"""Import a whole character from one .fnp: every part, its colours and shape.

A character file names dozens of parts by id - costume, hair, eyes, ears and
so on - but nothing loads them together. This walks the file's part ids, pulls
each one out of the game data through the object database, and drops the whole
character into the scene in one step. Accessories are left out by default: a
character often stacks a dozen and they rarely help a base import.

Parts come in two shapes. Most (hair, costume, ears, teeth) carry their own
model, so they import like any object. Eyes, eyebrows and eyelashes are
texture-only - their mesh already lives in the face model - so their textures
are wired onto the face's materials instead, and face paints are blended over
the face's skin at their opacity sliders. The character's own skin set loads
before any model so the face and body colour the neck from the same textures.

Colours are read into the scene *before* any model loads, because a material
bakes the scene colour it sees at import time; setting it afterwards leaves the
already-imported look untouched. Body proportions are applied last, once the
body has brought an armature into the scene to pose.
"""

import tempfile
from contextlib import closing
from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper

from . import (
    char_colors,
    charfile,
    classes,
    face_shape,
    game_normals,
    ice,
    import_fnp,
    import_model,
    objects,
    parts,
    proportions,
    scene_props,
)
from .colors import ColorId
from .debug import debug_print
from .preferences import get_preferences
from .shaders import shader_1104, shader_1105
from .shaders.colorize import ShaderNodePso2SrgbDecode, ShaderNodePso2SrgbEncode
from .util import OperatorResult

# Character-file field suffix -> object-database getter, in load order. The
# body parts come first so the proportion pass has an armature to pose. The
# four body slots are grouped so "Body / Outfit" can switch them off together.
_BODY_PARTS = (
    ("basewearPart", "get_basewear"),
    ("costumePart", "get_costumes"),
    ("innerwearPart", "get_innerwear"),
    ("outerwearPart", "get_outerwear"),
)
_HEAD_PARTS = (
    ("faceTypePart", "get_faces"),
    ("hairPart", "get_hair"),
    ("earsPart", "get_ears"),
    ("teethPart", "get_teeth"),
    ("hornPart", "get_horns"),
)

# Texture-only face parts: field suffix -> getter, the material name fragments
# to paint, and a fragment to skip. The eyelash shadow material carries the
# skin texture, not the eyelash one, so it is left alone.
_FACE_TEXTURE_PARTS = (
    ("eyePart", "get_eyes", ("eye_l", "eye_r", "tear_l", "tear_r"), None),
    ("eyebrowPart", "get_eyebrows", ("eyebrow",), None),
    ("eyelashPart", "get_eyelashes", ("eyelash",), "shadow"),
)

# Texture file suffix -> the image node that carries it, matching what the
# model importer wires for skin (Diffuse<-_d, Color Mask<-_m, ...).
_TEXTURE_NODES = {
    "d": "Diffuse",
    "m": "Color Mask",
    "s": "Multi Map",
    "n": "Normal Map",
    # the eye part's matcap, which the tears over the eyes read
    "v": shader_1105.ENV_MAP,
}

# Face paints: part-id field -> its opacity slider. The game layers the
# first over the skin and the second over that.
_FACE_PAINT_PARTS = (
    ("makeup1Part", "facePaint1Opacity"),
    ("makeup2Part", "facePaint2Opacity"),
)

# The opacity slider scales the paint's own alpha by (v + 127) / 127:
# nothing at -127, the paint as drawn at 0. Fitted against the face texture
# the game composited for a character with both paints at -106 (capture
# frame 16602): the eyeshadow fits best at 0.17 and the lipstick at 0.165,
# where the line gives 0.1654. An earlier curve through 0.73 at 0 was
# measured on frames while the colours here were still blended in linear
# space, which made every paint look weaker than its alpha. Above 0 the
# line goes on to 2 and the alpha saturates; that half is not measured.
_PAINT_FULL_AT = 127.0

# Where a face paint lands on the face texture. Its width matches the face
# texture's, so it covers the whole width, and it is an eighth as tall,
# sitting over the eyes. Measured against the game: a face UV of v 0.25 to
# 0.375 holds the eyeshadow, and the eye region of the face mesh (u 0.202
# to 0.781) lines up with the painted part of the texture (0.202 to 0.797)
# with no horizontal scaling at all.
_PAINT_ORIGIN_V = 0.25

# Eye size and iris size both scale the UVs the eye textures are sampled at,
# about the middle of the texture where the iris is drawn (its dark disc's
# centroid sits at 0.5000, 0.4937). A bigger scale samples a wider crop, so
# the iris draws smaller. Neither slider moves a bone, which is why reading
# the skeleton says they do nothing.
#
# The two ends are the CMX's eyeDict unkFloat0/unkFloat1, the same pair on
# every eye item in the game. Sweeping each slider in a running game and
# measuring the iris confirms them, the blend between, and that the two
# scales multiply:
#
#     eye size   iris size   iris width   scale(eye) * scale(iris)
#          0           0         56.5          1.0000  -> 56.3
#          0         127         67.5          0.8333  -> 67.5
#        127        -127         53.5          1.0417  -> 54.0
#        127         127         80.0          0.6944  -> 81.0
#
# Slider 0 landing on a scale of exactly 1.0 is what makes this the same
# neutral-anchored two-segment blend as every other slider: a straight line
# between the ends would put the pair at (0, 0) 5% smaller than measured.
_IRIS_UV_SCALE_MIN = 1.25
_IRIS_UV_SCALE_MAX = 0.8333333


def _face_paint_alpha(char: charfile.CharacterFile, suffix: str) -> float:
    """The blend a -127..127 opacity slider asks for, as 0..1."""
    for name in char:
        if name.split(".")[-1] == suffix:
            value = char[name]
            if not isinstance(value, int):
                continue
            value = max(-127, min(127, value))
            return (value + 127) / _PAINT_FULL_AT
    return 1.0


def _find_part_id(char: charfile.CharacterFile, suffix: str) -> int:
    """The part id stored under any of the ...Part fields ending in `suffix`.

    Part selections are spread across baseSLCT, baseSLCT2 and baseSLCTNGS
    depending on when NGS added them, so the block prefix is matched loosely.
    Returns 0 when the character has nothing in that slot.
    """
    for name in char:
        if name.split(".")[-1] == suffix:
            value = char[name]
            if isinstance(value, int):
                return value
    return 0


def _load_part_images(
    obj: objects.CmxObjectBase, data_path: Path
) -> dict[str, bpy.types.Image]:
    """Load a texture-only part's images, keyed by their `_d/_m/...` suffix."""
    files = obj.get_files()
    if not files:
        return {}

    ice_path = import_model._get_ice_path(files[0], data_path, True)
    if ice_path is None or not ice_path.exists():
        return {}

    images: dict[str, bpy.types.Image] = {}
    with tempfile.TemporaryDirectory(prefix="pso2_char_") as tmp:
        for entry in ice.IceFile.load(ice_path).get_files():
            if not entry.name.lower().endswith(".dds"):
                continue
            out = Path(tmp) / entry.name
            out.write_bytes(entry.data)
            suffix = entry.name.rsplit("_", 1)[-1].split(".")[0].lower()
            image = bpy.data.images.load(str(out), check_existing=True)
            image.pack()  # the temp file is about to be removed
            # Only the diffuse and the eye's matcap are colour (the game
            # reads both as sRGB). The mask, multi and normal maps are data,
            # and read through the sRGB curve a mask at half strength paints
            # the iris at a fifth - which is what left the eyes near black.
            colour = suffix in ("d", "v")
            image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
            images[suffix] = image

    return images


def _paint_face_textures(
    obj: objects.CmxObjectBase,
    data_path: Path,
    fragments: tuple[str, ...],
    skip: str | None,
) -> bool:
    """Wire a texture-only part's images onto the face's materials.

    Eyes/brows/lashes ship as a bare set of `_d/_m/_s/_n` textures with no
    model, so the face model's matching materials are painted with them
    directly. Returns whether any image was placed.
    """
    images = _load_part_images(obj, data_path)
    if not images:
        return False

    painted = False
    for material in bpy.data.materials:
        name = material.name.lower()
        if not material.use_nodes:
            continue
        if skip and skip in name:
            continue
        if not any(fragment in name for fragment in fragments):
            continue

        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE":
                continue
            for suffix, label in _TEXTURE_NODES.items():
                if node.label == label and suffix in images:
                    node.image = images[suffix]
                    painted = True

    return painted


def _find_slider(char: charfile.CharacterFile, name: str) -> int | None:
    """A top-level slider's value, whichever block this file keeps it in."""
    for field in char:
        if field.split(".")[-1] == name:
            value = char[field]
            if isinstance(value, int):
                return value
    return None


def _iris_uv_scale(value: int) -> float:
    """One eye slider's share of the UV scale."""
    value = max(-127, min(127, value))
    end = _IRIS_UV_SCALE_MAX if value >= 0 else _IRIS_UV_SCALE_MIN
    return 1.0 + (end - 1.0) * abs(value) / 127.0


def _eye_materials() -> list[bpy.types.Material]:
    """The face's two eye materials, which the eye part's textures went on,
    and the tears over them, whose matcap the game scales with the iris."""
    return [
        m
        for m in bpy.data.materials
        if m.use_nodes
        and any(side in m.name for side in ("eye_l", "eye_r", "tear_l", "tear_r"))
    ]


def _scale_iris(material: bpy.types.Material, scale: float) -> bool:
    """Sample the eye's textures at `scale`, centred on the iris."""
    tree = material.node_tree
    textures = [n for n in tree.nodes if n.type == "TEX_IMAGE"]
    if not textures:
        return False

    mapping = tree.nodes.get(shader_1104.IRIS_SIZE)
    if mapping is None:
        uv_node = tree.nodes.new("ShaderNodeUVMap")
        uv_node.name = uv_node.label = shader_1104.IRIS_SIZE_UV
        uv_node.location = (min(t.location.x for t in textures) - 700, 0)

        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.name = mapping.label = shader_1104.IRIS_SIZE
        mapping.location = (uv_node.location.x + 250, 0)
        tree.links.new(uv_node.outputs["UV"], mapping.inputs["Vector"])

    mapping.inputs["Scale"].default_value = (scale, scale, 1.0)
    offset = 0.5 * (1.0 - scale)
    mapping.inputs["Location"].default_value = (offset, offset, 0.0)

    for tex in textures:
        if not tex.inputs["Vector"].links:
            tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    return True


def _body_armature(context) -> bpy.types.Object | None:
    """The armature that is the whole character's skeleton, not a part's."""
    for obj in context.scene.objects:
        if obj.type == "ARMATURE" and any(
            bone.name.split("#")[0] == "body_root" for bone in obj.data.bones
        ):
            return obj
    return None


def _weight_deficit(obj: bpy.types.Object) -> list[float]:
    """How much of each vertex's skin weight never landed on a bone."""
    return [max(0.0, 1.0 - sum(g.weight for g in v.groups)) for v in obj.data.vertices]


def _root_share(obj: bpy.types.Object, root_name: str | None) -> list[float]:
    """Each vertex's weight that belongs to the part's root node.

    The FBX import dropped it, leaving a deficit; the native import keeps it
    on a bone of its own named after the part. Both mean the same thing.
    """
    root = obj.vertex_groups.get(root_name) if root_name else None
    shares = []
    for vertex in obj.data.vertices:
        total = 0.0
        on_root = 0.0
        for g in vertex.groups:
            total += g.weight
            if root is not None and g.group == root.index:
                on_root += g.weight
        shares.append(max(0.0, 1.0 - total) + on_root)
    return shares


# A mesh either rides the root node for a real share of its skin or not at
# all. Below this, the gap is the rounding a handful of vertices pick up
# from influences the import dropped for other reasons, and moving it onto
# a body bone would drag scattered vertices out of the face.
_ROOT_WEIGHT_FLOOR = 0.05

# How far the part's origin may sit from a body bone and still be taken for
# it. The neighbouring candidates are 4cm away, so this only has to absorb
# the wobble in where a part model is authored.
_ROOT_BONE_TOLERANCE = 0.02


def _bone_at(body: bpy.types.Object, point, tolerance: float):
    """The body bone resting at `point`, if one rests close enough to it."""
    best = None
    best_distance = tolerance
    for bone in body.data.bones:
        distance = ((body.matrix_world @ bone.head_local) - point).length
        if distance < best_distance:
            best, best_distance = bone, distance
    return best


def _add_proxy_bone(
    armature: bpy.types.Object, name: str, matrix, length: float
) -> None:
    """Give `armature` a bone standing in for a body bone.

    It takes the body bone's name so the proportion pass, which matches by
    name, drives it like any other, and its rest transform so the scales
    that pass writes mean the same thing on both. Everything about the
    body bone is passed in by value: switching an armature into edit mode
    re-points the Bone objects held on another armature, so a reference
    read afterwards names some unrelated bone.
    """
    previous = bpy.context.view_layer.objects.active
    if previous is not None and previous.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit_bones = armature.data.edit_bones
        if name in edit_bones:
            edit_bones.remove(edit_bones[name])
        edit_bone = edit_bones.new(name)
        edit_bone.head = (0.0, 0.0, 0.0)
        edit_bone.tail = (0.0, length, 0.0)
        edit_bone.matrix = matrix
        edit_bone.length = length
        edit_bone.use_deform = True
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.context.view_layer.objects.active = previous


def _restore_root_node_weights(context) -> int:
    """Re-attach the skin weights the model's dropped root node carried.

    A part model's first node is not one of the part's own bones: it is
    the body bone the part hangs from, and for a face that is the neck.
    The importer folds that node into the armature object instead of
    keeping it as a bone, so the weights aimed at it have nowhere to land
    and are dropped - here, half the neck skirt's skin, and three quarters
    of it on some copies. Those vertices then ride the armature rigidly,
    holding the width the model was authored at while the body's neck
    shrinks under the sliders, which reads as a collar standing off the
    neck instead of as the neck itself.
    """
    body = _body_armature(context)
    if body is None:
        return 0

    by_armature: dict[bpy.types.Object, list[bpy.types.Object]] = {}
    for obj in context.scene.objects:
        if obj.type == "MESH" and obj.parent and obj.parent.type == "ARMATURE":
            by_armature.setdefault(obj.parent, []).append(obj)

    restored = 0
    for armature, meshes in by_armature.items():
        if armature == body:
            continue

        # The native import keeps the root node as a parentless bone named
        # after the part, and the weights arrive on it intact - but nothing
        # poses that bone, so they ride it as rigidly as they rode the
        # armature when the FBX import dropped them. Either way they belong
        # to the body bone the root rests on. Only the bone named after the
        # part is that node: under the FBX import the top bone is the part's
        # own head, whose weights must stay where they are.
        root = next(
            (
                b
                for b in armature.data.bones
                if b.parent is None
                and b.name.split("#")[0] == armature.name.split("#")[0]
            ),
            None,
        )
        root_name = root.name if root is not None else None
        anchor = (
            armature.matrix_world @ root.head_local
            if root is not None
            else armature.matrix_world.translation
        )

        needy = []
        for mesh in meshes:
            share = _root_share(mesh, root_name)
            if share and sum(share) / len(share) >= _ROOT_WEIGHT_FLOOR:
                needy.append((mesh, share))
        if not needy:
            continue

        bone = _bone_at(body, anchor, _ROOT_BONE_TOLERANCE)
        if bone is None:
            debug_print(
                f"No body bone under {armature.name}; left its root weights off"
            )
            continue

        name = bone.name
        matrix = (
            armature.matrix_world.inverted() @ body.matrix_world @ bone.matrix_local
        )
        _add_proxy_bone(armature, name, matrix, bone.length)
        for mesh, share in needy:
            group = mesh.vertex_groups.get(name) or mesh.vertex_groups.new(name=name)
            root_group = mesh.vertex_groups.get(root_name) if root_name else None
            for index, weight in enumerate(share):
                if weight > 1e-4:
                    group.add([index], weight, "REPLACE")
                    if root_group is not None:
                        root_group.remove([index])
                    restored += 1

    return restored


# The face carries its neck as a skirt below the jaw, and ships one copy per
# body region a costume can cover, so that whichever region the outfit hides
# takes the neck with it. The copies are coincident - all four wrap the full
# 360 degrees over the same 7cm of height with the same 129cm2 of surface,
# differing only by ~2mm of radius and by vertex count - so drawing them all
# stacks their rims into visible steps under the chin.
_FACE_NECK_MESH_IDS = (
    parts.MeshId.BreastNeck,
    parts.MeshId.Front,
    parts.MeshId.Ornament1,
    parts.MeshId.Back,
)


def _is_face_mesh(obj: bpy.types.Object) -> bool:
    """Whether a mesh is part of a face model, by its [fc] skin material."""
    return any(m and "[fc]" in m.name for m in obj.data.materials)


def _keep_one_neck_variant(context) -> int:
    """Hide all but one of the face's interchangeable neck skirts.

    BreastNeck is the bare-skin copy and the one to keep: it is what shows
    with nothing covering the throat, which is the state we import into.

    A costume uses these same ids for its own breast, front, ornament and
    back pieces, where they are separate parts rather than copies, so this
    only ever looks at meshes wearing a face's [fc] material.
    """
    found: dict[parts.MeshId, list[bpy.types.Object]] = {}
    for obj in context.selected_objects:
        if obj.type != "MESH" or not _is_face_mesh(obj):
            continue
        try:
            mesh_id = parts.get_mesh_id(obj.name)
        except ValueError:
            continue
        if mesh_id in _FACE_NECK_MESH_IDS:
            found.setdefault(mesh_id, []).append(obj)

    if len(found) < 2:
        return 0

    keep = next(i for i in _FACE_NECK_MESH_IDS if i in found)
    hidden = 0
    for mesh_id, objects_ in found.items():
        if mesh_id == keep:
            continue
        for obj in objects_:
            obj.hide_viewport = True
            obj.hide_render = True
            hidden += 1
    return hidden


def _face_skin_materials() -> list[bpy.types.Material]:
    """The face's skin materials, tagged [fc]: shader 1102 (T2) or 1101 (T1)."""
    return [
        m
        for m in bpy.data.materials
        if m.use_nodes
        and "[fc]" in m.name
        and ("(1102p" in m.name or "(1101p" in m.name)
    ]


def _clear_face_paint(material: bpy.types.Material) -> None:
    """Remove any face-paint layers a previous import left on the material."""
    tree = material.node_tree
    skin_group = tree.nodes.get("PSO2 NGS Skin")
    if skin_group is None:
        return

    # Walk the paint chain back to whatever originally fed the shader, so
    # the link can be put back once the paint nodes are gone.
    source = None
    if skin_group.inputs["Diffuse"].links:
        source = skin_group.inputs["Diffuse"].links[0].from_socket
        while source is not None and source.node.name.startswith("Face Paint"):
            upstream = next(
                (
                    s
                    for s in source.node.inputs
                    if s.name in ("A", "Color") and s.is_linked
                ),
                None,
            )
            source = upstream.links[0].from_socket if upstream else None

    # Takes the paint's UV and placement nodes with it - they share the
    # "Face Paint N" prefix.
    for node in list(tree.nodes):
        if node.name.startswith("Face Paint"):
            tree.nodes.remove(node)

    if source is not None:
        tree.links.new(source, skin_group.inputs["Diffuse"])


def _paint_rect(
    placement: tuple[float, float, float, float] | None,
    image: bpy.types.Image,
    face_image: bpy.types.Image | None,
) -> tuple[float, float, float, float] | None:
    """Where a paint sits on the face, as (u, v, u scale, v scale).

    The CMX rectangle is (u, v measured down from the top, width, height);
    Blender measures v up from the bottom. With no rectangle the paint is
    assumed to be a full-width band over the eyes, which is what every
    1024-wide paint is and what the eyeshadow this was worked out on reads
    out of the CMX anyway.
    """
    if placement is not None:
        u, v_from_top, width, height = placement
        return u, 1.0 - v_from_top - height, 1.0 / width, 1.0 / height

    if face_image and all(image.size) and all(face_image.size):
        return (
            0.0,
            _PAINT_ORIGIN_V,
            face_image.size[0] / image.size[0],
            face_image.size[1] / image.size[1],
        )
    return None


def _layer_face_paint(
    material: bpy.types.Material,
    image: bpy.types.Image,
    opacity: float,
    index: int,
    placement: tuple[float, float, float, float] | None = None,
    mask: bpy.types.Image | None = None,
    colors: tuple[ColorId, ColorId] | None = None,
) -> bool:
    """Blend one face paint over the skin, before the shader group.

    The game composites face paints onto the face texture, in slot order, at
    the file's opacity slider. The same blend goes between the skin colorize
    and the shader group here, factored by the paint's own alpha times that
    opacity. Like the rest of the compositing it runs in sRGB space.

    A paint whose CMX entry names colours is tinted by its mask first: red
    and green each blend towards one character colour, over the paint's own
    colour. The hair's scalp paint is black with a mask of about two thirds,
    so it lays the hair colour at two thirds strength over the head.

    A paint is not a whole face texture: it is a strip that covers a band
    of the face's UV space, an eighth of the texture's height over the
    eyes, at the texture's own resolution. Sampled with the face's UVs
    unchanged it stretches over the entire head, which puts eyeshadow on
    the cheeks and forehead - the shape a running game never draws there.
    """
    tree = material.node_tree
    skin_group = tree.nodes.get("PSO2 NGS Skin")
    if skin_group is None or not skin_group.inputs["Diffuse"].links:
        return False

    current = skin_group.inputs["Diffuse"].links[0].from_socket
    base_x, base_y = skin_group.location
    x = base_x - 900
    y = base_y + 300 + index * 350
    prefix = f"Face Paint {index}"

    def new(node_type: str, name: str, dx: float, dy: float = 0.0):
        node = tree.nodes.new(node_type)
        node.name = node.label = name
        node.location = (x + dx, y + dy)
        return node

    tex = new("ShaderNodeTexImage", prefix, 0)
    tex.image = image
    # Outside its band the paint must not draw at all, so no repeats.
    tex.extension = "CLIP"

    mask_tex = None
    if mask is not None and colors:
        mask_tex = new("ShaderNodeTexImage", f"{prefix} Mask", 0, -280)
        mask_tex.image = mask
        mask_tex.extension = "CLIP"

    face_texture = tree.nodes.get("Diffuse")
    face_image = face_texture.image if face_texture else None
    rect = _paint_rect(placement, image, face_image)
    if rect is not None:
        origin_u, origin_v, scale_u, scale_v = rect

        uv_node = new("ShaderNodeUVMap", f"{prefix} UV", -700)
        mapping = new("ShaderNodeMapping", f"{prefix} Placement", -450)
        mapping.inputs["Scale"].default_value = (scale_u, scale_v, 1.0)
        mapping.inputs["Location"].default_value = (
            -origin_u * scale_u,
            -origin_v * scale_v,
            0.0,
        )

        tree.links.new(uv_node.outputs["UV"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        if mask_tex is not None:
            tree.links.new(mapping.outputs["Vector"], mask_tex.inputs["Vector"])

    base = new(ShaderNodePso2SrgbEncode.__name__, f"{prefix} Base sRGB", 560, 200)
    tree.links.new(current, base.inputs["Color"])

    paint = new(ShaderNodePso2SrgbEncode.__name__, f"{prefix} sRGB", 320)
    tree.links.new(tex.outputs["Color"], paint.inputs["Color"])
    painted = paint.outputs["Color"]

    channels = tree.nodes.get("Colors")
    if mask_tex is not None and colors and channels is not None:
        split = new("ShaderNodeSeparateColor", f"{prefix} Mask RGB", 320, -280)
        split.mode = "RGB"
        tree.links.new(mask_tex.outputs["Color"], split.inputs["Color"])
        for step, (channel, color_id) in enumerate(
            zip(("Red", "Green"), colors, strict=True)
        ):
            if color_id == ColorId.UNUSED:
                continue
            encode = new(
                ShaderNodePso2SrgbEncode.__name__,
                f"{prefix} {channel} sRGB",
                320 + 160 * step,
                -520,
            )
            tree.links.new(channels.outputs[color_id.value - 1], encode.inputs["Color"])
            squeeze = new(
                "ShaderNodeVectorMath",
                f"{prefix} {channel} Range",
                480 + 160 * step,
                -520,
            )
            squeeze.operation = "MULTIPLY_ADD"
            squeeze.inputs[1].default_value = (0.99, 0.99, 0.99)
            squeeze.inputs[2].default_value = (0.005, 0.005, 0.005)
            tree.links.new(encode.outputs["Color"], squeeze.inputs[0])

            tint = new("ShaderNodeMix", f"{prefix} {channel}", 480 + 160 * step, -260)
            tint.data_type = "RGBA"
            tint.blend_type = "MIX"
            tint.clamp_factor = True
            tree.links.new(painted, tint.inputs["A"])
            tree.links.new(squeeze.outputs["Vector"], tint.inputs["B"])
            tree.links.new(split.outputs[channel], tint.inputs["Factor"])
            painted = tint.outputs["Result"]

    fac = new("ShaderNodeMath", f"{prefix} Opacity", 320, -120)
    fac.operation = "MULTIPLY"
    fac.use_clamp = True
    fac.inputs[1].default_value = opacity

    mix = new("ShaderNodeMix", f"{prefix} Mix", 800)
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.clamp_factor = True

    linear = new(ShaderNodePso2SrgbDecode.__name__, f"{prefix} Linear", 1040)

    tree.links.new(tex.outputs["Alpha"], fac.inputs[0])
    tree.links.new(fac.outputs["Value"], mix.inputs["Factor"])
    tree.links.new(base.outputs["Color"], mix.inputs["A"])
    tree.links.new(painted, mix.inputs["B"])
    tree.links.new(mix.outputs["Result"], linear.inputs["Color"])
    tree.links.new(linear.outputs["Color"], skin_group.inputs["Diffuse"])
    return True


@classes.register
class PSO2_OT_ImportCharacter(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ImportHelper
):
    """Load a whole PSO2 character from a .fnp file: parts, colours and body shape"""

    bl_label = "Import Character (.fnp)"
    bl_idname = "pso2.import_character"
    bl_options = {"UNDO"}

    filename_ext = ".fnp"
    # A save is named [gender][race]p: f or m, then d/n/h/c for the four
    # races, and a trailing u where the body is not encrypted. Listing a
    # few by hand left the male ones invisible in the file dialog, so the
    # whole set is generated.
    filter_glob: bpy.props.StringProperty(
        default=";".join(
            f"*.{gender}{race}p{plain}"
            for gender in "fm"
            for race in "dnhc"
            for plain in ("", "u")
        ),
        options={"HIDDEN"},
    )

    import_colors: bpy.props.BoolProperty(
        name="Colours",
        description="Read the character's skin, hair, eye and outfit colours from the file",
        default=True,
    )
    import_proportions: bpy.props.BoolProperty(
        name="Body Proportions",
        description="Pose the body's bones to the character's body-shape sliders",
        default=True,
    )
    include_body: bpy.props.BoolProperty(
        name="Body / Outfit",
        description="Load basewear, costume, innerwear and outerwear",
        default=True,
    )
    expression: bpy.props.EnumProperty(
        name="Expression",
        description=(
            "Which of the file's expression presets the face rests in."
            " The character creator shows Natural"
        ),
        items=[
            (name, label, f"Pose the face with the file's {label} preset")
            for name, label in face_shape.EXPRESSIONS
        ],
        default=face_shape.DEFAULT_EXPRESSION,
    )
    game_normals: bpy.props.BoolProperty(
        name="Game Normals",
        description=(
            "Turn the normals with the bones the way the game does, so the"
            " face's neck meets the body without a band (Blender 4.5 or later)"
        ),
        default=True,
    )

    def draw(self, context):
        assert self.layout is not None
        self.layout.prop(self, "import_colors")
        self.layout.prop(self, "import_proportions")
        self.layout.prop(self, "include_body")
        self.layout.prop(self, "expression")
        row = self.layout.row()
        row.enabled = game_normals.supported()
        row.prop(self, "game_normals")

    def execute(self, context) -> OperatorResult:
        path = Path(self.filepath)  # type: ignore

        try:
            char = charfile.CharacterFile.load(path)
        except (OSError, ValueError, KeyError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        data_path = get_preferences(context).get_pso2_data_path()
        existing = {obj.as_pointer() for obj in bpy.data.objects}

        # Colours first: a material bakes the scene colour it sees when it is
        # built, so they have to be in place before anything imports.
        if self.import_colors:
            char_colors.apply_to_scene(context, char)
            self._apply_muscularity(context, char)

        model_parts = _HEAD_PARTS + (_BODY_PARTS if self.include_body else ())
        loaded: list[str] = []
        missing: list[str] = []

        with closing(objects.ObjectDatabase(context)) as db:
            # The character's skin goes into the file before any model:
            # each part's import takes whatever skin images are already
            # loaded, and the first one would otherwise pull in the
            # preference default instead. The face and body sharing one
            # skin set is what keeps the neck seamless.
            skin_id = _find_part_id(char, "skinTextureSet")
            if skin_id > 0:
                if import_model._import_skin_textures(
                    context, high_quality=True, use_t2_skin=False, skin_id=skin_id
                ):
                    loaded.append(f"skin={skin_id}")
                else:
                    missing.append(f"skinTextureSet={skin_id}")

            for suffix, getter in model_parts:
                part_id = _find_part_id(char, suffix)
                if part_id <= 0:
                    continue
                obj = next(iter(getattr(db, getter)(part_id)), None)
                if obj is None:
                    missing.append(f"{suffix}={part_id}")
                    continue
                import_model.import_object(self, context, obj, high_quality=True)
                if suffix == "faceTypePart":
                    debug_print(
                        f"Face neck: hid {_keep_one_neck_variant(context)} spare copies"
                    )
                loaded.append(obj.name)

            # Head parts import before the body, so this waits until the
            # skeleton it has to name its bones after is in the scene.
            debug_print(
                f"Rewired {_restore_root_node_weights(context)} root-node weights"
            )

            for suffix, getter, fragments, skip in _FACE_TEXTURE_PARTS:
                part_id = _find_part_id(char, suffix)
                if part_id <= 0:
                    continue
                obj = next(iter(getattr(db, getter)(part_id)), None)
                if obj is None:
                    missing.append(f"{suffix}={part_id}")
                    continue
                if _paint_face_textures(obj, data_path, fragments, skip):
                    loaded.append(obj.name)
                else:
                    missing.append(f"{suffix}={part_id} (no face to paint)")

            iris = face_shape.slider(char, "<expr>.irisSize", self.expression)
            eye_size = _find_slider(char, "eyeSize")
            if iris is not None or eye_size is not None:
                scale = _iris_uv_scale(iris or 0) * _iris_uv_scale(eye_size or 0)
                scaled = [m for m in _eye_materials() if _scale_iris(m, scale)]
                debug_print(
                    f"Eye size {eye_size}, iris size {iris}"
                    f" -> UV scale {scale:.4f} on {len(scaled)} eye materials"
                )

            face_materials = _face_skin_materials()
            for material in face_materials:
                _clear_face_paint(material)

            paint_cmx = objects.get_facepaint_cmx(data_path.parent)
            paint_placement = paint_cmx.placement

            # The hairstyle's own paint goes on first: the game lays it over
            # the scalp in the hair's colours before any makeup.
            hair_id = _find_part_id(char, "hairPart")
            scalp_id = paint_cmx.scalp.get(hair_id)
            if scalp_id is not None:
                obj = next(iter(db.get_facepaint(scalp_id)), None)
                images = _load_part_images(obj, data_path) if obj else {}
                painted = "d" in images and [
                    m
                    for m in face_materials
                    if _layer_face_paint(
                        m,
                        images["d"],
                        1.0,
                        0,
                        paint_placement.get(scalp_id),
                        images.get("m"),
                        paint_cmx.colors.get(scalp_id),
                    )
                ]
                if painted:
                    loaded.append(obj.name)
                else:
                    missing.append(f"scalp paint {scalp_id} for hair {hair_id}")

            for layer, (part_field, opacity_field) in enumerate(
                _FACE_PAINT_PARTS, start=1
            ):
                part_id = _find_part_id(char, part_field)
                if part_id <= 0:
                    continue
                obj = next(iter(db.get_facepaint(part_id)), None)
                if obj is None:
                    missing.append(f"{part_field}={part_id}")
                    continue
                images = _load_part_images(obj, data_path)
                diffuse = images.get("d")
                opacity = _face_paint_alpha(char, opacity_field)
                placement = paint_placement.get(part_id)
                painted = diffuse is not None and [
                    m
                    for m in face_materials
                    if _layer_face_paint(
                        m,
                        diffuse,
                        opacity,
                        layer,
                        placement,
                        images.get("m"),
                        paint_cmx.colors.get(part_id),
                    )
                ]
                if painted:
                    loaded.append(obj.name)
                else:
                    missing.append(f"{part_field}={part_id} (no face to paint)")

        if self.import_proportions:
            self._apply_proportions(context, char)
            self._apply_face_shape(context, char, self.expression)

        if self.game_normals:
            added = game_normals.add_to(
                obj for obj in bpy.data.objects if obj.as_pointer() not in existing
            )
            debug_print(f"Game normals on {added} meshes")

        if missing:
            shown = ", ".join(missing[:6])
            more = "..." if len(missing) > 6 else ""
            self.report(
                {"WARNING"},
                f"Loaded {len(loaded)} parts; {len(missing)} not found"
                f" ({shown}{more}).",
            )
        else:
            self.report({"INFO"}, f"Loaded {len(loaded)} parts.")

        return {"FINISHED"}

    def _apply_face_shape(
        self,
        context,
        char: charfile.CharacterFile,
        expression: str = face_shape.DEFAULT_EXPRESSION,
    ) -> None:
        """Shape the face with its own sliders, after the body proportions.

        It has to run after them because they reset every pose they touch,
        and the head-part fit has to run again afterwards because reshaping
        the face moves the mouth the teeth sit in.
        """
        face_id = _find_part_id(char, "faceTypePart")
        if face_id <= 0:
            return

        summary = face_shape.apply(context, char, face_id, expression)
        debug_print(
            f"Face shape: posed {summary['bones']} bones"
            f" on {summary['posed']} armatures"
        )
        if summary["posed"]:
            self._attach_head_parts(context)

    def _apply_muscularity(self, context, char: charfile.CharacterFile) -> None:
        """Set the scene's muscle blend to the character's muscle mass.

        The skin shaders mix their base and muscular texture sets by this
        value, so leaving it at the default renders every character at the
        same half-muscled skin regardless of the file.
        """
        try:
            muscle_mass = float(char["baseDOC.muscleMass"])
        except (KeyError, TypeError, ValueError):
            return

        value = max(0.0, min(1.0, muscle_mass / import_fnp._MUSCLE_MASS_MAX))
        try:
            setattr(context.scene, scene_props.MUSCULARITY, value)
        except (AttributeError, TypeError):
            debug_print("Could not set scene muscularity")

    def _apply_proportions(self, context, char: charfile.CharacterFile) -> None:
        """Pose every imported armature to the body-shape sliders.

        Each part keeps its own armature - body, head, hair, ears - and the
        proportion table matches by bone name, so it has to run on all of
        them. Posing only the body left the head at its bind size, which
        parted the neck; the shared spine/neck/head bones exist on the other
        armatures too, so the same pose keeps them lined up.
        """
        try:
            result = proportions.compute(char)
        except (OSError, KeyError, ValueError) as ex:
            debug_print("Could not compute proportions:", ex)
            return

        posed = 0
        for obj in context.scene.objects:
            if obj.type != "ARMATURE":
                continue
            summary = import_fnp.apply_proportions(obj, result["bones"])
            if summary["applied"]:
                posed += 1
                debug_print(f"Posed {obj.name} ({summary['applied']} bones)")

        debug_print(f"Applied proportions to {posed} armatures")
        self._attach_head_parts(context)

    def _attach_head_parts(self, context) -> None:
        """Move each head part's armature onto the posed body's head bone.

        Every part imports as its own armature, parked where the model was
        authored, but the body sliders move the head attach point. The game
        runs all the parts on one skeleton so they can never drift; here
        the face's neck skirt is alpha-faded over the body's neck, and the
        couple of centimetres of drift open a see-through ring where the
        fade has nothing behind it.
        """
        context.view_layer.update()

        body = None
        parts = []
        for obj in context.scene.objects:
            if obj.type != "ARMATURE":
                continue
            bases = {b.name.split("#")[0] for b in obj.pose.bones}
            if "body_root" in bases:
                body = obj
            else:
                parts.append(obj)
        if body is None or not parts:
            return

        head = next(
            (
                b
                for name in ("head", "neck2", "neck1")
                for b in body.pose.bones
                if b.name.split("#")[0] == name
            ),
            None,
        )
        if head is None:
            return
        target = body.matrix_world @ head.head

        moved = 0
        for obj in parts:
            root = next(
                (
                    b
                    for b in obj.pose.bones
                    if b.name.split("#")[0].rstrip("0123456789") == "head"
                ),
                None,
            )
            if root is None:
                continue
            current = obj.matrix_world @ root.head
            delta = target - current
            if delta.length > 1e-5:
                obj.matrix_world.translation += delta
                moved += 1

        debug_print(f"Attached {moved} head parts to {head.name}")
        self._follow_body_proxies(context, body, parts)
        self._follow_face_bones(context, parts)

    def _follow_body_proxies(self, context, body, parts_armatures) -> None:
        """Put each part's stand-in body bones exactly where the body's are.

        The proxy bones _restore_root_node_weights adds carry the body bone's
        name, so the proportion pass scales them like the body's - but they
        hang in the part's armature, which is placed by the head. The neck
        sliders also move the neck against the head (a shorter neck puts it
        higher), so a proxy left at its rest offset sits below the real neck
        and the skirt riding it hangs low and long. The game skins those
        vertices to the body's own bone; copying its posed matrix does the
        same here.
        """
        for obj in parts_armatures:
            for pose_bone in obj.pose.bones:
                if pose_bone.parent is not None:
                    continue
                source = body.pose.bones.get(pose_bone.name)
                if source is None:
                    continue
                # Refresh before each write, as with _follow_face_bones.
                context.view_layer.update()
                pose_bone.matrix = (
                    obj.matrix_world.inverted() @ body.matrix_world @ source.matrix
                )

    def _follow_face_bones(self, context, parts_armatures) -> None:
        """Copy the face's posed bone matrices onto the parts' shared bones.

        A part rig is a cut-down copy of the face's chain - the teeth carry
        the mouth bones but not the morph-driver bones between them and the
        head - so a scale the proportion pass puts on those drivers moves
        the face's mouth and not the teeth, and the teeth poke out of the
        closed lips. Matching bones by their counter-stripped names and
        copying the posed matrix keeps the parts inside the face the way
        the game's single skeleton does.
        """
        face = next(
            (o for o in parts_armatures if "_rhd_" in o.name),
            None,
        )
        if face is None:
            return

        face_by_base: dict[str, bpy.types.PoseBone] = {}
        for pose_bone in face.pose.bones:
            key = pose_bone.name.split("#")[0].rstrip("0123456789")
            face_by_base.setdefault(key, pose_bone)

        for obj in parts_armatures:
            if obj is face:
                continue
            for pose_bone in obj.pose.bones:
                key = pose_bone.name.split("#")[0].rstrip("0123456789")
                source = face_by_base.get(key)
                if source is None:
                    continue
                # Writing consecutive pose matrices without an update in
                # between reads stale parents, so refresh before each one.
                context.view_layer.update()
                pose_bone.matrix = (
                    obj.matrix_world.inverted() @ face.matrix_world @ source.matrix
                )
