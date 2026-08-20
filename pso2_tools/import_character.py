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
    ice,
    import_fnp,
    import_model,
    objects,
    proportions,
    scene_props,
)
from .debug import debug_print
from .preferences import get_preferences
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
    ("eyePart", "get_eyes", ("eye_l", "eye_r"), None),
    ("eyebrowPart", "get_eyebrows", ("eyebrow",), None),
    ("eyelashPart", "get_eyelashes", ("eyelash",), "shadow"),
)

# Texture file suffix -> the image node that carries it, matching what the
# model importer wires for skin (Diffuse<-_d, Color Mask<-_m, ...).
_TEXTURE_NODES = {"d": "Diffuse", "m": "Color Mask", "s": "Multi Map", "n": "Normal Map"}

# Face paints: part-id field -> its opacity slider. The game layers the
# first over the skin and the second over that.
_FACE_PAINT_PARTS = (
    ("makeup1Part", "facePaint1Opacity"),
    ("makeup2Part", "facePaint2Opacity"),
)


def _find_slider_ratio(char: charfile.CharacterFile, suffix: str) -> float:
    """A -127..127 slider under any field ending in `suffix`, as 0..1.

    Read forward: -127 is transparent, 127 opaque. Both test characters
    store their paints around -105 and draw softly in game; read the other
    way round the paints cover the face in blocks the game never shows.
    """
    for name in char:
        if name.split(".")[-1] == suffix:
            value = char[name]
            if isinstance(value, int):
                return max(0.0, min(1.0, (value + 127) / 254.0))
    return 0.5


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
            upstream = source.node.inputs["A"]
            source = upstream.links[0].from_socket if upstream.links else None

    for node in list(tree.nodes):
        if node.name.startswith("Face Paint"):
            tree.nodes.remove(node)

    if source is not None:
        tree.links.new(source, skin_group.inputs["Diffuse"])


def _layer_face_paint(
    material: bpy.types.Material,
    image: bpy.types.Image,
    opacity: float,
    index: int,
) -> bool:
    """Blend one face paint over the skin, before the shader group.

    The game composites face paints onto the face texture at the file's
    opacity slider. The same blend goes between the skin colorize and the
    shader group here, factored by the paint's own alpha times that
    opacity, so paints stack in slot order like they do in game.
    """
    tree = material.node_tree
    skin_group = tree.nodes.get("PSO2 NGS Skin")
    if skin_group is None or not skin_group.inputs["Diffuse"].links:
        return False

    current = skin_group.inputs["Diffuse"].links[0].from_socket
    base_x, base_y = skin_group.location

    tex = tree.nodes.new("ShaderNodeTexImage")
    tex.name = tex.label = f"Face Paint {index}"
    tex.image = image
    tex.location = (base_x - 900, base_y + 300 + index * 350)

    fac = tree.nodes.new("ShaderNodeMath")
    fac.name = f"Face Paint {index} Opacity"
    fac.label = fac.name
    fac.operation = "MULTIPLY"
    fac.inputs[1].default_value = opacity
    fac.location = (tex.location.x + 320, tex.location.y - 120)

    mix = tree.nodes.new("ShaderNodeMix")
    mix.name = f"Face Paint {index} Mix"
    mix.label = mix.name
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.clamp_factor = True
    mix.location = (tex.location.x + 560, tex.location.y)

    tree.links.new(tex.outputs["Alpha"], fac.inputs[0])
    tree.links.new(fac.outputs["Value"], mix.inputs["Factor"])
    tree.links.new(current, mix.inputs["A"])
    tree.links.new(tex.outputs["Color"], mix.inputs["B"])
    tree.links.new(mix.outputs["Result"], skin_group.inputs["Diffuse"])
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
    filter_glob: bpy.props.StringProperty(
        default="*.fnp;*.fkp;*.fcp", options={"HIDDEN"}
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

    def draw(self, context):
        assert self.layout is not None
        self.layout.prop(self, "import_colors")
        self.layout.prop(self, "import_proportions")
        self.layout.prop(self, "include_body")

    def execute(self, context) -> OperatorResult:
        path = Path(self.filepath)  # type: ignore

        try:
            char = charfile.CharacterFile.load(path)
        except (OSError, ValueError, KeyError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        data_path = get_preferences(context).get_pso2_data_path()

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
                loaded.append(obj.name)

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

            face_materials = _face_skin_materials()
            for material in face_materials:
                _clear_face_paint(material)

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
                diffuse = _load_part_images(obj, data_path).get("d")
                opacity = _find_slider_ratio(char, opacity_field)
                painted = diffuse is not None and [
                    m for m in face_materials if _layer_face_paint(m, diffuse, opacity, layer)
                ]
                if painted:
                    loaded.append(obj.name)
                else:
                    missing.append(f"{part_field}={part_id} (no face to paint)")

        if self.import_proportions:
            self._apply_proportions(context, char)

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
