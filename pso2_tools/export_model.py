import json
import re
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypedDict, cast, get_type_hints

import bpy
from mathutils import Matrix

from . import dotnet, fbx_wrapper
from .util import OperatorResult


class FbxExportOptions(TypedDict, total=False):
    use_selection: bool
    use_visible: bool
    use_active_collection: bool
    collection: str

    global_matrix: Matrix
    apply_unit_scale: bool
    global_scale: float
    apply_scale_options: str
    axis_up: str
    axis_forward: str
    context_objects: Any
    object_types: Any
    use_mesh_modifiers: bool
    use_mesh_modifiers_render: bool
    mesh_smooth_type: str
    use_subsurf: bool
    use_armature_deform_only: bool
    bake_anim: bool
    bake_anim_use_all_bones: bool
    bake_anim_use_nla_strips: bool
    bake_anim_use_all_actions: bool
    bake_anim_step: float
    bake_anim_simplify_factor: float
    bake_anim_force_startend_keying: bool
    add_leaf_bones: bool
    primary_bone_axis: str
    secondary_bone_axis: str
    use_metadata: bool
    path_mode: str
    use_mesh_edges: bool
    use_tspace: bool
    use_triangles: bool
    embed_textures: bool
    use_custom_props: bool
    bake_space_transform: bool
    armature_nodetype: str
    colors_type: str
    prioritize_active_color: bool


class ExportOptions(FbxExportOptions, total=False):
    rigid: bool
    override_bounding_radius: bool
    bounding_radius: float


# The package stores each entry's name in a fixed 0x20 byte field, and the
# writer appends "_l1.aqo" to the file name. Measured against real exports: a
# stem of 22 still leaves room for the terminator, 23 and 24 lose the ".aqo"
# and the package reads back with no models at all, and 25 or more is
# truncated so two different names can collide.
MAX_MODEL_NAME = 22

# What the game's own player body models carry in the bounding radius field.
GAME_BOUNDING_RADIUS = -10.0


def export(
    operator: bpy.types.Operator,
    context: bpy.types.Context,
    path: Path,
    is_ngs=True,
    overwrite_aqn=False,
    options: ExportOptions | None = None,
) -> OperatorResult:
    from AquaModelLibrary.Core.General import AssimpModelImporter
    from AquaModelLibrary.Data.PSO2.Aqua import AquaNode, AquaObject, AquaPackage

    dotnet.set_assimp_probing_paths()

    options = options or {}

    if len(path.stem) > MAX_MODEL_NAME:
        operator.report(
            {"WARNING"},
            f"'{path.stem}' is {len(path.stem)} characters. The package stores"
            " the entry name in 32 bytes and appends '_l1.aqo' to the file"
            " name, so a longer one is cut off - the file still writes, but it"
            " can read back with no model in it. Keep the name to"
            f" {MAX_MODEL_NAME} characters.",
        )

    with TemporaryDirectory() as tempdir:
        fbxfile = Path(tempdir) / path.with_suffix(".fbx").name

        # Make sure the armature is included for everything that will be exported,
        # or the exported FBX will convert to a broken AQP.
        with _include_parents(context, options):
            fbx_options = _get_fbx_options(options)

            result = fbx_wrapper.save(
                operator, context, filepath=str(fbxfile), **fbx_options
            )

        if "FINISHED" not in result:
            return result

        AssimpModelImporter.scaleHandling = (
            AssimpModelImporter.ScaleHandling.FileScaling
        )

        # TODO: support exporting motions
        model, aqn = cast(
            "tuple[AquaObject, AquaNode]",
            AssimpModelImporter.AssimpAquaConvertFull(
                initialFilePath=str(fbxfile),
                scaleFactor=1,
                preAssignNodeIds=False,
                isNGS=is_ngs,
                aqn=AquaNode(),
                rigidImport=options.get("rigid", False),
            ),
        )

    restore_bone_flags(context, aqn)
    clean_effect_nodes(context, aqn)
    name_root_node(context, aqn, path.stem)

    restored, missing = restore_material_textures(model)
    if missing:
        shown = ", ".join(sorted(missing)[:4])
        more = "..." if len(missing) > 4 else ""
        operator.report(
            {"WARNING"},
            f"{len(missing)} materials have no saved texture list"
            f" ({shown}{more}), so their texture names are whatever the"
            " conversion guessed. Re-import the model with a current version"
            " of this add-on to record them.",
        )

    if stripped := strip_padded_uvs(model):
        operator.report(
            {"INFO"},
            f"Dropped {stripped} zero-filled UV blocks the FBX conversion"
            " padded in.",
        )

    if options.get("override_bounding_radius"):
        set_bounding_radius(model, options.get("bounding_radius", GAME_BOUNDING_RADIUS))

    package = AquaPackage(model)
    package.WritePackage(str(path))

    aqn_path = path.with_suffix(".aqn")
    if overwrite_aqn or not aqn_path.exists():
        aqn_path.write_bytes(aqn.GetBytesNIFL())  # type: ignore

    return {"FINISHED"}


def set_bounding_radius(model, radius: float) -> float:
    """Overwrite the culling radius the conversion measured off the mesh.

    The game's own body models ship -10 here rather than a measured value.
    A radius that is merely correct for the bind pose is too small once an
    animation swings a limb out, and the model blinks out when its sphere
    leaves the view.
    """
    # OBJC and BoundingVolume are value types: read a copy, edit it, and put
    # the whole chain back, or the write goes nowhere.
    objc = model.objc
    bounds = objc.bounds
    previous = float(bounds.boundingRadius)
    bounds.boundingRadius = radius
    objc.bounds = bounds
    model.objc = objc
    return previous


def name_root_node(context: bpy.types.Context, aqn, model_name: str) -> bool:
    """Rebuild node 0, the skeleton root the conversion leaves degenerate.

    The game writes it named after the model and with flags, as in
    `pl_rbd_216590_bw#1CF#0`, sitting at the top of the hierarchy. What
    comes out of the conversion is blank and zeroed: no name, no flags,
    scale (0, 0, 0), and parent/child/sibling all 0, so the node is its own
    parent. A zero scale makes its inverse bind matrix NaN, and importing
    that file back gives an armature and meshes with NaN transforms.

    The importer puts the root's whole string on the armature object, so
    read the name and flags back from there and fall back to the file name.
    """
    from AquaModelLibrary.Data.DataTypes.SetLengthStrings import PSO2String
    from System.Numerics import Matrix4x4, Vector3

    if not len(aqn.nodeList):
        return False

    name, flags = model_name, None
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        # Blender appends .001 to duplicate names; the flags are hex.
        parts = re.sub(r"\.\d+$", "", obj.name).split("#")
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            flags = (int(parts[1], 16), int(parts[2], 16))
        except ValueError:
            continue
        name = parts[0]
        break

    if not name:
        return False

    # Both NODE and PSO2String are value types, so the whole field has to be
    # replaced and the node written back.
    node = aqn.nodeList[0]
    if not str(node.boneName.GetString()):
        node.boneName = PSO2String.GeneratePSO2String(name)
    if flags is not None:
        node.boneShort1, node.boneShort2 = flags

    node.parentId = -1
    node.firstChild = 1 if len(aqn.nodeList) > 1 else -1
    node.nextSibling = -1
    if min(abs(node.scale.X), abs(node.scale.Y), abs(node.scale.Z)) < 1e-6:
        node.scale = Vector3(1.0, 1.0, 1.0)
        node.SetInverseBindPoseMatrix(Matrix4x4.Identity)

    aqn.nodeList[0] = node
    return True


def restore_bone_flags(context: bpy.types.Context, aqn) -> int:
    """Put the second bone flag back on the nodes the conversion flattened.

    Bone names carry two flags, `name#short1#short2`. Coming back out of
    the FBX the second one arrives as a copy of the first, so a skeleton
    that went in with l_breast at (0x1C0, 0) comes out at (0x1C0, 0x1C0),
    and physics bones lose the 0x400 that marks them - the game then
    treats them as ordinary bones. Blender still has the real names, so
    read the flags off those and write them back.
    """
    flags: dict[str, tuple[int, int]] = {}
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        for bone in obj.data.bones:  # type: ignore
            parts = bone.name.split("#")
            if len(parts) < 3:
                continue
            try:
                flags.setdefault(parts[0], (int(parts[1], 16), int(parts[2], 16)))
            except ValueError:
                continue

    if not flags:
        return 0

    fixed = 0
    for index in range(len(aqn.nodeList)):
        node = aqn.nodeList[index]
        name = str(node.boneName.GetString())
        known = flags.get(name)
        if known is None or (node.boneShort1, node.boneShort2) == known:
            continue

        # NODE is a value type: mutate a copy, then put it back.
        node.boneShort1, node.boneShort2 = known
        aqn.nodeList[index] = node
        fixed += 1

    return fixed


def clean_effect_nodes(context: bpy.types.Context, aqn) -> tuple[int, int]:
    """Rebuild the NODO list from the bones that actually belong in it.

    The FBX conversion files every scene node it does not take for a bone
    into the effect-node list, so an export picks up one entry per mesh
    object plus the model empty - a body that left with 10 effect nodes
    comes back with 30. It also keeps the raw Blender names on the real
    ones, "l_finger_03#2#0" instead of "l_finger_03", which is why
    restore_bone_flags never matched them. Re-importing such a file turns
    the junk entries into the whole skeleton: 20 bones instead of 232.

    Effect nodes that map to a Blender bone are kept, renamed to their
    stem, and get their flags back; everything else is dropped. Returns
    (kept, dropped).
    """
    from AquaModelLibrary.Data.DataTypes.SetLengthStrings import PSO2String

    bone_flags: dict[str, tuple[int, int]] = {}
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        for bone in obj.data.bones:  # type: ignore
            parts = bone.name.split("#")
            if len(parts) < 3:
                continue
            try:
                bone_flags.setdefault(
                    parts[0], (int(parts[1], 16), int(parts[2], 16))
                )
            except ValueError:
                continue

    keep = []
    dropped = 0
    for node in aqn.nodoList:
        stem = str(node.boneName.GetString()).split("#")[0]
        flags = bone_flags.get(stem)
        if flags is None:
            dropped += 1
            continue
        node.boneName = PSO2String.GeneratePSO2String(stem)
        node.boneShort1, node.boneShort2 = flags
        keep.append(node)

    aqn.nodoList.Clear()
    for node in keep:
        aqn.nodoList.Add(node)

    return len(keep), dropped


# Blender material names encode "(shaders){blend}[special]name@twoSided@cutoff",
# plus the ".001" Blender adds to duplicates. The plain name is what the
# converted model's MATE entries carry.
_MATERIAL_NAME = re.compile(
    r"^(?:\([^)]*\))?(?:\{[^}]*\})?(?:\[[^\]]*\])?(?P<name>.+?)(?:@-?\d+)*(?:\.\d+)?$"
)


def _material_texture_data() -> dict[str, list[dict]]:
    """The texture register lists saved on materials at import, by name."""
    result: dict[str, list[dict]] = {}
    for mat in bpy.data.materials:
        raw = mat.get("pso2_tsta")
        if not raw:
            continue
        match = _MATERIAL_NAME.match(mat.name)
        if match is None:
            continue
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if data:
            result.setdefault(match.group("name"), data)
    return result


def restore_material_textures(model) -> tuple[int, set[str]]:
    """Rebuild the texture registers from the lists saved at import.

    The FBX between Blender and the conversion names textures after the
    images in the scene, and a material whose images were never found -
    eyes, eyelashes and eyebrows live in other ICE files - comes out as
    placeholder "tex0_d.dds" entries. Environment maps are not in the node
    tree at all and vanish. The game then has dangling texture references,
    which is most of what "the face breaks when exported from Blender"
    looks like. Returns (materials restored, material names with nothing
    saved).
    """
    saved = _material_texture_data()
    missing: set[str] = set()
    if not model.meshList.Count or not model.mateList.Count:
        return 0, missing

    tsta_type = type(model.tstaList[0]) if model.tstaList.Count else None
    tset_type = type(model.tsetList[0]) if model.tsetList.Count else None
    if tsta_type is None or tset_type is None:
        return 0, missing

    from AquaModelLibrary.Data.DataTypes.SetLengthStrings import PSO2String
    from System.Collections.Generic import List as CsList
    from System import Int32

    new_tsta: list = []
    tsta_cache: dict[tuple, int] = {}
    tset_cache: dict[str, int] = {}
    new_tset: list = []
    restored = 0

    def tsta_index(entry: dict, template_index: int) -> int:
        key = (
            entry["name"],
            entry.get("tag", 23),
            entry.get("usage", 0),
            entry.get("uv", 0),
            entry.get("i3", 1),
            entry.get("i4", 1),
            entry.get("i5", 1),
        )
        if key in tsta_cache:
            return tsta_cache[key]
        # Indexing the .NET list boxes a fresh copy each time. Reusing one
        # copy for several entries would leave every entry with the fields
        # of whichever was written last.
        tsta = model.tstaList[template_index]
        tsta.texName = PSO2String.GeneratePSO2String(entry["name"])
        tsta.tag = entry.get("tag", 23)
        tsta.texUsageOrder = entry.get("usage", 0)
        tsta.modelUVSet = entry.get("uv", 0)
        tsta.unkInt3 = entry.get("i3", 1)
        tsta.unkInt4 = entry.get("i4", 1)
        tsta.unkInt5 = entry.get("i5", 1)
        new_tsta.append(tsta)
        tsta_cache[key] = len(new_tsta) - 1
        return tsta_cache[key]

    for mesh_index in range(model.meshList.Count):
        mesh = model.meshList[mesh_index]
        if not 0 <= mesh.mateIndex < model.mateList.Count:
            continue
        mate_name = str(model.mateList[mesh.mateIndex].matName.GetString())
        entries = saved.get(mate_name)
        if not entries:
            missing.add(mate_name)
            continue

        if mate_name not in tset_cache:
            # Take field defaults from whatever the conversion produced for
            # this mesh, then overwrite everything the saved list knows.
            old_tset = model.tsetList[mesh.tsetIndex]
            template_index = (
                old_tset.tstaTexIDs[0]
                if old_tset.tstaTexIDs.Count
                and 0 <= old_tset.tstaTexIDs[0] < model.tstaList.Count
                else 0
            )
            ids = CsList[Int32]()
            for entry in entries:
                ids.Add(tsta_index(entry, template_index))
            tset = old_tset  # value type copy
            tset.tstaTexIDs = ids
            tset.texCount = ids.Count
            new_tset.append(tset)
            tset_cache[mate_name] = len(new_tset) - 1
            restored += 1

        mesh.tsetIndex = tset_cache[mate_name]
        model.meshList[mesh_index] = mesh

    if not restored:
        return 0, missing

    # Any mesh whose material had nothing saved still points into the old
    # lists, so bring its entries across unchanged.
    if missing:
        carried: dict[int, int] = {}
        for mesh_index in range(model.meshList.Count):
            mesh = model.meshList[mesh_index]
            mate_name = str(model.mateList[mesh.mateIndex].matName.GetString())
            if mate_name not in missing:
                continue
            old_index = mesh.tsetIndex
            if old_index not in carried:
                old_tset = model.tsetList[old_index]
                ids = CsList[Int32]()
                for k in range(old_tset.tstaTexIDs.Count):
                    old_tsta_index = old_tset.tstaTexIDs[k]
                    if not 0 <= old_tsta_index < model.tstaList.Count:
                        continue
                    old_tsta = model.tstaList[old_tsta_index]
                    entry = {
                        "name": str(old_tsta.texName.GetString()),
                        "tag": int(old_tsta.tag),
                        "usage": int(old_tsta.texUsageOrder),
                        "uv": int(old_tsta.modelUVSet),
                        "i3": int(old_tsta.unkInt3),
                        "i4": int(old_tsta.unkInt4),
                        "i5": int(old_tsta.unkInt5),
                    }
                    ids.Add(tsta_index(entry, old_tsta_index))
                tset = old_tset
                tset.tstaTexIDs = ids
                tset.texCount = ids.Count
                new_tset.append(tset)
                carried[old_index] = len(new_tset) - 1
            mesh.tsetIndex = carried[old_index]
            model.meshList[mesh_index] = mesh

    model.tstaList.Clear()
    for tsta in new_tsta:
        model.tstaList.Add(tsta)
    model.tsetList.Clear()
    for tset in new_tset:
        model.tsetList.Add(tset)

    _rebuild_texf(model, new_tsta)
    _sync_texture_counts(model)

    return restored, missing


def _sync_texture_counts(model) -> None:
    """Put the header's texture counts back in step with the lists.

    OBJC stores its own count for each list, and the writer emits exactly
    that many entries rather than measuring the list. Rewriting the
    registers without it writes the old number of them - a face rebuilt
    with 19 comes out holding 15, the last four silently cut, so the eyes
    lose their textures. OBJC is a value type, so the whole struct has to
    go back.
    """
    objc = model.objc
    objc.tstaCount = model.tstaList.Count
    objc.tsetCount = model.tsetList.Count
    objc.texfCount = model.texfList.Count
    model.objc = objc


def _rebuild_texf(model, new_tsta: list) -> None:
    """Put the file's texture name table in step with the registers.

    TEXF is a second, separate list of texture names, and it is the one the
    written file carries - the header's texfCount comes from it. Rewriting
    only the registers leaves it holding whatever the conversion guessed,
    so the model still ships placeholder names and the game finds no
    textures. TEXF entries are unique by name, unlike registers.
    """
    from AquaModelLibrary.Data.DataTypes.SetLengthStrings import PSO2String

    if not model.texfList.Count:
        return

    template = model.texfList[0]
    seen: set[str] = set()
    entries = []
    for tsta in new_tsta:
        name = str(tsta.texName.GetString())
        if name in seen:
            continue
        seen.add(name)
        texf = template  # TEXF is a value type; indexing boxed a copy.
        texf.texName = PSO2String.GeneratePSO2String(name)
        entries.append(texf)

    model.texfList.Clear()
    for texf in entries:
        model.texfList.Add(texf)

    if hasattr(model, "texFUnicodeNames"):
        model.texFUnicodeNames.Clear()
        for name in (str(t.texName.GetString()) for t in entries):
            model.texFUnicodeNames.Add(name)


def strip_padded_uvs(model) -> int:
    """Clear secondary UV blocks that are zero at every vertex.

    A scene imported before the importer learned to drop the conversion's
    padding still has eight UV layers on every mesh, and they come through
    the FBX as real uv2-uv4 blocks full of zeros - a face grows by half
    its file size and carries a vertex layout the game never wrote.
    """
    cleared = 0
    for index in range(model.vtxlList.Count):
        vtxl = model.vtxlList[index]
        for attribute in ("uv2List", "uv3List", "uv4List"):
            uv_list = getattr(vtxl, attribute, None)
            if uv_list is None or not uv_list.Count:
                continue
            if all(
                abs(uv_list[k].X) < 1e-9 and abs(uv_list[k].Y) < 1e-9
                for k in range(uv_list.Count)
            ):
                uv_list.Clear()
                cleared += 1

    return cleared


@contextmanager
def _include_parents(context: bpy.types.Context, fbx_options: ExportOptions):
    shown_objects: set[bpy.types.Object] = set()
    viewport_shown_objects: set[bpy.types.Object] = set()
    restore: list[tuple] = []

    use_visible = fbx_options.get("use_visible", False)
    use_selection = fbx_options.get("use_selection", False)

    if use_selection:
        ctx_objects = context.selected_objects
    else:
        assert context.view_layer is not None
        ctx_objects = context.view_layer.objects

    if ctx_objects is None:
        raise TypeError()

    if use_selection:
        meshes = list(_get_selected_meshes(ctx_objects))
    elif use_visible:
        meshes = list(_get_visible_meshes(ctx_objects))
    else:
        meshes = [obj for obj in ctx_objects if obj.type == "MESH"]

    required: set[bpy.types.Object] = set()
    for mesh in meshes:
        required |= _required_objects(mesh)

    try:
        _reveal_collections(context, required, restore)

        # If we are only including visible objects, make sure everything the
        # exported meshes need is visible too.
        if use_visible:
            for obj in required:
                if obj.hide_get():
                    obj.hide_set(False)
                    shown_objects.add(obj)

                if obj.hide_viewport:
                    obj.hide_viewport = False
                    viewport_shown_objects.add(obj)

        # Same for selection.
        if use_selection:
            selection = set(context.selected_objects or []) | required

            with context.temp_override(selected_objects=list(selection)):  # type: ignore
                yield
        else:
            yield
    finally:
        for obj in shown_objects:
            obj.hide_set(True)

        for obj in viewport_shown_objects:
            obj.hide_viewport = True

        for target, attribute in reversed(restore):
            setattr(target, attribute, True)


def _required_objects(mesh: bpy.types.Object) -> set[bpy.types.Object]:
    """What has to go into the file alongside this mesh.

    Parents, and the armatures that deform it. A mesh imported by this
    add-on is parented to its armature, but a mesh does not have to be: an
    Armature modifier binds it just as well, and one that has been joined,
    duplicated or rebuilt in Blender often ends up bound that way with no
    parent at all. Following only the parent link then finds nothing, the
    rig is left out of the export, and the model is written with no
    skeleton - which is the shape of "it exported without the armature".
    """
    found: set[bpy.types.Object] = set()
    queue = [mesh]

    while queue:
        obj = queue.pop()
        related = [obj.parent]
        related += [
            modifier.object for modifier in obj.modifiers if modifier.type == "ARMATURE"
        ]

        for other in related:
            if other is not None and other not in found:
                found.add(other)
                queue.append(other)

    return found


def _reveal_collections(
    context: bpy.types.Context, objects: Iterable[bpy.types.Object], restore: list
):
    """Put the collections these objects live in back in the view layer.

    Hiding a collection is not the same as hiding an object: an excluded
    collection drops its objects out of the view layer entirely, so the FBX
    exporter never sees them however the include options are set. Someone
    who sorts a character into collections and switches the rig's off gets
    a model exported with no skeleton - and the export reports success,
    since a file is written either way. It reads back as 974 KB of nothing
    where a whole one is 1.06 MB.

    Only the collections holding the objects passed in are touched, which
    is the parents of what is being exported. A collection switched off to
    leave its meshes out stays off.
    """
    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return

    for obj in objects:
        for layer in _layer_chain(view_layer.layer_collection, obj) or ():
            for target, attribute in (
                (layer, "exclude"),
                (layer, "hide_viewport"),
                (layer.collection, "hide_viewport"),
            ):
                if getattr(target, attribute, False):
                    setattr(target, attribute, False)
                    restore.append((target, attribute))


def _layer_chain(layer_collection, obj, chain: list | None = None) -> list | None:
    """The view layer's collections from the root down to the one holding obj."""
    chain = (chain or []) + [layer_collection]
    if obj.name in layer_collection.collection.objects:
        return chain

    for child in layer_collection.children:
        if found := _layer_chain(child, obj, chain):
            return found

    return None


def _get_visible_meshes(objects: Iterable[bpy.types.Object]):
    return (obj for obj in objects if obj.type == "MESH" and obj.visible_get())


def _get_selected_meshes(objects: Iterable[bpy.types.Object]):
    return (obj for obj in objects if obj.type == "MESH" and obj.select_get())


def _get_fbx_options(options: ExportOptions):
    result = FbxExportOptions()

    for key in get_type_hints(FbxExportOptions):
        if key in options:
            result[key] = options[key]

    return result
