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
    name_root_node(context, aqn, path.stem)

    package = AquaPackage(model)
    package.WritePackage(str(path))

    aqn_path = path.with_suffix(".aqn")
    if overwrite_aqn or not aqn_path.exists():
        aqn_path.write_bytes(aqn.GetBytesNIFL())  # type: ignore

    return {"FINISHED"}


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


@contextmanager
def _include_parents(context: bpy.types.Context, fbx_options: ExportOptions):
    shown_objects: set[bpy.types.Object] = set()
    viewport_shown_objects: set[bpy.types.Object] = set()

    use_visible = fbx_options.get("use_visible", False)
    use_selection = fbx_options.get("use_selection", False)

    if use_selection:
        ctx_objects = context.selected_objects
    else:
        assert context.view_layer is not None
        ctx_objects = context.view_layer.objects

    if ctx_objects is None:
        raise TypeError()

    try:
        # If we are only including visible objects, make sure the parents of any
        # visible objects are also visible.
        if use_visible:
            for obj in _get_visible_meshes(ctx_objects):
                while obj := obj.parent:
                    if obj.hide_get():
                        obj.hide_set(False)
                        shown_objects.add(obj)

                    if obj.hide_viewport:
                        obj.hide_viewport = False
                        viewport_shown_objects.add(obj)

        # If we are only including selected objects, make sure the parents of any
        # selected objects are also selected.
        if use_selection:
            selection = set(context.selected_objects or [])
            for obj in _get_selected_meshes(ctx_objects):
                while obj := obj.parent:
                    if not obj.select_get():
                        selection.add(obj)

            with context.temp_override(selected_objects=list(selection)):  # type: ignore
                yield
        else:
            yield
    finally:
        for obj in shown_objects:
            obj.hide_set(True)

        for obj in viewport_shown_objects:
            obj.hide_viewport = True


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
