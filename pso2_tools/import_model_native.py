"""Build the Blender scene for a model directly, with no FBX in between.

The Windows import path runs AquaModelLibrary.Native (C++/CLI wrapping the
Autodesk FBX SDK) to write a temporary FBX, then hands it to Blender's FBX
importer. C++/CLI does not exist off Windows, so this module produces the
same scene straight from the AquaObject/AquaNode data.

"The same scene" is a real contract, not a goal: everything downstream -
material building, character files, motions, shape adjust, and export back
to .aqp - was written against what io_scene_fbx makes of the converter's
FBX. So this follows the two of them step for step:

- The armature object is aqn node 0 (the converter marks it eRoot, and
  io_scene_fbx turns 'Root' models into armatures, not bones), named
  "name#BS1#BS2" with the flags in hex, carrying the Y-up-to-Z-up global
  rotation.
- Bones are the remaining nodes plus the NODO effect nodes, named
  "name#BS1#BS2"; NODE bones also get their index in a pso2_bone_id
  property, matching fbx_wrapper's rename of the "(id)" prefixes.
- An edit bone's matrix is the bind chain times the X,Y bone correction, so
  bones run down PSO2's +X. Tail lengths and use_connect come from
  io_scene_fbx's rules.
- NODO rest matrices replicate a converter quirk: it writes the INVERSE
  world matrix into their bind pose entries, so their rest pose lands
  there and a pose transform puts them back. Model export restores that
  pose before converting, so the round trip still writes correct values -
  but only if this importer reproduces the quirk rather than fixing it.
- UVs come in V-flipped on layers named UVChannel_1..10 (9 and 10 carry
  vertColor2s), vertex colors land in a corner byte-color attribute named
  by Blender ("Attribute"), and zero-filled padding channels are never
  created - the same net result as the FBX path plus its strip pass.
"""

from typing import TYPE_CHECKING

import bpy
import numpy as np
from bpy_extras.io_utils import axis_conversion
from mathutils import Matrix

from . import scene_props
from .debug import debug_print

if TYPE_CHECKING:
    from .import_model import ImportOptions

# FBX "OpenGL" axes (Y up, -Z forward) into Blender's Z up, as io_scene_fbx
# computes for the converter's files.
_GLOBAL_MATRIX = axis_conversion(from_forward="-Z", from_up="Y").to_4x4()

# io_scene_fbx's bone_correction_matrix for primary_bone_axis=X,
# secondary_bone_axis=Y - the axes every armature here is built with.
_BONE_CORRECTION = axis_conversion(
    from_forward="X", from_up="Y", to_forward="Y", to_up="X"
).to_4x4()

_MIN_BONE_LENGTH = 0.01


class _Bone:
    """One future edit bone: an aqn NODE (index >= 1) or NODO."""

    def __init__(self, name: str, node_index: int | None):
        self.name = name
        self.node_index = node_index  # None for NODOs
        self.children: list[_Bone] = []
        self.local_bind = Matrix()  # parent-relative bind, unnormalized
        self.local_trs = Matrix()  # the converter's Lcl transform
        self.edit_matrix = Matrix()  # armature space, correction applied
        self.chain_matrix = Matrix()  # normalized bind chain, no correction
        self.bl_name = ""  # actual Blender name after any dedup


def import_model(
    operator: bpy.types.Operator,
    context: bpy.types.Context,
    model,
    skeleton,
    name: str,
    options: "ImportOptions",
):
    from Pso2Tools.Interop import ModelInterop

    nodes = ModelInterop.GetNodes(skeleton)
    materials = ModelInterop.GetMaterials(model)

    global_matrix = _get_global_matrix(context)

    if bpy.ops.object.select_all.poll():
        bpy.ops.object.select_all(action="DESELECT")

    armature_obj, bone_names = _build_armature(context, nodes, global_matrix)

    material_map = _create_materials(materials)
    mesh_mapping = np.frombuffer(bytes(materials.MeshMapping), dtype=np.int32)

    for mesh_id in range(ModelInterop.GetMeshCount(model)):
        mesh_data = ModelInterop.GetMesh(model, mesh_id)
        material = (
            material_map[mesh_mapping[mesh_id]]
            if mesh_id < len(mesh_mapping)
            else None
        )
        _build_mesh(context, mesh_data, material, armature_obj, bone_names, options)

    if context.view_layer:
        context.view_layer.objects.active = armature_obj

    debug_print(
        f"Native import: {name}:"
        f" {nodes.NodeCount} nodes, {nodes.NodoCount} effect nodes,"
        f" {ModelInterop.GetMeshCount(model)} meshes"
    )

    return {"FINISHED"}


def _get_global_matrix(context: bpy.types.Context) -> Matrix:
    # The converter stamps its FBX with meter units (UnitScaleFactor 100),
    # and io_scene_fbx scales by that over the scene's own unit factor.
    try:
        from io_scene_fbx.fbx_utils import units_blender_to_fbx_factor

        scale = 100.0 / units_blender_to_fbx_factor(context.scene)
    except Exception:
        scale = 1.0

    return Matrix.Scale(scale, 4) @ _GLOBAL_MATRIX


def _matrix_from_numerics(flat: np.ndarray) -> Matrix:
    """System.Numerics stores row-vector matrices; mathutils uses columns."""
    return Matrix(
        (
            (flat[0], flat[4], flat[8], flat[12]),
            (flat[1], flat[5], flat[9], flat[13]),
            (flat[2], flat[6], flat[10], flat[14]),
            (flat[3], flat[7], flat[11], flat[15]),
        )
    )


def _build_armature(context: bpy.types.Context, nodes, global_matrix: Matrix):
    """Create the armature object and its bones.

    Returns the object and a map of aqn node index -> Blender bone name for
    the skinning to target.
    """
    node_count = nodes.NodeCount
    node_names = list(nodes.NodeNames)
    node_parents = np.frombuffer(bytes(nodes.NodeParents), dtype=np.int32)
    node_shorts = np.frombuffer(bytes(nodes.NodeShorts), dtype=np.int32)
    inv_bind = np.frombuffer(bytes(nodes.NodeInvBind), dtype=np.float32)

    nodo_count = nodes.NodoCount
    nodo_names = list(nodes.NodoNames)
    nodo_parents = np.frombuffer(bytes(nodes.NodoParents), dtype=np.int32)
    nodo_shorts = np.frombuffer(bytes(nodes.NodoShorts), dtype=np.int32)
    nodo_local = np.frombuffer(bytes(nodes.NodoLocal), dtype=np.float32)

    world_bind = [
        _matrix_from_numerics(inv_bind[i * 16 : i * 16 + 16]).inverted_safe()
        for i in range(node_count)
    ]

    def metadata_name(name: str, shorts: np.ndarray, index: int) -> str:
        return f"{name}#{shorts[index * 2]:X}#{shorts[index * 2 + 1]:X}"

    # Node 0 becomes the armature object itself, exactly as io_scene_fbx
    # treats the converter's eRoot skeleton node. Its bind pose entry in the
    # FBX ends up identity (the converter writes it twice, identity last),
    # so the bone chain hangs off identity while the object's own transform
    # carries node 0's local matrix - both replicated here.
    armature_name = (
        metadata_name(node_names[0], node_shorts, 0) if node_count else "Armature"
    )

    armature_data = bpy.data.armatures.new(armature_name)
    armature_obj = bpy.data.objects.new(armature_name, armature_data)
    armature_obj.matrix_basis = global_matrix @ world_bind[0] if node_count else global_matrix

    assert context.view_layer is not None
    context.view_layer.active_layer_collection.collection.objects.link(armature_obj)
    armature_obj.select_set(True)

    # Build the helper hierarchy: NODE bones in index order, then NODOs, so
    # children keep the converter's order (nodes first, effect nodes after).
    bones: list[_Bone | None] = [None] * node_count
    roots: list[_Bone] = []
    nodos: list[tuple[_Bone, int]] = []

    for i in range(1, node_count):
        bone = _Bone(metadata_name(node_names[i], node_shorts, i), i)
        bones[i] = bone

    for i in range(1, node_count):
        bone = bones[i]
        assert bone is not None
        parent_id = int(node_parents[i])
        if parent_id == 0:
            roots.append(bone)
        elif 0 < parent_id < node_count and bones[parent_id] is not None:
            bones[parent_id].children.append(bone)  # type: ignore[union-attr]
        else:
            # The converter leaves such nodes unparented and the FBX drops
            # them; match that rather than inventing a place for them.
            debug_print(f"Skipping node {i} with parent {parent_id}")
            bones[i] = None
            continue

        parent_world = world_bind[parent_id] if parent_id > 0 else Matrix()
        bone.local_bind = parent_world.inverted_safe() @ world_bind[i]
        bone.local_trs = bone.local_bind

    for i in range(nodo_count):
        parent_id = int(nodo_parents[i])
        if not 0 <= parent_id < node_count:
            debug_print(f"Skipping effect node {i} with parent {parent_id}")
            continue

        bone = _Bone(metadata_name(nodo_names[i], nodo_shorts, i), None)
        local = _matrix_from_numerics(nodo_local[i * 16 : i * 16 + 16])
        parent_world = world_bind[parent_id] if parent_id > 0 else Matrix()

        # Converter quirk, kept on purpose: NODO bind pose entries hold the
        # INVERSE world matrix, so the rest pose lands there and the pose
        # transform set below carries the difference.
        world = parent_world @ local
        bone.local_bind = parent_world.inverted_safe() @ world.inverted_safe()
        bone.local_trs = local

        if parent_id == 0:
            roots.append(bone)
        else:
            parent_bone = bones[parent_id]
            if parent_bone is None:
                debug_print(f"Skipping effect node {i} under dropped node {parent_id}")
                continue
            parent_bone.children.append(bone)
        nodos.append((bone, parent_id))

    # Compute edit matrices: parent chain of normalized local binds, times
    # the bone correction (io_scene_fbx telescopes the per-bone pre/post
    # correction pair down to exactly this).
    def compute_matrices(bone: _Bone, parent_chain: Matrix):
        bone.chain_matrix = parent_chain @ bone.local_bind.normalized()
        bone.edit_matrix = bone.chain_matrix @ _BONE_CORRECTION
        for child in bone.children:
            compute_matrices(child, bone.chain_matrix)

    for bone in roots:
        compute_matrices(bone, Matrix())

    # Create edit bones, sizing tails like io_scene_fbx build_skeleton.
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")

    try:
        from io_scene_fbx.fbx_utils import similar_values_iter
    except Exception:  # pragma: no cover - io_scene_fbx ships with Blender

        def similar_values_iter(v1, v2, e=1e-6):
            return all(
                a == b or ((a + b) and abs((a - b) / (a + b)) <= e)
                for a, b in zip(v1, v2, strict=False)
            )

    edit_bones = armature_data.edit_bones

    def build_bones(bone: _Bone, parent_edit, parent_size: float):
        edit = edit_bones.new(name=bone.name)
        bone.bl_name = edit.name

        sizes = [
            child.local_bind.normalized().to_translation().magnitude
            for child in bone.children
        ]
        size = (sum(sizes) / len(sizes)) if sizes else parent_size

        edit.tail = (0.0, max(_MIN_BONE_LENGTH, size), 0.0)
        edit.matrix = bone.edit_matrix

        if parent_edit is not None:
            edit.parent = parent_edit

        for child in bone.children:
            child_edit = build_bones(child, edit, size)
            if similar_values_iter(edit.tail, child_edit.head):
                child_edit.use_connect = True

        return edit

    for bone in roots:
        build_bones(bone, None, 1.0)

    # Store bone IDs where fbx_wrapper's import patch puts them. Custom
    # properties on edit bones persist onto the bones.
    def store_ids(bone: _Bone):
        if bone.node_index is not None:
            edit_bones[bone.bl_name][scene_props.BONE_ID] = bone.node_index
        for child in bone.children:
            store_ids(child)

    for bone in roots:
        store_ids(bone)

    bpy.ops.object.mode_set(mode="OBJECT")
    view_layer.objects.active = previous_active

    # NODE bones rest exactly at their bind pose, so their pose transform is
    # identity and the default is already right. NODOs rest at the quirk
    # position; the pose puts their heads back on the model.
    correction_inv = _BONE_CORRECTION.inverted()
    for bone, _parent_id in nodos:
        if not bone.bl_name:
            continue
        pose_bone = armature_obj.pose.bones[bone.bl_name]
        pose_bone.matrix_basis = (
            correction_inv
            @ bone.local_bind.inverted_safe()
            @ bone.local_trs
            @ _BONE_CORRECTION
        )

    bone_names = {
        bone.node_index: bone.bl_name
        for bone in filter(None, bones)
        if bone.node_index is not None and bone.bl_name
    }

    return armature_obj, bone_names


def _create_materials(materials) -> list[bpy.types.Material]:
    """One new Blender material per unique converter material.

    The FBX path also creates fresh datablocks each import and lets Blender
    dedup names with .001 suffixes, which _import_models relies on to tell
    new materials from old.
    """
    return [bpy.data.materials.new(name) for name in materials.Names]


def _build_mesh(
    context: bpy.types.Context,
    mesh_data,
    material: bpy.types.Material | None,
    armature_obj: bpy.types.Object,
    bone_names: dict[int, str],
    options: "ImportOptions",
):
    name = mesh_data.MeshName
    vertex_count = mesh_data.VertexCount

    positions = np.frombuffer(bytes(mesh_data.Positions), dtype=np.float32)
    triangles = np.frombuffer(bytes(mesh_data.Triangles), dtype=np.int32)

    mesh = bpy.data.meshes.new(f"{name}_mesh")

    mesh.vertices.add(vertex_count)
    mesh.vertices.foreach_set("co", positions)

    loop_count = len(triangles)
    mesh.loops.add(loop_count)
    mesh.loops.foreach_set("vertex_index", triangles)

    poly_count = loop_count // 3
    mesh.polygons.add(poly_count)
    mesh.polygons.foreach_set("loop_start", np.arange(0, loop_count, 3, dtype=np.int32))

    mesh.validate(clean_customdata=False)
    mesh.update()

    # validate() may drop invalid geometry, so per-corner data is gathered
    # with the loops the mesh actually kept.
    final_loops = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", final_loops)

    if len(mesh_data.Normals) and options.get("use_custom_normals", True):
        normals = np.frombuffer(bytes(mesh_data.Normals), dtype=np.float32)
        mesh.normals_split_custom_set_from_vertices(
            normals.reshape(vertex_count, 3).tolist()
        )

    uv_channels = [np.frombuffer(bytes(blob), dtype=np.float32) for blob in mesh_data.Uvs]
    last_used = max(
        (i for i, data in enumerate(uv_channels) if np.any(np.abs(data) >= 1e-9)),
        default=0,
    )
    for channel, data in enumerate(uv_channels[: last_used + 1]):
        # Retain intermediate empty channels so FBX export keeps the later
        # channel's semantic index, matching the FBX import path.
        if not len(data):
            data = np.zeros(vertex_count * 2, dtype=np.float32)

        layer = mesh.uv_layers.new(name=f"UVChannel_{channel + 1}", do_init=False)
        if layer is None:
            debug_print(f"{name}: could not add UVChannel_{channel + 1}")
            continue
        per_loop = data.reshape(vertex_count, 2)[final_loops]
        layer.uv.foreach_set("vector", per_loop.ravel())

    colors_type = options.get("colors_type", "SRGB")
    if len(mesh_data.Colors) and colors_type != "NONE":
        bgra = np.frombuffer(bytes(mesh_data.Colors), dtype=np.uint8)
        bgra = bgra.reshape(len(bgra) // 4, 4)
        if len(bgra) < vertex_count:
            bgra = np.pad(bgra, ((0, vertex_count - len(bgra)), (0, 0)))

        rgba = bgra[:, [2, 1, 0, 3]].astype(np.float32) / 255.0
        per_loop = rgba[final_loops].ravel()

        # Same layer setup as io_scene_fbx: the converter leaves the FBX
        # color element unnamed, and Blender names the attribute for it.
        if colors_type == "SRGB":
            attribute = mesh.color_attributes.new(
                name="", type="BYTE_COLOR", domain="CORNER"
            )
            attribute.data.foreach_set("color_srgb", per_loop)
        else:
            attribute = mesh.color_attributes.new(
                name="", type="FLOAT_COLOR", domain="CORNER"
            )
            attribute.data.foreach_set("color", per_loop)

    if material is not None:
        mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)

    assert context.view_layer is not None
    context.view_layer.active_layer_collection.collection.objects.link(obj)
    obj.select_set(True)

    obj.parent = armature_obj
    modifier = obj.modifiers.new(name=armature_obj.name, type="ARMATURE")
    modifier.object = armature_obj  # type: ignore[attr-defined]

    _apply_weights(obj, mesh_data, bone_names)

    return obj


def _apply_weights(obj: bpy.types.Object, mesh_data, bone_names: dict[int, str]):
    if not len(mesh_data.Weights):
        return

    vertex_count = mesh_data.VertexCount
    weights = np.frombuffer(bytes(mesh_data.Weights), dtype=np.float32).reshape(
        vertex_count, 4
    )
    indices = np.frombuffer(bytes(mesh_data.WeightIndices), dtype=np.int32).reshape(
        vertex_count, 4
    )
    palette = np.frombuffer(bytes(mesh_data.BonePalette), dtype=np.int32)

    for palette_index, node_index in enumerate(palette):
        # Palette entries the converter could not resolve fall back to node
        # 0, which is the armature, not a bone - io_scene_fbx drops those
        # clusters and so does this.
        bone_name = bone_names.get(int(node_index))
        if bone_name is None:
            continue

        mask = indices == palette_index
        if not mask.any():
            continue

        vertex_ids, _slots = np.nonzero(mask)
        vertex_weights = weights[mask]

        group = obj.vertex_groups.get(bone_name) or obj.vertex_groups.new(
            name=bone_name
        )

        # Group identical weights per add() like io_scene_fbx's
        # add_vgroup_to_objects; zero weights are kept, as the converter
        # emitted a cluster entry for every in-range palette slot.
        order = np.argsort(vertex_weights, kind="stable")
        sorted_weights = vertex_weights[order]
        sorted_vertices = vertex_ids[order]
        boundaries = np.nonzero(np.diff(sorted_weights))[0] + 1

        for chunk_ids, chunk_weights in zip(
            np.split(sorted_vertices, boundaries),
            np.split(sorted_weights, boundaries),
            strict=True,
        ):
            group.add(chunk_ids.tolist(), float(chunk_weights[0]), "REPLACE")
