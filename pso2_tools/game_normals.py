"""Turn a skinned mesh's normals the way the game's vertex shader does.

The game skins a normal like a position: it multiplies it by the weighted
sum of its bones' matrices and normalizes the result, scale and all. Blender
keeps custom normals relative to the faces around each vertex instead, so
they follow what the surface does rather than what the bones do. The two
agree on a rigid bone and part ways under the body sliders' uneven scales.

That matters where two meshes meet. The face's neck skirt ends on a ring of
vertices that sits on the body's neck with the very same normals (0.04
degrees apart in the files, 0.0 in a captured frame), but the faces around
the ring are the skirt's on one side and the neck's on the other. Deformed
by Blender, the two normals part by about 18 degrees (up to 21) and the
ring shows as a band around the neck, with a V at each side of the jaw.

Geometry nodes reproduce the game's sum around the armature modifier. The
first group adds a probe vertex one unit along each vertex's normal, with
the vertex's own weights; the armature moves the pair together, and since
skinning is linear, the probe's offset from its vertex is then the game's
normal before normalizing. The second group reads it off, sets it as the
vertex's normal and drops the probes. Measured against the sum done in
Python on the posed bones: 0.007 degrees on average.
"""

import bpy

PROBES_NAME = "PSO2 Normal Probes"
NORMALS_NAME = "PSO2 Game Normals"

# Rebuilt when this changes.
NODES_VERSION = 1
_VERSION_KEY = "pso2_nodes_version"

# How far out each probe sits. Skinning is linear, so the length drops out;
# a long one keeps the difference well above the positions' rounding.
PROBE_LENGTH = 1.0

_PROBE = "pso2_probe"
_PROBE_OFFSET = "pso2_probe_offset"
_GAME_NORMAL = "pso2_game_normal"


def supported() -> bool:
    """Free custom normals from geometry nodes came in Blender 4.5."""
    return hasattr(bpy.types, "GeometryNodeSetMeshNormal")


def _new_tree(name: str) -> bpy.types.NodeTree:
    tree = bpy.data.node_groups.get(name)
    if tree is not None and tree.bl_idname != "GeometryNodeTree":
        tree = None
    if tree is None:
        tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.nodes.clear()
    tree.interface.clear()
    tree.interface.new_socket(
        "Geometry", in_out="INPUT", socket_type="NodeSocketGeometry"
    )
    tree.interface.new_socket(
        "Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry"
    )
    tree[_VERSION_KEY] = NODES_VERSION
    return tree


def _current(name: str) -> bpy.types.NodeTree | None:
    tree = bpy.data.node_groups.get(name)
    if tree is not None and tree.get(_VERSION_KEY) == NODES_VERSION:
        return tree
    return None


def _probes_tree() -> bpy.types.NodeTree:
    """Mesh in, mesh out with a probe vertex after the last, one per vertex."""
    if tree := _current(PROBES_NAME):
        return tree

    tree = _new_tree(PROBES_NAME)
    nodes, links = tree.nodes, tree.links
    inputs = nodes.new("NodeGroupInput")
    outputs = nodes.new("NodeGroupOutput")

    # The normal on the corners carries the custom normals; on a vertex it
    # is their average, which is exact for the importer's per-vertex ones.
    normal = nodes.new("GeometryNodeInputNormal")
    on_corners = nodes.new("GeometryNodeFieldOnDomain")
    on_corners.domain = "CORNER"
    on_corners.data_type = "FLOAT_VECTOR"
    links.new(normal.outputs["Normal"], on_corners.inputs[0])
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "SCALE"
    length.inputs["Scale"].default_value = PROBE_LENGTH  # type: ignore
    links.new(on_corners.outputs[0], length.inputs[0])

    # Read on the mesh, then carried by the copies with the vertex groups.
    offset = nodes.new("GeometryNodeStoreNamedAttribute")
    offset.data_type = "FLOAT_VECTOR"
    offset.domain = "POINT"
    offset.inputs["Name"].default_value = _PROBE_OFFSET  # type: ignore
    links.new(inputs.outputs[0], offset.inputs["Geometry"])
    links.new(length.outputs[0], offset.inputs["Value"])

    # On a mesh this gives loose vertices, not a point cloud.
    copies = nodes.new("GeometryNodeDuplicateElements")
    copies.domain = "POINT"
    links.new(offset.outputs[0], copies.inputs["Geometry"])
    move = nodes.new("GeometryNodeSetPosition")
    read_offset = nodes.new("GeometryNodeInputNamedAttribute")
    read_offset.data_type = "FLOAT_VECTOR"
    read_offset.inputs["Name"].default_value = _PROBE_OFFSET  # type: ignore
    links.new(copies.outputs["Geometry"], move.inputs["Geometry"])
    links.new(read_offset.outputs["Attribute"], move.inputs["Offset"])
    mark = nodes.new("GeometryNodeStoreNamedAttribute")
    mark.data_type = "BOOLEAN"
    mark.domain = "POINT"
    mark.inputs["Name"].default_value = _PROBE  # type: ignore
    mark.inputs["Value"].default_value = True  # type: ignore
    links.new(move.outputs[0], mark.inputs["Geometry"])

    join = nodes.new("GeometryNodeJoinGeometry")
    links.new(mark.outputs[0], join.inputs[0])
    links.new(inputs.outputs[0], join.inputs[0])
    tidy = nodes.new("GeometryNodeRemoveAttribute")
    tidy.inputs["Name"].default_value = _PROBE_OFFSET  # type: ignore
    links.new(join.outputs[0], tidy.inputs["Geometry"])
    links.new(tidy.outputs[0], outputs.inputs[0])
    return tree


def _normals_tree() -> bpy.types.NodeTree:
    """Skinned mesh and probes in, the mesh with the game's normals out."""
    if tree := _current(NORMALS_NAME):
        return tree

    tree = _new_tree(NORMALS_NAME)
    nodes, links = tree.nodes, tree.links
    inputs = nodes.new("NodeGroupInput")
    outputs = nodes.new("NodeGroupOutput")

    # Each vertex's probe is half the vertices away, whichever half the join
    # put first.
    size = nodes.new("GeometryNodeAttributeDomainSize")
    links.new(inputs.outputs[0], size.inputs[0])
    half = nodes.new("FunctionNodeIntegerMath")
    half.operation = "DIVIDE_FLOOR"
    links.new(size.outputs["Point Count"], half.inputs[0])
    half.inputs[1].default_value = 2  # type: ignore
    index = nodes.new("GeometryNodeInputIndex")
    across = nodes.new("FunctionNodeIntegerMath")
    across.operation = "ADD"
    links.new(index.outputs[0], across.inputs[0])
    links.new(half.outputs[0], across.inputs[1])
    wrap = nodes.new("FunctionNodeIntegerMath")
    wrap.operation = "MODULO"
    links.new(across.outputs[0], wrap.inputs[0])
    links.new(size.outputs["Point Count"], wrap.inputs[1])

    position = nodes.new("GeometryNodeInputPosition")
    probe = nodes.new("GeometryNodeSampleIndex")
    probe.data_type = "FLOAT_VECTOR"
    probe.domain = "POINT"
    links.new(inputs.outputs[0], probe.inputs["Geometry"])
    links.new(position.outputs[0], probe.inputs["Value"])
    links.new(wrap.outputs[0], probe.inputs["Index"])
    towards = nodes.new("ShaderNodeVectorMath")
    towards.operation = "SUBTRACT"
    links.new(probe.outputs[0], towards.inputs[0])
    links.new(position.outputs[0], towards.inputs[1])
    unit = nodes.new("ShaderNodeVectorMath")
    unit.operation = "NORMALIZE"
    links.new(towards.outputs[0], unit.inputs[0])

    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type = "FLOAT_VECTOR"
    store.domain = "POINT"
    store.inputs["Name"].default_value = _GAME_NORMAL  # type: ignore
    links.new(inputs.outputs[0], store.inputs["Geometry"])
    links.new(unit.outputs[0], store.inputs["Value"])

    is_probe = nodes.new("GeometryNodeInputNamedAttribute")
    is_probe.data_type = "BOOLEAN"
    is_probe.inputs["Name"].default_value = _PROBE  # type: ignore
    drop = nodes.new("GeometryNodeDeleteGeometry")
    drop.domain = "POINT"
    links.new(store.outputs[0], drop.inputs["Geometry"])
    links.new(is_probe.outputs["Attribute"], drop.inputs["Selection"])

    set_normal = nodes.new("GeometryNodeSetMeshNormal")
    set_normal.mode = "FREE"
    set_normal.domain = "POINT"
    read = nodes.new("GeometryNodeInputNamedAttribute")
    read.data_type = "FLOAT_VECTOR"
    read.inputs["Name"].default_value = _GAME_NORMAL  # type: ignore
    links.new(drop.outputs[0], set_normal.inputs["Mesh"])
    links.new(read.outputs["Attribute"], set_normal.inputs["Custom Normal"])

    tidy = set_normal
    for name in (_GAME_NORMAL, _PROBE):
        remove = nodes.new("GeometryNodeRemoveAttribute")
        remove.inputs["Name"].default_value = name  # type: ignore
        links.new(tidy.outputs[0], remove.inputs["Geometry"])
        tidy = remove
    links.new(tidy.outputs[0], outputs.inputs[0])
    return tree


def _runs(modifier: bpy.types.Modifier, name: str) -> bool:
    return bool(
        modifier.type == "NODES"
        and modifier.node_group  # type: ignore
        and modifier.node_group.name == name  # type: ignore
    )


def is_probes(modifier: bpy.types.Modifier) -> bool:
    return _runs(modifier, PROBES_NAME)


def has_game_normals(obj: bpy.types.Object) -> bool:
    return any(_runs(m, NORMALS_NAME) for m in obj.modifiers)


def _move(obj: bpy.types.Object, modifier: bpy.types.Modifier, index: int) -> None:
    modifiers = obj.modifiers
    modifiers.move(list(modifiers).index(modifier), index)  # type: ignore


def add(obj: bpy.types.Object) -> bool:
    """Wrap the mesh's armature modifier in the two groups; False if it has
    none, already has them, or this Blender cannot set free normals."""
    if obj.type != "MESH" or not supported() or has_game_normals(obj):
        return False
    armature = next(
        (m for m in obj.modifiers if m.type == "ARMATURE" and m.object), None
    )
    if armature is None:
        return False

    probes = obj.modifiers.new(PROBES_NAME, "NODES")
    probes.node_group = _probes_tree()  # type: ignore
    normals = obj.modifiers.new(NORMALS_NAME, "NODES")
    normals.node_group = _normals_tree()  # type: ignore
    # In edit mode the probes would show as loose vertices the cage cannot
    # map, so both stay out of it together.
    for modifier in (probes, normals):
        modifier.show_in_editmode = False
        modifier.show_on_cage = False

    _move(obj, probes, list(obj.modifiers).index(armature))
    _move(obj, normals, list(obj.modifiers).index(armature) + 1)
    return True


def add_to(objects) -> int:
    return sum(add(obj) for obj in objects)


def evaluated_normals(obj: bpy.types.Object, depsgraph) -> list | None:
    """The mesh's corner normals as the game's normals leave them, in its
    own space - to keep through a bake that applies the armature."""
    if not has_game_normals(obj):
        return None
    mesh = obj.evaluated_get(depsgraph).to_mesh()
    try:
        if len(mesh.loops) != len(obj.data.loops):  # type: ignore
            return None
        return [c.vector.copy() for c in mesh.corner_normals]
    finally:
        obj.evaluated_get(depsgraph).to_mesh_clear()
