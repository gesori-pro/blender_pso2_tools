import re
from enum import IntEnum

import bpy

from . import util


class MeshId(IntEnum):
    Costume = 0
    BreastNeck = 1
    Front = 2
    Ornament1 = 3
    Back = 4
    Shoulder = 5
    Forearm = 6
    Legs = 7
    Ornament2 = 8
    HeadOrnament = 9
    CastBodyOrnament = 10
    CastLegsOrnament = 11
    CastArmsOrnament = 12
    OuterOrnament = 13


MESH_ID_NAMES = {
    MeshId.Costume: "None",
    MeshId.BreastNeck: "Breast & Neck",
    MeshId.Front: "Front",
    MeshId.Ornament1: "Basewear Ornament 1",
    MeshId.Back: "Back",
    MeshId.Shoulder: "Shoulder",
    MeshId.Forearm: "Arms",
    MeshId.Legs: "Legs",
    MeshId.Ornament2: "Basewear Ornament 2",
    MeshId.HeadOrnament: "Head Ornament",
    MeshId.CastBodyOrnament: "Cast Body Ornament",
    MeshId.CastLegsOrnament: "Cast Legs Ornament",
    MeshId.CastArmsOrnament: "Cast Arms Ornament",
    MeshId.OuterOrnament: "Outerwear Ornament",
}


# GetMeshName now appends a face-group ID. Ornament IDs remain the second
# # field in both old and new names. AML's reverse parser is private, so
# this Blender UI adapter still needs to identify the editable field.
MESH_ID_RE = re.compile(r"mesh\[\d+\]_[^#]+#-?\d+#(?P<dummy>-?\d+)(?:#\d+)?(?:_mesh)?$")


def get_mesh_id(name: str) -> MeshId | None:
    if m := MESH_ID_RE.search(util.remove_blender_suffix(name)):
        try:
            return MeshId(int(m.group("dummy")))
        except ValueError:
            return None

    return None


def set_mesh_id(obj: bpy.types.Object, mesh_id: MeshId):
    match = MESH_ID_RE.search(util.remove_blender_suffix(obj.name))
    if match is None:
        return
    start, end = match.span("dummy")
    new_obj_name = obj.name[:start] + str(int(mesh_id)) + obj.name[end:]

    obj.name = new_obj_name
    if obj.data:
        obj.data.name = util.remove_blender_suffix(new_obj_name) + "_mesh"
