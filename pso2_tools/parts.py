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


MESH_ID_RE = re.compile(r"mesh\[\d+\]_.*#.*#(\d+)$")
MESH_ID_SUB_RE = re.compile(r"(?<=#)\d+(?=(?:\.\d+)?$)")


def get_mesh_id(name: str) -> MeshId | None:
    if m := MESH_ID_RE.search(util.remove_blender_suffix(name)):
        return MeshId(int(m.group(1)))

    return None


def set_mesh_id(obj: bpy.types.Object, mesh_id: MeshId):
    new_obj_name = MESH_ID_SUB_RE.sub(str(mesh_id), obj.name)
    new_mesh_name = MESH_ID_SUB_RE.sub(f"{mesh_id}_mesh", obj.name)

    obj.name = new_obj_name
    if obj.data:
        obj.data.name = new_mesh_name
