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


# The Aqua library names each mesh "mesh[i]_mate_rend_shad_tset" followed by
# "#baseMeshNodeId#baseMeshDummyId", and since it learned face groups, by
# "#faceGroupId" too. The mesh id is the dummy id, the second number, not
# the last one: reading the last would take every mesh's face group (0).
MESH_ID_RE = re.compile(r"^(mesh\[\d+\]_[^#]*#-?\d+#)(-?\d+)((?:#-?\d+)?)$")


def get_mesh_id(name: str) -> MeshId | None:
    if m := MESH_ID_RE.match(util.remove_blender_suffix(name)):
        try:
            return MeshId(int(m.group(2)))
        except ValueError:
            return None

    return None


def set_mesh_id(obj: bpy.types.Object, mesh_id: MeshId):
    name = util.remove_blender_suffix(obj.name)
    m = MESH_ID_RE.match(name)
    if not m:
        return

    suffix = obj.name[len(name) :]
    obj.name = f"{m.group(1)}{int(mesh_id)}{m.group(3)}{suffix}"
    if obj.data:
        obj.data.name = f"{m.group(1)}{int(mesh_id)}{m.group(3)}_mesh{suffix}"
