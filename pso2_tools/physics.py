"""
Read an outfit's cloth physics definition (.fltd) and mark the bones it
drives.

The game swings these bones at runtime; Blender does not simulate them,
so they sit at their bind position here and somewhere else in game. That
is not an error to fix - it is a part of the model no static scene can
show. What the add-on can do is say which bones those are, so a pose is
not built against a skirt or a frill that will not be there.

Measured on pl_rbd_205990_bw.fltd: four chains, driving the neck string,
the tube skirt and the left and right frills. Every bone the file names
after the chain root is a collider, not a driven bone - c_breast and hip
appear there because cloth hits them, not because they swing.
"""

from __future__ import annotations

import bpy

from . import datafile
from .debug import debug_print

# Armature custom property: the names of every bone a .fltd swings, and
# the chain each one belongs to. {bone name: chain name}
PHYSICS_BONES_PROP = "pso2_physics_bones"

# Bone group / collection name the driven bones are put in, so they are
# visible in the outliner without opening a panel.
PHYSICS_COLLECTION = "PSO2 Physics"


def read_chains(data: bytes) -> dict[str, list[str]]:
    """Chain root -> the bone names that chain drives.

    Returns an empty dict when the file cannot be parsed; a costume with
    no cloth is normal and must not fail the import.
    """
    try:
        from System import Array, Byte  # type: ignore

        from AquaModelLibrary.Data.PSO2.Aqua import FLTDPhysics  # type: ignore

        physics = FLTDPhysics(Array[Byte](data))
    except Exception as ex:  # any parse failure is non-fatal
        debug_print("fltd: could not parse:", ex)
        return {}

    chains: dict[str, list[str]] = {}
    for index in range(physics.mainNodes.Count):
        node = physics.mainNodes[index]
        name = str(node.name) if node.name else ""
        if not name or name == "None":
            continue
        chains[name] = _chain_bones(name)

    return chains


def _chain_bones(root: str) -> list[str]:
    """The bones a chain covers, from its root's naming.

    A chain root is the first link of a numbered strip - drs2_tube_skirt_000_00
    heads drs2_tube_skirt_000_10, _020_00 and the rest. The file only names
    the root, so the members are found on the armature by prefix instead.
    """
    head = root.rsplit("_", 2)[0]
    return [head] if head else []


def mark_armature(
    armature: bpy.types.Object, chains: dict[str, list[str]]
) -> dict[str, str]:
    """Record which bones the game swings, and collect them for the user.

    Matching is by prefix: the file names one root per chain, and the
    armature holds the whole strip under the same stem.
    """
    if not chains:
        return {}

    prefixes = {}
    for root in chains:
        head = root.rsplit("_", 2)[0]
        if head:
            prefixes[head] = root

    found: dict[str, str] = {}
    for bone in armature.data.bones:  # type: ignore[union-attr]
        stem = bone.name.split("#")[0]
        for head, root in prefixes.items():
            if stem.startswith(head):
                found[bone.name] = root
                break

    if not found:
        return {}

    armature[PHYSICS_BONES_PROP] = found

    bones = armature.data.bones  # type: ignore[union-attr]
    collections = getattr(armature.data, "collections", None)  # type: ignore[union-attr]
    if collections is not None:
        existing = collections.get(PHYSICS_COLLECTION)
        if existing is not None:
            collections.remove(existing)
        collection = collections.new(PHYSICS_COLLECTION)
        for name in found:
            bone = bones.get(name)
            if bone is not None:
                collection.assign(bone)

    return found


def get_physics_bones(armature: bpy.types.Object | None) -> dict[str, str]:
    """What a previous import recorded, bone name -> chain root."""
    if armature is None:
        return {}
    stored = armature.get(PHYSICS_BONES_PROP)
    if not stored:
        return {}
    return {str(k): str(v) for k, v in dict(stored).items()}


def collect_physics_files(sources) -> list[datafile.DataFile]:
    files: list[datafile.DataFile] = []
    for source in sources:
        files.extend(source.glob("*.fltd"))
    return files
