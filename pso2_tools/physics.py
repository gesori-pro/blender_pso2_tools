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

import base64
from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper

from . import classes, datafile, fltd
from .debug import debug_print
from .util import OperatorResult

# Armature custom property: the names of every bone a .fltd swings, and
# the chain each one belongs to. {bone name: chain name}
PHYSICS_BONES_PROP = "pso2_physics_bones"

# Bone group / collection name the driven bones are put in, so they are
# visible in the outliner without opening a panel.
PHYSICS_COLLECTION = "PSO2 Physics"

# The file's own bytes, kept so an edited copy can be written back.
PHYSICS_SOURCE_PROP = "pso2_physics_source"


def read_chains(data: bytes) -> dict[str, list[str]]:
    """Chain root -> the bone names that chain drives.

    Returns an empty dict when the file cannot be parsed; a costume with
    no cloth is normal and must not fail the import.
    """
    try:
        parsed = fltd.chains(data)
    except Exception as ex:  # any parse failure is non-fatal
        debug_print("fltd: could not parse:", ex)
        return {}

    return {
        chain["name"]: _chain_stems(chain)
        for chain in parsed
        if chain["name"] and chain["name"] != "None"
    }


def _chain_stems(chain) -> list[str]:
    """The name stems a chain covers, one per strand.

    Every strand root is named in the file - drs2_tube_skirt_000_00 through
    _110_00 for a twelve-strand skirt. Only the root is named, though: its
    links (_000_10, _000_20 ...) are found on the armature by prefix, so the
    strand's trailing link number is dropped to get the stem. A name that
    does not end in a number is left whole rather than cut at its last
    underscore, which would leave a stem short enough to match anything.
    """
    stems = []
    for strand in chain.get("strands") or [chain["name"]]:
        head, sep, tail = strand.rpartition("_")
        stem = head if sep and tail.isdigit() and head else strand
        if stem not in stems:
            stems.append(stem)
    return stems


def mark_armature(
    armature: bpy.types.Object, chains: dict[str, list[str]]
) -> dict[str, str]:
    """Record which bones the game swings, and collect them for the user.

    Matching is by prefix on each strand's stem: the file names every
    strand root, and the armature holds that strand's links under it.
    """
    if not chains:
        return {}

    prefixes = {}
    for root, stems in chains.items():
        for stem in stems:
            prefixes.setdefault(stem, root)

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


# ---------------------------------------------------------------------------
# Editing


def scale_chain(raw: bytearray, chain_name: str, factor: float) -> int:
    """Multiply every non-zero parameter of one chain. Returns values changed.

    The sixteen floats per sub-node are the chain's simulation settings.
    What each one means is not documented anywhere and has not been pinned
    down, so this scales them together rather than pretending to label
    them: larger values swing further on the outfits tested. Zeros are left
    alone, since a zero is a switch rather than an amount.
    """
    changed = 0
    for chain in fltd.chains(bytes(raw)):
        if chain["name"] != chain_name:
            continue
        for sub in chain["subs"]:
            values = sub["floats"]
            scaled = [v * factor if abs(v) > 1e-9 else v for v in values]
            if scaled != values:
                fltd.write_floats(raw, sub["offset"], scaled)
                changed += sum(1 for a, b in zip(values, scaled, strict=True) if a != b)
    return changed


@classes.register
class PSO2_OT_ExportPhysics(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ExportHelper
):
    """Write the outfit's cloth physics out, optionally scaled"""

    bl_label = "Export Cloth Physics"
    bl_idname = "pso2.export_physics"
    bl_options = {"PRESET"}

    filename_ext = ".fltd"
    filter_glob: bpy.props.StringProperty(default="*.fltd", options={"HIDDEN"})

    factor: bpy.props.FloatProperty(
        name="Swing",
        description=(
            "Multiply every cloth setting by this. Above 1 swings further,"
            " below 1 stiffens. 1.0 writes the file back unchanged"
        ),
        default=1.0,
        min=0.1,
        max=5.0,
    )

    def execute(self, context) -> OperatorResult:
        armature = _find_armature(context)
        source = get_source(armature)
        if not source:
            self.report(
                {"ERROR"},
                "No cloth physics on this armature. Import a model that has"
                " a .fltd first.",
            )
            return {"CANCELLED"}

        raw = bytearray(source)
        chains = fltd.chains(bytes(raw))
        changed = 0
        if abs(self.factor - 1.0) > 1e-6:
            for chain in chains:
                changed += scale_chain(raw, chain["name"], self.factor)

        path = Path(self.filepath)  # type: ignore
        path.write_bytes(bytes(raw))

        self.report(
            {"INFO"},
            f"Wrote {path.name}: {len(chains)} chains, {len(raw):,} bytes"
            + (f", {changed} settings scaled by {self.factor:g}" if changed else ""),
        )
        return {"FINISHED"}


def _find_armature(context):
    from . import import_aqm

    return import_aqm._find_target_armature(context)


def store_source(armature: bpy.types.Object, data: bytes) -> None:
    """Keep the file's bytes so it can be written back out with edits.

    Base64 rather than a byte list: a custom property holding a list of
    ints comes back four times the size and no longer parses.
    """
    armature[PHYSICS_SOURCE_PROP] = base64.b64encode(bytes(data)).decode("ascii")


def get_source(armature: bpy.types.Object | None) -> bytes:
    if armature is None:
        return b""
    stored = armature.get(PHYSICS_SOURCE_PROP)
    if not stored:
        return b""
    try:
        return base64.b64decode(str(stored))
    except (ValueError, TypeError):
        return b""
