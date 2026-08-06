"""
PSO2 AQM motion file support.

Reading and writing NIFL/VTBF bytes is delegated to AquaModelLibrary's
AquaMotion class (PSO2-Aqua-Library/AquaModelLibrary.Data/PSO2/Aqua/
AquaMotion.cs) via pythonnet, rather than reimplementing the format here -
that's the same code the rest of this add-on already loads for models, so
there's one parser to keep in sync with the game instead of two.

Everything below the .NET boundary (the AqmMotion/AqmNode/AqmKeySet
dataclasses, shape-adjust detection, PSO2's non-inherited-scale
conversion) is plain Python: none of it exists in AquaModelLibrary, which
only reads and writes the byte format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

KEY_TYPE_POSITION = 0x1
KEY_TYPE_ROTATION = 0x2
KEY_TYPE_SCALE = 0x3

VARIANT_STD_ANIM = 0x10002
VARIANT_PLAYER_ANIM = 0x10012
VARIANT_CAMERA_ANIM = 0x10004
VARIANT_MATERIAL_ANIM = 0x20

# The game's own name for a shape adjust; nothing else uses the suffix.
SHAPE_ADJUST_SUFFIX = "_sa.aqm"

NODE_TYPE_STANDARD = 0x2
NODE_TYPE_NODE_TREE_FLAG = 0x10
NODE_TYPE_MATERIAL = 0x20


class AqmError(Exception):
    """Raised when a motion file cannot be parsed."""


@dataclass
class AqmKeySet:
    """MKEY: one channel (position/rotation/scale/...) of one node."""

    key_type: int
    data_type: int
    unk_int0: int
    # Raw stored timings, including the low flag bits (0x1 = first key,
    # 0x2 = last key). Empty if the channel has a single key.
    timings: list[int] = field(default_factory=list)
    # For data types 1/2/3, a list of (x, y, z, w) tuples.
    vec4_keys: list[tuple[float, float, float, float]] = field(default_factory=list)
    float_keys: list[float] = field(default_factory=list)
    int_keys: list[int] = field(default_factory=list)

    @property
    def time_multiplier(self) -> int:
        return 0x100 if self.data_type & 0x80 else 0x10

    @property
    def key_count(self) -> int:
        return max(len(self.vec4_keys), len(self.float_keys), len(self.int_keys))

    def frames(self) -> list[int]:
        """Timings decoded to frame numbers (flag bits masked off)."""
        if not self.timings:
            return [0] * self.key_count

        mult = self.time_multiplier
        return [t // mult for t in self.timings]

    def is_constant(self) -> bool:
        """Does the channel hold one value, however many keys it stores?

        Pose mods write the same value at every frame they key, so counting
        keys alone says nothing about whether a channel moves.
        """
        for keys in (self.vec4_keys, self.float_keys, self.int_keys):
            if len(keys) < 2:
                continue

            first = keys[0]
            if isinstance(first, tuple):
                if any(
                    abs(a - b) > 1e-6
                    for key in keys[1:]
                    for a, b in zip(first, key, strict=True)
                ):
                    return False
            elif any(abs(first - key) > 1e-6 for key in keys[1:]):
                return False

        return True


@dataclass
class AqmNode:
    """MSEG plus its key sets: the animation data for one node."""

    node_type: int
    node_id: int
    name: str
    key_sets: list[AqmKeySet] = field(default_factory=list)

    def get_key_set(self, key_type: int) -> AqmKeySet | None:
        for key_set in self.key_sets:
            if key_set.key_type == key_type:
                return key_set

        return None


@dataclass
class AqmMotion:
    variant: int
    loop_point: int
    end_frame: int
    frame_speed: float
    node_count: int
    nodes: list[AqmNode] = field(default_factory=list)

    @property
    def is_camera_motion(self) -> bool:
        return self.variant == VARIANT_CAMERA_ANIM

    @property
    def is_material_motion(self) -> bool:
        return self.variant == VARIANT_MATERIAL_ANIM

    @property
    def is_shape_adjust(self) -> bool:
        """Does this look like a shape adjust rather than a motion?

        Shape adjusts share the standard animation variant and cover the
        whole skeleton, so the giveaway is how little of it moves: a handful
        of channels change between the two frames and the rest sit still.

        This is only needed for files that do not say so themselves - see
        is_shape_adjust_file(), which trusts the game's own naming first.
        """
        if self.end_frame > 1 or self.is_camera_motion or self.is_material_motion:
            return False

        # Player animations are motions by definition. Pose mods live here:
        # one frame, every channel a single key.
        if self.variant == VARIANT_PLAYER_ANIM:
            return False

        static = adjusted = 0
        for node in self.nodes:
            for key_set in node.key_sets:
                if not key_set.key_count:
                    continue
                if key_set.is_constant():
                    static += 1
                else:
                    adjusted += 1

        # A shape adjust puts the rest value at frame 0 and the adjusted one
        # at frame 1, so something has to change between the two. A pose mod
        # writes the same value at both frames - on hundreds of channels in
        # some files - which counting keys alone read as an adjustment.
        #
        # What is left moves a handful of channels against a whole skeleton:
        # the files in use move between 0.8% and 7% of theirs, while the two
        # frame emotes that look the same move a fifth at the very least. A
        # tenth sits in that gap. Guessing motion on a borderline file only
        # costs a warning; guessing shape adjust turns a motion away.
        return adjusted > 0 and adjusted * 10 < static + adjusted

    def max_key_frame(self) -> int:
        """Highest frame number found in any node's decoded keyframes.

        MOHeader.end_frame is not always reliable (some mod tools write it
        incorrectly), so callers that need the true animation length should
        use this instead of trusting end_frame on its own.
        """
        return max(
            (
                frame
                for node in self.nodes
                for key_set in node.key_sets
                for frame in key_set.frames()
            ),
            default=0,
        )


# ---- AquaMotion (.NET) <-> AqmMotion (Python) conversion -------------------


def _load_aqua_motion_type():
    from . import dotnet

    dotnet.load()

    from AquaModelLibrary.Data.PSO2.Aqua import AquaMotion

    return AquaMotion


def read_aqm(path: Path | str) -> AqmMotion:
    return parse_aqm(Path(path).read_bytes())


def parse_aqm(data: bytes) -> AqmMotion:
    """Parse NIFL or VTBF motion bytes via AquaMotion."""
    from System import Array, Byte

    AquaMotion = _load_aqua_motion_type()

    try:
        dotnet_motion = AquaMotion(Array[Byte](data))
    # AquaMotion throws a range of CLR exceptions on malformed input.
    except Exception as ex:
        raise AqmError(f"Could not parse AQM file: {ex}") from ex

    if dotnet_motion.moHeader.nodeCount <= 0 or not len(dotnet_motion.motionKeys):
        raise AqmError("Not a motion file, or the file has no animated nodes")

    return _from_dotnet(dotnet_motion)


def write_aqm(path: Path | str, motion: AqmMotion) -> int:
    data = serialize_aqm(motion)
    return Path(path).write_bytes(data)


def serialize_aqm(motion: AqmMotion) -> bytes:
    """Serialize a motion to NIFL AQM bytes via AquaMotion.GetBytesNIFL()."""
    dotnet_motion = _to_dotnet(motion)

    try:
        return bytes(dotnet_motion.GetBytesNIFL())
    # AquaMotion throws a range of CLR exceptions on malformed input.
    except Exception as ex:
        raise AqmError(f"Could not serialize AQM file: {ex}") from ex


def _from_dotnet(dotnet_motion) -> AqmMotion:
    header = dotnet_motion.moHeader

    motion = AqmMotion(
        variant=header.variant,
        loop_point=header.loopPoint,
        end_frame=header.endFrame,
        frame_speed=header.frameSpeed,
        node_count=header.nodeCount,
    )

    for node_index in range(dotnet_motion.motionKeys.Count):
        key_data = dotnet_motion.motionKeys[node_index]
        mseg = key_data.mseg

        node = AqmNode(
            node_type=mseg.nodeType,
            node_id=mseg.nodeId,
            name=str(mseg.nodeName.GetString()),
        )

        for key_index in range(key_data.keyData.Count):
            mkey = key_data.keyData[key_index]

            node.key_sets.append(
                AqmKeySet(
                    key_type=mkey.keyType,
                    data_type=mkey.dataType,
                    unk_int0=mkey.unkInt0,
                    timings=[int(t) for t in mkey.frameTimings],
                    vec4_keys=[(v.X, v.Y, v.Z, v.W) for v in mkey.vector4Keys],
                    float_keys=[float(f) for f in mkey.floatKeys],
                    int_keys=[int(i) for i in mkey.intKeys],
                )
            )

        motion.nodes.append(node)

    return motion


def _to_dotnet(motion: AqmMotion):
    from System import Activator, UInt32
    from System.Numerics import Vector4

    AquaMotion = _load_aqua_motion_type()
    from AquaModelLibrary.Data.DataTypes.SetLengthStrings import PSO2String
    from AquaModelLibrary.Data.PSO2.Aqua.AquaMotionData import (
        KeyData,
        MKEY,
        MOHeader,
        MSEG,
    )

    dotnet_motion = AquaMotion()

    # MOHeader and MSEG are C# structs (value types): the whole field has to
    # be built up on a local copy and reassigned, not mutated through the
    # containing object's property.
    header = Activator.CreateInstance[MOHeader]()
    header.variant = motion.variant
    header.loopPoint = motion.loop_point
    header.endFrame = motion.end_frame
    header.frameSpeed = motion.frame_speed
    header.unkInt0 = 2
    header.nodeCount = len(motion.nodes)
    header.boneTableOffset = 0x50
    header.testString = PSO2String.GeneratePSO2String("test")
    dotnet_motion.moHeader = header

    for node in motion.nodes:
        key_data = KeyData()

        mseg = Activator.CreateInstance[MSEG]()
        mseg.nodeType = node.node_type
        mseg.nodeDataCount = len(node.key_sets)
        mseg.nodeName = PSO2String.GeneratePSO2String(node.name)
        mseg.nodeId = node.node_id
        key_data.mseg = mseg

        for key_set in node.key_sets:
            mkey = MKEY()
            mkey.keyType = key_set.key_type
            mkey.dataType = key_set.data_type
            mkey.unkInt0 = key_set.unk_int0
            mkey.keyCount = key_set.key_count

            for timing in key_set.timings:
                mkey.frameTimings.Add(UInt32(timing))
            for vec in key_set.vec4_keys:
                mkey.vector4Keys.Add(Vector4(*vec))
            for value in key_set.int_keys:
                mkey.intKeys.Add(value)
            for value in key_set.float_keys:
                mkey.floatKeys.Add(value)

            key_data.keyData.Add(mkey)

        dotnet_motion.motionKeys.Add(key_data)

    return dotnet_motion


# ---- Plain-Python motion logic (none of this exists in AquaMotion) --------


def is_shape_adjust_file(path: Path | str, motion: AqmMotion) -> bool:
    """Is this file a shape adjust rather than a motion?

    The game names them <model>_sa.aqm and reads them by that name, so the
    suffix settles it whatever the contents look like - a few carry their
    adjustment as static values, or run past the two frames the game reads.
    Anything else, including shape adjusts renamed by hand, has to be judged
    on its shape.
    """
    if Path(path).name.lower().endswith(SHAPE_ADJUST_SUFFIX):
        return True

    return motion.is_shape_adjust


def make_baked_timings(end_frame: int, multiplier: int = 0x10) -> list[int]:
    """Raw timings for one key per frame from 0 to end_frame.

    The first key is stored as 0x1 and the last has 0x2 added as a flag,
    matching the game's files and the community 3ds Max exporter.
    """
    if end_frame <= 0:
        return []

    timings = [0x1]
    timings += [frame * multiplier for frame in range(1, end_frame)]
    timings.append(end_frame * multiplier + 0x2)
    return timings


def prepare_scaling(motion: AqmMotion, parent_ids: dict[int, int]) -> None:
    """Convert PSO2's non-inherited scale keys to hierarchical scaling.

    PSO2 bones do not inherit scale from their parents, but Blender bones do,
    so each node's scale keys have to be divided by the parent's. That is
    exactly what AquaMotion.PrepareScalingForExport() does, so the work is
    handed to it rather than repeated here.

    It takes the hierarchy as an AquaNode. When a motion is imported onto an
    existing armature there is no .aqn to hand over, so a bare one carrying
    just the parent links is built from the armature's own bones.

    :param parent_ids: Maps a node index to its parent node index.
    """
    # Fast path: nothing to do if no node is actually scaled.
    if not any(
        abs(value - 1.0) > 1e-4
        for node in motion.nodes
        if (scale := node.get_key_set(KEY_TYPE_SCALE))
        for key in scale.vec4_keys
        for value in key[:3]
    ):
        return

    dotnet_motion = _to_dotnet(motion)
    timed = _time_single_scale_keys(dotnet_motion)
    dotnet_motion.PrepareScalingForExport(_hierarchy_node(motion, parent_ids))

    scaled = _from_dotnet(dotnet_motion)
    for index, (node, rescaled) in enumerate(
        zip(motion.nodes, scaled.nodes, strict=True)
    ):
        scale = node.get_key_set(KEY_TYPE_SCALE)
        new_scale = rescaled.get_key_set(KEY_TYPE_SCALE)
        if scale is None or new_scale is None:
            continue

        timings = list(new_scale.timings)
        if index in timed and timings == [0x1]:
            # Nothing was added to the channel, so leave it as the file had
            # it: a single key stores no timings at all.
            timings = []

        # Cancelling a parent that is scaled to zero divides by zero. Blender
        # would carry the resulting nan through the whole armature, so those
        # components go back to neutral: the parent has already collapsed the
        # bone to nothing, so the value under it makes no visible difference.
        # W is unused by scale keys and files leave it at zero.
        scale.timings = timings
        scale.vec4_keys = [
            (
                key[0] if isfinite(key[0]) else 1.0,
                key[1] if isfinite(key[1]) else 1.0,
                key[2] if isfinite(key[2]) else 1.0,
                key[3] if isfinite(key[3]) else 0.0,
            )
            for key in new_scale.vec4_keys
        ]


def _time_single_scale_keys(dotnet_motion) -> set[int]:
    """Give one-key scale channels a timing so the upstream pass sees them.

    PrepareScalingForExport walks a channel's frameTimings to cancel the
    parent, and files store no timings at all for a single key - so a static
    scale under a scaled parent is skipped and keeps the parent's influence.
    A lone timing of 0x1, the same value files use for a first key, puts the
    channel back in range without changing what it holds.

    :return: The nodes that were given one, so it can be taken back off.
    """
    from System import UInt32

    timed = set()
    for index in range(dotnet_motion.motionKeys.Count):
        for mkey in dotnet_motion.motionKeys[index].keyData:
            if (
                mkey.keyType == KEY_TYPE_SCALE
                and mkey.vector4Keys.Count == 1
                and mkey.frameTimings.Count == 0
            ):
                mkey.frameTimings.Add(UInt32(0x1))
                timed.add(index)

    return timed


def _hierarchy_node(motion: AqmMotion, parent_ids: dict[int, int]):
    """An AquaNode holding nothing but the parent links.

    PrepareScalingForExport only reads parentId (and boneShort1, for a mode
    switch that is commented out upstream), so the rest is left at zero.
    """
    from System import Activator

    from AquaModelLibrary.Data.PSO2.Aqua import AquaNode
    from AquaModelLibrary.Data.PSO2.Aqua.AquaNodeData import NODE

    aqn = AquaNode()
    for index in range(len(motion.nodes)):
        # NODE is a C# struct, so it has to be built up and then added.
        node = Activator.CreateInstance[NODE]()
        node.parentId = parent_ids.get(index, -1)
        aqn.nodeList.Add(node)

    return aqn
