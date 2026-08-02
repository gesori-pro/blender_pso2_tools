"""
Parser and writer for PSO2 AQM motion files (NIFL format).

Based on AquaModelLibrary's AquaMotion reader/writer
(PSO2-Aqua-Library/AquaModelLibrary.Data/PSO2/Aqua/AquaMotion.cs) and the
PSO2AQM_IO.ms 3ds Max script from Phantasy-Star-Online-2-Model-Tools.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGIC_NIFL = b"NIFL"
MAGIC_VTBF = b"VTBF"

KEY_TYPE_POSITION = 0x1
KEY_TYPE_ROTATION = 0x2
KEY_TYPE_SCALE = 0x3

VARIANT_STD_ANIM = 0x10002
VARIANT_PLAYER_ANIM = 0x10012
VARIANT_CAMERA_ANIM = 0x10004
VARIANT_MATERIAL_ANIM = 0x20

NODE_TYPE_STANDARD = 0x2
NODE_TYPE_NODE_TREE_FLAG = 0x10
NODE_TYPE_MATERIAL = 0x20

_UINT_TIMING_FLAG = 0x80


class AqmError(Exception):
    """Raised when a motion file cannot be parsed."""


def _decode_pso2_string(data: bytes) -> str:
    data = data.split(b"\0", 1)[0]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp932", errors="replace")


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
        return 0x100 if self.data_type & _UINT_TIMING_FLAG else 0x10

    @property
    def key_count(self) -> int:
        return max(len(self.vec4_keys), len(self.float_keys), len(self.int_keys))

    def frames(self) -> list[int]:
        """Timings decoded to frame numbers (flag bits masked off)."""
        if not self.timings:
            return [0] * self.key_count

        mult = self.time_multiplier
        return [t // mult for t in self.timings]


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


def read_aqm(path: Path | str) -> AqmMotion:
    return parse_aqm(Path(path).read_bytes())


def parse_aqm(data: bytes) -> AqmMotion:
    base = _skip_ice_envelope(data)
    magic = data[base : base + 4]

    if magic == MAGIC_VTBF:
        raise AqmError(
            "VTBF format motions are not supported."
            " Convert the file to NIFL format with Aqua Model Tool first."
        )

    if magic != MAGIC_NIFL:
        raise AqmError(f"Not an AQM file (unexpected magic {magic!r})")

    return _parse_nifl(data, base)


def _skip_ice_envelope(data: bytes) -> int:
    """Return the offset where NIFL/VTBF data starts.

    Files extracted from ICE archives may retain an AFP envelope header whose
    size is stored at offset 0xC.
    """
    if len(data) < 0x20:
        raise AqmError("File is too small to be an AQM file")

    if data[:4] in (MAGIC_NIFL, MAGIC_VTBF):
        return 0

    (header_size,) = struct.unpack_from("<i", data, 0xC)
    if 0x10 <= header_size < len(data) - 4:
        if data[header_size : header_size + 4] in (MAGIC_NIFL, MAGIC_VTBF):
            return header_size

    raise AqmError("Not an AQM file (no NIFL or VTBF header found)")


def _parse_nifl(data: bytes, base: int) -> AqmMotion:
    # NIFL header is 0x20 bytes. All offsets stored in the file are relative
    # to the end of it.
    offset0 = base + 0x20

    rel0_magic, _rel0_size, rel0_data_start, _version = struct.unpack_from(
        "<4siii", data, offset0
    )
    if rel0_magic != b"REL0":
        raise AqmError("Invalid AQM file (missing REL0 header)")

    mo_offset = offset0 + rel0_data_start
    (
        variant,
        loop_point,
        end_frame,
        frame_speed,
        _unk_int0,
        node_count,
        bone_table_offset,
    ) = struct.unpack_from("<iiifiii", data, mo_offset)
    # Followed by a 0x20 byte "test" string and a reserved int.

    motion = AqmMotion(
        variant=variant,
        loop_point=loop_point,
        end_frame=end_frame,
        frame_speed=frame_speed,
        node_count=node_count,
    )

    if node_count <= 0 or node_count > 0x10000:
        raise AqmError(f"Invalid AQM file (node count {node_count})")

    # MSEG table
    mseg_offset = offset0 + bone_table_offset
    node_offsets: list[int] = []

    for i in range(node_count):
        node_type, node_data_count, node_offset = struct.unpack_from(
            "<iii", data, mseg_offset
        )
        name = _decode_pso2_string(data[mseg_offset + 0xC : mseg_offset + 0x2C])
        (node_id,) = struct.unpack_from("<i", data, mseg_offset + 0x2C)
        mseg_offset += 0x30

        motion.nodes.append(AqmNode(node_type=node_type, node_id=node_id, name=name))
        node_offsets.append((node_offset, node_data_count))

    # MKEY data
    for node, (node_offset, node_data_count) in zip(motion.nodes, node_offsets):
        mkey_offset = offset0 + node_offset

        headers = []
        for _ in range(node_data_count):
            headers.append(struct.unpack_from("<6i", data, mkey_offset))
            mkey_offset += 0x18

        for key_type, data_type, unk_int0, key_count, frame_addr, time_addr in headers:
            key_set = AqmKeySet(
                key_type=key_type, data_type=data_type, unk_int0=unk_int0
            )

            if key_count > 1:
                fmt = "<%d%s" % (
                    key_count,
                    "I" if data_type & _UINT_TIMING_FLAG else "H",
                )
                key_set.timings = list(
                    struct.unpack_from(fmt, data, offset0 + time_addr)
                )

            base_type = data_type & ~_UINT_TIMING_FLAG
            offset = offset0 + frame_addr

            if base_type in (0x1, 0x2, 0x3):
                for value in struct.iter_unpack("<4f", data[offset : offset + key_count * 16]):
                    key_set.vec4_keys.append(value)
            elif base_type == 0x5:
                key_set.int_keys = list(
                    struct.unpack_from("<%di" % key_count, data, offset)
                )
            elif base_type in (0x4, 0x6):
                key_set.float_keys = list(
                    struct.unpack_from("<%df" % key_count, data, offset)
                )
            else:
                raise AqmError(
                    f"Unexpected data type 0x{data_type:X}"
                    f" (key type 0x{key_type:X}) in node {node.name}"
                )

            node.key_sets.append(key_set)

    return motion


def write_aqm(path: Path | str, motion: AqmMotion) -> int:
    data = serialize_aqm(motion)
    return Path(path).write_bytes(data)


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


def serialize_aqm(motion: AqmMotion) -> bytes:
    """Serialize a motion to NIFL AQM bytes.

    Port of AquaMotion.GetBytesNIFL(). All offsets are relative to the end
    of the NIFL header (offset 0x20), which equals a position in the REL0
    block built here since the NIFL header is prepended last.
    """
    rel = bytearray()
    pointer_locations: list[int] = []

    def w_i32(*values: int):
        for value in values:
            rel.extend(struct.pack("<i", value))

    def align16():
        rel.extend(b"\0" * (-len(rel) % 0x10))

    def backfill(location: int, value: int):
        rel[location : location + 4] = struct.pack("<i", value)

    # REL0 header
    rel.extend(b"REL0")
    rel0_size_at = len(rel)
    w_i32(0, 0x10, 0)

    # MOHeader
    w_i32(motion.variant, motion.loop_point, motion.end_frame)
    rel.extend(struct.pack("<f", motion.frame_speed))
    w_i32(0x2, len(motion.nodes))
    pointer_locations.append(len(rel))
    w_i32(0x50)  # Bone table offset. The MOHeader always ends at 0x50.
    rel.extend(b"test".ljust(0x20, b"\0"))
    w_i32(0)

    # MSEG table
    node_offset_locations: list[int] = []
    for node in motion.nodes:
        w_i32(node.node_type, len(node.key_sets))
        node_offset_locations.append(len(rel))
        pointer_locations.append(len(rel))
        w_i32(0)  # Node offset, filled in below.
        rel.extend(node.name.encode("utf-8")[:0x1F].ljust(0x20, b"\0"))
        w_i32(node.node_id)

    # Key data
    for node, node_offset_at in zip(motion.nodes, node_offset_locations):
        backfill(node_offset_at, len(rel))

        frame_address_locations: list[int] = []
        time_address_locations: list[int | None] = []

        for key_set in node.key_sets:
            frame_address_locations.append(len(rel) + 0x10)
            pointer_locations.append(len(rel) + 0x10)

            if key_set.key_count > 1:
                time_address_locations.append(len(rel) + 0x14)
                pointer_locations.append(len(rel) + 0x14)
            else:
                time_address_locations.append(None)

            w_i32(
                key_set.key_type,
                key_set.data_type,
                key_set.unk_int0,
                key_set.key_count,
                0,  # Frame address, filled in below.
                0,  # Time address, filled in below.
            )

        align16()

        for key_set, frame_at, time_at in zip(
            node.key_sets, frame_address_locations, time_address_locations
        ):
            if time_at is not None and key_set.timings:
                backfill(time_at, len(rel))
                fmt = "H" if key_set.time_multiplier == 0x10 else "I"
                for timing in key_set.timings:
                    rel.extend(struct.pack("<" + fmt, timing))
                align16()

            backfill(frame_at, len(rel))

            if key_set.vec4_keys:
                for key in key_set.vec4_keys:
                    rel.extend(struct.pack("<4f", *key))
            elif key_set.int_keys:
                for key in key_set.int_keys:
                    rel.extend(struct.pack("<i", key))
            elif key_set.float_keys:
                for key in key_set.float_keys:
                    rel.extend(struct.pack("<f", key))

            align16()

    # REL0 size
    backfill(rel0_size_at, len(rel) - 0x8)

    # NOF0 pointer table
    nof0_offset = len(rel)
    nof0_size = (len(pointer_locations) + 2) * 4
    rel.extend(b"NOF0")
    w_i32(nof0_size, len(pointer_locations), 0)
    for location in pointer_locations:
        w_i32(location)

    nof0_full_size = nof0_size + 0x8 + (-len(rel) % 0x10)
    align16()

    # NEND
    rel.extend(b"NEND")
    w_i32(0x8, 0, 0)

    # NIFL header
    nifl = struct.pack(
        "<4siiiiiii",
        MAGIC_NIFL,
        0x18,
        1,
        0x20,
        nof0_offset,
        nof0_offset + 0x20,
        nof0_full_size,
        0,
    )

    return bytes(nifl + rel)


def prepare_scaling(motion: AqmMotion, parent_ids: dict[int, int]) -> None:
    """Convert PSO2's non-inherited scale keys to hierarchical scaling.

    PSO2 bones do not inherit scale from their parents, but Blender bones do,
    so each node's scale keys are divided by the parent's interpolated scale.
    Ported from AquaMotion.PrepareScalingForExport().

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

    # Add scale keys to each child at its parent's key timings so the parent
    # influence can be cancelled exactly.
    for index, node in enumerate(motion.nodes):
        parent_id = parent_ids.get(index, -1)
        if parent_id < 0 or parent_id >= len(motion.nodes):
            continue

        scale = node.get_key_set(KEY_TYPE_SCALE)
        parent_scale = motion.nodes[parent_id].get_key_set(KEY_TYPE_SCALE)
        if scale is None or parent_scale is None:
            continue

        if len(parent_scale.timings) > 1 and len(scale.timings) < 2:
            scale.timings = [0x1, motion.end_frame * 0x10 + 0x2]
            scale.vec4_keys.append(scale.vec4_keys[0])

        for timing in parent_scale.timings:
            if timing not in scale.timings:
                _insert_vec4_key_at_time(scale, timing)

    # Cancel the parent scale, leaf to root.
    raw_keys = {
        index: list(scale.vec4_keys)
        for index, node in enumerate(motion.nodes)
        if (scale := node.get_key_set(KEY_TYPE_SCALE))
    }

    for index in range(len(motion.nodes) - 1, 0, -1):
        parent_id = parent_ids.get(index, -1)
        if parent_id < 0 or parent_id >= len(motion.nodes):
            continue

        scale = motion.nodes[index].get_key_set(KEY_TYPE_SCALE)
        parent_scale = motion.nodes[parent_id].get_key_set(KEY_TYPE_SCALE)
        if scale is None or parent_scale is None:
            continue

        parent_raw = raw_keys[parent_id]
        for i, timing in enumerate(scale.timings or [0]):
            value = _interpolate_vec4(
                parent_raw, parent_scale.timings, parent_scale.time_multiplier, timing
            )
            key = scale.vec4_keys[i]
            scale.vec4_keys[i] = tuple(
                k / v if v else k for k, v in zip(key, value)
            )  # type: ignore


def _insert_vec4_key_at_time(key_set: AqmKeySet, timing: int) -> None:
    for i, existing in enumerate(key_set.timings):
        if existing > timing:
            value = _interpolate_vec4(
                key_set.vec4_keys, key_set.timings, key_set.time_multiplier, timing
            )
            key_set.timings.insert(i, timing)
            key_set.vec4_keys.insert(i, value)
            return


def _interpolate_vec4(
    keys: list[tuple[float, float, float, float]],
    timings: list[int],
    multiplier: int,
    timing: int,
) -> tuple[float, float, float, float]:
    """Linearly interpolate a key value at a raw stored timing."""
    if not timings or len(keys) == 1:
        return keys[0]

    frame = timing // multiplier
    frames = [t // multiplier for t in timings]

    if frame <= frames[0]:
        return keys[0]
    if frame >= frames[-1]:
        return keys[-1]

    for i in range(len(frames) - 1):
        low, high = frames[i], frames[i + 1]
        if low <= frame <= high:
            if frame == low or high == low:
                return keys[i]
            if frame == high:
                return keys[i + 1]

            ratio = (frame - low) / (high - low)
            a, b = keys[i], keys[i + 1]
            return tuple(x + (y - x) * ratio for x, y in zip(a, b))  # type: ignore

    return keys[-1]
