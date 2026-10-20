"""Name-mapped AQM sources for the captured native local-pose kernels.

This adapter uses native_motion_math's verified sampler/frame wrapper.
Channel key_type selects position/rotation/scale; data_type independently
selects the native payload codec. Packed timings retain their flag bits.
There is no scale preparation, axis conversion, body/FNP composition,
rest-pose subtraction, or inferred model/root-name substitution here.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from . import native_motion_math


@dataclass(frozen=True)
class NativePoseSample:
    """Sampled native TRS plus the explicit source-to-target association."""

    trs: np.ndarray
    node_names: tuple[str, ...]
    source_indices: tuple[int | None, ...]
    source_names: tuple[str | None, ...]
    evaluated_frame: np.float32
    skipped_metadata: tuple[str, ...]


def _field(record, name):
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def sample_motion_pose(
    motion,
    node_names,
    default_trs,
    frame,
    *,
    source_name_map=None,
    frame_mode=None,
    frame_start=None,
    frame_end=None,
):
    """Sample an AqmMotion/dataclass or equivalent exported mapping.

    ``node_names`` is the explicit native NODE order. ``default_trs`` is a
    caller-supplied (N,3,4) bind or identity pose: unmatched targets and
    absent channels retain it. ``source_name_map`` explicitly maps source
    names to target names, including any model-specific root alias.
    Duplicate names/channels, unknown node types, and unmatched standard
    source nodes raise ValueError; node_type16 metadata is reported/skipped.

    Input frame is a native frame, not seconds. No loop/clamp is inferred
    from AQM headers. Optional frame_mode requires explicit start/end and
    delegates to the captured native wrapper. Its -1 sentinel is rejected
    rather than inventing a pose for an inactive/ambiguous wrapper result.
    """
    names = tuple(node_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Native node names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate target node names make motion mapping ambiguous")
    defaults = np.asarray(default_trs, dtype=np.float32)
    if defaults.shape != (len(names), 3, 4) or not np.isfinite(defaults).all():
        raise ValueError("default_trs must be a finite (node_count,3,4) native pose")
    evaluated = np.float32(frame)
    if not np.isfinite(evaluated):
        raise ValueError("Motion frame must be finite in float32")
    if frame_mode is None:
        if frame_start is not None or frame_end is not None:
            raise ValueError("Explicit frame bounds require frame_mode")
    else:
        if frame_mode not in (1, 2, 4, 8) or frame_start is None or frame_end is None:
            raise ValueError(
                "Native frame wrapping requires mode1/2/4/8 and explicit bounds"
            )
        evaluated = native_motion_math.wrap_frame(
            evaluated, frame_start, frame_end, frame_mode
        )
        if evaluated == native_motion_math.FRAME_SENTINEL:
            raise ValueError("Native frame wrapper returned its -1 sentinel")

    aliases = {} if source_name_map is None else dict(source_name_map)
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in aliases.items()
    ):
        raise ValueError("source_name_map must map source strings to target strings")
    target_indices = {name: index for index, name in enumerate(names)}
    result = defaults.copy()
    source_indices = [None] * len(names)
    source_names = [None] * len(names)
    seen_sources = set()
    skipped = []
    for source_index, node in enumerate(_field(motion, "nodes")):
        name = _field(node, "name")
        if not isinstance(name, str) or not name:
            raise ValueError("Source node names must be nonempty strings")
        if name in seen_sources:
            raise ValueError(f"Duplicate motion node name: {name}")
        seen_sources.add(name)
        node_type = int(_field(node, "node_type"))
        if node_type == 0x10:
            skipped.append(name)
            continue
        if node_type != 2:
            raise ValueError(f"Unsupported motion node type {node_type:#x}: {name}")
        target_name = aliases.get(name, name)
        if target_name not in target_indices:
            raise ValueError(f"Motion node has no explicit target mapping: {name}")
        target = target_indices[target_name]
        if source_indices[target] is not None:
            raise ValueError(f"Multiple source nodes map to target: {target_name}")
        source_indices[target] = source_index
        source_names[target] = name
        channels = set()
        for key_set in _field(node, "key_sets"):
            channel = int(_field(key_set, "key_type"))
            if channel not in (1, 2, 3):
                raise ValueError(
                    f"Unsupported native pose channel {channel:#x}: {name}"
                )
            if channel in channels:
                raise ValueError(f"Duplicate motion channel {channel}: {name}")
            channels.add(channel)
            keys = _field(key_set, "vec4_keys")
            if not len(keys):
                raise ValueError(f"Native pose channel has no float4 payload: {name}")
            timings = _field(key_set, "timings")
            result[target, channel - 1] = native_motion_math.sample_track(
                keys,
                timings if len(timings) else None,
                evaluated,
                format_byte=int(_field(key_set, "data_type")),
            )
    unused_aliases = aliases.keys() - seen_sources
    if unused_aliases:
        raise ValueError(
            f"Source aliases refer to missing nodes: {sorted(unused_aliases)}"
        )
    return NativePoseSample(
        result,
        names,
        tuple(source_indices),
        tuple(source_names),
        evaluated,
        tuple(skipped),
    )
