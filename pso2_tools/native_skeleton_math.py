"""Native PSO2 world-pose evaluation, without Blender or host calls.

Port of captured kernel RVA 0x193AF60 (5098 bytes, no calls), source image
SHA256 2830432bdb089913e2ebc83afc1fe61cef6603de7d5648a062dfae985ed49248.
Inputs are native row matrices and XYZW quaternions. Translation precedes
parent-axis normalization; compensation restores actor/root scale. All
arithmetic retains explicit float32 SSE ordering, including pairwise DPPS.
Live mode also implements 561E70's 02000000 linked-output write. The shared
arithmetic was checked with 199 live NODE records in an isolated oracle.
This evaluates FK only: it does not compose motion/FNP/costume inputs or
run native callbacks, skinning, or physics. No quaternion normalization is
implied. Unobserved node/override bits are rejected explicitly.
"""

import numpy as np

SOURCE_SHA256 = "2830432bdb089913e2ebc83afc1fe61cef6603de7d5648a062dfae985ed49248"
WORLD_POSE_KERNEL_RVA = 0x193AF60
LOCAL_POSE_MERGE_RVA = 0x1F9040

_F = np.float32
_EPSILON = _F(2**-16)
# Low metadata bits 2/4/8/20 are ignored by this kernel. Other listed bits
# select branches directly observed in its instructions.
_NODE_FLAGS = 0x380057FF
_OVERRIDE_FLAGS = 0x2F


def _floats(value, tail, name):
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != len(tail) + 1 or result.shape[1:] != tail:
        raise ValueError(f"{name} must have shape (count, {', '.join(map(str, tail))})")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite float32 values")
    return result


def _length(row):
    # DPPS 7F masks W before pairwise reduction; no BLAS/reassociation.
    p = row[:3] * row[:3]
    return _F(np.sqrt(_F(_F(p[0] + p[1]) + p[2])))


def _compose(left, right):
    result = np.empty((4, 4), dtype=np.float32)
    for axis in range(3):
        row = left[axis]
        result[axis] = (row[2] * right[2] + row[0] * right[0]) + row[1] * right[1]
    row = left[3]
    result[3] = (row[2] * right[2] + row[0] * right[0]) + (row[1] * right[1] + right[3])
    return result


def _rotate(basis, quaternion):
    # Literal vector operations from 193B858..193B912, including where
    # doubling occurs. Algebraically equivalent quaternion formulas can
    # round differently and are intentionally not substituted here.
    q = quaternion
    squared = q * q
    squared = squared + squared
    weighted = q[3] * q
    weighted = weighted + weighted
    cross = q[[2, 0, 1, 3]] * q[[1, 2, 0, 3]]
    cross = cross + cross
    diagonal = np.ones(4, dtype=np.float32) - (squared[[1, 2, 0, 3]] + squared)
    plus = weighted + cross
    minus = cross - weighted
    result = np.empty((3, 4), dtype=np.float32)
    result[0] = (diagonal[1] * basis[0] + plus[2] * basis[1]) + minus[1] * basis[2]
    result[1] = (minus[2] * basis[0] + diagonal[2] * basis[1]) + plus[0] * basis[2]
    result[2] = (plus[1] * basis[0] + minus[0] * basis[1]) + diagonal[0] * basis[2]
    return result


def _multiply_quaternions(left, right):
    ax, ay, az, aw = left
    bx, by, bz, bw = right
    return np.array(
        [
            ((ay * bz - az * by) + bx * aw) + ax * bw,
            ((bx * az - ax * bz) + aw * by) + bw * ay,
            ((ax * by - bx * ay) + aw * bz) + bw * az,
            ((bw * aw - ax * bx) - by * ay) - az * bz,
        ],
        dtype=np.float32,
    )


def merge_local_poses(
    first,
    second,
    *,
    bind_trs=None,
    output_before=None,
    first_flags=1,
    second_flags=1,
):
    """Port the numerical output of native local-pose merge RVA 0x1F9040.

    Arrays use (count,3,4) position/XYZW quaternion/scale. Positions add;
    raw quaternions multiply first times second. At index1 only, scale is
    ``1 + (second_scale - 1) * first_scale``; elsewhere scales multiply.
    All four lanes participate. If first has no entry, an available bind
    entry replaces it, using ordinary scale multiplication even at index1.

    Output count is retained when output_before is supplied, otherwise it
    equals second's count. Unwritten entries require output_before. Native
    header flags bit1 gate evaluation; this numerical API does not mutate
    headers or clear input dirty flags (the native caller does that).
    """
    first = _floats(first, (3, 4), "first")
    second = _floats(second, (3, 4), "second")
    bind = None if bind_trs is None else _floats(bind_trs, (3, 4), "bind_trs")
    dirty = bool(int(first_flags) & 1 or int(second_flags) & 1)
    if output_before is None:
        if not dirty or (
            len(first) < len(second) and (bind is None or len(bind) < len(second))
        ):
            raise ValueError("Unwritten native merge entries require output_before")
        output = np.empty_like(second)
    else:
        output = _floats(output_before, (3, 4), "output_before").copy()
    if not dirty:
        return output
    for index in range(min(len(output), len(second))):
        primary = index < len(first)
        if primary:
            left = first[index]
        elif bind is not None and index < len(bind):
            left = bind[index]
        else:
            continue
        right = second[index]
        output[index, 0] = left[0] + right[0]
        output[index, 1] = _multiply_quaternions(left[1], right[1])
        output[index, 2] = (
            _F(1) + (right[2] - _F(1)) * left[2]
            if primary and index == 1
            else left[2] * right[2]
        )
    return output


def evaluate_world_pose(
    node_flags,
    parent_ids,
    bind_trs,
    animated_trs=None,
    *,
    root_world=None,
    overrides=None,
    links=None,
    world_before=None,
    pass_selector=0,
    node_data_flags=0,
    limit_override=True,
    live_runtime=False,
):
    """Return native world matrices from NODE fields and local TRS arrays.

    ``bind_trs`` and optional ``animated_trs`` have shape (N,3,4), ordered
    position, XYZW quaternion, scale. NODE flags/parents are integer arrays;
    ``links`` is (N,2) signed auxiliary joint indices. ``world_before`` is
    (world_count,4,4), copied so skipped/extra slots retain their old values.
    It is required for nonzero overrides or auxiliary-pass evaluation.
    Root defaults to identity. The optional byte overrides are the actual
    native per-node mask, not a Boolean inclusion mask.

    Pass0 processes flags4000-clear nodes; pass1 processes flags4000-set
    nodes only when node_data_flags contains10. With limit_override=False,
    an animated array also bounds the evaluated count. This is the kernel's
    argument7 behavior, not Blender bone inheritance or an AQM flag field.
    live_runtime=True additionally permits the observed 02000000 alternate
    joint writeback; demo callers retain their previous flag boundary.
    """
    bind = _floats(bind_trs, (3, 4), "bind_trs")
    if not isinstance(live_runtime, bool):
        raise ValueError('live_runtime must explicitly select a Boolean backend')
    count = len(bind)
    flags = np.asarray(node_flags)
    parents = np.asarray(parent_ids)
    if flags.shape != (count,) or flags.dtype.kind not in "iu":
        raise ValueError("node_flags must be one unsigned native flag word per node")
    if parents.shape != (count,) or parents.dtype.kind not in "iu":
        raise ValueError("parent_ids must contain one native integer per node")
    allowed_flags = _NODE_FLAGS | (0x02000000 if live_runtime else 0)
    if any(int(v) < 0 or int(v) > 0xFFFFFFFF or int(v) & ~allowed_flags for v in flags):
        raise ValueError("Unobserved native NODE flag bits")
    if pass_selector not in (0, 1):
        raise ValueError("pass_selector must be 0 or 1")
    animated = (
        None if animated_trs is None else _floats(animated_trs, (3, 4), "animated_trs")
    )
    mask = (
        np.zeros(count, dtype=np.uint8) if overrides is None else np.asarray(overrides)
    )
    if (
        mask.shape != (count,)
        or mask.dtype.kind not in "iu"
        or any(int(v) < 0 or int(v) > 255 or int(v) & ~_OVERRIDE_FLAGS for v in mask)
    ):
        raise ValueError("overrides must contain only captured native byte-mask bits")
    if links is None:
        linked = np.full((count, 2), -1, dtype=np.int32)
    else:
        linked = np.asarray(links)
        if linked.shape != (count, 2) or linked.dtype.kind not in "iu":
            raise ValueError("links must contain two native joint indices per node")
        if np.any(linked < -32768) or np.any(linked > 32767):
            raise ValueError("Native auxiliary links must fit signed int16")
    if world_before is None:
        if np.any(mask) or pass_selector:
            raise ValueError(
                "Preserved world matrices are required for overrides/auxiliary pass"
            )
        world = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    else:
        world = _floats(world_before, (4, 4), "world_before").copy()
    if pass_selector and not int(node_data_flags) & 0x10:
        return world
    if not len(world):
        raise ValueError("Native root evaluation requires at least one world matrix")
    root = (
        np.eye(4, dtype=np.float32)
        if root_world is None
        else np.asarray(root_world, dtype=np.float32)
    )
    if root.shape != (4, 4) or not np.isfinite(root).all():
        raise ValueError("root_world must be a finite native 4x4 row matrix")
    root_scale = np.array([_length(row) for row in root[:3]], dtype=np.float32)
    first_mask = int(mask[0]) if count else 0
    if first_mask & 2 and root_world is None:
        raise ValueError("Native root skip override requires a supplied root matrix")
    if not first_mask & 2:
        normalized = root.copy()
        for axis in range(3):
            if root_scale[axis] != 0 or (root_world is not None and first_mask & 1):
                normalized[axis] = normalized[axis] / root_scale[axis]
        world[0] = (
            _compose(world[0], normalized)
            if root_world is not None and first_mask & 1
            else normalized
        )
    limit = min(count, len(world))
    if animated is not None and not limit_override:
        limit = min(limit, len(animated))
    for index in range(limit):
        flag, override = int(flags[index]), int(mask[index])
        if override & 2 or bool(flag & 0x4000) != bool(pass_selector):
            continue
        before = world[index].copy()
        a, b = map(int, linked[index])
        if flag & 0x08000000 and a >= 0 and b >= 0:
            if a < len(world) and b < len(world):
                world[index] = (world[b] - world[a]) * _F(0.5) + world[a]
                world[index, 3, 3] = _F(1)
            continue
        if index:
            parent = int(parents[index])
            world[index] = (
                world[parent]
                if 0 <= parent < len(world)
                else np.eye(4, dtype=np.float32)
            )
        current = world[index]
        local_exists = animated is not None and index < len(animated)
        if flag & 0x20000000 and a >= 0:
            if a < len(world):
                current[3] = (
                    ((world[b, 3] - world[a, 3]) * _F(0.5) + world[a, 3])
                    if 0 <= b < len(world)
                    else world[a, 3]
                )
                current[3, 3] = _F(1)
        else:
            position = None
            if local_exists:
                if override & 4:
                    position = animated[index, 0]
                elif flag & 0x200:
                    position = bind[index, 0] * animated[index, 2, 0]
                elif flag & 0x1000:
                    position = bind[index, 0]
                else:
                    position = animated[index, 0]
            elif not flag & 1:
                position = bind[index, 0]
            if position is not None:
                current[3] = (position[2] * current[2] + position[0] * current[0]) + (
                    position[1] * current[1] + current[3]
                )
        threshold = _EPSILON if animated is not None else _F(0)
        if animated is not None and flag & 0x10000000 and a >= 0:
            if a < len(world):
                for axis in range(3):
                    length = _length(current[axis])
                    if length > threshold:
                        current[axis] = (
                            world[a, axis] / _length(world[a, axis])
                        ) * length
            continue
        scale = root_scale.copy()
        for axis, bit in enumerate((0x40, 0x80, 0x100)):
            if flag & bit:
                length = _length(current[axis])
                if length > threshold:
                    current[axis] = current[axis] / length
            elif index:
                scale[axis] = _F(1)
        quaternion = (
            animated[index, 1] if local_exists and not flag & 0x400 else bind[index, 1]
        )
        current[:3] = _rotate(current[:3], quaternion)
        if override & 8:
            current[:] = _compose(before, current)
        local_scale = (
            animated[index, 2] if local_exists and not flag & 0x10 else bind[index, 2]
        )
        scale = local_scale[:3] * scale
        for axis in range(3):
            current[axis] = current[axis] * scale[axis]
        if override & 1:
            current[:] = _compose(before, current)
        if override & 0x20:
            current[3] = current[3] + before[3]
            current[3, 3] = _F(1)
        # Live 561E70 writes an alternate joint's completed matrix into its
        # linked output slot, before evaluating following joints. This is
        # observed on thigh/calf/foot *_alt nodes, not a final pose overlay.
        if (live_runtime and animated is not None and flag & 0x02000000
                and not (flag & 0x10000000 and a >= 0) and 0 <= a < len(world)):
            world[a] = current
    return world
