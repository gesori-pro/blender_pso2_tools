"""Generate a native local body layer from FNP fields and source motions.

The bundled proportion tables have finite extraction precision. This is a
source-derived calculation, not a claim of bit-exact native FNP generation.
No captured local/final character pose enters this function. The caller
supplies an asset-derived uniform root multiplier explicitly.
"""

import numpy as np

from . import native_pose_sources, proportions


def _multiply(left, right):
    x, y, z, w = left
    a, b, c, d = right
    return np.asarray(
        [
            w * a + x * d + y * c - z * b,
            w * b - x * c + y * d + z * a,
            w * c + x * b - y * a + z * d,
            w * d - x * a - y * b - z * c,
        ],
        dtype=np.float32,
    )


def compose_body_pose(
    node_names,
    neutral_trs,
    character_fields,
    shape_motion,
    *,
    root_multiplier,
    shape_name_map=None,
):
    """Return parent-local deltas in native position/XYZW/scale order.

    FNP rotations are parent-space left deltas. Native pose merging uses
    local right deltas, hence inv(neutralQ)*FNPdelta*neutralQ, followed by SA.
    SA frame1 is an adjustment pose, not an absolute neutral pose. CMX's
    legLength is multiplied into body_root last, as native callback8FE160.
    A model's selected metadata must be supplied by its actual asset identity.
    """
    names = tuple(node_names)
    if len(set(names)) != len(names):
        raise ValueError("Body node names must be unique")
    neutral = np.asarray(neutral_trs, dtype=np.float32)
    if neutral.shape != (len(names), 3, 4) or not np.isfinite(neutral).all():
        raise ValueError("Neutral body pose must match named native nodes")
    factor = np.float32(root_multiplier)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("CMX body multiplier must be finite and positive")
    body = proportions.compute(character_fields, apply_outfit_adjust=False)["bones"]
    result = np.zeros_like(neutral)
    result[:, 1, 3] = 1
    result[:, 2, :3] = 1
    shape = native_pose_sources.sample_motion_pose(
        shape_motion, names, result, 1, source_name_map=shape_name_map
    ).trs
    for i, name in enumerate(names):
        delta = body.get(name)
        if delta:
            result[i, 0, :3] = delta["pos"]
            result[i, 2, :3] = delta["scale"]
            q = neutral[i, 1]
            squared = np.sum(q * q, dtype=np.float32)
            if squared <= 0:
                raise ValueError(f"Neutral quaternion is not invertible: {name}")
            inverse = q * np.array([-1, -1, -1, 1], dtype=np.float32) / squared
            result[i, 1] = _multiply(
                _multiply(inverse, np.asarray(delta["rotQuat"], dtype=np.float32)), q
            )
        result[i, 0] += shape[i, 0]
        result[i, 1] = _multiply(result[i, 1], shape[i, 1])
        result[i, 2, :3] *= shape[i, 2, :3]
    if "body_root" not in names:
        raise ValueError("Body metadata requires a body_root node")
    result[names.index("body_root"), 2, :3] *= factor
    return result
