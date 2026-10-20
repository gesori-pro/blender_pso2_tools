"""Root extraction from an already sampled native local TRS pose.

Captured routine RVA 0x18B9880, extraction half 0x18B993E..0x18B9B62;
source image SHA256
2830432bdb089913e2ebc83afc1fe61cef6603de7d5648a062dfae985ed49248.
The preceding sampler calls are outside this API. Flags10/20 select those
samplers, so they do not alter extraction of a supplied pose. No Blender,
axis conversion, quaternion normalization, or root/ground fitting occurs.
"""

import numpy as np

SOURCE_SHA256 = "2830432bdb089913e2ebc83afc1fe61cef6603de7d5648a062dfae985ed49248"
ROOT_EXTRACTION_RVA = 0x18B9880


def _quaternion_rows(quaternion):
    # Literal SSE operation order, applied to the initial identity basis.
    # The native kernel does not normalize or canonicalize its XYZW input.
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
    x, y, z = np.eye(4, dtype=np.float32)[:3]
    rows = np.empty((3, 4), dtype=np.float32)
    rows[0] = (diagonal[1] * x + plus[2] * y) + minus[1] * z
    rows[1] = (diagonal[2] * y + minus[2] * x) + plus[0] * z
    rows[2] = (plus[1] * x + minus[0] * y) + diagonal[0] * z
    return rows


def extract_root_motion(pose, flags):
    """Return (new_pose, extracted_native_row_matrix) without mutating input.

    Pose shape is (N,3,4): position, XYZW quaternion, scale. Bits2/4/8
    extract root X/Y/Z translation and set those pose lanes to zero. Bit1
    extracts raw quaternion rotation and resets the root quaternion to
    (0,0,0,1). Whenever any low-nibble bit is set, all three scale axes are
    applied to the external basis and the full root scale becomes1111.
    With a zero low nibble the pose is unchanged and extraction is identity.

    Flags are a native uint32; higher bits belong to the preceding sampler
    or callers and have no effect within this extraction half. A root entry
    is required. The remaining pose nodes and position W are preserved.
    """
    if (
        not isinstance(flags, (int, np.integer))
        or isinstance(flags, (bool, np.bool_))
        or not 0 <= int(flags) <= 0xFFFFFFFF
    ):
        raise ValueError("Root extraction flags must fit a native uint32")
    data = np.asarray(pose, dtype=np.float32)
    if data.ndim != 3 or data.shape[1:] != (3, 4) or len(data) < 1:
        raise ValueError("Root extraction requires a nonempty (N,3,4) native pose")
    if not np.isfinite(data).all():
        raise ValueError("Root extraction requires finite float32 pose values")
    result = data.copy()
    matrix = np.eye(4, dtype=np.float32)
    bits = int(flags) & 0xF
    if not bits:
        return result, matrix
    if bits & 0xE:
        extracted = np.array((0, 0, 0, 1), dtype=np.float32)
        for axis, bit in enumerate((2, 4, 8)):
            if bits & bit:
                extracted[axis] = result[0, 0, axis]
                result[0, 0, axis] = np.float32(0)
        matrix[3] = (extracted[2] * matrix[2] + extracted[0] * matrix[0]) + (
            extracted[1] * matrix[1] + matrix[3]
        )
    if bits & 1:
        matrix[:3] = _quaternion_rows(result[0, 1])
        result[0, 1] = (0, 0, 0, 1)
    for axis in range(3):
        matrix[axis] = matrix[axis] * result[0, 2, axis]
    # Native global 0x7C22A60 is four ones, including the scale W lane.
    result[0, 2] = np.float32(1)
    return result, matrix
