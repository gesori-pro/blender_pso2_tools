"""Collider matrix preparation recovered from live RVA 28CCCF0.

The manager copies a selected bone matrix through this preparation before
collision. In particular the tie's .1 scale blend changes its basis. Source
buffer selection and name-to-index binding remain independent requirements.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_writeback import F, SMALL, _array, _cross, _dot


SCALE_BLEND_MAX = F(0.9900152683258057)


@dataclass(frozen=True)
class PreparedTransform:
    matrix: np.ndarray
    flags: int
    movement_distance: object


def prepare_transform(previous, incoming, flags, blend0, blend1, *, seeds):
    """Prepare one slot, preserving the native once-per-shape distance flag.

    movement_distance is None when bit 0x1000 was already set: that scratch
    slot is not written by this call. It is not assigned a made-up zero.
    """
    old = _array(previous, (4, 4), 'Previous collider matrix')
    out = _array(incoming, (4, 4), 'Incoming collider matrix').copy()
    coefficients = _array([blend0, blend1], (2,), 'Collider scale blend')
    if not isinstance(flags, (int, np.integer)) or not 0 <= flags <= 65535:
        raise ValueError('Collider flags must be an unsigned short')
    displacement = old[3] - out[3]
    norm = _dot(displacement, displacement)
    distance = F(F(seeds.rsqrt(norm))*norm) if norm > 0 else F(0)
    written = None if flags & 0x1000 else distance
    flags = int(flags) | 0x1000
    alpha = min(F(1), max(F(0), F(coefficients[0]*coefficients[1])))
    if flags & 2:
        if flags & 0x380:
            inverse_lengths = np.array([F(seeds.rsqrt(_dot(v, v))) for v in out[:3]], dtype=np.float32)
            selected = [i for i, bit in enumerate((0x80, 0x100, 0x200)) if flags & bit]
            summed = F(0)
            for i in selected:
                summed = F(summed + inverse_lengths[i])
            common = summed
            if abs(summed) >= SMALL:
                common = F(F(F(1)/summed) * F(len(selected)))
            for i in range(3):
                out[i] = out[i] * F(common * inverse_lengths[i])
        if SMALL <= alpha <= SCALE_BLEND_MAX:
            if abs(_dot(out[0], _cross(out[2], out[1]))) >= SMALL:
                residual = F(F(1)-alpha)
                for i in range(3):
                    inv = F(seeds.rsqrt(_dot(out[i], out[i])))
                    out[i] = out[i] * F(F(residual*inv) + alpha)
    if not np.isfinite(out).all():
        raise ValueError('Collider preparation produced a degenerate non-finite basis')
    return PreparedTransform(out, flags, written)
