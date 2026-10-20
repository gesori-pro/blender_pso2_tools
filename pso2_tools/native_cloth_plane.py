"""Observed variant-tag 5 collider, live RVA 28D9760.

Its code corrects native Y against a transformed origin plus particle radius,
then applies the native tangential correction. This does not implement tag 0
or 7, nor infer where animated collider transforms come from.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_stages import _particles
from .native_cloth_writeback import F, SMALL, _array, _dot, _small


@dataclass(frozen=True)
class PlaneCollider5:
    transform: np.ndarray
    origin: np.ndarray
    flags: int
    filter_flags: int
    friction_sleeping: float
    friction_moving: float
    particle_filter: tuple = ()


def _transform(point, matrix, *, z_first):
    x, y, z = point[:3]
    first = z * matrix[2] + x * matrix[0] if z_first else x * matrix[0] + z * matrix[2]
    # Native 28D983F/43/49 and 28D994D/51/55 form two pairs. Flattening the
    # decompiler expression changes rounding, even without FMA instructions.
    return first + (y * matrix[1] + matrix[3])


def apply_plane(positions, history, inverse_mass, radii, friction0, friction1,
                collider, sleep_mask, exclusion_mask, *, seeds):
    """Return adjusted native float4 positions. Temporal/mask inputs explicit."""
    p = _particles(positions)
    n = len(p)
    old = _array(history, p.shape, 'Collider history')
    mass = _array(inverse_mass, (n,), 'Collider inverse mass')
    radius = _array(radii, (n,), 'Particle radius')
    f0 = _array(friction0, (n,), 'Collision weights 0')
    f1 = _array(friction1, (n,), 'Collision weights 1')
    matrix = _array(collider.transform, (4, 4), 'Collider transform')
    origin = _array(collider.origin, (4,), 'Collider origin')
    coefficients = np.asarray([collider.friction_sleeping, collider.friction_moving], dtype=np.float32)
    if not np.isfinite(coefficients).all():
        raise ValueError('Collider friction coefficients must be finite')
    for mask in (sleep_mask, exclusion_mask):
        if not isinstance(mask, (int, np.integer)) or not 0 <= mask < 1 << 64:
            raise ValueError('Collider masks must be unsigned 64-bit integers')
    ids = tuple(collider.particle_filter) or tuple(range(n))
    if any(not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in ids):
        raise ValueError('Collider filter must index existing particles')
    first = np.eye(4, dtype=np.float32) if collider.flags & 2 else matrix
    plane = _transform(origin, first, z_first=False)
    second = matrix if collider.flags & 2 else np.eye(4, dtype=np.float32)
    for i in ids:
        if collider.filter_flags & 0xFFFFFFFE and exclusion_mask >> (i & 63) & 1:
            continue
        if mass[i] <= 0:
            continue
        plane_here = _transform(plane, second, z_first=True)
        before_y = p[i, 1]
        depth = F(F(before_y - plane_here[1]) - radius[i])
        if depth >= 0:
            continue
        correction = np.zeros(4, dtype=np.float32)
        correction[1] = depth
        new = p[i] - correction
        p[i] = new
        coefficient = (F(f1[i] * coefficients[0]) if sleep_mask >> (i & 63) & 1
                       else F(f0[i] * coefficients[1]))
        limit = F(coefficient * depth)
        limit2 = F(limit * limit)
        if limit2 < SMALL:
            continue
        reference = old[i].copy()
        vertical_gap = F(reference[1] - before_y)
        if vertical_gap >= SMALL:
            fraction = F(F(seeds.rcp(vertical_gap)) * F(vertical_gap + depth))
            reference = (new - reference) * fraction + reference
        tangent = reference - new
        tangent[1] = 0
        if _small(tangent):
            continue
        length2 = _dot(tangent, tangent)
        minimum = min(length2, limit2)
        scale = F(F(seeds.rsqrt(F(minimum * length2))) * minimum)
        p[i] = tangent * scale + new
    if not np.isfinite(p).all():
        raise ValueError('Native plane collision produced nonfinite coordinates')
    return p
