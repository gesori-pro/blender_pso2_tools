"""Tag-2 collider (FLTD file kind 3): a capsule on one bone.

Built by 28BBE60 (case 3): origin +0xB0 and axis +0xC0 (file values 0..2 and
3..5, each with the option 0x20/0x40/0x80 sign flips) and radius +0xD0 (file
value 6, not clamped). Point response RVA 28D6DA0 (flag 4 clear).
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_plane import _transform
from .native_cloth_stages import _particles
from .native_cloth_writeback import F, SMALL, _array, _dot


@dataclass(frozen=True)
class PointSegmentCollider2:
    transform: np.ndarray
    inverse_transform: np.ndarray
    origin: np.ndarray
    axis: np.ndarray
    radius: float
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def _frame(collider):
    matrix = _array(collider.transform, (4, 4), 'Segment world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Segment inverse transform')
    identity = np.eye(4, dtype=np.float32)
    local = bool(collider.flags & 2)
    frame = identity if local else matrix
    origin = _transform(_array(collider.origin, (4,), 'Segment origin'), frame, z_first=True)
    a = _array(collider.axis, (4,), 'Segment axis')
    axis = (a[2]*frame[2] + a[0]*frame[0]) + a[1]*frame[1]
    return origin, axis, (inverse if local else identity), (matrix if local else identity)


def apply_point_segment(positions, inverse_mass, radii, collider, exclusion_mask, *, seeds):
    """RVA 28D6DA0: push movable particles out of the one-bone capsule.

    Per particle the shape's radius lane holds r + radius[i] (no clamp; the
    original r is put back at the end). The axis parameter is clamped to
    [0, 1], so the ends are hemispherical. The filter list (+0x18) selects and
    orders particles; local mode (flag 2) works in the bone frame (+0x60 in,
    +0x48 out), world mode against the transformed origin and axis.
    """
    p = _particles(positions)
    n = len(p)
    if collider.flags & 4:
        raise ValueError('Edge one-bone capsule (28D51F0) is not recovered')
    mass = _array(inverse_mass, (n,), 'Segment inverse mass')
    radius = _array(radii, (n,), 'Particle radii')
    r0 = F(collider.radius)
    if not np.isfinite(r0):
        raise ValueError('Segment radius must be finite')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Segment particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    origin, axis, into, out_of = _frame(collider)
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        r = F(r0 + radius[i])
        if not mass[i] >= SMALL:
            continue
        point = _transform(p[i], into, z_first=True)
        norm = _dot(axis, axis)
        if not norm >= SMALL:
            continue
        offset = point - origin
        t = F(_dot(offset, axis) * F(seeds.rcp(norm)))
        t = t if t > 0 else F(0)    # MAXSS(t, 0)
        t = t if t < 1 else F(1)    # MINSS(t, 1)
        along = axis*t
        perpendicular = offset - along
        q = _dot(perpendicular, perpendicular)
        if not F(r*r) > q or not q >= SMALL:
            continue
        s = F(F(seeds.rsqrt(q)) * r)
        p[i] = _transform((perpendicular*s + along) + origin, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native segment response produced nonfinite coordinates')
    return p


@dataclass(frozen=True)
class EdgeSegmentCollider2:
    transform: np.ndarray
    inverse_transform: np.ndarray
    origin: np.ndarray
    axis: np.ndarray
    radius: float
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def _clamp01(x):
    x = x if x > 0 else F(0)       # MAXSS(x, 0)
    return x if x < 1 else F(1)    # MINSS(x, 1)


def _masked_sqrt(x, seeds):
    # rsqrt(x)*x, kept only where 0 < x (CMPLTSS mask)
    return F(F(seeds.rsqrt(x)) * x) if F(0) < x else F(0)


def closest_segment_parameters_eps(origin0, direction0, origin1, direction1, *, seeds):
    """RVA 28D1570: like 28D17D0, but the parallel test is b*b > 2^-32, a
    near-singular system picks s = 0 or 1 against half the denominator, and
    s is recomputed when t leaves [0, 1]."""
    eps = F(2.0**-32)
    r = origin0 - origin1
    a = _dot(direction0, direction0)
    e = _dot(direction1, direction1)
    f = _dot(direction1, r)
    if not a >= SMALL:
        if not e >= SMALL:
            return F(0), F(0)
        return F(0), _clamp01(F(F(seeds.rcp(e)) * f))
    c = _dot(direction0, r)
    if not e >= SMALL:
        return _clamp01(F(F(seeds.rcp(a)) * F(-c))), F(0)
    b = _dot(direction0, direction1)
    bb = F(b*b)
    s = F(0)
    if bb > eps:
        denominator = F(F(e*a) - bb)
        number = F(F(b*f) - F(c*e))
        if denominator > eps:
            s = _clamp01(F(F(seeds.rcp(denominator)) * number))
        elif not number < F(denominator*F(.5)):
            s = F(1)
    t = F(F(seeds.rcp(e)) * F(F(s*b) + f))
    if F(0) > t:
        return _clamp01(F(F(seeds.rcp(a)) * F(-c))), F(0)
    if t > F(1):
        return _clamp01(F(F(seeds.rcp(a)) * F(b - c))), F(1)
    return s, t


def apply_edge_segment(positions, inverse_mass, radii, constraints, collider, *, seeds):
    """RVA 28D51F0: spring edges against the one-bone capsule (flag 4 set).

    Springs come from edge_items (all springs, or the pair list with flag 8).
    Pinned endpoints already inside the bare capsule (radius r, no particle
    radius) skip the spring unless the spring has flag 8. With both endpoint
    radii below 2^-16 the contact subsegment is tested as a line (28D17D0);
    otherwise as a tapered band: the subsegment itself (28D1570) and, from the
    capsule origin's perpendicular, the subsegment offset by the endpoint
    radii, the deeper of the two pushing both movable endpoints.
    """
    from .native_cloth_contact import closest_segment_parameters, edge_items
    p = _particles(positions)
    n = len(p)
    if not collider.flags & 4:
        raise ValueError('Point one-bone capsule collisions use apply_point_segment (28D6DA0)')
    mass = _array(inverse_mass, (n,), 'Segment inverse mass')
    radius = _array(radii, (n,), 'Particle radii')
    R = F(collider.radius)
    if not np.isfinite(R):
        raise ValueError('Segment radius must be finite')
    origin, axis, into, out_of = _frame(collider)
    norm = _dot(axis, axis)

    def inside(point):
        if not norm >= SMALL:
            return False
        offset = point - origin
        t = _clamp01(F(_dot(offset, axis) * F(seeds.rcp(norm))))
        d = offset - axis*t
        return bool(F(R*R) > _dot(d, d))

    def push(a, b, correction, free_a, free_b):
        world = (correction[2]*out_of[2] + correction[0]*out_of[0]) + correction[1]*out_of[1]
        if free_a:
            p[a] = world + p[a]
        if free_b:
            p[b] = world + p[b]

    for a, b, r in edge_items(constraints, collider, n):
        free_a, free_b = mass[a] > 0, mass[b] > 0
        pa = _transform(p[a], into, z_first=True)
        pb = _transform(p[b], into, z_first=True)
        if not r.flags & 8:
            if not free_a and inside(pa):
                continue
            if not free_b and inside(pb):
                continue
        if not free_a and not free_b:
            continue
        start, end = F(r.contact_start), F(r.contact_end)
        ra, rb = radius[a], radius[b]
        edge = pb - pa
        ca = edge*start + pa
        cb = edge*end + pa
        direction = cb - ca
        size0 = F(F(F(F(1)-start)*ra) + F(start*rb))
        size1 = F(F(F(F(1)-end)*ra) + F(rb*end))
        if abs(ra) < SMALL and abs(rb) < SMALL:
            s, t = closest_segment_parameters(ca, direction, origin, axis, seeds=seeds)
            delta = (direction*s + ca) - (axis*t + origin)
            effective = F(F(F(F(F(1)-s)*size0) + F(size1*s)) + R)
            q = _dot(delta, delta)
            if not F(effective*effective) > q or not q >= SMALL:
                continue
            push(a, b, delta*F(F(F(seeds.rsqrt(q))*effective)-F(1)), free_a, free_b)
            continue
        length2 = _dot(edge, edge)
        s, t = closest_segment_parameters_eps(ca, direction, origin, axis, seeds=seeds)
        first = (direction*s + ca) - (axis*t + origin)
        depth1 = F(F(F(F(F(1)-s)*size0) + F(s*size1)) + R) - _masked_sqrt(_dot(first, first), seeds)
        depth1 = F(depth1)
        depth2, second = F(0), np.zeros(4, np.float32)
        if length2 >= SMALL:
            w = origin - pa
            perpendicular = w - edge*_dot(edge, w)
            p2 = _dot(perpendicular, perpendicular)
            if p2 >= SMALL:
                normal = perpendicular*F(seeds.rsqrt(p2))
                q1 = normal*size0 + ca
                q2 = normal*size1 + cb
                band = q2 - q1
                s2, t2 = closest_segment_parameters_eps(q1, band, origin, axis, seeds=seeds)
                second = (band*s2 + q1) - (axis*t2 + origin)
                depth2 = F(R - _masked_sqrt(_dot(second, second), seeds))
        if not (depth1 >= SMALL or depth2 >= SMALL):
            continue
        delta = second if depth2 > depth1 else first
        q = _dot(delta, delta)
        if not q >= SMALL:
            continue
        depth = depth2 if depth1 < depth2 else depth1
        push(a, b, delta*F(F(seeds.rsqrt(q))*depth), free_a, free_b)
    if not np.isfinite(p).all():
        raise ValueError('Native segment response produced nonfinite coordinates')
    return p


@dataclass(frozen=True)
class InnerSegmentCollider3:
    """Runtime tag 3 (FLTD file kind 4): keeps particles INSIDE the capsule.

    Built by the same 28BBE60 code as file kind 3 (origin +0xB0, axis +0xC0,
    radius +0xD0); its edge entry 28D7890 calls the point leaf 28D78B0.
    """
    transform: np.ndarray
    inverse_transform: np.ndarray
    origin: np.ndarray
    axis: np.ndarray
    radius: float
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def apply_inner_segment(positions, inverse_mass, collider, exclusion_mask, *, seeds):
    """RVA 28D78B0: each movable particle (inverse mass > 0) farther than the
    shape's own radius (no particle radius) from the clamped axis segment is
    put back on the capsule surface. Filter list and local mode as tag 2."""
    p = _particles(positions)
    n = len(p)
    mass = _array(inverse_mass, (n,), 'Segment inverse mass')
    r = F(collider.radius)
    if not np.isfinite(r):
        raise ValueError('Segment radius must be finite')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Segment particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    origin, axis, into, out_of = _frame(collider)
    norm = _dot(axis, axis)
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        if not mass[i] > 0 or not norm >= SMALL:
            continue
        offset = _transform(p[i], into, z_first=True) - origin
        along = axis*_clamp01(F(_dot(offset, axis) * F(seeds.rcp(norm))))
        perpendicular = offset - along
        q = _dot(perpendicular, perpendicular)
        if not F(r*r) < q or not q >= SMALL:
            continue
        s = F(F(seeds.rsqrt(q)) * r)
        p[i] = _transform((perpendicular*s + along) + origin, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native segment response produced nonfinite coordinates')
    return p
