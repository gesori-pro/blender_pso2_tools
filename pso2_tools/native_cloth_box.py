"""Tag-8 collider (FLTD file kind 8): an oriented box.

Built by 28BBE60 (case 8) -> B70B10 from file values 0..2 (centre, mirrored
by options 0x20/0x40/0x80), 3..5 (full size; half of it spans each axis) and
6..8 (X/Y/Z angles in degrees, turned into a quaternion from half angles in
16-bit units through the game's sincos 273A3A0). The shape keeps four rows
(+0xB0 centre, +0xC0/+0xD0/+0xE0 half axes), re-orthogonalized by
5EF7B0/5F0230 whenever it is transformed. Particles are pushed out of it.
"""

from dataclasses import dataclass
import math

import numpy as np

from .native_cloth_plane import _transform
from .native_cloth_sector import _ANGLE, sincos
from .native_cloth_stages import _particles
from .native_cloth_writeback import F, SMALL, _array


def _dot(a, b):
    # DPPS 0x7f: (x + y) + (z + masked w), so an all-zero sum is +0 even
    # when z*z' is -0 (the box rows carry signed-zero lanes into later sums).
    p = a * b
    return F(F(p[0] + p[1]) + F(p[2] + F(0)))

_INSIDE = F(0.0003662109375)    # 2C7B3F0


def _inverse_or_zero(value):
    return F(0) if SMALL > abs(value) else F(F(1) / value)


def orthogonalize_last(rows):
    """5F0230: row 3 loses its row-1 and row-2 parts (rows as given)."""
    r1, r2 = rows[1], rows[2]
    d1, d2 = _dot(r1, r1), _dot(r2, r2)
    out = rows.copy()
    if d1 >= SMALL:
        out[3] = out[3] - (r1*_dot(out[3], r1))*_inverse_or_zero(d1)
    if d2 >= SMALL:
        out[3] = out[3] - (r2*_dot(out[3], r2))*_inverse_or_zero(d2)
    return out


def _second_against_first(rows):
    d = _dot(rows[1], rows[1])
    out = rows.copy()
    if d >= SMALL:
        out[2] = out[2] - (rows[1]*_dot(out[2], rows[1]))*_inverse_or_zero(d)
    return out


def transform_rows(rows, matrix):
    """5EF7B0: row 0 as a point, rows 1..3 as vectors, then the Gram-Schmidt
    pass of rows 2 and 3."""
    m = np.asarray(matrix, np.float32)
    out = np.empty((4, 4), np.float32)
    out[0] = _transform(rows[0], m, z_first=True)
    for i in (1, 2, 3):
        v = rows[i]
        out[i] = (v[2]*m[2] + v[0]*m[0]) + v[1]*m[1]
    return orthogonalize_last(_second_against_first(out))


def _half_angle(degrees):
    value = F(F(F(32768)*F(degrees)) / F(180))
    if not np.isfinite(value) or abs(value) >= 2**31:
        raise ValueError('Box angle must be finite')
    return F(F(int(int(value) / 2))*_ANGLE)


def _shuffle(a, b, imm):
    return np.array([a[imm & 3], a[imm >> 2 & 3], b[imm >> 4 & 3], b[imm >> 6 & 3]], np.float32)


def rotation_rows(angles):
    """B70B10: quaternion from half angles, then its row matrix."""
    sz, cz = sincos(_half_angle(angles[2]))
    sx, cx = sincos(_half_angle(angles[0]))
    sy, cy = sincos(_half_angle(angles[1]))
    v = lambda *x: np.array(x, np.float32)
    first = (v(cy, cy, sy, cy)*v(sx, sx, sx, cx))*(v(cz, sz, cz, cz)*v(1, -1, -1, 1))
    second = (v(sy, sy, cy, sy)*v(cx, cx, cx, sx))*v(sz, cz, sz, sz)
    q = first + second
    cross = q[[2, 0, 1, 3]]*q[[1, 2, 0, 3]]
    cross = cross + cross
    wq = q[3]*q
    wq = wq + wq
    qq = q*q
    qq = qq + qq
    diagonal = np.ones(4, np.float32) - (qq[[1, 2, 0, 3]] + qq)
    plus, minus = wq + cross, cross - wq
    plus[3] = minus[3] = diagonal[3] = 0
    rows = np.empty((4, 4), np.float32)
    rows[0] = _shuffle(_shuffle(diagonal, plus, 0xed), minus, 0xd8)
    rows[1] = _shuffle(_shuffle(minus, diagonal, 0xee), plus, 0xc8)
    rows[2] = _shuffle(_shuffle(plus, minus, 0xcd), diagonal, 0xc8)
    rows[3] = (0, 0, 0, 1)
    return rows


def build_box(centre, size, angles, flips):
    """B70B10: the four stored rows (+0xB0..+0xEF)."""
    a = np.asarray(centre, np.float32)
    x, y, z = (-a[i] if flips[i] else a[i] for i in range(3))
    centre = np.array([x, y, z, 0], np.float32)
    b = np.asarray(size, np.float32)
    rows = np.zeros((4, 4), np.float32)
    rows[1] = np.array([.5, 0, 0, 0], np.float32)*b[0]
    rows[2] = np.array([0, .5, 0, 0], np.float32)*b[1]
    rows[3] = np.array([0, 0, .5, 0], np.float32)*b[2]
    rows = orthogonalize_last(_second_against_first(rows))
    out = transform_rows(rows, rotation_rows(angles))
    out[0] = centre + out[0]
    return orthogonalize_last(_second_against_first(out))


def box_surface_parameter(rows, t):
    """28E7740: snap the clamped box parameter to the face(s) with the
    smallest axis_length^2 * (1 - |t|)."""
    sq = np.array([_dot(rows[1], rows[1]), _dot(rows[2], rows[2]), _dot(rows[3], rows[3]), 0], np.float32)
    g = sq*(np.ones(4, np.float32) - np.abs(t))
    out = t.copy()

    def snap(*axes):
        for i in axes:
            out[i] = np.floor(t[i]) if F(0) > t[i] else np.ceil(t[i])

    dxy = F(g[0] - g[1])
    if SMALL > abs(dxy):
        dxz = F(g[0] - g[2])
        snap(*((0, 1, 2) if SMALL > abs(dxz) else (0, 1) if F(0) > dxz else (2,)))
    elif F(0) > dxy:
        dxz = F(g[0] - g[2])
        snap(*((0, 2) if SMALL > abs(dxz) else (0,) if F(0) > dxz else (2,)))
    else:
        dyz = F(g[1] - g[2])
        snap(*((1, 2) if SMALL > abs(dyz) else (1,) if F(0) > dyz else (2,)))
    return out


def box_contact(rows, point, *, seeds):
    """28E71B0 (and its inline copy in 28EA050): the box parameter of
    `point`, the snapped surface point and whether the point is inside."""
    c, x, y, z = rows
    d = point - c
    t = np.array([F(F(seeds.rcp(_dot(x, x)))*_dot(d, x)), F(F(seeds.rcp(_dot(y, y)))*_dot(d, y)),
                  F(F(seeds.rcp(_dot(z, z)))*_dot(d, z)), 0], np.float32)
    low, high = np.array([-1, -1, -1, 0], np.float32), np.array([1, 1, 1, 0], np.float32)
    clamped = np.where(t > low, t, low)
    clamped = np.where(clamped < high, clamped, high).astype(np.float32)
    p = box_surface_parameter(rows, clamped)
    inside = all(F(F(t[i]*t[i]) - F(1)) < _INSIDE for i in range(3))
    surface = ((x*p[0] + y*p[1]) + z*p[2]) + c
    return surface, inside


@dataclass(frozen=True)
class BoxCollider8:
    transform: np.ndarray
    inverse_transform: np.ndarray
    rows: np.ndarray
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def apply_point_box(positions, inverse_mass, radii, collider, exclusion_mask, *, seeds):
    """RVA 28EA050 (flag 4 clear): push particles (inverse mass >= 2^-16) out
    of the box. A particle with |radius| >= 2^-16 inside the box, or outside
    but nearer than its radius, ends `radius` beyond the snapped surface
    point; with a smaller radius only inside particles move, onto it."""
    p = _particles(positions)
    n = len(p)
    if collider.flags & 4:
        raise ValueError('Edge box collisions (28E8940) are not implemented')
    mass = _array(inverse_mass, (n,), 'Box inverse mass')
    radius = _array(radii, (n,), 'Particle radii')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Box particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    matrix = _array(collider.transform, (4, 4), 'Box world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Box inverse transform')
    local = bool(collider.flags & 2)
    identity = np.eye(4, dtype=np.float32)
    rows = transform_rows(_array(collider.rows, (4, 4), 'Box rows'), identity if local else matrix)
    into = inverse if local else identity
    out_of = matrix if local else identity
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        r = radius[i]
        if not mass[i] >= SMALL:
            continue
        point = _transform(p[i], into, z_first=True)
        surface, inside = box_contact(rows, point, seeds=seeds)
        if SMALL > abs(r):
            if not inside:
                continue
            result = surface
        else:
            normal = surface - point if inside else point - surface
            q = _dot(normal, normal)
            if not inside and not F(r*r) > q:
                continue
            if not q >= SMALL:
                continue
            result = normal*F(F(seeds.rsqrt(q))*r) + surface
        p[i] = _transform(result, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native box response produced nonfinite coordinates')
    return p


_NEG_SMALL = F(-1.52587890625e-05)       # 2C7B214
_FLT_MAX = np.float32(3.4028234663852886e+38)


def _masked_length(q, seeds):
    return F(F(seeds.rsqrt(q)) * q) if F(0) < q else F(0)


def sphere_overlaps_box(rows, centre, radius):
    """28E7070: squared distance from the box (centre + unit axes scaled by
    their lengths) to the point, against radius squared."""
    c = rows[0]
    total = None
    for axis in rows[1:]:
        length = F(np.sqrt(np.float32(_dot(axis, axis))))
        e = _dot(axis / length, centre - c)
        if F(-length) <= e:
            if not length < e:
                continue
            d = F(e - length)
        else:
            d = F(e + length)
        total = F(F(d*d) + F(0)) if total is None else F(total + F(d*d))
    total = F(0) if total is None else total
    return bool(total <= F(radius*radius))


def segment_box_slabs(rows, origin, direction, *, seeds):
    """28E7310: slab test of origin + t * unit(direction), t in [0, |direction|].
    Returns None on a miss, else (t_enter, t_exit)."""
    c = rows[0]
    span = F(np.sqrt(np.float32(_dot(direction, direction))))
    unit = direction / span
    offset = c - origin
    t_low, t_high = F(-_FLT_MAX), _FLT_MAX
    for axis in rows[1:]:
        q = _dot(axis, axis)
        half = _masked_length(q, seeds)
        normal = axis / half
        e = _dot(normal, offset)
        f = _dot(normal, unit)
        if abs(f) < SMALL:
            if q < F(e*e):
                return None
            continue
        inv = F(seeds.rcp(f))
        t1, t2 = F(F(e + half)*inv), F(F(e - half)*inv)
        if t2 <= t1:
            t_low = t2 if t2 > t_low else t_low
            t_high = t1 if t1 < t_high else t_high
        else:
            t_low = t1 if t1 > t_low else t_low
            t_high = t2 if t2 < t_high else t_high
        if t_high < t_low or t_high < F(0) or span < t_low:
            return None
    return t_low, t_high


def nearest_box_face(rows, p, q, *, seeds):
    """28EA990: of the six faces (-X, +X, -Y, +Y, -Z, +Z), the plane with the
    smallest |n.(p - f)| + |n.(q - f)| (ties within 2^-16 go to the smaller p
    distance). Returns (unit normal xyz, w = -n.face_centre)."""
    c, x, y, z = rows
    zero = np.zeros(4, np.float32)
    faces = []
    for axis in (x, y, z):
        negative = zero - axis
        faces += [(negative, c + negative), (axis, c + axis)]
    best, normal, centre = _FLT_MAX, zero, zero
    for n, fc in faces:
        nn2 = _dot(n, n)
        if not nn2 >= SMALL:
            continue
        unit = n * F(seeds.rsqrt(nn2))
        d1 = abs(_dot(unit, p - fc))
        score = F(F(d1 + F(0)) + abs(_dot(unit, q - fc)))
        if score < best or (SMALL > abs(F(score - best)) and d1 < abs(_dot(normal, p - centre))):
            best, normal, centre = score, unit, fc
    out = normal.copy()
    out[3] = -_dot(normal, centre)       # XOR sign, as 28EAF5B
    return out


def apply_edge_box(positions, inverse_mass, radii, constraints, collider, *, seeds):
    """RVA 28E8940 (flag 4): spring edges against the oriented box.

    Springs come from edge_items. Without spring flag 8 a pinned endpoint
    already touching the box (radius max(r, 0)) skips the spring. The contact
    subsegment is tested against the box grown by the larger end radius
    (axes lengthened, then re-orthogonalized); from the slab entry/exit points
    the nearest face plane is chosen and each movable endpoint whose entry/exit
    point lies deeper than (grown radius - its own radius) is pushed out along
    the face normal by that depth."""
    from .native_cloth_contact import edge_items
    p = _particles(positions)
    n = len(p)
    if not collider.flags & 4:
        raise ValueError('Point box collisions use apply_point_box (28EA050)')
    mass = _array(inverse_mass, (n,), 'Box inverse mass')
    radius = _array(radii, (n,), 'Particle radii')
    matrix = _array(collider.transform, (4, 4), 'Box world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Box inverse transform')
    local = bool(collider.flags & 2)
    identity = np.eye(4, dtype=np.float32)
    rows = transform_rows(_array(collider.rows, (4, 4), 'Box rows'), identity if local else matrix)
    into = inverse if local else identity
    out_of = matrix if local else identity

    def push(i, depth, normal):
        v = normal * (F(0) - depth)
        p[i] = ((v[2]*out_of[2] + v[0]*out_of[0]) + v[1]*out_of[1]) + p[i]

    for a, b, r in edge_items(constraints, collider, n):
        start, end = F(r.contact_start), F(r.contact_end)
        ra, rb = radius[a], radius[b]
        pa = _transform(p[a], into, z_first=True)
        pb = _transform(p[b], into, z_first=True)
        free_a, free_b = mass[a] > 0, mass[b] > 0
        if not r.flags & 8:
            if not free_a and sphere_overlaps_box(rows, pa, ra if ra > F(0) else F(0)):
                continue
            if not free_b and sphere_overlaps_box(rows, pb, rb if rb > F(0) else F(0)):
                continue
        if not free_a and not free_b:
            continue
        d = pb - pa
        ca = d*start + pa
        segment = (d*end + pa) - ca
        sa = F(F(F(F(1) - start)*ra) + F(start*rb))
        sb = F(F(F(F(1) - end)*ra) + F(end*rb))
        grow = sa if sa > sb else sb
        grown = rows.copy()
        if abs(grow) >= SMALL:
            for i in (1, 2, 3):
                grown[i] = grown[i]*F(F(seeds.rsqrt(_dot(rows[i], rows[i])))*grow) + rows[i]
            g = _dot(grown[1], grown[1])
            if g >= SMALL:
                grown[2] = grown[2] - (grown[1]*_dot(grown[2], grown[1]))*_inverse_or_zero(g)
            grown = orthogonalize_last(grown)
        hit = segment_box_slabs(grown, ca, segment, seeds=seeds)
        if hit is None:
            continue
        t_low, t_high = hit
        q = _dot(segment, segment)
        s = F(seeds.rsqrt(q))
        span = F(s*q) if F(0) < q else F(0)
        beyond = t_high > span
        if F(0) > t_low:
            enter = pa
            exit_ = pb if beyond else segment*F(s*t_high) + ca
        elif beyond:
            enter, exit_ = segment*F(s*t_low) + ca, pb
        else:
            unit = segment*s
            enter, exit_ = unit*t_low + ca, unit*t_high + ca
        plane = nearest_box_face(grown, enter, exit_, seeds=seeds)
        if free_a:
            depth = F(F(_dot(plane, enter) + plane[3]) + F(grow - sa))
            if _NEG_SMALL >= depth:
                push(a, depth, plane)
        if free_b:
            depth = F(F(_dot(plane, exit_) + plane[3]) + F(grow - sb))
            if _NEG_SMALL >= depth:
                push(b, depth, plane)
    if not np.isfinite(p).all():
        raise ValueError('Native box response produced nonfinite coordinates')
    return p
