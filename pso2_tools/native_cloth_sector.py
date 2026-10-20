"""Tag-6 collider (FLTD file kind 6): an annular cylinder sector.

Built by 28BBE60 (case 6) -> 28DA1D0 -> 28E1910 from file values 0..2
(origin), 3..5 (axis; its length is the height), 6/7 (inner and outer radius,
sorted), 8 (start angle, degrees) and 9 (angular width, degrees). Angles
become 16-bit units (v * 32768 / 180, truncated) and pass through the
game's own float sincos (273A3A0). Particles are kept INSIDE the wedge,
between the two radii and between the two caps.
"""

from dataclasses import dataclass
import math
import struct

import numpy as np

from .native_cloth_plane import _transform
from .native_cloth_stages import _particles
from .native_cloth_writeback import F, SMALL, _array, _dot

_ANGLE = np.float32(9.58738019107841e-05)    # 2*pi/65536 (2C80AA0)
_TINY = F(2.0**-23)                          # 0x34000000 lane test in 28E1910


def _double(bits):
    return struct.unpack('<d', struct.pack('<Q', bits))[0]


_INV_PI = _double(0x3fd45f306dc9c883)
_PI = _double(0x400921fb54442d18)
_HALF_PI = _double(0x3ff921fb54442d18)
_S1 = _double(0x3f81100380000000)
_S2 = _double(0xbf2973e4e0000000)
_S3 = _double(0xbfc5555300000000)
_C1 = _double(0xbf567a8620000000)
_C2 = _double(0xbfdfffff20000000)
_C3 = _double(0x3fa554df60000000)


def _f32bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def sincos(x):
    """RVA 273A3A0: float sin and cos with double polynomials.

    The argument is reduced by multiples of pi; the quotient parity (floor and
    round-to-nearest-even of x/pi + 1/2) picks the signs."""
    d = float(np.float32(x))
    v = d*_INV_PI + .5
    n = math.floor(v)
    r = (v - n) - .5
    lane = _f32bits(np.float32(r)) & 0x7fffffff
    r = r*_PI
    floor_odd = int(n) & 1
    round_odd = round(v) & 1
    if lane >= 0x3efffffd:
        return (F(1) if round_odd else F(-1)), F(0)
    r2 = r*r

    def odd(h):
        h2 = h*h
        y = h2*h2
        return np.float32((_S2*y + _S3)*(h2*h) + (_S1*y + 1.)*h)

    def even(h2):
        y = h2*h2
        return np.float32((_C1*y + _C2)*h2 + (_C3*y + 1.))

    if lane < 0x3e981c44:
        s = np.float32((_S2*(r2*r2) + _S3)*(r2*r) + (_S1*(r2*r2) + 1.)*r)
        sin = -s if floor_odd else s
        if lane < 0x3e527d50:
            c = even(r2)
            return F(sin), F(-c if floor_odd else c)
    else:
        h = _HALF_PI - abs(r)
        c = even(h*h)
        sin = c if round_odd else -c
    c = odd(_HALF_PI - abs(r))
    return F(sin), F(-c if floor_odd else c)


def _perm(v, order):
    return v[list(order)]


def _cross(a, b):
    # 28E1A48..60 and its repeats: (a.yzx*b - b.yzx*a).yzxy
    return _perm(_perm(a, (1, 2, 0, 3))*b - _perm(b, (1, 2, 0, 3))*a, (1, 2, 0, 1))


def _flip(v, flips):
    x, y, z = (-v[i] if flips[i] else v[i] for i in range(3))
    return np.array([x, y, z, 0], np.float32)


def angle_units(degrees):
    """28BCF06..5C: float(v*32768)/180, truncated toward zero."""
    value = F(F(F(32768)*F(degrees)) / F(180))
    if not np.isfinite(value) or abs(value) >= 2**31:
        raise ValueError('Sector angle must be finite')
    return int(value)


@dataclass(frozen=True)
class SectorShape:
    origin: np.ndarray       # +0xB0
    axis: np.ndarray         # +0xC0
    normal0: np.ndarray      # +0xD0
    normal1: np.ndarray      # +0xE0
    edge0: np.ndarray        # +0xF0
    edge1: np.ndarray        # +0x100
    middle: np.ndarray       # +0x110
    inner: np.float32        # +0x120
    outer: np.float32        # +0x124


def build_sector(origin, axis, radius0, radius1, start, width, flips, *, seeds):
    """28DA1D0/28DA380/28E1910. `origin`/`axis` are the raw file vectors,
    `start`/`width` 16-bit angle units, `flips` the option 0x20/0x40/0x80
    axis mirrors (applied to every stored vector)."""
    o = np.array([*np.asarray(origin, np.float32)[:3], 0], np.float32)
    b = np.array([*np.asarray(axis, np.float32)[:3], 0], np.float32)
    zero = np.zeros(4, np.float32)
    lanes = np.abs(b[:3]) < _TINY
    if lanes.all():
        u = zero
    else:
        ax, ay, az = np.abs(b[:3])
        x, y, z = b[:3]
        if ay >= ax:
            v = (-z, z, F(x - y)) if az >= ay else (-y, F(x - z), y)
        else:
            v = (-z, z, F(x - y)) if az >= ax else (F(y - z), -x, x)
        v = np.array([*v, 0], np.float32)
        u = v*F(seeds.rsqrt(_dot(v, v)))
    c = _cross(u, b)
    fields = [zero] * 5
    if not lanes.all() and not (np.abs(c[:3]) < _TINY).all():
        cn = c / F(math.sqrt(_dot(c, c)))
        bn = b / F(math.sqrt(_dot(b, b)))
        s, k = sincos(F(F(start)*_ANGLE))
        n0 = (-s)*u + k*cn
        e0 = _cross(bn, n0)
        s, k = sincos(F(F(start + width)*_ANGLE))
        n1 = s*u - k*cn
        e1 = _perm(_perm(n1, (1, 2, 0, 3))*bn - n1*_perm(bn, (1, 2, 0, 3)), (1, 2, 0, 1))
        s, k = sincos(F(F(start + int(width / 2))*_ANGLE))
        m = s*u - k*cn
        middle = _perm(_perm(m, (1, 2, 0, 3))*bn - _perm(bn, (1, 2, 0, 3))*m, (1, 2, 0, 1))
        fields = [_flip(f, flips) for f in (n0, n1, e0, e1, middle)]
    if sum(bool(f) for f in flips) & 1:
        fields[0], fields[1], fields[2], fields[3] = fields[1], fields[0], fields[3], fields[2]
    r0, r1 = F(radius0), F(radius1)
    if r0 > r1:
        r0, r1 = r1, r0
    return SectorShape(_flip(o, flips), _flip(b, flips), *fields, r0, r1)


@dataclass(frozen=True)
class SectorCollider6:
    transform: np.ndarray
    inverse_transform: np.ndarray
    shape: SectorShape
    flags: int
    filter_flags: int
    particle_filter: tuple = ()


def _vector(v, m):
    return (v[2]*m[2] + v[0]*m[0]) + v[1]*m[1]


def apply_point_sector(positions, inverse_mass, collider, exclusion_mask, *, seeds):
    """RVA 28E04B0 (flag 4 clear): keep movable particles inside the sector.

    Outside a side plane (d.n < 0) the offset is first moved back: for a wedge
    of at most 180 degrees ((e1 x e0) . axis >= 0) onto the axis plus the
    nearer edge direction, for a reflex wedge (only when outside both planes)
    by the larger of the two plane distances. Then the radial part is pushed
    into [inner, outer] (on the axis: out along the middle direction), and
    the axial parameter into [0, 1]. Filter list, mask and local mode as the
    other leaves."""
    p = _particles(positions)
    n = len(p)
    if collider.flags & 4:
        raise ValueError('Edge sector collisions use apply_edge_sector (28DCA60)')
    mass = _array(inverse_mass, (n,), 'Sector inverse mass')
    order = tuple(int(i) for i in collider.particle_filter) or tuple(range(n))
    if any(not 0 <= i < n for i in order):
        raise ValueError('Sector particle filter must index existing particles')
    mask = int(exclusion_mask) if exclusion_mask is not None else 0
    test_mask = bool(collider.filter_flags & 0xFFFFFFFE)
    matrix = _array(collider.transform, (4, 4), 'Sector world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Sector inverse transform')
    s = collider.shape
    local = bool(collider.flags & 2)
    identity = np.eye(4, dtype=np.float32)
    if local:
        o, h, n0, n1, e0, e1, middle = s.origin, s.axis, s.normal0, s.normal1, s.edge0, s.edge1, s.middle
    else:
        o = _transform(s.origin, matrix, z_first=True)
        h, n0, n1, e0, e1 = (_vector(v, matrix) for v in (s.axis, s.normal0, s.normal1, s.edge0, s.edge1))
        middle = _vector(s.middle, matrix)
    into = inverse if local else identity
    out_of = matrix if local else identity
    reflex = _dot(_cross(e1, e0), h) < 0
    for i in order:
        if test_mask and mask >> (i & 63) & 1:
            continue
        if not mass[i] > 0:
            continue
        hh = _dot(h, h)
        if not hh >= SMALL:
            continue
        d = _transform(p[i], into, z_first=True) - o
        a, b = _dot(d, n0), _dot(d, n1)
        side = False
        if a < 0 or b < 0:
            if not reflex:
                side = True
                edge = e0 if a < 0 else e1
                m = _dot(d, edge)
                m = F(0) if F(0) > m else m
                d = h*F(F(seeds.rcp(hh))*_dot(d, h)) + edge*m
            elif a < 0 and b < 0:
                side = True
                d = d - (n1*b if b > a else n0*a)
        rcp = F(seeds.rcp(hh))
        along_dot = _dot(d, h)
        t = F(along_dot*rcp)
        along = h*t
        radial = d - along
        q = _dot(radial, radial)
        moved = False
        if q >= SMALL:
            r = s.inner
            if not F(r*r) > q:
                r = s.outer
                r = r if q > F(r*r) else None
            if r is not None:
                d = radial*F(F(F(seeds.rsqrt(q))*r) - F(1)) + d
                moved = True
        elif s.inner >= SMALL:
            d = middle*s.inner + d
            moved = True
        if F(0) > along_dot:
            d = d - along
        elif along_dot > hh:
            d = d - h*F(t - F(1))
        elif not moved and not side:
            continue
        p[i] = _transform(d + o, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native sector response produced nonfinite coordinates')
    return p


TRACE = None    # set to a list to record the branches taken (validation only)


def _max0(x):
    return F(0) if F(0) > x else x          # MAXSS(0, x)


def segment_axis_parameter(a, b, axis, *, seeds):
    """28E1EF0, as compiled: clamp((a.h)(s.h) - rcp(det) * ((a.s)(h.h)), 0, 1)
    with s = b - a and det = (h.h)(s.s) - (s.h)^2 (only the second term is
    divided by det); 0 when det < 2^-16."""
    s = b - a
    ss, sd, dd = _dot(s, s), _dot(s, axis), _dot(axis, axis)
    det = F(F(dd*ss) - F(sd*sd))
    if not det >= SMALL:
        return F(0)
    v = F(F(_dot(a, axis)*sd) - F(F(seeds.rcp(det)) * F(_dot(a, s)*dd)))
    v = v if v > F(0) else F(0)
    return v if v < F(1) else F(1)


def _drop_weaker(v, n0, n1):
    """28E1F90: remove the component along the normal with the smaller abs dot."""
    a, b = _dot(v, n0), _dot(v, n1)
    return v - n0*a if abs(b) > abs(a) else v - n1*b


def apply_edge_sector(positions, inverse_mass, constraints, collider, *, seeds):
    """RVA 28DCA60 (flag 4, all-springs mode): keep spring edges inside the sector.

    For each spring with a movable endpoint the two endpoint offsets are first
    brought back across the side planes (rules differ for a wedge of at most
    or beyond 180 degrees), then the segment point nearest the axis
    (segment_axis_parameter) is pushed past the inner radius by moving both
    endpoints, and each endpoint is held between the radii and the caps as in
    the point leaf. If anything moved, both endpoints are rewritten, pinned
    ones included. Particle radii are not read.
    """
    from .native_cloth_contact import edge_items
    p = _particles(positions)
    n = len(p)
    if not collider.flags & 4:
        raise ValueError('Point sector collisions use apply_point_sector (28E04B0)')
    if collider.flags & 8:
        raise ValueError('Pair-list edge sector collisions are not implemented')
    mass = _array(inverse_mass, (n,), 'Sector inverse mass')
    matrix = _array(collider.transform, (4, 4), 'Sector world transform')
    inverse = _array(collider.inverse_transform, (4, 4), 'Sector inverse transform')
    s = collider.shape
    local = bool(collider.flags & 2)
    identity = np.eye(4, dtype=np.float32)
    if local:
        o, h, n0, n1, e0, e1, middle = s.origin, s.axis, s.normal0, s.normal1, s.edge0, s.edge1, s.middle
    else:
        o = _transform(s.origin, matrix, z_first=True)
        h, n0, n1, e0, e1, middle = (_vector(v, matrix) for v in
                                     (s.axis, s.normal0, s.normal1, s.edge0, s.edge1, s.middle))
    into = inverse if local else identity
    out_of = matrix if local else identity
    r0, r1 = s.inner, s.outer
    reflex = _dot(_cross(e1, e0), h) < 0

    def recentre(A, B):
        t = segment_axis_parameter(A, B, h, seeds=seeds)
        P = A*F(F(1) - t) + B*t
        delta = h*F(F(seeds.rcp(_dot(h, h)))*_dot(h, P)) - P
        return A + delta, B + delta, P

    trace = TRACE if TRACE is not None else []
    for ia, ib, record in edge_items(constraints, collider, n):
        if not (mass[ia] > 0 or mass[ib] > 0):
            continue
        A = _transform(p[ia], into, z_first=True) - o
        B = _transform(p[ib], into, z_first=True) - o
        a_n0, a_n1 = _dot(n0, A), _dot(n1, A)
        b_n0, b_n1 = _dot(n0, B), _dot(n1, B)
        out_a0, out_a1 = a_n0 < 0, a_n1 < 0
        in_b0, in_b1 = F(0) <= b_n0, F(0) <= b_n1
        sided = False
        rh = F(seeds.rcp(_dot(h, h)))

        def onto(v, e, m):
            return h*F(_dot(v, h)*rh) + e*_max0(m)

        if reflex:
            if not ((not out_a0 and in_b0) or (not out_a1 and in_b1)):
                both_a = out_a0 and out_a1
                both_b = not in_b0 and not in_b1
                sided = True
                if not both_a:
                    if both_b:
                        if out_a0:
                            B = B - n1*_dot(B, n1)
                        elif out_a1:
                            B = B - n0*_dot(B, n0)
                        else:
                            B = _drop_weaker(B, n0, n1)
                    else:
                        t = segment_axis_parameter(A, B, h, seeds=seeds)
                        P = A*F(F(1) - t) + B*t
                        if F(0) <= _dot(n0, P) or F(0) <= _dot(n1, P):
                            sided = False
                        else:
                            delta = h*F(rh*_dot(h, P)) - P
                            A, B = A + delta, B + delta
                elif both_b:
                    a0, b0, a1, b1 = _dot(A, e0), _dot(B, e0), _dot(A, e1), _dot(B, e1)
                    on_a0 = a0 > F(0) and not F(0) < a1
                    on_a1 = F(0) < a1 and a0 <= F(0)
                    on_b0 = b0 > F(0) and not F(0) < b1
                    on_b1 = F(0) < b1 and b0 <= F(0)
                    use0 = on_a0 and on_b0
                    use1 = (not use0) and on_a1 and on_b1
                    fallback = False
                    if not use0 and not use1:
                        weaker_b = _dot(n1, B) < _dot(n0, B)
                        if a_n0 <= a_n1:
                            use1 = not weaker_b
                        else:
                            use0 = weaker_b
                        fallback = not use0 and not use1
                    if use0:
                        A, B = onto(A, e0, a0), onto(B, e0, b0)
                    elif use1:
                        A, B = onto(A, e1, a1), onto(B, e1, b1)
                    elif fallback:
                        t = segment_axis_parameter(A, B, h, seeds=seeds)
                        P = A*F(F(1) - t) + B*t
                        delta = h*F(rh*_dot(h, P)) - P
                        A, B = A + delta, B + delta
                elif in_b0:
                    if in_b1:
                        d0, d1 = _dot(A, n0), _dot(A, n1)
                        A = A - n1*d1 if abs(d1) <= abs(d0) else A - n0*d0
                    else:
                        A = A - n0*_dot(A, n0)
                else:
                    A = A - n1*_dot(A, n1)
        else:
            if F(0) <= _dot(e0, e1):
                if out_a0:
                    A = onto(A, e0, _dot(A, e0))
                    sided = True
                elif out_a1:
                    A = onto(A, e1, _dot(A, e1))
                    sided = True
                if not (in_b0 and in_b1):
                    edge = e1 if in_b0 else e0
                    B = onto(B, edge, _dot(B, edge))
                    sided = True
            else:
                if out_a0:
                    A = A - n0*_dot(A, n0)
                sided = out_a0
                if _dot(n1, A) < 0:
                    A = A - n1*_dot(A, n1)
                    sided = True
                if not in_b0:
                    B = B - n0*_dot(B, n0)
                    sided = True
                if _dot(n1, B) < 0:
                    B = B - n1*_dot(B, n1)
                    sided = True
        trace.append(('reflex' if reflex else 'convex', bool(out_a0), bool(out_a1), bool(in_b0), bool(in_b1), sided))
        # the segment point nearest the axis against the inner radius
        t = segment_axis_parameter(A, B, h, seeds=seeds)
        P = A*F(F(1) - t) + B*t
        radial = P - h*F(_dot(h, P)*rh)
        q = _dot(radial, radial)
        if q < F(r0*r0):
            if q < SMALL:
                shift = middle*r0
            else:
                length = F(F(seeds.rsqrt(q))*q) if F(0) < q else F(0)
                shift = (radial*F(seeds.rsqrt(q)))*F(r0 - length)     # 28DE40F/413
            A, B = shift + A, shift + B
            sided = True
        hh = _dot(h, h)

        def hold(v):
            dot = _dot(v, h)
            k = F(dot*rh)
            along = h*k
            rad = v - along
            q = _dot(rad, rad)
            moved = False
            if q < SMALL:
                if SMALL <= r0:
                    v = middle*r0 + v
                    moved = True
            elif q < F(r0*r0):
                v = rad*F(F(F(seeds.rsqrt(q))*r0) - F(1)) + v
                moved = True
            elif F(r1*r1) < q:
                v = rad*F(F(F(seeds.rsqrt(q))*r1) - F(1)) + v
                moved = True
            return v, moved, dot, k, along

        A, moved_a, dot_a, k_a, along_a = hold(A)
        if F(0) <= dot_a:
            if hh < dot_a:
                A = A - h*F(k_a - F(1))
                moved_a = True
            else:
                moved_a = moved_a or sided
        else:
            A = A - along_a
            moved_a = True
        B, moved_b, dot_b, k_b, along_b = hold(B)
        if F(0) <= dot_b:
            if dot_b <= hh:
                if not moved_b and not moved_a:
                    continue
            else:
                B = B - h*F(k_b - F(1))
        else:
            B = B - along_b
        p[ia] = _transform(A + o, out_of, z_first=True)
        p[ib] = _transform(B + o, out_of, z_first=True)
    if not np.isfinite(p).all():
        raise ValueError('Native sector response produced nonfinite coordinates')
    return p
