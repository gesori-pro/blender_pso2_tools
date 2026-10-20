"""Live angular constraint RVA 28F70F0.

The observed live 0x4FF mode uses a 2D capsule/box bound, not an
interchangeable circular cone or guessed elliptical cone. With the 0x2000 basis-propagation mode (FLTD config modes & 2), each record
reads the basis of its reference particle and, after the correction, writes
that basis rotated onto the corrected edge into the parent's basis slot
(28F78B7..28F7B78), so frames travel down the chain within a pass.
"""

from dataclasses import dataclass
import math

import numpy as np

from .native_cloth_stages import constrain_cone
from .native_cloth_writeback import EPS, F, SMALL, _array, _cross, _dot, _rotate, _small


def _masked_sqrt(x, seeds):
    # rsqrt(x)*x, kept only where 0 < x (CMPLTSS mask)
    return F(F(seeds.rsqrt(x)) * x) if F(0) < x else F(0)


def half_unit_basis(basis):
    """28F7C96..CED (solver version 32 wrapper 28F7BE0): a basis longer than
    0.5 is scaled to length 0.5 (sqrt, divide, times 0.5 in every lane)."""
    b = np.asarray(basis, np.float32)
    d = _dot(b, b)
    if d > F(.25):
        b = (b / F(math.sqrt(d))) * F(.5)
    return b


def propagate_basis_frame(basis0, basis1, parent, child, *, seeds):
    """28F78C4: rotate (basis0, basis1) so that their x axis (basis0 x basis1)
    follows the corrected edge parent -> child. None when the native code
    leaves the destination unchanged (degenerate axis/edge, or opposed)."""
    y, z = np.asarray(basis0, np.float32), np.asarray(basis1, np.float32)
    x = _cross(y, z)
    edge = np.asarray(child, np.float32) - np.asarray(parent, np.float32)
    if np.all(np.abs(x[:3]) < EPS) or np.all(np.abs(edge[:3]) < EPS):
        return None
    u = edge * F(seeds.rsqrt(_dot(edge, edge)))
    v = x * F(seeds.rsqrt(_dot(x, x)))
    c = _cross(v, u)
    n2 = _dot(c, c)
    d = _dot(v, u)
    if n2 > 0:
        h = F(d * F(.5))
        h = h if h > F(-.5) else F(-.5)    # MAXSS(h, -0.5)
        h = h if h < F(.5) else F(.5)      # MINSS(h, 0.5)
        w = _masked_sqrt(F(h + F(.5)), seeds)
        s = _masked_sqrt(F(F(seeds.rcp(n2)) * F(F(.5) - h)), seeds)
        q = c * s
        q[3] = w
    elif d >= SMALL:
        q = np.array([0, 0, 0, 1], dtype=np.float32)
    else:
        return None
    return _rotate(q, y), _rotate(q, z)


@dataclass(frozen=True)
class AngularConstraint:
    cosine0: float
    sine0: float
    offset0: float
    offset1: float
    cosine1: float
    sine1: float
    scale: float
    flags: int
    parent: int
    child: int
    reference: int
    target_relative: bool = False


def _capsule_inside(shape, point):
    x, y, dx, dy, radius = shape
    norm = F(F(dx*dx) + F(dy*dy))
    along = F(0)
    if norm >= SMALL:
        inverse = F(F(1) / norm)
        along = F(F(F(F(point[1]-y)*dy) + F(F(point[0]-x)*dx)) * inverse)
        along = min(F(1), max(F(0), along))
    a = F(point[0] - F(F(dx*along)+x))
    b = F(point[1] - F(F(dy*along)+y))
    return F(F(b*b)+F(a*a)) <= F(radius*radius)


def _capsule_boundary(shape, point, seeds):
    x, y, dx, dy, radius = shape
    norm = F(F(dx*dx) + F(dy*dy))
    along = F(0)
    if norm >= SMALL:
        along = F(F(seeds.rcp(norm)) * F(F(F(point[0]-x)*dx) + F(F(point[1]-y)*dy)))
        along = min(F(1), max(F(0), along))
    near_x, near_y = F(F(dx*along)+x), F(F(dy*along)+y)
    a, b = F(point[0]-near_x), F(point[1]-near_y)
    norm = F(F(b*b)+F(a*a))
    if norm >= SMALL:
        inverse = F(seeds.rsqrt(norm))
        a, b = F(a*inverse), F(b*inverse)
    return np.array([F(F(a*radius)+near_x), F(F(b*radius)+near_y)], dtype=np.float32)


def constrain_angle(parent, child, reference, basis0, basis1, parent_mass,
                    child_mass, record, *, seeds, pin_parent=False):
    """Return adjusted float4 endpoints, preserving native float32 order."""
    a =_array(parent, (4,), 'Angular parent').copy()
    b = _array(child, (4,), 'Angular child').copy()
    ref = _array(reference, (4,), 'Angular reference')
    y = _array(basis0, (4,), 'Angular basis 0')
    z = _array(basis1, (4,), 'Angular basis 1')
    p = np.asarray([parent_mass, child_mass, record.cosine0, record.sine0,
                    record.offset0, record.offset1, record.cosine1,
                    record.sine1, record.scale], dtype=np.float32)
    if not np.isfinite(p).all():
        raise ValueError('Angular coefficients must be finite')
    pm, cm, c0, s0, off0, off1, c1, s1, scale = p
    if c0 <= 0:
        return a, b
    if abs(c1) < SMALL or abs(F(c0-c1)) < SMALL:
        return constrain_cone(a, b, ref, y, z, pm, cm, c0, s0, off0, off1,
                              scale, seeds=seeds, pin_parent=pin_parent)
    if c1 <= 0 or s0 <= 0 or s1 <= 0:
        return a, b
    edge = b - a
    axis = (F(scale*off0)*y + (a-ref)) + F(scale*off1)*z
    side0 = _cross(axis, z)
    side1 = _cross(side0, axis)
    if not _small(side0):
        side0 = side0 * F(seeds.rsqrt(_dot(side0, side0)))
    if not _small(side1):
        side1 = side1 * F(seeds.rsqrt(_dot(side1, side1)))
    dot, edge2, axis2 = _dot(axis, edge), _dot(edge, edge), _dot(axis, axis)
    if dot < 0:
        # Behind the axis: the native wrapper temporarily swaps the smaller
        # cosine/sine pair into the circular leaf, then restores the record.
        if c1 < c0:
            c0, s0 = c1, s1
        return constrain_cone(a, b, ref, y, z, pm, cm, c0, s0, off0, off1,
                              scale, seeds=seeds, pin_parent=pin_parent)
    if axis2 <= 0 or edge2 <= 0:
        return a, b
    perpendicular = edge - axis * F(F(seeds.rcp(axis2))*dot)
    if _dot(perpendicular, perpendicular) <= 0:
        return a, b
    axis_distance = abs(_dot(edge, axis * F(seeds.rsqrt(axis2))))
    radius0 = abs(F(F(F(seeds.rcp(c0))*s0)*axis_distance))
    radius1 = abs(F(F(F(seeds.rcp(c1))*s1)*axis_distance))
    diff = abs(F(radius0-radius1))
    xmajor, ymajor = F(radius0 >= radius1), F(radius0 < radius1)
    dx, dy = F(xmajor*diff), F(ymajor*diff)
    segment_x = F(F(xmajor * F(-diff)) - dx)
    segment_y = F(F(ymajor * F(-diff)) - dy)
    point = np.array([_dot(perpendicular, side0), _dot(perpendicular, side1)], dtype=np.float32)
    selected = np.zeros(2, dtype=np.float32)
    if F(F(segment_x*segment_x)+F(segment_y*segment_y)) < SMALL or record.flags & 1:
        if (point[0] < -radius0 or radius0 <= point[0]
                or point[1] < -radius1 or radius1 <= point[1]):
            selected[0] = min(radius0, max(F(-radius0), point[0]))
            selected[1] = min(radius1, max(F(-radius1), point[1]))
    else:
        shape = (dx, dy, segment_x, segment_y, max(F(0), min(radius0, radius1)))
        if not _capsule_inside(shape, point):
            selected = _capsule_boundary(shape, point, seeds)
    norm = F(F(selected[0]*selected[0]) + F(selected[1]*selected[1]))
    if norm <= 0:
        return a, b
    axial = F(F(seeds.rsqrt(axis2))*axis_distance)
    desired = (axis * axial + side0 * selected[0]) + side1 * selected[1]
    inv = F(seeds.rsqrt(F(F(seeds.rcp(edge2))*_dot(desired, desired))))
    new_child = desired * inv + a
    half = (new_child - b) * F(.5)
    change_a, change_b = F(0)-half, half
    if pin_parent or pm <= 0 or norm < SMALL:
        change_b = change_b * F(2)
        change_a = np.zeros(4, dtype=np.float32)
    if cm <= 0:
        change_a = change_a * F(2)
        change_b = np.zeros(4, dtype=np.float32)
    return a + change_a, b + change_b


def constrain_angles(positions, targets, basis0, basis1, inverse_mass,
                     target_constraints, angular_constraints, *, seeds,
                     pin_parent=False, propagate_basis=False, return_bases=False,
                     half_unit_bases=False):
    """Target-angle then three-particle-angle order in 29034A0/28FE440.

    With propagate_basis the pass works on its own copy of the bases (the
    S+1E0/S+1F8 scratch filled from the incoming bases each pass): a record
    reads its reference particle's basis (a target record: its parent's) and
    then overwrites its parent's with that basis turned onto the new edge.

    half_unit_bases is solver version 32 (wrapper 28F7BE0 instead of 28F70F0,
    the only difference between 2903B20/28FF030 and 29034A0/28FE440): a
    record without flag 2 first scales its two source bases to length <= 0.5
    and uses those for both the correction and the propagated frame.
    """
    p = np.asarray(positions, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError('Angular positions must be N x 4')
    p = _array(p, p.shape, 'Angular positions').copy()
    n = len(p)
    target = _array(targets, p.shape, 'Angular targets')
    y = _array(basis0, p.shape, 'Angular bases 0')
    z = _array(basis1, p.shape, 'Angular bases 1')
    mass = _array(inverse_mass, (n,), 'Angular inverse mass')
    target_constraints, angular_constraints = tuple(target_constraints), tuple(angular_constraints)
    for records, has_reference in ((target_constraints, False), (angular_constraints, True)):
        for r in records:
            ids = (r.parent, r.child, r.reference) if has_reference else (r.parent, r.child)
            if any(not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in ids):
                raise ValueError('Angular endpoint/reference IDs must index existing particles')
    if propagate_basis:
        y, z = y.copy(), z.copy()

    def bases(source, record):
        if half_unit_bases and not record.flags & 2:
            return half_unit_basis(y[source]), half_unit_basis(z[source])
        return y[source], z[source]

    def turn(by, bz, a, parent, child):
        if propagate_basis:
            turned = propagate_basis_frame(by, bz, parent, child, seeds=seeds)
            if turned is not None:
                y[a], z[a] = turned

    for r in target_constraints:
        a, b = r.parent, r.child
        reference = (p[a] + p[a]) - target[b]
        if r.target_relative:
            reference = (target[a] - target[b]) + p[a]
        # 29038BB: the parent goes in as a stack copy and is not written back;
        # only the child moves (the moved copy still sets the propagated edge).
        by, bz = bases(a, r)
        parent, p[b] = constrain_angle(p[a], p[b], reference, by, bz,
                                        mass[a], mass[b], r, seeds=seeds,
                                        pin_parent=pin_parent)
        turn(by, bz, a, parent, p[b])
    for r in angular_constraints:
        a, b, previous = r.parent, r.child, r.reference
        source = previous if propagate_basis else a
        by, bz = bases(source, r)
        p[a], p[b] = constrain_angle(p[a], p[b], p[previous], by, bz,
                                      mass[a], mass[b], r, seeds=seeds,
                                      pin_parent=pin_parent)
        turn(by, bz, a, p[a], p[b])
    # The collision pass's normal reflection (flag 0x80) reads these same
    # scratch bases, so propagated frames reach it too.
    return (p, y, z) if return_bases else p
