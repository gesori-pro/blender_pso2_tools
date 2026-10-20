"""Recovered version-33 stages, not a replacement complete cloth solver.

RVA 28F3200: external force; 28F3640: pin/axis attachment;
28F3DC0: target pull; distance sections of 29034A0/28FE440.
Angular constraints and collider dispatch must also execute for real garments.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_writeback import F, SMALL, _array, _cross, _dot


@dataclass(frozen=True)
class DistanceConstraint:
    parent: int
    child: int
    rest_length: float
    minimum_ratio: float
    maximum_ratio: float
    flags: int = 0
    stiffness: float = 0
    damping: float = 0
    contact_start: float = 0
    contact_end: float = 1


def _particles(positions):
    p = np.asarray(positions, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 4:
        raise ValueError('Positions must be N x 4')
    return _array(p, p.shape, 'Positions').copy()


def external_force(force_accumulator, uniform_force, weights0, weights1, multiplier):
    """RVA 28F3200, version >= 14 branch. Preserve multiply order."""
    force = _particles(force_accumulator)
    n = len(force)
    uniform = _array(uniform_force, (4,), 'Uniform force')
    w0 = _array(weights0, (n,), 'Force weights 0')
    w1 = _array(weights1, (n,), 'Force weights 1')
    scale = F(multiplier)
    if not np.isfinite(scale):
        raise ValueError('Force multiplier must be finite')
    return force + ((w1[:, None] * uniform) * (w0 * scale)[:, None])


def spring_forces(force_accumulator, positions, history, constraints, reciprocal_dt, *, seeds):
    """Version >=17 dispatcher filter and RVA 28B5A00 spring/damping force."""
    force = _particles(force_accumulator)
    p = _array(positions, force.shape, 'Positions')
    h = _array(history, force.shape, 'History')
    inv = F(reciprocal_dt)
    if not np.isfinite(inv) or inv <= 0:
        raise ValueError('Reciprocal timestep must be finite and positive')
    for r in constraints:
        if r.flags & 0x10 or not r.flags & 4:
            continue
        a, b = r.parent, r.child
        if not (0 <= a < len(p) and 0 <= b < len(p)):
            raise ValueError('Spring endpoints must index existing particles')
        delta = p[b] - p[a]
        length2 = _dot(delta, delta)
        value = np.zeros(4, dtype=np.float32)
        if length2 >= SMALL:
            seed = F(seeds.rsqrt(length2))
            direction = delta * seed
            relative = (p[b] - h[b]) * inv - (p[a] - h[a]) * inv
            scale = F(F(F(F(seed*length2) - F(r.rest_length))*F(r.stiffness))
                      + F(_dot(direction, relative)*F(r.damping)))
            value = direction * scale
        force[a] = force[a] + value
        force[b] = force[b] - value
    return force


def velocity_drag(force_accumulator, positions, history, constraints, weights0,
                  wind_reference, relative_velocity, reciprocal_dt,
                  damping, air_resistance, *, seeds, version=33, mass_coefficient=None):
    """RVA 28F2CA0, versions 17..33. Both endpoints get equal drag.

    Version >=33 caps damping/air per spring at 30 * mean endpoint force
    weight (S+148); 17..32 cap every spring at 30 * S+108, the FLTD config's
    first float (28F4587). The rest of the arithmetic is the same sequence.
    """
    force = _particles(force_accumulator)
    p = _array(positions, force.shape, 'Positions')
    h = _array(history, force.shape, 'History')
    weights = _array(weights0, (len(p),), 'Force weights 0')
    wind = _array(wind_reference, (4,), 'Wind reference')
    relative = _array(relative_velocity, (4,), 'Relative environment velocity')
    inv, damping, air = F(reciprocal_dt), F(damping), F(air_resistance)
    if not np.isfinite([inv, damping, air]).all() or inv <= 0:
        raise ValueError('Drag parameters must be finite with positive reciprocal timestep')
    if damping <= 0 and air <= 0:
        return force
    if not 17 <= int(version) <= 33:
        raise ValueError('Drag is recovered for solver versions 17..33 only')
    uniform_cap = None
    if version < 33:
        if mass_coefficient is None or not np.isfinite(F(mass_coefficient)):
            raise ValueError('Solver versions below 33 need the native S+108 coefficient')
        uniform_cap = F(F(30) * F(mass_coefficient))
    for r in constraints:
        if r.flags & 0x10:
            continue
        a, b = r.parent, r.child
        if not (0 <= a < len(p) and 0 <= b < len(p)):
            raise ValueError('Drag endpoints must index existing particles')
        cap = uniform_cap if uniform_cap is not None else F(F(F(weights[a]+weights[b]) * F(.5)) * F(30))
        d, resistance = min(damping, cap), min(air, cap)
        delta = p[b] - p[a]
        average_velocity = ((p[b]-h[b]) + (p[a]-h[a])) * F(inv * F(.5))
        velocity = (average_velocity-relative) * resistance + (relative-wind) * d
        length2 = _dot(delta, delta)
        if length2 >= SMALL:
            projection = delta * F(_dot(velocity, delta) * F(seeds.rcp(length2)))
            value = (velocity - projection) * F(-.5)
            force[a] = force[a] + value
            force[b] = force[b] + value
    return force


def attach_particles(positions, targets, inverse_mass, basis0, basis1, *, flags, seeds):
    """RVA 28F3640: pin immovable particles and remove forbidden motion.

    A zero axis deliberately leaves a free particle unchanged in the
    single-axis branch. Zero mass pins all four components to the target.
    """
    p = _particles(positions)
    n = len(p)
    t = _array(targets, (n, 4), 'Targets')
    mass = _array(inverse_mass, (n,), 'Inverse mass')
    b0 = _array(basis0, (n, 4), 'Basis 0')
    b1 = _array(basis1, (n, 4), 'Basis 1')
    for i in range(n):
        if mass[i] <= 0 or flags & 0x60 == 0:
            p[i] = t[i]
        elif flags & 0x60 != 0x60:
            axis = b0[i] if not flags & 0x20 else b1[i]
            norm = _dot(axis, axis)
            if norm > 0:
                distance = _dot(p[i] - t[i], axis)
                p[i] = p[i] - (distance * axis) * F(seeds.rcp(norm))
    return p


def pull_targets(positions, targets, weights, dt, *, seeds):
    """RVA 28F3DC0 after the first constraint pass (all float4 lanes)."""
    p = _particles(positions)
    t = _array(targets, p.shape, 'Targets')
    weights = _array(weights, (len(p),), 'Target weights')
    step = F(dt)
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Timestep must be finite and positive')
    for i, weight in enumerate(weights):
        if weight <= 0:
            continue
        factor = F(F(F(F(F(weight * weight) * step) * F(29.999998)) * step) * F(29.999998))
        if factor > 0:
            scale = min(F(1), F(seeds.rcp(F(F(seeds.rcp(factor)) + F(1)))))
            p[i] = p[i] + (t[i] - p[i]) * scale
    return p


def constrain_distances(positions, inverse_mass, constraints, *, seeds,
                        constrain_all=False, flags=0, post_collision=False):
    """One ordered distance pass of 29034A0 / 28FE440.

    The native rule splits a correction equally between movable endpoints;
    it does not distribute it in proportion to their inverse masses.
    This is only the distance section, not either complete function.
    """
    p = _particles(positions)
    n = len(p)
    mass = _array(inverse_mass, (n,), 'Inverse mass')
    constraints = tuple(constraints)
    for record in constraints:
        a, b = record.parent, record.child
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)):
            raise ValueError('Distance endpoint IDs must be integers')
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError('Distance endpoint IDs must identify distinct particles')
        params = np.array([record.rest_length, record.minimum_ratio, record.maximum_ratio], dtype=np.float32)
        if not np.isfinite(params).all() or np.any(params < 0) or params[1] > params[2]:
            raise ValueError('Distance lengths and bounds must be finite and ordered')
    for r in constraints:
        if r.flags & 0x10:
            continue
        a, b = r.parent, r.child
        parent_free = mass[a] > 0
        if post_collision and flags & 0x400 and not r.flags & 0x20:
            parent_free = False
        if not parent_free and mass[b] <= 0:
            continue
        rest, low, high = F(r.rest_length), F(r.minimum_ratio), F(r.maximum_ratio)
        low_length, high_length = F(low*rest), F(high*rest)
        delta = p[b] - p[a]
        length2 = _dot(delta, delta)
        over = F(high_length * high_length) < length2
        force_exact = constrain_all or bool(r.flags & 1)
        if not force_exact and not over and F(low_length * low_length) <= length2:
            continue
        if length2 < SMALL:
            continue
        wanted = rest if force_exact else F((high if over else low) * rest)
        correction = delta - delta * F(F(seeds.rsqrt(length2)) * wanted)
        half = correction * F(.5)
        child_delta = F(0) - half
        parent_delta = half
        if not parent_free:
            child_delta = child_delta * F(2)
            parent_delta = np.zeros(4, dtype=np.float32)
        if mass[b] <= 0:
            parent_delta = parent_delta * F(2)
            child_delta = np.zeros(4, dtype=np.float32)
        p[a] = p[a] + parent_delta
        p[b] = p[b] + child_delta
    return p


def constrain_cone(parent, child, reference, basis0, basis1, parent_mass,
                   child_mass, cosine, sine, offset0, offset1, offset_scale,
                   *, seeds, pin_parent=False):
    """Recovered circular-cone leaf RVA 28F5A50, not angular wrapper 28F70F0.

    The wrapper also handles elliptical limits and mutable basis/state; those
    operations are not silently replaced with this circular case.
    """
    a = _array(parent, (4,), 'Cone parent').copy()
    b = _array(child, (4,), 'Cone child').copy()
    ref = _array(reference, (4,), 'Cone reference')
    b0 = _array(basis0, (4,), 'Cone basis 0')
    b1 = _array(basis1, (4,), 'Cone basis 1')
    params = np.asarray([parent_mass, child_mass, cosine, sine, offset0,
                         offset1, offset_scale], dtype=np.float32)
    if not np.isfinite(params).all():
        raise ValueError('Cone coefficients must be finite')
    pm, cm, cos, sin, off0, off1, scale = params
    if cos <= 0:
        return a, b
    edge = b - a
    edge2 = _dot(edge, edge)
    axis = (b0 * F(scale * off0) + (a - ref)) + b1 * F(scale * off1)
    dot = _dot(axis, edge)
    axis2 = _dot(axis, axis)
    cross = _cross(edge, axis)
    outside = dot < 0 or F(F(F(sin*sin) * axis2) * edge2) < _dot(cross, cross)
    if not outside or edge2 <= 0 or axis2 <= 0:
        return a, b
    projection = F(F(seeds.rcp(axis2)) * dot)
    perpendicular = edge - axis * projection
    perp2 = _dot(perpendicular, perpendicular)
    inv_edge2 = F(seeds.rcp(edge2))
    if perp2 <= 0:
        return a, b
    perp_scale = F(F(seeds.rsqrt(F(perp2 * inv_edge2))) * sin)
    axis_scale = F(F(seeds.rsqrt(F(inv_edge2 * axis2))) * cos)
    desired = axis * axis_scale + perpendicular * perp_scale
    half = (desired - edge) * F(.5)
    change_a = F(0) - half
    change_b = half
    if pin_parent or pm <= 0:
        change_b = change_b * F(2)
        change_a = np.zeros(4, dtype=np.float32)
    if cm <= 0:
        change_a = change_a * F(2)
        change_b = np.zeros(4, dtype=np.float32)
    return a + change_a, b + change_b
