"""Stateful restricted test backend for recovered live version-33 stages.

Angular constraints support the observed non-propagating-basis mode, and
observed variants 0/5/7 include prefix/reflection ordering. Animated transform
providers, wrapper initialization/controller inputs and live output parity
remain required; unobserved variants/modes are explicitly rejected.
"""

from dataclasses import dataclass

import numpy as np

from . import native_cloth_stages as stages
from . import native_cloth_angles as angles
from .native_cloth_plane import PlaneCollider5, apply_plane
from .native_cloth_sphere import (EdgeSphereCollider0, InnerSphereCollider1, PointSphereCollider0,
                                  apply_edge_sphere, apply_inner_sphere, apply_point_sphere)
from .native_cloth_capsule import (EdgeCapsuleCollider7, PointCapsuleCollider7,
                                   apply_edge_capsule, apply_point_capsule)

from .native_cloth_segment import (EdgeSegmentCollider2, InnerSegmentCollider3, PointSegmentCollider2,
                                   apply_edge_segment, apply_inner_segment, apply_point_segment)
from .native_cloth_sector import SectorCollider6, apply_edge_sector, apply_point_sector
from .native_cloth_box import BoxCollider8, apply_edge_box, apply_point_box

SUPPORTED_COLLIDERS = (PlaneCollider5, EdgeSphereCollider0, PointSphereCollider0,
                       EdgeCapsuleCollider7, PointCapsuleCollider7, PointSegmentCollider2,
                       EdgeSegmentCollider2, InnerSphereCollider1, InnerSegmentCollider3,
                       SectorCollider6, BoxCollider8)
from .native_cloth_writeback import F, SMALL, _array, _dot
from .native_physics_math import integrate_particles

FULL_SOLVER_IMPLEMENTED = False
DISPATCH_RVA = 0x29048F0


@dataclass(frozen=True)
class SolverState:
    positions: np.ndarray
    previous_positions: np.ndarray
    previous_dt: float
    sleep_mask: int = 0


@dataclass(frozen=True)
class StepResult:
    state: SolverState
    force_scratch: np.ndarray
    sleep_mask: int
    iterations: tuple
    advanced: bool


def _distance_iterations(positions, mass, constraints, flags, iterations,
                         reciprocal_dt, seeds, post_collision, scratch,
                         targets, basis0, basis1, target_constraints, angular_constraints,
                         collision_inputs, solver_version=33):
    count = 0
    out = positions
    for _ in range(iterations):
        scratch = out.copy()
        out = stages.constrain_distances(
            out, mass, constraints, seeds=seeds, constrain_all=not bool(flags & 8),
            flags=flags, post_collision=post_collision)
        out, scratch0, _scratch1 = angles.constrain_angles(
            out, targets, basis0, basis1, mass, target_constraints,
            angular_constraints, seeds=seeds,
            pin_parent=bool(post_collision and flags & 0x400),
            propagate_basis=bool(flags & 0x2000), return_bases=True,
            half_unit_bases=solver_version == 32)
        if post_collision and collision_inputs['colliders']:
            before_collision = out.copy()
            prefix = collision_inputs['prefix']
            def collide(items):
                nonlocal out
                for collider in items:
                    if collider.flags & 0x11 == 1:
                        if isinstance(collider, EdgeSphereCollider0):
                            out = apply_edge_sphere(out, mass, collision_inputs['radii'],
                                                    constraints, collider, seeds=seeds)
                        elif isinstance(collider, EdgeCapsuleCollider7):
                            out = apply_edge_capsule(out, mass, collision_inputs['radii'],
                                                     constraints, collider, seeds=seeds)
                        elif isinstance(collider, PointSphereCollider0):
                            out = apply_point_sphere(out, mass, collision_inputs['radii'], collider,
                                                     collision_inputs['exclusion_mask'], seeds=seeds)
                        elif isinstance(collider, PointCapsuleCollider7):
                            out = apply_point_capsule(out, mass, collision_inputs['radii'], collider,
                                                      collision_inputs['exclusion_mask'], seeds=seeds)
                        elif isinstance(collider, EdgeSegmentCollider2):
                            out = apply_edge_segment(out, mass, collision_inputs['radii'],
                                                     constraints, collider, seeds=seeds)
                        elif isinstance(collider, InnerSphereCollider1):
                            out = apply_inner_sphere(out, mass, collider,
                                                     collision_inputs['exclusion_mask'], seeds=seeds)
                        elif isinstance(collider, InnerSegmentCollider3):
                            out = apply_inner_segment(out, mass, collider,
                                                      collision_inputs['exclusion_mask'], seeds=seeds)
                        elif isinstance(collider, BoxCollider8):
                            if collider.flags & 4:
                                out = apply_edge_box(out, mass, collision_inputs['radii'],
                                                     constraints, collider, seeds=seeds)
                            else:
                                out = apply_point_box(out, mass, collision_inputs['radii'], collider,
                                                      collision_inputs['exclusion_mask'], seeds=seeds)
                        elif isinstance(collider, SectorCollider6):
                            if collider.flags & 4:
                                out = apply_edge_sector(out, mass, constraints, collider, seeds=seeds)
                            else:
                                out = apply_point_sector(out, mass, collider,
                                                         collision_inputs['exclusion_mask'], seeds=seeds)
                        elif isinstance(collider, PointSegmentCollider2):
                            out = apply_point_segment(out, mass, collision_inputs['radii'], collider,
                                                      collision_inputs['exclusion_mask'], seeds=seeds)
                        else:
                            out = apply_plane(out, collision_inputs['history'], mass,
                                          collision_inputs['radii'], collision_inputs['weights0'],
                                          collision_inputs['weights1'], collider,
                                          collision_inputs['sleep_mask'], collision_inputs['exclusion_mask'],
                                          seeds=seeds)
            collide(collision_inputs['colliders'][:prefix])
            if flags & 0x80:
                for i in range(len(out)):
                    nibble = collision_inputs['modes'][i//2] >> (4*(i & 1)) & 0xF
                    if nibble == 1:
                        normal = scratch0[i]
                    elif nibble == 2:
                        normal = F(0)-scratch0[i]
                    elif nibble == 3:
                        # 28FED5D: modes 3/4 read the incoming basis 1, not its
                        # (propagated) scratch copy; modes 1/2 read scratch basis 0.
                        normal = basis1[i]
                    elif nibble == 4:
                        normal = F(0)-basis1[i]
                    else:
                        continue
                    projection = _dot(out[i]-before_collision[i], normal)
                    if projection < 0:
                        out[i] = out[i] - normal * F(projection+projection)
            collide(collision_inputs['colliders'][prefix:])
        count += 1
        # Native stopping rule uses displacement squared * reciprocal dt twice.
        moving = any(F(F(_dot(v, v) * reciprocal_dt) * reciprocal_dt) >= F(1)
                     for v in out - scratch)
        if not moving:
            break
    return out, scratch, count


def step(state, targets, animation_basis0, animation_basis1, inverse_mass,
         constraints, force_weights0, force_weights1, target_weights0,
         target_weights1, uniform_force, dt, *, seeds, flags,
         force_multiplier=1, normal_iterations=5, alternate_iterations=1,
         reciprocal_dt=None, angular_count=0, target_angle_count=0,
         collider_count=0, damping=0, air_resistance=0,
         wind_reference=(0, 0, 0, 0), relative_velocity=(0, 0, 0, 0),
         angular_constraints=(), target_angle_constraints=(), colliders=(),
         collision_radii=None, collision_weights0=None, collision_weights1=None,
         collision_prefix_count=None, packed_normal_modes=None, exclusion_mask=None,
         solver_version=33, mass_coefficient=None):
    """Execute the observed dispatcher order and audited constraint branches.

    No captured final pose is used. Caller supplies explicit evolving history,
    live/FK targets, native timestep and decoded constraint parameters.
    Unknown configurations fail before a state is changed. This is not the
    complete live controller: wrapper initialization and actual environment/
    animated transform inputs must be supplied independently.
    """
    if solver_version not in (31, 32, 33):
        raise ValueError('Only solver versions 31..33 are recovered')
    colliders = tuple(colliders)
    if collider_count != len(colliders):
        raise ValueError('Collider count must match explicitly supplied decoded colliders')
    if any(not isinstance(c, SUPPORTED_COLLIDERS) for c in colliders):
        raise ValueError('Unsupported collider variant; no shape substitution')
    angular_constraints, target_angle_constraints = tuple(angular_constraints), tuple(target_angle_constraints)
    if angular_count != len(angular_constraints) or target_angle_count != len(target_angle_constraints):
        raise ValueError('Angular counts must match the explicitly decoded records')
    # Bit 2 is retained from native configuration; the audited 33 dispatcher
    # has no direct test for it. Basis propagation/unknown modes are rejected.
    # 0x100/0x200 are the mirror bits 28F4440 records; nothing in 29048F0's
    # path reads them.
    if not isinstance(flags, int) or flags < 0 or flags & ~0x3FFF:
        raise ValueError('Unsupported solver flags: complete constraint stages are required')
    p = stages._particles(state.positions)
    n = len(p)
    previous = _array(state.previous_positions, p.shape, 'Previous positions')
    t = _array(targets, p.shape, 'Targets')
    mass = _array(inverse_mass, (n,), 'Inverse mass')
    collision_inputs = dict(colliders=colliders)
    if colliders:
        if not isinstance(collision_prefix_count, int) or not 0 <= collision_prefix_count <= len(colliders):
            raise ValueError('Supply the actual native collider prefix count')
        if exclusion_mask is None:
            raise ValueError('Supply the actual native collider exclusion mask')
        if flags & 0x80 and (packed_normal_modes is None or len(packed_normal_modes) < (n+1)//2):
            raise ValueError('Supply packed native normal modes for collider reflection')
        collision_inputs.update(
            prefix=collision_prefix_count, exclusion_mask=exclusion_mask,
            modes=packed_normal_modes,
            radii=_array(collision_radii, (n,), 'Collision radii'),
            weights0=_array(collision_weights0, (n,), 'Collision weights 0'),
            weights1=_array(collision_weights1, (n,), 'Collision weights 1'),
            sleep_mask=state.sleep_mask)
    old_dt, new_dt = F(state.previous_dt), F(dt)
    if not np.isfinite(old_dt) or old_dt < 0 or not np.isfinite(new_dt) or new_dt < 0:
        raise ValueError('Native timesteps must be finite and nonnegative')
    if new_dt < SMALL:
        raise ValueError('Subthreshold native dispatch needs preserved force/sleep scratch state')
    inv_dt = F(seeds.rcp(new_dt))
    inv_stop = inv_dt if reciprocal_dt is None else F(reciprocal_dt)
    if not np.isfinite(inv_stop) or inv_stop <= 0:
        raise ValueError('Constraint stopping reciprocal must be finite and positive')
    for limit in (normal_iterations, alternate_iterations):
        if not isinstance(limit, int) or limit < 0 or limit > 65535:
            raise ValueError('Native iteration limits must be bounded nonnegative integers')
    constraints = tuple(constraints)
    force = np.zeros_like(p)
    if flags & 1:
        force = stages.spring_forces(force, p, previous, constraints, inv_dt, seeds=seeds)
    force = stages.external_force(force, uniform_force, force_weights0,
                                  force_weights1, force_multiplier)
    if flags & 2:
        force = stages.velocity_drag(force, p, previous, constraints, force_weights0,
                                     wind_reference, relative_velocity, inv_dt,
                                     damping, air_resistance, seeds=seeds,
                                     version=solver_version, mass_coefficient=mass_coefficient)
    integrated = integrate_particles(p, previous, force, mass, old_dt, new_dt,
                                     reciprocal_previous_dt=seeds.rcp(old_dt) if old_dt >= SMALL else None)
    collision_inputs['history'] = integrated.previous_positions
    out = stages.attach_particles(integrated.positions, t, mass, animation_basis0,
                                  animation_basis1, flags=flags, seeds=seeds)
    iterations = (1 if flags & 0x1000 else alternate_iterations if flags & 0x800 else normal_iterations)
    out, scratch, first = _distance_iterations(out, mass, constraints, flags, iterations,
                                               inv_stop, seeds, False, force, t,
                                               animation_basis0, animation_basis1,
                                               target_angle_constraints, angular_constraints,
                                               collision_inputs, solver_version)
    weights = target_weights0 if flags & 0x10 else target_weights1
    out = stages.pull_targets(out, t, weights, new_dt, seeds=seeds)
    out, scratch, second = _distance_iterations(out, mass, constraints, flags, iterations,
                                                inv_stop, seeds, True, scratch, t,
                                                animation_basis0, animation_basis1,
                                                target_angle_constraints, angular_constraints,
                                                collision_inputs, solver_version)
    sleep_mask = 0
    for i, delta in enumerate(integrated.previous_positions - out):
        bit = 1 << (i & 63)
        sleep_mask = (sleep_mask | bit) if _dot(delta, delta) < SMALL else (sleep_mask & ~bit)
    return StepResult(SolverState(out, integrated.previous_positions, new_dt, sleep_mask),
                      scratch, sleep_mask, (first, second), True)
