"""Recovered live PSO2 solver arithmetic, independent of Blender.

Live DLL source SHA256:
1131296fa373f2eadd47715759b1653cb5c60427e9da6090b9e39c016674fe16.
Captured .text SHA256:
7a979c0ff137d10a131be7c441a1a7b1e7251569d6c28d72be6cb79010bd6319.

This module implements the version-33 integrator. Related native_cloth_*
modules now supply recovered force/attachment/distance/target stages and
bone reconstruction, with an explicit restricted temporal test driver.
Character import does not enable that driver: full garment collision dispatch,
initialization and complete live dynamics remain required. Observed angular
limits are implemented separately in native_cloth_angles.
Unicorn 2.1.4 AVX arithmetic is not the validation oracle. The captured math
was executed in a separate x64 process with a Win64 ABI bridge. In 108
synthetic cases, 13,284 float32 components matched, using the actual host's
RCPPS seed. That is not a live-pose, Rosetta or GPU-output comparison.
"""

from dataclasses import dataclass

import numpy as np

SOURCE_SHA256 = "1131296fa373f2eadd47715759b1653cb5c60427e9da6090b9e39c016674fe16"
INTEGRATOR_RVA = 0x2907570
FULL_SOLVER_IMPLEMENTED = False
_F = np.float32
_MIN_PREVIOUS_DT = _F(2**-16)


@dataclass(frozen=True)
class IntegratedParticles:
    """Native float4 arrays. No Blender axis or unit conversion is applied."""

    positions: np.ndarray
    previous_positions: np.ndarray
    displacement: np.ndarray
    previous_dt: np.float32


def integrate_particles(
    positions,
    previous_positions,
    force_accumulator,
    inverse_mass,
    previous_dt,
    dt,
    *,
    reciprocal_previous_dt=None,
):
    """Evaluate the captured solver-33 integration leaf, in float32 order.

    For previous_dt >= 2^-16, reciprocal_previous_dt must supply the actual
    RCPPS low-lane result for that timestep. Exact division is not an
    interchangeable substitute and is not silently used as a fallback.

    Inverse mass zero removes acceleration here, but does not remove
    previous-frame velocity. Attachment is a separate native solver stage.
    The caller remains responsible for the dispatcher's timestep gate.
    Inputs are not mutated. Returned history is a copy of the input pose.
    """
    current = np.asarray(positions, dtype=np.float32)
    history = np.asarray(previous_positions, dtype=np.float32)
    force = np.asarray(force_accumulator, dtype=np.float32)
    mass = np.asarray(inverse_mass, dtype=np.float32)
    if current.ndim != 2 or current.shape[1] != 4:
        raise ValueError("Particle positions must be an N x 4 native float32 array")
    if history.shape != current.shape or force.shape != current.shape:
        raise ValueError("History and force must match the particle positions")
    if mass.shape != (len(current),):
        raise ValueError("Inverse mass must contain one scalar per particle")
    if not all(np.isfinite(x).all() for x in (current, history, force, mass)):
        raise ValueError("Particle inputs must be finite")
    if np.any(mass < 0):
        raise ValueError("Inverse mass cannot be negative")
    old_dt, new_dt = _F(previous_dt), _F(dt)
    if not np.isfinite(old_dt) or old_dt < 0:
        raise ValueError("Previous timestep must be finite and nonnegative")
    if not np.isfinite(new_dt) or new_dt <= 0:
        raise ValueError("Timestep must be finite and positive")

    factor = _F(_F(_F(old_dt + new_dt) * _F(.5)) * new_dt)
    displacement = force * (factor * mass)[:, None]
    if old_dt >= _MIN_PREVIOUS_DT:
        if reciprocal_previous_dt is None:
            raise ValueError("The native RCPPS seed is required for the previous timestep")
        seed = _F(reciprocal_previous_dt)
        if not np.isfinite(seed) or seed <= 0:
            raise ValueError("The native reciprocal seed must be finite and positive")
        ratio = _F(seed * new_dt)
        displacement = displacement + (current - history) * ratio
    result = current + displacement
    if not np.isfinite(result).all() or not np.isfinite(displacement).all():
        raise ValueError("Particle integration overflowed float32")
    return IntegratedParticles(result, current.copy(), displacement, new_dt)
