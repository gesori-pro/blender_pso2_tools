"""Live wrapper LOD/iteration controller (RVA 29084C0).

Its returned scalar is a convergence scale, not generally 1/dt. All policy,
observer and state values are explicit inputs; no view or quality is guessed.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_writeback import F, SMALL, _array, _dot


@dataclass(frozen=True)
class ControllerState:
    flags: int
    core_flags: int
    normal_iterations: int
    alternate_iterations: int


@dataclass(frozen=True)
class ControllerPolicy:
    scale: float
    override: float
    minimum_iterations: int
    maximum_iterations: int
    alternate_iterations: int
    version: int


@dataclass(frozen=True)
class ControllerResult:
    state: ControllerState
    convergence_scale: np.float32


def update_controller(state, policy, root, observer, distance_scale, quality,
                      force_iterations, *, seeds):
    root = _array(root, (4,), 'Controller root')
    observer = _array(observer, (4,), 'Controller observer')
    scalars = _array([distance_scale, quality, policy.scale, policy.override],
                     (4,), 'Controller scalars')
    distance_scale, quality, scale, override = scalars
    if not (0 <= state.flags <= 255 and 0 <= state.core_flags <= 65535):
        raise ValueError('Native controller flags must fit their byte/ushort fields')
    if not (0 <= policy.minimum_iterations <= policy.maximum_iterations <= 65535
            and 0 <= policy.alternate_iterations <= 65535):
        raise ValueError('Native controller iteration policy must be ordered and bounded')
    flags, core_flags = int(state.flags), int(state.core_flags)
    normal = int(state.normal_iterations)
    result = F(1000)
    if force_iterations:
        core_flags |= 0x1F
        flags = flags & ~8 | 1
        normal = int(policy.maximum_iterations) * (3 if policy.version < 31 else 1)
    else:
        difference = root-observer
        squared = _dot(difference, difference)
        if squared > F(10000):
            flags &= ~1
            result = F(1)
        else:
            distance = F(F(seeds.rsqrt(squared))*squared) if squared > 0 else F(0)
            scaled_distance = F(distance*distance_scale)
            if abs(scaled_distance) >= SMALL:
                result = F(F(seeds.rcp(scaled_distance))*quality)
                value = override if override >= 0 else F(result*scale)
                if value < F(4):
                    flags &= ~1
                elif value > F(8):
                    flags |= 1
                    if flags & 4:
                        flags |= 8
                        core_flags &= 0xFFE0
                        if flags & 0x10:
                            flags &= ~1
                        normal = 1
                    elif value > F(12):
                        core_flags |= 0x1F
                        flags = flags & ~8 | 1
                        denominator = int(quality)-8
                        if denominator > 0:
                            numerator = ((int(value)-8) *
                                         (int(policy.maximum_iterations)-int(policy.minimum_iterations)))
                            interpolated = (abs(numerator)//denominator) * (-1 if numerator < 0 else 1)
                            interpolated += int(policy.minimum_iterations)
                            normal = min(int(policy.maximum_iterations),
                                         max(int(policy.minimum_iterations), interpolated))
    return ControllerResult(ControllerState(flags, core_flags, normal,
                                            int(policy.alternate_iterations)), result)
