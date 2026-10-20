"""Observed model-to-cloth preparation (2908A60/2908E70/28F3750).

Input bones must come from the correct pre-physics model buffer. These
functions do not substitute final simulated bones for that source.
"""

from dataclasses import dataclass, replace

import numpy as np

from .native_cloth_solver33 import SolverState
from .native_cloth_writeback import F, _array, _dot


@dataclass(frozen=True)
class FramePreparation:
    state: SolverState
    targets: np.ndarray
    animation_axes: np.ndarray
    collision_radii: np.ndarray


def _selected(world, mapping):
    matrices = np.asarray(world, dtype=np.float32)
    indices = np.asarray(mapping)
    if (matrices.ndim != 3 or matrices.shape[1:] != (4, 4) or not np.isfinite(matrices).all()
            or indices.ndim != 1 or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= len(matrices))):
        raise ValueError('Cloth preparation requires finite native rows and valid integer node mapping')
    return matrices[indices].copy()


def initial_targets(world, mapping):
    """2908E70 copies four translation lanes into positions and targets."""
    return _selected(world, mapping)[:, 3].copy()


def reference_lengths(constraints, reference_positions, *, seeds):
    """28F3750 derives every edge length from an explicit reference pose."""
    positions = np.asarray(reference_positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 4 or not np.isfinite(positions).all():
        raise ValueError('Cloth reference positions must be finite N x 4')
    result = []
    for record in constraints:
        if not (0 <= record.parent < len(positions) and 0 <= record.child < len(positions)):
            raise ValueError('Reference edge outside the particle array')
        delta = positions[record.parent] - positions[record.child]
        squared = _dot(delta, delta)
        length = F(F(seeds.rsqrt(squared))*squared) if squared > 0 else F(0)
        result.append(replace(record, rest_length=length))
    return tuple(result)


def prepare_frame(state, world, mapping, base_radii, particle_flags, pin_mask,
                  *, seeds):
    """Prepare observed version-33 mode with an empty modifier list.

    Pins move before forces are calculated, as in 2908A60. The kernel's
    later attachment pass does not replace this earlier state update.
    pin_mask includes the caller's actual fixed and dynamic pin masks.
    """
    selected = _selected(world, mapping)
    n = len(selected)
    radii = _array(base_radii, (n,), 'File-side collision radii')
    flags = np.asarray(particle_flags)
    if flags.shape != (n,) or flags.dtype.kind not in 'iu':
        raise ValueError('Particle flags must contain one integer per mapped node')
    if not isinstance(pin_mask, (int, np.integer)) or not 0 <= pin_mask <= (1 << 64)-1:
        raise ValueError('Cloth pin mask must be an unsigned 64-bit integer')
    positions = _array(state.positions, (n, 4), 'Cloth positions').copy()
    previous = _array(state.previous_positions, (n, 4), 'Cloth history').copy()
    targets = selected[:, 3].copy()
    axes = selected[:, :3].transpose(1, 0, 2).copy()
    output_radii = radii.copy()
    for i, matrix in enumerate(selected):
        if pin_mask >> (i & 63) & 1:
            positions[i] = targets[i]
        if flags[i] & 7:
            lengths = np.zeros(3, dtype=np.float32)
            count = 0
            for axis in range(3):
                if int(flags[i]) & (1 << axis):
                    squared = _dot(matrix[axis], matrix[axis])
                    lengths[axis] = F(F(seeds.rsqrt(squared))*squared) if squared > 0 else F(0)
                    count += 1
            scale = F(F(seeds.rcp(F(count))) * F(F(lengths[1]+lengths[0])+lengths[2]))
            output_radii[i] = F(scale*radii[i])
    return FramePreparation(SolverState(positions, previous, state.previous_dt, state.sleep_mask),
                            targets, axes, output_radii)
