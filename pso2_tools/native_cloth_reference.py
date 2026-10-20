"""Reference setup from live 2909A80/2908F20/28F3750/2908E70.

The model generates a separate reference pose (28BED60 -> 54A450). It is
not generally the current animated pose, the AQN inverse-bind, or a final
physics snapshot. Callers must provide that reference explicitly.
"""

import copy
from dataclasses import dataclass

import numpy as np

from .native_cloth_frame import initial_targets, reference_lengths
from .native_cloth_solver33 import SolverState
from .native_cloth_writeback import F, _dot


@dataclass(frozen=True)
class ReferenceInitialization:
    state: SolverState
    solver_arguments: dict
    reference_positions: np.ndarray
    reference_span: np.float32


def initialize_reference(definition, reference_world, current_world, mapping, *, seeds):
    """Build post-pin parameters and an explicit start state, without a dump.

    Constraint lengths use the native estimate, while the controller's
    first-to-last span uses SQRTPS. They must not share a replacement sqrt.
    This composes recovered stages; it does not implement the body's
    reference generator or its resource/parent-link overrides.
    """
    # Type-2 modifiers (2908F20 -> 2979150) act on reference_world itself:
    # the caller applies definition.reference_overrides before this point
    # (FileClothPreview33 does, part by part); other types are rejected at load.
    if definition.modifier_count and not hasattr(definition, 'reference_overrides'):
        raise ValueError('Reference setup requires actual native modifier callbacks')
    reference = initial_targets(reference_world, mapping)
    current = initial_targets(current_world, mapping)
    if len(reference) != definition.particle_count:
        raise ValueError('Reference mapping differs from the file particle count')
    if len(set(np.asarray(mapping).tolist())) != len(reference):
        raise ValueError('Reference mapping must not repeat a model node')
    parameters = copy.deepcopy(definition.solver_arguments)
    parameters['constraints'] = reference_lengths(parameters['constraints'], reference, seeds=seeds)
    delta = reference[0]-reference[-1]
    span = F(np.sqrt(_dot(delta, delta)))
    # The wrapper's first-history synchronization starts at these model
    # positions, rather than retaining an unrelated previous capture.
    state = SolverState(current.copy(), current.copy(), F(0), 0)
    return ReferenceInitialization(state, parameters, reference, span)
