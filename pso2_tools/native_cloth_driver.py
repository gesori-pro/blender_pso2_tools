"""Dynamic drivers for the explicitly supported solver 33 branches.

FileClothPreview33 connects supported imports. Other drivers remain explicit.
Observed Schoolu Prima core modes are ported;
animated source/controller correspondence remains required. This connects
audited stages without fixed final-pose input or claiming full live parity.
"""

import copy
import math

import numpy as np

from . import native_cloth_solver33 as solver
from . import native_cloth_writeback as reconstruction


class DistanceClothDriver33:
    full_solver_implemented = False

    def __init__(self, node_names, mapping, records, initial_state, initial_frame,
                 solver_parameters, writeback_parameters, *, seeds):
        self.node_names = tuple(node_names)
        self.mapping = np.asarray(mapping).copy()
        self.records = np.asarray(records).copy()
        self.parameters = copy.deepcopy(solver_parameters)
        self.writeback_parameters = dict(writeback_parameters)
        self.seeds = seeds
        self.revision = 0
        # Reject real, unsupported garment branches at binding, before the
        # caller can accidentally present this as live cloth equivalence.
        self._validate_parameters()
        self.reset(initial_state, initial_frame)

    def _validate_parameters(self):
        if any(self.parameters.get(k, 0) for k in
               ('angular_count', 'target_angle_count', 'collider_count')):
            raise ValueError('Actual garment requires angular and collider stages')
        if self.parameters.get('flags', 0) & ~0x1C7B:
            raise ValueError('Actual garment requires unsupported force/constraint stages')

    def _step_parameters(self, world, names, frame):
        return dict(self.parameters)

    def _prepare_step(self, state, world, targets, axes):
        return state, targets, axes, {}

    def reset(self, state, frame):
        if not math.isfinite(float(frame)):
            raise ValueError('Initial frame must be finite')
        self.state = copy.deepcopy(state)
        self.frame = float(frame)
        self._before = copy.deepcopy(state)
        self._before_frame = float(frame)
        self._fps = None
        self.revision += 1

    def evaluate(self, world_rows, node_names, frame, fps):
        if tuple(node_names) != self.node_names:
            raise ValueError('Cloth node order differs from its decoded AQN mapping')
        world = np.asarray(world_rows, dtype=np.float32)
        if world.shape != (len(self.node_names), 4, 4) or not np.isfinite(world).all():
            raise ValueError('Cloth input must be finite native world rows')
        frame, fps = float(frame), float(fps)
        if not math.isfinite(frame) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cloth frame and FPS must be finite and valid')
        if self._fps is not None and fps != self._fps:
            raise ValueError('Reset cloth explicitly after changing FPS')
        if frame < self.frame or frame > self.frame + 1:
            raise ValueError('Reset/replay cloth explicitly on rewind or skipped frames')
        if self.mapping.ndim != 1 or self.mapping.dtype.kind not in 'iu':
            raise ValueError('Cloth mapping must contain integer node IDs')
        if np.any(self.mapping < 0) or np.any(self.mapping >= len(world)):
            raise ValueError('Cloth mapping exceeds native world rows')
        selected = world[self.mapping]
        targets = selected[:, 3].copy()
        axes = selected[:, :3].transpose(1, 0, 2).copy()
        # Re-evaluating a frame starts from the state BEFORE that frame. This
        # allows pose edits at the same time without accumulating extra steps.
        before = self._before if frame == self.frame else self.state
        before_frame = self._before_frame if frame == self.frame else self.frame
        candidate = before
        if frame > before_frame:
            parameters = self._step_parameters(world, tuple(node_names), frame)
            prepared, targets, axes, overrides = self._prepare_step(before, world, targets, axes)
            parameters.update(overrides)
            step_dt = parameters.pop('dt', (frame - before_frame)/fps)
            active = parameters.pop('_native_controller_active', True)
            if self.writeback_parameters.get('flags', 0) & 0x20:
                active = False      # wrapper bypass: particles follow their targets
            parameters.setdefault('reciprocal_dt', np.float32(fps / (frame - before_frame)))
            if active:
                result = solver.step(prepared, targets, axes[1], axes[2],
                                     dt=step_dt, seeds=self.seeds, **parameters)
                candidate = result.state
            else:
                # 29082C0's disabled-controller path copies both arrays,
                # without advancing the core's previous timestep/sleep mask.
                candidate = solver.SolverState(targets.copy(), targets.copy(),
                                               prepared.previous_dt, prepared.sleep_mask)
        out = reconstruction.writeback(
            world, candidate.positions, targets, axes, self.mapping, self.records,
            seeds=self.seeds, **self.writeback_parameters)
        # Publish state only after the full step AND reconstruction succeed.
        self._before, self._before_frame = before, before_frame
        self.state, self.frame, self._fps = candidate, frame, fps
        return out


class KernelClothDriver33(DistanceClothDriver33):
    """Audited angle and observed variant 0/5/7 collision paths.

    A collider provider is required when collisions are supplied. It receives
    this frame's native FK rows and must return current typed collider objects
    and the actual exclusion mask. Stored transform snapshots are not silently
    treated as an animated bone attachment.

    Optional step_input_provider(world, names, frame) supplies all six explicit
    timing/environment inputs. Without it, scene timing and solver_parameters
    are controlled development inputs, not measured live caller values.
    """
    def __init__(self, *args, collider_provider=None, step_input_provider=None, **kwargs):
        self.collider_provider = collider_provider
        self.step_input_provider = step_input_provider
        super().__init__(*args, **kwargs)

    def _validate_parameters(self):
        for count, records in [('angular_count', 'angular_constraints'),
                               ('target_angle_count', 'target_angle_constraints'),
                               ('collider_count', 'colliders')]:
            if self.parameters.get(count, 0) != len(self.parameters.get(records, ())):
                raise ValueError('Native stage counts must match explicitly decoded inputs')
        if self.parameters.get("flags", 0) & ~0x3FFF:
            raise ValueError('Native basis propagation or unknown mode is not implemented')
        colliders = self.parameters.get('colliders', ())
        if any(not isinstance(c, solver.SUPPORTED_COLLIDERS) for c in colliders):
            raise ValueError('Unknown collider variants cannot be replaced by planes')
        if colliders and not callable(self.collider_provider):
            raise ValueError('An animated collider transform provider is required')

    def _step_parameters(self, world, names, frame):
        parameters = dict(self.parameters)
        if self.step_input_provider is not None:
            inputs = self.step_input_provider(world.copy(), names, frame)
            required = {'dt', 'reciprocal_dt', 'uniform_force', 'wind_reference',
                        'relative_velocity', 'force_multiplier'}
            if (not isinstance(inputs, dict) or not required <= set(inputs)
                    or set(inputs)-required-{'controller_result'}):
                raise ValueError('Step input provider must supply explicit timing and environment inputs')
            inputs = dict(inputs)
            control = inputs.pop('controller_result', None)
            if control is not None:
                from .native_cloth_controller import ControllerResult
                if not isinstance(control, ControllerResult):
                    raise ValueError('Controller inputs require an explicit native ControllerResult')
                parameters.update(flags=control.state.core_flags,
                    normal_iterations=control.state.normal_iterations,
                    alternate_iterations=control.state.alternate_iterations,
                    _native_controller_active=bool(control.state.flags & 1))
            parameters.update(inputs)
        if parameters.get('collider_count', 0):
            inputs = self.collider_provider(world.copy(), names, frame)
            if not isinstance(inputs, dict) or set(inputs) != {'colliders', 'exclusion_mask'}:
                raise ValueError('Collider provider must supply current colliders and exclusion_mask')
            parameters.update(inputs)
        return parameters


class FileClothDriver33(KernelClothDriver33):
    """Connect decoded file pins/radii to the observed pre-force stage.

    This is still an explicit development driver: the caller supplies the
    correct initial/reference model pose, controller/environment inputs and
    animated collider sources. It is not an automatic live-parity claim.
    """
    @classmethod
    def from_reference(cls, node_names, mapping, records, reference_world,
                       current_world, initial_frame, writeback_parameters, *,
                       file_definition, seeds, parameter_overrides=None, **kwargs):
        """Initialize from separate model reference/current poses, not a capture.

        Explicit overrides represent later caller settings (e.g. damping),
        which must not be inferred from a file byte or silently hard-coded.
        """
        from .native_cloth_reference import initialize_reference
        setup = initialize_reference(file_definition, reference_world, current_world,
                                     mapping, seeds=seeds)
        parameters = setup.solver_arguments
        parameters.update(parameter_overrides or {})
        result = cls(node_names, mapping, records, setup.state, initial_frame,
                     parameters, writeback_parameters, file_definition=file_definition,
                     seeds=seeds, **kwargs)
        result.reference_span = setup.reference_span
        return result

    def __init__(self, *args, file_definition, **kwargs):
        # Frame preparation (2908A60) calls only modifiers with kind bit 1 (types
        # 1/3, rejected at load); type 2 already acted on the reference pose.
        if file_definition.modifier_count and not hasattr(file_definition, 'reference_overrides'):
            raise ValueError('Native cloth modifiers require their actual callback implementations')
        if not callable(kwargs.get('step_input_provider')):
            raise ValueError('File cloth requires explicit native timing and environment inputs')
        self.file_definition = copy.deepcopy(file_definition)
        super().__init__(*args, **kwargs)
        if len(self.mapping) != file_definition.particle_count:
            raise ValueError('File cloth particle count differs from the native mapping')

    def _prepare_step(self, state, world, targets, axes):
        from .native_cloth_frame import prepare_frame
        source = self.file_definition
        result = prepare_frame(state, world, self.mapping,
            source.solver_arguments['collision_radii'], source.particle_flags,
            source.skip_mask, seeds=self.seeds)
        return result.state, result.targets, result.animation_axes, {
            'collision_radii': result.collision_radii}
