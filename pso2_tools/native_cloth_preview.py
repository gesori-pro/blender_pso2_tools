"""File-driven, stateful FLTD preview with explicit scene timing and gravity.

This selects the recovered solver, not a claimed native game environment or
LOD controller. It regenerates shapes from evolving FK, never from a dump.
Rewinds/jumps/FPS changes reinitialize and settle at the requested pose; they
do not fabricate animation samples for frames Blender has not evaluated.
"""
import copy
import math

import numpy as np

from .native_cloth_driver import FileClothDriver33


class FileClothPreview33:
    input_policy = 'SCENE'
    live_dynamic_parity_verified = False

    def __init__(self, names, parts, reference_world, *, settle_steps=30, seeds, related_worlds=None):
        # Source banks 1..9 (parent/grandparent/sibling resources) for colliders
        # bound outside this model; scene inputs, held fixed in model space.
        self.related_worlds = {int(k): np.asarray(v, dtype='f4').copy()
                               for k, v in (related_worlds or {}).items()}
        self.node_names = tuple(names)
        self.parts = tuple(parts)
        if not self.parts:
            raise ValueError('FLTD has no cloth parts to preview')
        self.reference = np.asarray(reference_world, dtype='f4').copy()
        if self.reference.shape != (len(names), 4, 4) or not np.isfinite(self.reference).all():
            raise ValueError('Cloth reference must match finite native world matrices')
        # 2908F20 runs each part's type-2 modifiers on the shared reference
        # model before its rest lengths: the overrides accumulate part by part.
        self.part_references, current = [], self.reference
        for part in self.parts:
            current = self._override_reference(current, part)
            self.part_references.append(current)
        if not isinstance(settle_steps, int) or not 0 <= settle_steps <= 120:
            raise ValueError('Cloth settle steps must be within 0..120')
        self.settle_steps, self.seeds = settle_steps, seeds
        self.drivers = None
        self.revision = 1
        self.last_reset_reason = 'initialization'
        self.context(np.eye(4, dtype='f4'), np.array([0, -9.81, 0, 0], 'f4'))

    def context(self, model_to_simulation, gravity):
        transform = np.asarray(model_to_simulation, dtype='f4')
        force = np.asarray(gravity, dtype='f4')
        if (transform.shape != (4, 4) or force.shape != (4,)
                or not np.isfinite(transform).all() or not np.isfinite(force).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])):
            raise ValueError('Cloth scene transform and gravity must be finite affine inputs')
        self.model_to_simulation = transform.copy()
        self.simulation_to_model = np.linalg.inv(transform).astype('f4')
        self.gravity = force.copy()
        self.context_metric = transform[:3,:3].T.astype('f8') @ transform[:3,:3].astype('f8')
        self.context_handedness = np.sign(np.linalg.det(transform[:3,:3]))

    def _inputs(self, multiplier):
        def provide(_world, _names, _frame):
            dt = np.float32(self.step_dt)
            return dict(dt=dt, reciprocal_dt=np.float32(np.float32(1) / dt * multiplier),
                uniform_force=self.gravity.copy(), wind_reference=np.zeros(4, 'f4'),
                relative_velocity=np.zeros(4, 'f4'), force_multiplier=np.float32(1))
        return provide

    def _build(self, references, current, frame):
        drivers = []
        for part, reference in zip(self.parts, references):
            def colliders(world, _names, _frame, source=part):
                return dict(colliders=tuple(c.evaluate(world, seeds=self.seeds, related_worlds=self.related_worlds)
                                            for c in source.colliders),
                    exclusion_mask=source.definition.basis_masks.get('exclusion_mask', 0))
            initial = colliders(current, self.node_names, frame)
            definition = part.definition
            writeback = dict(version=definition.solver_arguments.get('solver_version', 33),
                flags=definition.wrapper_flags,
                fltd_flags=definition.solver_arguments['flags'], skip_mask=definition.skip_mask, **{
                    k: v for k, v in definition.basis_masks.items() if k != 'exclusion_mask'})
            driver = FileClothDriver33.from_reference(self.node_names, part.mapping, part.records,
                reference, current, frame, writeback, file_definition=definition,
                parameter_overrides=dict(colliders=initial['colliders'],
                    collider_count=len(initial['colliders']),
                    collision_prefix_count=part.collision_prefix_count,
                    exclusion_mask=initial['exclusion_mask']),
                seeds=self.seeds, collider_provider=colliders,
                step_input_provider=self._inputs(definition.convergence_multiplier))
            drivers.append(driver)
        return drivers

    def _override_reference(self, reference, part):
        """2979150 -> node +0x20 (5494F0) -> 54C250: the particle's bone gets
        the matrix M (rotation rows from the half-angle quaternion, position
        row with w = 1) in the model's per-node override bank, which the pose
        pass (54C690) folds in. Read here as a turn in the bone's own frame,
        M @ local: every file entry has position 0 and one small angle on a
        bone whose AQN local already carries an offset and a large rotation,
        so a plain replacement would fold it onto its parent. Bones below it
        keep their reference local transforms.""" 
        overrides = getattr(part.definition, 'reference_overrides', ())
        if not overrides:
            return reference
        from .native_cloth_box import rotation_rows
        parents = part.parents
        if not parents:
            raise ValueError('Reference overrides need the model hierarchy')
        out = reference.astype('f8')
        local = {i: out[i] @ np.linalg.inv(out[p]) for i, p in enumerate(parents) if p >= 0}
        turned = {}
        for particle, position, angles in overrides:
            node = int(part.mapping[particle])
            matrix = rotation_rows(np.asarray(angles, 'f4')).astype('f8')
            matrix[3] = (*position, 1)
            turned[node] = matrix @ turned.get(node, np.eye(4))
        moved = set()
        for i, parent in enumerate(parents):     # AQN parents precede children
            if i in turned:
                out[i] = turned[i] @ (local[i] @ out[parent] if parent >= 0 else out[i])
                moved.add(i)
            elif parent in moved:
                out[i] = local[i] @ out[parent]
                moved.add(i)
        return out.astype('f4')

    def _repose(self, world, pose, written):
        """54C690 stand-in: bones the cloth wrote keep their matrices, and
        every other bone below one follows it with its pre-physics local
        transform (native rows: world = local @ parent world)."""
        parents = next((p.parents for p in self.parts if p.parents), ())
        out = pose.copy()
        moved = set(written)
        for i, parent in enumerate(parents):
            if i in written or parent < 0 or parent not in moved:
                continue
            local = world[i].astype('f8') @ np.linalg.inv(world[parent].astype('f8'))
            out[i] = (local @ out[parent].astype('f8')).astype('f4')
            moved.add(i)
        return out

    def _evaluate(self, drivers, world, frame, fps):
        out = world.copy()
        written, reposed = set(), False
        for part, driver in zip(self.parts, drivers):
            # Parts read the pre-physics FK (manager 28C2880) unless their group
            # follows the earlier parts (wrapper 0x8000, 28C2A5E): then the
            # model is re-posed once per frame and later writes show through.
            source = world
            if part.follows_earlier:
                if not reposed:
                    out = self._repose(world, out, written)
                    reposed = True
                source = out
            changed = driver.evaluate(source, self.node_names, frame, fps)
            out[driver.mapping] = changed[driver.mapping]
            written.update(int(i) for i in driver.mapping)
        if not np.isfinite(out).all():
            raise ValueError('Cloth generated non-finite output')
        return out

    def evaluate(self, world_rows, names, frame, fps):
        world = np.asarray(world_rows, dtype='f4')
        if tuple(names) != self.node_names or world.shape != self.reference.shape or not np.isfinite(world).all():
            raise ValueError('Cloth input no longer matches its source skeleton')
        frame, fps = float(frame), float(fps)
        if not math.isfinite(frame) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cloth needs finite frame and positive FPS')
        simulation = np.matmul(self.model_to_simulation, world.transpose(0, 2, 1)).transpose(0, 2, 1).copy()
        reset = self.drivers is None
        reason = 'initialization'
        if not reset:
            if fps != self.fps:
                reset, reason = True, 'FPS changed'
            elif (not np.allclose(self.context_metric,self.last_context_metric,rtol=1e-5,atol=1e-7)
                    or self.context_handedness != self.last_context_handedness):
                reset, reason = True, 'actor scale changed'
            elif frame < self.frame or frame > self.frame + 1:
                reset, reason = True, 'timeline jump'
            elif frame == self.frame and self.settled_at_frame:
                reset = not np.array_equal(simulation, self.last_world) or not np.array_equal(self.gravity, self.last_gravity)
                reason = 'initial pose edited'
        if not reset and frame == self.frame and np.array_equal(simulation, self.last_world) and np.array_equal(self.gravity, self.last_gravity):
            result = self.last_output
        elif reset:
            self.step_dt = 1 / fps
            references = [np.matmul(self.model_to_simulation, r.transpose(0, 2, 1)).transpose(0, 2, 1).copy()
                          for r in self.part_references]
            start = frame - self.settle_steps
            candidate = self._build(references, simulation, start)
            result = self._evaluate(candidate, simulation, start, fps)
            for i in range(1, self.settle_steps + 1):
                result = self._evaluate(candidate, simulation, start + i, fps)
            self.drivers = candidate
            self.last_reset_reason = reason
            self.settled_at_frame = True
        else:
            before = self.drivers[0]._before_frame if frame == self.frame else self.frame
            self.step_dt = (frame - before) / fps
            # Transactional across parts: a later failing collider must not
            # leave an earlier garment advanced to a different frame.
            fields = ('state', 'frame', '_before', '_before_frame', '_fps', 'revision')
            snapshots = [{k: copy.deepcopy(getattr(d, k)) for k in fields} for d in self.drivers]
            try:
                result = self._evaluate(self.drivers, simulation, frame, fps)
            except Exception:
                for driver, snapshot in zip(self.drivers, snapshots):
                    for key, value in snapshot.items():
                        setattr(driver, key, value)
                raise
            self.settled_at_frame = False
        self.last_context_metric = self.context_metric.copy()
        self.last_context_handedness = self.context_handedness
        self.frame, self.fps = frame, fps
        self.last_world, self.last_gravity, self.last_output = simulation.copy(), self.gravity.copy(), result.copy()
        return np.matmul(self.simulation_to_model, result.transpose(0, 2, 1)).transpose(0, 2, 1).copy()
