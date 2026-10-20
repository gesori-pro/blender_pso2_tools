"""Reference/source-bank contracts; isolated native oracles are separate."""

from dataclasses import replace
import importlib
from pathlib import Path
import struct
import sys
import types
import unittest

import numpy as np

package = types.ModuleType('_cloth_reference_test')
package.__path__ = [str(Path(__file__).parents[1]/'pso2_tools')]
sys.modules[package.__name__] = package
reference = importlib.import_module(package.__name__+'.native_cloth_reference')
bindings = importlib.import_module(package.__name__+'.native_cloth_bindings')
initialization = importlib.import_module(package.__name__+'.native_cloth_initialization')
stages = importlib.import_module(package.__name__+'.native_cloth_stages')
fk = importlib.import_module(package.__name__+'.native_skeleton_math')
drivers = importlib.import_module(package.__name__+'.native_cloth_driver')


class Estimates:
    rcp = staticmethod(lambda x: np.float32(1/np.float32(x)))
    rsqrt = staticmethod(lambda x: np.float32(1/np.sqrt(np.float32(x))))


def definition():
    parameters = dict(constraints=(stages.DistanceConstraint(0, 1, 0, 1, 1),),
        inverse_mass=[0, 1], force_weights0=[1]*2, force_weights1=[1]*2,
        target_weights0=[0]*2, target_weights1=[0]*2, flags=0x60)
    return initialization.FileClothDefinition(0, 2, parameters, (0, 0), {},
        1, 3, 2, 5, 1, 0, np.float32(1), 'explicit-test-fixture')


class ReferenceContracts(unittest.TestCase):
    def test_reference_lengths_do_not_follow_current_animated_pose(self):
        base = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
        base[1, 3, 0] = 2
        current = base.copy()
        current[1, 3, 0] = 3
        original = definition()
        setup = reference.initialize_reference(original, base, current, [0, 1], seeds=Estimates)
        self.assertEqual(setup.reference_span, 2)
        self.assertEqual(setup.solver_arguments['constraints'][0].rest_length, 2)
        self.assertEqual(original.solver_arguments['constraints'][0].rest_length, 0)
        np.testing.assert_array_equal(setup.state.positions, current[:, 3])
        np.testing.assert_array_equal(setup.state.previous_positions, current[:, 3])
        with self.assertRaisesRegex(ValueError, 'repeat'):
            reference.initialize_reference(original, base, current, [0, 0], seeds=Estimates)

    def test_file_driver_factory_takes_no_final_pose_or_captured_history(self):
        source = definition()
        base = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
        base[1, 3, 0] = 2
        driver = drivers.FileClothDriver33.from_reference(
            ['a', 'b'], [0, 1], [[0, 1, -1, -1, -1]], base, base, 0,
            dict(version=33, flags=3, fltd_flags=0x60), file_definition=source,
            seeds=Estimates, step_input_provider=lambda *_: None)
        self.assertEqual(driver.reference_span, 2)
        self.assertEqual(driver.parameters['constraints'][0].rest_length, 2)
        # Decoded type-2 modifiers act on the reference pose the caller passes.
        setup = reference.initialize_reference(replace(source, modifier_count=1, reference_overrides=()),
                                               base, base, [0, 1], seeds=Estimates)
        self.assertEqual(setup.solver_arguments['constraints'][0].rest_length, 2)

    def test_binding_records_retain_bank_and_slot_and_reject_bad_vectors(self):
        raw = struct.pack('<QIIII', 1234, 5, 2, 1, 0)
        row, = bindings.read_binding_records(raw)
        self.assertEqual((row.node_index, row.source_bank, row.slot), (5, 2, 1))
        for data in [raw[:-1], struct.pack('<QIIII', 1234, 5, 10, 1, 0)]:
            with self.assertRaises(ValueError):
                bindings.read_binding_records(data)

    def test_binding_prepares_actual_bank_with_scale_blend_without_mutating_inputs(self):
        matrices = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
        shape = bindings.BoundShapeState(matrices.copy(), matrices.copy(),
                                         np.zeros(2, 'f4'), 7, .1, 1)
        source = matrices.copy()
        source[1, :3, :3] *= 2
        source[1, 3, 0] = 4
        row = bindings.ColliderBoneBinding(1234, 1, 2, 0)
        out = bindings.prepare_bound_shapes([row], {2: source}, {1234: shape}, seeds=Estimates)[1234]
        self.assertEqual(out.transforms[0, 0, 0], np.float32(1.1))
        self.assertEqual(out.transforms[0, 3, 0], 4)
        self.assertEqual(out.inverse_transforms[0, 3, 3], 1)
        np.testing.assert_array_equal(shape.transforms, matrices)
        np.testing.assert_array_equal(out.transforms[1], matrices[1])
        with self.assertRaisesRegex(ValueError, 'source bone bank'):
            bindings.prepare_bound_shapes([row], {0: source}, {1234: shape}, seeds=Estimates)

    def test_live_alt_joint_write_occurs_before_later_joints_are_evaluated(self):
        bind = np.zeros((4, 3, 4), 'f4')
        bind[:, 1, 3] = 1
        bind[:, 2, :3] = 1
        bind[1, 0, 0], bind[2, 0, 0], bind[3, 0, 1] = 1, 3, 2
        # Joint 2 writes joint 1; joint 3 then reads that updated parent.
        flags = [0x1c0, 0x1c0, 0x20001c0, 0x1c0]
        parents = [-1, 0, 0, 1]
        links = [[-1, -1], [-1, -1], [1, -1], [-1, -1]]
        out = fk.evaluate_world_pose(flags, parents, bind, bind, links=links, live_runtime=True)
        np.testing.assert_array_equal(out[1], out[2])
        np.testing.assert_array_equal(out[3, 3, :3], [3, 2, 0])
        with self.assertRaisesRegex(ValueError, 'flag'):
            fk.evaluate_world_pose(flags, parents, bind, bind, links=links)
        # Native bind-only pass does not perform this linked-output write.
        out = fk.evaluate_world_pose(flags, parents, bind, links=links, live_runtime=True)
        np.testing.assert_array_equal(out[1, 3, :3], [1, 0, 0])


if __name__ == '__main__':
    unittest.main()
