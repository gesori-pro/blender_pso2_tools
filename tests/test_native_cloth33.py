"""Temporal/ordering contracts; exact-code numeric oracles are in the dossier."""

import copy
import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np

package = types.ModuleType('_cloth33_test')
package.__path__ = [str(Path(__file__).parents[1] / 'pso2_tools')]
sys.modules[package.__name__] = package
writeback = importlib.import_module(package.__name__ + '.native_cloth_writeback')
stages = importlib.import_module(package.__name__ + '.native_cloth_stages')
solver = importlib.import_module(package.__name__ + '.native_cloth_solver33')
driver_module = importlib.import_module(package.__name__ + '.native_cloth_driver')


class ExactTestEstimates:
    """Contract tests only: not an SSE parity oracle or a production fallback."""
    @staticmethod
    def rcp(value):
        return np.float32(1 / np.float32(value))

    @staticmethod
    def rsqrt(value):
        return np.float32(1 / np.sqrt(np.float32(value)))


class ClothContracts(unittest.TestCase):
    def setUp(self):
        self.world = np.tile(np.eye(4, dtype=np.float32), (3, 1, 1))
        self.world[:, 3, 0] = [0, 1, 2]
        self.pos = self.world[:, 3].copy()
        self.params = dict(
            inverse_mass=[0, 1, 1],
            constraints=[stages.DistanceConstraint(0, 1, 1, 1, 1),
                         stages.DistanceConstraint(1, 2, 1, 1, 1)],
            force_weights0=[1, 1, 1], force_weights1=[1, 1, 1],
            target_weights0=[0, 0, 0], target_weights1=[0, 0, 0],
            uniform_force=[0, -9.8, 0, 0], flags=0x60)
        self.state = solver.SolverState(self.pos, self.pos.copy(), 0)
        self.write_params = dict(version=33, flags=3, fltd_flags=0x60)

    def driver(self, params=None):
        return driver_module.DistanceClothDriver33(
            ['a', 'b', 'c'], [0, 1, 2], [[0, 1, -1, -1, -1], [1, 2, 0, -1, -1]],
            self.state, 0, params or self.params, self.write_params,
            seeds=ExactTestEstimates)

    def test_duplicate_frame_evaluation_does_not_integrate_twice(self):
        driver = self.driver()
        first = driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        state = copy.deepcopy(driver.state)
        second = driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(state.positions, driver.state.positions)
        next_pose = driver.evaluate(self.world, ['a', 'b', 'c'], 2, 60)
        self.assertGreater(float(abs(next_pose - first).max()), 0)

    def test_edit_same_frame_restarts_from_before_that_frame(self):
        driver = self.driver()
        driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        moved = self.world.copy()
        moved[:, 3, 1] += .2
        result = driver.evaluate(moved, ['a', 'b', 'c'], 1, 60)
        fresh = self.driver().evaluate(moved, ['a', 'b', 'c'], 1, 60)
        np.testing.assert_array_equal(result, fresh)
        self.assertAlmostEqual(result[0, 3, 1], .2)

    def test_jump_rewind_fps_change_do_not_mutate_state(self):
        driver = self.driver()
        driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        prior = driver.state.positions.copy()
        for frame, fps in [(3, 60), (0, 60), (2, 30)]:
            with self.assertRaisesRegex(ValueError, 'Reset|reset'):
                driver.evaluate(self.world, ['a', 'b', 'c'], frame, fps)
            np.testing.assert_array_equal(prior, driver.state.positions)
        driver.reset(self.state, 20)
        driver.evaluate(self.world, ['a', 'b', 'c'], 21, 30)

    def test_real_garment_is_not_silently_approximated(self):
        for field, value in [('angular_count', 24), ('target_angle_count', 12),
                             ('collider_count', 5), ('flags', 0x4FF)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.driver(dict(self.params, **{field: value}))
        self.assertFalse(solver.FULL_SOLVER_IMPLEMENTED)

    def test_kernel_uses_explicit_caller_timing_and_rejects_partial_environment(self):
        inputs = dict(dt=np.float32(1/30), reciprocal_dt=np.float32(30),
                      uniform_force=[0, -1, 0, 0], wind_reference=[0]*4,
                      relative_velocity=[0]*4, force_multiplier=1)
        def create(provider):
            return driver_module.KernelClothDriver33(
                ['a', 'b', 'c'], [0, 1, 2], [[0, 1, -1, -1, -1], [1, 2, 0, -1, -1]],
                self.state, 0, self.params, self.write_params,
                seeds=ExactTestEstimates, step_input_provider=provider)
        driver = create(lambda *_: inputs)
        result = driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        self.assertEqual(driver.state.previous_dt, np.float32(1/30))
        np.testing.assert_array_equal(result, driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60))
        bad = create(lambda *_: {'dt': 1/30})
        with self.assertRaisesRegex(ValueError, 'explicit timing'):
            bad.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        self.assertEqual(bad.frame, 0)
        np.testing.assert_array_equal(bad.state.positions, self.pos)

    def test_writeback_forces_homogeneous_one_even_on_degenerate_basis(self):
        for zero in (False, True):
            parent = self.world[0].copy()
            if zero:
                parent[0] = 0
            pos = self.pos[1].copy()
            pos[3] = .95
            a, b = writeback.reconstruct_edge(self.pos[0], pos, parent[:3], parent,
                                              self.world[1], self.pos[0], self.pos[1],
                                              seeds=ExactTestEstimates)
            self.assertEqual(b[3, 3], 1)
            np.testing.assert_array_equal(b[3, :3], pos[:3])

    def test_unmapped_identity_hemroot_is_preserved(self):
        world = np.concatenate((self.world, np.eye(4, dtype=np.float32)[None]))
        result = writeback.writeback(
            world, self.pos, self.pos, self.world[:, :3].transpose(1, 0, 2),
            [0, 1, 2], [[0, 1, -1, -1, -1], [1, 2, 0, -1, -1]],
            seeds=ExactTestEstimates, **self.write_params)
        np.testing.assert_array_equal(result[3], np.eye(4))
        np.testing.assert_array_equal(world[3], np.eye(4))

    def test_pin_copies_four_lanes_and_distance_uses_equal_split(self):
        p = self.pos.copy()
        p[1, 0] = 3
        actual = stages.constrain_distances(
            p, [1, .01, 0], [stages.DistanceConstraint(0, 1, 1, 1, 1)],
            seeds=ExactTestEstimates, constrain_all=True)
        np.testing.assert_array_equal(actual[:2, 0], [1, 2])
        pinned = stages.attach_particles(p, self.pos, [0, 1, 1],
                                          self.world[:, 1], self.world[:, 2],
                                          flags=0x60, seeds=ExactTestEstimates)
        np.testing.assert_array_equal(pinned[0], self.pos[0])

    def test_bad_mapping_and_forward_record_are_rejected_before_publication(self):
        driver = self.driver()
        driver.mapping[0] = 2
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            driver.evaluate(self.world, ['a', 'b', 'c'], 1, 60)
        self.assertEqual(driver.frame, 0)
        np.testing.assert_array_equal(driver.state.positions, self.pos)

    def test_circular_cone_moves_free_child_without_moving_pinned_parent(self):
        parent = np.array([0, 0, 0, 1], dtype=np.float32)
        child = np.array([0, 1, 0, 1], dtype=np.float32)
        ref = np.array([-1, 0, 0, 1], dtype=np.float32)
        a, b = stages.constrain_cone(
            parent, child, ref, [0]*4, [0]*4, 0, 1,
            np.sqrt(.75), .5, 0, 0, 1, seeds=ExactTestEstimates)
        np.testing.assert_array_equal(a, parent)
        np.testing.assert_allclose(b[:3], [np.sqrt(.75), .5, 0], atol=1e-7)
        np.testing.assert_array_equal(child, [0, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
