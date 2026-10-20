"""File/metadata and wrapper lifecycle contracts; native oracles are separate."""

import importlib
from pathlib import Path
import struct
import sys
import types
import unittest

import numpy as np

package = types.ModuleType('_cloth_initialization_test')
package.__path__ = [str(Path(__file__).parents[1]/'pso2_tools')]
sys.modules[package.__name__] = package
initialization = importlib.import_module(package.__name__+'.native_cloth_initialization')
assets = importlib.import_module(package.__name__+'.cloth_assets')
controller = importlib.import_module(package.__name__+'.native_cloth_controller')
transforms = importlib.import_module(package.__name__+'.native_cloth_transforms')
drivers = importlib.import_module(package.__name__+'.native_cloth_driver')
solver = importlib.import_module(package.__name__+'.native_cloth_solver33')
stages = importlib.import_module(package.__name__+'.native_cloth_stages')
environment = importlib.import_module(package.__name__+'.native_cloth_environment')


def source_fixture():
    data = bytearray(0x500)
    data[:4] = b'NIFL'
    struct.pack_into('<I', data, 12, 32)
    data[32:36] = b'REL0'
    struct.pack_into('<I', data, 40, 32)
    data[64:68] = bytes([10, 1, 1, 0])
    struct.pack_into('<II', data, 68, 0x100-32, 0x200-32)
    data[0x100:0x104] = bytes([0, 1, 0, 0])
    struct.pack_into('<I', data, 0x104, 16)
    data[0x200:0x208] = bytes([0x1F, 2, 0, 0, 1, 2, 5, 0])
    struct.pack_into('<II', data, 0x208, 0x458-32, 0x300-32)
    struct.pack_into('<I', data, 0x214, 1)
    struct.pack_into('<3f', data, 0x300, 6, .1, 80)
    data[0x30C:0x30F] = bytes([33, 1, 1])
    struct.pack_into('<I', data, 0x314, 0x440-32)
    struct.pack_into('<II', data, 0x32C, 0x448-32, 0x450-32)
    struct.pack_into('<I', data, 0x338, 0x380-32)
    struct.pack_into('<I', data, 0x380, 0x400-32)
    struct.pack_into('<5f', data, 0x400, 6, -9.8, .1, .2, .3)
    struct.pack_into('<5f', data, 0x420, 12, -4.9, .2, 1, 1)
    struct.pack_into('<2f', data, 0x448, 0, .2)
    struct.pack_into('<2f', data, 0x450, 1, .5)
    return data


class Estimates:
    rcp = staticmethod(lambda x: np.float32(1/np.float32(x)))
    rsqrt = staticmethod(lambda x: np.float32(1/np.sqrt(np.float32(x))))


class InitializationContracts(unittest.TestCase):
    def test_pins_are_applied_after_file_mass_and_weights_are_preserved(self):
        part, = initialization.read_file_definitions(source_fixture())
        self.assertEqual((part.particle_count, part.skip_mask), (2, 1))
        args = part.solver_arguments
        np.testing.assert_array_equal(args['inverse_mass'], np.array([0, 1/12], 'f4'))
        np.testing.assert_array_equal(args['force_weights0'], [6, 12])
        np.testing.assert_array_equal(args['force_weights1'], [1, .5])
        self.assertFalse(hasattr(part, 'final_matrices'))
        self.assertNotIn('uniform_force', args)
        self.assertNotIn('colliders', args)

    def test_file_pointer_and_unknown_solver_version_are_rejected(self):
        data = source_fixture()
        struct.pack_into('<I', data, 0x32C, 0xFFFFFF00)
        with self.assertRaisesRegex(ValueError, 'outside'):
            initialization.read_file_definitions(data)
        data = source_fixture()
        data[0x30C] = 30
        with self.assertRaisesRegex(ValueError, 'versions 31..33'):
            initialization.read_file_definitions(data)

    def test_version_32_keeps_angle_flag_bit_1(self):
        data = source_fixture()
        base = initialization.read_file_definitions(data)[0].solver_arguments
        data[0x30C] = 32
        args = initialization.read_file_definitions(data)[0].solver_arguments
        self.assertEqual(args['solver_version'], 32)
        for key in ('angular_constraints', 'target_angle_constraints'):
            for old, new in zip(base[key], args[key]):
                self.assertEqual(new.flags, old.flags)

    def test_version_31_shares_the_33_path_with_its_own_drag_and_angle_flags(self):
        data = source_fixture()
        base = initialization.read_file_definitions(data)[0].solver_arguments
        data[0x30C] = 31
        args = initialization.read_file_definitions(data)[0].solver_arguments
        self.assertEqual((base['solver_version'], args['solver_version']), (33, 31))
        self.assertEqual(args['mass_coefficient'], base['mass_coefficient'])
        # 28F4921/28F4DE5: angle record flag bit 1 needs solver version 32.
        for key in ('angular_constraints', 'target_angle_constraints'):
            for old, new in zip(base[key], args[key]):
                self.assertEqual(new.flags, old.flags & 1)

    def test_stored_asset_keeps_source_and_rejects_tampering_without_enabling_physics(self):
        data = source_fixture()
        payload = assets.capture_file(data, 'fixture.fltd')
        self.assertTrue(payload['initialization_decoded'])
        self.assertFalse(payload['live_dynamic_parity_verified'])
        self.assertEqual(assets.restore_file(payload), data)
        payload['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            assets.restore_file(payload)

    def test_controller_convergence_value_is_independent_of_scene_fps(self):
        state = controller.ControllerState(0x13, 0x4FF, 5, 1)
        policy = controller.ControllerPolicy(.3, -1, 5, 6, 1, 33)
        result = controller.update_controller(state, policy, [0, 0, 0, 1],
                       [0, 0, 3, 1], .78, 1242, True, seeds=Estimates)
        self.assertEqual(result.convergence_scale, 1000)
        self.assertEqual(result.state.normal_iterations, 6)

    def test_collider_scale_blend_and_once_per_shape_distance_are_distinct(self):
        previous = np.eye(4, dtype='f4')
        incoming = np.eye(4, dtype='f4')
        incoming[:3, :3] *= 2
        incoming[3, 0] = 3
        result = transforms.prepare_transform(previous, incoming, 7, .1, 1, seeds=Estimates)
        np.testing.assert_array_equal(result.matrix[:3, :3], np.eye(3, dtype='f4')*np.float32(1.1))
        self.assertEqual(result.movement_distance, 3)
        again = transforms.prepare_transform(previous, incoming, result.flags, .1, 1, seeds=Estimates)
        self.assertIsNone(again.movement_distance)
        np.testing.assert_array_equal(result.matrix, again.matrix)

    def test_file_driver_moves_pins_before_neighbor_force_calculation(self):
        definition, = initialization.read_file_definitions(source_fixture())
        world = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
        world[1, 3, 0] = 1
        positions = world[:, 3].copy()
        state = solver.SolverState(positions, positions.copy(), 0)
        world[0, 3, 0] = .2
        params = dict(definition.solver_arguments, flags=0x61,
            constraints=[stages.DistanceConstraint(0, 1, 1, 0, 2, flags=4, stiffness=3)],
            target_weights0=[0]*2, target_weights1=[0]*2,
            uniform_force=[0]*4, normal_iterations=0)
        inputs = lambda *_: dict(dt=1/60, reciprocal_dt=1000,
            uniform_force=[0]*4, wind_reference=[0]*4,
            relative_velocity=[0]*4, force_multiplier=1)
        def build(driver_type, **kwargs):
            return driver_type(['a', 'b'], [0, 1], [[0, 1, -1, -1, -1]],
                state, 0, params, dict(version=33, flags=3, fltd_flags=0x60),
                seeds=Estimates, step_input_provider=inputs, **kwargs)
        native = build(drivers.FileClothDriver33, file_definition=definition)
        diagnostic = build(drivers.KernelClothDriver33)
        native.evaluate(world, ['a', 'b'], 1, 60)
        diagnostic.evaluate(world, ['a', 'b'], 1, 60)
        self.assertGreater(float(abs(native.state.positions[1]-diagnostic.state.positions[1]).max()), 1e-6)
        first = native.state.positions.copy()
        native.evaluate(world, ['a', 'b'], 1, 60)
        np.testing.assert_array_equal(native.state.positions, first)
        np.testing.assert_array_equal(state.positions, positions)

    def test_disabled_controller_copies_targets_and_history_without_advancing_dt(self):
        definition, = initialization.read_file_definitions(source_fixture())
        world = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
        positions = np.array([[0, 0, 0, 1], [1, 0, 0, 1]], dtype='f4')
        world[:, 3] = positions
        world[:, 3, 1] += .25
        control = controller.ControllerResult(controller.ControllerState(0, 0x4FF, 3, 1), np.float32(1000))
        params = dict(definition.solver_arguments, uniform_force=[0]*4)
        driver = drivers.FileClothDriver33(['a', 'b'], [0, 1], [[0, 1, -1, -1, -1]],
            solver.SolverState(positions, positions.copy(), .01, 5), 0, params,
            dict(version=33, flags=3, fltd_flags=0x60), seeds=Estimates,
            file_definition=definition, step_input_provider=lambda *_: dict(
                dt=1/60, reciprocal_dt=1000, uniform_force=[0]*4,
                wind_reference=[0]*4, relative_velocity=[0]*4, force_multiplier=1,
                controller_result=control))
        driver.evaluate(world, ['a', 'b'], 1, 60)
        np.testing.assert_array_equal(driver.state.positions, world[:, 3])
        np.testing.assert_array_equal(driver.state.previous_positions, world[:, 3])
        self.assertEqual(driver.state.previous_dt, .01)
        self.assertEqual(driver.state.sleep_mask, 5)

    def test_negative_wind_phase_truncates_instead_of_floor_wrapping(self):
        wave = environment.EnvironmentWave(np.array([1, 0, 0, 1], 'f4'),
                                            np.array([1, 0, 1, 0], 'f4'))
        result = environment.evaluate_wave(wave, [.25, 0, 0, 1], 0)
        np.testing.assert_array_equal(result, [-.25, 0, 0, -.25])


if __name__ == '__main__':
    unittest.main()
