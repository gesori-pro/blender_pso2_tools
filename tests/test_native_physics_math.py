"""Contract checks; captured-code comparisons live in the analysis dossier."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

_path = Path(__file__).parents[1] / "pso2_tools/native_physics_math.py"
_spec = importlib.util.spec_from_file_location("tested_native_physics_math", _path)
native = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = native
_spec.loader.exec_module(native)


class IntegrationContract(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[1, 2, 3, 1]], dtype=np.float32)
        self.h = self.p - np.float32(.25)
        self.force = np.array([[0, -9.8, 0, 0]], dtype=np.float32)

    def test_does_not_silently_substitute_division(self):
        with self.assertRaisesRegex(ValueError, "RCPPS"):
            native.integrate_particles(self.p, self.h, self.force, [1], 1/60, 1/60)

    def test_zero_mass_does_not_cancel_history_velocity(self):
        result = native.integrate_particles(
            self.p, self.h, self.force, [0], 1, 1, reciprocal_previous_dt=1
        )
        np.testing.assert_array_equal(result.positions, self.p + (self.p - self.h))

    def test_initial_step_does_not_use_uninitialized_history(self):
        a = native.integrate_particles(self.p, self.h, self.force, [1], 0, 1/60)
        b = native.integrate_particles(self.p, self.h * 10, self.force, [1], 0, 1/60)
        np.testing.assert_array_equal(a.positions, b.positions)

    def test_inputs_and_outputs_do_not_alias(self):
        before = self.p.copy()
        result = native.integrate_particles(self.p, self.h, self.force, [1], 0, 1/60)
        result.previous_positions[0, 0] = 100
        result.positions[0, 1] = 100
        np.testing.assert_array_equal(before, self.p)

    def test_requires_float4_arrays(self):
        with self.assertRaisesRegex(ValueError, "N x 4"):
            native.integrate_particles(self.p[:, :3], self.h, self.force, [1], 0, 1/60)

    def test_rejects_nonfinite_state_and_invalid_timestep(self):
        for dt in (0, -1, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                native.integrate_particles(self.p, self.h, self.force, [1], 0, dt)
        with self.assertRaisesRegex(ValueError, "finite"):
            native.integrate_particles(self.p * np.nan, self.h, self.force, [1], 0, 1/60)

    def test_empty_set_has_correct_native_shapes(self):
        empty = np.empty((0, 4), dtype=np.float32)
        result = native.integrate_particles(empty, empty, empty, [], 0, 1/60)
        self.assertEqual(result.positions.shape, (0, 4))
        self.assertFalse(native.FULL_SOLVER_IMPLEMENTED)


if __name__ == "__main__":
    unittest.main()
