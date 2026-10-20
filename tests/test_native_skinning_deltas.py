"""Local editing must precede game FK, not Blender's inherited world pose."""
import copy
import importlib
from pathlib import Path
import sys
import types
import unittest

import numpy as np

_package = types.ModuleType('_native_skinning_delta_test')
_package.__path__ = [str(Path(__file__).parents[1] / 'pso2_tools')]
sys.modules[_package.__name__] = _package
profile = importlib.import_module(_package.__name__ + '.native_pose_profile')


class NativeLocalDeltaTests(unittest.TestCase):
    def setUp(self):
        trs = np.zeros((2, 3, 4), dtype=np.float32)
        trs[:, 1, 3] = 1
        trs[:, 2, :3] = 1
        trs[1, 0, 1] = 1
        second = trs.copy()
        second[:, 0] = 0
        inv = np.eye(4)
        inv[1, 3] = -1
        self.data = {
            'format': profile.FORMAT, 'version': profile.VERSION,
            'node_names': ['root', 'child'], 'bind_trs': trs.tolist(),
            'node_flags': [0, 0], 'parent_ids': [-1, 0],
            'links': [[-1, -1], [-1, -1]], 'root_world': np.eye(4).tolist(),
            'local_layers': {'first': trs.tolist(), 'second': second.tolist()},
            'palette': ['child'], 'inverse_bind': {'child': inv.tolist()},
            'meshes': [{'positions': [[0, 1, 0]], 'weights_u16': [[65535, 0, 0, 0]],
                        'bone_indices': [[0, 0, 0, 0]], 'triangles': [[0, 0, 0]]}],
        }
        self.delta = second.copy()

    def test_identity_delta_does_not_change_positions_or_input(self):
        original = copy.deepcopy(self.data)
        expected = profile.evaluate_profile(self.data)
        self.data['local_pose_deltas'] = self.delta.tolist()
        actual = profile.evaluate_profile(self.data)
        np.testing.assert_array_equal(actual['meshes'][0]['positions'], expected['meshes'][0]['positions'])
        del self.data['local_pose_deltas']
        self.assertEqual(self.data, original)

    def test_parent_translation_reaches_child_before_skinning(self):
        expected = np.array(profile.evaluate_profile(self.data)['meshes'][0]['positions'])
        self.delta[0, 0, 0] = .125
        self.data['local_pose_deltas'] = self.delta.tolist()
        actual = np.array(profile.evaluate_profile(self.data)['meshes'][0]['positions'])
        np.testing.assert_allclose(actual - expected, [[.125, 0, 0]], atol=1e-7)

    def test_unanimated_tail_uses_bind_without_claiming_physics(self):
        self.data['local_layers']['first'] = self.data['local_layers']['first'][:1]
        self.data['local_layers']['second'] = self.data['local_layers']['second'][:1]
        self.data['bind_tail_fallback'] = True
        self.delta[1, 0, 0] = .125
        self.data['local_pose_deltas'] = self.delta.tolist()
        result = profile.evaluate_profile(self.data)
        np.testing.assert_allclose(result['meshes'][0]['positions'], [[.125, 1, 0]], atol=1e-7)
        self.assertFalse(result['native_physics_implemented'])

    def test_reject_nonfinite_or_wrong_shape_before_fk(self):
        for value in ([[1]], np.full((2, 3, 4), np.nan).tolist()):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite bind TRS'):
                profile.evaluate_profile(dict(self.data, local_pose_deltas=value))

    def test_cloth_runs_before_inverse_bind_and_preserves_source_inputs(self):
        original = copy.deepcopy(self.data)
        seen = []
        def cloth(world, names):
            self.assertEqual(names, ('root', 'child'))
            # Native rows: child translation Y is one BEFORE inverse bind.
            self.assertEqual(world[1, 3, 1], 1)
            world[1, 3, 1] += np.float32(.25)
            seen.append(True)
            return world
        result = profile.evaluate_profile(self.data, cloth_step=cloth)
        np.testing.assert_allclose(result['skin_matrices']['child'][1][3], .25)
        np.testing.assert_allclose(result['meshes'][0]['positions'], [[0, 1.25, 0]])
        self.assertTrue(result['cloth_stage_applied'])
        self.assertFalse(result['native_physics_implemented'])
        self.assertEqual(self.data, original)
        self.assertEqual(len(seen), 1)

    def test_cloth_rejects_nonfinite_or_wrong_shape(self):
        for value in (np.eye(4), np.full((2, 4, 4), np.nan)):
            with self.assertRaisesRegex(ValueError, 'finite native world rows'):
                profile.evaluate_profile(self.data, cloth_step=lambda world, names: value)


if __name__ == '__main__':
    unittest.main()
