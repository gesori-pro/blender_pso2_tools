"""Run outside Blender: python -m unittest discover -s tests -p test_skinning_math.py"""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "pso2_tools" / "skinning_math.py"
spec = importlib.util.spec_from_file_location("skinning_math", MODULE)
skin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skin)


class SkinningMathTests(unittest.TestCase):
    def test_native_model_transform_places_head_parts(self):
        body = skin.NATIVE_TO_BLENDER
        np.testing.assert_allclose(skin.native_model_transform(body), np.eye(4), atol=1e-12)
        # A hair rig imported on the head: object rotated and lifted to 1.498.
        hair = np.array([[0, 1, 0, 0], [-0.137, 0, 0.9906, -0.0015],
                         [0.9906, 0, 0.137, 1.4984], [0, 0, 0, 1]])
        model = skin.native_model_transform(hair)
        head3 = np.array([0, 1.5429, 0.0076, 1])  # native AQN position
        expected = body @ head3                   # where the body puts that point
        np.testing.assert_allclose(hair @ model @ head3, expected, atol=1e-9)

    def test_implicit_fourth_weight_and_no_renormalization(self):
        w = skin.gpu_weights([[32767, 16383, 0, 65535], [65535, 65535, 0, 0]])
        self.assertAlmostEqual(float(w[0, 3]), 1 - 49150 / 65535, places=6)
        self.assertEqual(float(w[1, 3]), 0)
        self.assertEqual(float(w[1].sum()), 2)

    def test_palette_indirection_changes_the_selected_transform(self):
        a, b = np.eye(4), np.eye(4)
        a[1, 3] = 0.078529
        b[0, 3] = 1
        out = skin.skin_positions(
            [[0, 1, 0]],
            [[65535, 0, 0, 0]],
            [[1, 0, 0, 0]],
            ["b", "a"],
            {"a": a, "b": b},
        )
        np.testing.assert_allclose(out, [[0, 1.078529, 0]], atol=1e-7)

    def test_fourth_stored_component_is_ignored(self):
        m = {"a": np.eye(4)}
        m["a"][0, 3] = 0.25
        out = skin.skin_positions([[1, 2, 3]], [[0, 0, 0, 0]], [[0, 0, 0, 0]], ["a"], m)
        np.testing.assert_allclose(out, [[1.25, 2, 3]])

    def test_reject_invalid_influence_and_matrix(self):
        with self.assertRaises(ValueError):
            skin.skin_positions(
                [[0, 0, 0]], [[65535, 0, 0, 0]], [[1, 0, 0, 0]], ["a"], {"a": np.eye(4)}
            )
        with self.assertRaises(ValueError):
            skin.skin_positions(
                [[0, 0, 0]],
                [[65535, 0, 0, 0]],
                [[0, 0, 0, 0]],
                ["a"],
                {"a": np.full((4, 4), np.nan)},
            )
        with self.assertRaises(ValueError):
            skin.gpu_weights([[65536, 0, 0, 0]])

    def test_profile_validation_before_scene_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.json"
            path.write_text(json.dumps({"format": skin.CAPTURE_FORMAT, "version": 99}))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                skin.load_capture(path)


if __name__ == "__main__":
    unittest.main()
