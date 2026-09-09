"""Run in factory-startup Blender with the bundled AML binaries available."""

import os
import site
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(os.environ.get("PSO2_TEST_ADDON_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
if dependencies := os.environ.get("PSO2_TEST_DEPENDENCIES"):
    site.addsitedir(dependencies)

from pso2_tools import aqm, dotnet, export_aqm, ice, objects  # noqa: E402

dotnet.load()


class AmlDelegationTests(unittest.TestCase):
    def test_variant_compatibility(self):
        for name, expected in {
            "VARIANT_STD_ANIM": 0x10002,
            "VARIANT_PLAYER_ANIM": 0x10012,
            "VARIANT_CAMERA_ANIM": 0x10004,
            "VARIANT_MATERIAL_ANIM": 0x20,
        }.items():
            self.assertEqual(getattr(aqm, name), expected)
        with self.assertRaises(AttributeError):
            _ = aqm.nonexistent_constant

    def test_motion_classification(self):
        for variant, camera, material in [
            (0x10002, False, False),
            (0x10012, False, False),
            (0x10004, True, False),
            (0x20, False, True),
        ]:
            motion = aqm.AqmMotion(variant, 0, 0, 30, 0)
            self.assertEqual(motion.is_camera_motion, camera)
            self.assertEqual(motion.is_material_motion, material)
            self.assertFalse(motion.is_shape_adjust)

    def test_timing_boundary_and_key_types(self):
        self.assertEqual(aqm.baked_timing_format(4095), (16, 0))
        self.assertEqual(aqm.baked_timing_format(4096), (256, 128))
        for key_type, data_type in [(1, 1), (2, 3), (3, 1), (16, 5), (17, 5)]:
            self.assertEqual(aqm.key_data_type(key_type), data_type)
            for flag, multiplier in [(0, 16), (128, 256)]:
                key = aqm.AqmKeySet(
                    key_type, data_type | flag, 0, timings=[1, 3 * multiplier + 2]
                )
                self.assertEqual(key.frames(), [0, 3])

    def test_aqm_readback_at_both_timing_widths(self):
        for end in [1, 4095, 4096]:
            multiplier, flag = aqm.baked_timing_format(end)
            node = aqm.AqmNode(2, 0, "root")
            for key_type, vectors in [
                (1, [(0, 0, 0, 0), (1, 2, 3, 0)]),
                (2, [(0, 0, 0, 1), (0, 0, 0, 1)]),
                (3, [(1, 1, 1, 0), (1, 1, 1, 0)]),
            ]:
                node.key_sets.append(
                    aqm.AqmKeySet(
                        key_type,
                        aqm.key_data_type(key_type) | flag,
                        0,
                        timings=[1, end * multiplier + 2],
                        vec4_keys=vectors,
                    )
                )
            motion = aqm.AqmMotion(aqm.VARIANT_STD_ANIM, 0, end, 30, 1, [node])
            reread = aqm.parse_aqm(aqm.serialize_aqm(motion))
            self.assertEqual(reread.end_frame, end)
            for key, expected in zip(
                reread.nodes[0].key_sets, node.key_sets, strict=True
            ):
                self.assertEqual(key.frames(), [0, end])
                self.assertEqual(key.data_type, expected.data_type)
                self.assertEqual(key.vec4_keys, expected.vec4_keys)

    def test_player_tree_flags_keep_existing_channels(self):
        for flag, multiplier in [(0, 16), (128, 256)]:
            node = export_aqm._make_node_tree_flag(2, 3, multiplier, flag)
            self.assertEqual([k.key_type for k in node.key_sets], [16, 17])
            self.assertTrue(all(k.data_type == (5 | flag) for k in node.key_sets))
            self.assertEqual(node.key_sets[0].frames(), [0, 1, 2])

    def test_face_mapping_orientation_and_empty_crop(self):
        source = b"""language = "face_a"
crop_name = "facevar100070"
language = "unused"
crop_name = ""
language = "face_b"
crop_name = "facevar200090"
\x00\x00"""
        actual = objects._parse_face_variation_lua(SimpleNamespace(data=source))
        self.assertEqual(actual, {"face_a": 100070, "face_b": 200090})

    @unittest.skipUnless(
        os.environ.get("PSO2_TEST_ICE"), "Set PSO2_TEST_ICE for asset test"
    )
    def test_real_archive_payloads_match_original_entries(self):
        import struct

        from System.IO import FileMode, FileStream
        from Zamboni import IceFile

        path = os.environ["PSO2_TEST_ICE"]
        stream = FileStream(path, FileMode.Open)
        try:
            archive = IceFile.LoadIceFile(stream)
            expected = {}
            for group in [archive.groupOneFiles, archive.groupTwoFiles]:
                for entry in group:
                    raw = bytes(entry)
                    header = struct.unpack_from("<i", raw, 12)[0]
                    expected[str(IceFile.getFileName(entry))] = raw[header:]
        finally:
            stream.Close()
        actual = {
            entry.name: entry.data for entry in ice.IceFile.load(path).get_files()
        }
        self.assertGreater(len(actual), 0)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(AmlDelegationTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
