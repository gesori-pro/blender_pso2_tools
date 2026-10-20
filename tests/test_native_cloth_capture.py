"""Measured parameter decoding and native-stage guard contracts."""
import hashlib
from dataclasses import replace
import importlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest

import numpy as np

package = types.ModuleType('_measured_cloth_test')
package.__path__ = [str(Path(__file__).parents[1] / 'pso2_tools')]
sys.modules[package.__name__] = package
capture = importlib.import_module(package.__name__ + '.native_cloth_capture')
angles = importlib.import_module(package.__name__ + '.native_cloth_angles')
plane = importlib.import_module(package.__name__ + '.native_cloth_plane')
solver = importlib.import_module(package.__name__ + '.native_cloth_solver33')
drivers = importlib.import_module(package.__name__ + '.native_cloth_driver')
contact = importlib.import_module(package.__name__ + '.native_cloth_contact')


class Estimates:
    # Contract tests only; the independent source-code oracle uses SSE seeds.
    @staticmethod
    def rcp(x):
        return np.float32(1/np.float32(x))
    @staticmethod
    def rsqrt(x):
        return np.float32(1/np.sqrt(np.float32(x)))


class ParameterCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        folder = self.root/'accepted-raw/stationary-instance1/sample-0002'
        folder.mkdir(parents=True)
        self.folder = folder
        core = bytearray(0x210)
        outer = bytearray(0x430)
        struct.pack_into('<4f3i', core, 0x108, 6, .1, 80, 1, 5, 1, 33)
        struct.pack_into('<H', core, 0x144, 0x60)
        struct.pack_into('<I', outer, 0x400, 3)
        struct.pack_into('<I', outer, 0x418, 3)
        blobs = {}
        for offset in [0x60, 0xD8, 0xF0, 0x148, 0x160, 0x178, 0x190, 0x1A8]:
            struct.pack_into('<3Q', core, offset, 0x1000, 0x100C, 0x100C)
            blobs[f'cloth0_core_{offset:03x}.bin'] = np.array([0, 1, 1], '<f4').tobytes()
        distance = bytearray(48)
        struct.pack_into('<2i5f', distance, 8, 0, 1, 0, 0, 1, 0, 1)
        struct.pack_into('<I', distance, 0x2C, 0x12345603)
        struct.pack_into('<3Q', core, 0x78, 0x2000, 0x2030, 0x2030)
        blobs['cloth0_core_078.bin'] = distance
        blobs['cloth0_core_before.bin'] = core
        blobs['cloth0_outer_before.bin'] = outer
        blobs['cloth0_outer_310.bin'] = np.array([0, 1, 2], '<u4').tobytes()
        blobs['cloth0_outer_328.bin'] = np.array([[0, 1, -1, -1, -1]], '<i4').tobytes()
        items = []
        for name, content in blobs.items():
            path = folder/name
            path.write_bytes(content)
            items.append({'path': path.relative_to(self.root).as_posix(), 'bytes': len(content),
                          'sha256': hashlib.sha256(content).hexdigest()})
        (self.root/'artifact-manifest.json').write_text(json.dumps({'items': items}))
        (self.root/'samples.json').write_text(json.dumps({'samples': [dict(
            collection='stationary-instance1', sample=2, accepted=True)]}))

    def load(self, **changes):
        args = dict(instance=1, sample=2, cloth='skirt')
        args.update(changes)
        return capture.load_parameters(self.root, **args)

    def test_loads_parameters_without_requesting_pose_or_executing_helpers(self):
        result = self.load()
        self.assertEqual(result.solver_arguments['constraints'][0].flags, 3)
        self.assertEqual(result.particle_count, 3)
        self.assertFalse(result.evidence['final_pose_imported'])
        self.assertFalse(result.evidence['screen_instance_verified'])
        self.assertFalse(hasattr(result, 'positions'))
        self.assertFalse(result.mapping.flags.writeable)
        result.require_complete_backend()

    def test_sample_and_instance_are_never_guessed(self):
        for args in [dict(instance=0), dict(sample=1), dict(cloth='body')]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.load(**args)

    def test_tampered_scalar_vector_is_rejected(self):
        path = self.folder/'cloth0_core_060.bin'
        path.write_bytes(np.array([1, 2, 3], '<f4').tobytes())
        with self.assertRaisesRegex(ValueError, 'hash/length'):
            self.load()

    def test_collider_factory_requires_explicit_current_and_inverse_transforms(self):
        raw = bytearray(0x110)
        struct.pack_into('<H', raw, 0x90, 0x1007)
        struct.pack_into('<4f', raw, 0xB0, 0, 0, 0, .13)
        config = replace(self.load(), collider_snapshots=(dict(tag=0, raw=raw, vectors={}),))
        with self.assertRaises(ValueError):
            config.make_colliders([])
        with self.assertRaises(KeyError):
            config.make_colliders([{'current': np.eye(4)[None]}])
        with self.assertRaisesRegex(ValueError, 'inverse'):
            config.make_colliders([{'current': np.eye(4)[None], 'inverse': np.eye(4)}])
        matrices = np.eye(4)[None]
        result = config.make_colliders([{'current': matrices, 'inverse': matrices}])[0]
        self.assertEqual(result.flags, 0x1007)
        self.assertAlmostEqual(result.origin_radius[3], .13)
        matrices[:] = 0
        np.testing.assert_array_equal(result.transform, np.eye(4))

    def test_distance_contact_interval_is_not_discarded(self):
        raw = bytearray(48)
        struct.pack_into('<ii', raw, 8, 0, 1)
        struct.pack_into('<5f', raw, 0x18, .2, .9, 1.1, .33, .63)
        record = capture.decode_distance_records(raw, 2)[0]
        self.assertAlmostEqual(record.contact_start, .33)
        self.assertAlmostEqual(record.contact_end, .63)

    def test_padding_is_not_misread_as_angle_or_distance_data(self):
        raw = bytearray(64)
        struct.pack_into('<7fI3iI', raw, 0x10, .9, .4, 0, 0, .6, .8, 1,
                         0x12345602, 0, 1, 2, 0x7FC00000)
        a = capture.decode_angle_records(raw, 3, target=False)[0]
        self.assertEqual((a.reference, a.parent, a.child, a.flags), (0, 1, 2, 2))
        t = capture.decode_angle_records(raw, 3, target=True)[0]
        self.assertEqual((t.parent, t.child, t.reference), (0, 1, -1))

    def test_angle_propagation_turns_the_reference_basis_onto_the_edge(self):
        # 28F78C4: the reference particle's frame, turned so its x axis
        # (basis0 x basis1) follows the parent -> child edge, lands in the
        # parent's slot; the caller's bases are not modified.
        r = angles.AngularConstraint(0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0)
        p = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 1, 0, 1]], np.float32)
        y = np.tile(np.array([0, 1, 0, 0], np.float32), (3, 1))
        z = np.tile(np.array([0, 0, 1, 0], np.float32), (3, 1))
        before = y.copy()
        angles.constrain_angles(p, p, y, z, [1, 1, 1], (), (r,), seeds=Estimates,
                                propagate_basis=True)
        np.testing.assert_array_equal(y, before)
        turned = angles.propagate_basis_frame(y[0], z[0], p[1], p[2], seeds=Estimates)
        x = angles._cross(*turned)
        self.assertGreater(float(x[1]), .99)
        self.assertIsNone(angles.propagate_basis_frame(y[0], z[0], p[1], p[1], seeds=Estimates))

    def test_plane_respects_pins_filter_and_exclusion_mask(self):
        obj = plane.PlaneCollider5(np.eye(4), np.array([0, .05, 0, 1]), 3, 2, 0, 0)
        p = np.array([[0, -1, 0, 1]]*3, dtype=np.float32)
        out = plane.apply_plane(p, p, [0, 1, 1], [.1]*3, [0]*3, [0]*3,
                                obj, 0, 4, seeds=Estimates)
        self.assertEqual(out[0, 1], -1)
        self.assertAlmostEqual(out[1, 1], .15, places=6)
        self.assertEqual(out[2, 1], -1)
        np.testing.assert_array_equal(p[:, 1], [-1]*3)

    def test_kernel_driver_rejects_frozen_collider_and_unknown_variant(self):
        state = solver.SolverState(np.array([[0, 0, 0, 1]], dtype=np.float32),
                                   np.array([[0, 0, 0, 1]], dtype=np.float32), 0)
        obj = plane.PlaneCollider5(np.eye(4), np.array([0, .05, 0, 1]), 3, 0, 0, 0)
        for colliders, message in [([obj], 'animated collider'), ([{'tag': 7}], 'cannot be replaced')]:
            with self.subTest(colliders=colliders), self.assertRaisesRegex(ValueError, message):
                drivers.KernelClothDriver33(['root'], [0], np.empty((0, 5), dtype=np.int32),
                    state, 0, {'flags': 0x4FF, 'colliders': colliders, 'collider_count': 1},
                    {}, seeds=Estimates)

    def test_contact_sphere_strict_boundary_and_degenerate_segment_rule(self):
        self.assertTrue(contact.point_in_sphere([0, 0, 0, 1], [0, 0, 0, 1], np.eye(4), 2))
        self.assertFalse(contact.point_in_sphere([1, 0, 0, 1], [0, 0, 0, 1], np.eye(4), 2))
        self.assertFalse(contact.point_in_tapered_segment([0]*4, [0]*4, [0]*4, 1, 1,
                                                         seeds=Estimates))

    def test_segment_contact_returns_fractions_not_endpoint_coordinates(self):
        s, t = contact.closest_segment_parameters([0, 0, 0, 1], [1, 0, 0, 0],
                                                   [.5, -1, 0, 1], [0, 2, 0, 0], seeds=Estimates)
        np.testing.assert_allclose([s, t], [.5, .5], atol=1e-7)

    def test_fltd_declared_bindings_do_not_invent_runtime_tag_or_solver_version(self):
        data = bytearray(0x200)
        data[:4] = b'NIFL'
        struct.pack_into('<I', data, 12, 32)
        data[32:36] = b'REL0'
        struct.pack_into('<I', data, 40, 32)
        data[64:66] = bytes([10, 1])
        struct.pack_into('<I', data, 68, 64)
        data[0x62] = 1
        struct.pack_into('<II', data, 0x64, 0x150, 0x60)
        data[0x80] = 1  # File kind 1 is deliberately not relabelled runtime tag 1.
        struct.pack_into('<I', data, 0xC4, 0x170)
        struct.pack_into('<I', data, 0xD0, 0x15C)
        struct.pack_into('<I', data, 0x170, 0x160)
        struct.pack_into('<I', data, 0x17C, 0x190)
        for offset, value in [(0x180, b'root\0'), (0x190, b'l_thigh_alt\0'),
                               (0x1B0, b'l_thigh_tw_alt\0')]:
            data[offset:offset+len(value)] = value
        result = capture.read_fltd_collider_bindings(data)
        node = result['groups'][0]['colliders_in_file_order'][0]
        self.assertEqual((node['file_kind'], node['primary_node'], node['secondary_node']),
                         (1, 'l_thigh_alt', 'l_thigh_tw_alt'))
        self.assertEqual(result['file_header_version'], 10)
        self.assertIsNone(result['runtime_solver_version'])
        self.assertFalse(result['runtime_parent_provider_verified'])
        struct.pack_into('<I', data, 0xC4, 0xFFFFFF00)
        with self.assertRaisesRegex(ValueError, 'outside'):
            capture.read_fltd_collider_bindings(data)


if __name__ == '__main__':
    unittest.main()
