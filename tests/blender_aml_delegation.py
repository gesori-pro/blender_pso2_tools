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

from pso2_tools import (  # noqa: E402
    aqm,
    dotnet,
    export_aqm,
    ice,
    import_model,
    material,
    objects,
    parts,
)

dotnet.load()


class AmlDelegationTests(unittest.TestCase):
    def test_material_split_preserves_sparse_face_groups(self):
        from AquaModelLibrary.Data.PSO2.Aqua import AquaObject
        from AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData import VTXL
        from AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData.Intermediary import (
            GenericTriangles,
        )
        from System import Int32
        from System.Collections.Generic import List
        from System.Numerics import Vector3

        # Leading/interior/trailing empty groups, plus material partitions
        # that cross several source groups while selecting only some faces.
        for groups, selected, expected in [
            ([1, 0, 1], [0, 1], [1, 0, 1]),
            ([0, 0, 2, 0], [0, 1], [0, 0, 2, 0]),
            ([2, 0, 2, 1, 0], [1, 4], [1, 0, 0, 1, 0]),
            ([2, 0, 2, 1, 0], [0, 2, 3], [1, 0, 2, 0, 0]),
        ]:
            with self.subTest(groups=groups, selected=selected):
                model, tris, vertices = AquaObject(), GenericTriangles(), VTXL()
                for group in groups:
                    tris.faceGroups.Add(group)
                for face in range(sum(groups)):
                    tris.triList.Add(Vector3(face * 3, face * 3 + 1, face * 3 + 2))
                    tris.matIdList.Add(0)
                    for corner in range(3):
                        vertices.vertPositions.Add(Vector3(face, corner, 0))
                model.tempTris.Add(tris)
                model.vtxlList.Add(vertices)
                clone = List[Int32]()
                for face in selected:
                    clone.Add(face)
                partitions = List[List[Int32]]()
                partitions.Add(clone)
                model.SplitMeshTempData(0, partitions)
                self.assertEqual(list(model.tempTris[0].faceGroups), expected)
                self.assertEqual(model.tempTris[0].triList.Count, len(selected))
                self.assertEqual(
                    [float(v.X) for v in model.vtxlList[0].vertPositions],
                    [float(face) for face in selected for _ in range(3)],
                )

    def test_shape_adjust_keeps_unflagged_two_frame_layout(self):
        from pso2_tools import shape_sliders

        adjusted = {1: {"pos": (1, 2, 3), "quat": (0, 0, 0, 1), "scale": (2, 3, 4)}}
        motion = shape_sliders.PSO2_OT_ExportShapeAdjust._build_motion(
            SimpleNamespace(NODE_COUNT=2), {0: "root", 1: "hip"}, adjusted
        )
        reread = aqm.parse_aqm(aqm.serialize_aqm(motion))
        for index, node in enumerate(reread.nodes):
            self.assertEqual([key.data_type for key in node.key_sets], [1, 3, 1])
            for key in node.key_sets:
                self.assertEqual(key.timings, [0, 16] if index else [])
        self.assertEqual(reread.nodes[1].key_sets[2].vec4_keys[-1], (2, 3, 4, 0))

    def test_cmx_uses_named_aml_color_fields(self):
        from AquaModelLibrary.Data.PSO2.Aqua.CharacterMakingIndexData import (
            BBLYObject,
            HAIRObject,
            NGS_EarObject,
        )
        from pso2_tools.colors import ColorId

        for obj, attribute, convert in [
            (BBLYObject(), "bbly", objects.CmxColorMapping.from_bodypaint_obj),
            (HAIRObject(), "hair", objects.CmxColorMapping.from_hair_obj),
            (NGS_EarObject(), "ngsEar", objects.CmxColorMapping.from_ear_obj),
        ]:
            data = getattr(obj, attribute)
            mapping = data.maskColorMapping
            mapping.redIndex, mapping.greenIndex = (
                type(mapping.redIndex)(int(ColorId.HAIR1)),
                type(mapping.greenIndex)(int(ColorId.HAIR2)),
            )
            if attribute != "bbly":
                mapping.blueIndex = type(mapping.blueIndex)(0)
                mapping.alphaIndex = type(mapping.alphaIndex)(int(ColorId.MAIN_SKIN))
            data.maskColorMapping = mapping
            setattr(obj, attribute, data)
            actual = convert(obj)
            self.assertEqual(actual.red, ColorId.HAIR1)
            self.assertEqual(actual.green, ColorId.HAIR2)
            self.assertEqual(actual.blue, ColorId.UNUSED)
            self.assertEqual(
                actual.alpha,
                ColorId.UNUSED if attribute == "bbly" else ColorId.MAIN_SKIN,
            )

    def test_material_lookup_matches_export_parser(self):
        expected = material.Material(
            name="body",
            shaders=["1100", "1100"],
            blend_type="opaque",
            special_type="p",
            two_sided=1,
            alpha_cutoff=128,
        )
        for suffix in ("", ".001", "(1)"):
            name = "(1100,1100){opaque}[p]body@1@128" + suffix
            self.assertIs(material.find_material(name, [expected]), expected)
        classic = material.Material(
            name="cloth", shaders=["0100", "0100"], blend_type="blendalpha"
        )
        self.assertIs(
            material.find_material("(0100,0100){blendalpha}cloth", [classic]), classic
        )
        self.assertIsNone(
            material.find_material("Foot_Normal_Corrected_Texture", [expected])
        )
        self.assertIsNone(
            material.find_material("(1103,1103){opaque}[p]body@1@128", [expected])
        )

    def test_missing_mate_keeps_existing_tables(self):
        from AquaModelLibrary.Data.PSO2.Aqua import AquaObject
        from AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData.Intermediary import (
            GenericMaterial,
        )

        source = AquaObject()
        generic = GenericMaterial()
        generic.matName = "source"
        generic.blendType = "opaque"
        source.GenerateMaterial(generic)
        expected = source.mateList[0]
        source.mateList.Clear()
        tables = [
            source.shadList,
            source.rendList,
            source.tsetList,
            source.tstaList,
            source.texfList,
        ]
        snapshots = [list(table) for table in tables]
        self.assertTrue(import_model._add_missing_material(source, "restored"))
        mate = source.mateList[0]
        for member in mate.GetType().GetFields():
            if member.Name != "matName":
                self.assertEqual(
                    member.GetValue(mate), member.GetValue(expected), member.Name
                )
        self.assertEqual(mate.matName.GetString(), "restored")
        self.assertEqual(source.objc.mateCount, 1)
        self.assertFalse(import_model._add_missing_material(source, "ignored"))
        for table, snapshot in zip(tables, snapshots, strict=True):
            self.assertEqual(list(table), snapshot)

    def test_tsta_order_repeats_and_invalid_ids(self):
        from AquaModelLibrary.Data.PSO2.Aqua import AquaObject
        from AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData import MESH, TSET, TSTA
        from System import Activator, Int32
        from System.Collections.Generic import List

        model = AquaObject()
        model.meshList.Add(Activator.CreateInstance(MESH))
        for name, tag in [("diffuse.dds", 23), ("mask.dds", 150)]:
            tsta = Activator.CreateInstance(TSTA)
            tex_name = tsta.texName
            tex_name.SetString(name)
            tsta.texName = tex_name
            tsta.tag = tag
            tsta.modelUVSet = -1
            tsta.unkInt3, tsta.unkInt4, tsta.unkInt5 = 9, 11, 13
            model.tstaList.Add(tsta)
        tset = TSET()
        model.tsetList.Add(tset)
        mapping = List[Int32]()
        mapping.Add(0)
        for indices in [(1, -1, 0, 1), (1, 500, -2, 0, 1)]:
            tset.tstaTexIDs.Clear()
            for index in indices:
                tset.tstaTexIDs.Add(index)
            mats = [material.Material()]
            import_model._attach_tsta_data(model, mapping, mats)
            self.assertEqual([t["tag"] for t in mats[0].tsta_data], [150, 23, 150])
            self.assertEqual(
                [t["name"] for t in mats[0].tsta_data],
                ["mask.dds", "diffuse.dds", "mask.dds"],
            )
            self.assertTrue(
                all(
                    (t["uv"], t["i3"], t["i4"], t["i5"]) == (-1, 9, 11, 13)
                    for t in mats[0].tsta_data
                )
            )
            self.assertEqual(list(tset.tstaTexIDs), list(indices))

    def test_mesh_ornament_id_is_not_face_group(self):
        import bpy

        for tail in ("", "#2"):
            name = "mesh[0]_1_2_3_4#0#3" + tail + ".001"
            self.assertEqual(parts.get_mesh_id(name), parts.MeshId.Ornament1)
            mesh = bpy.data.meshes.new("test")
            obj = bpy.data.objects.new(name, mesh)
            try:
                parts.set_mesh_id(obj, parts.MeshId.Ornament2)
                self.assertEqual(obj.name, "mesh[0]_1_2_3_4#0#8" + tail + ".001")
                self.assertEqual(
                    parts.get_mesh_id(obj.data.name), parts.MeshId.Ornament2
                )
            finally:
                bpy.data.objects.remove(obj)
                bpy.data.meshes.remove(mesh)
        self.assertIsNone(parts.get_mesh_id("mesh[0]_0_0_0_0#0#-1#0"))

    def test_native_face_groups_preserve_channels_and_source(self):
        import numpy as np
        from AquaModelLibrary.Data.PSO2.Aqua import AquaObject
        from AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData import MESH, StripData, VTXL
        from Pso2Tools.Interop import ModelInterop
        from System import Activator, Array, Byte, Int32
        from System.Numerics import Vector2, Vector3, Vector4

        model = AquaObject()
        mesh = Activator.CreateInstance(MESH)
        mesh.baseMeshDummyId = 3
        model.meshList.Add(mesh)
        vertices = VTXL()
        vertices.bonePalette.Add(7)
        for i in range(4):
            vertices.vertPositions.Add(Vector3(i, i + 1, i + 2))
            vertices.uv1List.Add(Vector2(i / 4, i / 8))
            vertices.vertColors.Add(Array[Byte]([i, 2, 3, 255]))
            vertices.trueVertWeights.Add(Vector4(1, 0, 0, 0))
            vertices.trueVertWeightIndices.Add(Array[Int32]([0]))
        model.vtxlList.Add(vertices)
        strip = StripData()
        strip.format0xC31 = True
        for index in [0, 1, 2, 1, 3, 2]:
            strip.triStrips.Add(index)
        for count in [3, 0, 3]:
            strip.faceGroups.Add(count)
        model.strips.Add(strip)
        self.assertEqual(list(ModelInterop.GetFaceGroupIds(model, 0)), [0, 2])
        for group, expected in [(0, [0, 1, 2]), (2, [1, 3, 2])]:
            actual = ModelInterop.GetMesh(model, 0, group)
            positions = np.frombuffer(
                bytes(actual.Positions), dtype=np.float32
            ).reshape(-1, 3)
            self.assertEqual(positions[:, 0].tolist(), expected)
            self.assertEqual(actual.VertexCount, 3)
            self.assertEqual(
                np.frombuffer(bytes(actual.Triangles), dtype=np.int32).tolist(),
                [0, 1, 2],
            )
            colors = np.frombuffer(bytes(actual.Colors), dtype=np.uint8).reshape(-1, 4)
            self.assertEqual(colors[:, 0].tolist(), expected)
            weights = np.frombuffer(bytes(actual.Weights), dtype=np.float32).reshape(
                -1, 4
            )
            self.assertEqual(weights[:, 0].tolist(), [1, 1, 1])
            self.assertEqual(
                np.frombuffer(bytes(actual.BonePalette), dtype=np.int32).tolist(), [7]
            )
            self.assertTrue(actual.MeshName.endswith(f"#0#3#{group}"))
        self.assertEqual(vertices.vertPositions.Count, 4)
        self.assertEqual(list(strip.faceGroups), [3, 0, 3])

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
        for variant, camera, is_material in [
            (0x10002, False, False),
            (0x10012, False, False),
            (0x10004, True, False),
            (0x20, False, True),
        ]:
            motion = aqm.AqmMotion(variant, 0, 0, 30, 0)
            self.assertEqual(motion.is_camera_motion, camera)
            self.assertEqual(motion.is_material_motion, is_material)
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
