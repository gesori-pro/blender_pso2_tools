"""Real FBX/AML roundtrips, including a regression reproduction without the fix."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bpy

sys.path.insert(0, str(Path(__file__).parent))
import blender_export_shape_keys  # noqa: F401
from pso2_tools import export_model, import_model, material


class Report:
    def __init__(self):
        self.messages = []

    def report(self, level, message):
        self.messages.append((level, message))
        print(level, message)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if not mesh.users:
            bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        if not mat.users:
            bpy.data.materials.remove(mat)


def make_scene():
    clear_scene()
    rig_data = bpy.data.armatures.new("Skeleton")
    rig = bpy.data.objects.new("pl_test#1C0#0", rig_data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bone = rig_data.edit_bones.new("Root#1C0#0")
    bone.tail = (0, 1, 0)
    bpy.ops.object.mode_set(mode="OBJECT")
    rig_data.bones[0]["pso2_bone_id"] = 1
    for i, shader in enumerate(("1100", "1103")):
        mesh = bpy.data.meshes.new(f"mesh[{i}]_0_0_0_0#0#0")
        mesh.from_pydata(
            [(i * 2, 0, 0), (i * 2 + 1, 0, 0), (i * 2, 1, 0)], [], [(0, 1, 2)]
        )
        uv = mesh.uv_layers.new(name="UVChannel_1")
        for loop, value in zip(uv.data, ((0, 0), (1, 0), (0, 1)), strict=True):
            loop.uv = value
        for index in (2, 3):
            extra = mesh.uv_layers.new(name=f"UVChannel_{index}")
            for loop in extra.data:
                loop.uv = (0, 0) if index == 2 else (0.25, 0.75)
        obj = bpy.data.objects.new(f"Part {i}", mesh)
        bpy.context.collection.objects.link(obj)
        obj.parent = rig
        obj.vertex_groups.new(name="Root#1C0#0").add([0, 1, 2], 1, "REPLACE")
        obj.modifiers.new("Skeleton", "ARMATURE").object = rig
        obj.select_set(True)
        mat = bpy.data.materials.new(f"({shader}p,{shader}){{opaque}}Garment{i}@1@64")
        mat["pso2_tsta"] = json.dumps(
            [
                {"name": f"garment{i}_d.dds", "tag": 23, "usage": 0, "uv": 0},
                {"name": f"garment{i}_s.dds", "tag": 23, "usage": 1, "uv": 0},
            ]
        )
        mesh.materials.append(mat)


def read_model(path):
    from AquaModelLibrary.Data.PSO2.Aqua import AquaPackage

    return AquaPackage(path.read_bytes()).models[0]


def shader_pairs(model):
    return sorted(
        (str(s.pixelShader.GetString()), str(s.vertexShader.GetString()))
        for s in model.shadList
    )


def texture_names(model):
    return sorted(str(t.texName.GetString()) for t in model.texfList)


class MaterialExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        make_scene()
        self.report = Report()

    def export(self, name="model", **options):
        path = self.root / f"{name}.aqp"
        result = export_model.export(self.report, bpy.context, path, options=options)
        self.assertEqual(result, {"FINISHED"})
        return path

    def reload(self, path):
        clear_scene()
        result, _ = import_model._import_aqp(
            self.report, bpy.context, path, path.with_suffix(".aqn")
        )
        self.assertEqual(result, {"FINISHED"})
        meshes = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        self.assertEqual(len(meshes), 2)
        self.assertEqual(sum(len(o.data.vertices) for o in meshes), 6)
        for obj in meshes:
            self.assertTrue(obj.data.materials[0].get("pso2_material_name"))
        return meshes

    def test_old_plain_name_reproduces_0398(self):
        for mat in bpy.data.materials:
            mat.name = "Foot_Normal_Corrected_Texture"
        # Deliberately bypass only the new metadata adapter: the actual AML
        # converter must reproduce the old failure, not a mocked shader list.
        with patch.object(
            material, "get_export_material_name", side_effect=lambda mat: mat.name
        ):
            model = read_model(self.export("old_behavior"))
        self.assertIn(("0398p", "0398"), shader_pairs(model))

    def test_texture_register_vectors_and_floats_survive_export(self):
        from AquaModelLibrary.Data.PSO2.Aqua import AquaObject

        for n, mat in enumerate(bpy.data.materials):
            mat["pso2_tsta"] = json.dumps(
                [
                    {
                        "name": "shared_d.dds",
                        "tag": 150,
                        "usage": 0,
                        "uv": 0,
                        "i3": 1,
                        "i4": 3,
                        "i5": 1,
                        "vector": [0.125, n + 0.25, 1.0],
                        "floats": [0.125, 0.25, 0.5, 1.0, 2.0],
                    }
                ]
            )
        model = read_model(self.export("registers"))
        vectors = []
        for mesh in model.meshList:
            tex = next(iter(AquaObject.GetTexListTSTAs(model, mesh.tsetIndex)))
            vectors.append((tex.unkVector0.X, tex.unkVector0.Y, tex.unkVector0.Z))
            self.assertEqual(
                [getattr(tex, f"unkFloat{i}") for i in range(5)],
                [0.125, 0.25, 0.5, 1.0, 2.0],
            )
        self.assertEqual(sorted(vectors), [(0.125, 0.25, 1.0), (0.125, 1.25, 1.0)])

    def test_duplicated_legacy_mesh_names_export_separately(self):
        obj = next(obj for obj in bpy.context.scene.objects if obj.type == "MESH")
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.select_set(True)
        before = (obj.name, duplicate.name, obj.data.name, duplicate.data.name)
        result = read_model(self.export("legacy_duplicate"))
        self.assertEqual(result.meshList.Count, 3)
        self.assertEqual(
            (obj.name, duplicate.name, obj.data.name, duplicate.data.name), before
        )

    def test_sparse_face_groups_roundtrip(self):
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        for obj, group in zip(meshes, (0, 2), strict=True):
            obj.name = f"mesh[0]_0_0_0_0#0#3#{group}"
            obj.data.name = obj.name + "_mesh"
        meshes[1].data.materials[0] = meshes[0].data.materials[0]
        first = self.export("face_groups")
        model = read_model(first)
        self.assertEqual(model.meshList.Count, 1)
        self.assertEqual(list(model.strips[0].faceGroups), [3, 0, 3])
        reloaded = self.reload(first)
        self.assertEqual({obj.name.rsplit("#", 1)[1] for obj in reloaded}, {"0", "2"})
        final = read_model(self.export("face_groups_again"))
        self.assertEqual(final.meshList.Count, 1)
        self.assertEqual(list(final.strips[0].faceGroups), [3, 0, 3])

    def test_import_rename_copy_and_roundtrip(self):
        source = self.export("source")
        baseline = read_model(source)
        expected_shaders, expected_textures = (
            shader_pairs(baseline),
            texture_names(baseline),
        )
        meshes = self.reload(source)
        for index, obj in enumerate(meshes):
            copy = obj.data.materials[0].copy()
            copy.name = (
                "Foot_Normal_Corrected_Texture"
                if index == 0
                else "Custom (braces)[x]@name"
            )
            obj.data.materials[0] = copy
        names = [obj.data.materials[0].name for obj in meshes]
        # Existing AQN must survive a real successful export byte for byte.
        target = self.root / "renamed.aqp"
        target.with_suffix(".aqn").write_bytes(source.with_suffix(".aqn").read_bytes())
        protected = target.with_suffix(".aqn").read_bytes()
        output = self.export("renamed")
        result = read_model(output)
        self.assertEqual(shader_pairs(result), expected_shaders)
        self.assertEqual(texture_names(result), expected_textures)
        self.assertEqual([obj.data.materials[0].name for obj in meshes], names)
        self.assertEqual(target.with_suffix(".aqn").read_bytes(), protected)
        self.assertEqual(result.meshList.Count, baseline.meshList.Count)
        for before, after in zip(baseline.vtxlList, result.vtxlList, strict=True):
            for name in ("uv1List", "uv2List", "uv3List", "uv4List"):
                self.assertEqual(
                    [(v.X, v.Y) for v in getattr(before, name)],
                    [(v.X, v.Y) for v in getattr(after, name)],
                    name,
                )
        self.reload(output)

    def test_missing_metadata_cancels_without_touching_files(self):
        path = self.root / "protected.aqp"
        path.write_bytes(b"existing model")
        path.with_suffix(".aqn").write_bytes(b"existing skeleton")
        bpy.data.materials[0].name = "Unknown custom material"
        result = export_model.export(self.report, bpy.context, path)
        self.assertEqual(result, {"CANCELLED"})
        self.assertEqual(path.read_bytes(), b"existing model")
        self.assertEqual(path.with_suffix(".aqn").read_bytes(), b"existing skeleton")
        self.assertIn("no saved PSO2 shader metadata", self.report.messages[-1][1])

    def test_explicit_shader_edit_overrides_saved_name(self):
        mat = bpy.data.materials[0]
        material.remember_material_name(mat)
        mat.name = "(1101p,1101){opaque}Intentional@1@64"
        model = read_model(self.export())
        self.assertIn(("1101p", "1101"), shader_pairs(model))

    def test_export_failure_restores_names(self):
        for mat in bpy.data.materials:
            material.remember_material_name(mat)
            mat.name = "Renamed garment"
        names = [mat.name for mat in bpy.data.materials]
        with (
            patch.object(
                export_model.fbx_wrapper,
                "save",
                side_effect=RuntimeError("test failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "test failure"),
        ):
            export_model.export(self.report, bpy.context, self.root / "fail.aqp")
        self.assertEqual([mat.name for mat in bpy.data.materials], names)


if __name__ == "__main__":
    bpy.ops.preferences.addon_enable(
        module=os.environ.get("PSO2_TEST_ADDON_MODULE", "pso2_tools")
    )
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MaterialExportTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
