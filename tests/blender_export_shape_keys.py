"""Run with Blender --background --factory-startup --python this_file.

Requires the add-on's dependencies in Blender's extension site-packages.
All objects and output files are synthetic; no user scene is opened.
"""

import json
import os
import site
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bpy

ROOT = Path(os.environ.get("PSO2_TEST_ADDON_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
site.addsitedir(
    str(
        Path(bpy.utils.user_resource("EXTENSIONS"))
        / ".local/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
)
if dependencies := os.environ.get("PSO2_TEST_DEPENDENCIES"):
    site.addsitedir(dependencies)

from pso2_tools import export_model, export_shape_keys  # noqa: E402


def coordinates(points):
    return [tuple(point.co) for point in points]


def state(obj):
    mesh = obj.data
    keys = mesh.shape_keys
    return {
        "mesh": mesh.as_pointer(),
        "name": mesh.name,
        "vertices": coordinates(mesh.vertices),
        "keys": [
            (k.name, k.value, k.mute, k.vertex_group, coordinates(k.data))
            for k in keys.key_blocks
        ],
        "key_id": keys.as_pointer(),
        "action": str(keys.animation_data.action) if keys.animation_data else None,
        "index": obj.active_shape_key_index,
        "pin": obj.show_only_shape_key,
        "selected": obj.select_get(),
        "hidden": obj.hide_get(),
        "uv": [tuple(loop.uv) for loop in mesh.uv_layers.active.data],
        "weights": [[(g.group, g.weight) for g in v.groups] for v in mesh.vertices],
        "modifiers": [(m.name, m.type, m.show_viewport) for m in obj.modifiers],
        "custom_normals": mesh.has_custom_normals,
        "normals": [tuple(normal.vector) for normal in mesh.corner_normals],
        "mesh_count": len(bpy.data.meshes),
        "key_count": len(bpy.data.shape_keys),
        "object_count": len(bpy.data.objects),
    }


class ShapeKeyExportTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        mesh = bpy.data.meshes.new("mesh[0]_0_0_0_0#0#0")
        mesh.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
        mesh.uv_layers.new()
        self.obj = bpy.data.objects.new("Test garment", mesh)
        bpy.context.collection.objects.link(self.obj)
        self.obj.select_set(True)
        bpy.context.view_layer.objects.active = self.obj
        self.obj.shape_key_add(name="Basis")
        key = self.obj.shape_key_add(name="Arbitrary key")
        key.data[1].co.z = 2
        key.value = 0.35
        self.obj.active_shape_key_index = 1
        bpy.context.view_layer.update()

    def assertCoordinates(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            for x, y in zip(a, b, strict=True):
                self.assertAlmostEqual(x, y, places=5)

    def evaluated(self, obj=None):
        bpy.context.view_layer.update()
        return coordinates(
            (obj or self.obj)
            .evaluated_get(bpy.context.evaluated_depsgraph_get())
            .data.vertices
        )

    def check_mix(self):
        expected = self.evaluated()
        before = state(self.obj)
        with export_shape_keys.applied(bpy.context, [self.obj]):
            self.assertIsNone(self.obj.data.shape_keys)
            self.assertCoordinates(coordinates(self.obj.data.vertices), expected)
        self.assertEqual(state(self.obj), before)

    def test_partial_relative_mix_and_muted_key(self):
        other = self.obj.shape_key_add(name="Second key")
        other.data[0].co.y = 2
        other.value = 0.8
        muted = self.obj.shape_key_add(name="Muted key")
        muted.data[0].co.x = 10
        muted.value = 1
        muted.mute = True
        self.check_mix()

    def test_mask_and_relative_reference(self):
        mask = self.obj.vertex_groups.new(name="Mask")
        mask.add([1], 0.25, "REPLACE")
        self.obj.data.shape_keys.key_blocks[1].vertex_group = mask.name
        other = self.obj.shape_key_add(name="Relative to another key")
        other.relative_key = self.obj.data.shape_keys.key_blocks[1]
        other.data[2].co.x = 0.6
        other.value = 0.5
        self.check_mix()

    def test_negative_value(self):
        key = self.obj.data.shape_keys.key_blocks[1]
        key.slider_min = -2
        key.value = -0.5
        self.check_mix()

    def test_absolute_keys(self):
        keys = self.obj.data.shape_keys
        keys.use_relative = False
        keys.key_blocks[0].interpolation = "KEY_LINEAR"
        keys.key_blocks[1].interpolation = "KEY_LINEAR"
        keys.eval_time = 4
        self.check_mix()

    def test_pinned_active_key(self):
        self.obj.show_only_shape_key = True
        self.check_mix()

    def test_edit_mode(self):
        original_keys = self.obj.data.shape_keys.as_pointer()
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            with export_shape_keys.applied(bpy.context, [self.obj]):
                self.assertIsNone(self.obj.data.shape_keys)
                self.assertEqual(self.obj.mode, "OBJECT")
            self.assertEqual(self.obj.mode, "EDIT")
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
        self.assertEqual(self.obj.data.shape_keys.as_pointer(), original_keys)

    def test_custom_normals_and_color_attributes(self):
        mesh = self.obj.data
        mesh.normals_split_custom_set([(0, 0, 1)] * len(mesh.loops))
        colors = mesh.color_attributes.new("Color", "FLOAT_COLOR", "CORNER")
        for loop in colors.data:
            loop.color = (0.1, 0.3, 0.5, 1)
        before = state(self.obj)
        with export_shape_keys.applied(bpy.context, [self.obj]):
            self.assertTrue(self.obj.data.has_custom_normals)
            self.assertEqual(
                [tuple(c.color) for c in self.obj.data.color_attributes["Color"].data],
                [tuple(c.color) for c in colors.data],
            )
            self.assertEqual(
                [tuple(loop.uv) for loop in self.obj.data.uv_layers.active.data],
                before["uv"],
            )
        self.assertEqual(state(self.obj), before)

    def test_keyframe(self):
        key = self.obj.data.shape_keys.key_blocks[1]
        key.value = 0
        key.keyframe_insert("value", frame=1)
        key.value = 1
        key.keyframe_insert("value", frame=9)
        bpy.context.scene.frame_set(5)
        self.check_mix()

    def test_driver(self):
        curve = self.obj.data.shape_keys.key_blocks[1].driver_add("value")
        curve.driver.expression = "0.625"
        self.check_mix()

    def test_shared_mesh_and_failure_restore(self):
        other = self.obj.copy()
        bpy.context.collection.objects.link(other)
        other.show_only_shape_key = True
        expected = [self.evaluated(o) for o in (self.obj, other)]
        before = [state(o) for o in (self.obj, other)]
        with (
            self.assertRaisesRegex(RuntimeError, "export failed"),
            export_shape_keys.applied(bpy.context, [self.obj, other]),
        ):
            for obj, coords in zip((self.obj, other), expected, strict=True):
                self.assertCoordinates(coordinates(obj.data.vertices), coords)
            raise RuntimeError("export failed")
        self.assertEqual([state(o) for o in (self.obj, other)], before)
        self.assertIs(self.obj.data, other.data)

    def test_modifier_runs_after_shape_mix(self):
        mod = self.obj.modifiers.new("Thickness", "SOLIDIFY")
        mod.thickness = 0.2
        expected = self.evaluated()
        before = state(self.obj)
        with export_shape_keys.applied(bpy.context, [self.obj]):
            self.assertEqual(len(self.obj.modifiers), 1)
            self.assertCoordinates(self.evaluated(), expected)
        self.assertEqual(state(self.obj), before)

    def test_selection_visibility_and_collections(self):
        other = self.obj.copy()
        collection = bpy.data.collections.new("Other collection")
        bpy.context.scene.collection.children.link(collection)
        collection.objects.link(other)
        other.select_set(False)
        self.assertEqual(
            export_model._get_export_meshes(bpy.context, {"use_selection": True}),
            [self.obj],
        )
        self.assertEqual(
            export_model._get_export_meshes(
                bpy.context, {"collection": collection.name}
            ),
            [other],
        )
        self.assertEqual(
            export_model._get_export_meshes(bpy.context, {"collection": "Missing"}), []
        )
        other.hide_set(True)
        self.assertEqual(
            export_model._get_export_meshes(bpy.context, {"use_visible": True}),
            [self.obj],
        )

    def test_fbx_failure_restores_source(self):
        from pso2_tools import dotnet

        dotnet.load()
        before = state(self.obj)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                export_model.fbx_wrapper, "save", side_effect=RuntimeError("FBX failed")
            ),
            self.assertRaisesRegex(RuntimeError, "FBX failed"),
        ):
            export_model.export(None, bpy.context, Path(directory) / "shape.aqp")
        self.assertEqual(state(self.obj), before)

    def test_aqp_contains_mix_with_modifiers_on_and_off(self):
        from pso2_tools import dotnet

        dotnet.load()
        from AquaModelLibrary.Data.PSO2.Aqua import AquaPackage

        rig_data = bpy.data.armatures.new("Skeleton")
        rig = bpy.data.objects.new("pl_test#1C0#0", rig_data)
        bpy.context.collection.objects.link(rig)
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bone = rig_data.edit_bones.new("Root#1C0#0")
        bone.head = (0, 0, 0)
        bone.tail = (0, 1, 0)
        bpy.ops.object.mode_set(mode="OBJECT")
        rig_data.bones[0]["pso2_bone_id"] = 1
        group = self.obj.vertex_groups.new(name=rig_data.bones[0].name)
        group.add([0, 1, 2], 1, "REPLACE")
        modifier = self.obj.modifiers.new("Skeleton", "ARMATURE")
        modifier.object = rig
        self.obj.parent = rig
        material = bpy.data.materials.new("Test material")
        self.obj.data.materials.append(material)
        expected = self.evaluated()
        before = state(self.obj)

        class Report:
            def report(self, _level, _message):
                pass

        def exported_coordinates(path):
            model = AquaPackage(path.read_bytes()).models[0]
            return sorted(
                (round(v.X, 5), round(v.Y, 5), round(v.Z, 5))
                for buffer in model.vtxlList
                for v in buffer.vertPositions
            )

        with tempfile.TemporaryDirectory() as directory:
            for modifiers in (False, True):
                shape_path = Path(directory) / f"shape_{modifiers}.aqp"
                options = {"use_selection": True, "use_mesh_modifiers": modifiers}
                self.assertEqual(
                    export_model.export(
                        Report(), bpy.context, shape_path, options=options
                    ),
                    {"FINISHED"},
                )
                self.assertEqual(state(self.obj), before)
                # Independent reference: build plain vertices from Blender's
                # evaluated result, then run the same file conversion.
                reference = self.obj.data.copy()
                original = self.obj.data
                self.obj.data = reference
                reference_keys = reference.shape_keys
                self.obj.shape_key_clear()
                for vertex, co in zip(reference.vertices, expected, strict=True):
                    vertex.co = co
                reference.update()
                ref_path = Path(directory) / f"plain_{modifiers}.aqp"
                try:
                    export_model.export(
                        Report(), bpy.context, ref_path, options=options
                    )
                    self.assertEqual(
                        exported_coordinates(shape_path), exported_coordinates(ref_path)
                    )
                    self.assertEqual(
                        bytes(
                            AquaPackage(shape_path.read_bytes())
                            .models[0]
                            .GetBytesNIFL()
                        ),
                        bytes(
                            AquaPackage(ref_path.read_bytes()).models[0].GetBytesNIFL()
                        ),
                    )
                    options["apply_shape_keys"] = False
                finally:
                    self.obj.data = original
                    self.obj.active_shape_key_index = before["index"]
                    bpy.data.batch_remove(ids=(reference, reference_keys))
                base_path = Path(directory) / f"base_{modifiers}.aqp"
                export_model.export(Report(), bpy.context, base_path, options=options)
                self.assertNotEqual(
                    exported_coordinates(shape_path), exported_coordinates(base_path)
                )
                # Existing skeletons must not be overwritten.
                aqn_path = shape_path.with_suffix(".aqn")
                aqn_path.write_bytes(b"protected existing skeleton")
                options["apply_shape_keys"] = True
                export_model.export(Report(), bpy.context, shape_path, options=options)
                self.assertEqual(aqn_path.read_bytes(), b"protected existing skeleton")
                operator_path = Path(directory) / f"operator_{modifiers}.aqp"
                self.assertEqual(
                    bpy.ops.pso2.export_aqp(
                        filepath=str(operator_path),
                        use_selection=True,
                        use_space_transform=False,
                        use_mesh_modifiers=modifiers,
                        apply_shape_keys=True,
                    ),
                    {"FINISHED"},
                )
                self.assertCoordinates(
                    exported_coordinates(operator_path), sorted(expected)
                )
        self.assertEqual(state(self.obj), before)


if __name__ == "__main__":
    bpy.ops.preferences.addon_enable(module="pso2_tools")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ShapeKeyExportTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(
        json.dumps(
            {
                "blender": bpy.app.version_string,
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
            }
        )
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
