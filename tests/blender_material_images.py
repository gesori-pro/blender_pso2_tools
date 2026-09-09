"""Regression tests with real Blender Image RNA, in a factory-startup process."""

import os
import site
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import bpy

ROOT = Path(os.environ.get("PSO2_TEST_ADDON_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
if dependencies := os.environ.get("PSO2_TEST_DEPENDENCIES"):
    site.addsitedir(dependencies)

from pso2_tools import import_model, material  # noqa: E402


class ImageLifetimeTests(unittest.TestCase):
    def setUp(self):
        for image in list(bpy.data.images):
            bpy.data.images.remove(image)

    def image(self, name):
        return bpy.data.images.new(name, width=2, height=2)

    def test_removed_reference_does_not_hide_live_match(self):
        removed = self.image("pl_rbd_sk_000_d.dds")
        live = self.image("pl_rbd_sk_001_d.dds")
        cached = [removed, live]
        bpy.data.images.remove(removed)
        with self.assertRaises(ReferenceError):
            _ = removed.name
        self.assertEqual(material.find_textures("rbd", "sk", images=cached), [live])
        model = material.ModelMaterials(textures=cached)
        self.assertEqual(model._get_texture_by_name(live.name), live)

    def test_empty_explicit_set_does_not_pick_unrelated_scene_image(self):
        live = self.image("pl_rbd_sk_001_d.dds")
        self.assertEqual(material.find_textures("sk"), [live])
        self.assertEqual(material.find_textures("sk", images=[]), [])
        self.assertIsNone(
            material.ModelMaterials()
            ._get_texture_set("pl_body_skin_diffuse.dds")
            .skin_0.diffuse
        )

    def test_cleanup_then_skin_material_lookup(self):
        empty = self.image("pl_rbd_sk_000_d.dds")
        empty.source = "FILE"
        empty.filepath = str(ROOT / "tests" / "missing-skin-image.dds")
        self.assertEqual(tuple(empty.size), (0, 0))
        first = self.image("pl_rbd_sk_001_d.dds")
        second = self.image("pl_rbd_sk_002_d.dds")
        model = material.ModelMaterials(
            textures=[empty],
            skin_textures=[empty, second, first],
            extra_textures=[empty],
        )
        import_model._delete_empty_images()
        model.discard_removed_images()
        self.assertEqual(model.textures, [])
        self.assertEqual(model.extra_textures, [])
        self.assertEqual(model.skin_textures, [second, first])
        result = model.get_textures(
            material.Material(textures=["pl_body_skin_diffuse.dds"])
        )
        self.assertEqual(result.skin_0.diffuse, first)
        self.assertEqual(result.skin_1.diffuse, second)
        model.discard_removed_images()
        self.assertEqual(model.skin_textures, [second, first])

    def test_all_removed_skin_textures_are_missing_without_crashing(self):
        removed = self.image("pl_rbd_sk_000_d.dds")
        model = material.ModelMaterials(skin_textures=[removed])
        bpy.data.images.remove(removed)
        result = model.get_textures(
            material.Material(textures=["pl_body_skin_diffuse.dds"])
        )
        self.assertIsNone(result.skin_0.diffuse)
        self.assertIsNone(result.skin_1.diffuse)

    def test_missing_external_image_is_not_a_usable_texture(self):
        missing = self.image("pl_rbd_sk_000_d.dds")
        missing.source = "FILE"
        missing.filepath = str(ROOT / "tests" / "missing-skin-image.dds")
        self.assertEqual(tuple(missing.size), (0, 0))
        self.assertEqual(material.find_textures("rbd", "sk"), [])
        self.assertIsNone(
            material.ModelMaterials(textures=[missing])._get_texture_by_name(
                missing.name
            )
        )

    def test_import_reloads_skin_when_scene_contains_only_missing_images(self):
        missing = self.image("pl_rbd_sk_000_d.dds")
        missing.source = "FILE"
        missing.filepath = str(ROOT / "tests" / "missing-skin-image.dds")
        skin_material = material.Material(
            textures=["pl_body_skin_diffuse.dds"], shaders=["1102p", "1102"]
        )
        files = import_model.ModelFiles(model_files=[SimpleNamespace(name="test.aqp")])

        def import_mesh(*args, **kwargs):
            bpy.data.materials.new("Test imported skin")
            return {"FINISHED"}, [skin_material]

        recovered = []

        def import_skin(*args, **kwargs):
            recovered.extend(
                [self.image("pl_rbd_sk_001_d.dds"), self.image("pl_rbd_sk_002_d.dds")]
            )
            return recovered

        with (
            patch.object(import_model, "debug_print"),
            patch.object(import_model, "debug_pprint"),
            patch.object(import_model, "collect_model_files", return_value=files),
            patch.object(import_model, "_import_aqp", side_effect=import_mesh),
            patch.object(material, "find_material", return_value=skin_material),
            patch.object(
                import_model, "_import_skin_textures", side_effect=import_skin
            ) as loader,
            patch.object(import_model.shaders, "build_material") as builder,
        ):
            result = import_model._import_models(
                Mock(),
                bpy.context,
                [],
                options={"apply_shape_adjust": False},
                use_t2_skin=True,
            )
        self.assertEqual(result, {"FINISHED"})
        loader.assert_called_once_with(bpy.context, True, True)
        textures = builder.call_args.args[2].textures
        self.assertEqual(textures.skin_0.diffuse, recovered[0])
        self.assertEqual(textures.skin_1.diffuse, recovered[1])


suite = unittest.defaultTestLoader.loadTestsFromTestCase(ImageLifetimeTests)
result = unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful():
    raise SystemExit(1)
