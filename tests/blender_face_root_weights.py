"""Exercise real AQP/FBX skinning with node 0, including a translated root."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).parent))
from blender_export_materials import Report, clear_scene, make_scene
from pso2_tools import import_model


def read_vertices(path):
    from AquaModelLibrary.Data.PSO2.Aqua import AquaPackage
    from Pso2Tools.Interop import ModelInterop

    model = AquaPackage(path.read_bytes()).models[0]
    model.splitVSETPerMesh()
    result = []
    for mesh_id in range(model.meshList.Count):
        data = ModelInterop.GetMesh(model, mesh_id)
        positions = np.frombuffer(bytes(data.Positions), dtype=np.float32).reshape(
            -1, 3
        )
        weights = np.frombuffer(bytes(data.Weights), dtype=np.float32).reshape(-1, 4)
        indices = np.frombuffer(bytes(data.WeightIndices), dtype=np.int32).reshape(
            -1, 4
        )
        palette = np.frombuffer(bytes(data.BonePalette), dtype=np.int32)
        for position, values, ids in zip(positions, weights, indices, strict=True):
            resolved = {}
            for weight, index in zip(values, ids, strict=True):
                if weight > 0 and 0 <= index < len(palette):
                    node = int(palette[index])
                    resolved[node] = resolved.get(node, 0) + float(weight)
            result.append((tuple(float(v) for v in position), resolved))
    return result


class RootWeightsTests(unittest.TestCase):
    def check_roundtrip(self, native, translated):
        make_scene()
        rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        root = rig.data.edit_bones.new("ModelRoot#4#0")
        root.head = (0.15, 0.2, 0.3) if translated else (0, 0, 0)
        root.tail = (root.head.x, root.head.y + 0.1, root.head.z)
        rig.data.edit_bones["Root#1C0#0"].parent = root
        bpy.ops.object.mode_set(mode="OBJECT")
        rig.data.bones["ModelRoot#4#0"]["pso2_bone_id"] = 0
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH":
                continue
            child_group = obj.vertex_groups["Root#1C0#0"]
            child_group.remove([0, 1])
            child_group.add([1], 0.75, "REPLACE")
            root_group = obj.vertex_groups.new(name="ModelRoot#4#0")
            root_group.add([0], 1.0, "REPLACE")
            root_group.add([1], 0.25, "REPLACE")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.aqp"
            self.assertEqual(
                bpy.ops.pso2.export_aqp(filepath=str(source)), {"FINISHED"}
            )
            original = read_vertices(source)
            skeleton = source.with_suffix(".aqn").read_bytes()
            self.assertTrue(any(weights.get(0, 0) == 1 for _, weights in original))
            self.assertTrue(any(0 < weights.get(0, 0) < 1 for _, weights in original))
            clear_scene()
            with patch.dict(
                os.environ, {"PSO2_TOOLS_NATIVE_IMPORT": "1" if native else "0"}
            ):
                result, _ = import_model._import_aqp(
                    Report(), bpy.context, source, source.with_suffix(".aqn")
                )
            self.assertEqual(result, {"FINISHED"})
            rig = next(o for o in bpy.context.selected_objects if o.type == "ARMATURE")
            self.assertEqual(sum(b.get("pso2_bone_id") == 0 for b in rig.data.bones), 1)
            target = Path(directory) / "roundtrip.aqp"
            target.with_suffix(".aqn").write_bytes(skeleton)
            self.assertEqual(
                bpy.ops.pso2.export_aqp(filepath=str(target)), {"FINISHED"}
            )
            self.assertEqual(target.with_suffix(".aqn").read_bytes(), skeleton)
            final = read_vertices(target)
            # Check both directions so dropped source vertices cannot go unnoticed.
            for before, after in ((original, final), (final, original)):
                tree = KDTree(len(before))
                for i, (point, _) in enumerate(before):
                    tree.insert(point, i)
                tree.balance()
                for point, weights in after:
                    candidates = tree.find_range(point, 1e-5)
                    self.assertTrue(candidates, f"Vertex moved: {point}")

                    def difference(i, before=before, weights=weights):
                        previous = before[i][1]
                        return max(
                            abs(weights.get(k, 0) - previous.get(k, 0))
                            for k in weights.keys() | previous.keys()
                        )

                    self.assertLessEqual(
                        min(difference(i) for _, i, _ in candidates), 1e-4
                    )

    def test_fbx_root_weights(self):
        self.check_roundtrip(False, False)

    def test_fbx_translated_root_weights(self):
        self.check_roundtrip(False, True)

    def test_native_root_weights(self):
        self.check_roundtrip(True, False)

    def test_native_translated_root_weights(self):
        self.check_roundtrip(True, True)


if __name__ == "__main__":
    bpy.ops.preferences.addon_enable(
        module=os.environ.get("PSO2_TEST_ADDON_MODULE", "pso2_tools")
    )
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(RootWeightsTests)
    )
    if not result.wasSuccessful():
        raise SystemExit(1)
