"""AQP -> direct macOS import -> AQP, with root skinning and a moved root.

Run in a factory-startup Blender with the extension enabled. Outputs are
temporary, and no user model or game data is required.
"""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bpy
import numpy as np

sys.path.insert(0, os.getenv("PSO2_TEST_ADDON_ROOT", str(Path(__file__).parents[1])))
bpy.ops.preferences.addon_enable(
    module=os.getenv("PSO2_TEST_ADDON_MODULE", "pso2_tools")
)

from pso2_tools import import_model, scene_props  # noqa: E402
from AquaModelLibrary.Data.PSO2.Aqua import AquaNode, AquaPackage  # noqa: E402
from Pso2Tools.Interop import ModelInterop  # noqa: E402


class Report:
    def report(self, level, text):
        print(level, text)


def snapshot(path):
    model = AquaPackage(path.read_bytes()).models[0]
    skeleton = AquaNode(path.with_suffix(".aqn").read_bytes())
    model.splitVSETPerMesh()
    rows = []
    for mesh_id in range(ModelInterop.GetMeshCount(model)):
        for group_id in ModelInterop.GetFaceGroupIds(model, mesh_id):
            data = ModelInterop.GetMesh(model, mesh_id, group_id)
            positions = np.frombuffer(bytes(data.Positions), dtype=np.float32).reshape(
                -1, 3
            )
            weights = np.frombuffer(bytes(data.Weights), dtype=np.float32).reshape(
                -1, 4
            )
            indices = np.frombuffer(bytes(data.WeightIndices), dtype=np.int32).reshape(
                -1, 4
            )
            palette = np.frombuffer(bytes(data.BonePalette), dtype=np.int32)
            for position, values, ids in zip(positions, weights, indices, strict=True):
                resolved = {}
                for value, index in zip(values, ids, strict=True):
                    if value > 0 and 0 <= index < len(palette):
                        name = str(skeleton.GetNodeName(int(palette[index])))
                        resolved[name] = resolved.get(name, 0) + float(value)
                rows.append((tuple(float(v) for v in position), resolved))
    return sorted(rows, key=lambda row: row[0])


def roundtrip(translated):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    data = bpy.data.armatures.new("Skeleton")
    rig = bpy.data.objects.new("pl_test#1C0#0", data)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root = data.edit_bones.new("ModelRoot#4#0")
    root.head = (0.15, 0.2, 0.3) if translated else (0, 0, 0)
    root.tail = (root.head.x, root.head.y + 0.1, root.head.z)
    child = data.edit_bones.new("Child#1C0#0")
    child.head, child.tail = (0, 0.2, 0), (0, 0.3, 0)
    child.parent = root
    bpy.ops.object.mode_set(mode="OBJECT")
    for index, name in enumerate(("ModelRoot#4#0", "Child#1C0#0")):
        data.bones[name][scene_props.BONE_ID] = index
    mesh = bpy.data.meshes.new("mesh[0]_0_0_0_0#0#0")
    mesh.from_pydata([(0, 0, 0), (0.1, 0.2, 0), (0, 0.1, 0.3)], [], [(0, 1, 2)])
    mesh.uv_layers.new(name="UVChannel_1")
    mesh.uv_layers[0].data.foreach_set("uv", [0, 0, 1, 0, 0, 1])
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = rig
    obj.modifiers.new("Rig", "ARMATURE").object = rig
    obj.vertex_groups.new(name="ModelRoot#4#0").add([0], 1, "REPLACE")
    obj.vertex_groups[0].add([1], 0.25, "REPLACE")
    obj.vertex_groups.new(name="Child#1C0#0").add([1], 0.75, "REPLACE")
    obj.vertex_groups[1].add([2], 1, "REPLACE")
    mesh.materials.append(bpy.data.materials.new("(1103p,1103){opaque}test@0@0"))
    with TemporaryDirectory() as directory:
        first = Path(directory) / "source.aqp"
        second = Path(directory) / "second.aqp"
        assert bpy.ops.pso2.export_aqp(filepath=str(first)) == {"FINISHED"}
        original = snapshot(first)
        assert any(row[1].get("ModelRoot") == 1 for row in original)
        assert any(0 < row[1].get("ModelRoot", 0) < 1 for row in original)
        for item in list(bpy.data.objects):
            bpy.data.objects.remove(item, do_unlink=True)
        assert import_model._use_native_import()
        result, _ = import_model._import_aqp(
            Report(), bpy.context, first, first.with_suffix(".aqn")
        )
        assert result == {"FINISHED"}
        imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        assert imported and imported[0].data.uv_layers[0].name == "UVChannel_1"
        assert all(o.find_armature() is not None for o in imported)
        assert bpy.ops.pso2.export_aqp(filepath=str(second)) == {"FINISHED"}
        exported = snapshot(second)
        assert len(original) == len(exported)
        for (pos_a, weights_a), (pos_b, weights_b) in zip(
            original, exported, strict=True
        ):
            assert np.allclose(pos_a, pos_b, atol=1e-5), (pos_a, pos_b)
            assert weights_a.keys() == weights_b.keys(), (weights_a, weights_b)
            for bone, value in weights_a.items():
                assert abs(value - weights_b[bone]) < 1e-4


for moved in (False, True):
    roundtrip(moved)
print("MACOS_MODEL_RESULT", json.dumps({"root_weight_roundtrips": 2, "passed": True}))
