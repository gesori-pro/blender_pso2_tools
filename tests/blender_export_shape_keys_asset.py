"""Optional real-model check; set PSO2_TEST_AQP to an existing AQP path.

Run in a factory-startup background Blender. The input AQP/AQN are read
only. Outputs go in a temporary directory and are compared via Aqua.
"""

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from blender_export_shape_keys import coordinates, state
from pso2_tools import import_model, material as material_utils


def main():
    bpy.ops.preferences.addon_enable(
        module=os.environ.get("PSO2_TEST_ADDON_MODULE", "pso2_tools")
    )
    from AquaModelLibrary.Data.PSO2.Aqua import AquaNode, AquaPackage

    source = Path(os.environ["PSO2_TEST_AQP"])
    skeleton = source.with_suffix(".aqn")
    input_hashes = [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, skeleton)
    ]
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    class Report:
        def report(self, level, message):
            print(level, message)

    result, materials = import_model._import_aqp(
        Report(), bpy.context, source, skeleton
    )
    assert result == {"FINISHED"}
    for material in bpy.data.materials:
        source_material = material_utils.find_material(material.name, materials)
        if source_material and source_material.tsta_data:
            material["pso2_tsta"] = json.dumps(source_material.tsta_data)

    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    assert meshes
    if os.environ.get("PSO2_TEST_BASELINE"):
        with tempfile.TemporaryDirectory() as directory:
            assert bpy.ops.pso2.export_aqp(
                filepath=str(Path(directory) / "baseline.aqp"), apply_shape_keys=False
            ) == {"FINISHED"}
        print("Baseline import/export completed without shape keys")
        return
    for obj in meshes:
        obj.shape_key_add(name="Basis")
        shape = obj.shape_key_add(name="Generic export check")
        for vertex in shape.data:
            vertex.co.z += 0.013
        shape.value = 0.6

    before = [state(obj) for obj in meshes]
    expected = {}
    # Compute the shape independently, without armature or other modifiers.
    modifiers = [(mod, mod.show_viewport) for obj in meshes for mod in obj.modifiers]
    try:
        for mod, _ in modifiers:
            mod.show_viewport = False
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        for obj in meshes:
            expected[obj.name] = coordinates(obj.evaluated_get(graph).data.vertices)
    finally:
        for mod, visible in modifiers:
            mod.show_viewport = visible
        bpy.context.view_layer.update()

    def export(path, apply=True):
        assert bpy.ops.pso2.export_aqp(
            filepath=str(path), apply_shape_keys=apply, use_selection=False
        ) == {"FINISHED"}
        assert [state(obj) for obj in meshes] == before

    def model_bytes(path):
        return bytes(AquaPackage(path.read_bytes()).models[0].GetBytesNIFL())

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.aqp"
        export(path, False)
        base = model_bytes(path)
        export(path)
        mixed = model_bytes(path)
        assert mixed != base, "Export still contains the basis"
        mixed_skeleton = AquaNode(path.with_suffix(".aqn").read_bytes())

        swapped = []
        try:
            for obj in meshes:
                original = obj.data
                copy = original.copy()
                key_uid = copy.shape_keys.session_uid
                swapped.append(
                    (obj, original, copy, key_uid, obj.active_shape_key_index)
                )
                obj.data = copy
                obj.shape_key_clear()
                for vertex, co in zip(copy.vertices, expected[obj.name], strict=True):
                    vertex.co = co
                copy.update()
            bpy.context.view_layer.update()
            assert bpy.ops.pso2.export_aqp(filepath=str(path), use_selection=False) == {
                "FINISHED"
            }
            assert model_bytes(path) == mixed, (
                "AQP differs from the evaluated plain mesh"
            )
        finally:
            for obj, original, copy, key_uid, index in reversed(swapped):
                obj.data = original
                obj.active_shape_key_index = index
                remaining_keys = [
                    key for key in bpy.data.shape_keys if key.session_uid == key_uid
                ]
                bpy.data.batch_remove(ids=(copy, *remaining_keys))
            bpy.context.view_layer.update()
        assert [state(obj) for obj in meshes] == before
        model = AquaPackage(path.read_bytes()).models[0]
        report = {
            "blender": bpy.app.version_string,
            "source": source.name,
            "meshes": len(meshes),
            "vertices": sum(len(obj.data.vertices) for obj in meshes),
            "exported_nodes": mixed_skeleton.nodeList.Count,
            "materials": model.mateList.Count,
            "texture_names": [entry.texName.GetString() for entry in model.texfList],
            "model_bytes_match_evaluated_reference": True,
            "scene_restored": True,
        }
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        result, _ = import_model._import_aqp(
            Report(), bpy.context, path, path.with_suffix(".aqn")
        )
        assert result == {"FINISHED"}
        reloaded = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
        assert len(reloaded) == report["meshes"]
        assert all(
            math.isfinite(c)
            for obj in reloaded
            for v in obj.data.vertices
            for c in v.co
        )
        report["blender_reimport_meshes"] = len(reloaded)
    assert input_hashes == [
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, skeleton)
    ]
    print(json.dumps(report))


if __name__ == "__main__":
    main()
