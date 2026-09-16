"""Asset regression: set PSO2_STAGE_ORIGINAL and PSO2_STAGE_ADDITIONS.

Run in factory-startup background Blender; never writes into the asset folder.
"""

import os
import sys
import tempfile
from pathlib import Path

import bpy
from bpy_extras.io_utils import axis_conversion

sys.path.insert(0, str(Path(__file__).parent))
import blender_export_shape_keys  # noqa: F401

bpy.ops.preferences.addon_enable(module="pso2_tools")
from AquaModelLibrary.Data.PSO2.Aqua import AquaPackage  # noqa: E402
from pso2_tools import export_model, import_model, stage_model  # noqa: E402


class Report:
    def report(self, level, message):
        print(level, message)


def read(path):
    return AquaPackage(path.read_bytes()).models[0]


def payload(model, index):
    layout = model.vtxeList[model.vsetList[index].vtxeCount]
    return bytes(model.vtxlList[index].GetBytes(layout, layout.GetVTXESize()))


original_path = Path(os.environ["PSO2_STAGE_ORIGINAL"])
additions_path = Path(os.environ["PSO2_STAGE_ADDITIONS"])
original = read(original_path)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
report = Report()
for path in (original_path, additions_path):
    result, _ = import_model._import_aqp(
        report, bpy.context, path, path.with_suffix(".aqn")
    )
    assert result == {"FINISHED"}
stages = [o for o in bpy.context.scene.objects if o.get(stage_model.SOURCE)]
assert len(stages) == original.meshList.Count
options = {
    "global_matrix": axis_conversion(to_forward="-Z", to_up="Y").to_4x4(),
    "bake_anim": False,
}
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "stage_test.aqp"
    assert export_model.export(report, bpy.context, path, options=options) == {
        "FINISHED"
    }
    actual = read(path)
    assert actual.objc.type == original.objc.type
    # Compare to AML's own model classification, not a hardcoded scene ID or flag.
    unpreserved = Path(directory) / "converted.aqp"
    keys = [(o, o[stage_model.SOURCE]) for o in stages]
    for obj, _ in keys:
        del obj[stage_model.SOURCE]
    try:
        assert export_model.export(
            report, bpy.context, unpreserved, options=options
        ) == {"FINISHED"}
    finally:
        for obj, key in keys:
            obj[stage_model.SOURCE] = key
    converted = read(unpreserved)
    assert actual.objc.unkMeshValue == converted.objc.unkMeshValue
    assert payload(actual, 0) == payload(original, 0)
    assert (
        actual.meshList.Count
        == original.meshList.Count + read(additions_path).meshList.Count
    )
    assert actual.mesh2List.Count == original.mesh2List.Count
    for i in range(original.meshList.Count):
        assert list(actual.strips[i].triStrips) == list(original.strips[i].triStrips)
    print(
        "PASS: original stage payload, model type, runtime classification, added meshes"
    )

    # Provenance survives user renames and blend-file save/reload.
    obj = stages[0]
    obj.name = "Renamed stage part"
    obj.data.materials[0].name = "User renamed material"
    blend = Path(directory) / "saved.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    obj = bpy.data.objects["Renamed stage part"]
    assert export_model.export(report, bpy.context, path, options=options) == {
        "FINISHED"
    }
    assert payload(read(path), 0) == payload(original, 0)
    print("PASS: material/object rename and blend reload")

    # Positional edits must survive; a passthrough that discards edits is wrong.
    for vertex in obj.data.vertices:
        vertex.co.x += 0.025
    obj.data.update()
    assert export_model.export(report, bpy.context, path, options=options) == {
        "FINISHED"
    }
    edited = read(path)
    assert payload(edited, 0) != payload(original, 0)
    changed = {v.value for v in obj.data.attributes[stage_model.VERTEX_ID].data}
    for i, p in enumerate(original.vtxlList[0].vertPositions):
        expected = p.X + (0.025 if i in changed else 0)
        assert abs(edited.vtxlList[0].vertPositions[i].X - expected) < 1e-5
    obj.shape_key_add(name="Basis")
    key = obj.shape_key_add(name="Test shape")
    for vertex in key.data:
        vertex.co.z += 0.01
    key.value = 0.75
    assert export_model.export(report, bpy.context, path, options=options) == {
        "FINISHED"
    }
    assert payload(read(path), 0) != payload(edited, 0)
    assert obj.data.shape_keys.key_blocks["Test shape"].value == 0.75
    print("PASS: positional edits and non-destructive shape keys")

    # Missing a required part must cancel before replacing an existing file.
    marker = path.read_bytes()
    obj.select_set(True)
    for other in bpy.context.scene.objects:
        if other != obj:
            other.select_set(False)
    assert export_model.export(
        report, bpy.context, path, options={**options, "use_selection": True}
    ) == {"CANCELLED"}
    assert path.read_bytes() == marker
    print("PASS: incomplete stage export leaves destination untouched")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    result, _ = import_model._import_aqp(
        report, bpy.context, path, path.with_suffix(".aqn")
    )
    assert result == {"FINISHED"}
    imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    assert len(imported) == read(path).meshList.Count
    print("PASS: exported mixed model re-imports with all meshes")

print("STAGE_MODEL_TESTS_PASSED")
