"""Integration on a copy of a real scene; never saves the input blend.

blender -b source.blend --python this.py -- capture.json reference.aqn output.blend result.json
"""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback

import bpy
import numpy as np

capture_path, reference_path, output_path, result_path = sys.argv[
    sys.argv.index("--") + 1 :
]
report = {}
try:
    module = importlib.import_module("bl_ext.user_default.pso2_tools.game_skinning")
    math = importlib.import_module("bl_ext.user_default.pso2_tools.skinning_math")
    data = math.load_capture(capture_path)
    reference = module.read_reference_aqn(reference_path)
    targets = [
        o
        for o in bpy.data.objects
        if o.type == "ARMATURE" and "pl_rbd_218620_bw#" in o.name
    ]
    source = next(o for o in targets if o.name == "pl_rbd_218620_bw#1CF#0")
    bpy.context.view_layer.objects.active = source
    source.hide_set(False)
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    source.select_set(True)

    def fingerprint():
        payload = [
            (
                bone.name,
                [list(row) for row in bone.matrix_basis],
                [list(row) for row in bone.bone.matrix_local],
                bone.bone.use_connect,
            )
            for bone in source.pose.bones
        ]
        payload.extend(
            (
                obj.name,
                [list(v.co) for v in obj.data.vertices],
                [[(g.group, g.weight) for g in v.groups] for v in obj.data.vertices],
            )
            for obj in source.children
            if obj.type == "MESH"
        )
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    before = fingerprint()
    # Exercise the real operators, registration, error reporting and file dialogs' properties.
    assert bpy.ops.pso2.import_skinning_capture(filepath=capture_path) == {"FINISHED"}
    snapshots = [
        o for o in bpy.context.scene.objects if o.get(module.PREVIEW_PROP) == "CAPTURE"
    ]
    capture_errors = []
    for record in data["meshes"]:
        obj = next(o for o in snapshots if o["pso2_capture_event"] == record["event"])
        # Independent ground truth recovered from RenderDoc SV_Position,
        # not a second call to the implementation under test.
        expected = np.asarray(record["gpu_positions"])
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        me = ev.to_mesh()
        got = np.array([list(v.co) for v in me.vertices])
        ev.to_mesh_clear()
        error = float(np.linalg.norm(got - expected, axis=1).max() * 1000)
        if error >= 0.005:
            report["snapshot_debug"] = {
                "event": record["event"],
                "expected": expected[:3].tolist(),
                "actual": got[:3].tolist(),
                "keys": [
                    [k.name, k.value, k.relative_key.name, list(k.data[0].co)]
                    for k in obj.data.shape_keys.key_blocks
                ],
            }
        assert error < 0.005, error
        capture_errors.append({"event": record["event"], "max_mm": error})
    report["capture_snapshot_errors"] = capture_errors
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    for obj in snapshots:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = snapshots[0]
    for pose in ["BLENDER", "REST", "CAPTURE"]:
        assert bpy.ops.pso2.skinning_comparison(pose=pose) == {"FINISHED"}
        for obj in snapshots:
            assert obj.data.shape_keys.key_blocks["Captured game"].value == (
                pose == "CAPTURE"
            )
    front = next(o for o in snapshots if o["pso2_capture_event"] == 305)
    bpy.context.view_layer.objects.active = front
    assert bpy.ops.pso2.inspect_skinning_vertex(vertex=160) == {"FINISHED"}
    detail = json.loads(bpy.data.texts[front["pso2_capture_last_report"]].as_string())
    assert abs(detail["game_minus_blender_mm"][1] - 63.4475) < 0.01
    assert abs(sum(x["effective_weight"] for x in detail["influences"]) - 1) < 1e-6
    report["vertex_inspection"] = True
    report["comparison_switches"] = True

    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    source.select_set(True)
    bpy.context.view_layer.objects.active = source
    assert bpy.ops.pso2.preview_reference_skinning(
        filepath=reference_path, bone_filter="hemroot,hem_,drs2_tube_skirt_"
    ) == {"FINISHED"}
    rig = bpy.context.object
    assert rig.get(module.PREVIEW_PROP) == "REFERENCE"
    corrected = len(rig["pso2_reference_offsets"])
    assert corrected > 0
    report["corrected_bones"] = corrected
    src_root = source.pose.bones["hemroot#2#0"]
    ref_root = rig.pose.bones[src_root.name]
    actual_delta = (ref_root.head - src_root.head).y * 1000
    assert abs(actual_delta - 78.52906177) < 0.01, actual_delta
    report["hemroot_correction_mm"] = actual_delta
    for n in [f"drs2_tube_skirt_{i:03d}_00#4#0" for i in range(0, 120, 10)]:
        assert (
            abs(
                (rig.pose.bones[n].head - source.pose.bones[n].head).y * 1000
                - actual_delta
            )
            < 0.01
        )
    report["chain_roots_corrected_once"] = True
    previous = ref_root.head.copy()
    old_scale = source.pose.bones["hip#1C4#0"].scale.copy()
    source.pose.bones["hip#1C4#0"].scale = old_scale * 1.02
    bpy.context.view_layer.update()
    assert (ref_root.head - previous).length > 0.001
    # Dynamic driver re-evaluation must keep the edited/source offsets scaled by the parent.
    dynamic_delta = (ref_root.head - src_root.head).y * 1000
    assert abs(dynamic_delta - actual_delta * 1.02) < 0.01, dynamic_delta
    source.pose.bones["hip#1C4#0"].scale = old_scale
    bpy.context.view_layer.update()
    report["dynamic_parent_scale"] = True
    assert fingerprint() == before, "Original source data changed"
    report["source_fingerprint_unchanged"] = before
    # Preview file starts with the captured skirt, with live comparison kept as an optional collection.
    for collection in rig.users_collection:
        collection.hide_viewport = True
        collection.hide_render = True
    for name in [
        "mesh[0]_4_2_0_0#-1#2#0",
        "mesh[1]_4_2_0_0#-1#4#0",
        "mesh[2]_4_2_0_0#-1#7#0",
    ]:
        bpy.data.objects[name].hide_set(True)
    for obj in list(bpy.context.selected_objects):
        obj.select_set(False)
    for obj in snapshots:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = snapshots[0]
    bpy.ops.wm.save_as_mainfile(filepath=output_path)
    report["saved_preview"] = output_path
except Exception:
    report["error"] = traceback.format_exc()
finally:
    Path(result_path).write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
