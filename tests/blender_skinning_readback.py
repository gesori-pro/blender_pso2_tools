"""Reopen the delivered preview and verify persisted drivers/profile data."""

import importlib
import json
from pathlib import Path
import sys
import traceback

import bpy
import numpy as np

report = {}
try:
    module = importlib.import_module("bl_ext.user_default.pso2_tools.game_skinning")
    snapshots = [
        o for o in bpy.context.scene.objects if o.get(module.PREVIEW_PROP) == "CAPTURE"
    ]
    assert len(snapshots) == 3
    rig = next(
        o
        for o in bpy.context.scene.objects
        if o.type == "ARMATURE" and o.get(module.PREVIEW_PROP) == "REFERENCE"
    )
    source = bpy.data.objects[rig[module.SOURCE_PROP]]
    for collection in rig.users_collection:
        collection.hide_viewport = False
    bpy.context.view_layer.update()
    before = rig.pose.bones["hemroot#2#0"].head.copy()
    hip = source.pose.bones["hip#1C4#0"]
    old = hip.scale.copy()
    hip.scale = old * 1.01
    bpy.context.view_layer.update()
    delta = (
        rig.pose.bones["hemroot#2#0"].head - source.pose.bones["hemroot#2#0"].head
    ).y * 1000
    assert abs(delta - 78.52905989 * 1.01) < 0.01, delta
    assert (rig.pose.bones["hemroot#2#0"].head - before).length > 0.001
    hip.scale = old
    bpy.context.view_layer.update()
    report["persisted_drivers"] = len(rig.animation_data.drivers)
    assert all(c.driver.is_valid for c in rig.animation_data.drivers)
    errors = []
    for obj in snapshots:
        data = json.loads(bpy.data.texts[obj["pso2_capture_text"]].as_string())
        record = data["meshes"][obj["pso2_capture_mesh_index"]]
        ev = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        mesh = ev.to_mesh()
        positions = np.array([list(v.co) for v in mesh.vertices])
        ev.to_mesh_clear()
        error = float(
            np.linalg.norm(positions - np.array(record["gpu_positions"]), axis=1).max()
            * 1000
        )
        assert error < 0.005, error
        errors.append(error)
    report["postvs_readback_max_mm"] = max(errors)
    report["embedded_capture_profile"] = True
    report["success"] = True
except Exception:
    report["error"] = traceback.format_exc()
Path(sys.argv[sys.argv.index("--") + 1]).write_text(
    json.dumps(report, indent=2), encoding="utf8"
)
print(json.dumps(report, indent=2))
