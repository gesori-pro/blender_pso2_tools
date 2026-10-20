"""Actual-file regression for AQN rotation order and automatic FLTD import.

Run Blender in background with --python this_file -- <evidence-folder>.
The folder supplies import-inputs/pl_rbd_218620_bw.{aqp,aqn,fltd,aqm}.
No client files are modified and no blend file is written by this test.
"""
from pathlib import Path
import copy
import hashlib
import importlib
import json
import sys
import traceback

import bpy
import numpy as np

ROOT = Path(sys.argv[sys.argv.index("--") + 1])
PREFIX = "bl_ext.user_default.pso2_tools."
auto = importlib.import_module(PREFIX + "cloth_auto")
live = importlib.import_module(PREFIX + "live_skinning")
importer = importlib.import_module(PREFIX + "import_model")
objects = importlib.import_module(PREFIX + "objects_aqp")
report = {"success": False, "live_dynamic_parity_verified": False}


class Reports:
    def report(self, level, message):
        print(level, message, flush=True)


try:
    source = ROOT / "import-inputs/pl_rbd_218620_bw.aqp"
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.parent.iterdir()}
    result = importer._import_models(
        Reports(), bpy.context, [objects.AqpDataFileSource(source)],
        options={"automatic_bone_orientation": False,
                 "primary_bone_axis": "X", "secondary_bone_axis": "Y",
                 "automatic_cloth_physics": True})
    assert result == {"FINISHED"}, result
    rig = next(o for o in bpy.context.scene.objects
               if o.type == "ARMATURE" and o.get(live.SOURCE))
    raw = live._read_text(rig[live.SOURCE])
    template = copy.deepcopy(raw["native_template"])
    assert template["source_aqn_euler_order"] == "XZY"
    reference = live.native_pose_profile.evaluate_profile(template)
    errors = []
    for mesh in raw["meshes"]:
        restored = live.skinning_math.skin_positions(
            mesh["positions"], mesh["weights_u16"], mesh["bone_indices"],
            mesh["palette"], reference["skin_matrices"])
        errors.append(float(np.linalg.norm(
            restored - np.asarray(mesh["positions"]), axis=1).max()))
    # The previous XYZ conversion displaced this fixture by over one metre.
    assert max(errors) < 2e-5, errors
    assert rig.get(live.MODE) == "NATIVE_INPUTS"
    assert rig.get(auto.AUTO), rig.get(auto.STATUS)
    original = [np.asarray([tuple(v.co) for v in o.data.vertices])
                for o in live._meshes(rig)]
    frames = []
    for frame in (1, 2, 3, 4, 5):
        bpy.context.scene.frame_set(frame)
        rig.location.x = .025 * np.sin(frame * .4)
        bpy.context.view_layer.update()
        live.update_rig(rig, bpy.context.scene,
                        bpy.context.evaluated_depsgraph_get(), force=True)
        assert not rig.get(live.ERROR), rig.get(live.ERROR)
        driver = live._CLOTH_DRIVERS[rig.as_pointer()]
        state = [d.state.positions.copy() for d in driver.drivers]
        live.update_rig(rig, bpy.context.scene,
                        bpy.context.evaluated_depsgraph_get(), force=True)
        for first, d in zip(state, driver.drivers):
            np.testing.assert_array_equal(first, d.state.positions)
            assert np.isfinite(d.state.positions).all()
        frames.append(state[0])
    assert float(np.ptp(np.asarray(frames), axis=0).max()) > .001
    for original_mesh, obj in zip(original, live._meshes(rig)):
        np.testing.assert_array_equal(
            original_mesh, [tuple(v.co) for v in obj.data.vertices])
    for p, digest in before.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    report.update(success=True, max_reference_error_mm=max(errors) * 1000,
                  meshes=len(errors), frames=5, source_files_unchanged=True,
                  base_mesh_unchanged=True, duplicate_frame_idempotent=True)
except BaseException:
    report["error"] = traceback.format_exc()
    raise
finally:
    (ROOT / "new-import-regression.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
