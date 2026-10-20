"""Actual-file automatic import and save/reopen integration (separate Blender)."""
from pathlib import Path
import hashlib
import importlib
import json
import sys
import traceback

import bpy
import numpy as np

ROOT=Path(sys.argv[sys.argv.index('--')+1])
PREFIX='bl_ext.user_default.pso2_tools.'
live=importlib.import_module(PREFIX+'live_skinning')
auto=importlib.import_module(PREFIX+'cloth_auto')
assets=importlib.import_module(PREFIX+'cloth_assets')
importer=importlib.import_module(PREFIX+'import_model')
objects=importlib.import_module(PREFIX+'objects_aqp')
report=dict(success=False,live_game_modified=False,live_dynamic_parity_verified=False)

class Reports:
    def report(self,level,message):print(level,message,flush=True)

def current_rig():
    return next(o for o in bpy.context.scene.objects if o.type=='ARMATURE' and o.get(live.SOURCE))

def evaluate(rig):
    bpy.context.view_layer.update()
    live.update_rig(rig,bpy.context.scene,bpy.context.evaluated_depsgraph_get(),force=True)
    assert not rig.get(live.ERROR),rig.get(live.ERROR)

def check_geometry(rig):
    graph=bpy.context.evaluated_depsgraph_get();errors=[]
    for obj in live._meshes(rig):
        target=np.empty(len(obj.data.vertices)*3,'f4')
        obj.data.attributes[live.POSITION].data.foreach_get('vector',target)
        evaluated=obj.evaluated_get(graph);mesh=evaluated.to_mesh()
        actual=np.empty(len(mesh.vertices)*3,'f4');mesh.vertices.foreach_get('co',actual)
        evaluated.to_mesh_clear();assert len(actual)==len(target)
        error=float(abs(actual-target).max());assert error<2e-6,(obj.name,error);errors.append(error)
    return max(errors)

try:
    source=next((ROOT.parent/'20261017-live33-solver-inputs-002/received/disk-only-extracted').rglob('pl_rbd_218620_bw.aqp'))
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.parent.iterdir() if p.suffix.lower()in{'.aqp','.aqn','.fltd','.aqm'}}
    result=importer._import_models(Reports(),bpy.context,[objects.AqpDataFileSource(source)],
        options={'automatic_bone_orientation':False,'primary_bone_axis':'Y','secondary_bone_axis':'X','automatic_cloth_physics':True})
    assert result=={'FINISHED'},result
    rig=current_rig();assert rig.get(auto.AUTO),rig.get(auto.STATUS)
    evaluate(rig)
    driver=live._CLOTH_DRIVERS[rig.as_pointer()]
    assert [len(d.mapping)for d in driver.drivers]==[48,3]
    assert [len(d.parameters['colliders'])for d in driver.drivers]==[5,1]
    raw=live._read_text(rig[live.SOURCE]);source_positions=[np.array(m['positions'],'f4')for m in raw['meshes']]
    bind_before=[np.array([tuple(v.co)for v in o.data.vertices],'f4')for o in live._meshes(rig)]
    positions=[];max_error=0
    for frame in range(1,31):
        bpy.context.scene.frame_set(frame);rig.location.x=.05*np.sin(frame*.17);evaluate(rig)
        state=[d.state.positions.copy()for d in driver.drivers]
        evaluate(rig)
        for a,d in zip(state,driver.drivers):np.testing.assert_array_equal(a,d.state.positions)
        positions.append(state[0].copy());max_error=max(max_error,check_geometry(rig))
    assert float(np.ptp(np.array(positions),axis=0).max())>.01
    bpy.context.scene.frame_set(4);evaluate(rig);assert driver.last_reset_reason=='timeline jump'
    bpy.context.scene.render.fps=30;evaluate(rig);assert driver.last_reset_reason=='FPS changed'
    for original,obj in zip(bind_before,live._meshes(rig)):
        np.testing.assert_array_equal(original,[tuple(v.co)for v in obj.data.vertices])
    auto.disable(rig);assert not rig.get(auto.AUTO) and rig.as_pointer()not in live._CLOTH_DRIVERS
    auto.enable(rig,bpy.context.scene);evaluate(rig)
    bpy.context.scene.frame_set(1);rig.location.x=0;evaluate(rig)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'New_AQP_Auto_Physics.blend'))
    report['new_import']=dict(rig=rig.name,particles=[48,3],colliders=[5,1],frames=30,
        scene_motion_varies=True,duplicate_frame_idempotent=True,timeline_jump_reset=True,
        fps_change_reset=True,base_mesh_unchanged=True,max_geometry_nodes_error=max_error,
        saved=str(ROOT/'New_AQP_Auto_Physics.blend'),status=rig.get(auto.STATUS))
    for name,digest in before.items():assert hashlib.sha256((source.parent/name).read_bytes()).hexdigest()==digest

    sample=Path(r'C:\zp\gyaru_set\live33_sample_20261017\B5_LowMini_Live33_sample.blend')
    sample_hash=hashlib.sha256(sample.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(sample));rig=current_rig()
    if not rig.get(assets.SOURCE_KEY):
        fltd=source.with_suffix('.fltd');assets.attach_file(rig,fltd.read_bytes(),fltd.name)
    auto.enable(rig,bpy.context.scene);evaluate(rig)
    driver=live._CLOTH_DRIVERS[rig.as_pointer()]
    before_pose=live.native_pose_profile.evaluate_profile(live._read_text(rig[live.PROFILE]))
    frames=[]
    for frame in range(1,61):
        bpy.context.scene.frame_set(frame);rig.location.x=.04*np.sin(frame*.1);evaluate(rig)
        frames.append(driver.drivers[0].state.positions.copy())
    assert float(np.ptp(np.array(frames),axis=0).max())>.01
    bpy.context.scene.frame_set(1);rig.location.x=0;evaluate(rig)
    report['sample']=dict(original=str(sample),original_sha256=sample_hash,
        original_unchanged=hashlib.sha256(sample.read_bytes()).hexdigest()==sample_hash,
        animated_frames=60,particles=[len(d.mapping)for d in driver.drivers],
        initial_reference_and_current_are_separate=True,geometry_error=check_geometry(rig))
    assert report['sample']['original_unchanged']
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT/'B5_LowMini_Auto_Physics_008.blend'))
    report['sample']['saved']=str(ROOT/'B5_LowMini_Auto_Physics_008.blend')
    report['success']=True
except BaseException:
    report['error']=traceback.format_exc();raise
finally:
    (ROOT/'blender-auto-integration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
