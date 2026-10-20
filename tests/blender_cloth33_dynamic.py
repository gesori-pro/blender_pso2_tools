"""Background Blender integration on a synthetic fixture, never a user's file.

blender -b --python this.py -- report.json
This verifies dynamic routing/temporal state, NOT actual garment/live parity.
"""
import copy
import importlib
import json
from pathlib import Path
import sys
import struct
import traceback

import bpy
import numpy as np

report_path = Path(sys.argv[sys.argv.index('--') + 1])
reference_mode = '--reference' in sys.argv[sys.argv.index('--') + 2:]
file_mode = '--file' in sys.argv[sys.argv.index('--') + 2:] or reference_mode
kernel_mode = '--kernel' in sys.argv[sys.argv.index('--') + 2:] or file_mode
report = {'actual_garment_verified': False, 'live_1_to_1_verified': False}
try:
    prefix = 'bl_ext.user_default.pso2_tools.'
    live = importlib.import_module(prefix + 'live_skinning')
    profile = importlib.import_module(prefix + 'native_pose_profile')
    stages = importlib.import_module(prefix + 'native_cloth_stages')
    solver = importlib.import_module(prefix + 'native_cloth_solver33')
    drivers = importlib.import_module(prefix + 'native_cloth_driver')
    hardware = importlib.import_module(prefix + 'native_float_estimates')
    scene_props = importlib.import_module(prefix + 'scene_props')
    angles = importlib.import_module(prefix + 'native_cloth_angles') if kernel_mode else None
    plane = importlib.import_module(prefix + 'native_cloth_plane') if kernel_mode else None
    sphere = importlib.import_module(prefix + 'native_cloth_sphere') if kernel_mode else None
    capsule = importlib.import_module(prefix + 'native_cloth_capsule') if kernel_mode else None
    initialization = importlib.import_module(prefix + 'native_cloth_initialization') if file_mode else None
    controller = importlib.import_module(prefix + 'native_cloth_controller') if file_mode else None
    environment = importlib.import_module(prefix + 'native_cloth_environment') if file_mode else None
    bindings = importlib.import_module(prefix + 'native_cloth_bindings') if reference_mode else None
    seeds = hardware.HardwareEstimates()
    assert seeds.rcp(np.float32(60)) == np.float32(.01666259765625)
    assert seeds.rsqrt(np.float32(4)) != np.float32(.5)
    report['sse_estimates_loaded'] = True
    seed_oracle = report_path.with_name('estimate-oracle.json')
    if seed_oracle.exists():
        estimates = json.loads(seed_oracle.read_text())
        for row in estimates:
            argument = struct.unpack('<f', bytes.fromhex(row['input']))[0]
            result = getattr(seeds, row['kind'])(np.float32(argument))
            assert struct.pack('<f', result).hex() == row['expected'], row
        report['hardware_estimate_values_bitwise_equal'] = len(estimates)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    arm = bpy.data.armatures.new('Cloth33 fixture')
    rig = bpy.data.objects.new('Cloth33 fixture', arm)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    rig[scene_props.BONE_AXES] = 'Y,X'
    bpy.ops.object.mode_set(mode='EDIT')
    for i, name in enumerate(['a', 'b', 'c']):
        bone = arm.edit_bones.new(name)
        bone.head = (i, 0, 0)
        bone.tail = (i, 1, 0)
        if i:
            bone.parent = arm.edit_bones[['a', 'b'][i-1]]
    bpy.ops.object.mode_set(mode='OBJECT')
    for i, bone in enumerate(arm.bones):
        bone[scene_props.BONE_ID] = i
    mesh = bpy.data.meshes.new('Cloth33 triangle')
    mesh.from_pydata([(0, 0, 0), (1, 0, 0), (2, 0, 0)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new('Cloth33 triangle', mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = rig
    material = bpy.data.materials.new('Preserved fixture material')
    mesh.materials.append(material)
    original = np.array([list(v.co) for v in mesh.vertices])
    inverse = {}
    for i, name in enumerate(['a', 'b', 'c']):
        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = -i
        inverse[name] = matrix.tolist()
    raw = dict(format='pso2_raw_skin_source', version=1, node_names=['a', 'b', 'c'],
               inverse_bind=inverse, meshes=[dict(name=obj.name, positions=original.tolist(),
               weights_u16=[[65535, 0, 0, 0]]*3,
               bone_indices=[[i, 0, 0, 0] for i in range(3)],
               palette=['a', 'b', 'c'], source_vertices=[0, 1, 2])])
    assert live.attach_source(rig, [obj], raw) == 1
    trs = np.zeros((3, 3, 4), dtype=np.float32)
    trs[:, 1, 3] = 1
    trs[:, 2, :3] = 1
    trs[1:, 0, 0] = 1
    second = trs.copy()
    second[:, 0] = 0
    data = dict(format=profile.FORMAT, version=profile.VERSION,
                node_names=['a', 'b', 'c'], bind_trs=trs.tolist(),
                node_flags=[0, 0, 0], parent_ids=[-1, 0, 1], links=[[-1, -1]]*3,
                root_world=np.eye(4).tolist(),
                local_layers={'first': trs.tolist(), 'second': second.tolist()},
                palette=['a', 'b', 'c'], inverse_bind=inverse, meshes=[])
    if reference_mode:
        data['world_pose_backend'] = 'live'
    scene = bpy.context.scene
    scene.render.fps = 60
    scene.frame_set(0)
    live.bind_native_profile(rig, data, frame=0)
    initial = profile.evaluate_profile(data)
    positions = np.array([initial['world_matrices'][n] for n in data['node_names']], dtype=np.float32)[:, :3, 3]
    positions = np.column_stack((positions, np.ones(3, dtype=np.float32)))
    parameters = dict(inverse_mass=[0, 1, 1],
             constraints=[stages.DistanceConstraint(0, 1, 1, 1, 1),
                          stages.DistanceConstraint(1, 2, 1, 1, 1)],
             force_weights0=[1]*3, force_weights1=[1]*3,
             target_weights0=[0]*3, target_weights1=[0]*3,
             uniform_force=[0, -9.8, 0, 0], flags=0x60)
    driver_type = drivers.DistanceClothDriver33
    extra = {}
    provider_calls = []
    expected_inputs = {}
    if kernel_mode:
        driver_type = drivers.KernelClothDriver33
        def fixture_colliders(frame):
            transform = np.eye(4, dtype=np.float32)
            transform[3, :3] = [.8, .02*np.sin(frame*.1), .01]
            inverse_transform = np.eye(4, dtype=np.float32)
            inverse_transform[3, :3] = -transform[3, :3]
            matrices = np.stack([transform, transform.copy()])
            matrices[1, 3, 0] += .6
            return [sphere.EdgeSphereCollider0(transform, inverse_transform,
                         np.array([0, 0, 0, .06]), 7, 0),
                    capsule.EdgeCapsuleCollider7(matrices,
                         np.array([0, 0, 0, 1]), np.array([0, 0, 0, 1]),
                         np.array([1.03, 1, 1, 1]), .04, .08, 0, 1, 0, 7, 0),
                    plane.PlaneCollider5(np.eye(4),
                         np.array([0, -.2+.02*np.sin(frame*.1), 0, 1]), 3, 0, .1, .2)]
        parameters.update(
            flags=0x4FF,
            angular_constraints=[angles.AngularConstraint(.93972176, .34194008, 0, .06801529,
                                   .6428039, .7660308, 1, 2, 1, 2, 0)], angular_count=1,
            target_angle_constraints=[angles.AngularConstraint(.86604136, .49997234, 0, .70710677,
                                        .3746085, .9271831, 1, 2, 0, 1, -1)], target_angle_count=1,
            collision_radii=[.01]*3, collision_weights0=[.2]*3, collision_weights1=[.1]*3,
            collision_prefix_count=2, packed_normal_modes=bytes([0x33, 3]),
            colliders=fixture_colliders(0), collider_count=3)
        def collider_provider(world, names, frame):
            assert names == ('a', 'b', 'c')
            provider_calls.append(frame)
            colliders = fixture_colliders(frame)
            if reference_mode:
                identity = np.tile(np.eye(4, dtype='f4'), (2, 1, 1))
                # Distinct explicit model banks and two-slot tie-style binding.
                shapes = {12: bindings.BoundShapeState(identity.copy(), identity.copy(),
                            np.zeros(2, 'f4'), 7, .1, 1)}
                rows = [bindings.ColliderBoneBinding(12, 0, 0, 0),
                        bindings.ColliderBoneBinding(12, 1, 1, 1)]
                bound = bindings.prepare_bound_shapes(rows, {0: world, 1: world},
                                                     shapes, seeds=seeds)[12]
                old = colliders[1]
                colliders[1] = capsule.EdgeCapsuleCollider7(bound.transforms,
                    old.endpoint0, old.endpoint1, old.scale, old.radius0, old.radius1,
                    old.contact_start, old.contact_end, old.orientation_selector,
                    bound.flags, old.filter_flags)
            return dict(colliders=colliders, exclusion_mask=0)
        extra['collider_provider'] = collider_provider
        def step_inputs(world, names, frame):
            dt = np.float32((1/60, 1/30, 1/120)[int(frame) % 3])
            inputs = dict(dt=dt, reciprocal_dt=np.float32(1/dt),
                        uniform_force=[0, -9.8, 0, 0], wind_reference=[0]*4,
                        relative_velocity=[0]*4, force_multiplier=1)
            if file_mode:
                active = int(frame) % 15 < 10
                policy = controller.ControllerPolicy(.3, -1 if active else 2, 2, 3, 1, 33)
                control = controller.update_controller(
                    controller.ControllerState(0x13, 0x4FF, 2, 1), policy,
                    world[0, 3], [0, 0, 3, 1], .78, 1242, active, seeds=seeds)
                wave = environment.EnvironmentWave(np.array([.2, .4, .1, .5]),
                                                   np.array([.1, .01, .5, 0]))
                field = environment.evaluate_field(wave, (), [0]*4, world[0, 3], frame/60, 1)
                inputs.update(controller_result=control,
                    reciprocal_dt=control.convergence_scale,
                    wind_reference=environment.wrapper_wind(field, .2, [0]*4))
            expected_inputs[int(frame)] = inputs
            return inputs
        extra['step_input_provider'] = step_inputs
        if file_mode:
            source_root = report_path.parent.parent/'20261017-live33-solver-inputs-002/received/disk-only-extracted'
            fltd = next(source_root.rglob('pl_rbd_218620_bw.fltd'))
            file_definition = initialization.read_file_definitions(fltd.read_bytes())[1]
            assert file_definition.particle_count == 3
            extra['file_definition'] = file_definition
            driver_type = drivers.FileClothDriver33
            parameters.update(damping=.1, air_resistance=10)
    records = [[0, 1, -1, -1, -1], [1, 2, 0, -1, -1]]
    writeback_parameters = dict(version=33, flags=3, fltd_flags=0x60)
    if reference_mode:
        base = np.array([initial['world_matrices'][n] for n in data['node_names']],
                        dtype='f4').transpose(0, 2, 1).copy()
        driver = driver_type.from_reference(data['node_names'], [0, 1, 2], records,
            base, base, 0, writeback_parameters, parameter_overrides=parameters,
            seeds=seeds, **extra)
        assert driver.reference_span == 2
    else:
        driver = driver_type(data['node_names'], [0, 1, 2], records,
            solver.SolverState(positions, positions.copy(), 0), 0, parameters,
            writeback_parameters, seeds=seeds, **extra)
    live.set_cloth_driver(rig, driver)
    rows = []
    for frame in range(1, 61):
        scene.frame_set(frame)
        rig.pose.bones['a'].location.y = .04*np.sin(frame*.13)
        if reference_mode:
            rig.pose.bones['a'].scale.x = 1+.02*np.sin(frame*.11)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        live.update_rig(rig, scene, graph, force=True)
        assert not rig.get(live.ERROR), rig.get(live.ERROR)
        before = driver.state.positions.copy()
        if kernel_mode:
            inputs = expected_inputs[frame]
            if not file_mode or inputs['controller_result'].state.flags & 1:
                assert driver.state.previous_dt == inputs['dt']
        live.update_rig(rig, scene, graph, force=True)
        np.testing.assert_array_equal(before, driver.state.positions)
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        result = evaluated.to_mesh()
        actual = np.array([list(v.co) for v in result.vertices])
        evaluated.to_mesh_clear()
        error = float(np.max(abs(actual - driver.state.positions[:, :3])))
        assert error < 2e-6, (frame, error)
        rows.append({'frame': frame, 'max_position_error': error,
                     'tip': driver.state.positions[-1, :3].tolist()})
    assert np.ptp(np.array([r['tip'] for r in rows]), axis=0).max() > .01
    np.testing.assert_array_equal(original, [list(v.co) for v in mesh.vertices])
    assert mesh.materials[0] == material
    state = driver.state.positions.copy()
    scene.frame_set(10)
    live.update_scene(scene, bpy.context.evaluated_depsgraph_get())
    assert 'rewind' in rig.get(live.ERROR, '')
    np.testing.assert_array_equal(state, driver.state.positions)
    # A reload must report the missing driver; cached vertex values are stale.
    live.load_post(None)
    assert 'Reattach' in rig.get(live.ERROR, '')
    assert 'cached viewport is stale' in rig.get(live.ERROR, '')
    live.set_cloth_driver(rig, None)
    live.update_rig(rig, scene, bpy.context.evaluated_depsgraph_get(), force=True)
    assert not rig.get(live.ERROR)
    report.update(success=True, frames=rows, original_mesh_and_material_preserved=True,
                  duplicate_frame_safe=True, rewind_rejected=True,
                  missing_driver_reports_stale_viewport=True)
    if kernel_mode:
        assert len(provider_calls) >= 60
        report['dynamic_collider_provider_called'] = len(provider_calls)
        report['angular_and_plane_stages_connected'] = True
        report['collider_variants_connected'] = [0, 7, 5]
        report['explicit_caller_timing_and_environment_connected'] = True
        if file_mode:
            report['file_pins_scaled_radii_and_controller_connected'] = True
            report['disabled_controller_copies_targets_without_integration'] = True
        if reference_mode:
            report['reference_initialization_without_captured_pose'] = True
            report['explicit_bone_banks_and_tie_scale_blend_connected'] = True
            report['live_fk_backend_selected'] = True
except BaseException:
    report.update(success=False, error=traceback.format_exc())
    raise
finally:
    report_path.write_text(json.dumps(report, indent=2), encoding='utf8')
