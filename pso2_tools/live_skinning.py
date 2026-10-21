"""Persistent raw NGS influences and an affine viewport deformation path.

GPU_WEIGHTS uses Blender's final bones, so it proves weight semantics only.
NATIVE_INPUTS evaluates source local layers through the recovered FK kernel.
Supported FLTD imports may attach a persistent source-driven cloth preview.
Its scene timing/gravity policy is explicit; live dynamic parity is unverified.
No client executable, skeleton resource, rest mesh or vertex group is edited.
"""

import copy
import json
from dataclasses import asdict
from pathlib import Path

import bpy
import numpy as np
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix

from . import aqm, charfile, classes, import_aqm, native_pose_profile, scene_props, skinning_math

SOURCE = "pso2_raw_skin_source"
RIG = "pso2_skin_rig"
MODE = "pso2_skin_mode"
PROFILE = "pso2_skin_native_inputs"
BASELINE = "pso2_skin_pose_baseline"
ERROR = "pso2_skin_error"
MODIFIER = "PSO2 Affine Skinning"
POSITION = "pso2_skin_position"
_CACHE = {}
_CLOTH_DRIVERS = {}
_BUSY = False


def set_cloth_driver(rig, driver):
    """Development API: attach an explicit stateful native-stage driver.

    driver.evaluate(world_rows, node_names, frame, fps) must handle duplicate
    frame evaluations and reject unsupported timeline jumps. A driver is not
    persisted in .blend and cannot be silently restored as a fixed pose.
    """
    if rig.get(MODE) != 'NATIVE_INPUTS':
        raise ValueError('A cloth driver requires Native Local Inputs')
    if driver is not None and not callable(getattr(driver, 'evaluate', None)):
        raise ValueError('Cloth driver must supply evaluate')
    key = rig.as_pointer()
    if driver is None:
        _CLOTH_DRIVERS.pop(key, None)
        if rig.get('pso2_skin_cloth_driver_required'):
            del rig['pso2_skin_cloth_driver_required']
    else:
        _CLOTH_DRIVERS[key] = driver
        rig['pso2_skin_cloth_driver_required'] = True
    _CACHE.pop(('pose', key), None)


def _matrix(value):
    return np.asarray(value, dtype=np.float32)


def capture_source(model, skeleton):
    """Mirror the FBX exporter's face-group/control-point mapping exactly.

    Do not infer weights from normalized FBX clusters or nearest positions.
    Classic float-weight formats are deliberately excluded from this path.
    """
    names = [str(skeleton.GetNodeName(i)) for i in range(skeleton.nodeList.Count)]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate AQN node names cannot identify a skin palette")
    inverse = {}
    for i, node in enumerate(skeleton.nodeList):
        rows = [[float(getattr(getattr(node, f'm{r}'), c)) for c in 'XYZW']
                for r in range(1, 5)]
        inverse[names[i]] = _matrix(rows).T.tolist()
    meshes = []
    for i, mesh in enumerate(model.meshList):
        vertices = model.vtxlList[mesh.vsetIndex]
        if not vertices.vertWeightsNGS.Count:
            continue
        palette = (model.bonePalette if model.objc.bonePaletteOffset > 0
                   else vertices.bonePalette)
        palette_names = [names[int(k)] for k in palette]
        for j, group in enumerate(model.strips[mesh.psetIndex].faceGroups):
            if not group:
                continue
            faces = model.strips[mesh.psetIndex].GetTrianglesFaceGroup(j, True)
            count = (faces.vertexMappingList.Count if faces.vertexMappingList is not None
                     else vertices.vertPositions.Count)
            mapping = [int(faces.GetVertexMapping(k)) for k in range(count)]
            record = {
                'name': str(model.GetMeshName(i, True, j)),
                'positions': [[float(getattr(vertices.vertPositions[k], c))
                               for c in 'XYZ'] for k in mapping],
                'weights_u16': [[int(w) for w in vertices.vertWeightsNGS[k]] for k in mapping],
                'bone_indices': [[int(w) for w in vertices.vertWeightIndices[k]] for k in mapping],
                'palette': palette_names,
                'source_vertices': mapping,
            }
            # Validate all slots, even a zero-weight slot, before changing a scene.
            skinning_math.skin_positions(record['positions'], record['weights_u16'],
                                         record['bone_indices'], palette_names, inverse)
            meshes.append(record)
    return {'format': 'pso2_raw_skin_source', 'version': 1, 'node_names': names,
            'inverse_bind': inverse, 'meshes': meshes}


# Texts already read during the running viewport update. Nothing writes them
# inside one pass, and as_string() on the multi-megabyte inputs was read about
# fourteen times a frame.
_PASS = None


def _read_text(name):
    if _PASS is not None and name in _PASS:
        return _PASS[name]
    text = bpy.data.texts.get(name)
    if text is None:
        raise ValueError(f"Missing embedded skin inputs: {name}")
    value = text.as_string()
    cached = _CACHE.get(name)
    if cached and cached[0] == value:
        result = cached[1]
    else:
        result = json.loads(value)
        _CACHE[name] = (value, result)
    if _PASS is not None:
        _PASS[name] = result
    return result


def _new_text(name, value):
    text = bpy.data.texts.new(name)
    # Blender's Text.write has pathological cost for a multi-megabyte single line.
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    text.from_string(payload.replace('],', '],\n'))
    return text.name


def _bone_names(rig, names):
    by_id = {int(b.bone[scene_props.BONE_ID]): b.name for b in rig.pose.bones
             if scene_props.BONE_ID in b.bone}
    # Names in the AQN are not necessarily the FBX metadata-suffixed names.
    return {name: by_id[i] for i, name in enumerate(names) if i in by_id}


def _deformer(obj):
    existing = obj.modifiers.get(MODIFIER)
    if existing:
        if (existing.type != 'NODES' or existing.node_group is None
                or not existing.node_group.get('pso2_affine_skinning')):
            raise ValueError(f"Modifier name is already in use: {obj.name}")
        return existing
    group = bpy.data.node_groups.new(MODIFIER, 'GeometryNodeTree')
    # Arbitrary modifier ID properties do not survive Blender file reload.
    group['pso2_affine_skinning'] = True
    group.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    group.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    source = group.nodes.new('NodeGroupInput')
    output = group.nodes.new('NodeGroupOutput')
    position = group.nodes.new('GeometryNodeInputNamedAttribute')
    position.data_type = 'FLOAT_VECTOR'
    position.inputs['Name'].default_value = POSITION
    apply = group.nodes.new('GeometryNodeSetPosition')
    group.links.new(source.outputs['Geometry'], apply.inputs['Geometry'])
    group.links.new(position.outputs['Attribute'], apply.inputs['Position'])
    group.links.new(apply.outputs['Geometry'], output.inputs['Geometry'])
    modifier = obj.modifiers.new(MODIFIER, 'NODES')
    modifier.node_group = group
    # Skin before subdivision/other downstream modifiers, at the armature's position.
    index = next((i for i, m in enumerate(obj.modifiers) if m.type == 'ARMATURE'), 0)
    obj.modifiers.move(len(obj.modifiers) - 1, index)
    return modifier


def attach_source(rig, mesh_objects, data, *, enabled=True):
    """Attach only after exact ordered control-point and bone identity checks."""
    if rig.type != 'ARMATURE':
        raise ValueError('Select the imported armature')
    if rig.get(scene_props.BONE_AXES) == 'AUTO':
        raise ValueError('Re-import without Automatic Bone Orientation')
    # FBX has just assigned parents; matrix_world is stale until evaluation.
    bpy.context.view_layer.update()
    bone_names = _bone_names(rig, data['node_names'])
    pending = []
    available = list(mesh_objects)
    for record in data['meshes']:
        # FBX suffixes duplicates on a second import; mesh datablocks retain the source name.
        candidates = [o for o in available if o.type == 'MESH' and
                      (o.name == record['name'] or o.name.startswith(record['name'] + '.'))]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous imported face group: {record['name']}")
        obj = candidates[0]
        positions = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
        obj.data.vertices.foreach_get('co', positions)
        raw = _matrix(record['positions'])
        if positions.size != raw.size or not np.allclose(positions.reshape(-1, 3), raw, atol=2e-6, rtol=0):
            raise ValueError(f"FBX vertex order/coordinates changed: {obj.name}")
        if any(name not in bone_names and name != data["node_names"][0] for name in record['palette']):
            raise ValueError(f"A skin palette node is missing from the imported rig: {obj.name}")
        if obj.data.users != 1 or obj.get(SOURCE):
            raise ValueError(f"Skin inputs already attached or mesh data is shared: {obj.name}")
        if obj.modifiers.get(MODIFIER):
            raise ValueError(f"Reserved skin modifier already exists: {obj.name}")
        if obj.data.attributes.get(POSITION):
            raise ValueError(f"Reserved skin attribute already exists: {obj.name}")
        pending.append((obj, record))
    if not pending:
        return 0
    inputs = copy.deepcopy(data)
    inputs['blender_bones'] = bone_names
    # Object-space calibration is explicit, not inferred from current world placement.
    inputs['rig_bind_world'] = [list(r) for r in rig.matrix_world]
    text = _new_text('PSO2 Raw Skin Inputs.json', inputs)
    rig[SOURCE] = text
    for index, (obj, record) in enumerate(pending):
        obj[SOURCE] = text
        obj[RIG] = rig
        obj['pso2_skin_mesh'] = index
        obj['pso2_skin_mesh_bind_world'] = [list(r) for r in obj.matrix_world]
        obj['pso2_skin_saved_armature'] = json.dumps([
            [m.name, m.show_viewport, m.show_render] for m in obj.modifiers if m.type == 'ARMATURE'
        ])
        attr = obj.data.attributes.new(POSITION, 'FLOAT_VECTOR', 'POINT')
        attr.data.foreach_set('vector', np.asarray(record['positions'], dtype=np.float32).ravel())
        _deformer(obj)
    rig[MODE] = 'GPU_WEIGHTS' if enabled else 'BLENDER'
    set_mode(rig, rig[MODE])
    return len(pending)


def _meshes(rig):
    return [o for o in bpy.data.objects if o.type == 'MESH' and o.get(RIG) == rig
            and o.get(SOURCE) == rig.get(SOURCE)]


def set_mode(rig, mode):
    if mode not in {'BLENDER', 'GPU_WEIGHTS', 'NATIVE_INPUTS'}:
        raise ValueError('Unknown skinning mode')
    if mode == 'NATIVE_INPUTS' and not rig.get(PROFILE):
        raise ValueError('Bind source native pose inputs first')
    targets = [(obj, _deformer(obj)) for obj in _meshes(rig)]
    rig[MODE] = mode
    for obj, modifier in targets:
        modifier.show_viewport = modifier.show_render = mode != 'BLENDER'
        for name, viewport, render in json.loads(obj['pso2_skin_saved_armature']):
            if armature := obj.modifiers.get(name):
                armature.show_viewport = viewport if mode == 'BLENDER' else False
                armature.show_render = render if mode == 'BLENDER' else False
    _CACHE.pop(('pose', rig.as_pointer()), None)
    update_rig(rig, bpy.context.scene, bpy.context.evaluated_depsgraph_get(), force=True)


def _native_channel_deltas(rig, inputs):
    """Editable local channels layered onto source motion, before native FK.

    Parent evaluated world matrices never enter this conversion. Thus Blender
    parent scale handling cannot contaminate the recovered FK calculation.
    Bone constraints are not folded into these explicit local channels.
    """
    baseline = json.loads(rig.original[BASELINE])
    correction = import_aqm.bone_correction(rig).to_3x3()
    result = np.zeros((len(inputs['node_names']), 3, 4), dtype=np.float32)
    result[:, 1, 3] = 1
    result[:, 2, :3] = 1
    for i, name in enumerate(inputs['node_names']):
        bone_name = inputs['blender_bones'].get(name)
        if bone_name is None:
            continue
        pb = rig.pose.bones[bone_name]
        old = Matrix(baseline[bone_name])
        current = pb.matrix_basis
        rest = (pb.bone.parent.matrix_local.inverted() @ pb.bone.matrix_local
                if pb.parent else pb.bone.matrix_local)
        parent_c = correction if pb.parent else Matrix.Identity(3)
        position = parent_c @ rest.to_quaternion().to_matrix() @ (current.translation - old.translation)
        delta = old.to_quaternion().inverted() @ current.to_quaternion()
        rotation = (correction @ delta.to_matrix() @ correction.inverted()).to_quaternion()
        old_scale = np.asarray(old.to_scale())
        if (abs(old_scale) < 1e-8).any():
            raise ValueError(f"Zero baseline scale on {bone_name}")
        ratio = np.asarray(current.to_scale()) / old_scale
        scale = correction @ Matrix.Diagonal(ratio.tolist()) @ correction.inverted()
        result[i, 0, :3] = position
        result[i, 1] = [rotation.x, rotation.y, rotation.z, rotation.w]
        result[i, 2, :3] = [scale[k][k] for k in range(3)]
    return result.tolist()


def bind_native_profile(rig, data, *, frame=1):
    """Match by source bone identities AND inverse binds, never vertex proximity."""
    inputs = _read_text(rig[SOURCE])
    reference = data['inverse_bind']
    needed = {n for m in inputs['meshes'] for n in m['palette']}
    if not needed.issubset(reference):
        raise ValueError('Native profile does not supply this model skin palette')
    for name in needed:
        if not np.allclose(reference[name], inputs['inverse_bind'][name], atol=2e-6, rtol=0):
            raise ValueError(f"Native profile uses a different inverse bind: {name}")
    if not set(inputs['node_names']).issubset(data['node_names']):
        raise ValueError('Native profile does not supply this imported AQN node set')
    value = copy.deepcopy(data)
    # Profiles built from the import template leave model space at the
    # identity; place it by where the import put this rig (head parts).
    if np.allclose(value.get('model_transform', np.eye(4)), np.eye(4)) and inputs.get('rig_bind_world'):
        value['model_transform'] = skinning_math.native_model_transform(inputs['rig_bind_world']).tolist()
    value['meshes'] = []  # evaluate matrices once; meshes keep their original material/UV data
    value['bind_tail_fallback'] = True  # unanimated costume nodes, not a physics solver
    native_pose_profile.evaluate_profile(value)
    rig[PROFILE] = _new_text('PSO2 Driven Native Inputs.json', value)
    rig[BASELINE] = json.dumps({p.name: [list(r) for r in p.matrix_basis] for p in rig.pose.bones})
    rig['pso2_skin_frame_origin'] = float(frame)
    rig['pso2_skin_time_scale'] = 1.0
    rig['pso2_skin_live_runtime_verified'] = False
    set_mode(rig, 'NATIVE_INPUTS')


def update_rig(rig, scene, depsgraph, *, force=False):
    mode = rig.get(MODE, 'BLENDER')
    if mode == 'BLENDER':
        return
    inputs = _read_text(rig[SOURCE])
    evaluated = rig.evaluated_get(depsgraph)
    signature = (mode, scene.frame_current_final, rig.get(PROFILE),
                 rig[SOURCE], hash(_CACHE[rig[SOURCE]][0]),
                 rig.get('pso2_skin_time_scale', 1.0),
                 tuple(float(v) for row in evaluated.matrix_world for v in row),
                 tuple(float(v) for p in evaluated.pose.bones
                       for row in (p.matrix if mode == 'GPU_WEIGHTS' else p.matrix_basis) for v in row))
    cache_key = ('pose', rig.as_pointer())
    # Also invalidate when embedded inputs are edited.
    if mode == 'NATIVE_INPUTS':
        signature += (bpy.data.texts[rig[PROFILE]].as_string(),)
        from . import cloth_auto
        driver = cloth_auto.ensure_driver(rig,scene,_read_text(rig[PROFILE]))
        if rig.get('pso2_skin_cloth_driver_required') and driver is None:
            raise ValueError('Reattach the native cloth driver after loading; cached viewport is stale')
        signature += (id(driver), getattr(driver, 'revision', 0),
                      scene.render.fps, scene.render.fps_base,
                      bool(scene.use_gravity), tuple(scene.gravity))
        # Colliders on the body (bank 1) move with the body's pose.
        signature += cloth_auto.related_signature(rig, scene, depsgraph)
    if not force and _CACHE.get(cache_key) == signature:
        return
    matrices = {}
    if mode == 'GPU_WEIGHTS':
        root_name = inputs['node_names'][0]
        matrices[root_name] = _matrix(evaluated.matrix_world) @ _matrix(inputs['inverse_bind'][root_name])
        correction = import_aqm.bone_correction(rig)
        # Undo the importer bone-axis correction, retain affine/sheared world matrices.
        for name, bone in inputs['blender_bones'].items():
            world = evaluated.matrix_world @ evaluated.pose.bones[bone].matrix @ correction.inverted()
            matrices[name] = _matrix(world) @ _matrix(inputs['inverse_bind'][name])
    else:
        # evaluate_profile only reads its inputs: copy just what changes here
        # (a deep copy of the whole profile cost ~15 ms a frame).
        data = dict(_read_text(rig[PROFILE]))
        if data.get('first_motion'):
            data['first_motion'] = dict(data['first_motion'])
            data['first_motion']['frame'] += ((scene.frame_current_final - rig['pso2_skin_frame_origin'])
                                              * rig.get('pso2_skin_time_scale', 1.0))
        # Runtime NODE arrays may be reordered relative to file node IDs.
        delta_inputs = dict(inputs, node_names=data['node_names'])
        data['local_pose_deltas'] = _native_channel_deltas(evaluated, delta_inputs)
        cloth_step = None
        if driver is not None:
            cloth_step = lambda world, names: cloth_auto.evaluate(driver,world,names,
                float(scene.frame_current_final),scene,evaluated.matrix_world,data,
                rig=rig,depsgraph=depsgraph)
        result = native_pose_profile.evaluate_profile(data, cloth_step=cloth_step)
        matrices = {n: _matrix(evaluated.matrix_world) @ _matrix(m)
                    for n, m in result['skin_matrices'].items()}
    # Validate every target before writing any position attribute.
    pending = []
    for obj in _meshes(rig):
        if obj.mode == 'EDIT':
            raise ValueError('Leave Edit Mode before evaluating raw skin inputs')
        record = inputs['meshes'][obj['pso2_skin_mesh']]
        attr = obj.data.attributes.get(POSITION)
        if attr is None or len(attr.data) != len(record['positions']):
            raise ValueError(f"Vertex count changed; re-import original skin inputs: {obj.name}")
        # The imported object may have independent placement relative to its rig.
        bind_mesh = _matrix(obj['pso2_skin_mesh_bind_world'])
        bind_rig = _matrix(inputs['rig_bind_world'])
        relative_bind = np.linalg.inv(bind_rig) @ bind_mesh
        # Rig/model coordinates -> local mesh coordinates; object animation stays intact.
        local = np.linalg.inv(_matrix(evaluated.matrix_world) @ relative_bind)
        palette_matrices = {n: local @ matrices[n] for n in record['palette']}
        positions = skinning_math.skin_positions(record['positions'], record['weights_u16'],
                                                record['bone_indices'], record['palette'], palette_matrices)
        pending.append((obj, attr, positions))
    for obj, attr, positions in pending:
        attr.data.foreach_set('vector', positions.ravel())
        obj.data.update()
    _CACHE[cache_key] = signature
    if rig.get(ERROR):
        del rig[ERROR]


@persistent
def update_scene(scene, depsgraph=None):
    global _BUSY, _PASS
    if _BUSY:
        return
    _BUSY = True
    _PASS = {}
    try:
        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        for rig in scene.objects:
            if rig.type != 'ARMATURE' or not rig.get(SOURCE) or rig.get(MODE) == 'BLENDER':
                continue
            try:
                update_rig(rig, scene, depsgraph)
            except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
                message = str(error)
                if rig.get(ERROR) != message:
                    rig[ERROR] = message
                    print(f"PSO2 Skinning [{rig.name}]: {message}")
                # Never silently switch to normalized Blender weights on failure.
    finally:
        _BUSY = False
        _PASS = None


@persistent
def load_post(_):
    _CACHE.clear()
    _CLOTH_DRIVERS.clear()
    update_scene(bpy.context.scene)


def register_handlers():
    for handlers, function in ((bpy.app.handlers.depsgraph_update_post, update_scene),
                               (bpy.app.handlers.frame_change_post, update_scene),
                               (bpy.app.handlers.load_post, load_post)):
        if function not in handlers:
            handlers.append(function)


def unregister_handlers():
    for handlers, function in ((bpy.app.handlers.depsgraph_update_post, update_scene),
                               (bpy.app.handlers.frame_change_post, update_scene),
                               (bpy.app.handlers.load_post, load_post)):
        if function in handlers:
            handlers.remove(function)
    _CACHE.clear()
    _CLOTH_DRIVERS.clear()


@classes.register
class PSO2_OT_SkinMode(bpy.types.Operator):
    bl_idname = 'pso2.skin_mode'
    bl_label = 'Set PSO2 Skinning Mode'
    bl_options = {'UNDO'}
    mode: bpy.props.EnumProperty(items=[('BLENDER', 'Blender', ''),
                                       ('GPU_WEIGHTS', 'Raw NGS Weights', ''),
                                       ('NATIVE_INPUTS', 'Native Local Inputs', '')], translation_context="PSO2Tools")

    def execute(self, context):
        rig = import_aqm._find_target_armature(context)
        try:
            if rig is None or not rig.get(SOURCE):
                raise ValueError('Re-import the NGS AQP with Preserve NGS Weights enabled')
            set_mode(rig, self.mode)
        except (ValueError, KeyError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


@classes.register
class PSO2_OT_BindNativeSkinning(bpy.types.Operator, ImportHelper):
    """Drive an imported mesh from source local layers and recovered native FK"""
    bl_idname = 'pso2.bind_native_skinning'
    bl_label = 'Bind Native Pose Inputs to Imported Model'
    bl_options = {'UNDO'}
    filename_ext = '.json'
    filter_glob: bpy.props.StringProperty(default='*.native-pose.json', options={'HIDDEN'})

    def execute(self, context):
        rig = import_aqm._find_target_armature(context)
        try:
            if rig is None or not rig.get(SOURCE):
                raise ValueError('Re-import the NGS AQP with Preserve NGS Weights enabled')
            data, _ = native_pose_profile.load_profile(self.filepath)
            bind_native_profile(rig, data, frame=context.scene.frame_current_final)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}


@classes.register
class PSO2_OT_SkinSource(bpy.types.Operator, ImportHelper):
    """Replace a native input layer while retaining the original outfit inverse binds"""
    bl_idname = 'pso2.skin_source'
    bl_label = 'Load Character or Motion for Native Skinning'
    bl_options = {'UNDO'}
    filename_ext = '.fnp'
    filter_glob: bpy.props.StringProperty(
        default='*.fnp;*.mhp;*.fhp;*.fcp;*.mcp;*.aqm', options={'HIDDEN'})
    frame: bpy.props.FloatProperty(name='Motion Frame', default=0, min=0, translation_context="PSO2Tools")
    in_place: bpy.props.BoolProperty(name='In Place', default=True, translation_context="PSO2Tools")

    def execute(self, context):
        rig = import_aqm._find_target_armature(context)
        try:
            if rig is None or not rig.get(PROFILE):
                raise ValueError('Bind source native pose inputs to this model first')
            data = copy.deepcopy(_read_text(rig[PROFILE]))
            path = Path(self.filepath)
            if path.suffix.lower() == '.aqm':
                motion = aqm.read_aqm(path)
                if (motion.is_camera_motion or motion.is_material_motion
                        or motion.is_shape_adjust or not motion.nodes):
                    raise ValueError('Choose a skeletal AQM motion')
                data['first_motion'] = {
                    'motion': asdict(motion), 'frame': self.frame,
                    'source_name_map': {motion.nodes[0].name: data['node_names'][0]},
                    'root_motion_flags': 15, 'apply_extracted_root': not self.in_place,
                    'source_path': str(path.resolve()),
                }
            else:
                if not data.get('body_source'):
                    raise ValueError('This profile has no source body provider')
                character = charfile.CharacterFile.load(path)
                if (data['body_source']['character_fields']['baseDOC.gender']
                        != character['baseDOC.gender']):
                    raise ValueError('This character requires another body-type profile')
                data['body_source']['character_fields'] = character.to_dict()
                data['body_source']['character_path'] = str(path.resolve())
            native_pose_profile.evaluate_profile(data)
            rig[PROFILE] = _new_text('PSO2 Driven Native Inputs.json', data)
            rig['pso2_skin_frame_origin'] = context.scene.frame_current_final
            set_mode(rig, 'NATIVE_INPUTS')
        except (OSError, ValueError, KeyError, TypeError, aqm.AqmError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        return {'FINISHED'}
