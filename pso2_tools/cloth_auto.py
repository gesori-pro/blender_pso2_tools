"""Automatic supported FLTD preview. Scene inputs are never labelled live inputs."""
import copy
import json
from dataclasses import asdict

import bpy
import numpy as np

from . import classes, cloth_assets, import_aqm, live_skinning, native_pose_profile, native_pose_sources
from .native_cloth_file import declares_cloth, read_parts
from .native_cloth_preview import FileClothPreview33
from .native_float_estimates import HardwareEstimates

AUTO = 'pso2_cloth_auto'
STATUS = 'pso2_cloth_status'
SETTLE = 'pso2_cloth_settle_steps'
CHARACTER_BANK = ('character_top',)
_BLENDER_TO_NATIVE =np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], 'f4')


def skeleton_template(skeleton, raw):
    """Retain file NODE/TRS, without a runtime/final-pose dependency."""
    from AquaModelLibrary.Helpers.MathHelpers import MathExtras, RotationOrder
    names = raw['node_names']
    bind, flags, parents, links = [], [], [], []
    for node in skeleton.nodeList:
        # NODE Euler angles use XZY. The helper defaults to XYZ, which
        # agrees on single-axis joints but twists compound-rotation joints
        # (pelvis/clavicles) before the solver even runs. Keep the source
        # Euler fields, rather than deriving a rest pose from edited binds.
        q = MathExtras.EulerToQuaternion(node.eulRot, RotationOrder.XZY)
        bind.append([[float(getattr(node.pos,c)) for c in 'XYZ']+[0],
                     [float(getattr(q,c)) for c in 'XYZW'],
                     [float(getattr(node.scale,c)) for c in 'XYZ']+[0]])
        # boneShort2 is file metadata (motion exclusion/socket IDs), NOT
        # the upper half of the runtime FK flags. All 199 source low words
        # match the observed runtime NODE flag words for this resource.
        flags.append(int(node.boneShort1))
        parents.append(int(node.parentId))
        # Runtime auxiliary link slots are constructed by the resource loader;
        # file unkNode is not their packed representation. These slots are
        # unused by the supported ordinary pass (no 02000000 flag).
        links.append([0,0])
    second = np.zeros((len(names),3,4), 'f4')
    second[:,1,3] = 1
    second[:,2,:3] = 1
    return dict(format=native_pose_profile.FORMAT,version=native_pose_profile.VERSION,
        source='Source AQN NODE/TRS and inverse binds',source_aqn_euler_order='XZY',
        local_input_origin='File AQN plus explicit source/Blender channel inputs; no final pose',
        node_names=names,node_flags=flags,parent_ids=parents,links=links,bind_trs=bind,
        root_world=np.eye(4).tolist(),model_transform=np.eye(4).tolist(),
        local_layers=dict(first=bind,second=second.tolist()),inverse_bind=raw['inverse_bind'],
        palette=list(dict.fromkeys(n for m in raw['meshes'] for n in m['palette'])),
        meshes=[],bind_tail_fallback=True,world_pose_backend='live')


def _world(data, *, reference=False):
    value = copy.deepcopy(data)
    value['meshes'] = []
    value['model_transform'] = np.eye(4).tolist()
    value.pop('local_pose_deltas',None)
    if reference:
        value.pop('first_motion',None)
        value['local_layers']['first'] = value['bind_trs'][:len(value['local_layers']['first'])]
    result = native_pose_profile.evaluate_profile(value)
    return np.array([result['world_matrices'][n] for n in value['node_names']], 'f4').transpose(0,2,1).copy()


def _base(name):
    return name.split('#')[0]


def find_body_rig(rig, scene):
    """The character's body armature, the cloth's parent resource (bank 1).

    A model whose own skeleton has body_root is the body itself. Otherwise
    (hair, ears, accessories, imported as their own armatures) the scene's
    armature carrying body_root is used, as the character import lays them
    out; None when there is none."""
    if any(_base(b.name) == 'body_root' for b in rig.pose.bones):
        return rig
    bodies = sorted((o for o in scene.objects if o.type == 'ARMATURE' and o is not rig
                     and any(_base(b.name) == 'body_root' for b in o.pose.bones)), key=lambda o: o.name)
    return bodies[0] if bodies else None


def _body_bones(body):
    """(native node name, Blender bone name or None for the model root) pairs."""
    if body.get(live_skinning.SOURCE):
        inputs = live_skinning._read_text(body[live_skinning.SOURCE])
        bones = inputs.get('blender_bones', {})
        names = inputs['node_names']
        return [(n, bones.get(n)) for i, n in enumerate(names) if n in bones or i == 0]
    return [(_base(b.name), b.name) for b in body.pose.bones]


SIBLING_BANKS = 7    # source banks 3..9


def sibling_rigs(rig, body, scene):
    """Banks 3..: the character's other imported models (28BFE30 searches the
    sibling resources after the parent and grandparent), e.g. the basewear
    whose tube-skirt bones an outerwear's colliders name. Scene armatures with
    PSO2 source inputs, those sharing a collection with this model first."""
    found = [o for o in scene.objects if o.type == 'ARMATURE' and o is not rig and o is not body
             and o.get(live_skinning.SOURCE)]
    collections = set(rig.users_collection)
    same = [o for o in found if collections & set(o.users_collection)]
    return sorted(same or found, key=lambda o: o.name)[:SIBLING_BANKS]


def related_bank(rig, scene):
    """Source objects and node names per bank: 1 = the character's top node
    and the body's nodes, 3.. = sibling models' nodes."""
    body = find_body_rig(rig, scene)
    pairs = [] if body is None or body is rig else _body_bones(body)
    names = [n for n, _ in pairs if n != CHARACTER_BANK[0]]
    sources = {1: body}
    banks = {1: CHARACTER_BANK + tuple(dict.fromkeys(names))}
    for i, other in enumerate(sibling_rigs(rig, body, scene)):
        sources[3 + i] = other
        banks[3 + i] = tuple(dict.fromkeys(n for n, _ in _body_bones(other)))
    return sources, banks


def _rows(to_model, source, names, lookup, correction):
    out = np.tile(np.eye(4), (len(names), 1, 1))
    world = np.asarray(source.matrix_world, dtype='f8')
    for i, name in enumerate(names):
        bone = lookup.get(name)
        matrix = world
        if bone is not None and bone in source.pose.bones:
            matrix = world @ np.asarray(source.pose.bones[bone].matrix, dtype='f8') @ np.linalg.inv(correction)
        out[i] = to_model @ matrix
    # column (Blender) matrices -> native rows, as _world() hands the solver
    return out.transpose(0, 2, 1).astype('f4')


def related_worlds(rig, sources, banks, data):
    """Native world rows of every related bank, in this model's native space.

    character_top is the body armature's origin; bones follow their model's
    evaluated pose (world @ pose bone @ inverse bone-axis correction, as the
    GPU-weights mode reads skin matrices). Scene inputs, not live parents."""
    model = np.asarray(data.get('model_transform', np.eye(4)), dtype='f8')
    rig_world = np.asarray(rig.matrix_world, dtype='f8')
    to_model = np.linalg.inv(model) @ np.linalg.inv(rig_world)
    out = {}
    for bank, names in banks.items():
        source = sources.get(bank)
        if source is None:
            out[bank] = np.tile(np.eye(4, dtype='f4'), (len(names), 1, 1))
            continue
        lookup = dict(_body_bones(source)) if source is not rig else {}
        correction = np.asarray(import_aqm.bone_correction(source), dtype='f8') if lookup else np.eye(4)
        out[bank] = _rows(to_model, source, names, lookup, correction)
    return out


def related_signature(rig, scene, depsgraph):
    sources, _banks = related_bank(rig, scene)
    out = ()
    for bank in sorted(sources):
        source = sources[bank]
        if source is None or source is rig:
            continue
        source = source.evaluated_get(depsgraph)
        out += (bank, source.name, tuple(float(v) for row in source.matrix_world for v in row),
                tuple(float(v) for p in source.pose.bones for row in p.matrix for v in row))
    return out


def ensure_driver(rig, scene, data):
    """Restore from persisted SOURCE inputs, not serialized particle positions."""
    old = live_skinning._CLOTH_DRIVERS.get(rig.as_pointer())
    if not rig.get(AUTO):
        return old
    text = bpy.data.texts.get(rig.get(cloth_assets.SOURCE_KEY,''))
    if text is None:
        raise ValueError('Missing source FLTD; re-import the matching model archive')
    settle = int(rig.get(SETTLE,30))
    sources, banks = related_bank(rig, scene)
    key = (bpy.data.texts[rig[live_skinning.PROFILE]].as_string(), text.as_string(), settle,
           tuple((b, o.name if o is not None else None) for b, o in sorted(sources.items())),
           tuple(sorted(banks.items())))
    if isinstance(old,FileClothPreview33) and old.source_key == key:
        return old
    raw = cloth_assets.restore_file(json.loads(text.as_string()))
    # Collider bones missing from this model are looked up in its parent
    # resource (bank 1), as 28BFE30 does: the character's top node (floor
    # planes) and, for hair/accessories, the body's bones (head, neck, spine,
    # arms...). Their matrices come from the scene each frame (scene inputs).
    # Sibling models (banks 3..) give e.g. an outerwear the basewear's skirt
    # bones; a name found nowhere switches only that collider off, as in-game.
    parts = read_parts(raw,data['node_names'],data['parent_ids'],related=banks)
    candidate = FileClothPreview33(data['node_names'],parts,_world(data,reference=True),
        settle_steps=settle,seeds=HardwareEstimates(),
        related_worlds=related_worlds(rig, sources, banks, data))
    candidate.source_key = key
    candidate.related_bank = banks
    live_skinning._CLOTH_DRIVERS[rig.as_pointer()] = candidate
    rig['pso2_skin_cloth_driver_required'] = True
    off = sum(c.disabled for p in parts for c in p.colliders)
    rig[STATUS] = (f'FLTD 33 / {sum(len(p.mapping) for p in parts)} particles / Scene inputs'
                   + (f' / {off} colliders off (bones not in scene)' if off else ''))
    rig['pso2_cloth_live_dynamic_verified'] = False
    return candidate


def evaluate(driver, world, names, frame, scene, rig_matrix, data, rig=None, depsgraph=None):
    if isinstance(driver,FileClothPreview33):
        banks = getattr(driver, 'related_bank', None)
        if rig is not None and isinstance(banks, dict):
            sources, _names = related_bank(rig, scene)
            if depsgraph is not None:
                rig = rig.evaluated_get(depsgraph)
                sources = {b: (o.evaluated_get(depsgraph) if o is not None else None) for b, o in sources.items()}
            driver.related_worlds = related_worlds(rig, sources, banks, data)
        model = np.asarray(data.get('model_transform',np.eye(4)),dtype='f4')
        transform = _BLENDER_TO_NATIVE @ np.asarray(rig_matrix,dtype='f4') @ model
        gravity = np.zeros(4,'f4')
        if scene.use_gravity:
            gravity[:3] = _BLENDER_TO_NATIVE[:3,:3] @ np.asarray(scene.gravity,dtype='f4')
        driver.context(transform,gravity)
    return driver.evaluate(world,names,frame,scene.render.fps/scene.render.fps_base)


def enable(rig, scene, *, shape_data=None):
    if not rig.get(live_skinning.SOURCE) or not rig.get(cloth_assets.SOURCE_KEY):
        raise ValueError('Import matching NGS AQP/AQN and FLTD before enabling physics')
    text = bpy.data.texts.get(rig[cloth_assets.SOURCE_KEY])
    if text is not None and not declares_cloth(cloth_assets.restore_file(json.loads(text.as_string()))):
        # Placeholder FLTD of a model without cloth: leave the rig as imported.
        rig[STATUS] = 'No cloth parts (this FLTD declares none)'
        return None
    keys = (live_skinning.MODE,live_skinning.PROFILE,live_skinning.BASELINE,AUTO,SETTLE,
        'pso2_skin_frame_origin','pso2_skin_time_scale','pso2_skin_live_runtime_verified',
        'pso2_skin_cloth_driver_required')
    previous = {k:rig.get(k) for k in keys}
    old_driver = live_skinning._CLOTH_DRIVERS.get(rig.as_pointer())
    try:
        if not rig.get(live_skinning.PROFILE):
            raw = live_skinning._read_text(rig[live_skinning.SOURCE])
            if not raw.get('native_template'):
                raise ValueError(raw.get('native_template_error','Re-import this model to retain source AQN inputs'))
            data = copy.deepcopy(raw['native_template'])
            if shape_data:
                from . import aqm
                motion = aqm.parse_aqm(shape_data)
                second = data['local_layers']['second']
                data['local_layers']['second'] = native_pose_sources.sample_motion_pose(
                    asdict(motion),data['node_names'],second,1,
                    source_name_map={motion.nodes[0].name:data['node_names'][0]}).trs.tolist()
            live_skinning.bind_native_profile(rig,data,frame=scene.frame_current_final)
        rig[SETTLE] = int(rig.get(SETTLE,30))
        rig.id_properties_ui(SETTLE).update(min=0,max=120,description='Initial settling steps; timeline jumps restart at the current pose')
        rig[AUTO] = True
        rig[live_skinning.MODE] = 'NATIVE_INPUTS'
        # Use set_mode to preserve the affine deformer/armature switch rules.
        live_skinning.set_mode(rig,'NATIVE_INPUTS')
        if rig.get(live_skinning.ERROR):
            raise ValueError(rig[live_skinning.ERROR])
    except Exception:
        for key,value in previous.items():
            if value is None:
                if key in rig:del rig[key]
            else:rig[key]=value
        if old_driver is None:live_skinning._CLOTH_DRIVERS.pop(rig.as_pointer(),None)
        else:live_skinning._CLOTH_DRIVERS[rig.as_pointer()]=old_driver
        live_skinning._CACHE.pop(('pose',rig.as_pointer()),None)
        if previous[live_skinning.MODE]:
            live_skinning.set_mode(rig,previous[live_skinning.MODE])
        raise


def disable(rig):
    rig[AUTO] = False
    live_skinning.set_cloth_driver(rig,None)
    rig[STATUS] = 'Off'
    if rig.get(live_skinning.PROFILE):live_skinning.set_mode(rig,'NATIVE_INPUTS')


@classes.register
class PSO2_OT_ClothAuto(bpy.types.Operator):
    """Enable or disable the supported file-driven cloth preview"""
    bl_idname='pso2.cloth_auto'
    bl_label='FLTD Cloth Physics'
    bl_options={'UNDO'}
    enabled:bpy.props.BoolProperty(default=True)

    @classmethod
    def poll(cls,context):
        rig=import_aqm._find_target_armature(context)
        return rig is not None and bool(rig.get(cloth_assets.SOURCE_KEY))

    def execute(self,context):
        rig=import_aqm._find_target_armature(context)
        try:
            if self.enabled:enable(rig,context.scene)
            else:disable(rig)
        except (ValueError,KeyError,TypeError,IndexError,OSError) as error:
            rig[STATUS]='Unavailable: '+str(error)
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}


@classes.register
class PSO2_OT_ClothReset(bpy.types.Operator):
    """Reinitialize cloth from source reference and the current animated pose"""
    bl_idname='pso2.cloth_reset'
    bl_label='Reset Cloth'
    bl_options={'UNDO'}

    @classmethod
    def poll(cls,context):
        rig=import_aqm._find_target_armature(context)
        return rig is not None and bool(rig.get(AUTO))

    def execute(self,context):
        rig=import_aqm._find_target_armature(context)
        live_skinning._CLOTH_DRIVERS.pop(rig.as_pointer(),None)
        live_skinning._CACHE.pop(('pose',rig.as_pointer()),None)
        try:
            live_skinning.update_rig(rig,context.scene,context.evaluated_depsgraph_get(),force=True)
        except (ValueError,KeyError,TypeError,IndexError) as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}
        return {'FINISHED'}
