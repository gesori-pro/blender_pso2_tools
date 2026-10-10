"""Per-face, reversible editing of the verified face-motion blend weights.

These are character sliders (-127..127), not replacement vertex weights.
Every edit rebuilds the selected rig from its saved pose and the face's own
AQM, so returning a slider to zero cannot leave its last deformation behind.
No other character, mesh data, action, or rest pose is modified.
"""

import re
from contextlib import contextmanager
from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix

from . import aqm, charfile, classes, face_shape, import_aqm
from .debug import debug_print

SETTINGS = "pso2_face_settings"
SOURCE = "pso2_face_source"
BASE = "pso2_face_base"
AVAILABLE = "pso2_face_available"
_BUSY: set[int] = set()

# Readable expression fields without verified transform keys. Their live UI
# mapping is known, but pupil/animated lip-sync behavior is not implemented.
UNSUPPORTED = (
    "pupilShape",
    "lipSyncStrength",
)

_LABELS = {
    "eyeHorizontalPosition": ("Eyes", "Horizontal Position"),
    "ngsSLID.mouthVertical": ("Mouth", "Vertical Position"),
    "ngsSLID.eyebrowHoriz": ("Brows", "Horizontal Position"),
    "ngsSLID.irisVertical": ("Eyes", "Iris Vertical Position"),
    "leftEyebrowVertical": ("Brows", "Left Vertical Position"),
    "rightEyebrowVertical": ("Brows", "Right Vertical Position"),
    "leftEyebrowExpression": ("Brows", "Left Expression"),
    "rightEyebrowExpression": ("Brows", "Right Expression"),
    "leftMouthVertical": ("Mouth", "Left Mouth Corner"),
    "rightMouthVertical": ("Mouth", "Right Mouth Corner"),
    "eyeCorner": ("Eyes", "Eye Corner"),
    "leftEyelidVertical": ("Eyes", "Left Eyelid"),
    "rightEyelidVertical": ("Eyes", "Right Eyelid"),
    "mouthA": ("Mouth", "A"),
    "mouthI": ("Mouth", "I"),
    "mouthU": ("Mouth", "U"),
    "mouthE": ("Mouth", "E"),
    "mouthO": ("Mouth", "O"),
    "tongue": ("Tongue", "Extension"),
    "tongueVertical": ("Tongue", "Vertical Position"),
    "tongueHorizontal": ("Tongue", "Horizontal Position"),
    "mouthOpenClose": ("Lip Customization", "Open / Close"),
    "mouthCornerPosition": ("Lip Customization", "Mouth Corner Position"),
    "mouthCornerShape": ("Lip Customization", "Mouth Corner Shape"),
    "upperLipShape": ("Lip Customization", "Upper Lip Shape"),
    "lowerLipShape": ("Lip Customization", "Lower Lip Shape"),
    "upperLowerLipRatio": ("Lip Customization", "Upper / Lower Lip Ratio"),
    "teethOpenClose": ("Teeth / Jaw", "Teeth Open / Close"),
    "teethVerticalPosition": ("Teeth / Jaw", "Teeth Vertical Position"),
    "lowerJawOpenClose": ("Teeth / Jaw", "Lower Jaw Open / Close"),
    "eyelashAngle": ("Eyelashes / Shadow", "Eyelash Angle"),
    "eyelashLength": ("Eyelashes / Shadow", "Eyelash Length"),
    "eyeShadows": ("Eyelashes / Shadow", "Eye Shadow"),
}
_GROUPS = {
    "headVerts": "Head Shape",
    "faceShapeVerts": "Face Shape",
    "eyeShapeVerts": "Eye Shape",
    "noseHeightVerts": "Nose Height",
    "noseShapeVerts": "Nose Shape",
    "mouthVerts": "Mouth Shape",
}


def _controls():
    result = []
    seen = set()
    for field, minimum, _ in face_shape._FACE_SLIDERS:
        if field in seen:
            continue  # eyeShapeVerts.Z has two frame pairs, one shared control.
        seen.add(field)
        if field.startswith("baseFIGR."):
            stem, axis = field.removeprefix("baseFIGR.").split(".")
            group, label = _GROUPS[stem], axis
        else:
            group, label = _LABELS[field.removeprefix("<expr>.")]
        key = "weight_" + str(len(result))
        result.append((key, field, group, label, 0 if minimum is None else -127))
    return tuple(result)


CONTROLS = _controls()


@contextmanager
def _suspend(armature):
    pointer = armature.as_pointer()
    _BUSY.add(pointer)
    try:
        yield
    finally:
        _BUSY.discard(pointer)


def is_face(armature) -> bool:
    if armature is None or armature.type != "ARMATURE":
        return False
    _, names = import_aqm._get_bone_maps(armature)
    return "eyestop" in names and "nosetop" in names


def selected_face(context):
    armature = import_aqm._find_target_armature(context)
    return armature if is_face(armature) else None


def _field(field, expression):
    return field.replace("<expr>.", expression + ".")


def _restore(armature):
    for name, saved in armature.get(BASE, {}).items():
        bone = armature.pose.bones.get(name)
        if bone is None:
            continue
        values = saved["basis"]
        bone.rotation_mode = saved["mode"]
        bone.matrix_basis = Matrix([values[i : i + 4] for i in range(0, 16, 4)])


def _check_static(armature):
    animation = armature.animation_data
    if animation and (
        animation.action or any(not track.mute for track in animation.nla_tracks)
    ):
        raise ValueError(
            "An action/NLA is active on this face. Use a static copy first."
        )


def apply(context, armature):
    settings = getattr(armature, SETTINGS)
    if not settings.loaded:
        raise ValueError("Load a face character file first.")
    _check_static(armature)
    motion = face_shape._load_motion(context, settings.face_id)
    if motion is None:
        raise ValueError(
            f"Face motion {settings.face_id} is missing from the game data."
        )
    source = dict(armature[SOURCE])
    _, names = import_aqm._get_bone_maps(armature)
    targets = face_shape._blend(source, motion, settings.expression)
    # Validate before restoring/writing any pose.
    connected = [
        names[name.lower()]
        for name in targets
        if name.lower() in names
        and armature.data.bones[names[name.lower()]].use_connect
    ]
    if connected:
        raise ValueError(
            "Disconnect connected facial bones before editing their position."
        )
    _restore(armature)
    count = face_shape._pose(
        armature, targets, names, import_aqm.bone_correction(armature)
    )
    context.view_layer.update()
    settings.status = f"{count} facial bones posed (selected rig only)"
    return count


def _apply_update(context, armature):
    try:
        apply(context, armature)
        return True
    except (OSError, KeyError, ValueError, aqm.AqmError) as error:
        getattr(armature, SETTINGS).status = str(error)
        debug_print("Face customization:", error)
        return False


def _weight_update(field):
    def update(settings, context):
        armature = settings.id_data
        if not settings.loaded or armature.as_pointer() in _BUSY:
            return
        key = next(key for key, item, *_ in CONTROLS if item == field)
        source = dict(armature[SOURCE])
        name = _field(field, settings.expression)
        if name not in source or not armature.get(AVAILABLE, {}).get(field, True):
            with _suspend(armature):
                setattr(settings, key, int(source.get(name, 0)))
            settings.status = "This control has no field / deformation on this face."
            return
        source[name] = getattr(settings, key)
        armature[SOURCE] = source
        _apply_update(context, armature)

    return update


def _expression_update(settings, context):
    armature = settings.id_data
    if not settings.loaded or armature.as_pointer() in _BUSY:
        return
    _read_controls(armature)
    _apply_update(context, armature)


def _read_controls(armature):
    settings = getattr(armature, SETTINGS)
    source = armature[SOURCE]
    with _suspend(armature):
        for key, field, _, _, _ in CONTROLS:
            setattr(
                settings, key, int(source.get(_field(field, settings.expression), 0))
            )


@classes.register
class PSO2FaceSettings(bpy.types.PropertyGroup):
    loaded: bpy.props.BoolProperty(default=False, options={"HIDDEN"})
    face_id: bpy.props.IntProperty(default=0, options={"HIDDEN"})
    source_name: bpy.props.StringProperty(options={"HIDDEN"})
    status: bpy.props.StringProperty(options={"HIDDEN"})
    expression: bpy.props.EnumProperty(
        name="Expression",
        items=[(name, label, label) for name, label in face_shape.EXPRESSIONS],
        default=face_shape.DEFAULT_EXPRESSION,
        update=_expression_update,
    )


for _key, _name, _group, _label, _minimum in CONTROLS:
    PSO2FaceSettings.__annotations__[_key] = bpy.props.IntProperty(
        name=_label,
        description=f"{_name}: face-motion blend weight / 127",
        min=_minimum,
        max=127,
        default=0,
        options=set(),  # Static pose controls; frame-driven evaluation is not implemented.
        update=_weight_update(_name),
    )


def add_properties():
    setattr(
        bpy.types.Object, SETTINGS, bpy.props.PointerProperty(type=PSO2FaceSettings)
    )


def remove_properties():
    if hasattr(bpy.types.Object, SETTINGS):
        delattr(bpy.types.Object, SETTINGS)


def load(context, armature, char, expression=face_shape.DEFAULT_EXPRESSION):
    if not is_face(armature):
        raise ValueError(
            "Select the face armature or a mesh bound to it, not the body rig."
        )
    _check_static(armature)
    face_id = int(char["baseSLCT.faceTypePart"])
    match = re.search(r"pl_r?fc_(\d+)", armature.name, re.IGNORECASE)
    if match and int(match[1]) != face_id:
        raise ValueError(
            "The character file's face type does not match this face model."
        )
    motion = face_shape._load_motion(context, face_id)
    if motion is None:
        raise ValueError(f"Face motion {face_id} is missing from the game data.")
    _, names = import_aqm._get_bone_maps(armature)
    matched = [names[n.name.lower()] for n in motion.nodes if n.name.lower() in names]
    if not matched:
        raise ValueError("No motion bones match this face rig.")
    if any(armature.data.bones[name].use_connect for name in matched):
        raise ValueError(
            "Disconnect connected facial bones before editing their position."
        )
    if BASE not in armature:
        armature[BASE] = {
            name: {
                "basis": [
                    value
                    for row in armature.pose.bones[name].matrix_basis
                    for value in row
                ],
                "mode": armature.pose.bones[name].rotation_mode,
            }
            for name in matched
        }
    source = {}
    for _, field, _, _, _ in CONTROLS:
        presets = face_shape.EXPRESSIONS if "<expr>" in field else [(expression, "")]
        for preset, _ in presets:
            value = face_shape.slider(char, field, preset)
            if value is not None:
                source[_field(field, preset)] = value
    armature[SOURCE] = source
    armature[AVAILABLE] = {
        field: face_shape.has_slider_motion(motion, field)
        for _, field, _, _, _ in CONTROLS
    }
    settings = getattr(armature, SETTINGS)
    with _suspend(armature):
        settings.loaded = True
        settings.face_id = face_id
        settings.expression = expression
    _read_controls(armature)
    return apply(context, armature)


@classes.register
class PSO2_OT_LoadFaceCustomization(bpy.types.Operator, ImportHelper):
    """Load the file's facial proportions and expressions onto the selected face"""

    bl_idname = "pso2.load_face_customization"
    bl_label = "Load Face Customization"
    bl_options = {"UNDO"}
    filename_ext = ".fnp"
    filter_glob: bpy.props.StringProperty(
        default="*.fhp;*.fnp;*.fcp;*.fdp;*.mhp;*.mnp;*.mcp;*.mdp;"
        "*.fhpu;*.fnpu;*.fcpu;*.fdpu;*.mhpu;*.mnpu;*.mcpu;*.mdpu",
        options={"HIDDEN"},
    )

    @classmethod
    def poll(cls, context):
        return selected_face(context) is not None

    def execute(self, context):
        armature = selected_face(context)
        if armature is None:
            self.report({"ERROR"}, "Select the face armature or its mesh first.")
            return {"CANCELLED"}
        try:
            count = load(context, armature, charfile.CharacterFile.load(self.filepath))
        except (OSError, KeyError, ValueError, aqm.AqmError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        settings = getattr(armature, SETTINGS)
        settings.source_name = Path(self.filepath).name
        self.report({"INFO"}, f"Applied facial customization to {count} bones")
        return {"FINISHED"}


@classes.register
class PSO2_OT_ResetFaceCustomization(bpy.types.Operator):
    """Set the selected expression's weights and common face sliders to zero"""

    bl_idname = "pso2.reset_face_customization"
    bl_label = "Neutral Sliders"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        face = selected_face(context)
        return face is not None and getattr(face, SETTINGS).loaded

    def execute(self, context):
        armature = selected_face(context)
        try:
            _check_static(armature)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        settings = getattr(armature, SETTINGS)
        source = dict(armature[SOURCE])
        for _, field, _, _, _ in CONTROLS:
            name = _field(field, settings.expression)
            if name in source:
                source[name] = 0
        armature[SOURCE] = source
        _read_controls(armature)
        return {"FINISHED"} if _apply_update(context, armature) else {"CANCELLED"}


@classes.register
class PSO2_OT_ClearFaceCustomization(bpy.types.Operator):
    """Restore the facial pose saved before loading customization"""

    bl_idname = "pso2.clear_face_customization"
    bl_label = "Restore Previous Face"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        face = selected_face(context)
        return face is not None and BASE in face

    def execute(self, context):
        armature = selected_face(context)
        try:
            _check_static(armature)
        except ValueError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        _restore(armature)
        for prop in (SOURCE, BASE, AVAILABLE):
            if prop in armature:
                del armature[prop]
        getattr(armature, SETTINGS).loaded = False
        context.view_layer.update()
        return {"FINISHED"}
