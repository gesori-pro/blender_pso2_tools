"""Pose an imported face to the character file's face-shape sliders.

A face model does not ship one shape: it ships a motion of its own,
`pl_rfm_<faceId>_00.aqm` inside `pl_fm_<faceId>.ice`, whose frames are the
slider extremes. Frame 0 is the neutral face and each slider owns a pair of
frames - eye shape at 34 and 36, mouth at 66 to 76, and so on - which the
game blends between exactly the way the body proportions work:

    t = slider / 127
    t < 0 : blend frame 0 towards the min frame by -t
    t >= 0: blend frame 0 towards the max frame by t

Positions add, scales multiply, rotations compose. Without this the face
imports at its neutral shape, which is a different face: the eyes sit at
the wrong spacing and the nose and mouth are the model's defaults rather
than the character's.

The frame numbers come from Salon Tool's CharacterProportionConstants,
which defines them but never uses them. Two of its names are misleading -
checked against a running game, "FaceShape1" is headVerts and "FaceShape2"
is faceShapeVerts, and its two nose groups are swapped - so the mapping
below is the corrected one.
"""

import tempfile
from pathlib import Path

from mathutils import Matrix, Quaternion, Vector

from . import aqm, ice, import_aqm, import_model, objects
from .debug import debug_print
from .preferences import get_preferences

# Character-file field -> the motion frames holding that slider's extremes.
# `<expr>` stands for the expression preset the resting face uses. A slider
# with no min frame is a one-sided one: it blends frame 0 towards its single
# frame by value / 127.
_FACE_SLIDERS: tuple[tuple[str, int | None, int], ...] = (
    ("baseFIGR.noseHeightVerts.Z", 6, 8),
    ("baseFIGR.headVerts.X", 10, 12),
    ("baseFIGR.headVerts.Y", 14, 16),
    ("baseFIGR.headVerts.Z", 18, 20),
    ("baseFIGR.faceShapeVerts.X", 22, 24),
    ("baseFIGR.faceShapeVerts.Y", 26, 28),
    ("baseFIGR.faceShapeVerts.Z", 30, 32),
    ("baseFIGR.eyeShapeVerts.X", 34, 36),
    ("baseFIGR.eyeShapeVerts.Y", 38, 40),
    ("baseFIGR.eyeShapeVerts.Z", 42, 44),
    ("baseFIGR.noseHeightVerts.X", 46, 48),
    ("baseFIGR.noseHeightVerts.Y", 50, 52),
    ("baseFIGR.noseShapeVerts.X", 54, 56),
    ("baseFIGR.noseShapeVerts.Y", 58, 60),
    ("baseFIGR.noseShapeVerts.Z", 62, 64),
    ("baseFIGR.mouthVerts.X", 66, 68),
    ("baseFIGR.mouthVerts.Y", 70, 72),
    ("baseFIGR.mouthVerts.Z", 74, 76),
    ("eyeHorizontalPosition", 82, 84),
    # 86-92 hold no keys at all: the eye-size slider moves nothing in game.
    ("<expr>.leftEyebrowVertical", 94, 96),
    ("<expr>.leftMouthVertical", 98, 100),
    ("<expr>.rightEyebrowVertical", 102, 104),
    ("<expr>.rightMouthVertical", 106, 108),
    ("<expr>.eyeCorner", 110, 112),
    ("<expr>.leftEyelidVertical", 114, 116),
    ("<expr>.leftEyebrowExpression", 118, 120),
    ("<expr>.rightEyelidVertical", 122, 124),
    ("<expr>.rightEyebrowExpression", 126, 128),
    ("<expr>.mouthA", None, 130),
    ("<expr>.mouthI", None, 132),
    ("<expr>.mouthU", None, 134),
    ("<expr>.mouthE", None, 136),
    ("<expr>.mouthO", None, 138),
    ("<expr>.tongue", None, 144),
    ("ngsSLID.mouthVertical", 146, 148),
    ("ngsSLID.eyebrowHoriz", 150, 152),
    ("ngsSLID.irisVertical", 154, 156),
)

# The expression the face rests in. The file carries ten presets; this is
# the one the character creator shows when nothing else is playing.
DEFAULT_EXPRESSION = "faceNatural"


def _slider(char, field: str, expression: str) -> int | None:
    name = field.replace("<expr>", expression)
    try:
        value = char[name]
    except (KeyError, TypeError):
        return None
    return int(value) if isinstance(value, int) else None


def _load_motion(context, face_id: int) -> aqm.AqmMotion | None:
    """The face's own shape motion, or None when it is not in the game data."""
    data_path = get_preferences(context).get_pso2_data_path()
    name = objects.CmxFileName(f"character/making_reboot/pl_fm_{face_id}.ice")
    path = import_model._get_ice_path(name, data_path, True)
    if path is None or not path.exists():
        debug_print(f"No face-shape motion for face {face_id}")
        return None

    entry = next(
        (f for f in ice.IceFile.load(path).get_files() if f.name.lower().endswith(".aqm")),
        None,
    )
    if entry is None:
        return None

    with tempfile.TemporaryDirectory(prefix="pso2_face_") as tmp:
        motion_path = Path(tmp) / entry.name
        motion_path.write_bytes(entry.data)
        return aqm.read_aqm(str(motion_path))


def _key_at(node, key_type, frame: int):
    key_set = node.get_key_set(key_type)
    if key_set is None:
        return None
    frames = list(key_set.frames())
    if frame not in frames:
        return None
    return list(key_set.vec4_keys[frames.index(frame)])


def _blend(char, motion, expression: str) -> dict[str, dict]:
    """Node name -> the PSO2-space transform the sliders ask for.

    Only nodes some slider actually moves are returned; the rest keep
    whatever the body proportion pass left on them, which is where the head's
    own size lives.
    """
    neutral = {}
    for node in motion.nodes:
        neutral[node.name] = (
            _key_at(node, aqm.KEY_TYPE_POSITION, 0) or [0.0, 0.0, 0.0, 0.0],
            _key_at(node, aqm.KEY_TYPE_ROTATION, 0) or [0.0, 0.0, 0.0, 1.0],
            _key_at(node, aqm.KEY_TYPE_SCALE, 0) or [1.0, 1.0, 1.0, 0.0],
        )

    result: dict[str, dict] = {}
    for field, min_frame, max_frame in _FACE_SLIDERS:
        value = _slider(char, field, expression)
        if value is None:
            continue

        t = max(-1.0, min(1.0, value / 127.0))
        if min_frame is None:
            frame, weight = max_frame, max(0.0, t)
        elif t < 0:
            frame, weight = min_frame, -t
        else:
            frame, weight = max_frame, t
        if weight == 0.0:
            continue

        for node in motion.nodes:
            position = _key_at(node, aqm.KEY_TYPE_POSITION, frame)
            rotation = _key_at(node, aqm.KEY_TYPE_ROTATION, frame)
            scale = _key_at(node, aqm.KEY_TYPE_SCALE, frame)
            if position is None and rotation is None and scale is None:
                continue

            base_position, base_rotation, base_scale = neutral[node.name]
            entry = result.setdefault(
                node.name,
                {
                    "position": list(base_position[:3]),
                    "rotation": Quaternion(
                        (
                            base_rotation[3],
                            base_rotation[0],
                            base_rotation[1],
                            base_rotation[2],
                        )
                    ),
                    "scale": list(base_scale[:3]),
                },
            )

            if position is not None:
                for axis in range(3):
                    entry["position"][axis] += (
                        position[axis] - base_position[axis]
                    ) * weight

            if scale is not None:
                for axis in range(3):
                    if abs(base_scale[axis]) > 1e-9:
                        ratio = scale[axis] / base_scale[axis]
                        entry["scale"][axis] *= 1.0 + (ratio - 1.0) * weight

            if rotation is not None:
                target = Quaternion(
                    (rotation[3], rotation[0], rotation[1], rotation[2])
                )
                base = Quaternion(
                    (
                        base_rotation[3],
                        base_rotation[0],
                        base_rotation[1],
                        base_rotation[2],
                    )
                )
                delta = target @ base.inverted()
                entry["rotation"] = (
                    Quaternion().slerp(delta, weight) @ entry["rotation"]
                )

    return result


def apply(context, char, face_id: int, expression: str = DEFAULT_EXPRESSION) -> dict:
    """Pose every face armature in the scene to the character's face sliders."""
    motion = _load_motion(context, face_id)
    if motion is None:
        return {"posed": 0, "bones": 0}

    targets = _blend(char, motion, expression)
    if not targets:
        return {"posed": 0, "bones": 0}

    posed = bones = 0
    for armature in context.scene.objects:
        if armature.type != "ARMATURE":
            continue
        _, bones_by_name = import_aqm._get_bone_maps(armature)
        if "eyestop" not in bones_by_name and "nosetop" not in bones_by_name:
            continue

        correction = import_aqm.bone_correction(armature)
        applied = _pose(armature, targets, bones_by_name, correction)
        if applied:
            posed += 1
            bones += applied

    context.view_layer.update()
    return {"posed": posed, "bones": bones}


def _pose(armature, targets, bones_by_name, correction: Matrix) -> int:
    """Write the blended transforms onto one armature, in its own axes.

    The conversion is the one motion import uses: a key is in the PSO2
    node's axes, which the FBX importer rotated by the correction when it
    built the bone, so it has to be undone on the way in.
    """
    correction3 = correction.to_3x3()
    correction3_inv = correction3.inverted()
    applied = 0

    for name, target in targets.items():
        bone_name = bones_by_name.get(name.lower())
        if bone_name is None:
            continue
        pose_bone = armature.pose.bones[bone_name]
        bone = pose_bone.bone

        if bone.parent is not None:
            rest = bone.parent.matrix_local.inverted() @ bone.matrix_local
            parent_correction3_inv = correction3_inv
        else:
            rest = bone.matrix_local.copy()
            # A parentless bone hangs off the armature object, not a bone,
            # so nothing corrected its axes.
            parent_correction3_inv = Matrix.Identity(3)

        rest_translation = rest.to_translation()
        rest_rotation_inv = rest.to_quaternion().inverted()

        pose_bone.location = rest_rotation_inv @ (
            (parent_correction3_inv @ Vector(target["position"])) - rest_translation
        )

        rotation = rest_rotation_inv @ (
            parent_correction3_inv @ target["rotation"].to_matrix() @ correction3
        ).to_quaternion()
        rotation.normalize()
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = rotation

        pose_bone.scale = (
            correction3_inv @ Matrix.Diagonal(Vector(target["scale"])) @ correction3
        ).to_scale()

        applied += 1

    return applied
