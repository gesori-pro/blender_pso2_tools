"""
Export the active PSO2 armature's animation to a PSO2 AQM motion file.

The pose is baked one key per frame, matching the layout of game motion
files and of the community 3ds Max exporter (PSO2AQM_IO.ms). Nodes are
written by index (the bone's pso2_bone_id custom property), position and
rotation as local-to-parent transforms and scale as absolute scale, which
is what PSO2 expects (bones do not inherit scale).
"""

import re
from math import radians
from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector

from . import aqm, classes, import_aqm, scene_props
from .util import OperatorResult

# FBX to Blender axis conversion baked into the armature object on import.
_CONVERSION = Matrix.Rotation(radians(90), 4, "X")

_TRAILING_EXCLUDE_FLAG = 0x400

# The bone and channel out of an f-curve path, as in
# `pose.bones["hip#3C6#0"].location`.
_BONE_PATH = re.compile(r'pose\.bones\["(.+)"\]\.(\w+)$')


class ExportError(Exception):
    """Raised when the armature cannot be exported."""


@classes.register
class PSO2_OT_ExportAqm(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ExportHelper
):
    """Save the active PSO2 armature's animation as a PSO2 AQM motion file"""

    bl_label = "Export AQM"
    bl_idname = "pso2.export_aqm"
    bl_options = {"PRESET"}

    filename_ext = ".aqm"
    filter_glob: bpy.props.StringProperty(default="*.aqm;*.trm", options={"HIDDEN"})

    frame_source: bpy.props.EnumProperty(
        name="Frame Range",
        description="Which frame range to export",
        items=[
            ("SCENE", "Scene", "Export the scene frame range"),
            ("ACTION", "Action", "Export the active action's frame range"),
        ],
        default="SCENE",
    )
    player_anim: bpy.props.BoolProperty(
        name="Add __NodeTreeFlag__ Node",
        description=(
            "Add the __NodeTreeFlag__ node used by official player"
            " animations. Emote mods generally work without it"
        ),
        default=False,
    )
    ignore_applied_shape: bpy.props.BoolProperty(
        name="Ignore Body Shape",
        description=(
            "Leave the character's proportions out of the motion. Keys carry"
            " each bone's size and its offset from its parent, so a body"
            " shape - whether applied to the rest pose or sitting in the pose"
            " layer - is written into every frame, and every character"
            " playing the motion then takes on that body on top of their own."
            " Channels the action drives are kept: those are the pose. Has no"
            " effect without a character file loaded"
        ),
        default=True,
    )

    # Captured at invoke time: the file browser's context has no armature
    # to inspect by the time draw() runs.
    shape_state: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    def invoke(self, context, event):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
        from . import bake_rest

        self.shape_state = bake_rest.shape_state(
            import_aqm._find_target_armature(context)
        )
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        assert self.layout is not None

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "frame_source")
        layout.prop(self, "player_anim")
        layout.prop(self, "ignore_applied_shape")

        from . import bake_rest

        bake_rest.draw_shape_state(layout, self.shape_state)

    def execute(self, context) -> OperatorResult:
        armature = import_aqm._find_target_armature(context)
        if armature is None:
            self.report(
                {"ERROR"},
                "No target armature. Select a PSO2 armature (or a model parented"
                " to one) and try again.",
            )
            return {"CANCELLED"}

        from . import bake_rest

        shaped = armature if self.ignore_applied_shape else None
        keyed = _keyed_channels(armature) if self.ignore_applied_shape else None

        try:
            frame_start, frame_end = self._get_frame_range(context, armature)
            with (
                bake_rest.bake_suspended(shaped),
                bake_rest.shape_in_pose_suspended(shaped, keyed),
            ):
                motion = build_motion(
                    context,
                    armature,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    player_anim=self.player_anim,
                    shape_scale=bake_rest.keyed_shape_scale(shaped, keyed),
                )
        except ExportError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}

        path = Path(self.filepath)  # type: ignore
        size = aqm.write_aqm(path, motion)

        message = (
            f"Exported {path.name}: {len(motion.nodes)} nodes,"
            f" frames 0-{motion.end_frame}, {size:,} bytes"
        )

        # The exporter samples the posed transforms, so anything sitting in
        # the pose layer that the action does not drive - a body shape, most
        # often - ends up baked into the motion (SPEC §6-8).
        if stuck := _unkeyed_posed_bones(armature):
            shown = ", ".join(sorted(stuck)[:4])
            more = f" and {len(stuck) - 4} more" if len(stuck) > 4 else ""
            self.report(
                {"WARNING"},
                message + f". {len(stuck)} bones are posed but not animated"
                f" ({shown}{more}), so whatever holds them - a body shape,"
                " most often - is now part of this motion. Reset the pose on"
                " those bones if that was not intended.",
            )
            return {"FINISHED"}

        self.report({"INFO"}, message)

        return {"FINISHED"}

    def _get_frame_range(self, context, armature) -> tuple[int, int]:
        if self.frame_source == "ACTION":
            animation_data = armature.animation_data
            if animation_data is None or animation_data.action is None:
                raise ExportError("The armature has no active action")

            start, end = animation_data.action.frame_range
            return round(start), round(end)

        return context.scene.frame_start, context.scene.frame_end


def _keyed_channels(armature: bpy.types.Object) -> dict[str, set[str]]:
    """Bone name -> the transform channels the active action drives.

    Actions held their curves in a flat `fcurves` list up to 4.3 and in
    slotted channelbags after; 5.0 dropped the flat list. Both are read so
    the check works either way. The armature's own slot is preferred, since
    one action can hold curves for several data-blocks at once.
    """
    animation_data = armature.animation_data
    action = animation_data.action if animation_data else None
    if action is None:
        return {}

    slot = getattr(animation_data, "action_slot", None)
    curves = list(getattr(action, "fcurves", ()))
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            bags = [strip.channelbag(slot)] if slot else list(strip.channelbags)
            curves.extend(curve for bag in bags if bag for curve in bag.fcurves)

    driven: dict[str, set[str]] = {}
    for curve in curves:
        if match := _BONE_PATH.match(curve.data_path):
            driven.setdefault(match.group(1), set()).add(match.group(2))

    return driven


def _unkeyed_posed_bones(
    armature: bpy.types.Object, epsilon: float = 1e-4
) -> list[str]:
    """Posed bones the action leaves alone, so their pose is a constant.

    A motion carries every bone at every frame, so a bone the action never
    touches is still written out - holding whatever the pose layer had on
    it. That is where a loaded body shape goes.

    The comparison is against the pose the import left (import_fnp
    .store_model_pose), not the rest pose: the fingertip bones come in
    already transformed, and measuring those against rest would report ten
    bones on every export of an untouched model.
    """
    from . import import_fnp

    keyed = _keyed_channels(armature)
    out = []

    for pose_bone in armature.pose.bones:
        if pose_bone.name in keyed:
            continue

        baseline = pose_bone.get(import_fnp.MODEL_POSE_PROP)
        if baseline is None or len(baseline) != 10:
            continue

        current = (
            *pose_bone.location,
            *pose_bone.matrix_basis.to_quaternion(),
            *pose_bone.scale,
        )
        if any(abs(a - b) > epsilon for a, b in zip(current, baseline, strict=True)):
            out.append(pose_bone.name.split("#")[0])

    return out


def build_motion(
    context: bpy.types.Context,
    armature: bpy.types.Object,
    frame_start: int,
    frame_end: int,
    player_anim=False,
    shape_scale: dict | None = None,
) -> aqm.AqmMotion:
    """Bake the armature's animation into a motion, one key per frame."""
    if frame_end < frame_start:
        raise ExportError("Invalid frame range")

    scene = context.scene
    export_bones = _get_export_bones(armature)

    # Bone axes run differently in Blender and PSO2 (see
    # import_aqm.bone_correction). Keys have to go back the way they came.
    correction = import_aqm.bone_correction(armature)
    correction_inv = correction.inverted()
    correction3 = correction.to_3x3()
    correction3_inv = correction_inv.to_3x3()

    end_frame = frame_end - frame_start
    frame_count = end_frame + 1

    # Sample the posed transforms at each frame.
    node_names = [armature.name.split("#")[0]] + [
        name.split("#")[0] for name in export_bones
    ]
    samples: list[list[tuple]] = [[] for _ in range(len(export_bones) + 1)]

    current_frame = scene.frame_current
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            absolute: dict[str, Vector] = {}

            # Node 0 is the skeleton root: the armature object itself,
            # relative to the FBX axis conversion.
            _sample_matrix(
                samples[0], _CONVERSION.inverted() @ armature.matrix_basis, None
            )

            for index, name in enumerate(export_bones, start=1):
                pose_bone = armature.pose.bones[name]

                if pose_bone.parent is not None:
                    # Motions hide a bone by scaling it to zero, and a zero
                    # scale leaves the parent's matrix with no inverse.
                    # inverted_safe falls back instead of throwing, which
                    # otherwise took the whole export down.
                    local = pose_bone.parent.matrix.inverted_safe() @ pose_bone.matrix
                    parent_correction = correction
                else:
                    local = pose_bone.matrix
                    # Parented to the armature object, which is not a bone
                    # and so carries no correction.
                    parent_correction = Matrix.Identity(4)

                scale = _absolute_scale(pose_bone, absolute, shape_scale)
                _sample_matrix(
                    samples[index],
                    parent_correction @ local @ correction_inv,
                    (correction3 @ Matrix.Diagonal(scale) @ correction3_inv).to_scale(),
                )
    finally:
        scene.frame_set(current_frame)

    # Build the motion.
    multiplier = 0x100 if end_frame > 4095 else 0x10
    data_type_flag = 0x80 if end_frame > 4095 else 0
    timings = aqm.make_baked_timings(end_frame, multiplier)

    motion = aqm.AqmMotion(
        variant=aqm.VARIANT_PLAYER_ANIM if player_anim else aqm.VARIANT_STD_ANIM,
        loop_point=0,
        end_frame=end_frame,
        frame_speed=scene.render.fps / scene.render.fps_base,
        node_count=len(samples) + (1 if player_anim else 0),
    )

    for index, (name, keys) in enumerate(zip(node_names, samples, strict=True)):
        node = aqm.AqmNode(node_type=aqm.NODE_TYPE_STANDARD, node_id=index, name=name)

        for key_type, data_type, offset in (
            (aqm.KEY_TYPE_POSITION, 0x1, 0),
            (aqm.KEY_TYPE_ROTATION, 0x3, 1),
            (aqm.KEY_TYPE_SCALE, 0x1, 2),
        ):
            node.key_sets.append(
                aqm.AqmKeySet(
                    key_type=key_type,
                    data_type=data_type | data_type_flag,
                    unk_int0=0,
                    timings=list(timings) if frame_count > 1 else [],
                    vec4_keys=[keys[frame][offset] for frame in range(frame_count)],
                )
            )

        motion.nodes.append(node)

    if player_anim:
        motion.nodes.append(
            _make_node_tree_flag(len(samples), frame_count, multiplier, data_type_flag)
        )

    return motion


def _absolute_scale(pose_bone, memo: dict, shape: dict | None = None) -> Vector:
    """A bone's scale with its parents' folded back in.

    A PSO2 scale key is absolute, where a Blender bone's is relative to its
    parent, so motion import divides each bone's key by its parent's
    (aqm.prepare_scaling, which hands the work to AquaMotion
    .PrepareScalingForExport - a plain component-wise divide, one level).
    Multiplying back up the chain is the way out.

    Reading the scale off the bone's world matrix instead looks equivalent
    and is not: a scaled parent puts the scale between two rotations, which
    is a shear, and decomposing a shear spreads one number across three.
    """
    cached = memo.get(pose_bone.name)
    if cached is not None:
        return cached

    scale = Vector(pose_bone.scale)

    # A body shape the action keyed along with the pose cannot be taken off
    # the pose - stepping the frame puts it back - so it comes off here.
    if shape and (own := shape.get(pose_bone.name)):
        scale = Vector(
            c / s if abs(s) > 1e-9 else c for c, s in zip(scale, own, strict=True)
        )

    if pose_bone.parent is not None:
        parent = _absolute_scale(pose_bone.parent, memo, shape)
        scale = Vector((scale.x * parent.x, scale.y * parent.y, scale.z * parent.z))

    memo[pose_bone.name] = scale
    return scale


def _sample_matrix(node_samples: list, local: Matrix, bone_scale: Vector | None):
    """Decompose one frame's transforms into (pos, rot, scale) vec4 keys.

    Position and rotation come out of the local matrix, but scale does not:
    a bone that inherits its parent's scale has that scale sitting between
    two rotations in its matrix, which is a shear, and decomposing a shear
    spreads one number across three. A head scaled evenly to 1.030 read back
    as (0.975, 1.047, 1.101) that way.

    Pass the bone's own scale instead - the same channel motion import
    writes to - and the two are exact inverses.
    """
    location, rotation, local_scale = local.decompose()
    scale = local_scale if bone_scale is None else bone_scale

    # Keep consecutive quaternions on the same hemisphere.
    if node_samples:
        previous = node_samples[-1][1]
        if (
            previous[0] * rotation.x
            + previous[1] * rotation.y
            + previous[2] * rotation.z
            + previous[3] * rotation.w
        ) < 0:
            rotation.negate()

    node_samples.append(
        (
            (location.x, location.y, location.z, 0.0),
            (rotation.x, rotation.y, rotation.z, rotation.w),
            (scale.x, scale.y, scale.z, 0.0),
        )
    )


def _get_export_bones(armature: bpy.types.Object) -> list[str]:
    """Bone names to export, ordered by node index starting at 1.

    Trailing bones flagged with 0x400 in the second bone short (outfit
    physics bones, which the game never animates) are dropped, matching
    the layout of game files.
    """
    bones_by_id: dict[int, tuple[str, int]] = {}

    for bone in armature.data.bones:  # type: ignore
        bone_id = bone.get(scene_props.BONE_ID)
        if bone_id is None:
            continue

        parts = bone.name.split("#")
        flags = 0
        if len(parts) >= 3:
            try:
                flags = int(parts[2], 16)
            except ValueError:
                flags = 0

        bones_by_id.setdefault(int(bone_id), (bone.name, flags))

    if not bones_by_id:
        raise ExportError(
            "No bones with a pso2_bone_id property. The armature must be"
            " imported with PSO2 Tools to export motions."
        )

    ids = sorted(bones_by_id)

    while ids and bones_by_id[ids[-1]][1] & _TRAILING_EXCLUDE_FLAG:
        ids.pop()

    if not ids:
        raise ExportError("All bones are flagged as not animatable")

    expected = list(range(1, len(ids) + 1))
    if ids != expected:
        missing = sorted(set(expected) - set(ids))[:10]
        raise ExportError(f"Bone IDs are not contiguous. Missing node IDs: {missing}")

    return [bones_by_id[i][0] for i in ids]


def _make_node_tree_flag(
    node_id: int, frame_count: int, multiplier: int, data_type_flag: int
) -> aqm.AqmNode:
    """The __NodeTreeFlag__ pseudo-node official player animations carry."""
    timings = [0x8 | 0x1]
    timings += [frame * multiplier + 0x8 for frame in range(1, frame_count - 1)]
    if frame_count > 1:
        timings.append((frame_count - 1) * multiplier + 0x8 + 0x2)

    node = aqm.AqmNode(
        node_type=aqm.NODE_TYPE_NODE_TREE_FLAG,
        node_id=0,
        name="__NodeTreeFlag__",
    )

    for key_type in (0x10, 0x11, 0x12):
        node.key_sets.append(
            aqm.AqmKeySet(
                key_type=key_type,
                data_type=0x5 | data_type_flag,
                unk_int0=0,
                timings=list(timings) if frame_count > 1 else [],
                int_keys=[0x31] * frame_count,
            )
        )

    return node
