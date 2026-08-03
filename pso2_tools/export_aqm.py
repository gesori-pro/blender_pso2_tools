"""
Export the active PSO2 armature's animation to a PSO2 AQM motion file.

The pose is baked one key per frame, matching the layout of game motion
files and of the community 3ds Max exporter (PSO2AQM_IO.ms). Nodes are
written by index (the bone's pso2_bone_id custom property), position and
rotation as local-to-parent transforms and scale as absolute scale, which
is what PSO2 expects (bones do not inherit scale).
"""

from math import radians
from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

from . import aqm, classes, import_aqm, scene_props
from .util import OperatorResult

# FBX to Blender axis conversion baked into the armature object on import.
_CONVERSION = Matrix.Rotation(radians(90), 4, "X")

_TRAILING_EXCLUDE_FLAG = 0x400


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

    def draw(self, context):
        assert self.layout is not None

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "frame_source")
        layout.prop(self, "player_anim")

    def execute(self, context) -> OperatorResult:
        armature = import_aqm._find_target_armature(context)
        if armature is None:
            self.report(
                {"ERROR"},
                "No target armature. Select a PSO2 armature (or a model parented"
                " to one) and try again.",
            )
            return {"CANCELLED"}

        try:
            frame_start, frame_end = self._get_frame_range(context, armature)
            motion = build_motion(
                context,
                armature,
                frame_start=frame_start,
                frame_end=frame_end,
                player_anim=self.player_anim,
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

        # The exporter samples the posed transforms, so a body shape sitting
        # in the pose layer ends up baked into the motion (SPEC §6-8).
        from . import bake_rest

        if armature.animation_data is None or armature.animation_data.action is None:
            if bake_rest.pose_is_modified(armature):
                self.report(
                    {"WARNING"},
                    message + ". The pose holds a body shape and no action, so"
                    " that shape is now part of this motion - apply it to the"
                    " rest pose first if that was not intended.",
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


def build_motion(
    context: bpy.types.Context,
    armature: bpy.types.Object,
    frame_start: int,
    frame_end: int,
    player_anim=False,
) -> aqm.AqmMotion:
    """Bake the armature's animation into a motion, one key per frame."""
    if frame_end < frame_start:
        raise ExportError("Invalid frame range")

    scene = context.scene
    export_bones = _get_export_bones(armature)

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

            # Node 0 is the skeleton root: the armature object itself,
            # relative to the FBX axis conversion.
            _sample_matrix(
                samples[0], _CONVERSION.inverted() @ armature.matrix_basis, None
            )

            for index, name in enumerate(export_bones, start=1):
                pose_bone = armature.pose.bones[name]

                if pose_bone.parent is not None:
                    local = pose_bone.parent.matrix.inverted() @ pose_bone.matrix
                else:
                    local = pose_bone.matrix

                _sample_matrix(samples[index], local, pose_bone.matrix)
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

    for index, (name, keys) in enumerate(zip(node_names, samples)):
        node = aqm.AqmNode(
            node_type=aqm.NODE_TYPE_STANDARD, node_id=index, name=name
        )

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


def _sample_matrix(node_samples: list, local: Matrix, world: Matrix | None):
    """Decompose one frame's transforms into (pos, rot, scale) vec4 keys."""
    location, rotation, local_scale = local.decompose()

    # PSO2 bones do not inherit scale, so the scale channel holds the
    # absolute scale rather than the local one.
    scale = world.to_scale() if world is not None else local_scale

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
        raise ExportError(
            f"Bone IDs are not contiguous. Missing node IDs: {missing}"
        )

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
