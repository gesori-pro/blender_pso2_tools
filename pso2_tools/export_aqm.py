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
from mathutils import Matrix, Vector, kdtree

from . import aqm, classes, import_aqm, physics, scene_props
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
    drop_bone_scale: bpy.props.BoolProperty(
        name="Drop Bone Scale",
        description=(
            "Write every bone's scale as 1. A jiggle add-on baked to"
            " keyframes keys scale alongside rotation, and the scale then"
            " travels with the motion and resizes the body of whoever plays"
            " it. The bounce itself is in the rotation and survives; only"
            " the swelling goes. Every scale key in the game's own player"
            " animations is 1. Turned off automatically for a shape adjust,"
            " whose whole content is scale"
        ),
        default=True,
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
        layout.prop(self, "drop_bone_scale")
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
        # Only keys someone added over the imported motion can hold the
        # shape - the file's own curves never did.
        user_keyed = (
            _keyed_channels(armature, exclude_imported=True)
            if self.ignore_applied_shape
            else None
        )

        try:
            frame_start, frame_end = self._get_frame_range(context, armature)
            with (
                bake_rest.bake_suspended(shaped),
                bake_rest.shape_in_pose_suspended(shaped, keyed),
            ):
                # An action the import stamped carries the body in every
                # keyed channel; strip exactly what the stamp says went in.
                # Without a stamp, only keys someone added over the motion
                # can hold the body.
                shape = bake_rest.composed_body_deltas(shaped, keyed)
                if shape is None:
                    shape = bake_rest.keyed_shape_deltas(shaped, user_keyed)

                motion = build_motion(
                    context,
                    armature,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    player_anim=self.player_anim,
                    shape=shape,
                )
        except ExportError as ex:
            self.report({"ERROR"}, str(ex))
            return {"CANCELLED"}

        path = Path(self.filepath)  # type: ignore

        # A shape adjust is nothing but scale, so leave that one alone.
        dropped = 0
        if self.drop_bone_scale and not path.name.endswith(aqm.SHAPE_ADJUST_SUFFIX):
            dropped = _drop_bone_scale(motion)

        size = aqm.write_aqm(path, motion)

        message = (
            f"Exported {path.name}: {len(motion.nodes)} nodes,"
            f" frames 0-{motion.end_frame}, {size:,} bytes"
        )
        if dropped:
            message += f", scale reset to 1 on {dropped} bones"

        # Cloth the game swings is the one thing no accuracy here can
        # match: Blender holds it at its bind position, the game does not.
        # A hand lined up against a skirt or a frill is therefore lined up
        # against something that will not be there.
        if touching := _pose_touches_physics(context, armature):
            shown = ", ".join(sorted(touching)[:3])
            more = f" and {len(touching) - 3} more" if len(touching) > 3 else ""
            self.report(
                {"WARNING"},
                message + f". The pose puts a hand on cloth the game"
                f" simulates ({shown}{more}). Blender shows that cloth at"
                " rest, so it sits somewhere else in game - place the hand"
                " against the body, not against the cloth.",
            )
            return {"FINISHED"}

        # Position keys carry each bone's offset from its parent, so they
        # are where the skeleton's proportions and the character's placement
        # live - not just movement. An action that only rotates writes the
        # armature's rest offsets instead, which silently swaps the source
        # motion's limb lengths and root height for the model's own. One
        # file built this way put the hips 31 cm low and both shins at an
        # identical 1.92 cm short of the rig they were meant to match.
        if missing := _bones_missing_location(armature):
            shown = ", ".join(sorted(missing)[:4])
            more = f" and {len(missing) - 4} more" if len(missing) > 4 else ""
            self.report(
                {"WARNING"},
                message + f". {len(missing)} bones have no position keys"
                f" ({shown}{more}), so this motion carries the skeleton's"
                " own rest offsets for them. If you started from an existing"
                " motion, its root placement and bone lengths have been"
                " lost - re-import it and pose on top rather than posing"
                " from the rest pose.",
            )
            return {"FINISHED"}

        # The exporter samples the posed transforms, so anything sitting in
        # the pose layer that the action does not drive - a body shape, most
        # often - ends up baked into the motion (SPEC §6-8). With Ignore
        # Body Shape on, those channels were just sampled at their baseline
        # instead, so there is nothing left to warn about.
        # A baked simulation keys every channel it touched, scale included,
        # so the posed-but-unkeyed check above walks straight past it. The
        # game multiplies a motion's scale onto the character's own body, so
        # any bone that leaves here at other than 1 resizes whoever plays it.
        if scaled := _bones_with_scale(motion):
            shown = ", ".join(name for name, _ in sorted(scaled)[:4])
            more = f" and {len(scaled) - 4} more" if len(scaled) > 4 else ""
            worst = max(dev for _, dev in scaled)
            self.report(
                {"WARNING"},
                message + f". {len(scaled)} bones carry a scale other than 1"
                f" ({shown}{more}, up to {worst:+.0%}), which travels with"
                " the motion and resizes the body of whoever plays it."
                " Every scale key in the game's own player animations is 1"
                " - turn on Drop Bone Scale, or clear the scale before"
                " exporting.",
            )
            return {"FINISHED"}

        if not self.ignore_applied_shape and (stuck := _unkeyed_posed_bones(armature)):
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


def _keyed_channels(
    armature: bpy.types.Object, exclude_imported=False
) -> dict[str, set[str]]:
    """Bone name -> the transform channels the active action drives.

    With `exclude_imported`, channels the motion import created are left
    out (import_aqm.IMPORTED_CHANNELS_PROP), keeping only the ones someone
    keyframed afterwards. The difference matters for a loaded body shape:
    an imported curve holds the file's own values, but a new keyframe
    records the pose, and the pose is where the shape sits - so only the
    keys someone added can be carrying it.

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

    imported: set[str] = set()
    if exclude_imported:
        imported = set(action.get(import_aqm.IMPORTED_CHANNELS_PROP, ()))

    driven: dict[str, set[str]] = {}
    for curve in curves:
        if curve.data_path in imported:
            continue
        if match := _BONE_PATH.match(curve.data_path):
            driven.setdefault(match.group(1), set()).add(match.group(2))

    return driven


def _pose_touches_physics(context, armature, limit: float = 0.005) -> set[str]:
    """Cloth bones a hand is resting on.

    The game swings these; Blender does not. A pose lined up against a
    skirt or a frill therefore lands somewhere else in game, which is the
    one difference no amount of accuracy in this add-on can close.
    Reported in cm terms: anything within 5 mm counts as touching.
    """
    marked = physics.get_physics_bones(armature)
    if not marked:
        return set()

    depsgraph = context.evaluated_depsgraph_get()
    hand = re.compile(r"(^|_)(hand|finger|thumb)")

    hand_points: list = []
    cloth_points: list[tuple] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.find_armature() is not armature:
            continue
        groups = {g.index: g.name for g in obj.vertex_groups}
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        matrix = evaluated.matrix_world
        for vertex in mesh.vertices:
            best = max(vertex.groups, key=lambda g: g.weight, default=None)
            if best is None or best.weight < 0.5:
                continue
            name = groups.get(best.group, "")
            stem = name.split("#")[0]
            if name in marked:
                cloth_points.append((matrix @ vertex.co, marked[name]))
            elif hand.search(stem):
                hand_points.append(matrix @ vertex.co)
        evaluated.to_mesh_clear()

    if not hand_points or not cloth_points:
        return set()

    tree = kdtree.KDTree(len(cloth_points))
    for index, (point, _) in enumerate(cloth_points):
        tree.insert(point, index)
    tree.balance()

    touching: set[str] = set()
    for point in hand_points:
        _, index, distance = tree.find(point)
        if distance is not None and distance < limit:
            touching.add(cloth_points[index][1])
    return touching


def _drop_bone_scale(motion) -> int:
    """Reset every scale key to 1. Returns how many bones were changed.

    Only the scale channel: the rotation and position keys are the motion,
    and a jiggle add-on's bounce lives in the rotation, so it comes through
    intact. What goes is the swelling that rides along with it.
    """
    changed = 0
    for node in motion.nodes:
        key_set = node.get_key_set(aqm.KEY_TYPE_SCALE)
        if key_set is None or not key_set.vec4_keys:
            continue
        reset = [(1.0, 1.0, 1.0, w) for _, _, _, w in key_set.vec4_keys]
        if reset != key_set.vec4_keys:
            key_set.vec4_keys = reset
            changed += 1
    return changed


def _bones_with_scale(motion, limit: float = 0.002) -> list[tuple[str, float]]:
    """(bone, worst deviation) for every node the motion resizes.

    Read off the built motion rather than the pose, so it catches the scale
    whatever put it there - a body shape, a baked simulation, a stray S in
    the viewport. Measured across the game's own player animations, every
    scale key is exactly 1; whether the game composes a motion's scale with
    the character's own proportions or replaces them has not been pinned
    down, and either way the body on screen is not the one that was built.
    """
    found = []
    for node in motion.nodes:
        key_set = node.get_key_set(aqm.KEY_TYPE_SCALE)
        if key_set is None:
            continue
        worst = 0.0
        for x, y, z, _ in key_set.vec4_keys:
            worst = max(worst, abs(x - 1.0), abs(y - 1.0), abs(z - 1.0))
        if worst > limit:
            found.append((node.name, worst))
    return found


def _bones_missing_location(armature) -> set[str]:
    """Export bones the action moves without ever keying their position.

    A motion imported by this add-on keys every channel, so an animated
    bone with no position keys means the source motion's own values were
    dropped somewhere along the way.
    """
    keyed = _keyed_channels(armature)
    return {
        name
        for name, channels in keyed.items()
        if channels and "location" not in channels
    }


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
    shape: dict | None = None,
) -> aqm.AqmMotion:
    """Bake the armature's animation into a motion, one key per frame.

    `shape` maps bone names to body-shape deltas the action keyed along
    with the pose (bake_rest.keyed_shape_deltas); they are taken back off
    each sample so the motion stays body-neutral.
    """
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

                # The bone's transform relative to its parent is the rest
                # offset with the pose basis on top. Dividing the posed world
                # matrices instead gives the same thing only while nothing up
                # the chain is scaled unevenly: that puts a scale between two
                # rotations, which is a shear, and the position and rotation
                # read back off a shear are approximations. Rest matrices
                # carry no scale, so this way there is nothing to shear.
                basis = _shape_free_basis(pose_bone, shape)
                if pose_bone.parent is not None:
                    rest = (
                        pose_bone.parent.bone.matrix_local.inverted_safe()
                        @ pose_bone.bone.matrix_local
                    )
                    local = rest @ basis
                    parent_correction = correction
                else:
                    local = pose_bone.bone.matrix_local @ basis
                    # Parented to the armature object, which is not a bone
                    # and so carries no correction.
                    parent_correction = Matrix.Identity(4)

                scale = _absolute_scale(pose_bone, absolute, shape)
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


def _shape_free_basis(pose_bone, shape: dict | None) -> Matrix:
    """The pose basis with any keyed body-shape deltas taken back off.

    The basis is the pose channels as a matrix, T @ R @ S, so each delta
    comes off its own channel exactly: the shape's location offset was
    added, subtract it; its rotation delta was composed innermost,
    multiply the inverse on the right. Scale is left alone here - the
    scale channel is written from _absolute_scale, which does its own
    division, and the basis scale never reaches the position or rotation
    of rest @ basis.
    """
    basis = pose_bone.matrix_basis
    entry = shape.get(pose_bone.name) if shape else None
    if not entry or (entry["location"] is None and entry["rotation"] is None):
        return basis

    location, rotation, scale = basis.decompose()
    if entry["location"] is not None:
        location = location - entry["location"]
    if entry["rotation"] is not None:
        rotation = rotation @ entry["rotation"].inverted()

    return Matrix.LocRotScale(location, rotation, scale)


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
    if shape and (own := (shape.get(pose_bone.name) or {}).get("scale")):
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
    scale sitting between two rotations in a matrix is a shear, and
    decomposing a shear spreads one number across three - a head scaled
    evenly to 1.030 read back as (0.975, 1.047, 1.101) that way. The scale
    passed in is accumulated per channel instead (_absolute_scale), never
    having been inside a matrix with a rotation.
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
    """The __NodeTreeFlag__ pseudo-node official player animations carry.

    Two channels, not three. Both forms ship: over a 250 motion sample,
    140 lobby actions carry 0x10 and 0x11 and 33 carry 0x12 as well, and
    nothing about the file says which - node count, frame count and the
    camera variants all appear on both sides. Two is the common form and
    the one the emotes people replace use, among them pl_hum_lacf_004_yes,
    so an export that always wrote three put a channel into a slot that
    never had one.
    """
    timings = [0x8 | 0x1]
    timings += [frame * multiplier + 0x8 for frame in range(1, frame_count - 1)]
    if frame_count > 1:
        timings.append((frame_count - 1) * multiplier + 0x8 + 0x2)

    node = aqm.AqmNode(
        node_type=aqm.NODE_TYPE_NODE_TREE_FLAG,
        node_id=0,
        name="__NodeTreeFlag__",
    )

    for key_type in (0x10, 0x11):
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
