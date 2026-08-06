"""
Import PSO2 AQM motion files as actions on an existing PSO2 armature.

Motion keys are absolute local-to-parent transforms that replace the bind
pose local transform, applied by node index (matching the bone's
pso2_bone_id custom property). See AquaModelLibrary.Native/FbxExporterCore.cpp
CreateAnimationTakeFromAquaMotion() for the reference implementation.
"""

from dataclasses import dataclass, field
from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper, axis_conversion
from mathutils import Matrix, Quaternion, Vector

from . import aqm, classes, scene_props
from .debug import debug_print
from .util import OperatorResult

_LINEAR = 1  # bpy.types.Keyframe.interpolation enum value for 'LINEAR'

# Nodes that keep their translation keys when "Ignore Translation Keys" is
# enabled: root, body_root, hip.
_ROOT_NODE_COUNT = 3

# Set on an action by apply_motion: the f-curve data paths the import
# created. Motion export uses it to tell file-borne keys, which are
# body-neutral, from keys someone added later off the pose.
IMPORTED_CHANNELS_PROP = "pso2_imported_channels"


@dataclass
class ImportSummary:
    actions: list[bpy.types.Action] = field(default_factory=list)
    matched: int = 0
    matched_by_name: int = 0
    unmatched: list[str] = field(default_factory=list)
    disconnected: list[str] = field(default_factory=list)


@classes.register
class PSO2_OT_ImportAqm(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ImportHelper
):
    """Load a PSO2 AQM motion file onto the active PSO2 armature"""

    bl_label = "Import AQM"
    bl_idname = "pso2.import_aqm"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".aqm"
    filter_glob: bpy.props.StringProperty(
        default="*.aqm;*.trm;*.aqv;*.aqw", options={"HIDDEN"}
    )

    files: bpy.props.CollectionProperty(
        type=bpy.types.OperatorFileListElement, options={"HIDDEN", "SKIP_SAVE"}
    )
    directory: bpy.props.StringProperty(subtype="DIR_PATH", options={"HIDDEN"})

    set_frame_range: bpy.props.BoolProperty(
        name="Update Scene Frame Range",
        description="Set the scene frame range to match the motion",
        default=True,
    )
    set_fps: bpy.props.BoolProperty(
        name="Update Scene Frame Rate",
        description="Set the scene frame rate to the motion's frame rate",
        default=True,
    )
    ignore_translation_keys: bpy.props.BoolProperty(
        name="Ignore Non-Root Translation Keys",
        description=(
            "Only apply position keys to the root nodes and use rotation for"
            " everything else. This keeps the armature's own proportions, which"
            " helps when a motion distorts outfits with customized skeletons"
        ),
        default=False,
    )
    ignore_scale_keys: bpy.props.BoolProperty(
        name="Ignore Scale Keys",
        description=(
            "Drop the motion's scale keys and keep the armature at its own"
            " size. A pose exported from Blender takes the scale straight off"
            " the bones, so an author who had a body shape loaded writes"
            " their proportions into the file - one arm thicker than the"
            " other is the usual sign. Turn this on to take the pose without"
            " the body it was made on"
        ),
        default=False,
    )
    disconnect_bones: bpy.props.BoolProperty(
        name="Disconnect Animated Bones",
        description=(
            "Disconnect bones that have position keys from their parents."
            " Blender ignores the location of connected bones, which would"
            " break the motion. This does not change the rest pose"
        ),
        default=True,
    )

    # Captured at invoke time: the file browser's context has no armature
    # to inspect by the time draw() runs.
    shape_state: bpy.props.StringProperty(options={"HIDDEN", "SKIP_SAVE"})

    def draw(self, context):
        assert self.layout is not None

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "set_frame_range")
        layout.prop(self, "set_fps")
        layout.prop(self, "ignore_translation_keys")
        layout.prop(self, "ignore_scale_keys")
        layout.prop(self, "disconnect_bones")

        from . import bake_rest

        bake_rest.draw_shape_state(layout, self.shape_state)

    def execute(self, context) -> OperatorResult:
        armature = _find_target_armature(context)
        if armature is None:
            self.report(
                {"ERROR"},
                "No target armature. Select a PSO2 armature (or a model parented"
                " to one) and try again.",
            )
            return {"CANCELLED"}

        paths = [
            Path(self.directory) / file.name
            for file in self.files  # type: ignore
            if file.name
        ]
        if not paths:
            paths = [Path(self.filepath)]  # type: ignore

        summary = ImportSummary()

        for path in paths:
            try:
                motion = aqm.read_aqm(path)
            except (OSError, aqm.AqmError) as ex:
                self.report({"ERROR"}, f"{path.name}: {ex}")
                return {"CANCELLED"}

            if motion.is_camera_motion:
                self.report({"ERROR"}, f"{path.name}: camera motions are not supported")
                return {"CANCELLED"}

            if motion.is_material_motion:
                self.report(
                    {"ERROR"}, f"{path.name}: material motions are not supported"
                )
                return {"CANCELLED"}

            # Shape adjusts are .aqm files too, so they turn up in this file
            # browser. Loading one as a motion keys every bone to its static
            # value and twists the legs, and the damage is easy to mistake
            # for the shape itself being wrong.
            if aqm.is_shape_adjust_file(path, motion):
                self.report(
                    {"ERROR"},
                    f"{path.name} looks like a shape adjust, not a motion."
                    " Load it from Scene > PSO2 Appearance > Shape Adjust"
                    " instead.",
                )
                return {"CANCELLED"}

            if self.disconnect_bones and not self.ignore_translation_keys:
                _disconnect_animated_bones(context, armature, motion, summary)

            apply_motion(
                armature,
                motion,
                name=path.stem,
                summary=summary,
                ignore_translation_keys=self.ignore_translation_keys,
                ignore_scale_keys=self.ignore_scale_keys,
            )

            if self.set_frame_range:
                # MOHeader.end_frame is sometimes wrong (some mod tools
                # write it incorrectly), so use the real last keyframe if
                # it's later than what the header claims.
                context.scene.frame_start = 0
                context.scene.frame_end = max(motion.end_frame, motion.max_key_frame())

            if self.set_fps:
                fps = max(1, round(motion.frame_speed))
                context.scene.render.fps = fps
                context.scene.render.fps_base = fps / motion.frame_speed

        if summary.unmatched:
            debug_print("AQM nodes with no matching bone:", summary.unmatched)

        names = ", ".join(action.name for action in summary.actions)
        message = f"Imported {names}: {summary.matched} nodes matched"
        if summary.matched_by_name:
            message += f" ({summary.matched_by_name} by name)"
        if summary.unmatched:
            message += f", {len(summary.unmatched)} skipped"
        if summary.disconnected:
            message += f", {len(summary.disconnected)} bones disconnected"

        # An action drives the same pose channels the body shape lives in,
        # so playing it will discard the shape unless it has been baked
        # into the rest pose first (SPEC §6-8).
        from . import bake_rest

        if bake_rest.pose_is_modified(armature):
            self.report(
                {"WARNING"},
                message + ". The armature has a body shape in its pose, which"
                " this animation will override - use Apply Shape to Rest Pose"
                " (PSO2 Appearance panel) first.",
            )
            return {"FINISHED"}

        self.report({"INFO"}, message)

        return {"FINISHED"}

    def invoke(self, context, event):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
        from . import bake_rest

        self.shape_state = bake_rest.shape_state(_find_target_armature(context))
        return self.invoke_popup(context)


def _find_target_armature(context: bpy.types.Context) -> bpy.types.Object | None:
    def armature_of(obj: bpy.types.Object | None) -> bpy.types.Object | None:
        if obj is None:
            return None

        if obj.type == "ARMATURE":
            return obj

        if obj.parent is not None and obj.parent.type == "ARMATURE":
            return obj.parent

        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object is not None:
                return modifier.object

        return None

    # Not every context exposes a selection: right after a file load, or when
    # an operator runs from a non-3D-view area, reading these raises instead
    # of returning None.
    if armature := armature_of(getattr(context, "active_object", None)):
        return armature

    for obj in getattr(context, "selected_objects", None) or []:
        if armature := armature_of(obj):
            return armature

    view_layer = getattr(context, "view_layer", None)
    if view_layer is None:
        return None

    armatures = [obj for obj in view_layer.objects if obj.type == "ARMATURE"]
    if len(armatures) == 1:
        return armatures[0]

    return None


def _disconnect_animated_bones(
    context: bpy.types.Context,
    armature: bpy.types.Object,
    motion: aqm.AqmMotion,
    summary: ImportSummary,
):
    """Disconnect bones with position keys so their location can animate.

    Blender ignores the pose location of connected bones. Disconnecting only
    removes that restriction; it does not move the bone or change the rest
    pose.
    """
    bones_by_id, bones_by_name = _get_bone_maps(armature)

    names = []
    for index, node in enumerate(motion.nodes):
        if node.get_key_set(aqm.KEY_TYPE_POSITION) is None:
            continue

        name = bones_by_id.get(index) or bones_by_name.get(node.name.lower())
        if name is None:
            continue

        names.append(name)

    summary.disconnected.extend(disconnect_bones(context, armature, names))


def disconnect_bones(
    context: bpy.types.Context, armature: bpy.types.Object, names
) -> list[str]:
    """Detach the named bones from their parents, and say which moved.

    Blender ignores the pose location of a connected bone, so anything that
    means to translate one has to do this first. Names that are absent or
    already disconnected are skipped. Disconnecting only lifts the
    restriction: it does not move the bone or change the rest pose.
    """
    data: bpy.types.Armature = armature.data  # type: ignore

    names = [
        name
        for name in dict.fromkeys(names)
        if (bone := data.bones.get(name)) is not None and bone.use_connect
    ]
    if not names:
        return []

    disconnected: list[str] = []
    view_layer = context.view_layer
    previous_active = view_layer.objects.active
    previous_mode = previous_active.mode if previous_active else "OBJECT"
    was_hidden = armature.hide_get()
    was_hidden_viewport = armature.hide_viewport

    try:
        if previous_active is not None and previous_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        armature.hide_viewport = False
        armature.hide_set(False)
        view_layer.objects.active = armature

        bpy.ops.object.mode_set(mode="EDIT")
        try:
            for name in names:
                data.edit_bones[name].use_connect = False
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")

        disconnected = names
    except RuntimeError as ex:
        debug_print("Could not disconnect bones:", ex)
    finally:
        armature.hide_set(was_hidden)
        armature.hide_viewport = was_hidden_viewport

        try:
            view_layer.objects.active = previous_active
            if previous_active is not None and previous_mode != "OBJECT":
                bpy.ops.object.mode_set(mode=previous_mode)
        except RuntimeError:
            pass

    return disconnected


def apply_motion(
    armature: bpy.types.Object,
    motion: aqm.AqmMotion,
    name: str,
    summary: ImportSummary | None = None,
    ignore_translation_keys=False,
    ignore_scale_keys=False,
):
    """Create an action from a motion and assign it to the armature."""
    summary = summary if summary is not None else ImportSummary()

    bones_by_id, bones_by_name = _get_bone_maps(armature)

    resolved = _resolve_nodes(motion, armature, bones_by_id, bones_by_name)
    parent_ids = _get_parent_ids(armature, resolved)
    aqm.prepare_scaling(motion, parent_ids)
    correction = bone_correction(armature)

    action = bpy.data.actions.new(name)
    action.use_fake_user = True

    slot = action.slots.new(id_type="OBJECT", name=armature.name)
    layer = action.layers.new("Layer")
    strip = layer.strips.new(type="KEYFRAME")
    channelbag = strip.channelbag(slot, ensure=True)

    for index, node in enumerate(motion.nodes):
        pose_bone = None

        if (bone_name := bones_by_id.get(index)) is not None:
            pose_bone = armature.pose.bones.get(bone_name)

        if pose_bone is None:
            key = node.name.lower()
            if (bone_name := bones_by_name.get(key)) is not None:
                pose_bone = armature.pose.bones.get(bone_name)
                if pose_bone is not None:
                    summary.matched_by_name += 1

        if pose_bone is None:
            if index == 0:
                # Node 0 is the skeleton root, which the model importer turns
                # into the armature object itself.
                _apply_object_motion(armature, node, channelbag)
            elif any(
                key_set.key_type
                in (aqm.KEY_TYPE_POSITION, aqm.KEY_TYPE_ROTATION, aqm.KEY_TYPE_SCALE)
                for key_set in node.key_sets
            ):
                summary.unmatched.append(node.name or f"node {index}")
            continue

        ignore_translation = ignore_translation_keys and index >= _ROOT_NODE_COUNT
        _apply_bone_motion(
            pose_bone,
            node,
            channelbag,
            ignore_translation,
            correction,
            ignore_scale_keys,
        )
        summary.matched += 1

    # Every curve on the action right now came from the file, so its keyed
    # values are the motion's own, with no body shape in them. A channel
    # keyframed later starts from the pose instead, and the pose is where a
    # loaded shape lives - motion export tells the two apart by this stamp
    # to know which keys carry the shape (SPEC §6-8).
    action[IMPORTED_CHANNELS_PROP] = sorted(
        {curve.data_path for curve in channelbag.fcurves}
    )

    animation_data = armature.animation_data_create()
    animation_data.action = action
    animation_data.action_slot = slot

    summary.actions.append(action)
    return action


def bone_correction(armature: bpy.types.Object) -> Matrix:
    """The rotation the FBX importer put on every bone's local axes.

    A Blender bone always runs down its own +Y, while a PSO2 node runs down
    +X, so io_scene_fbx rotates each bone as it builds the armature. Motion
    keys are written in PSO2's axes, so reading one back has to undo the same
    rotation - otherwise a key that only restates the bind offset comes out
    as a translation of root(2) times the bone's length, and the model tears
    itself apart.

    This mirrors io_scene_fbx.import_fbx's own bone_correction_matrix, built
    from the axes the model importer recorded on the armature.
    """
    axes = str(armature.get(scene_props.BONE_AXES) or scene_props.DEFAULT_BONE_AXES)
    parts = axes.split(",")
    if len(parts) != 2:
        # "AUTO", or a rig from somewhere else: bones point at their
        # children and no fixed correction applies.
        return Matrix.Identity(4)

    primary, secondary = (part.strip() for part in parts)
    if (primary, secondary) == ("Y", "X"):
        # What io_scene_fbx treats as no correction at all.
        return Matrix.Identity(4)

    try:
        return axis_conversion(
            from_forward="X",
            from_up="Y",
            to_forward=secondary,
            to_up=primary,
        ).to_4x4()
    except ValueError:
        return Matrix.Identity(4)


def _get_bone_maps(armature: bpy.types.Object):
    """Maps of PSO2 node index -> bone name and node name -> bone name."""
    bones_by_id: dict[int, str] = {}
    bones_by_name: dict[str, str] = {}

    for bone in armature.data.bones:  # type: ignore
        bone_id = bone.get(scene_props.BONE_ID)
        if bone_id is not None and int(bone_id) not in bones_by_id:
            bones_by_id[int(bone_id)] = bone.name

        base_name = bone.name.split("#")[0].lower()
        if base_name and base_name not in bones_by_name:
            bones_by_name[base_name] = bone.name

    return bones_by_id, bones_by_name


def _resolve_nodes(
    motion,
    armature: bpy.types.Object,
    bones_by_id: dict[int, str],
    bones_by_name: dict[str, str],
) -> dict[int, str]:
    """Node index -> the bone its keys land on, by ID and then by name.

    The hierarchy the scale conversion needs is built from this, so it has
    to be worked out the same way the keys themselves are matched. An
    armature that came from somewhere other than this add-on carries no
    pso2_bone_id at all - swapping the rig for one out of another file is
    enough - and going by ID alone then finds nothing, leaving every scale
    key undivided. Blender bones inherit scale, so the error compounds down
    each limb: measured on one pose, 209 of 211 bones came out wrong and
    the worst reached 3.5 times its size.
    """
    resolved: dict[int, str] = {}

    for index, node in enumerate(motion.nodes):
        bone_name = bones_by_id.get(index)
        if bone_name is None:
            bone_name = bones_by_name.get(node.name.lower())

        if bone_name is not None and bone_name in armature.pose.bones:
            resolved[index] = bone_name

    return resolved


def _get_parent_ids(armature: bpy.types.Object, bones_by_index: dict[int, str]):
    """Maps a PSO2 node index to its parent's node index."""
    ids_by_bone = {name: index for index, name in bones_by_index.items()}
    parent_ids: dict[int, int] = {}

    for index, name in bones_by_index.items():
        bone = armature.data.bones[name]  # type: ignore
        if bone.parent is not None:
            parent_ids[index] = ids_by_bone.get(bone.parent.name, -1)
        else:
            # Parentless bones hang off the skeleton root (node 0).
            parent_ids[index] = 0

    return parent_ids


def _apply_bone_motion(
    pose_bone: bpy.types.PoseBone,
    node: aqm.AqmNode,
    channelbag,
    ignore_translation: bool,
    correction: Matrix,
    ignore_scale: bool = False,
):
    """Convert a node's local-space keys to pose-space keyframes.

    A key is in the PSO2 node's own axes, which the FBX importer rotated by
    C when it built the bone (see bone_correction). Undoing that turns the
    node's local matrix into the bone's:

        L_bone = C_parent⁻¹ @ L_anim @ C

    With rest local matrix L_rest = T_r @ R_r (PSO2 bind poses have no scale)
    and L_anim = T(p) @ R(q) @ S(s), that splits back into one term per
    channel, because C is a signed axis permutation:

        p' = C_parent⁻¹ @ p
        R' = C_parent⁻¹ @ R(q) @ C
        S' = C⁻¹ @ S(s) @ C                     (still diagonal)

        matrix_basis = L_rest⁻¹ @ T(p') @ R' @ S'
                     = T(R_r⁻¹ @ (p' - t_r)) @ (R_r⁻¹ @ R') @ S'

    so each channel still depends only on its own key set and native key
    timings can be kept, matching the game's linear interpolation.
    """
    bone = pose_bone.bone

    if bone.parent is not None:
        rest = bone.parent.matrix_local.inverted() @ bone.matrix_local
        parent_correction = correction
    else:
        rest = bone.matrix_local.copy()
        # A parentless bone hangs off the skeleton root, which the model
        # importer turns into the armature object. That is not a bone, so it
        # carries no correction.
        parent_correction = Matrix.Identity(4)

    rest_translation = rest.to_translation()
    rest_rotation_inv = rest.to_quaternion().inverted()

    correction3 = correction.to_3x3()
    correction3_inv = correction3.inverted()
    parent_correction3_inv = parent_correction.to_3x3().inverted()

    prefix = f'pose.bones["{pose_bone.name}"]'
    group = pose_bone.name

    if not ignore_translation and (key_set := node.get_key_set(aqm.KEY_TYPE_POSITION)):
        values = [
            rest_rotation_inv
            @ ((parent_correction3_inv @ Vector(key[:3])) - rest_translation)
            for key in key_set.vec4_keys
        ]
        _add_fcurves(
            channelbag, f"{prefix}.location", group, key_set.frames(), values, 3
        )

    if key_set := node.get_key_set(aqm.KEY_TYPE_ROTATION):
        values = []
        for key in key_set.vec4_keys:
            # AQM quaternions are stored XYZW.
            source = Quaternion((key[3], key[0], key[1], key[2])).normalized()
            rotation = (
                rest_rotation_inv
                @ (
                    parent_correction3_inv @ source.to_matrix() @ correction3
                ).to_quaternion()
            )
            rotation.normalize()

            # Keep consecutive keys on the same hemisphere so per-component
            # interpolation does not flip.
            if values and values[-1].dot(rotation) < 0:
                rotation.negate()

            values.append(rotation)

        pose_bone.rotation_mode = "QUATERNION"
        _add_fcurves(
            channelbag,
            f"{prefix}.rotation_quaternion",
            group,
            key_set.frames(),
            values,
            4,
        )

    if not ignore_scale and (key_set := node.get_key_set(aqm.KEY_TYPE_SCALE)):
        values = [
            (
                correction3_inv @ Matrix.Diagonal(Vector(key[:3])) @ correction3
            ).to_scale()
            for key in key_set.vec4_keys
        ]

        if any((value - Vector((1, 1, 1))).length > 1e-4 for value in values):
            _add_fcurves(
                channelbag, f"{prefix}.scale", group, key_set.frames(), values, 3
            )


def _apply_object_motion(armature: bpy.types.Object, node: aqm.AqmNode, channelbag):
    """Apply skeleton root motion to the armature object itself.

    The armature object's current transform holds the FBX-to-Blender axis
    conversion, so animated transforms are applied relative to it.
    """
    position = node.get_key_set(aqm.KEY_TYPE_POSITION)
    rotation = node.get_key_set(aqm.KEY_TYPE_ROTATION)
    scale = node.get_key_set(aqm.KEY_TYPE_SCALE)

    identity = True
    if position and any(Vector(key[:3]).length > 1e-6 for key in position.vec4_keys):
        identity = False
    if rotation and any(abs(1 - abs(key[3])) > 1e-6 for key in rotation.vec4_keys):
        identity = False
    if scale and any(
        (Vector(key[:3]) - Vector((1, 1, 1))).length > 1e-4 for key in scale.vec4_keys
    ):
        identity = False

    if identity:
        # The root is static: leave the armature object untouched.
        return

    base = armature.matrix_basis.copy()
    base_rotation = base.to_quaternion()

    if position:
        values = [base @ Vector(key[:3]) for key in position.vec4_keys]
        _add_fcurves(channelbag, "location", None, position.frames(), values, 3)

    if rotation:
        values = []
        for key in rotation.vec4_keys:
            value = base_rotation @ Quaternion((key[3], key[0], key[1], key[2]))
            value.normalize()

            if values and values[-1].dot(value) < 0:
                value.negate()

            values.append(value)

        armature.rotation_mode = "QUATERNION"
        _add_fcurves(
            channelbag, "rotation_quaternion", None, rotation.frames(), values, 4
        )

    if scale:
        values = [Vector(key[:3]) for key in scale.vec4_keys]
        _add_fcurves(channelbag, "scale", None, scale.frames(), values, 3)


def _add_fcurves(
    channelbag, data_path: str, group_name: str | None, frames, values, count: int
):
    """Create keyframes for one property, one F-curve per array index."""
    # Duplicate frames can appear (the final frame is often keyed twice, with
    # and without the end-flag timing). dict() keeps the last value.
    unique = dict(zip(frames, values, strict=True))
    items = sorted(unique.items())

    group = None
    if group_name is not None:
        group = channelbag.groups.get(group_name) or channelbag.groups.new(group_name)

    for index in range(count):
        fcurve = channelbag.fcurves.new(data_path, index=index)
        if group is not None:
            fcurve.group = group

        points = fcurve.keyframe_points
        points.add(len(items))

        coordinates = [0.0] * (len(items) * 2)
        coordinates[0::2] = [float(frame) for frame, _ in items]
        coordinates[1::2] = [value[index] for _, value in items]

        points.foreach_set("co", coordinates)
        points.foreach_set("interpolation", [_LINEAR] * len(items))
        fcurve.update()
