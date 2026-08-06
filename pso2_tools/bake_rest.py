"""
Bake the current body shape into the armature's rest pose.

Blender bones carry two layers: the rest pose the mesh is bound to, and a
pose transform on top of it. Body proportions and shape adjustments are
written to the pose layer - and so are imported animations, as actions.
Playing an animation therefore overwrites the body shape, and exporting a
motion bakes the body shape into it.

The game does not have this problem: proportions reshape the skeleton
itself and motions play on top of the reshaped skeleton. Applying the pose
as the rest pose reproduces that, freeing the pose layer for animation
(SPEC §6-8).

This is deliberately a button rather than something the character import
does on its own: once baked, the shape is part of the skeleton and the
sliders no longer have a pose to modify, so re-importing the character
file is the way back.
"""

import contextlib
from collections.abc import Iterable

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import classes, import_aqm, import_fnp, shape_sliders
from .util import OperatorResult

# Where the body shape currently lives.
STATE_REST = "REST"  # nothing in the pose - ready for animation
STATE_SHAPE_IN_POSE = "SHAPE_IN_POSE"  # shape in the pose, not baked yet
STATE_ANIMATED = "ANIMATED"  # an action owns the pose

# Set on the armature by the bake: what it takes to put things back. The
# datablock copies it names carry a fake user, so a revert still works after
# saving and reloading, where Blender's own undo stack is gone.
SNAPSHOT_PROP = "pso2_prebake"

# Pose bone properties the bake clears, kept so a revert can restore them.
_SNAPSHOT_BONE_PROPS = (import_fnp.MODEL_POSE_PROP, shape_sliders.BASE_PROP)

# Panel help. Blender labels do not wrap, so these are pre-wrapped;
# an empty string draws as a separator.
WORKFLOW_HELP = (
    "A Blender skeleton has two layers: the rest pose",
    "the mesh is built on, and a pose on top of it.",
    "",
    "The sliders write the body shape into the pose.",
    "Animations use that same pose, so importing one",
    "overwrites the shape, and exporting one bakes the",
    "shape into the motion.",
    "",
    "Apply Shape to Rest Pose moves the shape down into",
    "the skeleton itself, leaving the pose free for",
    "animation. That is how the game works: proportions",
    "reshape the skeleton, motions play on top of it.",
    "",
    "Order of work:",
    "     1.  Import a character file",
    "     2.  Adjust the sliders",
    "     3.  Export AQM  (only for a shape mod)",
    "     4.  Apply Shape to Rest Pose",
    "     5.  Import or export animations",
)


def take_snapshot(armature: bpy.types.Object, meshes) -> None:
    """Copy everything the bake is about to overwrite onto the armature.

    Baking rewrites three things at once - every deformed mesh's vertices,
    the armature's rest pose, and the pose layer with the properties hanging
    off it - so putting it back means keeping all three. The meshes and the
    armature are kept as datablock copies with a fake user; the pose is small
    enough to store as plain numbers.
    """
    discard_snapshot(armature)

    rest_name = armature.data.name  # type: ignore
    rest = armature.data.copy()  # type: ignore
    rest.use_fake_user = True
    rest.name = f"{rest_name}.prebake"

    mesh_copies = {}
    for mesh in meshes:
        original = mesh.data.name  # type: ignore
        copy = mesh.data.copy()  # type: ignore
        copy.use_fake_user = True
        copy.name = f"{original}.prebake"
        # Keep the name the data had, so a revert can put it back rather
        # than leaving the object on something called ".prebake.001".
        mesh_copies[mesh.name] = [copy.name, original]

    pose = {}
    bone_props = {}
    for bone in armature.pose.bones:
        bone.rotation_mode = "QUATERNION"
        pose[bone.name] = (
            list(bone.location) + list(bone.rotation_quaternion) + list(bone.scale)
        )
        kept = {
            name: list(bone[name])
            for name in _SNAPSHOT_BONE_PROPS
            if bone.get(name) is not None
        }
        if kept:
            bone_props[bone.name] = kept

    armature[SNAPSHOT_PROP] = {
        "armature": [rest.name, rest_name],
        "meshes": mesh_copies,
        "pose": pose,
        "bone_props": bone_props,
    }


def has_snapshot(armature: bpy.types.Object | None) -> bool:
    """Is there a stored pre-bake state, with its copies still present?"""
    if armature is None:
        return False

    snapshot = armature.get(SNAPSHOT_PROP)
    if not snapshot:
        return False

    if bpy.data.armatures.get(next(iter(snapshot["armature"]))) is None:
        return False

    return all(
        bpy.data.meshes.get(next(iter(entry))) is not None
        for entry in dict(snapshot["meshes"]).values()
    )


def discard_snapshot(armature: bpy.types.Object) -> None:
    """Drop a stored pre-bake state and the copies it holds."""
    snapshot = armature.get(SNAPSHOT_PROP)
    if not snapshot:
        return

    rest = bpy.data.armatures.get(next(iter(snapshot["armature"])))
    if rest is not None and rest is not armature.data:
        rest.use_fake_user = False
        if rest.users == 0:
            bpy.data.armatures.remove(rest)

    for entry in dict(snapshot["meshes"]).values():
        mesh = bpy.data.meshes.get(next(iter(entry)))
        if mesh is None:
            continue
        mesh.use_fake_user = False
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    del armature[SNAPSHOT_PROP]


def restore_snapshot(armature: bpy.types.Object) -> list[str]:
    """Put back the state the bake replaced. Returns what could not be found."""
    snapshot = armature[SNAPSHOT_PROP]
    missing: list[str] = []

    for object_name, entry in dict(snapshot["meshes"]).items():
        copy_name, original_name = list(entry)
        obj = bpy.data.objects.get(object_name)
        mesh = bpy.data.meshes.get(copy_name)
        if obj is None or mesh is None:
            missing.append(object_name)
            continue
        stale = obj.data
        obj.data = mesh
        mesh.use_fake_user = False
        if stale is not None and stale.users == 0:
            bpy.data.meshes.remove(stale)  # type: ignore
        mesh.name = original_name

    copy_name, original_name = list(snapshot["armature"])
    rest = bpy.data.armatures.get(copy_name)
    if rest is None:
        missing.append(str(armature.data.name))  # type: ignore
    else:
        stale = armature.data
        armature.data = rest
        rest.use_fake_user = False
        if stale is not None and stale.users == 0:
            bpy.data.armatures.remove(stale)  # type: ignore
        rest.name = original_name

    pose = dict(snapshot["pose"])
    bone_props = dict(snapshot["bone_props"])
    for bone in armature.pose.bones:
        values = pose.get(bone.name)
        if values is None:
            continue
        bone.rotation_mode = "QUATERNION"
        bone.location = values[0:3]
        bone.rotation_quaternion = values[3:7]
        bone.scale = values[7:10]

        for name in _SNAPSHOT_BONE_PROPS:
            if bone.get(name) is not None:
                del bone[name]
        for name, value in dict(bone_props.get(bone.name, {})).items():
            bone[name] = list(value)

    discard_snapshot(armature)
    return missing


@contextlib.contextmanager
def bake_suspended(armature: bpy.types.Object | None):
    """Put the model back to its unbaked shape for a while.

    Applying the shape to the rest pose writes it into the mesh, so a model
    exported afterwards carries the shape - and the game reshapes the
    skeleton by the character's own proportions on top of that, applying it
    twice. Swapping the stored pre-bake copies in for the export keeps the
    file neutral without disturbing what is on screen.

    Does nothing when there is no stored state to swap in. Everything goes
    back even if the block raises.
    """
    if armature is None or not has_snapshot(armature):
        yield
        return

    snapshot = armature[SNAPSHOT_PROP]
    swapped = []

    def update():
        if view_layer := getattr(bpy.context, "view_layer", None):
            view_layer.update()

    def swap(obj, replacement):
        """Put `replacement` on `obj`, borrowing the live data's name.

        The conversion reads a mesh's PSO2 settings out of its data name, so
        a copy called "..._mesh.prebake" fails to parse. Both names are put
        back when the block ends.
        """
        live = obj.data
        live_name, copy_name = live.name, replacement.name
        swapped.append((obj, live, live_name, replacement, copy_name))
        obj.data = replacement
        live.name = f"{live_name}.swapped"
        replacement.name = live_name

    try:
        for object_name, entry in dict(snapshot["meshes"]).items():
            obj = bpy.data.objects.get(object_name)
            mesh = bpy.data.meshes.get(next(iter(entry)))
            if obj is not None and mesh is not None:
                swap(obj, mesh)

        rest = bpy.data.armatures.get(next(iter(snapshot["armature"])))
        if rest is not None:
            swap(armature, rest)

        update()
        yield
    finally:
        for obj, live, live_name, replacement, copy_name in swapped:
            obj.data = live
            replacement.name = copy_name
            live.name = live_name
        update()


def _deformed_meshes(armature: bpy.types.Object) -> list[bpy.types.Object]:
    """Meshes this armature deforms through an Armature modifier."""
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and any(
            modifier.type == "ARMATURE" and modifier.object is armature
            for modifier in obj.modifiers
        )
    ]


def _freeze_deformation(
    context: bpy.types.Context, mesh: bpy.types.Object, armature: bpy.types.Object
) -> None:
    """Bake the mesh's current deformation into its vertices.

    Applies a *copy* of the armature modifier, so the mesh keeps the shape
    it has right now while the original modifier stays in place to receive
    animation afterwards.
    """
    source = next(
        modifier
        for modifier in mesh.modifiers
        if modifier.type == "ARMATURE" and modifier.object is armature
    )

    context.view_layer.objects.active = mesh
    copied = mesh.modifiers.new(name="pso2_freeze", type="ARMATURE")
    copied.object = armature  # type: ignore
    copied.use_deform_preserve_volume = source.use_deform_preserve_volume  # type: ignore

    # Apply it where the original sits, so it sees the same input. A single
    # move call, never a loop: if it cannot move, applying it at the end of
    # the stack is still correct for the usual single-modifier mesh.
    with contextlib.suppress(RuntimeError, ValueError):
        bpy.ops.object.modifier_move_to_index(
            modifier=copied.name, index=list(mesh.modifiers).index(source)
        )

    bpy.ops.object.modifier_apply(modifier=copied.name)


def pose_is_modified(
    armature: bpy.types.Object,
    epsilon: float = 1e-6,
    bones: Iterable[bpy.types.PoseBone] | None = None,
) -> bool:
    """Is any pose bone transformed away from its rest position?

    Pass `bones` to ask about part of the skeleton rather than all of it.

    The basis matrix is what gets compared, rather than the location,
    rotation and scale fields: a bone posed in euler mode leaves
    rotation_quaternion at identity, and reading that field alone would call
    the bone unposed.
    """
    identity = Matrix.Identity(4)
    for pose_bone in armature.pose.bones if bones is None else bones:
        if any(
            abs(a - b) > epsilon
            for row, rest in zip(pose_bone.matrix_basis, identity, strict=True)
            for a, b in zip(row, rest, strict=True)
        ):
            return True

    return False


@contextlib.contextmanager
def shape_in_pose_suspended(
    armature: bpy.types.Object | None, keyed: dict[str, set[str]] | None
):
    """Take a body shape back off the channels an action does not drive.

    A character file writes the body's proportions into the pose layer, and
    a motion exported from there carries them - the file then reshapes
    every character that plays it, on top of their own proportions.

    An action loaded over the top hides most of this, because its keys win
    on the channels it covers. Only most: a pose mod keys scale on the
    bones it moves and leaves the rest alone, so whatever the shape put on
    those bones is what gets written out. Measured with a character file
    loaded and a pose mod over it, nine finger bones came through carrying
    the character's proportions - which is enough to put a fingertip
    somewhere else in game than it sits on screen.

    The channels the action drives are left alone: those are the pose, and
    for a motion the pose is the point. Everything else goes back to what
    the model import stored, which is the body before any character file
    (import_fnp.store_model_pose).

    Does nothing without that baseline, so a model that never had a
    character file applied is untouched.
    """
    if armature is None or keyed is None or not import_fnp.has_model_pose(armature):
        yield
        return

    saved = []
    channels = (
        ("location", slice(0, 3)),
        ("rotation_quaternion", slice(3, 7)),
        ("scale", slice(7, 10)),
    )

    try:
        for pose_bone in armature.pose.bones:
            baseline = pose_bone.get(import_fnp.MODEL_POSE_PROP)
            if baseline is None or len(baseline) != 10:
                continue

            driven = keyed.get(pose_bone.name, set())
            for name, part in channels:
                if name in driven:
                    continue

                current = getattr(pose_bone, name).copy()
                wanted = list(baseline)[part]
                if all(abs(a - b) < 1e-6 for a, b in zip(current, wanted, strict=True)):
                    continue

                saved.append((pose_bone, name, current))
                if name == "rotation_quaternion":
                    pose_bone.rotation_mode = "QUATERNION"
                setattr(pose_bone, name, wanted)

        _update()
        yield
    finally:
        for pose_bone, name, value in reversed(saved):
            setattr(pose_bone, name, value)
        _update()


def composed_body_deltas(
    armature: bpy.types.Object | None, keyed: dict[str, set[str]] | None
) -> dict[str, dict] | None:
    """The body the motion import composed into the active action's curves.

    Returns the same shape build_motion consumes - bone name to per-channel
    deltas, gated to the channels the action drives - or None when the
    action carries no composition stamp, in which case the caller falls
    back to the user-keyed guesswork. Taking off exactly what the stamp
    says went in keeps the pair exact whatever the body was.
    """
    if armature is None or keyed is None:
        return None

    animation_data = armature.animation_data
    action = animation_data.action if animation_data else None
    stamped = action.get(import_aqm.COMPOSED_BODY_PROP) if action else None
    if not stamped:
        return None

    out: dict[str, dict] = {}
    for name, packed in dict(stamped).items():
        if len(packed) != 10:
            continue
        driven = keyed.get(name, set())
        if not driven:
            continue

        scale = tuple(packed[0:3])
        location = Vector(packed[3:6])
        rotation = Quaternion(packed[6:10])

        piece = {
            "scale": (
                scale
                if "scale" in driven and any(abs(v - 1.0) > 1e-9 for v in scale)
                else None
            ),
            "location": (
                location if "location" in driven and location.length > 1e-9 else None
            ),
            "rotation": (
                rotation
                if "rotation_quaternion" in driven
                and abs(abs(rotation.w) - 1.0) > 1e-12
                else None
            ),
        }
        if any(v is not None for v in piece.values()):
            out[name] = piece

    return out


def keyed_shape_deltas(
    armature: bpy.types.Object | None, keyed: dict[str, set[str]] | None
) -> dict[str, dict]:
    """The shape's own deltas on channels the action also keys.

    Putting a channel back to its baseline only works while the action
    leaves that channel alone. Someone who turns a bone and presses the
    keyframe button keys location, rotation and scale together, so the
    shape's deltas on that bone become keys and read as part of the pose.
    Move the arm on a body whose file thickens the arms, and the arm goes
    out thickened - and every bone below it with it, since a child's
    absolute scale is its parents' multiplied through. The position deltas
    are worse than the scale ones: one file on hand carries 100 mm on the
    hip twists and 50 mm on the breasts.

    Nor can any of it be corrected on the pose: sampling steps the frame,
    which re-runs the action and puts the keyed values straight back. It
    has to come off the sampled numbers instead, which is what this feeds.

    The deltas a loaded file asked for are known exactly - shape_sliders
    keeps them - so they are taken off rather than guessed at from what is
    or is not keyed. Scale divides and location subtracts, both exact.
    Rotation multiplies the delta's inverse on the right, exact because
    the delta went on innermost and later edits compose outside it.

    `keyed` must be the channels someone keyframed over the imported
    motion (export_aqm._keyed_channels with exclude_imported), not
    everything the action drives: the file's own curves hold the motion's
    values with no shape in them, and taking a delta off those would put
    the error in rather than out. A channel keyed before the shape was
    loaded is indistinguishable from one keyed after and is treated as
    carrying it.

    Returns bone name -> {"scale", "location", "rotation"}, each None when
    that channel is unkeyed or the delta is identity.
    """
    if armature is None or keyed is None:
        return {}

    carried = shape_sliders.get_carried(bpy.context)
    if not carried:
        return {}

    bones_by_id, bones_by_name = import_aqm._get_bone_maps(armature)
    out: dict[str, dict] = {}

    for index, entry in carried.items():
        name = bones_by_id.get(index) or bones_by_name.get(entry["name"].lower())
        pose_bone = armature.pose.bones.get(name) if name else None
        if pose_bone is None:
            continue

        driven = keyed.get(name, set())
        if not driven:
            continue

        scale_mul, loc_off, delta_local = shape_sliders.delta_to_blender(
            pose_bone, entry["scale"], entry["pos"], entry["quat"]
        )

        piece = {
            "scale": (
                scale_mul
                if "scale" in driven and any(abs(v - 1.0) > 1e-6 for v in scale_mul)
                else None
            ),
            "location": (
                loc_off if "location" in driven and loc_off.length > 1e-9 else None
            ),
            "rotation": (
                delta_local
                if "rotation_quaternion" in driven
                and abs(abs(delta_local.w) - 1.0) > 1e-9
                else None
            ),
        }
        if any(piece.values()):
            out[name] = piece

    return out


def _update():
    if view_layer := getattr(bpy.context, "view_layer", None):
        view_layer.update()


@contextlib.contextmanager
def pose_suspended(armature: bpy.types.Object | None):
    """Put the skeleton back the way the model import left it, for a while.

    Model export writes the skeleton as it stands, so a body shape or an
    animation frame sitting in the pose ends up in the .aqp and .aqn -
    silently, because the files come out the same size either way.

    "The way the model import left it" is not the rest pose: an imported
    model carries real pose transforms on its finger bones, and clearing
    the pose would drop them from the exported skeleton. The baseline the
    character import stored (import_fnp.store_model_pose) is what gets
    restored here instead.

    Does nothing at all when there is no baseline, so exporting a model
    that never had a character file applied is untouched.

    A linked action is unlinked as well, since it would be applied again
    on the next depsgraph evaluation. Everything goes back even if the
    block raises.
    """
    if (
        armature is None
        or armature.type != "ARMATURE"
        or not import_fnp.has_model_pose(armature)
    ):
        yield
        return

    animation_data = armature.animation_data
    action = animation_data.action if animation_data else None
    slot = getattr(animation_data, "action_slot", None) if action else None

    saved = [
        (
            bone,
            bone.location.copy(),
            bone.rotation_quaternion.copy(),
            bone.rotation_mode,
            bone.scale.copy(),
        )
        for bone in armature.pose.bones
    ]

    def update():
        if view_layer := getattr(bpy.context, "view_layer", None):
            view_layer.update()

    try:
        if action is not None:
            animation_data.action = None

        for bone in armature.pose.bones:
            baseline = bone.get(import_fnp.MODEL_POSE_PROP)
            if baseline is None or len(baseline) != 10:
                continue

            bone.rotation_mode = "QUATERNION"
            bone.location = baseline[0:3]
            bone.rotation_quaternion = baseline[3:7]
            bone.scale = baseline[7:10]

        update()
        yield
    finally:
        for bone, location, quaternion, rotation_mode, scale in saved:
            bone.rotation_mode = "QUATERNION"
            bone.location = location
            bone.rotation_quaternion = quaternion
            bone.scale = scale
            bone.rotation_mode = rotation_mode

        if action is not None:
            animation_data.action = action
            # Reassigning an action drops the slot it was played through.
            if slot is not None:
                with contextlib.suppress(AttributeError, TypeError):
                    animation_data.action_slot = slot

        update()


def shape_state(armature: bpy.types.Object | None) -> str:
    """Which layer the body shape lives in, for the UI to report.

    An action owning the pose means the shape is either already baked or
    was never there, so only an actionless modified pose is a warning.
    """
    if armature is None:
        return STATE_REST

    if (
        armature.animation_data is not None
        and armature.animation_data.action is not None
    ):
        return STATE_ANIMATED

    if pose_is_modified(armature):
        return STATE_SHAPE_IN_POSE

    return STATE_REST


def draw_shape_state(layout, state: str, *, hint: str = "location") -> None:
    """Warn, in place, that the shape and animation want the same layer.

    Drawn in the Shape Adjust panel and in the motion import/export
    dialogs, which are the points where the collision actually bites.
    The dialogs cannot host the operator, so they point at it instead;
    the panel, which already shows the button, passes hint="none".
    """
    if state != STATE_SHAPE_IN_POSE:
        return

    box = layout.box()
    column = box.column(align=True)
    column.label(text="Body shape is in the pose layer.", icon="ERROR")
    column.label(text="Animation uses that same layer and")
    column.label(text="will overwrite the shape.")

    if hint == "location":
        column.separator()
        column.label(text="Apply Shape to Rest Pose first:", icon="ARMATURE_DATA")
        column.label(text="Scene > PSO2 Appearance > Shape Adjust")


@classes.register
class PSO2_OT_BakeShapeToRest(bpy.types.Operator):
    """Make the current body shape part of the skeleton, freeing the pose
    for animation. Do this after the shape is final - to change it again,
    re-import the character file"""

    bl_label = "Apply Shape to Rest Pose"
    bl_idname = "pso2.bake_shape_to_rest"
    bl_options = {"UNDO"}

    def invoke(self, context, event):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
        return context.window_manager.invoke_props_dialog(
            self, width=400, confirm_text="Apply"
        )

    def draw(self, context):
        assert self.layout is not None

        column = self.layout.column(align=True)
        column.label(text="Makes the current body shape part of the skeleton,")
        column.label(text="so animations no longer overwrite it.")
        column.separator()
        column.label(text="Mesh data is changed permanently.", icon="ERROR")
        column.label(text="The sliders can no longer edit the shape afterwards")
        column.label(text="- re-import the character file to change it.")

    def execute(self, context) -> OperatorResult:
        armature = import_aqm._find_target_armature(context)
        if armature is None:
            self.report(
                {"ERROR"},
                "No target armature. Select a PSO2 armature (or a model parented"
                " to one) and try again.",
            )
            return {"CANCELLED"}

        if not pose_is_modified(armature):
            self.report({"WARNING"}, "The pose is already at rest; nothing to bake")
            return {"CANCELLED"}

        if armature.animation_data and armature.animation_data.action:
            self.report(
                {"ERROR"},
                "The armature has an animation action. Baking now would capture"
                " the current animation frame. Unlink the action first.",
            )
            return {"CANCELLED"}

        meshes = _deformed_meshes(armature)
        blocked = [m.name for m in meshes if m.data.shape_keys]  # type: ignore
        if blocked:
            self.report(
                {"ERROR"},
                f"{blocked[0]} has shape keys, which block freezing its"
                " deformation. Remove them and try again.",
            )
            return {"CANCELLED"}

        previous_active = context.view_layer.objects.active
        previous_mode = armature.mode
        selected = list(context.selected_objects)

        take_snapshot(armature, meshes)

        try:
            # Freeze each mesh in its deformed shape first. Rest bones cannot
            # hold scale, so applying the pose alone would snap every mesh
            # back to the unshaped body.
            for mesh in meshes:
                _freeze_deformation(context, mesh, armature)

            context.view_layer.objects.active = armature
            armature.select_set(True)
            bpy.ops.object.mode_set(mode="POSE")
            bpy.ops.pose.armature_apply(selected=False)
        except RuntimeError as ex:
            discard_snapshot(armature)
            self.report({"ERROR"}, f"Could not apply the pose: {ex}")
            return {"CANCELLED"}
        finally:
            if armature.mode != previous_mode:
                with contextlib.suppress(RuntimeError):
                    bpy.ops.object.mode_set(mode=previous_mode)
            for obj in selected:
                obj.select_set(True)
            if previous_active is not None:
                context.view_layer.objects.active = previous_active

        # The pose is empty now, so the sliders' stored baseline is gone
        # with it; they start again from the new rest pose. The model pose
        # goes too - it is part of the rest pose from here on, and putting
        # it back on top would double it.
        shape_sliders.clear_base(armature)
        import_fnp.clear_model_pose(armature)
        settings = shape_sliders.get_settings(context)
        if settings is not None:
            settings.reset()

        self.report(
            {"INFO"},
            "Body shape applied to the rest pose. Animations can now be"
            " imported without losing it. Revert Applied Shape puts it back.",
        )
        return {"FINISHED"}


@classes.register
class PSO2_OT_RevertShapeBake(bpy.types.Operator):
    """Put the skeleton and meshes back the way they were before the shape
    was applied to the rest pose, and return the shape to the pose"""

    bl_label = "Revert Applied Shape"
    bl_idname = "pso2.revert_shape_bake"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return has_snapshot(import_aqm._find_target_armature(context))

    def invoke(self, context, event):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
        return context.window_manager.invoke_props_dialog(
            self, width=400, confirm_text="Revert"
        )

    def draw(self, context):
        assert self.layout is not None

        column = self.layout.column(align=True)
        column.label(text="Restores the skeleton and the meshes as they were")
        column.label(text="before Apply Shape to Rest Pose, and puts the")
        column.label(text="shape back in the pose so the sliders can edit it.")
        column.separator()
        column.label(text="Anything sculpted since then is lost.", icon="ERROR")

    def execute(self, context) -> OperatorResult:
        armature = import_aqm._find_target_armature(context)
        if armature is None:
            self.report(
                {"ERROR"},
                "No target armature. Select a PSO2 armature (or a model parented"
                " to one) and try again.",
            )
            return {"CANCELLED"}

        if not has_snapshot(armature):
            self.report(
                {"ERROR"},
                "Nothing to revert: no stored state from Apply Shape to Rest"
                " Pose, or the copies it kept have been deleted.",
            )
            return {"CANCELLED"}

        if armature.animation_data and armature.animation_data.action:
            self.report(
                {"ERROR"},
                "The armature has an animation action, which drives the same"
                " pose the shape goes back into. Unlink the action first.",
            )
            return {"CANCELLED"}

        missing = restore_snapshot(armature)
        context.view_layer.update()

        settings = shape_sliders.get_settings(context)
        if settings is not None:
            settings.reset()

        if missing:
            self.report(
                {"WARNING"},
                f"Reverted, but {len(missing)} object(s) had no stored copy"
                f" left: {', '.join(missing[:3])}",
            )
            return {"FINISHED"}

        self.report(
            {"INFO"},
            "Reverted to the state before the shape was applied. The shape is"
            " back in the pose.",
        )
        return {"FINISHED"}
