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

import bpy

from . import classes, import_aqm, shape_sliders
from .util import OperatorResult

# Where the body shape currently lives.
STATE_REST = "REST"  # nothing in the pose - ready for animation
STATE_SHAPE_IN_POSE = "SHAPE_IN_POSE"  # shape in the pose, not baked yet
STATE_ANIMATED = "ANIMATED"  # an action owns the pose

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
    try:
        bpy.ops.object.modifier_move_to_index(
            modifier=copied.name, index=list(mesh.modifiers).index(source)
        )
    except (RuntimeError, ValueError):
        pass

    bpy.ops.object.modifier_apply(modifier=copied.name)


def pose_is_modified(armature: bpy.types.Object, epsilon: float = 1e-6) -> bool:
    """Is any pose bone transformed away from its rest position?"""
    for pose_bone in armature.pose.bones:
        if any(abs(v) > epsilon for v in pose_bone.location):
            return True
        if any(abs(v - 1.0) > epsilon for v in pose_bone.scale):
            return True
        quaternion = pose_bone.rotation_quaternion
        if abs(abs(quaternion.w) - 1.0) > epsilon:
            return True

    return False


def shape_state(armature: bpy.types.Object | None) -> str:
    """Which layer the body shape lives in, for the UI to report.

    An action owning the pose means the shape is either already baked or
    was never there, so only an actionless modified pose is a warning.
    """
    if armature is None:
        return STATE_REST

    if armature.animation_data is not None and armature.animation_data.action is not None:
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
            self.report({"ERROR"}, f"Could not apply the pose: {ex}")
            return {"CANCELLED"}
        finally:
            if armature.mode != previous_mode:
                try:
                    bpy.ops.object.mode_set(mode=previous_mode)
                except RuntimeError:
                    pass
            for obj in selected:
                obj.select_set(True)
            if previous_active is not None:
                context.view_layer.objects.active = previous_active

        # The pose is empty now, so the sliders' stored baseline is gone
        # with it; they start again from the new rest pose.
        shape_sliders.clear_base(armature)
        settings = shape_sliders.get_settings(context)
        if settings is not None:
            settings.reset()

        self.report(
            {"INFO"},
            "Body shape applied to the rest pose. Animations can now be"
            " imported without losing it.",
        )
        return {"FINISHED"}
