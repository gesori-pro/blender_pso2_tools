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


@classes.register
class PSO2_OT_BakeShapeToRest(bpy.types.Operator):
    """Make the current body shape part of the skeleton, freeing the pose
    for animation. Do this after the shape is final - to change it again,
    re-import the character file"""

    bl_label = "Apply Shape to Rest Pose"
    bl_idname = "pso2.bake_shape_to_rest"
    bl_options = {"UNDO"}

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

        previous_active = context.view_layer.objects.active
        previous_mode = armature.mode
        selected = list(context.selected_objects)

        try:
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
