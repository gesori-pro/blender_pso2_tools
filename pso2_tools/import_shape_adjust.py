"""
Import a PSO2 shape-adjust motion (pl_rbd_*_sa.aqm) onto a PSO2 armature.

Outfits ship these to correct the body shape for a specific costume, and
mod managers use exactly the same format for custom body shapes (breast
size adjustments and the like). The file is a two-frame motion: frame 0
is neutral and frame 1 carries scale/position/rotation deltas that apply
on top of whatever pose the character already has - the same convention
as the proportion motion's frame-1 base correction (SPEC §6-1-1, §6-11).
"""

from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Quaternion, Vector

from . import aqm, classes, import_aqm, import_fnp
from .util import OperatorResult


@classes.register
class PSO2_OT_ImportShapeAdjust(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ImportHelper
):
    """Load a PSO2 shape-adjust motion (_sa.aqm) and compose its frame-1
    deltas onto the active PSO2 armature's pose"""

    bl_label = "Import Shape Adjust"
    bl_idname = "pso2.import_shape_adjust"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".aqm"
    filter_glob: bpy.props.StringProperty(default="*.aqm", options={"HIDDEN"})

    compose: bpy.props.BoolProperty(
        name="Compose With Current Pose",
        description=(
            "Apply on top of the current pose (how the game layers it over"
            " the character's proportions). Disable to reset the pose first"
            " and see the adjustment alone"
        ),
        default=True,
    )
    ground_contact: bpy.props.BoolProperty(
        name="Ground Contact",
        description="Re-solve the body_root ground contact after applying",
        default=True,
    )

    def draw(self, context):
        assert self.layout is not None

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "compose")
        layout.prop(self, "ground_contact")

    def execute(self, context) -> OperatorResult:
        armature = import_aqm._find_target_armature(context)
        if armature is None:
            self.report(
                {"ERROR"},
                "No target armature. Select a PSO2 armature (or a model parented"
                " to one) and try again.",
            )
            return {"CANCELLED"}

        path = Path(self.filepath)  # type: ignore

        try:
            motion = aqm.read_aqm(path)
        except (OSError, aqm.AqmError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        deltas = extract_frame1_deltas(motion)
        if not deltas:
            self.report(
                {"ERROR"},
                f"{path.name}: no frame-1 adjustment data. This does not look"
                " like a shape-adjust motion.",
            )
            return {"CANCELLED"}

        if not self.compose:
            import_fnp.reset_pose(armature)

        summary = apply_shape_adjust(armature, deltas)

        if self.ground_contact:
            summary["ground"] = import_fnp.solve_ground_contact(armature)

        # The pose changed underneath the shape sliders; their stored base
        # is stale. Local import: shape_sliders imports this module.
        from . import shape_sliders

        shape_sliders.clear_base(armature)

        message = f"Adjusted {summary['applied']} bones from {path.name}"
        if motion.end_frame != 1:
            message += (
                f" (warning: motion has {motion.end_frame + 1} frames, only"
                " frame 1 was used)"
            )
        if summary.get("ground", {}).get("solved"):
            message += f", grounded at body_root {summary['ground']['scale']}"
        if summary["missing"]:
            message += f", {len(summary['missing'])} bones not in armature"

        self.report({"INFO"}, message)
        return {"FINISHED"}

    def invoke(self, context, event):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
        return self.invoke_popup(context)


def extract_frame1_deltas(motion: aqm.AqmMotion) -> dict:
    """Per-node adjustment deltas, in PSO2 axis order.

    Same convention as the weight tables' _baseCorrection: scale is a
    multiplier (frame1 / frame0), position is a difference, rotation is
    the left delta (frame1 * frame0^-1).

    A channel with a single key is a static value, not a change - and what
    that means differs by channel. The rest pose's scale is 1.0, so a
    static scale IS the multiplier; real files rely on this, storing
    pelvis 1.3/1.15/1.0 as one key. Static position and rotation channels
    hold the rest pose itself (hip at 0.898 and so on), so for those,
    nothing to compare against means no adjustment.
    """
    deltas: dict[int, dict] = {}

    for index, node in enumerate(motion.nodes):
        entry = {
            "name": node.name,
            "scale": None,
            "pos": None,
            "rotQuat": None,
        }

        for key_set in node.key_sets:
            if not key_set.vec4_keys:
                continue

            keys = dict(zip(key_set.frames(), key_set.vec4_keys))

            if key_set.key_type == aqm.KEY_TYPE_SCALE:
                if 0 in keys and 1 in keys:
                    f0, f1 = keys[0], keys[1]
                    mul = [
                        a / b if abs(b) > 1e-9 else 1.0 for a, b in zip(f1[:3], f0[:3])
                    ]
                elif len(keys) == 1:
                    mul = list(next(iter(keys.values()))[:3])
                    # A few nodes carry an all-zero scale (l_legadd in
                    # real files); the game does not collapse the bone,
                    # so treat it as neutral rather than degenerate.
                    if all(abs(m) < 1e-9 for m in mul):
                        continue
                else:
                    continue

                if any(abs(m - 1.0) > 1e-6 for m in mul):
                    entry["scale"] = mul
                continue

            if 0 not in keys or 1 not in keys:
                continue

            f0, f1 = keys[0], keys[1]

            if key_set.key_type == aqm.KEY_TYPE_POSITION:
                diff = [a - b for a, b in zip(f1[:3], f0[:3])]
                if any(abs(d) > 1e-9 for d in diff):
                    entry["pos"] = diff
            elif key_set.key_type == aqm.KEY_TYPE_ROTATION:
                q0 = Quaternion((f0[3], f0[0], f0[1], f0[2])).normalized()
                q1 = Quaternion((f1[3], f1[0], f1[1], f1[2])).normalized()
                delta = q1 @ q0.inverted()
                if delta.angle > 1e-6:
                    # Store xyzw, PSO2 space, left-delta (SPEC §6-10).
                    entry["rotQuat"] = [delta.x, delta.y, delta.z, delta.w]

        if entry["scale"] or entry["pos"] or entry["rotQuat"]:
            deltas[index] = entry

    return deltas


def apply_shape_adjust(armature: bpy.types.Object, deltas: dict) -> dict:
    """Compose frame-1 deltas onto the current pose.

    Uses the same axis permutation as the proportion importer (SPEC §6-2,
    §6-3, §6-10); scale multiplies, position adds in the bone-local frame,
    rotation left-multiplies the pose quaternion.
    """
    import_fnp.set_inherit_scale(armature)

    bones_by_id, bones_by_name = import_aqm._get_bone_maps(armature)

    applied = 0
    missing: list[str] = []

    for index, entry in deltas.items():
        bone_name = bones_by_id.get(index)
        if bone_name is None:
            bone_name = bones_by_name.get(entry["name"].lower())
        if bone_name is None:
            missing.append(entry["name"] or f"node {index}")
            continue

        pose_bone = armature.pose.bones[bone_name]
        bone = pose_bone.bone

        if bone.parent is not None:
            rest = bone.parent.matrix_local.inverted() @ bone.matrix_local
        else:
            rest = bone.matrix_local.copy()
        rest_rotation = rest.to_quaternion()

        if entry["scale"] is not None:
            mx, my, mz = entry["scale"]
            # SPEC §6-2 permutation, then component-wise multiply.
            pose_bone.scale = (
                pose_bone.scale[0] * my,
                pose_bone.scale[1] * mx,
                pose_bone.scale[2] * mz,
            )

        if entry["pos"] is not None:
            d = entry["pos"]
            pose_bone.location = pose_bone.location + (
                rest_rotation.inverted() @ Vector((d[1], d[0], -d[2]))
            )

        if entry["rotQuat"] is not None:
            q = entry["rotQuat"]
            m_pso2 = Quaternion((q[3], q[0], q[1], q[2])).to_matrix()
            m_blender = import_fnp._PERM @ m_pso2 @ import_fnp._PERM.transposed()
            delta_local = (
                rest_rotation.inverted().to_matrix()
                @ m_blender
                @ rest_rotation.to_matrix()
            ).to_quaternion()

            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = delta_local @ pose_bone.rotation_quaternion

        applied += 1

    bpy.context.view_layer.update()

    return {"applied": applied, "missing": missing}
