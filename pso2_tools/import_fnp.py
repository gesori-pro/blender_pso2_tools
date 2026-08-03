"""
Import PSO2 NGS character customization files (.fnp) and apply the body
proportion sliders to a PSO2 armature as a single pose.

Blender-side port of the reverse-engineering reference implementation
(RE/handoff/python/apply_proportions_blender.py), verified bone-for-bone
against the running game (SPEC §1: scale 133/133, position 134/134,
rotation 132/134). The slider math lives in the proportions package;
this module owns everything Blender-specific:

  * the PSO2 -> Blender axis permutation for scale, position and rotation
    (SPEC §6-2, §6-3, §6-10 - PSO2's bone length axis is X, Blender's is Y)
  * inherit_scale set explicitly from the AQN flags (SPEC §6-4)
  * ground-contact normalization of body_root (SPEC §6-9)

Deliberately NOT done, both established by reading the game's runtime
bone array (SPEC §6-5, §6-9):

  * No driver-bone folding: the game keeps the value on the weightless
    bone itself and leaves the parent alone. Folding puts a ~3% error on
    the calves and pelvis.
  * No proportion value on body_root: its runtime scale is ground-contact
    normalization, solved here by bisection instead.
"""

from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector

from . import char_colors, charfile, classes, import_aqm, proportions, scene_props
from .debug import debug_print
from .util import OperatorResult

# PSO2 X is the bone length axis; Blender's is Y. Components 0 and 1 swap,
# Z flips. det(+1), a proper rotation (SPEC §6-2).
_PERM = Matrix(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))

# NOTE (SPEC §9-2, unverified): muscleMass is a shader/normal-map blend,
# not a bone deform. Whether the addon's MUSCULARITY property maps as
# value/60000 or /65535 has not been confirmed.
_MUSCLE_MASS_MAX = 60000.0


@classes.register
class PSO2_OT_ImportFnp(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ImportHelper
):
    """Load a PSO2 NGS character file and pose the active PSO2 armature
    with its body proportion sliders"""

    bl_label = "Import Character File"
    bl_idname = "pso2.import_fnp"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".fnp"
    # [m|f][h|n|c|d]p = (male/female) x (human/newman/cast/deuman),
    # trailing "u" = unencrypted body.
    filter_glob: bpy.props.StringProperty(
        default=(
            "*.fhp;*.fnp;*.fcp;*.fdp;*.mhp;*.mnp;*.mcp;*.mdp;"
            "*.fhpu;*.fnpu;*.fcpu;*.fdpu;*.mhpu;*.mnpu;*.mcpu;*.mdpu"
        ),
        options={"HIDDEN"},
    )

    ground_contact: bpy.props.BoolProperty(
        name="Ground Contact",
        description=(
            "Scale body_root so the lowest vertex sits on Z=0, matching the"
            " game's ground-contact normalization. Only moves the body; the"
            " character's size is unaffected"
        ),
        default=True,
    )
    outfit_adjust: bpy.props.BoolProperty(
        name="Outfit Shape Adjust",
        description=(
            "Apply the outfit's own shape-adjust motion (_sa.aqm frame 1)"
            " on top of the sliders, routed from the file's costumePart."
            " Most outfits don't have one, in which case this does nothing"
        ),
        default=True,
    )
    import_colors: bpy.props.BoolProperty(
        name="Import Colors",
        description=(
            "Set the scene's skin, hair, eye and outfit colors from the"
            " file. Materials pick these up when a model is imported, so"
            " re-import the model to see them on an existing one"
        ),
        default=True,
    )
    set_muscularity: bpy.props.BoolProperty(
        name="Set Muscularity",
        description=(
            "Set the scene muscularity (skin texture mix) from the file's"
            " muscleMass value. The scale factor is unverified (SPEC §9-2)"
        ),
        default=False,
    )

    def draw(self, context):
        assert self.layout is not None

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.prop(self, "ground_contact")
        layout.prop(self, "outfit_adjust")
        layout.prop(self, "import_colors")
        layout.prop(self, "set_muscularity")

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
            char = charfile.CharacterFile.load(path)
        except (OSError, ValueError, KeyError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        try:
            result = proportions.compute(char, apply_outfit_adjust=self.outfit_adjust)
        except (OSError, KeyError, ValueError) as ex:
            self.report({"ERROR"}, f"Could not compute proportions: {ex}")
            return {"CANCELLED"}

        summary = apply_proportions(armature, result["bones"])

        if self.ground_contact:
            summary["ground"] = solve_ground_contact(armature)

        # The pose was rebuilt, so the shape sliders' stored baseline is
        # stale and their values belong to whatever character was loaded
        # before. Clear both, or the next slider touch would apply the old
        # character's shape all at once.
        # Local import: shape_sliders imports this module at load time.
        from . import shape_sliders

        shape_sliders.clear_base(armature)
        if (sliders := shape_sliders.get_settings(context)) is not None:
            sliders.reset()

        colors_applied = 0
        if self.import_colors:
            colors_applied = char_colors.apply_to_scene(context, char)

        if self.set_muscularity:
            muscle_mass = float(char["baseDOC.muscleMass"])
            value = max(0.0, min(1.0, muscle_mass / _MUSCLE_MASS_MAX))
            try:
                setattr(context.scene, scene_props.MUSCULARITY, value)
            except (AttributeError, TypeError):
                debug_print("Could not set scene muscularity")

        message = (
            f"Applied {len(result['sliders'])} sliders to {summary['applied']} bones"
        )
        if result["outfit_adjust_bones"] and self.outfit_adjust:
            message += f", outfit adjust on {result['outfit_adjust_bones']} bones"
        if colors_applied:
            message += f", {colors_applied} colors"
        if summary.get("ground", {}).get("solved"):
            message += f", grounded at body_root {summary['ground']['scale']}"
        if summary["missing"]:
            message += f", {len(summary['missing'])} bones not in armature"

        self.report({"INFO"}, message)
        return {"FINISHED"}

    def invoke(self, context, event):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
        return self.invoke_popup(context)


def set_inherit_scale(armature: bpy.types.Object) -> dict:
    """Set inherit_scale from the AQN flags carried in the bone names.

    PSO2 standard skeleton bones (boneShort1 & 0x1C0 != 0) do not inherit
    parent scale; costume/decoration bones do (SPEC §6-4). Imports leave
    inconsistent values behind - a scene where every bone is FULL
    compounds scale down the chain and blows l_calf_alt up from 1.24 to
    3.49 - so never trust the scene, set it explicitly.
    """
    counts = {"NONE": 0, "FULL": 0}

    for bone in armature.data.bones:  # type: ignore
        parts = bone.name.split("#")
        try:
            flags = int(parts[1], 16) if len(parts) > 2 else 0
        except ValueError:
            flags = 0

        mode = "NONE" if flags & 0x1C0 else "FULL"
        bone.inherit_scale = mode
        counts[mode] += 1

    return counts


def reset_pose(armature: bpy.types.Object) -> None:
    for pose_bone in armature.pose.bones:
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)


# Pose-bone custom property holding the pose the model import produced,
# captured before proportions overwrite it: [loc xyz, quat wxyz, scale xyz].
MODEL_POSE_PROP = "pso2_model_pose"


def store_model_pose(armature: bpy.types.Object) -> int:
    """Remember the pose as the model import left it.

    An imported model is not sitting at its rest pose - the finger bones
    carry real transforms - so "put the skeleton back" cannot mean "clear
    the pose", and model export needs somewhere to put it back to. Written
    once and then left alone, so importing a second character file does not
    overwrite the baseline with an already shaped pose.
    """
    stored = 0

    for pose_bone in armature.pose.bones:
        if MODEL_POSE_PROP in pose_bone:
            continue

        pose_bone.rotation_mode = "QUATERNION"
        pose_bone[MODEL_POSE_PROP] = [
            *pose_bone.location,
            *pose_bone.rotation_quaternion,
            *pose_bone.scale,
        ]
        stored += 1

    return stored


def clear_model_pose(armature: bpy.types.Object) -> None:
    """Forget the stored model pose (call once it is part of the rest pose)."""
    for pose_bone in armature.pose.bones:
        if MODEL_POSE_PROP in pose_bone:
            del pose_bone[MODEL_POSE_PROP]


def has_model_pose(armature: bpy.types.Object) -> bool:
    return any(MODEL_POSE_PROP in pose_bone for pose_bone in armature.pose.bones)


def apply_proportions(armature: bpy.types.Object, bones: dict) -> dict:
    """Write computed proportion deltas (PSO2 axis order) into the pose."""
    store_model_pose(armature)
    reset_pose(armature)
    inherit = set_inherit_scale(armature)

    bones_by_id, bones_by_name = import_aqm._get_bone_maps(armature)

    applied = 0
    missing: list[str] = []

    for name, delta in bones.items():
        index = delta.get("index")

        bone_name = bones_by_id.get(index) if index is not None else None
        if bone_name is None:
            bone_name = bones_by_name.get(name.lower())
        if bone_name is None:
            missing.append(name)
            continue

        pose_bone = armature.pose.bones[bone_name]
        bone = pose_bone.bone

        # SPEC §6-2: PSO2 X (length axis) == Blender Y, swap components 0/1.
        sx, sy, sz = delta["scale"]
        pose_bone.scale = (sy, sx, sz)

        # SPEC §6-3: same permutation plus a Z sign flip; posDelta is in
        # the parent-local frame, pose location is bone-local, so run it
        # through the rest rotation as well.
        if bone.parent is not None:
            rest = bone.parent.matrix_local.inverted() @ bone.matrix_local
        else:
            rest = bone.matrix_local.copy()
        rest_rotation = rest.to_quaternion()

        d = delta["pos"]
        pose_bone.location = rest_rotation.inverted() @ Vector((d[1], d[0], -d[2]))

        # SPEC §6-10: the game composes delta * rest in the parent-local
        # frame, so the bone-local pose is rest^-1 * delta * rest, with the
        # delta moved into Blender's basis by the same permutation.
        q = delta.get("rotQuat")
        if q is not None and any(abs(component) > 1e-9 for component in q[:3]):
            m_pso2 = Quaternion((q[3], q[0], q[1], q[2])).to_matrix()
            m_blender = _PERM @ m_pso2 @ _PERM.transposed()
            pose_bone.rotation_quaternion = (
                rest_rotation.inverted().to_matrix()
                @ m_blender
                @ rest_rotation.to_matrix()
            ).to_quaternion()

        applied += 1

    bpy.context.view_layer.update()

    return {"applied": applied, "missing": missing, "inherit_scale": inherit}


def _lowest_vertex_z(armature: bpy.types.Object) -> float | None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    lowest = None

    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.parent is not armature:
            continue

        evaluated = obj.evaluated_get(depsgraph)
        matrix = obj.matrix_world
        for vertex in evaluated.data.vertices:  # type: ignore
            z = (matrix @ vertex.co).z
            if lowest is None or z < lowest:
                lowest = z

    return lowest


def solve_ground_contact(armature: bpy.types.Object, low=0.9, high=1.6) -> dict:
    """Scale body_root until the lowest vertex sits on Z = 0 (SPEC §6-9).

    body_root's runtime scale is not a proportion value: the game uses it
    to stand the character on the ground, so it depends on the footwear
    geometry. With inherit_scale='NONE' it only translates the body; the
    character's size is unaffected.
    """
    base_names = {
        b.name.split("#")[0]: b.name
        for b in armature.data.bones  # type: ignore
    }
    bone_name = base_names.get("body_root")
    if bone_name is None:
        return {"solved": False}

    pose_bone = armature.pose.bones[bone_name]
    keep = tuple(pose_bone.scale)

    def floor_at(scale: float) -> float | None:
        pose_bone.scale = (scale, scale, scale)
        bpy.context.view_layer.update()
        return _lowest_vertex_z(armature)

    floor_low = floor_at(low)
    floor_high = floor_at(high)

    if floor_low is None or floor_high is None or (floor_low < 0) == (floor_high < 0):
        # No meshes, or no sign change: leave the proportion value alone.
        pose_bone.scale = keep
        bpy.context.view_layer.update()
        return {"solved": False, "scale": list(keep)}

    for _ in range(60):
        mid = (low + high) / 2.0
        if ((floor_at(mid) or 0.0) < 0) == (floor_low < 0):
            low = mid
        else:
            high = mid

    scale = (low + high) / 2.0
    pose_bone.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    return {
        "solved": True,
        "scale": round(scale, 6),
        "floor_z": round(_lowest_vertex_z(armature) or 0.0, 6),
    }
