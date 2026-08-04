"""
Interactive body shape sliders, editing the same values a shape-adjust
motion (pl_rbd_*_sa.aqm) carries.

The sliders hold PSO2-space values exactly as they would be written to an
AQM file: scale multipliers around 1.0, position deltas in meters in the
parent-local frame, rotations in degrees (stored as quaternions in the
file). Moving a slider re-poses the bones live through the same verified
math path as the shape-adjust importer (SPEC §6-2, §6-3, §6-10), and the
result can be exported as a mod-manager-compatible _sa.aqm.

Idempotency: every application recomputes pose = base ∘ sliders, where
the base pose is snapshotted into a custom property on each managed pose
bone the first time a slider moves. Re-importing a character clears the
snapshots (the pose was rebuilt), and "Freeze Current" re-captures.

Mirroring follows the convention observed in the game's own _sa files and
in hand-made ones: the right side negates the position's PSO2 Y component
and the quaternion's X and Z components; scale is copied as is.
"""

import math
from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Quaternion, Vector

from . import aqm, classes, import_aqm, import_shape_adjust
from .util import OperatorResult

# Pose-bone custom property holding the pre-slider pose:
# [loc xyz, quat wxyz, scale xyz].
BASE_PROP = "pso2_shape_base"

# Scene property name for the slider group.
SHAPE_SLIDERS = "pso2_shape_sliders"


class Group:
    """One slider group: a property prefix and the bones it drives.

    `node_ids` are the PSO2 standard-skeleton indices, used when exporting
    against an armature whose bones carry no pso2_bone_id.
    """

    def __init__(self, key, label, left, right, node_ids, rotate=True):
        self.key = key
        self.label = label
        self.left = left
        self.right = right
        self.node_ids = node_ids
        self.rotate = rotate

    @property
    def bones(self):
        return [b for b in (self.left, self.right) if b is not None]


# Order here is the panel order. c_breast is intentionally absent: it
# barely deforms anything in practice, so it only added clutter.
GROUPS = [
    Group("breast", "Breast", "l_breast", "r_breast", {"l_breast": 41, "r_breast": 43}),
    Group(
        "breast2",
        "Breast Scale",
        "l_breast_scale",
        "r_breast_scale",
        {"l_breast_scale": 124, "r_breast_scale": 125},
    ),
    Group(
        "cbreast2",
        "Center Breast Scale",
        "c_breast_scale",
        None,
        {"c_breast_scale": 130},
    ),
    Group(
        "clav",
        "Clavicle",
        "l_clavicle",
        "r_clavicle",
        {"l_clavicle": 22, "r_clavicle": 30},
        rotate=False,
    ),
    Group(
        "thigh",
        "Thigh",
        "l_thigh_alt",
        "r_thigh_alt",
        {"l_thigh_alt": 52, "r_thigh_alt": 63},
    ),
    Group(
        "thightw",
        "Thigh Twist",
        "l_thigh_tw_alt",
        "r_thigh_tw_alt",
        {"l_thigh_tw_alt": 53, "r_thigh_tw_alt": 64},
    ),
    Group(
        "hiptw", "Hip Twist", "l_hip_tw", "r_hip_tw", {"l_hip_tw": 50, "r_hip_tw": 51}
    ),
    Group("pelvis", "Pelvis", "pelvis", None, {"pelvis": 3}),
]

GROUPS_BY_KEY = {g.key: g for g in GROUPS}

# Bones a loaded file may carry that no slider covers. Warned about on
# load so a round trip never silently drops data.
UNCOVERED_BONES = {"c_breast"}

IDENTITY_SCALE = (1.0, 1.0, 1.0)
IDENTITY_VEC = (0.0, 0.0, 0.0)


def euler_deg_to_quat(deg) -> tuple[float, float, float, float]:
    """PSO2-space euler degrees -> xyzw quaternion, ZYX composition.

    The game's own euler convention (SPEC §6-10): q = Qz * Qy * Qx, i.e.
    the X rotation applied first. Wrong order costs several degrees on
    the shoulders and bust, so this is centralized here.
    """
    q = (
        Quaternion((0.0, 0.0, 1.0), math.radians(deg[2]))
        @ Quaternion((0.0, 1.0, 0.0), math.radians(deg[1]))
        @ Quaternion((1.0, 0.0, 0.0), math.radians(deg[0]))
    )
    return (q.x, q.y, q.z, q.w)


def quat_to_euler_deg(q_xyzw) -> tuple[float, float, float]:
    """Inverse of euler_deg_to_quat. Blender's 'XYZ' euler mode composes
    the same matrix (X applied first), verified by round trip."""
    q = Quaternion((q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])).normalized()
    e = q.to_euler("XYZ")
    return (math.degrees(e.x), math.degrees(e.y), math.degrees(e.z))


def mirror_pos(pos):
    """Left -> right position across the sagittal plane (PSO2 Y negates)."""
    return (pos[0], -pos[1], pos[2])


def mirror_quat(q):
    """Left -> right rotation: reflect across the plane with normal Y."""
    return (-q[0], q[1], -q[2], q[3])


def _on_update(self, context):
    apply_sliders(context)


def _vector_prop(name, default, precision=5, **kwargs):
    return bpy.props.FloatVectorProperty(
        name=name,
        size=3,
        default=default,
        precision=precision,
        update=_on_update,
        **kwargs,
    )


def _scale_prop():
    return _vector_prop("Scale", IDENTITY_SCALE, soft_min=0.25, soft_max=4.0)


def _pos_prop():
    return _vector_prop(
        "Position", IDENTITY_VEC, soft_min=-0.2, soft_max=0.2, subtype="TRANSLATION"
    )


def _rot_prop():
    return _vector_prop(
        "Rotation", IDENTITY_VEC, precision=3, soft_min=-45.0, soft_max=45.0
    )


def _slider_annotations():
    """Build the PropertyGroup fields from the GROUPS table."""
    fields = {}
    for group in GROUPS:
        fields[f"{group.key}_scale"] = _scale_prop()
        fields[f"{group.key}_pos"] = _pos_prop()
        if group.rotate:
            fields[f"{group.key}_rot"] = _rot_prop()
    return fields


@classes.register
class Pso2ShapeSliders(bpy.types.PropertyGroup):
    """PSO2-space shape-adjust values, exactly as written to a _sa.aqm."""

    __annotations__ = _slider_annotations()

    def group_values(self, key: str) -> dict:
        return {
            "scale": tuple(getattr(self, f"{key}_scale")),
            "pos": tuple(getattr(self, f"{key}_pos")),
            "rot": tuple(getattr(self, f"{key}_rot", IDENTITY_VEC)),
        }

    def set_group_values(self, key: str, scale=None, pos=None, rot=None):
        if scale is not None:
            setattr(self, f"{key}_scale", scale)
        if pos is not None:
            setattr(self, f"{key}_pos", pos)
        if rot is not None and hasattr(self, f"{key}_rot"):
            setattr(self, f"{key}_rot", rot)

    def is_neutral(self, key: str) -> bool:
        values = self.group_values(key)
        return (
            all(abs(s - 1.0) < 1e-9 for s in values["scale"])
            and all(abs(p) < 1e-9 for p in values["pos"])
            and all(abs(r) < 1e-9 for r in values["rot"])
        )

    def reset(self):
        for prop in self.bl_rna.properties.keys():
            if prop not in {"rna_type", "name"}:
                self.property_unset(prop)


def add_scene_property():
    setattr(
        bpy.types.Scene,
        SHAPE_SLIDERS,
        bpy.props.PointerProperty(type=Pso2ShapeSliders),
    )


def get_settings(context) -> Pso2ShapeSliders | None:
    return getattr(context.scene, SHAPE_SLIDERS, None)


# ---------------------------------------------------------------------------
# Applying sliders to the pose


def _ensure_base(pose_bone):
    stored = pose_bone.get(BASE_PROP)
    if stored is None or len(stored) != 10:
        pose_bone.rotation_mode = "QUATERNION"
        stored = [
            *pose_bone.location,
            *pose_bone.rotation_quaternion,
            *pose_bone.scale,
        ]
        pose_bone[BASE_PROP] = stored

    return Vector(stored[0:3]), Quaternion(stored[3:7]), Vector(stored[7:10])


def clear_base(armature: bpy.types.Object):
    """Forget the stored base pose (call after the pose is rebuilt)."""
    for pose_bone in armature.pose.bones:
        if BASE_PROP in pose_bone:
            del pose_bone[BASE_PROP]


def _apply_to_bone(pose_bone, scale, pos, quat):
    """pose = base ∘ delta, through the verified axis conversion."""
    base_loc, base_rot, base_scale = _ensure_base(pose_bone)
    bone = pose_bone.bone

    if bone.parent is not None:
        rest = bone.parent.matrix_local.inverted() @ bone.matrix_local
    else:
        rest = bone.matrix_local.copy()
    rest_rotation = rest.to_quaternion()

    # SPEC §6-2: swap components 0/1 into Blender order, multiply.
    pose_bone.scale = (
        base_scale[0] * scale[1],
        base_scale[1] * scale[0],
        base_scale[2] * scale[2],
    )

    # SPEC §6-3: permutation + Z flip, rest rotation into bone-local space.
    pose_bone.location = base_loc + (
        rest_rotation.inverted() @ Vector((pos[1], pos[0], -pos[2]))
    )

    # The game applies shape-adjust rotations in the bone's own frame
    # (right delta, verified against its composed bone array), where the
    # PSO2 -> Blender axis change is the fixed (x,y,z) -> (y,x,-z) swap.
    # The preview does the same, so what is on screen is what the
    # exported file will do in game.
    delta_local = Quaternion((quat[3], quat[1], quat[0], -quat[2]))

    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = base_rot @ delta_local


def _sides(group, values):
    """(bone name, position, quaternion) for each side of a group."""
    quat = euler_deg_to_quat(values["rot"])
    out = [(group.left, values["pos"], quat)]
    if group.right is not None:
        out.append((group.right, mirror_pos(values["pos"]), mirror_quat(quat)))
    return out


def apply_sliders(context) -> dict:
    settings = get_settings(context)
    armature = import_aqm._find_target_armature(context)
    if settings is None or armature is None:
        return {"applied": 0}

    _, bones_by_name = import_aqm._get_bone_maps(armature)

    targets = [
        name
        for group in GROUPS
        for bone in group.bones
        if (name := bones_by_name.get(bone)) is not None
    ]

    # Blender throws away the pose location of a bone that is connected to
    # its parent, and the model import leaves a few of these connected -
    # asymmetrically, so l_thigh_tw_alt arrives connected while its mirror
    # does not, and the Thigh Twist sliders used to move only one leg.
    disconnected = import_aqm.disconnect_bones(context, armature, targets)

    applied = 0
    for group in GROUPS:
        values = settings.group_values(group.key)
        for bone_name, pos, quat in _sides(group, values):
            name = bones_by_name.get(bone_name)
            if name is None:
                continue
            _apply_to_bone(armature.pose.bones[name], values["scale"], pos, quat)
            applied += 1

    bpy.context.view_layer.update()
    return {"applied": applied, "disconnected": disconnected}


# ---------------------------------------------------------------------------
# Operators


@classes.register
class PSO2_OT_ShapeSlidersReset(bpy.types.Operator):
    """Reset all shape sliders to neutral, restoring the base pose"""

    bl_label = "Reset Sliders"
    bl_idname = "pso2.shape_sliders_reset"
    bl_options = {"UNDO"}

    def execute(self, context) -> OperatorResult:
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}

        settings.reset()
        apply_sliders(context)
        return {"FINISHED"}


@classes.register
class PSO2_OT_ShapeSlidersFreeze(bpy.types.Operator):
    """Treat the current pose as the new starting point: the sliders reset
    to neutral and further edits build on top of what you see now"""

    bl_label = "Freeze Current"
    bl_idname = "pso2.shape_sliders_freeze"
    bl_options = {"UNDO"}

    def execute(self, context) -> OperatorResult:
        settings = get_settings(context)
        armature = import_aqm._find_target_armature(context)
        if settings is None or armature is None:
            self.report({"ERROR"}, "No target armature")
            return {"CANCELLED"}

        clear_base(armature)
        settings.reset()

        self.report({"INFO"}, "Current pose is now the slider baseline")
        return {"FINISHED"}


@classes.register
class PSO2_OT_ShapeSlidersFromAqm(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ImportHelper
):
    """Load a shape-adjust motion (_sa.aqm) into the sliders for editing"""

    bl_label = "Load AQM Into Sliders"
    bl_idname = "pso2.shape_sliders_from_aqm"
    bl_options = {"UNDO"}

    filename_ext = ".aqm"
    filter_glob: bpy.props.StringProperty(default="*.aqm", options={"HIDDEN"})

    def execute(self, context) -> OperatorResult:
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}

        path = Path(self.filepath)  # type: ignore
        try:
            motion = aqm.read_aqm(path)
        except (OSError, aqm.AqmError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        deltas = import_shape_adjust.extract_frame1_deltas(motion)
        by_name = {
            entry["name"].lower(): entry for entry in deltas.values() if entry["name"]
        }

        settings.reset()
        loaded = 0
        warnings: list[str] = []

        for group in GROUPS:
            entry = by_name.get(group.left)
            if entry is None:
                continue

            scale = entry["scale"] or IDENTITY_SCALE
            pos = entry["pos"] or IDENTITY_VEC
            rot = (
                quat_to_euler_deg(entry["rotQuat"])
                if entry["rotQuat"]
                else IDENTITY_VEC
            )
            settings.set_group_values(group.key, scale=scale, pos=pos, rot=rot)
            loaded += 1

            # The sliders are symmetric; note when the file is not.
            other = by_name.get(group.right) if group.right else None
            if other is not None:
                mirrored = mirror_pos(pos)
                if any(
                    abs(a - b) > 1e-4
                    for a, b in zip(other["scale"] or IDENTITY_SCALE, scale)
                ) or any(
                    abs(a - b) > 1e-4
                    for a, b in zip(other["pos"] or IDENTITY_VEC, mirrored)
                ):
                    warnings.append(f"{group.label} L/R differ, left side used")

        dropped = sorted(UNCOVERED_BONES & set(by_name))
        if dropped:
            warnings.append(
                f"no slider for {', '.join(dropped)} (will not be exported)"
            )

        apply_sliders(context)

        message = f"Loaded {loaded} slider groups from {path.name}"
        if warnings:
            message += " - " + "; ".join(warnings)
        self.report({"WARNING" if warnings else "INFO"}, message)
        return {"FINISHED"}


@classes.register
class PSO2_OT_ExportShapeAdjust(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ExportHelper
):
    """Export the current slider values as a shape-adjust motion (_sa.aqm)
    compatible with the game and mod managers"""

    bl_label = "Export Shape Adjust"
    bl_idname = "pso2.export_shape_adjust"

    filename_ext = ".aqm"
    filter_glob: bpy.props.StringProperty(default="*.aqm", options={"HIDDEN"})

    # The player standard skeleton in every observed _sa.aqm.
    NODE_COUNT = 172

    def execute(self, context) -> OperatorResult:
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}

        armature = import_aqm._find_target_armature(context)

        # Node names come from the armature when available so the file
        # reads naturally in other tools; the game maps nodes by index.
        names: dict[int, str] = {}
        if armature is not None:
            bones_by_id, _ = import_aqm._get_bone_maps(armature)
            names = {
                index: bone_name.split("#")[0]
                for index, bone_name in bones_by_id.items()
            }
        ids_by_name = {name: index for index, name in names.items()}

        adjusted: dict[int, dict] = {}
        for group in GROUPS:
            if settings.is_neutral(group.key):
                continue

            values = settings.group_values(group.key)
            for bone_name, pos, quat in _sides(group, values):
                index = ids_by_name.get(bone_name, group.node_ids.get(bone_name))
                if index is None:
                    continue
                adjusted[index] = {
                    "scale": values["scale"],
                    "pos": pos,
                    "quat": quat,
                }

        if not adjusted:
            self.report({"ERROR"}, "All sliders are neutral; nothing to export")
            return {"CANCELLED"}

        motion = self._build_motion(names, adjusted)
        path = Path(self.filepath)  # type: ignore
        aqm.write_aqm(path, motion)

        self.report(
            {"INFO"},
            f"Exported shape adjust for {len(adjusted)} bones to {path.name}",
        )
        return {"FINISHED"}

    def _build_motion(self, names: dict[int, str], adjusted: dict[int, dict]):
        """A two-frame motion in the exact shape of real _sa.aqm files.

        Frame 0 is neutral and frame 1 carries the deltas; the game only
        uses the frame1-relative-to-frame0 change, so untouched nodes get
        single neutral keys (observed files put arbitrary static values
        there - two different mods disagree on them and both work).
        """
        motion = aqm.AqmMotion(
            variant=aqm.VARIANT_STD_ANIM,
            loop_point=0,
            end_frame=1,
            frame_speed=30.0,
            node_count=self.NODE_COUNT,
        )

        def key_set(key_type, data_type, keys):
            return aqm.AqmKeySet(
                key_type=key_type,
                data_type=data_type,
                unk_int0=0,
                # Real files store plain [0, 0x10] on two-key channels.
                timings=[0, 0x10] if len(keys) > 1 else [],
                vec4_keys=keys,
            )

        for index in range(self.NODE_COUNT):
            node = aqm.AqmNode(
                node_type=aqm.NODE_TYPE_STANDARD,
                node_id=index,
                name=names.get(index, f"node{index}"),
            )

            entry = adjusted.get(index)
            if entry is None:
                node.key_sets = [
                    key_set(aqm.KEY_TYPE_POSITION, 0x1, [(0.0, 0.0, 0.0, 0.0)]),
                    key_set(aqm.KEY_TYPE_ROTATION, 0x3, [(0.0, 0.0, 0.0, 1.0)]),
                    key_set(aqm.KEY_TYPE_SCALE, 0x1, [(1.0, 1.0, 1.0, 0.0)]),
                ]
            else:
                px, py, pz = entry["pos"]
                qx, qy, qz, qw = entry["quat"]
                sx, sy, sz = entry["scale"]
                node.key_sets = [
                    key_set(
                        aqm.KEY_TYPE_POSITION,
                        0x1,
                        [(0.0, 0.0, 0.0, 0.0), (px, py, pz, 0.0)],
                    ),
                    key_set(
                        aqm.KEY_TYPE_ROTATION,
                        0x3,
                        [(0.0, 0.0, 0.0, 1.0), (qx, qy, qz, qw)],
                    ),
                    key_set(
                        aqm.KEY_TYPE_SCALE,
                        0x1,
                        [(1.0, 1.0, 1.0, 0.0), (sx, sy, sz, 0.0)],
                    ),
                ]

            motion.nodes.append(node)

        return motion


# The panel lives in panels/shape_adjust.py: it parents onto the PSO2
# Appearance panel, which must be registered first, and the panels package
# is imported after this module.
