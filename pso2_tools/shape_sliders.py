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

# Scene property holding every adjustment a loaded _sa.aqm carried, keyed by
# node index. The sliders only reach a handful of bones, but real files also
# scale knees, fingers, the neck and the spine - arm thickness and height are
# built out of those - so an export that only wrote the slider bones would
# quietly drop the rest. Kept here so it can be written back out.
CARRIED = "pso2_shape_carried"

# Armature custom properties recording the body as Blender-space deltas,
# bone name -> [sx,sy,sz, lx,ly,lz, qw,qx,qy,qz]. One for the character
# file's proportions, one for the sliders and the shape file, so either
# can be redone without recomputing the other. Motion import composes
# their product into the action's curves - the game keeps the body under
# every frame of every motion, and a preview that loses it on the keyed
# channels puts the hand somewhere the game does not (measured: 6.5 cm at
# the fingertip on one body).
BODY_FNP_PROP = "pso2_body_fnp"
BODY_SA_PROP = "pso2_body_sa"

_IDENTITY_PIECES = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def pack_pieces(scale_mul, loc_off, delta_local) -> list[float]:
    """One bone's body delta as the flat list the records store."""
    return [
        scale_mul[0],
        scale_mul[1],
        scale_mul[2],
        loc_off[0],
        loc_off[1],
        loc_off[2],
        delta_local.w,
        delta_local.x,
        delta_local.y,
        delta_local.z,
    ]


def pieces_are_identity(packed) -> bool:
    return all(abs(a - b) < 1e-9 for a, b in zip(packed, _IDENTITY_PIECES, strict=True))


def get_body_deltas(armature) -> dict[str, dict]:
    """The whole loaded body, bone name -> composed Blender-space delta.

    The character file's part and the shape file's part are composed the
    way they were applied: character first, shape on top, each a right
    delta in the bone's own frame.
    """
    out: dict[str, dict] = {}

    for prop in (BODY_FNP_PROP, BODY_SA_PROP):
        stored = armature.get(prop)
        if not stored:
            continue
        for name, packed in dict(stored).items():
            if name not in armature.pose.bones or len(packed) != 10:
                continue
            scale = (packed[0], packed[1], packed[2])
            loc = Vector((packed[3], packed[4], packed[5]))
            quat = Quaternion((packed[6], packed[7], packed[8], packed[9]))

            entry = out.get(name)
            if entry is None:
                out[name] = {"scale": scale, "location": loc, "rotation": quat}
            else:
                entry["scale"] = tuple(
                    a * b for a, b in zip(entry["scale"], scale, strict=True)
                )
                entry["location"] = entry["location"] + loc
                entry["rotation"] = entry["rotation"] @ quat

    return {
        name: entry
        for name, entry in out.items()
        if not pieces_are_identity(
            pack_pieces(entry["scale"], entry["location"], entry["rotation"])
        )
    }


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
# Panel order, and the order these are written out. Top of the body to the
# bottom, so a slider is where the part it moves is - the arrangement Tae
# asked for after working with the first version, where the leg groups sat
# in the order they happened to be added.
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
    # Shoulder width, really: the collarbone carries l_upperarm and the arm
    # with it, and 2760 chest vertices ride along at up to 0.45 weight.
    Group(
        "clav",
        "Clavicle",
        "l_clavicle",
        "r_clavicle",
        {"l_clavicle": 22, "r_clavicle": 30},
        rotate=False,
    ),
    # hip (2) is the parent of pelvis (3), so it comes first. hip carries
    # 19622 vertices of its own; pelvis has none but every leg bone inherits
    # its scale, which makes it the whole-leg control.
    Group("hip", "Hip", "hip", None, {"hip": 2}),
    Group("pelvis", "Pelvis", "pelvis", None, {"pelvis": 3}),
    Group(
        "hiptw", "Hip Twist", "l_hip_tw", "r_hip_tw", {"l_hip_tw": 50, "r_hip_tw": 51}
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
        "thightw2",
        "Thigh Twist 2",
        "l_thigh_tw2_alt",
        "r_thigh_tw2_alt",
        {"l_thigh_tw2_alt": 54, "r_thigh_tw2_alt": 65},
    ),
    Group(
        "calf0",
        "Calf Upper",
        "l_calf0_alt",
        "r_calf0_alt",
        {"l_calf0_alt": 55, "r_calf0_alt": 66},
    ),
    Group(
        "calf",
        "Calf",
        "l_calf_alt",
        "r_calf_alt",
        {"l_calf_alt": 56, "r_calf_alt": 67},
    ),
    Group(
        "foot",
        "Foot",
        "l_foot_alt",
        "r_foot_alt",
        {"l_foot_alt": 57, "r_foot_alt": 68},
    ),
]

GROUPS_BY_KEY = {g.key: g for g in GROUPS}

IDENTITY_SCALE = (1.0, 1.0, 1.0)
IDENTITY_VEC = (0.0, 0.0, 0.0)


def store_carried(context, deltas: dict) -> int:
    """Remember every adjustment a loaded file made, slider or not."""
    kept = {}
    for index, entry in deltas.items():
        if not (entry.get("scale") or entry.get("pos") or entry.get("rotQuat")):
            continue
        kept[str(index)] = {
            "name": entry.get("name") or "",
            "scale": list(entry["scale"] or IDENTITY_SCALE),
            "pos": list(entry["pos"] or IDENTITY_VEC),
            "quat": list(entry["rotQuat"] or (0.0, 0.0, 0.0, 1.0)),
        }

    context.scene[CARRIED] = kept
    return len(kept)


def get_carried(context) -> dict[int, dict]:
    """The stored adjustments, keyed by node index."""
    stored = context.scene.get(CARRIED)
    if not stored:
        return {}

    out = {}
    for index, entry in dict(stored).items():
        try:
            out[int(index)] = {
                "name": str(entry["name"]),
                "scale": tuple(entry["scale"]),
                "pos": tuple(entry["pos"]),
                "quat": tuple(entry["quat"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

    return out


def clear_carried(context) -> None:
    if CARRIED in context.scene:
        del context.scene[CARRIED]


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
        # keys() is required here: iterating an RNA collection directly
        # yields property objects, and property_unset() wants the name.
        for prop in self.bl_rna.properties.keys():  # noqa: SIM118
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


def delta_to_blender(pose_bone, scale, pos, quat):
    """One PSO2 shape delta as Blender pose-channel terms.

    Returns (scale multiplier, location offset, local rotation delta), the
    three pieces `_apply_to_bone` composes onto the base pose. Export needs
    the same pieces to take a delta back off a sampled key, so the axis
    conversion lives here once rather than in both directions.

    Scale swaps components 0/1 into Blender order (SPEC §6-2). Position is
    the (x,y,z) -> (y,x,-z) permutation carried into bone-local space by
    the rest rotation (SPEC §6-3). Rotation is the same permutation as a
    right-side delta in the bone's own frame, verified against the game's
    composed bone array.
    """
    bone = pose_bone.bone

    if bone.parent is not None:
        rest = bone.parent.matrix_local.inverted() @ bone.matrix_local
    else:
        rest = bone.matrix_local.copy()
    rest_rotation = rest.to_quaternion()

    return (
        (scale[1], scale[0], scale[2]),
        rest_rotation.inverted() @ Vector((pos[1], pos[0], -pos[2])),
        Quaternion((quat[3], quat[1], quat[0], -quat[2])),
    )


def _apply_to_bone(pose_bone, scale, pos, quat):
    """pose = base ∘ delta, through the verified axis conversion."""
    base_loc, base_rot, base_scale = _ensure_base(pose_bone)
    scale_mul, loc_off, delta_local = delta_to_blender(pose_bone, scale, pos, quat)

    pose_bone.scale = (
        base_scale[0] * scale_mul[0],
        base_scale[1] * scale_mul[1],
        base_scale[2] * scale_mul[2],
    )
    pose_bone.location = base_loc + loc_off

    # The preview composes the same way the game does, so what is on
    # screen is what the exported file will do in game.
    pose_bone.rotation_mode = "QUATERNION"
    pose_bone.rotation_quaternion = base_rot @ delta_local

    return scale_mul, loc_off, delta_local


def _sides(group, values):
    """(bone name, position, quaternion) for each side of a group."""
    quat = euler_deg_to_quat(values["rot"])
    out = [(group.left, values["pos"], quat)]
    if group.right is not None:
        out.append((group.right, mirror_pos(values["pos"]), mirror_quat(quat)))
    return out


def has_shape(context) -> bool:
    """Is there a body shape to put back on a pose that was rebuilt?"""
    if get_carried(context):
        return True

    settings = get_settings(context)
    return settings is not None and not all(
        settings.is_neutral(group.key) for group in GROUPS
    )


def apply_sliders(context, armature=None) -> dict:
    settings = get_settings(context)
    if armature is None:
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
    covered: set[str] = set()
    record: dict[str, list[float]] = {}
    for group in GROUPS:
        values = settings.group_values(group.key)
        for bone_name, pos, quat in _sides(group, values):
            name = bones_by_name.get(bone_name)
            if name is None:
                continue
            pieces = _apply_to_bone(
                armature.pose.bones[name], values["scale"], pos, quat
            )
            packed = pack_pieces(*pieces)
            if not pieces_are_identity(packed):
                record[name] = packed
            covered.add(name)
            applied += 1

    # The rest of what a loaded file adjusted - arms, knees, fingers, neck -
    # has no slider, and was kept for export without ever reaching the pose.
    # The body on screen was then not the body in the game, so a pose built
    # against it does not land where it looked like it would: one file on
    # hand thickens the arms by a tenth, and that is what an author sees as
    # a hand coming out fatter in game than it went in.
    bones_by_id, _ = import_aqm._get_bone_maps(armature)
    carried = 0
    for index, entry in get_carried(context).items():
        name = bones_by_id.get(index) or bones_by_name.get(entry["name"].lower())
        if name is None or name in covered:
            continue

        pieces = _apply_to_bone(
            armature.pose.bones[name], entry["scale"], entry["pos"], entry["quat"]
        )
        packed = pack_pieces(*pieces)
        if not pieces_are_identity(packed):
            record[name] = packed
        carried += 1

    # What this pass put on the pose, for motion import to keep composed
    # under any animation the way the game does.
    armature[BODY_SA_PROP] = record

    bpy.context.view_layer.update()
    return {"applied": applied, "carried": carried, "disconnected": disconnected}


# ---------------------------------------------------------------------------
# Operators


def load_shape_adjust(context, motion, source_name: str, armature=None) -> dict | None:
    """Put a shape adjust's values on the sliders and onto the pose.

    Shared by the Load AQM button and by model import, which finds the
    outfit's own _sa.aqm sitting in the same archive and names the armature
    it belongs to. Returns None when there is nowhere to put the values.
    """
    settings = get_settings(context)
    if settings is None:
        return None

    deltas = import_shape_adjust.extract_frame1_deltas(motion)
    by_name = {
        entry["name"].lower(): entry for entry in deltas.values() if entry["name"]
    }

    settings.reset()
    groups = 0
    warnings: list[str] = []

    for group in GROUPS:
        entry = by_name.get(group.left)
        if entry is None:
            continue

        scale = entry["scale"] or IDENTITY_SCALE
        pos = entry["pos"] or IDENTITY_VEC
        rot = quat_to_euler_deg(entry["rotQuat"]) if entry["rotQuat"] else IDENTITY_VEC
        settings.set_group_values(group.key, scale=scale, pos=pos, rot=rot)
        groups += 1

        # The sliders are symmetric; note when the file is not.
        other = by_name.get(group.right) if group.right else None
        if other is not None:
            mirrored = mirror_pos(pos)
            if any(
                abs(a - b) > 1e-4
                for a, b in zip(other["scale"] or IDENTITY_SCALE, scale, strict=True)
            ) or any(
                abs(a - b) > 1e-4
                for a, b in zip(other["pos"] or IDENTITY_VEC, mirrored, strict=True)
            ):
                warnings.append(f"{group.label} L/R differ, left side used")

    # Everything the file touched is kept, so exporting later writes the
    # bones the sliders cannot reach back out unchanged.
    carried = store_carried(context, deltas)
    applied = apply_sliders(context, armature)

    return {
        "groups": groups,
        "carried": carried,
        "warnings": warnings,
        "source": source_name,
        "applied": applied.get("applied", 0),
    }


@classes.register
class PSO2_OT_ShapeSlidersReset(bpy.types.Operator):
    """Reset all shape sliders to neutral, restoring the base pose. Also
    forgets a loaded file's other adjustments, so the next export starts
    from nothing"""

    bl_label = "Reset Sliders"
    bl_idname = "pso2.shape_sliders_reset"
    bl_options = {"UNDO"}

    def execute(self, context) -> OperatorResult:
        settings = get_settings(context)
        if settings is None:
            return {"CANCELLED"}

        settings.reset()
        clear_carried(context)
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
        path = Path(self.filepath)  # type: ignore
        try:
            motion = aqm.read_aqm(path)
        except (OSError, aqm.AqmError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        loaded = load_shape_adjust(context, motion, path.name)
        if loaded is None:
            return {"CANCELLED"}

        message = (
            f"Loaded {loaded['groups']} slider groups from {path.name}"
            f"; {loaded['carried']} adjusted bones kept for export"
        )
        if loaded["warnings"]:
            message += " - " + "; ".join(loaded["warnings"])
        self.report({"WARNING" if loaded["warnings"] else "INFO"}, message)
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

        # Start from whatever a loaded file adjusted - knees, fingers, neck,
        # spine and so on - then let the sliders override their own bones.
        adjusted: dict[int, dict] = {
            index: {"scale": entry["scale"], "pos": entry["pos"], "quat": entry["quat"]}
            for index, entry in get_carried(context).items()
        }
        carried = len(adjusted)

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
            self.report(
                {"ERROR"},
                "All sliders are neutral and no file has been loaded;"
                " nothing to export",
            )
            return {"CANCELLED"}

        from_sliders = len(adjusted) - carried

        motion = self._build_motion(names, adjusted)
        path = Path(self.filepath)  # type: ignore
        aqm.write_aqm(path, motion)

        message = f"Exported shape adjust for {len(adjusted)} bones to {path.name}"
        if carried:
            message += (
                f" ({carried} kept from the loaded file,"
                f" {from_sliders} added by the sliders)"
            )
        self.report({"INFO"}, message)
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
