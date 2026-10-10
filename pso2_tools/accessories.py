"""Hang a character file's accessories where the game puts them.

A character holds up to 18 accessories (12 before file version 15). Each
slot keeps an item id, an attach point, nine adjustment sliders (position,
rotation and scale, three axes each, -127..127) and two colour choices.

Placing one takes three things from three places:

- The CMX pairs the accessory's nodes with character bones (see
  objects.get_accessory_cmx). The accessory's root is authored exactly on
  its bone's bind pose - a ring's root is r_finger_21's bind matrix to the
  last digit - so a node follows its bone by keeping that rest offset.
- The attach point, when it is not 0, moves the root onto another bone. Its
  numbers index the creator's Attach_PointList (ui_charamake_text_reb.text),
  which names 66 points from Default through Right_ForthFinger3.
- The sliders pose the accessory's adjust node(s) - acce_top, acce_top1 for
  a pair, l_foot_skin/r_foot_skin for boots - through the accessory's own
  _fig.aqm. That motion keeps every slider on a stretch of its own frames:
  the rest pose in the middle and the slider's two ends one frame either
  side (position about frames 2, 6 and 10, rotation 14, 18 and 22, scale
  26, 30 and 34). Ranges and axes differ per accessory - a ring moves 0.1
  along its own Y first, a horn 0.2 along -Y - so they are read, not
  assumed. All 40 accessories sampled keep this layout.

Not yet checked against the game: which end of a stretch +127 lands on,
how several rotation sliders compose, the bone behind a few attach points
(the jaw, thumbs), whether a moved accessory keeps its authored offset, and
what the colour choices tint. Classic (pl_ah_*) accessories have no
_fig.aqm, so they are attached but not adjusted.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector

from . import aqm, charfile, ice, import_model, objects
from .colors import ColorId, ColorMapping
from .debug import debug_print

SLOTS = 18

# PSO2 is Y-up, Blender Z-up: a PSO2 point (x, y, z) lands at (x, -z, y).
_TO_BLENDER = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
_FROM_BLENDER = _TO_BLENDER.transposed()

# An imported bone's rest is its node's frame turned 180 degrees about
# (1, 1, 0): PSO2 bones run along X and Blender's along Y. Measured on
# body, face and accessory rigs, through FBX and the direct importer alike,
# so it converts any posed bone back to the game's frame. It is its own
# inverse.
_BONE_AXES = Matrix(((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))

# AQM key types
_POSITION = 1
_ROTATION = 2
_SCALE = 3

# The middle frame of each slider's stretch in an accessory's _fig.aqm, in
# the file's slider order.
_SLIDER_FRAMES = {
    _POSITION: (2, 6, 10),
    _ROTATION: (14, 18, 22),
    _SCALE: (26, 30, 34),
}

# Which end of the stretch +127 reaches. Taken as the frame before the
# middle, which is the positive end for every channel: +0.1 on a ring's
# first position stretch, +179 degrees on its rotation stretches and 1.5x
# on its scale ones. Not yet measured in the game.
_SLIDER_SIGN = -1.0

# The creator's Attach_PointList, in file order: the display name and the
# character bone it is taken to mean. Index 0 keeps the CMX's own bone.
# Where the CMX's default attach bones use the name, the match is direct
# (headtop_efhd, eyesroot, nosebridge, neck0, spine1_efce, *_breast_top,
# *_foot_eff*, *_finger_21 ...); the jaw, head-of-head and thumb bones are
# the likeliest candidates, not yet confirmed in the game.
ATTACH_POINTS: tuple[tuple[str, str | None], ...] = (
    ("Default", None),
    ("HeadTop", "headtop_efhd"),
    ("EyesRoot", "eyesroot"),
    ("NoseBridge", "nosebridge"),
    ("JawUpper", "mouthattach"),
    ("JawLower", "mouthattach2"),
    ("RightEar", "rear_2"),
    ("LeftEar", "lear_2"),
    ("HeadofHead", "head2"),
    ("Neck", "neck"),
    ("NeckRoot", "neck0"),
    ("Right_Breast", "r_breast_top"),
    ("Center_Breast", "c_breast_top"),
    ("Left_Breast", "l_breast_top"),
    ("Spine1", "spine1_efce"),
    ("Spine2", "spine2"),
    ("Waist", "hip"),
    ("Crotch", "crotch"),
    ("Left_Clavicle", "l_clavicle"),
    ("Left_UpperArm1", "l_upperarm"),
    ("Left_UpperArm2", "l_upperarm_tw"),
    ("Left_ForeArm1", "l_forearm"),
    ("Left_ForeArm2", "l_forearm_tw"),
    ("Left_Hand", "l_hand"),
    ("Right_Clavicle", "r_clavicle"),
    ("Right_UpperArm1", "r_upperarm"),
    ("Right_UpperArm2", "r_upperarm_tw"),
    ("Right_ForeArm1", "r_forearm"),
    ("Right_ForeArm2", "r_forearm_tw"),
    ("Right_Hand", "r_hand"),
    ("Left_Thigh1", "l_thigh"),
    ("Left_Thigh2", "l_thigh_tw"),
    ("Left_Knee", "l_knee"),
    ("Left_Ankle", "l_foot_effl"),
    ("Right_Thigh1", "r_thigh"),
    ("Right_Thigh2", "r_thigh_tw"),
    ("Right_Knee", "r_knee"),
    ("Right_Ankle", "r_foot_effr"),
    *(
        (f"{side}_{finger}{joint}", f"{side[0].lower()}_finger_{digit}{joint}")
        for side in ("Left", "Right")
        for finger, digit, joints in (
            ("Thumb", 0, (1, 2)),
            ("FirstFinger", 1, (1, 2, 3)),
            ("SecondFinger", 2, (1, 2, 3)),
            ("ThirdFinger", 3, (1, 2, 3)),
            ("ForthFinger", 4, (1, 2, 3)),
        )
        for joint in joints
    ),
)


@dataclass
class Slot:
    """One accessory as the character file sets it."""

    number: int  # 1-based, as the creator numbers them
    item_id: int
    attach: int = 0
    position: tuple[int, int, int] = (0, 0, 0)
    rotation: tuple[int, int, int] = (0, 0, 0)
    scale: tuple[int, int, int] = (0, 0, 0)
    colors: tuple[int, int] = (0, 0)

    @property
    def adjusted(self) -> bool:
        return any(self.position) or any(self.rotation) or any(self.scale)

    def sliders(self, key_type: int) -> tuple[int, int, int]:
        return {
            _POSITION: self.position,
            _ROTATION: self.rotation,
            _SCALE: self.scale,
        }[key_type]


def _field(char: charfile.CharacterFile, suffix: str):
    for name in char:
        if name.split(".")[-1] == suffix:
            return char[name]
    return None


def read_slots(char: charfile.CharacterFile) -> list[Slot]:
    """The accessories a character file wears, in slot order."""
    position = _field(char, "posSliders") or []
    rotation = _field(char, "rotSliders") or []
    scale = _field(char, "scaleSliders") or []
    attach = _field(char, "accessoryAttach") or []
    colors = _field(char, "accessoryColorChoices") or []

    def triple(values, i) -> tuple[int, int, int]:
        part = tuple(values[3 * i : 3 * i + 3])
        return part if len(part) == 3 else (0, 0, 0)  # type: ignore[return-value]

    slots = []
    for number in range(1, SLOTS + 1):
        item_id = _field(char, f"acc{number}Part")
        if not isinstance(item_id, int) or item_id <= 0:
            continue
        i = number - 1
        slots.append(
            Slot(
                number=number,
                item_id=item_id,
                attach=attach[i] if i < len(attach) else 0,
                position=triple(position, i),
                rotation=triple(rotation, i),
                scale=triple(scale, i),
                colors=(
                    (colors[2 * i], colors[2 * i + 1])
                    if 2 * i + 1 < len(colors)
                    else (0, 0)
                ),
            )
        )
    return slots


def attach_bone(slot: Slot) -> str | None:
    """The bone the slot's attach point moves the accessory to, if any."""
    if 0 < slot.attach < len(ATTACH_POINTS):
        return ATTACH_POINTS[slot.attach][1]
    return None


def tint(slot: Slot) -> ColorMapping | None:
    """The character colours the slot paints the accessory's mask with.

    Each choice is a ColorId (0 = the accessory's own colours, 3 = base 1,
    17 = hair 1, ...), the same numbering the CMX gives face paints, and
    like a face paint's they are taken to go to the mask's red and green.
    """
    if not any(slot.colors):
        return None

    def color(value: int) -> ColorId:
        try:
            return ColorId(value)
        except ValueError:
            return ColorId.UNUSED

    return ColorMapping(red=color(slot.colors[0]), green=color(slot.colors[1]))


@dataclass
class _TintedAccessory(objects.CmxAccessory):
    """An accessory imported in the colours its slot chose."""

    tint: ColorMapping = field(default_factory=ColorMapping)

    def get_color_map(self) -> ColorMapping:
        return self.tint


def with_tint(obj: objects.CmxAccessory, slot: Slot) -> objects.CmxAccessory:
    mapping = tint(slot)
    if mapping is None:
        return obj
    values = {f.name: getattr(obj, f.name) for f in fields(obj)}
    return _TintedAccessory(**values, tint=mapping)


# ---- the adjust motion ------------------------------------------------------


def adjust_motion(obj: objects.CmxAccessory, data_path: Path) -> aqm.AqmMotion | None:
    """The accessory's _fig.aqm, which carries its slider ranges."""
    path = import_model._get_ice_path(obj.file, data_path, True)
    if path is None:
        return None
    for data in ice.IceFile.load(path).glob("*_fig.aqm"):
        try:
            return aqm.parse_aqm(data.data)
        except aqm.AqmError as ex:
            debug_print(f"Could not read {data.name}:", ex)
    return None


def _sample(key_set: aqm.AqmKeySet, frame: float, rotation: bool):
    frames = key_set.frames()
    values = key_set.vec4_keys
    if len(values) == 1 or frame <= frames[0]:
        return values[0]
    for i in range(1, len(values)):
        if frame <= frames[i]:
            start, end = frames[i - 1], frames[i]
            t = (frame - start) / (end - start) if end > start else 0.0
            a, b = values[i - 1], values[i]
            if rotation:
                qa = Quaternion((a[3], a[0], a[1], a[2]))
                qb = Quaternion((b[3], b[0], b[1], b[2]))
                q = qa.slerp(qb, t)
                return (q.x, q.y, q.z, q.w)
            return tuple(x + (y - x) * t for x, y in zip(a, b, strict=True))
    return values[-1]


def _quaternion(value) -> Quaternion:
    return Quaternion((value[3], value[0], value[1], value[2]))


def _frame(centre: int, value: int) -> float:
    return centre + _SLIDER_SIGN * max(-127, min(127, value)) / 127.0


def adjust_delta(node: aqm.AqmNode, slot: Slot) -> Matrix | None:
    """How the slot's sliders move one adjust node, in the node's own frame.

    None for a node the motion does not adjust. Position stretches add up,
    scale ones multiply, and rotation ones turn the node about its own axes
    in slider order.
    """
    channels = {
        key_type: key_set
        for key_type in (_POSITION, _ROTATION, _SCALE)
        if (key_set := node.get_key_set(key_type)) is not None and key_set.key_count > 1
    }
    if not channels:
        return None

    def rest(key_type: int, default):
        key_set = node.get_key_set(key_type)
        return _sample(key_set, 0, key_type == _ROTATION) if key_set else default

    rest_position = Vector(rest(_POSITION, (0, 0, 0, 0))[:3])
    rest_rotation = _quaternion(rest(_ROTATION, (0, 0, 0, 1)))
    rest_scale = Vector(rest(_SCALE, (1, 1, 1, 0))[:3])

    position = rest_position.copy()
    rotation = rest_rotation.copy()
    scale = rest_scale.copy()

    for key_type, key_set in channels.items():
        is_rotation = key_type == _ROTATION
        for centre, value in zip(
            _SLIDER_FRAMES[key_type], slot.sliders(key_type), strict=True
        ):
            if not value:
                continue
            middle = _sample(key_set, centre, is_rotation)
            end = _sample(key_set, _frame(centre, value), is_rotation)
            if key_type == _POSITION:
                position += Vector(end[:3]) - Vector(middle[:3])
            elif is_rotation:
                rotation = rotation @ (
                    _quaternion(middle).inverted() @ _quaternion(end)
                )
            else:
                scale = Vector(
                    s * (e / m if abs(m) > 1e-9 else 1.0)
                    for s, e, m in zip(scale, end[:3], middle[:3], strict=True)
                )

    before = Matrix.LocRotScale(rest_position, rest_rotation, rest_scale)
    after = Matrix.LocRotScale(position, rotation, scale)
    return before.inverted() @ after


# ---- placing it -------------------------------------------------------------


def _base(name: str) -> str:
    return name.split("#")[0]


def _game_matrix(armature: bpy.types.Object, bone_matrix: Matrix) -> Matrix:
    """A bone's armature-space matrix as the game's node frame, in PSO2 space."""
    return _FROM_BLENDER @ armature.matrix_world @ bone_matrix @ _BONE_AXES


def _set_game_matrix(
    armature: bpy.types.Object, pose_bone: bpy.types.PoseBone, world: Matrix
) -> None:
    pose_bone.matrix = (
        armature.matrix_world.inverted() @ _TO_BLENDER @ world @ _BONE_AXES
    )


def character_bones(
    armatures: list[bpy.types.Object],
) -> dict[str, tuple[bpy.types.Object, bpy.types.PoseBone]]:
    """Every character bone by its bare name.

    A name several parts share (head, neck, ...) is taken from the body
    first, then the face - the rigs the proportion pass poses directly and
    the others are aligned to.
    """

    def rank(obj: bpy.types.Object) -> int:
        bases = {_base(b.name) for b in obj.data.bones}
        if "body_root" in bases:
            return 0
        if "_rhd_" in obj.name:
            return 1
        return 2

    found: dict[str, tuple[bpy.types.Object, bpy.types.PoseBone]] = {}
    for obj in sorted(armatures, key=rank):
        for pose_bone in obj.pose.bones:
            found.setdefault(_base(pose_bone.name), (obj, pose_bone))
    return found


@dataclass
class Placed:
    """An imported accessory waiting to be hung on the character."""

    slot: Slot
    item: objects.CmxAccessory
    armature: bpy.types.Object
    motion: aqm.AqmMotion | None = None


def place(
    context: bpy.types.Context,
    accessory: Placed,
    pairs: tuple[tuple[str, str], ...],
    character: dict[str, tuple[bpy.types.Object, bpy.types.PoseBone]],
) -> list[str]:
    """Pose one accessory onto the character. Returns what could not be done."""
    problems: list[str] = []
    armature = accessory.armature
    nodes = {_base(b.name): b for b in armature.pose.bones}

    # The CMX calls the accessory's root "root"; the model names it after
    # itself (pl_rac_106140).
    root = next((b for b in armature.pose.bones if b.parent is None), None)
    if root is None:
        return ["no root node"]
    nodes.setdefault("root", root)
    if not pairs:
        pairs = (("", "root"),)

    # "all" (77 records, nails and eyelash pieces) hangs every node that
    # shares a character bone's name - r_finger_43, eyelid_l_01 - on that
    # bone, and leaves the root with the character.
    if any(bone == "all" for bone, _ in pairs):
        pairs = tuple(
            (_base(pose_bone.name), _base(pose_bone.name))
            for pose_bone in armature.pose.bones
            if pose_bone is not root and _base(pose_bone.name) in character
        )
        moved_to = None
    else:
        moved_to = attach_bone(accessory.slot)

    for index, (bone, node) in enumerate(pairs):
        pose_bone = nodes.get(node)
        if pose_bone is None:
            problems.append(f"no node {node}")
            continue
        default = character.get(bone)
        target = character.get(moved_to) if index == 0 and moved_to else default
        if target is None:
            problems.append(f"no bone {moved_to if index == 0 and moved_to else bone}")
            continue
        target_armature, target_bone = target

        # The node keeps the rest offset it was authored with against its
        # default bone, and follows whichever bone now carries it.
        node_rest = _game_matrix(armature, pose_bone.bone.matrix_local)
        if default is not None:
            default_rest = _game_matrix(default[0], default[1].bone.matrix_local)
            offset = default_rest.inverted() @ node_rest
        else:
            offset = Matrix.Identity(4)
        context.view_layer.update()
        world = _game_matrix(target_armature, target_bone.matrix) @ offset
        _set_game_matrix(armature, pose_bone, world)

    if accessory.slot.adjusted:
        if accessory.motion is None:
            problems.append("no adjust motion, sliders not applied")
        else:
            for motion_node in accessory.motion.nodes:
                delta = adjust_delta(motion_node, accessory.slot)
                pose_bone = nodes.get(motion_node.name)
                if delta is None or pose_bone is None:
                    continue
                context.view_layer.update()
                current = _game_matrix(armature, pose_bone.matrix)
                _set_game_matrix(armature, pose_bone, current @ delta)

    context.view_layer.update()
    return problems
