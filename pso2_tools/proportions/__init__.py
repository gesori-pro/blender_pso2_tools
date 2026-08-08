"""
Compute PSO2 NGS bone proportions from a character file.

Port of the reverse-engineering reference implementation (RE/handoff/
python/proportions.py), which was verified bone-for-bone against the
running game: scale 133/133 within 0.02%, position 134/134 within 0.2 mm,
rotation 132/134 within 0.25° (SPEC §1, §6-10).

Everything here stays in PSO2's own axis order (X = bone length axis).
Converting to Blender (the component permutation of SPEC §6-2/§6-3) is
import_fnp's job.

The weight tables bundled next to this module were extracted from the
game's proportion motions (character/making/pl_system_data.ice) with
propdump; regenerate them from the game data if SEGA changes the motions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..charfile import CharacterFile

_DATA_DIR = Path(__file__).parent

MIN_KEY = "atMin(slider=-127)"
MAX_KEY = "atMax(slider=+127)"

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)

# Weight-table slider keys are prose ("bodyVerts.X physique CenterY");
# character-file fields are paths ("baseFIGR.bodyVerts.X"). These are the
# fields that live at the top level rather than under baseFIGR (SPEC §2-2).
_TOP_LEVEL_FIELDS = {
    "neckVerts.X",
    "neckVerts.Y",
    "neckVerts.Z",
    "waistVerts.X",
    "waistVerts.Y",
    "waistVerts.Z",
    "hands.X",
    "hands.Y",
    "hands.Z",
    "neckAngle",
}


def slider_to_field(key: str) -> str:
    """Weight-table slider key -> character-file field path (SPEC §2-2)."""
    head = key.split(" ")[0]
    if key.endswith("(ngsSLID)"):
        return "ngsSLID." + head
    if head in _TOP_LEVEL_FIELDS:
        return head
    return "baseFIGR." + head


def blend(value: int, neutral, at_min, at_max):
    """Signed blend away from the neutral (frame 0) pose - SPEC §6-1.

    NOT lerp(min, max, (v+127)/254): that is Salon Tool's formula and the
    game's memory disproves it (SPEC appendix A-2). Slider 0 = exactly no
    deformation.
    """
    t = value / 127.0
    if t < 0.0:
        return [n + (lo - n) * -t for n, lo in zip(neutral, at_min)]
    return [n + (hi - n) * t for n, hi in zip(neutral, at_max)]


def quat_mul(a, b):
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_scale(q, t: float):
    """Scale a rotation by t along the shortest arc from identity."""
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    w = min(1.0, max(-1.0, w))
    half = math.acos(w)
    s = math.sin(half)
    if s < 1e-9:
        return IDENTITY_QUAT
    a = math.sin((1.0 - t) * half) / s
    b = math.sin(t * half) / s
    return (x * b, y * b, z * b, a + w * b)


def blend_quat(value: int, at_min, at_max):
    """Signed blend of the rotation delta: slerp from identity by |v|/127.

    Uses rotDeltaQuat, never rotDeltaEulerDeg - rebuilding the euler form
    needs ZYX order and a wrong guess costs several degrees on the
    shoulders and bust (SPEC §6-10).
    """
    t = value / 127.0
    return quat_scale(tuple(at_min if t < 0.0 else at_max), abs(t))


def split_bone_key(key: str) -> tuple[int | None, str]:
    """Weight-table bone key '41:l_breast' -> (41, 'l_breast')."""
    head, _, name = key.partition(":")
    try:
        return int(head), name
    except ValueError:
        return None, key


def _new_slot() -> dict:
    return {
        "index": None,
        "scale": [1.0, 1.0, 1.0],
        "pos": [0.0, 0.0, 0.0],
        "rotQuat": list(IDENTITY_QUAT),
    }


def weights_path_for(char: CharacterFile) -> Path:
    """NGS body motion table for this character's gender (SPEC §6-6).

    NOTE (SPEC §9-3, unverified): whether NGS casts (race 2) use the same
    _rb body motions has not been confirmed against real files.
    """
    female = int(char["baseDOC.gender"]) == 1
    name = "pl_cmakemot_b_fh_rb.json" if female else "pl_cmakemot_b_mh_rb.json"
    return _DATA_DIR / name


def outfit_adjust(char: CharacterFile) -> dict[str, list[float]]:
    """Per-outfit shape adjustment, keyed by bone (SPEC §6-11).

    Some outfits ship pl_rbd_<id>_<slot>_sa.aqm whose frame 1 multiplies
    the bone scales on top of the proportion result. baseSLCT.costumePart
    is routed through the CMX id links to find which parts apply. Most
    outfits have none.
    """
    adjust_path = _DATA_DIR / "outfit_adjust.json"
    links_path = _DATA_DIR / "cmx_idlinks.json"
    if not (adjust_path.is_file() and links_path.is_file()):
        return {}

    with adjust_path.open(encoding="utf-8") as f:
        adjust = json.load(f)
    with links_path.open(encoding="utf-8") as f:
        links = json.load(f)

    try:
        costume = int(char["baseSLCT.costumePart"])
    except (KeyError, ValueError, TypeError):
        return {}

    result: dict[str, list[float]] = {}
    for slot, table in (
        ("ow", "outerWearIdLink"),
        ("bw", "baseWearIdLink"),
        ("b1", "innerWearIdLink"),
    ):
        part_id = links.get(table, {}).get(str(costume))
        if part_id is None:
            continue
        for bone, mul in adjust.get(slot, {}).get(str(part_id), {}).items():
            current = result.setdefault(bone, [1.0, 1.0, 1.0])
            result[bone] = [a * b for a, b in zip(current, mul)]

    return result


def compute(
    char: CharacterFile,
    weights_path: Path | None = None,
    apply_outfit_adjust=True,
) -> dict:
    """Slider values -> per-bone accumulated TRS deltas, in PSO2 axis order.

    Returns bones keyed by name: {"l_breast": {"index", "scale", "pos",
    "rotQuat"}, ...}. Position deltas are in the parent-local frame; scale
    is the non-inherited per-bone multiplier; rotQuat is the xyzw rotation
    delta composed as delta * rest (SPEC §6-10).
    """
    if weights_path is None:
        weights_path = weights_path_for(char)

    with Path(weights_path).open(encoding="utf-8") as f:
        table = json.load(f)

    accum: dict[str, dict] = {}
    applied: list[tuple[str, str, int]] = []

    for key, entry in table["sliders"].items():
        bones = entry.get("affectedBones") if isinstance(entry, dict) else None
        if not isinstance(bones, dict):
            continue  # slider absent from this body type (classic motions)

        field = slider_to_field(key)
        if field not in char:
            continue

        value = int(char[field])
        applied.append((key, field, value))

        for bone_key, deltas in bones.items():
            index, name = split_bone_key(bone_key)
            slot = accum.setdefault(name, _new_slot())
            if index is not None:
                slot["index"] = index
            lo, hi = deltas[MIN_KEY], deltas[MAX_KEY]

            s = blend(value, (1.0, 1.0, 1.0), lo["scaleMul"], hi["scaleMul"])
            slot["scale"] = [x * y for x, y in zip(slot["scale"], s)]

            p = blend(value, (0.0, 0.0, 0.0), lo["posDelta"], hi["posDelta"])
            slot["pos"] = [x + y for x, y in zip(slot["pos"], p)]

            if "rotDeltaQuat" in lo:
                q = blend_quat(value, lo["rotDeltaQuat"], hi["rotDeltaQuat"])
                # Sliders compose in table order: each new delta multiplies
                # on the right of what is already there, so the first slider
                # listed is applied first. The other way round (new delta on
                # the left) only shows on a bone three sliders drive with
                # different axes - the clavicles - where it left the shoulder
                # 0.28 deg off the game, ~6 mm at the fingertip. One-slider
                # bones are unaffected, which is why it read as 132/134.
                slot["rotQuat"] = list(quat_mul(tuple(slot["rotQuat"]), q))

    # Frame 1 of the motion is not a slider: it multiplies into every
    # character no matter what (SPEC §6-1-1). Skipping it puts a 5-10%
    # error on eleven upper-body bones, and skipping its position channel
    # leaves l/r_breast 2.5 mm out.
    def slot_for(bone_key: str) -> dict:
        index, name = split_bone_key(bone_key)
        slot = accum.setdefault(name, _new_slot())
        if index is not None and slot["index"] is None:
            slot["index"] = index
        return slot

    for bone_key, mul in table.get("_baseCorrection", {}).items():
        slot = slot_for(bone_key)
        slot["scale"] = [x * y for x, y in zip(slot["scale"], mul)]

    for bone_key, delta in table.get("_baseCorrectionPos", {}).items():
        slot = slot_for(bone_key)
        slot["pos"] = [x + y for x, y in zip(slot["pos"], delta)]

    for bone_key, q in table.get("_baseCorrectionRot", {}).items():
        slot = slot_for(bone_key)
        slot["rotQuat"] = list(quat_mul(tuple(q), tuple(slot["rotQuat"])))

    # Outfit-side shape adjustment, last (SPEC §6-11).
    outfit_bones = outfit_adjust(char) if apply_outfit_adjust else {}
    for bone_key, mul in outfit_bones.items():
        slot = slot_for(bone_key)
        slot["scale"] = [x * y for x, y in zip(slot["scale"], mul)]

    return {
        "source": Path(weights_path).name,
        "sliders": applied,
        "outfit_adjust_bones": len(outfit_bones),
        "bones": accum,
    }
