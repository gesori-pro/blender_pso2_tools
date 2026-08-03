"""
Read the colour choices out of a character file into the scene's PSO2
colour properties.

Character files store colours in the `ngsCOL2` block: eighteen entries of
four bytes each, matching the addon's eighteen ColorId channels one for
one. The block is identical across file versions (V10 through V16) apart
from where it starts, which the layout tables already handle.

The three colour bytes are ordered **blue, green, red**. Two things say
so independently: no skin tone has more blue in it than red, and reading
a character's skinColor1 of 199/217/255 forwards gives pale blue skin
where backwards gives (255, 217, 199), an ordinary light complexion; and
skinColor2 reads backwards as a dark red, which is what the sub-skin
channel defaults to here (DEFAULT_SUB_SKIN is pure red) and forwards as a
blue that nothing would use.

The stored bytes are the sRGB values the in-game colour picker shows,
while Blender's COLOR properties are linear, so they are converted on the
way in. The alpha byte is not an opacity - it varies independently of the
visible colour (skinColor2 stores 39, hair stores 0) and no use for it has
been established - so it is left at 1.0.
"""

import bpy

from .charfile import CharacterFile
from .colors import COLOR_CHANNELS, ColorId

# ColorId -> character-file field. The four "cast" channels map onto the
# NGS main/sub colour set, which is what cast parts are tinted with.
COLOR_FIELDS = {
    ColorId.OUTER1: "ngsCOL2.outerColor1",
    ColorId.OUTER2: "ngsCOL2.outerColor2",
    ColorId.BASE1: "ngsCOL2.baseColor1",
    ColorId.BASE2: "ngsCOL2.baseColor2",
    ColorId.INNER1: "ngsCOL2.innerColor1",
    ColorId.INNER2: "ngsCOL2.innerColor2",
    ColorId.CAST1: "ngsCOL2.mainColor",
    ColorId.CAST2: "ngsCOL2.subColor1",
    ColorId.CAST3: "ngsCOL2.subColor2",
    ColorId.CAST4: "ngsCOL2.subColor3",
    ColorId.MAIN_SKIN: "ngsCOL2.skinColor1",
    ColorId.SUB_SKIN: "ngsCOL2.skinColor2",
    ColorId.RIGHT_EYE: "ngsCOL2.rightEyeColor",
    ColorId.LEFT_EYE: "ngsCOL2.leftEyeColor",
    ColorId.EYEBROW: "ngsCOL2.eyebrowColor",
    ColorId.EYELASH: "ngsCOL2.eyelashColor",
    ColorId.HAIR1: "ngsCOL2.hairColor1",
    ColorId.HAIR2: "ngsCOL2.hairColor2",
}


def srgb_to_linear(value: float) -> float:
    """sRGB 0-1 -> linear 0-1, the standard piecewise transfer function."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def read_colors(
    char: CharacterFile,
) -> dict[ColorId, tuple[float, float, float, float]]:
    """Character file -> linear RGBA per colour channel."""
    result = {}

    for color_id, field in COLOR_FIELDS.items():
        if field not in char:
            continue

        raw = char[field]
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            continue

        # Stored blue, green, red - reverse into Blender's order.
        rgb = tuple(srgb_to_linear(int(c) / 255.0) for c in reversed(raw[:3]))
        result[color_id] = (*rgb, 1.0)

    return result


def apply_to_scene(context: bpy.types.Context, char: CharacterFile) -> int:
    """Write the file's colours into the scene properties.

    Returns how many channels were set. Materials read these when a model
    is imported, so an already-imported model keeps its current look until
    it is re-imported - the values are still correct for the next import.
    """
    scene = context.scene
    applied = 0

    for color_id, rgba in read_colors(char).items():
        channel = COLOR_CHANNELS.get(color_id)
        if channel is None:
            continue

        name = channel.custom_property_name
        if not hasattr(scene, name):
            continue

        try:
            setattr(scene, name, rgba)
            applied += 1
        except (AttributeError, TypeError, ValueError):
            continue

    return applied
