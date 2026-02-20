from dataclasses import dataclass, field

from .. import colors as clr
from .. import material as mat


@dataclass
class ShaderData:
    material: mat.Material
    textures: mat.MaterialTextures
    color_map: clr.ColorMapping | None = field(default_factory=clr.ColorMapping)
    uv_map: mat.UVMapping | None = None
