import bpy

from ..material import UVMapping


def set_uv_map_range(map_range: bpy.types.ShaderNodeMapRange, uv_map: UVMapping):
    map_range.data_type = "FLOAT_VECTOR"
    map_range.clamp = False
    map_range.inputs[7].default_value[0] = uv_map.from_u_min  # type: ignore
    map_range.inputs[8].default_value[0] = uv_map.from_u_max  # type: ignore
    map_range.inputs[9].default_value[0] = uv_map.to_u_min  # type: ignore
    map_range.inputs[10].default_value[0] = uv_map.to_u_max  # type: ignore
