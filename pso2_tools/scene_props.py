import bpy

from . import parts
from .colors import COLOR_CHANNELS
from .preferences import get_preferences

# Scene
HIDE_INNERWEAR = "pso2_hide_innerwear"
MUSCULARITY = "pso2_muscularity"

# Object
ALPHA_THRESHOLD = "pso2_alpha_threshold"
MESH_ID = "pso2_mesh_id"

# Bone
BONE_ID = "pso2_bone_id"


def add_custom_properties():
    _add_material_properties()
    _add_object_properties()
    _add_scene_properties()


def _add_scene_properties():
    preferences = get_preferences(bpy.context)

    setattr(
        bpy.types.Scene,
        HIDE_INNERWEAR,
        bpy.props.BoolProperty(name="Hide Innerwear", default=False),
    )

    setattr(
        bpy.types.Scene,
        MUSCULARITY,
        bpy.props.FloatProperty(
            name="Muscularity",
            min=0,
            max=1,
            default=preferences.default_muscularity,
            subtype="FACTOR",
        ),
    )

    for channel in COLOR_CHANNELS.values():
        name = channel.custom_property_name
        setattr(
            bpy.types.Scene,
            name,
            bpy.props.FloatVectorProperty(
                name=channel.name,
                subtype="COLOR",
                default=getattr(preferences, channel.prop),
                min=0,
                max=1,
                size=4,
            ),
        )


def _add_material_properties():
    setattr(
        bpy.types.Material,
        ALPHA_THRESHOLD,
        bpy.props.IntProperty(
            name="Alpha Threshold",
            min=0,
            max=255,
            default=0,
            subtype="FACTOR",
        ),
    )


def _add_object_properties():
    def _enum(mesh_id: parts.MeshId):
        name = parts.MESH_ID_NAMES[mesh_id]
        return (str(mesh_id), name, name, int(mesh_id))

    def _get_mesh_id(self: bpy.types.Object):
        return int(parts.get_mesh_id(self.name) or 0)

    def _set_mesh_id(self: bpy.types.Object, value: int):
        parts.set_mesh_id(self, parts.MeshId(value))

    setattr(
        bpy.types.Object,
        MESH_ID,
        bpy.props.EnumProperty(
            name="Mesh Part",
            items=[
                _enum(parts.MeshId.Costume),
                _enum(parts.MeshId.BreastNeck),
                _enum(parts.MeshId.Front),
                _enum(parts.MeshId.Back),
                _enum(parts.MeshId.Shoulder),
                _enum(parts.MeshId.Forearm),
                _enum(parts.MeshId.Legs),
                _enum(parts.MeshId.Ornament1),
                _enum(parts.MeshId.Ornament2),
                _enum(parts.MeshId.OuterOrnament),
                _enum(parts.MeshId.CastBodyOrnament),
                _enum(parts.MeshId.CastLegsOrnament),
                _enum(parts.MeshId.CastArmsOrnament),
                _enum(parts.MeshId.HeadOrnament),
            ],
            get=_get_mesh_id,
            set=_set_mesh_id,
        ),
    )
