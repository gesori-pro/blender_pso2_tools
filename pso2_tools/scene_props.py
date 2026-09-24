import bpy

from . import parts
from .colors import COLOR_CHANNELS
from .preferences import get_preferences

# Scene
HIDE_INNERWEAR = "pso2_hide_innerwear"
MUSCULARITY = "pso2_muscularity"

# Scene: the game's lighting (shaders/game_lighting.py). The materials read
# these through View Layer attributes; the sun's two come from drivers on
# the "PSO2 Sun" object that the Game Lighting setup makes.
GAME_SHADING = "pso2_game_shading"
LIGHT_DIRECTION = "pso2_light_direction"
LIGHT_STRENGTH = "pso2_light_strength"
EXPOSURE = "pso2_exposure"
ENVIRONMENT_COLOR = "pso2_environment_color"
HEADLIGHT = "pso2_headlight"

# World: the camera's zoom, for the hair's strand noise. It sits on the world
# because its driver reads the scene's resolution, which a driver on the
# scene itself cannot; View Layer attributes look there after the scene.
CAMERA_ZOOM = "pso2_camera_zoom"

# The character creator's sun (towards the light, Blender axes) and scene,
# from the constants of a captured frame.
CREATOR_LIGHT_DIRECTION = (0.29143327, -0.95006776, 0.11152586)
CREATOR_LIGHT_STRENGTH = 12.56
CREATOR_EXPOSURE = 0.84
CREATOR_ENVIRONMENT_COLOR = (0.0375, 0.075, 0.125)
CREATOR_CAMERA_ZOOM = 11.605
CREATOR_HEADLIGHT = 1.665

# Object
ALPHA_THRESHOLD = "pso2_alpha_threshold"
MESH_ID = "pso2_mesh_id"

# Armature: the primary and secondary bone axes the FBX importer was given,
# as "X,Y", or "AUTO" when it oriented bones from their children instead.
# Motion import needs them to undo the rotation the importer put on every
# bone's local axes.
BONE_AXES = "pso2_bone_axes"
DEFAULT_BONE_AXES = "X,Y"

# Bone
BONE_ID = "pso2_bone_id"


def add_custom_properties():
    _add_material_properties()
    _add_object_properties()
    _add_scene_properties()
    _add_world_properties()


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

    setattr(
        bpy.types.Scene,
        GAME_SHADING,
        bpy.props.FloatProperty(
            name="Game Shading",
            description=(
                "Light PSO2 materials the way the game does (EEVEE only)."
                " 0 keeps the Principled shading Cycles can render"
            ),
            min=0,
            max=1,
            default=0,
            subtype="FACTOR",
        ),
    )
    setattr(
        bpy.types.Scene,
        LIGHT_DIRECTION,
        bpy.props.FloatVectorProperty(
            name="Sun Direction",
            description="Towards the sun; driven by the PSO2 Sun object",
            size=3,
            subtype="DIRECTION",
            default=CREATOR_LIGHT_DIRECTION,
        ),
    )
    setattr(
        bpy.types.Scene,
        LIGHT_STRENGTH,
        bpy.props.FloatProperty(
            name="Sun Strength",
            description="The sun's strength; driven by the PSO2 Sun object",
            min=0,
            default=CREATOR_LIGHT_STRENGTH,
        ),
    )
    setattr(
        bpy.types.Scene,
        EXPOSURE,
        bpy.props.FloatProperty(
            name="Exposure",
            description=(
                "The game's tone-mapping exposure. Characters never see the sun"
                " dimmer than 1 / (0.7 x this), and get a fill light of 0.3 of that"
            ),
            min=0.01,
            default=CREATOR_EXPOSURE,
        ),
    )
    setattr(
        bpy.types.Scene,
        ENVIRONMENT_COLOR,
        bpy.props.FloatVectorProperty(
            name="Sky Tint",
            description="Added to the ambient from above",
            size=3,
            subtype="COLOR",
            min=0,
            default=CREATOR_ENVIRONMENT_COLOR,
        ),
    )
    setattr(
        bpy.types.Scene,
        HEADLIGHT,
        bpy.props.FloatProperty(
            name="Headlight",
            description=(
                "The character creator's light on the camera, which lights"
                " characters only. 0 turns it off"
            ),
            min=0,
            default=CREATOR_HEADLIGHT,
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


def _add_world_properties():
    setattr(
        bpy.types.World,
        CAMERA_ZOOM,
        bpy.props.FloatProperty(
            name="Camera Zoom",
            description=(
                "One over the tangent of half the camera's vertical field of"
                " view; hair picks its strand noise by how big it is on screen."
                " Driven by the scene camera"
            ),
            min=0.01,
            default=CREATOR_CAMERA_ZOOM,
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
