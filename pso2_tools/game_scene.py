"""Set a scene up to be lit the way the game lights characters (EEVEE).

The materials carry the game's lighting beside their Principled BSDF
(shaders/game_lighting.py); this makes what that lighting reads: a sun whose
direction and strength drive the scene's lighting properties, an environment
image for the ambient, a world that shows a backdrop but lights nothing (the
ambient is in the materials, and the sun's shadows are read back from what
EEVEE's lights do to a plain diffuse surface), and the game's glare and tone
curve in the compositor. The defaults are the character creator's.
"""

import math

import bpy
from mathutils import Vector

from . import classes, scene_props
from .shaders.game_lighting import ENVIRONMENT_IMAGE
from .util import OperatorResult

SUN_NAME = "PSO2 Sun"
WORLD_NAME = "PSO2 Game World"
TONE_MAP_NAME = "PSO2 Tone Map"

# The character creator's backdrop, as bright before tone mapping as its
# light grey after it.
BACKDROP = 1.13

# The game's glare: the image blends 5% towards a blur of itself, made from
# a pyramid of three levels. From the captured frame, as one Gaussian: a
# sigma of 0.178 of the frame's height, at 0.72 of the brightness.
GLARE_RATE = 0.05
GLARE_GAIN = 0.72
GLARE_SIGMA = 0.178

# What a driver reads for Camera.sensor_fit
SENSOR_FIT_HORIZONTAL = 1
SENSOR_FIT_VERTICAL = 2


def _eevee_engine() -> str:
    engines = [
        e.identifier
        for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    return "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"


def _sun(context) -> bpy.types.Object:
    sun = bpy.data.objects.get(SUN_NAME)
    if sun is None or sun.type != "LIGHT":
        light = bpy.data.lights.new(SUN_NAME, "SUN")
        sun = bpy.data.objects.new(SUN_NAME, light)
        context.scene.collection.objects.link(sun)
        towards = Vector(scene_props.CREATOR_LIGHT_DIRECTION).normalized()
        # a sun shines down its -Z, so +Z points at it
        sun.rotation_euler = towards.to_track_quat("Z", "Y").to_euler()
        light.energy = scene_props.CREATOR_LIGHT_STRENGTH
        light.color = (1, 1, 1)
        # The game filters its shadow map; on a character that softens the
        # edges about as much as a sun five degrees across.
        light.angle = math.radians(5)
    return sun


def _drive(scene: bpy.types.Scene, sun: bpy.types.Object) -> None:
    """Scene lighting properties follow the sun object, with no Python."""
    scene.driver_remove(scene_props.LIGHT_DIRECTION)
    for index in range(3):
        driver = scene.driver_add(scene_props.LIGHT_DIRECTION, index).driver
        driver.type = "AVERAGE"
        var = driver.variables.new()
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "OBJECT"
        var.targets[0].id = sun
        # A driver reads matrix_world[a][b] as column a, row b: this is the
        # sun's +Z axis, which points at it.
        var.targets[0].data_path = f"matrix_world[2][{index}]"

    scene.driver_remove(scene_props.LIGHT_STRENGTH)
    driver = scene.driver_add(scene_props.LIGHT_STRENGTH).driver
    driver.type = "AVERAGE"
    var = driver.variables.new()
    var.type = "SINGLE_PROP"
    var.targets[0].id_type = "OBJECT"
    var.targets[0].id = sun
    var.targets[0].data_path = "data.energy"


def _drive_zoom(scene: bpy.types.Scene, world: bpy.types.World) -> None:
    """Hair noise is picked by how big it is on screen: the world's zoom
    follows the scene camera's vertical field of view."""
    world.driver_remove(scene_props.CAMERA_ZOOM)
    driver = world.driver_add(scene_props.CAMERA_ZOOM).driver
    driver.type = "SCRIPTED"
    for name, path in (
        ("lens", "camera.data.lens"),
        ("width", "camera.data.sensor_width"),
        ("height", "camera.data.sensor_height"),
        ("fit", "camera.data.sensor_fit"),
        ("x", "render.resolution_x"),
        ("y", "render.resolution_y"),
    ):
        var = driver.variables.new()
        var.name = name
        var.type = "SINGLE_PROP"
        var.targets[0].id_type = "SCENE"
        var.targets[0].id = scene
        var.targets[0].data_path = path
    # A simple expression, so it runs without Python: the sensor's height
    # when fitted vertically, else its width scaled to the frame's height.
    # With no camera it reads zeros, and must not fail on them: a driver
    # that errors once stays switched off.
    sensor = (
        f"(height if fit == {SENSOR_FIT_VERTICAL}"
        f" else width * (y / max(x, 1) if fit == {SENSOR_FIT_HORIZONTAL} or x >= y"
        " else 1))"
    )
    driver.expression = (
        f"2 * lens / max({sensor}, 0.001) if lens > 0"
        f" else {scene_props.CREATOR_CAMERA_ZOOM}"
    )


def _environment(path: str) -> bpy.types.Image:
    image = None
    if path:
        image = bpy.data.images.load(bpy.path.abspath(path), check_existing=True)
    if image is None:
        image = bpy.data.images.get(ENVIRONMENT_IMAGE)
    if image is None:
        # a flat neutral grey until a real environment is given
        image = bpy.data.images.new(ENVIRONMENT_IMAGE, 2, 1, float_buffer=True)
        image.pixels[:] = (0.35, 0.33, 0.31, 1.0) * 2
        image.pack()
    image.name = ENVIRONMENT_IMAGE
    image.colorspace_settings.name = "Linear Rec.709" if image.is_float else "sRGB"
    for tree in bpy.data.node_groups:
        for node in tree.nodes:
            if node.type == "TEX_ENVIRONMENT" and node.name.startswith("Environment"):
                node.image = image
    return image


def _world(scene: bpy.types.Scene) -> bpy.types.World:
    world = bpy.data.worlds.get(WORLD_NAME) or bpy.data.worlds.new(WORLD_NAME)
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    backdrop = nodes.new("ShaderNodeBackground")
    backdrop.inputs["Color"].default_value = (BACKDROP, BACKDROP, BACKDROP, 1)
    dark = nodes.new("ShaderNodeBackground")
    dark.inputs["Color"].default_value = (0, 0, 0, 1)
    path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    out = nodes.new("ShaderNodeOutputWorld")
    # Seen by the camera it is the backdrop; it lights nothing, so what
    # EEVEE reports for a diffuse surface is the sun alone.
    links.new(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(dark.outputs["Background"], mix.inputs[1])
    links.new(backdrop.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], out.inputs["Surface"])
    scene.world = world
    return world


def _tone_map(scene: bpy.types.Scene) -> None:
    """The game's glare, then its curve: y = x(3x + 0.03) / (x(3x + 1) +
    0.14) on x = exposure * colour, then gamma 1/2.2. The view transform is
    left at Standard, so the output is what that encodes to y^(1/2.2). The
    glare's size is set from the render resolution."""
    modern = hasattr(scene, "compositing_node_group")
    if modern:
        tree = bpy.data.node_groups.get(TONE_MAP_NAME)
        if tree is None or tree.bl_idname != "CompositorNodeTree":
            tree = bpy.data.node_groups.new(TONE_MAP_NAME, "CompositorNodeTree")
        tree.nodes.clear()
        if not any(item.in_out == "OUTPUT" for item in tree.interface.items_tree):
            tree.interface.new_socket(
                "Image", in_out="OUTPUT", socket_type="NodeSocketColor"
            )
        scene.compositing_node_group = tree
        math_type = "ShaderNodeMath"
    else:
        scene.use_nodes = True
        tree = scene.node_tree
        tree.nodes.clear()
        math_type = "CompositorNodeMath"

    nodes, links = tree.nodes, tree.links
    layers = nodes.new("CompositorNodeRLayers")
    split = nodes.new("CompositorNodeSeparateColor")
    join = nodes.new("CompositorNodeCombineColor")
    links.new(layers.outputs["Image"], split.inputs["Image"])

    # A Gaussian blur's size is three sigmas.
    height = scene.render.resolution_y * scene.render.resolution_percentage / 100
    radius = max(1, round(3 * GLARE_SIGMA * height))
    blur = nodes.new("CompositorNodeBlur")
    if "Size" in blur.inputs and blur.inputs["Size"].type == "VECTOR":
        blur.inputs["Size"].default_value = (radius, radius)  # type: ignore
    else:
        blur.filter_type = "GAUSS"  # type: ignore
        blur.size_x = blur.size_y = radius  # type: ignore
    blurred = nodes.new("CompositorNodeSeparateColor")
    links.new(layers.outputs["Image"], blur.inputs["Image"])
    links.new(blur.outputs["Image"], blurred.inputs["Image"])

    def math(op, a, b=None, c=None):
        node = nodes.new(math_type)
        node.operation = op
        for index, operand in enumerate((a, b, c)):
            if operand is None:
                continue
            if isinstance(operand, (int, float)):
                node.inputs[index].default_value = operand
            else:
                links.new(operand, node.inputs[index])
        return node.outputs[0]

    exposure = getattr(scene, scene_props.EXPOSURE, scene_props.CREATOR_EXPOSURE)
    for channel in ("Red", "Green", "Blue"):
        glare = math(
            "MULTIPLY_ADD",
            split.outputs[channel],
            1 - GLARE_RATE,
            math("MULTIPLY", blurred.outputs[channel], GLARE_RATE * GLARE_GAIN),
        )
        x = math("MULTIPLY", glare, float(exposure))
        x = math("MAXIMUM", x, 0.0)
        top = math("MULTIPLY", x, math("MULTIPLY_ADD", x, 3.0, 0.03))
        bottom = math("MULTIPLY_ADD", x, math("MULTIPLY_ADD", x, 3.0, 1.0), 0.14)
        y = math("MINIMUM", math("DIVIDE", top, bottom), 1.0)
        shown = math("POWER", y, 1 / 2.2)
        # what the sRGB view transform turns back into `shown`
        linear = math(
            "POWER", math("MULTIPLY_ADD", shown, 1 / 1.055, 0.055 / 1.055), 2.4
        )
        links.new(linear, join.inputs[channel])
    links.new(split.outputs["Alpha"], join.inputs["Alpha"])

    if modern:
        out = nodes.new("NodeGroupOutput")
        links.new(join.outputs["Image"], out.inputs[0])
    else:
        out = nodes.new("CompositorNodeComposite")
        links.new(join.outputs["Image"], out.inputs["Image"])
    viewer = nodes.new("CompositorNodeViewer")
    links.new(join.outputs["Image"], viewer.inputs["Image"])


@classes.register
class PSO2_OT_SetupGameLighting(bpy.types.Operator):
    """Light this scene's PSO2 characters the way the game does (EEVEE)"""

    bl_idname = "pso2.setup_game_lighting"
    bl_label = "Set Up Game Lighting"
    bl_options = {"REGISTER", "UNDO"}

    environment: bpy.props.StringProperty(
        name="Environment",
        description=(
            "An equirectangular image of the lighting environment for the"
            " ambient light. Leave empty for a neutral grey"
        ),
        subtype="FILE_PATH",
        default="",
    )
    tone_map: bpy.props.BoolProperty(
        name="Game Tone Curve",
        description="Replace the compositor with the game's tone curve",
        default=True,
    )

    def execute(self, context) -> OperatorResult:
        scene = context.scene
        sun = _sun(context)
        _drive(scene, sun)
        setattr(scene, scene_props.GAME_SHADING, 1.0)
        _environment(self.environment)
        _drive_zoom(scene, _world(scene))

        scene.render.engine = _eevee_engine()
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0
        if self.tone_map:
            _tone_map(scene)
        if scene.camera is None:
            self.report(
                {"WARNING"},
                "No scene camera: hair noise assumes the character creator's"
                " camera until this is run again with one",
            )
        else:
            self.report({"INFO"}, f"Game lighting set up with {sun.name}")
        return {"FINISHED"}
