import bpy

from .. import classes, scene_props
from ..colors import ColorId, ColorMapping
from . import builder, group
from .colorize import ShaderNodePso2Colorize
from .colors import ShaderNodePso2Colorchannels
from .game_lighting import Expr, ShaderNodePso2GameLight, Val
from .ngs import GAME_ALBEDO

# The mapping the character importer sets to the eye and iris size sliders.
IRIS_SIZE = "Iris Size"
IRIS_SIZE_UV = "Iris Size UV"

# What the game's eye G-buffer shader (1104g) does, from the compiled shader
# and a captured frame. The diffuse's alpha is a height map: 1 on the white
# of the eye, down to about half across the iris, which sits in a dish
# half a UV unit deep. The game marches the view through it (sixteen steps,
# then sixteen more between the last two) and reads every texture where the
# ray lands, takes the normal from the height map's slope, and darkens the
# colour by the height.
EYE_UV_MAP = "UVChannel_1"
EYE_DEPTH = 0.5
EYE_STEPS = 16
EYE_ALBEDO_SCALE = 0.61
# the height map's slope is read one texel of a 256 texture away
EYE_SLOPE_TEXEL = 1 / 256
EYE_SLOPE_Z = 0.005


class Shader1104(builder.ShaderBuilder):
    """NGS eye shader"""

    @property
    def textures(self):
        return self.data.textures

    @property
    def colors(self) -> ColorMapping:
        return self.data.color_map or ColorMapping()

    def build(self, context):
        tree = self.init_tree()

        output = tree.add_node(bpy.types.ShaderNodeOutputMaterial, (40, 6))

        shader_group = tree.add_node(ShaderNodePso2NgsEye, (34, 6))
        tree.add_link(shader_group.outputs["BSDF"], output.inputs["Surface"])

        # The UVs every texture is read at, scaled about the iris by the
        # eye and iris size sliders (the character importer sets the scale).
        uv = tree.add_node(bpy.types.ShaderNodeUVMap, (-60, 0), name=IRIS_SIZE_UV)
        uv.uv_map = EYE_UV_MAP
        mapping = tree.add_node(bpy.types.ShaderNodeMapping, (-56, 0), name=IRIS_SIZE)
        tree.add_link(uv.outputs["UV"], mapping.inputs["Vector"])

        # Diffuse
        diffuse = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 18), name="Diffuse")
        diffuse.image = self.textures.default.diffuse
        diffuse.extension = "EXTEND"

        landing, slope_u, slope_v = _parallax(
            Expr(tree, -50, 40), mapping.outputs["Vector"], diffuse.image
        )
        tree.add_link(landing.socket, diffuse.inputs["Vector"])
        tree.add_link(diffuse.outputs["Alpha"], shader_group.inputs["Height"])
        tree.add_link(slope_u, shader_group.inputs["Height U"])
        tree.add_link(slope_v, shader_group.inputs["Height V"])

        # Color Mask
        mask = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 12), name="Color Mask")
        mask.image = self.textures.default.mask
        mask.extension = "EXTEND"
        tree.add_link(landing.socket, mask.inputs["Vector"])

        colorize = tree.add_node(ShaderNodePso2Colorize, (12, 14))

        tree.add_link(diffuse.outputs["Color"], colorize.inputs["Input"])
        tree.add_link(colorize.outputs["Result"], shader_group.inputs["Diffuse"])

        tree.add_link(mask.outputs["Color"], colorize.inputs["Mask RGB"])
        tree.add_link(mask.outputs["Alpha"], colorize.inputs["Mask A"])

        channels = tree.add_node(ShaderNodePso2Colorchannels, (7, 10), name="Colors")

        color = ColorId.LEFT_EYE if "eye_l" in self.material.name else ColorId.RIGHT_EYE
        tree.add_color_link(color, channels, colorize.inputs["Color 1"])
        colorize.set_colors_used([1])

        # Multi Map
        multi = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 6), name="Multi Map")
        multi.image = self.textures.default.multi
        multi.extension = "EXTEND"
        tree.add_link(landing.socket, multi.inputs["Vector"])

        tree.add_link(multi.outputs["Color"], shader_group.inputs["Multi RGB"])
        tree.add_link(multi.outputs["Alpha"], shader_group.inputs["Multi A"])

        # Normal Map: only its alpha is read, where the ray started
        normal = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 0), name="Normal Map")
        normal.image = self.textures.default.normal
        tree.add_link(mapping.outputs["Vector"], normal.inputs["Vector"])

        tree.add_link(normal.outputs["Alpha"], shader_group.inputs["Normal A"])


def _parallax(
    e: Expr, uv: bpy.types.NodeSocket, image: bpy.types.Image | None
) -> tuple[Val, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Where the view lands in the eye's height map, and the height one
    texel along u and along v from there.

    The height samples are Diffuse texture nodes, so painting a new eye part
    onto the material updates them with the rest.
    """
    start = e.v(uv)

    def height(at: Val) -> bpy.types.NodeSocket:
        node = e._node(bpy.types.ShaderNodeTexImage, name="Diffuse")
        node.image = image
        node.extension = "EXTEND"
        e.tree.add_link(at.socket, node.inputs["Vector"])
        return node.outputs["Alpha"]

    # The view in the eye's tangent space: along u, along v, and into the
    # surface. The game's v runs down the texture, Blender's up.
    geometry = e._node(bpy.types.ShaderNodeNewGeometry)
    normal = e.v(geometry.outputs["Normal"])
    eye = e.mul(e.v(geometry.outputs["Incoming"]), -1.0)
    tangent_node = e._node(bpy.types.ShaderNodeTangent)
    tangent_node.direction_type = "UV_MAP"
    tangent_node.uv_map = EYE_UV_MAP
    tangent = e.normalize(e.v(tangent_node.outputs["Tangent"]))
    bitangent = e.normalize(e._vmath("CROSS_PRODUCT", normal, tangent))
    view = e.normalize(
        e.xyz(e.dot(eye, tangent), e.dot(eye, bitangent), e.dot(eye, normal))
    )
    vx, vy, vz = e.split(view)
    step = e.xyz(vx, vy, 0.0)
    sink = e.mul(vz, -1.0)

    def missed(t) -> Val:
        """1 where the ray, t along, is still above the surface."""
        surface = e.mul(
            e.sub(1.0, e.f(height(e.add(start, e.mul(step, t))))), EYE_DEPTH
        )
        return e.gt(surface, e.mul(sink, t))

    # Sixteenths until the ray is under the surface (the game tries up to
    # 14/16, then gives up on the sixteenth after).
    last = (EYE_STEPS - 1) / EYE_STEPS
    hit = None
    for k in range(1, EYE_STEPS - 1):
        t = k / EYE_STEPS
        candidate = e.add(missed(t), t)  # t where it hit, above 1 where not
        hit = candidate if hit is None else e.min(hit, candidate)
    hit = e.min(hit, last)
    below = e.sub(hit, 1 / EYE_STEPS)

    # Then the first of sixteen steps between that and the one before where
    # it is under (or the fifteenth); the game walks them, a halving search
    # finds the same step in four.
    fine = 1 / EYE_STEPS**2
    count = e.const(0.0)
    for size in (8, 4, 2, 1):
        t = e.add(below, e.mul(e.add(count, size - 1), fine))
        count = e.add(count, e.mul(missed(t), float(size)))
    t = e.add(below, e.mul(e.min(count, EYE_STEPS - 1.0), fine))

    # Where the white of the eye starts flat, the game does not march at all.
    flat = e.gt(e.f(height(start)), 0.9999)
    t = e.mul(t, e.sub(1.0, flat))

    landing = e.add(start, e.mul(step, t))
    slope_u = height(e.add(landing, (EYE_SLOPE_TEXEL, 0.0, 0.0)))
    slope_v = height(e.add(landing, (0.0, -EYE_SLOPE_TEXEL, 0.0)))
    return landing, slope_u, slope_v


def _to_game(e: Expr, v: Val) -> Val:
    """A Blender world vector in the game's axes (Y up, the character
    facing +Z)."""
    x, y, z = e.split(v)
    return e.xyz(x, z, e.mul(y, -1.0))


@classes.register
class ShaderNodePso2NgsEye(group.ShaderNodeCustomGroup):
    bl_name = "ShaderNodePso2Eye"
    bl_label = "PSO2 Eye"
    bl_icon = "NONE"

    # 2: the game's eye G-buffer and lighting beside the Principled BSDF
    tree_version = 2

    def init(self, context):
        super().init(context)

        self.input(bpy.types.NodeSocketColor, "Diffuse").default_value = (1, 0, 1, 1)  # type: ignore
        self.input(bpy.types.NodeSocketFloat, "Height").default_value = 1
        self.input(bpy.types.NodeSocketFloat, "Height U").default_value = 1
        self.input(bpy.types.NodeSocketFloat, "Height V").default_value = 1
        self.input(bpy.types.NodeSocketColor, "Multi RGB").default_value = (0, 1, 1, 1)  # type: ignore

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput, (-40, 0))
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput, (40, 0))

        tree.new_input(bpy.types.NodeSocketColor, "Diffuse")
        tree.new_input(bpy.types.NodeSocketFloat, "Height")
        tree.new_input(bpy.types.NodeSocketFloat, "Height U")
        tree.new_input(bpy.types.NodeSocketFloat, "Height V")
        tree.new_input(bpy.types.NodeSocketColor, "Multi RGB")
        tree.new_input(bpy.types.NodeSocketFloat, "Multi A")
        tree.new_input(bpy.types.NodeSocketFloat, "Normal A")

        tree.new_output(bpy.types.NodeSocketShader, "BSDF")

        e = Expr(tree, -34, 30)
        geometry = e._node(bpy.types.ShaderNodeNewGeometry)
        view = e.v(geometry.outputs["Incoming"])
        normal = e.v(geometry.outputs["Normal"])
        tangent_node = e._node(bpy.types.ShaderNodeTangent)
        tangent_node.direction_type = "UV_MAP"
        tangent_node.uv_map = EYE_UV_MAP
        tangent = e.normalize(e.v(tangent_node.outputs["Tangent"]))
        bitangent = e.normalize(e._vmath("CROSS_PRODUCT", normal, tangent))

        diffuse = e.v(group_inputs.outputs["Diffuse"])
        height = e.f(group_inputs.outputs["Height"])
        metal_root, _, occlusion = e.split(e.v(group_inputs.outputs["Multi RGB"]))
        gate = e.sat(e.sub(e.f(group_inputs.outputs["Multi A"]), 0.02))
        normal_a = e.min(e.mul(e.f(group_inputs.outputs["Normal A"]), 1.02), 1.0)

        # ---- G-buffer
        albedo = e.mul(diffuse, e.mul(height, EYE_ALBEDO_SCALE))
        soft = e.mul(e.min(e.mul(normal_a, 2.0), 1.0), 0.25)
        facing = e.abs(e.sub(1.0, e.dot(normal, view)))
        rim = e.min(
            e.mul(
                e.mul(e.mul(facing, facing), facing),
                e.max(e.madd(normal_a, 2.0, -1.0), 0.0),
            ),
            1.0,
        )

        # Glowing parts: 50 x the albedo squared, through the game's
        # brightness boost (x1.5, ^1.3), which undoes the exposure. They
        # are not soft, and carry the glow where the soft colour would be.
        glow = e.add(e.mul(e.mul(albedo, albedo), 0.995), (0.005, 0.005, 0.005))
        glow = e.mul(glow, e.mul(e.mul(gate, gate), 75.0))
        gx, gy, gz = e.split(glow)
        glow = e.div(
            e.xyz(e.pow(gx, 1.3), e.pow(gy, 1.3), e.pow(gz, 1.3)),
            e.max(e.attribute(scene_props.EXPOSURE), 1e-3),
        )
        glowing = e.gt(e.add(e.add(gx, gy), gz), 1e-6)
        soft = e.mul(soft, e.sub(1.0, glowing))
        scatter = e.lerp(e.mul(albedo, e.gt(soft, 1 / 255 - 1e-6)), glow, glowing)

        # The normal is the height map's slope. The game turns it into the
        # world with the tangent frame transposed, in its own axes; eyes
        # face it square on, where that comes out right.
        slope = e.normalize(
            e.xyz(
                e.sub(e.f(group_inputs.outputs["Height U"]), height),
                e.sub(height, e.f(group_inputs.outputs["Height V"])),
                EYE_SLOPE_Z,
            )
        )
        turned = e.xyz(
            e.mul(e.dot(slope, _to_game(e, tangent)), -1.0),
            e.mul(e.dot(slope, _to_game(e, bitangent)), -1.0),
            e.dot(slope, _to_game(e, normal)),
        )
        tx, ty, tz = e.split(turned)
        surface_normal = e.normalize(e.xyz(tx, e.mul(tz, -1.0), ty))

        gbuffer_albedo = e.mul(albedo, e.sub(1.0, gate))

        # ---- Principled, for Cycles
        bsdf = e._node(bpy.types.ShaderNodeBsdfPrincipled, name="Principled BSDF")
        game_albedo = e._node(bpy.types.ShaderNodeValue, name=GAME_ALBEDO)
        game_albedo.outputs[0].default_value = 1  # type: ignore
        base = e.lerp(diffuse, gbuffer_albedo, e.f(game_albedo.outputs[0]))
        tree.add_link(base.socket, bsdf.inputs["Base Color"])
        tree.add_link(e.mul(metal_root, metal_root).socket, bsdf.inputs["Metallic"])
        bsdf.inputs["Roughness"].default_value = 1  # type: ignore
        tree.add_link(surface_normal.socket, bsdf.inputs["Normal"])
        tree.add_link(glow.socket, bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1  # type: ignore

        # ---- the game's lighting, for EEVEE: model 0, fully rough. Like
        # the face, the eyes take no shadow in the character creator.
        light = e._node(ShaderNodePso2GameLight)
        for name, value in (
            ("Albedo", gbuffer_albedo),
            ("Soft", soft),
            ("Metal Root", metal_root),
            ("AO", occlusion),
            ("Scatter", scatter),
            ("Rim", e.mul(rim, soft)),
            ("Normal", surface_normal),
        ):
            tree.add_link(value.socket, light.inputs[name])
        light.inputs["Roughness"].default_value = 1  # type: ignore
        light.inputs["Skin"].default_value = 0  # type: ignore
        light.inputs["Shadow"].default_value = 1  # type: ignore

        lit = e._node(bpy.types.ShaderNodeEmission, name="Game Light")
        tree.add_link(light.outputs["Color"], lit.inputs["Color"])

        choose = e._node(bpy.types.ShaderNodeMixShader, name="Game Shading")
        tree.add_link(
            e.attribute(scene_props.GAME_SHADING).socket, choose.inputs["Fac"]
        )
        tree.add_link(bsdf.outputs["BSDF"], choose.inputs[1])
        tree.add_link(lit.outputs["Emission"], choose.inputs[2])
        tree.add_link(choose.outputs["Shader"], group_outputs.inputs["BSDF"])
