import bpy

from .. import classes, scene_props
from ..colors import ColorId, ColorMapping
from . import builder, group
from .game_lighting import Expr, ShaderNodePso2GameHairLight, ShaderNodePso2GameShadow
from .ngs import GAME_ALBEDO
from .colorize import ShaderNodePso2Colorize
from .colors import ShaderNodePso2Colorchannels


class Shader1103(builder.ShaderBuilder):
    """NGS hair shader"""

    @property
    def textures(self):
        return self.data.textures

    @property
    def colors(self) -> ColorMapping:
        return self.data.color_map or ColorMapping()

    def build(self, context):
        tree = self.init_tree()

        output = tree.add_node(bpy.types.ShaderNodeOutputMaterial, (24, 6))

        shader_group = tree.add_node(ShaderNodePso2NgsHair, (18, 6))
        tree.add_link(shader_group.outputs["BSDF"], output.inputs["Surface"])

        # Non-alpha Texture UVs
        uv = tree.add_node(bpy.types.ShaderNodeUVMap, (-8, 0), name="UVs")
        uv.uv_map = "UVChannel_2"

        # Diffuse
        diffuse = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 18), name="Diffuse")
        diffuse.image = self.textures.default.diffuse

        tree.add_link(uv.outputs[0], diffuse.inputs["Vector"])

        # Color Mask
        mask = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 12), name="Color Mask")
        mask.image = self.textures.default.mask

        tree.add_link(uv.outputs[0], mask.inputs["Vector"])

        colorize = tree.add_node(ShaderNodePso2Colorize, (12, 14))

        tree.add_link(diffuse.outputs["Color"], colorize.inputs["Input"])
        tree.add_link(colorize.outputs["Result"], shader_group.inputs["Diffuse"])

        tree.add_link(mask.outputs["Color"], colorize.inputs["Mask RGB"])
        if self.colors.alpha != ColorId.UNUSED:
            tree.add_link(mask.outputs["Alpha"], colorize.inputs["Mask A"])

        channels = tree.add_node(ShaderNodePso2Colorchannels, (7, 10), name="Colors")

        tree.add_color_link(self.colors.red, channels, colorize.inputs["Color 1"])
        tree.add_color_link(self.colors.green, channels, colorize.inputs["Color 2"])
        tree.add_color_link(self.colors.blue, channels, colorize.inputs["Color 3"])
        tree.add_color_link(self.colors.alpha, channels, colorize.inputs["Color 4"])
        colorize.set_colors_used(self.colors)

        # Alpha
        alpha = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 6), name="Alpha")
        alpha.image = self.textures.default.alpha

        tree.add_link(alpha.outputs["Color"], shader_group.inputs["Alpha"])

        # Multi Map
        multi = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 0), name="Multi Map")
        multi.image = self.textures.default.multi

        tree.add_link(uv.outputs[0], multi.inputs["Vector"])
        tree.add_link(multi.outputs["Color"], shader_group.inputs["Multi RGB"])
        tree.add_link(multi.outputs["Alpha"], shader_group.inputs["Multi A"])

        # Normal Map
        normal = tree.add_node(bpy.types.ShaderNodeTexImage, (0, -6), name="Normal Map")
        normal.image = self.textures.default.normal

        tree.add_link(uv.outputs[0], normal.inputs["Vector"])
        tree.add_link(normal.outputs["Color"], shader_group.inputs["Normal"])


# What the game's hair G-buffer shader (1103g) does, from the compiled shader
# and the constants of a captured frame (u_ColorNoise 0.2, u_AONoise 1,
# u_Block 0.5, u_MaskNoise 0, u_HairWidth 1; the vertex colours it reads are
# black with alpha 1 on NGS hair).
HAIR_COLOR_NOISE = 0.2
HAIR_AO_NOISE = 1.0
HAIR_ALBEDO_SCALE = 0.61


@classes.register
class ShaderNodePso2NgsHair(group.ShaderNodeCustomGroup):
    bl_name = "ShaderNodePso2NgsHair"
    bl_label = "PSO2 NGS Hair"
    bl_icon = "NONE"

    # 2: the game's hair G-buffer and lighting beside the Principled BSDF
    # 3: the game's strand noise: checkered, filtered, mip from the camera
    # 4: the Principled BSDF's highlight as wide as the game's
    tree_version = 4

    def init(self, context):
        super().init(context)

        self.input(bpy.types.NodeSocketColor, "Diffuse").default_value = (1, 0, 1, 1)  # type: ignore
        self.input(bpy.types.NodeSocketFloat, "Alpha").default_value = 1
        self.input(bpy.types.NodeSocketColor, "Multi RGB").default_value = (0, 1, 1, 1)  # type: ignore

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput, (-40, 0))
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput, (40, 0))

        tree.new_input(bpy.types.NodeSocketColor, "Diffuse")
        tree.new_input(bpy.types.NodeSocketFloat, "Alpha")
        tree.new_input(bpy.types.NodeSocketColor, "Multi RGB")
        tree.new_input(bpy.types.NodeSocketFloat, "Multi A")
        tree.new_input(bpy.types.NodeSocketColor, "Normal")

        tree.new_output(bpy.types.NodeSocketShader, "BSDF")

        e = Expr(tree, -34, 30)
        geometry = e._node(bpy.types.ShaderNodeNewGeometry)
        view = e.v(geometry.outputs["Incoming"])
        vertex_normal = e.v(geometry.outputs["Normal"])

        # The strand runs along the texture's v: the binormal of its UVs.
        tangent_node = e._node(bpy.types.ShaderNodeTangent)
        tangent_node.direction_type = "UV_MAP"
        tangent_node.uv_map = "UVChannel_2"
        across = e.normalize(e.v(tangent_node.outputs["Tangent"]))
        along = e.normalize(e._vmath("CROSS_PRODUCT", vertex_normal, across))

        normal_map = e._node(bpy.types.ShaderNodeNormalMap)
        normal_map.uv_map = "UVChannel_2"
        tree.add_link(group_inputs.outputs["Normal"], normal_map.inputs["Color"])
        mapped_normal = e.v(normal_map.outputs["Normal"])

        _, green, blue = e.split(e.v(group_inputs.outputs["Multi RGB"]))

        # ---- strand noise, on the alpha UVs: thin across, long along.
        # The game reads it from a 256x256 texture the engine makes, whose
        # every mip is a checkerboard of its own with random amplitudes
        # (0.5 -/+ up to 0.5), so neighbouring strands go light and dark in
        # turn. It filters bilinearly, at a mip picked from how big the hair
        # is on screen and dithered between mips.
        uv = e._node(bpy.types.ShaderNodeUVMap)
        uv.uv_map = "UVChannel_1"
        u, v, _ = e.split(e.v(uv.outputs["UV"]))

        def texel(x, y, mip):
            """The noise texel at whole coordinates x, y of a mip."""
            if isinstance(mip, (int, float)):
                size = 256 / 2**mip
            else:
                size = e.div(256.0, e.exp2(mip))
            x = e._math("FLOORED_MODULO", x, size)
            y = e._math("FLOORED_MODULO", y, size)
            node = e._node(bpy.types.ShaderNodeTexWhiteNoise)
            node.noise_dimensions = "3D"
            tree.add_link(e.xyz(x, y, mip).socket, node.inputs["Vector"])
            odd = e._math("FLOORED_MODULO", e.add(x, y), 2.0)
            # 0.5 - (-1)^(x+y) * amplitude / 2
            return e.madd(
                e.mul(e.f(node.outputs["Value"]), 0.5), e.madd(odd, 2.0, -1.0), 0.5
            )

        def filtered(x, y, mip, step=None):
            """A bilinear sample at texel coordinates x, y; with `step`, the
            average of four of them `step` texels apart across the strand."""
            px = e.sub(x, 0.5)
            py = e.sub(y, 0.5)
            ix = e._math("FLOOR", px)
            iy = e._math("FLOOR", py)
            fx = e.sub(px, ix)
            fy = e.sub(py, iy)

            def column(offset):
                cx = e.add(ix, offset) if offset else ix
                return e.lerp(texel(cx, iy, mip), texel(cx, e.add(iy, 1.0), mip), fy)

            if step is None:
                return e.lerp(column(0), column(1), fx)
            # Each of the four samples blends two of three columns: add up
            # how much of each the four take.
            first = e.sub(1.0, fx)
            last = None
            for k in range(1, 4):
                reach = e.mul(step, float(k))
                first = e.add(first, e.max(e.sub(e.sub(1.0, fx), reach), 0.0))
                over = e.max(e.sub(e.add(fx, reach), 1.0), 0.0)
                last = over if last is None else e.add(last, over)
            first = e.mul(first, 0.25)
            last = e.mul(last, 0.25)
            middle = e.sub(e.sub(1.0, first), last)
            return e.add(
                e.add(e.mul(column(0), first), e.mul(column(1), middle)),
                e.mul(column(2), last),
            )

        def contrast(n, lod):
            if isinstance(lod, (int, float)):
                gain, power = 1.5 - lod / 12, 1 + lod / 12
            else:
                gain, power = e.madd(lod, -1 / 12, 1.5), e.madd(lod, 1 / 12, 1.0)
            n = e.sat(e.madd(e.sub(n, 0.5), gain, 0.5))
            return e.pow(e.max(n, 1e-6), power)

        # The mip: log2 of 16 half-heights of the view at the hair's depth.
        camera = e._node(bpy.types.ShaderNodeCameraData)
        depth = e.f(camera.outputs["View Z Depth"])
        # (the zoom is on the world; with no world, the creator's camera)
        zoom = e.attribute(scene_props.CAMERA_ZOOM)
        zoom = e.add(
            zoom,
            e.mul(e._math("LESS_THAN", zoom, 0.01), scene_props.CREATOR_CAMERA_ZOOM),
        )
        half_height = e.div(depth, zoom)
        coords = e._node(bpy.types.ShaderNodeTexCoord)
        pixel = e._vmath(
            "FLOOR", e.mul(e.v(coords.outputs["Window"]), (8192.0, 8192.0, 0.0))
        )
        dither_node = e._node(bpy.types.ShaderNodeTexWhiteNoise)
        dither_node.noise_dimensions = "3D"
        tree.add_link(
            e.add(pixel, (0.0, 0.0, 0.5)).socket, dither_node.inputs["Vector"]
        )
        # the game's 4x4 ordered dither; a fine random threshold that EEVEE's
        # samples average does the same
        dither = e.f(dither_node.outputs["Value"])
        lod = e._math(
            "FLOOR", e.add(e.log2(e.max(e.mul(half_height, 16.0), 1e-6)), dither)
        )
        mip = e.min(e.max(lod, 0.0), 8.0)
        below = e.min(e.max(e.sub(lod, 1.0), 0.0), 6.0)
        above = e.min(e.max(e.add(lod, 1.0), 0.0), 6.0)
        texels_u = e.mul(u, 2048.0)
        texels_v = e.mul(v, 8.0)
        # hair nearer than the finest mip stretches it along the strand
        near = e.exp2(e.sub(lod, mip))

        def at(level):
            return e.div(texels_u, e.exp2(level))

        main = filtered(at(mip), e.mul(texels_v, near), mip, e.mul(near, 0.25))
        noise = e.add(
            e.mul(contrast(main, lod), 3.0),
            e.add(
                contrast(filtered(at(below), texels_v, below), below),
                contrast(filtered(at(above), texels_v, above), above),
            ),
        )
        noise = e.sat(e.madd(e.madd(noise, 0.2, -0.5), 1.3, 0.5))
        # coarse blocks that tilt whole locks: mip 3, 125 across and 2 along
        block = filtered(e.mul(u, 125.0), e.mul(v, 2.0), 3)
        grain = e.sub(1.0, noise)

        # views across the strand show the noise most
        rim = e.sub(1.0, e.pow(e.max(e.abs(e.dot(view, across)), 1e-6), 1.5))

        # ---- G-buffer
        albedo = e.mul(e.v(group_inputs.outputs["Diffuse"]), HAIR_ALBEDO_SCALE)
        gate = e.sat(e.sub(e.f(group_inputs.outputs["Multi A"]), 0.02))
        tint = e.sub(1.0, e.mul(e.mul(e.mul(rim, rim), HAIR_COLOR_NOISE), grain))
        hair_albedo = e.mul(e.mul(albedo, tint), e.sub(1.0, gate))
        light_dir = e.normalize(e.attribute(scene_props.LIGHT_DIRECTION, vector=True))
        self_shadow = e.sat(e.dot(mapped_normal, light_dir))
        occlusion = e.mul(blue, e.sub(1.0, e.mul(e.mul(rim, HAIR_AO_NOISE), grain)))
        jitter = e.add(e.mul(e.pow(block, 0.75), 0.5), e.madd(noise, 1.625, -0.75))
        strand = e.normalize(e.add(along, e.mul(vertex_normal, e.mul(jitter, 0.25))))
        alpha = e.sat(
            e.madd(
                e.mul(e.f(group_inputs.outputs["Alpha"]), e.madd(noise, 0.5, 0.5)),
                8.0,
                -3.5,
            )
        )

        # Glowing parts: 50 x the albedo squared, through the game's
        # brightness boost (x1.5, ^1.3), which undoes the exposure.
        noisy = e.mul(albedo, tint)
        glow = e.add(e.mul(e.mul(noisy, noisy), 0.995), (0.005, 0.005, 0.005))
        glow = e.mul(glow, e.mul(e.mul(gate, gate), 75.0))
        gx, gy, gz = e.split(glow)
        glow = e.div(
            e.xyz(e.pow(gx, 1.3), e.pow(gy, 1.3), e.pow(gz, 1.3)),
            e.max(e.attribute(scene_props.EXPOSURE), 1e-3),
        )

        # ---- Principled, for Cycles
        bsdf = e._node(bpy.types.ShaderNodeBsdfPrincipled, name="Principled BSDF")
        game_albedo = e._node(bpy.types.ShaderNodeValue, name=GAME_ALBEDO)
        game_albedo.outputs[0].default_value = 1  # type: ignore
        base = e.lerp(
            e.v(group_inputs.outputs["Diffuse"]),
            e.mul(hair_albedo, occlusion),
            e.f(game_albedo.outputs[0]),
        )
        tree.add_link(base.socket, bsdf.inputs["Base Color"])
        # The game's highlight is two lobes round the strand, of exponents
        # n = 2/r^2 - 2 and n/24, at the same height. The sharp one carries
        # next to no light; the wide one is as wide as a GGX alpha of
        # r sqrt(24 / (1 + 23 r^2)), and Blender's roughness is the square
        # root of alpha. Fed r itself (a median 0.06 on hair), the BSDF
        # was near a mirror and the hair looked wet. What the game reflects
        # of the surroundings is a flat 2% (0.02 x AO), half the BSDF's 4%.
        wide = e.mul(green, e.sqrt(e.div(24.0, e.madd(e.mul(green, green), 23.0, 1.0))))
        tree.add_link(e.sqrt(e.min(wide, 1.0)).socket, bsdf.inputs["Roughness"])
        bsdf.inputs["Specular IOR Level"].default_value = 0.25  # type: ignore
        tree.add_link(mapped_normal.socket, bsdf.inputs["Normal"])
        tree.add_link(alpha.socket, bsdf.inputs["Alpha"])
        tree.add_link(glow.socket, bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1  # type: ignore

        # ---- the game's lighting, for EEVEE
        shadow = e._node(ShaderNodePso2GameShadow)
        light = e._node(ShaderNodePso2GameHairLight)
        for name, value in (
            ("Albedo", hair_albedo),
            ("Self Shadow", self_shadow),
            ("Roughness", green),
            ("AO", occlusion),
            ("Tangent", strand),
            ("Emission", glow),
        ):
            tree.add_link(value.socket, light.inputs[name])
        tree.add_link(shadow.outputs["Shadow"], light.inputs["Shadow"])

        lit = e._node(bpy.types.ShaderNodeEmission, name="Game Light")
        tree.add_link(light.outputs["Color"], lit.inputs["Color"])
        clear = e._node(bpy.types.ShaderNodeBsdfTransparent)
        cutout = e._node(bpy.types.ShaderNodeMixShader, name="Game Alpha")
        tree.add_link(alpha.socket, cutout.inputs["Fac"])
        tree.add_link(clear.outputs["BSDF"], cutout.inputs[1])
        tree.add_link(lit.outputs["Emission"], cutout.inputs[2])

        choose = e._node(bpy.types.ShaderNodeMixShader, name="Game Shading")
        tree.add_link(
            e.attribute(scene_props.GAME_SHADING).socket, choose.inputs["Fac"]
        )
        tree.add_link(bsdf.outputs["BSDF"], choose.inputs[1])
        tree.add_link(cutout.outputs["Shader"], choose.inputs[2])
        tree.add_link(choose.outputs["Shader"], group_outputs.inputs["BSDF"])
