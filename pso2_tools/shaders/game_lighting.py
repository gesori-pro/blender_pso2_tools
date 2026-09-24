"""The game's own lighting, for EEVEE: what the deferred lighting pass does
with the G-buffer, rebuilt as node groups.

The material groups (ngs.py, shader_1103.py) already reproduce what the
game's G-buffer shaders write. Here that is lit the way the character
creator lights it, instead of by a Principled BSDF:

- models 0 and 4 (costume and skin): the diffuse is Lambert on the parts that
  are not soft, and wraps round by the soft amount on the rest, tinted by the
  albedo and the subsurface colour; silhouettes of soft areas take light from
  behind; skin (model 4) sees the sun with its vertical component cut to a
  quarter. Specular is GGX with the game's own geometry term. Characters also
  get a colourless fill from the opposite side, and a floor on the sun's
  colour so they never go dark. A screen-space ambient occlusion darkens the
  sun, the fill and the ambient. Eyes are model 0 too, fully rough.
- model 1 (hair): a strand model on the hair's tangent, with a normal built
  from the view, two specular lobes and an ambient from both sides.
- the character creator also hangs a point light on the camera, which only
  lights characters: the same diffuse and specular for models 0/4 without
  the extra terms of the sun, and a diffuse alone for hair.

Checked against the lighting pass of a captured frame before it was built
as nodes: 1.2% median per-pixel error for models 0/4, 0.3-0.5% for hair.
Rendered in EEVEE with the glare and tone curve of game_scene.py, the head
comes within 7/255 of the creator's final frame on average.

The sun comes in through scene properties the Game Lighting setup drives
from its sun object; the ambient from an environment image in world space.
The result is emitted, so it only works where lights can be read back into
a material (EEVEE's Shader to RGB), which is also where its shadows come
from. Cycles keeps the Principled path.
"""

import math

import bpy

from .. import classes, scene_props
from . import builder, group

ENVIRONMENT_IMAGE = "PSO2 Environment"


class Val:
    """A node output in an expression, and whether it is a float or a vector."""

    def __init__(self, socket, vector: bool):
        self.socket = socket
        self.vector = vector


class Expr:
    """Math on node sockets. Each operation adds the nodes it needs."""

    def __init__(self, tree: builder.NodeTreeBuilder, x: float = 0, y: float = 0):
        self.tree = tree
        self.x = x
        self.y = y
        self.column = 0

    def _node(self, node_type, name=None):
        node = self.tree.add_node(
            node_type,
            (self.x + (self.column // 12) * 4, self.y - (self.column % 12) * 3),
            name=name,
        )
        self.column += 1
        return node

    # ---- wrapping
    def f(self, socket) -> Val:
        return Val(socket, False)

    def v(self, socket) -> Val:
        return Val(socket, True)

    def const(self, value) -> Val:
        if isinstance(value, (tuple, list)):
            node = self._node(bpy.types.ShaderNodeCombineXYZ)
            for axis, component in zip("XYZ", value, strict=True):
                node.inputs[axis].default_value = component  # type: ignore
            return Val(node.outputs[0], True)
        node = self._node(bpy.types.ShaderNodeValue)
        node.outputs[0].default_value = value  # type: ignore
        return Val(node.outputs[0], False)

    def _in(self, node, index, operand, vector: bool):
        socket = node.inputs[index]
        if isinstance(operand, Val):
            if vector and not operand.vector:
                operand = self.splat(operand)
            self.tree.add_link(operand.socket, socket)
        elif vector:
            socket.default_value = (
                (operand,) * 3 if isinstance(operand, (int, float)) else operand
            )  # type: ignore
        else:
            socket.default_value = operand  # type: ignore

    @staticmethod
    def _is_vector(*operands) -> bool:
        return any(isinstance(o, Val) and o.vector for o in operands) or any(
            isinstance(o, (tuple, list)) for o in operands
        )

    def splat(self, value: Val) -> Val:
        node = self._node(bpy.types.ShaderNodeCombineXYZ)
        for axis in "XYZ":
            self.tree.add_link(value.socket, node.inputs[axis])
        return Val(node.outputs[0], True)

    def _math(self, operation, *operands, clamp=False) -> Val:
        node = self._node(bpy.types.ShaderNodeMath)
        node.operation = operation  # type: ignore
        node.use_clamp = clamp
        for index, operand in enumerate(operands):
            self._in(node, index, operand, False)
        return Val(node.outputs[0], False)

    def _vmath(self, operation, *operands, output=0) -> Val:
        node = self._node(bpy.types.ShaderNodeVectorMath)
        node.operation = operation  # type: ignore
        for index, operand in enumerate(operands):
            if operation == "SCALE" and index == 1:
                self._in(node, "Scale", operand, False)
            else:
                self._in(node, index, operand, True)
        return Val(node.outputs[output], output == 0)

    # ---- arithmetic, dispatched on float/vector
    def add(self, a, b):
        if self._is_vector(a, b):
            return self._vmath("ADD", a, b)
        return self._math("ADD", a, b)

    def sub(self, a, b):
        if self._is_vector(a, b):
            return self._vmath("SUBTRACT", a, b)
        return self._math("SUBTRACT", a, b)

    def mul(self, a, b):
        av = self._is_vector(a)
        bv = self._is_vector(b)
        if av and bv:
            return self._vmath("MULTIPLY", a, b)
        if av:
            return self._vmath("SCALE", a, b)
        if bv:
            return self._vmath("SCALE", b, a)
        return self._math("MULTIPLY", a, b)

    def div(self, a, b):
        if self._is_vector(a, b):
            return self._vmath("DIVIDE", a, b)
        return self._math("DIVIDE", a, b)

    def madd(self, a, b, c):
        """a * b + c on floats."""
        return self._math("MULTIPLY_ADD", a, b, c)

    def max(self, a, b):
        if self._is_vector(a, b):
            return self._vmath("MAXIMUM", a, b)
        return self._math("MAXIMUM", a, b)

    def min(self, a, b):
        if self._is_vector(a, b):
            return self._vmath("MINIMUM", a, b)
        return self._math("MINIMUM", a, b)

    def sat(self, a):
        if self._is_vector(a):
            return self._vmath("MINIMUM", self._vmath("MAXIMUM", a, 0.0), 1.0)
        return self._math("MULTIPLY", a, 1.0, clamp=True)

    def pow(self, a, b):
        return self._math("POWER", a, b)

    def exp2(self, a):
        return self._math("POWER", 2.0, a)

    def log2(self, a):
        return self._math("LOGARITHM", a, 2.0)

    def sqrt(self, a):
        return self._math("SQRT", a)

    def abs(self, a):
        return self._math("ABSOLUTE", a)

    def gt(self, a, b):
        return self._math("GREATER_THAN", a, b)

    def dot(self, a, b):
        return self._vmath("DOT_PRODUCT", a, b, output=1)

    def normalize(self, a):
        return self._vmath("NORMALIZE", a)

    def lerp(self, a, b, t):
        return self.add(a, self.mul(self.sub(b, a), t))

    def xyz(self, x, y, z):
        node = self._node(bpy.types.ShaderNodeCombineXYZ)
        for axis, component in zip("XYZ", (x, y, z), strict=True):
            self._in(node, axis, component, False)
        return Val(node.outputs[0], True)

    def split(self, a):
        node = self._node(bpy.types.ShaderNodeSeparateXYZ)
        self._in(node, 0, a, True)
        return tuple(Val(node.outputs[axis], False) for axis in "XYZ")

    def attribute(self, name, vector=False):
        node = self._node(bpy.types.ShaderNodeAttribute)
        node.attribute_type = "VIEW_LAYER"
        node.attribute_name = name
        return Val(node.outputs["Vector" if vector else "Fac"], vector)


def environment(e: Expr, direction: Val, name: str = "Environment") -> Val:
    """The lighting environment in a world direction."""
    node = e._node(bpy.types.ShaderNodeTexEnvironment, name=name)
    node.image = bpy.data.images.get(ENVIRONMENT_IMAGE)
    node.interpolation = "Linear"
    e.tree.add_link(direction.socket, node.inputs["Vector"])
    return Val(node.outputs["Color"], True)


def scene_light(e: Expr):
    """Sun direction (towards the light), its colour with the character floor,
    the fill strength, the environment tint and the sun's raw strength."""
    direction = e.normalize(e.attribute(scene_props.LIGHT_DIRECTION, vector=True))
    strength = e.attribute(scene_props.LIGHT_STRENGTH)
    exposure = e.max(e.attribute(scene_props.EXPOSURE), 1e-3)
    # Characters (the flag every character material carries) never see the
    # sun dimmer than 1 / (0.7 * exposure), and get a fill of 0.3 of that.
    floor = e.div(1.0, e.mul(exposure, 0.7))
    color = e.splat(e.max(strength, floor))
    fill = e.mul(floor, 0.3)
    env_color = e.attribute(scene_props.ENVIRONMENT_COLOR, vector=True)
    return direction, color, fill, env_color, strength


# How the creator's headlight fades: 1 / (1 + 0.24 * distance). (It is also
# cut off from 95 to 100 metres away, which no shot of a character reaches.)
HEADLIGHT_ATTENUATION = 0.24

# The game darkens costume and skin by a screen-space ambient occlusion, a
# wide, soft one, on top of the material's own. EEVEE's is sharper; at
# 0.4 m and 0.38 of its strength it darkens the creator's face and eyes as
# much on average (0.93).
SCREEN_AO_DISTANCE = 0.4
SCREEN_AO_STRENGTH = 0.38


def headlight(e: Expr) -> Val:
    """The strength of the light on the camera where this point is. The
    game divides it by the exposure, so it looks the same at any."""
    distance = e.f(e._node(bpy.types.ShaderNodeCameraData).outputs["View Distance"])
    strength = e.attribute(scene_props.HEADLIGHT)
    exposure = e.max(e.attribute(scene_props.EXPOSURE), 1e-3)
    return e.div(
        strength, e.mul(e.madd(distance, HEADLIGHT_ATTENUATION, 1.0), exposure)
    )


@classes.register
class ShaderNodePso2GameShadow(group.ShaderNodeCustomGroup):
    """How much of the sun reaches this point: EEVEE's own lighting of a
    white diffuse surface, divided by what the unshadowed sun would give."""

    bl_name = "ShaderNodePso2GameShadow"
    bl_label = "PSO2 Game Shadow"
    bl_icon = "NONE"

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)
        outputs = tree.add_node(bpy.types.NodeGroupOutput, (30, 0))
        tree.new_output(bpy.types.NodeSocketFloat, "Shadow")
        e = Expr(tree, -10, 10)

        geometry = e._node(bpy.types.ShaderNodeNewGeometry)
        normal = e.v(geometry.outputs["Normal"])
        diffuse = e._node(bpy.types.ShaderNodeBsdfDiffuse)
        diffuse.inputs["Color"].default_value = (1, 1, 1, 1)  # type: ignore
        tree.add_link(geometry.outputs["Normal"], diffuse.inputs["Normal"])
        to_rgb = e._node(bpy.types.ShaderNodeShaderToRGB)
        tree.add_link(diffuse.outputs["BSDF"], to_rgb.inputs["Shader"])
        lit = e.split(e.v(to_rgb.outputs["Color"]))[1]

        direction = e.normalize(e.attribute(scene_props.LIGHT_DIRECTION, vector=True))
        strength = e.max(e.attribute(scene_props.LIGHT_STRENGTH), 1e-4)
        ndl = e.dot(normal, direction)
        # Facing away the game's shadow map has them in shadow anyway.
        facing = e.gt(ndl, 0.02)
        expected = e.mul(e.max(ndl, 0.02), e.div(strength, math.pi))
        shadow = e.mul(e.sat(e.div(lit, expected)), facing)
        tree.add_link(shadow.socket, outputs.inputs["Shadow"])


@classes.register
class ShaderNodePso2GameLight(group.ShaderNodeCustomGroup):
    """Models 0 (costume) and 4 (skin) of the deferred lighting."""

    bl_name = "ShaderNodePso2GameLight"
    bl_label = "PSO2 Game Light"
    bl_icon = "NONE"

    # 2: the screen-space ambient occlusion
    tree_version = 2

    def init(self, context):
        super().init(context)
        self.input(bpy.types.NodeSocketFloat, "AO").default_value = 1
        self.input(bpy.types.NodeSocketFloat, "Shadow").default_value = 1

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)
        gi = tree.add_node(bpy.types.NodeGroupInput, (-60, 0))
        go = tree.add_node(bpy.types.NodeGroupOutput, (60, 0))
        for socket_type, name in (
            (bpy.types.NodeSocketColor, "Albedo"),
            (bpy.types.NodeSocketFloat, "Soft"),
            (bpy.types.NodeSocketFloat, "Metal Root"),
            (bpy.types.NodeSocketFloat, "Roughness"),
            (bpy.types.NodeSocketFloat, "AO"),
            (bpy.types.NodeSocketColor, "Scatter"),
            (bpy.types.NodeSocketFloat, "Rim"),
            (bpy.types.NodeSocketVector, "Normal"),
            (bpy.types.NodeSocketFloat, "Skin"),
            (bpy.types.NodeSocketFloat, "Shadow"),
        ):
            tree.new_input(socket_type, name)
        tree.new_output(bpy.types.NodeSocketColor, "Color")

        e = Expr(tree, -50, 20)
        albedo = e.v(gi.outputs["Albedo"])
        a = e.f(gi.outputs["Soft"])
        metal_root = e.f(gi.outputs["Metal Root"])
        rough = e.f(gi.outputs["Roughness"])
        occlusion = e.f(gi.outputs["AO"])
        screen_ao = e._node(bpy.types.ShaderNodeAmbientOcclusion)
        screen_ao.inputs["Distance"].default_value = SCREEN_AO_DISTANCE  # type: ignore
        screen = e.madd(
            e.f(screen_ao.outputs["AO"]), SCREEN_AO_STRENGTH, 1 - SCREEN_AO_STRENGTH
        )
        # the sun, the ambient and the fill see both; the headlight only
        # the material's
        occluded = e.mul(occlusion, screen)
        scatter = e.v(gi.outputs["Scatter"])
        rim = e.f(gi.outputs["Rim"])
        n = e.normalize(e.v(gi.outputs["Normal"]))
        skin = e.f(gi.outputs["Skin"])
        shadow = e.f(gi.outputs["Shadow"])
        view = e.v(e._node(bpy.types.ShaderNodeNewGeometry).outputs["Incoming"])

        sun, color, fill, env_color, _ = scene_light(e)
        # Skin sees the sun with its vertical component cut to a quarter.
        sx, sy, sz = e.split(sun)
        flat = e.normalize(e.xyz(sx, sy, e.mul(sz, 0.25)))
        light = e.lerp(sun, flat, skin)

        metal = e.mul(metal_root, metal_root)
        diffuse_color = e.mul(albedo, e.sub(1.0, metal))
        f0 = e.lerp((0.04, 0.04, 0.04), albedo, metal)
        tint_a = e.add(e.mul(albedo, e.sub(1.0, a)), e.mul(scatter, e.mul(a, a)))
        tint_b = e.add(e.mul(albedo, e.sub(1.0, a)), e.mul(scatter, a))

        alpha = e.pow(e.max(rough, 0.05), 2.0)
        a2 = e.mul(alpha, alpha)
        k = e.sub(1.0, a2)
        ndv = e.max(e.dot(n, view), 1e-3)

        def g1(x):
            x2 = e.mul(x, x)
            inner = e.madd(x2, k, a2)
            return e.div(e.mul(x2, 2.0), e.madd(inner, inner, x))

        g_view = g1(ndv)

        def specular(light_dir, ndl_s):
            half = e.normalize(e.add(view, light_dir))
            ndh = e.max(e.dot(n, half), 1e-3)
            vdh = e.max(e.dot(view, half), 1e-3)
            fres = e.add(
                f0, e.mul(e.sub((1.0, 1.0, 1.0), f0), e.exp2(e.mul(vdh, -12.5378895)))
            )
            den = e.madd(e.mul(ndh, ndh), e.sub(a2, 1.0), 1.0)
            d = e.div(a2, e.mul(e.mul(den, den), math.pi))
            g = e.mul(g_view, g1(ndl_s))
            brdf = e.div(e.mul(d, g), e.madd(e.mul(ndv, ndl_s), 4.0, 1e-7))
            return e.mul(fres, e.mul(brdf, ndl_s))

        # sun
        ndl = e.dot(n, light)
        ndl_s = e.sat(ndl)
        lambert = e.mul(e.mul(occluded, shadow), ndl_s)
        wrap = e.mul(e.max(e.div(e.mul(e.add(ndl, 2.0), a), 3.0), 0.0), shadow)
        back = e.mul(e.mul(e.madd(ndl, -0.5, 0.5), rim), shadow)
        direct = e.add(e.mul(tint_a, wrap), e.splat(e.mul(lambert, e.sub(1.0, a))))
        direct = e.add(e.mul(tint_b, back), direct)
        direct = e.mul(e.mul(direct, color), diffuse_color)

        # ambient: the environment round the normal, and the sky tint from above
        _, _, nz = e.split(n)
        up = e.sat(e.madd(nz, 0.5, 0.5))
        irradiance = e.add(environment(e, n), e.mul(env_color, up))
        indirect = e.lerp(albedo, scatter, e.mul(a, 0.5))
        indirect = e.mul(e.mul(indirect, irradiance), occluded)

        diffuse = e.add(e.mul(direct, 1.0 / math.pi), indirect)
        spec = e.mul(e.mul(specular(light, ndl_s), color), shadow)
        result = e.add(e.mul(diffuse, e.sub(1.0, metal)), spec)

        # the fill: colourless, from the opposite side, never shadowed
        back_light = e.mul(light, -1.0)
        ndl2 = e.dot(n, back_light)
        ndl2_s = e.sat(ndl2)
        wrap2 = e.max(e.div(e.mul(e.add(ndl2, 2.0), a), 3.0), 0.0)
        fill_diffuse = e.add(
            e.mul(tint_a, wrap2),
            e.splat(e.mul(e.mul(ndl2_s, occluded), e.sub(1.0, a))),
        )
        fill_diffuse = e.mul(
            e.mul(fill_diffuse, diffuse_color), e.mul(fill, e.sub(1.0, metal))
        )
        fill_spec = e.mul(specular(back_light, ndl2_s), e.mul(fill, 0.5))
        result = e.add(result, e.add(e.mul(fill_diffuse, 1.0 / math.pi), fill_spec))

        # the headlight, from the camera: no shadow, no light from behind, and
        # the ambient occlusion on its specular only
        lamp = headlight(e)
        ndl3 = e.dot(n, view)
        ndl3_s = e.sat(ndl3)
        wrap3 = e.max(e.div(e.mul(e.add(ndl3, 2.0), a), 3.0), 0.0)
        lamp_diffuse = e.add(
            e.mul(tint_a, wrap3), e.splat(e.mul(ndl3_s, e.sub(1.0, a)))
        )
        lamp_diffuse = e.mul(
            e.mul(lamp_diffuse, diffuse_color),
            e.mul(lamp, e.div(e.sub(1.0, metal), math.pi)),
        )
        lamp_spec = e.mul(specular(view, ndl3_s), e.mul(lamp, occlusion))
        result = e.add(result, e.add(lamp_diffuse, lamp_spec))

        # glow: what is not soft carries emission in the same channel
        glowing = e.sub(1.0, e.sat(e.mul(a, 1e5)))
        result = e.add(result, e.mul(scatter, glowing))
        tree.add_link(result.socket, go.inputs["Color"])


@classes.register
class ShaderNodePso2GameHairLight(group.ShaderNodeCustomGroup):
    """Model 1 (hair) of the deferred lighting."""

    bl_name = "ShaderNodePso2GameHairLight"
    bl_label = "PSO2 Game Hair Light"
    bl_icon = "NONE"

    def init(self, context):
        super().init(context)
        self.input(bpy.types.NodeSocketFloat, "AO").default_value = 1
        self.input(bpy.types.NodeSocketFloat, "Shadow").default_value = 1
        self.input(bpy.types.NodeSocketFloat, "Self Shadow").default_value = 1

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)
        gi = tree.add_node(bpy.types.NodeGroupInput, (-60, 0))
        go = tree.add_node(bpy.types.NodeGroupOutput, (60, 0))
        for socket_type, name in (
            (bpy.types.NodeSocketColor, "Albedo"),
            (bpy.types.NodeSocketFloat, "Self Shadow"),
            (bpy.types.NodeSocketFloat, "Roughness"),
            (bpy.types.NodeSocketFloat, "AO"),
            (bpy.types.NodeSocketVector, "Tangent"),
            (bpy.types.NodeSocketColor, "Emission"),
            (bpy.types.NodeSocketFloat, "Shadow"),
        ):
            tree.new_input(socket_type, name)
        tree.new_output(bpy.types.NodeSocketColor, "Color")

        e = Expr(tree, -50, 20)
        albedo = e.v(gi.outputs["Albedo"])
        self_shadow = e.f(gi.outputs["Self Shadow"])
        rough = e.f(gi.outputs["Roughness"])
        occlusion = e.f(gi.outputs["AO"])
        t = e.normalize(e.v(gi.outputs["Tangent"]))
        emission = e.v(gi.outputs["Emission"])
        shadow = e.f(gi.outputs["Shadow"])
        view = e.v(e._node(bpy.types.ShaderNodeNewGeometry).outputs["Incoming"])
        light, color, _, env_color, _ = scene_light(e)

        # a normal across the strand, from the view
        vt = e.dot(view, t)
        n = e.normalize(e.sub(e.mul(t, vt), view))
        tl = e.dot(light, t)
        sin_l = e.sqrt(e.max(e.madd(e.mul(tl, tl), -1.0, 1.0), 0.0))
        sin_v = e.sqrt(e.max(e.madd(e.mul(vt, vt), -1.0, 1.0), 0.0))
        nl = e.dot(n, light)
        abs_tl = e.abs(tl)
        abs_nl = e.abs(nl)
        one_minus_tl = e.sub(1.0, abs_tl)
        one_minus_nl = e.sub(1.0, abs_nl)
        nl2 = e.mul(abs_nl, abs_nl)
        omn2 = e.mul(one_minus_nl, one_minus_nl)
        shape = e.add(
            e.add(e.mul(one_minus_tl, one_minus_tl), e.mul(tl, tl)),
            e.add(e.mul(nl2, nl2), e.mul(omn2, omn2)),
        )
        shape = e.madd(shape, 0.25, 0.5)
        diffuse = e.mul(e.mul(albedo, color), e.mul(shape, self_shadow))
        wrapped = e.mul(
            e.mul(e.mul(albedo, albedo), color),
            e.mul(e.mul(e.madd(nl, 0.5, 0.5), shadow), 0.5),
        )
        diffuse = e.add(diffuse, wrapped)

        cos_sum = e.max(e.sub(e.mul(sin_l, sin_v), e.mul(vt, tl)), 1e-6)
        inv = e.div(1.0, e.max(rough, 0.01))
        exponent = e.max(e.sub(e.mul(e.mul(inv, inv), 2.0), 2.0), 1e-3)
        amplitude = e.mul(e.exp2(e.mul(rough, -2.88539)), 24.0)
        lg = e.log2(cos_sum)
        lobe1 = e.mul(e.exp2(e.mul(lg, exponent)), e.mul(amplitude, 0.002))
        lobe2 = e.mul(e.exp2(e.mul(lg, e.div(exponent, 24.0))), e.mul(amplitude, 0.002))
        ax, ay, az = e.split(albedo)
        tint = e.xyz(
            e.pow(e.max(ax, 1e-6), 0.25),
            e.pow(e.max(ay, 1e-6), 0.25),
            e.pow(e.max(az, 1e-6), 0.25),
        )
        spec = e.add(e.mul(tint, lobe2), e.splat(lobe1))
        spec = e.mul(e.mul(spec, color), self_shadow)

        # ambient from both sides of the strand normal, less round the top
        _, _, nz = e.split(n)
        up = e.madd(nz, 0.5, 0.5)
        both = e.mul(
            e.add(
                environment(e, n, "Environment Front"),
                environment(e, e.mul(n, -1.0), "Environment Back"),
            ),
            0.5,
        )
        irradiance = e.add(both, e.mul(env_color, e.sat(up)))
        side = e.madd(e.sub(1.0, e.abs(nz)), 0.5, 0.5)
        indirect = e.mul(
            e.mul(irradiance, albedo), e.mul(e.madd(occlusion, 0.5, 0.5), side)
        )

        reflected = e._vmath("REFLECT", e.mul(view, -1.0), n)
        mirror = e.mul(
            environment(e, reflected, "Environment Reflection"), e.mul(occlusion, 0.02)
        )

        lit = e.mul(
            e.add(e.mul(diffuse, 1.0 / math.pi), spec), e.mul(occlusion, shadow)
        )
        # The headlight, from the camera, lights the strand's diffuse alone
        # (by how far across the strand the view is), over 2 pi rather than pi.
        abs_vt = e.abs(vt)
        across = e.sub(1.0, abs_vt)
        edge = e.sub(1.0, sin_v)
        sin_v2 = e.mul(sin_v, sin_v)
        edge2 = e.mul(edge, edge)
        lamp_shape = e.add(
            e.add(e.mul(across, across), e.mul(vt, vt)),
            e.add(e.mul(sin_v2, sin_v2), e.mul(edge2, edge2)),
        )
        lamp_shape = e.madd(lamp_shape, 0.25, 0.5)
        lamp = e.mul(
            e.mul(headlight(e), e.mul(lamp_shape, sin_v)),
            e.mul(e.madd(occlusion, 0.5, 0.5), 0.5 / math.pi),
        )

        result = e.add(e.add(lit, indirect), e.add(mirror, emission))
        result = e.add(result, e.mul(albedo, lamp))
        tree.add_link(result.socket, go.inputs["Color"])
