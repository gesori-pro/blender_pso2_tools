import math

import bpy

from .. import classes, scene_props
from . import builder, group
from .game_lighting import Expr, ShaderNodePso2GameShadow, scene_light
from .shader_1104 import IRIS_SIZE

# What the game's eye tear shader does (the forward pass drawn over each eye
# after the lighting, added to the image), from the compiled shader and a
# captured frame. It is a thin shell over the eye that adds two things: a
# sharp reflection of the sun and of the fill (smooth: roughness 0.05), and
# a matcap - its texture, looked up by the normal in view space and scaled
# with the iris, is a dim blue rim that brightens the eye's edges.
TEAR_ROUGHNESS = 0.05
# the matcap moves sideways by a tenth of this, one way on each eye
TEAR_MATCAP_OFFSET = 0.5276
TEAR_MATCAP_GAIN = 2.0
TEAR_SPECULAR_LIMIT = 5.0
# The game adds only the lights' glints, nothing of the surroundings. The
# Principled BSDF cannot tell the two apart, so its reflection is turned
# down to 1/25 of the usual: a light's glint, far brighter than anything
# around it, still shows, and the eye no longer mirrors the environment
# (which in Material Preview hid the iris behind the studio's image).
TEAR_REFLECTION_LEVEL = 0.02

# The matcap's image node: the eye part's _v texture, which the character
# importer paints on when it loads the eye.
ENV_MAP = "Env Map"


class Shader1105(builder.ShaderBuilder):
    """NGS eye tear shader"""

    # Added over what is behind it: dithering would drop the surface where
    # it is fully transparent, and what it adds with it.
    render_method = "BLENDED"

    @property
    def textures(self):
        return self.data.textures

    def build(self, context):
        tree = self.init_tree()

        output = tree.add_node(bpy.types.ShaderNodeOutputMaterial, (30, 6))
        shader_group = tree.add_node(ShaderNodePso2NgsTear, (24, 6))
        tree.add_link(shader_group.outputs["BSDF"], output.inputs["Surface"])

        # The matcap coordinates, 0.5 + 0.5 x the normal in view space (x
        # right, y up), scaled about the middle like the iris (the character
        # importer sets the scale from the eye and iris size sliders).
        geometry = tree.add_node(bpy.types.ShaderNodeNewGeometry, (-24, 6))
        to_camera = tree.add_node(bpy.types.ShaderNodeVectorTransform, (-18, 6))
        to_camera.vector_type = "NORMAL"
        to_camera.convert_from = "WORLD"
        to_camera.convert_to = "CAMERA"
        tree.add_link(geometry.outputs["Normal"], to_camera.inputs["Vector"])
        centred = tree.add_node(bpy.types.ShaderNodeVectorMath, (-12, 6))
        centred.operation = "MULTIPLY_ADD"
        centred.inputs[1].default_value = (0.5, 0.5, 0.0)  # type: ignore
        centred.inputs[2].default_value = (0.5, 0.5, 0.0)  # type: ignore
        tree.add_link(to_camera.outputs["Vector"], centred.inputs[0])
        mapping = tree.add_node(bpy.types.ShaderNodeMapping, (-6, 6), name=IRIS_SIZE)
        tree.add_link(centred.outputs["Vector"], mapping.inputs["Vector"])
        side = -1.0 if "tear_l" in self.material.name else 1.0
        shift = tree.add_node(bpy.types.ShaderNodeVectorMath, (0, 6))
        shift.operation = "ADD"
        shift.inputs[1].default_value = (0.05 * TEAR_MATCAP_OFFSET * side, 0, 0)  # type: ignore
        tree.add_link(mapping.outputs["Vector"], shift.inputs[0])

        matcap = tree.add_node(bpy.types.ShaderNodeTexImage, (6, 6), name=ENV_MAP)
        matcap.image = self.textures.default.env
        tree.add_link(shift.outputs["Vector"], matcap.inputs["Vector"])
        tree.add_link(matcap.outputs["Color"], shader_group.inputs["Matcap Color"])


@classes.register
class ShaderNodePso2NgsTear(group.ShaderNodeCustomGroup):
    bl_name = "ShaderNodePso2NgsTear"
    bl_label = "PSO2 Eye Tear"
    bl_icon = "NONE"

    # 2: the Principled BSDF's reflection down to the lights' glints
    tree_version = 2

    def init(self, context):
        super().init(context)
        matcap = self.input(bpy.types.NodeSocketColor, "Matcap Color")
        matcap.default_value = (0, 0, 0, 1)  # type: ignore

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput, (-40, 0))
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput, (40, 0))
        tree.new_input(bpy.types.NodeSocketColor, "Matcap Color")
        tree.new_output(bpy.types.NodeSocketShader, "BSDF")

        e = Expr(tree, -34, 30)
        geometry = e._node(bpy.types.ShaderNodeNewGeometry)
        view = e.v(geometry.outputs["Incoming"])
        normal = e.v(geometry.outputs["Normal"])

        # ---- the game's pass, for EEVEE
        sun, color, fill, _, _ = scene_light(e)
        # like skin, the tear sees the sun with its vertical cut to a quarter
        sx, sy, sz = e.split(sun)
        light = e.normalize(e.xyz(sx, sy, e.mul(sz, 0.25)))

        alpha = TEAR_ROUGHNESS * TEAR_ROUGHNESS
        a2 = alpha * alpha
        ndv = e.max(e.dot(normal, view), 1e-3)

        def g1(x):
            x2 = e.mul(x, x)
            inner = e.madd(x2, 1 - a2, a2)
            return e.div(e.mul(x2, 2.0), e.madd(inner, inner, x))

        def specular(light_dir):
            ndl = e.sat(e.dot(normal, light_dir))
            half = e.normalize(e.add(view, light_dir))
            ndh = e.max(e.dot(normal, half), 1e-3)
            vdh = e.max(e.dot(view, half), 1e-3)
            fres = e.madd(e.exp2(e.mul(vdh, -12.5378895)), 0.96, 0.04)
            den = e.madd(e.mul(ndh, ndh), a2 - 1, 1.0)
            d = e.div(a2, e.mul(e.mul(den, den), math.pi))
            g = e.mul(g1(ndv), g1(ndl))
            # F D G / (4 N.V N.L), lit by N.L
            return e.div(e.mul(e.mul(d, g), fres), e.mul(ndv, 4.0))

        shadow = e.f(e._node(ShaderNodePso2GameShadow).outputs["Shadow"])
        sun_spec = e.mul(e.mul(specular(light), shadow), color)
        fill_spec = e.mul(specular(e.mul(light, -1.0)), e.mul(fill, 0.5))
        glint = e.min(e.add(sun_spec, fill_spec), TEAR_SPECULAR_LIMIT)
        exposure = e.max(e.attribute(scene_props.EXPOSURE), 1e-3)
        rim = e.mul(
            e.v(group_inputs.outputs["Matcap Color"]),
            e.div(TEAR_MATCAP_GAIN, exposure),
        )
        added = e._node(bpy.types.ShaderNodeEmission, name="Game Light")
        tree.add_link(e.add(glint, rim).socket, added.inputs["Color"])

        # ---- Cycles: the lights' glints and the matcap, added
        gloss = e._node(bpy.types.ShaderNodeBsdfPrincipled, name="Principled BSDF")
        gloss.inputs["Base Color"].default_value = (0, 0, 0, 1)  # type: ignore
        gloss.inputs["Roughness"].default_value = TEAR_ROUGHNESS  # type: ignore
        gloss.inputs["Specular IOR Level"].default_value = TEAR_REFLECTION_LEVEL  # type: ignore
        tree.add_link(rim.socket, gloss.inputs["Emission Color"])
        gloss.inputs["Emission Strength"].default_value = 1  # type: ignore

        choose = e._node(bpy.types.ShaderNodeMixShader, name="Game Shading")
        tree.add_link(
            e.attribute(scene_props.GAME_SHADING).socket, choose.inputs["Fac"]
        )
        tree.add_link(gloss.outputs["BSDF"], choose.inputs[1])
        tree.add_link(added.outputs["Emission"], choose.inputs[2])

        # Added over what is behind it.
        clear = e._node(bpy.types.ShaderNodeBsdfTransparent)
        over = e._node(bpy.types.ShaderNodeAddShader)
        tree.add_link(clear.outputs["BSDF"], over.inputs[0])
        tree.add_link(choose.outputs["Shader"], over.inputs[1])
        tree.add_link(over.outputs["Shader"], group_outputs.inputs["BSDF"])
