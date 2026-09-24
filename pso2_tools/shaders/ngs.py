from typing import cast

import bpy

from .. import classes
from . import builder, group


@classes.register
class ShaderNodePso2AlphaThreshold(group.ShaderNodeCustomGroup):
    bl_name = "ShaderNodePso2AlphaThreshold"
    bl_label = "PSO2 Alpha Threshold"
    bl_icon = "NONE"

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput)
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput)

        tree.new_input(bpy.types.NodeSocketFloat, "Alpha")
        tree.new_input(bpy.types.NodeSocketFloat, "Threshold")

        tree.new_output(bpy.types.NodeSocketFloat, "Alpha")

        threshold = tree.add_node(bpy.types.ShaderNodeMath, name="Above Threshold")
        threshold.operation = "GREATER_THAN"

        disabled = tree.add_node(bpy.types.ShaderNodeMath, name="Disabled")
        disabled.operation = "COMPARE"
        disabled.inputs[1].default_value = 0  # type: ignore
        disabled.inputs[2].default_value = 0  # type: ignore

        mix = tree.add_node(bpy.types.ShaderNodeMix, name="Mix")
        mix.data_type = "FLOAT"
        mix.clamp_result = True

        tree.add_link(group_inputs.outputs["Alpha"], threshold.inputs[0])
        tree.add_link(group_inputs.outputs["Threshold"], threshold.inputs[1])

        tree.add_link(group_inputs.outputs["Threshold"], disabled.inputs[0])

        tree.add_link(disabled.outputs["Value"], mix.inputs["Factor"])
        tree.add_link(threshold.outputs["Value"], mix.inputs["A"])
        tree.add_link(group_inputs.outputs["Alpha"], mix.inputs["B"])

        tree.add_link(mix.outputs["Result"], group_outputs.inputs["Alpha"])


# What the game's G-buffer shaders (1100g, 1101g, 1102g) do to a material's
# textures before its lighting pass sees them, read from the compiled shaders
# and the constants of a frame captured in the character creator. Only the
# material side is reproduced: the lighting is Blender's.

# Every character shader darkens the diffuse texture by this much. The game's
# sun makes up for it: 12.56, or 4 once the diffuse 1/pi is taken out.
ALBEDO_SCALE = 0.61

# How far a soft area wraps light round to its dark side, which is what the
# game's subsurface amounts to. Skin takes it from its material constants
# (u_ObjBlendColor4.w in the captured frame); costumes use a fixed quarter.
SKIN_SUBSURFACE = 0.153
COSTUME_SUBSURFACE = 0.25

# The character creator's skin wetness: -35/127. Below zero dries the skin
# towards fully rough, above zero makes it glossier and darker.
SKIN_WET = -35 / 127

# The multi map's alpha glows only above this.
EMISSION_THRESHOLD = 0.02
# glow = (gain * gate^2 * albedo^2) ^ power / exposure. The gain is the shader's
# 50 times the scene's brightness enhancement of 1.5, which also raises it to
# 1.3; dividing by the exposure (0.84) keeps it that bright after tone mapping.
EMISSION_GAIN = 75.0
EMISSION_POWER = 1.3
EXPOSURE = 0.84

# The node in each group that switches the game's adjustments to the diffuse
# off, so a bake reads the plain colour back instead of a darkened one.
GAME_ALBEDO = "Game Albedo"


def _math(
    tree: builder.NodeTreeBuilder,
    operation: str,
    location: builder.Vec2,
    name: str,
    *values: float | None,
    clamp=False,
) -> bpy.types.ShaderNodeMath:
    node = tree.add_node(bpy.types.ShaderNodeMath, location, name=name)
    node.operation = operation  # type: ignore
    node.use_clamp = clamp
    for index, value in enumerate(values):
        if value is not None:
            node.inputs[index].default_value = value  # type: ignore
    return node


def _vector(
    tree: builder.NodeTreeBuilder, operation: str, location: builder.Vec2, name: str
) -> bpy.types.ShaderNodeVectorMath:
    node = tree.add_node(bpy.types.ShaderNodeVectorMath, location, name=name)
    node.operation = operation  # type: ignore
    return node


class ShaderNodePso2NgsBase(group.ShaderNodeCustomGroup):
    tree_version = 2
    subsurface_amount = COSTUME_SUBSURFACE
    subsurface_radius = (1.0, 1.0, 1.0)

    def init(self, context):
        super().init(context)

        self.input(bpy.types.NodeSocketColor, "Diffuse").default_value = (1, 0, 1, 1)  # type: ignore
        self.input(bpy.types.NodeSocketFloat, "Alpha").default_value = 1
        self.input(bpy.types.NodeSocketFloat, "Alpha Threshold").default_value = 0
        self.input(bpy.types.NodeSocketColor, "Multi RGB").default_value = (0, 1, 1, 1)  # type: ignore
        self.input(bpy.types.NodeSocketFloat, "Normal A").default_value = 1
        self.input(
            bpy.types.NodeSocketFloat, "Subsurface"
        ).default_value = self.subsurface_amount
        self.input(bpy.types.NodeSocketFloat, "Emission Scale").default_value = 1

    def _new_inputs(self, tree: builder.NodeTreeBuilder) -> None:
        """Add a subclass's own inputs after the shared ones."""

    def _wetness(
        self,
        tree: builder.NodeTreeBuilder,
        group_inputs: bpy.types.Node,
        green: bpy.types.NodeSocket,
        subsurface: bpy.types.NodeSocket,
    ) -> tuple[bpy.types.NodeSocket, bpy.types.NodeSocket | None]:
        """The multi map's green as the lighting reads it, and a factor for
        the diffuse. Only skin has one."""
        return green, None

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput, (-34, 0))
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput, (28, 0))

        tree.new_input(bpy.types.NodeSocketColor, "Diffuse")
        tree.new_input(bpy.types.NodeSocketFloat, "Alpha")
        tree.new_input(bpy.types.NodeSocketFloat, "Alpha Threshold")
        tree.new_input(bpy.types.NodeSocketColor, "Multi RGB")
        tree.new_input(bpy.types.NodeSocketFloat, "Multi A")
        tree.new_input(bpy.types.NodeSocketColor, "Normal")
        tree.new_input(bpy.types.NodeSocketFloat, "Normal A").default_value = 1
        tree.new_input(
            bpy.types.NodeSocketFloat, "Subsurface"
        ).default_value = self.subsurface_amount
        tree.new_input(bpy.types.NodeSocketFloat, "Emission Scale").default_value = 1
        self._new_inputs(tree)

        tree.new_output(bpy.types.NodeSocketShader, "BSDF")

        bsdf = tree.add_node(
            bpy.types.ShaderNodeBsdfPrincipled, (22, 0), name="Principled BSDF"
        )
        bsdf.inputs["Subsurface Radius"].default_value = self.subsurface_radius  # type: ignore
        tree.add_link(bsdf.outputs["BSDF"], group_outputs.inputs["BSDF"])

        # ========== Normal Map ==========

        normal_map = tree.add_node(bpy.types.ShaderNodeNormalMap, (16, -16))

        tree.add_link(group_inputs.outputs["Normal"], normal_map.inputs["Color"])
        tree.add_link(normal_map.outputs[0], bsdf.inputs["Normal"])

        # ========== Multi Map ==========

        multi_rgb = tree.add_node(
            bpy.types.ShaderNodeSeparateColor, (-28, 8), name="Multi Map RGB"
        )
        multi_rgb.mode = "RGB"

        tree.add_link(group_inputs.outputs["Multi RGB"], multi_rgb.inputs["Color"])

        # Red is the square root of metalness.
        metallic = _math(tree, "MULTIPLY", (-4, 10), "Metallic")
        tree.add_link(multi_rgb.outputs["Red"], metallic.inputs[0])
        tree.add_link(multi_rgb.outputs["Red"], metallic.inputs[1])
        tree.add_link(metallic.outputs["Value"], bsdf.inputs["Metallic"])

        # ========== Subsurface ==========

        # The normal map's alpha marks how soft an area is: at or below one
        # half it wraps light by the full amount, fading out towards one.
        soft_alpha = _math(
            tree, "MULTIPLY", (-28, -8), "Normal A", None, 1.02, clamp=True
        )
        hardness = _math(
            tree, "MULTIPLY_ADD", (-24, -8), "Hardness", None, 2, -1, clamp=True
        )
        softness = _math(tree, "SUBTRACT", (-20, -8), "Softness", 1)
        subsurface = _math(tree, "MULTIPLY", (-16, -8), "Subsurface Amount")

        tree.add_link(group_inputs.outputs["Normal A"], soft_alpha.inputs[0])
        tree.add_link(soft_alpha.outputs["Value"], hardness.inputs[0])
        tree.add_link(hardness.outputs["Value"], softness.inputs[1])
        tree.add_link(softness.outputs["Value"], subsurface.inputs[0])
        tree.add_link(group_inputs.outputs["Subsurface"], subsurface.inputs[1])

        # ========== Roughness ==========

        green, albedo_factor = self._wetness(
            tree, group_inputs, multi_rgb.outputs["Green"], subsurface.outputs["Value"]
        )

        # Green is roughness as it stands - GGX with alpha = roughness^2, the
        # same convention as the Principled BSDF - kept off a perfect mirror.
        roughness = _math(tree, "MAXIMUM", (-4, 6), "Roughness", None, 0.05)
        tree.add_link(green, roughness.inputs[0])
        tree.add_link(roughness.outputs["Value"], bsdf.inputs["Roughness"])

        # ========== Emission ==========

        gate = _math(
            tree,
            "SUBTRACT",
            (-24, 20),
            "Emission Gate",
            None,
            EMISSION_THRESHOLD,
            clamp=True,
        )
        tree.add_link(group_inputs.outputs["Multi A"], gate.inputs[0])

        albedo = _vector(tree, "SCALE", (-24, 26), "Albedo")
        albedo.inputs["Scale"].default_value = ALBEDO_SCALE  # type: ignore
        tree.add_link(group_inputs.outputs["Diffuse"], albedo.inputs[0])

        albedo_sq = _vector(tree, "MULTIPLY", (-20, 30), "Albedo Squared")
        tree.add_link(albedo.outputs["Vector"], albedo_sq.inputs[0])
        tree.add_link(albedo.outputs["Vector"], albedo_sq.inputs[1])

        glow_base = _vector(tree, "MULTIPLY_ADD", (-16, 30), "Glow Colour")
        glow_base.inputs[1].default_value = (0.995, 0.995, 0.995)  # type: ignore
        glow_base.inputs[2].default_value = (0.005, 0.005, 0.005)  # type: ignore
        tree.add_link(albedo_sq.outputs["Vector"], glow_base.inputs[0])

        gate_sq = _math(tree, "MULTIPLY", (-20, 22), "Gate Squared")
        tree.add_link(gate.outputs["Value"], gate_sq.inputs[0])
        tree.add_link(gate.outputs["Value"], gate_sq.inputs[1])

        gain = _math(tree, "MULTIPLY", (-16, 22), "Glow Gain", None, EMISSION_GAIN)
        tree.add_link(gate_sq.outputs["Value"], gain.inputs[0])

        glow = _vector(tree, "SCALE", (-12, 28), "Glow")
        tree.add_link(glow_base.outputs["Vector"], glow.inputs[0])
        tree.add_link(gain.outputs["Value"], glow.inputs["Scale"])

        # Per channel: the vector node's Power only arrived after Blender 4.4.
        split = tree.add_node(bpy.types.ShaderNodeSeparateXYZ, (-10, 28))
        enhanced = tree.add_node(
            bpy.types.ShaderNodeCombineXYZ, (-6, 28), name="Glow Enhanced"
        )
        tree.add_link(glow.outputs["Vector"], split.inputs[0])
        for offset, axis in enumerate("XYZ"):
            power = _math(
                tree,
                "POWER",
                (-8, 30 - offset),
                f"Glow Enhanced {axis}",
                None,
                EMISSION_POWER,
            )
            tree.add_link(split.outputs[axis], power.inputs[0])
            tree.add_link(power.outputs["Value"], enhanced.inputs[axis])

        exposure = _math(tree, "DIVIDE", (-8, 22), "Glow Exposure", None, EXPOSURE)
        tree.add_link(group_inputs.outputs["Emission Scale"], exposure.inputs[0])

        emission = _vector(tree, "SCALE", (-4, 26), "Emission")
        tree.add_link(enhanced.outputs["Vector"], emission.inputs[0])
        tree.add_link(exposure.outputs["Value"], emission.inputs["Scale"])
        tree.add_link(emission.outputs["Vector"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = 1  # type: ignore

        # Glowing texels carry no subsurface.
        not_glowing = _math(tree, "LESS_THAN", (-12, -4), "Not Glowing", None, 1e-3)
        tree.add_link(gate.outputs["Value"], not_glowing.inputs[0])

        wrap = _math(tree, "MULTIPLY", (-8, -6), "Subsurface Weight")
        tree.add_link(subsurface.outputs["Value"], wrap.inputs[0])
        tree.add_link(not_glowing.outputs["Value"], wrap.inputs[1])
        tree.add_link(wrap.outputs["Value"], bsdf.inputs["Subsurface Weight"])

        # ========== Base color ==========

        # What glows is taken out of the diffuse; blue is occlusion, which
        # in game darkens only the diffuse light, not the reflections.
        unlit = _math(tree, "SUBTRACT", (-12, 18), "Unlit", 1)
        tree.add_link(gate.outputs["Value"], unlit.inputs[1])

        occlusion = _math(tree, "MULTIPLY", (-8, 16), "Ambient Occlusion")
        tree.add_link(unlit.outputs["Value"], occlusion.inputs[0])
        tree.add_link(multi_rgb.outputs["Blue"], occlusion.inputs[1])

        factor = occlusion.outputs["Value"]
        if albedo_factor is not None:
            wet = _math(tree, "MULTIPLY", (-4, 16), "Wet Darkening")
            tree.add_link(factor, wet.inputs[0])
            tree.add_link(albedo_factor, wet.inputs[1])
            factor = wet.outputs["Value"]

        game_color = _vector(tree, "SCALE", (0, 20), "Game Diffuse")
        tree.add_link(albedo.outputs["Vector"], game_color.inputs[0])
        tree.add_link(factor, game_color.inputs["Scale"])

        game_albedo = tree.add_node(
            bpy.types.ShaderNodeValue, (0, 14), name=GAME_ALBEDO
        )
        game_albedo.outputs[0].default_value = 1  # type: ignore

        base_color = tree.add_node(bpy.types.ShaderNodeMix, (6, 18), name="Base Color")
        base_color.data_type = "RGBA"
        base_color.clamp_factor = True

        tree.add_link(game_albedo.outputs[0], base_color.inputs["Factor"])
        tree.add_link(group_inputs.outputs["Diffuse"], base_color.inputs["A"])
        tree.add_link(game_color.outputs["Vector"], base_color.inputs["B"])
        tree.add_link(base_color.outputs["Result"], bsdf.inputs["Base Color"])

        # ========== Alpha ==========

        alpha = tree.add_node(ShaderNodePso2AlphaThreshold, (16, -8), name="Alpha")

        tree.add_link(group_inputs.outputs["Alpha"], alpha.inputs["Alpha"])
        tree.add_link(
            group_inputs.outputs["Alpha Threshold"], alpha.inputs["Threshold"]
        )

        tree.add_link(alpha.outputs["Alpha"], bsdf.inputs["Alpha"])


@classes.register
class ShaderNodePso2Ngs(ShaderNodePso2NgsBase):
    bl_name = "ShaderNodePso2Ngs"
    bl_label = "PSO2 NGS"
    bl_icon = "NONE"


@classes.register
class ShaderNodePso2NgsSkin(ShaderNodePso2NgsBase):
    bl_name = "ShaderNodePso2NgsSkin"
    bl_label = "PSO2 NGS Skin"
    bl_icon = "NONE"

    subsurface_amount = SKIN_SUBSURFACE
    # The game's wrap is tinted by the albedo itself - its deep red
    # (u_ObjBlendColor4.rgb) enters squared, a couple of percent - so the
    # scatter stays neutral rather than taking Blender's red-heavy skin
    # default. Against the captured frame the face regions came out a
    # little closer (mean error 0.0267 against 0.0278).
    subsurface_radius = (1.0, 1.0, 1.0)

    def init(self, context):
        super().init(context)

        self.input(bpy.types.NodeSocketFloat, "Skin Wet").default_value = SKIN_WET
        self.input(bpy.types.NodeSocketFloat, "Wet Mask").default_value = 0

    def _new_inputs(self, tree):
        tree.new_input(bpy.types.NodeSocketFloat, "Skin Wet").default_value = SKIN_WET
        # 0 wets all of it (the face), 1 only its soft areas (the body).
        tree.new_input(bpy.types.NodeSocketFloat, "Wet Mask").default_value = 0

    def _wetness(self, tree, group_inputs, green, subsurface):
        # The body only takes the wetness where it is soft.
        soft = _math(tree, "MULTIPLY", (-16, 2), "Wet Area", None, 8, clamp=True)
        tree.add_link(subsurface, soft.inputs[0])

        area = tree.add_node(bpy.types.ShaderNodeMix, (-12, 2), name="Wet Mask")
        area.data_type = "FLOAT"
        area.clamp_factor = True
        area.inputs["A"].default_value = 1  # type: ignore
        tree.add_link(group_inputs.outputs["Wet Mask"], area.inputs["Factor"])
        tree.add_link(soft.outputs["Value"], area.inputs["B"])

        wet = _math(tree, "MULTIPLY", (-8, 2), "Wetness")
        tree.add_link(group_inputs.outputs["Skin Wet"], wet.inputs[0])
        tree.add_link(area.outputs["Result"], wet.inputs[1])

        wetter = _math(tree, "MULTIPLY", (-4, 0), "Wetter", None, 1, clamp=True)
        drier = _math(tree, "MULTIPLY", (-4, -2), "Drier", None, -1, clamp=True)
        tree.add_link(wet.outputs["Value"], wetter.inputs[0])
        tree.add_link(wet.outputs["Value"], drier.inputs[0])

        # Wet skin is up to 70% glossier, dry skin up to fully rough...
        gloss = _math(tree, "MULTIPLY_ADD", (0, 2), "Wet Gloss", None, -0.7, 1)
        tree.add_link(wetter.outputs["Value"], gloss.inputs[0])

        glossed = _math(tree, "MULTIPLY", (4, 4), "Wet Roughness")
        tree.add_link(green, glossed.inputs[0])
        tree.add_link(gloss.outputs["Value"], glossed.inputs[1])

        dried = tree.add_node(bpy.types.ShaderNodeMix, (8, 4), name="Dry Roughness")
        dried.data_type = "FLOAT"
        dried.clamp_factor = True
        dried.inputs["B"].default_value = 1  # type: ignore
        tree.add_link(drier.outputs["Value"], dried.inputs["Factor"])
        tree.add_link(glossed.outputs["Value"], dried.inputs["A"])

        # ...and wet skin up to 20% darker.
        darker = _math(tree, "MULTIPLY_ADD", (0, -2), "Wet Colour", None, -0.2, 1)
        tree.add_link(wetter.outputs["Value"], darker.inputs[0])

        return dried.outputs["Result"], darker.outputs["Value"]

    def _build(self, node_tree):
        super()._build(node_tree)

        bsdf = cast(
            "bpy.types.ShaderNodeBsdfPrincipled", node_tree.nodes["Principled BSDF"]
        )

        bsdf.subsurface_method = "RANDOM_WALK_SKIN"


def set_game_albedo(enabled: bool) -> list[bpy.types.NodeTree]:
    """Switch the game's darkening of the diffuse on or off in every group.

    Returns the trees that were switched.
    """
    switched = []
    for tree in bpy.data.node_groups:
        node = tree.nodes.get(GAME_ALBEDO)
        if node is None or node.type != "VALUE":
            continue
        node.outputs[0].default_value = 1.0 if enabled else 0.0  # type: ignore
        switched.append(tree)
    return switched
