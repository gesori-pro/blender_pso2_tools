from typing import Literal

import bpy

from .. import classes
from ..colors import ColorId, ColorMapping
from . import builder, group


class _ShaderNodePso2SrgbBase(group.ShaderNodeCustomGroup):
    """Per-channel sRGB transfer curve, the exact piecewise form."""

    encode: bool = True

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput)
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput)

        tree.new_input(bpy.types.NodeSocketColor, "Color")
        tree.new_output(bpy.types.NodeSocketColor, "Color")

        split = tree.add_node(bpy.types.ShaderNodeSeparateColor)
        split.mode = "RGB"
        join = tree.add_node(bpy.types.ShaderNodeCombineColor)
        join.mode = "RGB"
        tree.add_link(group_inputs.outputs["Color"], split.inputs["Color"])
        tree.add_link(join.outputs["Color"], group_outputs.inputs["Color"])

        for channel in ("Red", "Green", "Blue"):
            value = tree.add_node(bpy.types.ShaderNodeMath)
            value.operation = "MAXIMUM"
            value.inputs[1].default_value = 0  # type: ignore
            tree.add_link(split.outputs[channel], value.inputs[0])

            low = tree.add_node(bpy.types.ShaderNodeMath)
            low.operation = "MULTIPLY" if self.encode else "DIVIDE"
            low.inputs[1].default_value = 12.92  # type: ignore
            tree.add_link(value.outputs[0], low.inputs[0])

            high = tree.add_node(bpy.types.ShaderNodeMath)
            if self.encode:
                # 1.055 * c^(1/2.4) - 0.055
                power = tree.add_node(bpy.types.ShaderNodeMath)
                power.operation = "POWER"
                power.inputs[1].default_value = 1 / 2.4  # type: ignore
                tree.add_link(value.outputs[0], power.inputs[0])
                high.operation = "MULTIPLY_ADD"
                high.inputs[1].default_value = 1.055  # type: ignore
                high.inputs[2].default_value = -0.055  # type: ignore
                tree.add_link(power.outputs[0], high.inputs[0])
            else:
                # ((c + 0.055) / 1.055)^2.4
                shift = tree.add_node(bpy.types.ShaderNodeMath)
                shift.operation = "MULTIPLY_ADD"
                shift.inputs[1].default_value = 1 / 1.055  # type: ignore
                shift.inputs[2].default_value = 0.055 / 1.055  # type: ignore
                tree.add_link(value.outputs[0], shift.inputs[0])
                high.operation = "POWER"
                high.inputs[1].default_value = 2.4  # type: ignore
                tree.add_link(shift.outputs[0], high.inputs[0])

            knee = tree.add_node(bpy.types.ShaderNodeMath)
            knee.operation = "GREATER_THAN"
            knee.inputs[1].default_value = 0.0031308 if self.encode else 0.04045  # type: ignore
            tree.add_link(value.outputs[0], knee.inputs[0])

            pick = tree.add_node(bpy.types.ShaderNodeMix)
            pick.data_type = "FLOAT"
            tree.add_link(knee.outputs[0], pick.inputs["Factor"])
            tree.add_link(low.outputs[0], pick.inputs["A"])
            tree.add_link(high.outputs[0], pick.inputs["B"])
            tree.add_link(pick.outputs["Result"], join.inputs[channel])


@classes.register
class ShaderNodePso2SrgbEncode(_ShaderNodePso2SrgbBase):
    bl_name = "ShaderNodePso2SrgbEncode"
    bl_label = "PSO2 sRGB Encode"
    bl_icon = "NONE"

    encode = True


@classes.register
class ShaderNodePso2SrgbDecode(_ShaderNodePso2SrgbBase):
    bl_name = "ShaderNodePso2SrgbDecode"
    bl_label = "PSO2 sRGB Decode"
    bl_icon = "NONE"

    encode = False


class ShaderNodePso2ColorizeBase(group.ShaderNodeCustomGroup):
    # 2: blends in sRGB space, as the game composites (see _build)
    tree_version = 2
    operation: Literal["MIX", "MULTIPLY"] = "MIX"

    def _set_channel_used(self, channel: int, used: bool):
        self.input(
            bpy.types.NodeSocketBool, f"Use Color {channel}"
        ).default_value = used

    def set_colors_used(self, colors: ColorMapping | list[int]):
        if isinstance(colors, ColorMapping):
            self._set_channel_used(1, colors.red != ColorId.UNUSED)
            self._set_channel_used(2, colors.green != ColorId.UNUSED)
            self._set_channel_used(3, colors.blue != ColorId.UNUSED)
            self._set_channel_used(4, colors.alpha != ColorId.UNUSED)
        else:
            self._set_channel_used(1, 1 in colors)
            self._set_channel_used(2, 2 in colors)
            self._set_channel_used(3, 3 in colors)
            self._set_channel_used(4, 4 in colors)

    def _build(self, node_tree):
        tree = builder.NodeTreeBuilder(node_tree)

        group_inputs = tree.add_node(bpy.types.NodeGroupInput)
        group_outputs = tree.add_node(bpy.types.NodeGroupOutput)

        tree.new_input(bpy.types.NodeSocketColor, "Input")
        tree.new_input(bpy.types.NodeSocketColor, "Mask RGB")
        tree.new_input(bpy.types.NodeSocketFloat, "Mask A")
        tree.new_input(bpy.types.NodeSocketBool, "Use Color 1")
        tree.new_input(bpy.types.NodeSocketBool, "Use Color 2")
        tree.new_input(bpy.types.NodeSocketBool, "Use Color 3")
        tree.new_input(bpy.types.NodeSocketBool, "Use Color 4")
        tree.new_input(bpy.types.NodeSocketColor, "Color 1")
        tree.new_input(bpy.types.NodeSocketColor, "Color 2")
        tree.new_input(bpy.types.NodeSocketColor, "Color 3")
        tree.new_input(bpy.types.NodeSocketColor, "Color 4")

        tree.new_output(bpy.types.NodeSocketColor, "Result")

        mask_rgb = tree.add_node(bpy.types.ShaderNodeSeparateColor, name="Mask RGB")
        mask_rgb.mode = "RGB"

        mask_rgb_used = tree.add_node(
            bpy.types.ShaderNodeCombineXYZ, name="Combine RGB Used"
        )

        tree.add_link(group_inputs.outputs["Use Color 1"], mask_rgb_used.inputs["X"])
        tree.add_link(group_inputs.outputs["Use Color 2"], mask_rgb_used.inputs["Y"])
        tree.add_link(group_inputs.outputs["Use Color 3"], mask_rgb_used.inputs["Z"])

        rgb_used = tree.add_node(bpy.types.ShaderNodeVectorMath, name="Mask RGB Used")
        rgb_used.operation = "MULTIPLY"

        alpha_used = tree.add_node(bpy.types.ShaderNodeMath, name="Mask A Used")
        alpha_used.operation = "MULTIPLY"

        tree.add_link(group_inputs.outputs["Mask RGB"], rgb_used.inputs[0])
        tree.add_link(mask_rgb_used.outputs["Vector"], rgb_used.inputs[1])

        tree.add_link(group_inputs.outputs["Mask A"], alpha_used.inputs[0])
        tree.add_link(group_inputs.outputs["Use Color 4"], alpha_used.inputs[1])

        tree.add_link(rgb_used.outputs[0], mask_rgb.inputs["Color"])

        # The game composites a character's textures once, at load, in
        # sRGB space: the diffuse as stored, the colours as their 0-255
        # values over 255. Blended in linear space instead, a half-strength
        # mask moves the colour far less than it does in game, and skin
        # comes out paler and less flushed than the character creator shows
        # it. So the chain runs on encoded values and decodes at the end.
        source = tree.add_node(ShaderNodePso2SrgbEncode, name="Input sRGB")
        tree.add_link(group_inputs.outputs["Input"], source.inputs["Color"])

        factors = [
            mask_rgb.outputs["Red"],
            mask_rgb.outputs["Green"],
            mask_rgb.outputs["Blue"],
            alpha_used.outputs[0],
        ]

        # Every compositing shader squeezes its colours to 0.005..0.995.
        colors = []
        for index in range(1, 5):
            encode = tree.add_node(ShaderNodePso2SrgbEncode, name=f"Color {index} sRGB")
            tree.add_link(
                group_inputs.outputs[f"Color {index}"], encode.inputs["Color"]
            )

            squeeze = tree.add_node(
                bpy.types.ShaderNodeVectorMath, name=f"Color {index} Range"
            )
            squeeze.operation = "MULTIPLY_ADD"
            squeeze.inputs[1].default_value = (0.99, 0.99, 0.99)  # type: ignore
            squeeze.inputs[2].default_value = (0.005, 0.005, 0.005)  # type: ignore
            tree.add_link(encode.outputs["Color"], squeeze.inputs[0])
            colors.append(squeeze.outputs["Vector"])

        # Mix blends each colour over the texture in turn. Multiply (skin)
        # builds its tint the same way but from white, and only then
        # multiplies the texture by it - not a product of one tint per
        # channel, which darkens wherever two masks overlap.
        if self.operation == "MULTIPLY":
            white = tree.add_node(bpy.types.ShaderNodeRGB, name="White")
            white.outputs[0].default_value = (1, 1, 1, 1)  # type: ignore
            previous = white.outputs[0]
        else:
            previous = source.outputs["Color"]

        for index, (color, factor) in enumerate(zip(colors, factors, strict=True), 1):
            mix = tree.add_node(bpy.types.ShaderNodeMix, name=f"Color {index}")
            mix.data_type = "RGBA"
            mix.blend_type = "MIX"
            mix.clamp_factor = True
            tree.add_link(previous, mix.inputs["A"])
            tree.add_link(color, mix.inputs["B"])
            tree.add_link(factor, mix.inputs["Factor"])
            previous = mix.outputs["Result"]

        if self.operation == "MULTIPLY":
            tint = tree.add_node(bpy.types.ShaderNodeMix, name="Tint")
            tint.data_type = "RGBA"
            tint.blend_type = "MULTIPLY"
            tint.inputs["Factor"].default_value = 1  # type: ignore
            tree.add_link(source.outputs["Color"], tint.inputs["A"])
            tree.add_link(previous, tint.inputs["B"])
            previous = tint.outputs["Result"]

        result = tree.add_node(ShaderNodePso2SrgbDecode, name="Result Linear")
        tree.add_link(previous, result.inputs["Color"])
        tree.add_link(result.outputs["Color"], group_outputs.inputs["Result"])


@classes.register
class ShaderNodePso2Colorize(ShaderNodePso2ColorizeBase):
    bl_name = "ShaderNodePso2Colorize"
    bl_label = "PSO2 Colorize Mix"
    bl_icon = "NONE"

    operation = "MIX"


@classes.register
class ShaderNodePso2ColorizeMultiply(ShaderNodePso2ColorizeBase):
    bl_name = "ShaderNodePso2ColorizeMultiply"
    bl_label = "PSO2 Colorize Multiply"
    bl_icon = "NONE"

    operation = "MULTIPLY"
