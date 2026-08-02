import bpy

from ..colors import ColorMapping
from . import attributes, builder
from .colorize import ShaderNodePso2Colorize, ShaderNodePso2ColorizeMultiply
from .colors import ShaderNodePso2Colorchannels
from .ngs import ShaderNodePso2Ngs, ShaderNodePso2NgsSkin


class Shader1101(builder.ShaderBuilder):
    """NGS horn/tooth/anime face shader"""

    @property
    def textures(self):
        return self.data.textures

    @property
    def colors(self) -> ColorMapping:
        return self.data.color_map or ColorMapping()

    def build(self, context):
        is_face = self.data.material.special_type == "fc"

        tree = self.init_tree()

        output = tree.add_node(bpy.types.ShaderNodeOutputMaterial, (24, 6))

        if is_face:
            shader_group = tree.add_node(ShaderNodePso2NgsSkin, (18, 6))
        else:
            shader_group = tree.add_node(ShaderNodePso2Ngs, (18, 6))

        attributes.add_alpha_threshold(
            target=shader_group.inputs["Alpha Threshold"],
            material=self.material,
        )

        tree.add_link(shader_group.outputs["BSDF"], output.inputs["Surface"])

        # Diffuse
        diffuse = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 18), name="Diffuse")
        diffuse.image = self.textures.default.diffuse

        tree.add_link(diffuse.outputs["Alpha"], shader_group.inputs["Alpha"])

        # Color Mask
        mask = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 12), name="Color Mask")
        mask.image = self.textures.default.mask

        if is_face:
            colorize = tree.add_node(ShaderNodePso2ColorizeMultiply, (12, 14))
        else:
            colorize = tree.add_node(ShaderNodePso2Colorize, (12, 14))

        tree.add_link(diffuse.outputs["Color"], colorize.inputs["Input"])
        tree.add_link(mask.outputs["Color"], colorize.inputs["Mask RGB"])
        tree.add_link(mask.outputs["Alpha"], colorize.inputs["Mask A"])
        tree.add_link(colorize.outputs["Result"], shader_group.inputs["Diffuse"])

        channels = tree.add_node(ShaderNodePso2Colorchannels, (7, 10), name="Colors")

        tree.add_color_link(self.colors.red, channels, colorize.inputs["Color 1"])
        tree.add_color_link(self.colors.green, channels, colorize.inputs["Color 2"])
        tree.add_color_link(self.colors.blue, channels, colorize.inputs["Color 3"])
        tree.add_color_link(self.colors.alpha, channels, colorize.inputs["Color 4"])
        colorize.set_colors_used(self.colors)

        # Multi Map
        multi = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 6), name="Multi Map")
        multi.image = self.textures.default.multi

        tree.add_link(multi.outputs["Color"], shader_group.inputs["Multi RGB"])
        tree.add_link(multi.outputs["Alpha"], shader_group.inputs["Multi A"])

        # Normal Map
        normal = tree.add_node(bpy.types.ShaderNodeTexImage, (0, 0), name="Normal Map")
        normal.image = self.textures.default.normal

        tree.add_link(normal.outputs["Color"], shader_group.inputs["Normal"])
