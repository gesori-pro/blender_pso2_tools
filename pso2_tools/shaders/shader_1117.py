import bpy

from . import builder, decal, shader_1102


class Shader1117(shader_1102.Shader1102):
    """NGS skin + decal shader"""

    def build(self, context):
        super().build(context)

        tree = builder.NodeTreeBuilder(self.tree)
        frame = tree.tree.nodes["Skin"]
        skin = tree.tree.nodes["PSO2 NGS Skin"]
        mix = tree.tree.nodes["Mix Shader"]
        output = tree.tree.nodes["Material Output"]
        try:
            diffuse = tree.tree.nodes["Skin Colorize"]
        except KeyError:
            diffuse = tree.tree.nodes["Colorize"]

        skin.location.x += 50 * 5
        mix.location.x += 50 * 3
        output.location.x += 50 * 3

        decal_uv = tree.add_node(bpy.types.ShaderNodeUVMap, (18, -5), name="Decal UV")
        decal_uv.parent = frame  # type: ignore
        decal_uv.uv_map = "UVChannel_3"

        decal_tex = tree.add_node(bpy.types.ShaderNodeTexImage, (24, 0), name="Decal")
        decal_tex.parent = frame  # type: ignore
        decal_tex.image = self.textures.decal.diffuse or decal.get_no_decal_image()
        decal_tex.extension = "CLIP"

        decal_mix = tree.add_node(bpy.types.ShaderNodeMix, (30, 6), name="Decal Mix")
        decal_mix.parent = frame  # type: ignore
        decal_mix.data_type = "RGBA"
        decal_mix.blend_type = "MIX"
        decal_mix.clamp_factor = True

        tree.add_link(decal_uv.outputs["UV"], decal_tex.inputs["Vector"])
        tree.add_link(diffuse.outputs["Result"], decal_mix.inputs["A"])
        tree.add_link(decal_tex.outputs["Color"], decal_mix.inputs["B"])
        tree.add_link(decal_tex.outputs["Alpha"], decal_mix.inputs["Factor"])
        tree.add_link(decal_mix.outputs["Result"], skin.inputs["Diffuse"])
