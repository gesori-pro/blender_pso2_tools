"""Bake a PSO2 material down to one diffuse, normal and multi map.

A PSO2 material is not the textures it ships. The diffuse in the game data
is close to a white sheet: skin tone, outfit colours and eye colour are a
mask plus a colorize step the game runs at draw time, and over that a face
carries its paints, a body mixes its innerwear in, and a muscular character
blends a second texture set on top of the first. Pulling the .dds out of the
game data and using it as-is is what gives the pale, plastic-looking skin -
the file really does look like that.

Baking collapses the whole chain into three images that read correctly in
any renderer, one set per material, leaving the UVs and the blend mode
alone.

The diffuse goes through Cycles' own diffuse-colour pass, which follows a
Mix Shader, so a body that mixes skin and innerwear bakes as what is really
drawn. Alpha, normal and multi map have no such pass, so those are read off
the shader group's inputs through a temporary emission node.
"""

from pathlib import Path

import bpy

from . import classes
from .debug import debug_print
from .util import OperatorResult

# Shader-group inputs that have no bake pass of their own.
_ALPHA_INPUT = "Alpha"
_EMIT_INPUTS = {
    "n": "Normal",
    "s": "Multi RGB",
}

# Maps that are data rather than colour, and must not be gamma corrected.
_DATA_MAPS = {"n", "s"}

_TARGET_NODE = "PSO2 Bake Target"
_EMISSION_NODE = "PSO2 Bake Emission"


def _output_node(tree: bpy.types.NodeTree):
    for node in tree.nodes:
        if node.type == "OUTPUT_MATERIAL" and node.inputs["Surface"].links:
            return node
    return None


def _shader_group(tree: bpy.types.NodeTree):
    """The PSO2 shader group a material is built around.

    A skin material mixes two of them, one for the skin and one for the
    innerwear layer over it. The first is the one describing the surface the
    model is mostly made of, so that is the one read here.
    """
    for node in tree.nodes:
        if node.bl_idname.startswith("ShaderNodePso2") and "BSDF" in node.outputs:
            return node
    return None


def _source_size(material: bpy.types.Material) -> tuple[int, int]:
    """The largest source texture in a material, or a sane default."""
    sizes = [
        (int(node.image.size[0]), int(node.image.size[1]))
        for node in material.node_tree.nodes
        if node.type == "TEX_IMAGE" and node.image and all(node.image.size)
    ]
    if not sizes:
        return 1024, 1024
    return max(sizes, key=lambda size: size[0] * size[1])


def _make_target(material: bpy.types.Material, suffix: str, size, alpha: bool):
    name = f"{material.name}_baked_{suffix}"
    image = bpy.data.images.get(name)
    if image is not None and tuple(image.size) != tuple(size):
        bpy.data.images.remove(image)
        image = None
    if image is None:
        image = bpy.data.images.new(name, size[0], size[1], alpha=alpha)
    image.colorspace_settings.name = "Non-Color" if suffix in _DATA_MAPS else "sRGB"
    return image


def _target_node(material: bpy.types.Material, image: bpy.types.Image):
    """A texture node holding `image`, made active so the bake lands in it."""
    tree = material.node_tree
    node = tree.nodes.get(_TARGET_NODE)
    if node is None:
        node = tree.nodes.new("ShaderNodeTexImage")
        node.name = node.label = _TARGET_NODE
        node.location = (0, 600)
    node.image = image
    for other in tree.nodes:
        other.select = False
    node.select = True
    tree.nodes.active = node
    return node


def _saved_surface(material: bpy.types.Material):
    output = _output_node(material.node_tree)
    if output is None:
        return None
    links = output.inputs["Surface"].links
    return links[0].from_socket if links else None


def _route_to_emission(material: bpy.types.Material, input_name: str) -> bool:
    """Send one shader-group input to the surface output on its own.

    Returns whether there was anything to send; `_restore` puts the
    original link back.
    """
    tree = material.node_tree
    output = _output_node(tree)
    group = _shader_group(tree)
    if output is None or group is None:
        return False

    socket = group.inputs.get(input_name)
    if socket is None or not socket.links:
        return False

    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = emission.label = _EMISSION_NODE
    emission.location = (output.location.x - 200, output.location.y + 300)
    tree.links.new(socket.links[0].from_socket, emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return True


def _restore(material: bpy.types.Material, saved) -> None:
    tree = material.node_tree
    emission = tree.nodes.get(_EMISSION_NODE)
    if emission is not None:
        tree.nodes.remove(emission)
    output = next((n for n in tree.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if output is not None and saved is not None:
        tree.links.new(saved, output.inputs["Surface"])


def _materials_of(objects) -> list[bpy.types.Material]:
    found: list[bpy.types.Material] = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for material in obj.data.materials:
            if material and material.use_nodes and material not in found:
                found.append(material)
    return found


def _write_alpha(diffuse: bpy.types.Image, mask: bpy.types.Image) -> None:
    """Move a baked alpha pass into the diffuse image's alpha channel.

    The face's neck skirt fades out along its bottom edge through the
    diffuse alpha, so a diffuse baked without it loses that join.
    """
    import numpy as np

    if tuple(mask.size) != tuple(diffuse.size):
        return

    count = diffuse.size[0] * diffuse.size[1] * 4
    rgba = np.empty(count, dtype=np.float32)
    diffuse.pixels.foreach_get(rgba)
    values = np.empty(count, dtype=np.float32)
    mask.pixels.foreach_get(values)

    rgba[3::4] = values[0::4]
    diffuse.pixels.foreach_set(rgba)
    diffuse.update()


@classes.register
class PSO2_OT_BakeTextures(bpy.types.Operator):  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    """Bake the selected objects' PSO2 materials into one diffuse, normal and
    multi map each, with the colours and layers the game applies at draw time
    already in them"""

    bl_label = "Bake PSO2 Textures"
    bl_idname = "pso2.bake_textures"
    bl_options = {"REGISTER", "UNDO"}

    bake_diffuse: bpy.props.BoolProperty(
        name="Diffuse",
        description=(
            "Bake the colour the game actually draws: the mask and colorize"
            " step, face paints, innerwear and the muscle blend"
        ),
        default=True,
    )
    bake_alpha: bpy.props.BoolProperty(
        name="Diffuse Alpha",
        description=(
            "Put the material's alpha into the baked diffuse. The face's neck"
            " skirt fades out through it"
        ),
        default=True,
    )
    bake_normal: bpy.props.BoolProperty(
        name="Normal",
        description="Bake the normal map, including the muscle blend",
        default=True,
    )
    bake_multi: bpy.props.BoolProperty(
        name="Multi Map",
        description="Bake the multi map, PSO2's packed specular set",
        default=True,
    )
    margin: bpy.props.IntProperty(
        name="Margin",
        description="Pixels to bleed past each UV island",
        default=16,
        min=0,
        soft_max=64,
    )
    save_to: bpy.props.StringProperty(
        name="Save To",
        description=(
            "Folder to write the baked images to as PNG. Leave empty to keep"
            " them in the blend file only"
        ),
        subtype="DIR_PATH",
        default="",
    )

    @classmethod
    def poll(cls, context):
        return any(obj.type == "MESH" for obj in context.selected_objects)

    def execute(self, context) -> OperatorResult:
        objects = [obj for obj in context.selected_objects if obj.type == "MESH"]
        materials = _materials_of(objects)
        if not materials:
            self.report({"ERROR"}, "No node materials on the selected objects")
            return {"CANCELLED"}

        scene = context.scene
        previous_engine = scene.render.engine
        previous_samples = getattr(scene.cycles, "samples", None)

        scene.render.engine = "CYCLES"
        scene.cycles.samples = 1
        scene.render.bake.margin = self.margin
        scene.render.bake.use_clear = True

        baked: dict[str, list[str]] = {}
        try:
            if self.bake_diffuse:
                baked["d"] = self._bake_diffuse(context, materials)
            if self.bake_normal:
                baked["n"] = self._bake_emit(context, materials, "n")
            if self.bake_multi:
                baked["s"] = self._bake_emit(context, materials, "s")
        finally:
            for material in materials:
                node = material.node_tree.nodes.get(_TARGET_NODE)
                if node is not None:
                    material.node_tree.nodes.remove(node)
            scene.render.engine = previous_engine
            if previous_samples is not None:
                scene.cycles.samples = previous_samples

        if self.save_to:
            self._save(baked)

        total = sum(len(names) for names in baked.values())
        counts = ", ".join(f"{key}:{len(names)}" for key, names in baked.items())
        self.report(
            {"INFO"},
            f"Baked {total} images for {len(materials)} materials ({counts})",
        )
        return {"FINISHED"}

    def _bake_diffuse(self, context, materials) -> list[str]:
        """Cycles' diffuse-colour pass, which follows a Mix Shader."""
        images = {}
        for material in materials:
            image = _make_target(material, "d", _source_size(material), alpha=True)
            _target_node(material, image)
            images[material.name] = image

        bake = context.scene.render.bake
        bake.use_pass_direct = False
        bake.use_pass_indirect = False
        bake.use_pass_color = True
        bpy.ops.object.bake(type="DIFFUSE")

        if self.bake_alpha:
            masks = self._bake_alpha(context, materials)
            for name, image in images.items():
                mask = masks.get(name)
                if mask is not None:
                    _write_alpha(image, mask)
                    bpy.data.images.remove(mask)

        return [image.name for image in images.values()]

    def _bake_alpha(self, context, materials) -> dict:
        masks = {}
        saved = {}
        for material in materials:
            saved[material.name] = _saved_surface(material)
            if not _route_to_emission(material, _ALPHA_INPUT):
                continue
            image = _make_target(material, "alpha", _source_size(material), alpha=False)
            _target_node(material, image)
            masks[material.name] = image

        if masks:
            bpy.ops.object.bake(type="EMIT")

        for material in materials:
            _restore(material, saved.get(material.name))
        return masks

    def _bake_emit(self, context, materials, suffix: str) -> list[str]:
        images = {}
        saved = {}
        for material in materials:
            saved[material.name] = _saved_surface(material)
            if not _route_to_emission(material, _EMIT_INPUTS[suffix]):
                debug_print(f"{material.name}: nothing feeding {_EMIT_INPUTS[suffix]}")
                continue
            image = _make_target(material, suffix, _source_size(material), alpha=False)
            _target_node(material, image)
            images[material.name] = image

        if images:
            bpy.ops.object.bake(type="EMIT")

        for material in materials:
            _restore(material, saved.get(material.name))
        return [image.name for image in images.values()]

    def _save(self, baked: dict[str, list[str]]) -> None:
        folder = Path(bpy.path.abspath(self.save_to))
        folder.mkdir(parents=True, exist_ok=True)
        for names in baked.values():
            for name in names:
                image = bpy.data.images.get(name)
                if image is None:
                    continue
                image.filepath_raw = str(folder / f"{name}.png")
                image.file_format = "PNG"
                image.save()
