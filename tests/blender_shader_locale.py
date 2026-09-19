"""Verify shader imports work when Blender localizes built-in node names."""

import os
import site
import sys
from pathlib import Path

import bpy

ROOT = Path(os.environ.get("PSO2_TEST_ADDON_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
site.addsitedir(
    str(
        Path(bpy.utils.user_resource("EXTENSIONS"))
        / ".local/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
)
if dependencies := os.environ.get("PSO2_TEST_DEPENDENCIES"):
    site.addsitedir(dependencies)

from pso2_tools import material  # noqa: E402
from pso2_tools.shaders import build_material, types  # noqa: E402

bpy.ops.preferences.addon_enable(module="pso2_tools")
language = bpy.context.preferences.view.language
try:
    for locale in ("zh_HANS", "ja_JP"):
        bpy.context.preferences.view.language = locale
        for shader in ("1110", "1117"):
            mat = bpy.data.materials.new(shader)
            data = types.ShaderData(
                material=material.Material(shaders=[shader + "p", shader]),
                textures=material.MaterialTextures(),
            )
            build_material(bpy.context, mat, data)
            assert any(node.type == "OUTPUT_MATERIAL" for node in mat.node_tree.nodes)
            assert mat.node_tree.nodes.get("Material Output") is None
            bpy.data.materials.remove(mat)
finally:
    bpy.context.preferences.view.language = language
    bpy.ops.preferences.addon_disable(module="pso2_tools")

print("LOCALIZED_SHADER_TESTS_PASSED")
