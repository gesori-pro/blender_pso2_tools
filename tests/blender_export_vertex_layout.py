"""Run with the add-on enabled in background Blender; uses real AML types."""

import os
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
import blender_export_shape_keys  # noqa: F401

bpy.ops.preferences.addon_enable(
    module=os.environ.get("PSO2_TEST_ADDON_MODULE", "pso2_tools")
)

from pso2_tools.export_model import strip_padded_uvs  # noqa: E402
from AquaModelLibrary.Data.PSO2.Aqua import AquaObject  # noqa: E402
from AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData import VTXE, VTXL, VSET  # noqa: E402
from System.Numerics import Vector2, Vector3  # noqa: E402


def reproduce_old_failure():
    vertices = VTXL()
    vertices.vertPositions.Add(Vector3(0, 0, 0))
    vertices.uv1List.Add(Vector2(0.2, 0.3))
    vertices.uv2List.Add(Vector2(0, 0))
    layout, stride = VTXE.ConstructFromVTXL(vertices)
    vertices.uv2List.Clear()
    try:
        vertices.GetBytes(layout, stride)
    except Exception as error:
        assert "ArgumentOutOfRangeException" in str(type(error)), type(error)
    else:
        raise AssertionError("Original empty-UV/layout mismatch did not reproduce")


def check(ngs):
    model = AquaObject()
    header = model.objc
    header.type = 0xC33 if ngs else 0xC2A
    model.objc = header
    for populated in (False, True):
        vertices = VTXL()
        for i in range(2):
            vertices.vertPositions.Add(Vector3(i, 0, 0))
            vertices.uv1List.Add(Vector2(0.2, 0.3))
            vertices.uv2List.Add(Vector2(0.5 if populated else 0, 0))
            # A later nonzero channel must stay at its semantic index.
            vertices.uv3List.Add(Vector2(0.7, 0.8))
        layout, stride = VTXE.ConstructFromVTXL(vertices)
        vset = VSET()
        vset.vtxlCount = 2
        vset.vertDataSize = stride
        vset.vtxeCount = 0 if ngs else layout.vertDataTypes.Count
        model.vtxlList.Add(vertices)
        model.vsetList.Add(vset)
        if not ngs or not model.vtxeList.Count:
            model.vtxeList.Add(layout)
    header = model.objc
    header.largetsVtxl = stride
    header.vtxeCount = model.vtxeList.Count
    model.objc = header
    original_layout = model.vtxeList[0]
    assert strip_padded_uvs(model) == 1
    for i in range(2):
        vset = model.vsetList[i]
        layout = model.vtxeList[vset.vtxeCount if ngs else i]
        flags = [e.dataType for e in layout.vertDataTypes]
        assert (0x11 in flags) == (i == 1)
        assert 0x12 in flags
        assert model.vtxlList[i].uv3List.Count == 2
        output = model.vtxlList[i].GetBytes(
            layout, model.objc.largetsVtxl if ngs else vset.vertDataSize
        )
        assert len(output) == 2 * (stride if ngs else vset.vertDataSize)
    if ngs:
        assert 0x11 in [e.dataType for e in original_layout.vertDataTypes]
        assert model.objc.vtxeCount == model.vtxeList.Count
    assert strip_padded_uvs(model) == 0


reproduce_old_failure()
check(True)
check(False)
print("VERTEX_LAYOUT_TESTS_PASSED: NGS shared layout, classic stride, nonzero UVs")
