"""Preserve rigid stage models across the FBX intermediary.

FBX cannot carry Aqua's second color channel, packed vertex declaration or
secondary stage tables. Keep a packed source in the blend and record corner
provenance, so positional edits can be applied to that source before adding
the other exported meshes. Never replace a mixed model's runtime settings
with the rigid source's settings.
"""

import base64
import hashlib
import re
import zlib
from collections import Counter

import bpy
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from . import material

SOURCE = "pso2_stage_source"
MESH_INDEX = "pso2_stage_mesh"
VERTEX_ID = "pso2_stage_vertex"
_MESH_NAME = re.compile(r"^mesh\[(\d+)\]")


def _triangles(model, mesh):
    return [
        tuple(int(getattr(t, c)) for c in "XYZ")
        for t in model.strips[mesh.psetIndex].GetTriangles(True)
    ]


def _cycle(t):
    return min(t, t[1:] + t[:1], t[2:] + t[:2])


def remember(data, objects):
    """Record only source models with rigid stage shaders, not characters."""
    from AquaModelLibrary.Data.PSO2.Aqua import AquaPackage

    model = AquaPackage(data).models[0]
    if not model.IsNGS or not model.meshList.Count:
        return
    if any(
        m.flags != 9
        or int(model.shadList[m.shadIndex].vertexShader.GetString()) >= 1000
        or model.vtxlList[m.vsetIndex].vertWeightIndices.Count
        for m in model.meshList
    ):
        return
    key = "PSO2 stage " + hashlib.sha256(bytes(data)).hexdigest()[:40]
    pending = []
    for obj in objects:
        if obj.type != "MESH" or not (match := _MESH_NAME.match(obj.name)):
            continue
        index = int(match[1])
        if index >= model.meshList.Count:
            continue
        source = model.vtxlList[model.meshList[index].vsetIndex]
        source.convertToLegacyTypes()
        tree = KDTree(source.vertPositions.Count)
        used = {i for t in _triangles(model, model.meshList[index]) for i in t}
        for i in used:
            p = source.vertPositions[i]
            tree.insert((p.X, p.Y, p.Z), i)
        tree.balance()
        ids = []
        mesh = obj.data
        for loop in mesh.loops:
            candidates = [
                i
                for _, i, _ in tree.find_range(
                    mesh.vertices[loop.vertex_index].co, 1e-5
                )
            ]
            for channel, values in (
                ("UVChannel_1", source.uv1List),
                ("UVChannel_2", source.uv2List),
            ):
                if (uv := mesh.uv_layers.get(channel)) is not None and values.Count:
                    u, v = uv.data[loop.index].uv
                    candidates = [
                        i
                        for i in candidates
                        if abs(values[i].X - u) < 1e-5
                        and abs(values[i].Y - (1 - v)) < 1e-5
                    ]
            if len(candidates) > 1 and source.vertNormals.Count:
                n = mesh.corner_normals[loop.index].vector
                candidates = [
                    i
                    for i in candidates
                    if (
                        Vector(tuple(getattr(source.vertNormals[i], c) for c in "XYZ"))
                        - n
                    ).length
                    < 1e-3
                ]
            if not candidates:
                raise ValueError(
                    f"Cannot preserve stage vertex provenance for '{obj.name}'. Import with default transforms."
                )
            if (
                len(candidates) > 1
                and source.vertColors.Count
                and mesh.color_attributes
            ):
                color = mesh.color_attributes[0]
                ci = loop.index if color.domain == "CORNER" else loop.vertex_index
                rgba = color.data[ci].color_srgb
                bgra = [round(rgba[i] * 255) for i in (2, 1, 0, 3)]
                candidates = [
                    i for i in candidates if list(source.vertColors[i]) == bgra
                ]
            if not candidates:
                raise ValueError(f"Cannot match stage vertex color on '{obj.name}'.")
            # Coincident vertices must not silently lose distinct secondary colors.
            if (
                source.vertColor2s.Count
                and len({tuple(source.vertColor2s[i]) for i in candidates}) != 1
            ):
                raise ValueError(f"Ambiguous stage vertex colors on '{obj.name}'.")
            ids.append(candidates[0])
        pending.append((obj, index, ids))
    if not pending:
        return
    if key not in bpy.data.texts:
        text = bpy.data.texts.new(key)
        text.use_fake_user = True
        text.write(base64.b64encode(zlib.compress(bytes(data))).decode("ascii"))
    for obj, index, ids in pending:
        attr = obj.data.attributes.get(VERTEX_ID) or obj.data.attributes.new(
            VERTEX_ID, "INT", "CORNER"
        )
        attr.data.foreach_set("value", ids)
        obj[SOURCE] = key
        obj[MESH_INDEX] = index


def prepare(context, objects, options):
    """Capture evaluated positional edits while the shape-key mix is active.

    Changing topology or UV seams needs remapping the auxiliary stage tables;
    reject those cases instead of silently dropping edits or writing bad data.
    """
    from AquaModelLibrary.Data.PSO2.Aqua import AquaPackage
    from System.Numerics import Vector3

    objects = list(objects)
    selected = [o for o in objects if o.get(SOURCE)]
    if not selected:
        return None
    if not options.get("apply_shape_keys", True) and any(
        o.data.shape_keys is not None for o in selected
    ):
        raise ValueError(
            "Enable Apply Shape Keys for preserved stage parts with shape keys."
        )
    keys = {o[SOURCE] for o in selected}
    if len(keys) != 1:
        raise ValueError(
            "Export one source stage model at a time, together with its added meshes."
        )
    key = keys.pop()
    text = bpy.data.texts.get(key)
    if text is None:
        raise ValueError(
            "The embedded stage source is missing. Re-import the original AQP."
        )
    package = AquaPackage(zlib.decompress(base64.b64decode(text.as_string())))
    base = package.models[0]
    found = set()
    names = set()
    edits = {}
    depsgraph = context.evaluated_depsgraph_get()
    global_matrix = options.get("global_matrix", Matrix.Identity(4))
    scale = options.get("global_scale", 1.0)
    for obj in selected:
        index = obj[MESH_INDEX]
        if index in found:
            raise ValueError(
                "Duplicated stage parts need a separate export; their source tables cannot be shared."
            )
        found.add(index)
        source_mesh = base.meshList[index]
        source = base.vtxlList[source_mesh.vsetIndex]
        source.convertToLegacyTypes()
        evaluated = (
            obj.evaluated_get(depsgraph)
            if options.get("use_mesh_modifiers", True)
            else obj
        )
        mesh = evaluated.data
        attr = mesh.attributes.get(VERTEX_ID)
        if attr is None or attr.domain != "CORNER" or len(attr.data) != len(mesh.loops):
            raise ValueError(
                f"Stage provenance was removed from '{obj.name}'. Re-import its original AQP."
            )
        ids = [v.value for v in attr.data]
        if any(i < 0 or i >= source.vertPositions.Count for i in ids):
            raise ValueError(f"Invalid stage vertex provenance on '{obj.name}'.")
        mesh.calc_loop_triangles()
        actual = Counter(
            _cycle(tuple(ids[i] for i in t.loops)) for t in mesh.loop_triangles
        )
        expected = Counter(_cycle(t) for t in _triangles(base, source_mesh))
        if actual != expected:
            raise ValueError(
                f"Topology of stage part '{obj.name}' changed. Stage preservation currently supports vertex edits and added separate meshes, not topology changes."
            )
        if len(obj.material_slots) != 1 or obj.material_slots[0].material is None:
            raise ValueError(f"Keep one source material on stage part '{obj.name}'.")
        mat = obj.material_slots[0].material
        encoded = material.get_export_material_name(mat)
        match = material.FBX_MATERIAL_RE.fullmatch(encoded)
        shad = base.shadList[source_mesh.shadIndex]
        if (
            match["shaders"]
            != f"{shad.pixelShader.GetString()},{shad.vertexShader.GetString()}"
        ):
            raise ValueError(f"Stage part '{obj.name}' must retain its source shader.")
        names.add(match["name"])
        transform = global_matrix @ evaluated.matrix_world
        for loop, vertex_id in zip(mesh.loops, ids, strict=True):
            p = (transform @ mesh.vertices[loop.vertex_index].co) * scale
            k = (source_mesh.vsetIndex, vertex_id)
            if k in edits and (edits[k] - p).length > 1e-5:
                raise ValueError(f"A shared stage vertex was split on '{obj.name}'.")
            edits[k] = p
            for channel, values in (
                ("UVChannel_1", source.uv1List),
                ("UVChannel_2", source.uv2List),
            ):
                if values.Count and mesh.uv_layers.get(channel) is None:
                    raise ValueError(
                        f"Source UV layer '{channel}' was removed from '{obj.name}'."
                    )
                if (uv := mesh.uv_layers.get(channel)) is not None and values.Count:
                    u, v = uv.data[loop.index].uv
                    if (
                        abs(values[vertex_id].X - u) > 1e-5
                        or abs(values[vertex_id].Y - (1 - v)) > 1e-5
                    ):
                        raise ValueError(
                            f"UV edits on stage part '{obj.name}' require remapping the original stage data; export cancelled."
                        )
    if found != set(range(base.meshList.Count)):
        raise ValueError(
            "Select all parts of the original stage model to preserve its auxiliary tables."
        )
    for obj in objects:
        if obj in selected:
            continue
        for slot in obj.material_slots:
            if (
                slot.material
                and material.FBX_MATERIAL_RE.fullmatch(
                    material.get_export_material_name(slot.material)
                )["name"]
                in names
            ):
                raise ValueError(
                    "Added meshes must use a separate material from the preserved stage parts."
                )
    for (vi, i), p in edits.items():
        # Preserve exact original bytes when the intermediary only rounded a coordinate.
        old = base.vtxlList[vi].vertPositions[i]
        if (Vector((old.X, old.Y, old.Z)) - p).length > 1e-6:
            base.vtxlList[vi].vertPositions[i] = Vector3(*p)
    return base, names


def merge(converted, prepared):
    """Append converted additions using AML records, preserving stage tables."""
    if prepared is None:
        return converted
    if not converted.IsNGS:
        raise ValueError("Preserved stage models require the NGS export format.")
    from System import Int32, UInt32
    from System.Collections.Generic import List

    base, stage_names = prepared
    maps = {
        n: {}
        for n in (
            "mateList",
            "rendList",
            "shadList",
            "tstaList",
            "tsetList",
            "vsetList",
        )
    }

    def copy_record(group, i):
        if i in maps[group]:
            return maps[group][i]
        source = getattr(converted, group)[i]
        record = source.Clone() if hasattr(source, "Clone") else source
        if group == "tsetList":
            record.tstaTexIDs = List[Int32]()
            for j in source.tstaTexIDs:
                record.tstaTexIDs.Add(copy_record("tstaList", j) if j >= 0 else j)
        if group == "vsetList":
            layout = converted.vtxeList[source.vtxeCount].Clone()
            record.vtxeCount = base.vtxeList.Count
            base.vtxeList.Add(layout)
            base.vtxlList.Add(converted.vtxlList[i].Clone())
        dest = getattr(base, group)
        index = dest.Count
        dest.Add(record)
        maps[group][i] = index
        return index

    for i in range(converted.meshList.Count):
        src = converted.meshList[i]
        name = str(converted.mateList[src.mateIndex].matName.GetString())
        if name in stage_names:
            continue
        m = converted.meshList[i]
        for field, group in (
            ("mateIndex", "mateList"),
            ("rendIndex", "rendList"),
            ("shadIndex", "shadList"),
            ("tsetIndex", "tsetList"),
            ("vsetIndex", "vsetList"),
        ):
            setattr(m, field, copy_record(group, getattr(src, field)))
        m.psetIndex = base.psetList.Count
        base.psetList.Add(converted.psetList[src.psetIndex])
        base.strips.Add(converted.strips[src.psetIndex].Clone())
        base.meshList.Add(m)
    # Original stage meshes use rigid node IDs. Do not transplant a different rig.
    if any(m.baseMeshNodeId != 0 for m in base.meshList):
        raise ValueError("Stage preservation currently requires a single root bone.")
    if any(int(x) != 0 for x in converted.bonePalette):
        raise ValueError("Added stage meshes must be bound to the stage root bone.")
    base.bonePalette = List[UInt32](converted.bonePalette)
    names = {str(t.texName.GetString()) for t in base.texfList}
    for t in converted.texfList:
        if str(t.texName.GetString()) not in names:
            base.texfList.Add(t)
            names.add(str(t.texName.GetString()))
    header = base.objc
    # Runtime-tested: copying the rigid source value here makes weighted additions
    # follow the camera/player. Retain AML's value for the actual exported scene.
    header.unkMeshValue = converted.objc.unkMeshValue
    header.largetsVtxl = max(v.GetVTXESize() for v in base.vtxeList)
    count = 0
    for i in range(base.vsetList.Count):
        v = base.vsetList[i]
        v.vertDataSize = header.largetsVtxl
        v.vtxlStartVert = count
        count += v.vtxlCount
        base.vsetList[i] = v
    header.totalVTXLCount = count
    count = 0
    for i in range(base.psetList.Count):
        p = base.psetList[i]
        p.stripStartCount = count
        count += base.strips[i].triStrips.Count
        base.psetList[i] = p
    header.totalStripFaces = count
    for name in (
        "vset",
        "pset",
        "mesh",
        "mate",
        "rend",
        "shad",
        "tsta",
        "tset",
        "texf",
        "vtxe",
    ):
        setattr(header, name + "Count", getattr(base, name + "List").Count)
    base.objc = header
    return base
