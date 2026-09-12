// Bulk accessors for the Python side of the add-on.
//
// The native model importer (pso2_tools/import_model_native.py) reads
// AquaObject/AquaNode data through Python.NET. Element-by-element property
// access crosses the interop boundary once per value, which is far too slow
// for vertex data, so everything array-shaped is flattened here into byte
// blobs that Python turns into numpy arrays in one copy.
//
// The values themselves mirror what AquaModelLibrary.Native's FbxExporterCore
// used to put into the intermediate FBX, so the Blender scene the native
// importer builds matches the one the Windows FBX path produces: UV V is
// flipped, vertex colors stay in their stored BGRA byte order, material and
// mesh names use the same metadata formats, and the bone palette is chosen
// the same way.
using AquaModelLibrary.Data.PSO2.Aqua;
using AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData;
using AquaModelLibrary.Data.PSO2.Aqua.AquaObjectData.Intermediary;
using System.Numerics;

namespace Pso2Tools.Interop;

public class NodeSet
{
    public int NodeCount;
    public int NodoCount;
    public string[] NodeNames = [];
    public string[] NodoNames = [];
    public byte[] NodeParents = [];   // int32 per node
    public byte[] NodoParents = [];   // int32 per nodo
    public byte[] NodeShorts = [];    // int32 pairs (boneShort1, boneShort2) per node
    public byte[] NodoShorts = [];    // int32 pairs per nodo
    public byte[] NodeInvBind = [];   // float32 x16 per node, System.Numerics element order
    public byte[] NodoLocal = [];     // float32 x16 per nodo, System.Numerics element order
}

public class MaterialSet
{
    public string[] Names = [];       // full metadata-format names, one per unique material
    public byte[] MeshMapping = [];   // int32 per mesh: index into Names
}

public class MeshSet
{
    public string MeshName = "";
    public int VertexCount;
    public byte[] Positions = [];     // float32 x3 per vertex
    public byte[] Normals = [];       // float32 x3 per vertex, or empty
    public byte[] Triangles = [];     // int32 x3 per triangle
    public byte[][] Uvs = [];         // 10 channels (UVChannel_1..10): float32 x2 per vertex, or empty
    public byte[] Colors = [];        // raw BGRA bytes x4 per vertex, or empty
    public byte[] Weights = [];       // float32 x4 per vertex, or empty
    public byte[] WeightIndices = []; // int32 x4 per vertex, -1 padded, or empty
    public byte[] BonePalette = [];   // int32 per palette entry
}

public static class ModelInterop
{
    public static NodeSet GetNodes(AquaNode aqn)
    {
        var result = new NodeSet
        {
            NodeCount = aqn.nodeList.Count,
            NodoCount = aqn.nodoList.Count,
        };

        var nodeNames = new string[aqn.nodeList.Count];
        var nodeParents = new int[aqn.nodeList.Count];
        var nodeShorts = new int[aqn.nodeList.Count * 2];
        var nodeInvBind = new float[aqn.nodeList.Count * 16];

        for (int i = 0; i < aqn.nodeList.Count; i++)
        {
            var node = aqn.nodeList[i];
            nodeNames[i] = aqn.GetNodeName(i);
            nodeParents[i] = node.parentId;
            nodeShorts[i * 2] = node.boneShort1;
            nodeShorts[i * 2 + 1] = node.boneShort2;
            WriteMatrix(nodeInvBind, i * 16, node.GetInverseBindPoseMatrix());
        }

        var nodoNames = new string[aqn.nodoList.Count];
        var nodoParents = new int[aqn.nodoList.Count];
        var nodoShorts = new int[aqn.nodoList.Count * 2];
        var nodoLocal = new float[aqn.nodoList.Count * 16];

        for (int i = 0; i < aqn.nodoList.Count; i++)
        {
            var nodo = aqn.nodoList[i];
            nodoNames[i] = aqn.GetNodoName(i);
            nodoParents[i] = nodo.parentId;
            nodoShorts[i * 2] = nodo.boneShort1;
            nodoShorts[i * 2 + 1] = nodo.boneShort2;
            WriteMatrix(nodoLocal, i * 16, nodo.GetLocalTransformMatrix());
        }

        result.NodeNames = nodeNames;
        result.NodeParents = ToBytes(nodeParents);
        result.NodeShorts = ToBytes(nodeShorts);
        result.NodeInvBind = ToBytes(nodeInvBind);
        result.NodoNames = nodoNames;
        result.NodoParents = ToBytes(nodoParents);
        result.NodoShorts = ToBytes(nodoShorts);
        result.NodoLocal = ToBytes(nodoLocal);

        return result;
    }

    public static MaterialSet GetMaterials(AquaObject aqo)
    {
        var materials = aqo.GetUniqueMaterials(out List<int> meshMatMapping);

        return new MaterialSet
        {
            Names = materials.Select(BuildMaterialName).ToArray(),
            MeshMapping = ToBytes(meshMatMapping.ToArray()),
        };
    }

    // Matches CreateFbxSurfacePhongFromMaterial's includeMetadata naming.
    public static string BuildMaterialName(GenericMaterial mat)
    {
        var shaders = mat.shaderNames ?? [];
        var shader0 = shaders.Count > 0 ? shaders[0] : "";
        var shader1 = shaders.Count > 1 ? shaders[1] : "";
        var specialType = string.IsNullOrEmpty(mat.specialType) ? "" : "[" + mat.specialType + "]";

        return "(" + shader0 + "," + shader1 + ")" + "{" + mat.blendType + "}" + specialType
            + mat.matName + "@" + mat.twoSided + "@" + mat.alphaCutoff;
    }

    public static int GetMeshCount(AquaObject aqo) => aqo.meshList.Count;

    public static int[] GetFaceGroupIds(AquaObject aqo, int meshId)
    {
        var groups = aqo.strips[aqo.meshList[meshId].psetIndex].faceGroups;
        return groups.Count == 0 ? [0]
            : Enumerable.Range(0, groups.Count).Where(i => groups[i] > 0).ToArray();
    }

    public static MeshSet GetMesh(AquaObject aqo, int meshId, int faceGroupId = 0)
    {
        var msh = aqo.meshList[meshId];
        var vtxl = aqo.vtxlList[msh.vsetIndex];
        var strips = aqo.strips[msh.psetIndex];
        var group = strips.faceGroups.Count > 0
            ? strips.GetTrianglesFaceGroup(faceGroupId, true)
            : new StripData.FaceGroupData { triangles = strips.GetTriangles(true) };
        if (group.vertexMappingList != null)
        {
            // AML owns the face-group remapping and vertex channel layout.
            // Copy only the group's vertices, without changing the source.
            var subset = new VTXL();
            foreach (int index in group.vertexMappingList)
                VTXL.AppendVertex(vtxl, subset, index);
            subset.bonePalette.AddRange(vtxl.bonePalette);
            vtxl = subset;
        }
        int count = vtxl.vertPositions.Count;

        var result = new MeshSet
        {
            MeshName = aqo.GetMeshName(meshId, true, faceGroupId),
            VertexCount = count,
            Positions = ToBytes(FlattenVector3(vtxl.vertPositions)),
        };

        if (vtxl.vertNormals.Count > 0)
        {
            result.Normals = ToBytes(FlattenVector3(vtxl.vertNormals));
        }

        var triangles = group.triangles;
        var tris = new int[triangles.Count * 3];
        for (int i = 0; i < triangles.Count; i++)
        {
            tris[i * 3] = (int)triangles[i].X;
            tris[i * 3 + 1] = (int)triangles[i].Y;
            tris[i * 3 + 2] = (int)triangles[i].Z;
        }
        result.Triangles = ToBytes(tris);

        result.Uvs =
        [
            FlattenUvs(vtxl.uv1List, count, fillWhenEmpty: true),
            FlattenUvs(vtxl.uv2List, count, fillWhenEmpty: false),
            FlattenUvs(vtxl.uv3List, count, fillWhenEmpty: false),
            FlattenUvs(vtxl.uv4List, count, fillWhenEmpty: false),
            FlattenUvsWithShortFallback(vtxl.uv5List, vtxl.vert0x22, count),
            FlattenUvsWithShortFallback(vtxl.uv6List, vtxl.vert0x23, count),
            FlattenUvsWithShortFallback(vtxl.uv7List, vtxl.vert0x24, count),
            FlattenUvsWithShortFallback(vtxl.uv8List, vtxl.vert0x25, count),
            FlattenColor2Uvs(vtxl.vertColor2s, 2, 1),
            FlattenColor2Uvs(vtxl.vertColor2s, 0, 3),
        ];

        if (vtxl.vertColors.Count > 0)
        {
            var colors = new byte[vtxl.vertColors.Count * 4];
            for (int i = 0; i < vtxl.vertColors.Count; i++)
            {
                var color = vtxl.vertColors[i];
                colors[i * 4] = color[0];
                colors[i * 4 + 1] = color[1];
                colors[i * 4 + 2] = color[2];
                colors[i * 4 + 3] = color[3];
            }
            result.Colors = colors;
        }

        if (vtxl.trueVertWeights.Count > 0)
        {
            var weights = new float[count * 4];
            var indices = new int[count * 4];
            Array.Fill(indices, -1);

            for (int i = 0; i < Math.Min(count, vtxl.trueVertWeights.Count); i++)
            {
                var weight = vtxl.trueVertWeights[i];
                var weightIndices = vtxl.trueVertWeightIndices[i];

                weights[i * 4] = weight.X;
                weights[i * 4 + 1] = weight.Y;
                weights[i * 4 + 2] = weight.Z;
                weights[i * 4 + 3] = weight.W;

                for (int wt = 0; wt < Math.Min(weightIndices.Length, 4); wt++)
                {
                    indices[i * 4 + wt] = weightIndices[wt];
                }
            }

            result.Weights = ToBytes(weights);
            result.WeightIndices = ToBytes(indices);

            // Same palette selection as CreateFbxNodeFromMesh.
            int[] paletteInts;
            if (aqo.objc.bonePaletteOffset > 0)
            {
                paletteInts = new int[aqo.bonePalette.Count];
                for (int i = 0; i < aqo.bonePalette.Count; i++)
                {
                    paletteInts[i] = (int)aqo.bonePalette[i];
                }
            }
            else
            {
                paletteInts = new int[vtxl.bonePalette.Count];
                for (int i = 0; i < vtxl.bonePalette.Count; i++)
                {
                    paletteInts[i] = vtxl.bonePalette[i];
                }
            }
            result.BonePalette = ToBytes(paletteInts);
        }

        return result;
    }

    private static byte[] FlattenUvs(List<Vector2> uvs, int count, bool fillWhenEmpty)
    {
        if (uvs.Count == 0)
        {
            return fillWhenEmpty ? new byte[count * 2 * sizeof(float)] : [];
        }

        var data = new float[count * 2];
        for (int i = 0; i < Math.Min(count, uvs.Count); i++)
        {
            data[i * 2] = uvs[i].X;
            data[i * 2 + 1] = 1 - uvs[i].Y;
        }
        return ToBytes(data);
    }

    private static byte[] FlattenUvsWithShortFallback(List<Vector2> uvs, List<short[]> shorts, int count)
    {
        if (uvs.Count > 0)
        {
            return FlattenUvs(uvs, count, fillWhenEmpty: false);
        }
        if (shorts.Count == 0)
        {
            return [];
        }

        var data = new float[count * 2];
        for (int i = 0; i < Math.Min(count, shorts.Count); i++)
        {
            data[i * 2] = (float)shorts[i][0] / 32767;
            data[i * 2 + 1] = (float)(1.0 - (float)shorts[i][1] / 32767);
        }
        return ToBytes(data);
    }

    // vertColor2s ride through the FBX as UVChannel_9 (b, g) and
    // UVChannel_10 (r, a), stored BGRA and not V-flipped.
    private static byte[] FlattenColor2Uvs(List<byte[]> colors, int first, int second)
    {
        if (colors.Count == 0)
        {
            return [];
        }

        var data = new float[colors.Count * 2];
        for (int i = 0; i < colors.Count; i++)
        {
            data[i * 2] = (float)colors[i][first] / 255;
            data[i * 2 + 1] = (float)colors[i][second] / 255;
        }
        return ToBytes(data);
    }

    private static float[] FlattenVector3(List<Vector3> vectors)
    {
        var data = new float[vectors.Count * 3];
        for (int i = 0; i < vectors.Count; i++)
        {
            data[i * 3] = vectors[i].X;
            data[i * 3 + 1] = vectors[i].Y;
            data[i * 3 + 2] = vectors[i].Z;
        }
        return data;
    }

    private static void WriteMatrix(float[] target, int offset, Matrix4x4 m)
    {
        target[offset] = m.M11; target[offset + 1] = m.M12; target[offset + 2] = m.M13; target[offset + 3] = m.M14;
        target[offset + 4] = m.M21; target[offset + 5] = m.M22; target[offset + 6] = m.M23; target[offset + 7] = m.M24;
        target[offset + 8] = m.M31; target[offset + 9] = m.M32; target[offset + 10] = m.M33; target[offset + 11] = m.M34;
        target[offset + 12] = m.M41; target[offset + 13] = m.M42; target[offset + 14] = m.M43; target[offset + 15] = m.M44;
    }

    private static byte[] ToBytes(float[] data)
    {
        var bytes = new byte[data.Length * sizeof(float)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static byte[] ToBytes(int[] data)
    {
        var bytes = new byte[data.Length * sizeof(int)];
        Buffer.BlockCopy(data, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
