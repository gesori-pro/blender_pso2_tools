// Non-Windows stand-in for the C++/CLI AquaModelLibrary.Native assembly.
// The Autodesk FBX SDK bridge is C++/CLI, which only exists on Windows; on
// other platforms Directory.Build.targets compiles this stub into
// AquaModelLibrary.Core instead so everything that mentions the type still
// builds. The Blender add-on never calls it off Windows - model import goes
// through pso2_tools/import_model_native.py there.
using AquaModelLibrary.Data.PSO2.Aqua;
using System.Numerics;

namespace AquaModelLibrary.Objects.Processing.Fbx
{
    public class FbxExporterCore
    {
        public virtual void ExportToFile(AquaObject aqo, AquaNode aqn, List<AquaMotion> aqmList, string destinationFilePath, List<string> aqmNameList, List<Matrix4x4> instanceTransforms, bool includeMetadata, int coordSystem)
            => throw new PlatformNotSupportedException("FBX export requires the Windows-only AquaModelLibrary.Native assembly.");

        public virtual void ExportToFileSets(List<AquaObject> aqoList, List<AquaNode> aqnList, List<string> modelNames, string destinationFilePath, List<List<Matrix4x4>> instanceTransformsList, bool includeMetadata, int coordSystem)
            => throw new PlatformNotSupportedException("FBX export requires the Windows-only AquaModelLibrary.Native assembly.");
    }
}
