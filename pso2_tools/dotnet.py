import sys
from pathlib import Path

import clr_loader
import pythonnet

from .paths import BIN_PATH

_DLL_NAMES = [
    "AquaModelLibrary.Core.dll",
    "AquaModelLibrary.Data.dll",
    "AquaModelLibrary.Helpers.dll",
    "SharpAssimp.dll",
    "ZamboniLib.dll",
]

# Bulk data access for the native model importer. Not present in a bin/
# folder built before it existed, and nothing else needs it, so its absence
# only turns off the native import path.
_INTEROP_DLL = "Pso2Tools.Interop.dll"

_PROBING_PATH_X64 = str(BIN_PATH / "x64")

if sys.platform == "win32":
    _DOTNET_ROOTS = [Path("C:/Program Files/dotnet")]
else:
    # Blender is a GUI app, so its PATH is too bare to find dotnet with.
    # These cover the official installer, the install script, and Homebrew.
    _DOTNET_ROOTS = [
        Path("/usr/local/share/dotnet"),
        Path.home() / ".dotnet",
        Path("/opt/homebrew/opt/dotnet/libexec"),
        Path("/usr/local/opt/dotnet/libexec"),
    ]

_loaded = False
_probing_paths_set = False


def _find_dotnet_root():
    return next((root for root in _DOTNET_ROOTS if root.exists()), None)


def load():
    global _loaded
    if _loaded:
        return

    try:
        if dotnet_root := _find_dotnet_root():
            rt = clr_loader.get_coreclr(dotnet_root=dotnet_root)
            pythonnet.load(rt)
        else:
            pythonnet.load("coreclr")
    except RuntimeError:
        # The runtime is already loaded, e.g. when the extension is
        # re-enabled after an update. It cannot be unloaded, but referencing
        # assemblies again is harmless.
        pass

    import clr

    names = list(_DLL_NAMES)
    if (BIN_PATH / _INTEROP_DLL).exists():
        names.append(_INTEROP_DLL)

    if sys.platform == "win32":
        for name in names:
            clr.AddReference(str(BIN_PATH / name))  # type: ignore
    else:
        # Python.NET 3.0's AddReference reads a POSIX path as an assembly
        # name and fails; LoadFrom takes paths everywhere and resolves the
        # assemblies' dependencies from the same folder.
        from System.Reflection import Assembly

        for name in names:
            Assembly.LoadFrom(str(BIN_PATH / name))

        _fix_windows_path_constants()

    _loaded = True


def _fix_windows_path_constants():
    """Rewrite AquaModelLibrary's hardcoded backslash game paths.

    CharacterMakingIndex joins pso2_bin paths with "data\\win32\\" style
    constants. Windows accepts either separator, but on other systems the
    backslashes become part of the file name, every File.Exists comes back
    false, and ReferenceGenerator quietly returns nulls - the model
    database then has no cmx and no item names. The fields are static and
    writable, so they can be corrected in place.
    """
    from AquaModelLibrary.Data.PSO2.Aqua import CharacterMakingIndex

    CharacterMakingIndex.dataDirPC = "data/win32/"
    CharacterMakingIndex.dataNADirPC = "data/win32_na/"
    CharacterMakingIndex.dataRebootPC = "data/win32reboot/"
    CharacterMakingIndex.dataRebootNAPC = "data/win32reboot_na/"


def has_fbx_converter() -> bool:
    """Whether the C++/CLI FBX bridge is present.

    It is Windows-only: model import goes .aqp -> FBX -> Blender's FBX
    importer there, and through import_model_native everywhere else.
    """
    return sys.platform == "win32" and (BIN_PATH / "AquaModelLibrary.Native.dll").exists()


def has_interop() -> bool:
    return (BIN_PATH / _INTEROP_DLL).exists()


def set_assimp_probing_paths():
    global _probing_paths_set
    if _probing_paths_set:
        return

    from SharpAssimp.Unmanaged import AssimpLibrary

    if sys.platform == "win32":
        AssimpLibrary.Instance.Resolver.SetProbingPaths64([_PROBING_PATH_X64])
    else:
        # dotnet publish drops libassimp for this machine into bin/ itself.
        AssimpLibrary.Instance.Resolver.SetProbingPaths64([str(BIN_PATH)])

    _probing_paths_set = True
