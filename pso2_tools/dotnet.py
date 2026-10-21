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

_PROBING_PATH_X64 = str(BIN_PATH / "x64")

_DOTNET_ROOT = Path("C:/Program Files/dotnet")
_RUNTIME_CONFIG = Path(__file__).with_name("dotnet.runtimeconfig.json")

_loaded = False
_probing_paths_set = False


def load():
    global _loaded
    if _loaded:
        return

    try:
        # Without a config, clr_loader asks the `dotnet` on PATH for the
        # newest runtime and demands exactly that one: a .NET 11 preview, or
        # one installed outside this root, then fails with 0x80008096
        # (FrameworkMissingFailure). The libraries target net9.0, so ask for
        # 9.0 and let the host roll forward to a newer major only without it.
        if _DOTNET_ROOT.exists():
            rt = clr_loader.get_coreclr(runtime_config=_RUNTIME_CONFIG, dotnet_root=_DOTNET_ROOT)
            pythonnet.load(rt)
        else:
            pythonnet.load("coreclr", runtime_config=_RUNTIME_CONFIG)
    except RuntimeError:
        # The runtime is already loaded, e.g. when the extension is
        # re-enabled after an update. It cannot be unloaded, but referencing
        # assemblies again is harmless.
        pass

    import clr

    for name in _DLL_NAMES:
        path = str(BIN_PATH / name)
        clr.AddReference(path)  # type: ignore

    _loaded = True


def set_assimp_probing_paths():
    global _probing_paths_set
    if _probing_paths_set:
        return

    from SharpAssimp.Unmanaged import AssimpLibrary

    AssimpLibrary.Instance.Resolver.SetProbingPaths64([_PROBING_PATH_X64])

    _probing_paths_set = True
