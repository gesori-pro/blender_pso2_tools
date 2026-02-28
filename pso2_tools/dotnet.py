from pathlib import Path
import sys

import clr_loader
import pythonnet

from .paths import BIN_PATH

_DLL_NAMES = [
    "AssimpNet.dll",
    "AquaModelLibrary.Core.dll",
    "AquaModelLibrary.Data.dll",
    "AquaModelLibrary.Helpers.dll",
    "ZamboniLib.dll",
]

_PROBING_PATH_X64 = str(BIN_PATH / "x64")


def _get_dotnet_roots() -> list[Path]:
    if sys.platform == "win32":
        return [Path("C:/Program Files/dotnet")]

    if sys.platform == "darwin":
        return [
            Path("/opt/homebrew/share/dotnet"),
            Path("/usr/local/share/dotnet"),
        ]

    return [Path("/usr/share/dotnet"), Path("/usr/local/share/dotnet")]


def _get_native_probing_paths() -> list[str]:
    runtime_dirs = [
        BIN_PATH,
        BIN_PATH / "x64",
        BIN_PATH / "runtimes" / "win-x64" / "native",
        BIN_PATH / "runtimes" / "osx-arm64" / "native",
        BIN_PATH / "runtimes" / "osx-x64" / "native",
        BIN_PATH / "runtimes" / "linux-x64" / "native",
    ]

    return [str(path) for path in runtime_dirs if path.exists()]

_loaded = False


def load():
    global _loaded
    if _loaded:
        return

    dotnet_root = next((p for p in _get_dotnet_roots() if p.exists()), None)
    if dotnet_root:
        rt = clr_loader.get_coreclr(dotnet_root=dotnet_root)
        pythonnet.load(rt)
    else:
        pythonnet.load("coreclr")

    import clr

    for name in _DLL_NAMES:
        path = str(BIN_PATH / name)
        clr.AddReference(path)  # type: ignore

    from Assimp.Unmanaged import AssimpLibrary

    resolver = AssimpLibrary.Instance.Resolver
    probing_paths = _get_native_probing_paths() or [_PROBING_PATH_X64]

    if hasattr(resolver, "SetProbingPaths64"):
        resolver.SetProbingPaths64(probing_paths)
    elif hasattr(resolver, "SetProbingPaths"):
        resolver.SetProbingPaths(probing_paths)

    _loaded = True
