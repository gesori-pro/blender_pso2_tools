#! /usr/bin/env python3
"""
Build .net dependencies.

On Windows this drives Visual Studio's MSBuild so the C++/CLI FBX bridge
(AquaModelLibrary.Native) builds alongside the managed code. On macOS that
bridge cannot exist - C++/CLI is Windows-only - so the managed stack is
published with the dotnet CLI instead (Directory.Build.targets swaps the
bridge for a stub), and an ooz build stands in for the Oodle DLL that
ZamboniLib expects for NGS ICE decompression.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

FBX_URL = "https://aps.autodesk.com/developer/overview/fbx-sdk"
VISUAL_STUDIO_URL = "https://visualstudio.microsoft.com/vs/community/"

VSWHERE = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")

FBX_INSTALL_ROOT = Path("C:/Program Files/Autodesk/FBX/FBX SDK")
FBX_DEST = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.Native/Dependencies/FBX"
BIN_PATH = ROOT / "pso2_tools/bin"

INTEROP_PROJECT = ROOT / "dotnet/Pso2Tools.Interop/Pso2Tools.Interop.csproj"

STUB_GENERATOR_SLN = ROOT / "pythonnet-stub-generator/csharp/PythonNetStubGenerator.sln"

PACKAGES_PATH = ROOT / "packages"

# .NET Framework 4.8 reference assemblies, needed to compile the legacy-format
# NvTriStripDotNet project where no Visual Studio provides them.
NET48_REFS_PACKAGE = "microsoft.netframework.referenceassemblies.net48"
NET48_REFS_VERSION = "1.0.3"
NET48_REFS_URL = (
    "https://api.nuget.org/v3-flatcontainer/"
    f"{NET48_REFS_PACKAGE}/{NET48_REFS_VERSION}/{NET48_REFS_PACKAGE}.{NET48_REFS_VERSION}.nupkg"
)
NET48_REFS_PATH = PACKAGES_PATH / f"{NET48_REFS_PACKAGE}.{NET48_REFS_VERSION}"

# Open-source Oodle decompressor. ZamboniLib P/Invokes oo2core_8_win64_ to
# unpack NGS ICE archives; dotnet/ooz/oodle_shim.cpp wraps ooz behind that ABI.
OOZ_REPO = "https://github.com/powzix/ooz.git"
OOZ_COMMIT = "05038060aa68f9187ae9923b2388ca8db40e58d1"
OOZ_SRC = PACKAGES_PATH / "ooz"
OOZ_SHIM_DIR = ROOT / "dotnet/ooz"
OOZ_DYLIB = "liboo2core_8_win64_.dylib"

SSE2NEON_COMMIT = "13a42df35dc7fcc94f987568e7274a998bb6cc86"
SSE2NEON_URL = (
    f"https://raw.githubusercontent.com/DLTcollab/sse2neon/{SSE2NEON_COMMIT}/sse2neon.h"
)


def check_dependencies():
    if not VSWHERE.exists():
        print(f"Please install Visual Studio: {VISUAL_STUDIO_URL}")
        sys.exit(1)


def find_fbx_sdk(requested: Path | None) -> Path:
    def usable(path):
        return (path / "include/fbxsdk.h").is_file() and any(
            (path / layout / "libfbxsdk-md.lib").is_file()
            for layout in ("lib/vs2017/x64/release", "lib/x64/release")
        )

    # Respect already configured junctions, including newer 2020.3 SDKs.
    if requested is None and usable(FBX_DEST):
        return FBX_DEST
    candidates = (
        [requested]
        if requested
        else sorted(
            FBX_INSTALL_ROOT.glob("2020.*"),
            key=lambda p: tuple(int(part) for part in p.name.split(".")),
            reverse=True,
        )
    )
    for path in candidates:
        if usable(path):
            return path
    raise RuntimeError(f"Install an FBX 2020 SDK or pass --fbx-sdk: {FBX_URL}")


def configure_fbx_sdk(requested: Path | None) -> None:
    source = find_fbx_sdk(requested)
    if source == FBX_DEST:
        return
    for name in ("lib", "include"):
        dest = FBX_DEST / name
        src = (source / name).resolve()
        if dest.exists():
            if dest.resolve() != src:
                raise RuntimeError(f"{dest} already points elsewhere; check --fbx-sdk")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)

        # PowerShell avoids cmd's parsing of metacharacters in SDK paths.
        def quote(path):
            return "'" + str(path).replace("'", "''") + "'"

        subprocess.check_call(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"New-Item -ItemType Junction -Path {quote(dest)} -Target {quote(src)} | Out-Null",
            ]
        )


def finish_bin(stage: Path, debug: bool, rid: str):
    """Validate a complete staged publish before replacing the previous bin."""
    required = [
        "AquaModelLibrary.Core.dll",
        "AquaModelLibrary.Data.dll",
        "Pso2Tools.Interop.dll",
        "Pso2Tools.Interop.deps.json",
        "ZamboniLib.dll",
    ]
    if rid == "win-x64":
        required += ["AquaModelLibrary.Native.X64.dll", "Ijwhost.dll", "assimp.dll"]
    else:
        required += [OOZ_DYLIB]
    for name in required:
        if not (stage / name).is_file():
            raise RuntimeError(f"Incomplete publish: missing {name}")
    if not debug:
        for pdb in stage.rglob("*.pdb"):
            pdb.unlink()
    revision = subprocess.check_output(
        ["git", "-C", ROOT / "PSO2-Aqua-Library", "rev-parse", "HEAD"], text=True
    ).strip()
    provenance = {
        "aml_commit": revision,
        "runtime": rid,
        "source_compatibility": "dotnet/AmlCompatibility.targets",
        "dll_sha256": {
            file.relative_to(stage).as_posix(): hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
            for file in sorted(stage.rglob("*.dll"))
        },
    }
    (stage / "build-info.json").write_text(json.dumps(provenance, indent=2) + "\n")
    # Both paths belong to our workspace. Never follow an installed add-on's
    # junction, and retain the entire old bin if publication/rename fails.
    if BIN_PATH.resolve() != ROOT.resolve() / "pso2_tools/bin":
        raise RuntimeError(f"Refusing to replace redirected bin: {BIN_PATH}")
    previous = stage.parent / "previous-bin"
    had_bin = BIN_PATH.exists()
    if had_bin:
        BIN_PATH.rename(previous)
    try:
        stage.rename(BIN_PATH)
    except OSError:
        if had_bin:
            previous.rename(BIN_PATH)
        raise


def call_msbuild(args: list[Path | str]):
    vs = json.loads(
        subprocess.check_output(
            # vswhere localizes text in the console code page, which need
            # not be valid UTF-8. Every field read here is ASCII.
            [VSWHERE, "-latest", "-format", "json"],
            encoding="utf-8",
            errors="replace",
        )
    )
    msbuild = Path(vs[0]["installationPath"]) / "Msbuild/Current/Bin/MSBuild.exe"

    subprocess.check_call([msbuild, *args], cwd=ROOT)


def build_windows(args):
    check_dependencies()

    config = "Debug" if args.debug else "Release"
    configure_fbx_sdk(args.fbx_sdk)
    PACKAGES_PATH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aml-build-", dir=PACKAGES_PATH) as temp:
        stage = Path(temp) / "publish"
        # Resolve versions, frameworks and native assets from AML's project
        # graph. Do not override them with a second hand-maintained list.
        call_msbuild(
            [
                INTEROP_PROJECT,
                "-restore",
                "-t:Rebuild;Publish" if args.clean else "-t:Publish",
                f"-p:Configuration={config}",
                "-p:Platform=x64",
                "-p:RuntimeIdentifier=win-x64",
                "-p:SelfContained=false",
                "-p:RestorePackagesConfig=true",
                f"-p:PublishDir={stage}/",
                "-verbosity:minimal",
            ]
        )
        if args.with_stubs:
            call_msbuild(
                [
                    STUB_GENERATOR_SLN,
                    "-restore",
                    "-t:Build",
                    "-p:RestorePackagesConfig=true",
                    "-p:Configuration=Release",
                ]
            )
        finish_bin(stage, args.debug, "win-x64")


def check_dependencies_macos():
    missing = []
    if not shutil.which("dotnet"):
        missing.append("dotnet (https://dotnet.microsoft.com/download)")
    if not shutil.which("git"):
        missing.append("git")
    if not shutil.which("clang++"):
        missing.append("clang++ (xcode-select --install)")

    if missing:
        for item in missing:
            print(f"Please install {item}")
        sys.exit(1)


def ensure_net48_reference_assemblies() -> Path:
    refs = NET48_REFS_PATH / "build/.NETFramework/v4.8"
    if refs.exists():
        return refs

    PACKAGES_PATH.mkdir(exist_ok=True)
    nupkg = PACKAGES_PATH / f"{NET48_REFS_PACKAGE}.{NET48_REFS_VERSION}.nupkg"
    print(f"Downloading {NET48_REFS_PACKAGE}...")
    urllib.request.urlretrieve(NET48_REFS_URL, nupkg)

    with zipfile.ZipFile(nupkg) as archive:
        archive.extractall(NET48_REFS_PATH)
    nupkg.unlink()

    if not refs.exists():
        raise Exception(f"Reference assemblies did not unpack to {refs}")
    return refs


def get_macos_rid() -> str:
    machine = platform.machine()
    return "osx-arm64" if machine == "arm64" else "osx-x64"


def build_ooz(args, output: Path):
    """Build ooz as a drop-in for the Oodle DLL ZamboniLib expects."""
    dylib = output / OOZ_DYLIB

    if not OOZ_SRC.exists():
        PACKAGES_PATH.mkdir(exist_ok=True)
        subprocess.check_call(["git", "clone", OOZ_REPO, OOZ_SRC])

    subprocess.check_call(["git", "-C", OOZ_SRC, "checkout", "--force", OOZ_COMMIT])

    sse2neon = OOZ_SRC / "sse2neon.h"
    if not sse2neon.exists():
        print("Downloading sse2neon...")
        urllib.request.urlretrieve(SSE2NEON_URL, sse2neon)

    shutil.copyfile(OOZ_SHIM_DIR / "stdafx_portable.h", OOZ_SRC / "stdafx.h")
    shutil.copyfile(OOZ_SHIM_DIR / "oodle_shim.cpp", OOZ_SRC / "oodle_shim.cpp")

    # Drop the CLI tool at the end of kraken.cpp: it needs Windows.h, and a
    # library build only wants the decompressor above it.
    kraken = OOZ_SRC / "kraken.cpp"
    source = kraken.read_text()
    cli_start = source.index("byte *load_file(")
    source = source[:cli_start]
    # Export the entry point ZamboniLib's ooz path names, in case the Oodle
    # ABI shim is ever bypassed.
    source = source.replace(
        "int Kraken_Decompress(const byte *src",
        'extern "C" __attribute__((visibility("default")))'
        " int Kraken_Decompress(const byte *src",
        1,
    )
    kraken.write_text(source)

    subprocess.check_call(
        [
            "clang++",
            "-std=c++14",
            "-O2",
            "-fPIC",
            "-shared",
            "-Wno-deprecated-declarations",
            "-Wno-unknown-pragmas",
            "-Wno-macro-redefined",
            "-o",
            dylib,
            OOZ_SRC / "kraken.cpp",
            OOZ_SRC / "bitknit.cpp",
            OOZ_SRC / "lzna.cpp",
            OOZ_SRC / "oodle_shim.cpp",
        ]
    )
    print(f"Built {dylib.name}")


def build_macos(args):
    check_dependencies_macos()

    config = "Debug" if args.debug else "Release"
    refs = ensure_net48_reference_assemblies()

    # Publishing with a runtime identifier resolves every NuGet package's
    # assemblies and native libraries (libassimp, libprs_rs) for this machine
    # into one flat folder, just as the Windows MSBuild publish does.
    command = [
        "dotnet",
        "publish",
        INTEROP_PROJECT,
        "--configuration",
        config,
        "--runtime",
        get_macos_rid(),
        "--no-self-contained",
        "-verbosity:minimal",
        f"-p:FrameworkPathOverride={refs}",
    ]
    PACKAGES_PATH.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aml-build-", dir=PACKAGES_PATH) as temp:
        stage = Path(temp) / "publish"
        if args.clean:
            subprocess.check_call(
                ["dotnet", "clean", INTEROP_PROJECT, "--configuration", config],
                cwd=ROOT,
            )
        subprocess.check_call([*command, "--output", stage], cwd=ROOT)
        build_ooz(args, stage)
        finish_bin(stage, args.debug, get_macos_rid())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--fbx-sdk", type=Path, help="Windows: FBX 2020 SDK directory")
    parser.add_argument(
        "--with-stubs",
        action="store_true",
        help="Also build the Windows typing generator",
    )

    args = parser.parse_args()

    if sys.platform == "win32":
        build_windows(args)
    elif sys.platform == "darwin":
        build_macos(args)
    else:
        print(f"Unsupported platform: {sys.platform}")
        sys.exit(1)


if __name__ == "__main__":
    main()
