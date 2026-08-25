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
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

FRAMEWORK = "net9.0"

FBX_URL = "https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-1/fbx20201_fbxsdk_vs2017_win.exe"
NUGET_URL = "https://learn.microsoft.com/en-us/nuget/consume-packages/install-use-packages-nuget-cli"
VISUAL_STUDIO_URL = "https://visualstudio.microsoft.com/vs/community/"

VSWHERE = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")

FBX_SRC = Path("C:/Program Files/Autodesk/FBX/FBX SDK/2020.1")
FBX_DEST = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.Native/Dependencies/FBX"
BIN_PATH = ROOT / "pso2_tools/bin"

AQUA_SLN = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.sln"
AQUA_CORE_PATH = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.Core"

INTEROP_PROJECT = ROOT / "dotnet/Pso2Tools.Interop/Pso2Tools.Interop.csproj"

STUB_GENERATOR_SLN = ROOT / "pythonnet-stub-generator/csharp/PythonNetStubGenerator.sln"

PACKAGES_PATH = ROOT / "packages"
PACKAGES = [
    ("BouncyCastle.Cryptography", "2.4.0"),
    ("DrSwizzler", "1.1.1"),
    ("prs_rs.Net.Sys", "1.0.4"),
    ("Pfim", "0.11.3"),
    ("Reloaded.Memory", "9.4.2"),
    ("SixLabors.ImageSharp", "3.1.6"),
    ("SharpAssimp", "6.0.12"),
    ("SharpZipLib", "1.4.2"),
    ("System.Drawing.Common", "8.0.11"),
    ("System.Data.DataSetExtensions", "4.6.0-preview3.19128.7"),
    ("System.Text.RegularExpressions", "4.3.1"),
    ("ZstdNet", "1.4.5"),
]

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
    if not shutil.which("nuget"):
        print(f"Please install nuget: {NUGET_URL}")

    if not VSWHERE.exists():
        print(f"Please install Visual Studio: {VISUAL_STUDIO_URL}")
        sys.exit(1)

    if not FBX_SRC.exists():
        print(f"Please install FBX SDK 2020.1: {FBX_URL}")
        sys.exit(1)


def make_junction(src: Path, dest: Path):
    if dest.exists():
        return

    subprocess.call(["mklink", "/J", dest, src], shell=True)


def install_packages():
    for package, version in PACKAGES:
        subprocess.check_call(
            [
                "nuget",
                "install",
                package,
                "-Version",
                version,
                "-Framework",
                FRAMEWORK,
                "-OutputDirectory",
                PACKAGES_PATH,
            ]
        )


def copy_package_dlls():
    frameworks = [
        "net9.0",
        "net8.0",
        "net7.0",
        "net6.0",
        "net5.0",
        "netstandard2.1",
        "netstandard2.0",
        "netstandard1.3",
    ]

    for package, version in PACKAGES:
        src = PACKAGES_PATH / f"{package}.{version}"
        lib = src / "lib"
        runtime_x64 = src / "runtimes/win-x64/native"

        if not src.exists():
            raise Exception(f"Couldn't find {src}")

        try:
            framework = next(lib / f for f in frameworks if (lib / f).exists())

            for dll in framework.glob("*.dll"):
                shutil.copyfile(dll, BIN_PATH / dll.name)
        except StopIteration:
            pass

        for dll in runtime_x64.glob("*.dll"):
            print(" ", dll.name)
            shutil.copyfile(dll, BIN_PATH / "x64" / dll.name)


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

    subprocess.check_call([msbuild, *args])


def build_windows(args):
    check_dependencies()

    target = "Rebuild" if args.clean else "Build"
    config = "Debug" if args.debug else "Release"

    # Set up Aqua Library dependencies
    # Use junction points instead of symlinks so Git sees them as directories
    # and they fit PSO2-Aqua-Library's .gitignore patterns.
    make_junction(FBX_SRC / "lib", FBX_DEST / "lib")
    make_junction(FBX_SRC / "include", FBX_DEST / "include")

    install_packages()

    # Build Aqua Library
    call_msbuild(
        [
            AQUA_SLN,
            "-p:RestorePackagesConfig=true",
            f"-p:Configuration={config}",
            f"-t:{target}",
            "-verbosity:minimal",
            "-restore",
        ]
    )

    # Build the interop helpers the native model importer uses. Windows still
    # imports through FBX, but building it everywhere keeps one bin layout.
    call_msbuild(
        [
            INTEROP_PROJECT,
            f"-p:Configuration={config}",
            f"-t:{target}",
            "-verbosity:minimal",
            "-restore",
        ]
    )

    # Copy to pso2_tools/bin folder
    out_path = AQUA_CORE_PATH / "bin" / config / FRAMEWORK

    ignore = None if args.debug else shutil.ignore_patterns("*.pdb")

    shutil.rmtree(BIN_PATH, ignore_errors=True)
    shutil.copytree(out_path, BIN_PATH, dirs_exist_ok=True, ignore=ignore)

    interop_out = INTEROP_PROJECT.parent / "bin" / config / FRAMEWORK
    for pattern in ("Pso2Tools.Interop.dll", "Pso2Tools.Interop.pdb"):
        for file in interop_out.glob(pattern):
            if not args.debug and file.suffix == ".pdb":
                continue
            shutil.copyfile(file, BIN_PATH / file.name)

    copy_package_dlls()

    # Build pythonnet-stub-generator
    call_msbuild(
        [
            STUB_GENERATOR_SLN,
            "-p:RestorePackagesConfig=true",
            "-p:Configuration=Release",
            f"-t:{target}",
            "-verbosity:minimal",
            "-restore",
        ]
    )


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


def build_ooz(args):
    """Build ooz as a drop-in for the Oodle DLL ZamboniLib expects."""
    dylib = BIN_PATH / OOZ_DYLIB

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

    shutil.rmtree(BIN_PATH, ignore_errors=True)

    # Publishing with a runtime identifier resolves every NuGet package's
    # assemblies and native libraries (libassimp, libprs_rs) for this machine
    # into one flat folder, so there is no hand-kept package list like the
    # Windows path needs.
    command = [
        "dotnet",
        "publish",
        INTEROP_PROJECT,
        "--configuration",
        config,
        "--runtime",
        get_macos_rid(),
        "--no-self-contained",
        "--output",
        BIN_PATH,
        "-verbosity:minimal",
        f"-p:FrameworkPathOverride={refs}",
    ]
    if args.clean:
        command.append("--no-incremental")

    subprocess.check_call(command)

    if not args.debug:
        for pdb in BIN_PATH.glob("*.pdb"):
            pdb.unlink()

    build_ooz(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--debug", action="store_true")

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
