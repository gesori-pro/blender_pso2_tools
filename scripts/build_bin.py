#! /usr/bin/env python3
"""
Build .net dependencies.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

FRAMEWORK = "net9.0"

FBX_URL = "https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-1/fbx20201_fbxsdk_vs2017_win.exe"
NUGET_URL = "https://learn.microsoft.com/en-us/nuget/consume-packages/install-use-packages-nuget-cli"
VISUAL_STUDIO_URL = "https://visualstudio.microsoft.com/vs/community/"
DOTNET_URL = "https://dotnet.microsoft.com/en-us/download/dotnet/9.0"

VSWHERE = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")

FBX_SRC = Path("C:/Program Files/Autodesk/FBX/FBX SDK/2020.1")
FBX_DEST = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.Native/Dependencies/FBX"
BIN_PATH = ROOT / "pso2_tools/bin"

AQUA_SLN = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.sln"
AQUA_CORE_PATH = ROOT / "PSO2-Aqua-Library/AquaModelLibrary.Core"
AQUA_CORE_PROJECT = AQUA_CORE_PATH / "AquaModelLibrary.Core.csproj"

STUB_GENERATOR_SLN = ROOT / "pythonnet-stub-generator/csharp/PythonNetStubGenerator.sln"

PACKAGES_PATH = ROOT / "packages"
PACKAGES = [
    ("AssimpNet", "5.0.0-beta1"),
    ("BouncyCastle.Cryptography", "2.4.0"),
    ("DrSwizzler", "1.1.1"),
    ("prs_rs.Net.Sys", "1.0.4"),
    ("Pfim", "0.11.3"),
    ("Reloaded.Memory", "9.4.2"),
    ("SixLabors.ImageSharp", "3.1.6"),
    ("SharpZipLib", "1.4.2"),
    ("System.Drawing.Common", "8.0.11"),
    ("System.Data.DataSetExtensions", "4.6.0-preview3.19128.7"),
    ("System.Net.Http", "4.3.4"),
    ("System.Text.Encoding.CodePages", "9.0.0"),
    ("System.Text.RegularExpressions", "4.3.1"),
    ("ZstdNet", "1.4.5"),
]


def is_windows() -> bool:
    return sys.platform == "win32"


def get_runtime_identifier() -> str | None:
    machine = platform.machine().lower()

    if sys.platform == "darwin":
        if machine in {"arm64", "aarch64"}:
            return "osx-arm64"
        return "osx-x64"

    if sys.platform.startswith("linux"):
        if machine in {"arm64", "aarch64"}:
            return "linux-arm64"
        return "linux-x64"

    return None


def check_common_dependencies():
    if not shutil.which("dotnet"):
        print(f"Please install .NET SDK 9.0: {DOTNET_URL}")
        sys.exit(1)


def check_windows_dependencies():
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

    subprocess.call(["mklink", "/J", str(dest), str(src)], shell=True)


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

        try:
            framework = next(lib / f for f in frameworks if (lib / f).exists())

            for dll in framework.glob("*.dll"):
                shutil.copyfile(dll, BIN_PATH / dll.name)
        except StopIteration:
            pass

        for dll in runtime_x64.glob("*.dll"):
            shutil.copyfile(dll, BIN_PATH / "x64" / dll.name)


def call_msbuild(args: list[Path | str]):
    vs = json.loads(
        subprocess.check_output([VSWHERE, "-latest", "-format", "json"], encoding="utf-8")
    )
    msbuild = Path(vs[0]["installationPath"]) / "Msbuild/Current/Bin/MSBuild.exe"

    subprocess.check_call([msbuild, *args])


def call_dotnet_build(
    project: Path, config: str, clean: bool, runtime_identifier: str | None = None
):
    build_args = [
        "dotnet",
        "build",
        project,
        "-c",
        config,
        "-nologo",
        "-p:CopyLocalLockFileAssemblies=true",
    ]

    if runtime_identifier:
        build_args.extend(["-r", runtime_identifier])

    if clean:
        clean_args = ["dotnet", "clean", project, "-c", config, "-nologo"]
        if runtime_identifier:
            clean_args.extend(["-r", runtime_identifier])
        subprocess.check_call(clean_args)

    subprocess.check_call(build_args)


def copy_bin_output(config: str, debug: bool, runtime_identifier: str | None = None):
    out_path = AQUA_CORE_PATH / "bin" / config / FRAMEWORK
    if runtime_identifier:
        out_path = out_path / runtime_identifier

    ignore = None if debug else shutil.ignore_patterns("*.pdb")

    shutil.rmtree(BIN_PATH, ignore_errors=True)
    shutil.copytree(out_path, BIN_PATH, dirs_exist_ok=True, ignore=ignore)


def build_windows(clean: bool, config: str, debug: bool):
    check_windows_dependencies()

    # Set up Aqua Library dependencies
    # Use junction points instead of symlinks so Git sees them as directories
    # and they fit PSO2-Aqua-Library's .gitignore patterns.
    make_junction(FBX_SRC / "lib", FBX_DEST / "lib")
    make_junction(FBX_SRC / "include", FBX_DEST / "include")

    install_packages()

    target = "Rebuild" if clean else "Build"

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

    # Copy to pso2_tools/bin folder
    copy_bin_output(config, debug)
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


def build_non_windows(clean: bool, config: str, debug: bool):
    runtime_identifier = get_runtime_identifier()

    # Build Aqua Library Core (without Windows native FBX project).
    call_dotnet_build(
        AQUA_CORE_PROJECT,
        config=config,
        clean=clean,
        runtime_identifier=runtime_identifier,
    )

    # Copy to pso2_tools/bin folder
    copy_bin_output(config, debug, runtime_identifier=runtime_identifier)

    # Build pythonnet-stub-generator
    call_dotnet_build(STUB_GENERATOR_SLN, config="Release", clean=clean)


def main():
    check_common_dependencies()

    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    config = "Debug" if args.debug else "Release"

    if is_windows():
        build_windows(clean=args.clean, config=config, debug=args.debug)
    else:
        build_non_windows(clean=args.clean, config=config, debug=args.debug)


if __name__ == "__main__":
    main()
