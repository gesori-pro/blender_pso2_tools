#! /usr/bin/env python3
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path

import tomlkit
from blender import blender_call

ROOT = Path(__file__).parent.parent
SOURCE_DIR = ROOT / "pso2_tools"
OUTPUT_DIR = ROOT / "dist"

MANIFEST = SOURCE_DIR / "blender_manifest.toml"

HOST_PLATFORM = {
    "win32": "windows-x64",
    "darwin": "macos-arm64",
}


@contextmanager
def _host_platform_only():
    """Limit the manifest to the build host's platform while building.

    pso2_tools/bin holds binaries scripts/build_bin.py made on this machine,
    so the zip is only usable where it was built; trimming the platform list
    makes the package say so and names the file after it.
    """
    platform = HOST_PLATFORM.get(sys.platform)
    if platform is None:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")

    original = MANIFEST.read_text()
    manifest = tomlkit.parse(original)
    manifest["platforms"] = [platform]
    # --split-platforms omits foreign wheels from the ZIP but leaves their
    # declarations in the manifest. Declare only the wheels being shipped.
    manifest["wheels"] = [
        wheel
        for wheel in manifest["wheels"]
        if _wheel_matches_platform(wheel, platform)
    ]
    MANIFEST.write_text(tomlkit.dumps(manifest))
    try:
        yield
    finally:
        MANIFEST.write_text(original)


def _wheel_matches_platform(wheel: str, platform: str) -> bool:
    tags = Path(wheel).stem.rsplit("-", 1)[-1].split(".")
    if "any" in tags:
        return True
    if platform == "windows-x64":
        return "win_amd64" in tags
    return any(
        tag.startswith("macosx_") and tag.endswith(("_arm64", "_universal2"))
        for tag in tags
    )


def build():
    OUTPUT_DIR.mkdir(exist_ok=True)
    with _host_platform_only():
        result = blender_call(
            [
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "-c",
                "extension",
                "build",
                "--source-dir",
                SOURCE_DIR,
                "--output-dir",
                OUTPUT_DIR,
                "--split-platforms",
            ]
        )
        if result:
            raise RuntimeError(f"Blender extension build failed (exit {result})")

    return get_package_path()


def get_package_path() -> Path:
    with MANIFEST.open("rb") as f:
        manifest = tomllib.load(f)

    platform = HOST_PLATFORM[sys.platform].replace("-", "_")
    name = f"{manifest['id']}-{manifest['version']}-{platform}.zip"

    return OUTPUT_DIR / name


if __name__ == "__main__":
    build()
