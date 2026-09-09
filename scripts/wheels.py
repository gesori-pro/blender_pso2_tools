#! /usr/bin/env python3
"""
Download wheels for the project's dependencies.
"""

import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

import tomlkit

ROOT = Path(__file__).parent.parent
ADDON_PATH = ROOT / "pso2_tools"
WHEELS = ADDON_PATH / "wheels"
MANIFEST = ADDON_PATH / "blender_manifest.toml"

DEPENDENCIES = ["pythonnet==3.0.5", "watchdog==6.0.0"]

PYTHON_VERSIONS = ["3.11", "3.13"]

# One entry per supported platform. Extra tags on an entry let pip accept
# the universal builds macOS packages often ship instead of arm64-only ones.
PLATFORMS = [
    ["win_amd64"],
    ["macosx_11_0_arm64", "macosx_10_9_universal2"],
]


def main():
    shutil.rmtree(WHEELS, ignore_errors=True)

    for dep, version, platforms in product(DEPENDENCIES, PYTHON_VERSIONS, PLATFORMS):
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                dep,
                "--dest",
                WHEELS,
                "--only-binary=:all:",
                f"--python-version={version}",
                *(f"--platform={platform}" for platform in platforms),
            ]
        )

    manifest = tomlkit.parse(MANIFEST.read_text())

    wheels = WHEELS.rglob("*.whl")

    a = tomlkit.array()
    for w in wheels:
        a.add_line(str(w.relative_to(ADDON_PATH).as_posix()), indent="  ")

    a.add_line(indent="")
    manifest["wheels"] = a

    MANIFEST.write_text(tomlkit.dumps(manifest))


if __name__ == "__main__":
    main()
