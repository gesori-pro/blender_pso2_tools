#! /usr/bin/env python3
"""
Download wheels for the project's dependencies.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
WHEELS = ROOT / "pso2_tools" / "wheels"

DEPENDENCIES = ["pythonnet==3.0.5", "watchdog==6.0.0"]

PYTHON_VERSION = "3.11"

# Download all target wheels so the extension package can be built from any host OS.
PLATFORMS = [
    "win_amd64",
    "macosx_10_9_x86_64",
    "macosx_11_0_arm64",
]


def main():
    shutil.rmtree(WHEELS, ignore_errors=True)
    WHEELS.mkdir(parents=True, exist_ok=True)

    for platform in PLATFORMS:
        for dep in DEPENDENCIES:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "download",
                    dep,
                    "--dest",
                    WHEELS,
                    "--only-binary=:all:",
                    f"--python-version={PYTHON_VERSION}",
                    f"--platform={platform}",
                ]
            )


if __name__ == "__main__":
    main()
