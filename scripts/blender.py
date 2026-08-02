"""
Find and run Blender.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def find_blender() -> Path:
    # TODO: add handling for multiple installed versions
    if blender_path := os.getenv("BLENDER_PATH"):
        path = Path(blender_path)
        if path.is_file():
            return path

    program_files = Path(os.getenv("PROGRAMFILES", "C:/Program Files"))
    roots = [program_files / "Blender Foundation"]
    roots += [
        Path(f"{drive}:/Program Files/Blender Foundation")
        for drive in "CDEFG"
        if Path(f"{drive}:/Program Files/Blender Foundation") not in roots
    ]

    for blender_root in roots:
        if not blender_root.exists():
            continue

        try:
            return next(blender_root.rglob("blender.exe"))
        except StopIteration:
            continue

    raise RuntimeError("Could not find Blender")


def blender_call(cmd, *args, **kwargs):
    blender = find_blender()
    return subprocess.call([blender, *cmd], *args, **kwargs)


def blender_check_output(cmd, *args, encoding="utf-8", **kwargs) -> str:
    blender = find_blender()
    return subprocess.check_output([blender, *cmd], *args, encoding=encoding, **kwargs)


def get_extension_repo(name: str) -> Path | None:
    output = blender_check_output(["-c", "extension", "repo-list"])

    current_repo = None

    for line in output.splitlines():
        if m := re.match(r"^(\w+):$", line):
            current_repo = m.group(1)

        if current_repo == name and line.startswith("    "):
            key, _, value = line.partition(":")
            key = key.strip()

            if key == "directory":
                value = value.strip().strip('"')
                return Path(value)

    return None


def main():
    blender_call(sys.argv[1:])


if __name__ == "__main__":
    main()
