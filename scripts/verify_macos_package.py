"""Verify the actual release ZIP, including provenance and native binaries."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def revision(path):
    return subprocess.check_output(
        ["git", "-C", path, "rev-parse", "HEAD"], text=True
    ).strip()


def verify(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        manifest = tomllib.loads(archive.read("blender_manifest.toml").decode())
        source_manifest = tomllib.loads(
            (ROOT / "pso2_tools/blender_manifest.toml").read_text()
        )
        assert manifest["id"] == "pso2_tools"
        assert manifest["version"] == source_manifest["version"]
        assert manifest["platforms"] == ["macos-arm64"]
        for wheel in manifest["wheels"]:
            assert wheel in names, wheel
            assert "win_amd64" not in wheel, wheel
        for abi in ("cp311", "cp313"):
            for dep in ("cffi", "watchdog"):
                assert any(
                    Path(wheel).name.startswith(dep + "-")
                    and abi in wheel
                    and "macosx" in wheel
                    for wheel in manifest["wheels"]
                ), (dep, abi)
        for dep in ("pythonnet", "clr_loader", "pycparser"):
            assert any(
                Path(wheel).name.startswith(dep + "-") for wheel in manifest["wheels"]
            ), dep
        info = json.loads(archive.read("bin/build-info.json"))
        assert info["runtime"] == "osx-arm64"
        assert info["source_commit"] == revision(ROOT)
        assert info["aml_commit"] == revision(ROOT / "PSO2-Aqua-Library")
        for name, digest in info["dll_sha256"].items():
            assert hashlib.sha256(archive.read("bin/" + name)).hexdigest() == digest
        for name in ("libassimp.dylib", "libprs_rs.dylib", "liboo2core_8_win64_.dylib"):
            data = archive.read("bin/" + name)
            assert data[:4] == bytes.fromhex("cffaedfe"), name
            assert struct.unpack_from("<I", data, 4)[0] == 0x100000C, name
        assert not any(
            name.endswith((".blend", ".fnp", ".aqp", ".aqn", ".aqm", ".ice", ".db"))
            for name in names
        ), "Unexpected user/game data in ZIP"
        for name in (
            "AquaModelLibrary.Native.X64.dll",
            "AquaModelLibrary.Native.X86.dll",
            "Ijwhost.dll",
            "assimp.dll",
            "oo2core_8_win64_.dll",
            "x64/libzstd.dll",
            "x86/libzstd.dll",
        ):
            assert "bin/" + name not in names, name
        sources = (
            subprocess.check_output(["git", "-C", ROOT, "ls-files", "-z", "pso2_tools"])
            .decode()
            .split("\0")
        )
        checked = 0
        for source in filter(None, sources):
            relative = source.removeprefix("pso2_tools/")
            if relative == "blender_manifest.toml":
                continue
            assert archive.read(relative) == (ROOT / source).read_bytes(), source
            checked += 1
    print(
        json.dumps(
            {
                "zip": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_commit": info["source_commit"],
                "aml_commit": info["aml_commit"],
                "source_files_matched": checked,
                "dll_hashes_matched": len(info["dll_sha256"]),
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    verify(parser.parse_args().zip)
