# macOS companion for PSO2 Tools 2.10.0

Validated on 2026-10-15. This companion ports the Windows `build-20261010`
release (`e11269ff4c6d027ef1c6f32cb9e57195a76c40ed`) to Apple Silicon. Its
source is tagged `build-20261010-macos`; the Mac ZIP is attached to the
existing `build-20261010` GitHub release. The Windows ZIP and tag remain the
original Windows build.

## Implementation

- Retains the Windows release's 51 static face controls, expression editing,
  per-rig pose restoration, outfit variant selection and locale-safe shaders.
- Reuses the direct model importer and bulk interop from `build-20260925`.
  AML still parses/writes AQP, AQN, AQM and ICE content. Only the Windows-only
  C++/CLI FBX import bridge is replaced. Export uses Blender FBX and Assimp.
- Pins the same AML revision as Windows:
  `7e9f8749ded800fb3c5b93952bfb9cb7a04734e5`, with the existing build-time
  compatibility adapters from the September Mac build. AML's checkout is not
  edited. Managed assemblies are AnyCPU and native dependencies are arm64.
- Uses host-specific wheel declarations, omits copied Windows Oodle/zstd DLLs
  from Mac output, and records the source commit and actual shipped DLL
  hashes in `bin/build-info.json`.
- Keeps Windows and Mac release source separate from the older `main`
  branch. This is a companion to the Windows feature set, not a merge of
  every later change from the September development branch.

## Validation

Environment: Apple Silicon, macOS 27.0 (26A428), Blender 5.2.0 LTS
(`fbe6228777e7`), .NET SDK 9.0.318 for building, .NET Runtime 10.0.3 in Blender.
The ZIP was installed into an isolated Blender profile using its bundled
wheels. Subsequent fresh Blender processes loaded Python modules and AML
assemblies from that installed ZIP, not from a development checkout.

| Check | Result |
| --- | --- |
| Clean install / fresh-process startup | Passed; version 2.10.0, direct importer and bundled assemblies loaded |
| Face controls | 166 checks, including real motions for face types 100070, 100000 and 200070 |
| Real face geometry | 130 bones, 7,272 vertices; left/right mouth, new lip, jaw, eyelash and shadow controls exercised |
| Face restoration / persistence | Zero pose restoration error; `.blend` save/reload retained pose and settings |
| Outfit lookup | 99 checks covering V10–V16, empty outfit slots and variant-to-base fallback |
| Japanese / Simplified Chinese | Shader variants 1110 and 1117 created successfully in both languages |
| AQP export -> direct import -> export | Two synthetic root-skinning round trips, including a translated root; positions and bone weights preserved |
| Full character | Real character file imported 13 parts, 5 rigs, 47 meshes and 29,039 vertices; face editor and game lighting initialized |
| Package | ZIP CRC, arm64 dylibs, wheel presence, DLL hashes, source-byte comparison and no bundled user/game data checked |
| Source checks | Ruff and `git diff --check` |

The local catalog does not contain the Windows regression fixture IDs
219421/219420, so the optional real-catalog test for those exact IDs cannot
run here; their lookup behavior is covered by the 99 synthetic checks.
Actual imports above use available Japanese game assets. No input character
file was modified, and no game assets or object database are distributed.

Intel Macs, older Blender versions, live-game appearance and accessory
placement are not established by this validation. The existing face-editor
limitations in the Windows release notes also apply. The Windows C++/CLI
execution path was not re-tested on this Mac.

## Reproduce

Use native arm64 Python 3.11+, Blender, Xcode command line tools and .NET SDK
9. Install `pip` and `tomlkit` in a build virtual environment, then run:

```sh
python scripts/wheels.py
python scripts/build_bin.py
python scripts/build_package.py
python scripts/verify_macos_package.py dist/pso2_tools-2.10.0-macos_arm64.zip
```

Run `scripts/test_macos_model.py`, `scripts/test_face_controls.py`,
`scripts/test_character_parts.py` and `scripts/test_shader_locale.py` in fresh
background Blender processes with `--python-exit-code 1`. The latter scripts
support `PSO2_TEST_ADDON_MODULE` for checking a ZIP installed as an extension.
Real-face test environment variables are documented at the top of
`test_face_controls.py`. Private fixture paths and local QA profiles are not
part of the release source.
