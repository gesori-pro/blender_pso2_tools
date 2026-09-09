# Windows release validation - 2026-09-09

The release is based on `release/20260906`, including the macOS merge
`0ebbe3f`. The downloadable ZIP targets Windows x64. No macOS binary package
or macOS runtime validation is claimed by this release.

## Export fixes

- Removing a zero-filled secondary UV buffer also removes its VTXE declaration.
  AML supplies element sizes and serialization. Shared NGS layouts are cloned;
  classic layouts get an updated stride. Nonzero and packed UV data are kept.
- Import removes only trailing empty UV layers. Intermediate empty layers stay
  in place so subsequent export cannot move UV3 data into UV2. Both FBX and
  direct import paths follow this rule.
- Import records the material's encoded PSO2 name on the material datablock.
  A copied or renamed material retains its shader, blend and alpha metadata.
  FBX receives an encoded export name without renaming the Blender material;
  AML still parses that metadata and creates the PSO2 material.
- Explicit PSO2 metadata in the current name takes precedence over saved
  metadata. A plain name with no saved metadata cancels export before either
  output file is touched. Restore the original encoded name or re-import and
  duplicate the correct source material for older scenes with lost metadata.
- Windows detects `AquaModelLibrary.Native.X64.dll` and uses the FBX importer
  by default. The bundled interop helper also permits explicit direct import.
- The existing shape-key export fix and Image RNA/default-skin fixes are retained.
  ICE envelope handling, AQP splitting, AQM constants and face-name parsing
  delegate to the bundled AML APIs. AML binaries were not replaced.

## Validation of the installable ZIP

Blender's extension builder created the ZIP. It was installed and enabled in
separate clean profiles in Blender 5.1.2 and 5.2.1. Tests loaded the installed
`bl_ext.user_default.pso2_tools` package and its installed wheel dependencies.

| Check | Blender 5.1.2 | Blender 5.2.1 |
| --- | --- | --- |
| Shape-key export suite | 14 passed | 14 passed |
| Material export suite with FBX import | 5 passed | 5 passed |
| Material export suite with direct import | 5 passed | 5 passed |
| AML delegation, including a real ICE archive | 7 passed | 7 passed |
| Removed Image RNA and missing default-skin data | 6 passed | 6 passed |
| UV writer: old failure, NGS shared layout, classic stride | Passed | Passed |
| Real garment: export, AML reparse, Blender re-import | Passed | Passed |

The real garment check used 13 meshes and 53,943 vertices. Applied shape-key
output matched an independently evaluated plain mesh, differed from the basis,
and left the source scene intact before re-import. Input AQP/AQN hashes stayed
unchanged. Existing destination AQN preservation is also checked by the export
and material tests.

The material regression test reproduces `0398` through the actual AML converter
when the metadata adapter is bypassed. With the fix, arbitrary names and copied
materials preserve `1100`/`1103`, their texture names and UV1-UV4 values through
export and re-import. Explicit shader edits and failure paths are also tested.

The previous garment-only `0398` to `1100` assignment was reported as fixed in
game by the user. These release checks establish file and Blender behavior;
they are not a fresh game-rendering or physics test of the new ZIP.

## Scope

The uncommitted legacy-accessory missing-MATE/null-list workaround is excluded
and retained locally. Private scenes, mod archives, textures, AQN files and
local QA logs are not part of the source commit or release assets. The package
keeps the upstream version `2.8.0`; dated release asset names distinguish builds.
