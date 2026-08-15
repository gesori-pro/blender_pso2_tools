# Blender Add-on for PSO2

This integrates the model import/export functions of [Aqua-Library](https://github.com/Shadowth117/PSO2-Aqua-Library) into Blender.

## Installation

The libraries used for import/export are only compiled for Windows x64. This add-on will not currently work on other OSes.

1. Install the [.NET Runtime 9.0](https://dotnet.microsoft.com/en-us/download/dotnet/9.0) for Windows (on the bottom right of the page).
2. Download `pso2_tools-****.zip` from the [latest release](https://github.com/dummycount/blender_pso2_tools/releases/latest).
3. In Blender, go to **Edit > Preferences > Add-ons**.
4. Click the down arrow in the upper-right corner and select **Install from Disk...**.
5. Select the .zip file you downloaded.
6. Make sure **PSO2 Tools** is checked in the add-ons list.
7. Expand **PSO2 Tools** and make sure **Path to pso2_bin/data** is correct. If not, set it to point to your game's install directory.

Tested on Blender 4.4 and 5.1. Importing and exporting models, motions and character files behaves the same on both, down to identical output files. Versions before 4.4 are untested.

## Usage

### Import

**Files > Import > PSO2 Model Search** opens a window to find an import items by name. Currently only character model items can be searched.

**Files > Import > PSO2 ICE Archive** imports models and textures from an ICE archive. If the file name matches a known item, settings such as color mapping are automatically read from that item.

**Files > Import > PSO2 AQP (.aqp)** imports from the `.aqp` model format. If an `aqn` skeleton file of the same name exists, it is also imported, as are any `.dds` textures in the same folder. If the file name matches a known item, settings such as color mapping are automatically read from that item.

For models that get their textures from other items, such as innerwear textures on basewear items, or eye/eyebrow/eyelash textures on faces, import the item with the textures first, then import the model, and it will automatically find the correct textures. If you import in the wrong order, you can go to Blender's **Shading** tab, select an object, then assign any missing textures in the shader editor area.

If an NGS model uses skin textures, they will automatically be imported. You can change which textures to use in the add-on preferences.

**Files > Import > PSO2 Character (.fnp/.mhp/...)** reads a character file saved by the game or the Character Creator and reshapes the imported model's skeleton to match, along with its skin, hair, and outfit colors. Import the model first, then the character file. All eight body types are accepted, in file versions 10 through 16, so both current live characters and older saves work.

The proportions the game applies to an outfit depend on the outfit itself, so the result matches what you see in game rather than a generic body.

**Files > Import > PSO2 AQM (.aqm)** loads a motion onto the imported armature as an action.

Shape adjusts - the extra per-outfit tweak the game applies on top of the body proportions - are loaded from **Scene > PSO2 Appearance > Shape Adjust**, alongside the sliders that edit them, rather than from the import menu.

### Export

**Files > Export > PSO2 AQP (.aqp)** exports the model back to an `.aqp` file.

By default, this will only write a matching `.aqn` file if it does not already exist. Check **Overwrite .aqn** to overwrite any existing file.

**Ignore Pose** is on by default and exports the skeleton as the model import left it. Without it, a body shape or animation frame sitting in the pose is written into the model, and since the file comes out the same size either way there is nothing to notice. It does nothing unless a character file has been applied, so plain model round trips are unaffected.

**Files > Export > PSO2 AQM (.aqm)** saves the armature's animation as a motion file, one key per frame.

Apply the body shape to the rest pose first (see below). A body shape left in the pose gets written into the motion, which is not what game motion files contain.

**Files > Export > PSO2 Shape Adjust (\_sa.aqm)** saves the current slider values as a shape-adjust motion that the game can load.

### Preferences

Go to **Edit > Preferences > Add-ons > PSO2 Tools** to edit the extension's settings.

| Setting                 | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| Path to pso2_bin/data   | Path to `pso2_bin/data` inside the game's install directory      |
| Hide armature on import | Automatically hide the armature object when importing a model    |
| Debug logging           | If enabled, debugging messages are written to the system console |
| Default Muscularity     | Default value for **Muscularity** scene property                 |
| Default T1 Skin Texture | Skin texture to import for T1 models                             |
| Default T2 Skin Texture | Skin texture to import for T2 models                             |
| Import Colors           | Default values for the color scene properties                    |

### Scene Properties

In the **Properties** area, go to the **Scene** tab. Two panels will appear here once a model has been imported:

#### PSO2 Appearance

| Property       | Description                                             |
| -------------- | ------------------------------------------------------- |
| Hide Innerwear | Hides the innerwear layer on skin materials             |
| Muscularity    | Adjusts the mix between skin textures on skin materials |
| Colors         | The color channels used by PSO2 materials               |
| Shape Adjust   | Sliders for editing the body shape                      |

##### Shape Adjust

These sliders edit the body the same way the in-game customization does, in scale, position, and rotation around 1.0, for the breasts, clavicles, thighs, hips, and pelvis. **Export AQM** writes the result as a shape-adjust motion; **Load AQM** reads one back in.

A Blender skeleton has two layers: the rest pose the mesh is built on, and a pose on top of it. The sliders write the body shape into the pose, and animations use that same pose, so importing a motion overwrites the shape and exporting one bakes the shape into the motion.

**Apply Shape to Rest Pose** moves the shape down into the skeleton itself and leaves the pose free for animation, which is how the game works: proportions reshape the skeleton, motions play on top of it. The usual order of work is:

1. Import a model, then a character file.
2. Adjust the sliders.
3. Export the shape adjust motion, if you are making one.
4. **Apply Shape to Rest Pose.**
5. Import or export animations.

Step 4 changes mesh data permanently and the sliders can no longer edit the shape afterwards, so re-import the character file if you need to change it. The panel warns you whenever a shape is sitting in the pose, and the **How This Works** section repeats this summary in Blender.

#### PSO2 Ornaments

Clicking the buttons here will show or hide meshes that are associated with toggleable ornaments.

### Working With Aqua-Toolset

The import/export functions of this add-on combine the FBX import/export from [Aqua-Toolset](https://github.com/Shadowth117/Aqua-Toolset) with Blender's FBX import/export. However, bones are renamed from the format used by Aqua-Toolset for better compatibility with Blender functions such as mirroring vertex weights. Aqua-Toolset exports bones in the format `(id)name#flags`, and this add-on strips the `(id)` prefixes and moves them to `pso2_bone_id` custom properties on each bone. These IDs are added back to the bone names when exporting to `.aqp` format.

If you have a model you imported from Aqua-Toolset's FBX export, you can run the **PSO2 bone IDs to properties** operator to move the bone IDs from names to custom properties.

If you want to export to FBX, you can run the **PSO2 bone IDs to names** operator to put the bone IDs back at the start of bone names.

## Development

To build and develop the extension, first install the following requirements:

- [Blender 5.1](https://www.blender.org/download/releases/) or newer.
- [uv](https://github.com/astral-sh/uv)
- [Visual Studio](https://visualstudio.microsoft.com/vs/community/) with the C# and C++ workflows installed.
- [.NET SDK 9.0](https://dotnet.microsoft.com/en-us/download/dotnet/9.0)
- [Autodesk FBX SDK](https://www.autodesk.com/content/dam/autodesk/www/adn/fbx/2020-1/fbx20201_fbxsdk_vs2017_win.exe) version 2020.1

First, clone the repo with submodules:

```pwsh
git clone --recurse https://github.com/dummycount/blender_pso2_tools.git
cd blender_pso2_tools
```

Then run the following commands to set up the development environment:

```pwsh
# Create a virtual environment
uv venv .venv
# You can optionally run the "activate" command this prints. Then you don't need
# to prefix some commands below with "uv run".

# Install Python modules needed for development
uv pip install .
# Set up Git hooks to format files
uv run prek install

# Download Python wheels for dependencies.
uv run scripts/wheels.py
# Build binaries needed by the add-on.
uv run scripts/build_bin.py
# Generate Python typings for the above binaries.
# (This will probably fail, but it will generate some useful typings first.)
uv run scripts/build_typings.py
```

[scripts/wheels.py](scripts/wheels.py) defines the Python dependencies used by the add-on. This script needs to be run any time the dependencies are updated, and [pso2_tools/blender_manifest.toml](pso2_tools/blender_manifest.toml) needs to be updated to list all the wheel files.

[scripts/build_bin.py](scripts/build_bin.py) needs to be run any time the PSO2-Aqua-Library submodule is updated. The `PACKAGES` array at the top also needs to be kept in sync with any nuget packages used by Aqua Library.

[scripts/build_typings.py](scripts/build_typings.py) does not need to be run for the add-on to function, but it generates typings that can be helpful when editing in an IDE.

To build and install the add-on, run:

```pwsh
uv run scripts/install.py --editable
```

This will install the add-on in Blender, then symlink it back to this repo. The extension will automatically reload itself when it detects a change to its own files. (This usually crashes Blender after a while, so save your work often.)

Run the script without `--editable` to install the add-on without a symlink.

To build the add-on without installing it, e.g. for a release, run:

```pwsh
uv run scripts/build_package.py
```

## Acknowledgements

This fork exists because two people spent years on this and shared all of it.

- **[dummycount](https://github.com/dummycount)** wrote this add-on. Everything in this fork is his design, extended.
- **[Shadowth117](https://github.com/Shadowth117)** reverse engineered the formats themselves: [PSO2-Aqua-Library](https://github.com/Shadowth117/PSO2-Aqua-Library), [Aqua-Toolset](https://github.com/Shadowth117/Aqua-Toolset), and [Zamboni](https://github.com/Shadowth117/Zamboni). Every model, motion, and archive this add-on touches goes through his work.

Boundless gratitude to both.
