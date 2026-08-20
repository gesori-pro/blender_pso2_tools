"""Import a whole character from one .fnp: every part, its colours and shape.

A character file names dozens of parts by id - costume, hair, eyes, ears and
so on - but nothing loads them together. This walks the file's part ids, pulls
each one out of the game data through the object database, and drops the whole
character into the scene in one step. Accessories are left out by default: a
character often stacks a dozen and they rarely help a base import.

Parts come in two shapes. Most (hair, costume, ears, teeth) carry their own
model, so they import like any object. Eyes, eyebrows and eyelashes are
texture-only - their mesh already lives in the face model - so their textures
are wired onto the face's materials instead.

Colours are read into the scene *before* any model loads, because a material
bakes the scene colour it sees at import time; setting it afterwards leaves the
already-imported look untouched. Body proportions are applied last, once the
body has brought an armature into the scene to pose.
"""

import tempfile
from contextlib import closing
from pathlib import Path

import bpy
from bpy_extras.io_utils import ImportHelper

from . import (
    char_colors,
    charfile,
    classes,
    ice,
    import_fnp,
    import_model,
    objects,
    proportions,
)
from .debug import debug_print
from .preferences import get_preferences
from .util import OperatorResult

# Character-file field suffix -> object-database getter, in load order. The
# body parts come first so the proportion pass has an armature to pose. The
# four body slots are grouped so "Body / Outfit" can switch them off together.
_BODY_PARTS = (
    ("basewearPart", "get_basewear"),
    ("costumePart", "get_costumes"),
    ("innerwearPart", "get_innerwear"),
    ("outerwearPart", "get_outerwear"),
)
_HEAD_PARTS = (
    ("faceTypePart", "get_faces"),
    ("hairPart", "get_hair"),
    ("earsPart", "get_ears"),
    ("teethPart", "get_teeth"),
    ("hornPart", "get_horns"),
)

# Texture-only face parts: field suffix -> getter, the material name fragments
# to paint, and a fragment to skip. The eyelash shadow material carries the
# skin texture, not the eyelash one, so it is left alone.
_FACE_TEXTURE_PARTS = (
    ("eyePart", "get_eyes", ("eye_l", "eye_r"), None),
    ("eyebrowPart", "get_eyebrows", ("eyebrow",), None),
    ("eyelashPart", "get_eyelashes", ("eyelash",), "shadow"),
)

# Texture file suffix -> the image node that carries it, matching what the
# model importer wires for skin (Diffuse<-_d, Color Mask<-_m, ...).
_TEXTURE_NODES = {"d": "Diffuse", "m": "Color Mask", "s": "Multi Map", "n": "Normal Map"}


def _find_part_id(char: charfile.CharacterFile, suffix: str) -> int:
    """The part id stored under any of the ...Part fields ending in `suffix`.

    Part selections are spread across baseSLCT, baseSLCT2 and baseSLCTNGS
    depending on when NGS added them, so the block prefix is matched loosely.
    Returns 0 when the character has nothing in that slot.
    """
    for name in char:
        if name.split(".")[-1] == suffix:
            value = char[name]
            if isinstance(value, int):
                return value
    return 0


def _paint_face_textures(
    obj: objects.CmxObjectBase,
    data_path: Path,
    fragments: tuple[str, ...],
    skip: str | None,
) -> bool:
    """Wire a texture-only part's images onto the face's materials.

    Eyes/brows/lashes ship as a bare set of `_d/_m/_s/_n` textures with no
    model, so the face model's matching materials are painted with them
    directly. Returns whether any image was placed.
    """
    files = obj.get_files()
    if not files:
        return False

    ice_path = import_model._get_ice_path(files[0], data_path, True)
    if ice_path is None or not ice_path.exists():
        return False

    images: dict[str, bpy.types.Image] = {}
    with tempfile.TemporaryDirectory(prefix="pso2_char_") as tmp:
        for entry in ice.IceFile.load(ice_path).get_files():
            if not entry.name.lower().endswith(".dds"):
                continue
            out = Path(tmp) / entry.name
            out.write_bytes(entry.data)
            suffix = entry.name.rsplit("_", 1)[-1].split(".")[0].lower()
            image = bpy.data.images.load(str(out), check_existing=True)
            image.pack()  # the temp file is about to be removed
            images[suffix] = image

    if not images:
        return False

    painted = False
    for material in bpy.data.materials:
        name = material.name.lower()
        if not material.use_nodes:
            continue
        if skip and skip in name:
            continue
        if not any(fragment in name for fragment in fragments):
            continue

        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE":
                continue
            for suffix, label in _TEXTURE_NODES.items():
                if node.label == label and suffix in images:
                    node.image = images[suffix]
                    painted = True

    return painted


@classes.register
class PSO2_OT_ImportCharacter(  # type: ignore https://github.com/nutti/fake-bpy-module/issues/376
    bpy.types.Operator, ImportHelper
):
    """Load a whole PSO2 character from a .fnp file: parts, colours and body shape"""

    bl_label = "Import Character (.fnp)"
    bl_idname = "pso2.import_character"
    bl_options = {"UNDO"}

    filename_ext = ".fnp"
    filter_glob: bpy.props.StringProperty(
        default="*.fnp;*.fkp;*.fcp", options={"HIDDEN"}
    )

    import_colors: bpy.props.BoolProperty(
        name="Colours",
        description="Read the character's skin, hair, eye and outfit colours from the file",
        default=True,
    )
    import_proportions: bpy.props.BoolProperty(
        name="Body Proportions",
        description="Pose the body's bones to the character's body-shape sliders",
        default=True,
    )
    include_body: bpy.props.BoolProperty(
        name="Body / Outfit",
        description="Load basewear, costume, innerwear and outerwear",
        default=True,
    )

    def draw(self, context):
        assert self.layout is not None
        self.layout.prop(self, "import_colors")
        self.layout.prop(self, "import_proportions")
        self.layout.prop(self, "include_body")

    def execute(self, context) -> OperatorResult:
        path = Path(self.filepath)  # type: ignore

        try:
            char = charfile.CharacterFile.load(path)
        except (OSError, ValueError, KeyError) as ex:
            self.report({"ERROR"}, f"{path.name}: {ex}")
            return {"CANCELLED"}

        data_path = get_preferences(context).get_pso2_data_path()

        # Colours first: a material bakes the scene colour it sees when it is
        # built, so they have to be in place before anything imports.
        if self.import_colors:
            char_colors.apply_to_scene(context, char)

        model_parts = _HEAD_PARTS + (_BODY_PARTS if self.include_body else ())
        loaded: list[str] = []
        missing: list[str] = []

        with closing(objects.ObjectDatabase(context)) as db:
            for suffix, getter in model_parts:
                part_id = _find_part_id(char, suffix)
                if part_id <= 0:
                    continue
                obj = next(iter(getattr(db, getter)(part_id)), None)
                if obj is None:
                    missing.append(f"{suffix}={part_id}")
                    continue
                import_model.import_object(self, context, obj, high_quality=True)
                loaded.append(obj.name)

            for suffix, getter, fragments, skip in _FACE_TEXTURE_PARTS:
                part_id = _find_part_id(char, suffix)
                if part_id <= 0:
                    continue
                obj = next(iter(getattr(db, getter)(part_id)), None)
                if obj is None:
                    missing.append(f"{suffix}={part_id}")
                    continue
                if _paint_face_textures(obj, data_path, fragments, skip):
                    loaded.append(obj.name)
                else:
                    missing.append(f"{suffix}={part_id} (no face to paint)")

        if self.import_proportions:
            self._apply_proportions(context, char)

        if missing:
            shown = ", ".join(missing[:6])
            more = "..." if len(missing) > 6 else ""
            self.report(
                {"WARNING"},
                f"Loaded {len(loaded)} parts; {len(missing)} not found"
                f" ({shown}{more}).",
            )
        else:
            self.report({"INFO"}, f"Loaded {len(loaded)} parts.")

        return {"FINISHED"}

    def _apply_proportions(self, context, char: charfile.CharacterFile) -> None:
        """Pose every imported armature to the body-shape sliders.

        Each part keeps its own armature - body, head, hair, ears - and the
        proportion table matches by bone name, so it has to run on all of
        them. Posing only the body left the head at its bind size, which
        parted the neck; the shared spine/neck/head bones exist on the other
        armatures too, so the same pose keeps them lined up.
        """
        try:
            result = proportions.compute(char)
        except (OSError, KeyError, ValueError) as ex:
            debug_print("Could not compute proportions:", ex)
            return

        posed = 0
        for obj in context.scene.objects:
            if obj.type != "ARMATURE":
                continue
            summary = import_fnp.apply_proportions(obj, result["bones"])
            if summary["applied"]:
                posed += 1
                debug_print(f"Posed {obj.name} ({summary['applied']} bones)")

        debug_print(f"Applied proportions to {posed} armatures")
