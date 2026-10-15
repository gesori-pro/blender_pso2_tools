"""Character part lookup regression tests in a fresh background Blender.

Run with --background --factory-startup --python-exit-code 1 --python.
PSO2_TEST_DEPENDENCIES and PSO2_TEST_ADDON_ROOT select existing dependencies
and an optional installed addon. PSO2_TEST_OBJECT_DB opens an existing cache
read-only; PSO2_TEST_GAME_DATA verifies the resolved ICE files. Set
PSO2_TEST_IMPORT_MODEL=1 to import the reported outfit into the fresh scene.
No user scene, character file, preference or persistent cache is changed.
"""

import json
import os
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.getenv("PSO2_TEST_ADDON_ROOT", str(REPO)))
if deps := os.getenv("PSO2_TEST_DEPENDENCIES"):
    sys.path.append(deps)
bpy.ops.preferences.addon_enable(
    module=os.getenv("PSO2_TEST_ADDON_MODULE", "pso2_tools")
)

from pso2_tools import charfile, import_character, import_model, objects  # noqa: E402
from pso2_tools.preferences import get_preferences  # noqa: E402

checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def connect(path=":memory:"):
    if path == ":memory:":
        con = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    else:
        con = sqlite3.connect(
            Path(path).resolve().as_uri() + "?mode=ro",
            uri=True,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
    con.row_factory = sqlite3.Row
    con.create_function("md5", 1, objects.md5digest, deterministic=True)
    con.create_function("md5_ex", 1, objects.md5digest_ex, deterministic=True)
    return con


def database(con):
    # Bypass _open_db: it can create or migrate a user's persistent cache.
    db = objects.ObjectDatabase.__new__(objects.ObjectDatabase)
    db.context = bpy.context
    db.con = con
    return db


def character(version):
    layout = charfile.load_layout(
        REPO / "pso2_tools" / "charfile" / f"xxpv{version}_layout.json"
    )
    size = max(f.offset + f.width * f.count for f in layout.values())
    char = charfile.CharacterFile(bytes((size + 7) // 8 * 8), layout, version)
    char["baseSLCT2.basewearPart"] = 219421
    char["baseSLCT.costumePart"] = 30000
    char["baseSLCT2.innerwearPart"] = 200061
    return char


def body(object_type, item_id, file_name):
    cls = objects._object_types[object_type]
    return cls(
        object_type=object_type,
        id=item_id,
        adjusted_id=item_id,
        file=objects.CmxFileName(file_name),
    )


con = connect()
con.executescript(
    "".join(cls.db_schema(typ) for typ, cls in objects._object_types.items())
    + "".join(objects.CmxColorSets.db_schema(typ) for typ in objects._COLOR_SET_TYPES)
)
db = database(con)

for typ, base_id, variant_id, getter in (
    (objects.ObjectType.BASEWEAR, 219420, 219421, "get_basewear"),
    (objects.ObjectType.INNERWEAR, 200060, 200061, "get_innerwear"),
    (objects.ObjectType.OUTERWEAR, 201000, 201001, "get_outerwear"),
    (objects.ObjectType.COSTUME, 10000, 10001, "get_costumes"),
):
    model = body(typ, base_id, f"character/making/test_{typ}_{base_id}.ice")
    model.db_insert(con)
    objects.CmxColorSets(
        base_id,
        [objects.CmxColorSet(variant_id, "variant", "variant", 123, 456)],
    ).db_insert(con, typ)
    changes = con.total_changes
    selected = getattr(db, getter)(variant_id)
    check(f"{typ}: variant resolves to base model", len(selected) == 1)
    check(f"{typ}: original ICE metadata", selected[0].file.name == model.file.name)
    check(f"{typ}: original model ID", selected[0].id == base_id)
    check(f"{typ}: query does not write cache", con.total_changes == changes)
    check(f"{typ}: direct base lookup", getattr(db, getter)(base_id)[0].id == base_id)
    check(f"{typ}: unknown ID stays missing", not getattr(db, getter)(999999))
    check(
        f"{typ}: hash lookup unchanged",
        getattr(db, getter)(file_hash=model.file.hash)[0].id == base_id,
    )
    check(
        f"{typ}: unknown hash does not fall back to ID",
        not getattr(db, getter)(variant_id, file_hash="missing"),
    )
    exact = body(typ, variant_id, f"character/making/exact_{typ}.ice")
    exact.db_insert(con)
    check(
        f"{typ}: exact model wins over colour link",
        getattr(db, getter)(variant_id)[0].file.name == exact.file.name,
    )
    con.execute(f"DELETE FROM {typ} WHERE id=?", (variant_id,))

# Bad links and an ID shared by different categories must remain isolated.
objects.CmxColorSets(999998, [objects.CmxColorSet(999997, "", "", 0, 0)]).db_insert(
    con, objects.ObjectType.BASEWEAR
)
check("dangling colour link remains missing", not db.get_basewear(999997))
check("basewear colour link is not an accessory", not db.get_accessories(219421))
check("innerwear colour link is not skin", not db.get_skins(200061))
check("enumeration still returns model rows only", len(db.get_basewear()) == 1)

classic = body(objects.ObjectType.COSTUME, 12345, "character/making/costume.ice")
classic.db_insert(con)
outer = body(objects.ObjectType.OUTERWEAR, 12345, "character/making/outer.ice")
outer.db_insert(con)
lookup = import_character._lookup_model_part
check(
    "exact costume takes priority",
    lookup(db, "get_costumes", 12345).file.name == classic.file.name,
)
check(
    "outfit slot accepts linked outerwear variant",
    lookup(db, "get_costumes", 201001).object_type == objects.ObjectType.OUTERWEAR,
)
check("unknown outfit stays missing", lookup(db, "get_costumes", 987654) is None)

selections = import_character._model_part_selections
for version in range(10, 17):
    char = character(version)
    original = bytes(char.body)
    requested = list(selections(char, True))
    check(f"V{version}: reported slots omit empty costume", len(requested) == 2)
    check(
        f"V{version}: both variants resolve without missing parts",
        all(
            lookup(db, getter, part_id) is not None for _, getter, part_id in requested
        ),
    )
    check(
        f"V{version}: lookup leaves character unchanged", bytes(char.body) == original
    )
    check(f"V{version}: body disabled", not list(selections(char, False)))
    char["baseSLCT.faceTypePart"] = 30000
    check(
        f"V{version}: face ID 30000 is not skipped",
        list(selections(char, False)) == [("faceTypePart", "get_faces", 30000)],
    )
    char["baseSLCT.costumePart"] = 20000
    check(
        f"V{version}: T1 empty outerwear is not missing",
        all(suffix != "costumePart" for suffix, _, _ in selections(char, True)),
    )
    char["baseSLCT2.basewearPart"] = 0
    check(
        f"V{version}: no blanket numeric-ID filter",
        ("costumePart", "get_costumes", 20000) in list(selections(char, True)),
    )
    char["baseSLCT.costumePart"] = 12345
    check(
        f"V{version}: classic costume stays selected",
        ("costumePart", "get_costumes", 12345) in list(selections(char, True)),
    )
con.close()

if cache_path := os.getenv("PSO2_TEST_OBJECT_DB"):
    db = database(connect(cache_path))
    data_path = Path(os.environ["PSO2_TEST_GAME_DATA"])
    real_parts = []
    for getter, variant_id, base_id in (
        ("get_basewear", 219421, 219420),
        ("get_innerwear", 200061, 200060),
    ):
        selected = getattr(db, getter)(variant_id)
        check(f"actual cache: {variant_id} resolves", len(selected) == 1)
        check(
            f"actual cache: {variant_id} links to {base_id}", selected[0].id == base_id
        )
        item = selected[0]
        for quality in (False, True):
            path = import_model._get_ice_path(item.file, data_path, quality)
            check(f"actual ICE: {variant_id}, high_quality={quality}", path is not None)
        real_parts.append(
            {"selected": variant_id, "model": base_id, "ice": item.file.name}
        )
    requested = list(selections(character(16), True))
    check(
        "actual cache: all reported outfit parts resolve",
        all(
            lookup(db, getter, part_id) is not None for _, getter, part_id in requested
        ),
    )
    check("actual cache: no writes", db.con.total_changes == 0)
    print("REAL_PARTS=" + json.dumps(real_parts))
    db.close()

    if os.getenv("PSO2_TEST_IMPORT_MODEL") == "1":
        get_preferences(bpy.context).pso2_data_path = str(data_path)
        reports = []
        with TemporaryDirectory(prefix="pso2_parts_test_") as tmp:
            path = Path(tmp) / "variant-regression.fnp"
            character(16).save(path)
            operator = SimpleNamespace(
                filepath=str(path),
                import_colors=False,
                import_proportions=False,
                include_body=True,
                include_accessories=False,
                game_normals=False,
                expression="faceNatural",
                report=lambda level, text: reports.append((sorted(level), text)),
            )
            # All DB reads during this real import stay read-only. Preferences
            # and imported meshes belong only to this factory-startup process.
            with patch.object(
                objects.ObjectDatabase,
                "_open_db",
                staticmethod(lambda: connect(cache_path)),
            ):
                result = import_character.PSO2_OT_ImportCharacter.execute(
                    operator, bpy.context
                )
            check("actual Blender import finishes", result == {"FINISHED"})
            check(
                "actual Blender import has no missing-part warning",
                any(
                    level == ["INFO"] and text == "Loaded 2 parts."
                    for level, text in reports
                ),
            )
            check(
                "actual Blender import no errors",
                not any("ERROR" in level for level, _ in reports),
            )
            meshes = [
                obj
                for obj in bpy.data.objects
                if obj.type == "MESH" and obj.name != "Cube"
            ]
            check("actual Blender body meshes exist", bool(meshes))
            print(
                "REAL_IMPORT="
                + json.dumps(
                    {
                        "reports": reports,
                        "meshes": len(meshes),
                        "vertices": sum(len(obj.data.vertices) for obj in meshes),
                    },
                    ensure_ascii=True,
                )
            )

print(json.dumps({"passed": len(checks), "checks": checks}, ensure_ascii=True))
