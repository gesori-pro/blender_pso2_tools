"""Run with Blender --background --factory-startup --python-exit-code 1.

PSO2_TEST_DEPENDENCIES may point at Blender's existing Python dependencies.
PSO2_TEST_FACE_MOTION may point at an offline extracted, real face AQM.
PSO2_TEST_EXTRA_FACE_MOTIONS may contain more AQM paths, separated by os.pathsep.
PSO2_TEST_FACE_MODEL_ICE and PSO2_TEST_GAME_DATA enable a real face-model test.
No installed extension, game file, open scene or user preference is changed.
"""

import json
import os
import sys
from pathlib import Path

import bpy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.getenv("PSO2_TEST_ADDON_ROOT", str(REPO)))
if deps := os.getenv("PSO2_TEST_DEPENDENCIES"):
    sys.path.append(deps)
bpy.ops.preferences.addon_enable(module="pso2_tools")

from pso2_tools import aqm, charfile, face_controls, face_shape, scene_props  # noqa: E402

checks = []


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def character(version):
    layout = charfile.pso2_xxp.load_layout(
        REPO / "pso2_tools" / "charfile" / f"xxpv{version}_layout.json"
    )
    size = max(f.offset + f.width * f.count for f in layout.values())
    char = charfile.CharacterFile(bytes(size), layout, version)
    char["baseSLCT.faceTypePart"] = 100070
    return char


def face(name):
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_name in ("eyestop", "nosetop", "lip_l", "lip_r", "jaw"):
        bone = data.edit_bones.new(bone_name + "#1C0#0")
        bone.head = (0, 0, 0)
        bone.tail = (0, 1, 0)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj[scene_props.BONE_AXES] = "Y,X"
    return obj


def basis(obj):
    return {
        p.name: tuple(v for row in p.matrix_basis for v in row) for p in obj.pose.bones
    }


def weights(obj):
    return [
        (v.index, [(g.group, g.weight) for g in v.groups]) for v in obj.data.vertices
    ]


def vertex(obj):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return tuple(evaluated.data.vertices[0].co)


def control(field):
    return next(key for key, item, *_ in face_controls.CONTROLS if item == field)


mock = aqm.AqmMotion(0x10012, 0, 164, 30, 5)
for index, name in enumerate(("eyestop", "nosetop", "lip_l", "lip_r", "jaw")):
    pairs = {
        "lip_l": [
            (98, (-0.01, 0, 0, 0)),
            (100, (0.01, 0, 0, 0)),
            (117, (0, -0.005, 0, 0)),
            (119, (0, 0.005, 0, 0)),
        ],
        "lip_r": [(106, (-0.02, 0, 0, 0)), (108, (0.02, 0, 0, 0))],
        "jaw": [(130, (0, -0.03, 0, 0))],
    }.get(name, [])
    key = aqm.AqmKeySet(
        1,
        1,
        0,
        [0] + [frame * 16 for frame, _ in pairs],
        [(0, 0, 0, 0)] + [value for _, value in pairs],
    )
    mock.nodes.append(aqm.AqmNode(2, index, name, [key]))

loader = face_shape._load_motion
face_shape._load_motion = lambda context, face_id: mock
try:
    first = face("Face A")
    second = face("Face B")
    original = basis(first)
    other = basis(second)
    char = character(16)
    char["faceNatural.expStruct.expStruct.leftMouthVertical"] = 127
    char["faceSmile.expStruct.expStruct.leftMouthVertical"] = 64
    check(
        "V16 nested expression read",
        face_shape.slider(char, "<expr>.leftMouthVertical", "faceNatural") == 127,
    )
    face_controls.load(bpy.context, first, char)
    settings = getattr(first, face_controls.SETTINGS)
    left = control("<expr>.leftMouthVertical")
    check("loaded left corner weight", getattr(settings, left) == 127)
    check(
        "mouth-corner endpoint",
        abs(first.pose.bones["lip_l#1C0#0"].location.x - 0.01) < 1e-7,
    )
    check("second character unchanged on load", basis(second) == other)

    data = bpy.data.meshes.new("Weighted lip")
    data.from_pydata([(0, 0, 0), (0.02, 0, 0), (0, 0.02, 0)], [], [(0, 1, 2)])
    mesh = bpy.data.objects.new("Weighted lip", data)
    bpy.context.collection.objects.link(mesh)
    mesh.parent = first
    group = mesh.vertex_groups.new(name="lip_l#1C0#0")
    group.add([0, 1, 2], 1, "REPLACE")
    mesh.modifiers.new("Face", "ARMATURE").object = first
    initial_weights = weights(mesh)
    posed = vertex(mesh)
    settings.expression = "faceNatural"
    for _ in range(10):
        for value in (-127, 0, 127):
            setattr(settings, left, value)
            expected = 0.01 * value / 127
            check(
                f"rebuild corner {len(checks)}",
                abs(first.pose.bones["lip_l#1C0#0"].location.x - expected) < 1e-7,
            )
    check("idempotent mesh position", vertex(mesh) == posed)
    check("vertex weights unchanged", weights(mesh) == initial_weights)
    check("inactive rig not touched by callback", basis(second) == other)

    settings.expression = "faceSmile"
    check("preset loads its own weights", getattr(settings, left) == 64)
    setattr(settings, left, 40)
    settings.expression = "faceNatural"
    check("previous preset preserved", getattr(settings, left) == 127)
    settings.expression = "faceSmile"
    check("edited preset preserved", getattr(settings, left) == 40)
    bpy.context.view_layer.objects.active = first
    check("neutral operator", bpy.ops.pso2.reset_face_customization() == {"FINISHED"})
    check(
        "neutral removes previous deformation",
        abs(first.pose.bones["lip_l#1C0#0"].location.x) < 1e-7,
    )
    check("clear operator", bpy.ops.pso2.clear_face_customization() == {"FINISHED"})
    check("clear restores original pose", basis(first) == original)
    check("clear removes persistent source", face_controls.SOURCE not in first)

    old = character(10)
    old["faceNatural.leftMouthVertical"] = -127
    face_controls.load(bpy.context, first, old)
    check("V10 flat expression read", getattr(settings, left) == -127)
    check(
        "unsupported tongue field absent",
        "faceNatural.tongueHorizontal" not in first[face_controls.SOURCE],
    )
    check(
        "unimplemented pupil and lip-sync controls excluded",
        not any(
            x[1] == "<expr>." + name
            for x in face_controls.CONTROLS
            for name in face_controls.UNSUPPORTED
        ),
    )
    check("51 independent controls", len(face_controls.CONTROLS) == 51)
    check(
        "new controls appended without renumbering the original 39",
        [item[1] for item in face_controls.CONTROLS[39:]]
        == [item[0] for item in face_shape.BORG_SLIDERS],
    )
    new_key = control("<expr>.mouthOpenClose")
    old_pose = basis(first)
    setattr(settings, new_key, 100)
    check(
        "missing V10 field cannot be invented by an edit",
        getattr(settings, new_key) == 0,
    )
    check("missing field edit preserves pose", basis(first) == old_pose)
    bpy.ops.pso2.clear_face_customization()
    new_char = character(16)
    new_char["faceNatural.mouthOpenClose"] = -4
    new_char["faceNatural.mouthCornerPosition"] = -54
    new_char["faceSmile.mouthOpenClose"] = 64
    face_controls.load(bpy.context, first, new_char)
    check("new direct V16 field read", getattr(settings, new_key) == -4)
    check(
        "new motion availability",
        first[face_controls.AVAILABLE]["<expr>.mouthOpenClose"],
    )
    check(
        "missing deformation disabled despite a saved file value",
        not first[face_controls.AVAILABLE]["<expr>.mouthCornerPosition"],
    )
    blocked_key = control("<expr>.mouthCornerPosition")
    before = basis(first)
    setattr(settings, blocked_key, 100)
    check(
        "unavailable motion preserves saved value",
        getattr(settings, blocked_key) == -54,
    )
    check("unavailable motion preserves pose", basis(first) == before)
    for value in (-127, 0, 127, 0):
        setattr(settings, new_key, value)
        check(
            f"new open-close endpoint / zero {value}",
            abs(first.pose.bones["lip_l#1C0#0"].location.y - 0.005 * value / 127)
            < 1e-7,
        )
    settings.expression = "faceSmile"
    check("new expression preset value read", getattr(settings, new_key) == 64)
    settings.expression = "faceNatural"
    check("new expression edit retained", getattr(settings, new_key) == 0)
    bpy.ops.pso2.clear_face_customization()
    check("clear removes motion availability", face_controls.AVAILABLE not in first)
    # The rig was restored above; make the manually shaped head the baseline.
    bpy.context.view_layer.objects.active = first
    first.pose.bones["eyestop#1C0#0"].scale = (1.2, 1.2, 1.2)
    mock.nodes[0].key_sets[0].timings.append(164 * 16)
    mock.nodes[0].key_sets[0].vec4_keys.append((0, 0, 0, 0))
    modern = character(16)
    modern["faceNatural.expStruct.tongueHorizontal"] = 127
    face_controls.load(bpy.context, first, modern)
    check(
        "terminal neutral key preserves existing head shape",
        abs(first.pose.bones["eyestop#1C0#0"].scale.x - 1.2) < 1e-6,
    )
    check(
        "equivalent negative quaternion is not deformation",
        not face_shape._changed_keys(
            None, [0, 0, 0, -1], None, ([0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0])
        ),
    )
    face_controls.load(bpy.context, first, old)
    settings.expression = "faceSmile"
    # ZERO values are present in V10; missing is not the same as zero.
    check(
        "zero-valued controls are available",
        "faceSmile.leftMouthVertical" in first[face_controls.SOURCE],
    )

    before = basis(first)
    first.animation_data_create().action = bpy.data.actions.new("Do not overwrite")
    try:
        face_controls.load(bpy.context, first, old)
        raise AssertionError("animated face accepted")
    except ValueError as error:
        check("active animation refused", "action/NLA" in str(error))
    check("rejected load preserves pose", basis(first) == before)
    first.animation_data_clear()
finally:
    face_shape._load_motion = loader

real = None
if path := os.getenv("PSO2_TEST_FACE_MOTION"):
    motion = aqm.read_aqm(path)
    result = {}
    for name, _frames in (
        ("leftMouthVertical", (98, 100)),
        ("rightMouthVertical", (106, 108)),
        *(
            (field.removeprefix("<expr>."), (low, high))
            for field, low, high in face_shape.BORG_SLIDERS
        ),
    ):
        char = {"faceNatural." + name: 0}
        neutral = face_shape._blend(char, motion, "faceNatural")
        endpoints = []
        for value in (-127, 127):
            char["faceNatural." + name] = value
            target = face_shape._blend(char, motion, "faceNatural")
            changed = []
            for node in motion.nodes:
                if node.name not in target:
                    continue
                base = neutral.get(node.name)
                if base is None:
                    base_position = face_shape._key_at(node, 1, 0) or [0, 0, 0, 0]
                    base_scale = face_shape._key_at(node, 3, 0) or [1, 1, 1, 0]
                    q = face_shape._key_at(node, 2, 0) or [0, 0, 0, 1]
                    base_rotation = (q[3], q[0], q[1], q[2])
                else:
                    base_position, base_scale = base["position"], base["scale"]
                    base_rotation = base["rotation"]
                entry = target[node.name]
                error = max(
                    *(
                        abs(a - b)
                        for a, b in zip(
                            entry["position"], base_position[:3], strict=True
                        )
                    ),
                    *(
                        abs(a - b)
                        for a, b in zip(entry["scale"], base_scale[:3], strict=True)
                    ),
                )
                q = entry["rotation"]
                rotation_error = min(
                    max(abs(a - b) for a, b in zip(q, base_rotation, strict=True)),
                    max(abs(a + b) for a, b in zip(q, base_rotation, strict=True)),
                )
                error = max(error, rotation_error)
                if error > 1e-7:
                    changed.append(node.name)
            check(f"real {name} {value} changes bones", bool(changed))
            if name in {
                field.removeprefix("<expr>.")
                for field, *_ in face_shape.BORG_SLIDERS[:6]
            }:
                check(
                    f"real {name} {value} affects only lips",
                    all(bone.startswith("lip_") for bone in changed),
                )
            endpoints.append({"value": value, "changed_bones": changed})
        result[name] = endpoints
    real = {
        "end_frame": motion.end_frame,
        "node_count": motion.node_count,
        "corners": result,
    }

for path in os.getenv("PSO2_TEST_EXTRA_FACE_MOTIONS", "").split(os.pathsep):
    if not path:
        continue
    motion = aqm.read_aqm(path)
    availability = {
        field: face_shape.has_slider_motion(motion, field)
        for field, *_ in face_shape.BORG_SLIDERS
    }
    print(
        "EXTRA_FACE_MOTION",
        json.dumps(
            {"path": path, "end_frame": motion.end_frame, "available": availability}
        ),
    )
    # Bone count / face ID alone does not establish control availability.
    if "200070" in Path(path).name:
        check("200070 has lip opening keys", availability["<expr>.mouthOpenClose"])
        check(
            "200070 has no upper-lip-shape keys",
            not availability["<expr>.upperLipShape"],
        )
    elif "100000" not in Path(path).name:
        for field, *_ in face_shape.BORG_SLIDERS:
            check(f"extra motion supports {field}", availability[field])

print("FACE_TEST_RESULT", json.dumps({"checks": len(checks), "real_motion": real}))

if model_path := os.getenv("PSO2_TEST_FACE_MODEL_ICE"):
    from pso2_tools import ice, import_model, preferences

    class Reporter:
        def report(self, level, text):
            print("MODEL_IMPORT", level, text)

    preferences.get_preferences(bpy.context).pso2_data_path = os.environ[
        "PSO2_TEST_GAME_DATA"
    ]
    before_objects = {obj.as_pointer() for obj in bpy.data.objects}
    result = import_model._import_models(
        Reporter(),
        bpy.context,
        [ice.IceFile.load(Path(model_path))],
        options={"use_anim": False},
        use_t2_skin=False,
    )
    check("real face-model import", result == {"FINISHED"})
    imported = [
        obj for obj in bpy.data.objects if obj.as_pointer() not in before_objects
    ]
    rig = next(obj for obj in imported if face_controls.is_face(obj))
    meshes = [obj for obj in imported if obj.type == "MESH"]
    saved_pose = basis(rig)
    saved_weights = [weights(obj) for obj in meshes]
    face_controls.load(bpy.context, rig, character(16))
    settings = getattr(rig, face_controls.SETTINGS)

    def vertices():
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        return [
            tuple(v.co)
            for obj in meshes
            for v in obj.evaluated_get(graph).data.vertices
        ]

    neutral_vertices = vertices()
    displacements = []
    required_mesh_controls = {
        "leftMouthVertical",
        "rightMouthVertical",
        "lowerJawOpenClose",
        *(field.removeprefix("<expr>.") for field, *_ in face_shape.BORG_SLIDERS[:6]),
    }
    for field in (
        "leftMouthVertical",
        "rightMouthVertical",
        *(field.removeprefix("<expr>.") for field, *_ in face_shape.BORG_SLIDERS),
    ):
        key = control("<expr>." + field)
        for value in (-127, 127):
            setattr(settings, key, value)
            moved = vertices()
            distances = [
                sum((a - b) ** 2 for a, b in zip(p, q, strict=True)) ** 0.5
                for p, q in zip(moved, neutral_vertices, strict=True)
            ]
            changed = sum(distance > 1e-7 for distance in distances)
            if field in required_mesh_controls:
                check(f"real mesh {field} {value} moves", changed > 0)
            displacements.append(
                {
                    "field": field,
                    "value": value,
                    "vertices": changed,
                    "max_meters": max(distances),
                }
            )
            setattr(settings, key, 0)
            returned = vertices()
            error = max(
                abs(a - b)
                for p, q in zip(returned, neutral_vertices, strict=True)
                for a, b in zip(p, q, strict=True)
            )
            check("real mesh returns to neutral", error < 1e-7)
    check(
        "real vertex weights preserved",
        saved_weights == [weights(obj) for obj in meshes],
    )
    bpy.context.view_layer.objects.active = rig
    bpy.ops.pso2.clear_face_customization()
    restore_error = max(
        abs(a - b)
        for name, pose in saved_pose.items()
        for a, b in zip(pose, basis(rig)[name], strict=True)
    )
    check("real face pose restored", restore_error < 1e-6)
    print(
        "REAL_FACE_MESH_RESULT",
        json.dumps(
            {
                "checks": len(checks),
                "bones": len(rig.pose.bones),
                "vertices": len(neutral_vertices),
                "changes": displacements,
                "restore_error": restore_error,
            }
        ),
    )

    if output := os.getenv("PSO2_TEST_OUTPUT"):
        output_path = Path(output)
        if output_path.exists():
            raise ValueError("Refusing to overwrite an existing validation scene.")
        face_controls.load(bpy.context, rig, character(16))
        settings.expression = "faceSmile"
        setattr(settings, control("<expr>.leftMouthVertical"), 64)
        setattr(settings, control("<expr>.mouthCornerPosition"), -54)
        stored_pose = basis(rig)
        rig_name = rig.name
        bpy.ops.wm.save_as_mainfile(filepath=str(output_path), check_existing=False)
        bpy.ops.wm.open_mainfile(filepath=str(output_path))
        rig = bpy.data.objects[rig_name]
        settings = getattr(rig, face_controls.SETTINGS)
        check(
            "saved face settings reload",
            settings.loaded and settings.expression == "faceSmile",
        )
        check(
            "saved corner weight reload",
            getattr(settings, control("<expr>.leftMouthVertical")) == 64,
        )
        check(
            "saved source weights reload",
            rig[face_controls.SOURCE]["faceSmile.leftMouthVertical"] == 64,
        )
        check("saved face pose reload", basis(rig) == stored_pose)
        check(
            "saved new lip value reload",
            getattr(settings, control("<expr>.mouthCornerPosition")) == -54
            and rig[face_controls.SOURCE]["faceSmile.mouthCornerPosition"] == -54,
        )
        check(
            "saved motion availability reload",
            rig[face_controls.AVAILABLE]["<expr>.mouthOpenClose"],
        )
        print(
            "FACE_PERSISTENCE_RESULT",
            json.dumps({"checks": len(checks), "scene": str(output_path)}),
        )
