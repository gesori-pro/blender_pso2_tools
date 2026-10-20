"""Blender inspection of calculated native poses, including affine shear."""
from .translations import report as _report


import json
from dataclasses import asdict
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper

from . import aqm, charfile, classes, game_skinning, import_aqm, native_pose_profile

PROFILE_PROP = "pso2_native_pose_input_text"


def create_preview(context, data, result):
    """Use calculated vertex positions so Blender TRS cannot discard shear."""
    collection = bpy.data.collections.new("PSO2 Native Pose Calculation")
    context.scene.collection.children.link(collection)
    profile = bpy.data.texts.new("PSO2 Native Pose Inputs.json")
    profile.write(json.dumps(data, ensure_ascii=False, indent=2))
    report = bpy.data.texts.new("PSO2 Native Pose Result.json")
    report.write(json.dumps(result, ensure_ascii=False, indent=2))
    rig = import_aqm._find_target_armature(context)
    objects = []
    for index, (source, calculated) in enumerate(
        zip(data["meshes"], result["meshes"], strict=True)
    ):
        mesh = bpy.data.meshes.new(source["name"] + " [Native Calculation]")
        mesh.from_pydata(source["positions"], [], source["triangles"])
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        collection.objects.link(obj)
        if rig:
            obj.matrix_world = rig.matrix_world.copy()
        else:
            obj.rotation_euler.x = np.pi / 2
        obj[game_skinning.PREVIEW_PROP] = "NATIVE_CALCULATION"
        obj[PROFILE_PROP] = profile.name
        obj["pso2_native_pose_result_text"] = report.name
        obj["pso2_native_mesh_index"] = index
        obj.shape_key_add(name="Edited bind input")
        key = obj.shape_key_add(name="Calculated native pose")
        key.data.foreach_set(
            "co", np.asarray(calculated["positions"], dtype=np.float32).ravel()
        )
        key.value = 1
        if "gpu_error_mm" in calculated:
            attr = mesh.attributes.new("pso2_native_gpu_error_mm", "FLOAT", "POINT")
            attr.data.foreach_set("value", calculated["gpu_error_mm"])
            obj["pso2_native_gpu_max_error_mm"] = calculated["gpu_max_error_mm"]
        for channel in (1, 2):
            uv = source.get(f"uv{channel}")
            if uv:
                layer = mesh.uv_layers.new(name=f"UVChannel_{channel}")
                layer.data.foreach_set(
                    "uv", [x for loop in mesh.loops for x in uv[loop.vertex_index]]
                )
        material = bpy.data.materials.get(source.get("material_name", ""))
        if material:
            mesh.materials.append(material)
        objects.append(obj)
    return objects


@classes.register
class PSO2_OT_ImportNativePose(bpy.types.Operator, ImportHelper):
    """Calculate game FK and skinning from explicit native local pose inputs"""

    bl_idname = "pso2.import_native_pose_inputs"
    bl_label = "Calculate Native Pose Inputs"
    bl_options = {"UNDO"}
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(
        default="*.native-pose.json", options={"HIDDEN"}
    )

    def execute(self, context):
        try:
            data, result = native_pose_profile.load_profile(self.filepath)
            create_preview(context, data, result)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            _report('Native pose calculated; local inputs and result are in the Text Editor'),
        )
        return {"FINISHED"}


def recalculate_preview(context, text_name, data):
    result = native_pose_profile.evaluate_profile(data)
    affected = [
        o
        for o in context.scene.objects
        if o.type == "MESH" and o.get(PROFILE_PROP) == text_name
    ]
    if not affected:
        raise ValueError("No native preview meshes use these inputs")
    for obj in affected:
        index = obj["pso2_native_mesh_index"]
        key = obj.data.shape_keys.key_blocks["Calculated native pose"]
        if len(key.data) != len(result["meshes"][index]["positions"]):
            raise ValueError("Input vertex count changed; import a new profile")
    for obj in affected:
        record = result["meshes"][obj["pso2_native_mesh_index"]]
        obj.data.shape_keys.key_blocks["Calculated native pose"].data.foreach_set(
            "co", np.asarray(record["positions"], dtype=np.float32).ravel()
        )
        if "gpu_error_mm" in record:
            obj.data.attributes["pso2_native_gpu_error_mm"].data.foreach_set(
                "value", record["gpu_error_mm"]
            )
            obj["pso2_native_gpu_max_error_mm"] = record["gpu_max_error_mm"]
        report = bpy.data.texts[obj["pso2_native_pose_result_text"]]
        report.clear()
        report.write(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@classes.register
class PSO2_OT_RecalculateNativePose(bpy.types.Operator):
    """Recalculate after editing the local pose input text"""

    bl_idname = "pso2.recalculate_native_pose"
    bl_label = "Recalculate Native Pose"
    bl_options = {"UNDO"}

    @classmethod
    def poll(cls, context):
        return context.object is not None and bool(context.object.get(PROFILE_PROP))

    def execute(self, context):
        try:
            name = context.object[PROFILE_PROP]
            recalculate_preview(
                context, name, json.loads(bpy.data.texts[name].as_string())
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}


@classes.register
class PSO2_OT_ImportNativeFnp(bpy.types.Operator, ImportHelper):
    """Recalculate the selected source profile from a character file"""

    bl_idname = "pso2.import_native_fnp"
    bl_label = "Import Character for Native Pose Preview"
    bl_options = {"UNDO"}
    filename_ext = ".fnp"
    filter_glob: bpy.props.StringProperty(
        default="*.fnp;*.mhp;*.fcp;*.mcp;*.fhp;*.fnp?;*.mhp?", options={"HIDDEN"}
    )

    @classmethod
    def poll(cls, context):
        return context.object is not None and bool(context.object.get(PROFILE_PROP))

    def execute(self, context):
        try:
            name = context.object[PROFILE_PROP]
            data = json.loads(bpy.data.texts[name].as_string())
            if not data.get("body_source"):
                raise ValueError(
                    "Load a source-derived body profile before importing a character"
                )
            character = charfile.CharacterFile.load(Path(self.filepath))
            old = data["body_source"]["character_fields"]
            if old["baseDOC.gender"] != character["baseDOC.gender"]:
                raise ValueError("This character requires a profile for its body type")
            data["body_source"]["character_fields"] = character.to_dict()
            data["body_source"]["character_path"] = str(Path(self.filepath).resolve())
            recalculate_preview(context, name, data)
            text = bpy.data.texts[name]
            text.clear()
            text.write(json.dumps(data, ensure_ascii=False, indent=2))
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report(
            {"INFO"}, _report("Character pose recalculated for the profile's target outfit")
        )
        return {"FINISHED"}


@classes.register
class PSO2_OT_ImportNativeAqm(bpy.types.Operator, ImportHelper):
    """Sample a source AQM with game interpolation on the native preview"""

    bl_idname = "pso2.import_native_aqm"
    bl_label = "Import Motion for Native Pose Preview"
    bl_options = {"UNDO"}
    filename_ext = ".aqm"
    filter_glob: bpy.props.StringProperty(default="*.aqm", options={"HIDDEN"})
    frame: bpy.props.FloatProperty(name="Frame", default=0, min=0, translation_context="PSO2Tools")
    in_place: bpy.props.BoolProperty(name="In Place", default=True, translation_context="PSO2Tools")

    @classmethod
    def poll(cls, context):
        return context.object is not None and bool(context.object.get(PROFILE_PROP))

    def execute(self, context):
        try:
            name = context.object[PROFILE_PROP]
            data = json.loads(bpy.data.texts[name].as_string())
            motion = aqm.read_aqm(Path(self.filepath))
            if (
                motion.is_camera_motion
                or motion.is_material_motion
                or motion.is_shape_adjust
            ):
                raise ValueError(
                    "Select a skeletal animation; shape adjustment uses its own source layer"
                )
            if not motion.nodes:
                raise ValueError("Motion has no skeletal nodes")
            data["first_motion"] = {
                "motion": asdict(motion),
                "frame": self.frame,
                "source_name_map": {motion.nodes[0].name: data["node_names"][0]},
                "root_motion_flags": 15,
                "apply_extracted_root": not self.in_place,
                "source_path": str(Path(self.filepath).resolve()),
            }
            recalculate_preview(context, name, data)
            text = bpy.data.texts[name]
            text.clear()
            text.write(json.dumps(data, ensure_ascii=False, indent=2))
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            aqm.AqmError,
        ) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}
