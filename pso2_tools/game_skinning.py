from .translations import status as _status
"""Explicit capture snapshots and live reference-position diagnostics.

Reference preview carries source pose channels onto a separate rig whose
bind stays edited, but whose neutral local head positions come from an
explicit AQN. It models the observed anchor mismatch, not FLTD dynamics.
"""
from .translations import iface as _iface, report as _report


import json
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Vector

from . import classes, import_aqm, skinning_math

PREVIEW_PROP = "pso2_skinning_preview"
SOURCE_PROP = "pso2_skinning_source"
REFERENCE_PROP = "pso2_skinning_reference"


def _base(name):
    return name.split("#", 1)[0]


def read_reference_aqn(path):
    from . import dotnet

    dotnet.load()
    import System
    from AquaModelLibrary.Data.PSO2.Aqua import AquaNode

    nodes = AquaNode(System.Array[System.Byte](Path(path).read_bytes())).nodeList
    names = [str(node.boneName.GetString()) for node in nodes]
    if len(set(names)) != len(names):
        raise ValueError("Reference AQN has duplicate bone names")
    result = {}
    for node, name in zip(nodes, names, strict=True):
        rows = [
            [float(getattr(row, field)) for field in "XYZW"]
            for row in (node.m1, node.m2, node.m3, node.m4)
        ]
        inverse_bind = Matrix(rows).transposed()
        if abs(inverse_bind.determinant()) < 1e-12:
            raise ValueError(f"Singular inverse bind: {name}")
        parent = int(node.parentId)
        if parent >= len(names):
            raise ValueError(f"Invalid parent for {name}")
        result[name] = {
            "head": inverse_bind.inverted().translation,
            "parent": names[parent] if parent >= 0 else None,
        }
    return result


def reference_offsets(source, reference, prefixes=()):
    """Offsets in each edited bone's local location-channel coordinates.

    Convert the difference of parent-relative head displacements, not
    world head differences: shifting a whole chain should affect its root
    once. Children inherit the correction through Blender's own flags.
    """
    offsets = {}
    for bone in source.data.bones:
        name = _base(bone.name)
        if prefixes and not name.startswith(prefixes):
            continue
        record = reference.get(name)
        if record is None:
            continue
        parent_name = _base(bone.parent.name) if bone.parent else None
        if bone.parent:
            parent_ref = reference.get(parent_name)
            if parent_ref is None or record["parent"] != parent_name:
                raise ValueError(f"Reference hierarchy differs at {name}")
            reference_local = record["head"] - parent_ref["head"]
            edited_local = bone.head_local - bone.parent.head_local
        else:
            # Aqua's envelope root is absent from Blender's deform rig.
            parent_ref = reference.get(record["parent"])
            if parent_ref and parent_ref["head"].length > 1e-6:
                raise ValueError(f"Nonzero omitted reference parent: {name}")
            reference_local = record["head"]
            edited_local = bone.head_local
        offset = bone.matrix_local.to_3x3().inverted() @ (
            reference_local - edited_local
        )
        if offset.length > 1e-6:
            if not bone.use_local_location:
                raise ValueError(f"Enable Local Location for {name} first")
            offsets[bone.name] = offset
    return offsets


def _copy_driver(target, source, path, index, offset=0.0):
    curve = target.driver_add(path, index)
    driver = curve.driver
    driver.type = "SCRIPTED"
    variable = driver.variables.new()
    variable.name = "source_value"
    variable.type = "SINGLE_PROP"
    variable.targets[0].id = source
    variable.targets[0].data_path = path + f"[{index}]"
    driver.expression = f"source_value + ({float(offset)!r})"


def create_reference_preview(
    context, source, reference, reference_path="", prefixes=()
):
    if source.get(PREVIEW_PROP):
        raise ValueError("Select the source armature, not another preview")
    if source.constraints or any(b.constraints for b in source.pose.bones):
        raise ValueError("Reference preview requires a source without pose constraints")
    offsets = reference_offsets(source, reference, prefixes)
    if not offsets:
        raise ValueError("Reference and edited bone positions already agree")
    meshes = [
        obj
        for obj in context.scene.objects
        if obj.type == "MESH"
        and any(m.type == "ARMATURE" and m.object == source for m in obj.modifiers)
    ]
    if not meshes:
        raise ValueError("No meshes are deformed by the source armature")
    if any(obj.constraints for obj in meshes):
        raise ValueError("Reference preview does not support constrained mesh objects")
    if any(len([m for m in o.modifiers if m.type == "ARMATURE"]) != 1 for o in meshes):
        raise ValueError("Each source mesh must have one Armature modifier")

    collection = bpy.data.collections.new("PSO2 Reference Skinning Preview")
    context.scene.collection.children.link(collection)
    rig = source.copy()
    rig.data = source.data.copy()
    rig.name = source.name + " [Game Reference]"
    rig.animation_data_clear()
    rig.parent = None
    rig.matrix_world = source.matrix_world.copy()
    rig[PREVIEW_PROP] = "REFERENCE"
    rig[SOURCE_PROP] = source.name
    rig[REFERENCE_PROP] = reference_path
    rig["pso2_reference_offsets"] = {k: list(v) for k, v in offsets.items()}
    collection.objects.link(rig)
    # Only the preview data is disconnected. Blender ignores location
    # channels on connected bones, including valid reference corrections.
    for selected in list(context.selected_objects):
        selected.select_set(False)
    rig.hide_viewport = False
    rig.hide_set(False)
    context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for name in offsets:
            rig.data.edit_bones[name].use_connect = False
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    follow = rig.constraints.new("COPY_TRANSFORMS")
    follow.name = "Follow source object"
    follow.target = source
    for bone in rig.pose.bones:
        original = source.pose.bones[bone.name]
        bone.rotation_mode = original.rotation_mode
        for prop, count in (("location", 3), ("scale", 3)):
            path = bone.path_from_id(prop)
            for index in range(count):
                offset = (
                    offsets.get(bone.name, Vector())[index] if prop == "location" else 0
                )
                _copy_driver(rig, source, path, index, offset)
        prop, count = (
            ("rotation_quaternion", 4)
            if bone.rotation_mode == "QUATERNION"
            else ("rotation_axis_angle", 4)
            if bone.rotation_mode == "AXIS_ANGLE"
            else ("rotation_euler", 3)
        )
        for index in range(count):
            _copy_driver(rig, source, bone.path_from_id(prop), index)

    duplicates = []
    for obj in meshes:
        clone = obj.copy()
        clone.data = obj.data.copy()
        clone.name = obj.name + " [Game Reference]"
        clone.animation_data_clear()
        clone.parent = None
        clone.matrix_world = obj.matrix_world.copy()
        collection.objects.link(clone)
        clone[PREVIEW_PROP] = "REFERENCE"
        clone[SOURCE_PROP] = obj.name
        follow = clone.constraints.new("COPY_TRANSFORMS")
        follow.target = obj
        for modifier in clone.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == source:
                modifier.object = rig
                modifier.use_deform_preserve_volume = False
        duplicates.append(clone)
    context.view_layer.update()
    return rig, duplicates


def create_capture_preview(context, data):
    """Exact captured geometry with rest/capture/source comparison keys."""
    collection = bpy.data.collections.new("PSO2 Captured Skinning")
    context.scene.collection.children.link(collection)
    profile_text = bpy.data.texts.new("PSO2 Skinning Capture Profile")
    profile_text.write(json.dumps(data, ensure_ascii=False))
    objects = []
    source_armature = import_aqm._find_target_armature(context)
    for mesh_index, record in enumerate(data["meshes"]):
        positions = record["positions"]
        skinned = skinning_math.skin_positions(
            positions,
            record["weights_u16"],
            record["bone_indices"],
            data["palette"],
            data["bone_matrices"],
        )
        mesh = bpy.data.meshes.new(record["name"] + " [Capture]")
        mesh.from_pydata(positions, [], record["triangles"])
        mesh.update()
        obj = bpy.data.objects.new(mesh.name, mesh)
        collection.objects.link(obj)
        # Standard imported PSO2 model orientation: model Y is Blender Z.
        if source_armature and not source_armature.get(PREVIEW_PROP):
            obj.matrix_world = source_armature.matrix_world.copy()
        else:
            obj.rotation_euler.x = np.pi / 2
        obj[PREVIEW_PROP] = "CAPTURE"
        obj["pso2_capture_text"] = profile_text.name
        obj["pso2_capture_mesh_index"] = mesh_index
        obj["pso2_capture_event"] = record.get("event", 0)
        obj["pso2_capture_source"] = data.get("source", "")
        obj.shape_key_add(name="Rest input").value = 0
        key = obj.shape_key_add(name="Captured game")
        key.data.foreach_set("co", skinned.ravel())
        key.value = 1
        if "blender_positions" in record:
            source = np.asarray(record["blender_positions"], dtype=np.float32)
            key = obj.shape_key_add(name="Blender comparison")
            key.value = 0
            key.data.foreach_set("co", source.ravel())
            delta = np.linalg.norm(skinned - source, axis=1) * 1000
            attr = mesh.attributes.new("pso2_capture_delta_mm", "FLOAT", "POINT")
            attr.data.foreach_set("value", delta)
            obj["pso2_capture_max_delta_mm"] = float(delta.max())
        weights = skinning_math.gpu_weights(record["weights_u16"])
        groups = [obj.vertex_groups.new(name=name) for name in data["palette"]]
        for vertex, (ids, values) in enumerate(
            zip(record["bone_indices"], weights, strict=True)
        ):
            combined = {}
            for group, weight in zip(ids, values, strict=True):
                combined[group] = combined.get(group, 0.0) + float(weight)
            for group, weight in combined.items():
                if weight > 0:
                    groups[group].add([vertex], weight, "REPLACE")
        for channel in (1, 2):
            uv = record.get(f"uv{channel}")
            if uv:
                layer = mesh.uv_layers.new(name=f"UVChannel_{channel}")
                layer.data.foreach_set(
                    "uv", [x for loop in mesh.loops for x in uv[loop.vertex_index]]
                )
        material = bpy.data.materials.get(record.get("material_name", ""))
        if material:
            mesh.materials.append(material)
        objects.append(obj)
    context.view_layer.update()
    return objects


@classes.register
class PSO2_OT_ReferenceSkinning(bpy.types.Operator, ImportHelper):
    bl_idname = "pso2.preview_reference_skinning"
    bl_label = "Create Reference Skinning Preview"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".aqn"
    filter_glob: bpy.props.StringProperty(default="*.aqn", options={"HIDDEN"})

    bone_filter: bpy.props.StringProperty(
        name="Bone prefixes",
        description="Comma-separated bone prefixes; empty includes all matched bones",
        default="hemroot,hem_,drs2_tube_skirt_", translation_context="PSO2Tools",
    )

    @classmethod
    def poll(cls, context):
        return (
            context.mode == "OBJECT"
            and import_aqm._find_target_armature(context) is not None
        )

    def execute(self, context):
        source = import_aqm._find_target_armature(context)
        try:
            reference = read_reference_aqn(self.filepath)
            prefixes = tuple(
                x.strip() for x in self.bone_filter.split(",") if x.strip()
            )
            rig, meshes = create_reference_preview(
                context, source, reference, self.filepath, prefixes
            )
        except (ValueError, OSError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report(
            {"INFO"}, _report('Created {value0}, {value1} meshes; source preserved').format(value0=rig.name,value1=len(meshes))
        )
        return {"FINISHED"}


@classes.register
class PSO2_OT_CapturedSkinning(bpy.types.Operator, ImportHelper):
    bl_idname = "pso2.import_skinning_capture"
    bl_label = "Import Captured Skinning"
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        try:
            data = skinning_math.load_capture(self.filepath)
            objects = create_capture_preview(context, data)
        except (ValueError, OSError, KeyError, TypeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, _report('Created {value0} captured meshes').format(value0=len(objects)))
        return {"FINISHED"}


@classes.register
class PSO2_OT_SkinningComparison(bpy.types.Operator):
    bl_idname = "pso2.skinning_comparison"
    bl_label = "Show Skinning Pose"
    bl_options = {"UNDO"}
    pose: bpy.props.EnumProperty(
        items=[
            ("CAPTURE", "Captured game", ""),
            ("BLENDER", "Blender comparison", ""),
            ("REST", "Rest input", ""),
        ], translation_context="PSO2Tools"
    )

    def execute(self, context):
        objects = [
            o for o in context.selected_objects if o.get(PREVIEW_PROP) == "CAPTURE"
        ]
        if not objects:
            self.report({"ERROR"}, _report('Select captured preview meshes'))
            return {"CANCELLED"}
        for obj in objects:
            keys = obj.data.shape_keys.key_blocks
            keys["Captured game"].value = float(self.pose == "CAPTURE")
            if "Blender comparison" in keys:
                keys["Blender comparison"].value = float(self.pose == "BLENDER")
        return {"FINISHED"}


@classes.register
class PSO2_OT_InspectSkinningVertex(bpy.types.Operator):
    bl_idname = "pso2.inspect_skinning_vertex"
    bl_label = 'Inspect Captured Vertex'
    bl_options = {"REGISTER", "UNDO"}
    vertex: bpy.props.IntProperty(name="Vertex index", min=0, default=0, translation_context="PSO2Tools")

    @classmethod
    def poll(cls, context):
        return (
            context.object is not None and context.object.get(PREVIEW_PROP) == "CAPTURE"
        )

    def invoke(self, context, event):
        if context.mode == "EDIT_MESH":
            import bmesh

            mesh = bmesh.from_edit_mesh(context.object.data)
            active = mesh.select_history.active
            if isinstance(active, bmesh.types.BMVert):
                self.vertex = active.index
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        obj = context.object
        try:
            text = bpy.data.texts[obj["pso2_capture_text"]]
            data = json.loads(text.as_string())
            report = skinning_math.inspect_vertex(
                data, obj["pso2_capture_mesh_index"], self.vertex
            )
        except (ValueError, KeyError, IndexError, TypeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        output = bpy.data.texts.new(
            f"Skinning EID{report['event']} vertex{self.vertex}"
        )
        output.write(json.dumps(report, indent=2, ensure_ascii=False))
        obj["pso2_capture_last_report"] = output.name
        self.report({"INFO"}, _report('Text Editor: {value0}').format(value0=output.name))
        return {"FINISHED"}


@classes.register
class PSO2GameSkinningPanel(bpy.types.Panel):
    bl_idname = "VIEW3D_PT_pso2_game_skinning"
    bl_label = 'Game Skinning'
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "PSO2"

    def draw(self, context):
        layout = self.layout
        rig = import_aqm._find_target_armature(context)
        if rig and rig.get("pso2_raw_skin_source"):
            layout.label(text=_iface('Skinning mode: {value0}').format(value0=rig.get('pso2_skin_mode', 'BLENDER')))
            row = layout.row(align=True)
            for mode, title in (("BLENDER", "Blender"), ("GPU_WEIGHTS", 'Source Weights')):
                row.operator("pso2.skin_mode", text=title, text_ctxt="PSO2Tools").mode = mode
            layout.operator("pso2.bind_native_skinning", text='Bind Source Bone Inputs', text_ctxt="PSO2Tools")
            if rig.get("pso2_skin_native_inputs"):
                layout.operator("pso2.skin_mode", text='Native Bone Evaluation', text_ctxt="PSO2Tools").mode = "NATIVE_INPUTS"
                layout.operator("pso2.skin_source", text='Load Body FNP / Motion AQM', text_ctxt="PSO2Tools")
                layout.prop(rig, '["pso2_skin_time_scale"]', text='Motion Time Scale', text_ctxt="PSO2Tools")
            if rig.get("pso2_skin_error"):
                layout.label(text=rig["pso2_skin_error"], icon="ERROR")
            if rig.get('pso2_cloth_source_text'):
                row=layout.row(align=True)
                row.operator('pso2.cloth_auto',text='Enable FLTD Physics', text_ctxt="PSO2Tools").enabled=True
                row.operator('pso2.cloth_auto',text='Disable', text_ctxt="PSO2Tools").enabled=False
                if rig.get('pso2_cloth_auto'):
                    layout.prop(rig,'["pso2_cloth_settle_steps"]',text='Initial Settling Steps', text_ctxt="PSO2Tools")
                    layout.operator('pso2.cloth_reset',text='Reset Physics at Current Pose', text_ctxt="PSO2Tools")
                if rig.get('pso2_cloth_status'):
                    layout.label(text=_status(rig['pso2_cloth_status']), translate=False)
                layout.label(text='Uses scene time and gravity; live motion parity is unverified', text_ctxt="PSO2Tools")
            else:
                layout.label(text='Physics requires the matching model FLTD', text_ctxt="PSO2Tools")
        layout.operator("pso2.import_native_pose_inputs", text='Calculate from Native Bone Inputs', text_ctxt="PSO2Tools")
        if context.object and context.object.get("pso2_native_pose_input_text"):
            layout.operator("pso2.recalculate_native_pose", text='Recalculate Edited Inputs', text_ctxt="PSO2Tools")
            layout.operator("pso2.import_native_fnp", text='Load FNP for Validation', text_ctxt="PSO2Tools")
            layout.operator("pso2.import_native_aqm", text='Load AQM for Validation', text_ctxt="PSO2Tools")
            layout.label(text='Bone-input validation; general FNP reproduction is under investigation', text_ctxt="PSO2Tools")
        layout.operator(
            "pso2.build_bone_pipeline", text='Build Body, SA and Idle Pipeline', text_ctxt="PSO2Tools"
        )
        layout.operator(
            "pso2.import_skinning_capture", text='Import Captured Skinning Preview', text_ctxt="PSO2Tools"
        )
        layout.operator(
            "pso2.preview_reference_skinning", text='Compare with Reference AQN', text_ctxt="PSO2Tools"
        )
        obj = context.object
        if obj and obj.get(PREVIEW_PROP) == "CAPTURE":
            layout.operator(
                "pso2.inspect_skinning_vertex", text='Inspect Vertex Weights', text_ctxt="PSO2Tools"
            )
            row = layout.row(align=True)
            for value, title in (
                ("CAPTURE", 'Game'),
                ("BLENDER", "Blender"),
                ("REST", 'Rest Vertices'),
            ):
                row.operator("pso2.skinning_comparison", text=title, text_ctxt="PSO2Tools").pose = value
            layout.label(
                text=_iface('EID {value0} / fixed captured pose').format(value0=obj.get('pso2_capture_event', 0))
            )
            layout.label(
                text=_iface('Maximum position difference {value0:.2f} mm').format(value0=obj.get('pso2_capture_max_delta_mm', 0))
            )
        rig = import_aqm._find_target_armature(context)
        if rig and rig.get("pso2_bone_pipeline_trace"):
            layout.operator("pso2.inspect_bone_pipeline", text='Inspect Bone Calculation Stages', text_ctxt="PSO2Tools")
            row = layout.row(align=True)
            for stage, title in (
                ("BIND", "AQN"),
                ("NEUTRAL", 'Neutral'),
                ("BODY", 'Body'),
                ("SA", "SA"),
                ("IDLE", 'Idle'),
            ):
                row.operator("pso2.show_bone_pipeline_stage", text=title, text_ctxt="PSO2Tools").stage = stage
            layout.label(text='No physics in this pipeline; compare residuals with a capture', text_ctxt="PSO2Tools")
        if obj and obj.get(PREVIEW_PROP) == "REFERENCE":
            layout.label(text='Reference-offset preview; physics simulation is not included', text_ctxt="PSO2Tools")
