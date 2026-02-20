from collections.abc import Sequence

import bpy

from .. import classes, fbx_wrapper, scene_props
from ..util import OperatorResult


@classes.register
class OBJECT_OT_pso2_rename_bones(bpy.types.Operator):
    """Move PSO2 bone IDs from names to custom properties"""

    bl_label = "PSO2 bone IDs to properties"
    bl_description = (
        "Move (id) prefixes to pso2_bone_id custom properties. "
        "This enables tools that use l_* and r_* prefixes on bones and vertex groups."
    )
    bl_idname = "object.pso2_rename_bones"
    bl_options = {"UNDO", "REGISTER"}

    @classmethod
    def poll(cls, context):
        return any(_get_bones_with_ids_in_names())

    def execute(self, context) -> OperatorResult:
        bones = list(_get_bones_with_ids_in_names())

        if dupes := list(_find_duplicate_bones(bones)):
            self.report(
                {"ERROR_INVALID_INPUT"},
                "Cannot rename bones. There are duplicates of the following bones:\n"
                + "\n".join(dupes),
            )
            return {"CANCELLED"}

        for bone in bones:
            if result := fbx_wrapper.split_bone_name(bone.name):
                bone.name = result[0]
                bone[scene_props.BONE_ID] = result[1]

        return {"FINISHED"}


@classes.register
class OBJECT_OT_pso2_restore_bones(bpy.types.Operator):
    """Move PSO2 bone IDs from custom properties back to names"""

    bl_label = "PSO2 bone IDs to names"
    bl_description = (
        "Move pso2_bone_id custom properties back to (id) prefixes on bone names. "
        "This can be used if you need to export to FBX with the bone name format "
        "expected by Aqua Model Tool."
    )
    bl_idname = "object.pso2_restore_bones"
    bl_options = {"UNDO", "REGISTER"}

    @classmethod
    def poll(cls, context):
        return any(_get_bones_with_id_props())

    def execute(self, context) -> OperatorResult:
        bones = list(_get_bones_with_id_props())

        if dupes := list(_find_duplicate_bones(bones)):
            self.report(
                {"ERROR_INVALID_INPUT"},
                "Cannot rename bones. There are duplicates of the following bones:\n"
                + "\n".join(dupes),
            )
            return {"CANCELLED"}

        for bone in bones:
            bone_id = bone[scene_props.BONE_ID]
            bone.name = fbx_wrapper.join_bone_name(bone.name, bone_id)
            # TODO: this doesn't actually seem to remove the property?
            del bone[scene_props.BONE_ID]

        return {"FINISHED"}


def menu_func(self: bpy.types.Operator, context: bpy.types.Context):
    assert self.layout is not None

    self.layout.operator(OBJECT_OT_pso2_rename_bones.bl_idname)
    self.layout.operator(OBJECT_OT_pso2_restore_bones.bl_idname)


def _get_bones_with_ids_in_names():
    for obj in bpy.data.objects:
        if not isinstance(obj.data, bpy.types.Armature):
            continue

        for bone in obj.data.bones:
            for pattern in (fbx_wrapper.BONE_PATTERN, fbx_wrapper.BONE_PATTERN_2):
                if pattern.match(bone.name):
                    yield bone


def _get_bones_with_id_props():
    for obj in bpy.data.objects:
        if not isinstance(obj.data, bpy.types.Armature):
            continue

        for bone in obj.data.bones:
            if scene_props.BONE_ID in bone:
                yield bone


def _find_duplicate_bones(bones: Sequence[bpy.types.Bone]):
    names = set[str]()
    for bone in bones:
        if bone.name in names:
            yield bone.name
        names.add(bone.name)
