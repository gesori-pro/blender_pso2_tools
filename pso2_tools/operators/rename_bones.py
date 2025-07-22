from typing import Sequence

import bpy

from .. import classes, fbx_wrapper
from ..util import OperatorResult


@classes.register
class OBJECT_OT_pso2_rename_bones(bpy.types.Operator):
    """Move PSO2 bone IDs to the ends of bone names"""

    bl_label = "Move PSO2 bone IDs"
    bl_description = (
        "Move (id) prefixes to the end of bone names. "
        "This enables tools that use l_* and r_* prefixes on bones and vertex groups."
    )
    bl_idname = "object.pso2_rename_bones"
    bl_options = {"UNDO", "REGISTER"}

    @classmethod
    def poll(cls, context):
        return any(_get_bones_with_id_prefixes())

    def execute(self, context) -> OperatorResult:
        bones = list(_get_bones_with_id_prefixes())

        if dupes := list(_find_duplicate_bones(bones)):
            self.report(
                {"ERROR_INVALID_INPUT"},
                "Cannot rename bones. There are duplicates of the following bones:\n"
                + "\n".join(dupes),
            )
            return {"CANCELLED"}

        for bone in bones:
            bone.name = fbx_wrapper.rename_bone_for_import(bone.name)

        return {"FINISHED"}


def menu_func(self: bpy.types.Operator, context: bpy.types.Context):
    assert self.layout is not None

    self.layout.operator(OBJECT_OT_pso2_rename_bones.bl_idname)


def _get_bones_with_id_prefixes():
    for obj in bpy.data.objects:
        if not isinstance(obj.data, bpy.types.Armature):
            continue

        for bone in obj.data.bones:
            if fbx_wrapper.IMPORT_BONE_PATTERN.match(bone.name):
                yield bone


def _find_duplicate_bones(bones: Sequence[bpy.types.Bone]):
    names = set[str]()
    for bone in bones:
        if bone.name in names:
            yield bone.name
        names.add(bone.name)
