import re
from contextlib import contextmanager

import io_scene_fbx.export_fbx_bin
import io_scene_fbx.import_fbx

from . import util


@util.copy_signature(io_scene_fbx.import_fbx.load)
def load(*args, **kwargs):
    """
    Wrapper around io_scene_fbx.import_fbx.load() which renames bones to move
    "(id)" prefixes to suffixes.
    """

    with _monkey_patch_FbxImportHelperNode():
        return io_scene_fbx.import_fbx.load(*args, **kwargs)


@util.copy_signature(io_scene_fbx.export_fbx_bin.save)
def save(*args, **kwargs):
    """
    Wrapper around io_scene_fbx.export_fbx_bin.save() which renames bones to
    move "(id)" suffixes back to prefixes.
    """

    with _monkey_patch_fbx_name_class():
        return io_scene_fbx.export_fbx_bin.save(*args, **kwargs)


IMPORT_BONE_PATTERN = re.compile(r"^\((\d+)\)(.+(?:#.+(?:#.+)?)?)$")
"Matches `(id)name#short1#short2`"

EXPORT_BONE_PATTERN = re.compile(r"^(.+(?:#.+(?:#.+)?)?)\((\d+)\)$")
"Matches `name#short1#short2(id)`"


def rename_bone_for_import(name: str):
    if m := IMPORT_BONE_PATTERN.match(name):
        return f"{m.group(2)}({m.group(1)})"
    return name


def rename_bone_for_export(name: str):
    if m := EXPORT_BONE_PATTERN.match(name):
        return f"({m.group(2)}){m.group(1)}"
    return name


@contextmanager
def _monkey_patch_FbxImportHelperNode():
    orig = io_scene_fbx.import_fbx.FbxImportHelperNode

    class FbxImportHelperNodePso2(orig):
        def __init__(self, fbx_elem, bl_data, fbx_transform_data, is_bone):
            super().__init__(fbx_elem, bl_data, fbx_transform_data, is_bone)

            self.fbx_name = rename_bone_for_import(self.fbx_name)

    try:
        io_scene_fbx.import_fbx.FbxImportHelperNode = FbxImportHelperNodePso2
        yield
    finally:
        io_scene_fbx.import_fbx.FbxImportHelperNode = orig


@contextmanager
def _monkey_patch_fbx_name_class():
    orig = io_scene_fbx.export_fbx_bin.fbx_name_class

    def fbx_name_class_pso2(name: bytes, cls: bytes):
        name = rename_bone_for_export(name.decode()).encode()

        return orig(name, cls)

    try:
        io_scene_fbx.export_fbx_bin.fbx_name_class = fbx_name_class_pso2
        yield
    finally:
        io_scene_fbx.export_fbx_bin.fbx_name_class = orig
