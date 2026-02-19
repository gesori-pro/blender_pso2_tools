import ast
import re
from contextlib import contextmanager
from pathlib import Path

import io_scene_fbx.export_fbx_bin
import io_scene_fbx.import_fbx

from . import scene_props, util


@util.copy_signature(io_scene_fbx.import_fbx.load)
def load(*args, **kwargs):
    """
    Wrapper around io_scene_fbx.import_fbx.load() which renames bones to move
    "(id)" prefixes to custom attributes.
    """

    with _monkey_patch_FbxImportHelperNode():
        return io_scene_fbx.import_fbx.load(*args, **kwargs)


@util.copy_signature(io_scene_fbx.export_fbx_bin.save)
def save(*args, **kwargs):
    """
    Wrapper around io_scene_fbx.export_fbx_bin.save() which renames bones to
    move "(id)" custom attributes back to prefixes.
    """

    with _monkey_patch_export_fbx_bin():
        return io_scene_fbx.export_fbx_bin.save(*args, **kwargs)


BONE_PATTERN = re.compile(r"^\((\d+)\)(.+(?:#.+(?:#.+)?)?)$")
"Matches `(id)name#short1#short2`, the format used by Aqua library"

BONE_PATTERN_2 = re.compile(r"^(.+(?:#.+(?:#.+)?)?)\((\d+)\)$")
"Matches `name#short1#short2(id)`, the format used by older versions of this add-on"


def split_bone_name(name: str) -> tuple[str, int] | None:
    if m := BONE_PATTERN.match(name):
        return m.group(2), int(m.group(1))

    # For models imported with the previous renaming that moved the ID to the end
    if m := BONE_PATTERN_2.match(name):
        return m.group(1), int(m.group(2))

    return None


def join_bone_name(name: str, bone_id: int):
    return f"({bone_id}){name}"


@contextmanager
def _monkey_patch_FbxImportHelperNode():
    orig = io_scene_fbx.import_fbx.FbxImportHelperNode

    class FbxImportHelperNodePso2(orig):
        def __init__(self, fbx_elem, bl_data, fbx_transform_data, is_bone):
            super().__init__(fbx_elem, bl_data, fbx_transform_data, is_bone)

            # Split fbx_name into the name and bone ID
            # pylint: disable-next=access-member-before-definition
            if result := split_bone_name(self.fbx_name):
                self.fbx_name = result[0]
                self.pso2_bone_id = result[1]
            else:
                self.pso2_bone_id = None

        def build_skeleton(self, *args, **kwargs):
            bone = super().build_skeleton(*args, **kwargs)

            # Store the saved bone ID in a custom property
            if self.pso2_bone_id is not None:
                bone[scene_props.BONE_ID] = self.pso2_bone_id

            return bone

    try:
        io_scene_fbx.import_fbx.FbxImportHelperNode = FbxImportHelperNodePso2
        yield
    finally:
        io_scene_fbx.import_fbx.FbxImportHelperNode = orig


def _find_function(mod: ast.Module, name: str):
    return next(
        n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _compile_function(name: str, source: str):
    mod = ast.parse(source, filename="<export_fbx_bin patched>")
    return _find_function(mod, name)


def _get_patched_export_funcs():
    # There's no convenient function to replace in the export code.
    # fbx_data_armature_elements() is the smallest scope that has access to the
    # data needed, but it's long. Instead of putting a patched copy of the whole
    # function here, parse the source file and generate a patched function.

    module_path = Path(io_scene_fbx.export_fbx_bin.__spec__.origin)
    source = module_path.read_text(encoding="utf-8")

    # If a bone has a "pso2_bone_id" custom property, returns "(id)name"
    # Parameter may be io_scene_fbx.fbx_utils.ObjectWrapper or bpy.types.Bone
    get_bone_name = _compile_function(
        "pso2_get_bone_name",
        """\
def pso2_get_bone_name(bone):
    search = [bone]

    if bdata := getattr(bone, 'bdata', None):
        search.append(bdata)

    for obj in search:
        try:
            bone_id = obj["pso2_bone_id"]
            return f"({bone_id}){obj.name}"
        except (KeyError, TypeError):
            pass

    return bone.name
""",
    )

    class RewriteBoneName(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)

            # replace fbx_name_class(<var>.name.encode(), <cls>)
            # with    fbx_name_class(pso2_get_bone_name(<var>).encode(), <cls>)
            match node:
                case ast.Call(
                    func=ast.Name("fbx_name_class"),
                    args=[
                        ast.Call(
                            func=ast.Attribute(
                                ast.Attribute(value=ast.Name(id=name), attr="name"),
                                attr="encode",
                            ),
                            args=[],
                            keywords=[],
                        ),
                        cls,
                    ],
                    keywords=[],
                ):
                    return ast.Call(
                        func=ast.Name("fbx_name_class"),
                        args=[
                            ast.Call(
                                ast.Attribute(
                                    value=ast.Call(
                                        func=ast.Name("pso2_get_bone_name"),
                                        args=[ast.Name(id=name)],
                                        keywords=[],
                                    ),
                                    attr="encode",
                                ),
                                args=[],
                                keywords=[],
                            ),
                            cls,
                        ],
                        keywords=[],
                    )

                case _:
                    return node

    patched_1 = _compile_function("fbx_data_armature_elements", source)
    patched_1.body.insert(0, get_bone_name)
    patched_1 = ast.fix_missing_locations(RewriteBoneName().visit(patched_1))

    patched_2 = _compile_function("fbx_data_object_elements", source)
    patched_2.body.insert(0, get_bone_name)
    patched_2 = ast.fix_missing_locations(RewriteBoneName().visit(patched_2))

    ns = {}
    # pylint: disable-next=exec-used
    exec(ast.unparse(patched_1), io_scene_fbx.export_fbx_bin.__dict__, ns)
    # pylint: disable-next=exec-used
    exec(ast.unparse(patched_2), io_scene_fbx.export_fbx_bin.__dict__, ns)

    return ns


@contextmanager
def _monkey_patch_export_fbx_bin():
    orig_armature_elements = io_scene_fbx.export_fbx_bin.fbx_data_armature_elements
    orig_object_elements = io_scene_fbx.export_fbx_bin.fbx_data_object_elements
    new = _get_patched_export_funcs()

    try:
        io_scene_fbx.export_fbx_bin.fbx_data_armature_elements = new[
            "fbx_data_armature_elements"
        ]
        io_scene_fbx.export_fbx_bin.fbx_data_object_elements = new[
            "fbx_data_object_elements"
        ]
        yield
    finally:
        io_scene_fbx.export_fbx_bin.fbx_data_armature_elements = orig_armature_elements
        io_scene_fbx.export_fbx_bin.fbx_data_object_elements = orig_object_elements
