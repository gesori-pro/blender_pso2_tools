from typing import ClassVar, TypeVar, cast

import bpy

_T = TypeVar("_T")


class ShaderNodeCustomGroup(bpy.types.ShaderNodeCustomGroup):
    # Set to True to not share node tree between instances
    has_attributes: ClassVar[bool] = False

    # Bump when a group's sockets or maths change. The shared tree is found by
    # name, so without this a file that already holds the older tree would
    # hand it to every new node - one missing the sockets the builders link.
    tree_version: ClassVar[int] = 0

    @property
    def group_name(self):
        name = self.bl_label
        if self.tree_version:
            name = f"{name} v{self.tree_version}"

        if self.has_attributes:
            return "." + name + "." + self.name

        return name

    def init(self, context):
        if not self.has_attributes and (
            tree := bpy.data.node_groups.get(self.group_name, None)
        ):
            self.node_tree = cast("bpy.types.ShaderNodeTree", tree)
        else:
            self.node_tree = cast(
                "bpy.types.ShaderNodeTree",
                bpy.data.node_groups.new(self.group_name, "ShaderNodeTree"),
            )
            self._build(self.node_tree)

    def free(self):
        if self.node_tree and self.node_tree.users == 1:
            bpy.data.node_groups.remove(self.node_tree, do_unlink=True)

    def _build(self, node_tree: bpy.types.ShaderNodeTree) -> None:
        raise NotImplementedError()

    def input(self, node_type: type[_T], name: str) -> _T:
        return cast("_T", self.inputs[name])

    def draw_buttons(
        self, context: bpy.types.Context, layout: bpy.types.UILayout | None
    ):
        pass
