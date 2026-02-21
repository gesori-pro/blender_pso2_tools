from typing import Literal, TypeVar, cast

import bpy

from ..colors import ColorId
from . import types

GRID = 50

Vec2 = tuple[float, float]

NODE = TypeVar("NODE", bound=bpy.types.Node)
SOCKET = TypeVar("SOCKET", bound=bpy.types.NodeSocket)


def add_driver(
    target: bpy.types.bpy_struct,
    prop: str,
    id_type: Literal["MATERIAL"],
    source: bpy.types.ID,
    data_path: str,
    index=-1,
    expression=None,
):
    driver = target.driver_add(prop, index)

    assert isinstance(driver, bpy.types.FCurve)
    assert driver.driver is not None

    var = driver.driver.variables.new()
    var.name = data_path
    var.targets[0].id_type = id_type
    var.targets[0].id = source
    var.targets[0].data_path = data_path

    driver.driver.expression = expression or data_path

    return driver


class ShaderBuilder:
    material: bpy.types.Material
    data: types.ShaderData

    def __init__(self, material: bpy.types.Material, data: types.ShaderData):
        self.material = material
        self.data = data

    @property
    def tree(self):
        if not self.material.node_tree:
            raise RuntimeError("Material tree missing")

        return self.material.node_tree

    def init_tree(self):

        self.material.use_nodes = True
        self.tree.nodes.clear()

        return NodeTreeBuilder(self.tree)

    def build(self, context: bpy.types.Context):
        raise NotImplementedError()


class NodeTreeBuilder:
    tree: bpy.types.ShaderNodeTree

    def __init__(self, tree: bpy.types.ShaderNodeTree):
        self.tree = tree

    def new_input(
        self,
        socket_type: type[SOCKET],
        name: str,
        parent: bpy.types.NodeTreeInterfacePanel | None = None,
    ):
        assert self.tree.interface is not None

        return self.tree.interface.new_socket(
            name=name,
            in_out="INPUT",
            socket_type=socket_type.__name__,  # type: ignore (incorrect type stub)
            parent=parent,
        )

    def new_output(
        self,
        socket_type: type[SOCKET],
        name: str,
        parent: bpy.types.NodeTreeInterfacePanel | None = None,
    ):
        assert self.tree.interface is not None

        return self.tree.interface.new_socket(
            name=name,
            in_out="OUTPUT",
            socket_type=socket_type.__name__,  # type: ignore (incorrect type stub)
            parent=parent,
        )

    def add_link(
        self, in_socket: bpy.types.NodeSocket, out_socket: bpy.types.NodeSocket
    ) -> bpy.types.NodeLink:
        return self.tree.links.new(in_socket, out_socket)

    def add_color_link(
        self,
        channel: ColorId,
        colors: bpy.types.ShaderNodeCustomGroup,
        output: bpy.types.NodeSocket,
    ):
        if channel == ColorId.UNUSED:
            return None

        return self.tree.links.new(colors.outputs[channel.value - 1], output)

    def add_node(
        self,
        node_type: type[NODE],
        location: Vec2 | None = None,
        name: str | None = None,
    ) -> NODE:
        return cast("node_type", self._add_node(node_type.__name__, location, name))

    def _add_node(self, node_type: str, location: Vec2 | None, name: str | None):
        x, y = location or (0, 0)

        node = self.tree.nodes.new(node_type)
        node.location = (x * GRID, y * GRID)

        if name:
            node.name = name
            node.label = name

        return node

    def add_group(self, name: str, offset: Vec2 = (0, 0)):
        return NodeTreeGroupBuilder(self.tree, name, offset)


class NodeTreeGroupBuilder(NodeTreeBuilder):
    def __init__(
        self, tree: bpy.types.ShaderNodeTree, name: str, offset: Vec2 = (0, 0)
    ):
        super().__init__(tree)
        self.offset = offset

        self.frame = super()._add_node("NodeFrame", offset, name)

    def _add_node(self, node_type: str, location: Vec2 | None, name: str | None):
        x, y = location or (0, 0)

        x += self.offset[0]
        y += self.offset[1]

        node = super()._add_node(node_type, (x, y), name)
        node.parent = self.frame

        return node
