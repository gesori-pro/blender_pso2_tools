"""Flatten shape keys on temporary meshes for model export."""

from collections.abc import Iterable
from contextlib import contextmanager

import bpy


@contextmanager
def applied(context: bpy.types.Context, objects: Iterable[bpy.types.Object]):
    """Use the current shape-key mix without changing the editable mesh.

    AQP stores vertices, not Blender shape keys. Leave modifiers on the
    object so FBX can still apply them after the mix, and keep object
    identity intact for parenting, constraints and export selection.
    Blender evaluates the mix itself, including relative keys and masks.
    """
    objects = [obj for obj in objects if obj.data.shape_keys is not None]
    for obj in objects:
        if not obj.is_editable:
            raise RuntimeError(
                f"Make '{obj.name}' local before exporting its shape keys"
            )

    # Blender ignores mesh-data assignment in Edit Mode: clearing keys
    # there would operate on the original mesh instead of our copy.
    editing = (
        context.edit_object if any(obj.mode == "EDIT" for obj in objects) else None
    )
    swapped = []
    try:
        if editing is not None:
            bpy.ops.object.mode_set(mode="OBJECT")
        for obj in objects:
            original = obj.data

            # Mesh copies get Blender's numeric suffix, which Aqua's mesh
            # name parser strips before reading the PSO2 mesh flags.
            temporary = original.copy()
            index = obj.active_shape_key_index
            keys = temporary.shape_keys
            swapped.append((obj, original, temporary, keys, index))
            obj.data = temporary
            if obj.data != temporary:
                raise RuntimeError(f"Could not prepare shape keys on '{obj.name}'")

            mixed = obj.shape_key_add(name="Export Mix", from_mix=True)
            coordinates = [0.0] * (len(mixed.data) * 3)
            mixed.data.foreach_get("co", coordinates)
            obj.shape_key_clear()
            temporary.vertices.foreach_set("co", coordinates)
            temporary.update()

        context.view_layer.update()
        yield
    finally:
        for obj, original, temporary, keys, index in reversed(swapped):
            obj.data = original
            obj.active_shape_key_index = index
            # Clearing keys unlinks their datablock but leaves it orphaned.
            bpy.data.batch_remove(ids=(temporary, keys))
        context.view_layer.update()
        if editing is not None:
            context.view_layer.objects.active = editing
            bpy.ops.object.mode_set(mode="EDIT")
