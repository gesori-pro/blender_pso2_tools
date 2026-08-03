import bpy

from .. import classes, shape_sliders


@classes.register
class PSO2ShapeAdjustPanel(bpy.types.Panel):
    """Interactive body shape sliders, editable like the in-game
    customization and exportable as a _sa.aqm (see shape_sliders.py)."""

    bl_idname = "OBJECT_PT_pso2_shape_adjust"
    bl_label = "Shape Adjust"
    bl_parent_id = "OBJECT_PT_pso2_appearance"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        assert self.layout is not None

        settings = shape_sliders.get_settings(context)
        if settings is None:
            return

        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        row = layout.row(align=True)
        row.operator(
            shape_sliders.PSO2_OT_ShapeSlidersFromAqm.bl_idname, text="Load AQM"
        )
        row.operator(
            shape_sliders.PSO2_OT_ExportShapeAdjust.bl_idname, text="Export AQM"
        )

        row = layout.row(align=True)
        row.operator(shape_sliders.PSO2_OT_ShapeSlidersReset.bl_idname, text="Reset")
        row.operator(
            shape_sliders.PSO2_OT_ShapeSlidersFreeze.bl_idname, text="Freeze Current"
        )

        for index, group in enumerate(shape_sliders.GROUPS):
            header, body = layout.panel(
                f"pso2_shape_{group.key}", default_closed=index > 0
            )
            header.label(text=f"{group.label}  ({', '.join(group.bones)})")
            if not body:
                continue

            body.prop(settings, f"{group.key}_scale")
            body.prop(settings, f"{group.key}_pos")
            if group.rotate:
                body.prop(settings, f"{group.key}_rot")
