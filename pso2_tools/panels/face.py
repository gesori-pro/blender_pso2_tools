import bpy

from .. import classes, face_controls


@classes.register
class PSO2FacePanel(bpy.types.Panel):
    bl_idname = "OBJECT_PT_pso2_face_customization"
    bl_label = "PSO2 Face Customization"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    @classmethod
    def poll(cls, context):
        return face_controls.selected_face(context) is not None

    def draw(self, context):
        layout = self.layout
        armature = face_controls.selected_face(context)
        settings = getattr(armature, face_controls.SETTINGS)
        layout.label(text=armature.name, icon="ARMATURE_DATA")
        layout.operator("pso2.load_face_customization", icon="FILE_FOLDER")
        if not settings.loaded:
            layout.label(text="Uses this face model's own game-data motion.")
            return
        layout.label(text=settings.source_name)
        layout.prop(settings, "expression")
        layout.label(text=settings.status, icon="INFO")
        row = layout.row(align=True)
        row.operator("pso2.reset_face_customization")
        row.operator("pso2.clear_face_customization")
        header, body = layout.panel("pso2_face_limitations", default_closed=True)
        header.label(text="Scope / Limitations", icon="INFO")
        if body:
            body.label(
                text="Static selected face rig only; vertex weights stay unchanged."
            )
            body.label(text="Separate teeth/hair rigs are not rebound by this editor.")
            body.label(text="Pupil shape and lip-sync strength are not implemented.")
            body.label(text="Character-file export is not provided here.")
        source = armature.get(face_controls.SOURCE, {})
        available = armature.get(face_controls.AVAILABLE, {})
        for group in dict.fromkeys(item[2] for item in face_controls.CONTROLS):
            header, body = layout.panel(f"pso2_face_{group}", default_closed=True)
            header.label(text=group)
            if not body:
                continue
            for key, field, item_group, _, _ in face_controls.CONTROLS:
                if group != item_group:
                    continue
                row = body.row()
                row.enabled = face_controls._field(
                    field, settings.expression
                ) in source and available.get(field, True)
                row.prop(settings, key, slider=True)
                if face_controls._field(
                    field, settings.expression
                ) in source and not available.get(field, True):
                    body.label(text="No deformation in this face motion.", icon="INFO")
