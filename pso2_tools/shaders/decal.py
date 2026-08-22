import bpy


def get_no_decal_image() -> bpy.types.Image:
    key = "PSO2 No Decal"

    image = bpy.data.images.get(key)
    if image:
        return image

    image = bpy.data.images.new(key, width=1, height=1, alpha=True)

    image.pixels[0:4] = (0.0, 0.0, 0.0, 0.0)  # type: ignore
    image.update()
    image.pack()

    return image
