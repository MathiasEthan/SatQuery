import os
import numpy as np


def load_image(path):
    from PIL import Image

    ext = path.lower().split(".")[-1]
    if ext in ["tif", "tiff"]:
        import rasterio

        with rasterio.open(path) as src:
            img = src.read()
            # rasterio reads as (C, H, W). convert to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            return img
    else:
        # PNG or JPEG
        img = Image.open(path)
        img_arr = np.array(img)
        # Ensure it has 3 dimensions (H, W, C) so transpose works as expected
        if len(img_arr.shape) == 2:
            img_arr = np.expand_dims(img_arr, axis=-1)
        elif img_arr.shape[2] == 4:
            # If RGBA, convert to RGB
            img_arr = np.array(img.convert("RGB"))
        return img_arr
