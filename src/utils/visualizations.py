from typing import Union
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
import torch


def add_text_to_image(
    img_in, text, origin_idx=(5, 5), anchor="lt", fill="red", debug=False
):
    """Add text to the the input image according to the specified arguments"""
    if isinstance(img_in, np.ndarray):
        if len(img_in.shape) < 3:
            img = np.tile(np.atleast_3d(img_in), (1, 1, 3))
        else:
            img = img_in
        img = Image.fromarray(img.astype(np.uint8), mode="RGB")
    elif not issubclass(img_in.__class__, Image.Image):
        raise Exception("Image should be either a numpy array or a PIL object")
    else:  # img is of type PIL.Image
        img = img_in
    draw = ImageDraw.Draw(img)
    draw.text(
        origin_idx, text, fill=fill, anchor=anchor, stroke_width=1, stroke_fill="black"
    )

    if debug:
        plt.figure()
        plt.imshow(img)
    return img


def full_dynamic_range(
    img: Union[np.ndarray, torch.tensor], vmin=0, vmax=255, dtype=np.uint8
):
    """Stretch input image to attain the full dynamic range indicated by vmin and vmax"""
    norm_img = (
        img - img.min()
    ) / img.ptp()  # all image values are in the closed interval [0, 1]
    dynamic_range = vmax - vmin
    full_range_img = norm_img * dynamic_range + vmin
    if dtype is not None:
        full_range_img = full_range_img.astype(dtype)
    return full_range_img


def get_next_matplotlib_color(format="hex"):
    def _hex2rgb(color_hex):
        color_hex_split = [color_hex[1:3], color_hex[3:5], color_hex[5:]]
        color_rgb = np.asarray([int(color, 16) for color in color_hex_split])
        return color_rgb

    cycle_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color in cycle_colors:
        if format == "rgb":
            yield _hex2rgb(color) / 255
        else:
            yield color
