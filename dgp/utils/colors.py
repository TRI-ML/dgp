# Copyright 2021 Toyota Research Institute.  All rights reserved.
"""This file contains definitions of predefined RGB colors, and color-related utility functions.
"""
import cv2
import matplotlib.cm
import numpy as np

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARKGRAY = (50, 50, 50)
YELLOW = (252, 226, 5)


def get_unique_colors(num_colors, in_bgr=False, cmap='tab20'):
    """Use matplotlib color maps ('tab10' or 'tab20') to get given number of unique colors.

    CAVEAT: matplotlib only supports default qualitative color palletes of fixed number, up to 20.

    Parameters
    ----------
    num_colors: int
        Number of colors. Must be less than 20.

    in_bgr: bool, optional
        Whether or not to return the color in BGR order. Default: False.

    cmap: str, optional
        matplotlib colormap name (https://matplotlib.org/tutorials/colors/colormaps.html)
        Must be a qualitative color map. Default: "tab20".

    Returns
    -------
    List[Tuple[int]]
        List of colors in 3-tuple.
    """
    colors = list(matplotlib.cm.get_cmap(cmap).colors)[:num_colors]

    if in_bgr:
        colors = [(clr[2], clr[1], clr[0]) for clr in colors]

    return [tuple([int(c * 255) for c in clr]) for clr in colors]


def color_borders(img, color, thickness=10):
    """Draw a frame (i.e. the 4 sides) of image.

    Parameters
    ----------
    img: np.ndarray
        Input image with shape of (H, W, 3) and unit8 type.

    color: Tuple[int]
        Color to draw the frame.

    thickness: int, optional
        Thickness of the frame. Default: 10.

    Returns
    -------
    img_w_frame: np.ndarray
        Image with colored frame.
    """
    assert img.shape[0] > 2 * thickness and img.shape[1] > 2 * thickness, "Image must be larger than border padding."

    img_w_frame = img.copy()
    img_w_frame[:thickness, :, :] = color
    img_w_frame[-thickness:, :, :] = color
    img_w_frame[:, :thickness, :] = color
    img_w_frame[:, -thickness:, :] = color

    return img_w_frame


def adjust_lightness(color, factor=1.0):
    """Adjust lightness of an RGB color.

    Parameters
    ----------
    color: Tuple[int]
        RGB color

    factor: float, optional
        Factor of lightness adjustment. Default: 1.0.

    Returns
    -------
    Tuple[int]
        Adjusted RGB color.
    """
    hls = cv2.cvtColor(np.uint8(color).reshape(1, 1, 3), cv2.COLOR_RGB2HLS).flatten()  # pylint: disable=too-many-function-args
    adjusted_hls = [hls[0], int(max(0, min(255, hls[1] * factor))), hls[2]]
    adjusted_rgb = cv2.cvtColor(np.uint8(adjusted_hls).reshape(1, 1, 3), cv2.COLOR_HLS2RGB).flatten()  # pylint: disable=too-many-function-args
    return (int(adjusted_rgb[0]), int(adjusted_rgb[1]), int(adjusted_rgb[2]))
