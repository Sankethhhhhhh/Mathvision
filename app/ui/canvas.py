"""Drawing-canvas wrapper: white canvas, dark stroke, pipeline-ready.

Returns a BGR uint8 image (white background, dark ink) or None when empty.
Uses the SAME downstream pipeline as uploaded images.
"""
from __future__ import annotations

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 860, 300


def draw_canvas(st, key: str = "mv_canvas"):
    """Render the drawable canvas. Returns (bgr_image | None, is_empty)."""
    from streamlit_drawable_canvas import st_canvas

    canvas = st_canvas(
        fill_color="rgba(255,255,255,1)",
        stroke_width=6,
        stroke_color="#111111",
        background_color="#ffffff",
        height=CANVAS_H,
        width=CANVAS_W,
        drawing_mode="freedraw",
        key=key,
        display_toolbar=True,
    )
    if canvas is None or canvas.image_data is None:
        return None, True
    rgba = canvas.image_data.astype(np.uint8)
    if rgba[..., :3].std() < 1e-6 and (rgba[..., 3] > 250).all():
        return None, True  # untouched blank canvas
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = (rgba[..., :3].astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    if (rgb > 250).all():
        return None, True
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, False
