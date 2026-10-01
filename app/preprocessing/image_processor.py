"""Grayscale / denoise / binarize / resize helpers (OpenCV + NumPy)."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(gray, 3)


def binarize(gray: np.ndarray, method: str = "otsu") -> np.ndarray:
    """Return binary uint8 image, white symbol (255) on black (0)."""
    if method == "adaptive":
        b = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 21, 10)
        return b
    # OTSU with inversion: ink (dark) -> white
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b


def _estimate_stroke_width(binary: np.ndarray) -> float:
    """Median stroke width via area/perimeter over large contours."""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    widths = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        per = cv2.arcLength(cnt, True)
        if per > 0:
            widths.append(2.0 * area / per)
    if not widths:
        return 3.0
    return float(np.median(widths))


def preprocess_for_segmentation(img: np.ndarray) -> np.ndarray:
    """BGR/Gray -> clean binary (white-on-black) ready for contour detection."""
    gray = to_grayscale(img)
    gray = denoise(gray)
    binary = binarize(gray, method="otsu")
    # light morphological open to remove specks
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # close small rendering gaps (thin '+' crossings split apart).
    # Kernel scales with image size: 1px dropouts on 1000px scans need a
    # wider bridge than pen strokes on 300px photos; inter-symbol gaps are
    # an order of magnitude larger in both cases, so they never fuse.
    k = int(min(9, max(3, round(min(binary.shape[:2]) / 150))))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # normalize hairline strokes (scans/renders) toward pen-like width
    sw = _estimate_stroke_width(binary)
    if sw < 2.5:
        iters = 2 if sw < 1.5 else 1
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=iters)
    return binary


def _center_of_mass_shift(img: np.ndarray) -> np.ndarray:
    """Shift so symbol centroid is at the canvas centre (MNIST-style)."""
    m = cv2.moments(img)
    if m["m00"] == 0:
        return img
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
    h, w = img.shape[:2]
    tx, ty = w / 2 - cx, h / 2 - cy
    m_mat = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, m_mat, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def resize_with_padding(crop: np.ndarray, size: int = 28) -> np.ndarray:
    """Aspect-preserving resize of a white-on-black crop into size x size.

    Keeps aspect ratio, pads to square, resizes, then centres by mass.
    Input: uint8 binary/gray crop. Output: uint8 size x size.
    """
    if crop.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = (size - 4) / max(h, w)  # 2px margin each side
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return _center_of_mass_shift(canvas)


def normalize(img28: np.ndarray) -> np.ndarray:
    """uint8 28x28 -> float32 28x28x1 in [0, 1]."""
    arr = img28.astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr
