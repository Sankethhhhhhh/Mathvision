"""Contour-based symbol segmentation, left-to-right, with multi-part merge.

Handles '=' (two bars), '÷' (bar + two dots) and ':'-like stacks by merging
boxes whose x-intervals overlap strongly and whose y-gap is small.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.preprocessing.image_processor import (
    preprocess_for_segmentation,
    resize_with_padding,
)


@dataclass
class SymbolCrop:
    image28: np.ndarray      # uint8 28x28, white-on-black
    bbox: tuple[int, int, int, int]  # x, y, w, h in source coords
    box: tuple[int, int, int, int]   # same as bbox (alias for UI overlay)


def _bar_like(b):
    """Wide-short box (a bar of '=' or the bar of 'div')."""
    return b[2] >= 2.5 * b[3]


def _merge_multipart_boxes(boxes):
    """Merge vertically-stacked parts of multi-part symbols ('=', 'div').

    Scale-invariant rule: two boxes merge when their x-projections overlap
    (>=50% of the narrower width), they are strictly stacked vertically with
    a gap proportional to their own stroke heights (gap <= 3x the taller
    box height), and at least one box is bar-like. Absolute pixel sizes are
    never used, so '=' bars merge whether the equation is 60px or 600px
    tall, and two separate text rows (large gap) never fuse. Digit bodies
    are never bar-like, so side-by-side symbols cannot merge.
    """
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
    merged = []
    has_bar = []  # parallel flag: merged box already contains a bar part
    for b in boxes:
        placed = False
        for k, m in enumerate(merged):
            x0 = max(b[0], m[0])
            x1 = min(b[0] + b[2], m[0] + m[2])
            overlap = max(0, x1 - x0)
            narrow = min(b[2], m[2])
            x_overlap = overlap / max(1, narrow)
            # signed vertical gap: >0 stacked/disjoint, <0 y-ranges overlap
            gap = max(b[1], m[1]) - min(b[1] + b[3], m[1] + m[3])
            nx = min(b[0], m[0])
            ny = min(b[1], m[1])
            nx2 = max(b[0] + b[2], m[0] + m[2])
            ny2 = max(b[1] + b[3], m[1] + m[3])
            if (x_overlap >= 0.5 and 0 <= gap <= 3.0 * max(b[3], m[3])
                    and (has_bar[k] or _bar_like(b))
                    and (ny2 - ny) <= 1.5 * (nx2 - nx)):
                merged[k] = (nx, ny, nx2 - nx, ny2 - ny)
                has_bar[k] = True
                placed = True
                break
        if not placed:
            merged.append(b)
            has_bar.append(_bar_like(b))
    # re-run once to collapse chains (e.g. 3-part 'div')
    if len(merged) != len(boxes):
        return _merge_multipart_boxes(merged)
    return _absorb_fragments(merged)


def _absorb_fragments(boxes):
    """Absorb speck/hat/dot fragments into a large overlapping neighbor.

    A box is a fragment when its area is <12% of a neighbor it overlaps
    (x-overlap >= 30% of the fragment width, vertical gap <= half the
    fragment height). Two full-size symbols (e.g. '11') can never satisfy
    the area ratio, so they never fuse. Fixes dot-of-'i'-style splits and
    rendering specks without touching symbol-level grouping.
    """
    boxes = list(boxes)
    while True:
        areas = [b[2] * b[3] for b in boxes]
        absorbed = False
        for i, b in enumerate(boxes):
            best, best_a = -1, 0
            for j, m in enumerate(boxes):
                if i == j or areas[i] >= 0.12 * areas[j]:
                    continue
                x0 = max(b[0], m[0])
                x1 = min(b[0] + b[2], m[0] + m[2])
                if (x1 - x0) < 0.3 * b[2]:
                    continue
                gap = max(b[1], m[1]) - min(b[1] + b[3], m[1] + m[3])
                if gap > 0.5 * b[3]:
                    continue
                if areas[j] > best_a:
                    best, best_a = j, areas[j]
            if best >= 0:
                m = boxes[best]
                nx, ny = min(b[0], m[0]), min(b[1], m[1])
                nx2 = max(b[0] + b[2], m[0] + m[2])
                ny2 = max(b[1] + b[3], m[1] + m[3])
                boxes[best] = (nx, ny, nx2 - nx, ny2 - ny)
                del boxes[i]
                absorbed = True
                break
        if not absorbed:
            return boxes


def segment_symbols(image: np.ndarray,
                    min_area: int = 80,
                    min_height: int = 8) -> list[SymbolCrop]:
    """Segment white-on-black (or raw BGR/gray) image into symbol crops.

    Returns crops sorted left-to-right.
    """
    if image.ndim == 3 or (image.ndim == 2 and image.dtype == np.uint8
                           and image.mean() > 127):
        # Heuristic: raw input (dark ink on light bg mean is high) -> preprocess.
        # Already-binary white-on-black has mean < ~60 for sparse equations.
        binary = preprocess_for_segmentation(image) if image.ndim == 3 else image
        if binary.mean() > 127:
            binary = 255 - binary
    else:
        binary = image
    if binary.ndim == 3:
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < min_area:
            continue
        if h < min_height and w < min_height:
            continue
        boxes.append((x, y, w, h))
    boxes = _merge_multipart_boxes(boxes)
    boxes.sort(key=lambda b: b[0])  # left-to-right

    crops: list[SymbolCrop] = []
    for (x, y, w, h) in boxes:
        pad = 3
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = x + w + pad, y + h + pad
        crop = binary[y0:y1, x0:x1]
        img28 = resize_with_padding(crop, 28)
        crops.append(SymbolCrop(image28=img28, bbox=(x, y, w, h), box=(x, y, w, h)))
    return crops


def segment_file(path: str) -> list[SymbolCrop]:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return segment_symbols(img)
