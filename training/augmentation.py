"""Controlled augmentation that preserves math-symbol identity.

Allowed: small rotations, translations, scaling, stroke-thickness jitter,
minor elastic wobble, Gaussian noise. Forbidden: flips, large rotations,
shears that turn '=' into '11', etc.
"""

from __future__ import annotations

import cv2
import numpy as np


def _translate(img: np.ndarray, tx: float, ty: float) -> np.ndarray:
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def _scale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(1, int(h * factor)), max(1, int(w * factor))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros_like(img)
    y0 = max(0, (h - nh) // 2)
    x0 = max(0, (w - nw) // 2)
    y1, x1 = y0 + min(nh, h), x0 + min(nw, w)
    canvas[y0:y1, x0:x1] = resized[: y1 - y0, : x1 - x0]
    return canvas


def elastic_wobble(img: np.ndarray, magnitude: float = 1.2,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Tiny smooth displacement field (keeps symbol identity)."""
    rng = np.random.default_rng() if rng is None else rng
    h, w = img.shape[:2]
    dx = rng.normal(0, magnitude, (h, w)).astype(np.float32)
    dy = rng.normal(0, magnitude, (h, w)).astype(np.float32)
    dx = cv2.GaussianBlur(dx, (5, 5), 1.0)
    dy = cv2.GaussianBlur(dy, (5, 5), 1.0)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(img, xs + dx, ys + dy,
                     interpolation=cv2.INTER_LINEAR, borderValue=0)


def augment_image(img: np.ndarray,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply one random, identity-preserving augmentation pass.

    Expects uint8 28x28, black background (0) / white symbol (>0).
    Returns uint8 image of the same shape.
    """
    rng = np.random.default_rng() if rng is None else rng
    out = img.copy()

    # small rotation: +/- 8 degrees
    if rng.random() < 0.7:
        out = _rotate(out, float(rng.uniform(-8.0, 8.0)))
    # small translation: +/- 2 px
    if rng.random() < 0.7:
        out = _translate(out, float(rng.uniform(-2, 2)),
                         float(rng.uniform(-2, 2)))
    # scaling 0.9 - 1.1
    if rng.random() < 0.5:
        out = _scale(out, float(rng.uniform(0.9, 1.1)))
    # stroke thickness jitter (incl. hairline scans/renders: eroded variants
    # teach the CNN thin-stroke styles instead of only bold pen strokes)
    r = rng.random()
    if r < 0.2:
        k = int(rng.integers(2, 3))
        out = cv2.dilate(out, np.ones((k, k), np.uint8))
    elif r < 0.35:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    elif r < 0.45:
        # hairline: two gentle erosions (topology-preserving, unlike 3x3)
        out = cv2.erode(out, np.ones((2, 2), np.uint8), iterations=2)
    # minor elastic distortion
    if rng.random() < 0.3:
        out = elastic_wobble(out, magnitude=float(rng.uniform(0.5, 1.2)), rng=rng)
    # slight brightness / noise
    if rng.random() < 0.3:
        noise = rng.normal(0, 6, out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def augment_batch(images: np.ndarray, n_views: int = 1,
                  seed: int | None = None) -> np.ndarray:
    """Return originals + `n_views` augmented copies stacked on axis 0."""
    rng = np.random.default_rng(seed)
    views = [images]
    for _ in range(n_views):
        aug = np.stack([augment_image(im, rng) for im in images])
        views.append(aug)
    return np.concatenate(views, axis=0)
