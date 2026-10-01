"""Synthetic + MNIST-hybrid dataset builder (offline-capable, local only).

Strategy
--------
* Try to load MNIST digits via tf.keras (needs one-time download). If that
  fails (offline), fall back to fully synthetic digits.
* Always synthesize the 6 non-digit symbols (+, -, ×, ÷, =, x) with OpenCV
  vector drawing + the identity-preserving augmentation in
  training/augmentation.py so train/val/test stay consistent with canvas ink.

Output: data/processed/mathvision.npz with X_train/y_train/X_val/y_val/
X_test/y_test (float32, [0,1], (N,28,28,1), one-hot labels) + meta dict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
NPZ_PATH = PROCESSED / "mathvision.npz"
META_PATH = PROCESSED / "meta.json"

import sys
sys.path.insert(0, str(ROOT))
from app.recognition.cnn_model import CLASS_NAMES, CLASS_TO_INDEX  # noqa: E402
from training.augmentation import augment_image  # noqa: E402

SYMBOL_CLASSES = ["+", "-", "\u00d7", "\u00f7", "=", "x"]
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
         cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]


def _blank(size=84):
    return np.zeros((size, size), dtype=np.uint8)


# Render scale: symbols are drawn large (mimicking real pen strokes at camera
# resolution) and then normalized with the EXACT pipeline function used at
# inference (tight-crop aspect resize + centre-of-mass). This removes the
# train/inference scale-mismatch that made bold segmented symbols (e.g. 'x')
# look like a different class (e.g. the times-cross).
RENDER = 3
CANVAS = 28 * RENDER
MID = CANVAS // 2


def _draw_symbol(symbol, rng):
    """Draw one 28x28 white-on-black symbol, pipeline-normalized."""
    from app.preprocessing.image_processor import resize_with_padding
    img = _blank(CANVAS)
    C = MID
    cx, cy = C + int(rng.integers(-6, 7)), C + int(rng.integers(-6, 7))
    th = int(rng.integers(3, 10))  # stroke thickness 3-9 (== 1-3 at 28px)
    L = int(rng.integers(21, 33))  # half-length of bars (== 7-11 at 28px)

    if symbol == "+":
        cv2.line(img, (cx - L, cy), (cx + L, cy), 255, th, cv2.LINE_AA)
        cv2.line(img, (cx, cy - L), (cx, cy + L), 255, th, cv2.LINE_AA)
    elif symbol == "-":
        cv2.line(img, (cx - L, cy), (cx + L, cy), 255, th + 3, cv2.LINE_AA)
    elif symbol == "\u00d7":  # geometric times-cross: straight uniform strokes
        o = L
        cv2.line(img, (cx - o, cy - o), (cx + o, cy + o), 255, th, cv2.LINE_AA)
        cv2.line(img, (cx - o, cy + o), (cx + o, cy - o), 255, th, cv2.LINE_AA)
    elif symbol == "x":  # handwritten 'x': font glyph, curved uneven strokes
        xfonts = [cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                  cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
                  cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_DUPLEX]
        xfont = xfonts[int(rng.integers(0, len(xfonts)))]
        xscale = float(rng.uniform(2.1, 3.0))
        xt = int(rng.integers(3, 7))
        (xtw, xth) = cv2.getTextSize("x", xfont, xscale, xt)[0]
        xorg = (C - xtw // 2 + int(rng.integers(-6, 7)),
                C + xth // 2 + int(rng.integers(-6, 7)))
        cv2.putText(img, "x", xorg, xfont, xscale, 255, xt, cv2.LINE_AA)
    elif symbol == "=":
        gap = 12
        cv2.line(img, (cx - L, cy - gap), (cx + L, cy - gap), 255, th + 3, cv2.LINE_AA)
        cv2.line(img, (cx - L, cy + gap), (cx + L, cy + gap), 255, th + 3, cv2.LINE_AA)
    elif symbol == "\u00f7":
        cv2.line(img, (cx - L, cy), (cx + L, cy), 255, th, cv2.LINE_AA)
        cv2.circle(img, (cx, cy - 21), th, 255, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy + 21), th, 255, -1, cv2.LINE_AA)
    elif symbol.isdigit():
        font = FONTS[int(rng.integers(0, len(FONTS)))]
        scale = float(rng.uniform(2.1, 3.0))
        t = int(rng.integers(3, 7))
        (tw, th_) = cv2.getTextSize(symbol, font, scale, t)[0]
        org = (C - tw // 2 + int(rng.integers(-6, 7)),
               C + th_ // 2 + int(rng.integers(-6, 7)))
        cv2.putText(img, symbol, org, font, scale, 255, t, cv2.LINE_AA)
    # random small rotation for handwriting feel (identity-safe range)
    if rng.random() < 0.6:
        ang = float(rng.uniform(-7, 7))
        m = cv2.getRotationMatrix2D((C, C), ang, 1.0)
        img = cv2.warpAffine(img, m, (CANVAS, CANVAS),
                             flags=cv2.INTER_LINEAR, borderValue=0)
    # tight bounding-box crop (like the segmenter) then pipeline normalization
    ys, xs = np.where(img > 20)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    pad = 6
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(CANVAS, x1 + pad), min(CANVAS, y1 + pad)
    return resize_with_padding(img[y0:y1, x0:x1], 28)


def _load_mnist_digits(per_class: int, seed: int):
    """Return (images uint8 NCHW-less, labels int) or (None, None) offline."""
    try:
        from tensorflow.keras.datasets import mnist
        (xtr, ytr), (xte, yte) = mnist.load_data()
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        X = np.concatenate([xtr, xte])
        Y = np.concatenate([ytr, yte])
        for d in range(10):
            idx = np.where(Y == d)[0]
            sel = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
            xs.append(X[sel])
            ys.append(Y[sel])
        return np.concatenate(xs), np.concatenate(ys)
    except Exception as e:  # offline or no tf data cache
        print(f"[dataset] MNIST unavailable ({e}); using synthetic digits.")
        return None, None


def build_dataset(samples_per_class: int = 2500, seed: int = 42,
                  mnist_per_class: int = 1500):
    rng = np.random.default_rng(seed)
    images: list[np.ndarray] = []
    labels: list[int] = []

    mX, mY = _load_mnist_digits(mnist_per_class, seed)
    if mX is not None:
        for im, lb in zip(mX, mY):
            images.append(im.astype(np.uint8))
            labels.append(int(lb))

    n_synth_digits = samples_per_class if mX is None else samples_per_class // 3
    for d in map(str, range(10)):
        for _ in range(n_synth_digits):
            images.append(augment_image(_draw_symbol(d, rng), rng))
            labels.append(CLASS_TO_INDEX[d])
    for s in SYMBOL_CLASSES:
        for _ in range(samples_per_class):
            images.append(augment_image(_draw_symbol(s, rng), rng))
            labels.append(CLASS_TO_INDEX[s])

    X = np.stack(images).astype(np.uint8)
    y = np.array(labels, dtype=np.int64)
    # shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def split_and_save(X: np.ndarray, y: np.ndarray):
    from tensorflow.keras.utils import to_categorical
    n = len(X)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    Xte, yte = X[n_train + n_val:], y[n_train + n_val:]

    def prep(a: np.ndarray) -> np.ndarray:
        return (a.astype(np.float32) / 255.0)[..., None]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(NPZ_PATH),
        X_train=prep(Xtr), y_train=to_categorical(ytr, len(CLASS_NAMES)),
        X_val=prep(Xva), y_val=to_categorical(yva, len(CLASS_NAMES)),
        X_test=prep(Xte), y_test=to_categorical(yte, len(CLASS_NAMES)),
        y_train_int=ytr, y_val_int=yva, y_test_int=yte,
    )
    META_PATH.write_text(json.dumps(
        {"classes": CLASS_NAMES, "sizes": {"train": len(Xtr), "val": len(Xva),
                                          "test": len(Xte)}}, indent=2))
    print(f"[dataset] saved {NPZ_PATH} train={len(Xtr)} val={len(Xva)} test={len(Xte)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-class", type=int, default=2500)
    ap.add_argument("--mnist-per-class", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    X, y = build_dataset(args.samples_per_class, args.seed, args.mnist_per_class)
    split_and_save(X, y)


if __name__ == "__main__":
    main()
