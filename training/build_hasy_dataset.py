"""Build hybrid MathVision training set from real + synthetic sources.

Per-class source policy (documented in dataset_report.json):
  digits 0-9 : MNIST (1500/class; HASY has only ~60-130/class)
  + - x ÷ x  : HASYv2 real samples (hasyv2_mathvision.npz), upsampled with
               identity-preserving augmentation to target counts
  x          : HASY (66) + synthetic font-x supplement (scarcity documented)
  =          : synthetic double-bar (NO HASYv2 '=' class exists)

Anti-leakage: sources are split 70/15/15 FIRST, augmentation happens
independently inside each split (seeded). HASY 32x32 images go through the
exact inference normalization (invert -> tight crop -> resize_with_padding).

Output: data/processed/mathvision_hasy.npz (same keys as mathvision.npz).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

import sys
sys.path.insert(0, str(ROOT))
from app.preprocessing.image_processor import resize_with_padding  # noqa: E402
from app.recognition.cnn_model import CLASS_NAMES, CLASS_TO_INDEX  # noqa: E402
from training.augmentation import augment_image  # noqa: E402
from training.prepare_data import _draw_symbol, _load_mnist_digits  # noqa: E402

TRAIN_N, VAL_N, TEST_N = 1400, 300, 300
# HASY symbol_id per MathVision class (from class_mapping.json)
HASY_ID = {"+": 196, "-": 195, "\u00d7": 513, "\u00f7": 526, "x": 113}
# Synthetic supplements are TRAIN-ONLY (val/test stay purely real: MNIST/HASY).
# Rationale: HASY 'x' has 66 writers and print-style digits (flagged '1',
# barred '7') are absent from MNIST/HASY; supplements cover those styles.
DIGIT_SYNTH_TRAIN = 400
X_SYNTH_TRAIN = 940


def hasy_to_28(im32: np.ndarray) -> np.ndarray:
    """HASY 32x32 black-on-white -> pipeline-normalized 28x28 white-on-black."""
    inv = 255 - np.asarray(im32, dtype=np.uint8)
    ys, xs = np.where(inv > 20)
    if len(xs) == 0:
        return np.zeros((28, 28), np.uint8)
    x0, y0 = max(0, xs.min() - 2), max(0, ys.min() - 2)
    x1, y1 = min(32, xs.max() + 3), min(32, ys.max() + 3)
    return resize_with_padding(inv[y0:y1, x0:x1], 28)


def split_sources(n: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    ntr, nva = int(0.7 * n), int(0.15 * n)
    return idx[:ntr], idx[ntr:ntr + nva], idx[ntr + nva:]


def fill(base: list[np.ndarray], target: int, seed: int) -> list[np.ndarray]:
    """Top up to target by augmenting random picks (seeded)."""
    rng = np.random.default_rng(seed)
    out = list(base)
    while len(out) < target:
        src = base[int(rng.integers(0, len(base)))]
        out.append(augment_image(src, rng))
    return out[:target]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    hz = np.load(PROCESSED / "hasyv2_mathvision.npz")
    HX, Hids = hz["X"], hz["symbol_id"]
    per_class: dict[str, dict[str, list]] = {c: {"train": [], "val": [], "test": []}
                                             for c in CLASS_NAMES}

    # 1. digits from MNIST (+ synthetic font supplement in train only)
    mX, mY = _load_mnist_digits(1500, args.seed)
    assert mX is not None, "MNIST required for digits"
    for d in range(10):
        pool = mX[mY == d]
        itr, iva, ite = split_sources(len(pool), args.seed + d)
        for key, ii, n in (("train", itr, TRAIN_N), ("val", iva, VAL_N), ("test", ite, TEST_N)):
            base = [pool[i].astype(np.uint8) for i in ii]
            if key == "train":
                base += [_draw_symbol(str(d), rng) for _ in range(DIGIT_SYNTH_TRAIN)]
            per_class[str(d)][key] = fill(base, n, args.seed + 2000 + d)

    # 2. HASY symbols + synthetic '=' / x-supplement
    for cls in ["+", "-", "\u00d7", "\u00f7"]:
        for key, n, extra in (("train", TRAIN_N, 0), ("val", VAL_N, 0), ("test", TEST_N, 0)):
            pool = HX[Hids == HASY_ID[cls]]
            itr, iva, ite = split_sources(len(pool), args.seed + CLASS_TO_INDEX[cls])
            ii = {"train": itr, "val": iva, "test": ite}[key]
            base = [hasy_to_28(pool[i]) for i in ii]
            per_class[cls][key] = fill(base, n, args.seed + 1000 + CLASS_TO_INDEX[cls])

    # 2b. 'x' and '=' anchored on CROHME InkML symbols (file-grouped split:
    # symbols from one expression never leak across splits).
    cz = np.load(PROCESSED / "crohme_symbols.npz")
    CX = cz["X"]
    CY = np.array([str(v) for v in cz["y"]])
    CF = np.array([str(v) for v in cz["fid"]])

    def crohme_pool(cls, key, seed):
        fids = np.unique(CF[CY == cls])
        rr = np.random.default_rng(seed)
        perm = rr.permutation(len(fids))
        ntr, nva = int(0.7 * len(fids)), int(0.15 * len(fids))
        sel = {"train": set(fids[perm[:ntr]]),
               "val": set(fids[perm[ntr:ntr + nva]]),
               "test": set(fids[perm[ntr + nva:]])}[key]
        return [CX[i] for i in np.where(CY == cls)[0] if CF[i] in sel]

    def take(pool, n, seed):
        rr = np.random.default_rng(seed)
        idx = rr.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]

    # x: 900 CROHME + 300 HASY-upsampled + 200 font-x  |  val/test: CROHME only
    hx = HX[Hids == HASY_ID["x"]]
    per_class["x"]["train"] = (take(crohme_pool("x", "train", 11), 900, 21)
                               + fill([hasy_to_28(im) for im in hx], 300, 22)
                               + [_draw_symbol("x", rng) for _ in range(200)])
    per_class["x"]["val"] = fill(crohme_pool("x", "val", 11), VAL_N, 23)
    per_class["x"]["test"] = fill(crohme_pool("x", "test", 11), TEST_N, 24)
    # '=': 900 real CROHME + 500 synthetic  |  val/test: CROHME only
    per_class["="]["train"] = (take(crohme_pool("=", "train", 12), 900, 25)
                               + [_draw_symbol("=", rng) for _ in range(500)])
    per_class["="]["val"] = fill(crohme_pool("=", "val", 12), VAL_N, 26)
    per_class["="]["test"] = fill(crohme_pool("=", "test", 12), TEST_N, 27)
    if False:  # placeholder replaced below
        pass
    # 3. assemble stratified npz
    from tensorflow.keras.utils import to_categorical
    out = {}
    for key in ("train", "val", "test"):
        Xs, ys = [], []
        for cls in CLASS_NAMES:
            for im in per_class[cls][key]:
                Xs.append(im)
                ys.append(CLASS_TO_INDEX[cls])
        Xs = np.stack(Xs).astype(np.uint8)
        ys = np.array(ys, np.int64)
        perm = rng.permutation(len(Xs))
        Xs, ys = Xs[perm], ys[perm]
        Xn = (Xs.astype(np.float32) / 255.0)[..., None]
        out[f"X_{key}"], out[f"y_{key}"] = Xn, to_categorical(ys, len(CLASS_NAMES))
        out[f"y_{key}_int"] = ys
        print(f"[hasy-build] {key}: {Xn.shape} (per-class ~{len(Xs) // 16})")
    np.savez_compressed(str(PROCESSED / "mathvision_hasy.npz"), **out)
    (PROCESSED / "hasy_build_meta.json").write_text(json.dumps(
        {"sources": "MNIST digits; HASY +,-,times,div,x; synthetic = and x-supplement",
         "per_class_target": {"train": TRAIN_N, "val": VAL_N, "test": TEST_N},
         "seed": args.seed}, indent=2))
    print("[hasy-build] saved mathvision_hasy.npz")


if __name__ == "__main__":
    main()
