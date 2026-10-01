"""Extract rasterized per-symbol crops from CROHME InkML traceGroups (read-only).

Uses symbol-level segmentation in InkML: traceGroup truth label + traceView
refs -> trace coordinates. Rasterizes each requested symbol at 3x and
normalizes with the exact inference function (resize_with_padding).

Output: data/processed/crohme_symbols.npz (X uint8 (N,28,28), y labels,
  fid source file ids for grouped splitting) + printed per-class counts.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INKML_DIRS = [ROOT / "data" / "raw" / "crohme" / "TC11_CROHME23" / "INKML" / "train" / d
              for d in ("CROHME2019", "CROHME2023_train")]
OUT = ROOT / "data" / "processed" / "crohme_symbols.npz"

import sys
sys.path.insert(0, str(ROOT))
from app.preprocessing.image_processor import resize_with_padding  # noqa: E402

TRACE_RE = re.compile(r'<trace id="(\d+)">(.*?)</trace>', re.S)
GROUP_RE = re.compile(r'<traceGroup xml:id="\d+">\s*<annotation type="truth">(.*?)</annotation>(.*?)</traceGroup>', re.S)
VIEW_RE = re.compile(r'traceDataRef="(\d+)"')

# InkML truth -> MathVision class (only single-symbol V1 entries kept)
WANT = {str(d): str(d) for d in range(10)}
WANT.update({"+": "+", "-": "-", "=": "=", "x": "x"})


def rasterize(traces: list[np.ndarray]) -> np.ndarray:
    pts = np.vstack(traces)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    w, h = max(1e-6, x1 - x0), max(1e-6, y1 - y0)
    S = 84
    sc = 60 / max(w, h)
    cv = np.zeros((S, S), np.uint8)
    th = max(2, int(round(0.05 * max(w, h) * sc)))
    for t in traces:
        p = ((t - [x0, y0]) * sc + (S - np.array([w, h]) * sc) / 2).astype(int)
        cv = cv2.polylines(cv, [p.reshape(-1, 1, 2)], False, 255, th, cv2.LINE_AA)
    return resize_with_padding(cv, 28)


def parse_file(path: Path):
    try:
        txt = path.read_text(encoding="utf-8", errors="strict")
    except Exception:
        return None
    traces: dict[str, np.ndarray] = {}
    for tid, body in TRACE_RE.findall(txt):
        nums = [float(x) for x in re.split(r"[\s,]+", body.strip()) if x]
        nums = nums[: len(nums) // 2 * 2]  # drop trailing orphan from ', ' separators
        if len(nums) >= 4:
            traces[tid] = np.array(nums, np.float32).reshape(-1, 2)
    if not traces:
        return None
    out = []
    for truth, gbody in GROUP_RE.findall(txt):
        truth = truth.strip()
        if truth not in WANT or "<" in truth:
            continue
        tids = VIEW_RE.findall(gbody)
        ts = [traces[i] for i in tids if i in traces]
        if ts:
            out.append((WANT[truth], ts))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-per-class", type=int, default=1500)
    args = ap.parse_args()
    from collections import Counter
    Xs, ys, fids, counts = [], [], [], Counter()
    files = [p for d in INKML_DIRS for p in sorted(d.rglob("*.inkml"))]
    print(f"[crohme-sym] files: {len(files)}")
    for i, f in enumerate(files):
        items = parse_file(f)
        if not items:
            continue
        for lab, ts in items:
            if counts[lab] >= args.cap_per_class and lab != "x":
                continue
            try:
                Xs.append(rasterize(ts))
                ys.append(lab)
                fids.append(f.stem)
                counts[lab] += 1
            except Exception:
                pass
        if (i + 1) % 2000 == 0:
            print(f"  ...{i + 1}/{len(files)} {dict(counts)}")
    X = np.stack(Xs).astype(np.uint8)
    np.savez_compressed(str(OUT), X=X,
                        y=np.array(ys), fid=np.array(fids))
    print(f"[crohme-sym] saved {OUT} {X.shape} {dict(counts)}")


if __name__ == "__main__":
    main()
