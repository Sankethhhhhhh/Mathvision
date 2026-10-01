"""Prepare the filtered MathVision symbol set from HASYv2 (reproducible).

Reads data/raw/hasyv2/ (extracted from HASYv2.tar.bz2, archives untouched),
selects the 15 HASYv2 classes covering the MathVision V1 vocabulary
('=' has NO HASYv2 class and stays synthetic — see mapping notes), verifies
every image decodes, and writes:
  data/processed/class_mapping.json   — hasy_symbol_id -> MathVision class
  data/processed/hasyv2_manifest.csv  — path,symbol_id,class,user_id (raw 32x32)
  data/processed/hasyv2_mathvision.npz — X uint8 (N,32,32) B/W, y int ids
Raw files are never modified; no resizing here (28x28 happens in training).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hasyv2"
PROCESSED = ROOT / "data" / "processed"

# Exact HASYv2 symbol_ids (from symbols.csv — never guessed).
# '=' does not exist in HASYv2 and is intentionally absent.
HASY_TO_CLASS: dict[str, str] = {
    "70": "0", "71": "1", "72": "2", "73": "3", "74": "4",
    "75": "5", "76": "6", "77": "7", "78": "8", "79": "9",
    "196": "+", "195": "-", "513": "\u00d7", "526": "\u00f7",
    "113": "x",
}
HASY_LATEX = {
    "70": "0", "71": "1", "72": "2", "73": "3", "74": "4",
    "75": "5", "76": "6", "77": "7", "78": "8", "79": "9",
    "196": "+", "195": "-", "513": "\\times", "526": "\\div",
    "113": "x",
}


def load_labels() -> list[dict]:
    with open(RAW / "hasy-data-labels.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["symbol_id"] in HASY_TO_CLASS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-npz", action="store_true", default=True)
    args = ap.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    rows = load_labels()
    print(f"[hasyv2] selected rows: {len(rows)}")

    mapping = {
        sid: {"class": cls, "latex": HASY_LATEX[sid],
              "samples": sum(1 for r in rows if r["symbol_id"] == sid)}
        for sid, cls in HASY_TO_CLASS.items()
    }
    mapping["_notes"] = {
        "=": "NO HASYv2 class exists for '=' (verified: no latex contains '='); "
             "keep the synthetic double-bar generator for '=' in Phase 2.",
        "X_vs_x": "HASYv2 id 54 is uppercase 'X' — excluded (looks like times); "
                  "only lowercase 'x' (id 113) is used.",
    }
    (PROCESSED / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(PROCESSED / "hasyv2_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "symbol_id", "class", "user_id"])
        for r in rows:
            w.writerow([r["path"], r["symbol_id"],
                        HASY_TO_CLASS[r["symbol_id"]], r["user_id"]])

    bad: list[str] = []
    imgs: list[np.ndarray] = []
    ids: list[int] = []
    for r in rows:
        p = RAW / r["path"]
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None or im.size == 0:
            bad.append(r["path"])
            continue
        imgs.append(im)
        ids.append(int(r["symbol_id"]))
    if bad:
        print(f"[hasyv2] WARNING: {len(bad)} unreadable images (see validation)")
        (PROCESSED / "hasyv2_bad_files.txt").write_text("\n".join(bad), encoding="utf-8")
    X = np.stack(imgs) if imgs else np.zeros((0, 32, 32), np.uint8)
    print(f"[hasyv2] decoded {len(imgs)}/{len(rows)}, shape={X.shape}")
    if args.save_npz:
        np.savez_compressed(str(PROCESSED / "hasyv2_mathvision.npz"),
                            X=X, symbol_id=np.array(ids, np.int32))
        print(f"[hasyv2] saved {PROCESSED / 'hasyv2_mathvision.npz'} "
              f"({X.nbytes / 1e6:.1f} MB, raw 32x32, unresized)")


if __name__ == "__main__":
    main()
