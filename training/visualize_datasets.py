"""Generate dataset sample visualizations (reproducible).
  assets/dataset_samples/hasy_grid.png   — HASYv2 classes for MathVision V1
  assets/dataset_samples/crohme_v1_*.png — CROHME test expressions using only
                                           V1 vocabulary, with LaTeX truth
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HASY = ROOT / "data" / "raw" / "hasyv2"
CROHME = ROOT / "data" / "raw" / "crohme" / "TC11_CROHME23"
OUT = ROOT / "assets" / "dataset_samples"
V1 = set("0123456789+-=x") | {"times", "div", "/"}
LG_LABEL = {"times": "\u00d7", "div": "\u00f7"}


def hasy_grid() -> None:
    rows = list(csv.DictReader((HASY / "hasy-data-labels.csv").open(encoding="utf-8")))
    by_cls: dict[str, list] = {}
    for r in rows:
        by_cls.setdefault(r["symbol_id"], []).append(r)
    order = ["70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
             "196", "195", "513", "526", "113"]
    names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
             "+", "-", "\u00d7", "\u00f7", "x"]
    rng = random.Random(0)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, sid, nm in zip(axes.flat, order + [None], names + ["= (synthetic)"]):
        ax.set_title(f"HASY id {sid} → '{nm}'" if sid else "'=' → synthetic (no HASY class)",
                     fontsize=9)
        ax.axis("off")
        if sid is None:
            canvas = np.zeros((32, 128), np.uint8)
            cv2.line(canvas, (20, 10), (108, 10), 255, 3, cv2.LINE_AA)
            cv2.line(canvas, (20, 22), (108, 22), 255, 3, cv2.LINE_AA)
            ax.imshow(canvas, cmap="gray")
            continue
        picks = rng.sample(by_cls[sid], 4)
        strip = np.hstack([cv2.imread(str(HASY / p["path"]), cv2.IMREAD_GRAYSCALE)
                           for p in picks])
        ax.imshow(255 - strip, cmap="gray")
    fig.suptitle("HASYv2 samples for MathVision V1 (white-on-black shown)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "hasy_grid.png", dpi=120)
    print("[viz] hasy_grid.png")


def crohme_v1_expressions(k: int = 4) -> None:
    import re
    truth_re = re.compile(r'<annotation type="truth">\s*(.*?)\s*</annotation>', re.S)
    found = []
    for lg in sorted((CROHME / "SymLG" / "test" / "CROHME2019_test").glob("*.lg")):
        txt = lg.read_text(encoding="utf-8", errors="replace")
        labels = [ln.split(",")[2].strip() for ln in txt.splitlines()
                  if ln.startswith("O,")]
        rels = [ln.split(",")[3].strip() for ln in txt.splitlines()
                if ln.startswith("R,")]
        if (labels and set(labels) <= V1 | {"="} and "=" in labels
                and not set(rels) & {"Sup", "Sub", "Above", "Below", "Inside"}):
            found.append((lg.stem, labels))
        if len(found) >= k:
            break
    for stem, labels in found:
        img = cv2.imread(str(CROHME / "IMG" / "test" / "CROHME2019_test" / f"{stem}.png"))
        ik = CROHME / "INKML" / "test" / "CROHME2019_test" / f"{stem}.inkml"
        truth = truth_re.search(ik.read_text(encoding="utf-8", errors="replace")).group(1)
        truth = truth.replace("$", "")  # plain text: avoid mathtext parsing
        disp = " ".join(LG_LABEL.get(l, l) for l in labels)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"{stem}\ntruth: {truth}\nsymbols: {disp}", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"crohme_v1_{stem}.png", dpi=120)
        plt.close(fig)
        print(f"[viz] crohme_v1_{stem}.png truth={truth}")


if __name__ == "__main__":
    hasy_grid()
    crohme_v1_expressions()
