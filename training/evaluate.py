"""Evaluate a trained model: accuracy, report, confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))
from app.recognition.cnn_model import load_class_names  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "mathvision.npz"))
    ap.add_argument("--model", default=str(ROOT / "models" / "mathvision_cnn.keras"))
    ap.add_argument("--outdir", default=str(ROOT / "models"))
    args = ap.parse_args()

    from tensorflow import keras

    data = np.load(args.data)
    Xte, yte = data["X_test"], data["y_test"]
    y_true = np.argmax(yte, axis=1)

    model = keras.models.load_model(args.model)
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"[eval] loss={loss:.4f} acc={acc:.4f}")

    y_pred = np.argmax(model.predict(Xte, verbose=0), axis=1)
    names = load_class_names()
    n_cls = len(names)

    # --- metrics without sklearn (sklearn/scipy may be blocked) ---
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    print(f"{'class':>8} {'prec':>7} {'rec':>7} {'f1':>7} {'support':>8}")
    precs, recs, f1s = [], [], []
    for i, name in enumerate(names):
        tp = cm[i, i]
        prec = tp / max(1, cm[:, i].sum())
        rec = tp / max(1, cm[i, :].sum())
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"{name:>8} {prec:7.4f} {rec:7.4f} {f1:7.4f} {cm[i, :].sum():8d}")
    print(f"macro-avg prec={np.mean(precs):.4f} rec={np.mean(recs):.4f} f1={np.mean(f1s):.4f}")
    # weakest classes
    worst = sorted(zip(names, recs), key=lambda t: t[1])[:3]
    print(f"[eval] weakest classes: {worst}")

    metrics = {
        "test_accuracy": float(acc), "test_loss": float(loss),
        "macro_precision": float(np.mean(precs)),
        "macro_recall": float(np.mean(recs)), "macro_f1": float(np.mean(f1s)),
        "per_class": {n: {"precision": float(p), "recall": float(r),
                          "f1": float(f), "support": int(cm[i].sum())}
                      for i, (n, p, r, f) in enumerate(zip(names, precs, recs, f1s))},
        "data": args.data,
    }
    with open(Path(args.outdir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] metrics -> {Path(args.outdir) / 'metrics.json'}")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n_cls), names)
    ax.set_yticks(range(n_cls), names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Confusion matrix")
    fig.tight_layout()
    out = Path(args.outdir) / "confusion_matrix.png"
    fig.savefig(str(out), dpi=120)
    print(f"[eval] confusion matrix -> {out}")


if __name__ == "__main__":
    main()
