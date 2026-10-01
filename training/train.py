"""Train the MathVision CNN on data/processed/mathvision.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(ROOT))
from app.recognition.cnn_model import (  # noqa: E402
    build_mathvision_cnn, save_class_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "mathvision.npz"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "models" / "mathvision_cnn.keras"))
    args = ap.parse_args()

    import numpy as np
    from tensorflow import keras

    data = np.load(args.data)
    Xtr, ytr = data["X_train"], data["y_train"]
    Xva, yva = data["X_val"], data["y_val"]
    print(f"[train] train={Xtr.shape} val={Xva.shape}")

    model = build_mathvision_cnn()
    model.summary(print_fn=print)

    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(args.out, monitor="val_accuracy",
                                        save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=args.epochs, batch_size=args.batch_size,
                     callbacks=cbs, verbose=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    save_class_names()
    print(f"[train] saved model -> {args.out}")

    # history plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist.history["accuracy"], label="train")
    ax[0].plot(hist.history["val_accuracy"], label="val")
    ax[0].set_title("accuracy"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(hist.history["loss"], label="train")
    ax[1].plot(hist.history["val_loss"], label="val")
    ax[1].set_title("loss"); ax[1].legend(); ax[1].grid(True)
    fig.tight_layout()
    fig.savefig(str(Path(args.out).parent / "training_history.png"), dpi=120)
    print("[train] history plot saved.")


if __name__ == "__main__":
    main()
