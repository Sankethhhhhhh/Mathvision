"""MathVision CNN architecture + vocabulary constants (local, no external APIs)."""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabulary (V1) — 16 classes. Order is the canonical label encoding.
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "\u00d7", "\u00f7", "=", "x",
]

CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

# Filesystem-safe aliases for per-class folders (avoid unicode issues).
SAFE_NAMES: dict[str, str] = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "+": "plus", "-": "minus", "\u00d7": "times",
    "\u00f7": "div", "=": "equals", "x": "x",
}
SAFE_TO_CLASS: dict[str, str] = {v: k for k, v in SAFE_NAMES.items()}

IMG_SIZE = 28
NUM_CLASSES = len(CLASS_NAMES)
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mathvision_cnn.keras"
DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[2] / "models" / "class_names.json"


def build_mathvision_cnn(input_shape: tuple = INPUT_SHAPE,
                         num_classes: int = NUM_CLASSES):
    """Custom CNN for 28x28 handwritten math-symbol classification.

    Arch: Conv32 -> MaxPool -> Conv64 -> MaxPool -> Conv128 -> MaxPool
          -> Flatten -> Dense128 -> Dropout(0.35) -> Softmax
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # 28 -> 14 -> 7 -> 3  => 3x3x128
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mathvision_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_class_names(path: Path | str = DEFAULT_LABELS_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_class_names(path: Path | str = DEFAULT_LABELS_PATH) -> list[str]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return list(CLASS_NAMES)
