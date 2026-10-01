"""Local CNN inference wrapper (no external APIs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.preprocessing.image_processor import normalize
from app.recognition.cnn_model import (
    CLASS_NAMES,
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    load_class_names,
)


class SymbolPredictor:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH,
                 labels_path: Path | str = DEFAULT_LABELS_PATH):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self._model = None
        self.class_names: list[str] = load_class_names(labels_path) \
            if Path(labels_path).exists() else list(CLASS_NAMES)

    def _load(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self.model_path}. "
                    "Train it first: python -m training.train"
                )
            from tensorflow import keras
            self._model = keras.models.load_model(str(self.model_path))
            if self.labels_path.exists():
                self.class_names = load_class_names(self.labels_path)
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def predict_one(self, img28: np.ndarray) -> tuple[str, float, np.ndarray]:
        """img28: uint8 (28,28) or float (28,28,1). -> (label, conf, probs)."""
        model = self._load()
        if img28.dtype != np.float32 or img28.max() > 1.5:
            if img28.max() > 1.5:
                x = normalize(img28 if img28.ndim == 2 else img28[:, :, 0])
            else:
                x = img28.astype(np.float32)
                if x.ndim == 2:
                    x = x[..., None]
        else:
            x = img28 if img28.ndim == 3 else img28[..., None]
        probs = np.asarray(model.predict(x[None], verbose=0)[0], dtype=float)
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx]), probs

    def predict_batch(self, images: list[np.ndarray] | np.ndarray):
        import numpy as _np  # local alias to keep method self-contained
        arrs = []
        for im in images:
            im = _np.asarray(im)
            if im.max() > 1.5:
                arrs.append(normalize(im if im.ndim == 2 else im[:, :, 0]))
            else:
                a = im.astype(_np.float32)
                arrs.append(a if a.ndim == 3 else a[..., None])
        model = self._load()
        probs = _np.asarray(model.predict(_np.stack(arrs), verbose=0))
        out = []
        for p in probs:
            i = int(_np.argmax(p))
            out.append((self.class_names[i], float(p[i]), _np.array(p, dtype=float)))
        return out
