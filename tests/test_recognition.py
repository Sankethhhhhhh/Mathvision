"""Recognition + segmentation scale-invariance tests (use trained model)."""
import cv2
import numpy as np

from app.preprocessing.image_processor import preprocess_for_segmentation
from app.recognition.cnn_model import CLASS_NAMES
from app.recognition.predictor import SymbolPredictor
from app.segmentation.symbol_segmenter import segment_symbols


def _draw_eq_bars(symbols, scale=3, seed=0):
    """Render a row of symbols large/smooth (dark on white), like pen strokes."""
    rng = np.random.default_rng(seed)
    H, cell, gap = 46 * scale, 34 * scale, 12 * scale
    W = len(symbols) * cell + (len(symbols) - 1) * gap + 20 * scale
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    x = 10 * scale
    cy = H // 2
    for sym in symbols:
        cx = x + cell // 2
        th = 2 * scale
        L = 9 * scale
        if sym == "=":
            g = 5 * scale
            cv2.line(img, (cx - L, cy - g), (cx + L, cy - g), (0, 0, 0), th + scale, cv2.LINE_AA)
            cv2.line(img, (cx - L, cy + g), (cx + L, cy + g), (0, 0, 0), th + scale, cv2.LINE_AA)
        elif sym == "div":
            cv2.line(img, (cx - L, cy), (cx + L, cy), (0, 0, 0), th, cv2.LINE_AA)
            cv2.circle(img, (cx, cy - 9 * scale), th, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy + 9 * scale), th, (0, 0, 0), -1, cv2.LINE_AA)
        x += cell + gap
    return img


def test_predictor_loads_and_maps_16_classes():
    p = SymbolPredictor()
    assert p.class_names == CLASS_NAMES
    assert len(p.class_names) == 16
    imgs = [np.zeros((28, 28), dtype=np.uint8) for _ in range(2)]
    out = p.predict_batch(imgs)
    assert len(out) == 2
    for lab, conf, probs in out:
        assert lab in CLASS_NAMES
        assert 0.0 <= conf <= 1.0
        assert probs.shape == (16,)
        assert abs(probs.sum() - 1.0) < 1e-5


def test_equals_merges_at_multiple_scales():
    for scale in (2, 3, 5):
        img = _draw_eq_bars(["="], scale=scale)
        crops = segment_symbols(preprocess_for_segmentation(img))
        assert len(crops) == 1, f"scale={scale}: got {len(crops)} crops"


def test_div_merges_at_multiple_scales():
    for scale in (2, 3, 5):
        img = _draw_eq_bars(["div"], scale=scale)
        crops = segment_symbols(preprocess_for_segmentation(img))
        assert len(crops) == 1, f"scale={scale}: got {len(crops)} crops"


def test_full_row_segment_counts():
    img = _draw_eq_bars(["=", "div", "="], scale=3)
    crops = segment_symbols(preprocess_for_segmentation(img))
    assert len(crops) == 3
    xs = [c.bbox[0] for c in crops]
    assert xs == sorted(xs), "crops must be left-to-right"
