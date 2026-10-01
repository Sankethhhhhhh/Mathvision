"""Solver + preprocessing + segmentation smoke tests (no trained model)."""
import cv2
import numpy as np

from app.preprocessing.image_processor import (
    normalize,
    preprocess_for_segmentation,
    resize_with_padding,
)
from app.segmentation.symbol_segmenter import segment_symbols
from app.solver.equation_solver import solve_expression


def test_solver_linear():
    r = solve_expression("2x+5=15")
    assert r.kind == "linear_solution"
    assert "5" in r.message


def test_solver_arithmetic():
    r = solve_expression("12+8=20")
    assert r.kind == "arithmetic_check"
    assert r.data["correct"] is True


def test_solver_division():
    r = solve_expression("20÷4=5")
    assert r.data["correct"] is True


def test_preprocess_and_segment_synthetic():
    canvas = np.ones((100, 300), dtype=np.uint8) * 255
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, "2+3", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    binary = preprocess_for_segmentation(canvas)
    assert binary.shape == canvas.shape[:2]
    crops = segment_symbols(binary)
    assert len(crops) >= 2
    for c in crops:
        assert c.image28.shape == (28, 28)
        x = normalize(c.image28)
        assert x.shape == (28, 28, 1)
        assert 0.0 <= x.max() <= 1.0


def test_resize_empty_crop():
    out = resize_with_padding(np.zeros((0, 0), dtype=np.uint8))
    assert out.shape == (28, 28)
