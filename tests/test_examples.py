"""End-to-end tests on the shipped demo images (Phase 14).

Inputs only — expected answers are test assertions, never used by the app.
"""
from pathlib import Path

import cv2

from app.main import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "example_2xp5e15.png": ("2x+5=15", "x = 5"),
    "example_3xm7e11.png": ("3x-7=11", "x = 6"),
    "example_7xe35.png": ("7x=35", "x = 5"),
    "example_4xp2e18.png": ("4x+2=18", "x = 4"),
    "example_12p8e20.png": ("12+8=20", None),  # arithmetic: assert correctness flag
    "example_20d4e5.png": ("20÷4=5", None),  # arithmetic: assert correctness flag
}


def test_demo_examples_end_to_end():
    for fname, (expr, solution) in CASES.items():
        img = cv2.imread(str(ROOT / "data" / "test" / fname), cv2.IMREAD_COLOR)
        assert img is not None, fname
        binary, crops, details, pred, result = run_pipeline(img)
        assert pred == expr, f"{fname}: {pred!r} != {expr!r}"
        assert result is not None and result.kind != "error", fname
        if solution is not None:
            assert result.message == solution, f"{fname}: {result.message!r}"
        else:
            assert result.data.get("correct") is True, fname
        assert all(c >= 0.6 for _, c in details), f"{fname}: low confidence {details}"
