"""Render V1 demo example images into data/test/ (reproducible, seed-fixed).

These are INPUTS ONLY. Answers are never stored — the app derives them live
from CNN -> reconstruction -> SymPy.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data" / "test"

EXAMPLES = ["2x+5=15", "3x-7=11", "7x=35", "12+8=20", "4x+2=18", "20/4=5"]


def draw_eq(eq: str, rng: np.random.Generator) -> np.ndarray:
    H, cell, gap = 140, 100, 34
    W = len(eq) * cell + (len(eq) - 1) * gap + 60
    img = np.ones((H, W, 3), dtype=np.uint8) * 255
    x = 30
    cy = H // 2 + int(rng.integers(-4, 5))
    for ch in eq:
        cx = x + cell // 2 + int(rng.integers(-3, 4))
        th = int(rng.integers(5, 8))
        L = int(rng.integers(24, 32))
        col = (0, 0, 0)
        if ch.isdigit() or ch == "x":
            font = cv2.FONT_HERSHEY_SIMPLEX if ch.isdigit() else cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
            sc = float(rng.uniform(2.6, 3.2))
            t = int(rng.integers(5, 8))
            (tw, th_) = cv2.getTextSize(ch, font, sc, t)[0]
            cv2.putText(img, ch, (cx - tw // 2, cy + th_ // 2), font, sc, col, t, cv2.LINE_AA)
        elif ch == "+":
            cv2.line(img, (cx - L, cy), (cx + L, cy), col, th, cv2.LINE_AA)
            cv2.line(img, (cx, cy - L), (cx, cy + L), col, th, cv2.LINE_AA)
        elif ch == "-":
            cv2.line(img, (cx - L, cy), (cx + L, cy), col, th + 2, cv2.LINE_AA)
        elif ch == "/":  # rendered as ÷ (dots + line)
            cv2.line(img, (cx - L, cy), (cx + L, cy), col, th + 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy - L - 8), th, col, -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy + L + 8), th, col, -1, cv2.LINE_AA)
        elif ch == "=":
            g = 15
            cv2.line(img, (cx - L, cy - g), (cx + L, cy - g), col, th + 2, cv2.LINE_AA)
            cv2.line(img, (cx - L, cy + g), (cx + L, cy + g), col, th + 2, cv2.LINE_AA)
        x += cell + gap
    return cv2.GaussianBlur(img, (3, 3), 0.5)


def main() -> None:
    TEST.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    for eq in EXAMPLES:
        p = TEST / f"example_{eq.replace('+', 'p').replace('-', 'm').replace('/', 'd').replace('=', 'e')}.png"
        cv2.imwrite(str(p), draw_eq(eq, rng))
        print(f"[examples] {p.name}")


if __name__ == "__main__":
    main()
