"""Reusable Streamlit rendering helpers (pipeline visualization)."""
from __future__ import annotations

import numpy as np


def render_predictions(predictions: list[tuple[str, float]], confidences: list[float] | None = None):
    """Render per-symbol predictions inside Streamlit (lazy import)."""
    import streamlit as st

    confs = confidences if confidences is not None else [p[1] for p in predictions]
    labels = [p[0] if isinstance(p, tuple) else p for p in predictions]
    st.subheader("Symbol predictions")
    cols = st.columns(max(1, len(labels)))
    for col, lab, conf in zip(cols, labels, confs):
        with col:
            st.markdown(f"### `{lab}`")
            st.progress(min(1.0, max(0.0, float(conf))))
            st.caption(f"{float(conf):.1%}")


def render_solution(expression: str, message: str):
    """Render recognized equation + solver output."""
    import streamlit as st

    st.subheader("Recognized equation")
    st.latex(expression.replace("×", r"\times").replace("÷", r"\div"))
    st.code(expression)
    st.subheader("Solution")
    st.success(message)


def render_crops(crops: list, predictions: list | None = None):
    """Render segmented 28x28 crops with optional predicted labels."""
    import streamlit as st

    st.subheader(f"Segmented symbols ({len(crops)})")
    cols = st.columns(max(1, min(len(crops), 8)))
    for i, crop in enumerate(crops):
        img = crop.image28 if hasattr(crop, "image28") else np.asarray(crop)
        label = ""
        if predictions is not None and i < len(predictions):
            p = predictions[i]
            label = f"{p[0]} ({p[1]:.0%})" if isinstance(p, tuple) else str(p)
        with cols[i % len(cols)]:
            st.image(img, width=64, caption=label)
