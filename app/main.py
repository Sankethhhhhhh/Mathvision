"""MathVision — AI mathematical vision workstation.

Backend contract (PROTECTED, unchanged behavior):
  preprocessing.preprocess_for_segmentation
  segmentation.symbol_segmenter.segment_symbols
  recognition.predictor.SymbolPredictor.predict_batch
  solver.equation_solver.solve_expression
This file is presentation only: it calls run_pipeline()/solve_image()
and renders real outputs. Nothing is hardcoded.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mathvision_cnn.keras"
METRICS_PATH = ROOT / "models" / "metrics.json"
TEST_DIR = ROOT / "data" / "test"

EXAMPLES = [
    ("2x + 5 = 15", "example_2xp5e15.png"),
    ("3x − 7 = 11", "example_3xm7e11.png"),
    ("7x = 35", "example_7xe35.png"),
    ("4x + 2 = 18", "example_4xp2e18.png"),
    ("12 + 8 = 20", "example_12p8e20.png"),
    ("20 ÷ 4 = 5", "example_20d4e5.png"),
]


def _decode_upload(upload) -> np.ndarray:
    data = np.asarray(bytearray(upload.read()), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode uploaded image.")
    return img


def run_pipeline(image: np.ndarray, predictor=None):
    """Preprocess -> segment -> recognize -> reconstruct -> solve."""
    from app.preprocessing.image_processor import preprocess_for_segmentation
    from app.recognition.predictor import SymbolPredictor
    from app.segmentation.symbol_segmenter import segment_symbols
    from app.solver.equation_solver import solve_expression

    binary = preprocess_for_segmentation(image)
    crops = segment_symbols(binary)
    labels: list[str] = []
    details: list = []
    if crops:
        try:
            predictor = predictor or SymbolPredictor()
            results = predictor.predict_batch([c.image28 for c in crops])
            for lab, conf, _probs in results:
                labels.append(lab)
                details.append((lab, conf))
        except Exception:
            labels = ["?" for _ in crops]
            details = [(lab, 0.0) for lab in labels]
    expression = "".join(labels)
    result = solve_expression(expression) if labels and "?" not in labels else None
    return binary, crops, details, expression, result


def solve_image(image: np.ndarray, predictor=None) -> dict:
    t0 = time.perf_counter()
    binary, crops, details, expression, result = run_pipeline(image, predictor)
    dt = (time.perf_counter() - t0) * 1000
    conf = float(np.mean([d[1] for d in details])) if details else 0.0
    return {"image": image, "binary": binary, "crops": crops, "details": details,
            "expression": expression, "result": result, "confidence": conf,
            "ms": dt}


def _push_history(expr: str, answer: str, conf: float, ms: float) -> None:
    import streamlit as st

    hist = st.session_state.get("history", [])
    if hist and hist[0].get("expr") == expr:
        return
    hist = [{"expr": expr, "answer": answer, "conf": conf, "ms": ms}] + hist
    st.session_state.history = hist[:8]


def main():
    import streamlit as st
    from app.ui.canvas import draw_canvas
    from app.ui.components import empty_result_card, history_row_html, section_label
    from app.ui.pipeline_view import render_inspection, render_pipeline
    from app.ui.result_view import render_result_card
    from app.ui.theme import apply_theme, status_pill

    st.set_page_config(page_title="MathVision", page_icon="∑", layout="wide")
    apply_theme(st)

    online = MODEL_PATH.exists()
    try:
        import json

        metrics = json.load(open(METRICS_PATH, encoding="utf-8")) if METRICS_PATH.exists() else {}
    except Exception:
        metrics = {}

    @st.cache_resource(show_spinner=False)
    def _predictor():
        from app.recognition.predictor import SymbolPredictor

        try:
            p = SymbolPredictor()
            p.predict_batch([np.zeros((28, 28), np.uint8)])
            return p
        except Exception:
            return None

    predictor = _predictor() if online else None
    is_online = bool(online and predictor is not None)

    for k, v in {"canvas_key": 0, "solution": None, "history": [],
                 "input_mode": "Draw", "nav": "Workspace"}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ---------- sidebar: purposeful product nav ----------
    with st.sidebar:
        st.markdown('<div class="mv-brand">∑ MATHVISION</div>', unsafe_allow_html=True)
        st.caption("Handwritten math · local CNN")
        nav = st.radio("Section", ["Workspace", "Model", "About"],
                       key="nav", label_visibility="collapsed")
        if st.button("＋ New Equation", use_container_width=True):
            st.session_state.canvas_key += 1
            st.session_state.solution = None
            st.session_state.nav = "Workspace"
            st.rerun()
        st.markdown(section_label("MODEL"), unsafe_allow_html=True)
        st.markdown(status_pill(is_online), unsafe_allow_html=True)
        st.markdown(section_label("RECENT EQUATIONS"), unsafe_allow_html=True)
        if not st.session_state.history:
            st.caption("Nothing solved yet this session.")
        else:
            for h in st.session_state.history:
                st.markdown(history_row_html(h["expr"], h["answer"], h["conf"], h["ms"]),
                            unsafe_allow_html=True)

    # ---------- top bar ----------
    st.markdown(
        f'<div class="mv-topbar"><div class="mv-brand">MATHVISION'
        f'<small>mathematical vision workstation</small></div>'
        f'<div>{status_pill(is_online)}</div></div>',
        unsafe_allow_html=True,
    )

    if nav == "Model":
        st.markdown("### MODEL")
        st.markdown(
            '<p class="mv-small"><b>MathVision CNN</b> · Handwritten symbol classification · '
            "Input 28 × 28 × 1 · Inference: local, offline after install<br>"
            "Classes: <b>0–9</b>, <b>+</b>, <b>−</b>, <b>×</b>, <b>÷</b>, <b>=</b>, <b>x</b></p>",
            unsafe_allow_html=True,
        )
        if metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("TEST ACCURACY", f"{metrics['test_accuracy']:.2%}")
            m2.metric("PRECISION", f"{metrics['macro_precision']:.4f}")
            m3.metric("RECALL", f"{metrics['macro_recall']:.4f}")
            m4.metric("F1", f"{metrics['macro_f1']:.4f}")
            weakest = min(metrics["per_class"].items(), key=lambda kv: kv[1]["recall"])
            st.caption(f"Weakest class: {weakest[0]} ({weakest[1]['recall']:.1%} recall) · "
                       "measured on the real-handwriting test set.")
        else:
            st.info("Metrics appear after `python -m training.evaluate`.")
        st.caption("No cloud APIs · no external OCR · weights load once via cache_resource.")
        return

    if nav == "About":
        st.markdown("### ABOUT")
        st.markdown(
            '<div class="mv-card"><b>MathVision</b> — NNDL mini-project.<br>'
            '<span class="mv-small">Local CNN · Computer Vision · Symbol Recognition · SymPy<br>'
            "Draw an equation, the CNN reads each symbol, SymPy solves it. "
            "Fully offline after install.</span></div>",
            unsafe_allow_html=True,
        )
        return

    # ---------- compact hero ----------
    st.markdown(
        '<div class="mv-hero"><h1>MATHVISION</h1>'
        "<h2>Handwritten mathematics, understood.</h2>"
        '<p class="mv-small">Draw an equation or upload an image. '
        "Our local neural network recognizes the symbols and solves the expression.</p></div>",
        unsafe_allow_html=True,
    )
    if not is_online:
        st.warning("No trained model found — run `python -m training.train`. Segmentation preview only.")

    mode = st.radio("Input mode", ["Draw", "Upload"], horizontal=True,
                    key="input_mode", label_visibility="collapsed")

    col_in, col_out = st.columns([1.05, 0.95], gap="large")

    # ---------- LEFT: input ----------
    with col_in:
        st.markdown(
            '<div class="mv-card">'
            f'{section_label("WRITE YOUR EQUATION")}'
            '<p class="mv-helper">Use clear, separated strokes for best recognition.</p>',
            unsafe_allow_html=True,
        )
        draw_image, upload_image = None, None
        draw_empty = True
        if mode == "Draw":
            try:
                st.markdown('<div class="mv-canvas-wrap">', unsafe_allow_html=True)
                draw_image, draw_empty = draw_canvas(st, key=f"mv_canvas_{st.session_state.canvas_key}")
                st.markdown("</div>", unsafe_allow_html=True)
            except (ImportError, ModuleNotFoundError):
                draw_image, draw_empty = None, True
                st.warning("Canvas package missing — switch to Upload or relaunch with "
                           "`.\\venv\\Scripts\\python.exe run.py`.")
        else:
            upload = st.file_uploader("Drop an equation image here", type=["png", "jpg", "jpeg"])
            st.caption("Supported: PNG · JPG · JPEG")
            if upload is not None:
                try:
                    upload_image = _decode_upload(upload)
                    st.image(cv2.cvtColor(upload_image, cv2.COLOR_BGR2RGB),
                             caption="Upload preview", width="stretch")
                except ValueError as e:
                    st.error(str(e))

        b1, b2 = st.columns(2)
        with b1:
            if st.button("CLEAR", use_container_width=True):
                st.session_state.canvas_key += 1
                st.session_state.solution = None
                st.rerun()
        with b2:
            ready = (draw_image is not None) if mode == "Draw" else (upload_image is not None)
            do_solve = st.button("SOLVE EQUATION", type="primary",
                                 use_container_width=True, disabled=not ready)
        if mode == "Draw" and draw_image is None and not draw_empty:
            st.info("Draw an equation on the canvas, then press SOLVE EQUATION.")
        if mode == "Upload" and upload_image is None:
            st.info("Drop an equation image above to begin.")
        st.markdown("</div>", unsafe_allow_html=True)

        if do_solve:
            target = draw_image if mode == "Draw" else upload_image
            if target is None:
                st.error("Draw or upload an equation first.")
            else:
                with st.spinner("Analyzing handwriting… Preprocessing → Detecting → Recognizing → Solving"):
                    try:
                        sol = solve_image(target, predictor)
                        st.session_state.solution = sol
                        if sol["crops"] and sol["result"] is not None and sol["result"].kind != "error":
                            _push_history(sol["expression"], sol["result"].message,
                                          sol["confidence"], sol["ms"])
                        st.rerun()
                    except Exception:
                        st.error("Could not process that image. Try a clearer photo or rewrite the equation.")

    # ---------- RIGHT: result ----------
    with col_out:
        sol = st.session_state.solution
        if sol is None:
            st.markdown(empty_result_card(), unsafe_allow_html=True)
        elif not sol["crops"]:
            st.error("No mathematical symbols were detected. Try writing larger, darker strokes.")
        else:
            render_result_card(st, sol)
            render_inspection(st, sol)

    # ---------- pipeline ----------
    st.markdown("<br>", unsafe_allow_html=True)
    render_pipeline(st, st.session_state.solution)

    # ---------- examples ----------
    st.markdown("### TRY IT")
    st.caption("Example inputs — every answer is computed live by the CNN + SymPy.")
    ex_cols = st.columns(len(EXAMPLES))
    for col, (label, fname) in zip(ex_cols, EXAMPLES):
        with col:
            st.markdown(f'<div class="mv-chip">{label}</div>', unsafe_allow_html=True)
            if st.button("Run", key=f"ex_{fname}", use_container_width=True):
                p = TEST_DIR / fname
                if p.exists():
                    with st.spinner("Analyzing handwriting…"):
                        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        sol = solve_image(img, predictor)
                        st.session_state.solution = sol
                        if sol["crops"] and sol["result"] is not None and sol["result"].kind != "error":
                            _push_history(sol["expression"], sol["result"].message,
                                          sol["confidence"], sol["ms"])
                        st.session_state.nav = "Workspace"
                        st.rerun()
                else:
                    st.error(f"Example file missing: {fname}")

    st.markdown(
        '<div class="mv-footer">MATHVISION<br>'
        '<span class="mv-muted">Local CNN · Computer Vision · Symbol Recognition · SymPy<br>'
        "NNDL Mini Project · Runs fully offline</span></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
