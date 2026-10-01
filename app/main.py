"""MathVision Streamlit app — dark themed two-column workspace.

DRAWING and UPLOAD share one backend: run_pipeline() below.
Nothing is hardcoded: predictions, confidences, equations, answers and
timings all come from the live CNN + SymPy pipeline.
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
    confs: list[float] = []
    details: list = []
    if crops:
        try:
            predictor = predictor or SymbolPredictor()
            results = predictor.predict_batch([c.image28 for c in crops])
            for lab, conf, _probs in results:
                labels.append(lab)
                confs.append(conf)
                details.append((lab, conf))
        except Exception:
            # No weights / no TF / anything else: recognition unavailable,
            # but segmentation preview still works (never crash the UI).
            labels = ["?" for _ in crops]
            confs = [0.0 for _ in crops]
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


def main():
    import streamlit as st
    from app.ui.theme import apply_theme, status_badge

    st.set_page_config(page_title="MathVision", page_icon="∑", layout="wide")
    apply_theme(st)

    online = MODEL_PATH.exists()
    try:
        metrics = __import__("json").load(open(METRICS_PATH, encoding="utf-8")) \
            if METRICS_PATH.exists() else {}
    except Exception:
        metrics = {}

    @st.cache_resource(show_spinner=False)
    def _predictor():
        from app.recognition.predictor import SymbolPredictor
        try:
            p = SymbolPredictor()
            # warm up (loads weights now so SOLVE timing is pure inference)
            p.predict_batch([np.zeros((28, 28), np.uint8)])
            return p
        except Exception:
            # Missing weights, missing TF, anything else: OFFLINE preview
            # mode instead of a raw traceback (see status badge below).
            return None

    predictor = _predictor() if online else None

    # ---------- header ----------
    st.markdown("# MATHVISION")
    st.markdown("## Handwritten Mathematics, Understood.")
    st.markdown('<p class="mv-muted">Local CNN-powered mathematical expression '
                "recognition and solving.</p>", unsafe_allow_html=True)
    st.markdown('<p class="mv-muted">Local CNN • Computer Vision • '
                "Symbol Recognition • SymPy</p>", unsafe_allow_html=True)
    status_badge(st, online and predictor is not None)

    if not online:
        st.warning("No trained model found — run `python -m training.train`. "
                   "Segmentation preview only.")

    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = 0
    if "solution" not in st.session_state:
        st.session_state.solution = None

    col_in, col_out = st.columns(2, gap="large")

    # ---------- INPUT ----------
    with col_in:
        st.markdown("### INPUT")
        tabs = st.tabs(["DRAW EQUATION", "UPLOAD IMAGE"])
        draw_image = None
        upload_image = None
        draw_empty = True
        do_solve_draw = False
        do_solve_upload = False
        with tabs[0]:
            try:
                from app.ui.canvas import draw_canvas
                draw_image, draw_empty = draw_canvas(st, key=f"mv_canvas_{st.session_state.canvas_key}")
            except (ImportError, ModuleNotFoundError):
                draw_image, draw_empty = None, True
                st.warning("Drawing canvas is unavailable in this Python environment "
                           "(`streamlit-drawable-canvas` not installed). Use the "
                           "UPLOAD IMAGE tab — or relaunch with "
                           "`.\\venv\\Scripts\\python.exe run.py`.")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("CLEAR", use_container_width=True):
                    st.session_state.canvas_key += 1
                    st.session_state.solution = None
                    st.rerun()
            with b2:
                do_solve_draw = st.button("SOLVE", type="primary", use_container_width=True,
                                          disabled=draw_image is None)
            if draw_image is None and not draw_empty:
                st.info("Draw an equation on the canvas, then press SOLVE.")
        with tabs[1]:
            upload = st.file_uploader("Equation image", type=["png", "jpg", "jpeg"])
            if upload is not None:
                try:
                    upload_image = _decode_upload(upload)
                except ValueError as e:
                    st.error(str(e))
                    upload_image = None
            if upload_image is not None:
                st.image(cv2.cvtColor(upload_image, cv2.COLOR_BGR2RGB), caption="Upload preview")
                do_solve_upload = st.button("SOLVE UPLOAD", type="primary", use_container_width=True)
            else:
                st.info("Upload a PNG/JPG photo of a handwritten equation.")

        if do_solve_draw and draw_image is not None:
            with st.spinner("Analyzing handwriting…"):
                try:
                    st.session_state.solution = solve_image(draw_image, predictor)
                except Exception:  # never show raw tracebacks
                    st.error("Could not process that image. "
                             "Try a clearer photo or rewrite the equation.")
                    st.session_state.solution = None
        elif do_solve_upload and upload_image is not None:
            with st.spinner("Analyzing handwriting…"):
                try:
                    st.session_state.solution = solve_image(upload_image, predictor)
                except Exception:  # never show raw tracebacks
                    st.error("Could not process that image. "
                             "Try a clearer photo or rewrite the equation.")
                    st.session_state.solution = None

    # ---------- RESULT ----------
    with col_out:
        st.markdown("### SOLUTION")
        sol = st.session_state.solution
        if sol is None:
            st.markdown('<div class="mv-card mv-muted">Draw or upload an equation, '
                        "then press SOLVE.</div>", unsafe_allow_html=True)
        elif not sol["crops"]:
            st.error("No mathematical symbols were detected. "
                     "Try writing larger, darker strokes.")
        else:
            low = [d for d in sol["details"] if d[1] < 0.60]
            st.markdown('<div class="mv-card">', unsafe_allow_html=True)
            st.markdown("RECOGNIZED EQUATION")
            st.markdown(f'<div class="mv-eq">{sol["expression"]}</div>',
                        unsafe_allow_html=True)
            st.divider()
            st.markdown("SOLUTION")
            if sol["result"] is None:
                st.error("Recognition incomplete (model missing).")
            elif sol["result"].kind == "error":
                st.error("We recognized the input, but it could not be interpreted "
                         f"as a valid equation. ({sol['result'].message})")
            else:
                st.markdown(f'<div class="mv-sol">{sol["result"].message}</div>',
                            unsafe_allow_html=True)
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("MODEL CONFIDENCE", f"{sol['confidence']:.1%}")
            c2.metric("INFERENCE TIME", f"{sol['ms']:.0f} ms")
            if low:
                st.warning("Low-confidence recognition. Try rewriting the equation.")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Recognition Details"):
                from app.ui.components import render_crops
                st.markdown("**Original input**")
                st.image(cv2.cvtColor(sol["image"], cv2.COLOR_BGR2RGB))
                st.markdown("**Preprocessed (binary)**")
                st.image(sol["binary"])
                render_crops(sol["crops"], sol["details"])
                st.markdown("**Symbol · Prediction · Confidence**")
                st.table([{"Symbol": f"`{lab}`", "Confidence": f"{c:.1%}"}
                          for lab, c in sol["details"]])

    # ---------- pipeline / examples / model ----------
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### HOW MATHVISION THINKS")
        steps = [("IMAGE", "Input"), ("FILTER", "Preprocess"),
                 ("GRID", "Segment"), ("NN", "CNN Recognition"),
                 ("EQUATION", "Reconstruct"), ("√", "Solve")]
        cols = st.columns(len(steps))
        for col, (icon, label) in zip(cols, steps):
            with col:
                st.markdown(f'<div class="mv-card mv-step-card">'
                            f"<strong>[ {icon} ]</strong><br>{label}</div>",
                            unsafe_allow_html=True)
        st.markdown("### TRY AN EXAMPLE")
        st.caption("Example inputs — every answer is computed live by the CNN + SymPy.")
        for label, fname in EXAMPLES:
            if st.button(label, key=f"ex_{fname}"):
                p = TEST_DIR / fname
                if p.exists():
                    with st.spinner("Analyzing handwriting…"):
                        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                        st.session_state.solution = solve_image(img, predictor)
                        st.rerun()
    with c2:
        st.markdown("### MODEL")
        st.markdown("**MathVision CNN** · Input 28 × 28 × 1 · "
                    "Handwritten symbol classification · Inference: local")
        st.markdown("Classes: `0–9`, `+`, `−`, `×`, `÷`, `=`, `x`")
        if metrics:
            st.metric("MEASURED TEST ACCURACY", f"{metrics['test_accuracy']:.2%}")
            st.caption(f"Macro F1 {metrics['macro_f1']:.4f} · "
                       f"weakest: {min(metrics['per_class'].items(), key=lambda kv: kv[1]['recall'])[0]} "
                       f"({min(v['recall'] for v in metrics['per_class'].values()):.1%} recall)")
        else:
            st.caption("Accuracy appears here after `python -m training.evaluate`.")
    st.caption("MathVision runs fully offline after install — no cloud APIs.")
    st.divider()
    st.markdown('<p class="mv-muted" style="text-align:center">MathVision<br>'
                "NNDL Mini Project · Local Deep Learning Pipeline</p>",
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
