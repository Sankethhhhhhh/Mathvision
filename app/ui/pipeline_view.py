"""Pipeline + inspection views — all values from the live result dict."""
from __future__ import annotations

import traceback

import cv2

from app.ui.components import section_label, symbol_grid_html

STEPS = [
    ("IMAGE", "Input"),
    ("FILTER", "Preprocess"),
    ("SEGMENTS", "Detection"),
    ("CNN", "Recognition"),
    ("EQUATION", "Expression"),
    ("√", "Solution"),
]


def render_pipeline(st, sol: dict | None = None) -> None:
    st.markdown("### HOW MATHVISION UNDERSTANDS YOUR EQUATION")
    st.markdown(
        '<p class="mv-small">Local CNN pipeline — every value below comes '
        "from the real inference run, nothing is mocked.</p>",
        unsafe_allow_html=True,
    )
    if sol is None:
        vals = ["—"] * 6
    else:
        n = len(sol.get("crops", []))
        vals = [
            "ready",
            f"{sol['binary'].shape[1]}×{sol['binary'].shape[0]}",
            f"{n} symbols",
            f"{sol['confidence']:.0%}",
            (sol["expression"] or "—"),
            (sol["result"].message if sol.get("result") else "—"),
        ]
    cols = st.columns(len(STEPS))
    for col, (icon, label), val in zip(cols, STEPS, vals):
        with col:
            st.markdown(
                f'<div class="mv-step"><span class="ic">{icon}</span>'
                f'<div class="lb">{label}</div>'
                f'<div class="vl">{val}</div></div>',
                unsafe_allow_html=True,
            )


def render_inspection(st, sol: dict) -> None:
    """Sophisticated technical expander — no st.table / pandas / pyarrow."""
    with st.expander("MODEL INSPECTION — preprocess · segments · predictions"):
        try:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(section_label("PREPROCESSED IMAGE"), unsafe_allow_html=True)
                # NOTE: Streamlit 1.44 (pinned venv) expects width: int | None.
                # width="stretch" raises TypeError here, so use the 1.44-compatible
                # use_container_width=True to fill the available column width.
                st.image(sol["binary"], caption="Binary — white ink on black",
                         use_container_width=True)
            with c2:
                st.markdown(section_label("ORIGINAL INPUT"), unsafe_allow_html=True)
                st.image(
                    cv2.cvtColor(sol["image"], cv2.COLOR_BGR2RGB),
                    caption="What the CNN saw",
                    use_container_width=True,
                )
            st.markdown(section_label(f"DETECTED SYMBOLS — {len(sol['crops'])}"), unsafe_allow_html=True)
            if sol["crops"]:
                cols = st.columns(min(len(sol["crops"]), 8))
                for i, crop in enumerate(sol["crops"]):
                    lab, conf = sol["details"][i] if i < len(sol["details"]) else ("?", 0.0)
                    with cols[i % len(cols)]:
                        st.image(crop.image28, caption=f"{lab} {conf:.0%}",
                                 use_container_width=True)
            st.markdown(symbol_grid_html(sol["details"]), unsafe_allow_html=True)
        except Exception:
            # Log full traceback for developers (terminal/logs) but show a
            # clean message to normal users instead of a raw traceback.
            traceback.print_exc()
            st.error("Unable to display model inspection results.")
