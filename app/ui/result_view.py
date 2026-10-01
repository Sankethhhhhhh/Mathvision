"""Result panel rendering — consumes only real pipeline output."""
from __future__ import annotations

from app.ui.components import esc, section_label


def render_result_card(st, sol: dict) -> None:
    """Right-panel result. sol comes from solve_image() — never fabricated."""
    low = [d for d in sol["details"] if d[1] < 0.60]
    expr = esc(sol["expression"] or "—")
    res = sol["result"]

    if res is None:
        st.error("Recognition unavailable — model could not be loaded.")
        return
    if res.kind == "error":
        st.error(
            "The recognized symbols could not be interpreted as a valid "
            f"equation. ({res.message})"
        )
        st.markdown(
            f'<div class="mv-card"><div class="mv-panel-title">RECOGNIZED</div>'
            f'<div class="mv-eq" style="font-size:1.2rem">{expr}</div></div>',
            unsafe_allow_html=True,
        )
        return

    answer = esc(res.message)
    st.markdown(
        '<div class="mv-card">'
        f'{section_label("RECOGNIZED EQUATION")}'
        f'<div class="mv-eq">{expr}</div>'
        '<hr class="mv-rule"/>'
        f'{section_label("SOLUTION")}'
        f'<div class="mv-sol">{answer}</div>'
        '<hr class="mv-rule"/>'
        f'<div class="mv-kv"><span class="mv-muted">Confidence</span><b>{sol["confidence"]:.1%}</b></div>'
        f'<div class="mv-kv"><span class="mv-muted">Inference</span><b>{sol["ms"]:.0f} ms</b></div>'
        f'<div class="mv-kv"><span class="mv-muted">Symbols</span><b>{len(sol["crops"])}</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if low:
        st.warning(
            "Recognition confidence is low — try writing the equation "
            "more clearly with separated strokes."
        )
