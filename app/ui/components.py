"""Reusable presentational helpers — no ML logic here."""
from __future__ import annotations

import html


def esc(s: object) -> str:
    return html.escape(str(s))


def section_label(title: str) -> str:
    return f'<div class="mv-panel-title">{esc(title)}</div>'


def empty_result_card() -> str:
    return (
        '<div class="mv-card" style="text-align:center">'
        '<div class="mv-panel-title">YOUR RESULT</div>'
        '<div class="mv-empty-ico">∑</div>'
        '<p class="mv-small">Draw an equation and click Solve.<br>'
        'Your solution will appear here.</p>'
        '</div>'
    )


def no_input_card() -> str:
    return (
        '<div class="mv-card mv-card-tight">'
        '<b>No equation yet.</b><br>'
        '<span class="mv-small">Draw or upload an equation to begin.</span>'
        '</div>'
    )


def conf_bar_html(label: str, conf: float) -> str:
    pct = max(0.0, min(1.0, float(conf)))
    cls = "mv-bar low" if pct < 0.60 else "mv-bar"
    return (
        f'<div class="mv-symrow"><div class="sym">{esc(label)}</div>'
        f'<div class="{cls}"><span style="width:{pct * 100:.1f}%"></span></div>'
        f'<div class="mv-conf">{pct:.1%}</div></div>'
    )


def symbol_grid_html(details: list[tuple[str, float]]) -> str:
    """Custom HTML grid — deliberately avoids st.table/pandas/pyarrow."""
    rows = "".join(conf_bar_html(lab, c) for lab, c in details)
    return (
        f'{section_label("SYMBOL PREDICTIONS — LIVE CNN OUTPUT")}'
        f'<div>{rows}</div>'
    )


def history_row_html(expr: str, answer: str, conf: float, ms: float) -> str:
    return (
        '<div class="mv-hist">'
        f'<div>{esc(expr)} &nbsp;→&nbsp; <span class="ans">{esc(answer)}</span></div>'
        f'<div class="meta">{conf:.0%} · {ms:.0f} ms</div>'
        '</div>'
    )
