"""MathVision dark-theme tokens + global CSS (no artwork, no heavy gradients)."""
from __future__ import annotations

BG = "#0e1113"
PANEL = "#161a1d"
BORDER = "#2a3136"
TEXT = "#f2f5f6"
MUTED = "#9aa4ab"
ACCENT = "#2fd597"   # green
ACCENT2 = "#59c2ff"  # cyan (sparingly)
WARN = "#ffb224"
ERROR = "#ff5d5d"

CSS = f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; }}
h1, h2, h3 {{ color: {TEXT} !important; letter-spacing: -0.02em; }}
.mv-muted {{ color: {MUTED}; }}
.mv-card {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 20px 22px; margin: 10px 0;
}}
.mv-eq {{ font-size: 2.0rem; font-weight: 700; text-align: center; margin: 6px 0; }}
.mv-sol {{ font-size: 1.6rem; font-weight: 700; text-align: center; color: {ACCENT}; margin: 6px 0; }}
.mv-badge {{
  display: inline-block; font-size: 0.8rem; font-weight: 600;
  border: 1px solid {BORDER}; border-radius: 999px; padding: 3px 12px;
}}
.mv-online {{ color: {ACCENT}; border-color: {ACCENT}; }}
.mv-offline {{ color: {WARN}; border-color: {WARN}; }}
.mv-step {{ color: {MUTED}; font-size: 0.85rem; }}
.mv-step-card {{ text-align: center; padding: 12px 6px !important; font-size: 0.8rem; }}
.mv-accent {{ color: {ACCENT}; }}
div.stButton > button[kind="primary"] {{
  background: {ACCENT}; color: #06110c; border: none; font-weight: 700;
  border-radius: 10px;
}}
div.stButton > button {{
  border-radius: 10px;
}}
canvas {{ border-radius: 12px; }}
</style>
"""


def apply_theme(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_badge(st, online: bool) -> None:
    cls = "mv-online" if online else "mv-offline"
    label = "MODEL ONLINE" if online else "MODEL OFFLINE"
    st.markdown(f'<span class="mv-badge {cls}">● {label}</span>', unsafe_allow_html=True)
