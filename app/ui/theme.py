"""MathVision design tokens + global CSS.

Single source of truth for colors, spacing, typography, cards,
buttons, status pills and confidence bars. No artwork, no heavy
gradients, no glassmorphism — restrained technical dark theme.
"""
from __future__ import annotations

BG = "#0b0e10"
PANEL = "#14181c"
PANEL_2 = "#181e23"
BORDER = "#242c33"
BORDER_SOFT = "#1e252b"
TEXT = "#f2f5f6"
MUTED = "#9aa4ab"
ACCENT = "#2fd597"    # electric green / mint
ACCENT_DIM = "#1a8f66"
ACCENT2 = "#59c2ff"   # cool blue/cyan
WARN = "#ffb224"
ERROR = "#ff6b6b"

FONT = ("Inter, 'Segoe UI', system-ui, -apple-system, Roboto, "
        "'Helvetica Neue', Arial, sans-serif")
MONO = ("'JetBrains Mono', 'Cascadia Code', Consolas, 'Courier New', monospace")

CSS = f"""
<style>
:root {{
  --mv-bg: {BG}; --mv-panel: {PANEL}; --mv-border: {BORDER};
  --mv-text: {TEXT}; --mv-muted: {MUTED};
  --mv-accent: {ACCENT}; --mv-accent2: {ACCENT2};
}}
.stApp {{ background: {BG}; color: {TEXT}; font-family: {FONT}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: {BG}; }}
section[data-testid="stSidebar"] {{
  background: {PANEL}; border-right: 1px solid {BORDER_SOFT};
}}
section[data-testid="stSidebar"] .stRadio > div {{ gap: 4px; }}
h1, h2, h3, h4 {{ color: {TEXT} !important; letter-spacing: -0.02em; font-family: {FONT}; }}

.mv-topbar {{
  display: flex; align-items: center; justify-content: space-between;
  border: 1px solid {BORDER_SOFT}; background: {PANEL};
  border-radius: 12px; padding: 10px 16px; margin: 4px 0 14px 0;
}}
.mv-brand {{ font-weight: 800; letter-spacing: 0.12em; font-size: 1.0rem; }}
.mv-brand small {{ color: {MUTED}; font-weight: 500; letter-spacing: 0.02em; margin-left: 10px; }}
.mv-hero {{ margin: 6px 0 2px 0; }}
.mv-hero h1 {{
  font-size: 2.6rem; font-weight: 800; letter-spacing: 0.02em;
  margin: 0; line-height: 1.05;
}}
.mv-hero h2 {{ font-size: 1.25rem; font-weight: 500; color: {TEXT}; margin: 6px 0 4px 0; }}
.mv-muted {{ color: {MUTED}; }}
.mv-small {{ font-size: 0.85rem; color: {MUTED}; line-height: 1.5; }}

.mv-card {{
  background: {PANEL}; border: 1px solid {BORDER_SOFT};
  border-radius: 14px; padding: 18px 20px; margin: 10px 0;
}}
.mv-card-tight {{ padding: 14px 16px; }}
.mv-panel-title {{
  font-size: 0.75rem; font-weight: 700; letter-spacing: 0.14em;
  color: {MUTED}; margin-bottom: 10px;
}}
.mv-helper {{ font-size: 0.82rem; color: {MUTED}; margin: -4px 0 10px 0; }}

.mv-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
  border: 1px solid {BORDER}; border-radius: 999px; padding: 4px 12px;
  white-space: nowrap;
}}
.mv-online {{ color: {ACCENT}; border-color: {ACCENT}; }}
.mv-offline {{ color: {WARN}; border-color: {WARN}; }}
.mv-dot {{ font-size: 0.65rem; }}

.mv-eq {{
  font-family: {MONO}; font-size: 2rem; font-weight: 700;
  text-align: center; margin: 8px 0; letter-spacing: 0.02em;
}}
.mv-sol {{
  font-family: {MONO}; font-size: 1.7rem; font-weight: 800;
  text-align: center; color: {ACCENT}; margin: 8px 0;
}}
.mv-rule {{ border: none; border-top: 1px solid {BORDER_SOFT}; margin: 14px 0; }}
.mv-kv {{ display: flex; justify-content: space-between; font-size: 0.85rem; padding: 3px 0; }}
.mv-kv b {{ font-family: {MONO}; }}
.mv-empty-ico {{ font-size: 2rem; color: {BORDER}; text-align: center; }}

.mv-steps {{ display: flex; gap: 8px; }}
.mv-step {{
  flex: 1; background: {PANEL}; border: 1px solid {BORDER_SOFT};
  border-radius: 10px; padding: 10px 6px; text-align: center;
}}
.mv-step .ic {{
  display: inline-block; font-family: {MONO}; font-weight: 700; font-size: 0.8rem;
  border: 1px solid {BORDER}; border-radius: 6px; padding: 2px 8px; margin-bottom: 6px;
  color: {TEXT}; background: {PANEL_2};
}}
.mv-step .lb {{ font-size: 0.72rem; color: {MUTED}; }}
.mv-step .vl {{ font-size: 0.75rem; color: {TEXT}; font-family: {MONO}; margin-top: 4px; }}

.mv-symrow {{
  display: flex; align-items: center; gap: 10px;
  border: 1px solid {BORDER_SOFT}; border-radius: 8px;
  padding: 7px 10px; margin: 6px 0; background: {PANEL_2};
  font-family: {MONO};
}}
.mv-symrow .sym {{
  width: 44px; text-align: center; font-weight: 800; font-size: 1.05rem;
  border: 1px solid {BORDER}; border-radius: 6px; padding: 2px 0; background: {PANEL};
}}
.mv-bar {{ flex: 1; height: 6px; background: #0e1316; border-radius: 4px; overflow: hidden; }}
.mv-bar > span {{ display: block; height: 100%; background: {ACCENT}; border-radius: 4px; }}
.mv-bar.low > span {{ background: {WARN}; }}
.mv-conf {{ width: 64px; text-align: right; color: {MUTED}; font-size: 0.82rem; }}

.mv-chip {{
  border: 1px solid {BORDER}; background: {PANEL_2}; color: {TEXT};
  border-radius: 8px; padding: 8px 10px; font-family: {MONO}; font-size: 0.88rem;
  text-align: center;
}}
.mv-hist {{
  border: 1px solid {BORDER_SOFT}; border-radius: 8px; padding: 8px 10px;
  margin: 6px 0; font-family: {MONO}; font-size: 0.8rem; background: {PANEL_2};
}}
.mv-hist .ans {{ color: {ACCENT}; font-weight: 700; }}
.mv-hist .meta {{ color: {MUTED}; font-size: 0.72rem; }}

.mv-canvas-wrap {{
  background: #ffffff; border: 1px solid {BORDER};
  border-radius: 12px; padding: 10px;
}}
.mv-canvas-wrap iframe {{ border-radius: 8px; }}
.mv-footer {{ text-align: center; color: {MUTED}; font-size: 0.78rem; margin-top: 18px; }}

/* Streamlit control tuning — restrained, not neon */
div.stButton > button {{
  border-radius: 10px; font-weight: 700; letter-spacing: 0.02em;
  border: 1px solid {BORDER}; background: {PANEL_2}; color: {TEXT};
  padding: 0.55rem 1rem;
}}
div.stButton > button:hover {{ border-color: {ACCENT}; color: {TEXT}; }}
div.stButton > button[kind="primary"] {{
  background: {ACCENT}; color: #06110c; border: none;
}}
div.stButton > button[kind="primary"]:hover {{ background: #45e0a8; color: #06110c; }}
div.stButton > button:disabled {{ opacity: 0.45; }}
div[data-testid="stRadio"] div[role="radiogroup"] {{
  border: 1px solid {BORDER_SOFT}; border-radius: 10px; padding: 4px;
  background: {PANEL}; gap: 4px;
}}
.stExpander {{ border: 1px solid {BORDER_SOFT}; border-radius: 12px; background: {PANEL}; }}
canvas {{ border-radius: 8px; }}
</style>
"""


def apply_theme(st) -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def status_pill(online: bool) -> str:
    cls = "mv-online" if online else "mv-offline"
    label = "MODEL ONLINE" if online else "MODEL OFFLINE"
    return f'<span class="mv-badge {cls}"><span class="mv-dot">●</span> {label}</span>'
