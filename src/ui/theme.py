"""Midnight Teal design system: colour tokens, injected CSS and the shared Plotly template.

UI accent (#2DD4BF) is for chrome only. Chart series use the validated palette below
(checked with the dataviz validator against the card surface #111A2E: lightness band,
chroma, CVD and normal-vision separation, 3:1 contrast all PASS).
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

COLORS = {
    "bg": "#0B1220",
    "sidebar": "#0E1628",
    "card": "#111A2E",
    "card_hi": "#16213A",
    "border": "#1F2A44",
    "accent": "#2DD4BF",
    "accent_dim": "rgba(45, 212, 191, 0.14)",
    "text": "#E2E8F0",
    "text_2": "#A9B6CA",
    "muted": "#7C8BA3",
    "grid": "#1C2740",
}
# Categorical series, fixed order (validated on the dark card surface).
SERIES = ["#17A898", "#3987e5", "#d95926"]
NEUTRAL = "#64748B"
# Reserved status colours: always shown with a text label, never colour alone.
STATUS = {"good": "#0ca30c", "warning": "#fab219", "critical": "#d03b3b"}

SEGMENT_COLORS = {"Gold": SERIES[0], "Silver": SERIES[1], "Bronze": SERIES[2], "Unknown": NEUTRAL}
RISK_COLORS = {"Low": STATUS["good"], "Medium": STATUS["warning"], "High": STATUS["critical"]}
SEVERITY_COLORS = {"error": STATUS["critical"], "warning": STATUS["warning"]}

FONT_STACK = ('-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, '
              '"Noto Sans", sans-serif')
TEMPLATE = "c360_midnight"

_CSS = f"""
<style>
:root {{
  --bg: {COLORS['bg']}; --card: {COLORS['card']}; --card-hi: {COLORS['card_hi']};
  --border: {COLORS['border']}; --accent: {COLORS['accent']}; --accent-dim: {COLORS['accent_dim']};
  --text: {COLORS['text']}; --text-2: {COLORS['text_2']}; --muted: {COLORS['muted']};
  --good: {STATUS['good']}; --warn: {STATUS['warning']}; --crit: {STATUS['critical']};
}}
html, body, .stApp, .stMarkdown, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
button, input, textarea, [data-baseweb="select"] {{ font-family: {FONT_STACK}; }}
/* keep icon ligatures on the Material Symbols font */
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: "Material Symbols Rounded" !important; }}
[data-testid="stAppViewContainer"], .stApp {{ background: var(--bg); }}
/* hide default chrome */
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1440px; }}
/* sidebar */
[data-testid="stSidebar"] {{ background: {COLORS['sidebar']}; border-right: 1px solid var(--border); }}
[data-testid="stSidebarNav"] a {{ border-radius: 10px; }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{ background: var(--accent-dim); }}
[data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: var(--accent) !important; }}
/* bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--card); border-color: var(--border) !important;
  border-radius: 14px !important; }}
[data-testid="stExpander"] details {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; }}
[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
/* buttons */
[data-testid="stBaseButton-primary"] {{ background: var(--accent); color: #04211D; border: 0; font-weight: 600; }}
[data-testid="stBaseButton-secondary"] {{ background: var(--card-hi); border: 1px solid var(--border); color: var(--text); }}
[data-testid="stBaseButton-secondary"]:hover {{ border-color: var(--accent); color: var(--accent); }}
[data-baseweb="tag"] {{ background: var(--accent-dim) !important; color: var(--accent) !important; }}
/* brand + header */
.c360-header {{ display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-end; gap: 10px;
  border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 18px; }}
.c360-header h1 {{ color: var(--text); font-size: 1.6rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.2; }}
.c360-header .sub {{ color: var(--muted); font-size: .9rem; margin-top: 4px; }}
/* chips + badges */
.c360-chips {{ display:flex; flex-wrap:wrap; gap: 6px; }}
.c360-chip {{ display:inline-flex; align-items:center; gap:6px; padding: 3px 10px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--card); color: var(--text-2); font-size: .78rem; white-space: nowrap; }}
.c360-chip b {{ color: var(--text); font-weight: 600; }}
.c360-chip.accent {{ border-color: rgba(45,212,191,.45); color: var(--accent); background: var(--accent-dim); }}
.c360-dot {{ width: 8px; height: 8px; border-radius: 50%; display:inline-block; }}
.c360-badge {{ display:inline-flex; align-items:center; gap:6px; padding: 2px 9px; border-radius: 6px;
  font-size: .76rem; font-weight: 600; border: 1px solid; }}
.c360-badge.good {{ color: #5EE35E; border-color: rgba(12,163,12,.5); background: rgba(12,163,12,.12); }}
.c360-badge.warning {{ color: var(--warn); border-color: rgba(250,178,25,.5); background: rgba(250,178,25,.10); }}
.c360-badge.critical {{ color: #FF8A8A; border-color: rgba(208,59,59,.6); background: rgba(208,59,59,.14); }}
.c360-badge.neutral {{ color: var(--text-2); border-color: var(--border); background: var(--card); }}
/* KPI cards */
.c360-kpis {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; margin: 2px 0 20px; }}
.c360-kpi {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px 16px 13px;
  position: relative; overflow: hidden; }}
.c360-kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; background: var(--border); }}
.c360-kpi.accent::before {{ background: var(--accent); }}
.c360-kpi.good::before {{ background: var(--good); }}
.c360-kpi.warning::before {{ background: var(--warn); }}
.c360-kpi.critical::before {{ background: var(--crit); }}
.c360-kpi .label {{ color: var(--muted); font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; font-weight: 600; }}
.c360-kpi .value {{ color: var(--text); font-size: 1.6rem; font-weight: 700; margin-top: 6px; line-height: 1.15;
  font-variant-numeric: tabular-nums; }}
.c360-kpi .sub {{ color: var(--text-2); font-size: .78rem; margin-top: 4px; }}
/* section header */
.c360-section {{ margin: 8px 0 10px; }}
.c360-section .title {{ color: var(--text); font-size: 1.02rem; font-weight: 650; }}
.c360-section .caption {{ color: var(--muted); font-size: .82rem; margin-top: 2px; }}
/* empty + info panels */
.c360-panel {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 22px 24px; }}
.c360-panel.center {{ text-align:center; padding: 56px 24px; }}
.c360-panel h2 {{ color: var(--text); font-size: 1.3rem; margin: 8px 0 6px; }}
.c360-panel p {{ color: var(--text-2); margin: 4px 0; }}
.c360-panel code {{ background: var(--bg); border: 1px solid var(--border); color: var(--accent); padding: 2px 8px; border-radius: 6px; }}
.c360-legend {{ display:flex; gap: 14px; flex-wrap: wrap; color: var(--text-2); font-size: .8rem; margin: 2px 0 8px; }}
.c360-kv {{ display:grid; grid-template-columns: 140px 1fr; gap: 6px 12px; font-size: .88rem; }}
.c360-kv .k {{ color: var(--muted); }} .c360-kv .v {{ color: var(--text); }}
</style>
"""


def _register_plotly_template() -> None:
    axis = dict(gridcolor=COLORS["grid"], linecolor=COLORS["border"], zeroline=False, automargin=True,
                tickfont=dict(color=COLORS["text_2"], size=11), title=dict(font=dict(color=COLORS["text_2"], size=12)),
                showline=False, ticks="")
    pio.templates[TEMPLATE] = go.layout.Template(layout=go.Layout(
        font=dict(family=FONT_STACK, color=COLORS["text"], size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=SERIES + [NEUTRAL],
        xaxis=axis, yaxis=axis,
        margin=dict(l=8, r=12, t=10, b=8),
        hoverlabel=dict(bgcolor=COLORS["card_hi"], bordercolor=COLORS["border"],
                        font=dict(color=COLORS["text"], family=FONT_STACK, size=12)),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(color=COLORS["text_2"], size=11),
                    bgcolor="rgba(0,0,0,0)"),
        bargap=0.32, barcornerradius=4,
    ))


def inject() -> None:
    """Inject CSS and make the Plotly template the default. Call once per script run."""
    st.markdown(_CSS, unsafe_allow_html=True)
    if TEMPLATE not in pio.templates:
        _register_plotly_template()
    pio.templates.default = TEMPLATE
