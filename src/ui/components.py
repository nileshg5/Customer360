"""Reusable HTML components and chart builders for the dashboard."""
from __future__ import annotations

import html
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.theme import COLORS, SERIES

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def esc(value) -> str:
    return html.escape(str(value))


def fmt_num(v, decimals: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return "-"
    return f"{v:,.{decimals}f}"


def fmt_compact(v) -> str:
    """64582401 -> 64.58M. Currency is not stated in the source data, so no symbol."""
    if v is None or pd.isna(v):
        return "-"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{suffix}"
    return f"{v:,.0f}"


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# ---------- layout pieces ----------

def kpi_row(cards: Sequence[Dict[str, str]]) -> None:
    """cards: [{label, value, sub?, tone? (accent|good|warning|critical)}]"""
    items = "".join(
        f'<div class="c360-kpi {esc(c.get("tone", ""))}"><div class="label">{esc(c["label"])}</div>'
        f'<div class="value">{esc(c["value"])}</div>'
        + (f'<div class="sub">{esc(c["sub"])}</div>' if c.get("sub") else "") + "</div>"
        for c in cards)
    md(f'<div class="c360-kpis">{items}</div>')


def section(title: str, caption: str = "") -> None:
    cap = f'<div class="caption">{esc(caption)}</div>' if caption else ""
    md(f'<div class="c360-section"><div class="title">{esc(title)}</div>{cap}</div>')


def chips(items: Iterable[str], accent: bool = False) -> str:
    cls = "c360-chip accent" if accent else "c360-chip"
    return "".join(f'<span class="{cls}">{i}</span>' for i in items)


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="c360-badge {esc(tone)}">{esc(text)}</span>'


def legend(items: Dict[str, str]) -> None:
    """Colour legend with text labels (identity is never colour-only)."""
    md('<div class="c360-legend">' + "".join(
        f'<span><span class="c360-dot" style="background:{c}"></span> {esc(label)}</span>'
        for label, c in items.items()) + "</div>")


def empty_state(title: str, body: str = "") -> None:
    md(f'<div class="c360-panel center"><h2>{esc(title)}</h2><p>{esc(body)}</p></div>')


def page_header(title: str, subtitle: str, meta_chips_html: str) -> None:
    md(f'<div class="c360-header"><div><h1>{esc(title)}</h1><div class="sub">{esc(subtitle)}</div></div>'
       f'<div class="c360-chips">{meta_chips_html}</div></div>')


def plot(fig: go.Figure, height: int = 320, key: Optional[str] = None) -> None:
    fig.update_layout(height=height)
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key)


# ---------- charts ----------

def hbar(df: pd.DataFrame, label: str, value: str, color: str = SERIES[0], value_fmt: str = ",.0f",
         color_by: Optional[Dict[str, str]] = None, hover_name: str = "") -> go.Figure:
    """Horizontal bars sorted largest at top, value labels at the bar end.
    color_by maps a label to its colour, so colour follows the entity, not its position."""
    d = df.sort_values(value, ascending=True)
    marker_color = [color_by[x] for x in d[label]] if color_by else color
    fig = go.Figure(go.Bar(
        x=d[value], y=d[label], orientation="h",
        marker=dict(color=marker_color, line=dict(width=0)),
        text=[format(v, value_fmt) for v in d[value]], textposition="outside", cliponaxis=False,
        textfont=dict(color=COLORS["text_2"], size=11),
        hovertemplate=f"%{{y}}<br>{hover_name or value}: %{{x:{value_fmt}}}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False),
                      margin=dict(l=8, r=60, t=6, b=6))
    return fig


def vbar(x, y, color=SERIES[0], hover: str = "%{x}: %{y:,}", colors=None) -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker=dict(color=colors if colors is not None else color, line=dict(width=0)),
                           hovertemplate=hover + "<extra></extra>"))
    fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=True))
    return fig


def donut(labels: List[str], values: List[float], colors: List[str], center_text: str,
          hover_fmt: str = ",.0f") -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.66, sort=False, direction="clockwise",
        marker=dict(colors=colors, line=dict(color=COLORS["card"], width=2)),
        textinfo="percent", textfont=dict(color="#FFFFFF", size=11),
        hovertemplate=f"%{{label}}: %{{value:{hover_fmt}}} (%{{percent}})<extra></extra>",
    ))
    fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5),
                      annotations=[dict(text=center_text, showarrow=False,
                                        font=dict(size=15, color=COLORS["text"]))],
                      margin=dict(l=4, r=4, t=4, b=4))
    return fig
