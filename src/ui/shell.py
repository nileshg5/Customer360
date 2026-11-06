"""App shell: page config, theme, data loading, global filters, header and navigation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import Settings, load_config
from src.ui import components as ui
from src.ui import theme
from src.ui.data import Bundle, Filters, apply_filters, current_run_id, latest_dir, load_bundle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ENV = "C360_CONFIG"
SEGMENTS = ["Gold", "Silver", "Bronze", "Unknown"]
RISK_BANDS = ["High", "Medium", "Low"]


@dataclass
class AppContext:
    cfg: Settings
    bundle: Bundle
    df: pd.DataFrame          # customers after global filters
    filters: Filters


def _pages() -> List[Tuple[str, str, str, Callable[[AppContext], None]]]:
    from src.ui.pages import ask_ai, data_quality, overview, pipeline_run, revenue, segments, support
    return [
        ("overview", "Overview", ":material/space_dashboard:", overview.render),
        ("revenue", "Revenue & Regions", ":material/payments:", revenue.render),
        ("segments", "Segments & Customers", ":material/groups:", segments.render),
        ("support", "Support", ":material/support_agent:", support.render),
        ("data-quality", "Data Quality", ":material/verified:", data_quality.render),
        ("ask-ai", "Ask AI", ":material/forum:", ask_ai.render),
        ("pipeline", "Pipeline Run", ":material/account_tree:", pipeline_run.render),
    ]




LOGO = Path(__file__).resolve().parent / "assets" / "logo.svg"


def _brand() -> None:
    st.logo(str(LOGO), size="large")


def _missing_data_screen(cfg: Settings) -> None:
    ui.md(f'''<div class="c360-panel center">
      <div style="font-size:2.2rem">&#9881;</div>
      <h2>No pipeline output yet</h2>
      <p>The dashboard reads <code>{ui.esc(latest_dir(cfg).relative_to(cfg.project_root).as_posix())}/</code>, which doesn't exist or isn't complete.</p>
      <p style="margin-top:14px">Run the pipeline first, then refresh this page:</p>
      <p style="margin-top:10px"><code>python main.py</code></p>
    </div>''')


def _sidebar_filters(customers: pd.DataFrame) -> Filters:
    st.sidebar.markdown('<div class="c360-section"><div class="title">Filters</div>'
                        '<div class="caption">Apply to every customer view</div></div>', unsafe_allow_html=True)
    cities = st.sidebar.multiselect("City", sorted(customers["city"].unique()), key="f_city",
                                    placeholder="All cities")
    segs = st.sidebar.multiselect("Segment", [s for s in SEGMENTS if s in set(customers["segment"])],
                                  key="f_segment", placeholder="All segments")
    bands = st.sidebar.multiselect("Risk band", [b for b in RISK_BANDS if b in set(customers["risk_band"])],
                                   key="f_risk", placeholder="All risk bands")
    f = Filters(cities, segs, bands)
    if f.active and st.sidebar.button("Clear filters", icon=":material/filter_alt_off:", width="stretch"):
        for k in ("f_city", "f_segment", "f_risk"):
            st.session_state[k] = []
        st.rerun()
    return f


def _header(ctx: AppContext, title: str, subtitle: str) -> None:
    m, dq = ctx.bundle.manifest, ctx.bundle.dq
    healthy = m.get("status") == "success" and dq["reconciliation"]["balanced"]
    status = ui.badge("Data OK" if healthy else "Check data", "good" if healthy else "warning")
    meta = ui.chips([f"As of <b>{ui.esc(m['as_of_date'])}</b>", f"Run <b>{ui.esc(m['run_id'])}</b>"]) + status
    ui.page_header(title, subtitle, meta)
    f = ctx.filters
    if f.active:
        labels = ([f"City: {c}" for c in f.cities] + [f"Segment: {s}" for s in f.segments]
                  + [f"Risk: {b}" for b in f.risk_bands])
        ui.md('<div class="c360-chips" style="margin:-6px 0 14px">'
              + ui.chips([ui.esc(x) for x in labels], accent=True)
              + f'<span class="c360-chip">{len(ctx.df)} of {len(ctx.bundle.customers)} customers</span></div>')


SUBTITLES: Dict[str, str] = {
    "overview": "Revenue, segments, support and data health at a glance",
    "revenue": "Where revenue comes from, by city and by customer",
    "segments": "Gold / Silver / Bronze mix and the full customer profile",
    "support": "Ticket load, heavy users and low-value high-effort customers",
    "data-quality": "Every source row accounted for: rejected, flagged or clean",
    "ask-ai": "Ask questions in plain English; answers come from read-only SQL",
    "pipeline": "Run metadata, row counts per stage, timings and downloads",
}


def _render(ctx: AppContext, key: str, title: str, fn: Callable[[AppContext], None]) -> None:
    _header(ctx, title, SUBTITLES[key])
    fn(ctx)


def main(page: Optional[str] = None, config_path: Optional[str] = None) -> None:
    """Entry point. `page` renders one page directly (used by tests); otherwise st.navigation."""
    st.set_page_config(page_title="Customer 360", page_icon=":material/hub:", layout="wide",
                       initial_sidebar_state="expanded")
    theme.inject()
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(Path(config_path or os.environ.get(CONFIG_ENV) or PROJECT_ROOT / "config.json"))
    _brand()

    run_id = current_run_id(latest_dir(cfg))
    if run_id is None:
        _missing_data_screen(cfg)
        return
    bundle = load_bundle(str(latest_dir(cfg)), run_id)
    filters = _sidebar_filters(bundle.customers)
    ctx = AppContext(cfg, bundle, apply_filters(bundle.customers, filters), filters)
    pages = _pages()

    if page is not None:
        key, title, _, fn = next(p for p in pages if p[0] == page)
        _render(ctx, key, title, fn)
        return

    def make(key: str, title: str, fn: Callable[[AppContext], None]):
        def run() -> None:
            _render(ctx, key, title, fn)
        run.__name__ = f"page_{key.replace('-', '_')}"
        return run

    nav = st.navigation([st.Page(make(k, t, fn), title=t, icon=icon, url_path=k, default=(k == "overview"))
                         for k, t, icon, fn in pages])
    nav.run()
