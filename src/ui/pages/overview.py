"""Overview: KPIs, revenue by city, segment mix, order recency trend, risk mix, DQ strip."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.qa.exec_summary import build_aggregates, generate, render_markdown
from src.qa.llm_client import client_from_env
from src.ui import components as ui
from src.ui.theme import COLORS, RISK_COLORS, SEGMENT_COLORS, SERIES


def render(ctx) -> None:
    df, dq = ctx.df, ctx.bundle.dq
    if df.empty:
        ui.empty_state("No customers match the filters", "Clear or change the filters in the sidebar.")
        return
    revenue = df["lifetime_value"].sum()
    pass_rate = dq["successful_records"] / dq["total_records"] * 100
    ui.kpi_row([
        {"label": "Customers", "value": ui.fmt_num(len(df)), "sub": f"of {len(ctx.bundle.customers)} in the 360", "tone": "accent"},
        {"label": "Total revenue", "value": ui.fmt_compact(revenue), "sub": "sum of lifetime value"},
        {"label": "Avg lifetime value", "value": ui.fmt_compact(df["lifetime_value"].mean()), "sub": "per customer with LTV"},
        {"label": "Support tickets", "value": ui.fmt_num(df["support_tickets"].sum()),
         "sub": f"{(df['support_tickets'] > ctx.cfg.support.high_ticket_threshold).sum()} customers > "
                f"{ctx.cfg.support.high_ticket_threshold} tickets"},
        {"label": "At-risk VIPs", "value": ui.fmt_num(int(df["at_risk_vip"].sum())), "sub": "Gold and High risk",
         "tone": "critical" if df["at_risk_vip"].any() else ""},
        {"label": "DQ pass rate", "value": f"{pass_rate:.2f}%", "sub": f"{dq['successful_records']} / {dq['total_records']} source rows",
         "tone": "good"},
    ])

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        ui.section("Revenue by city", "Sum of lifetime value; quarantined values excluded")
        by_city = df.groupby("city", as_index=False)["lifetime_value"].sum()
        ui.plot(ui.hbar(by_city, "city", "lifetime_value", hover_name="Revenue"), height=330, key="ov_city")
    with right, st.container(border=True):
        ui.section("Customer segments", "Gold > 100,000 · Silver 50,000-100,000 · Bronze < 50,000")
        seg = df["segment"].value_counts()
        order = [s for s in SEGMENT_COLORS if s in seg.index]
        ui.plot(ui.donut(order, [int(seg[s]) for s in order], [SEGMENT_COLORS[s] for s in order],
                         f"{len(df)}<br><span style='font-size:11px;color:{COLORS['text_2']}'>customers</span>"),
                height=330, key="ov_seg")

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        ui.section("Last order month", "Customers by the month of their most recent order")
        months = df["last_order_date"].dt.to_period("M").value_counts().sort_index()
        fig = go.Figure(go.Scatter(x=months.index.to_timestamp(), y=months.values, mode="lines+markers",
                                   line=dict(color=SERIES[0], width=2), marker=dict(size=8, color=SERIES[0],
                                   line=dict(color=COLORS["card"], width=2)),
                                   hovertemplate="%{x|%b %Y}: %{y} customers<extra></extra>"))
        fig.update_layout(xaxis=dict(showgrid=False, tickformat="%b"), yaxis=dict(rangemode="tozero"))
        ui.plot(fig, height=260, key="ov_trend")
    with right, st.container(border=True):
        ui.section("Risk bands", "Rule-based risk score; not a prediction")
        bands = [b for b in ("Low", "Medium", "High") if b in set(df["risk_band"])]
        counts = df["risk_band"].value_counts()
        ui.plot(ui.vbar(bands, [int(counts[b]) for b in bands], colors=[RISK_COLORS[b] for b in bands],
                        hover="%{x} risk: %{y} customers"), height=260, key="ov_risk")

    rec = dq["reconciliation"]
    ui.md('<div class="c360-panel" style="padding:14px 18px; display:flex; gap:10px; flex-wrap:wrap; align-items:center">'
          '<b style="color:var(--text)">Data quality</b>'
          + ui.badge(f"{dq['successful_records']} error-free rows", "good")
          + ui.badge(f"{dq['rejected_unique_records']} rejected", "critical")
          + ui.badge(f"{dq['negative_revenue']} value quarantined", "warning")
          + ui.badge(f"{dq['warning_records']} warnings", "warning")
          + ui.badge("Reconciled" if rec["balanced"] else "NOT reconciled", "good" if rec["balanced"] else "critical")
          + f'<span style="color:var(--muted); font-size:.82rem">input {rec["input"]} = clean {rec["clean"]} + rejected {rec["rejected"]}'
          ' · see the Data Quality page</span></div>')

    _executive_summary(ctx)


def _executive_summary(ctx) -> None:
    """Bonus: summary from aggregated numbers only; written to output/summary.md."""
    ui.md('<div style="height:14px"></div>')
    with st.expander("Executive summary", icon=":material/summarize:"):
        st.caption("Uses only aggregated numbers for the whole dataset (no customer rows, names or emails). "
                   "Without a GROQ_API_KEY a template summary is written instead.")
        if st.button("Generate executive summary", type="primary", key="exec_btn"):
            m = ctx.bundle.manifest
            agg = build_aggregates(ctx.bundle.customers, ctx.bundle.dq, ctx.cfg.support.high_ticket_threshold)
            with st.status("Writing summary...") as status:
                text, source = generate(agg, client_from_env(ctx.cfg), m["as_of_date"],
                                        on_status=lambda msg: status.update(label=msg))
                status.update(label=f"Done ({source})", state="complete")
            doc = render_markdown(text, source, m["run_id"], m["as_of_date"])
            (ctx.cfg.output_dir / "summary.md").write_text(doc, encoding="utf-8")
            st.session_state["exec_summary"] = doc
        doc = st.session_state.get("exec_summary")
        if doc:
            st.markdown(doc.split("\n", 1)[1])         # body without the file's H1 title
            st.download_button("summary.md", doc, file_name="summary.md", mime="text/markdown",
                               icon=":material/download:", key="exec_dl")
