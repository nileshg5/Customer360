"""Segments & Customers: segment mix, searchable customer table and the Customer 360 card."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.ui import components as ui
from src.ui.theme import COLORS, RISK_COLORS, SEGMENT_COLORS, SERIES

TABLE_COLUMNS = ["customer_id", "name", "email_masked", "city", "segment", "lifetime_value", "total_orders",
                 "support_tickets", "risk_score", "risk_band", "rfm_label"]
RISK_PARTS = [("risk_pts_order_recency", "Order recency"), ("risk_pts_ticket_count", "Ticket count"),
              ("risk_pts_ticket_recency", "Ticket recency"), ("risk_pts_low_ltv", "Low lifetime value")]


def _segment_summary(df):
    g = df.groupby("segment")
    out = g.agg(customers=("customer_id", "size"), revenue=("lifetime_value", lambda s: s.sum(min_count=1)),
                avg_ltv=("lifetime_value", "mean"), avg_tickets=("support_tickets", "mean")).reindex(
        [s for s in SEGMENT_COLORS if s in set(df["segment"])])
    out["avg_ltv"] = out["avg_ltv"].round(2)
    total = out["revenue"].sum()
    out["revenue_share"] = (out["revenue"] / total * 100).round(1) if total else 0.0
    return out.reset_index()


def _detail_card(c) -> None:
    flags = c["dq_flags"] or []
    seg_color = SEGMENT_COLORS.get(c["segment"], COLORS["muted"])
    band_tone = {"High": "critical", "Medium": "warning", "Low": "good"}.get(c["risk_band"], "neutral")
    band_badge = ui.badge(f"{c['risk_band']} risk", band_tone)
    with st.container(border=True):
        ui.md(f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;align-items:center">'
              f'<div><div style="color:var(--text);font-size:1.25rem;font-weight:700">{ui.esc(c["name"])}</div>'
              f'<div style="color:var(--muted);font-size:.85rem">{ui.esc(c["customer_id"])} · {ui.esc(c["city"])}</div></div>'
              f'<div class="c360-chips"><span class="c360-chip"><span class="c360-dot" style="background:{seg_color}"></span>'
              f'<b>{ui.esc(c["segment"])}</b></span>{band_badge}'
              + (ui.badge("At-risk VIP", "critical") if c["at_risk_vip"] else "")
              + f'<span class="c360-chip">RFM <b>{ui.esc(c["rfm_label"])}</b>&nbsp;{ui.esc(c["rfm_score"] or "")}</span></div></div>')
        ui.kpi_row([
            {"label": "Lifetime value", "value": ui.fmt_num(c["lifetime_value"]), "tone": "accent"},
            {"label": "Orders", "value": ui.fmt_num(c["total_orders"])},
            {"label": "Avg order value", "value": ui.fmt_num(c["avg_order_value"], 2)},
            {"label": "Support tickets", "value": ui.fmt_num(c["support_tickets"])},
            {"label": "Risk score", "value": ui.fmt_num(c["risk_score"], 1), "tone": band_tone},
        ])
        left, right = st.columns([2, 3], gap="medium")
        with left:
            ui.section("Profile")
            date = lambda d: d.strftime("%Y-%m-%d") if d == d and d is not None else "-"   # NaT-safe
            rows = [("Email", c["email_masked"]), ("Registered", date(c["registration_date"])),
                    ("Tenure", f'{ui.fmt_num(c["tenure_days"])} days'),
                    ("Last order", f'{date(c["last_order_date"])} ({ui.fmt_num(c["days_since_last_order"])} days ago)'),
                    ("Last ticket", date(c["last_ticket_date"])),
                    ("Tickets / order", ui.fmt_num(c["tickets_per_order"], 2)),
                    ("Sources", c["source_system"])]
            ui.md('<div class="c360-kv">' + "".join(f'<div class="k">{ui.esc(k)}</div><div class="v">{ui.esc(v)}</div>'
                                                    for k, v in rows) + "</div>")
            ui.section("Data-quality flags")
            ui.md(ui.chips([ui.esc(f) for f in flags]) if flags else ui.badge("No flags", "good"))
        with right:
            ui.section("Why this risk score", "Points per component; they add up to the score")
            parts = [(label, c[col]) for col, label in RISK_PARTS]
            fig = go.Figure(go.Bar(
                y=[p[0] for p in parts][::-1], x=[(p[1] if p[1] == p[1] and p[1] is not None else 0) for p in parts][::-1],
                orientation="h", marker=dict(color=RISK_COLORS.get(c["risk_band"], SERIES[0])),
                text=[("n/a" if p[1] is None or p[1] != p[1] else f"{p[1]:.1f}") for p in parts][::-1],
                textposition="outside", cliponaxis=False, textfont=dict(color=COLORS["text_2"]),
                hovertemplate="%{y}: %{x:.1f} pts<extra></extra>"))
            fig.update_layout(xaxis=dict(range=[0, 45], showticklabels=False), yaxis=dict(showgrid=False),
                              margin=dict(l=8, r=40, t=4, b=4))
            ui.plot(fig, height=210, key="seg_risk_parts")
            if any(p[1] is None or p[1] != p[1] for p in parts):
                st.caption("n/a: component could not be computed; remaining weights were rescaled.")


def render(ctx) -> None:
    df = ctx.df
    if df.empty:
        ui.empty_state("No customers match the filters", "Clear or change the filters in the sidebar.")
        return
    summary = _segment_summary(df)
    ui.kpi_row([{"label": f"{r.segment}", "value": ui.fmt_num(r.customers),
                 "sub": (f"{r.revenue_share}% of revenue" if r.segment != "Unknown" else "LTV quarantined"),
                 "tone": "accent" if r.segment == "Gold" else ""} for r in summary.itertuples()])

    left, right = st.columns([2, 3], gap="medium")
    with left, st.container(border=True):
        ui.section("Customers per segment")
        ui.plot(ui.vbar(summary["segment"], summary["customers"],
                        colors=[SEGMENT_COLORS[s] for s in summary["segment"]], hover="%{x}: %{y} customers"),
                height=280, key="seg_bar")
    with right, st.container(border=True):
        ui.section("Segment summary", "Revenue share excludes the quarantined (Unknown) value")
        st.dataframe(summary, hide_index=True, height=200, column_config={
            "segment": "Segment", "customers": "Customers",
            "revenue": st.column_config.NumberColumn("Revenue", format="localized"),
            "avg_ltv": st.column_config.NumberColumn("Avg LTV", format="localized"),
            "avg_tickets": st.column_config.NumberColumn("Avg tickets", format="%.2f"),
            "revenue_share": st.column_config.NumberColumn("Revenue share", format="%.1f%%")})
        ui.legend({k: v for k, v in SEGMENT_COLORS.items() if k in set(df["segment"])})

    ui.section("Customers", "Search by ID, name or city; select a row (or pick below) to open the Customer 360 card")
    query = st.text_input("Search customers", key="seg_search", placeholder="e.g. C0042, Priya, Pune",
                          label_visibility="collapsed")
    table = df
    if query.strip():
        q = query.strip().lower()
        table = df[df["customer_id"].str.lower().str.contains(q, regex=False)
                   | df["name"].str.lower().str.contains(q, regex=False)
                   | df["city"].str.lower().str.contains(q, regex=False)]
    if table.empty:
        ui.empty_state("No customers match your search", "Try another ID, name or city.")
        return
    table = table.sort_values(["lifetime_value", "customer_id"], ascending=[False, True], na_position="last")
    event = st.dataframe(
        table[TABLE_COLUMNS], hide_index=True, height=min(340, 38 + 35 * len(table)), key="seg_table",
        on_select="rerun", selection_mode="single-row",
        column_config={"customer_id": "ID", "name": "Name", "email_masked": "Email (masked)", "city": "City",
                       "segment": "Segment",
                       "lifetime_value": st.column_config.NumberColumn("Revenue", format="localized"),
                       "total_orders": "Orders", "support_tickets": "Tickets",
                       "risk_score": st.column_config.NumberColumn("Risk", format="%.1f"),
                       "risk_band": "Band", "rfm_label": "RFM"})
    ids = table["customer_id"].tolist()
    selected_rows = getattr(getattr(event, "selection", None), "rows", None) or []
    if selected_rows:
        st.session_state["seg_customer"] = ids[selected_rows[0]]
    if st.session_state.get("seg_customer") not in ids:
        st.session_state["seg_customer"] = ids[0]
    cid = st.selectbox("Customer", ids, key="seg_customer",
                       format_func=lambda i: f"{i} · {table.loc[table.customer_id == i, 'name'].iloc[0]}")
    _detail_card(table[table["customer_id"] == cid].iloc[0])
