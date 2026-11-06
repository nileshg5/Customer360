"""Support: ticket distribution, > threshold list, percentile view, ticket-heavy low-value."""
from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui.theme import COLORS, SERIES, STATUS

LIST_COLUMNS = ["customer_id", "name", "city", "segment", "support_tickets", "last_ticket_date", "lifetime_value"]
COLUMN_CONFIG = {"customer_id": "ID", "name": "Name", "city": "City", "segment": "Segment",
                 "support_tickets": "Tickets",
                 "last_ticket_date": st.column_config.DateColumn("Last ticket", format="YYYY-MM-DD"),
                 "lifetime_value": st.column_config.NumberColumn("Revenue", format="localized"),
                 "tickets_per_order": st.column_config.NumberColumn("Tickets / order", format="%.2f")}


def render(ctx) -> None:
    df = ctx.df
    if df.empty:
        ui.empty_state("No customers match the filters", "Clear or change the filters in the sidebar.")
        return
    t = ctx.cfg.support.high_ticket_threshold
    tickets = df["support_tickets"].dropna()
    heavy = df[df["support_tickets"] > t]
    low_value = heavy[heavy["segment"] == "Bronze"]
    ui.kpi_row([
        {"label": "Total tickets", "value": ui.fmt_num(tickets.sum()), "tone": "accent"},
        {"label": f"Customers > {t} tickets", "value": ui.fmt_num(len(heavy)),
         "sub": f"{len(heavy) / len(df) * 100:.0f}% of customers", "tone": "warning"},
        {"label": "Avg tickets / customer", "value": f"{tickets.mean():.2f}"},
        {"label": "Ticket-heavy, low value", "value": ui.fmt_num(len(low_value)), "sub": f"> {t} tickets and Bronze",
         "tone": "critical" if len(low_value) else ""},
    ])
    st.caption(f"{len(heavy) / len(df) * 100:.0f}% of customers exceed the fixed '> {t} tickets' rule, so it "
               "flags a majority. Use the percentile view below to focus on the heaviest users.")

    left, right = st.columns([3, 2], gap="medium")
    with left, st.container(border=True):
        ui.section("Ticket distribution", f"Customers by ticket count; bars right of the line have > {t} tickets")
        dist = tickets.astype(int).value_counts().sort_index()
        fig = ui.vbar(dist.index, dist.values, colors=[STATUS["warning"] if v > t else SERIES[0] for v in dist.index],
                      hover="%{x} tickets: %{y} customers")
        fig.add_vline(x=t + 0.5, line=dict(color=COLORS["text_2"], width=1, dash="dot"))
        fig.update_layout(xaxis=dict(dtick=1, title=dict(text="tickets")))
        ui.plot(fig, height=280, key="sup_dist")
    with right, st.container(border=True):
        ui.section("Percentile threshold", "Flag customers above a ticket percentile instead of a fixed number")
        pct = st.select_slider("Percentile", options=[50, 60, 70, 75, 80, 90, 95],
                               value=round(ctx.cfg.support.default_percentile * 100), key="sup_pct",
                               format_func=lambda p: f"P{p}")
        cut = float(tickets.quantile(pct / 100))
        above = df[df["support_tickets"] > cut]
        ui.kpi_row([{"label": f"P{pct} ticket count", "value": f"{cut:g}"},
                    {"label": "Customers above", "value": ui.fmt_num(len(above)),
                     "sub": f"{len(above) / len(df) * 100:.0f}% of customers", "tone": "accent"}])

    tab1, tab2, tab3 = st.tabs([f"More than {t} tickets ({len(heavy)})", f"Above P{pct} ({len(above)})",
                                f"Ticket-heavy, low value ({len(low_value)})"])
    order = ["support_tickets", "customer_id"]
    with tab1:
        st.dataframe(heavy.sort_values(order, ascending=[False, True])[LIST_COLUMNS], hide_index=True,
                     height=320, column_config=COLUMN_CONFIG)
    with tab2:
        if above.empty:
            ui.empty_state("Nobody is above this percentile", "Pick a lower percentile.")
        else:
            st.dataframe(above.sort_values(order, ascending=[False, True])[LIST_COLUMNS], hide_index=True,
                         height=320, column_config=COLUMN_CONFIG)
    with tab3:
        if low_value.empty:
            ui.empty_state("No ticket-heavy Bronze customers", "Nothing matches with the current filters.")
        else:
            st.dataframe(low_value.sort_values(order, ascending=[False, True])[LIST_COLUMNS + ["tickets_per_order"]],
                         hide_index=True, height=320, column_config=COLUMN_CONFIG)
