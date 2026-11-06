"""Revenue & Regions: revenue and customer count by city, revenue per customer, top 10."""
from __future__ import annotations

import streamlit as st

from src.ui import components as ui
from src.ui.theme import SERIES


def city_table(df):
    g = df.groupby("city")
    out = g.agg(customers=("customer_id", "size"), customers_with_ltv=("lifetime_value", "count"),
                revenue=("lifetime_value", "sum")).reset_index()
    out["revenue_per_customer"] = (out["revenue"] / out["customers_with_ltv"]).round(2)
    return out.sort_values(["revenue", "city"], ascending=[False, True])


def render(ctx) -> None:
    df = ctx.df
    if df.empty:
        ui.empty_state("No customers match the filters", "Clear or change the filters in the sidebar.")
        return
    cities = city_table(df)
    top = cities.iloc[0]
    ui.kpi_row([
        {"label": "Total revenue", "value": ui.fmt_compact(df["lifetime_value"].sum()), "tone": "accent"},
        {"label": "Cities", "value": str(len(cities))},
        {"label": "Top city", "value": str(top["city"]), "sub": f"{ui.fmt_compact(top['revenue'])} revenue"},
        {"label": "Revenue per customer", "value": ui.fmt_compact(df["lifetime_value"].mean()),
         "sub": "customers with a known LTV"},
    ])

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        ui.section("Revenue by city", "Sum of lifetime value")
        ui.plot(ui.hbar(cities, "city", "revenue", hover_name="Revenue"), height=320, key="rv_rev")
    with right, st.container(border=True):
        ui.section("Customer count by city", "Customers in the 360 after filters")
        ui.plot(ui.hbar(cities, "city", "customers", color=SERIES[1], hover_name="Customers"), height=320, key="rv_cnt")

    left, right = st.columns([2, 3], gap="medium")
    with left, st.container(border=True):
        ui.section("Revenue per customer by city", "Revenue / customers with a known LTV")
        ui.plot(ui.hbar(cities, "city", "revenue_per_customer", hover_name="Per customer"), height=390, key="rv_rpc")
    with right, st.container(border=True):
        ui.section("Top 10 customers by revenue", "Ties broken by customer_id")
        top10 = (df[df["lifetime_value"].notna()]
                 .sort_values(["lifetime_value", "customer_id"], ascending=[False, True]).head(10))
        st.dataframe(
            top10[["customer_id", "name", "city", "segment", "lifetime_value", "total_orders", "risk_band"]],
            hide_index=True, height="auto",
            column_config={
                "customer_id": "ID", "name": "Name", "city": "City", "segment": "Segment",
                "lifetime_value": st.column_config.NumberColumn("Revenue", format="localized"),
                "total_orders": st.column_config.NumberColumn("Orders"),
                "risk_band": "Risk",
            })
