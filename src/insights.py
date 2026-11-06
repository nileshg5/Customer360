"""Business insight tables built from the enriched Customer 360 frame.

Revenue = lifetime_value. Customers with a null LTV (quarantined) are excluded from
revenue sums and rankings but still counted as customers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd

from src.config import Settings

SEGMENT_ORDER = ["Gold", "Silver", "Bronze"]


def _ltv(df: pd.DataFrame) -> pd.Series:
    return df["lifetime_value"].astype("Float64")


def _segment_order(cfg: Settings):
    return SEGMENT_ORDER + [cfg.segments.null_label]


def top10_by_revenue(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    known = df[_ltv(df).notna()]
    top = known.sort_values(["lifetime_value", "customer_id"], ascending=[False, True]).head(10)
    top = top[["customer_id", "name", "city", "segment", "lifetime_value", "total_orders"]]
    return top.reset_index(drop=True).rename_axis("rank").reset_index().assign(rank=lambda d: d["rank"] + 1)


def segment_summary(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    g = df.groupby("segment")
    out = pd.DataFrame({
        "customers": g.size(),
        "total_revenue": g["lifetime_value"].sum(min_count=1),
        "avg_ltv": g["lifetime_value"].mean().round(2),
        "min_ltv": g["lifetime_value"].min(),
        "max_ltv": g["lifetime_value"].max(),
        "avg_orders": g["total_orders"].mean().round(2),
        "avg_tickets": g["support_tickets"].mean().round(2),
    })
    order = [s for s in _segment_order(cfg) if s in out.index]
    return out.loc[order].rename_axis("segment").reset_index()


def segment_revenue_share(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    g = df.groupby("segment")
    out = pd.DataFrame({"customers": g.size(), "revenue": g["lifetime_value"].sum(min_count=1)})
    out["revenue_share_pct"] = (out["revenue"] / _ltv(df).sum() * 100).round(2)
    out["customer_share_pct"] = (out["customers"] / len(df) * 100).round(2)
    order = [s for s in _segment_order(cfg) if s in out.index]
    return out.loc[order].rename_axis("segment").reset_index()


def customers_gt_threshold_tickets(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    t = cfg.support.high_ticket_threshold
    hit = df[df["support_tickets"].astype("Int64").gt(t).fillna(False)]
    hit = hit.sort_values(["support_tickets", "customer_id"], ascending=[False, True])
    return hit[["customer_id", "name", "city", "segment", "support_tickets", "last_ticket_date",
                "lifetime_value"]].reset_index(drop=True)


def revenue_by_city(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    out = df.groupby("city")["lifetime_value"].sum(min_count=1).rename("total_revenue").reset_index()
    return out.sort_values(["total_revenue", "city"], ascending=[False, True]).reset_index(drop=True)


def customer_count_by_city(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    out = df.groupby("city").size().rename("customers").reset_index()
    return out.sort_values(["customers", "city"], ascending=[False, True]).reset_index(drop=True)


def revenue_per_customer_by_city(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Revenue per customer uses only customers with a known LTV as the denominator."""
    g = df.groupby("city")
    out = pd.DataFrame({
        "customers": g.size(),
        "customers_with_ltv": g["lifetime_value"].count(),
        "total_revenue": g["lifetime_value"].sum(min_count=1),
    })
    out["revenue_per_customer"] = (out["total_revenue"] / out["customers_with_ltv"]).round(2)
    out = out.rename_axis("city").reset_index()
    return out.sort_values(["revenue_per_customer", "city"], ascending=[False, True]).reset_index(drop=True)


def at_risk_vips(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    vip = df[df["at_risk_vip"].astype(bool)]
    vip = vip.sort_values(["risk_score", "customer_id"], ascending=[False, True])
    cols = ["customer_id", "name", "city", "lifetime_value", "risk_score", "risk_band",
            "risk_pts_order_recency", "risk_pts_ticket_count", "risk_pts_ticket_recency",
            "risk_pts_low_ltv", "days_since_last_order", "support_tickets"]
    return vip[cols].reset_index(drop=True)


def ticket_heavy_low_value(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Tickets above the support threshold AND segment Bronze (Unknown is excluded)."""
    t = cfg.support.high_ticket_threshold
    hit = df[df["support_tickets"].astype("Int64").gt(t).fillna(False) & (df["segment"] == "Bronze")]
    hit = hit.sort_values(["support_tickets", "customer_id"], ascending=[False, True])
    return hit[["customer_id", "name", "city", "segment", "support_tickets", "lifetime_value",
                "tickets_per_order"]].reset_index(drop=True)


def ticket_percentiles(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    tickets = df["support_tickets"].astype("Int64").dropna().astype(float)
    rows = []
    for p in cfg.support.report_percentiles:
        value = float(tickets.quantile(p))
        rows.append({"percentile": f"P{round(p * 100)}", "tickets_at_percentile": value,
                     "customers_above": int((tickets > value).sum())})
    return pd.DataFrame(rows)


def dq_funnel(report: Dict[str, Any]) -> pd.DataFrame:
    stages = [
        ("source_records", report["total_records"]),
        ("after_rejections", report["reconciliation"]["clean"]),
        ("error_free_records", report["successful_records"]),
        ("customers_in_360", report["customers_in_360"]),
    ]
    return pd.DataFrame(stages, columns=["stage", "count"])


CUSTOMER_INSIGHTS: Dict[str, Callable[[pd.DataFrame, Settings], pd.DataFrame]] = {
    "top10_by_revenue": top10_by_revenue,
    "segment_summary": segment_summary,
    "customers_gt5_tickets": customers_gt_threshold_tickets,
    "revenue_by_city": revenue_by_city,
    "customer_count_by_city": customer_count_by_city,
    "segment_revenue_share": segment_revenue_share,
    "at_risk_vips": at_risk_vips,
    "revenue_per_customer_by_city": revenue_per_customer_by_city,
    "ticket_heavy_low_value": ticket_heavy_low_value,
    "ticket_percentiles": ticket_percentiles,
}


def build_insights(df: pd.DataFrame, report: Dict[str, Any], cfg: Settings) -> Dict[str, pd.DataFrame]:
    out = {name: fn(df, cfg) for name, fn in CUSTOMER_INSIGHTS.items()}
    out["dq_funnel"] = dq_funnel(report)
    return out
