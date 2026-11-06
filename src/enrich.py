"""Derived metrics, segments, rule-based risk score and RFM, all relative to as_of_date."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import RfmBounds, RfmRule, Settings

log = logging.getLogger("enrich")

RISK_COMPONENTS: List[str] = ["order_recency", "ticket_count", "ticket_recency", "low_ltv"]
DERIVED_COLUMNS: List[str] = [
    "segment", "avg_order_value", "days_since_last_order", "days_since_last_ticket",
    "tenure_days", "tickets_per_order", "risk_score", "risk_band",
    *[f"risk_pts_{c}" for c in RISK_COMPONENTS], "at_risk_vip",
    "r_score", "f_score", "m_score", "rfm_score", "rfm_label",
]


def _num(s: pd.Series) -> pd.Series:
    """Nullable/any numeric series -> float64 with NaN for missing."""
    return pd.Series(s.to_numpy(dtype=float, na_value=np.nan), index=s.index)


def _days_since(dates: pd.Series, as_of: pd.Timestamp) -> pd.Series:
    return (as_of - dates).dt.days.astype("Int64")


def _ratio(num: pd.Series, den: pd.Series, decimals: int) -> pd.Series:
    n, d = _num(num), _num(den)
    out = (n / d.where(d > 0)).round(decimals)
    return out.astype("Float64")


def add_derived(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["avg_order_value"] = _ratio(df["lifetime_value"], df["total_orders"], 2)
    df["days_since_last_order"] = _days_since(df["last_order_date"], as_of)
    df["days_since_last_ticket"] = _days_since(df["last_ticket_date"], as_of)
    df["tenure_days"] = _days_since(df["registration_date"], as_of)
    df["tickets_per_order"] = _ratio(df["support_tickets"], df["total_orders"], 4)
    return df


def assign_segments(ltv: pd.Series, cfg: Settings) -> pd.Series:
    """Gold > gold_min; Silver >= silver_min (and <= gold_min); Bronze below; null -> null_label."""
    seg = cfg.segments
    v = _num(ltv)
    if seg.mode == "percentile":
        lo, hi = v.dropna().quantile(seg.percentile_cuts).tolist()
    else:
        lo, hi = seg.silver_min_inclusive, seg.gold_min_exclusive
    out = pd.Series(seg.null_label, index=ltv.index, dtype=object)
    known = v.notna()
    out[known & (v < lo)] = "Bronze"
    out[known & (v >= lo) & (v <= hi)] = "Silver"
    out[known & (v > hi)] = "Gold"
    return out


def risk_components(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Each component in [0, 1] (1 = riskier); NaN when it cannot be computed."""
    r = cfg.risk
    tickets = _num(df["support_tickets"])
    order_days = _num(df["days_since_last_order"]).clip(lower=0)
    ticket_days = _num(df["days_since_last_ticket"]).clip(lower=0)
    ticket_recency = 1 - ticket_days.clip(upper=r.ticket_recency_window_days) / r.ticket_recency_window_days
    ticket_recency = ticket_recency.where(tickets != 0, 0.0).where(tickets.notna())
    return pd.DataFrame({
        "order_recency": order_days.clip(upper=r.order_recency_cap_days) / r.order_recency_cap_days,
        "ticket_count": tickets.clip(lower=0, upper=r.ticket_count_cap) / r.ticket_count_cap,
        "ticket_recency": ticket_recency,
        "low_ltv": 1 - _num(df["lifetime_value"]).rank(pct=True),
    }, index=df.index)


def risk_score(comp: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Weighted blend scaled to 0-100. Missing components are dropped and the remaining
    weights re-normalised, so a customer is never scored on an invented value."""
    w = pd.Series(cfg.risk.weights.model_dump())[RISK_COMPONENTS]
    avail_w = comp.notna().mul(w, axis=1).sum(axis=1)
    pts = comp.mul(w, axis=1).mul(100).div(avail_w.where(avail_w > 0), axis=0)
    score = pts.sum(axis=1, min_count=1).round(1)
    band = pd.Series(None, index=comp.index, dtype=object)
    band[score.notna()] = "Low"
    band[score >= cfg.risk.band_medium_min] = "Medium"
    band[score >= cfg.risk.band_high_min] = "High"
    out = pd.DataFrame({"risk_score": score.astype("Float64"), "risk_band": band})
    for c in RISK_COMPONENTS:
        out[f"risk_pts_{c}"] = pts[c].round(2).astype("Float64")
    return out


def quantile_score(values: pd.Series, bins: int, higher_is_better: bool = True) -> pd.Series:
    """1..bins by percentile rank; ties share a score; NaN stays null."""
    pct = _num(values).rank(method="average", pct=True, ascending=higher_is_better)
    return np.ceil(pct * bins).clip(1, bins).astype("Int64")


def _within(score: int, lo: Optional[int], hi: Optional[int]) -> bool:
    return (lo is None or score >= lo) and (hi is None or score <= hi)


def _all_bounds(r: int, f: int, m: int, b: RfmBounds) -> bool:
    return (_within(r, b.r_min, b.r_max) and _within(f, b.f_min, b.f_max)
            and _within(m, b.m_min, b.m_max))


def _any_bound(r: int, f: int, m: int, b: RfmBounds) -> bool:
    """True if any single specified bound in `b` is met (OR semantics)."""
    checks = [(r, b.r_min, b.r_max), (f, b.f_min, b.f_max), (m, b.m_min, b.m_max)]
    tests = []
    for v, lo, hi in checks:
        if lo is not None:
            tests.append(v >= lo)
        if hi is not None:
            tests.append(v <= hi)
    return any(tests)


def rfm_label(r, f, m, rules: List[RfmRule], default: str, unscored: str) -> str:
    if any(pd.isna(x) for x in (r, f, m)):
        return unscored
    for rule in rules:
        if _all_bounds(r, f, m, rule.all) and (rule.any is None or _any_bound(r, f, m, rule.any)):
            return rule.label
    return default


def add_rfm(df: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    df = df.copy()
    bins = cfg.rfm.bins
    df["r_score"] = quantile_score(df["days_since_last_order"], bins, higher_is_better=False)
    df["f_score"] = quantile_score(df["total_orders"], bins)
    df["m_score"] = quantile_score(df["lifetime_value"], bins)
    scored = df[["r_score", "f_score", "m_score"]].notna().all(axis=1)
    df["rfm_score"] = None
    df.loc[scored, "rfm_score"] = (df.loc[scored, "r_score"].astype(str) + df.loc[scored, "f_score"].astype(str)
                                   + df.loc[scored, "m_score"].astype(str))
    df["rfm_label"] = [rfm_label(r, f, m, cfg.rfm.rules, cfg.rfm.default_label, cfg.rfm.unscored_label)
                       for r, f, m in zip(df["r_score"], df["f_score"], df["m_score"])]
    return df


def enrich(customers: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    as_of = pd.Timestamp(cfg.as_of_date)
    df = add_derived(customers, as_of)
    df["segment"] = assign_segments(df["lifetime_value"], cfg)
    df = pd.concat([df, risk_score(risk_components(df, cfg), cfg)], axis=1)
    df["at_risk_vip"] = (df["segment"] == "Gold") & (df["risk_band"] == "High")
    df = add_rfm(df, cfg)
    lineage = ["dq_flags", "source_system", "run_id", "ingested_at"]
    core = [c for c in customers.columns if c not in lineage]
    df = df[core + DERIVED_COLUMNS + lineage]
    log.info("enriched | customers=%d segments=%s risk_bands=%s at_risk_vip=%d",
             len(df), df["segment"].value_counts().to_dict(),
             df["risk_band"].value_counts().to_dict(), int(df["at_risk_vip"].sum()))
    return df
