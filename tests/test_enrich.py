"""Merge, derived fields, segments, risk and RFM."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import main
from src.enrich import (assign_segments, enrich, quantile_score, rfm_label, risk_components,
                        risk_score)
from src.merge import CORE_FIELDS, merge
from src.publish import write_customer_outputs
from src.validate import validate
from tests.conftest import crm_row, order_row, staged, ticket_row


def build(cfg, crm, orders=None, support=None) -> pd.DataFrame:
    res = validate(staged(crm, orders, support), cfg, "TEST")
    return enrich(merge(res.clean, cfg), cfg).set_index("customer_id", drop=False)


# ---------- merge ----------

def test_join_count_equals_clean_crm_and_defaults(cfg):
    crm = [crm_row("C0001"), crm_row("C0002"), crm_row("C0003"), crm_row("C0003")]  # one dup
    df = build(cfg, crm, [order_row("C0001"), order_row("C0003")],
               [ticket_row("C0001"), ticket_row("C0002")])
    assert len(df) == 3 and df.customer_id.is_unique
    # C0002: no order row -> 0 orders, 0 LTV, null date, flagged
    assert df.loc["C0002", "total_orders"] == 0 and df.loc["C0002", "lifetime_value"] == 0
    assert pd.isna(df.loc["C0002", "last_order_date"])
    assert "NO_ORDER_ROW" in df.loc["C0002", "dq_flags"]
    # C0003: no ticket row -> 0 tickets, null date, flagged
    assert df.loc["C0003", "support_tickets"] == 0 and pd.isna(df.loc["C0003", "last_ticket_date"])
    assert "NO_TICKET_ROW" in df.loc["C0003", "dq_flags"]
    assert df.loc["C0001", "dq_flags"] == [] and df.loc["C0001", "source_system"] == "crm,orders,support"
    assert df.loc["C0003", "source_system"] == "crm,orders"


def test_flags_from_all_sources_are_combined(cfg):
    df = build(cfg, [crm_row("C0001", email="bad")],
               [order_row("C0001", total_orders=None)], [ticket_row("C0001", support_tickets="0")])
    assert df.loc["C0001", "dq_flags"] == ["BAD_EMAIL", "NULL_ORDERS", "ZERO_TICKETS_WITH_DATE"]


# ---------- derived fields ----------

def test_derived_fields(cfg):
    # as_of_date is 2026-10-28
    df = build(cfg, [crm_row("C0001", registration_date="2026-10-18")],
               [order_row("C0001", total_orders="4", lifetime_value="1000", last_order_date="2026-10-21")],
               [ticket_row("C0001", support_tickets="3", last_ticket_date="2026-10-27")])
    r = df.loc["C0001"]
    assert r.avg_order_value == 250.0
    assert r.days_since_last_order == 7
    assert r.days_since_last_ticket == 1
    assert r.tenure_days == 10
    assert r.tickets_per_order == 0.75


def test_derived_null_handling(cfg):
    df = build(cfg, [crm_row("C0001"), crm_row("C0002"), crm_row("C0003")],
               [order_row("C0001", total_orders=None), order_row("C0002", lifetime_value="-5")],
               [ticket_row("C0001"), ticket_row("C0002"), ticket_row("C0003")])
    assert pd.isna(df.loc["C0001", "avg_order_value"]) and pd.isna(df.loc["C0001", "tickets_per_order"])
    assert pd.isna(df.loc["C0002", "avg_order_value"])           # LTV quarantined
    assert df.loc["C0002", "segment"] == "Unknown"
    # C0003 has no order row: 0 orders -> ratios undefined, recency unknown
    assert pd.isna(df.loc["C0003", "avg_order_value"]) and pd.isna(df.loc["C0003", "days_since_last_order"])


# ---------- segments ----------

@pytest.mark.parametrize("ltv,segment", [
    (0, "Bronze"), (49999.99, "Bronze"), (50000, "Silver"), (75000, "Silver"),
    (100000, "Silver"), (100000.01, "Gold"), (None, "Unknown"),
])
def test_segment_boundaries(cfg, ltv, segment):
    s = pd.Series([ltv], dtype="Float64")
    assert assign_segments(s, cfg).iloc[0] == segment


def test_segment_percentile_mode(cfg):
    pct_cfg = cfg.model_copy(update={"segments": cfg.segments.model_copy(update={"mode": "percentile"})})
    s = pd.Series([float(v) for v in range(1, 10)] + [None], dtype="Float64")
    out = assign_segments(s, pct_cfg).tolist()
    assert out.count("Bronze") == 3 and out.count("Silver") == 3 and out.count("Gold") == 3
    assert out[-1] == "Unknown"


# ---------- risk ----------

def test_risk_points_sum_to_score_and_bands(cfg):
    comp = pd.DataFrame({"order_recency": [1.0, 0.0, 0.5], "ticket_count": [1.0, 0.0, 0.5],
                         "ticket_recency": [1.0, 0.0, 0.5], "low_ltv": [1.0, 0.0, 0.5]})
    out = risk_score(comp, cfg)
    assert out.risk_score.tolist() == [100.0, 0.0, 50.0]
    assert out.risk_band.tolist() == ["High", "Low", "Medium"]
    pts = out[[c for c in out.columns if c.startswith("risk_pts_")]].sum(axis=1)
    assert np.allclose(pts.astype(float), out.risk_score.astype(float))
    assert out.loc[0, "risk_pts_order_recency"] == 40.0          # weight 0.40


@pytest.mark.parametrize("score,band", [(39.9, "Low"), (40.0, "Medium"), (64.9, "Medium"), (65.0, "High")])
def test_risk_band_cutoffs(cfg, score, band):
    comp = pd.DataFrame({c: [score / 100] for c in ["order_recency", "ticket_count", "ticket_recency", "low_ltv"]})
    assert risk_score(comp, cfg).risk_band.iloc[0] == band


def test_risk_missing_component_reweighted(cfg):
    comp = pd.DataFrame({"order_recency": [1.0], "ticket_count": [1.0], "ticket_recency": [1.0],
                         "low_ltv": [np.nan]})
    out = risk_score(comp, cfg)
    assert out.risk_score.iloc[0] == 100.0 and pd.isna(out.risk_pts_low_ltv.iloc[0])


def test_risk_components_rules(cfg):
    df = pd.DataFrame({
        "support_tickets": pd.array([0, 12, 6], dtype="Int64"),
        "days_since_last_order": pd.array([0, 360, 90], dtype="Int64"),
        "days_since_last_ticket": pd.array([5, 0, 45], dtype="Int64"),
        "lifetime_value": pd.array([10.0, 30.0, 20.0], dtype="Float64"),
    })
    c = risk_components(df, cfg)
    assert c.order_recency.tolist() == [0.0, 1.0, 0.5]        # capped at 180 days
    assert c.ticket_count.tolist() == [0.0, 1.0, 0.5]
    assert c.ticket_recency.tolist() == [0.0, 1.0, 0.5]       # 0 tickets -> 0 regardless of date
    assert c.low_ltv.round(4).tolist() == [round(1 - 1 / 3, 4), 0.0, round(1 - 2 / 3, 4)]


def test_at_risk_vip(cfg):
    df = build(cfg, [crm_row("C0001"), crm_row("C0002")],
               [order_row("C0001", lifetime_value="200000", last_order_date="2026-01-01"),
                order_row("C0002", lifetime_value="20000", last_order_date="2026-01-01")],
               [ticket_row("C0001", support_tickets="12", last_ticket_date="2026-10-27"),
                ticket_row("C0002", support_tickets="12", last_ticket_date="2026-10-27")])
    assert df.loc["C0001", "segment"] == "Gold" and df.loc["C0001", "risk_band"] == "High"
    assert bool(df.loc["C0001", "at_risk_vip"]) is True
    assert bool(df.loc["C0002", "at_risk_vip"]) is False     # High risk but not Gold


# ---------- RFM ----------

def test_quantile_score():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None], dtype="Float64")
    assert quantile_score(s, 5).tolist()[:10] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert quantile_score(s, 5, higher_is_better=False).tolist()[:10] == [5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    assert pd.isna(quantile_score(s, 5).iloc[10])
    ties = quantile_score(pd.Series([7.0] * 4 + [1.0]), 5)
    assert len(set(ties.iloc[:4])) == 1                       # ties share a score


@pytest.mark.parametrize("rfm,label", [
    ((5, 5, 5), "Champions"), ((3, 4, 1), "Loyal"), ((3, 1, 4), "Big Spenders"),
    ((2, 3, 1), "At Risk"), ((2, 1, 3), "At Risk"), ((1, 4, 4), "At Risk"),   # old rule order said Loyal
    ((1, 4, 1), "At Risk"), ((1, 1, 1), "Dormant"),
    ((3, 3, 3), "Needs Attention"), ((pd.NA, 3, 3), "Unscored"),
])
def test_rfm_labels(cfg, rfm, label):
    assert rfm_label(*rfm, cfg.rfm.rules, cfg.rfm.default_label, cfg.rfm.unscored_label) == label


# ---------- output contract + real sample ----------

def test_customer_360_json_contract(cfg, tmp_path):
    df = build(cfg, [crm_row("C0001")], [order_row("C0001")], [ticket_row("C0001")])
    write_customer_outputs(df.reset_index(drop=True), tmp_path)
    recs = json.loads((tmp_path / "customer_360.json").read_text(encoding="utf-8"))
    assert list(recs[0]) == CORE_FIELDS
    assert recs[0] == {"customer_id": "C0001", "name": "Test 0001", "email": "user0001@example.com",
                       "city": "Pune", "registration_date": "2024-01-01", "total_orders": 5,
                       "lifetime_value": 60000, "last_order_date": "2026-05-01",
                       "support_tickets": 2, "last_ticket_date": "2026-06-01"}


def test_real_sample_enrichment(cfg, tmp_path):
    res = validate(main.stage_sources(cfg, "TEST", tmp_path / "staged"), cfg, "TEST")
    df = enrich(merge(res.clean, cfg), cfg).set_index("customer_id", drop=False)
    assert len(df) == 500
    assert df.segment.value_counts().to_dict() == {"Gold": 318, "Silver": 101, "Bronze": 80, "Unknown": 1}
    assert df.loc["C0420", "segment"] == "Unknown" and pd.isna(df.loc["C0420", "lifetime_value"])
    assert pd.isna(df.loc["C0310", "total_orders"]) and pd.isna(df.loc["C0310", "avg_order_value"])
    assert df.loc["C0450", "support_tickets"] == 0 and "NO_TICKET_ROW" in df.loc["C0450", "dq_flags"]
    assert df.lifetime_value.sum() == df[df.customer_id != "C0420"].lifetime_value.sum()
