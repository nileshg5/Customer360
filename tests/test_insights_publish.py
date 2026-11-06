"""Insight tables on a fixed fixture, atomic publish, SQLite and idempotency."""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

import main
import src.publish as publish
from src import insights as ins
from src.config import load_config
from src.enrich import enrich
from src.merge import merge
from src.validate import validate
from tests.conftest import PROJECT_ROOT, crm_row, order_row, staged, ticket_row
from tests.helpers import raw_snapshot, snapshot

# id, city, ltv, tickets
FIXTURE = [
    ("C0001", "Pune", "200000", "6"),
    ("C0002", "Pune", "200000", "7"),      # ties C0001 on revenue
    ("C0003", "Delhi", "60000", "2"),
    ("C0004", "Delhi", "10000", "9"),      # Bronze and ticket-heavy
    ("C0005", "Mumbai", "-5", "8"),        # quarantined LTV -> Unknown, ticket-heavy
    ("C0006", "Mumbai", "40000", "5"),     # exactly at threshold: not "more than 5"
]


@pytest.fixture()
def fx(cfg):
    res = validate(staged([crm_row(c, city=city) for c, city, _, _ in FIXTURE],
                          [order_row(c, lifetime_value=ltv) for c, _, ltv, _ in FIXTURE],
                          [ticket_row(c, support_tickets=t) for c, _, _, t in FIXTURE]), cfg, "TEST")
    df = enrich(merge(res.clean, cfg), cfg)
    return df, res


def test_top10_ranking_ties_and_nulls(cfg):
    ids = [f"C{i:04d}" for i in range(1, 13)]
    ltvs = ["5000", "9000", "9000", "1000", "8000", "7000", "6000", "4000", "3000", "2000", "1500", "-1"]
    res = validate(staged([crm_row(c) for c in ids], [order_row(c, lifetime_value=v) for c, v in zip(ids, ltvs)]),
                   cfg, "TEST")
    top = ins.top10_by_revenue(enrich(merge(res.clean, cfg), cfg), cfg)
    assert top["rank"].tolist() == list(range(1, 11))
    assert top.customer_id.tolist()[:3] == ["C0002", "C0003", "C0005"]   # 9000 tie -> by id
    assert "C0012" not in top.customer_id.tolist()                       # null LTV excluded
    assert top.lifetime_value.tolist() == sorted(top.lifetime_value.tolist(), reverse=True)


def test_segment_summary_and_share(fx, cfg):
    df, _ = fx
    s = ins.segment_summary(df, cfg).set_index("segment")
    assert s.index.tolist() == ["Gold", "Silver", "Bronze", "Unknown"]
    assert s.customers.tolist() == [2, 1, 2, 1]
    assert s.loc["Gold", "total_revenue"] == 400000 and s.loc["Bronze", "total_revenue"] == 50000
    assert pd.isna(s.loc["Unknown", "total_revenue"])
    share = ins.segment_revenue_share(df, cfg).set_index("segment")
    assert share.revenue_share_pct.sum() == pytest.approx(100, abs=0.05)
    assert share.loc["Gold", "revenue_share_pct"] == pytest.approx(400000 / 510000 * 100, abs=0.01)


def test_more_than_threshold_tickets(fx, cfg):
    df, _ = fx
    out = ins.customers_gt_threshold_tickets(df, cfg)
    assert out.customer_id.tolist() == ["C0004", "C0005", "C0002", "C0001"]   # by tickets desc
    assert "C0006" not in out.customer_id.tolist()


def test_city_tables(fx, cfg):
    df, _ = fx
    rev = ins.revenue_by_city(df, cfg)
    assert rev.values.tolist() == [["Pune", 400000.0], ["Delhi", 70000.0], ["Mumbai", 40000.0]]
    cnt = ins.customer_count_by_city(df, cfg)
    assert cnt.values.tolist() == [["Delhi", 2], ["Mumbai", 2], ["Pune", 2]]   # tie -> by city
    rpc = ins.revenue_per_customer_by_city(df, cfg).set_index("city")
    assert rpc.loc["Mumbai", "customers"] == 2 and rpc.loc["Mumbai", "customers_with_ltv"] == 1
    assert rpc.loc["Mumbai", "revenue_per_customer"] == 40000


def test_ticket_heavy_low_value_excludes_unknown(fx, cfg):
    df, _ = fx
    assert ins.ticket_heavy_low_value(df, cfg).customer_id.tolist() == ["C0004"]


def test_at_risk_vips_matches_flag(fx, cfg):
    df, _ = fx
    out = ins.at_risk_vips(df, cfg)
    assert set(out.customer_id) == set(df[df.at_risk_vip].customer_id)
    assert out.risk_score.is_monotonic_decreasing


def test_ticket_percentiles_and_funnel(fx, cfg):
    df, res = fx
    p = ins.ticket_percentiles(df, cfg).set_index("percentile")
    assert p.index.tolist() == ["P25", "P50", "P75", "P90"]
    assert p.loc["P50", "tickets_at_percentile"] == 6.5      # median of 2,5,6,7,8,9
    assert p.loc["P50", "customers_above"] == 3
    f = ins.dq_funnel(res.report)
    assert f.stage.tolist() == ["source_records", "after_rejections", "error_free_records",
                                "customers_in_360"]
    assert f["count"].tolist() == [18, 18, 17, 6]


# ---------- SQLite ----------

def test_sqlite_tables(fx, tmp_path):
    df, res = fx
    path = publish.write_sqlite(df, res.rejected, tmp_path)
    con = sqlite3.connect(path)
    cols = {r[1]: r[2] for r in con.execute("PRAGMA table_info(rejected_records)")}
    assert "raw_record" not in cols and cols["source_row"] == "INTEGER"
    assert con.execute("SELECT COUNT(*) FROM customer_360").fetchone()[0] == 6
    flags = con.execute("SELECT dq_flags FROM customer_360 WHERE customer_id='C0005'").fetchone()[0]
    assert json.loads(flags) == ["NEG_REVENUE"]
    assert con.execute("SELECT SUM(lifetime_value) FROM customer_360").fetchone()[0] == 510000
    con.close()


# ---------- atomic publish ----------

def _run_dir(tmp_path, name, content):
    d = tmp_path / name
    (d / "insights").mkdir(parents=True)
    (d / "customer_360.json").write_text(content)
    (d / "insights" / "a.csv").write_text(content)
    return d


def _leftovers(tmp_path):
    return [p.name for p in tmp_path.iterdir() if p.name.startswith(".latest")]


def test_publish_replaces_latest(tmp_path):
    latest = tmp_path / "latest"
    publish.publish_latest(_run_dir(tmp_path, "r1", "one"), latest)
    publish.publish_latest(_run_dir(tmp_path, "r2", "two"), latest)
    assert (latest / "customer_360.json").read_text() == "two"
    assert _leftovers(tmp_path) == []


def test_failure_during_copy_leaves_latest_unchanged(tmp_path, monkeypatch):
    latest = tmp_path / "latest"
    publish.publish_latest(_run_dir(tmp_path, "r1", "one"), latest)
    before = raw_snapshot(latest)

    real_copytree = shutil.copytree
    def half_copy(src, dst, *a, **k):
        result = real_copytree(src, dst, *a, **k)
        dst = Path(dst)
        if dst.name.startswith(".latest.tmp-"):          # top-level call only, not sub-dirs
            (dst / "customer_360.json").write_text("partial")
            raise OSError("disk full")
        return result
    monkeypatch.setattr(publish.shutil, "copytree", half_copy)

    with pytest.raises(OSError, match="disk full"):
        publish.publish_latest(_run_dir(tmp_path, "r2", "two"), latest)
    assert raw_snapshot(latest) == before and _leftovers(tmp_path) == []


def test_failure_during_swap_restores_latest(tmp_path, monkeypatch):
    latest = tmp_path / "latest"
    publish.publish_latest(_run_dir(tmp_path, "r1", "one"), latest)
    before = raw_snapshot(latest)

    real_replace = publish.os.replace
    def failing_replace(src, dst):
        if ".tmp-" in str(src):
            raise OSError("swap failed")
        return real_replace(src, dst)
    monkeypatch.setattr(publish.os, "replace", failing_replace)

    with pytest.raises(OSError, match="swap failed"):
        publish.publish_latest(_run_dir(tmp_path, "r2", "two"), latest)
    assert raw_snapshot(latest) == before and _leftovers(tmp_path) == []


# ---------- end-to-end: idempotency and run_id clash ----------

@pytest.fixture()
def project_copy(tmp_path):
    shutil.copy(PROJECT_ROOT / "config.json", tmp_path / "config.json")
    shutil.copytree(PROJECT_ROOT / "data" / "raw", tmp_path / "data" / "raw")
    return tmp_path


def test_idempotent_runs(project_copy, monkeypatch):
    cfg_path = str(project_copy / "config.json")
    for run_id in ("RUN_A", "RUN_B"):
        monkeypatch.setattr(main, "new_run_id", lambda r=run_id: r)
        assert main.main(["--config", cfg_path, "--as-of", "2026-10-28"]) == 0
    out = load_config(project_copy / "config.json").output_dir
    a, b = snapshot(out / "RUN_A"), snapshot(out / "RUN_B")
    assert len(a) == 20 and a == b
    assert snapshot(out / "latest") == b
    manifest = json.loads((out / "latest" / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "RUN_B" and manifest["status"] == "success"
    assert manifest["stage_rows"]["insights"]["customers_gt5_tickets"] == 268


def test_existing_run_dir_is_not_overwritten(project_copy, monkeypatch):
    monkeypatch.setattr(main, "new_run_id", lambda: "SAME")
    cfg_path = str(project_copy / "config.json")
    assert main.main(["--config", cfg_path]) == 0
    assert main.main(["--config", cfg_path]) == 2
