"""End-to-end DQ checks: the real sample, and an abort on a fake high-reject dataset."""
from __future__ import annotations

import csv
import json
import shutil

import main
from src.config import load_config
from src.validate import validate
from tests.conftest import PROJECT_ROOT


def test_real_sample_dq_counts(cfg, tmp_path):
    frames = main.stage_sources(cfg, "TEST", tmp_path / "staged")
    res = validate(frames, cfg, "TEST")
    r = res.report
    expected = {"total_records": 1501, "duplicates": 1, "invalid_emails": 3, "missing_fields": 1,
                "successful_records": 1494, "invalid_ids": 1, "negative_revenue": 1, "orphans": 0,
                "rejected_unique_records": 2, "flagged_kept_records": 5, "warning_records": 35,
                "customers_in_360": 500}
    assert {k: r[k] for k in expected} == expected
    assert r["reconciliation"] == {"input": 1501, "clean": 1499, "rejected": 2, "balanced": True}
    assert {s: (v["input"], v["clean"], v["rejected"]) for s, v in r["by_source"].items()} == {
        "crm_customers.json": (501, 500, 1), "orders.csv": (500, 500, 0),
        "support_tickets.xml": (500, 499, 1)}


def _write_fake_dataset(root, n=20, bad=4):
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    ids = [f"C{i:04d}" for i in range(1, n + 1)]
    crm = [{"customer_id": ("BAD" + c[1:]) if i < bad else c, "name": f"N{i}",
            "email": f"u{i}@x.com", "city": "Pune", "registration_date": "2024-01-01"}
           for i, c in enumerate(ids)]
    (raw / "crm_customers.json").write_text(json.dumps(crm), encoding="utf-8")
    with (raw / "orders.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cust_id", "total_orders", "lifetime_value", "last_order_date"])
        w.writerows([c, 1, 1000, "2026-01-01"] for c in ids)
    xml = "".join(f"<customer><customerId>{c}</customerId><tickets>1</tickets>"
                  f"<lastTicketDate>2026-01-01</lastTicketDate></customer>" for c in ids)
    (raw / "support_tickets.xml").write_text(f"<customers>{xml}</customers>", encoding="utf-8")


def test_abort_when_reject_rate_exceeded(tmp_path):
    shutil.copy(PROJECT_ROOT / "config.json", tmp_path / "config.json")
    _write_fake_dataset(tmp_path)          # 4 bad CRM ids -> 4 + 4 orphan orders + 4 orphan tickets

    exit_code = main.main(["--config", str(tmp_path / "config.json")])

    assert exit_code == 3
    cfg = load_config(tmp_path / "config.json")
    run_dirs = [p for p in cfg.output_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    report = json.loads((run_dirs[0] / "dq_report.json").read_text(encoding="utf-8"))
    assert report["total_records"] == 60 and report["rejected_unique_records"] == 12
    assert report["reject_rate"] == 0.2
    assert not (cfg.output_dir / cfg.paths.latest_dir_name).exists()   # nothing published
