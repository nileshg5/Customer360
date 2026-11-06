"""Unit tests: one or more per DQ rule, plus report and reconciliation."""
from __future__ import annotations

import re

import pytest

from src.validate import (RejectRateExceeded, check_reject_rate, is_valid_email, is_valid_id,
                          parse_count, validate)
from tests.conftest import crm_row, order_row, staged, ticket_row

REQUIRED_KEYS = ["total_records", "duplicates", "invalid_emails", "missing_fields", "successful_records"]


def run(cfg, crm, orders=None, support=None):
    return validate(staged(crm, orders, support), cfg, "TEST")


def flags(df, cid):
    return df.loc[df.customer_id == cid, "dq_flags"].iloc[0]


# ---------- field validators ----------

@pytest.mark.parametrize("email,ok", [
    ("user1@example.com", True),
    ("User1@Example.COM", True),        # uppercase is valid
    ("user1example.com", False),        # no @
    ("user 1@example.com", False),      # space
    ("user1@example", False),           # no domain dot
    ("a@b@c.com", False),
    (None, False),
])
def test_email_validator(cfg, email, ok):
    assert is_valid_email(email, re.compile(cfg.validation.email_regex)) is ok


@pytest.mark.parametrize("cid,ok", [
    ("C0001", True), (None, False), ("", False), ("X0001", False), ("c0001", False),
    ("C001", False), ("C00001", False), ("C000A", False),
])
def test_id_validator(cfg, cid, ok):
    assert is_valid_id(cid, re.compile(cfg.validation.id_regex)) is ok


@pytest.mark.parametrize("text,expected", [("5", 5), ("5.0", 5), ("0", 0)])
def test_parse_count_accepts_whole_numbers(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize("text", ["5.5", "-1", "abc", "nan", "inf"])
def test_parse_count_rejects_bad_numbers(text):
    with pytest.raises(ValueError):
        parse_count(text)


# ---------- ID rules ----------

def test_empty_and_bad_id_rejected(cfg):
    res = run(cfg, [crm_row("C0001"), crm_row(None), crm_row("X0002")])
    assert list(res.clean["crm"].customer_id) == ["C0001"]
    assert sorted(res.rejected.rule_codes) == ["BAD_ID", "EMPTY_ID"]
    assert res.report["invalid_ids"] == 2


def test_dedupe_keeps_first(cfg):
    first = crm_row("C0001", name="First")
    second = crm_row("C0001", name="Second")
    res = run(cfg, [first, second])
    clean = res.clean["crm"]
    assert len(clean) == 1 and clean.name.iloc[0] == "First"
    assert flags(clean, "C0001") == []
    assert res.rejected[["source_row", "rule_codes", "action"]].to_dict("records") == [
        {"source_row": 2, "rule_codes": "DUP_ID", "action": "rejected"}]
    assert res.report["duplicates"] == 1


def test_dup_email_flags_but_keeps(cfg):
    res = run(cfg, [crm_row("C0001", email="same@x.com"), crm_row("C0002", email="SAME@x.com")])
    assert len(res.clean["crm"]) == 2
    assert all("DUP_EMAIL" in f for f in res.clean["crm"].dq_flags)
    assert res.rejected.empty


# ---------- CRM field rules ----------

def test_bad_email_nulled_and_flagged(cfg):
    res = run(cfg, [crm_row("C0001", email="user1example.com"), crm_row("C0002", email="Up@Case.com")])
    clean = res.clean["crm"].set_index("customer_id")
    assert clean.loc["C0001", "email"] is None
    assert clean.loc["C0001", "dq_flags"] == ["BAD_EMAIL"]
    assert clean.loc["C0002", "email"] == "up@case.com"     # valid emails are lowercased
    assert res.report["invalid_emails"] == 1 and res.rejected.empty


@pytest.mark.parametrize("field", ["name", "city", "registration_date"])
def test_missing_mandatory_field_rejected(cfg, field):
    res = run(cfg, [crm_row("C0001", **{field: None})])
    assert res.clean["crm"].empty
    assert res.rejected.rule_codes.tolist() == ["MISSING_FIELD"]
    assert res.report["missing_fields"] == 1


def test_bad_dates(cfg):
    res = run(cfg, [crm_row("C0001", registration_date="2024-13-45"),
                    crm_row("C0002", registration_date="2030-01-01")])
    clean = res.clean["crm"].set_index("customer_id")
    assert clean.loc["C0001", "dq_flags"] == ["BAD_DATE_UNPARSEABLE"]
    assert clean.loc["C0001", "registration_date"] != clean.loc["C0001", "registration_date"]  # NaT
    assert clean.loc["C0002", "dq_flags"] == ["BAD_DATE_FUTURE"]
    assert res.rejected.empty


# ---------- order / ticket rules ----------

def test_null_orders_flagged_and_kept(cfg):
    res = run(cfg, [crm_row("C0001")], [order_row("C0001", total_orders=None)])
    o = res.clean["orders"]
    assert len(o) == 1 and o.total_orders.isna().all()
    assert flags(o, "C0001") == ["NULL_ORDERS"]
    assert res.report["missing_fields"] == 1


def test_negative_revenue_quarantined_and_excluded(cfg):
    res = run(cfg, [crm_row("C0001"), crm_row("C0002")],
              [order_row("C0001", lifetime_value="-5000"), order_row("C0002", lifetime_value="70000")])
    o = res.clean["orders"].set_index("customer_id")
    assert o.loc["C0001", "lifetime_value"] is None or str(o.loc["C0001", "lifetime_value"]) == "<NA>"
    assert o.loc["C0001", "total_orders"] == 5                  # rest of the row kept
    assert o.loc["C0001", "dq_flags"] == ["NEG_REVENUE"]
    assert res.clean["orders"].lifetime_value.sum() == 70000    # excluded from revenue
    q = res.rejected.to_dict("records")
    assert len(q) == 1 and q[0]["action"] == "quarantined_value" and q[0]["rule_codes"] == "NEG_REVENUE"
    assert '"-5000"' in q[0]["raw_record"]
    assert res.report["negative_revenue"] == 1 and res.report["rejected_unique_records"] == 0


@pytest.mark.parametrize("field,value", [("total_orders", "abc"), ("total_orders", "2.5"),
                                         ("total_orders", "-3"), ("lifetime_value", "12k")])
def test_bad_number_nulled_and_flagged(cfg, field, value):
    res = run(cfg, [crm_row("C0001")], [order_row("C0001", **{field: value})])
    o = res.clean["orders"]
    assert o[field].isna().all() and flags(o, "C0001") == ["BAD_NUMBER"]


def test_orphans_rejected(cfg):
    res = run(cfg, [crm_row("C0001")],
              [order_row("C0001"), order_row("C0999")], [ticket_row("C0001"), ticket_row("C0998")])
    assert list(res.clean["orders"].customer_id) == ["C0001"]
    assert list(res.clean["support"].customer_id) == ["C0001"]
    assert sorted(res.rejected.customer_id) == ["C0998", "C0999"]
    assert set(res.rejected.rule_codes) == {"ORPHAN"}
    assert res.report["orphans"] == 2


def test_orphan_when_crm_row_rejected(cfg):
    res = run(cfg, [crm_row("C0001", city=None)], [order_row("C0001")])
    assert res.rejected.rule_codes.tolist() == ["MISSING_FIELD", "ORPHAN"]


def test_cross_date_flag_only(cfg):
    res = run(cfg, [crm_row("C0001", registration_date="2026-06-01")],
              [order_row("C0001", last_order_date="2026-05-01")])
    assert flags(res.clean["orders"], "C0001") == ["CROSS_DATE"] and res.rejected.empty


def test_zero_tickets_with_date_flagged(cfg):
    res = run(cfg, [crm_row("C0001")], support=[ticket_row("C0001", support_tickets="0")])
    assert flags(res.clean["support"], "C0001") == ["ZERO_TICKETS_WITH_DATE"]


def test_empty_id_ticket_rejected(cfg):
    res = run(cfg, [crm_row("C0001")], support=[ticket_row(None, support_tickets="8")])
    assert res.clean["support"].empty
    assert res.rejected.to_dict("records")[0]["rule_codes"] == "EMPTY_ID"


# ---------- report and reconciliation ----------

def test_report_counts_unique_records_and_reconciles(cfg):
    crm = [crm_row("C0001"), crm_row("C0002", email="bad", registration_date="2030-01-01"),
           crm_row("C0002"), crm_row("X1")]
    res = run(cfg, crm, [order_row("C0001"), order_row("C0002", total_orders=None)])
    r = res.report
    assert list(r)[2:7] == REQUIRED_KEYS                     # required keys at top level, in order
    assert r["total_records"] == 6
    # issues: C0002 (2 rules, one record), dup C0002, X1, order C0002 -> 4 unique records
    assert r["successful_records"] == 2
    assert r["rejected_unique_records"] == 2 and r["flagged_kept_records"] == 2
    rec = r["reconciliation"]
    assert rec == {"input": 6, "clean": 4, "rejected": 2, "balanced": True}
    assert all(s["balanced"] for s in r["by_source"].values())


def test_reject_rate_threshold(cfg):
    check_reject_rate({"reject_rate": 0.05, "max_reject_rate": 0.05})      # equal is allowed
    with pytest.raises(RejectRateExceeded):
        check_reject_rate({"reject_rate": 0.0501, "max_reject_rate": 0.05})


# ---------- NULL_TICKETS / NULL_DATE / severity ----------

def test_null_tickets_error_kept(cfg):
    res = run(cfg, [crm_row("C0001")], support=[ticket_row("C0001", support_tickets=None)])
    t = res.clean["support"]
    assert len(t) == 1 and t.support_tickets.isna().all()
    assert flags(t, "C0001") == ["NULL_TICKETS"]
    assert res.report["missing_fields"] == 1 and res.report["successful_records"] == 1


def test_null_date_is_warning(cfg):
    res = run(cfg, [crm_row("C0001")], [order_row("C0001", last_order_date=None)],
              [ticket_row("C0001", last_ticket_date=None)])
    assert flags(res.clean["orders"], "C0001") == ["NULL_DATE"]
    assert flags(res.clean["support"], "C0001") == ["NULL_DATE"]
    r = res.report
    assert r["successful_records"] == 3 and r["warning_records"] == 2 and r["flagged_kept_records"] == 0


def test_warnings_do_not_reduce_successful_records(cfg):
    res = run(cfg, [crm_row("C0001"), crm_row("C0002", email="bad")],
              support=[ticket_row("C0001", support_tickets="0"), ticket_row("C0002")])
    r = res.report
    assert r["total_records"] == 4
    assert r["successful_records"] == 3          # only the BAD_EMAIL row is an error
    assert r["flagged_kept_records"] == 1 and r["warning_records"] == 1


def test_per_source_reject_rate(cfg):
    res = run(cfg, [crm_row("C0001"), crm_row("X1"), crm_row("C0002"), crm_row("C0003")])
    assert res.report["by_source"]["crm_customers.json"]["reject_rate"] == 0.25
    assert res.report["by_source"]["orders.csv"]["reject_rate"] == 0.0
