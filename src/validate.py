"""Data-quality rules, quarantine and the DQ report.

Each rule produces an Issue (rule code + affected field) on a source row. Every rule has a
severity (error / warning) used for counting, and an action that decides what happens to the row:
  reject               -> whole row goes to rejected_records.csv, excluded from clean
  flag                 -> row kept, code added to dq_flags
  null_and_flag        -> row kept, the affected field set to null, code added to dq_flags
  null_and_quarantine  -> as null_and_flag, plus the original value is written to
                          rejected_records.csv with action 'quarantined_value'
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Pattern, Set

import pandas as pd

from src.config import Settings

log = logging.getLogger("validate")

# DQ report key -> rule codes it counts (a row counts once even if it hits several of them).
REPORT_KEYS: Dict[str, List[str]] = {
    "duplicates": ["DUP_ID"],
    "invalid_emails": ["BAD_EMAIL"],
    "missing_fields": ["MISSING_FIELD", "NULL_ORDERS", "NULL_TICKETS"],
    "invalid_ids": ["EMPTY_ID", "BAD_ID"],
    "negative_revenue": ["NEG_REVENUE"],
    "orphans": ["ORPHAN"],
}
DATE_FIELDS = {"crm": ["registration_date"], "orders": ["last_order_date"], "support": ["last_ticket_date"]}
INT_FIELDS = {"crm": [], "orders": ["total_orders"], "support": ["support_tickets"]}
FLOAT_FIELDS = {"crm": [], "orders": ["lifetime_value"], "support": []}
REJECTED_COLUMNS = ["source_file", "source_row", "customer_id", "rule_codes", "action", "raw_record"]


class RejectRateExceeded(RuntimeError):
    """Raised when the share of rejected rows exceeds validation.max_reject_rate."""


@dataclass
class Issue:
    rule: str
    field: Optional[str] = None


@dataclass
class Row:
    """One staged source row being validated."""
    raw: Dict[str, Any]
    typed: Dict[str, Any]
    issues: List[Issue] = field(default_factory=list)

    @property
    def cid(self) -> Optional[str]:
        return self.raw.get("customer_id")

    def add(self, rule: str, fld: Optional[str] = None) -> None:
        self.issues.append(Issue(rule, fld))

    def has(self, *rules: str) -> bool:
        return any(i.rule in rules for i in self.issues)

    def codes(self) -> List[str]:
        return list(dict.fromkeys(i.rule for i in self.issues))


@dataclass
class ValidationResult:
    clean: Dict[str, pd.DataFrame]
    rejected: pd.DataFrame
    report: Dict[str, Any]


# ---------- field-level validators (pure, unit-tested) ----------

def is_valid_id(value: Optional[str], id_re: Pattern[str]) -> bool:
    return value is not None and id_re.fullmatch(value) is not None


def is_valid_email(value: Optional[str], email_re: Pattern[str]) -> bool:
    return value is not None and email_re.fullmatch(value) is not None


def parse_date(text: str, fmt: str) -> date:
    return datetime.strptime(text, fmt).date()


def parse_count(text: str) -> int:
    """Non-negative whole number ('5' or '5.0'); anything else raises ValueError."""
    num = float(text)
    if not math.isfinite(num) or not num.is_integer() or num < 0:
        raise ValueError(f"not a non-negative whole number: {text!r}")
    return int(num)


def parse_amount(text: str) -> float:
    num = float(text)
    if not math.isfinite(num):
        raise ValueError(f"not a finite number: {text!r}")
    return num


# ---------- rule passes ----------

def _check_ids(rows: List[Row], id_re: Pattern[str]) -> None:
    """EMPTY_ID / BAD_ID, then DUP_ID on valid IDs (first occurrence wins)."""
    seen: Set[str] = set()
    for r in rows:
        if r.cid is None:
            r.add("EMPTY_ID", "customer_id")
        elif not is_valid_id(r.cid, id_re):
            r.add("BAD_ID", "customer_id")
        elif r.cid in seen:
            r.add("DUP_ID", "customer_id")
        else:
            seen.add(r.cid)


def _check_dates(rows: List[Row], system: str, cfg: Settings) -> None:
    for r in rows:
        for f in DATE_FIELDS[system]:
            text = r.raw.get(f)
            if text is None:
                if system != "crm":          # CRM registration_date is covered by MISSING_FIELD
                    r.add("NULL_DATE", f)
                continue
            try:
                r.typed[f] = parse_date(text, cfg.validation.date_format)
            except ValueError:
                r.add("BAD_DATE_UNPARSEABLE", f)
                continue
            if r.typed[f] > cfg.as_of_date:
                r.add("BAD_DATE_FUTURE", f)


def _check_numbers(rows: List[Row], system: str) -> None:
    for r in rows:
        for f in INT_FIELDS[system] + FLOAT_FIELDS[system]:
            text = r.raw.get(f)
            if text is None:
                r.add("NULL_ORDERS" if system == "orders" else "NULL_TICKETS", f)
                continue
            try:
                r.typed[f] = parse_count(text) if f in INT_FIELDS[system] else parse_amount(text)
            except ValueError:
                r.add("BAD_NUMBER", f)
                continue
            if f == "lifetime_value" and r.typed[f] < 0:
                r.add("NEG_REVENUE", f)


def _check_crm(rows: List[Row], cfg: Settings) -> None:
    email_re = re.compile(cfg.validation.email_regex)
    for r in rows:
        for f in cfg.validation.mandatory_fields:
            if r.raw.get(f) is None:
                r.add("MISSING_FIELD", f)
        if is_valid_email(r.raw.get("email"), email_re):
            r.typed["email"] = r.raw["email"].lower()
        else:
            r.add("BAD_EMAIL", "email")
    # DUP_EMAIL: one email used by more than one distinct, valid customer_id.
    ids_by_email: Dict[str, Set[str]] = {}
    for r in rows:
        if r.typed.get("email") and not r.has("EMPTY_ID", "BAD_ID"):
            ids_by_email.setdefault(r.typed["email"], set()).add(r.cid)
    shared = {e for e, ids in ids_by_email.items() if len(ids) > 1}
    for r in rows:
        if r.typed.get("email") in shared:
            r.add("DUP_EMAIL", "email")


def _check_links(rows: List[Row], system: str, crm_reg: Dict[str, Optional[date]]) -> None:
    """ORPHAN against surviving CRM ids; CROSS_DATE against registration date."""
    date_field = DATE_FIELDS[system][0]
    for r in rows:
        if r.has("EMPTY_ID", "BAD_ID", "DUP_ID"):
            continue
        if r.cid not in crm_reg:
            r.add("ORPHAN", "customer_id")
            continue
        reg, when = crm_reg[r.cid], r.typed.get(date_field)
        if reg and when and when < reg:
            r.add("CROSS_DATE", date_field)


def _check_zero_tickets(rows: List[Row]) -> None:
    for r in rows:
        if r.typed.get("support_tickets") == 0 and r.raw.get("last_ticket_date") is not None:
            r.add("ZERO_TICKETS_WITH_DATE", "last_ticket_date")


# ---------- actions, outputs, report ----------

def _is_rejected(row: Row, cfg: Settings) -> bool:
    return any(cfg.validation.action(i.rule) == "reject" for i in row.issues)


def _has_severity(row: Row, cfg: Settings, error: bool) -> bool:
    return any(cfg.validation.is_error(i.rule) is error for i in row.issues)


def _raw_json(row: Row) -> str:
    return json.dumps({k: v for k, v in row.raw.items()
                       if k not in ("run_id", "ingested_at")}, ensure_ascii=False, default=str)


def _apply_actions(rows: List[Row], system: str, cfg: Settings):
    """Split rows into kept typed records and rejected/quarantined entries."""
    kept, out = [], []
    canonical = [f for f in cfg.sources[cfg.source_for(system)].field_map.values()]
    for r in rows:
        if _is_rejected(r, cfg):
            out.append({"source_file": r.raw["source_file"], "source_row": r.raw["source_row"],
                        "customer_id": r.cid, "rule_codes": "|".join(r.codes()),
                        "action": "rejected", "raw_record": _raw_json(r)})
            continue
        typed_fields = set(DATE_FIELDS[system] + INT_FIELDS[system] + FLOAT_FIELDS[system])
        # Typed fields that failed to parse are always null; text fields keep their raw value.
        rec = {f: r.typed.get(f) if f in typed_fields else r.typed.get(f, r.raw.get(f))
               for f in canonical}
        quarantined = []
        for i in r.issues:
            action = cfg.validation.action(i.rule)
            if action in ("null_and_flag", "null_and_quarantine") and i.field:
                rec[i.field] = None
            if action == "null_and_quarantine":
                quarantined.append(i.rule)
        if quarantined:
            out.append({"source_file": r.raw["source_file"], "source_row": r.raw["source_row"],
                        "customer_id": r.cid, "rule_codes": "|".join(dict.fromkeys(quarantined)),
                        "action": "quarantined_value", "raw_record": _raw_json(r)})
        for lin in ("source_system", "source_file", "source_row", "run_id", "ingested_at"):
            rec[lin] = r.raw.get(lin)
        rec["dq_flags"] = r.codes()
        kept.append(rec)
    return kept, out


def _typed_frame(records: List[Dict[str, Any]], system: str, columns: List[str]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records, columns=columns)
    for f in DATE_FIELDS[system]:
        df[f] = pd.to_datetime(df[f], errors="raise")
    for f in INT_FIELDS[system]:
        df[f] = df[f].astype("Int64")
    for f in FLOAT_FIELDS[system]:
        df[f] = df[f].astype("Float64")
    return df


def _rows_with(rows: List[Row], rules: List[str]) -> int:
    return sum(1 for r in rows if r.has(*rules))


def _by_rule(rows: List[Row]) -> Dict[str, int]:
    counts = Counter(code for r in rows for code in r.codes())
    return dict(sorted(counts.items()))


def _tally(rows: List[Row], cfg: Settings) -> Dict[str, int]:
    """rejected; with_error (any error rule); flagged_kept (kept, has an error rule);
    warning_records (kept, has a warning rule)."""
    t = {"rejected": 0, "with_error": 0, "flagged_kept": 0, "warning_records": 0}
    for r in rows:
        rejected, has_error = _is_rejected(r, cfg), _has_severity(r, cfg, error=True)
        t["rejected"] += rejected
        t["with_error"] += has_error
        t["flagged_kept"] += has_error and not rejected
        t["warning_records"] += _has_severity(r, cfg, error=False) and not rejected
    return t


def build_report(all_rows: Dict[str, List[Row]], clean: Dict[str, pd.DataFrame],
                 cfg: Settings, run_id: str) -> Dict[str, Any]:
    rows = [r for rs in all_rows.values() for r in rs]
    total = len(rows)
    tally = _tally(rows, cfg)
    counts = {key: _rows_with(rows, rules) for key, rules in REPORT_KEYS.items()}

    by_source = {}
    for system, rs in all_rows.items():
        t = _tally(rs, cfg)
        by_source[cfg.source_for(system)] = {
            "input": len(rs), "clean": len(clean[system]), "rejected": t["rejected"],
            "reject_rate": round(t["rejected"] / len(rs), 6) if rs else 0.0,
            "successful": len(rs) - t["with_error"],
            "flagged_kept": t["flagged_kept"], "warning_records": t["warning_records"],
            "balanced": len(rs) == len(clean[system]) + t["rejected"],
            "by_rule": _by_rule(rs),
        }
    rejected = tally["rejected"]
    n_clean = sum(len(df) for df in clean.values())
    return {
        "run_id": run_id,
        "as_of_date": cfg.as_of_date.isoformat(),
        "total_records": total,
        "duplicates": counts["duplicates"],
        "invalid_emails": counts["invalid_emails"],
        "missing_fields": counts["missing_fields"],
        "successful_records": total - tally["with_error"],
        "invalid_ids": counts["invalid_ids"],
        "negative_revenue": counts["negative_revenue"],
        "orphans": counts["orphans"],
        "rejected_unique_records": rejected,
        "flagged_kept_records": tally["flagged_kept"],
        "warning_records": tally["warning_records"],
        "customers_in_360": len(clean["crm"]),
        "reject_rate": round(rejected / total, 6) if total else 0.0,
        "max_reject_rate": cfg.validation.max_reject_rate,
        "by_rule": _by_rule(rows),
        "by_source": by_source,
        "reconciliation": {"input": total, "clean": n_clean, "rejected": rejected,
                           "balanced": total == n_clean + rejected},
    }


def validate(staged: Dict[str, pd.DataFrame], cfg: Settings, run_id: str) -> ValidationResult:
    """Run every rule over the staged frames (keys: crm / orders / support)."""
    id_re = re.compile(cfg.validation.id_regex)
    all_rows: Dict[str, List[Row]] = {
        s: [Row(raw=rec, typed={}) for rec in staged[s].to_dict("records")]
        for s in ("crm", "orders", "support")
    }
    crm = all_rows["crm"]
    _check_ids(crm, id_re)
    _check_crm(crm, cfg)
    _check_dates(crm, "crm", cfg)
    crm_reg = {r.cid: r.typed.get("registration_date") for r in crm if not _is_rejected(r, cfg)}

    for system in ("orders", "support"):
        rows = all_rows[system]
        _check_ids(rows, id_re)
        _check_numbers(rows, system)
        _check_dates(rows, system, cfg)
        _check_links(rows, system, crm_reg)
    _check_zero_tickets(all_rows["support"])

    clean: Dict[str, pd.DataFrame] = {}
    rejected_entries: List[Dict[str, Any]] = []
    for system, rows in all_rows.items():
        kept, out = _apply_actions(rows, system, cfg)
        columns = list(staged[system].columns) + ["dq_flags"]
        clean[system] = _typed_frame(kept, system, columns)
        rejected_entries.extend(out)
    rejected = pd.DataFrame(rejected_entries, columns=REJECTED_COLUMNS)

    report = build_report(all_rows, clean, cfg, run_id)
    log.info("validated | input=%d clean=%d rejected=%d successful=%d",
             report["total_records"], report["reconciliation"]["clean"],
             report["rejected_unique_records"], report["successful_records"])
    for src, s in report["by_source"].items():
        log.info("reconcile %-22s input=%d clean=%d rejected=%d balanced=%s",
                 src, s["input"], s["clean"], s["rejected"], s["balanced"])
    if not report["reconciliation"]["balanced"]:
        raise AssertionError(f"reconciliation failed: {report['reconciliation']}")
    return ValidationResult(clean=clean, rejected=rejected, report=report)


def check_reject_rate(report: Dict[str, Any]) -> None:
    """Fail fast before publishing if too many rows were rejected."""
    if report["reject_rate"] > report["max_reject_rate"]:
        raise RejectRateExceeded(
            f"reject rate {report['reject_rate']:.2%} exceeds max {report['max_reject_rate']:.2%}"
        )


def add_customer_level_flags(report: Dict[str, Any], customers: pd.DataFrame) -> None:
    """Record flags that only exist after the merge (e.g. NO_ORDER_ROW) and check the 360 count."""
    if report["customers_in_360"] != len(customers):
        raise AssertionError(f"customers_in_360={report['customers_in_360']} but merged {len(customers)}")
    merge_rules = ("NO_ORDER_ROW", "NO_TICKET_ROW")
    report["customer_level_flags"] = {
        rule: int(customers["dq_flags"].map(lambda f, r=rule: r in f).sum()) for rule in merge_rules
    }
