"""Q&A eval: 12 questions whose expected answers are computed independently with pandas
from the published JSON/CSV outputs (never from SQL or the LLM).

  python -m src.qa.evaluate            # live LLM if GROQ_API_KEY is set, else reference SQL
  python -m src.qa.evaluate --reference

Reference mode runs a hand-written SQL per question through the same guard + read-only
executor, so it checks the SQL layer, not the LLM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.config import Settings, load_config
from src.logging_setup import setup_logging
from src.qa.engine import QaEngine, QaResult, build_engine
from src.qa.llm_client import client_from_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class EvalItem:
    id: str
    question: str
    kind: str                      # scalar | label | mapping
    expected: Callable[[pd.DataFrame, pd.DataFrame, Settings], Any]
    reference_sql: str


def _days_before(df: pd.DataFrame, cfg: Settings, col: str) -> pd.Series:
    return (pd.Timestamp(cfg.as_of_date) - pd.to_datetime(df[col])).dt.days


EVAL_SET: List[EvalItem] = [
    EvalItem("top_customer", "Who is the top customer by revenue?", "label",
             lambda c, r, cfg: c.sort_values(["lifetime_value", "customer_id"], ascending=[False, True]).iloc[0]["customer_id"],
             "SELECT customer_id, name FROM customer_360 WHERE lifetime_value IS NOT NULL "
             "ORDER BY lifetime_value DESC, customer_id LIMIT 1"),
    EvalItem("revenue_by_city", "What is the total revenue by city?", "mapping",
             lambda c, r, cfg: c.groupby("city")["lifetime_value"].sum().to_dict(),
             "SELECT city, SUM(lifetime_value) AS revenue FROM customer_360 GROUP BY city"),
    EvalItem("gold_count", "How many Gold customers are there?", "scalar",
             lambda c, r, cfg: int((c.segment == "Gold").sum()),
             "SELECT COUNT(*) FROM customer_360 WHERE segment = 'Gold'"),
    EvalItem("gt5_tickets", "How many customers have more than 5 support tickets?", "scalar",
             lambda c, r, cfg: int((c.support_tickets > cfg.support.high_ticket_threshold).sum()),
             "SELECT COUNT(*) FROM customer_360 WHERE support_tickets > 5"),
    EvalItem("at_risk_vips", "How many at-risk VIP customers do we have?", "scalar",
             lambda c, r, cfg: int(((c.segment == "Gold") & (c.risk_band == "High")).sum()),
             "SELECT COUNT(*) FROM customer_360 WHERE at_risk_vip = 1"),
    EvalItem("rejected_count", "How many records were rejected by the data quality checks?", "scalar",
             lambda c, r, cfg: int((r.action == "rejected").sum()),
             "SELECT COUNT(*) FROM rejected_records WHERE action = 'rejected'"),
    EvalItem("recent_orders_90d", "How many customers placed their last order in the 90 days before the as-of date?",
             "scalar",
             lambda c, r, cfg: int((_days_before(c, cfg, "last_order_date") <= 90).sum()),
             "SELECT COUNT(*) FROM customer_360 WHERE last_order_date >= date('{as_of}', '-90 days')"),
    EvalItem("top_city_customers", "Which city has the most customers?", "label",
             lambda c, r, cfg: c.city.value_counts().idxmax(),
             "SELECT city, COUNT(*) AS n FROM customer_360 GROUP BY city ORDER BY n DESC, city LIMIT 1"),
    EvalItem("silver_avg_ltv", "What is the average lifetime value of Silver customers?", "scalar",
             lambda c, r, cfg: float(c.loc[c.segment == "Silver", "lifetime_value"].mean()),
             "SELECT AVG(lifetime_value) FROM customer_360 WHERE segment = 'Silver'"),
    EvalItem("high_risk_count", "How many customers are in the High risk band?", "scalar",
             lambda c, r, cfg: int((c.risk_band == "High").sum()),
             "SELECT COUNT(*) FROM customer_360 WHERE risk_band = 'High'"),
    EvalItem("top_segment_revenue", "Which segment has the highest total revenue?", "label",
             lambda c, r, cfg: c.groupby("segment")["lifetime_value"].sum().idxmax(),
             "SELECT segment, SUM(lifetime_value) AS rev FROM customer_360 GROUP BY segment ORDER BY rev DESC LIMIT 1"),
    EvalItem("registered_2025", "How many customers registered in 2025?", "scalar",
             lambda c, r, cfg: int(pd.to_datetime(c.registration_date).dt.year.eq(2025).sum()),
             "SELECT COUNT(*) FROM customer_360 WHERE registration_date BETWEEN '2025-01-01' AND '2025-12-31'"),
]


def _close(a: Any, b: float) -> bool:
    try:
        return abs(float(a) - b) <= max(0.01, abs(b) * 1e-4)
    except (TypeError, ValueError):
        return False


def check(kind: str, expected: Any, res: QaResult) -> bool:
    """Lenient on column names, strict on values."""
    if res.status != "ok" or not res.rows:
        return False
    first = res.rows[0]
    if kind == "scalar":
        return any(_close(v, expected) for v in first if not isinstance(v, str))
    if kind == "label":
        return any(isinstance(v, str) and v == expected for v in first)
    if kind == "mapping":
        df = res.frame()
        keys = [c for c in df.columns if df[c].map(lambda v: isinstance(v, str)).all()]
        nums = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]
        if not keys or not nums:
            return False
        got = dict(zip(df[keys[0]], df[nums[0]]))
        return set(got) == set(expected) and all(_close(got[k], v) for k, v in expected.items())
    raise ValueError(kind)


def _summarise_got(kind: str, res: QaResult) -> str:
    if res.status != "ok":
        return f"[{res.status}] {res.message[:60]}"
    if kind == "mapping":
        return f"{len(res.rows)} rows"
    return ", ".join(str(v) for v in (res.rows[0] if res.rows else []))[:60]


def run_eval(cfg: Settings, engine: QaEngine, live: bool, pause_s: float = 0.0) -> List[Dict[str, Any]]:
    latest = cfg.output_dir / cfg.paths.latest_dir_name
    customers = pd.DataFrame(json.loads((latest / "customer_360_enriched.json").read_text(encoding="utf-8")))
    rejected = pd.read_csv(latest / "rejected_records.csv")
    rows = []
    for item in EVAL_SET:
        expected = item.expected(customers, rejected, cfg)
        if live:
            res = engine.ask(item.question)
            time.sleep(pause_s)
        else:
            res = QaResult(question=item.question, status="ok", model="reference-sql")
            engine._execute(res, item.reference_sql.replace("{as_of}", cfg.as_of_date.isoformat()))
        rows.append({"id": item.id, "question": item.question,
                     "expected": round(expected, 2) if isinstance(expected, float) else
                     (f"{len(expected)} cities" if isinstance(expected, dict) else expected),
                     "got": _summarise_got(item.kind, res), "pass": check(item.kind, expected, res),
                     "model": res.model, "attempts": res.attempts, "sql": res.sql})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reference", action="store_true", help="use reference SQL instead of the LLM")
    p.add_argument("--pause", type=float, default=1.0, help="seconds between live questions (rate limits)")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(PROJECT_ROOT / "config.json")
    setup_logging(cfg.log_file, "qa-eval", secret_env_vars=[cfg.llm.api_key_env], level=30)
    client = None if args.reference else client_from_env(cfg)
    engine = build_engine(cfg, client)
    rows = run_eval(cfg, engine, live=client is not None, pause_s=args.pause)

    mode = "LIVE LLM" if client is not None else "REFERENCE SQL (no LLM)"
    print(f"Mode: {mode}")
    print(pd.DataFrame(rows)[["id", "expected", "got", "pass", "model", "attempts"]].to_string(index=False))
    passed = sum(r["pass"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed")
    out = cfg.output_dir / "qa_eval.json"
    out.write_text(json.dumps({"mode": mode, "passed": passed, "total": len(rows), "results": rows},
                              indent=2, default=str), encoding="utf-8")
    print(f"details: {out}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
