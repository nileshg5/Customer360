"""Executive summary from aggregated numbers only (no customer rows, names, emails or IDs)."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.qa.engine import _grounded, _numbers_in
from src.qa.llm_client import ChatClient, LlmError, StatusCallback

log = logging.getLogger("qa.exec_summary")


def build_aggregates(customers: pd.DataFrame, dq: Dict[str, Any], threshold: int) -> Dict[str, Any]:
    ltv = customers["lifetime_value"]
    total = float(ltv.sum())
    seg = customers.groupby("segment")["lifetime_value"].agg(["size", "sum"])
    city = customers.groupby("city")["lifetime_value"].sum().sort_values(ascending=False)
    return {
        "customers": int(len(customers)),
        "total_revenue": round(total), "total_revenue_millions": round(total / 1e6, 2),
        "avg_lifetime_value": round(float(ltv.mean()), 2),
        "segments": {s: {"customers": int(r["size"]), "revenue_share_pct": round(r["sum"] / total * 100, 1)}
                     for s, r in seg.iterrows()},
        "top_cities_by_revenue_millions": {c: round(v / 1e6, 2) for c, v in city.head(3).items()},
        "risk_bands": customers["risk_band"].value_counts().to_dict(),
        "at_risk_vips": int(customers["at_risk_vip"].sum()),
        f"customers_over_{threshold}_tickets": int((customers["support_tickets"] > threshold).sum()),
        "dq": {k: dq[k] for k in ("total_records", "successful_records", "rejected_unique_records",
                                  "warning_records", "customers_in_360")},
        "dq_pass_rate_pct": round(dq["successful_records"] / dq["total_records"] * 100, 2),
    }


def template_summary(a: Dict[str, Any]) -> str:
    gold = a["segments"].get("Gold", {})
    top_city = next(iter(a["top_cities_by_revenue_millions"]), "-")
    over = next(v for k, v in a.items() if k.startswith("customers_over_"))
    return "\n".join([
        f"- **Revenue:** {a['total_revenue_millions']}M across {a['customers']} customers "
        f"(average lifetime value {a['avg_lifetime_value']:,}).",
        f"- **Concentration:** Gold customers ({gold.get('customers', 0)}) bring {gold.get('revenue_share_pct', 0)}% "
        f"of revenue; {top_city} is the top city.",
        f"- **Risk:** {a['at_risk_vips']} Gold customers are in the High risk band (at-risk VIPs).",
        f"- **Support:** {over} customers have more than the support threshold of tickets.",
        f"- **Data quality:** {a['dq_pass_rate_pct']}% of {a['dq']['total_records']} source rows are error-free; "
        f"{a['dq']['rejected_unique_records']} rows were rejected.",
    ])


def _allowed_numbers(a: Dict[str, Any]) -> set:
    out: set = set()

    def walk(v):
        if isinstance(v, dict):
            for k, x in v.items():
                out.update(_numbers_in(k))
                walk(x)
        else:
            out.update(_numbers_in(v))
    walk(a)
    return out


def generate(a: Dict[str, Any], client: Optional[ChatClient], as_of: str,
             on_status: StatusCallback = None) -> Tuple[str, str]:
    """Return (markdown, source) where source is the model name or 'template'."""
    if client is None:
        return template_summary(a), "template"
    messages = [
        {"role": "system", "content": (
            "Write an executive summary of these customer metrics as 4-6 short markdown bullet points "
            "for a business leader: revenue, concentration, risk, support load, data quality, one recommendation. "
            "Use ONLY numbers present in the JSON; do not compute new numbers. No currency symbols. "
            "Reply with JSON only: {\"summary_markdown\": \"...\"}")},
        {"role": "user", "content": json.dumps({"as_of_date": as_of, "metrics": a})},
    ]
    try:
        reply = client.complete(messages, on_status)
        text = json.loads(reply.text).get("summary_markdown", "")
        if isinstance(text, str):
            text = text.replace("\\n", "\n")          # some replies double-escape newlines
    except (LlmError, ValueError, AttributeError) as exc:
        log.warning("executive summary failed: %s", exc)
        return template_summary(a), "template"
    if not isinstance(text, str) or not text.strip() or not _grounded(text, _allowed_numbers(a) | _numbers_in(as_of)):
        log.warning("executive summary rejected (empty or ungrounded numbers); using template")
        return template_summary(a), "template"
    return text.strip(), reply.model


def render_markdown(summary: str, source: str, run_id: str, as_of: str) -> str:
    return (f"# Customer 360: executive summary\n\nAs of **{as_of}** · run `{run_id}` · "
            f"written by {source}\n\n{summary}\n\n_Generated from aggregated metrics only; "
            "no customer-level data was sent to the AI._\n")
