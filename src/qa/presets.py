"""Preset questions with fixed, tested SQL. Used when the LLM is unavailable or refuses."""
from __future__ import annotations

from typing import Dict, List

PRESETS: List[Dict[str, str]] = [
    {"id": "top10", "question": "Who are the top 10 customers by revenue?",
     "sql": "SELECT customer_id, name, city, segment, lifetime_value FROM customer_360 "
            "WHERE lifetime_value IS NOT NULL ORDER BY lifetime_value DESC, customer_id LIMIT 10"},
    {"id": "revenue_by_city", "question": "What is the revenue by city?",
     "sql": "SELECT city, SUM(lifetime_value) AS total_revenue, COUNT(*) AS customers "
            "FROM customer_360 GROUP BY city ORDER BY total_revenue DESC, city"},
    {"id": "segment_mix", "question": "How are customers split across segments?",
     "sql": "SELECT segment, COUNT(*) AS customers, SUM(lifetime_value) AS revenue FROM customer_360 "
            "GROUP BY segment ORDER BY revenue DESC"},
    {"id": "gt5_tickets", "question": "Which customers have more than 5 support tickets?",
     "sql": "SELECT customer_id, name, city, support_tickets FROM customer_360 "
            "WHERE support_tickets > 5 ORDER BY support_tickets DESC, customer_id"},
    {"id": "at_risk_vips", "question": "Which high-value customers are at risk?",
     "sql": "SELECT customer_id, name, city, lifetime_value, risk_score FROM customer_360 "
            "WHERE at_risk_vip = 1 ORDER BY risk_score DESC, customer_id"},
    {"id": "dq_issues", "question": "What data quality issues were found?",
     "sql": "SELECT rule_codes, action, COUNT(*) AS records FROM rejected_records "
            "GROUP BY rule_codes, action ORDER BY records DESC, rule_codes"},
    {"id": "risk_bands", "question": "How many customers are in each risk band?",
     "sql": "SELECT risk_band, COUNT(*) AS customers FROM customer_360 GROUP BY risk_band "
            "ORDER BY CASE risk_band WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END"},
    {"id": "ticket_heavy_low_value", "question": "Which customers raise many tickets but bring little revenue?",
     "sql": "SELECT customer_id, name, city, support_tickets, lifetime_value FROM customer_360 "
            "WHERE support_tickets > 5 AND segment = 'Bronze' ORDER BY support_tickets DESC, customer_id"},
]


def get_preset(preset_id: str) -> Dict[str, str]:
    for p in PRESETS:
        if p["id"] == preset_id:
            return p
    raise KeyError(f"unknown preset: {preset_id}")
