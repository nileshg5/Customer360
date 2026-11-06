"""Executive summary: aggregates only, grounded numbers, template fallback."""
from __future__ import annotations

import json

import pandas as pd

from src.qa.exec_summary import build_aggregates, generate, render_markdown
from src.qa.llm_client import LlmReply

DQ = {"total_records": 10, "successful_records": 9, "rejected_unique_records": 1, "warning_records": 0,
      "customers_in_360": 3}


def customers() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["C0001", "C0002", "C0003"], "name": ["Asha Rao", "Ravi Kumar", "Meera Iyer"],
        "email_masked": ["a***@x.com"] * 3, "city": ["Pune", "Pune", "Delhi"],
        "segment": ["Gold", "Gold", "Bronze"], "lifetime_value": [200000.0, 150000.0, 20000.0],
        "risk_band": ["High", "Low", "Medium"], "at_risk_vip": [True, False, False],
        "support_tickets": [9, 1, 7]})


class Fake:
    def __init__(self, text):
        self.text, self.sent = text, None

    def complete(self, messages, on_status=None):
        self.sent = messages
        return LlmReply(json.dumps({"summary_markdown": self.text}), "fake")


def test_aggregates_contain_no_customer_level_data():
    a = build_aggregates(customers(), DQ, 5)
    blob = json.dumps(a)
    for secret in ("Asha", "Ravi", "C0001", "a***@x.com", "@"):
        assert secret not in blob
    assert a["total_revenue"] == 370000 and a["at_risk_vips"] == 1 and a["customers_over_5_tickets"] == 2


def test_template_without_client():
    text, source = generate(build_aggregates(customers(), DQ, 5), None, "2026-10-28")
    assert source == "template" and "0.37M" in text and "90.0%" in text


def test_grounded_llm_summary_used():
    fake = Fake("- Revenue is 0.37M across 3 customers.\n- 1 at-risk VIP.")
    text, source = generate(build_aggregates(customers(), DQ, 5), fake, "2026-10-28")
    assert source == "fake" and text.startswith("- Revenue is 0.37M")
    assert "Asha" not in json.dumps(fake.sent)


def test_invented_number_falls_back_to_template():
    text, source = generate(build_aggregates(customers(), DQ, 5), Fake("- Revenue grew 42% year on year."),
                            "2026-10-28")
    assert source == "template"


def test_render_markdown_header():
    doc = render_markdown("- x", "template", "RUN1", "2026-10-28")
    assert doc.startswith("# Customer 360: executive summary") and "RUN1" in doc and "2026-10-28" in doc


def test_double_escaped_newlines_are_fixed():
    text, source = generate(build_aggregates(customers(), DQ, 5), Fake("- 3 customers\\n- 1 at-risk VIP"),
                            "2026-10-28")
    assert source == "fake" and text == "- 3 customers\n- 1 at-risk VIP"
