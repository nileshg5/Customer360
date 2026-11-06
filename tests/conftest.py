"""Shared fixtures: project config and a builder for small staged datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

from src.config import Settings, load_config
from src.standardize import LINEAGE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = {
    "crm": ["customer_id", "name", "email", "city", "registration_date"],
    "orders": ["customer_id", "total_orders", "lifetime_value", "last_order_date"],
    "support": ["customer_id", "support_tickets", "last_ticket_date"],
}
FILES = {"crm": "crm_customers.json", "orders": "orders.csv", "support": "support_tickets.xml"}


@pytest.fixture(scope="session")
def cfg() -> Settings:
    return load_config(PROJECT_ROOT / "config.json")


def crm_row(cid: str = "C0001", **over) -> Dict[str, Optional[str]]:
    n = (cid or "none")[-4:]
    row = {"customer_id": cid, "name": f"Test {n}", "email": f"user{n}@example.com",
           "city": "Pune", "registration_date": "2024-01-01"}
    row.update(over)
    return row


def order_row(cid: str = "C0001", **over) -> Dict[str, Optional[str]]:
    row = {"customer_id": cid, "total_orders": "5", "lifetime_value": "60000",
           "last_order_date": "2026-05-01"}
    row.update(over)
    return row


def ticket_row(cid: str = "C0001", **over) -> Dict[str, Optional[str]]:
    row = {"customer_id": cid, "support_tickets": "2", "last_ticket_date": "2026-06-01"}
    row.update(over)
    return row


def staged(crm: List[dict], orders: Optional[List[dict]] = None,
           support: Optional[List[dict]] = None) -> Dict[str, pd.DataFrame]:
    """Build staged frames shaped exactly like standardize() output."""
    out = {}
    for system, rows in (("crm", crm), ("orders", orders or []), ("support", support or [])):
        df = pd.DataFrame(rows, columns=CANONICAL[system]).astype(object)
        df = df.where(df.notna(), None)
        df["source_system"] = system
        df["source_file"] = FILES[system]
        df["source_row"] = range(1, len(df) + 1)
        df["run_id"] = "TEST"
        df["ingested_at"] = "2026-01-01T00:00:00Z"
        out[system] = df[CANONICAL[system] + LINEAGE_COLUMNS]
    return out
