"""Join clean CRM, orders and support into one row per customer."""
from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd

from src.config import Settings

log = logging.getLogger("merge")

CORE_FIELDS: List[str] = [
    "customer_id", "name", "email", "city", "registration_date",
    "total_orders", "lifetime_value", "last_order_date", "support_tickets", "last_ticket_date",
]


def _combine_flags(*flag_lists) -> List[str]:
    """Merge per-source flag lists, keeping first-seen order and dropping repeats/missing."""
    out: List[str] = []
    for flags in flag_lists:
        if isinstance(flags, list):
            out.extend(f for f in flags if f not in out)
    return out


def merge(clean: Dict[str, pd.DataFrame], cfg: Settings) -> pd.DataFrame:
    """CRM LEFT JOIN orders LEFT JOIN support on customer_id; fill defaults for missing rows."""
    crm, orders, support = clean["crm"], clean["orders"], clean["support"]
    defaults = cfg.merge.missing_row_defaults

    o = orders[["customer_id", "total_orders", "lifetime_value", "last_order_date", "dq_flags"]]
    o = o.rename(columns={"dq_flags": "_flags_orders"}).assign(_has_order=True)
    t = support[["customer_id", "support_tickets", "last_ticket_date", "dq_flags"]]
    t = t.rename(columns={"dq_flags": "_flags_support"}).assign(_has_ticket=True)
    c = crm.rename(columns={"dq_flags": "_flags_crm"})

    df = (c.merge(o, on="customer_id", how="left", validate="one_to_one")
           .merge(t, on="customer_id", how="left", validate="one_to_one"))
    if len(df) != len(crm) or df.customer_id.duplicated().any():
        raise AssertionError(f"merge produced {len(df)} rows for {len(crm)} CRM customers")

    no_order = df["_has_order"].isna()
    no_ticket = df["_has_ticket"].isna()
    df.loc[no_order, "total_orders"] = defaults["total_orders"]
    df.loc[no_order, "lifetime_value"] = defaults["lifetime_value"]
    df.loc[no_ticket, "support_tickets"] = defaults["support_tickets"]
    df["total_orders"] = df["total_orders"].astype("Int64")
    df["support_tickets"] = df["support_tickets"].astype("Int64")
    df["lifetime_value"] = df["lifetime_value"].astype("Float64")

    merge_flags = [
        (["NO_ORDER_ROW"] if a else []) + (["NO_TICKET_ROW"] if b else [])
        for a, b in zip(no_order, no_ticket)
    ]
    df["dq_flags"] = [
        _combine_flags(fc, fo, fs, fm)
        for fc, fo, fs, fm in zip(df["_flags_crm"], df["_flags_orders"], df["_flags_support"], merge_flags)
    ]
    df["source_system"] = [
        ",".join(["crm"] + (["orders"] if not a else []) + (["support"] if not b else []))
        for a, b in zip(no_order, no_ticket)
    ]
    df = df[CORE_FIELDS + ["dq_flags", "source_system", "run_id", "ingested_at"]]
    df = df.sort_values("customer_id").reset_index(drop=True)

    log.info("merged | crm=%d orders=%d support=%d -> customers=%d | no_order_row=%d no_ticket_row=%d",
             len(crm), len(orders), len(support), len(df), int(no_order.sum()), int(no_ticket.sum()))
    return df
