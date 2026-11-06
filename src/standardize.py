"""Map each source onto the canonical schema. Values stay strings until validation.

Standardization here is deliberately minimal and loss-free:
  * rename source fields via `field_map` (cust_id / customerId -> customer_id)
  * trim surrounding whitespace; empty string -> None
  * add any canonical column the source lacks (as None) and drop unmapped columns (logged)
  * add lineage: source_system, source_file, run_id, ingested_at
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

from src.config import SourceSpec

log = logging.getLogger("standardize")

LINEAGE_COLUMNS: List[str] = ["source_system", "source_file", "source_row", "run_id", "ingested_at"]


def _clean_text(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def standardize(df: pd.DataFrame, spec: SourceSpec, source_file: str,
                run_id: str, ingested_at: str) -> pd.DataFrame:
    """Return a frame with canonical columns (in field_map order) followed by lineage columns."""
    canonical = list(spec.field_map.values())
    present = [c for c in df.columns if c != "source_row"]

    unmapped = [c for c in present if c not in spec.field_map]
    if unmapped:
        log.warning("%s: dropping unmapped columns %s", source_file, unmapped)
    absent = [src for src in spec.field_map if src not in present]
    if absent:
        log.warning("%s: expected columns missing, filled with null: %s", source_file, absent)

    out = pd.DataFrame(index=df.index)
    for src, dst in spec.field_map.items():
        col = df[src] if src in df.columns else pd.Series([None] * len(df), index=df.index, dtype=object)
        out[dst] = col.map(_clean_text).astype(object)

    out["source_system"] = spec.system
    out["source_file"] = source_file
    out["source_row"] = df["source_row"].values
    out["run_id"] = run_id
    out["ingested_at"] = ingested_at
    out = out[canonical + LINEAGE_COLUMNS].reset_index(drop=True)

    log.info("standardized %-22s rows_in=%d rows_out=%d", source_file, len(df), len(out))
    return out
