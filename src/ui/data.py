"""Read output/latest/ for the dashboard: retried reads, cached per run_id, emails masked."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd
import streamlit as st

from src.config import Settings

T = TypeVar("T")
DATE_COLUMNS = ["registration_date", "last_order_date", "last_ticket_date"]


@dataclass(frozen=True)
class Bundle:
    run_id: str
    latest: str
    manifest: Dict[str, Any]
    dq: Dict[str, Any]
    customers: pd.DataFrame          # enriched, with email replaced by email_masked
    rejected: pd.DataFrame
    insights: Dict[str, pd.DataFrame]


@dataclass
class Filters:
    cities: List[str]
    segments: List[str]
    risk_bands: List[str]

    @property
    def active(self) -> bool:
        return bool(self.cities or self.segments or self.risk_bands)


def latest_dir(cfg: Settings) -> Path:
    return cfg.output_dir / cfg.paths.latest_dir_name


def with_retry(fn: Callable[[], T], attempts: int = 4, delay: float = 0.25) -> T:
    """latest/ is swapped by rename; a read can briefly hit a missing or locked file."""
    for i in range(attempts):
        try:
            return fn()
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            if i == attempts - 1:
                raise
            time.sleep(delay)
    raise RuntimeError("unreachable")


def current_run_id(latest: Path) -> Optional[str]:
    """run_id of the published run, or None if nothing usable is published."""
    manifest = latest / "run_manifest.json"
    if not latest.exists():
        return None
    try:
        data = with_retry(lambda: json.loads(manifest.read_text(encoding="utf-8")))
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None
    return data.get("run_id") if data.get("status") == "success" else None


def mask_email(email: Optional[str]) -> str:
    """user125@example.com -> u***@example.com; missing -> '-'."""
    if not isinstance(email, str) or "@" not in email:
        return "-"
    local, domain = email.split("@", 1)
    return f"{local[:1]}***@{domain}"


def _read_json(path: Path):
    return with_retry(lambda: json.loads(path.read_text(encoding="utf-8")))


@st.cache_data(show_spinner=False, max_entries=3)
def load_bundle(latest: str, run_id: str) -> Bundle:
    """Cached on (latest path, run_id): a new pipeline run invalidates the cache automatically."""
    root = Path(latest)
    customers = pd.DataFrame(_read_json(root / "customer_360_enriched.json"))
    customers.insert(customers.columns.get_loc("email"), "email_masked", customers["email"].map(mask_email))
    customers = customers.drop(columns=["email"])
    for c in DATE_COLUMNS:
        customers[c] = pd.to_datetime(customers[c])
    customers["at_risk_vip"] = customers["at_risk_vip"].astype(bool)
    insights = {p.stem: with_retry(lambda p=p: pd.read_csv(p)) for p in sorted((root / "insights").glob("*.csv"))}
    return Bundle(
        run_id=run_id, latest=latest,
        manifest=_read_json(root / "run_manifest.json"),
        dq=_read_json(root / "dq_report.json"),
        customers=customers,
        rejected=with_retry(lambda: pd.read_csv(root / "rejected_records.csv")).drop(columns=["raw_record"]),
        insights=insights,
    )


def read_bytes(latest: str, relative: str) -> bytes:
    return with_retry(lambda: (Path(latest) / relative).read_bytes())


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    out = df
    if f.cities:
        out = out[out["city"].isin(f.cities)]
    if f.segments:
        out = out[out["segment"].isin(f.segments)]
    if f.risk_bands:
        out = out[out["risk_band"].isin(f.risk_bands)]
    return out
