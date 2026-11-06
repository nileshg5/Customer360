"""Write pipeline outputs to the run directory and promote a finished run to latest/."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.merge import CORE_FIELDS

log = logging.getLogger("publish")

DB_NAME = "customer360.db"
CUSTOMER_TABLE = "customer_360"
REJECTED_TABLE = "rejected_records"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def json_value(v: Any) -> Any:
    """Normalise a pandas/numpy cell for JSON: dates -> YYYY-MM-DD, whole floats -> int, missing -> null."""
    if isinstance(v, (list, tuple)):
        return list(v)
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        return int(f) if f.is_integer() else f
    return v


def to_records(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, Any]]:
    return [{c: json_value(v) for c, v in zip(columns, row)}
            for row in df[columns].itertuples(index=False, name=None)]


def write_dq_outputs(report: Dict[str, Any], rejected: pd.DataFrame, run_dir: Path) -> None:
    """dq_report.json and rejected_records.csv for a run."""
    write_json(report, run_dir / "dq_report.json")
    rejected.to_csv(run_dir / "rejected_records.csv", index=False)


def write_customer_outputs(enriched: pd.DataFrame, run_dir: Path) -> None:
    """customer_360.json (exactly the 10 contract fields) and customer_360_enriched.json (all)."""
    write_json(to_records(enriched, CORE_FIELDS), run_dir / "customer_360.json")
    write_json(to_records(enriched, list(enriched.columns)), run_dir / "customer_360_enriched.json")


def write_insights(insights: Dict[str, pd.DataFrame], run_dir: Path) -> None:
    out = run_dir / "insights"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in insights.items():
        csv_frame(df).to_csv(out / f"{name}.csv", index=False)


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Dates as YYYY-MM-DD; whole-number floats written without '.0'; real decimals kept."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(df[c]):
            # list comprehension, not .map: .map re-infers float dtype and brings the ".0" back
            df[c] = pd.Series([json_value(v) for v in df[c]], index=df.index, dtype=object)
    return df


def _sqlite_frame(df: pd.DataFrame):
    """Convert to SQLite-friendly columns and an explicit column type map."""
    out, types = pd.DataFrame(index=df.index), {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c], types[c] = s.dt.strftime("%Y-%m-%d").astype(object).where(s.notna(), None), "TEXT"
        elif pd.api.types.is_bool_dtype(s):
            out[c], types[c] = s.astype(int), "INTEGER"
        elif pd.api.types.is_integer_dtype(s):
            out[c], types[c] = s.astype("Int64"), "INTEGER"
        elif pd.api.types.is_float_dtype(s):
            out[c], types[c] = s.astype("Float64"), "REAL"
        else:
            out[c] = s.map(lambda v: json.dumps(v) if isinstance(v, list) else json_value(v))
            types[c] = "TEXT"
    return out, types


def write_sqlite(enriched: pd.DataFrame, rejected: pd.DataFrame, run_dir: Path) -> Path:
    """customer_360 (enriched; dq_flags as JSON text) and rejected_records (without raw_record)."""
    path = run_dir / DB_NAME
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for table, frame in ((CUSTOMER_TABLE, enriched),
                             (REJECTED_TABLE, rejected.drop(columns=["raw_record"]))):
            data, types = _sqlite_frame(frame)
            data.to_sql(table, con, index=False, dtype=types)
        con.execute(f"CREATE UNIQUE INDEX idx_{CUSTOMER_TABLE}_id ON {CUSTOMER_TABLE}(customer_id)")
        con.commit()
    finally:
        con.close()
    return path


def list_outputs(run_dir: Path) -> List[Dict[str, Any]]:
    return [{"path": p.relative_to(run_dir).as_posix(), "bytes": p.stat().st_size}
            for p in sorted(run_dir.rglob("*")) if p.is_file()]


def publish_latest(run_dir: Path, latest_dir: Path) -> None:
    """Copy a finished run into a temp dir next to latest/, then swap it in.

    latest/ is only replaced once the copy is complete. If the swap fails, the previous
    latest/ is restored. Temp/backup dirs are removed in every case.
    """
    parent = latest_dir.parent
    tmp = parent / f".{latest_dir.name}.tmp-{run_dir.name}"
    old = parent / f".{latest_dir.name}.old-{run_dir.name}"
    for leftover in (tmp, old):
        if leftover.exists():
            shutil.rmtree(leftover)
    try:
        shutil.copytree(run_dir, tmp)
        if latest_dir.exists():
            os.replace(latest_dir, old)
        try:
            os.replace(tmp, latest_dir)
        except OSError:
            if old.exists() and not latest_dir.exists():
                os.replace(old, latest_dir)
            raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        if old.exists() and latest_dir.exists():
            shutil.rmtree(old, ignore_errors=True)
    log.info("published %s -> %s", run_dir.name, latest_dir)
