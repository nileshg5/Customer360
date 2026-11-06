"""Run-output snapshot used to prove idempotency.

Each file is hashed after removing only the values that are expected to change between
runs: run_id, ingested_at and the manifest's wall-clock fields.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

VOLATILE_KEYS = {"run_id", "ingested_at", "started_at", "finished_at", "duration_seconds",
                 "timings_seconds"}


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _normalised_bytes(path: Path) -> bytes:
    if path.suffix == ".json":
        return json.dumps(_strip(json.loads(path.read_text(encoding="utf-8"))), sort_keys=True).encode()
    if path.suffix == ".csv":
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))
        return json.dumps(_strip(rows), sort_keys=True).encode()
    if path.suffix == ".db":
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            dump = {}
            for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
                cur = con.execute(f"SELECT * FROM {table}")
                cols = [d[0] for d in cur.description]
                dump[table] = _strip([dict(zip(cols, r)) for r in cur.fetchall()])
            return json.dumps(dump, sort_keys=True, default=str).encode()
        finally:
            con.close()
    return path.read_bytes()


def snapshot(run_dir: Path) -> Dict[str, str]:
    """{relative path: sha256 of normalised content} for every file in a run directory."""
    return {p.relative_to(run_dir).as_posix(): hashlib.sha256(_normalised_bytes(p)).hexdigest()
            for p in sorted(run_dir.rglob("*")) if p.is_file()}


def raw_snapshot(run_dir: Path) -> Dict[str, str]:
    return {p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.rglob("*")) if p.is_file()}
