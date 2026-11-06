"""Read raw JSON / CSV / XML sources into string-only DataFrames.

Every value is kept as text (or None when the source has no value) so that
validation, not parsing, decides what is wrong. Each row gets `source_row`
(1-based position in the source file).
"""
from __future__ import annotations

import csv
import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from src.config import SourceSpec

log = logging.getLogger("ingest")


def _to_text(value: Any) -> Optional[str]:
    """Keep None as None; render everything else as its raw text."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def read_json(path: Path, spec: SourceSpec) -> List[Dict[str, Optional[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise ValueError(f"{path.name}: expected a JSON array of objects")
    return [{k: _to_text(v) for k, v in rec.items()} for rec in data]


def read_csv(path: Path, spec: SourceSpec) -> List[Dict[str, Optional[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = []
        for rec in reader:
            if None in rec:
                raise ValueError(f"{path.name} line {reader.line_num}: more values than header columns")
            rows.append(dict(rec))
    return rows


def read_xml(path: Path, spec: SourceSpec) -> List[Dict[str, Optional[str]]]:
    if not spec.record_tag:
        raise ValueError(f"{path.name}: 'record_tag' is required for XML sources")
    root = ET.parse(path).getroot()
    rows = []
    for elem in root.iter(spec.record_tag):
        # An empty element (<customerId/>) becomes "", an absent one is simply missing (None later).
        rows.append({child.tag: (child.text or "") for child in elem})
    return rows


READERS: Dict[str, Callable[[Path, SourceSpec], List[Dict[str, Optional[str]]]]] = {
    ".json": read_json,
    ".csv": read_csv,
    ".xml": read_xml,
}


def ingest_source(path: Path, spec: SourceSpec) -> pd.DataFrame:
    """Read one source file. All columns are object dtype holding str or None."""
    rows = READERS[path.suffix.lower()](path, spec)
    df = pd.DataFrame.from_records(rows).astype(object)
    df = df.where(df.notna(), None)
    df.insert(0, "source_row", range(1, len(df) + 1))
    log.info("ingested %-22s rows=%d columns=%s", path.name, len(df),
             [c for c in df.columns if c != "source_row"])
    return df
