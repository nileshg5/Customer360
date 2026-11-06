"""Customer 360 pipeline CLI.

Usage: python main.py [--config config.json] [--as-of YYYY-MM-DD]
Exit codes: 0 success, 2 bad config / missing input / run_id clash, 3 aborted on reject rate.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from pydantic import ValidationError

from src.config import Settings, load_config
from src.enrich import enrich
from src.ingest import ingest_source
from src.insights import build_insights
from src.logging_setup import new_run_id, setup_logging
from src.merge import merge
from src.publish import (list_outputs, publish_latest, sha256_file, write_customer_outputs,
                         write_dq_outputs, write_insights, write_json, write_sqlite)
from src.standardize import standardize
from src.validate import (RejectRateExceeded, add_customer_level_flags, check_reject_rate,
                          validate)

PROJECT_ROOT = Path(__file__).resolve().parent
log = logging.getLogger("pipeline")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def timed(timings: Dict[str, float], stage: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    timings[stage] = round(time.perf_counter() - start, 3)


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Customer 360 batch pipeline.")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json",
                   help="path to config.json (default: ./config.json)")
    p.add_argument("--as-of", dest="as_of", default=None,
                   help="override as_of_date (YYYY-MM-DD)")
    return p.parse_args(argv)


def check_inputs(cfg: Settings) -> List[Path]:
    """Return the input file paths that do not exist."""
    return [cfg.source_path(n) for n in cfg.input_sources if not cfg.source_path(n).is_file()]


def stage_sources(cfg: Settings, run_id: str, staged_dir: Path) -> Dict[str, pd.DataFrame]:
    """Ingest + standardize every input source; write each to staged/<system>.csv."""
    ingested_at = _utc_now()
    staged_dir.mkdir(parents=True, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    for name in cfg.input_sources:
        spec = cfg.sources[name]
        raw = ingest_source(cfg.source_path(name), spec)
        frames[spec.system] = standardize(raw, spec, name, run_id, ingested_at)
        frames[spec.system].to_csv(staged_dir / f"{spec.system}.csv", index=False)
    return frames


def build_manifest(cfg: Settings, run_id: str, status: str, started_at: str,
                   timings: Dict[str, float], stage_rows: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    inputs = []
    for name in cfg.input_sources:
        path, system = cfg.source_path(name), cfg.sources[name].system
        inputs.append({"file": name, "system": system, "sha256": sha256_file(path),
                       "bytes": path.stat().st_size, "rows": stage_rows["ingest"][system]})
    return {
        "run_id": run_id,
        "status": status,
        "as_of_date": cfg.as_of_date.isoformat(),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(sum(timings.values()), 3),
        "inputs": inputs,
        "stage_rows": stage_rows,
        "timings_seconds": timings,
        "config": cfg.model_dump(mode="json"),
        "outputs": [o for o in list_outputs(run_dir) if o["path"] != "run_manifest.json"],
    }


def run_pipeline(cfg: Settings, run_id: str) -> int:
    run_dir = cfg.output_dir / run_id
    if run_dir.exists():
        log.error("run directory already exists, refusing to overwrite: %s", run_dir)
        return 2
    started_at, timings, rows = _utc_now(), {}, {}

    with timed(timings, "ingest_standardize"):
        staged = stage_sources(cfg, run_id, run_dir / "staged")
    rows["ingest"] = {s: len(df) for s, df in staged.items()}

    with timed(timings, "validate"):
        result = validate(staged, cfg, run_id)
    rows["validate"] = {"clean": {s: len(df) for s, df in result.clean.items()},
                        "rejected": result.report["rejected_unique_records"],
                        "quarantined_values": int((result.rejected.action == "quarantined_value").sum())}
    write_dq_outputs(result.report, result.rejected, run_dir)
    try:
        check_reject_rate(result.report)
    except RejectRateExceeded as exc:
        log.error("ABORT before publishing: %s (see %s)", exc, run_dir / "dq_report.json")
        write_json(build_manifest(cfg, run_id, "aborted", started_at, timings, rows, run_dir),
                   run_dir / "run_manifest.json")
        return 3

    with timed(timings, "merge"):
        customers = merge(result.clean, cfg)
    with timed(timings, "enrich"):
        enriched = enrich(customers, cfg)
    rows["merge"], rows["enrich"] = len(customers), len(enriched)
    add_customer_level_flags(result.report, enriched)

    with timed(timings, "insights"):
        insights = build_insights(enriched, result.report, cfg)
    rows["insights"] = {name: len(df) for name, df in insights.items()}

    with timed(timings, "write_outputs"):
        write_dq_outputs(result.report, result.rejected, run_dir)
        write_customer_outputs(enriched, run_dir)
        write_insights(insights, run_dir)
        write_sqlite(enriched, result.rejected, run_dir)
    write_json(build_manifest(cfg, run_id, "success", started_at, timings, rows, run_dir),
               run_dir / "run_manifest.json")

    publish_latest(run_dir, cfg.output_dir / cfg.paths.latest_dir_name)
    log.info("stage summary | ingest=%s | clean=%s rejected=%d | customers=%d | insights=%d | %.2fs",
             rows["ingest"], rows["validate"]["clean"], rows["validate"]["rejected"],
             rows["enrich"], len(insights), sum(timings.values()))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cfg = load_config(args.config, as_of_override=args.as_of)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"ERROR: invalid config {args.config}: {exc}", file=sys.stderr)
        return 2

    run_id = new_run_id()
    setup_logging(cfg.log_file, run_id, secret_env_vars=[cfg.llm.api_key_env])
    log.info("config loaded from %s | as_of_date=%s", args.config, cfg.as_of_date)
    log.info("input_sources=%s", cfg.input_sources)

    missing = check_inputs(cfg)
    if missing:
        log.error("missing input files: %s", [str(m) for m in missing])
        return 2
    return run_pipeline(cfg, run_id)


if __name__ == "__main__":
    sys.exit(main())
