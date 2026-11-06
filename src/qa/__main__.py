"""CLI: python -m src.qa "question" | --preset ID | --list-presets"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.logging_setup import setup_logging
from src.qa.engine import build_engine
from src.qa.llm_client import client_from_env
from src.qa.presets import PRESETS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description="Ask the Customer 360 data a question.")
    p.add_argument("question", nargs="?")
    p.add_argument("--preset", help="run a preset question by id")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.list_presets:
        for pr in PRESETS:
            print(f"{pr['id']:24} {pr['question']}")
        return 0
    if not args.question and not args.preset:
        p.error("give a question or --preset ID")

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(args.config)
    setup_logging(cfg.log_file, "qa-cli", secret_env_vars=[cfg.llm.api_key_env], level=30)
    engine = build_engine(cfg, client_from_env(cfg))
    r = engine.run_preset(args.preset) if args.preset else engine.ask(args.question)

    print(f"Q: {r.question}\nstatus: {r.status} | model: {r.model} | attempts: {r.attempts} | {r.latency_ms} ms")
    if r.message:
        print(f"message: {r.message}")
    if r.sql:
        print(f"SQL: {r.sql}")
    if r.columns:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(r.frame().head(10).to_string(index=False))
            if len(r.rows) > 10:
                print(f"... {len(r.rows)} rows")
    if r.summary:
        print(f"Summary: {r.summary}")
    return 0 if r.status in ("ok", "refused", "no_llm") else 1


if __name__ == "__main__":
    sys.exit(main())
