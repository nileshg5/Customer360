"""Logging to console and a file, with run_id on every line and secrets redacted."""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | run=%(run_id)s | %(name)s | %(message)s"


def new_run_id() -> str:
    """UTC timestamp run id, e.g. 20261106T101500Z."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class _ContextFilter(logging.Filter):
    """Stamps run_id on each record and masks any configured secret values."""

    def __init__(self, run_id: str, secret_env_vars: Iterable[str]) -> None:
        super().__init__()
        self.run_id = run_id
        self.secret_env_vars = list(secret_env_vars)

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        secrets = [os.environ.get(v) for v in self.secret_env_vars]
        secrets = [s for s in secrets if s and len(s) >= 8]
        if secrets:
            msg = record.getMessage()
            for s in secrets:
                msg = msg.replace(s, "***REDACTED***")
            record.msg, record.args = msg, None
        return True


def setup_logging(log_file: Path, run_id: str, secret_env_vars: Iterable[str] = (),
                  level: int = logging.INFO) -> None:
    """Configure the root logger. Safe to call more than once (handlers are replaced)."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    ctx = _ContextFilter(run_id, secret_env_vars)
    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%SZ")
    formatter.converter = time.gmtime
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(log_file, mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        handler.addFilter(ctx)
        root.addHandler(handler)
    root.setLevel(level)
