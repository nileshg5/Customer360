"""Load and validate config.json into typed settings."""
from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_EXTENSIONS = {".json", ".csv", ".xml"}
RuleAction = Literal["reject", "flag", "null_and_flag", "null_and_quarantine"]
REQUIRED_RULES = {
    "EMPTY_ID", "BAD_ID", "DUP_ID", "DUP_EMAIL", "BAD_EMAIL", "MISSING_FIELD", "NULL_ORDERS",
    "NULL_TICKETS", "NEG_REVENUE", "BAD_DATE_UNPARSEABLE", "BAD_DATE_FUTURE", "CROSS_DATE",
    "ORPHAN", "ZERO_TICKETS_WITH_DATE", "NULL_DATE", "NO_ORDER_ROW", "NO_TICKET_ROW", "BAD_NUMBER",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw_dir: str
    output_dir: str
    latest_dir_name: str
    log_file: str
    qa_log_file: str


class SourceSpec(_Strict):
    system: Literal["crm", "orders", "support"]
    field_map: Dict[str, str]
    record_tag: Optional[str] = None

    @field_validator("field_map")
    @classmethod
    def _maps_customer_id(cls, v: Dict[str, str]) -> Dict[str, str]:
        if "customer_id" not in v.values():
            raise ValueError("field_map must map one source field to 'customer_id'")
        return v


class RuleSpec(_Strict):
    action: RuleAction
    severity: Literal["error", "warning"]


class Validation(_Strict):
    id_regex: str
    email_regex: str
    date_format: str
    mandatory_fields: List[str]
    max_reject_rate: float = Field(gt=0, lt=1)
    rules: Dict[str, RuleSpec]

    @field_validator("id_regex", "email_regex")
    @classmethod
    def _compiles(cls, v: str) -> str:
        re.compile(v)
        return v

    @field_validator("rules")
    @classmethod
    def _all_rules_present(cls, v: Dict[str, RuleSpec]) -> Dict[str, RuleSpec]:
        missing = REQUIRED_RULES - set(v)
        unknown = set(v) - REQUIRED_RULES
        if missing or unknown:
            raise ValueError(f"rules mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")
        return v

    def action(self, rule: str) -> str:
        return self.rules[rule].action

    def is_error(self, rule: str) -> bool:
        return self.rules[rule].severity == "error"


class Merge(_Strict):
    missing_row_defaults: Dict[str, float]


class Segments(_Strict):
    mode: Literal["fixed", "percentile"]
    gold_min_exclusive: float
    silver_min_inclusive: float
    percentile_cuts: List[float] = Field(min_length=2, max_length=2)
    null_label: str

    @model_validator(mode="after")
    def _ordered(self) -> "Segments":
        if not self.silver_min_inclusive < self.gold_min_exclusive:
            raise ValueError("silver_min_inclusive must be below gold_min_exclusive")
        lo, hi = self.percentile_cuts
        if not 0 < lo < hi < 1:
            raise ValueError("percentile_cuts must satisfy 0 < low < high < 1")
        return self


class Support(_Strict):
    high_ticket_threshold: int = Field(ge=0)
    default_percentile: float = Field(gt=0, lt=1)
    report_percentiles: List[float] = Field(min_length=1)

    @field_validator("report_percentiles")
    @classmethod
    def _in_unit_interval(cls, v: List[float]) -> List[float]:
        if not all(0 < p < 1 for p in v):
            raise ValueError("report_percentiles must be between 0 and 1")
        return sorted(v)


class RiskWeights(_Strict):
    order_recency: float = Field(ge=0)
    ticket_count: float = Field(ge=0)
    ticket_recency: float = Field(ge=0)
    low_ltv: float = Field(ge=0)


class Risk(_Strict):
    weights: RiskWeights
    order_recency_cap_days: int = Field(gt=0)
    ticket_count_cap: int = Field(gt=0)
    ticket_recency_window_days: int = Field(gt=0)
    band_medium_min: float
    band_high_min: float

    @model_validator(mode="after")
    def _check(self) -> "Risk":
        total = sum(self.weights.model_dump().values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"risk weights must sum to 1.0, got {total}")
        if not 0 < self.band_medium_min < self.band_high_min <= 100:
            raise ValueError("risk bands must satisfy 0 < medium_min < high_min <= 100")
        return self


class RfmBounds(_Strict):
    r_min: Optional[int] = None
    r_max: Optional[int] = None
    f_min: Optional[int] = None
    f_max: Optional[int] = None
    m_min: Optional[int] = None
    m_max: Optional[int] = None


class RfmRule(_Strict):
    label: str
    all: RfmBounds
    any: Optional[RfmBounds] = None


class Rfm(_Strict):
    bins: int = Field(ge=2)
    rules: List[RfmRule]
    default_label: str
    unscored_label: str


class Llm(_Strict):
    provider: Literal["groq"]
    model: str
    fallback_model: Optional[str] = None
    api_key_env: str
    reasoning_effort: Optional[Literal["low", "medium", "high"]] = None
    timeout_seconds: float = Field(gt=0)
    max_retry_wait_seconds: float = Field(ge=0)
    sql_timeout_seconds: float = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    max_result_rows: int = Field(gt=0)
    max_summary_rows: int = Field(gt=0)
    hidden_columns: List[str]
    summary_excluded_columns: List[str]
    allowed_tables: List[str] = Field(min_length=1)
    schema_doc: str


class Settings(_Strict):
    input_sources: List[str] = Field(min_length=1)
    as_of_date: date
    paths: Paths
    sources: Dict[str, SourceSpec]
    validation: Validation
    merge: Merge
    segments: Segments
    support: Support
    risk: Risk
    rfm: Rfm
    llm: Llm
    project_root: Path = Field(exclude=True)

    @model_validator(mode="after")
    def _sources_consistent(self) -> "Settings":
        for name in self.input_sources:
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"unsupported input format: {name}")
            if name not in self.sources:
                raise ValueError(f"input source '{name}' has no entry under 'sources'")
        systems = [self.sources[n].system for n in self.input_sources]
        if sorted(systems) != ["crm", "orders", "support"]:
            raise ValueError(f"need exactly one crm, orders and support source, got {systems}")
        return self

    def resolve(self, relative: str) -> Path:
        """Resolve a config path relative to the project root."""
        return (self.project_root / relative).resolve()

    @property
    def raw_dir(self) -> Path:
        return self.resolve(self.paths.raw_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    @property
    def log_file(self) -> Path:
        return self.resolve(self.paths.log_file)

    def source_path(self, name: str) -> Path:
        return self.raw_dir / name

    def source_for(self, system: str) -> str:
        """Return the input_sources file name for a system (crm / orders / support)."""
        return next(n for n in self.input_sources if self.sources[n].system == system)


def load_config(path: Path, as_of_override: Optional[str] = None) -> Settings:
    """Read config.json, apply an optional as_of_date override, and validate."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if as_of_override:
        raw["as_of_date"] = as_of_override
    raw["project_root"] = path.parent
    return Settings.model_validate(raw)
