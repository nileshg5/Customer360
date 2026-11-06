"""Dashboard smoke tests with streamlit.testing AppTest (no browser, no network)."""
from __future__ import annotations

import html
import json
import re
import shutil

import pytest
from streamlit.testing.v1 import AppTest

import main
from src.ui.data import current_run_id, mask_email, with_retry
from tests.conftest import PROJECT_ROOT

PAGES = {
    "overview": ["Overview", "Revenue by city", "Customer segments", "DQ pass rate"],
    "revenue": ["Revenue & Regions", "Revenue by city", "Customer count by city", "Top 10 customers by revenue"],
    "segments": ["Segments & Customers", "Customers per segment", "Segment summary", "Why this risk score"],
    "support": ["Support", "Ticket distribution", "Percentile threshold"],
    "data-quality": ["Data Quality", "From source rows to customers", "Issues by rule", "Reconciliation by source",
                     "Errors vs warnings"],
    "ask-ai": ["Ask AI", "Preset questions"],
    "pipeline": ["Pipeline Run", "Rows at each stage", "Stage timings", "Downloads"],
}
RAW_EMAIL = re.compile(r"\buser\d+@example\.com\b")


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """Run the real pipeline into a temp project and return its config path."""
    root = tmp_path_factory.mktemp("dash")
    shutil.copy(PROJECT_ROOT / "config.json", root / "config.json")
    shutil.copytree(PROJECT_ROOT / "data" / "raw", root / "data" / "raw")
    shutil.copytree(PROJECT_ROOT / "src" / "qa", root / "src" / "qa")       # schema_doc.json path in config
    assert main.main(["--config", str(root / "config.json")]) == 0
    return root / "config.json"


def app(config_path, page=None) -> AppTest:
    script = f"from src.ui.shell import main\nmain(page={page!r}, config_path={str(config_path)!r})"
    return AppTest.from_string(script, default_timeout=60)


def page_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    labels = [w.label for kind in ("multiselect", "pills", "text_input", "select_slider", "selectbox")
              for w in getattr(at, kind)]
    return html.unescape("\n".join(str(p) for p in parts + labels))


def shown_values(at: AppTest) -> str:
    return "\n".join(df.value.to_string() for df in at.dataframe)


@pytest.mark.parametrize("page", list(PAGES))
def test_every_page_renders(published, page, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")                # no network from the dashboard tests
    at = app(published, page).run()
    assert not at.exception, [e.value for e in at.exception]
    text = page_text(at)
    for expected in PAGES[page]:
        assert expected in text, f"{expected!r} missing on {page}"
    assert "As of <b>2026-10-28</b>" in text               # header on every page
    assert RAW_EMAIL.search(shown_values(at) + text) is None


def test_navigation_app_runs(published, monkeypatch):
    monkeypatch.setenv("C360_CONFIG", str(published))
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "Revenue by city" in page_text(at)              # default page = Overview


def test_filters_apply_and_show_chips(published, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = app(published, "overview").run()
    at.sidebar.multiselect(key="f_segment").set_value(["Gold"]).run()
    text = page_text(at)
    assert "Segment: Gold" in text and "318 of 500 customers" in text
    assert '<div class="value">318</div>' in text          # KPI follows the filter
    at.sidebar.multiselect(key="f_city").set_value(["Pune"]).run()
    assert "City: Pune" in page_text(at)
    assert not at.exception


def test_empty_filter_result(published):
    at = app(published, "revenue").run()
    at.sidebar.multiselect(key="f_segment").set_value(["Unknown"]).run()   # only C0420 (Kolkata)
    at.sidebar.multiselect(key="f_city").set_value(["Pune"]).run()
    assert not at.exception
    assert "No customers match the filters" in page_text(at)


def test_customer_search_and_detail(published):
    at = app(published, "segments").run()
    at.text_input(key="seg_search").set_value("C0420").run()
    assert not at.exception
    text = page_text(at)
    assert "Vikram Jain 420" in text and "NEG_REVENUE" in text and "u***@example.com" in text
    assert RAW_EMAIL.search(shown_values(at) + text) is None


def test_dq_reason_filter(published):
    at = app(published, "data-quality").run()
    at.multiselect(key="dq_reason").set_value(["DUP_ID"]).run()
    tables = [df.value for df in at.dataframe if "rule_codes" in df.value.columns]
    assert len(tables) == 1 and tables[0]["rule_codes"].tolist() == ["DUP_ID"]


def test_ask_ai_without_key_uses_presets(published, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = app(published, "ask-ai").run()
    assert "AI not configured" in page_text(at)
    at.pills(key="ai_preset").set_value("dq_issues").run()
    assert not at.exception
    assert any(set(df.value.columns) == {"rule_codes", "action", "records"} for df in at.dataframe)
    at.chat_input(key="ai_input").set_value("How many Gold customers?").run()
    assert "GROQ_API_KEY" in page_text(at) + "\n".join(m.value for m in at.markdown)


def test_missing_latest_shows_friendly_screen(tmp_path):
    shutil.copy(PROJECT_ROOT / "config.json", tmp_path / "config.json")
    at = app(tmp_path / "config.json", "overview").run()
    assert not at.exception
    text = page_text(at)
    assert "No pipeline output yet" in text and "python main.py" in text


# ---------- data layer ----------

def test_mask_email():
    assert mask_email("user125@example.com") == "u***@example.com"
    assert mask_email(None) == "-" and mask_email("not-an-email") == "-"


def test_with_retry_survives_brief_swap():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FileNotFoundError("latest/ is being swapped")
        return "ok"
    assert with_retry(flaky, attempts=4, delay=0) == "ok" and calls["n"] == 3


def test_current_run_id_requires_successful_manifest(tmp_path):
    latest = tmp_path / "latest"
    assert current_run_id(latest) is None
    latest.mkdir()
    (latest / "run_manifest.json").write_text(json.dumps({"run_id": "R1", "status": "aborted"}))
    assert current_run_id(latest) is None
    (latest / "run_manifest.json").write_text(json.dumps({"run_id": "R1", "status": "success"}))
    assert current_run_id(latest) == "R1"


def test_exec_summary_button_writes_summary_md(published, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = app(published, "overview").run()
    at.button(key="exec_btn").click().run()
    assert not at.exception
    out = published.parent / "output" / "summary.md"
    assert out.is_file() and "executive summary" in out.read_text(encoding="utf-8")
    assert "written by template" in out.read_text(encoding="utf-8")
