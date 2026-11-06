"""Ask AI: chat over the read-only SQLite table, preset chips, SQL / table / chart / summary."""
from __future__ import annotations

import os

import streamlit as st

from src.qa.engine import QaEngine, QaResult, build_engine
from src.qa.llm_client import client_from_env
from src.qa.presets import PRESETS
from src.ui import components as ui
from src.ui.theme import SERIES

STATUS_TONE = {"ok": "good", "refused": "neutral", "blocked": "warning", "no_llm": "neutral", "error": "critical"}
STATUS_LABEL = {"ok": "Answered", "refused": "Declined", "blocked": "Blocked by safety check",
                "no_llm": "AI not configured", "error": "Could not answer"}


@st.cache_resource(show_spinner=False, max_entries=2)
def _engine(_cfg, run_id: str, has_key: bool) -> QaEngine:
    """One engine per published run (and key presence). It opens SQLite per query, so it never
    holds output/latest/ open."""
    return build_engine(_cfg, client_from_env(_cfg))


def _render_result(r: QaResult, idx: int) -> None:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(r.question)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        meta = [ui.badge(STATUS_LABEL.get(r.status, r.status), STATUS_TONE.get(r.status, "neutral"))]
        if r.model:
            meta.append(f'<span class="c360-chip">model <b>{ui.esc(r.model)}</b></span>')
        meta.append(f'<span class="c360-chip"><b>{r.latency_ms:,}</b> ms</span>')
        if r.attempts > 1:
            meta.append(f'<span class="c360-chip">self-repaired ({r.attempts} attempts)</span>')
        ui.md('<div class="c360-chips" style="margin-bottom:8px">' + "".join(meta) + "</div>")
        if r.status == "ok":
            st.markdown(r.summary or "")
        else:
            st.markdown(r.message)
        if r.sql:
            with st.expander("SQL that was run", icon=":material/code:"):
                st.code(r.sql, language="sql")
        if r.columns:
            df = r.frame()
            if r.chart and r.chart.get("type") == "bar":
                ui.plot(ui.hbar(df, r.chart["x"], r.chart["y"], color=SERIES[0], value_fmt=",.6g"),
                        height=max(180, 34 * len(df)), key=f"ai_chart_{idx}")
            st.dataframe(df, hide_index=True, height=min(400, 38 + 35 * len(df)))
            if len(r.rows) >= 200:
                st.caption("Result capped at 200 rows.")


def _run_preset() -> None:
    pid = st.session_state.get("ai_preset")
    if pid:
        engine = st.session_state["_ai_engine"]
        st.session_state.setdefault("ai_chat", []).append(engine.run_preset(pid))
        st.session_state["ai_preset"] = None


def render(ctx) -> None:
    cfg = ctx.cfg
    has_key = bool((os.environ.get(cfg.llm.api_key_env) or "").strip())
    try:
        engine = _engine(cfg, ctx.bundle.run_id, has_key)
    except FileNotFoundError:
        ui.empty_state("Database not found", "Run python main.py to publish output/latest/customer360.db.")
        return
    st.session_state["_ai_engine"] = engine
    chat = st.session_state.setdefault("ai_chat", [])

    if not has_key:
        ui.md('<div class="c360-panel" style="padding:14px 18px;margin-bottom:12px">'
              + ui.badge("AI not configured", "neutral")
              + '<span style="color:var(--text-2);margin-left:8px">Set <code>GROQ_API_KEY</code> in the environment or '
                '<code>.env</code> to ask free-form questions. The preset questions below work without it.</span></div>')
    st.caption(f"Answers come from read-only SQL over the whole dataset (sidebar filters don't apply). "
               f"Model: {cfg.llm.model} (fallback {cfg.llm.fallback_model}). Emails are never available to the AI.")

    st.pills("Preset questions (fixed, tested SQL)", [p["id"] for p in PRESETS], key="ai_preset",
             format_func=lambda pid: next(p["question"] for p in PRESETS if p["id"] == pid),
             on_change=_run_preset)

    question = st.chat_input("Ask about customers, revenue, segments, support or data quality", key="ai_input")
    if question:
        with st.status("Thinking...", expanded=False) as status:
            result = engine.ask(question, on_status=lambda m: status.update(label=m, state="running"))
            status.update(label="Done" if result.status == "ok" else STATUS_LABEL.get(result.status, "Done"),
                          state="complete" if result.status in ("ok", "refused", "no_llm") else "error")
        chat.append(result)

    if not chat:
        ui.empty_state("Ask a question", "For example: Which city has the most Gold customers?")
    for i, r in enumerate(chat):
        _render_result(r, i)
    if chat and st.button("Clear conversation", icon=":material/delete_sweep:", key="ai_clear"):
        st.session_state["ai_chat"] = []
        st.rerun()
