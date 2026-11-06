"""Data Quality: funnel, rule breakdown by severity, per-source reconciliation, quarantine table."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui import components as ui
from src.ui.theme import COLORS, SERIES, SEVERITY_COLORS


def _rule_table(ctx) -> pd.DataFrame:
    dq, rules = ctx.bundle.dq, ctx.cfg.validation.rules
    counts = dict(dq["by_rule"])
    counts.update({k: v for k, v in dq.get("customer_level_flags", {}).items() if v})
    rows = [{"rule": code, "records": n, "severity": rules[code].severity, "action": rules[code].action,
             "level": "customer" if code in dq.get("customer_level_flags", {}) else "source row"}
            for code, n in counts.items() if code in rules]
    return pd.DataFrame(rows).sort_values(["severity", "records", "rule"], ascending=[True, False, True])


def render(ctx) -> None:
    dq = ctx.bundle.dq
    rec = dq["reconciliation"]
    st.caption("Source-level data quality for the whole run. The sidebar customer filters do not apply here.")
    ui.kpi_row([
        {"label": "Source records", "value": ui.fmt_num(dq["total_records"]), "sub": "CRM + orders + tickets"},
        {"label": "Error-free", "value": ui.fmt_num(dq["successful_records"]),
         "sub": f"{dq['successful_records'] / dq['total_records'] * 100:.2f}% pass rate", "tone": "good"},
        {"label": "Rejected rows", "value": ui.fmt_num(dq["rejected_unique_records"]),
         "sub": f"reject rate {dq['reject_rate'] * 100:.2f}% (abort at {dq['max_reject_rate'] * 100:.0f}%)", "tone": "critical"},
        {"label": "Kept with errors", "value": ui.fmt_num(dq["flagged_kept_records"]), "sub": "value nulled or flagged",
         "tone": "warning"},
        {"label": "Warning rows", "value": ui.fmt_num(dq["warning_records"]), "sub": "kept, informational", "tone": "warning"},
        {"label": "Customers in 360", "value": ui.fmt_num(dq["customers_in_360"]), "tone": "accent"},
    ])

    left, right = st.columns([2, 3], gap="medium")
    with left, st.container(border=True):
        ui.section("From source rows to customers", "Each step is explained by the rules on the right")
        f = ctx.bundle.insights.get("dq_funnel")
        labels = {"source_records": "Source rows", "after_rejections": "After rejections",
                  "error_free_records": "Error-free rows", "customers_in_360": "Customers (merged)"}
        fig = go.Figure(go.Funnel(
            y=[labels.get(s, s) for s in f["stage"]], x=f["count"], textinfo="value",
            marker=dict(color=[SERIES[0]] * 3 + [SERIES[1]], line=dict(width=0)),
            textfont=dict(color="#FFFFFF"), connector=dict(fillcolor=COLORS["card_hi"]),
            hovertemplate="%{y}: %{x:,}<extra></extra>"))
        fig.update_layout(yaxis=dict(showgrid=False), margin=dict(l=8, r=8, t=6, b=6))
        ui.plot(fig, height=300, key="dq_funnel")
        st.caption(f"{rec['clean']:,} clean rows become {dq['customers_in_360']:,} customers because the three "
                   "sources are merged into one row per customer.")
    with right, st.container(border=True):
        ui.section("Issues by rule", "Rows affected per rule; colour = severity")
        rules = _rule_table(ctx)
        ui.legend({"Error": SEVERITY_COLORS["error"], "Warning": SEVERITY_COLORS["warning"]})
        ui.plot(ui.hbar(rules, "rule", "records",
                        color_by={r.rule: SEVERITY_COLORS[r.severity] for r in rules.itertuples()},
                        hover_name="Rows"), height=260, key="dq_rules")

    ui.section("Reconciliation by source", "input = clean + rejected must hold for every source")
    src = pd.DataFrame([{"source": k, **{c: v[c] for c in ("input", "clean", "rejected", "reject_rate", "successful",
                                                            "flagged_kept", "warning_records")},
                         "status": "Balanced" if v["balanced"] else "NOT balanced"}
                        for k, v in dq["by_source"].items()])
    src["reject_rate"] = src["reject_rate"] * 100
    ui.md('<div style="margin:-4px 0 8px">' + ui.badge(
        f"Overall: {rec['input']} = {rec['clean']} clean + {rec['rejected']} rejected"
        + (" · balanced" if rec["balanced"] else " · NOT balanced"), "good" if rec["balanced"] else "critical") + "</div>")
    st.dataframe(src, hide_index=True, column_config={
        "source": "Source file", "input": "Input", "clean": "Clean", "rejected": "Rejected",
        "reject_rate": st.column_config.NumberColumn("Reject rate", format="%.2f%%"),
        "successful": "Error-free", "flagged_kept": "Kept with errors", "warning_records": "Warnings",
        "status": "Reconciliation"})

    left, right = st.columns([3, 2], gap="medium")
    with left:
        ui.section("Quarantine: rejected rows and quarantined values", "Raw values are kept in rejected_records.csv")
        rej = ctx.bundle.rejected.copy()
        rej["customer_id"] = rej["customer_id"].fillna("-")
        codes = sorted({c for v in rej["rule_codes"] for c in str(v).split("|")})
        pick = st.multiselect("Reason", codes, key="dq_reason", placeholder="All reasons")
        if pick:
            rej = rej[rej["rule_codes"].map(lambda v: any(c in str(v).split("|") for c in pick))]
        if rej.empty:
            ui.empty_state("No records for this reason")
        else:
            st.dataframe(rej, hide_index=True, column_config={
                "source_file": "Source file", "source_row": "Row", "customer_id": "Customer ID",
                "rule_codes": "Rule codes", "action": "Action"})
    with right, st.container(border=True):
        ui.section("Errors vs warnings")
        ui.md(f'''<div style="color:var(--text-2); font-size:.88rem; line-height:1.55">
          {ui.badge("Error", "critical")} a value is wrong or missing. The row is <b>rejected</b> (bad or duplicate ID,
          missing mandatory field, orphan) or <b>kept with the bad value removed</b> (bad email, negative revenue,
          unparseable number or date). Errors reduce <i>successful_records</i>.<br><br>
          {ui.badge("Warning", "warning")} the row is usable but looks inconsistent (0 tickets with a ticket date,
          shared email, missing date, no order or ticket row). Warnings are reported but do not reduce
          <i>successful_records</i>.<br><br>
          Thresholds, actions and severities live in <code>config.json</code>.</div>''')
    with st.expander("All rules and their handling"):
        st.dataframe(pd.DataFrame([{"rule": k, "severity": v.severity, "action": v.action}
                                   for k, v in ctx.cfg.validation.rules.items()]), hide_index=True)
