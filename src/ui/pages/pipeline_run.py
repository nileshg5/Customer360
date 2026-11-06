"""Pipeline Run: run metadata, stage row counts, timings, inputs, config used and downloads."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import components as ui
from src.ui.data import read_bytes

# Files without personal data. customer_360*.json and the DB contain emails, so they are not offered here.
DOWNLOADS = [("dq_report.json", "application/json"), ("rejected_records.csv", "text/csv"),
             ("run_manifest.json", "application/json")]


def render(ctx) -> None:
    m = ctx.bundle.manifest
    rows = m["stage_rows"]
    ui.kpi_row([
        {"label": "Status", "value": m["status"].capitalize(), "tone": "good" if m["status"] == "success" else "critical"},
        {"label": "As-of date", "value": m["as_of_date"], "sub": "fixed in config.json"},
        {"label": "Duration", "value": f"{m['duration_seconds']:.2f}s", "sub": f"started {m['started_at']}"},
        {"label": "Customers published", "value": ui.fmt_num(rows["enrich"]), "tone": "accent"},
    ])

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        ui.section("Rows at each stage")
        v = rows["validate"]
        stages = pd.DataFrame([
            {"stage": "Ingest", "crm": rows["ingest"]["crm"], "orders": rows["ingest"]["orders"], "support": rows["ingest"]["support"]},
            {"stage": "Validate (clean)", "crm": v["clean"]["crm"], "orders": v["clean"]["orders"], "support": v["clean"]["support"]},
        ])
        st.dataframe(stages, hide_index=True)
        ui.md(ui.chips([f"Rejected <b>{v['rejected']}</b>", f"Quarantined values <b>{v['quarantined_values']}</b>",
                        f"Merged <b>{rows['merge']}</b>", f"Enriched <b>{rows['enrich']}</b>",
                        f"Insight tables <b>{len(rows['insights'])}</b>"]))
    with right, st.container(border=True):
        ui.section("Stage timings", "Seconds per stage")
        t = pd.DataFrame(list(m["timings_seconds"].items()), columns=["stage", "seconds"])
        ui.plot(ui.hbar(t, "stage", "seconds", value_fmt=".3f", hover_name="Seconds"), height=230, key="pr_time")

    ui.section("Inputs", "SHA-256 checksums prove which files produced this run")
    inputs = pd.DataFrame(m["inputs"])
    inputs["sha256"] = inputs["sha256"].str[:16] + "..."
    st.dataframe(inputs, hide_index=True)

    ui.section("Downloads", "Files with customer emails (customer_360*.json, customer360.db) are not offered here")
    cols = st.columns(len(DOWNLOADS) + 1)
    for col, (name, mime) in zip(cols, DOWNLOADS):
        with col:
            st.download_button(name, read_bytes(ctx.bundle.latest, name), file_name=name, mime=mime,
                               icon=":material/download:", width="stretch", key=f"dl_{name}")
    with cols[-1]:
        name = st.selectbox("Insight table", sorted(ctx.bundle.insights), key="pr_insight", label_visibility="collapsed")
        st.download_button(f"{name}.csv", read_bytes(ctx.bundle.latest, f"insights/{name}.csv"),
                           file_name=f"{name}.csv", mime="text/csv", icon=":material/download:", width="stretch",
                           key="dl_insight")

    with st.expander("Config used for this run", icon=":material/settings:"):
        st.json(m["config"], expanded=1)
    with st.expander("Output files", icon=":material/folder:"):
        st.dataframe(pd.DataFrame(m["outputs"]), hide_index=True)
