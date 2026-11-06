# Customer 360 Pipeline

Live demo: https://customer360-5c9f.onrender.com/

Merges customer data from a CRM (JSON), an order system (CSV) and a support system (XML) into one clean customer profile. It checks data quality, produces business insights, and serves a dashboard with an AI assistant that answers questions in plain English.

Note: the demo runs on Render's free tier. It sleeps when idle, so the first load can take about a minute.

## What it does

```
raw files --> ingest --> standardize --> validate --> merge --> enrich --> publish
 json/csv/xml                              |                                 |
                                           +--> rejected_records.csv         +--> JSON, CSV, SQLite, reports
                                                                             +--> dashboard + AI Q&A
```

1. **Ingest:** reads the three files and keeps every value as raw text.
2. **Standardize:** renames fields to one schema (`cust_id`, `customerId` become `customer_id`), trims whitespace, adds lineage columns (source, run id, timestamp).
3. **Validate:** applies data quality rules. Bad rows are never silently dropped; they go to `rejected_records.csv` with a reason code. Rules are either **errors** (affect `successful_records`) or **warnings** (flagged only).
4. **Merge:** CRM left join orders left join tickets on `customer_id`. Output has exactly one row per valid CRM customer.
5. **Enrich:** segments (Gold / Silver / Bronze), average order value, recency, tenure, a rule-based risk score, at-risk VIP flag, and RFM labels.
6. **Publish:** writes outputs to `output/<run_id>/`, then swaps them into `output/latest/` so the dashboard never reads a half-written run.

Everything is driven by `config.json` (input sources, rules, thresholds, weights, model names).

## Data quality results (sample data)

| Check | Result |
|---|---|
| Total records (3 sources) | 1501 |
| Duplicates | 1 (C0105) |
| Invalid emails | 3 (email set to null, customer kept) |
| Missing fields | 1 (C0310, orders) |
| Invalid IDs | 1 (blank ID in tickets, rejected) |
| Negative revenue | 1 (C0420, value quarantined) |
| Successful records | 1494 |
| Customers in Customer 360 | 500 |

`as_of_date` is fixed to **2026-10-28** (latest date in the data) so results are reproducible. Override with `--as-of YYYY-MM-DD`.

## Outputs

In `output/latest/`:
- `customer_360.json`: the 10 fields from the assignment
- `customer_360_enriched.json`: plus segments, risk, RFM, flags
- `dq_report.json`, `rejected_records.csv`, `run_manifest.json`
- `insights/`: top 10 customers, segments, revenue by city, customers by city, customers with more than 5 tickets, and more
- `customer360.db`: SQLite, used by the AI assistant

## Dashboard

Overview, Revenue and Regions, Segments and Customers, Support, Data Quality, Ask AI, Pipeline Run. Filters by city, segment and risk band apply everywhere. Emails are always masked.

## AI assistant (Groq)

Ask questions like "which city has the most revenue?". The model writes one SQL query, a guard checks it, and it runs on a read-only database. The guard allows only a single SELECT, and blocks email columns, `SELECT *`, comments and any write. Only capped, aggregated rows are used for the summary, and names and emails are never sent to the model. If there is no API key, or the service is rate limited, preset questions still work. Tested on 12 questions checked against pandas: 12 of 12 correct.

The key is optional. Copy `.env.example` to `.env` and set `GROQ_API_KEY`. Never commit `.env`.

## How to run

Three ways. Run locally with Streamlit if you can; use the live link for a quick look.

### 1. Live demo (Render)

Open https://customer360-5c9f.onrender.com/ (first load may take about a minute).

### 2. Local with Streamlit

Requires Python 3.10+ (tested on 3.12).

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py                    # run the pipeline first
streamlit run app.py              # open http://localhost:8501
python -m pytest                  # 189 tests
```

`streamlit run app.py` needs `python main.py` to run first, because the dashboard reads `output/latest/`.

### 3. Docker

```
docker build -t customer360 .
docker run -p 8501:8501 --env-file .env customer360
```

Open http://localhost:8501. The pipeline runs when the container starts. Without a `.env` file, drop `--env-file .env` and the app runs with preset questions only.

## Deployment (Render)

The live demo runs from the same Dockerfile.

| Setting | Value |
|---|---|
| Service type | Web Service |
| Runtime | Docker |
| Instance type | Free |
| Port | Set by Render through `$PORT` (the Dockerfile falls back to 8501) |
| Environment | `GROQ_API_KEY` (optional; without it the demo uses preset questions) |

The container runs `python main.py` on every start, so `output/` is rebuilt from `data/raw/`. Secrets live only in Render's environment settings and are never in the repo or image (`.env` is in `.gitignore` and `.dockerignore`).

## Notes

- Risk components that are missing (e.g. null revenue) are left out and the remaining weights are rescaled.
- On Windows the `latest/` swap has a very short gap between two renames.
- A production version would use cloud storage, Spark or Glue, an orchestrator such as Airflow, and a warehouse.
