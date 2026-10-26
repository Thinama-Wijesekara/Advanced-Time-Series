# Advanced Time Series Sales Pipeline

A reproducible daily store-sales forecasting pipeline using the three selected
models:

1. Chronos-T5-large zero-shot with calendar adjustment
2. Chronos-T5-large + gradient-boosting ensemble (Chronos weight 0.7)
3. Level + calendar baseline

The complete master and cleaned daily-total series and the holiday/store
references are in `data/`.
The zero-shot Chronos model weights are downloaded from Hugging Face on first
run; the selected local model metadata and ensemble/calendar artifacts are in
`models/selected_models/`.

## Setup

Use Python 3.12.14 (see `.python-version`) and install the pipeline dependencies:

```powershell
python -m pip install -r pipeline/requirements.txt
```

Chronos inference downloads `amazon/chronos-t5-large`, so the first run needs
internet access and several GB of free disk space for the model cache. Inference
can run on CPU but will be substantially slower than on a supported GPU.

## Run

From the repository root, forecast the next 16 days using the included daily
training data:

```powershell
python pipeline/data_pipeline.py
```

To append new store-level data before preprocessing, refitting the calendar/GBM
artifacts, and forecasting, provide a CSV with `date`, `store_nbr`, `family`,
and `sales` columns. Extra columns are preserved in the archived original:

```powershell
python pipeline/data_pipeline.py path/to/store_sales_upload.csv
```

Legacy daily-total CSVs with `timestamp_utc` and `sales` columns are also
accepted. The pipeline checks dates and store coverage against
`data/reference/stores.csv`, archives each valid upload untouched under
`data/raw/uploads/`, aggregates store-level sales by date, and lets uploaded
values replace existing dates. The master and cleaned series are backed up
before each overwrite.

Refitted model artifacts are versioned under `models/selected_models/*/versions/`;
original model files are never overwritten. Timestamped and latest forecast
files are written under `data/forecasts/` in long format with `date`, `model`,
and `forecast` columns. The generated outputs, upload archives, backups, and
model versions are excluded from Git; the master and cleaned histories remain
tracked. The weekly GitHub Actions workflow can also be started manually with
an optional `upload_path`.

## Validation

```powershell
python -m compileall -q pipeline
python pipeline/validate_assets.py
```

## Local web dashboard

The local dashboard accepts store-level CSV and Excel (`.xlsx`) uploads and
runs the same six pipeline steps for each valid file. It listens only on the
loopback interface and does not send uploaded sales data to a hosted service.
Chronos is not retrained; the calendar Ridge and GBM components are refit.

Install the pinned Python 3.12 dependencies from the repository root:

```powershell
python -m pip install -r pipeline/requirements.txt
```

Start the local backend and open `http://127.0.0.1:8000/`:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

On first start, the app creates an ignored `runtime/` working copy of the
tracked `data/` and `models/selected_models/` files. The store-level
`data/raw/train.csv` is excluded and is never copied. The pipeline writes
uploads, job state, refits, cleaned data, and forecasts only in this runtime
copy, leaving tracked files untouched. The Chronos-large weights must already
be cached or downloaded on the first forecast run.

The page shows the current history and latest forecast before any upload.
Accepted `.csv` and `.xlsx` files must contain `date`, `store_nbr`, `family`,
and `sales` columns; each accepted upload runs validation, aggregation,
history merge, cleaning, Ridge/GBM refit, and forecasting in sequence. The
runtime keeps the latest forecast available if a later job fails.

### Optional GitHub publishing

Publishing is **off by default**. Upload results stay in `runtime/` unless you
explicitly enable this feature. To configure it, copy the example file and edit
the local `.env`:

```powershell
Copy-Item .env.example .env
```

Set `PUBLISH_TO_GITHUB=true` and provide a fine-grained GitHub token in
`GITHUB_TOKEN_PUBLISH`. The token needs **Contents: read and write** and
**Actions: write** access to this repository. Restart the app after changing
`.env`. Never paste the token into the browser or commit `.env`; it is
gitignored and the app does not log, return, or save the token.

When enabled, the dashboard displays a warning that this GitHub repository is
public. After a successful local pipeline run, the app publishes only
`data/clean/train_cleaned_data/daily_total_sales.csv`,
`data/clean/train_cleaned_data/daily_total_sales_clean.csv`, and
`data/forecasts/latest_forecast.csv`. The original uploaded CSV/XLSX, `.env`,
`data/raw/train.csv`, backups, and timestamped forecasts are never published by
this feature. The app uses an isolated Git worktree, rebases on the latest
`main` before pushing, then dispatches `data_pipeline.yml` on GitHub Actions.
Publishing or dispatch errors are shown in the dashboard; local results remain
available. If the push succeeds but the dispatch fails, the UI indicates that
the derived files are already public and the workflow must be started manually.