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

The runtime is the local source of truth for the dashboard:

- Untouched uploads are stored in `runtime/data/raw/uploads/`.
- The current master and cleaned series are stored in
  `runtime/data/clean/train_cleaned_data/`.
- Immutable dated copies and their JSON metadata are stored beside those
  live series. `runtime/state/manifest.json` tracks successful uploads and
  their archive/copy filenames; `runtime/state/jobs.json` stores job status.
- The dashboard reads current files from disk and offers downloads for dated
  copies. It checks every five seconds for deleted upload assets or direct
  edits to the live master/cleaned files, waits for a 15-second quiet period,
  then queues one rebuild/refit job. A rebuild starts from the tracked daily
  baseline and replays the remaining upload archives. If an archive cannot
  continue the contiguous history, it is retained and reported as ignored.
- The chart/history refreshes when the live data or latest forecast changes.
  The existing forecast remains in place if a job fails.

For a PowerShell start with a clear port-8000 conflict message:

```powershell
.\start_app.ps1
```

Open the runtime files in Explorer/Finder/the system file browser:

```powershell
python -m api.open_runtime
```

Reset the runtime data to the repository baseline only when intended. The
command requires `--confirm`, creates a timestamped backup under
`runtime/runtime_backups/`, clears upload archives and dated copies from the
active runtime, resets the manifest and job history, and preserves the
current forecast by default. Add `--forecast` to clear that forecast too:

```powershell
python -m api.reset_data --confirm
python -m api.reset_data --confirm --forecast
```

The runtime and its backups are local and gitignored. Do not copy client
uploads or runtime data into Git.