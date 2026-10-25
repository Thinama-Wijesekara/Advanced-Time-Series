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