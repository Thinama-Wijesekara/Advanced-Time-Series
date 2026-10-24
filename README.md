# Advanced Time Series Sales Pipeline

A reproducible daily store-sales forecasting pipeline using the three selected
models:

1. Chronos-T5-large zero-shot with calendar adjustment
2. Chronos-T5-large + gradient-boosting ensemble (Chronos weight 0.7)
3. Level + calendar baseline

The complete daily-total training series and holiday reference are in `data/`.
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

To append newly observed daily totals before preprocessing, refitting the
calendar/GBM artifacts, and forecasting, provide a CSV with `timestamp_utc` and
`sales` columns:

```powershell
python pipeline/data_pipeline.py path/to/daily_sales_upload.csv
```

For overlapping dates, uploaded values replace existing rows. The pipeline
backs up the current raw and cleaned data before updating them. It writes
timestamped model versions under `models/selected_models/` and forecast CSVs
under `data/forecasts/`; generated versions and outputs are intentionally
excluded from Git.

The committed seed series is pre-aggregated by date. New uploaded observations
must also be daily totals. The forecast file is in long format with `date`,
`model`, and `forecast` columns. See `pipeline/` for ingestion, preprocessing,
refitting, and inference implementation. Model parameters are stored in
`pipeline/config.json`, separate from exploratory notebooks.

## Validation

```powershell
python -m compileall -q pipeline
python pipeline/validate_assets.py
```