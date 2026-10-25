# Data

- `raw/daily_total_sales_initial.csv`: the original daily-total seed history.
- `raw/uploads/`: timestamped, untouched client-upload archives (ignored by Git).
- `clean/train_cleaned_data/daily_total_sales.csv`: the tracked full daily-total
  master history used by the pipeline.
- `clean/train_cleaned_data/daily_total_sales_clean.csv`: the tracked cleaned
  history, including closure and holiday flags.
- `reference/holidays_events.csv`: Ecuador holiday/event reference used by the
  calendar-adjustment model.
- `reference/stores.csv`: store list used to validate upload coverage.
- `backups/` and `forecasts/`: runtime backups and generated forecast files;
  these are ignored by Git.

The store-level `raw/train.csv` is intentionally excluded from the repository.
New store-level uploads are validated, archived, aggregated by date, and merged
into the tracked daily master; existing `timestamp_utc,sales` daily-total
uploads are also accepted.

The source competition dataset is distributed by Kaggle. Check its current
license and competition rules before redistributing it outside this repository.
