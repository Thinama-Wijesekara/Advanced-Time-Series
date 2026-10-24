# Data

- `raw/daily_total_sales_initial.csv`: the complete daily-total training
  series, ending on 2017-08-15. On first run, the pipeline copies it to
  `raw/daily_total_sales.csv`, which becomes the mutable master series.
- `reference/holidays_events.csv`: Ecuador holiday/event reference used by the
  calendar-adjustment model.

`data/clean/`, `data/backups/`, and `data/forecasts/` are created or updated by
the pipeline. Generated files and uploaded data are not committed.

The source competition dataset is distributed by Kaggle. Check its current
license and competition rules before redistributing it outside this repository.
