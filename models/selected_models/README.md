# Selected models

Only the three chosen forecasting models are included:

| Folder | Model |
|---|---|
| `chronos_t5_large_zero_shot_adjusted/` | Chronos-T5-large zero-shot with calendar adjustment |
| `ensemble_chronos_t5_large_gbm_w0.7_2/` | Chronos-T5-large + GBM ensemble (weights 0.7 / 0.3) |
| `level_calendar_no_model/` | 28-day level + calendar baseline |

The zero-shot folder contains metadata only: its pretrained weights are fetched
from Hugging Face (`amazon/chronos-t5-large`) at runtime. The ensemble folder
contains the local gradient-boosting and calendar Ridge artifacts needed by the
pipeline. The baseline folder contains its calendar Ridge artifacts. Each
folder's `model_info.json` records its settings and evaluation metrics.

The pipeline refits the Ridge and GBM components and stores timestamped
versions beneath the relevant model folders; generated versions are not
committed.
