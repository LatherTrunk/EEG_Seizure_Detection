# Data

**Dataset:** Epileptic Seizure Recognition (Bonn University EEG, hosted by the
UCI Machine Learning Repository) —
<http://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition>.
Original source: Andrzejak et al. (2001), *Physical Review E*, 64, 061907.

11,500 rows × 178 raw EEG amplitude values (one second of recording each) plus a
label `y` in {1..5}: 1 = seizure, 2 = tumor-region tissue, 3 = healthy region,
4 = eyes closed, 5 = eyes open. 2,300 rows per class (perfectly balanced).

## Layout

| Folder | Contents | In git? |
|---|---|---|
| `raw/` | Put the downloaded `Data.csv` here | No (7.6 MB, public) |
| `sample/` | `Data_sample.csv`: 250 rows, 50 per class, seed 42 | Yes |
| `processed/` | Feature tables and SQLite databases written by the scripts | No (regenerated) |

## Reproducing the processed files
Run `R/01_feature_engineering.R` (writes `processed/engineered_features_R.csv`)
and `python/eeg_feature_extraction.py` + `python/nonlinear_entropy_features.py`
(write the two Python feature tables). See the root README for the run order.
