# EEG Seizure Detection — end-to-end analytics in R (with a Python comparison track)

Classifies one-second EEG windows into five brain states (seizure, tumor-region,
healthy-region, eyes closed, eyes open) and investigates which engineered
features actually separate them — in particular, whether **spectral entropy**
and **time-domain entropy** (sample / permutation entropy) behave alike.
They do not.

## Objectives
1. Build a complete data pipeline in R: acquisition, feature engineering,
   exploratory analysis, statistical testing, modeling, validation.
2. Compare classical ML on engineered features against deep learning on raw signal.
3. Keep every result reproducible from scripts, with the R pipeline as the primary path.

## Repository structure
```
README.md
requirements.txt        Python dependencies
data/                   README + 250-row sample (full data: see data/README.md)
R/                      PRIMARY pipeline: 01-06 scripts + eda_script.R (+ R/README.md)
python/                 comparison track: features, 16 models, SHAP, SQLite (+ python/README.md)
docs/                   DA1 proposal (problem statement, literature review)
results/plots/          all figures (plot1-9 Python/EDA, plot10-13 R)
results/models/         comparison tables, ANOVA and descriptive statistics (CSV)
presentation/           slide deck
```

## How to run
Download `Data.csv` into `data/raw/` (see `data/README.md`). Run each script
from inside its own folder (paths are relative).

**R pipeline** — R >= 4.x with `caret`, `randomForest`, `e1071`, `kernlab`, `rpart`,
`gbm`, `nnet`, `pROC`, `DBI`, `RSQLite`, `ggplot2`, `reshape2`, `corrplot`, `gridExtra`:
```
cd R
Rscript 01_feature_engineering.R      # ~1 min; 23 features, entropy measures hand-implemented
Rscript 02_classical_ml_models.R      # 9 models via caret
Rscript 03_database_connectivity.R    # SQLite via DBI / RSQLite
Rscript 04_hyperparameter_tuning.R    # CV tuning of the top 2 models
Rscript 05_comparative_visualizations.R
Rscript 06_statistical_analysis.R     # normality, ANOVA + effect sizes, Tukey HSD, PCA
```
`eda_script.R` (the DA1 plots) reads the Python feature table, so run
`python/eeg_feature_extraction.py` first if you want to regenerate plots 1-5.

**Python track** — `pip install -r requirements.txt`, then see `python/README.md`
for the run order.

## Results at a glance
5-class accuracy on a stratified 80/20 split (seizure recall in brackets):

| Pipeline | Best model | Accuracy | Seizure recall |
|---|---|---|---|
| R (9 models) | Random Forest, tuned | 79.7% | 97.4% |
| Python, engineered features (11 models) | LightGBM, tuned | 80.9% | 99.1% |
| Python, raw signal (4 deep models) | CNN-LSTM hybrid | 71.5% | 98.5% |

R and Python use different random-number generators, so their splits are not
row-identical; the numbers are comparable in kind, not row for row.

**Statistics (R, no model involved):** only 1 of 25 feature × class combinations
passes Shapiro-Wilk, which motivates tree-based models. One-way ANOVA effect sizes
(η²) over all 23 features: sample entropy 0.50 (rank 5), variance 0.45,
permutation entropy 0.43, **spectral entropy 0.11 (rank 20)**. Tukey HSD with
Cohen's d shows seizure differs from eyes-open / eyes-closed by |d| = 2.8 / 2.3
on sample entropy but only 0.12 / 0.19 on spectral entropy. The same picture
appears independently in Python SHAP (sample entropy 3rd of 19 features,
spectral entropy last) and in R Random Forest importance.

## Limitations
- Deep-learning models ran on a single CPU core with capped epochs and small
  architectures; they should not be read as a general verdict on deep learning.
- Tumor-region vs healthy-region and eyes-open vs eyes-closed remain the main
  confusions for every model.
- The top-ranked ANOVA features are four magnitude features that were pruned as redundant
  with `variance`; comparisons against `variance` are sensitive to its skewness (see `R/README.md`).

## Repository note
The commit history groups the completed work into its logical stages; later work
is committed incrementally.
