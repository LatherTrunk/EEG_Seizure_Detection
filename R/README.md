# R pipeline — EEG seizure detection

Everything here runs **natively in R**, independently of the Python
pipeline — not a wrapper around it. Run in order from inside `R/`:

1. **`01_feature_engineering.R`** — raw `data/raw/Data.csv` → all 23
   features (9 time-domain, 12 frequency-domain via base R's `fft()`,
   plus sample entropy and permutation entropy) computed purely in R.
   Sample entropy and permutation entropy are **hand-implemented**
   (standard algorithms, vectorized via `outer()`/`pmax()` for speed —
   ~55 sec for all 11,500 rows) since neither is available via apt in
   this environment and CRAN isn't reachable here.
   Saves `data/processed/engineered_features_R.csv`.

   **Validated against the Python output before being trusted** — this
   caught two real bugs: `spectral_entropy` was first normalized by
   `log(n_bins)` (wrong scale), then used natural log instead of log
   base 2 (consistent ~1.44x = 1/ln(2) ratio vs. Python). Fixed; now
   matches Python to within 0.02 across all 11,500 rows. `sample_entropy`
   is close but not bit-identical to Python's `antropy` package (minor
   boundary/tie-handling convention difference between implementations —
   both are valid standard sample entropy computations). `permutation_entropy`,
   `beta_power`, and `dominant_freq` match Python exactly.

2. **`02_classical_ml_models.R`** — 9 models via `caret`: Random Forest,
   SVM (radial + linear), k-NN, Decision Tree, Gradient Boosting (`gbm`),
   Neural Net (`nnet`), Multinomial Logistic, and Naive Bayes (via
   `e1071::naiveBayes` directly — caret's `nb` method needs the
   unavailable `klaR` package). Same 80/20 stratified split logic as the
   Python roster, though **not row-identical** — R and Python use
   different RNG algorithms, so `set.seed(42)` doesn't reproduce the same
   split as `random_state=42`. Saves `results/models/classical_ml_comparison_R.csv`.

3. **`03_database_connectivity.R`** — loads the R-generated feature set
   into SQLite via `DBI`/`RSQLite`, with real SQL retrieval queries.
   Saves `data/processed/eeg_features_R.db`.

4. **`04_hyperparameter_tuning.R`** — `caret`'s native cross-validated
   tuning (`trainControl(method="cv")` + `tuneGrid`) on the top 2
   baseline performers (Random Forest, Gradient Boosting) only.
   Saves `results/models/tuned_model_comparison_R.csv`.

5. **`05_comparative_visualizations.R`** — confusion matrices (`ggplot2`),
   ROC curves for Seizure-vs-rest (`pROC`), and Random Forest feature
   importance (Gini). Saves 3 PNGs to `results/plots/` (`plot10`–`plot12`).

6. **`06_statistical_analysis.R`** — genuine statistical data science, as
   distinct from everything above (which is all modeling): descriptive
   statistics by class, Shapiro-Wilk normality tests, one-way ANOVA with
   eta-squared effect sizes (plus a rank-based Kruskal-Wallis check) for
   **all 23 features**, Tukey HSD post-hoc comparisons with Cohen's d, and
   PCA for class separability — **no trained classifier involved anywhere in
   this script**. Uses only base R's `stats` package (`aov`, `TukeyHSD`,
   `kruskal.test`, `shapiro.test`, `prcomp`) plus `ggplot2` for the PCA plot.
   Saves to `results/models/`: `descriptive_statistics_R.csv`,
   `shapiro_wilk_R.csv`, `anova_effect_sizes_R.csv`, `tukey_hsd_R.csv`,
   `pca_variance_R.csv`, `pca_loadings_R.csv`, and
   `results/plots/plot13_R_pca_separability.png`.

7. **`eda_script.R`** (unchanged from DA1) — the original 5 EDA plots.

## Results

**Classical ML roster (9 models, R-native):**

| Model | Accuracy | Macro F1 | Seizure recall |
|---|---|---|---|
| Random Forest | 79.1% | 0.791 | 97.0% |
| Gradient Boosting | 77.6% | 0.775 | 97.2% |
| Neural Net (MLP) | 75.5% | 0.754 | 96.1% |
| SVM (Radial) | 73.7% | 0.736 | 96.5% |
| SVM (Linear) | 72.8% | 0.729 | 93.9% |
| Multinomial Logistic | 72.7% | 0.727 | 93.5% |
| k-NN | 70.4% | 0.704 | 92.8% |
| Decision Tree | 67.7% | 0.648 | 96.3% |
| Naive Bayes | 60.6% | 0.586 | 60.2% |

**After tuning:** Random Forest 79.1% → 79.7% (mtry=6), Gradient Boosting
77.6% → 78.4% (trees=150, depth=4). Both improved on seizure recall too.

**Feature importance (Random Forest, Gini)** independently confirms the
Python SHAP finding: `sample_entropy` ranks 6th of 19 features (genuinely
important), `spectral_entropy` ranks near the bottom — the same
qualitative pattern as the Python SHAP result, now also visible from
an R-native analysis rather than only a Python one.

## Statistical analysis (06_statistical_analysis.R) — the "pure data science" layer

Only 1 of 25 (feature × class) combinations passed the Shapiro-Wilk
normality test — real, data-grounded justification for preferring
tree-based methods over normality-assuming ones like LDA, rather than
just defaulting to them.

**One-way ANOVA effect sizes (eta-squared), all 23 features, no model involved.**
Selected rows (full table in `anova_effect_sizes_R.csv`):

| Rank | Feature | eta-squared | Kruskal-Wallis eta-squared |
|---|---|---|---|
| 1-4 | std, rms, peak_to_peak, mean_abs_diff | 0.56-0.65 | 0.53-0.67 |
| 5 | **sample_entropy** | **0.502** | 0.539 |
| 6 | variance | 0.454 | 0.528 |
| 7 | permutation_entropy | 0.426 | 0.411 |
| 15 | beta_power | 0.232 | 0.674 |
| 20 | **spectral_entropy** | **0.105** | 0.145 |
| 21-23 | kurtosis, skewness, mean | 0.007-0.022 | 0.005-0.011 |

How to read this precisely:
- The four top-ranked magnitude features are the redundant cluster pruned in
  the modeling pipeline (r >= 0.90 with `variance`); they are included here so
  the ranking covers every feature. Among the 19 features actually modeled,
  sample entropy ranks first.
- `variance` is the cluster's representative but scores lower than `std`
  because it is strongly right-skewed, which flatters sample entropy in the
  comparison with it. The rank-based column removes much of that artifact,
  and under it two band-power features (`beta_power`, `alpha_power`) rank above
  sample entropy.
- Spectral entropy is a **medium** effect (rank 20 of 23), not the weakest
  feature. The defensible claim is that time-domain entropies separate the five
  classes roughly 4-5x better than spectral entropy, not that spectral entropy
  carries no information.

This is a *third* line of evidence (alongside Python SHAP and this
pipeline's own Random Forest Gini importance), obtained with classical
inferential statistics and no trained model.

**Tukey HSD with Cohen's d** (`tukey_hsd_R.csv`), Seizure vs. each other class:

| Comparison | spectral_entropy |d| | sample_entropy |d| |
|---|---|---|
| vs Tumor | 0.78 (medium) | 0.84 (large) |
| vs Healthy | 0.81 (large) | 1.16 (large) |
| vs EyesClosed | 0.19 (negligible) | 2.27 (large) |
| vs EyesOpen | 0.12 (negligible) | 2.79 (large) |

All eight comparisons are statistically significant (smallest adjusted p =
0.0004, Seizure vs EyesOpen on spectral entropy). With n = 2,300 per class
even tiny differences are significant, so the right reading is "significant
but practically negligible" for spectral entropy against the resting-state
classes, while sample entropy separates seizure from them by more than two
standard deviations.

PCA on all 19 features (no classifier involved) shows Seizure fanning out
clearly along PC1, while the other four classes stay clustered near the
origin — visual, unsupervised confirmation of the same Tumor/Healthy and
EyesOpen/EyesClosed confusions every trained model's confusion matrix
also shows.

## What this means for the DA2 rubric
Every item now has a demonstrable R implementation, not just the EDA:
feature engineering, database connectivity, 9 ML models, hyperparameter
tuning, and comparative visualizations all run natively in R. Track B
(deep learning on raw signal) remains Python-only — porting CNN/LSTM to R
would go through `keras3`, which still calls the same Python/TensorFlow
backend underneath, so it wouldn't add a materially different R
demonstration even if built.
