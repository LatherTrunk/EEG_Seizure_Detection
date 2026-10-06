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
   split as `random_state=42`. Saves `outputs/models/classical_ml_comparison_R.csv`.

3. **`03_database_connectivity.R`** — loads the R-generated feature set
   into SQLite via `DBI`/`RSQLite`, with real SQL retrieval queries.
   Saves `data/processed/eeg_features_R.db`.

4. **`04_hyperparameter_tuning.R`** — `caret`'s native cross-validated
   tuning (`trainControl(method="cv")` + `tuneGrid`) on the top 2
   baseline performers (Random Forest, Gradient Boosting) only.
   Saves `outputs/models/tuned_model_comparison_R.csv`.

5. **`05_comparative_visualizations.R`** — confusion matrices (`ggplot2`),
   ROC curves for Seizure-vs-rest (`pROC`), and Random Forest feature
   importance (Gini). Saves 3 PNGs to `outputs/plots/` (`plot10`–`plot12`).

6. **`06_statistical_analysis.R`** — genuine statistical data science, as
   distinct from everything above (which is all modeling): descriptive
   statistics by class, Shapiro-Wilk normality tests, one-way ANOVA with
   eta-squared effect sizes across all 5 classes, Tukey HSD post-hoc
   comparisons, and PCA for class separability — **no trained classifier
   involved anywhere in this script**. Uses only base R's `stats` package
   (`aov`, `TukeyHSD`, `shapiro.test`, `prcomp`) — no external packages
   needed. Saves `outputs/models/descriptive_statistics_R.csv`,
   `outputs/models/anova_effect_sizes_R.csv`, and
   `outputs/plots/plot13_R_pca_separability.png`.

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

**One-way ANOVA effect sizes (eta-squared), no model involved at all:**

| Feature | eta-squared | Effect size |
|---|---|---|
| sample_entropy | 0.502 | Large — largest of all 5 |
| variance | 0.454 | Large |
| permutation_entropy | 0.426 | Large |
| beta_power | 0.232 | Large |
| spectral_entropy | 0.105 | Medium — smallest by far |

**Of these five features, sample entropy has the largest effect size —
larger than variance, and about 5x spectral entropy.** (These five were
hand-picked for the entropy comparison. Across all 23 engineered features,
the four magnitude features later pruned for redundancy — `std`, `rms`,
`peak_to_peak`, `mean_abs_diff` — score higher still, so this is a
comparison within the analyzed set, not a ranking of every feature.) This is a *third* independent line of
evidence (alongside Python SHAP and this pipeline's own Random Forest
Gini importance) that sample entropy genuinely separates the classes and
spectral entropy does not — this time via classical inferential
statistics, with no trained model anywhere in the analysis.

One nuance worth being precise about: Tukey HSD shows Seizure-vs-EyesOpen
for `spectral_entropy` **is** statistically significant (p=0.0004) — not
non-significant as might be assumed from the DA1 EDA alone — but the
effect magnitude is 4-6x smaller than Seizure-vs-Tumor/Healthy. With
n=2,300 per class there's enough statistical power to detect even small
differences as "significant," so the right framing is "statistically
significant but practically small," not "not significant."

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
