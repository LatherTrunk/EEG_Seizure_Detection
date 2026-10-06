# Python pipeline — EEG seizure detection

The Python track of the project. It computes the same features independently
of the R pipeline, adds the 16-model comparison (including the deep-learning
Track B, which has no R equivalent), SHAP explainability, and an SQLite
database. The **primary, R-native pipeline lives in [`../R/`](../R/README.md)**;
the two were cross-validated against each other (see that README).

Scripts assume they're run **from inside `python/`** — all paths are relative
to that (`../data/...`, `../results/...`). Run in this order; each script's
output feeds the next. `data/raw/Data.csv` must be downloaded first (see
`../data/README.md`).

1. **`python/eeg_feature_extraction.py`** (DA1) — `data/raw/Data.csv` → `data/processed/engineered_features.csv`.
   21 time- and frequency-domain features per sample, including spectral entropy.

2. **`R/eda_script.R`** (DA1) — reads from `data/`, writes 5 plots to `results/plots/`.

3. **`python/nonlinear_entropy_features.py`** (new) — `data/raw/Data.csv` → `data/processed/nonlinear_entropy_features.csv`.
   Adds sample entropy and permutation entropy — the time-domain entropy
   family, as distinct from the frequency-domain spectral entropy in step 1.
   Requires `antropy` (`pip install antropy --break-system-packages`).

4. **`python/train_model_and_shap.py`** (new) — reads both processed CSVs → trained model + real SHAP values in `results/models/`.
   - Drops 4 redundant magnitude features (r ≥ 0.90 correlated with `variance`).
   - Trains a 5-class XGBoost classifier (not binary — all five original classes are kept).
   - Runs `shap.TreeExplainer` (not impurity importance) on a held-out test set.
   Requires `xgboost`, `shap`, `scikit-learn`.

5. **`python/train_classical_ml_roster.py`** (new, DA2) — the classical-ML
   half of DA2's "10-15 ML/DL algorithms" requirement. 11 models (Logistic
   Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting,
   AdaBoost, LightGBM, SVM, k-NN, Naive Bayes, MLP), evaluated on the exact
   same 19-feature set and train/test split as the interpretable model, for
   a fair comparison. Saves `results/models/classical_ml_comparison.csv`.
   Requires `lightgbm` in addition to the above.

6. **`python/setup_database.py`** (new, DA2) — database connectivity
   requirement. Loads the full feature set into SQLite (chosen over
   MySQL/PostgreSQL because this is single-user file-based analysis with
   no real benefit from a server DB, and over MongoDB because the data is
   tabular, not document-shaped) and demonstrates real SQL retrieval
   queries. Saves `data/processed/eeg_features.db`.

7. **`python/train_deep_learning_roster.py`** (new, DA2) — Track B:
   1D CNN, LSTM, BiLSTM, and CNN-LSTM hybrid trained directly on the raw
   178-point signal (no feature engineering), on the same train/test split
   as everything else. Run ONE model at a time (`--model cnn|lstm|bilstm|cnn_lstm`)
   rather than all four in one process — LSTM/BiLSTM are sequential and
   don't parallelize on CPU, so this avoids losing all progress if one
   long-running model times out. Each run appends to
   `results/models/deep_learning_comparison.csv` rather than overwriting it.
   Requires `tensorflow-cpu`.

8. **`python/generate_comparative_visualizations.py`** (new, DA2) —
   confusion matrices, ROC curves, PR curves (all one-vs-rest for the
   Seizure class), and a SHAP feature-importance bar chart, for 4
   representative models (XGBoost, Random Forest, LightGBM, CNN-LSTM
   hybrid) rather than all 16 (unreadable). Saves 4 PNGs to `results/plots/`.

9. **`python/hyperparameter_tuning.py`** (new, DA2) — RandomizedSearchCV
   tuning of the top 2 baseline performers only (Random Forest, LightGBM;
   `n_jobs=1` throughout — nested parallelism with the single-core
   container causes contention, not speedup). Saves
   `results/models/tuned_model_comparison.csv`.

## DA2 status: complete
All rubric items done — feature engineering/selection, database
connectivity, 16 ML/DL models (Track A + Track B), hyperparameter tuning,
comparative metrics, and comparative visualizations.

## Results so far (16 models total: 1 interpretable + 11 classical + 4 deep learning)

**Track A — engineered features + classical ML:**

| Model | Accuracy | Macro F1 | Seizure recall |
|---|---|---|---|
| LightGBM | 80.7% | 0.807 | 98.9% |
| Extra Trees | 80.7% | 0.805 | 98.9% |
| XGBoost (interpretable model) | 80.6% | — | 99.1% |
| Random Forest | 80.0% | 0.799 | **99.35% (best)** |
| Gradient Boosting | 79.4% | 0.792 | 98.5% |
| MLP | 76.1% | 0.761 | 97.2% |
| SVM (RBF) | 73.3% | 0.731 | 97.4% |
| Logistic Regression | 71.3% | 0.713 | 94.8% |
| k-NN | 71.3% | 0.711 | 95.7% |
| Decision Tree | 70.6% | 0.704 | 97.4% |
| AdaBoost | 67.7% | 0.668 | 99.1% |
| Naive Bayes | 64.4% | 0.622 | 80.2% (worst) |

**Track B — raw signal + deep learning** (trained on the same split; 1 CPU core, capped at 15 epochs, 32 RNN units — see caveat below):

| Model | Accuracy | Macro F1 | Seizure recall |
|---|---|---|---|
| CNN-LSTM hybrid | 71.5% | 0.702 | 98.5% |
| 1D CNN | 71.0% | 0.701 | 97.0% |
| BiLSTM | 65.9% | 0.647 | 95.4% |
| LSTM | 47.4% | 0.431 | 96.7% |

**Track A beats Track B across the board** (except Naive Bayes). Worth stating carefully in the report: this isn't "features beat deep learning" as a general claim — Track B ran under real compute constraints (1 CPU core, no GPU, capped epochs, small architectures, no DWT preprocessing). It's consistent with Kashefi Amiri et al. (2025) needing DWT decomposition before their CNN-LSTM reached 97% on this same dataset — they didn't train on raw amplitude directly either. Plain LSTM struggling most and CNN-LSTM doing best among Track B fits that pattern: the CNN layers are doing lightweight feature extraction before the LSTM sees the sequence.

## Environment
Python 3.14. Install with `pip install -r requirements.txt` (in the repo root).
Deep-learning runs were done on a single CPU core, which is why Track B is
capped at 15 epochs with small architectures.
