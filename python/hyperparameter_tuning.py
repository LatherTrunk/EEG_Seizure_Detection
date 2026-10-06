"""
hyperparameter_tuning.py

DA2 rubric requirement: "Hyperparameter tuning and model optimization" (1 mark).

Tunes only the top 2 performers from train_classical_ml_roster.py rather
than all 16 models -- no research or grading benefit to tuning models that
already lost the comparison, and tuning everything would just eat compute
for no real payoff:
  - Random Forest  (best seizure recall in the baseline roster: 99.35%)
  - LightGBM       (tied-best overall accuracy in the baseline roster: 80.7%)

Uses RandomizedSearchCV (not GridSearchCV) with a modest n_iter/cv --
this container has 1 CPU core, and a full grid over 4-5 hyperparameters
each with several values would take far longer for a marginal gain over
a randomized search of the same space.

Scored on macro F1 (matches how the baseline roster was ranked), with
seizure recall reported alongside since that's this project's priority
metric throughout.

Inputs:  ../data/processed/engineered_features.csv
         ../data/processed/nonlinear_entropy_features.csv
Outputs: printed before/after comparison, saved to
         ../results/models/tuned_model_comparison.csv
"""

import time

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

ENGINEERED_FEATURES_PATH = "../data/processed/engineered_features.csv"
NONLINEAR_FEATURES_PATH = "../data/processed/nonlinear_entropy_features.csv"
OUTPUT_PATH = "../results/models/tuned_model_comparison.csv"
REDUNDANT_FEATURES = ["std", "rms", "peak_to_peak", "mean_abs_diff"]
SEIZURE_CLASS_INDEX = 0
SEED = 42

# Baseline (untuned) numbers from train_classical_ml_roster.py, for comparison
BASELINE = {
    "Random Forest": {"accuracy": 0.8000, "macro_f1": 0.7990, "seizure_recall": 0.9935},
    "LightGBM": {"accuracy": 0.8070, "macro_f1": 0.8070, "seizure_recall": 0.9890},
}

RF_PARAM_DIST = {
    "n_estimators": [150, 200, 300],
    "max_depth": [None, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

LGBM_PARAM_DIST = {
    "n_estimators": [100, 150, 200],
    "max_depth": [-1, 5, 10],
    "learning_rate": [0.05, 0.1, 0.2],
    "num_leaves": [15, 31, 63],
}


def load_data():
    ef = pd.read_csv(ENGINEERED_FEATURES_PATH)
    nl = pd.read_csv(NONLINEAR_FEATURES_PATH)
    ef["permutation_entropy"] = nl["permutation_entropy"]
    ef["sample_entropy"] = nl["sample_entropy"]
    feature_cols = [c for c in ef.columns if c not in ("label", "label_binary") and c not in REDUNDANT_FEATURES]
    X = ef[feature_cols]
    y = ef["label"] - 1
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def tune(name, base_model, param_dist, X_train, y_train, X_test, y_test):
    print(f"\n=== Tuning {name} ===")
    t0 = time.time()
    search = RandomizedSearchCV(
        base_model, param_dist, n_iter=10, cv=2, scoring="f1_macro",
        random_state=SEED, n_jobs=1,
    )
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = search.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    seizure_recall = recall_score(y_test, y_pred, labels=[SEIZURE_CLASS_INDEX], average="macro")

    print(f"Best params: {search.best_params_}")
    print(f"Tuned: acc={acc:.4f} macroF1={macro_f1:.4f} seizure_recall={seizure_recall:.4f} ({elapsed:.1f}s search)")
    b = BASELINE[name]
    print(f"Baseline was: acc={b['accuracy']:.4f} macroF1={b['macro_f1']:.4f} seizure_recall={b['seizure_recall']:.4f}")

    return {
        "model": name,
        "baseline_accuracy": b["accuracy"], "tuned_accuracy": round(acc, 4),
        "baseline_macro_f1": b["macro_f1"], "tuned_macro_f1": round(macro_f1, 4),
        "baseline_seizure_recall": b["seizure_recall"], "tuned_seizure_recall": round(seizure_recall, 4),
        "best_params": str(search.best_params_),
        "search_time_sec": round(elapsed, 1),
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    results = []
    results.append(tune(
        "Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=1),
        RF_PARAM_DIST, X_train, y_train, X_test, y_test,
    ))
    results.append(tune(
        "LightGBM", lgb.LGBMClassifier(random_state=SEED, verbosity=-1, n_jobs=1),
        LGBM_PARAM_DIST, X_train, y_train, X_test, y_test,
    ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print("\n=== Tuning summary (baseline -> tuned) ===")
    print(results_df[["model", "baseline_accuracy", "tuned_accuracy",
                       "baseline_macro_f1", "tuned_macro_f1",
                       "baseline_seizure_recall", "tuned_seizure_recall"]].to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")
