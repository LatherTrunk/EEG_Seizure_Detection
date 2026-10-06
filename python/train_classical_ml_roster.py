"""
train_classical_ml_roster.py

DA2 rubric requirement: "Implementation of 10-15 ML/DL algorithms" (3 marks).
This script builds the classical-ML half of that roster (Track A: engineered
features + classical ML), evaluated on the SAME 19-feature set and the SAME
train/test split as the interpretable XGBoost model (train_model_and_shap.py),
so every number in the final comparison table is a fair apples-to-apples
comparison, not different models on different data splits.

Track B (deep learning directly on the raw 178-point signal) is a separate
script, since it needs different input shaping (sequences, not tabular rows)
and a different train/test split strategy is not required as long as the
same random_state is reused.

This script intentionally does NOT tune hyperparameters yet — these are
default/lightly-set baselines. Hyperparameter tuning (DA2 rubric, 1 mark)
is applied afterward to whichever 2-3 models come out strongest here,
not to all of them (no research or grading benefit to tuning everything).

Requires: scikit-learn, lightgbm, xgboost (already installed for the
interpretable model script)

Inputs:  ../data/processed/engineered_features.csv
         ../data/processed/nonlinear_entropy_features.csv
Outputs: printed comparison table (accuracy, macro F1, seizure recall)
         saved to ../results/models/classical_ml_comparison.csv
"""

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ENGINEERED_FEATURES_PATH = "../data/processed/engineered_features.csv"
NONLINEAR_FEATURES_PATH = "../data/processed/nonlinear_entropy_features.csv"
OUTPUT_PATH = "../results/models/classical_ml_comparison.csv"
REDUNDANT_FEATURES = ["std", "rms", "peak_to_peak", "mean_abs_diff"]
SEIZURE_CLASS_INDEX = 0  # after y - 1, class 0 = Seizure (label 1)


def load_data():
    ef = pd.read_csv(ENGINEERED_FEATURES_PATH)
    nl = pd.read_csv(NONLINEAR_FEATURES_PATH)
    assert (ef["label"].values == nl["y"].values).all(), "row order mismatch"
    ef["permutation_entropy"] = nl["permutation_entropy"]
    ef["sample_entropy"] = nl["sample_entropy"]
    feature_cols = [
        c for c in ef.columns
        if c not in ("label", "label_binary") and c not in REDUNDANT_FEATURES
    ]
    X = ef[feature_cols]
    y = ef["label"] - 1
    return X, y


# Models that need scaled input (distance-/gradient-based) vs. tree models that don't.
NEEDS_SCALING = {"Logistic Regression", "SVM (RBF)", "k-NN", "MLP", "Naive Bayes"}

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42, n_estimators=200),
    "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    "k-NN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
}


if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    for name, model in MODELS.items():
        t0 = time.time()
        if name in NEEDS_SCALING:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elapsed = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro")
        seizure_recall = recall_score(
            y_test, y_pred, labels=[SEIZURE_CLASS_INDEX], average="macro"
        )
        results.append(
            {
                "model": name,
                "accuracy": round(acc, 4),
                "macro_f1": round(macro_f1, 4),
                "seizure_recall": round(seizure_recall, 4),
                "train_time_sec": round(elapsed, 2),
            }
        )
        print(f"{name:20s} acc={acc:.4f}  macroF1={macro_f1:.4f}  seizure_recall={seizure_recall:.4f}  ({elapsed:.2f}s)")

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n=== Ranked by macro F1 ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")
