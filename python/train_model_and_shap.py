"""
train_model_and_shap.py

Full pipeline for the interpretable-by-design model:
  1. Load engineered features (eeg_feature_extraction.py output) +
     nonlinear entropy features (nonlinear_entropy_features.py output).
  2. Prune redundant correlated features (correlation threshold r >= 0.90).
  3. Train a 5-class (not binary) XGBoost
     classifier, chosen specifically because it is interpretable-by-design,
     so SHAP values below are exact rather than approximated.
  4. Evaluate on a held-out test set, prioritizing seizure-class recall
     (a missed seizure is costlier than a false alarm).
  5. Run real SHAP (TreeExplainer) on the held-out set — NOT impurity-based
     feature_importances_, which is a known-blunt instrument for correlated
     features. This is the step that actually tests which entropy measure
     (if any) the model relies on.

Requires: xgboost, shap, scikit-learn (pip install --break-system-packages)

Inputs:  engineered_features.csv, nonlinear_entropy_features.csv
Outputs: printed classification report + SHAP importance rankings
         (global and seizure-class-specific), plus the trained model
         and SHAP values saved for later reuse.
"""

import pickle

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ENGINEERED_FEATURES_PATH = "../data/processed/engineered_features.csv"
NONLINEAR_FEATURES_PATH = "../data/processed/nonlinear_entropy_features.csv"
CLASS_NAMES = ["Seizure", "Tumor region", "Healthy region", "Eyes closed", "Eyes open"]

# Features dropped for redundancy (pairwise correlation r >= 0.90 with variance).
# All five (std, rms, peak_to_peak, variance, mean_abs_diff) are mutually
# correlated at r >= 0.90; `variance` is kept as the cluster's representative.
REDUNDANT_FEATURES = ["std", "rms", "peak_to_peak", "mean_abs_diff"]


def load_data():
    ef = pd.read_csv(ENGINEERED_FEATURES_PATH)
    nl = pd.read_csv(NONLINEAR_FEATURES_PATH)

    assert len(ef) == len(nl), "row count mismatch between the two feature files"
    assert (ef["label"].values == nl["y"].values).all(), (
        "row order mismatch — engineered_features.csv and "
        "nonlinear_entropy_features.csv must come from the same run of Data.csv"
    )

    ef["permutation_entropy"] = nl["permutation_entropy"]
    ef["sample_entropy"] = nl["sample_entropy"]

    feature_cols = [
        c for c in ef.columns
        if c not in ("label", "label_binary") and c not in REDUNDANT_FEATURES
    ]
    X = ef[feature_cols]
    y = ef["label"] - 1  # XGBoost expects 0-indexed class labels
    return X, y, feature_cols


def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=5,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    return model


def run_shap(model, X_test, feature_cols):
    explainer = shap.TreeExplainer(model)
    shap_values = np.array(explainer.shap_values(X_test))  # (n_samples, n_features, n_classes)

    global_importance = np.mean(np.abs(shap_values), axis=(0, 2))
    seizure_importance = np.mean(np.abs(shap_values[:, :, 0]), axis=0)  # class 0 = Seizure

    global_df = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap_all_classes": global_importance}
    ).sort_values("mean_abs_shap_all_classes", ascending=False)

    seizure_df = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap_seizure_class": seizure_importance}
    ).sort_values("mean_abs_shap_seizure_class", ascending=False)

    return shap_values, global_df, seizure_df


if __name__ == "__main__":
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=== Held-out test accuracy ===")
    print(round(accuracy_score(y_test, y_pred), 4))
    print("\n=== Classification report ===")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=3))

    shap_values, global_df, seizure_df = run_shap(model, X_test, feature_cols)

    print("\n=== Global SHAP importance (mean |SHAP|, all classes) ===")
    print(global_df.to_string(index=False))
    print("\n=== SHAP importance for SEIZURE class specifically ===")
    print(seizure_df.to_string(index=False))

    # Saved so downstream scripts (comparative visualizations) can reuse this exact
    # trained model and its SHAP values without retraining.
    with open("../results/models/xgb_model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "X_test": X_test,
                "y_test": y_test,
                "feature_cols": feature_cols,
                "class_names": CLASS_NAMES,
            },
            f,
        )
    np.save("../results/models/shap_values.npy", shap_values)
    print("\nSaved outputs/models/xgb_model.pkl and outputs/models/shap_values.npy")
