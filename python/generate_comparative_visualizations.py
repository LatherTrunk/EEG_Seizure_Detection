"""
generate_comparative_visualizations.py

DA2 rubric requirement: "Comparative visualizations (ROC, Confusion Matrix,
Precision-Recall, Feature Importance)" (1 mark).

Rather than plotting all 16 models (unreadable), this focuses on a
representative set: the strongest model from each family, so the
comparison is meaningful rather than cluttered:
  - XGBoost           (the interpretable model from train_model_and_shap.py)
  - Random Forest      (best seizure recall of the classical roster)
  - LightGBM           (best overall accuracy of the classical roster)
  - CNN-LSTM hybrid     (best of Track B / raw-signal deep learning)

All four are retrained here (fast for the tree models; ~40s for the
CNN-LSTM) specifically to get class probabilities, which the saved
comparison CSVs don't contain -- only point predictions were needed for
the accuracy/F1/recall table, but ROC and PR curves need probabilities.

ROC and PR curves are computed one-vs-rest for the Seizure class
specifically, since that's the clinically relevant distinction this
project has prioritized throughout (recall over raw accuracy).

Requires: matplotlib, scikit-learn, xgboost, lightgbm, tensorflow-cpu

Inputs:  ../data/processed/engineered_features.csv
         ../data/processed/nonlinear_entropy_features.csv
         ../data/raw/Data.csv
         ../results/models/shap_values.npy (from train_model_and_shap.py)
Outputs: 4 PNGs in ../results/plots/
"""

import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks

CLASS_NAMES = ["Seizure", "Tumor region", "Healthy region", "Eyes closed", "Eyes open"]
SEIZURE_IDX = 0
SEED = 42
PLOTS_DIR = "../results/plots/"
REDUNDANT_FEATURES = ["std", "rms", "peak_to_peak", "mean_abs_diff"]


# ---------- Track A data (engineered features) ----------
def load_tabular_data():
    ef = pd.read_csv("../data/processed/engineered_features.csv")
    nl = pd.read_csv("../data/processed/nonlinear_entropy_features.csv")
    ef["permutation_entropy"] = nl["permutation_entropy"]
    ef["sample_entropy"] = nl["sample_entropy"]
    feature_cols = [c for c in ef.columns if c not in ("label", "label_binary") and c not in REDUNDANT_FEATURES]
    X = ef[feature_cols]
    y = ef["label"] - 1
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y), feature_cols


# ---------- Track B data (raw signal) ----------
def load_sequence_data():
    raw = pd.read_csv("../data/raw/Data.csv")
    signal_cols = [c for c in raw.columns if c.startswith("X")]
    X = raw[signal_cols].values.astype("float32")
    y = raw["y"].values - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train).reshape(-1, 178, 1)
    X_test_s = scaler.transform(X_test).reshape(-1, 178, 1)
    return X_train_s, X_test_s, y_train, y_test


def get_probs_and_preds():
    """Trains/loads the 4 representative models and returns (name -> (y_test, y_pred, y_proba))."""
    (X_train, X_test, y_train, y_test), feature_cols = load_tabular_data()
    results = {}

    # XGBoost -- reuse the already-trained, already-saved model
    with open("../results/models/xgb_model.pkl", "rb") as f:
        d = pickle.load(f)
    xgb_model, xgb_X_test, xgb_y_test = d["model"], d["X_test"], d["y_test"]
    results["XGBoost"] = (xgb_y_test, xgb_model.predict(xgb_X_test), xgb_model.predict_proba(xgb_X_test))

    # Random Forest -- retrain (fast)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    results["Random Forest"] = (y_test, rf.predict(X_test), rf.predict_proba(X_test))

    # LightGBM -- retrain (fast)
    lgbm = lgb.LGBMClassifier(random_state=SEED, verbosity=-1)
    lgbm.fit(X_train, y_train)
    results["LightGBM"] = (y_test, lgbm.predict(X_test), lgbm.predict_proba(X_test))

    # CNN-LSTM hybrid -- retrain (Track B representative, ~40s)
    Xtr_s, Xte_s, ytr_s, yte_s = load_sequence_data()
    tf.random.set_seed(SEED)
    cnn_lstm = models.Sequential([
        layers.Input(shape=(178, 1)),
        layers.Conv1D(32, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(5, activation="softmax"),
    ])
    cnn_lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnn_lstm.fit(
        Xtr_s, ytr_s, validation_split=0.15, epochs=15, batch_size=128,
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=0,
    )
    proba = cnn_lstm.predict(Xte_s, verbose=0)
    results["CNN-LSTM hybrid"] = (yte_s, np.argmax(proba, axis=1), proba)

    return results


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (name, (y_true, y_pred, _)) in zip(axes, results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=CLASS_NAMES, ax=ax, colorbar=False, xticks_rotation=45
        )
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}plot6_confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved plot6_confusion_matrices.png")


def plot_roc_curves(results):
    plt.figure(figsize=(7, 6))
    for name, (y_true, _, y_proba) in results.items():
        y_true_binary = (y_true == SEIZURE_IDX).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_proba[:, SEIZURE_IDX])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Seizure vs. Rest (one-vs-rest)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}plot7_roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved plot7_roc_curves.png")


def plot_pr_curves(results):
    plt.figure(figsize=(7, 6))
    for name, (y_true, _, y_proba) in results.items():
        y_true_binary = (y_true == SEIZURE_IDX).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_binary, y_proba[:, SEIZURE_IDX])
        plt.plot(recall, precision, label=f"{name} (AUC={auc(recall, precision):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall — Seizure vs. Rest (one-vs-rest)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}plot8_precision_recall_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved plot8_precision_recall_curves.png")


def plot_shap_feature_importance():
    with open("../results/models/xgb_model.pkl", "rb") as f:
        d = pickle.load(f)
    feature_cols = d["feature_cols"]
    shap_values = np.load("../results/models/shap_values.npy")
    seizure_importance = np.mean(np.abs(shap_values[:, :, SEIZURE_IDX]), axis=0)

    order = np.argsort(seizure_importance)
    plt.figure(figsize=(8, 7))
    plt.barh(np.array(feature_cols)[order], seizure_importance[order], color="#2E86AB")
    plt.xlabel("Mean |SHAP value| (Seizure class)")
    plt.title("SHAP Feature Importance — Seizure Class")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}plot9_shap_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved plot9_shap_feature_importance.png")


if __name__ == "__main__":
    print("Training/loading representative models for probability-based comparisons...")
    results = get_probs_and_preds()

    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_pr_curves(results)
    plot_shap_feature_importance()
    print("\nAll comparative visualizations saved to ../results/plots/")
