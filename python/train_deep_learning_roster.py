"""
train_deep_learning_roster.py

DA2 rubric requirement: "Implementation of 10-15 ML/DL algorithms" (3 marks)
-- this is the deep-learning half (Track B: raw signal, no feature
engineering) to compare against the classical-ML half
(train_classical_ml_roster.py, Track A: engineered features).

Unlike Track A, these models see the raw 178-point signal directly and
have to learn their own representations -- this is the point of comparison
the report is built around: does hand-engineered feature extraction
(variance, band power, sample entropy, ...) beat what a network learns on
its own, or not?

Uses the SAME train/test split (random_state=42, stratify=y, test_size=0.2)
as Track A and the interpretable model, on the same row order, so results
are directly comparable rather than confounded by different splits.

Four architectures, in increasing complexity:
  1. 1D CNN            -- local waveform-shape patterns
  2. LSTM               -- sequential/temporal dependencies
  3. BiLSTM              -- temporal dependencies in both directions
  4. CNN-LSTM hybrid     -- CNN feature extraction feeding an LSTM,
                            mirroring the general architecture pattern used
                            in Kashefi Amiri et al. (2025) on this same
                            dataset (see the DA1 literature review), though
                            without their DWT preprocessing step.

Kept deliberately lightweight (32 recurrent units, capped epochs, early
stopping) since this environment has 1 CPU core and no GPU -- LSTM/BiLSTM
are inherently sequential and do not parallelize well on CPU, so this
script runs ONE model per invocation (see --model) rather than all four
in a single long-running process, appending each result to the shared
output CSV so partial progress is never lost.

Requires: tensorflow-cpu (pip install tensorflow-cpu --break-system-packages)

Usage:  python3 train_deep_learning_roster.py --model cnn
        python3 train_deep_learning_roster.py --model lstm
        python3 train_deep_learning_roster.py --model bilstm
        python3 train_deep_learning_roster.py --model cnn_lstm

Input:  ../data/raw/Data.csv
Output: appended to ../results/models/deep_learning_comparison.csv
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, callbacks

RAW_DATA_PATH = "../data/raw/Data.csv"
OUTPUT_PATH = "../results/models/deep_learning_comparison.csv"
CLASS_NAMES = ["Seizure", "Tumor region", "Healthy region", "Eyes closed", "Eyes open"]
SEIZURE_CLASS_INDEX = 0
N_CLASSES = 5
EPOCHS = 15
BATCH_SIZE = 128
SEED = 42
RNN_UNITS = 32  # trimmed from 64 -- CPU-only, LSTM/BiLSTM don't parallelize

tf.random.set_seed(SEED)


def load_data():
    raw = pd.read_csv(RAW_DATA_PATH)
    signal_cols = [c for c in raw.columns if c.startswith("X")]
    X = raw[signal_cols].values.astype("float32")
    y = raw["y"].values - 1  # 0-indexed classes, matches Track A scripts
    return X, y


def build_cnn():
    return models.Sequential([
        layers.Input(shape=(178, 1)),
        layers.Conv1D(32, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="1D_CNN")


def build_lstm():
    return models.Sequential([
        layers.Input(shape=(178, 1)),
        layers.LSTM(RNN_UNITS),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="LSTM")


def build_bilstm():
    return models.Sequential([
        layers.Input(shape=(178, 1)),
        layers.Bidirectional(layers.LSTM(RNN_UNITS)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="BiLSTM")


def build_cnn_lstm():
    return models.Sequential([
        layers.Input(shape=(178, 1)),
        layers.Conv1D(32, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPooling1D(2),
        layers.LSTM(RNN_UNITS),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(N_CLASSES, activation="softmax"),
    ], name="CNN_LSTM_hybrid")


MODEL_BUILDERS = {
    "cnn": ("1D CNN", build_cnn),
    "lstm": ("LSTM", build_lstm),
    "bilstm": ("BiLSTM", build_bilstm),
    "cnn_lstm": ("CNN-LSTM hybrid", build_cnn_lstm),
}


def run_one(model_key):
    display_name, builder = MODEL_BUILDERS[model_key]

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train).reshape(-1, 178, 1)
    X_test_scaled = scaler.transform(X_test).reshape(-1, 178, 1)

    print(f"\n=== Training {display_name} ===")
    tf.random.set_seed(SEED)
    model = builder()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    t0 = time.time()
    model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )
    elapsed = time.time() - t0

    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    seizure_recall = recall_score(y_test, y_pred, labels=[SEIZURE_CLASS_INDEX], average="macro")

    row = {
        "model": display_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "seizure_recall": round(seizure_recall, 4),
        "train_time_sec": round(elapsed, 1),
        "params": model.count_params(),
    }
    print(f"{display_name}: acc={acc:.4f} macroF1={macro_f1:.4f} seizure_recall={seizure_recall:.4f} ({elapsed:.1f}s, {model.count_params()} params)")

    # Append (not overwrite) so running one model at a time never loses
    # results from a model already run.
    existing = pd.read_csv(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else pd.DataFrame(columns=["model"])
    existing = existing[existing["model"] != display_name]  # replace if re-run
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated.to_csv(OUTPUT_PATH, index=False)
    print(f"Appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_BUILDERS.keys()))
    args = parser.parse_args()
    run_one(args.model)

