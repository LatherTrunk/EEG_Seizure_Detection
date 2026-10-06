"""
EEG Feature Extraction — Epileptic Seizure Recognition Dataset (UCI)
---------------------------------------------------------------------
Turns each raw 1-second EEG signal (178 amplitude readings) into a
feature vector combining time-domain and frequency-domain (FFT band
power) features, for use with classical ML models (DA2).

Expected input (any of these layouts are handled automatically):
    Unnamed: 0, X1, X2, ..., X178, y      <- Kaggle mirror (has ID column)
    X1, X2, ..., X178, y                  <- UCI original (no ID column)

- Every row = one 1-second EEG sample
- Columns X1...X178 = the 178 raw signal readings for that sample
- Column y = class label (1-5): 1 = seizure, 2-5 = non-seizure variants

Output: engineered_features.csv (one row per sample, ~23 engineered
        features + multi-class label + binary label)
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.stats import skew, kurtosis

INPUT_FILE = "../data/raw/Data.csv"
OUTPUT_FILE = "../data/processed/engineered_features.csv"
SAMPLING_RATE = 178  # 178 readings represent 1 second -> 178 Hz effective rate

# ---------------------------------------------------------------------
# 1. Load data and auto-detect signal vs. label vs. ID columns
# ---------------------------------------------------------------------
def load_and_identify_columns(path):
    df = pd.read_csv(path)

    if "y" not in df.columns:
        raise ValueError(
            f"Expected a label column named 'y' but got columns: {list(df.columns)}. "
            "Check you downloaded the correct file."
        )

    # Signal columns: anything that matches X<number> (X1, X2, ..., X178)
    signal_cols = [c for c in df.columns if c.startswith("X") and c[1:].isdigit()]
    signal_cols = sorted(signal_cols, key=lambda c: int(c[1:]))  # ensure X1..X178 order

    if len(signal_cols) != 178:
        raise ValueError(
            f"Expected 178 signal columns (X1..X178), found {len(signal_cols)}. "
            "Check the file wasn't truncated or reformatted."
        )

    # Anything that isn't a signal column and isn't 'y' is treated as an ID/index
    # column (e.g. 'Unnamed: 0') and simply ignored, not fed into the model.
    ignored_cols = [c for c in df.columns if c not in signal_cols + ["y"]]
    if ignored_cols:
        print(f"Ignoring non-signal columns (ID/index, not features): {ignored_cols}")

    return df, signal_cols


df, signal_cols = load_and_identify_columns(INPUT_FILE)

X_raw = df[signal_cols].to_numpy(dtype=float)   # shape: (n_samples, 178)
y = df["y"].to_numpy()                          # classes 1-5

# Sanity checks
assert not np.isnan(X_raw).any(), "Found NaNs in signal data — inspect the raw file before proceeding."
assert set(np.unique(y)).issubset({1, 2, 3, 4, 5}), f"Unexpected label values found: {np.unique(y)}"

# Binarize: class 1 = seizure, classes 2-5 = non-seizure
y_binary = np.where(y == 1, 1, 0)

print(f"Loaded {X_raw.shape[0]} samples, {X_raw.shape[1]} signal points each.")
print(f"Class distribution (multi-class): {dict(zip(*np.unique(y, return_counts=True)))}")
print(f"Class distribution (binary): {dict(zip(*np.unique(y_binary, return_counts=True)))}")

# ---------------------------------------------------------------------
# 2. Time-domain features
# ---------------------------------------------------------------------
def time_domain_features(signal):
    diffs = np.diff(signal)
    return {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "variance": np.var(signal),
        "skewness": skew(signal),
        "kurtosis": kurtosis(signal),
        "peak_to_peak": np.max(signal) - np.min(signal),
        "zero_crossing_rate": ((signal[:-1] * signal[1:]) < 0).sum() / len(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
        "mean_abs_diff": np.mean(np.abs(diffs)),   # signal "roughness" between consecutive points
    }

# ---------------------------------------------------------------------
# 3. Frequency-domain features (FFT band power)
# ---------------------------------------------------------------------
NYQUIST = SAMPLING_RATE / 2  # 89 Hz — max representable frequency for this sampling rate

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, NYQUIST),  # capped at Nyquist, not the usual 100 Hz gamma upper bound
}

def frequency_domain_features(signal, fs=SAMPLING_RATE):
    n = len(signal)
    freqs = rfftfreq(n, d=1.0 / fs)
    power = np.abs(rfft(signal)) ** 2

    total_power = np.sum(power) + 1e-12  # avoid divide-by-zero on a flat/zero signal
    feats = {}
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_power = np.sum(power[mask])
        feats[f"{band_name}_power"] = band_power
        feats[f"{band_name}_rel_power"] = band_power / total_power  # normalized 0-1

    feats["dominant_freq"] = freqs[np.argmax(power)]
    feats["spectral_entropy"] = _spectral_entropy(power)
    return feats


def _spectral_entropy(power):
    """Measures how 'spread out' vs. 'peaked' the frequency content is —
    seizures tend to concentrate energy in narrower bands, lowering entropy."""
    p_norm = power / (np.sum(power) + 1e-12)
    p_norm = p_norm[p_norm > 0]  # avoid log(0)
    return -np.sum(p_norm * np.log2(p_norm))

# ---------------------------------------------------------------------
# 4. Build feature matrix (vectorized where possible, row-wise otherwise)
# ---------------------------------------------------------------------
feature_rows = []
for i in range(X_raw.shape[0]):
    signal = X_raw[i, :]
    row = {}
    row.update(time_domain_features(signal))
    row.update(frequency_domain_features(signal))
    feature_rows.append(row)

    if (i + 1) % 2000 == 0:
        print(f"  processed {i + 1}/{X_raw.shape[0]} samples...")

features_df = pd.DataFrame(feature_rows)
features_df["label"] = y                 # multi-class (1-5)
features_df["label_binary"] = y_binary    # seizure (1) vs non-seizure (0)

# ---------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------
features_df.to_csv(OUTPUT_FILE, index=False)
n_features = features_df.shape[1] - 2  # excluding the two label columns
print(f"\nDone. Engineered {n_features} features for {features_df.shape[0]} samples.")
print(f"Saved to {OUTPUT_FILE}")
print(features_df.head())

"""
Notes for DA2:
- This produces ~23 engineered features per sample (9 time-domain +
  ~13 frequency-domain incl. spectral entropy) from the raw 178-point signal.
- Use `label_binary` for a seizure vs. non-seizure task, or `label` if your
  professor wants the full 5-class problem — decide this once, up front,
  since it changes which metrics/plots make sense later.
- For the deep learning branch (1D-CNN / LSTM) in DA2, feed X_raw directly
  (reshaped to (n_samples, 178, 1)), NOT these engineered features — that
  contrast (engineered features + classical ML vs. raw signal + DL) is the
  comparison described in the DA1 proposal.
- Recall/sensitivity matters more than plain accuracy here (a missed seizure
  is costlier than a false alarm) — keep this in mind when choosing your
  evaluation metric and any class-weighting during model training.
"""
