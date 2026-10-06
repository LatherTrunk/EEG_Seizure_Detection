"""
nonlinear_entropy_features.py

Extracts TIME-DOMAIN entropy measures (sample entropy, permutation entropy)
from the raw 178-point EEG signal — as opposed to the frequency-domain
spectral entropy already computed in eeg_feature_extraction.py.

Why this is a separate script / separate feature family:
Spectral entropy (Shannon entropy over the FFT power spectrum) and
sample/permutation entropy (measures of how irregular a waveform is over
time) are mathematically distinct constructions that both get called
"entropy" in the literature. They are NOT interchangeable — the project
compares the two families empirically (SHAP, ANOVA effect sizes). This
script produces the time-domain half of that comparison.

Requires: antropy (pip install antropy --break-system-packages)

Input:  Data.csv (raw dataset, columns X1..X178 + y)
Output: nonlinear_entropy_features.csv (label + permutation_entropy + sample_entropy)
"""

import pandas as pd
import antropy as ant

INPUT_PATH = "../data/raw/Data.csv"
OUTPUT_PATH = "../data/processed/nonlinear_entropy_features.csv"

# Parameters chosen to match common usage in the classic Bonn-dataset
# entropy literature (see the DA1 proposal for citations):
PERM_ENTROPY_ORDER = 4       # embedding dimension for permutation entropy
SAMPLE_ENTROPY_ORDER = 2     # embedding dimension (m) for sample entropy


def extract_nonlinear_features(df: pd.DataFrame) -> pd.DataFrame:
    signal_cols = [c for c in df.columns if c.startswith("X")]
    perm_ent, samp_ent = [], []

    for row in df[signal_cols].values:
        perm_ent.append(ant.perm_entropy(row, order=PERM_ENTROPY_ORDER, delay=1, normalize=True))
        samp_ent.append(ant.sample_entropy(row, order=SAMPLE_ENTROPY_ORDER))

    out = df[["y"]].copy()
    out["permutation_entropy"] = perm_ent
    out["sample_entropy"] = samp_ent
    return out


if __name__ == "__main__":
    raw = pd.read_csv(INPUT_PATH)
    result = extract_nonlinear_features(raw)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(result)} rows to {OUTPUT_PATH}")
    print("\nMedian entropy by class (1=Seizure, 2=Tumor, 3=Healthy, 4=EyesClosed, 5=EyesOpen):")
    print(result.groupby("y")[["permutation_entropy", "sample_entropy"]].median().round(4))
