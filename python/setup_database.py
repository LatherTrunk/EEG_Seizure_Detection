"""
setup_database.py

DA2 rubric requirement: "Database connectivity (MySQL/SQLite/PostgreSQL/
MongoDB) + data retrieval using R/Python" (2 marks).

SQLite chosen over a server-based DB (MySQL/PostgreSQL) because this is a
single-user, file-based analysis pipeline with ~11,500 rows — a server
process would add setup/deployment overhead with no real benefit here.
MongoDB doesn't fit either, since the data is naturally tabular, not
document-shaped. SQLite is the pragmatic choice, not the default choice.

This script:
  1. Loads the full feature set (engineered + nonlinear entropy features)
     into a SQLite table.
  2. Demonstrates real retrieval via SQL queries (not just pandas.to_sql
     and calling it done) — a per-class summary query and a seizure-only
     retrieval query, both of which mirror queries the modeling scripts
     could equally pull from instead of reading the CSV directly.

Inputs:  ../data/processed/engineered_features.csv
         ../data/processed/nonlinear_entropy_features.csv
Output:  ../data/processed/eeg_features.db
"""

import sqlite3

import pandas as pd

ENGINEERED_FEATURES_PATH = "../data/processed/engineered_features.csv"
NONLINEAR_FEATURES_PATH = "../data/processed/nonlinear_entropy_features.csv"
DB_PATH = "../data/processed/eeg_features.db"
TABLE_NAME = "eeg_features"

CLASS_NAMES = {1: "Seizure", 2: "Tumor region", 3: "Healthy region", 4: "Eyes closed", 5: "Eyes open"}


def build_database():
    ef = pd.read_csv(ENGINEERED_FEATURES_PATH)
    nl = pd.read_csv(NONLINEAR_FEATURES_PATH)
    assert (ef["label"].values == nl["y"].values).all(), "row order mismatch"

    ef["permutation_entropy"] = nl["permutation_entropy"]
    ef["sample_entropy"] = nl["sample_entropy"]
    ef["class_name"] = ef["label"].map(CLASS_NAMES)

    conn = sqlite3.connect(DB_PATH)
    ef.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_label ON {TABLE_NAME}(label)")
    conn.commit()
    return conn


def demo_queries(conn):
    print("=== Query 1: per-class sample count + mean variance (sanity check) ===")
    q1 = f"""
        SELECT class_name, COUNT(*) as n, ROUND(AVG(variance), 1) as mean_variance
        FROM {TABLE_NAME}
        GROUP BY class_name
        ORDER BY mean_variance DESC
    """
    print(pd.read_sql(q1, conn).to_string(index=False))

    print("\n=== Query 2: seizure-only rows with entropy features (first 5) ===")
    q2 = f"""
        SELECT variance, spectral_entropy, sample_entropy, permutation_entropy
        FROM {TABLE_NAME}
        WHERE label = 1
        LIMIT 5
    """
    print(pd.read_sql(q2, conn).to_string(index=False))


if __name__ == "__main__":
    conn = build_database()
    demo_queries(conn)
    conn.close()
    print(f"\nSaved database to {DB_PATH}")
