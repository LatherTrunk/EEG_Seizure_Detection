# 03_database_connectivity.R
#
# R-native database connectivity (DA2 rubric requirement) -- the R
# counterpart to setup_database.py. Loads the R-generated feature set
# into SQLite via DBI/RSQLite and demonstrates real retrieval with SQL,
# not just read.csv().

suppressMessages({
  library(DBI)
  library(RSQLite)
})

features <- read.csv("../data/processed/engineered_features_R.csv")
class_names <- c("Seizure", "Tumor region", "Healthy region", "Eyes closed", "Eyes open")
features$class_name <- class_names[features$label]

con <- dbConnect(RSQLite::SQLite(), "../data/processed/eeg_features_R.db")
dbWriteTable(con, "eeg_features", features, overwrite = TRUE)
dbExecute(con, "CREATE INDEX IF NOT EXISTS idx_label ON eeg_features(label)")

cat("=== Query 1: per-class sample count + mean variance ===\n")
q1 <- dbGetQuery(con, "
  SELECT class_name, COUNT(*) as n, ROUND(AVG(variance), 1) as mean_variance
  FROM eeg_features
  GROUP BY class_name
  ORDER BY mean_variance DESC
")
print(q1, row.names = FALSE)

cat("\n=== Query 2: seizure-only rows with entropy features (first 5) ===\n")
q2 <- dbGetQuery(con, "
  SELECT variance, spectral_entropy, sample_entropy, permutation_entropy
  FROM eeg_features
  WHERE label = 1
  LIMIT 5
")
print(q2, row.names = FALSE)

dbDisconnect(con)
cat("\nSaved database to ../data/processed/eeg_features_R.db\n")
