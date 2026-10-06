# 02_classical_ml_models.R
#
# R-native classical ML model roster via caret -- the R counterpart to
# train_classical_ml_roster.py. Uses engineered_features_R.csv (the
# output of 01_feature_engineering.R), so this whole half of the pipeline
# now runs in R without depending on the Python feature extraction at all.
#
# NOTE ON COMPARABILITY: R and Python use different RNG algorithms, so
# set.seed(42) here does NOT reproduce the exact same row-level train/test
# split as random_state=42 in Python -- both are proper stratified 80/20
# splits of the same data, independently drawn, not identical splits.
# Results are comparable in aggregate, not row-for-row.

suppressMessages({
  library(caret)
  library(randomForest)
  library(e1071)
  library(kernlab)
  library(rpart)
  library(gbm)
  library(nnet)
})

set.seed(42)

features <- read.csv("../data/processed/engineered_features_R.csv")
redundant <- c("std", "rms", "peak_to_peak", "mean_abs_diff")
feature_cols <- setdiff(colnames(features), c("label", "label_binary", "class_name", redundant))

X <- features[, feature_cols]
y <- factor(features$label, labels = c("Seizure", "Tumor", "Healthy", "EyesClosed", "EyesOpen"))

train_idx <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_idx, ]; y_train <- y[train_idx]
X_test <- X[-train_idx, ]; y_test <- y[-train_idx]

cat(sprintf("Train: %d rows | Test: %d rows\n", nrow(X_train), nrow(X_test)))

ctrl <- trainControl(method = "none")  # baseline roster -- tuning is a separate script

macro_f1 <- function(cm) {
  mean(cm$byClass[, "F1"], na.rm = TRUE)
}
seizure_recall <- function(cm) {
  cm$byClass["Class: Seizure", "Sensitivity"]
}

model_specs <- list(
  list(name = "Random Forest", method = "rf", preProcess = NULL, tuneGrid = data.frame(mtry = 4)),
  list(name = "SVM (Radial)", method = "svmRadial", preProcess = c("center", "scale"), tuneGrid = data.frame(sigma = 0.05, C = 1)),
  list(name = "SVM (Linear)", method = "svmLinear", preProcess = c("center", "scale"), tuneGrid = data.frame(C = 1)),
  list(name = "k-NN", method = "knn", preProcess = c("center", "scale"), tuneGrid = data.frame(k = 7)),
  # Naive Bayes deliberately excluded here -- caret's "nb" method requires
  # the klaR package, which isn't available via apt in this environment
  # (no CRAN access). Handled separately below via e1071::naiveBayes
  # directly, so the comparison table still includes it.
  list(name = "Decision Tree", method = "rpart", preProcess = NULL, tuneGrid = data.frame(cp = 0.01)),
  list(name = "Gradient Boosting", method = "gbm", preProcess = NULL, tuneGrid = data.frame(n.trees = 150, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10)),
  list(name = "Neural Net (MLP)", method = "nnet", preProcess = c("center", "scale"), tuneGrid = data.frame(size = 10, decay = 0.1)),
  list(name = "Multinomial Logistic", method = "multinom", preProcess = c("center", "scale"), tuneGrid = data.frame(decay = 0.1))
)

results <- list()
for (spec in model_specs) {
  cat(sprintf("\n=== Training %s ===\n", spec$name))
  t0 <- Sys.time()
  # nnet/multinom print a noisy per-iteration log unless trace=FALSE is
  # passed through -- but that argument doesn't exist for the other
  # methods here (rf, svmRadial, rpart, ...), so it's only added for the
  # two methods that actually accept it, rather than passed to all.
  extra_args <- if (spec$method %in% c("nnet", "multinom")) list(trace = FALSE) else list()
  fit <- tryCatch({
    do.call(train, c(
      list(x = X_train, y = y_train, method = spec$method,
           trControl = ctrl, tuneGrid = spec$tuneGrid, preProcess = spec$preProcess),
      extra_args
    ))
  }, error = function(e) { cat("FAILED:", conditionMessage(e), "\n"); NULL })

  if (is.null(fit)) next
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  pred <- predict(fit, X_test)
  cm <- confusionMatrix(pred, y_test)

  results[[spec$name]] <- data.frame(
    model = spec$name,
    accuracy = round(cm$overall["Accuracy"], 4),
    macro_f1 = round(macro_f1(cm), 4),
    seizure_recall = round(seizure_recall(cm), 4),
    train_time_sec = round(elapsed, 1)
  )
  cat(sprintf("%s: acc=%.4f macroF1=%.4f seizure_recall=%.4f (%.1fs)\n",
              spec$name, cm$overall["Accuracy"], macro_f1(cm), seizure_recall(cm), elapsed))
}

# Naive Bayes via e1071 directly (caret's "nb" wrapper needs the
# unavailable klaR package -- this is the same underlying algorithm,
# called directly instead of through caret).
cat("\n=== Training Naive Bayes (via e1071 directly) ===\n")
t0 <- Sys.time()
nb_fit <- naiveBayes(X_train, y_train)
nb_pred <- predict(nb_fit, X_test)
nb_cm <- confusionMatrix(nb_pred, y_test)
elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
results[["Naive Bayes"]] <- data.frame(
  model = "Naive Bayes",
  accuracy = round(nb_cm$overall["Accuracy"], 4),
  macro_f1 = round(macro_f1(nb_cm), 4),
  seizure_recall = round(seizure_recall(nb_cm), 4),
  train_time_sec = round(elapsed, 1)
)
cat(sprintf("Naive Bayes: acc=%.4f macroF1=%.4f seizure_recall=%.4f (%.1fs)\n",
            nb_cm$overall["Accuracy"], macro_f1(nb_cm), seizure_recall(nb_cm), elapsed))

results_df <- do.call(rbind, results)
results_df <- results_df[order(-results_df$macro_f1), ]
write.csv(results_df, "../results/models/classical_ml_comparison_R.csv", row.names = FALSE)
cat("\n=== R classical ML roster, ranked by macro F1 ===\n")
print(results_df, row.names = FALSE)
cat("\nSaved to ../results/models/classical_ml_comparison_R.csv\n")
