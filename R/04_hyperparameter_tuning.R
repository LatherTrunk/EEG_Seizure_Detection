# 04_hyperparameter_tuning.R
#
# R-native hyperparameter tuning (DA2 rubric requirement) -- caret's
# built-in cross-validated tuning (trainControl(method="cv") + tuneGrid),
# applied to the top 2 performers from 02_classical_ml_models.R only
# (Random Forest, Gradient Boosting) -- same reasoning as the Python
# version: no benefit to tuning every model, only the ones that already
# won the baseline comparison.

suppressMessages({
  library(caret)
  library(randomForest)
  library(gbm)
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

macro_f1 <- function(cm) mean(cm$byClass[, "F1"], na.rm = TRUE)
seizure_recall <- function(cm) cm$byClass["Class: Seizure", "Sensitivity"]

baseline <- read.csv("../results/models/classical_ml_comparison_R.csv")

cv_ctrl <- trainControl(method = "cv", number = 3)  # 3-fold, kept modest for runtime

# ---------- Random Forest ----------
cat("=== Tuning Random Forest (cv=3, mtry grid) ===\n")
t0 <- Sys.time()
rf_grid <- expand.grid(mtry = c(3, 4, 6, 8, 10))
rf_tuned <- train(x = X_train, y = y_train, method = "rf", trControl = cv_ctrl, tuneGrid = rf_grid, ntree = 300)
rf_elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
rf_pred <- predict(rf_tuned, X_test)
rf_cm <- confusionMatrix(rf_pred, y_test)
cat(sprintf("Best mtry: %d\n", rf_tuned$bestTune$mtry))
cat(sprintf("Tuned: acc=%.4f macroF1=%.4f seizure_recall=%.4f (%.1fs)\n",
            rf_cm$overall["Accuracy"], macro_f1(rf_cm), seizure_recall(rf_cm), rf_elapsed))

# ---------- Gradient Boosting ----------
cat("\n=== Tuning Gradient Boosting (cv=3, depth/shrinkage/trees grid) ===\n")
t0 <- Sys.time()
gbm_grid <- expand.grid(
  n.trees = c(100, 150),
  interaction.depth = c(3, 4),
  shrinkage = 0.1,
  n.minobsinnode = 10
)
invisible(capture.output(
  gbm_tuned <- train(x = X_train, y = y_train, method = "gbm", trControl = cv_ctrl, tuneGrid = gbm_grid, verbose = FALSE)
))
gbm_elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
gbm_pred <- predict(gbm_tuned, X_test)
gbm_cm <- confusionMatrix(gbm_pred, y_test)
cat(sprintf("Best params: trees=%d, depth=%d, shrinkage=%.2f\n",
            gbm_tuned$bestTune$n.trees, gbm_tuned$bestTune$interaction.depth, gbm_tuned$bestTune$shrinkage))
cat(sprintf("Tuned: acc=%.4f macroF1=%.4f seizure_recall=%.4f (%.1fs)\n",
            gbm_cm$overall["Accuracy"], macro_f1(gbm_cm), seizure_recall(gbm_cm), gbm_elapsed))

results_df <- data.frame(
  model = c("Random Forest", "Gradient Boosting"),
  baseline_accuracy = baseline$accuracy[match(c("Random Forest", "Gradient Boosting"), baseline$model)],
  tuned_accuracy = round(c(rf_cm$overall["Accuracy"], gbm_cm$overall["Accuracy"]), 4),
  baseline_macro_f1 = baseline$macro_f1[match(c("Random Forest", "Gradient Boosting"), baseline$model)],
  tuned_macro_f1 = round(c(macro_f1(rf_cm), macro_f1(gbm_cm)), 4),
  baseline_seizure_recall = baseline$seizure_recall[match(c("Random Forest", "Gradient Boosting"), baseline$model)],
  tuned_seizure_recall = round(c(seizure_recall(rf_cm), seizure_recall(gbm_cm)), 4),
  search_time_sec = round(c(rf_elapsed, gbm_elapsed), 1)
)
write.csv(results_df, "../results/models/tuned_model_comparison_R.csv", row.names = FALSE)
cat("\n=== Tuning summary (baseline -> tuned) ===\n")
print(results_df, row.names = FALSE)
