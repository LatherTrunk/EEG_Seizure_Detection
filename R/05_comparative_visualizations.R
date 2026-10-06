# 05_comparative_visualizations.R
#
# R-native comparative visualizations (DA2 rubric requirement) -- the R
# counterpart to generate_comparative_visualizations.py. Produces
# confusion matrices, ROC curves (Seizure vs. rest, one-vs-rest), and a
# Random Forest feature importance plot, using the tuned Random Forest
# from 04_hyperparameter_tuning.R as the representative model.

suppressMessages({
  library(caret)
  library(randomForest)
  library(gbm)
  library(pROC)
  library(ggplot2)
  library(reshape2)
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

# Retrain the two representative models with their tuned hyperparameters
# (from 04_hyperparameter_tuning.R) specifically to get class probabilities
# -- the saved comparison CSVs only have point predictions.
rf_model <- randomForest(x = X_train, y = y_train, mtry = 6, ntree = 300)
gbm_model <- train(x = X_train, y = y_train, method = "gbm",
                    trControl = trainControl(method = "none"),
                    tuneGrid = data.frame(n.trees = 150, interaction.depth = 4, shrinkage = 0.1, n.minobsinnode = 10),
                    verbose = FALSE)

rf_proba <- predict(rf_model, X_test, type = "prob")
gbm_proba <- predict(gbm_model, X_test, type = "prob")
rf_pred <- predict(rf_model, X_test)
gbm_pred <- predict(gbm_model, X_test)

# ---------- Confusion matrices (ggplot2 heatmap) ----------
plot_confusion <- function(y_true, y_pred, title) {
  cm_table <- as.data.frame(table(Predicted = y_pred, Actual = y_true))
  ggplot(cm_table, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), size = 4) +
    scale_fill_gradient(low = "white", high = "#2E86AB") +
    labs(title = title) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

p_rf_cm <- plot_confusion(y_test, rf_pred, "Random Forest (R, tuned)")
p_gbm_cm <- plot_confusion(y_test, gbm_pred, "Gradient Boosting (R, tuned)")

library(gridExtra)
png("../results/plots/plot10_R_confusion_matrices.png", width = 1400, height = 600, res = 120)
grid.arrange(p_rf_cm, p_gbm_cm, ncol = 2)
dev.off()
cat("Saved plot10_R_confusion_matrices.png\n")

# ---------- ROC curves (Seizure vs. rest, one-vs-rest) ----------
y_test_binary <- ifelse(y_test == "Seizure", 1, 0)
roc_rf <- roc(y_test_binary, rf_proba[, "Seizure"], quiet = TRUE)
roc_gbm <- roc(y_test_binary, gbm_proba[, "Seizure"], quiet = TRUE)

png("../results/plots/plot11_R_roc_curves.png", width = 800, height = 700, res = 120)
plot(roc_rf, col = "#2E86AB", lwd = 2, main = "ROC - Seizure vs. Rest (R models)")
plot(roc_gbm, col = "#E67E22", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(sprintf("Random Forest (AUC=%.3f)", auc(roc_rf)),
                  sprintf("Gradient Boosting (AUC=%.3f)", auc(roc_gbm))),
       col = c("#2E86AB", "#E67E22"), lwd = 2)
dev.off()
cat("Saved plot11_R_roc_curves.png\n")

# ---------- Feature importance (Random Forest, Gini) ----------
imp <- importance(rf_model)
imp_df <- data.frame(feature = rownames(imp), importance = imp[, "MeanDecreaseGini"])
imp_df <- imp_df[order(imp_df$importance), ]
imp_df$feature <- factor(imp_df$feature, levels = imp_df$feature)

p_imp <- ggplot(imp_df, aes(x = feature, y = importance)) +
  geom_bar(stat = "identity", fill = "#2E86AB") +
  coord_flip() +
  labs(title = "Random Forest Feature Importance (R, Gini)", x = "", y = "Mean Decrease in Gini") +
  theme_minimal()
ggsave("../results/plots/plot12_R_feature_importance.png", p_imp, width = 8, height = 7, dpi = 120)
cat("Saved plot12_R_feature_importance.png\n")

cat("\nAll R comparative visualizations saved to ../results/plots/\n")
