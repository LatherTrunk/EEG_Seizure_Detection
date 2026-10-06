# 06_statistical_analysis.R
#
# Genuine statistical data science, as distinct from the modeling work in
# 02/04 -- hypothesis testing, effect sizes, and dimensionality reduction,
# not "train a model and report accuracy." Everything here uses base R's
# `stats` package (aov, TukeyHSD, shapiro.test, cor.test, prcomp) -- no
# external packages needed, which is itself worth noting: this is the
# kind of work R was built for.
#
# Three things this does that nothing else in the project does:
#   1. Tests whether the class differences already SEEN in the EDA plots
#      are actually statistically significant, not just visually suggestive.
#   2. Produces an effect-size-based answer (eta-squared) to "does spectral
#      entropy or time-domain entropy separate the classes better" -- a
#      THIRD independent line of evidence alongside Python SHAP and R's own
#      Random Forest Gini importance (05_comparative_visualizations.R).
#   3. PCA for 2D class-separability visualization -- a standard
#      exploratory technique that involves no trained classifier at all.

suppressMessages(library(ggplot2))

features <- read.csv("../data/processed/engineered_features_R.csv")
class_names <- c("Seizure", "Tumor", "Healthy", "EyesClosed", "EyesOpen")
features$class_name <- factor(class_names[features$label], levels = class_names)

key_features <- c("variance", "spectral_entropy", "sample_entropy", "permutation_entropy", "beta_power")

# ---------- 1. Descriptive statistics by class ----------
cat("=== Descriptive statistics by class ===\n")
desc_stats <- do.call(rbind, lapply(key_features, function(f) {
  do.call(rbind, lapply(class_names, function(cl) {
    x <- features[[f]][features$class_name == cl]
    data.frame(feature = f, class = cl, mean = round(mean(x), 3), sd = round(sd(x), 3),
               median = round(median(x), 3), skewness = round(mean((x - mean(x))^3) / sd(x)^3, 3))
  }))
}))
print(desc_stats, row.names = FALSE)
write.csv(desc_stats, "../results/models/descriptive_statistics_R.csv", row.names = FALSE)

# ---------- 2. Normality check (Shapiro-Wilk, sampled to 500/class -- test has a 5000-row cap) ----------
cat("\n=== Shapiro-Wilk normality test (p < 0.05 = NOT normal), n=500 sample per class ===\n")
set.seed(42)
normality_results <- do.call(rbind, lapply(key_features, function(f) {
  do.call(rbind, lapply(class_names, function(cl) {
    x <- features[[f]][features$class_name == cl]
    x_sample <- sample(x, min(500, length(x)))
    sw <- shapiro.test(x_sample)
    data.frame(feature = f, class = cl, W = round(sw$statistic, 4), p_value = signif(sw$p.value, 3),
               normal = sw$p.value >= 0.05)
  }))
}))
print(normality_results, row.names = FALSE)
cat(sprintf("\n%d of %d (feature, class) combinations are normally distributed.\n",
            sum(normality_results$normal), nrow(normality_results)))
cat("-> Non-normal data is why Random Forest / tree-based methods (no normality assumption)\n")
cat("   are a better default here than parametric models like LDA.\n")

# ---------- 3. One-way ANOVA + effect size (eta-squared) + Tukey HSD ----------
cat("\n=== One-way ANOVA across 5 classes (which features REALLY separate classes?) ===\n")
anova_results <- list()
for (f in key_features) {
  model <- aov(as.formula(paste(f, "~ class_name")), data = features)
  s <- summary(model)[[1]]
  ss_between <- s["class_name", "Sum Sq"]
  ss_total <- sum(s[, "Sum Sq"])
  eta_sq <- ss_between / ss_total
  anova_results[[f]] <- data.frame(
    feature = f, F_statistic = round(s["class_name", "F value"], 1),
    p_value = signif(s["class_name", "Pr(>F)"], 3), eta_squared = round(eta_sq, 4)
  )
  cat(sprintf("%-20s F=%.1f  p=%s  eta^2=%.4f  %s\n", f, s["class_name", "F value"],
              signif(s["class_name", "Pr(>F)"], 3), eta_sq,
              ifelse(eta_sq > 0.14, "(large effect)", ifelse(eta_sq > 0.06, "(medium effect)", "(small effect)"))))
}
anova_df <- do.call(rbind, anova_results)
anova_df <- anova_df[order(-anova_df$eta_squared), ]
write.csv(anova_df, "../results/models/anova_effect_sizes_R.csv", row.names = FALSE)

cat("\n=== Ranked by effect size (eta-squared) -- which feature ACTUALLY separates the 5 classes best ===\n")
print(anova_df, row.names = FALSE)

# ---------- Tukey HSD specifically for spectral_entropy (does Seizure actually differ from EyesOpen/EyesClosed?) ----------
cat("\n=== Tukey HSD post-hoc: spectral_entropy, Seizure vs. each other class ===\n")
se_model <- aov(spectral_entropy ~ class_name, data = features)
tukey_se <- TukeyHSD(se_model)
seizure_rows <- grep("Seizure", rownames(tukey_se$class_name))
print(round(tukey_se$class_name[seizure_rows, ], 4))
cat("-> If the Seizure-EyesOpen / Seizure-EyesClosed rows include 0 in their confidence\n")
cat("   interval (lwr<0<upr), the difference is NOT statistically significant -- this is the\n")
cat("   formal statistical confirmation of the Section 3.3 finding that the binary\n")
cat("   'entropy is higher in seizure' claim doesn't hold against normal resting states.\n")

cat("\n=== Tukey HSD post-hoc: sample_entropy, Seizure vs. each other class ===\n")
samp_model <- aov(sample_entropy ~ class_name, data = features)
tukey_samp <- TukeyHSD(samp_model)
seizure_rows2 <- grep("Seizure", rownames(tukey_samp$class_name))
print(round(tukey_samp$class_name[seizure_rows2, ], 4))

# ---------- 4. PCA for class separability (no classifier involved) ----------
cat("\n=== PCA: how separable are the 5 classes in reduced dimensions? ===\n")
redundant <- c("std", "rms", "peak_to_peak", "mean_abs_diff")
feature_cols <- setdiff(colnames(features), c("label", "label_binary", "class_name", redundant))
pca_input <- scale(features[, feature_cols])
pca <- prcomp(pca_input, center = FALSE, scale. = FALSE)  # already scaled above

var_explained <- summary(pca)$importance[2, 1:5] * 100
cat("Variance explained by PC1-PC5:", paste(round(var_explained, 1), collapse = "%, "), "%\n")

pca_df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2], class = features$class_name)
p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = class)) +
  geom_point(alpha = 0.4, size = 0.8) +
  labs(title = "PCA of 19 Engineered Features",
       subtitle = sprintf("PC1 (%.1f%% var) vs. PC2 (%.1f%% var) -- no classifier involved", var_explained[1], var_explained[2]),
       x = sprintf("PC1 (%.1f%%)", var_explained[1]), y = sprintf("PC2 (%.1f%%)", var_explained[2])) +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")
ggsave("../results/plots/plot13_R_pca_separability.png", p_pca, width = 8, height = 6, dpi = 120)
cat("Saved plot13_R_pca_separability.png\n")

cat("\nAll statistical analysis outputs saved to ../results/models/ and ../results/plots/\n")
