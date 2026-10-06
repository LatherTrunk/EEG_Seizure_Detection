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
#   2. Produces an effect-size-based answer (eta-squared, with a rank-based
#      Kruskal-Wallis check) to "does spectral entropy or time-domain entropy
#      separate the classes better", for ALL engineered features -- a THIRD
#      independent line of evidence alongside Python SHAP and R's own
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
write.csv(normality_results, "../results/models/shapiro_wilk_R.csv", row.names = FALSE)
cat(sprintf("\n%d of %d (feature, class) combinations are normally distributed.\n",
            sum(normality_results$normal), nrow(normality_results)))
cat("-> Non-normal data is why Random Forest / tree-based methods (no normality assumption)\n")
cat("   are a better default here than parametric models like LDA.\n")

# ---------- 3. One-way ANOVA + effect size (eta-squared) for ALL features ----------
# Run on every engineered feature, not just the five used in the entropy comparison,
# so the ranking below is a ranking of the whole feature set. Because Shapiro-Wilk
# says the data are almost never normal, a rank-based Kruskal-Wallis effect size
# (eta^2_H) is reported next to the ANOVA eta^2 as a robustness check.
cat("\n=== One-way ANOVA across 5 classes, ALL features (which features REALLY separate classes?) ===\n")
all_features <- setdiff(colnames(features), c("label", "label_binary", "class_name"))
n_total <- nrow(features)
n_groups <- length(class_names)
anova_results <- lapply(all_features, function(f) {
  s <- summary(aov(as.formula(paste(f, "~ class_name")), data = features))[[1]]
  eta_sq <- s["class_name", "Sum Sq"] / sum(s[, "Sum Sq"])
  kw <- kruskal.test(features[[f]] ~ features$class_name)
  eta_h <- (unname(kw$statistic) - n_groups + 1) / (n_total - n_groups)
  data.frame(
    feature = f, F_statistic = round(s["class_name", "F value"], 1),
    p_value = signif(s["class_name", "Pr(>F)"], 3), eta_squared = round(eta_sq, 4),
    kw_eta_squared = round(eta_h, 4),
    effect_size = ifelse(eta_sq > 0.14, "large", ifelse(eta_sq > 0.06, "medium", "small")),
    in_entropy_comparison = f %in% key_features
  )
})
anova_df <- do.call(rbind, anova_results)
anova_df <- anova_df[order(-anova_df$eta_squared), ]
anova_df$rank <- seq_len(nrow(anova_df))
write.csv(anova_df, "../results/models/anova_effect_sizes_R.csv", row.names = FALSE)

cat("\n=== All features ranked by effect size (eta-squared) ===\n")
print(anova_df[, c("rank", "feature", "eta_squared", "kw_eta_squared", "effect_size")], row.names = FALSE)
cat("\nNote: std, rms, peak_to_peak and mean_abs_diff are the redundant magnitude features\n")
cat("pruned in the modeling pipeline (r >= 0.90 with variance); they are kept here so the\n")
cat("ranking covers every engineered feature.\n")

# ---------- Tukey HSD post-hoc: which class pairs differ, and by how much? ----------
# Significance and magnitude are reported separately. With n = 2,300 per class even tiny
# differences are "significant", so Cohen's d (difference / residual SD) is the better
# guide to practical importance.
tukey_table <- function(f) {
  m <- aov(as.formula(paste(f, "~ class_name")), data = features)
  tk <- TukeyHSD(m)$class_name
  resid_sd <- sqrt(deviance(m) / df.residual(m))
  data.frame(feature = f, comparison = rownames(tk), diff = round(tk[, "diff"], 4),
             lwr = round(tk[, "lwr"], 4), upr = round(tk[, "upr"], 4),
             p_adj = signif(tk[, "p adj"], 3), cohens_d = round(tk[, "diff"] / resid_sd, 3),
             row.names = NULL)
}
tukey_all <- do.call(rbind, lapply(c("spectral_entropy", "sample_entropy"), tukey_table))
write.csv(tukey_all, "../results/models/tukey_hsd_R.csv", row.names = FALSE)

for (f in c("spectral_entropy", "sample_entropy")) {
  cat(sprintf("\n=== Tukey HSD post-hoc: %s, Seizure vs. each other class ===\n", f))
  rows <- tukey_all[tukey_all$feature == f & grepl("Seizure", tukey_all$comparison), ]
  for (k in seq_len(nrow(rows))) {
    d <- abs(rows$cohens_d[k])
    cat(sprintf("%-22s diff=%7.4f  p_adj=%-8s |d|=%.2f  %s, %s\n", rows$comparison[k], rows$diff[k],
                format(rows$p_adj[k]), d,
                ifelse(rows$p_adj[k] < 0.05, "significant", "NOT significant"),
                ifelse(d >= 0.8, "large effect", ifelse(d >= 0.5, "medium effect",
                       ifelse(d >= 0.2, "small effect", "negligible effect")))))
  }
}
cat("\n-> Read significance and magnitude together: a comparison can be statistically\n")
cat("   significant yet practically small, which is how the Seizure vs. eyes-open/closed\n")
cat("   spectral-entropy differences should be described.\n")

# ---------- 4. PCA for class separability (no classifier involved) ----------
cat("\n=== PCA: how separable are the 5 classes in reduced dimensions? ===\n")
redundant <- c("std", "rms", "peak_to_peak", "mean_abs_diff")
feature_cols <- setdiff(colnames(features), c("label", "label_binary", "class_name", redundant))
pca_input <- scale(features[, feature_cols])
pca <- prcomp(pca_input, center = FALSE, scale. = FALSE)  # already scaled above

var_explained <- summary(pca)$importance[2, 1:5] * 100
cat("Variance explained by PC1-PC5:", paste(round(var_explained, 1), collapse = "%, "), "%\n")
pca_variance <- data.frame(component = paste0("PC", 1:5), variance_pct = round(var_explained, 2),
                           cumulative_pct = round(cumsum(var_explained), 2))
write.csv(pca_variance, "../results/models/pca_variance_R.csv", row.names = FALSE)
pca_loadings <- data.frame(feature = rownames(pca$rotation), PC1 = round(pca$rotation[, 1], 3),
                           PC2 = round(pca$rotation[, 2], 3), row.names = NULL)
write.csv(pca_loadings, "../results/models/pca_loadings_R.csv", row.names = FALSE)

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
