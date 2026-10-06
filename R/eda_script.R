# =============================================================================
# EDA — Epileptic Seizure Recognition Dataset (DA1 requirement: min 5 visualizations)
# =============================================================================
# Needs two files in the same folder as this script:
#   1. Data.csv                 -> raw EEG signal (from Kaggle/UCI download)
#   2. engineered_features.csv  -> output of eeg_feature_extraction.py
#
# Install required packages ONCE (uncomment the line below and run it):
# install.packages(c("ggplot2", "reshape2", "corrplot"))

library(ggplot2)
library(reshape2)
library(corrplot)

# -----------------------------------------------------------------------------
# 0. Load data
# -----------------------------------------------------------------------------
raw <- read.csv("../data/raw/Data.csv")
feat <- read.csv("../data/processed/engineered_features.csv")

# Identify the 178 raw signal columns (X1..X178), regardless of any extra
# ID column (e.g. "Unnamed") the Kaggle mirror may include
signal_cols <- grep("^X[0-9]+$", names(raw), value = TRUE)
signal_cols <- signal_cols[order(as.numeric(sub("X", "", signal_cols)))]

# Labels as factors, with readable names for plots
class_labels <- c("1" = "Seizure", "2" = "Tumor region", "3" = "Healthy region",
                   "4" = "Eyes closed", "5" = "Eyes open")
raw$class_name <- factor(class_labels[as.character(raw$y)],
                          levels = class_labels)
feat$class_name <- factor(class_labels[as.character(feat$label)],
                           levels = class_labels)
feat$seizure <- factor(feat$label_binary, levels = c(0, 1),
                        labels = c("Non-seizure", "Seizure"))

# =============================================================================
# PLOT 1 — Class distribution
# =============================================================================
p1 <- ggplot(raw, aes(x = class_name, fill = class_name)) +
  geom_bar() +
  labs(title = "Class Distribution",
       x = "Class", y = "Number of samples") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 20, hjust = 1))

ggsave("../results/plots/plot1_class_distribution.png", p1, width = 7, height = 5)

# =============================================================================
# PLOT 2 — Raw EEG waveform: seizure vs. non-seizure sample
# =============================================================================
# Grab one example row from class 1 (seizure) and one from class 5 (eyes open,
# a "healthy baseline") so the contrast is as clean as possible
seizure_row   <- raw[raw$y == 1, signal_cols][1, ]
healthy_row   <- raw[raw$y == 5, signal_cols][1, ]

waveform_df <- data.frame(
  time_point = rep(1:178, 2),
  amplitude  = c(as.numeric(seizure_row), as.numeric(healthy_row)),
  type = rep(c("Seizure", "Non-seizure (eyes open)"), each = 178)
)

p2 <- ggplot(waveform_df, aes(x = time_point, y = amplitude, color = type)) +
  geom_line() +
  facet_wrap(~type, ncol = 1, scales = "free_y") +
  labs(title = "Raw EEG Waveform: Seizure vs. Non-seizure",
       x = "Time point (within 1-second window)", y = "Signal amplitude") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("../results/plots/plot2_raw_waveform_comparison.png", p2, width = 8, height = 6)

# =============================================================================
# PLOT 3 — Boxplots of key engineered features by class
# =============================================================================
box_df <- feat[, c("variance", "dominant_freq", "spectral_entropy", "seizure")]
box_df_melt <- melt(box_df, id.vars = "seizure")

p3 <- ggplot(box_df_melt, aes(x = seizure, y = value, fill = seizure)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  facet_wrap(~variable, scales = "free_y") +
  labs(title = "Engineered Feature Distributions: Seizure vs. Non-seizure",
       x = "", y = "") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("../results/plots/plot3_feature_boxplots.png", p3, width = 9, height = 5)

# =============================================================================
# PLOT 4 — Correlation heatmap of engineered features
# =============================================================================
feature_cols <- c("mean", "std", "variance", "skewness", "kurtosis",
                   "peak_to_peak", "zero_crossing_rate", "rms", "mean_abs_diff",
                   "delta_rel_power", "theta_rel_power", "alpha_rel_power",
                   "beta_rel_power", "gamma_rel_power", "dominant_freq",
                   "spectral_entropy")

corr_matrix <- cor(feat[, feature_cols])

png("../results/plots/plot4_correlation_heatmap.png", width = 800, height = 800)
corrplot(corr_matrix, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.8, tl.srt = 45,
         title = "Correlation Between Engineered Features",
         mar = c(0, 0, 2, 0))
dev.off()

# =============================================================================
# PLOT 5 — FFT power spectrum comparison across classes
# =============================================================================
# Average power spectrum per class, computed directly from raw signals
# (independent check on top of the relative-band features already engineered)
compute_avg_spectrum <- function(class_val, n_samples = 100) {
  rows <- raw[raw$y == class_val, signal_cols]
  rows <- rows[1:min(n_samples, nrow(rows)), ]  # cap for speed

  fs <- 178  # sampling rate: 178 points represent 1 second
  n <- ncol(rows)
  freqs <- (0:(n / 2)) * fs / n

  spectra <- apply(rows, 1, function(sig) {
    fft_vals <- fft(as.numeric(sig))
    power <- Mod(fft_vals[1:(n / 2 + 1)])^2
    power
  })
  rowMeans(spectra)
}

spectrum_df <- do.call(rbind, lapply(1:5, function(cls) {
  power <- compute_avg_spectrum(cls)
  data.frame(
    freq = (0:(length(power) - 1)) * 178 / 178,
    power = power,
    class_name = class_labels[as.character(cls)]
  )
}))

# Restrict to 0-60 Hz for readability (most seizure-relevant activity lives here)
spectrum_df <- spectrum_df[spectrum_df$freq <= 60, ]

p5 <- ggplot(spectrum_df, aes(x = freq, y = power, color = class_name)) +
  geom_line() +
  labs(title = "Average FFT Power Spectrum by Class",
       x = "Frequency (Hz)", y = "Power", color = "Class") +
  theme_minimal()

ggsave("../results/plots/plot5_fft_power_spectrum.png", p5, width = 8, height = 5)

# =============================================================================
cat("All 5 plots saved to the working directory:\n",
    "plot1_class_distribution.png\n",
    "plot2_raw_waveform_comparison.png\n",
    "plot3_feature_boxplots.png\n",
    "plot4_correlation_heatmap.png\n",
    "plot5_fft_power_spectrum.png\n")
