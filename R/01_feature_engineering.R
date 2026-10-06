# 01_feature_engineering.R
#
# Pure-R port of the feature engineering pipeline (previously Python-only:
# eeg_feature_extraction.py + nonlinear_entropy_features.py). This is the
# core preprocessing step of the whole project, and it now runs end-to-end
# in R: raw 178-point EEG signal -> 23 engineered features, with no
# dependency on the Python output.
#
# 21 of the 23 features mirror the Python version exactly (time-domain +
# FFT-based frequency-domain). Sample entropy and permutation entropy are
# HAND-IMPLEMENTED below (standard algorithms, no package) since this
# environment has no CRAN access and these aren't in Ubuntu's apt-packaged
# R library -- they are not approximations of the Python results, they are
# the same mathematical definitions computed independently in R.

suppressMessages(library(reshape2))

raw <- read.csv("../data/raw/Data.csv")
signal_cols <- grep("^X", colnames(raw), value = TRUE)
signal_matrix <- as.matrix(raw[, signal_cols])
n_samples <- nrow(signal_matrix)
fs <- 178  # samples per 1-second window -> sampling rate in Hz

cat(sprintf("Loaded %d samples x %d signal points\n", n_samples, length(signal_cols)))

# ---------- time-domain feature helpers ----------
zero_crossing_rate <- function(x) {
  sum(diff(sign(x)) != 0) / (length(x) - 1)
}

# ---------- frequency-domain helpers (FFT via base R) ----------
band_power <- function(freqs, power, lo, hi) {
  idx <- which(freqs >= lo & freqs < hi)
  if (length(idx) == 0) return(0)
  sum(power[idx])
}

spectral_entropy <- function(power) {
  # Raw Shannon entropy in BITS (log base 2), not normalized by log(n_bins)
  # -- matches the Python definition used throughout this project (range
  # ~0-4.5). Two bugs were caught here by validating against the Python
  # output before trusting this script: an earlier version normalized by
  # log(n_bins) (wrong scale entirely), and a second version used natural
  # log instead of log2 (consistent ~1.44x = 1/ln(2) ratio vs. Python).
  p <- power / sum(power)
  p <- p[p > 0]
  -sum(p * log2(p))
}

# ---------- sample entropy (hand-implemented, m=2, r=0.2*SD) ----------
# Vectorized via outer()/pmax() rather than a row-by-row apply loop --
# ~50x faster in pure R, same Chebyshev-distance definition, verified to
# give identical counts to the (slower) loop version on test rows.
sample_entropy <- function(x, m = 2, r_mult = 0.2) {
  r <- r_mult * sd(x)

  count_matches <- function(m_len) {
    templates <- embed(x, m_len)
    n <- nrow(templates)
    dist_mat <- matrix(0, n, n)
    for (k in 1:m_len) {
      dist_mat <- pmax(dist_mat, abs(outer(templates[, k], templates[, k], "-")))
    }
    diag(dist_mat) <- Inf  # exclude self-matches
    sum(dist_mat <= r) / 2  # symmetric matrix double-counts each pair
  }

  B <- count_matches(m)
  A <- count_matches(m + 1)
  if (B == 0 || A == 0) return(NA)
  -log(A / B)
}

# ---------- permutation entropy (hand-implemented, order=4, normalized) ----------
permutation_entropy <- function(x, order = 4, normalize = TRUE) {
  N <- length(x)
  n_patterns <- N - order + 1
  pattern_counts <- list()

  for (i in 1:n_patterns) {
    window <- x[i:(i + order - 1)]
    pattern <- paste(rank(window, ties.method = "first"), collapse = "")
    pattern_counts[[pattern]] <- (pattern_counts[[pattern]] %||% 0) + 1
  }

  counts <- unlist(pattern_counts)
  p <- counts / sum(counts)
  pe <- -sum(p * log(p))
  if (normalize) pe <- pe / log(factorial(order))
  pe
}
`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------- main extraction loop ----------
cat("Extracting features (this includes O(n^2) sample entropy per row, ~minutes for 11,500 rows)...\n")

results <- vector("list", n_samples)
freqs <- (0:(fs - 1)) * (fs / fs) / fs * fs / 2  # placeholder, replaced below properly
freq_bins <- seq(0, fs / 2, length.out = floor(fs / 2) + 1)

t0 <- Sys.time()
for (i in 1:n_samples) {
  x <- signal_matrix[i, ]

  # time domain
  mean_v <- mean(x); sd_v <- sd(x); var_v <- var(x)
  skew_v <- sum((x - mean_v)^3) / (length(x) * sd_v^3)
  kurt_v <- sum((x - mean_v)^4) / (length(x) * sd_v^4) - 3
  ptp_v <- max(x) - min(x)
  zcr_v <- zero_crossing_rate(x)
  rms_v <- sqrt(mean(x^2))
  mad_v <- mean(abs(diff(x)))

  # frequency domain via FFT
  fft_vals <- fft(x)
  n_half <- floor(length(x) / 2)
  power <- (Mod(fft_vals[1:n_half])^2)
  freqs_hz <- (0:(n_half - 1)) * (fs / length(x))

  delta_p <- band_power(freqs_hz, power, 0.5, 4)
  theta_p <- band_power(freqs_hz, power, 4, 8)
  alpha_p <- band_power(freqs_hz, power, 8, 13)
  beta_p  <- band_power(freqs_hz, power, 13, 30)
  gamma_p <- band_power(freqs_hz, power, 30, 60)
  total_p <- sum(power)
  dom_freq <- freqs_hz[which.max(power)]
  spec_ent <- spectral_entropy(power)

  results[[i]] <- data.frame(
    mean = mean_v, std = sd_v, variance = var_v, skewness = skew_v, kurtosis = kurt_v,
    peak_to_peak = ptp_v, zero_crossing_rate = zcr_v, rms = rms_v, mean_abs_diff = mad_v,
    delta_power = delta_p, delta_rel_power = delta_p / total_p,
    theta_power = theta_p, theta_rel_power = theta_p / total_p,
    alpha_power = alpha_p, alpha_rel_power = alpha_p / total_p,
    beta_power = beta_p, beta_rel_power = beta_p / total_p,
    gamma_power = gamma_p, gamma_rel_power = gamma_p / total_p,
    dominant_freq = dom_freq, spectral_entropy = spec_ent,
    sample_entropy = sample_entropy(x), permutation_entropy = permutation_entropy(x),
    label = raw$y[i]
  )

  if (i %% 1000 == 0) cat(sprintf("  %d/%d  (%.1f min elapsed)\n", i, n_samples,
                                   as.numeric(difftime(Sys.time(), t0, units = "mins"))))
}

features_r <- do.call(rbind, results)
features_r$label_binary <- ifelse(features_r$label == 1, 1, 0)

write.csv(features_r, "../data/processed/engineered_features_R.csv", row.names = FALSE)
cat(sprintf("\nDone in %.1f minutes. Saved engineered_features_R.csv (%d rows x %d cols)\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins")), nrow(features_r), ncol(features_r)))
