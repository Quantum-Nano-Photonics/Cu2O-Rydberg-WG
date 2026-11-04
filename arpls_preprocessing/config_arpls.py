
from pathlib import Path
import os

REPOSITORY_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# DATA DIRECTORY
# ============================================================

# Path to the folder containing the raw experimental spectra.
# Replace this example path with your own data directory.

SPECTRA_DIR = Path(
    os.environ.get("CU2O_RAW_DATA_DIR", str(REPOSITORY_DIR / "data" / "raw"))
).expanduser()


# ============================================================
# EXPERIMENTAL CONFIGURATION
# ============================================================

# Collection geometry: "cI", "cG1", or "cG2"
COLLECTION = "cI"

# Excitation geometry: island
EXCITATION = "eI"

# Excitation polarization waveplate angle (degrees)
EXPOL_TO_ANALYZE = 0.0

# Regions of interest to analyze
ROIS_TO_ANALYZE = (1, 2)


# ============================================================
# INPUT FILE SELECTION
# ============================================================

FILE_PATTERN = "*.csv"

# Only files containing all these strings are selected.
MEASUREMENT_REQUIRED_TEXT = (
    f"_{COLLECTION}_",
    f"_{EXCITATION}_",
    "_120s_",
)

# Expected CSV column names
WAVELENGTH_COLUMN = "Wavelength"
INTENSITY_COLUMN = "Intensity"


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Separate output directories for each configuration.
RESULTS_DIR = (
    REPOSITORY_DIR
    / "results"
    / "yellow"
    / f"results_arpls_{COLLECTION}_{EXCITATION}_no_background"
)

OUTPUT_LABEL = (
    f"{COLLECTION}_{EXCITATION}_"
    f"expol{EXPOL_TO_ANALYZE:g}_"
    f"yellow_noBG_arpls"
)


# ============================================================
# BACKGROUND HANDLING
# ============================================================

# No separate background measurement is subtracted.
#
# The original 120 s spectrum is processed directly.
# No acquisition-time scaling is applied.
#
# A smooth spectral baseline is estimated and subtracted
# using the arPLS algorithm.


# ============================================================
# WAVELENGTH REGION — YELLOW EXCITONS
# ============================================================

# Full spectral window used for baseline estimation.
BASELINE_MIN_NM = 565.0
BASELINE_MAX_NM = 590.0

# Integration window covering the yellow excitonic peaks.
INTEGRATION_MIN_NM = 573.5
INTEGRATION_MAX_NM = 581.5


# ============================================================
# BASELINE CORRECTION
# ============================================================

# Adaptive iteratively reweighted penalized least squares
BASELINE_METHOD = "arpls"

# Baseline smoothness regularization
BASELINE_LAMBDA = 1e5

# arPLS convergence parameters
ARPLS_CONVERGENCE_RATIO = 1e-6
ARPLS_MAX_ITERATIONS = 100

# ALS settings retained for compatibility with the
# shared baseline estimation interface.
# These are not used when BASELINE_METHOD = "arpls".

ALS_ASYMMETRY_P = 0.001
ALS_ITERATIONS = 20


# ============================================================
# SPIKE DETECTION AND CORRECTION
# ============================================================

# Rolling median window (number of spectral points)
SPIKE_WINDOW_POINTS = 9

# Threshold based on robust noise standard deviation
SPIKE_SIGMA_THRESHOLD = 8.0


# ============================================================
# ROI COLLECTION-EFFICIENCY CORRECTION
# ============================================================

# Experimental calibration:
# ROI1 / ROI2 = 0.728
#
# The ROI2 baseline-corrected spectrum is multiplied
# by this factor to compensate for relative
# collection efficiency.
#
# ROI1 is left unchanged.
#
# Both unscaled and scaled corrected spectra
# are retained in the output CSV files.

APPLY_ROI2_COLLECTION_CORRECTION = True

ROI2_COLLECTION_SCALE_FACTOR = 0.728


# ============================================================
# FIGURE SETTINGS
# ============================================================

FIGURE_WIDTH = 11
FIGURE_HEIGHT = 7

# Overview figures
SHOW_BASELINE_CORRECTED_SPECTRA = True
SHOW_INTEGRAL_PLOT = True
SHOW_ROI_DIFFERENCE_PLOT = True

# Individual baseline diagnostics:
#   - Raw experimental spectrum
#   - Despiked spectrum
#   - Estimated arPLS baseline
#   - Baseline-corrected spectrum

SHOW_BASELINE_DIAGNOSTICS = False
SAVE_BASELINE_DIAGNOSTICS = True


# ============================================================
# SAVING SETTINGS
# ============================================================

SAVE_FIGURES = True
SAVE_RESULTS_TABLE = True

# Save the corrected spectrum for every
# collection angle and ROI.
#
# Each CSV contains:
#   wavelength_nm
#   raw_intensity_counts
#   despiked_intensity_counts
#   arpls_baseline_counts
#   corrected_intensity_original_counts
#   collection_scale_factor
#   corrected_intensity_analysis_counts

SAVE_CORRECTED_SPECTRA = True
