from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


# ============================================================
# FOLDERS
# ============================================================

# main.py:
#
# paper/code/lorentzian_fit_ROI2-ROI1/main.py

# This is the original joint fitting/integration implementation, with ONLY
# configuration of input/output folders generalized for command-line use.
# Fitting, analytical integration, result columns and plotting are unchanged.
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Joint H/V two-Lorentzian fitting and analytic 2p integration"
    )
    parser.add_argument(
        "--collection", choices=("island", "G1"), default="island",
        help="Collection path to use when --input-dir is not explicitly supplied"
    )
    parser.add_argument(
        "--paper-root", type=Path, default=Path(__file__).resolve().parent,
        help="Base directory containing results/yellow/... (default: script folder)"
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Directory containing already ARPLS-corrected ROI1/ROI2 CSV files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for fit-result CSVs, saved_spectra and figures"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Write figures but do not launch interactive windows"
    )
    return parser.parse_args()


ARGS = parse_args()
SCRIPT_FOLDER = Path(__file__).resolve().parent
PAPER_FOLDER = ARGS.paper_root.resolve()
COLLECTION_SUFFIX = "cI_eI" if ARGS.collection == "island" else "cG1_eI"

DATA_FOLDER = ARGS.input_dir if ARGS.input_dir is not None else (
    PAPER_FOLDER / "results" / "yellow"
    / f"results_arpls_{COLLECTION_SUFFIX}_no_background" / "corrected_spectra"
)
OUTPUT_FOLDER = ARGS.output_dir if ARGS.output_dir is not None else (
    PAPER_FOLDER / "results" / "yellow"
    / f"results_lorentzian_fit_ROI2_minus_ROI1_{COLLECTION_SUFFIX}"
)
DATA_FOLDER = DATA_FOLDER.resolve()
OUTPUT_FOLDER = OUTPUT_FOLDER.resolve()
if ARGS.no_show:
    plt.switch_backend("Agg")


OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# Folder specifically for all saved spectral data

SPECTRA_OUTPUT_FOLDER = (
    OUTPUT_FOLDER
    / "saved_spectra"
)


SPECTRA_OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ANGLES TO ANALYZE
# ============================================================

ANGLE_MIN = 150.0
ANGLE_MAX = 360.0


# ============================================================
# CSV COLUMNS
# ============================================================

# These spectra are ALREADY ARPLS corrected.

WAVELENGTH_COLUMN = "wavelength_nm"

INTENSITY_COLUMN = (
    "corrected_intensity_analysis_counts"
)


# ============================================================
# FIT RANGE
# ============================================================

FIT_MIN_NM = 573.5
FIT_MAX_NM = 581.5


# ============================================================
# PEAK 1
# ============================================================

PEAK_1_CENTER_GUESS_NM = 575.7

PEAK_1_CENTER_BOUNDS_NM = (
    574.8,
    576.5,
)


# ============================================================
# PEAK 2 = 2p
# ============================================================

PEAK_2_CENTER_GUESS_NM = 579.1

PEAK_2_CENTER_BOUNDS_NM = (
    578.2,
    580.0,
)


# ============================================================
# WIDTH SETTINGS
# ============================================================

FWHM_GUESS_NM = 0.8

FWHM_BOUNDS_NM = (
    0.35,
    2.5,
)


# ============================================================
# 2p INTEGRATION WINDOW
# ============================================================

P2_INTEGRATION_MIN_NM = 577.2
P2_INTEGRATION_MAX_NM = 581.5


# ============================================================
# PLOT SETTINGS
# ============================================================

FIGURE_WIDTH = 10
FIGURE_HEIGHT = 7

SHOW_RAW_DIFFERENCE = True

SAVE_INDIVIDUAL_FITS = True
SAVE_COMBINED_FIGURE = True

SAVE_2P_AREA_PLOT = True
SAVE_ROI_2P_AREA_PLOT = True


# ============================================================
# LORENTZIAN
# ============================================================

def lorentzian_height(
    wavelength,
    amplitude,
    center,
    fwhm,
):
    """
    Lorentzian parameterized by peak height.

    amplitude:
        Lorentzian height at the center.

    center:
        Peak center in nm.

    fwhm:
        Full width at half maximum in nm.
    """

    gamma = (
        fwhm / 2.0
    )

    return (
        amplitude
        * gamma**2
        / (
            (wavelength - center) ** 2
            + gamma**2
        )
    )


# ============================================================
# FULL LORENTZIAN AREA
# ============================================================

def lorentzian_full_area(
    amplitude,
    fwhm,
):
    """
    Full area from -infinity to +infinity.

    area = pi * amplitude * FWHM / 2

    Units:
        counts * nm
    """

    return (
        np.pi
        * amplitude
        * fwhm
        / 2.0
    )


# ============================================================
# FINITE LORENTZIAN AREA
# ============================================================

def lorentzian_area_between(
    amplitude,
    center,
    fwhm,
    wavelength_min,
    wavelength_max,
):
    """
    Exact Lorentzian integral between wavelength_min and
    wavelength_max.
    """

    gamma = (
        fwhm / 2.0
    )


    upper = np.arctan(
        (
            wavelength_max
            - center
        )
        / gamma
    )


    lower = np.arctan(
        (
            wavelength_min
            - center
        )
        / gamma
    )


    return float(
        amplitude
        * gamma
        * (
            upper
            - lower
        )
    )


# ============================================================
# READ SPECTRUM
# ============================================================

def read_spectrum(
    file_path,
):
    """
    Read one ARPLS-corrected spectrum.
    """

    df = pd.read_csv(
        file_path
    )


    if (
        WAVELENGTH_COLUMN
        not in df.columns
    ):
        raise ValueError(
            f"Column '{WAVELENGTH_COLUMN}' "
            f"not found in:\n{file_path}"
        )


    if (
        INTENSITY_COLUMN
        not in df.columns
    ):
        raise ValueError(
            f"Column '{INTENSITY_COLUMN}' "
            f"not found in:\n{file_path}"
        )


    wavelength = (
        df[
            WAVELENGTH_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )


    intensity = (
        df[
            INTENSITY_COLUMN
        ]
        .to_numpy(
            dtype=float
        )
    )


    valid = (
        np.isfinite(
            wavelength
        )
        & np.isfinite(
            intensity
        )
    )


    wavelength = (
        wavelength[
            valid
        ]
    )


    intensity = (
        intensity[
            valid
        ]
    )


    order = np.argsort(
        wavelength
    )


    wavelength = (
        wavelength[
            order
        ]
    )


    intensity = (
        intensity[
            order
        ]
    )


    # --------------------------------------------------------
    # COLLECTION SCALE FACTOR
    # --------------------------------------------------------

    if (
        "collection_scale_factor"
        in df.columns
    ):

        factors = (
            df[
                "collection_scale_factor"
            ]
            .dropna()
            .unique()
        )


        if len(factors) > 0:

            scale_factor = float(
                factors[0]
            )

        else:

            scale_factor = np.nan

    else:

        scale_factor = np.nan


    return (
        wavelength,
        intensity,
        scale_factor,
    )


# ============================================================
# EXTRACT ROI AND COLLECTION ANGLE
# ============================================================

def extract_file_information(
    file_path,
):
    """
    Example:

    ARPLS_corrected_ROI1_ColPolWP154p5deg.csv

    returns

    roi = 1
    angle = 154.5
    """

    pattern = re.compile(
        r"ARPLS_corrected_ROI"
        r"([12])"
        r"_ColPolWP"
        r"([0-9]+(?:p[0-9]+)?)"
        r"deg",
        flags=re.IGNORECASE,
    )


    match = pattern.search(
        file_path.stem
    )


    if match is None:

        return None


    roi = int(
        match.group(1)
    )


    angle = float(
        match.group(2)
        .replace(
            "p",
            ".",
        )
    )


    return (
        roi,
        angle,
    )


# ============================================================
# FIND ROI1 / ROI2 PAIRS
# ============================================================

def find_pairs():

    files = {}


    for path in DATA_FOLDER.glob(
        "*.csv"
    ):

        information = (
            extract_file_information(
                path
            )
        )


        if information is None:

            continue


        roi, angle = (
            information
        )


        if not (
            ANGLE_MIN
            <= angle
            <= ANGLE_MAX
        ):

            continue


        if angle not in files:

            files[
                angle
            ] = {}


        files[
            angle
        ][
            roi
        ] = path


    paired = {}


    for angle, roi_files in (
        files.items()
    ):

        if (
            1 in roi_files
            and 2 in roi_files
        ):

            paired[
                angle
            ] = roi_files


    return dict(
        sorted(
            paired.items()
        )
    )


# ============================================================
# JOINT FIT
# ============================================================

def fit_joint_spectra(
    wavelength,
    roi1,
    roi2,
):
    """
    Joint fit of ROI1 and ROI2.

    Shared parameters:
        center 1
        FWHM 1
        center 2
        FWHM 2

    Separate parameters:
        ROI1 amplitudes
        ROI2 amplitudes
        residual offsets
    """

    mask = (
        np.isfinite(
            wavelength
        )
        & np.isfinite(
            roi1
        )
        & np.isfinite(
            roi2
        )
        & (
            wavelength
            >= FIT_MIN_NM
        )
        & (
            wavelength
            <= FIT_MAX_NM
        )
    )


    x = (
        wavelength[
            mask
        ]
    )


    y1 = (
        roi1[
            mask
        ]
    )


    y2 = (
        roi2[
            mask
        ]
    )


    if x.size < 20:

        raise ValueError(
            "Not enough points "
            "inside the fitting range."
        )


    # ========================================================
    # INITIAL HEIGHT ESTIMATE
    # ========================================================

    def estimate_height(
        y,
        center,
        half_width=0.7,
    ):

        local = (
            np.abs(
                x - center
            )
            <= half_width
        )


        if not np.any(
            local
        ):

            return max(
                float(
                    np.max(
                        y
                    )
                ),
                1.0,
            )


        local_maximum = float(
            np.max(
                y[
                    local
                ]
            )
        )


        return max(
            local_maximum,
            1.0,
        )


    # ROI1

    roi1_peak1_guess = (
        estimate_height(
            y1,
            PEAK_1_CENTER_GUESS_NM,
        )
    )


    roi1_peak2_guess = (
        estimate_height(
            y1,
            PEAK_2_CENTER_GUESS_NM,
        )
    )


    # ROI2

    roi2_peak1_guess = (
        estimate_height(
            y2,
            PEAK_1_CENTER_GUESS_NM,
        )
    )


    roi2_peak2_guess = (
        estimate_height(
            y2,
            PEAK_2_CENTER_GUESS_NM,
        )
    )


    # ========================================================
    # PARAMETER ARRAY
    # ========================================================
    #
    # 0  center 1
    # 1  FWHM 1
    #
    # 2  center 2
    # 3  FWHM 2
    #
    # 4  ROI1 peak 1 amplitude
    # 5  ROI1 peak 2 amplitude
    # 6  ROI1 residual offset
    #
    # 7  ROI2 peak 1 amplitude
    # 8  ROI2 peak 2 amplitude
    # 9  ROI2 residual offset
    #
    # ========================================================

    p0 = np.array(
        [
            PEAK_1_CENTER_GUESS_NM,
            FWHM_GUESS_NM,

            PEAK_2_CENTER_GUESS_NM,
            FWHM_GUESS_NM,

            roi1_peak1_guess,
            roi1_peak2_guess,
            0.0,

            roi2_peak1_guess,
            roi2_peak2_guess,
            0.0,
        ],
        dtype=float,
    )


    lower = np.array(
        [
            PEAK_1_CENTER_BOUNDS_NM[0],
            FWHM_BOUNDS_NM[0],

            PEAK_2_CENTER_BOUNDS_NM[0],
            FWHM_BOUNDS_NM[0],

            0.0,
            0.0,
            -np.inf,

            0.0,
            0.0,
            -np.inf,
        ],
        dtype=float,
    )


    upper = np.array(
        [
            PEAK_1_CENTER_BOUNDS_NM[1],
            FWHM_BOUNDS_NM[1],

            PEAK_2_CENTER_BOUNDS_NM[1],
            FWHM_BOUNDS_NM[1],

            np.inf,
            np.inf,
            np.inf,

            np.inf,
            np.inf,
            np.inf,
        ],
        dtype=float,
    )


    # ========================================================
    # MODEL
    # ========================================================

    def evaluate_models(
        parameters,
        x_values,
    ):

        (
            center_1,
            fwhm_1,

            center_2,
            fwhm_2,

            roi1_amp_1,
            roi1_amp_2,
            roi1_offset,

            roi2_amp_1,
            roi2_amp_2,
            roi2_offset,
        ) = parameters


        line_1 = lorentzian_height(
            x_values,
            1.0,
            center_1,
            fwhm_1,
        )


        line_2 = lorentzian_height(
            x_values,
            1.0,
            center_2,
            fwhm_2,
        )


        model_roi1 = (
            roi1_offset
            + roi1_amp_1
            * line_1
            + roi1_amp_2
            * line_2
        )


        model_roi2 = (
            roi2_offset
            + roi2_amp_1
            * line_1
            + roi2_amp_2
            * line_2
        )


        return (
            model_roi1,
            model_roi2,
            line_1,
            line_2,
        )


    # ========================================================
    # RESIDUALS
    # ========================================================

    def residuals(
        parameters,
    ):

        (
            model_roi1,
            model_roi2,
            _,
            _,
        ) = evaluate_models(
            parameters,
            x,
        )


        residual_roi1 = (
            y1
            - model_roi1
        )


        residual_roi2 = (
            y2
            - model_roi2
        )


        return np.concatenate(
            [
                residual_roi1,
                residual_roi2,
            ]
        )


    # ========================================================
    # NOISE ESTIMATE
    # ========================================================

    combined = np.concatenate(
        [
            y1
            - np.median(
                y1
            ),

            y2
            - np.median(
                y2
            ),
        ]
    )


    median_combined = np.median(
        combined
    )


    mad = np.median(
        np.abs(
            combined
            - median_combined
        )
    )


    noise_sigma = (
        1.4826
        * mad
    )


    if (
        not np.isfinite(
            noise_sigma
        )
        or noise_sigma <= 0
    ):

        noise_sigma = 1.0


    # ========================================================
    # FIT
    # ========================================================

    result = least_squares(
        residuals,
        p0,
        bounds=(
            lower,
            upper,
        ),
        loss="soft_l1",
        f_scale=noise_sigma,
        x_scale="jac",
        max_nfev=100000,
    )


    parameters = (
        result.x
    )


    (
        center_1,
        fwhm_1,

        center_2,
        fwhm_2,

        roi1_amp_1,
        roi1_amp_2,
        roi1_offset,

        roi2_amp_1,
        roi2_amp_2,
        roi2_offset,
    ) = parameters


    # ========================================================
    # DENSE WAVELENGTH GRID
    # ========================================================

    dense_x = np.linspace(
        FIT_MIN_NM,
        FIT_MAX_NM,
        3000,
    )


    (
        dense_roi1,
        dense_roi2,
        dense_line_1,
        dense_line_2,
    ) = evaluate_models(
        parameters,
        dense_x,
    )


    # ========================================================
    # BASELINE-FREE INDIVIDUAL FITS
    # ========================================================

    roi1_peak1_dense = (
        roi1_amp_1
        * dense_line_1
    )


    roi1_peak2_dense = (
        roi1_amp_2
        * dense_line_2
    )


    roi2_peak1_dense = (
        roi2_amp_1
        * dense_line_1
    )


    roi2_peak2_dense = (
        roi2_amp_2
        * dense_line_2
    )


    roi1_peaks_only_dense = (
        roi1_peak1_dense
        + roi1_peak2_dense
    )


    roi2_peaks_only_dense = (
        roi2_peak1_dense
        + roi2_peak2_dense
    )


    # ========================================================
    # ROI2 - ROI1
    # ========================================================

    delta_amp_1 = (
        roi2_amp_1
        - roi1_amp_1
    )


    delta_amp_2 = (
        roi2_amp_2
        - roi1_amp_2
    )


    differential_peak_1 = (
        delta_amp_1
        * dense_line_1
    )


    differential_peak_2 = (
        delta_amp_2
        * dense_line_2
    )


    differential_offset = (
        roi2_offset
        - roi1_offset
    )


    differential_fit = (
        differential_offset
        + differential_peak_1
        + differential_peak_2
    )


    differential_peaks_only = (
        differential_peak_1
        + differential_peak_2
    )


    raw_difference = (
        y2
        - y1
    )


    # ========================================================
    # 2p AREAS
    # ========================================================

    # --------------------------------------------------------
    # FULL LORENTZIAN AREA
    # --------------------------------------------------------

    roi1_2p_area_full = (
        lorentzian_full_area(
            amplitude=roi1_amp_2,
            fwhm=fwhm_2,
        )
    )


    roi2_2p_area_full = (
        lorentzian_full_area(
            amplitude=roi2_amp_2,
            fwhm=fwhm_2,
        )
    )


    difference_2p_area_full = (
        roi2_2p_area_full
        - roi1_2p_area_full
    )


    # --------------------------------------------------------
    # AREA IN SELECTED WINDOW
    # --------------------------------------------------------

    roi1_2p_area_window = (
        lorentzian_area_between(
            amplitude=roi1_amp_2,
            center=center_2,
            fwhm=fwhm_2,
            wavelength_min=(
                P2_INTEGRATION_MIN_NM
            ),
            wavelength_max=(
                P2_INTEGRATION_MAX_NM
            ),
        )
    )


    roi2_2p_area_window = (
        lorentzian_area_between(
            amplitude=roi2_amp_2,
            center=center_2,
            fwhm=fwhm_2,
            wavelength_min=(
                P2_INTEGRATION_MIN_NM
            ),
            wavelength_max=(
                P2_INTEGRATION_MAX_NM
            ),
        )
    )


    difference_2p_area_window = (
        roi2_2p_area_window
        - roi1_2p_area_window
    )


    # ========================================================
    # DIFFERENTIAL R-SQUARED
    # ========================================================

    line_1_data = lorentzian_height(
        x,
        1.0,
        center_1,
        fwhm_1,
    )


    line_2_data = lorentzian_height(
        x,
        1.0,
        center_2,
        fwhm_2,
    )


    differential_on_data = (
        differential_offset
        + delta_amp_1
        * line_1_data
        + delta_amp_2
        * line_2_data
    )


    ss_res = np.sum(
        (
            raw_difference
            - differential_on_data
        ) ** 2
    )


    ss_tot = np.sum(
        (
            raw_difference
            - np.mean(
                raw_difference
            )
        ) ** 2
    )


    if ss_tot > 0:

        difference_r_squared = (
            1.0
            - ss_res
            / ss_tot
        )

    else:

        difference_r_squared = np.nan


    # ========================================================
    # RETURN
    # ========================================================

    return {

        # Experimental grid

        "x": x,

        "roi1": y1,
        "roi2": y2,

        "raw_difference": (
            raw_difference
        ),


        # Dense grid

        "dense_x": (
            dense_x
        ),


        # Full ROI fits including offset

        "dense_roi1": (
            dense_roi1
        ),

        "dense_roi2": (
            dense_roi2
        ),


        # Baseline-free ROI fits

        "roi1_peaks_only_dense": (
            roi1_peaks_only_dense
        ),

        "roi2_peaks_only_dense": (
            roi2_peaks_only_dense
        ),


        # Individual ROI peak components

        "roi1_peak1_dense": (
            roi1_peak1_dense
        ),

        "roi1_peak2_dense": (
            roi1_peak2_dense
        ),

        "roi2_peak1_dense": (
            roi2_peak1_dense
        ),

        "roi2_peak2_dense": (
            roi2_peak2_dense
        ),


        # Differential fit

        "difference_fit": (
            differential_fit
        ),

        "difference_peaks_only": (
            differential_peaks_only
        ),

        "difference_peak_1": (
            differential_peak_1
        ),

        "difference_peak_2": (
            differential_peak_2
        ),


        # Fit parameters

        "center_1": (
            center_1
        ),

        "fwhm_1": (
            fwhm_1
        ),

        "center_2": (
            center_2
        ),

        "fwhm_2": (
            fwhm_2
        ),


        # Amplitudes

        "roi1_amp_1": (
            roi1_amp_1
        ),

        "roi1_amp_2": (
            roi1_amp_2
        ),

        "roi2_amp_1": (
            roi2_amp_1
        ),

        "roi2_amp_2": (
            roi2_amp_2
        ),

        "delta_amp_1": (
            delta_amp_1
        ),

        "delta_amp_2": (
            delta_amp_2
        ),


        # 2p areas

        "roi1_2p_area_full": (
            roi1_2p_area_full
        ),

        "roi2_2p_area_full": (
            roi2_2p_area_full
        ),

        "difference_2p_area_full": (
            difference_2p_area_full
        ),

        "roi1_2p_area_window": (
            roi1_2p_area_window
        ),

        "roi2_2p_area_window": (
            roi2_2p_area_window
        ),

        "difference_2p_area_window": (
            difference_2p_area_window
        ),


        # Fit quality

        "difference_r_squared": (
            difference_r_squared
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "ROI2 - ROI1 JOINT LORENTZIAN ANALYSIS"
    )

    print(
        "=" * 78
    )


    print(
        "\nInput folder:"
    )

    print(
        DATA_FOLDER
    )


    print(
        "\nOutput folder:"
    )

    print(
        OUTPUT_FOLDER
    )


    print(
        "\nSaved spectra folder:"
    )

    print(
        SPECTRA_OUTPUT_FOLDER
    )


    print(
        "\nCollection-angle range:"
    )

    print(
        f"{ANGLE_MIN:g}° - "
        f"{ANGLE_MAX:g}°"
    )


    print(
        "\n2p integration range:"
    )

    print(
        f"{P2_INTEGRATION_MIN_NM:g} - "
        f"{P2_INTEGRATION_MAX_NM:g} nm"
    )


    # ========================================================
    # FIND FILE PAIRS
    # ========================================================

    pairs = find_pairs()


    if not pairs:

        raise FileNotFoundError(
            "No matching ROI1/ROI2 file pairs were found."
        )


    print(
        "\nAngles found:"
    )


    for angle in pairs:

        print(
            f"  {angle:g}°"
        )


    fit_results = {}

    rows = []


    # ========================================================
    # PROCESS EACH ANGLE
    # ========================================================

    for angle, files in (
        pairs.items()
    ):

        print(
            "\n"
            + "=" * 78
        )

        print(
            f"ANGLE = {angle:g}°"
        )

        print(
            "=" * 78
        )


        # ----------------------------------------------------
        # READ ROI1
        # ----------------------------------------------------

        (
            wavelength_1,
            roi1,
            roi1_scale,
        ) = read_spectrum(
            files[1]
        )


        # ----------------------------------------------------
        # READ ROI2
        # ----------------------------------------------------

        (
            wavelength_2,
            roi2,
            roi2_scale,
        ) = read_spectrum(
            files[2]
        )


        print(
            f"ROI1 collection scale factor: "
            f"{roi1_scale}"
        )


        print(
            f"ROI2 collection scale factor: "
            f"{roi2_scale}"
        )


        # ====================================================
        # COMMON WAVELENGTH GRID
        # ====================================================

        common_min = max(
            wavelength_1.min(),
            wavelength_2.min(),
        )


        common_max = min(
            wavelength_1.max(),
            wavelength_2.max(),
        )


        common_mask = (
            (
                wavelength_1
                >= common_min
            )
            & (
                wavelength_1
                <= common_max
            )
        )


        wavelength = (
            wavelength_1[
                common_mask
            ]
        )


        roi1_common = (
            roi1[
                common_mask
            ]
        )


        roi2_common = np.interp(
            wavelength,
            wavelength_2,
            roi2,
        )


        # ====================================================
        # JOINT FIT
        # ====================================================

        try:

            fit = fit_joint_spectra(
                wavelength,
                roi1_common,
                roi2_common,
            )

        except Exception as error:

            print(
                f"\n[FAILED] "
                f"{angle:g}°"
            )

            print(
                f"Reason: {error}"
            )

            continue


        fit_results[
            angle
        ] = fit


        # ====================================================
        # SAFE ANGLE STRING
        # ====================================================

        safe_angle = (
            f"{angle:g}"
            .replace(
                ".",
                "p",
            )
        )


        # ====================================================
        # SAVE MEASURED SPECTRA
        # ====================================================
        #
        # Actual data used in the fit.
        #
        # Includes:
        # ROI1
        # ROI2
        # ROI2 - ROI1
        #
        # ====================================================

        measured_dataframe = pd.DataFrame(
            {
                "wavelength_nm": (
                    fit[
                        "x"
                    ]
                ),

                "ROI1_counts": (
                    fit[
                        "roi1"
                    ]
                ),

                "ROI2_counts": (
                    fit[
                        "roi2"
                    ]
                ),

                "ROI2_minus_ROI1_counts": (
                    fit[
                        "raw_difference"
                    ]
                ),
            }
        )


        measured_file = (
            SPECTRA_OUTPUT_FOLDER
            / (
                "ROI2_minus_ROI1_"
                f"WP{safe_angle}deg_"
                "measured_spectrum.csv"
            )
        )


        measured_dataframe.to_csv(
            measured_file,
            index=False,
        )


        # ====================================================
        # SAVE FITTED SPECTRA
        # ====================================================
        #
        # This contains all smooth curves.
        #
        # ====================================================

        fitted_dataframe = pd.DataFrame(
            {
                "wavelength_nm": (
                    fit[
                        "dense_x"
                    ]
                ),

                # ROI1

                "ROI1_fit_with_offset_counts": (
                    fit[
                        "dense_roi1"
                    ]
                ),

                "ROI1_peaks_only_counts": (
                    fit[
                        "roi1_peaks_only_dense"
                    ]
                ),

                "ROI1_peak1_counts": (
                    fit[
                        "roi1_peak1_dense"
                    ]
                ),

                "ROI1_2p_counts": (
                    fit[
                        "roi1_peak2_dense"
                    ]
                ),


                # ROI2

                "ROI2_fit_with_offset_counts": (
                    fit[
                        "dense_roi2"
                    ]
                ),

                "ROI2_peaks_only_counts": (
                    fit[
                        "roi2_peaks_only_dense"
                    ]
                ),

                "ROI2_peak1_counts": (
                    fit[
                        "roi2_peak1_dense"
                    ]
                ),

                "ROI2_2p_counts": (
                    fit[
                        "roi2_peak2_dense"
                    ]
                ),


                # Differential

                "ROI2_minus_ROI1_fit_counts": (
                    fit[
                        "difference_fit"
                    ]
                ),

                "ROI2_minus_ROI1_peaks_only_counts": (
                    fit[
                        "difference_peaks_only"
                    ]
                ),

                "ROI2_minus_ROI1_peak1_counts": (
                    fit[
                        "difference_peak_1"
                    ]
                ),

                "ROI2_minus_ROI1_2p_counts": (
                    fit[
                        "difference_peak_2"
                    ]
                ),
            }
        )


        fitted_file = (
            SPECTRA_OUTPUT_FOLDER
            / (
                "ROI2_minus_ROI1_"
                f"WP{safe_angle}deg_"
                "fitted_spectrum.csv"
            )
        )


        fitted_dataframe.to_csv(
            fitted_file,
            index=False,
        )


        print(
            "\nSaved measured spectrum:"
        )

        print(
            measured_file
        )


        print(
            "\nSaved fitted spectrum:"
        )

        print(
            fitted_file
        )


        # ====================================================
        # RESULT ROW
        # ====================================================

        rows.append(
            {
                "angle_deg": angle,

                "roi1_scale_factor": (
                    roi1_scale
                ),

                "roi2_scale_factor": (
                    roi2_scale
                ),


                # --------------------------------------------
                # PEAK 1
                # --------------------------------------------

                "peak_1_center_nm": (
                    fit[
                        "center_1"
                    ]
                ),

                "peak_1_fwhm_nm": (
                    fit[
                        "fwhm_1"
                    ]
                ),

                "roi1_peak_1_height_counts": (
                    fit[
                        "roi1_amp_1"
                    ]
                ),

                "roi2_peak_1_height_counts": (
                    fit[
                        "roi2_amp_1"
                    ]
                ),

                "peak_1_roi2_minus_roi1_height_counts": (
                    fit[
                        "delta_amp_1"
                    ]
                ),


                # --------------------------------------------
                # PEAK 2 = 2p
                # --------------------------------------------

                "2p_center_nm": (
                    fit[
                        "center_2"
                    ]
                ),

                "2p_fwhm_nm": (
                    fit[
                        "fwhm_2"
                    ]
                ),

                "roi1_2p_height_counts": (
                    fit[
                        "roi1_amp_2"
                    ]
                ),

                "roi2_2p_height_counts": (
                    fit[
                        "roi2_amp_2"
                    ]
                ),

                "2p_roi2_minus_roi1_height_counts": (
                    fit[
                        "delta_amp_2"
                    ]
                ),


                # --------------------------------------------
                # FULL 2p AREA
                # --------------------------------------------

                "roi1_2p_area_full_counts_nm": (
                    fit[
                        "roi1_2p_area_full"
                    ]
                ),

                "roi2_2p_area_full_counts_nm": (
                    fit[
                        "roi2_2p_area_full"
                    ]
                ),

                "2p_roi2_minus_roi1_area_full_counts_nm": (
                    fit[
                        "difference_2p_area_full"
                    ]
                ),


                # --------------------------------------------
                # WINDOW 2p AREA
                # --------------------------------------------

                "roi1_2p_area_window_counts_nm": (
                    fit[
                        "roi1_2p_area_window"
                    ]
                ),

                "roi2_2p_area_window_counts_nm": (
                    fit[
                        "roi2_2p_area_window"
                    ]
                ),

                "2p_roi2_minus_roi1_area_window_counts_nm": (
                    fit[
                        "difference_2p_area_window"
                    ]
                ),


                # --------------------------------------------
                # FIT QUALITY
                # --------------------------------------------

                "difference_r_squared": (
                    fit[
                        "difference_r_squared"
                    ]
                ),
            }
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print(
            "\nPeak 1:"
        )

        print(
            f"  Center = "
            f"{fit['center_1']:.4f} nm"
        )

        print(
            f"  FWHM   = "
            f"{fit['fwhm_1']:.4f} nm"
        )

        print(
            f"  ROI2 - ROI1 height = "
            f"{fit['delta_amp_1']:.2f} counts"
        )


        print(
            "\n2p transition:"
        )

        print(
            f"  Center = "
            f"{fit['center_2']:.4f} nm"
        )

        print(
            f"  FWHM   = "
            f"{fit['fwhm_2']:.4f} nm"
        )


        print(
            "\n2p integrated area "
            f"{P2_INTEGRATION_MIN_NM:g}-"
            f"{P2_INTEGRATION_MAX_NM:g} nm:"
        )

        print(
            f"  ROI1 = "
            f"{fit['roi1_2p_area_window']:.2f} "
            f"counts·nm"
        )

        print(
            f"  ROI2 = "
            f"{fit['roi2_2p_area_window']:.2f} "
            f"counts·nm"
        )

        print(
            f"  ROI2 - ROI1 = "
            f"{fit['difference_2p_area_window']:.2f} "
            f"counts·nm"
        )


        print(
            f"\nDifference R² = "
            f"{fit['difference_r_squared']:.5f}"
        )


        # ====================================================
        # INDIVIDUAL FIGURE
        # ====================================================

        if SAVE_INDIVIDUAL_FITS:

            figure, axis = plt.subplots(
                figsize=(
                    8,
                    5,
                )
            )


            axis.plot(
                fit[
                    "x"
                ],
                fit[
                    "raw_difference"
                ],
                ".",
                markersize=3,
                alpha=0.45,
                label=(
                    "Measured ROI2 - ROI1"
                ),
            )


            axis.plot(
                fit[
                    "dense_x"
                ],
                fit[
                    "difference_fit"
                ],
                linewidth=2.2,
                label=(
                    "Difference of joint fits"
                ),
            )


            axis.plot(
                fit[
                    "dense_x"
                ],
                fit[
                    "difference_peak_1"
                ],
                "--",
                linewidth=1.3,
                label=(
                    "Peak 1 contribution"
                ),
            )


            axis.plot(
                fit[
                    "dense_x"
                ],
                fit[
                    "difference_peak_2"
                ],
                "--",
                linewidth=1.3,
                label=(
                    "2p contribution"
                ),
            )


            axis.axvspan(
                P2_INTEGRATION_MIN_NM,
                P2_INTEGRATION_MAX_NM,
                alpha=0.08,
                label=(
                    "2p integration window"
                ),
            )


            axis.axhline(
                0,
                linestyle=":",
                linewidth=1,
            )


            axis.set_xlabel(
                "Wavelength (nm)",
                fontsize=13,
            )


            axis.set_ylabel(
                "ROI2 - ROI1 intensity (counts)",
                fontsize=13,
            )


            axis.set_title(
                (
                    f"Collection WP "
                    f"{angle:g}°\n"
                    f"2p ROI2 - ROI1 area = "
                    f"{fit['difference_2p_area_window']:.1f} "
                    f"counts·nm"
                ),
                fontsize=13,
            )


            axis.tick_params(
                axis="both",
                labelsize=11,
            )


            axis.legend(
                fontsize=8,
            )


            figure.tight_layout()


            figure.savefig(
                OUTPUT_FOLDER
                / (
                    "ROI2_minus_ROI1_"
                    f"WP{safe_angle}deg.png"
                ),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )


            plt.close(
                figure
            )


    # ========================================================
    # CHECK SUCCESS
    # ========================================================

    if not rows:

        raise RuntimeError(
            "No successful fits were obtained."
        )


    # ========================================================
    # PARAMETER / AREA RESULTS TABLE
    # ========================================================

    results = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "angle_deg"
        )
        .reset_index(
            drop=True
        )
    )


    results_output_file = (
        OUTPUT_FOLDER
        / (
            "ROI2_minus_ROI1_"
            "joint_lorentzian_results.csv"
        )
    )


    results.to_csv(
        results_output_file,
        index=False,
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULT TABLE SAVED:"
    )

    print(
        results_output_file
    )


    # ========================================================
    # SAVE ALL FITTED DIFFERENCE SPECTRA TO ONE CSV
    # ========================================================
    #
    # THIS FILE CONTAINS THE SOLID COLORED CURVES IN YOUR
    # COMBINED FIGURE.
    #
    # ========================================================

    combined_fitted_spectra = None


    for angle in sorted(
        fit_results
    ):

        fit = (
            fit_results[
                angle
            ]
        )


        safe_angle = (
            f"{angle:g}"
            .replace(
                ".",
                "p",
            )
        )


        if combined_fitted_spectra is None:

            combined_fitted_spectra = pd.DataFrame(
                {
                    "wavelength_nm": (
                        fit[
                            "dense_x"
                        ]
                    )
                }
            )


        combined_fitted_spectra[
            f"WP{safe_angle}deg"
        ] = (
            fit[
                "difference_fit"
            ]
        )


    combined_fitted_file = (
        SPECTRA_OUTPUT_FOLDER
        / (
            "ROI2_minus_ROI1_"
            "ALL_FITTED_SPECTRA.csv"
        )
    )


    combined_fitted_spectra.to_csv(
        combined_fitted_file,
        index=False,
    )


    print(
        "\nAll fitted spectra saved:"
    )

    print(
        combined_fitted_file
    )


    # ========================================================
    # SAVE ALL BASELINE-FREE FITTED DIFFERENCE SPECTRA
    # ========================================================

    combined_peaks_only_spectra = None


    for angle in sorted(
        fit_results
    ):

        fit = (
            fit_results[
                angle
            ]
        )


        safe_angle = (
            f"{angle:g}"
            .replace(
                ".",
                "p",
            )
        )


        if (
            combined_peaks_only_spectra
            is None
        ):

            combined_peaks_only_spectra = (
                pd.DataFrame(
                    {
                        "wavelength_nm": (
                            fit[
                                "dense_x"
                            ]
                        )
                    }
                )
            )


        combined_peaks_only_spectra[
            f"WP{safe_angle}deg"
        ] = (
            fit[
                "difference_peaks_only"
            ]
        )


    combined_peaks_only_file = (
        SPECTRA_OUTPUT_FOLDER
        / (
            "ROI2_minus_ROI1_"
            "ALL_FITTED_PEAKS_ONLY_SPECTRA.csv"
        )
    )


    combined_peaks_only_spectra.to_csv(
        combined_peaks_only_file,
        index=False,
    )


    # ========================================================
    # SAVE ALL 2p FITTED CONTRIBUTIONS TO ONE CSV
    # ========================================================

    combined_2p_spectra = None


    for angle in sorted(
        fit_results
    ):

        fit = (
            fit_results[
                angle
            ]
        )


        safe_angle = (
            f"{angle:g}"
            .replace(
                ".",
                "p",
            )
        )


        if (
            combined_2p_spectra
            is None
        ):

            combined_2p_spectra = (
                pd.DataFrame(
                    {
                        "wavelength_nm": (
                            fit[
                                "dense_x"
                            ]
                        )
                    }
                )
            )


        combined_2p_spectra[
            f"WP{safe_angle}deg"
        ] = (
            fit[
                "difference_peak_2"
            ]
        )


    combined_2p_file = (
        SPECTRA_OUTPUT_FOLDER
        / (
            "ROI2_minus_ROI1_"
            "ALL_2P_FITTED_SPECTRA.csv"
        )
    )


    combined_2p_spectra.to_csv(
        combined_2p_file,
        index=False,
    )


    print(
        "\nAll differential 2p fitted spectra saved:"
    )

    print(
        combined_2p_file
    )


    # ========================================================
    # SAVE ALL MEASURED DIFFERENCE SPECTRA TO ONE CSV
    # ========================================================
    #
    # THESE ARE THE COLORED POINTS IN YOUR COMBINED FIGURE.
    #
    # ========================================================

    combined_measured_spectra = None


    for angle in sorted(
        fit_results
    ):

        fit = (
            fit_results[
                angle
            ]
        )


        safe_angle = (
            f"{angle:g}"
            .replace(
                ".",
                "p",
            )
        )


        current_dataframe = pd.DataFrame(
            {
                "wavelength_nm": (
                    fit[
                        "x"
                    ]
                ),

                f"WP{safe_angle}deg": (
                    fit[
                        "raw_difference"
                    ]
                ),
            }
        )


        if (
            combined_measured_spectra
            is None
        ):

            combined_measured_spectra = (
                current_dataframe
            )

        else:

            combined_measured_spectra = (
                pd.merge(
                    combined_measured_spectra,
                    current_dataframe,
                    on="wavelength_nm",
                    how="outer",
                )
            )


    combined_measured_spectra = (
        combined_measured_spectra
        .sort_values(
            "wavelength_nm"
        )
        .reset_index(
            drop=True
        )
    )


    combined_measured_file = (
        SPECTRA_OUTPUT_FOLDER
        / (
            "ROI2_minus_ROI1_"
            "ALL_MEASURED_SPECTRA.csv"
        )
    )


    combined_measured_spectra.to_csv(
        combined_measured_file,
        index=False,
    )


    print(
        "\nAll measured spectra saved:"
    )

    print(
        combined_measured_file
    )


    # ========================================================
    # COMBINED DIFFERENTIAL FIT FIGURE
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT,
        )
    )


    for angle in sorted(
        fit_results
    ):

        fit = (
            fit_results[
                angle
            ]
        )


        line, = axis.plot(
            fit[
                "dense_x"
            ],
            fit[
                "difference_fit"
            ],
            linewidth=2,
            label=(
                f"{angle:g}°"
            ),
        )


        if SHOW_RAW_DIFFERENCE:

            axis.plot(
                fit[
                    "x"
                ],
                fit[
                    "raw_difference"
                ],
                ".",
                markersize=2,
                alpha=0.15,
                color=(
                    line.get_color()
                ),
            )


    axis.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )


    axis.axvspan(
        P2_INTEGRATION_MIN_NM,
        P2_INTEGRATION_MAX_NM,
        alpha=0.05,
    )


    axis.set_xlim(
        FIT_MIN_NM,
        FIT_MAX_NM,
    )


    axis.set_xlabel(
        "Wavelength (nm)",
        fontsize=14,
    )


    axis.set_ylabel(
        "ROI2 - ROI1 intensity (counts)",
        fontsize=14,
    )


    axis.set_title(
        (
            "ROI2 - ROI1 from joint Lorentzian fits\n"
            f"Collection WP "
            f"{ANGLE_MIN:g}°-"
            f"{ANGLE_MAX:g}°"
        ),
        fontsize=15,
    )


    axis.tick_params(
        axis="both",
        labelsize=12,
    )


    axis.legend(
        title="Collection WP",
        fontsize=9,
        title_fontsize=10,
        ncol=2,
    )


    figure.tight_layout()


    if SAVE_COMBINED_FIGURE:

        combined_figure_file = (
            OUTPUT_FOLDER
            / (
                "ROI2_minus_ROI1_"
                "joint_fits_all_angles.png"
            )
        )


        figure.savefig(
            combined_figure_file,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


    plt.show()


    # ========================================================
    # ROI1 AND ROI2 2p AREAS
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )


    axis.plot(
        results[
            "angle_deg"
        ],
        results[
            "roi1_2p_area_window_counts_nm"
        ],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="ROI1",
    )


    axis.plot(
        results[
            "angle_deg"
        ],
        results[
            "roi2_2p_area_window_counts_nm"
        ],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="ROI2",
    )


    axis.set_xlabel(
        "Collection waveplate angle (degrees)",
        fontsize=14,
    )


    axis.set_ylabel(
        r"Integrated 2p intensity (counts$\cdot$nm)",
        fontsize=14,
    )


    axis.set_title(
        (
            "Integrated 2p Lorentzian area\n"
            f"{P2_INTEGRATION_MIN_NM:g}-"
            f"{P2_INTEGRATION_MAX_NM:g} nm"
        ),
        fontsize=14,
    )


    axis.tick_params(
        axis="both",
        labelsize=12,
    )


    axis.legend(
        fontsize=11,
    )


    figure.tight_layout()


    if SAVE_ROI_2P_AREA_PLOT:

        figure.savefig(
            OUTPUT_FOLDER
            / (
                "2p_integrated_area_"
                "ROI1_ROI2_vs_angle.png"
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


    plt.show()


    # ========================================================
    # DIFFERENTIAL 2p AREA
    # ========================================================

    figure, axis = plt.subplots(
        figsize=(
            8,
            5,
        )
    )


    axis.plot(
        results[
            "angle_deg"
        ],
        results[
            "2p_roi2_minus_roi1_area_window_counts_nm"
        ],
        marker="o",
        linewidth=1.8,
        markersize=6,
    )


    axis.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )


    axis.set_xlabel(
        "Collection waveplate angle (degrees)",
        fontsize=14,
    )


    axis.set_ylabel(
        (
            "Integrated 2p ROI2 - ROI1 "
            r"(counts$\cdot$nm)"
        ),
        fontsize=14,
    )


    axis.set_title(
        (
            "2p differential integrated intensity\n"
            f"{P2_INTEGRATION_MIN_NM:g}-"
            f"{P2_INTEGRATION_MAX_NM:g} nm"
        ),
        fontsize=14,
    )


    axis.tick_params(
        axis="both",
        labelsize=12,
    )


    figure.tight_layout()


    if SAVE_2P_AREA_PLOT:

        figure.savefig(
            OUTPUT_FOLDER
            / (
                "2p_ROI2_minus_ROI1_"
                "integrated_area_vs_angle.png"
            ),
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )


    plt.show()


    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "2p INTEGRATED AREA RESULTS"
    )

    print(
        "=" * 78
    )


    print(
        results[
            [
                "angle_deg",

                "2p_center_nm",
                "2p_fwhm_nm",

                "roi1_2p_area_window_counts_nm",
                "roi2_2p_area_window_counts_nm",

                "2p_roi2_minus_roi1_area_window_counts_nm",

                "difference_r_squared",
            ]
        ].to_string(
            index=False
        )
    )


    print(
        "\n"
        + "=" * 78
    )

    print(
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 78
    )


    print(
        "\nMost important output files:"
    )


    print(
        "\n1. All solid fitted curves:"
    )

    print(
        combined_fitted_file
    )


    print(
        "\n2. All measured ROI2 - ROI1 spectra:"
    )

    print(
        combined_measured_file
    )


    print(
        "\n3. All fitted 2p differential curves:"
    )

    print(
        combined_2p_file
    )


    print(
        "\n4. Fit parameters and integrated areas:"
    )

    print(
        results_output_file
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()