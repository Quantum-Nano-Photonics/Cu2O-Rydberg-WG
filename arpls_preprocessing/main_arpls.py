from __future__ import annotations

import numpy as np
import pandas as pd

import config_arpls as config

from arpls_baseline import estimate_baseline
from plotting_arpls import (
    plot_baseline_corrected_spectra,
    plot_baseline_diagnostic,
    plot_integrated_intensity,
    plot_roi2_minus_roi1,
)
from processing_arpls import (
    detect_isolated_spikes,
    integrate_spectrum,
)
from spectrum_io_arpls import (
    find_spectrum_files,
    read_spectrum,
)


def make_output_filename(
    base_name: str,
    extension: str = ".csv",
) -> str:
    output_label = str(
        getattr(
            config,
            "OUTPUT_LABEL",
            "",
        )
    ).strip()

    if output_label:
        return (
            f"{base_name}_"
            f"{output_label}"
            f"{extension}"
        )

    return (
        f"{base_name}"
        f"{extension}"
    )


def save_corrected_spectrum(
    wavelength: np.ndarray,
    raw_intensity: np.ndarray,
    despiked_intensity: np.ndarray,
    baseline: np.ndarray,
    corrected_intensity_original: np.ndarray,
    corrected_intensity_analysis: np.ndarray,
    collection_scale_factor: float,
    angle: float,
    roi: int,
    output_folder,
) -> None:
    """
    Save the complete ARPLS-processed spectrum for one collection angle
    and ROI.

    For ROI 1, the analysis spectrum is identical to the original
    ARPLS-corrected spectrum.

    For ROI 2, the analysis spectrum is the ARPLS-corrected spectrum
    multiplied by ROI2_COLLECTION_SCALE_FACTOR (0.728 by default).

    Both the unscaled and analysis spectra are retained in the CSV.
    """
    spectra_folder = output_folder / "corrected_spectra"
    spectra_folder.mkdir(parents=True, exist_ok=True)

    output_data = pd.DataFrame(
        {
            "wavelength_nm": wavelength,
            "raw_intensity_counts": raw_intensity,
            "despiked_intensity_counts": despiked_intensity,
            "arpls_baseline_counts": baseline,
            "corrected_intensity_original_counts": (
                corrected_intensity_original
            ),
            "collection_scale_factor": np.full(
                wavelength.shape,
                collection_scale_factor,
                dtype=float,
            ),
            "corrected_intensity_analysis_counts": (
                corrected_intensity_analysis
            ),
        }
    )

    safe_angle = (
        f"{angle:g}"
        .replace("-", "minus_")
        .replace(".", "p")
    )

    output_file = (
        spectra_folder
        / (
            f"ARPLS_corrected_"
            f"ROI{roi}_"
            f"ColPolWP{safe_angle}deg.csv"
        )
    )

    output_data.to_csv(output_file, index=False)

    print("\nSaved ARPLS-corrected spectrum:")
    print(f"  {output_file}")


def calculate_roi_differences(
    results: pd.DataFrame,
) -> pd.DataFrame | None:
    """Calculate ROI 2 - ROI 1 using ARPLS-corrected integrated areas."""
    if results.empty:
        return None

    pivot = results.pivot(
        index="colpol_angle_deg",
        columns="roi",
        values="integrated_intensity",
    )

    if (
        1 not in pivot.columns
        or 2 not in pivot.columns
    ):
        return None

    common = pivot[
        [1, 2]
    ].dropna()

    if common.empty:
        return None

    output = pd.DataFrame(
        {
            "colpol_angle_deg": (
                common.index.to_numpy(
                    dtype=float
                )
            ),
            "roi1_integrated_intensity": (
                common[1].to_numpy(
                    dtype=float
                )
            ),
            "roi2_integrated_intensity": (
                common[2].to_numpy(
                    dtype=float
                )
            ),
        }
    )

    output[
        "roi2_minus_roi1"
    ] = (
        output[
            "roi2_integrated_intensity"
        ]
        - output[
            "roi1_integrated_intensity"
        ]
    )

    return (
        output
        .sort_values(
            "colpol_angle_deg"
        )
        .reset_index(
            drop=True
        )
    )


def main() -> None:
    print("=" * 78)
    print("ARPLS BASELINE CORRECTION — NO BACKGROUND SUBTRACTION")
    print("=" * 78)

    print(
        "\nNo background file is loaded."
    )

    print(
        "No acquisition-time scaling is applied."
    )

    print(
        f"\nBaseline method: "
        f"{config.BASELINE_METHOD.upper()}"
    )

    print(
        f"Baseline window: "
        f"{config.BASELINE_MIN_NM:g}–"
        f"{config.BASELINE_MAX_NM:g} nm"
    )

    print(
        f"Integration window: "
        f"{config.INTEGRATION_MIN_NM:g}–"
        f"{config.INTEGRATION_MAX_NM:g} nm"
    )

    print(
        f"Baseline lambda: "
        f"{config.BASELINE_LAMBDA:g}"
    )

    print(
        "ROI2 collection correction: "
        + (
            f"ENABLED (ROI2 × {config.ROI2_COLLECTION_SCALE_FACTOR:g})"
            if getattr(
                config,
                "APPLY_ROI2_COLLECTION_CORRECTION",
                True,
            )
            else "DISABLED"
        )
    )

    # ========================================================
    # FIND MEASUREMENT FILES ONLY
    # ========================================================

    spectrum_files = find_spectrum_files(
        folder=config.SPECTRA_DIR,
        file_pattern=config.FILE_PATTERN,
        expol_to_analyze=config.EXPOL_TO_ANALYZE,
        required_text=config.MEASUREMENT_REQUIRED_TEXT,
    )

    selected_spectrum_files = {
        key: path
        for key, path in spectrum_files.items()
        if key[1] in config.ROIS_TO_ANALYZE
    }

    print(
        f"\nValid measurement spectra: "
        f"{len(selected_spectrum_files)}"
    )

    if not selected_spectrum_files:
        raise FileNotFoundError(
            "No measurement spectra matched the selected settings."
        )

    processed_spectra: dict[
        tuple[float, int],
        tuple[np.ndarray, np.ndarray],
    ] = {}

    result_rows: list[dict] = []
    failed_rows: list[dict] = []

    config.RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # PROCESS EACH RAW MEASUREMENT
    # ========================================================

    for key in sorted(
        selected_spectrum_files
    ):
        angle, roi = key

        spectrum_path = (
            selected_spectrum_files[
                key
            ]
        )

        print(
            "\n"
            + "-" * 78
        )

        print(
            f"Processing ColPol WP "
            f"{angle:g}°, ROI {roi}"
        )

        print(
            "-" * 78
        )

        print(
            f"Measurement: "
            f"{spectrum_path.name}"
        )

        try:
            # =================================================
            # READ RAW MEASUREMENT
            # =================================================

            (
                spectrum_wavelength,
                spectrum_intensity,
            ) = read_spectrum(
                file_path=spectrum_path,
                wavelength_column=(
                    config.WAVELENGTH_COLUMN
                ),
                intensity_column=(
                    config.INTENSITY_COLUMN
                ),
            )

            # =================================================
            # SELECT WIDE ARPLS BASELINE WINDOW
            # =================================================

            baseline_mask = (
                np.isfinite(
                    spectrum_wavelength
                )
                & np.isfinite(
                    spectrum_intensity
                )
                & (
                    spectrum_wavelength
                    >= config.BASELINE_MIN_NM
                )
                & (
                    spectrum_wavelength
                    <= config.BASELINE_MAX_NM
                )
            )

            wavelength = (
                spectrum_wavelength[
                    baseline_mask
                ]
            )

            intensity = (
                spectrum_intensity[
                    baseline_mask
                ]
            )

            if wavelength.size < 20:
                raise ValueError(
                    "Too few points exist inside "
                    "the selected baseline window."
                )

            order = np.argsort(
                wavelength
            )

            wavelength = wavelength[
                order
            ]

            intensity = intensity[
                order
            ]

            # =================================================
            # SPIKE DETECTION
            # =================================================

            (
                spike_mask,
                robust_sigma,
                local_median,
            ) = detect_isolated_spikes(
                intensity=intensity,
                window_points=(
                    config.SPIKE_WINDOW_POINTS
                ),
                sigma_threshold=(
                    config.SPIKE_SIGMA_THRESHOLD
                ),
            )

            despiked_intensity = (
                intensity.copy()
            )

            despiked_intensity[
                spike_mask
            ] = local_median[
                spike_mask
            ]

            # =================================================
            # ARPLS BASELINE
            # =================================================

            baseline = estimate_baseline(
                intensity=despiked_intensity,
                method="arpls",
                smoothness_lambda=(
                    config.BASELINE_LAMBDA
                ),
                als_asymmetry_p=(
                    config.ALS_ASYMMETRY_P
                ),
                als_iterations=(
                    config.ALS_ITERATIONS
                ),
                arpls_convergence_ratio=(
                    config.ARPLS_CONVERGENCE_RATIO
                ),
                arpls_max_iterations=(
                    config.ARPLS_MAX_ITERATIONS
                ),
            )

            # Original ARPLS-corrected experimental spectrum.
            peak_only_intensity_original = (
                despiked_intensity
                - baseline
            )

            # =================================================
            # ROI COLLECTION-EFFICIENCY CORRECTION
            # =================================================

            collection_scale_factor = 1.0

            if (
                roi == 2
                and getattr(
                    config,
                    "APPLY_ROI2_COLLECTION_CORRECTION",
                    True,
                )
            ):
                collection_scale_factor = float(
                    config.ROI2_COLLECTION_SCALE_FACTOR
                )

            peak_only_intensity_analysis = (
                peak_only_intensity_original
                * collection_scale_factor
            )

            # The combined plots use the calibrated spectrum.
            # ROI1 is unchanged; ROI2 is multiplied by 0.728.
            processed_spectra[key] = (
                wavelength,
                peak_only_intensity_analysis,
            )

            # =================================================
            # SAVE COMPLETE ARPLS-CORRECTED SPECTRUM
            # =================================================

            if getattr(
                config,
                "SAVE_CORRECTED_SPECTRA",
                True,
            ):
                save_corrected_spectrum(
                    wavelength=wavelength,
                    raw_intensity=intensity,
                    despiked_intensity=despiked_intensity,
                    baseline=baseline,
                    corrected_intensity_original=(
                        peak_only_intensity_original
                    ),
                    corrected_intensity_analysis=(
                        peak_only_intensity_analysis
                    ),
                    collection_scale_factor=(
                        collection_scale_factor
                    ),
                    angle=angle,
                    roi=roi,
                    output_folder=config.RESULTS_DIR,
                )

            # =================================================
            # NUMERICAL INTEGRATION
            # =================================================

            # Keep the original integral for traceability.
            integrated_intensity_original = integrate_spectrum(
                wavelength=wavelength,
                intensity=peak_only_intensity_original,
                wavelength_min=config.INTEGRATION_MIN_NM,
                wavelength_max=config.INTEGRATION_MAX_NM,
            )

            # This is the integral used in all downstream analysis.
            # For ROI2 it is based on the 0.728-scaled spectrum.
            integrated_intensity = integrate_spectrum(
                wavelength=wavelength,
                intensity=peak_only_intensity_analysis,
                wavelength_min=config.INTEGRATION_MIN_NM,
                wavelength_max=config.INTEGRATION_MAX_NM,
            )

            result_rows.append(
                {
                    "colpol_angle_deg": (
                        angle
                    ),
                    "roi": roi,
                    # Main quantity used for plots/differences.
                    # ROI2 is collection-efficiency corrected.
                    "integrated_intensity": (
                        integrated_intensity
                    ),
                    "integrated_intensity_original": (
                        integrated_intensity_original
                    ),
                    "collection_scale_factor": (
                        collection_scale_factor
                    ),
                    "roi2_collection_corrected": (
                        bool(roi == 2 and collection_scale_factor != 1.0)
                    ),
                    "baseline_method": (
                        "arpls"
                    ),
                    "baseline_lambda": (
                        config.BASELINE_LAMBDA
                    ),
                    "baseline_min_nm": (
                        config.BASELINE_MIN_NM
                    ),
                    "baseline_max_nm": (
                        config.BASELINE_MAX_NM
                    ),
                    "integration_min_nm": (
                        config.INTEGRATION_MIN_NM
                    ),
                    "integration_max_nm": (
                        config.INTEGRATION_MAX_NM
                    ),
                    "initial_spike_count": (
                        int(
                            np.count_nonzero(
                                spike_mask
                            )
                        )
                    ),
                    "robust_noise_sigma": (
                        float(
                            robust_sigma
                        )
                    ),
                    "background_subtracted": (
                        False
                    ),
                    "spectrum_scale_factor": (
                        1.0
                    ),
                    "spectrum_file": (
                        spectrum_path.name
                    ),
                }
            )

            # =================================================
            # INDIVIDUAL ARPLS DIAGNOSTIC
            # =================================================

            if (
                config.SHOW_BASELINE_DIAGNOSTICS
                or config.SAVE_BASELINE_DIAGNOSTICS
            ):
                plot_baseline_diagnostic(
                    wavelength=wavelength,
                    raw_intensity=intensity,
                    despiked_intensity=(
                        despiked_intensity
                    ),
                    baseline=baseline,
                    peak_only_intensity=(
                        peak_only_intensity_original
                    ),
                    angle=angle,
                    roi=roi,
                    integration_min=(
                        config.INTEGRATION_MIN_NM
                    ),
                    integration_max=(
                        config.INTEGRATION_MAX_NM
                    ),
                    integrated_area=(
                        integrated_intensity_original
                    ),
                    method_name="arpls",
                    save_figure=(
                        config.SAVE_BASELINE_DIAGNOSTICS
                    ),
                    show_figure=(
                        config.SHOW_BASELINE_DIAGNOSTICS
                    ),
                    output_folder=(
                        config.RESULTS_DIR
                    ),
                )

            print(
                f"Original integrated area: "
                f"{integrated_intensity_original:.8g} "
                f"counts·nm"
            )

            print(
                f"Analysis integrated area: "
                f"{integrated_intensity:.8g} "
                f"counts·nm "
                f"(scale factor = {collection_scale_factor:g})"
            )

            print(
                f"Detected isolated spikes: "
                f"{np.count_nonzero(spike_mask)}"
            )

        except Exception as error:
            processed_spectra.pop(
                key,
                None,
            )

            print(
                f"[failed] "
                f"ColPol WP {angle:g}°, "
                f"ROI {roi}"
            )

            print(
                f"Reason: {error}"
            )

            failed_rows.append(
                {
                    "colpol_angle_deg": (
                        angle
                    ),
                    "roi": roi,
                    "spectrum_file": (
                        spectrum_path.name
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": (
                        str(error)
                    ),
                }
            )

    if not result_rows:
        raise RuntimeError(
            "All ARPLS spectra failed."
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results = (
        pd.DataFrame(
            result_rows
        )
        .sort_values(
            [
                "roi",
                "colpol_angle_deg",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    difference_results = (
        calculate_roi_differences(
            results
        )
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "FINAL ARPLS NUMERICAL INTEGRATION RESULTS"
    )

    print(
        "=" * 78
    )

    print(
        results[
            [
                "colpol_angle_deg",
                "roi",
                "integrated_intensity_original",
                "collection_scale_factor",
                "integrated_intensity",
                "initial_spike_count",
            ]
        ].to_string(
            index=False
        )
    )

    if difference_results is not None:
        print(
            "\nROI 2 − ROI 1"
        )

        print(
            difference_results.to_string(
                index=False
            )
        )

    # ========================================================
    # SAVE TABLES
    # ========================================================

    if config.SAVE_RESULTS_TABLE:
        config.RESULTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        results.to_csv(
            config.RESULTS_DIR
            / make_output_filename(
                "arpls_integrated_results"
            ),
            index=False,
        )

        if (
            difference_results
            is not None
        ):
            difference_results.to_csv(
                config.RESULTS_DIR
                / make_output_filename(
                    "arpls_ROI2_minus_ROI1"
                ),
                index=False,
            )

        if failed_rows:
            pd.DataFrame(
                failed_rows
            ).to_csv(
                config.RESULTS_DIR
                / make_output_filename(
                    "arpls_failed_spectra"
                ),
                index=False,
            )

    # ========================================================
    # PLOTS
    # ========================================================

    if config.SHOW_BASELINE_CORRECTED_SPECTRA:
        plot_baseline_corrected_spectra(
            processed_spectra=(
                processed_spectra
            ),
            integration_min=(
                config.INTEGRATION_MIN_NM
            ),
            integration_max=(
                config.INTEGRATION_MAX_NM
            ),
            method_name="arpls",
            figure_width=(
                config.FIGURE_WIDTH
            ),
            figure_height=(
                config.FIGURE_HEIGHT
            ),
            save_figure=(
                config.SAVE_FIGURES
            ),
            output_folder=(
                config.RESULTS_DIR
            ),
        )

    if config.SHOW_INTEGRAL_PLOT:
        plot_integrated_intensity(
            results=results,
            method_name="arpls",
            figure_width=(
                config.FIGURE_WIDTH
            ),
            figure_height=(
                config.FIGURE_HEIGHT
            ),
            save_figure=(
                config.SAVE_FIGURES
            ),
            output_folder=(
                config.RESULTS_DIR
            ),
        )

    if (
        config.SHOW_ROI_DIFFERENCE_PLOT
        and difference_results
        is not None
    ):
        plot_roi2_minus_roi1(
            difference_results=(
                difference_results
            ),
            method_name="arpls",
            figure_width=(
                config.FIGURE_WIDTH
            ),
            figure_height=(
                config.FIGURE_HEIGHT
            ),
            save_figure=(
                config.SAVE_FIGURES
            ),
            output_folder=(
                config.RESULTS_DIR
            ),
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "ARPLS ANALYSIS COMPLETED"
    )

    print(
        "=" * 78
    )

    print(
        f"Successful spectra: "
        f"{len(results)}"
    )

    print(
        f"Failed spectra: "
        f"{len(failed_rows)}"
    )

    if getattr(
        config,
        "SAVE_CORRECTED_SPECTRA",
        True,
    ):
        print(
            "\nCorrected spectra saved in:"
        )
        print(
            config.RESULTS_DIR
            / "corrected_spectra"
        )


if __name__ == "__main__":
    main()
