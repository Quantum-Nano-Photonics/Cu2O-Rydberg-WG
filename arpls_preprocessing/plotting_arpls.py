from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_baseline_diagnostic(
    wavelength: np.ndarray,
    raw_intensity: np.ndarray,
    despiked_intensity: np.ndarray,
    baseline: np.ndarray,
    peak_only_intensity: np.ndarray,
    angle: float,
    roi: int,
    integration_min: float,
    integration_max: float,
    integrated_area: float,
    method_name: str,
    save_figure: bool = False,
    show_figure: bool = False,
    output_folder: Path | None = None,
) -> None:
    """Plot one ARPLS baseline diagnostic figure."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    top_axis, bottom_axis = axes

    top_axis.plot(
        wavelength,
        raw_intensity,
        linewidth=1.0,
        label="Raw spectrum",
    )

    top_axis.plot(
        wavelength,
        despiked_intensity,
        linewidth=1.2,
        label="Despiked spectrum",
    )

    top_axis.plot(
        wavelength,
        baseline,
        linestyle="--",
        linewidth=2.0,
        label=f"{method_name.upper()} baseline",
    )

    top_axis.set_ylabel("Intensity (counts)")

    top_axis.set_title(
        (
            f"ARPLS baseline diagnostic — ROI {roi}, "
            f"ColPol WP {angle:g}°\n"
            f"Integrated area = {integrated_area:.6g} counts·nm"
        )
    )

    top_axis.grid(alpha=0.25)
    top_axis.legend(fontsize=9)

    bottom_axis.plot(
        wavelength,
        peak_only_intensity,
        linewidth=1.4,
        label="ARPLS-corrected spectrum",
    )

    bottom_axis.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    bottom_axis.axvspan(
        integration_min,
        integration_max,
        alpha=0.12,
        label=(
            f"Integration range: "
            f"{integration_min:g}–{integration_max:g} nm"
        ),
    )

    integration_mask = (
        (wavelength >= integration_min)
        & (wavelength <= integration_max)
    )

    if np.any(integration_mask):
        bottom_axis.fill_between(
            wavelength[integration_mask],
            0,
            peak_only_intensity[integration_mask],
            alpha=0.12,
        )

    bottom_axis.set_xlabel("Wavelength (nm)")
    bottom_axis.set_ylabel("Baseline-corrected intensity (counts)")
    bottom_axis.grid(alpha=0.25)
    bottom_axis.legend(fontsize=9)

    figure.tight_layout()

    if save_figure:
        if output_folder is None:
            raise ValueError("output_folder is required when saving figures.")

        diagnostic_folder = output_folder / "baseline_diagnostics"
        diagnostic_folder.mkdir(parents=True, exist_ok=True)

        safe_angle = (
            f"{angle:g}"
            .replace("-", "minus_")
            .replace(".", "p")
        )

        output_file = diagnostic_folder / (
            f"arpls_baseline_ROI{roi}_WP{safe_angle}deg.png"
        )

        figure.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    if show_figure:
        plt.show()
    else:
        plt.close(figure)


def plot_baseline_corrected_spectra(
    processed_spectra: dict[
        tuple[float, int],
        tuple[np.ndarray, np.ndarray],
    ],
    integration_min: float,
    integration_max: float,
    method_name: str,
    figure_width: float = 11,
    figure_height: float = 7,
    save_figure: bool = False,
    output_folder: Path | None = None,
) -> None:
    """Plot all ARPLS-corrected spectra, one figure per ROI."""
    if not processed_spectra:
        return

    roi_numbers = sorted(
        {
            roi
            for _, roi in processed_spectra
        }
    )

    for roi in roi_numbers:
        figure, axis = plt.subplots(
            figsize=(figure_width, figure_height)
        )

        roi_keys = sorted(
            key
            for key in processed_spectra
            if key[1] == roi
        )

        for angle, current_roi in roi_keys:
            wavelength, intensity = processed_spectra[
                (angle, current_roi)
            ]

            axis.plot(
                wavelength,
                intensity,
                linewidth=1.4,
                label=f"ColPol WP {angle:g}°",
            )

        axis.axvspan(
            integration_min,
            integration_max,
            alpha=0.12,
            label=(
                f"Integration: {integration_min:g}–"
                f"{integration_max:g} nm"
            ),
        )

        axis.axhline(
            0,
            linestyle="--",
            linewidth=1.0,
        )

        axis.set_xlabel(
            "Wavelength (nm)",
            fontsize=14,
        )

        axis.set_ylabel(
            "ARPLS-corrected intensity (counts)",
            fontsize=14,
        )

        axis.set_title(
            f"ARPLS-corrected spectra — ROI {roi}",
            fontsize=16,
        )

        axis.grid(alpha=0.25)
        axis.legend(fontsize=9, ncol=2)

        figure.tight_layout()

        if save_figure:
            if output_folder is None:
                raise ValueError(
                    "output_folder is required when saving figures."
                )

            output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_file = (
                output_folder
                / f"arpls_corrected_spectra_ROI{roi}.png"
            )

            figure.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
            )

    plt.show()


def plot_integrated_intensity(
    results: pd.DataFrame,
    method_name: str,
    figure_width: float = 10,
    figure_height: float = 6,
    save_figure: bool = False,
    output_folder: Path | None = None,
) -> None:
    """Plot ARPLS integrated intensity versus ColPol angle."""
    if results.empty:
        return

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    for roi, roi_results in results.groupby("roi"):
        roi_results = roi_results.sort_values(
            "colpol_angle_deg"
        )

        axis.plot(
            roi_results["colpol_angle_deg"],
            roi_results["integrated_intensity"],
            marker="o",
            linewidth=1.8,
            markersize=6,
            label=f"ROI {roi}",
        )

    axis.set_xlabel(
        "Collection waveplate angle (degrees)",
        fontsize=14,
    )

    axis.set_ylabel(
        "Integrated peak intensity (counts·nm)",
        fontsize=14,
    )

    axis.set_title(
        "Numerical peak area after ARPLS baseline correction",
        fontsize=15,
    )

    axis.grid(alpha=0.25)
    axis.legend(fontsize=11)

    figure.tight_layout()

    if save_figure:
        if output_folder is None:
            raise ValueError(
                "output_folder is required when saving figures."
            )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_folder / "arpls_integrated_intensity_vs_colpol.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()


def plot_roi2_minus_roi1(
    difference_results: pd.DataFrame,
    method_name: str,
    figure_width: float = 10,
    figure_height: float = 6,
    save_figure: bool = False,
    output_folder: Path | None = None,
) -> None:
    """Plot ROI 2 minus ROI 1 ARPLS integrated areas."""
    if difference_results.empty:
        return

    difference_results = difference_results.sort_values(
        "colpol_angle_deg"
    )

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height)
    )

    axis.plot(
        difference_results["colpol_angle_deg"],
        difference_results["roi2_minus_roi1"],
        marker="o",
        linewidth=1.8,
        markersize=6,
    )

    axis.axhline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    axis.set_xlabel(
        "Collection waveplate angle (degrees)",
        fontsize=14,
    )

    axis.set_ylabel(
        "ROI 2 − ROI 1 area (counts·nm)",
        fontsize=14,
    )

    axis.set_title(
        "ROI 2 − ROI 1 after ARPLS baseline correction",
        fontsize=15,
    )

    axis.grid(alpha=0.25)

    figure.tight_layout()

    if save_figure:
        if output_folder is None:
            raise ValueError(
                "output_folder is required when saving figures."
            )

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_folder / "arpls_ROI2_minus_ROI1.png",
            dpi=300,
            bbox_inches="tight",
        )

    plt.show()
