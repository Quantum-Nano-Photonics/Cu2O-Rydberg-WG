from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def integrate_spectrum(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
) -> float:
    """Numerically integrate a spectrum over a selected wavelength interval."""
    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if wavelength.shape != intensity.shape:
        raise ValueError("wavelength and intensity must have the same shape.")

    if wavelength_min >= wavelength_max:
        raise ValueError("wavelength_min must be smaller than wavelength_max.")

    mask = (
        np.isfinite(wavelength)
        & np.isfinite(intensity)
        & (wavelength >= wavelength_min)
        & (wavelength <= wavelength_max)
    )

    x = wavelength[mask]
    y = intensity[mask]

    if x.size < 2:
        raise ValueError(
            f"Not enough valid points between {wavelength_min:g} and "
            f"{wavelength_max:g} nm."
        )

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    return float(np.trapezoid(y, x=x))


def detect_isolated_spikes(
    intensity: np.ndarray,
    window_points: int = 9,
    sigma_threshold: float = 8.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Detect isolated narrow spikes with a rolling median and robust MAD noise.

    The rolling median is used only to identify and replace points classified
    as isolated spikes. It does not smooth the remainder of the spectrum.
    """
    intensity = np.asarray(intensity, dtype=float)

    if intensity.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")

    if intensity.size < 5:
        raise ValueError("At least five intensity points are required.")

    if window_points < 3:
        raise ValueError("window_points must be at least 3.")

    if sigma_threshold <= 0:
        raise ValueError("sigma_threshold must be greater than zero.")

    if window_points % 2 == 0:
        window_points += 1

    if window_points > intensity.size:
        window_points = intensity.size
        if window_points % 2 == 0:
            window_points -= 1

    local_median = median_filter(
        intensity,
        size=window_points,
        mode="nearest",
    )

    residual = intensity - local_median
    residual_center = float(np.median(residual))

    mad = float(
        np.median(
            np.abs(residual - residual_center)
        )
    )

    if not np.isfinite(mad) or mad <= 0:
        robust_sigma = float(np.std(residual))

        if not np.isfinite(robust_sigma) or robust_sigma <= 0:
            robust_sigma = 1.0
    else:
        robust_sigma = 1.4826 * mad

    spike_mask = (
        np.abs(residual - residual_center)
        > sigma_threshold * robust_sigma
    )

    return (
        spike_mask,
        float(robust_sigma),
        local_median,
    )
