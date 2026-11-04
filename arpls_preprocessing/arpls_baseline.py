from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def _second_difference_matrix(number_of_points: int) -> sparse.csc_matrix:
    """Return the second-difference matrix used for baseline smoothness."""
    if number_of_points < 3:
        raise ValueError("At least three points are required for baseline fitting.")

    return sparse.diags(
        diagonals=(
            np.ones(number_of_points - 2),
            -2.0 * np.ones(number_of_points - 2),
            np.ones(number_of_points - 2),
        ),
        offsets=(0, 1, 2),
        shape=(number_of_points - 2, number_of_points),
        format="csc",
    )


def asymmetric_least_squares(
    intensity: np.ndarray,
    smoothness_lambda: float = 1e5,
    asymmetry_p: float = 0.001,
    iterations: int = 20,
) -> np.ndarray:
    """
    Estimate a smooth baseline with asymmetric least squares (ALS/AsLS).
    """
    y = np.asarray(intensity, dtype=float)

    if y.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")
    if y.size < 3:
        raise ValueError("At least three points are required.")
    if smoothness_lambda <= 0:
        raise ValueError("smoothness_lambda must be greater than zero.")
    if not 0 < asymmetry_p < 1:
        raise ValueError("asymmetry_p must be between 0 and 1.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    difference_matrix = _second_difference_matrix(y.size)
    penalty = smoothness_lambda * (difference_matrix.T @ difference_matrix)

    weights = np.ones(y.size, dtype=float)

    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, y.size, y.size)
        baseline = spsolve(
            (weight_matrix + penalty).tocsc(),
            weights * y,
        )

        weights = np.where(
            y > baseline,
            asymmetry_p,
            1.0 - asymmetry_p,
        )

    return np.asarray(baseline, dtype=float)


def arpls_baseline(
    intensity: np.ndarray,
    smoothness_lambda: float = 1e5,
    convergence_ratio: float = 1e-6,
    max_iterations: int = 100,
) -> np.ndarray:
    """
    Estimate a baseline with adaptive iteratively reweighted penalized
    least squares (arPLS).
    """
    y = np.asarray(intensity, dtype=float)

    if y.ndim != 1:
        raise ValueError("intensity must be one-dimensional.")
    if y.size < 3:
        raise ValueError("At least three points are required.")
    if smoothness_lambda <= 0:
        raise ValueError("smoothness_lambda must be greater than zero.")
    if convergence_ratio <= 0:
        raise ValueError("convergence_ratio must be greater than zero.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1.")

    difference_matrix = _second_difference_matrix(y.size)
    penalty = smoothness_lambda * (difference_matrix.T @ difference_matrix)

    weights = np.ones(y.size, dtype=float)
    baseline = np.zeros_like(y)

    for _ in range(max_iterations):
        weight_matrix = sparse.spdiags(weights, 0, y.size, y.size)
        baseline = spsolve(
            (weight_matrix + penalty).tocsc(),
            weights * y,
        )

        residual = y - baseline
        negative_residuals = residual[residual < 0]

        if negative_residuals.size < 2:
            break

        mean_negative = float(np.mean(negative_residuals))
        std_negative = float(np.std(negative_residuals))

        if not np.isfinite(std_negative) or std_negative <= 0:
            break

        exponent = (
            2.0
            * (
                residual
                - (2.0 * std_negative - mean_negative)
            )
            / std_negative
        )

        exponent = np.clip(exponent, -60.0, 60.0)
        new_weights = 1.0 / (1.0 + np.exp(exponent))

        denominator = np.linalg.norm(weights)
        if denominator == 0:
            weights = new_weights
            break

        relative_change = (
            np.linalg.norm(weights - new_weights)
            / denominator
        )

        weights = new_weights

        if relative_change < convergence_ratio:
            break

    return np.asarray(baseline, dtype=float)


def estimate_baseline(
    intensity: np.ndarray,
    method: str = "arpls",
    *,
    smoothness_lambda: float = 1e5,
    als_asymmetry_p: float = 0.001,
    als_iterations: int = 20,
    arpls_convergence_ratio: float = 1e-6,
    arpls_max_iterations: int = 100,
) -> np.ndarray:
    """Dispatch to ALS or arPLS using one common interface."""
    method_normalized = method.strip().lower()

    if method_normalized == "arpls":
        return arpls_baseline(
            intensity=intensity,
            smoothness_lambda=smoothness_lambda,
            convergence_ratio=arpls_convergence_ratio,
            max_iterations=arpls_max_iterations,
        )

    if method_normalized in {"als", "asls"}:
        return asymmetric_least_squares(
            intensity=intensity,
            smoothness_lambda=smoothness_lambda,
            asymmetry_p=als_asymmetry_p,
            iterations=als_iterations,
        )

    raise ValueError(
        "BASELINE_METHOD must be either 'arpls' or 'als'. "
        f"Received: {method!r}"
    )
