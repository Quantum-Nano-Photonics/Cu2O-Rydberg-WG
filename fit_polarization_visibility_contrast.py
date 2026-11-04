#!/usr/bin/env python3
"""Fit the 2p polarization visibility from efficiency-corrected H and V areas.

For each collection angle theta:
    P(theta) = [H(theta) - V(theta)] / [H(theta) + V(theta)]

Fit the measured contrast with:
    P(theta) = P0 + a*cos(2*theta) + b*sin(2*theta)
             = P0 + A*cos(2*theta + phase)
    A = sqrt(a**2 + b**2)

For balanced, correctly calibrated orthogonal polarization channels (P0=0),
A is the conventional polarization visibility (Imax-Imin)/(Imax+Imin).
Nonzero fitted P0 must be investigated before making that identification
exactly; this script prints it with an uncertainty.

The CSVs are assumed to contain ALREADY efficiency-corrected fitted H/V
integrated intensities; the ROI2 correction factor is NOT applied twice.

Requirements: pip install numpy pandas matplotlib
Usage: python fit_polarization_visibility_contrast.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results" / "yellow"
INPUTS = {
    "Island (I)": RESULTS_ROOT / "results_lorentzian_fit_ROI2_minus_ROI1_cI_eI" / "ROI2_minus_ROI1_joint_lorentzian_results.csv",
    "Grating (G1)": RESULTS_ROOT / "results_lorentzian_fit_ROI2_minus_ROI1_cG1_eI" / "ROI2_minus_ROI1_joint_lorentzian_results.csv",
}
COLORS = {"Island (I)": "#d62728", "Grating (G1)": "#2ca02c"}
H_COL = "roi2_2p_area_window_counts_nm"   # ROI2 = H
V_COL = "roi1_2p_area_window_counts_nm"   # ROI1 = V
HWP_ZERO_DEG = 154.5

# None = use all ten angles (the last corresponds to 405° collection angle).
# 360.0 = use only the nine angles shown in Fig. 3(b).
MAX_COLLECTION_ANGLE = 360.0
OUT = HERE / "polarization_contrast_fit_results"


def model(theta_deg, p0, a, b):
    rad = np.deg2rad(2.0 * np.asarray(theta_deg, dtype=float))
    return p0 + a * np.cos(rad) + b * np.sin(rad)


def fit_configuration(label, csv_path):
    df = pd.read_csv(csv_path)
    required = ["angle_deg", H_COL, V_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing columns {missing}")

    df = df.dropna(subset=required).sort_values("angle_deg").copy()
    df["collection_angle_deg"] = 2.0 * (df["angle_deg"] - HWP_ZERO_DEG)
    if MAX_COLLECTION_ANGLE is not None:
        df = df.loc[df["collection_angle_deg"] <= MAX_COLLECTION_ANGLE].copy()
    if len(df) <= 3:
        raise ValueError(f"{label}: need at least 4 valid angle measurements")

    theta = df["collection_angle_deg"].to_numpy(float)
    H = df[H_COL].to_numpy(float)
    V = df[V_COL].to_numpy(float)
    if np.any(H + V <= 0):
        raise ValueError(f"{label}: some H+V values are zero or negative")

    P = (H - V) / (H + V)
    alpha = np.deg2rad(2.0 * theta)
    X = np.column_stack((np.ones(len(theta)), np.cos(alpha), np.sin(alpha)))

    # Linear least squares: jointly fit offset and sine/cosine coefficients.
    pars, _, rank, _ = np.linalg.lstsq(X, P, rcond=None)
    if rank != 3:
        raise ValueError(f"{label}: the angles cannot determine all 3 parameters")
    p0, a, b = map(float, pars)
    Pfit = X @ pars
    residuals = P - Pfit
    dof = len(P) - 3
    s2 = float(np.sum(residuals**2) / dof)
    covariance = s2 * np.linalg.inv(X.T @ X)

    A = float(np.hypot(a, b))
    grad = np.array([0.0, a / A, b / A]) if A > 0 else np.zeros(3)
    A_se = float(np.sqrt(max(0, grad @ covariance @ grad))) if A > 0 else float("nan")
    p0_se = float(np.sqrt(covariance[0, 0]))
    phase_deg = float(np.degrees(np.arctan2(-b, a)))
    sst = float(np.sum((P - np.mean(P)) ** 2))
    r2 = 1.0 - float(np.sum(residuals**2)) / sst if sst > 0 else float("nan")

    table = pd.DataFrame({
        "configuration": label,
        "hwp_angle_deg": df["angle_deg"].to_numpy(float),
        "collection_angle_deg": theta,
        "H_integrated_2p_counts_nm": H,
        "V_integrated_2p_counts_nm": V,
        "H_minus_V": H - V,
        "H_plus_V": H + V,
        "contrast_measured": P,
        "contrast_fitted": Pfit,
        "residual": residuals,
    })
    result = {
        "configuration": label,
        "n_points": len(P),
        "contrast_amplitude_A": A,
        "amplitude_standard_error_1sigma": A_se,
        "offset_P0": p0,
        "offset_standard_error_1sigma": p0_se,
        "phase_deg": phase_deg,
        "R_squared": r2,
    }
    return table, result, pars


def main():
    OUT.mkdir(exist_ok=True)
    measurements, results, params = [], [], {}

    for label, path in INPUTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Cannot find: {path}\nPut this script next to both CSV files.")
        data, result, par = fit_configuration(label, path)
        measurements.append(data)
        results.append(result)
        params[label] = par
        print(f"\n{label}: {result['n_points']} measured angles")
        print(f"  Fitted polarization contrast amplitude A = "
              f"{result['contrast_amplitude_A']:.4f} ± "
              f"{result['amplitude_standard_error_1sigma']:.4f} (1σ)")
        print(f"  Fitted offset P0 = {result['offset_P0']:+.4f} ± "
              f"{result['offset_standard_error_1sigma']:.4f} (1σ)")
        print(f"  R² = {result['R_squared']:.4f}")

    all_data = pd.concat(measurements, ignore_index=True)
    all_data.to_csv(OUT / "measured_and_fitted_contrast.csv", index=False)
    pd.DataFrame(results).to_csv(OUT / "visibility_fit_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 5.1), layout="constrained")
    angle_max = max(360.0, float(all_data["collection_angle_deg"].max()))
    fine = np.linspace(0, angle_max, 900)
    for result, data in zip(results, measurements):
        label = result["configuration"]
        color = COLORS[label]
        ax.scatter(data["collection_angle_deg"], data["contrast_measured"],
                   s=50, color=color, zorder=4, label=f"{label}: data")
        ax.plot(fine, model(fine, *params[label]), color=color, lw=2.2,
                label=(f"{label}: fit A = {result['contrast_amplitude_A']:.3f} "
                       f"± {result['amplitude_standard_error_1sigma']:.3f}"))
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set(xlabel="Collection polarization angle (degrees)",
           ylabel="Polarization contrast (H − V) / (H + V)",
           title="2p polarization contrast: measured and cosine fits",
           xlim=(0, angle_max))
    ax.set_xticks(np.arange(0, angle_max + 1, 45))
    ax.grid(alpha=0.16)
    ax.legend(fontsize=9, frameon=False)
    fig.savefig(OUT / "polarization_contrast_cosine_fits.png", dpi=250)
    fig.savefig(OUT / "polarization_contrast_cosine_fits.pdf")
    plt.close(fig)
    print("\nSaved figure and CSV results in:", OUT.resolve())
    print("Warning: A equals the conventional intensity visibility only if")
    print("the corrected orthogonal H/V signals satisfy the complementary model;")
    print("check fitted offsets and efficiency calibration before publication.")


if __name__ == "__main__":
    main()
