# Cu2O Rydberg Exciton Spectroscopy — analysis scripts

Python analysis workflows accompanying a study of polarization-resolved Cu2O exciton photoluminescence collected from island and grating/waveguide geometries. **This is a code-only repository scaffold**: experimental raw data and final figure datasets are not included.

## Analysis pipeline

1. **`arpls_preprocessing/`** — read raw spectral CSVs, detect isolated spikes, fit an adaptive iteratively reweighted penalized least-squares (arPLS) background, subtract it, apply the relative ROI2 collection-efficiency factor, save corrected spectra, and perform a diagnostic trapezoidal integration of the baseline-corrected spectra.
2. **`joint_lorentzian_fit.py`** — load the corrected ROI1/ROI2 spectra, interpolate onto a common wavelength grid, fit two Lorentzian peaks jointly with shared centers/FWHMs and ROI-specific amplitudes/offsets, calculate the fitted 2p area and ROI differences, and save figures/results. The integrated area of the second fitted Lorentzian, corresponding to the 2p transition, is used for the subsequent polarization contrast and visibility analysis.
3. **`fit_polarization_visibility_contrast.py`** — load the 2p fitted areas from the island and G1 geometries; calculate normalized H/V contrast and its sinusoidal modulation amplitude, offset and residual-based standard errors.

## Installation

Requires Python 3.10+ (NumPy >= 2.0 for `np.trapezoid`). From the repository root:

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell):

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Step 1 — preprocess raw spectra

Place the input CSV spectra in `data/raw/`, or set environment variable `CU2O_RAW_DATA_DIR` to a folder containing them. Input filenames must match the selectors in `arpls_preprocessing/config_arpls.py`. The supplied configuration expects `ExPol = 0°`, `ROI1/ROI2`, `_eI_` excitation, `_120s_` spectra, and **`COLLECTION = "cI"`**. Input CSVs ordinarily have `Wavelength` and `Intensity` columns; see `spectrum_io_arpls.py` for accepted fallbacks and filename parsing.

Run from the repository root:

```bash
python arpls_preprocessing/main_arpls.py
```

Then edit `COLLECTION = "cG1"` in `arpls_preprocessing/config_arpls.py` and repeat to prepare the G1 collection configuration. You can switch back to `cI` without deleting the G1 output.

Outputs appear in:

```text
results/yellow/results_arpls_cI_eI_no_background/corrected_spectra/
results/yellow/results_arpls_cG1_eI_no_background/corrected_spectra/
```

The preprocessing uses **arPLS rather than a separately measured background file**. After baseline subtraction, it multiplies ROI2 by the experimentally supplied relative-efficiency factor `0.728` and leaves ROI1 unchanged. Both original and corrected intensities are saved; this factor must not be applied again downstream. 

## Step 2 — jointly fit ROI1/ROI2 spectra

From the repository root:

```bash
python joint_lorentzian_fit.py --collection island --no-show
python joint_lorentzian_fit.py --collection G1 --no-show
```

The default `--paper-root` is the repository root. To fit files stored elsewhere:

```bash
python joint_lorentzian_fit.py --collection island --input-dir "path/to/corrected_spectra" --output-dir "path/to/results" --no-show
```

Outputs include `ROI2_minus_ROI1_joint_lorentzian_results.csv` under each `results/yellow/results_lorentzian_fit_ROI2_minus_ROI1_...` directory.


## Step 3 — fit polarization contrast

After completing *both* island and G1 Lorentzian analyses:

```bash
python fit_polarization_visibility_contrast.py
```

Outputs are saved in `polarization_contrast_fit_results/`. This script assumes **ROI2 = H and ROI1 = V**, uses the already corrected integrated intensities, and maps waveplate angle `w` to `theta = 2 * (w - 154.5°)` for its collection polarization angle. It fits

```text
P(theta) = (H - V) / (H + V)
         = P0 + a*cos(2*theta) + b*sin(2*theta)
A = sqrt(a² + b²)
```

The fitted amplitude (A) quantifies the polarization modulation of the normalized H/V contrast. Under appropriately calibrated, complementary detection channels, it provides an estimate of the polarization visibility. The reported 1σ uncertainties are derived from the least-squares fit residuals and do not include uncertainties propagated from the preceding spectral analysis.


## Contents

```text
arpls_preprocessing/
joint_lorentzian_fit.py
fit_polarization_visibility_contrast.py
README.md
```

