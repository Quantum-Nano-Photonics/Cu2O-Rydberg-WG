# Cu2O Rydberg Exciton Spectroscopy — analysis scripts

Python analysis workflows accompanying a study of polarization-resolved Cu2O exciton photoluminescence collected from island and grating/waveguide geometries. **This is a code-only repository scaffold**: experimental raw data and final figure datasets are not included.

## Analysis pipeline

1. **`arpls\\\_preprocessing/`** — read raw spectral CSVs, detect isolated spikes, fit an adaptive iteratively reweighted penalized least-squares (arPLS) background, subtract it, apply the relative ROI2 collection-efficiency factor, save corrected spectra, and perform a diagnostic trapezoidal integration of the baseline-corrected spectra.
2. **`joint\\\_lorentzian\\\_fit.py`** — load the corrected ROI1/ROI2 spectra, interpolate onto a common wavelength grid, fit two Lorentzian peaks jointly with shared centers/FWHMs and ROI-specific amplitudes/offsets, calculate the fitted 2p area and ROI differences, and save figures/results. The integrated area of the second fitted Lorentzian, corresponding to the 2p transition, is used for the subsequent polarization contrast and visibility analysis.
3. **`fit\\\_polarization\\\_visibility\\\_contrast.py`** — load the 2p fitted areas from the island and G1 geometries; calculate normalized H/V contrast and its sinusoidal modulation amplitude, offset and residual-based standard errors.

## Installation

Requires Python 3.10+ (NumPy >= 2.0 for `np.trapezoid`). From the repository root:

```bash
python -m venv .venv
```

Activate the environment (Windows PowerShell):

```powershell
.venv\\\\Scripts\\\\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Step 1 — preprocess raw spectra

Place the input CSV spectra in `data/raw/`, or set environment variable `CU2O\\\_RAW\\\_DATA\\\_DIR` to a folder containing them. Input filenames must match the selectors in `arpls\\\_preprocessing/config\\\_arpls.py`. The supplied configuration expects `ExPol = 0°`, `ROI1/ROI2`, `\\\_eI\\\_` excitation, `\\\_120s\\\_` spectra, and **`COLLECTION = "cI"`**. Input CSVs ordinarily have `Wavelength` and `Intensity` columns; see `spectrum\\\_io\\\_arpls.py` for accepted fallbacks and filename parsing.

Run from the repository root:

```bash
python arpls\\\_preprocessing/main\\\_arpls.py
```

Then edit `COLLECTION = "cG1"` in `arpls\\\_preprocessing/config\\\_arpls.py` and repeat to prepare the G1 collection configuration. You can switch back to `cI` without deleting the G1 output.

Outputs appear in:

```text
results/yellow/results\\\_arpls\\\_cI\\\_eI\\\_no\\\_background/corrected\\\_spectra/
results/yellow/results\\\_arpls\\\_cG1\\\_eI\\\_no\\\_background/corrected\\\_spectra/
```

The preprocessing uses **arPLS rather than a separately measured background file**. After baseline subtraction, it multiplies ROI2 by the experimentally supplied relative-efficiency factor `0.728` and leaves ROI1 unchanged. Both original and corrected intensities are saved; this factor must not be applied again downstream. 

## Step 2 — jointly fit ROI1/ROI2 spectra

From the repository root:

```bash
python joint\\\_lorentzian\\\_fit.py --collection island --no-show
python joint\\\_lorentzian\\\_fit.py --collection G1 --no-show
```

The default `--paper-root` is the repository root. To fit files stored elsewhere:

```bash
python joint\\\_lorentzian\\\_fit.py --collection island --input-dir "path/to/corrected\\\_spectra" --output-dir "path/to/results" --no-show
```

Outputs include `ROI2\\\_minus\\\_ROI1\\\_joint\\\_lorentzian\\\_results.csv` under each `results/yellow/results\\\_lorentzian\\\_fit\\\_ROI2\\\_minus\\\_ROI1\\\_...` directory.


## Step 3 — fit polarization contrast

After completing *both* island and G1 Lorentzian analyses:

```bash
python fit\\\_polarization\\\_visibility\\\_contrast.py
```

Outputs are saved in `polarization\\\_contrast\\\_fit\\\_results/`. This script assumes **ROI2 = H and ROI1 = V**, uses the already corrected integrated intensities, and maps waveplate angle `w` to `theta = 2 \\\* (w - 154.5°)` for its collection polarization angle. It fits

```text
P(theta) = (H - V) / (H + V)
         = P0 + a\\\*cos(2\\\*theta) + b\\\*sin(2\\\*theta)
A = sqrt(a² + b²)
```

`A` is the fitted **contrast modulation amplitude**. Its correspondence to conventional `(I\\\_max-I\\\_min)/(I\\\_max+I\\\_min)` visibility assumes appropriately calibrated complementary channels; check `P0` and polarization calibration. The printed 1σ errors are from an unweighted linear least-squares residual covariance estimate, **not** propagation of independently measured spectrum/fit uncertainties.


## Contents

```text
arpls\\\_preprocessing/
    arpls\\\_baseline.py
    config\\\_arpls.py
    main\\\_arpls.py
    plotting\\\_arpls.py
    processing\\\_arpls.py
    spectrum\\\_io\\\_arpls.py
    README.txt
joint\\\_lorentzian\\\_fit.py
fit\\\_polarization\\\_visibility\\\_contrast.py
.gitignore
README.md
data/raw/.gitkeep
```

### Scope of changes from the supplied archive

The analysis scripts are organized to support reproducible spectral preprocessing, Lorentzian peak fitting, and polarization visibility calculations. Input and output directories can be configured for different computing environments.

