from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROI_PATTERN = re.compile(
    r"(?:^|[-_ ])roi[-_ ]?(\d+)",
    re.IGNORECASE,
)

# Supports angles written as 0, 22.5, or 22p5.
EXPOL_PATTERN = re.compile(
    r"expol[_ -]?(?:wp[_ -]?)?(?P<angle>[-+]?\d+(?:[pP.]\d+)?)\s*deg",
    re.IGNORECASE,
)

COLPOL_PATTERNS = (
    re.compile(
        r"colpol[_ -]?wp[_ -]?(?P<angle>[-+]?\d+(?:[pP.]\d+)?)\s*deg",
        re.IGNORECASE,
    ),
    re.compile(
        r"colpol[_ -]?(?P<angle>[-+]?\d+(?:[pP.]\d+)?)\s*deg",
        re.IGNORECASE,
    ),
)


def _angle_to_float(angle_text: str) -> float:
    """Convert filename angle notation such as 19p5 -> 19.5."""
    return float(angle_text.lower().replace("p", "."))


def _parse_roi(filename: str) -> int | None:
    match = ROI_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def _parse_expol(filename: str) -> float | None:
    match = EXPOL_PATTERN.search(filename)
    return _angle_to_float(match.group("angle")) if match else None


def _parse_colpol(filename: str) -> float | None:
    for pattern in COLPOL_PATTERNS:
        match = pattern.search(filename)
        if match:
            return _angle_to_float(match.group("angle"))
    return None


def find_spectrum_files(
    folder: Path,
    file_pattern: str = "*.csv",
    expol_to_analyze: float | None = None,
    required_text: tuple[str, ...] = (),
) -> dict[tuple[float, int], Path]:
    """Find spectra and index them by ``(ColPol angle, ROI)``.

    Supported examples include::

        ..._expol_0deg_colpol_WP19p5deg-Roi-1.csv
        ..._expol_0deg_colpol_WP42deg-Roi-2.csv
        ..._expol_22p5deg_colpol_64p5deg-Roi-1.csv

    Decimal angles may use either ``p`` or ``.``.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist:\n{folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"This path is not a folder:\n{folder}")

    files = sorted(folder.glob(file_pattern))
    output: dict[tuple[float, int], Path] = {}
    duplicates: dict[tuple[float, int], list[Path]] = {}

    required_lower = tuple(text.lower() for text in required_text)

    for path in files:
        name_lower = path.name.lower()

        if any(text not in name_lower for text in required_lower):
            continue

        roi = _parse_roi(path.name)
        angle = _parse_colpol(path.name)
        expol = _parse_expol(path.name)

        if roi is None or angle is None:
            continue

        if expol_to_analyze is not None:
            if expol is None:
                continue
            if not np.isclose(expol, float(expol_to_analyze), atol=1e-6):
                continue

        key = (float(angle), int(roi))

        if key in output:
            duplicates.setdefault(key, [output[key]]).append(path)
        else:
            output[key] = path

    if duplicates:
        lines = ["Duplicate spectra were found for the same ColPol angle and ROI:"]
        for (angle, roi), paths in sorted(duplicates.items()):
            lines.append(f"  ColPol {angle:g}°, ROI {roi}:")
            lines.extend(f"    {p.name}" for p in paths)
        raise RuntimeError("\n".join(lines))

    return output


def _find_column_case_insensitive(columns, requested_name: str) -> str | None:
    requested = requested_name.strip().lower()
    for column in columns:
        if str(column).strip().lower() == requested:
            return column
    return None


def read_spectrum(
    file_path: Path,
    wavelength_column: str = "Wavelength",
    intensity_column: str = "Intensity",
) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectrum CSV and return wavelength and intensity arrays."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        frame = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as error:
        raise ValueError(f"Could not read CSV file:\n{file_path}\n{error}") from error

    if frame.empty:
        raise ValueError(f"Spectrum file is empty:\n{file_path}")

    wl_col = _find_column_case_insensitive(frame.columns, wavelength_column)
    in_col = _find_column_case_insensitive(frame.columns, intensity_column)

    if wl_col is not None and in_col is not None:
        wavelength = pd.to_numeric(
            frame[wl_col], errors="coerce"
        ).to_numpy(dtype=float)
        intensity = pd.to_numeric(
            frame[in_col], errors="coerce"
        ).to_numpy(dtype=float)
    else:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        usable_columns = [
            col for col in numeric.columns
            if numeric[col].notna().sum() >= 2
        ]

        if len(usable_columns) < 2:
            raise ValueError(
                "Could not identify wavelength and intensity columns in:\n"
                f"{file_path}\n"
                f"Columns found: {list(frame.columns)}\n"
                f"Expected: {wavelength_column!r} and {intensity_column!r}."
            )

        wl_col, in_col = usable_columns[:2]
        wavelength = numeric[wl_col].to_numpy(dtype=float)
        intensity = numeric[in_col].to_numpy(dtype=float)

    valid = np.isfinite(wavelength) & np.isfinite(intensity)
    wavelength = wavelength[valid]
    intensity = intensity[valid]

    if wavelength.size < 2:
        raise ValueError(f"Fewer than two valid numerical rows in:\n{file_path}")

    order = np.argsort(wavelength)
    return wavelength[order], intensity[order]
