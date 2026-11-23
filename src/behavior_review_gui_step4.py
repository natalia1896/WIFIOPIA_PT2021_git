"""Behavior review GUI for Step4 WIFIOPIA/OISI data.

Purpose
-------
A lightweight manual GUI that works directly from Step4 outputs, without Step5:

- iterates through Step4 stems using ``build_step4_index``;
- lets the user manually choose a behavior video for the current stem;
- lets the user mark video start/stop frames;
- shows a dynamic selected-signal map at the current trace cursor and a simple rectangular ROI;
- can use per-frame contrast or global contrast computed once from the whole recording inside the Step5-like brain mask;
- can optionally open stems missing a confirmed ``review_status=reviewed`` (pending),
  or only stems with no saved RUN/REST intervals (including ``[]``);
- extracts an HbT/HbTvreg mean trace from the selected ROI in Step4 processed zarr;
- synchronizes a cursor on the trace with the behavior video frame;
- saves video mapping, ROI mask, and optional run/rest intervals.

Expected location
-----------------
Put this file in the same ``src`` folder as ``step4_signal_processing.py``
or run it with that folder on PYTHONPATH.

Example
-------
python behavior_review_gui_step4.py --config path/to/config.yaml

Resume from stems not yet present in behavior_review_annotations.csv:
python behavior_review_gui_step4.py --config path/to/config.yaml --review-mode unreviewed

Example CLI filters (adjust values to the metadata of your recordings):
python behavior_review_gui_step4.py --config path/to/config.yaml --filter exp_day=Day8 --filter state=Awake

Optional global signal-map contrast, computed once per stem/signal over the whole
recording inside the Step5-like brain mask:
python behavior_review_gui_step4.py --config path/to/config.yaml --map-contrast global --map-percentiles 2 98
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import xarray as xr

try:
    from qtpy import QtCore, QtGui, QtWidgets
except Exception as e:  # pragma: no cover
    raise ImportError(
        "This GUI needs a Qt binding through qtpy. In the napari environment this is usually already installed. "
        "Try installing qtpy + PyQt5/PySide6 or run from the same environment where napari works."
    ) from e

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

# The GUI lives next to the current Step4 module in src.
from step4_signal_processing import (
    build_step4_index, cfg_get, open_raw_zarr,
    select_raw_zarr_path_for_processing,
)
from common import canonical_mask_paths


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v")


@dataclass
class VideoState:
    """Store one recording's video-file mapping and frame bounds."""
    stem: str
    group_id: str = ""
    behavior_video_path: str = ""
    video_exists: bool = False
    video_fps: float = float("nan")
    video_n_frames: int = 0
    video_width: int = 0
    video_height: int = 0
    video_start_frame: int = 0
    video_stop_frame: int = 0
    updated_at: str = ""


@dataclass
class AnnotationState:
    """Store a stem-level behavior annotation row for CSV persistence."""
    stem: str
    group_id: str = ""
    mouse_id: str = ""
    exp_day: str = ""
    state: str = ""
    protocol: str = ""
    raw_zarr_path_resolved: str = ""
    processed_zarr_path_resolved: str = ""
    behavior_video_path: str = ""
    signal_name: str = ""
    roi_y0: int = -1
    roi_x0: int = -1
    roi_y1: int = -1
    roi_x1: int = -1
    roi_mask_path: str = ""
    label_intervals_json: str = "[]"
    review_status: str = "reviewed"
    notes: str = ""
    updated_at: str = ""


def now_iso() -> str:
    """Return the current local timestamp in ISO-8601 seconds precision."""
    return datetime.now().isoformat(timespec="seconds")


def safe_str(value: Any) -> str:
    """Convert a possibly missing scalar to a string; map missing values to empty text."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def normalize_to_uint8(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize a display image to unsigned 8-bit intensities.
    
    Parameters
    ----------
    img : numpy.ndarray
        Grayscale (y, x) or RGB/RGBA (y, x, channel) image.
    p_low, p_high : float
        Percentile limits used for non-uint8 input; in percent.
    
    Returns
    -------
    numpy.ndarray
        Display-only uint8 image, with non-finite values mapped to black.
    
    Notes
    -----
    This operation does not modify the raw or processed Zarr."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr.copy()
    x = arr.astype(np.float32, copy=False)
    finite = x[np.isfinite(x)]
    if finite.size < 10:
        return np.zeros(x.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(finite, [float(p_low), float(p_high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(x.shape, dtype=np.uint8)
    out = np.clip((x - lo) / (hi - lo), 0, 1)
    # Newer NumPy warns when NaN is cast to uint8.  NaNs can appear outside
    # valid signal pixels, so make display-only invalid values black.
    out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.round(out * 255).astype(np.uint8)


def numpy_to_qimage(img: np.ndarray) -> QtGui.QImage:
    """Copy a grayscale, RGB, or RGBA uint8 image into a Qt QImage."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr8 = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        h, w = arr8.shape
        qimg = QtGui.QImage(arr8.data, w, h, int(arr8.strides[0]), QtGui.QImage.Format_Grayscale8)
        return qimg.copy()
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr8 = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        h, w, _ = arr8.shape
        qimg = QtGui.QImage(arr8.data, w, h, int(arr8.strides[0]), QtGui.QImage.Format_RGB888)
        return qimg.copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr8 = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        h, w, _ = arr8.shape
        qimg = QtGui.QImage(arr8.data, w, h, int(arr8.strides[0]), QtGui.QImage.Format_RGBA8888)
        return qimg.copy()
    raise ValueError(f"Unsupported image shape for Qt display: {arr.shape}")


def csv_safe_value(value: Any) -> str:
    """Return a pandas/CSV-safe scalar string for old and new pandas versions.

    Newer pandas can use Arrow-backed string columns and refuses assigning bool,
    int, float or None directly into them.  The behavior-review CSV files are
    metadata logs, so storing scalar values as text is the most portable option.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (np.bool_, bool)):
        return "True" if bool(value) else "False"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if not np.isfinite(f):
            return ""
        return repr(f)
    return str(value)


def read_csv_text_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a behavior CSV as plain object strings, never Arrow string dtype."""
    path = Path(path)
    df = pd.read_csv(
        path,
        dtype=object,
        keep_default_na=False,
        encoding="utf-8-sig",
        **kwargs,
    )
    # Make sure all existing scalar cells are object strings, not pyarrow strings.
    df = df.astype(object)
    for col in df.columns:
        df[col] = df[col].map(csv_safe_value).astype(object)
    return df


def upsert_csv_row(path: Path, row: dict[str, Any], key: str = "stem") -> None:
    """Insert or update the latest CSV row matching the selected key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_safe = {str(col): csv_safe_value(val) for col, val in row.items()}
    key = str(key)

    if path.exists():
        # Keep everything as object strings. This avoids pandas 2.x/Arrow errors
        # such as assigning False into a string column or "" into a float column.
        df = read_csv_text_table(path)
    else:
        df = pd.DataFrame(dtype=object)

    for col in row_safe:
        if col not in df.columns:
            df[col] = ""
    if key not in df.columns:
        df[key] = ""

    df = df.astype(object)
    key_value = csv_safe_value(row_safe.get(key, "")).strip()
    key_series = df[key].map(csv_safe_value).astype(str).str.strip()
    mask = key_series.eq(key_value)

    if mask.any():
        idx = df.index[mask][-1]
        for col, val in row_safe.items():
            df.at[idx, col] = csv_safe_value(val)
    else:
        df = pd.concat([df, pd.DataFrame([row_safe], dtype=object)], ignore_index=True, sort=False)

    ordered = list(row_safe.keys()) + [c for c in df.columns if c not in row_safe]
    df = df[ordered].astype(object)
    for col in df.columns:
        df[col] = df[col].map(csv_safe_value).astype(object)
    df.to_csv(path, index=False, encoding="utf-8")


def read_latest_row(path: Path, stem: str) -> pd.Series | None:
    """Return the last saved annotation row for a stem, if available."""
    path = Path(path)
    if not path.exists():
        return None
    df = read_csv_text_table(path)
    if df.empty or "stem" not in df.columns:
        return None
    target = csv_safe_value(stem).strip()
    stems = df["stem"].map(csv_safe_value).astype(str).str.strip()
    sub = df[stems.eq(target)]
    if sub.empty:
        return None
    return sub.iloc[-1]


def read_stems_present_in_csv(path: Path) -> set[str]:
    """Return stems that already have at least one row in a CSV file.

    This is intentionally based only on stem presence, not on review_status, so
    old behavior-review CSVs remain compatible and no new columns are required.
    """
    path = Path(path)
    if not path.exists():
        return set()
    try:
        df = read_csv_text_table(path, usecols=["stem"])
    except Exception:
        df = read_csv_text_table(path)
    if df.empty or "stem" not in df.columns:
        return set()
    stems = df["stem"].map(csv_safe_value).astype(str).str.strip()
    return {s for s in stems if s}


def read_stems_explicitly_reviewed(path: Path) -> set[str]:
    """Return stems whose latest CSV row is explicitly marked ``reviewed``.

    Missing file, missing review_status column, empty/unknown statuses and all
    ``in_progress`` rows stay eligible for ``--review-mode pending``.  For
    duplicate stem rows the last row is authoritative, as in read_latest_row().
    """
    path = Path(path)
    if not path.exists():
        return set()
    df = read_csv_text_table(path)
    if df.empty or "stem" not in df.columns or "review_status" not in df.columns:
        return set()
    work = df.copy()
    work["stem"] = work["stem"].map(csv_safe_value).astype(str).str.strip()
    work = work[work["stem"].ne("")].drop_duplicates(subset=["stem"], keep="last")
    reviewed = work["review_status"].map(csv_safe_value).astype(str).str.strip().str.lower().eq("reviewed")
    return set(work.loc[reviewed, "stem"].tolist())


def read_stems_with_nonempty_intervals(path: Path) -> set[str]:
    """Return stems whose latest annotation row contains at least one interval.

    This deliberately uses only ``label_intervals_json`` and does not change the
    CSV schema. A stem already present in the CSV with ``[]`` is therefore still
    considered unannotated and can be opened again; saving will update the same
    stem row via the existing ``upsert_csv_row(..., key="stem")`` logic.
    """
    path = Path(path)
    if not path.exists():
        return set()
    df = read_csv_text_table(path)
    if df.empty or "stem" not in df.columns:
        return set()
    annotated: set[str] = set()
    # read_latest_row() semantics: the last row for a stem is the active state.
    for stem_raw in df["stem"].map(csv_safe_value).astype(str).str.strip().unique():
        stem = str(stem_raw).strip()
        if not stem:
            continue
        sub = df[df["stem"].map(csv_safe_value).astype(str).str.strip().eq(stem)]
        if sub.empty:
            continue
        latest = sub.iloc[-1]
        text = safe_str(latest.get("label_intervals_json")).strip() or "[]"
        try:
            intervals = json.loads(text)
        except Exception:
            intervals = []
        if isinstance(intervals, list) and len(intervals) > 0:
            annotated.add(stem)
    return annotated


def parse_cli_filters(items: Sequence[str] | None) -> dict[str, Any] | None:
    """Parse repeated CLI KEY=VALUE and comma-separated metadata filters."""
    if not items:
        return None
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Filter must be KEY=VALUE or KEY=a,b, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty filter key in {item!r}")
        if "," in value:
            out[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            out[key] = value
    return out


def normalise_wavelength_label(value: Any) -> str:
    """Normalize numerical wavelength labels to comparable strings."""
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(f)
    except Exception:
        return str(value).strip()


def select_wavelength_index(raw: xr.DataArray, wanted: str | int | float | None) -> int:
    """Select the requested raw wavelength index, with a fixed fallback.
    
    Parameters
    ----------
    raw : xarray.DataArray
        Raw array containing a wavelength coordinate.
    wanted : str, int, float, or None
        Requested wavelength label, typically in nm.
    
    Returns
    -------
    int
        Index of the matching wavelength, or the 505-nm channel when present,
        otherwise the first available channel.
    
    Notes
    -----
    The 505-nm fallback is hard-coded in this helper. The GUI's explicit
    ``--raw-preview-wavelength`` setting can select a different channel."""
    values = list(raw.coords["wavelength"].values)
    labels = [normalise_wavelength_label(v) for v in values]
    if wanted is not None:
        w = normalise_wavelength_label(wanted)
        if w in labels:
            return labels.index(w)
    # Backward-compatible 505-nm fallback; otherwise use the first available channel.
    # This fallback is not a value read from the YAML configuration.
    if "505" in labels:
        return labels.index("505")
    return 0


def resize_bool_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a spatial boolean mask using nearest-neighbor sampling."""
    mask = np.asarray(mask, dtype=bool)
    h, w = int(shape[0]), int(shape[1])
    if tuple(mask.shape) == (h, w):
        return mask
    return cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _load_npy_mask_for_display(
    path: Path,
    *,
    image_shape: tuple[int, int],
    resize_if_needed: bool,
) -> np.ndarray:
    """Read a canonical NPY mask and optionally resize to the signal grid."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = np.load(path).astype(bool)
    if tuple(mask.shape) != tuple(image_shape):
        if not resize_if_needed:
            raise ValueError(
                f"Mask shape mismatch for {path}: {mask.shape} vs {image_shape}. "
                "Set signal_processing.masks.resize_to_raw_if_needed: true or rebuild matching masks/zarr."
            )
        mask = resize_bool_mask(mask, image_shape)
    return mask.astype(bool)


def load_step5_like_brain_signal_mask(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    image_shape: tuple[int, int],
    raw_valid: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    """Construct the display-domain brain-signal validity mask.
    
    Parameters
    ----------
    row : pandas.Series
        Step4 index row containing the recording ``stem``.
    config : dict
        Full YAML configuration; reads
        ``signal_processing.masks.resize_to_raw_if_needed``.
    paths : dict[str, pathlib.Path]
        Project paths used to resolve canonical Step2 masks.
    image_shape : tuple[int, int]
        Target (y, x) signal-map shape in pixels.
    raw_valid : numpy.ndarray
        Boolean validity mask on the processed (y, x) grid.
    
    Returns
    -------
    mask : numpy.ndarray
        Boolean array with shape (y, x). True marks valid brain pixels.
    sources : dict[str, str]
        Resolved mask filenames and selected manual/automatic sources.
    
    Notes
    -----
    The implemented combination is ``brain & ~bad & ~bregma & raw_valid``.
    Existing manual brain/bregma masks take precedence when their files exist.
    This function does not modify analysis outputs."""
    stem = str(row["stem"])
    cps = canonical_mask_paths(paths, stem)
    resize_if_needed = bool(cfg_get(config, "signal_processing", "masks", "resize_to_raw_if_needed", default=True))

    brain_path = cps["brain_manual"] if cps["brain_manual"].exists() else cps["brain_auto"]
    bregma_path = cps["bregma_manual"] if cps["bregma_manual"].exists() else cps["bregma_auto"]
    bad_path = cps["bad_auto"]

    brain = _load_npy_mask_for_display(brain_path, image_shape=image_shape, resize_if_needed=resize_if_needed)
    bregma = _load_npy_mask_for_display(bregma_path, image_shape=image_shape, resize_if_needed=resize_if_needed)
    bad = _load_npy_mask_for_display(bad_path, image_shape=image_shape, resize_if_needed=resize_if_needed)

    if tuple(raw_valid.shape) != tuple(image_shape):
        raise ValueError("raw_valid grid does not match processed Hb grid")
    brain_signal = brain & (~bad) & (~bregma) & np.asarray(raw_valid, dtype=bool)
    sources = {
        "brain_path": str(brain_path),
        "brain_source": "manual" if brain_path.name == "brain_manual.npy" else "auto",
        "bregma_path": str(bregma_path),
        "bregma_source": "manual" if bregma_path.name == "bregma_manual.npy" else "auto",
        "bad_path": str(bad_path),
        "mask_formula": "brainvessels_signal = brain & ~bad & ~bregma & raw_valid",
    }
    return brain_signal.astype(bool), sources


def gaussian_masked_display_frame(frame: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Spatial Gaussian for display only, without bleeding across invalid pixels."""
    arr = np.asarray(frame, dtype=np.float32)
    allowed = np.asarray(mask, dtype=bool)
    if arr.ndim != 2 or allowed.shape != arr.shape:
        raise ValueError("Display Gaussian requires matching 2D frame and mask")
    if not np.isfinite(sigma_px) or sigma_px < 0:
        raise ValueError("Display Gaussian sigma must be finite and nonnegative")
    valid = allowed & np.isfinite(arr)
    if sigma_px == 0:
        return np.where(valid, arr, np.nan)
    weights = valid.astype(np.float32)
    numerator = gaussian_filter(np.where(valid, arr, 0.0), float(sigma_px), mode="constant", cval=0.0)
    denominator = gaussian_filter(weights, float(sigma_px), mode="constant", cval=0.0)
    result = np.full(arr.shape, np.nan, dtype=np.float32)
    result[valid] = numerator[valid] / np.maximum(denominator[valid], 1e-12)
    return result


def gaussian_temporal_display_trace(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian smooth a plotted 1D trace in *frames* (not seconds), display-only.

    Finite samples are smoothed using normalized weights; invalid samples stay NaN.
    No filtering, normalization, or changes to the behavior time axis occur.
    """
    x = np.asarray(values, dtype=np.float64)
    sigma = float(sigma_frames)
    if x.ndim != 1:
        raise ValueError("Temporal Gaussian display smoothing requires a 1D trace")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Temporal Gaussian sigma must be finite and nonnegative")
    if sigma == 0 or x.size == 0:
        return x.copy()
    finite = np.isfinite(x)
    if not finite.any():
        return x.copy()
    numerator = gaussian_filter1d(np.where(finite, x, 0.0), sigma=sigma, mode="reflect")
    denominator = gaussian_filter1d(finite.astype(float), sigma=sigma, mode="reflect")
    result = np.full(x.shape, np.nan, dtype=np.float64)
    result[finite] = numerator[finite] / np.maximum(denominator[finite], 1e-12)
    return result


def bandpass_display_trace(
    values: np.ndarray, fps: float, low_hz: float = 0.08, high_hz: float = 0.40,
    order: int = 4, pad_sec: float = 20.0,
) -> np.ndarray:
    """Filter a one-dimensional ROI trace for display only.
    
    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional time series.
    fps : float
        Sampling rate in Hz.
    low_hz, high_hz : float
        Band-pass frequency bounds in Hz.
    order : int
        Butterworth filter order.
    pad_sec : float
        Requested edge padding in seconds, capped by the trace length.
    
    Returns
    -------
    numpy.ndarray
        Zero-phase filtered display trace, in the input units.
    
    Notes
    -----
    This function never changes stored signals or behavior intervals."""
    x = np.asarray(values, dtype=np.float64)
    fs = float(fps)
    if x.ndim != 1 or x.size < 4:
        raise ValueError("A 1D trace with at least four frames is required")
    if not np.all(np.isfinite(x)):
        raise ValueError("Cannot band-pass an ROI trace containing NaN/inf; turn filter off or fix invalid pixels")
    if not np.isfinite(fs) or not (0 < low_hz < high_hz < fs / 2):
        raise ValueError(f"Invalid frequency limits {low_hz}–{high_hz} Hz for fps={fps}")
    sos = butter(int(order), [float(low_hz), float(high_hz)], btype="bandpass", fs=fs, output="sos")
    padlen = min(int(round(pad_sec * fs)), max(0, x.size - 2))
    return np.asarray(sosfiltfilt(sos, x, padtype="odd" if padlen else None, padlen=padlen), dtype=np.float64)


def signal_frame_to_jet_rgb(
    frame: np.ndarray,
    mask: np.ndarray | None,
    *,
    percentile_range: tuple[float, float] = (2.0, 98.0),
    fixed_limits: tuple[float, float] | None = None,
    outside_value: int = 0,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Convert one signal frame to an RGB jet image.

    If ``fixed_limits`` is None, vmin/vmax are computed from this frame only
    inside ``mask``.  If ``fixed_limits`` is provided, the same vmin/vmax are
    reused for every frame of the current recording/signal (global contrast).
    """
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Signal map frame must be 2D, got shape={arr.shape}")

    finite = np.isfinite(arr)
    if mask is not None:
        mask2 = resize_bool_mask(mask, arr.shape)
        valid = finite & mask2
    else:
        mask2 = np.ones(arr.shape, dtype=bool)
        valid = finite

    vals = arr[valid]
    if vals.size < 10:
        raise ValueError("Too few finite pixels inside brain_signal mask to draw signal map")

    if fixed_limits is None:
        p_low, p_high = map(float, percentile_range)
        vmin, vmax = np.nanpercentile(vals, [p_low, p_high])
    else:
        vmin, vmax = map(float, fixed_limits)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    # Non-finite signal pixels are drawn as outside/black; avoid NumPy cast warnings.
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    gray = np.round(norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~mask2 | ~finite] = int(outside_value)
    return rgb.astype(np.uint8), (float(vmin), float(vmax))


def compute_global_signal_contrast_limits(
    var: xr.DataArray,
    mask: np.ndarray | None,
    *,
    percentile_range: tuple[float, float] = (2.0, 98.0),
    time_chunk: int = 50,
    max_values: int = 2_000_000,
) -> tuple[tuple[float, float], str]:
    """Determine common display contrast limits across one recording.
    
    Parameters
    ----------
    var : xarray.DataArray
        Signal with dimensions (time, y, x).
    mask : numpy.ndarray or None
        Valid display pixels with shape (y, x).
    percentile_range : tuple[float, float]
        Lower and upper percentiles, in percent.
    time_chunk : int
        Number of time frames materialized per read.
    max_values : int
        Cap on collected values; when exceeded, sample brain pixels
        deterministically while retaining every time point.
    
    Returns
    -------
    limits : tuple[float, float]
        Lower and upper contrast limits in the input signal's units.
    mode_note : str
        Whether the scan used all pixels or a spatial subsample.
    
    Notes
    -----
    Contrast affects the displayed colormap only and is not saved into Zarr."""
    if set(var.dims) != {"time", "y", "x"}:
        raise ValueError(f"Global contrast requires a time,y,x variable, got dims={var.dims}")

    v = var.transpose("time", "y", "x")
    n_time = int(v.sizes["time"])
    h = int(v.sizes["y"])
    w = int(v.sizes["x"])
    if n_time <= 0 or h <= 0 or w <= 0:
        raise ValueError("Empty signal array")

    if mask is not None:
        mask2 = resize_bool_mask(mask, (h, w))
    else:
        mask2 = np.ones((h, w), dtype=bool)
    flat_idx_all = np.flatnonzero(mask2.ravel())
    if flat_idx_all.size < 10:
        raise ValueError("Too few pixels inside brain_signal mask for global contrast")

    max_values_i = int(max_values)
    total_values = int(n_time) * int(flat_idx_all.size)
    if max_values_i <= 0 or total_values <= max_values_i:
        flat_idx = flat_idx_all
        mode_note = f"exact all masked values, n_time={n_time}, n_pixels={flat_idx_all.size}"
    else:
        # Keep all time points, but use a deterministic spatial subsample so the
        # GUI does not allocate hundreds of millions of values on large recordings.
        n_px = max(10, min(int(flat_idx_all.size), max_values_i // max(n_time, 1)))
        rng = np.random.default_rng(0)
        flat_idx = np.sort(rng.choice(flat_idx_all, size=n_px, replace=False))
        mode_note = f"sampled all time points, n_time={n_time}, sampled_pixels={n_px}/{flat_idx_all.size}, max_values={max_values_i}"

    chunks: list[np.ndarray] = []
    chunk = max(1, int(time_chunk))
    for t0 in range(0, n_time, chunk):
        t1 = min(n_time, t0 + chunk)
        arr = np.asarray(v.isel(time=slice(t0, t1)).compute().values, dtype=np.float32)
        vals = arr.reshape(arr.shape[0], -1)[:, flat_idx].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            chunks.append(vals)

    if not chunks:
        raise ValueError("No finite masked values found for global contrast")
    all_vals = np.concatenate(chunks)
    if all_vals.size < 10:
        raise ValueError("Too few finite masked values found for global contrast")

    p_low, p_high = map(float, percentile_range)
    vmin, vmax = np.nanpercentile(all_vals, [p_low, p_high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return (float(vmin), float(vmax)), mode_note


def candidate_hbt_signals(ds: xr.Dataset) -> list[tuple[str, str]]:
    """Return display-label / variable-name pairs for HbT-like variables."""
    names = list(ds.data_vars)
    lower_to_name = {n.lower(): n for n in names}

    def first_existing(candidates: Sequence[str]) -> str | None:
        """Find the first available processed signal variable among candidate names."""
        for c in candidates:
            if c in names:
                return c
            if c.lower() in lower_to_name:
                return lower_to_name[c.lower()]
        return None

    choices: list[tuple[str, str]] = []
    hbt = first_existing(["dcHbT", "HbT", "hbt", "dcHBT"])
    if hbt is not None:
        choices.append((f"HbT ({hbt})", hbt))

    # Common possible names for vessel-regressed HbT.  The exact variable depends
    # on output/global-signal settings, so keep this intentionally broad.
    hbt_vreg = first_existing([
        "dcHbT_vreg",
        "dcHbT_vessel_regression",
        "HbT_vreg",
        "HbTvreg",
        "dcHbTvreg",
        "hbt_vreg",
    ])
    if hbt_vreg is not None and hbt_vreg != hbt:
        choices.append((f"HbTvreg ({hbt_vreg})", hbt_vreg))

    # Add every 3D/1D variable containing hbt that is not already listed.
    seen = {v for _, v in choices}
    for name in names:
        if "hbt" not in name.lower():
            continue
        var = ds[name]
        dims = set(var.dims)
        if (dims == {"time", "y", "x"} or list(var.dims) == ["time"]) and name not in seen:
            label = name
            if "vreg" in name.lower() and hbt_vreg is None:
                label = f"HbTvreg ({name})"
            elif name != hbt:
                label = f"HbT-like ({name})"
            choices.append((label, name))
            seen.add(name)
    return choices


# -----------------------------------------------------------------------------
# Custom widgets
# -----------------------------------------------------------------------------

class RoiImageWidget(QtWidgets.QWidget):
    """Show a raw image and let the user draw a rectangular spatial ROI."""
    roiChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize this GUI object and its display state."""
        super().__init__(parent)
        self.setMinimumSize(320, 320)
        self.setMouseTracking(True)
        self._image_uint8: np.ndarray | None = None
        self._qimage: QtGui.QImage | None = None
        self._pixmap: QtGui.QPixmap | None = None
        self._display_rect = QtCore.QRectF()
        self._roi_xyxy: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1 inclusive-ish
        self._dragging = False
        self._drag_start_xy: tuple[int, int] | None = None

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """Return the current image dimensions in (y, x) order."""
        if self._image_uint8 is None:
            return None
        return int(self._image_uint8.shape[0]), int(self._image_uint8.shape[1])

    def set_image(self, img: np.ndarray) -> None:
        """Replace the displayed image and update the ROI widget."""
        self._image_uint8 = normalize_to_uint8(img)
        self._qimage = numpy_to_qimage(self._image_uint8)
        self._pixmap = QtGui.QPixmap.fromImage(self._qimage)
        self.update()

    def set_roi_yxyx(self, y0: int, x0: int, y1: int, x1: int) -> None:
        """Set the rectangular ROI bounds in (y0, x0, y1, x1) order."""
        if self.image_shape is None:
            return
        h, w = self.image_shape
        x0 = int(np.clip(x0, 0, w - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        self._roi_xyxy = (x0, y0, x1, y1)
        self.update()
        self.roiChanged.emit()

    def set_default_roi(self) -> None:
        """Initialize a default rectangular ROI on the current image."""
        if self.image_shape is None:
            return
        h, w = self.image_shape
        size = int(max(20, min(h, w) * 0.18))
        cy, cx = h // 2, w // 2
        self.set_roi_yxyx(cy - size // 2, cx - size // 2, cy + size // 2, cx + size // 2)

    def clear_roi(self) -> None:
        """Remove the current rectangular ROI."""
        self._roi_xyxy = None
        self.update()
        self.roiChanged.emit()

    def roi_yxyx(self) -> tuple[int, int, int, int] | None:
        """Return the current rectangle bounds in image-pixel coordinates."""
        if self._roi_xyxy is None:
            return None
        x0, y0, x1, y1 = self._roi_xyxy
        return y0, x0, y1, x1

    def roi_mask(self) -> np.ndarray | None:
        """Rasterize the current rectangle into a boolean (y, x) mask."""
        if self.image_shape is None or self._roi_xyxy is None:
            return None
        h, w = self.image_shape
        x0, y0, x1, y1 = self._roi_xyxy
        m = np.zeros((h, w), dtype=bool)
        m[y0 : y1 + 1, x0 : x1 + 1] = True
        return m

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        """Draw the current image and ROI overlay in the Qt widget."""
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("black"))
        if self._pixmap is None:
            painter.setPen(QtGui.QColor("white"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "signal map\n(no image loaded)")
            return

        pix = self._pixmap
        scaled = pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) / 2
        y = (self.height() - scaled.height()) / 2
        self._display_rect = QtCore.QRectF(x, y, scaled.width(), scaled.height())
        painter.drawPixmap(int(x), int(y), scaled)

        if self._roi_xyxy is not None and self.image_shape is not None:
            x0, y0, x1, y1 = self._roi_xyxy
            r = self._image_rect_to_widget_rect(x0, y0, x1, y1)
            pen = QtGui.QPen(QtGui.QColor(255, 230, 0), 2)
            painter.setPen(pen)
            painter.drawRect(r)
            painter.setPen(QtGui.QColor(255, 230, 0))
            painter.drawText(r.topLeft() + QtCore.QPointF(4, 14), "ROI")

    def _widget_to_image_xy(self, pos: QtCore.QPoint | QtCore.QPointF) -> tuple[int, int] | None:
        """Map Qt widget coordinates to image coordinates (x, y)."""
        if self.image_shape is None or self._display_rect.isNull():
            return None
        if not self._display_rect.contains(QtCore.QPointF(pos)):
            return None
        h, w = self.image_shape
        rel_x = (float(pos.x()) - self._display_rect.left()) / max(self._display_rect.width(), 1)
        rel_y = (float(pos.y()) - self._display_rect.top()) / max(self._display_rect.height(), 1)
        x = int(np.clip(round(rel_x * (w - 1)), 0, w - 1))
        y = int(np.clip(round(rel_y * (h - 1)), 0, h - 1))
        return x, y

    def _image_rect_to_widget_rect(self, x0: int, y0: int, x1: int, y1: int) -> QtCore.QRectF:
        """Map an image rectangle to a Qt widget rectangle."""
        if self.image_shape is None:
            return QtCore.QRectF()
        h, w = self.image_shape
        left = self._display_rect.left() + (x0 / max(w - 1, 1)) * self._display_rect.width()
        right = self._display_rect.left() + (x1 / max(w - 1, 1)) * self._display_rect.width()
        top = self._display_rect.top() + (y0 / max(h - 1, 1)) * self._display_rect.height()
        bottom = self._display_rect.top() + (y1 / max(h - 1, 1)) * self._display_rect.height()
        return QtCore.QRectF(QtCore.QPointF(left, top), QtCore.QPointF(right, bottom)).normalized()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Start editing the selected ROI from a mouse press."""
        if event.button() != QtCore.Qt.LeftButton:
            return
        xy = self._widget_to_image_xy(event.position() if hasattr(event, "position") else event.pos())
        if xy is None:
            return
        self._dragging = True
        self._drag_start_xy = xy
        x, y = xy
        self._roi_xyxy = (x, y, x, y)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Update the selected ROI as the pointer moves."""
        if not self._dragging or self._drag_start_xy is None:
            return
        xy = self._widget_to_image_xy(event.position() if hasattr(event, "position") else event.pos())
        if xy is None:
            return
        x0, y0 = self._drag_start_xy
        x1, y1 = xy
        self._roi_xyxy = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        """Complete editing the current ROI from a mouse release."""
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self._dragging:
            self._dragging = False
            self._drag_start_xy = None
            self.roiChanged.emit()


class VideoWidget(QtWidgets.QLabel):
    """Show video frames in a resizable Qt image widget."""
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize this GUI object and its display state."""
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("background: black; color: white;")
        self._frame_rgb: np.ndarray | None = None
        self.setText("No video loaded")

    def set_frame(self, frame_rgb: np.ndarray | None, text: str | None = None) -> None:
        """Display a new video frame in the widget."""
        self._frame_rgb = frame_rgb
        if frame_rgb is None:
            self.setPixmap(QtGui.QPixmap())
            self.setText(text or "No video loaded")
        else:
            qimg = numpy_to_qimage(frame_rgb)
            pix = QtGui.QPixmap.fromImage(qimg)
            pix = pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.setPixmap(pix)
            self.setText("")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        """Update the video display when the widget is resized."""
        super().resizeEvent(event)
        if self._frame_rgb is not None:
            self.set_frame(self._frame_rgb)


class TraceCanvas(FigureCanvas):
    """Display ROI signals, behavior interval spans, and the time cursor."""
    cursorChanged = QtCore.Signal(int)

    RUN_COLOR = "red"
    REST_COLOR = "blue"
    INTERVAL_ALPHA = 0.18

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize this GUI object and its display state."""
        self.fig = Figure(figsize=(8, 3), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.times = np.array([], dtype=float)
        self.values = np.array([], dtype=float)
        self.cursor_index = 0
        self._cursor_line = None
        self._interval_spans: list[Any] = []
        self._interval_legend = None
        self._mouse_is_down = False
        self._full_xlim: tuple[float, float] | None = None
        self._full_ylim: tuple[float, float] | None = None
        self.mpl_connect("button_press_event", self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event", self._on_motion)

    def set_trace(self, times: np.ndarray, values: np.ndarray, *, title: str, signal_name: str) -> None:
        """Set the signal trace data and redraw the plot."""
        self.times = np.asarray(times, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.cursor_index = min(max(self.cursor_index, 0), max(len(self.times) - 1, 0))
        self.ax.clear()
        self._interval_legend = None
        if len(self.times) and len(self.values):
            self.ax.plot(self.times, self.values, linewidth=1.0, zorder=2)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel(signal_name)
            self.ax.set_title(title)
            self._cursor_line = self.ax.axvline(self.times[self.cursor_index], linewidth=1.2, zorder=3)
        else:
            self.ax.text(0.5, 0.5, "No trace", transform=self.ax.transAxes, ha="center", va="center")
            self.ax.set_title(title)
            self._cursor_line = None
        self._interval_spans = []
        if len(self.times) and len(self.values):
            self.ax.relim()
            self.ax.autoscale_view()
            self._full_xlim = tuple(map(float, self.ax.get_xlim()))
            self._full_ylim = tuple(map(float, self.ax.get_ylim()))
        else:
            self._full_xlim = None
            self._full_ylim = None
        self.draw_idle()

    def zoom_x(self, factor: float) -> None:
        """Zoom X around the current trace cursor. factor < 1 zooms in."""
        if len(self.times) == 0:
            return
        left, right = map(float, self.ax.get_xlim())
        width = max(right - left, np.finfo(float).eps)
        new_width = width * float(factor)
        center = float(self.times[int(np.clip(self.cursor_index, 0, len(self.times) - 1))])
        new_left = center - new_width / 2.0
        new_right = center + new_width / 2.0
        if self._full_xlim is not None and factor >= 1.0:
            full_left, full_right = self._full_xlim
            if new_width >= (full_right - full_left):
                new_left, new_right = full_left, full_right
            else:
                if new_left < full_left:
                    new_right += full_left - new_left
                    new_left = full_left
                if new_right > full_right:
                    new_left -= new_right - full_right
                    new_right = full_right
        self.ax.set_xlim(new_left, new_right)
        self.draw_idle()

    def zoom_y(self, factor: float) -> None:
        """Zoom Y around the current displayed Y-axis center. factor < 1 zooms in."""
        bottom, top = map(float, self.ax.get_ylim())
        height = max(top - bottom, np.finfo(float).eps)
        center = (bottom + top) / 2.0
        new_height = height * float(factor)
        new_bottom = center - new_height / 2.0
        new_top = center + new_height / 2.0
        if self._full_ylim is not None and factor >= 1.0:
            full_bottom, full_top = self._full_ylim
            if new_height >= (full_top - full_bottom):
                new_bottom, new_top = full_bottom, full_top
        self.ax.set_ylim(new_bottom, new_top)
        self.draw_idle()

    def reset_zoom(self) -> None:
        """Restore the original trace-axis view limits."""
        if self._full_xlim is not None:
            self.ax.set_xlim(*self._full_xlim)
        if self._full_ylim is not None:
            self.ax.set_ylim(*self._full_ylim)
        self.draw_idle()

    @staticmethod
    def _interval_color(label: Any) -> str | None:
        """Map a behavior label to its display color."""
        text = str(label or "").strip().lower()
        if text == "run":
            return TraceCanvas.RUN_COLOR
        if text == "rest":
            return TraceCanvas.REST_COLOR
        return None

    def _refresh_interval_legend(self, has_run: bool, has_rest: bool) -> None:
        """Refresh the legend for annotated behavior intervals."""
        if self._interval_legend is not None:
            try:
                self._interval_legend.remove()
            except Exception:
                pass
            self._interval_legend = None
        handles = []
        if has_run or has_rest:
            from matplotlib.patches import Patch
            if has_run:
                handles.append(Patch(facecolor=self.RUN_COLOR, alpha=self.INTERVAL_ALPHA, label="RUN"))
            if has_rest:
                handles.append(Patch(facecolor=self.REST_COLOR, alpha=self.INTERVAL_ALPHA, label="REST"))
        if handles:
            self._interval_legend = self.ax.legend(handles=handles, loc="upper right", framealpha=0.8)

    def set_cursor_index(self, idx: int, emit: bool = False) -> None:
        """Move the displayed trace cursor to the selected sample index."""
        if len(self.times) == 0:
            return
        idx = int(np.clip(idx, 0, len(self.times) - 1))
        self.cursor_index = idx
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([self.times[idx], self.times[idx]])
            self.draw_idle()
        if emit:
            self.cursorChanged.emit(idx)

    def nearest_index(self, x: float) -> int:
        """Find the sample index nearest to the requested cursor coordinate."""
        if len(self.times) == 0:
            return 0
        return int(np.nanargmin(np.abs(self.times - float(x))))

    def add_interval_spans(self, intervals: list[dict[str, Any]]) -> None:
        """Draw saved/manual behavior intervals over the ROI trace.

        RUN intervals are red, REST intervals are blue.  This same function is
        called both after adding a new interval and after a stem with previously
        saved ``label_intervals_json`` is loaded and the trace is recomputed.
        """
        for span in self._interval_spans:
            try:
                span.remove()
            except Exception:
                pass
        self._interval_spans = []
        has_run = False
        has_rest = False
        for item in intervals:
            try:
                t0 = float(item["t0_sec"])
                t1 = float(item["t1_sec"])
            except Exception:
                continue
            color = self._interval_color(item.get("label", ""))
            if color is None:
                continue
            if str(item.get("label", "")).strip().lower() == "run":
                has_run = True
            if str(item.get("label", "")).strip().lower() == "rest":
                has_rest = True
            span = self.ax.axvspan(
                min(t0, t1),
                max(t0, t1),
                facecolor=color,
                edgecolor="none",
                alpha=self.INTERVAL_ALPHA,
                zorder=0.5,
            )
            self._interval_spans.append(span)
        self._refresh_interval_legend(has_run=has_run, has_rest=has_rest)
        if self._cursor_line is not None and len(self.times):
            self._cursor_line.set_xdata([self.times[self.cursor_index], self.times[self.cursor_index]])
        self.draw_idle()

    def _event_to_cursor(self, event: Any) -> None:
        """Move the trace cursor to the position of a Matplotlib pointer event."""
        if event.inaxes != self.ax or event.xdata is None:
            return
        idx = self.nearest_index(float(event.xdata))
        self.set_cursor_index(idx, emit=True)

    def _on_press(self, event: Any) -> None:
        """Handle the start of an interactive trace-cursor drag."""
        if event.button == 1:
            self._mouse_is_down = True
            self._event_to_cursor(event)

    def _on_release(self, event: Any) -> None:
        """Handle the end of an interactive trace-cursor drag."""
        if event.button == 1:
            self._mouse_is_down = False

    def _on_motion(self, event: Any) -> None:
        """Update the cursor during an interactive trace drag."""
        if self._mouse_is_down:
            self._event_to_cursor(event)


class VideoReader:
    """Read individual frames and metadata from a behavior video with OpenCV."""
    def __init__(self) -> None:
        """Initialize this GUI object and its display state."""
        self.path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.fps: float = float("nan")
        self.n_frames: int = 0
        self.width: int = 0
        self.height: int = 0

    def close(self) -> None:
        """Release the current OpenCV video capture handle."""
        if self.cap is not None:
            self.cap.release()
        self.path = None
        self.cap = None
        self.fps = float("nan")
        self.n_frames = 0
        self.width = 0
        self.height = 0

    def open(self, path: str | Path) -> None:
        """Open and inspect a user-selected video file."""
        self.close()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Video not found: {p}")
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {p}")
        self.path = p
        self.cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or float("nan"))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def read_frame(self, index: int) -> np.ndarray | None:
        """Read an indexed video frame and return it in RGB order."""
        if self.cap is None or self.n_frames <= 0:
            return None
        idx = int(np.clip(index, 0, self.n_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------------------------------
# Main GUI
# -----------------------------------------------------------------------------

class BehaviorReviewWindow(QtWidgets.QMainWindow):
    """Coordinate Step4 signal viewing, video alignment, and behavior annotations."""
    def __init__(
        self,
        *,
        config_path: str | Path,
        filters: dict[str, Any] | None = None,
        groups: Sequence[str] | None = None,
        stems: Sequence[str] | None = None,
        include_bad: bool | None = None,
        preview_n_frames: int = 200,
        raw_preview_wavelength: str | None = "505",
        signal_map_contrast_mode: str = "frame",
        signal_map_percentiles: tuple[float, float] = (2.0, 98.0),
        global_contrast_time_chunk: int = 50,
        global_contrast_max_values: int = 2_000_000,
        review_mode: str = "all",
        trace_bandpass_enabled: bool = False,
        trace_gaussian_sigma_frames: float = 0.0,
        map_smoothing_sigma_px: float = 0.0,
    ) -> None:
        """Initialize the video-and-signal annotation interface.
        
        Parameters
        ----------
        config_path : str or pathlib.Path
            Path to the complete project YAML configuration.
        filters : dict or None
            Metadata filter override passed to ``build_step4_index``.
        groups, stems : sequence of str or None
            Optional subsets of Step4 recordings.
        include_bad : bool or None
            Whether to include Step2 rows flagged as bad.
        preview_n_frames : int
            Raw frames averaged into the fallback display image.
        raw_preview_wavelength : str or None
            Preferred raw preview wavelength label (typically nm).
        signal_map_contrast_mode : {'frame', 'global'}
            Compute map limits per frame or across the selected recording.
        signal_map_percentiles : tuple[float, float]
            Contrast percentiles; display-only.
        global_contrast_time_chunk, global_contrast_max_values : int
            Memory-control limits for global display contrast.
        review_mode : str
            Existing-annotation selection mode: all, pending, unreviewed,
            or unannotated.
        trace_bandpass_enabled : bool
            Enable the optional display-only 0.08-0.40 Hz trace filter.
        trace_gaussian_sigma_frames : float
            Temporal Gaussian display sigma, in frames.
        map_smoothing_sigma_px : float
            Spatial Gaussian display sigma, in pixels.
        
        Notes
        -----
        Saves behavior annotations to ``out/behavior_review/`` beneath the
        configured project output. No processed Zarr is rewritten."""
        super().__init__()
        self.config_path = Path(config_path)
        self.preview_n_frames = int(preview_n_frames)
        self.raw_preview_wavelength = raw_preview_wavelength
        contrast_mode = str(signal_map_contrast_mode or "frame").strip().lower()
        if contrast_mode not in {"frame", "global"}:
            raise ValueError("signal_map_contrast_mode must be 'frame' or 'global'")
        self.signal_map_contrast_mode = contrast_mode
        p_low, p_high = map(float, signal_map_percentiles)
        if not (0 <= p_low < p_high <= 100):
            raise ValueError("signal_map_percentiles must satisfy 0 <= LOW < HIGH <= 100")
        self.signal_map_percentiles = (p_low, p_high)
        self.global_contrast_time_chunk = max(1, int(global_contrast_time_chunk))
        self.global_contrast_max_values = int(global_contrast_max_values)
        if not np.isfinite(map_smoothing_sigma_px) or map_smoothing_sigma_px < 0:
            raise ValueError("map_smoothing_sigma_px must be finite and nonnegative")
        if not np.isfinite(trace_gaussian_sigma_frames) or trace_gaussian_sigma_frames < 0:
            raise ValueError("trace_gaussian_sigma_frames must be finite and nonnegative")
        if trace_bandpass_enabled and trace_gaussian_sigma_frames > 0:
            raise ValueError("Select either trace band-pass or temporal Gaussian, not both")
        self.trace_bandpass_enabled = bool(trace_bandpass_enabled)
        self.trace_gaussian_sigma_frames = float(trace_gaussian_sigma_frames)
        self._requested_trace_sigma_frames = float(trace_gaussian_sigma_frames) if trace_gaussian_sigma_frames > 0 else 1.0
        self.map_smoothing_sigma_px = float(map_smoothing_sigma_px)  # active value, zero disables
        self._requested_map_sigma_px = float(map_smoothing_sigma_px) if map_smoothing_sigma_px > 0 else 1.0
        mode = str(review_mode or "all").strip().lower()
        aliases = {
            "all": "all",
            "all_stems": "all",
            "old": "all",
            "previous": "all",
            "unreviewed": "unreviewed",
            "missing": "unreviewed",
            "new": "unreviewed",
            "not_in_annotations": "unreviewed",
            "pending": "pending",
            "unannotated": "unannotated",
            "empty_intervals": "unannotated",
            "no_intervals": "unannotated",
        }
        if mode not in aliases:
            raise ValueError("review_mode must be 'all', 'unreviewed', 'pending', or 'unannotated'")
        self.review_mode = aliases[mode]
        self.reviewed_stems_at_start: set[str] = set()
        self.annotated_stems_at_start: set[str] = set()
        self.skipped_reviewed_n = 0

        self.df, self.config, self.paths = build_step4_index(
            self.config_path,
            include_bad=include_bad,
            filters=filters,
            groups=groups,
            stems=stems,
        )
        n_indexed = len(self.df)
        missing_rows = self.df.loc[~self.df["processed_zarr_exists"].astype(bool)] if "processed_zarr_exists" in self.df.columns else self.df.iloc[0:0]
        for _, miss in missing_rows.head(3).iterrows():
            print(f"[Behavior] [MISSING PROCESSED] stem={miss.get('stem')} path={miss.get('processed_zarr_path_resolved')}")
        if "processed_zarr_exists" in self.df.columns:
            self.df = self.df[self.df["processed_zarr_exists"].astype(bool)].reset_index(drop=True)
        self.n_indexed = n_indexed
        self.n_with_processed = len(self.df)
        print(f"[Behavior] Step4 indexed={n_indexed}, processed Zarr found={self.n_with_processed}, missing={n_indexed - self.n_with_processed}")

        self.behavior_root = Path(self.paths["out_root"]) / "behavior_review"
        self.behavior_root.mkdir(parents=True, exist_ok=True)
        self.video_map_csv = self.behavior_root / "behavior_video_map.csv"
        self.annotations_csv = self.behavior_root / "behavior_review_annotations.csv"
        self.roi_root = self.behavior_root / "roi_masks"
        self.roi_root.mkdir(parents=True, exist_ok=True)

        if self.review_mode == "unreviewed":
            n_before = len(self.df)
            self.reviewed_stems_at_start = read_stems_present_in_csv(self.annotations_csv)
            if self.reviewed_stems_at_start:
                self.df = self.df[~self.df["stem"].astype(str).isin(self.reviewed_stems_at_start)].reset_index(drop=True)
            self.skipped_reviewed_n = n_before - len(self.df)
            print(
                f"[INFO] review_mode=unreviewed | annotations CSV: {self.annotations_csv} | "
                f"stems in CSV={len(self.reviewed_stems_at_start)} | "
                f"selected after metadata filters={n_before} | skipped={self.skipped_reviewed_n} | "
                f"remaining={len(self.df)}"
            )
        elif self.review_mode == "pending":
            n_before = len(self.df)
            self.reviewed_stems_at_start = read_stems_explicitly_reviewed(self.annotations_csv)
            if self.reviewed_stems_at_start:
                self.df = self.df[~self.df["stem"].astype(str).isin(self.reviewed_stems_at_start)].reset_index(drop=True)
            self.skipped_reviewed_n = n_before - len(self.df)
            print(
                f"[INFO] review_mode=pending | annotations CSV: {self.annotations_csv} | "
                f"stems explicitly reviewed={len(self.reviewed_stems_at_start)} | "
                f"selected after metadata filters={n_before} | skipped reviewed={self.skipped_reviewed_n} | "
                f"remaining={len(self.df)}"
            )
        elif self.review_mode == "unannotated":
            n_before = len(self.df)
            self.annotated_stems_at_start = read_stems_with_nonempty_intervals(self.annotations_csv)
            if self.annotated_stems_at_start:
                self.df = self.df[~self.df["stem"].astype(str).isin(self.annotated_stems_at_start)].reset_index(drop=True)
            self.skipped_reviewed_n = n_before - len(self.df)
            print(
                f"[INFO] review_mode=unannotated | annotations CSV: {self.annotations_csv} | "
                f"stems with non-empty label_intervals_json={len(self.annotated_stems_at_start)} | "
                f"selected after metadata filters={n_before} | skipped already annotated={self.skipped_reviewed_n} | "
                f"remaining={len(self.df)}"
            )
        else:
            print(f"[INFO] review_mode=all; behavior annotations CSV: {self.annotations_csv}")

        if self.df.empty:
            if self.n_indexed == 0:
                raise RuntimeError("No stems matched Step4 metadata selection. Check --protocol, --filter and excluded_stems.")
            if self.n_with_processed == 0:
                raise FileNotFoundError(
                    "No selected Step4 processed Zarr found. Check path_resolving, "
                    "project out_root, Step4 processed output and protocol filters. "
                    "A behavior annotation CSV is not required for the first run."
                )
            if self.review_mode == "unreviewed":
                raise RuntimeError(
                    "No unreviewed Step4 rows were found. All selected stems are already present "
                    f"in behavior_review_annotations.csv: {self.annotations_csv}"
                )
            if self.review_mode == "pending":
                raise RuntimeError(
                    "No pending Step4 rows were found for the selected filters. All selected stems "
                    f"have latest review_status='reviewed' in: {self.annotations_csv}"
                )
            if self.review_mode == "unannotated":
                raise RuntimeError(
                    "No unannotated Step4 rows were found for the selected filters. Every selected stem already has "
                    f"a non-empty label_intervals_json in: {self.annotations_csv}"
                )
            raise RuntimeError("No Step4 rows with existing processed zarr were found for the selected filters.")

        self.current_index = 0
        self.current_row: pd.Series | None = None
        self.current_ds: xr.Dataset | None = None
        self.current_raw_image: np.ndarray | None = None
        self.current_brain_signal_mask: np.ndarray | None = None
        self.current_mask_sources: dict[str, str] = {}
        self._signal_map_cache: dict[tuple[str, int, str, float], tuple[np.ndarray, float, float]] = {}
        self._signal_map_last_key: tuple[str, int, str, float] | None = None
        self._global_contrast_limits: dict[str, tuple[float, float]] = {}
        self._global_contrast_notes: dict[str, str] = {}
        self.current_trace_times = np.array([], dtype=float)
        self.current_trace_values = np.array([], dtype=float)
        self.signal_choices: list[tuple[str, str]] = []
        self.video = VideoReader()
        self.video_state: VideoState | None = None
        self.label_intervals: list[dict[str, Any]] = []
        self.interval_start_sec: float | None = None
        self.last_video_dir = Path.home()
        self._updating_slider = False
        self._updating_trace = False
        self._loading = False

        self._build_ui()
        self.load_current_stem()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        """Create widgets, layouts, and signal connections for the interface."""
        self.setWindowTitle("Behavior review from Step4 v5.3")
        self.resize(1400, 900)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        top = QtWidgets.QHBoxLayout()
        main_layout.addLayout(top)

        self.prev_btn = QtWidgets.QPushButton("◀ Prev stem")
        self.next_btn = QtWidgets.QPushButton("Next stem ▶")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.choose_video_btn = QtWidgets.QPushButton("Choose video")
        self.clear_video_btn = QtWidgets.QPushButton("Clear video")
        self.no_video_btn = QtWidgets.QPushButton("No video")
        self.default_roi_btn = QtWidgets.QPushButton("Default ROI")
        self.clear_roi_btn = QtWidgets.QPushButton("Clear ROI")

        self.signal_combo = QtWidgets.QComboBox()
        self.stem_label = QtWidgets.QLabel()
        self.stem_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        top.addWidget(self.prev_btn)
        top.addWidget(self.next_btn)
        top.addSpacing(10)
        top.addWidget(QtWidgets.QLabel("Signal:"))
        top.addWidget(self.signal_combo, stretch=0)
        top.addSpacing(10)
        top.addWidget(self.choose_video_btn)
        top.addWidget(self.clear_video_btn)
        top.addWidget(self.no_video_btn)
        top.addSpacing(10)
        top.addWidget(self.default_roi_btn)
        top.addWidget(self.clear_roi_btn)
        top.addSpacing(10)
        top.addWidget(self.save_btn)
        top.addWidget(self.stem_label, stretch=1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.video_widget = VideoWidget()
        left_layout.addWidget(QtWidgets.QLabel("Behavior video"))
        left_layout.addWidget(self.video_widget, stretch=1)

        video_ctrl = QtWidgets.QHBoxLayout()
        self.video_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(0)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.video_info_label = QtWidgets.QLabel("No video")
        video_ctrl.addWidget(self.video_slider, stretch=1)
        video_ctrl.addWidget(QtWidgets.QLabel("frame"))
        video_ctrl.addWidget(self.frame_spin)
        left_layout.addLayout(video_ctrl)
        left_layout.addWidget(self.video_info_label)

        clip_ctrl = QtWidgets.QHBoxLayout()
        self.set_start_btn = QtWidgets.QPushButton("Set video start")
        self.set_stop_btn = QtWidgets.QPushButton("Set video stop")
        self.video_start_label = QtWidgets.QLabel("start: -")
        self.video_stop_label = QtWidgets.QLabel("stop: -")
        clip_ctrl.addWidget(self.set_start_btn)
        clip_ctrl.addWidget(self.set_stop_btn)
        clip_ctrl.addWidget(self.video_start_label)
        clip_ctrl.addWidget(self.video_stop_label)
        left_layout.addLayout(clip_ctrl)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.addWidget(QtWidgets.QLabel(
            f"Signal map at trace cursor, masked as in Step5; contrast={self.signal_map_contrast_mode}; draw rectangular ROI with mouse"
        ))
        self.roi_widget = RoiImageWidget()
        right_layout.addWidget(self.roi_widget, stretch=1)
        self.roi_info_label = QtWidgets.QLabel("ROI: -")
        right_layout.addWidget(self.roi_info_label)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([700, 700])

        bottom = QtWidgets.QVBoxLayout()
        main_layout.addLayout(bottom)
        self.trace_canvas = TraceCanvas()
        bottom.addWidget(self.trace_canvas, stretch=0)

        display_controls = QtWidgets.QHBoxLayout()
        self.trace_filter_checkbox = QtWidgets.QCheckBox("Trace band-pass 0.08–0.40 Hz (display only)")
        self.trace_filter_checkbox.setChecked(self.trace_bandpass_enabled)
        self.trace_gaussian_checkbox = QtWidgets.QCheckBox(
            f"Trace Gaussian σ={self._requested_trace_sigma_frames:g} frame(s) (display only)"
        )
        self.trace_gaussian_checkbox.setChecked(self.trace_gaussian_sigma_frames > 0)
        self.map_smoothing_checkbox = QtWidgets.QCheckBox(f"Map Gaussian σ={self._requested_map_sigma_px:g} px (display only)")
        self.map_smoothing_checkbox.setChecked(self.map_smoothing_sigma_px > 0)
        display_controls.addWidget(self.trace_filter_checkbox)
        display_controls.addWidget(self.trace_gaussian_checkbox)
        display_controls.addWidget(self.map_smoothing_checkbox)
        display_controls.addStretch(1)
        bottom.addLayout(display_controls)

        trace_zoom_ctrl = QtWidgets.QHBoxLayout()
        self.x_zoom_in_btn = QtWidgets.QPushButton("X zoom +")
        self.x_zoom_out_btn = QtWidgets.QPushButton("X zoom -")
        self.y_zoom_in_btn = QtWidgets.QPushButton("Y zoom +")
        self.y_zoom_out_btn = QtWidgets.QPushButton("Y zoom -")
        self.trace_reset_btn = QtWidgets.QPushButton("Reset graph")
        trace_zoom_ctrl.addWidget(QtWidgets.QLabel("Trace scale:"))
        trace_zoom_ctrl.addWidget(self.x_zoom_in_btn)
        trace_zoom_ctrl.addWidget(self.x_zoom_out_btn)
        trace_zoom_ctrl.addWidget(self.y_zoom_in_btn)
        trace_zoom_ctrl.addWidget(self.y_zoom_out_btn)
        trace_zoom_ctrl.addWidget(self.trace_reset_btn)
        trace_zoom_ctrl.addStretch(1)
        bottom.addLayout(trace_zoom_ctrl)

        interval_ctrl = QtWidgets.QHBoxLayout()
        self.interval_start_btn = QtWidgets.QPushButton("Set interval start at cursor")
        self.add_run_btn = QtWidgets.QPushButton("Set RUN interval")
        self.add_rest_btn = QtWidgets.QPushButton("Set REST interval")
        self.clear_intervals_btn = QtWidgets.QPushButton("Clear intervals")
        self.interval_info_label = QtWidgets.QLabel("Intervals: 0")
        interval_ctrl.addWidget(self.interval_start_btn)
        interval_ctrl.addWidget(self.add_run_btn)
        interval_ctrl.addWidget(self.add_rest_btn)
        interval_ctrl.addWidget(self.clear_intervals_btn)
        interval_ctrl.addWidget(self.interval_info_label, stretch=1)
        bottom.addLayout(interval_ctrl)

        # Connections
        self.prev_btn.clicked.connect(self.prev_stem)
        self.next_btn.clicked.connect(self.next_stem)
        self.save_btn.clicked.connect(lambda: self.save_current_state(explicit=True))
        self.choose_video_btn.clicked.connect(self.choose_video)
        self.clear_video_btn.clicked.connect(self.clear_video)
        self.no_video_btn.clicked.connect(self.mark_no_video)
        self.default_roi_btn.clicked.connect(self.set_default_roi_and_trace)
        self.clear_roi_btn.clicked.connect(self.clear_roi_and_trace)
        self.set_start_btn.clicked.connect(self.set_video_start)
        self.set_stop_btn.clicked.connect(self.set_video_stop)
        self.video_slider.valueChanged.connect(self.on_video_slider_changed)
        self.frame_spin.valueChanged.connect(self.on_frame_spin_changed)
        self.signal_combo.currentIndexChanged.connect(self.on_signal_changed)
        self.trace_filter_checkbox.toggled.connect(self.on_trace_filter_toggled)
        self.trace_gaussian_checkbox.toggled.connect(self.on_trace_gaussian_toggled)
        self.map_smoothing_checkbox.toggled.connect(self.on_map_smoothing_toggled)
        self.roi_widget.roiChanged.connect(self.on_roi_changed)
        self.trace_canvas.cursorChanged.connect(self.on_trace_cursor_changed)
        self.x_zoom_in_btn.clicked.connect(lambda: self.trace_canvas.zoom_x(0.5))
        self.x_zoom_out_btn.clicked.connect(lambda: self.trace_canvas.zoom_x(2.0))
        self.y_zoom_in_btn.clicked.connect(lambda: self.trace_canvas.zoom_y(0.5))
        self.y_zoom_out_btn.clicked.connect(lambda: self.trace_canvas.zoom_y(2.0))
        self.trace_reset_btn.clicked.connect(self.trace_canvas.reset_zoom)
        self.interval_start_btn.clicked.connect(self.set_interval_start)
        self.add_run_btn.clicked.connect(lambda: self.add_interval("run"))
        self.add_rest_btn.clicked.connect(lambda: self.add_interval("rest"))
        self.clear_intervals_btn.clicked.connect(self.clear_intervals)

        self.statusBar().showMessage("Ready")

    # ---------------- Loading ----------------

    def load_current_stem(self) -> None:
        """Load the selected stem and restore its saved review state."""
        self._loading = True
        try:
            self._close_current_ds()
            self.video.close()
            self.video_widget.set_frame(None, "No video loaded")
            self.current_row = self.df.iloc[self.current_index]
            row = self.current_row
            stem = str(row["stem"])
            mode_txt = ""
            if self.review_mode == "unreviewed":
                mode_txt = f" | unreviewed only, skipped already saved={self.skipped_reviewed_n}"
            elif self.review_mode == "pending":
                mode_txt = f" | pending only, skipped confirmed reviewed={self.skipped_reviewed_n}"
            elif self.review_mode == "unannotated":
                mode_txt = f" | unannotated only, skipped non-empty intervals={self.skipped_reviewed_n}"
            self.stem_label.setText(
                f"{self.current_index + 1}/{len(self.df)} | stem: {stem} | "
                f"group: {safe_str(row.get('group_id'))} | "
                f"{safe_str(row.get('exp_day'))} / {safe_str(row.get('state'))} / {safe_str(row.get('protocol'))}"
                f"{mode_txt}"
            )
            self.statusBar().showMessage(f"Loading {stem} ...")
            QtWidgets.QApplication.processEvents()

            self.current_brain_signal_mask = None
            self.current_mask_sources = {}
            self._signal_map_cache = {}
            self._signal_map_last_key = None
            self._global_contrast_limits = {}
            self._global_contrast_notes = {}
            self.load_raw_preview(row)
            self.load_processed_dataset(row)
            self.load_signal_choices()
            self.load_saved_roi_and_annotations(stem)
            # Load mask after old annotations are read, because a saved
            # signal_name can restore a different selected signal.
            self.load_display_mask_for_current_signal_map(row)
            self.load_saved_video_state(stem)
            self.compute_and_plot_trace()
            self.sync_video_to_trace_cursor()
            self.update_signal_map_at_cursor(force=True)
            self.statusBar().showMessage(f"Loaded {stem}")
        except Exception as e:
            self.show_error("Could not load stem", e)
        finally:
            self._loading = False

    def _close_current_ds(self) -> None:
        """Close the currently open processed Zarr dataset."""
        if self.current_ds is not None:
            try:
                self.current_ds.close()
            except Exception:
                pass
        self.current_ds = None

    def load_raw_preview(self, row: pd.Series) -> None:
        """Average initial raw frames to generate the GUI background image."""
        raw_path = select_raw_zarr_path_for_processing(row, self.config)
        raw = open_raw_zarr(raw_path, self.config)
        wave = self.raw_preview_wavelength or cfg_get(self.config, "raw", "reference_wave", default="505")
        widx = select_wavelength_index(raw, wave)
        n = min(max(int(self.preview_n_frames), 1), int(raw.sizes["time"]))
        img = raw.isel(wavelength=widx, time=slice(0, n)).mean("time").compute().values
        self.current_raw_image = np.asarray(img)
        self.roi_widget.set_image(self.current_raw_image)

    def load_processed_dataset(self, row: pd.Series) -> None:
        """Open the selected Step4 processed Zarr dataset for viewing."""
        p = Path(str(row["processed_zarr_path_resolved"]))
        if not p.exists():
            raise FileNotFoundError(f"Processed zarr not found: {p}")
        try:
            self.current_ds = xr.open_zarr(p, chunks="auto", consolidated=False, mask_and_scale=False)
        except Exception:
            self.current_ds = xr.open_zarr(p, chunks=None, consolidated=False, mask_and_scale=False)

    def load_display_mask_for_current_signal_map(self, row: pd.Series) -> None:
        """Load the canonical display mask for a spatial processed signal.
        
        Parameters
        ----------
        row : pandas.Series
            Metadata row for the current recording.
        
        Notes
        -----
        For a (time, y, x) signal, this function requires ``raw_valid`` in the
        processed Zarr and reads the corresponding canonical Step2 masks. If mask
        loading fails, it clears cached mask state and re-raises the exception;
        it does not silently fall back to an unmasked trace."""
        self.current_brain_signal_mask = None
        self.current_mask_sources = {}
        if self.current_ds is None:
            return
        signal_name = self.selected_signal_name()
        if not signal_name or signal_name not in self.current_ds:
            return
        var = self.current_ds[signal_name]
        if set(var.dims) != {"time", "y", "x"}:
            return
        image_shape = (int(var.sizes["y"]), int(var.sizes["x"]))
        try:
            if "raw_valid" not in self.current_ds:
                raise KeyError("Step4 processed Zarr has no raw_valid; rebuild with current Step4")
            rv = np.asarray(self.current_ds["raw_valid"].transpose("y", "x").values)
            mask, sources = load_step5_like_brain_signal_mask(
                row, self.config, self.paths, image_shape, raw_valid=(rv == 1)
            )
            self.current_brain_signal_mask = mask
            self.current_mask_sources = sources
        except Exception:
            self.current_brain_signal_mask = None
            self.current_mask_sources = {}
            raise

    def load_signal_choices(self) -> None:
        """Populate the list of available HbT-like signal variables."""
        if self.current_ds is None:
            return
        previous_var = self.selected_signal_name()
        self.signal_choices = candidate_hbt_signals(self.current_ds)
        self.signal_combo.blockSignals(True)
        self.signal_combo.clear()
        if not self.signal_choices:
            self.signal_combo.addItem("No HbT/HbTvreg variable found", "")
        else:
            for label, varname in self.signal_choices:
                self.signal_combo.addItem(label, varname)
            if previous_var:
                for i, (_, varname) in enumerate(self.signal_choices):
                    if varname == previous_var:
                        self.signal_combo.setCurrentIndex(i)
                        break
        self.signal_combo.blockSignals(False)

    def load_saved_roi_and_annotations(self, stem: str) -> None:
        """Restore a saved rectangle and behavior intervals for the current stem."""
        self.label_intervals = []
        self.interval_start_sec = None
        row = read_latest_row(self.annotations_csv, stem)
        if row is None:
            self.roi_widget.set_default_roi()
            self.update_roi_info()
            self.trace_canvas.add_interval_spans([])
            return

        # Restore signal choice if possible.
        saved_signal = safe_str(row.get("signal_name"))
        if saved_signal:
            for i in range(self.signal_combo.count()):
                if str(self.signal_combo.itemData(i)) == saved_signal:
                    self.signal_combo.setCurrentIndex(i)
                    break

        try:
            y0 = int(float(row.get("roi_y0")))
            x0 = int(float(row.get("roi_x0")))
            y1 = int(float(row.get("roi_y1")))
            x1 = int(float(row.get("roi_x1")))
            if y0 >= 0 and x0 >= 0 and y1 >= y0 and x1 >= x0:
                self.roi_widget.set_roi_yxyx(y0, x0, y1, x1)
            else:
                self.roi_widget.set_default_roi()
        except Exception:
            self.roi_widget.set_default_roi()

        try:
            intervals = json.loads(safe_str(row.get("label_intervals_json")) or "[]")
            if isinstance(intervals, list):
                self.label_intervals = intervals
        except Exception:
            self.label_intervals = []
        self.update_interval_info()
        self.update_roi_info()

    def load_saved_video_state(self, stem: str) -> None:
        """Restore a previously saved behavior-video mapping for the current stem."""
        row0 = self.current_row
        group_id = safe_str(row0.get("group_id")) if row0 is not None else ""
        saved = read_latest_row(self.video_map_csv, stem)
        self.video_state = VideoState(stem=stem, group_id=group_id, updated_at=now_iso())
        if saved is None:
            self.update_video_controls_no_video("Choose video for this stem")
            return

        path = safe_str(saved.get("behavior_video_path"))
        video_exists_flag = str(saved.get("video_exists", "")).strip().lower() in {"true", "1", "yes", "y"}
        if path:
            self.last_video_dir = Path(path).parent
        if video_exists_flag and path and Path(path).exists():
            try:
                self.open_video_path(path)
                start = int(float(saved.get("video_start_frame", 0)))
                stop = int(float(saved.get("video_stop_frame", max(self.video.n_frames - 1, 0))))
                self.video_state.video_start_frame = int(np.clip(start, 0, max(self.video.n_frames - 1, 0)))
                self.video_state.video_stop_frame = int(np.clip(stop, 0, max(self.video.n_frames - 1, 0)))
                if self.video_state.video_stop_frame < self.video_state.video_start_frame:
                    self.video_state.video_start_frame, self.video_state.video_stop_frame = (
                        self.video_state.video_stop_frame,
                        self.video_state.video_start_frame,
                    )
                self.update_video_start_stop_labels()
                return
            except Exception as e:
                self.statusBar().showMessage(f"Saved video could not be loaded: {e}")

        self.video_state.video_exists = False
        self.video_state.behavior_video_path = path
        self.update_video_controls_no_video("No valid video for this stem")

    # ---------------- Video ----------------

    def open_video_path(self, path: str | Path) -> None:
        """Open the selected video and refresh associated controls."""
        self.video.open(path)
        stem = str(self.current_row["stem"]) if self.current_row is not None else ""
        group_id = safe_str(self.current_row.get("group_id")) if self.current_row is not None else ""
        self.video_state = VideoState(
            stem=stem,
            group_id=group_id,
            behavior_video_path=str(Path(path)),
            video_exists=True,
            video_fps=self.video.fps,
            video_n_frames=self.video.n_frames,
            video_width=self.video.width,
            video_height=self.video.height,
            video_start_frame=0,
            video_stop_frame=max(self.video.n_frames - 1, 0),
            updated_at=now_iso(),
        )
        self.video_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(max(self.video.n_frames - 1, 0))
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(max(self.video.n_frames - 1, 0))
        self.video_slider.setValue(0)
        self.frame_spin.setValue(0)
        self.video_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.update_video_info()
        self.update_video_start_stop_labels()
        self.show_video_frame(0)

    def update_video_controls_no_video(self, text: str = "No video loaded") -> None:
        """Refresh controls when no behavior video is currently loaded."""
        self.video.close()
        self.video_widget.set_frame(None, text)
        self.video_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(0)
        self.video_slider.setValue(0)
        self.frame_spin.setMinimum(0)
        self.frame_spin.setMaximum(0)
        self.frame_spin.setValue(0)
        self.video_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.video_info_label.setText(text)
        self.video_start_label.setText("start: -")
        self.video_stop_label.setText("stop: -")

    def update_video_info(self) -> None:
        """Show the selected video filename and metadata."""
        if self.video.cap is None or self.video.path is None:
            self.video_info_label.setText("No video")
            return
        self.video_info_label.setText(
            f"{self.video.path.name} | fps={self.video.fps:.3g} | "
            f"frames={self.video.n_frames} | size={self.video.width}x{self.video.height}"
        )

    def update_video_start_stop_labels(self) -> None:
        """Refresh the displayed video frame-boundary labels."""
        if self.video_state is None or not self.video_state.video_exists:
            self.video_start_label.setText("start: -")
            self.video_stop_label.setText("stop: -")
            return
        self.video_start_label.setText(f"start: {self.video_state.video_start_frame}")
        self.video_stop_label.setText(f"stop: {self.video_state.video_stop_frame}")

    def choose_video(self) -> None:
        """Prompt the user to select a video file for the current stem."""
        start_dir = str(self.last_video_dir if self.last_video_dir.exists() else Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose behavior video for this stem",
            start_dir,
            "Video files (*.avi *.mp4 *.mov *.mkv *.wmv *.mpg *.mpeg *.m4v);;All files (*.*)",
        )
        if not path:
            return
        try:
            self.last_video_dir = Path(path).parent
            self.open_video_path(path)
            self.save_video_map_only()
            self.sync_video_to_trace_cursor()
            self.statusBar().showMessage(f"Video selected: {path}")
        except Exception as e:
            self.show_error("Could not open selected video", e)

    def clear_video(self) -> None:
        """Clear the selected behavior video mapping."""
        if self.current_row is None:
            return
        stem = str(self.current_row["stem"])
        group_id = safe_str(self.current_row.get("group_id"))
        self.video_state = VideoState(stem=stem, group_id=group_id, video_exists=False, updated_at=now_iso())
        self.update_video_controls_no_video("Video cleared")
        self.save_video_map_only()

    def mark_no_video(self) -> None:
        """Mark the current stem as having no behavior video."""
        if self.current_row is None:
            return
        stem = str(self.current_row["stem"])
        group_id = safe_str(self.current_row.get("group_id"))
        self.video_state = VideoState(stem=stem, group_id=group_id, video_exists=False, updated_at=now_iso())
        self.update_video_controls_no_video("Marked: no video for this stem")
        self.save_video_map_only()

    def show_video_frame(self, frame_idx: int) -> None:
        """Display the requested video frame."""
        if self.video.cap is None:
            return
        frame_idx = int(np.clip(frame_idx, 0, max(self.video.n_frames - 1, 0)))
        frame = self.video.read_frame(frame_idx)
        self.video_widget.set_frame(frame, text="Could not read video frame" if frame is None else None)
        if not self._updating_slider:
            self._updating_slider = True
            self.video_slider.setValue(frame_idx)
            self.frame_spin.setValue(frame_idx)
            self._updating_slider = False

    def on_video_slider_changed(self, value: int) -> None:
        """Update the video display after moving the frame slider."""
        if self._updating_slider or self._loading:
            return
        self._updating_slider = True
        self.frame_spin.setValue(int(value))
        self._updating_slider = False
        self.show_video_frame(int(value))
        self.sync_trace_to_video_frame(int(value))

    def on_frame_spin_changed(self, value: int) -> None:
        """Update the video display after editing the frame selector."""
        if self._updating_slider or self._loading:
            return
        self._updating_slider = True
        self.video_slider.setValue(int(value))
        self._updating_slider = False
        self.show_video_frame(int(value))
        self.sync_trace_to_video_frame(int(value))

    def set_video_start(self) -> None:
        """Set the synchronization start frame in the behavior video."""
        if self.video_state is None or not self.video_state.video_exists:
            return
        f = int(self.video_slider.value())
        self.video_state.video_start_frame = f
        if self.video_state.video_stop_frame < f:
            self.video_state.video_stop_frame = f
        self.update_video_start_stop_labels()
        self.save_video_map_only()

    def set_video_stop(self) -> None:
        """Set the synchronization stop frame in the behavior video."""
        if self.video_state is None or not self.video_state.video_exists:
            return
        f = int(self.video_slider.value())
        self.video_state.video_stop_frame = f
        if self.video_state.video_start_frame > f:
            self.video_state.video_start_frame = f
        self.update_video_start_stop_labels()
        self.save_video_map_only()

    # ---------------- Trace and ROI ----------------

    def selected_signal_name(self) -> str:
        """Return the selected processed HbT-like variable name."""
        idx = self.signal_combo.currentIndex() if hasattr(self, "signal_combo") else -1
        if idx < 0:
            return ""
        value = self.signal_combo.itemData(idx)
        return str(value or "")

    def on_signal_changed(self) -> None:
        """Update signal maps and ROI traces when the selected signal changes."""
        if self._loading:
            return
        if self.current_row is not None:
            self._signal_map_cache = {}
            self._signal_map_last_key = None
            self._global_contrast_limits = {}
            self._global_contrast_notes = {}
            self.load_display_mask_for_current_signal_map(self.current_row)
        self.compute_and_plot_trace()
        self.sync_video_to_trace_cursor()
        self.update_signal_map_at_cursor(force=True)

    def on_roi_changed(self) -> None:
        """Recompute visualization when the rectangular ROI is edited."""
        if self._loading:
            return
        self.update_roi_info()
        self.compute_and_plot_trace()
        self.sync_video_to_trace_cursor()
        self.update_signal_map_at_cursor(force=True)

    def set_default_roi_and_trace(self) -> None:
        """Reset the ROI to its default and refresh the trace."""
        self.roi_widget.set_default_roi()
        self.compute_and_plot_trace()

    def clear_roi_and_trace(self) -> None:
        """Clear the ROI and refresh the displayed time series."""
        self.roi_widget.clear_roi()
        self.compute_and_plot_trace()

    def update_roi_info(self) -> None:
        """Display the current ROI bounds and pixel area."""
        roi = self.roi_widget.roi_yxyx()
        if roi is None:
            self.roi_info_label.setText("ROI: not set")
        else:
            y0, x0, y1, x1 = roi
            area = max(0, y1 - y0 + 1) * max(0, x1 - x0 + 1)
            self.roi_info_label.setText(f"ROI: y={y0}:{y1}, x={x0}:{x1}, area={area} px")

    def compute_and_plot_trace(self) -> None:
        """Extract and display the selected ROI time course.
        
        Notes
        -----
        For (time, y, x) signals, the rectangular ROI is intersected with the
        brain-signal validity mask and spatially averaged over finite pixels.
        A one-dimensional (time,) input is displayed directly. Optional temporal
        Gaussian or band-pass operations are display-only. Time values come from
        the processed signal coordinate, normally expressed in seconds."""
        row = self.current_row
        ds = self.current_ds
        signal_name = self.selected_signal_name()
        if row is None or ds is None or not signal_name or signal_name not in ds:
            self.current_trace_times = np.array([], dtype=float)
            self.current_trace_values = np.array([], dtype=float)
            self.trace_canvas.set_trace(self.current_trace_times, self.current_trace_values, title="No signal", signal_name=signal_name)
            return

        try:
            var = ds[signal_name]
            if set(var.dims) == {"time", "y", "x"}:
                mask = self.roi_widget.roi_mask()
                if mask is None or int(mask.sum()) == 0:
                    self.current_trace_times = np.array([], dtype=float)
                    self.current_trace_values = np.array([], dtype=float)
                    self.trace_canvas.set_trace(self.current_trace_times, self.current_trace_values, title="Draw ROI to extract trace", signal_name=signal_name)
                    return
                target_shape = (int(var.sizes["y"]), int(var.sizes["x"]))
                mask = resize_bool_mask(mask, target_shape)
                if self.current_brain_signal_mask is None:
                    raise RuntimeError("Processed brainvessels/raw_valid display mask is unavailable")
                mask &= self.current_brain_signal_mask
                if not np.any(mask):
                    raise ValueError("Behavior ROI has no valid signal pixels")
                coords: dict[str, Any] = {}
                if "y" in var.coords and len(var.coords["y"]) == target_shape[0]:
                    coords["y"] = var.coords["y"].values
                if "x" in var.coords and len(var.coords["x"]) == target_shape[1]:
                    coords["x"] = var.coords["x"].values
                mask_da = xr.DataArray(mask, dims=("y", "x"), coords=coords)
                trace = var.where(mask_da).mean(dim=("y", "x"), skipna=True).compute()
                values = np.asarray(trace.values, dtype=float)
                times = np.asarray(trace["time"].values, dtype=float) if "time" in trace.coords else np.arange(len(values), dtype=float)
            elif list(var.dims) == ["time"] or var.dims == ("time",):
                trace = var.compute()
                values = np.asarray(trace.values, dtype=float)
                times = np.asarray(trace["time"].values, dtype=float) if "time" in trace.coords else np.arange(len(values), dtype=float)
            else:
                raise ValueError(f"Signal {signal_name!r} must have dims time,y,x or time. Got dims={var.dims}")

            display_values = values
            filter_note = ""
            if self.trace_gaussian_sigma_frames > 0:
                display_values = gaussian_temporal_display_trace(values, self.trace_gaussian_sigma_frames)
                filter_note = f" | display temporal Gaussian σ={self.trace_gaussian_sigma_frames:g} frames"
            elif self.trace_bandpass_enabled:
                fps = float(cfg_get(self.config, "raw", "fps", default=float("nan")))
                try:
                    display_values = bandpass_display_trace(values, fps)
                    filter_note = " | display band-pass 0.08–0.40 Hz"
                except ValueError as exc:
                    # Do not silently show filtered-looking invalid data.
                    filter_note = " | unfiltered (band-pass unavailable)"
                    self.statusBar().showMessage(f"Display band-pass unavailable: {exc}")
            self.current_trace_times = times  # annotation and video cursor remain unchanged
            self.current_trace_values = display_values
            title = f"{row['stem']} | {signal_name} | ROI mean{filter_note}"
            self.trace_canvas.set_trace(times, display_values, title=title, signal_name=signal_name)
            self.trace_canvas.add_interval_spans(self.label_intervals)
            self.update_signal_map_at_cursor(force=True)
        except Exception as e:
            self.current_trace_times = np.array([], dtype=float)
            self.current_trace_values = np.array([], dtype=float)
            self.trace_canvas.set_trace(self.current_trace_times, self.current_trace_values, title=f"Trace error: {e}", signal_name=signal_name)
            self.statusBar().showMessage(f"Trace error: {e}")

    def on_trace_filter_toggled(self, enabled: bool) -> None:
        """Enable or disable display-only trace band-pass filtering."""
        self.trace_bandpass_enabled = bool(enabled)
        if enabled and self.trace_gaussian_sigma_frames > 0:
            self.trace_gaussian_sigma_frames = 0.0
            self.trace_gaussian_checkbox.blockSignals(True)
            self.trace_gaussian_checkbox.setChecked(False)
            self.trace_gaussian_checkbox.blockSignals(False)
        if not self._loading:
            self.compute_and_plot_trace()

    def on_trace_gaussian_toggled(self, enabled: bool) -> None:
        """Enable or disable display-only temporal Gaussian smoothing."""
        self.trace_gaussian_sigma_frames = self._requested_trace_sigma_frames if enabled else 0.0
        if enabled and self.trace_bandpass_enabled:
            self.trace_bandpass_enabled = False
            self.trace_filter_checkbox.blockSignals(True)
            self.trace_filter_checkbox.setChecked(False)
            self.trace_filter_checkbox.blockSignals(False)
        if not self._loading:
            self.compute_and_plot_trace()

    def on_map_smoothing_toggled(self, enabled: bool) -> None:
        """Enable or disable display-only spatial Gaussian smoothing."""
        self.map_smoothing_sigma_px = self._requested_map_sigma_px if enabled else 0.0
        self._signal_map_cache.clear()
        self._signal_map_last_key = None
        if not self._loading:
            self.update_signal_map_at_cursor(force=True)

    def global_contrast_limits_for_signal(self, signal_name: str, var: xr.DataArray) -> tuple[float, float] | None:
        """Return global contrast limits for the current stem/signal if requested."""
        if self.signal_map_contrast_mode != "global":
            return None
        if signal_name in self._global_contrast_limits:
            return self._global_contrast_limits[signal_name]

        self.statusBar().showMessage(
            f"Computing GLOBAL contrast for {signal_name}: "
            f"p{self.signal_map_percentiles[0]:g}-p{self.signal_map_percentiles[1]:g} inside Step5-like brain mask ..."
        )
        QtWidgets.QApplication.processEvents()

        limits, note = compute_global_signal_contrast_limits(
            var,
            self.current_brain_signal_mask,
            percentile_range=self.signal_map_percentiles,
            time_chunk=self.global_contrast_time_chunk,
            max_values=self.global_contrast_max_values,
        )
        self._global_contrast_limits[signal_name] = limits
        self._global_contrast_notes[signal_name] = note
        self.statusBar().showMessage(
            f"GLOBAL contrast ready for {signal_name}: "
            f"[{limits[0]:.4g}, {limits[1]:.4g}] | {note}"
        )
        QtWidgets.QApplication.processEvents()
        return limits

    def update_signal_map_at_cursor(self, *, force: bool = False) -> None:
        """Show selected signal map at the current trace cursor.

        The display is masked with brain_signal exactly like Step5
        (brain & ~bad & ~bregma & raw_valid) when masks are available. This affects only
        visualization; ROI coordinates and saved interval JSON remain unchanged.
        """
        row = self.current_row
        ds = self.current_ds
        signal_name = self.selected_signal_name()
        if row is None or ds is None or not signal_name or signal_name not in ds:
            return
        var = ds[signal_name]
        if set(var.dims) != {"time", "y", "x"}:
            # For 1D variables there is no image map. Keep whatever fallback/last image exists.
            return
        if int(var.sizes.get("time", 0)) <= 0:
            return

        idx = int(np.clip(self.trace_canvas.cursor_index, 0, int(var.sizes["time"]) - 1))
        key = (signal_name, idx, self.signal_map_contrast_mode, self.map_smoothing_sigma_px)
        if not force and self._signal_map_last_key == key:
            return
        self._signal_map_last_key = key

        try:
            try:
                fixed_limits = self.global_contrast_limits_for_signal(signal_name, var)
            except Exception as ge:
                # Keep the GUI usable if global scanning fails: fall back to
                # per-frame contrast for display only, without touching saved CSVs.
                fixed_limits = None
                if self.signal_map_contrast_mode == "global":
                    self.statusBar().showMessage(f"Global contrast failed; using frame contrast for display: {ge}")
                    QtWidgets.QApplication.processEvents()
            cached = self._signal_map_cache.get(key)
            if cached is None:
                frame = var.isel(time=idx).compute().values
                if self.map_smoothing_sigma_px > 0:
                    if self.current_brain_signal_mask is None:
                        raise RuntimeError("Cannot smooth map without brainvessels/raw_valid mask")
                    frame = gaussian_masked_display_frame(
                        frame, self.current_brain_signal_mask, self.map_smoothing_sigma_px
                    )
                rgb, (vmin, vmax) = signal_frame_to_jet_rgb(
                    frame,
                    self.current_brain_signal_mask,
                    percentile_range=self.signal_map_percentiles,
                    fixed_limits=fixed_limits,
                )
                cached = (rgb, float(vmin), float(vmax))
                # Keep the cache small so dragging remains responsive without
                # growing memory across long recordings.
                if len(self._signal_map_cache) > 40:
                    self._signal_map_cache.pop(next(iter(self._signal_map_cache)))
                self._signal_map_cache[key] = cached
            rgb, vmin, vmax = cached
            self.roi_widget.set_image(rgb)
            self.update_roi_info()
            mask_txt = "brainvessels + raw_valid" if self.current_brain_signal_mask is not None else "no mask"
            time_txt = ""
            if "time" in var.coords:
                try:
                    time_txt = f" | t={float(var['time'].values[idx]):.3f}s"
                except Exception:
                    time_txt = ""
            contrast_txt = "global" if self.signal_map_contrast_mode == "global" else "frame"
            note = self._global_contrast_notes.get(signal_name, "") if self.signal_map_contrast_mode == "global" else ""
            note_txt = f" | {note}" if note else ""
            sigma_txt = f" | Gaussian σ={self.map_smoothing_sigma_px:g} px" if self.map_smoothing_sigma_px > 0 else ""
            self.statusBar().showMessage(
                f"Signal map: {signal_name} frame={idx}{time_txt} | {mask_txt}{sigma_txt} | jet {contrast_txt} p{self.signal_map_percentiles[0]:g}-p{self.signal_map_percentiles[1]:g} [{vmin:.4g}, {vmax:.4g}]{note_txt}"
            )
        except Exception as e:
            # Never break annotation because of map display.  Fall back to the old
            # static raw preview if available.
            if self.current_raw_image is not None:
                self.roi_widget.set_image(self.current_raw_image)
            self.statusBar().showMessage(f"Signal map fallback to raw preview: {e}")

    # ---------------- Synchronization ----------------

    def on_trace_cursor_changed(self, idx: int) -> None:
        """Synchronize the video frame when the signal cursor moves."""
        if self._updating_trace or self._loading:
            return
        self.trace_canvas.set_cursor_index(idx, emit=False)
        self.sync_video_to_trace_cursor()
        self.update_signal_map_at_cursor()

    def sync_video_to_trace_cursor(self) -> None:
        """Map the trace cursor to a video frame by relative progress.
        
        Notes
        -----
        The mapping interpolates between ``video_start_frame`` and
        ``video_stop_frame`` using cursor index / total signal indices; it does
        not independently estimate temporal offsets or camera drift."""
        if self.video.cap is None or self.video_state is None or not self.video_state.video_exists:
            return
        if len(self.current_trace_times) == 0:
            return
        idx = self.trace_canvas.cursor_index
        if len(self.current_trace_times) <= 1:
            frac = 0.0
        else:
            frac = idx / max(len(self.current_trace_times) - 1, 1)
        start = int(self.video_state.video_start_frame)
        stop = int(self.video_state.video_stop_frame)
        if stop < start:
            start, stop = stop, start
        frame = int(round(start + frac * max(stop - start, 0)))
        self.show_video_frame(frame)

    def sync_trace_to_video_frame(self, frame_idx: int) -> None:
        """Map a selected video frame back to a trace cursor index.
        
        Notes
        -----
        The relative position within the marked video start/stop frames is
        linearly mapped to the time-series sample indices."""
        if self._updating_trace or self.video_state is None or not self.video_state.video_exists:
            return
        if len(self.current_trace_times) == 0:
            return
        start = int(self.video_state.video_start_frame)
        stop = int(self.video_state.video_stop_frame)
        if stop < start:
            start, stop = stop, start
        if stop == start:
            frac = 0.0
        else:
            frac = (int(frame_idx) - start) / max(stop - start, 1)
        frac = float(np.clip(frac, 0, 1))
        idx = int(round(frac * max(len(self.current_trace_times) - 1, 0)))
        self._updating_trace = True
        self.trace_canvas.set_cursor_index(idx, emit=False)
        self._updating_trace = False
        self.update_signal_map_at_cursor()

    # ---------------- Interval labels ----------------

    def current_cursor_time(self) -> float | None:
        """Return the current trace cursor coordinate in seconds."""
        if len(self.current_trace_times) == 0:
            return None
        idx = int(np.clip(self.trace_canvas.cursor_index, 0, len(self.current_trace_times) - 1))
        return float(self.current_trace_times[idx])

    def set_interval_start(self) -> None:
        """Mark the first boundary of a behavior interval."""
        t = self.current_cursor_time()
        if t is None:
            return
        self.interval_start_sec = t
        self.update_interval_info()

    def add_interval(self, label: str) -> None:
        """Append one RUN or REST interval using the current trace cursor.
        
        Parameters
        ----------
        label : str
            Behavior class. The saved schema uses lower-case ``run`` or ``rest``.
        
        Notes
        -----
        Boundaries are stored as ``t0_sec`` and ``t1_sec`` in seconds in the
        ``label_intervals_json`` field. This assumes the displayed Step4 time
        coordinate represents seconds. Equal boundaries are ignored."""
        t1 = self.current_cursor_time()
        if t1 is None:
            return
        if self.interval_start_sec is None:
            self.interval_start_sec = t1
            self.update_interval_info()
            return
        t0 = float(self.interval_start_sec)
        if abs(t1 - t0) < 1e-9:
            return
        item = {
            "label": str(label),
            "t0_sec": float(min(t0, t1)),
            "t1_sec": float(max(t0, t1)),
        }
        self.label_intervals.append(item)
        self.interval_start_sec = None
        self.update_interval_info()
        self.trace_canvas.add_interval_spans(self.label_intervals)
        self.statusBar().showMessage(f"Added {str(label).upper()} interval: {item['t0_sec']:.3f}-{item['t1_sec']:.3f}s")

    def clear_intervals(self) -> None:
        """Clear the current unsaved behavior interval annotations."""
        self.label_intervals = []
        self.interval_start_sec = None
        self.update_interval_info()
        self.trace_canvas.add_interval_spans(self.label_intervals)

    def update_interval_info(self) -> None:
        """Refresh interval counts and the pending start time."""
        start_txt = ""
        if self.interval_start_sec is not None:
            start_txt = f" | interval start={self.interval_start_sec:.3f}s"
        self.interval_info_label.setText(f"Intervals: {len(self.label_intervals)}{start_txt}")

    # ---------------- Saving ----------------

    def save_video_map_only(self) -> None:
        """Save the video mapping separately from behavior labels."""
        if self.video_state is None:
            return
        self.video_state.updated_at = now_iso()
        upsert_csv_row(self.video_map_csv, asdict(self.video_state), key="stem")

    def save_current_state(self, *, explicit: bool = False) -> None:
        """Save the current stem's video mapping, ROI, and behavior intervals.
        
        Parameters
        ----------
        explicit : bool
            Whether the user explicitly confirmed the stem. Automatic navigation
            or closing creates ``in_progress`` unless previously ``reviewed``.
        
        Notes
        -----
        Writes ``behavior_video_map.csv``, ``behavior_review_annotations.csv``
        and an optional per-stem uint8 ROI mask. CSV rows are keyed by ``stem``.
        The saved interval labels and field names are part of the downstream
        REST analysis interface and must not be translated."""
        try:
            if self.current_row is None:
                return
            row = self.current_row
            stem = str(row["stem"])
            self.save_video_map_only()

            roi = self.roi_widget.roi_yxyx()
            mask = self.roi_widget.roi_mask()
            roi_path = ""
            y0 = x0 = y1 = x1 = -1
            if roi is not None and mask is not None and int(mask.sum()) > 0:
                y0, x0, y1, x1 = roi
                roi_path_p = self.roi_root / f"{stem}_behavior_roi.npy"
                np.save(roi_path_p, mask.astype(np.uint8))
                roi_path = str(roi_path_p)

            # Automatic navigation/close saves are drafts, never silently declare
            # a previously unreviewed stem complete. Explicit Save confirms it.
            previous = read_latest_row(self.annotations_csv, stem)
            was_reviewed = (previous is not None and
                            safe_str(previous.get("review_status")).strip().lower() == "reviewed")
            review_status = "reviewed" if (explicit or was_reviewed) else "in_progress"
            video_path = self.video_state.behavior_video_path if self.video_state is not None else ""
            ann = AnnotationState(
                stem=stem,
                group_id=safe_str(row.get("group_id")),
                mouse_id=safe_str(row.get("mouse_id")),
                exp_day=safe_str(row.get("exp_day")),
                state=safe_str(row.get("state")),
                protocol=safe_str(row.get("protocol")),
                raw_zarr_path_resolved=safe_str(row.get("raw_zarr_selected_for_processing_path_resolved", row.get("raw_zarr_path_resolved"))),
                processed_zarr_path_resolved=safe_str(row.get("processed_zarr_path_resolved")),
                behavior_video_path=video_path,
                signal_name=self.selected_signal_name(),
                roi_y0=int(y0),
                roi_x0=int(x0),
                roi_y1=int(y1),
                roi_x1=int(x1),
                roi_mask_path=roi_path,
                label_intervals_json=json.dumps(self.label_intervals, ensure_ascii=False),
                review_status=review_status,
                notes="",
                updated_at=now_iso(),
            )
            upsert_csv_row(self.annotations_csv, asdict(ann), key="stem")
            self.statusBar().showMessage(f"Saved behavior review for {stem}")
        except Exception as e:
            self.show_error("Could not save current state", e)

    # ---------------- Navigation ----------------

    def prev_stem(self) -> None:
        """Save and navigate to the previous selected recording."""
        self.save_current_state()
        self.current_index = max(0, self.current_index - 1)
        self.load_current_stem()

    def next_stem(self) -> None:
        """Save and navigate to the next selected recording."""
        self.save_current_state()
        self.current_index = min(len(self.df) - 1, self.current_index + 1)
        self.load_current_stem()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Persist the current state and release video/Zarr resources on closing."""
        try:
            self.save_current_state()
        except Exception:
            pass
        self.video.close()
        self._close_current_ds()
        event.accept()

    # ---------------- Error display ----------------

    def show_error(self, title: str, err: BaseException) -> None:
        """Display and print an error message with its traceback."""
        msg = f"{err}\n\n{traceback.format_exc()}"
        print(msg)
        QtWidgets.QMessageBox.critical(self, title, msg)
        self.statusBar().showMessage(f"{title}: {err}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    """Parse CLI options and launch the interactive behavior-review window.
    
    Notes
    -----
    ``--config`` supplies the full project YAML path; additional CLI options
    select review scope and display settings. Direct execution from ``src/``
    uses the existing sibling-module imports."""
    parser = argparse.ArgumentParser(description="Manual behavior review GUI using Step4 processed signal maps and behavior video.")
    parser.add_argument("--config", required=True, help="Path to the project YAML configuration file.")
    parser.add_argument("--filter", action="append", default=None, help="Metadata filter KEY=VALUE or KEY=a,b. If provided, replaces step4.debug.filters.")
    parser.add_argument("--protocol", default=None, help="Protocol name, e.g. Rest-state. With variant_keys enabled selects protocol_base after keep-highest-number.")
    parser.add_argument("--trace-bandpass", action="store_true", help="Initially show 0.08–0.40 Hz band-pass ROI trace; can toggle in GUI; no changes to data/annotations.")
    parser.add_argument("--trace-gaussian-sigma", type=float, default=0.0, metavar="FRAMES", help="Initially smooth displayed ROI trace in time (e.g. 1 frame); 0 disables. Alternative to --trace-bandpass.")
    parser.add_argument("--map-sigma", type=float, default=0.0, metavar="PX", help="Initially smooth displayed signal maps with spatial Gaussian sigma in pixels (e.g. 1); can toggle in GUI.")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group_id filter.")
    parser.add_argument("--stems", nargs="*", default=None, help="Optional stem filter.")
    parser.add_argument("--include-bad", action="store_true", help="Include rows marked is_bad in Step4 metadata.")
    parser.add_argument("--preview-n-frames", type=int, default=200, help="Number of raw-zarr frames averaged for the fallback preview image.")
    parser.add_argument("--raw-preview-wavelength", default="505", help="Wavelength shown for fallback raw preview; default: 505.")
    parser.add_argument(
        "--map-contrast",
        choices=["frame", "global"],
        default="frame",
        help="Signal-map contrast mode. frame = vmin/vmax from the current frame; global = vmin/vmax computed once from the whole recording inside the Step5-like brain mask.",
    )
    parser.add_argument(
        "--map-percentiles",
        nargs=2,
        type=float,
        default=(2.0, 98.0),
        metavar=("LOW", "HIGH"),
        help="Percentiles for signal-map contrast, default: 2 98.",
    )
    parser.add_argument(
        "--global-contrast-time-chunk",
        type=int,
        default=50,
        help="Number of time frames read at once while computing global contrast, default: 50.",
    )
    parser.add_argument(
        "--global-contrast-max-values",
        type=int,
        default=2_000_000,
        help="Maximum masked values kept for global contrast. Uses all time points and samples brain pixels if needed. Set 0 for exact all masked values; may be slow/heavy.",
    )
    parser.add_argument(
        "--review-mode",
        choices=["all", "unreviewed", "unannotated", "pending"],
        default="all",
        help=(
            "Which stems to open after metadata filters. all = all selected rows; "
            "unreviewed = only stems absent from behavior_review_annotations.csv; "
            "unannotated = only stems whose latest label_intervals_json is empty [] or missing; "
            "pending = stems not explicitly confirmed with review_status=reviewed "
            "(including missing CSV rows and imported in_progress rows)."
        ),
    )
    parser.add_argument(
        "--unreviewed-only",
        action="store_true",
        help="Shortcut for --review-mode unreviewed.",
    )
    parser.add_argument(
        "--unannotated-only",
        action="store_true",
        help="Shortcut for --review-mode unannotated (keeps rows already in CSV when label_intervals_json is []).",
    )
    args = parser.parse_args()

    filters = parse_cli_filters(args.filter)
    if args.protocol:
        filters = dict(filters or {})
        if "protocol" in filters:
            parser.error("Use either --protocol or --filter protocol=..., not both")
        filters["protocol"] = str(args.protocol)
    if args.map_sigma < 0 or not np.isfinite(args.map_sigma):
        parser.error("--map-sigma must be finite and nonnegative")
    if args.trace_gaussian_sigma < 0 or not np.isfinite(args.trace_gaussian_sigma):
        parser.error("--trace-gaussian-sigma must be finite and nonnegative")
    if args.trace_bandpass and args.trace_gaussian_sigma > 0:
        parser.error("--trace-bandpass and --trace-gaussian-sigma are mutually exclusive")
    if args.unannotated_only:
        review_mode = "unannotated"
    elif args.unreviewed_only:
        review_mode = "unreviewed"
    else:
        review_mode = args.review_mode
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = BehaviorReviewWindow(
        config_path=args.config,
        filters=filters,
        groups=args.groups,
        stems=args.stems,
        include_bad=True if args.include_bad else None,
        preview_n_frames=args.preview_n_frames,
        raw_preview_wavelength=args.raw_preview_wavelength,
        signal_map_contrast_mode=args.map_contrast,
        signal_map_percentiles=tuple(args.map_percentiles),
        global_contrast_time_chunk=args.global_contrast_time_chunk,
        global_contrast_max_values=args.global_contrast_max_values,
        review_mode=review_mode,
        trace_bandpass_enabled=args.trace_bandpass,
        trace_gaussian_sigma_frames=args.trace_gaussian_sigma,
        map_smoothing_sigma_px=args.map_sigma,
    )
    win.show()
    sys.exit(app.exec_() if hasattr(app, "exec_") else app.exec())


if __name__ == "__main__":
    main()
