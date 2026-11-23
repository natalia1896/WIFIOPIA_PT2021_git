"""REST-segment rescue GUI for final connectivity candidates.

Purpose
-------
Specialized QC viewer using existing manually annotated behavior intervals.
It reads the source behavior annotations without modifying them.

The viewer:
* by default reviews eligible stems except states listed in ``rest_qc.anesthesia_states``, without requiring experimental-group assignment;
* optional legacy ``analysis_selection`` filter is explicit (not the default);
* keeps only manually annotated REST intervals;
* splits REST intervals into full, non-overlapping windows with configurable duration (default 60 s);
* asks for an explicit USE / REJECT / UNCLEAR decision for every candidate window;
* allows the current candidate window to be shifted inside its parent REST interval before deciding;
* shows the selected HbT window in Napari, with segment-centered display selected by default;
* can load matching raw wavelength chunks for visual artifact inspection;
* provides one drawable activity ROI per stem and plots whole-brain + ROI HbT traces;
* optional display-only spatial Gaussian for the Napari HbT stack and temporal Gaussian for both traces;
* keeps one editable rescue shadow mask per stem while all candidates are reviewed;
* stores every accepted candidate interval in label_intervals_json using the same REST interval format;
* writes rescue decisions to a separate CSV and never changes behavior_review_annotations.csv.

No 0.08-0.4 Hz band-pass is applied in this QC viewer. The target band is
recorded as context; this stage checks whether the source interval is clean
enough to enter the later frequency/connectivity analysis.

Example
-------
python rest_segment_review_gui.py --config path/to/config.yaml

"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
import faulthandler
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter, gaussian_filter1d

try:
    import napari
    from qtpy import QtCore, QtWidgets
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure
except Exception as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "This viewer requires napari, qtpy, a Qt binding, and Matplotlib Qt backend. "
        "Run it in the same environment as the existing WIFIOPIA Napari tools."
    ) from exc


# Current Step4 is authoritative for the recording index and spatial masks.
from step4_signal_processing import (
    build_step4_index, open_raw_zarr, select_raw_zarr_path_for_processing,
    load_masks_for_row, cfg_get,
)


# =============================================================================
# Generic helpers
# =============================================================================


def configure_console_encoding_fallback() -> None:
    """Do not let Windows cp1251 console/log streams crash on Unicode text.

    This only changes error handling for stdout/stderr, not their encoding and
    not any CSV or Zarr files. Unsupported characters become ASCII U+XXXX escape text.
    Qt widgets remain Unicode and show the original labels normally.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="backslashreplace")
            except (OSError, ValueError, TypeError):
                pass


def safe_str(value: Any) -> str:
    """Normalize missing values to empty text while preserving other values as strings."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def bool_series(series: pd.Series) -> pd.Series:
    """Recognize true-like metadata values in a pandas Series."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def parse_cli_filters(items: Sequence[str] | None) -> dict[str, Any] | None:
    """Parse CLI KEY=VALUE metadata filters into a dictionary."""
    if not items:
        return None
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Filter must be KEY=VALUE or KEY=a,b, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        values = [part.strip() for part in value.split(",") if part.strip()]
        if not key:
            raise ValueError(f"Empty filter key in {item!r}")
        out[key] = values if len(values) > 1 else (values[0] if values else "")
    return out


def normalize_wavelength_label(value: Any) -> str:
    """Normalize a numeric wavelength label to comparable string form."""
    try:
        number = float(value)
        if np.isfinite(number) and abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return str(number)
    except Exception:
        return str(value).strip()


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean (y, x) mask with nearest-neighbor interpolation."""
    mask = np.asarray(mask, dtype=bool)
    if tuple(mask.shape) == tuple(shape):
        return mask
    height, width = int(shape[0]), int(shape[1])
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def polygon_mask(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize Napari polygon vertices into a boolean image mask."""
    height, width = int(shape[0]), int(shape[1])
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return np.zeros((height, width), dtype=bool)
    yx = points[:, -2:]
    if not np.all(np.isfinite(yx)):
        return np.zeros((height, width), dtype=bool)
    xy = np.column_stack([yx[:, 1], yx[:, 0]])
    xy[:, 0] = np.clip(np.rint(xy[:, 0]), 0, width - 1)
    xy[:, 1] = np.clip(np.rint(xy[:, 1]), 0, height - 1)
    out = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(out, [xy.astype(np.int32)], 1)
    return out.astype(bool)


def read_csv_text(path: Path) -> pd.DataFrame:
    """Read annotation CSV without pandas dtype inference surprises."""
    return pd.read_csv(
        Path(path),
        dtype=object,
        keep_default_na=False,
        encoding="utf-8-sig",
    ).astype(object)


def parse_intervals_json(value: Any) -> list[dict[str, Any]]:
    """Parse and normalize manually labeled behavior intervals.
    
    Parameters
    ----------
    value : Any
        JSON-encoded list stored in ``label_intervals_json``.
    
    Returns
    -------
    list[dict[str, Any]]
        Valid RUN or REST intervals with ordered start/end times in seconds.
    
    Notes
    -----
    Malformed items are ignored. Values of ``label`` are data-schema values
    and therefore remain ``run`` and ``rest``."""
    try:
        raw = json.loads(safe_str(value) or "[]")
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = safe_str(item.get("label")).strip().lower()
        if label not in {"run", "rest"}:
            continue
        try:
            t0 = float(item.get("t0_sec"))
            t1 = float(item.get("t1_sec"))
        except Exception:
            continue
        if not np.isfinite(t0) or not np.isfinite(t1) or abs(t1 - t0) < 1e-12:
            continue
        out.append(
            {
                "label": label,
                "t0_sec": float(min(t0, t1)),
                "t1_sec": float(max(t0, t1)),
            }
        )
    return out


def build_mouse_group_map(experimental_groups: dict[str, Any]) -> dict[str, str]:
    """Assign each mouse identifier to its configured experimental group."""
    mapping: dict[str, str] = {}
    for group, mice in dict(experimental_groups or {}).items():
        if mice is None:
            continue
        if isinstance(mice, (str, bytes)):
            mice = [mice]
        for mouse in mice:
            key = safe_str(mouse).strip()
            if not key:
                continue
            if key in mapping and mapping[key] != str(group):
                raise ValueError(f"Mouse {key!r} is assigned to more than one experimental group")
            mapping[key] = str(group)
    return mapping


def apply_analysis_selection(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
    *,
    require_block: bool = True,
) -> pd.DataFrame:
    """Apply the same analysis_selection convention used by Behavior_rest_summary_v1."""
    selection = dict(cfg_get(config, "analysis_selection", default={}) or {})
    if not selection:
        if require_block:
            raise RuntimeError(
                "config has no analysis_selection block. Add the analysis_selection used for the final analysis "
                "or launch with --ignore-analysis-selection explicitly."
            )
        out = dataframe.copy()
        if "experimental_group" not in out.columns:
            out["experimental_group"] = "UNASSIGNED"
        return out.reset_index(drop=True)

    out = dataframe.copy()
    experimental_groups = dict(selection.get("experimental_groups", {}) or {})
    mouse_to_group = build_mouse_group_map(experimental_groups)
    if "mouse_id" in out.columns:
        out["experimental_group"] = out["mouse_id"].map(lambda v: mouse_to_group.get(safe_str(v), "UNASSIGNED"))
    else:
        out["experimental_group"] = "UNASSIGNED"

    exp_days = selection.get("exp_days")
    if exp_days:
        if "exp_day" not in out.columns:
            raise KeyError("analysis_selection.exp_days is set but Step4 index has no exp_day column")
        wanted = {safe_str(v) for v in exp_days}
        out = out[out["exp_day"].map(safe_str).isin(wanted)]

    include_mice = selection.get("include_mice")
    if include_mice:
        if "mouse_id" not in out.columns:
            raise KeyError("analysis_selection.include_mice is set but Step4 index has no mouse_id column")
        wanted = {safe_str(v) for v in include_mice}
        out = out[out["mouse_id"].map(safe_str).isin(wanted)]

    extra_filters = dict(selection.get("filters", {}) or {})
    for key, wanted in extra_filters.items():
        if key not in out.columns:
            raise KeyError(f"analysis_selection.filters contains {key!r}, but Step4 index has no such column")
        if isinstance(wanted, (list, tuple, set)):
            wanted_set = {safe_str(v) for v in wanted}
            out = out[out[key].map(safe_str).isin(wanted_set)]
        else:
            out = out[out[key].map(safe_str).eq(safe_str(wanted))]

    if experimental_groups and bool(selection.get("require_group_assignment", True)):
        out = out[~out["experimental_group"].eq("UNASSIGNED")]
    return out.reset_index(drop=True)


def segment_id(stem: str, t0_sec: float, t1_sec: float) -> str:
    """Build a stable segment identifier from stem and boundaries in seconds."""
    return f"{stem}|{float(t0_sec):.6f}|{float(t1_sec):.6f}"


def split_rest_intervals_into_chunks(
    stem: str,
    intervals: list[dict[str, Any]],
    segment_sec: float,
) -> list[dict[str, Any]]:
    """Split annotated REST periods into full non-overlapping candidate windows.
    
    Parameters
    ----------
    stem : str
        Recording identifier used to build segment IDs.
    intervals : list[dict[str, Any]]
        Behavior intervals with ``label`` and boundaries in seconds.
    segment_sec : float
        Candidate duration in seconds (CLI default 60).
    
    Returns
    -------
    list[dict[str, Any]]
        Full-duration REST candidates with parent-window provenance,
        within-parent chunk index, and stable ``segment_id``.
    
    Notes
    -----
    Trailing fragments shorter than ``segment_sec`` are omitted. Candidates
    are only generated from manually annotated REST intervals."""
    width = float(segment_sec)
    if not np.isfinite(width) or width <= 0:
        raise ValueError("segment_sec must be positive")
    chunks: list[dict[str, Any]] = []
    for parent_index, item in enumerate(intervals):
        if safe_str(item.get("label")).strip().lower() != "rest":
            continue
        parent_t0 = float(item["t0_sec"])
        parent_t1 = float(item["t1_sec"])
        duration = max(0.0, parent_t1 - parent_t0)
        n_full = int(np.floor((duration + 1e-9) / width))
        for chunk_index in range(n_full):
            t0 = parent_t0 + chunk_index * width
            t1 = t0 + width
            chunks.append({
                "label": "rest",
                "t0_sec": float(t0),
                "t1_sec": float(t1),
                "parent_rest_t0_sec": float(parent_t0),
                "parent_rest_t1_sec": float(parent_t1),
                "parent_rest_index": int(parent_index),
                "chunk_index_in_parent": int(chunk_index),
                "segment_id": segment_id(stem, t0, t1),
                "_qc_status": "",
            })
    return chunks


def default_qc_path(paths: dict[str, Path], segment_sec: float) -> Path:
    """Construct the legacy primary-QC CSV path for the segment duration."""
    value = float(segment_sec)
    tag = str(int(round(value))) if abs(value - round(value)) < 1e-9 else str(value).replace(".", "p")
    root = Path(paths["out_root"]) / "behavior_review" / f"rest_segment_qc_{tag}s"
    return root / "rest_segment_qc.csv"


def read_qc_status_map(path: Path) -> dict[str, str]:
    """Load the most recent QC status for each segment identifier."""
    path = Path(path)
    if not path.exists():
        return {}
    df = read_csv_text(path)
    if df.empty or "segment_id" not in df.columns or "qc_status" not in df.columns:
        return {}
    df = df.drop_duplicates(subset=["segment_id"], keep="last")
    return {
        safe_str(row["segment_id"]): safe_str(row["qc_status"]).strip().lower()
        for _, row in df.iterrows()
        if safe_str(row["segment_id"]).strip()
    }


def csv_safe_value(value: Any) -> str:
    """Serialize a scalar into a plain CSV-compatible string."""
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
        v = float(value)
        return "" if not np.isfinite(v) else repr(v)
    return str(value)


def upsert_qc_row(path: Path, row: dict[str, Any]) -> None:
    """Insert or update a primary QC record keyed by segment_id."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {str(k): csv_safe_value(v) for k, v in row.items()}
    if path.exists():
        df = read_csv_text(path)
    else:
        df = pd.DataFrame(dtype=object)
    for col in safe:
        if col not in df.columns:
            df[col] = ""
    if "segment_id" not in df.columns:
        df["segment_id"] = ""
    target = safe.get("segment_id", "")
    mask = df["segment_id"].map(safe_str).eq(target)
    if mask.any():
        idx = df.index[mask][-1]
        for col, val in safe.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([safe], dtype=object)], ignore_index=True, sort=False)
    ordered = list(safe.keys()) + [c for c in df.columns if c not in safe]
    df = df[ordered].astype(object)
    df.to_csv(path, index=False, encoding="utf-8")


def candidate_hbt_signals(ds: xr.Dataset) -> list[str]:
    """List available HbT-like processed signal variable names."""
    names = list(ds.data_vars)
    lower = {name.lower(): name for name in names}
    preferred: list[str] = []
    for candidate in (
        "dcHbT",
        "HbT",
        "dcHbT_vreg",
        "dcHbTvreg",
        "HbTvreg",
        "HbT_vreg",
    ):
        hit = lower.get(candidate.lower())
        if hit and hit not in preferred:
            preferred.append(hit)
    for name in names:
        if "hbt" in name.lower() and set(ds[name].dims) == {"time", "y", "x"} and name not in preferred:
            preferred.append(name)
    return preferred


def time_values_numeric(var: xr.DataArray) -> np.ndarray:
    """Return numeric time coordinates, or sample indices as a fallback."""
    n_time = int(var.sizes["time"])
    if "time" not in var.coords:
        return np.arange(n_time, dtype=float)
    values = np.asarray(var.coords["time"].values)
    try:
        result = values.astype(float)
        if result.ndim == 1 and result.size == n_time and np.all(np.isfinite(result)):
            return result
    except Exception:
        pass
    return np.arange(n_time, dtype=float)


def interval_indices(var: xr.DataArray, t0: float, t1: float) -> tuple[int, int]:
    """Find inclusive sample bounds for an interval on the stored time coordinate."""
    times = time_values_numeric(var)
    if times.size == 0:
        raise ValueError("Signal has no time points")
    lo, hi = float(min(t0, t1)), float(max(t0, t1))
    inside = np.flatnonzero((times >= lo) & (times <= hi))
    if inside.size:
        return int(inside[0]), int(inside[-1])
    i0 = int(np.nanargmin(np.abs(times - lo)))
    i1 = int(np.nanargmin(np.abs(times - hi)))
    if i1 < i0:
        i0, i1 = i1, i0
    return i0, i1


def annotation_fps(config: dict[str, Any]) -> float:
    """Sampling frequency used to convert annotation seconds to frame indices.

    Existing behavior annotations store t0_sec/t1_sec in seconds.  Frame selection
    must therefore be performed as round(seconds * fps), independently of how
    the processed zarr happens to encode its ``time`` coordinate.
    """
    fps = float(cfg_get(config, "raw", "fps", default=float("nan")))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("config raw.fps must be a positive finite number")
    return fps


def resolve_annotation_interval(
    var: xr.DataArray,
    t0: float,
    t1: float,
    config: dict[str, Any],
    *,
    requested_unit: str = "seconds",
) -> dict[str, Any]:
    """Convert recorded behavior boundaries to Step4 frame indices.
    
    Parameters
    ----------
    var : xarray.DataArray
        Time-dependent Step4 signal with a time dimension.
    t0, t1 : float
        Annotation boundaries, normally in seconds.
    config : dict
        Full YAML configuration; ``raw.fps`` is used for seconds-to-frame
        conversion.
    requested_unit : {'seconds', 'auto', 'frames'}
        Explicit unit interpretation. ``auto`` uses seconds, while ``frames``
        is a compatibility/debugging mode.
    
    Returns
    -------
    dict[str, Any]
        Inclusive frame bounds, requested bounds, clipping flags, physical
        seconds, and diagnostics of the stored time coordinate.
    
    Notes
    -----
    For second-based annotations, the requested interval is treated as
    half-open [t0, t1) in samples so adjacent windows do not share frames.
    Boundary indices are clipped to the available recording."""
    unit = str(requested_unit or "seconds").strip().lower()
    if unit not in {"auto", "seconds", "frames"}:
        raise ValueError("annotation time unit must be auto, seconds, or frames")

    n = int(var.sizes.get("time", 0))
    if n <= 0:
        raise ValueError("Signal has no time points")
    fps = annotation_fps(config)
    lo, hi = float(min(t0, t1)), float(max(t0, t1))

    if unit in {"auto", "seconds"}:
        inferred = "seconds"
        reason = f"t0_sec/t1_sec are seconds; converted with config raw.fps={fps:g} Hz"
        requested_i0 = int(round(lo * fps))
        # QC chunks use half-open [start, stop) sampling so adjacent 60-s chunks
        # never share the boundary frame. i1 remains the last INCLUDED frame.
        requested_stop_exclusive = int(round(hi * fps))
        requested_i1 = max(requested_i0, requested_stop_exclusive - 1)
        annotation_duration_sec = max(0.0, hi - lo)
    else:
        inferred = "frames"
        reason = "explicit debugging mode: annotation values interpreted as frame indices"
        requested_i0 = int(round(lo))
        requested_i1 = int(round(hi))
        annotation_duration_sec = max(0.0, (requested_i1 - requested_i0) / fps)

    if requested_i1 < requested_i0:
        requested_i0, requested_i1 = requested_i1, requested_i0

    i0 = int(np.clip(requested_i0, 0, n - 1))
    i1 = int(np.clip(requested_i1, 0, n - 1))
    if i1 < i0:
        i0, i1 = i1, i0

    actual_start_sec = i0 / fps
    actual_end_sec = i1 / fps
    sampled_span_sec = max(0.0, (i1 - i0) / fps)
    clipped = (i0 != requested_i0) or (i1 != requested_i1)

    coord = time_values_numeric(var)
    coord_start = float(coord[i0]) if coord.size > i0 else float("nan")
    coord_end = float(coord[i1]) if coord.size > i1 else float("nan")

    return {
        "unit": inferred,
        "reason": reason,
        "fps": fps,
        "i0": i0,
        "i1": i1,
        "requested_i0": requested_i0,
        "requested_i1": requested_i1,
        "clipped": bool(clipped),
        "annotation_lo": lo,
        "annotation_hi": hi,
        "annotation_numeric_span": max(0.0, hi - lo),
        "actual_start_sec": actual_start_sec,
        "actual_end_sec": actual_end_sec,
        "sampled_span_sec": sampled_span_sec,
        "duration_sec": annotation_duration_sec,
        "n_time": n,
        "coord_start": coord_start,
        "coord_end": coord_end,
    }


def time_axis_diagnostics(var: xr.DataArray, config: dict[str, Any]) -> dict[str, Any]:
    """Describe stored zarr coordinate separately from physical sampling time."""
    coord = time_values_numeric(var)
    fps = annotation_fps(config)
    out: dict[str, Any] = {
        "n_time": int(coord.size),
        "coord_min": float(coord[0]) if coord.size else float("nan"),
        "coord_max": float(coord[-1]) if coord.size else float("nan"),
        "coord_step_median": float("nan"),
        "coord_looks_like_frames": False,
        "config_raw_fps": fps,
        "effective_dt_sec": 1.0 / fps,
    }
    if coord.size >= 2:
        diffs = np.diff(coord)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            step = float(np.nanmedian(diffs))
            out["coord_step_median"] = step
            max_expected_frame = float(coord.size - 1)
            looks_step = np.isclose(step, 1.0, rtol=0.0, atol=1e-6)
            looks_range = np.isfinite(coord[-1]) and abs(float(coord[-1]) - max_expected_frame) <= max(2.0, 0.01 * max_expected_frame)
            out["coord_looks_like_frames"] = bool(looks_step and looks_range)
    return out

def normalized_roi_vertices(vertices: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Store a 2-D polygon in shape-independent [0, 1] y/x coordinates."""
    pts = np.asarray(vertices, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    yx = pts[:, -2:].copy()
    if not np.all(np.isfinite(yx)):
        return None
    h, w = int(shape[0]), int(shape[1])
    yx[:, 0] = yx[:, 0] / max(h - 1, 1)
    yx[:, 1] = yx[:, 1] / max(w - 1, 1)
    return np.clip(yx, 0.0, 1.0)


def denormalized_roi_vertices(vertices01: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Map normalized ROI polygon vertices back to pixel coordinates."""
    pts = np.asarray(vertices01, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
        return None
    if not np.all(np.isfinite(pts)):
        return None
    h, w = int(shape[0]), int(shape[1])
    yx = np.clip(pts[:, -2:], 0.0, 1.0).copy()
    yx[:, 0] *= max(h - 1, 1)
    yx[:, 1] *= max(w - 1, 1)
    return yx


def compute_segment_contrast_limits(
    var: xr.DataArray,
    mask: np.ndarray,
    *,
    percentiles: tuple[float, float],
    time_chunk: int = 50,
    max_values: int = 2_000_000,
) -> tuple[float, float, str]:
    """Percentile limits over the selected segment only, inside brain mask."""
    data = var.transpose("time", "y", "x")
    n_time = int(data.sizes["time"])
    shape = (int(data.sizes["y"]), int(data.sizes["x"]))
    mask2 = resize_bool(mask, shape)
    all_idx = np.flatnonzero(mask2.ravel())
    if n_time <= 0 or all_idx.size < 10:
        raise ValueError("Too little data inside brain mask for segment-global contrast")

    total = int(n_time) * int(all_idx.size)
    if int(max_values) <= 0 or total <= int(max_values):
        chosen = all_idx
        note = f"all pixels, frames={n_time}, pixels={all_idx.size}"
    else:
        n_px = max(10, min(int(all_idx.size), int(max_values) // max(n_time, 1)))
        rng = np.random.default_rng(0)
        chosen = np.sort(rng.choice(all_idx, size=n_px, replace=False))
        note = f"all frames, sampled pixels={n_px}/{all_idx.size}"

    values: list[np.ndarray] = []
    chunk = max(1, int(time_chunk))
    for start in range(0, n_time, chunk):
        stop = min(n_time, start + chunk)
        arr = np.asarray(data.isel(time=slice(start, stop)).compute().values, dtype=np.float32)
        vals = arr.reshape(arr.shape[0], -1)[:, chosen].reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            values.append(vals)
    if not values:
        raise ValueError("No finite values for segment-global contrast")
    merged = np.concatenate(values)
    p_low, p_high = map(float, percentiles)
    vmin, vmax = np.nanpercentile(merged, [p_low, p_high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(merged)), float(np.nanmax(merged))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax), note


def gaussian_temporal_display_trace(values: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Gaussian-smooth a plotted trace in optical frames; do not alter the source.

    NaN/inf never contribute to neighboring samples and remain NaN at their
    original positions. Sigma 0 means no smoothing.
    """
    x = np.asarray(values, dtype=np.float64)
    sigma = float(sigma_frames)
    if x.ndim != 1:
        raise ValueError("Temporal display Gaussian expects a one-dimensional trace")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Temporal Gaussian sigma must be finite and nonnegative")
    if sigma == 0 or x.size == 0:
        return x.copy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.full(x.shape, np.nan, dtype=np.float64)
    numerator = gaussian_filter1d(np.where(finite, x, 0.0), sigma, mode="reflect")
    denominator = gaussian_filter1d(finite.astype(float), sigma, mode="reflect")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    out[finite] = numerator[finite] / np.maximum(denominator[finite], 1e-12)
    return out


def gaussian_masked_display_frame(frame: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """Display-only spatial Gaussian, normalized within valid brain pixels.

    Neither excluded pixels nor nonfinite data can bleed into the displayed
    brain area. Only the originally finite, valid pixels remain visible.
    """
    frame = np.asarray(frame, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    sigma = float(sigma_px)
    if frame.ndim != 2 or frame.shape != mask.shape:
        raise ValueError("Spatial display Gaussian expects same-size 2D frame and mask")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Spatial Gaussian sigma must be finite and nonnegative")
    valid = mask & np.isfinite(frame)
    if sigma == 0:
        return np.where(valid, frame, np.nan).astype(np.float32)
    numerator = gaussian_filter(np.where(valid, frame, 0.0), sigma, mode="constant", cval=0.0)
    denominator = gaussian_filter(valid.astype(np.float32), sigma, mode="constant", cval=0.0)
    out = np.full(frame.shape, np.nan, dtype=np.float32)
    out[valid] = numerator[valid] / np.maximum(denominator[valid], 1e-12)
    return out


def gaussian_spatial_display_stack(var: xr.DataArray, mask: np.ndarray, sigma_px: float) -> xr.DataArray:
    """Smooth every Napari HbT display frame in y/x only (never across time).

    ``apply_ufunc(dask='parallelized')`` keeps chunked Step4 Zarr lazy. Its
    spatial core dimensions are rechunked if necessary, not time-smoothed.
    Returned data has identical time/y/x coordinates; input is untouched.
    """
    data = var.transpose("time", "y", "x")
    image_shape = (int(data.sizes["y"]), int(data.sizes["x"]))
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_shape:
        raise ValueError(f"Spatial display mask shape {mask.shape} != HbT grid {image_shape}")
    sigma = float(sigma_px)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Spatial Gaussian sigma must be finite and nonnegative")
    mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": data.coords["y"], "x": data.coords["x"]})
    if sigma == 0:
        return data.where(mask_da)
    smoothed = xr.apply_ufunc(
        gaussian_masked_display_frame, data, mask_da,
        input_core_dims=[["y", "x"], ["y", "x"]],
        output_core_dims=[["y", "x"]],
        kwargs={"sigma_px": sigma},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
        keep_attrs=True,
    )
    return smoothed.transpose("time", "y", "x")


def segment_centered_var(var: xr.DataArray, *, max_eager_bytes: int = 128 * 1024 * 1024) -> xr.DataArray:
    """Subtract each pixel's temporal mean from a displayed segment.
    
    Parameters
    ----------
    var : xarray.DataArray
        Selected signal with dimensions (time, y, x).
    max_eager_bytes : int
        Upper size limit for eager float32 materialization.
    
    Returns
    -------
    xarray.DataArray
        Display-only segment-centered signal with original dimensions and units.
    
    Notes
    -----
    Smaller stacks are computed once; larger stacks retain a lazy data array
    with an eagerly evaluated temporal mean. Saved Zarr is not modified."""
    data = var.transpose("time", "y", "x")
    n_elements = int(np.prod([int(data.sizes[d]) for d in ("time", "y", "x")]))
    estimated_bytes = n_elements * np.dtype(np.float32).itemsize
    if estimated_bytes <= int(max_eager_bytes):
        # A single float32 display stack is substantially smaller than repeated
        # dask-backed segment mean calculations. The offset is subtracted in place.
        arr = np.asarray(data.astype(np.float32).compute().values, dtype=np.float32).copy()
        counts = np.count_nonzero(np.isfinite(arr), axis=0)
        means = np.zeros(arr.shape[1:], dtype=np.float32)
        valid_pixels = counts > 0
        if np.any(valid_pixels):
            summed = np.nansum(arr, axis=0, dtype=np.float64)
            means[valid_pixels] = (summed[valid_pixels] / counts[valid_pixels]).astype(np.float32)
        np.subtract(arr, means[None, :, :], out=arr)
        return xr.DataArray(arr, dims=data.dims, coords=data.coords,
                            attrs=data.attrs, name=data.name)
    # Do not eagerly materialize 464x344 or larger high-resolution stacks.
    # Computing the mean separately prevents Dask from rescanning the whole
    # temporal window for every individual contrast chunk.
    mean_frame = data.mean(dim="time", skipna=True).compute()
    return data.astype(np.float32) - mean_frame.astype(np.float32)


def nearest_indices(source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Nearest monotonic source index for every target time."""
    src = np.asarray(source_times, dtype=float)
    tgt = np.asarray(target_times, dtype=float)
    if src.size == 0:
        return np.zeros(tgt.size, dtype=int)
    if src.size == 1:
        return np.zeros(tgt.size, dtype=int)
    order = np.argsort(src)
    sorted_src = src[order]
    pos = np.searchsorted(sorted_src, tgt, side="left")
    pos = np.clip(pos, 1, len(sorted_src) - 1)
    left = sorted_src[pos - 1]
    right = sorted_src[pos]
    choose_right = np.abs(right - tgt) < np.abs(tgt - left)
    sorted_idx = pos - 1 + choose_right.astype(int)
    return order[sorted_idx].astype(int)


def manual_review_root(paths: dict[str, Path], config: dict[str, Any]) -> Path:
    """Resolve the optional Step4 manual ROI review directory from YAML."""
    manual_cfg = dict(cfg_get(config, "step4", "manual_roi_review", default={}) or {})
    folder_name = str(manual_cfg.get("folder_name", "manual_roi_review"))
    return Path(paths["step4_root"]) / folder_name


def jet_colormap() -> Any:
    """Return a Napari custom JET colormap, with a safe string fallback."""
    try:
        from napari.utils.colormaps import Colormap
        try:
            from matplotlib import colormaps
            colors = colormaps.get_cmap("jet")(np.linspace(0.0, 1.0, 256))
        except Exception:
            from matplotlib import cm
            colors = cm.get_cmap("jet")(np.linspace(0.0, 1.0, 256))
        return Colormap(colors=np.asarray(colors), name="segment_jet")
    except Exception:
        return "turbo"




# =============================================================================
# Rescue index / persistence
# =============================================================================


def default_annotations_path(paths: dict[str, Path]) -> Path:
    """Return the default behavior annotations CSV path."""
    return Path(paths["out_root"]) / "behavior_review" / "behavior_review_annotations.csv"


def default_primary_qc_path(paths: dict[str, Path], segment_sec: float) -> Path:
    """Return the default legacy primary REST-QC CSV path."""
    value = float(segment_sec)
    tag = str(int(round(value))) if abs(value - round(value)) < 1e-9 else str(value).replace(".", "p")
    return Path(paths["out_root"]) / "behavior_review" / f"rest_segment_qc_{tag}s" / "rest_segment_qc.csv"


def default_rescue_root(paths: dict[str, Path], segment_sec: float) -> Path:
    """Return the rescue review directory for the selected window duration."""
    value = float(segment_sec)
    tag = str(int(round(value))) if abs(value - round(value)) < 1e-9 else str(value).replace(".", "p")
    return Path(paths["out_root"]) / "behavior_review" / f"rest_segment_rescue_{tag}s"


def read_latest_by_key(path: Path, key: str) -> pd.DataFrame:
    """Read a CSV and retain the latest row per recording key."""
    if not Path(path).exists():
        return pd.DataFrame(dtype=object)
    df = read_csv_text(Path(path))
    if df.empty or key not in df.columns:
        return df
    return df.drop_duplicates(subset=[key], keep="last").copy()


def csv_safe_value(value: Any) -> str:
    """Serialize a scalar into a plain CSV-compatible string."""
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
        v = float(value)
        return "" if not np.isfinite(v) else repr(v)
    return str(value)


def upsert_csv_row(path: Path, row: dict[str, Any], *, key: str = "stem") -> None:
    """Insert or update one stem-keyed CSV row without dropping other columns."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = {str(k): csv_safe_value(v) for k, v in row.items()}
    if path.exists():
        df = read_csv_text(path)
    else:
        df = pd.DataFrame(dtype=object)
    for col in safe:
        if col not in df.columns:
            df[col] = ""
    if key not in df.columns:
        df[key] = ""
    target = safe.get(key, "")
    mask = df[key].map(safe_str).eq(target)
    if mask.any():
        idx = df.index[mask][-1]
        for col, val in safe.items():
            df.at[idx, col] = val
    else:
        df = pd.concat([df, pd.DataFrame([safe], dtype=object)], ignore_index=True, sort=False)
    ordered = list(safe.keys()) + [c for c in df.columns if c not in safe]
    df = df[ordered].astype(object)
    # Write transactionally: interrupted saves must not destroy existing manual QC.
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def good_counts_by_stem(qc_path: Path) -> dict[str, int]:
    """Count previously accepted primary-QC chunks per recording."""
    path = Path(qc_path)
    if not path.exists():
        return {}
    qc = read_csv_text(path)
    if qc.empty or "stem" not in qc.columns or "qc_status" not in qc.columns:
        return {}
    if "segment_id" in qc.columns:
        qc = qc.drop_duplicates(subset=["segment_id"], keep="last")
    qc["qc_status"] = qc["qc_status"].map(safe_str).str.strip().str.lower()
    good = qc[qc["qc_status"].eq("good")]
    if good.empty:
        return {}
    return good.groupby(good["stem"].map(safe_str)).size().astype(int).to_dict()


def build_rescue_index(
    config_path: Path,
    *,
    filters: dict[str, Any] | None,
    manual_stems: Sequence[str] | None,
    include_bad: bool,
    annotations_path: Path | None,
    primary_qc_path: Path | None,
    segment_sec: float,
    auto_low_good: bool,
    max_good_for_auto: int,
    use_analysis_selection: bool,
    review_all: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path], Path, Path, Path]:
    """Select REST recordings for the combined QC/rescue interface.
    
    Parameters
    ----------
    config_path : pathlib.Path
        Path to the complete project YAML configuration.
    filters : dict or None
        Explicit Step4 index filters; by default an empty filter overrides
        Step4 debug filters.
    manual_stems : sequence of str or None
        Optional user-selected recording identifiers.
    include_bad : bool
        Include metadata rows marked ``is_bad``.
    annotations_path, primary_qc_path : pathlib.Path or None
        Optional existing behavior annotation and legacy primary-QC CSV paths.
    segment_sec : float
        Full REST candidate-window duration, in seconds.
    auto_low_good : bool
        Enable legacy low-GOOD selection when ``review_all`` is false.
    max_good_for_auto : int
        Threshold on legacy primary-QC GOOD window count.
    use_analysis_selection : bool
        Opt into cohort filtering configured under ``analysis_selection``.
    review_all : bool
        Select all eligible manually annotated REST recordings when true.
    
    Returns
    -------
    dataframe : pandas.DataFrame
        Eligible Step4 stems with usable parent REST intervals.
    config : dict
        Loaded project configuration.
    paths : dict[str, pathlib.Path]
        Standard project output paths.
    annotations_path, primary_qc_path, rescue_csv : pathlib.Path
        Resolved input and output CSV locations.
    
    Notes
    -----
    By default this tool excludes states listed in
    ``rest_qc.anesthesia_states`` (default ``['Iso']``), requires existing
    processed Zarr, and does not apply ``analysis_selection`` cohort filters.
    It does not alter the behavior annotation source CSV."""
    dataframe, config, paths = build_step4_index(
        config_path,
        include_bad=include_bad,
        filters=filters if filters is not None else {},
    )
    # This review excludes states listed in rest_qc.anesthesia_states.
    # Experimental-group selection is optional; explicit CLI filters still apply.
    anesthesia_states = {
        safe_str(s).strip().lower()
        for s in cfg_get(config, "rest_qc", "anesthesia_states", default=["Iso"])
    }
    if "state" not in dataframe.columns:
        raise KeyError("Step4 index has no state column: cannot omit anesthesia from REST QC")
    dataframe = dataframe[~dataframe["state"].map(
        lambda s: safe_str(s).strip().lower() in anesthesia_states
    )].copy()
    if "processed_zarr_exists" in dataframe.columns:
        dataframe = dataframe[bool_series(dataframe["processed_zarr_exists"])].copy()
    else:
        dataframe = dataframe[
            dataframe["processed_zarr_path_resolved"].map(lambda value: Path(str(value)).exists())
        ].copy()

    if use_analysis_selection:
        dataframe = apply_analysis_selection(dataframe, config, require_block=True)
    else:
        dataframe = dataframe.reset_index(drop=True)
        selection = dict(cfg_get(config, "analysis_selection", default={}) or {})
        mapping = build_mouse_group_map(dict(selection.get("experimental_groups", {}) or {}))
        dataframe["experimental_group"] = dataframe.get("mouse_id", pd.Series([""] * len(dataframe))).map(
            lambda v: mapping.get(safe_str(v), "UNASSIGNED")
        )

    ann_path = Path(annotations_path) if annotations_path is not None else default_annotations_path(paths)
    if not ann_path.exists():
        raise FileNotFoundError(f"Behavior annotations CSV not found: {ann_path}")
    ann = read_csv_text(ann_path)
    if ann.empty or "stem" not in ann.columns or "label_intervals_json" not in ann.columns:
        raise RuntimeError("Behavior annotations must contain stem and label_intervals_json")
    ann["stem"] = ann["stem"].map(safe_str)
    ann = ann.drop_duplicates(subset=["stem"], keep="last")
    ann["_behavior_intervals"] = ann["label_intervals_json"].map(parse_intervals_json)

    metadata = dataframe.copy()
    metadata["stem"] = metadata["stem"].map(safe_str)
    ann_cols = ["stem", "_behavior_intervals"]
    renames: dict[str, str] = {}
    for source, target in [
        ("processed_zarr_path_resolved", "annotation_processed_zarr_path"),
        ("raw_zarr_path_resolved", "annotation_raw_zarr_path"),
        ("signal_name", "annotation_signal_name"),
    ]:
        if source in ann.columns:
            ann_cols.append(source)
            renames[source] = target
    merged = metadata.merge(ann[ann_cols].rename(columns=renames), on="stem", how="inner")

    # Only stems with at least one manually annotated REST interval that can contain 60 s.
    width = float(segment_sec)
    def usable_parents(value: Any) -> list[dict[str, Any]]:
        """Keep REST intervals long enough to contain one full candidate chunk."""
        intervals = list(value) if isinstance(value, list) else []
        out = []
        for idx, item in enumerate(intervals):
            if safe_str(item.get("label")).strip().lower() != "rest":
                continue
            t0, t1 = float(item["t0_sec"]), float(item["t1_sec"])
            if t1 - t0 + 1e-9 >= width:
                out.append({
                    "label": "rest",
                    "t0_sec": t0,
                    "t1_sec": t1,
                    "parent_rest_index": idx,
                    "duration_sec": t1 - t0,
                })
        return out
    merged["_rest_parents"] = merged["_behavior_intervals"].map(usable_parents)
    merged = merged[merged["_rest_parents"].map(len).gt(0)].copy()

    qc_path = Path(primary_qc_path) if primary_qc_path is not None else default_primary_qc_path(paths, width)
    if auto_low_good and not review_all and not qc_path.exists():
        raise FileNotFoundError(
            f"Primary QC CSV not found: {qc_path}. "
            "Use --no-auto-low-good if you want to open only manually supplied --stems."
        )
    # No legacy primary-QC dependency in the default combined review mode.
    good_counts = good_counts_by_stem(qc_path) if not review_all else {}
    merged["n_good_primary_qc"] = (np.nan if review_all else
                                   merged["stem"].map(lambda s: int(good_counts.get(safe_str(s), 0))))

    manual = {safe_str(s).strip() for s in (manual_stems or []) if safe_str(s).strip()}
    merged["manual_requested"] = merged["stem"].isin(manual)
    if review_all:
        # Combined QC/Rescue: any annotated REST >=60s is eligible on first pass.
        # --stems/--stem narrows the view instead of silently relying on primary QC.
        merged["auto_low_good"] = False
        merged["rescue_reason"] = "combined_review"
        if manual:
            merged = merged[merged["manual_requested"]].copy()
    else:
        merged["auto_low_good"] = merged["n_good_primary_qc"].le(int(max_good_for_auto)) if auto_low_good else False
        merged["rescue_reason"] = np.where(
            merged["manual_requested"] & merged["auto_low_good"],
            "manual + low_good",
            np.where(merged["manual_requested"], "manual", "low_good"),
        )
        merged = merged[merged["manual_requested"] | merged["auto_low_good"]].copy()

    if manual:
        available = set(metadata["stem"].map(safe_str))
        missing = sorted(manual - available)
        if missing:
            print("[WARN] Manual stems not found after current Step4/analysis_selection filters:", missing)

    sort_cols = [c for c in ["experimental_group", "mouse_id", "exp_day", "stem"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols)
    merged = merged.reset_index(drop=True)
    if merged.empty:
        raise RuntimeError("No rescue stems remain after filters / analysis_selection / REST >= segment length.")

    rescue_root = default_rescue_root(paths, width)
    rescue_csv = rescue_root / "rest_segment_rescue_review.csv"
    return merged, config, paths, ann_path, qc_path, rescue_csv


# =============================================================================
# Rescue GUI
# =============================================================================


class RestSegmentReviewController(QtWidgets.QWidget):
    """Manage the Napari combined QC/rescue workflow for candidate REST windows."""
    def __init__(
        self,
        *,
        viewer: napari.Viewer,
        dataframe: pd.DataFrame,
        config: dict[str, Any],
        paths: dict[str, Path],
        annotations_path: Path,
        rescue_csv: Path,
        signal_name: str,
        load_raw: bool,
        segment_sec: float,
        map_percentiles: tuple[float, float],
        contrast_time_chunk: int,
        contrast_max_values: int,
        trace_gaussian_sigma_frames: float,
        map_smoothing_sigma_px: float,
        annotation_time_unit: str,
        contour_width: int,
        start_stem: str | None,
    ) -> None:
        """Initialize the Napari review controller for candidate REST windows.
        
        Parameters
        ----------
        viewer : napari.Viewer
            Existing viewer to which QC image layers and the control dock are added.
        dataframe : pandas.DataFrame
            Eligible Step4 recordings and parent REST intervals.
        config : dict
            Project YAML configuration, including ``raw.fps``.
        paths : dict[str, pathlib.Path]
            Standard Step4 and project output locations.
        annotations_path, rescue_csv : pathlib.Path
            Read-only behavior annotation file and writable rescue/QC CSV.
        signal_name : str
            Preferred HbT-like variable in processed Zarr.
        load_raw : bool
            Show matching raw-wavelength frames during QC.
        segment_sec : float
            Length of one candidate interval, in seconds.
        map_percentiles : tuple[float, float]
            Display-only map contrast bounds, in percent.
        contrast_time_chunk, contrast_max_values : int
            Controls memory use while computing map contrast.
        trace_gaussian_sigma_frames : float
            Display-only temporal smoothing sigma, in optical frames.
        map_smoothing_sigma_px : float
            Display-only spatial smoothing sigma, in pixels.
        annotation_time_unit : str
            Interpret annotation boundaries as seconds, auto, or frames.
        contour_width : int
            Width of manual overlay contours, in pixels.
        start_stem : str or None
            Optional initial recording identifier.
        
        Notes
        -----
        Writes decisions and edit masks separately from
        ``behavior_review_annotations.csv``. The window may be moved within its
        parent REST interval while the decision remains attached to the
        original canonical candidate slot."""
        super().__init__()
        self.viewer = viewer
        self.df = dataframe.reset_index(drop=True)
        self.config = config
        self.paths = paths
        self.annotations_path = Path(annotations_path)
        self.rescue_csv = Path(rescue_csv)
        self.rescue_root = self.rescue_csv.parent
        self.mask_root = self.rescue_root / "masks"
        self.mask_root.mkdir(parents=True, exist_ok=True)
        self.requested_signal = str(signal_name or "dcHbT")
        self.load_raw = bool(load_raw)
        self.segment_sec = float(segment_sec)
        self.map_percentiles = tuple(map(float, map_percentiles))
        self.contrast_time_chunk = max(1, int(contrast_time_chunk))
        self.contrast_max_values = int(contrast_max_values)
        if not np.isfinite(trace_gaussian_sigma_frames) or trace_gaussian_sigma_frames < 0:
            raise ValueError("Temporal display Gaussian sigma must be finite and >= 0")
        if not np.isfinite(map_smoothing_sigma_px) or map_smoothing_sigma_px < 0:
            raise ValueError("Spatial display Gaussian sigma must be finite and >= 0")
        self.trace_gaussian_sigma_frames = float(trace_gaussian_sigma_frames)
        self.map_smoothing_sigma_px = float(map_smoothing_sigma_px)
        # If not passed on CLI, the checkboxes can still activate sigma=1.
        self._requested_trace_sigma_frames = float(trace_gaussian_sigma_frames) if trace_gaussian_sigma_frames > 0 else 1.0
        self._requested_map_sigma_px = float(map_smoothing_sigma_px) if map_smoothing_sigma_px > 0 else 1.0
        self.annotation_time_unit = str(annotation_time_unit or "seconds")
        self.contour_width = max(0, int(contour_width))

        self.stem_index = 0
        if start_stem:
            hits = np.flatnonzero(self.df["stem"].astype(str).eq(str(start_stem)).to_numpy())
            if hits.size:
                self.stem_index = int(hits[0])

        self.ds: xr.Dataset | None = None
        self.raw: xr.DataArray | None = None
        self.current_signal = ""
        self.brain_mask = np.zeros((1, 1), dtype=bool)
        self.bad_mask = np.zeros((1, 1), dtype=bool)
        self.bregma_mask = np.zeros((1, 1), dtype=bool)
        self.brain_signal_mask = np.zeros((1, 1), dtype=bool)
        self.hbt_layer: Any | None = None
        self.shadow_layer: Any | None = None
        self.activity_roi_layer: Any | None = None
        self.shadow_mask_buffer: np.ndarray | None = None
        self.activity_roi_vertices_buffer: np.ndarray | None = None
        self.current_var: xr.DataArray | None = None
        # Current canonical 60-s candidate within the selected parent REST.
        # The displayed/accepted window may be shifted, but its decision remains
        # attached to this canonical slot so every candidate is reviewed once.
        self.current_chunk_index = 0
        self.raw_layers: dict[str, Any] = {}
        self.mask_layers: dict[str, Any] = {}
        self.segment_i0 = 0
        self.segment_i1 = -1
        self.segment_times = np.asarray([], dtype=float)
        self.segment_plot_times = np.asarray([], dtype=float)
        self.segment_var: xr.DataArray | None = None
        self.segment_display_var: xr.DataArray | None = None
        self.parent_trace_times = np.asarray([], dtype=float)
        self.parent_trace_values = np.asarray([], dtype=float)
        self.parent_roi_trace_values = np.asarray([], dtype=float)
        self._loading = False
        self._layer_callbacks: list[tuple[Any, Any]] = []
        self._time_cursor_line: Any | None = None
        self._saved_rows = read_latest_by_key(self.rescue_csv, "stem")
        # Only cache arrays belonging to the CURRENT stem (and signal grid).
        self._stem_masks_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        self._parent_trace_cache_key: tuple[Any, ...] | None = None
        # The 3-D Napari time slider belongs to the viewer's Qt widgets.  Keep
        # the layers alive while moving between windows of the same stem.
        self._visible_stem: str | None = None
        self._loaded_window_key: tuple[Any, ...] | None = None

        self._build_ui()
        try:
            self.viewer.dims.events.current_step.connect(self._on_napari_time_changed)
        except Exception:
            pass
        self._populate_stems()
        self.load_current_stem()

    @property
    def row(self) -> pd.Series:
        """Return the currently selected Step4 metadata row."""
        return self.df.iloc[self.stem_index]

    @property
    def stem(self) -> str:
        """Return the current recording identifier."""
        return safe_str(self.row["stem"])

    def parent_intervals(self) -> list[dict[str, Any]]:
        """Return eligible parent REST intervals for the current recording."""
        value = self.row.get("_rest_parents", [])
        return list(value) if isinstance(value, list) else []

    def current_parent(self) -> dict[str, Any] | None:
        """Return the currently selected parent REST interval."""
        intervals = self.parent_intervals()
        idx = self.parent_combo.currentIndex()
        if idx < 0 or idx >= len(intervals):
            return None
        return intervals[idx]

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        """Construct the Napari dock controls and connect their callbacks."""
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(5)

        self.title_label = QtWidgets.QLabel("REST review / 60-s QC")
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 11pt;")
        outer.addWidget(self.title_label)

        row = QtWidgets.QHBoxLayout()
        self.prev_stem_btn = QtWidgets.QPushButton("◀ Stem")
        self.stem_combo = QtWidgets.QComboBox()
        self.next_stem_btn = QtWidgets.QPushButton("Stem ▶")
        row.addWidget(self.prev_stem_btn)
        row.addWidget(self.stem_combo, stretch=1)
        row.addWidget(self.next_stem_btn)
        outer.addLayout(row)

        parent_row = QtWidgets.QHBoxLayout()
        self.parent_combo = QtWidgets.QComboBox()
        parent_row.addWidget(QtWidgets.QLabel("Annotated REST"))
        parent_row.addWidget(self.parent_combo, stretch=1)
        outer.addLayout(parent_row)

        candidate_row = QtWidgets.QHBoxLayout()
        self.prev_chunk_btn = QtWidgets.QPushButton("◀ 60 s")
        self.chunk_label = QtWidgets.QLabel()
        self.next_chunk_btn = QtWidgets.QPushButton("60 s ▶")
        candidate_row.addWidget(self.prev_chunk_btn)
        candidate_row.addWidget(self.chunk_label, stretch=1)
        candidate_row.addWidget(self.next_chunk_btn)
        outer.addLayout(candidate_row)

        window_box = QtWidgets.QGroupBox(f"Current {self.segment_sec:g} s candidate")
        wl = QtWidgets.QVBoxLayout(window_box)
        start_row = QtWidgets.QHBoxLayout()
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(1.0)
        self.start_spin.setKeyboardTracking(False)
        self.window_label = QtWidgets.QLabel()
        start_row.addWidget(QtWidgets.QLabel("start (s)"))
        start_row.addWidget(self.start_spin)
        start_row.addWidget(self.window_label, stretch=1)
        wl.addLayout(start_row)
        shifts = QtWidgets.QHBoxLayout()
        for text, delta in (("−5 s", -5.0), ("−1 s", -1.0), ("+1 s", 1.0), ("+5 s", 5.0)):
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda checked=False, d=delta: self.shift_window(d))
            shifts.addWidget(btn)
        wl.addLayout(shifts)
        outer.addWidget(window_box)

        display_row = QtWidgets.QHBoxLayout()
        self.signal_combo = QtWidgets.QComboBox()
        self.raw_checkbox = QtWidgets.QCheckBox("Show raw")
        self.raw_checkbox.setChecked(False)
        display_row.addWidget(QtWidgets.QLabel("HbT"))
        display_row.addWidget(self.signal_combo, stretch=1)
        display_row.addWidget(self.raw_checkbox)
        outer.addLayout(display_row)

        smooth_row = QtWidgets.QHBoxLayout()
        self.trace_gaussian_checkbox = QtWidgets.QCheckBox(
            f"Trace Gaussian σ={self._requested_trace_sigma_frames:g} frame(s) (display only)"
        )
        self.trace_gaussian_checkbox.setChecked(self.trace_gaussian_sigma_frames > 0)
        self.map_smoothing_checkbox = QtWidgets.QCheckBox(
            f"Map Gaussian σ={self._requested_map_sigma_px:g} px (display only)"
        )
        self.map_smoothing_checkbox.setChecked(self.map_smoothing_sigma_px > 0)
        smooth_row.addWidget(self.trace_gaussian_checkbox)
        smooth_row.addWidget(self.map_smoothing_checkbox)
        smooth_row.addStretch(1)
        outer.addLayout(smooth_row)

        roi_row = QtWidgets.QHBoxLayout()
        self.activate_roi_btn = QtWidgets.QPushButton("Draw activity ROI")
        self.clear_roi_btn = QtWidgets.QPushButton("Clear activity ROI")
        roi_row.addWidget(self.activate_roi_btn)
        roi_row.addWidget(self.clear_roi_btn)
        outer.addLayout(roi_row)

        mask_row = QtWidgets.QHBoxLayout()
        self.activate_shadow_btn = QtWidgets.QPushButton("Draw/edit shadow mask")
        self.clear_shadow_btn = QtWidgets.QPushButton("Clear shadow mask")
        mask_row.addWidget(self.activate_shadow_btn)
        mask_row.addWidget(self.clear_shadow_btn)
        outer.addLayout(mask_row)

        decision_box = QtWidgets.QGroupBox("REST decision (shift is optional)")
        dl = QtWidgets.QHBoxLayout(decision_box)
        self.use_btn = QtWidgets.QPushButton("USE THIS 60 s")
        self.reject_btn = QtWidgets.QPushButton("REJECT THIS 60 s")
        self.unclear_btn = QtWidgets.QPushButton("UNCLEAR THIS 60 s")
        dl.addWidget(self.use_btn)
        dl.addWidget(self.reject_btn)
        dl.addWidget(self.unclear_btn)
        outer.addWidget(decision_box)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setWordWrap(True)
        outer.addWidget(self.info_label)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        try:
            screen = QtWidgets.QApplication.primaryScreen()
            screen_h = int(screen.availableGeometry().height()) if screen is not None else 900
        except Exception:
            screen_h = 900
        # Do not force the Napari/Qt main window taller than the monitor.
        # A tall minimum canvas previously triggered repeated Windows geometry
        # warnings on 1080p displays (minimum dock height > available area).
        self.canvas.setMinimumHeight(240)
        outer.addWidget(self.toolbar)
        outer.addWidget(self.canvas, stretch=10)

        self.prev_stem_btn.clicked.connect(lambda: self.change_stem(-1))
        self.next_stem_btn.clicked.connect(lambda: self.change_stem(1))
        self.stem_combo.currentIndexChanged.connect(self._stem_changed)
        self.parent_combo.currentIndexChanged.connect(self._parent_changed)
        self.prev_chunk_btn.clicked.connect(lambda: self.change_chunk(-1))
        self.next_chunk_btn.clicked.connect(lambda: self.change_chunk(1))
        self.start_spin.editingFinished.connect(self._window_start_edited)
        self.signal_combo.currentIndexChanged.connect(self._signal_changed)
        self.raw_checkbox.toggled.connect(self._apply_raw_visibility)
        self.trace_gaussian_checkbox.toggled.connect(self._on_trace_gaussian_toggled)
        self.map_smoothing_checkbox.toggled.connect(self._on_map_smoothing_toggled)
        self.activate_roi_btn.clicked.connect(self._activate_activity_roi)
        self.clear_roi_btn.clicked.connect(self._clear_activity_roi)
        self.activate_shadow_btn.clicked.connect(self._activate_shadow)
        self.clear_shadow_btn.clicked.connect(self._clear_shadow)
        self.use_btn.clicked.connect(lambda: self.save_decision("use"))
        self.reject_btn.clicked.connect(lambda: self.save_decision("reject"))
        self.unclear_btn.clicked.connect(lambda: self.save_decision("unclear"))

        self.setMinimumWidth(520)
        self.resize(760, min(780, max(550, screen_h - 60)))

    # ---------------- navigation / saved state ----------------

    def _populate_stems(self) -> None:
        """Populate the recording selection control."""
        self.stem_combo.blockSignals(True)
        self.stem_combo.clear()
        for _, row in self.df.iterrows():
            self.stem_combo.addItem(
                f"{safe_str(row.get('stem'))} | {safe_str(row.get('experimental_group'))} | "
                f"{safe_str(row.get('mouse_id'))} | {safe_str(row.get('exp_day'))} | "
                f"GOOD={safe_str(row.get('n_good_primary_qc'))} | {safe_str(row.get('rescue_reason'))}",
                safe_str(row.get("stem")),
            )
        self.stem_combo.setCurrentIndex(self.stem_index)
        self.stem_combo.blockSignals(False)

    def _populate_parents(self) -> None:
        """Populate the parent REST-interval selection control."""
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        for i, item in enumerate(self.parent_intervals()):
            self.parent_combo.addItem(
                f"{i+1}/{len(self.parent_intervals())} | REST {float(item['t0_sec']):.3f}–{float(item['t1_sec']):.3f} s "
                f"({float(item['duration_sec']):.1f} s)"
            )
        self.parent_combo.setCurrentIndex(0 if self.parent_combo.count() else -1)
        self.parent_combo.blockSignals(False)

    def _saved_row_for_stem(self) -> pd.Series | None:
        """Return the latest saved decision row for the given recording."""
        df = self._saved_rows
        if df.empty or "stem" not in df.columns:
            return None
        sub = df[df["stem"].map(safe_str).eq(self.stem)]
        return None if sub.empty else sub.iloc[-1]

    def change_stem(self, delta: int) -> None:
        """Navigate to another recording without changing the source annotations."""
        new = int(np.clip(self.stem_index + int(delta), 0, len(self.df) - 1))
        if new == self.stem_index:
            return
        self.stem_index = new
        self.stem_combo.blockSignals(True)
        self.stem_combo.setCurrentIndex(new)
        self.stem_combo.blockSignals(False)
        try:
            self.load_current_stem()
        except Exception as exc:
            self._report_gui_error("Could not load REST stem", exc)

    def _stem_changed(self, index: int) -> None:
        """Refresh the data and controls after selecting a recording."""
        if self._loading or index < 0 or index >= len(self.df):
            return
        self.stem_index = int(index)
        try:
            self.load_current_stem()
        except Exception as exc:
            self._report_gui_error("Could not load REST stem", exc)

    def _n_chunks_for_parent(self, parent: dict[str, Any] | None = None) -> int:
        """Count full candidate slots within one parent REST interval."""
        parent = parent if parent is not None else self.current_parent()
        if parent is None:
            return 0
        duration = max(0.0, float(parent["t1_sec"]) - float(parent["t0_sec"]))
        return int(np.floor((duration + 1e-9) / self.segment_sec))

    def _current_slot(self) -> dict[str, Any] | None:
        """Return identity and nominal bounds of the selected canonical REST slot.
        
        Notes
        -----
        Window shifts do not change the slot identifier used for decisions."""
        parent = self.current_parent()
        if parent is None:
            return None
        n = self._n_chunks_for_parent(parent)
        if n <= 0:
            return None
        ci = int(np.clip(self.current_chunk_index, 0, n - 1))
        parent_idx = int(parent.get("parent_rest_index", self.parent_combo.currentIndex()))
        nominal_t0 = float(parent["t0_sec"]) + ci * self.segment_sec
        nominal_t1 = nominal_t0 + self.segment_sec
        return {
            "slot_id": f"{self.stem}|parent{parent_idx}|chunk{ci}",
            "parent_rest_index": parent_idx,
            "chunk_index_in_parent": ci,
            "nominal_t0_sec": nominal_t0,
            "nominal_t1_sec": nominal_t1,
        }

    def _all_slots(self) -> list[dict[str, Any]]:
        """Enumerate canonical candidate slots for the current recording.
        
        Returns
        -------
        list[dict[str, Any]]
            Stable slot IDs and nominal start/end times for all parent REST periods."""
        slots: list[dict[str, Any]] = []
        for combo_idx, parent in enumerate(self.parent_intervals()):
            n = self._n_chunks_for_parent(parent)
            parent_idx = int(parent.get("parent_rest_index", combo_idx))
            for ci in range(n):
                t0 = float(parent["t0_sec"]) + ci * self.segment_sec
                slots.append({
                    "slot_id": f"{self.stem}|parent{parent_idx}|chunk{ci}",
                    "parent_combo_index": combo_idx,
                    "parent_rest_index": parent_idx,
                    "chunk_index_in_parent": ci,
                    "nominal_t0_sec": t0,
                    "nominal_t1_sec": t0 + self.segment_sec,
                })
        return slots

    def _decision_records(self) -> list[dict[str, Any]]:
        """Read the saved candidate decisions for the current stem."""
        saved = self._saved_row_for_stem()
        if saved is None:
            return []
        text = safe_str(saved.get("segment_decisions_json")).strip()
        if not text:
            return []
        try:
            value = json.loads(text)
            return [dict(x) for x in value if isinstance(x, dict)] if isinstance(value, list) else []
        except Exception:
            return []

    def _decision_map(self) -> dict[str, dict[str, Any]]:
        """Map stored decisions by canonical slot identifier."""
        out: dict[str, dict[str, Any]] = {}
        for item in self._decision_records():
            sid = safe_str(item.get("slot_id")).strip()
            if sid:
                out[sid] = item
        return out

    def _set_candidate(self, parent_combo_index: int, chunk_index: int, *, start_override: float | None = None, load: bool = True) -> None:
        """Select a canonical REST candidate and optionally load its window."""
        parents = self.parent_intervals()
        if not parents:
            return
        pi = int(np.clip(parent_combo_index, 0, len(parents) - 1))
        parent = parents[pi]
        n = self._n_chunks_for_parent(parent)
        if n <= 0:
            return
        ci = int(np.clip(chunk_index, 0, n - 1))
        self.current_chunk_index = ci
        self.parent_combo.blockSignals(True)
        self.parent_combo.setCurrentIndex(pi)
        self.parent_combo.blockSignals(False)
        lo = float(parent["t0_sec"])
        hi = float(parent["t1_sec"]) - self.segment_sec
        self._set_start_limits(lo, hi)
        nominal = lo + ci * self.segment_sec
        # When revisiting an already rated slot, restore the actual shifted window
        # that was used for that slot rather than snapping back to the nominal start.
        if start_override is None:
            slot_id = f"{self.stem}|parent{int(parent.get('parent_rest_index', pi))}|chunk{ci}"
            previous = self._decision_map().get(slot_id)
            if previous is not None:
                try:
                    start_override = float(previous.get("t0_sec"))
                except Exception:
                    start_override = None
        value = nominal if start_override is None else float(np.clip(start_override, lo, max(lo, hi)))
        self.start_spin.setValue(value)
        self._update_chunk_label()
        if load:
            self.load_selected_window()

    def _parent_changed(self) -> None:
        """Update candidate selection after changing the parent REST interval."""
        if self._loading:
            return
        idx = self.parent_combo.currentIndex()
        if idx < 0:
            return
        self._set_candidate(idx, 0, load=True)

    def change_chunk(self, delta: int) -> None:
        """Navigate among full candidate windows, including across parent intervals."""
        parent = self.current_parent()
        if parent is None:
            return
        n = self._n_chunks_for_parent(parent)
        new = self.current_chunk_index + int(delta)
        if 0 <= new < n:
            self._set_candidate(self.parent_combo.currentIndex(), new, load=True)
            return
        # Allow manual navigation across parent REST boundaries.
        pi = self.parent_combo.currentIndex()
        if new >= n and pi + 1 < self.parent_combo.count():
            self._set_candidate(pi + 1, 0, load=True)
        elif new < 0 and pi > 0:
            prev_parent = self.parent_intervals()[pi - 1]
            self._set_candidate(pi - 1, max(0, self._n_chunks_for_parent(prev_parent) - 1), load=True)

    def _set_start_limits(self, lo: float, hi: float) -> None:
        """Constrain the shiftable-window start to the parent REST interval."""
        hi = max(lo, hi)
        self.start_spin.setRange(float(lo), float(hi))

    def shift_window(self, delta: float) -> None:
        """Shift the current review window by the requested time in seconds."""
        self.start_spin.setValue(self.start_spin.value() + float(delta))
        self.load_selected_window()

    def _window_request_key(self) -> tuple[Any, ...]:
        """Identity of the *display request*, not a decision or saved CSV row."""
        return (
            self.stem, safe_str(self.signal_combo.currentData() or self.current_signal),
            int(self.parent_combo.currentIndex()), int(self.current_chunk_index),
            float(self.start_spin.value()),
        )

    def _window_start_edited(self) -> None:
        # NEVER remove/recreate the Napari 3-D layers synchronously inside a
        # QDoubleSpinBox editingFinished / keyboard-focus event. On Windows Qt
        # can still be inside napari.qt_dims_slider.slider_focused_listener.
        """Schedule safe window loading after the user edits the start time."""
        if self._loading:
            return
        request = self._window_request_key()
        QtCore.QTimer.singleShot(0, lambda key=request: self._apply_manual_start_edit(key))

    def _apply_manual_start_edit(self, request: tuple[Any, ...]) -> None:
        # Ignore a queued event if the user has already changed stem/slot/value.
        # A second editingFinished for the same number is deduplicated in the
        # window loader, so entering a number + losing focus is safe.
        """Apply a queued window edit only if its request is still current."""
        if not self._loading and self.ds is not None and request == self._window_request_key():
            self.load_selected_window()

    def _update_chunk_label(self) -> None:
        """Refresh the active candidate slot and window labels."""
        parent = self.current_parent()
        n = self._n_chunks_for_parent(parent)
        slot = self._current_slot()
        decisions = self._decision_map()
        status = "unreviewed"
        if slot is not None and slot["slot_id"] in decisions:
            status = safe_str(decisions[slot["slot_id"]].get("decision")) or "unreviewed"
        self.chunk_label.setText(
            f"60-s candidate {self.current_chunk_index + 1}/{max(n, 1)} | decision: {status}"
        )

    # ---------------- data loading ----------------

    def _close_data(self) -> None:
        """Close open Zarr resources and reset cached display data."""
        if self.ds is not None:
            try: self.ds.close()
            except Exception: pass
        if self.raw is not None:
            try: self.raw.close()
            except Exception: pass
        self.ds = None
        self.raw = None

    def _disconnect_layer_callbacks(self) -> None:
        """Disconnect temporary Napari layer event handlers."""
        for emitter, callback in self._layer_callbacks:
            try: emitter.disconnect(callback)
            except Exception: pass
        self._layer_callbacks.clear()

    def _connect(self, emitter: Any, callback: Any) -> None:
        """Attach an event handler to a Napari layer."""
        emitter.connect(callback)
        self._layer_callbacks.append((emitter, callback))

    def load_current_stem(self) -> None:
        """Load Step4 signals and saved review state for the selected stem."""
        self._loading = True
        self._disconnect_layer_callbacks()
        self.viewer.layers.clear()
        # CRITICAL: old stem's Label layer must not survive into the new stem.
        # load_selected_window() calls _capture_edit_buffers() and would otherwise
        # copy the PREVIOUS animal's shadow into the next one's blank buffer.
        self.hbt_layer = None
        self.shadow_layer = None
        self.activity_roi_layer = None
        self.raw_layers = {}
        self.mask_layers = {}
        self.segment_var = None
        self.segment_display_var = None
        self._stem_masks_cache.clear()
        self._parent_trace_cache_key = None
        self._visible_stem = None
        self._loaded_window_key = None
        self._close_data()
        self.shadow_mask_buffer = None
        self.activity_roi_vertices_buffer = None
        self.current_var = None
        try:
            row = self.row
            current_path = Path(str(row["processed_zarr_path_resolved"]))
            # Behavior annotations are independent of numerical Step4 reruns.
            # Always display the currently selected Step4 Zarr for this stem.
            processed_path = current_path
            if not processed_path.exists():
                raise FileNotFoundError(f"Processed zarr not found: {processed_path}")
            try:
                self.ds = xr.open_zarr(processed_path, chunks="auto", consolidated=False, mask_and_scale=False)
            except Exception:
                self.ds = xr.open_zarr(processed_path, chunks=None, consolidated=False, mask_and_scale=False)

            choices = candidate_hbt_signals(self.ds)
            if not choices:
                raise RuntimeError(f"No HbT-like time,y,x signal in {processed_path}")
            self.signal_combo.blockSignals(True)
            self.signal_combo.clear()
            for name in choices:
                self.signal_combo.addItem(name, name)
            idx = self.signal_combo.findData(self.requested_signal)
            self.signal_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.signal_combo.blockSignals(False)
            self.current_signal = safe_str(self.signal_combo.currentData())

            if self.load_raw:
                raw_path = Path(select_raw_zarr_path_for_processing(row, self.config))
                if raw_path.exists():
                    self.raw = open_raw_zarr(raw_path, self.config).transpose("time", "wavelength", "y", "x")

            self._populate_parents()
            # Resume at the first canonical 60-s candidate without a decision.
            # If all candidates were already rated, reopen the first one for review/editing.
            slots = self._all_slots()
            if not slots:
                raise RuntimeError("No 60-s candidates available for rescue")
            decided = self._decision_map()
            target = next((s for s in slots if s["slot_id"] not in decided), slots[0])
            saved_start = None
            if target["slot_id"] in decided:
                try:
                    saved_start = float(decided[target["slot_id"]].get("t0_sec"))
                except Exception:
                    saved_start = None
            self._set_candidate(
                int(target["parent_combo_index"]),
                int(target["chunk_index_in_parent"]),
                start_override=saved_start,
                load=False,
            )
            self._loading = False
            self.load_selected_window()
        except Exception:
            self._loading = False
            raise

    def _signal_changed(self) -> None:
        """Refresh the current REST window after changing the HbT signal."""
        if self._loading:
            return
        self.current_signal = safe_str(self.signal_combo.currentData())
        self.load_selected_window()

    def _current_window(self) -> tuple[float, float]:
        """Return the shifted candidate interval boundaries in seconds."""
        t0 = float(self.start_spin.value())
        return t0, t0 + self.segment_sec

    def _activity_roi_vertices(self) -> np.ndarray | None:
        """Read the current editable activity ROI polygon vertices."""
        if self.activity_roi_layer is not None:
            try:
                data = list(self.activity_roi_layer.data)
                if data:
                    pts = np.asarray(data[0], dtype=float)
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        return pts[:, -2:]
            except Exception:
                pass
        if self.activity_roi_vertices_buffer is not None:
            pts = np.asarray(self.activity_roi_vertices_buffer, dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                return pts[:, -2:]
        return None

    def _activity_roi_mask(self) -> np.ndarray | None:
        """Convert the current activity ROI into a boolean spatial mask."""
        pts = self._activity_roi_vertices()
        if pts is None:
            return None
        h, w = self.brain_signal_mask.shape
        poly = np.round(pts[:, [1, 0]]).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 1)
        out = mask.astype(bool) & self.brain_signal_mask
        return out if np.any(out) else None

    def _build_parent_trace(self, var: xr.DataArray, parent: dict[str, Any]) -> None:
        # A 160-s REST parent may contain multiple 60-s slots. Its whole-brain
        # and ROI traces are identical for all slots until signal/ROI changes.
        # Avoid re-reading the entire parent every time the user clicks Next.
        """Build whole-brain and editable-ROI traces for a parent REST interval.
        
        Notes
        -----
        The traces aid visual review only. Their optional smoothing does not
        change accepted interval boundaries or stored Step4 signals."""
        pts = self._activity_roi_vertices()
        roi_key = None if pts is None else (pts.shape, np.asarray(pts, dtype=np.float64).tobytes())
        key = (self.stem, safe_str(var.name), float(parent["t0_sec"]),
               float(parent["t1_sec"]), roi_key)
        if key == self._parent_trace_cache_key:
            return
        res = resolve_annotation_interval(
            var,
            float(parent["t0_sec"]),
            float(parent["t1_sec"]),
            self.config,
            requested_unit=self.annotation_time_unit,
        )
        i0, i1 = int(res["i0"]), int(res["i1"])
        parent_var = var.isel(time=slice(i0, i1 + 1))
        mask_da = xr.DataArray(
            self.brain_signal_mask,
            dims=("y", "x"),
            coords={"y": parent_var.coords["y"], "x": parent_var.coords["x"]},
        )
        trace = parent_var.where(mask_da).mean(dim=("y", "x"), skipna=True).compute()
        self.parent_trace_values = np.asarray(trace.values, dtype=float)

        self.parent_roi_trace_values = np.asarray([], dtype=float)
        roi_mask = self._activity_roi_mask()
        if roi_mask is not None:
            roi_da = xr.DataArray(
                roi_mask,
                dims=("y", "x"),
                coords={"y": parent_var.coords["y"], "x": parent_var.coords["x"]},
            )
            roi_trace = parent_var.where(roi_da).mean(dim=("y", "x"), skipna=True).compute()
            self.parent_roi_trace_values = np.asarray(roi_trace.values, dtype=float)

        fps = annotation_fps(self.config)
        frames = np.arange(i0, i1 + 1, dtype=float)
        self.parent_trace_times = frames / fps
        self._parent_trace_cache_key = key

    def _report_gui_error(self, title: str, exc: Exception) -> None:
        # Exceptions raised inside Qt signal handlers may otherwise terminate
        # the interpreter on some PyQt versions without a useful traceback.
        """Report a review error in Napari and standard output."""
        traceback.print_exc()
        print(f"[REST review ERROR] {title}: {exc}", flush=True)
        try:
            self.viewer.status = f"{title}: {exc}"
        except Exception:
            pass
        try:
            QtWidgets.QMessageBox.warning(self, title, str(exc))
        except Exception:
            pass

    def load_selected_window(self) -> None:
        """Load the selected REST review window with GUI error handling."""
        try:
            self._load_selected_window_impl()
        except Exception as exc:
            self._report_gui_error("Could not load REST 60-s window", exc)

    def _load_selected_window_impl(self) -> None:
        """Load the selected candidate window and matching Napari display layers.
        
        Notes
        -----
        Annotation boundaries are converted using ``raw.fps`` unless frame
        interpretation was explicitly requested. Segment-centered HbT display,
        raw layers and QC masks are for inspection, not signal processing."""
        if self.ds is None or self.current_parent() is None:
            return
        request_key = self._window_request_key()
        if request_key == self._loaded_window_key and self.hbt_layer is not None:
            # QDoubleSpinBox can emit editingFinished on Return *and* focus-out.
            # Identical 60-s windows must never delete/rebuild Napari's slider.
            return
        self._capture_edit_buffers()
        self._loading = True
        try:
            signal = safe_str(self.signal_combo.currentData() or self.current_signal)
            var = self.ds[signal].transpose("time", "y", "x")
            self.current_var = var
            image_shape = (int(var.sizes["y"]), int(var.sizes["x"]))
            cached_masks = self._stem_masks_cache.get(image_shape)
            if cached_masks is None:
                # Read and resize Step2 masks only once per stem + processed grid.
                masks = load_masks_for_row(self.row, self.config, image_shape)
                brain_mask = np.asarray(masks["brain"].values, dtype=bool)
                bad_mask = np.asarray(masks["bad"].values, dtype=bool)
                bregma_mask = np.asarray(masks["bregma"].values, dtype=bool)
                if "raw_valid" not in self.ds:
                    raise KeyError("Step4 processed Zarr has no raw_valid")
                raw_valid = np.asarray(self.ds["raw_valid"].transpose("y", "x").values) == 1
                if tuple(raw_valid.shape) != tuple(image_shape):
                    raise ValueError("raw_valid shape differs from selected Hb grid")
                brain_signal_mask = brain_mask & (~bad_mask) & (~bregma_mask) & raw_valid
                cached_masks = (brain_mask, bad_mask, bregma_mask, brain_signal_mask)
                self._stem_masks_cache[image_shape] = cached_masks
            self.brain_mask, self.bad_mask, self.bregma_mask, self.brain_signal_mask = cached_masks
            self._load_saved_stem_buffers_if_needed()

            t0, t1 = self._current_window()
            res = resolve_annotation_interval(var, t0, t1, self.config, requested_unit=self.annotation_time_unit)
            if res["clipped"]:
                raise ValueError(f"60-s REST window outside current processed Zarr: {t0}..{t1}")
            i0, i1 = int(res["i0"]), int(res["i1"])
            expected_n = int(round(self.segment_sec * float(res["fps"])))
            if (i1 - i0 + 1) != expected_n:
                raise ValueError("REST candidate is not a full 60-s sampling window")
            self.segment_i0, self.segment_i1 = i0, i1
            print(f"[REST review] Loading stem={self.stem} window={t0:.3f}-{t1:.3f}s "
                  f"frames={i0}:{i1 + 1} map_sigma={self.map_smoothing_sigma_px:g}", flush=True)
            self.segment_var = var.isel(time=slice(i0, i1 + 1))
            self.segment_display_var = segment_centered_var(self.segment_var)
            fps = float(res["fps"])
            frames = np.arange(i0, i1 + 1, dtype=float)
            self.segment_times = frames / fps
            self.segment_plot_times = (frames - float(i0)) / fps

            display_var = gaussian_spatial_display_stack(
                self.segment_display_var, self.brain_signal_mask, self.map_smoothing_sigma_px
            )
            vmin, vmax, _ = compute_segment_contrast_limits(
                display_var,
                self.brain_signal_mask,
                percentiles=self.map_percentiles,
                time_chunk=self.contrast_time_chunk,
                max_values=self.contrast_max_values,
            )
            # All 60-s slots in one stem use the same time/y/x geometry.  Update
            # the existing layers instead of destroying Napari Qt dims sliders.
            # In particular, never call viewer.layers.clear() from a number edit.
            reuse_layers = (
                self._visible_stem == self.stem
                and self.hbt_layer is not None
                and self.shadow_layer is not None
                and self.activity_roi_layer is not None
                and (self.raw is None or len(self.raw_layers) == int(self.raw.sizes["wavelength"]))
            )
            if reuse_layers:
                # All candidates have the same frame count. Napari keeps the
                # current time index when a layer's data is replaced by a stack
                # of identical shape. Do NOT touch Qt dims sliders here.
                self.hbt_layer.data = display_var.data
                self.hbt_layer.name = f"{signal}_segment_centered"
                self.hbt_layer.contrast_limits = (vmin, vmax)
                if self.raw is not None:
                    self._update_raw_layer_data()
            else:
                self._disconnect_layer_callbacks()
                self.viewer.layers.clear()
                self.raw_layers = {}
                self.mask_layers = {}
                self.hbt_layer = None
                self.shadow_layer = None
                self.activity_roi_layer = None
                self.hbt_layer = self.viewer.add_image(
                    display_var.data,
                    name=f"{signal}_segment_centered",
                    colormap=jet_colormap(),
                    contrast_limits=(vmin, vmax),
                    visible=True,
                )
                self._add_project_masks()
                self._add_existing_manual_contours()
                if self.raw is not None:
                    self._add_raw_layers()
                self._add_shadow_edit_layer()
                self._add_activity_roi_layer()
            self._apply_raw_visibility()
            self._build_parent_trace(var, self.current_parent())
            self.refresh_plot()
            self.viewer.title = f"REST rescue | {self.stem}"
            self._update_info()
            self._visible_stem = self.stem
            self._loaded_window_key = request_key
        finally:
            self._loading = False

    def _add_project_masks(self) -> None:
        """Overlay the project brain, bad, and bregma masks in Napari."""
        for name, mask, opacity in (
            ("brain_mask", self.brain_mask, 0.12),
            ("bad_auto", self.bad_mask, 0.30),
            ("bregma_mask", self.bregma_mask, 0.30),
        ):
            layer = self.viewer.add_labels(mask.astype(np.uint8), name=name, opacity=opacity, visible=False)
            layer.editable = False
            self.mask_layers[name] = layer

    def _load_saved_manual_mask(self, filename: str) -> np.ndarray | None:
        """Load any previously saved per-stem rescue shadow mask."""
        path = manual_review_root(self.paths, self.config) / self.stem / filename
        if not path.exists():
            return None
        try:
            return resize_bool(np.load(path).astype(bool), self.brain_signal_mask.shape)
        except Exception:
            return None

    def _add_existing_manual_contours(self) -> None:
        """Show saved manual exclusion contours over the current image."""
        for filename, opacity in (("shadow_union_manual.npy", 0.55), ("hemorrhage_manual.npy", 0.75)):
            mask = self._load_saved_manual_mask(filename)
            if mask is None or not mask.any():
                continue
            layer = self.viewer.add_labels(mask.astype(np.uint8), name=filename.replace(".npy", ""), opacity=opacity, visible=True)
            layer.editable = False
            try: layer.contour = self.contour_width
            except Exception: pass
            self.mask_layers[filename] = layer

    def _add_raw_layers(self) -> None:
        """Add raw-wavelength inspection layers to the Napari viewer."""
        if self.raw is None:
            return
        raw_n = int(self.raw.sizes.get("time", 0))
        if raw_n <= 0:
            return
        indices = np.arange(self.segment_i0, self.segment_i1 + 1, dtype=int)
        indices = np.clip(indices, 0, raw_n - 1)
        for wi, value in enumerate(self.raw.coords["wavelength"].values):
            wave = normalize_wavelength_label(value)
            channel = self.raw.isel(wavelength=wi, time=indices).transpose("time", "y", "x")
            layer = self.viewer.add_image(channel.data, name=f"raw_{wave}", colormap="gray", visible=False)
            self.raw_layers[wave] = layer

    def _update_raw_layer_data(self) -> None:
        """Keep raw 3-D layers (and their Qt slider) across same-stem windows."""
        if self.raw is None:
            return
        raw_n = int(self.raw.sizes.get("time", 0))
        if raw_n <= 0:
            return
        indices = np.arange(self.segment_i0, self.segment_i1 + 1, dtype=int)
        indices = np.clip(indices, 0, raw_n - 1)
        for wi, value in enumerate(self.raw.coords["wavelength"].values):
            wave = normalize_wavelength_label(value)
            channel = self.raw.isel(wavelength=wi, time=indices).transpose("time", "y", "x")
            self.raw_layers[wave].data = channel.data

    def _mask_path_for_stem(self) -> Path:
        """Return the saved shadow-mask path for the current stem."""
        return self.mask_root / f"{self.stem}_shadow.npy"

    def _roi_mask_path_for_stem(self) -> Path:
        """Return the activity ROI mask path for the current stem."""
        return self.mask_root / f"{self.stem}_activity_roi.npy"

    def _capture_edit_buffers(self) -> None:
        """Capture current per-stem manual editing layers before navigating."""
        if self.shadow_layer is not None:
            try:
                arr = np.asarray(self.shadow_layer.data, dtype=np.uint8)
                if arr.ndim == 2:
                    self.shadow_mask_buffer = (arr > 0).astype(np.uint8)
            except Exception:
                pass
        if self.activity_roi_layer is not None:
            try:
                data = list(self.activity_roi_layer.data)
                if data:
                    pts = np.asarray(data[0], dtype=float)
                    if pts.ndim == 2 and pts.shape[0] >= 3:
                        self.activity_roi_vertices_buffer = pts[:, -2:].copy()
                else:
                    self.activity_roi_vertices_buffer = None
            except Exception:
                pass

    def _load_saved_stem_buffers_if_needed(self) -> None:
        """Restore cached or saved edits for the current recording."""
        saved = self._saved_row_for_stem()
        if self.shadow_mask_buffer is None:
            initial = np.zeros(self.brain_signal_mask.shape, dtype=np.uint8)
            if saved is not None:
                path_text = safe_str(saved.get("shadow_mask_path")).strip()
                if path_text:
                    path = Path(path_text)
                    if path.exists():
                        try:
                            loaded_shadow = np.load(path, allow_pickle=False).astype(bool)
                            if loaded_shadow.shape != self.brain_signal_mask.shape:
                                raise ValueError(f"Saved shadow shape {loaded_shadow.shape} does not match processed grid {self.brain_signal_mask.shape}")
                            initial = loaded_shadow.astype(np.uint8)
                        except Exception as exc:
                            raise RuntimeError(f"Could not safely load existing shadow mask: {path}: {exc}") from exc
            self.shadow_mask_buffer = initial

        if self.activity_roi_vertices_buffer is None and saved is not None:
            text = safe_str(saved.get("activity_roi_vertices_json")).strip()
            if text:
                try:
                    pts01 = np.asarray(json.loads(text), dtype=float)
                    pts = denormalized_roi_vertices(pts01, self.brain_signal_mask.shape)
                    if pts is not None:
                        self.activity_roi_vertices_buffer = pts
                except Exception:
                    pass

    def _add_shadow_edit_layer(self) -> None:
        """Create the editable exclusion/shadow mask layer."""
        initial = self.shadow_mask_buffer
        if initial is None or tuple(np.asarray(initial).shape) != tuple(self.brain_signal_mask.shape):
            initial = np.zeros(self.brain_signal_mask.shape, dtype=np.uint8)
        self.shadow_layer = self.viewer.add_labels(
            np.asarray(initial, dtype=np.uint8),
            name="shadow_exclude_rescue",
            opacity=0.50,
            visible=True,
        )
        try: self.shadow_layer.selected_label = 1
        except Exception: pass

    def _add_activity_roi_layer(self) -> None:
        """Create the editable activity-ROI polygon layer."""
        data: list[np.ndarray] = []
        if self.activity_roi_vertices_buffer is not None:
            pts = np.asarray(self.activity_roi_vertices_buffer, dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                data = [pts[:, -2:]]
        try:
            self.activity_roi_layer = self.viewer.add_shapes(
                data=data,
                shape_type="polygon",
                name="activity_roi",
                edge_width=2,
                opacity=0.35,
                visible=True,
            )
        except Exception:
            self.activity_roi_layer = self.viewer.add_shapes(data=data, shape_type="polygon", name="activity_roi")
        try:
            self._connect(self.activity_roi_layer.events.data, self._on_activity_roi_changed)
        except Exception:
            pass

    def _activate_activity_roi(self) -> None:
        """Activate the ROI editing tool."""
        if self.activity_roi_layer is None:
            return
        try: self.viewer.layers.selection.active = self.activity_roi_layer
        except Exception: pass
        try:
            has_roi = bool(len(self.activity_roi_layer.data))
        except Exception:
            has_roi = False
        try:
            self.activity_roi_layer.mode = "direct" if has_roi else "add_polygon"
        except Exception:
            try: self.activity_roi_layer.mode = "select" if has_roi else "add_polygon"
            except Exception: pass

    def _clear_activity_roi(self) -> None:
        """Clear the current activity ROI without changing behavior annotations."""
        if self.activity_roi_layer is None:
            return
        try: self.activity_roi_layer.data = []
        except Exception: pass
        self.activity_roi_vertices_buffer = None
        self.parent_roi_trace_values = np.asarray([], dtype=float)
        if self.current_var is not None and self.current_parent() is not None:
            self._build_parent_trace(self.current_var, self.current_parent())
        self.refresh_plot()
        self._activate_activity_roi()

    def _on_activity_roi_changed(self, event: Any | None = None) -> None:
        """Refresh activity ROI traces after a polygon edit."""
        if self._loading:
            return
        self._capture_edit_buffers()
        if self.current_var is not None and self.current_parent() is not None:
            try:
                self._build_parent_trace(self.current_var, self.current_parent())
                self.refresh_plot()
                self._update_info()
            except Exception as exc:
                try: self.viewer.status = f"ROI trace error: {exc}"
                except Exception: pass

    def _activate_shadow(self) -> None:
        """Activate manual shadow-mask editing."""
        if self.shadow_layer is None:
            return
        try: self.viewer.layers.selection.active = self.shadow_layer
        except Exception: pass
        try: self.shadow_layer.mode = "paint"
        except Exception: pass

    def _clear_shadow(self) -> None:
        """Clear the current editable shadow mask."""
        if self.shadow_layer is None:
            return
        self.shadow_layer.data = np.zeros(self.brain_signal_mask.shape, dtype=np.uint8)
        self.shadow_mask_buffer = np.zeros(self.brain_signal_mask.shape, dtype=np.uint8)
        self._activate_shadow()

    def _persist_edit_layers(self) -> tuple[str, int, str, str]:
        """Persist per-stem shadow and activity ROI edits to review output files.
        
        Returns
        -------
        shadow_path : str
            Path to the saved exclusion/shadow mask, if present.
        shadow_pixels : int
            Number of marked shadow pixels.
        roi_vertices_json : str
            Serialized ROI contour vertices.
        roi_mask_path : str
            Saved activity ROI mask location, if any."""
        self._capture_edit_buffers()

        shadow_path_text = ""
        shadow_n = 0
        shadow_path = self._mask_path_for_stem()
        if self.shadow_mask_buffer is not None and np.any(np.asarray(self.shadow_mask_buffer) > 0):
            arr = (np.asarray(self.shadow_mask_buffer) > 0).astype(np.uint8)
            np.save(shadow_path, arr)
            shadow_path_text = str(shadow_path)
            shadow_n = int(arr.sum())
        elif shadow_path.exists():
            try: shadow_path.unlink()
            except Exception: pass

        roi_vertices_json = ""
        roi_mask_path_text = ""
        roi_mask = self._activity_roi_mask()
        verts = self._activity_roi_vertices()
        roi_path = self._roi_mask_path_for_stem()
        if verts is not None and roi_mask is not None:
            pts01 = normalized_roi_vertices(verts, self.brain_signal_mask.shape)
            if pts01 is not None:
                roi_vertices_json = json.dumps(pts01.tolist(), ensure_ascii=False)
            np.save(roi_path, roi_mask.astype(np.uint8))
            roi_mask_path_text = str(roi_path)
        elif roi_path.exists():
            try: roi_path.unlink()
            except Exception: pass

        return shadow_path_text, shadow_n, roi_vertices_json, roi_mask_path_text

    def _on_trace_gaussian_toggled(self, enabled: bool) -> None:
        """Update only Matplotlib lines; never modify the original traces."""
        self.trace_gaussian_sigma_frames = self._requested_trace_sigma_frames if enabled else 0.0
        if not self._loading:
            self.refresh_plot()

    def _update_hbt_display_layer(self) -> None:
        """Replace only Napari HbT display pixels; preserve ROI/shadow and slot."""
        if self.segment_display_var is None or self.hbt_layer is None:
            return
        display_var = gaussian_spatial_display_stack(
            self.segment_display_var, self.brain_signal_mask, self.map_smoothing_sigma_px
        )
        vmin, vmax, _ = compute_segment_contrast_limits(
            display_var, self.brain_signal_mask,
            percentiles=self.map_percentiles,
            time_chunk=self.contrast_time_chunk,
            max_values=self.contrast_max_values,
        )
        # No viewer.layers.clear(): drawn masks, ROI, selected window stay intact.
        try:
            previous_frame = self._current_frame_index()
        except Exception:
            previous_frame = 0
        self.hbt_layer.data = display_var.data
        self.hbt_layer.contrast_limits = (vmin, vmax)
        try:
            self.viewer.dims.set_current_step(0, previous_frame)
        except Exception:
            pass

    def _on_map_smoothing_toggled(self, enabled: bool) -> None:
        """Toggle display-only Gaussian smoothing on the HbT map."""
        previous_sigma = self.map_smoothing_sigma_px
        self.map_smoothing_sigma_px = self._requested_map_sigma_px if enabled else 0.0
        if self._loading or self.hbt_layer is None:
            return
        old_data = self.hbt_layer.data
        old_contrast = self.hbt_layer.contrast_limits
        try:
            self._update_hbt_display_layer()
        except Exception as exc:
            # Display errors must not interrupt editing/rest decisions.
            self.map_smoothing_sigma_px = previous_sigma
            self.map_smoothing_checkbox.blockSignals(True)
            self.map_smoothing_checkbox.setChecked(previous_sigma > 0)
            self.map_smoothing_checkbox.blockSignals(False)
            try:
                self.hbt_layer.data = old_data
                self.hbt_layer.contrast_limits = old_contrast
            except Exception:
                pass
            try:
                self.viewer.status = f"Map Gaussian display unavailable: {exc}"
            except Exception:
                pass
            print(f"[REST review] Display-only map smoothing failed: {exc}")

    def _apply_raw_visibility(self) -> None:
        """Show or hide the matching raw-wavelength Napari layers."""
        visible = bool(self.raw_checkbox.isChecked()) if hasattr(self, "raw_checkbox") else False
        for layer in self.raw_layers.values():
            try: layer.visible = visible
            except Exception: pass

    # ---------------- plot ----------------

    def refresh_plot(self) -> None:
        """Update whole-brain and ROI traces plus the time cursor."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.parent_trace_times.size and self.parent_trace_values.size:
            brain_plot = self.parent_trace_values
            roi_plot = self.parent_roi_trace_values
            if self.trace_gaussian_sigma_frames > 0:
                brain_plot = gaussian_temporal_display_trace(brain_plot, self.trace_gaussian_sigma_frames)
                if roi_plot.size == self.parent_trace_times.size:
                    roi_plot = gaussian_temporal_display_trace(roi_plot, self.trace_gaussian_sigma_frames)
            ax.plot(self.parent_trace_times, brain_plot, linewidth=1.0, label="whole-brain HbT")
            if roi_plot.size == self.parent_trace_times.size:
                ax.plot(self.parent_trace_times, roi_plot, linewidth=1.0, label="activity ROI HbT")
            t0, t1 = self._current_window()
            ax.axvspan(t0, t1, alpha=0.20, label=f"selected {self.segment_sec:g} s")
            current_abs = self._current_recording_time()
            self._time_cursor_line = ax.axvline(current_abs, linestyle=":", linewidth=1.0)
            parent = self.current_parent()
            if parent is not None:
                ax.set_xlim(float(parent["t0_sec"]), float(parent["t1_sec"]))
            ax.set_xlabel("Recording time (s)")
            ax.set_ylabel(safe_str(self.signal_combo.currentData()))
            if self.trace_gaussian_sigma_frames > 0:
                ax.set_title(f"Display-only temporal Gaussian σ={self.trace_gaussian_sigma_frames:g} frame(s)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.25)
        else:
            ax.text(0.5, 0.5, "No parent REST trace", transform=ax.transAxes, ha="center", va="center")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _current_frame_index(self) -> int:
        """Read the currently displayed Napari temporal sample index."""
        if self.segment_times.size == 0:
            return 0
        try:
            steps = tuple(self.viewer.dims.current_step)
            frame = int(round(float(steps[0]))) if steps else 0
        except Exception:
            frame = 0
        return int(np.clip(frame, 0, len(self.segment_times) - 1))

    def _current_recording_time(self) -> float:
        """Map the displayed temporal sample to recording seconds."""
        if not self.segment_times.size:
            return float(self.start_spin.value())
        return float(self.segment_times[self._current_frame_index()])

    def _on_napari_time_changed(self, event: Any | None = None) -> None:
        """Move the trace cursor to the current Napari frame time."""
        if self._loading:
            return
        if self._time_cursor_line is not None:
            x = self._current_recording_time()
            try: self._time_cursor_line.set_xdata([x, x])
            except Exception: pass
            self.canvas.draw_idle()

    # ---------------- persistence ----------------

    def _replace_slot_decision(self, records: list[dict[str, Any]], record: dict[str, Any]) -> list[dict[str, Any]]:
        """Replace the saved decision for one canonical candidate slot.
        
        Notes
        -----
        The decision history is reduced to one current row per ``slot_id``;
        slot ordering follows the parent REST interval and chunk index."""
        sid = safe_str(record.get("slot_id")).strip()
        out = [dict(x) for x in records if safe_str(x.get("slot_id")).strip() != sid]
        out.append(dict(record))
        slot_order = {s["slot_id"]: i for i, s in enumerate(self._all_slots())}
        out.sort(key=lambda x: slot_order.get(safe_str(x.get("slot_id")), 10**9))
        return out

    @staticmethod
    def _intervals_from_decisions(records: list[dict[str, Any]], decision: str) -> list[dict[str, Any]]:
        """Build sorted REST intervals from decisions with the requested status."""
        out: list[dict[str, Any]] = []
        for item in records:
            if safe_str(item.get("decision")).strip().lower() != decision:
                continue
            try:
                out.append({
                    "label": "rest",
                    "t0_sec": float(item["t0_sec"]),
                    "t1_sec": float(item["t1_sec"]),
                })
            except Exception:
                continue
        out.sort(key=lambda x: (float(x["t0_sec"]), float(x["t1_sec"])))
        return out

    def _advance_after_chunk_decision(self) -> None:
        """Advance to the next candidate or selected recording."""
        parent = self.current_parent()
        if parent is None:
            return
        n = self._n_chunks_for_parent(parent)
        if self.current_chunk_index + 1 < n:
            self._set_candidate(self.parent_combo.currentIndex(), self.current_chunk_index + 1, load=True)
            return
        pi = self.parent_combo.currentIndex()
        if pi + 1 < self.parent_combo.count():
            self._set_candidate(pi + 1, 0, load=True)
            return
        if self.stem_index < len(self.df) - 1:
            self.change_stem(1)

    def save_decision(self, status: str) -> None:
        """Save a USE, REJECT, or UNCLEAR decision for the selected REST slot.
        
        Parameters
        ----------
        status : str
            One of ``use``, ``reject``, or ``unclear``. ``selected`` is accepted
            as a legacy synonym for ``use``.
        
        Notes
        -----
        Stores all slot decisions in ``segment_decisions_json`` and accepted
        intervals in ``label_intervals_json``. A stem is ``complete`` only when
        every canonical slot has a decision. Saves one CSV row per stem and
        advances the GUI to the next candidate."""
        status = safe_str(status).strip().lower()
        if status == "selected":
            status = "use"
        if status not in {"use", "reject", "unclear"}:
            raise ValueError(status)
        from datetime import datetime

        t0, t1 = self._current_window()
        parent = self.current_parent()
        slot = self._current_slot()
        if parent is None or slot is None:
            return

        records = self._decision_records()
        record = {
            "slot_id": slot["slot_id"],
            "decision": status,
            "parent_rest_index": int(slot["parent_rest_index"]),
            "chunk_index_in_parent": int(slot["chunk_index_in_parent"]),
            "parent_rest_t0_sec": float(parent["t0_sec"]),
            "parent_rest_t1_sec": float(parent["t1_sec"]),
            "nominal_t0_sec": float(slot["nominal_t0_sec"]),
            "nominal_t1_sec": float(slot["nominal_t1_sec"]),
            "t0_sec": float(t0),
            "t1_sec": float(t1),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        records = self._replace_slot_decision(records, record)

        selected_intervals = self._intervals_from_decisions(records, "use")
        rejected = self._intervals_from_decisions(records, "reject")
        unclear = self._intervals_from_decisions(records, "unclear")
        all_slots = self._all_slots()
        reviewed_ids = {safe_str(x.get("slot_id")) for x in records if safe_str(x.get("decision")) in {"use", "reject", "unclear"}}
        complete = bool(all_slots) and all(s["slot_id"] in reviewed_ids for s in all_slots)
        review_status = "complete" if complete else "in_progress"

        shadow_path_text, shadow_n, roi_vertices_json, roi_mask_path_text = self._persist_edit_layers()

        payload = {
            "stem": self.stem,
            "mouse_id": safe_str(self.row.get("mouse_id")),
            "experimental_group": safe_str(self.row.get("experimental_group")),
            "exp_day": safe_str(self.row.get("exp_day")),
            "state": safe_str(self.row.get("state")),
            "protocol": safe_str(self.row.get("protocol")),
            "rescue_reason": safe_str(self.row.get("rescue_reason")),
            "n_good_primary_qc": safe_str(self.row.get("n_good_primary_qc")),
            # Unified accepted REST format: ALL accepted 60-s pieces for this stem.
            "label_intervals_json": json.dumps(selected_intervals, ensure_ascii=False),
            "rejected_60s_intervals_json": json.dumps(rejected, ensure_ascii=False),
            "unclear_60s_intervals_json": json.dumps(unclear, ensure_ascii=False),
            "segment_decisions_json": json.dumps(records, ensure_ascii=False),
            "n_total_60s_candidates": len(all_slots),
            "n_reviewed_60s": len(reviewed_ids),
            "n_selected_60s": len(selected_intervals),
            "n_rejected_60s": len(rejected),
            "n_unclear_60s": len(unclear),
            "segment_duration_sec": self.segment_sec,
            "target_band_low_hz": 0.08,
            "target_band_high_hz": 0.4,
            "signal_name": safe_str(self.signal_combo.currentData()),
            "display_mode": "segment_centered",
            "shadow_mask_path": shadow_path_text,
            "shadow_mask_n_px": shadow_n,
            "activity_roi_vertices_json": roi_vertices_json,
            "activity_roi_mask_path": roi_mask_path_text,
            "last_slot_id": slot["slot_id"],
            "last_segment_t0_sec": float(t0),
            "last_segment_t1_sec": float(t1),
            "last_segment_decision": status,
            "review_status": review_status,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        upsert_csv_row(self.rescue_csv, payload, key="stem")
        self._saved_rows = read_latest_by_key(self.rescue_csv, "stem")
        self._update_chunk_label()
        self._update_info()
        try:
            self.viewer.status = f"Saved {status} for current 60 s: {self.stem}"
        except Exception:
            pass
        self._advance_after_chunk_decision()

    def _update_info(self) -> None:
        """Refresh window times, pixel counts, and review decision totals."""
        t0, t1 = self._current_window()
        records = self._decision_records()
        counts = {k: 0 for k in ("use", "reject", "unclear")}
        for item in records:
            d = safe_str(item.get("decision")).strip().lower()
            if d in counts:
                counts[d] += 1
        mask_n = int(np.sum(np.asarray(self.shadow_layer.data) > 0)) if self.shadow_layer is not None else 0
        roi_mask = self._activity_roi_mask()
        roi_n = int(np.sum(roi_mask)) if roi_mask is not None else 0
        self.window_label.setText(f"→ {t1:.3f} s")
        self._update_chunk_label()
        self.title_label.setText(
            f"{self.stem} | {safe_str(self.row.get('experimental_group'))} | {safe_str(self.row.get('mouse_id'))} | "
            f"{safe_str(self.row.get('exp_day'))}"
        )
        self.info_label.setText(
            f"Window {t0:.3f}–{t1:.3f} s | ROI={roi_n} px | shadow={mask_n} px | "
            f"USE={counts['use']} REJECT={counts['reject']} UNCLEAR={counts['unclear']}"
        )


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the combined REST QC/rescue viewer."""
    parser = argparse.ArgumentParser(description="Unified REST QC/Rescue GUI: rate every eligible REST 60-s window and optionally shift it or draw shadows.")
    parser.add_argument("--config", required=True, type=Path, help="Project YAML config")
    parser.add_argument("--filter", action="append", default=None, help="Additional Step4 filter KEY=VALUE or KEY=a,b")
    parser.add_argument("--stems", nargs="+", default=None, help="Manual stems to add to rescue list")
    parser.add_argument("--stem", action="append", default=None, help="Manual stem; repeatable (alias for --stems)")
    parser.add_argument("--start-stem", default=None)
    parser.add_argument("--include-bad", action=argparse.BooleanOptionalAction, default=True,
                        help="Include Step2 is_bad rows for manual review (default: yes); use --no-include-bad to exclude")
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--primary-qc-csv", type=Path, default=None, help="Primary rest_segment_qc.csv; default inferred from out")
    parser.add_argument("--rescue-csv", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--no-auto-low-good", action="store_true", help="Use only manually supplied stems; do not auto-add stems with 0/1 GOOD chunks")
    parser.add_argument("--max-good-for-auto", type=int, default=1, help="For --legacy-rescue-only: select stems with <=N primary GOOD chunks")
    parser.add_argument("--legacy-rescue-only", action="store_true", help="Use old rescue selection; default is combined QC for ALL annotated REST stems")
    parser.add_argument("--use-analysis-selection", action="store_true",
                        help="Opt into legacy analysis_selection filtering; default is NOT to apply it")
    parser.add_argument("--ignore-analysis-selection", action="store_true",
                        help="Deprecated no-op: analysis_selection is already ignored by default")
    parser.add_argument("--only-unreviewed", action="store_true",
                        help="Open only stems without a saved review_status=complete; in_progress stems remain")
    parser.add_argument("--list", action="store_true",
                        help="Print selected stems/review statuses without opening Napari")
    parser.add_argument("--signal", default="dcHbT")
    parser.add_argument("--no-raw", action="store_true")
    parser.add_argument("--segment-sec", type=float, default=60.0)
    parser.add_argument("--map-percentiles", nargs=2, type=float, default=(2.0, 98.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--contrast-time-chunk", type=int, default=50)
    parser.add_argument("--contrast-max-values", type=int, default=2_000_000)
    parser.add_argument("--trace-gaussian-sigma", type=float, default=0.0, metavar="FRAMES",
                        help="Temporal Gaussian sigma in optical frames for both plotted traces; 0 disables (default). Display only.")
    parser.add_argument("--map-sigma", type=float, default=0.0, metavar="PX",
                        help="Spatial Gaussian sigma in pixels for Napari HbT maps; 0 disables (default). Display only.")
    parser.add_argument("--annotation-time-unit", choices=["seconds", "auto", "frames"], default="seconds")
    parser.add_argument("--contour-width", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Parse REST QC CLI arguments and open the interactive Napari viewer.
    
    Notes
    -----
    ``--config`` supplies the project YAML. ``--segment-sec`` is a CLI
    parameter with default 60 seconds; the format of the saved field names
    uses the historical ``60s`` naming even when configured otherwise."""
    configure_console_encoding_fallback()
    # If Qt/VisPy/Dask terminates the Python process without a normal exception,
    # -X faulthandler or this handler may still print the active Python stacks.
    try:
        faulthandler.enable(all_threads=True)
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    if not np.isfinite(args.segment_sec) or float(args.segment_sec) <= 0:
        raise ValueError("--segment-sec must be positive")
    p_low, p_high = map(float, args.map_percentiles)
    if not (0 <= p_low < p_high <= 100):
        raise ValueError("--map-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    for name in ("trace_gaussian_sigma", "map_sigma"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    manual = list(args.stems or []) + list(args.stem or [])
    dataframe, config, paths, annotations_path, primary_qc_path, default_rescue_csv = build_rescue_index(
        args.config,
        filters=parse_cli_filters(args.filter),
        manual_stems=manual,
        include_bad=bool(args.include_bad),
        annotations_path=args.annotations,
        primary_qc_path=args.primary_qc_csv,
        segment_sec=float(args.segment_sec),
        auto_low_good=not bool(args.no_auto_low_good),
        max_good_for_auto=int(args.max_good_for_auto),
        use_analysis_selection=bool(args.use_analysis_selection) and not bool(args.ignore_analysis_selection),
        review_all=not bool(args.legacy_rescue_only),
    )
    rescue_csv = Path(args.rescue_csv) if args.rescue_csv is not None else default_rescue_csv
    saved = read_latest_by_key(rescue_csv, "stem")
    completed = set()
    saved_statuses: dict[str, str] = {}
    if not saved.empty and "stem" in saved.columns:
        for _, saved_row in saved.iterrows():
            key = safe_str(saved_row.get("stem")).strip()
            status = safe_str(saved_row.get("review_status")).strip().lower()
            if key:
                saved_statuses[key] = status
                if status == "complete":
                    completed.add(key)
    dataframe = dataframe.copy()
    dataframe["saved_review_status"] = dataframe["stem"].map(
        lambda s: saved_statuses.get(safe_str(s).strip(), "unreviewed")
    )
    if args.only_unreviewed:
        dataframe = dataframe[~dataframe["stem"].map(safe_str).isin(completed)].copy()
    dataframe = dataframe.reset_index(drop=True)
    print(f"REST review recordings: {len(dataframe)} (already complete in saved CSV: {len(completed)})")
    print(f"Behavior annotations (read-only): {annotations_path}")
    print(f"Legacy primary QC path (not used in default combined review): {primary_qc_path}")
    print(f"Rescue output: {rescue_csv}")
    print(f"Segment length: {float(args.segment_sec):g} s | display: segment-centered HbT")
    print(f"Display-only Gaussian: trace sigma={args.trace_gaussian_sigma:g} frames, map sigma={args.map_sigma:g} px")
    if manual:
        print("Manual stems requested:", manual)
    if args.list:
        columns = [k for k in ("stem", "mouse_id", "exp_day", "state", "experimental_group", "saved_review_status") if k in dataframe]
        print(dataframe[columns].to_string(index=False) if not dataframe.empty else "No remaining eligible stems")
        return
    if dataframe.empty:
        print("No remaining eligible REST QC stems. Existing CSV was not modified.")
        return
    # Backup the authoritative manual-review CSV before the first interactive edit.
    # Saving new stems uses the existing stem-keyed upsert; reviewed stems are untouched.
    if rescue_csv.is_file():
        from datetime import datetime
        import shutil
        backup_dir = rescue_csv.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = backup_dir / f"{rescue_csv.stem}_before_gui_{stamp}.csv"
        shutil.copy2(rescue_csv, backup)
        print(f"Backup of existing manual QC: {backup}")

    viewer = napari.Viewer(title="REST segment review: QC + optional rescue")
    controller = RestSegmentReviewController(
        viewer=viewer,
        dataframe=dataframe,
        config=config,
        paths=paths,
        annotations_path=annotations_path,
        rescue_csv=rescue_csv,
        signal_name=args.signal,
        load_raw=not bool(args.no_raw),
        segment_sec=float(args.segment_sec),
        map_percentiles=(p_low, p_high),
        contrast_time_chunk=int(args.contrast_time_chunk),
        contrast_max_values=int(args.contrast_max_values),
        trace_gaussian_sigma_frames=float(args.trace_gaussian_sigma),
        map_smoothing_sigma_px=float(args.map_sigma),
        annotation_time_unit=args.annotation_time_unit,
        contour_width=int(args.contour_width),
        start_stem=args.start_stem,
    )
    viewer.window.add_dock_widget(controller, name="REST review", area="right")
    napari.run()


if __name__ == "__main__":
    main()
