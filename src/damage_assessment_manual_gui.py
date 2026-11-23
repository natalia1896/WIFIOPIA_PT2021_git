"""Manual lesion review GUI for WIFIOPIA/OISI damage assessment.

Purpose
-------
For each selected post-PT stem, prepare a fixed set of display-only helper maps and
let an annotator manually outline the lesion in Napari. This module deliberately
DOES NOT perform automatic lesion segmentation and DOES NOT feed lesion or
hemorrhage masks back into Step4 processing.

GUI image layers (current-stem rotated_fullres display coordinates):
The wavelength labels below describe the default 505/656-nm configuration;
other available layers depend on the project YAML and input data.
- raw_505 / baseline_505_registered / diff_signed_505;
- raw_505_mean / baseline_505_registered_mean / diff_signed_505_mean;
- raw_656 / baseline_656_registered / diff_signed_656;
- raw_656_mean / baseline_656_registered_mean / diff_signed_656_mean;
- run_sd_dcHbT;
- homotopic_step7_authoritative;
- dcHbT_movie: lazy read-only processed Zarr movie (any configured day);
- raw_505_movie / raw_656_movie: on-demand existing raw Zarr movies (Napari dock).

Overlays:
- brain_mask_display_current: read-only, hidden by default;
- lesion_polygon_EDIT: the only editable layer.

Baseline fallback
-----------------
If Day-PT metadata/BMP/Raw Zarr is missing, retain the current reference BMP,
any available current raw mean, on-demand raw movies and processed HbT movie.
Suppress all baseline and signed-difference layers, with an explicit warning.
Missing baseline alone never prevents manual lesion annotation.

Mean-image contract
-------------------
raw_<wl>_mean is computed ONLY from an already existing Step4 raw_zarr store.
The zarr mean is calculated on the native rotated_scaled grid and then resized once
to the current Step2/Step3 rotated_fullres display shape. The module never rebuilds
raw_zarr and never rereads the source BMP sequence for mean images. Baseline mean
images are resized to baseline fullres and then warped baseline -> reference ->
current stem exactly like the single-frame baseline display. No CLAHE layers are produced.

Geometry contract
-----------------
Step1/2/3 images and final lesion mask are in rotated_fullres. Step4 processed
signals and Step7 FC maps are lower-resolution. Those maps are resized only for
GUI display. Quantitative source arrays remain in their native grids.

Typical commands
----------------
One stem test:
  python damage_assessment_manual_gui.py --config config.yaml --stem STEM

One mouse/group:
  python damage_assessment_manual_gui.py --config config.yaml --mouse m12

Batch helper preparation only:
  python damage_assessment_manual_gui.py --config config.yaml --batch --prepare-only

Batch sequential review:
  python damage_assessment_manual_gui.py --config config.yaml --batch --only-unreviewed
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import traceback
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, detrend as scipy_detrend, sosfiltfilt
from skimage.draw import polygon as sk_polygon
from skimage.measure import find_contours
from skimage.transform import resize, warp

try:
    import napari
except Exception:  # pragma: no cover
    napari = None

from common import (
    apply_raw_rotation,
    canonical_mask_paths,
    get_path_resolving_config,
    list_raw_images,
    resolve_flexible_path,
)

try:
    import damage_assessment as damage
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "damage_assessment_manual_gui.py requires damage_assessment.py "
        "in the same src folder."
    ) from exc

# Step4 is imported lazily. This keeps the GUI module importable in lightweight
# environments while still using the current Step4 index in the real project.
def _get_build_step4_index():
    """Lazily import the Step4 index builder when processed data are needed."""
    from step4_signal_processing import build_step4_index
    return build_step4_index


# =============================================================================
# Config
# =============================================================================


def _deep_update_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Fill absent nested GUI settings without overwriting configured values."""
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_update_missing(dst[key], value)
    return dst


def default_manual_gui_config() -> dict[str, Any]:
    """Return fallback settings for the manual lesion-review interface."""
    return {
        "enabled": True,
        "helpers_folder": "manual_gui_helpers",
        "tables_folder": "tables",
        "gui_index_name": "damage_gui_index.csv",
        "review_log_name": "damage_manual_review.csv",
        "error_log_name": "damage_manual_gui_errors.csv",
        "raw_layers": {
            "wavelengths": ["505", "656"],
            "normalization": "mean_in_brain",  # none / mean_in_brain, used for difference only
            "mean_images": {
                "enabled": True,
                # current uses ONLY an existing Step4 raw_zarr. It never rebuilds zarr
                # and never rereads raw BMP frames for the temporal mean.
                "source": "existing_raw_zarr",
                "raw_zarr_variant": "selected_for_processing",
                "resize_interpolation": "linear",
            },
        },
        "run_sd": {
            "enabled": True,
            "annotations_path": None,  # default: out/behavior_review/behavior_review_annotations.csv
            "signal": "dcHbT",
            "allow_signal_fallback": True,
            "fallback_signals": ["dcHbT_gsub"],
            "min_run_duration_sec": 2.0,
            "interval_source": "complement_of_rest",  # non-REST is RUN by project convention

            "save_segment_maps": True,
            "aggregation": "sample_weighted_segment_sd",
        },
        "hbt_movie": {
            "enabled": True,
            "exp_days": [],  # empty = any selected exp_day with a processed Zarr
            "signal": "dcHbT",
            "colormap": "turbo",  # keep existing Napari automatic contrast limits
        },
        "raw_movie": {
            "enabled": True,
            "wavelengths": ["505", "656"],
            "colormap": "gray",
            "load_mode": "on_demand",  # Napari dock: 505, 656, or both
        },
        "homotopic": {
            # Existing frozen Step7 output, kept as a separate comparison layer.
            "authoritative": {
                "enabled": True,
                "folder_name": "step7_functional_connectivity",
                "rest_variant": "with_rest",
                "variant_id": "hbt_bandpass",
                "subset_output_key": None,  # optional known _subset_runs/<key>
                "mask_set": "stroke_longitudinal",
                "display_units": "pearson_r",  # pearson_r / fisher_z
                "save_resized_z": True,
                "show_gbc": False,  # optionally load existing gbc_z_mean.npy, never compute FC
            },
            # New current DISPLAY-ONLY continuous map. It deliberately ignores
            # analytical-ROI labels/buffers but preserves Step7-like REST temporal processing.
            "full_preview": {
                "enabled": False,
                "folder_name": "step7_functional_connectivity_v4",
                "mask_set": "stroke_longitudinal",
                "signal": "dcHbT",
                "allow_signal_fallback": True,
                "fallback_signals": ["dcHbT_gsub"],
                "rest_segments_path": None,  # auto: Step7 rest_segments_used.csv, then global final REST
                "display_units": "pearson_r",  # pearson_r / fisher_z
                "bilateral_display": True,
                "midline_exclusion_px_fc": 1.0,
                "mutual_roundtrip": True,
                "fisher_clip": 0.999999,
                "save_segment_maps": False,
                "temporal": {
                    "inherit_functional_connectivity": True,
                    "detrend_enabled": True,
                    "detrend_type": "linear",
                    "filter_enabled": True,
                    "low_hz": 0.08,
                    "high_hz": 0.40,
                    "order": 4,
                    "synthetic_odd_pad_sec": 20.0,
                    "variance_epsilon": 1.0e-12,
                },
                "spatial": {
                    "inherit_functional_connectivity": True,
                    "scale": 0.50,
                    "min_valid_fraction": 0.50,
                },
            },
        },
        "hemorrhage": {
            "enabled": False,
            # Legacy display-only overlay (not loaded).
            "step4_manual_roi_folder": None,
            "filename": "hemorrhage_manual.npy",
            "opacity": 0.35,
            "contour_width": 2,
        },
        "display": {
            "percentiles": [2.0, 98.0],
            "percentiles_use_brain_mask": True,
            "default_visible_layer": "raw_505",
            "layer_opacity": 1.0,
            "colormaps": {
                "raw": "gray",
                "diff_signed": "coolwarm",
                "run_sd": "turbo",
                "homotopic": "turbo",
            },
        },
        "review": {
            "preload_existing_lesion": True,
            "save_empty_mask": False,
            "lesion_edge_color": "yellow",
            "lesion_face_color": [1.0, 1.0, 0.0, 0.0],
            "lesion_edge_width": 2.0,
            "skip_reviewed": False,
        },
        "batch": {
            "continue_on_error": True,
        },
    }


def ensure_manual_gui_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Merge GUI defaults and disable obsolete preview/hemorrhage overlays."""
    damage.ensure_damage_defaults(config)
    config.setdefault("damage_assessment", {})
    config["damage_assessment"].setdefault("manual_gui", {})
    _deep_update_missing(config["damage_assessment"]["manual_gui"], default_manual_gui_config())
    # The GUI intentionally never rebuilds FC previews or loads hemorrhage,
    # including when an older YAML still has these options enabled.
    manual = config["damage_assessment"]["manual_gui"]
    manual.setdefault("homotopic", {}).setdefault("full_preview", {})["enabled"] = False
    manual.setdefault("hemorrhage", {})["enabled"] = False
    return config


def gui_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return the effective damage_assessment.manual_gui configuration."""
    return ensure_manual_gui_defaults(config)["damage_assessment"]["manual_gui"]


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get a nested configuration value or the specified fallback."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# =============================================================================
# Paths / logging
# =============================================================================


def build_gui_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create and return helper, index, review and error-output paths."""
    cfg = gui_cfg(config)
    damage_root = Path(paths.get("damage_root") or (Path(paths["out_root"]) / config["damage_assessment"].get("folder_name", "damage_assessment")))
    helpers_root = damage_root / str(cfg.get("helpers_folder", "manual_gui_helpers"))
    tables_root = damage_root / str(cfg.get("tables_folder", "tables"))
    helpers_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    return {
        "damage_root": damage_root,
        "helpers_root": helpers_root,
        "tables_root": tables_root,
        "gui_index_csv": tables_root / str(cfg.get("gui_index_name", "damage_gui_index.csv")),
        "review_log_csv": tables_root / str(cfg.get("review_log_name", "damage_manual_review.csv")),
        "error_log_csv": tables_root / str(cfg.get("error_log_name", "damage_manual_gui_errors.csv")),
    }


def helper_dir(paths: dict[str, Path], stem: str) -> Path:
    """Return the directory for cached display helpers of one stem."""
    p = Path(paths["helpers_root"]) / str(stem)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _now_iso() -> str:
    """Return a seconds-resolution local ISO timestamp."""
    return _dt.datetime.now().isoformat(timespec="seconds")


def _upsert_csv(path: Path, row: dict[str, Any], keys: Sequence[str]) -> None:
    """Update or append one uniquely keyed CSV record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        if not old.empty and all(k in old.columns for k in keys):
            same_record = pd.Series(True, index=old.index)
            for k in keys:
                same_record &= old[k].astype(str).eq(str(row.get(k, "")))
            old = old.loc[~same_record].copy()
        new = pd.concat([old, new], ignore_index=True, sort=False)
    new.to_csv(path, index=False)


# =============================================================================
# Small image helpers
# =============================================================================


def _read_gray(path: Path) -> np.ndarray:
    """Read a full-resolution reference BMP as a two-dimensional float32 image."""
    arr = np.asarray(iio.imread(path))
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr[..., :3].mean(axis=-1).astype(np.float32)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Unsupported image shape {arr.shape}: {path}")


def _resolve_row_path(row: pd.Series, columns: Sequence[str], path_cfg: dict[str, Any]) -> Path:
    """Find and resolve the first existing path in a metadata row."""
    tried: list[str] = []
    for col in columns:
        if col not in row.index:
            continue
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)) or not str(val).strip():
            continue
        tried.append(f"{col}={val}")
        p = resolve_flexible_path(
            val,
            from_drive=path_cfg.get("from_drive"),
            to_drive=path_cfg.get("to_drive"),
            from_root=path_cfg.get("from_root"),
            to_root=path_cfg.get("to_root"),
            must_exist=False,
        )
        if p is not None and p.exists():
            return Path(p)
    raise FileNotFoundError("No existing image path. Tried: " + "; ".join(tried or columns))


def load_wave_image(row: pd.Series, wavelength: str, path_cfg: dict[str, Any]) -> tuple[np.ndarray, Path]:
    """Read the requested wavelength reference image for a stem."""
    wl = str(wavelength)
    p = _resolve_row_path(
        row,
        [f"ref_{wl}_path_resolved", f"ref_{wl}_path", f"bmp_{wl}"],
        path_cfg,
    )
    return _read_gray(p), p


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask with nearest-neighbour interpolation."""
    arr = np.asarray(mask, dtype=bool)
    if tuple(arr.shape) == tuple(shape):
        return arr
    return cv2.resize(
        arr.astype(np.uint8),
        (int(shape[1]), int(shape[0])),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def resize_float_nan(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a continuous 2-D map while preserving NaN support.

    Values and finite-pixel weights are interpolated separately, then divided.
    A nearest-neighbour validity map is used as the final support so values do
    not bleed into the missing homotopic hemisphere/background.
    """
    a = np.asarray(arr, dtype=np.float32)
    if tuple(a.shape) == tuple(shape):
        return a.copy()
    valid = np.isfinite(a)
    values = np.where(valid, a, 0.0).astype(np.float32)
    size = (int(shape[1]), int(shape[0]))
    weighted = cv2.resize(values, size, interpolation=cv2.INTER_LINEAR)
    weights = cv2.resize(valid.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    valid_nn = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    out = np.full(shape, np.nan, dtype=np.float32)
    good = valid_nn & (weights > 1e-6)
    out[good] = (weighted[good] / weights[good]).astype(np.float32)
    return out


def contrast_limits(arr: np.ndarray, brain: np.ndarray, p_low: float, p_high: float) -> tuple[float, float]:
    """Determine finite display limits from percentile values, favoring brain pixels."""
    a = np.asarray(arr, dtype=np.float32)
    m = np.asarray(brain, dtype=bool)
    if a.shape != m.shape:
        m = resize_bool(m, a.shape)
    vals = a[m & np.isfinite(a)]
    if vals.size == 0:
        vals = a[np.isfinite(a)]
    if vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(vals, [float(p_low), float(p_high)])
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.nanmean(vals)) if vals.size else 0.0
        spread = float(np.nanstd(vals)) if vals.size else 1.0
        spread = spread if np.isfinite(spread) and spread > 0 else 1.0
        return center - spread, center + spread
    return lo, hi


def normalize_mean_in_mask(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Divide an image by its mean over finite pixels of the selected mask."""
    a = np.asarray(arr, dtype=np.float32).copy()
    vals = a[np.asarray(mask, bool) & np.isfinite(a)]
    if vals.size == 0:
        return a
    denom = float(np.mean(vals))
    if abs(denom) < 1e-12:
        return a
    return a / denom


def warp_native_to_reference(
    image_native: np.ndarray,
    tform_native_to_ref: Any,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a native image and validity support into reference coordinates."""
    if tform_native_to_ref is None:
        img = image_native.astype(np.float32)
        if tuple(img.shape) != tuple(output_shape):
            img = resize(img, output_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        return img, np.ones(output_shape, dtype=bool)
    out = warp(
        image_native.astype(np.float32),
        inverse_map=tform_native_to_ref.inverse,
        output_shape=output_shape,
        order=1,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    support = warp(
        np.ones_like(image_native, dtype=np.float32),
        inverse_map=tform_native_to_ref.inverse,
        output_shape=output_shape,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    ) > 0.5
    return out, support


def warp_baseline_to_current(
    baseline_native: np.ndarray,
    baseline_tform: Any,
    current_tform: Any,
    current_shape: tuple[int, int],
    ref_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Warp a baseline image through reference space to the current image grid.

    Parameters
    ----------
    baseline_native : numpy.ndarray
        Two-dimensional baseline image in rotated full-resolution coordinates.
    baseline_tform, current_tform : transform or None
        Native-baseline-to-reference and current-to-reference transforms.
    current_shape, ref_shape : tuple of int
        ``(y, x)`` shapes in pixels of current and reference images.

    Returns
    -------
    warped : numpy.ndarray
        Baseline image on the current-stem grid, with NaN outside support.
    support : numpy.ndarray
        Boolean footprint of pixels covered by the warped baseline.

    Notes
    -----
    This is a display-only helper; source images and quantitative signals are not
    modified.
    """
    baseline_ref, support_ref = warp_native_to_reference(baseline_native, baseline_tform, ref_shape)
    baseline_cur = damage.warp_reference_to_current(baseline_ref, current_tform, current_shape, order=1)
    support_cur = damage.warp_reference_to_current(support_ref.astype(np.float32), current_tform, current_shape, order=0) > 0.5
    baseline_cur = baseline_cur.astype(np.float32)
    baseline_cur[~support_cur] = np.nan
    return baseline_cur, support_cur


# =============================================================================
# Brain / raw / difference helpers
# =============================================================================


def load_fullres_brain(row: pd.Series, paths: dict[str, Path], shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Load a stem brain mask, preferring the manually corrected version."""
    c = canonical_mask_paths(paths, str(row["stem"]))
    if c["brain_manual"].exists():
        p = c["brain_manual"]
    elif c["brain_auto"].exists():
        p = c["brain_auto"]
    else:
        raise FileNotFoundError(f"No brain_manual.npy or brain_auto.npy for stem={row['stem']}")
    mask = np.load(p).astype(bool)
    if tuple(mask.shape) != tuple(shape):
        mask = resize_bool(mask, shape)
    return mask, str(p)


_MANIFEST_CACHE: dict[str, pd.DataFrame] = {}


def _manifest_row_for_stem(stem: str, paths: dict[str, Path]) -> pd.Series:
    """Locate a stem record in the Step1 manifest."""
    manifest_path = Path(paths["manifest_path"])
    key = str(manifest_path.resolve()) if manifest_path.exists() else str(manifest_path)
    if key not in _MANIFEST_CACHE:
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest_recordings.csv not found: {manifest_path}")
        _MANIFEST_CACHE[key] = pd.read_csv(manifest_path)
    manifest = _MANIFEST_CACHE[key]
    if "stem" not in manifest.columns:
        raise ValueError(f"Manifest has no stem column: {manifest_path}")
    rows = manifest[manifest["stem"].astype(str).eq(str(stem))]
    if rows.empty:
        raise KeyError(f"Manifest row not found for stem={stem}")
    return rows.iloc[-1]


def _existing_step4_path(value: Any, config: dict[str, Any] | None = None) -> Path | None:
    """Resolve an existing Step4 output with the project's F: -> H: remap.

    Paths in an older Step4 index may still use the original drive. Read only:
    this function never creates, moves, or recalculates a Zarr store.
    """
    text = _safe_str(value).strip()
    if not text:
        return None
    p = Path(text)
    if p.exists():
        return p
    mapping = get_path_resolving_config(config or {}, "step4")
    try:
        resolved = resolve_flexible_path(
            text,
            from_drive=mapping.get("from_drive"),
            to_drive=mapping.get("to_drive"),
            from_root=mapping.get("from_root"),
            to_root=mapping.get("to_root"),
            must_exist=True,
        )
        return Path(resolved) if resolved is not None else None
    except FileNotFoundError:
        return None


def _raw_zarr_path_from_step4_row(
    row: pd.Series | None, config: dict[str, Any] | None = None,
) -> Path:
    """Find the existing Step4 raw Zarr, respecting global path remapping."""
    if row is None:
        raise FileNotFoundError("Step4 index row is missing; existing raw_zarr cannot be resolved")
    candidates = [
        "raw_zarr_selected_for_processing_path_resolved",
        "raw_zarr_motion_corrected_path_resolved",
        "raw_zarr_unaligned_path_resolved",
        "raw_zarr_path_resolved",
        "raw_zarr_path",
    ]
    tried: list[str] = []
    for col in candidates:
        if col not in row.index:
            continue
        value = _safe_str(row.get(col)).strip()
        if not value:
            continue
        tried.append(f"{col}={value}")
        resolved = _existing_step4_path(value, config)
        if resolved is not None:
            return resolved
    raise FileNotFoundError("No existing raw_zarr found. Tried: " + "; ".join(tried or candidates))


def _raw_zarr_key(config: dict[str, Any]) -> str:
    """Return the configured Zarr data-array key for raw image data."""
    return str(
        cfg_get(
            config,
            "signal_processing", "input", "raw_zarr_key",
            default=cfg_get(config, "raw", "zarr_key", default="raw_record"),
        )
    )


def _normalise_wave_label(value: Any) -> str:
    """Normalize a wavelength label used for raw-Zarr channel selection."""
    try:
        x = float(value)
        if np.isfinite(x) and abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return str(x)
    except Exception:
        return str(value).strip()


def load_existing_raw_zarr_mean_resized(
    step4_row: pd.Series | None,
    wavelength: str,
    fullres_shape: tuple[int, int],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mean(time) from an EXISTING Step4 raw_zarr, resized once for GUI display.

    Source grid is Step4 ``rotated_scaled``.  The returned array is a display copy
    in Step2/Step3 ``rotated_fullres``.  No BMP files are opened and no raw_zarr is
    built or overwritten here.
    """
    zarr_path = _raw_zarr_path_from_step4_row(step4_row, config)
    key = _raw_zarr_key(config)
    ds = xr.open_zarr(zarr_path, chunks="auto", consolidated=False)
    try:
        if key not in ds:
            raise KeyError(f"Variable {key!r} not found in raw_zarr {zarr_path}; available={list(ds.data_vars)}")
        raw = ds[key]
        required = {"time", "wavelength", "y", "x"}
        if not required.issubset(set(raw.dims)):
            raise ValueError(f"raw_zarr variable {key!r} must contain dims {sorted(required)}, got {raw.dims}")
        wave_vals = [_normalise_wave_label(v) for v in np.asarray(raw.coords["wavelength"].values)]
        target = _normalise_wave_label(wavelength)
        if target not in wave_vals:
            raise KeyError(f"Wavelength {target!r} not found in raw_zarr {zarr_path}; available={wave_vals}")
        wi = wave_vals.index(target)
        plane = raw.isel(wavelength=wi).transpose("time", "y", "x")
        n_frames = int(plane.sizes["time"])
        native_mean = np.asarray(plane.mean(dim="time", skipna=True).compute().values, dtype=np.float32)
    finally:
        try:
            ds.close()
        except Exception:
            pass

    if native_mean.ndim != 2:
        raise ValueError(f"Expected 2-D raw_zarr mean, got {native_mean.shape} from {zarr_path}")
    native_shape = tuple(int(v) for v in native_mean.shape)
    if native_shape == tuple(fullres_shape):
        display = native_mean.copy()
    else:
        display = cv2.resize(
            native_mean,
            (int(fullres_shape[1]), int(fullres_shape[0])),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
    return display, {
        "status": "existing_raw_zarr",
        "raw_zarr_path": str(zarr_path),
        "raw_zarr_key": key,
        "wavelength": target,
        "n_frames": n_frames,
        "native_shape": list(native_shape),
        "native_space": "rotated_scaled",
        "display_shape": [int(fullres_shape[0]), int(fullres_shape[1])],
        "display_space": "rotated_fullres",
        "resize_interpolation": "linear",
        "raw_zarr_rebuilt": False,
        "source_bmp_read": False,
    }


def _signed_difference(
    current: np.ndarray,
    baseline_registered: np.ndarray,
    brain: np.ndarray,
    support: np.ndarray,
    norm_mode: str,
) -> np.ndarray:
    """Prepare a brain-masked current-minus-baseline contrast map for display.

    Parameters
    ----------
    current, baseline_registered : numpy.ndarray
        Two-dimensional full-resolution images on the current-stem grid.
    brain : numpy.ndarray
        Current-stem brain mask, resized if necessary.
    support : numpy.ndarray
        Valid footprint of the warped baseline image.
    norm_mode : str
        ``none`` or ``mean_in_brain``; the latter scales each image by its
        finite mean over the intersection of brain and support.

    Returns
    -------
    numpy.ndarray
        Signed ``float32`` difference image with NaN outside supported brain.

    Notes
    -----
    Only display helper arrays are altered; this operation does not replace
    processed Step4 signals.
    """
    brain_here = resize_bool(brain, current.shape) if tuple(brain.shape) != tuple(current.shape) else np.asarray(brain, bool)
    current_work = np.asarray(current, np.float32).copy()
    baseline_work = np.asarray(baseline_registered, np.float32).copy()
    norm_mask = brain_here & np.asarray(support, bool) & np.isfinite(baseline_work) & np.isfinite(current_work)
    if norm_mode == "mean_in_brain":
        current_work = normalize_mean_in_mask(current_work, norm_mask)
        baseline_work = normalize_mean_in_mask(baseline_work, norm_mask)
    elif norm_mode != "none":
        raise ValueError("manual_gui.raw_layers.normalization must be none or mean_in_brain")
    diff = current_work - baseline_work
    diff[~brain_here] = np.nan
    diff[~np.asarray(support, bool)] = np.nan
    return diff.astype(np.float32)


def prepare_raw_layers(
    row: pd.Series,
    baseline_row: pd.Series | None,
    ref_row: pd.Series | None,
    brain: np.ndarray,
    config: dict[str, Any],
    paths: dict[str, Path],
    manual_overrides: pd.DataFrame,
    current_step4_row: pd.Series | None,
    baseline_step4_row: pd.Series | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Prepare current-only layers when the Day-PT baseline is unavailable.

    This is a *display* fallback, never a change to registration or signal
    processing. Baseline/difference layers are prepared as one coherent set;
    they are all omitted if the baseline, its BMPs or (when means are enabled)
    its existing Raw Zarr cannot be read. Current BMP and current mean layers
    remain independently available. No Zarr is built.
    """
    cfg = gui_cfg(config)["raw_layers"]
    path_cfg = get_path_resolving_config(config, "step3")
    norm_mode = str(cfg.get("normalization", "mean_in_brain")).lower()
    mean_enabled = bool((cfg.get("mean_images", {}) or {}).get("enabled", True))
    wavelengths = [str(v) for v in cfg.get("wavelengths", ["505", "656"])]

    layers: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {
        "current_registration_source": "not_needed_for_current_only",
        "baseline_registration_source": "not_available",
        "raw_sources": {},
        "mean_sources": {},
    }

    # Current reference BMPs are independent of baseline and Step4 Zarr.
    current_images: dict[str, tuple[np.ndarray, Path]] = {}
    for wl in wavelengths:
        try:
            current, current_path = load_wave_image(row, wl, path_cfg)
        except (FileNotFoundError, KeyError) as exc:
            meta["raw_sources"][wl] = {"status": "missing_current_bmp", "reason": str(exc)}
            print(f"[WARN] {row['stem']}: current {wl} nm BMP unavailable: {exc}")
            continue
        current_images[wl] = (current, current_path)
        layers[f"raw_{wl}"] = current.astype(np.float32)
        meta["raw_sources"][wl] = {"current_path": str(current_path)}
    if not current_images:
        raise FileNotFoundError(f"No current 505/656 reference BMPs for stem={row['stem']}")

    # A missing current Raw Zarr only removes the current mean images.
    current_means: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    if mean_enabled:
        for wl, (current, _) in current_images.items():
            try:
                arr, info = load_existing_raw_zarr_mean_resized(
                    current_step4_row, wl, tuple(current.shape), config,
                )
                current_means[wl] = (arr, info)
                layers[f"raw_{wl}_mean"] = arr.astype(np.float32)
                meta["mean_sources"].setdefault(wl, {})["current"] = info
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                meta["mean_sources"].setdefault(wl, {})["current"] = {
                    "status": "unavailable", "reason": str(exc),
                }
                print(f"[WARN] {row['stem']}: current {wl} nm mean unavailable: {exc}")

    # Preflight the entire baseline before creating *any* comparison layers.
    baseline_reason = ""
    baseline_images: dict[str, tuple[np.ndarray, Path]] = {}
    baseline_means: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    baseline_tform = current_tform = None
    baseline_reg_source = current_reg_source = "not_needed_for_current_only"
    if baseline_row is None:
        baseline_reason = "No Day-PT baseline row in metadata_registration.csv"
    elif ref_row is None:
        baseline_reason = "Reference registration row missing; comparison not possible"
    else:
        try:
            current_tform, current_reg_source, _ = damage.transform_for_row(
                row, manual_overrides, config,
            )
            baseline_tform, baseline_reg_source, _ = damage.transform_for_row(
                baseline_row, manual_overrides, config,
            )
            for wl in current_images:
                baseline_images[wl] = load_wave_image(baseline_row, wl, path_cfg)
            if mean_enabled:
                # Do not require Raw Zarr just to show the current image/movie,
                # but never present an incomplete baseline/diff mean set.
                _raw_zarr_path_from_step4_row(baseline_step4_row, config)
                for wl, (baseline, _) in baseline_images.items():
                    baseline_means[wl] = load_existing_raw_zarr_mean_resized(
                        baseline_step4_row, wl, tuple(baseline.shape), config,
                    )
        except Exception as exc:
            baseline_reason = str(exc)

    # An available baseline follows the pre-existing image-warp/difference
    # algorithms without changes. The set is committed only if all succeed.
    comparison_layers: dict[str, np.ndarray] = {}
    comparison_raw_meta: dict[str, dict[str, Any]] = {}
    comparison_mean_meta: dict[str, dict[str, Any]] = {}
    if not baseline_reason:
        try:
            for wl, (current, _) in current_images.items():
                baseline, baseline_path = baseline_images[wl]
                try:
                    ref_img, _ = load_wave_image(ref_row, wl, path_cfg)
                    ref_shape = tuple(ref_img.shape)
                except (FileNotFoundError, KeyError):
                    ref_shape = tuple(current.shape)

                brain_w = (resize_bool(brain, current.shape)
                           if tuple(brain.shape) != tuple(current.shape) else brain)
                baseline_cur, support = warp_baseline_to_current(
                    baseline, baseline_tform, current_tform, tuple(current.shape), ref_shape,
                )
                baseline_cur = baseline_cur.astype(np.float32, copy=False)
                baseline_cur[~support] = np.nan
                comparison_layers[f"baseline_{wl}_registered"] = baseline_cur
                comparison_layers[f"diff_signed_{wl}"] = _signed_difference(
                    current, baseline_cur, brain_w, support, norm_mode,
                )
                comparison_raw_meta[wl] = {
                    "baseline_path": str(baseline_path),
                    "baseline_display_space": "rotated_fullres_current_stem",
                    "baseline_display_intensity_normalized": False,
                    "difference_normalization": norm_mode,
                    "support_fraction_in_brain": float(
                        (support & brain_w).sum() / max(int(brain_w.sum()), 1)
                    ),
                }
                if mean_enabled and wl in current_means:
                    current_mean, current_mean_info = current_means[wl]
                    baseline_mean, baseline_mean_info = baseline_means[wl]
                    baseline_mean_cur, mean_support = warp_baseline_to_current(
                        baseline_mean, baseline_tform, current_tform,
                        tuple(current.shape), ref_shape,
                    )
                    baseline_mean_cur = baseline_mean_cur.astype(np.float32, copy=False)
                    baseline_mean_cur[~mean_support] = np.nan
                    comparison_layers[f"baseline_{wl}_registered_mean"] = baseline_mean_cur
                    comparison_layers[f"diff_signed_{wl}_mean"] = _signed_difference(
                        current_mean, baseline_mean_cur, brain_w, mean_support, norm_mode,
                    )
                    comparison_mean_meta[wl] = {
                        "current": current_mean_info,
                        "baseline": baseline_mean_info,
                        "baseline_display_space": "rotated_fullres_current_stem",
                        "difference_normalization": norm_mode,
                        "support_fraction_in_brain": float(
                            (mean_support & brain_w).sum() / max(int(brain_w.sum()), 1)
                        ),
                    }
        except Exception as exc:
            baseline_reason = f"Cannot prepare baseline comparisons: {exc}"
            comparison_layers.clear()
            comparison_raw_meta.clear()
            comparison_mean_meta.clear()

    if baseline_reason:
        meta["baseline_status"] = "unavailable_current_only"
        meta["baseline_reason"] = baseline_reason
        print(f"[WARN] {row['stem']}: baseline unavailable; current images/movies only: {baseline_reason}")
    else:
        meta["baseline_status"] = "available"
        meta["baseline_reason"] = ""
        meta["current_registration_source"] = current_reg_source
        meta["baseline_registration_source"] = baseline_reg_source
        layers.update(comparison_layers)
        for wl, info in comparison_raw_meta.items():
            meta["raw_sources"][wl].update(info)
        for wl, info in comparison_mean_meta.items():
            meta["mean_sources"][wl] = info
    return layers, meta


# =============================================================================
# Behavior RUN-SD
# =============================================================================


def behavior_annotations_path(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    """Resolve the behavior-annotation CSV path from configuration."""
    raw = gui_cfg(config)["run_sd"].get("annotations_path")
    if raw:
        p = resolve_flexible_path(raw, must_exist=False)
        return Path(p) if p is not None else Path(raw)
    return Path(paths["out_root"]) / "behavior_review" / "behavior_review_annotations.csv"


def _safe_str(v: Any) -> str:
    """Convert a possibly missing table value into a stable string."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def parse_run_intervals(value: Any) -> list[dict[str, float]]:
    """Parse and validate RUN interval pairs stored in a JSON cell."""
    try:
        items = json.loads(_safe_str(value) or "[]")
    except Exception:
        return []
    out: list[dict[str, float]] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict) or _safe_str(item.get("label")).strip().lower() != "run":
            continue
        try:
            t0 = float(item.get("t0_sec"))
            t1 = float(item.get("t1_sec"))
        except Exception:
            continue
        if not np.isfinite(t0) or not np.isfinite(t1):
            continue
        lo, hi = sorted((t0, t1))
        if hi > lo:
            out.append({"t0_sec": float(lo), "t1_sec": float(hi)})
    return out



def parse_rest_intervals(value: Any) -> list[dict[str, float]]:
    """Read explicitly labelled REST intervals from behavior_review_annotations.csv.

    A missing/invalid REST annotation is never interpreted as 100% RUN.
    """
    try:
        items = json.loads(_safe_str(value) or "[]")
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid label_intervals_json") from exc
    if not isinstance(items, list):
        raise ValueError("label_intervals_json must contain a list")
    out: list[dict[str, float]] = []
    for item in items:
        if not isinstance(item, dict) or _safe_str(item.get("label")).strip().lower() != "rest":
            continue
        try:
            t0, t1 = float(item["t0_sec"]), float(item["t1_sec"])
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError("REST interval missing valid t0_sec/t1_sec") from exc
        if not (np.isfinite(t0) and np.isfinite(t1)):
            raise ValueError("REST interval has non-finite time")
        lo, hi = sorted((t0, t1))
        if hi > lo:
            out.append({"t0_sec": lo, "t1_sec": hi})
    return out


def run_intervals_outside_rest(rest: Sequence[dict[str, float]], duration_sec: float) -> list[dict[str, float]]:
    """Complement of merged and clipped REST intervals on [0, duration_sec]."""
    if not np.isfinite(duration_sec) or duration_sec <= 0:
        raise ValueError("Recording duration must be positive")
    merged: list[list[float]] = []
    for r in sorted(rest, key=lambda item: item["t0_sec"]):
        lo = max(0.0, min(float(duration_sec), float(r["t0_sec"])))
        hi = max(0.0, min(float(duration_sec), float(r["t1_sec"])))
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    out: list[dict[str, float]] = []
    cursor = 0.0
    for lo, hi in merged:
        if lo > cursor:
            out.append({"t0_sec": cursor, "t1_sec": lo})
        cursor = hi
    if cursor < duration_sec:
        out.append({"t0_sec": cursor, "t1_sec": float(duration_sec)})
    return out


def load_latest_behavior_row(stem: str, config: dict[str, Any], paths: dict[str, Path]) -> tuple[pd.Series | None, Path]:
    """Read the latest available behavioral annotation for a stem."""
    p = behavior_annotations_path(config, paths)
    if not p.exists():
        return None, p
    df = pd.read_csv(p, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    if df.empty or "stem" not in df.columns:
        return None, p
    df = df[df["stem"].astype(str).eq(str(stem))].copy()
    if df.empty:
        return None, p
    return df.iloc[-1], p


def choose_processed_signal(ds: xr.Dataset, config: dict[str, Any]) -> tuple[str, xr.DataArray, bool]:
    """Choose the requested processed signal or an allowed fallback."""
    cfg = gui_cfg(config)["run_sd"]
    requested = str(cfg.get("signal", "dcHbT"))
    candidates = [requested]
    if bool(cfg.get("allow_signal_fallback", True)):
        candidates.extend(str(v) for v in cfg.get("fallback_signals", []) if str(v) not in candidates)
    for i, name in enumerate(candidates):
        if name in ds.data_vars:
            var = ds[name]
            if set(("time", "y", "x")).issubset(set(var.dims)):
                return name, var.transpose("time", "y", "x"), i > 0
    raise KeyError(f"None of requested RUN-SD signals exist. Tried {candidates}; available={list(ds.data_vars)}")


def compute_run_sd(
    stem: str,
    processed_path: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    out_dir: Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Compute a per-pixel RUN-period standard-deviation display map.

    Parameters
    ----------
    stem : str
        Recording identifier used to locate behavior annotations.
    processed_path : pathlib.Path
        Existing Step4 processed Zarr with time-resolved signals.
    config : dict
        Configuration under ``damage_assessment.manual_gui.run_sd``.
    paths : dict
        Paths to the behavior annotation table and outputs.
    out_dir : pathlib.Path
        Output location for cached display maps and optional segment outputs.

    Returns
    -------
    map_or_none : numpy.ndarray or None
        Per-pixel standard deviation map on the processed-signal ``(y, x)``
        grid, or None when the required REST annotation or signals are absent.
    metadata : dict
        RUN-interval selection and aggregation provenance or a skip reason.

    Notes
    -----
    RUN is derived from the complement of explicitly annotated REST intervals
    when the configured convention requests it. No missing REST labels are
    silently interpreted as full-recording RUN. The map is a display aid, not an
    independent behavioral measurement.
    """
    cfg = gui_cfg(config)["run_sd"]
    ann_row, ann_path = load_latest_behavior_row(stem, config, paths)
    meta: dict[str, Any] = {
        "enabled": bool(cfg.get("enabled", True)),
        "annotations_path": str(ann_path),
        "processed_path": str(processed_path),
        "status": "not_run",
    }
    if not bool(cfg.get("enabled", True)):
        meta["status"] = "disabled"
        return None, meta
    if ann_row is None:
        meta["status"] = "missing_behavior_annotation"
        return None, meta

    try:
        rest_intervals = parse_rest_intervals(ann_row.get("label_intervals_json", "[]"))
    except ValueError as exc:
        meta.update({"status": "invalid_rest_annotation", "detail": str(exc)})
        return None, meta
    if not rest_intervals:
        meta["status"] = "no_rest_intervals"
        return None, meta
    if not processed_path.exists():
        meta["status"] = "missing_processed_zarr"
        return None, meta

    try:
        import dask  # noqa: F401
        _chunks: Any = "auto"
    except Exception:
        _chunks = None
    ds = xr.open_zarr(processed_path, chunks=_chunks, consolidated=False)
    try:
        signal_name, signal, used_fallback = choose_processed_signal(ds, config)
        fps = float(cfg_get(config, "raw", "fps", default=np.nan))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("config raw.fps must be positive")
        n_time = int(signal.sizes["time"])
        min_sec = float(cfg.get("min_run_duration_sec", 2.0) or 0.0)
        intervals = run_intervals_outside_rest(rest_intervals, n_time / fps)
        intervals = [r for r in intervals if r["t1_sec"] - r["t0_sec"] >= min_sec]
        meta.update({"interval_source": "complement_of_rest", "n_rest_intervals": len(rest_intervals)})
        if not intervals:
            meta["status"] = "no_run_intervals_after_rest_complement"
            return None, meta

        numerator: np.ndarray | None = None
        denominator: np.ndarray | None = None
        segment_rows: list[dict[str, Any]] = []
        segment_dir = out_dir / "run_sd_segments_processed"
        if bool(cfg.get("save_segment_maps", True)):
            segment_dir.mkdir(parents=True, exist_ok=True)

        for run_index, r in enumerate(intervals):
            requested_i0 = int(round(float(r["t0_sec"]) * fps))
            requested_stop = int(round(float(r["t1_sec"]) * fps))
            i0 = max(0, requested_i0)
            i_stop = min(n_time, requested_stop)
            clipped = i0 != requested_i0 or i_stop != requested_stop
            if i_stop <= i0:
                segment_rows.append({
                    "run_index": run_index,
                    **r,
                    "i0": i0,
                    "i_stop": i_stop,
                    "n_samples": 0,
                    "interval_clipped": clipped,
                    "status": "empty_after_clipping",
                })
                continue

            seg = signal.isel(time=slice(i0, i_stop))
            sd = np.asarray(seg.std(dim="time", ddof=0).compute().values, dtype=np.float32)
            n_samples = int(i_stop - i0)
            finite = np.isfinite(sd)
            if numerator is None:
                numerator = np.zeros_like(sd, dtype=np.float64)
                denominator = np.zeros_like(sd, dtype=np.float64)
            numerator[finite] += sd[finite].astype(np.float64) * n_samples
            denominator[finite] += n_samples

            sd_path = ""
            if bool(cfg.get("save_segment_maps", True)):
                p = segment_dir / f"run_{run_index:03d}.npy"
                np.save(p, sd.astype(np.float32))
                sd_path = str(p)
            segment_rows.append({
                "run_index": run_index,
                **r,
                "duration_sec": float(r["t1_sec"] - r["t0_sec"]),
                "i0": i0,
                "i_stop": i_stop,
                "n_samples": n_samples,
                "interval_clipped": clipped,
                "sd_map_path": sd_path,
                "status": "ok",
            })

        pd.DataFrame(segment_rows).to_csv(out_dir / "run_sd_segments.csv", index=False)
        good = [r for r in segment_rows if r.get("status") == "ok"]
        if numerator is None or denominator is None or not good:
            meta.update({"status": "no_valid_run_segments", "segments": segment_rows})
            return None, meta
        stem_sd = np.full_like(numerator, np.nan, dtype=np.float32)
        valid = denominator > 0
        stem_sd[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
        native_path = out_dir / "run_sd_stem_processed.npy"
        np.save(native_path, stem_sd)
        meta.update({
            "status": "ok",
            "signal_requested": str(cfg.get("signal", "dcHbT")),
            "signal_used": signal_name,
            "signal_fallback_used": bool(used_fallback),
            "fps": fps,
            "n_time": n_time,
            "n_run_segments": len(good),
            "run_total_samples": int(sum(int(r["n_samples"]) for r in good)),
            "run_total_sec": float(sum(float(r["duration_sec"]) for r in good)),
            "aggregation": "sample_weighted_segment_sd",
            "native_map_path": str(native_path),
            "segments": segment_rows,
        })
        return stem_sd, meta
    finally:
        try:
            ds.close()
        except Exception:
            pass


# =============================================================================
# Homotopic FC display helpers (legacy hemorrhage overlay is disabled)
# =============================================================================


def _homotopic_cfg_parts(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate frozen Step7 and optional display-preview configuration."""
    raw = gui_cfg(config).get("homotopic", {}) or {}
    # Backward compatibility with the older flat homotopic block.
    auth = dict(raw.get("authoritative", {}) or {})
    for key in ("enabled", "folder_name", "mask_set", "display_units", "save_resized_z"):
        if key in raw:
            auth[key] = raw[key]
    full = dict(raw.get("full_preview", {}) or {})
    return auth, full


def _step7_root(paths: dict[str, Path], cfg: dict[str, Any]) -> Path:
    """Step7 root for one explicitly chosen REST branch and signal/filter variant."""
    branch = str(cfg.get("rest_variant", "with_rest"))
    variant = str(cfg.get("variant_id", "hbt_bandpass"))
    if branch not in {"with_rest", "without_rest"}:
        raise ValueError(f"Invalid GUI Step7 rest_variant: {branch}")
    if not variant or not all(c.isalnum() or c in "_-" for c in variant):
        raise ValueError(f"Invalid GUI Step7 variant_id: {variant!r}")
    return (Path(paths["out_root"]) /
            str(cfg.get("folder_name", "step7_functional_connectivity")) /
            branch / variant)


def _step7_stem_dirs(row: pd.Series, cfg: dict[str, Any], paths: dict[str, Path]) -> tuple[Path, Path, Path]:
    """Construct Step7 output paths for a recording and selected mask set."""
    mouse_id = _safe_str(row.get("mouse_id")) or _safe_str(row.get("group_id"))
    stem = _safe_str(row.get("stem"))
    mask_set = str(cfg.get("mask_set", "stroke_longitudinal"))
    root = _step7_root(paths, cfg)
    key = Path(mask_set) / mouse_id / stem
    return root / "stem_maps" / key, root / "fc_geometry" / key, root


def _resolve_step7_saved_dirs(
    row: pd.Series, cfg: dict[str, Any], paths: dict[str, Path]
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    """Find exact frozen Step7 maps, never recomputing or mixing variants.

    Full-run results take priority. If absent, an explicit subset key can be
    selected; otherwise use a subset only if EXACTLY one contains the stem.
    """
    maps_dir, geom_dir, root = _step7_stem_dirs(row, cfg, paths)
    expected = maps_dir / "homotopic_z_mean.npy"
    if expected.is_file():
        return maps_dir, geom_dir, {"status": "ok", "source": "full_run"}
    key = _safe_str(cfg.get("subset_output_key", "")).strip()
    if key and key.lower() not in {"none", "null"}:
        candidate_root = root / "_subset_runs" / key
        sub_maps = candidate_root / maps_dir.relative_to(root)
        sub_geom = candidate_root / geom_dir.relative_to(root)
        if (sub_maps / "homotopic_z_mean.npy").is_file():
            return sub_maps, sub_geom, {"status": "ok", "source": "selected_subset", "subset_output_key": key}
        return None, None, {"status": "missing", "path": str(sub_maps / "homotopic_z_mean.npy")}

    hits = sorted((root / "_subset_runs").glob(
        "*/" + str(expected.relative_to(root)).replace("\\", "/")
    ))
    hits = [h for h in hits if h.is_file()]
    if len(hits) == 1:
        sub_maps = hits[0].parent
        sub_geom = sub_maps.parents[3] / "fc_geometry" / maps_dir.relative_to(root / "stem_maps")
        return sub_maps, sub_geom, {"status": "ok", "source": "unique_subset", "subset_output_key": sub_maps.parents[3].name}
    if len(hits) > 1:
        return None, None, {"status": "ambiguous_subset", "candidates": [str(p) for p in hits]}
    return None, None, {"status": "missing", "path": str(expected)}


def load_homotopic_authoritative(
    row: pd.Series, config: dict[str, Any], paths: dict[str, Path]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Load one existing Step7 homotopic Fisher-z map; display transform only."""
    cfg, _ = _homotopic_cfg_parts(config)
    if not bool(cfg.get("enabled", True)):
        return None, {"status": "disabled"}
    maps_dir, geom_dir, info = _resolve_step7_saved_dirs(row, cfg, paths)
    if maps_dir is None or geom_dir is None:
        return None, {**info, "mask_set": str(cfg.get("mask_set", "stroke_longitudinal"))}
    p = maps_dir / "homotopic_z_mean.npy"
    z = np.load(p, allow_pickle=False).astype(np.float32)
    units = str(cfg.get("display_units", "pearson_r")).lower()
    if units not in {"pearson_r", "fisher_z"}:
        raise ValueError(f"Unsupported Step7 display_units: {units!r}")
    display = np.tanh(z).astype(np.float32) if units == "pearson_r" else z.copy()
    geom_path = geom_dir / "fc_geometry.json"
    geom_meta: dict[str, Any] = {}
    if geom_path.exists():
        try:
            geom_meta = json.loads(geom_path.read_text(encoding="utf-8"))
        except Exception:
            geom_meta = {}
    return display, {
        **info,
        "path": str(p),
        "geometry_path": str(geom_path),
        "geometry_available": bool(geom_path.exists()),
        "geometry_fc_shape": geom_meta.get("fc_shape"),
        "geometry_processed_shape": geom_meta.get("processed_shape"),
        "homotopic_map_side": geom_meta.get("homotopic_map_side") or geom_meta.get("homotopic_map_hemisphere"),
        "mask_set": str(cfg.get("mask_set", "stroke_longitudinal")),
        "rest_variant": str(cfg.get("rest_variant", "with_rest")),
        "variant_id": str(cfg.get("variant_id", "hbt_bandpass")),
        "source_units": "fisher_z",
        "display_units": units,
        "source_shape": list(z.shape),
        "z_map": z,
    }


def load_existing_step7_gbc(
    row: pd.Series, config: dict[str, Any], paths: dict[str, Path]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Optional read-only existing Step7 GBC map, same selected branch/variant."""
    cfg, _ = _homotopic_cfg_parts(config)
    if not bool(cfg.get("enabled", True)) or not bool(cfg.get("show_gbc", False)):
        return None, {"status": "disabled"}
    maps_dir, _, info = _resolve_step7_saved_dirs(row, cfg, paths)
    if maps_dir is None:
        return None, info
    p = maps_dir / "gbc_z_mean.npy"
    if not p.is_file():
        return None, {"status": "missing", "path": str(p)}
    z = np.load(p, allow_pickle=False).astype(np.float32)
    units = str(cfg.get("display_units", "pearson_r")).lower()
    out = np.tanh(z).astype(np.float32) if units == "pearson_r" else z.copy()
    return out, {**info, "path": str(p), "source_units": "fisher_z", "display_units": units}


def _full_preview_temporal_cfg(config: dict[str, Any], full_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve temporal preprocessing settings for display-only FC previews."""
    defaults = {
        "detrend_enabled": True,
        "detrend_type": "linear",
        "filter_enabled": True,
        "low_hz": 0.08,
        "high_hz": 0.40,
        "order": 4,
        "synthetic_odd_pad_sec": 20.0,
        "variance_epsilon": 1.0e-12,
    }
    own = dict(full_cfg.get("temporal", {}) or {})
    if bool(own.get("inherit_functional_connectivity", True)):
        fc = dict((config.get("functional_connectivity", {}) or {}).get("temporal", {}) or {})
        defaults.update({k: v for k, v in fc.items() if v is not None})
    defaults.update({k: v for k, v in own.items() if k != "inherit_functional_connectivity" and v is not None})
    return defaults


def _full_preview_spatial_cfg(config: dict[str, Any], full_cfg: dict[str, Any]) -> dict[str, Any]:
    """Resolve spatial resizing settings for display-only FC previews."""
    defaults = {"scale": 0.50, "min_valid_fraction": 0.50}
    own = dict(full_cfg.get("spatial", {}) or {})
    if bool(own.get("inherit_functional_connectivity", True)):
        fc = dict((config.get("functional_connectivity", {}) or {}).get("spatial", {}) or {})
        defaults.update({k: v for k, v in fc.items() if v is not None})
    defaults.update({k: v for k, v in own.items() if k != "inherit_functional_connectivity" and v is not None})
    return defaults


def _rest_table_for_full_preview(config: dict[str, Any], paths: dict[str, Path], full_cfg: dict[str, Any]) -> Path | None:
    """Locate the selected table containing REST intervals for preview."""
    custom = full_cfg.get("rest_segments_path")
    if custom:
        p = resolve_flexible_path(custom, must_exist=False)
        return Path(p) if p is not None else Path(str(custom))
    root = _step7_root(paths, full_cfg)
    candidates = [
        root / "tables" / "rest_segments_used.csv",
        root / "tables" / "rest_segments_final.csv",
        Path(paths["out_root"]) / "behavior_review" / "rest_segments_final.csv",
    ]
    return next((p for p in candidates if p.exists()), candidates[0])


def _load_rest_intervals_for_stem(stem: str, config: dict[str, Any], paths: dict[str, Path], full_cfg: dict[str, Any]) -> tuple[list[dict[str, float]], str]:
    """Read accepted REST intervals for a selected recording."""
    p = _rest_table_for_full_preview(config, paths, full_cfg)
    if p is None or not p.exists():
        return [], str(p or "")
    df = pd.read_csv(p, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    if df.empty or "stem" not in df.columns or "t0_sec" not in df.columns or "t1_sec" not in df.columns:
        return [], str(p)
    df = df[df["stem"].astype(str).eq(str(stem))].copy()
    out: list[dict[str, float]] = []
    seen: set[tuple[float, float]] = set()
    for _, r in df.iterrows():
        try:
            t0 = float(r["t0_sec"]); t1 = float(r["t1_sec"])
        except Exception:
            continue
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            key = (round(t0, 9), round(t1, 9))
            if key in seen:
                continue
            seen.add(key)
            out.append({"t0_sec": t0, "t1_sec": t1})
    return out, str(p)


def _select_preview_signal(ds: xr.Dataset, full_cfg: dict[str, Any]) -> tuple[str, bool]:
    """Choose a processed signal for the display-only homotopic preview."""
    requested = str(full_cfg.get("signal", "dcHbT"))
    if requested in ds.data_vars:
        return requested, False
    if bool(full_cfg.get("allow_signal_fallback", False)):
        for name in [str(v) for v in full_cfg.get("fallback_signals", [])]:
            if name in ds.data_vars:
                return name, True
    raise KeyError(f"Full homotopic preview signal {requested!r} missing. Available: {list(ds.data_vars)}")


def _mask_aware_resize_movie_area_local(movie: np.ndarray, valid_mask: np.ndarray, shape: tuple[int, int], min_valid_fraction: float) -> np.ndarray:
    """Resize a time series to an FC grid using validity-weighted averaging.

    Parameters
    ----------
    movie : numpy.ndarray
        Input ``float32`` movie with shape ``(time, y, x)``.
    valid_mask : numpy.ndarray
        Boolean source-grid support of shape ``(y, x)``.
    shape : tuple of int
        Target FC grid shape ``(y, x)``.
    min_valid_fraction : float
        Required source-pixel support fraction per resized pixel.

    Returns
    -------
    numpy.ndarray
        Resized movie of shape ``(time, *shape)`` with invalid values marked NaN.

    Notes
    -----
    Used only for the optional display-only homotopic preview and does not
    change analytical Step7 maps.
    """
    arr = np.asarray(movie, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    h, w = map(int, shape)
    if arr.ndim != 3 or arr.shape[1:] != valid.shape:
        raise ValueError("movie/valid_mask shape mismatch for homotopic preview")
    static_fraction = cv2.resize(valid.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    out = np.full((arr.shape[0], h, w), np.nan, dtype=np.float32)
    for i in range(arr.shape[0]):
        frame = arr[i]
        finite_valid = valid & np.isfinite(frame)
        weights = finite_valid.astype(np.float32)
        numerator = cv2.resize(np.where(finite_valid, frame, 0.0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        denom = cv2.resize(weights, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        good = (denom >= float(min_valid_fraction)) & (static_fraction > 0)
        temp = np.full((h, w), np.nan, dtype=np.float32)
        temp[good] = numerator[good] / np.maximum(denom[good], 1e-12)
        out[i] = temp
    return out


def _bandpass_matrix(X: np.ndarray, fps: float, temporal: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Apply the configured temporal band-pass filter to signal columns.

    Parameters
    ----------
    X : numpy.ndarray
        Two-dimensional time-by-pixel matrix ``(time, pixels)``.
    fps : float
        Sampling frequency in Hz.
    temporal : dict
        Filter enable flag, lower and upper cutoff frequencies (Hz), and order.

    Returns
    -------
    filtered : numpy.ndarray
        Filtered ``float32`` matrix with the same shape as ``X``.
    pad_frames : int
        Number of synthetic padding frames used for edge handling, or zero.

    Notes
    -----
    This function belongs to the disabled-by-default visualization preview;
    its output is not substituted for authoritative Step7 connectivity.
    """
    if not bool(temporal.get("filter_enabled", True)):
        return np.asarray(X, dtype=np.float32), 0
    low = float(temporal.get("low_hz", 0.08)); high = float(temporal.get("high_hz", 0.40)); order = int(temporal.get("order", 4))
    if not (0 < low < high < fps / 2):
        raise ValueError(f"Invalid homotopic preview bandpass {low}-{high} Hz for fps={fps}")
    sos = butter(order, [low, high], btype="bandpass", fs=fps, output="sos")
    requested = max(0, int(round(float(temporal.get("synthetic_odd_pad_sec", 20.0)) * fps)))
    actual = min(requested, max(0, int(X.shape[0]) - 2))
    if actual == 0:
        out = sosfiltfilt(sos, X, axis=0, padtype=None, padlen=0)
    else:
        out = sosfiltfilt(sos, X, axis=0, padtype="odd", padlen=actual)
    return np.asarray(out, dtype=np.float32), int(actual)


def _signed_midline_distance_grid(shape: tuple[int, int], origin_xy: np.ndarray, hat_xy: np.ndarray) -> np.ndarray:
    """Compute signed pixel distances from an image midline."""
    h, w = map(int, shape)
    yy, xx = np.mgrid[:h, :w]
    hat = np.asarray(hat_xy, dtype=float)
    norm = float(np.linalg.norm(hat))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid midline_hat_xy_fc")
    hat = hat / norm
    normal = np.array([-hat[1], hat[0]], dtype=float)
    return (xx - float(origin_xy[0])) * normal[0] + (yy - float(origin_xy[1])) * normal[1]


def _reflect_xy(x: np.ndarray, y: np.ndarray, origin_xy: np.ndarray, hat_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reflect image coordinates across the supplied midline."""
    hat = np.asarray(hat_xy, dtype=float)
    hat = hat / max(float(np.linalg.norm(hat)), 1e-12)
    normal = np.array([-hat[1], hat[0]], dtype=float)
    dx = np.asarray(x, dtype=float) - float(origin_xy[0])
    dy = np.asarray(y, dtype=float) - float(origin_xy[1])
    d = dx * normal[0] + dy * normal[1]
    return np.asarray(x, dtype=float) - 2.0 * d * normal[0], np.asarray(y, dtype=float) - 2.0 * d * normal[1]


def _infer_source_sign(authoritative_z: np.ndarray | None, dist: np.ndarray, geom_dir: Path, map_side: str) -> tuple[int, str]:
    """Determine which midline side is used as the homotopic source."""
    if authoritative_z is not None and authoritative_z.shape == dist.shape:
        finite = np.isfinite(authoritative_z)
        vals = dist[finite]
        vals = vals[np.isfinite(vals) & (np.abs(vals) > 0.25)]
        if vals.size:
            med = float(np.median(vals))
            if abs(med) > 1e-9:
                return (1 if med > 0 else -1), "authoritative_homotopic_map"
    side = str(map_side or "").strip().lower()
    names = [
        ("left", ["left_hemisphere_fc.npy", "left_fc.npy"]),
        ("right", ["right_hemisphere_fc.npy", "right_fc.npy"]),
    ]
    for label, files in names:
        if side and label != side:
            continue
        for name in files:
            p = geom_dir / name
            if p.exists():
                m = np.load(p).astype(bool)
                if m.shape == dist.shape and m.any():
                    med = float(np.median(dist[m]))
                    if abs(med) > 1e-9:
                        return (1 if med > 0 else -1), f"{name}:{label}"
    return 1, "fallback_positive_signed_side"


def _build_continuous_homotopic_pairs(brain_fc: np.ndarray, origin_xy: np.ndarray, hat_xy: np.ndarray, source_sign: int, midline_exclusion_px: float, mutual_roundtrip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Pair eligible brain pixels with their reflections across the image midline.

    Parameters
    ----------
    brain_fc : numpy.ndarray
        Boolean FC-grid spatial support of shape ``(y, x)``.
    origin_xy, hat_xy : numpy.ndarray
        Midline origin and direction in FC-grid pixels, ordered ``(x, y)``.
    source_sign : int
        Selected side of the midline.
    midline_exclusion_px : float
        Exclusion distance near the midline in FC-grid pixels.
    mutual_roundtrip : bool
        Require geometrically reciprocal pairing when enabled.

    Returns
    -------
    source_indices, target_indices : tuple of numpy.ndarray
        Flat image indices of corresponding reflected pixel pairs.

    Notes
    -----
    Uses continuous brain support without the atlas ROI labels and boundary
    buffer rules applied by the authoritative Step7 analysis.
    """
    shape = tuple(map(int, brain_fc.shape))
    h, w = shape
    dist = _signed_midline_distance_grid(shape, origin_xy, hat_xy)
    src_mask = np.asarray(brain_fc, bool) & ((dist * int(source_sign)) > float(midline_exclusion_px))
    ys, xs = np.nonzero(src_mask)
    xr, yr = _reflect_xy(xs.astype(float), ys.astype(float), origin_xy, hat_xy)
    xi = np.rint(xr).astype(int); yi = np.rint(yr).astype(int)
    good = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    good_idx = np.flatnonzero(good)
    if good_idx.size:
        keep = brain_fc[yi[good_idx], xi[good_idx]] & ((dist[yi[good_idx], xi[good_idx]] * int(source_sign)) < -float(midline_exclusion_px))
        good[good_idx] &= keep
    if bool(mutual_roundtrip):
        idx = np.flatnonzero(good)
        if idx.size:
            xback, yback = _reflect_xy(xi[idx].astype(float), yi[idx].astype(float), origin_xy, hat_xy)
            roundtrip = (np.rint(xback).astype(int) == xs[idx]) & (np.rint(yback).astype(int) == ys[idx])
            good[idx] &= roundtrip
    src_flat = (ys[good] * w + xs[good]).astype(np.int64)
    dst_flat = (yi[good] * w + xi[good]).astype(np.int64)
    # Remove accidental duplicate destinations caused by raster rounding: closest geometric pair wins.
    if src_flat.size:
        _, first = np.unique(dst_flat, return_index=True)
        first = np.sort(first)
        src_flat = src_flat[first]; dst_flat = dst_flat[first]
    return src_flat, dst_flat


def _pearson_pairs(X: np.ndarray, src_cols: np.ndarray, dst_cols: np.ndarray) -> np.ndarray:
    """Compute vectorized Pearson correlations for selected pixel pairs.

    Parameters
    ----------
    X : numpy.ndarray
        Time-by-pixel signal matrix ``(time, pixels)``.
    src_cols, dst_cols : numpy.ndarray
        Matching integer column indices for each homotopic pair.

    Returns
    -------
    numpy.ndarray
        Pearson coefficients, one per selected pair; invalid denominators
        yield NaN according to the implementation.
    """
    a = np.asarray(X[:, src_cols], dtype=np.float64)
    b = np.asarray(X[:, dst_cols], dtype=np.float64)
    a -= np.mean(a, axis=0, keepdims=True); b -= np.mean(b, axis=0, keepdims=True)
    den = np.sqrt(np.sum(a * a, axis=0) * np.sum(b * b, axis=0))
    r = np.full(src_cols.shape, np.nan, dtype=np.float32)
    good = np.isfinite(den) & (den > 0)
    r[good] = (np.sum(a[:, good] * b[:, good], axis=0) / den[good]).astype(np.float32)
    return r


def compute_full_homotopic_preview(
    row: pd.Series,
    processed_path: Path,
    brain_fullres: np.ndarray,
    config: dict[str, Any],
    paths: dict[str, Path],
    out_dir: Path,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Build a continuous bilateral homotopic map for lesion-review DISPLAY only.

    Differences from authoritative Step7:
    - uses current-stem brain mask as continuous spatial support;
    - does NOT require matching analytical ROI labels and therefore has no ROI boundary gaps;
    - writes the same pair value on both mirrored members for intuitive bilateral display;
    - never replaces or modifies frozen Step7 outputs.
    """
    _, cfg = _homotopic_cfg_parts(config)
    if not bool(cfg.get("enabled", True)):
        return None, {"status": "disabled", "preview_only": True}
    maps_dir, geom_dir, root = _step7_stem_dirs(row, cfg, paths)
    geom_path = geom_dir / "fc_geometry.json"
    if not geom_path.exists():
        return None, {"status": "missing_step7_geometry", "geometry_path": str(geom_path), "preview_only": True}
    try:
        geom = json.loads(geom_path.read_text(encoding="utf-8"))
        fc_shape = tuple(int(v) for v in geom["fc_shape"])
        processed_shape = tuple(int(v) for v in geom["processed_shape"])
        origin = np.asarray(geom["midline_origin_xy_fc"], dtype=float)
        hat = np.asarray(geom["midline_hat_xy_fc"], dtype=float)
    except Exception as exc:
        return None, {"status": "invalid_step7_geometry", "error": repr(exc), "geometry_path": str(geom_path), "preview_only": True}

    rest, rest_path = _load_rest_intervals_for_stem(str(row["stem"]), config, paths, cfg)
    if not rest:
        return None, {"status": "no_accepted_rest", "rest_segments_path": rest_path, "geometry_path": str(geom_path), "preview_only": True}
    if processed_path is None or not Path(processed_path).exists():
        return None, {"status": "missing_processed_zarr", "processed_path": str(processed_path), "preview_only": True}

    brain_processed = resize_bool(brain_fullres, processed_shape)
    brain_fc = resize_bool(brain_fullres, fc_shape)
    dist = _signed_midline_distance_grid(fc_shape, origin, hat)
    auth_z = None
    auth_path = maps_dir / "homotopic_z_mean.npy"
    if auth_path.exists():
        try:
            auth_z = np.load(auth_path).astype(np.float32)
        except Exception:
            auth_z = None
    source_sign, sign_source = _infer_source_sign(auth_z, dist, geom_dir, geom.get("homotopic_map_side", ""))
    src_flat, dst_flat = _build_continuous_homotopic_pairs(
        brain_fc,
        origin,
        hat,
        source_sign,
        float(cfg.get("midline_exclusion_px_fc", 1.0)),
        bool(cfg.get("mutual_roundtrip", True)),
    )
    if src_flat.size == 0:
        return None, {"status": "no_continuous_pairs", "preview_only": True, "geometry_path": str(geom_path)}

    temporal = _full_preview_temporal_cfg(config, cfg)
    spatial = _full_preview_spatial_cfg(config, cfg)
    fps = float(cfg_get(config, "raw", "fps", default=np.nan))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("config raw.fps must be positive for full homotopic preview")

    try:
        ds = xr.open_zarr(processed_path, chunks="auto", consolidated=False)
    except Exception:
        ds = xr.open_zarr(processed_path, chunks=None, consolidated=False)
    segment_maps: list[np.ndarray] = []
    segment_rows: list[dict[str, Any]] = []
    try:
        signal_name, used_fallback = _select_preview_signal(ds, cfg)
        var = ds[signal_name].transpose("time", "y", "x")
        n_total = int(var.sizes["time"])
        if tuple(var.shape[1:]) != tuple(processed_shape):
            processed_shape = tuple(int(v) for v in var.shape[1:])
            brain_processed = resize_bool(brain_fullres, processed_shape)
        eps = float(temporal.get("variance_epsilon", 1e-12))
        fisher_clip = float(cfg.get("fisher_clip", 0.999999))
        bilateral_display = bool(cfg.get("bilateral_display", True))
        seg_dir = out_dir / "homotopic_full_preview_segments_fc"
        if bool(cfg.get("save_segment_maps", False)):
            seg_dir.mkdir(parents=True, exist_ok=True)

        for seg_i, rr in enumerate(rest):
            t0 = float(rr["t0_sec"]); t1 = float(rr["t1_sec"])
            i0 = int(round(t0 * fps)); i1 = i0 + int(round((t1 - t0) * fps))
            rec = {"segment_index": seg_i, "t0_sec": t0, "t1_sec": t1, "i0": i0, "i_stop": i1}
            if i0 < 0 or i0 >= n_total or i1 <= i0 or i1 > n_total:
                rec.update({"status": "out_of_bounds", "n_pairs": 0}); segment_rows.append(rec); continue
            movie = var.isel(time=slice(i0, i1)).load().values.astype(np.float32, copy=False)
            if movie.shape[0] < 4:
                rec.update({"status": "too_short", "n_pairs": 0}); segment_rows.append(rec); continue
            movie_fc = _mask_aware_resize_movie_area_local(movie, brain_processed, fc_shape, float(spatial.get("min_valid_fraction", 0.50)))
            finite_all = np.all(np.isfinite(movie_fc), axis=0)
            raw_std = np.full(fc_shape, np.nan, dtype=np.float32)
            if finite_all.any():
                raw_std[finite_all] = np.std(movie_fc[:, finite_all], axis=0, ddof=0)
            functional = brain_fc & finite_all & np.isfinite(raw_std) & (raw_std > eps)
            valid_flat = np.flatnonzero(functional.ravel()).astype(np.int64)
            if valid_flat.size == 0:
                rec.update({"status": "no_valid_pixels", "n_pairs": 0}); segment_rows.append(rec); continue
            X = movie_fc.reshape(movie_fc.shape[0], -1)[:, valid_flat].astype(np.float32, copy=False)
            if bool(temporal.get("detrend_enabled", True)):
                X = scipy_detrend(X, axis=0, type=str(temporal.get("detrend_type", "linear")), overwrite_data=False).astype(np.float32, copy=False)
            X, pad_frames = _bandpass_matrix(X, fps, temporal)
            good_cols = np.all(np.isfinite(X), axis=0) & (np.std(X, axis=0, ddof=1) > eps)
            X = X[:, good_cols]
            valid_flat = valid_flat[good_cols]
            colmap = np.full(int(np.prod(fc_shape)), -1, dtype=np.int64)
            colmap[valid_flat] = np.arange(valid_flat.size, dtype=np.int64)
            scols = colmap[src_flat]; dcols = colmap[dst_flat]
            pair_good = (scols >= 0) & (dcols >= 0)
            if not pair_good.any():
                rec.update({"status": "no_valid_pairs", "n_pairs": 0, "pad_frames": pad_frames}); segment_rows.append(rec); continue
            r = _pearson_pairs(X, scols[pair_good], dcols[pair_good])
            finite_r = np.isfinite(r)
            pair_src = src_flat[pair_good][finite_r]; pair_dst = dst_flat[pair_good][finite_r]
            r = np.clip(r[finite_r], -fisher_clip, fisher_clip)
            z = np.arctanh(r).astype(np.float32)
            zmap = np.full(int(np.prod(fc_shape)), np.nan, dtype=np.float32)
            zmap[pair_src] = z
            if bilateral_display:
                zmap[pair_dst] = z
            zmap = zmap.reshape(fc_shape)
            segment_maps.append(zmap)
            if bool(cfg.get("save_segment_maps", False)):
                np.save(seg_dir / f"segment_{seg_i:03d}_z.npy", zmap)
            rec.update({"status": "ok", "n_pairs": int(len(z)), "pad_frames": int(pad_frames)}); segment_rows.append(rec)

        pd.DataFrame(segment_rows).to_csv(out_dir / "homotopic_full_brain_preview_segments.csv", index=False)
        if not segment_maps:
            return None, {
                "status": "no_valid_rest_segments",
                "preview_only": True,
                "rest_segments_path": rest_path,
                "segments": segment_rows,
            }
        stack = np.stack(segment_maps, axis=0)
        count = np.sum(np.isfinite(stack), axis=0).astype(np.int16)
        total = np.nansum(stack, axis=0, dtype=np.float64)
        zmean = np.full(fc_shape, np.nan, dtype=np.float32)
        good = count > 0
        zmean[good] = (total[good] / count[good]).astype(np.float32)
        np.save(out_dir / "homotopic_full_brain_preview_z_fc.npy", zmean)
        np.save(out_dir / "homotopic_full_brain_preview_n_segments_fc.npy", count)
        units = str(cfg.get("display_units", "pearson_r")).lower()
        display = np.tanh(zmean).astype(np.float32) if units == "pearson_r" else zmean.copy()
        return display, {
            "status": "ok",
            "preview_only": True,
            "not_for_statistics": True,
            "same_analysis_roi_required": False,
            "analytical_roi_boundary_buffer_used": False,
            "bilateral_display_duplication": bool(cfg.get("bilateral_display", True)),
            "source_units": "fisher_z",
            "display_units": units,
            "signal_requested": str(cfg.get("signal", "dcHbT")),
            "signal_used": signal_name,
            "signal_fallback_used": bool(used_fallback),
            "processed_zarr": str(processed_path),
            "rest_segments_path": rest_path,
            "n_rest_segments_ok": int(len(segment_maps)),
            "n_geometry_pairs": int(src_flat.size),
            "source_sign": int(source_sign),
            "source_sign_inferred_from": sign_source,
            "midline_exclusion_px_fc": float(cfg.get("midline_exclusion_px_fc", 1.0)),
            "fc_shape": list(fc_shape),
            "processed_shape": list(processed_shape),
            "geometry_path": str(geom_path),
            "temporal": temporal,
            "spatial": spatial,
            "segments": segment_rows,
        }
    finally:
        try:
            ds.close()
        except Exception:
            pass


def processed_index_map(config_path: Path, stems: Sequence[str]) -> dict[str, pd.Series]:
    """Build a stem-keyed lookup table of Step4 processed recordings."""
    build_step4_index = _get_build_step4_index()
    df, _, _ = build_step4_index(config_path, stems=list(stems), include_bad=True, filters={})
    return {str(r["stem"]): r for _, r in df.iterrows()}


def _processed_path_from_row(
    row: pd.Series | None, config: dict[str, Any] | None = None,
) -> Path | None:
    """Find an existing processed Zarr path from a Step4 index record."""
    if row is None:
        return None
    for col in [
        "processed_zarr_path_resolved",
        "processed_zarr_motion_corrected_path_resolved",
        "processed_zarr_unaligned_path_resolved",
        "processed_zarr_path",
    ]:
        if col in row.index:
            found = _existing_step4_path(row.get(col), config)
            if found is not None:
                return found
    return None


def prepare_helpers_for_row(
    row: pd.Series,
    baseline_row: pd.Series | None,
    ref_row: pd.Series | None,
    config: dict[str, Any],
    paths: dict[str, Path],
    manual_overrides: pd.DataFrame,
    step4_row: pd.Series | None,
    baseline_step4_row: pd.Series | None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare display-only image, RUN-SD and frozen-FC helpers for one stem.

    Parameters
    ----------
    row : pandas.Series
        Current recording's Step2 metadata.
    baseline_row, ref_row : pandas.Series or None
        Optional baseline and reference metadata, respectively.
    config : dict
        Effective project YAML configuration.
    paths : dict
        Working analysis paths and GUI helper output roots.
    manual_overrides : pandas.DataFrame
        Step3 manual registration transforms.
    step4_row, baseline_step4_row : pandas.Series or None
        Step4 index records for available processed/raw Zarr stores.
    force : bool, optional
        Rebuild previously cached helpers if requested.

    Returns
    -------
    dict
        Paths and metadata for cached visualization arrays.

    Notes
    -----
    All arrays are prepared for GUI display in current-stem rotated-fullres
    coordinates; source Zarr and authoritative Step7 outputs are not modified.
    When baseline inputs are incomplete, only current-stem helpers are retained.
    """
    stem = str(row["stem"])
    out = helper_dir(paths, stem)
    meta_path = out / "display_meta.json"
    if meta_path.exists() and not force:
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            # Explicitly reuse the existing cache. --force-prepare rebuilds it.
            if old.get("status") == "ok":
                return old
        except Exception:
            pass

    raw_cfg = gui_cfg(config)["raw_layers"]
    path_cfg = get_path_resolving_config(config, "step3")
    first_wave = None
    current_first = None
    for candidate_wave in [str(v) for v in (raw_cfg.get("wavelengths") or ["505", "656"])]:
        try:
            current_first, _ = load_wave_image(row, candidate_wave, path_cfg)
            first_wave = candidate_wave
            break
        except (FileNotFoundError, KeyError):
            continue
    if current_first is None:
        raise FileNotFoundError(f"No current 505/656 reference image for stem={stem}")
    fullres_shape = tuple(current_first.shape)
    brain, brain_path = load_fullres_brain(row, paths, fullres_shape)

    layers, raw_meta = prepare_raw_layers(
        row, baseline_row, ref_row, brain, config, paths, manual_overrides,
        step4_row, baseline_step4_row,
    )

    proc_path = _processed_path_from_row(step4_row, config)
    run_sd_native = None
    if proc_path is None:
        run_meta: dict[str, Any] = {"status": "missing_step4_index_row"}
    else:
        run_sd_native, run_meta = compute_run_sd(stem, proc_path, config, paths, out)
    if run_sd_native is not None:
        run_sd_full = resize_float_nan(run_sd_native, fullres_shape)
        run_sd_full[~brain] = np.nan
        layers["run_sd_dcHbT"] = run_sd_full

    # Frozen authoritative Step7 map (kept separate).
    auth_map, auth_meta = load_homotopic_authoritative(row, config, paths)
    if auth_map is not None:
        afull = resize_float_nan(auth_map, fullres_shape)
        afull[~brain] = np.nan
        layers["homotopic_step7_authoritative"] = afull
        auth_cfg, _ = _homotopic_cfg_parts(config)
        if bool(auth_cfg.get("save_resized_z", True)) and isinstance(auth_meta.get("z_map"), np.ndarray):
            zfull = resize_float_nan(auth_meta["z_map"], fullres_shape)
            zfull[~brain] = np.nan
            np.save(out / "homotopic_step7_authoritative_z_resized_display.npy", zfull.astype(np.float32))
        auth_meta.pop("z_map", None)

    gbc_map, gbc_meta = load_existing_step7_gbc(row, config, paths)
    if gbc_map is not None:
        gfull = resize_float_nan(gbc_map, fullres_shape)
        gfull[~brain] = np.nan
        layers["gbc_step7_authoritative"] = gfull

    # No correlation recomputation and no hemorrhage file access.
    full_meta: dict[str, Any] = {"status": "disabled"}

    # Rename current brain_fullres -> explicit display-only brain mask.
    np.save(out / "brain_mask_display_current.npy", brain.astype(np.uint8))
    for name, arr in layers.items():
        np.save(out / f"{name}.npy", np.asarray(arr, dtype=np.float32))

    dcfg = gui_cfg(config)["display"]
    p_low, p_high = [float(v) for v in dcfg.get("percentiles", [2.0, 98.0])]
    contrasts: dict[str, list[float]] = {}
    for name, arr in layers.items():
        lo, hi = contrast_limits(arr, brain, p_low, p_high)
        contrasts[name] = [float(lo), float(hi)]

    meta = {
        "status": "ok",
        "stem": stem,
        "group_id": _safe_str(row.get("group_id")),
        "mouse_id": _safe_str(row.get("mouse_id")),
        "exp_day": _safe_str(row.get("exp_day")),
        "baseline_stem": str(baseline_row["stem"]) if baseline_row is not None else "",
        "baseline_status": raw_meta.get("baseline_status", "unavailable_current_only"),
        "baseline_reason": raw_meta.get("baseline_reason", ""),
        "ref_stem": _safe_str(row.get("ref_stem")),
        "fullres_shape": list(fullres_shape),
        "image_space": "rotated_fullres_current_stem",
        "brain_mask_display_path": brain_path,
        "brain_mask_role": "display_contrast_and_helper_support_only",
        "display_percentiles": [p_low, p_high],
        "contrast_limits": contrasts,
        "layers": sorted(layers.keys()),
        "raw": raw_meta,
        "run_sd": run_meta,
        "homotopic_authoritative": auth_meta,
        "gbc_authoritative": gbc_meta,
        "homotopic_full_preview": full_meta,
        "updated_at": _now_iso(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_prepared_helpers(stem: str, paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any], np.ndarray | None]:
    """Load cached helper metadata and display maps for one stem."""
    d = helper_dir(paths, stem)
    mp = d / "display_meta.json"
    if not mp.exists():
        raise FileNotFoundError(f"Prepared helper metadata not found: {mp}")
    with open(mp, "r", encoding="utf-8") as f:
        meta = json.load(f)
    brain_path = d / "brain_mask_display_current.npy"
    if not brain_path.exists():  # backward-compatible fallback
        brain_path = d / "brain_fullres.npy"
    brain = np.load(brain_path).astype(bool)
    layers: dict[str, np.ndarray] = {}
    for name in meta.get("layers", []):
        if name == "homotopic_full_brain_preview" or name.startswith("hemorrhage"):
            continue  # Ignore obsolete preview/hemorrhage in existing helper caches.
        p = d / f"{name}.npy"
        if p.exists():
            layers[name] = np.load(p).astype(np.float32)
    return layers, brain, meta, None


# =============================================================================
# Lesion polygon GUI
# =============================================================================


def mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Convert the boundary of a binary lesion mask to editable polygons."""
    if mask is None or not np.asarray(mask, bool).any():
        return []
    polygons: list[np.ndarray] = []
    for contour in find_contours(np.asarray(mask, dtype=np.uint8), 0.5):
        if contour.shape[0] >= 3:
            polygons.append(np.asarray(contour[:, :2], dtype=np.float32))
    return polygons


def polygons_to_mask(polygons: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Rasterize editable lesion polygons into a full-resolution binary mask."""
    out = np.zeros(shape, dtype=bool)
    for vertices in polygons:
        pts = np.asarray(vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] < 2:
            continue
        rr, cc = sk_polygon(pts[:, -2], pts[:, -1], shape=shape)
        out[rr, cc] = True
    return out


def _colormap_for_layer(name: str, config: dict[str, Any]) -> str:
    """Choose a Napari colormap based on layer name and configuration."""
    cmap = gui_cfg(config)["display"].get("colormaps", {}) or {}
    if name.startswith("raw_") or name.startswith("baseline_"):
        return str(cmap.get("raw", "gray"))
    if name.startswith("diff_signed"):
        return str(cmap.get("diff_signed", "coolwarm"))
    if name.startswith("run_sd"):
        return str(cmap.get("run_sd", "turbo"))
    if name.startswith("homotopic") or name.startswith("gbc_"):
        return str(cmap.get("homotopic", "turbo"))
    return "gray"



def add_processed_hbt_movie(
    viewer: Any,
    stem: str,
    exp_day: str,
    processed_path: Path | None,
    config: dict[str, Any],
    fullres_shape: tuple[int, int],
) -> xr.Dataset | None:
    """Display existing dcHbT Zarr lazily in Napari; caller closes returned dataset."""
    cfg = gui_cfg(config).get("hbt_movie", {})
    if not bool(cfg.get("enabled", True)):
        return None
    days = cfg.get("exp_days", [])
    if days and exp_day not in {str(v) for v in days}:
        return None
    if processed_path is None or not processed_path.exists():
        print(f"[WARN] {stem}: no existing processed Zarr for HbT movie: {processed_path}")
        return None
    try:
        import dask.array  # noqa: F401 (lazy movie; no full memory load)
    except ImportError:
        print(f"[WARN] {stem}: Dask is required to display HbT movie lazily; layer omitted")
        return None
    ds = None
    try:
        ds = xr.open_zarr(processed_path, chunks="auto", consolidated=False)
        signal_name = str(cfg.get("signal", "dcHbT"))
        if signal_name not in ds.data_vars:
            raise KeyError(f"{signal_name} not in {list(ds.data_vars)}")
        data = ds[signal_name]
        if not {"time", "y", "x"}.issubset(data.dims):
            raise ValueError(f"{signal_name} has dims {data.dims}; expected time,y,x")
        movie = data.transpose("time", "y", "x")
        h, w = int(movie.sizes["y"]), int(movie.sizes["x"])
        if min(h, w) <= 0:
            raise ValueError("HbT movie has zero spatial dimensions")
        viewer.add_image(
            movie.data, name="dcHbT_movie", visible=False,
            colormap=str(cfg.get("colormap", "turbo")),
            scale=(1.0, fullres_shape[0] / h, fullres_shape[1] / w),
        )
        print(f"[HbT] {stem}: {signal_name}, frames={movie.sizes['time']}, "
              f"native={h}x{w}, fps={cfg_get(config, 'raw', 'fps', default='?')}; "
              "frame number / fps = seconds")
        return ds
    except Exception as exc:
        if ds is not None:
            ds.close()
        print(f"[WARN] {stem}: could not open HbT movie: {exc}")
        return None


def install_raw_movie_selector(
    viewer: Any,
    stem: str,
    raw_zarr_path: Path | None,
    config: dict[str, Any],
    fullres_shape: tuple[int, int],
    lesion_layer: Any,
) -> dict[str, Any]:
    """Add an on-demand 505/656/both Qt selector; never write or recompute Zarr.

    The returned holder owns the lazily opened xarray dataset until Napari exits.
    The UI is created for each stem, so users can choose independently per review.
    """
    holder: dict[str, Any] = {"dataset": None, "widget": None}
    cfg = gui_cfg(config).get("raw_movie", {}) or {}
    if not bool(cfg.get("enabled", False)):
        return holder
    try:
        from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget
    except ImportError as exc:
        print(f"[WARN] {stem}: Qt movie selector unavailable: {exc}")
        return holder

    allowed = [str(w) for w in cfg.get("wavelengths", ["505", "656"])]
    allowed = [w for w in ["505", "656"] if w in allowed]
    if not allowed:
        return holder

    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.addWidget(QLabel("Existing raw Zarr movies (on demand)"))
    select = QComboBox()
    for w in allowed:
        select.addItem(f"{w} nm", [w])
    if len(allowed) > 1:
        select.addItem("Both 505 + 656", allowed)
    layout.addWidget(select)
    load_button = QPushButton("Load selected movie(s)")
    remove_button = QPushButton("Remove raw movies")
    layout.addWidget(load_button)
    layout.addWidget(remove_button)
    names = {"505": "raw_505_movie", "656": "raw_656_movie"}

    def _remove_layers() -> None:
        """Remove any previously loaded on-demand raw movie layers."""
        for name in names.values():
            try:
                viewer.layers.remove(viewer.layers[name])
            except KeyError:
                pass

    def _on_load() -> None:
        """Load selected raw movie wavelength layers from the existing Zarr store."""
        try:
            if raw_zarr_path is None or not raw_zarr_path.exists():
                raise FileNotFoundError(f"Existing raw Zarr not found: {raw_zarr_path}")
            if holder["dataset"] is None:
                ds = xr.open_zarr(raw_zarr_path, chunks="auto", consolidated=False)
                key = _raw_zarr_key(config)
                if key not in ds:
                    ds.close()
                    raise KeyError(f"{key!r} absent from raw Zarr; vars={list(ds.data_vars)}")
                raw = ds[key]
                if not {"time", "wavelength", "y", "x"}.issubset(raw.dims):
                    ds.close()
                    raise ValueError(f"Expected time,wavelength,y,x in {key}, got {raw.dims}")
                holder["dataset"] = ds
            ds = holder["dataset"]
            raw = ds[_raw_zarr_key(config)]
            wave_labels = [_normalise_wave_label(v) for v in np.asarray(raw.coords["wavelength"].values)]
            selection = [str(w) for w in select.currentData()]
            for wave in selection:
                if wave not in wave_labels:
                    raise KeyError(f"{wave} nm absent from raw Zarr, available={wave_labels}")
            # Validate requested channels before removing any previous layers.
            movies = {}
            for wave in selection:
                movie = raw.isel(wavelength=wave_labels.index(wave)).transpose("time", "y", "x")
                h, w = int(movie.sizes["y"]), int(movie.sizes["x"])
                if min(h, w) <= 0:
                    raise ValueError(f"Invalid raw movie shape for {wave}: {h}x{w}")
                movies[wave] = (movie, h, w)
            _remove_layers()
            for wave, (movie, h, w) in movies.items():
                viewer.add_image(
                    movie.data,
                    name=names[wave],
                    visible=True,
                    colormap=str(cfg.get("colormap", "gray")),
                    scale=(1.0, fullres_shape[0] / h, fullres_shape[1] / w),
                )
            # Keep lesion outline above dynamically inserted movie layers.
            viewer.layers.move(viewer.layers.index(lesion_layer), len(viewer.layers) - 1)
            print(f"[Raw movie] {stem}: loaded {', '.join(selection)} nm from {raw_zarr_path}")
        except Exception as exc:
            print(f"[WARN] {stem}: could not load raw movie: {exc}")

    load_button.clicked.connect(_on_load)
    remove_button.clicked.connect(_remove_layers)
    viewer.window.add_dock_widget(panel, name="Raw movies 505/656", area="right")
    holder["widget"] = panel  # retain callbacks and controls during Napari review
    return holder


def review_lesion_napari(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    processed_path: Path | None = None,
    raw_zarr_path: Path | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Open Napari to review and optionally edit a lesion polygon.

    Parameters
    ----------
    row : pandas.Series
        Current stem metadata including the selected lesion file path.
    config : dict
        GUI settings under ``damage_assessment.manual_gui``.
    paths : dict
        Paths to cached display helpers and masks.
    processed_path, raw_zarr_path : pathlib.Path or None, optional
        Existing Zarr stores for optional read-only movie layers.

    Returns
    -------
    lesion_or_none : numpy.ndarray or None
        Boolean full-resolution lesion mask, or None when nothing should be saved.
    review_metadata : dict
        Review action, polygon count, and baseline availability metadata.

    Notes
    -----
    ``lesion_polygon_EDIT`` is the only editable lesion layer. Raw, difference,
    RUN-SD and Step7 helper layers are display-only. Unchanged polygons are
    recognized before rasterization to avoid rewriting legacy lesion masks.
    """
    if napari is None:
        raise ImportError("Napari is required for manual lesion review")
    stem = str(row["stem"])
    layers, brain, meta, hemorrhage = load_prepared_helpers(stem, paths)
    shape = tuple(int(v) for v in meta["fullres_shape"])
    mpaths = damage.damage_mask_paths(paths, stem)
    existing = None
    if mpaths["lesion_final"].exists():
        if not bool(gui_cfg(config)["review"].get("preload_existing_lesion", True)):
            raise ValueError("Refusing to edit an existing mask while preload_existing_lesion=False")
        existing = np.load(mpaths["lesion_final"]).astype(bool)
        if tuple(existing.shape) != shape:
            raise ValueError(
                f"Existing lesion mask shape {existing.shape} differs from display {shape}; "
                "check image geometry; no automatic resizing is allowed"
            )

    viewer = napari.Viewer(title=f"Manual lesion review: {stem}")
    default_visible = str(gui_cfg(config)["display"].get("default_visible_layer", "raw_505"))
    # If the preferred wavelength is technically absent, show another current
    # still image rather than opening Napari with every display layer hidden.
    if default_visible not in layers:
        default_visible = next(
            (name for name in ("raw_505", "raw_656", "raw_505_mean", "raw_656_mean") if name in layers),
            next(iter(layers), default_visible),
        )
    opacity = float(gui_cfg(config)["display"].get("layer_opacity", 1.0))
    contrast_map = meta.get("contrast_limits", {}) or {}

    preferred_order = [
        "raw_505", "baseline_505_registered", "diff_signed_505",
        "raw_505_mean", "baseline_505_registered_mean", "diff_signed_505_mean",
        "raw_656", "baseline_656_registered", "diff_signed_656",
        "raw_656_mean", "baseline_656_registered_mean", "diff_signed_656_mean",
        "run_sd_dcHbT", "homotopic_step7_authoritative", "gbc_step7_authoritative",
    ]
    names = [n for n in preferred_order if n in layers] + [n for n in layers if n not in preferred_order]
    for name in names:
        arr = layers[name]
        kwargs: dict[str, Any] = {
            "name": name,
            "visible": name == default_visible,
            "opacity": opacity,
            "colormap": _colormap_for_layer(name, config),
        }
        lim = contrast_map.get(name)
        if lim and len(lim) == 2 and np.isfinite(lim).all():
            kwargs["contrast_limits"] = tuple(float(v) for v in lim)
        try:
            viewer.add_image(arr, **kwargs)
        except Exception:
            kwargs.pop("colormap", None)
            viewer.add_image(arr, **kwargs)

    movie_ds = add_processed_hbt_movie(
        viewer, stem, _safe_str(row.get("exp_day")), processed_path, config, shape,
    )

    brain_layer = viewer.add_labels(brain.astype(np.uint8), name="brain_mask_display_current", opacity=0.12, visible=False)
    brain_layer.editable = False

    initial_polygons = mask_to_polygons(existing) if existing is not None else []
    rcfg = gui_cfg(config)["review"]
    # Keep lesion visible across every helper image as an OUTLINE ONLY.
    # A fully transparent polygon face avoids obscuring raw/diff/FC layers while
    # preserving normal Napari polygon editing and vertex handles.
    lesion_layer = viewer.add_shapes(
        initial_polygons,
        shape_type="polygon" if initial_polygons else "polygon",
        name="lesion_polygon_EDIT",
        edge_color=rcfg.get("lesion_edge_color", "yellow"),
        face_color=[1.0, 1.0, 0.0, 0.0],
        edge_width=float(rcfg.get("lesion_edge_width", 2.0)),
    )

    print()
    print("Manual lesion GUI")
    print("stem:", stem)
    print("baseline:", meta.get("baseline_stem", "") or "not available")
    if meta.get("baseline_status") == "unavailable_current_only":
        print("[WARN] Current-only review: baseline/difference layers omitted.")
        print("[WARN] Reason:", meta.get("baseline_reason", ""))
    print("Draw one or more polygons in layer 'lesion_polygon_EDIT'. Only the lesion outline is displayed; polygon fill is transparent.")
    print("Close Napari: unchanged existing masks will NOT be rewritten.")
    raw_movies = install_raw_movie_selector(
        viewer, stem, raw_zarr_path, config, shape, lesion_layer,
    )

    print("RUN is defined as all intervals outside annotated REST.")
    print("Raw 505/656 movies: select and load in the Napari dock when needed.")
    print()
    baseline_review_meta = {
        "baseline_status": str(meta.get("baseline_status", "unknown")),
        "baseline_reason": str(meta.get("baseline_reason", "")),
        "baseline_used_for_comparison": bool(meta.get("baseline_status") == "available"),
    }
    try:
        napari.run()
    finally:
        if movie_ds is not None:
            movie_ds.close()
        if raw_movies.get("dataset") is not None:
            raw_movies["dataset"].close()

    polygons = [np.asarray(p, dtype=float) for p in lesion_layer.data]
    # The original polygon -> raster conversion can change pixels in a legacy
    # mask (holes, contour boundaries) even if Napari was just opened and closed.
    # Compare vertices BEFORE rasterization and never touch an unchanged file.
    same_shapes = len(polygons) == len(initial_polygons) and all(
        np.array_equal(a, b) for a, b in zip(polygons, initial_polygons)
    )
    if existing is not None and same_shapes:
        return None, {"review_action": "unchanged_existing", "n_polygons": len(polygons), **baseline_review_meta}
    lesion = polygons_to_mask(polygons, shape)
    if existing is not None and np.array_equal(lesion, existing):
        return None, {"review_action": "unchanged_existing", "n_polygons": len(polygons), **baseline_review_meta}
    save_empty = bool(rcfg.get("save_empty_mask", False))
    if not lesion.any() and not save_empty:
        if existing is not None:
            print("[WARN] Empty polygons: existing lesion mask is preserved; nothing saved")
            return None, {"review_action": "kept_existing_empty_shapes", "n_polygons": 0, **baseline_review_meta}
        return None, {"review_action": "empty_not_saved", "n_polygons": 0, **baseline_review_meta}
    return lesion, {"review_action": "saved_manual_polygon", "n_polygons": len(polygons), **baseline_review_meta}


def save_lesion_review(
    row: pd.Series,
    lesion: np.ndarray | None,
    review_meta: dict[str, Any],
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Persist only a new or changed manual lesion mask with its review metadata.

    Parameters
    ----------
    row : pandas.Series
        Metadata of the reviewed recording.
    lesion : numpy.ndarray or None
        Current-stem full-resolution ``(y, x)`` mask or no new mask.
    review_meta : dict
        Review status and provenance supplied by Napari.
    config : dict
        Manual GUI configuration.
    paths : dict
        Output paths for the final mask and review logs.

    Returns
    -------
    dict
        Save/review status and, when a mask was written, its output path and
        full-resolution lesion area in pixels.

    Notes
    -----
    Existing masks and JSON are deliberately preserved when the review did not
    change a polygon. The GUI does not modify hemorrhage masks.
    """
    stem = str(row["stem"])
    if review_meta.get("review_action") in {"unchanged_existing", "kept_existing_empty_shapes"}:
        # Do not touch lesion_final.npy, metadata JSON, or old review CSV.
        return {"stem": stem, "status": "unchanged_existing", **review_meta}
    mpaths = damage.damage_mask_paths(paths, stem)
    record = {
        "group_id": _safe_str(row.get("group_id")),
        "mouse_id": _safe_str(row.get("mouse_id")),
        "stem": stem,
        "exp_day": _safe_str(row.get("exp_day")),
        "baseline_stem": _safe_str(row.get("baseline_stem")),
        "status": "not_saved" if lesion is None else "reviewed_manual",
        "lesion_final_path": str(mpaths["lesion_final"]),
        "lesion_area_px": int(np.asarray(lesion, bool).sum()) if lesion is not None else np.nan,
        **review_meta,
        "updated_at": _now_iso(),
    }
    if lesion is not None:
        mpaths["lesion_final"].parent.mkdir(parents=True, exist_ok=True)
        np.save(mpaths["lesion_final"], np.asarray(lesion, dtype=np.uint8))
        review_json = mpaths["lesion_final"].with_name("manual_gui_review_meta.json")
        with open(review_json, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        record["review_meta_path"] = str(review_json)
    _upsert_csv(paths["review_log_csv"], record, keys=["group_id", "stem"])
    return record


# =============================================================================
# Index / controller
# =============================================================================


def build_gui_index(
    config_path: str | Path,
    *,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    include_bad: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Select recording stems for manual lesion review.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Main project YAML file.
    filters : dict or None, optional
        Additional metadata filters.
    groups, stems, mice : sequence of str or None, optional
        Group, stem and mouse selection restrictions.
    include_bad : bool or None, optional
        Whether to include Step2 bad records.

    Returns
    -------
    tuple of (pandas.DataFrame, dict, dict)
        Selected GUI index, effective configuration and output paths.

    Notes
    -----
    Delegates to ``damage.build_damage_index(..., for_manual_gui=True)``;
    therefore the hard-coded manual-GUI day/protocol selection overrides the
    regular baseline and post-stroke YAML selectors.
    """
    index, config, paths = damage.build_damage_index(
        config_path,
        filters=filters,
        groups=groups,
        stems=stems,
        include_bad=include_bad,
        for_manual_gui=True,
        write_index=False,
    )
    config = ensure_manual_gui_defaults(config)
    paths = {**paths, **build_gui_paths(config, paths)}
    if mice and not index.empty:
        wanted = {str(v) for v in mice}
        if "mouse_id" in index.columns:
            index = index[index["mouse_id"].astype(str).isin(wanted)].copy()
        else:
            index = index[index["group_id"].astype(str).isin(wanted)].copy()
    if not index.empty:
        helper_exists = []
        for stem in index["stem"].astype(str):
            helper_exists.append((helper_dir(paths, stem) / "display_meta.json").exists() if stem else False)
        index["helpers_exist"] = helper_exists
        index.to_csv(paths["gui_index_csv"], index=False)
    return index.reset_index(drop=True), config, paths


def _find_metadata_row(metadata: pd.DataFrame, stem: str) -> pd.Series:
    """Find the latest metadata-registration row matching a stem."""
    sub = metadata[metadata["stem"].astype(str).eq(str(stem))]
    if sub.empty:
        raise KeyError(f"metadata_registration row not found for stem={stem}")
    return sub.iloc[-1]


def run_manual_gui(
    config_path: str | Path,
    *,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    include_bad: bool | None = None,
    prepare_only: bool = False,
    review_only: bool = False,
    only_unreviewed: bool = False,
    force_prepare: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Prepare and optionally review manual lesion masks for selected recordings.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Project YAML configuration.
    filters : dict or None, optional
        Additional selection criteria.
    groups, stems, mice : sequence of str or None, optional
        Limit processing to the specified metadata identifiers.
    include_bad : bool or None, optional
        Include Step2 flagged records when enabled.
    prepare_only, review_only : bool, optional
        Restrict execution to caching or to interactive review.
    only_unreviewed : bool, optional
        Skip stems with existing reviewed lesion masks.
    force_prepare : bool, optional
        Replace existing cached display helpers.
    dry_run : bool, optional
        Report the selection without executing GUI work.

    Returns
    -------
    pandas.DataFrame
        Per-stem result table including review or preparation status.

    Notes
    -----
    The GUI may operate in current-only mode if the baseline is absent; it does
    not perform automatic lesion segmentation.
    """
    config_path = Path(config_path)
    index, config, paths = build_gui_index(
        config_path,
        filters=filters,
        groups=groups,
        stems=stems,
        mice=mice,
        include_bad=include_bad,
    )
    print("Damage GUI index:", paths["gui_index_csv"])
    if index.empty:
        print("No post-PT stems selected.")
        return index

    ready = index[index["index_status"].astype(str).eq("ready")].copy()
    if only_unreviewed or bool(gui_cfg(config)["review"].get("skip_reviewed", False)):
        ready = ready[~ready["lesion_final_exists"].astype(bool)].copy()

    cols = [c for c in ["group_id", "mouse_id", "stem", "baseline_stem", "baseline_status", "exp_day", "state", "protocol", "lesion_final_exists", "helpers_exist"] if c in ready.columns]
    print(ready[cols].to_string(index=False))
    if dry_run:
        return ready

    metadata = pd.read_csv(paths["metadata_registration_csv"])
    manual_overrides = damage.load_manual_overrides(config, paths)
    _, _full_hcfg = _homotopic_cfg_parts(config)
    mean_enabled = bool((gui_cfg(config)["raw_layers"].get("mean_images", {}) or {}).get("enabled", True))
    hbt_cfg = gui_cfg(config).get("hbt_movie", {})
    raw_movie_cfg = gui_cfg(config).get("raw_movie", {})
    need_movie = (bool(hbt_cfg.get("enabled", True)) or bool(raw_movie_cfg.get("enabled", False))) and not prepare_only
    need_step4 = ((not review_only) and (
        mean_enabled
        or bool(gui_cfg(config)["run_sd"].get("enabled", True))
    )) or need_movie
    step4_stems = set(ready["stem"].dropna().astype(str).tolist())
    if "baseline_stem" in ready.columns:
        step4_stems.update(v for v in ready["baseline_stem"].dropna().astype(str).tolist() if v)
    step4_map = processed_index_map(config_path, sorted(step4_stems)) if need_step4 else {}
    continue_on_error = bool(gui_cfg(config)["batch"].get("continue_on_error", True))
    records: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(ready.iterrows(), start=1):
        stem = str(row["stem"])
        print(f"\n[{i}/{len(ready)}] {stem}")
        try:
            baseline_stem = _safe_str(row.get("baseline_stem")).strip()
            try:
                baseline_row = _find_metadata_row(metadata, baseline_stem) if baseline_stem else None
            except KeyError:
                baseline_row = None  # current-only mode; do not discard this stem
            try:
                ref_row = _find_metadata_row(metadata, str(row["ref_stem"]))
            except KeyError:
                ref_row = None  # current-only mode if registration reference is missing
            if not review_only:
                pmeta = prepare_helpers_for_row(
                    row,
                    baseline_row,
                    ref_row,
                    config,
                    paths,
                    manual_overrides,
                    step4_map.get(stem),
                    step4_map.get(str(row["baseline_stem"])),
                    force=force_prepare,
                )
                print(
                    "prepare:", pmeta.get("status"),
                    "RUN-SD:", (pmeta.get("run_sd") or {}).get("status"),
                    "FC-full:", (pmeta.get("homotopic_full_preview") or {}).get("status"),
                    "FC-Step7:", (pmeta.get("homotopic_authoritative") or {}).get("status"),
                    "baseline:", pmeta.get("baseline_status", "unknown"),
                )
            if prepare_only:
                records.append({"stem": stem, "status": "prepared", "updated_at": _now_iso()})
                continue

            try:
                raw_movie_path = _raw_zarr_path_from_step4_row(step4_map.get(stem), config)
            except FileNotFoundError:
                raw_movie_path = None  # optional on-demand layer
            lesion, rmeta = review_lesion_napari(
                row, config, paths,
                processed_path=_processed_path_from_row(step4_map.get(stem), config),
                raw_zarr_path=raw_movie_path,
            )
            record = save_lesion_review(row, lesion, rmeta, config, paths)
            print("review:", record["status"], "area_px=", record.get("lesion_area_px"))
            records.append(record)
        except Exception as exc:
            err = {
                "group_id": _safe_str(row.get("group_id")),
                "stem": stem,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "updated_at": _now_iso(),
            }
            _upsert_csv(paths["error_log_csv"], err, keys=["group_id", "stem"])
            records.append({**err, "status": "error"})
            print("[ERROR]", exc)
            if not continue_on_error:
                raise
    return pd.DataFrame(records)


# =============================================================================
# CLI
# =============================================================================


def parse_filters(items: Sequence[str] | None) -> dict[str, Any] | None:
    """Parse repeated command-line metadata filters."""
    if not items:
        return None
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Filter must be KEY=VALUE or KEY=a,b, got {item!r}")
        key, value = item.split("=", 1)
        vals = [v.strip() for v in value.split(",") if v.strip()]
        out[key.strip()] = vals if len(vals) > 1 else (vals[0] if vals else "")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser for manual lesion review."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--stem", action="append", default=None, help="Test/review a specific stem; repeatable")
    p.add_argument("--mouse", action="append", default=None, help="Select mouse_id/group_id; repeatable")
    p.add_argument("--group", action="append", default=None, help="Select group_id; repeatable")
    p.add_argument("--filter", action="append", default=None, help="Extra exact metadata filter KEY=VALUE or KEY=a,b")
    p.add_argument("--batch", action="store_true", help="Explicitly select all post-PT rows matching config")
    p.add_argument("--include-bad", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Prepare all helper maps but do not open Napari")
    p.add_argument("--review-only", action="store_true", help="Open GUI using already prepared helpers")
    p.add_argument("--only-unreviewed", action="store_true", help="Skip stems with lesion_final.npy")
    p.add_argument("--force-prepare", action="store_true", help="Recompute helper maps even if display_meta.json exists")
    p.add_argument("--dry-run", action="store_true", help="Build/print selection only")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Validate CLI options and run manual lesion-review preparation or review."""
    args = build_arg_parser().parse_args(argv)
    if args.prepare_only and args.review_only:
        raise ValueError("Use only one of --prepare-only or --review-only")
    if not args.batch and not args.stem and not args.mouse and not args.group:
        raise ValueError("Select --stem, --mouse/--group, or explicitly use --batch")
    run_manual_gui(
        args.config,
        filters=parse_filters(args.filter),
        groups=args.group,
        stems=args.stem,
        mice=args.mouse,
        include_bad=True if args.include_bad else None,
        prepare_only=bool(args.prepare_only),
        review_only=bool(args.review_only),
        only_unreviewed=bool(args.only_unreviewed),
        force_prepare=bool(args.force_prepare),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
