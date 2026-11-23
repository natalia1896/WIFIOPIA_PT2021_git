"""Step 4 raw zarr + signal processing for WIFIOPIA/OISI, v1_13 geometry + raw motion + GSR/preprocessing merge.

This module is designed to be used from Step4_SignalProcessing.ipynb.
It follows the current pipeline identity model: stem / group_id / ref_stem.
It intentionally uses config['step4'] only.  Rename the old config block step5 -> step4
before using this module.

Recommended workflow
--------------------
1) One-stem notebook test: raw -> raw_zarr -> processed_zarr with static previews.
2) Batch phase A: build raw_zarr for all selected stems.
3) Batch phase B: process all selected raw_zarr into processed_signals.
4) Optional napari preview only for a manually selected stem, never inside loops.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import gc
import hashlib
import json
import math
import re
import shutil
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Sequence

import cv2
import dask.array as da
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed as dask_delayed
from scipy.ndimage import gaussian_filter, shift as ndi_shift
from tqdm.auto import tqdm

try:
    from numcodecs import Blosc
except Exception:  # pragma: no cover
    Blosc = None

try:
    import zarr
except Exception:  # pragma: no cover
    zarr = None

try:
    from skimage.registration import phase_cross_correlation
except Exception:  # pragma: no cover
    phase_cross_correlation = None

from common import (
    build_standard_paths,
    create_or_load_project,
    get_path_resolving_config,
    list_raw_images,
    load_config,
    resolve_flexible_path,
    rotate_bound_image,
    canonical_mask_paths,
)


# ============================================================
# Generic helpers
# ============================================================


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested configuration value, returning the supplied default when a key is absent."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def require_step4(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Step4 configuration block or raise if the block is missing."""
    if "step4" not in config:
        raise KeyError("config['step4'] is required. Rename old config block 'step5' to 'step4'.")
    return config["step4"] or {}


def bool_series(s: pd.Series) -> pd.Series:
    """Convert common textual representations of true values to a Boolean series."""
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def as_list_or_none(value: Any) -> list[str] | None:
    """Represent a scalar or collection as a list of strings, preserving missing values."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def normalise_chunks(chunks: Sequence[int] | None, shape: Sequence[int]) -> tuple[int, ...]:
    """Clamp configured chunk lengths to array dimensions and expand -1 to the full axis."""
    if chunks is None:
        return tuple(int(s) for s in shape)
    out: list[int] = []
    for c, s in zip(chunks, shape):
        c = int(c)
        s = int(s)
        if c == -1:
            out.append(s)
        elif c <= 0:
            raise ValueError(f"Invalid chunk size {c}; use positive integer or -1")
        else:
            out.append(min(c, s))
    return tuple(out)


def make_compressor(config: dict[str, Any], block: str) -> Any | None:
    """Return a numcodecs compressor for zarr v2 stores.

    Step4 writes zarr stores as zarr format 2 by default, because the existing
    WIFIOPIA environment can have zarr >= 3 installed while xarray/numcodecs
    compression is still most stable with v2 metadata.
    """
    if Blosc is None:
        return None
    if block == "raw":
        name = cfg_get(config, "raw", "compressor_name", default="zstd")
        level = int(cfg_get(config, "raw", "compressor_level", default=3))
    else:
        name = cfg_get(config, "signal_processing", "output", "compressor_name", default="zstd")
        level = int(cfg_get(config, "signal_processing", "output", "compressor_level", default=3))
    if not name:
        return None
    cname = "zstd" if str(name).lower() in {"zstd", "zstandard"} else str(name)
    return Blosc(cname=cname, clevel=level, shuffle=Blosc.BITSHUFFLE)


def _is_empty_path_value(value: Any) -> bool:
    """Check whether an optional path is missing or represented by an empty token."""
    return value is None or not str(value).strip() or str(value).strip().lower() in {"nan", "none"}

def _remap_windows_like_path_no_exist(value: Any, path_cfg: dict[str, Any]) -> Path | None:
    """Apply configured root/drive remapping without requiring the target to exist.

    This is intentionally stricter than resolve_flexible_path for Step4 output
    stores.  raw_zarr_path and processed_zarr_path often point to .zarr folders
    that have not been created yet, so existence-based remapping can leave stale
    paths from an old manifest, e.g. H:\\... instead of F:\\...
    """
    if _is_empty_path_value(value):
        return None

    s = str(value).strip()

    from_root = path_cfg.get("from_root")
    to_root = path_cfg.get("to_root")
    if from_root and to_root:
        s_norm = s.replace("/", "\\")
        from_norm = str(from_root).rstrip("\\/").replace("/", "\\")
        to_norm = str(to_root).rstrip("\\/").replace("/", "\\")
        if s_norm.lower().startswith(from_norm.lower()):
            tail = s_norm[len(from_norm):].lstrip("\\/")
            # Build from a single string so Windows drive syntax survives both
            # Windows and non-Windows test environments.
            return Path(str(PureWindowsPath(to_norm) / PureWindowsPath(tail)))

    from_drive = path_cfg.get("from_drive")
    to_drive = path_cfg.get("to_drive")
    if from_drive and to_drive:
        wp = PureWindowsPath(s)
        from_drive_norm = str(from_drive).rstrip("\\/").lower()
        to_drive_norm = str(to_drive).rstrip("\\/")
        if wp.drive and wp.drive.lower() == from_drive_norm:
            tail_parts = wp.parts[1:]
            return Path(str(PureWindowsPath(to_drive_norm + "\\", *tail_parts)))

    return Path(s)

def resolve_optional_path(value: Any, path_cfg: dict[str, Any], *, must_exist: bool = False) -> Path | None:
    """Resolve real input paths using the shared, existence-aware resolver."""
    if _is_empty_path_value(value):
        if must_exist:
            raise FileNotFoundError(f"Empty path: {value!r}")
        return None
    return resolve_flexible_path(
        value,
        from_drive=path_cfg.get("from_drive"),
        to_drive=path_cfg.get("to_drive"),
        from_root=path_cfg.get("from_root"),
        to_root=path_cfg.get("to_root"),
        must_exist=must_exist,
    )

def resolve_output_path(value: Any, path_cfg: dict[str, Any], fallback: Path) -> Path:
    """Resolve Step4 output zarr path with non-existing-safe remapping.

    If manifest_recordings.csv contains an old output path, remap it first.
    If the manifest value is empty, use the current project-derived fallback.
    """
    p = _remap_windows_like_path_no_exist(value, path_cfg)
    if p is None:
        return Path(fallback)
    return p


def safe_scalar_attrs(row: pd.Series) -> dict[str, Any]:
    """Extract non-missing scalar row metadata suitable for dataset attributes."""
    out: dict[str, Any] = {}
    for k, v in row.to_dict().items():
        if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or v is None:
            if pd.isna(v) if not isinstance(v, str) else False:
                continue
            out[str(k)] = v.item() if isinstance(v, (np.integer, np.floating)) else v
    return out


# ============================================================
# Step4 index
# ============================================================


def build_step4_index(
    config_path: str | Path,
    *,
    include_bad: bool | None = None,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Build the Step4 recording index from Step1 and Step2 metadata.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        YAML configuration file used for this analysis.
    include_bad : bool, optional
        Override `step4.review.include_bad` when selecting records.
    filters : dict, optional
        Restrict records by metadata fields.
    groups : sequence of str, optional
        Select `group_id` values.
    stems : sequence of str, optional
        Select recording `stem` identifiers.
    
    Returns
    -------
    index : pandas.DataFrame
        Recording-level table with resolved paths and file-existence indicators.
    config : dict
        Parsed project YAML settings.
    paths : dict
        Standard analysis and output directories.
    
    Notes
    -----
    Step2 `metadata_registration.csv` is authoritative for registration, masks,
    and QC. Step1 `manifest_recordings.csv` provides raw input directory paths.
    The index does not apply inter-recording spatial transforms to raw data."""
    config_path = Path(config_path)
    config = load_config(config_path)
    step4_cfg = require_step4(config)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)

    manifest_path = paths["manifest_path"]
    metadata_path = paths["metadata_registration_csv"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_recordings.csv not found: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_registration.csv not found: {metadata_path}")

    manifest = pd.read_csv(manifest_path)
    meta_reg = pd.read_csv(metadata_path)
    # Step2 metadata defines the analysis cohort; report manifest entries that
    # were not admitted (repeat selection, missing H5, manual exclusions).
    missing_from_step2 = manifest.loc[~manifest["stem"].astype(str).isin(meta_reg["stem"].astype(str)), "stem"].astype(str).tolist()
    print(f"[Step4 selection] Step1 manifest={len(manifest)}, Step2 metadata={len(meta_reg)}, missing from Step2={len(missing_from_step2)}")
    for stem in missing_from_step2:
        print(f"  [NOT IN STEP2] {stem} (see step2_missing_inputs.csv)")
    if "stem" not in manifest.columns or "stem" not in meta_reg.columns:
        raise ValueError("Both manifest_recordings.csv and metadata_registration.csv must contain 'stem'.")

    if "img_dir" not in manifest.columns:
        raise ValueError("manifest_recordings.csv must contain img_dir")
    if manifest["stem"].duplicated().any() or meta_reg["stem"].duplicated().any():
        raise ValueError("Duplicate stem in manifest_recordings.csv or metadata_registration.csv")
    # Manifest is authoritative for raw directories, never stale Step2 metadata.
    manifest_extra_cols = [c for c in manifest.columns if c == "stem" or c == "img_dir" or c not in meta_reg.columns]
    df = meta_reg.drop(columns=["img_dir"], errors="ignore").merge(
        manifest[manifest_extra_cols], on="stem", how="left", validate="one_to_one"
    )
    # Step2 already selects highest protocol_repeat. Never resurrect older repeats
    # if an old metadata file still contains them.
    if bool(cfg_get(config, "variant_keys", "protocol", "enabled", default=False)):
        if all(c in df.columns for c in ("mouse_id", "exp_day", "state", "protocol_base", "protocol_repeat")):
            group_cols = ["mouse_id", "exp_day", "state", "protocol_base"]
            repeat = pd.to_numeric(df["protocol_repeat"], errors="coerce").fillna(0)
            kept = repeat.eq(repeat.groupby([df[c] for c in group_cols]).transform("max"))
            for stem in df.loc[~kept, "stem"].astype(str):
                print(f"[SKIPPED] Step4 stem={stem}: superseded_by_higher_repeat")
            df = df.loc[kept].copy()
    # Always exclude technically invalid recordings, including when Step2 metadata is older.
    excluded_stems = set(config.get("excluded_stems") or [])
    if excluded_stems:
        skipped = df[df["stem"].astype(str).isin(excluded_stems)]
        for stem in skipped["stem"].astype(str):
            print(f"[EXCLUDED] Step4 stem={stem}: excluded_by_config")
        df = df[~df["stem"].astype(str).isin(excluded_stems)].copy()

    review_cfg = step4_cfg.get("review", {}) or {}
    if include_bad is None:
        include_bad = bool(review_cfg.get("include_bad", False))
    if not include_bad and "is_bad" in df.columns:
        df = df[~bool_series(df["is_bad"])].copy()

    debug_filters = filters
    if debug_filters is None:
        debug_filters = (step4_cfg.get("debug", {}) or {}).get("filters") or {}
    for key, allowed_raw in (debug_filters or {}).items():
        compare_key = "protocol_base" if key == "protocol" and bool(cfg_get(config, "variant_keys", "protocol", "enabled", default=False)) else key
        if compare_key not in df.columns:
            raise ValueError(f"Step4 filter column not found: {compare_key}")
        allowed = set(as_list_or_none(allowed_raw) or [])
        if allowed:
            if compare_key == "protocol_repeat":
                df = df[pd.to_numeric(df[compare_key], errors="coerce").isin({float(v) for v in allowed})].copy()
            else:
                df = df[df[compare_key].astype(str).isin(allowed)].copy()

    if groups:
        if "group_id" not in df.columns:
            raise ValueError("groups filter requires group_id column in metadata_registration.csv")
        df = df[df["group_id"].astype(str).isin({str(g) for g in groups})].copy()
    if stems:
        df = df[df["stem"].astype(str).isin({str(s) for s in stems})].copy()

    path_cfg = get_path_resolving_config(config, "step4")
    raw_root = paths["raw_zarr_root"]
    processed_root = paths["processed_signals_root"]

    img_dirs: list[str] = []
    raw_paths: list[str] = []
    raw_mc_paths: list[str] = []
    proc_paths: list[str] = []
    proc_unaligned_paths: list[str] = []
    proc_mc_paths: list[str] = []
    proc_selected_variants: list[str] = []
    raw_selected_paths: list[str] = []
    raw_exists: list[bool] = []
    raw_mc_exists: list[bool] = []
    proc_exists: list[bool] = []
    proc_unaligned_exists: list[bool] = []
    proc_mc_exists: list[bool] = []

    for _, row in df.iterrows():
        stem = str(row["stem"])
        try:
            img_dir = resolve_optional_path(row.get("img_dir", ""), path_cfg, must_exist=True)
            img_dirs.append(str(img_dir) if img_dir else "")
        except FileNotFoundError:
            img_dirs.append("")  # record-level missing input; reported in raw phase
            print(f"[WARN] Step4 missing raw folder for stem={stem}: {row.get('img_dir', '')}")

        raw_path = row.get("raw_zarr_path", "")
        raw_p = resolve_output_path(raw_path, path_cfg, raw_root / f"{stem}.zarr")
        raw_paths.append(str(raw_p))
        raw_exists.append(raw_p.exists())

        raw_mc_p = motion_corrected_raw_store_path(raw_p, config)
        raw_mc_paths.append(str(raw_mc_p))
        raw_mc_exists.append(raw_mc_p.exists())

        processed_path = row.get("processed_zarr_path", "")
        proc_p = resolve_output_path(processed_path, path_cfg, processed_root / f"{stem}_processed.zarr")
        proc_mc_p = motion_corrected_processed_store_path(proc_p, config)
        variant_cfg = raw_processing_variant(config)
        if variant_cfg == "motion_corrected":
            selected_raw_p = raw_mc_p
            selected_proc_p = proc_mc_p
            selected_variant = "motion_corrected"
        elif variant_cfg == "auto" and raw_mc_p.exists():
            selected_raw_p = raw_mc_p
            selected_proc_p = proc_mc_p
            selected_variant = "motion_corrected"
        else:
            selected_raw_p = raw_p
            selected_proc_p = proc_p
            selected_variant = "unaligned"

        proc_unaligned_paths.append(str(proc_p))
        proc_mc_paths.append(str(proc_mc_p))
        proc_paths.append(str(selected_proc_p))
        raw_selected_paths.append(str(selected_raw_p))
        proc_selected_variants.append(selected_variant)
        proc_exists.append(selected_proc_p.exists())
        proc_unaligned_exists.append(proc_p.exists())
        proc_mc_exists.append(proc_mc_p.exists())

    df["img_dir_resolved"] = img_dirs
    df["raw_zarr_path_resolved"] = raw_paths
    df["raw_zarr_unaligned_path_resolved"] = raw_paths
    df["raw_zarr_motion_corrected_path_resolved"] = raw_mc_paths
    df["raw_zarr_selected_for_processing_path_resolved"] = raw_selected_paths
    df["raw_zarr_selected_variant"] = proc_selected_variants
    df["processed_zarr_unaligned_path_resolved"] = proc_unaligned_paths
    df["processed_zarr_motion_corrected_path_resolved"] = proc_mc_paths
    df["processed_zarr_path_resolved"] = proc_paths
    df["processed_zarr_selected_variant"] = proc_selected_variants
    df["raw_zarr_exists"] = raw_exists
    df["raw_zarr_motion_corrected_exists"] = raw_mc_exists
    df["processed_zarr_exists"] = proc_exists
    df["processed_zarr_unaligned_exists"] = proc_unaligned_exists
    df["processed_zarr_motion_corrected_exists"] = proc_mc_exists
    print(f"[Step4 selection] selected for processing: {len(df)} stem(s)")
    return df.reset_index(drop=True), config, paths


# ============================================================
# Raw image discovery / splitting / resampling
# ============================================================


def split_paths_by_wavelength(
    sorted_paths: list[Path],
    wave_pattern: dict[str, Sequence[int]],
    read_end: int | str = "MAX",
) -> dict[str, list[Path]]:
    """Split acquisition-ordered frame paths into wavelength-specific sequences.
    
    Parameters
    ----------
    sorted_paths : list of pathlib.Path
        Source images in acquisition sort order.
    wave_pattern : dict
        Configured mapping from wavelength to (start_index, stride).
    read_end : int or str
        Maximum number of source frames or MAX for all frames.
    
    Returns
    -------
    dict of str to list of pathlib.Path
        Wavelength-indexed source paths in acquisition order.
    
    Notes
    -----
    The mapping in ``raw.wave_pattern`` contains a zero-based start index and
    an acquisition stride for each wavelength. Only the selected leading frames
    (``raw.read_end``) are considered."""
    n = len(sorted_paths) if str(read_end).upper() == "MAX" else int(read_end)
    subset = sorted_paths[:n]
    out: dict[str, list[Path]] = {}
    for wave, pattern in wave_pattern.items():
        start, step = int(pattern[0]), int(pattern[1])
        out[str(wave)] = subset[start::step]
    return out


def reference_indices(n_ref: int, ref_step: int, step: int, mode: str) -> np.ndarray:
    """Calculate indices that align one wavelength sampling interval to a reference interval.
    
    Parameters
    ----------
    n_ref : int
        Number of frames at the reference wavelength.
    ref_step : int
        Acquisition stride of the reference wavelength.
    step : int
        Acquisition stride of the source wavelength.
    mode : str
        Resampling mode: repeat, nearest, or mean_surround.
    
    Returns
    -------
    numpy.ndarray
        Source-channel indices or fractional positions, one per reference frame.
    
    Notes
    -----
    The output indexes are on the source channel's frame grid. The
    ``repeat`` and ``nearest`` modes return integer indices, whereas
    ``mean_surround`` returns fractional positions for bracketing."""
    idx_float = np.arange(n_ref, dtype=np.float64) * (ref_step / step)
    if mode == "repeat":
        return np.floor(idx_float).astype(np.int64)
    if mode == "nearest":
        return np.round(idx_float).astype(np.int64)
    if mode == "mean_surround":
        return idx_float
    raise ValueError(f"Unknown raw.resample_mode={mode!r}")


def resample_paths_to_reference(
    wave_to_paths: dict[str, list[Path]],
    wave_pattern: dict[str, Sequence[int]],
    reference_wave: str,
    mode: str = "repeat",
) -> dict[str, list[Path | tuple[Path, Path]]]:
    """Align wavelength frame paths to the configured reference-wavelength time grid.
    
    Parameters
    ----------
    wave_to_paths : dict
        Raw file paths for each wavelength.
    wave_pattern : dict
        Configured mapping from wavelength to (start_index, stride).
    reference_wave : str
        Reference wavelength identifier from ``raw.reference_wave``.
    mode : object
        See the corresponding Step4 call site and YAML settings.
    
    Returns
    -------
    dict
        Wavelength-indexed frame paths or bracketing path pairs at the reference time grid.
    
    Notes
    -----
    The reference wavelength comes from ``raw.reference_wave``. The
    configured ``raw.resample_mode`` can repeat or select nearest images, or
    request the mean of bracketing frames. Missing source frames cause errors
    rather than silent padding."""
    reference_wave = str(reference_wave)
    if reference_wave not in wave_to_paths:
        raise KeyError(f"reference_wave={reference_wave!r} not found in wave_pattern")
    ref_paths = wave_to_paths[reference_wave]
    n_ref = len(ref_paths)
    _, ref_step = [int(x) for x in wave_pattern[reference_wave]]
    out: dict[str, list[Path | tuple[Path, Path]]] = {reference_wave: list(ref_paths)}

    for wave, paths in wave_to_paths.items():
        wave = str(wave)
        if wave == reference_wave:
            continue
        if int(wave_pattern[wave][1]) == ref_step and len(paths) != n_ref:
            raise ValueError(f"Unequal wavelength frame counts: {reference_wave}={n_ref}, {wave}={len(paths)}; no automatic duplication/truncation")
        if not paths:
            raise RuntimeError(f"No images found for wavelength {wave}")
        _, step = [int(x) for x in wave_pattern[wave]]
        idx = reference_indices(n_ref, ref_step, step, mode)
        if mode in {"repeat", "nearest"}:
            idx_i = idx.astype(np.int64)
            if np.any((idx_i < 0) | (idx_i >= len(paths))):
                raise ValueError(f"Wavelength {wave}: {len(paths)} frame(s) cannot cover {n_ref} reference frames; refusing to duplicate an absent frame")
            out[wave] = [paths[i] for i in idx_i]
        else:
            idx_low = np.floor(idx).astype(np.int64)
            idx_high = np.ceil(idx).astype(np.int64)
            if np.any((idx_low < 0) | (idx_high >= len(paths))):
                raise ValueError(f"Wavelength {wave}: missing frame for mean_surround interpolation")
            out[wave] = [paths[i] if i == j else (paths[i], paths[j]) for i, j in zip(idx_low, idx_high)]
    return out


def inspect_raw_files_for_row(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    """Inspect a recording and build the wavelength-specific input frame lists."""
    raw_cfg = config.get("raw", {}) or {}
    if not str(row.get("img_dir_resolved", "")).strip():
        raise FileNotFoundError(f"No raw image directory for stem={row.get('stem', '')}")
    img_dir = Path(row["img_dir_resolved"])
    paths = list_raw_images(
        img_dir,
        suffixes=tuple(raw_cfg.get("suffixes", [".bmp"])),
        file_regex=raw_cfg.get("file_regex"),
        sort_key=raw_cfg.get("sort_key", "index"),
    )
    wave_pattern = {str(k): v for k, v in raw_cfg["wave_pattern"].items()}
    wave_to_paths0 = split_paths_by_wavelength(paths, wave_pattern, raw_cfg.get("read_end", "MAX"))
    reference_wave = str(raw_cfg.get("reference_wave", list(wave_pattern)[0]))
    wave_to_paths = resample_paths_to_reference(
        wave_to_paths0,
        wave_pattern=wave_pattern,
        reference_wave=reference_wave,
        mode=raw_cfg.get("resample_mode", "repeat"),
    )
    return {
        "all_paths": paths,
        "wave_to_paths_raw": wave_to_paths0,
        "wave_to_paths_resampled": wave_to_paths,
        "reference_wave": reference_wave,
        "n_source_files": len(paths),
        "n_time": len(wave_to_paths[reference_wave]),
    }


# ============================================================
# Image reading and preview
# ============================================================


def read_gray_frame(
    obj: Path | tuple[Path, Path],
    *,
    scale: float = 1.0,
    rotation: float = 0.0,
    read_dtype: str | np.dtype | None = None,
    mean_pair_dtype: str | np.dtype | None = None,
    source_frame_callback: Callable[[Path, np.ndarray], None] | None = None,
) -> np.ndarray:
    """Read an image path or frame selection as a two-dimensional grayscale array."""
    def read_one(path: Path) -> np.ndarray:
        """Read one frame or image pair for the current wavelength."""
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        if source_frame_callback is not None:
            source_frame_callback(path, img)  # Original uint8 BMP, before rotation/resize.
        if rotation != 0:
            # Exact technical rotation before downsampling (Step1–Step4 geometry).
            img = rotate_bound_image(img, float(rotation), interpolation=cv2.INTER_LINEAR)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA)
        if read_dtype is not None:
            img = img.astype(read_dtype, copy=False)
        return img

    if isinstance(obj, tuple):
        a = read_one(obj[0]).astype(np.float32)
        b = read_one(obj[1]).astype(np.float32)
        out = (a + b) / 2.0
        if mean_pair_dtype is not None:
            if np.issubdtype(np.dtype(mean_pair_dtype), np.integer):
                out = np.rint(out)
            out = out.astype(mean_pair_dtype, copy=False)
        return out
    return read_one(obj)


def raw_read_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Collect raw image reading and resampling options from the YAML configuration."""
    raw_cfg = config.get("raw", {}) or {}
    return {
        "scale": float(raw_cfg.get("scale", 1.0)),
        "rotation": float(raw_cfg.get("rotation", 0.0)),
        "read_dtype": raw_cfg.get("read_dtype"),
        "mean_pair_dtype": raw_cfg.get("mean_pair_dtype"),
    }


def preview_raw_frames_from_bmp(row: pd.Series, config: dict[str, Any], *, frame_index: int | None = None, save_path: str | Path | None = None) -> plt.Figure:
    """Plot reference frames for the configured wavelengths before creating raw Zarr."""
    info = inspect_raw_files_for_row(row, config)
    wave_to_paths = info["wave_to_paths_resampled"]
    frame_index = int(frame_index if frame_index is not None else cfg_get(config, "step4", "notebook", "preview_frame_index", default=10))
    kwargs = raw_read_kwargs(config)

    waves = list(wave_to_paths.keys())
    fig, axes = plt.subplots(1, len(waves), figsize=(4 * len(waves), 4))
    if len(waves) == 1:
        axes = [axes]
    for ax, wave in zip(axes, waves):
        paths = wave_to_paths[wave]
        idx = min(max(frame_index, 0), len(paths) - 1)
        img = read_gray_frame(paths[idx], **kwargs)
        vmin, vmax = np.nanpercentile(img, [2, 98])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{wave} nm | frame {idx}")
        ax.axis("off")
    fig.suptitle(f"Raw BMP preview: {row['stem']}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig



def raw_valid_from_min_max(
    raw_min: np.ndarray,
    raw_max: np.ndarray,
    config: dict[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Construct the all-frame valid-pixel mask from source-image extrema.
    
    Parameters
    ----------
    raw_min : numpy.ndarray
        Per-pixel minimum across the original BMP exposures (y, x).
    raw_max : numpy.ndarray
        Per-pixel maximum across the original BMP exposures (y, x).
    config : dict
        Configuration containing `signal_processing.raw_valid.min_intensity`
        and `max_intensity`, plus the raw rotation settings.
    image_shape : tuple of int
        Spatial shape (y, x) of the Step4 raw recording.
    
    Returns
    -------
    valid : numpy.ndarray
        Boolean (y, x) mask; True indicates that no source exposure violated
        the configured limits at the mapped spatial location.
    
    Notes
    -----
    Pixels violating either intensity limit in any source exposure are marked
    invalid. The invalid mask is rotated with nearest-neighbor interpolation
    and conservatively mapped to raw Zarr geometry; spatial resampling uses
    an area-based criterion for invalid pixels."""
    lo = int(cfg_get(config, "signal_processing", "raw_valid", "min_intensity", default=1))
    hi = int(cfg_get(config, "signal_processing", "raw_valid", "max_intensity", default=250))
    if not 0 <= lo <= hi <= 255:
        raise ValueError("raw_valid requires 0 <= min_intensity <= max_intensity <= 255")
    if raw_min.shape != raw_max.shape:
        raise ValueError("raw_min/raw_max source shapes do not match")
    invalid = (raw_min < lo) | (raw_max > hi)
    rotation = float(cfg_get(config, "raw", "rotation", default=0))
    if rotation:
        invalid = rotate_bound_image(invalid.astype(np.uint8), rotation,
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
    if invalid.shape != image_shape:
        invalid = cv2.resize(invalid.astype(np.float32),
                             (image_shape[1], image_shape[0]),
                             interpolation=cv2.INTER_AREA) > 0
    return ~invalid


def load_raw_valid(
    path: str | Path,
    image_shape: tuple[int, int],
    *,
    config: dict[str, Any],
) -> xr.DataArray:
    """Rebuild the spatial raw-validity mask from saved source-image extrema.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Existing raw Zarr store with `source_intensity_stats/raw_min` and
        `source_intensity_stats/raw_max`.
    image_shape : tuple of int
        Requested Step4 spatial shape (y, x).
    config : dict
        Configuration containing `signal_processing.raw_valid` thresholds.
    
    Returns
    -------
    valid : xarray.DataArray
        Boolean spatial mask with dimensions (y, x).
    
    Notes
    -----
    The stored source extrema allow the validity thresholds to be applied
    without reopening the original BMP stack. The mask is not written as a
    source-space array inside the raw Zarr store."""
    root = zarr_open_group_v2_compat(Path(path), mode="r")
    if "source_intensity_stats" not in root:
        raise ValueError(f"source_intensity_stats not found in {path}; rebuild raw Zarr")
    group = root["source_intensity_stats"]
    raw_min = np.asarray(group["raw_min"][:])
    raw_max = np.asarray(group["raw_max"][:])
    valid = raw_valid_from_min_max(raw_min, raw_max, config, image_shape)
    return xr.DataArray(valid, dims=("y", "x"),
                        coords={"y": np.arange(image_shape[0]), "x": np.arange(image_shape[1])},
                        name="raw_valid")


# ============================================================
# Raw zarr writing: batch/chunk loop with progress bars
# ============================================================


def zarr_create_dataset_compat(group: Any, name: str, **kwargs: Any) -> Any:
    """Create a zarr array with compatibility for zarr 2 and zarr 3 APIs.

    In zarr >= 3, ``Group.create_dataset`` is deprecated but still works.
    We intentionally write zarr-format-2 stores in Step4, so the old
    ``compressor=numcodecs.Blosc(...)`` encoding remains valid.
    """
    if hasattr(group, "create_dataset"):
        return group.create_dataset(name, **kwargs)
    if hasattr(group, "create_array"):
        return group.create_array(name, **kwargs)
    raise RuntimeError("Unsupported zarr group object: no create_dataset/create_array")


def zarr_open_group_v2_compat(store: Path, mode: str = "w") -> Any:
    """Open a zarr group, preferring zarr format 2 for numcodecs compatibility."""
    if zarr is None:
        raise ImportError("zarr is required")
    try:
        return zarr.open_group(str(store), mode=mode, zarr_format=2)
    except TypeError:
        # Older zarr uses zarr_version instead of zarr_format.
        try:
            return zarr.open_group(str(store), mode=mode, zarr_version=2)
        except TypeError:
            return zarr.open_group(str(store), mode=mode)


def zarr_group_has_array(store: str | Path, key: str) -> bool:
    """Return True if a zarr group contains a requested array.

    This is mainly used to detect half-created/corrupted stores left after
    an interrupted write. Without this check, overwrite=False would silently
    reuse an empty .zarr folder and xarray would later show a dataset with no
    variables.
    """
    if zarr is None:
        return False
    try:
        root = zarr_open_group_v2_compat(Path(store), mode="r")
        if hasattr(root, "array_keys"):
            return str(key) in {str(k) for k in root.array_keys()}
        return str(key) in root
    except Exception:
        return False


def numeric_wavelength_values(wavelengths: Sequence[str]) -> np.ndarray:
    """Store wavelength coordinates as int32 when possible.

    Numeric coordinates are more robust across zarr/xarray versions than fixed
    width unicode arrays. If a non-numeric channel label is ever used, fall back
    to short unicode strings.
    """
    try:
        return np.asarray([int(str(w)) for w in wavelengths], dtype="int32")
    except Exception:
        return np.asarray([str(w) for w in wavelengths], dtype="U16")


def write_xarray_compatible_raw_zarr(
    store: Path,
    *,
    data_shape: tuple[int, int, int, int],
    chunks: tuple[int, int, int, int],
    dtype: np.dtype,
    wavelengths: Sequence[str],
    time_values: np.ndarray,
    attrs: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    """Write coordinate metadata in the raw store for xarray-compatible reading."""
    if zarr is None:
        raise ImportError("zarr is required for batch raw writing")
    compressor = make_compressor(config, "raw")
    # Force zarr format 2 so numcodecs compressors work even when the
    # installed Python package is zarr >= 3.
    root = zarr_open_group_v2_compat(store, mode="w")
    raw_key = cfg_get(config, "raw", "zarr_key", default="raw_record")

    create_kwargs = dict(shape=data_shape, chunks=chunks, dtype=dtype)
    if compressor is not None:
        create_kwargs["compressor"] = compressor
    arr = zarr_create_dataset_compat(root, raw_key, **create_kwargs)
    arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "wavelength", "y", "x"]

    time_coord = time_values.astype("float64")
    t = zarr_create_dataset_compat(root, "time", data=time_coord, shape=time_coord.shape, dtype="float64")
    t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    wave_coord = numeric_wavelength_values(wavelengths)
    w = zarr_create_dataset_compat(root, "wavelength", data=wave_coord, shape=wave_coord.shape, dtype=wave_coord.dtype)
    w.attrs["_ARRAY_DIMENSIONS"] = ["wavelength"]
    y_coord = np.arange(data_shape[2], dtype="int32")
    y = zarr_create_dataset_compat(root, "y", data=y_coord, shape=y_coord.shape, dtype="int32")
    y.attrs["_ARRAY_DIMENSIONS"] = ["y"]
    x_coord = np.arange(data_shape[3], dtype="int32")
    x = zarr_create_dataset_compat(root, "x", data=x_coord, shape=x_coord.shape, dtype="int32")
    x.attrs["_ARRAY_DIMENSIONS"] = ["x"]

    root.attrs.update(attrs)
    return arr


# ============================================================
# Optional intra-recording motion correction for raw zarr
# ============================================================


def raw_motion_correction_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the optional Step4 raw motion-correction config block.

    This block is intentionally independent from Step2/Step3 registration:
    it estimates frame-to-frame translation inside one recording while raw zarr
    is being built. The uncorrected raw zarr is always kept at the normal
    ``raw_zarr_path_resolved``; the corrected copy is written to a sibling
    zarr store only when this block is enabled.
    """
    cfg = dict(config.get("raw_motion_correction", {}) or {})
    cfg.setdefault("enabled", False)
    cfg.setdefault("estimate_wavelength", None)
    cfg.setdefault("output_suffix", "motion_corrected")
    cfg.setdefault("store_dtype", "float32")
    cfg.setdefault("reference_mode", "mean_first_n")
    cfg.setdefault("reference_frame_index", 0)
    cfg.setdefault("reference_n_frames", 20)
    cfg.setdefault("preprocess", "normalize")
    cfg.setdefault("highpass_sigma", 0.0)
    cfg.setdefault("ignore_border_px", 0)
    cfg.setdefault("upsample_factor", 10)
    cfg.setdefault("normalization", "phase")
    cfg.setdefault("max_shift_px", 30)
    cfg.setdefault("fail_policy", "use_previous")
    cfg.setdefault("interpolation_order", 1)
    cfg.setdefault("shift_mode", "nearest")
    cfg.setdefault("save_transforms_csv", True)
    return cfg


def raw_motion_correction_enabled(config: dict[str, Any]) -> bool:
    """Report whether raw motion correction is enabled in the configuration."""
    return bool(raw_motion_correction_config(config).get("enabled", False))


def motion_corrected_raw_store_path(store: str | Path, config: dict[str, Any]) -> Path:
    """Derive the motion-corrected raw Zarr path from the base raw store path."""
    store = Path(store)
    mc_cfg = raw_motion_correction_config(config)
    suffix = str(mc_cfg.get("output_suffix", "motion_corrected")).strip() or "motion_corrected"
    if store.suffix:
        return store.with_name(f"{store.stem}_{suffix}{store.suffix}")
    return store.with_name(f"{store.name}_{suffix}")


def motion_transforms_csv_path(row: pd.Series, config: dict[str, Any]) -> Path:
    """Build the CSV path used for frame-wise motion correction transforms."""
    qc_dir = get_qc_dir(row, config)
    qc_dir.mkdir(parents=True, exist_ok=True)
    suffix = str(raw_motion_correction_config(config).get("output_suffix", "motion_corrected")).strip() or "motion_corrected"
    return qc_dir / f"{row['stem']}_{suffix}_transforms.csv"


def raw_processing_variant(config: dict[str, Any]) -> str:
    """Return which raw zarr variant Step4 processing should use.

    Supported config:

        signal_processing:
          input:
            raw_variant: unaligned          # default, old behavior
            # raw_variant: motion_corrected # process <stem>_motion_corrected.zarr
            # raw_variant: auto             # use motion-corrected only if it exists

    The raw-build phase always writes the normal unaligned store first.  This
    selector is used only by the signal-processing phase and QC helpers.
    """
    input_cfg = cfg_get(config, "signal_processing", "input", default={}) or {}
    value = input_cfg.get("raw_variant", input_cfg.get("raw_zarr_variant", "unaligned"))
    value = str(value or "unaligned").strip().lower()
    aliases = {
        "raw": "unaligned",
        "original": "unaligned",
        "uncorrected": "unaligned",
        "not_corrected": "unaligned",
        "motion": "motion_corrected",
        "corrected": "motion_corrected",
        "motion_corrected": "motion_corrected",
        "mc": "motion_corrected",
    }
    value = aliases.get(value, value)
    if value not in {"unaligned", "motion_corrected", "auto"}:
        raise ValueError(
            "signal_processing.input.raw_variant must be one of: "
            "'unaligned', 'motion_corrected', or 'auto'"
        )
    return value


def motion_corrected_processed_store_path(store: str | Path, config: dict[str, Any]) -> Path:
    """Return the processed-zarr sibling path for motion-corrected raw input."""
    store = Path(store)
    input_cfg = cfg_get(config, "signal_processing", "input", default={}) or {}
    suffix = str(input_cfg.get("motion_corrected_processed_suffix", "motion_corrected")).strip()
    if not suffix:
        return store
    if store.suffix:
        return store.with_name(f"{store.stem}_{suffix}{store.suffix}")
    return store.with_name(f"{store.name}_{suffix}")


def _row_path(row: pd.Series, column: str, default: Path | None = None) -> Path:
    """Return a path from a row column, using the optional fallback when needed."""
    value = row.get(column, "")
    if isinstance(value, str) and value.strip():
        return Path(value)
    if value is not None and not (isinstance(value, float) and pd.isna(value)) and str(value).strip():
        return Path(str(value))
    if default is None:
        raise KeyError(f"Row does not contain a usable path column: {column}")
    return Path(default)


def selected_raw_variant_for_row(row: pd.Series, config: dict[str, Any]) -> str:
    """Return the effective raw variant for this row.

    For raw_variant='auto', this returns 'motion_corrected' only when the
    corrected store already exists; otherwise it falls back to 'unaligned'.
    """
    variant = raw_processing_variant(config)
    if variant == "auto":
        raw_unaligned = _row_path(row, "raw_zarr_path_resolved")
        raw_mc = _row_path(
            row,
            "raw_zarr_motion_corrected_path_resolved",
            default=motion_corrected_raw_store_path(raw_unaligned, config),
        )
        return "motion_corrected" if raw_mc.exists() else "unaligned"
    return variant


def select_raw_zarr_path_for_processing(row: pd.Series, config: dict[str, Any]) -> Path:
    """Return the raw zarr path used by the signal-processing phase."""
    raw_unaligned = _row_path(row, "raw_zarr_path_resolved")
    variant = selected_raw_variant_for_row(row, config)
    if variant == "motion_corrected":
        return _row_path(
            row,
            "raw_zarr_motion_corrected_path_resolved",
            default=motion_corrected_raw_store_path(raw_unaligned, config),
        )
    return raw_unaligned


def select_processed_zarr_path_for_processing(row: pd.Series, config: dict[str, Any]) -> Path:
    """Return the processed zarr path matching the selected raw input variant."""
    proc_unaligned = _row_path(row, "processed_zarr_unaligned_path_resolved", default=_row_path(row, "processed_zarr_path_resolved"))
    variant = selected_raw_variant_for_row(row, config)
    if variant == "motion_corrected":
        return _row_path(
            row,
            "processed_zarr_motion_corrected_path_resolved",
            default=motion_corrected_processed_store_path(proc_unaligned, config),
        )
    return proc_unaligned


def _motion_preprocess_image(img: np.ndarray, mc_cfg: dict[str, Any]) -> np.ndarray:
    """Build a stable 2D feature image for phase-cross-correlation."""
    arr = np.asarray(img, dtype=np.float32)
    b = int(mc_cfg.get("ignore_border_px", 0) or 0)
    if b > 0 and arr.shape[0] > 2 * b and arr.shape[1] > 2 * b:
        arr = arr[b:-b, b:-b]

    mode = str(mc_cfg.get("preprocess", "normalize")).strip().lower()
    if mode in {"none", "raw"}:
        out = arr
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size < 10:
            out = np.zeros_like(arr, dtype=np.float32)
        else:
            lo, hi = np.nanpercentile(finite, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
            out = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

        if mode in {"highpass", "normalize_highpass", "norm_highpass"}:
            sigma = float(mc_cfg.get("highpass_sigma", 12) or 0)
            if sigma > 0:
                out = out - gaussian_filter(out, sigma=sigma)

    out = np.asarray(out, dtype=np.float32)
    out[~np.isfinite(out)] = 0
    return out


def _read_motion_reference_image(
    paths: Sequence[Path | tuple[Path, Path]],
    *,
    kwargs: dict[str, Any],
    mc_cfg: dict[str, Any],
) -> np.ndarray:
    """Load the frame or frame sequence used to estimate raw motion correction."""
    mode = str(mc_cfg.get("reference_mode", "mean_first_n")).strip().lower()
    idx0 = int(mc_cfg.get("reference_frame_index", 0) or 0)
    idx0 = min(max(idx0, 0), len(paths) - 1)

    if mode in {"first", "first_frame", "frame"}:
        return read_gray_frame(paths[idx0], **kwargs).astype(np.float32)

    if mode in {"mean_first_n", "mean", "average_first_n"}:
        n = int(mc_cfg.get("reference_n_frames", 20) or 1)
        n = max(1, min(n, len(paths) - idx0))
        frames = [read_gray_frame(paths[i], **kwargs).astype(np.float32) for i in range(idx0, idx0 + n)]
        return np.mean(np.stack(frames, axis=0), axis=0).astype(np.float32)

    raise ValueError("raw_motion_correction.reference_mode must be 'first_frame' or 'mean_first_n'")


def _estimate_translation_shift(
    reference_feature: np.ndarray,
    moving_frame: np.ndarray,
    *,
    mc_cfg: dict[str, Any],
    previous_shift: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float, bool, str]:
    """Estimate (dy, dx) to apply to moving_frame to align it to reference."""
    if phase_cross_correlation is None:
        raise ImportError("skimage.registration.phase_cross_correlation is required for raw motion correction")

    moving_feature = _motion_preprocess_image(moving_frame, mc_cfg)
    try:
        shift_yx, error, phasediff = phase_cross_correlation(
            reference_feature,
            moving_feature,
            upsample_factor=int(mc_cfg.get("upsample_factor", 10) or 1),
            normalization=mc_cfg.get("normalization", "phase"),
        )
        dy = float(shift_yx[0])
        dx = float(shift_yx[1])
        error_f = float(error) if np.isfinite(error) else float("nan")
        phasediff_f = float(phasediff) if np.isfinite(phasediff) else float("nan")
    except Exception as e:
        dy, dx = float("nan"), float("nan")
        error_f, phasediff_f = float("nan"), float("nan")
        reason = f"estimate_failed:{type(e).__name__}"
    else:
        reason = "ok"

    max_shift = float(mc_cfg.get("max_shift_px", 30) or 0)
    invalid = (
        not np.isfinite(dy)
        or not np.isfinite(dx)
        or (max_shift > 0 and (abs(dy) > max_shift or abs(dx) > max_shift))
    )
    used_fallback = False
    if invalid:
        policy = str(mc_cfg.get("fail_policy", "use_previous")).strip().lower()
        used_fallback = True
        if policy == "use_previous":
            dy, dx = previous_shift
            reason = reason if reason != "ok" else "shift_out_of_bounds_use_previous"
        elif policy in {"identity", "zero"}:
            dy, dx = 0.0, 0.0
            reason = reason if reason != "ok" else "shift_out_of_bounds_identity"
        elif policy == "error":
            raise RuntimeError(f"Invalid motion-correction shift: dy={dy}, dx={dx}, max_shift_px={max_shift}")
        else:
            raise ValueError("raw_motion_correction.fail_policy must be 'use_previous', 'identity', or 'error'")

    return float(dy), float(dx), error_f, phasediff_f, used_fallback, reason


def _apply_translation_shift(frame: np.ndarray, dy: float, dx: float, *, mc_cfg: dict[str, Any], out_dtype: np.dtype) -> np.ndarray:
    """Apply a spatial translation to one raw frame in pixel coordinates."""
    shifted = ndi_shift(
        np.asarray(frame, dtype=np.float32),
        shift=(float(dy), float(dx)),
        order=int(mc_cfg.get("interpolation_order", 1) or 1),
        mode=str(mc_cfg.get("shift_mode", "nearest")),
        prefilter=False,
    )
    if np.issubdtype(out_dtype, np.integer):
        info = np.iinfo(out_dtype)
        shifted = np.clip(np.rint(shifted), info.min, info.max)
    return shifted.astype(out_dtype, copy=False)


def save_motion_corrected_raw_zarr_for_row(
    row: pd.Series,
    config: dict[str, Any],
    *,
    info: dict[str, Any] | None = None,
    overwrite: bool | None = None,
    show_progress: bool = True,
) -> Path:
    """Write a second raw zarr corrected for within-recording translation.

    The shift is estimated on one selected wavelength and then applied to every
    wavelength at the same time index, preserving cross-wavelength pixel
    correspondence for hemoglobin calculations.
    """
    mc_cfg = raw_motion_correction_config(config)
    if not bool(mc_cfg.get("enabled", False)):
        return motion_corrected_raw_store_path(row["raw_zarr_path_resolved"], config)

    raw_cfg = config.get("raw", {}) or {}
    if overwrite is None:
        overwrite = bool(require_step4(config).get("overwrite_raw", False))

    source_store = Path(row["raw_zarr_path_resolved"])
    store = motion_corrected_raw_store_path(source_store, config)
    raw_key = cfg_get(config, "raw", "zarr_key", default="raw_record")
    if store.exists():
        if not overwrite:
            if zarr_group_has_array(store, raw_key):
                return store
            raise RuntimeError(
                f"Motion-corrected raw zarr exists but does not contain {raw_key!r}: {store}. "
                "Delete it or rerun with overwrite_raw=True."
            )
        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)

    if info is None:
        info = inspect_raw_files_for_row(row, config)
    wave_to_paths = info["wave_to_paths_resampled"]
    wavelengths = list(wave_to_paths.keys())
    n_time = int(info["n_time"])
    fps = float(raw_cfg.get("fps", 20))
    kwargs = raw_read_kwargs(config)

    estimate_wave = mc_cfg.get("estimate_wavelength")
    if estimate_wave is None or not str(estimate_wave).strip():
        estimate_wave = str(raw_cfg.get("reference_wave", wavelengths[0]))
    estimate_wave = str(estimate_wave)
    if estimate_wave not in wave_to_paths:
        raise KeyError(f"raw_motion_correction.estimate_wavelength={estimate_wave!r} is not present in raw.wave_pattern")

    sample = read_gray_frame(wave_to_paths[wavelengths[0]][0], **kwargs)
    y_size, x_size = sample.shape
    out_dtype = np.dtype(mc_cfg.get("store_dtype") or raw_cfg.get("store_dtype") or "float32")
    shape = (n_time, len(wavelengths), y_size, x_size)
    chunks = normalise_chunks(raw_cfg.get("chunks", [64, 1, -1, -1]), shape)
    time_values = np.arange(n_time, dtype=np.float64) / fps

    attrs = {
        "stem": str(row["stem"]),
        "group_id": str(row.get("group_id", "")),
        "ref_stem": str(row.get("ref_stem", "")),
        "source_dir": str(row.get("img_dir_resolved", "")),
        "fps": fps,
        "scale": float(raw_cfg.get("scale", 1.0)),
        "rotation": float(raw_cfg.get("rotation", 0.0)),
        "reference_wave": str(raw_cfg.get("reference_wave", wavelengths[0])),
        "resample_mode": str(raw_cfg.get("resample_mode", "repeat")),
        "wave_pattern_json": json.dumps(raw_cfg.get("wave_pattern", {}), ensure_ascii=False),
        "created_at": _dt.datetime.now().isoformat(),
        "motion_correction": "phase_cross_correlation_translation",
        "motion_correction_estimate_wavelength": estimate_wave,
        "motion_correction_config_json": json.dumps(mc_cfg, ensure_ascii=False),
        "source_unaligned_raw_zarr": str(source_store),
    }
    attrs.update(safe_scalar_attrs(row))

    arr = write_xarray_compatible_raw_zarr(
        store,
        data_shape=shape,
        chunks=chunks,
        dtype=out_dtype,
        wavelengths=wavelengths,
        time_values=time_values,
        attrs=attrs,
        config=config,
    )

    reference_img = _read_motion_reference_image(wave_to_paths[estimate_wave], kwargs=kwargs, mc_cfg=mc_cfg)
    reference_feature = _motion_preprocess_image(reference_img, mc_cfg)
    batch = int(chunks[0])
    records: list[dict[str, Any]] = []
    prev_shift = (0.0, 0.0)

    time_iter = range(0, n_time, batch)
    if show_progress:
        time_iter = tqdm(time_iter, desc=f"Motion-corrected raw zarr [{row['stem']}]", leave=False)

    for start in time_iter:
        stop = min(start + batch, n_time)
        for ti in range(start, stop):
            estimate_frame = read_gray_frame(wave_to_paths[estimate_wave][ti], **kwargs).astype(np.float32)
            dy, dx, error, phasediff, used_fallback, reason = _estimate_translation_shift(
                reference_feature,
                estimate_frame,
                mc_cfg=mc_cfg,
                previous_shift=prev_shift,
            )
            if not used_fallback or str(mc_cfg.get("fail_policy", "use_previous")).strip().lower() == "use_previous":
                prev_shift = (dy, dx)

            for wi, wave in enumerate(wavelengths):
                frame = read_gray_frame(wave_to_paths[wave][ti], **kwargs)
                arr[ti, wi, :, :] = _apply_translation_shift(frame, dy, dx, mc_cfg=mc_cfg, out_dtype=out_dtype)

            records.append({
                "stem": str(row["stem"]),
                "time_index": ti,
                "time_sec": ti / fps,
                "estimate_wavelength": estimate_wave,
                "shift_y_px": dy,
                "shift_x_px": dx,
                "error": error,
                "phasediff": phasediff,
                "used_fallback": bool(used_fallback),
                "status": reason,
            })

    if bool(mc_cfg.get("save_transforms_csv", True)):
        csv_path = motion_transforms_csv_path(row, config)
        pd.DataFrame(records).to_csv(csv_path, index=False)
        try:
            root = zarr_open_group_v2_compat(store, mode="a")
            root.attrs["motion_correction_transforms_csv"] = str(csv_path)
        except Exception:
            pass

    return store



def _fingerprint(payload: Any) -> str:
    """Compute a stable fingerprint for a serializable processing-information payload."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    """Compute a content hash for a file used by the processing fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_fingerprint(row: pd.Series, config: dict[str, Any], info: dict[str, Any] | None = None) -> str:
    """Fingerprint the raw recording and raw-processing configuration."""
    if info is None:
        info = inspect_raw_files_for_row(row, config)
    files = [p for paths in info["wave_to_paths_raw"].values() for p in paths]
    # The file list and stats detect changed acquisition inputs without hashing
    # gigabytes of BMP frames for every subsequent process run.
    return _fingerprint({"raw": config.get("raw", {}),
                         "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files]})


def processed_fingerprint(row: pd.Series, config: dict[str, Any], raw_store: Path) -> str:
    """Fingerprint the signal-processing configuration and dependent input files."""
    masks_root = build_standard_paths(config, create_or_load_project(config, Path(".")))["masks_root"]
    mask_paths = canonical_mask_paths(masks_root, str(row["stem"]))
    brain = mask_paths["brain_manual"] if mask_paths["brain_manual"].exists() else mask_paths["brain_auto"]
    bregma = mask_paths["bregma_manual"] if mask_paths["bregma_manual"].exists() else mask_paths["bregma_auto"]
    req = [brain, bregma, mask_paths["bad_auto"], mask_paths["vessels_auto"]]
    files = {str(p): _file_hash(p) if p.is_file() else "missing" for p in req}
    hb = cfg_get(config,"signal_processing","hemoglobin",default={}) or {}
    for key in ("coeff_abs_path", "pathlength_path"):
        path = Path(str(hb.get(key, "")))
        files[key] = _file_hash(path) if path.is_file() else "missing"
    root = zarr_open_group_v2_compat(raw_store, mode="r")
    return _fingerprint({"raw_fingerprint": root.attrs.get("raw_fingerprint", ""),
                         "signal_processing": config.get("signal_processing", {}), "mask_and_tables": files})


def is_complete_zarr(store: Path, config: dict[str, Any], *, raw: bool, fingerprint: str | None = None) -> bool:
    """Check raw or processed Zarr completion markers and optional fingerprints."""
    if not store.exists():
        return False
    try:
        root = zarr_open_group_v2_compat(store, mode="r")
        if not bool(root.attrs.get("step4_complete", False)):
            return False
        if fingerprint is not None and root.attrs.get("step4_fingerprint") != fingerprint:
            return False
        if raw:
            key = str(cfg_get(config,"raw","zarr_key",default="raw_record"))
            return (key in root and "time" in root
                    and "source_intensity_stats" in root
                    and "raw_min" in root["source_intensity_stats"]
                    and "raw_max" in root["source_intensity_stats"])
        return len(list(root.array_keys())) > 0
    except Exception:
        return False


def assert_existing_raw_compatible_for_processing(raw_root: Any, config: dict[str, Any], raw_path: Path) -> None:
    """Check the saved raw pixels/time/channel interpretation matches current raw settings.

    Explicit reuse should not require source BMP stat/fingerprint equality, because
    the processed phase reads the saved raw Zarr and its saved raw intensity stats.
    Differences in the settings that change raw pixels/time/channel order are unsafe.
    """
    stored_json = raw_root.attrs.get("raw_config_json")
    if not isinstance(stored_json, str) or not stored_json:
        raise RuntimeError(
            f"Cannot reuse {raw_path}: missing saved raw_config_json. "
            "Do not bypass validation for an unidentified raw Zarr."
        )
    try:
        saved = json.loads(stored_json)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot decode saved raw_config_json in {raw_path}: {exc}") from exc
    current = config.get("raw", {}) or {}
    # Exclude purely runtime/storage controls such as compression, chunk shape,
    # worker count, overwrite and Step1's reference-image frame indices.
    critical_keys = (
        "suffixes", "file_regex", "sort_key", "wave_pattern",
        "reference_wave", "resample_mode", "fps", "scale", "rotation",
        "read_end", "read_dtype", "mean_pair_dtype", "store_dtype", "zarr_key",
    )
    mismatch = [key for key in critical_keys if saved.get(key) != current.get(key)]
    if mismatch:
        details = "; ".join(
            f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
            for key in mismatch
        )
        raise RuntimeError(
            "Cannot reuse existing raw Zarr: settings affecting raw data differ. "
            + details + ". Restore the original raw settings or rebuild raw intentionally."
        )
    expected_waves = [str(w) for w in (current.get("wave_pattern") or {})]
    try:
        actual_waves = [str(w) for w in raw_root["wavelength"][:]]
    except Exception as exc:
        raise RuntimeError(f"Cannot read wavelength coordinates in {raw_path}: {exc}") from exc
    if actual_waves != expected_waves:
        raise RuntimeError(
            f"Cannot reuse {raw_path}: raw wavelengths {actual_waves} != configured {expected_waves}"
        )



def save_raw_zarr_for_row_batch(
    row: pd.Series,
    config: dict[str, Any],
    *,
    overwrite: bool | None = None,
    show_progress: bool = True,
) -> Path:
    """Construct and save a raw Zarr recording from wavelength-interleaved BMP frames.
    
    Parameters
    ----------
    row : pandas.Series
        One selected recording row with its ``stem`` and paths.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    overwrite : bool, optional
        Override the configured reuse/overwrite setting.
    show_progress : bool
        Whether to print progress information.
    
    Returns
    -------
    pathlib.Path
        Path to the created or validated raw Zarr store.
    
    Notes
    -----
    Reads BMP sequences using ``raw.wave_pattern``, ``raw.rotation``,
    ``raw.scale`` and the configured temporal sampling. Raw source-image minima
    and maxima are recorded separately for later pixel-validity checks.
    Completion and fingerprint metadata govern reuse of existing stores."""
    raw_cfg = config.get("raw", {}) or {}
    step4_cfg = require_step4(config)
    motion_enabled = raw_motion_correction_enabled(config)
    if overwrite is None:
        overwrite = bool(step4_cfg.get("overwrite_raw", False))
    final_store = Path(row["raw_zarr_path_resolved"])
    raw_key = cfg_get(config, "raw", "zarr_key", default="raw_record")
    if show_progress:
        print(f"[RAW] {row['stem']}: inspecting input BMPs...", flush=True)
    info = inspect_raw_files_for_row(row, config)
    current_fp = raw_fingerprint(row, config, info)
    if final_store.exists() and not overwrite:
        if is_complete_zarr(final_store, config, raw=True, fingerprint=current_fp):
            if show_progress:
                print(f"[RAW] {row['stem']}: complete Zarr already exists; skipped (overwrite=False)", flush=True)
            return final_store
        raise RuntimeError(f"Raw Zarr incomplete or its inputs/config changed: {final_store}; use overwrite_raw=True")
    store = final_store.with_name(final_store.name + ".incomplete")
    if store.exists():
        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    wave_to_paths = info["wave_to_paths_resampled"]
    wavelengths = list(wave_to_paths.keys())
    n_time = int(info["n_time"])
    fps = float(raw_cfg.get("fps", 20))
    kwargs = raw_read_kwargs(config)
    raw_min_full: np.ndarray | None = None
    raw_max_full: np.ndarray | None = None
    inspected_paths: set[Path] = set()

    def accumulate_source_stats(path: Path, frame: np.ndarray) -> None:
        """Update per-pixel minimum and maximum values in source-image geometry."""
        nonlocal raw_min_full, raw_max_full
        if path in inspected_paths:
            return
        if raw_min_full is None:
            raw_min_full = frame.copy()
            raw_max_full = frame.copy()
        elif frame.shape != raw_min_full.shape:
            raise ValueError(f"Inconsistent BMP shape: {path}")
        else:
            np.minimum(raw_min_full, frame, out=raw_min_full)
            np.maximum(raw_max_full, frame, out=raw_max_full)
        inspected_paths.add(path)

    sample = read_gray_frame(wave_to_paths[wavelengths[0]][0], **kwargs,
                             source_frame_callback=accumulate_source_stats)
    y_size, x_size = sample.shape
    dtype = np.dtype(raw_cfg.get("store_dtype") or sample.dtype)
    shape = (n_time, len(wavelengths), y_size, x_size)
    chunks = normalise_chunks(raw_cfg.get("chunks", [64, 1, -1, -1]), shape)
    time_values = np.arange(n_time, dtype=np.float64) / fps

    attrs = {
        "stem": str(row["stem"]),
        "group_id": str(row.get("group_id", "")),
        "ref_stem": str(row.get("ref_stem", "")),
        "source_dir": str(row.get("img_dir_resolved", "")),
        "fps": fps,
        "scale": float(raw_cfg.get("scale", 1.0)),
        "rotation": float(raw_cfg.get("rotation", 0.0)),
        "reference_wave": str(raw_cfg.get("reference_wave", wavelengths[0])),
        "resample_mode": str(raw_cfg.get("resample_mode", "repeat")),
        "wave_pattern_json": json.dumps(raw_cfg.get("wave_pattern", {}), ensure_ascii=False),
        "raw_config_json": json.dumps(raw_cfg, ensure_ascii=False, sort_keys=True, default=str),
        "created_at": _dt.datetime.now().isoformat(),
        "raw_fingerprint": current_fp,
        "step4_fingerprint": current_fp,
        "step4_complete": False,
    }
    attrs.update(safe_scalar_attrs(row))
    arr = write_xarray_compatible_raw_zarr(
        store,
        data_shape=shape,
        chunks=chunks,
        dtype=dtype,
        wavelengths=wavelengths,
        time_values=time_values,
        attrs=attrs,
        config=config,
    )

    root = zarr_open_group_v2_compat(store, mode="a")
    batch = int(chunks[0])
    progress = (tqdm(total=n_time * len(wavelengths),
                     desc=f"Building raw + intensity stats [{row['stem']}]", unit="frame")
                if show_progress else None)
    try:
        for wi, wave in enumerate(wavelengths):
            paths = wave_to_paths[wave]
            if progress is not None:
                progress.set_postfix_str(f"{wave} nm")
            for start in range(0, n_time, batch):
                stop = min(start + batch, n_time)
                frames = [read_gray_frame(obj, **kwargs, source_frame_callback=accumulate_source_stats)
                          .astype(dtype, copy=False) for obj in paths[start:stop]]
                arr[start:stop, wi, :, :] = np.stack(frames, axis=0)
                if progress is not None:
                    progress.update(stop - start)
    finally:
        if progress is not None:
            progress.close()

    # Include source exposures unused by a resampling mode; every BMP counts.
    for paths in info["wave_to_paths_raw"].values():
        for path in paths:
            if path not in inspected_paths:
                frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    raise ValueError(f"Cannot read raw BMP for source intensity stats: {path}")
                accumulate_source_stats(path, frame)

    if raw_min_full is None or raw_max_full is None:
        raise ValueError("No BMP frames for source intensity statistics")
    stats = root.create_group("source_intensity_stats")
    stats.attrs["space"] = "source_bmp_before_rotation_and_resize"
    for name, data in (("raw_min", raw_min_full), ("raw_max", raw_max_full)):
        dataset = zarr_create_dataset_compat(stats, name, data=data,
                                            shape=data.shape, dtype=data.dtype)
        dataset.attrs["_ARRAY_DIMENSIONS"] = ["source_y", "source_x"]

    root.attrs["step4_complete"] = True
    if final_store.exists():
        shutil.rmtree(final_store)
    store.rename(final_store)
    if show_progress:
        print(f"[RAW STATS] {row['stem']}: source raw_min/raw_max saved ({raw_min_full.shape})")
    if motion_enabled:
        save_motion_corrected_raw_zarr_for_row(
            row, config, info=info, overwrite=overwrite, show_progress=show_progress,
        )
    return final_store


def build_raw_dataarray_for_row_lazy(row: pd.Series, config: dict[str, Any]) -> xr.DataArray:
    """Optional lazy builder. Useful for experiments; batch writer is preferred."""
    raw_cfg = config.get("raw", {}) or {}
    info = inspect_raw_files_for_row(row, config)
    wave_to_paths = info["wave_to_paths_resampled"]
    wavelengths = list(wave_to_paths.keys())
    kwargs = raw_read_kwargs(config)
    sample = read_gray_frame(wave_to_paths[wavelengths[0]][0], **kwargs)
    y_size, x_size = sample.shape
    dtype = np.dtype(raw_cfg.get("store_dtype") or sample.dtype)
    n_time = int(info["n_time"])
    fps = float(raw_cfg.get("fps", 20))

    wave_arrays: list[da.Array] = []
    for wave in wavelengths:
        delayed_frames = [
            da.from_delayed(
                dask_delayed(read_gray_frame)(obj, **kwargs),
                shape=(y_size, x_size),
                dtype=dtype,
            )
            for obj in wave_to_paths[wave]
        ]
        wave_arrays.append(da.stack(delayed_frames, axis=0))
    data = da.stack(wave_arrays, axis=1).astype(dtype)
    data = data.rechunk(normalise_chunks(raw_cfg.get("chunks", [64, 1, -1, -1]), data.shape))
    return xr.DataArray(
        data,
        dims=("time", "wavelength", "y", "x"),
        coords={
            "time": np.arange(n_time, dtype=np.float64) / fps,
            "wavelength": np.asarray(wavelengths, dtype=str),
            "y": np.arange(y_size, dtype=np.int32),
            "x": np.arange(x_size, dtype=np.int32),
        },
        name=raw_cfg.get("zarr_key", "raw_record"),
        attrs=safe_scalar_attrs(row),
    )


def open_raw_zarr(path: str | Path, config: dict[str, Any]) -> xr.DataArray:
    """Open a saved raw recording as a labeled xarray DataArray."""
    key = cfg_get(config, "signal_processing", "input", "raw_zarr_key", default=cfg_get(config, "raw", "zarr_key", default="raw_record"))
    # The stores are not consolidated by default. Setting consolidated=False avoids
    # the slow warning path where xarray first searches for .zmetadata.
    ds = xr.open_zarr(path, chunks="auto", consolidated=False, mask_and_scale=False)
    if key not in ds:
        available = list(ds.data_vars)
        raise KeyError(
            f"No variable named {key!r} in raw zarr store: {path}. "
            f"Available variables: {available}. "
            "Most likely the store is empty/partial from a failed previous write. "
            "Delete it or rebuild with overwrite_raw=True."
        )
    return ds[key]


def preview_raw_frames_from_zarr(row: pd.Series, config: dict[str, Any], *, frame_index: int | None = None, save_path: str | Path | None = None) -> plt.Figure:
    """Plot a selection of raw wavelength frames from an existing Zarr store."""
    raw_path = select_raw_zarr_path_for_processing(row, config)
    raw = open_raw_zarr(raw_path, config)
    frame_index = int(frame_index if frame_index is not None else cfg_get(config, "step4", "notebook", "preview_frame_index", default=10))
    waves = [str(w) for w in raw.coords["wavelength"].values]
    fig, axes = plt.subplots(1, len(waves), figsize=(4 * len(waves), 4))
    if len(waves) == 1:
        axes = [axes]
    idx = min(max(frame_index, 0), raw.sizes["time"] - 1)
    for wi, (ax, wave) in enumerate(zip(axes, waves)):
        img = raw.isel(wavelength=wi, time=idx).compute().values
        vmin, vmax = np.nanpercentile(img, [2, 98])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{wave} nm | zarr frame {idx}")
        ax.axis("off")
    fig.suptitle(f"Raw zarr preview ({selected_raw_variant_for_row(row, config)}): {row['stem']}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# Masks, ROI and traces
# ============================================================


def resize_mask_to_shape(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to the raw zarr image shape using nearest-neighbour interpolation."""
    target_h, target_w = int(image_shape[0]), int(image_shape[1])
    if tuple(mask.shape) == (target_h, target_w):
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def load_mask_array(
    path_value: Any,
    *,
    image_shape: tuple[int, int] | None = None,
    resize_if_needed: bool = False,
) -> xr.DataArray:
    """Load a spatial mask from its saved array path.
    
    Parameters
    ----------
    path_value : str or pathlib.Path
        Saved path to a spatial mask.
    image_shape : tuple of int
        Requested spatial shape (y, x).
    resize_if_needed : object
        See the corresponding Step4 call site and YAML settings.
    
    Notes
    -----
    Masks are spatial (y, x) arrays; they refer to the geometry used
    when Step4 processes the corresponding recording. Their semantic meaning
    is determined by ``signal_processing.masks``."""
    path = Path(str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = np.load(path).astype(bool)
    if image_shape is not None and tuple(mask.shape) != tuple(image_shape):
        if resize_if_needed:
            old_shape = tuple(mask.shape)
            mask = resize_mask_to_shape(mask, image_shape)
            print(f"[Step4] Resized mask to raw zarr shape: {path.name}: {old_shape} -> {tuple(mask.shape)}")
        else:
            raise ValueError(
                f"Mask shape mismatch for {path}: {mask.shape} vs {image_shape}. "
                "Either rebuild raw_zarr with raw.scale=1.0 to match Step2 masks, "
                "or set signal_processing.masks.resize_to_raw_if_needed: true."
            )
    return xr.DataArray(mask, dims=("y", "x"), coords={"y": np.arange(mask.shape[0]), "x": np.arange(mask.shape[1])})


def _zeros_mask_like(mask: xr.DataArray) -> xr.DataArray:
    """Create a false-valued spatial mask with the same shape as a reference mask."""
    return xr.zeros_like(mask).astype(bool)


def _load_optional_mask_array(
    path: Path,
    *,
    image_shape: tuple[int, int],
    resize_if_needed: bool,
    like: xr.DataArray | None = None,
) -> xr.DataArray:
    """Read an optional mask or return a missing-value indicator."""
    if path.exists():
        return load_mask_array(path, image_shape=image_shape, resize_if_needed=resize_if_needed)
    if like is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    return _zeros_mask_like(like)


def load_masks_for_row(row: pd.Series, config: dict[str, Any], image_shape: tuple[int, int], raw_valid: xr.DataArray | None = None) -> dict[str, xr.DataArray]:
    """Load canonical per-recording masks and construct Step4 processing ROIs.
    
    Parameters
    ----------
    row : pandas.Series
        Recording metadata including its `stem` identifier.
    config : dict
        Configuration; mask resizing is controlled by
        `signal_processing.masks.resize_to_raw_if_needed`.
    image_shape : tuple of int
        Raw signal image shape (y, x).
    raw_valid : xarray.DataArray, optional
        Spatial validity mask (y, x); defaults to all valid pixels.
    
    Returns
    -------
    masks : dict of str to xarray.DataArray
        Boolean (y, x) masks including `brain`, `bad`, `bregma`, `vessels`,
        `brainvessels_signal`, `brain_signal`, `vessels_signal`, `background`,
        and `raw_valid`.
    
    Notes
    -----
    Canonical mask paths come from Step2/Step3 under the project's configured
    Step2 output directory. `brain_manual.npy` and `bregma_manual.npy` take
    precedence over their automatic counterparts when present. The files
    `bad_auto.npy` and `vessels_auto.npy` are read directly. Derived signal
    ROIs exclude invalid pixels and selected anatomical/artifact masks.
    The stored canonical source masks are not modified by this function."""
    if "stem" not in row.index or pd.isna(row.get("stem")) or not str(row.get("stem")).strip():
        raise ValueError("Step4 row must contain a non-empty stem")

    project_paths = create_or_load_project(config, Path("."))
    paths = build_standard_paths(config, project_paths)
    mask_paths = canonical_mask_paths(paths, str(row["stem"]))
    resize_if_needed = bool(cfg_get(config, "signal_processing", "masks", "resize_to_raw_if_needed", default=False))

    brain_path = mask_paths["brain_manual"] if mask_paths["brain_manual"].exists() else mask_paths["brain_auto"]
    if not brain_path.exists():
        raise FileNotFoundError(
            f"brain mask is required for stem={row.get('stem', '')}. "
            f"Expected {mask_paths['brain_manual'].name} or {mask_paths['brain_auto'].name} in {brain_path.parent}"
        )
    brain = load_mask_array(brain_path, image_shape=image_shape, resize_if_needed=resize_if_needed)

    bad = load_mask_array(mask_paths["bad_auto"], image_shape=image_shape, resize_if_needed=resize_if_needed)
    bregma_path = mask_paths["bregma_manual"] if mask_paths["bregma_manual"].exists() else mask_paths["bregma_auto"]
    bregma = load_mask_array(bregma_path, image_shape=image_shape, resize_if_needed=resize_if_needed)
    vessels = load_mask_array(mask_paths["vessels_auto"], image_shape=image_shape, resize_if_needed=resize_if_needed)

    brain_bool = brain.astype(bool)
    bad_bool = bad.astype(bool)
    bregma_bool = bregma.astype(bool)
    vessels_bool = vessels.astype(bool)

    # Derived signal masks.  Canonical source masks stay unchanged; these masks
    # are processing ROIs in raw_zarr space.
    valid = raw_valid.astype(bool) if raw_valid is not None else xr.ones_like(brain_bool, dtype=bool)
    if tuple(valid.shape) != tuple(brain_bool.shape):
        raise ValueError("raw_valid shape does not match canonical masks")
    # Pixel masks must share spatial coordinates; xarray may decode zero-valued
    # integer Zarr coordinate labels as NaN and silently drop row/column zero.
    valid = valid.assign_coords(y=brain_bool.coords["y"], x=brain_bool.coords["x"])
    brainvessels_signal = brain_bool & (~bad_bool) & (~bregma_bool) & valid
    brain_signal = brainvessels_signal & (~vessels_bool)
    vessels_signal = brainvessels_signal & vessels_bool
    background = (~brain_bool) & (~bad_bool) & (~bregma_bool) & valid

    out = {
        "brain": brain_bool,
        "brain_source": brain_bool,
        "bad": bad_bool,
        "bregma": bregma_bool,
        "vessels": vessels_bool,
        "raw_valid": valid,
        "brainvessels_signal": brainvessels_signal.astype(bool),
        "brain_signal": brain_signal.astype(bool),
        "vessels_signal": vessels_signal.astype(bool),
        "background": background.astype(bool),
    }
    out["brain"].attrs["source_path"] = str(brain_path)
    out["bad"].attrs["source_path"] = str(mask_paths["bad_auto"])
    out["bregma"].attrs["source_path"] = str(bregma_path)
    out["vessels"].attrs["source_path"] = str(mask_paths["vessels_auto"])
    return out


def make_signal_roi(masks: dict[str, xr.DataArray], config: dict[str, Any]) -> xr.DataArray:
    """Select the configured ROI for extracting biological signal traces.
    
    Parameters
    ----------
    masks : dict
        Named spatial masks used for signal processing.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Notes
    -----
    The ROI name comes from ``signal_processing.roi.signal_roi``.
    Its mask is used to extract traces, not to replace the full spatial maps."""
    mode = cfg_get(config, "signal_processing", "roi", "signal_roi", default="brain_signal")
    roi = _mask_roi_from_name(masks, mode)
    min_pixels = int(cfg_get(config, "signal_processing", "roi", "min_signal_pixels", default=100))
    return _require_min_pixels(roi, min_pixels, "signal ROI")

def make_background_roi(masks: dict[str, xr.DataArray], config: dict[str, Any]) -> xr.DataArray:
    """Select the configured background ROI for drift correction.
    
    Parameters
    ----------
    masks : dict
        Named spatial masks used for signal processing.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Notes
    -----
    The selected name comes from ``signal_processing.roi.background_roi``.
    This ROI is only required when ``signal_processing.drift_correction.enabled``
    requires background correction."""
    mode = cfg_get(config, "signal_processing", "roi", "background_roi", default="background")
    roi = _mask_roi_from_name(masks, mode)
    min_pixels = int(cfg_get(config, "signal_processing", "roi", "min_background_pixels", default=100))
    return _require_min_pixels(roi, min_pixels, "background ROI")


def _mask_roi_from_name(masks: dict[str, xr.DataArray], name: Any) -> xr.DataArray:
    """Return a named ROI from the mask dictionary.

    Preferred names are the explicit derived masks:
      brainvessels_signal = brain & ~bad & ~bregma
      brain_signal        = brain & ~bad & ~bregma & ~vessels
      vessels_signal      = brain & ~bad & ~bregma & vessels
      background          = ~brain & ~bad & ~bregma
    """
    mode = str(name or "brain_signal").strip().lower()
    aliases = {
        "brain_with_vessels": "brainvessels_signal",
        "brain-vessels-included": "brainvessels_signal",
        "brain_minus_bad_bregma": "brainvessels_signal",
        "brain_without_bad_bregma": "brainvessels_signal",
        "brain_no_vessels": "brain_signal",
        "brain_without_vessels": "brain_signal",
        "brain_minus_vessels": "brain_signal",
        "vessels": "vessels_signal",
        "vessel_signal": "vessels_signal",
    }
    key = aliases.get(mode, mode)
    if key in masks:
        return masks[key].astype(bool)
    raise ValueError(f"Unsupported ROI/mask name: {name!r}. Available: {sorted(masks)}")

def _require_min_pixels(roi: xr.DataArray, min_pixels: int, label: str) -> xr.DataArray:
    """Require at least the configured number of valid pixels in an ROI."""
    n = int(roi.sum().compute().item()) if hasattr(roi.data, "compute") else int(roi.sum().item())
    if n < int(min_pixels):
        raise RuntimeError(f"Too few {label} pixels: {n} < {int(min_pixels)}")
    return roi.astype(bool)

def make_named_roi(masks: dict[str, xr.DataArray], name: Any, *, min_pixels: int = 1, label: str = "ROI") -> xr.DataArray:
    """Select a named mask or ROI from the calculated mask dictionary."""
    return _require_min_pixels(_mask_roi_from_name(masks, name), int(min_pixels), label)

def mean_trace(data: xr.DataArray, roi: xr.DataArray) -> xr.DataArray:
    """Calculate the spatial mean time course inside the supplied ROI.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    roi : xarray.DataArray
        Spatial Boolean mask with dimensions (y, x).
    
    Returns
    -------
    xarray.DataArray
        ROI-mean trace with its original time coordinate.
    
    Notes
    -----
    Spatial pixels are averaged over the (y, x) mask while preserving
    the time coordinate. Temporal units follow the input signal."""
    return data.where(roi).mean(dim=("y", "x"), skipna=True)


def preview_masks(row: pd.Series, masks: dict[str, xr.DataArray], signal_roi: xr.DataArray | None = None, *, save_path: str | Path | None = None) -> plt.Figure:
    """Plot the masks and selected signal ROI for visual quality control."""
    items = [
        ("brain_source", masks["brain"]),
        ("bad", masks["bad"]),
        ("bregma", masks["bregma"]),
        ("vessels", masks["vessels"]),
        ("brainvessels_signal", masks["brainvessels_signal"]),
        ("brain_signal", masks["brain_signal"]),
        ("vessels_signal", masks["vessels_signal"]),
        ("background", masks["background"]),
    ]
    if signal_roi is not None:
        items.append(("signal_roi", signal_roi))
    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4))
    if len(items) == 1:
        axes = [axes]
    for ax, (name, mask) in zip(axes, items):
        ax.imshow(mask.values.astype(float), cmap="gray")
        ax.set_title(f"{name}\nN={int(mask.sum().compute().item())}")
        ax.axis("off")
    fig.suptitle(f"Masks: {row['stem']}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ============================================================
# Signal processing math
# ============================================================


def background_division(raw: xr.DataArray, background_roi: xr.DataArray, config: dict[str, Any]) -> tuple[xr.DataArray, xr.DataArray]:
    """Divide each intensity channel by its spatial background time course.
    
    Parameters
    ----------
    raw : xarray.DataArray
        Input intensity with wavelength, time, and spatial axes.
    background_roi : xarray.DataArray
        Background Boolean mask with spatial dimensions (y, x).
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Returns
    -------
    tuple of xarray.DataArray
        Background-divided intensity and the background trace.
    
    Notes
    -----
    The per-channel background trace is the spatial mean over the
    background ROI. Values with absolute magnitude at or below
    ``signal_processing.drift_correction.epsilon`` are masked before division."""
    eps = float(cfg_get(config, "signal_processing", "drift_correction", "epsilon", default=1e-6))
    bg_trace = mean_trace(raw.astype("float32"), background_roi)
    bg_trace.name = "background_trace"
    corrected = raw.astype("float32") / bg_trace.where(abs(bg_trace) > eps)
    corrected.name = "raw_background_divided"
    return corrected, bg_trace.astype("float32")


def prepare_intensity_for_processing(
    raw: xr.DataArray,
    background_roi: xr.DataArray | None,
    config: dict[str, Any],
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Apply the configured background-drift and baseline normalization order.
    
    Parameters
    ----------
    raw : xarray.DataArray
        Input multiwavelength intensity with spatial and temporal axes.
    background_roi : xarray.DataArray or None
        Boolean spatial (y, x) mask for the background time course. It is
        required when the configured drift correction is active.
    config : dict
        Configuration containing `signal_processing.drift_correction`,
        `signal_processing.baseline`, and `preprocessing_order`.
    
    Returns
    -------
    preprocessed : xarray.DataArray
        Float32 intensity-like array, preserving the wavelength/time/spatial axes.
    background_trace : xarray.DataArray or None
        Background time course, or None when drift correction is disabled.
    preprocessing_mode : str
        Actual processing-order indicator for later channel-specific functions.
    
    Notes
    -----
    With `drift_then_normalize`, background division precedes channel-specific
    I/I0 or dF/F normalization. With `normalize_then_drift`, I/I0 is computed
    first and then divided by the background trace; downstream reflectance
    processing avoids duplicate normalization. Disabling drift correction does
    not disable baseline calculation."""
    order = str(cfg_get(config, "signal_processing", "preprocessing_order", default="drift_then_normalize")).strip().lower()
    drift_enabled = bool(cfg_get(config, "signal_processing", "drift_correction", "enabled", default=True))
    method = cfg_get(config, "signal_processing", "drift_correction", "method", default="background_division")
    if method != "background_division":
        raise ValueError("Step4 supports drift_correction.method='background_division'")

    raw_f = raw.astype("float32")
    if order in {"drift_then_normalize", "drift_first", "background_then_normalize"}:
        if drift_enabled:
            corrected, bg_trace = background_division(raw_f, background_roi, config)
            return corrected.astype("float32"), bg_trace.astype("float32"), "drift_then_normalize"
        return raw_f, None, "no_drift_then_normalize"

    if order in {"normalize_then_drift", "normalize_first", "baseline_then_background"}:
        normalized = normalize_i_over_i0(raw_f, config)
        if drift_enabled:
            corrected, bg_trace = background_division(normalized, background_roi, config)
            return corrected.astype("float32"), bg_trace.astype("float32"), "normalize_then_drift"
        return normalized.astype("float32"), None, "normalize_no_drift"

    raise ValueError(
        "Unsupported signal_processing.preprocessing_order. "
        "Use 'drift_then_normalize' or 'normalize_then_drift'."
    )

def channel_i_over_i0(preprocessed: xr.DataArray, wavelength: int | float | str, config: dict[str, Any], preprocessing_mode: str) -> xr.DataArray:
    """Select a reflectance wavelength as I/I0, avoiding double normalization."""
    arr = select_wave(preprocessed, wavelength)
    if str(preprocessing_mode).startswith("normalize"):
        out = arr.astype("float32")
        out.name = "i_over_i0"
        return out
    return normalize_i_over_i0(arr, config)

def fluorescence_dff_from_preprocessed(preprocessed: xr.DataArray, wavelength: int | float | str, config: dict[str, Any], preprocessing_mode: str) -> xr.DataArray:
    """Compute fluorescence dF/F after the selected drift/normalization order."""
    arr = select_wave(preprocessed, wavelength)
    if str(preprocessing_mode).startswith("normalize"):
        out = arr - 1.0
        units = str(cfg_get(config, "signal_processing", "fluorescence", "output_units", default="percent")).lower()
        if units in {"percent", "%"}:
            out = out * 100.0
            out.attrs["units"] = "percent_dff"
        else:
            out.attrs["units"] = "fraction_dff"
        out.name = "fluorescence_dff"
        return out.astype("float32")
    return fluorescence_dff(arr, config)

def compute_baseline(data: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Calculate the configured reference intensity over the time dimension.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Returns
    -------
    xarray.DataArray
        Baseline intensity with the time dimension removed.
    
    Notes
    -----
    Supported methods are ``mean_all``, ``first_n`` (including
    ``first_n_frames``), and ``first_frame``, configured in
    ``signal_processing.baseline``. The output has no time axis."""
    method = cfg_get(config, "signal_processing", "baseline", "method", default="mean_all")
    frames = cfg_get(config, "signal_processing", "baseline", "frames", default=None)
    if method == "mean_all":
        return data.mean("time")
    if method in {"first_n", "first_n_frames"}:
        if isinstance(frames, (list, tuple)) and len(frames) == 2:
            return data.isel(time=slice(int(frames[0]), int(frames[1]))).mean("time")
        n = int(frames or 100)
        return data.isel(time=slice(0, n)).mean("time")
    if method == "first_frame":
        return data.isel(time=0)
    raise ValueError(f"Unknown baseline method: {method}")


def normalize_i_over_i0(data: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Normalize intensity relative to the configured baseline.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Returns
    -------
    xarray.DataArray
        Dimensionless I/I0 with the input time and spatial axes.
    
    Notes
    -----
    The ratio is dimensionless. The configured baseline epsilon masks
    near-zero denominator values; see ``signal_processing.baseline.epsilon``."""
    eps = float(cfg_get(config, "signal_processing", "baseline", "epsilon", default=1e-6))
    baseline = compute_baseline(data, config)
    out = data / baseline.where(abs(baseline) > eps)
    out.name = "i_over_i0"
    return out.astype("float32")


def reflectance_dA(i_over_i0: xr.DataArray) -> xr.DataArray:
    """Convert relative reflectance intensity to natural-log attenuation.
    
    Parameters
    ----------
    i_over_i0 : xarray.DataArray
        Baseline-normalized reflectance ratio; dimensionless.
    
    Returns
    -------
    xarray.DataArray
        Dimensionless natural-log attenuation with dimensions (time, y, x).
    
    Notes
    -----
    For positive input I/I0, the implementation computes
    ``-ln(I/I0)`` (natural logarithm). Nonpositive values are masked before
    applying the logarithm. The result is dimensionless."""
    out = -xr.apply_ufunc(np.log, i_over_i0.where(i_over_i0 > 0), dask="allowed")
    out.name = "dA"
    return out.astype("float32")


def fluorescence_dff(data: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Calculate baseline-normalized fluorescence changes.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Returns
    -------
    xarray.DataArray
        Fluorescence change in configured fraction or percent units.
    
    Notes
    -----
    Computes ``(F - F0) / (F0 + epsilon)``; the configured output
    unit can be a fraction or percentage. ``F0`` uses the configured baseline."""
    eps = float(cfg_get(config, "signal_processing", "baseline", "epsilon", default=1e-6))
    out = (data - compute_baseline(data, config)) / (compute_baseline(data, config) + eps)
    units = str(cfg_get(config, "signal_processing", "fluorescence", "output_units", default="percent")).lower()
    if units in {"percent", "%"}:
        out = out * 100.0
        out.attrs["units"] = "percent_dff"
    else:
        out.attrs["units"] = "fraction_dff"
    out.name = "fluorescence_dff"
    return out.astype("float32")


def read_coeff_table(path: str | Path) -> pd.DataFrame:
    """Read a hemoglobin extinction-coefficient table from CSV or whitespace-delimited text.
    
    Parameters
    ----------
    path : str or pathlib.Path
        Path to the relevant input file or store.
    
    Returns
    -------
    pandas.DataFrame
        Parsed extinction table."""
    try:
        df = pd.read_csv(path)
        if {"Wave", "HbO", "HbR"}.issubset(df.columns):
            return df
    except Exception:
        pass
    return pd.read_csv(path, sep=r"\s+", engine="python")


def lookup_pathlength(wave: int, table: pd.DataFrame) -> float:
    """Read or interpolate the optical pathlength at an acquisition wavelength.
    
    Parameters
    ----------
    wave : int
        Requested wavelength in nanometers.
    table : pandas.DataFrame
        Configured wavelength/pathlength lookup table.
    
    Returns
    -------
    float
        Optical pathlength in the units stored in the table (mm for the provided table).
    
    Notes
    -----
    The table columns are ``Wavelength`` and ``X``. Exact wavelength
    lookup is preferred; otherwise the mean of the entries at wavelengths
    ``wave - 1`` and ``wave + 1`` is used when both exist. The caller converts
    table pathlengths from millimeters to centimeters."""
    if "Wavelength" not in table.columns or "X" not in table.columns:
        raise ValueError("Pathlength table must contain Wavelength and X columns")
    waves = table["Wavelength"].astype(int)
    wave = int(wave)
    if wave in set(waves):
        return float(table.loc[waves == wave, "X"].iloc[0])
    lo, hi = wave - 1, wave + 1
    if lo in set(waves) and hi in set(waves):
        return float(table.loc[waves == lo, "X"].iloc[0] + table.loc[waves == hi, "X"].iloc[0]) / 2.0
    raise ValueError(f"No pathlength for wavelength {wave}")


def _convert_reflectance_to_hb_base(dA1: xr.DataArray, dA2: xr.DataArray, wave1: int, wave2: int, config: dict[str, Any]) -> xr.Dataset:
    """Solve the two-wavelength hemoglobin inversion using configured optical tables.
    
    Parameters
    ----------
    dA1 : xarray.DataArray
        Natural-log attenuation for the first reflectance wavelength (time, y, x).
    dA2 : xarray.DataArray
        Natural-log attenuation for the second reflectance wavelength (time, y, x).
    wave1 : int
        First reflectance wavelength in nanometers.
    wave2 : int
        Second reflectance wavelength in nanometers.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Returns
    -------
    xarray.Dataset
        HbO, HHb, and HbT concentration-change maps (time, y, x).
    
    Notes
    -----
    The source tables come from ``signal_processing.hemoglobin.coeff_abs_path``
    and ``pathlength_path``. Extinction coefficients are converted from decadic
    to natural-log units by multiplication with ``ln(10)``; pathlengths are
    converted from mm to cm. A two-by-two system estimates HbO and HHb, and
    HbT is their sum. This base function performs the optical inversion without output scaling.
    ``convert_reflectance_to_hb`` calls it and applies configured hemoglobin
    scaling, including ``signal_processing.hemoglobin.scale_to_uM``."""
    hb_cfg = cfg_get(config, "signal_processing", "hemoglobin", default={}) or {}
    coeff = read_coeff_table(hb_cfg["coeff_abs_path"])
    pathlength = pd.read_csv(hb_cfg["pathlength_path"])

    def eps(wave: int, col: str) -> float:
        """Look up an extinction coefficient, interpolating between even wavelengths when needed."""
        wave = int(wave)
        waves = coeff["Wave"].astype(int)
        if wave % 2 == 0:
            rows = coeff.loc[waves == wave, col]
            if rows.empty:
                raise ValueError(f"No extinction coefficient for wavelength {wave}")
            value = float(rows.iloc[0])
        else:
            # Odd wavelengths: interpolate between the nearest even-numbered
            # source rows. Ignore any manually added odd-numbered table row.
            even = coeff.loc[waves % 2 == 0].sort_values("Wave")
            lower = even[even["Wave"] < wave].tail(1)
            upper = even[even["Wave"] > wave].head(1)
            if lower.empty or upper.empty:
                raise ValueError(f"Cannot interpolate extinction coefficient at {wave} nm")
            x0, y0 = float(lower["Wave"].iloc[0]), float(lower[col].iloc[0])
            x1, y1 = float(upper["Wave"].iloc[0]), float(upper[col].iloc[0])
            value = y0 + (y1 - y0) * (wave - x0) / (x1 - x0)
        return value * np.log(10.0)

    X1 = lookup_pathlength(wave1, pathlength) * 0.1
    X2 = lookup_pathlength(wave2, pathlength) * 0.1
    dmu1 = dA1 / X1
    dmu2 = dA2 / X2
    epsO1, epsR1 = eps(wave1, "HbO"), eps(wave1, "HbR")
    epsO2, epsR2 = eps(wave2, "HbO"), eps(wave2, "HbR")

    denom_r = epsO1 * epsR2 - epsO2 * epsR1
    denom_o = epsR1 * epsO2 - epsR2 * epsO1
    dc_hhb = (epsO1 * dmu2 - epsO2 * dmu1) / denom_r
    dc_hbo = (epsR1 * dmu2 - epsR2 * dmu1) / denom_o
    dc_hbt = dc_hbo + dc_hhb
    names = hb_cfg.get("output_names", {}) or {}
    hb_ds = xr.Dataset({
        names.get("HbO", "dcHbO"): dc_hbo.astype("float32"),
        names.get("HHb", "dcHHb"): dc_hhb.astype("float32"),
        names.get("HbT", "dcHbT"): dc_hbt.astype("float32"),
    })
    # Record the actual coefficients used above (including the 505 nm
    # interpolation) in the processed Zarr metadata, not in a separate CSV.
    pathlength_waves = set(pathlength["Wavelength"].astype(int))
    used = {
        "extinction_table": str(hb_cfg["coeff_abs_path"]),
        "pathlength_table": str(hb_cfg["pathlength_path"]),
        "extinction_table_units": "cm^-1 M^-1 (decadic)",
        "pathlength_table_units": "mm",
        "inversion_conversion": "extinction * ln(10); pathlength_mm * 0.1 = pathlength_cm",
        "wavelengths": [
            {
                "wavelength_nm": int(wave),
                "HbO_table": round(float(eps_o / np.log(10.0)), 10),
                "HbR_table": round(float(eps_r / np.log(10.0)), 10),
                "pathlength_mm": float(x_cm / 0.1),
                "HbO_inversion": float(eps_o),
                "HbR_inversion": float(eps_r),
                "pathlength_cm": float(x_cm),
                "extinction_method": "linear_interpolation" if int(wave) % 2 else "direct",
                "pathlength_method": "direct" if int(wave) in pathlength_waves else "linear_interpolation",
            }
            for wave, eps_o, eps_r, x_cm in (
                (wave1, epsO1, epsR1, X1),
                (wave2, epsO2, epsR2, X2),
            )
        ],
    }
    hb_ds.attrs["hb_coefficients_used_json"] = json.dumps(used, ensure_ascii=False)
    return hb_ds


def _normalise_wavelength_label(value: Any) -> str:
    """Return a stable text label for wavelength coordinates.

    xarray/zarr may round-trip wavelength coordinates as strings, ints, or floats
    (for example 530, "530", or 530.0).  For channel selection these should be
    treated as the same wavelength.
    """
    try:
        f = float(value)
        if np.isfinite(f) and abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return str(f)
    except Exception:
        text = str(value).strip()
        try:
            f = float(text)
            if np.isfinite(f) and abs(f - round(f)) < 1e-9:
                return str(int(round(f)))
        except Exception:
            pass
        return text


def select_wave(raw: xr.DataArray, wavelength: int | float | str) -> xr.DataArray:
    """Select a wavelength channel after normalizing its coordinate representation."""
    wanted = _normalise_wavelength_label(wavelength)
    values = list(raw.coords["wavelength"].values)
    labels = [_normalise_wavelength_label(v) for v in values]

    if wanted not in labels:
        raise KeyError(
            f"Wavelength {wanted} not found in raw zarr. "
            f"Available raw labels: {labels}; raw coordinate values: {[str(v) for v in values]}"
        )

    idx = labels.index(wanted)
    # Use isel rather than sel because the stored coordinate may be 530.0 while
    # the requested value is 530 or "530".
    return raw.isel(wavelength=idx).astype("float32")


# ============================================================
# Regression helpers
# ============================================================


def residual_regression(
    y: np.ndarray,
    *regs: np.ndarray,
    include_intercept: bool = True,
    subtract_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit temporal regressors by least squares and remove their fitted component.
    
    Parameters
    ----------
    y : numpy.ndarray
        One-dimensional target time series.
    *regs : numpy.ndarray
        One-dimensional regressors with the same number of time samples as `y`.
    include_intercept : bool
        Add a constant column to the fitted design matrix.
    subtract_intercept : bool
        Subtract the fitted intercept as well as regressor-dependent components.
    
    Returns
    -------
    residual : numpy.ndarray
        Corrected one-dimensional series, with invalid samples marked NaN.
    beta : numpy.ndarray
        Fitted coefficients, including the intercept when requested.
    r2 : float
        Coefficient of determination for the full fitted model.
    
    Notes
    -----
    The fit uses samples finite in the target and every regressor, and requires
    at least `max(5, n_parameters + 1)` valid samples. If `include_intercept`
    is true and `subtract_intercept` false, the intercept is retained while
    regressor-dependent components are removed. Failed fits return NaNs; the
    reported R2 always refers to the complete fitted model."""
    y = np.asarray(y, dtype=np.float64)
    R = [np.asarray(r, dtype=np.float64) for r in regs]

    n_regs = len(R)
    n_beta = n_regs + int(include_intercept)
    beta_nan = np.full((n_beta,), np.nan, dtype=np.float32)

    if y.ndim != 1 or n_regs == 0:
        return np.full_like(y, np.nan, dtype=np.float32), beta_nan, np.nan

    ok = np.isfinite(y)
    for r in R:
        if r.ndim != 1 or r.shape[0] != y.shape[0]:
            return np.full_like(y, np.nan, dtype=np.float32), beta_nan, np.nan
        ok &= np.isfinite(r)

    # Need more valid time points than fitted coefficients; keep a small
    # conservative lower bound to avoid unstable fits on tiny traces.
    if ok.sum() < max(5, n_beta + 1):
        return np.full_like(y, np.nan, dtype=np.float32), beta_nan, np.nan

    cols = []
    if include_intercept:
        cols.append(np.ones(ok.sum(), dtype=np.float64))
    cols.extend([r[ok] for r in R])
    X = np.column_stack(cols)

    try:
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    except Exception:
        return np.full_like(y, np.nan, dtype=np.float32), beta_nan, np.nan

    cols_all = []
    if include_intercept:
        cols_all.append(np.ones(y.size, dtype=np.float64))
    cols_all.extend(R)
    X_all = np.column_stack(cols_all)

    pred_full = X_all @ beta

    if include_intercept and not subtract_intercept:
        # Preserve beta0 in the corrected signal. Remove only the component
        # explained by the regressors.
        pred_to_subtract = X_all[:, 1:] @ beta[1:]
    else:
        # Either the intercept should be removed, or there is no intercept in
        # the model. In both cases subtract the full prediction.
        pred_to_subtract = pred_full

    resid = y - pred_to_subtract
    resid[~ok] = np.nan

    # R2 is still computed for the full fitted model, including intercept when
    # present, because it describes model fit rather than the correction mode.
    ss_res = np.nansum((y[ok] - (X @ beta)) ** 2)
    ss_tot = np.nansum((y[ok] - np.nanmean(y[ok])) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + np.finfo(float).eps)

    beta_out = beta_nan.copy()
    beta_out[: len(beta)] = beta.astype(np.float32)
    return resid.astype(np.float32), beta_out, float(r2)


def _rechunk_time_for_regression(arr: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Make `time` a single chunk before pixel-wise regression.

    xarray.apply_ufunc(..., dask="parallelized") requires every core
    dimension to be contained in one chunk. Here the core dimension is time.
    Spatial dimensions stay chunked according to signal_processing.output.regression_chunks
    when available, so memory use is controlled by y/x tile size.
    """
    if "time" not in arr.dims:
        return arr

    reg_chunks = cfg_get(config, "signal_processing", "output", "regression_chunks", default=[-1, 32, 32])
    try:
        # Expected order in config: [time, y, x]. Force time=-1 for regression.
        y_chunk = int(reg_chunks[1]) if len(reg_chunks) > 1 and int(reg_chunks[1]) > 0 else 32
        x_chunk = int(reg_chunks[2]) if len(reg_chunks) > 2 and int(reg_chunks[2]) > 0 else 32
    except Exception:
        y_chunk, x_chunk = 32, 32

    chunks = {"time": -1}
    if "y" in arr.dims:
        chunks["y"] = y_chunk
    if "x" in arr.dims:
        chunks["x"] = x_chunk
    return arr.chunk(chunks)


def regress_fluorescence_from_regressors(fluo: xr.DataArray, regressors: list[xr.DataArray], config: dict[str, Any]) -> xr.Dataset:
    """Regress configured hemodynamic components from a fluorescence time series.
    
    Parameters
    ----------
    fluo : object
        See the corresponding Step4 call site and YAML settings.
    regressors : object
        See the corresponding Step4 call site and YAML settings.
    config : dict
        Parsed YAML configuration used by this Step4 operation."""
    corr_cfg = cfg_get(config, "signal_processing", "fluorescence", "correction", default={}) or {}
    include_intercept = bool(corr_cfg.get("include_intercept", True))
    subtract_intercept = bool(corr_cfg.get("subtract_intercept", False))
    corrected_name = corr_cfg.get("corrected_name", "fluo_hemo_regressed_percent")
    beta_prefix = corr_cfg.get("beta_prefix", "fluo_regression_beta")
    r2_name = corr_cfg.get("r2_name", "fluo_regression_r2")

    # Generic but with fixed support for 1-3 regressors for simpler xarray outputs.
    n_regs = len(regressors)
    if n_regs not in {1, 2, 3}:
        raise ValueError("pixelwise_hemo_regression currently supports 1-3 regressors")

    # Regression is solved along the full time axis for every pixel.
    # Therefore time must be a single dask chunk; y/x remain tiled.
    fluo = _rechunk_time_for_regression(fluo, config)
    regressors = [_rechunk_time_for_regression(r, config) for r in regressors]

    def wrapped(y, *rs):
        """Apply the current regression operation to an array block."""
        resid, beta, r2 = residual_regression(
            y,
            *rs,
            include_intercept=include_intercept,
            subtract_intercept=subtract_intercept,
        )
        max_beta = 1 + n_regs if include_intercept else n_regs
        beta_padded = np.full((max_beta,), np.nan, dtype=np.float32)
        beta_padded[: len(beta)] = beta
        return resid, beta_padded, np.float32(r2)

    residual, beta, r2 = xr.apply_ufunc(
        wrapped,
        fluo,
        *regressors,
        input_core_dims=[["time"], *([["time"]] * n_regs)],
        output_core_dims=[["time"], ["beta"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32", "float32", "float32"],
        dask_gufunc_kwargs={"output_sizes": {"beta": (1 + n_regs if include_intercept else n_regs)}},
    )
    residual = residual.transpose("time", "y", "x")
    beta_labels = (["intercept"] if include_intercept else []) + [r.name or f"reg{i+1}" for i, r in enumerate(regressors)]
    ds = xr.Dataset({corrected_name: residual.astype("float32"), r2_name: r2.astype("float32")})
    for i, label in enumerate(beta_labels):
        ds[f"{beta_prefix}_{label}"] = beta.isel(beta=i).astype("float32")
    return ds


def global_signal_regression(
    data: xr.DataArray,
    roi: xr.DataArray,
    *,
    include_intercept: bool = True,
    subtract_intercept: bool = False,
    config: dict[str, Any] | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Regress the ROI-average temporal component from each spatial signal.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    roi : xarray.DataArray
        Spatial Boolean mask with dimensions (y, x).
    include_intercept : object
        See the corresponding Step4 call site and YAML settings.
    subtract_intercept : object
        See the corresponding Step4 call site and YAML settings.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Notes
    -----
    The spatially averaged ROI trace is used as a time-varying
    regressor. The returned arrays include corrected data and regression
    statistics. Configuration in ``signal_processing.global_signal`` selects
    which variants use this operation."""
    global_trace = mean_trace(data, roi).astype("float32")
    if config is not None:
        data = _rechunk_time_for_regression(data, config)
        global_trace = global_trace.chunk({"time": -1})

    def wrapped(y, r):
        """Apply the current regression operation to an array block."""
        resid, beta, r2 = residual_regression(
            y,
            r,
            include_intercept=include_intercept,
            subtract_intercept=subtract_intercept,
        )
        max_beta = 2 if include_intercept else 1
        beta_padded = np.full((max_beta,), np.nan, dtype=np.float32)
        beta_padded[: len(beta)] = beta
        return resid, beta_padded, np.float32(r2)

    residual, beta, r2 = xr.apply_ufunc(
        wrapped,
        data,
        global_trace,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"], ["beta"], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32", "float32", "float32"],
        dask_gufunc_kwargs={"output_sizes": {"beta": (2 if include_intercept else 1)}},
    )
    residual = residual.transpose("time", "y", "x")
    slope_index = 1 if include_intercept else 0
    return residual.astype("float32"), global_trace.astype("float32"), beta.isel(beta=slope_index).astype("float32"), r2.astype("float32")


def _as_string_list(value: Any) -> list[str]:
    """Convert an optional configuration value to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(_as_string_list(v))
        return out
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]

def _flatten_apply_to(value: Any) -> list[str]:
    """Flatten configured target signal names to one string list."""
    return list(dict.fromkeys(_as_string_list(value)))

def _row_matches_rule(row: pd.Series, match: dict[str, Any] | None) -> bool:
    """Test whether recording metadata match a configured variant-selection rule."""
    for key, wanted in (match or {}).items():
        key = "protocol_base" if key == "protocol" and pd.notna(row.get("protocol_base")) else str(key)
        allowed = set(_as_string_list(wanted))
        if key == "protocol_repeat" and allowed:
            if not pd.notna(row.get(key)) or float(row.get(key)) not in {float(v) for v in allowed}:
                return False
        elif allowed and str(row.get(key, "")) not in allowed:
            return False
    return True

def resolve_global_signal_variants(row: pd.Series, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve active GSR variants for one recording row.

    New config shape:
      signal_processing.global_signal.variants: list of variant dicts
      signal_processing.global_signal.default_variants: list of variant names
      signal_processing.global_signal.rules: list of metadata match rules

    Rules are evaluated in order; the last matching rule wins.
    """
    gs_cfg = cfg_get(config, "signal_processing", "global_signal", default={}) or {}
    raw_variants = gs_cfg.get("variants") or []
    variants_by_name: dict[str, dict[str, Any]] = {}
    for item in raw_variants:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Every global_signal variant must have a non-empty 'name'.")
        variants_by_name[name] = dict(item)

    selected_names = _as_string_list(gs_cfg.get("default_variants", []))
    for rule in gs_cfg.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        if _row_matches_rule(row, rule.get("match") or {}):
            selected_names = _as_string_list(rule.get("variants", selected_names))

    selected: list[dict[str, Any]] = []
    for name in selected_names:
        if name not in variants_by_name:
            raise KeyError(f"global_signal rule selected unknown variant: {name!r}")
        variant = dict(variants_by_name[name])
        if bool(variant.get("enabled", True)):
            selected.append(variant)
    return selected

def _gsr_suffix(variant: dict[str, Any]) -> str:
    """Determine the output suffix for a global-signal correction variant."""
    suffix = str(variant.get("suffix") or variant.get("name") or "gsr").strip()
    if not suffix:
        raise ValueError(f"Global signal variant has empty suffix/name: {variant}")
    return suffix

def global_signal_subtract_mean(data: xr.DataArray, roi: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Subtract the ROI-average time course from the input spatial signal.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    roi : xarray.DataArray
        Spatial Boolean mask with dimensions (y, x).
    
    Notes
    -----
    The ROI-average time course is subtracted directly (no regression)
    from the spatial signal."""
    trace = mean_trace(data.astype("float32"), roi).astype("float32")
    corrected = (data.astype("float32") - trace).astype("float32")
    return corrected, trace

def apply_global_signal_variants(
    ds_out: xr.Dataset,
    masks: dict[str, xr.DataArray],
    variants: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """Apply configured global-signal corrections and add new dataset variables.
    
    Parameters
    ----------
    ds_out : xarray.Dataset
        Input signal dataset; this function adds new variables in place.
    masks : dict
        Spatial ROIs used to derive correction time courses.
    variants : list of dict
        Active variants selected using `signal_processing.global_signal`.
    config : dict
        Parsed project configuration, including ROI-size requirements.
    
    Returns
    -------
    created_names : list of str
        Names of the newly created corrected spatial signal variables.
    
    Notes
    -----
    Supported variants subtract a mean ROI time course directly or regress
    it from each target signal. Depending on per-variant options, the function
    also saves temporal regressors, beta maps, and R2 maps. Original source
    signals remain available in the output dataset."""
    made: list[str] = []
    default_min = int(cfg_get(config, "signal_processing", "roi", "min_signal_pixels", default=100))

    for variant in variants:
        method = str(variant.get("method", "none")).strip().lower()
        if method in {"none", "no_gsr", "off", "disabled"}:
            continue

        suffix = _gsr_suffix(variant)
        roi_name = variant.get("roi", "brainvessels_signal")
        min_pixels = int(variant.get("min_pixels", default_min))
        roi = make_named_roi(masks, roi_name, min_pixels=min_pixels, label=f"global_signal[{suffix}] ROI")
        apply_to = _flatten_apply_to(variant.get("apply_to", []))
        save_trace = bool(variant.get("save_trace", True))
        save_beta = bool(variant.get("save_beta_map", False))
        save_r2 = bool(variant.get("save_r2_map", False))

        for name in apply_to:
            if name not in ds_out:
                print(f"[WARN] global_signal variant={suffix}: variable not found, skipping: {name}")
                continue
            data = ds_out[name]
            if set(data.dims) != {"time", "y", "x"}:
                print(f"[WARN] global_signal variant={suffix}: variable is not time/y/x, skipping: {name}")
                continue

            out_name = f"{name}_{suffix}"
            if method in {"subtract_mean", "mean_subtraction", "brain_mean_subtract"}:
                corrected, trace = global_signal_subtract_mean(data, roi)
                ds_out[out_name] = corrected.astype("float32")
                if save_trace:
                    ds_out[f"{out_name}_trace"] = trace.astype("float32")
                made.append(out_name)
                continue

            if method in {"roi_mean_regression", "mean_regression", "vessel_regression", "brain_mean_regression"}:
                corrected, trace, beta, r2 = global_signal_regression(
                    data,
                    roi,
                    include_intercept=bool(variant.get("include_intercept", True)),
                    subtract_intercept=bool(variant.get("subtract_intercept", False)),
                    config=config,
                )
                ds_out[out_name] = corrected.astype("float32")
                if save_trace:
                    ds_out[f"{out_name}_trace"] = trace.astype("float32")
                if save_beta:
                    ds_out[f"{out_name}_beta"] = beta.astype("float32")
                if save_r2:
                    ds_out[f"{out_name}_r2"] = r2.astype("float32")
                made.append(out_name)
                continue

            raise ValueError(
                f"Unsupported global_signal variant method={method!r}. "
                "Use 'none', 'subtract_mean', or 'roi_mean_regression'."
            )
    return made

def fluorescence_global_signal_variant_names(corr_cfg: dict[str, Any], active_variants: list[dict[str, Any]]) -> list[str]:
    """Select applicable global-signal variants for fluorescence correction."""
    raw = corr_cfg.get("global_signal_variants", [])
    if raw in (None, False):
        return []
    if raw is True or str(raw).strip().lower() == "auto":
        return [str(v.get("name")) for v in active_variants if bool(v.get("apply_to_fluorescence_correction", False))]
    return _as_string_list(raw)

def global_signal_trace_for_fluorescence(fluo_dff: xr.DataArray, masks: dict[str, xr.DataArray], variant: dict[str, Any], config: dict[str, Any]) -> xr.DataArray:
    """Construct the configured ROI-average regressor for fluorescence."""
    suffix = _gsr_suffix(variant)
    default_min = int(cfg_get(config, "signal_processing", "roi", "min_signal_pixels", default=100))
    roi = make_named_roi(
        masks,
        variant.get("roi", "brainvessels_signal"),
        min_pixels=int(variant.get("min_pixels", default_min)),
        label=f"fluorescence global_signal[{suffix}] ROI",
    )
    trace = mean_trace(fluo_dff.astype("float32"), roi).astype("float32")
    trace.name = f"global_{suffix}"
    return trace

def smooth_signal_yx(data: xr.DataArray, config: dict[str, Any]) -> xr.DataArray:
    """Apply configuration-controlled spatial or temporal Gaussian smoothing.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input spatial signal with a time axis.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Notes
    -----
    The enabled setting and Gaussian sigma values are read from
    ``signal_processing.smoothing``. Spatial and temporal smoothing are kept
    in the configured axis order; units are inherited from the input signal."""
    sm_cfg = cfg_get(config, "signal_processing", "smoothing", default={}) or {}
    if not bool(sm_cfg.get("enabled", False)):
        return data
    sigma_yx = float(sm_cfg.get("sigma_yx", 1))
    sigma_time = float(sm_cfg.get("sigma_time", 0))
    if sigma_yx <= 0 and sigma_time <= 0:
        return data
    sigma = (sigma_time, sigma_yx, sigma_yx)
    depth = {0: int(math.ceil(3 * sigma_time)), 1: int(math.ceil(3 * sigma_yx)), 2: int(math.ceil(3 * sigma_yx))}
    return xr.apply_ufunc(
        lambda x: da.map_overlap(lambda b: gaussian_filter(b, sigma=sigma), x, depth=depth, boundary="reflect"),
        data,
        dask="allowed",
    ).astype("float32")


# ============================================================
# Main processing
# ============================================================



def rechunk_dataset_for_zarr(ds: xr.Dataset, chunks_3d: Sequence[int] | None = None) -> xr.Dataset:
    """Rechunk dask-backed variables so xarray.to_zarr safe_chunks validation passes.

    xarray requires that the requested zarr encoding chunks do not overlap
    multiple dask chunks. After raw loading, many variables have time chunks
    inherited from raw_zarr (for example 64), while output encoding may request
    larger chunks (for example 256). Pixelwise regression can also temporarily
    create time=-1, y=32, x=32 chunks. Before writing, make dask chunks match
    the intended output chunks for 3D time/y/x arrays.
    """
    if chunks_3d is None:
        chunks_3d = [256, 64, 64]

    out = ds.copy()
    for name, var in list(out.data_vars.items()):
        if not hasattr(var.data, "chunks"):
            continue

        chunk_map: dict[str, int] = {}
        if "time" in var.dims and "y" in var.dims and "x" in var.dims:
            # Config order is [time, y, x], but variable dims are not assumed.
            base = {"time": int(chunks_3d[0]), "y": int(chunks_3d[1]), "x": int(chunks_3d[2])}
            for d in var.dims:
                if d in base:
                    c = base[d]
                    if c <= 0:
                        c = int(var.sizes[d])
                    chunk_map[d] = min(c, int(var.sizes[d]))
        elif "time" in var.dims and var.ndim == 1:
            # Small traces are cheap; one chunk makes them easy to read.
            chunk_map["time"] = int(var.sizes["time"])
        elif "y" in var.dims and "x" in var.dims and var.ndim == 2:
            # Maps such as beta/r2.
            y_chunk = int(chunks_3d[1]) if len(chunks_3d) > 1 and int(chunks_3d[1]) > 0 else int(var.sizes["y"])
            x_chunk = int(chunks_3d[2]) if len(chunks_3d) > 2 and int(chunks_3d[2]) > 0 else int(var.sizes["x"])
            chunk_map = {"y": min(y_chunk, int(var.sizes["y"])), "x": min(x_chunk, int(var.sizes["x"]))}

        if chunk_map:
            out[name] = var.chunk(chunk_map)
    return out



# ============================================================
# Step-level debug / progress report helpers
# ============================================================


class Step4ProcessReporter:
    """Small helper for notebook-friendly Step4 processing diagnostics.

    It prints coarse status messages, drives one progress bar over processing
    stages, and saves a per-stem report table.  It is intentionally lightweight:
    it does not compute large arrays just for reporting.
    """

    def __init__(
        self,
        row: pd.Series,
        config: dict[str, Any],
        *,
        steps: Sequence[str],
        show_progress: bool = True,
        debug: bool = False,
        save_report: bool = True,
    ) -> None:
        """Initialize the progress reporter for one recording."""
        self.row = row
        self.config = config
        self.steps = list(steps)
        self.show_progress = bool(show_progress)
        self.debug = bool(debug)
        self.save_report = bool(save_report)
        self.records: list[dict[str, Any]] = []
        self._active_name: str | None = None
        self._active_t0: _dt.datetime | None = None
        self._pbar = tqdm(
            total=len(self.steps),
            desc=f"Step4 process [{row.get('stem', '')}]",
            unit="step",
            disable=not self.show_progress,
        )

    def _now(self) -> str:
        """Return the current timestamp for the processing log."""
        return _dt.datetime.now().isoformat(timespec="seconds")

    def log(self, message: str) -> None:
        """Append one progress message to the active processing report."""
        if self.debug or self.show_progress:
            print(message)

    def begin(self, name: str, message: str | None = None) -> None:
        """Record the beginning of a processing substep."""
        if self._active_name is not None:
            # Auto-close a forgotten active step as ok; this prevents broken reports.
            self.end(status="ok")
        self._active_name = str(name)
        self._active_t0 = _dt.datetime.now()
        if message is None:
            message = f"[PROCESS] {name}"
        self.log(message)
        if self._pbar is not None:
            self._pbar.set_postfix_str(str(name)[:60])

    def end(self, *, status: str = "ok", details: str = "") -> None:
        """Finalize the active processing substep with a status and details."""
        if self._active_name is None:
            return
        t1 = _dt.datetime.now()
        t0 = self._active_t0 or t1
        rec = {
            "stem": str(self.row.get("stem", "")),
            "group_id": str(self.row.get("group_id", "")),
            "step": self._active_name,
            "status": status,
            "details": details,
            "started_at": t0.isoformat(timespec="seconds"),
            "finished_at": t1.isoformat(timespec="seconds"),
            "duration_sec": round((t1 - t0).total_seconds(), 3),
        }
        self.records.append(rec)
        if self._pbar is not None:
            self._pbar.update(1)
        if self.debug or status not in {"ok", "skipped"}:
            tag = "OK" if status == "ok" else status.upper()
            msg = f"[{tag}] {self._active_name}"
            if details:
                msg += f": {details}"
            print(msg)
        self._active_name = None
        self._active_t0 = None

    def fail(self, error: BaseException | str) -> None:
        """Mark the active processing substep as failed."""
        self.end(status="failed", details=str(error))

    def close(self) -> None:
        """Close the progress display and finalize any active step."""
        if self._active_name is not None:
            self.end(status="ok")
        if self._pbar is not None:
            self._pbar.close()

    def report_dir(self) -> Path:
        """Return the quality-control output directory for the current recording."""
        return get_qc_dir(self.row, self.config)

    def save(self, *, filename_prefix: str = "process_step_report") -> tuple[Path, Path] | None:
        """Save the processing-step report to CSV and JSON, if enabled."""
        if not self.save_report:
            return None
        out_dir = self.report_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        csv_path = out_dir / f"{filename_prefix}.csv"
        json_path = out_dir / f"{filename_prefix}.json"
        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, ensure_ascii=False, indent=2)
        return csv_path, json_path


def _step4_debug_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Step4 processing-debug settings."""
    step4_cfg = require_step4(config)
    return (step4_cfg.get("process_debug", {}) or {})


def print_step4_dataset_summary(ds: xr.Dataset, *, title: str = "Dataset summary") -> None:
    """Print a compact variable/chunk summary for debugging xarray/dask/zarr."""
    print(f"\n{title}")
    print("=" * len(title))
    for name, var in ds.data_vars.items():
        chunks = getattr(var.data, "chunks", None)
        print(f"{name}: dims={var.dims}, shape={tuple(var.shape)}, dtype={var.dtype}, chunks={chunks}")


def print_step4_array_summary(arr: xr.DataArray, *, title: str = "DataArray summary") -> None:
    """Print array dimensions, shape, data type, and chunk layout."""
    chunks = getattr(arr.data, "chunks", None)
    print(f"\n{title}")
    print("=" * len(title))
    print(f"name={arr.name}, dims={arr.dims}, shape={tuple(arr.shape)}, dtype={arr.dtype}, chunks={chunks}")


def process_raw_zarr_for_row(
    row: pd.Series,
    config: dict[str, Any],
    *,
    overwrite: bool | None = None,
    save_qc: bool = True,
    show_progress: bool | None = None,
    debug: bool | None = None,
    save_step_report: bool | None = None,
) -> Path:
    """Process one raw recording into configured hemoglobin and fluorescence signals.
    
    Parameters
    ----------
    row : pandas.Series
        Recording metadata and resolved input/output paths.
    config : dict
        Parsed YAML settings, primarily `step4` and `signal_processing`.
    overwrite : bool, optional
        Override `step4.overwrite_processed`.
    save_qc : bool
        Request quality-control handling, subject to the configured save flags.
    show_progress : bool, optional
        Override the processing progress setting.
    debug : bool, optional
        Override diagnostic output.
    save_step_report : bool, optional
        Override saving a per-recording step report.
    
    Returns
    -------
    processed_path : pathlib.Path
        Path of the completed or reusable processed Zarr store.
    
    Notes
    -----
    The implementation checks raw-store completion and fingerprints; loads
    raw-validity and canonical masks; performs configured drift and baseline
    operations; converts selected reflectance channels to hemoglobin signals;
    optionally processes fluorescence, regression variants, and smoothing;
    selects requested outputs and writes Zarr with completion metadata.
    The sequence of these operations is part of the verified scientific code.
    The hemoglobin conversion uses the same optional scaling wrapper for
    imported modules and direct CLI execution. Raw-store validation uses
    the saved raw configuration if
    ``step4.reuse_existing_raw_for_processing`` is true; otherwise it
    checks the current raw-input fingerprint."""
    step4_cfg = require_step4(config)
    sp_cfg = config.get("signal_processing", {}) or {}
    out_cfg = sp_cfg.get("output", {}) or {}
    dbg_cfg = _step4_debug_config(config)

    if overwrite is None:
        overwrite = bool(step4_cfg.get("overwrite_processed", False))
    if show_progress is None:
        show_progress = bool(dbg_cfg.get("show_progress", True))
    if debug is None:
        debug = bool(dbg_cfg.get("enabled", False))
    if save_step_report is None:
        save_step_report = bool(dbg_cfg.get("save_step_report", True))

    steps = [
        "check output paths",
        "open raw zarr",
        "load masks and ROI",
        "background correction",
        "hemoglobin conversion",
        "fluorescence processing",
        "global signal regression",
        "smoothing",
        "select outputs and rechunk",
        "write processed zarr",
        "save QC",
        "cleanup",
    ]
    rep = Step4ProcessReporter(
        row,
        config,
        steps=steps,
        show_progress=bool(show_progress),
        debug=bool(debug),
        save_report=bool(save_step_report),
    )

    raw = raw_corr = ds_out = ds_save = masks = signal_roi = None  # for cleanup/report robustness
    out_path: Path | None = None
    raw_path: Path | None = None

    try:
        # ------------------------------------------------------------
        # 1. Paths / overwrite
        # ------------------------------------------------------------
        rep.begin("check output paths", f"[LOAD] Preparing Step4 processing for stem={row.get('stem', '')}")
        raw_variant = selected_raw_variant_for_row(row, config)
        raw_path = select_raw_zarr_path_for_processing(row, config)
        out_path = select_processed_zarr_path_for_processing(row, config)
        if not raw_path.exists():
            raise FileNotFoundError(
                f"raw_zarr not found for signal processing: {raw_path} "
                f"(raw_variant={raw_variant!r}). "
                "If you selected raw_variant='motion_corrected', run Step4 raw build with "
                "raw_motion_correction.enabled=true first, or switch signal_processing.input.raw_variant to 'unaligned'."
            )
        if not is_complete_zarr(raw_path, config, raw=True):
            raise RuntimeError(f"Raw Zarr is incomplete: {raw_path}")
        raw_root = zarr_open_group_v2_compat(raw_path, mode="r")
        if bool(cfg_get(config, "step4", "reuse_existing_raw_for_processing", default=False)):
            # Explicitly requested processing of the stored raw pixels. The
            # current BMP files' mtimes/paths may differ without affecting
            # what is already saved inside this complete raw Zarr.
            assert_existing_raw_compatible_for_processing(raw_root, config, raw_path)
            print(f"[REUSE EXISTING RAW] {row['stem']}: raw Zarr complete and raw settings compatible; "
                  "not checking current BMP paths/size/mtime. Results use SAVED raw pixels.", flush=True)
        else:
            stored_fp = raw_root.attrs.get("raw_fingerprint", "")
            if stored_fp != raw_fingerprint(row, config):
                raise RuntimeError(f"Raw Zarr is stale for stem={row['stem']}; rebuild raw with overwrite_raw=True")
        current_fp = processed_fingerprint(row, config, raw_path)
        if out_path.exists() and not overwrite:
            if not is_complete_zarr(out_path, config, raw=False, fingerprint=current_fp):
                raise RuntimeError(f"Processed Zarr incomplete or its inputs/config changed: {out_path}; use overwrite_processed=True")
            rep.end(status="skipped", details=f"processed zarr complete and up-to-date: {out_path}")
            for step_name in steps[1:]:
                rep.begin(step_name, f"[SKIP] {step_name}")
                rep.end(status="skipped")
            rep.save()
            rep.close()
            return out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rep.end(details=f"raw_variant={raw_variant}; raw={raw_path}; out={out_path}")

        # ------------------------------------------------------------
        # 2. Open raw zarr
        # ------------------------------------------------------------
        rep.begin("open raw zarr", "[LOAD] Opening raw zarr")
        raw = open_raw_zarr(raw_path, config)
        if debug:
            print_step4_array_summary(raw, title="Raw zarr after open_raw_zarr")
        image_shape = (raw.sizes["y"], raw.sizes["x"])
        rep.end(details=f"shape={tuple(raw.shape)}, chunks={getattr(raw.data, 'chunks', None)}")

        # ------------------------------------------------------------
        # 3. Masks and ROI
        # ------------------------------------------------------------
        rep.begin("load masks and ROI", "[LOAD] Loading masks and building ROI")
        raw_valid = load_raw_valid(raw_path, image_shape=image_shape, config=config)
        masks = load_masks_for_row(row, config, image_shape=image_shape, raw_valid=raw_valid)
        signal_roi = make_signal_roi(masks, config)
        drift_enabled = bool(cfg_get(config, "signal_processing", "drift_correction", "enabled", default=True))
        background_roi = make_background_roi(masks, config) if drift_enabled else None
        signal_pixels = int(signal_roi.sum().compute().item())
        background_pixels = int(background_roi.sum().compute().item()) if background_roi is not None else None
        rep.end(details=f"signal_pixels={signal_pixels}; background_pixels={background_pixels}; drift_enabled={drift_enabled}")

        # ------------------------------------------------------------
        # 4. Preprocessing: drift/background correction and baseline order
        # ------------------------------------------------------------
        rep.begin("background correction", "[PROCESS] Preprocessing order and background drift correction")
        raw_corr, bg_trace, preprocessing_mode = prepare_intensity_for_processing(raw.where(raw_valid), background_roi, config)
        active_gs_variants = resolve_global_signal_variants(row, config)
        rep.end(
            details=(
                f"preprocessing_mode={preprocessing_mode}; "
                f"active_gsr_variants={[v.get('name') for v in active_gs_variants]}"
            )
        )

        ds_out = xr.Dataset(attrs={
            "stem": str(row["stem"]),
            "group_id": str(row.get("group_id", "")),
            "ref_stem": str(row.get("ref_stem", "")),
            "source_raw_zarr": str(raw_path),
            "created_at": _dt.datetime.now().isoformat(),
            "preprocessing_mode": preprocessing_mode,
            "smoothing_enabled": bool(cfg_get(config, "signal_processing", "smoothing", "enabled", default=False)),
        })
        if bg_trace is not None:
            ds_out["background_trace"] = bg_trace.astype("float32")
        # Keep 0/1 values distinct from missing data in default xarray.open_zarr().
        ds_out["raw_valid"] = raw_valid.astype("uint8")
        ds_out["raw_valid"].encoding["_FillValue"] = 255
        ds_out.attrs["raw_valid_min_intensity"] = int(cfg_get(config, "signal_processing", "raw_valid", "min_intensity", default=1))
        ds_out.attrs["raw_valid_max_intensity"] = int(cfg_get(config, "signal_processing", "raw_valid", "max_intensity", default=250))
        ds_out.attrs["raw_valid_pixels"] = int(raw_valid.sum().item())
        ds_out.attrs["raw_valid_fraction"] = float(raw_valid.mean().item())
        ds_out.attrs["led_drift_correction_enabled"] = drift_enabled
        ds_out.attrs["baseline_method"] = str(cfg_get(config, "signal_processing", "baseline", "method", default="mean_all"))
        ds_out.attrs["signal_processing_config_json"] = json.dumps(sp_cfg, ensure_ascii=False, sort_keys=True, default=str)

        # ------------------------------------------------------------
        # 5. Hemoglobin
        # ------------------------------------------------------------
        rep.begin("hemoglobin conversion", "[PROCESS] Hemoglobin conversion")
        hb_names = cfg_get(config, "signal_processing", "hemoglobin", "output_names", default={}) or {}
        hb_ds: xr.Dataset | None = None
        if bool(cfg_get(config, "signal_processing", "hemoglobin", "enabled", default=True)):
            hb_cfg = cfg_get(config, "signal_processing", "hemoglobin", default={}) or {}
            refl_waves = [int(w) for w in hb_cfg.get("reflectance_wavelengths", [530, 656])]
            save_dA_single = bool(hb_cfg.get("save_dA_if_single_reflectance", True))
            save_dA_intermediate = bool(hb_cfg.get("save_intermediate_dA", False))

            if len(refl_waves) == 1:
                refl = channel_i_over_i0(raw_corr, refl_waves[0], config, preprocessing_mode)
                dA = reflectance_dA(refl)
                dA_name = hb_cfg.get("single_reflectance_dA_name") or f"dA_{refl_waves[0]}"
                if save_dA_single:
                    ds_out[dA_name] = dA.astype("float32")
                rep.end(details=f"single_reflectance_wavelength={refl_waves[0]}; outputs={[dA_name] if save_dA_single else []}")
            elif len(refl_waves) == 2:
                refl1 = channel_i_over_i0(raw_corr, refl_waves[0], config, preprocessing_mode)
                refl2 = channel_i_over_i0(raw_corr, refl_waves[1], config, preprocessing_mode)
                dA1 = reflectance_dA(refl1)
                dA2 = reflectance_dA(refl2)
                if save_dA_intermediate:
                    ds_out[f"dA_{refl_waves[0]}"] = dA1.astype("float32")
                    ds_out[f"dA_{refl_waves[1]}"] = dA2.astype("float32")
                hb_ds = convert_reflectance_to_hb(dA1, dA2, refl_waves[0], refl_waves[1], config)
                for name, var in hb_ds.data_vars.items():
                    ds_out[name] = var.astype("float32")
                ds_out.attrs["hb_coefficients_used_json"] = hb_ds.attrs["hb_coefficients_used_json"]
                rep.end(details=f"reflectance_wavelengths={refl_waves}; outputs={list(hb_ds.data_vars)}; save_dA_intermediate={save_dA_intermediate}")
            else:
                raise ValueError("signal_processing.hemoglobin.reflectance_wavelengths must contain one or two wavelengths")
        else:
            rep.end(status="skipped", details="hemoglobin.enabled=False")

        # ------------------------------------------------------------
        # 6. Fluorescence
        # ------------------------------------------------------------
        rep.begin("fluorescence processing", "[PROCESS] Fluorescence dF/F and hemodynamic regression")
        if bool(cfg_get(config, "signal_processing", "fluorescence", "enabled", default=True)):
            fluo_wave = cfg_get(config, "signal_processing", "fluorescence", "channel", default=470)
            fluo_dff = fluorescence_dff_from_preprocessed(raw_corr, fluo_wave, config, preprocessing_mode)
            uncorr_name = cfg_get(config, "signal_processing", "fluorescence", "uncorrected_name", default="fluo_dff_percent")
            added = []
            if bool(cfg_get(config, "signal_processing", "fluorescence", "save_uncorrected", default=True)):
                ds_out[uncorr_name] = fluo_dff.astype("float32")
                added.append(uncorr_name)

            corr_cfg = cfg_get(config, "signal_processing", "fluorescence", "correction", default={}) or {}
            if bool(corr_cfg.get("enabled", True)):
                regressor_names = [str(x) for x in corr_cfg.get("regressors", [hb_names.get("HbO", "dcHbO"), hb_names.get("HHb", "dcHHb")])]
                regressors = []
                for name in regressor_names:
                    if name not in ds_out:
                        raise KeyError(f"Fluorescence regression regressor not found: {name}")
                    r = ds_out[name]
                    r.name = name
                    regressors.append(r)

                reg_ds = regress_fluorescence_from_regressors(fluo_dff, regressors, config)
                for name, var in reg_ds.data_vars.items():
                    save_betas = bool(corr_cfg.get("save_betas", True))
                    save_r2 = bool(corr_cfg.get("save_r2", True))
                    if "beta" in name and not save_betas:
                        continue
                    if name == corr_cfg.get("r2_name", "fluo_regression_r2") and not save_r2:
                        continue
                    ds_out[name] = var.astype("float32")
                    added.append(name)

                variant_lookup = {str(v.get("name")): v for v in active_gs_variants}
                for variant_name in fluorescence_global_signal_variant_names(corr_cfg, active_gs_variants):
                    if variant_name not in variant_lookup:
                        raise KeyError(f"fluorescence.correction.global_signal_variants selected unknown active variant: {variant_name!r}")
                    variant = variant_lookup[variant_name]
                    method = str(variant.get("method", "none")).strip().lower()
                    if method in {"none", "no_gsr", "off", "disabled"}:
                        continue
                    suffix = _gsr_suffix(variant)
                    gtrace = global_signal_trace_for_fluorescence(fluo_dff, masks, variant, config)
                    variant_regressors = list(regressors) + [gtrace]
                    corrected_name = f"{corr_cfg.get('corrected_name', 'fluo_hemo_regressed_percent')}_{suffix}"
                    beta_prefix = f"{corr_cfg.get('beta_prefix', 'fluo_regression_beta')}_{suffix}"
                    r2_name = f"{corr_cfg.get('r2_name', 'fluo_regression_r2')}_{suffix}"
                    reg_ds_g = regress_fluorescence_from_regressors(
                        fluo_dff,
                        variant_regressors,
                        config,
                        corrected_name_override=corrected_name,
                        beta_prefix_override=beta_prefix,
                        r2_name_override=r2_name,
                    )
                    for name, var in reg_ds_g.data_vars.items():
                        if "beta" in name and not bool(corr_cfg.get("save_betas", True)):
                            continue
                        if name == r2_name and not bool(corr_cfg.get("save_r2", True)):
                            continue
                        ds_out[name] = var.astype("float32")
                        added.append(name)
                    if bool(variant.get("save_trace", True)):
                        ds_out[f"fluo_global_{suffix}_trace"] = gtrace.astype("float32")
                        added.append(f"fluo_global_{suffix}_trace")

                rep.end(details=f"channel={fluo_wave}; regressors={regressor_names}; outputs={added}")
            else:
                rep.end(details=f"channel={fluo_wave}; correction disabled; outputs={added}")
        else:
            rep.end(status="skipped", details="fluorescence.enabled=False")

        # ------------------------------------------------------------
        # 7. Global signal regression
        # ------------------------------------------------------------
        rep.begin("global signal regression", "[PROCESS] Global signal regression")
        made = apply_global_signal_variants(ds_out, masks, active_gs_variants, config)
        if made:
            rep.end(details=f"outputs={made}")
        else:
            rep.end(status="skipped", details="no active non-none global_signal variants")

        # ------------------------------------------------------------
        # 8. Smoothing
        # ------------------------------------------------------------
        rep.begin("smoothing", "[PROCESS] Spatial smoothing")
        sm_cfg = sp_cfg.get("smoothing", {}) or {}
        if bool(sm_cfg.get("enabled", False)):
            apply_to = sm_cfg.get("apply_to") or [name for name, var in ds_out.data_vars.items() if set(var.dims) == {"time", "y", "x"}]
            save_as_main = bool(sm_cfg.get("save_smoothed_as_main", True))
            save_unsmoothed = bool(sm_cfg.get("save_unsmoothed", False))
            suffix = sm_cfg.get("suffix", "smooth")
            smoothed_names = []
            for name in list(apply_to):
                if name not in ds_out or set(ds_out[name].dims) != {"time", "y", "x"}:
                    continue
                smoothed = smooth_signal_yx(ds_out[name], config)
                if save_as_main:
                    if save_unsmoothed:
                        ds_out[f"{name}_unsmoothed"] = ds_out[name]
                    ds_out[name] = smoothed
                    smoothed_names.append(name)
                else:
                    ds_out[f"{name}_{suffix}"] = smoothed
                    smoothed_names.append(f"{name}_{suffix}")
            rep.end(details=f"smoothed={smoothed_names}")
        else:
            rep.end(status="skipped", details="smoothing.enabled=False")

        # ------------------------------------------------------------
        # 9. Select outputs and rechunk
        # ------------------------------------------------------------
        rep.begin("select outputs and rechunk", "[PROCESS] Selecting outputs and aligning chunks for zarr write")
        save_signals = set(str(x) for x in (out_cfg.get("save_signals") or []))
        if save_signals:
            always_keep = {"background_trace", "raw_valid"}
            keep_vars = [
                name for name, var in ds_out.data_vars.items()
                if (
                    name in save_signals
                    or name in always_keep
                    or name.endswith("_trace")
                    or name.endswith("_beta")
                    or name.endswith("_r2")
                    or "regression_beta" in name
                    or "regression_r2" in name
                )
            ]
            ds_save = ds_out[keep_vars].copy()
        else:
            ds_save = ds_out

        chunks_3d = out_cfg.get("chunks", [256, 64, 64])
        ds_save = rechunk_dataset_for_zarr(ds_save, chunks_3d=chunks_3d)
        if debug:
            print_step4_dataset_summary(ds_save, title="Dataset selected for processed zarr write")
        rep.end(details=f"n_variables={len(ds_save.data_vars)}; output_chunks={chunks_3d}")

        # ------------------------------------------------------------
        # 10. Encoding + write
        # ------------------------------------------------------------
        rep.begin("write processed zarr", "[SAVE] Writing processed zarr")
        encoding: dict[str, dict[str, Any]] = {}
        compressor = make_compressor(config, "signal_processing")
        for name, var in ds_save.data_vars.items():
            enc: dict[str, Any] = {}
            if "time" in var.dims and "y" in var.dims and "x" in var.dims:
                enc["chunks"] = normalise_chunks(chunks_3d, [var.sizes[d] for d in var.dims])
                enc["dtype"] = out_cfg.get("store_dtype", "float32")
            elif var.dtype.kind == "f":
                enc["dtype"] = out_cfg.get("store_dtype", "float32")
            if compressor is not None and enc:
                enc["compressor"] = compressor
            if enc:
                encoding[name] = enc

        temp_path = out_path.with_name(out_path.name + ".incomplete")
        if temp_path.exists():
            shutil.rmtree(temp_path)
        ds_save.attrs["step4_fingerprint"] = current_fp
        ds_save.attrs["step4_complete"] = True
        try:
            ds_save.to_zarr(temp_path, mode="w", encoding=encoding, zarr_format=2, consolidated=False)
        except TypeError:
            try:
                ds_save.to_zarr(temp_path, mode="w", encoding=encoding, zarr_version=2, consolidated=False)
            except TypeError:
                ds_save.to_zarr(temp_path, mode="w", encoding=encoding)
        if out_path.exists():
            shutil.rmtree(out_path)
        temp_path.rename(out_path)
        rep.end(details=f"saved={out_path}")

        # ------------------------------------------------------------
        # 11. QC
        # ------------------------------------------------------------
        rep.begin("save QC", "[SAVE] Saving Step4 QC figures and traces")
        if save_qc:
            save_step4_qc_for_row(row, config, ds_save=ds_save, raw=raw, masks=masks, signal_roi=signal_roi)
            rep.end(details=f"qc_dir={get_qc_dir(row, config)}")
        else:
            rep.end(status="skipped", details="save_qc=False")

        # ------------------------------------------------------------
        # 12. Cleanup
        # ------------------------------------------------------------
        rep.begin("cleanup", "[CLEAN] Releasing temporary objects")
        rep.end()
        rep.save()
        return out_path

    except Exception as e:
        rep.fail(e)
        rep.save()
        raise

    finally:
        try:
            del raw, raw_corr, ds_out, ds_save, masks, signal_roi
        except Exception:
            pass
        gc.collect()
        rep.close()



# ============================================================
# QC figures and traces
# ============================================================


def get_qc_dir(row: pd.Series, config: dict[str, Any], paths: dict[str, Path] | None = None) -> Path:
    """Resolve the per-recording directory for Step4 quality-control artifacts."""
    step4_cfg = require_step4(config)
    if paths is not None:
        root = paths["step4_root"]
    else:
        root = Path(row["processed_zarr_path_resolved"]).parent.parent
    return root / step4_cfg.get("qc_folder", "qc_previews") / str(row["stem"])


def get_traces_dir(row: pd.Series, config: dict[str, Any], paths: dict[str, Path] | None = None) -> Path:
    """Resolve the directory for Step4 ROI trace tables."""
    step4_cfg = require_step4(config)
    if paths is not None:
        root = paths["step4_root"]
    else:
        root = Path(row["processed_zarr_path_resolved"]).parent.parent
    return root / step4_cfg.get("traces_folder", "roi_traces")


def dataset_roi_traces(ds: xr.Dataset, roi: xr.DataArray, *, max_vars: Sequence[str] | None = None) -> pd.DataFrame:
    """Extract ROI-mean traces and saved one-dimensional signals from a dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Processed dataset with spatial and temporal signal variables.
    roi : xarray.DataArray
        Spatial Boolean mask with dimensions (y, x).
    max_vars : object
        See the corresponding Step4 call site and YAML settings.
    
    Returns
    -------
    pandas.DataFrame
        Table of time coordinates and selected ROI traces.
    
    Notes
    -----
    Spatial variables with dimensions (time, y, x) are averaged over
    ROI pixels; one-dimensional time variables are copied. The output is a
    table with a ``time`` column and one column per selected trace."""
    traces: dict[str, np.ndarray] = {}
    time_values: np.ndarray | None = None
    allowed = set(max_vars) if max_vars else None
    for name, var in ds.data_vars.items():
        if allowed is not None and name not in allowed:
            continue
        if set(var.dims) == {"time", "y", "x"}:
            tr = mean_trace(var, roi).compute()
            traces[f"{name}_mean"] = np.asarray(tr.values)
            if time_values is None:
                time_values = np.asarray(tr["time"].values)
        elif var.dims == ("time",) or list(var.dims) == ["time"]:
            vv = var.compute()
            traces[name] = np.asarray(vv.values)
            if time_values is None and "time" in vv.coords:
                time_values = np.asarray(vv["time"].values)
    if time_values is None:
        time_values = np.arange(len(next(iter(traces.values())))) if traces else np.array([])
    df = pd.DataFrame({"time": time_values})
    for name, values in traces.items():
        if len(values) == len(df):
            df[name] = values
    return df


def plot_traces(df: pd.DataFrame, *, title: str, save_path: str | Path | None = None) -> plt.Figure:
    """Plot saved signals against the dataset time coordinate."""
    cols = [c for c in df.columns if c != "time"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in cols:
        ax.plot(df["time"], df[c], label=c, linewidth=1)
    ax.set_xlabel("time")
    ax.set_title(title)
    if cols:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def preview_processed_frames(row: pd.Series, ds: xr.Dataset, config: dict[str, Any], *, frame_index: int | None = None, save_path: str | Path | None = None) -> plt.Figure | None:
    """Plot selected processed two-dimensional signal frames at one time index."""
    frame_index = int(frame_index if frame_index is not None else cfg_get(config, "step4", "notebook", "preview_frame_index", default=10))
    save_signals = cfg_get(config, "signal_processing", "output", "save_signals", default=[])
    vars_3d = [name for name in save_signals if name in ds and set(ds[name].dims) == {"time", "y", "x"}]
    if not vars_3d:
        return None
    vars_3d = vars_3d[:6]
    fig, axes = plt.subplots(1, len(vars_3d), figsize=(4 * len(vars_3d), 4))
    if len(vars_3d) == 1:
        axes = [axes]
    for ax, name in zip(axes, vars_3d):
        idx = min(max(frame_index, 0), ds[name].sizes["time"] - 1)
        img = ds[name].isel(time=idx).compute().values
        vmin, vmax = np.nanpercentile(img, [2, 98])
        ax.imshow(img, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\nframe {idx}")
        ax.axis("off")
    fig.suptitle(f"Processed frames: {row['stem']}")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_step4_qc_for_row(
    row: pd.Series,
    config: dict[str, Any],
    *,
    ds_save: xr.Dataset | None = None,
    raw: xr.DataArray | None = None,
    masks: dict[str, xr.DataArray] | None = None,
    signal_roi: xr.DataArray | None = None,
) -> None:
    """Create configured Step4 quality-control figures and trace files."""
    if not bool(cfg_get(config, "signal_processing", "output", "save_static_qc_png", default=True)):
        return
    qc_dir = get_qc_dir(row, config)
    traces_dir = get_traces_dir(row, config)
    qc_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    if raw is None:
        raw = open_raw_zarr(select_raw_zarr_path_for_processing(row, config), config)
    if masks is None:
        masks = load_masks_for_row(row, config, image_shape=(raw.sizes["y"], raw.sizes["x"]))
    if signal_roi is None:
        signal_roi = make_signal_roi(masks, config)
    if ds_save is None:
        ds_save = xr.open_zarr(row["processed_zarr_path_resolved"], chunks="auto", mask_and_scale=False)

    try:
        preview_raw_frames_from_zarr(row, config, save_path=qc_dir / "raw_channels_from_zarr.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save raw zarr preview for {row['stem']}: {e}")
    try:
        preview_masks(row, masks, signal_roi, save_path=qc_dir / "masks_and_signal_roi.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save mask preview for {row['stem']}: {e}")
    try:
        save_vars = cfg_get(config, "signal_processing", "output", "save_signals", default=None)
        traces = dataset_roi_traces(ds_save, signal_roi, max_vars=save_vars)
        traces_path = traces_dir / f"{row['stem']}_roi_traces.csv"
        traces.to_csv(traces_path, index=False)
        plot_traces(traces, title=f"ROI traces: {row['stem']}", save_path=qc_dir / "final_roi_traces.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save ROI traces for {row['stem']}: {e}")
    try:
        preview_processed_frames(row, ds_save, config, save_path=qc_dir / "processed_signal_frames.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save processed frame preview for {row['stem']}: {e}")


# ============================================================
# Batch runners
# ============================================================


def write_processing_log(paths: dict[str, Path], config: dict[str, Any], rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Write recording-level processing status to the configured CSV log."""
    step4_cfg = require_step4(config)
    log = pd.DataFrame(rows)
    log_path = paths["step4_root"] / step4_cfg.get("processing_log_name", "processing_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        old = pd.read_csv(log_path)
        log_all = pd.concat([old, log], ignore_index=True)
    else:
        log_all = log
    log_all.to_csv(log_path, index=False)
    return log


def run_raw_build_phase(
    df: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    overwrite: bool | None = None,
    save_bmp_preview: bool = True,
) -> pd.DataFrame:
    """Build or reuse raw Zarr stores for every selected recording.
    
    Parameters
    ----------
    df : object
        See the corresponding Step4 call site and YAML settings.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    paths : dict
        Resolved project output directories.
    overwrite : bool, optional
        Override the configured reuse/overwrite setting.
    save_bmp_preview : object
        See the corresponding Step4 call site and YAML settings.
    
    Notes
    -----
    This is Phase A of the two-phase Step4 workflow. It operates
    recording by recording and writes a status log."""
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step4A raw_zarr", unit="stem"):
        rec = {
            "phase": "raw_build",
            "stem": row["stem"],
            "group_id": row.get("group_id", ""),
            "raw_zarr_path": str(select_raw_zarr_path_for_processing(row, config)),
            "raw_zarr_variant": selected_raw_variant_for_row(row, config),
            "raw_zarr_unaligned_path": row.get("raw_zarr_path_resolved", ""),
            "raw_zarr_motion_corrected_path": row.get("raw_zarr_motion_corrected_path_resolved", ""),
            "processed_zarr_path": str(select_processed_zarr_path_for_processing(row, config)),
            "status": "skipped",
            "error": "",
            "updated_at": _dt.datetime.now().isoformat(),
        }
        try:
            if save_bmp_preview and bool(cfg_get(config, "step4", "notebook", "save_static_previews", default=True)):
                qc_dir = get_qc_dir(row, config, paths)
                preview_raw_frames_from_bmp(row, config, save_path=qc_dir / "raw_channels_from_bmp.png")
                plt.close("all")
            prior = is_complete_zarr(Path(row["raw_zarr_path_resolved"]), config, raw=True)
            save_raw_zarr_for_row_batch(row, config, overwrite=overwrite, show_progress=True)
            rec["status"] = "reused" if prior and not (overwrite if overwrite is not None else config["step4"].get("overwrite_raw", False)) else "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = repr(e)
        rows.append(rec)
        gc.collect()
    result = write_processing_log(paths, config, rows)
    print(f"[Step4 {rows[0]['phase'] if rows else 'batch'}] " + ", ".join(f"{status}={sum(r['status'] == status for r in rows)}" for status in ("ok", "reused", "error")))
    for r in rows:
        if r["status"] == "error":
            print(f"  [MISSING/ERROR] {r['stem']}: {r['error']}")
    return result


def run_signal_processing_phase(
    df: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    overwrite: bool | None = None,
) -> pd.DataFrame:
    """Convert selected raw Zarr recordings to processed signal stores.
    
    Parameters
    ----------
    df : object
        See the corresponding Step4 call site and YAML settings.
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    paths : dict
        Resolved project output directories.
    overwrite : bool, optional
        Override the configured reuse/overwrite setting.
    
    Notes
    -----
    This is Phase B. It requires completed raw Zarr stores and writes
    processed datasets and recording-level status logs."""
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step4B processed_signals", unit="stem"):
        rec = {
            "phase": "signal_processing",
            "stem": row["stem"],
            "group_id": row.get("group_id", ""),
            "raw_zarr_path": str(select_raw_zarr_path_for_processing(row, config)),
            "raw_zarr_variant": selected_raw_variant_for_row(row, config),
            "raw_zarr_unaligned_path": row.get("raw_zarr_path_resolved", ""),
            "raw_zarr_motion_corrected_path": row.get("raw_zarr_motion_corrected_path_resolved", ""),
            "processed_zarr_path": str(select_processed_zarr_path_for_processing(row, config)),
            "status": "skipped",
            "error": "",
            "updated_at": _dt.datetime.now().isoformat(),
        }
        try:
            raw_path = select_raw_zarr_path_for_processing(row, config)
            if not is_complete_zarr(raw_path, config, raw=True):
                raise RuntimeError(f"Raw build not complete for stem={row['stem']}: {raw_path}")
            prior = is_complete_zarr(select_processed_zarr_path_for_processing(row, config), config, raw=False)
            process_raw_zarr_for_row(row, config, overwrite=overwrite, save_qc=True)
            rec["status"] = "reused" if prior and not (overwrite if overwrite is not None else config["step4"].get("overwrite_processed", False)) else "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = repr(e)
        rows.append(rec)
        gc.collect()
    result = write_processing_log(paths, config, rows)
    print(f"[Step4 {rows[0]['phase'] if rows else 'batch'}] " + ", ".join(f"{status}={sum(r['status'] == status for r in rows)}" for status in ("ok", "reused", "error")))
    for r in rows:
        if r["status"] == "error":
            print(f"  [MISSING/ERROR] {r['stem']}: {r['error']}")
    return result


def run_one_stem_full_test(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    overwrite_raw: bool | None = None,
    overwrite_processed: bool | None = None,
) -> Path:
    """Run raw conversion and signal processing for one recording with QC previews."""
    qc_dir = get_qc_dir(row, config, paths)
    preview_raw_frames_from_bmp(row, config, save_path=qc_dir / "raw_channels_from_bmp.png")
    plt.show()
    save_raw_zarr_for_row_batch(row, config, overwrite=overwrite_raw, show_progress=True)
    preview_raw_frames_from_zarr(row, config, save_path=qc_dir / "raw_channels_from_zarr.png")
    plt.show()
    out = process_raw_zarr_for_row(row, config, overwrite=overwrite_processed, save_qc=True)
    ds = xr.open_zarr(out, chunks="auto", mask_and_scale=False)
    raw = open_raw_zarr(select_raw_zarr_path_for_processing(row, config), config)
    masks = load_masks_for_row(row, config, image_shape=(raw.sizes["y"], raw.sizes["x"]), raw_valid=(ds["raw_valid"] == 1))
    roi = make_signal_roi(masks, config)
    preview_masks(row, masks, roi)
    plt.show()
    traces = dataset_roi_traces(ds, roi, max_vars=cfg_get(config, "signal_processing", "output", "save_signals", default=None))
    plot_roi_traces_dual_axis(traces, title=f"ROI traces: {row['stem']}", config=config)
    plt.show()
    preview_processed_frames(row, ds, config)
    plt.show()
    return out


def run_step4_two_phase(
    config_path: str | Path,
    *,
    include_bad: bool | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    overwrite_raw: bool | None = None,
    overwrite_processed: bool | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run the configured raw-build and signal-processing phases sequentially."""
    df, config, paths = build_step4_index(config_path, include_bad=include_bad, groups=groups, stems=stems)
    step4_cfg = require_step4(config)
    raw_log = None
    sig_log = None
    if bool(step4_cfg.get("run_raw_build", True)):
        raw_log = run_raw_build_phase(df, config, paths, overwrite=overwrite_raw)
    # Rebuild index after raw phase so raw_zarr_exists flags are fresh.
    df, config, paths = build_step4_index(config_path, include_bad=include_bad, groups=groups, stems=stems)
    if bool(step4_cfg.get("run_signal_processing", True)):
        sig_log = run_signal_processing_phase(df, config, paths, overwrite=overwrite_processed)
    return raw_log, sig_log


# ============================================================
# Optional napari helpers — manual use only, not batch
# ============================================================


def open_raw_in_napari_for_row(row: pd.Series, config: dict[str, Any]) -> None:
    """Open one recording of raw wavelength channels in Napari."""
    import napari
    raw = open_raw_zarr(select_raw_zarr_path_for_processing(row, config), config)
    viewer = napari.Viewer(title=f"Step4 raw preview ({selected_raw_variant_for_row(row, config)}): {row['stem']}")
    for wi, wave in enumerate(raw.coords["wavelength"].values):
        viewer.add_image(raw.isel(wavelength=wi).data, name=f"raw_{wave}")
    napari.run()


def open_processed_in_napari_for_row(row: pd.Series, config: dict[str, Any], signals: Sequence[str] | None = None) -> None:
    """Open configured processed signal maps for one recording in Napari."""
    import napari
    ds = xr.open_zarr(row["processed_zarr_path_resolved"], chunks="auto", mask_and_scale=False)
    if signals is None:
        signals = cfg_get(config, "visualization", "napari", "default_signals", default=cfg_get(config, "signal_processing", "output", "save_signals", default=[]))
    viewer = napari.Viewer(title=f"Step4 processed preview: {row['stem']}")
    for name in signals:
        if name in ds and set(ds[name].dims) == {"time", "y", "x"}:
            viewer.add_image(ds[name].data, name=name)
    napari.run()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    """Parse Step4 CLI arguments and execute the requested processing phases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--groups", nargs="*", default=None)
    parser.add_argument("--stems", nargs="*", default=None)
    parser.add_argument("--include-bad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--raw-only", action="store_true")
    parser.add_argument("--signals-only", action="store_true")
    parser.add_argument(
        "--reuse-existing-raw", action="store_true",
        help="For --signals-only: process the SAVED raw pixels even if source BMP paths/mtimes changed; "
             "still validate completeness and raw pixel/geometry settings",
    )
    parser.add_argument("--overwrite-raw", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite-processed", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    if args.reuse_existing_raw and (not args.signals_only or args.raw_only):
        parser.error("--reuse-existing-raw must be combined with --signals-only")
    df, config, paths = build_step4_index(args.config, include_bad=args.include_bad, groups=args.groups, stems=args.stems)
    if args.reuse_existing_raw:
        config.setdefault("step4", {})["reuse_existing_raw_for_processing"] = True
    if not args.signals_only:
        print(run_raw_build_phase(df, config, paths, overwrite=args.overwrite_raw))
    if not args.raw_only:
        df, config, paths = build_step4_index(args.config, include_bad=args.include_bad, groups=args.groups, stems=args.stems)
        if args.reuse_existing_raw:
            config.setdefault("step4", {})["reuse_existing_raw_for_processing"] = True
        print(run_signal_processing_phase(df, config, paths, overwrite=args.overwrite_processed))





# ============================================================
# v1_7 QC visualization overrides and helpers
# ============================================================
# These functions intentionally keep the computation pipeline unchanged and only
# add the visual/QC behavior discussed for the notebook:
# - masked image previews with a white/transparent background, similar to the
#   old signal_converter.show_figures() logic;
# - optional hemoglobin scaling to microM for stored outputs and plots;
# - one compact ROI trace panel with Hb on the left y-axis and fluorescence on
#   the right y-axis.

try:
    import matplotlib.ticker as ticker
except Exception:  # pragma: no cover
    ticker = None



def _hb_scale_factor_and_units(config: dict[str, Any]) -> tuple[float, str]:
    """Resolve the configured hemoglobin display and storage scale and unit label.
    
    Parameters
    ----------
    config : dict
        Parsed YAML configuration used by this Step4 operation.
    
    Notes
    -----
    ``signal_processing.hemoglobin.scale_to_uM`` takes precedence.
    Legacy unit strings, including Cyrillic spellings, remain valid input
    values for backward compatibility."""
    hb_cfg = cfg_get(config, "signal_processing", "hemoglobin", default={}) or {}
    # Preferred explicit key.  Set this true in config to save/display Hb in µM.
    if bool(hb_cfg.get("scale_to_uM", False)):
        return 1_000_000.0, "µM"
    units = str(hb_cfg.get("output_units", hb_cfg.get("units", ""))).strip().lower()
    if units in {"um", "µm", "μm", "umol", "µmol", "μmol", "micromolar", "micromol", "micromoles", "mkм", "мкм", "мкмоли"}:
        return 1_000_000.0, "µM"
    return float(hb_cfg.get("scale_factor", 1.0)), str(hb_cfg.get("units", "a.u."))


def convert_reflectance_to_hb(dA1: xr.DataArray, dA2: xr.DataArray, wave1: int, wave2: int, config: dict[str, Any]) -> xr.Dataset:
    """Convert two reflectance attenuation maps into hemoglobin signals.
    
    Parameters
    ----------
    dA1 : xarray.DataArray
        Natural-log attenuation for the first wavelength (time, y, x).
    dA2 : xarray.DataArray
        Natural-log attenuation for the second wavelength (time, y, x).
    wave1 : int
        First reflectance wavelength in nm.
    wave2 : int
        Second reflectance wavelength in nm.
    config : dict
        Configuration containing `signal_processing.hemoglobin` tables,
        output names, and output-scaling settings.
    
    Returns
    -------
    hemoglobin : xarray.Dataset
        Spatial HbO, HHb, and HbT changes with dimensions (time, y, x).
    
    Notes
    -----
    ``_convert_reflectance_to_hb_base`` performs the two-wavelength inversion
    using configured extinction coefficients and optical pathlengths. This
    function applies optional output scaling via
    ``signal_processing.hemoglobin.scale_to_uM`` or legacy unit labels.
    It preserves coefficient provenance and sets the dataset and variable
    attributes for the actual units and scale. Both CLI and notebook imports
    use this scaling wrapper."""
    ds = _convert_reflectance_to_hb_base(dA1, dA2, wave1, wave2, config)
    used_json = ds.attrs.get("hb_coefficients_used_json")
    scale, units = _hb_scale_factor_and_units(config)
    if scale != 1.0:
        ds = ds * np.float32(scale)
    if used_json is not None:
        ds.attrs["hb_coefficients_used_json"] = used_json
    for name in ds.data_vars:
        ds[name].attrs["units"] = units
        ds[name].attrs["hb_scale_factor"] = scale
    ds.attrs["hemoglobin_units"] = units
    ds.attrs["hemoglobin_scale_factor"] = scale
    return ds.astype("float32")


def _copy_cmap_with_bad(cmap_name: str, *, background: str = "white"):
    """Copy a matplotlib colormap with the configured color for missing values."""
    cmap = plt.get_cmap(cmap_name).copy()
    bg = str(background).lower()
    if bg == "transparent":
        cmap.set_bad(color=(1, 1, 1, 0))
    else:
        cmap.set_bad(color="white")
    return cmap


def _as_2d_frame(data: Any, frame_index: int | None = None) -> np.ndarray:
    """Select a spatial image from a time-dependent or two-dimensional signal."""
    if isinstance(data, xr.DataArray):
        if "time" in data.dims:
            idx = int(frame_index or 0)
            idx = min(max(idx, 0), int(data.sizes["time"]) - 1)
            data = data.isel(time=idx)
        data = data.compute().values
    arr = np.asarray(data)
    if arr.ndim == 3:
        idx = int(frame_index or 0)
        idx = min(max(idx, 0), arr.shape[0] - 1)
        arr = arr[idx]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D frame or 3D time stack, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def masked_frame_for_display(
    data: Any,
    mask: Any,
    *,
    frame_index: int | None = None,
    outside_value: float = np.nan,
) -> np.ndarray:
    """Return a 2D frame where pixels outside mask are NaN/transparent.

    This is the xarray/dask-friendly equivalent of the old show_figures()
    pattern from signal_converter.py: mask non-brain pixels, compute color scale
    from valid pixels only, and leave the background white/transparent.
    """
    img = _as_2d_frame(data, frame_index=frame_index)
    if isinstance(mask, xr.DataArray):
        m = mask.compute().values
    else:
        m = np.asarray(mask)
    if img.shape != m.shape:
        raise ValueError(f"Mask shape mismatch for display: image {img.shape} vs mask {m.shape}")
    out = img.copy()
    out[~m.astype(bool)] = outside_value
    return out


def _display_limits(arr: np.ndarray, config: dict[str, Any]) -> tuple[float, float]:
    """Choose image-display limits from configured intensity percentiles."""
    viz_cfg = cfg_get(config, "step4", "visualization", default={}) or {}
    p = viz_cfg.get("percentile_range", [2, 98])
    try:
        lo, hi = float(p[0]), float(p[1])
    except Exception:
        lo, hi = 2.0, 98.0
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(vals, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def show_masked_frames(
    data_list: Sequence[Any],
    names: Sequence[str],
    *,
    frame_index: int,
    mask: Any,
    config: dict[str, Any],
    save_path: str | Path | None = None,
    title: str | None = None,
    cmap: str | None = None,
) -> plt.Figure:
    """Show several masked frames side-by-side with old show_figures-like scaling."""
    viz_cfg = cfg_get(config, "step4", "visualization", default={}) or {}
    cmap_name = cmap or viz_cfg.get("frame_cmap", "jet")
    background = viz_cfg.get("background", "white")
    cm = _copy_cmap_with_bad(cmap_name, background=background)
    n = len(data_list)
    font_scale = float(viz_cfg.get("font_scale", 1.0))
    fig, axes = plt.subplots(1, n, figsize=(4 * n * max(1.0, font_scale / 2), 4 * max(1.0, font_scale / 2)))
    if n == 1:
        axes = [axes]
    for ax, data, name in zip(axes, data_list, names):
        arr = masked_frame_for_display(data, mask, frame_index=frame_index)
        vmin, vmax = _display_limits(arr, config)
        ax.set_facecolor("white")
        im = ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax)
        ax.set_title(str(name), fontsize=10 * font_scale)
        ax.axis("off")
        if bool(viz_cfg.get("show_colorbar", False)):
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8 * font_scale)
    if title:
        fig.suptitle(title, fontsize=12 * font_scale)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(viz_cfg.get("dpi", 150)), bbox_inches="tight", transparent=(background == "transparent"))
    return fig


def _var_is_3d_time_yx(var: xr.DataArray) -> bool:
    """Check whether a signal has the spatial and temporal dimensions needed for frame QC."""
    return "time" in var.dims and "y" in var.dims and "x" in var.dims


def preview_processed_frames(
    row: pd.Series,
    ds: xr.Dataset,
    config: dict[str, Any],
    *,
    frame_index: int | None = None,
    save_path: str | Path | None = None,
    mask: xr.DataArray | None = None,
) -> plt.Figure | None:
    """Masked processed-signal frame preview.

    Unlike v1_6, this masks the background and uses the old show_figures-style
    percentile color scaling.  By default it uses the Step4 signal ROI
    (brain_minus_bad) so the outside background is white/transparent.
    """
    frame_index = int(frame_index if frame_index is not None else cfg_get(config, "step4", "notebook", "preview_frame_index", default=10))
    save_signals = cfg_get(config, "signal_processing", "output", "save_signals", default=[])
    vars_3d = [name for name in save_signals if name in ds and _var_is_3d_time_yx(ds[name])]
    if not vars_3d:
        return None
    vars_3d = vars_3d[: int(cfg_get(config, "step4", "visualization", "max_frame_panels", default=6))]

    if mask is None:
        raw = open_raw_zarr(select_raw_zarr_path_for_processing(row, config), config)
        raw_valid = (ds["raw_valid"] == 1)
        masks = load_masks_for_row(row, config, image_shape=(raw.sizes["y"], raw.sizes["x"]), raw_valid=raw_valid)
        mask = make_signal_roi(masks, config)

    fig = show_masked_frames(
        [ds[name] for name in vars_3d],
        vars_3d,
        frame_index=frame_index,
        mask=mask,
        config=config,
        save_path=save_path,
        title=f"Processed masked frames: {row['stem']} | frame {frame_index}",
    )
    return fig


def _trace_groups_from_columns(columns: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separate hemoglobin and fluorescence columns for QC plotting."""
    hb_cols: list[str] = []
    fluo_cols: list[str] = []
    for c in columns:
        cl = c.lower()
        if c == "time":
            continue
        if any(k in cl for k in ["dchbo", "dchhb", "dchbt", "hbo", "hhb", "hbt"]):
            hb_cols.append(c)
        elif any(k in cl for k in ["fluo", "ca", "gcamp"]):
            fluo_cols.append(c)
    return hb_cols, fluo_cols


def _plot_roi_traces_by_variant(
    df: pd.DataFrame,
    *,
    title: str,
    config: dict[str, Any],
    save_path: str | Path | None,
    hb_cols: Sequence[str],
    fluo_cols: Sequence[str],
) -> plt.Figure:
    """Separate Hb raw/gsub/vreg panels with independent y-axis limits (QC only)."""
    viz_cfg = cfg_get(config, "step4", "visualization", "traces", default={}) or {}
    scale = float(viz_cfg.get("font_scale", cfg_get(config, "step4", "visualization", "font_scale", default=1.0)))
    colors = viz_cfg.get("colors", {}) or {}
    grouped: dict[str, list[str]] = {"raw": [], "gsub": [], "vreg": []}
    for name in hb_cols:
        if name not in df:
            continue
        base = name[:-5] if name.endswith("_mean") else name
        if base.endswith("_vreg"):
            grouped["vreg"].append(name)
        elif base.endswith("_gsub"):
            grouped["gsub"].append(name)
        else:
            grouped["raw"].append(name)
    groups = [(key, grouped[key]) for key in ("raw", "gsub", "vreg") if grouped[key]]
    if fluo_cols:
        groups.append(("fluorescence", [name for name in fluo_cols if name in df]))
    if not groups:
        raise ValueError("No Hb or fluorescence traces available for QC plot")

    width, height = viz_cfg.get("figsize", [20, 18])
    fig, axes = plt.subplots(len(groups), 1, figsize=(width, height), sharex=True, squeeze=False)
    axes = axes[:, 0]
    x = df["time"].values
    labels = {"raw": "Hb — without GSR (raw Hb)", "gsub": "Hb — gsub", "vreg": "Hb — vreg", "fluorescence": "Fluorescence"}
    for ax, (kind, columns) in zip(axes, groups):
        for c in columns:
            name = c[:-5] if c.endswith("_mean") else c
            ax.plot(x, df[c].values, label=name, color=colors.get(name), linewidth=1.5)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(labels[kind], fontsize=11 * scale)
        ax.set_ylabel(viz_cfg.get("y_fluo", "ΔF/F₀ (%)") if kind == "fluorescence"
                      else viz_cfg.get("y_hb", "Δ[Hb] (µM)"), fontsize=10 * scale)
        ax.tick_params(axis="both", labelsize=8 * scale)
        ax.legend(fontsize=8 * scale, ncol=min(3, len(columns)), loc="best")
        # Intentionally no shared Hb y limits: raw/gsub/vreg need their own scales.
        if kind == "fluorescence" and viz_cfg.get("fluo_y_lim") is not None:
            ax.set_ylim(viz_cfg["fluo_y_lim"])
        if kind != "fluorescence" and viz_cfg.get("hb_y_lim") is not None:
            ax.set_ylim(viz_cfg["hb_y_lim"])
        if ticker is not None:
            if viz_cfg.get("hb_tick_step") is not None and kind != "fluorescence":
                ax.yaxis.set_major_locator(ticker.MultipleLocator(float(viz_cfg["hb_tick_step"])))
            if viz_cfg.get("fluo_tick_step") is not None and kind == "fluorescence":
                ax.yaxis.set_major_locator(ticker.MultipleLocator(float(viz_cfg["fluo_tick_step"])))
    axes[-1].set_xlabel(viz_cfg.get("x_label", "Time (s)"), fontsize=10 * scale)
    if viz_cfg.get("x_lim") is not None:
        axes[-1].set_xlim(viz_cfg["x_lim"])
    if ticker is not None and viz_cfg.get("x_tick_step") is not None:
        axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(float(viz_cfg["x_tick_step"])))
    fig.suptitle(title, fontsize=12 * scale)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(viz_cfg.get("dpi", 200)), bbox_inches="tight")
    return fig


def plot_roi_traces_dual_axis(
    df: pd.DataFrame,
    *,
    title: str,
    config: dict[str, Any],
    save_path: str | Path | None = None,
    hb_cols: Sequence[str] | None = None,
    fluo_cols: Sequence[str] | None = None,
) -> plt.Figure:
    """Plot ROI traces with Hb on left axis and fluorescence on right axis.

    This is a simplified single-record analogue of plot_single_panel_signals_df:
    no seaborn, no grouping, no stimulus annotations by default; just aligned
    axes and stable colors for QC.
    """
    if "time" not in df.columns:
        raise ValueError("ROI traces DataFrame must contain a 'time' column")
    auto_hb, auto_fluo = _trace_groups_from_columns(df.columns)
    hb_cols = list(hb_cols) if hb_cols is not None else auto_hb
    fluo_cols = list(fluo_cols) if fluo_cols is not None else auto_fluo

    viz_cfg = cfg_get(config, "step4", "visualization", "traces", default={}) or {}
    if viz_cfg.get("layout", "overlay") == "by_variant":
        return _plot_roi_traces_by_variant(
            df, title=title, config=config, save_path=save_path,
            hb_cols=hb_cols, fluo_cols=fluo_cols,
        )
    colors = viz_cfg.get("colors", {}) or {
        "dcHbO": "red",
        "dcHHb": "blue",
        "dcHbT": "magenta",
        "fluo_dff_percent": "black",
        "fluo_hemo_regressed_percent": "green",
        "fluo_hemo_regressed_percent_gsr": "darkgreen",
        "dcHbT_gsr": "purple",
    }
    fig, ax_hb = plt.subplots(figsize=tuple(viz_cfg.get("figsize", [10, 5])))
    ax_fluo = ax_hb.twinx()
    x = df["time"].values

    def color_for(col: str) -> str:
        """Select the configured plotting color for one signal name."""
        base = col[:-5] if col.endswith("_mean") else col
        return colors.get(base, colors.get(col, "black"))

    for c in hb_cols:
        if c in df.columns:
            label = c[:-5] if c.endswith("_mean") else c
            ax_hb.plot(x, df[c].values, label=label, color=color_for(c), linewidth=1.2)
    for c in fluo_cols:
        if c in df.columns:
            label = c[:-5] if c.endswith("_mean") else c
            ax_fluo.plot(x, df[c].values, label=label, color=color_for(c), linewidth=1.2)

    ax_hb.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_hb.set_xlabel(viz_cfg.get("x_label", "Time (s)"))
    hb_units = _hb_scale_factor_and_units(config)[1]
    ax_hb.set_ylabel(viz_cfg.get("y_hb", f"Δ[Hb] ({hb_units})"), color=viz_cfg.get("hb_axis_color", "red"))
    ax_fluo.set_ylabel(viz_cfg.get("y_fluo", "ΔF/F₀ fluorescence (%)"), color=viz_cfg.get("fluo_axis_color", "green"))
    ax_hb.tick_params(axis="y", colors=viz_cfg.get("hb_axis_color", "red"))
    ax_fluo.tick_params(axis="y", colors=viz_cfg.get("fluo_axis_color", "green"))
    ax_fluo.spines["right"].set_color(viz_cfg.get("fluo_axis_color", "green"))

    if viz_cfg.get("x_lim") is not None:
        ax_hb.set_xlim(viz_cfg["x_lim"])
    if viz_cfg.get("hb_y_lim") is not None:
        ax_hb.set_ylim(viz_cfg["hb_y_lim"])
    if viz_cfg.get("fluo_y_lim") is not None:
        ax_fluo.set_ylim(viz_cfg["fluo_y_lim"])
    if ticker is not None:
        if viz_cfg.get("x_tick_step") is not None:
            ax_hb.xaxis.set_major_locator(ticker.MultipleLocator(float(viz_cfg["x_tick_step"])))
        if viz_cfg.get("hb_tick_step") is not None:
            ax_hb.yaxis.set_major_locator(ticker.MultipleLocator(float(viz_cfg["hb_tick_step"])))
        if viz_cfg.get("fluo_tick_step") is not None:
            ax_fluo.yaxis.set_major_locator(ticker.MultipleLocator(float(viz_cfg["fluo_tick_step"])))

    lines1, labels1 = ax_hb.get_legend_handles_labels()
    lines2, labels2 = ax_fluo.get_legend_handles_labels()
    if lines1 or lines2:
        ax_hb.legend(lines1 + lines2, labels1 + labels2, fontsize=8, ncol=2, loc="best")
    ax_hb.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=int(viz_cfg.get("dpi", 200)), bbox_inches="tight")
    return fig


def plot_traces(df: pd.DataFrame, *, title: str, save_path: str | Path | None = None) -> plt.Figure:
    """Backward-compatible wrapper used by older calls.

    If the caller does not pass config, keep a simple single-axis plot.  The new
    QC saving path calls plot_roi_traces_dual_axis() directly.
    """
    cols = [c for c in df.columns if c != "time"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for c in cols:
        ax.plot(df["time"], df[c], label=c, linewidth=1)
    ax.set_xlabel("time")
    ax.set_title(title)
    if cols:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_step4_qc_for_row(
    row: pd.Series,
    config: dict[str, Any],
    *,
    ds_save: xr.Dataset | None = None,
    raw: xr.DataArray | None = None,
    masks: dict[str, xr.DataArray] | None = None,
    signal_roi: xr.DataArray | None = None,
) -> None:
    """Save QC previews using masked images and dual-axis ROI traces."""
    if not bool(cfg_get(config, "signal_processing", "output", "save_static_qc_png", default=True)):
        return
    qc_dir = get_qc_dir(row, config)
    traces_dir = get_traces_dir(row, config)
    qc_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    if ds_save is None:
        ds_save = xr.open_zarr(select_processed_zarr_path_for_processing(row, config), chunks="auto", consolidated=False, mask_and_scale=False)
    if raw is None:
        raw = open_raw_zarr(select_raw_zarr_path_for_processing(row, config), config)
    if masks is None:
        raw_valid = (ds_save["raw_valid"] == 1)
        masks = load_masks_for_row(row, config, image_shape=(raw.sizes["y"], raw.sizes["x"]), raw_valid=raw_valid)
    if signal_roi is None:
        signal_roi = make_signal_roi(masks, config)

    frame_index = int(cfg_get(config, "step4", "notebook", "preview_frame_index", default=10))

    try:
        preview_raw_frames_from_zarr(row, config, save_path=qc_dir / "raw_channels_from_zarr.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save raw zarr preview for {row['stem']}: {e}")
    try:
        preview_masks(row, masks, signal_roi, save_path=qc_dir / "masks_and_signal_roi.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save mask preview for {row['stem']}: {e}")
    try:
        save_vars = cfg_get(config, "signal_processing", "output", "save_signals", default=None)
        traces = dataset_roi_traces(ds_save, signal_roi, max_vars=save_vars)
        traces_path = traces_dir / f"{row['stem']}_roi_traces.csv"
        traces.to_csv(traces_path, index=False)
        plot_roi_traces_dual_axis(traces, title=f"ROI traces: {row['stem']}", config=config, save_path=qc_dir / "final_roi_traces_dual_axis.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save ROI traces for {row['stem']}: {e}")
    try:
        preview_processed_frames(row, ds_save, config, frame_index=frame_index, mask=signal_roi, save_path=qc_dir / "processed_signal_frames_masked.png")
        plt.close("all")
    except Exception as e:
        print(f"[WARN] Could not save processed frame preview for {row['stem']}: {e}")


def preview_processed_qc_for_row(row: pd.Series, config: dict[str, Any]) -> tuple[plt.Figure | None, plt.Figure | None]:
    """Notebook helper: open processed zarr and display masked frames + dual-axis traces."""
    ds = xr.open_zarr(select_processed_zarr_path_for_processing(row, config), chunks="auto", consolidated=False, mask_and_scale=False)
    raw = open_raw_zarr(select_raw_zarr_path_for_processing(row, config), config)
    raw_valid = (ds["raw_valid"] == 1)
    masks = load_masks_for_row(row, config, image_shape=(raw.sizes["y"], raw.sizes["x"]), raw_valid=raw_valid)
    signal_roi = make_signal_roi(masks, config)
    fig_frames = preview_processed_frames(row, ds, config, mask=signal_roi)
    save_vars = cfg_get(config, "signal_processing", "output", "save_signals", default=None)
    traces = dataset_roi_traces(ds, signal_roi, max_vars=save_vars)
    fig_traces = plot_roi_traces_dual_axis(traces, title=f"ROI traces: {row['stem']}", config=config)
    return fig_frames, fig_traces

# End of v1_13 merged from v1_11_geometry GSR/preprocessing + v1_11_geometry_motion raw motion.


if __name__ == "__main__":
    main()
