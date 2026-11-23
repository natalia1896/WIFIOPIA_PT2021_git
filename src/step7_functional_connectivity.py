"""Current audited Step7 functional connectivity with current Step6b API.

Read-only Step6b mask-set selections and Finalizer Awake REST windows.
One of two Step6b branches (with_rest / without_rest), and independent
signal x temporal-filter variants in separate result directories.
Homotopic FC / GBC mathematical kernels and Fisher-z aggregation are preserved.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import gc
import json
import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import butter, detrend as scipy_detrend, sosfiltfilt
from skimage.transform import warp

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional progress dependency
    _tqdm = None

try:
    import step6b_fc_geometry_coverage as s6b
except Exception as exc:  # pragma: no cover - project environment dependent
    raise ImportError(
        "Step7 requires the current audited step6b_fc_geometry_coverage.py next to it."
    ) from exc


# =============================================================================
# Config / paths
# =============================================================================


def _deep_fill_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively fill missing nested configuration keys from a defaults mapping.
    
    Parameters
    ----------
    dst : dict[str, Any]
        Destination dictionary modified in place.
    src : dict[str, Any]
        Source dictionary providing defaults or overrides.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation."""
    for key, value in src.items():
        if key not in dst:
            dst[key] = copy.deepcopy(value)
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_fill_missing(dst[key], value)
    return dst


def _deep_update(dst: dict[str, Any], src: dict[str, Any] | None) -> dict[str, Any]:
    """Recursively apply explicitly supplied configuration overrides.
    
    Parameters
    ----------
    dst : dict[str, Any]
        Destination dictionary modified in place.
    src : dict[str, Any] | None
        Source dictionary providing defaults or overrides.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation."""
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def default_step7_config() -> dict[str, Any]:
    """Return the default settings for the functional-connectivity pipeline.
    
    
    Returns
    -------
    dict[str, Any]
        New dictionary of default Step7 settings.
    
    Notes
    -----
    Settings belong to the YAML section ``functional_connectivity``. The four default
    definitions span ``dcHbT``/``dcHbT_vreg`` and ``bandpass``/``lowpass``. A single run uses
    only its selected variant and REST branch."""
    return {
        "folder_name": "step7_functional_connectivity",
        "rest_variant": "with_rest",
        "active_variant_id": "hbt_bandpass",
        "variants": {
            "selected": ["hbt_bandpass", "hbt_lowpass", "hbt_vreg_bandpass", "hbt_vreg_lowpass"],
            "definitions": {
                "hbt_bandpass": {"signal": "dcHbT", "filter_type": "bandpass"},
                "hbt_lowpass": {"signal": "dcHbT", "filter_type": "lowpass"},
                "hbt_vreg_bandpass": {"signal": "dcHbT_vreg", "filter_type": "bandpass"},
                "hbt_vreg_lowpass": {"signal": "dcHbT_vreg", "filter_type": "lowpass"},
            },
        },
        "mask_sets": ["stroke_longitudinal"],
        "signal": "dcHbT",
        "rest_segments_csv": None,
        "build_rest_segments_if_missing": False,
        "expected_segment_sec": 60.0,
        "temporal": {
            "detrend_enabled": True,
            "detrend_type": "linear",
            "filter_enabled": True,
            "filter_type": "bandpass",
            "trim_left_sec": 2.0,
            "trim_right_sec": 2.0,
            "low_hz": 0.08,
            "high_hz": 0.40,
            "order": 4,
            "synthetic_odd_pad_sec": 20.0,
            "variance_epsilon": 1e-12,
        },
        "spatial": {
            "scale": 0.50,
            "signal_resize": "mask_aware_area",
            # Fraction of the processed-grid pixel area that must contain finite
            # valid signal for the resized FC pixel to be finite.
            "min_valid_fraction": 0.50,
        },
        "homotopic": {
            "require_same_analysis_roi": True,
            "mutual_nearest_partner": True,
            "fisher_clip": 0.999999,
        },
        "global": {
            # Used only when Step6b global_analysis_hemisphere == "both".
            # separate_hemispheres: left pixels correlate only with left pool,
            #                       right pixels only with right pool.
            # whole_cortex: one bilateral pool; each pixel correlates with all
            #               other valid pixels across both hemispheres.
            # In both modes Step6b left/right masks are respected, therefore
            # midline_exclusion_px remains excluded from the pool.
            "both_hemispheres_mode": "separate_hemispheres",
            "correlation_block_size": 512,
            "fisher_clip": 0.999999,
            "min_pool_pixels": 2,
        },
        "final_grid_qc": {
            # These are additional diagnostic gates after Step6b. Step6b remains
            # authoritative for canonical/cohort eligibility.
            "min_roi_pixels": 5,
            "min_homotopic_pairs": 5,
            "min_final_grid_coverage_fraction": 0.25,
        },
        "output": {
            "overwrite": False,
            "save_geometry": True,
            "save_segment_maps": True,
            "save_stem_maps": True,
            "save_qc_png": True,
            "save_roi_traces_one_mouse": True,
            "save_roi_traces_batch": False,
        },
        "qc": {
            "map_percentiles": [2.0, 98.0],
            "trace_rois": None,
        },
        "batch": {
            "continue_on_error": True,
            "show_progress": True,
        },
    }


def ensure_step7_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Insert any missing functional-connectivity defaults into the supplied configuration.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    dict[str, Any]
        Input configuration augmented in place with missing Step7 defaults."""
    config.setdefault("functional_connectivity", {})
    _deep_fill_missing(config["functional_connectivity"], default_step7_config())
    return config


def step7_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return the functional_connectivity configuration section with defaults applied.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    dict[str, Any]
        Mutable ``functional_connectivity`` subsection of the configuration."""
    ensure_step7_defaults(config)
    return config["functional_connectivity"]


def build_step7_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create variant-specific output directories and return their paths.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    paths : dict[str, Path]
        Project paths returned by the upstream project/index builder.
    
    Returns
    -------
    dict[str, Path]
        Named absolute or relative paths for the variant output products.
    
    Notes
    -----
    Output layout: ``<out_root>/<folder_name>/<rest_variant>/<variant_id>/``. Specifying a
    subset adds a separate ``_subset_runs/<hash>`` directory."""
    cfg = step7_cfg(config)
    branch = str(cfg.get("rest_variant", "with_rest"))
    variant = str(cfg.get("active_variant_id", "hbt_bandpass"))
    if branch not in {"with_rest", "without_rest"}:
        raise ValueError(f"Invalid rest_variant: {branch}")
    if not variant or not all(ch.isalnum() or ch in "_-" for ch in variant):
        raise ValueError(f"Unsafe variant id: {variant!r}")
    root = Path(paths["out_root"]) / str(cfg.get("folder_name", "step7_functional_connectivity")) / branch / variant
    subset = cfg.get("_subset_output_key")
    if subset:
        root = root / "_subset_runs" / str(subset)
    tables = root / "tables"
    out = {
        "root": root,
        "tables_root": tables,
        "geometry_root": root / "fc_geometry",
        "segments_root": root / "segments",
        "stem_maps_root": root / "stem_maps",
        "roi_traces_root": root / "roi_traces",
        "qc_root": root / "qc",
        "step7_index_csv": tables / "step7_index.csv",
        "rest_used_csv": tables / "rest_segments_used.csv",
        "windows_csv": tables / "windows_used_60s.csv",
        "run_metadata_json": tables / "run_metadata.json",
        "segment_log_csv": tables / "segment_processing_log.csv",
        "homotopic_roi_csv": tables / "homotopic_roi_values.csv",
        "global_roi_csv": tables / "global_roi_values.csv",
        "roi_long_csv": tables / "fc_roi_values_long.csv",
        "errors_csv": tables / "step7_errors.csv",
        "run_summary_csv": tables / "step7_run_summary.csv",
    }
    for p in out.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return out


def safe_str(value: Any) -> str:
    """Convert a scalar to a string while mapping missing values to an empty string.
    
    Parameters
    ----------
    value : Any
        Scalar or missing value to normalize.
    
    Returns
    -------
    str
        Result of the documented operation."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def bool_value(value: Any) -> bool:
    """Interpret boolean scalars and common textual truth values.
    
    Parameters
    ----------
    value : Any
        Scalar or missing value to normalize.
    
    Returns
    -------
    bool
        Result of the documented operation."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return safe_str(value).strip().lower() in {"true", "1", "yes", "y"}


def safe_float(value: Any, default: float = np.nan) -> float:
    """Convert a value to a finite float or use the specified fallback.
    
    Parameters
    ----------
    value : Any
        Scalar or missing value to normalize.
    default : float
        Fallback returned when conversion fails or yields a nonfinite result.
    
    Returns
    -------
    float
        Result of the documented operation."""
    try:
        x = float(value)
        return x if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def now_iso() -> str:
    """Return the current local timestamp in ISO format.
    
    
    Returns
    -------
    str
        Result of the documented operation."""
    return _dt.datetime.now().isoformat(timespec="seconds")


# =============================================================================
# Read-only Finalizer REST windows; no QC reconciliation, no regeneration.
# =============================================================================


def default_final_rest_path(paths: dict[str, Path]) -> Path:
    """Locate the default read-only Finalizer REST-segment CSV.
    
    Parameters
    ----------
    paths : dict[str, Path]
        Project paths returned by the upstream project/index builder.
    
    Returns
    -------
    Path
        Result of the documented operation."""
    return Path(paths["out_root"]) / "behavior_review" / "rest_segments_final.csv"


def load_final_rest_segments_readonly(
    target_step7_index: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, Path]:
    """Validate and load accepted Awake REST windows without modifying the source CSV.
    
    Parameters
    ----------
    target_step7_index : pd.DataFrame
        Index of selected recordings whose Awake stems require accepted REST windows.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    paths : dict[str, Path]
        Project paths returned by the upstream project/index builder.
    
    Returns
    -------
    tuple[pd.DataFrame, Path]
        Validated accepted REST windows and path to the source CSV.
    
    Notes
    -----
    Only Awake recordings in the ``with_rest`` branch consume these accepted windows. The CSV
    must contain ``stem``, ``t0_sec`` and ``t1_sec``; windows must be positive, nonduplicated
    and within 0.75 seconds of the configured expected duration."""
    cfg = step7_cfg(config)
    chosen = cfg.get("rest_segments_csv") or s6b.cfg_get(
        config, "step6b_fc_geometry", "rest_segments_final_csv", default=None
    )
    if chosen:
        path = s6b.resolve_declared_path(str(chosen), config)
    else:
        path = default_final_rest_path(paths)
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"Accepted Finalizer REST CSV is required, read-only: {path}")
    path = Path(path)
    df = pd.read_csv(path)
    required = {"stem", "t0_sec", "t1_sec"}
    if missing := (required - set(df.columns)):
        raise KeyError(f"Finalizer REST CSV missing {sorted(missing)}: {path}")
    df = df.copy()
    df["t0_sec"] = pd.to_numeric(df["t0_sec"], errors="coerce")
    df["t1_sec"] = pd.to_numeric(df["t1_sec"], errors="coerce")
    if df[["t0_sec", "t1_sec"]].isna().any().any():
        raise ValueError(f"REST CSV contains non-numeric t0/t1: {path}")
    if (df["t0_sec"] < 0).any() or (df["t1_sec"] <= df["t0_sec"]).any():
        raise ValueError(f"REST CSV contains invalid time windows: {path}")
    expected = float(cfg.get("expected_segment_sec", 60.0))
    if not np.allclose((df["t1_sec"] - df["t0_sec"]).to_numpy(), expected, atol=0.75, rtol=0):
        raise ValueError(f"REST CSV has window(s) outside {expected}±0.75 sec: {path}")
    if "state" in df.columns and df["state"].astype(str).ne("Awake").any():
        raise ValueError("Finalizer REST CSV must contain Awake windows only")
    if df.duplicated(["stem", "t0_sec", "t1_sec"]).any():
        raise ValueError(f"Duplicate accepted REST windows in {path}")
    wanted = (target_step7_index[target_step7_index["state"].astype(str).eq("Awake")]["stem"].astype(str).unique()
              if not target_step7_index.empty else [])
    existing = set(df["stem"].astype(str))
    absent = sorted(set(wanted) - existing)
    if absent:
        raise RuntimeError(f"Step6b accepted Awake stems without Finalizer REST windows: {absent}")
    if "segment_number" not in df.columns:
        df = df.sort_values(["stem", "t0_sec", "t1_sec"]).copy()
        df["segment_number"] = df.groupby("stem").cumcount() + 1
    return df.reset_index(drop=True), path


# Compatibility alias for earlier notebooks: it NEVER builds or changes Finalizer REST.
load_or_build_rest_segments_final = load_final_rest_segments_readonly


def whole_recording_windows(stem: str, n_frames: int, fps: float, expected_sec: float) -> pd.DataFrame:
    """Partition a recording into consecutive nonoverlapping complete windows.
    
    Parameters
    ----------
    stem : str
        Unique recording stem identifier.
    n_frames : int
        Total number of acquired or processed frames.
    fps : float
        Acquisition frame rate in Hz (frames per second).
    expected_sec : float
        Nominal complete-window duration in seconds.
    
    Returns
    -------
    pd.DataFrame
        Table with ``stem``, window number, boundaries in seconds and source tag.
    
    Notes
    -----
    Frames per window are ``round(expected_sec * fps)``. Any incomplete trailing window is
    discarded; times are derived from frame indices, not assumed exact integer seconds."""
    per = int(round(float(expected_sec) * fps))
    if per < 4:
        raise ValueError("Window shorter than four frames")
    count = int(n_frames) // per
    return pd.DataFrame([
        {"stem": stem, "segment_number": i + 1, "t0_sec": (i * per) / fps,
         "t1_sec": ((i + 1) * per) / fps,
         "source": "full_recording_nonoverlap"}
        for i in range(count)
    ], columns=["stem", "segment_number", "t0_sec", "t1_sec", "source"])


def variant_config(config: dict[str, Any], variant_id: str, rest_variant: str | None = None) -> dict[str, Any]:
    """Create an independent configuration for one signal/filter variant.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    variant_id : str
        Identifier of the configured signal/filter combination.
    rest_variant : str | None
        ``with_rest`` or ``without_rest`` branch.
    
    Returns
    -------
    dict[str, Any]
        Deep-copied configuration with the selected signal, filter and REST branch.
    
    Notes
    -----
    Input config is deep-copied. Only ``dcHbT`` and ``dcHbT_vreg`` and filter types
    ``bandpass``/``lowpass`` are accepted."""
    out = copy.deepcopy(config)
    fc = step7_cfg(out)
    specs = fc.get("variants", {}).get("definitions", {})
    if variant_id not in specs:
        raise KeyError(f"Unconfigured Step7 variant {variant_id!r}; available: {list(specs)}")
    spec = specs[variant_id]
    signal = str(spec.get("signal", ""))
    if signal not in {"dcHbT", "dcHbT_vreg"}:
        raise ValueError(f"Unsupported Step7 signal: {signal!r}")
    filt = str(spec.get("filter_type", "")).lower()
    if filt not in {"bandpass", "lowpass"}:
        raise ValueError(f"Unsupported Step7 filter: {filt!r}")
    fc["active_variant_id"] = variant_id
    fc["signal"] = signal
    fc["temporal"]["filter_type"] = filt
    fc["temporal"]["filter_enabled"] = True
    if rest_variant is not None:
        fc["rest_variant"] = rest_variant
    if fc["rest_variant"] not in {"with_rest", "without_rest"}:
        raise ValueError(f"Unsupported Step6b rest variant: {fc['rest_variant']!r}")
    return out


def _analysis_fingerprint(config: dict[str, Any], row: pd.Series, segs: pd.DataFrame,
                          processed_path: Path, s6b_paths: dict[str, Path]) -> str:
    """Hash analysis settings, source metadata and selected time windows for reuse checks.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    segs : pd.DataFrame
        Sorted selected time-window table for the recording.
    processed_path : Path
        Path to the processed Step4 Zarr directory.
    s6b_paths : dict[str, Path]
        Path mapping for saved Step6b tables and geometry.
    
    Returns
    -------
    str
        Result of the documented operation.
    
    Notes
    -----
    The hash uses Step6b input paths, sizes and modification times, the processed Zarr
    path/mtime, selected windows and processing settings. It is a reuse safeguard rather than a
    content hash of every data byte."""
    fc = step7_cfg(config)
    # File source metadata prevents accidental reuse after a Zarr/Step6b update.
    stat = processed_path.stat()
    ms = str(row["mask_set"])
    gid = str(row["group_id"])
    stem = str(row["stem"])
    refstem = str(row.get("ref_stem", ""))
    mask_root = Path(s6b_paths["mask_sets_root"]) / s6b.safe_key(ms, gid)
    ref_dir = Path(s6b_paths["reference_root"]) / s6b.safe_key(gid, refstem)
    stem_dir = mask_root / "by_stem" / s6b.safe_key(stem)
    upstream = [
        mask_root / "analysis_roi_ref_buffered_repaired.npy",
        stem_dir / "bilateral_final_usable_ref.npy",
        stem_dir / "global_healthy_final_usable_ref.npy",
        ref_dir / "left_hemisphere_ref.npy",
        ref_dir / "right_hemisphere_ref.npy",
        Path(s6b_paths["selection_csv"]),
    ]
    provenance = [
        [str(p), int(p.stat().st_mtime_ns), int(p.stat().st_size)]
        if p.is_file() else [str(p), "MISSING"] for p in upstream
    ]
    payload = {
        "step6b_sources": provenance,
        "signal": fc["signal"], "rest_variant": fc["rest_variant"],
        "variant_id": fc["active_variant_id"],
        "temporal": fc["temporal"], "spatial": fc["spatial"],
        "homotopic": fc["homotopic"], "global": fc["global"],
        "final_grid_qc": fc["final_grid_qc"],
        "step6b_config": config.get("step6b_fc_geometry", {}),
        "mask_set": str(row["mask_set"]), "stem": str(row["stem"]),
        "windows": [[round(float(t), 7), round(float(u), 7)] for t, u in zip(segs["t0_sec"], segs["t1_sec"])],
        "zarr": [str(processed_path), int(stat.st_mtime_ns)],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# =============================================================================
# Step7 index / Step6b tables
# =============================================================================


def _selected_mask_sets(config: dict[str, Any], explicit: Sequence[str] | None) -> list[str]:
    """Resolve explicitly requested mask sets or the configured default list.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    explicit : Sequence[str] | None
        Explicitly requested mask-set identifiers, if any.
    
    Returns
    -------
    list[str]
        Result of the documented operation."""
    if explicit:
        return [str(x) for x in explicit]
    raw = step7_cfg(config).get("mask_sets", ["stroke_longitudinal"])
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def build_step7_index(
    config_path: str | Path,
    *,
    mask_sets: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path], dict[str, Path], dict[str, pd.DataFrame]]:
    """Construct the Step7 stem index from the authoritative Step6b selection.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_sets : Sequence[str] | None
        Optional mask-set names to select; ``None`` uses configuration defaults.
    mice : Sequence[str] | None
        Optional mouse IDs used to restrict the selected recordings.
    stems : Sequence[str] | None
        Optional recording stems used to restrict the selection.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    
    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any], dict[str, Path], dict[str, Path], dict[str, pd.DataFrame]]
        Selected index, updated configuration, project paths, Step6b paths and loaded Step6b tables.
    
    Notes
    -----
    Step6b selection-group completeness is checked against the full original Step6b selection
    before filtering mice or stems. Only successful selection rows from complete groups enter
    Step7."""
    step6_df, config, paths = s6b.build_step6_index(
        config_path,
        include_bad=True,
        filters={},
        require_response_roi=False,
    )
    s6b.ensure_step6b_defaults(config)
    ensure_step7_defaults(config)
    _deep_update(config["functional_connectivity"], runtime_overrides)

    branch = str(step7_cfg(config).get("rest_variant", "with_rest"))
    if branch not in {"with_rest", "without_rest"}:
        raise ValueError(f"Invalid Step6b rest_variant={branch!r}")
    config["step6b_fc_geometry"]["active_rest_variant"] = branch
    s6b_paths = s6b.build_step6b_paths(config, paths)
    tables = s6b.load_step6b_tables(config_path, rest_variant=branch)
    selection = tables.get("selection", pd.DataFrame()).copy()
    if selection.empty:
        raise RuntimeError("Step6b mask_set_selection.csv is empty. Run Step6b v4.10 first.")

    if "rest_variant" not in selection.columns or not selection["rest_variant"].astype(str).eq(branch).all():
        raise RuntimeError(f"Step6b selection is not exclusively rest_variant={branch}")
    for field in ("mask_set", "group_id", "slot", "stem", "status"):
        if field not in selection.columns:
            raise KeyError(f"Step6b selection missing {field!r}")
    if selection.duplicated(["mask_set", "group_id", "slot"]).any():
        raise RuntimeError("Step6b has duplicate mask_set × group_id × slot keys")
    skipped = tables.get("skipped_mask_sets", pd.DataFrame())
    if not skipped.empty and skipped.duplicated(["mask_set", "group_id"]).any():
        raise RuntimeError("Step6b has duplicate skipped_mask_sets keys")
    # Apply the identical Step6b slot-policy rule to the ORIGINAL full selection,
    # before filtering mice/stems; no variant_keys/protocol resolution here.
    keep = set()
    for (ms, gid), sg in selection.groupby(["mask_set", "group_id"], sort=False):
        complete = s6b.selection_group_complete(sg, config)
        skipped_here = (not skipped.empty and ((skipped["mask_set"].astype(str) == str(ms)) &
                                         (skipped["group_id"].astype(str) == str(gid))).any())
        if complete and skipped_here:
            raise RuntimeError(f"Step6b contradictory selection/skip: {ms}/{gid}")
        if (not complete) and (not skipped_here):
            raise RuntimeError(f"Step6b incomplete selection not declared skipped: {ms}/{gid}")
        if complete:
            keep.add((str(ms), str(gid)))
    selection = selection[selection.apply(lambda r: (str(r["mask_set"]), str(r["group_id"])) in keep and str(r["status"]) == "ok", axis=1)].copy()
    wanted_sets = set(_selected_mask_sets(config, mask_sets))
    selection = selection[selection["mask_set"].astype(str).isin(wanted_sets)].copy()

    if mice:
        wanted = {str(v) for v in mice}
        selection = selection[selection["mouse_id"].astype(str).isin(wanted)].copy()
    if stems:
        wanted = {str(v) for v in stems}
        selection = selection[selection["stem"].astype(str).isin(wanted)].copy()

    rows: list[dict[str, Any]] = []
    for _, sr in selection.iterrows():
        stem = safe_str(sr.get("stem"))
        gid = safe_str(sr.get("group_id"))
        hit = step6_df[
            step6_df["stem"].astype(str).eq(stem)
            & step6_df["group_id"].astype(str).eq(gid)
        ]
        if len(hit) != 1:
            raise RuntimeError(
                f"mask_set_selection row does not map uniquely to Step6 index: group_id={gid}, stem={stem}, n={len(hit)}"
            )
        rr = hit.iloc[0]
        rec = rr.to_dict()
        rec.update(
            {
                "mask_set": safe_str(sr.get("mask_set")),
                "slot": safe_str(sr.get("slot")),
                "experimental_group": safe_str(sr.get("experimental_group", rec.get("experimental_group", ""))),
                "comparison_group": safe_str(sr.get("comparison_group", "")),
                "rest_variant": branch,
                "selection_status": safe_str(sr.get("status", "ok")),
            }
        )
        rows.append(rec)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, config, paths, s6b_paths, tables
    sort_cols = [c for c in ["mask_set", "mouse_id", "slot", "stem"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out, config, paths, s6b_paths, tables


# =============================================================================
# Geometry helpers
# =============================================================================


def resize_bool_nearest(mask: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask using nearest-neighbor interpolation.
    
    Parameters
    ----------
    mask : np.ndarray
        Binary input mask.
    shape_yx : tuple[int, int]
        Output image shape as ``(height, width)``.
    
    Returns
    -------
    np.ndarray
        Result of the documented operation."""
    arr = np.asarray(mask, dtype=bool)
    h, w = map(int, shape_yx)
    if arr.shape == (h, w):
        return arr
    return cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def resize_labels_nearest(labels: np.ndarray, shape_yx: tuple[int, int]) -> np.ndarray:
    """Resize an integer ROI-label image without interpolating label identities.
    
    Parameters
    ----------
    labels : np.ndarray
        Two-dimensional integer anatomical ROI label image.
    shape_yx : tuple[int, int]
        Output image shape as ``(height, width)``.
    
    Returns
    -------
    np.ndarray
        Result of the documented operation."""
    arr = np.asarray(labels, dtype=np.int32)
    h, w = map(int, shape_yx)
    if arr.shape == (h, w):
        return arr
    out = cv2.resize(arr, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.rint(out).astype(np.int32)


def infer_stem_fullres_shape(row: pd.Series, fallback: tuple[int, int]) -> tuple[int, int]:
    """Identify the original image height and width from available registration images.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    fallback : tuple[int, int]
        Reference image shape used when no registration image is readable.
    
    Returns
    -------
    tuple[int, int]
        Result of the documented operation."""
    candidates = [
        "moving_registration_path_resolved",
        "moving_registration_path",
        "fixed_registration_path_resolved",
        "fixed_registration_path",
        "ref_image_path",
    ]
    for col in candidates:
        text = safe_str(row.get(col)).strip()
        if not text:
            continue
        p = Path(text)
        if not p.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim >= 2:
            return int(img.shape[0]), int(img.shape[1])
    return tuple(map(int, fallback))


def warp_ref_to_stem_nearest(
    arr: np.ndarray,
    tform_stem_to_ref: Any,
    output_shape: tuple[int, int],
    *,
    as_bool: bool,
) -> np.ndarray:
    # skimage.warp inverse_map maps output(stem) -> input(reference). The saved
    # tform already maps stem -> reference, so it is passed directly here.
    """Warp a reference-space mask or label image into full-resolution stem coordinates.
    
    Parameters
    ----------
    arr : np.ndarray
        Image or numerical array used by the operation.
    tform_stem_to_ref : Any
        Saved spatial transform mapping stem pixel coordinates to reference coordinates.
    output_shape : tuple[int, int]
        Destination image shape in ``(y, x)`` pixels.
    as_bool : bool
        Return a binary mask if true; otherwise return rounded integer labels.
    
    Returns
    -------
    np.ndarray
        Result of the documented operation.
    
    Notes
    -----
    ``skimage.transform.warp`` receives the saved stem-to-reference transform as
    ``inverse_map`` because output stem pixels must sample the input reference image.
    Nearest-neighbor resampling preserves binary and label identities."""
    out = warp(
        np.asarray(arr, dtype=np.float32),
        inverse_map=tform_stem_to_ref,
        output_shape=tuple(map(int, output_shape)),
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0,
    )
    if as_bool:
        return out > 0.5
    return np.rint(out).astype(np.int32)


def scale_xy_between_shapes(
    xy: np.ndarray,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> np.ndarray:
    """Rescale x/y coordinates using endpoint-aligned pixel-grid dimensions.
    
    Parameters
    ----------
    xy : np.ndarray
        Array of points whose last dimension is ``(x, y)``.
    src_shape : tuple[int, int]
        Source grid shape as ``(height, width)``.
    dst_shape : tuple[int, int]
        Destination grid shape as ``(height, width)``.
    
    Returns
    -------
    np.ndarray
        Result of the documented operation.
    
    Notes
    -----
    The coordinate convention is ``[..., 0] = x`` and ``[..., 1] = y``; shapes are ``(height,
    width)`` in pixels."""
    pts = np.asarray(xy, dtype=float).copy()
    sh, sw = map(int, src_shape)
    dh, dw = map(int, dst_shape)
    sx = (dw - 1) / max(sw - 1, 1)
    sy = (dh - 1) / max(sh - 1, 1)
    pts[..., 0] *= sx
    pts[..., 1] *= sy
    return pts


def transformed_midline_on_fc_grid(
    row: pd.Series,
    config: dict[str, Any],
    tform_stem_to_ref: Any,
    stem_fullres_shape: tuple[int, int],
    fc_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform the reference AP midline into an origin and direction on the FC grid.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    tform_stem_to_ref : Any
        Saved spatial transform mapping stem pixel coordinates to reference coordinates.
    stem_fullres_shape : tuple[int, int]
        Full-resolution stem image shape as ``(y, x)``.
    fc_shape : tuple[int, int]
        Target FC image shape as ``(y, x)`` pixel counts.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Result of the documented operation.
    
    Notes
    -----
    The reference AP axis is transformed to stem coordinates with the inverse affine transform
    and rescaled to the FC pixel grid."""
    basis = s6b.reference_basis_from_row(row, config)
    origin = np.asarray(basis["origin_xy"], dtype=float)
    ap = np.asarray(basis["ap_hat_xy"], dtype=float)
    length = max(stem_fullres_shape) * 0.75
    ref_pts = np.vstack([origin - ap * length, origin + ap * length])
    stem_pts = np.asarray(tform_stem_to_ref.inverse(ref_pts), dtype=float)
    fc_pts = scale_xy_between_shapes(stem_pts, stem_fullres_shape, fc_shape)
    direction = fc_pts[1] - fc_pts[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise RuntimeError("Transformed Step3 midline axis collapsed on FC grid")
    return fc_pts[0], direction / norm


def reflect_xy_across_line(points_xy: np.ndarray, origin_xy: np.ndarray, line_hat_xy: np.ndarray) -> np.ndarray:
    """Reflect two-dimensional points across the anatomical midline.
    
    Parameters
    ----------
    points_xy : np.ndarray
        Points with a trailing coordinate axis ``(x, y)``.
    origin_xy : np.ndarray
        A point on the reflection axis in ``(x, y)`` coordinates.
    line_hat_xy : np.ndarray
        Unit direction vector of the reflection axis.
    
    Returns
    -------
    np.ndarray
        Result of the documented operation.
    
    Notes
    -----
    ``origin_xy`` is a point on the line and ``line_hat_xy`` is assumed to be a unit direction
    vector."""
    pts = np.asarray(points_xy, dtype=float)
    origin = np.asarray(origin_xy, dtype=float)
    u = np.asarray(line_hat_xy, dtype=float)
    rel = pts - origin
    proj = origin + np.outer(rel @ u, u)
    return 2.0 * proj - pts


def _resolve_side(choice: str, healthy_side: str, lesion_side: str) -> str:
    """Resolve anatomical or lesion-relative hemisphere selectors.
    
    Parameters
    ----------
    choice : str
        Requested anatomical side or lesion-relative selector.
    healthy_side : str
        Anatomical side assigned to the healthy hemisphere.
    lesion_side : str
        Anatomical side assigned to the affected hemisphere.
    
    Returns
    -------
    str
        Result of the documented operation."""
    choice = safe_str(choice).strip().lower()
    if choice == "healthy":
        return healthy_side
    if choice == "lesion":
        return lesion_side
    if choice in {"left", "right"}:
        return choice
    raise ValueError(f"Hemisphere choice must be healthy/lesion/left/right, got {choice!r}")


def homotopic_map_side_for_mask_set(config: dict[str, Any], mask_set: str, healthy_side: str, lesion_side: str) -> str:
    """Resolve the hemisphere receiving homotopic correlation values.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    mask_set : str
        Name of the selected Step6b mask set.
    healthy_side : str
        Anatomical side assigned to the healthy hemisphere.
    lesion_side : str
        Anatomical side assigned to the affected hemisphere.
    
    Returns
    -------
    str
        Result of the documented operation."""
    local = s6b.cfg_get(
        config, "step6b_fc_geometry", "mask_sets", mask_set, "homotopic_map_hemisphere", default=None
    )
    choice = safe_str(local).strip().lower() if local is not None and safe_str(local).strip() else safe_str(
        s6b.cfg_get(config, "step6b_fc_geometry", "hemisphere", "homotopic_map_hemisphere", default="lesion")
    ).strip().lower()
    return _resolve_side(choice, healthy_side, lesion_side)


def normalize_global_both_mode(config: dict[str, Any]) -> str:
    """Normalize configured bilateral GBC pool modes and their aliases.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    str
        Result of the documented operation."""
    raw = safe_str(
        step7_cfg(config).get("global", {}).get(
            "both_hemispheres_mode", "separate_hemispheres"
        )
    ).strip().lower()
    aliases = {
        "separate": "separate_hemispheres",
        "separate_hemisphere": "separate_hemispheres",
        "separate_hemispheres": "separate_hemispheres",
        "two_separate": "separate_hemispheres",
        "whole": "whole_cortex",
        "bilateral": "whole_cortex",
        "whole_brain": "whole_cortex",
        "whole_cortex": "whole_cortex",
    }
    if raw not in aliases:
        raise ValueError(
            "functional_connectivity.global.both_hemispheres_mode must be "
            "'separate_hemispheres' or 'whole_cortex'"
        )
    return aliases[raw]


def global_plan_for_mask_set(
    config: dict[str, Any],
    mask_set: str,
    healthy_side: str,
    lesion_side: str,
) -> dict[str, Any]:
    """Select the hemispheric pool layout for one Step6b mask set.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    mask_set : str
        Name of the selected Step6b mask set.
    healthy_side : str
        Anatomical side assigned to the healthy hemisphere.
    lesion_side : str
        Anatomical side assigned to the affected hemisphere.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation.
    
    Notes
    -----
    The source of hemisphere selection is Step6b. Bilateral GBC may use independent left/right
    pools or a single bilateral ``whole_cortex`` pool; Step6b midline exclusions remain in
    force."""
    choice = safe_str(
        s6b.global_analysis_choice_for_mask_set(config, mask_set)
    ).strip().lower()
    if choice == "both":
        mode = normalize_global_both_mode(config)
        return {
            "analysis_choice": "both",
            "both_mode": mode,
            # ROI outcomes remain side-specific because Step6b coverage/pass
            # tables are side-specific even when the correlation pool is bilateral.
            "roi_sides": ["left", "right"],
            "pool_keys": (
                ["left", "right"]
                if mode == "separate_hemispheres"
                else ["whole_cortex"]
            ),
        }

    side = _resolve_side(choice, healthy_side, lesion_side)
    return {
        "analysis_choice": choice,
        "both_mode": "single_hemisphere",
        "roi_sides": [side],
        "pool_keys": [side],
    }


def _global_pool_key_for_roi_side(geometry: "StemGeometry", side: str) -> str:
    """Find the GBC pool containing the requested ROI hemisphere.
    
    Parameters
    ----------
    geometry : 'StemGeometry'
        Recording-specific ``StemGeometry`` object on the FC grid.
    side : str
        Left/right anatomical side of the selected ROI or GBC pool.
    
    Returns
    -------
    str
        Result of the documented operation."""
    if (
        geometry.global_analysis_choice == "both"
        and geometry.global_both_mode == "whole_cortex"
    ):
        return "whole_cortex"
    return side


@dataclass
class StemGeometry:
    """Frozen geometry for one stem, carrying Step6b anatomy into FC space.
    
        Attributes
        ----------
        fc_shape : tuple of int
            Height and width of the reduced connectivity grid in pixels.
        bilateral_fc, left_fc, right_fc : numpy.ndarray
            Cortical and hemispheric validity masks with dimensions ``(y, x)``.
        labels_fc : numpy.ndarray
            Repaired analytical ROI labels transformed into FC coordinates.
        pair_src_flat, pair_dst_flat : numpy.ndarray
            Row-major flattened indices of contralateral homotopic pixel partners.
    
        Notes
        -----
        Anatomical left and right are taken from Step6b masks; do not infer them
        from image columns. Both technical validity and midline exclusion apply."""
    mask_set: str
    stem: str
    group_id: str
    ref_stem: str
    mouse_id: str
    healthy_side: str
    lesion_side: str
    homotopic_map_side: str
    global_analysis_choice: str
    global_both_mode: str
    global_roi_sides: list[str]
    processed_shape: tuple[int, int]
    fc_shape: tuple[int, int]
    stem_fullres_shape: tuple[int, int]
    bilateral_processed: np.ndarray
    bilateral_fc: np.ndarray
    left_fc: np.ndarray
    right_fc: np.ndarray
    global_pool_masks_fc: dict[str, np.ndarray]
    labels_fc: np.ndarray
    labels_source_path: str
    midline_origin_xy: np.ndarray
    midline_hat_xy: np.ndarray
    transform_source: str
    transform_meta: dict[str, float]
    pair_src_flat: np.ndarray
    pair_dst_flat: np.ndarray
    pair_roi_id: np.ndarray
    pair_potential_count_by_roi: dict[int, int]
    pair_geometry_count_by_roi: dict[int, int]



def build_homotopic_pairs_fc(
    labels_fc: np.ndarray,
    bilateral_fc: np.ndarray,
    left_fc: np.ndarray,
    right_fc: np.ndarray,
    *,
    map_side: str,
    midline_origin_xy: np.ndarray,
    midline_hat_xy: np.ndarray,
    require_same_roi: bool = True,
    mutual_nearest: bool = True,
) -> dict[str, Any]:
    """Construct valid contralateral pixel pairs on the FC grid.
    
    Parameters
    ----------
    labels_fc : np.ndarray
        Integer analytical ROI labels on the FC grid ``(y, x)``.
    bilateral_fc : np.ndarray
        Binary valid bilateral cortex mask on the FC grid.
    left_fc : np.ndarray
        Binary mask of the left hemisphere on the FC grid.
    right_fc : np.ndarray
        Binary mask of the right hemisphere on the FC grid.
    map_side : str
        Hemisphere on which homotopic source pixels are located.
    midline_origin_xy : np.ndarray
        Origin of the transformed midline on the FC grid.
    midline_hat_xy : np.ndarray
        Unit vector along the transformed anatomical midline.
    require_same_roi : bool
        Require mirrored partners to have the same analytical ROI label.
    mutual_nearest : bool
        Require reciprocal nearest-neighbor pixel correspondence.
    
    Returns
    -------
    dict[str, Any]
        Mapping containing source and destination flattened indices, ROI IDs, and geometric candidate counts per ROI.
    
    Notes
    -----
    Pixel centers are reflected across the transformed anatomical midline, then rounded to
    nearest integer pixels. Optional constraints require equal analysis ROI labels and
    reciprocal nearest partner mapping. Returned flattened indices use row-major ``(y,x)``
    order."""
    h, w = labels_fc.shape
    source_hemi = left_fc if map_side == "left" else right_fc
    opposite_hemi = right_fc if map_side == "left" else left_fc

    sy, sx = np.nonzero(source_hemi & (labels_fc > 0))
    if sy.size == 0:
        return {
            "src_flat": np.empty(0, dtype=np.int64),
            "dst_flat": np.empty(0, dtype=np.int64),
            "roi_id": np.empty(0, dtype=np.int32),
            "potential_count_by_roi": {},
            "geometry_count_by_roi": {},
        }

    src_xy = np.column_stack([sx.astype(float), sy.astype(float)])
    mirrored = reflect_xy_across_line(src_xy, midline_origin_xy, midline_hat_xy)
    dx = np.rint(mirrored[:, 0]).astype(int)
    dy = np.rint(mirrored[:, 1]).astype(int)
    inside = (dx >= 0) & (dx < w) & (dy >= 0) & (dy < h)

    partner_roi = np.zeros(sy.size, dtype=np.int32)
    partner_opposite = np.zeros(sy.size, dtype=bool)
    partner_valid = np.zeros(sy.size, dtype=bool)
    partner_roi[inside] = labels_fc[dy[inside], dx[inside]]
    partner_opposite[inside] = opposite_hemi[dy[inside], dx[inside]]
    partner_valid[inside] = bilateral_fc[dy[inside], dx[inside]]

    src_roi = labels_fc[sy, sx].astype(np.int32)
    potential = inside & partner_opposite & (partner_roi > 0)
    if require_same_roi:
        potential &= partner_roi == src_roi

    if mutual_nearest and np.any(potential):
        idx = np.flatnonzero(potential)
        back_xy = reflect_xy_across_line(
            np.column_stack([dx[idx].astype(float), dy[idx].astype(float)]),
            midline_origin_xy,
            midline_hat_xy,
        )
        back_x = np.rint(back_xy[:, 0]).astype(int)
        back_y = np.rint(back_xy[:, 1]).astype(int)
        mutual = (back_x == sx[idx]) & (back_y == sy[idx])
        tmp = np.zeros_like(potential)
        tmp[idx[mutual]] = True
        potential &= tmp

    valid_geometry = potential & bilateral_fc[sy, sx] & partner_valid

    potential_count_by_roi: dict[int, int] = {}
    geometry_count_by_roi: dict[int, int] = {}
    for rid in [int(v) for v in np.unique(src_roi[potential]) if int(v) > 0]:
        potential_count_by_roi[rid] = int(np.sum(potential & (src_roi == rid)))
        geometry_count_by_roi[rid] = int(np.sum(valid_geometry & (src_roi == rid)))

    keep = np.flatnonzero(valid_geometry)
    src_flat = np.ravel_multi_index((sy[keep], sx[keep]), (h, w)).astype(np.int64)
    dst_flat = np.ravel_multi_index((dy[keep], dx[keep]), (h, w)).astype(np.int64)
    roi_id = src_roi[keep].astype(np.int32)

    return {
        "src_flat": src_flat,
        "dst_flat": dst_flat,
        "roi_id": roi_id,
        "potential_count_by_roi": potential_count_by_roi,
        "geometry_count_by_roi": geometry_count_by_roi,
    }


def _hemisphere_assignment(tables: dict[str, pd.DataFrame], group_id: str, config: dict[str, Any]) -> tuple[str, str]:
    """Identify lesion and healthy hemispheres from Step6b metadata or fallback settings.
    
    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    group_id : str
        Identifier of the Step6b registration group.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    tuple[str, str]
        Result of the documented operation."""
    hemi = tables.get("hemisphere", pd.DataFrame())
    if not hemi.empty and "group_id" in hemi.columns:
        hit = hemi[hemi["group_id"].astype(str).eq(group_id)]
        if not hit.empty:
            lesion = safe_str(hit.iloc[-1].get("lesion_side", "")).lower()
            healthy = safe_str(hit.iloc[-1].get("healthy_side", "")).lower()
            if lesion in {"left", "right"} and healthy in {"left", "right"}:
                return lesion, healthy
    lesion = safe_str(s6b.cfg_get(config, "step6b_fc_geometry", "hemisphere", "lesion_side", default="right")).lower()
    if lesion not in {"left", "right"}:
        lesion = "right"
    return lesion, ("left" if lesion == "right" else "right")


def load_stem_geometry(
    row: pd.Series,
    mask_set: str,
    config: dict[str, Any],
    paths: dict[str, Path],
    s6b_paths: dict[str, Path],
    tables: dict[str, pd.DataFrame],
    processed_shape: tuple[int, int],
) -> StemGeometry:
    """Load and transform the Step6b ROI geometry required for one recording.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    mask_set : str
        Name of the selected Step6b mask set.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    paths : dict[str, Path]
        Project paths returned by the upstream project/index builder.
    s6b_paths : dict[str, Path]
        Path mapping for saved Step6b tables and geometry.
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    processed_shape : tuple[int, int]
        Processed signal image shape as ``(y, x)`` pixels.
    
    Returns
    -------
    StemGeometry
        Fully assembled ``StemGeometry`` with transformed masks and pixel pairings.
    
    Notes
    -----
    The repaired Step6b label product ``analysis_roi_ref_buffered_repaired.npy`` is mandatory;
    there is no fallback to legacy labels. Saved reference-space masks are transformed into
    stem and processed grids, then into the FC grid."""
    cfg = step7_cfg(config)
    spatial_scale = float(cfg["spatial"].get("scale", 0.5))
    if spatial_scale <= 0:
        raise ValueError("functional_connectivity.spatial.scale must be > 0")
    fc_shape = (
        max(1, int(round(processed_shape[0] * spatial_scale))),
        max(1, int(round(processed_shape[1] * spatial_scale))),
    )

    stem = safe_str(row.get("stem"))
    gid = safe_str(row.get("group_id"))
    ref_stem = safe_str(row.get("ref_stem"))
    mouse = safe_str(row.get("mouse_id")) or gid

    mask_set_dir = Path(s6b_paths["mask_sets_root"]) / s6b.safe_key(mask_set, gid)
    stem_dir = mask_set_dir / "by_stem" / s6b.safe_key(stem)
    ref_dir = Path(s6b_paths["reference_root"]) / s6b.safe_key(gid, ref_stem)

    bilateral_path = stem_dir / "bilateral_final_usable_ref.npy"
    # Current Step6b authoritative repaired-before-erosion analytical atlas.
    labels_path = mask_set_dir / "analysis_roi_ref_buffered_repaired.npy"
    left_path = ref_dir / "left_hemisphere_ref.npy"
    right_path = ref_dir / "right_hemisphere_ref.npy"
    global_healthy_path = stem_dir / "global_healthy_final_usable_ref.npy"

    required = [bilateral_path, labels_path, left_path, right_path, global_healthy_path]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                f"Required current Step6b geometry missing: {p}. "
                "Step7 v4 does not fall back to legacy refspace_broad_buffered_qc.npy."
            )

    bilateral_ref = np.load(bilateral_path).astype(bool)
    labels_ref = np.load(labels_path).astype(np.int32)
    left_ref = np.load(left_path).astype(bool)
    right_ref = np.load(right_path).astype(bool)
    global_healthy_ref = np.load(global_healthy_path).astype(bool)
    ref_shape = tuple(bilateral_ref.shape)
    if not (
        labels_ref.shape == left_ref.shape == right_ref.shape
        == bilateral_ref.shape == global_healthy_ref.shape
    ):
        raise ValueError("Current Step6b reference geometry arrays have inconsistent shapes")

    manual = s6b.load_manual_registration_overrides(config, paths)
    tform, transform_source, transform_meta = s6b.effective_tform_stem_to_ref(
        row, manual, config
    )
    stem_fullres_shape = infer_stem_fullres_shape(row, ref_shape)

    bilateral_full = warp_ref_to_stem_nearest(
        bilateral_ref, tform, stem_fullres_shape, as_bool=True
    )
    labels_full = warp_ref_to_stem_nearest(
        labels_ref, tform, stem_fullres_shape, as_bool=False
    )
    left_full = warp_ref_to_stem_nearest(
        left_ref, tform, stem_fullres_shape, as_bool=True
    )
    right_full = warp_ref_to_stem_nearest(
        right_ref, tform, stem_fullres_shape, as_bool=True
    )
    global_healthy_full = warp_ref_to_stem_nearest(
        global_healthy_ref, tform, stem_fullres_shape, as_bool=True
    )

    bilateral_processed = resize_bool_nearest(bilateral_full, processed_shape)
    labels_processed = resize_labels_nearest(labels_full, processed_shape)
    left_processed = resize_bool_nearest(left_full, processed_shape)
    right_processed = resize_bool_nearest(right_full, processed_shape)
    global_healthy_processed = resize_bool_nearest(global_healthy_full, processed_shape)

    bilateral_fc = resize_bool_nearest(bilateral_processed, fc_shape)
    labels_fc = resize_labels_nearest(labels_processed, fc_shape)
    left_fc = resize_bool_nearest(left_processed, fc_shape)
    right_fc = resize_bool_nearest(right_processed, fc_shape)
    global_healthy_fc = resize_bool_nearest(global_healthy_processed, fc_shape)

    # left_fc/right_fc are anatomical side masks that already contain Step6b
    # midline_exclusion_px. Keep them separate from bilateral technical validity.
    lesion_side, healthy_side = _hemisphere_assignment(tables, gid, config)
    hom_side = homotopic_map_side_for_mask_set(
        config, mask_set, healthy_side, lesion_side
    )
    plan = global_plan_for_mask_set(
        config, mask_set, healthy_side, lesion_side
    )

    left_valid = bilateral_fc & left_fc
    right_valid = bilateral_fc & right_fc
    # Constrain the explicit healthy product by the transformed hemisphere mask
    # to protect the midline exclusion after nearest-neighbour resampling.
    healthy_hemi_fc = left_fc if healthy_side == "left" else right_fc
    global_healthy_fc = global_healthy_fc & healthy_hemi_fc & bilateral_fc

    side_valid = {"left": left_valid, "right": right_valid}
    side_valid[healthy_side] = global_healthy_fc

    global_pool_masks: dict[str, np.ndarray] = {}
    if plan["analysis_choice"] == "both":
        if plan["both_mode"] == "separate_hemispheres":
            global_pool_masks = {
                "left": left_valid,
                "right": right_valid,
            }
        else:
            # A single bilateral pool. The union is explicitly made from
            # left/right masks, so Step6b midline_exclusion_px remains excluded.
            global_pool_masks = {
                "whole_cortex": bilateral_fc & (left_fc | right_fc)
            }
    else:
        side = plan["roi_sides"][0]
        global_pool_masks = {side: side_valid[side]}

    mid_origin, mid_hat = transformed_midline_on_fc_grid(
        row, config, tform, stem_fullres_shape, fc_shape
    )
    pair = build_homotopic_pairs_fc(
        labels_fc,
        bilateral_fc,
        left_fc,
        right_fc,
        map_side=hom_side,
        midline_origin_xy=mid_origin,
        midline_hat_xy=mid_hat,
        require_same_roi=bool(
            cfg["homotopic"].get("require_same_analysis_roi", True)
        ),
        mutual_nearest=bool(
            cfg["homotopic"].get("mutual_nearest_partner", True)
        ),
    )

    return StemGeometry(
        mask_set=mask_set,
        stem=stem,
        group_id=gid,
        ref_stem=ref_stem,
        mouse_id=mouse,
        healthy_side=healthy_side,
        lesion_side=lesion_side,
        homotopic_map_side=hom_side,
        global_analysis_choice=plan["analysis_choice"],
        global_both_mode=plan["both_mode"],
        global_roi_sides=list(plan["roi_sides"]),
        processed_shape=tuple(map(int, processed_shape)),
        fc_shape=tuple(map(int, fc_shape)),
        stem_fullres_shape=stem_fullres_shape,
        bilateral_processed=bilateral_processed,
        bilateral_fc=bilateral_fc,
        left_fc=left_fc,
        right_fc=right_fc,
        global_pool_masks_fc=global_pool_masks,
        labels_fc=labels_fc,
        labels_source_path=str(labels_path),
        midline_origin_xy=np.asarray(mid_origin, dtype=float),
        midline_hat_xy=np.asarray(mid_hat, dtype=float),
        transform_source=transform_source,
        transform_meta=transform_meta,
        pair_src_flat=pair["src_flat"],
        pair_dst_flat=pair["dst_flat"],
        pair_roi_id=pair["roi_id"],
        pair_potential_count_by_roi=pair["potential_count_by_roi"],
        pair_geometry_count_by_roi=pair["geometry_count_by_roi"],
    )


# =============================================================================
# Signal preprocessing
# =============================================================================


def make_bandpass_sos(config: dict[str, Any], fps: float) -> np.ndarray | None:
    """Design the configured Butterworth band-pass or low-pass filter in SOS form.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    fps : float
        Acquisition frame rate in Hz (frames per second).
    
    Returns
    -------
    np.ndarray | None
        Butterworth second-order-section coefficients for the configured filter.
    
    Notes
    -----
    Despite the historical function name, ``temporal.filter_type`` selects either a Butterworth
    low-pass or a Butterworth band-pass filter. Frequencies are in Hz and ``fps`` is samples
    per second."""
    tc = step7_cfg(config)["temporal"]
    if not bool(tc.get("filter_enabled", True)):
        raise ValueError("Step7 variants require a filter")
    filt = str(tc.get("filter_type", "bandpass"))
    high = float(tc.get("high_hz", 0.4))
    order = int(tc.get("order", 4))
    if not (order >= 1 and 0 < high < fps/2):
        raise ValueError(f"Invalid Step7 filter high={high}, order={order}, fps={fps}")
    if filt == "lowpass":
        return butter(order, high, btype="lowpass", fs=fps, output="sos")
    if filt == "bandpass":
        low = float(tc.get("low_hz", 0.08))
        if not (0 < low < high):
            raise ValueError(f"Invalid Step7 bandpass {low}-{high} Hz")
        return butter(order, [low, high], btype="bandpass", fs=fps, output="sos")
    raise ValueError(f"Invalid Step7 filter_type={filt!r}")


def mask_aware_resize_movie_area(
    movie: np.ndarray,
    valid_mask_processed: np.ndarray,
    fc_shape: tuple[int, int],
    *,
    min_valid_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-downsample a movie without averaging invalid or background pixels into cortex.
    
    Parameters
    ----------
    movie : np.ndarray
        Hemodynamic movie with axes ``(time, y, x)``.
    valid_mask_processed : np.ndarray
        Binary cortex-valid mask on the processed signal grid.
    fc_shape : tuple[int, int]
        Target FC image shape as ``(y, x)`` pixel counts.
    min_valid_fraction : float
        Minimum finite source-area coverage required for a target pixel.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Downsampled movie ``(time, y, x)`` and static source-mask coverage fraction ``(y, x)``.
    
    Notes
    -----
    The movie has axes ``(time, y, x)``. A resized pixel is NaN unless enough of its source
    area contains static-mask-valid, finite data. The companion array is static mask coverage
    fraction on the FC grid."""
    arr = np.asarray(movie, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected movie (time,y,x), got {arr.shape}")
    valid_static = np.asarray(valid_mask_processed, dtype=bool)
    if valid_static.shape != arr.shape[1:]:
        raise ValueError("valid_mask_processed shape does not match movie")
    h, w = map(int, fc_shape)
    out = np.full((arr.shape[0], h, w), np.nan, dtype=np.float32)
    min_frac = float(min_valid_fraction)
    if not (0 <= min_frac <= 1):
        raise ValueError("spatial.min_valid_fraction must be in [0,1]")

    static_fraction = cv2.resize(
        valid_static.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
    ).astype(np.float32)

    for i in range(arr.shape[0]):
        frame = arr[i]
        finite_valid = valid_static & np.isfinite(frame)
        weights = finite_valid.astype(np.float32)
        numerator = cv2.resize(
            np.where(finite_valid, frame, 0.0).astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_AREA,
        )
        denom = cv2.resize(weights, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        good = denom >= min_frac
        # Do not create a value where the static mask itself had no support.
        good &= static_fraction > 0
        out_i = np.full((h, w), np.nan, dtype=np.float32)
        out_i[good] = numerator[good] / np.maximum(denom[good], 1e-12)
        out[i] = out_i
    return out, static_fraction


def filter_traces_sosfiltfilt(X: np.ndarray, sos: np.ndarray, fps: float, pad_sec: float) -> tuple[np.ndarray, int]:
    """Apply a zero-phase SOS filter with a bounded odd-reflection padding length.
    
    Parameters
    ----------
    X : np.ndarray
        Time-series matrix with shape ``(n_time, n_pixels)``.
    sos : np.ndarray
        Second-order-sections filter coefficient matrix.
    fps : float
        Acquisition frame rate in Hz (frames per second).
    pad_sec : float
        Requested synthetic boundary padding in seconds.
    
    Returns
    -------
    tuple[np.ndarray, int]
        Filtered time-series matrix and actual number of padding frames.
    
    Notes
    -----
    The requested odd-extension pad is limited to at most ``n_time - 2`` frames; filtering uses
    ``scipy.signal.sosfiltfilt`` along the time axis."""
    n_time = int(X.shape[0])
    requested = max(0, int(round(float(pad_sec) * float(fps))))
    actual = min(requested, max(0, n_time - 2))
    if actual == 0:
        out = sosfiltfilt(sos, X, axis=0, padtype=None, padlen=0)
    else:
        out = sosfiltfilt(sos, X, axis=0, padtype="odd", padlen=actual)
    return np.asarray(out, dtype=np.float32), int(actual)


def _roi_trace_names_and_masks(geometry: StemGeometry, roi_defs: pd.DataFrame) -> tuple[list[str], list[np.ndarray], list[int], list[str]]:
    """Define valid left/right ROI masks for optional time-trace export.
    
    Parameters
    ----------
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    roi_defs : pd.DataFrame
        Step6b analysis ROI definitions table.
    
    Returns
    -------
    tuple[list[str], list[np.ndarray], list[int], list[str]]
        Result of the documented operation."""
    names: list[str] = []
    masks: list[np.ndarray] = []
    ids: list[int] = []
    sides: list[str] = []
    if roi_defs.empty:
        return names, masks, ids, sides
    id_col = "broad_structure_id" if "broad_structure_id" in roi_defs.columns else "analysis_structure_id"
    acro_col = "analysis_roi_acronym" if "analysis_roi_acronym" in roi_defs.columns else "broad_acronym"
    seen: set[tuple[int, str]] = set()
    for _, rr in roi_defs.iterrows():
        rid = int(rr[id_col])
        acro = safe_str(rr.get(acro_col, rr.get("broad_acronym", rid)))
        for side, side_mask in [("left", geometry.left_fc), ("right", geometry.right_fc)]:
            key = (rid, side)
            if key in seen:
                continue
            seen.add(key)
            m = (geometry.labels_fc == rid) & geometry.bilateral_fc & side_mask
            if not m.any():
                continue
            names.append(f"{acro}_{side}")
            masks.append(m)
            ids.append(rid)
            sides.append(side)
    return names, masks, ids, sides


def preprocess_segment(
    var: xr.DataArray,
    t0_sec: float,
    t1_sec: float,
    fps: float,
    geometry: StemGeometry,
    config: dict[str, Any],
    *,
    roi_defs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Extract, resize, detrend, filter, and trim one time window for FC analysis.
    
    Parameters
    ----------
    var : xr.DataArray
        Lazy-loaded Step4 signal DataArray with axes ``(time, y, x)``.
    t0_sec : float
        Start time of the selected window in seconds.
    t1_sec : float
        End time of the selected window in seconds.
    fps : float
        Acquisition frame rate in Hz (frames per second).
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    roi_defs : pd.DataFrame | None
        Step6b analysis ROI definitions table.
    
    Returns
    -------
    dict[str, Any]
        Metadata, retained FC pixels, filtered traces, validity mask, preview and optional ROI traces.
    
    Notes
    -----
    Window boundaries are converted to frames by rounding ``time_sec * raw.fps``. Static and
    dynamic validity filtering precede detrending/filtering. The default temporal window is
    trimmed by 2 seconds on each side. ROI traces, when enabled, are pixel means at raw,
    detrended and filtered stages."""
    cfg = step7_cfg(config)
    tc = cfg["temporal"]
    sc = cfg["spatial"]
    n_time_total = int(var.sizes["time"])
    i0 = int(round(float(t0_sec) * fps))
    n_expected = int(round((float(t1_sec) - float(t0_sec)) * fps))
    i1 = i0 + n_expected
    if i0 < 0 or i0 >= n_time_total or i1 > n_time_total:
        raise IndexError(
            f"REST segment [{t0_sec}, {t1_sec}] -> [{i0}, {i1}) outside recording n={n_time_total}"
        )
    movie = (
        var.isel(time=slice(i0, i1)).load().values.astype(np.float32, copy=False)
    )
    if movie.shape[0] < 4:
        raise RuntimeError("REST segment has too few samples")

    movie_fc, spatial_fraction = mask_aware_resize_movie_area(
        movie,
        geometry.bilateral_processed,
        geometry.fc_shape,
        min_valid_fraction=float(sc.get("min_valid_fraction", 0.50)),
    )

    finite_all = np.all(np.isfinite(movie_fc), axis=0)
    # Avoid np.nanstd on all-NaN/background slices.  Only pixels finite for
    # the entire accepted REST segment can enter FC anyway, so compute the
    # same population SD (ddof=0) only for those pixels.
    raw_std = np.full(movie_fc.shape[1:], np.nan, dtype=np.float32)
    if np.any(finite_all):
        raw_std[finite_all] = np.std(
            movie_fc[:, finite_all], axis=0, ddof=0
        ).astype(np.float32, copy=False)
    eps = float(tc.get("variance_epsilon", 1e-12))
    functional = geometry.bilateral_fc & finite_all & np.isfinite(raw_std) & (raw_std > eps)
    valid_flat = np.flatnonzero(functional.ravel()).astype(np.int64)
    if valid_flat.size == 0:
        raise RuntimeError("No finite non-constant bilateral pixels remain in this REST segment")

    X = movie_fc.reshape(movie_fc.shape[0], -1)[:, valid_flat].astype(np.float32, copy=False)
    X_raw = X
    if bool(tc.get("detrend_enabled", True)):
        X_det = scipy_detrend(
            X_raw,
            axis=0,
            type=str(tc.get("detrend_type", "linear")),
            overwrite_data=False,
        ).astype(np.float32, copy=False)
    else:
        X_det = X_raw.copy()

    sos = make_bandpass_sos(config, fps)
    pad_frames = 0
    if sos is not None:
        X_filt, pad_frames = filter_traces_sosfiltfilt(
            X_det, sos, fps, float(tc.get("synthetic_odd_pad_sec", 20.0))
        )
    else:
        X_filt = X_det.copy()

    left_sec = float(tc.get("trim_left_sec", 2.0))
    right_sec = float(tc.get("trim_right_sec", 2.0))
    if not np.isfinite(left_sec) or not np.isfinite(right_sec) or min(left_sec, right_sec) < 0:
        raise ValueError("trim_left_sec and trim_right_sec must be nonnegative and finite")
    trim_left = int(round(left_sec * fps))
    trim_right = int(round(right_sec * fps))
    if trim_left + trim_right >= X_filt.shape[0] - 3:
        raise ValueError("Window too short after left/right trimming")
    end = X_filt.shape[0] - trim_right if trim_right else X_filt.shape[0]
    X_filt = X_filt[trim_left:end]
    X_det = X_det[trim_left:end]
    X_raw = X_raw[trim_left:end]

    filt_good = np.all(np.isfinite(X_filt), axis=0)
    filt_std = np.std(X_filt, axis=0, ddof=1)
    filt_good &= np.isfinite(filt_std) & (filt_std > eps)
    X_raw = X_raw[:, filt_good]
    X_det = X_det[:, filt_good]
    X_filt = X_filt[:, filt_good]
    valid_flat = valid_flat[filt_good]

    functional_final = np.zeros(np.prod(geometry.fc_shape), dtype=bool)
    functional_final[valid_flat] = True
    functional_final = functional_final.reshape(geometry.fc_shape)

    roi_trace_payload: dict[str, Any] = {}
    if roi_defs is not None and not roi_defs.empty:
        names, masks, ids, sides = _roi_trace_names_and_masks(geometry, roi_defs)
        colmap = np.full(np.prod(geometry.fc_shape), -1, dtype=np.int64)
        colmap[valid_flat] = np.arange(valid_flat.size, dtype=np.int64)
        raw_traces = []
        det_traces = []
        filt_traces = []
        n_pixels = []
        kept_names = []
        kept_ids = []
        kept_sides = []
        for name, mask, rid, side in zip(names, masks, ids, sides):
            cols = colmap[np.flatnonzero(mask.ravel())]
            cols = cols[cols >= 0]
            if cols.size == 0:
                continue
            kept_names.append(name)
            kept_ids.append(rid)
            kept_sides.append(side)
            n_pixels.append(int(cols.size))
            raw_traces.append(np.mean(X_raw[:, cols], axis=1))
            det_traces.append(np.mean(X_det[:, cols], axis=1))
            filt_traces.append(np.mean(X_filt[:, cols], axis=1))
        if kept_names:
            roi_trace_payload = {
                "names": np.asarray(kept_names, dtype="U96"),
                "roi_ids": np.asarray(kept_ids, dtype=np.int32),
                "sides": np.asarray(kept_sides, dtype="U8"),
                "n_pixels": np.asarray(n_pixels, dtype=np.int32),
                "raw": np.asarray(raw_traces, dtype=np.float32),
                "detrended": np.asarray(det_traces, dtype=np.float32),
                "filtered": np.asarray(filt_traces, dtype=np.float32),
                "time_sec": trim_left / fps + np.arange(X_filt.shape[0], dtype=np.float32) / float(fps),
            }

    preview_frame = movie_fc[movie_fc.shape[0] // 2].astype(np.float32)
    del movie, movie_fc
    gc.collect()
    return {
        "i0": i0,
        "i1": i1,
        "n_frames": int(X_filt.shape[0]),
        "valid_flat": valid_flat,
        "X_filtered": X_filt,
        "functional_valid_fc": functional_final,
        "spatial_valid_fraction_fc": spatial_fraction,
        "pad_frames": int(pad_frames),
        "pad_sec_actual": float(pad_frames / fps),
        "trim_left_frames": trim_left,
        "trim_right_frames": trim_right,
        "roi_traces": roi_trace_payload,
        "preview_frame": preview_frame,
    }


# =============================================================================
# Correlation kernels
# =============================================================================


def zscore_columns(X: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Standardize finite nonconstant signal columns using sample standard deviation.
    
    Parameters
    ----------
    X : np.ndarray
        Time-series matrix with shape ``(n_time, n_pixels)``.
    eps : float
        Minimum nonzero sample standard deviation.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Standardized retained columns and a boolean mask of accepted input columns.
    
    Notes
    -----
    The input matrix has shape ``(n_time, n_pixels)``. Only finite columns with sample standard
    deviation ``ddof=1`` greater than ``eps`` are retained."""
    arr = np.asarray(X, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    sd = np.std(arr, axis=0, ddof=1)
    good = np.isfinite(sd) & (sd > eps) & np.all(np.isfinite(arr), axis=0)
    if not np.any(good):
        return np.empty((arr.shape[0], 0), dtype=np.float32), good
    z = (arr[:, good] - mean[good]) / sd[good]
    return z.astype(np.float32), good


def compute_homotopic_segment_z(
    X_filtered: np.ndarray,
    valid_flat: np.ndarray,
    geometry: StemGeometry,
    *,
    fisher_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise homotopic Pearson correlations in Fisher-z space.
    
    Parameters
    ----------
    X_filtered : np.ndarray
        Filtered pixel traces with shape ``(n_time, n_valid_pixels)``.
    valid_flat : np.ndarray
        Row-major flattened FC-grid indices of retained pixel columns.
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    fisher_clip : float
        Positive clipping limit for Pearson ``r`` before ``arctanh``.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Fisher-z map ``(y, x)``, valid source ROI IDs and corresponding Fisher-z values.
    
    Notes
    -----
    Pearson ``r`` is calculated between source pixels and reflected contralateral partners on
    filtered time courses; ``arctanh(clip(r))`` is stored at source pixels only. Noneligible
    locations remain NaN."""
    h, w = geometry.fc_shape
    colmap = np.full(h * w, -1, dtype=np.int64)
    colmap[valid_flat] = np.arange(valid_flat.size, dtype=np.int64)
    sc = colmap[geometry.pair_src_flat]
    dc = colmap[geometry.pair_dst_flat]
    ok = (sc >= 0) & (dc >= 0)
    zmap = np.full((h, w), np.nan, dtype=np.float32)
    if not np.any(ok):
        return zmap, np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    A = X_filtered[:, sc[ok]]
    B = X_filtered[:, dc[ok]]
    Am = A - np.mean(A, axis=0, keepdims=True)
    Bm = B - np.mean(B, axis=0, keepdims=True)
    As = np.std(A, axis=0, ddof=1)
    Bs = np.std(B, axis=0, ddof=1)
    good = (As > 0) & (Bs > 0) & np.isfinite(As) & np.isfinite(Bs)
    r = np.full(A.shape[1], np.nan, dtype=np.float64)
    if np.any(good):
        r[good] = np.sum(Am[:, good] * Bm[:, good], axis=0) / (
            (A.shape[0] - 1) * As[good] * Bs[good]
        )
    clip = float(fisher_clip)
    z = np.arctanh(np.clip(r, -clip, clip)).astype(np.float32)
    src_flat = geometry.pair_src_flat[ok]
    good_z = np.isfinite(z)
    zmap.ravel()[src_flat[good_z]] = z[good_z]
    roi_ids = geometry.pair_roi_id[ok][good_z].astype(np.int32)
    return zmap, roi_ids, z[good_z]


def compute_gbc_blockwise(
    X: np.ndarray,
    *,
    block_size: int,
    fisher_clip: float,
) -> np.ndarray:
    """Compute per-pixel global brain connectivity from pairwise Fisher-z correlations.
    
    Parameters
    ----------
    X : np.ndarray
        Time-series matrix with shape ``(n_time, n_pixels)``.
    block_size : int
        Number of seed pixels processed per matrix-correlation block.
    fisher_clip : float
        Positive clipping limit for Pearson ``r`` before ``arctanh``.
    
    Returns
    -------
    np.ndarray
        Per-input-pixel mean Fisher-z GBC, with NaN for invalid pixels.
    
    Notes
    -----
    The input matrix is ``(n_time, n_pixels)``. Off-diagonal correlations are Fisher
    transformed; the mean Fisher z excludes self-correlation and uses a configurable pixel
    block size for memory efficiency."""
    Xz, good = zscore_columns(X)
    if Xz.shape[1] < 2:
        return np.full(X.shape[1], np.nan, dtype=np.float32)
    # If zscore dropped columns, compute for retained columns and scatter back.
    kept = np.flatnonzero(good)
    n = Xz.shape[1]
    t = Xz.shape[0]
    out_kept = np.full(n, np.nan, dtype=np.float32)
    bs = max(1, int(block_size))
    clip = float(fisher_clip)
    XT = Xz.T.astype(np.float32, copy=False)
    for start in range(0, n, bs):
        stop = min(n, start + bs)
        R = (XT[start:stop] @ Xz) / float(t - 1)
        R = np.clip(R, -clip, clip)
        Z = np.arctanh(R).astype(np.float32)
        local = np.arange(stop - start)
        Z[local, start + local] = np.nan
        out_kept[start:stop] = np.nanmean(Z, axis=1).astype(np.float32)
        del R, Z
    out = np.full(X.shape[1], np.nan, dtype=np.float32)
    out[kept] = out_kept
    return out


def compute_global_segment_z(
    X_filtered: np.ndarray,
    valid_flat: np.ndarray,
    geometry: StemGeometry,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute GBC maps for the configured hemisphere-specific correlation pools.
    
    Parameters
    ----------
    X_filtered : np.ndarray
        Filtered pixel traces with shape ``(n_time, n_valid_pixels)``.
    valid_flat : np.ndarray
        Row-major flattened FC-grid indices of retained pixel columns.
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    tuple[np.ndarray, dict[str, int]]
        Spatial GBC Fisher-z map and contributing pixel counts by pool.
    
    Notes
    -----
    When both hemispheres are analyzed, ``separate_hemispheres`` computes within-side pools
    independently, whereas ``whole_cortex`` includes cross-hemisphere pairs. Each pool excludes
    self-correlations."""
    cfg = step7_cfg(config)["global"]
    h, w = geometry.fc_shape
    map_out = np.full((h, w), np.nan, dtype=np.float32)
    colmap = np.full(h * w, -1, dtype=np.int64)
    colmap[valid_flat] = np.arange(valid_flat.size, dtype=np.int64)
    counts: dict[str, int] = {}

    for pool_key, mask in geometry.global_pool_masks_fc.items():
        flats = np.flatnonzero(mask.ravel())
        cols = colmap[flats]
        good = cols >= 0
        flats = flats[good]
        cols = cols[good]
        counts[pool_key] = int(cols.size)
        if cols.size < int(cfg.get("min_pool_pixels", 2)):
            continue

        gbc = compute_gbc_blockwise(
            X_filtered[:, cols],
            block_size=int(cfg.get("correlation_block_size", 512)),
            fisher_clip=float(cfg.get("fisher_clip", 0.999999)),
        )
        finite = np.isfinite(gbc)
        map_out.ravel()[flats[finite]] = gbc[finite]

    return map_out, counts


def nanmean_maps(maps: list[np.ndarray], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Average maps over valid segments, retaining the number of finite contributions.
    
    Parameters
    ----------
    maps : list[np.ndarray]
        Sequence of segment-wise two-dimensional Fisher-z maps.
    shape : tuple[int, int]
        Expected output image shape in pixels ``(height, width)``.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Finite-segment mean map and per-pixel finite contribution count.
    
    Notes
    -----
    Each output pixel is the arithmetic mean of finite segment-level Fisher z values. The count
    map is an unsigned 16-bit count of finite contributions; Pearson r is not averaged
    directly."""
    if not maps:
        return np.full(shape, np.nan, dtype=np.float32), np.zeros(shape, dtype=np.uint16)
    stack = np.stack(maps, axis=0).astype(np.float32)
    count = np.sum(np.isfinite(stack), axis=0).astype(np.uint16)
    total = np.nansum(stack, axis=0, dtype=np.float64)
    mean = np.full(shape, np.nan, dtype=np.float32)
    good = count > 0
    mean[good] = (total[good] / count[good]).astype(np.float32)
    return mean, count


# =============================================================================
# ROI tables / metadata
# =============================================================================


def roi_definitions(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Retrieve the Step6b table of analytical ROI definitions.
    
    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    
    Returns
    -------
    pd.DataFrame
        Result of the documented operation."""
    df = tables.get("analysis_roi_definitions", pd.DataFrame()).copy()
    if df.empty:
        raise RuntimeError("Step6b analysis_roi_definitions.csv is empty")
    return df


def roi_meta_lookup(roi_defs: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Index ROI names and anatomical identifiers by structure ID.
    
    Parameters
    ----------
    roi_defs : pd.DataFrame
        Step6b analysis ROI definitions table.
    
    Returns
    -------
    dict[int, dict[str, Any]]
        Result of the documented operation."""
    out: dict[int, dict[str, Any]] = {}
    for _, rr in roi_defs.iterrows():
        rid = int(rr.get("broad_structure_id", rr.get("analysis_structure_id")))
        out[rid] = {
            "broad_structure_id": rid,
            "broad_acronym": safe_str(rr.get("broad_acronym", rr.get("analysis_roi_acronym", ""))),
            "broad_name": safe_str(rr.get("broad_name", rr.get("analysis_roi_name", ""))),
            "analysis_structure_id": int(rr.get("analysis_structure_id", rid)),
            "analysis_roi_acronym": safe_str(rr.get("analysis_roi_acronym", rr.get("broad_acronym", ""))),
            "analysis_roi_name": safe_str(rr.get("analysis_roi_name", rr.get("broad_name", ""))),
            "broad_family_structure_id": rr.get("broad_family_structure_id", rid),
            "broad_family_acronym": safe_str(rr.get("broad_family_acronym", rr.get("broad_acronym", ""))),
            "broad_family_name": safe_str(rr.get("broad_family_name", rr.get("broad_name", ""))),
        }
    return out


def selected_include_lookup(tables: dict[str, pd.DataFrame]) -> dict[tuple[str, str, str], bool]:
    """Read Step6b ROI inclusion flags into a lookup keyed by mask set and analysis.
    
    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    
    Returns
    -------
    dict[tuple[str, str, str], bool]
        Result of the documented operation."""
    df = tables.get("selected_rois", pd.DataFrame())
    out: dict[tuple[str, str, str], bool] = {}
    if df.empty:
        return out
    for _, rr in df.iterrows():
        out[(safe_str(rr.get("mask_set")), safe_str(rr.get("analysis")), safe_str(rr.get("roi_key")))] = bool_value(rr.get("include", False))
    return out


def _first_matching_row(df: pd.DataFrame, conditions: dict[str, Any]) -> pd.Series | None:
    """Find exactly one matching Step6b table row or return no match.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input metadata table.
    conditions : dict[str, Any]
        Column-to-value equality constraints identifying the desired row.
    
    Returns
    -------
    pd.Series | None
        Result of the documented operation."""
    if df.empty:
        return None
    work = df
    for col, val in conditions.items():
        if col not in work.columns:
            raise KeyError(f"Required Step6b matching column missing: {col}")
        work = work[work[col].astype(str).eq(str(val))]
    if len(work) > 1:
        raise RuntimeError(f"Ambiguous Step6b ROI eligibility rows for {conditions}: n={len(work)}")
    return work.iloc[0] if not work.empty else None


def _copy_columns_from_row(dst: dict[str, Any], row: pd.Series | None, columns: Iterable[str]) -> None:
    """Copy selected metadata fields from a row when present.
    
    Parameters
    ----------
    dst : dict[str, Any]
        Destination dictionary modified in place.
    row : pd.Series | None
        Metadata for the selected Step6b/Step7 recording.
    columns : Iterable[str]
        Source column names to copy when available."""
    if row is None:
        return
    for col in columns:
        if col in row.index:
            dst[col] = row.get(col)


def summarize_homotopic_rois(
    row: pd.Series,
    mask_set: str,
    geometry: StemGeometry,
    mean_map: np.ndarray,
    tables: dict[str, pd.DataFrame],
    roi_defs: pd.DataFrame,
    n_segments: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize stem-level homotopic Fisher-z maps for selected analytical ROIs.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    mask_set : str
        Name of the selected Step6b mask set.
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    mean_map : np.ndarray
        Mean Fisher-z connectivity map on the FC grid.
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    roi_defs : pd.DataFrame
        Step6b analysis ROI definitions table.
    n_segments : int
        Number of accepted, processed segments.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    pd.DataFrame
        Per-ROI table with Fisher-z estimates, counts, coverage and QC fields.
    
    Notes
    -----
    ROI results describe the mean of eligible map values in Fisher-z space and apply Step6b
    eligibility and additional FC-grid QC."""
    meta = roi_meta_lookup(roi_defs)
    selected = selected_include_lookup(tables)
    qcfg = step7_cfg(config)["final_grid_qc"]
    hstem = tables.get("homotopic_stem_usable_coverage", pd.DataFrame())
    rows: list[dict[str, Any]] = []
    ids = sorted(set(int(v) for v in geometry.pair_roi_id.tolist()))
    for rid in ids:
        m = meta.get(rid, {"broad_structure_id": rid, "analysis_roi_acronym": str(rid)})
        acro = safe_str(m.get("analysis_roi_acronym", m.get("broad_acronym", rid)))
        pair_key = f"{safe_str(m.get('broad_acronym', acro))}_LR"
        # Prefer the exact Step6b pair key if available.
        hs = _first_matching_row(
            hstem,
            {"mask_set": mask_set, "group_id": geometry.group_id, "stem": geometry.stem, "broad_structure_id": rid},
        )
        if hs is not None:
            pair_key = safe_str(hs.get("pair_key", pair_key))

        vals = mean_map.ravel()[geometry.pair_src_flat[geometry.pair_roi_id == rid]]
        finite = np.isfinite(vals)
        computed_mean = float(np.nanmean(vals)) if np.any(finite) else np.nan
        computed_sd = float(np.nanstd(vals, ddof=1)) if np.sum(finite) > 1 else np.nan
        n_valid = int(np.sum(finite))
        n_geom = int(geometry.pair_geometry_count_by_roi.get(rid, 0))
        n_potential = int(geometry.pair_potential_count_by_roi.get(rid, 0))
        grid_cov = float(n_geom / n_potential) if n_potential > 0 else np.nan
        selected_include = selected.get((mask_set, "homotopic", pair_key), False)
        stem_pass = bool_value(hs.get("homotopic_stem_pass", False)) if hs is not None else False
        fc_grid_pass = bool(
            n_geom >= int(qcfg.get("min_homotopic_pairs", 5))
            and np.isfinite(grid_cov)
            and grid_cov >= float(qcfg.get("min_final_grid_coverage_fraction", 0.25))
        )
        analysis_pass = bool(selected_include and stem_pass and fc_grid_pass and n_valid > 0 and np.isfinite(computed_mean))
        rec = {
            "mask_set": mask_set,
            "slot": safe_str(row.get("slot")),
            "group_id": geometry.group_id,
            "mouse_id": geometry.mouse_id,
            "experimental_group": safe_str(row.get("experimental_group")),
            "comparison_group": safe_str(row.get("comparison_group")),
            "rest_variant": safe_str(row.get("rest_variant")),
            "variant_id": safe_str(step7_cfg(config).get("active_variant_id")),
            "stem": geometry.stem,
            "exp_day": safe_str(row.get("exp_day")),
            "state": safe_str(row.get("state")),
            "protocol": safe_str(row.get("protocol")),
            **m,
            "pair_key": pair_key,
            "map_hemisphere": geometry.homotopic_map_side,
            "computed_mean_z_qc": computed_mean,
            "computed_sd_z_qc": computed_sd,
            "mean_z": computed_mean if analysis_pass else np.nan,
            "sd_z": computed_sd if analysis_pass else np.nan,
            "n_valid_pairs_fc": n_valid,
            "n_geometry_pairs_fc": n_geom,
            "n_potential_pairs_fc": n_potential,
            "fc_pair_coverage_fraction": grid_cov,
            "n_rest_segments": int(n_segments),
            "selected_include": bool(selected_include),
            "homotopic_stem_pass": bool(stem_pass),
            "fc_grid_pass": fc_grid_pass,
            "analysis_pass": analysis_pass,
        }
        _copy_columns_from_row(
            rec,
            hs,
            [
                "homotopic_technical_pass",
                "homotopic_relative_to_cohort",
                "stem_usable_pair_coverage_fraction",
                "canonical_homotopic_pair_px",
                "canonical_final_usable_pair_px",
                "canonical_fixed_technical_pair_coverage_fraction",
            ],
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_global_rois(
    row: pd.Series,
    mask_set: str,
    geometry: StemGeometry,
    mean_map: np.ndarray,
    tables: dict[str, pd.DataFrame],
    roi_defs: pd.DataFrame,
    n_segments: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize stem-level GBC maps by ROI and permitted hemispheric pool.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    mask_set : str
        Name of the selected Step6b mask set.
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    mean_map : np.ndarray
        Mean Fisher-z connectivity map on the FC grid.
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    roi_defs : pd.DataFrame
        Step6b analysis ROI definitions table.
    n_segments : int
        Number of accepted, processed segments.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    pd.DataFrame
        Per-ROI and hemisphere table with GBC estimates and QC metadata.
    
    Notes
    -----
    Global ROI aggregation respects the configured GBC pool and the allowed hemisphere of each
    ROI; it does not conflate left/right and bilateral pools."""
    meta = roi_meta_lookup(roi_defs)
    selected = selected_include_lookup(tables)
    qcfg = step7_cfg(config)["final_grid_qc"]
    gstem = tables.get("global_stem_usable_coverage", pd.DataFrame())
    rows: list[dict[str, Any]] = []

    # ROI outcomes remain left/right-specific even when the GBC correlation
    # pool is the whole bilateral cortex. This preserves Step6b side-specific
    # coverage/eligibility and lets left/right ROIs remain independent outcomes.
    for side in geometry.global_roi_sides:
        side_mask = geometry.left_fc if side == "left" else geometry.right_fc
        pool_key = _global_pool_key_for_roi_side(geometry, side)
        global_pool_mask = geometry.global_pool_masks_fc[pool_key]
        side_valid_mask = global_pool_mask & side_mask

        ids = [
            int(v)
            for v in np.unique(geometry.labels_fc[side_mask])
            if int(v) > 0
        ]
        for rid in ids:
            m = meta.get(
                rid,
                {"broad_structure_id": rid, "analysis_roi_acronym": str(rid)},
            )
            acro = safe_str(
                m.get("analysis_roi_acronym", m.get("broad_acronym", rid))
            )
            roi_key = f"{safe_str(m.get('broad_acronym', acro))}_{side}"
            gs = _first_matching_row(
                gstem,
                {
                    "mask_set": mask_set,
                    "group_id": geometry.group_id,
                    "stem": geometry.stem,
                    "broad_structure_id": rid,
                    "hemisphere": side,
                },
            )
            if gs is not None:
                roi_key = safe_str(
                    gs.get("global_roi_key", gs.get("roi_key", roi_key))
                )

            potential = (geometry.labels_fc == rid) & side_mask
            geom = (geometry.labels_fc == rid) & side_valid_mask
            vals = mean_map[geom]
            finite = np.isfinite(vals)
            computed_mean = (
                float(np.nanmean(vals)) if np.any(finite) else np.nan
            )
            computed_sd = (
                float(np.nanstd(vals, ddof=1))
                if np.sum(finite) > 1
                else np.nan
            )
            n_potential = int(potential.sum())
            n_geom = int(geom.sum())
            n_valid = int(np.sum(finite))
            grid_cov = (
                float(n_geom / n_potential) if n_potential > 0 else np.nan
            )
            selected_include = selected.get(
                (mask_set, "global", roi_key), False
            )
            stem_pass = (
                bool_value(gs.get("global_stem_pass", False))
                if gs is not None
                else False
            )
            fc_grid_pass = bool(
                n_geom >= int(qcfg.get("min_roi_pixels", 5))
                and np.isfinite(grid_cov)
                and grid_cov
                >= float(qcfg.get("min_final_grid_coverage_fraction", 0.25))
            )
            analysis_pass = bool(
                selected_include and stem_pass and fc_grid_pass and n_valid > 0 and np.isfinite(computed_mean)
            )

            rec = {
                "mask_set": mask_set,
                "slot": safe_str(row.get("slot")),
                "group_id": geometry.group_id,
                "mouse_id": geometry.mouse_id,
                "experimental_group": safe_str(row.get("experimental_group")),
            "comparison_group": safe_str(row.get("comparison_group")),
            "rest_variant": safe_str(row.get("rest_variant")),
            "variant_id": safe_str(step7_cfg(config).get("active_variant_id")),
                "stem": geometry.stem,
                "exp_day": safe_str(row.get("exp_day")),
                "state": safe_str(row.get("state")),
                "protocol": safe_str(row.get("protocol")),
                **m,
                "hemisphere": side,
                "roi_key": roi_key,
                "global_analysis_choice_step7": geometry.global_analysis_choice,
                "global_both_hemispheres_mode": geometry.global_both_mode,
                "global_pool": pool_key,
                "global_pool_includes_cross_hemisphere": bool(
                    geometry.global_analysis_choice == "both"
                    and geometry.global_both_mode == "whole_cortex"
                ),
                "global_pool_geometry_pixels_fc": int(global_pool_mask.sum()),
                "computed_mean_GBC_z_qc": computed_mean,
                "computed_sd_GBC_z_qc": computed_sd,
                "mean_GBC_z": computed_mean if analysis_pass else np.nan,
                "sd_GBC_z": computed_sd if analysis_pass else np.nan,
                "n_valid_pixels_fc": n_valid,
                "n_geometry_pixels_fc": n_geom,
                "n_potential_pixels_fc": n_potential,
                "fc_coverage_fraction": grid_cov,
                "n_rest_segments": int(n_segments),
                "selected_include": bool(selected_include),
                "global_stem_pass": bool(stem_pass),
                "fc_grid_pass": fc_grid_pass,
                "analysis_pass": analysis_pass,
            }
            _copy_columns_from_row(
                rec,
                gs,
                [
                    "global_technical_pass",
                    "global_relative_to_cohort",
                    "stem_usable_coverage_fraction",
                    "canonical_buffered_area_px",
                    "canonical_final_usable_px",
                    "canonical_fixed_technical_coverage_fraction",
                    "global_analysis_choice",
                    "global_analysis_hemisphere",
                ],
            )
            rows.append(rec)
    return pd.DataFrame(rows)


# =============================================================================
# Saving / visualization
# =============================================================================


def _roi_boundary_contours(ax: Any, labels: np.ndarray, *, linewidth: float = 0.45, alpha: float = 0.7) -> None:
    """Overlay analytical ROI boundaries on a Matplotlib axes object.
    
    Parameters
    ----------
    ax : Any
        Matplotlib axes receiving ROI boundary contours.
    labels : np.ndarray
        Two-dimensional integer anatomical ROI label image.
    linewidth : float
        Width of the plotted ROI contours.
    alpha : float
        Opacity of the ROI contour overlay."""
    for rid in [int(v) for v in np.unique(labels) if int(v) > 0]:
        m = labels == rid
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors="white", linewidths=linewidth, alpha=alpha)


def _robust_limits(arr: np.ndarray, mask: np.ndarray | None, percentiles: Sequence[float]) -> tuple[float, float]:
    """Estimate display limits from finite masked values and chosen percentiles.
    
    Parameters
    ----------
    arr : np.ndarray
        Image or numerical array used by the operation.
    mask : np.ndarray | None
        Binary input mask.
    percentiles : Sequence[float]
        Lower and upper percentiles for display limits.
    
    Returns
    -------
    tuple[float, float]
        Result of the documented operation."""
    x = np.asarray(arr, dtype=float)
    good = np.isfinite(x)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    vals = x[good]
    if vals.size == 0:
        return -1.0, 1.0
    p0, p1 = map(float, percentiles)
    lo, hi = np.nanpercentile(vals, [p0, p1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def save_stem_qc_png(
    path: Path,
    geometry: StemGeometry,
    preview_frame: np.ndarray,
    hom_map: np.ndarray,
    gbc_map: np.ndarray,
    config: dict[str, Any],
    *,
    title: str,
) -> None:
    """Save a diagnostic overview of anatomical geometry and stem FC maps.
    
    Parameters
    ----------
    path : Path
        Destination file path or path to the required artifact.
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    preview_frame : np.ndarray
        Representative hemodynamic image on the reduced FC grid.
    hom_map : np.ndarray
        Mean homotopic Fisher-z map with shape ``(y, x)``.
    gbc_map : np.ndarray
        Mean global connectivity Fisher-z map with shape ``(y, x)``.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    title : str
        Descriptive title for the saved QC figure."""
    pcts = step7_cfg(config)["qc"].get("map_percentiles", [2.0, 98.0])
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    v0, v1 = _robust_limits(preview_frame, geometry.bilateral_fc, pcts)
    im = axes[0, 0].imshow(np.where(geometry.bilateral_fc, preview_frame, np.nan), cmap="jet", vmin=v0, vmax=v1)
    _roi_boundary_contours(axes[0, 0], geometry.labels_fc)
    axes[0, 0].set_title("resized HbT preview + Allen")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    side_code = np.zeros(geometry.fc_shape, dtype=float)
    side_code[geometry.left_fc] = 1
    side_code[geometry.right_fc] = 2
    axes[0, 1].imshow(np.where(geometry.bilateral_fc, side_code, np.nan), cmap="viridis")
    _roi_boundary_contours(axes[0, 1], geometry.labels_fc)
    axes[0, 1].set_title("left/right after midline exclusion")

    pair_mask = np.zeros(geometry.fc_shape, dtype=bool)
    pair_mask.ravel()[geometry.pair_src_flat] = True
    axes[0, 2].imshow(pair_mask, cmap="gray")
    _roi_boundary_contours(axes[0, 2], geometry.labels_fc)
    axes[0, 2].set_title(f"final homotopic pairs on {geometry.homotopic_map_side}")

    h0, h1 = _robust_limits(hom_map, np.isfinite(hom_map), pcts)
    imh = axes[1, 0].imshow(hom_map, cmap="jet", vmin=h0, vmax=h1)
    _roi_boundary_contours(axes[1, 0], geometry.labels_fc)
    axes[1, 0].set_title("mean homotopic Fisher-z")
    fig.colorbar(imh, ax=axes[1, 0], fraction=0.046, pad=0.04)

    g0, g1 = _robust_limits(gbc_map, np.isfinite(gbc_map), pcts)
    img = axes[1, 1].imshow(gbc_map, cmap="jet", vmin=g0, vmax=g1)
    _roi_boundary_contours(axes[1, 1], geometry.labels_fc)
    axes[1, 1].set_title(
        "mean GBC Fisher-z\n"
        + f"choice={geometry.global_analysis_choice}; mode={geometry.global_both_mode}"
    )
    fig.colorbar(img, ax=axes[1, 1], fraction=0.046, pad=0.04)

    global_union = np.zeros(geometry.fc_shape, dtype=bool)
    for m in geometry.global_pool_masks_fc.values():
        global_union |= m
    axes[1, 2].imshow(global_union, cmap="gray")
    _roi_boundary_contours(axes[1, 2], geometry.labels_fc)
    axes[1, 2].set_title("global pool mask: " + ",".join(geometry.global_pool_masks_fc.keys()))

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_geometry(geometry: StemGeometry, root: Path, config: dict[str, Any], n_segments: int) -> dict[str, str]:
    """Persist the frozen FC geometry, masks, homotopic partners and provenance.
    
    Parameters
    ----------
    geometry : StemGeometry
        Recording-specific ``StemGeometry`` object on the FC grid.
    root : Path
        Directory in which to save the frozen geometry products.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    n_segments : int
        Number of accepted, processed segments.
    
    Returns
    -------
    dict[str, str]
        Result of the documented operation.
    
    Notes
    -----
    Frozen geometry and metadata support reproducible QC, reuse compatibility checks and later
    summary-only recalculation."""
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "bilateral_fc": root / "bilateral_final_usable_fc.npy",
        "left_fc": root / "left_hemisphere_fc.npy",
        "right_fc": root / "right_hemisphere_fc.npy",
        "labels_fc": root / "analysis_roi_labels_fc.npy",
        "pairs": root / "homotopic_pairs_fc.npz",
        "geometry_json": root / "fc_geometry.json",
    }
    np.save(files["bilateral_fc"], geometry.bilateral_fc.astype(np.uint8))
    np.save(files["left_fc"], geometry.left_fc.astype(np.uint8))
    np.save(files["right_fc"], geometry.right_fc.astype(np.uint8))
    np.save(files["labels_fc"], geometry.labels_fc.astype(np.int32))
    for pool_key, mask in geometry.global_pool_masks_fc.items():
        np.save(root / f"global_pool_{pool_key}_final_usable_fc.npy", mask.astype(np.uint8))
    np.savez_compressed(
        files["pairs"],
        src_flat=geometry.pair_src_flat,
        dst_flat=geometry.pair_dst_flat,
        roi_id=geometry.pair_roi_id,
    )
    payload = {
        "mask_set": geometry.mask_set,
        "stem": geometry.stem,
        "group_id": geometry.group_id,
        "ref_stem": geometry.ref_stem,
        "mouse_id": geometry.mouse_id,
        "healthy_side": geometry.healthy_side,
        "lesion_side": geometry.lesion_side,
        "homotopic_map_side": geometry.homotopic_map_side,
        "global_analysis_choice": geometry.global_analysis_choice,
        "global_both_hemispheres_mode": geometry.global_both_mode,
        "global_roi_sides": geometry.global_roi_sides,
        "global_pool_keys": list(geometry.global_pool_masks_fc.keys()),
        "analysis_roi_labels_source": geometry.labels_source_path,
        "processed_shape": list(geometry.processed_shape),
        "fc_shape": list(geometry.fc_shape),
        "stem_fullres_shape": list(geometry.stem_fullres_shape),
        "spatial_scale": float(step7_cfg(config)["spatial"].get("scale", 0.5)),
        "midline_origin_xy_fc": geometry.midline_origin_xy.tolist(),
        "midline_hat_xy_fc": geometry.midline_hat_xy.tolist(),
        "transform_source": geometry.transform_source,
        "transform_meta": geometry.transform_meta,
        "n_geometry_homotopic_pairs": int(len(geometry.pair_src_flat)),
        "n_accepted_rest_segments": int(n_segments),
        "variant_id": str(step7_cfg(config).get("active_variant_id")),
        "rest_variant": str(step7_cfg(config).get("rest_variant")),
        "signal": str(step7_cfg(config).get("signal")),
        "filter_type": str(step7_cfg(config)["temporal"].get("filter_type")),
        "trim_left_sec": float(step7_cfg(config)["temporal"].get("trim_left_sec", 2.0)),
        "trim_right_sec": float(step7_cfg(config)["temporal"].get("trim_right_sec", 2.0)),
        "updated_at": now_iso(),
    }
    with open(files["geometry_json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return {k: str(v) for k, v in files.items()}


# =============================================================================
# Stem processing / batch
# =============================================================================


def _open_signal(row: pd.Series, config: dict[str, Any]) -> tuple[xr.Dataset, xr.DataArray, Path]:
    """Open the requested Step4 Zarr signal without substituting another signal variant.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    tuple[xr.Dataset, xr.DataArray, Path]
        Result of the documented operation.
    
    Notes
    -----
    A missing requested variable is an error; no implicit substitution between ``dcHbT`` and
    ``dcHbT_vreg`` occurs. Output variable dimensions are ordered ``(time, y, x)``."""
    zarr_text = safe_str(row.get("processed_zarr_path_resolved")).strip()
    if not zarr_text:
        raise FileNotFoundError(f"Processed zarr missing: empty processed_zarr_path_resolved for {safe_str(row.get('stem'))}")
    p = Path(zarr_text)
    if not p.is_dir():
        raise FileNotFoundError(f"Processed zarr missing: {p}")
    try:
        ds = xr.open_zarr(p, chunks="auto", consolidated=False)
    except Exception:
        ds = xr.open_zarr(p, chunks=None, consolidated=False)
    signal = safe_str(step7_cfg(config).get("signal", "dcHbT"))
    if signal not in ds.data_vars:
        raise KeyError(
            f"Step7 signal {signal!r} missing in {p}. Available: {list(ds.data_vars)}. "
            "Step7 does not silently substitute a GSR/non-GSR variant."
        )
    var = ds[signal].transpose("time", "y", "x")
    return ds, var, p


def _stem_output_dirs(step7_paths: dict[str, Path], mask_set: str, row: pd.Series) -> dict[str, Path]:
    """Create and return the per-mask-set, per-mouse, per-stem output directories.
    
    Parameters
    ----------
    step7_paths : dict[str, Path]
        Path mapping for Step7 outputs.
    mask_set : str
        Name of the selected Step6b mask set.
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    
    Returns
    -------
    dict[str, Path]
        Result of the documented operation."""
    mouse = safe_str(row.get("mouse_id")) or safe_str(row.get("group_id"))
    stem = safe_str(row.get("stem"))
    key = Path(s6b.safe_key(mask_set)) / s6b.safe_key(mouse) / s6b.safe_key(stem)
    out = {
        "geometry": Path(step7_paths["geometry_root"]) / key,
        "segments": Path(step7_paths["segments_root"]) / key,
        "stem_maps": Path(step7_paths["stem_maps_root"]) / key,
        "roi_traces": Path(step7_paths["roi_traces_root"]) / key,
        "qc": Path(step7_paths["qc_root"]) / key,
    }
    for p in out.values():
        p.mkdir(parents=True, exist_ok=True)
    return out


def _expected_existing_signature(config: dict[str, Any], mask_set: str) -> dict[str, str]:
    """Describe the geometry and GBC settings required for reusable output.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    mask_set : str
        Name of the selected Step6b mask set.
    
    Returns
    -------
    dict[str, str]
        Result of the documented operation."""
    choice = safe_str(
        s6b.global_analysis_choice_for_mask_set(config, mask_set)
    ).strip().lower()
    mode = normalize_global_both_mode(config) if choice == "both" else "single_hemisphere"
    return {
        "global_analysis_choice": choice,
        "global_both_hemispheres_mode": mode,
        "analysis_roi_labels_source_name": "analysis_roi_ref_buffered_repaired.npy",
    }


def _assert_existing_output_compatible(
    geometry_json_path: Path,
    config: dict[str, Any],
    mask_set: str,
) -> None:
    """Reject saved stem outputs with incompatible frozen geometry metadata.
    
    Parameters
    ----------
    geometry_json_path : Path
        Path to previously saved stem geometry metadata.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    mask_set : str
        Name of the selected Step6b mask set.
    
    Notes
    -----
    Checks saved global hemisphere mode, selected analysis side and authoritative Step6b ROI
    label filename. A separate analysis fingerprint is checked when reusing a complete stem."""
    if not geometry_json_path.exists():
        raise RuntimeError(
            f"Existing Step7 maps found but geometry metadata is missing: {geometry_json_path}. "
            "Re-run with overwrite=True."
        )
    payload = json.loads(geometry_json_path.read_text(encoding="utf-8"))
    expected = _expected_existing_signature(config, mask_set)
    actual_choice = safe_str(payload.get("global_analysis_choice", ""))
    actual_mode = safe_str(payload.get("global_both_hemispheres_mode", ""))
    actual_source = Path(
        safe_str(payload.get("analysis_roi_labels_source", ""))
    ).name
    problems = []
    if actual_choice != expected["global_analysis_choice"]:
        problems.append(
            f"global_analysis_choice existing={actual_choice!r}, expected={expected['global_analysis_choice']!r}"
        )
    if actual_mode != expected["global_both_hemispheres_mode"]:
        problems.append(
            f"global_both_hemispheres_mode existing={actual_mode!r}, expected={expected['global_both_hemispheres_mode']!r}"
        )
    if actual_source != expected["analysis_roi_labels_source_name"]:
        problems.append(
            f"atlas source existing={actual_source!r}, expected={expected['analysis_roi_labels_source_name']!r}"
        )
    if problems:
        raise RuntimeError(
            "Existing Step7 output was generated with incompatible geometry/global settings: "
            + "; ".join(problems)
            + ". Re-run this stem with overwrite=True (or use a different functional_connectivity.folder_name)."
        )


def process_stem(
    row: pd.Series,
    rest_segments: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    s6b_paths: dict[str, Path],
    step7_paths: dict[str, Path],
    tables: dict[str, pd.DataFrame],
    *,
    overwrite: bool,
    save_roi_traces: bool,
    save_qc: bool,
) -> dict[str, Any]:
    """Process or safely reuse all FC windows for a single selected recording.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    rest_segments : pd.DataFrame
        Finalizer REST windows, consumed only for Awake/with_rest.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    paths : dict[str, Path]
        Project paths returned by the upstream project/index builder.
    s6b_paths : dict[str, Path]
        Path mapping for saved Step6b tables and geometry.
    step7_paths : dict[str, Path]
        Path mapping for Step7 outputs.
    tables : dict[str, pd.DataFrame]
        Saved Step6b tables keyed by their logical names.
    overwrite : bool
        Whether existing stem results should be removed and recomputed.
    save_roi_traces : bool
        Whether to save per-ROI traces for each segment.
    save_qc : bool
        Whether to save QC preview figures.
    
    Returns
    -------
    dict[str, Any]
        Status, windows, per-stem ROI tables, segment log and saved-product paths.
    
    Notes
    -----
    The ``with_rest`` branch uses Finalizer windows only for Awake; all other cases use
    consecutive complete windows. On ``overwrite=True`` old products are invalidated before
    recomputation. Segment FC maps are averaged in Fisher-z space."""
    mask_set = safe_str(row.get("mask_set"))
    stem = safe_str(row.get("stem"))
    out_dirs = _stem_output_dirs(step7_paths, mask_set, row)
    mean_h_path = out_dirs["stem_maps"] / "homotopic_z_mean.npy"
    mean_g_path = out_dirs["stem_maps"] / "gbc_z_mean.npy"
    stem_hroi_csv = out_dirs["stem_maps"] / "homotopic_roi_values.csv"
    stem_groi_csv = out_dirs["stem_maps"] / "global_roi_values.csv"
    stem_seglog_csv = out_dirs["stem_maps"] / "segment_processing_log.csv"

    ds, var, processed_path = _open_signal(row, config)
    fps = float(config.get("raw", {}).get("fps", np.nan))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("config raw.fps must be positive")
    mode = str(step7_cfg(config).get("rest_variant", "with_rest"))
    if mode == "with_rest" and str(row.get("state")) == "Awake":
        segs = rest_segments[rest_segments["stem"].astype(str).eq(stem)].copy()
        segs["source"] = "accepted_final_rest"
    else:
        segs = whole_recording_windows(stem, int(var.sizes["time"]), fps,
                                       float(step7_cfg(config).get("expected_segment_sec", 60.0)))
    segs = segs.sort_values(["t0_sec", "t1_sec"]).reset_index(drop=True)
    if segs.empty:
        ds.close()
        raise RuntimeError(f"No 60s windows for {stem} ({mode})")
    fingerprint = _analysis_fingerprint(config, row, segs, processed_path, s6b_paths)
    # In overwrite mode invalidate old maps BEFORE any new computation so a failed
    # recalculation cannot leave old, scientifically incompatible maps behind.
    if overwrite:
        for name in ("homotopic_z_mean.npy", "gbc_z_mean.npy",
                     "homotopic_n_segments.npy", "gbc_n_segments.npy",
                     "homotopic_roi_values.csv", "global_roi_values.csv",
                     "segment_processing_log.csv", "windows_used_60s.csv"):
            (out_dirs["stem_maps"] / name).unlink(missing_ok=True)
        (out_dirs["geometry"] / "fc_geometry.json").unlink(missing_ok=True)
        for pp in out_dirs["segments"].glob("segment_*.npz"):
            pp.unlink()
        for pp in out_dirs["roi_traces"].glob("segment_*_roi_traces.npz"):
            pp.unlink()
    # Existing completed stem can be loaded by batch when overwrite=False.
    if (
        mean_h_path.exists()
        and mean_g_path.exists()
        and stem_hroi_csv.exists()
        and stem_groi_csv.exists()
        and not overwrite
    ):
        _assert_existing_output_compatible(
            out_dirs["geometry"] / "fc_geometry.json", config, mask_set
        )
        metadata = json.loads((out_dirs["geometry"] / "fc_geometry.json").read_text(encoding="utf-8"))
        if metadata.get("analysis_fingerprint") != fingerprint:
            ds.close()
            raise RuntimeError(f"Stale Step7 output for {stem}: input/window/processing fingerprint differs; overwrite required")
        ds.close()
        return {
            "status": "skipped_existing",
            "windows_used": segs,
            "mask_set": mask_set,
            "stem": stem,
            "mouse_id": safe_str(row.get("mouse_id")),
            "homotopic_roi": pd.read_csv(stem_hroi_csv),
            "global_roi": pd.read_csv(stem_groi_csv),
            "segment_log": pd.read_csv(stem_seglog_csv) if stem_seglog_csv.exists() else pd.DataFrame(),
            "output_dir": str(out_dirs["stem_maps"]),
            "qc_png": str(out_dirs["qc"] / "fc_maps_allen_overlay.png"),
        }

    processed_shape = (int(var.sizes["y"]), int(var.sizes["x"]))

    geometry = load_stem_geometry(
        row, mask_set, config, paths, s6b_paths, tables, processed_shape
    )
    roi_defs = roi_definitions(tables)
    cfg = step7_cfg(config)

    hom_maps: list[np.ndarray] = []
    gbc_maps: list[np.ndarray] = []
    segment_log_rows: list[dict[str, Any]] = []
    first_preview: np.ndarray | None = None

    for local_idx, (_, sr) in enumerate(segs.iterrows(), start=1):
        t0 = float(sr["t0_sec"])
        t1 = float(sr["t1_sec"])
        try:
            prep = preprocess_segment(var, t0, t1, fps, geometry, config, roi_defs=roi_defs if save_roi_traces else None)
            hmap, pair_roi_ids, pair_z = compute_homotopic_segment_z(
                prep["X_filtered"],
                prep["valid_flat"],
                geometry,
                fisher_clip=float(cfg["homotopic"].get("fisher_clip", 0.999999)),
            )
            gmap, global_counts = compute_global_segment_z(
                prep["X_filtered"], prep["valid_flat"], geometry, config
            )
            if not np.isfinite(hmap).any() and not np.isfinite(gmap).any():
                raise RuntimeError("Both segment FC maps contain no finite correlation values")
            if bool(cfg["output"].get("save_segment_maps", True)):
                np.savez_compressed(
                    out_dirs["segments"] / f"segment_{local_idx:02d}.npz",
                    homotopic_z=hmap,
                    gbc_z=gmap,
                    functional_valid=prep["functional_valid_fc"].astype(np.uint8),
                    t0_sec=np.float64(t0),
                    t1_sec=np.float64(t1),
                    n_valid_homotopic_pairs=np.int32(np.isfinite(hmap).sum()),
                    n_valid_global_pixels=np.int32(np.isfinite(gmap).sum()),
                )

            if save_roi_traces and prep.get("roi_traces"):
                rt = prep["roi_traces"]
                np.savez_compressed(
                    out_dirs["roi_traces"] / f"segment_{local_idx:02d}_roi_traces.npz",
                    **rt,
                    t0_sec=np.float64(t0),
                    t1_sec=np.float64(t1),
                )

            hom_maps.append(hmap)
            gbc_maps.append(gmap)
            if first_preview is None:
                first_preview = prep["preview_frame"]
            segment_log_rows.append(
                {
                    "mask_set": mask_set,
                    "slot": safe_str(row.get("slot")),
                    "mouse_id": geometry.mouse_id,
                    "group_id": geometry.group_id,
                    "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")),
                    "segment_number": local_idx,
                    "source": safe_str(sr.get("source", "")),
                    "t0_sec": t0,
                    "t1_sec": t1,
                    "n_frames": prep["n_frames"],
                    "filter_pad_sec_actual": prep["pad_sec_actual"],
                    "trim_left_frames": prep["trim_left_frames"],
                    "trim_right_frames": prep["trim_right_frames"],
                    "filter_type": str(cfg["temporal"].get("filter_type")),
                    "variant_id": str(cfg.get("active_variant_id")),
                    "rest_variant": str(cfg.get("rest_variant")),
                    "n_functional_valid_pixels": int(len(prep["valid_flat"])),
                    "n_valid_homotopic_pairs": int(np.isfinite(hmap).sum()),
                    "n_valid_global_pixels": int(np.isfinite(gmap).sum()),
                    "global_analysis_choice": geometry.global_analysis_choice,
                    "global_both_hemispheres_mode": geometry.global_both_mode,
                    "global_pool_counts_json": json.dumps(global_counts, ensure_ascii=False),
                    "status": "ok",
                    "error": "",
                }
            )
            del prep
            gc.collect()
        except Exception as exc:
            hom_maps.append(hmap)
            gbc_maps.append(gmap)
            if first_preview is None:
                first_preview = prep["preview_frame"]
            segment_log_rows.append(
                {
                    "mask_set": mask_set,
                    "slot": safe_str(row.get("slot")),
                    "mouse_id": geometry.mouse_id,
                    "group_id": geometry.group_id,
                    "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")),
                    "segment_number": local_idx,
                    "source": safe_str(sr.get("source", "")),
                    "t0_sec": t0,
                    "t1_sec": t1,
                    "status": "error",
                    "error": repr(exc),
                }
            )
            if not bool(cfg["batch"].get("continue_on_error", True)):
                raise

    segment_log = pd.DataFrame(segment_log_rows)
    n_ok = int((segment_log.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum())
    if n_ok == 0:
        raise RuntimeError(f"All accepted REST segments failed for stem={stem}")

    hom_mean, hom_n = nanmean_maps(hom_maps, geometry.fc_shape)
    gbc_mean, gbc_n = nanmean_maps(gbc_maps, geometry.fc_shape)

    if bool(cfg["output"].get("save_stem_maps", True)):
        np.save(mean_h_path, hom_mean.astype(np.float32))
        np.save(out_dirs["stem_maps"] / "homotopic_n_segments.npy", hom_n)
        np.save(mean_g_path, gbc_mean.astype(np.float32))
        np.save(out_dirs["stem_maps"] / "gbc_n_segments.npy", gbc_n)

    geom_files = save_geometry(geometry, out_dirs["geometry"], config, n_ok)
    geometry_path = out_dirs["geometry"] / "fc_geometry.json"
    gm = json.loads(geometry_path.read_text(encoding="utf-8"))
    gm["analysis_fingerprint"] = fingerprint
    geometry_path.write_text(json.dumps(gm, indent=2, ensure_ascii=False), encoding="utf-8")
    segs.to_csv(out_dirs["stem_maps"] / "windows_used_60s.csv", index=False)

    hroi = summarize_homotopic_rois(row, mask_set, geometry, hom_mean, tables, roi_defs, n_ok, config)
    groi = summarize_global_rois(row, mask_set, geometry, gbc_mean, tables, roi_defs, n_ok, config)
    for df in (hroi, groi):
        if df.empty:
            continue
        df["signal"] = str(cfg["signal"])
        df["filter_type"] = str(cfg["temporal"]["filter_type"])
        df["filter_low_hz"] = (float(cfg["temporal"]["low_hz"]) if cfg["temporal"]["filter_type"] == "bandpass" else np.nan)
        df["filter_high_hz"] = float(cfg["temporal"]["high_hz"])
        df["trim_left_sec"] = float(cfg["temporal"]["trim_left_sec"])
        df["trim_right_sec"] = float(cfg["temporal"]["trim_right_sec"])
        df["analysis_fingerprint"] = fingerprint
    hroi.to_csv(stem_hroi_csv, index=False)
    groi.to_csv(stem_groi_csv, index=False)
    segment_log.to_csv(stem_seglog_csv, index=False)

    if save_qc and bool(cfg["output"].get("save_qc_png", True)):
        if first_preview is None:
            first_preview = np.full(geometry.fc_shape, np.nan, dtype=np.float32)
        qc_path = out_dirs["qc"] / "fc_maps_allen_overlay.png"
        save_stem_qc_png(
            qc_path,
            geometry,
            first_preview,
            hom_mean,
            gbc_mean,
            config,
            title=f"{geometry.mouse_id} | {safe_str(row.get('exp_day'))} | {stem} | {mask_set}",
        )

    try:
        ds.close()
    except Exception:
        pass

    return {
        "status": "ok",
        "mask_set": mask_set,
        "stem": stem,
        "mouse_id": geometry.mouse_id,
        "n_segments_ok": n_ok,
        "windows_used": segs,
        "processed_zarr": str(processed_path),
        "geometry_files": geom_files,
        "homotopic_roi": hroi,
        "global_roi": groi,
        "segment_log": segment_log,
        "output_dir": str(out_dirs["stem_maps"]),
        "qc_png": str(out_dirs["qc"] / "fc_maps_allen_overlay.png"),
    }


def _build_long_roi_table(hdf: pd.DataFrame, gdf: pd.DataFrame) -> pd.DataFrame:
    """Combine homotopic and global ROI results into a long-format table.
    
    Parameters
    ----------
    hdf : pd.DataFrame
        Homotopic ROI values table.
    gdf : pd.DataFrame
        Global connectivity ROI values table.
    
    Returns
    -------
    pd.DataFrame
        Result of the documented operation."""
    parts: list[pd.DataFrame] = []
    if not hdf.empty:
        x = hdf.copy()
        x["analysis"] = "homotopic"
        x["roi_key"] = x.get("pair_key", "")
        x["value_z"] = x.get("mean_z", np.nan)
        x["sd_z_out"] = x.get("sd_z", np.nan)
        parts.append(x)
    if not gdf.empty:
        x = gdf.copy()
        x["analysis"] = "global"
        x["value_z"] = x.get("mean_GBC_z", np.nan)
        x["sd_z_out"] = x.get("sd_GBC_z", np.nan)
        parts.append(x)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def run_step7(
    config_path: str | Path,
    *,
    mask_sets: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    overwrite: bool | None = None,
    save_qc: bool | None = None,
    save_roi_traces: bool | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    rest_variant: str | None = None,
    variant_id: str | None = None,
) -> dict[str, Any]:
    # Read configuration before Step6b index to materialize the variant.
    """Execute one Step7 signal/filter/REST variant across its selected recordings.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_sets : Sequence[str] | None
        Optional mask-set names to select; ``None`` uses configuration defaults.
    mice : Sequence[str] | None
        Optional mouse IDs used to restrict the selected recordings.
    stems : Sequence[str] | None
        Optional recording stems used to restrict the selection.
    overwrite : bool | None
        Whether existing stem results should be removed and recomputed.
    save_qc : bool | None
        Whether to save QC preview figures.
    save_roi_traces : bool | None
        Whether to save per-ROI traces for each segment.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    rest_variant : str | None
        ``with_rest`` or ``without_rest`` branch.
    variant_id : str | None
        Identifier of the configured signal/filter combination.
    
    Returns
    -------
    dict[str, Any]
        Index, windows, ROI tables, errors, run metadata, saved paths and run configuration.
    
    Notes
    -----
    Writes variant-specific index, windows, ROI summary, segment log, error and provenance
    tables. A missing processed Zarr is recorded as a technical skip; other failures are logged
    or re-raised according to batch settings."""
    import yaml
    with Path(config_path).open(encoding="utf-8-sig") as fin:
        project_cfg = yaml.safe_load(fin) or {}
    ensure_step7_defaults(project_cfg)
    fc_raw = step7_cfg(project_cfg)
    chosen = variant_id or str(fc_raw.get("active_variant_id", "hbt_bandpass"))
    chosen_rest = rest_variant or str(fc_raw.get("rest_variant", "with_rest"))
    materialized = variant_config(project_cfg, chosen, chosen_rest)["functional_connectivity"]
    _deep_update(materialized, runtime_overrides)
    # The scientific variant identifier defines signal and filter: do not let
    # stale user top-level settings override the named combination.
    materialized = variant_config({"functional_connectivity": materialized}, chosen, chosen_rest)["functional_connectivity"]
    # Partial runs get distinct roots, preventing truncation of full cohort tables.
    if mice or stems:
        subset = {"mice": sorted(str(x) for x in (mice or [])),
                  "stems": sorted(str(x) for x in (stems or [])),
                  "mask_sets": sorted(str(x) for x in (mask_sets or []))}
        materialized["_subset_output_key"] = hashlib.sha256(json.dumps(subset, sort_keys=True).encode()).hexdigest()[:12]
    index, config, paths, s6b_paths, tables = build_step7_index(
        config_path,
        mask_sets=mask_sets,
        mice=mice,
        stems=stems,
        runtime_overrides=materialized,
    )
    step7_paths = build_step7_paths(config, paths)
    cfg = step7_cfg(config)
    if index.empty:
        raise RuntimeError(f"No complete Step6b selections for rest_variant={cfg['rest_variant']!r}, requested mask_sets={mask_sets}")
    if overwrite is None:
        overwrite = bool(cfg["output"].get("overwrite", False))
    if save_qc is None:
        save_qc = bool(cfg["output"].get("save_qc_png", True))
    if save_roi_traces is None:
        save_roi_traces = bool(
            cfg["output"].get(
                "save_roi_traces_one_mouse" if mice and len(mice) == 1 else "save_roi_traces_batch",
                bool(mice and len(mice) == 1),
            )
        )

    branch = str(cfg["rest_variant"])
    if branch == "with_rest" and not index.empty and index["state"].astype(str).eq("Awake").any():
        rest, rest_path = load_final_rest_segments_readonly(index, config, paths)
        target = set(index.loc[index["state"].astype(str).eq("Awake"), "stem"].astype(str))
        rest_used = rest[rest["stem"].astype(str).isin(target)].copy()
    else:
        rest_used = pd.DataFrame(columns=["stem", "t0_sec", "t1_sec", "source"])
        rest_path = "not_used_full_recording"
    # Finalizer windows are read-only. Actual Iso/full-recording windows are
    # logged after reading the Zarr time length in process_stem().
    rest_used.to_csv(step7_paths["rest_used_csv"], index=False)

    if not index.empty:
        counts = rest_used.groupby(rest_used["stem"].astype(str)).size().to_dict() if not rest_used.empty else {}
        index = index.copy()
        index["n_accepted_rest_segments"] = index["stem"].astype(str).map(lambda s: int(counts.get(s, 0)))
        index["step7_output_root"] = index.apply(
            lambda r: str(_stem_output_dirs(step7_paths, safe_str(r["mask_set"]), r)["stem_maps"]), axis=1
        )
    index.to_csv(step7_paths["step7_index_csv"], index=False)

    hom_parts: list[pd.DataFrame] = []
    glob_parts: list[pd.DataFrame] = []
    seg_parts: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stem_results: list[dict[str, Any]] = []
    windows_parts: list[pd.DataFrame] = []

    _stem_iter = index.iterrows()
    _progress = None
    if (
        bool(cfg.get("batch", {}).get("show_progress", True))
        and len(index) > 1
        and _tqdm is not None
    ):
        _progress = _tqdm(
            _stem_iter,
            total=len(index),
            desc="Step7 FC",
            unit="stem",
            dynamic_ncols=True,
        )
        _stem_iter = _progress

    for _, row in _stem_iter:
        stem = safe_str(row.get("stem"))
        mask_set = safe_str(row.get("mask_set"))
        if _progress is not None:
            _progress.set_postfix_str(f"{mask_set}:{stem}", refresh=False)
        try:
            result = process_stem(
                row,
                rest_used,
                config,
                paths,
                s6b_paths,
                step7_paths,
                tables,
                overwrite=bool(overwrite),
                save_roi_traces=bool(save_roi_traces),
                save_qc=bool(save_qc),
            )
            stem_results.append(result)
            if "windows_used" in result:
                w = result["windows_used"].copy()
                for col in ("mask_set", "slot", "mouse_id", "group_id", "state", "comparison_group", "experimental_group"):
                    w[col] = safe_str(row.get(col))
                w["rest_variant"] = branch
                w["variant_id"] = str(cfg["active_variant_id"])
                windows_parts.append(w)
            if not result["homotopic_roi"].empty:
                hom_parts.append(result["homotopic_roi"])
            if not result["global_roi"].empty:
                glob_parts.append(result["global_roi"])
            if not result["segment_log"].empty:
                seg_parts.append(result["segment_log"])
            summary_rows.append(
                {
                    "mask_set": mask_set,
                    "comparison_group": safe_str(row.get("comparison_group")),
                    "experimental_group": safe_str(row.get("experimental_group")),
                    "rest_variant": branch,
                    "variant_id": str(cfg["active_variant_id"]),
                    "mouse_id": safe_str(row.get("mouse_id")),
                    "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")),
                    "status": result.get("status", "ok"),
                    "n_segments_ok": result.get("n_segments_ok", np.nan),
                    "output_dir": result.get("output_dir", ""),
                    "updated_at": now_iso(),
                }
            )
        except Exception as exc:
            # Technical absence of a processed Zarr is an expected skip, not a
            # geometry/FC computation failure (e.g. m16 in this project).
            if isinstance(exc, FileNotFoundError) and str(exc).startswith("Processed zarr missing:"):
                summary_rows.append({
                    "mask_set": mask_set, "mouse_id": safe_str(row.get("mouse_id")),
                    "group_id": safe_str(row.get("group_id")), "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")), "status": "skipped_missing_zarr",
                    "n_segments_ok": 0, "output_dir": "", "reason": str(exc),
                    "updated_at": now_iso(),
                })
                continue
            errors.append(
                {
                    "mask_set": mask_set,
                    "mouse_id": safe_str(row.get("mouse_id")),
                    "group_id": safe_str(row.get("group_id")),
                    "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")),
                    "error": repr(exc),
                    "updated_at": now_iso(),
                }
            )
            summary_rows.append(
                {
                    "mask_set": mask_set,
                    "mouse_id": safe_str(row.get("mouse_id")),
                    "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")),
                    "status": "error",
                    "n_segments_ok": 0,
                    "output_dir": "",
                    "updated_at": now_iso(),
                }
            )
            if not bool(cfg["batch"].get("continue_on_error", True)):
                raise

    hdf = pd.concat(hom_parts, ignore_index=True, sort=False) if hom_parts else pd.DataFrame()
    gdf = pd.concat(glob_parts, ignore_index=True, sort=False) if glob_parts else pd.DataFrame()
    sdf = pd.concat(seg_parts, ignore_index=True, sort=False) if seg_parts else pd.DataFrame()
    edf = pd.DataFrame(errors)
    summary = pd.DataFrame(summary_rows)
    long_df = _build_long_roi_table(hdf, gdf)

    windows_df = pd.concat(windows_parts, ignore_index=True) if windows_parts else pd.DataFrame()
    windows_df.to_csv(step7_paths["windows_csv"], index=False)
    metadata = {"rest_variant": branch, "variant_id": str(cfg["active_variant_id"]),
                "signal": str(cfg["signal"]), "temporal": cfg["temporal"],
                "mask_sets": _selected_mask_sets(config, mask_sets),
                "source_step6b_root": str(s6b_paths["root"]),
                "source_final_rest": str(rest_path),
                "n_stems_selected": int(len(index)), "n_windows": int(len(windows_df)),
                "generated_at": now_iso()}
    step7_paths["run_metadata_json"].write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    hdf.to_csv(step7_paths["homotopic_roi_csv"], index=False)
    gdf.to_csv(step7_paths["global_roi_csv"], index=False)
    sdf.to_csv(step7_paths["segment_log_csv"], index=False)
    edf.to_csv(step7_paths["errors_csv"], index=False)
    summary.to_csv(step7_paths["run_summary_csv"], index=False)
    long_df.to_csv(step7_paths["roi_long_csv"], index=False)

    return {
        "index": index,
        "rest_segments": rest_used,
        "windows_used": windows_df,
        "rest_segments_source": str(rest_path),
        "homotopic_roi_values": hdf,
        "global_roi_values": gdf,
        "roi_values_long": long_df,
        "segment_log": sdf,
        "errors": edf,
        "run_summary": summary,
        "stem_results": stem_results,
        "paths": step7_paths,
        "config": config,
    }



def run_step7_variants(
    config_path: str | Path,
    *,
    rest_variant: str | None = None,
    variants: Sequence[str] | None = None,
    mask_sets: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    overwrite: bool = False,
    save_qc: bool = True,
    save_roi_traces: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run independent configured FC variants and write their registry.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    rest_variant : str | None
        ``with_rest`` or ``without_rest`` branch.
    variants : Sequence[str] | None
        Optional list of configured variant IDs.
    mask_sets : Sequence[str] | None
        Optional mask-set names to select; ``None`` uses configuration defaults.
    mice : Sequence[str] | None
        Optional mouse IDs used to restrict the selected recordings.
    stems : Sequence[str] | None
        Optional recording stems used to restrict the selection.
    overwrite : bool
        Whether existing stem results should be removed and recomputed.
    save_qc : bool
        Whether to save QC preview figures.
    save_roi_traces : bool
        Whether to save per-ROI traces for each segment.
    
    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary of individual Step7 run results keyed by variant ID.
    
    Notes
    -----
    Each selected ID has its own signal/filter configuration and separate output root. The
    helper also runs the frozen-map whole-cortex addon when enabled and writes
    ``variant_registry.csv``."""
    import yaml
    with Path(config_path).open(encoding="utf-8-sig") as fin:
        raw = yaml.safe_load(fin) or {}
    fc = step7_cfg(raw)
    selected = list(variants if variants is not None else fc.get("variants", {}).get("selected", []))
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Expected nonempty, unique list of Step7 variants")
    results: dict[str, dict[str, Any]] = {}
    for name in selected:
        print(f"[Step7] {rest_variant or fc['rest_variant']} / {name}")
        results[name] = run_step7(
            config_path, mask_sets=mask_sets, mice=mice, stems=stems,
            overwrite=overwrite, save_qc=save_qc, save_roi_traces=save_roi_traces,
            rest_variant=rest_variant, variant_id=name,
        )
        result = results[name]
        if not result["run_summary"].empty and bool(step7_cfg(result["config"]).get("whole_cortex_summary", {}).get("enabled", True)):
            result["whole_cortex_summary"] = run_whole_cortex_summary_from_existing_step7(
                config_path, mask_sets=mask_sets, mice=mice, stems=stems,
                runtime_overrides=result["config"]["functional_connectivity"],
                continue_on_error=True, show_progress=False,
            )
    if results:
        first = next(iter(results.values()))
        first_root = Path(first["paths"]["root"])
        is_subset = bool(mice or stems)
        root = first_root.parents[2] if is_subset else first_root.parent  # <step7>/<rest_variant>
        registry = []
        for name, result in results.items():
            p = result["paths"]
            registry.append({"rest_variant": str(result["config"]["functional_connectivity"]["rest_variant"]),
                             "variant_id": name, "signal": str(result["config"]["functional_connectivity"]["signal"]),
                             "filter_type": str(result["config"]["functional_connectivity"]["temporal"]["filter_type"]),
                             "root": str(p["root"]), "homotopic_roi_values_csv": str(p["homotopic_roi_csv"]),
                             "global_roi_values_csv": str(p["global_roi_csv"]),
                             "whole_cortex_fc_mouse_values_csv": str(p["tables_root"] / "whole_cortex_fc_mouse_values.csv"),
                             "step7_index_csv": str(p["step7_index_csv"]),
                             "n_selected": len(result["index"]),
                             "n_errors": len(result["errors"])})
        if is_subset:
            registry_dir = root / "subset_registries"
            registry_dir.mkdir(exist_ok=True, parents=True)
            registry_path = registry_dir / (str(first["config"]["functional_connectivity"]["_subset_output_key"]) + ".csv")
        else:
            registry_path = root / "variant_registry.csv"
        pd.DataFrame(registry).to_csv(registry_path, index=False)
    return results


# =============================================================================
# v4_1 publication addon: frozen-map whole-cortex / whole-hemisphere summaries
# =============================================================================

STEP7_CODE_VERSION = "step7_integrated"
STEP7_FROZEN_MAP_SOURCE_VERSION = "step7_integrated"


def default_whole_cortex_summary_config() -> dict[str, Any]:
    """Return default settings for the frozen-output whole-cortex summary addon.
    
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation."""
    return {
        "enabled": True,
        "save_stem_table": True,
        "save_mouse_table": True,
        "homotopic": {"enabled": True},
        "global": {
            "enabled": True,
            "save_sides_when_both": True,
            "save_bilateral_when_both": True,
            "single_hemisphere_primary": True,
        },
        "qc": {
            "min_valid_homotopic_pairs": 1,
            "min_valid_gbc_pixels": 1,
        },
    }


def whole_cortex_summary_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Merge the optional whole-cortex summary settings with defaults.
    
    Parameters
    ----------
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation."""
    raw = copy.deepcopy(
        (config.get("functional_connectivity", {}) or {}).get(
            "whole_cortex_summary", {}
        )
        or {}
    )
    return _deep_fill_missing(raw, default_whole_cortex_summary_config())


def _summary_table_paths(step7_paths: dict[str, Path]) -> dict[str, Path]:
    """Resolve the addon paths for stem, mouse and error CSV tables.
    
    Parameters
    ----------
    step7_paths : dict[str, Path]
        Path mapping for Step7 outputs.
    
    Returns
    -------
    dict[str, Path]
        Result of the documented operation."""
    tables = Path(step7_paths["tables_root"])
    return {
        "stem_csv": tables / "whole_cortex_fc_stem_values.csv",
        "mouse_csv": tables / "whole_cortex_fc_mouse_values.csv",
        "errors_csv": tables / "whole_cortex_fc_summary_errors.csv",
    }


def _existing_step7_index_for_summary(
    config_path: str | Path,
    *,
    mask_sets: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path], dict[str, Path]]:
    """Load the previously saved Step7 index without recalculating FC maps.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_sets : Sequence[str] | None
        Optional mask-set names to select; ``None`` uses configuration defaults.
    mice : Sequence[str] | None
        Optional mouse IDs used to restrict the selected recordings.
    stems : Sequence[str] | None
        Optional recording stems used to restrict the selection.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    
    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any], dict[str, Path], dict[str, Path]]
        Result of the documented operation."""
    _, config, paths = s6b.build_step6_index(
        config_path,
        include_bad=True,
        filters={},
        require_response_roi=False,
    )
    ensure_step7_defaults(config)
    _deep_update(config["functional_connectivity"], runtime_overrides)
    step7_paths = build_step7_paths(config, paths)
    index_path = Path(step7_paths["step7_index_csv"])
    if not index_path.exists():
        raise FileNotFoundError(
            f"Saved Step7 index not found: {index_path}. "
            "The v4_1 addon requires an already completed Step7 v4 run."
        )
    index = pd.read_csv(index_path)
    if index.empty:
        raise RuntimeError(f"Saved Step7 index is empty: {index_path}")

    out = index.copy()
    if mask_sets:
        wanted = {str(x) for x in mask_sets}
        out = out[out["mask_set"].astype(str).isin(wanted)].copy()
    if mice:
        wanted = {str(x) for x in mice}
        if "mouse_id" not in out.columns:
            raise KeyError("Saved Step7 index has no mouse_id column")
        out = out[out["mouse_id"].astype(str).isin(wanted)].copy()
    if stems:
        wanted = {str(x) for x in stems}
        out = out[out["stem"].astype(str).isin(wanted)].copy()

    sort_cols = [c for c in ["mask_set", "mouse_id", "slot", "stem"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out, config, paths, step7_paths


def _load_frozen_geometry_bundle(
    step7_paths: dict[str, Path],
    row: pd.Series,
) -> dict[str, Any]:
    """Load the frozen geometry and FC maps needed for summary-only analysis.
    
    Parameters
    ----------
    step7_paths : dict[str, Path]
        Path mapping for Step7 outputs.
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation.
    
    Notes
    -----
    Uses the stored stem geometry, pair lists and maps; it does not reproject the Allen atlas
    or recalculate time-series correlations."""
    mask_set = safe_str(row.get("mask_set"))
    dirs = _stem_output_dirs(step7_paths, mask_set, row)
    geometry_dir = dirs["geometry"]
    maps_dir = dirs["stem_maps"]

    required = {
        "geometry_json": geometry_dir / "fc_geometry.json",
        "pairs": geometry_dir / "homotopic_pairs_fc.npz",
        "left": geometry_dir / "left_hemisphere_fc.npy",
        "right": geometry_dir / "right_hemisphere_fc.npy",
        "hom_map": maps_dir / "homotopic_z_mean.npy",
        "gbc_map": maps_dir / "gbc_z_mean.npy",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "v4_1 summary addon cannot run because frozen Step7 v4 outputs are missing: "
            + "; ".join(missing)
        )

    geom = json.loads(required["geometry_json"].read_text(encoding="utf-8"))
    hom_map = np.load(required["hom_map"]).astype(np.float32, copy=False)
    gbc_map = np.load(required["gbc_map"]).astype(np.float32, copy=False)
    left = np.load(required["left"]).astype(bool)
    right = np.load(required["right"]).astype(bool)
    if not (hom_map.shape == gbc_map.shape == left.shape == right.shape):
        raise ValueError(
            f"Frozen Step7 arrays have inconsistent shapes for stem={row.get('stem')}: "
            f"hom={hom_map.shape}, gbc={gbc_map.shape}, left={left.shape}, right={right.shape}"
        )

    with np.load(required["pairs"], allow_pickle=False) as d:
        pair_src_flat = np.asarray(d["src_flat"], dtype=np.int64)
        pair_dst_flat = np.asarray(d["dst_flat"], dtype=np.int64)
        pair_roi_id = np.asarray(d["roi_id"], dtype=np.int32)

    if pair_src_flat.size and (
        int(pair_src_flat.min()) < 0 or int(pair_src_flat.max()) >= hom_map.size
    ):
        raise ValueError("Saved homotopic pair source indices are outside the saved homotopic map")

    pool_keys = [safe_str(x) for x in geom.get("global_pool_keys", []) if safe_str(x)]
    pools: dict[str, np.ndarray] = {}
    for key in pool_keys:
        p = geometry_dir / f"global_pool_{key}_final_usable_fc.npy"
        if not p.exists():
            raise FileNotFoundError(
                f"Saved global pool mask listed in fc_geometry.json is missing: {p}"
            )
        arr = np.load(p).astype(bool)
        if arr.shape != gbc_map.shape:
            raise ValueError(f"Global pool {key} shape {arr.shape} != GBC map {gbc_map.shape}")
        pools[key] = arr

    if not pools:
        # Defensive fallback for a very old frozen geometry JSON: discover masks
        # by filename, but never reconstruct them from current Step6b geometry.
        for p in sorted(geometry_dir.glob("global_pool_*_final_usable_fc.npy")):
            key = p.name[len("global_pool_") : -len("_final_usable_fc.npy")]
            arr = np.load(p).astype(bool)
            if arr.shape == gbc_map.shape:
                pools[key] = arr
    if not pools:
        raise FileNotFoundError(
            f"No frozen global_pool_*_final_usable_fc.npy masks found in {geometry_dir}"
        )

    return {
        "dirs": dirs,
        "geometry": geom,
        "hom_map": hom_map,
        "gbc_map": gbc_map,
        "left": left,
        "right": right,
        "pair_src_flat": pair_src_flat,
        "pair_dst_flat": pair_dst_flat,
        "pair_roi_id": pair_roi_id,
        "global_pools": pools,
    }


def _finite_summary(
    values: np.ndarray,
    *,
    n_geometry: int,
    n_rest_segments: int,
    min_valid: int,
) -> dict[str, Any]:
    """Calculate finite-element Fisher-z summaries and minimum-data QC outcomes.
    
    Parameters
    ----------
    values : np.ndarray
        Input numerical values; nonfinite elements are excluded from summaries.
    n_geometry : int
        Number of geometrically eligible FC elements.
    n_rest_segments : int
        Number of successfully processed windows.
    min_valid : int
        Minimum number of finite map elements for QC PASS.
    
    Returns
    -------
    dict[str, Any]
        Computed z/r estimates, counts, gate flags, reasons and publication-facing values.
    
    Notes
    -----
    Computes ``mean_z`` and its inverse-Fisher ``tanh(mean_z)``. Publication-facing values
    become NaN when minimum valid-element or minimum-segment gates fail; diagnostic values
    remain available."""
    vals = np.asarray(values, dtype=np.float64).ravel()
    vals = vals[np.isfinite(vals)]
    n_valid = int(vals.size)
    mean_z = float(np.mean(vals)) if n_valid else np.nan
    sd_z = float(np.std(vals, ddof=1)) if n_valid > 1 else np.nan

    reasons: list[str] = []
    if int(n_rest_segments) <= 0:
        reasons.append("no_rest_segments")
    if n_valid < int(min_valid):
        reasons.append(f"n_valid<{int(min_valid)}")
    if not np.isfinite(mean_z):
        reasons.append("mean_z_nonfinite")
    qc_pass = len(reasons) == 0
    computed_r = float(np.tanh(mean_z)) if np.isfinite(mean_z) else np.nan
    return {
        "computed_value_z_qc": mean_z,
        "computed_value_r_qc": computed_r,
        "value_sd_z": sd_z,
        "n_valid_elements_fc": n_valid,
        "n_geometry_elements_fc": int(n_geometry),
        "n_rest_segments": int(n_rest_segments),
        "qc_pass": bool(qc_pass),
        "qc_reason": "ok" if qc_pass else ";".join(reasons),
        "analysis_pass": bool(qc_pass),
        "value_z": mean_z if qc_pass else np.nan,
        "value_r": computed_r if qc_pass else np.nan,
    }


def _base_whole_cortex_record(
    row: pd.Series,
    config: dict[str, Any],
    geom: dict[str, Any],
) -> dict[str, Any]:
    """Assemble provenance and recording metadata for scalar FC summaries.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    geom : dict[str, Any]
        Frozen geometry metadata loaded from the completed Step7 run.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation."""
    tc = step7_cfg(config).get("temporal", {}) or {}
    sc = step7_cfg(config).get("spatial", {}) or {}
    hc = step7_cfg(config).get("homotopic", {}) or {}
    global_choice = safe_str(geom.get("global_analysis_choice", ""))
    both_mode = safe_str(geom.get("global_both_hemispheres_mode", ""))
    return {
        "mask_set": safe_str(row.get("mask_set")),
        "slot": safe_str(row.get("slot")),
        "group_id": safe_str(row.get("group_id")),
        "mouse_id": safe_str(row.get("mouse_id")) or safe_str(row.get("group_id")),
        "experimental_group": safe_str(row.get("experimental_group")),
            "comparison_group": safe_str(row.get("comparison_group")),
            "rest_variant": safe_str(row.get("rest_variant")),
            "variant_id": safe_str(step7_cfg(config).get("active_variant_id")),
        "stem": safe_str(row.get("stem")),
        "exp_day": safe_str(row.get("exp_day")),
        "state": safe_str(row.get("state")),
        "protocol": safe_str(row.get("protocol")),
        "signal": safe_str(step7_cfg(config).get("signal", "dcHbT")),
        "filter_type": safe_str(tc.get("filter_type")),
        "trim_left_sec": safe_float(tc.get("trim_left_sec", 2.0)),
        "trim_right_sec": safe_float(tc.get("trim_right_sec", 2.0)),
        "rest_variant": safe_str(step7_cfg(config).get("rest_variant")),
        "variant_id": safe_str(step7_cfg(config).get("active_variant_id")),
        "comparison_group": safe_str(row.get("comparison_group")),
        "filter_low_hz": safe_float(tc.get("low_hz", np.nan)) if str(tc.get("filter_type")) == "bandpass" else np.nan,
        "filter_high_hz": safe_float(tc.get("high_hz", np.nan)),
        "spatial_scale": safe_float(geom.get("spatial_scale", sc.get("scale", np.nan))),
        "global_analysis_choice_step7": global_choice,
        "global_both_hemispheres_mode": both_mode,
        "global_pool_includes_cross_hemisphere": bool(
            global_choice == "both" and both_mode == "whole_cortex"
        ),
        "homotopic_map_hemisphere": safe_str(geom.get("homotopic_map_side", "")),
        "require_same_analysis_roi": bool(hc.get("require_same_analysis_roi", True)),
        "mutual_nearest_partner": bool(hc.get("mutual_nearest_partner", True)),
        "source_step7_version": STEP7_CODE_VERSION,
        "source_frozen_maps_version": STEP7_FROZEN_MAP_SOURCE_VERSION,
    }


def _homotopic_whole_cortex_row(
    row: pd.Series,
    bundle: dict[str, Any],
    config: dict[str, Any],
    summary_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate homotopic map values across accepted source pixels.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    bundle : dict[str, Any]
        Saved FC maps, geometry metadata and pairing/pooling masks.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    summary_cfg : dict[str, Any]
        Effective settings for the frozen-output whole-cortex addon.
    
    Returns
    -------
    dict[str, Any]
        Result of the documented operation."""
    geom = bundle["geometry"]
    src = bundle["pair_src_flat"]
    hom = bundle["hom_map"]
    vals = hom.ravel()[src] if src.size else np.empty(0, dtype=np.float32)
    n_rest = int(geom.get("n_accepted_rest_segments", 0) or 0)
    stats = _finite_summary(
        vals,
        n_geometry=int(src.size),
        n_rest_segments=n_rest,
        min_valid=int(summary_cfg["qc"].get("min_valid_homotopic_pairs", 1)),
    )
    return {
        **_base_whole_cortex_record(row, config, geom),
        "fc_kind": "homotopic",
        "metric": "homotopic_pixelwise_whole_cortex_z",
        "hemisphere_scope": "whole_cortex_unique_pairs",
        "global_pool": "",
        **stats,
        # Explicit aliases requested by the publication contract.
        "computed_whole_cortex_homotopic_mean_z_qc": stats["computed_value_z_qc"],
        "computed_whole_cortex_homotopic_sd_z_qc": stats["value_sd_z"],
        "n_valid_homotopic_pairs_whole_cortex_fc": stats["n_valid_elements_fc"],
        "n_geometry_homotopic_pairs_whole_cortex_fc": stats["n_geometry_elements_fc"],
        "whole_cortex_homotopic_qc_pass": stats["qc_pass"],
        "whole_cortex_homotopic_qc_reason": stats["qc_reason"],
    }


def _gbc_scope_specs(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate the saved GBC spatial pools and compatible hemisphere scopes.
    
    Parameters
    ----------
    bundle : dict[str, Any]
        Saved FC maps, geometry metadata and pairing/pooling masks.
    
    Returns
    -------
    list[dict[str, Any]]
        Result of the documented operation."""
    geom = bundle["geometry"]
    left = bundle["left"]
    right = bundle["right"]
    pools: dict[str, np.ndarray] = bundle["global_pools"]
    choice = safe_str(geom.get("global_analysis_choice", "")).strip().lower()
    mode = safe_str(geom.get("global_both_hemispheres_mode", "")).strip().lower()
    healthy_side = safe_str(geom.get("healthy_side", "")).strip().lower()
    lesion_side = safe_str(geom.get("lesion_side", "")).strip().lower()

    specs: list[dict[str, Any]] = []
    if choice == "both":
        if mode == "whole_cortex":
            if "whole_cortex" not in pools:
                raise KeyError("Frozen geometry says whole_cortex mode but whole_cortex pool mask is missing")
            whole = pools["whole_cortex"]
            left_seed = whole & left
            right_seed = whole & right
            bilateral = left_seed | right_seed
            specs.extend(
                [
                    {
                        "metric": "GBC_pixelwise_left_cortex_z",
                        "hemisphere_scope": "left",
                        "mask": left_seed,
                        "global_pool": "whole_cortex",
                    },
                    {
                        "metric": "GBC_pixelwise_right_cortex_z",
                        "hemisphere_scope": "right",
                        "mask": right_seed,
                        "global_pool": "whole_cortex",
                    },
                    {
                        "metric": "GBC_pixelwise_bilateral_cortex_z",
                        "hemisphere_scope": "bilateral",
                        "mask": bilateral,
                        "global_pool": "whole_cortex",
                    },
                ]
            )
        else:
            if "left" not in pools or "right" not in pools:
                raise KeyError("Frozen geometry says separate hemispheres but left/right pool masks are missing")
            left_seed = pools["left"] & left
            right_seed = pools["right"] & right
            bilateral = left_seed | right_seed
            specs.extend(
                [
                    {
                        "metric": "GBC_pixelwise_left_cortex_z",
                        "hemisphere_scope": "left",
                        "mask": left_seed,
                        "global_pool": "left",
                    },
                    {
                        "metric": "GBC_pixelwise_right_cortex_z",
                        "hemisphere_scope": "right",
                        "mask": right_seed,
                        "global_pool": "right",
                    },
                    {
                        "metric": "GBC_pixelwise_bilateral_cortex_z",
                        "hemisphere_scope": "bilateral",
                        # Pixel-weighted union, NOT (left_mean + right_mean)/2.
                        "mask": bilateral,
                        "global_pool": "left+right_separate",
                    },
                ]
            )
        return specs

    # Single primary analysis hemisphere. The frozen pool key is the actual side
    # (left/right), while the semantic choice may be healthy/lesion/left/right.
    if len(pools) != 1:
        raise RuntimeError(
            f"Expected one frozen GBC pool for global_analysis_choice={choice!r}, got {list(pools)}"
        )
    pool_key, pool_mask = next(iter(pools.items()))
    actual_side = pool_key if pool_key in {"left", "right"} else ""
    if choice == "healthy":
        metric = "GBC_pixelwise_healthy_hemisphere_z"
        scope = "healthy"
        if healthy_side in {"left", "right"} and actual_side and actual_side != healthy_side:
            raise RuntimeError(
                f"Frozen healthy GBC pool side mismatch: pool={actual_side}, healthy_side={healthy_side}"
            )
    elif choice == "lesion":
        metric = "GBC_pixelwise_lesion_hemisphere_z"
        scope = "lesion"
        if lesion_side in {"left", "right"} and actual_side and actual_side != lesion_side:
            raise RuntimeError(
                f"Frozen lesion GBC pool side mismatch: pool={actual_side}, lesion_side={lesion_side}"
            )
    elif choice in {"left", "right"}:
        metric = f"GBC_pixelwise_{choice}_cortex_z"
        scope = choice
    else:
        metric = "GBC_pixelwise_whole_hemisphere_z"
        scope = choice or actual_side or "primary"

    specs.append(
        {
            "metric": metric,
            "hemisphere_scope": scope,
            "mask": pool_mask,
            "global_pool": pool_key,
        }
    )
    return specs


def _gbc_summary_rows(
    row: pd.Series,
    bundle: dict[str, Any],
    config: dict[str, Any],
    summary_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Construct whole-cortex or hemisphere-level scalar GBC result rows.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    bundle : dict[str, Any]
        Saved FC maps, geometry metadata and pairing/pooling masks.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    summary_cfg : dict[str, Any]
        Effective settings for the frozen-output whole-cortex addon.
    
    Returns
    -------
    list[dict[str, Any]]
        Result of the documented operation."""
    geom = bundle["geometry"]
    gbc = bundle["gbc_map"]
    n_rest = int(geom.get("n_accepted_rest_segments", 0) or 0)
    out: list[dict[str, Any]] = []
    global_cfg = summary_cfg.get("global", {}) or {}
    choice = safe_str(geom.get("global_analysis_choice", "")).strip().lower()
    for spec in _gbc_scope_specs(bundle):
        if choice == "both":
            if spec["hemisphere_scope"] in {"left", "right"} and not bool(
                global_cfg.get("save_sides_when_both", True)
            ):
                continue
            if spec["hemisphere_scope"] == "bilateral" and not bool(
                global_cfg.get("save_bilateral_when_both", True)
            ):
                continue
        elif not bool(global_cfg.get("single_hemisphere_primary", True)):
            continue
        mask = np.asarray(spec["mask"], dtype=bool)
        vals = gbc[mask]
        stats = _finite_summary(
            vals,
            n_geometry=int(mask.sum()),
            n_rest_segments=n_rest,
            min_valid=int(summary_cfg["qc"].get("min_valid_gbc_pixels", 1)),
        )
        out.append(
            {
                **_base_whole_cortex_record(row, config, geom),
                "fc_kind": "global",
                "metric": spec["metric"],
                "hemisphere_scope": spec["hemisphere_scope"],
                "global_pool": spec["global_pool"],
                **stats,
            }
        )
    return out


def summarize_existing_step7_stem(
    row: pd.Series,
    config: dict[str, Any],
    step7_paths: dict[str, Path],
) -> pd.DataFrame:
    """Derive scalar summaries for one recording from completed FC maps.
    
    Parameters
    ----------
    row : pd.Series
        Metadata for the selected Step6b/Step7 recording.
    config : dict[str, Any]
        Project configuration including the Step7 settings.
    step7_paths : dict[str, Path]
        Path mapping for Step7 outputs.
    
    Returns
    -------
    pd.DataFrame
        Result of the documented operation."""
    summary_cfg = whole_cortex_summary_cfg(config)
    bundle = _load_frozen_geometry_bundle(step7_paths, row)
    rows: list[dict[str, Any]] = []
    if bool(summary_cfg.get("homotopic", {}).get("enabled", True)):
        rows.append(_homotopic_whole_cortex_row(row, bundle, config, summary_cfg))
    if bool(summary_cfg.get("global", {}).get("enabled", True)):
        rows.extend(_gbc_summary_rows(row, bundle, config, summary_cfg))
    return pd.DataFrame(rows)


def _unique_or_empty(series: pd.Series) -> tuple[str, bool]:
    """Represent unique metadata values and report conflicts.
    
    Parameters
    ----------
    series : pd.Series
        Sequence of source metadata values to check for consistency.
    
    Returns
    -------
    tuple[str, bool]
        Result of the documented operation."""
    vals = [safe_str(v) for v in series if safe_str(v).strip()]
    uniq = list(dict.fromkeys(vals))
    if len(uniq) <= 1:
        return (uniq[0] if uniq else ""), True
    return "|".join(uniq), False


def build_whole_cortex_mouse_table(stem_df: pd.DataFrame) -> pd.DataFrame:
    """Average valid stem Fisher-z estimates with equal stem weight within each mouse.
    
    Parameters
    ----------
    stem_df : pd.DataFrame
        Stem-level whole-cortex connectivity summary table.
    
    Returns
    -------
    pd.DataFrame
        One row per mouse and analysis scope, containing the equal-weight Fisher-z mean and QC provenance.
    
    Notes
    -----
    Only stems passing analysis QC contribute, with each valid stem receiving equal weight.
    Provenance conflicts for global pooling, signal or other critical methods block publication
    PASS; averaging occurs in Fisher-z space before the inverse transform."""
    if stem_df.empty:
        return pd.DataFrame()
    group_cols = [
        "mask_set",
        "slot",
        "mouse_id",
        "experimental_group",
        "comparison_group",
        "rest_variant",
        "variant_id",
        "fc_kind",
        "metric",
        "hemisphere_scope",
    ]
    missing = [c for c in group_cols if c not in stem_df.columns]
    if missing:
        raise KeyError(f"Stem summary table missing columns needed for mouse aggregation: {missing}")

    rows: list[dict[str, Any]] = []
    for key, g in stem_df.groupby(group_cols, dropna=False, sort=False):
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        valid = g[
            g["analysis_pass"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
            & pd.to_numeric(g["value_z"], errors="coerce").notna()
        ].copy()

        incompat: list[str] = []
        provenance_cols = [
            "global_analysis_choice_step7",
            "global_both_hemispheres_mode",
            "global_pool_includes_cross_hemisphere",
            "global_pool",
            "homotopic_map_hemisphere",
            "require_same_analysis_roi",
            "mutual_nearest_partner",
            "signal",
            "filter_type",
            "trim_left_sec", "trim_right_sec",
            "filter_low_hz",
            "filter_high_hz",
            "rest_variant", "variant_id", "comparison_group",
            "spatial_scale",
        ]
        prov: dict[str, Any] = {}
        for col in provenance_cols:
            if col not in g.columns:
                continue
            val, ok = _unique_or_empty(valid[col] if not valid.empty else g[col])
            prov[col] = val
            # GBC pool semantics and FC settings must not be silently mixed.
            if not ok and col in {
                "global_analysis_choice_step7",
                "global_both_hemispheres_mode",
                "global_pool_includes_cross_hemisphere",
                "global_pool",
                "homotopic_map_hemisphere",
                "require_same_analysis_roi",
                "mutual_nearest_partner",
                "signal",
                "filter_low_hz",
                "filter_high_hz",
                "spatial_scale",
            }:
                incompat.append(f"mixed_{col}")

        n_source = int(len(valid))
        if incompat:
            mean_z = np.nan
            reason = ";".join(incompat)
            pass_pub = False
        elif n_source == 0:
            mean_z = np.nan
            reason = "no_valid_source_stems"
            pass_pub = False
        else:
            mean_z = float(pd.to_numeric(valid["value_z"], errors="coerce").mean())
            reason = "ok" if np.isfinite(mean_z) else "mouse_mean_nonfinite"
            pass_pub = bool(np.isfinite(mean_z))

        source_stems = sorted({safe_str(x) for x in valid.get("stem", pd.Series(dtype=str)) if safe_str(x)})
        exp_days = sorted({safe_str(x) for x in valid.get("exp_day", pd.Series(dtype=str)) if safe_str(x)})
        states = sorted({safe_str(x) for x in valid.get("state", pd.Series(dtype=str)) if safe_str(x)})
        protocols = sorted({safe_str(x) for x in valid.get("protocol", pd.Series(dtype=str)) if safe_str(x)})

        rows.append(
            {
                **rec,
                "value_z": mean_z if pass_pub else np.nan,
                "value_r": float(np.tanh(mean_z)) if pass_pub else np.nan,
                "n_source_stems": n_source,
                "n_source_stems_total": int(len(g)),
                "source_stems": "|".join(source_stems),
                "source_exp_days": "|".join(exp_days),
                "source_states": "|".join(states),
                "source_protocols": "|".join(protocols),
                "analysis_pass_publication": bool(pass_pub),
                "publication_qc_reason": reason,
                **prov,
                "source_step7_version": STEP7_CODE_VERSION,
                "source_frozen_maps_version": STEP7_FROZEN_MAP_SOURCE_VERSION,
            }
        )
    return pd.DataFrame(rows)


def run_whole_cortex_summary_from_existing_step7(
    config_path: str | Path,
    *,
    mask_sets: Sequence[str] | None = None,
    mice: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    runtime_overrides: dict[str, Any] | None = None,
    continue_on_error: bool = True,
    show_progress: bool = True,
    save: bool = True,
) -> dict[str, Any]:
    """Rebuild stem- and mouse-level whole-cortex tables from frozen Step7 products.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_sets : Sequence[str] | None
        Optional mask-set names to select; ``None`` uses configuration defaults.
    mice : Sequence[str] | None
        Optional mouse IDs used to restrict the selected recordings.
    stems : Sequence[str] | None
        Optional recording stems used to restrict the selection.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    continue_on_error : bool
        Continue to subsequent recordings after an individual summary error.
    show_progress : bool
        Display a progress bar when the optional dependency is present.
    save : bool
        Whether to write summary CSV files.
    
    Returns
    -------
    dict[str, Any]
        Stem and mouse summary tables, errors, paths and run metadata.
    
    Notes
    -----
    Reads previously saved Step7 index/geometry/maps and can write scalar stem/mouse CSVs
    without reopening processed signal Zarr."""
    index, config, paths, step7_paths = _existing_step7_index_for_summary(
        config_path,
        mask_sets=mask_sets,
        mice=mice,
        stems=stems,
        runtime_overrides=runtime_overrides,
    )
    summary_cfg = whole_cortex_summary_cfg(config)
    if not bool(summary_cfg.get("enabled", True)):
        raise RuntimeError("functional_connectivity.whole_cortex_summary.enabled is false")

    parts: list[pd.DataFrame] = []
    errors: list[dict[str, Any]] = []
    iterator: Any = index.iterrows()
    if show_progress and len(index) > 1 and _tqdm is not None:
        iterator = _tqdm(
            iterator,
            total=len(index),
            desc="Step7 v4_1 summaries",
            unit="stem",
            dynamic_ncols=True,
        )

    for _, row in iterator:
        stem = safe_str(row.get("stem"))
        mask_set = safe_str(row.get("mask_set"))
        try:
            part = summarize_existing_step7_stem(row, config, step7_paths)
            if not part.empty:
                parts.append(part)
        except Exception as exc:
            # Technical absence of a processed Zarr is an expected skip, not a
            # geometry/FC computation failure (e.g. m16 in this project).
            if isinstance(exc, FileNotFoundError) and str(exc).startswith("Processed zarr missing:"):
                summary_rows.append({
                    "mask_set": mask_set, "mouse_id": safe_str(row.get("mouse_id")),
                    "group_id": safe_str(row.get("group_id")), "stem": stem,
                    "exp_day": safe_str(row.get("exp_day")), "status": "skipped_missing_zarr",
                    "n_segments_ok": 0, "output_dir": "", "reason": str(exc),
                    "updated_at": now_iso(),
                })
                continue
            errors.append(
                {
                    "mask_set": mask_set,
                    "mouse_id": safe_str(row.get("mouse_id")),
                    "group_id": safe_str(row.get("group_id")),
                    "stem": stem,
                    "error": repr(exc),
                    "updated_at": now_iso(),
                }
            )
            if not continue_on_error:
                raise

    stem_df = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    mouse_df = build_whole_cortex_mouse_table(stem_df) if not stem_df.empty else pd.DataFrame()
    error_df = pd.DataFrame(errors)
    out_paths = _summary_table_paths(step7_paths)

    if save:
        if bool(summary_cfg.get("save_stem_table", True)):
            stem_df.to_csv(out_paths["stem_csv"], index=False)
        if bool(summary_cfg.get("save_mouse_table", True)):
            mouse_df.to_csv(out_paths["mouse_csv"], index=False)
        error_df.to_csv(out_paths["errors_csv"], index=False)

    return {
        "index": index,
        "stem_values": stem_df,
        "mouse_values": mouse_df,
        "errors": error_df,
        "paths": out_paths,
        "step7_paths": step7_paths,
        "config": config,
        "summary_config": summary_cfg,
    }

# =============================================================================
# Notebook-friendly QC loaders/plots
# =============================================================================


def _find_index_row_for_output(
    config_path: str | Path,
    mask_set: str,
    stem: str,
    runtime_overrides: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any], dict[str, Path]]:
    """Locate one selected recording and its variant-specific output paths.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_set : str
        Name of the selected Step6b mask set.
    stem : str
        Unique recording stem identifier.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    
    Returns
    -------
    tuple[pd.Series, dict[str, Any], dict[str, Path]]
        Result of the documented operation."""
    idx, config, paths, _, _ = build_step7_index(
        config_path,
        mask_sets=[mask_set],
        stems=[stem],
        runtime_overrides=runtime_overrides,
    )
    if idx.empty:
        raise KeyError(f"No Step7 index row for mask_set={mask_set}, stem={stem}")
    return idx.iloc[0], config, build_step7_paths(config, paths)


def plot_stem_maps(
    config_path: str | Path,
    *,
    mask_set: str,
    stem: str,
    figsize: tuple[float, float] = (13, 6),
    runtime_overrides: dict[str, Any] | None = None,
) -> plt.Figure:
    """Plot the saved homotopic and GBC maps for a selected recording.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_set : str
        Name of the selected Step6b mask set.
    stem : str
        Unique recording stem identifier.
    figsize : tuple[float, float]
        Figure width and height in Matplotlib inches.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure displaying saved FC maps.
    
    Notes
    -----
    This is a visualization of saved Fisher-z arrays, not a new correlation calculation."""
    row, config, p7 = _find_index_row_for_output(
        config_path, mask_set, stem, runtime_overrides=runtime_overrides
    )
    dirs = _stem_output_dirs(p7, mask_set, row)
    labels = np.load(dirs["geometry"] / "analysis_roi_labels_fc.npy")
    hmap = np.load(dirs["stem_maps"] / "homotopic_z_mean.npy")
    gmap = np.load(dirs["stem_maps"] / "gbc_z_mean.npy")
    pcts = step7_cfg(config)["qc"].get("map_percentiles", [2.0, 98.0])
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    h0, h1 = _robust_limits(hmap, np.isfinite(hmap), pcts)
    g0, g1 = _robust_limits(gmap, np.isfinite(gmap), pcts)
    im0 = axes[0].imshow(hmap, cmap="jet", vmin=h0, vmax=h1)
    _roi_boundary_contours(axes[0], labels, linewidth=0.6)
    axes[0].set_title("Homotopic mean Fisher-z")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(gmap, cmap="jet", vmin=g0, vmax=g1)
    _roi_boundary_contours(axes[1], labels, linewidth=0.6)
    axes[1].set_title("GBC mean Fisher-z")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    geom_json_path = dirs["geometry"] / "fc_geometry.json"
    mode_note = ""
    if geom_json_path.exists():
        gj = json.loads(geom_json_path.read_text(encoding="utf-8"))
        mode_note = (
            f" | global={safe_str(gj.get('global_analysis_choice'))}"
            f"/{safe_str(gj.get('global_both_hemispheres_mode'))}"
        )
    fig.suptitle(
        f"{safe_str(row.get('mouse_id'))} | {safe_str(row.get('exp_day'))} | {stem}{mode_note}"
    )
    fig.tight_layout()
    return fig


def plot_stem_roi_traces(
    config_path: str | Path,
    *,
    mask_set: str,
    stem: str,
    roi_acronym: str,
    hemisphere: str = "healthy",
    stage: str = "filtered",
    figsize: tuple[float, float] = (12, 6),
    runtime_overrides: dict[str, Any] | None = None,
) -> plt.Figure:
    """Plot saved segment-wise ROI time courses for a selected recording.
    
    Parameters
    ----------
    config_path : str | Path
        Path to the project YAML configuration.
    mask_set : str
        Name of the selected Step6b mask set.
    stem : str
        Unique recording stem identifier.
    roi_acronym : str
        Atlas-based ROI acronym for optional time-trace plotting.
    hemisphere : str
        Hemisphere selector for optional ROI trace plotting.
    stage : str
        ``raw``, ``detrended`` or ``filtered`` trace stage.
    figsize : tuple[float, float]
        Figure width and height in Matplotlib inches.
    runtime_overrides : dict[str, Any] | None
        Optional overrides for the Step7 configuration, applied at runtime.
    
    Returns
    -------
    plt.Figure
        Matplotlib figure displaying ROI traces by accepted segment.
    
    Notes
    -----
    Traces are loaded from optional per-segment NPZ products, not recomputed from raw
    recordings."""
    row, config, p7 = _find_index_row_for_output(
        config_path, mask_set, stem, runtime_overrides=runtime_overrides
    )
    dirs = _stem_output_dirs(p7, mask_set, row)
    geom_json = json.loads((dirs["geometry"] / "fc_geometry.json").read_text(encoding="utf-8"))
    side = hemisphere.strip().lower()
    if side == "healthy":
        side = safe_str(geom_json.get("healthy_side"))
    elif side == "lesion":
        side = safe_str(geom_json.get("lesion_side"))
    if side not in {"left", "right"}:
        raise ValueError("hemisphere must be healthy/lesion/left/right")
    target = f"{roi_acronym}_{side}".lower()

    files = sorted(dirs["roi_traces"].glob("segment_*_roi_traces.npz"))
    if not files:
        raise FileNotFoundError(
            f"No saved ROI traces for {stem}. Run one-mouse Step7 with save_roi_traces=True."
        )
    fig, ax = plt.subplots(figsize=figsize)
    n = 0
    for fp in files:
        with np.load(fp, allow_pickle=False) as d:
            names = [str(x) for x in d["names"]]
            idxs = [i for i, name in enumerate(names) if name.lower() == target]
            if not idxs:
                continue
            i = idxs[0]
            if stage not in d.files:
                raise KeyError(f"Trace stage {stage!r} not in {fp}; available={d.files}")
            y = np.asarray(d[stage][i], dtype=float)
            t = np.asarray(d["time_sec"], dtype=float)
            t0 = float(d["t0_sec"])
            ax.plot(t, y, linewidth=1.1, alpha=0.8, label=f"REST {n+1} (t0={t0:g}s)")
            n += 1
    if n == 0:
        raise KeyError(f"ROI trace {roi_acronym}_{side} not found in saved trace files")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time within accepted REST (s)")
    ax.set_ylabel(f"{stage} {step7_cfg(config).get('signal', 'dcHbT')} mean")
    ax.set_title(f"{safe_str(row.get('mouse_id'))} | {safe_str(row.get('exp_day'))} | {roi_acronym}_{side}")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser for Step7.
    
    
    Returns
    -------
    argparse.ArgumentParser
        Result of the documented operation."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mask-set", action="append", default=None)
    p.add_argument("--mouse", action="append", default=None)
    p.add_argument("--stem", action="append", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rest-variant", choices=["with_rest", "without_rest"], default=None)
    p.add_argument("--variant", action="append", default=None)
    p.add_argument("--no-qc", action="store_true")
    p.add_argument("--save-roi-traces", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI options and execute the requested Step7 variants.
    
    Parameters
    ----------
    argv : Sequence[str] | None
        Optional CLI arguments; ``None`` uses command-line inputs."""
    args = build_arg_parser().parse_args(argv)
    results = run_step7_variants(
        args.config, rest_variant=args.rest_variant, variants=args.variant,
        mask_sets=args.mask_set, mice=args.mouse, stems=args.stem,
        overwrite=bool(args.overwrite), save_qc=not bool(args.no_qc),
        save_roi_traces=bool(args.save_roi_traces),
    )
    for name, result in results.items():
        print(f"\n{name}: {len(result['run_summary'])} stem rows, {len(result['errors'])} errors; root={result['paths']['root']}")


if __name__ == "__main__":
    main()
