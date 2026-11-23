"""Step 5 response ROI detection/extraction for the stem-key WIFIOPIA pipeline, v1_5.

This module ports the old ``Step3b_ResponseDetection.ipynb`` idea to the current
Step1-Step4 pipeline:

- inputs are Step4 processed zarr stores, one row/stem at a time;
- the response drawing mask is a manual napari polygon ROI;
- the display/extraction mask is Step4-compatible brainvessels_signal:

    brain = brain_manual if it exists else brain_auto
    bregma = bregma_manual if it exists else bregma_auto
    bad = bad_auto
    brainvessels_signal = brain & ~bad & ~bregma & saved_raw_valid

The intended workflow is interactive and notebook-friendly:

1. ``build_step5_index(...)``
2. choose one ``row``
3. ``preview = build_response_preview_for_row(row, config, paths)``
4. ``roi_mask = draw_response_roi_napari(preview, row, config)``
5. ``save_response_roi_mask(...)`` and ``extract_and_save_response_traces(...)``

The CLI can also be used for one or several selected stems.  It will open napari
for each selected row unless ``--no-napari`` is passed, in which case it expects
an existing ROI mask and only extracts traces/plots.  v1_5 keeps notebook-friendly
separation of napari drawing and mask saving, plus optional batch metadata filters
for selecting only stimulation recordings.

Publication note
----------------
The current implementation deliberately restricts eligible recordings to
Day8/Awake/Stim and unaligned processed Step4 signals. These checks are
in the Python code, not generalized by YAML metadata filters. The
stimulation parameters and visualization values in the built-in
defaults are fallback examples; user-provided YAML values take priority."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

try:  # napari is optional until the interactive function is called.
    import napari
except Exception:  # pragma: no cover
    napari = None

try:
    from skimage.draw import polygon as sk_polygon
except Exception:  # pragma: no cover
    sk_polygon = None

from common import (
    build_standard_paths,
    canonical_mask_paths,
    create_or_load_project,
    get_path_resolving_config,
    load_config,
    resolve_flexible_path,
)


# ============================================================
# Config and paths
# ============================================================


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return a nested configuration value without modifying the mapping.

    Parameters
    ----------
    config : dict
        Project configuration mapping.
    *keys : str
        Nested YAML keys to access, in order.
    default : Any, optional
        Value returned if any key is absent or a level is not a mapping.

    Returns
    -------
    Any
        Retrieved value or ``default``.
    """
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_update_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Fill missing nested keys in ``dst`` from ``src`` without overwriting user values."""
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_update_missing(dst[key], value)
    return dst


def default_step5_config() -> dict[str, Any]:
    """Return the built-in default settings for the Step5 workflow.

    Returns
    -------
    dict
        Default ``step5`` configuration block, including output paths,
        response signals, stimulation timing, plotting and interactive review.

    Notes
    -----
    These are fallback settings, not universal experimental parameters. Existing
    YAML values take precedence via ``ensure_step5_defaults``. Some Step5
    eligibility conditions are hard-coded elsewhere and cannot be changed by
    these defaults.
    """
    return {
        "folder_name": "step5_response_detection",
        "response_masks_drawn_folder": "response_masks_drawn",
        "response_tables_folder": "response_tables",
        "response_plots_folder": "response_plots",
        "response_previews_folder": "response_previews",
        "roi_log_name": "response_roi_log.csv",
        "overwrite_roi": False,
        "input": {
            "hemoglobin_signals": ["dcHbT_gsub", "dcHbO_gsub", "dcHHb_gsub"],
        },
        "napari": {
            "image_layer_name": "Avg hbt Stimulus",
            "roi_layer_name": "ROI",
            "brain_mask_layer_name": "brain_signal_mask",
            "existing_roi_layer_name": "existing_response_roi",
            "colormap": "turbo",
            "contrast_limits": [0, 1],
            "display_smoothing_sigma_px": 1.0,  # Napari preview only; no change to Hb/ROI traces.
            "title_size": 28,
            "show_brain_mask": True,
            "show_existing_roi": True,
        },
        "stimulus": {
            "stim_count": 4,
            "stim_start_frame": 6000,
            "stim_step": 1500,
            "baseline_frames": 20,
            "preview_window_frames": [0, 100],
            "trial_window_frames": [-20, 1480],
            # The pulse frequency comes from the YAML settings, not a fixed frame interval.
            "pulse_count": 4,
            "pulse_frequency_hz": 1.0,
            "pulse_duration_sec": 0.3,
        },
        "plot": {
            "enabled": True,
            "error_type": "sem",
            "label_position": "center",
            "dpi": 200,
            "save_formats": ["png", "tiff"],
            "figsize": [10, 5],
            "hb_y_label": "Δ[Hb] (µM)",
            "x_label": "Time from trial window start (s)",
            "x_lim": None,
            "hb_y_lim": None,
            "line_width": 1.4,
            "error_alpha": 0.10,
            "colors": {
                "dcHbT_gsub": "magenta",
                "dcHbO_gsub": "red",
                "dcHHb_gsub": "blue",
                "stimulus": "gray",
                "zero_line": "black",
            },
            "labels": {
                "dcHbT_gsub": "HbT",
                "dcHbO_gsub": "HbO",
                "dcHHb_gsub": "HHb",
            },
            "stimulus_boxes": {
                "enabled": True,
                "alpha": 0.20,
                "label": "Stimulus",
                "label_y_fraction": 0.95,
            },
            "signal_annotations": {
                "enabled": True,
                "use_arrows": True,
                "use_bboxes": True,
                "fontsize": 13,
                "bbox_facecolor": "white",
                "x_offsets_sec": {
                    "dcHbT_gsub": 2,
                    "dcHbO_gsub": -4,
                    "dcHHb_gsub": 5,
                },
                "y_fractions": {
                    "dcHbT_gsub": 0.65,
                    "dcHbO_gsub": 0.45,
                    "dcHHb_gsub": 0.25,
                },
            },
        },
        "output": {
            "save_roi_npy": True,
            "save_roi_bmp": True,
            "save_roi_png": True,
            "save_preview_png": True,
            "save_traces_csv": True,
            "save_stim_trials_csv": True,
            "save_summary_json": True,
        },
        "batch": {
            # Extra exact/regex metadata filters used when build_step5_index(..., batch_mode=True)
            # or when the CLI is called with --batch. Examples:
            # metadata_filters: {"state": "Awake", "protocol": {"regex": "RHL|LHL"}}
            "metadata_filters": None,
            # Optional broad stimulation selector. Keep disabled by default so old behavior is unchanged.
            "only_stimulation": False,
            "stimulation_column": "protocol",
            "stimulation_include_regex": "(?i)(?:RHL|LHL|RE|LE|RV|LV|stim)",
            "stimulation_exclude_values": ["Awake", "Baseline", "Rest", "Spontaneous"],
            # Legacy batch option; enabling it is rejected by build_step5_index for safety.
            "skip_existing_roi": False,
            # Keep batch runs moving if one stem has truncated frames or another
            # recoverable Step5 problem.
            "continue_on_error": True,
        },
        "debug": {
            "filters": None,
            "selected_stems": None,
            "selected_groups": None,
        },
    }


def ensure_step5_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Populate missing Step5 settings without overriding user configuration.

    Parameters
    ----------
    config : dict
        Mutable full project configuration; updated in place.

    Returns
    -------
    dict
        The same configuration mapping with missing ``step5`` keys filled.
    """
    config.setdefault("step5", {})
    _deep_update_missing(config["step5"], default_step5_config())
    return config


def load_step5_config(config_path: str | Path) -> dict[str, Any]:
    """Load the YAML project configuration and fill missing Step5 defaults.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the project YAML file.

    Returns
    -------
    dict
        Full configuration with an initialized ``step5`` block.
    """
    config = load_config(config_path)
    return ensure_step5_defaults(config)


def build_step5_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create and return the Step5 output directories and log path.

    Parameters
    ----------
    config : dict
        Full configuration; directory names are read from ``step5``.
    paths : dict of pathlib.Path
        Existing paths from ``build_standard_paths``; must include ``out_root``.

    Returns
    -------
    dict of pathlib.Path
        Step5 root and subdirectories for ROI masks, tables, previews, plots,
        and the CSV review log.

    Notes
    -----
    This function creates directories as a side effect.
    """
    step5_cfg = ensure_step5_defaults(config)["step5"]
    root = Path(paths["out_root"]) / step5_cfg.get("folder_name", "step5_response_detection")
    out = {
        "step5_root": root,
        "response_masks_drawn_root": root / step5_cfg.get("response_masks_drawn_folder", "response_masks_drawn"),
        "response_tables_root": root / step5_cfg.get("response_tables_folder", "response_tables"),
        "response_plots_root": root / step5_cfg.get("response_plots_folder", "response_plots"),
        "response_previews_root": root / step5_cfg.get("response_previews_folder", "response_previews"),
        "response_roi_log_csv": root / step5_cfg.get("roi_log_name", "response_roi_log.csv"),
    }
    for p in out.values():
        if str(p).lower().endswith(".csv"):
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# Index
# ============================================================


def _as_list_or_none(value: Any) -> list[str] | None:
    """Normalize an optional scalar or collection to a list of strings.

    Returns
    -------
    list of str or None
        String representations of provided values; ``None`` remains ``None``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _bool_series(s: pd.Series) -> pd.Series:
    """Interpret a pandas Series of flag values as booleans.

    Notes
    -----
    Only case-insensitive ``true``, ``1``, ``yes`` and ``y`` map to True.
    """
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _merge_filter_dicts(*items: dict[str, Any] | None) -> dict[str, Any] | None:
    """Combine optional metadata filter mappings, with later keys winning.

    Returns
    -------
    dict or None
        Merged filter mapping, or ``None`` if all inputs are empty.
    """
    merged: dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        merged.update(dict(item))
    return merged or None


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Filter Step5 rows by exact values, inclusion/exclusion or text matching.

    Parameters
    ----------
    df : pandas.DataFrame
        Recording index with metadata columns.
    filters : dict or None
        Mapping of metadata column names to exact values, lists or dictionaries
        using ``include``, ``exclude``, ``regex`` and/or ``contains``.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of the input table.

    Notes
    -----
    Example values such as ``Awake`` or ``RHL`` are illustrative, not required
    by this filter. All requested metadata columns must already exist.
    """
    out = df.copy()
    for key, allowed_raw in (filters or {}).items():
        if key not in out.columns:
            raise ValueError(f"Step5 filter column not found: {key}")
        values = out[key].astype(str)

        if isinstance(allowed_raw, dict):
            mask = pd.Series(True, index=out.index)
            include = set(_as_list_or_none(allowed_raw.get("include")) or [])
            exclude = set(_as_list_or_none(allowed_raw.get("exclude")) or [])
            if include:
                mask &= values.isin(include)
            if exclude:
                mask &= ~values.isin(exclude)
            regex = allowed_raw.get("regex")
            if regex not in (None, ""):
                mask &= values.str.contains(str(regex), regex=True, na=False)
            contains = allowed_raw.get("contains")
            if contains not in (None, ""):
                mask &= values.str.contains(str(contains), regex=False, na=False)
            out = out[mask].copy()
            continue

        allowed = set(_as_list_or_none(allowed_raw) or [])
        if allowed:
            out = out[values.isin(allowed)].copy()
    return out


def _apply_stimulation_filter(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    only_stimulation: bool | None = None,
    stimulation_column: str | None = None,
    stimulation_include_regex: str | None = None,
    stimulation_exclude_values: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Optionally filter recordings using protocol metadata text.

    Parameters
    ----------
    df : pandas.DataFrame
        Step5 recording index.
    config : dict
        Full configuration with optional ``step5.batch`` filter settings.
    only_stimulation : bool or None, optional
        Overrides ``step5.batch.only_stimulation``.
    stimulation_column : str or None, optional
        Metadata column to inspect, from ``step5.batch.stimulation_column`` by
        default.
    stimulation_include_regex : str or None, optional
        Inclusion expression from ``step5.batch.stimulation_include_regex``.
    stimulation_exclude_values : sequence of str or None, optional
        Exact metadata labels to exclude.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of the recording index.

    Notes
    -----
    This optional selector runs *after* the hard-coded Day8/Awake/Stim
    restriction in ``build_step5_index``. It does not generalize that cohort.
    """
    batch_cfg = cfg_get(config, "step5", "batch", default={}) or {}
    if only_stimulation is None:
        only_stimulation = bool(batch_cfg.get("only_stimulation", False))
    if not bool(only_stimulation):
        return df.copy()

    col = str(stimulation_column or batch_cfg.get("stimulation_column", "protocol"))
    if col not in df.columns:
        raise ValueError(f"Step5 stimulation filter column not found: {col}")
    values = df[col].astype(str)

    regex = stimulation_include_regex
    if regex is None:
        regex = batch_cfg.get("stimulation_include_regex")
    exclude_values = list(stimulation_exclude_values or batch_cfg.get("stimulation_exclude_values") or [])

    mask = pd.Series(True, index=df.index)
    if regex not in (None, ""):
        mask &= values.str.contains(str(regex), regex=True, na=False)
    if exclude_values:
        mask &= ~values.str.strip().isin({str(v) for v in exclude_values})
    return df[mask].copy()


def _parse_cli_metadata_filters(filter_items: Sequence[str] | None, filters_json: str | None) -> dict[str, Any] | None:
    """Convert CLI metadata filter arguments to a Step5 filter mapping.

    Parameters
    ----------
    filter_items : sequence of str or None
        Repeated ``KEY=VALUE`` or ``KEY=a,b`` arguments.
    filters_json : str or None
        JSON object containing filter expressions.

    Returns
    -------
    dict or None
        Parsed metadata filters or ``None`` when not provided.

    Raises
    ------
    ValueError
        If a filter is malformed or JSON does not decode to a mapping.
    """
    filters: dict[str, Any] = {}
    if filters_json:
        loaded = json.loads(filters_json)
        if not isinstance(loaded, dict):
            raise ValueError("--filters-json must decode to a JSON object")
        filters.update(loaded)
    for item in filter_items or []:
        if "=" not in item:
            raise ValueError(f"Metadata filter must be KEY=VALUE or KEY=a,b, got: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty metadata filter key in: {item!r}")
        if "," in value:
            filters[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            filters[key] = value
    return filters or None


def _resolve_optional_path(value: Any, path_cfg: dict[str, Any], *, must_exist: bool = False) -> Path | None:
    """Resolve a potentially empty path using configured drive/root remapping.

    Parameters
    ----------
    value : Any
        Original path value, potentially missing.
    path_cfg : dict
        Path mappings from the project configuration.
    must_exist : bool, optional
        Whether to require an existing path.

    Returns
    -------
    pathlib.Path or None
        Resolved path, or ``None`` for missing optional values.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
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


def build_step5_index(
    config_path: str | Path,
    *,
    include_bad: bool | None = None,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    require_processed: bool = False,
    batch_mode: bool = False,
    only_stimulation: bool | None = None,
    stimulation_column: str | None = None,
    stimulation_include_regex: str | None = None,
    stimulation_exclude_values: Sequence[str] | None = None,
    skip_existing_roi: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Build a recording-level Step5 index from Step1 and Step2 metadata.

    Parameters
    ----------
    config_path : str or pathlib.Path
        YAML project configuration file.
    include_bad : bool or None, optional
        Include rows flagged as bad in Step2; default is False.
    filters : dict or None, optional
        Exact or text-based additional metadata filters.
    groups, stems : sequence of str or None, optional
        Restrict to the specified group IDs or recording stems.
    require_processed : bool, optional
        Keep only recordings with an existing processed Zarr.
    batch_mode : bool, optional
        Apply batch-specific defaults. YAML ``step5.batch.metadata_filters`` is
        merged only when the explicit ``filters`` argument is ``None``.
    only_stimulation, stimulation_column, stimulation_include_regex,
    stimulation_exclude_values : optional
        Additional metadata-based stimulation selector options.
    skip_existing_roi : bool or None, optional
        Request to skip existing ROIs; if enabled, an error is deliberately raised.

    Returns
    -------
    pandas.DataFrame
        Selected recordings with resolved raw/processed paths and expected
        ROI/trace/preview output paths.
    dict
        Loaded project configuration with Step5 defaults.
    dict of pathlib.Path
        Standard project paths extended with Step5 output directories.

    Notes
    -----
    The current implementation has a fixed experimental cohort restriction:
    ``exp_day == 'Day8'``, ``state == 'Awake'`` and
    ``protocol_base == 'Stim'``. Optional YAML/CLI filters only narrow this
    selection; they do not replace it. The module also requires
    ``signal_processing.input.raw_variant == 'unaligned'`` and, when protocol
    variants are enabled, selects the maximum repeat for each identity group.
    These are current code restrictions, not general YAML-configurable defaults.
    """
    config_path = Path(config_path)
    config = load_step5_config(config_path)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)
    paths = {**paths, **build_step5_paths(config, paths)}

    debug_cfg = cfg_get(config, "step5", "debug", default={}) or {}
    batch_cfg = cfg_get(config, "step5", "batch", default={}) or {}
    if filters is None:
        filters = debug_cfg.get("filters")
        if batch_mode:
            filters = _merge_filter_dicts(filters, batch_cfg.get("metadata_filters"))
    if groups is None:
        groups = _as_list_or_none(debug_cfg.get("selected_groups"))
    if stems is None:
        stems = _as_list_or_none(debug_cfg.get("selected_stems"))

    manifest_path = paths["manifest_path"]
    metadata_path = paths["metadata_registration_csv"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_recordings.csv not found: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_registration.csv not found: {metadata_path}")

    manifest = pd.read_csv(manifest_path)
    meta_reg = pd.read_csv(metadata_path)
    if "stem" not in manifest.columns or "stem" not in meta_reg.columns:
        raise ValueError("Both manifest_recordings.csv and metadata_registration.csv must contain 'stem'.")

    if manifest["stem"].duplicated().any() or meta_reg["stem"].duplicated().any():
        raise ValueError("Duplicate stem in Step5 manifest or Step2 metadata")
    manifest_extra_cols = [c for c in manifest.columns if c in {"stem", "img_dir"} or c not in meta_reg.columns]
    df = meta_reg.drop(columns=["img_dir"], errors="ignore").merge(
        manifest[manifest_extra_cols], on="stem", how="left", validate="one_to_one"
    )
    if "img_dir" not in df.columns or df["img_dir"].isna().any():
        raise ValueError("Step5 metadata contains stem(s) missing from Step1 manifest")

    if include_bad is None:
        include_bad = False
    if not include_bad and "is_bad" in df.columns:
        df = df[~_bool_series(df["is_bad"])].copy()

    # Hard-coded eligibility for the validated Day8 / Awake / Stim cohort.
    # Additional YAML and CLI filters can only narrow this selection.
    # Numeric protocol repeats are evaluated through protocol_base; stem is unchanged.
    if "protocol_base" not in df.columns:
        raise ValueError("Step5 requires protocol_base in Step2 metadata_registration.csv")
    df = df[
        (df["exp_day"].astype(str) == "Day8")
        & (df["state"].astype(str) == "Awake")
        & (df["protocol_base"].astype(str) == "Stim")
    ].copy()
    excluded = set(config.get("excluded_stems") or [])
    df = df[~df["stem"].astype(str).isin(excluded)].copy()
    if df["stem"].duplicated().any() or manifest["stem"].duplicated().any() or meta_reg["stem"].duplicated().any():
        raise ValueError("Duplicate stem in Step5 manifest / metadata index")
    if bool(cfg_get(config, "variant_keys", "protocol", "enabled", default=False)):
        key_cols = ["mouse_id", "exp_day", "state", "protocol_base"]
        if "protocol_repeat" not in df.columns or any(k not in df.columns for k in key_cols):
            raise ValueError("Step5 requires protocol_repeat and protocol_base for variant selection")
        repeats = pd.to_numeric(df["protocol_repeat"], errors="coerce").fillna(0)
        df = df[repeats.eq(repeats.groupby([df[k] for k in key_cols]).transform("max"))].copy()
    if filters and bool(cfg_get(config, "variant_keys", "protocol", "enabled", default=False)):
        filters = {("protocol_base" if k == "protocol" else k): v for k, v in filters.items()}
    df = _apply_filters(df, filters)
    df = _apply_stimulation_filter(
        df,
        config,
        only_stimulation=only_stimulation if only_stimulation is not None else (batch_cfg.get("only_stimulation") if batch_mode else False),
        stimulation_column=stimulation_column,
        stimulation_include_regex=stimulation_include_regex,
        stimulation_exclude_values=stimulation_exclude_values,
    )
    if groups:
        if "group_id" not in df.columns:
            raise ValueError("groups filter requires group_id column in metadata_registration.csv")
        df = df[df["group_id"].astype(str).isin({str(g) for g in groups})].copy()
    if stems:
        df = df[df["stem"].astype(str).isin({str(s) for s in stems})].copy()

    if str(cfg_get(config, "signal_processing", "input", "raw_variant", default="unaligned")) != "unaligned":
        raise ValueError("Step5 approved experiment requires signal_processing.input.raw_variant: unaligned")
    path_cfg = get_path_resolving_config(config, "step4")
    raw_root = paths["raw_zarr_root"]
    processed_root = paths["processed_signals_root"]

    img_dirs: list[str] = []
    raw_paths: list[str] = []
    proc_paths: list[str] = []
    raw_exists: list[bool] = []
    proc_exists: list[bool] = []
    for _, row in df.iterrows():
        stem = str(row["stem"])

        img_dir = _resolve_optional_path(row.get("img_dir", ""), path_cfg, must_exist=False)
        img_dirs.append(str(img_dir) if img_dir else "")

        raw_p = None
        raw_path = row.get("raw_zarr_path", "")
        if str(raw_path).strip():
            raw_p = _resolve_optional_path(raw_path, path_cfg, must_exist=False)
        if raw_p is None:
            raw_p = raw_root / f"{stem}.zarr"
        raw_paths.append(str(raw_p))
        raw_exists.append(raw_p.exists())

        proc_p = None
        processed_path = row.get("processed_zarr_path", "")
        if str(processed_path).strip():
            proc_p = _resolve_optional_path(processed_path, path_cfg, must_exist=False)
        if proc_p is None:
            proc_p = processed_root / f"{stem}_processed.zarr"
        proc_paths.append(str(proc_p))
        proc_exists.append(proc_p.exists())

    df["img_dir_resolved"] = img_dirs
    df["raw_zarr_path_resolved"] = raw_paths
    df["processed_zarr_path_resolved"] = proc_paths
    df["raw_zarr_exists"] = raw_exists
    df["processed_zarr_exists"] = proc_exists

    if require_processed:
        df = df[df["processed_zarr_exists"].astype(bool)].copy()

    roi_paths: list[str] = []
    roi_exists: list[bool] = []
    trace_paths: list[str] = []
    plot_paths: list[str] = []
    preview_paths: list[str] = []
    for _, row in df.iterrows():
        stem = str(row["stem"])
        roi_path = paths["response_masks_drawn_root"] / f"{stem}_response_roi.npy"
        roi_paths.append(str(roi_path))
        roi_exists.append(roi_path.exists())
        trace_paths.append(str(paths["response_tables_root"] / f"{stem}_roi_traces.csv"))
        plot_paths.append(str(paths["response_plots_root"] / f"{stem}_response_plot.png"))
        preview_paths.append(str(paths["response_previews_root"] / f"{stem}_response_preview.png"))

    df["response_roi_path"] = roi_paths
    df["response_roi_exists"] = roi_exists
    df["response_traces_csv_path"] = trace_paths
    df["response_plot_png_path"] = plot_paths
    df["response_preview_png_path"] = preview_paths

    if skip_existing_roi is None:
        skip_existing_roi = bool(batch_cfg.get("skip_existing_roi", False)) if batch_mode else False
    if bool(skip_existing_roi):
        raise ValueError("Step5 skip_existing_roi by filename is unsafe; review ROI status and Step4 fingerprint instead")

    return df.reset_index(drop=True), config, paths


# ============================================================
# Masks
# ============================================================


def resize_mask_to_shape(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a spatial mask to the requested image shape by nearest neighbor.

    Parameters
    ----------
    mask : numpy.ndarray
        Input mask of shape (y, x).
    image_shape : tuple of int
        Target (height, width), in pixels.

    Returns
    -------
    numpy.ndarray
        Boolean mask with shape ``image_shape``.

    Notes
    -----
    Nearest-neighbor resizing preserves discrete mask values; it does not
    recompute the mask in the target image's coordinate system.
    """
    target_h, target_w = int(image_shape[0]), int(image_shape[1])
    if tuple(mask.shape) == (target_h, target_w):
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _load_npy_mask(path: Path, *, image_shape: tuple[int, int], resize_if_needed: bool) -> np.ndarray:
    """Read a canonical NumPy mask and optionally resize it.

    Parameters
    ----------
    path : pathlib.Path
        Existing ``.npy`` mask file.
    image_shape : tuple of int
        Expected (y, x) array shape.
    resize_if_needed : bool
        Whether nearest-neighbor spatial resizing is allowed.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the requested spatial shape.

    Raises
    ------
    FileNotFoundError
        If the mask file does not exist.
    ValueError
        If the shape differs and resizing is disabled.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")
    mask = np.load(path).astype(bool)
    if tuple(mask.shape) != tuple(image_shape):
        if not resize_if_needed:
            raise ValueError(
                f"Mask shape mismatch for {path}: {mask.shape} vs {image_shape}. "
                "Set signal_processing.masks.resize_to_raw_if_needed: true or rebuild matching zarr/masks."
            )
        mask = resize_mask_to_shape(mask, image_shape)
    return mask.astype(bool)


def load_canonical_brain_signal_mask(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    image_shape: tuple[int, int],
    *,
    processed_ds: xr.Dataset,
) -> tuple[np.ndarray, dict[str, str]]:
    """Construct the Step5 spatial signal mask from saved Step2 and Step4 masks.

    Parameters
    ----------
    row : pandas.Series
        Recording metadata, including ``stem``.
    config : dict
        Full configuration, including ``signal_processing.masks``.
    paths : dict of pathlib.Path
        Canonical Step2 mask root and project paths.
    image_shape : tuple of int
        Processed image dimensions (y, x) in pixels.
    processed_ds : xarray.Dataset
        Completed Step4 dataset with saved ``raw_valid`` (y, x).

    Returns
    -------
    numpy.ndarray
        Boolean ``brainvessels_signal`` mask, shape (y, x).
    dict of str
        Source paths, mask precedence and coordinate provenance.

    Notes
    -----
    The mask is ``brain & ~bad & ~bregma & (raw_valid == 1)``. Existing
    manual brain/bregma masks take precedence over automatic ones, while
    ``bad_auto`` is always used. Mask resizing follows
    ``signal_processing.masks.resize_to_raw_if_needed``. All masks are applied
    in the processed-Zarr spatial grid.
    """
    stem = str(row["stem"])
    cps = canonical_mask_paths(paths, stem)
    resize_if_needed = bool(cfg_get(config, "signal_processing", "masks", "resize_to_raw_if_needed", default=True))

    brain_path = cps["brain_manual"] if cps["brain_manual"].exists() else cps["brain_auto"]
    bregma_path = cps["bregma_manual"] if cps["bregma_manual"].exists() else cps["bregma_auto"]
    bad_path = cps["bad_auto"]

    brain = _load_npy_mask(brain_path, image_shape=image_shape, resize_if_needed=resize_if_needed)
    bregma = _load_npy_mask(bregma_path, image_shape=image_shape, resize_if_needed=resize_if_needed)
    bad = _load_npy_mask(bad_path, image_shape=image_shape, resize_if_needed=resize_if_needed)

    valid = processed_ds["raw_valid"].transpose("y", "x").compute().values
    if tuple(valid.shape) != tuple(image_shape):
        raise ValueError(f"raw_valid shape {valid.shape} != processed image {image_shape}")
    brain_signal = brain & (~bad) & (~bregma) & (valid == 1)
    sources = {
        "brain_path": str(brain_path),
        "brain_source": "manual" if brain_path.name == "brain_manual.npy" else "auto",
        "bregma_path": str(bregma_path),
        "bregma_source": "manual" if bregma_path.name == "bregma_manual.npy" else "auto",
        "bad_path": str(bad_path),
        "mask_formula": "brainvessels_signal = brain & ~bad & ~bregma & (saved raw_valid == 1)",
        "raw_valid_source": "processed_zarr:raw_valid",
        "image_shape": json.dumps(list(map(int, image_shape))),
    }
    return brain_signal.astype(bool), sources


# ============================================================
# Processed zarr and signal selection
# ============================================================


def open_processed_zarr(path: str | Path) -> xr.Dataset:
    """Open and validate a completed Step4 processed Zarr dataset.

    Parameters
    ----------
    path : str or pathlib.Path
        Processed Zarr directory.

    Returns
    -------
    xarray.Dataset
        Dataset containing ``raw_valid`` and recorded Step4 provenance.

    Raises
    ------
    ValueError
        If ``step4_complete``, ``step4_fingerprint`` or ``raw_valid`` is absent.

    Notes
    -----
    Uses ``mask_and_scale=False``. If automatic Dask chunking cannot be imported,
    it retries opening without Dask chunks.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed zarr not found: {path}")
    
    try:
        ds = xr.open_zarr(path, chunks="auto", consolidated=False, mask_and_scale=False)
    except (ImportError, ModuleNotFoundError):
        ds = xr.open_zarr(path, chunks=None, consolidated=False, mask_and_scale=False)
    if ds.attrs.get("step4_complete") is not True or not ds.attrs.get("step4_fingerprint"):
        raise ValueError(f"Incomplete or untracked Step4 processed Zarr: {path}")
    if "raw_valid" not in ds:
        raise ValueError(f"Step4 processed Zarr lacks saved raw_valid: {path}")
    return ds


def _normalise_candidates(value: Any) -> list[str]:
    """Normalize one or more candidate variable names to strings.

    Returns
    -------
    list of str
        Nonempty candidate names; an absent value gives an empty list.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def first_existing_signal(ds: xr.Dataset, candidates: Any, *, required: bool = True) -> str | None:
    """Find the first candidate variable present in a processed dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Available processed signals.
    candidates : Any
        One or more requested variable names in priority order.
    required : bool, optional
        Raise ``KeyError`` instead of returning ``None`` if none is found.

    Returns
    -------
    str or None
        First available candidate signal name.
    """
    for name in _normalise_candidates(candidates):
        if name in ds.data_vars:
            return name
    if required:
        raise KeyError(
            f"None of the requested signals were found. candidates={_normalise_candidates(candidates)}; "
            f"available={list(ds.data_vars)}"
        )
    return None


def selected_hemoglobin_signals(ds: xr.Dataset, config: dict[str, Any]) -> list[str]:
    """Validate and return the explicitly selected Step5 Hb signals.

    Parameters
    ----------
    ds : xarray.Dataset
        Processed Step4 signals.
    config : dict
        Full configuration with ``step5.input.hemoglobin_signals``.

    Returns
    -------
    list of str
        One to three distinct names from ``dcHbT_gsub``, ``dcHbO_gsub`` and
        ``dcHHb_gsub``, all required to exist in the dataset.

    Raises
    ------
    ValueError
        If the requested channels are not distinct or not approved.
    KeyError
        If any requested signal is missing from the dataset.

    Notes
    -----
    There is no automatic substitution of other processing variants.
    """
    names = list(cfg_get(config, "step5", "input", "hemoglobin_signals", default=[
        "dcHbT_gsub", "dcHbO_gsub", "dcHHb_gsub",
    ]))
    allowed = {"dcHbT_gsub", "dcHbO_gsub", "dcHHb_gsub"}
    if not 1 <= len(names) <= 3 or len(set(names)) != len(names) or not set(names) <= allowed:
        raise ValueError(f"Step5 hemoglobin_signals must be 1–3 distinct approved *_gsub names; got {names}")
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"missing_selected_signal: missing={missing}, requested={names}, available={list(ds.data_vars)}")
    return names


def configured_signal_map(ds: xr.Dataset, config: dict[str, Any]) -> dict[str, str]:
    """Return an identity mapping for the selected Step5 Hb signals.

    Returns
    -------
    dict of str to str
        Configured output labels mapped to the same processed variable names.

    Notes
    -----
    Only the approved ``*_gsub`` hemoglobin channels are accepted.
    """
    names = selected_hemoglobin_signals(ds, config)
    return {name: name for name in names}


def signal_name_for_roi_image(ds: xr.Dataset, config: dict[str, Any]) -> str:
    """Return the first configured Hb channel used for ROI preview display.

    Returns
    -------
    str
        Existing processed-Zarr variable name.
    """
    return selected_hemoglobin_signals(ds, config)[0]


def _as_time_y_x(var: xr.DataArray) -> xr.DataArray:
    """Require spatial-temporal dimensions and reorder a signal to (time, y, x).

    Parameters
    ----------
    var : xarray.DataArray
        Input signal with ``time``, ``y`` and ``x`` dimensions.

    Returns
    -------
    xarray.DataArray
        Reordered signal with dimensions (time, y, x).
    """
    required = {"time", "y", "x"}
    if not required.issubset(set(var.dims)):
        raise ValueError(f"Signal {var.name!r} must have dims including time/y/x, got {var.dims}")
    return var.transpose("time", "y", "x")


def signal_to_numpy(ds: xr.Dataset, name: str) -> np.ndarray:
    """Materialize a processed signal as float32 (time, y, x) data.

    Parameters
    ----------
    ds : xarray.Dataset
        Processed Step4 dataset.
    name : str
        Name of the signal variable to load.

    Returns
    -------
    numpy.ndarray
        Three-dimensional float32 array, dimensions (time, y, x).
        Units are inherited from the selected signal.
    """
    arr = _as_time_y_x(ds[name]).astype("float32")
    return arr.compute().values.astype(np.float32, copy=False)


# ============================================================
# Stimulus windows
# ============================================================


def _tuple2(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    """Interpret a configuration value as a pair of integer frame offsets.

    Parameters
    ----------
    value : Any
        Two-element sequence or comma-separated textual pair.
    default : tuple of int
        Returned when the input cannot be interpreted as a pair.

    Returns
    -------
    tuple of int
        Two integer values, preserving input order.
    """
    if value is None:
        return default
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")]
        parts = [p for p in parts if p]
        if len(parts) == 2:
            return (int(float(parts[0])), int(float(parts[1])))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return default


def stimulus_cfg_for_signal(config: dict[str, Any], signal_name: str) -> dict[str, Any]:
    """The same Hb preview window applies to all three selected *_gsub channels."""
    stim_cfg = dict(cfg_get(config, "step5", "stimulus", default={}) or {})
    stim_cfg["stim_window"] = _tuple2(stim_cfg.get("preview_window_frames"), (0, 100))
    stim_cfg["subtr_baseline"] = "prestart_mean"
    stim_cfg["subtr_baseline_interval"] = int(stim_cfg.get("baseline_frames", 20))
    return stim_cfg


def trial_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Use a single onset, repeat period, pre-onset baseline and trial window."""
    stim_cfg = cfg_get(config, "step5", "stimulus", default={}) or {}
    onset = int(stim_cfg.get("stim_start_frame", 6000))
    count = int(stim_cfg.get("stim_count", 4))
    interval = int(stim_cfg.get("stim_step", 1500))
    window = _tuple2(stim_cfg.get("trial_window_frames"), (-20, 1480))
    baseline = int(stim_cfg.get("baseline_frames", 20))
    if count != 4 or interval <= 0 or baseline <= 0 or window[0] >= window[1]:
        raise ValueError("Step5 requires 4 stimulation blocks, a positive period/baseline and a valid trial window")
    return {
        "stim_start_frame": onset,
        "stim_count": count,
        "stim_step": interval,
        "stim_window": window,
        "subtr_baseline": "prestart_mean",
        "subtr_baseline_interval": baseline,
        "skip_incomplete_trials": False,
        "min_valid_trials": count,
        "stim_onset_frame": onset,
    }


def pulse_schedule(config: dict[str, Any], sampling_rate: float) -> list[dict[str, float | int]]:
    """Calculate pulse onset and offset times from the configured protocol.

    Parameters
    ----------
    config : dict
        Full configuration with ``step5.stimulus.pulse_count``,
        ``pulse_frequency_hz`` and ``pulse_duration_sec``.
    sampling_rate : float
        Effective acquisition frame rate, in Hz.

    Returns
    -------
    list of dict
        Per-pulse onset/offset times in seconds, rounded onset frames and
        rounded pulse durations in frames.

    Notes
    -----
    Pulse onsets are computed independently using ``np.rint(i * fps / hz)``;
    no fixed integer frame period is accumulated. A 1-Hz frequency and
    16.67-Hz acquisition rate are examples only; actual values are determined
    by configuration and the processed dataset.
    """
    stim_cfg = cfg_get(config, "step5", "stimulus", default={}) or {}
    count = int(stim_cfg.get("pulse_count", 4))
    hz = float(stim_cfg.get("pulse_frequency_hz", 1.0))
    duration = float(stim_cfg.get("pulse_duration_sec", 0.3))
    fps = float(sampling_rate)
    if count < 1 or hz <= 0 or fps <= 0 or duration <= 0 or duration > 1.0/hz:
        raise ValueError("Invalid Step5 pulse_count, pulse_frequency_hz, pulse_duration_sec or sampling rate")
    return [
        {
            "onset_seconds": float(i / hz),
            "offset_seconds": float(i / hz + duration),
            "onset_frame": int(np.rint(i * fps / hz)),
            "duration_frames": int(np.rint(duration * fps)),
        }
        for i in range(count)
    ]


def stim_sum_flexible(
    dataset: np.ndarray,
    stim_count: int = 4,
    stim_start_frame: int = 6000,
    stim_step: int = 1500,
    stim_window: tuple[int, int] = (0, 100),
    subtr_baseline: str = "prestart_mean",
    subtr_baseline_interval: int = 20,
    substim_count: int | None = None,
    substim_interval: int | None = None,
    substim_window: tuple[int, int] = (1, 6),
    skip_incomplete_trials: bool = True,
    min_valid_trials: int = 1,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract repeated spatial stimulus clips and sum baseline-corrected frames.

    Parameters
    ----------
    dataset : numpy.ndarray
        Float array of shape (time, y, x), in the source signal's units.
    stim_count : int
        Number of repeated stimulation windows.
    stim_start_frame, stim_step : int
        First onset and interval between successive trial onsets, in frames.
    stim_window : tuple of int
        Start/end offsets from each onset, in frames (end exclusive).
    subtr_baseline : str
        Baseline mode: ``mean``, ``prestart_mean`` or no subtraction.
    subtr_baseline_interval : int
        Number of preceding frames used for ``prestart_mean``.
    substim_count, substim_interval, substim_window : optional
        Optional additional within-trial summation windows, in frames.
    skip_incomplete_trials : bool
        Whether to skip trials extending past the recording length.
    min_valid_trials : int
        Minimum accepted trial count; the implementation also requires every
        requested trial to be valid at return time.

    Returns
    -------
    dict of str to numpy.ndarray
        Baseline-subtracted clips, each (trial_time, y, x).
    numpy.ndarray
        Sum over selected frames and trials, shape (y, x), float32.

    Notes
    -----
    The returned sum is not divided by the number of frames or trials.
    ``average_stimulus_from_dict`` performs a separate mean across repeats.
    Current Step5 calls this function with all preview trials required.
    """
    arr = np.asarray(dataset, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"dataset must be 3D (time, y, x), got {arr.shape}")
    stim_dict: dict[str, np.ndarray] = {}
    sum_img = np.zeros(arr.shape[1:], dtype=np.float32)
    skipped: list[str] = []

    for i in range(int(stim_count)):
        stim_base = int(stim_start_frame) + i * int(stim_step)
        stim_start = stim_base + int(stim_window[0])
        stim_end = stim_base + int(stim_window[1])
        if stim_start < 0 or stim_end > arr.shape[0] or stim_end <= stim_start:
            msg = f"Stimulus window out of bounds for st{i + 1}: {stim_start}:{stim_end}, T={arr.shape[0]}"
            if skip_incomplete_trials:
                skipped.append(msg)
                continue
            raise IndexError(msg)

        mode = str(subtr_baseline).strip().lower()
        if mode == "mean":
            baseline: np.ndarray | float = np.nanmean(arr, axis=0)
        elif mode == "prestart_mean":
            base_start = stim_base - int(subtr_baseline_interval)
            base_end = stim_base
            if base_start < 0 or base_end <= base_start:
                msg = f"Baseline window out of bounds for st{i + 1}: {base_start}:{base_end}, T={arr.shape[0]}"
                if skip_incomplete_trials:
                    skipped.append(msg)
                    continue
                raise IndexError(msg)
            baseline = np.nanmean(arr[base_start:base_end], axis=0)
        elif mode in {"no", "none", "false"}:
            baseline = 0.0
        else:
            raise ValueError(f"Unknown baseline mode: {subtr_baseline!r}")

        stim_data = arr[stim_start:stim_end] - baseline
        stim_dict[f"st{i + 1}"] = stim_data.astype(np.float32, copy=False)

        if substim_count is None:
            sum_img += np.nansum(stim_data, axis=0)
        else:
            if substim_interval is None:
                raise ValueError("substim_interval is required when substim_count is not None")
            step = int(substim_interval)  # Inter-onset period, not total train duration.
            for j in range(int(substim_count)):
                s_start = j * step + int(substim_window[0])
                s_end = j * step + int(substim_window[1])
                if 0 <= s_start < s_end <= stim_data.shape[0]:
                    sum_img += np.nansum(stim_data[s_start:s_end], axis=0)

    if skipped:
        print(f"[Step5] Skipped {len(skipped)} incomplete 3D stimulus repeat(s): " + "; ".join(skipped))
    if len(stim_dict) != int(stim_count) or len(stim_dict) < int(min_valid_trials):
        raise ValueError(
            f"Not enough valid 3D stimulus repeats: valid={len(stim_dict)}, "
            f"required={int(min_valid_trials)}, T={arr.shape[0]}"
        )
    return stim_dict, sum_img.astype(np.float32, copy=False)


def average_stimulus_from_dict(stim_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Compute the framewise mean of equally sized stimulus clips.

    Parameters
    ----------
    stim_dict : dict of str to numpy.ndarray
        Trial arrays with common shape (trial_time, y, x).

    Returns
    -------
    numpy.ndarray
        Float32 trial-averaged response, shape (trial_time, y, x).
        The physical units match the input signal.

    Raises
    ------
    ValueError
        If the collection is empty or trial lengths differ.
    """
    if not stim_dict:
        raise ValueError("stim_dict is empty")
    lengths = {np.asarray(v).shape[0] for v in stim_dict.values()}
    if len(lengths) != 1:
        raise ValueError(f"Stimulus clips have different lengths: {sorted(lengths)}")
    return np.nanmean(np.stack(list(stim_dict.values()), axis=0), axis=0).astype(np.float32)


def normalise_3d_percentile(img: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Normalize finite array values to the clipped display range [0, 1].

    Parameters
    ----------
    img : numpy.ndarray
        Array of image intensities, commonly (time, y, x).
    low, high : float
        Global percentile limits used for display contrast.

    Returns
    -------
    numpy.ndarray
        Float32 array with display values clipped to [0, 1].

    Notes
    -----
    This is visualization normalization, not normalization of quantitative
    hemoglobin concentrations or extracted ROI signals.
    """
    arr = np.asarray(img, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    p_low, p_high = np.nanpercentile(valid, [low, high])
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - float(p_low)) / (float(p_high) - float(p_low))
    return np.clip(out, 0, 1).astype(np.float32)


def smooth_preview_with_mask(
    img: np.ndarray, mask: np.ndarray, sigma_px: float,
) -> np.ndarray:
    """Smooth preview frames spatially without bleeding from invalid pixels.

    Parameters
    ----------
    img : numpy.ndarray
        Three-dimensional preview stack (time, y, x).
    mask : numpy.ndarray
        Boolean spatial validity mask (y, x).
    sigma_px : float
        Gaussian smoothing sigma in spatial pixels; zero disables smoothing.

    Returns
    -------
    numpy.ndarray
        Float32 display stack with NaN outside the valid mask.

    Notes
    -----
    Each frame is normalized by a smoothed validity weight map. The operation
    is restricted to Napari preview frames and does not alter saved processed
    hemoglobin signals or quantitative ROI-mean traces.
    """
    arr = np.asarray(img, dtype=np.float32)
    valid_mask = np.asarray(mask, dtype=bool)
    if arr.ndim != 3 or arr.shape[1:] != valid_mask.shape:
        raise ValueError(f"Preview and mask shape mismatch: {arr.shape} vs {valid_mask.shape}")
    sigma = float(sigma_px)
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("step5.napari.display_smoothing_sigma_px must be >= 0")
    result = np.full(arr.shape, np.nan, dtype=np.float32)
    for t, frame in enumerate(arr):
        usable = valid_mask & np.isfinite(frame)
        if sigma == 0:
            result[t, usable] = frame[usable]
            continue
        weights = usable.astype(np.float32)
        numerator = cv2.GaussianBlur(np.where(usable, frame, 0).astype(np.float32),
                                     (0, 0), sigmaX=sigma, sigmaY=sigma)
        denominator = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma, sigmaY=sigma)
        result[t, usable] = (numerator[usable] / denominator[usable]).astype(np.float32)
    return result


# ============================================================
# Preview and napari ROI drawing
# ============================================================


def build_response_preview_for_row(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    save_preview: bool | None = None,
) -> dict[str, Any]:
    """Build time-resolved, trial-averaged images for interactive response ROI review.

    Parameters
    ----------
    row : pandas.Series
        Step5 recording row with a processed-Zarr path.
    config : dict
        Full configuration, including ``step5.stimulus``, ``step5.input``
        and ``step5.napari`` settings.
    paths : dict of pathlib.Path
        Step5 output directories and canonical mask locations.
    save_preview : bool or None, optional
        Override ``step5.output.save_preview_png``.

    Returns
    -------
    dict
        Loaded dataset, mask sources, selected Hb signals, baseline-corrected
        trial clips, (trial_time, y, x) averaged response stacks, display-only
        normalized stacks, and Step4 fingerprint metadata.

    Notes
    -----
    The code requires Day8/Awake/Stim and four complete stimulation trials.
    For each selected Hb channel, a pre-onset baseline is subtracted from each
    trial before averaging. Display stacks may be Gaussian-smoothed and
    percentile-scaled, but quantitative averages are preserved separately.
    The ROI is restricted to the canonical valid brain signal mask.
    """
    config = ensure_step5_defaults(config)
    if str(row.get("exp_day")) != "Day8" or str(row.get("state")) != "Awake" or str(row.get("protocol_base")) != "Stim":
        raise ValueError(f"Step5 only supports Day8 / Awake / Stim, got stem={row.get('stem')}")
    if str(row.get("stem")) in set(config.get("excluded_stems") or []):
        raise ValueError(f"Excluded Step5 stem={row.get('stem')}")
    ds = open_processed_zarr(row["processed_zarr_path_resolved"])
    names = selected_hemoglobin_signals(ds, config)
    trial_cfg = trial_kwargs_from_config(config)
    tstart = int(trial_cfg["stim_start_frame"])
    count = int(trial_cfg["stim_count"])
    step = int(trial_cfg["stim_step"])
    twindow = trial_cfg["stim_window"]
    preview_window = _tuple2(cfg_get(config, "step5", "stimulus", "preview_window_frames", default=[0,100]), (0,100))
    if (tstart + twindow[0] < 0 or tstart - int(trial_cfg["subtr_baseline_interval"]) < 0
            or tstart + (count - 1)*step + twindow[1] > int(ds.sizes["time"])):
        raise ValueError(f"Step5 requires four complete 1D trials: T={ds.sizes['time']}, "
                         f"first_onset={tstart}, count={count}, step={step}, window={twindow}")
    if (preview_window[0] >= preview_window[1] or tstart + preview_window[0] < 0
            or tstart + (count - 1)*step + preview_window[1] > int(ds.sizes["time"])):
        raise ValueError(f"Step5 requires four complete preview windows: T={ds.sizes['time']}, "
                         f"first_onset={tstart}, window={preview_window}")
    image_shape = (int(ds.sizes["y"]), int(ds.sizes["x"]))
    brain_signal, mask_sources = load_canonical_brain_signal_mask(
        row, config, paths, image_shape, processed_ds=ds
    )
    stacks: dict[str, np.ndarray] = {}
    averages: dict[str, np.ndarray] = {}
    stim_dicts: dict[str, dict[str, np.ndarray]] = {}
    stim_cfg = stimulus_cfg_for_signal(config, names[0])
    display_sigma = float(cfg_get(config, "step5", "napari", "display_smoothing_sigma_px", default=1.0))
    for name in names:
        data = signal_to_numpy(ds, name)
        if tuple(data.shape[1:]) != image_shape:
            raise ValueError(f"Processed signal geometry mismatch: {name} {data.shape}")
        clips, _ = stim_sum_flexible(
            data,
            stim_count=int(stim_cfg.get("stim_count", 4)),
            stim_start_frame=int(stim_cfg.get("stim_start_frame", 6000)),
            stim_step=int(stim_cfg.get("stim_step", 1500)),
            stim_window=_tuple2(stim_cfg.get("stim_window"), (0, 100)),
            subtr_baseline=str(stim_cfg.get("subtr_baseline", "prestart_mean")),
            subtr_baseline_interval=int(stim_cfg.get("subtr_baseline_interval", 20)),
            substim_count=None,
            skip_incomplete_trials=False,
            min_valid_trials=int(stim_cfg.get("stim_count", 4)),
        )
        avg = average_stimulus_from_dict(clips)
        averages[name] = avg
        stim_dicts[name] = clips
        # Never alter averages or extracted ROI traces; smooth display stack only.
        display_stack = smooth_preview_with_mask(avg, brain_signal, display_sigma)
        stacks[name] = normalise_3d_percentile(display_stack, 1, 99)
    first = names[0]
    preview = {
        "ds": ds,
        "signal_name": first,
        "displayed_signals": names,
        "brain_signal": brain_signal,
        "mask_sources": mask_sources,
        "stim_cfg": stim_cfg,
        "stim_dict": stim_dicts[first],
        "stim_dicts": stim_dicts,
        "avg_stack": averages[first],
        "avg_stacks": averages,
        "preview_stack": stacks[first],
        "preview_stacks": stacks,
        "stimulus_repeats_used": int(stim_cfg.get("stim_count", 4)),
        "display_smoothing_sigma_px": display_sigma,
        "pulse_schedule": pulse_schedule(config, sampling_rate_from_dataset_or_config(ds, config)),
        "step4_fingerprint": str(ds.attrs["step4_fingerprint"]),
        "processed_zarr_path": str(row["processed_zarr_path_resolved"]),
    }

    if save_preview is None:
        save_preview = bool(cfg_get(config, "step5", "output", "save_preview_png", default=True))
    if save_preview:
        save_response_preview_png(preview, row, config, paths)
    return preview


def save_response_preview_png(preview: dict[str, Any], row: pd.Series, config: dict[str, Any], paths: dict[str, Path]) -> Path:
    """Save a static PNG preview of the first selected response signal.

    Parameters
    ----------
    preview : dict
        Output of ``build_response_preview_for_row`` with ``preview_stack``.
    row : pandas.Series
        Recording metadata, including ``stem``.
    config : dict
        Full YAML-derived configuration.
    paths : dict
        Step5 output directories.

    Returns
    -------
    pathlib.Path
        Path of the saved preview PNG.
    """
    stem = str(row["stem"])
    out_path = Path(paths["response_previews_root"]) / f"{stem}_response_preview.png"
    stack = np.asarray(preview["preview_stack"])
    if stack.ndim == 3:
        img = np.nanmean(stack, axis=0)
    else:
        img = stack
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap=cfg_get(config, "step5", "napari", "colormap", default="turbo"), vmin=0, vmax=1)
    ax.set_title(f"{stem}\n{preview.get('signal_name', '')}")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _existing_roi_path(row: pd.Series, paths: dict[str, Path]) -> Path:
    """Construct the canonical NPY filename for a recording's response ROI.

    Returns
    -------
    pathlib.Path
        Path within ``response_masks_drawn_root``.
    """
    return Path(paths["response_masks_drawn_root"]) / f"{row['stem']}_response_roi.npy"


def load_existing_response_roi(
    row: pd.Series, paths: dict[str, Path], image_shape: tuple[int, int],
    *, expected_step4_fingerprint: str | None = None,
) -> np.ndarray | None:
    """Load an existing ROI, optionally validating its Step4 provenance.

    Parameters
    ----------
    row : pandas.Series
        Recording row containing ``stem``.
    paths : dict
        Step5 output directories.
    image_shape : tuple of int
        Expected ROI shape (y, x), in processed-Zarr pixels.
    expected_step4_fingerprint : str or None, optional
        When given, require a matching sidecar JSON fingerprint.

    Returns
    -------
    numpy.ndarray or None
        Saved boolean ROI mask, or ``None`` when no ROI file exists.

    Raises
    ------
    ValueError
        If provenance is missing/incompatible or the mask geometry differs.
    """
    path = _existing_roi_path(row, paths)
    if not path.exists():
        return None
    if expected_step4_fingerprint is not None:
        provenance_file = path.with_suffix(".json")
        if not provenance_file.exists():
            raise ValueError(f"Existing ROI missing provenance: {provenance_file}; redraw/review required")
        provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
        if provenance.get("step4_fingerprint") != str(expected_step4_fingerprint):
            raise ValueError(f"Existing ROI was drawn for a different Step4 processed dataset: {path}")
    mask = np.load(path).astype(bool)
    if tuple(mask.shape) != tuple(image_shape):
        raise ValueError(
            f"Saved response ROI geometry mismatch for {row['stem']}: "
            f"{mask.shape} vs processed {image_shape}; review ROI anew"
        )
    return mask.astype(bool)


def roi_layer_to_mask(roi_layer: Any, image_shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a Napari Shapes layer into a two-dimensional response mask.

    Parameters
    ----------
    roi_layer : Any
        Napari Shapes layer containing polygon(s).
    image_shape : tuple of int
        Target (y, x) image dimensions.

    Returns
    -------
    numpy.ndarray
        Boolean ROI mask, shape (y, x).

    Notes
    -----
    First tries ``to_labels``. If that fails, polygons are rasterized from
    the final two shape coordinates, accommodating both (y, x) and (t, y, x)
    shape data. No physical-coordinate transform is applied here.
    """
    if roi_layer is None or len(getattr(roi_layer, "data", [])) == 0:
        return np.zeros(image_shape, dtype=bool)

    # Preferred napari conversion.  Some napari versions/layer ndim combinations
    # are picky, so keep a polygon fallback below.
    try:
        labels = roi_layer.to_labels(list(image_shape))
        return np.asarray(labels) > 0
    except Exception:
        pass

    if sk_polygon is None:
        raise RuntimeError("Could not convert napari ROI to mask and skimage.draw.polygon is unavailable")

    mask = np.zeros(image_shape, dtype=bool)
    for shape in roi_layer.data:
        pts = np.asarray(shape, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        # If shape was drawn on a 3D stack, coordinates are usually (t, y, x).
        # If it was drawn on 2D image, they are (y, x).  The last two coordinates
        # are therefore always y/x for this use-case.
        y = pts[:, -2]
        x = pts[:, -1]
        rr, cc = sk_polygon(y, x, shape=image_shape)
        mask[rr, cc] = True
    return mask


def open_response_roi_napari_viewer(
    preview: dict[str, Any],
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    """Open a Napari viewer to inspect stimulus-averaged Hb and draw a polygon.

    Parameters
    ----------
    preview : dict
        Result of ``build_response_preview_for_row``.
    row : pandas.Series
        Recording metadata.
    config : dict
        Full YAML-derived configuration with ``step5.napari`` options.
    paths : dict
        ROI root used to show any compatible saved response mask.

    Returns
    -------
    dict
        Live Napari viewer/layer session and source metadata.

    Notes
    -----
    Does not call ``napari.run``. Call it separately and then use
    ``response_roi_mask_from_napari_session``. Closing Napari alone does not
    confirm that a response is absent; the CLI separately asks for an action.
    """
    if napari is None:
        raise ImportError("napari is required for interactive Step5 ROI drawing")

    nap_cfg = cfg_get(config, "step5", "napari", default={}) or {}
    stack = np.asarray(preview["preview_stack"], dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError(f"preview_stack must be 3D, got {stack.shape}")
    t, h, w = stack.shape
    stem = str(row["stem"])
    title = f"{stem}"
    if "protocol" in row.index and pd.notna(row.get("protocol")):
        title = f"{stem} | {row.get('protocol')}"

    viewer = napari.Viewer(title=f"Step5 response ROI: {title}")
    image_layers = {}
    for i, (signal_name, signal_stack) in enumerate(preview["preview_stacks"].items()):
        image_layers[signal_name] = viewer.add_image(
            np.asarray(signal_stack, dtype=np.float32),
            name=signal_name,
            colormap=nap_cfg.get("colormap", "turbo"),
            contrast_limits=tuple(nap_cfg.get("contrast_limits", [0, 1])),
            visible=(i == 0),
        )
    image_layer = next(iter(image_layers.values()))

    brain_layer = None
    if bool(nap_cfg.get("show_brain_mask", True)):
        brain_signal_u8 = np.asarray(preview["brain_signal"], dtype=np.uint8)
        brain_layer = viewer.add_labels(
            brain_signal_u8,
            name=nap_cfg.get("brain_mask_layer_name", "brain_signal_mask"),
            opacity=0.18,
        )

    existing_layer = None
    try:
        existing = load_existing_response_roi(
            row, paths, (h, w), expected_step4_fingerprint=preview["step4_fingerprint"]
        )
    except ValueError as exc:
        print(f"[Step5] Existing ROI cannot be shown without re-review: {exc}")
        existing = None
    if existing is not None and bool(nap_cfg.get("show_existing_roi", True)):
        existing_layer = viewer.add_labels(
            existing.astype(np.uint8),
            name=nap_cfg.get("existing_roi_layer_name", "existing_response_roi"),
            opacity=0.35,
        )

    # Old visual title overlay: one text point per time plane.
    title_positions = [[frame_i, -5, w // 2] for frame_i in range(t)]
    title_layer = viewer.add_points(title_positions, name="Title", size=0.1)
    title_layer.text = {
        "string": [title] * t,
        "anchor": "center",
        "color": "white",
        "size": int(nap_cfg.get("title_size", 28)),
    }

    roi_layer_name = nap_cfg.get("roi_layer_name", "ROI")
    roi_layer = viewer.add_shapes(name=roi_layer_name, shape_type="polygon")

    print("[Step5] Draw one polygon in 'ROI', or leave it empty to review the saved ROI.")
    print("[Step5] After closing Napari, confirm the decision in PowerShell.")
    print("[Step5] Simply closing Napari NEVER means 'no visible response'.")

    return {
        "viewer": viewer,
        "image_layer": image_layer,
        "image_layers": image_layers,
        "brain_layer": brain_layer,
        "existing_layer": existing_layer,
        "existing_roi_mask": existing,
        "title_layer": title_layer,
        "roi_layer": roi_layer,
        "roi_layer_name": roi_layer_name,
        "image_shape": (h, w),
        "preview": preview,
        "row": row,
        "config": config,
        "paths": paths,
    }


def _roi_layer_has_shapes(roi_layer: Any) -> bool:
    """Return True if the napari Shapes layer contains at least one drawn shape."""
    if roi_layer is None:
        return False
    try:
        return len(getattr(roi_layer, "data", [])) > 0
    except Exception:
        return False


def response_roi_mask_from_napari_session(session: dict[str, Any]) -> np.ndarray | None:
    """Extract exactly one newly drawn polygon ROI from a Napari session.

    Parameters
    ----------
    session : dict
        Viewer session returned by ``open_response_roi_napari_viewer``.

    Returns
    -------
    numpy.ndarray or None
        Boolean response ROI in (y, x) processed image space, intersected with
        the canonical ``brain_signal`` mask; ``None`` if no polygon was drawn.

    Raises
    ------
    ValueError
        If multiple polygons exist or the drawn ROI has no eligible pixels.

    Notes
    -----
    An empty Shapes layer is not a negative response decision.
    """
    if not isinstance(session, dict):
        raise TypeError("session must be the dict returned by open_response_roi_napari_viewer(...)")
    preview = session["preview"]
    image_shape = tuple(session["image_shape"])
    roi_layer = session.get("roi_layer")
    if roi_layer is None:
        viewer = session.get("viewer")
        layer_name = session.get("roi_layer_name", "ROI")
        if viewer is None:
            raise RuntimeError("No ROI layer or viewer stored in napari session")
        roi_layer = viewer.layers[layer_name]

    if not _roi_layer_has_shapes(roi_layer):
        session["roi_mask_source"] = "none"
        return None

    if len(roi_layer.data) != 1:
        raise ValueError(f"Step5 requires exactly one final response ROI polygon; found {len(roi_layer.data)}")
    roi_mask = roi_layer_to_mask(roi_layer, image_shape).astype(bool)
    roi_mask &= np.asarray(preview["brain_signal"], dtype=bool)
    if not np.any(roi_mask):
        raise ValueError("Drawn ROI has no valid pixels inside brainvessels_signal; correct the polygon")
    session["roi_mask_source"] = "drawn"
    return roi_mask


def choose_response_roi_action(
    session: dict[str, Any], *, input_fn: Any = None,
) -> tuple[str, np.ndarray | None]:
    """Request an explicit post-Napari response-ROI decision.

    Parameters
    ----------
    session : dict
        Live review session with new and optional existing ROI data.
    input_fn : callable or None, optional
        Input reader for operator response; defaults to ``input``.

    Returns
    -------
    str
        One of ``save_new``, ``keep_existing``, ``no_visible_response``
        or ``review_cancelled``.
    numpy.ndarray or None
        New or existing mask for the corresponding positive action.

    Notes
    -----
    Empty input, EOF and KeyboardInterrupt cancel without changing saved
    results. Keeping an existing ROI is allowed only when its Step4 fingerprint
    and image geometry matched the current processed dataset.
    """
    if input_fn is None:
        input_fn = input
    new_mask = response_roi_mask_from_napari_session(session)
    existing_mask = session.get("existing_roi_mask")
    has_existing = existing_mask is not None and np.any(existing_mask)
    stem = str(session["row"]["stem"])
    print(f"\n[Step5] Confirm the decision for {stem} in PowerShell:")
    choices: dict[str, str] = {}
    if new_mask is not None:
        print("  [n] Save the newly drawn ROI (archive the previous ROI)")
        choices["n"] = "save_new"
    if has_existing:
        print("  [k] Keep the existing ROI and all its results unchanged")
        choices["k"] = "keep_existing"
    print("  [0] Confirm no visible response (archive the previous ROI)")
    print("  [c] Cancel without changing anything (default)")
    choices.update({"0": "no_visible_response", "c": "review_cancelled", "": "review_cancelled"})
    while True:
        try:
            user_choice = str(input_fn("Your choice n / k / 0 / c [c]: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            user_choice = "c"
        if user_choice in choices:
            action = choices[user_choice]
            return action, (new_mask if action == "save_new" else (
                np.asarray(existing_mask, dtype=bool) if action == "keep_existing" else None
            ))
        print("[Step5] Unrecognized choice; previously saved results are unchanged.")

def draw_response_roi_napari(
    preview: dict[str, Any],
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> np.ndarray:
    """Open napari, block until it is closed, and return the drawn ROI mask.

    Only reads the new polygon (if any); it does not save an ROI or make an
    operator decision. CLI callers must use ``choose_response_roi_action``.
    For notebooks: open viewer, call napari.run(), then explicitly choose an
    action before saving or invalidating any ROI.
    """
    session = open_response_roi_napari_viewer(preview, row, config, paths)
    napari.run()
    return response_roi_mask_from_napari_session(session)


# ============================================================
# ROI saving / logging
# ============================================================


def save_response_roi_mask(
    roi_mask: np.ndarray | None,
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    overwrite: bool | None = None,
) -> dict[str, Path]:
    """Save a confirmed response ROI mask in the configured file formats.

    Parameters
    ----------
    roi_mask : numpy.ndarray or None
        Nonempty boolean ROI in processed (y, x) image space.
    row : pandas.Series
        Recording row containing ``stem``.
    config : dict
        Full configuration with ``step5.output`` and ``step5.overwrite_roi``.
    paths : dict
        Step5 output directories.
    overwrite : bool or None, optional
        Override the configured permission to replace saved masks.

    Returns
    -------
    dict of str to pathlib.Path
        Saved ROI paths, or an empty mapping if no nonempty mask was passed.

    Notes
    -----
    Identical existing masks are retained when overwrite is disabled.
    Conflicting existing files raise ``FileExistsError`` instead of being
    silently replaced. Sidecar provenance JSON is written separately by
    ``process_step5_row``.
    """
    config = ensure_step5_defaults(config)
    if overwrite is None:
        overwrite = bool(cfg_get(config, "step5", "overwrite_roi", default=False))
    stem = str(row["stem"])
    roi_root = Path(paths["response_masks_drawn_root"])
    roi_root.mkdir(parents=True, exist_ok=True)

    if roi_mask is None:
        print(f"[Step5] No ROI mask for stem={stem}; nothing was saved.")
        return {}
    mask_bool = np.asarray(roi_mask).astype(bool)
    if not np.any(mask_bool):
        print(f"[Step5] Empty ROI mask for stem={stem}; nothing was saved.")
        return {}

    all_paths = {
        "npy": roi_root / f"{stem}_response_roi.npy",
        "bmp": roi_root / f"{stem}_response_roi.bmp",
        "png": roi_root / f"{stem}_response_roi.png",
    }
    out_paths: dict[str, Path] = {}
    for kind, path in all_paths.items():
        enabled = bool(cfg_get(config, "step5", "output", f"save_roi_{kind}", default=True))
        if enabled:
            out_paths[kind] = path

    npy_path = all_paths["npy"]
    if npy_path.exists() and not overwrite:
        try:
            old = np.load(npy_path).astype(bool)
            same = old.shape == mask_bool.shape and np.array_equal(old, mask_bool)
        except Exception:
            same = False
        if same:
            print(f"[Step5] Existing ROI is identical; keeping saved mask for stem={stem}.")
            return {kind: path for kind, path in out_paths.items() if path.exists()}
        raise FileExistsError(f"ROI already exists and overwrite=False: {npy_path}")

    # If some image sidecars exist without an npy file, protect them too unless overwriting.
    for kind, path in out_paths.items():
        if kind == "npy":
            continue
        if path.exists() and not overwrite and not npy_path.exists():
            raise FileExistsError(f"ROI sidecar already exists and overwrite=False: {path}")

    if "npy" in out_paths:
        np.save(out_paths["npy"], mask_bool.astype(np.uint8))
    img = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    if "bmp" in out_paths:
        img.save(out_paths["bmp"])
    if "png" in out_paths:
        img.save(out_paths["png"])
    return out_paths

def update_response_roi_log(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    roi_mask: np.ndarray | None,
    roi_paths: dict[str, Path] | None = None,
    mask_sources: dict[str, str] | None = None,
    answer_mask: int | None = None,
) -> Path:
    """Append a review outcome to the per-stem ROI history CSV.

    Parameters
    ----------
    row : pandas.Series
        Recording identifiers and metadata.
    config : dict
        Full configuration mapping (kept for the Step5 interface).
    paths : dict
        Must contain ``response_roi_log_csv``.
    roi_mask : numpy.ndarray or None
        ROI mask in processed ``rotated_scaled`` image space (y, x).
    roi_paths : dict or None, optional
        Saved ROI files.
    mask_sources : dict or None, optional
        Spatial-mask provenance and review metadata.
    answer_mask : int or None, optional
        Explicit binary response decision; inferred from the ROI if absent.

    Returns
    -------
    pathlib.Path
        Updated CSV log path.

    Notes
    -----
    Previous entries are retained; each new entry receives a revision number.
    """
    log_path = Path(paths["response_roi_log_csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    if answer_mask is None:
        answer_mask = int(roi_mask is not None and np.any(roi_mask))
    area = int(np.asarray(roi_mask).astype(bool).sum()) if roi_mask is not None else 0

    record: dict[str, Any] = {
        "stem": str(row.get("stem", "")),
        "group_id": str(row.get("group_id", "")),
        "ref_stem": str(row.get("ref_stem", "")),
        "answer_mask": int(answer_mask),
        "roi_area_px": area,
        "roi_npy_path": str((roi_paths or {}).get("npy", "")),
        "roi_bmp_path": str((roi_paths or {}).get("bmp", "")),
        "roi_png_path": str((roi_paths or {}).get("png", "")),
        "updated_at": now,
        "status": "has_response" if int(answer_mask) else "no_visible_response",
        "roi_image_space": "rotated_scaled",
        "roi_image_shape": json.dumps(list(np.asarray(roi_mask).shape)) if roi_mask is not None else str((mask_sources or {}).get("image_shape", "")),
    }
    for col in ["mouse_id", "exp_day", "state", "protocol"]:
        if col in row.index:
            record[col] = row.get(col, "")
    if mask_sources:
        record.update(mask_sources)

    if log_path.exists():
        df = pd.read_csv(log_path)
    else:
        df = pd.DataFrame()
    earlier = df[df["stem"].astype(str) == str(record["stem"])] if not df.empty and "stem" in df.columns else pd.DataFrame()
    record["roi_revision"] = int(pd.to_numeric(earlier.get("roi_revision", pd.Series(dtype=float)), errors="coerce").max()) + 1 if not earlier.empty and "roi_revision" in earlier else (len(earlier) + 1)
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df.to_csv(log_path, index=False)
    return log_path


# ============================================================
# Trace extraction, filtering, trial CSVs
# ============================================================


def sampling_rate_from_dataset_or_config(ds: xr.Dataset, config: dict[str, Any]) -> float:
    # The recorded Step4 time coordinate is authoritative; raw.fps is fallback.
    """Determine the effective sampling frequency of the processed recording.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a ``time`` coordinate in seconds, when available.
    config : dict
        Full configuration used for fallback ``raw.fps``.

    Returns
    -------
    float
        Sampling rate, in Hz.

    Notes
    -----
    Uses the reciprocal of the median time-coordinate spacing when positive and
    finite. Falls back to ``raw.fps`` otherwise.
    """
    if "time" in ds.coords and ds.sizes.get("time", 0) > 2:
        t = np.asarray(ds["time"].values, dtype=float)
        dt = np.nanmedian(np.diff(t))
        if np.isfinite(dt) and dt > 0:
            return float(1.0 / dt)
    return float(cfg_get(config, "raw", "fps", default=20))


def masked_mean_xarray(var: xr.DataArray, mask: np.ndarray) -> np.ndarray:
    """Calculate the spatial mean of a signal within a boolean ROI.

    Parameters
    ----------
    var : xarray.DataArray
        Signal with time, y and x axes.
    mask : numpy.ndarray
        Boolean ROI, shape (y, x); True pixels are included.

    Returns
    -------
    numpy.ndarray
        Float32 signal trace, shape (time,); units match ``var``.

    Notes
    -----
    Averages over y and x with ``skipna=True`` and no extra temporal filtering.
    """
    v = _as_time_y_x(var).astype("float32")
    mask_da = xr.DataArray(mask.astype(bool), dims=("y", "x"), coords={"y": v.coords["y"], "x": v.coords["x"]})
    trace = v.where(mask_da).mean(dim=("y", "x"), skipna=True).compute().values
    return np.asarray(trace, dtype=np.float32)



def stim_avg_flexible_1d(
    dataset: np.ndarray,
    stim_start_frame: int = 5980,
    stim_count: int = 4,
    stim_step: int = 1500,
    stim_window: tuple[int, int] = (0, 1500),
    subtr_baseline: str = "prestart_mean",
    subtr_baseline_interval: int = 20,
    skip_incomplete_trials: bool = True,
    min_valid_trials: int = 1,
    stim_onset_frame: int | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract baseline-corrected one-dimensional stimulus trials.

    Parameters
    ----------
    dataset : numpy.ndarray
        Temporal ROI-averaged signal, shape (time,).
    stim_start_frame, stim_step : int
        First trial-window anchor and interval between trials, in frames.
    stim_count : int
        Number of stimulation repetitions.
    stim_window : tuple of int
        Trial start/end offsets in frames (end exclusive).
    subtr_baseline : str
        Baseline subtraction mode.
    subtr_baseline_interval : int
        Number of pre-onset baseline samples.
    skip_incomplete_trials : bool
        Whether out-of-bounds trials are skipped.
    min_valid_trials : int
        Requested minimum trial count; the implementation also requires all
        ``stim_count`` trials at return time.
    stim_onset_frame : int or None
        Onset used as the prestart-baseline anchor when specified.

    Returns
    -------
    dict of str to numpy.ndarray
        Baseline-corrected 1D trial traces.
    numpy.ndarray
        Float32 mean trial trace with the source signal's units.

    Notes
    -----
    Used for per-stimulus CSV output and plotting. The baseline is subtracted
    once per trial; plot assembly does not repeat baseline subtraction.
    """
    arr = np.asarray(dataset, dtype=np.float32)
    stim_dict: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for i in range(int(stim_count)):
        stim_base = int(stim_start_frame) + i * int(stim_step)
        stim_start = stim_base + int(stim_window[0])
        stim_end = stim_base + int(stim_window[1])
        if stim_start < 0 or stim_end > arr.shape[0] or stim_end <= stim_start:
            msg = f"1D stimulus window out of bounds for st{i + 1}: {stim_start}:{stim_end}, T={arr.shape[0]}"
            if skip_incomplete_trials:
                skipped.append(msg)
                continue
            raise IndexError(msg)
        mode = str(subtr_baseline).strip().lower()
        if mode == "mean":
            baseline = np.nanmean(arr)
        elif mode == "prestart_mean":
            onset = int(stim_onset_frame) + i * int(stim_step) if stim_onset_frame is not None else stim_start
            base_start = onset - int(subtr_baseline_interval)
            base_end = onset
            if base_start < 0 or base_end <= base_start:
                msg = f"1D baseline window out of bounds for st{i + 1}: {base_start}:{base_end}, T={arr.shape[0]}"
                if skip_incomplete_trials:
                    skipped.append(msg)
                    continue
                raise IndexError(msg)
            baseline = np.nanmean(arr[base_start:base_end])
        elif mode in {"no", "none", "false"}:
            baseline = 0.0
        else:
            raise ValueError(f"Unknown baseline mode: {subtr_baseline!r}")
        stim_dict[f"st{i + 1}"] = (arr[stim_start:stim_end] - baseline).astype(np.float32)
    if skipped:
        print(f"[Step5] Skipped {len(skipped)} incomplete 1D stimulus repeat(s): " + "; ".join(skipped))
    if len(stim_dict) != int(stim_count) or len(stim_dict) < int(min_valid_trials):
        raise ValueError(
            f"Not enough valid 1D stimulus repeats: valid={len(stim_dict)}, "
            f"required={int(min_valid_trials)}, T={arr.shape[0]}"
        )
    avg = np.nanmean(np.stack(list(stim_dict.values()), axis=0), axis=0).astype(np.float32)
    return stim_dict, avg


def save_stim_trials_csv(
    stim_dict: dict[str, np.ndarray],
    out_path: Path,
    row: pd.Series,
    *,
    signal_label: str,
    source_signal_name: str,
    sampling_rate: float,
) -> Path:
    """Save baseline-corrected individual stimulus trials to a CSV table.

    Parameters
    ----------
    stim_dict : dict of str to numpy.ndarray
        One 1D trace per trial, indexed by its stimulation label.
    out_path : pathlib.Path
        Output CSV filename.
    row : pandas.Series
        Recording identifiers and metadata.
    signal_label, source_signal_name : str
        Display/metadata label and processed source variable.
    sampling_rate : float
        Sampling frequency, in Hz, used for relative time-column labels.

    Returns
    -------
    pathlib.Path
        Path to the saved trial CSV.

    Notes
    -----
    Time column labels begin at each trial window's first sample, not at the
    stimulus onset itself.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(stim_dict).T.reset_index(names="stim#")
    time_cols = [f"{t:.3f}" for t in np.arange(df.shape[1] - 1, dtype=float) / float(sampling_rate)]
    df.columns = ["stim#"] + time_cols
    meta = {
        "stem": str(row.get("stem", "")),
        "group_id": str(row.get("group_id", "")),
        "ref_stem": str(row.get("ref_stem", "")),
        "signal": signal_label,
        "source_signal": source_signal_name,
    }
    for col in ["mouse_id", "exp_day", "state", "protocol"]:
        if col in row.index:
            meta[col] = row.get(col, "")
    for key, value in reversed(list(meta.items())):
        df.insert(0, key, value)
    df.to_csv(out_path, index=False)
    return out_path


def extract_and_save_response_traces(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    roi_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract quantitative Hb traces from a confirmed functional ROI.

    Parameters
    ----------
    row : pandas.Series
        Recording with a processed-Zarr path and identifying metadata.
    config : dict
        Full project configuration including ``step5.input``,
        ``step5.stimulus``, ``step5.plot`` and ``step5.output``.
    paths : dict of pathlib.Path
        Step5 response table and plot output directories.
    roi_mask : numpy.ndarray or None, optional
        Boolean ROI (y, x); if omitted, load a previously saved ROI.

    Returns
    -------
    dict
        Paths and provenance for traces, trial tables, plots and summary JSON.
        When no usable ROI exists, returns a corresponding skip status.

    Notes
    -----
    ROI pixels are intersected with the canonical valid brain mask. Each
    trace is the NaN-aware spatial mean of the selected processed Step4 Hb
    signal, with no additional low-pass filter. Time is reported in seconds;
    hemoglobin concentration units follow the source processed variables.
    Four full trials are extracted according to ``step5.stimulus`` when enabled.
    """
    config = ensure_step5_defaults(config)
    ds = open_processed_zarr(row["processed_zarr_path_resolved"])
    image_shape = (int(ds.sizes["y"]), int(ds.sizes["x"]))
    if roi_mask is None:
        roi_path = _existing_roi_path(row, paths)
        if roi_path.exists():
            roi_mask = load_existing_response_roi(row, paths, image_shape)
        else:
            summary = {
                "stem": str(row.get("stem", "")),
                "status": "no_response_roi",
                "trace_csv": "",
                "stim_trial_csvs": {},
                "plot_paths": [],
                "signals": {},
                "roi_area_px": 0,
            }
            if bool(cfg_get(config, "step5", "output", "save_summary_json", default=True)):
                summary_path = Path(paths["response_tables_root"]) / f"{row['stem']}_step5_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                summary["summary_json"] = str(summary_path)
            print(f"[Step5] No ROI mask found for stem={row['stem']}; traces/plot skipped.")
            return summary

    # Always restrict traces to the canonical signal mask requested for Step5.
    brain_signal, _ = load_canonical_brain_signal_mask(row, config, paths, image_shape, processed_ds=ds)
    if tuple(np.asarray(roi_mask).shape) != tuple(image_shape):
        raise ValueError(f"Response ROI shape mismatch: {np.asarray(roi_mask).shape} vs {image_shape}")
    roi_mask = np.asarray(roi_mask, dtype=bool) & brain_signal
    if roi_mask is None or not np.any(roi_mask):
        summary = {
            "stem": str(row.get("stem", "")),
            "status": "empty_response_roi",
            "trace_csv": "",
            "stim_trial_csvs": {},
            "plot_paths": [],
            "signals": {},
            "roi_area_px": 0,
        }
        if bool(cfg_get(config, "step5", "output", "save_summary_json", default=True)):
            summary_path = Path(paths["response_tables_root"]) / f"{row['stem']}_step5_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            summary["summary_json"] = str(summary_path)
        print(f"[Step5] ROI mask is empty after brain_signal intersection for stem={row['stem']}; traces/plot skipped.")
        return summary

    sampling_rate = sampling_rate_from_dataset_or_config(ds, config)
    signal_map = configured_signal_map(ds, config)
    if not signal_map:
        raise KeyError(f"No configured Step5 signals found in processed zarr. Available: {list(ds.data_vars)}")

    traces: dict[str, np.ndarray] = {}
    source_names: dict[str, str] = {}
    for label, source_name in signal_map.items():
        trace = masked_mean_xarray(ds[source_name], roi_mask.astype(bool))
        # Quantitative Step5 traces must equal processed-Zarr ROI means (no extra low-pass).
        traces[label] = trace.astype(np.float32, copy=False)
        source_names[label] = source_name

    n_time = min(len(v) for v in traces.values())
    time_s = np.arange(n_time, dtype=np.float32) / float(sampling_rate)
    trace_df = pd.DataFrame({"time_s": time_s})
    for label, trace in traces.items():
        trace_df[label] = trace[:n_time]
    for col in ["stem", "group_id", "ref_stem", "mouse_id", "exp_day", "state", "protocol"]:
        if col in row.index:
            trace_df[col] = row.get(col, "")

    trace_path = Path(paths["response_tables_root"]) / f"{row['stem']}_roi_traces.csv"
    if bool(cfg_get(config, "step5", "output", "save_traces_csv", default=True)):
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_df.to_csv(trace_path, index=False)

    stim_paths: dict[str, str] = {}
    stim_warnings: dict[str, str] = {}
    trial_args = trial_kwargs_from_config(config)
    if bool(cfg_get(config, "step5", "output", "save_stim_trials_csv", default=True)):
        for label, trace in traces.items():
            try:
                stim_dict, _ = stim_avg_flexible_1d(trace, **trial_args)
            except (IndexError, ValueError) as e:
                stim_warnings[label] = str(e)
                print(f"[Step5] Skipping stimulus-trial CSV for {row['stem']} / {label}: {e}")
                continue
            p = Path(paths["response_tables_root"]) / f"{row['stem']}_{label}_stim_trials.csv"
            save_stim_trials_csv(
                stim_dict,
                p,
                row,
                signal_label=label,
                source_signal_name=source_names[label],
                sampling_rate=sampling_rate,
            )
            stim_paths[label] = str(p)

    plot_paths: list[str] = []
    plot_warning = ""
    if bool(cfg_get(config, "step5", "plot", "enabled", default=True)):
        try:
            plot_paths = [str(p) for p in plot_response_traces(traces, row, config, paths, sampling_rate=sampling_rate)]
        except (IndexError, ValueError) as e:
            plot_warning = str(e)
            print(f"[Step5] Skipping response plot for {row['stem']}: {e}")

    summary = {
        "stem": str(row.get("stem", "")),
        "trace_csv": str(trace_path),
        "stim_trial_csvs": stim_paths,
        "stim_trial_warnings": stim_warnings,
        "plot_paths": plot_paths,
        "plot_warning": plot_warning,
        "signals": source_names,
        "displayed_signals": list(source_names.values()),
        "processed_zarr_path": str(row["processed_zarr_path_resolved"]),
        "step4_fingerprint": str(ds.attrs.get("step4_fingerprint", "")),
        "roi_image_space": "rotated_scaled",
        "roi_image_shape": list(image_shape),
        "baseline_frames": [int(trial_args["stim_start_frame"]) - int(trial_args["subtr_baseline_interval"]), int(trial_args["stim_start_frame"])],
        "stimulus_onset_frames": [int(trial_args["stim_start_frame"]) + i*int(trial_args["stim_step"]) for i in range(4)],
        "pulse_schedule": pulse_schedule(config, sampling_rate),
        "sampling_rate": sampling_rate,
        "roi_area_px": int(np.asarray(roi_mask).astype(bool).sum()),
    }
    if bool(cfg_get(config, "step5", "output", "save_summary_json", default=True)):
        summary_path = Path(paths["response_tables_root"]) / f"{row['stem']}_step5_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
    return summary


# ============================================================
# Plotting
# ============================================================


def _plot_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return the ``step5.plot`` configuration mapping or an empty dict.
    """
    return cfg_get(config, "step5", "plot", default={}) or {}


def _plot_color(config: dict[str, Any], label: str, default: str) -> str:
    """Return signal color from Step5 config, with Step4 visualization fallback."""
    plot_colors = (_plot_cfg(config).get("colors") or {})
    if label in plot_colors:
        return str(plot_colors[label])
    if label == "dcHbR" and "dcHHb" in plot_colors:
        return str(plot_colors["dcHHb"])
    # Optional compatibility with Step4 visualization.traces.colors.
    step4_colors = cfg_get(config, "step4", "visualization", "traces", "colors", default={}) or {}
    aliases = {
        "Ca": ["fluo_hemo_regressed_percent", "Ca"],
        "Cawocorr": ["fluo_dff_percent", "Cawocorr"],
        "dcHbR": ["dcHHb", "dcHbR"],
    }
    for key in [label] + aliases.get(label, []):
        if key in step4_colors:
            return str(step4_colors[key])
    return default


def _plot_label(config: dict[str, Any], label: str) -> str:
    """Select the configured or built-in display label for a signal.

    Returns
    -------
    str
        Human-readable label used in plots.
    """
    labels = (_plot_cfg(config).get("labels") or {})
    if label in labels:
        return str(labels[label])
    defaults = {
        "Cawocorr": "Ca",
        "Ca": "Ca (corrected)",
        "dcHbT": "HbT",
        "dcHbO": "HbO",
        "dcHbR": "HbR",
        "dcHHb": "HHb",
    }
    if label.endswith("_gsub"):
        return str(labels.get(label, defaults.get(label[:-5], label)))
    return defaults.get(label, label)


def _label_position_index(n: int, position: str) -> int:
    """Select an index at the requested left/center/right position.

    Returns
    -------
    int
        Index clipped to the sample range.
    """
    position = str(position or "center").lower()
    if position == "left":
        frac = 0.20
    elif position == "right":
        frac = 0.80
    else:
        frac = 0.50
    return int(np.clip(round((n - 1) * frac), 0, max(n - 1, 0)))


def _trials_for_plot(trace: np.ndarray, config: dict[str, Any], sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute trial-averaged response curves and variability envelopes.

    Parameters
    ----------
    trace : numpy.ndarray
        One-dimensional recorded signal, in its native concentration units.
    config : dict
        Full configuration, including ``step5.plot.error_type`` and timing.
    sampling_rate : float
        Effective sample rate in Hz; used by callers to construct the x-axis.

    Returns
    -------
    numpy.ndarray
        Mean baseline-corrected response trace.
    numpy.ndarray
        SD, SEM or zero error envelope according to the configuration.

    Notes
    -----
    The SEM is computed as ``nanstd(trials, axis=0) / sqrt(number of trials)``;
    there is no second baseline subtraction.
    """
    plot_cfg = _plot_cfg(config)
    stim_dict, _ = stim_avg_flexible_1d(trace, **trial_kwargs_from_config(config))
    trials = np.stack(list(stim_dict.values()), axis=0).astype(np.float32)

    # Baseline has already been subtracted once relative to each true onset.
    # Never re-baseline for plotting (legacy normalize_to_pre_stim is ignored).

    mean = np.nanmean(trials, axis=0)
    err_type = str(plot_cfg.get("error_type", "sem")).lower()
    if err_type == "sd":
        err = np.nanstd(trials, axis=0)
    elif err_type == "sem":
        err = np.nanstd(trials, axis=0) / math.sqrt(max(trials.shape[0], 1))
    else:
        err = np.zeros_like(mean)
    return mean.astype(np.float32), err.astype(np.float32)


def _stimulus_box_times(
    config: dict[str, Any], x_max: float, sampling_rate: float,
) -> list[tuple[float, float]]:
    """Use the physical pulse protocol in seconds, independent of frame rounding."""
    plot_cfg = _plot_cfg(config)
    box_cfg = plot_cfg.get("stimulus_boxes", {}) or {}
    if not bool(box_cfg.get("enabled", True)):
        return []
    onset_rel = -int(trial_kwargs_from_config(config)["stim_window"][0]) / float(sampling_rate)
    boxes = []
    for pulse in pulse_schedule(config, sampling_rate):
        start = onset_rel + float(pulse["onset_seconds"])
        stop = min(onset_rel + float(pulse["offset_seconds"]), float(x_max))
        if start < x_max and stop > start:
            boxes.append((start, stop))
    return boxes


def _draw_stimulus_boxes(ax: Any, x: np.ndarray, config: dict[str, Any], sampling_rate: float) -> None:
    """Draw shaded stimulus pulse intervals on a Matplotlib axis.

    Notes
    -----
    Pulse timings are derived from ``step5.stimulus`` and the selected trial
    window. Plot appearance is configured under ``step5.plot.stimulus_boxes``.
    """
    plot_cfg = _plot_cfg(config)
    box_cfg = plot_cfg.get("stimulus_boxes", {}) or {}
    color = _plot_color(config, "stimulus", str(box_cfg.get("color", "gray")))
    alpha = float(box_cfg.get("alpha", 0.2))
    boxes = _stimulus_box_times(config, float(x[-1]), sampling_rate)
    for start, stop in boxes:
        ax.axvspan(start, stop, facecolor=color, alpha=alpha, zorder=0)

    label = box_cfg.get("label")
    if label and boxes:
        y0, y1 = ax.get_ylim()
        y = y0 + float(box_cfg.get("label_y_fraction", 0.95)) * (y1 - y0)
        mid = (boxes[0][0] + boxes[-1][1]) / 2.0
        ax.text(mid, y, str(label), ha="center", va="top", color=color, fontsize=9)


def _annotate_signal_labels(
    ax_hb: Any,
    ax_ca: Any,
    plotted: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Annotate plotted response curves using their configured text positions.

    Notes
    -----
    Uses ``step5.plot.signal_annotations`` for offsets, colors, arrows and
    label boxes. This changes visualization only.
    """
    plot_cfg = _plot_cfg(config)
    ann_cfg = plot_cfg.get("signal_annotations", {}) or {}
    if not bool(ann_cfg.get("enabled", True)):
        return
    label_position = str(plot_cfg.get("label_position", "center"))
    x_offsets = ann_cfg.get("x_offsets_sec", {}) or {}
    y_fractions = ann_cfg.get("y_fractions", {}) or {}
    fontsize = int(ann_cfg.get("fontsize", 13))
    use_arrows = bool(ann_cfg.get("use_arrows", True))
    use_bboxes = bool(ann_cfg.get("use_bboxes", True))
    bbox_fc = str(ann_cfg.get("bbox_facecolor", "white"))

    for item in plotted:
        label = str(item["label"])
        ax = item["axis"]
        x = item["x"]
        y = item["mean"]
        color = item["color"]
        text = _plot_label(config, label)
        idx = _label_position_index(len(x), label_position)
        y0, y1 = ax.get_ylim()
        y_frac = float(y_fractions.get(label, 0.5))
        xytext = (
            float(x[idx]) + float(x_offsets.get(label, 0.0)),
            y0 + y_frac * (y1 - y0),
        )
        arrowprops = dict(arrowstyle="->", color=color, lw=1.5) if use_arrows else None
        bbox = dict(boxstyle="round,pad=0.3", fc=bbox_fc, ec=color, lw=1) if use_bboxes else None
        ax.annotate(
            text,
            xy=(float(x[idx]), float(y[idx])),
            xytext=xytext,
            color=color,
            fontsize=fontsize,
            arrowprops=arrowprops,
            bbox=bbox,
            ha="left",
            va="center",
        )


def _align_yaxis_zero(ax1: Any, ax2: Any) -> None:
    """Small local replacement for the old visualisation.align_yaxis helper."""
    try:
        y1 = ax1.get_ylim()
        y2 = ax2.get_ylim()
        lo = min(y1[0], y2[0])
        hi = max(y1[1], y2[1])
        if lo < 0 < hi:
            ax1.set_ylim(lo, hi)
            ax2.set_ylim(lo, hi)
    except Exception:
        return


def plot_response_traces(
    traces: dict[str, np.ndarray],
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    sampling_rate: float,
    show: bool = False,
) -> list[Path]:
    """Plot trial-averaged ROI signals with stimulus timing annotations.

    Parameters
    ----------
    traces : dict of str to numpy.ndarray
        ROI-mean signals, each shape (time,) in their source units.
    row : pandas.Series
        Recording metadata for plot title and output basename.
    config : dict
        Full configuration containing ``step5.plot`` and ``step5.stimulus``.
    paths : dict
        Step5 output directories.
    sampling_rate : float
        Sampling rate in Hz used for seconds on the time axis.
    show : bool, optional
        Display the figure instead of closing it after saving.

    Returns
    -------
    list of pathlib.Path
        Paths to saved plots in configured formats.

    Notes
    -----
    Trial means and SD/SEM envelopes are calculated by ``_trials_for_plot``.
    Pulse boxes use physical durations and the processed sampling rate.
    The current Step5 selection accepts Hb ``*_gsub`` signals; optional Ca
    plotting branches are legacy-compatible but are not selected by Step5.
    """
    plot_cfg = _plot_cfg(config)
    figsize = tuple(plot_cfg.get("figsize", [10, 5]))
    fig, ax_ca = plt.subplots(figsize=figsize)
    # This experiment displays Hb only; avoid creating an empty Ca y-axis.
    hb_only = not any(label in traces for label in ("Ca", "Cawocorr"))
    ax_hb = ax_ca if hb_only else ax_ca.twinx()

    # Preserve the legacy plot order; the current Step5 selection contains Hb only.
    ca_labels = [label for label in ["Cawocorr", "Ca"] if label in traces]
    hb_labels = [name for name in ("dcHbT_gsub", "dcHbO_gsub", "dcHHb_gsub") if name in traces]

    trial_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_len = None
    skipped_labels: dict[str, str] = {}
    for label in ca_labels + hb_labels:
        try:
            mean, err = _trials_for_plot(traces[label], config, sampling_rate)
        except (IndexError, ValueError) as e:
            skipped_labels[label] = str(e)
            print(f"[Step5] Skipping plot line {row.get('stem', '')} / {label}: {e}")
            continue
        trial_stats[label] = (mean, err)
        max_len = len(mean) if max_len is None else min(max_len, len(mean))
    if max_len is None:
        plt.close(fig)
        details = "; ".join(f"{k}: {v}" for k, v in skipped_labels.items())
        raise ValueError("No valid stimulus trials available for plotting" + (f": {details}" if details else ""))
    x = np.arange(max_len, dtype=float) / float(sampling_rate)

    line_width = float(plot_cfg.get("line_width", 1.4))
    error_alpha = float(plot_cfg.get("error_alpha", 0.10))
    plotted: list[dict[str, Any]] = []

    defaults = {
        "Cawocorr": "black",
        "Ca": "green",
        "dcHbT": "magenta",
        "dcHbO": "red",
        "dcHbR": "blue",
        "dcHHb": "blue",
        "dcHbT_gsub": "magenta",
        "dcHbO_gsub": "red",
        "dcHHb_gsub": "blue",
    }

    for label in ca_labels:
        if label not in trial_stats:
            continue
        mean, err = trial_stats[label]
        mean, err = mean[:max_len], err[:max_len]
        color = _plot_color(config, label, defaults.get(label, "green"))
        ax_ca.plot(x, mean, linewidth=line_width, color=color, label=_plot_label(config, label))
        if np.any(err):
            ax_ca.fill_between(x, mean - err, mean + err, color=color, alpha=error_alpha)
        plotted.append({"label": label, "axis": ax_ca, "x": x, "mean": mean, "color": color})

    for label in hb_labels:
        if label not in trial_stats:
            continue
        mean, err = trial_stats[label]
        mean, err = mean[:max_len], err[:max_len]
        color = _plot_color(config, label, defaults.get(label, "red"))
        ax_hb.plot(x, mean, linewidth=line_width, color=color, label=_plot_label(config, label))
        if np.any(err):
            ax_hb.fill_between(x, mean - err, mean + err, color=color, alpha=error_alpha)
        plotted.append({"label": label, "axis": ax_hb, "x": x, "mean": mean, "color": color})

    stem = str(row.get("stem", ""))
    title = stem
    if "protocol" in row.index and pd.notna(row.get("protocol")):
        title = f"{stem} | {row.get('protocol')}"
    ax_ca.set_title(title)
    ax_ca.set_xlabel(plot_cfg.get("x_label", "Time from trial window start (s)"))
    if not hb_only:
        ax_ca.set_ylabel(plot_cfg.get("calcium_y_label", "ΔF/F₀ fluorescence (%)"), color=_plot_color(config, "Ca", "green"))
        ax_ca.tick_params(axis="y", colors=_plot_color(config, "Ca", "green"))
    ax_hb.set_ylabel(plot_cfg.get("hb_y_label", "Δ[Hb] (µM)"), color=_plot_color(config, "dcHbO", "red"))
    ax_hb.tick_params(axis="y", colors=_plot_color(config, "dcHbO", "red"))
    ax_hb.spines["left" if hb_only else "right"].set_color(_plot_color(config, "dcHbO", "red"))

    x_lim = plot_cfg.get("x_lim")
    if isinstance(x_lim, (list, tuple)) and len(x_lim) == 2:
        ax_ca.set_xlim(float(x_lim[0]), float(x_lim[1]))
    calcium_y_lim = plot_cfg.get("calcium_y_lim")
    if not hb_only and isinstance(calcium_y_lim, (list, tuple)) and len(calcium_y_lim) == 2:
        ax_ca.set_ylim(float(calcium_y_lim[0]), float(calcium_y_lim[1]))
    hb_y_lim = plot_cfg.get("hb_y_lim")
    if isinstance(hb_y_lim, (list, tuple)) and len(hb_y_lim) == 2:
        ax_hb.set_ylim(float(hb_y_lim[0]), float(hb_y_lim[1]))

    zero_color = _plot_color(config, "zero_line", "black")
    ax_ca.axhline(y=0.0, color=zero_color, linestyle="--", linewidth=0.8)
    if not hb_only:
        ax_hb.axhline(y=0.0, color=zero_color, linestyle="--", linewidth=0.8, alpha=0.35)

    # Draw stimulus boxes after limits are initialized but before annotation.
    _draw_stimulus_boxes(ax_ca, x, config, sampling_rate)
    _annotate_signal_labels(ax_hb, ax_ca, plotted, config)

    if bool(plot_cfg.get("show_legend", False)):
        lines1, labels1 = ax_ca.get_legend_handles_labels()
        lines2, labels2 = ax_hb.get_legend_handles_labels()
        if lines1 or lines2:
            ax_ca.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=9)

    if bool(plot_cfg.get("align_zero", False)):
        _align_yaxis_zero(ax_ca, ax_hb)

    fig.tight_layout()
    out_paths: list[Path] = []
    root = Path(paths["response_plots_root"])
    root.mkdir(parents=True, exist_ok=True)
    for fmt in plot_cfg.get("save_formats", ["png", "tiff"]):
        out_path = root / f"{stem}_response_plot.{fmt}"
        fig.savefig(out_path, dpi=int(plot_cfg.get("dpi", 200)), bbox_inches="tight")
        out_paths.append(out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_paths


def display_saved_response_plot(summary: dict[str, Any]) -> None:
    """Notebook helper: display the first saved PNG response plot if IPython is available."""
    try:
        from IPython.display import Image, display  # type: ignore
    except Exception:
        print("IPython display is not available")
        return
    plot_paths = [Path(p) for p in summary.get("plot_paths", [])]
    pngs = [p for p in plot_paths if p.suffix.lower() == ".png" and p.exists()]
    if not pngs:
        print("No saved PNG plot found in summary['plot_paths']")
        return
    display(Image(filename=str(pngs[0])))




def _write_step5_skip_summary(
    row: pd.Series,
    paths: dict[str, Path],
    *,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    """Write a small summary JSON for rows skipped by recoverable Step5 errors."""
    summary = {
        "stem": str(row.get("stem", "")),
        "status": str(status),
        "error": str(error),
        "trace_csv": "",
        "stim_trial_csvs": {},
        "stim_trial_warnings": {},
        "plot_paths": [],
        "plot_warning": "",
        "signals": {},
        "roi_area_px": 0,
    }
    try:
        summary_path = Path(paths["response_tables_root"]) / f"{row['stem']}_step5_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
    except Exception as e:
        summary["summary_write_error"] = repr(e)
    return summary

def invalidate_response_outputs(row: pd.Series, paths: dict[str, Path]) -> list[str]:
    """Archive superseded positive response ROI and derived analysis files.

    Parameters
    ----------
    row : pandas.Series
        Recording identifier ``stem``.
    paths : dict
        Step5 root and relevant output directories.

    Returns
    -------
    list of str
        Paths of files that were moved to the timestamped ``superseded`` area.

    Notes
    -----
    Moves ROI mask/sidecar, trace CSV, stimulus-trial CSVs and response plots.
    The ROI log is not removed, and unrelated files are left unchanged.
    """
    stem = str(row["stem"])
    archive = Path(paths["step5_root"]) / "superseded" / stem / _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    items = []
    for root, globs in (
        (Path(paths["response_masks_drawn_root"]), [f"{stem}_response_roi.*"]),
        (Path(paths["response_tables_root"]), [f"{stem}_roi_traces.csv", f"{stem}_*_stim_trials.csv"]),
        (Path(paths["response_plots_root"]), [f"{stem}_response_plot.*"]),
    ):
        for pattern in globs:
            for candidate in root.glob(pattern):
                if candidate.is_file():
                    items.append(candidate)
    if items:
        archive.mkdir(parents=True, exist_ok=True)
        for item in items:
            shutil.move(str(item), str(archive / item.name))
    return [str(p) for p in items]


# ============================================================
# End-to-end row helpers and CLI
# ============================================================


def process_step5_row(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    open_napari: bool | None = None,
    overwrite_roi: bool | None = None,
) -> dict[str, Any]:
    """Review one stimulation recording and save the confirmed Step5 outcome.

    Parameters
    ----------
    row : pandas.Series
        Selected Step5 recording, including ``stem`` and processed-Zarr path.
    config : dict
        Full project configuration, including ``step5`` defaults and decisions.
    paths : dict of pathlib.Path
        Output directories returned by ``build_step5_paths``.
    open_napari : bool or None, optional
        Open an interactive Napari review unless explicitly disabled.
    overwrite_roi : bool or None, optional
        Whether a newly drawn ROI may replace an existing one; falls back to
        ``step5.overwrite_roi`` when omitted.

    Returns
    -------
    dict
        Recording status, explicit response decision, ROI paths, log path and
        optional trace-extraction summary.

    Notes
    -----
    Closing Napari is not equivalent to a negative response decision. The user
    must choose to save a new ROI, keep an existing ROI, declare no visible
    response or cancel. Cancel/keep preserve existing outputs. Confirmed negative
    reviews archive superseded positive ROI outputs. Noninteractive reuse checks
    the Step4 fingerprint recorded with the saved ROI.
    """
    config = ensure_step5_defaults(config)
    if open_napari is None:
        open_napari = True  # CLI --no-napari controls noninteractive re-extraction.
    if overwrite_roi is None:
        overwrite_roi = bool(cfg_get(config, "step5", "overwrite_roi", default=False))
    continue_on_error = bool(cfg_get(config, "step5", "batch", "continue_on_error", default=True))

    try:
        # On cancel / keep-existing, even a QC preview PNG must not be overwritten.
        preview = build_response_preview_for_row(
            row, config, paths, save_preview=False if open_napari else None
        )
    except KeyError as e:
        if not continue_on_error:
            raise
        status = "missing_selected_signal" if "missing_selected_signal" in str(e) else "preview_data_error"
        print(f"[Step5] {status}: stem={row.get('stem', '')}: {e}")
        return {
            "stem": str(row.get("stem", "")), "status": status, "answer_mask": None,
            "roi_source": "none", "roi_paths": {}, "roi_log": "",
            "trace_summary": _write_step5_skip_summary(row, paths, status=status, error=str(e)),
        }
    except (IndexError, ValueError, FileNotFoundError) as e:
        if not continue_on_error:
            raise
        print(f"[Step5] Preview error: stem={row.get('stem', '')}: {e}")
        return {
            "stem": str(row.get("stem", "")), "status": "preview_error", "answer_mask": None,
            "roi_source": "none", "roi_paths": {}, "roi_log": "",
            "trace_summary": _write_step5_skip_summary(row, paths, status="preview_error", error=str(e)),
        }
    image_shape = tuple(np.asarray(preview["preview_stack"]).shape[1:])

    roi_source = "none"
    if open_napari:
        session = open_response_roi_napari_viewer(preview, row, config, paths)
        napari.run()
        action, roi_mask = choose_response_roi_action(session)
        if action == "review_cancelled":
            print(f"[Step5] Review cancelled: {row['stem']}; existing ROI/results unchanged")
            return {
                "stem": str(row["stem"]), "status": "review_cancelled", "answer_mask": None,
                "roi_source": "none", "roi_paths": {}, "roi_log": "", "trace_summary": {},
            }
        if action == "keep_existing":
            print(f"[Step5] Previous ROI confirmed: {row['stem']}; no files changed")
            return {
                "stem": str(row["stem"]), "status": "confirmed_existing", "answer_mask": 1,
                "roi_source": "confirmed_existing",
                "roi_paths": {"npy": str(_existing_roi_path(row, paths))},
                "roi_log": "", "trace_summary": {},
            }
        # Only an explicit save_new/no_visible_response decision may write files.
        if bool(cfg_get(config, "step5", "output", "save_preview_png", default=True)):
            save_response_preview_png(preview, row, config, paths)
        if action == "save_new":
            if roi_mask is None or not np.any(roi_mask):
                raise RuntimeError("Cannot save a missing/empty newly drawn ROI")
            roi_source = "drawn"
            if _existing_roi_path(row, paths).exists() and bool(overwrite_roi):
                invalidate_response_outputs(row, paths)
            roi_paths = save_response_roi_mask(roi_mask, row, config, paths, overwrite=overwrite_roi)
            provenance = {
                "stem": str(row["stem"]),
                "source_processed_zarr": preview["processed_zarr_path"],
                "step4_fingerprint": preview["step4_fingerprint"],
                "displayed_signals": preview["displayed_signals"],
                "mask_sources": preview["mask_sources"],
                "roi_image_space": "rotated_scaled",
                "roi_shape": list(image_shape),
                "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            }
            _existing_roi_path(row, paths).with_suffix(".json").write_text(
                json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif action == "no_visible_response":
            roi_source = "explicit_no_response"
            roi_mask = None
            roi_paths = {}
            invalidate_response_outputs(row, paths)
        else:
            raise RuntimeError(f"Unexpected Step5 operator decision: {action}")
    else:
        # Explicit no-Napari mode may reuse only a compatible, non-superseded ROI;
        # absence of an ROI is not an operator's decision of no response.
        roi_mask = load_existing_response_roi(
            row, paths, image_shape, expected_step4_fingerprint=preview["step4_fingerprint"]
        )
        if roi_mask is None:
            print(f"[Step5] No confirmed ROI for stem={row['stem']} (no-Napari mode)")
            return {
                "stem": str(row["stem"]), "status": "no_existing_roi", "answer_mask": None,
                "roi_source": "none", "roi_paths": {}, "roi_log": "",
                "trace_summary": _write_step5_skip_summary(row, paths, status="no_existing_roi"),
            }
        roi_paths = {"npy": _existing_roi_path(row, paths)}
        roi_source = "confirmed_existing"

    answer_mask = int(roi_mask is not None and np.any(roi_mask))
    mask_sources = dict(preview.get("mask_sources") or {})
    mask_sources.update({
        "roi_source": roi_source,
        "roi_status": "has_response" if answer_mask else "no_visible_response",
        "source_processed_zarr": preview["processed_zarr_path"],
        "source_step4_fingerprint": preview["step4_fingerprint"],
        "displayed_signals": json.dumps(preview["displayed_signals"]),
        "trace_signals": json.dumps(preview["displayed_signals"]),
        "display_mode": "hb_only",
        "averaging_mode": "four_full_stimulus_trials",
        "stimulus_onset_frames": json.dumps([
            int(cfg_get(config, "step5", "stimulus", "stim_start_frame", default=6000)) + i *
            int(cfg_get(config, "step5", "stimulus", "stim_step", default=1500)) for i in range(4)
        ]),
        "baseline_interval_frames": int(cfg_get(config, "step5", "stimulus", "baseline_frames", default=20)),
        "sampling_rate": sampling_rate_from_dataset_or_config(preview["ds"], config),
        "pulse_schedule": json.dumps(preview["pulse_schedule"]),
        "display_smoothing_sigma_px": float(preview["display_smoothing_sigma_px"]),
    })
    log_path = update_response_roi_log(
        row, config, paths, roi_mask=roi_mask, roi_paths=roi_paths,
        mask_sources=mask_sources, answer_mask=answer_mask,
    )

    if not answer_mask:
        trace_summary = _write_step5_skip_summary(row, paths, status="no_visible_response")
    else:
        try:
            trace_summary = extract_and_save_response_traces(row, config, paths, roi_mask=roi_mask)
        except (KeyError, IndexError, ValueError) as e:
            if not continue_on_error:
                raise
            print(f"[Step5] Trace error for stem={row.get('stem', '')}: {e}")
            trace_summary = _write_step5_skip_summary(row, paths, status="trace_extraction_error", error=str(e))
    return {
        "stem": str(row.get("stem", "")),
        "status": "has_response" if answer_mask else "no_visible_response",
        "answer_mask": answer_mask,
        "roi_source": roi_source,
        "roi_paths": {k: str(v) for k, v in roi_paths.items()},
        "roi_log": str(log_path),
        "trace_summary": trace_summary,
    }

def _print_index(df: pd.DataFrame, max_rows: int = 40) -> None:
    """Print a short index view of available Step5 recordings.

    Parameters
    ----------
    df : pandas.DataFrame
        Step5 index table.
    max_rows : int, optional
        Maximum number of rows displayed.
    """
    cols = [c for c in ["stem", "group_id", "mouse_id", "exp_day", "state", "protocol", "processed_zarr_exists", "response_roi_exists"] if c in df.columns]
    print(df[cols].head(max_rows).to_string(index=True))
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Step5 command-line interface.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Explicit CLI arguments for programmatic use, or ``None`` to read
        arguments from the process command line.

    Notes
    -----
    Requires ``--config``. Supports interactive ROI review, filtering,
    ``--no-napari`` re-extraction from existing compatible ROIs, and ``--list``.
    The selection is additionally restricted to Day8/Awake/Stim by
    ``build_step5_index`` regardless of CLI metadata filters.
    """
    parser = argparse.ArgumentParser(description="Step5 response ROI drawing and trace extraction")
    parser.add_argument("--config", required=True, help="Path to current stem-key config YAML")
    parser.add_argument("--stems", nargs="*", default=None, help="Optional stem(s) to process")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group_id(s) to process")
    parser.add_argument("--idx", type=int, default=None, help="Process one row by index after filtering")
    parser.add_argument("--include-bad", action="store_true", help="Include rows marked is_bad")
    parser.add_argument("--require-processed", action="store_true", help="Keep only rows with processed zarr present")
    parser.add_argument("--batch", action="store_true", help="Process all rows after filters; also applies step5.batch.metadata_filters")
    parser.add_argument("--filter", action="append", default=None, help="Metadata exact filter KEY=VALUE or KEY=a,b; can be repeated")
    parser.add_argument("--filters-json", default=None, help='Metadata filters as JSON, e.g. {"protocol": {"regex": "RHL|LHL"}}')
    parser.add_argument("--only-stimulation", action="store_true", help="Keep only stimulation recordings using step5.batch stimulation filter")
    parser.add_argument("--all-recordings", action="store_true", help="Disable config step5.batch.only_stimulation")
    parser.add_argument("--stim-column", default=None, help="Metadata column used by --only-stimulation, default from config")
    parser.add_argument("--stim-regex", default=None, help="Regex used by --only-stimulation, default from config")
    parser.add_argument("--stim-exclude", nargs="*", default=None, help="Exact metadata values excluded by --only-stimulation")
    parser.add_argument("--skip-existing-roi", action="store_true", help="Skip rows where response ROI already exists")
    parser.add_argument("--no-napari", action="store_true", help="Do not open napari; use existing ROI and only extract traces")
    parser.add_argument("--overwrite-roi", action="store_true", default=None, help="Allow replacing an existing ROI mask. If omitted, uses step5.overwrite_roi from config")
    parser.add_argument("--list", action="store_true", help="Only print the Step5 index")
    args = parser.parse_args(argv)

    cli_filters = _parse_cli_metadata_filters(args.filter, args.filters_json)
    only_stimulation_arg: bool | None
    if args.only_stimulation:
        only_stimulation_arg = True
    elif args.all_recordings:
        only_stimulation_arg = False
    else:
        only_stimulation_arg = None

    df, config, paths = build_step5_index(
        args.config,
        include_bad=bool(args.include_bad),
        filters=cli_filters,
        stems=args.stems,
        groups=args.groups,
        require_processed=bool(args.require_processed),
        batch_mode=bool(args.batch),
        only_stimulation=only_stimulation_arg,
        stimulation_column=args.stim_column,
        stimulation_include_regex=args.stim_regex,
        stimulation_exclude_values=args.stim_exclude,
        skip_existing_roi=bool(args.skip_existing_roi),
    )
    if df.empty:
        raise RuntimeError("Step5 index is empty after filters")
    if args.list:
        _print_index(df)
        return

    if not args.no_napari and napari is None:
        parser.error("Napari is unavailable in this Python environment; activate the Napari environment before Step5 ROI drawing")

    if args.idx is not None:
        rows = [df.iloc[int(args.idx)]]
    elif args.batch or args.stems or args.groups or cli_filters or args.only_stimulation:
        rows = [r for _, r in df.iterrows()]
    else:
        _print_index(df)
        idx = int(input("Enter row index to process: "))
        rows = [df.iloc[idx]]

    if len(rows) > 1:
        print(f"Step5 will process {len(rows)} row(s).")

    summaries = []
    continue_on_error = bool(cfg_get(config, "step5", "batch", "continue_on_error", default=True))
    for row in rows:
        try:
            summaries.append(
                process_step5_row(
                    row,
                    config,
                    paths,
                    open_napari=not bool(args.no_napari),
                    overwrite_roi=args.overwrite_roi,
                )
            )
        except Exception as e:
            if not continue_on_error:
                raise
            print(f"[Step5] Skipping stem={row.get('stem', '')} after unexpected error: {e!r}")
            summaries.append({
                "stem": str(row.get("stem", "")),
                "answer_mask": 0,
                "roi_source": "none",
                "roi_paths": {},
                "roi_log": "",
                "trace_summary": _write_step5_skip_summary(row, paths, status="unexpected_step5_error", error=repr(e)),
            })
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
