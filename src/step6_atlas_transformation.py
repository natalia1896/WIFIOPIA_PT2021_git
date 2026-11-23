"""Step 6 atlas transformation for WIFIOPIA/OISI.

This module adds a group-reference atlas coordinate workflow on top of the current
stem-key pipeline:

    stem space <-> group reference space <-> group-specific Allen atlas space

Workflow assumptions
--------------------
* The Step3 reference frame (``group_id`` + ``ref_stem``) is the anatomical and
  functional reference for the registration group. Step6 does not select another frame.
* Step3 annotations are in rotated full-resolution space.
* Step5 response ROI masks are saved as ``.npy`` and may be in Step4/processed
  scaled space. Step6 resizes them back to the Step3 reference full-resolution
  shape before applying manual Step3 or automatic Step2 registration.
* Stimulated side is detected from the drawn response mask relative to the
  AP/midline axis, not from the protocol string.

Outputs
-------
* response masks transformed to reference space;
* side detection tables;
* group-specific atlas transform JSON files;
* atlas label maps in reference space;
* atlas label maps in each stem/processed space;
* coverage tables: percent of every atlas region covered by brain_signal.

The atlas-dependent part is optional until needed. If ``brainglobe-atlasapi`` is
not installed, index/response-side/coverage utilities can still be imported and
used. The BrainGlobe atlas projection follows the pitch-first logic prototyped in
``OISI_pitch_atlas_registration_3points_APline_pitchfirst_widget_v6_contours``
but removes lambda from the required landmarks. The atlas workflow preserves the validated AP sign convention and makes response-anchor handling configurable: atlas fit first uses response ROI on the reference stem, can optionally use same-group non-reference response ROI registered into reference space, and falls back to a population bregma/AP-axis template only when the group has no usable response ROI.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree
from skimage.transform import AffineTransform, EuclideanTransform, warp

try:  # Optional until interactive atlas annotation is requested.
    import napari
except Exception:  # pragma: no cover
    napari = None

try:  # Optional until processed zarr shape is requested.
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None

try:  # Optional until atlas loading is requested.
    from brainglobe_atlasapi import BrainGlobeAtlas
except Exception:  # pragma: no cover
    BrainGlobeAtlas = None

from common import (
    build_standard_paths,
    canonical_mask_paths,
    create_or_load_project,
    get_path_resolving_config,
    load_config,
    raw_scale_factor,
    resolve_flexible_path,
)


# ============================================================
# Generic config / paths
# ============================================================


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested YAML configuration key with a fallback value.
    
    Parameters
    ----------
    config : dict
        Complete project configuration.
    *keys : str
        Nested key sequence.
    default : Any, optional
        Returned if a key is missing or an intermediate value is not a dict.
    
    Returns
    -------
    Any
        Retrieved configuration value or ``default``.
    """
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_update_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively add missing default settings without replacing existing keys.
    
    Parameters
    ----------
    dst : dict
        Configuration to modify in place.
    src : dict
        Fallback values.
    
    Returns
    -------
    dict
        The updated destination mapping.
    """
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_update_missing(dst[key], value)
    return dst


def default_step6_config() -> dict[str, Any]:
    """Return built-in defaults for atlas registration and output organization.
    
    Returns
    -------
    dict
        Default ``step6`` YAML subtree.
    
    Notes
    -----
    These values are fallbacks, not universal experimental parameters. Explicit
    user values in ``step6`` take precedence when defaults are merged.
    """
    return {
        "folder_name": "step6_atlas_transform",
        "strict": False,
        "reference": {
            # Step6 fits one atlas transform per group_id/ref_stem.
            # The anatomical transform is always estimated in the Step3 reference
            # frame. Response anchors are selected by step6.response_anchor.
            "policy": "step3_ref_stem",
            "assumption": "ref_stem_is_anatomical_reference; response_anchor_can_be_registered_same_group_or_population_template",
            "require_step3_annotation": True,
            "require_response_mask_on_ref_stem": False,
            # Backward-compatible aliases; new code reads step6.response_anchor.
            "response_anchor_policy": "ref_then_registered_same_group_then_population",
            "use_nonref_response_masks_for_fit": True,
        },
        "response_anchor": {
            # ref_then_registered_same_group_then_population:
            #   1) use ROI on ref_stem if present;
            #   2) otherwise use any same-group non-ref ROI after Step2 registration
            #      into the reference frame;
            #   3) otherwise use population template in bregma/AP coordinates.
            # ref_then_population disables same-group non-ref response anchors.
            # ref_only disables all fallback response anchors.
            "policy": "ref_then_registered_same_group_then_population",
            "allow_nonreference_response": True,
            "fallback_to_population_template": True,
            "side_detection": "auto_from_ap_axis",
            # If present in the Step6 index, these columns can gate non-ref response.
            # Missing QC columns do not block by default; status is reported instead.
            "require_registration_qc_ok": False,
            "registration_qc_ok_values": ["ok", "good", "true", "1", "yes", "y", "да"],
            # Optional source reliability multipliers for response-center loss.
            "source_weight_multiplier": {
                "ref_stem_response_roi": 1.0,
                "same_group_registered_response_roi": 1.0,
                "population_ref_stem_response_template": 0.8,
            },
        },
        "input": {
            "target_space": "step3_reference_rotated_fullres",
            "response_roi_source": "step5_npy",
            "response_roi_filename_template": "{stem}_response_roi.npy",
            "use_step2_registration_for_response_masks": True,
            "resize_response_roi_to_reference": True,
            "processed_zarr_required_for_stem_labelmaps": False,
        },
        "geometry": {
            "pixel_size_um": 22.0,
            # Match the tested notebook convention: AP+ is caudal/posterior.
            # With a rostral/frontal landmark this means rostral AP should be
            # negative relative to bregma. Set to "rostral" only for the old
            # experimental Step6 v1_0/v1_1 convention.
            "ap_positive_direction": "caudal",  # caudal/posterior or rostral/anterior
            "rostral_ap_sign": "negative",      # negative/positive, used if ap_positive_direction is auto/custom
            "rostral_landmark_columns": [
                ["rostral_x", "rostral_y"],
                ["frontal_vessel_cross_x", "frontal_vessel_cross_y"],
            ],
            # A single visual calibration parameter. Flip it if the atlas overlay
            # or side detection is mirrored across the AP axis.
            "ml_to_image_sign": 1.0,
        },
        "side_detection": {
            "method": "response_mask_relative_to_ap_axis",
            # Positive ML is a property of the image basis; assign it to Allen
            # left/right once after visual QC. This changes only the text/atlas-side
            # label, not the response-mask transform. For the current PT2021 test,
            # the refspace PNG looked correct and only left/right text was swapped.
            "ml_positive_label": "right",
            "ambiguous_min_abs_ml_px": 20.0,
            "min_side_fraction": 0.65,
        },
        "atlas": {
            "name": "allen_mouse_25um",
            "brainglobe_dir": None,
            # Raw atlas volume axes -> canonical order AP, DV, ML.
            "axis_ap": 0,
            "axis_dv": 1,
            "axis_ml": 2,
            "ml_sign": 1.0,
            "ap_sign": 1.0,
            "dv_sign": 1.0,
            "dorsal_from_low_dv": True,
            "landmark_names": ["bregma", "rostral"],
            "landmarks_csv_name": "allen_landmarks_bregma_rostral.csv",
            "sshl_structure_queries": ["SSp-ll", "Primary somatosensory area, lower limb"],
            "left_ml_relation_to_bregma": "negative",  # negative/positive, atlas-grid convention
            "include_descendants": True,
        },
        "fit": {
            "pitch_bounds_deg": [-60.0, 60.0],
            "scale0": 1.0,
            "allow_scale_after_pitch": True,
            "scale_bounds": [0.8, 1.5],
            "scale_prior": 1.0,
            "scale_prior_sigma": 0.25,
            "pitch_local_window_deg": 4.0,
            "pitch_anchor_sigma_deg": 2.0,
            "landmark_sigma_um": 50.0,
            "response_sigma_um": 150.0,
            "allow_fit_without_response_mask": True,
            "weights": {
                "rostral_ap": 2.0,
                "response_ap": 1.0,
                "response_ml": 1.0,
                "scale_prior": 1.0,
            },
        },
        "projection": {
            # Atlas flat AP/ML label map -> reference image label map.
            "raster_stride": 1,
            "fill_label_max_dist_px": 20.0,
            "save_contours_npz": True,
            "save_reference_labelmap": True,
            "save_stem_labelmaps": True,
            "save_overlay_png": True,
            "overlay_alpha": 0.35,
            "overlay_dpi": 180,
            # QC points: projected SSp-ll / lower-limb atlas pixels used as the
            # response anchor. These are drawn on overlay to diagnose wrong side,
            # AP sign and pitch/scale problems.
            "show_response_match_points": True,
            "response_match_points_max": 250,
            "response_match_points_alpha": 0.45,
        },
        "missing_response_masks": {
            "enabled": True,
            # Build fallback response templates from reliable sources. Default is
            # ref_stem_only so the population prior is not contaminated by group
            # registration errors. Same-group non-ref masks are used directly before
            # this fallback when step6.response_anchor allows them.
            "template_source": "ref_stem_only",
            "min_masks_per_side": 2,
            "template_bin_px": 4,
            "template_radius_px": 250,
            "probability_threshold": 0.5,
            "save_imputed_masks": True,
            "imputed_fit_weight_multiplier": 0.8,
            # No response ROI -> side is unknown, no population imputation by default.
            # 'fixed' is permitted ONLY with an independently verified manual side.
            "side_policy": "missing",
            "fixed_side": None,
        },
        "full_atlas_overlay": {
            "enabled": False,
            "show_cortex": True,
            "show_cerebellum": True,
            "show_olfactory_bulbs": True,
            "show_target": True,
            "target_side_mode": "detected",  # detected / both
            "contour_min_points": 25,
            "contour_min_area_flat_px": 8.0,
            "contour_smooth_window": 7,
            "boundary_linewidth": 0.8,
            "target_linewidth": 1.6,
            "fill_alpha": 0.16,
            "target_fill_alpha": 0.28,
            "margin_px": 80,
            "dpi": 220,
            "show_image_landmarks": True,
            "show_projected_atlas_landmarks": True,
            "show_response_mask": True,
        },
        "coverage": {
            "enabled": True,
            "min_region_coverage_fraction": 0.5,
            "warn_region_coverage_fraction": 0.1,
            "min_region_area_px": 10,
        },
        "batch": {
            "filters": None,
            "selected_groups": None,
            "selected_stems": None,
            "include_bad": False,
            "skip_existing_transforms": False,
            "continue_on_error": True,
        },
        "output": {
            "response_refspace_folder": "response_masks_refspace",
            "response_template_folder": "response_mask_templates",
            "atlas_landmarks_folder": "atlas_landmarks",
            "transforms_folder": "transforms",
            "labelmaps_refspace_folder": "atlas_labelmaps_refspace",
            "labelmaps_by_stem_folder": "atlas_labelmaps_by_stem",
            "overlays_folder": "overlays",
            "tables_folder": "tables",
            "logs_folder": "logs",
            "step6_index_name": "step6_index.csv",
            "transform_log_name": "step6_transform_log.csv",
            "side_detection_log_name": "step6_side_detection_log.csv",
            "error_log_name": "step6_errors.csv",
        },
    }


def ensure_step6_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Add missing Step6 default keys to the full project configuration.
    
    Parameters
    ----------
    config : dict
        Full YAML mapping; modified in place.
    
    Returns
    -------
    dict
        Configuration including ``step6`` defaults without overwriting existing
        experiment settings.
    """
    config.setdefault("step6", {})
    manual_cfg = config["step6"].get("missing_response_masks", {}) or {}
    obsolete = {"protocol_side_to_response_side", "protocol_column", "right_protocol_regex", "left_protocol_regex"}
    present = sorted(obsolete.intersection(manual_cfg))
    if present:
        raise ValueError(
            "step6.missing_response_masks contains obsolete protocol-based side inference keys: "
            f"{present}. Remove these keys; response hemisphere must not be inferred from a protocol name."
        )
    _deep_update_missing(config["step6"], default_step6_config())
    side_cfg = config["step6"]["missing_response_masks"]
    side_policy = str(side_cfg.get("side_policy", "missing")).strip().lower()
    if side_policy not in {"missing", "fixed"}:
        raise ValueError(
            "step6.missing_response_masks.side_policy supports only 'missing' or 'fixed'; "
            "inference from protocol names is not supported."
        )
    fixed_side = str(side_cfg.get("fixed_side") or "").strip().lower()
    if side_policy == "fixed" and fixed_side not in {"left", "right"}:
        raise ValueError("side_policy='fixed' requires a manually verified fixed_side: left or right")
    if side_policy == "missing" and fixed_side in {"left", "right"}:
        raise ValueError("fixed_side is set while side_policy='missing'; use 'fixed' only after independent verification")
    return config


def load_step6_config(config_path: str | Path) -> dict[str, Any]:
    """Load the project YAML and initialize the Step6 configuration subtree.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the project YAML file.
    
    Returns
    -------
    dict
        Full configuration with missing Step6 defaults.
    """
    config = load_config(config_path)
    return ensure_step6_defaults(config)


def build_step6_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create and return output directories for Step6 products.
    
    Parameters
    ----------
    config : dict
        Project configuration; names are read from ``step6.output``.
    paths : dict
        Standard project paths, including ``out_root``.
    
    Returns
    -------
    dict of pathlib.Path
        Step6 directories and CSV file paths.
    
    Notes
    -----
    Creates the configured output folders as a side effect.
    """
    step6 = ensure_step6_defaults(config)["step6"]
    out_cfg = step6.get("output", {}) or {}
    root = Path(paths["out_root"]) / step6.get("folder_name", "step6_atlas_transform")

    out = {
        "step6_root": root,
        "response_refspace_root": root / out_cfg.get("response_refspace_folder", "response_masks_refspace"),
        "response_template_root": root / out_cfg.get("response_template_folder", "response_mask_templates"),
        "atlas_landmarks_root": root / out_cfg.get("atlas_landmarks_folder", "atlas_landmarks"),
        "transforms_root": root / out_cfg.get("transforms_folder", "transforms"),
        "labelmaps_refspace_root": root / out_cfg.get("labelmaps_refspace_folder", "atlas_labelmaps_refspace"),
        "labelmaps_by_stem_root": root / out_cfg.get("labelmaps_by_stem_folder", "atlas_labelmaps_by_stem"),
        "overlays_root": root / out_cfg.get("overlays_folder", "overlays"),
        "tables_root": root / out_cfg.get("tables_folder", "tables"),
        "logs_root": root / out_cfg.get("logs_folder", "logs"),
    }
    out["step6_index_csv"] = out["tables_root"] / out_cfg.get("step6_index_name", "step6_index.csv")
    out["transform_log_csv"] = out["tables_root"] / out_cfg.get("transform_log_name", "step6_transform_log.csv")
    out["side_detection_log_csv"] = out["tables_root"] / out_cfg.get("side_detection_log_name", "step6_side_detection_log.csv")
    out["error_log_csv"] = out["logs_root"] / out_cfg.get("error_log_name", "step6_errors.csv")
    out["atlas_landmarks_csv"] = out["atlas_landmarks_root"] / cfg_get(
        config, "step6", "atlas", "landmarks_csv_name", default="allen_landmarks_bregma_rostral.csv"
    )

    for p in out.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return out


def standard_project_paths(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    """Load Step6 settings and resolve standard and Step6-specific project paths.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        Project YAML path.
    
    Returns
    -------
    config : dict
        Full configuration with Step6 defaults.
    paths : dict of pathlib.Path
        Standard and Step6-specific paths.
    """
    config_path = Path(config_path)
    config = load_step6_config(config_path)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)
    paths = {**paths, **build_step6_paths(config, paths)}
    return config, paths


# ============================================================
# Small IO helpers
# ============================================================


def _as_list_or_none(value: Any) -> list[str] | None:
    """Normalize optional scalar and collection filters to strings.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _bool_series(s: pd.Series) -> pd.Series:
    """Interpret common textual boolean markers in a pandas Series.
    """
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _to_num(value: Any, default: float = float("nan")) -> float:
    """Convert a value to float, using the supplied default for invalid data.
    """
    try:
        v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(v) if pd.notna(v) else float(default)
    except Exception:
        return float(default)


def _resolve_optional_path(value: Any, path_cfg: dict[str, Any], *, must_exist: bool = False) -> Path | None:
    """Resolve a possibly absent path using the project root/drive mappings.
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


def normalize01(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize finite image intensities to [0, 1] for visualization only.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image in any intensity units.
    p_low, p_high : float
        Lower and upper percentile limits for display normalization.
    
    Returns
    -------
    numpy.ndarray
        Float32 array with the same shape and display values within [0, 1].
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [float(p_low), float(p_high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    return np.clip((arr - lo) / max(hi - lo, 1e-12), 0, 1).astype(np.float32)


def save_json(path: str | Path, obj: dict[str, Any]) -> None:
    """Serialize a mapping to an indented UTF-8 JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_or_create_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append a metadata row to a CSV, creating the table if necessary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        df_old = pd.read_csv(path)
        df = pd.concat([df_old, df_new], ignore_index=True, sort=False)
    else:
        df = df_new
    df.to_csv(path, index=False)


# ============================================================
# Step6 index
# ============================================================


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """Restrict a stem-level table using optional metadata equality filters.
    """
    out = df.copy()
    for key, allowed_raw in (filters or {}).items():
        if key not in out.columns:
            raise ValueError(f"Step6 filter column not found: {key}")
        values = out[key].astype(str)
        if isinstance(allowed_raw, dict):
            mask = pd.Series(True, index=out.index)
            if allowed_raw.get("regex") not in (None, ""):
                mask &= values.str.contains(str(allowed_raw["regex"]), regex=True, na=False)
            include = set(_as_list_or_none(allowed_raw.get("include")) or [])
            exclude = set(_as_list_or_none(allowed_raw.get("exclude")) or [])
            if include:
                mask &= values.isin(include)
            if exclude:
                mask &= ~values.isin(exclude)
            if allowed_raw.get("contains") not in (None, ""):
                mask &= values.str.contains(str(allowed_raw["contains"]), regex=False, na=False)
            out = out[mask].copy()
        else:
            allowed = set(_as_list_or_none(allowed_raw) or [])
            if allowed:
                out = out[values.isin(allowed)].copy()
    return out


def _reference_annotation_complete(row: pd.Series, config: dict[str, Any]) -> bool:
    """Test whether all Step3 landmarks required by Step6 are available.
    """
    required = ["ann_bregma_x", "ann_bregma_y", "ann_axis_p1_x", "ann_axis_p1_y", "ann_axis_p2_x", "ann_axis_p2_y"]
    has_rostral = False
    for xcol, ycol in cfg_get(config, "step6", "geometry", "rostral_landmark_columns", default=[]):
        ax, ay = f"ann_{xcol}", f"ann_{ycol}"
        if ax in row.index and ay in row.index and np.isfinite(_to_num(row.get(ax))) and np.isfinite(_to_num(row.get(ay))):
            has_rostral = True
            break
    if not has_rostral:
        return False
    for col in required:
        if col not in row.index or not np.isfinite(_to_num(row.get(col))):
            return False
    return True


def _load_reference_annotations(path: Path) -> pd.DataFrame:
    """Load Step3 reference annotations with prefixed columns for merging.
    """
    if not path.exists():
        return pd.DataFrame()
    ann = pd.read_csv(path)
    if ann.empty:
        return ann
    # Last annotation per (group_id, ref_stem) wins.
    ann = ann.copy()
    if "updated_at" in ann.columns:
        ann = ann.sort_values("updated_at")
    ann = ann.groupby([ann["group_id"].astype(str), ann["ref_stem"].astype(str)], sort=False).tail(1)
    rename = {c: f"ann_{c}" for c in ann.columns if c not in {"group_id", "ref_stem"}}
    return ann.rename(columns=rename)


def _first_existing_path_from_columns(row: pd.Series, columns: list[str], path_cfg: dict[str, Any], *, what: str) -> Path | None:
    """Return the first existing image path listed in a reference metadata row.
    """
    tried: list[str] = []
    for col in columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            tried.append(f"{col}={row[col]}")
            p = _resolve_optional_path(row[col], path_cfg, must_exist=False)
            if p is not None and p.exists():
                return p
    return None


def _get_reference_image_path_for_group(ref_row: pd.Series, config: dict[str, Any], path_cfg: dict[str, Any]) -> Path | None:
    # Match Step3 selection logic without importing the Step3 runner.
    """Resolve the configured reference image for a Step3 registration group.
    """
    ref_cfg = cfg_get(config, "step3", "reference_image", default={}) or {}
    source = str(ref_cfg.get("source", "registration")).strip().lower()
    if source in {"registration", "registration_fixed", "fixed_registration"}:
        return _first_existing_path_from_columns(
            ref_row,
            ["fixed_registration_path_resolved", "fixed_registration_path"],
            path_cfg,
            what="Step3 reference image",
        )
    if source in {"wavelength", "ref_wavelength", "reference_wavelength"}:
        wl = ref_cfg.get("wavelength")
        if wl is not None and str(wl).strip():
            return _first_existing_path_from_columns(
                ref_row,
                [f"ref_{wl}_path_resolved", f"ref_{wl}_path"],
                path_cfg,
                what=f"Step3 reference image wavelength {wl}",
            )
    if source == "column":
        col = ref_cfg.get("column")
        if col is not None and str(col).strip():
            return _first_existing_path_from_columns(ref_row, [f"{col}_resolved", str(col)], path_cfg, what="Step3 reference image column")
    # Fallbacks.
    dynamic = [c for c in ref_row.index if str(c).startswith("ref_") and str(c).endswith("_path")]
    return _first_existing_path_from_columns(
        ref_row,
        ["fixed_registration_path_resolved", "fixed_registration_path"] + dynamic,
        path_cfg,
        what="Step3 reference image fallback",
    )


def _step5_response_root(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    """Construct the Step5 response-ROI folder from the configured project paths.
    """
    step5_cfg = config.get("step5", {}) or {}
    root = Path(paths["out_root"]) / step5_cfg.get("folder_name", "step5_response_detection")
    folder = step5_cfg.get("response_masks_drawn_folder", "response_masks_drawn")
    return root / folder


def build_step6_index(
    config_path: str | Path,
    *,
    include_bad: bool | None = None,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    require_step3_annotation: bool | None = None,
    require_response_roi: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Construct the Step6 stem table from recorded and anatomical metadata.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        Full project YAML path.
    include_bad : bool or None, optional
        Whether to retain rows marked ``is_bad``. The default comes from
        ``step6.batch.include_bad``.
    filters : dict or None, optional
        Metadata filters, overriding ``step6.batch.filters`` when provided.
    groups, stems : sequence of str or None, optional
        Group and recording identifiers to retain. Defaults come from
        ``step6.batch.selected_groups`` and ``selected_stems``.
    require_step3_annotation : bool or None, optional
        Require complete anatomical annotations; if None, read
        ``step6.reference.require_step3_annotation``.
    require_response_roi : bool, optional
        Keep only rows with an available, non-superseded Step5 response ROI.
    
    Returns
    -------
    df : pandas.DataFrame
        Stem-level table including ``group_id``, ``ref_stem``, resolved paths,
        reference landmarks, registration information and response-ROI status.
    config : dict
        Full configuration including Step6 defaults.
    paths : dict of pathlib.Path
        Project and Step6 output paths.
    
    Notes
    -----
    A single atlas fit is shared by each ``group_id``/``ref_stem`` pair. This step
    prepares anatomical projection inputs; it does not independently apply
    ``analysis_exclusions`` for scientific cohort selection. It does honor any
    explicit batch filters and the configured ``include_bad`` flag. Existing Step3
    manual registration overrides are merged when their CSV is present.
    The index CSV is saved as a side effect.
    """
    config, paths = standard_project_paths(config_path)
    step6 = config["step6"]
    batch_cfg = step6.get("batch", {}) or {}

    if include_bad is None:
        include_bad = bool(batch_cfg.get("include_bad", False))
    if filters is None:
        filters = batch_cfg.get("filters")
    if groups is None:
        groups = _as_list_or_none(batch_cfg.get("selected_groups"))
    if stems is None:
        stems = _as_list_or_none(batch_cfg.get("selected_stems"))
    if require_step3_annotation is None:
        require_step3_annotation = bool(cfg_get(config, "step6", "reference", "require_step3_annotation", default=True))

    manifest_path = paths["manifest_path"]
    metadata_path = paths["metadata_registration_csv"]
    ann_path = paths["reference_annotations_csv"]
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_recordings.csv not found: {manifest_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_registration.csv not found: {metadata_path}")

    manifest = pd.read_csv(manifest_path)
    meta = pd.read_csv(metadata_path)
    if "stem" not in manifest.columns or "stem" not in meta.columns:
        raise ValueError("Both manifest_recordings.csv and metadata_registration.csv must contain 'stem'.")

    manifest_extra_cols = [c for c in manifest.columns if c == "stem" or c not in meta.columns]
    df = meta.merge(manifest[manifest_extra_cols], on="stem", how="left", suffixes=("", "_manifest"))
    # Step6 is an anatomical projection stage, not a scientific cohort selection.
    # The old is_bad field is retained for diagnostic output; true missing/invalid
    # geometry is handled by the existing per-group errors.
    # A caller may explicitly ask for this Step2 legacy filter, but the default
    # in the project YAML is include_bad=True, not unapproved automated QC.
    if not include_bad and "is_bad" in df.columns:
        df = df[~_bool_series(df["is_bad"])].copy()
    df = _apply_filters(df, filters)
    if groups:
        df = df[df["group_id"].astype(str).isin({str(g) for g in groups})].copy()
    if stems:
        df = df[df["stem"].astype(str).isin({str(s) for s in stems})].copy()

    ann = _load_reference_annotations(ann_path)
    if not ann.empty:
        df = df.merge(ann, on=["group_id", "ref_stem"], how="left")

    # Attach saved final original-stem -> reference registration, if any.
    # Step3 GUI persists the full 3x3 affine in final_transform_xy_json.
    man_cfg = cfg_get(config, "step3", "manual_registration", default={}) or {}
    override_csv = (Path(paths["step3_root"]) / str(man_cfg.get("folder_name", "manual_registration"))
                    / str(man_cfg.get("overrides_csv_name", "manual_registration_overrides.csv")))
    if override_csv.is_file():
        overrides = pd.read_csv(override_csv)
        if not overrides.empty:
            if "stem" not in overrides.columns:
                raise KeyError(f"Manual registration overrides lack stem: {override_csv}")
            keys = [k for k in ["group_id", "stem"] if k in overrides.columns]
            if "updated_at" in overrides:
                overrides = overrides.sort_values("updated_at")
            overrides = overrides.drop_duplicates(keys, keep="last")
            overrides["manual_override_present"] = True
            keep = keys + [col for col in ["final_transform_xy_json", "final_translation_x", "final_translation_y", "final_rotation_rad", "manual_override_present"] if col in overrides]
            overrides = overrides[keep].rename(columns={col: "step3_" + col for col in keep if col not in keys})
            df = df.merge(overrides, on=keys, how="left", validate="many_to_one")

    df["step3_annotation_complete"] = df.apply(lambda r: _reference_annotation_complete(r, config), axis=1)
    if require_step3_annotation:
        df = df[df["step3_annotation_complete"].astype(bool)].copy()

    path_cfg = get_path_resolving_config(config, "step6")
    processed_root = paths["processed_signals_root"]
    response_root = _step5_response_root(config, paths)

    # Reference image path per group/ref_stem.
    ref_paths_by_key: dict[tuple[str, str], str] = {}
    for (gid, ref), g in df.groupby([df["group_id"].astype(str), df["ref_stem"].astype(str)], sort=False):
        ref_rows = g[g["stem"].astype(str).eq(str(ref))]
        ref_row = ref_rows.iloc[0] if not ref_rows.empty else g.iloc[0]
        ref_path = _get_reference_image_path_for_group(ref_row, config, path_cfg)
        ref_paths_by_key[(str(gid), str(ref))] = str(ref_path) if ref_path is not None else ""

    proc_paths: list[str] = []
    proc_exists: list[bool] = []
    response_paths: list[str] = []
    response_exists: list[bool] = []
    ref_image_paths: list[str] = []
    ref_image_exists: list[bool] = []
    response_refspace_paths: list[str] = []
    stem_label_paths: list[str] = []
    usable_label_paths: list[str] = []

    template = cfg_get(config, "step6", "input", "response_roi_filename_template", default="{stem}_response_roi.npy")
    for _, row in df.iterrows():
        stem = str(row["stem"])
        gid = str(row.get("group_id", ""))
        ref = str(row.get("ref_stem", ""))
        p = processed_root / f"{stem}_processed.zarr"
        if "processed_zarr_path" in row.index and pd.notna(row.get("processed_zarr_path")) and str(row.get("processed_zarr_path")).strip():
            q = _resolve_optional_path(row.get("processed_zarr_path"), path_cfg, must_exist=False)
            if q is not None:
                p = q
        proc_paths.append(str(p))
        proc_exists.append(p.exists())

        roi = response_root / str(template).format(stem=stem, group_id=gid, ref_stem=ref)
        response_paths.append(str(roi))
        response_exists.append(roi.exists())

        ref_path = ref_paths_by_key.get((gid, ref), "")
        ref_image_paths.append(ref_path)
        ref_image_exists.append(bool(ref_path and Path(ref_path).exists()))

        response_refspace_paths.append(str(paths["response_refspace_root"] / f"{stem}_response_roi_refspace.npy"))
        stem_label_paths.append(str(paths["labelmaps_by_stem_root"] / f"{stem}_atlas_labels_processed.npy"))
        usable_label_paths.append(str(paths["labelmaps_by_stem_root"] / f"{stem}_atlas_usable_labels_processed.npy"))

    df["processed_zarr_path_resolved"] = proc_paths
    df["processed_zarr_exists"] = proc_exists
    df["response_roi_path"] = response_paths
    # Step5's explicit confirmation of no visible response overrides a stale mask
    # left on disk from an earlier review iteration.
    step5_cfg = config.get("step5", {}) or {}
    roi_log = (Path(paths["out_root"]) / str(step5_cfg.get("folder_name", "step5_response_detection"))
               / str(step5_cfg.get("response_tables_folder", "response_tables"))
               / str(step5_cfg.get("roi_log_name", "response_roi_log.csv")))
    if roi_log.is_file():
        log = pd.read_csv(roi_log)
        if "stem" in log:
            if "updated_at" in log:
                log = log.sort_values("updated_at")
            log = log.drop_duplicates("stem", keep="last")
            status_cols = [c for c in ["roi_status", "status", "response_status", "decision"] if c in log.columns]
            if status_cols:
                bad = {"no_visible_response", "no_response_roi", "no_response", "absent"}
                stale = set()
                for _, entry in log.iterrows():
                    if any(str(entry[c]).strip().lower() in bad for c in status_cols):
                        stale.add(str(entry["stem"]))
                response_exists = [bool(ok and stem not in stale) for ok, stem in zip(response_exists, df["stem"].astype(str))]
    df["response_roi_exists"] = response_exists
    df["ref_image_path"] = ref_image_paths
    df["ref_image_exists"] = ref_image_exists
    df["response_roi_refspace_path"] = response_refspace_paths
    df["atlas_labels_processed_path"] = stem_label_paths
    df["atlas_usable_labels_processed_path"] = usable_label_paths

    if require_response_roi:
        df = df[df["response_roi_exists"].astype(bool)].copy()

    df = df.reset_index(drop=True)
    df.to_csv(paths["step6_index_csv"], index=False)
    return df, config, paths


# ============================================================
# Reference image and AP/ML basis
# ============================================================


def load_reference_image_for_row(row: pd.Series) -> np.ndarray:
    """Read the registered reference image as a 2D or RGB NumPy array.
    """
    path = Path(str(row.get("ref_image_path", "")))
    if not path.exists():
        raise FileNotFoundError(f"Reference image not found for group={row.get('group_id', '')}: {path}")
    img = np.asarray(iio.imread(path))
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        return img[..., :3]
    if img.ndim == 3:
        return np.nanmean(img, axis=0)
    return img


def _annotation_xy(row: pd.Series, prefix: str, x_col: str, y_col: str) -> np.ndarray | None:
    """Read an annotated (x, y) landmark in full-resolution image pixels.
    """
    x = _to_num(row.get(f"{prefix}{x_col}"))
    y = _to_num(row.get(f"{prefix}{y_col}"))
    if np.isfinite(x) and np.isfinite(y):
        return np.array([x, y], dtype=float)
    return None


def choose_rostral_xy(row: pd.Series, config: dict[str, Any], prefix: str = "ann_") -> tuple[np.ndarray, str]:
    """Choose the first available rostral landmark specified in the YAML.
    
    Parameters
    ----------
    row : pandas.Series
        Step3 reference annotation and registration metadata.
    config : dict
        Contains ``step6.geometry.rostral_landmark_columns``.
    prefix : str, optional
        Prefix of merged annotation column names.
    
    Returns
    -------
    xy : numpy.ndarray
        Two-element (x, y) point in full-resolution reference pixels.
    source : str
        Name of the coordinate pair used.
    
    Raises
    ------
    RuntimeError
        If no configured rostral landmark is available.
    """
    for pair in cfg_get(config, "step6", "geometry", "rostral_landmark_columns", default=[]):
        if len(pair) != 2:
            continue
        xy = _annotation_xy(row, prefix, pair[0], pair[1])
        if xy is not None:
            return xy, f"{pair[0]}/{pair[1]}"
    raise RuntimeError(
        "No rostral landmark found in Step3 annotation. Expected one of: "
        f"{cfg_get(config, 'step6', 'geometry', 'rostral_landmark_columns', default=[])}"
    )


def project_xy_to_axis(xy: np.ndarray, axis_p1_xy: np.ndarray, ap_hat_xy: np.ndarray) -> np.ndarray:
    """Orthogonally project an (x, y) point onto the AP midline axis.
    """
    return axis_p1_xy + np.dot(np.asarray(xy, dtype=float) - axis_p1_xy, ap_hat_xy) * ap_hat_xy


def _desired_rostral_ap_sign(config: dict[str, Any]) -> float:
    """Return the expected sign of the rostral landmark on the AP axis.
    
    Notes
    -----
    The sign follows ``step6.geometry.ap_positive_direction`` when its value is
    recognized, otherwise ``step6.geometry.rostral_ap_sign``. A caudal-positive
    AP axis places the rostral landmark at negative AP.
    """
    direction = str(cfg_get(config, "step6", "geometry", "ap_positive_direction", default="caudal")).strip().lower()
    if direction in {"caudal", "posterior", "lambda", "bregma_to_lambda"}:
        return -1.0
    if direction in {"rostral", "anterior", "frontal", "bregma_to_rostral"}:
        return 1.0
    sign_text = str(cfg_get(config, "step6", "geometry", "rostral_ap_sign", default="negative")).strip().lower()
    return 1.0 if sign_text in {"positive", "+", "+1", "1"} else -1.0


def reference_basis_from_row(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    """Construct the anatomical AP/ML coordinate basis for a reference frame.
    
    Parameters
    ----------
    row : pandas.Series
        Step3 reference annotations including bregma, a rostral landmark and
        two midline endpoints, all in full-resolution (x, y) image pixels.
    config : dict
        Geometry keys under ``step6.geometry`` include
        ``ap_positive_direction``, ``ml_to_image_sign``, ``pixel_size_um`` and
        ``rostral_landmark_columns``.
    
    Returns
    -------
    dict
        Reference origin and anatomical landmarks as (x, y) pixel vectors;
        unit AP/ML direction vectors; pixel size in µm/px; and rostral
        coordinates in µm.
    
    Notes
    -----
    The AP direction is oriented so the rostral landmark has the configured sign.
    The origin is the orthogonal projection of bregma onto the annotated midline,
    not necessarily the unprojected bregma point. ML is perpendicular to AP and
    its sign is controlled by ``step6.geometry.ml_to_image_sign``.
    
    Raises
    ------
    RuntimeError
        If bregma, a rostral landmark or a usable midline axis is missing.
    """
    bregma_xy = _annotation_xy(row, "ann_", "bregma_x", "bregma_y")
    if bregma_xy is None:
        raise RuntimeError("Step3 annotation has no bregma for this group/ref_stem")
    rostral_xy, rostral_source = choose_rostral_xy(row, config, prefix="ann_")

    p1 = _annotation_xy(row, "ann_", "axis_p1_x", "axis_p1_y")
    p2 = _annotation_xy(row, "ann_", "axis_p2_x", "axis_p2_y")
    if p1 is None or p2 is None:
        raise RuntimeError("Step3 annotation has no AP/midline axis")
    ap_hat = p2 - p1
    ap_norm = float(np.linalg.norm(ap_hat))
    if ap_norm < 1e-9:
        raise RuntimeError("Step3 AP-axis length is zero")
    ap_hat = ap_hat / ap_norm

    bregma_proj = project_xy_to_axis(bregma_xy, p1, ap_hat)
    rostral_proj = project_xy_to_axis(rostral_xy, p1, ap_hat)
    rostral_ap0 = float(np.dot(rostral_proj - bregma_proj, ap_hat))
    desired = _desired_rostral_ap_sign(config)
    if rostral_ap0 == 0 or np.sign(rostral_ap0) != np.sign(desired):
        ap_hat = -ap_hat
        bregma_proj = project_xy_to_axis(bregma_xy, p1, ap_hat)
        rostral_proj = project_xy_to_axis(rostral_xy, p1, ap_hat)

    ml_sign = float(cfg_get(config, "step6", "geometry", "ml_to_image_sign", default=1.0))
    ml_hat = ml_sign * np.array([-ap_hat[1], ap_hat[0]], dtype=float)

    pixel_size_um = float(cfg_get(config, "step6", "geometry", "pixel_size_um", default=22.0))
    rostral_ap_um = float(np.dot(rostral_proj - bregma_proj, ap_hat) * pixel_size_um)
    return {
        "origin_xy": bregma_proj.astype(float),
        "bregma_xy": bregma_xy.astype(float),
        "bregma_proj_xy": bregma_proj.astype(float),
        "rostral_xy": rostral_xy.astype(float),
        "rostral_proj_xy": rostral_proj.astype(float),
        "axis_p1_xy": p1.astype(float),
        "axis_p2_xy": p2.astype(float),
        "ap_hat_xy": ap_hat.astype(float),
        "ml_hat_xy": ml_hat.astype(float),
        "pixel_size_um": pixel_size_um,
        "rostral_source": rostral_source,
        "ap_positive_direction": str(cfg_get(config, "step6", "geometry", "ap_positive_direction", default="caudal")),
        "rostral_expected_ap_sign": "positive" if desired > 0 else "negative",
        "rostral_ap_um": rostral_ap_um,
        "rostral_ml_um": float(np.dot(rostral_proj - bregma_proj, ml_hat) * pixel_size_um),
    }


def xy_to_ml_ap_px(xy: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """Convert reference (x, y) pixel coordinates to (ML, AP) pixel offsets.
    
    Parameters
    ----------
    xy : numpy.ndarray
        Point(s) ending in a length-2 (x, y) coordinate axis.
    basis : dict
        Reference AP/ML basis from ``reference_basis_from_row``.
    
    Returns
    -------
    numpy.ndarray
        Coordinate array ending in (ML, AP), in full-resolution pixels.
    """
    xy = np.asarray(xy, dtype=float)
    delta = xy - np.asarray(basis["origin_xy"], dtype=float)
    return np.stack([
        np.dot(delta, basis["ml_hat_xy"]),
        np.dot(delta, basis["ap_hat_xy"]),
    ], axis=-1)


def xy_to_ml_ap_um(xy: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """Convert reference (x, y) image pixels to (ML, AP) in µm.
    """
    return xy_to_ml_ap_px(xy, basis) * float(basis["pixel_size_um"])


def ml_ap_um_to_xy(ml_ap_um: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """Convert (ML, AP) offsets in µm to reference-image (x, y) pixels.
    """
    v = np.asarray(ml_ap_um, dtype=float)
    ml_um = v[..., 0]
    ap_um = v[..., 1]
    pix = float(basis["pixel_size_um"])
    return (
        np.asarray(basis["origin_xy"], dtype=float)
        + (ml_um[..., None] / pix) * np.asarray(basis["ml_hat_xy"], dtype=float)
        + (ap_um[..., None] / pix) * np.asarray(basis["ap_hat_xy"], dtype=float)
    )


# ============================================================
# Response masks in reference space and side detection
# ============================================================


def resize_mask_nearest(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D binary ROI to (height, width) using nearest neighbors.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    if tuple(mask.shape) == (h, w):
        return mask.astype(bool)
    out = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)


def resize_label_nearest(labels: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D integer atlas label map without interpolating region IDs.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    if tuple(labels.shape) == (h, w):
        return labels.astype(np.int32, copy=False)
    out = cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.int32, copy=False)


def registration_tform_stem_to_ref(row: pd.Series) -> AffineTransform:
    """Select the saved stem-to-reference spatial transform.
    
    Parameters
    ----------
    row : pandas.Series
        Registration metadata for the recording, including optional final
        Step3 manual 3×3 affine and Step2 Euclidean parameters.
    
    Returns
    -------
    skimage.transform.AffineTransform
        Forward transform from the stem's full-resolution (x, y) pixels to the
        Step3 reference image. Reference stems yield the identity transform.
    
    Notes
    -----
    The final Step3 manual affine takes precedence when available, followed by
    the Step2 registration. These are image-pixel transformations, not µm.
    """
    if str(row.get("stem", "")) == str(row.get("ref_stem", "")):
        return AffineTransform()
    raw = row.get("step3_final_transform_xy_json")
    if raw is not None and not pd.isna(raw) and str(raw).strip():
        try:
            mat = np.asarray(json.loads(str(raw)), dtype=float)
            if (mat.shape != (3, 3) or not np.isfinite(mat).all()
                or not np.allclose(mat[2], [0, 0, 1], atol=1e-9)
                or abs(np.linalg.det(mat[:2, :2])) < 1e-10):
                raise ValueError("invalid 3x3 affine")
            return AffineTransform(matrix=mat)
        except Exception as exc:
            raise RuntimeError(f"Malformed Step3 final_transform_xy_json for {row.get('stem')}: {exc}") from exc
    # Legacy Step3 numeric final fields, if JSON column was not saved in older QC.
    manual = [row.get("step3_final_translation_x"), row.get("step3_final_translation_y"), row.get("step3_final_rotation_rad")]
    provided = [v is not None and pd.notna(v) and str(v).strip() for v in manual]
    if any(provided):
        vals = [_to_num(v) for v in manual]
        if not all(np.isfinite(v) for v in vals):
            raise RuntimeError(f"Incomplete Step3 manual registration for stem={row.get('stem')}")
        return AffineTransform(matrix=EuclideanTransform(rotation=float(vals[2]), translation=(float(vals[0]), float(vals[1]))).params)
    if str(row.get("step3_manual_override_present", False)).strip().lower() in {"true", "1"}:
        raise RuntimeError(f"Step3 manual override exists but has no valid final transform for stem={row.get('stem')}")
    tx, ty, rot = (_to_num(row.get(k)) for k in ["translation_x", "translation_y", "rotation_rad"])
    if not (np.isfinite(tx) and np.isfinite(ty) and np.isfinite(rot)):
        raise RuntimeError(f"Missing Step2 stem->reference transform for stem={row.get('stem', '')}")
    return AffineTransform(matrix=EuclideanTransform(rotation=float(rot), translation=(float(tx), float(ty))).params)


def warp_mask_stem_to_reference(mask_fullres: np.ndarray, row: pd.Series, ref_shape: tuple[int, int]) -> np.ndarray:
    """Warp a binary stem mask to the full-resolution reference image.
    
    Parameters
    ----------
    mask_fullres : numpy.ndarray
        Binary mask in the stem's registered-input image pixels, (y, x).
    row : pandas.Series
        Stem registration metadata.
    ref_shape : tuple of int
        Target reference (height, width), in pixels.
    
    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``ref_shape``.
    
    Notes
    -----
    Uses nearest-neighbor interpolation (order 0) and the authoritative
    stem-to-reference transform. Outside pixels are set to False.
    """
    if str(row.get("stem", "")) == str(row.get("ref_stem", "")):
        return resize_mask_nearest(mask_fullres, ref_shape)
    tform = registration_tform_stem_to_ref(row)
    warped = warp(
        mask_fullres.astype(np.float32),
        inverse_map=tform.inverse,
        output_shape=tuple(ref_shape),
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0,
    )
    return warped > 0.5


def mask_centroid_xy(mask: np.ndarray) -> np.ndarray | None:
    """Return the binary-mask centroid in (x, y) pixels or None if empty.
    """
    yy, xx = np.where(np.asarray(mask, dtype=bool))
    if yy.size == 0:
        return None
    return np.array([float(xx.mean()), float(yy.mean())], dtype=float)


def load_response_roi_for_row(row: pd.Series) -> np.ndarray | None:
    """Load an optional 2D boolean Step5 response ROI from its resolved path.
    """
    p = Path(str(row.get("response_roi_path", "")))
    if not p.exists():
        return None
    arr = np.load(p)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D response ROI mask, got {arr.shape}: {p}")
    return arr.astype(bool)


def response_roi_to_reference_space(row: pd.Series, config: dict[str, Any], ref_shape: tuple[int, int]) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Resize and register a Step5 response ROI into Step3 reference space.
    
    Parameters
    ----------
    row : pandas.Series
        Stem metadata containing the Step5 response ROI path and spatial transform.
    config : dict
        ``step6.input.resize_response_roi_to_reference`` selects whether spatial
        resizing is allowed before registration.
    ref_shape : tuple of int
        Full-resolution reference (height, width) in pixels.
    
    Returns
    -------
    mask_ref : numpy.ndarray or None
        Boolean (y, x) mask in the reference frame, or None if the ROI is missing.
    metadata : dict
        Response status, source path and original/reference shapes and area.
    
    Notes
    -----
    Step5 masks may be stored at the processed image resolution. They are resized
    before application of the Step2 or Step3 stem-to-reference transform.
    """
    mask = load_response_roi_for_row(row)
    if mask is None:
        return None, {
            "response_mask_status": "missing",
            "response_roi_path": str(row.get("response_roi_path", "")),
        }

    original_shape = tuple(mask.shape)
    if bool(cfg_get(config, "step6", "input", "resize_response_roi_to_reference", default=True)):
        mask_fullres = resize_mask_nearest(mask, ref_shape)
    else:
        mask_fullres = mask.astype(bool)
        if tuple(mask_fullres.shape) != tuple(ref_shape):
            raise ValueError(f"response ROI shape {mask_fullres.shape} != ref_shape {ref_shape}; enable resize_response_roi_to_reference")

    mask_ref = warp_mask_stem_to_reference(mask_fullres, row, ref_shape)
    status = "existing_ref_stem" if str(row.get("stem", "")) == str(row.get("ref_stem", "")) else "registered_to_ref"
    return mask_ref.astype(bool), {
        "response_mask_status": status,
        "response_roi_path": str(row.get("response_roi_path", "")),
        "response_original_shape": json.dumps(list(map(int, original_shape))),
        "response_ref_shape": json.dumps(list(map(int, ref_shape))),
        "response_area_px_refspace": int(mask_ref.sum()),
    }


def save_response_refspace_mask(row: pd.Series, mask_ref: np.ndarray, paths: dict[str, Path]) -> Path:
    """Save the reference-space response mask as NumPy and PNG images.
    """
    stem = str(row["stem"])
    out = paths["response_refspace_root"] / f"{stem}_response_roi_refspace.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, mask_ref.astype(np.uint8))
    png = out.with_suffix(".png")
    iio.imwrite(png, (mask_ref.astype(np.uint8) * 255))
    return out


def detect_response_side(mask_ref: np.ndarray, basis: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Classify the hemisphere of a response ROI relative to the AP axis.
    
    Parameters
    ----------
    mask_ref : numpy.ndarray
        Boolean 2D (y, x) mask in full-resolution reference pixels.
    basis : dict
        Reference anatomical basis; ML/AP coordinates use its configured sign.
    config : dict
        Side label mapping and thresholds are read from ``step6.side_detection``.
    
    Returns
    -------
    dict
        Side label (``left``, ``right``, ``ambiguous`` or ``missing``), confidence,
        centroid in reference pixels and ML/AP pixel/µm coordinates, and the
        proportions of mask pixels on either side of the midline.
    
    Notes
    -----
    Uses the centroid's absolute ML distance and the configured minimum fraction
    of mask pixels on one side. Changing ``ml_positive_label`` changes the
    left/right **text labels**, not the mask geometry.
    """
    mask = np.asarray(mask_ref, dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "detected_side": "missing",
            "side_confidence": 0.0,
            "response_center_x": np.nan,
            "response_center_y": np.nan,
            "response_ml_px": np.nan,
            "response_ap_px": np.nan,
            "positive_side_fraction": np.nan,
            "negative_side_fraction": np.nan,
            "side_source": "empty_response_mask",
        }

    yy, xx = np.where(mask)
    xy = np.column_stack([xx.astype(float), yy.astype(float)])
    ml_ap_px = xy_to_ml_ap_px(xy, basis)
    ml_vals = ml_ap_px[:, 0]
    pos_frac = float(np.mean(ml_vals > 0))
    neg_frac = float(np.mean(ml_vals < 0))
    center = mask_centroid_xy(mask)
    center_ml_ap_px = xy_to_ml_ap_px(center, basis)

    ml_positive_label = str(cfg_get(config, "step6", "side_detection", "ml_positive_label", default="right")).strip().lower()
    ml_negative_label = "right" if ml_positive_label == "left" else "left"
    min_side_frac = float(cfg_get(config, "step6", "side_detection", "min_side_fraction", default=0.65))
    min_abs_ml = float(cfg_get(config, "step6", "side_detection", "ambiguous_min_abs_ml_px", default=20.0))

    if abs(float(center_ml_ap_px[0])) < min_abs_ml:
        ml_axis_side = "midline"
        side = "ambiguous"
        conf = max(pos_frac, neg_frac)
    elif pos_frac >= min_side_frac:
        ml_axis_side = "ml_positive"
        side = ml_positive_label
        conf = pos_frac
    elif neg_frac >= min_side_frac:
        ml_axis_side = "ml_negative"
        side = ml_negative_label
        conf = neg_frac
    else:
        ml_axis_side = "mixed"
        side = "ambiguous"
        conf = max(pos_frac, neg_frac)

    return {
        # ``detected_side`` is a label obtained from the configurable mapping
        # ml_positive_label/ml_negative_label. The mask geometry itself is not
        # mirrored here; only the text label changes when that mapping is flipped.
        "detected_side": side,
        "response_ml_axis_side": ml_axis_side,
        "ml_positive_label": ml_positive_label,
        "ml_negative_label": ml_negative_label,
        "side_confidence": float(conf),
        "response_center_x": float(center[0]),
        "response_center_y": float(center[1]),
        "response_ml_px": float(center_ml_ap_px[0]),
        "response_ap_px": float(center_ml_ap_px[1]),
        "response_ml_um": float(center_ml_ap_px[0] * basis["pixel_size_um"]),
        "response_ap_um": float(center_ml_ap_px[1] * basis["pixel_size_um"]),
        "positive_side_fraction": pos_frac,
        "negative_side_fraction": neg_frac,
        "side_source": "response_mask_relative_to_ap_axis_configured_label",
        "side_mapping_note": "Flip step6.side_detection.ml_positive_label if left/right text is swapped; this does not change refspace mask geometry.",
    }


def process_response_masks_to_reference(
    df: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    save: bool = True,
) -> pd.DataFrame:
    """Convert available response ROIs to reference space and log side detection.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Step6 stem index with response ROI paths and registration metadata.
    config : dict
        Full project configuration.
    paths : dict
        Step6 output paths.
    save : bool, optional
        Whether to write the reference masks and side-detection CSV.
    
    Returns
    -------
    pandas.DataFrame
        Stem-level response geometry and side classification records.
    """
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            ref_img = load_reference_image_for_row(row)
            ref_shape = tuple(ref_img.shape[:2])
            basis = reference_basis_from_row(row, config)
            mask_ref, status = response_roi_to_reference_space(row, config, ref_shape)
            out_path = ""
            side = {
                "detected_side": "missing",
                "side_confidence": 0.0,
                "side_source": "missing_response_mask",
            }
            if mask_ref is not None:
                if save:
                    out_path = str(save_response_refspace_mask(row, mask_ref, paths))
                side = detect_response_side(mask_ref, basis, config)
            rec = {
                "stem": str(row.get("stem", "")),
                "group_id": str(row.get("group_id", "")),
                "ref_stem": str(row.get("ref_stem", "")),
                "response_roi_refspace_path": out_path,
                "updated_at": _dt.datetime.now().isoformat(),
                **status,
                **side,
            }
            rows.append(rec)
            append_or_create_csv(paths["side_detection_log_csv"], rec)
        except Exception as e:
            rec = {
                "stem": str(row.get("stem", "")),
                "group_id": str(row.get("group_id", "")),
                "ref_stem": str(row.get("ref_stem", "")),
                "error": repr(e),
                "updated_at": _dt.datetime.now().isoformat(),
            }
            rows.append(rec)
            append_or_create_csv(paths["error_log_csv"], rec)
            if bool(config["step6"].get("strict", False)):
                raise
    return pd.DataFrame(rows)


# ============================================================
# Missing response-mask templates
# ============================================================


def _mask_to_relative_ml_ap_px(mask_ref: np.ndarray, basis: dict[str, Any]) -> np.ndarray:
    """Convert all nonzero mask pixels into bregma-relative (ML, AP) pixels.
    """
    yy, xx = np.where(np.asarray(mask_ref, dtype=bool))
    if yy.size == 0:
        return np.empty((0, 2), dtype=float)
    xy = np.column_stack([xx.astype(float), yy.astype(float)])
    return xy_to_ml_ap_px(xy, basis)


def build_response_template_for_side(
    mask_records: list[tuple[np.ndarray, dict[str, Any], str]],
    config: dict[str, Any],
    side: str,
) -> dict[str, Any] | None:
    """Construct a side-specific, bregma-aligned response probability template.
    
    Parameters
    ----------
    mask_records : list of tuple
        Tuples of (reference-space boolean mask, AP/ML basis, detected side).
    config : dict
        ``step6.missing_response_masks`` controls bin size, radius and minimum
        number of donors.
    side : str
        Hemisphere label used to select donor masks.
    
    Returns
    -------
    dict or None
        Relative (AP, ML) probability grid and binning metadata, or None if too
        few eligible reference masks are available.
    
    Notes
    -----
    The accumulator is divided by the number of selected masks. Template
    coordinates are offsets in full-resolution pixels; templates are not
    independent proof of the response side for another group.
    """
    selected = [(m, b) for m, b, s in mask_records if str(s).lower() == str(side).lower() and m is not None and int(m.sum()) > 0]
    min_n = int(cfg_get(config, "step6", "missing_response_masks", "min_masks_per_side", default=2))
    if len(selected) < min_n:
        return None
    radius = int(cfg_get(config, "step6", "missing_response_masks", "template_radius_px", default=250))
    bin_px = int(cfg_get(config, "step6", "missing_response_masks", "template_bin_px", default=4))
    n = int(math.ceil(2 * radius / bin_px)) + 1
    acc = np.zeros((n, n), dtype=np.float32)
    center = n // 2

    positive_label = str(cfg_get(config, "step6", "side_detection", "ml_positive_label", default="right")).lower()
    keep_positive = (side == positive_label)
    for mask, basis in selected:
        pts = _mask_to_relative_ml_ap_px(mask, basis)
        if pts.size == 0:
            continue
        pts = pts[pts[:, 0] > 0] if keep_positive else pts[pts[:, 0] < 0]
        if pts.size == 0:
            continue
        ml = np.round(pts[:, 0] / bin_px).astype(int) + center
        ap = np.round(pts[:, 1] / bin_px).astype(int) + center
        valid = (ml >= 0) & (ml < n) & (ap >= 0) & (ap < n)
        acc[ap[valid], ml[valid]] += 1.0
    prob = acc / max(len(selected), 1)
    return {
        "side": str(side),
        "probability": prob,
        "bin_px": int(bin_px),
        "radius_px": int(radius),
        "n_masks": int(len(selected)),
    }


def place_response_template_on_reference(template: dict[str, Any], basis: dict[str, Any], ref_shape: tuple[int, int], config: dict[str, Any]) -> np.ndarray:
    """Rasterize a bregma-relative response template in one reference image.
    """
    prob = np.asarray(template["probability"], dtype=np.float32)
    bin_px = int(template["bin_px"])
    center = prob.shape[0] // 2
    thr = float(cfg_get(config, "step6", "missing_response_masks", "probability_threshold", default=0.5))
    ap_i, ml_i = np.where(prob >= thr)
    if ap_i.size == 0:
        return np.zeros(ref_shape, dtype=bool)
    ml_px = (ml_i - center).astype(float) * bin_px
    ap_px = (ap_i - center).astype(float) * bin_px
    xy = (
        np.asarray(basis["origin_xy"], dtype=float)
        + ml_px[:, None] * np.asarray(basis["ml_hat_xy"], dtype=float)
        + ap_px[:, None] * np.asarray(basis["ap_hat_xy"], dtype=float)
    )
    xx = np.round(xy[:, 0]).astype(int)
    yy = np.round(xy[:, 1]).astype(int)
    h, w = ref_shape
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    mask = np.zeros(ref_shape, dtype=bool)
    mask[yy[valid], xx[valid]] = True
    # Smooth tiny holes caused by binning.
    if bin_px > 1:
        mask = ndimage.binary_dilation(mask, iterations=max(1, bin_px // 2))
        mask = ndimage.binary_fill_holes(mask)
    # Enforce donor-side isolation again after dilation/filling.
    template_side = str(template.get("side", "missing")).lower()
    positive_label = str(cfg_get(config, "step6", "side_detection", "ml_positive_label", default="right")).lower()
    yy, xx = np.where(mask)
    if template_side in {"left", "right"} and yy.size:
        ml_ap = xy_to_ml_ap_px(np.column_stack([xx, yy]), basis)
        valid_side = ml_ap[:, 0] > 0 if template_side == positive_label else ml_ap[:, 0] < 0
        restricted = np.zeros_like(mask)
        restricted[yy[valid_side], xx[valid_side]] = True
        mask = restricted
    return mask.astype(bool)


# ============================================================
# Atlas loading and atlas landmarks
# ============================================================


@dataclass
class AtlasData:
    """BrainGlobe atlas volumes, dorsal projection, physical resolution and metadata.

    Notes
    -----
    Canonical volume axes are (AP, DV, ML). The 2D flat projection uses
    (AP, ML), while 3D physical coordinates use (ML, AP, DV) in µm.
    """
    atlas: Any
    reference: np.ndarray
    annotation: np.ndarray
    ann_canon: np.ndarray
    ref_canon: np.ndarray
    flat_label: np.ndarray
    flat_ref: np.ndarray
    flat_valid: np.ndarray
    flat_dv_idx: np.ndarray
    flat_ref_disp: np.ndarray
    res_ap_um: float
    res_dv_um: float
    res_ml_um: float
    id_to_acronym: dict[int, str]
    id_to_name: dict[int, str]
    structures: dict[Any, Any]
    n_ap: int
    n_dv: int
    n_ml: int


def _move_to_ap_dv_ml(vol: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Reorder a 3D atlas volume into configured (AP, DV, ML) axes.
    """
    axes = [
        int(cfg_get(config, "step6", "atlas", "axis_ap", default=0)),
        int(cfg_get(config, "step6", "atlas", "axis_dv", default=1)),
        int(cfg_get(config, "step6", "atlas", "axis_ml", default=2)),
    ]
    if sorted(axes) != [0, 1, 2]:
        raise ValueError("step6.atlas.axis_ap/dv/ml must be a permutation of 0,1,2")
    return np.moveaxis(vol, axes, [0, 1, 2])


def load_allen_atlas(config: dict[str, Any]) -> AtlasData:
    """Load a BrainGlobe Allen atlas and form its dorsal flat projection.
    
    Parameters
    ----------
    config : dict
        ``step6.atlas`` supplies the atlas name, axis order, optional atlas
        directory, axis sign conventions and dorsal surface direction.
    
    Returns
    -------
    AtlasData
        Original and canonical (AP, DV, ML) atlas volumes, dorsal-surface
        2D (AP, ML) reference and integer label arrays, valid-pixel mask,
        per-axis resolution in µm/voxel and structure metadata.
    
    Notes
    -----
    The dorsal pixel is selected as the first valid brain voxel along the DV
    axis (or from the reverse direction if configured). BrainGlobe is initialized
    with ``check_latest=False``, so this function does not request atlas updates.
    
    Raises
    ------
    ImportError
        If ``brainglobe-atlasapi`` is unavailable.
    """
    if BrainGlobeAtlas is None:
        raise ImportError(
            "brainglobe-atlasapi is required for atlas loading. "
            "Install it with: pip install brainglobe-atlasapi"
        )

    atlas_name = str(
        cfg_get(
            config,
            "step6", "atlas", "name",
            default="allen_mouse_25um",
        )
    )

    bg_dir = cfg_get(
        config,
        "step6", "atlas", "brainglobe_dir",
        default=None,
    )

    # IMPORTANT:
    # Do not contact the BrainGlobe server to check for newer atlas versions.
    atlas_kwargs = {
        "check_latest": False,
    }

    # Do not explicitly pass None because some BrainGlobe versions
    # handle omitted brainglobe_dir more reliably.
    if bg_dir is not None and str(bg_dir).strip().lower() not in {
        "", "none", "null"
    }:
        atlas_kwargs["brainglobe_dir"] = bg_dir

    atlas = BrainGlobeAtlas(
        atlas_name,
        **atlas_kwargs,
    )

    reference = np.asarray(atlas.reference)
    annotation = np.asarray(atlas.annotation)

    ann_canon = _move_to_ap_dv_ml(annotation, config)
    ref_canon = _move_to_ap_dv_ml(reference, config)

    brain_mask = ann_canon > 0
    n_ap, n_dv, n_ml = ann_canon.shape

    res = atlas.resolution
    if np.isscalar(res):
        res_ap_um = res_dv_um = res_ml_um = float(res)
    else:
        res_ap_um = float(
            res[int(cfg_get(config, "step6", "atlas", "axis_ap", default=0))]
        )
        res_dv_um = float(
            res[int(cfg_get(config, "step6", "atlas", "axis_dv", default=1))]
        )
        res_ml_um = float(
            res[int(cfg_get(config, "step6", "atlas", "axis_ml", default=2))]
        )

    flat_valid = brain_mask.any(axis=1)

    dorsal_from_low = bool(
        cfg_get(
            config,
            "step6", "atlas", "dorsal_from_low_dv",
            default=True,
        )
    )

    if dorsal_from_low:
        flat_dv_idx = np.argmax(brain_mask, axis=1)
    else:
        flat_dv_idx = (
            n_dv - 1
            - np.argmax(brain_mask[:, ::-1, :], axis=1)
        )

    ap_grid, ml_grid = np.indices((n_ap, n_ml))

    flat_label = np.zeros((n_ap, n_ml), dtype=np.int32)
    flat_ref = np.zeros((n_ap, n_ml), dtype=np.float32)

    flat_label[flat_valid] = ann_canon[
        ap_grid[flat_valid],
        flat_dv_idx[flat_valid],
        ml_grid[flat_valid],
    ].astype(np.int32)

    flat_ref[flat_valid] = ref_canon[
        ap_grid[flat_valid],
        flat_dv_idx[flat_valid],
        ml_grid[flat_valid],
    ].astype(np.float32)

    id_to_acronym: dict[int, str] = {}
    id_to_name: dict[int, str] = {}

    for sid, meta in atlas.structures.items():
        try:
            sid_int = int(sid)
        except Exception:
            continue

        id_to_acronym[sid_int] = str(
            meta.get("acronym", sid_int)
        )
        id_to_name[sid_int] = str(
            meta.get("name", sid_int)
        )

    return AtlasData(
        atlas=atlas,
        reference=reference,
        annotation=annotation,
        ann_canon=ann_canon,
        ref_canon=ref_canon,
        flat_label=flat_label,
        flat_ref=flat_ref,
        flat_valid=flat_valid,
        flat_dv_idx=flat_dv_idx,
        flat_ref_disp=normalize01(flat_ref),
        res_ap_um=res_ap_um,
        res_dv_um=res_dv_um,
        res_ml_um=res_ml_um,
        id_to_acronym=id_to_acronym,
        id_to_name=id_to_name,
        structures=atlas.structures,
        n_ap=n_ap,
        n_dv=n_dv,
        n_ml=n_ml,
    )


def ap_dv_ml_indices_to_xyz_um(ap: Any, dv: Any, ml: Any, atlas_data: AtlasData, config: dict[str, Any]) -> np.ndarray:
    """Convert canonical atlas (AP, DV, ML) voxel indices to (ML, AP, DV) µm.
    """
    ml_sign = float(cfg_get(config, "step6", "atlas", "ml_sign", default=1.0))
    ap_sign = float(cfg_get(config, "step6", "atlas", "ap_sign", default=1.0))
    dv_sign = float(cfg_get(config, "step6", "atlas", "dv_sign", default=1.0))
    x_ml_um = ml_sign * np.asarray(ml, dtype=float) * atlas_data.res_ml_um
    y_ap_um = ap_sign * np.asarray(ap, dtype=float) * atlas_data.res_ap_um
    z_dv_um = dv_sign * np.asarray(dv, dtype=float) * atlas_data.res_dv_um
    return np.stack([x_ml_um, y_ap_um, z_dv_um], axis=-1)


def _label_boundaries(label_img: np.ndarray) -> np.ndarray:
    """Compute binary boundaries between atlas region labels.
    """
    lab = np.asarray(label_img)
    b = np.zeros(lab.shape, dtype=bool)
    b[:-1, :] |= lab[:-1, :] != lab[1:, :]
    b[1:, :] |= lab[:-1, :] != lab[1:, :]
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    return b & (lab > 0)


def open_atlas_landmark_napari_viewer(atlas_data: AtlasData, paths: dict[str, Path], config: dict[str, Any]) -> dict[str, Any]:
    """Create a Napari session for bregma and rostral atlas annotation.
    
    Parameters
    ----------
    atlas_data : AtlasData
        Loaded atlas with its dorsal flat projection.
    paths : dict
        Contains the atlas landmarks CSV location.
    config : dict
        Project configuration retained in the returned session.
    
    Returns
    -------
    dict
        Viewer, named point layers, atlas data and paths. The caller must run
        the Napari event loop and explicitly save the completed session.
    
    Notes
    -----
    Existing landmark coordinates are loaded into the point layers if a saved
    CSV is present. Opening and closing the viewer alone does not write the CSV.
    """
    if napari is None:
        raise RuntimeError("napari is not available")
    lm_csv = Path(paths["atlas_landmarks_csv"])
    existing = pd.read_csv(lm_csv) if lm_csv.exists() else pd.DataFrame()
    existing_yx: dict[str, tuple[float, float]] = {}
    if {"name", "ap_idx", "ml_idx"}.issubset(existing.columns):
        for _, r in existing.iterrows():
            existing_yx[str(r["name"])] = (float(r["ap_idx"]), float(r["ml_idx"]))

    viewer = napari.Viewer(title="Allen atlas landmarks: bregma + rostral")
    viewer.add_image(atlas_data.flat_ref_disp, name="dorsal_reference", colormap="gray")
    viewer.add_labels(atlas_data.flat_label.astype(np.int32), name="atlas_labels", opacity=0.25)
    viewer.add_image(_label_boundaries(atlas_data.flat_label).astype(np.float32), name="label_boundaries", opacity=0.6, blending="additive")
    layers = {}
    colors = {"bregma": "red", "rostral": "green"}
    for name in ["bregma", "rostral"]:
        layer = viewer.add_points(name=f"{name}_atlas", ndim=2, size=8, face_color=colors.get(name, "white"))
        if name in existing_yx:
            layer.data = np.array([existing_yx[name]], dtype=float)
        layers[name] = layer
    print("Place one point in each layer: bregma_atlas and rostral_atlas. Then close napari.")
    return {"viewer": viewer, "layers": layers, "atlas_data": atlas_data, "paths": paths, "config": config}


def _nearest_valid_flat_ap_ml(atlas_data: AtlasData, ap_i: int, ml_i: int) -> tuple[int, int, float]:
    """Snap an atlas (AP, ML) pixel to the nearest valid dorsal position.
    """
    ap_i = int(np.clip(ap_i, 0, atlas_data.n_ap - 1))
    ml_i = int(np.clip(ml_i, 0, atlas_data.n_ml - 1))
    if atlas_data.flat_valid[ap_i, ml_i]:
        return ap_i, ml_i, 0.0
    valid_ap_ml = np.column_stack(np.where(atlas_data.flat_valid))
    kd = cKDTree(valid_ap_ml.astype(float))
    dist, j = kd.query(np.array([[ap_i, ml_i]], dtype=float), k=1)
    ap2, ml2 = valid_ap_ml[int(j[0])]
    return int(ap2), int(ml2), float(dist[0])


def save_atlas_landmarks_from_napari_session(session: dict[str, Any]) -> pd.DataFrame:
    """Validate and save atlas landmarks selected in a Napari session.
    
    Parameters
    ----------
    session : dict
        Session created by ``open_atlas_landmark_napari_viewer`` after user edits.
    
    Returns
    -------
    pandas.DataFrame
        Bregma and rostral atlas landmarks, their canonical voxel indices,
        (ML, AP, DV) coordinates in µm, and snapping diagnostics.
    
    Notes
    -----
    The saved AP positions are projected onto a single medial ML index selected
    by the function and snapped to valid atlas pixels. The output replaces the
    configured atlas landmarks CSV.
    """
    atlas_data: AtlasData = session["atlas_data"]
    paths: dict[str, Path] = session["paths"]
    config: dict[str, Any] = session["config"]
    layers = session["layers"]

    rows = []
    clicked: dict[str, tuple[int, int]] = {}
    for name, layer in layers.items():
        pts = np.asarray(layer.data, dtype=float)
        if len(pts) != 1:
            raise RuntimeError(f"Expected one point in layer {name}_atlas, got {len(pts)}")
        ap_click = int(round(float(pts[0, 0])))
        ml_click = int(round(float(pts[0, 1])))
        clicked[name] = (ap_click, ml_click)

    # One AP line in atlas flat projection = same ML coordinate for all landmarks.
    midline_ml_idx = int(clicked["bregma"][1])
    for name in ["bregma", "rostral"]:
        ap_click, _ = clicked[name]
        ap_use, ml_use, snap_dist = _nearest_valid_flat_ap_ml(atlas_data, ap_click, midline_ml_idx)
        dv_use = int(atlas_data.flat_dv_idx[ap_use, ml_use])
        label_id = int(atlas_data.flat_label[ap_use, ml_use])
        xyz_um = ap_dv_ml_indices_to_xyz_um(ap_use, dv_use, ml_use, atlas_data, config)
        rows.append({
            "name": name,
            "ap_idx": int(ap_use),
            "dv_idx": int(dv_use),
            "ml_idx": int(ml_use),
            "x_ml_um": float(xyz_um[0]),
            "y_ap_um": float(xyz_um[1]),
            "z_dv_um": float(xyz_um[2]),
            "label_id": label_id,
            "acronym": atlas_data.id_to_acronym.get(label_id, str(label_id)),
            "clicked_ap_idx": int(ap_click),
            "clicked_ml_idx": int(clicked[name][1]),
            "fixed_ml_idx_for_one_AP_line": int(midline_ml_idx),
            "snap_dist_px": float(snap_dist),
            "updated_at": _dt.datetime.now().isoformat(),
        })
    df = pd.DataFrame(rows)
    out = Path(paths["atlas_landmarks_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def load_atlas_landmarks(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Read the saved bregma and rostral atlas positions in (ML, AP, DV) µm.
    """
    path = Path(paths["atlas_landmarks_csv"])
    if not path.exists():
        raise FileNotFoundError(f"Atlas landmarks CSV not found: {path}. Run atlas landmark annotation first.")
    df = pd.read_csv(path)
    out = {}
    for _, r in df.iterrows():
        out[str(r["name"])] = np.array([float(r["x_ml_um"]), float(r["y_ap_um"]), float(r["z_dv_um"])], dtype=float)
    for name in ["bregma", "rostral"]:
        if name not in out:
            raise RuntimeError(f"Atlas landmark {name!r} not found in {path}")
    return out


# ============================================================
# Atlas structures / response anchor
# ============================================================


def _structure_path_ids(meta: dict[str, Any]) -> list[int]:
    """Extract integer structure ancestry IDs from BrainGlobe metadata.
    """
    path = None
    for key in ("structure_id_path", "structure_id_path_ids", "id_path", "path"):
        if key in meta:
            path = meta.get(key)
            break
    if path is None:
        return []
    if isinstance(path, str):
        return [int(x) for x in re.findall(r"\d+", path)]
    if isinstance(path, (list, tuple, np.ndarray)):
        out = []
        for x in path:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    return []


def find_structure_ids(atlas_data: AtlasData, queries: Sequence[str], *, include_descendants: bool = True) -> set[int]:
    """Resolve atlas structure names/acronyms and optional descendant IDs.
    """
    q_lower = [str(q).lower() for q in queries]
    exact: list[int] = []
    fallback: list[int] = []
    for sid_key, meta in atlas_data.structures.items():
        try:
            sid = int(sid_key)
        except Exception:
            continue
        acr = str(meta.get("acronym", "")).lower()
        name = str(meta.get("name", "")).lower()
        for q in q_lower:
            if q == acr or q == name:
                exact.append(sid)
            elif q in acr or q in name:
                fallback.append(sid)
    roots = set(exact if exact else fallback)
    if not include_descendants:
        return roots
    descendants = set(roots)
    for sid_key, meta in atlas_data.structures.items():
        try:
            sid = int(sid_key)
        except Exception:
            continue
        path = set(_structure_path_ids(meta))
        if sid in roots or (roots and len(path & roots) > 0):
            descendants.add(sid)
    return descendants


def atlas_structure_center_xyz_um(atlas_data: AtlasData, config: dict[str, Any], side: str, structure_queries: Sequence[str] | None = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Find the mean 3D atlas coordinate of a selected structure and side.
    """
    if structure_queries is None:
        structure_queries = cfg_get(config, "step6", "atlas", "sshl_structure_queries", default=[])
    ids = find_structure_ids(
        atlas_data,
        structure_queries,
        include_descendants=bool(cfg_get(config, "step6", "atlas", "include_descendants", default=True)),
    )
    if not ids:
        return None, {"atlas_response_anchor_status": "structure_not_found", "queries": json.dumps(list(structure_queries), ensure_ascii=False)}

    mask = np.isin(atlas_data.flat_label, np.array(sorted(ids), dtype=np.int32))
    if not mask.any():
        return None, {"atlas_response_anchor_status": "structure_not_in_flat_projection", "structure_ids": json.dumps(sorted(ids))}

    # Use bregma landmark ML index if available in the landmarks CSV via config cache. If not, midline = middle of atlas.
    bregma_ml_idx = None
    if "atlas_landmark_df" in config.get("_step6_cache", {}):
        df = config["_step6_cache"]["atlas_landmark_df"]
        b = df[df["name"].astype(str).eq("bregma")]
        if not b.empty:
            bregma_ml_idx = int(b.iloc[-1]["ml_idx"])
    if bregma_ml_idx is None:
        bregma_ml_idx = atlas_data.n_ml // 2

    ap_idx, ml_idx = np.where(mask)
    left_rel = str(cfg_get(config, "step6", "atlas", "left_ml_relation_to_bregma", default="negative")).strip().lower()
    side_l = str(side).strip().lower()
    if side_l in {"left", "l"}:
        side_mask = ml_idx < bregma_ml_idx if left_rel == "negative" else ml_idx > bregma_ml_idx
    elif side_l in {"right", "r"}:
        side_mask = ml_idx > bregma_ml_idx if left_rel == "negative" else ml_idx < bregma_ml_idx
    else:
        side_mask = np.ones_like(ml_idx, dtype=bool)
    ap_idx = ap_idx[side_mask]
    ml_idx = ml_idx[side_mask]
    if ap_idx.size == 0:
        return None, {"atlas_response_anchor_status": "structure_side_empty", "structure_ids": json.dumps(sorted(ids)), "side": side}

    dv_idx = atlas_data.flat_dv_idx[ap_idx, ml_idx]
    xyz = ap_dv_ml_indices_to_xyz_um(ap_idx, dv_idx, ml_idx, atlas_data, config)
    center = np.nanmean(xyz, axis=0)
    return center.astype(float), {
        "atlas_response_anchor_status": "ok",
        "structure_ids": json.dumps(sorted(int(x) for x in ids)),
        "structure_queries": json.dumps(list(structure_queries), ensure_ascii=False),
        "structure_n_flat_pixels": int(ap_idx.size),
        "side": str(side),
    }


def atlas_structure_flat_points_xyz_um(
    atlas_data: AtlasData,
    config: dict[str, Any],
    side: str,
    structure_queries: Sequence[str] | None = None,
    *,
    max_points: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample dorsal-flat atlas structure points in (ML, AP, DV) µm.
    
    Parameters
    ----------
    atlas_data : AtlasData
        BrainGlobe atlas with canonical dorsal-flat labels.
    config : dict
        ``step6.atlas`` structure and hemisphere mapping options.
    side : str
        Selected hemisphere label.
    structure_queries : sequence of str or None, optional
        Named atlas regions; defaults are read from the configuration.
    max_points : int, optional
        Upper bound on returned sampled points.
    
    Returns
    -------
    numpy.ndarray
        Two-dimensional array shaped (n_points, 3), with columns
        (ML, AP, DV) in µm.
    """
    if structure_queries is None:
        structure_queries = cfg_get(config, "step6", "atlas", "sshl_structure_queries", default=[])
    ids = find_structure_ids(
        atlas_data,
        structure_queries,
        include_descendants=bool(cfg_get(config, "step6", "atlas", "include_descendants", default=True)),
    )
    if not ids:
        return np.empty((0, 3), dtype=float), {"sshl_match_points_status": "structure_not_found"}
    mask = np.isin(atlas_data.flat_label, np.array(sorted(ids), dtype=np.int32))
    if not mask.any():
        return np.empty((0, 3), dtype=float), {"sshl_match_points_status": "structure_not_in_flat_projection", "structure_ids": json.dumps(sorted(ids))}

    bregma_ml_idx = None
    if "atlas_landmark_df" in config.get("_step6_cache", {}):
        df = config["_step6_cache"]["atlas_landmark_df"]
        b = df[df["name"].astype(str).eq("bregma")]
        if not b.empty:
            bregma_ml_idx = int(b.iloc[-1]["ml_idx"])
    if bregma_ml_idx is None:
        bregma_ml_idx = atlas_data.n_ml // 2

    ap_idx, ml_idx = np.where(mask)
    left_rel = str(cfg_get(config, "step6", "atlas", "left_ml_relation_to_bregma", default="negative")).strip().lower()
    side_l = str(side).strip().lower()
    if side_l in {"left", "l"}:
        side_mask = ml_idx < bregma_ml_idx if left_rel == "negative" else ml_idx > bregma_ml_idx
    elif side_l in {"right", "r"}:
        side_mask = ml_idx > bregma_ml_idx if left_rel == "negative" else ml_idx < bregma_ml_idx
    else:
        side_mask = np.ones_like(ml_idx, dtype=bool)
    ap_idx = ap_idx[side_mask]
    ml_idx = ml_idx[side_mask]
    if ap_idx.size == 0:
        return np.empty((0, 3), dtype=float), {"sshl_match_points_status": "structure_side_empty", "structure_ids": json.dumps(sorted(ids)), "side": side}

    max_points_cfg = cfg_get(config, "step6", "projection", "response_match_points_max", default=250)
    if max_points is None:
        max_points = int(max_points_cfg if max_points_cfg is not None else 250)
    if max_points > 0 and ap_idx.size > max_points:
        # Deterministic evenly spaced subsample, stable across runs.
        take = np.linspace(0, ap_idx.size - 1, int(max_points)).round().astype(int)
        ap_idx = ap_idx[take]
        ml_idx = ml_idx[take]

    dv_idx = atlas_data.flat_dv_idx[ap_idx, ml_idx]
    xyz = ap_dv_ml_indices_to_xyz_um(ap_idx, dv_idx, ml_idx, atlas_data, config)
    return xyz.astype(float), {
        "sshl_match_points_status": "ok",
        "sshl_match_points_n": int(xyz.shape[0]),
        "structure_ids": json.dumps(sorted(int(x) for x in ids)),
        "side": str(side),
    }


def project_atlas_xyz_to_reference_xy(xyz_um: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    """Project atlas coordinates in (ML, AP, DV) µm to reference (x, y) pixels.
    """
    xyz = np.asarray(xyz_um, dtype=float)
    if xyz.size == 0:
        return np.empty((0, 2), dtype=float)
    uv, _ = pitch_project_xyz_um(xyz, float(transform["pitch_final_deg"]), np.asarray(transform["pivot_xyz_um"], dtype=float))
    basis = {
        "origin_xy": np.asarray(transform["origin_xy"], dtype=float),
        "ap_hat_xy": np.asarray(transform["ap_hat_xy"], dtype=float),
        "ml_hat_xy": np.asarray(transform["ml_hat_xy"], dtype=float),
        "pixel_size_um": float(transform["pixel_size_um"]),
    }
    return atlas_uv_um_to_reference_xy_px(uv, float(transform["scale_final"]), basis)


# ============================================================
# Pitch/scale fit and projection
# ============================================================


def pitch_project_xyz_um(xyz_um: np.ndarray, pitch_deg: float, pivot_xyz_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a pitch rotation about atlas bregma and return projected ML/AP in µm.
    
    Parameters
    ----------
    xyz_um : numpy.ndarray
        Atlas points with final coordinate axis (ML, AP, DV) in µm.
    pitch_deg : float
        Pitch angle in degrees.
    pivot_xyz_um : numpy.ndarray
        Rotation pivot, usually atlas bregma, in µm.
    
    Returns
    -------
    uv_um : numpy.ndarray
        Projected (ML, AP) values in µm.
    xyz_rot_um : numpy.ndarray
        Rotated 3D atlas coordinates in µm.
    """
    xyz = np.asarray(xyz_um, dtype=float)
    pivot = np.asarray(pivot_xyz_um, dtype=float)
    rel = xyz - pivot
    th = np.deg2rad(float(pitch_deg))
    c, s = np.cos(th), np.sin(th)
    ml_rot = rel[..., 0]
    ap_rot = rel[..., 1] * c - rel[..., 2] * s
    dv_rot = rel[..., 1] * s + rel[..., 2] * c
    uv_um = np.stack([ml_rot, ap_rot], axis=-1)
    return uv_um, dv_rot


def atlas_uv_um_to_reference_xy_px(uv_um: np.ndarray, scale_factor: float, basis: dict[str, Any]) -> np.ndarray:
    """Scale projected atlas (ML, AP) µm into reference (x, y) pixels.
    """
    uv = np.asarray(uv_um, dtype=float)
    ml_um = uv[..., 0]
    ap_um = uv[..., 1]
    pixel_size = float(basis["pixel_size_um"])
    xy = (
        np.asarray(basis["origin_xy"], dtype=float)
        + (float(scale_factor) / pixel_size)
        * (ml_um[..., None] * np.asarray(basis["ml_hat_xy"], dtype=float) + ap_um[..., None] * np.asarray(basis["ap_hat_xy"], dtype=float))
    )
    return xy


def fit_pitch_scale(
    basis: dict[str, Any],
    atlas_landmarks_xyz_um: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    response_ml_ap_um: np.ndarray | None = None,
    atlas_response_xyz_um: np.ndarray | None = None,
    response_anchor_status: str = "missing",
    response_weight_multiplier: float = 1.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit atlas pitch and scale from anatomical and optional response anchors.
    
    Parameters
    ----------
    basis : dict
        Step3 reference AP/ML basis, including observed rostral AP offset in µm.
    atlas_landmarks_xyz_um : dict of numpy.ndarray
        Atlas bregma and rostral points in (ML, AP, DV) µm.
    config : dict
        Optimization bounds, priors, uncertainties and anchor weights are read
        from ``step6.fit``.
    response_ml_ap_um : numpy.ndarray or None, optional
        Observed response center in reference (ML, AP) µm.
    atlas_response_xyz_um : numpy.ndarray or None, optional
        Matching atlas response point in 3D (ML, AP, DV) µm.
    response_anchor_status : str, optional
        Indicates whether the optional response pair is valid for fitting.
    response_weight_multiplier : float, optional
        Additional weight of the response residual terms.
    
    Returns
    -------
    fit : dict
        Pitch in degrees, dimensionless scale, optimizer status, residual
        diagnostics in µm and fit settings for provenance.
    fit_table : pandas.DataFrame
        Observed, predicted and residual landmark positions in µm.
    
    Notes
    -----
    Stage 1 performs bounded one-dimensional pitch optimization at ``scale0``.
    If enabled, stage 2 jointly fits pitch and scale using bounded L-BFGS-B,
    including the scale prior and a penalty for moving away from stage-1 pitch.
    The weighted objective uses rostral AP and, only with a valid response pair,
    response ML/AP offsets. Lambda is not part of this fit.
    """
    pivot = np.asarray(atlas_landmarks_xyz_um["bregma"], dtype=float)
    rostral_xyz = np.asarray(atlas_landmarks_xyz_um["rostral"], dtype=float)
    obs_rostral_ap = float(basis["rostral_ap_um"])

    pitch_bounds = tuple(float(x) for x in cfg_get(config, "step6", "fit", "pitch_bounds_deg", default=[-60, 60]))
    scale0 = float(cfg_get(config, "step6", "fit", "scale0", default=1.0))
    scale_bounds = tuple(float(x) for x in cfg_get(config, "step6", "fit", "scale_bounds", default=[0.8, 1.5]))
    scale_prior = float(cfg_get(config, "step6", "fit", "scale_prior", default=1.0))
    scale_prior_sigma = float(cfg_get(config, "step6", "fit", "scale_prior_sigma", default=0.25))
    pitch_local_window = float(cfg_get(config, "step6", "fit", "pitch_local_window_deg", default=4.0))
    pitch_anchor_sigma = float(cfg_get(config, "step6", "fit", "pitch_anchor_sigma_deg", default=2.0))
    lm_sigma = float(cfg_get(config, "step6", "fit", "landmark_sigma_um", default=50.0))
    resp_sigma = float(cfg_get(config, "step6", "fit", "response_sigma_um", default=150.0))
    weights = cfg_get(config, "step6", "fit", "weights", default={}) or {}

    has_response = response_ml_ap_um is not None and atlas_response_xyz_um is not None and str(response_anchor_status).lower() in {"ok", "imputed", "existing", "registered_to_ref", "ref_stem_response", "same_group_registered_response", "imputed_population_template"}
    response_weight_multiplier = float(response_weight_multiplier) if np.isfinite(float(response_weight_multiplier)) else 1.0

    def pred_rostral_ap(pitch: float, scale: float) -> float:
        """Predict the rostral landmark's AP offset in µm for pitch and scale.
        """
        uv, _ = pitch_project_xyz_um(rostral_xyz, pitch, pivot)
        return float(scale * uv[1])

    def pred_response_ml_ap(pitch: float, scale: float) -> np.ndarray | None:
        """Predict response-center (ML, AP) in µm for an eligible anchor.
        """
        if not has_response:
            return None
        uv, _ = pitch_project_xyz_um(np.asarray(atlas_response_xyz_um, dtype=float), pitch, pivot)
        return np.asarray(scale * uv, dtype=float)

    def core_loss(pitch: float, scale: float) -> float:
        """Evaluate the weighted rostral and optional response-anchor squared residuals.
        """
        loss = 0.0
        r_ap = pred_rostral_ap(pitch, scale) - obs_rostral_ap
        loss += float(weights.get("rostral_ap", 1.0)) * (r_ap / lm_sigma) ** 2
        if has_response:
            pred = pred_response_ml_ap(pitch, scale)
            obs = np.asarray(response_ml_ap_um, dtype=float)
            r = pred - obs
            loss += response_weight_multiplier * float(weights.get("response_ml", 1.0)) * (r[0] / resp_sigma) ** 2
            loss += response_weight_multiplier * float(weights.get("response_ap", 1.0)) * (r[1] / resp_sigma) ** 2
        return float(loss)

    def stage1_loss(pitch: float) -> float:
        """Evaluate the pitch-only fitting objective with scale fixed to scale0.
        """
        return core_loss(float(pitch), scale0)

    stage1 = minimize_scalar(stage1_loss, bounds=pitch_bounds, method="bounded")
    pitch1 = float(stage1.x)
    rmse1 = abs(pred_rostral_ap(pitch1, scale0) - obs_rostral_ap)

    allow_scale = bool(cfg_get(config, "step6", "fit", "allow_scale_after_pitch", default=True))
    if allow_scale:
        lo = max(pitch_bounds[0], pitch1 - pitch_local_window)
        hi = min(pitch_bounds[1], pitch1 + pitch_local_window)

        def final_loss(v: np.ndarray) -> float:
            """Evaluate the joint pitch/scale objective including regularization priors.
            """
            pitch, scale = float(v[0]), float(v[1])
            loss = core_loss(pitch, scale)
            loss += float(weights.get("scale_prior", 1.0)) * ((scale - scale_prior) / max(scale_prior_sigma, 1e-12)) ** 2
            loss += ((pitch - pitch1) / max(pitch_anchor_sigma, 1e-12)) ** 2
            return float(loss)

        # Match the tested notebook: bounded L-BFGS-B, not unconstrained/simplex
        # optimization. This prevents accidental excursions to implausible pitch
        # values when the response anchor is noisy or side mapping is wrong.
        opt = minimize(
            final_loss,
            x0=np.array([pitch1, scale0], dtype=float),
            bounds=[(lo, hi), scale_bounds],
            method="L-BFGS-B",
            options={"maxiter": 500},
        )
        pitch_final = float(opt.x[0])
        scale_final = float(opt.x[1])
        success = bool(opt.success)
        message = str(opt.message)
    else:
        pitch_final = pitch1
        scale_final = scale0
        success = bool(stage1.success)
        message = str(stage1.message)

    rows = []
    pred_rostral = pred_rostral_ap(pitch_final, scale_final)
    rows.append({
        "anchor": "rostral_ap",
        "observed_um": obs_rostral_ap,
        "predicted_um": pred_rostral,
        "residual_um": pred_rostral - obs_rostral_ap,
        "used": True,
    })
    if has_response:
        pred = pred_response_ml_ap(pitch_final, scale_final)
        obs = np.asarray(response_ml_ap_um, dtype=float)
        rows.extend([
            {"anchor": "response_ml", "observed_um": float(obs[0]), "predicted_um": float(pred[0]), "residual_um": float(pred[0] - obs[0]), "used": True},
            {"anchor": "response_ap", "observed_um": float(obs[1]), "predicted_um": float(pred[1]), "residual_um": float(pred[1] - obs[1]), "used": True},
        ])
    fit_table = pd.DataFrame(rows)
    rmse_final = float(np.sqrt(np.mean(fit_table.loc[fit_table["used"], "residual_um"].astype(float).values ** 2)))

    fit = {
        "created_at": _dt.datetime.now().isoformat(),
        "model": "mouse reference AP-axis basis; Allen bregma/rostral; pitch around ML; scale after pitch; optional response anchor",
        "landmarks_used": ["bregma", "rostral"] + (["response_center"] if has_response else []),
        "response_anchor_used": bool(has_response),
        "response_anchor_status": str(response_anchor_status),
        "pitch_stage1_deg": float(pitch1),
        "scale_stage1": float(scale0),
        "rmse_stage1_um": float(rmse1),
        "pitch_final_deg": float(pitch_final),
        "scale_final": float(scale_final),
        "rmse_final_um": rmse_final,
        "scale_change_percent": float((scale_final / scale0 - 1.0) * 100.0),
        "pitch_change_from_stage1_deg": float(pitch_final - pitch1),
        "success": bool(success),
        "message": message,
        "pivot_landmark": "bregma",
        "pivot_xyz_um": pivot.tolist(),
        "fit_settings": {
            "pitch_bounds_deg": list(pitch_bounds),
            "scale_bounds": list(scale_bounds),
            "scale_prior": scale_prior,
            "scale_prior_sigma": scale_prior_sigma,
            "pitch_local_window_deg": pitch_local_window,
            "pitch_anchor_sigma_deg": pitch_anchor_sigma,
            "landmark_sigma_um": lm_sigma,
            "response_sigma_um": resp_sigma,
            "weights": weights,
            "response_weight_multiplier": float(response_weight_multiplier),
        },
    }
    return fit, fit_table


def transform_json_from_fit(row: pd.Series, basis: dict[str, Any], fit: dict[str, Any], fit_table: pd.DataFrame, config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the fitted atlas transform and provenance into a JSON-safe mapping.
    """
    out = {
        "group_id": str(row.get("group_id", "")),
        "ref_stem": str(row.get("ref_stem", "")),
        "reference_policy": cfg_get(config, "step6", "reference", "policy", default="step3_ref_stem"),
        "reference_assumption": cfg_get(config, "step6", "reference", "assumption", default="ref_stem_is_stimulation_recording"),
        "target_space": cfg_get(config, "step6", "input", "target_space", default="step3_reference_rotated_fullres"),
        "pixel_size_um": float(basis["pixel_size_um"]),
        "reference_image_path": str(row.get("ref_image_path", "")),
        "reference_shape": None,
        "bregma_xy": np.asarray(basis["bregma_xy"], dtype=float).tolist(),
        "bregma_projected_xy": np.asarray(basis["bregma_proj_xy"], dtype=float).tolist(),
        "origin_xy": np.asarray(basis["origin_xy"], dtype=float).tolist(),
        "rostral_xy": np.asarray(basis["rostral_xy"], dtype=float).tolist(),
        "rostral_projected_xy": np.asarray(basis["rostral_proj_xy"], dtype=float).tolist(),
        "rostral_source": str(basis.get("rostral_source", "")),
        "ap_positive_direction": str(basis.get("ap_positive_direction", "")),
        "rostral_expected_ap_sign": str(basis.get("rostral_expected_ap_sign", "")),
        "rostral_ap_um": float(basis["rostral_ap_um"]),
        "ap_hat_xy": np.asarray(basis["ap_hat_xy"], dtype=float).tolist(),
        "ml_hat_xy": np.asarray(basis["ml_hat_xy"], dtype=float).tolist(),
        "atlas_name": cfg_get(config, "step6", "atlas", "name", default="allen_mouse_25um"),
        **fit,
        "fit_residuals": fit_table.to_dict(orient="records"),
    }
    if extra:
        out.update(extra)
    return out


# ============================================================
# Atlas label-map projection and stem-space label maps
# ============================================================


def project_atlas_flat_labels_to_reference(
    atlas_data: AtlasData,
    transform: dict[str, Any],
    ref_shape: tuple[int, int],
    config: dict[str, Any],
) -> np.ndarray:
    """Rasterize the dorsal atlas labels into the Step3 reference image.
    
    Parameters
    ----------
    atlas_data : AtlasData
        Dorsal (AP, ML) atlas labels and associated surface depths.
    transform : dict
        Fitted atlas pitch, scale, pivot and reference AP/ML basis.
    ref_shape : tuple of int
        Reference output shape (height, width), in full-resolution pixels.
    config : dict
        Projection stride and gap-fill distance from ``step6.projection``.
    
    Returns
    -------
    numpy.ndarray
        Integer atlas region IDs of shape ``ref_shape``; zero indicates no label.
    
    Notes
    -----
    Projects sampled atlas surface voxels and assigns labels at rounded pixel
    positions. Optional nearest-seed filling is bounded by
    ``step6.projection.fill_label_max_dist_px`` (pixels).
    """
    stride = int(cfg_get(config, "step6", "projection", "raster_stride", default=1))
    stride = max(1, stride)
    ap_idx, ml_idx = np.where(atlas_data.flat_label > 0)
    if stride > 1:
        keep = (ap_idx % stride == 0) & (ml_idx % stride == 0)
        ap_idx = ap_idx[keep]
        ml_idx = ml_idx[keep]
    dv_idx = atlas_data.flat_dv_idx[ap_idx, ml_idx]
    labels = atlas_data.flat_label[ap_idx, ml_idx].astype(np.int32)
    xyz = ap_dv_ml_indices_to_xyz_um(ap_idx, dv_idx, ml_idx, atlas_data, config)
    uv, _ = pitch_project_xyz_um(xyz, float(transform["pitch_final_deg"]), np.asarray(transform["pivot_xyz_um"], dtype=float))
    basis = {
        "origin_xy": np.asarray(transform["origin_xy"], dtype=float),
        "ap_hat_xy": np.asarray(transform["ap_hat_xy"], dtype=float),
        "ml_hat_xy": np.asarray(transform["ml_hat_xy"], dtype=float),
        "pixel_size_um": float(transform["pixel_size_um"]),
    }
    xy = atlas_uv_um_to_reference_xy_px(uv, float(transform["scale_final"]), basis)
    xx = np.round(xy[:, 0]).astype(int)
    yy = np.round(xy[:, 1]).astype(int)
    h, w = int(ref_shape[0]), int(ref_shape[1])
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    labelmap = np.zeros((h, w), dtype=np.int32)
    labelmap[yy[valid], xx[valid]] = labels[valid]

    fill_max = float(cfg_get(config, "step6", "projection", "fill_label_max_dist_px", default=20.0))
    if fill_max > 0 and np.any(labelmap > 0):
        seeds = labelmap > 0
        dist, inds = ndimage.distance_transform_edt(~seeds, return_indices=True)
        filled = labelmap[inds[0], inds[1]]
        filled[dist > fill_max] = 0
        labelmap = filled.astype(np.int32)
    return labelmap.astype(np.int32)


def atlas_label_table_from_labelmap(labelmap: np.ndarray, atlas_data: AtlasData) -> pd.DataFrame:
    """Build a region-ID/acronym/name table for labels present in an image.
    """
    ids = sorted(int(x) for x in np.unique(labelmap) if int(x) > 0)
    return pd.DataFrame([
        {
            "atlas_structure_id": sid,
            "acronym": atlas_data.id_to_acronym.get(sid, str(sid)),
            "name": atlas_data.id_to_name.get(sid, str(sid)),
        }
        for sid in ids
    ])


def save_reference_labelmap_outputs(
    labelmap_ref: np.ndarray,
    atlas_data: AtlasData,
    row: pd.Series,
    paths: dict[str, Path],
) -> tuple[Path, Path]:
    """Write the reference atlas label map and corresponding region table.
    """
    key = f"{row.get('group_id', '')}__{row.get('ref_stem', '')}".replace("/", "-").replace("\\", "-")
    out = paths["labelmaps_refspace_root"] / f"{key}_atlas_labels_refspace.npy"
    table_path = paths["labelmaps_refspace_root"] / f"{key}_atlas_label_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, labelmap_ref.astype(np.int32))
    atlas_label_table_from_labelmap(labelmap_ref, atlas_data).to_csv(table_path, index=False)
    return out, table_path


def labelmap_reference_to_stem_fullres(labelmap_ref: np.ndarray, row: pd.Series, stem_shape_fullres: tuple[int, int]) -> np.ndarray:
    """Inverse-warp reference atlas labels into a stem's full-resolution pixels.
    """
    if str(row.get("stem", "")) == str(row.get("ref_stem", "")):
        return resize_label_nearest(labelmap_ref, stem_shape_fullres)
    tform = registration_tform_stem_to_ref(row)
    warped = warp(
        labelmap_ref.astype(np.float32),
        inverse_map=tform,
        output_shape=tuple(stem_shape_fullres),
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0,
    )
    return np.rint(warped).astype(np.int32)


def processed_zarr_image_shape(row: pd.Series) -> tuple[int, int] | None:
    """Read a Step4 processed Zarr spatial (y, x) shape if available.
    """
    if xr is None:
        return None
    p = Path(str(row.get("processed_zarr_path_resolved", "")))
    if not p.exists():
        return None
    try:
        ds = xr.open_zarr(p, chunks=None, consolidated=False)
        for var in ds.data_vars:
            da = ds[var]
            if "y" in da.dims and "x" in da.dims:
                return int(da.sizes["y"]), int(da.sizes["x"])
    except Exception:
        return None
    return None


def load_canonical_brain_signal_mask(row: pd.Series, config: dict[str, Any], paths: dict[str, Path], image_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, str]]:
    """Construct the usable brain-signal mask for a stem.
    
    Parameters
    ----------
    row : pandas.Series
        Stem identifier and canonical mask locations.
    config : dict
        Mask resizing follows ``signal_processing.masks.resize_to_raw_if_needed``.
    paths : dict
        Paths to canonical Step2/Step3 masks.
    image_shape : tuple of int
        Requested output (height, width) in pixels.
    
    Returns
    -------
    brain_signal : numpy.ndarray
        Boolean mask in the requested image shape.
    mask_sources : dict
        Paths and provenance of the brain, bad and bregma masks.
    
    Notes
    -----
    The expression is ``brain & ~bad & ~bregma``. Existing manual brain and
    bregma masks take precedence over automatic versions. Missing optional masks
    are replaced by zero masks according to the existing implementation.
    """
    cps = canonical_mask_paths(paths, str(row["stem"]))
    resize_if_needed = bool(cfg_get(config, "signal_processing", "masks", "resize_to_raw_if_needed", default=True))

    def load_optional(path: Path, default: np.ndarray | None = None) -> np.ndarray:
        """Load an optional canonical mask, using the documented fallback if missing.
        """
        if path.exists():
            arr = np.load(path).astype(bool)
        elif default is not None:
            arr = default.astype(bool)
        else:
            arr = np.zeros(image_shape, dtype=bool)
        if tuple(arr.shape) != tuple(image_shape):
            if not resize_if_needed:
                raise ValueError(f"Mask shape mismatch for {path}: {arr.shape} vs {image_shape}")
            arr = resize_mask_nearest(arr, image_shape)
        return arr.astype(bool)

    brain_path = cps["brain_manual"] if cps["brain_manual"].exists() else cps["brain_auto"]
    if not brain_path.exists():
        raise FileNotFoundError(f"Brain mask not found for stem={row.get('stem', '')}: {brain_path}")
    brain = load_optional(brain_path)
    bad = load_optional(cps["bad_auto"], default=np.zeros_like(brain))
    bregma_path = cps["bregma_manual"] if cps["bregma_manual"].exists() else cps["bregma_auto"]
    bregma = load_optional(bregma_path, default=np.zeros_like(brain))
    brain_signal = brain & (~bad) & (~bregma)
    return brain_signal.astype(bool), {
        "brain_path": str(brain_path),
        "brain_source": "manual" if brain_path.name == "brain_manual.npy" else "auto",
        "bad_path": str(cps["bad_auto"]),
        "bregma_path": str(bregma_path),
        "bregma_source": "manual" if bregma_path.name == "bregma_manual.npy" else "auto",
        "mask_formula": "brain_signal = brain & ~bad & ~bregma",
    }


def make_stem_labelmaps_and_coverage(
    row: pd.Series,
    labelmap_ref: np.ndarray,
    atlas_table: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    reference_shape: tuple[int, int],
) -> dict[str, Any]:
    """Transfer atlas labels to one stem and compute regional signal coverage.
    
    Parameters
    ----------
    row : pandas.Series
        Stem registration metadata and processed-data paths.
    labelmap_ref : numpy.ndarray
        Integer atlas labels in the full-resolution reference (y, x) grid.
    atlas_table : pandas.DataFrame
        Reference atlas structure names and IDs.
    config : dict
        ``raw.scale`` and canonical-mask resizing settings are used if needed.
    paths : dict
        Output directories for stem labels and coverage tables.
    reference_shape : tuple of int
        Full-resolution reference image (height, width).
    
    Returns
    -------
    dict
        Paths to saved processed-space atlas maps and coverage CSV, processed
        image shape and canonical mask provenance.
    
    Notes
    -----
    Inverse Step2/Step3 registration is performed before resizing to processed
    resolution. If the processed Zarr shape cannot be read, the implementation
    uses the response ROI shape or derives a shape from ``raw.scale``.
    """
    # Fullres target for inverse Step2 transform. Step2 transform coordinates are
    # fullres. If raw/processed shape is available, final resize happens after.
    stem_shape_fullres = tuple(reference_shape)
    label_fullres = labelmap_reference_to_stem_fullres(labelmap_ref, row, stem_shape_fullres)

    processed_shape = processed_zarr_image_shape(row)
    if processed_shape is None:
        # Fallback: response ROI shape if present, else raw.scale from config.
        roi = load_response_roi_for_row(row)
        if roi is not None:
            processed_shape = tuple(roi.shape)
        else:
            scale = float(raw_scale_factor(config))
            processed_shape = (max(1, int(round(stem_shape_fullres[0] * scale))), max(1, int(round(stem_shape_fullres[1] * scale))))

    label_processed = resize_label_nearest(label_fullres, processed_shape)
    brain_signal, mask_sources = load_canonical_brain_signal_mask(row, config, paths, processed_shape)
    usable = np.where(brain_signal, label_processed, 0).astype(np.int32)

    stem = str(row["stem"])
    label_path = paths["labelmaps_by_stem_root"] / f"{stem}_atlas_labels_processed.npy"
    usable_path = paths["labelmaps_by_stem_root"] / f"{stem}_atlas_usable_labels_processed.npy"
    cov_path = paths["tables_root"] / f"{stem}_atlas_region_coverage.csv"
    np.save(label_path, label_processed.astype(np.int32))
    np.save(usable_path, usable.astype(np.int32))

    coverage = coverage_table_for_labelmaps(label_processed, usable, atlas_table, row, config)
    coverage.to_csv(cov_path, index=False)
    return {
        "stem": stem,
        "atlas_labels_processed_path": str(label_path),
        "atlas_usable_labels_processed_path": str(usable_path),
        "coverage_csv_path": str(cov_path),
        "processed_shape": json.dumps(list(map(int, processed_shape))),
        **mask_sources,
    }


def coverage_table_for_labelmaps(
    labelmap: np.ndarray,
    usable_labelmap: np.ndarray,
    atlas_table: pd.DataFrame,
    row: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize atlas-region coverage by the usable brain-signal mask.
    
    Parameters
    ----------
    labelmap : numpy.ndarray
        Atlas labels in processed (y, x) pixels.
    usable_labelmap : numpy.ndarray
        Same labels with unusable pixels replaced by zero.
    atlas_table : pandas.DataFrame
        Region metadata from the reference label map.
    row : pandas.Series
        Recording identifiers.
    config : dict
        Region area and coverage thresholds from ``step6.coverage``.
    
    Returns
    -------
    pandas.DataFrame
        Per-region total/usable pixel counts, coverage fraction and percent,
        fraction of eligible signal pixels and coverage status.
    
    Notes
    -----
    For each region, ``coverage_fraction`` is the number of retained pixels
    divided by the total projected region pixels. Status is based on the
    configured minimum area, warning and acceptance thresholds.
    """
    ids = sorted(int(x) for x in np.unique(labelmap) if int(x) > 0)
    table_by_id = {}
    if atlas_table is not None and not atlas_table.empty and "atlas_structure_id" in atlas_table.columns:
        for _, r in atlas_table.iterrows():
            table_by_id[int(r["atlas_structure_id"])] = r.to_dict()
    min_ok = float(cfg_get(config, "step6", "coverage", "min_region_coverage_fraction", default=0.5))
    warn = float(cfg_get(config, "step6", "coverage", "warn_region_coverage_fraction", default=0.1))
    min_area = int(cfg_get(config, "step6", "coverage", "min_region_area_px", default=10))
    brain_signal_area = int(np.asarray(usable_labelmap > 0).sum())
    rows = []
    for sid in ids:
        region = labelmap == sid
        usable = usable_labelmap == sid
        area = int(region.sum())
        usable_area = int(usable.sum())
        cov = float(usable_area / area) if area > 0 else np.nan
        if area < min_area or usable_area == 0 or (np.isfinite(cov) and cov < warn):
            status = "missing_or_outside"
        elif cov < min_ok:
            status = "low_coverage"
        else:
            status = "ok"
        meta = table_by_id.get(sid, {})
        rows.append({
            "stem": str(row.get("stem", "")),
            "group_id": str(row.get("group_id", "")),
            "ref_stem": str(row.get("ref_stem", "")),
            "atlas_structure_id": sid,
            "acronym": meta.get("acronym", str(sid)),
            "name": meta.get("name", str(sid)),
            "atlas_area_px": area,
            "usable_area_px": usable_area,
            "coverage_fraction": cov,
            "coverage_percent": float(100.0 * cov) if np.isfinite(cov) else np.nan,
            "brain_signal_area_px": brain_signal_area,
            "region_fraction_of_brain_signal": float(usable_area / brain_signal_area) if brain_signal_area > 0 else np.nan,
            "status": status,
        })
    return pd.DataFrame(rows)


# ============================================================
# QC rendering
# ============================================================


def render_reference_overlay(
    ref_img: np.ndarray,
    labelmap_ref: np.ndarray,
    row: pd.Series,
    basis: dict[str, Any],
    paths: dict[str, Path],
    config: dict[str, Any],
    *,
    response_mask_ref: np.ndarray | None = None,
    response_center_xy: np.ndarray | None = None,
    atlas_response_center_xy: np.ndarray | None = None,
    atlas_response_points_xy: np.ndarray | None = None,
    title: str | None = None,
) -> Path:
    """Save a reference-image overlay of atlas regions and optional anchors.
    """
    key = f"{row.get('group_id', '')}__{row.get('ref_stem', '')}".replace("/", "-").replace("\\", "-")
    out = paths["overlays_root"] / f"{key}_atlas_overlay.png"
    alpha = float(cfg_get(config, "step6", "projection", "overlay_alpha", default=0.35))
    dpi = int(cfg_get(config, "step6", "projection", "overlay_dpi", default=180))
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = normalize01(ref_img)
    ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
    masked = np.ma.masked_where(labelmap_ref <= 0, labelmap_ref)
    ax.imshow(masked, alpha=alpha, cmap="tab20")
    b = _label_boundaries(labelmap_ref)
    if b.any():
        ax.contour(b.astype(float), levels=[0.5], colors="white", linewidths=0.4)
    if response_mask_ref is not None and np.any(response_mask_ref):
        ax.contour(response_mask_ref.astype(float), levels=[0.5], colors="yellow", linewidths=1.2)
    if atlas_response_points_xy is not None and np.asarray(atlas_response_points_xy).size > 0:
        pts = np.asarray(atlas_response_points_xy, dtype=float)
        point_alpha = float(cfg_get(config, "step6", "projection", "response_match_points_alpha", default=0.45))
        ax.scatter(pts[:, 0], pts[:, 1], c="cyan", s=8, alpha=point_alpha, marker=".", label="projected SSp-ll points")
    if response_center_xy is not None and np.all(np.isfinite(response_center_xy)):
        cxy = np.asarray(response_center_xy, dtype=float)
        ax.scatter([cxy[0]], [cxy[1]], c="yellow", s=65, marker="x", linewidths=2.0, label="response center")
    if atlas_response_center_xy is not None and np.all(np.isfinite(atlas_response_center_xy)):
        axy = np.asarray(atlas_response_center_xy, dtype=float)
        ax.scatter([axy[0]], [axy[1]], c="cyan", s=65, marker="o", edgecolors="black", linewidths=0.7, label="projected SSp-ll center")
    origin = basis["origin_xy"]
    ap_end = origin + basis["ap_hat_xy"] * 100
    ml_end = origin + basis["ml_hat_xy"] * 100
    ax.plot([origin[0], ap_end[0]], [origin[1], ap_end[1]], "r-", lw=1.5, label="AP+")
    ax.plot([origin[0], ml_end[0]], [origin[1], ml_end[1]], "b-", lw=1.5, label="ML+")
    ax.scatter([basis["bregma_xy"][0]], [basis["bregma_xy"][1]], c="red", s=30)
    ax.scatter([basis["rostral_xy"][0]], [basis["rostral_xy"][1]], c="green", s=30)
    ax.set_title(title or f"Step6 atlas overlay: {row.get('group_id', '')} / {row.get('ref_stem', '')}")
    ax.axis("off")
    ax.legend(loc="lower right")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out



# ============================================================
# Reference-only response anchors and population fallback
# ============================================================


def _group_key_from_df(group_df: pd.DataFrame) -> tuple[str, str]:
    """Extract the common group/reference identity from a nonempty group table.
    """
    if group_df.empty:
        return "", ""
    return str(group_df["group_id"].iloc[0]), str(group_df["ref_stem"].iloc[0])


def _reference_row_for_group(group_df: pd.DataFrame) -> pd.Series:
    """Select the unique Step3 reference row for one group/reference pair.
    """
    if group_df.empty:
        raise ValueError("group_df is empty")
    ref = str(group_df["ref_stem"].iloc[0])
    ref_rows = group_df[group_df["stem"].astype(str).eq(ref)]
    if not ref_rows.empty:
        return ref_rows.iloc[0]
    return group_df.iloc[0]



def build_reference_response_population_templates(
    step6_index: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    save: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build diagnostic population response templates from reference-stem ROIs.
    
    Parameters
    ----------
    step6_index : pandas.DataFrame
        Step6 index with eligible response paths.
    config : dict
        Template settings under ``step6.missing_response_masks``.
    paths : dict
        Output locations for template arrays and provenance.
    save : bool, optional
        Whether to save template products.
    
    Returns
    -------
    dict
        Available templates indexed by detected response side.
    
    Notes
    -----
    Only response masks drawn on their group's reference stem contribute to
    these templates. A donor template is not automatically used as a fitting
    anchor when another group's side is unknown.
    """
    records: list[tuple[np.ndarray, dict[str, Any], str]] = []
    rows_log: list[dict[str, Any]] = []
    if step6_index is None or step6_index.empty:
        return {}
    for (gid, ref), g in step6_index.groupby([step6_index["group_id"].astype(str), step6_index["ref_stem"].astype(str)], sort=False):
        try:
            ref_row_df = g[g["stem"].astype(str).eq(str(ref))]
            if ref_row_df.empty:
                continue
            row = ref_row_df.iloc[0]
            if not bool(row.get("response_roi_exists", False)):
                continue
            ref_img = load_reference_image_for_row(row)
            ref_shape = tuple(ref_img.shape[:2])
            basis = reference_basis_from_row(row, config)
            mask_ref, status = response_roi_to_reference_space(row, config, ref_shape)
            if mask_ref is None or int(mask_ref.sum()) == 0:
                continue
            side = detect_response_side(mask_ref, basis, config)
            detected = str(side.get("detected_side", "missing"))
            if detected not in {"left", "right"}:
                continue
            records.append((mask_ref, basis, detected))
            rows_log.append({
                "group_id": str(gid),
                "ref_stem": str(ref),
                "stem": str(row.get("stem", "")),
                "detected_side": detected,
                "response_area_px_refspace": int(mask_ref.sum()),
                **{k: v for k, v in status.items() if k not in {"response_roi_path"}},
            })
        except Exception as e:
            rows_log.append({"group_id": str(gid), "ref_stem": str(ref), "error": repr(e)})
            if bool(config.get("step6", {}).get("strict", False)):
                raise

    templates: dict[str, dict[str, Any]] = {}
    for side in ["left", "right"]:
        tmpl = build_response_template_for_side(records, config, side)
        if tmpl is not None:
            templates[side] = tmpl
            if save:
                out_npz = paths["response_template_root"] / f"reference_response_template_{side}.npz"
                out_npz.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    out_npz,
                    probability=np.asarray(tmpl["probability"], dtype=np.float32),
                    side=str(side),
                    bin_px=int(tmpl["bin_px"]),
                    radius_px=int(tmpl["radius_px"]),
                    n_masks=int(tmpl["n_masks"]),
                )
    if save:
        pd.DataFrame(rows_log).to_csv(paths["response_template_root"] / "reference_response_template_sources.csv", index=False)
    return templates


def _template_center_ml_ap_um(template: dict[str, Any], config: dict[str, Any], pixel_size_um: float) -> np.ndarray | None:
    """Calculate the weighted template center in (ML, AP) µm.
    """
    prob = np.asarray(template.get("probability"), dtype=np.float32)
    if prob.size == 0:
        return None
    thr = float(cfg_get(config, "step6", "missing_response_masks", "probability_threshold", default=0.5))
    yy, xx = np.where(prob >= thr)
    weights = prob[yy, xx] if yy.size else np.asarray([], dtype=float)
    if yy.size == 0 or float(np.sum(weights)) <= 0:
        yy, xx = np.where(prob > 0)
        weights = prob[yy, xx] if yy.size else np.asarray([], dtype=float)
    if yy.size == 0 or float(np.sum(weights)) <= 0:
        return None
    bin_px = int(template["bin_px"])
    center = prob.shape[0] // 2
    ml_px = float(np.average((xx - center).astype(float) * bin_px, weights=weights))
    ap_px = float(np.average((yy - center).astype(float) * bin_px, weights=weights))
    return np.array([ml_px * pixel_size_um, ap_px * pixel_size_um], dtype=float)



def resolve_response_anchor_for_group(
    group_df: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    row0: pd.Series,
    ref_shape: tuple[int, int],
    basis: dict[str, Any],
    *,
    response_templates: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any], np.ndarray | None]:
    """Select and register a response anchor for fitting one atlas transform.
    
    Parameters
    ----------
    group_df : pandas.DataFrame
        All selected stem rows sharing one ``group_id`` and ``ref_stem``.
    config : dict
        Anchor selection and template policies under ``step6.response_anchor``
        and ``step6.missing_response_masks``.
    paths : dict
        Reference ROI and optional imputed-mask output locations.
    row0 : pandas.Series
        Step3 reference row for the group.
    ref_shape : tuple of int
        Reference image (height, width), in pixels.
    basis : dict
        Reference AP/ML anatomical coordinate basis.
    response_templates : dict or None, optional
        Side-specific donor templates from other reference stems.
    
    Returns
    -------
    response_mask_ref : numpy.ndarray or None
        Chosen response mask in reference pixels, if available.
    info : dict
        Provenance, detected or explicitly specified hemisphere, QC and
        response weighting details.
    response_ml_ap_um : numpy.ndarray or None
        Selected response-center (ML, AP) in µm, or None for anatomical-only fit.
    
    Notes
    -----
    Priority is configurable: real reference-stem ROI, registered ROI from the
    same group, then a population template only when its side has been explicitly
    set via ``step6.missing_response_masks.side_policy`` and ``fixed_side``.
    Without an anchor the atlas can be fitted anatomically when permitted.
    """
    response_templates = response_templates or {}
    policy = _response_anchor_policy(config)
    anchor_row = choose_response_anchor_for_group(group_df, config)
    if anchor_row is not None:
        response_mask_ref, status = response_roi_to_reference_space(anchor_row, config, ref_shape)
        if response_mask_ref is not None and int(response_mask_ref.sum()) > 0:
            save_response_refspace_mask(anchor_row, response_mask_ref, paths)
            side = detect_response_side(response_mask_ref, basis, config)
            ml_ap_um = np.array([float(side["response_ml_um"]), float(side["response_ap_um"])], dtype=float)
            is_ref = str(anchor_row.get("stem", "")) == str(anchor_row.get("ref_stem", ""))
            source = str(anchor_row.get("_response_anchor_candidate_source", "ref_stem_response_roi" if is_ref else "same_group_registered_response_roi"))
            anchor_status = "ref_stem_response" if is_ref else "same_group_registered_response"
            info = {
                **status,
                **side,
                "response_anchor_status": anchor_status,
                "response_anchor_source": source,
                "response_anchor_policy": policy,
                "response_anchor_stem": str(anchor_row.get("stem", "")),
                "response_anchor_is_ref_stem": bool(is_ref),
                "response_anchor_registered_to_ref": bool(not is_ref),
                "response_anchor_qc_status": str(anchor_row.get("_response_anchor_qc_status", "")),
                "response_anchor_used_for_fit": True,
                "response_weight_multiplier": _source_weight_multiplier(config, source),
                "side_detection_method_used": "response_mask_relative_to_ap_axis",
            }
            return response_mask_ref, info, ml_ap_um

    info_base = {
        "response_anchor_status": "missing_group_response",
        "response_anchor_source": "missing",
        "response_anchor_policy": policy,
        "response_anchor_stem": str(row0.get("ref_stem", "")),
        "response_anchor_is_ref_stem": True,
        "response_anchor_registered_to_ref": False,
        "response_anchor_used_for_fit": False,
        "response_weight_multiplier": 0.0,
        "detected_side": "missing",
    }
    fallback_allowed = bool(cfg_get(config, "step6", "response_anchor", "fallback_to_population_template", default=True))
    if policy in {"ref_only", "ref_stem_only"}:
        fallback_allowed = False
    if not fallback_allowed or not bool(cfg_get(config, "step6", "missing_response_masks", "enabled", default=True)):
        return None, info_base, None

    policy_side = str(cfg_get(config, "step6", "missing_response_masks", "side_policy", default="missing")).lower()
    fixed_side = str(cfg_get(config, "step6", "missing_response_masks", "fixed_side", default="") or "").lower()
    side = fixed_side if policy_side == "fixed" and fixed_side in {"left", "right"} else "missing"
    side_info = {"expected_response_side": side, "expected_side_source": "explicit_fixed_side" if side != "missing" else "unspecified_side"}
    # Never infer the missing anchor side from protocol or the mere existence
    # of a donor template on one hemisphere.
    if side not in response_templates:
        return None, {**info_base, **side_info, "response_anchor_status": "NO_POPULATION_ANCHOR"}, None

    template = response_templates[side]
    response_mask_ref = place_response_template_on_reference(template, basis, ref_shape, config)
    if response_mask_ref is None or int(response_mask_ref.sum()) == 0:
        return None, {**info_base, **side_info, "response_anchor_status": "missing_empty_population_template"}, None

    if bool(cfg_get(config, "step6", "missing_response_masks", "save_imputed_masks", default=True)):
        key = f"{row0.get('group_id', '')}__{row0.get('ref_stem', '')}".replace("/", "-").replace("\\", "-")
        out = paths["response_template_root"] / f"{key}_imputed_response_{side}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, response_mask_ref.astype(np.uint8))

    side_detected = detect_response_side(response_mask_ref, basis, config)
    # Keep the requested template side explicit even if thresholded contour is too small/ambiguous.
    detected_side = str(side_detected.get("detected_side", side))
    if detected_side not in {"left", "right"}:
        detected_side = side
    ml_ap_um = _template_center_ml_ap_um(template, config, float(basis["pixel_size_um"]))
    if ml_ap_um is None:
        ml_ap_um = np.array([float(side_detected.get("response_ml_um", np.nan)), float(side_detected.get("response_ap_um", np.nan))], dtype=float)
    if not np.all(np.isfinite(ml_ap_um)):
        return response_mask_ref, {**info_base, **side_info, "response_anchor_status": "missing_invalid_population_template_center"}, None

    info = {
        **side_detected,
        **side_info,
        "detected_side": detected_side,
        "response_anchor_status": "imputed_population_template",
        "response_anchor_source": "population_ref_stem_response_template",
        "response_anchor_policy": _response_anchor_policy(config),
        "response_anchor_stem": str(row0.get("ref_stem", "")),
        "response_anchor_is_ref_stem": True,
        "response_anchor_registered_to_ref": False,
        "response_anchor_used_for_fit": True,
        "response_weight_multiplier": _source_weight_multiplier(config, "population_ref_stem_response_template"),
        "response_template_side": side,
        "response_template_n_masks": int(template.get("n_masks", 0)),
        "response_ml_um": float(ml_ap_um[0]),
        "response_ap_um": float(ml_ap_um[1]),
        "response_ml_px": float(ml_ap_um[0] / float(basis["pixel_size_um"])),
        "response_ap_px": float(ml_ap_um[1] / float(basis["pixel_size_um"])),
    }
    # Center in image coordinates for plotting.
    center_xy = ml_ap_um_to_xy(ml_ap_um, basis)
    info["response_center_x"] = float(center_xy[0])
    info["response_center_y"] = float(center_xy[1])
    return response_mask_ref, info, ml_ap_um


# ============================================================
# Full atlas extent overlay
# ============================================================


def _structure_path_ids_full_overlay(meta: dict[str, Any]) -> list[int]:
    """Extract integer ancestry IDs for the full-atlas overlay.
    """
    path = None
    for key in ("structure_id_path", "structure_id_path_ids", "id_path", "path"):
        if key in meta:
            path = meta.get(key)
            break
    if path is None:
        return []
    if isinstance(path, str):
        return [int(x) for x in re.findall(r"\d+", path)]
    if isinstance(path, (list, tuple, np.ndarray)):
        out = []
        for x in path:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    return []


def _find_root_structure_ids_full_overlay(atlas_data: AtlasData, queries: Sequence[str]) -> list[int]:
    """Find structure IDs matching the configured overlay region queries.
    """
    q_lower = [str(q).lower() for q in queries]
    exact: list[int] = []
    fallback: list[int] = []
    for sid_key, meta in atlas_data.structures.items():
        sid = int(sid_key)
        acr = str(meta.get("acronym", "")).lower()
        name = str(meta.get("name", "")).lower()
        for q in q_lower:
            if q == acr or q == name:
                exact.append(sid)
            elif q in acr or q in name:
                fallback.append(sid)
    return sorted(set(exact if exact else fallback))


def _collect_descendant_structure_ids_full_overlay(atlas_data: AtlasData, queries: Sequence[str]) -> set[int]:
    """Find matching region IDs together with their atlas descendants.
    """
    roots = set(_find_root_structure_ids_full_overlay(atlas_data, queries))
    descendants = set(roots)
    for sid_key, meta in atlas_data.structures.items():
        sid = int(sid_key)
        path = set(_structure_path_ids_full_overlay(meta))
        if sid in roots or (roots and len(path & roots) > 0):
            descendants.add(sid)
    return descendants


def _binary_contours_ap_ml_full_overlay(mask: np.ndarray, level: float = 0.5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Extract atlas flat-projection contours in (AP, ML) index coordinates.
    """
    mask = np.asarray(mask, dtype=float)
    try:
        from skimage import measure
        segs = measure.find_contours(mask, level=level)
        return [(seg[:, 0].astype(float), seg[:, 1].astype(float)) for seg in segs if len(seg) > 1]
    except Exception:
        fig_tmp, ax_tmp = plt.subplots(figsize=(1, 1))
        cs = ax_tmp.contour(mask, levels=[level])
        segs: list[tuple[np.ndarray, np.ndarray]] = []
        if hasattr(cs, "allsegs") and len(cs.allsegs) > 0:
            for seg in cs.allsegs[0]:
                if len(seg) > 1:
                    segs.append((seg[:, 1].astype(float), seg[:, 0].astype(float)))
        plt.close(fig_tmp)
        return segs


def _contour_area_flat_px_full_overlay(ap: np.ndarray, ml: np.ndarray) -> float:
    """Compute planar polygon area in dorsal-projection pixels.
    """
    ap = np.asarray(ap, dtype=float)
    ml = np.asarray(ml, dtype=float)
    if len(ap) < 3:
        return 0.0
    return float(abs(0.5 * np.sum(ml * np.roll(ap, -1) - np.roll(ml, -1) * ap)))


def _smooth_closed_contour_full_overlay(xy: np.ndarray, window: int) -> np.ndarray:
    """Smooth a closed atlas contour with the requested moving-average window.
    """
    xy = np.asarray(xy, dtype=float)
    w = int(window)
    if w <= 1 or len(xy) < w:
        return xy
    if w % 2 == 0:
        w += 1
    pad = w // 2
    out = np.empty_like(xy)
    for dim in range(2):
        vals = xy[:, dim]
        ext = np.concatenate([vals[-pad:], vals, vals[:pad]])
        kernel = np.ones(w, dtype=float) / float(w)
        out[:, dim] = np.convolve(ext, kernel, mode="valid")
    return out


def _region_id_sets_full_overlay(atlas_data: AtlasData) -> dict[str, set[int]]:
    """Resolve atlas structure ID groups for full-field contour display.
    """
    return {
        "cortex": _collect_descendant_structure_ids_full_overlay(atlas_data, ["Isocortex"]),
        "cerebellum": _collect_descendant_structure_ids_full_overlay(atlas_data, ["CB", "Cerebellum"]),
        "olfactory_bulbs": _collect_descendant_structure_ids_full_overlay(atlas_data, ["MOB", "Main olfactory bulb", "AOB", "Accessory olfactory bulb"]),
    }


def _target_structure_ids_full_overlay(atlas_data: AtlasData, config: dict[str, Any]) -> set[int]:
    """Find the configured response-target atlas structure IDs.
    """
    queries = cfg_get(
        config,
        "step6", "atlas", "sshl_structure_queries",
        default=["SSp-ll", "Primary somatosensory area, lower limb"],
    )
    return _collect_descendant_structure_ids_full_overlay(atlas_data, queries)


def _build_flat_contour_records_full_overlay(
    atlas_data: AtlasData,
    id_to_region: dict[int, str],
    *,
    min_points: int,
    min_area_px: float,
) -> list[dict[str, Any]]:
    """Build filtered flat-atlas contour records for projected overlay regions.
    """
    label_img = np.asarray(atlas_data.flat_label, dtype=np.int32)
    records: list[dict[str, Any]] = []
    for sid in sorted(int(x) for x in id_to_region):
        mask = label_img == int(sid)
        if int(mask.sum()) < max(float(min_area_px), 2):
            continue
        for contour_index, (ap, ml) in enumerate(_binary_contours_ap_ml_full_overlay(mask)):
            if len(ap) < int(min_points):
                continue
            area = _contour_area_flat_px_full_overlay(ap, ml)
            if area < float(min_area_px):
                continue
            records.append({
                "label_id": int(sid),
                "acronym": atlas_data.id_to_acronym.get(int(sid), str(int(sid))),
                "name": atlas_data.id_to_name.get(int(sid), str(int(sid))),
                "region": str(id_to_region[int(sid)]),
                "contour_index": int(contour_index),
                "area_flat_px": float(area),
                "ap": np.asarray(ap, dtype=float),
                "ml": np.asarray(ml, dtype=float),
            })
    return records


def _atlas_bregma_ml_idx_full_overlay(paths: dict[str, Path], atlas_data: AtlasData) -> int:
    """Find the bregma-related atlas ML index used for side display.
    """
    try:
        lm = pd.read_csv(paths["atlas_landmarks_csv"])
        b = lm[lm["name"].astype(str).eq("bregma")]
        if not b.empty:
            return int(b.iloc[-1]["ml_idx"])
    except Exception:
        pass
    return int(atlas_data.n_ml // 2)


def _side_filter_records_full_overlay(records: list[dict[str, Any]], side: str, paths: dict[str, Path], atlas_data: AtlasData, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select contour records according to the requested response hemisphere.
    """
    side = str(side or "both").lower()
    if side not in {"left", "right"}:
        return records
    bregma_ml = _atlas_bregma_ml_idx_full_overlay(paths, atlas_data)
    left_rel = str(cfg_get(config, "step6", "atlas", "left_ml_relation_to_bregma", default="negative")).lower()
    out: list[dict[str, Any]] = []
    for r in records:
        mean_ml = float(np.nanmean(r["ml"]))
        is_left = mean_ml < bregma_ml if left_rel == "negative" else mean_ml > bregma_ml
        if (side == "left" and is_left) or (side == "right" and not is_left):
            out.append(r)
    return out


def _project_flat_contour_to_reference_xy_full_overlay(record: dict[str, Any], atlas_data: AtlasData, transform: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    """Project one atlas (AP, ML) contour into reference (x, y) pixels.
    """
    ap = np.asarray(record["ap"], dtype=float)
    ml = np.asarray(record["ml"], dtype=float)
    ap_i = np.clip(np.round(ap).astype(int), 0, atlas_data.n_ap - 1)
    ml_i = np.clip(np.round(ml).astype(int), 0, atlas_data.n_ml - 1)
    dv_i = atlas_data.flat_dv_idx[ap_i, ml_i]
    xyz = ap_dv_ml_indices_to_xyz_um(ap, dv_i, ml, atlas_data, config)
    uv, _ = pitch_project_xyz_um(
        xyz,
        float(transform["pitch_final_deg"]),
        np.asarray(transform["pivot_xyz_um"], dtype=float),
    )
    basis_for_projection = {
        "origin_xy": np.asarray(transform["origin_xy"], dtype=float),
        "ap_hat_xy": np.asarray(transform["ap_hat_xy"], dtype=float),
        "ml_hat_xy": np.asarray(transform["ml_hat_xy"], dtype=float),
        "pixel_size_um": float(transform["pixel_size_um"]),
    }
    xy = atlas_uv_um_to_reference_xy_px(uv, float(transform["scale_final"]), basis_for_projection)
    smooth_window = int(cfg_get(config, "step6", "full_atlas_overlay", "contour_smooth_window", default=7))
    return _smooth_closed_contour_full_overlay(xy, smooth_window)


def _load_projected_atlas_landmark_points(transform: dict[str, Any], paths: dict[str, Path]) -> dict[str, np.ndarray]:
    """Load the atlas landmark positions and map them into reference pixels.
    """
    try:
        lm = pd.read_csv(paths["atlas_landmarks_csv"])
    except Exception:
        return {}
    out: dict[str, np.ndarray] = {}
    for _, r in lm.iterrows():
        name = str(r.get("name", ""))
        if not name:
            continue
        xyz = np.array([[float(r["x_ml_um"]), float(r["y_ap_um"]), float(r["z_dv_um"])]], dtype=float)
        out[name] = project_atlas_xyz_to_reference_xy(xyz, transform)[0]
    return out


def render_full_atlas_extent_overlay(
    ref_img: np.ndarray,
    atlas_data: AtlasData,
    transform: dict[str, Any],
    row0: pd.Series,
    group_df: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    response_mask_ref: np.ndarray | None = None,
    detected_side: str = "missing",
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, Path, dict[str, Any]]:
    """Render the full dorsal atlas extent around the reference image.
    
    Parameters
    ----------
    ref_img : numpy.ndarray
        Reference image in full-resolution pixels.
    atlas_data : AtlasData
        BrainGlobe labels and flat dorsal projection.
    transform : dict
        Fitted atlas-to-reference transform with pitch, scale and basis.
    row0 : pandas.Series
        Group/reference metadata.
    group_df : pandas.DataFrame
        Group stems used for context and overlays.
    config : dict
        Display settings under ``step6.full_atlas_overlay``.
    paths : dict
        Output locations.
    response_mask_ref : numpy.ndarray or None, optional
        Optional response mask in reference-space (y, x) pixels.
    detected_side : str, optional
        Selected hemisphere label.
    save_path : str or pathlib.Path or None, optional
        Explicit output path, overriding the generated name.
    
    Returns
    -------
    figure : matplotlib.figure.Figure
        Full-extent annotated plot.
    path : pathlib.Path
        Saved figure path.
    info : dict
        Rendering and coordinate-offset metadata.
    
    Notes
    -----
    The canvas may extend beyond the reference image footprint to show all
    selected projected atlas contours. This is a visualization product and does
    not alter saved label maps or fitted transforms.
    """
    cfg = cfg_get(config, "step6", "full_atlas_overlay", default={}) or {}
    ref_h, ref_w = ref_img.shape[:2]

    region_sets = _region_id_sets_full_overlay(atlas_data)
    id_to_region: dict[int, str] = {}
    if bool(cfg.get("show_cortex", True)):
        id_to_region.update({int(s): "cortex" for s in region_sets["cortex"]})
    if bool(cfg.get("show_cerebellum", True)):
        id_to_region.update({int(s): "cerebellum" for s in region_sets["cerebellum"]})
    if bool(cfg.get("show_olfactory_bulbs", True)):
        id_to_region.update({int(s): "olfactory_bulbs" for s in region_sets["olfactory_bulbs"]})

    broad_records = _build_flat_contour_records_full_overlay(
        atlas_data,
        id_to_region,
        min_points=int(cfg.get("contour_min_points", 25)),
        min_area_px=float(cfg.get("contour_min_area_flat_px", 8.0)),
    )

    target_records: list[dict[str, Any]] = []
    if bool(cfg.get("show_target", True)):
        target_ids = _target_structure_ids_full_overlay(atlas_data, config)
        target_records = _build_flat_contour_records_full_overlay(
            atlas_data,
            {int(s): "target" for s in target_ids},
            min_points=8,
            min_area_px=2.0,
        )
        if str(cfg.get("target_side_mode", "detected")).lower() == "detected":
            target_records = _side_filter_records_full_overlay(target_records, detected_side, paths, atlas_data, config)

    projected = []
    for rec in broad_records:
        xy = _project_flat_contour_to_reference_xy_full_overlay(rec, atlas_data, transform, config)
        if np.all(np.isfinite(xy)):
            projected.append((rec, xy))
    projected_target = []
    for rec in target_records:
        xy = _project_flat_contour_to_reference_xy_full_overlay(rec, atlas_data, transform, config)
        if np.all(np.isfinite(xy)):
            projected_target.append((rec, xy))

    all_xy = [xy for _, xy in projected] + [xy for _, xy in projected_target]
    margin = float(cfg.get("margin_px", 80))
    if all_xy:
        xy_cat = np.concatenate(all_xy, axis=0)
        min_x = min(0.0, float(np.nanmin(xy_cat[:, 0]))) - margin
        max_x = max(float(ref_w), float(np.nanmax(xy_cat[:, 0]))) + margin
        min_y = min(0.0, float(np.nanmin(xy_cat[:, 1]))) - margin
        max_y = max(float(ref_h), float(np.nanmax(xy_cat[:, 1]))) + margin
    else:
        min_x, max_x, min_y, max_y = -margin, ref_w + margin, -margin, ref_h + margin

    fig_w = max(7.0, min(13.0, (max_x - min_x) / 140.0))
    fig_h = max(7.0, min(13.0, (max_y - min_y) / 140.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    disp = normalize01(ref_img)
    ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None, zorder=0)
    ax.plot([0, ref_w, ref_w, 0, 0], [0, 0, ref_h, ref_h, 0], color="white", lw=1.0, alpha=0.9, zorder=2)

    colors = {
        "cortex": "tab:blue",
        "cerebellum": "tab:purple",
        "olfactory_bulbs": "tab:green",
        "target": "cyan",
    }
    fill_alpha = float(cfg.get("fill_alpha", 0.16))
    target_fill_alpha = float(cfg.get("target_fill_alpha", 0.28))
    boundary_lw = float(cfg.get("boundary_linewidth", 0.8))
    target_lw = float(cfg.get("target_linewidth", 1.6))

    for rec, xy in projected:
        color = colors.get(rec["region"], "white")
        if fill_alpha > 0:
            ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=fill_alpha, lw=0, zorder=3)
        ax.plot(xy[:, 0], xy[:, 1], color="white", lw=boundary_lw, alpha=0.85, zorder=4)

    for rec, xy in projected_target:
        color = colors.get("target", "cyan")
        if target_fill_alpha > 0:
            ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=target_fill_alpha, lw=0, zorder=5)
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=target_lw, alpha=0.95, zorder=6)

    if bool(cfg.get("show_response_mask", True)) and response_mask_ref is not None and np.any(response_mask_ref):
        ax.contour(response_mask_ref.astype(float), levels=[0.5], colors="yellow", linewidths=1.4, zorder=8)

    basis = reference_basis_from_row(row0, config)
    origin = np.asarray(basis["origin_xy"], dtype=float)
    ap_end = origin + np.asarray(basis["ap_hat_xy"], dtype=float) * 150
    ml_end = origin + np.asarray(basis["ml_hat_xy"], dtype=float) * 150
    ax.plot([origin[0], ap_end[0]], [origin[1], ap_end[1]], "r-", lw=1.4, label="image AP+", zorder=10)
    ax.plot([origin[0], ml_end[0]], [origin[1], ml_end[1]], "b-", lw=1.4, label="image ML+", zorder=10)

    if bool(cfg.get("show_image_landmarks", True)):
        ax.scatter([basis["bregma_xy"][0]], [basis["bregma_xy"][1]], c="red", s=40, label="image bregma", zorder=11)
        ax.scatter([basis["rostral_xy"][0]], [basis["rostral_xy"][1]], c="lime", s=40, label="image rostral", zorder=11)

    if bool(cfg.get("show_projected_atlas_landmarks", True)):
        atlas_lm_xy = _load_projected_atlas_landmark_points(transform, paths)
        if "bregma" in atlas_lm_xy:
            xy = atlas_lm_xy["bregma"]
            ax.scatter([xy[0]], [xy[1]], c="none", edgecolors="red", s=95, marker="o", linewidths=1.8, label="atlas bregma projected", zorder=12)
        if "rostral" in atlas_lm_xy:
            xy = atlas_lm_xy["rostral"]
            ax.scatter([xy[0]], [xy[1]], c="none", edgecolors="lime", s=95, marker="s", linewidths=1.8, label="atlas rostral projected", zorder=12)

    title = (
        f"Full atlas extent on reference | {row0.get('group_id', '')} / {row0.get('ref_stem', '')}\n"
        f"pitch={float(transform['pitch_final_deg']):.2f}°, scale={float(transform['scale_final']):.4f}, "
        f"response={transform.get('response_anchor_status', 'missing')}, side={detected_side}, "
        f"contours={len(projected)}, target={len(projected_target)}"
    )
    ax.set_title(title)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_path is None:
        key = f"{row0.get('group_id', '')}__{row0.get('ref_stem', '')}".replace("/", "-").replace("\\", "-")
        save_path = paths["overlays_root"] / f"{key}_full_atlas_extent_overlay.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=int(cfg.get("dpi", 220)), bbox_inches="tight")
    return fig, save_path, {
        "n_broad_contours": int(len(projected)),
        "n_target_contours": int(len(projected_target)),
        "detected_side": str(detected_side),
        "xlim": json.dumps([float(min_x), float(max_x)]),
        "ylim": json.dumps([float(min_y), float(max_y)]),
    }

# ============================================================
# High-level processing
# ============================================================


def _response_anchor_policy(config: dict[str, Any]) -> str:
    """Resolve the active response-anchor selection policy from Step6 YAML.
    """
    policy = cfg_get(config, "step6", "response_anchor", "policy", default=None)
    if policy in (None, ""):
        policy = cfg_get(config, "step6", "reference", "response_anchor_policy", default="ref_then_registered_same_group_then_population")
    return str(policy).strip().lower()


def _source_weight_multiplier(config: dict[str, Any], source: str) -> float:
    """Return the configured reliability multiplier for a response-anchor source.
    """
    weights = cfg_get(config, "step6", "response_anchor", "source_weight_multiplier", default={}) or {}
    if source in weights:
        return float(weights[source])
    # Backward-compatible population multiplier.
    if source == "population_ref_stem_response_template":
        return float(cfg_get(config, "step6", "missing_response_masks", "imputed_fit_weight_multiplier", default=0.35))
    return 1.0


def _row_registration_qc_ok(row: pd.Series, config: dict[str, Any]) -> tuple[bool, str]:
    """Determine whether registration-QC metadata permits a non-reference anchor.
    """
    if not bool(cfg_get(config, "step6", "response_anchor", "require_registration_qc_ok", default=False)):
        return True, "not_required"
    ok_values = {str(v).strip().lower() for v in cfg_get(config, "step6", "response_anchor", "registration_qc_ok_values", default=["ok", "good", "true", "1", "yes", "y"])}
    for col in ["registration_qc", "registration_status", "review_ok"]:
        if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip():
            text = str(row.get(col)).strip().lower()
            return text in ok_values, f"{col}={text}"
    # No QC column: do not silently claim it is good.
    return False, "registration_qc_missing"


def choose_response_anchor_for_group(group_df: pd.DataFrame, config: dict[str, Any]) -> pd.Series | None:
    """Select a real Step5 response ROI according to the configured source priority.
    
    Parameters
    ----------
    group_df : pandas.DataFrame
        Records sharing ``group_id`` and ``ref_stem``.
    config : dict
        ``step6.response_anchor`` policy and optional registration QC gate.
    
    Returns
    -------
    pandas.Series or None
        The chosen row with anchor status fields, or None if no eligible ROI
        is available.
    
    Notes
    -----
    Reference-stem ROI takes precedence. A non-reference ROI is eligible only
    when permitted by policy and the relevant registration QC settings.
    """
    if group_df.empty or "response_roi_exists" not in group_df.columns:
        return None

    ref = str(group_df["ref_stem"].iloc[0])
    have = group_df[group_df["response_roi_exists"].astype(bool)].copy()
    if have.empty:
        return None

    # 1) True reference response, highest confidence.
    ref_have = have[have["stem"].astype(str).eq(ref)].copy()
    if not ref_have.empty:
        row = ref_have.iloc[0].copy()
        row["_response_anchor_candidate_source"] = "ref_stem_response_roi"
        row["_response_anchor_qc_status"] = "ref_stem_no_registration_needed"
        return row

    policy = _response_anchor_policy(config)
    allow_nonref = bool(cfg_get(config, "step6", "response_anchor", "allow_nonreference_response", default=True))
    if policy in {"ref_only", "ref_stem_only", "ref_then_population", "ref_stem_only_then_population_template"}:
        allow_nonref = False
    if not allow_nonref:
        return None

    # 2) Same-group non-reference responses are allowed only after Step2 registration
    # into the reference frame. Pick the first QC-allowed candidate.
    nonref = have[~have["stem"].astype(str).eq(ref)].copy()
    if nonref.empty:
        return None
    # Preserve the authoritative Step2 metadata row order. Neither protocol
    # names nor their suffixes are used to prioritize a response ROI.
    first_rejected_status = ""
    for _, rr in nonref.iterrows():
        ok, qc_status = _row_registration_qc_ok(rr, config)
        if ok:
            row = rr.copy()
            row["_response_anchor_candidate_source"] = "same_group_registered_response_roi"
            row["_response_anchor_qc_status"] = qc_status
            return row
        first_rejected_status = qc_status

    # Return no anchor; caller can fall back to population template.
    return None


def process_step6_reference(
    group_df: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    atlas_data: AtlasData | None = None,
    save_stem_labelmaps: bool = True,
    response_templates: dict[str, dict[str, Any]] | None = None,
    save_full_atlas_overlay: bool | None = None,
) -> dict[str, Any]:
    """Fit and save one atlas transform and optional per-stem label maps.
    
    Parameters
    ----------
    group_df : pandas.DataFrame
        Stem rows for one Step3 ``group_id``/``ref_stem`` pair.
    config : dict
        Full Step6 project configuration.
    paths : dict
        Step6 folders and output paths.
    atlas_data : AtlasData or None, optional
        Loaded atlas, if already available.
    save_stem_labelmaps : bool, optional
        Whether to create processed-resolution label maps and coverage products.
    response_templates : dict or None, optional
        Previously assembled reference-stem donor templates.
    save_full_atlas_overlay : bool or None, optional
        Overrides ``step6.full_atlas_overlay.enabled`` for this invocation.
    
    Returns
    -------
    dict
        Paths to saved transform, reference label map and overlays, fit metrics,
        response provenance, and number of created stem maps.
    
    Notes
    -----
    Uses the Step3 reference and the saved atlas landmarks. It fits pitch/scale,
    projects the 2D dorsal atlas labels, then optionally inverse-registers labels
    into each stem's processed image resolution. Per-stem failures can be logged
    without aborting the group unless ``step6.strict`` is enabled.
    """
    if group_df.empty:
        raise ValueError("group_df is empty")

    # Always work from the actual reference row when it is present.
    row0 = _reference_row_for_group(group_df)
    ref_img = load_reference_image_for_row(row0)
    ref_shape = tuple(ref_img.shape[:2])
    basis = reference_basis_from_row(row0, config)

    response_mask_ref, response_info, response_ml_ap_um = resolve_response_anchor_for_group(
        group_df,
        config,
        paths,
        row0,
        ref_shape,
        basis,
        response_templates=response_templates,
    )

    if atlas_data is None:
        return {
            "group_id": str(row0.get("group_id", "")),
            "ref_stem": str(row0.get("ref_stem", "")),
            "status": "response_processed_no_atlas",
            **response_info,
        }

    # Cache landmarks dataframe for side-specific atlas structure center.
    lm_df = pd.read_csv(paths["atlas_landmarks_csv"])
    config.setdefault("_step6_cache", {})["atlas_landmark_df"] = lm_df
    atlas_landmarks = load_atlas_landmarks(paths)

    atlas_response_xyz = None
    atlas_response_points_xyz = np.empty((0, 3), dtype=float)
    atlas_anchor_extra: dict[str, Any] = {"atlas_response_anchor_status": "missing"}
    atlas_points_extra: dict[str, Any] = {"sshl_match_points_status": "missing"}
    if response_info.get("detected_side") in {"left", "right"} and response_ml_ap_um is not None:
        atlas_response_xyz, atlas_anchor_extra = atlas_structure_center_xyz_um(atlas_data, config, str(response_info["detected_side"]))
        if bool(cfg_get(config, "step6", "projection", "show_response_match_points", default=True)):
            atlas_response_points_xyz, atlas_points_extra = atlas_structure_flat_points_xyz_um(atlas_data, config, str(response_info["detected_side"]))
        if atlas_response_xyz is None:
            response_ml_ap_um = None

    fit, fit_table = fit_pitch_scale(
        basis,
        atlas_landmarks,
        config,
        response_ml_ap_um=response_ml_ap_um,
        atlas_response_xyz_um=atlas_response_xyz,
        response_anchor_status=atlas_anchor_extra.get("atlas_response_anchor_status", response_info.get("response_anchor_status", "missing")),
        response_weight_multiplier=float(response_info.get("response_weight_multiplier", 1.0) or 1.0),
    )
    transform = transform_json_from_fit(
        row0,
        basis,
        fit,
        fit_table,
        config,
        extra={
            **response_info,
            **atlas_anchor_extra,
            **atlas_points_extra,
            "reference_shape": list(map(int, ref_shape)),
        },
    )
    key = f"{row0.get('group_id', '')}__{row0.get('ref_stem', '')}".replace("/", "-").replace("\\", "-")
    transform_path = paths["transforms_root"] / f"{key}_atlas_transform.json"
    save_json(transform_path, transform)

    labelmap_ref = project_atlas_flat_labels_to_reference(atlas_data, transform, ref_shape, config)
    ref_label_path, table_path = save_reference_labelmap_outputs(labelmap_ref, atlas_data, row0, paths)
    atlas_table = pd.read_csv(table_path)
    overlay_path = ""
    if bool(cfg_get(config, "step6", "projection", "save_overlay_png", default=True)):
        response_center_xy = None
        if response_info.get("response_center_x") is not None:
            try:
                response_center_xy = np.array([float(response_info["response_center_x"]), float(response_info["response_center_y"])], dtype=float)
            except Exception:
                response_center_xy = None
        atlas_center_xy = project_atlas_xyz_to_reference_xy(atlas_response_xyz[None, :], transform)[0] if atlas_response_xyz is not None else None
        atlas_points_xy = project_atlas_xyz_to_reference_xy(atlas_response_points_xyz, transform) if atlas_response_points_xyz is not None and np.asarray(atlas_response_points_xyz).size > 0 else None
        overlay_path = str(render_reference_overlay(
            ref_img,
            labelmap_ref,
            row0,
            basis,
            paths,
            config,
            response_mask_ref=response_mask_ref,
            response_center_xy=response_center_xy,
            atlas_response_center_xy=atlas_center_xy,
            atlas_response_points_xy=atlas_points_xy,
        ))

    full_overlay_path = ""
    full_overlay_info: dict[str, Any] = {}
    if save_full_atlas_overlay is None:
        save_full_atlas_overlay = bool(cfg_get(config, "step6", "full_atlas_overlay", "enabled", default=False))
    if bool(save_full_atlas_overlay):
        fig_full, full_path, full_overlay_info = render_full_atlas_extent_overlay(
            ref_img,
            atlas_data,
            transform,
            row0,
            group_df,
            config,
            paths,
            response_mask_ref=response_mask_ref,
            detected_side=str(response_info.get("detected_side", "missing")),
        )
        plt.close(fig_full)
        full_overlay_path = str(full_path)

    stem_rows = []
    if save_stem_labelmaps and bool(cfg_get(config, "step6", "projection", "save_stem_labelmaps", default=True)):
        for _, rr in group_df.iterrows():
            try:
                stem_rows.append(make_stem_labelmaps_and_coverage(rr, labelmap_ref, atlas_table, config, paths, reference_shape=ref_shape))
            except Exception as e:
                err = {"stem": str(rr.get("stem", "")), "group_id": str(rr.get("group_id", "")), "error": repr(e), "updated_at": _dt.datetime.now().isoformat()}
                append_or_create_csv(paths["error_log_csv"], err)
                if bool(config["step6"].get("strict", False)):
                    raise

    rec = {
        "group_id": str(row0.get("group_id", "")),
        "ref_stem": str(row0.get("ref_stem", "")),
        "transform_path": str(transform_path),
        "labelmap_refspace_path": str(ref_label_path),
        "label_table_path": str(table_path),
        "overlay_path": overlay_path,
        "full_atlas_overlay_path": full_overlay_path,
        "pitch_final_deg": float(fit["pitch_final_deg"]),
        "scale_final": float(fit["scale_final"]),
        "rmse_final_um": float(fit["rmse_final_um"]),
        "n_stem_labelmaps": int(len(stem_rows)),
        "updated_at": _dt.datetime.now().isoformat(),
        **response_info,
        **atlas_anchor_extra,
        **atlas_points_extra,
        **{f"full_overlay_{k}": v for k, v in full_overlay_info.items()},
    }
    append_or_create_csv(paths["transform_log_csv"], rec)
    return rec

def run_step6_batch(
    config_path: str | Path,
    *,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    filters: dict[str, Any] | None = None,
    load_atlas: bool = True,
    save_full_atlas_overlay: bool | None = None,
) -> pd.DataFrame:
    """Process Step6 atlas fits for selected registration groups.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the project YAML.
    groups, stems : sequence of str or None, optional
        Explicit group and recording filters.
    filters : dict or None, optional
        Additional metadata filters.
    load_atlas : bool, optional
        Whether to load BrainGlobe atlas before processing.
    save_full_atlas_overlay : bool or None, optional
        Override the full-extent overlay setting for this run.
    
    Returns
    -------
    pandas.DataFrame
        Group-level atlas transform log and output paths.
    
    Notes
    -----
    Population templates are computed once from selected reference-stem ROIs.
    For each group the anchor-selection policy chooses a real ROI, an allowed
    template or an anatomical-only fit. Error continuation is controlled by
    ``step6.batch.continue_on_error``.
    """
    df, config, paths = build_step6_index(config_path, groups=groups, stems=stems, filters=filters)
    atlas_data = load_allen_atlas(config) if load_atlas else None
    response_templates = build_reference_response_population_templates(df, config, paths, save=True)
    out_rows = []
    for (_, _), g in df.groupby([df["group_id"].astype(str), df["ref_stem"].astype(str)], sort=False):
        try:
            out_rows.append(process_step6_reference(
                g.reset_index(drop=True),
                config,
                paths,
                atlas_data=atlas_data,
                response_templates=response_templates,
                save_full_atlas_overlay=save_full_atlas_overlay,
            ))
        except Exception as e:
            rec = {
                "group_id": str(g["group_id"].iloc[0]) if not g.empty else "",
                "ref_stem": str(g["ref_stem"].iloc[0]) if not g.empty else "",
                "error": repr(e),
                "updated_at": _dt.datetime.now().isoformat(),
            }
            out_rows.append(rec)
            append_or_create_csv(paths["error_log_csv"], rec)
            if not bool(cfg_get(config, "step6", "batch", "continue_on_error", default=True)):
                raise
    return pd.DataFrame(out_rows)


# ============================================================
# CLI
# ============================================================


def _parse_filter_items(items: Sequence[str] | None) -> dict[str, Any] | None:
    """Parse repeated ``KEY=VALUE`` CLI metadata filters into a mapping.
    """
    filters: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Filter must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if "," in v:
            filters[k] = [x.strip() for x in v.split(",") if x.strip()]
        else:
            filters[k] = v
    return filters or None


def main(argv: Sequence[str] | None = None) -> None:
    """Parse Step6 CLI switches for index, reference-ROI and atlas batch workflows.
    
    Parameters
    ----------
    argv : sequence of str or None, optional
        Explicit CLI tokens, or process arguments if omitted.
    
    Notes
    -----
    The supported entry points are ``--list``, ``--prepare-response-refspace``
    and ``--batch``. Atlas annotation itself remains an interactive Napari step.
    """
    p = argparse.ArgumentParser(description="Step6 atlas transformation for WIFIOPIA/OISI")
    p.add_argument("--config", required=True, help="Path to project YAML config")
    p.add_argument("--list", action="store_true", help="Only build/list Step6 index")
    p.add_argument("--prepare-response-refspace", action="store_true", help="Transform Step5 response masks to Step3 reference space and detect side")
    p.add_argument("--batch", action="store_true", help="Run atlas transform batch by group/ref_stem")
    p.add_argument("--no-atlas", action="store_true", help="Do not load BrainGlobe atlas; useful for response-mask/side testing")
    p.add_argument("--group", dest="groups", action="append", help="group_id filter; can be repeated")
    p.add_argument("--stem", dest="stems", action="append", help="stem filter; can be repeated")
    p.add_argument("--filter", dest="filters", action="append", help="metadata filter KEY=VALUE; can be repeated")
    p.add_argument("--require-response-roi", action="store_true", help="Keep only stems with existing Step5 response ROI")
    p.add_argument("--full-atlas-overlay", action="store_true", help="Save full-extent atlas overlay during batch")
    args = p.parse_args(argv)

    filters = _parse_filter_items(args.filters)
    df, config, paths = build_step6_index(
        args.config,
        groups=args.groups,
        stems=args.stems,
        filters=filters,
        require_response_roi=bool(args.require_response_roi),
    )
    if args.list:
        cols = [
            c for c in [
                "stem", "group_id", "ref_stem", "exp_day", "state", "protocol",
                "step3_annotation_complete", "response_roi_exists", "ref_image_exists",
                "processed_zarr_exists", "response_roi_path",
            ] if c in df.columns
        ]
        print(df[cols].to_string(index=False))
        print(f"Saved Step6 index: {paths['step6_index_csv']}")
        return

    if args.prepare_response_refspace:
        out = process_response_masks_to_reference(df, config, paths, save=True)
        print(out.to_string(index=False))
        print(f"Saved side log: {paths['side_detection_log_csv']}")
        return

    if args.batch:
        out = run_step6_batch(
            args.config,
            groups=args.groups,
            stems=args.stems,
            filters=filters,
            load_atlas=not bool(args.no_atlas),
            save_full_atlas_overlay=bool(args.full_atlas_overlay),
        )
        print(out.to_string(index=False))
        print(f"Saved transform log: {paths['transform_log_csv']}")
        return

    print("Nothing to do. Use --list, --prepare-response-refspace, or --batch.")


if __name__ == "__main__":
    main()
