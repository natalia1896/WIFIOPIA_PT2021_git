"""Step6b: canonical masks, ROI representation, cohort QC and stem-specific usability.

This stage sits after Step6 atlas registration and before functional-connectivity
calculation. It DOES NOT calculate correlations and DOES NOT define the final
Step7 correlation grid.

Architecture
-----------------
1. Broad Isocortex ROIs are represented as independent canonical side ROIs.
   Optional selective refinement can replace only chosen families (for example
   MO -> MOp + MOs) while all unlisted families stay broad. Full and buffered
   canonical areas are fixed for the experiment; reference-image pixel counts
   are saved separately for audit.
2. A mask_set defines the temporal comparison. Its fixed technical mask is the
   intersection of technically valid pixels across the selected stems. Rescue
   shadow and pathology are NOT part of this fixed intersection.
3. Three spatial mask products are saved for every mouse x mask_set:
   bilateral technical, healthy-hemisphere global, and one-sided homotopic
   pairable mask displayed on the configured hemisphere.
4. Rescue shadow and pathology are applied per stem. A failed stem removes only
   that mouse x ROI x stem data point; it never removes the mouse/ROI from all
   other time points.
5. ROI eligibility is based on fixed technical representation: minimum pixels,
   cohort-relative coverage and optional absolute coverage. all_selected_stems_pass
   is QC-only and is never an ROI-selection gate.
6. Thresholds can be overridden separately for each mask_set, allowing stricter
   healthy stability / anesthesia comparisons without changing longitudinal rules.
7. Cohort frequency maps are created in canonical Allen space by summing one
   fixed technical mask per mouse and dividing by the number of included mice.
8. Canonical-to-reference left/right consistency is checked for every mouse. A
   left/right swap can be configured to fail fast rather than silently corrupt
   side-specific coverage.
9. Final Step7 masks/pairs are still rebuilt after FC spatial resizing; Step6b is
   the geometry, coverage and eligibility QC stage.
10. Reference-space analytical ROI labels use the same repair-before-erosion
    principle as the canonical branch. The repaired buffered reference labelmap is
    the authoritative analytical atlas source for Step7; canonical geometry remains
    authoritative for coverage denominators and eligibility.
11. The workflow optionally erodes the mask-set-specific common_brain before fixed
    technical artifacts are subtracted. This removes a configurable outer brain-edge
    safety band while preserving all existing downstream filenames and the existing
    left/right hemisphere-mask logic. Rescue shadow and pathology remain stem-specific.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import EuclideanTransform, warp

from common import (
    canonical_mask_paths,
    get_path_resolving_config,
    load_config,
    resolve_flexible_path,
)

try:
    from step6_atlas_transformation import (
        build_step6_index,
        load_allen_atlas,
        load_reference_image_for_row,
        reference_basis_from_row,
        ap_dv_ml_indices_to_xyz_um,
        pitch_project_xyz_um,
        atlas_uv_um_to_reference_xy_px,
        registration_tform_stem_to_ref,
    )
except Exception as exc:  # pragma: no cover - environment/project dependent
    raise ImportError(
        "Step6b requires step6_atlas_transformation.py next to this file "
        "and the same environment used for Step6."
    ) from exc



# =============================================================================
# Defaults / config
# =============================================================================


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested configuration value, returning the supplied fallback if absent."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def default_step6b_config() -> dict[str, Any]:
    """Return the built-in Step6b defaults for masks, selection, QC, and eligibility."""
    return {
        "folder_name": "step6b_fc_geometry",
        "active_rest_variant": "without_rest",
        "include_bad_rows": True,  # Step2 is_bad is NOT an approved quantitative QC gate.
        "use_manual_registration_overrides": True,
        "strict_missing_declared_masks": True,
        "continue_on_error": False,
        "selection": {
            "require_assigned_experimental_group": True,  # legacy default, now overridden by mask_set
            "unassigned_label": "UNASSIGNED",
            "fail_on_required_selection_problem": False,
        },
        "broad_atlas": {
            "parent_acronym": "Isocortex",
            "save_original_to_broad_mapping": True,
            # Optional selective refinement. Unlisted broad families stay unchanged.
            # Example: MO -> MOp + MOs while SS/VIS/RSP/... remain broad.
            "roi_refinements": {},
        },
        "technical_sources": {
            "bad_auto": True,
            "bregma": True,
            # v4.9: rescue shadow is loaded here but applied per stem AFTER the
            # fixed longitudinal technical mask is built.
            "rescue_shadow": True,
            "step4_shadow_union_manual": False,
        },
        "technical_geometry": {
            # Optional safety margin along the boundary of the
            # mask-set-specific common brain. Erosion is applied BEFORE fixed
            # bad/bregma artifacts are subtracted, so those internal artifact
            # masks are not enlarged by this operation.
            "common_brain_edge_erosion": {
                "enabled": False,
                "radius_px": 0,
            },
        },
        # Hemorrhage is authoritative ONLY at step2_masks/masks/<stem>/hemorrhage_manual.npy.

        "hemisphere": {
            "lesion_side": "right",
            "per_mouse_lesion_side": {},
            # Optional Step6b-only override. If null, use step6.side_detection.ml_positive_label.
            "ml_positive_label": None,
            "midline_exclusion_px": 0.5,
            # Where the homotopic value/map is assigned. The correlation itself is symmetric.
            "homotopic_map_hemisphere": "lesion",   # lesion | healthy | left | right
            # Primary GBC analysis hemisphere.
            "global_analysis_hemisphere": "healthy", # healthy | lesion | left | right
        },
        "roi_geometry": {
            # v4.9: buffer is defined on the canonical Allen dorsal grid.
            "boundary_exclusion_canonical_px": 2,
            # Fill enclosed zero-label holes INSIDE each analytical side ROI
            # before erosion. This prevents a 1-pixel atlas artifact from
            # expanding into a large artificial exclusion after buffering.
            "fill_internal_zero_holes_before_buffer": True,
            # Use 4-connectivity for BACKGROUND when identifying internal holes.
            # This is intentional: a 1-pixel hole that touches outside only
            # diagonally is still treated as an internal raster hole.
            "internal_hole_background_connectivity": 4,
            # Optional maximum hole size. null/None = fill every enclosed
            # zero-label hole.
            "max_internal_zero_hole_area_px": None,
            # v4.9: allow reassignment of a TINY non-zero analytical island
            # that is completely enclosed inside another ROI. This repairs
            # raster/ontology single-pixel artifacts such as the MOs hole.
            # 0 disables; 1 allows only a single enclosed pixel.
            "max_internal_nonzero_hole_area_px": 1,
            # Reference-space equivalent is calculated from physical pixel sizes
            # and saved only as an audit/QC measurement.
            "save_refspace_equivalent_buffer_qc": True,
        },
        "roi_thresholds": {
            "min_technical_coverage": 0.50,
            "min_homotopic_technical_coverage": 0.50,
            "min_global_technical_coverage": 0.50,
            "use_absolute_coverage_gate": False,
            "cohort_relative": {
                "enabled": True,
                "reference_stat": "median",
                "min_relative_fraction": 0.85,
                "max_relative_fraction": 1.25,
                # flag: >max is reported but not excluded; exclude: it becomes a gate.
                "upper_bound_mode": "flag",
            },
            "min_stem_usable_coverage": 0.50,  # fraction of FIXED technical ROI; review per analysis
            # Applied to canonical ROI pixels / canonical homotopic pairs.
            "min_pixels": 50,
            "min_mouse_fraction_per_group": 0.80,
            "require_all_groups": True,
        },
        "side_consistency_qc": {
            "enabled": True,
            "fail_on_swap": True,
            # expected-side overlap minus opposite-side overlap must exceed this.
            "min_expected_minus_opposite": 0.20,
        },
        "qc": {
            "organize_png_by_mask_set": True,
            "save_pathology_png_only_if_nonzero": True,
            "save_shadow_png_only_if_nonzero": True,
            "save_canonical_mask_pngs": True,
            "save_cohort_frequency_maps": True,
            "save_reference_atlas_repair_png": True,
        },
        "primary_roi_mask_set": "stroke_longitudinal",
        "mask_sets": {
            "stroke_longitudinal": {
                "enabled": True,
                "required": True,
                "whole_hemisphere_only": False,
                "require_assigned_experimental_group": True,
                "slot_policy": "per_stem",
                "min_stems": 2,
                "spatial_masks": ["homotopic", "global_healthy", "bilateral"],
                "thresholds": {},
                "slots": {
                    "baseline": {
                        "match": {
                            "exp_day": "Before-PT",
                            "state": "Awake",
                            "protocol": {"regex": "(?i)rest[-_ ]?state"},
                        }
                    },
                    "post_1_5h": {
                        "match": {
                            "exp_day": {"regex": "(?i)(?:1[\\.,]?5.*h|90.*min)"},
                            "state": "Awake",
                            "protocol": {"regex": "(?i)rest[-_ ]?state"},
                        }
                    },
                    "day1": {
                        "match": {
                            "exp_day": "1d-PT",
                            "state": "Awake",
                            "protocol": {"regex": "(?i)rest[-_ ]?state"},
                        }
                    },
                    "day3": {
                        "match": {
                            "exp_day": "3d-PT",
                            "state": "Awake",
                            "protocol": {"regex": "(?i)rest[-_ ]?state"},
                        }
                    },
                },
                "per_mouse_slot_overrides": {},
            },
            "prestroke_stability": {
                "enabled": True,
                "required": False,
                "whole_hemisphere_only": True,
                "require_assigned_experimental_group": False,
                "slot_policy": "all_slots",
                "spatial_masks": ["homotopic", "global_healthy", "bilateral"],
                "thresholds": {},
                "slots": {
                    "before": {
                        "match": {
                            "exp_day": "Before-PT",
                            "state": "Awake",
                            "protocol": {"regex": "(?i)rest[-_ ]?state"},
                        }
                    },
                    "daypt_awake": {
                        "match": {
                            "exp_day": "Day-PT",
                            "state": "Awake",
                            "protocol": {"regex": "(?i)rest[-_ ]?state"},
                        }
                    },
                },
            },
            "daypt_anesthesia": {
                "enabled": True,
                "required": False,
                "whole_hemisphere_only": True,
                "require_assigned_experimental_group": False,
                "slot_policy": "one_of",
                "expected_comparison_groups": ["Awake", "Iso"],
                "spatial_masks": ["homotopic", "global_healthy", "bilateral"],
                "thresholds": {},
                "slots": {
                    "daypt_awake": {
                        "match": {"exp_day": "Day-PT", "state": "Awake", "protocol": {"regex": "(?i)rest[-_ ]?state"}}
                    },
                    "daypt_iso": {
                        "match": {"exp_day": "Day-PT", "state": "Iso", "protocol": {"regex": "(?i)rest[-_ ]?state"}}
                    },
                },
            },
        },
    }


def deep_fill_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Insert missing nested defaults without overwriting user-specified values."""
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            deep_fill_missing(dst[key], value)
    return dst


def ensure_step6b_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Populate missing step6b_fc_geometry settings in the full configuration."""
    config.setdefault("step6b_fc_geometry", {})
    deep_fill_missing(config["step6b_fc_geometry"], default_step6b_config())
    return config


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively overwrite values from src into dst."""
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def thresholds_for_mask_set(config: dict[str, Any], mask_set: str) -> dict[str, Any]:
    """Return global ROI thresholds with optional mask-set-specific overrides."""
    base = copy.deepcopy(
        dict(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", default={}) or {})
    )
    local = dict(
        cfg_get(
            config,
            "step6b_fc_geometry",
            "mask_sets",
            str(mask_set),
            "thresholds",
            default={},
        )
        or {}
    )
    return deep_update(base, local)


def config_for_mask_set(config: dict[str, Any], mask_set: str) -> dict[str, Any]:
    """Copy config and materialize the selected mask-set thresholds as roi_thresholds."""
    out = copy.deepcopy(config)
    ensure_step6b_defaults(out)
    out["step6b_fc_geometry"]["roi_thresholds"] = thresholds_for_mask_set(config, mask_set)
    return out


# =============================================================================
# Paths
# =============================================================================


def build_step6b_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create and return output paths for the selected REST branch."""
    cfg = ensure_step6b_defaults(config)["step6b_fc_geometry"]
    variant = str(cfg.get("active_rest_variant", "without_rest"))
    if variant not in {"without_rest", "with_rest"}:
        raise ValueError(f"Invalid step6b rest variant: {variant!r}")
    root = Path(paths["out_root"]) / str(cfg.get("folder_name", "step6b_fc_geometry")) / variant
    tables = root / "tables"
    out = {
        "root": root,
        "canonical_root": root / "canonical_allen",
        "reference_root": root / "reference",
        "stem_inputs_root": root / "stem_inputs_refspace",
        "mask_sets_root": root / "mask_sets_refspace",
        "canonical_masks_root": root / "mask_sets_canonical",
        "cohort_canonical_root": root / "cohort_canonical",
        "qc_root": root / "qc",
        "qc_technical_root": root / "qc" / "technical",
        "qc_pathology_root": root / "qc" / "pathology",
        "qc_shadow_root": root / "qc" / "shadow",
        "qc_brain_edge_root": root / "qc" / "brain_edge",
        "qc_summary_root": root / "qc" / "summary",
        "qc_reference_repair_root": root / "qc" / "reference_atlas_repair",
        "tables_root": tables,
        "mapping_csv": tables / "allen_surface_to_analysis_roi.csv",
        "direct_children_csv": tables / "analysis_roi_definitions.csv",
        "roi_refinement_csv": tables / "roi_refinement_definitions.csv",
        "canonical_roi_csv": tables / "canonical_roi_side_areas.csv",
        "canonical_hole_repair_csv": tables / "canonical_internal_hole_repairs.csv",
        "reference_hole_repair_csv": tables / "reference_space_internal_hole_repairs.csv",
        "canonical_pair_csv": tables / "canonical_homotopic_pair_areas.csv",
        "selection_csv": tables / "mask_set_selection.csv",
        "excluded_stems_csv": tables / "analysis_exclusions_by_stem.csv",
        "exclusion_rules_csv": tables / "analysis_exclusion_rule_matches.csv",
        "rest_availability_csv": tables / "rest_temporal_eligibility.csv",
        "skipped_mask_sets_csv": tables / "skipped_mask_sets.csv",
        "mask_set_cohort_summary_csv": tables / "mask_set_cohort_summary.csv",
        # Human-readable unique-mouse funnel: ALL + actual comparison groups.
        "cohort_overview_csv": tables / "cohort_overview.csv",
        "final_roi_pass_summary_csv": tables / "final_roi_pass_summary.csv",
        "assignment_audit_csv": tables / "mouse_group_assignment_audit.csv",
        "stem_mask_log_csv": tables / "stem_mask_sources.csv",
        "hemisphere_csv": tables / "hemisphere_assignment.csv",
        "mask_set_summary_csv": tables / "mask_set_summary.csv",
        "side_consistency_csv": tables / "left_right_projection_consistency.csv",
        "cohort_frequency_summary_csv": tables / "cohort_frequency_roi_summary.csv",
        "side_technical_csv": tables / "roi_side_technical_coverage_by_mouse.csv",
        "homotopic_technical_csv": tables / "roi_homotopic_pair_coverage_by_mouse.csv",
        "global_technical_csv": tables / "roi_global_coverage_by_mouse.csv",
        # Backward-friendly key used by notebook loader; points to side table in v4.10.
        "technical_roi_csv": tables / "roi_side_technical_coverage_by_mouse.csv",
        "cohort_reference_roi_csv": tables / "roi_cohort_reference_coverage.csv",
        "side_stem_usable_csv": tables / "roi_side_stem_usable_coverage.csv",
        "homotopic_stem_usable_csv": tables / "roi_homotopic_stem_usable_coverage.csv",
        "global_stem_usable_csv": tables / "roi_global_stem_usable_coverage.csv",
        "stem_usable_roi_csv": tables / "roi_side_stem_usable_coverage.csv",
        "pathology_nonzero_csv": tables / "pathology_nonzero_cases.csv",
        "shadow_nonzero_csv": tables / "shadow_nonzero_cases.csv",
        "eligibility_csv": tables / "roi_analysis_eligibility.csv",
        "completeness_funnel_csv": tables / "roi_completeness_funnel.csv",
        "group_summary_csv": tables / "roi_group_eligibility_summary.csv",
        "selected_by_set_csv": tables / "selected_rois_by_mask_set.csv",
        "selected_longitudinal_csv": tables / "selected_rois_longitudinal.csv",
        "errors_csv": tables / "step6b_errors.csv",
    }
    for p in out.values():
        if p.suffix:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return out


def safe_key(*parts: Any) -> str:
    """Join identifier components into a filesystem-safe directory or file key."""
    text = "__".join(str(p) for p in parts)
    return text.replace("/", "-").replace("\\", "-").replace(":", "-")

# =============================================================================
# Small helpers
# =============================================================================


def safe_str(value: Any) -> str:
    """Convert a scalar to text, treating missing values as an empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def bool_series(s: pd.Series) -> pd.Series:
    """Interpret common textual Boolean values in a pandas series."""
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def resize_bool(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a spatial boolean mask using nearest-neighbor interpolation."""
    arr = np.asarray(mask, dtype=bool)
    if tuple(arr.shape) == tuple(shape):
        return arr
    h, w = int(shape[0]), int(shape[1])
    return cv2.resize(arr.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def load_bool_mask(path: Path, *, target_shape: tuple[int, int] | None = None) -> np.ndarray:
    """Read a two-dimensional mask and optionally resize it."""
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {arr.shape}: {path}")
    out = arr.astype(bool)
    return resize_bool(out, target_shape) if target_shape is not None else out


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows into a CSV table at the given path."""
    pd.DataFrame(rows).to_csv(path, index=False)


def experimental_group_map(config: dict[str, Any]) -> dict[str, str]:
    """Map each configured mouse ID to its experimental-group label."""
    groups = dict(cfg_get(config, "analysis_selection", "experimental_groups", default={}) or {})
    out: dict[str, str] = {}
    for group_name, members in groups.items():
        if isinstance(members, str):
            members = [members]
        for mouse in members or []:
            out[str(mouse)] = str(group_name)
    return out


def build_mouse_group_assignment_audit(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """One row per Step6 group showing whether the mouse has an analysis group."""
    exp_groups = experimental_group_map(config)
    unassigned_label = str(
        cfg_get(
            config,
            "step6b_fc_geometry",
            "selection",
            "unassigned_label",
            default="UNASSIGNED",
        )
    )
    rows: list[dict[str, Any]] = []
    for group_id, group in df.groupby(df["group_id"].astype(str), sort=False):
        mouse_id = identity_value(group)
        exp_group = exp_groups.get(mouse_id, unassigned_label)
        rows.append({
            "group_id": str(group_id),
            "mouse_id": mouse_id,
            "experimental_group": exp_group,
            "assigned": bool(exp_group != unassigned_label),
            "n_stems": int(len(group)),
        })
    return pd.DataFrame(rows)


def _comparison_group_is_eligible(comp: dict[str, Any], experimental_group: str) -> bool:
    """Optional design-level restriction of a comparison to experimental groups."""
    include = comp.get("experimental_groups")
    exclude = comp.get("exclude_experimental_groups")

    if include is not None:
        if isinstance(include, str):
            include = [include]
        if experimental_group not in {str(v) for v in (include or [])}:
            return False

    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [exclude]
        if experimental_group in {str(v) for v in (exclude or [])}:
            return False

    return True



# =============================================================================
# Comparison row selection
# =============================================================================


def _match_one(series: pd.Series, spec: Any) -> pd.Series:
    """Evaluate one flexible matching specification against a metadata column."""
    values = series.map(safe_str)
    if isinstance(spec, dict):
        mask = pd.Series(True, index=series.index)
        if "regex" in spec:
            mask &= values.str.contains(str(spec["regex"]), regex=True, na=False)
        if "include" in spec:
            raw = spec["include"]
            wanted = {safe_str(v) for v in (raw if isinstance(raw, (list, tuple, set)) else [raw])}
            mask &= values.isin(wanted)
        if "exclude" in spec:
            raw = spec["exclude"]
            unwanted = {safe_str(v) for v in (raw if isinstance(raw, (list, tuple, set)) else [raw])}
            mask &= ~values.isin(unwanted)
        if "equals" in spec:
            mask &= values.eq(safe_str(spec["equals"]))
        return mask
    if isinstance(spec, (list, tuple, set)):
        return values.isin({safe_str(v) for v in spec})
    return values.eq(safe_str(spec))


def apply_match(df: pd.DataFrame, match: dict[str, Any]) -> pd.DataFrame:
    """Select metadata rows satisfying all supplied column match conditions."""
    mask = pd.Series(True, index=df.index)
    for col, spec in (match or {}).items():
        if col not in df.columns:
            raise KeyError(f"Comparison match uses missing metadata column {col!r}")
        mask &= _match_one(df[col], spec)
    return df[mask].copy()


def identity_value(row_or_group: pd.Series | pd.DataFrame, preferred: str = "mouse_id") -> str:
    """Extract the configured comparison identity from a record or table."""
    if isinstance(row_or_group, pd.DataFrame):
        if row_or_group.empty:
            return ""
        row = row_or_group.iloc[0]
    else:
        row = row_or_group
    value = safe_str(row.get(preferred, ""))
    return value or safe_str(row.get("group_id", ""))


def slot_spec_for_mouse(comparison_cfg: dict[str, Any], slot_name: str, mouse_id: str, group_id: str) -> dict[str, Any]:
    """Resolve a temporal slot definition including mouse-specific overrides."""
    base = dict((comparison_cfg.get("slots", {}) or {}).get(slot_name, {}) or {})
    overrides = dict(comparison_cfg.get("per_mouse_slot_overrides", {}) or {})
    override_block = overrides.get(mouse_id, overrides.get(group_id, {})) or {}
    if slot_name in override_block:
        override = dict(override_block[slot_name] or {})
        merged = dict(base)
        # replace match as a complete unit unless explicitly absent
        if "match" in override:
            merged["match"] = dict(override["match"] or {})
        for k, v in override.items():
            if k != "match":
                merged[k] = v
        return merged
    return base


def load_rest_eligible_stems(config: dict[str, Any], paths: dict[str, Path]) -> set[str]:
    """Only authoritative Finalizer 60-s USE windows count for awake REST eligibility.

    Iso is handled by actual Step2 state and never needs behavioral REST QC.
    No fallback to legacy rest_segment_qc.csv or behavioral labels.
    """
    explicit = cfg_get(config, "step6b_fc_geometry", "rest_segments_final_csv", default=None)
    if explicit:
        candidate = resolve_declared_path(str(explicit), config)
    else:
        candidate = Path(paths["out_root"]) / "behavior_review" / "rest_segments_final.csv"
    if candidate is None or not candidate.is_file():
        raise FileNotFoundError(f"with_rest requires final REST segments CSV: {candidate}")
    segments = pd.read_csv(candidate)
    if "stem" not in segments.columns:
        raise KeyError(f"Final REST table lacks stem: {candidate}")
    if "duration_sec" in segments.columns:
        duration = pd.to_numeric(segments["duration_sec"], errors="coerce")
    elif {"t0_sec", "t1_sec"}.issubset(segments.columns):
        duration = pd.to_numeric(segments["t1_sec"], errors="coerce") - pd.to_numeric(segments["t0_sec"], errors="coerce")
    else:
        raise KeyError(f"Final REST table lacks duration_sec or t0_sec/t1_sec: {candidate}")
    required = float(cfg_get(config, "step6b_fc_geometry", "rest_min_duration_sec", default=60.0))
    if not (np.isfinite(required) and required > 0):
        raise ValueError("rest_min_duration_sec must be positive")
    # Finalizer guarantees that these are approved USE segments, not pending candidates.
    good = segments[duration.ge(required - 1e-6)]
    return set(good["stem"].astype(str))


def selection_group_complete(sg: pd.DataFrame, config: dict[str, Any]) -> bool:
    """A partial longitudinal series is valid; paired analyses are all-slots."""
    if sg.empty:
        return False
    mask_set = safe_str(sg.iloc[0]["mask_set"])
    comp = cfg_get(config, "step6b_fc_geometry", "mask_sets", mask_set, default={}) or {}
    policy = safe_str(comp.get("slot_policy", "all_slots")).lower()
    n_ok = int(sg["status"].eq("ok").sum())
    if sg["status"].isin(["ambiguous", "match_error"]).any():
        return False
    if policy == "per_stem":
        return n_ok >= int(comp.get("min_stems", 2))
    if policy == "one_of":
        return n_ok == 1  # two states for one mouse must not be intersected together
    if policy == "all_slots":
        return n_ok == len(sg)
    raise ValueError(f"Unknown slot_policy={policy!r} for {mask_set}")


def resolve_comparison_selection(
    df: pd.DataFrame,
    config: dict[str, Any],
    *,
    mask_sets: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    rest_variant: str = "without_rest",
    rest_eligible: set[str] | None = None,
) -> pd.DataFrame:
    """Select temporal comparison slots before constructing fixed technical masks.

    Parameters
    ----------
    df : pandas.DataFrame
        Step6/Step2 index after the configured scientific exclusions.
    config : dict
        Project YAML, including ``step6b_fc_geometry.mask_sets``.
    mask_sets, groups : sequence of str, optional
        Restrict the mask sets and registration groups to inspect.
    rest_variant : {"without_rest", "with_rest"}, optional
        Temporal eligibility branch.
    rest_eligible : set[str], optional
        Stems with accepted final REST segments in the ``with_rest`` branch.

    Returns
    -------
    pandas.DataFrame
        Selection records with comparison slots, source stems and selection status.

    Notes
    -----
    REST eligibility affects the temporal selection, not the definition of
    fixed technical-mask geometry. Existing scientific exclusions are applied
    before this function is invoked by ``run_step6b``."""
    if rest_variant not in {"without_rest", "with_rest"}:
        raise ValueError(f"Unknown rest variant: {rest_variant}")
    cfg = ensure_step6b_defaults(config)["step6b_fc_geometry"]
    comp_cfgs = dict(cfg.get("mask_sets", {}) or {})
    wanted_comp = set(map(str, mask_sets)) if mask_sets else None
    wanted_groups = set(map(str, groups)) if groups else None
    exp_groups = experimental_group_map(config)
    unassigned = safe_str(cfg_get(config, "step6b_fc_geometry", "selection", "unassigned_label", default="UNASSIGNED"))
    eligible = rest_eligible or set()
    records: list[dict[str, Any]] = []
    for group_id, group in df.groupby(df["group_id"].astype(str), sort=False):
        group = group.reset_index(drop=True)
        if wanted_groups is not None and str(group_id) not in wanted_groups:
            continue
        mouse_id = identity_value(group)
        exp_group = exp_groups.get(mouse_id, unassigned)
        for comp_name, comp in comp_cfgs.items():
            if not bool((comp or {}).get("enabled", True)) or (wanted_comp is not None and comp_name not in wanted_comp):
                continue
            if bool(comp.get("require_assigned_experimental_group", cfg["selection"].get("require_assigned_experimental_group", True))) and exp_group == unassigned:
                continue
            if not _comparison_group_is_eligible(comp or {}, exp_group):
                continue
            for slot_name in dict(comp.get("slots", {}) or {}):
                spec = slot_spec_for_mouse(comp, str(slot_name), mouse_id, str(group_id))
                match = dict(spec.get("match", {}) or {})
                match_error = ""
                try:
                    candidates = apply_match(group, match)
                    status = "ok" if len(candidates) == 1 else ("missing" if len(candidates) == 0 else "ambiguous")
                except Exception as exc:
                    candidates = group.iloc[0:0]
                    status, match_error = "match_error", repr(exc)
                if len(candidates) == 1:
                    r = candidates.iloc[0]
                    if bool(r.get("analysis_excluded", False)):
                        status = "analysis_excluded"
                    elif rest_variant == "with_rest" and safe_str(r.get("state")) == "Awake" and safe_str(r.get("stem")) not in eligible:
                        status = "no_rest"
                    elif rest_variant == "with_rest" and safe_str(r.get("state")) not in {"Awake", "Iso"}:
                        status = "unsupported_state"
                    # Iso is intentionally eligible without manual REST QC.
                base = {
                    "rest_variant": rest_variant,
                    "mask_set": str(comp_name),
                    "group_id": str(group_id),
                    "mouse_id": mouse_id,
                    "experimental_group": exp_group,
                    "comparison_group": exp_group if comp_name != "daypt_anesthesia" else safe_str(candidates.iloc[0].get("state")) if len(candidates)==1 else safe_str(match.get("state", "")),
                    "slot": str(slot_name),
                    "match_json": json.dumps(match, ensure_ascii=False),
                    "status": status,
                    "n_candidates": int(len(candidates)),
                    "candidate_stems": json.dumps(candidates["stem"].astype(str).tolist(), ensure_ascii=False) if "stem" in candidates else "[]",
                    "error": match_error,
                }
                if len(candidates) == 1:
                    for col in ["stem", "ref_stem", "exp_day", "state", "protocol", "exclusion_rule_ids", "exclusion_reason"]:
                        base[col] = safe_str(r.get(col, ""))
                else:
                    for col in ["stem", "ref_stem", "exp_day", "state", "protocol", "exclusion_rule_ids", "exclusion_reason"]:
                        base[col] = ""
                records.append(base)
    return pd.DataFrame(records, columns=["rest_variant", "mask_set", "group_id", "mouse_id", "experimental_group", "comparison_group", "slot", "match_json", "status", "n_candidates", "candidate_stems", "error", "stem", "ref_stem", "exp_day", "state", "protocol", "exclusion_rule_ids", "exclusion_reason"])


def apply_analysis_exclusions_to_step6b(df: pd.DataFrame, config: dict[str, Any], metadata_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shared exclusion rules are evaluated on the COMPLETE Step2-selected index.

    Step6 remains unaffected; excluded records are retained for diagnostic audits.
    """
    from analysis_exclusions import build_exclusion_registry
    registry = build_exclusion_registry(metadata_full, pd.DataFrame(), config)
    audit = registry["records"]
    fields = ["stem", "analysis_excluded", "exclusion_rule_ids", "exclusion_reasons", "exclusion_reason"]
    out = df.merge(audit[fields], on="stem", how="left", validate="one_to_one")
    out["analysis_excluded"] = out["analysis_excluded"].fillna(False).astype(bool)
    return out, audit, registry["rule_report"]


# =============================================================================
# Manual registration overrides / stem -> reference transform
# =============================================================================


def manual_registration_overrides_csv(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    """Locate the Step3 manual-registration override table."""
    manual_cfg = dict(cfg_get(config, "step3", "manual_registration", default={}) or {})
    folder = str(manual_cfg.get("folder_name", "manual_registration"))
    name = str(manual_cfg.get("overrides_csv_name", "manual_registration_overrides.csv"))
    return Path(paths["step3_root"]) / folder / name


def load_manual_registration_overrides(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    """Read the most recent applicable manual transforms per stem/group."""
    path = manual_registration_overrides_csv(config, paths)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "stem" not in df.columns:
        return pd.DataFrame()
    if "updated_at" in df.columns:
        df = df.sort_values("updated_at")
    keys = [c for c in ["group_id", "stem"] if c in df.columns]
    if keys:
        df = df.groupby(keys, sort=False).tail(1)
    else:
        df = df.groupby(df["stem"].astype(str), sort=False).tail(1)
    return df.reset_index(drop=True)


def finite_float(value: Any) -> float | None:
    """Convert a scalar to a finite float or return None."""
    try:
        x = float(value)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def effective_tform_stem_to_ref(
    row: pd.Series,
    manual_overrides: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[AffineTransform, str, dict[str, float]]:
    """Obtain the effective stem-to-reference transform and its provenance."""
    tform = registration_tform_stem_to_ref(row)
    stem = safe_str(row.get("stem"))
    ref = safe_str(row.get("ref_stem"))
    has_manual = str(row.get("step3_manual_override_present", "")).strip().lower() in {"true", "1"}
    source = "identity_reference" if stem == ref else ("step3_manual_final_transform" if has_manual else "step2_auto_transform")
    mat = np.asarray(tform.params, dtype=float)
    info = {"translation_x": float(mat[0,2]), "translation_y": float(mat[1,2]),
            "rotation_rad": float(np.arctan2(mat[1,0], mat[0,0])),
            "linear_det": float(np.linalg.det(mat[:2,:2]))}
    return tform, source, info


def warp_bool_stem_to_ref(mask: np.ndarray, tform: EuclideanTransform, ref_shape: tuple[int, int]) -> np.ndarray:
    """Warp a binary stem-space mask into the full-resolution reference grid.

    Parameters
    ----------
    mask : numpy.ndarray
        Two-dimensional stem-space mask, indexed by ``(y, x)``.
    tform : skimage.transform.EuclideanTransform
        Forward transform from stem to reference coordinates.
    ref_shape : tuple[int, int]
        Reference-grid shape ``(y, x)`` in pixels.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``ref_shape``.

    Notes
    -----
    Uses inverse mapping, nearest-neighbor interpolation, and zero fill;
    no floating-point mask values are retained."""
    out = warp(
        np.asarray(mask, dtype=np.float32),
        inverse_map=tform.inverse,
        output_shape=tuple(ref_shape),
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0,
    )
    return out > 0.5




# =============================================================================
# Technical / pathology mask sources
# =============================================================================


def manual_roi_review_root(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    """Locate the Step4 manual ROI review directory."""
    block = dict(cfg_get(config, "step4", "manual_roi_review", default={}) or {})
    return Path(paths["step4_root"]) / str(block.get("folder_name", "manual_roi_review"))


def default_rescue_csv(paths: dict[str, Path]) -> Path:
    """Return the default location of REST rescue-review decisions."""
    return Path(paths["out_root"]) / "behavior_review" / "rest_segment_rescue_60s" / "rest_segment_rescue_review.csv"


def rest_segments_final_candidates(config: dict[str, Any], paths: dict[str, Path]) -> list[Path]:
    """List declared and standard candidate paths for final REST segments."""
    custom = safe_str(cfg_get(config, "step6b_fc_geometry", "rest_segments_final_csv", default="")).strip()
    out: list[Path] = []
    if custom:
        p = resolve_declared_path(custom, config)
        if p is not None:
            out.append(p)
    out.extend([
        Path(paths["out_root"]) / "behavior_review" / "rest_segments_final.csv",
        Path(paths["out_root"]) / "rest_segments_final.csv",
    ])
    return out


def load_shadow_map(config: dict[str, Any], paths: dict[str, Path]) -> tuple[dict[str, str], str]:
    """One authoritative path per stem, including stems without accepted REST windows."""
    custom = cfg_get(config, "step6b_fc_geometry", "rest_shadow_masks_by_stem_csv", default=None)
    primary = resolve_declared_path(str(custom), config) if custom else Path(paths["out_root"]) / "behavior_review" / "rest_shadow_masks_by_stem.csv"
    if primary is None or not primary.exists():
        raise FileNotFoundError(f"Authoritative REST shadow mapping not found: {primary}")
    df = pd.read_csv(primary, dtype=object, keep_default_na=False, encoding="utf-8-sig")
    if "stem" not in df.columns or "shadow_mask_path" not in df.columns:
        raise KeyError(f"REST shadow mapping lacks stem/shadow_mask_path: {primary}")
    if df["stem"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate stem in authoritative shadow mapping: {primary}")
    return {safe_str(r["stem"]): safe_str(r["shadow_mask_path"]).strip() for _, r in df.iterrows() if safe_str(r["shadow_mask_path"]).strip()}, str(primary)


def resolve_declared_path(value: str, config: dict[str, Any]) -> Path | None:
    """Resolve a configured path using project-level path remapping."""
    text = safe_str(value).strip()
    if not text:
        return None
    path_cfg = get_path_resolving_config(config, "step6b")
    try:
        p = resolve_flexible_path(
            text,
            from_drive=path_cfg.get("from_drive"),
            to_drive=path_cfg.get("to_drive"),
            from_root=path_cfg.get("from_root"),
            to_root=path_cfg.get("to_root"),
            must_exist=False,
        )
        return Path(p) if p is not None else None
    except Exception:
        return Path(text)


@dataclass
class StemReferenceMasks:
    """Reference-space masks for one stem and their provenance.

    All spatial arrays are on the full-resolution Step6 reference grid
    with shape ``(y, x)``. Masks are separated by technical validity,
    REST rescue shadow, and stem-specific pathology."""
    brain_ref: np.ndarray
    technical_artifact_ref: np.ndarray
    bad_ref: np.ndarray
    bregma_ref: np.ndarray
    rescue_shadow_ref: np.ndarray
    pathology_ref: np.ndarray
    source_meta: dict[str, Any]


def build_stem_reference_masks(
    row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    ref_shape: tuple[int, int],
    manual_overrides: pd.DataFrame,
    shadow_map: dict[str, str],
    shadow_table_source: str,
) -> StemReferenceMasks:
    """Load technical and stem-specific masks into Step6 reference coordinates.

    Parameters
    ----------
    row : pandas.Series
        Step6 index row identifying the stem and its registration reference.
    config : dict
        Full project configuration, including ``step6b_fc_geometry.technical_sources``.
    paths : dict[str, pathlib.Path]
        Paths produced by the preceding pipeline stages.
    ref_shape : tuple[int, int]
        Reference image height and width in full-resolution pixels, ``(y, x)``.
    manual_overrides : pandas.DataFrame
        Step3 manual registration overrides, if available.
    shadow_map : dict[str, str]
        Stem-to-file mapping of REST rescue shadow masks.
    shadow_table_source : str
        Provenance of the shadow-mask mapping table.

    Returns
    -------
    StemReferenceMasks
        Boolean reference-grid masks ``(y, x)`` and their source metadata.

    Notes
    -----
    The fixed technical mask and stem-specific shadow/pathology masks have
    separate roles. Shadow and pathology are applied downstream per stem,
    not incorporated into the cross-stem fixed technical intersection."""
    stem = safe_str(row.get("stem"))
    tech_cfg = dict(cfg_get(config, "step6b_fc_geometry", "technical_sources", default={}) or {})
    strict_declared = bool(cfg_get(config, "step6b_fc_geometry", "strict_missing_declared_masks", default=True))

    cps = canonical_mask_paths(paths, stem)
    brain_path = cps["brain_manual"] if cps["brain_manual"].exists() else cps["brain_auto"]
    if not brain_path.exists():
        raise FileNotFoundError(f"Canonical brain mask missing for stem={stem}: {brain_path}")
    brain_native = load_bool_mask(brain_path)
    native_shape = tuple(brain_native.shape)

    bad_native = np.zeros(native_shape, dtype=bool)
    if bool(tech_cfg.get("bad_auto", True)):
        if not cps["bad_auto"].exists():
            raise FileNotFoundError(f"Canonical bad_auto mask missing for stem={stem}: {cps['bad_auto']}")
        bad_native = load_bool_mask(cps["bad_auto"], target_shape=native_shape)

    bregma_native = np.zeros(native_shape, dtype=bool)
    bregma_path = cps["bregma_manual"] if cps["bregma_manual"].exists() else cps["bregma_auto"]
    if bool(tech_cfg.get("bregma", True)):
        if not bregma_path.exists():
            raise FileNotFoundError(f"Canonical bregma mask missing for stem={stem}: {bregma_path}")
        bregma_native = load_bool_mask(bregma_path, target_shape=native_shape)

    if bool(tech_cfg.get("step4_shadow_union_manual", False)):
        p = manual_roi_review_root(config, paths) / stem / "shadow_union_manual.npy"
        if p.exists():
            bad_native |= load_bool_mask(p, target_shape=native_shape)

    rescue_native = np.zeros(native_shape, dtype=bool)
    rescue_declared = shadow_map.get(stem, "")
    rescue_resolved = ""
    if bool(tech_cfg.get("rescue_shadow", True)) and rescue_declared:
        p = resolve_declared_path(rescue_declared, config)
        if p is None or not p.exists():
            if strict_declared:
                raise FileNotFoundError(f"Residual shadow mask declared but missing for stem={stem}: {rescue_declared}")
        else:
            rescue_native = load_bool_mask(p, target_shape=native_shape)
            rescue_resolved = str(p)

    # Only the single canonical manual mask. Missing file denotes zero hemorrhage;
    # the source/missing status is explicit in per-stem audit table.
    hemorrhage_path = Path(paths["masks_root"]) / stem / "hemorrhage_manual.npy"
    hemorrhage_exists = hemorrhage_path.is_file()
    # Before-/Day-PT may be unreviewed; after the lesion no missing mask may
    # silently mean zero pathology. Reviewed absence is an explicit ZERO .npy.
    postpt_days = set(cfg_get(config, "hemorrhage_reference_review", "postpt_exp_days",
                              default=["1-5h-PT", "1d-PT", "3d-PT"]) or [])
    if not hemorrhage_exists and safe_str(row.get("exp_day")) in postpt_days and strict_declared:
        raise FileNotFoundError(f"Poststroke hemorrhage_manual.npy is missing (not confirmed zero): {hemorrhage_path}")
    pathology_native = load_bool_mask(hemorrhage_path, target_shape=native_shape) if hemorrhage_exists else np.zeros(native_shape, dtype=bool)
    pathology_sources = [str(hemorrhage_path)] if hemorrhage_exists else []

    technical_artifact_native = bad_native | bregma_native

    tform, registration_source, tvals = effective_tform_stem_to_ref(row, manual_overrides, config)
    brain_ref = warp_bool_stem_to_ref(brain_native, tform, ref_shape)
    bad_ref = warp_bool_stem_to_ref(bad_native, tform, ref_shape)
    bregma_ref = warp_bool_stem_to_ref(bregma_native, tform, ref_shape)
    rescue_ref = warp_bool_stem_to_ref(rescue_native, tform, ref_shape)
    pathology_ref = warp_bool_stem_to_ref(pathology_native, tform, ref_shape)
    technical_artifact_ref = bad_ref | bregma_ref

    meta = {
        "group_id": safe_str(row.get("group_id")),
        "mouse_id": safe_str(row.get("mouse_id")),
        "stem": stem,
        "ref_stem": safe_str(row.get("ref_stem")),
        "exp_day": safe_str(row.get("exp_day")),
        "state": safe_str(row.get("state")),
        "protocol": safe_str(row.get("protocol")),
        "brain_source": "manual" if brain_path.name == "brain_manual.npy" else "auto",
        "brain_path": str(brain_path),
        "bad_path": str(cps["bad_auto"]) if cps["bad_auto"].exists() else "",
        "bregma_path": str(bregma_path) if bregma_path.exists() else "",
        "shadow_table_source": shadow_table_source,
        "rescue_shadow_declared_path": rescue_declared,
        "rescue_shadow_resolved_path": rescue_resolved,
        "pathology_sources_json": json.dumps(pathology_sources, ensure_ascii=False),
        "hemorrhage_manual_path": str(hemorrhage_path),
        "hemorrhage_manual_exists": bool(hemorrhage_exists),
        "pathology_exists": bool(pathology_sources) and bool(pathology_native.any()),
        "registration_source": registration_source,
        **{f"registration_{k}": v for k, v in tvals.items()},
        "native_brain_px": int(brain_native.sum()),
        "native_bad_px": int(bad_native.sum()),
        "native_bregma_px": int(bregma_native.sum()),
        "native_rescue_shadow_px": int(rescue_native.sum()),
        "native_pathology_px": int(pathology_native.sum()),
        "ref_brain_px": int(brain_ref.sum()),
        "ref_technical_artifact_px": int(technical_artifact_ref.sum()),
        "ref_pathology_px": int(pathology_ref.sum()),
        "technical_formula": "common brain AND NOT(bad OR bregma); shadow/hemorrhage stem-specific",
    }
    return StemReferenceMasks(
        brain_ref=brain_ref,
        technical_artifact_ref=technical_artifact_ref,
        bad_ref=bad_ref,
        bregma_ref=bregma_ref,
        rescue_shadow_ref=rescue_ref,
        pathology_ref=pathology_ref,
        source_meta=meta,
    )

# =============================================================================
# Allen broad parent mapping
# =============================================================================


def structure_path_ids(meta: dict[str, Any]) -> list[int]:
    """Extract numeric Allen hierarchy IDs from structure metadata."""
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
        out: list[int] = []
        for x in path:
            try:
                out.append(int(x))
            except Exception:
                pass
        return out
    return []


def find_exact_structure_id(atlas_data: Any, query: str) -> int:
    """Resolve an exact Allen structure acronym or name to its ID."""
    q = str(query).strip().lower()
    hits: list[int] = []
    for sid_key, meta in atlas_data.structures.items():
        try:
            sid = int(sid_key)
        except Exception:
            continue
        acr = safe_str(meta.get("acronym", "")).strip().lower()
        name = safe_str(meta.get("name", "")).strip().lower()
        if q in {acr, name}:
            hits.append(sid)
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one Allen structure matching {query!r}, found {hits}")
    return hits[0]


def direct_children_of_structure(atlas_data: Any, parent_id: int) -> list[int]:
    """Find the direct descendants of an Allen ontology structure."""
    children: set[int] = set()
    for sid_key, meta in atlas_data.structures.items():
        try:
            sid = int(sid_key)
        except Exception:
            continue
        path = structure_path_ids(meta)
        if parent_id not in path:
            continue
        idx = path.index(parent_id)
        if idx + 1 < len(path):
            children.add(int(path[idx + 1]))
        elif sid != parent_id:
            # fallback for unusual metadata where path may omit self
            parent_field = meta.get("parent_structure_id", meta.get("parent_id"))
            try:
                if int(parent_field) == int(parent_id):
                    children.add(sid)
            except Exception:
                pass
    children.discard(int(parent_id))
    return sorted(children)


def broad_parent_for_id(atlas_data: Any, sid: int, parent_id: int, direct_children: set[int]) -> int:
    """Map an Allen structure ID to the matching broad cortical family."""
    sid = int(sid)
    if sid in direct_children:
        return sid
    meta = atlas_data.structures.get(sid, atlas_data.structures.get(str(sid), {})) or {}
    path = structure_path_ids(meta)
    if parent_id in path:
        idx = path.index(parent_id)
        if idx + 1 < len(path):
            child = int(path[idx + 1])
            if child in direct_children:
                return child
    return 0


def structure_is_descendant_or_self(atlas_data: Any, sid: int, ancestor_id: int) -> bool:
    """Return True when ``sid`` is ``ancestor_id`` or lies below it in Allen hierarchy."""
    sid = int(sid)
    ancestor_id = int(ancestor_id)
    if sid == ancestor_id:
        return True
    meta = atlas_data.structures.get(sid, atlas_data.structures.get(str(sid), {})) or {}
    return ancestor_id in structure_path_ids(meta)


def resolve_roi_refinements(
    atlas_data: Any,
    parent_id: int,
    direct_children: Sequence[int],
    config: dict[str, Any] | None,
) -> tuple[dict[int, dict[str, Any]], pd.DataFrame]:
    """Resolve optional selective refinement definitions from config.

    The config is intentionally local to Step6b analytical ROIs.  Families not
    listed in ``roi_refinements`` remain at the direct-child-of-Isocortex level.

    Example::

        broad_atlas:
          roi_refinements:
            MO:
              mode: split
              children: [MOp, MOs]
              unmatched_policy: error

    ``children`` may contain any Allen descendants of the selected family; they
    do not need to be literal direct children.  ``unmatched_policy`` controls
    pixels belonging to the broad family but to none of the requested targets:
    ``error`` (recommended for an intended complete split), ``keep_parent``, or
    ``drop``.
    """
    refinements_cfg = dict(
        cfg_get(
            config or {},
            "step6b_fc_geometry",
            "broad_atlas",
            "roi_refinements",
            default={},
        )
        or {}
    )
    direct_set = {int(x) for x in direct_children}
    resolved: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for family_query, raw_block in refinements_cfg.items():
        block = dict(raw_block or {})
        mode = safe_str(block.get("mode", "split")).strip().lower()
        if mode in {"none", "off", "disabled"}:
            continue
        if mode != "split":
            raise ValueError(
                f"ROI refinement {family_query!r}: only mode='split' is supported, got {mode!r}"
            )

        family_id = find_exact_structure_id(atlas_data, safe_str(family_query))
        if family_id not in direct_set:
            raise ValueError(
                f"ROI refinement parent {family_query!r} (Allen id={family_id}) is not a direct "
                "child of the configured broad parent. Select a broad family such as MO/SS/VIS."
            )

        target_queries = block.get("children", block.get("subregions", []))
        if isinstance(target_queries, str):
            target_queries = [target_queries]
        target_queries = list(target_queries or [])
        if not target_queries:
            raise ValueError(f"ROI refinement {family_query!r} has no children/subregions")

        target_ids: list[int] = []
        for q in target_queries:
            tid = find_exact_structure_id(atlas_data, safe_str(q))
            if tid == family_id or not structure_is_descendant_or_self(atlas_data, tid, family_id):
                raise ValueError(
                    f"ROI refinement target {q!r} (Allen id={tid}) is not a descendant of "
                    f"{family_query!r} (Allen id={family_id})"
                )
            target_ids.append(int(tid))

        if len(set(target_ids)) != len(target_ids):
            raise ValueError(f"ROI refinement {family_query!r} contains duplicate targets")

        # Overlapping targets (one target is an ancestor of another) would make
        # assignment ambiguous.  Require disjoint branches.
        for i, a in enumerate(target_ids):
            for b in target_ids[i + 1:]:
                if structure_is_descendant_or_self(atlas_data, a, b) or structure_is_descendant_or_self(atlas_data, b, a):
                    raise ValueError(
                        f"ROI refinement {family_query!r} has overlapping targets "
                        f"{atlas_data.id_to_acronym.get(a, a)!r} and {atlas_data.id_to_acronym.get(b, b)!r}"
                    )

        unmatched_policy = safe_str(block.get("unmatched_policy", "error")).strip().lower()
        if unmatched_policy not in {"error", "keep_parent", "drop"}:
            raise ValueError(
                f"ROI refinement {family_query!r}: unmatched_policy must be "
                f"error/keep_parent/drop, got {unmatched_policy!r}"
            )

        resolved[family_id] = {
            "family_query": safe_str(family_query),
            "family_id": int(family_id),
            "target_ids": target_ids,
            "unmatched_policy": unmatched_policy,
        }

        for tid in target_ids:
            rows.append({
                "broad_family_structure_id": int(family_id),
                "broad_family_acronym": atlas_data.id_to_acronym.get(family_id, safe_str(family_query)),
                "broad_family_name": atlas_data.id_to_name.get(family_id, safe_str(family_query)),
                "analysis_structure_id": int(tid),
                "analysis_roi_acronym": atlas_data.id_to_acronym.get(tid, str(tid)),
                "analysis_roi_name": atlas_data.id_to_name.get(tid, str(tid)),
                "refinement_mode": "split",
                "unmatched_policy": unmatched_policy,
            })

    return resolved, pd.DataFrame(rows)


def analysis_roi_for_structure(
    atlas_data: Any,
    sid: int,
    family_id: int,
    refinement: dict[str, Any] | None,
) -> tuple[int, str]:
    """Map one original Allen structure to the configured analytical ROI id."""
    if family_id <= 0:
        return 0, "outside_parent"
    if refinement is None:
        return int(family_id), "broad_family"

    matches = [
        int(tid)
        for tid in refinement["target_ids"]
        if structure_is_descendant_or_self(atlas_data, int(sid), int(tid))
    ]
    if len(matches) == 1:
        return int(matches[0]), "refined_target"
    if len(matches) > 1:
        raise RuntimeError(
            f"Allen structure id={sid} matched multiple refinement targets {matches}; "
            "refinement targets must be non-overlapping"
        )

    policy = safe_str(refinement.get("unmatched_policy", "error")).lower()
    if policy == "keep_parent":
        return int(family_id), "refinement_unmatched_keep_parent"
    if policy == "drop":
        return 0, "refinement_unmatched_drop"
    return -1, "refinement_unmatched_error"


def build_broad_labelmap(
    original_labelmap: np.ndarray,
    atlas_data: Any,
    parent_query: str,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Build the analytical Step6b ROI labelmap.

    Backward compatibility: when no ``roi_refinements`` are configured this is
    identical to the previous broad direct-child-of-Isocortex recoding.

    With selective refinement, only explicitly selected broad families are
    replaced by the requested Allen descendants.  The returned numerical labels
    are always genuine Allen structure IDs.  Historical ``broad_*`` columns are
    kept as aliases for the *analysis ROI* so downstream Step6b code remains
    compatible; explicit ``broad_family_*`` and ``analysis_roi_*`` columns make
    the hierarchy unambiguous in tables.
    """
    parent_id = find_exact_structure_id(atlas_data, parent_query)
    families = direct_children_of_structure(atlas_data, parent_id)
    family_set = set(families)
    refinements, refinement_table = resolve_roi_refinements(
        atlas_data, parent_id, families, config
    )

    analysis = np.zeros_like(original_labelmap, dtype=np.int32)
    records: list[dict[str, Any]] = []
    encountered_analysis_ids: set[int] = set()
    unmatched_error_records: list[dict[str, Any]] = []

    for sid in sorted(int(x) for x in np.unique(original_labelmap) if int(x) > 0):
        family_id = broad_parent_for_id(atlas_data, sid, parent_id, family_set)
        refinement = refinements.get(int(family_id)) if family_id > 0 else None
        aid, assignment_status = analysis_roi_for_structure(
            atlas_data, sid, family_id, refinement
        )

        if aid == -1:
            unmatched_error_records.append({
                "atlas_structure_id": int(sid),
                "acronym": atlas_data.id_to_acronym.get(sid, str(sid)),
                "name": atlas_data.id_to_name.get(sid, str(sid)),
                "broad_family_structure_id": int(family_id),
                "broad_family_acronym": atlas_data.id_to_acronym.get(family_id, ""),
                "n_pixels": int((original_labelmap == sid).sum()),
            })
            aid_for_table = 0
        else:
            aid_for_table = int(aid)
            if aid > 0:
                analysis[original_labelmap == sid] = int(aid)
                encountered_analysis_ids.add(int(aid))

        records.append({
            "atlas_structure_id": int(sid),
            "acronym": atlas_data.id_to_acronym.get(sid, str(sid)),
            "name": atlas_data.id_to_name.get(sid, str(sid)),
            "is_isocortex_descendant": bool(family_id > 0),
            "broad_family_structure_id": int(family_id),
            "broad_family_acronym": atlas_data.id_to_acronym.get(family_id, "") if family_id > 0 else "",
            "broad_family_name": atlas_data.id_to_name.get(family_id, "") if family_id > 0 else "",
            "analysis_structure_id": int(aid_for_table),
            "analysis_roi_acronym": atlas_data.id_to_acronym.get(aid_for_table, "") if aid_for_table > 0 else "",
            "analysis_roi_name": atlas_data.id_to_name.get(aid_for_table, "") if aid_for_table > 0 else "",
            "roi_refinement_applied": bool(refinement is not None),
            "roi_assignment_status": assignment_status,
            # Backward-compatible aliases used by downstream Step6b code.
            "broad_structure_id": int(aid_for_table),
            "broad_acronym": atlas_data.id_to_acronym.get(aid_for_table, "") if aid_for_table > 0 else "",
            "broad_name": atlas_data.id_to_name.get(aid_for_table, "") if aid_for_table > 0 else "",
        })

    if unmatched_error_records:
        detail = pd.DataFrame(unmatched_error_records)
        raise RuntimeError(
            "Selective ROI refinement left Allen surface pixels unmatched while "
            "unmatched_policy='error'. Either add the missing target(s), or set "
            "unmatched_policy: keep_parent/drop. Unmatched structures:\n"
            + detail.to_string(index=False)
        )

    # Definition table for every analytical ROI actually present.  This includes
    # untouched broad families and any optional keep-parent residual.
    definition_rows: list[dict[str, Any]] = []
    mapping_df = pd.DataFrame(records)
    for aid in sorted(encountered_analysis_ids):
        sub = mapping_df[mapping_df["analysis_structure_id"].eq(aid)]
        family_ids = sorted(int(x) for x in sub["broad_family_structure_id"].dropna().unique() if int(x) > 0)
        if len(family_ids) != 1:
            raise RuntimeError(f"Analysis ROI id={aid} maps to unexpected broad families {family_ids}")
        fid = family_ids[0]
        is_refined = bool(fid in refinements and aid != fid)
        definition_rows.append({
            # Backward-compatible aliases = analysis ROI.
            "broad_structure_id": int(aid),
            "broad_acronym": atlas_data.id_to_acronym.get(aid, str(aid)),
            "broad_name": atlas_data.id_to_name.get(aid, str(aid)),
            # Explicit hierarchy.
            "analysis_structure_id": int(aid),
            "analysis_roi_acronym": atlas_data.id_to_acronym.get(aid, str(aid)),
            "analysis_roi_name": atlas_data.id_to_name.get(aid, str(aid)),
            "broad_family_structure_id": int(fid),
            "broad_family_acronym": atlas_data.id_to_acronym.get(fid, str(fid)),
            "broad_family_name": atlas_data.id_to_name.get(fid, str(fid)),
            "roi_refinement_applied": is_refined,
            "roi_definition_source": "refined_target" if is_refined else "broad_family",
            "parent_structure_id": int(parent_id),
            "parent_acronym": atlas_data.id_to_acronym.get(parent_id, parent_query),
        })

    definition_table = pd.DataFrame(definition_rows)
    if not refinement_table.empty:
        # Attach for callers that want the fully declared config, including
        # targets absent from a particular labelmap. Stored in attrs only; the
        # canonical caller saves a separate table below.
        definition_table.attrs["roi_refinement_table"] = refinement_table

    return analysis, mapping_df, definition_table


# =============================================================================
# Broad-ROI boundary buffer
# =============================================================================


def buffered_broad_labelmap(
    broad_labelmap: np.ndarray,
    boundary_exclusion_px: float,
) -> np.ndarray:
    """Erode each broad Allen ROI inward by the configured reference-space pixels.

    Pixels removed at boundaries become label 0.  The operation is performed
    once in Step6b reference space. Step7 should resize this buffered labelmap
    rather than eroding again after downsampling.
    """
    radius = int(round(float(boundary_exclusion_px)))
    src = np.asarray(broad_labelmap, dtype=np.int32)
    if radius <= 0:
        return src.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    out = np.zeros(src.shape, dtype=np.int32)
    for bid in sorted(int(v) for v in np.unique(src) if int(v) > 0):
        mask = (src == bid).astype(np.uint8)
        eroded = cv2.erode(
            mask,
            kernel,
            iterations=1,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        out[eroded] = bid
    return out


# =============================================================================
# AP/midline reflection and paired homotopic masks
# =============================================================================


def reference_ml_coordinate_px(shape: tuple[int, int], basis: dict[str, Any]) -> np.ndarray:
    """Project reference-image pixels onto the annotated mediolateral axis.

    Parameters
    ----------
    shape : tuple[int, int]
        Image dimensions in pixels as ``(y, x)``.
    basis : dict
        Reference anatomical basis containing ``origin_xy`` and ``ml_hat_xy``.

    Returns
    -------
    numpy.ndarray
        Signed mediolateral coordinates, shape ``(y, x)``, in reference pixels.

    Notes
    -----
    The anatomical basis stores coordinate vectors in ``(x, y)`` order while
    the returned array is indexed by ``(y, x)``."""
    h, w = map(int, shape)
    yy, xx = np.indices((h, w), dtype=np.float64)
    origin = np.asarray(basis["origin_xy"], dtype=float)
    ml_hat = np.asarray(basis["ml_hat_xy"], dtype=float)
    return (xx - origin[0]) * ml_hat[0] + (yy - origin[1]) * ml_hat[1]


def mirror_partner_indices(shape: tuple[int, int], basis: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest integer partner after reflection around the annotated AP/midline axis."""
    h, w = map(int, shape)
    yy, xx = np.indices((h, w), dtype=np.float64)
    origin = np.asarray(basis["origin_xy"], dtype=float)
    ml_hat = np.asarray(basis["ml_hat_xy"], dtype=float)
    ml = (xx - origin[0]) * ml_hat[0] + (yy - origin[1]) * ml_hat[1]
    xr = xx - 2.0 * ml * ml_hat[0]
    yr = yy - 2.0 * ml * ml_hat[1]
    xi = np.rint(xr).astype(np.int32)
    yi = np.rint(yr).astype(np.int32)
    inside = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
    yi_safe = np.clip(yi, 0, h - 1)
    xi_safe = np.clip(xi, 0, w - 1)
    yi_safe[~inside] = -1
    xi_safe[~inside] = -1
    return yi_safe, xi_safe, inside


def sample_partner(arr: np.ndarray, partner_y: np.ndarray, partner_x: np.ndarray, inside: np.ndarray, fill: int | bool = 0) -> np.ndarray:
    """Sample a spatial array at precomputed mirrored-partner coordinates."""
    out = np.full(arr.shape, fill, dtype=arr.dtype)
    valid = np.asarray(inside, dtype=bool)
    out[valid] = arr[partner_y[valid], partner_x[valid]]
    return out


def build_homotopic_masks(
    common_valid: np.ndarray,
    broad_labelmap: np.ndarray,
    basis: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Construct reference-grid mirrored-partner masks for technical QC.

    Parameters
    ----------
    common_valid : numpy.ndarray
        Boolean mask of pixels valid across the comparison, shape ``(y, x)``.
    broad_labelmap : numpy.ndarray
        Analytical ROI labels on the same reference grid.
    basis : dict
        Anterior-posterior and mediolateral anatomical reference basis.

    Returns
    -------
    dict[str, numpy.ndarray]
        Boolean masks and mirrored-partner coordinate arrays.

    Notes
    -----
    Partner indices are derived from reflection about the annotated
    midline; the same anatomical ROI label must be valid on both sides."""
    py, px, inside = mirror_partner_indices(common_valid.shape, basis)
    partner_valid = sample_partner(common_valid.astype(bool), py, px, inside, fill=False).astype(bool)
    partner_broad = sample_partner(broad_labelmap.astype(np.int32), py, px, inside, fill=0).astype(np.int32)
    paired_geometry = common_valid.astype(bool) & inside & partner_valid
    # Atlas-only denominator for homotopic ROI coverage: pixels whose mirrored
    # partner exists and belongs to the same broad Allen region, irrespective of
    # experimental brain/artifact masks.  This separates atlas asymmetry/FOV from
    # mask-driven loss of valid pixels.
    atlas_broad_pairable = inside & (broad_labelmap > 0) & (partner_broad == broad_labelmap)
    paired_broad = paired_geometry & atlas_broad_pairable
    return {
        "partner_y": py,
        "partner_x": px,
        "partner_inside": inside,
        "partner_broad": partner_broad,
        "atlas_broad_pairable": atlas_broad_pairable,
        "paired_geometry": paired_geometry,
        "paired_broad": paired_broad,
    }


def hemisphere_masks(shape: tuple[int, int], basis: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute left and right reference masks from the annotated midline."""
    ml = reference_ml_coordinate_px(shape, basis)

    override = cfg_get(
        config, "step6b_fc_geometry", "hemisphere", "ml_positive_label", default=None
    )
    if override is None or safe_str(override).strip().lower() in {"", "none", "null"}:
        positive_label = str(
            cfg_get(config, "step6", "side_detection", "ml_positive_label", default="right")
        ).strip().lower()
    else:
        positive_label = safe_str(override).strip().lower()

    if positive_label not in {"left", "right"}:
        raise ValueError(
            f"Step6b hemisphere.ml_positive_label must be left/right/null, got {positive_label!r}"
        )

    pos = ml > 0
    neg = ml < 0
    if positive_label == "left":
        left, right = pos, neg
    else:
        left, right = neg, pos
    midline = ~(left | right)
    return left, right, midline





def lesion_and_healthy_side(mouse_id: str, group_id: str, config: dict[str, Any]) -> tuple[str, str]:
    """Resolve lesion and contralateral sides for a mouse/group."""
    block = dict(cfg_get(config, "step6b_fc_geometry", "hemisphere", default={}) or {})
    per_mouse = dict(block.get("per_mouse_lesion_side", {}) or {})
    lesion = safe_str(per_mouse.get(mouse_id, per_mouse.get(group_id, block.get("lesion_side", "right")))).strip().lower()
    if lesion not in {"left", "right"}:
        raise ValueError(f"lesion_side must be left/right, got {lesion!r} for mouse={mouse_id}")
    healthy = "left" if lesion == "right" else "right"
    return lesion, healthy


def analysis_hemisphere_masks(
    shape: tuple[int, int],
    basis: dict[str, Any],
    config: dict[str, Any],
    mouse_id: str,
    group_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Resolve anatomical hemisphere masks and lesion laterality for one mouse.

    Parameters
    ----------
    shape : tuple[int, int]
        Reference image dimensions ``(y, x)`` in pixels.
    basis : dict
        Anatomical coordinate basis from the reference annotation.
    config : dict
        Project configuration, including Step6b hemisphere settings.
    mouse_id, group_id : str
        Mouse and registration group identifiers.

    Returns
    -------
    tuple
        Healthy-side mask, lesion-side mask, midline exclusion mask,
        lesion-side label, and healthy-side label, in that order. Spatial masks
        have reference-grid shape ``(y, x)``.

    Notes
    -----
    The ``midline_exclusion_px`` setting is expressed in reference pixels.
    Mouse-specific lesion-side overrides are read from the YAML settings."""
    left, right, midline = hemisphere_masks(shape, basis, config)
    margin = float(cfg_get(config, "step6b_fc_geometry", "hemisphere", "midline_exclusion_px", default=0.5))
    if margin > 0:
        ml = np.abs(reference_ml_coordinate_px(shape, basis))
        band = ml <= margin
        left = left & (~band)
        right = right & (~band)
        midline = midline | band
    lesion_side, healthy_side = lesion_and_healthy_side(mouse_id, group_id, config)
    healthy = left if healthy_side == "left" else right
    lesion = right if lesion_side == "right" else left
    return healthy, lesion, midline, lesion_side, healthy_side



def choose_hemisphere_mask(
    choice: str,
    *,
    healthy_mask: np.ndarray,
    lesion_mask: np.ndarray,
    healthy_side: str,
    lesion_side: str,
) -> tuple[np.ndarray, str]:
    """Resolve biological/anatomical hemisphere choice to a boolean mask."""
    choice = safe_str(choice).strip().lower()
    if choice == "healthy":
        return healthy_mask.astype(bool), f"healthy_{healthy_side}"
    if choice == "lesion":
        return lesion_mask.astype(bool), f"lesion_{lesion_side}"
    if choice == healthy_side:
        return healthy_mask.astype(bool), choice
    if choice == lesion_side:
        return lesion_mask.astype(bool), choice
    if choice in {"left", "right"}:
        # left/right must correspond to either healthy or lesion in this experiment.
        if choice == healthy_side:
            return healthy_mask.astype(bool), choice
        return lesion_mask.astype(bool), choice
    raise ValueError(
        f"Hemisphere choice must be lesion/healthy/left/right, got {choice!r}"
    )


def analysis_role_hemisphere_masks(
    *,
    healthy_mask: np.ndarray,
    lesion_mask: np.ndarray,
    healthy_side: str,
    lesion_side: str,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Resolve which hemisphere is used for homotopic display and global analysis."""
    hchoice = cfg_get(
        config,
        "step6b_fc_geometry",
        "hemisphere",
        "homotopic_map_hemisphere",
        default="lesion",
    )
    gchoice = cfg_get(
        config,
        "step6b_fc_geometry",
        "hemisphere",
        "global_analysis_hemisphere",
        default="healthy",
    )
    hom_mask, hom_label = choose_hemisphere_mask(
        hchoice,
        healthy_mask=healthy_mask,
        lesion_mask=lesion_mask,
        healthy_side=healthy_side,
        lesion_side=lesion_side,
    )
    global_mask, global_label = choose_hemisphere_mask(
        gchoice,
        healthy_mask=healthy_mask,
        lesion_mask=lesion_mask,
        healthy_side=healthy_side,
        lesion_side=lesion_side,
    )
    return hom_mask, global_mask, hom_label, global_label


def atlas_and_technical_pairable_map_side(
    broad_labelmap: np.ndarray,
    technical_mask: np.ndarray,
    homotopic_map_mask: np.ndarray,
    basis: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Pairable homotopic pixels represented on the configured map hemisphere."""
    py, px, inside = mirror_partner_indices(broad_labelmap.shape, basis)
    partner_broad = sample_partner(
        broad_labelmap.astype(np.int32), py, px, inside, fill=0
    ).astype(np.int32)
    partner_technical = sample_partner(
        technical_mask.astype(bool), py, px, inside, fill=False
    ).astype(bool)

    atlas_pairable = (
        homotopic_map_mask.astype(bool)
        & inside
        & (broad_labelmap > 0)
        & (partner_broad == broad_labelmap)
    )
    technical_pairable = (
        atlas_pairable
        & technical_mask.astype(bool)
        & partner_technical
    )
    return {
        "partner_y": py,
        "partner_x": px,
        "partner_inside": inside,
        "partner_broad": partner_broad,
        "atlas_pairable_map": atlas_pairable,
        "technical_pairable_map": technical_pairable,
    }


def pathology_adjusted_pairable_map_side(
    technical_pairable_map: np.ndarray,
    pathology_ref: np.ndarray,
    partner_y: np.ndarray,
    partner_x: np.ndarray,
    partner_inside: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude pathological pixels and their reflected partners from pairability."""
    partner_pathology = sample_partner(pathology_ref.astype(bool), partner_y, partner_x, partner_inside, fill=False).astype(bool)
    pathology_hits_pair = pathology_ref.astype(bool) | partner_pathology
    usable = technical_pairable_map.astype(bool) & (~pathology_hits_pair)
    lost = technical_pairable_map.astype(bool) & pathology_hits_pair
    return usable, lost


# =============================================================================
# QC plots
# =============================================================================


def compact_label_display(labelmap: np.ndarray) -> np.ndarray:
    """Map nonzero atlas labels to consecutive integers for plotting."""
    labels = np.asarray(labelmap, dtype=np.int32)
    out = np.zeros(labels.shape, dtype=np.int16)
    for i, sid in enumerate(sorted(int(x) for x in np.unique(labels) if int(x) > 0), start=1):
        out[labels == sid] = i
    return out


def _imshow_mask(ax, mask: np.ndarray, title: str) -> None:
    """Draw a binary spatial mask on a Matplotlib axis."""
    ax.imshow(mask.astype(float), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")


def save_mask_set_qc_png(
    path: Path,
    reference_image: np.ndarray,
    broad_labelmap: np.ndarray,
    common_brain: np.ndarray,
    technical_artifact: np.ndarray,
    technical_mask: np.ndarray,
    global_analysis_mask: np.ndarray,
    technical_pairable: np.ndarray,
    *,
    title: str,
) -> None:
    """Save a QC figure showing reference geometry and fixed technical masks."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes[0, 0].imshow(reference_image, cmap="gray")
    axes[0, 0].set_title("reference")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(compact_label_display(broad_labelmap), interpolation="nearest")
    axes[0, 1].set_title("analysis ROI atlas")
    axes[0, 1].axis("off")
    _imshow_mask(axes[0, 2], common_brain, "common brain")
    _imshow_mask(axes[0, 3], technical_artifact, "technical artifact OR")
    _imshow_mask(axes[1, 0], technical_mask, "technical mask")
    axes[1, 1].imshow(np.where(technical_mask, compact_label_display(broad_labelmap), 0), interpolation="nearest")
    axes[1, 1].set_title("holey analysis ROI atlas")
    axes[1, 1].axis("off")
    _imshow_mask(axes[1, 2], global_analysis_mask & technical_mask, "global analysis technical")
    _imshow_mask(axes[1, 3], technical_pairable, "homotopic technical pairs")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_common_brain_edge_qc_png(
    path: Path,
    reference_image: np.ndarray,
    common_brain: np.ndarray,
    common_brain_for_technical: np.ndarray,
    excluded_edge: np.ndarray,
    fixed_technical_ref: np.ndarray,
    *,
    title: str,
) -> None:
    """QC for the v4.12 common-brain edge exclusion."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(reference_image, cmap="gray")
    axes[0].set_title("reference")
    _imshow_mask(axes[1], common_brain, "common brain (v4.10 input)")
    _imshow_mask(axes[2], excluded_edge, "edge excluded")
    _imshow_mask(axes[3], common_brain_for_technical, "common brain after erosion")
    _imshow_mask(axes[4], fixed_technical_ref, "fixed technical after artifacts")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_stem_pathology_qc_png(
    path: Path,
    reference_image: np.ndarray,
    pathology: np.ndarray,
    usable: np.ndarray,
    technical_pairable: np.ndarray,
    usable_pairable: np.ndarray,
    technical_global: np.ndarray,
    usable_global: np.ndarray,
    *,
    title: str,
) -> None:
    """Save a QC figure of stem-specific pathology and usable paired pixels."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes[0, 0].imshow(reference_image, cmap="gray")
    axes[0, 0].set_title("reference")
    axes[0, 0].axis("off")
    _imshow_mask(axes[0, 1], pathology, "pathology / hemorrhage")
    _imshow_mask(axes[0, 2], usable, "usable after pathology")
    _imshow_mask(axes[0, 3], technical_pairable, "technical homotopic")
    _imshow_mask(axes[1, 0], usable_pairable, "usable homotopic")
    _imshow_mask(axes[1, 1], technical_pairable & (~usable_pairable), "lost homotopic pairs")
    _imshow_mask(axes[1, 2], technical_global, "technical global analysis")
    _imshow_mask(axes[1, 3], usable_global, "usable global analysis")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Coverage tables / ROI eligibility
# =============================================================================


def child_meta_map(child_table: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Metadata keyed by analytical ROI Allen structure id."""
    out: dict[int, dict[str, Any]] = {}
    for _, r in child_table.iterrows():
        aid = int(r["broad_structure_id"])
        out[aid] = {
            "broad_acronym": safe_str(r.get("broad_acronym", "")),
            "broad_name": safe_str(r.get("broad_name", "")),
            "analysis_structure_id": int(r.get("analysis_structure_id", aid)),
            "analysis_roi_acronym": safe_str(r.get("analysis_roi_acronym", r.get("broad_acronym", ""))),
            "analysis_roi_name": safe_str(r.get("analysis_roi_name", r.get("broad_name", ""))),
            "broad_family_structure_id": int(r.get("broad_family_structure_id", aid)),
            "broad_family_acronym": safe_str(r.get("broad_family_acronym", r.get("broad_acronym", ""))),
            "broad_family_name": safe_str(r.get("broad_family_name", r.get("broad_name", ""))),
            "roi_refinement_applied": bool(r.get("roi_refinement_applied", False)),
            "roi_definition_source": safe_str(r.get("roi_definition_source", "broad_family")),
        }
    return out


def safe_fraction(num: int, den: int) -> float:
    """Compute a ratio, returning NaN for a nonpositive denominator."""
    return float(num / den) if den > 0 else np.nan


def analysis_roi_metadata(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Explicit analytical-ROI hierarchy fields for human-facing tables."""
    getter = row.get
    aid = int(getter("analysis_structure_id", getter("broad_structure_id", 0)) or 0)
    fid = int(getter("broad_family_structure_id", aid) or aid)
    return {
        "analysis_structure_id": aid,
        "analysis_roi_acronym": safe_str(getter("analysis_roi_acronym", getter("broad_acronym", ""))),
        "analysis_roi_name": safe_str(getter("analysis_roi_name", getter("broad_name", ""))),
        "broad_family_structure_id": fid,
        "broad_family_acronym": safe_str(getter("broad_family_acronym", getter("broad_acronym", ""))),
        "broad_family_name": safe_str(getter("broad_family_name", getter("broad_name", ""))),
        "roi_refinement_applied": bool(getter("roi_refinement_applied", False)),
        "roi_definition_source": safe_str(getter("roi_definition_source", "broad_family")),
    }


# Legacy pre-v4.3 aggregate coverage/completeness helpers were removed in v4.10.
# The active implementation below uses side-specific canonical coverage,
# pair-specific homotopic coverage, and stem-specific pointwise eligibility.


def build_skipped_mask_sets_table(
    selection: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """One row per incomplete mouse × mask_set.

    Missing/ambiguous required slots are analysis exclusions, not a reason to
    abort the rest of the batch. We never build a common mask from a partial
    required mask set.
    """
    columns = [
        "mask_set", "group_id", "mouse_id", "experimental_group",
        "mask_set_required", "n_slots", "n_ok", "n_unresolved",
        "unresolved_slots_json", "unresolved_status_json",
        "resolved_stems_json", "skip_reason",
    ]
    if selection.empty:
        return pd.DataFrame(columns=columns)

    cfg_sets = dict(cfg_get(config, "step6b_fc_geometry", "mask_sets", default={}) or {})
    rows: list[dict[str, Any]] = []

    for (mask_set, group_id), sg in selection.groupby(["mask_set", "group_id"], sort=False):
        sg = sg.copy()
        bad = sg[~sg["status"].eq("ok")]
        if selection_group_complete(sg, config):
            continue

        unresolved_slots = bad["slot"].astype(str).tolist()
        unresolved_status = {
            str(r["slot"]): str(r["status"])
            for _, r in bad.iterrows()
        }
        resolved_stems = (
            sg.loc[sg["status"].eq("ok"), "stem"].astype(str).tolist()
            if "stem" in sg.columns else []
        )

        rows.append({
            "mask_set": str(mask_set),
            "group_id": str(group_id),
            "mouse_id": safe_str(sg.iloc[0].get("mouse_id")),
            "experimental_group": safe_str(sg.iloc[0].get("experimental_group")),
            "mask_set_required": bool((cfg_sets.get(str(mask_set), {}) or {}).get("required", False)),
            "n_slots": int(len(sg)),
            "n_ok": int(sg["status"].eq("ok").sum()),
            "n_unresolved": int(len(bad)),
            "unresolved_slots_json": json.dumps(unresolved_slots, ensure_ascii=False),
            "unresolved_status_json": json.dumps(unresolved_status, ensure_ascii=False),
            "resolved_stems_json": json.dumps(resolved_stems, ensure_ascii=False),
            "skip_reason": "incomplete_mask_set_selection",
        })

    return pd.DataFrame(rows, columns=columns)


def build_mask_set_cohort_summary(
    selection: pd.DataFrame,
    skipped: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarise expected, complete and skipped mice per group × mask_set."""
    columns = [
        "mask_set", "experimental_group",
        "n_mice_expected", "n_mice_complete", "n_mice_skipped",
    ]
    if selection.empty:
        return pd.DataFrame(columns=columns)

    base = (
        selection[["mask_set", "group_id", "mouse_id", "experimental_group"]]
        .drop_duplicates()
        .copy()
    )

    complete_keys = []
    for (mask_set, group_id), sg in selection.groupby(["mask_set", "group_id"], sort=False):
        if selection_group_complete(sg, config):
            complete_keys.append((str(mask_set), str(group_id)))

    complete_key_set = set(complete_keys)
    base["complete"] = [
        (str(ms), str(gid)) in complete_key_set
        for ms, gid in zip(base["mask_set"], base["group_id"])
    ]

    out = (
        base.groupby(["mask_set", "experimental_group"], dropna=False)
        .agg(
            n_mice_expected=("mouse_id", "nunique"),
            n_mice_complete=("complete", "sum"),
        )
        .reset_index()
    )
    out["n_mice_complete"] = out["n_mice_complete"].astype(int)
    out["n_mice_skipped"] = out["n_mice_expected"] - out["n_mice_complete"]
    return out[columns]



def build_cohort_overview(
    selection: pd.DataFrame,
    config: dict[str, Any],
    built_mask_sets: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Count distinct mice by *analysis comparison*, not overlapping experiment groups.

    The ALL row is a unique count for the mask set; group rows partition only
    that mask set, in particular Awake/Iso for daypt_anesthesia. "Candidates"
    means considered for this mask set, NOT all Step2 animals. Complete means
    slot-selection complete; built additionally requires successful mask creation.
    """
    cols = ["mask_set", "comparison_group", "n_mice_considered",
            "n_mice_selection_complete", "n_mice_masks_built", "n_selected_stems"]
    if selection.empty:
        return pd.DataFrame(columns=cols)
    entries = []
    for (mask_set, group_id), g in selection.groupby(["mask_set", "group_id"], sort=False):
        mouse = safe_str(g.iloc[0].get("mouse_id", group_id))
        complete = selection_group_complete(g, config)
        ok = g[g["status"].eq("ok")]
        comparison = (safe_str(ok.iloc[0].get("state")) if mask_set == "daypt_anesthesia" and complete and len(ok) == 1
                      else safe_str(g.iloc[0].get("experimental_group")) if mask_set != "daypt_anesthesia"
                      else "UNRESOLVED")
        entries.append({"mask_set": mask_set, "group_id": group_id, "mouse_id": mouse,
                        "comparison_group": comparison, "complete": bool(complete),
                        "n_selected_stems": len(ok) if complete else 0})
    base = pd.DataFrame(entries)
    # Successful built cases, not inferred from selection alone.
    built = pd.DataFrame() if built_mask_sets is None else built_mask_sets.copy()
    if built_mask_sets is None:
        base["built"] = pd.NA
    elif built.empty:
        base["built"] = False
    else:
        required = {"mask_set", "group_id"}
        if not required.issubset(built.columns):
            raise KeyError(f"built_mask_sets missing {required-set(built.columns)}")
        keyset = set(zip(built["mask_set"].astype(str), built["group_id"].astype(str)))
        base["built"] = [(str(ms), str(gid)) in keyset for ms, gid in zip(base["mask_set"], base["group_id"])]
    result = []
    for mask_set, g in base.groupby("mask_set", sort=False):
        for label, sub in [("ALL", g)] + [(key, part) for key, part in g.groupby("comparison_group", sort=True)]:
            complete_sub = sub[sub["complete"]]
            n_built = pd.NA if built_mask_sets is None else int(sub["built"].sum())
            result.append({"mask_set": mask_set, "comparison_group": label,
                           "n_mice_considered": int(sub["mouse_id"].nunique()),
                           "n_mice_selection_complete": int(complete_sub["mouse_id"].nunique()),
                           "n_mice_masks_built": n_built,
                           "n_selected_stems": int(complete_sub["n_selected_stems"].sum())})
    return pd.DataFrame(result, columns=cols)


def build_final_roi_pass_summary(selected: pd.DataFrame, group_summary: pd.DataFrame) -> pd.DataFrame:
    """Readable ROI-level decisions. Uses the *existing* authoritative selection.

    Does NOT redefine PASS, does NOT use stem-level PASS as a cohort gate.
    """
    cols = ["mask_set", "analysis", "roi_key", "hemisphere", "roi", "PASS",
            "min_group_technical_pass_fraction", "required_fraction",
            "n_comparison_groups", "group_technical_pass_details", "reason"]
    if selected.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, r in selected.iterrows():
        mask_set, analysis, key = map(safe_str, (r["mask_set"], r["analysis"], r["roi_key"]))
        group = group_summary[(group_summary["mask_set"].astype(str)==mask_set) &
                              (group_summary["analysis"].astype(str)==analysis) &
                              (group_summary["roi_key"].astype(str)==key)] if not group_summary.empty else pd.DataFrame()
        if not group.empty:
            detail = "; ".join(f"{rr['experimental_group']}: {int(rr['n_mice_technical_pass'])}/{int(rr['n_mice_total'])}"
                                for _, rr in group.sort_values("experimental_group").iterrows())
        else:
            detail = ""
        rows.append({"mask_set": mask_set, "analysis": analysis, "roi_key": key,
                     "hemisphere": safe_str(r.get("hemisphere")), "roi": safe_str(r.get("broad_acronym")),
                     "PASS": bool(r["include"]),
                     "min_group_technical_pass_fraction": float(r["min_group_technical_pass_fraction"]),
                     "required_fraction": float(r["min_mouse_fraction_required"]),
                     "n_comparison_groups": int(r["n_groups"]),
                     "group_technical_pass_details": detail,
                     "reason": safe_str(r["exclusion_reason"])})
    return pd.DataFrame(rows, columns=cols).sort_values(["mask_set", "analysis", "roi_key"]).reset_index(drop=True)


# =============================================================================
# v4.9 canonical Allen side/pair coverage helpers
# =============================================================================


def load_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object and reject non-mapping JSON content."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def group_transform_path(paths: dict[str, Path], group_id: str, ref_stem: str) -> Path:
    """Build the path of a saved group-level atlas transform."""
    key = safe_key(group_id, ref_stem)
    return Path(paths["transforms_root"]) / f"{key}_atlas_transform.json"


def atlas_midline_ml_idx(paths: dict[str, Path], atlas_data: Any) -> tuple[float, str]:
    """Return the geometric midline of the canonical Allen ML grid.

    v4.9 geometric-midline rule:
    Canonical left/right splitting must use the geometric center of the ML axis,
    not a rounded integer bregma ML index. For an even number of ML columns the
    midline therefore lies between the two central columns (e.g. 49.5 for 100
    columns), giving equal numbers of columns on the left and right. For an odd
    number of columns it lies on the single central column.

    The Step6 bregma landmark remains untouched and is still used by Step6 for
    atlas fitting/coordinates; this helper is only for Step6b canonical
    hemisphere splitting.
    """
    n_ml = int(atlas_data.flat_label.shape[1])
    if n_ml <= 0:
        raise ValueError(f"Invalid canonical Allen ML size: {n_ml}")
    return (n_ml - 1) / 2.0, "canonical_geometric_center"


def canonical_left_right_masks(
    atlas_data: Any,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, str]:
    """Split the canonical Allen flat projection at its geometric ML center.

    v4.9 uses one authoritative helper for the center. With an even ML width the
    midline lies BETWEEN the two central columns; with an odd width it lies ON
    the single central column. This guarantees equal left/right grid widths.
    """
    mid, source = atlas_midline_ml_idx(paths, atlas_data)

    _, ml = np.indices(atlas_data.flat_label.shape)
    left_rel = safe_str(
        cfg_get(config, "step6", "atlas", "left_ml_relation_to_bregma", default="negative")
    ).strip().lower()
    if left_rel == "negative":
        left = ml < mid
        right = ml > mid
    elif left_rel == "positive":
        left = ml > mid
        right = ml < mid
    else:
        raise ValueError(
            "step6.atlas.left_ml_relation_to_bregma must be negative/positive"
        )
    midline = ~(left | right)
    return left, right, midline, float(mid), source



def internal_hole_candidates_bool(
    mask: np.ndarray,
    *,
    background_connectivity: int = 4,
) -> np.ndarray:
    """Return enclosed background components of a binary ROI.

    `background_connectivity=4` is the preferred raster convention here:
    a background pixel that touches the exterior only diagonally is treated
    as an internal hole rather than as an exterior-connected opening.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape={m.shape}")

    conn = int(background_connectivity)
    if conn not in {4, 8}:
        raise ValueError(
            f"background_connectivity must be 4 or 8, got {background_connectivity!r}"
        )

    background = (~m).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(background, connectivity=conn)

    if n_labels <= 1:
        return np.zeros_like(m, dtype=bool)

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    outside = np.isin(labels, border_labels)
    return (~m) & (~outside)


def fill_internal_holes_bool(
    mask: np.ndarray,
    *,
    allowed_fill_mask: np.ndarray | None = None,
    max_hole_area_px: int | None = None,
    background_connectivity: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill enclosed ROI holes, using 4-connected background by default.

    Only pixels permitted by `allowed_fill_mask` are written. In Step6b the
    allowed mask is `analysis_labelmap == 0`, so pixels belonging to another
    analytical Allen ROI remain protected.
    """
    m = np.asarray(mask, dtype=bool)

    hole_candidates = internal_hole_candidates_bool(
        m,
        background_connectivity=background_connectivity,
    )

    holes = hole_candidates.copy()

    if allowed_fill_mask is not None:
        allowed = np.asarray(allowed_fill_mask, dtype=bool)
        if allowed.shape != m.shape:
            raise ValueError(
                f"allowed_fill_mask shape {allowed.shape} != mask shape {m.shape}"
            )
        holes &= allowed

    if max_hole_area_px is not None:
        max_area = int(max_hole_area_px)
        if max_area < 0:
            raise ValueError("max_internal_zero_hole_area_px must be >= 0 or null")

        # Re-label only the candidate pixels that are actually fillable.
        n_labels, labels = cv2.connectedComponents(
            holes.astype(np.uint8),
            connectivity=int(background_connectivity),
        )
        keep = np.zeros_like(holes, dtype=bool)
        for lab in range(1, int(n_labels)):
            comp = labels == lab
            if int(comp.sum()) <= max_area:
                keep |= comp
        holes = keep

    return (m | holes), holes


def erode_bool(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """Apply disk-shaped binary erosion using the requested pixel radius."""
    radius = int(round(float(radius_px)))
    m = np.asarray(mask, dtype=np.uint8)
    if radius <= 0:
        return m.astype(bool)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.erode(
        m,
        kernel,
        iterations=1,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def common_brain_for_technical_analysis(
    common_brain: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the optional v4.12 common-brain edge erosion.

    This operation is deliberately isolated from hemisphere geometry and Allen ROI
    buffering. ``left_hemisphere_ref.npy`` / ``right_hemisphere_ref.npy`` therefore
    keep their v4.10 definition (geometric side + midline exclusion only).

    The returned ``for_technical`` mask is the only mask used downstream when
    building ``fixed_technical_ref``. With the option disabled, it is an exact
    copy of ``common_brain`` so v4.12 reproduces the v4.10 fixed-technical logic.
    """
    src = np.asarray(common_brain, dtype=bool)
    block = dict(
        cfg_get(
            config,
            "step6b_fc_geometry",
            "technical_geometry",
            "common_brain_edge_erosion",
            default={},
        )
        or {}
    )
    enabled = bool(block.get("enabled", False))
    requested_radius = float(block.get("radius_px", 0) or 0)
    if requested_radius < 0:
        raise ValueError(
            "step6b_fc_geometry.technical_geometry.common_brain_edge_erosion.radius_px "
            "must be >= 0"
        )
    applied_radius = int(round(requested_radius)) if enabled else 0
    if applied_radius > 0:
        out = erode_bool(src, applied_radius)
    else:
        out = src.copy()
    excluded = src & (~out)
    return {
        "original": src,
        "for_technical": out,
        "excluded_edge": excluded,
        "enabled": bool(enabled),
        "requested_radius_px": float(requested_radius),
        "applied_radius_px": int(applied_radius),
        "before_px": int(src.sum()),
        "after_px": int(out.sum()),
        "excluded_px": int(excluded.sum()),
        "retained_fraction": safe_fraction(int(out.sum()), int(src.sum())),
    }


def build_canonical_side_rois(
    atlas_data: Any,
    config: dict[str, Any],
    paths: dict[str, Path],
    parent_query: str,
) -> dict[str, Any]:
    """Construct fixed side-specific analytical ROI masks in canonical Allen space.

    Parameters
    ----------
    atlas_data : object
        Loaded Allen atlas geometry, label grid and ontology.
    config : dict
        Project YAML, including ROI refinements, internal-hole repair and buffers.
    paths : dict[str, pathlib.Path]
        Step6/Step6b output and atlas paths.
    parent_query : str
        Parent structure acronym or name defining the broad ROI family.

    Returns
    -------
    dict[str, Any]
        Canonical full/buffered masks, ROI definitions, labelmaps and QC details.

    Notes
    -----
    Only configured ROI families are selectively refined. Enclosed zero-label
    holes and eligible tiny nonzero islands are repaired before boundary
    exclusion. Canonical areas supply denominators for quantitative coverage;
    reference-grid pixel counts are retained separately for auditing."""
    broad_full_raw, mapping, children = build_broad_labelmap(
        atlas_data.flat_label, atlas_data, parent_query, config=config
    )
    left, right, midline, mid_idx, mid_source = canonical_left_right_masks(
        atlas_data, config, paths
    )

    boundary_px = float(
        cfg_get(
            config, "step6b_fc_geometry", "roi_geometry",
            "boundary_exclusion_canonical_px", default=2,
        )
    )
    fill_holes = bool(
        cfg_get(
            config, "step6b_fc_geometry", "roi_geometry",
            "fill_internal_zero_holes_before_buffer", default=True,
        )
    )
    hole_connectivity = int(
        cfg_get(
            config, "step6b_fc_geometry", "roi_geometry",
            "internal_hole_background_connectivity", default=4,
        )
    )
    if hole_connectivity not in {4, 8}:
        raise ValueError(
            "step6b_fc_geometry.roi_geometry.internal_hole_background_connectivity "
            "must be 4 or 8"
        )

    max_zero_raw = cfg_get(
        config, "step6b_fc_geometry", "roi_geometry",
        "max_internal_zero_hole_area_px", default=None,
    )
    max_zero_area = (
        None if max_zero_raw is None
        or safe_str(max_zero_raw).strip().lower() in {"", "none", "null"}
        else int(max_zero_raw)
    )

    max_nonzero_raw = cfg_get(
        config, "step6b_fc_geometry", "roi_geometry",
        "max_internal_nonzero_hole_area_px", default=1,
    )
    max_nonzero_area = (
        0 if max_nonzero_raw is None
        or safe_str(max_nonzero_raw).strip().lower() in {"", "none", "null"}
        else int(max_nonzero_raw)
    )
    if max_nonzero_area < 0:
        raise ValueError("max_internal_nonzero_hole_area_px must be >= 0")

    meta = child_meta_map(children)
    roi_ids = sorted(int(v) for v in np.unique(broad_full_raw) if int(v) > 0)

    # ------------------------------------------------------------------
    # Pass 1: collect repair claims on the RAW analytical labelmap.
    # A claim is attached to an entire enclosed background component of
    # the target ROI. Small non-zero islands can be absorbed if configured.
    # ------------------------------------------------------------------
    claim_owner = np.zeros_like(broad_full_raw, dtype=np.int32)
    claim_count = np.zeros_like(broad_full_raw, dtype=np.uint8)
    claim_kind = np.zeros_like(broad_full_raw, dtype=np.uint8)  # 1=zero, 2=nonzero
    diagnostics: dict[tuple[int, str], dict[str, Any]] = {}

    for bid in roi_ids:
        for side, side_mask in (("left", left), ("right", right)):
            raw_roi = (broad_full_raw == bid) & side_mask
            candidates = (
                internal_hole_candidates_bool(
                    raw_roi,
                    background_connectivity=hole_connectivity,
                )
                if fill_holes
                else np.zeros_like(raw_roi, dtype=bool)
            )

            diag = {
                "candidate_px": int(candidates.sum()),
                "zero_claim_px": 0,
                "nonzero_claim_px": 0,
                "protected_px": 0,
                "conflict_px": 0,
                "nonzero_original_labels": [],
            }

            if candidates.any():
                n_comp, comp_labels = cv2.connectedComponents(
                    candidates.astype(np.uint8),
                    connectivity=hole_connectivity,
                )
                for lab in range(1, int(n_comp)):
                    comp = comp_labels == lab
                    area = int(comp.sum())
                    if area <= 0:
                        continue
                    vals = np.unique(broad_full_raw[comp])
                    vals_nonzero = [int(v) for v in vals if int(v) != 0]

                    if len(vals_nonzero) == 0:
                        eligible = max_zero_area is None or area <= max_zero_area
                        kind = 1
                    else:
                        eligible = max_nonzero_area > 0 and area <= max_nonzero_area
                        kind = 2

                    if not eligible:
                        diag["protected_px"] += area
                        if vals_nonzero:
                            diag["nonzero_original_labels"].extend(vals_nonzero)
                        continue

                    claim_count[comp] = np.minimum(
                        claim_count[comp].astype(np.uint16) + 1, 255
                    ).astype(np.uint8)
                    # owner is meaningful only where exactly one claim survives;
                    # conflicts are detected below and never applied.
                    claim_owner[comp] = int(bid)
                    claim_kind[comp] = int(kind)

                    if kind == 1:
                        diag["zero_claim_px"] += area
                    else:
                        diag["nonzero_claim_px"] += area
                        diag["nonzero_original_labels"].extend(vals_nonzero)

            diagnostics[(bid, side)] = diag

    conflict_mask = claim_count > 1
    unique_claim = claim_count == 1

    # Apply only unique claims -> repaired analytical labelmap is exclusive.
    broad_full = broad_full_raw.copy()
    broad_full[unique_claim] = claim_owner[unique_claim]

    # Attribute conflicts back to target diagnostics.
    if conflict_mask.any():
        for bid in roi_ids:
            for side, side_mask in (("left", left), ("right", right)):
                diagnostics[(bid, side)]["conflict_px"] = int(
                    (conflict_mask & side_mask & (claim_owner == bid)).sum()
                )

    # ------------------------------------------------------------------
    # Pass 2: build all ROI masks from the SAME repaired labelmap.
    # This prevents overlap after a non-zero pixel is reassigned.
    # ------------------------------------------------------------------
    full_side_labels = np.zeros(broad_full.shape, dtype=np.int32)
    buffered_side_labels = np.zeros(broad_full.shape, dtype=np.int32)
    buffered_broad = np.zeros(broad_full.shape, dtype=np.int32)
    roi_masks_raw: dict[str, np.ndarray] = {}
    roi_masks_full: dict[str, np.ndarray] = {}
    roi_masks_buffered: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    code = 0

    for bid in roi_ids:
        m = meta.get(bid, {})
        acr = safe_str(m.get("analysis_roi_acronym", m.get("broad_acronym", str(bid)))) or str(bid)
        name = safe_str(m.get("analysis_roi_name", m.get("broad_name", str(bid)))) or str(bid)
        family_id = int(m.get("broad_family_structure_id", bid))
        family_acr = safe_str(m.get("broad_family_acronym", acr)) or acr
        family_name = safe_str(m.get("broad_family_name", name)) or name
        refined = bool(m.get("roi_refinement_applied", False))
        definition_source = safe_str(m.get("roi_definition_source", "broad_family"))

        for side, side_mask in (("left", left), ("right", right)):
            raw_full = (broad_full_raw == bid) & side_mask
            full = (broad_full == bid) & side_mask
            buffered = erode_bool(full, boundary_px)

            gained = full & (~raw_full)
            lost = raw_full & (~full)
            zero_gained = gained & (broad_full_raw == 0)
            nonzero_gained = gained & (broad_full_raw != 0)

            diag = diagnostics[(bid, side)]
            orig_nonzero = sorted(set(int(v) for v in diag["nonzero_original_labels"]))

            code += 1
            roi_key = f"{acr}_{side}"
            full_side_labels[full] = code
            buffered_side_labels[buffered] = code
            buffered_broad[buffered] = bid
            roi_masks_raw[roi_key] = raw_full
            roi_masks_full[roi_key] = full
            roi_masks_buffered[roi_key] = buffered

            rows.append({
                "side_roi_code": code,
                "broad_structure_id": bid,
                "broad_acronym": acr,
                "broad_name": name,
                "analysis_structure_id": bid,
                "analysis_roi_acronym": acr,
                "analysis_roi_name": name,
                "broad_family_structure_id": family_id,
                "broad_family_acronym": family_acr,
                "broad_family_name": family_name,
                "roi_refinement_applied": refined,
                "roi_definition_source": definition_source,
                "hemisphere": side,
                "roi_key": roi_key,

                "canonical_raw_full_area_px": int(raw_full.sum()),
                "canonical_enclosed_hole_candidate_px": int(diag["candidate_px"]),
                "canonical_internal_zero_hole_filled_px": int(zero_gained.sum()),
                "canonical_internal_nonzero_hole_reassigned_px": int(nonzero_gained.sum()),
                "canonical_reassigned_away_px": int(lost.sum()),
                "canonical_protected_hole_px": int(diag["protected_px"]),
                "canonical_hole_claim_conflict_px": int(diag["conflict_px"]),
                "canonical_nonzero_hole_original_labels_json": json.dumps(orig_nonzero),
                "canonical_hole_background_connectivity": int(hole_connectivity),
                "canonical_hole_fill_enabled": bool(fill_holes),
                "canonical_max_internal_zero_hole_area_px": max_zero_area,
                "canonical_max_internal_nonzero_hole_area_px": int(max_nonzero_area),

                "canonical_full_area_px": int(full.sum()),
                "canonical_buffered_area_px": int(buffered.sum()),
                "full_allen_area_px": int(full.sum()),
                "buffered_full_allen_area_px": int(buffered.sum()),
                "canonical_boundary_excluded_px": int(full.sum() - buffered.sum()),
                "canonical_boundary_retained_fraction": safe_fraction(
                    int(buffered.sum()), int(full.sum())
                ),
                "boundary_exclusion_canonical_px": boundary_px,
                "atlas_midline_ml_idx": float(mid_idx),
                "atlas_midline_source": mid_source,
                "canonical_n_ml_columns": int(broad_full.shape[1]),
                "canonical_left_columns": int(np.any(left, axis=0).sum()),
                "canonical_right_columns": int(np.any(right, axis=0).sum()),
                "canonical_midline_columns": int(np.any(midline, axis=0).sum()),
            })

    repair_mask = broad_full != broad_full_raw
    repair_rows: list[dict[str, Any]] = []
    yy, xx = np.where(repair_mask)
    for y, x in zip(yy, xx):
        old_id = int(broad_full_raw[y, x])
        new_id = int(broad_full[y, x])
        repair_rows.append({
            "ap_idx": int(y),
            "ml_idx": int(x),
            "old_analysis_structure_id": old_id,
            "new_analysis_structure_id": new_id,
            "repair_kind": "zero_fill" if old_id == 0 else "nonzero_reassignment",
        })

    return {
        "broad_full": broad_full,
        "broad_full_raw": broad_full_raw,
        "buffered_broad": buffered_broad,
        "full_side_labels": full_side_labels,
        "buffered_side_labels": buffered_side_labels,
        "roi_masks_raw": roi_masks_raw,
        "roi_masks_full": roi_masks_full,
        "roi_masks_buffered": roi_masks_buffered,
        "roi_table": pd.DataFrame(rows),
        "hole_repair_table": pd.DataFrame(repair_rows),
        "mapping": mapping,
        "children": children,
        "roi_refinements": children.attrs.get("roi_refinement_table", pd.DataFrame()).copy(),
        "left_mask": left,
        "right_mask": right,
        "midline_mask": midline,
        "midline_ml_idx": mid_idx,
        "midline_source": mid_source,
        "boundary_px": boundary_px,
        "fill_internal_zero_holes_before_buffer": bool(fill_holes),
        "internal_hole_background_connectivity": int(hole_connectivity),
        "max_internal_zero_hole_area_px": max_zero_area,
        "max_internal_nonzero_hole_area_px": int(max_nonzero_area),
    }



# =============================================================================
# v4.10 reference-space analytical atlas repair
# =============================================================================


def repair_reference_analysis_labelmap(
    analysis_roi_ref_raw: np.ndarray,
    ref_left: np.ndarray,
    ref_right: np.ndarray,
    config: dict[str, Any],
    *,
    group_id: str,
    mask_set: str,
    mouse_id: str,
    roi_definitions: pd.DataFrame,
) -> dict[str, Any]:
    """Repair a mouse-specific analytical ROI labelmap before reference erosion.

    This mirrors the *principle* used by the canonical v4.9 branch without
    replacing it.  The input is the Step6 reference atlas after Allen IDs have
    already been recoded to Step6b analytical ROI IDs (including optional
    selective refinement such as MO -> MOp/MOs).

    Repair rules are read from the existing roi_geometry config:
      * fill_internal_zero_holes_before_buffer
      * internal_hole_background_connectivity
      * max_internal_zero_hole_area_px
      * max_internal_nonzero_hole_area_px

    A repair is applied to the shared analytical labelmap first, so a non-zero
    reassignment remains exclusive: the pixel is removed from its old ROI and
    assigned to the surrounding ROI.  Boundary erosion happens only afterwards.
    """
    raw = np.asarray(analysis_roi_ref_raw, dtype=np.int32)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D reference analytical labelmap, got {raw.shape}")
    left = np.asarray(ref_left, dtype=bool)
    right = np.asarray(ref_right, dtype=bool)
    if left.shape != raw.shape or right.shape != raw.shape:
        raise ValueError(
            f"Reference side masks {left.shape}/{right.shape} do not match labelmap {raw.shape}"
        )

    fill_holes = bool(
        cfg_get(
            config, "step6b_fc_geometry", "roi_geometry",
            "fill_internal_zero_holes_before_buffer", default=True,
        )
    )
    connectivity = int(
        cfg_get(
            config, "step6b_fc_geometry", "roi_geometry",
            "internal_hole_background_connectivity", default=4,
        )
    )
    if connectivity not in {4, 8}:
        raise ValueError(
            "step6b_fc_geometry.roi_geometry.internal_hole_background_connectivity "
            "must be 4 or 8"
        )

    max_zero_raw = cfg_get(
        config, "step6b_fc_geometry", "roi_geometry",
        "max_internal_zero_hole_area_px", default=None,
    )
    max_zero_area = (
        None
        if max_zero_raw is None
        or safe_str(max_zero_raw).strip().lower() in {"", "none", "null"}
        else int(max_zero_raw)
    )
    if max_zero_area is not None and max_zero_area < 0:
        raise ValueError("max_internal_zero_hole_area_px must be >= 0 or null")

    max_nonzero_raw = cfg_get(
        config, "step6b_fc_geometry", "roi_geometry",
        "max_internal_nonzero_hole_area_px", default=1,
    )
    max_nonzero_area = (
        0
        if max_nonzero_raw is None
        or safe_str(max_nonzero_raw).strip().lower() in {"", "none", "null"}
        else int(max_nonzero_raw)
    )
    if max_nonzero_area < 0:
        raise ValueError("max_internal_nonzero_hole_area_px must be >= 0")

    meta = child_meta_map(roi_definitions)
    roi_ids = sorted(int(v) for v in np.unique(raw) if int(v) > 0)

    # Pixel-level exclusive claims, exactly as in the canonical principle.
    claim_owner = np.zeros(raw.shape, dtype=np.int32)
    claim_count = np.zeros(raw.shape, dtype=np.uint8)
    component_records: list[dict[str, Any]] = []

    for bid in roi_ids:
        m = meta.get(bid, {})
        acr = safe_str(m.get("analysis_roi_acronym", m.get("broad_acronym", str(bid)))) or str(bid)
        family_id = int(m.get("broad_family_structure_id", bid))
        family_acr = safe_str(m.get("broad_family_acronym", acr)) or acr

        for side, side_mask in (("left", left), ("right", right)):
            raw_roi = (raw == bid) & side_mask
            candidates = (
                internal_hole_candidates_bool(
                    raw_roi,
                    background_connectivity=connectivity,
                )
                if fill_holes
                else np.zeros_like(raw_roi, dtype=bool)
            )
            if not candidates.any():
                continue

            n_comp, comp_labels = cv2.connectedComponents(
                candidates.astype(np.uint8),
                connectivity=connectivity,
            )
            for comp_id in range(1, int(n_comp)):
                comp = comp_labels == comp_id
                area = int(comp.sum())
                if area <= 0:
                    continue
                old_ids = sorted(int(v) for v in np.unique(raw[comp]))
                old_nonzero = [v for v in old_ids if v != 0]

                if not old_nonzero:
                    repair_kind = "zero_fill"
                    eligible = max_zero_area is None or area <= max_zero_area
                else:
                    repair_kind = "nonzero_reassignment"
                    eligible = max_nonzero_area > 0 and area <= max_nonzero_area

                flat_idx = np.flatnonzero(comp)
                if eligible:
                    # Count all independent target claims. Any conflicting pixel
                    # will be left unchanged below.
                    current = claim_count.ravel()[flat_idx].astype(np.uint16) + 1
                    claim_count.ravel()[flat_idx] = np.minimum(current, 255).astype(np.uint8)
                    claim_owner.ravel()[flat_idx] = int(bid)

                component_records.append({
                    "group_id": str(group_id),
                    "mask_set": str(mask_set),
                    "mouse_id": str(mouse_id),
                    "analysis_structure_id": int(bid),
                    "analysis_roi_acronym": acr,
                    "broad_family_structure_id": int(family_id),
                    "broad_family_acronym": family_acr,
                    "hemisphere": side,
                    "repair_kind": repair_kind,
                    "component_area_px": int(area),
                    "old_analysis_structure_ids_json": json.dumps(old_ids),
                    "eligible_by_size": bool(eligible),
                    "_flat_idx": flat_idx,
                })

    unique_claim = claim_count == 1
    repaired = raw.copy()
    repaired[unique_claim] = claim_owner[unique_claim]

    audit_rows: list[dict[str, Any]] = []
    for rec in component_records:
        idx = rec.pop("_flat_idx")
        target = int(rec["analysis_structure_id"])
        applied = unique_claim.ravel()[idx] & (claim_owner.ravel()[idx] == target)
        n_corrected = int(applied.sum())
        area = int(rec["component_area_px"])
        if not bool(rec["eligible_by_size"]):
            status = "protected_by_size"
        elif n_corrected == area:
            status = "applied"
        elif n_corrected == 0:
            status = "conflict_not_applied"
        else:
            status = "partial_conflict"
        rec["repair_status"] = status
        rec["n_corrected_px"] = n_corrected
        audit_rows.append(rec)

    repair_mask = repaired != raw
    return {
        "raw": raw,
        "repaired": repaired,
        "repair_mask": repair_mask,
        "repair_table": pd.DataFrame(audit_rows),
        "n_repaired_px": int(repair_mask.sum()),
        "fill_internal_zero_holes_before_buffer": bool(fill_holes),
        "internal_hole_background_connectivity": int(connectivity),
        "max_internal_zero_hole_area_px": max_zero_area,
        "max_internal_nonzero_hole_area_px": int(max_nonzero_area),
    }


def save_reference_atlas_repair_qc_png(
    path: Path,
    reference_image: np.ndarray,
    raw_labelmap: np.ndarray,
    repaired_labelmap: np.ndarray,
    buffered_repaired_labelmap: np.ndarray,
    *,
    title: str,
) -> None:
    """Save raw vs repaired vs buffered reference analytical atlas QC.

    The lower row zooms around the changed pixels so single-pixel repairs are
    actually visible in the saved PNG.
    """
    raw = np.asarray(raw_labelmap, dtype=np.int32)
    repaired = np.asarray(repaired_labelmap, dtype=np.int32)
    buffered = np.asarray(buffered_repaired_labelmap, dtype=np.int32)
    changed = raw != repaired
    if not changed.any():
        return

    yy, xx = np.where(changed)
    pad = 30
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(raw.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(raw.shape[1], int(xx.max()) + pad + 1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(reference_image, cmap="gray")
    axes[0, 0].set_title("reference image")
    axes[0, 1].imshow(compact_label_display(raw), interpolation="nearest")
    axes[0, 1].set_title("analysis_roi_ref_raw")
    axes[0, 2].imshow(compact_label_display(repaired), interpolation="nearest")
    axes[0, 2].imshow(np.where(changed, 1.0, np.nan), cmap="Reds", alpha=0.85, interpolation="nearest")
    axes[0, 2].set_title(f"repaired; changed={int(changed.sum())} px")
    axes[0, 3].imshow(compact_label_display(buffered), interpolation="nearest")
    axes[0, 3].set_title("buffered repaired")

    axes[1, 0].imshow(reference_image[y0:y1, x0:x1], cmap="gray", interpolation="nearest")
    axes[1, 0].set_title("repair zoom: reference")
    axes[1, 1].imshow(compact_label_display(raw[y0:y1, x0:x1]), interpolation="nearest")
    axes[1, 1].set_title("zoom raw")
    axes[1, 2].imshow(compact_label_display(repaired[y0:y1, x0:x1]), interpolation="nearest")
    axes[1, 2].imshow(
        np.where(changed[y0:y1, x0:x1], 1.0, np.nan),
        cmap="Reds", alpha=0.9, interpolation="nearest",
    )
    axes[1, 2].set_title("zoom repaired")
    axes[1, 3].imshow(compact_label_display(buffered[y0:y1, x0:x1]), interpolation="nearest")
    axes[1, 3].set_title("zoom buffered")

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_canonical_homotopic_pairs(
    canonical: dict[str, Any],
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    """Build unique left/right analytical atlas pixel pairs on the canonical grid.

    Parameters
    ----------
    canonical : dict[str, Any]
        Canonical side-specific analytical ROI masks and geometry.

    Returns
    -------
    pandas.DataFrame
        Pair keys, ROI identifiers and canonical pair counts.
    dict[int, dict[str, numpy.ndarray]]
        Partner indices for each analytical ROI structure ID.

    Notes
    -----
    Each mirrored pair is counted once rather than once in each hemisphere."""
    buffered_broad = np.asarray(canonical["buffered_broad"], dtype=np.int32)
    left = np.asarray(canonical["left_mask"], dtype=bool)
    right = np.asarray(canonical["right_mask"], dtype=bool)
    # Keep the exact geometric center.  For an even ML width this is a
    # half-integer (e.g. 49.5), so mirroring column 49 gives column 50.
    mid = float(canonical["midline_ml_idx"])
    roi_table = canonical["roi_table"]
    pair_rows: list[dict[str, Any]] = []
    pair_map: dict[int, dict[str, np.ndarray]] = {}
    n_ml = buffered_broad.shape[1]

    pair_meta_cols = [
        "broad_structure_id", "broad_acronym", "broad_name",
        "analysis_structure_id", "analysis_roi_acronym", "analysis_roi_name",
        "broad_family_structure_id", "broad_family_acronym", "broad_family_name",
        "roi_refinement_applied", "roi_definition_source",
    ]
    pair_meta_cols = [c for c in pair_meta_cols if c in roi_table.columns]
    broad_meta = roi_table[pair_meta_cols].drop_duplicates("broad_structure_id")
    for _, mr in broad_meta.iterrows():
        bid = int(mr["broad_structure_id"])
        ap_l, ml_l = np.where((buffered_broad == bid) & left)
        # Reflection around a geometric ML center.  2*mid is always integer
        # for integer-width arrays, therefore mirrored pixel centers land
        # exactly on integer columns for both even and odd widths.
        ml_r = np.rint(2.0 * mid - ml_l.astype(float)).astype(np.int32)
        in_bounds = (ml_r >= 0) & (ml_r < n_ml)
        ap_l = ap_l[in_bounds]
        ml_l = ml_l[in_bounds]
        ml_r = ml_r[in_bounds]
        valid = right[ap_l, ml_r] & (buffered_broad[ap_l, ml_r] == bid)
        ap_l = ap_l[valid].astype(np.int32)
        ml_l = ml_l[valid].astype(np.int32)
        ap_r = ap_l.copy()
        ml_r = ml_r[valid].astype(np.int32)
        pair_map[bid] = {
            "left_ap": ap_l,
            "left_ml": ml_l,
            "right_ap": ap_r,
            "right_ml": ml_r,
        }
        pair_rows.append({
            "broad_structure_id": bid,
            "broad_acronym": safe_str(mr["broad_acronym"]),
            "broad_name": safe_str(mr["broad_name"]),
            "analysis_structure_id": int(mr.get("analysis_structure_id", bid)),
            "analysis_roi_acronym": safe_str(mr.get("analysis_roi_acronym", mr["broad_acronym"])),
            "analysis_roi_name": safe_str(mr.get("analysis_roi_name", mr["broad_name"])),
            "broad_family_structure_id": int(mr.get("broad_family_structure_id", bid)),
            "broad_family_acronym": safe_str(mr.get("broad_family_acronym", mr["broad_acronym"])),
            "broad_family_name": safe_str(mr.get("broad_family_name", mr["broad_name"])),
            "roi_refinement_applied": bool(mr.get("roi_refinement_applied", False)),
            "roi_definition_source": safe_str(mr.get("roi_definition_source", "broad_family")),
            "pair_key": f"{safe_str(mr['broad_acronym'])}_LR",
            "left_roi_key": f"{safe_str(mr['broad_acronym'])}_left",
            "right_roi_key": f"{safe_str(mr['broad_acronym'])}_right",
            "canonical_homotopic_pair_px": int(len(ap_l)),
        })
    return pd.DataFrame(pair_rows), pair_map


def project_canonical_grid_to_reference(
    atlas_data: Any,
    transform: dict[str, Any],
    ref_shape: tuple[int, int],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Project every canonical dorsal Allen pixel to one nearest reference pixel."""
    flat_shape = atlas_data.flat_label.shape
    ap_idx, ml_idx = np.where(atlas_data.flat_label > 0)
    dv_idx = atlas_data.flat_dv_idx[ap_idx, ml_idx]
    xyz = ap_dv_ml_indices_to_xyz_um(ap_idx, dv_idx, ml_idx, atlas_data, config)
    uv, _ = pitch_project_xyz_um(
        xyz,
        float(transform["pitch_final_deg"]),
        np.asarray(transform["pivot_xyz_um"], dtype=float),
    )
    basis = {
        "origin_xy": np.asarray(transform["origin_xy"], dtype=float),
        "ap_hat_xy": np.asarray(transform["ap_hat_xy"], dtype=float),
        "ml_hat_xy": np.asarray(transform["ml_hat_xy"], dtype=float),
        "pixel_size_um": float(transform["pixel_size_um"]),
    }
    xy = atlas_uv_um_to_reference_xy_px(
        uv, float(transform["scale_final"]), basis
    )
    xx = np.rint(xy[:, 0]).astype(np.int32)
    yy = np.rint(xy[:, 1]).astype(np.int32)
    h, w = map(int, ref_shape)
    inside_vec = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)

    ry = np.full(flat_shape, -1, dtype=np.int32)
    rx = np.full(flat_shape, -1, dtype=np.int32)
    inside = np.zeros(flat_shape, dtype=bool)
    ry[ap_idx, ml_idx] = yy
    rx[ap_idx, ml_idx] = xx
    inside[ap_idx, ml_idx] = inside_vec
    return {"ref_y": ry, "ref_x": rx, "inside": inside}


def sample_ref_mask_on_canonical(
    projection: dict[str, np.ndarray],
    ref_mask: np.ndarray,
) -> np.ndarray:
    """Resample a reference-grid mask onto projected canonical atlas pixels."""
    out = np.zeros(projection["inside"].shape, dtype=bool)
    inside = projection["inside"]
    if np.any(inside):
        out[inside] = np.asarray(ref_mask, dtype=bool)[
            projection["ref_y"][inside], projection["ref_x"][inside]
        ]
    return out


def canonical_reference_side_consistency(
    *,
    canonical: dict[str, Any],
    projection: dict[str, np.ndarray],
    ref_left: np.ndarray,
    ref_right: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Check that canonical anatomical LEFT/RIGHT project to the same reference labels."""
    cortex = np.asarray(canonical["broad_full"]) > 0
    inside = np.asarray(projection["inside"], dtype=bool)
    left_can = np.asarray(canonical["left_mask"], dtype=bool) & cortex & inside
    right_can = np.asarray(canonical["right_mask"], dtype=bool) & cortex & inside

    def frac(can_mask: np.ndarray, ref_mask: np.ndarray) -> float:
        """Calculate the fraction of a canonical ROI overlapping a reference mask."""
        n = int(can_mask.sum())
        if n <= 0:
            return np.nan
        yy = projection["ref_y"][can_mask]
        xx = projection["ref_x"][can_mask]
        return float(np.asarray(ref_mask, dtype=bool)[yy, xx].mean())

    ll = frac(left_can, ref_left)
    lr = frac(left_can, ref_right)
    rl = frac(right_can, ref_left)
    rr = frac(right_can, ref_right)
    expected = float(np.nanmean([ll, rr]))
    opposite = float(np.nanmean([lr, rl]))
    margin = expected - opposite
    min_margin = float(
        cfg_get(
            config,
            "step6b_fc_geometry",
            "side_consistency_qc",
            "min_expected_minus_opposite",
            default=0.20,
        )
    )
    if not np.isfinite(margin):
        status = "insufficient_projection"
    elif margin >= min_margin:
        status = "ok"
    elif -margin >= min_margin:
        status = "swapped"
    else:
        status = "ambiguous"
    return {
        "canonical_left_to_ref_left": ll,
        "canonical_left_to_ref_right": lr,
        "canonical_right_to_ref_left": rl,
        "canonical_right_to_ref_right": rr,
        "expected_overlap_mean": expected,
        "opposite_overlap_mean": opposite,
        "expected_minus_opposite": margin,
        "min_expected_minus_opposite": min_margin,
        "side_mapping_status": status,
        "n_canonical_left_inside": int(left_can.sum()),
        "n_canonical_right_inside": int(right_can.sum()),
    }


def canonical_spatial_masks(
    *,
    valid_can: np.ndarray,
    canonical: dict[str, Any],
    pair_map: dict[int, dict[str, np.ndarray]],
    healthy_side: str,
    lesion_side: str,
    homotopic_display_choice: str,
) -> dict[str, np.ndarray]:
    """Derive Step6b spatial products on the canonical Allen grid."""
    valid = np.asarray(valid_can, dtype=bool)
    cortex_buffered = np.asarray(canonical["buffered_broad"]) > 0
    bilateral = valid & cortex_buffered
    left = np.asarray(canonical["left_mask"], dtype=bool)
    right = np.asarray(canonical["right_mask"], dtype=bool)
    healthy_mask = left if str(healthy_side).lower() == "left" else right
    lesion_mask = left if str(lesion_side).lower() == "left" else right
    global_healthy = bilateral & healthy_mask

    pairable_bilateral = np.zeros_like(valid, dtype=bool)
    for pinfo in pair_map.values():
        la, lm = pinfo["left_ap"], pinfo["left_ml"]
        ra, rm = pinfo["right_ap"], pinfo["right_ml"]
        ok = valid[la, lm] & valid[ra, rm]
        if np.any(ok):
            pairable_bilateral[la[ok], lm[ok]] = True
            pairable_bilateral[ra[ok], rm[ok]] = True

    choice = safe_str(homotopic_display_choice).strip().lower()
    if choice == "lesion":
        display_side = str(lesion_side).lower()
    elif choice == "healthy":
        display_side = str(healthy_side).lower()
    elif choice in {"left", "right"}:
        display_side = choice
    else:
        raise ValueError(
            "homotopic_display_choice must be lesion/healthy/left/right"
        )
    display_hemi = left if display_side == "left" else right
    homotopic_display = pairable_bilateral & display_hemi

    return {
        "bilateral": bilateral,
        "global_healthy": global_healthy,
        "homotopic_pairable_bilateral": pairable_bilateral,
        "homotopic_pairable_display": homotopic_display,
    }


def save_canonical_frequency_png(
    path: Path,
    frequency: np.ndarray,
    canonical: dict[str, Any],
    title: str,
) -> None:
    """Save a 0..1 cohort representation map with broad Allen boundaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(frequency, dtype=float)
    atlas = np.asarray(canonical["buffered_broad"], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.where(atlas > 0, freq, np.nan), vmin=0.0, vmax=1.0, cmap="viridis")
    # One contour call per broad ID makes anatomical boundaries visible without relabeling.
    for bid in [int(v) for v in np.unique(atlas) if int(v) > 0]:
        m = atlas == bid
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors="white", linewidths=0.35, alpha=0.75)
    ax.set_title(title)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("fraction of mice with canonical pixel valid")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def reference_side_masks_for_mouse(
    ref_shape: tuple[int, int],
    basis: dict[str, Any],
    config: dict[str, Any],
    mouse_id: str,
    group_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    """Build reference-space left/right/healthy masks for one mouse."""
    healthy, lesion, midline, lesion_side, healthy_side = analysis_hemisphere_masks(
        ref_shape, basis, config, mouse_id, group_id
    )
    left = healthy if healthy_side == "left" else lesion
    right = healthy if healthy_side == "right" else lesion
    return left, right, midline, lesion_side, healthy_side


def reference_equivalent_buffer_px(
    canonical_boundary_px: float,
    atlas_data: Any,
    basis: dict[str, Any],
) -> float:
    """Convert a canonical atlas edge buffer to approximate reference pixels."""
    canonical_um = float(canonical_boundary_px) * float(
        np.mean([atlas_data.res_ap_um, atlas_data.res_ml_um])
    )
    ref_um = float(basis.get("pixel_size_um", 1.0))
    return canonical_um / ref_um if ref_um > 0 else float(canonical_boundary_px)


def build_side_technical_raw_table(
    *,
    mask_set: str,
    group_id: str,
    mouse_id: str,
    experimental_group: str,
    canonical: dict[str, Any],
    canonical_technical: np.ndarray,
    broad_labelmap_ref_full: np.ndarray,
    technical_mask_ref: np.ndarray,
    ref_left: np.ndarray,
    ref_right: np.ndarray,
    ref_buffer_px: float,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Measure fixed technical representation for each side-specific atlas ROI.

    Parameters
    ----------
    mask_set, group_id, mouse_id, experimental_group : str
        Analysis and animal identifiers.
    canonical : dict
        Canonical analytical ROI masks and atlas-grid geometry.
    canonical_technical : numpy.ndarray
        Fixed boolean technical-validity mask on the canonical atlas grid.
    broad_labelmap_ref_full : numpy.ndarray
        Reference-space analytical atlas labels before the comparison-specific mask.
    technical_mask_ref : numpy.ndarray
        Fixed technical mask on the full-resolution reference grid.
    ref_left, ref_right : numpy.ndarray
        Reference-grid hemisphere masks of shape ``(y, x)``.
    ref_buffer_px : float
        Reference-space equivalent of the canonical boundary buffer, in pixels.
    config : dict
        Full configuration, including ROI thresholds.

    Returns
    -------
    pandas.DataFrame
        Per-ROI canonical pixel counts, coverage fractions and gate inputs.

    Notes
    -----
    Canonical ROI areas are the quantitative coverage denominators; reference
    pixel counts are retained for spatial QC."""
    rows: list[dict[str, Any]] = []
    min_px = int(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", "min_pixels", default=50))
    min_abs = float(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", "min_technical_coverage", default=0.5))

    for _, rr in canonical["roi_table"].iterrows():
        bid = int(rr["broad_structure_id"])
        side = safe_str(rr["hemisphere"])
        roi_key = safe_str(rr["roi_key"])
        can_buf = canonical["roi_masks_buffered"][roi_key]
        can_valid = int((can_buf & canonical_technical).sum())
        can_den = int(rr["canonical_buffered_area_px"])
        can_cov = safe_fraction(can_valid, can_den)

        side_ref = ref_left if side == "left" else ref_right
        ref_raw_full_mask = (broad_labelmap_ref_full == bid) & side_ref

        fill_holes = bool(
            cfg_get(
                config,
                "step6b_fc_geometry",
                "roi_geometry",
                "fill_internal_zero_holes_before_buffer",
                default=True,
            )
        )
        hole_connectivity = int(
            cfg_get(
                config,
                "step6b_fc_geometry",
                "roi_geometry",
                "internal_hole_background_connectivity",
                default=4,
            )
        )
        if hole_connectivity not in {4, 8}:
            raise ValueError(
                "step6b_fc_geometry.roi_geometry.internal_hole_background_connectivity "
                "must be 4 or 8"
            )

        max_hole_area_raw = cfg_get(
            config,
            "step6b_fc_geometry",
            "roi_geometry",
            "max_internal_zero_hole_area_px",
            default=None,
        )
        max_hole_area = (
            None
            if max_hole_area_raw is None
            or safe_str(max_hole_area_raw).strip().lower() in {"", "none", "null"}
            else int(max_hole_area_raw)
        )

        if fill_holes:
            ref_allowed_fill = (broad_labelmap_ref_full == 0) & side_ref
            ref_hole_candidates = internal_hole_candidates_bool(
                ref_raw_full_mask,
                background_connectivity=hole_connectivity,
            )
            ref_full_mask, ref_filled_holes = fill_internal_holes_bool(
                ref_raw_full_mask,
                allowed_fill_mask=ref_allowed_fill,
                max_hole_area_px=max_hole_area,
                background_connectivity=hole_connectivity,
            )
            ref_protected_holes = ref_hole_candidates & (~ref_allowed_fill)
        else:
            ref_full_mask = ref_raw_full_mask.copy()
            ref_hole_candidates = np.zeros_like(ref_raw_full_mask, dtype=bool)
            ref_filled_holes = np.zeros_like(ref_raw_full_mask, dtype=bool)
            ref_protected_holes = np.zeros_like(ref_raw_full_mask, dtype=bool)

        ref_buffered_mask = erode_bool(ref_full_mask, ref_buffer_px)
        ref_valid = int((ref_buffered_mask & technical_mask_ref).sum())
        ref_den = int(ref_buffered_mask.sum())

        rows.append({
            "mask_set": mask_set,
            "group_id": group_id,
            "mouse_id": mouse_id,
            "experimental_group": experimental_group,
            "broad_structure_id": bid,
            "broad_acronym": safe_str(rr["broad_acronym"]),
            "broad_name": safe_str(rr["broad_name"]),
            **analysis_roi_metadata(rr),
            "hemisphere": side,
            "roi_key": roi_key,
            # Original/reference-image QC counts.
            "refspace_raw_full_roi_px": int(ref_raw_full_mask.sum()),
            "refspace_enclosed_hole_candidate_px": int(ref_hole_candidates.sum()),
            "refspace_internal_zero_hole_filled_px": int(ref_filled_holes.sum()),
            "refspace_protected_nonzero_hole_px": int(ref_protected_holes.sum()),
            "refspace_hole_background_connectivity": int(hole_connectivity),
            "refspace_full_roi_px": int(ref_full_mask.sum()),
            "refspace_buffered_roi_px": ref_den,
            "refspace_technical_valid_px": ref_valid,
            "refspace_technical_coverage_fraction": safe_fraction(ref_valid, ref_den),
            "refspace_equivalent_boundary_buffer_px": float(ref_buffer_px),
            # Common canonical analysis counts.
            "canonical_full_area_px": int(rr["canonical_full_area_px"]),
            "canonical_buffered_area_px": can_den,
            "canonical_technical_valid_px": can_valid,
            # Main human-facing aliases. These are canonical, not refspace.
            "full_allen_area_px": int(rr["canonical_full_area_px"]),
            "buffered_full_allen_area_px": can_den,
            "technical_px": can_valid,
            "technical_coverage_fraction": can_cov,
            "technical_pixel_pass": bool(can_den >= min_px and can_valid >= min_px),
            "technical_absolute_pass": bool(
                can_den >= min_px and can_valid >= min_px
                and np.isfinite(can_cov) and can_cov >= min_abs
            ),
        })
    return pd.DataFrame(rows)


def build_homotopic_technical_raw_table(
    *,
    mask_set: str,
    group_id: str,
    mouse_id: str,
    experimental_group: str,
    pair_table: pd.DataFrame,
    pair_map: dict[int, dict[str, np.ndarray]],
    canonical_technical: np.ndarray,
    refspace_pairable_full: np.ndarray,
    refspace_pairable_technical: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Measure fixed technical coverage of unique homotopic atlas pixel pairs.

    Parameters
    ----------
    mask_set, group_id, mouse_id, experimental_group : str
        Identifiers for this comparison and mouse.
    pair_table : pandas.DataFrame
        Canonical ROI-pair definitions and pair denominators.
    pair_map : dict
        Canonical left/right partner indices for each analytical ROI.
    canonical_technical : numpy.ndarray
        Fixed technical-validity mask in canonical atlas coordinates.
    refspace_pairable_full, refspace_pairable_technical : numpy.ndarray
        Boolean reference-grid pairability masks used for QC.
    config : dict
        Project configuration and ``step6b_fc_geometry.roi_thresholds``.

    Returns
    -------
    pandas.DataFrame
        One row per homotopic ROI pair and mouse, including valid-pair counts.

    Notes
    -----
    A left/right pair is counted once. Both partners must pass the fixed
    technical mask for that pair to be counted as valid."""
    rows: list[dict[str, Any]] = []
    min_px = int(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", "min_pixels", default=50))
    min_abs = float(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", "min_homotopic_technical_coverage", default=0.5))
    for _, pr in pair_table.iterrows():
        bid = int(pr["broad_structure_id"])
        p = pair_map[bid]
        l_ok = canonical_technical[p["left_ap"], p["left_ml"]]
        r_ok = canonical_technical[p["right_ap"], p["right_ml"]]
        valid_n = int((l_ok & r_ok).sum())
        den = int(pr["canonical_homotopic_pair_px"])
        cov = safe_fraction(valid_n, den)
        rows.append({
            "mask_set": mask_set,
            "group_id": group_id,
            "mouse_id": mouse_id,
            "experimental_group": experimental_group,
            "broad_structure_id": bid,
            "broad_acronym": safe_str(pr["broad_acronym"]),
            "broad_name": safe_str(pr["broad_name"]),
            **analysis_roi_metadata(pr),
            "pair_key": safe_str(pr["pair_key"]),
            "left_roi_key": safe_str(pr["left_roi_key"]),
            "right_roi_key": safe_str(pr["right_roi_key"]),
            "refspace_atlas_pairable_px": int(refspace_pairable_full.sum()) if len(pair_table) == 1 else np.nan,
            "refspace_technical_pairable_px": int(refspace_pairable_technical.sum()) if len(pair_table) == 1 else np.nan,
            "canonical_homotopic_pair_px": den,
            "canonical_homotopic_technical_pair_px": valid_n,
            "homotopic_technical_coverage_fraction": cov,
            "homotopic_pixel_pass": bool(den >= min_px and valid_n >= min_px),
            "homotopic_absolute_pass": bool(
                den >= min_px and valid_n >= min_px
                and np.isfinite(cov) and cov >= min_abs
            ),
        })
    return pd.DataFrame(rows)


def add_relative_gate(
    df: pd.DataFrame,
    *,
    value_col: str,
    reference_col: str,
    relative_col: str,
    key_cols: list[str],
    pixel_pass_col: str,
    absolute_pass_col: str,
    final_pass_col: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate relative technical-coverage criteria within matching cohorts.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-mouse coverage table before relative gating.
    value_col, reference_col, relative_col : str
        Column names for measured coverage, cohort reference and normalized ratio.
    key_cols : list[str]
        Grouping keys defining the cohort comparison for each reference statistic.
    pixel_pass_col, absolute_pass_col, final_pass_col : str
        Input and output columns carrying the separate gate decisions.
    config : dict
        Configuration of ``step6b_fc_geometry.roi_thresholds.cohort_relative``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Annotated coverage rows and cohort-reference summary rows.

    Notes
    -----
    Cohort reference is the configured mean or median. The upper bound is
    reported without excluding records when ``upper_bound_mode='flag'``;
    ``'exclude'`` enables the upper bound as a selection gate."""
    if df.empty:
        return df.copy(), pd.DataFrame()
    out = df.copy()
    rel_cfg = dict(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", "cohort_relative", default={}) or {})
    enabled = bool(rel_cfg.get("enabled", True))
    stat = safe_str(rel_cfg.get("reference_stat", "median")).lower()
    low = float(rel_cfg.get("min_relative_fraction", 0.85))
    high = float(rel_cfg.get("max_relative_fraction", 1.25))
    upper_mode = safe_str(rel_cfg.get("upper_bound_mode", "flag")).lower()
    use_abs = bool(cfg_get(config, "step6b_fc_geometry", "roi_thresholds", "use_absolute_coverage_gate", default=False))
    if stat not in {"median", "mean"}:
        raise ValueError("cohort_relative.reference_stat must be median/mean")
    if upper_mode not in {"flag", "exclude"}:
        raise ValueError("cohort_relative.upper_bound_mode must be flag/exclude")

    refs: list[dict[str, Any]] = []
    for vals, g in out.groupby(key_cols, sort=False, dropna=False):
        if not isinstance(vals, tuple):
            vals = (vals,)
        nums = pd.to_numeric(g[value_col], errors="coerce").dropna()
        ref = float(nums.mean()) if stat == "mean" and not nums.empty else (
            float(nums.median()) if not nums.empty else np.nan
        )
        rec = {k: v for k, v in zip(key_cols, vals)}
        rec.update({
            reference_col: ref,
            "reference_stat": stat,
            "n_mice_reference": int(g["mouse_id"].nunique()),
        })
        refs.append(rec)
    ref_df = pd.DataFrame(refs)
    out = out.merge(ref_df, on=key_cols, how="left", validate="many_to_one")
    ref_num = pd.to_numeric(out[reference_col], errors="coerce")
    val_num = pd.to_numeric(out[value_col], errors="coerce")
    out[relative_col] = np.where(ref_num > 0, val_num / ref_num, np.nan)
    out[f"{relative_col}_low"] = pd.to_numeric(out[relative_col], errors="coerce") < low
    out[f"{relative_col}_high"] = pd.to_numeric(out[relative_col], errors="coerce") > high
    out[f"{relative_col}_in_range"] = (
        pd.to_numeric(out[relative_col], errors="coerce").ge(low)
        & pd.to_numeric(out[relative_col], errors="coerce").le(high)
    )
    lower_pass = pd.to_numeric(out[relative_col], errors="coerce").ge(low) if enabled else pd.Series(True, index=out.index)
    upper_pass = (
        pd.to_numeric(out[relative_col], errors="coerce").le(high)
        if enabled and upper_mode == "exclude"
        else pd.Series(True, index=out.index)
    )
    final = out[pixel_pass_col].astype(bool) & lower_pass & upper_pass
    if use_abs:
        final &= out[absolute_pass_col].astype(bool)
    out[final_pass_col] = final.astype(bool)
    out["cohort_relative_enabled"] = enabled
    out["cohort_relative_min"] = low
    out["cohort_relative_max"] = high
    out["cohort_upper_bound_mode"] = upper_mode
    out["absolute_coverage_gate_enabled"] = use_abs
    return out, ref_df


def add_relative_gate_by_mask_set(
    df: pd.DataFrame,
    *,
    value_col: str,
    reference_col: str,
    relative_col: str,
    key_cols: list[str],
    pixel_pass_col: str,
    absolute_pass_col: str,
    final_pass_col: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply cohort-relative gates separately so each mask_set can use its own thresholds."""
    if df.empty:
        return df.copy(), pd.DataFrame()
    out_parts: list[pd.DataFrame] = []
    ref_parts: list[pd.DataFrame] = []
    for mask_set, g in df.groupby("mask_set", sort=False, dropna=False):
        local_cfg = config_for_mask_set(config, safe_str(mask_set))
        out_g, ref_g = add_relative_gate(
            g,
            value_col=value_col,
            reference_col=reference_col,
            relative_col=relative_col,
            key_cols=key_cols,
            pixel_pass_col=pixel_pass_col,
            absolute_pass_col=absolute_pass_col,
            final_pass_col=final_pass_col,
            config=local_cfg,
        )
        out_parts.append(out_g)
        ref_parts.append(ref_g)
    return (
        pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy(),
        pd.concat(ref_parts, ignore_index=True, sort=False) if ref_parts else pd.DataFrame(),
    )


def add_relative_pass_column_by_mask_set(
    df: pd.DataFrame,
    *,
    relative_col: str,
    pass_col: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate an existing relative-coverage column with per-mask-set bounds."""
    if df.empty:
        return df.copy()
    pieces = []
    for mask_set, g in df.groupby("mask_set", sort=False, dropna=False):
        gg = g.copy()
        thr = thresholds_for_mask_set(config, safe_str(mask_set))
        rel_cfg = dict(thr.get("cohort_relative", {}) or {})
        enabled = bool(rel_cfg.get("enabled", True))
        low = float(rel_cfg.get("min_relative_fraction", 0.85))
        high = float(rel_cfg.get("max_relative_fraction", 1.25))
        mode = safe_str(rel_cfg.get("upper_bound_mode", "flag")).lower()
        rel = pd.to_numeric(gg[relative_col], errors="coerce")
        if enabled:
            passed = rel.ge(low)
            if mode == "exclude":
                passed &= rel.le(high)
        else:
            passed = pd.Series(True, index=gg.index)
        gg[pass_col] = passed.astype(bool)
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0].copy()


def global_analysis_choice_for_mask_set(config: dict[str, Any], mask_set: str) -> str:
    """Resolve the global-analysis hemisphere for the specified mask set."""
    local = cfg_get(
        config,
        "step6b_fc_geometry",
        "mask_sets",
        str(mask_set),
        "global_analysis_hemisphere",
        default=None,
    )
    if local is not None and safe_str(local).strip():
        return safe_str(local).strip().lower()
    return safe_str(
        cfg_get(
            config,
            "step6b_fc_geometry",
            "hemisphere",
            "global_analysis_hemisphere",
            default="healthy",
        )
    ).strip().lower()


def select_global_analysis_mask(
    bilateral: np.ndarray, left: np.ndarray, right: np.ndarray,
    *, choice: str, healthy_side: str, lesion_side: str,
) -> np.ndarray:
    """Build the explicitly named global-analysis spatial product.

    Historical `global_healthy_*` files remain healthy-only for compatibility.
    The new `global_analysis_*` reflects the actual mask-set choice, including both.
    """
    choice = safe_str(choice).lower().strip()
    sides = {"left": np.asarray(left, dtype=bool), "right": np.asarray(right, dtype=bool)}
    if choice == "both":
        selected = sides["left"] | sides["right"]
    else:
        side = healthy_side if choice == "healthy" else lesion_side if choice == "lesion" else choice
        if side not in sides:
            raise ValueError(f"Unknown global analysis hemisphere {choice!r}")
        selected = sides[side]
    return np.asarray(bilateral, dtype=bool) & selected


def build_global_technical_table(
    side_df: pd.DataFrame,
    hemisphere_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Select global-analysis side coverage for the configured hemisphere policy.

    Parameters
    ----------
    side_df : pandas.DataFrame
        Side-specific fixed technical coverage rows.
    hemisphere_df : pandas.DataFrame
        Per-mouse hemisphere assignments.
    config : dict
        Project YAML with the global-analysis hemisphere for each mask set.

    Returns
    -------
    pandas.DataFrame
        Applicable side ROI rows, with global-specific coverage and pass fields.

    Notes
    -----
    The policy can select ``healthy``, ``lesion``, ``left``, ``right``, or
    ``both``. The fixed healthy-only mask remains separate from the more
    general global-analysis product."""
    if side_df.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    hemi_lookup = hemisphere_df.set_index(["mask_set", "group_id"]) if not hemisphere_df.empty else pd.DataFrame()
    for _, r in side_df.iterrows():
        gid = safe_str(r["group_id"])
        side = safe_str(r["hemisphere"])
        choice = global_analysis_choice_for_mask_set(config, safe_str(r["mask_set"]))
        hemi_key = (safe_str(r["mask_set"]), gid)
        if hemi_key not in hemi_lookup.index:
            raise KeyError(f"Missing hemisphere assignment for mask_set/group_id={hemi_key}")
        lesion_side = safe_str(hemi_lookup.loc[hemi_key, "lesion_side"])
        healthy_side = safe_str(hemi_lookup.loc[hemi_key, "healthy_side"])
        if choice == "healthy":
            wanted = {healthy_side}
        elif choice == "lesion":
            wanted = {lesion_side}
        elif choice in {"left", "right"}:
            wanted = {choice}
        elif choice == "both":
            wanted = {"left", "right"}
        else:
            raise ValueError("global_analysis_hemisphere must be healthy/lesion/left/right/both")
        if side not in wanted:
            continue
        rec = r.copy()
        rec["global_analysis_choice"] = choice
        rec["global_analysis_hemisphere"] = side
        rec["global_roi_key"] = safe_str(r["roi_key"])
        rec["global_technical_coverage_fraction"] = r["technical_coverage_fraction"]
        rec["global_relative_to_cohort"] = r["technical_relative_to_cohort"]
        # Global uses the same side-specific canonical coverage but can impose its
        # own optional absolute threshold when absolute gating is enabled.
        local_thr = thresholds_for_mask_set(config, safe_str(r["mask_set"]))
        global_abs_thr = float(local_thr.get("min_global_technical_coverage", 0.5))
        global_abs_pass = bool(
            r["technical_pixel_pass"]
            and np.isfinite(float(r["technical_coverage_fraction"]))
            and float(r["technical_coverage_fraction"]) >= global_abs_thr
        )
        rec["global_absolute_pass"] = global_abs_pass
        use_abs = bool(local_thr.get("use_absolute_coverage_gate", False))
        rec["global_technical_pass"] = bool(r["technical_pass"] and (global_abs_pass if use_abs else True))
        rows.append(rec)
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def build_side_stem_raw_table(
    *,
    mask_set: str,
    slot: str,
    row: pd.Series,
    experimental_group: str,
    canonical: dict[str, Any],
    fixed_technical_can: np.ndarray,
    shadow_can: np.ndarray,
    pathology_can: np.ndarray,
) -> pd.DataFrame:
    """Measure a side ROI's usability at one selected recording stem.

    Parameters
    ----------
    mask_set, slot, experimental_group : str
        Comparison/temporal slot and assigned animal group.
    row : pandas.Series
        Stem metadata from the resolved selection.
    canonical : dict
        Fixed analytical ROI definitions in canonical atlas space.
    fixed_technical_can, shadow_can, pathology_can : numpy.ndarray
        Boolean canonical-grid technical, shadow, and pathology masks.

    Returns
    -------
    pandas.DataFrame
        Per-ROI stem-level usable pixel counts and coverage fractions.

    Notes
    -----
    Shadow and pathology act per recording and do not retroactively remove
    the mouse from the fixed ROI-level cohort selection."""
    rows: list[dict[str, Any]] = []
    for _, rr in canonical["roi_table"].iterrows():
        roi_key = safe_str(rr["roi_key"])
        roi = canonical["roi_masks_buffered"][roi_key]
        fixed = roi & fixed_technical_can
        shadow_lost = fixed & shadow_can
        after_shadow = fixed & (~shadow_can)
        pathology_lost = after_shadow & pathology_can
        final = after_shadow & (~pathology_can)
        fixed_n = int(fixed.sum())
        rows.append({
            "mask_set": mask_set,
            "slot": slot,
            "group_id": safe_str(row.get("group_id")),
            "mouse_id": safe_str(row.get("mouse_id")) or safe_str(row.get("group_id")),
            "experimental_group": experimental_group,
            "stem": safe_str(row.get("stem")),
            "exp_day": safe_str(row.get("exp_day")),
            "state": safe_str(row.get("state")),
            "protocol": safe_str(row.get("protocol")),
            "broad_structure_id": int(rr["broad_structure_id"]),
            "broad_acronym": safe_str(rr["broad_acronym"]),
            "broad_name": safe_str(rr["broad_name"]),
            **analysis_roi_metadata(rr),
            "hemisphere": safe_str(rr["hemisphere"]),
            "roi_key": roi_key,
            "canonical_buffered_area_px": int(rr["canonical_buffered_area_px"]),
            "canonical_fixed_technical_px": fixed_n,
            "canonical_fixed_technical_coverage_fraction": safe_fraction(
                fixed_n, int(rr["canonical_buffered_area_px"])
            ),
            "canonical_shadow_lost_px": int(shadow_lost.sum()),
            "canonical_after_shadow_px": int(after_shadow.sum()),
            "canonical_pathology_lost_px": int(pathology_lost.sum()),
            "canonical_final_usable_px": int(final.sum()),
            "stem_usable_coverage_fraction": safe_fraction(
                int(final.sum()), int(rr["canonical_buffered_area_px"])
            ),  # A: diagnostic only; NOT used as stem PASS
            "stem_usable_fraction_B": safe_fraction(int(final.sum()), fixed_n),
            "shadow_lost_fraction_of_fixed_technical": safe_fraction(int(shadow_lost.sum()), fixed_n),
            "pathology_lost_fraction_after_shadow": safe_fraction(int(pathology_lost.sum()), int(after_shadow.sum())),
            "final_usable_fraction_of_fixed_technical": safe_fraction(int(final.sum()), fixed_n),
        })
    return pd.DataFrame(rows)


def build_homotopic_stem_raw_table(
    *,
    mask_set: str,
    slot: str,
    row: pd.Series,
    experimental_group: str,
    pair_table: pd.DataFrame,
    pair_map: dict[int, dict[str, np.ndarray]],
    fixed_technical_can: np.ndarray,
    shadow_can: np.ndarray,
    pathology_can: np.ndarray,
) -> pd.DataFrame:
    """Measure stem-level usability of the canonical homotopic ROI pairs.

    Parameters
    ----------
    mask_set, slot, experimental_group : str
        Comparison identifiers.
    row : pandas.Series
        Selected recording metadata.
    pair_table : pandas.DataFrame
        Canonical homotopic ROI-pair definitions.
    pair_map : dict
        Left/right partner indices on the canonical atlas grid.
    fixed_technical_can, shadow_can, pathology_can : numpy.ndarray
        Boolean masks on the canonical atlas grid.

    Returns
    -------
    pandas.DataFrame
        Stem-specific valid-pair counts and coverage fractions.

    Notes
    -----
    Each pair contributes only when both mirrored pixels remain usable."""
    rows: list[dict[str, Any]] = []
    for _, pr in pair_table.iterrows():
        bid = int(pr["broad_structure_id"])
        p = pair_map[bid]
        lfix = fixed_technical_can[p["left_ap"], p["left_ml"]]
        rfix = fixed_technical_can[p["right_ap"], p["right_ml"]]
        fixed = lfix & rfix
        shadow_hit = (
            shadow_can[p["left_ap"], p["left_ml"]]
            | shadow_can[p["right_ap"], p["right_ml"]]
        )
        shadow_lost = fixed & shadow_hit
        after_shadow = fixed & (~shadow_hit)
        pathology_hit = (
            pathology_can[p["left_ap"], p["left_ml"]]
            | pathology_can[p["right_ap"], p["right_ml"]]
        )
        pathology_lost = after_shadow & pathology_hit
        final = after_shadow & (~pathology_hit)
        fixed_n = int(fixed.sum())
        rows.append({
            "mask_set": mask_set,
            "slot": slot,
            "group_id": safe_str(row.get("group_id")),
            "mouse_id": safe_str(row.get("mouse_id")) or safe_str(row.get("group_id")),
            "experimental_group": experimental_group,
            "stem": safe_str(row.get("stem")),
            "exp_day": safe_str(row.get("exp_day")),
            "state": safe_str(row.get("state")),
            "protocol": safe_str(row.get("protocol")),
            "broad_structure_id": bid,
            "broad_acronym": safe_str(pr["broad_acronym"]),
            "broad_name": safe_str(pr["broad_name"]),
            **analysis_roi_metadata(pr),
            "pair_key": safe_str(pr["pair_key"]),
            "canonical_homotopic_pair_px": int(pr["canonical_homotopic_pair_px"]),
            "canonical_fixed_technical_pair_px": fixed_n,
            "canonical_fixed_technical_pair_coverage_fraction": safe_fraction(
                fixed_n, int(pr["canonical_homotopic_pair_px"])
            ),
            "canonical_shadow_lost_pair_px": int(shadow_lost.sum()),
            "canonical_after_shadow_pair_px": int(after_shadow.sum()),
            "canonical_pathology_lost_pair_px": int(pathology_lost.sum()),
            "canonical_final_usable_pair_px": int(final.sum()),
            "stem_usable_pair_coverage_fraction": safe_fraction(
                int(final.sum()), int(pr["canonical_homotopic_pair_px"])
            ),  # A: diagnostic only
            "stem_usable_pair_fraction_B": safe_fraction(int(final.sum()), fixed_n),
            "shadow_lost_fraction_of_fixed_technical": safe_fraction(int(shadow_lost.sum()), fixed_n),
            "pathology_lost_fraction_after_shadow": safe_fraction(int(pathology_lost.sum()), int(after_shadow.sum())),
            "final_usable_fraction_of_fixed_technical": safe_fraction(int(final.sum()), fixed_n),
        })
    return pd.DataFrame(rows)


def finalize_side_stem_passes(side_stem: pd.DataFrame, side_tech: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Attach fixed eligibility and calculate side-ROI usability pass per stem."""
    if side_stem.empty:
        return side_stem.copy()
    keys = ["mask_set", "group_id", "mouse_id", "experimental_group", "broad_structure_id", "broad_acronym", "broad_name", "hemisphere", "roi_key"]
    right = side_tech[keys + ["technical_pass", "technical_relative_to_cohort"]].drop_duplicates(keys)
    out = side_stem.merge(right, on=keys, how="left", validate="many_to_one")
    pieces = []
    for mask_set, g in out.groupby("mask_set", sort=False, dropna=False):
        gg = g.copy()
        thr = thresholds_for_mask_set(config, safe_str(mask_set))
        min_px = int(thr.get("min_pixels", 50))
        min_frac = float(thr.get("min_stem_usable_coverage", 0.5))
        gg["stem_min_pixels"] = min_px
        gg["stem_min_usable_coverage"] = min_frac
        gg["side_stem_pass"] = (
            gg["technical_pass"].fillna(False).astype(bool)
            & pd.to_numeric(gg["canonical_final_usable_px"], errors="coerce").ge(min_px)
            & pd.to_numeric(gg["canonical_fixed_technical_px"], errors="coerce").gt(0)
            & pd.to_numeric(gg["stem_usable_fraction_B"], errors="coerce").ge(min_frac)
        )
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True) if pieces else out.iloc[0:0].copy()


def finalize_homotopic_stem_passes(hstem: pd.DataFrame, htech: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Attach fixed eligibility and calculate homotopic usability pass per stem."""
    if hstem.empty:
        return hstem.copy()
    keys = ["mask_set", "group_id", "mouse_id", "experimental_group", "broad_structure_id", "broad_acronym", "broad_name", "pair_key"]
    right = htech[keys + ["homotopic_technical_pass", "homotopic_relative_to_cohort"]].drop_duplicates(keys)
    out = hstem.merge(right, on=keys, how="left", validate="many_to_one")
    pieces = []
    for mask_set, g in out.groupby("mask_set", sort=False, dropna=False):
        gg = g.copy()
        thr = thresholds_for_mask_set(config, safe_str(mask_set))
        min_px = int(thr.get("min_pixels", 50))
        min_frac = float(thr.get("min_stem_usable_coverage", 0.5))
        gg["stem_min_pairs"] = min_px
        gg["stem_min_usable_pair_coverage"] = min_frac
        gg["homotopic_stem_pass"] = (
            gg["homotopic_technical_pass"].fillna(False).astype(bool)
            & pd.to_numeric(gg["canonical_final_usable_pair_px"], errors="coerce").ge(min_px)
            & pd.to_numeric(gg["canonical_fixed_technical_pair_px"], errors="coerce").gt(0)
            & pd.to_numeric(gg["stem_usable_pair_fraction_B"], errors="coerce").ge(min_frac)
        )
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True) if pieces else out.iloc[0:0].copy()


def build_global_stem_table(side_stem: pd.DataFrame, global_tech: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Combine stem-level side usability with the selected global ROI coverage.

    Parameters
    ----------
    side_stem : pandas.DataFrame
        Stem-level side ROI usability table.
    global_tech : pandas.DataFrame
        Global-analysis side selection and technical coverage table.
    config : dict
        Project configuration, including per-mask-set usability thresholds.

    Returns
    -------
    pandas.DataFrame
        Applicable stem-level global-analysis rows and pass decisions."""
    if side_stem.empty or global_tech.empty:
        return pd.DataFrame()
    keys = ["mask_set", "group_id", "mouse_id", "experimental_group", "broad_structure_id", "broad_acronym", "broad_name", "hemisphere", "roi_key"]
    gt = global_tech[keys + ["global_analysis_choice", "global_analysis_hemisphere", "global_roi_key", "global_technical_pass", "global_relative_to_cohort"]].drop_duplicates(keys)
    out = side_stem.merge(gt, on=keys, how="inner", validate="many_to_one")
    pieces = []
    for mask_set, g in out.groupby("mask_set", sort=False, dropna=False):
        gg = g.copy()
        thr = thresholds_for_mask_set(config, safe_str(mask_set))
        min_px = int(thr.get("min_pixels", 50))
        min_frac = float(thr.get("min_stem_usable_coverage", 0.5))
        gg["stem_min_pixels"] = min_px
        gg["stem_min_usable_coverage"] = min_frac
        gg["global_stem_pass"] = (
            gg["global_technical_pass"].fillna(False).astype(bool)
            & pd.to_numeric(gg["canonical_final_usable_px"], errors="coerce").ge(min_px)
            & pd.to_numeric(gg["canonical_fixed_technical_px"], errors="coerce").gt(0)
            & pd.to_numeric(gg["stem_usable_fraction_B"], errors="coerce").ge(min_frac)
        )
        pieces.append(gg)
    return pd.concat(pieces, ignore_index=True) if pieces else out.iloc[0:0].copy()



def build_v45_eligibility(
    htech: pd.DataFrame,
    hstem: pd.DataFrame,
    gtech: pd.DataFrame,
    gstem: pd.DataFrame,
) -> pd.DataFrame:
    """Combine fixed and stem-level tables into per-mouse analytical ROI eligibility.

    Parameters
    ----------
    htech, hstem : pandas.DataFrame
        Fixed and stem-specific homotopic coverage tables.
    gtech, gstem : pandas.DataFrame
        Fixed and stem-specific global-analysis coverage tables.

    Returns
    -------
    pandas.DataFrame
        Per-mouse, ROI and analysis eligibility, including diagnostic stem fields.

    Notes
    -----
    ROI-level eligibility follows fixed technical representation. The
    ``all_selected_stems_pass`` field is only diagnostic, not a selection gate."""
    rows: list[dict[str, Any]] = []
    if not htech.empty:
        keycols = ["mask_set", "group_id", "mouse_id", "experimental_group", "broad_structure_id", "broad_acronym", "broad_name", "pair_key"]
        for vals, t in htech.groupby(keycols, sort=False, dropna=False):
            rec = dict(zip(keycols, vals))
            t0 = t.iloc[0]
            s = hstem[
                (hstem["mask_set"].astype(str) == str(rec["mask_set"]))
                & (hstem["group_id"].astype(str) == str(rec["group_id"]))
                & (hstem["pair_key"].astype(str) == str(rec["pair_key"]))
            ]
            n = len(s)
            n_pass = int(s["homotopic_stem_pass"].astype(bool).sum()) if n else 0
            rows.append({
                **rec,
                "analysis": "homotopic",
                "roi_key": rec["pair_key"],
                "hemisphere": "LR_pair",
                "pixel_pass": bool(t0.get("homotopic_pixel_pass", False)),
                "relative_pass": bool(t0.get("homotopic_relative_pass", False)),
                "technical_pass": bool(t0.get("homotopic_technical_pass", False)),
                "relative_to_cohort": t0.get("homotopic_relative_to_cohort", np.nan),
                "n_selected_stems": int(n),
                "n_stems_pass": n_pass,
                "n_stems_fail": int(n - n_pass),
                "any_selected_stem_pass": bool(n_pass > 0),
                "all_selected_stems_pass": bool(n > 0 and n_pass == n),
            })
    if not gtech.empty:
        keycols = ["mask_set", "group_id", "mouse_id", "experimental_group", "broad_structure_id", "broad_acronym", "broad_name", "hemisphere", "global_roi_key"]
        for vals, t in gtech.groupby(keycols, sort=False, dropna=False):
            rec = dict(zip(keycols, vals))
            t0 = t.iloc[0]
            s = gstem[
                (gstem["mask_set"].astype(str) == str(rec["mask_set"]))
                & (gstem["group_id"].astype(str) == str(rec["group_id"]))
                & (gstem["global_roi_key"].astype(str) == str(rec["global_roi_key"]))
            ]
            n = len(s)
            n_pass = int(s["global_stem_pass"].astype(bool).sum()) if n else 0
            rows.append({
                **rec,
                "analysis": "global",
                "roi_key": rec["global_roi_key"],
                "pixel_pass": bool(t0.get("technical_pixel_pass", False)),
                "relative_pass": bool(t0.get("technical_relative_pass", False)),
                "technical_pass": bool(t0.get("global_technical_pass", False)),
                "relative_to_cohort": t0.get("global_relative_to_cohort", np.nan),
                "n_selected_stems": int(n),
                "n_stems_pass": n_pass,
                "n_stems_fail": int(n - n_pass),
                "any_selected_stem_pass": bool(n_pass > 0),
                "all_selected_stems_pass": bool(n > 0 and n_pass == n),
            })
    return pd.DataFrame(rows)


def build_v45_group_tables(
    eligibility: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select group-level analytical ROIs from mouse-level technical eligibility.

    Parameters
    ----------
    eligibility : pandas.DataFrame
        Per-mouse analytical ROI inclusion and QC decisions.
    config : dict
        Project configuration, including group fraction and ROI gate settings.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Group eligibility summaries, selected ROI tables and completeness QC.

    Notes
    -----
    Stem-level completeness is not used to exclude all other valid time
    points for the same mouse and ROI."""
    if eligibility.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    group_rows: list[dict[str, Any]] = []
    gkeys = ["mask_set", "analysis", "experimental_group", "broad_structure_id", "broad_acronym", "broad_name", "roi_key", "hemisphere"]
    for vals, g in eligibility.groupby(gkeys, sort=False, dropna=False):
        rec = dict(zip(gkeys, vals))
        n = int(g["mouse_id"].nunique())
        nt = int(g.loc[g["technical_pass"].astype(bool), "mouse_id"].nunique())
        na = int(g.loc[g["all_selected_stems_pass"].astype(bool), "mouse_id"].nunique())
        ny = int(g.loc[g["any_selected_stem_pass"].astype(bool), "mouse_id"].nunique())
        group_rows.append({
            **rec,
            "n_mice_total": n,
            "n_mice_technical_pass": nt,
            "technical_pass_fraction": safe_fraction(nt, n),
            "n_mice_with_any_stem_pass": ny,
            "any_stem_pass_fraction": safe_fraction(ny, n),
            "n_mice_all_selected_stems_pass": na,
            "all_selected_stems_pass_fraction": safe_fraction(na, n),
        })
    group_df = pd.DataFrame(group_rows)

    selected_rows: list[dict[str, Any]] = []
    skeys = ["mask_set", "analysis", "broad_structure_id", "broad_acronym", "broad_name", "roi_key", "hemisphere"]
    for vals, g in group_df.groupby(skeys, sort=False, dropna=False):
        rec = dict(zip(skeys, vals))
        thr = thresholds_for_mask_set(config, safe_str(rec["mask_set"]))
        min_frac = float(thr.get("min_mouse_fraction_per_group", 0.80))
        require_all = bool(thr.get("require_all_groups", True))
        ok = g["technical_pass_fraction"].ge(min_frac)
        expected = list((cfg_get(config, "step6b_fc_geometry", "mask_sets", str(vals[0]), default={}) or {}).get("expected_comparison_groups", []))
        present = set(g["experimental_group"].astype(str))
        missing_groups = sorted(set(map(str, expected)) - present)
        include = (bool(ok.all()) if require_all else bool(ok.any())) and not missing_groups
        selected_rows.append({
            "missing_expected_comparison_groups_json": json.dumps(missing_groups, ensure_ascii=False),
            **rec,
            "include": include,
            "selection_basis": "technical_pass_fraction_only",
            "min_mouse_fraction_required": min_frac,
            "require_all_groups": require_all,
            "n_groups": int(g["experimental_group"].nunique()),
            "min_group_technical_pass_fraction": float(g["technical_pass_fraction"].min()),
            "exclusion_reason": ("coverage_ok" if include else
                                 "missing_expected_comparison_groups" if missing_groups else
                                 "insufficient_mouse_technical_representation"),
            "group_details_json": json.dumps(
                g[[
                    "experimental_group",
                    "n_mice_total",
                    "n_mice_technical_pass",
                    "technical_pass_fraction",
                    "n_mice_all_selected_stems_pass",
                    "all_selected_stems_pass_fraction",
                ]].to_dict("records"),
                ensure_ascii=False,
            ),
        })
    selected = pd.DataFrame(selected_rows)
    primary = safe_str(
        cfg_get(config, "step6b_fc_geometry", "primary_roi_mask_set", default="stroke_longitudinal")
    )
    selected_primary = selected[selected["mask_set"].astype(str).eq(primary)].copy()

    funnel_rows: list[dict[str, Any]] = []
    fkeys = ["mask_set", "analysis", "experimental_group", "roi_key", "broad_acronym", "hemisphere"]
    for vals, g in eligibility.groupby(fkeys, sort=False, dropna=False):
        rec = dict(zip(fkeys, vals))
        n = int(g["mouse_id"].nunique())
        funnel_rows.append({
            **rec,
            "n_expected": n,
            "n_pixel_pass": int(g.loc[g["pixel_pass"].astype(bool), "mouse_id"].nunique()),
            "n_relative_pass": int(g.loc[g["relative_pass"].astype(bool), "mouse_id"].nunique()),
            "n_technical_pass": int(g.loc[g["technical_pass"].astype(bool), "mouse_id"].nunique()),
            "n_with_any_stem_pass": int(g.loc[g["any_selected_stem_pass"].astype(bool), "mouse_id"].nunique()),
            "n_all_selected_stems_pass": int(g.loc[g["all_selected_stems_pass"].astype(bool), "mouse_id"].nunique()),
            "all_selected_stems_pass_is_qc_only": True,
        })
    return group_df, selected, selected_primary, pd.DataFrame(funnel_rows)


# =============================================================================
# Main processing
# =============================================================================


def group_reference_labelmap_path(paths: dict[str, Path], group_id: str, ref_stem: str) -> Path:
    """Locate the group-specific atlas labels in reference-image coordinates."""
    key = safe_key(group_id, ref_stem)
    return Path(paths["labelmaps_refspace_root"]) / f"{key}_atlas_labels_refspace.npy"


def rows_for_resolved_selection(df: pd.DataFrame, selection_group: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Resolve selected slot/stem records to unique Step6 index rows."""
    rows: list[tuple[str, pd.Series]] = []
    for _, selrow in selection_group.iterrows():
        stem = safe_str(selrow.get("stem"))
        match = df[df["stem"].astype(str).eq(stem)]
        match = match[match["group_id"].astype(str).eq(safe_str(selrow.get("group_id")))]
        if len(match) != 1:
            raise RuntimeError(f"Resolved selection stem={stem} does not map uniquely back to Step6 index")
        rows.append((safe_str(selrow.get("slot")), match.iloc[0]))
    return rows


def _save_stem_inputs(root: Path, masks: StemReferenceMasks) -> dict[str, str]:
    """Save reference-space stem masks and return their output paths."""
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "brain_ref": root / "brain_ref.npy",
        "technical_artifact_ref": root / "technical_artifact_ref.npy",
        "bad_ref": root / "bad_ref.npy",
        "bregma_ref": root / "bregma_ref.npy",
        "rescue_shadow_ref": root / "rescue_shadow_ref.npy",
        "pathology_ref": root / "pathology_ref.npy",
    }
    np.save(files["brain_ref"], masks.brain_ref.astype(np.uint8))
    np.save(files["technical_artifact_ref"], masks.technical_artifact_ref.astype(np.uint8))
    np.save(files["bad_ref"], masks.bad_ref.astype(np.uint8))
    np.save(files["bregma_ref"], masks.bregma_ref.astype(np.uint8))
    np.save(files["rescue_shadow_ref"], masks.rescue_shadow_ref.astype(np.uint8))
    np.save(files["pathology_ref"], masks.pathology_ref.astype(np.uint8))
    return {k + "_path": str(v) for k, v in files.items()}


def run_step6b(
    config_path: str | Path,
    *,
    mask_sets: Sequence[str] | None = None,
    groups: Sequence[str] | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    rest_variant: str = "without_rest",
) -> dict[str, pd.DataFrame]:
    """Run Step6b spatial geometry, cohort coverage, and ROI eligibility analysis.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the complete project YAML configuration.
    mask_sets : sequence of str, optional
        Named ``step6b_fc_geometry.mask_sets`` to consider; ``None`` uses configured sets.
    groups : sequence of str, optional
        Restrict execution to selected Step6 registration groups.
    dry_run : bool, optional
        Perform selection-oriented checks without constructing spatial outputs.
        Selection and audit tables may still be written.
    overwrite : bool, optional
        Allow overwriting files where the workflow supports it.
    rest_variant : {"without_rest", "with_rest", "both"}, optional
        Temporal eligibility branch. ``both`` runs the two branches separately.

    Returns
    -------
    dict
        Named DataFrames for selection, ROI geometry, technical coverage,
        stem-level usability, and cohort eligibility. For ``both`` the result
        is keyed first by REST variant.

    Notes
    -----
    Technical cohort selection precedes stem-specific shadow and pathology
    exclusions. ``all_selected_stems_pass`` is diagnostic rather than a group
    ROI-selection gate. This function writes analysis tables even during the
    selection-oriented dry run; do not treat it as read-only."""
    if rest_variant == "both":
        return {variant: run_step6b(config_path, mask_sets=mask_sets, groups=groups,
                                   dry_run=dry_run, overwrite=overwrite, rest_variant=variant)
                for variant in ["without_rest", "with_rest"]}
    if rest_variant not in {"without_rest", "with_rest"}:
        raise ValueError("rest_variant must be without_rest, with_rest, or both")
    pre_config = ensure_step6b_defaults(load_config(Path(config_path)))
    include_bad = bool(cfg_get(pre_config, "step6b_fc_geometry", "include_bad_rows", default=False))

    df, config, paths = build_step6_index(
        config_path,
        include_bad=True,
        require_response_roi=False,
    )
    ensure_step6b_defaults(config)
    # is_bad is logged but not a science gate; technical missing-data failures remain explicit.
    # Never filter on Step2's unapproved automated score thresholds.
    metadata_full = pd.read_csv(paths["metadata_registration_csv"])
    df, exclusion_audit, exclusion_rules = apply_analysis_exclusions_to_step6b(df, config, metadata_full)
    config["step6b_fc_geometry"]["active_rest_variant"] = rest_variant
    out_paths = build_step6b_paths(config, paths)
    exclusion_audit.to_csv(out_paths["excluded_stems_csv"], index=False)
    exclusion_rules.to_csv(out_paths["exclusion_rules_csv"], index=False)
    rest_eligible = load_rest_eligible_stems(config, paths) if rest_variant == "with_rest" else set()
    df[["stem", "mouse_id", "exp_day", "state", "analysis_excluded"]].assign(
        rest_60s_accepted=lambda x: x["stem"].astype(str).isin(rest_eligible),
        rest_policy=lambda x: np.where(x["state"].eq("Iso"), "not_applicable_anest",
            "not_checked" if rest_variant == "without_rest" else np.where(x["stem"].isin(rest_eligible), "accepted_rest", "no_rest"))
    ).to_csv(out_paths["rest_availability_csv"], index=False)
    assignment_audit = build_mouse_group_assignment_audit(df, config)
    assignment_audit.to_csv(out_paths["assignment_audit_csv"], index=False)
    selection = resolve_comparison_selection(df, config, mask_sets=mask_sets, groups=groups,
                                             rest_variant=rest_variant, rest_eligible=rest_eligible)
    selection.to_csv(out_paths["selection_csv"], index=False)
    skipped_mask_sets = build_skipped_mask_sets_table(selection, config)
    skipped_mask_sets.to_csv(out_paths["skipped_mask_sets_csv"], index=False)
    mask_set_cohort_summary = build_mask_set_cohort_summary(selection, skipped_mask_sets, config)
    mask_set_cohort_summary.to_csv(out_paths["mask_set_cohort_summary_csv"], index=False)
    # Dry-run overview deliberately has unknown n_mice_masks_built.
    # Keep an existing completed overview intact during a selection-only dry-run.
    cohort_overview = build_cohort_overview(selection, config)

    cfg_sets = dict(cfg_get(config, "step6b_fc_geometry", "mask_sets", default={}) or {})
    required_problem: list[dict[str, Any]] = []
    if not selection.empty:
        for _, r in selection[~selection["status"].eq("ok")].iterrows():
            mset = safe_str(r.get("mask_set"))
            if bool((cfg_sets.get(mset, {}) or {}).get("required", False)):
                required_problem.append(r.to_dict())
    if dry_run:
        return {
            "selection": selection,
            "assignment_audit": assignment_audit,
            "required_selection_problems": pd.DataFrame(required_problem),
            "skipped_mask_sets": skipped_mask_sets,
            "mask_set_cohort_summary": mask_set_cohort_summary,
            "cohort_overview": cohort_overview,
            "analysis_exclusions": exclusion_audit,
            "exclusion_rules": exclusion_rules,
        }
    if required_problem and bool(cfg_get(config, "step6b_fc_geometry", "selection", "fail_on_required_selection_problem", default=False)):
        raise RuntimeError(
            f"Required Step6b mask-set selection has {len(required_problem)} unresolved slots. "
            f"Inspect {out_paths['selection_csv']}."
        )
    if not skipped_mask_sets.empty:
        print(f"[Step6b v4.12] Skipping {len(skipped_mask_sets)} incomplete mouse × mask-set combinations.")
        for _, sr in skipped_mask_sets.iterrows():
            print(f"  SKIP mouse={sr['mouse_id']} mask_set={sr['mask_set']} unresolved={sr['unresolved_slots_json']}")

    complete_keys: list[tuple[str, str]] = []
    for (mset, gid), sg in selection.groupby(["mask_set", "group_id"], sort=False):
        if selection_group_complete(sg, config):
            complete_keys.append((str(mset), str(gid)))

    atlas_data = load_allen_atlas(config)
    parent_query = safe_str(cfg_get(config, "step6b_fc_geometry", "broad_atlas", "parent_acronym", default="Isocortex"))
    canonical = build_canonical_side_rois(atlas_data, config, paths, parent_query)
    pair_table, pair_map = build_canonical_homotopic_pairs(canonical)
    canonical["roi_table"].to_csv(out_paths["canonical_roi_csv"], index=False)
    canonical.get("hole_repair_table", pd.DataFrame()).to_csv(
        out_paths["canonical_hole_repair_csv"], index=False
    )
    pair_table.to_csv(out_paths["canonical_pair_csv"], index=False)
    # Backward-compatible filenames retained. In v4.9 the default analytical
    # canonical map is the hole-repaired geometry used downstream; raw geometry
    # is saved separately for QC/audit.
    np.save(out_paths["canonical_root"] / "broad_isocortex_canonical_raw.npy", canonical["broad_full_raw"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "broad_isocortex_canonical.npy", canonical["broad_full"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "analysis_roi_canonical_raw.npy", canonical["broad_full_raw"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "broad_side_labels_canonical.npy", canonical["full_side_labels"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "broad_side_labels_buffered_canonical.npy", canonical["buffered_side_labels"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "analysis_roi_canonical.npy", canonical["broad_full"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "analysis_roi_side_labels_canonical.npy", canonical["full_side_labels"].astype(np.int32))
    np.save(out_paths["canonical_root"] / "analysis_roi_side_labels_buffered_canonical.npy", canonical["buffered_side_labels"].astype(np.int32))

    mapping_df = canonical["mapping"].copy()
    children_df = canonical["children"].copy()
    refinement_df = canonical.get("roi_refinements", pd.DataFrame()).copy()
    mapping_df.to_csv(out_paths["mapping_csv"], index=False)
    children_df.to_csv(out_paths["direct_children_csv"], index=False)
    refinement_df.to_csv(out_paths["roi_refinement_csv"], index=False)

    manual_overrides = load_manual_registration_overrides(config, paths)
    shadow_map, shadow_table_source = load_shadow_map(config, paths)
    exp_groups = experimental_group_map(config)
    stem_cache: dict[tuple[str, str], StemReferenceMasks] = {}
    stem_log_records: dict[tuple[str, str], dict[str, Any]] = {}
    hemisphere_rows: list[dict[str, Any]] = []
    mask_set_summary_rows: list[dict[str, Any]] = []
    side_technical_parts: list[pd.DataFrame] = []
    homotopic_technical_parts: list[pd.DataFrame] = []
    side_stem_parts: list[pd.DataFrame] = []
    homotopic_stem_parts: list[pd.DataFrame] = []
    error_rows: list[dict[str, Any]] = []
    side_consistency_rows: list[dict[str, Any]] = []
    reference_hole_repair_parts: list[pd.DataFrame] = []
    cohort_fixed_masks: dict[str, list[tuple[str, str, np.ndarray]]] = {}
    continue_on_error = bool(cfg_get(config, "step6b_fc_geometry", "continue_on_error", default=False))

    # Cache per mouse/mask-set canonical projection and fixed technical validity.
    for mask_set, gid in complete_keys:
        sg = selection[(selection["mask_set"].astype(str) == mask_set) & (selection["group_id"].astype(str) == gid) & selection["status"].eq("ok")].copy()
        mouse_id = safe_str(sg.iloc[0].get("mouse_id")) or gid
        experimental_group = exp_groups.get(mouse_id, safe_str(sg.iloc[0].get("experimental_group")) or "UNASSIGNED")
        try:
            local_config = config_for_mask_set(config, mask_set)
            selected = rows_for_resolved_selection(df, sg)
            selected_rows = [r for _, r in selected]
            ref_stems = {safe_str(r.get("ref_stem")) for r in selected_rows}
            if len(ref_stems) != 1:
                raise RuntimeError(f"Mask set {mask_set}/{gid} spans multiple ref_stems: {sorted(ref_stems)}")
            ref_stem = next(iter(ref_stems))
            group_rows = df[df["group_id"].astype(str).eq(gid)]
            ref_rows = group_rows[group_rows["stem"].astype(str).eq(ref_stem)]
            ref_row = ref_rows.iloc[0] if not ref_rows.empty else selected_rows[0]
            ref_img = load_reference_image_for_row(ref_row)
            ref_shape = tuple(ref_img.shape[:2])
            basis = reference_basis_from_row(ref_row, config)

            label_path = group_reference_labelmap_path(paths, gid, ref_stem)
            if not label_path.exists():
                raise FileNotFoundError(f"Step6 reference atlas labelmap missing: {label_path}")
            original_labelmap = np.load(label_path).astype(np.int32)
            if tuple(original_labelmap.shape) != ref_shape:
                raise ValueError(f"Reference labelmap shape {original_labelmap.shape} != reference image {ref_shape}")
            broad_ref_full, _, _ = build_broad_labelmap(original_labelmap, atlas_data, parent_query, config=config)

            transform_path = group_transform_path(paths, gid, ref_stem)
            if not transform_path.exists():
                raise FileNotFoundError(f"Step6 atlas transform missing: {transform_path}")
            transform = load_json_file(transform_path)
            projection = project_canonical_grid_to_reference(atlas_data, transform, ref_shape, config)

            ref_left, ref_right, midline_ref, lesion_side, healthy_side = reference_side_masks_for_mouse(
                ref_shape, basis, config, mouse_id, gid
            )
            side_qc = canonical_reference_side_consistency(
                canonical=canonical,
                projection=projection,
                ref_left=ref_left,
                ref_right=ref_right,
                config=config,
            )
            side_qc.update({
                "mask_set": mask_set,
                "group_id": gid,
                "mouse_id": mouse_id,
                "experimental_group": experimental_group,
                "ref_stem": ref_stem,
                "atlas_ml_sign": float(cfg_get(config, "step6", "atlas", "ml_sign", default=1.0)),
                "atlas_left_ml_relation_to_bregma": safe_str(cfg_get(config, "step6", "atlas", "left_ml_relation_to_bregma", default="negative")),
                "image_ml_to_image_sign": float(cfg_get(config, "step6", "geometry", "ml_to_image_sign", default=1.0)),
                "image_ml_positive_label": safe_str(cfg_get(config, "step6", "side_detection", "ml_positive_label", default="right")),
            })
            side_consistency_rows.append(side_qc)
            pd.DataFrame(side_consistency_rows).to_csv(out_paths["side_consistency_csv"], index=False)
            side_qc_enabled = bool(cfg_get(config, "step6b_fc_geometry", "side_consistency_qc", "enabled", default=True))
            fail_on_swap = bool(cfg_get(config, "step6b_fc_geometry", "side_consistency_qc", "fail_on_swap", default=True))
            if side_qc_enabled and fail_on_swap and side_qc["side_mapping_status"] == "swapped":
                raise RuntimeError(
                    "Canonical/reference LEFT-RIGHT mapping is swapped for "
                    f"mouse={mouse_id}, group={gid}, mask_set={mask_set}. "
                    f"Check step6.atlas.ml_sign and Step6 transforms. QC={side_qc}"
                )
            hom_choice = safe_str(cfg_get(config, "step6b_fc_geometry", "hemisphere", "homotopic_map_hemisphere", default="lesion"))
            global_choice = global_analysis_choice_for_mask_set(config, mask_set)
            hemisphere_rows.append({
                "mask_set": mask_set,
                "group_id": gid,
                "mouse_id": mouse_id,
                "experimental_group": experimental_group,
                "ref_stem": ref_stem,
                "lesion_side": lesion_side,
                "healthy_side": healthy_side,
                "homotopic_map_hemisphere": hom_choice,
                "global_analysis_hemisphere": global_choice,
                "global_mask_for_actual_choice": "global_analysis_fixed_technical_ref.npy",
                "legacy_global_healthy_is_always_healthy_only": True,
                "canonical_left_right_source": "Allen ML relative to canonical bregma midline",
            })

            # Fixed longitudinal technical mask excludes rescue shadow in v4.10.
            brain_refs: list[np.ndarray] = []
            fixed_artifact_refs: list[np.ndarray] = []
            masks_by_stem: dict[str, StemReferenceMasks] = {}
            for _, row in selected:
                stem = safe_str(row.get("stem"))
                cache_key = (gid, stem)
                if cache_key not in stem_cache:
                    masks = build_stem_reference_masks(
                        row, config, paths, ref_shape, manual_overrides,
                        shadow_map, shadow_table_source,
                    )
                    stem_cache[cache_key] = masks
                    input_dir = out_paths["stem_inputs_root"] / safe_key(gid, stem)
                    saved_paths = _save_stem_inputs(input_dir, masks)
                    meta = dict(masks.source_meta)
                    meta.update(saved_paths)
                    meta["rescue_shadow_role_v45"] = "stem_specific_after_fixed_technical"
                    stem_log_records[cache_key] = meta
                masks = stem_cache[cache_key]
                masks_by_stem[stem] = masks
                brain_refs.append(masks.brain_ref)
                fixed_artifact_refs.append(masks.technical_artifact_ref)

            common_brain = np.logical_and.reduce(brain_refs)
            common_brain_edge = common_brain_for_technical_analysis(
                common_brain, local_config
            )
            common_brain_for_technical = common_brain_edge["for_technical"]
            fixed_artifact_union = np.logical_or.reduce(fixed_artifact_refs)
            fixed_technical_ref = common_brain_for_technical & (~fixed_artifact_union)
            fixed_technical_can = sample_ref_mask_on_canonical(projection, fixed_technical_ref)

            spatial_fixed_can = canonical_spatial_masks(
                valid_can=fixed_technical_can,
                canonical=canonical,
                pair_map=pair_map,
                healthy_side=healthy_side,
                lesion_side=lesion_side,
                homotopic_display_choice=safe_str(
                    cfg_get(config, "step6b_fc_geometry", "hemisphere", "homotopic_map_hemisphere", default="lesion")
                ),
            )
            canonical_mask_dir = out_paths["canonical_masks_root"] / safe_key(mask_set, gid)
            canonical_mask_dir.mkdir(parents=True, exist_ok=True)
            np.save(canonical_mask_dir / "fixed_technical_canonical.npy", fixed_technical_can.astype(np.uint8))
            np.save(canonical_mask_dir / "bilateral_fixed_technical_canonical.npy", spatial_fixed_can["bilateral"].astype(np.uint8))
            np.save(canonical_mask_dir / "global_healthy_fixed_technical_canonical.npy", spatial_fixed_can["global_healthy"].astype(np.uint8))
            np.save(canonical_mask_dir / "global_analysis_fixed_technical_canonical.npy",
                    select_global_analysis_mask(spatial_fixed_can["bilateral"], canonical["left_mask"],
                                                canonical["right_mask"], choice=global_choice,
                                                healthy_side=healthy_side, lesion_side=lesion_side).astype(np.uint8))
            np.save(canonical_mask_dir / "homotopic_pairable_bilateral_canonical.npy", spatial_fixed_can["homotopic_pairable_bilateral"].astype(np.uint8))
            np.save(canonical_mask_dir / "homotopic_pairable_display_canonical.npy", spatial_fixed_can["homotopic_pairable_display"].astype(np.uint8))
            cohort_fixed_masks.setdefault(mask_set, []).append(
                (mouse_id, gid, spatial_fixed_can["bilateral"].astype(bool))
            )

            # v4.10 reference analytical atlas branch. Canonical geometry remains
            # authoritative for denominators/eligibility; this repaired reference
            # atlas is authoritative for Step7 reference->stem geometry.
            hom_display_mask = ref_left if ((hom_choice == "left") or (hom_choice == "healthy" and healthy_side == "left") or (hom_choice == "lesion" and lesion_side == "left")) else ref_right
            ref_buffer_px = reference_equivalent_buffer_px(canonical["boundary_px"], atlas_data, basis)

            ref_analysis_repair = repair_reference_analysis_labelmap(
                broad_ref_full,
                ref_left,
                ref_right,
                local_config,
                group_id=gid,
                mask_set=mask_set,
                mouse_id=mouse_id,
                roi_definitions=canonical["children"],
            )
            analysis_roi_ref_raw = ref_analysis_repair["raw"]
            analysis_roi_ref_repaired = ref_analysis_repair["repaired"]
            analysis_roi_ref_buffered_repaired = buffered_broad_labelmap(
                analysis_roi_ref_repaired,
                ref_buffer_px,
            )
            # Legacy pre-v4.10 QC product: erosion directly from the un-repaired
            # analytical labelmap. Kept only for backward compatibility/audit.
            ref_broad_buffer_legacy_qc = buffered_broad_labelmap(
                analysis_roi_ref_raw,
                ref_buffer_px,
            )
            # All active reference-space spatial products below use the repaired
            # buffered analytical atlas.
            ref_broad_buffer_qc = analysis_roi_ref_buffered_repaired

            ref_repair_table = ref_analysis_repair["repair_table"].copy()
            if not ref_repair_table.empty:
                reference_hole_repair_parts.append(ref_repair_table)

            pair_qc = atlas_and_technical_pairable_map_side(
                ref_broad_buffer_qc, fixed_technical_ref, hom_display_mask, basis
            )

            side_raw = build_side_technical_raw_table(
                mask_set=mask_set,
                group_id=gid,
                mouse_id=mouse_id,
                experimental_group=experimental_group,
                canonical=canonical,
                canonical_technical=fixed_technical_can,
                broad_labelmap_ref_full=analysis_roi_ref_repaired,
                technical_mask_ref=fixed_technical_ref,
                ref_left=ref_left,
                ref_right=ref_right,
                ref_buffer_px=ref_buffer_px,
                config=local_config,
            )
            side_technical_parts.append(side_raw)

            # Per-broad reference-space pair QC counts (not denominators for v4.9 eligibility).
            hraw_parts: list[dict[str, Any]] = []
            hraw = build_homotopic_technical_raw_table(
                mask_set=mask_set,
                group_id=gid,
                mouse_id=mouse_id,
                experimental_group=experimental_group,
                pair_table=pair_table,
                pair_map=pair_map,
                canonical_technical=fixed_technical_can,
                refspace_pairable_full=pair_qc["atlas_pairable_map"],
                refspace_pairable_technical=pair_qc["technical_pairable_map"],
                config=local_config,
            )
            # Replace generic whole-map QC counts with ROI-specific refspace counts.
            for i, rr in hraw.iterrows():
                bid = int(rr["broad_structure_id"])
                roi_ref = ref_broad_buffer_qc == bid
                hraw.at[i, "refspace_atlas_pairable_px"] = int((roi_ref & pair_qc["atlas_pairable_map"]).sum())
                hraw.at[i, "refspace_technical_pairable_px"] = int((roi_ref & pair_qc["technical_pairable_map"]).sum())
            homotopic_technical_parts.append(hraw)

            reference_dir = out_paths["reference_root"] / safe_key(gid, ref_stem)
            reference_dir.mkdir(parents=True, exist_ok=True)
            np.save(reference_dir / "broad_isocortex_labels_ref.npy", broad_ref_full.astype(np.int32))
            np.save(reference_dir / "left_hemisphere_ref.npy", ref_left.astype(np.uint8))
            np.save(reference_dir / "right_hemisphere_ref.npy", ref_right.astype(np.uint8))
            np.save(reference_dir / "fixed_technical_mask_ref.npy", fixed_technical_ref.astype(np.uint8))

            mask_set_dir = out_paths["mask_sets_root"] / safe_key(mask_set, gid)
            mask_set_dir.mkdir(parents=True, exist_ok=True)
            # Keep the v4.10 authoritative filenames unchanged. Additional v4.12
            # masks are audit/QC only and make the new edge exclusion explicit.
            np.save(mask_set_dir / "technical_common_brain_ref.npy", common_brain.astype(np.uint8))
            np.save(mask_set_dir / "technical_common_brain_eroded_ref.npy", common_brain_for_technical.astype(np.uint8))
            np.save(mask_set_dir / "technical_common_brain_edge_excluded_ref.npy", common_brain_edge["excluded_edge"].astype(np.uint8))
            np.save(mask_set_dir / "fixed_technical_artifact_union_ref.npy", fixed_artifact_union.astype(np.uint8))
            np.save(mask_set_dir / "fixed_technical_mask_ref.npy", fixed_technical_ref.astype(np.uint8))

            # v4.10 authoritative reference-space analytical atlas chain.
            np.save(mask_set_dir / "analysis_roi_ref_raw.npy", analysis_roi_ref_raw.astype(np.int32))
            np.save(mask_set_dir / "analysis_roi_ref_repaired.npy", analysis_roi_ref_repaired.astype(np.int32))
            np.save(
                mask_set_dir / "analysis_roi_ref_buffered_repaired.npy",
                analysis_roi_ref_buffered_repaired.astype(np.int32),
            )
            # Backward compatibility/QC only: old direct raw->erosion product.
            np.save(mask_set_dir / "refspace_broad_buffered_qc.npy", ref_broad_buffer_legacy_qc.astype(np.int32))

            if not ref_repair_table.empty:
                ref_repair_table.to_csv(mask_set_dir / "reference_space_hole_repairs.csv", index=False)
                if bool(cfg_get(config, "step6b_fc_geometry", "qc", "save_reference_atlas_repair_png", default=True)):
                    save_reference_atlas_repair_qc_png(
                        out_paths["qc_reference_repair_root"] / safe_key(mask_set) / f"{safe_key(mouse_id)}__{safe_key(gid)}__reference_atlas_repair.png",
                        ref_img,
                        analysis_roi_ref_raw,
                        analysis_roi_ref_repaired,
                        analysis_roi_ref_buffered_repaired,
                        title=f"{mask_set} | {mouse_id} | reference analytical atlas repair",
                    )

            bilateral_fixed_ref = fixed_technical_ref & (analysis_roi_ref_buffered_repaired > 0)
            healthy_ref_mask = ref_left if healthy_side == "left" else ref_right
            global_healthy_fixed_ref = bilateral_fixed_ref & healthy_ref_mask
            np.save(mask_set_dir / "bilateral_fixed_technical_ref.npy", bilateral_fixed_ref.astype(np.uint8))
            np.save(mask_set_dir / "global_healthy_fixed_technical_ref.npy", global_healthy_fixed_ref.astype(np.uint8))
            np.save(mask_set_dir / "global_analysis_fixed_technical_ref.npy",
                    select_global_analysis_mask(bilateral_fixed_ref, ref_left, ref_right,
                                                choice=global_choice, healthy_side=healthy_side,
                                                lesion_side=lesion_side).astype(np.uint8))
            np.save(mask_set_dir / "homotopic_pairable_display_ref.npy", pair_qc["technical_pairable_map"].astype(np.uint8))

            technical_qc = out_paths["qc_technical_root"] / safe_key(mask_set) / f"{safe_key(mouse_id)}__{safe_key(gid)}__technical.png"
            save_mask_set_qc_png(
                technical_qc,
                ref_img,
                ref_broad_buffer_qc,
                common_brain,
                fixed_artifact_union,
                fixed_technical_ref,
                select_global_analysis_mask(np.ones_like(ref_left, dtype=bool), ref_left, ref_right,
                                            choice=global_choice, healthy_side=healthy_side,
                                            lesion_side=lesion_side),
                pair_qc["technical_pairable_map"],
                title=f"{mask_set} | {mouse_id} | fixed technical; rescue shadow is stem-specific",
            )
            brain_edge_qc = (
                out_paths["qc_brain_edge_root"]
                / safe_key(mask_set)
                / f"{safe_key(mouse_id)}__{safe_key(gid)}__common_brain_edge.png"
            )
            if common_brain_edge["enabled"]:
                save_common_brain_edge_qc_png(
                    brain_edge_qc,
                    ref_img,
                    common_brain,
                    common_brain_for_technical,
                    common_brain_edge["excluded_edge"],
                    fixed_technical_ref,
                    title=(
                        f"{mask_set} | {mouse_id} | common-brain edge erosion "
                        f"{common_brain_edge['applied_radius_px']} px"
                    ),
                )

            selected_stems: list[str] = []
            for slot, row in selected:
                stem = safe_str(row.get("stem"))
                selected_stems.append(stem)
                masks = masks_by_stem[stem]
                shadow_can = sample_ref_mask_on_canonical(projection, masks.rescue_shadow_ref)
                pathology_can = sample_ref_mask_on_canonical(projection, masks.pathology_ref)
                final_usable_can = fixed_technical_can & (~shadow_can) & (~pathology_can)
                spatial_stem_can = canonical_spatial_masks(
                    valid_can=final_usable_can,
                    canonical=canonical,
                    pair_map=pair_map,
                    healthy_side=healthy_side,
                    lesion_side=lesion_side,
                    homotopic_display_choice=hom_choice,
                )
                canonical_stem_dir = canonical_mask_dir / "by_stem" / safe_key(stem)
                canonical_stem_dir.mkdir(parents=True, exist_ok=True)
                np.save(canonical_stem_dir / "rescue_shadow_canonical.npy", shadow_can.astype(np.uint8))
                np.save(canonical_stem_dir / "pathology_canonical.npy", pathology_can.astype(np.uint8))
                np.save(canonical_stem_dir / "final_usable_canonical.npy", final_usable_can.astype(np.uint8))
                np.save(canonical_stem_dir / "bilateral_final_usable_canonical.npy", spatial_stem_can["bilateral"].astype(np.uint8))
                np.save(canonical_stem_dir / "global_healthy_final_usable_canonical.npy", spatial_stem_can["global_healthy"].astype(np.uint8))
                np.save(canonical_stem_dir / "global_analysis_final_usable_canonical.npy",
                        select_global_analysis_mask(spatial_stem_can["bilateral"], canonical["left_mask"],
                                                    canonical["right_mask"], choice=global_choice,
                                                    healthy_side=healthy_side, lesion_side=lesion_side).astype(np.uint8))
                np.save(canonical_stem_dir / "homotopic_pairable_bilateral_canonical.npy", spatial_stem_can["homotopic_pairable_bilateral"].astype(np.uint8))
                np.save(canonical_stem_dir / "homotopic_pairable_display_canonical.npy", spatial_stem_can["homotopic_pairable_display"].astype(np.uint8))
                side_stem_parts.append(build_side_stem_raw_table(
                    mask_set=mask_set,
                    slot=slot,
                    row=row,
                    experimental_group=experimental_group,
                    canonical=canonical,
                    fixed_technical_can=fixed_technical_can,
                    shadow_can=shadow_can,
                    pathology_can=pathology_can,
                ))
                homotopic_stem_parts.append(build_homotopic_stem_raw_table(
                    mask_set=mask_set,
                    slot=slot,
                    row=row,
                    experimental_group=experimental_group,
                    pair_table=pair_table,
                    pair_map=pair_map,
                    fixed_technical_can=fixed_technical_can,
                    shadow_can=shadow_can,
                    pathology_can=pathology_can,
                ))

                stem_technical_ref = fixed_technical_ref & (~masks.rescue_shadow_ref)
                final_usable_ref = stem_technical_ref & (~masks.pathology_ref)
                stem_dir = mask_set_dir / "by_stem" / safe_key(stem)
                stem_dir.mkdir(parents=True, exist_ok=True)
                np.save(stem_dir / "fixed_technical_mask_ref.npy", fixed_technical_ref.astype(np.uint8))
                np.save(stem_dir / "rescue_shadow_ref.npy", masks.rescue_shadow_ref.astype(np.uint8))
                np.save(stem_dir / "stem_technical_after_shadow_ref.npy", stem_technical_ref.astype(np.uint8))
                np.save(stem_dir / "pathology_ref.npy", masks.pathology_ref.astype(np.uint8))
                np.save(stem_dir / "final_usable_mask_ref.npy", final_usable_ref.astype(np.uint8))
                bilateral_final_ref = final_usable_ref & (ref_broad_buffer_qc > 0)
                global_healthy_final_ref = bilateral_final_ref & healthy_ref_mask
                final_pair_qc = atlas_and_technical_pairable_map_side(
                    ref_broad_buffer_qc, final_usable_ref, hom_display_mask, basis
                )
                np.save(stem_dir / "bilateral_final_usable_ref.npy", bilateral_final_ref.astype(np.uint8))
                np.save(stem_dir / "global_healthy_final_usable_ref.npy", global_healthy_final_ref.astype(np.uint8))
                np.save(stem_dir / "global_analysis_final_usable_ref.npy",
                        select_global_analysis_mask(bilateral_final_ref, ref_left, ref_right,
                                                    choice=global_choice, healthy_side=healthy_side,
                                                    lesion_side=lesion_side).astype(np.uint8))
                np.save(stem_dir / "homotopic_pairable_display_ref.npy", final_pair_qc["technical_pairable_map"].astype(np.uint8))

                if bool(masks.rescue_shadow_ref.any()):
                    shadow_png = out_paths["qc_shadow_root"] / safe_key(mask_set) / f"{safe_key(mouse_id)}__{safe_key(row.get('exp_day'))}__{safe_key(stem)}__shadow.png"
                    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                    axes[0].imshow(ref_img, cmap="gray"); axes[0].set_title("reference")
                    _imshow_mask(axes[1], masks.rescue_shadow_ref, "rescue shadow")
                    _imshow_mask(axes[2], fixed_technical_ref, "fixed technical")
                    _imshow_mask(axes[3], stem_technical_ref, "after shadow")
                    for ax in axes: ax.axis("off")
                    fig.suptitle(f"{mask_set} | {mouse_id} | {stem}")
                    fig.tight_layout(); shadow_png.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(shadow_png, dpi=160, bbox_inches="tight"); plt.close(fig)

                if bool(masks.pathology_ref.any()):
                    pathology_png = out_paths["qc_pathology_root"] / safe_key(mask_set) / f"{safe_key(mouse_id)}__{safe_key(row.get('exp_day'))}__{safe_key(stem)}__pathology.png"
                    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                    axes[0].imshow(ref_img, cmap="gray"); axes[0].set_title("reference")
                    _imshow_mask(axes[1], masks.pathology_ref, "pathology")
                    _imshow_mask(axes[2], stem_technical_ref, "after shadow")
                    _imshow_mask(axes[3], final_usable_ref, "final usable")
                    for ax in axes: ax.axis("off")
                    fig.suptitle(f"{mask_set} | {mouse_id} | {stem}")
                    fig.tight_layout(); pathology_png.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(pathology_png, dpi=160, bbox_inches="tight"); plt.close(fig)

            mask_set_summary_rows.append({
                "mask_set": mask_set,
                "group_id": gid,
                "mouse_id": mouse_id,
                "experimental_group": experimental_group,
                "ref_stem": ref_stem,
                "lesion_side": lesion_side,
                "healthy_side": healthy_side,
                "n_slots": len(selected),
                "selected_stems_json": json.dumps(selected_stems, ensure_ascii=False),
                "spatial_masks_json": json.dumps(
                    list((cfg_get(config, "step6b_fc_geometry", "mask_sets", mask_set, "spatial_masks", default=[]) or [])),
                    ensure_ascii=False,
                ),
                "effective_thresholds_json": json.dumps(
                    thresholds_for_mask_set(config, mask_set), ensure_ascii=False
                ),
                "fixed_common_brain_ref_px": int(common_brain.sum()),
                "common_brain_edge_erosion_enabled": bool(common_brain_edge["enabled"]),
                "common_brain_edge_erosion_requested_px": float(common_brain_edge["requested_radius_px"]),
                "common_brain_edge_erosion_applied_px": int(common_brain_edge["applied_radius_px"]),
                "fixed_common_brain_after_edge_erosion_ref_px": int(common_brain_edge["after_px"]),
                "fixed_common_brain_edge_excluded_ref_px": int(common_brain_edge["excluded_px"]),
                "fixed_common_brain_edge_retained_fraction": common_brain_edge["retained_fraction"],
                "fixed_technical_formula": "eroded_common_brain AND NOT fixed_artifact_union" if int(common_brain_edge["applied_radius_px"]) > 0 else "common_brain AND NOT fixed_artifact_union",
                "fixed_technical_artifact_union_ref_px": int(fixed_artifact_union.sum()),
                "fixed_technical_mask_ref_px": int(fixed_technical_ref.sum()),
                "canonical_fixed_technical_px": int(fixed_technical_can.sum()),
                "boundary_exclusion_canonical_px": float(canonical["boundary_px"]),
                "refspace_equivalent_boundary_buffer_px": float(ref_buffer_px),
                "reference_analysis_roi_atlas_source": "analysis_roi_ref_buffered_repaired.npy",
                "reference_analysis_roi_repaired_px": int(ref_analysis_repair["n_repaired_px"]),
                "technical_qc_png": str(technical_qc),
                "brain_edge_qc_png": str(brain_edge_qc) if common_brain_edge["enabled"] else "",
            })
        except Exception as exc:
            error_rows.append({
                "mask_set": mask_set,
                "group_id": gid,
                "mouse_id": mouse_id,
                "error_type": type(exc).__name__,
                "error": repr(exc),
            })
            if not continue_on_error:
                raise

    # Canonical cohort representation maps: sum one fixed technical mask per mouse / N.
    cohort_frequency_rows: list[dict[str, Any]] = []
    for mset, entries in cohort_fixed_masks.items():
        if not entries:
            continue
        stack = np.stack([np.asarray(mask, dtype=np.uint8) for _, _, mask in entries], axis=0)
        count_map = stack.sum(axis=0).astype(np.uint16)
        n_mice = int(stack.shape[0])
        frequency_map = count_map.astype(np.float32) / float(n_mice)
        cohort_dir = out_paths["cohort_canonical_root"] / safe_key(mset)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        np.save(cohort_dir / "fixed_technical_count_canonical.npy", count_map)
        np.save(cohort_dir / "fixed_technical_frequency_canonical.npy", frequency_map)
        pd.DataFrame([
            {"mouse_id": mid, "group_id": gid, "mask_set": mset}
            for mid, gid, _ in entries
        ]).to_csv(cohort_dir / "included_mice.csv", index=False)
        if bool(cfg_get(config, "step6b_fc_geometry", "qc", "save_cohort_frequency_maps", default=True)):
            save_canonical_frequency_png(
                cohort_dir / "fixed_technical_frequency_canonical.png",
                frequency_map,
                canonical,
                title=f"{mset} | fixed technical representation | N={n_mice}",
            )
        for _, rr in canonical["roi_table"].iterrows():
            roi_key = safe_str(rr["roi_key"])
            roi = np.asarray(canonical["roi_masks_buffered"][roi_key], dtype=bool)
            vals = frequency_map[roi]
            vals = vals[np.isfinite(vals)]
            cohort_frequency_rows.append({
                "mask_set": mset,
                "broad_structure_id": int(rr["broad_structure_id"]),
                "broad_acronym": safe_str(rr["broad_acronym"]),
                "broad_name": safe_str(rr["broad_name"]),
                "hemisphere": safe_str(rr["hemisphere"]),
                "roi_key": roi_key,
                "n_mice_frequency_map": n_mice,
                "canonical_buffered_area_px": int(rr["canonical_buffered_area_px"]),
                "mean_pixel_mouse_fraction": float(np.mean(vals)) if vals.size else np.nan,
                "median_pixel_mouse_fraction": float(np.median(vals)) if vals.size else np.nan,
                "min_pixel_mouse_fraction": float(np.min(vals)) if vals.size else np.nan,
                "max_pixel_mouse_fraction": float(np.max(vals)) if vals.size else np.nan,
                "fraction_roi_pixels_present_in_all_mice": float(np.mean(vals >= 1.0)) if vals.size else np.nan,
                "fraction_roi_pixels_present_in_majority": float(np.mean(vals >= 0.5)) if vals.size else np.nan,
            })
    cohort_frequency_df = pd.DataFrame(cohort_frequency_rows)
    cohort_frequency_df.to_csv(out_paths["cohort_frequency_summary_csv"], index=False)
    side_consistency_df = pd.DataFrame(side_consistency_rows)
    side_consistency_df.to_csv(out_paths["side_consistency_csv"], index=False)

    # Cohort-relative gates after all mice are collected.
    side_raw_df = pd.concat(side_technical_parts, ignore_index=True) if side_technical_parts else pd.DataFrame()
    side_df, side_ref_df = add_relative_gate_by_mask_set(
        side_raw_df,
        value_col="technical_coverage_fraction",
        reference_col="cohort_reference_technical_coverage",
        relative_col="technical_relative_to_cohort",
        key_cols=["mask_set", "broad_structure_id", "broad_acronym", "broad_name", "hemisphere", "roi_key"],
        pixel_pass_col="technical_pixel_pass",
        absolute_pass_col="technical_absolute_pass",
        final_pass_col="technical_pass",
        config=config,
    )
    side_df = add_relative_pass_column_by_mask_set(
        side_df,
        relative_col="technical_relative_to_cohort",
        pass_col="technical_relative_pass",
        config=config,
    )
    side_df.to_csv(out_paths["side_technical_csv"], index=False)

    hraw_df = pd.concat(homotopic_technical_parts, ignore_index=True) if homotopic_technical_parts else pd.DataFrame()
    htech_df, h_ref_df = add_relative_gate_by_mask_set(
        hraw_df,
        value_col="homotopic_technical_coverage_fraction",
        reference_col="cohort_reference_homotopic_coverage",
        relative_col="homotopic_relative_to_cohort",
        key_cols=["mask_set", "broad_structure_id", "broad_acronym", "broad_name", "pair_key"],
        pixel_pass_col="homotopic_pixel_pass",
        absolute_pass_col="homotopic_absolute_pass",
        final_pass_col="homotopic_technical_pass",
        config=config,
    )
    htech_df = add_relative_pass_column_by_mask_set(
        htech_df,
        relative_col="homotopic_relative_to_cohort",
        pass_col="homotopic_relative_pass",
        config=config,
    )
    htech_df.to_csv(out_paths["homotopic_technical_csv"], index=False)

    hemisphere_df = (pd.DataFrame(hemisphere_rows).drop_duplicates(
        subset=["mask_set", "group_id"], keep="last") if hemisphere_rows else pd.DataFrame())
    hemisphere_df.to_csv(out_paths["hemisphere_csv"], index=False)
    global_df = build_global_technical_table(side_df, hemisphere_df, config)
    global_df.to_csv(out_paths["global_technical_csv"], index=False)

    cohort_refs = []
    if not side_ref_df.empty:
        s = side_ref_df.copy(); s["coverage_kind"] = "side"; cohort_refs.append(s)
    if not h_ref_df.empty:
        h = h_ref_df.copy(); h["coverage_kind"] = "homotopic_pair"; cohort_refs.append(h)
    cohort_ref_df = pd.concat(cohort_refs, ignore_index=True, sort=False) if cohort_refs else pd.DataFrame()
    cohort_ref_df.to_csv(out_paths["cohort_reference_roi_csv"], index=False)

    side_stem_raw = pd.concat(side_stem_parts, ignore_index=True) if side_stem_parts else pd.DataFrame()
    side_stem_df = finalize_side_stem_passes(side_stem_raw, side_df, config)
    side_stem_df.to_csv(out_paths["side_stem_usable_csv"], index=False)
    hstem_raw = pd.concat(homotopic_stem_parts, ignore_index=True) if homotopic_stem_parts else pd.DataFrame()
    hstem_df = finalize_homotopic_stem_passes(hstem_raw, htech_df, config)
    hstem_df.to_csv(out_paths["homotopic_stem_usable_csv"], index=False)
    gstem_df = build_global_stem_table(side_stem_df, global_df, config)
    gstem_df.to_csv(out_paths["global_stem_usable_csv"], index=False)

    # Save per mouse/mask-set and stem local coverage CSVs for browsing.
    if not side_df.empty:
        for (mset, gid), g in side_df.groupby(["mask_set", "group_id"], sort=False):
            d = out_paths["mask_sets_root"] / safe_key(mset, gid); d.mkdir(parents=True, exist_ok=True)
            g.to_csv(d / "roi_side_technical_coverage.csv", index=False)
    if not htech_df.empty:
        for (mset, gid), g in htech_df.groupby(["mask_set", "group_id"], sort=False):
            d = out_paths["mask_sets_root"] / safe_key(mset, gid); d.mkdir(parents=True, exist_ok=True)
            g.to_csv(d / "roi_homotopic_pair_coverage.csv", index=False)
    if not side_stem_df.empty:
        for (mset, gid, stem), g in side_stem_df.groupby(["mask_set", "group_id", "stem"], sort=False):
            d = out_paths["mask_sets_root"] / safe_key(mset, gid) / "by_stem" / safe_key(stem); d.mkdir(parents=True, exist_ok=True)
            g.to_csv(d / "roi_side_usable_coverage.csv", index=False)
    if not hstem_df.empty:
        for (mset, gid, stem), g in hstem_df.groupby(["mask_set", "group_id", "stem"], sort=False):
            d = out_paths["mask_sets_root"] / safe_key(mset, gid) / "by_stem" / safe_key(stem); d.mkdir(parents=True, exist_ok=True)
            g.to_csv(d / "roi_homotopic_usable_coverage.csv", index=False)

    eligibility_df = build_v45_eligibility(htech_df, hstem_df, global_df, gstem_df)
    eligibility_df.to_csv(out_paths["eligibility_csv"], index=False)
    # Only group PASS denominators use the actual comparison group (Day-PT state).
    # Experimental_group remains the true experimental-group provenance everywhere.
    comparison_lookup = selection[selection["status"].eq("ok")][["mask_set", "group_id", "comparison_group"]].drop_duplicates()
    if not comparison_lookup.empty:
        if comparison_lookup.duplicated(["mask_set", "group_id"]).any():
            raise ValueError("One mouse has multiple Day-PT states: separate analysis cases are required; never intersect states")
        eligibility_df = eligibility_df.merge(comparison_lookup, on=["mask_set", "group_id"], how="left", validate="many_to_one")
    else:
        eligibility_df["comparison_group"] = ""
    eligibility_df.to_csv(out_paths["eligibility_csv"], index=False)
    eligibility_for_grouping = eligibility_df.copy()
    eligibility_for_grouping["experimental_group"] = eligibility_for_grouping["comparison_group"].astype(str)
    group_summary_df, selected_df, selected_primary_df, funnel_df = build_v45_group_tables(eligibility_for_grouping, config)
    group_summary_df.to_csv(out_paths["group_summary_csv"], index=False)
    selected_df.to_csv(out_paths["selected_by_set_csv"], index=False)
    final_roi_pass_summary = build_final_roi_pass_summary(selected_df, group_summary_df)
    final_roi_pass_summary.to_csv(out_paths["final_roi_pass_summary_csv"], index=False)
    selected_primary_df.to_csv(out_paths["selected_longitudinal_csv"], index=False)
    funnel_df.to_csv(out_paths["completeness_funnel_csv"], index=False)

    pathology_parts = []
    if not side_stem_df.empty:
        p = side_stem_df[pd.to_numeric(side_stem_df["canonical_pathology_lost_px"], errors="coerce").fillna(0) > 0].copy(); p["analysis_kind"] = "side"; pathology_parts.append(p)
    if not hstem_df.empty:
        p = hstem_df[pd.to_numeric(hstem_df["canonical_pathology_lost_pair_px"], errors="coerce").fillna(0) > 0].copy(); p["analysis_kind"] = "homotopic_pair"; pathology_parts.append(p)
    pathology_nonzero_df = pd.concat(pathology_parts, ignore_index=True, sort=False) if pathology_parts else pd.DataFrame()
    pathology_nonzero_df.to_csv(out_paths["pathology_nonzero_csv"], index=False)

    shadow_parts = []
    if not side_stem_df.empty:
        s = side_stem_df[pd.to_numeric(side_stem_df["canonical_shadow_lost_px"], errors="coerce").fillna(0) > 0].copy(); s["analysis_kind"] = "side"; shadow_parts.append(s)
    if not hstem_df.empty:
        s = hstem_df[pd.to_numeric(hstem_df["canonical_shadow_lost_pair_px"], errors="coerce").fillna(0) > 0].copy(); s["analysis_kind"] = "homotopic_pair"; shadow_parts.append(s)
    shadow_nonzero_df = pd.concat(shadow_parts, ignore_index=True, sort=False) if shadow_parts else pd.DataFrame()
    shadow_nonzero_df.to_csv(out_paths["shadow_nonzero_csv"], index=False)

    reference_hole_repair_df = (
        pd.concat(reference_hole_repair_parts, ignore_index=True, sort=False)
        if reference_hole_repair_parts else pd.DataFrame()
    )
    reference_hole_repair_df.to_csv(out_paths["reference_hole_repair_csv"], index=False)

    stem_log_df = pd.DataFrame(list(stem_log_records.values()))
    stem_log_df.to_csv(out_paths["stem_mask_log_csv"], index=False)
    mask_set_summary_df = pd.DataFrame(mask_set_summary_rows)
    mask_set_summary_df.to_csv(out_paths["mask_set_summary_csv"], index=False)
    cohort_overview = build_cohort_overview(selection, config, mask_set_summary_df)
    cohort_overview.to_csv(out_paths["cohort_overview_csv"], index=False)
    errors_df = pd.DataFrame(error_rows, columns=["mask_set", "group_id", "mouse_id", "error_type", "error"])
    errors_df.to_csv(out_paths["errors_csv"], index=False)

    return {
        "selection": selection,
        "assignment_audit": assignment_audit,
        "analysis_exclusions": exclusion_audit,
        "exclusion_rules": exclusion_rules,
        "skipped_mask_sets": skipped_mask_sets,
        "mask_set_cohort_summary": mask_set_cohort_summary,
        "cohort_overview": cohort_overview,
        "mapping": mapping_df,
        "direct_children": children_df,
        "analysis_roi_definitions": children_df,
        "roi_refinements": refinement_df,
        "canonical_rois": canonical["roi_table"],
        "canonical_hole_repairs": canonical.get("hole_repair_table", pd.DataFrame()),
        "reference_hole_repairs": reference_hole_repair_df,
        "canonical_homotopic_pairs": pair_table,
        "hemisphere": hemisphere_df,
        "mask_set_summary": mask_set_summary_df,
        "left_right_consistency": side_consistency_df,
        "cohort_frequency_summary": cohort_frequency_df,
        "side_technical_coverage": side_df,
        "technical_coverage": side_df,
        "homotopic_technical_coverage": htech_df,
        "global_technical_coverage": global_df,
        "cohort_reference_coverage": cohort_ref_df,
        "side_stem_usable_coverage": side_stem_df,
        "stem_usable_coverage": side_stem_df,
        "homotopic_stem_usable_coverage": hstem_df,
        "global_stem_usable_coverage": gstem_df,
        "pathology_nonzero": pathology_nonzero_df,
        "shadow_nonzero": shadow_nonzero_df,
        "eligibility": eligibility_df,
        "completeness_funnel": funnel_df,
        "group_summary": group_summary_df,
        "selected_rois": selected_df,
        "final_roi_pass_summary": final_roi_pass_summary,
        "selected_rois_longitudinal": selected_primary_df,
        "errors": errors_df,
    }


def load_step6b_tables(config_path: str | Path, *, rest_variant: str = "without_rest") -> dict[str, pd.DataFrame]:
    """Load previously generated Step6b CSV tables without rebuilding ROI geometry.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Project YAML used to locate the analysis output directory.
    rest_variant : {"without_rest", "with_rest"}, optional
        Saved branch to load.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Named tables; missing or empty output files are represented by empty
        DataFrames.

    Notes
    -----
    This is the read-existing-results path for notebook quality control."""
    _, config, paths = build_step6_index(config_path, include_bad=True, require_response_roi=False)
    ensure_step6b_defaults(config)
    config["step6b_fc_geometry"]["active_rest_variant"] = rest_variant
    out = build_step6b_paths(config, paths)
    mapping = {
        "mapping": "mapping_csv",
        "direct_children": "direct_children_csv",
        "analysis_roi_definitions": "direct_children_csv",
        "roi_refinements": "roi_refinement_csv",
        "canonical_rois": "canonical_roi_csv",
        "canonical_hole_repairs": "canonical_hole_repair_csv",
        "reference_hole_repairs": "reference_hole_repair_csv",
        "canonical_homotopic_pairs": "canonical_pair_csv",
        "selection": "selection_csv",
        "analysis_exclusions": "excluded_stems_csv",
        "exclusion_rules": "exclusion_rules_csv",
        "rest_availability": "rest_availability_csv",
        "skipped_mask_sets": "skipped_mask_sets_csv",
        "mask_set_cohort_summary": "mask_set_cohort_summary_csv",
        "cohort_overview": "cohort_overview_csv",
        "final_roi_pass_summary": "final_roi_pass_summary_csv",
        "assignment_audit": "assignment_audit_csv",
        "hemisphere": "hemisphere_csv",
        "mask_set_summary": "mask_set_summary_csv",
        "left_right_consistency": "side_consistency_csv",
        "cohort_frequency_summary": "cohort_frequency_summary_csv",
        "side_technical_coverage": "side_technical_csv",
        "technical_coverage": "side_technical_csv",
        "homotopic_technical_coverage": "homotopic_technical_csv",
        "global_technical_coverage": "global_technical_csv",
        "cohort_reference_coverage": "cohort_reference_roi_csv",
        "side_stem_usable_coverage": "side_stem_usable_csv",
        "stem_usable_coverage": "side_stem_usable_csv",
        "homotopic_stem_usable_coverage": "homotopic_stem_usable_csv",
        "global_stem_usable_coverage": "global_stem_usable_csv",
        "pathology_nonzero": "pathology_nonzero_csv",
        "shadow_nonzero": "shadow_nonzero_csv",
        "eligibility": "eligibility_csv",
        "completeness_funnel": "completeness_funnel_csv",
        "group_summary": "group_summary_csv",
        "selected_rois": "selected_by_set_csv",
        "selected_rois_longitudinal": "selected_longitudinal_csv",
        "errors": "errors_csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for name, key in mapping.items():
        p = out[key]
        if p.exists() and p.stat().st_size > 0:
            try:
                result[name] = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                result[name] = pd.DataFrame()
        else:
            result[name] = pd.DataFrame()
    return result


# =============================================================================
# CLI
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the documented command-line argument parser for Step6b."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path, help="Project YAML config")
    p.add_argument("--mask-set", action="append", default=None, help="Process only this mask set; repeatable")
    p.add_argument("--group", action="append", default=None, help="Process only this group_id; repeatable")
    p.add_argument("--rest-variant", choices=["without_rest", "with_rest", "both"], default="without_rest")
    p.add_argument("--dry-run", action="store_true", help="Resolve mask-set slots only")
    p.add_argument("--overwrite", action="store_true", help="Overwrite saved QC/reference outputs")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Parse CLI arguments and invoke the Step6b workflow."""
    args = build_arg_parser().parse_args(argv)
    result = run_step6b(
        args.config,
        mask_sets=args.mask_set,
        groups=args.group,
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
        rest_variant=args.rest_variant,
    )
    if args.rest_variant == "both":
        for variant, sub in result.items():
            print(f"{variant}: selection={len(sub['selection'])}, skipped={len(sub.get('skipped_mask_sets', []))}")
        return
    if args.dry_run:
        problems = result["selection"][~result["selection"]["status"].eq("ok")]
        print(problems.to_string(index=False) if not problems.empty else "No selection problems")
        print(f"Incomplete mouse × mask-set combinations: {len(result.get('skipped_mask_sets', pd.DataFrame()))}")
    else:
        print(f"Saved Step6b v4.12 outputs. Mask sets: {len(result['mask_set_summary'])}")
        print(f"Side ROI technical rows: {len(result['side_technical_coverage'])}")
        print(f"Homotopic pair rows: {len(result['homotopic_technical_coverage'])}")
        print(f"Global ROI rows: {len(result['global_technical_coverage'])}")
        print(f"Selected primary ROI rows: {len(result['selected_rois_longitudinal'])}")


if __name__ == "__main__":
    main()
