"""Lesion-assessment helper module for the WIFIOPIA stem-key pipeline.

The original automatic assessment workflow is preserved for compatibility.
The manual GUI imports this module only for indexing, geometry and masks.
The original workflow keeps the following behavior:

* one output mask set per post-stroke ``stem``;
* masks stay in that stem's native Step2/Step3 image space
  (rotated full resolution, before Step4 ``raw.scale``);
* automatic lesion candidate comes from the difference between a baseline image
  and the current post-stroke image;
* only the current stem brain mask limits automatic detection;
* Step2 bad/glare masks and cross-day common masks are not used;
* hemorrhage is a separate, manually drawn Napari labels layer;
* outputs are not consumed by Step4.

The automatic detector originates from an earlier difference-image review
workflow; indexing follows the current ``group_id / stem / ref_stem`` pipeline.

Typical use
-----------

    python damage_assessment.py --config config.yaml

Dry-run/index check without opening Napari:

    python damage_assessment.py --config config.yaml --dry-run

Review selected stems:

    python damage_assessment.py --config config.yaml \
        --stems STEM_A STEM_B
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
import traceback
from pathlib import Path
from typing import Any, Iterable, Sequence

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
from skimage.transform import EuclideanTransform, resize, warp

try:  # Napari is optional until interactive review is requested.
    import napari
except Exception:  # pragma: no cover - depends on the user's GUI environment.
    napari = None

from common import (
    build_standard_paths,
    canonical_mask_paths,
    create_or_load_project,
    get_path_resolving_config,
    load_config,
    resolve_flexible_path,
)


# ============================================================================
# Defaults / config
# ============================================================================


def _deep_update_missing(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Fill missing nested configuration keys without overwriting user values."""
    for key, value in src.items():
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            _deep_update_missing(dst[key], value)
    return dst


def default_damage_config() -> dict[str, Any]:
    """Return the fallback settings for automatic lesion assessment."""
    return {
        "folder_name": "damage_assessment",
        "annotator": "user",
        "input": {
            # The original default policy: use the Step2 ref_stem as baseline.
            # Set policy=match only if baseline is not the reference stem.
            "baseline": {
                "policy": "ref_stem",  # ref_stem / match
                "match": {},
            },
            # Example fallback labels; set actual days in damage_assessment.input.poststroke.match in YAML.
            "poststroke": {
                "match": {
                    "exp_day": ["PT01", "PT03"],
                },
            },
            # Current native image is read from moving_registration_path; reference
            # rows use fixed_registration_path. All are rotated_fullres images.
            "use_manual_registration_overrides": True,
            "resize_brain_mask_if_needed": False,
        },
        "detection": {
            "normalization": "mean",  # none / mean / zscore
            "difference_mode": "positive",  # positive / negative / absolute
            "gaussian_sigma": 1.0,
            "threshold": 0.15,
            "close_radius": 2,
            "open_radius": 1,
            "min_area_px": 50,
            # nearest_expected is adapted from the old pipeline. Set largest for
            # a detector that does not depend on lesion coordinates.
            "component_selection": "nearest_expected",  # nearest_expected / largest / all
            "expected_center": {
                "enabled": True,
                # Anatomical offsets relative to final reference bregma and
                # the Step3 annotated axis: dx<0 = right, dx>0 = left;
                # dy<0 = down along the axis, dy>0 = up along the axis.
                "dx_mm": -2.0,
                "dy_mm": 1.0,
                "pixel_size_mm": 0.053,
                "axis_angle_source": "step3_annotation",  # step3_annotation / fixed
                "fallback_axis_angle_deg": -10.0,
                "max_component_distance_px": None,
                "lesion_radius_mm": 0.8,
            },
        },
        "review": {
            "enabled": True,
            "preload_existing_final_masks": True,
            "skip_reviewed": False,
            # --no-viewer normally saves only auto/intermediate files. Enabling this
            # copies lesion_auto to lesion_final in non-interactive runs.
            "save_auto_as_final_without_viewer": False,
            "lesion_opacity": 0.55,
            "hemorrhage_opacity": 0.55,
        },
        "batch": {
            "filters": None,
            "include_bad": False,
            "continue_on_error": True,
        },
        "output": {
            "masks_folder": "masks",
            "diff_maps_folder": "diff_maps",
            "previews_folder": "previews",
            "tables_folder": "tables",
            "index_name": "damage_index.csv",
            "review_log_name": "damage_review.csv",
            "error_log_name": "damage_errors.csv",
            "save_diff_maps": True,
            "save_preview_png": True,
        },
    }


def ensure_damage_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Populate absent damage_assessment settings while retaining the YAML values."""
    config.setdefault("damage_assessment", {})
    _deep_update_missing(config["damage_assessment"], default_damage_config())
    return config


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested configuration value or return the requested default."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ============================================================================
# Paths / small IO
# ============================================================================


def build_damage_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create and return damage assessment output directories and table paths."""
    cfg = ensure_damage_defaults(config)["damage_assessment"]
    out_cfg = cfg.get("output", {}) or {}
    root = Path(paths["out_root"]) / str(cfg.get("folder_name", "damage_assessment"))

    out = {
        "damage_root": root,
        "damage_masks_root": root / out_cfg.get("masks_folder", "masks"),
        "damage_diff_root": root / out_cfg.get("diff_maps_folder", "diff_maps"),
        "damage_previews_root": root / out_cfg.get("previews_folder", "previews"),
        "damage_tables_root": root / out_cfg.get("tables_folder", "tables"),
    }
    out["damage_index_csv"] = out["damage_tables_root"] / out_cfg.get("index_name", "damage_index.csv")
    out["damage_review_csv"] = out["damage_tables_root"] / out_cfg.get("review_log_name", "damage_review.csv")
    out["damage_errors_csv"] = out["damage_tables_root"] / out_cfg.get("error_log_name", "damage_errors.csv")

    for path in out.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    return out


def damage_mask_paths(paths: dict[str, Path], stem: str) -> dict[str, Path]:
    """Return canonical automatic and final mask paths for one recording stem."""
    root = Path(paths["damage_masks_root"]) / str(stem)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "lesion_auto": root / "lesion_auto.npy",
        "lesion_final": root / "lesion_final.npy",
        "hemorrhage_final": root / "hemorrhage_final.npy",
        "brain_used": root / "brain_used.npy",
        "review_meta": root / "review_meta.json",
    }


def damage_diff_paths(paths: dict[str, Path], stem: str) -> dict[str, Path]:
    """Return difference-map paths for one recording stem."""
    root = Path(paths["damage_diff_root"]) / str(stem)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "baseline_warped": root / "baseline_warped_to_current.npy",
        "diff_signed": root / "diff_signed.npy",
        "diff_selected": root / "diff_selected.npy",
        "diff_smooth": root / "diff_smooth.npy",
        "candidate_raw": root / "lesion_candidate_raw.npy",
        "candidate_clean": root / "lesion_candidate_clean.npy",
    }


def _resolve_optional_path(value: Any, path_cfg: dict[str, Any], *, must_exist: bool = False) -> Path | None:
    """Resolve an optional path using the project drive/root remapping rules."""
    if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
        return None
    try:
        return resolve_flexible_path(
            value,
            from_drive=path_cfg.get("from_drive"),
            to_drive=path_cfg.get("to_drive"),
            from_root=path_cfg.get("from_root"),
            to_root=path_cfg.get("to_root"),
            must_exist=must_exist,
        )
    except FileNotFoundError:
        p = Path(str(value))
        if must_exist:
            raise
        return p


def _first_existing_path(row: pd.Series, columns: Sequence[str], path_cfg: dict[str, Any], *, what: str) -> Path:
    """Find the first existing path among the specified metadata columns."""
    tried: list[str] = []
    for col in columns:
        if col not in row.index:
            continue
        value = row.get(col)
        if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
            continue
        tried.append(f"{col}={value}")
        path = _resolve_optional_path(value, path_cfg, must_exist=False)
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(f"{what} not found. Tried: " + "; ".join(tried or columns))


def _read_gray(path: Path) -> np.ndarray:
    """Read a grayscale or multichannel image as a two-dimensional float32 array."""
    arr = np.asarray(iio.imread(path))
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return arr[..., :3].mean(axis=-1).astype(np.float32)
    if arr.ndim == 3:
        return np.asarray(arr[0], dtype=np.float32)
    raise ValueError(f"Unsupported image shape {arr.shape} for {path}")


def _save_json(path: Path, obj: dict[str, Any]) -> None:
    """Write a metadata dictionary to an indented UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _append_or_upsert_csv(path: Path, row: dict[str, Any], keys: Sequence[str]) -> None:
    """Replace matching CSV records and append the supplied record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame()

    for col in row:
        if col not in df.columns:
            df[col] = ""
    for key in keys:
        if key not in df.columns:
            df[key] = ""

    if not df.empty:
        mask = pd.Series(True, index=df.index)
        for key in keys:
            mask &= df[key].astype(str).eq(str(row.get(key, "")))
    else:
        mask = pd.Series(dtype=bool)

    if len(mask) and mask.any():
        idx = df.index[mask][-1]
        for col, value in row.items():
            df.loc[idx, col] = value
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False)

    ordered = list(row.keys()) + [c for c in df.columns if c not in row]
    df[ordered].to_csv(path, index=False)


def _now_iso() -> str:
    """Return the local time as a seconds-resolution ISO timestamp."""
    return _dt.datetime.now().isoformat(timespec="seconds")


# ============================================================================
# Filtering / index
# ============================================================================


def _as_list(value: Any) -> list[str]:
    """Convert scalar or iterable filter values into a list of strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def _bool_series(series: pd.Series) -> pd.Series:
    """Convert common textual truth values into a boolean pandas Series."""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _match_mask(df: pd.DataFrame, match: dict[str, Any] | None) -> pd.Series:
    """Build a per-row selection mask from exact metadata matching rules."""
    mask = pd.Series(True, index=df.index)
    for key, rule in (match or {}).items():
        if key not in df.columns:
            raise ValueError(f"Damage filter column not found: {key}")
        values = df[key].astype(str)

        if isinstance(rule, dict):
            local = pd.Series(True, index=df.index)
            include = set(_as_list(rule.get("include")))
            exclude = set(_as_list(rule.get("exclude")))
            regex = rule.get("regex")
            contains = rule.get("contains")
            if include:
                local &= values.isin(include)
            if exclude:
                local &= ~values.isin(exclude)
            if regex not in (None, ""):
                local &= values.str.contains(str(regex), regex=True, na=False)
            if contains not in (None, ""):
                local &= values.str.contains(str(contains), regex=False, na=False)
            mask &= local
        else:
            allowed = set(_as_list(rule))
            if allowed:
                mask &= values.isin(allowed)
    return mask


def _parse_cli_filters(items: Sequence[str] | None) -> dict[str, Any] | None:
    """Parse repeated CLI KEY=VALUE filters into matching rules."""
    if not items:
        return None
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Filter must be KEY=VALUE or KEY=a,b, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        vals = [v.strip() for v in value.split(",") if v.strip()]
        if not key:
            raise ValueError(f"Empty filter key in {item!r}")
        out[key] = vals if len(vals) > 1 else (vals[0] if vals else "")
    return out


def _merge_filters(*filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Combine optional filters, with later dictionaries overriding earlier ones."""
    out: dict[str, Any] = {}
    for item in filters:
        if item:
            out.update(item)
    return out or None


def _select_baseline_row(group: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Select a unique baseline row under the configured baseline policy."""
    baseline_cfg = cfg_get(config, "damage_assessment", "input", "baseline", default={}) or {}
    policy = str(baseline_cfg.get("policy", "ref_stem")).strip().lower()

    if policy == "ref_stem":
        ref_values = [str(v) for v in group.get("ref_stem", pd.Series(dtype=str)).dropna().astype(str).unique() if str(v).strip()]
        if not ref_values:
            raise RuntimeError(f"No ref_stem in group_id={group.iloc[0].get('group_id', '')}")
        rows = group[group["stem"].astype(str).eq(ref_values[0])]
        if rows.empty:
            raise RuntimeError(f"ref_stem={ref_values[0]} row not found in group_id={group.iloc[0].get('group_id', '')}")
        return rows.iloc[0]

    if policy == "match":
        rows = group[_match_mask(group, baseline_cfg.get("match") or {})]
        if rows.empty:
            raise RuntimeError(
                f"No baseline row matched for group_id={group.iloc[0].get('group_id', '')}; "
                f"match={baseline_cfg.get('match')}"
            )
        if len(rows) > 1:
            if bool(cfg_get(config, "damage_assessment", "manual_gui", "selection", "require_unique_baseline", default=False)):
                raise RuntimeError(
                    "Ambiguous Day-PT baseline for group_id="
                    f"{group.iloc[0].get('group_id', '')}: "
                    f"{rows['stem'].astype(str).tolist()}"
                )
            rows = rows.sort_values([c for c in ["exp_day", "state", "protocol", "stem"] if c in rows.columns])
        return rows.iloc[0]

    raise ValueError("damage_assessment.input.baseline.policy must be 'ref_stem' or 'match'")


def build_damage_index(
    config_path: str | Path,
    *,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    include_bad: bool | None = None,
    for_manual_gui: bool = False,
    write_index: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Path]]:
    """Build the lesion-assessment index from Step2 registration metadata.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the project YAML configuration.
    filters : dict or None, optional
        Additional metadata filters for post-stroke targets.
    groups, stems : sequence of str or None, optional
        Restrict review to selected group or recording identifiers.
    include_bad : bool or None, optional
        Whether to include Step2 rows flagged as bad.
    for_manual_gui : bool, optional
        Use the manual GUI selection branch, which **overrides** the baseline
        and post-stroke day/protocol filters with hard-coded values; this branch
        is not controlled solely by ``damage_assessment.input`` in YAML.
    write_index : bool, optional
        Write the index CSV when nonempty.

    Returns
    -------
    tuple of (pandas.DataFrame, dict, dict)
        Per-stem selection/index table, effective configuration, and project
        paths. Entries with baseline-selection errors can be retained as error
        records rather than silently assigned to a different baseline.

    Notes
    -----
    The manual GUI branch selects ``Day-PT`` / ``Rest-state`` for the baseline
    and ``1-5h-PT``, ``1d-PT``, ``3d-PT`` or ``PT`` / ``Rest-state`` for targets.
    It may retain a target without a baseline; the automatic workflow does not.
    These are implementation-specific selection rules, not universal study days.
    """
    config_path = Path(config_path)
    config = ensure_damage_defaults(load_config(config_path))
    if for_manual_gui:
        # These selection changes are GUI-local: the YAML and analysis workflows
        # retain their original selection criteria and experimental cohorts.
        config = copy.deepcopy(config)
        dcfg = config["damage_assessment"]
        baseline = dcfg.setdefault("input", {}).setdefault("baseline", {})
        baseline["policy"] = "match"
        baseline_match = baseline.setdefault("match", {})
        baseline_match["exp_day"] = "Day-PT"
        baseline_match["protocol"] = "Rest-state"
        baseline_match.pop("state", None)  # Awake / Iso baseline are equally valid.
        post = dcfg["input"].setdefault("poststroke", {}).setdefault("match", {})
        post["exp_day"] = ["1-5h-PT", "1d-PT", "3d-PT", "PT"]
        post["protocol"] = "Rest-state"
        post.pop("state", None)
        dcfg.setdefault("manual_gui", {}).setdefault("selection", {})["require_unique_baseline"] = True
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)
    paths = {**paths, **build_damage_paths(config, paths)}

    metadata_path = paths["metadata_registration_csv"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_registration.csv not found: {metadata_path}")
    df = pd.read_csv(metadata_path)
    required = {"stem", "group_id", "ref_stem"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"metadata_registration.csv is missing columns: {missing}")

    batch_cfg = cfg_get(config, "damage_assessment", "batch", default={}) or {}
    if include_bad is None:
        include_bad = bool(batch_cfg.get("include_bad", False))
    if not include_bad and "is_bad" in df.columns:
        df = df[~_bool_series(df["is_bad"])].copy()

    # Group/stem/metadata filters select post-stroke targets, but baseline rows
    # must remain available for comparison. Therefore baseline selection is done
    # from the full usable group before target-only filters are applied.
    if groups:
        df = df[df["group_id"].astype(str).isin({str(v) for v in groups})].copy()
    combined_filters = _merge_filters(batch_cfg.get("filters"), filters)

    post_match = cfg_get(config, "damage_assessment", "input", "poststroke", "match", default={}) or {}
    rows: list[dict[str, Any]] = []
    for group_id, group in df.groupby(df["group_id"].astype(str), sort=False):
        group = group.copy().reset_index(drop=True)
        baseline = None
        baseline_issue = ""
        try:
            baseline = _select_baseline_row(group, config)
        except Exception as exc:
            # Only the MANUAL GUI may review a target without a Day-PT baseline.
            # Ambiguous baselines and other selection errors still require review;
            # they must not silently select a reference or change analysis results.
            if for_manual_gui and str(exc).startswith("No baseline row matched"):
                baseline_issue = str(exc)
            else:
                rows.append({
                    "group_id": str(group_id),
                    "stem": "",
                    "ref_stem": str(group.iloc[0].get("ref_stem", "")),
                    "baseline_stem": "",
                    "index_status": "baseline_error",
                    "index_error": str(exc),
                })
                continue

        post_rows = group[_match_mask(group, post_match)].copy()
        if baseline is not None:
            post_rows = post_rows[~post_rows["stem"].astype(str).eq(str(baseline["stem"]))]
        if combined_filters:
            post_rows = post_rows[_match_mask(post_rows, combined_filters)].copy()
        if stems:
            post_rows = post_rows[post_rows["stem"].astype(str).isin({str(v) for v in stems})].copy()
        sort_cols = [c for c in ["exp_day", "state", "protocol", "stem"] if c in post_rows.columns]
        if sort_cols:
            post_rows = post_rows.sort_values(sort_cols)

        for _, current in post_rows.iterrows():
            row = current.to_dict()
            row.update({
                "baseline_stem": str(baseline["stem"]) if baseline is not None else "",
                "baseline_ref_stem": str(baseline.get("ref_stem", "")) if baseline is not None else "",
                "baseline_status": "selected" if baseline is not None else "missing",
                "index_status": "ready",
                "index_error": baseline_issue,
            })
            mpaths = damage_mask_paths(paths, str(current["stem"]))
            row.update({
                "lesion_auto_path": str(mpaths["lesion_auto"]),
                "lesion_final_path": str(mpaths["lesion_final"]),
                "lesion_auto_exists": mpaths["lesion_auto"].exists(),
                "lesion_final_exists": mpaths["lesion_final"].exists(),
            })
            if not for_manual_gui:
                row.update({
                    "hemorrhage_final_path": str(mpaths["hemorrhage_final"]),
                    "hemorrhage_final_exists": mpaths["hemorrhage_final"].exists(),
                })
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and write_index:
        out.to_csv(paths["damage_index_csv"], index=False)
    return out.reset_index(drop=True), config, paths


# ============================================================================
# Manual registration overrides / transforms
# ============================================================================


def _manual_overrides_csv(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    """Resolve the Step3 manual-registration override CSV path."""
    manual_cfg = cfg_get(config, "step3", "manual_registration", default={}) or {}
    folder = manual_cfg.get("folder_name", "manual_registration")
    name = manual_cfg.get("overrides_csv_name", "manual_registration_overrides.csv")
    return Path(paths["step3_root"]) / str(folder) / str(name)


def load_manual_overrides(config: dict[str, Any], paths: dict[str, Path]) -> pd.DataFrame:
    """Read saved Step3 manual registration overrides, if available."""
    path = _manual_overrides_csv(config, paths)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "stem" not in df.columns:
        return pd.DataFrame()
    if "updated_at" in df.columns:
        df = df.sort_values("updated_at")
    subset = [c for c in ["group_id", "stem"] if c in df.columns]
    if subset:
        df = df.groupby(subset, sort=False).tail(1)
    else:
        df = df.groupby([df["stem"].astype(str)], sort=False).tail(1)
    return df.reset_index(drop=True)


def _finite_number(value: Any) -> float | None:
    """Convert a value to a finite float, or return None."""
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def transform_for_row(
    row: pd.Series,
    manual_overrides: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[EuclideanTransform | None, str, dict[str, float]]:
    """Resolve the rigid current-stem-to-reference transformation.

    Parameters
    ----------
    row : pandas.Series
        Step2 registration metadata for the selected stem.
    manual_overrides : pandas.DataFrame
        Saved Step3 manual registration overrides.
    config : dict
        Effective YAML configuration.

    Returns
    -------
    transform : skimage.transform.EuclideanTransform or None
        Current-to-reference transform; None denotes the reference stem itself.
    source : str
        Provenance of the selected transform.
    values : dict
        Translation in pixels and rotation in radians.

    Notes
    -----
    When enabled, valid Step3 manual transforms take precedence over Step2
    ``translation_x``, ``translation_y``, and ``rotation_rad``.
    """
    stem = str(row.get("stem", ""))
    ref_stem = str(row.get("ref_stem", ""))
    if stem == ref_stem:
        return None, "identity_reference", {"translation_x": 0.0, "translation_y": 0.0, "rotation_rad": 0.0}

    use_manual = bool(cfg_get(config, "damage_assessment", "input", "use_manual_registration_overrides", default=True))
    if use_manual and not manual_overrides.empty:
        sub = manual_overrides[manual_overrides["stem"].astype(str).eq(stem)]
        if "group_id" in manual_overrides.columns and str(row.get("group_id", "")):
            sub = sub[sub["group_id"].astype(str).eq(str(row.get("group_id", "")))]
        if not sub.empty:
            r = sub.iloc[-1]
            tx = _finite_number(r.get("final_translation_x"))
            ty = _finite_number(r.get("final_translation_y"))
            rot = _finite_number(r.get("final_rotation_rad"))
            if tx is not None and ty is not None and rot is not None:
                return (
                    EuclideanTransform(rotation=rot, translation=(tx, ty)),
                    "step3_manual_final_transform",
                    {"translation_x": tx, "translation_y": ty, "rotation_rad": rot},
                )

    tx = _finite_number(row.get("translation_x"))
    ty = _finite_number(row.get("translation_y"))
    rot = _finite_number(row.get("rotation_rad"))
    if tx is None or ty is None or rot is None:
        raise RuntimeError(
            f"Missing current->reference transform for stem={stem}: "
            f"translation_x={row.get('translation_x')}, translation_y={row.get('translation_y')}, "
            f"rotation_rad={row.get('rotation_rad')}"
        )
    return (
        EuclideanTransform(rotation=rot, translation=(tx, ty)),
        "step2_auto_transform",
        {"translation_x": tx, "translation_y": ty, "rotation_rad": rot},
    )


def load_registered_image_for_row(
    row: pd.Series,
    manual_overrides: pd.DataFrame,
    config: dict[str, Any],
    path_cfg: dict[str, Any],
) -> tuple[np.ndarray, Path, str]:
    """Load the available fixed, manually registered or automatic registered BMP."""
    stem = str(row.get("stem", ""))
    ref_stem = str(row.get("ref_stem", ""))
    if stem == ref_stem:
        path = _first_existing_path(
            row,
            ["fixed_registration_path_resolved", "fixed_registration_path"],
            path_cfg,
            what=f"reference image for stem={stem}",
        )
        return _read_gray(path), path, "reference_fixed"

    use_manual = bool(cfg_get(config, "damage_assessment", "input", "use_manual_registration_overrides", default=True))
    if use_manual and not manual_overrides.empty:
        sub = manual_overrides[manual_overrides["stem"].astype(str).eq(stem)]
        if "group_id" in manual_overrides.columns and str(row.get("group_id", "")):
            sub = sub[sub["group_id"].astype(str).eq(str(row.get("group_id", "")))]
        if not sub.empty:
            p = _resolve_optional_path(sub.iloc[-1].get("manual_registered_path", ""), path_cfg, must_exist=False)
            if p is not None and p.exists():
                return _read_gray(p), p, "step3_manual_registered_bmp"

    path = _first_existing_path(
        row,
        ["saved_registered_path_resolved", "saved_registered_path"],
        path_cfg,
        what=f"registered image for stem={stem}",
    )
    return _read_gray(path), path, "step2_registered_bmp"


def load_native_image_for_row(row: pd.Series, path_cfg: dict[str, Any]) -> tuple[np.ndarray, Path]:
    """Read the current stem image in rotated full-resolution coordinates."""
    stem = str(row.get("stem", ""))
    if stem == str(row.get("ref_stem", "")):
        cols = ["fixed_registration_path_resolved", "fixed_registration_path"]
    else:
        cols = ["moving_registration_path_resolved", "moving_registration_path"]
    path = _first_existing_path(row, cols, path_cfg, what=f"native registration image for stem={stem}")
    return _read_gray(path), path


# ============================================================================
# Brain / geometry helpers
# ============================================================================


def load_current_brain_mask(row: pd.Series, paths: dict[str, Path], image_shape: tuple[int, int], config: dict[str, Any]) -> tuple[np.ndarray, Path, str]:
    """Load the manual or automatic canonical brain mask for the current stem.

    Parameters
    ----------
    row : pandas.Series
        Step2 registration metadata for the recording.
    paths : dict of str to pathlib.Path
        Standard project and mask paths.
    image_shape : tuple of int
        Expected full-resolution image shape ``(y, x)`` in pixels.
    config : dict
        Effective project configuration.

    Returns
    -------
    mask : numpy.ndarray
        Boolean mask with shape ``(y, x)`` in the current stem's rotated
        full-resolution image space.
    path : pathlib.Path
        Mask file selected from the canonical Step2/Step3 paths.
    source : str
        Provenance label distinguishing the manual and automatic brain masks.

    Notes
    -----
    The optional resize behavior is governed by
    ``damage_assessment.input.resize_brain_mask_if_needed``.
    """
    stem = str(row["stem"])
    candidates = canonical_mask_paths(paths, stem)
    if candidates["brain_manual"].exists():
        path = candidates["brain_manual"]
        source = "brain_manual"
    elif candidates["brain_auto"].exists():
        path = candidates["brain_auto"]
        source = "brain_auto"
    else:
        raise FileNotFoundError(
            f"No current brain mask for stem={stem}. Tried: "
            f"{candidates['brain_manual']}; {candidates['brain_auto']}"
        )

    mask = np.load(path).astype(bool)
    if tuple(mask.shape) != tuple(image_shape):
        if not bool(cfg_get(config, "damage_assessment", "input", "resize_brain_mask_if_needed", default=False)):
            raise ValueError(f"Brain mask shape mismatch for stem={stem}: {mask.shape} vs image {image_shape}")
        mask = resize(
            mask.astype(np.uint8),
            image_shape,
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ) > 0.5
    return mask.astype(bool), path, source


def load_reference_axis_angle(config: dict[str, Any], paths: dict[str, Path], row: pd.Series) -> tuple[float, str]:
    """Get the reference midline angle from Step3 annotation or a configured fallback."""
    expected_cfg = cfg_get(config, "damage_assessment", "detection", "expected_center", default={}) or {}
    source = str(expected_cfg.get("axis_angle_source", "step3_annotation")).strip().lower()
    fallback = float(expected_cfg.get("fallback_axis_angle_deg", -10.0))
    if source == "fixed":
        return fallback, "fixed_config"

    ann_path = paths["reference_annotations_csv"]
    if ann_path.exists():
        ann = pd.read_csv(ann_path)
        if not ann.empty and "group_id" in ann.columns:
            sub = ann[ann["group_id"].astype(str).eq(str(row.get("group_id", "")))]
            if "ref_stem" in ann.columns:
                sub = sub[sub["ref_stem"].astype(str).eq(str(row.get("ref_stem", "")))]
            if not sub.empty:
                if "updated_at" in sub.columns:
                    sub = sub.sort_values("updated_at")
                value = _finite_number(sub.iloc[-1].get("axis_angle_deg"))
                if value is not None:
                    return value, "step3_reference_annotation"
    return fallback, "fallback_config"


def anatomical_offset_to_image_yx(
    dx_px: float,
    dy_px: float,
    axis_angle_deg: float,
) -> tuple[float, float, dict[str, float]]:
    """Convert anatomical lesion offsets to image-coordinate displacement.

    User-facing convention:
      * dx is perpendicular to the annotated midline/AP axis;
      * dx < 0 moves to image/anatomical right;
      * dx > 0 moves to left;
      * dy is parallel to the annotated axis;
      * dy < 0 moves down along that axis;
      * dy > 0 moves up along that axis.

    ``axis_angle_deg`` comes from the Step3 line angle. The order in which the
    two line endpoints were drawn must not change the result, so the axis vector
    is always re-oriented toward increasing image y ("down"). A right-pointing
    perpendicular is then chosen by requiring a positive image-x component.

    Returns ``(delta_y, delta_x, basis_metadata)``.
    """
    angle = np.deg2rad(float(axis_angle_deg))

    # Raw axis direction in image (x, y) coordinates.
    axis_x = float(np.cos(angle))
    axis_y = float(np.sin(angle))

    # Make the axis direction deterministic: it must point image-down.
    # The tie-break handles a theoretically horizontal line.
    if axis_y < 0 or (abs(axis_y) < 1e-12 and axis_x < 0):
        axis_x = -axis_x
        axis_y = -axis_y

    # Basis vectors stored as (y, x).
    down_y, down_x = axis_y, axis_x
    right_y, right_x = -axis_x, axis_y

    # Requested signs: negative dx -> right; negative dy -> down.
    right_amount_px = -float(dx_px)
    down_amount_px = -float(dy_px)

    delta_y = down_amount_px * down_y + right_amount_px * right_y
    delta_x = down_amount_px * down_x + right_amount_px * right_x

    return float(delta_y), float(delta_x), {
        "axis_down_unit_y": float(down_y),
        "axis_down_unit_x": float(down_x),
        "axis_right_unit_y": float(right_y),
        "axis_right_unit_x": float(right_x),
        "requested_right_px": float(right_amount_px),
        "requested_down_px": float(down_amount_px),
    }


def expected_point_in_reference(row: pd.Series, config: dict[str, Any], paths: dict[str, Path]) -> tuple[tuple[float, float] | None, dict[str, Any]]:
    """Compute the configured expected lesion center relative to reference bregma.

    Parameters
    ----------
    row : pandas.Series
        Registration record with reference bregma coordinates in pixels.
    config : dict
        Uses ``damage_assessment.detection.expected_center``.
    paths : dict
        Paths to Step3 reference annotations and other project outputs.

    Returns
    -------
    point_yx : tuple of float or None
        Expected center as ``(y, x)`` on the reference full-resolution grid;
        None when disabled or when the reference bregma is unavailable.
    metadata : dict
        Expected offsets, axis source, or reason the point is unavailable.

    Notes
    -----
    Offsets ``dx_mm`` and ``dy_mm`` are divided by ``pixel_size_mm`` before
    applying the Step3 midline basis. The fallback axis angle is used only as
    specified by the function's existing configuration logic.
    """
    ecfg = cfg_get(config, "damage_assessment", "detection", "expected_center", default={}) or {}
    if not bool(ecfg.get("enabled", True)):
        return None, {"expected_center_enabled": False}

    bregma_y = _finite_number(row.get("bregma_ref_cy"))
    bregma_x = _finite_number(row.get("bregma_ref_cx"))
    if bregma_y is None or bregma_x is None:
        return None, {"expected_center_enabled": True, "expected_center_error": "reference bregma missing"}

    pixel_size_mm = float(ecfg.get("pixel_size_mm", 0.053))
    if pixel_size_mm <= 0:
        raise ValueError("damage_assessment.detection.expected_center.pixel_size_mm must be > 0")
    dx_px = float(ecfg.get("dx_mm", -2.0)) / pixel_size_mm
    dy_px = float(ecfg.get("dy_mm", 1.0)) / pixel_size_mm
    axis_angle, angle_source = load_reference_axis_angle(config, paths, row)
    delta_y, delta_x, basis_meta = anatomical_offset_to_image_yx(dx_px, dy_px, axis_angle)
    point = (float(bregma_y + delta_y), float(bregma_x + delta_x))
    return point, {
        "expected_center_enabled": True,
        "bregma_ref_y": bregma_y,
        "bregma_ref_x": bregma_x,
        "expected_dx_mm": float(ecfg.get("dx_mm", -2.0)),
        "expected_dy_mm": float(ecfg.get("dy_mm", 1.0)),
        "pixel_size_mm": pixel_size_mm,
        "axis_angle_deg": axis_angle,
        "axis_angle_source": angle_source,
        **basis_meta,
        "expected_ref_y": point[0],
        "expected_ref_x": point[1],
    }


def point_ref_to_current(point_yx: tuple[float, float] | None, tform_current_to_ref: EuclideanTransform | None) -> tuple[float, float] | None:
    """Map a reference-image point into the current stem coordinate space."""
    if point_yx is None:
        return None
    y_ref, x_ref = point_yx
    if tform_current_to_ref is None:
        return float(y_ref), float(x_ref)
    current_xy = tform_current_to_ref.inverse(np.asarray([[x_ref, y_ref]], dtype=float))[0]
    return float(current_xy[1]), float(current_xy[0])


# ============================================================================
# Detection
# ============================================================================


def warp_reference_to_current(
    reference_image: np.ndarray,
    tform_current_to_ref: EuclideanTransform | None,
    output_shape: tuple[int, int],
    *,
    order: int = 1,
) -> np.ndarray:
    """Resample a reference image onto the current stem's image grid.

    Parameters
    ----------
    reference_image : numpy.ndarray
        Reference-space image of shape ``(y, x)``.
    tform_current_to_ref : skimage.transform.EuclideanTransform or None
        Forward transformation from current coordinates to reference coordinates.
    output_shape : tuple of int
        Output ``(y, x)`` in current-stem pixels.
    order : int, optional
        Interpolation order passed to ``skimage.transform.warp``.

    Returns
    -------
    numpy.ndarray
        ``float32`` current-space image of shape ``output_shape``.

    Notes
    -----
    The forward current-to-reference transform is used as the inverse sampling
    map: each current output pixel samples its corresponding reference position.
    If the transform is None, the input is copied or resized as needed.
    """
    if tform_current_to_ref is None:
        if tuple(reference_image.shape) != tuple(output_shape):
            return resize(reference_image, output_shape, order=order, preserve_range=True, anti_aliasing=(order > 0)).astype(np.float32)
        return reference_image.astype(np.float32)

    # Output coordinates are in the current image. To sample the reference input,
    # map current -> reference with the forward Step2/Step3 transform.
    out = warp(
        reference_image.astype(np.float32),
        inverse_map=tform_current_to_ref,
        output_shape=output_shape,
        order=order,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    return out.astype(np.float32)


def normalize_pair(current: np.ndarray, baseline_warped: np.ndarray, mask: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalize current and warped baseline images inside the current brain mask.

    Parameters
    ----------
    current, baseline_warped : numpy.ndarray
        Two-dimensional intensity images on the same full-resolution grid.
    mask : numpy.ndarray
        Boolean current-brain support, shape ``(y, x)``.
    mode : str
        ``none``, ``mean`` or ``zscore``.

    Returns
    -------
    current_normalized, baseline_normalized : tuple of numpy.ndarray
        Two ``float32`` images of shape ``(y, x)`` with zeros outside the mask.

    Notes
    -----
    ``mean`` divides each image by its own mean inside the mask; ``zscore``
    subtracts its own mean and divides by its own standard deviation. The
    original small numerical stabilizer is retained in the implementation.
    """
    current = np.asarray(current, dtype=np.float32).copy()
    baseline = np.asarray(baseline_warped, dtype=np.float32).copy()
    m = mask.astype(bool)
    mode = str(mode).strip().lower()

    if not m.any():
        raise RuntimeError("Current brain mask is empty")

    if mode == "none":
        current[~m] = 0.0
        baseline[~m] = 0.0
        return current, baseline

    cur_vals = current[m]
    base_vals = baseline[m]
    if mode == "mean":
        current[m] = cur_vals / (float(np.mean(cur_vals)) + 1e-6)
        baseline[m] = base_vals / (float(np.mean(base_vals)) + 1e-6)
    elif mode == "zscore":
        current[m] = (cur_vals - float(np.mean(cur_vals))) / (float(np.std(cur_vals)) + 1e-6)
        baseline[m] = (base_vals - float(np.mean(base_vals))) / (float(np.std(base_vals)) + 1e-6)
    else:
        raise ValueError("damage_assessment.detection.normalization must be none, mean, or zscore")

    current[~m] = 0.0
    baseline[~m] = 0.0
    return current, baseline


def choose_difference(diff_signed: np.ndarray, mode: str) -> np.ndarray:
    """Select positive, negative or absolute contrast from a signed difference."""
    mode = str(mode).strip().lower()
    if mode == "positive":
        return np.clip(diff_signed, 0.0, None).astype(np.float32)
    if mode == "negative":
        return np.clip(-diff_signed, 0.0, None).astype(np.float32)
    if mode == "absolute":
        return np.abs(diff_signed).astype(np.float32)
    raise ValueError("damage_assessment.detection.difference_mode must be positive, negative, or absolute")


def clean_candidate(mask: np.ndarray, brain: np.ndarray, *, close_radius: int, open_radius: int, min_area_px: int) -> np.ndarray:
    """Apply morphological cleanup and brain masking to a lesion candidate."""
    out = mask.astype(bool)
    if close_radius > 0:
        out = binary_closing(out, disk(int(close_radius)))
    out = binary_fill_holes(out)
    if open_radius > 0:
        out = binary_opening(out, disk(int(open_radius)))
    if min_area_px > 0:
        out = remove_small_objects(out, min_size=int(min_area_px))
    out &= brain.astype(bool)
    return out.astype(bool)


def select_component(
    candidate: np.ndarray,
    *,
    method: str,
    target_yx: tuple[float, float] | None,
    max_distance_px: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select an automatic lesion candidate connected component.

    Parameters
    ----------
    candidate : numpy.ndarray
        Binary ``(y, x)`` candidate mask.
    method : str
        ``all``, ``largest`` or ``nearest_expected``.
    target_yx : tuple of float or None
        Expected center in current-stem pixel coordinates.
    max_distance_px : float or None
        Optional maximal centroid-to-target distance in pixels.

    Returns
    -------
    selected : numpy.ndarray
        Boolean ``(y, x)`` selected-component mask.
    metadata : dict
        Number of components, selected area and applicable centroid/distance.

    Notes
    -----
    When the expected center is missing, component selection uses the largest
    component. An empty selection is possible if the configured distance gate
    rejects all components.
    """
    method = str(method).strip().lower()
    if method == "all":
        area = int(candidate.astype(bool).sum())
        return candidate.astype(bool), {"component_method": method, "n_components": int(label(candidate).max()), "selected_area_px": area}

    lab = label(candidate.astype(bool))
    props = list(regionprops(lab.astype(np.int32)))
    if not props:
        return np.zeros_like(candidate, dtype=bool), {
            "component_method": method,
            "n_components": 0,
            "selected_area_px": 0,
            "selected_centroid_y": np.nan,
            "selected_centroid_x": np.nan,
            "selected_distance_px": np.nan,
        }

    selected = None
    distance = np.nan
    if method == "largest" or target_yx is None:
        selected = max(props, key=lambda rp: rp.area)
    elif method == "nearest_expected":
        ty, tx = target_yx
        ranked = sorted(props, key=lambda rp: float(np.hypot(rp.centroid[0] - ty, rp.centroid[1] - tx)))
        for rp in ranked:
            d = float(np.hypot(rp.centroid[0] - ty, rp.centroid[1] - tx))
            if max_distance_px is None or d <= float(max_distance_px):
                selected = rp
                distance = d
                break
    else:
        raise ValueError("damage_assessment.detection.component_selection must be nearest_expected, largest, or all")

    if selected is None:
        return np.zeros_like(candidate, dtype=bool), {
            "component_method": method,
            "n_components": len(props),
            "selected_area_px": 0,
            "selected_centroid_y": np.nan,
            "selected_centroid_x": np.nan,
            "selected_distance_px": np.nan,
        }

    cy, cx = selected.centroid
    if not np.isfinite(distance) and target_yx is not None:
        distance = float(np.hypot(cy - target_yx[0], cx - target_yx[1]))
    return (lab == selected.label), {
        "component_method": method,
        "n_components": len(props),
        "selected_area_px": int(selected.area),
        "selected_centroid_y": float(cy),
        "selected_centroid_x": float(cx),
        "selected_distance_px": float(distance) if np.isfinite(distance) else np.nan,
    }


def prune_to_expected_radius(mask: np.ndarray, target_yx: tuple[float, float] | None, radius_px: float | None) -> np.ndarray:
    """Restrict a candidate mask to a radius around its expected center."""
    if target_yx is None or radius_px is None or radius_px <= 0:
        return mask.astype(bool)
    yy, xx = np.indices(mask.shape)
    keep = np.hypot(yy - float(target_yx[0]), xx - float(target_yx[1])) <= float(radius_px)
    return mask.astype(bool) & keep


def mask_metrics(mask: np.ndarray, pixel_size_mm: float | None = None, prefix: str = "") -> dict[str, Any]:
    """Compute mask area and centroid, optionally expressing area in mm²."""
    m = mask.astype(bool)
    area = int(m.sum())
    out: dict[str, Any] = {f"{prefix}area_px": area}
    if area:
        ys, xs = np.nonzero(m)
        out[f"{prefix}centroid_y"] = float(np.mean(ys))
        out[f"{prefix}centroid_x"] = float(np.mean(xs))
    else:
        out[f"{prefix}centroid_y"] = np.nan
        out[f"{prefix}centroid_x"] = np.nan
    if pixel_size_mm is not None and pixel_size_mm > 0:
        out[f"{prefix}area_mm2"] = float(area) * float(pixel_size_mm) ** 2
    return out


def detect_damage_for_row(
    row: pd.Series,
    baseline_row: pd.Series,
    config: dict[str, Any],
    paths: dict[str, Path],
    manual_overrides: pd.DataFrame,
) -> dict[str, Any]:
    """Construct an automatic lesion candidate from baseline/current BMP differences.

    Parameters
    ----------
    row, baseline_row : pandas.Series
        Current and baseline metadata records.
    config : dict
        Project configuration, including ``damage_assessment.detection``.
    paths : dict
        Analysis paths for images, masks, annotations and output files.
    manual_overrides : pandas.DataFrame
        Step3 registration overrides for the applicable stems.

    Returns
    -------
    dict
        Full-resolution current/baseline images, signed and selected difference
        maps, candidate and final automatic masks of shape ``(y, x)``, plus
        configuration, geometry, and selection provenance.

    Notes
    -----
    The implemented order is: load native current and registered baseline;
    warp baseline to current coordinates; load the current brain mask; normalize
    both images; select and smooth their difference; threshold; clean connected
    components; optionally restrict by anatomical center and radius. This is an
    automatic candidate only: the manual GUI uses its own display helpers and
    manual lesion polygons.
    """
    path_cfg = get_path_resolving_config(config, "step3")
    current, current_path = load_native_image_for_row(row, path_cfg)
    baseline_ref, baseline_registered_path, baseline_registered_source = load_registered_image_for_row(
        baseline_row, manual_overrides, config, path_cfg
    )
    tform, registration_source, transform_values = transform_for_row(row, manual_overrides, config)
    baseline_warped = warp_reference_to_current(baseline_ref, tform, current.shape, order=1)

    brain, brain_path, brain_source = load_current_brain_mask(row, paths, current.shape, config)
    normalization = str(cfg_get(config, "damage_assessment", "detection", "normalization", default="mean"))
    current_work, baseline_work = normalize_pair(current, baseline_warped, brain, normalization)

    diff_signed = current_work - baseline_work
    diff_signed[~brain] = 0.0
    difference_mode = str(cfg_get(config, "damage_assessment", "detection", "difference_mode", default="positive"))
    diff_selected = choose_difference(diff_signed, difference_mode)
    diff_selected[~brain] = 0.0

    sigma = float(cfg_get(config, "damage_assessment", "detection", "gaussian_sigma", default=1.0))
    diff_smooth = gaussian(diff_selected, sigma=sigma, preserve_range=True).astype(np.float32) if sigma > 0 else diff_selected.copy()
    diff_smooth[~brain] = 0.0

    threshold = float(cfg_get(config, "damage_assessment", "detection", "threshold", default=0.15))
    candidate_raw = (diff_smooth > threshold) & brain
    candidate_clean = clean_candidate(
        candidate_raw,
        brain,
        close_radius=int(cfg_get(config, "damage_assessment", "detection", "close_radius", default=2)),
        open_radius=int(cfg_get(config, "damage_assessment", "detection", "open_radius", default=1)),
        min_area_px=int(cfg_get(config, "damage_assessment", "detection", "min_area_px", default=50)),
    )

    expected_ref_yx, expected_meta = expected_point_in_reference(row, config, paths)
    expected_current_yx = point_ref_to_current(expected_ref_yx, tform)
    ecfg = cfg_get(config, "damage_assessment", "detection", "expected_center", default={}) or {}
    max_dist = ecfg.get("max_component_distance_px")
    max_dist = None if max_dist in (None, "") else float(max_dist)
    selected, component_meta = select_component(
        candidate_clean,
        method=str(cfg_get(config, "damage_assessment", "detection", "component_selection", default="nearest_expected")),
        target_yx=expected_current_yx,
        max_distance_px=max_dist,
    )

    radius_mm = ecfg.get("lesion_radius_mm")
    pixel_size_mm = float(ecfg.get("pixel_size_mm", 0.053))
    radius_px = None if radius_mm in (None, "") else float(radius_mm) / pixel_size_mm
    lesion_auto = prune_to_expected_radius(selected, expected_current_yx, radius_px)
    lesion_auto = binary_fill_holes(lesion_auto)
    min_area = int(cfg_get(config, "damage_assessment", "detection", "min_area_px", default=50))
    if min_area > 0:
        lesion_auto = remove_small_objects(lesion_auto.astype(bool), min_size=min_area)
    lesion_auto &= brain

    return {
        "current_image": current.astype(np.float32),
        "current_image_path": current_path,
        "baseline_ref_image": baseline_ref.astype(np.float32),
        "baseline_registered_path": baseline_registered_path,
        "baseline_registered_source": baseline_registered_source,
        "baseline_warped": baseline_warped.astype(np.float32),
        "brain": brain.astype(bool),
        "brain_path": brain_path,
        "brain_source": brain_source,
        "diff_signed": diff_signed.astype(np.float32),
        "diff_selected": diff_selected.astype(np.float32),
        "diff_smooth": diff_smooth.astype(np.float32),
        "candidate_raw": candidate_raw.astype(bool),
        "candidate_clean": candidate_clean.astype(bool),
        "lesion_auto": lesion_auto.astype(bool),
        "expected_current_yx": expected_current_yx,
        "registration_source": registration_source,
        "transform_values": transform_values,
        "normalization": normalization,
        "difference_mode": difference_mode,
        "threshold": threshold,
        "gaussian_sigma": sigma,
        "expected_meta": expected_meta,
        "component_meta": component_meta,
        "pixel_size_mm": pixel_size_mm,
        "lesion_radius_px": radius_px,
    }


# ============================================================================
# Saving / preview / Napari
# ============================================================================


def _normalise01(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Scale finite image values to the unit display interval."""
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    return np.clip((arr - lo) / max(float(hi - lo), 1e-12), 0, 1).astype(np.float32)


def save_preview_png(
    path: Path,
    result: dict[str, Any],
    lesion_final: np.ndarray,
    hemorrhage_final: np.ndarray,
    title: str,
) -> None:
    """Save a visual summary of lesion and hemorrhage masks with difference maps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(result["current_image"], cmap="gray")
    axes[0].set_title("Current native")
    axes[1].imshow(result["baseline_warped"], cmap="gray")
    axes[1].set_title("Baseline warped")
    axes[2].imshow(result["diff_smooth"], cmap="magma")
    axes[2].set_title("Difference")
    axes[3].imshow(result["current_image"], cmap="gray")
    axes[3].contour(lesion_final.astype(bool), levels=[0.5], linewidths=1.2)
    if hemorrhage_final.astype(bool).any():
        axes[3].contour(hemorrhage_final.astype(bool), levels=[0.5], linewidths=1.2, linestyles="--")
    axes[3].set_title("Final masks")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_automatic_outputs(row: pd.Series, result: dict[str, Any], config: dict[str, Any], paths: dict[str, Path], *, overwrite_auto: bool) -> None:
    """Write automatic lesion candidates and intermediate difference arrays."""
    stem = str(row["stem"])
    mpaths = damage_mask_paths(paths, stem)
    dpaths = damage_diff_paths(paths, stem)

    if overwrite_auto or not mpaths["lesion_auto"].exists():
        np.save(mpaths["lesion_auto"], result["lesion_auto"].astype(np.uint8))
        np.save(mpaths["brain_used"], result["brain"].astype(np.uint8))

    if bool(cfg_get(config, "damage_assessment", "output", "save_diff_maps", default=True)):
        if overwrite_auto or not dpaths["diff_smooth"].exists():
            np.save(dpaths["baseline_warped"], result["baseline_warped"].astype(np.float32))
            np.save(dpaths["diff_signed"], result["diff_signed"].astype(np.float32))
            np.save(dpaths["diff_selected"], result["diff_selected"].astype(np.float32))
            np.save(dpaths["diff_smooth"], result["diff_smooth"].astype(np.float32))
            np.save(dpaths["candidate_raw"], result["candidate_raw"].astype(np.uint8))
            np.save(dpaths["candidate_clean"], result["candidate_clean"].astype(np.uint8))


def _load_initial_final_masks(row: pd.Series, result: dict[str, Any], config: dict[str, Any], paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load existing final masks for review or initialize editable masks."""
    mpaths = damage_mask_paths(paths, str(row["stem"]))
    preload = bool(cfg_get(config, "damage_assessment", "review", "preload_existing_final_masks", default=True))
    if preload and mpaths["lesion_final"].exists():
        lesion = np.load(mpaths["lesion_final"]).astype(bool)
    else:
        lesion = result["lesion_auto"].astype(bool).copy()
    if preload and mpaths["hemorrhage_final"].exists():
        hemorrhage = np.load(mpaths["hemorrhage_final"]).astype(bool)
    else:
        hemorrhage = np.zeros_like(lesion, dtype=bool)
    if lesion.shape != result["current_image"].shape or hemorrhage.shape != result["current_image"].shape:
        raise ValueError(f"Existing final mask shape mismatch for stem={row['stem']}")
    return lesion, hemorrhage


def review_in_napari(row: pd.Series, result: dict[str, Any], config: dict[str, Any], paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Open the interactive lesion and hemorrhage review layers in Napari."""
    if napari is None:
        raise ImportError("Napari is required for interactive damage review. Run in the same environment where Step3 Napari works.")

    lesion_initial, hemorrhage_initial = _load_initial_final_masks(row, result, config, paths)
    viewer = napari.Viewer(title=f"Damage assessment: {row['stem']}")
    viewer.add_image(result["current_image"], name="current_day_native", visible=True)
    viewer.add_image(result["baseline_warped"], name="baseline_warped_to_current", visible=False)
    viewer.add_image(result["diff_signed"], name="diff_signed", visible=False)
    viewer.add_image(result["diff_selected"], name=f"diff_{result['difference_mode']}", visible=False)
    viewer.add_image(result["diff_smooth"], name="diff_smooth", visible=True, colormap="magma")
    viewer.add_labels(result["brain"].astype(np.uint8), name="brain_current", opacity=0.20, visible=False)
    viewer.add_labels(result["candidate_raw"].astype(np.uint8), name="lesion_candidate_raw", opacity=0.25, visible=False)
    viewer.add_labels(result["lesion_auto"].astype(np.uint8), name="lesion_auto", opacity=0.35, visible=False)
    lesion_layer = viewer.add_labels(
        lesion_initial.astype(np.uint8),
        name="lesion_final_EDIT",
        opacity=float(cfg_get(config, "damage_assessment", "review", "lesion_opacity", default=0.55)),
        visible=True,
    )
    hemorrhage_layer = viewer.add_labels(
        hemorrhage_initial.astype(np.uint8),
        name="hemorrhage_final_EDIT",
        opacity=float(cfg_get(config, "damage_assessment", "review", "hemorrhage_opacity", default=0.55)),
        visible=True,
    )

    expected = result.get("expected_current_yx")
    if expected is not None and np.all(np.isfinite(expected)):
        viewer.add_points(
            np.asarray([[expected[0], expected[1]]], dtype=float),
            name="expected_lesion_center",
            size=9,
            face_color="yellow",
            visible=True,
        )

    print()
    print("Damage assessment")
    print("stem:", row["stem"])
    print("baseline:", row["baseline_stem"])
    print("Edit lesion in layer: lesion_final_EDIT")
    print("Draw hemorrhage only if present in layer: hemorrhage_final_EDIT")
    print("Use label value 1. Erase with label 0. Close Napari to save.")
    print()
    napari.run()

    return (np.asarray(lesion_layer.data) > 0), (np.asarray(hemorrhage_layer.data) > 0)


def _review_status(lesion_auto: np.ndarray, lesion_final: np.ndarray) -> str:
    """Classify a reviewed lesion mask relative to its automatic candidate."""
    auto = lesion_auto.astype(bool)
    final = lesion_final.astype(bool)
    if not final.any():
        return "empty"
    if np.array_equal(auto, final):
        return "accepted_auto"
    if not auto.any():
        return "drawn_manual"
    return "edited_manual"


def save_final_outputs(
    row: pd.Series,
    baseline_row: pd.Series,
    result: dict[str, Any],
    lesion_final: np.ndarray,
    hemorrhage_final: np.ndarray,
    config: dict[str, Any],
    paths: dict[str, Path],
    *,
    overwrite_final: bool,
    viewer_used: bool,
) -> dict[str, Any]:
    """Write reviewed lesion and hemorrhage masks with provenance metadata.

    Parameters
    ----------
    row, baseline_row : pandas.Series
        Current and baseline metadata records.
    result : dict
        Automatic detection output and its provenance.
    lesion_final, hemorrhage_final : numpy.ndarray
        Full-resolution binary masks in current-stem coordinates ``(y, x)``.
    config : dict
        Effective ``damage_assessment`` settings.
    paths : dict
        Canonical mask paths and summary table paths.
    overwrite_final, viewer_used : bool
        Determine whether existing manual masks are preserved in noninteractive
        mode and whether an actual Napari review occurred.

    Returns
    -------
    dict
        Review record containing mask paths, areas, origins and timestamps.

    Notes
    -----
    A noninteractive run with ``overwrite_final=False`` can preserve existing
    final masks. The saved JSON records ``used_by_step4=False``.
    """
    stem = str(row["stem"])
    mpaths = damage_mask_paths(paths, stem)
    if not overwrite_final and mpaths["lesion_final"].exists() and not viewer_used:
        # Non-interactive runs must never silently replace existing manual work.
        lesion_final = np.load(mpaths["lesion_final"]).astype(bool)
    else:
        np.save(mpaths["lesion_final"], lesion_final.astype(np.uint8))

    if not overwrite_final and mpaths["hemorrhage_final"].exists() and not viewer_used:
        hemorrhage_final = np.load(mpaths["hemorrhage_final"]).astype(bool)
    else:
        np.save(mpaths["hemorrhage_final"], hemorrhage_final.astype(np.uint8))

    pixel_size_mm = result.get("pixel_size_mm")
    record: dict[str, Any] = {
        "group_id": str(row.get("group_id", "")),
        "stem": stem,
        "ref_stem": str(row.get("ref_stem", "")),
        "baseline_stem": str(baseline_row.get("stem", "")),
        "registration_source": result["registration_source"],
        "current_image_path": str(result["current_image_path"]),
        "baseline_registered_path": str(result["baseline_registered_path"]),
        "brain_path": str(result["brain_path"]),
        "brain_source": result["brain_source"],
        "normalization": result["normalization"],
        "difference_mode": result["difference_mode"],
        "threshold": result["threshold"],
        "gaussian_sigma": result["gaussian_sigma"],
        "lesion_review_status": _review_status(result["lesion_auto"], lesion_final),
        "hemorrhage_present": bool(hemorrhage_final.any()),
        "viewer_used": bool(viewer_used),
        "annotator": str(cfg_get(config, "damage_assessment", "annotator", default="user")),
        "updated_at": _now_iso(),
        "lesion_auto_path": str(mpaths["lesion_auto"]),
        "lesion_final_path": str(mpaths["lesion_final"]),
        "hemorrhage_final_path": str(mpaths["hemorrhage_final"]),
    }
    record.update(result["transform_values"])
    record.update(result["expected_meta"])
    if result.get("expected_current_yx") is not None:
        record["expected_current_y"] = float(result["expected_current_yx"][0])
        record["expected_current_x"] = float(result["expected_current_yx"][1])
    record.update(result["component_meta"])
    record.update(mask_metrics(result["lesion_auto"], pixel_size_mm, prefix="lesion_auto_"))
    record.update(mask_metrics(lesion_final, pixel_size_mm, prefix="lesion_final_"))
    record.update(mask_metrics(hemorrhage_final, pixel_size_mm, prefix="hemorrhage_"))

    meta = {
        **record,
        "image_space": "rotated_fullres_native_stem",
        "mask_dtype": "uint8_0_1",
        "bad_glare_used": False,
        "cross_day_common_mask_used": False,
        "used_by_step4": False,
    }
    _save_json(mpaths["review_meta"], meta)
    _append_or_upsert_csv(paths["damage_review_csv"], record, keys=["group_id", "stem"])

    if bool(cfg_get(config, "damage_assessment", "output", "save_preview_png", default=True)):
        preview_path = Path(paths["damage_previews_root"]) / f"{stem}_damage_preview.png"
        save_preview_png(
            preview_path,
            result,
            lesion_final,
            hemorrhage_final,
            title=f"{stem} | baseline={baseline_row['stem']}",
        )
        record["preview_path"] = str(preview_path)
    return record


# ============================================================================
# Runner
# ============================================================================


def process_damage_row(
    row: pd.Series,
    all_metadata: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
    manual_overrides: pd.DataFrame,
    *,
    show_viewer: bool,
    overwrite_auto: bool,
    overwrite_final: bool,
) -> dict[str, Any]:
    """Process one stem through automatic detection and optional manual review."""
    baseline_stem = str(row["baseline_stem"])
    baseline_rows = all_metadata[all_metadata["stem"].astype(str).eq(baseline_stem)]
    if baseline_rows.empty:
        raise RuntimeError(f"Baseline row not found in metadata_registration.csv: {baseline_stem}")
    baseline_row = baseline_rows.iloc[-1]

    result = detect_damage_for_row(row, baseline_row, config, paths, manual_overrides)
    save_automatic_outputs(row, result, config, paths, overwrite_auto=overwrite_auto)

    review_enabled = bool(cfg_get(config, "damage_assessment", "review", "enabled", default=True))
    viewer_used = bool(show_viewer and review_enabled)
    if viewer_used:
        lesion_final, hemorrhage_final = review_in_napari(row, result, config, paths)
    else:
        lesion_final, hemorrhage_final = _load_initial_final_masks(row, result, config, paths)
        if not bool(cfg_get(config, "damage_assessment", "review", "save_auto_as_final_without_viewer", default=False)):
            # Save only automatic outputs; do not claim a reviewed final mask.
            return {
                "group_id": str(row.get("group_id", "")),
                "stem": str(row["stem"]),
                "baseline_stem": baseline_stem,
                "status": "auto_saved_not_reviewed",
                "lesion_auto_path": str(damage_mask_paths(paths, str(row["stem"]))["lesion_auto"]),
            }

    record = save_final_outputs(
        row,
        baseline_row,
        result,
        lesion_final,
        hemorrhage_final,
        config,
        paths,
        overwrite_final=overwrite_final,
        viewer_used=viewer_used,
    )
    record["status"] = "reviewed" if viewer_used else "auto_saved_as_final"
    return record


def run_damage_assessment(
    config_path: str | Path,
    *,
    filters: dict[str, Any] | None = None,
    groups: Sequence[str] | None = None,
    stems: Sequence[str] | None = None,
    include_bad: bool | None = None,
    show_viewer: bool = True,
    only_unreviewed: bool = False,
    overwrite_auto: bool = False,
    overwrite_final: bool = False,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Run the existing automatic candidate and optional Napari-review workflow.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Project YAML configuration path.
    filters : dict or None, optional
        Additional metadata filters.
    groups, stems : sequence of str or None, optional
        Target selection by group and recording identifiers.
    include_bad : bool or None, optional
        Whether to include Step2 bad records.
    show_viewer : bool, optional
        Open Napari to review each generated candidate.
    only_unreviewed : bool, optional
        Skip stems with existing final lesion masks.
    overwrite_auto, overwrite_final : bool, optional
        Control replacement of automatic and final files, respectively.
    dry_run : bool, optional
        Return the selection table without running detection.

    Returns
    -------
    pandas.DataFrame
        Selection table in dry-run mode, otherwise per-stem outcome records.
    """
    index, config, paths = build_damage_index(
        config_path,
        filters=filters,
        groups=groups,
        stems=stems,
        include_bad=include_bad,
    )
    print("Damage index:", paths["damage_index_csv"])
    if index.empty:
        print("No post-stroke rows matched damage_assessment.input.poststroke.match")
        return index

    display_cols = [c for c in ["group_id", "stem", "baseline_stem", "exp_day", "state", "protocol", "index_status", "index_error"] if c in index.columns]
    print(index[display_cols].to_string(index=False))
    if dry_run:
        return index

    ready = index[index["index_status"].astype(str).eq("ready")].copy()
    if only_unreviewed:
        ready = ready[~ready["lesion_final_exists"].astype(bool)].copy()
    if bool(cfg_get(config, "damage_assessment", "review", "skip_reviewed", default=False)):
        ready = ready[~ready["lesion_final_exists"].astype(bool)].copy()

    metadata = pd.read_csv(paths["metadata_registration_csv"])
    manual_overrides = load_manual_overrides(config, paths)
    continue_on_error = bool(cfg_get(config, "damage_assessment", "batch", "continue_on_error", default=True))
    records: list[dict[str, Any]] = []

    for i, (_, row) in enumerate(ready.iterrows(), start=1):
        print(f"\n[{i}/{len(ready)}] Damage assessment: {row['stem']}")
        try:
            record = process_damage_row(
                row,
                metadata,
                config,
                paths,
                manual_overrides,
                show_viewer=show_viewer,
                overwrite_auto=overwrite_auto,
                overwrite_final=overwrite_final,
            )
            records.append(record)
            print("status:", record.get("status", "ok"))
        except Exception as exc:
            error = {
                "group_id": str(row.get("group_id", "")),
                "stem": str(row.get("stem", "")),
                "baseline_stem": str(row.get("baseline_stem", "")),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "updated_at": _now_iso(),
            }
            _append_or_upsert_csv(paths["damage_errors_csv"], error, keys=["group_id", "stem"])
            records.append({**error, "status": "error"})
            print("[ERROR]", exc)
            if not continue_on_error:
                raise

    return pd.DataFrame(records)


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """Parse and execute the automatic damage assessment command-line interface."""
    parser = argparse.ArgumentParser(description="Simple per-stem lesion and hemorrhage assessment")
    parser.add_argument("--config", required=True, help="Path to the main project YAML config")
    parser.add_argument("--filter", action="append", default=None, help="Extra exact filter KEY=VALUE or KEY=a,b")
    parser.add_argument("--groups", nargs="+", default=None, help="Selected group_id values")
    parser.add_argument("--stems", nargs="+", default=None, help="Selected post-stroke stem values")
    parser.add_argument("--include-bad", action="store_true", help="Include rows marked is_bad")
    parser.add_argument("--no-viewer", action="store_true", help="Build automatic candidates without opening Napari")
    parser.add_argument("--only-unreviewed", action="store_true", help="Skip stems with an existing lesion_final.npy")
    parser.add_argument("--overwrite-auto", action="store_true", help="Recompute and overwrite automatic/intermediate files")
    parser.add_argument("--overwrite-final", action="store_true", help="Allow replacement of final masks in non-interactive mode")
    parser.add_argument("--dry-run", action="store_true", help="Only build and print damage_index.csv")
    args = parser.parse_args()

    filters = _parse_cli_filters(args.filter)
    results = run_damage_assessment(
        args.config,
        filters=filters,
        groups=args.groups,
        stems=args.stems,
        include_bad=True if args.include_bad else None,
        show_viewer=not args.no_viewer,
        only_unreviewed=args.only_unreviewed,
        overwrite_auto=args.overwrite_auto,
        overwrite_final=args.overwrite_final,
        dry_run=args.dry_run,
    )
    if not results.empty:
        print("\nRun summary")
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
