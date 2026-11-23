"""Step 3 runner for group-level Napari QC and anatomical reference annotation."""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from common import (
    load_config,
    create_or_load_project,
    build_standard_paths,
    get_stem_keys,
    get_path_resolving_config,
    resolve_flexible_path,
    raw_rotation_degrees,
    raw_scale_factor,
    scaled_coordinate_value,
    canonical_mask_paths,
    mask_for_reg_union_path,
    mask_for_reg_per_stem_path,
    drop_mask_path_metadata_columns,
    parse_variant_value,
)
from step3_review_group import review_group_df
from step3_annotation import annotate_reference_group


# ============================================================
# Helpers
# ============================================================

def normalize_bool_series(s: pd.Series) -> pd.Series:
    """Interpret common truthy CSV values as a boolean pandas series."""
    return s.astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES"])


def _as_optional_set(x: Any) -> set[str] | None:
    """Normalize an optional scalar or collection to a set of strings."""
    if x is None:
        return None
    if isinstance(x, (str, int, float)):
        return {str(x)}
    return {str(v) for v in x}


def apply_metadata_filters_df(
    df: pd.DataFrame, filters: dict[str, Any] | None, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Keep metadata rows matching the configured allowed values."""
    out = df.copy()
    variants = (config or {}).get("variant_keys", {}) or {}
    for key, allowed_raw in (filters or {}).items():
        if key not in out.columns:
            raise ValueError(f"Filter column not found in metadata_registration.csv: {key}")
        allowed = _as_optional_set(allowed_raw)
        if allowed is None:
            continue
        variant_cfg = variants.get(key, {}) or {}
        if variant_cfg.get("enabled", False):
            separators = variant_cfg.get("separators", ["", "-", "_"])
            actual = out[key].apply(lambda v: parse_variant_value(v, separators)[0])
            allowed = {parse_variant_value(v, separators)[0] for v in allowed}
        else:
            actual = out[key].astype(str)
        out = out[actual.isin(allowed)].copy()
    return out


def resolve_path_column(
    df: pd.DataFrame,
    column: str,
    path_cfg: dict[str, Any],
    allow_missing: bool = True,
) -> pd.DataFrame:
    """Add resolved-path and existence columns for a metadata path field."""
    df = df.copy()
    resolved_vals: list[str] = []
    exists_vals: list[bool] = []

    for v in df[column]:
        if pd.isna(v) or not str(v).strip():
            resolved_vals.append("")
            exists_vals.append(False)
            continue
        try:
            p = resolve_flexible_path(
                v,
                from_drive=path_cfg.get("from_drive"),
                to_drive=path_cfg.get("to_drive"),
                from_root=path_cfg.get("from_root"),
                to_root=path_cfg.get("to_root"),
                must_exist=not allow_missing,
            )
            resolved_vals.append(str(p) if p is not None else "")
            exists_vals.append(bool(p and p.exists()))
        except FileNotFoundError:
            if allow_missing:
                resolved_vals.append(str(v))
                exists_vals.append(False)
            else:
                raise

    df[f"{column}_resolved"] = resolved_vals
    df[f"{column}_exists"] = exists_vals
    return df


def resolve_metadata_paths(df: pd.DataFrame, path_cfg: dict[str, Any]) -> pd.DataFrame:
    """Resolve reference and registration image paths in Step2 metadata."""
    generic_cols = [
        "fixed_registration_path",
        "moving_registration_path",
        "saved_registered_path",
    ]
    dynamic_ref_cols = [c for c in df.columns if c.startswith("ref_") and c.endswith("_path")]
    for col in list(dict.fromkeys(generic_cols + dynamic_ref_cols)):
        if col in df.columns:
            df = resolve_path_column(df, col, path_cfg, allow_missing=True)
    return df


def get_sort_columns(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Choose available columns for deterministic ordering of group records."""
    candidates = [k for k in get_stem_keys(config) if k in df.columns]
    if not candidates:
        candidates = [c for c in ["exp_day", "state", "protocol", "stem"] if c in df.columns]
    if "stem" not in candidates and "stem" in df.columns:
        candidates.append("stem")
    return candidates


def reorder_reference_first(df_group: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Place the Step2 reference recording first within its group."""
    df_group = df_group.copy()
    ref_stems = [str(v) for v in df_group.get("ref_stem", pd.Series(dtype=str)).dropna().astype(str).unique() if str(v).strip()]
    ref_stem = ref_stems[0] if ref_stems else ""

    sort_cols = get_sort_columns(df_group, config)
    if ref_stem and "stem" in df_group.columns:
        ref_rows = df_group[df_group["stem"].astype(str) == ref_stem]
        if not ref_rows.empty:
            ref_row = ref_rows.iloc[[0]]
            others = df_group[df_group["stem"].astype(str) != ref_stem]
            if sort_cols:
                others = others.sort_values(sort_cols)
            return pd.concat([ref_row, others], axis=0).reset_index(drop=True)

    if sort_cols:
        return df_group.sort_values(sort_cols).reset_index(drop=True)
    return df_group.reset_index(drop=True)


def choose_reference_row(df_group: pd.DataFrame) -> pd.Series:
    """Select the unique reference recording designated by Step2."""
    if not {"stem", "ref_stem"}.issubset(df_group.columns):
        raise ValueError("Step3 requires stem and ref_stem columns")
    ref_stems = [str(v) for v in df_group["ref_stem"].dropna().astype(str).unique() if str(v).strip()]
    if len(ref_stems) != 1:
        raise ValueError(f"Step3 group has {len(ref_stems)} distinct ref_stem values: {ref_stems}")
    selected = df_group[df_group["stem"].astype(str).eq(ref_stems[0])]
    if len(selected) != 1:
        raise ValueError(f"Step3 ref_stem={ref_stems[0]!r} has {len(selected)} matching rows")
    return selected.iloc[0]

def _ring_like_roi_mode(value: Any) -> bool:
    """Identify registration ROI modes that use ring-like spatial masks."""
    mode = str(value or "ring").strip().lower()
    return mode in {"ring", "ring_sobel", "mask_mode", "legacy", "old"}


def choose_mask_paths(df_group: pd.DataFrame, masks_root: Path) -> Path | list[Path]:
    """Locate registration mask images for QC of a Step2 group.
    
    Parameters
    ----------
    df_group : pandas.DataFrame
        Metadata for one registration group, including ``stem`` and ROI mode.
    masks_root : pathlib.Path
        Root of canonical Step2 masks.
    
    Returns
    -------
    pathlib.Path or list of pathlib.Path
        A shared group-union mask when available for a ring-like ROI, or
        per-stem ``mask_for_reg.npy`` paths in group row order.
    
    Notes
    -----
    The configured registration ``roi_mode`` determines which mask is used
    for QC. A saved group-union file is not necessarily used when
    ``roi_mode`` is ``full_image``.
    """
    if df_group.empty:
        raise ValueError("df_group is empty")
    if "stem" not in df_group.columns:
        raise ValueError("df_group must contain stem")

    row0 = df_group.iloc[0]
    group_id = str(row0.get("group_id", ""))
    mask_mode = str(row0.get("mask_mode", "pairwise") or "pairwise").strip().lower()
    roi_mode = str(row0.get("registration_roi_mode", "ring") or "ring").strip().lower()

    if mask_mode in {"group_union", "hybrid"} and _ring_like_roi_mode(roi_mode):
        p = mask_for_reg_union_path(masks_root, group_id)
        if p.exists():
            return p
        print(f"[WARN] group-union registration mask not found, falling back to per-stem masks: {p}")

    paths = [mask_for_reg_per_stem_path(masks_root, stem) for stem in df_group["stem"].astype(str).tolist()]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Per-stem registration mask(s) not found: " + "; ".join(missing[:10]))
    return paths

def get_reference_image_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read Step3 reference-image selection settings with defaults."""
    cfg = ((config.get("step3", {}) or {}).get("reference_image", {}) or {}).copy()
    cfg.setdefault("source", "registration")
    cfg.setdefault("wavelength", None)
    cfg.setdefault("column", None)
    return cfg


def _first_existing_path_from_columns(row: pd.Series, columns: list[str], *, what: str) -> Path:
    """Return the first existing file from the candidate metadata columns."""
    tried: list[str] = []
    for col in columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            tried.append(f"{col}={row[col]}")
            p = Path(str(row[col]))
            if p.exists():
                return p
    raise FileNotFoundError(f"{what} not found. Tried: " + "; ".join(tried or columns))


def get_reference_image_path(ref_row: pd.Series, config: dict[str, Any]) -> Path:
    """Choose the anatomical annotation image using Step3 YAML settings.
    
    Parameters
    ----------
    ref_row : pandas.Series
        Step2 reference-stem metadata including image path columns.
    config : dict
        Full project configuration. Reads ``step3.reference_image``.
    
    Returns
    -------
    pathlib.Path
        Existing image selected by ``source``: ``registration`` uses
        Step2's fixed image; ``wavelength`` selects ``ref_<wavelength>_path``;
        ``column`` selects a named metadata path column.
    
    Raises
    ------
    FileNotFoundError
        If no candidate input image exists.
    ValueError
        If the chosen selection mode lacks a required setting or is invalid.
    """
    ref_cfg = get_reference_image_config(config)
    source = str(ref_cfg.get("source", "registration")).strip().lower()

    if source in {"registration", "registration_fixed", "fixed_registration"}:
        return _first_existing_path_from_columns(
            ref_row,
            ["fixed_registration_path_resolved", "fixed_registration_path"],
            what="Step3 reference image from Step2 registration",
        )

    if source in {"wavelength", "ref_wavelength", "reference_wavelength"}:
        wl = ref_cfg.get("wavelength")
        if wl is None or not str(wl).strip():
            raise ValueError("step3.reference_image.wavelength is required when source='wavelength'")
        wl = str(wl)
        return _first_existing_path_from_columns(
            ref_row,
            [f"ref_{wl}_path_resolved", f"ref_{wl}_path"],
            what=f"Step3 reference image for wavelength {wl}",
        )

    if source == "column":
        col = ref_cfg.get("column")
        if col is None or not str(col).strip():
            raise ValueError("step3.reference_image.column is required when source='column'")
        col = str(col)
        return _first_existing_path_from_columns(
            ref_row,
            [f"{col}_resolved", col],
            what=f"Step3 reference image from column {col}",
        )

    raise ValueError(
        "Unsupported step3.reference_image.source. "
        "Use 'registration', 'wavelength', or 'column'."
    )





def annotation_row_is_complete(row: pd.Series, annotation_cfg: dict[str, Any] | None = None) -> bool:
    """Check whether all required landmarks exist in an annotation row."""
    cfg = annotation_cfg or {}
    checks = []
    if bool(cfg.get("require_bregma", True)):
        checks.extend(["bregma_x", "bregma_y"])
    if bool(cfg.get("require_lambda", True)):
        checks.extend(["lambda_x", "lambda_y"])
    if bool(cfg.get("require_frontal_vessel_cross", True)):
        checks.extend(["frontal_vessel_cross_x", "frontal_vessel_cross_y"])
    if bool(cfg.get("require_axis", True)):
        checks.extend(["axis_p1_x", "axis_p1_y", "axis_p2_x", "axis_p2_y", "axis_angle_deg"])
    for col in checks:
        if col not in row.index:
            return False
        v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        if pd.isna(v):
            return False
    return True


def annotated_group_keys(reference_csv: Path, annotation_cfg: dict[str, Any] | None = None) -> set[tuple[str, str]]:
    """Find reference groups with complete anatomical annotations."""
    reference_csv = Path(reference_csv)
    if not reference_csv.exists():
        return set()
    ann = pd.read_csv(reference_csv)
    if ann.empty or "group_id" not in ann.columns or "ref_stem" not in ann.columns:
        return set()
    out: set[tuple[str, str]] = set()
    # Last row per pair wins if a historical file has duplicates.
    for (_, _), rows in ann.groupby([ann["group_id"].astype(str), ann["ref_stem"].astype(str)], sort=False):
        row = rows.iloc[-1]
        if annotation_row_is_complete(row, annotation_cfg):
            out.add((str(row["group_id"]), str(row["ref_stem"])))
    return out

def _to_num(value: Any) -> float | None:
    """Convert a scalar to a finite numeric value when possible."""
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else None


def _truthy_manual(value: Any) -> bool:
    """Check whether a metadata value indicates manual annotation."""
    return "manual" in str(value).strip().lower()


def _find_qc_coordinate_pair(qc_row: pd.Series) -> tuple[str, str] | None:
    """Locate a compatible bregma-coordinate column pair in QC metadata."""
    candidates = [
        ("bregma_cy", "bregma_cx"),
        ("manual_bregma_y", "manual_bregma_x"),
        ("bregma_manual_y", "bregma_manual_x"),
        ("bregma_y", "bregma_x"),
        ("manual_y", "manual_x"),
    ]
    for y_col, x_col in candidates:
        if y_col in qc_row.index and x_col in qc_row.index:
            y = _to_num(qc_row.get(y_col))
            x = _to_num(qc_row.get(x_col))
            if y is not None and x is not None:
                return y_col, x_col
    return None


def _qc_row_is_manual_bregma(qc_row: pd.Series, coord_pair: tuple[str, str]) -> bool:
    """Check whether a QC row represents a manual bregma correction."""
    source_cols = [
        "bregma_source",
        "bregma_mode",
        "bregma_status",
        "source",
        "annotation_source",
    ]
    for col in source_cols:
        if col in qc_row.index and _truthy_manual(qc_row.get(col)):
            return True

    # If the coordinate columns themselves are explicitly manual, accept them.
    joined_names = " ".join(coord_pair).lower()
    if "manual" in joined_names:
        return True

    return False


def update_metadata_bregma_from_frame_qc(
    metadata_df: pd.DataFrame,
    *,
    metadata_path: Path,
    qc_path: Path,
    group_id: str,
    annotator: str,
) -> tuple[pd.DataFrame, int]:
    """Import manually corrected bregma positions from an existing QC CSV.
    
    Parameters
    ----------
    metadata_df : pandas.DataFrame
        Registration metadata table to update.
    metadata_path : pathlib.Path
        Full registration metadata CSV that receives the updates.
    qc_path : pathlib.Path
        Existing frame-level QC CSV containing the manual coordinates.
    group_id : str
        Restrict corrections to this registration group.
    annotator : str
        Operator label recorded with each applied correction.
    
    Returns
    -------
    tuple of (pandas.DataFrame, int)
        Updated metadata and number of applied manual bregma values.
    
    Notes
    -----
    Supports several historical manual-coordinate column names. This
    helper is not automatically invoked by the current ``run_step3`` flow.
    """
    if not qc_path.exists():
        return metadata_df, 0

    qc = pd.read_csv(qc_path)
    if qc.empty or "stem" not in qc.columns:
        return metadata_df, 0

    group_stems = set(metadata_df.loc[metadata_df["group_id"].astype(str).eq(str(group_id)), "stem"].astype(str))
    qc_group = qc[qc["stem"].astype(str).isin(group_stems)].copy()
    if "group_id" in qc_group.columns:
        qc_group = qc_group[qc_group["group_id"].astype(str).eq(str(group_id))].copy()

    if qc_group.empty:
        return metadata_df, 0

    for col in ["bregma_source", "bregma_updated_at", "bregma_annotator"]:
        if col not in metadata_df.columns:
            metadata_df[col] = ""

    n_updated = 0
    now = _dt.datetime.now().isoformat()

    # Last row per stem wins, because frame_qc can accumulate repeated reviews.
    for stem, rows in qc_group.groupby(qc_group["stem"].astype(str), sort=False):
        qc_row = rows.iloc[-1]
        pair = _find_qc_coordinate_pair(qc_row)
        if pair is None:
            continue
        if not _qc_row_is_manual_bregma(qc_row, pair):
            continue

        y_col, x_col = pair
        cy = _to_num(qc_row.get(y_col))
        cx = _to_num(qc_row.get(x_col))
        if cy is None or cx is None:
            continue

        idx = metadata_df["stem"].astype(str).eq(str(stem))
        if not idx.any():
            continue

        metadata_df.loc[idx, "bregma_cy"] = cy
        metadata_df.loc[idx, "bregma_cx"] = cx
        metadata_df.loc[idx, "bregma_found"] = True
        metadata_df.loc[idx, "bregma_source"] = "manual"
        metadata_df.loc[idx, "bregma_updated_at"] = now
        metadata_df.loc[idx, "bregma_annotator"] = annotator
        n_updated += 1

    if n_updated:
        group_mask = metadata_df["group_id"].astype(str).eq(str(group_id))
        ref_stems = metadata_df.loc[group_mask, "ref_stem"].dropna().astype(str).unique().tolist() if "ref_stem" in metadata_df.columns else []
        ref_stem = ref_stems[0] if ref_stems else ""
        if ref_stem:
            ref_row = metadata_df[metadata_df["stem"].astype(str).eq(ref_stem)]
            if not ref_row.empty and str(ref_row.iloc[0].get("bregma_source", "")).lower() == "manual":
                metadata_df.loc[group_mask, "bregma_ref_found"] = True
                metadata_df.loc[group_mask, "bregma_ref_cy"] = ref_row.iloc[0].get("bregma_cy")
                metadata_df.loc[group_mask, "bregma_ref_cx"] = ref_row.iloc[0].get("bregma_cx")
                metadata_df.loc[group_mask, "bregma_ref_source"] = "manual"

        metadata_df = drop_mask_path_metadata_columns(metadata_df)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Updated metadata_registration.csv from Step3 manual bregma: group_id={group_id}, n={n_updated}")

    return drop_mask_path_metadata_columns(metadata_df), n_updated


def update_metadata_bregma_from_manual_updates(
    metadata_df: pd.DataFrame,
    *,
    metadata_path: Path,
    manual_updates: dict[str, tuple[float, float]],
    annotator: str,
    raw_scale: float = 1.0,
    rotation_deg: float = 0.0,
) -> tuple[pd.DataFrame, int]:
    """Persist manual bregma coordinates for selected stems without losing rows.
    
    Parameters
    ----------
    metadata_df : pandas.DataFrame
        Available metadata table; may have been filtered for review.
    metadata_path : pathlib.Path
        Full Step2 metadata CSV, reloaded before updating.
    manual_updates : dict of str to tuple of float
        Mapping ``stem -> (cy, cx)`` in rotated full-resolution pixels.
    annotator : str
        Operator label saved with the correction provenance.
    raw_scale : float, optional
        Project scaling factor used for downstream raw-Zarr coordinates.
    rotation_deg : float, optional
        Project rotation in degrees, recorded in metadata.
    
    Returns
    -------
    tuple of (pandas.DataFrame, int)
        Full metadata table and the number of updated recording rows.
    
    Notes
    -----
    Manual point coordinates are stored in Step2/Step3 full-resolution
    space; scaled versions use the same ``raw.scale`` conversion helper as
    other pipeline stages. Unselected metadata rows are retained.
    """
    if not manual_updates:
        # Return a fresh full table when possible, but do not save anything.
        if metadata_path.exists():
            return pd.read_csv(metadata_path), 0
        return metadata_df, 0

    if metadata_path.exists():
        full_df = pd.read_csv(metadata_path)
    else:
        full_df = metadata_df.copy()

    if "stem" not in full_df.columns:
        raise ValueError("metadata_registration.csv must contain stem")

    for col in ["bregma_source", "bregma_updated_at", "bregma_annotator",
                "step3_image_space", "raw_zarr_image_space"]:
        if col not in full_df.columns:
            full_df[col] = pd.Series(pd.NA, index=full_df.index, dtype="string")
        else:
            full_df[col] = full_df[col].astype("string")
    for col in ["bregma_cy", "bregma_cx"]:
        if col not in full_df.columns:
            full_df[col] = np.nan
    if "bregma_found" not in full_df.columns:
        full_df["bregma_found"] = False

    now = _dt.datetime.now().isoformat()
    n_updated = 0
    for stem, (cy, cx) in manual_updates.items():
        idx = full_df["stem"].astype(str).eq(str(stem))
        if not idx.any():
            print(f"[WARN] manual bregma stem not found in full metadata_registration.csv: {stem}")
            continue
        full_df.loc[idx, "bregma_cy"] = float(cy)
        full_df.loc[idx, "bregma_cx"] = float(cx)
        full_df.loc[idx, "bregma_cy_raw"] = scaled_coordinate_value(cy, raw_scale)
        full_df.loc[idx, "bregma_cx_raw"] = scaled_coordinate_value(cx, raw_scale)
        full_df.loc[idx, "bregma_found"] = True
        full_df.loc[idx, "bregma_source"] = "manual"
        full_df.loc[idx, "bregma_updated_at"] = now
        full_df.loc[idx, "bregma_annotator"] = annotator
        full_df.loc[idx, "project_raw_rotation_deg"] = float(rotation_deg)
        full_df.loc[idx, "project_raw_scale"] = float(raw_scale)
        full_df.loc[idx, "step3_image_space"] = "rotated_fullres"
        full_df.loc[idx, "raw_zarr_image_space"] = "rotated_scaled"
        n_updated += int(idx.sum())

    full_df = drop_mask_path_metadata_columns(full_df)
    if n_updated:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(metadata_path, index=False)
        print(f"Patched full metadata_registration.csv from Step3 manual bregma: n={n_updated}")
    return full_df, n_updated



def _resolve_optional_existing_path(value: Any, path_cfg: dict[str, Any]) -> Path | None:
    """Resolve an optional path only when an existing file is available."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip():
        return None
    try:
        p = resolve_flexible_path(
            value,
            from_drive=path_cfg.get("from_drive"),
            to_drive=path_cfg.get("to_drive"),
            from_root=path_cfg.get("from_root"),
            to_root=path_cfg.get("to_root"),
            must_exist=True,
        )
        return p
    except FileNotFoundError:
        p = Path(str(value))
        return p if p.exists() else None


def _load_bool_mask(path: Path | None, shape: tuple[int, int] | None = None, *, default: bool = False) -> np.ndarray:
    """Load a boolean spatial mask or construct a requested default."""
    if path is None:
        if shape is None:
            raise FileNotFoundError("Cannot create default mask without shape")
        return np.full(shape, bool(default), dtype=bool)
    m = np.load(path).astype(bool)
    if shape is not None and tuple(m.shape) != tuple(shape):
        raise ValueError(f"Mask shape mismatch for {path}: {m.shape} vs {shape}")
    return m


def _disk_mask(shape: tuple[int, int], cy: float, cx: float, diameter_px: float = 25.0) -> np.ndarray:
    """Build a circular mask around a bregma point in full-resolution pixels."""
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.ogrid[:h, :w]
    radius = float(diameter_px) / 2.0
    return ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= (radius ** 2)


def update_metadata_masks_from_step3_manual_edits(
    metadata_df: pd.DataFrame,
    *,
    metadata_path: Path,
    group_id: str,
    manual_brain_mask_paths: dict[str, str],
    manual_bregma_updates: dict[str, tuple[float, float]],
    manual_brain_reverted_stems: list[str] | None = None,
    masks_root: Path,
    path_cfg: dict[str, Any],
    raw_scale: float,
    rotation_deg: float,
    bregma_diameter_px: float = 25.0,
) -> tuple[pd.DataFrame, int]:
    """Persist selected manual masks and recalculate associated mask metadata.
    
    Parameters
    ----------
    metadata_df : pandas.DataFrame
        In-memory metadata; the complete CSV is reloaded before updating.
    metadata_path : pathlib.Path
        Canonical ``metadata_registration.csv`` to update.
    group_id : str
        Identifier of the group being reviewed.
    manual_brain_mask_paths : dict of str to str
        Manual brain-mask file paths keyed by recording ``stem``.
    manual_bregma_updates : dict of str to tuple of float
        Corrected ``(cy, cx)`` positions in rotated full-resolution pixels.
    manual_brain_reverted_stems : list of str or None, optional
        Stems for which an existing manual brain mask should be removed.
    masks_root : pathlib.Path
        Root directory containing canonical per-stem Step2 masks.
    path_cfg : dict
        Global path-remapping parameters.
    raw_scale : float
        Scale used downstream when building the raw Zarr image geometry.
    rotation_deg : float
        Project raw-image rotation, in degrees.
    bregma_diameter_px : float, optional
        Diameter, in full-resolution pixels, of a circular manual bregma mask.
    
    Returns
    -------
    tuple of (pandas.DataFrame, int)
        Full updated metadata table and the number of touched rows.
    
    Notes
    -----
    Brain and bregma masks are saved under ``masks/<stem>/``; pixel masks
    are boolean spatial arrays with shape ``(y, x)``. When supplied, the
    reverted-stem list removes stale canonical ``brain_manual.npy`` files.
    The caller must actually forward that list for reversion to take effect.
    """
    manual_brain_reverted_stems = manual_brain_reverted_stems or []
    touched_stems = ({str(s) for s in manual_brain_mask_paths}
                     | {str(s) for s in manual_bregma_updates}
                     | {str(s) for s in manual_brain_reverted_stems})
    if not touched_stems:
        if metadata_path.exists():
            return drop_mask_path_metadata_columns(pd.read_csv(metadata_path)), 0
        return drop_mask_path_metadata_columns(metadata_df.copy()), 0

    if metadata_path.exists():
        full_df = pd.read_csv(metadata_path)
    else:
        full_df = metadata_df.copy()
    if "stem" not in full_df.columns:
        raise ValueError("metadata_registration.csv must contain stem")

    now = _dt.datetime.now().isoformat()
    n_updated = 0

    for col in [
        "brain_source", "background_source", "bregma_mask_source",
        "bregma_mask_updated_at", "step3_masks_updated_at", "mask_file_logic",
        "step3_image_space", "mask_image_space", "raw_zarr_image_space",
    ]:
        if col not in full_df.columns:
            full_df[col] = pd.Series(pd.NA, index=full_df.index, dtype="string")
        else:
            full_df[col] = full_df[col].astype("string")
    for col in [
        "bregma_mask_diameter_px", "brain_area_px", "brain_signal_area_px",
        "background_area_px", "bregma_mask_area_px",
    ]:
        full_df[col] = pd.to_numeric(full_df[col], errors="coerce") if col in full_df.columns else np.nan

    for stem in sorted(touched_stems):
        idx = full_df["stem"].astype(str).eq(str(stem))
        if not idx.any():
            print(f"[WARN] Step3 manual mask stem not found in metadata_registration.csv: {stem}")
            continue

        row = full_df.loc[idx].iloc[0]
        paths = canonical_mask_paths(masks_root, stem)
        paths["brain_auto"].parent.mkdir(parents=True, exist_ok=True)

        # Save manual brain mask, if edited in Step3; remove it if user reverted to auto.
        if stem in manual_brain_reverted_stems and paths["brain_manual"].exists():
            paths["brain_manual"].unlink()

        if stem in manual_brain_mask_paths:
            manual_brain_path = Path(manual_brain_mask_paths[stem])
            if not manual_brain_path.exists():
                raise FileNotFoundError(f"Manual brain mask disappeared: {manual_brain_path}")
            source_brain = _load_bool_mask(manual_brain_path)
            np.save(paths["brain_manual"], source_brain.astype(np.uint8))
            full_df.loc[idx, "brain_source"] = "manual"
        elif paths["brain_manual"].exists():
            source_brain = _load_bool_mask(paths["brain_manual"])
            full_df.loc[idx, "brain_source"] = "manual"
        elif paths["brain_auto"].exists():
            source_brain = _load_bool_mask(paths["brain_auto"])
            full_df.loc[idx, "brain_source"] = "auto"
        else:
            raise FileNotFoundError(f"No brain_auto.npy or brain_manual.npy found for stem={stem}: {paths['brain_auto'].parent}")

        image_shape = tuple(source_brain.shape)

        bad = _load_bool_mask(paths["bad_auto"], image_shape, default=False) if paths["bad_auto"].exists() else np.zeros(image_shape, dtype=bool)

        # Save manual bregma as canonical bregma_manual.npy if user placed/corrected it.
        if stem in manual_bregma_updates:
            cy, cx = manual_bregma_updates[stem]
            bregma_mask = _disk_mask(image_shape, cy, cx, diameter_px=float(bregma_diameter_px))
            np.save(paths["bregma_manual"], bregma_mask.astype(np.uint8))
            full_df.loc[idx, "bregma_mask_source"] = "manual_circle"
            full_df.loc[idx, "bregma_mask_diameter_px"] = float(bregma_diameter_px)
            full_df.loc[idx, "bregma_mask_updated_at"] = now
        elif paths["bregma_manual"].exists():
            bregma_mask = _load_bool_mask(paths["bregma_manual"], image_shape, default=False)
            full_df.loc[idx, "bregma_mask_source"] = "manual_circle"
        elif paths["bregma_auto"].exists():
            bregma_mask = _load_bool_mask(paths["bregma_auto"], image_shape, default=False)
            full_df.loc[idx, "bregma_mask_source"] = "auto"
        else:
            bregma_mask = np.zeros(image_shape, dtype=bool)
            full_df.loc[idx, "bregma_mask_source"] = "missing"

        brain_signal = source_brain.astype(bool) & (~bad.astype(bool)) & (~bregma_mask.astype(bool))
        background = (~source_brain.astype(bool)) & (~bad.astype(bool)) & (~bregma_mask.astype(bool))

        full_df.loc[idx, "background_source"] = "computed_in_step4_from_brain_bad_bregma"
        full_df.loc[idx, "step3_masks_updated_at"] = now
        full_df.loc[idx, "brain_area_px"] = int(source_brain.sum())
        full_df.loc[idx, "brain_signal_area_px"] = int(brain_signal.sum())
        full_df.loc[idx, "background_area_px"] = int(background.sum())
        full_df.loc[idx, "bregma_mask_area_px"] = int(bregma_mask.sum())
        full_df.loc[idx, "mask_file_logic"] = "canonical_by_stem"
        full_df.loc[idx, "project_raw_rotation_deg"] = float(rotation_deg)
        full_df.loc[idx, "project_raw_scale"] = float(raw_scale)
        full_df.loc[idx, "step3_image_space"] = "rotated_fullres"
        full_df.loc[idx, "mask_image_space"] = "rotated_fullres"
        full_df.loc[idx, "raw_zarr_image_space"] = "rotated_scaled"
        n_updated += int(idx.sum())

    full_df = drop_mask_path_metadata_columns(full_df)
    if n_updated:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(metadata_path, index=False)
        print(f"Saved Step3 manual masks as canonical files and removed mask paths from metadata: n={n_updated}")
    return full_df, n_updated

def get_manual_bregma_groups(df: pd.DataFrame) -> set[str]:
    """Identify groups containing at least one manually marked bregma."""
    if "bregma_source" not in df.columns:
        return set()
    manual = df[df["bregma_source"].astype(str).str.strip().str.lower().eq("manual")]
    if "group_id" not in manual.columns:
        return set()
    return {str(v) for v in manual["group_id"].dropna().astype(str).tolist() if str(v).strip()}



# ============================================================
# Step3 registration QC prompt
# ============================================================

REGISTRATION_QC_COLUMNS = [
    "group_id", "ref_stem", "n_records",
    "registration_qc", "needs_manual_registration",
    "answer", "notes", "annotator", "updated_at",
    "review_ok", "annotation_ok", "error",
]


def registration_qc_config(step3_cfg: dict[str, Any]) -> dict[str, Any]:
    """Read group-level registration QC settings from the Step3 block."""
    cfg = dict((step3_cfg.get("registration_qc", {}) or {}))
    cfg.setdefault("ask_after_group", False)
    cfg.setdefault("csv_name", "registration_qc_groups.csv")
    cfg.setdefault("bad_values", ["bad", "no", "n"])
    return cfg


def registration_qc_csv_path(step3_root_path: Path, step3_cfg: dict[str, Any]) -> Path:
    """Construct the configured path for group-level registration QC."""
    cfg = registration_qc_config(step3_cfg)
    return step3_root_path / str(cfg.get("csv_name", "registration_qc_groups.csv"))


def load_registration_qc_table(path: Path) -> pd.DataFrame:
    """Read or initialize a group-level registration QC table."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=REGISTRATION_QC_COLUMNS)
    for col in REGISTRATION_QC_COLUMNS:
        if col not in df.columns:
            if col in {"n_records"}:
                df[col] = 0
            elif col in {"needs_manual_registration", "review_ok", "annotation_ok"}:
                df[col] = False
            else:
                df[col] = ""
    ordered = REGISTRATION_QC_COLUMNS + [c for c in df.columns if c not in REGISTRATION_QC_COLUMNS]
    return df[ordered]


def append_registration_qc_row(path: Path, row: dict[str, Any]) -> None:
    """Append a registration QC decision to the persistent CSV table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_registration_qc_table(path)
    row_full = {col: row.get(col, "") for col in REGISTRATION_QC_COLUMNS}
    df = pd.concat([df, pd.DataFrame([row_full])], ignore_index=True)
    df.to_csv(path, index=False)


def _normalise_console_answer(text: str, default: str = "ok") -> str:
    """Normalize English and legacy Russian registration QC responses."""
    ans = str(text or "").strip().lower()
    if not ans:
        return default
    if ans in {"y", "yes", "ok", "good", "д", "да", "+", "1"}:
        return "ok"
    if ans in {"n", "no", "bad", "плохо", "нет", "н", "-", "0"}:
        return "bad"
    if ans in {"s", "skip", "пропустить"}:
        return "skip"
    if ans in {"q", "quit", "exit", "stop", "стоп", "выход"}:
        return "quit"
    return ans


def ask_registration_qc_after_group(
    *,
    group_id: str,
    ref_stem: str,
    n_records: int,
    review_ok: bool,
    annotation_ok: bool,
    error: str,
    annotator: str,
    qc_csv: Path,
    default_ok: bool = True,
) -> str:
    """Request and persist the operator's group-level registration assessment.
    
    Parameters
    ----------
    group_id, ref_stem : str
        Group and reference recording identifiers.
    n_records : int
        Number of recordings evaluated in the group.
    review_ok, annotation_ok : bool
        Whether the corresponding preceding interactive steps succeeded.
    error : str
        Existing error text; skips interactive input when nonempty.
    annotator : str
        Operator label recorded in the QC row.
    qc_csv : pathlib.Path
        Destination CSV for group-level registration QC decisions.
    default_ok : bool, optional
        Select the default response shown in the console.
    
    Returns
    -------
    str
        ``'continue'`` or ``'stop'`` to control the outer review loop.
    
    Notes
    -----
    The decision is appended to ``registration_qc_groups.csv`` (or the
    configured filename). It does not modify Step2 registration transforms.
    Legacy Russian keyboard responses remain accepted.
    """
    if error:
        answer = "error"
        registration_qc = "error"
        needs_manual = False
        notes = str(error)
    else:
        default_label = "ok" if default_ok else "bad"
        print()
        print("Registration QC")
        print(f"group_id: {group_id}")
        print(f"ref_stem: {ref_stem}")
        print("Is registration within this group acceptable?")
        print("  [Enter/y/yes] = accept registration")
        print("  [n/no/bad] = mark the group for manual registration")
        print("  [s/skip] = do not save a QC decision")
        print("  [q/quit] = stop Step3 after saving the current decision")
        raw = input(f"Answer [{default_label}]: ")
        answer = _normalise_console_answer(raw, default=default_label)
        if answer == "skip":
            print("Registration QC skipped for this group.")
            return "continue"
        if answer == "quit":
            registration_qc = "quit"
            needs_manual = False
            notes = "user requested stop after group"
        elif answer == "bad":
            registration_qc = "bad"
            needs_manual = True
            notes = input("Comment/reason (press Enter to skip): ").strip()
        elif answer == "ok":
            registration_qc = "ok"
            needs_manual = False
            notes = ""
        else:
            registration_qc = str(answer)
            needs_manual = str(answer).lower() in {"bad", "no", "n", "плохо", "нет"}
            notes = "unrecognized answer saved as-is"

    append_registration_qc_row(qc_csv, {
        "group_id": str(group_id),
        "ref_stem": str(ref_stem),
        "n_records": int(n_records),
        "registration_qc": registration_qc,
        "needs_manual_registration": bool(needs_manual),
        "answer": answer,
        "notes": notes,
        "annotator": annotator,
        "updated_at": _dt.datetime.now().isoformat(),
        "review_ok": bool(review_ok),
        "annotation_ok": bool(annotation_ok),
        "error": str(error or ""),
    })
    print(f"Registration QC saved: {qc_csv}")
    if registration_qc == "bad":
        print("Group marked for manual registration widget.")
    return "stop" if answer == "quit" else "continue"


def latest_bad_registration_groups(qc_csv: str | Path) -> list[str]:
    """Select groups whose latest registration decision requests manual review."""
    path = Path(qc_csv)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "group_id" not in df.columns:
        return []
    if "updated_at" in df.columns:
        df = df.sort_values("updated_at")
    last = df.groupby(df["group_id"].astype(str), sort=False).tail(1)
    if "needs_manual_registration" in last.columns:
        mask = last["needs_manual_registration"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
    else:
        mask = last.get("registration_qc", pd.Series(index=last.index, dtype=str)).astype(str).str.lower().isin({"bad", "no", "n"})
    return last.loc[mask, "group_id"].astype(str).tolist()

# ============================================================
# Main
# ============================================================

def run_step3(
    config_path: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    binmasks_dir: str | Path | None = None,
    step3_root: str | Path | None = None,
    qc_csv: str | Path | None = None,
    reference_csv: str | Path | None = None,
    annotator: str | None = None,
    include_bad: bool | None = None,
    groups: list[str] | None = None,
    manual_groups_only: bool = False,
    only_unannotated: bool | None = None,
    auto_bregma_size: int | None = None,
    manual_bregma_size: int | None = None,
    ask_registration_qc: bool | None = None,
) -> pd.DataFrame:
    """Review selected Step2 groups, save manual edits, and annotate references.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        Project YAML used to locate Step2 outputs and read ``step3`` settings.
    metadata_csv : str or pathlib.Path or None, optional
        Override for the Step2 ``metadata_registration.csv`` input.
    binmasks_dir : str or pathlib.Path or None, optional
        Override for the canonical per-stem mask directory.
    step3_root : str or pathlib.Path or None, optional
        Override for the Step3 output directory.
    qc_csv : str or pathlib.Path or None, optional
        Optional path reserved for frame-level QC metadata.
    reference_csv : str or pathlib.Path or None, optional
        Override for ``reference_annotations.csv``.
    annotator : str or None, optional
        Operator label; defaults to ``step3.annotator``.
    include_bad : bool or None, optional
        Whether to include records marked bad in Step2; falls back to
        ``step3.review.include_bad``.
    groups : list of str or None, optional
        Optional subset of ``group_id`` values.
    manual_groups_only : bool, optional
        Process only groups with manual bregma provenance in Step2 metadata.
    only_unannotated : bool or None, optional
        Use annotation completeness as a group-selection criterion.
    auto_bregma_size, manual_bregma_size : int or None, optional
        Napari point sizes; defaults are read from ``step3.review``.
    ask_registration_qc : bool or None, optional
        Enable a console registration QC prompt after each group; defaults to
        ``step3.registration_qc.ask_after_group``.
    
    Returns
    -------
    pandas.DataFrame
        Per-group review and annotation summary, also written to the
        configured Step3 ``run_summary.csv``.
    
    Notes
    -----
    Step3 operates in rotated, full-resolution Step2 image coordinates.
    The configured ``raw.scale`` applies to downstream raw Zarr geometry,
    not the Napari editing grid. Manual bregma and brain masks are saved
    against the original recording ``stem``. Individual group errors are
    reported in the summary rather than always aborting the whole run.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)

    step3_cfg = config.get("step3", {}) or {}
    review_cfg = step3_cfg.get("review", {}) or {}
    annotation_cfg = step3_cfg.get("annotation", {}) or {}
    path_cfg = get_path_resolving_config(config, "step3")
    rotation_deg = raw_rotation_degrees(config)
    raw_scale = raw_scale_factor(config)

    metadata_path = Path(metadata_csv) if metadata_csv else paths["metadata_registration_csv"]
    step3_root_path = Path(step3_root) if step3_root else paths["step3_root"]
    qc_path = Path(qc_csv) if qc_csv else step3_root_path / step3_cfg.get("frame_qc_name", "frame_qc.csv")
    reference_path = Path(reference_csv) if reference_csv else step3_root_path / step3_cfg.get("reference_annotations_name", "reference_annotations.csv")
    manual_masks_root = step3_root_path / step3_cfg.get("manual_masks_folder", "manual_brain_masks")
    run_summary_csv = step3_root_path / step3_cfg.get("run_summary_name", "run_summary.csv")
    reg_qc_cfg = registration_qc_config(step3_cfg)
    if ask_registration_qc is None:
        ask_registration_qc = bool(reg_qc_cfg.get("ask_after_group", False))
    reg_qc_csv = registration_qc_csv_path(step3_root_path, step3_cfg)
    binmasks_path = Path(binmasks_dir) if binmasks_dir else paths["masks_root"]

    step3_root_path.mkdir(parents=True, exist_ok=True)
    manual_masks_root.mkdir(parents=True, exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_registration.csv not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise RuntimeError(f"Metadata table is empty: {metadata_path}")
    excluded_stems = set(config.get("excluded_stems") or [])
    if excluded_stems:
        skipped = df[df["stem"].astype(str).isin(excluded_stems)]
        for stem in skipped["stem"].astype(str):
            print(f"[EXCLUDED] Step3 stem={stem}: excluded_by_config")
        df = df[~df["stem"].astype(str).isin(excluded_stems)].copy()
        if df.empty:
            print("[WARN] Step3: no records remain after excluded_stems.")
            summary_df = pd.DataFrame(columns=["group_id", "n_records", "review_ok", "annotation_ok", "error"])
            run_summary_csv.parent.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(run_summary_csv, index=False)
            return summary_df
    if "group_id" not in df.columns:
        raise ValueError("metadata_registration.csv must contain group_id for Step3 v2_14")

    filters = ((step3_cfg.get("debug", {}) or {}).get("filters", {}) or {})
    if filters:
        df = apply_metadata_filters_df(df, filters, config)

    if include_bad is None:
        include_bad = bool(review_cfg.get("include_bad", False))
    if not include_bad and "is_bad" in df.columns:
        df = df[~normalize_bool_series(df["is_bad"])].copy()

    if manual_groups_only:
        manual_groups = get_manual_bregma_groups(df)
        if not manual_groups:
            raise RuntimeError("No bregma_source=manual groups found in metadata_registration.csv")
        df = df[df["group_id"].astype(str).isin(manual_groups)].copy()

    if groups:
        wanted = {str(g) for g in groups}
        df = df[df["group_id"].astype(str).isin(wanted)].copy()

    if df.empty:
        raise RuntimeError("No Step3 rows after filtering")

    df = resolve_metadata_paths(df, path_cfg)

    if only_unannotated is None:
        only_unannotated = bool(annotation_cfg.get("only_unannotated", False))
    if only_unannotated:
        done = annotated_group_keys(reference_path, annotation_cfg)
        keep_groups: list[str] = []
        for gid, g in df.groupby("group_id", sort=True):
            g = reorder_reference_first(g, config)
            ref_row = choose_reference_row(g)
            ref_stem = str(ref_row["ref_stem"]) if "ref_stem" in ref_row and pd.notna(ref_row["ref_stem"]) and str(ref_row["ref_stem"]).strip() else str(ref_row["stem"])
            if (str(gid), str(ref_stem)) not in done:
                keep_groups.append(str(gid))
        df = df[df["group_id"].astype(str).isin(set(keep_groups))].copy()
        if df.empty:
            print("All reference groups are already annotated according to current step3.annotation requirements.")
            summary_df = pd.DataFrame(columns=["group_id", "n_records", "review_ok", "annotation_ok", "error"])
            summary_df.to_csv(run_summary_csv, index=False)
            return summary_df

    if annotator is None:
        annotator = step3_cfg.get("annotator", "user")
    if auto_bregma_size is None:
        auto_bregma_size = int(review_cfg.get("auto_bregma_size", 6))
    if manual_bregma_size is None:
        manual_bregma_size = int(review_cfg.get("manual_bregma_size", 8))

    summary: list[dict[str, Any]] = []
    missing_inputs: list[dict[str, str]] = []

    for group_id, df_group in df.groupby("group_id", sort=True):
        df_group = reorder_reference_first(df_group, config)
        review_ok = False
        annotation_ok = False
        error = ""
        manual_bregma_updates = {}
        manual_brain_mask_paths = {}
        manual_brain_reverted_stems = []
        n_masks_updated = 0
        ref_stem_for_qc = ""

        print()
        print("=" * 80)
        print(f"STEP3 group_id: {group_id}")
        print(f"n_records: {len(df_group)}")

        try:
            mask_paths = choose_mask_paths(df_group, paths["masks_root"])
            review_result = review_group_df(
                df_group=df_group,
                group_id=str(group_id),
                mask_path=mask_paths,
                manual_masks_dir=manual_masks_root / str(group_id),
                binmasks_dir=binmasks_path,
                annotator=annotator,
                auto_bregma_size=auto_bregma_size,
                manual_bregma_size=manual_bregma_size,
            )
            manual_bregma_updates = review_result.get("manual_bregma_updates", {}) or {}
            manual_brain_mask_paths = review_result.get("manual_brain_mask_paths", {}) or {}
            manual_brain_reverted_stems = review_result.get("manual_brain_reverted_stems", []) or []
            review_ok = True

            df, n_bregma_updated = update_metadata_bregma_from_manual_updates(
                df,
                metadata_path=metadata_path,
                manual_updates=manual_bregma_updates,
                annotator=annotator,
                raw_scale=raw_scale,
                rotation_deg=rotation_deg,
            )

            manual_update_cfg = (step3_cfg.get("manual_mask_updates", {}) or {})
            if bool(manual_update_cfg.get("enabled", True)):
                df, n_masks_updated = update_metadata_masks_from_step3_manual_edits(
                    df,
                    metadata_path=metadata_path,
                    group_id=str(group_id),
                    manual_brain_mask_paths={str(k): str(v) for k, v in manual_brain_mask_paths.items()},
                    manual_bregma_updates={str(k): v for k, v in manual_bregma_updates.items()},
                    masks_root=binmasks_path,
                    path_cfg=path_cfg,
                    raw_scale=raw_scale,
                    rotation_deg=rotation_deg,
                    bregma_diameter_px=float(manual_update_cfg.get("bregma_diameter_px", 25)),
                )
            else:
                n_masks_updated = 0

            if n_bregma_updated or n_masks_updated:
                # Refresh this group's view so reference annotation sees the latest metadata.
                df_group = df[df["group_id"].astype(str).eq(str(group_id))].copy()
                df_group = resolve_metadata_paths(df_group, path_cfg)
                df_group = reorder_reference_first(df_group, config)

            ref_row = choose_reference_row(df_group)
            ref_stem = str(ref_row["ref_stem"]) if "ref_stem" in ref_row and pd.notna(ref_row["ref_stem"]) and str(ref_row["ref_stem"]).strip() else str(ref_row["stem"])
            ref_stem_for_qc = ref_stem
            ref_image_path = get_reference_image_path(ref_row, config)

            annotate_reference_group(
                group_id=str(group_id),
                ref_stem=ref_stem,
                ref_image_path=ref_image_path,
                annotations_csv=reference_path,
                annotator=annotator,
                group_keys_json=str(ref_row.get("group_keys_json", "")),
                annotation_cfg=annotation_cfg,
            )
            annotation_ok = True

        except Exception as e:
            error = repr(e)
            if isinstance(e, (FileNotFoundError, ValueError)) and "brain_auto.npy" in str(e):
                missing_inputs.append({"group_id": str(group_id), "reason": str(e)})
                print(f"[WARN] Skipping Step3 QC group={group_id}: {e}")
            else:
                print(f"[ERROR] group_id={group_id}: {error}")

        if ask_registration_qc:
            if not ref_stem_for_qc:
                try:
                    ref_row_qc = choose_reference_row(df_group)
                    ref_stem_for_qc = str(ref_row_qc["ref_stem"]) if "ref_stem" in ref_row_qc and pd.notna(ref_row_qc["ref_stem"]) and str(ref_row_qc["ref_stem"]).strip() else str(ref_row_qc["stem"])
                except Exception:
                    ref_stem_for_qc = ""
            action = ask_registration_qc_after_group(
                group_id=str(group_id),
                ref_stem=ref_stem_for_qc,
                n_records=len(df_group),
                review_ok=review_ok,
                annotation_ok=annotation_ok,
                error=error,
                annotator=str(annotator),
                qc_csv=reg_qc_csv,
                default_ok=bool(reg_qc_cfg.get("default_ok", True)),
            )
            if action == "stop":
                print("Stopping Step3 by user request after registration QC prompt.")
                summary.append({
                    "group_id": group_id,
                    "n_records": len(df_group),
                    "review_ok": review_ok,
                    "annotation_ok": annotation_ok,
                    "project_raw_rotation_deg": rotation_deg,
                    "project_raw_scale": raw_scale,
                    "step3_image_space": "rotated_fullres",
                    "raw_zarr_image_space": "rotated_scaled",
                    "registration_qc_csv": str(reg_qc_csv),
                    "error": error,
                })
                break

        summary.append({
            "group_id": group_id,
            "n_records": len(df_group),
            "review_ok": review_ok,
            "annotation_ok": annotation_ok,
            "project_raw_rotation_deg": rotation_deg,
            "project_raw_scale": raw_scale,
            "step3_image_space": "rotated_fullres",
            "raw_zarr_image_space": "rotated_scaled",
            "registration_qc_csv": str(reg_qc_csv) if ask_registration_qc else "",
            "error": error,
        })

    missing_path = step3_root_path / "step3_missing_inputs.csv"
    pd.DataFrame(missing_inputs, columns=["group_id", "reason"]).to_csv(missing_path, index=False)
    if missing_inputs:
        print(f"[WARN] {len(missing_inputs)} group(s) skipped because of missing/invalid brain masks: {missing_path}")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(run_summary_csv, index=False)
    print()
    print("=" * 80)
    print("Step3 finished")
    print("Run summary:", run_summary_csv)
    return summary_df


def main() -> None:
    """Parse command-line options and execute Step3."""
    parser = argparse.ArgumentParser(description="Step 3 v2_14: napari QC + reference annotation + optional registration QC prompt")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--metadata", default=None, help="Override metadata_registration.csv")
    parser.add_argument("--binmasks-dir", default=None, help="Root dir with canonical <stem>/brain_auto.npy etc.; defaults to Step2 masks root")
    parser.add_argument("--step3-root", default=None, help="Override output root for Step3")
    parser.add_argument("--qc-csv", default=None, help="Override frame_qc.csv path")
    parser.add_argument("--reference-csv", default=None, help="Override reference_annotations.csv path")
    parser.add_argument("--annotator", default=None)
    parser.add_argument("--include-bad", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group_id list to process")
    parser.add_argument("--manual-groups-only", action="store_true", help="Run only groups where metadata_registration.csv has bregma_source=manual")
    parser.add_argument("--only-unannotated", action=argparse.BooleanOptionalAction, default=None, help="Run only groups missing a complete row in reference_annotations.csv")
    parser.add_argument("--auto-bregma-size", type=int, default=None)
    parser.add_argument("--manual-bregma-size", type=int, default=None)
    qc_group = parser.add_mutually_exclusive_group()
    qc_group.add_argument("--ask-registration-qc", action="store_true", help="Ask after each group whether registration looked OK and save bad groups for the manual-registration widget")
    qc_group.add_argument("--no-ask-registration-qc", action="store_true", help="Disable the after-group registration QC prompt even if config enables it")
    args = parser.parse_args()

    run_step3(
        config_path=args.config,
        metadata_csv=args.metadata,
        binmasks_dir=args.binmasks_dir,
        step3_root=args.step3_root,
        qc_csv=args.qc_csv,
        reference_csv=args.reference_csv,
        annotator=args.annotator,
        include_bad=args.include_bad,
        groups=args.groups,
        manual_groups_only=args.manual_groups_only,
        only_unannotated=args.only_unannotated,
        auto_bregma_size=args.auto_bregma_size,
        manual_bregma_size=args.manual_bregma_size,
        ask_registration_qc=True if args.ask_registration_qc else (False if args.no_ask_registration_qc else None),
    )


if __name__ == "__main__":
    main()
