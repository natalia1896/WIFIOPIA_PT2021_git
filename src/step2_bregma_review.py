from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd

from common import (
    load_config,
    create_or_load_project,
    build_standard_paths,
    get_path_resolving_config,
    resolve_flexible_path,
    raw_rotation_degrees,
    raw_scale_factor,
    scaled_coordinate_value,
)


BREGMA_MISSING_REASONS = {"bregma_empty", "ref_bregma_empty"}


def _resolve_path(value: Any, path_cfg: dict[str, Any], must_exist: bool = True) -> Path:
    """Resolve a stored image path using project-wide remapping rules.

    Parameters
    ----------
    value : Any
        Path value from metadata.
    path_cfg : dict
        Global ``path_resolving`` configuration.
    must_exist : bool, default=True
        Whether the resulting path must be present.

    Returns
    -------
    pathlib.Path
        Resolved image path.
    """
    p = resolve_flexible_path(
        value,
        from_drive=path_cfg.get("from_drive"),
        to_drive=path_cfg.get("to_drive"),
        from_root=path_cfg.get("from_root"),
        to_root=path_cfg.get("to_root"),
        must_exist=must_exist,
    )
    if p is None:
        raise FileNotFoundError(f"Empty path: {value!r}")
    return p


def _bregma_source_wavelengths(config: dict[str, Any]) -> list[str]:
    """List wavelengths configured for the bregma mask source.

    Parameters
    ----------
    config : dict
        Full YAML configuration containing ``step2.mask_sources.bregma``.

    Returns
    -------
    list[str]
        Enabled wavelength names, in first-occurrence order.

    Notes
    -----
        A role may contain one source or several ``sources`` entries.
    """
    step2_cfg = config.get("step2", {}) or {}
    mask_sources = step2_cfg.get("mask_sources", {}) or {}
    bregma_cfg = mask_sources.get("bregma")
    if not isinstance(bregma_cfg, dict) or not bregma_cfg.get("enabled", True):
        raise ValueError("step2.mask_sources.bregma must be configured")

    raw_sources = bregma_cfg.get("sources")
    if raw_sources is None:
        raw_sources = [bregma_cfg]
    elif isinstance(raw_sources, dict):
        raw_sources = [raw_sources]

    waves: list[str] = []
    for src in raw_sources or []:
        if not isinstance(src, dict) or not src.get("enabled", True):
            continue
        wl = src.get("wavelength")
        if wl is not None and str(wl).strip():
            waves.append(str(wl))
    if not waves:
        raise ValueError("step2.mask_sources.bregma has no wavelength")
    return list(dict.fromkeys(waves))


def _choose_image_path(row: pd.Series, path_cfg: dict[str, Any], config: dict[str, Any]) -> Path:
    """Find the stem-specific image used for manual bregma correction.

    Parameters
    ----------
    row : pandas.Series
        Step2 metadata row for exactly one recording.
    path_cfg : dict
        Global path-remapping settings.
    config : dict
        Full project configuration.

    Returns
    -------
    pathlib.Path
        First available image for a configured bregma source wavelength,
        falling back to registration images as implemented.

    Notes
    -----
        A moving stem does not use another stem's fixed reference image.
    """
    cols: list[str] = []
    for wl in _bregma_source_wavelengths(config):
        cols.extend([f"ref_{wl}_path", f"bmp_{wl}"])

    # Never show the group's fixed reference image for a moving stem.
    cols.append("moving_registration_path")
    if str(row.get("stem", "")) == str(row.get("ref_stem", "")):
        cols.append("fixed_registration_path")

    for col in dict.fromkeys(cols):
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            try:
                return _resolve_path(row[col], path_cfg, must_exist=True)
            except FileNotFoundError:
                continue
    raise FileNotFoundError(f"No image path found for stem={row.get('stem', '')}")


def _bool_missing(value: Any) -> bool:
    """Interpret false-like scalar strings as a missing bregma detection."""
    text = str(value).strip().lower()
    return text in {"false", "0", "no", "nan", "", "none"}


def _needs_bregma_review(row: pd.Series) -> bool:
    """Check whether a Step2 record needs manual bregma review.

    Parameters
    ----------
    row : pandas.Series
        One registration metadata record.

    Returns
    -------
    bool
        True for missing bregma coordinates or relevant QC reasons.

    Notes
    -----
        ``ref_bregma_empty`` selects the reference record, and reviewing it
        subsequently opens its entire group.
    """
    reason_text = str(row.get("bad_reasons", ""))
    reasons = {r.strip() for r in reason_text.split(",") if r.strip()}

    if "bregma_empty" in reasons:
        return True

    # ref_bregma_empty should select the group for review, but not every row as
    # an individual target. Because this script annotates the whole group once a
    # group is selected, it is enough to return True for the reference row.
    if "ref_bregma_empty" in reasons:
        stem = str(row.get("stem", ""))
        ref_stem = str(row.get("ref_stem", ""))
        if stem == ref_stem:
            return True

    if _bool_missing(row.get("bregma_found", "")):
        return True

    if pd.isna(row.get("bregma_cy", np.nan)) or pd.isna(row.get("bregma_cx", np.nan)):
        return True

    return False


def _normalise_image(img: np.ndarray) -> np.ndarray:
    """Convert an image into a two-dimensional frame for Napari stacking.

    Parameters
    ----------
    img : numpy.ndarray
        Image in grayscale, RGB/RGBA or supported stacked format.

    Returns
    -------
    numpy.ndarray
        Two-dimensional image with shape ``(y, x)``.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # RGB/RGBA -> first 3 channels are display data; convert to luminance-like mean.
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1).astype(arr.dtype, copy=False)
        # If a tiny leading dimension sneaks in, use first plane.
        return arr[0]
    raise ValueError(f"Unsupported image shape for bregma review: {arr.shape}")


def _load_group_stack(df_group: pd.DataFrame, path_cfg: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, list[pd.Series], list[Path]]:
    """Load equally sized images for every recording in one group.

    Parameters
    ----------
    df_group : pandas.DataFrame
        Metadata rows belonging to one ``group_id``.
    path_cfg : dict
        Global path-remapping settings.
    config : dict
        Full project configuration.

    Returns
    -------
    stack : numpy.ndarray
        Image stack with axes ``(stem_index, y, x)``.
    rows : list[pandas.Series]
        Metadata rows in the same order as the first stack dimension.
    paths : list[pathlib.Path]
        Input images in the same order.

    Raises
    ------
    ValueError
        If images cannot be stacked because their shapes differ.
    """
    rows: list[pd.Series] = []
    paths: list[Path] = []
    images: list[np.ndarray] = []

    for _, row in df_group.iterrows():
        p = _choose_image_path(row, path_cfg, config)
        img = _normalise_image(iio.imread(p))
        rows.append(row)
        paths.append(p)
        images.append(img)

    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        details = "; ".join(f"{r.get('stem', '')}: {im.shape}" for r, im in zip(rows, images))
        raise ValueError(f"Images in one group have different shapes and cannot be stacked: {details}")

    stack = np.stack(images, axis=0)
    return stack, rows, paths


def _annotate_group_bregma(df_group: pd.DataFrame, path_cfg: dict[str, Any], config: dict[str, Any], annotator: str) -> dict[str, tuple[float, float]]:
    """Review and save one bregma coordinate for every stem in Napari.

    Parameters
    ----------
    df_group : pandas.DataFrame
        All recordings for a selected group.
    path_cfg : dict
        Global path-remapping settings.
    config : dict
        Full project configuration.
    annotator : str
        Name recorded for the correction.

    Returns
    -------
    dict[str, tuple[float, float]]
        Stem-specific bregma coordinates in ``(cy, cx)`` full-resolution
        rotated Step2 pixels.

    Notes
    -----
        The third-dimensional Napari stack index identifies the stem, not time.
        The last point placed on each z-index is used. The function requires a
        point for each stem before returning.
    """
    group_id = str(df_group["group_id"].iloc[0])
    df_group = df_group.reset_index(drop=True)
    # Stack order maps directly to the stem identity shown in the viewer.
    stack, rows, image_paths = _load_group_stack(df_group, path_cfg, config)

    viewer = napari.Viewer(title=f"Manual bregma group: {group_id}")
    viewer.add_image(stack, name="group_images_z_stem", axis_labels=("stem_index", "y", "x"))

    points_data: list[list[float]] = []
    point_stems: list[str] = []
    for z, row in enumerate(rows):
        cy = pd.to_numeric(pd.Series([row.get("bregma_cy", np.nan)]), errors="coerce").iloc[0]
        cx = pd.to_numeric(pd.Series([row.get("bregma_cx", np.nan)]), errors="coerce").iloc[0]
        if np.isfinite(cy) and np.isfinite(cx):
            points_data.append([float(z), float(cy), float(cx)])
            point_stems.append(str(row.get("stem", "")))

    layer = viewer.add_points(
        np.asarray(points_data, dtype=float) if points_data else np.empty((0, 3), dtype=float),
        name="bregma_manual_one_point_per_slice",
        ndim=3,
        size=9,
        face_color="red",
    )

    print()
    print("Manual bregma review for one full group")
    print("group_id:", group_id)
    print("Place exactly one red point on bregma on every z-slice/stem, then close napari.")
    print("z-index -> stem -> image")
    for z, (row, p) in enumerate(zip(rows, image_paths)):
        print(f"  {z}: {row.get('stem', '')} -> {p}")
    print("annotator:", annotator)
    print()

    # Collect user-edited points only after the interactive session closes.
    napari.run()

    if len(layer.data) == 0:
        raise RuntimeError(f"No bregma points placed for group_id={group_id}")

    # Use the last point placed per z-index if the user corrected a point by adding a new one.
    by_stem: dict[str, tuple[float, float]] = {}
    by_z: dict[int, tuple[float, float]] = {}
    for point in np.asarray(layer.data, dtype=float):
        z = int(round(float(point[0])))
        if z < 0 or z >= len(rows):
            raise RuntimeError(f"Point has invalid z-index {point[0]} for group_id={group_id}")
        by_z[z] = (float(point[1]), float(point[2]))

    missing_z = [z for z in range(len(rows)) if z not in by_z]
    if missing_z:
        missing = ", ".join(f"z={z} stem={rows[z].get('stem', '')}" for z in missing_z)
        raise RuntimeError(f"Bregma is missing for slices in group_id={group_id}: {missing}")

    for z, row in enumerate(rows):
        by_stem[str(row.get("stem", ""))] = by_z[z]

    return by_stem


def select_groups_for_review(
    df: pd.DataFrame,
    *,
    groups: list[str] | None = None,
    stems: list[str] | None = None,
    only_missing: bool = True,
) -> list[str]:
    """Choose registration groups for manual bregma inspection.

    Parameters
    ----------
    df : pandas.DataFrame
        Step2 metadata table.
    groups : list[str] or None, optional
        Optional explicit group-ID selection.
    stems : list[str] or None, optional
        Optional stems whose entire groups should be selected.
    only_missing : bool, default=True
        If true, retain only groups with detected bregma issues.

    Returns
    -------
    list[str]
        Sorted selected group IDs.
    """
    work = df.copy()
    if groups:
        wanted = {str(g) for g in groups}
        work = work[work["group_id"].astype(str).isin(wanted)].copy()
    if stems:
        wanted_stems = {str(s) for s in stems}
        selected_groups = set(work.loc[work["stem"].astype(str).isin(wanted_stems), "group_id"].astype(str))
        work = work[work["group_id"].astype(str).isin(selected_groups)].copy()
    if only_missing:
        group_has_missing = work.groupby("group_id").apply(lambda g: bool(g.apply(_needs_bregma_review, axis=1).any()))
        selected = [str(g) for g, ok in group_has_missing.items() if ok]
    else:
        selected = sorted(work["group_id"].dropna().astype(str).unique().tolist())
    return sorted(selected)


def update_metadata_bregma(
    metadata_csv: str | Path | None,
    *,
    config_path: str | Path,
    groups: list[str] | None = None,
    stems: list[str] | None = None,
    only_missing: bool = True,
    annotator: str = "user",
) -> pd.DataFrame:
    """Interactively correct bregma coordinates in Step2 metadata.

    Parameters
    ----------
    metadata_csv : str or pathlib.Path or None
        Override path to ``metadata_registration.csv``; ``None`` uses the project path.
    config_path : str or pathlib.Path
        Complete YAML configuration file.
    groups, stems : list[str] or None, optional
        Optional group or stem selection.
    only_missing : bool, default=True
        Review only groups with bregma issues unless disabled.
    annotator : str, default='user'
        Correction provenance stored in Step2 metadata.

    Returns
    -------
    pandas.DataFrame
        Metadata table with manual bregma positions and provenance.

    Notes
    -----
        The Napari corrections are recorded in rotated full-resolution Step2
        ``(cy, cx)`` pixels; corresponding ``*_raw`` columns use ``raw.scale``.
        Metadata are saved after each group. Registration must be rerun separately
        for groups with changed bregma coordinates.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)
    path_cfg = get_path_resolving_config(config, "step2")
    rotation_deg = raw_rotation_degrees(config)
    raw_scale = raw_scale_factor(config)

    metadata_path = Path(metadata_csv) if metadata_csv else paths["metadata_registration_csv"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_registration.csv not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if df.empty:
        raise RuntimeError(f"Metadata table is empty: {metadata_path}")
    excluded_stems = set(config.get("excluded_stems") or [])
    if excluded_stems:
        skipped = df[df["stem"].astype(str).isin(excluded_stems)]
        for stem in skipped["stem"].astype(str):
            print(f"[EXCLUDED] Step2 bregma stem={stem}: excluded_by_config")
        df = df[~df["stem"].astype(str).isin(excluded_stems)].copy()
        if df.empty:
            print("[WARN] No Step2 bregma records remain after excluded_stems.")
            return df
    if "group_id" not in df.columns:
        if "mouse" in df.columns:
            df["group_id"] = df["mouse"].astype(str)
        else:
            raise ValueError("metadata_registration.csv must contain group_id")

    for col in ["bregma_source", "bregma_updated_at", "bregma_annotator"]:
        if col not in df.columns:
            df[col] = ""

    target_groups = select_groups_for_review(df, groups=groups, stems=stems, only_missing=only_missing)
    if not target_groups:
        print("No groups selected for manual bregma review.")
        return df

    for group_id in target_groups:
        group_mask = df["group_id"].astype(str).eq(str(group_id))
        df_group = df.loc[group_mask].copy()
        df_group = df_group.sort_values([c for c in ["is_reference", "ref_stem", "stem"] if c in df_group.columns], ascending=[False, True, True][:len([c for c in ["is_reference", "ref_stem", "stem"] if c in df_group.columns])]).reset_index(drop=True)

        # Save one bregma location per stem before any registration rerun.
        manual_points = _annotate_group_bregma(df_group, path_cfg, config, annotator=annotator)
        now = _dt.datetime.now().isoformat()

        for stem, (cy, cx) in manual_points.items():
            idx = df["stem"].astype(str).eq(stem)
            df.loc[idx, "bregma_cy"] = cy
            df.loc[idx, "bregma_cx"] = cx
            df.loc[idx, "bregma_cy_raw"] = scaled_coordinate_value(cy, raw_scale)
            df.loc[idx, "bregma_cx_raw"] = scaled_coordinate_value(cx, raw_scale)
            df.loc[idx, "bregma_found"] = True
            df.loc[idx, "bregma_source"] = "manual"
            df.loc[idx, "bregma_updated_at"] = now
            df.loc[idx, "bregma_annotator"] = annotator
            df.loc[idx, "project_raw_rotation_deg"] = rotation_deg
            df.loc[idx, "project_raw_scale"] = raw_scale
            df.loc[idx, "step2_image_space"] = "rotated_fullres"
            df.loc[idx, "raw_zarr_image_space"] = "rotated_scaled"
            print(f"Saved manual bregma for stem={stem}: cy={cy:.2f}, cx={cx:.2f}")

        # Keep reference-bregma columns consistent for every row in the group.
        ref_stem = str(df.loc[group_mask, "ref_stem"].dropna().astype(str).iloc[0]) if "ref_stem" in df.columns and not df.loc[group_mask, "ref_stem"].dropna().empty else ""
        if ref_stem and ref_stem in manual_points:
            cy, cx = manual_points[ref_stem]
            df.loc[group_mask, "bregma_ref_found"] = True
            df.loc[group_mask, "bregma_ref_cy"] = cy
            df.loc[group_mask, "bregma_ref_cx"] = cx
            df.loc[group_mask, "bregma_ref_cy_raw"] = scaled_coordinate_value(cy, raw_scale)
            df.loc[group_mask, "bregma_ref_cx_raw"] = scaled_coordinate_value(cx, raw_scale)
            df.loc[group_mask, "bregma_ref_source"] = "manual"

        # Save after every group so completed manual work is not lost if a later group fails.
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(metadata_path, index=False)

    manual_groups = sorted(df.loc[df["bregma_source"].astype(str).str.lower().eq("manual"), "group_id"].dropna().astype(str).unique())
    print()
    print("Updated metadata:", metadata_path)
    print("Groups with manual bregma:", ", ".join(manual_groups) if manual_groups else "none")
    print("Next: re-run Step2 with --manual-groups-only, then run Step3 for those groups.")
    return df


def main() -> None:
    """Parse CLI options for the standalone Napari bregma-review tool."""
    parser = argparse.ArgumentParser(description="Manual group-level bregma review for Step2 v2_3 metadata_registration.csv")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--metadata", default=None, help="metadata_registration.csv; defaults to Step2 output")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group_id list. Every stem in each group is reviewed.")
    parser.add_argument("--stems", nargs="*", default=None, help="Optional stem list. Their whole groups are reviewed.")
    parser.add_argument("--all-selected", action="store_true", help="Review all selected groups, not only groups with missing bregma")
    parser.add_argument("--annotator", default="user")
    args = parser.parse_args()

    update_metadata_bregma(
        metadata_csv=args.metadata,
        config_path=args.config,
        groups=args.groups,
        stems=args.stems,
        only_missing=not args.all_selected,
        annotator=args.annotator,
    )


if __name__ == "__main__":
    main()
