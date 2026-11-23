from __future__ import annotations

import os
import json
import shutil
import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import imageio.v3 as iio
from tqdm.auto import tqdm

from common import (
    infer_stem_from_config,
    list_raw_images,
    get_stem_keys,
    parse_variant_value,
    get_path_resolving_config,
    resolve_flexible_path,
    apply_raw_rotation,
    raw_rotation_degrees,
    raw_scale_factor,
)

# ============================================================
# RAW RECORDING DISCOVERY
# ============================================================

def count_matching_files(
    folder: Path,
    suffixes: Sequence[str],
) -> int:
    """Count files in one directory with the requested extensions.

    Parameters
    ----------
    folder : pathlib.Path
        Directory to inspect.
    suffixes : sequence of str
        Extensions compared case-insensitively.

    Returns
    -------
    int
        Number of matching regular files."""
    suffixes_lower = {s.lower() for s in suffixes}

    n = 0
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in suffixes_lower:
            n += 1

    return n


def discover_recording_folders_recursive(
    raw_root: str | Path,
    *,
    suffixes: Sequence[str],
    min_files: int = 100,
    max_depth: int | None = None,
    skip_folder_names: Sequence[str] = (),
    show_progress: bool = True,
) -> list[Path]:
    """Find recording directories by a minimum number of raw image files.

    Parameters
    ----------
    raw_root : str or pathlib.Path
        Root directory traversed recursively.
    suffixes : sequence of str
        Filename extensions used to count raw images.
    min_files : int, optional
        Minimum number of matching files in a folder; default is 100.
    max_depth : int or None, optional
        Maximum traversal depth relative to ``raw_root``; None is unlimited.
    skip_folder_names : sequence of str, optional
        Directory names excluded from traversal.
    show_progress : bool, optional
        Display an indeterminate folder-scanning progress bar.

    Returns
    -------
    list[pathlib.Path]
        Sorted recording directories containing at least ``min_files`` images.

    Raises
    ------
    FileNotFoundError
        If the configured raw root does not exist.

    Notes
    -----
    When a directory qualifies, its subdirectories are not traversed. The
    search therefore treats qualifying recording folders as leaves; it does
    not inspect their nested folders for further recordings."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    suffixes_lower = {s.lower() for s in suffixes}
    skip_set = set(skip_folder_names or [])
    candidates: list[Path] = []

    root_depth = len(raw_root.parts)

    walker = os.walk(raw_root)

    if show_progress:
        walker = tqdm(
            walker,
            desc="Searching recording folders",
            unit="folder",
        )

    for dirpath, dirnames, filenames in walker:
        current = Path(dirpath)

        if skip_set:
            dirnames[:] = [d for d in dirnames if d not in skip_set]

        if max_depth is not None:
            depth = len(current.parts) - root_depth
            if depth > max_depth:
                dirnames[:] = []
                continue

        n_bmp = sum(
            1
            for name in filenames
            if Path(name).suffix.lower() in suffixes_lower
        )
        
        if show_progress:
            try:
                current_display = str(current.relative_to(raw_root))
            except ValueError:
                current_display = str(current)
        
            max_path_len = 80
            if len(current_display) > max_path_len:
                current_display = "..." + current_display[-max_path_len:]
        
            walker.set_postfix(
                found=len(candidates),
                bmp=n_bmp,
                current=current_display,
            )

        if n_bmp >= min_files:
            candidates.append(current)

            # Treat a qualifying recording folder as a leaf directory.
            # Do not descend into technical subfolders nested within a recording.
            dirnames[:] = []

    return sorted(candidates)


def discover_recording_folders_folder_structure(
    raw_root: str | Path,
    *,
    skip_folder_names: Sequence[str] = ("m#",),
) -> list[Path]:
    """Enumerate directories in the legacy four-level hierarchy.

    Parameters
    ----------
    raw_root : str or pathlib.Path
        Root of ``exp_day/mouse_id/state/protocol`` folders.
    skip_folder_names : sequence of str, optional
        Directory names to skip at each traversed level.

    Returns
    -------
    list[pathlib.Path]
        Protocol directories encountered under the four-level structure.

    Notes
    -----
    This mode does not validate image-file counts before returning a folder."""
    raw_root = Path(raw_root)
    skip_set = set(skip_folder_names or [])

    out: list[Path] = []

    for exp_day in sorted(raw_root.iterdir()):
        if not exp_day.is_dir() or exp_day.name in skip_set:
            continue

        for mouse_dir in sorted(exp_day.iterdir()):
            if not mouse_dir.is_dir() or mouse_dir.name in skip_set:
                continue

            for state_dir in sorted(mouse_dir.iterdir()):
                if not state_dir.is_dir() or state_dir.name in skip_set:
                    continue

                for protocol_dir in sorted(state_dir.iterdir()):
                    if not protocol_dir.is_dir() or protocol_dir.name in skip_set:
                        continue

                    out.append(protocol_dir)

    return out


def discover_recording_folders(config: dict[str, Any]) -> list[Path]:
    """Select the configured strategy for discovering raw recording directories.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration containing ``discovery`` and global ``path_resolving``.

    Returns
    -------
    list[pathlib.Path]
        Recording folders under the resolved ``discovery.raw_root``.

    Raises
    ------
    ValueError
        If ``discovery.raw_root`` is absent or discovery mode is unknown.

    Notes
    -----
    ``recursive_folders`` respects ``min_files``; ``folder_structure``
    traverses the legacy four-level hierarchy without checking file counts."""
    discovery_cfg = config.get("discovery", {})
    mode = discovery_cfg.get("mode", "recursive_folders")

    raw_root = discovery_cfg.get("raw_root")
    if raw_root is None:
        raise ValueError("discovery.raw_root is required")

    path_cfg = get_path_resolving_config(config, "step1")
    raw_root_resolved = resolve_flexible_path(
        raw_root,
        from_drive=path_cfg.get("from_drive"),
        to_drive=path_cfg.get("to_drive"),
        from_root=path_cfg.get("from_root"),
        to_root=path_cfg.get("to_root"),
        must_exist=True,
    )

    suffixes = discovery_cfg.get("suffixes", [".bmp"])
    skip_folder_names = discovery_cfg.get("skip_folder_names", [])
    show_progress = bool(discovery_cfg.get("show_progress", True))

    if mode == "recursive_folders":
        return discover_recording_folders_recursive(
            raw_root=raw_root_resolved,
            suffixes=suffixes,
            min_files=int(discovery_cfg.get("min_files", 100)),
            max_depth=discovery_cfg.get("max_depth"),
            skip_folder_names=skip_folder_names,
            show_progress=show_progress,
        )

    if mode == "folder_structure":
        return discover_recording_folders_folder_structure(
            raw_root=raw_root_resolved,
            skip_folder_names=skip_folder_names,
        )

    raise ValueError(
        "discovery.mode must be 'recursive_folders' or 'folder_structure'"
    )

# ============================================================
# FILTERS
# ============================================================

def _as_optional_set(x: Any) -> set[str] | None:
    """Normalize a scalar or sequence of filter values to strings.

    Parameters
    ----------
    x : Any
        Allowed values or None.

    Returns
    -------
    set[str] or None
        String values as a set, or None when no filter is specified."""
    if x is None:
        return None
    if isinstance(x, (str, int, float)):
        return {str(x)}
    return {str(v) for v in x}


def metadata_passes_filters(meta: dict[str, str], config: dict[str, Any]) -> bool:
    """Check whether recording metadata satisfies discovery filters.

    Parameters
    ----------
    meta : dict[str, str]
        Recording metadata extracted from its directory.
    config : dict[str, Any]
        Configuration with ``discovery.metadata_filters`` and optional
        ``variant_keys`` repeat-parsing rules.

    Returns
    -------
    bool
        True when every configured non-None filter admits the recording.

    Notes
    -----
    Values are compared as strings. For enabled variant fields, both
    candidate and allowed values are compared after parsing off their
    trailing repeat suffixes; this function does not select which repeat
    will ultimately be retained."""
    discovery_cfg = config.get("discovery", {}) or {}
    filters = discovery_cfg.get("metadata_filters", {}) or {}

    variants = config.get("variant_keys", {}) or {}
    for key, allowed_raw in filters.items():
        allowed = _as_optional_set(allowed_raw)
        if allowed is None:
            continue
        key = str(key)
        variant_cfg = variants.get(key, {}) or {}
        if variant_cfg.get("enabled", False):
            separators = variant_cfg.get("separators", ["", "-", "_"])
            actual = parse_variant_value(meta.get(key, ""), separators)[0]
            allowed = {parse_variant_value(v, separators)[0] for v in allowed}
        else:
            actual = str(meta.get(key, ""))
        if actual not in allowed:
            return False

    return True

# ============================================================
# MANIFEST
# ============================================================

def build_recording_row(
    img_dir: Path,
    stem: str,
    meta: dict[str, str],
    replacements: dict[str, dict[str, str]],
    paths: dict[str, Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Create one manifest record with paths and metadata provenance.

    Parameters
    ----------
    img_dir : pathlib.Path
        Raw recording directory.
    stem : str
        Recording identifier formed from ``metadata.stem_keys``.
    meta : dict[str, str]
        Metadata extracted from the recording path.
    replacements : dict[str, dict[str, str]]
        Audit of normalized metadata values.
    paths : dict[str, pathlib.Path]
        Project paths from ``build_standard_paths``.
    config : dict[str, Any]
        Project metadata and raw geometry settings.

    Returns
    -------
    dict[str, Any]
        Manifest row with raw, mask, and processed-output paths, metadata,
        timestamps, and geometry provenance.

    Notes
    -----
    Step1 reference images use ``rotated_fullres`` geometry. The row also
    records the intended later raw-Zarr geometry as ``rotated_scaled``."""
    raw_zarr_path = paths["raw_zarr_root"] / f"{stem}.zarr"
    processed_zarr_path = paths["processed_signals_root"] / f"{stem}_processed.zarr"
    masks_dir = paths["masks_root"] / stem
    stem_keys = get_stem_keys(config)

    row = {
        "stem": stem,
        "stem_keys_json": json.dumps(stem_keys, ensure_ascii=False),
        "metadata_json": json.dumps(meta, ensure_ascii=False),
        "replacements_json": json.dumps(replacements, ensure_ascii=False),
        "img_dir": str(img_dir),
        "raw_zarr_path": str(raw_zarr_path),
        "processed_zarr_path": str(processed_zarr_path),
        "masks_dir": str(masks_dir),
        "created_at": datetime.datetime.now().isoformat(),
        "project_raw_rotation_deg": raw_rotation_degrees(config),
        "project_raw_scale": raw_scale_factor(config),
        "step1_image_space": "rotated_fullres",
        "raw_zarr_image_space": "rotated_scaled",
    }

    # Preserve every metadata column as its own CSV column.
    for key, value in meta.items():
        if key not in row:
            row[key] = value

    return row

def build_recording_manifest(
    config: dict[str, Any],
    paths: dict[str, Path],
) -> pd.DataFrame:
    """Build and audit the manifest of discovered recording directories.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration defining discovery, metadata, and filtering.
    paths : dict[str, pathlib.Path]
        Project paths, including the future manifest CSV location.

    Returns
    -------
    pandas.DataFrame
        One row per discovered recording that passes metadata filters.

    Raises
    ------
    RuntimeError
        If any discovered directory raises an error, including duplicate
        recording stems; details are written to ``manifest_errors.csv``.

    Notes
    -----
    Every discovered directory is recorded in ``manifest_preflight.csv``
    with an inclusion, exclusion, or error status. The function does not
    save the accepted main manifest; call ``save_manifest`` separately."""
    recording_dirs = discover_recording_folders(config)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    seen_stems: dict[str, str] = {}

    for img_dir in recording_dirs:
        stem = ""
        duplicate = False
        try:
            stem, meta, replacements = infer_stem_from_config(img_dir, config)
            if not metadata_passes_filters(meta, config):
                checked.append({"img_dir": str(img_dir), "stem": stem,
                                "status": "excluded_metadata_filter",
                                "metadata_source": meta.get("stem_metadata_source", "")})
                continue
            if stem in seen_stems:
                duplicate = True
                raise ValueError(f"Duplicate stem {stem}: {seen_stems[stem]} and {img_dir}")
            seen_stems[stem] = str(img_dir)
            rows.append(build_recording_row(
                img_dir=img_dir, stem=stem, meta=meta,
                replacements=replacements, paths=paths, config=config,
            ))
            checked.append({"img_dir": str(img_dir), "stem": stem,
                            "status": "included",
                            "metadata_source": meta.get("stem_metadata_source", "")})
        except Exception as exc:
            errors.append({"img_dir": str(img_dir), "stem": stem, "error": repr(exc)})
            checked.append({"img_dir": str(img_dir), "stem": stem,
                            "status": "duplicate_stem" if duplicate else "error", "metadata_source": ""})

    report_path = paths["manifest_path"].with_name("manifest_preflight.csv")
    pd.DataFrame(checked, columns=["img_dir", "stem", "status", "metadata_source"]).to_csv(report_path, index=False)
    error_path = paths["manifest_path"].with_name("manifest_errors.csv")
    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False)
        raise RuntimeError(f"Step1 manifest preflight failed: {len(errors)} error(s); see {error_path}")
    if error_path.exists():
        error_path.unlink()
    return pd.DataFrame(rows)


def save_manifest(
    df: pd.DataFrame,
    manifest_path: str | Path,
    overwrite: bool = True,
) -> None:
    """Write recording manifest rows to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Manifest table.
    manifest_path : str or pathlib.Path
        Output CSV file path.
    overwrite : bool, optional
        Whether an existing manifest may be replaced; default is True.

    Raises
    ------
    FileExistsError
        If the manifest exists and ``overwrite`` is False."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {manifest_path}")

    df.to_csv(manifest_path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Read a previously saved recording manifest.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing manifest CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded recording metadata table.

    Raises
    ------
    FileNotFoundError
        If the specified CSV does not exist."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    return pd.read_csv(path)

# ============================================================
# Reference image selection across wavelengths
# ============================================================

def save_reference_images_for_manifest(
    manifest_df,
    config,
    paths,
    show_progress: bool = True,
):
    """Select per-wavelength reference frames and record Ilastik file paths.

    Parameters
    ----------
    manifest_df : pandas.DataFrame
        Step1 manifest containing at least ``stem`` and ``img_dir``.
    config : dict
        Configuration containing ``raw.wave_pattern``, ``raw.rotation``,
        ``raw.ref_image_idx_per_wavelength``, and ``step1.reference_selection``.
    paths : dict
        Project path mapping containing ``reference_images_root``.
    show_progress : bool, optional
        Whether to show per-recording progress information.

    Returns
    -------
    pandas.DataFrame
        Copy of manifest rows with per-wavelength ``ref_<wavelength>_path``,
        ``prob_<wavelength>_path``, image provenance, and probability-file
        existence indicators.

    Notes
    -----
    Raw image paths are ordered by ``raw.file_regex`` and ``raw.sort_key``.
    Each wavelength selects the positional subsequence specified by
    ``raw.wave_pattern`` as ``files[start::step]``. The selected frame index
    is capped at the final available frame. Only ``raw.rotation`` is applied
    before exporting a full-resolution BMP; ``raw.scale`` is not applied.
    Probability H5 paths are recorded for subsequent Ilastik exports but
    this function does not create the H5 files. Existing reference BMPs
    can be retained when ``step1.reference_selection.overwrite`` is False."""
    raw_cfg = config["raw"]
    step1_cfg = config["step1"]["reference_selection"]

    output_root = paths["reference_images_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    suffixes = raw_cfg.get("suffixes", [".bmp"])
    file_regex = raw_cfg.get("file_regex")
    sort_key = raw_cfg.get("sort_key", "index")

    wave_pattern = raw_cfg["wave_pattern"]
    ref_idx_by_wave = raw_cfg.get("ref_image_idx_per_wavelength", {})

    filename_template = step1_cfg.get("filename_template", "{stem}.bmp")
    probability_template = step1_cfg.get(
        "probability_filename_template",
        "{stem}_Probabilities.h5",
    )

    overwrite = bool(step1_cfg.get("overwrite", True))
    rotation_deg = raw_rotation_degrees(config)
    raw_scale = raw_scale_factor(config)

    updated_rows = []

    iterator = manifest_df.iterrows()
    if show_progress:
        iterator = tqdm(
            iterator,
            total=len(manifest_df),
            desc="Saving Ilastik reference images",
            unit="recording",
        )

    for _, row in iterator:
        row = row.copy()

        stem = row["stem"]

        path_cfg = get_path_resolving_config(config, "step1")
        img_dir = resolve_flexible_path(
            row["img_dir"],
            from_drive=path_cfg.get("from_drive"),
            to_drive=path_cfg.get("to_drive"),
            from_root=path_cfg.get("from_root"),
            to_root=path_cfg.get("to_root"),
            must_exist=True,
        )

        if show_progress:
            iterator.set_postfix(stem=str(stem)[:40])

        files = list_raw_images(
            img_dir,
            suffixes=suffixes,
            file_regex=file_regex,
            sort_key=sort_key,
        )

        for wavelength, pattern in wave_pattern.items():
            start, step = pattern
            wave_files = files[start::step]

            wave_dir = output_root / str(wavelength)
            wave_dir.mkdir(parents=True, exist_ok=True)

            ref_filename = filename_template.format(
                stem=stem,
                wavelength=wavelength,
            )
            prob_filename = probability_template.format(
                stem=stem,
                wavelength=wavelength,
            )

            ref_path = wave_dir / ref_filename
            prob_path = wave_dir / prob_filename

            row[f"ref_{wavelength}_path"] = str(ref_path)
            row[f"prob_{wavelength}_path"] = str(prob_path)

            if not wave_files:
                row[f"ref_{wavelength}_status"] = "no_files"
                row[f"prob_{wavelength}_exists"] = prob_path.exists()
                continue

            ref_idx = int(ref_idx_by_wave.get(str(wavelength), 0))
            ref_idx = min(ref_idx, len(wave_files) - 1)

            src = wave_files[ref_idx]

            if ref_path.exists() and not overwrite:
                row[f"ref_{wavelength}_status"] = "exists"
            else:
                if abs(rotation_deg) < 1e-12:
                    # No geometry change: keep the original file byte-for-byte.
                    shutil.copy2(src, ref_path)
                else:
                    # Project geometry starts here: Ilastik references are rotated,
                    # but deliberately not scaled. Masks/annotations remain full-res.
                    img = iio.imread(src)
                    img_rot = apply_raw_rotation(img, config)
                    iio.imwrite(ref_path, img_rot)
                row[f"ref_{wavelength}_status"] = "saved"

            row[f"ref_{wavelength}_source_path"] = str(src)
            row[f"ref_{wavelength}_index"] = ref_idx
            row[f"ref_{wavelength}_rotation_deg"] = rotation_deg
            row[f"ref_{wavelength}_scale_applied"] = 1.0
            row[f"ref_{wavelength}_image_space"] = "rotated_fullres"
            row[f"project_raw_scale"] = raw_scale
            row[f"prob_{wavelength}_exists"] = prob_path.exists()

        updated_rows.append(row)

    return pd.DataFrame(updated_rows)