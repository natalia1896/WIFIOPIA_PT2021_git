"""Early manual hemorrhage-only review for every selected Step2 stem.

Authoritative cohort: out/step2_masks/metadata_registration.csv (no new variant
selection, no condition/day/processed-signal filtering).

Only mask: out/step2_masks/masks/<stem>/hemorrhage_manual.npy, in native
rotated_fullres coordinates of that stem's own raw BMP frames.

Display images are the final frame of EACH wavelength in the recording,
not the early Step1 reference frame; Step1 reference is used only for a
geometry/shape check and is never shown as a review image.

This module never edits Step2/Step3 files or Step6b/lesion/FC outputs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import tempfile
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd

from common import (
    load_config,
    create_or_load_project,
    build_standard_paths,
    get_path_resolving_config,
    resolve_flexible_path,
    stem_masks_dir,
    apply_raw_rotation,
    list_raw_images,
    parse_raw_filename,
)

MASK_FILENAME = "hemorrhage_manual.npy"
IMAGE_SPACE = "rotated_fullres"
STATUSES = {"reviewed_present", "reviewed_absent"}
LOG_COLUMNS = [
    "stem", "group_id", "mouse_id", "exp_day", "state", "protocol",
    "protocol_base", "protocol_repeat", "review_status", "area_px",
    "hemorrhage_mask_path", "image_shape", "image_space", "last_raw_505_path",
    "last_raw_656_path", "review_revision", "updated_at", "annotator",
]


def _text(value: Any) -> str:
    """Return a stripped string, or an empty string for missing values.

    Parameters
    ----------
    value : Any
        Scalar metadata value, possibly missing.

    Returns
    -------
    str
        Normalized text for metadata comparisons.
    """
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _bool(value: Any) -> bool:
    """Recognize a small set of true-like metadata values.

    Parameters
    ----------
    value : Any
        Value to interpret as a boolean.

    Returns
    -------
    bool
        ``True`` for ``true``, ``1``, ``yes``, or ``y`` (case-insensitive).
    """
    return _text(value).lower() in {"true", "1", "yes", "y"}


def review_paths(config: dict, paths: dict) -> tuple[Path, Path]:
    """Return the directory and CSV path for hemorrhage review records.

    Parameters
    ----------
    config : dict
        Project configuration; accepted for the common helper interface but not used.
    paths : dict
        Project paths returned by ``build_standard_paths``.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Review directory and ``hemorrhage_mask_review.csv`` path.

    Notes
    -----
    The review log is separate from the canonical per-stem Step2 mask folders.
    """
    root = Path(paths["out_root"]) / "hemorrhage_mask_review"
    return root, root / "hemorrhage_mask_review.csv"


def mask_path(paths: dict, stem: str) -> Path:
    """Return the canonical hemorrhage mask path for one recording.

    Parameters
    ----------
    paths : dict
        Project paths used to locate the Step2 mask root.
    stem : str
        Step2 recording identifier; path separators and traversal components are rejected.

    Returns
    -------
    pathlib.Path
        Path to ``masks/<stem>/hemorrhage_manual.npy``.

    Raises
    ------
    ValueError
        If the stem is unsafe for use as a directory component.
    """
    if not stem or stem in {".", ".."} or "/" in stem or "\\" in stem:
        raise ValueError(f"Unsafe/invalid stem from Step2 metadata: {stem!r}")
    return stem_masks_dir(paths, stem) / MASK_FILENAME


def _resolved_image_path(row: pd.Series, wavelength: str, config: dict, *, required: bool) -> Path | None:
    """Resolve a Step1 reference BMP belonging to the requested stem.

    Parameters
    ----------
    row : pandas.Series
        Recording metadata, including ``stem`` and reference image paths.
    wavelength : str
        Channel identifier used in ``ref_<wavelength>_path``.
    config : dict
        Project configuration, including ``path_resolving`` and the Step1 filename template.
    required : bool
        Whether to reject a missing reference BMP.

    Returns
    -------
    pathlib.Path or None
        Verified reference BMP path, or ``None`` when optional and unavailable.

    Notes
    -----
    This image is used only to validate the rotated full-resolution geometry.
    It is never substituted for the final raw frame in the Napari review.
    """
    path_cfg = get_path_resolving_config(config)
    kwargs = {key: path_cfg.get(key) for key in ("from_drive", "to_drive", "from_root", "to_root")}
    for col in (f"ref_{wavelength}_path", f"bmp_{wavelength}"):
        value = _text(row.get(col, ""))
        if not value:
            continue
        try:
            resolved = resolve_flexible_path(value, **kwargs, must_exist=True)
            template = (config.get("step1", {}).get("reference_selection", {}) or {}).get(
                "filename_template", "{stem}.bmp")
            expected = template.format(stem=_text(row["stem"]), wavelength=wavelength)
            if resolved.name.lower() != expected.lower():
                raise ValueError(
                    f"stem={row['stem']}: wrong reference BMP identity: {resolved.name!r}; "
                    f"expected {expected!r} for wavelength {wavelength}"
                )
            return resolved
        except FileNotFoundError:
            continue
    if required:
        raise FileNotFoundError(
            f"stem={row['stem']}: missing own ref_{wavelength}_path/BMP in Step2 metadata. "
            "Do not substitute fixed_registration_path or another stem's image."
        )
    return None


def _read_gray(path: Path) -> np.ndarray:
    """Read a two-dimensional grayscale BMP without resizing it.

    Parameters
    ----------
    path : pathlib.Path
        Path to the image.

    Returns
    -------
    numpy.ndarray
        Image with shape ``(y, x)``; original pixel dtype is preserved.

    Raises
    ------
    ValueError
        If the loaded image is not two-dimensional.
    """
    arr = np.asarray(iio.imread(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected original 2D grayscale BMP, got {arr.shape}: {path}")
    return arr


def _last_raw_paths_by_wavelength(row: pd.Series, config: dict) -> dict[str, Path]:
    """Find the last recorded raw BMP for each supported review wavelength.

    Parameters
    ----------
    row : pandas.Series
        Step2 metadata containing the original image directory ``img_dir``.
    config : dict
        Project configuration containing ``raw.file_regex``, ``raw.sort_key``,
        ``raw.suffixes``, and ``raw.wave_pattern``.

    Returns
    -------
    dict[str, pathlib.Path]
        Last matching file for each available supported wavelength (``505`` and
        optionally ``656``).

    Raises
    ------
    FileNotFoundError
        If the raw images or the required 505-nm frame are unavailable.
    ValueError
        If filename indices cannot safely define the configured channel ordering,
        or if channel splitting parameters are invalid.

    Notes
    -----
    Raw frames are ordered using the same selection rules as Step1. When index
    sorting and a filename regex are configured, nonconsecutive indices are
    rejected to avoid silently shifting alternating wavelength assignments.
    The current implementation explicitly requires wavelength ``505`` and only
    processes ``505``/``656``; it is not wavelength-agnostic.
    """
    path_cfg = get_path_resolving_config(config)
    kwargs = {key: path_cfg.get(key) for key in ("from_drive", "to_drive", "from_root", "to_root")}
    raw_dir_value = _text(row.get("img_dir"))
    if not raw_dir_value:
        raise ValueError(f"stem={row['stem']}: Step2 metadata has no img_dir")
    raw_dir = resolve_flexible_path(raw_dir_value, **kwargs, must_exist=True)
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"stem={row['stem']}: raw img_dir is not a folder: {raw_dir}")

    raw_cfg = config.get("raw", {}) or {}
    file_regex = raw_cfg.get("file_regex")
    files = list_raw_images(
        raw_dir,
        suffixes=tuple(raw_cfg.get("suffixes", [".bmp"])),
        file_regex=file_regex,
        sort_key=raw_cfg.get("sort_key", "index"),
    )
    if not files:
        raise FileNotFoundError(f"stem={row['stem']}: no matching raw BMPs in {raw_dir}")

    # Step1 allocates wavelengths by alternating positions in this sorted list.
    # Interior missing indices or duplicates invalidate that interpretation.
    if file_regex and raw_cfg.get("sort_key", "index") == "index":
        index_texts = [parse_raw_filename(p, file_regex).get("index") for p in files]
        if all(index_texts) and all(str(x).isdigit() for x in index_texts):
            indices = [int(x) for x in index_texts]
            if any(b != a + 1 for a, b in zip(indices, indices[1:])):
                raise ValueError(
                    f"stem={row['stem']}: duplicated/missing BMP indices; "
                    "cannot safely assign last 505/656 frames by wave_pattern"
                )

    wave_pattern = raw_cfg.get("wave_pattern", {})
    if not isinstance(wave_pattern, dict) or "505" not in {str(k) for k in wave_pattern}:
        raise ValueError("raw.wave_pattern must define wavelength 505")
    result: dict[str, Path] = {}
    for wave, pattern in wave_pattern.items():
        wave = str(wave)
        if wave not in {"505", "656"}:
            continue
        if not isinstance(pattern, (list, tuple)) or len(pattern) != 2:
            raise ValueError(f"Invalid raw.wave_pattern[{wave}]: {pattern!r}")
        start, step = int(pattern[0]), int(pattern[1])
        if start < 0 or step <= 0 or start >= step:
            raise ValueError(f"Invalid raw.wave_pattern[{wave}]: {pattern!r}")
        matching = files[start::step]
        if matching:
            result[wave] = matching[-1]
    if "505" not in result:
        raise FileNotFoundError(f"stem={row['stem']}: no raw 505 frame found")
    return result


def load_stem_images(row: pd.Series, config: dict) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load final raw frames in the native Step2/Step3 coordinate space.

    Parameters
    ----------
    row : pandas.Series
        Step2 metadata for the recording to review.
    config : dict
        Project configuration, including ``raw.rotation`` and image path settings.

    Returns
    -------
    images : dict[str, numpy.ndarray]
        Rotated, unscaled grayscale frames with shape ``(y, x)`` for supported
        wavelength keys; 505-nm data must be present.
    origins : dict[str, str]
        Source BMP paths for ``505`` and ``656``; the missing 656-nm channel has
        an empty path.

    Notes
    -----
    Reference BMP dimensions are checked against the rotated final raw images.
    Masks are never resized to force a match. ``raw.scale`` is not applied here.
    """
    last_paths = _last_raw_paths_by_wavelength(row, config)
    images: dict[str, np.ndarray] = {}
    origins = {"505": "", "656": ""}
    for wave in ("505", "656"):
        if wave not in last_paths:
            continue
        path = last_paths[wave]
        images[wave] = apply_raw_rotation(_read_gray(path), config)
        origins[wave] = str(path)

    # Step2/Step3 masks use rotated_fullres. Checking the early reference here
    # verifies only the grid shape, never substitutes it for the last raw frame.
    ref_path = _resolved_image_path(row, "505", config, required=True)
    assert ref_path is not None
    expected_shape = _read_gray(ref_path).shape
    for wave, img in images.items():
        if img.shape != expected_shape:
            raise ValueError(
                f"stem={row['stem']}: last raw {wave} rotated shape {img.shape} "
                f"differs from Step1/Step2 rotated_fullres shape {expected_shape}; "
                "no mask resize allowed"
            )
    return images, origins


def load_step2_index(config_path: str | Path) -> tuple[pd.DataFrame, dict, dict]:
    """Load the authoritative cohort for hemorrhage mask review.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the project's YAML configuration.

    Returns
    -------
    metadata : pandas.DataFrame
        Selected Step2 rows, with configured ``excluded_stems`` removed.
    config : dict
        Parsed project configuration.
    paths : dict
        Standard project directories and CSV paths.

    Notes
    -----
    No additional selection by experimental day, recording state, or signal QC
    is applied. Previous protocol-repeat selection is not recalculated, and
    rows marked ``is_bad`` remain available for review.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    paths = build_standard_paths(config, create_or_load_project(config, config_path))
    metadata_path = paths["metadata_registration_csv"]
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run Step2 first: {metadata_path}")
    df = pd.read_csv(metadata_path)
    necessary = {"stem", "group_id", "mouse_id", "exp_day", "state", "protocol"}
    missing = necessary - set(df.columns)
    if missing:
        raise ValueError(f"Step2 metadata missing required columns: {sorted(missing)}")
    if df["stem"].isna().any() or df["stem"].astype(str).str.strip().eq("").any():
        raise ValueError("Step2 metadata contains empty stems")
    if any(s in {".", ".."} or "/" in s or "\\" in s for s in df["stem"].astype(str)):
        raise ValueError("Step2 metadata contains invalid stem path components")
    if df["stem"].astype(str).duplicated().any():
        stems = df.loc[df["stem"].astype(str).duplicated(keep=False), "stem"].tolist()
        raise ValueError(f"Step2 metadata has duplicate stem(s): {stems}")

    # Step2 has already applied variant_keys.highest_number. Do NOT recompute on
    # its filtered output: a missing highest repeat must not promote an older one.
    excluded = set(str(s) for s in (config.get("excluded_stems") or []))
    df = df.loc[~df["stem"].astype(str).isin(excluded)].copy()
    # 'is_bad' is intentionally NOT filtered: user asked to review all Step2 rows.
    return df.reset_index(drop=True), config, paths


def read_review_log(log_csv: Path) -> pd.DataFrame:
    """Load the hemorrhage review CSV or create an empty schema.

    Parameters
    ----------
    log_csv : pathlib.Path
        Review log location.

    Returns
    -------
    pandas.DataFrame
        One row per stem with columns in ``LOG_COLUMNS`` order.

    Raises
    ------
    ValueError
        If the existing CSV lacks valid unique stem identifiers.

    Notes
    -----
    Missing optional columns are added as empty strings for compatibility with
    previous review log versions.
    """
    if log_csv.exists():
        df = pd.read_csv(log_csv, dtype={"stem": str})
        if "stem" not in df.columns or df["stem"].isna().any():
            raise ValueError(f"Invalid hemorrhage review CSV: {log_csv}")
        if df["stem"].duplicated().any():
            raise ValueError(f"Duplicate stem(s) in hemorrhage review CSV: {log_csv}")
        for col in LOG_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[LOG_COLUMNS].copy()
    return pd.DataFrame(columns=LOG_COLUMNS)


def validate_mask_file(mask_file: Path, shape: tuple[int, int]) -> np.ndarray:
    """Validate a saved binary hemorrhage mask on the native image grid.

    Parameters
    ----------
    mask_file : pathlib.Path
        Saved ``hemorrhage_manual.npy`` path.
    shape : tuple[int, int]
        Expected native rotated full-resolution image shape ``(y, x)`` in pixels.

    Returns
    -------
    numpy.ndarray
        Binary mask with shape ``(y, x)`` and dtype ``uint8``; value 1 marks
        hemorrhage and 0 marks background.

    Raises
    ------
    ValueError
        If the dimensions or binary label values are invalid.
    """
    mask = np.load(mask_file, allow_pickle=False)
    if mask.ndim != 2 or tuple(mask.shape) != tuple(shape):
        raise ValueError(f"Mask shape {mask.shape} != native image shape {shape}: {mask_file}")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"Mask must contain only 0/1: {mask_file}")
    return mask.astype(np.uint8)


def review_status(
    row: pd.Series, paths: dict, log_df: pd.DataFrame, image_shape: tuple[int, int],
    image_paths: dict[str, str] | None = None,
) -> str:
    """Determine whether an existing mask and its review log agree.

    Parameters
    ----------
    row : pandas.Series
        Step2 recording metadata, including ``stem``.
    paths : dict
        Standard project paths.
    log_df : pandas.DataFrame
        Previously saved hemorrhage review log.
    image_shape : tuple[int, int]
        Native image dimensions ``(y, x)`` in pixels.
    image_paths : dict[str, str] or None, optional
        Source last-frame BMP paths for 505/656 nm. If supplied, a review based
        on different input frames is marked invalid.

    Returns
    -------
    str
        ``unreviewed``, ``invalid``, ``reviewed_present``, or ``reviewed_absent``.

    Notes
    -----
    The saved mask is checked against image shape, coordinate space, filename,
    recorded pixel count, and the claimed positive/negative review decision.
    A mask alone is not treated as a completed review.
    """
    stem = _text(row["stem"])
    mp = mask_path(paths, stem)
    match = log_df.loc[log_df["stem"].astype(str).eq(stem)]
    if match.empty and not mp.exists():
        return "unreviewed"
    if match.empty or not mp.exists():
        return "invalid"  # Mask without explicit review != reviewed_absent.
    record = match.iloc[0]
    status = _text(record["review_status"])
    if status not in STATUSES:
        return "invalid"
    if _text(record.get("image_shape", "")) != f"{image_shape[0]}x{image_shape[1]}":
        return "invalid"
    if _text(record.get("image_space", "")) != IMAGE_SPACE:
        return "invalid"
    if _text(record.get("hemorrhage_mask_path", "")) != str(mp):
        return "invalid"
    if image_paths is not None:
        # Old reference-based reviews are not silently treated as reviewed on
        # the new last-frame images. Review them again explicitly.
        for wave in ("505", "656"):
            if _text(record.get(f"last_raw_{wave}_path", "")) != image_paths.get(wave, ""):
                return "invalid"
    try:
        mask = validate_mask_file(mp, image_shape)
        area = int(mask.sum())
        old_area = int(record["area_px"])
        if area != old_area:
            return "invalid"
        if (status == "reviewed_present") != bool(area):
            return "invalid"
        return status
    except (ValueError, TypeError, OSError):
        return "invalid"


def _atomic_write_mask(path: Path, value: np.ndarray) -> None:
    """Replace a mask file through a temporary file in its directory.

    Parameters
    ----------
    path : pathlib.Path
        Destination ``.npy`` file.
    value : numpy.ndarray
        Mask values to store as ``uint8``.

    Notes
    -----
    The temporary file is replaced atomically when supported by the filesystem;
    leftover temporary files are removed when possible.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".hemorrhage_", suffix=".npy", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, value.astype(np.uint8, copy=False), allow_pickle=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    """Replace a review log through a temporary CSV file.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV file.
    df : pandas.DataFrame
        Review records to save without a DataFrame index.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".hemorrhage_log_", suffix=".csv", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save_confirmed_review(
    row: pd.Series, mask: np.ndarray, expected_shape: tuple[int, int],
    paths: dict, log_csv: Path, *, annotator: str,
    image_paths: dict[str, str] | None = None,
) -> str:
    """Save a manually confirmed binary hemorrhage mask and audit record.

    Parameters
    ----------
    row : pandas.Series
        Step2 metadata for the reviewed stem.
    mask : numpy.ndarray
        Native rotated full-resolution mask with shape ``(y, x)`` and labels 0/1.
    expected_shape : tuple[int, int]
        Required mask shape in pixels, matching the displayed raw image.
    paths : dict
        Project paths locating canonical per-stem masks.
    log_csv : pathlib.Path
        Destination ``hemorrhage_mask_review.csv``.
    annotator : str
        User label stored in the review record.
    image_paths : dict[str, str] or None, optional
        Source last-frame BMP paths for provenance checks.

    Returns
    -------
    str
        ``reviewed_present`` when any pixel is labeled 1, otherwise
        ``reviewed_absent``.

    Notes
    -----
    The measured ``area_px`` is the count of positive pixels, not an area in
    physical units. Every explicit confirmation increments ``review_revision``.
    The mask and CSV are replaced separately; if log writing fails, the code
    attempts to restore the preceding mask and leaves the old log unchanged.
    Closing or navigating the viewer never calls this function implicitly.
    """
    if mask.ndim != 2 or tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(f"stem={row['stem']}: wrong ROI shape {mask.shape} != {expected_shape}")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError(f"stem={row['stem']}: hemorrhage_EDIT may use only labels 0 and 1")
    mask = np.asarray(mask, dtype=np.uint8).copy()
    stem = _text(row["stem"])
    area = int(mask.sum())
    status = "reviewed_present" if area else "reviewed_absent"
    mp = mask_path(paths, stem)
    log = read_review_log(log_csv)
    prev = log.loc[log["stem"].astype(str).eq(stem)]
    old_revision = int(pd.to_numeric(prev.iloc[0]["review_revision"], errors="coerce")) if (not prev.empty and pd.notna(pd.to_numeric(prev.iloc[0]["review_revision"], errors="coerce"))) else 0
    origin = image_paths or {}
    entry = {
        "stem": stem,
        "group_id": _text(row.get("group_id")),
        "mouse_id": _text(row.get("mouse_id")),
        "exp_day": _text(row.get("exp_day")),
        "state": _text(row.get("state")),
        "protocol": _text(row.get("protocol")),
        "protocol_base": _text(row.get("protocol_base")),
        "protocol_repeat": _text(row.get("protocol_repeat")),
        "review_status": status,
        "area_px": area,
        "hemorrhage_mask_path": str(mp),
        "image_shape": f"{expected_shape[0]}x{expected_shape[1]}",
        "image_space": IMAGE_SPACE,
        "last_raw_505_path": origin.get("505", ""),
        "last_raw_656_path": origin.get("656", ""),
        "review_revision": old_revision + 1,
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "annotator": str(annotator),
    }
    rest = log.loc[~log["stem"].astype(str).eq(stem)]
    next_log = pd.concat([rest, pd.DataFrame([entry], columns=LOG_COLUMNS)], ignore_index=True)

    # Two files cannot be atomically replaced together. If logging fails after
    # the mask replacement, roll back the prior mask and leave the old CSV intact.
    previous_mask_bytes = mp.read_bytes() if mp.exists() else None
    try:
        _atomic_write_mask(mp, mask)
        _atomic_write_csv(log_csv, next_log)
    except Exception:
        if previous_mask_bytes is None:
            mp.unlink(missing_ok=True)
        else:
            fd, temp = tempfile.mkstemp(prefix=".hemorrhage_restore_", suffix=".npy", dir=mp.parent)
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(previous_mask_bytes)
                os.replace(temp, mp)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)
        raise
    return status


def open_gui(df: pd.DataFrame, config: dict, paths: dict, *, annotator: str) -> None:
    """Open the interactive Napari review for selected Step2 recordings.

    Parameters
    ----------
    df : pandas.DataFrame
        Step2-selected recording rows; current order controls GUI navigation.
    config : dict
        Project configuration for raw BMP loading and geometry.
    paths : dict
        Standard project paths.
    annotator : str
        User identifier written to confirmed reviews.

    Notes
    -----
    The GUI presents final raw 505/656-nm frames and one editable binary Labels
    layer at a time. Only an explicit confirmation saves a mask and CSV record.
    The Napari and Qt dependencies are imported when the GUI is opened.
    """
    try:
        import napari
        from qtpy.QtWidgets import (
            QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QHBoxLayout,
        )
    except ImportError as exc:
        raise ImportError("The review GUI requires napari and qtpy in your project environment") from exc

    _, log_csv = review_paths(config, paths)
    viewer = napari.Viewer(title="Hemorrhage manual review — native stem space")

    class ReviewPanel(QWidget):
        """Qt controls for navigating, editing, and confirming stem masks.

        Notes
        -----
        This nested class is used only inside ``open_gui``. It keeps the current
        editable image layer and last loaded/saved binary mask in memory.
        """
        def __init__(self):
            """Create the navigation, instructions, and explicit-save controls.
            """
            super().__init__()
            self.position = -1
            self.layer = None
            self.source_mask = None
            self.current_shape = None
            self.origins = {}
            outer = QVBoxLayout(self)
            self.header = QLabel("Loading...")
            self.header.setWordWrap(True)
            outer.addWidget(self.header)
            self.help = QLabel(
                "Paint only label 1 (hemorrhage); label 0 erases. "
                "Save requires explicit confirmation, including an empty mask. "
                "Navigating without saving leaves files unchanged."
            )
            self.help.setWordWrap(True)
            outer.addWidget(self.help)
            nav = QHBoxLayout()
            previous_button = QPushButton("← Previous")
            next_button = QPushButton("Next →")
            previous_button.clicked.connect(lambda: self.navigate(-1))
            next_button.clicked.connect(lambda: self.navigate(1))
            nav.addWidget(previous_button)
            nav.addWidget(next_button)
            outer.addLayout(nav)
            save_button = QPushButton("Confirm and save current mask")
            save_button.clicked.connect(self.save_current)
            outer.addWidget(save_button)
            save_next_button = QPushButton("Confirm, save and next →")
            save_next_button.clicked.connect(self.save_and_next)
            outer.addWidget(save_next_button)
            outer.addStretch()
            self.load_position(0)

        def modified(self) -> bool:
            """Return whether the editable layer differs from the loaded mask.
            """
            return (self.layer is not None and self.source_mask is not None
                    and not np.array_equal(np.asarray(self.layer.data), self.source_mask))

        def navigate(self, delta: int) -> None:
            """Move between stems after confirming the discard of unsaved edits.

            Parameters
            ----------
            delta : int
                Relative step in the selected recording table, usually -1 or +1.
            """
            target = self.position + delta
            if not 0 <= target < len(df):
                return
            if self.modified():
                yes = QMessageBox.question(
                    self, "Discard unsaved edits?",
                    "Current drawing differs from the last loaded/saved mask. Discard it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
                )
                if yes != QMessageBox.Yes:
                    return
            self.load_position(target)

        def load_position(self, pos: int) -> None:
            """Load one recording's final raw images and existing mask into Napari.

            Parameters
            ----------
            pos : int
                Zero-based row index in the selected Step2 table.

            Notes
            -----
            An invalid saved mask is rejected rather than resized. Review status is
            validated against both the mask and its CSV provenance before display.
            """
            self.position = pos
            row = df.iloc[pos]
            stem = _text(row["stem"])
            self.layer = None
            self.source_mask = None
            self.current_shape = None
            self.origins = {}
            viewer.layers.clear()
            try:
                images, origins = load_stem_images(row, config)
                self.origins = origins
                self.current_shape = tuple(images["505"].shape)
                for wave in ("505", "656"):
                    if wave in images:
                        viewer.add_image(images[wave], name=f"last_raw_{wave} (this stem)", visible=(wave == "505"))
                mp = mask_path(paths, stem)
                if mp.exists():
                    # Incorrect shape/labels should never be silently resized.
                    loaded = validate_mask_file(mp, self.current_shape)
                else:
                    loaded = np.zeros(self.current_shape, dtype=np.uint8)
                log = read_review_log(log_csv)
                status = review_status(row, paths, log, self.current_shape, origins)
                self.source_mask = loaded.copy()
                self.layer = viewer.add_labels(
                    loaded.copy(), name="hemorrhage_EDIT", opacity=0.48,
                    visible=True,
                )
                self.layer.selected_label = 1
                viewer.title = f"Hemorrhage {pos + 1}/{len(df)}: {stem}"
                self.header.setText(
                    f"{pos + 1}/{len(df)} — {stem}\n"
                    f"group_id={_text(row.get('group_id'))}\n"
                    f"Status: {status} | native {self.current_shape[0]}x{self.current_shape[1]}\n"
                    f"Last 505: {Path(origins['505']).name}\n"
                    f"Last 656: {Path(origins['656']).name if origins['656'] else 'not available'}"
                )
                if status == "invalid":
                    QMessageBox.warning(self, "Invalid earlier review",
                        "Mask or review CSV is inconsistent. Review carefully and confirm a new save.")
            except Exception as exc:
                self.header.setText(f"{pos + 1}/{len(df)} — {stem}\nERROR: {exc}\nNo data modified.")
                QMessageBox.warning(self, f"Cannot load stem {stem}", str(exc))

        def save_current(self) -> bool:
            """Request confirmation and persist the current binary mask.

            Returns
            -------
            bool
                ``True`` when the mask and log were successfully saved, otherwise ``False``.

            Notes
            -----
            An empty mask is allowed but must be explicitly confirmed; it is logged
            as ``reviewed_absent``.
            """
            if self.layer is None or self.current_shape is None:
                QMessageBox.warning(self, "No editable mask", "This stem did not load successfully.")
                return False
            row = df.iloc[self.position]
            mask = np.asarray(self.layer.data)
            if mask.ndim != 2 or tuple(mask.shape) != self.current_shape or not np.isin(mask, [0, 1]).all():
                QMessageBox.warning(self, "Invalid mask", "Only label values 0/1 and exact native image shape are allowed.")
                return False
            area = int(mask.sum())
            decision = "reviewed_present" if area else "reviewed_absent"
            answer = QMessageBox.question(
                self, "Confirm hemorrhage review",
                f"stem: {row['stem']}\nStatus: {decision}\nArea: {area} pixels\n"
                "Overwrite this stem's hemorrhage_manual.npy and update the review CSV?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return False
            try:
                result = save_confirmed_review(
                    row, mask, self.current_shape, paths, log_csv,
                    annotator=annotator, image_paths=self.origins,
                )
                self.source_mask = mask.astype(np.uint8).copy()
                self.header.setText(self.header.text().split("\nStatus:")[0] + f"\nStatus: {result} (saved)")
                return True
            except Exception as exc:
                QMessageBox.critical(self, "Save failed", str(exc))
                return False

        def save_and_next(self) -> None:
            """Save the current review and advance only after successful saving.
            """
            if self.save_current() and self.position + 1 < len(df):
                self.load_position(self.position + 1)

    panel = ReviewPanel()
    viewer.window.add_dock_widget(panel, area="right", name="Hemorrhage review")
    napari.run()


def main() -> None:
    """Parse command-line arguments and launch or list hemorrhage reviews.

    Notes
    -----
    ``--config`` selects the YAML configuration. ``--stem`` restricts review to
    one exact Step2-selected recording, while ``--list`` reports status without
    opening Napari. No additional experiment-specific cohort filters are added.
    """
    parser = argparse.ArgumentParser(description="Manual hemorrhage masks for ALL Step2 selected stems")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stem", help="Review one exact selected Step2 stem")
    parser.add_argument("--annotator", default="user")
    parser.add_argument("--list", action="store_true", help="Print selected stems without starting Napari")
    args = parser.parse_args()
    df, config, paths = load_step2_index(args.config)
    if args.stem:
        if args.stem not in set(df["stem"].astype(str)):
            parser.error(f"stem not in Step2 selected metadata: {args.stem}")
        df = df.loc[df["stem"].astype(str).eq(args.stem)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No selected stems in Step2 metadata")
    print(f"Selected {len(df)} Step2 stems; excluded_stems and superseded repeats are not reintroduced.")
    if args.list:
        _, log_csv = review_paths(config, paths)
        log = read_review_log(log_csv)
        for _, row in df.iterrows():
            stem = _text(row["stem"])
            try:
                image, origins = load_stem_images(row, config)
                status = review_status(row, paths, log, image["505"].shape, origins)
            except Exception as exc:
                status = f"missing_data: {exc}"
            print(f"{stem}: {status}")
        return
    open_gui(df, config, paths, annotator=args.annotator)


if __name__ == "__main__":
    main()
