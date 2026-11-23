from __future__ import annotations

import datetime
import re
import yaml
import numpy as np
import cv2
from pathlib import Path, PureWindowsPath
from typing import Any, Optional, Sequence

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of silently overwriting them."""


def _construct_mapping_without_duplicates(loader: yaml.SafeLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict[Any, Any]:
    """Construct a YAML mapping while rejecting duplicate keys.

    Parameters
    ----------
    loader : yaml.SafeLoader
        Loader instance processing the mapping.
    node : yaml.nodes.MappingNode
        YAML mapping node to construct.
    deep : bool, optional
        Whether to construct nested keys and values deeply.

    Returns
    -------
    dict
        Constructed mapping with unique keys.

    Raises
    ------
    ValueError
        If a key appears more than once in the same mapping."""
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = getattr(key_node.start_mark, "line", -1) + 1
            raise ValueError(f"Duplicate YAML key {key!r} at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_without_duplicates,
)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a project YAML configuration without accepting duplicate keys.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        YAML file path; defaults to ``config.yaml``.

    Returns
    -------
    dict[str, Any]
        Root configuration mapping.

    Raises
    ------
    FileNotFoundError
        If the YAML file is absent.
    ValueError
        If the YAML root is empty or is not a mapping, or if a mapping
        contains a duplicate key.

    Notes
    -----
    The file is decoded with ``utf-8-sig`` to allow a UTF-8 BOM."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.load(f, Loader=_UniqueKeyLoader)
    if cfg is None:
        raise ValueError(f"Empty config: {path}")
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write the project configuration as UTF-8 YAML.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration mapping to serialize.
    path : str or pathlib.Path
        Output file path. Parent directories are created if necessary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ============================================================
# PROJECT STRUCTURE
# ============================================================

def create_or_load_project(config: dict[str, Any], config_path: str | Path) -> dict[str, Path]:
    """Create the analysis directories or reuse an explicitly configured root.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration. If ``project.directory_to_save_main_analysis``
        is empty, this function sets that key and saves the updated YAML.
    config_path : str or pathlib.Path
        Configuration file used for saving the inferred analysis location.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths named ``analysis_root``, ``out_root``, and ``step1_root`` through
        ``step4_root``.

    Raises
    ------
    FileExistsError
        If a newly selected analysis directory already contains files.

    Notes
    -----
    An existing ``project.directory_to_save_main_analysis`` takes precedence
    regardless of ``create_new_analysis_folder``. Only the branch without an
    explicit directory writes the updated configuration and its optional
    snapshot. The function creates directories and can modify the config file."""
    config_path = Path(config_path)
    project_cfg = config.setdefault("project", {})
    create_new = bool(project_cfg.get("create_new_analysis_folder", True))
    out_folder_name = project_cfg.get("out_folder_name", "out")
    existing_dir = project_cfg.get("directory_to_save_main_analysis")
    path_cfg = get_path_resolving_config(config)
    remap_kwargs = {key: path_cfg.get(key) for key in ("from_drive", "to_drive", "from_root", "to_root")}

    if existing_dir:
        analysis_root = resolve_flexible_path(existing_dir, **remap_kwargs, must_exist=False)
        analysis_root.mkdir(parents=True, exist_ok=True)
    else:
        analysis_parent = resolve_flexible_path(
            project_cfg.get("analysis_parent_dir", config_path.parent), **remap_kwargs, must_exist=False
        )
        analysis_parent.mkdir(parents=True, exist_ok=True)
        if create_new:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            analysis_root = analysis_parent / (
                project_cfg.get("analysis_folder_name") or f"Main_analysis_{today}"
            )
            if analysis_root.exists() and any(analysis_root.iterdir()):
                raise FileExistsError(f"New project directory is not empty: {analysis_root}")
        else:
            analysis_root = analysis_parent
        analysis_root.mkdir(parents=True, exist_ok=True)
        project_cfg["directory_to_save_main_analysis"] = str(analysis_root)
        save_config(config, config_path)
        snapshot_name = project_cfg.get("config_snapshot_name")
        if snapshot_name:
            save_config(config, analysis_root / snapshot_name)

    out_root = analysis_root / out_folder_name
    out_root.mkdir(parents=True, exist_ok=True)

    step1_root = out_root / config.get("step1", {}).get("folder_name", "step1_project")
    step2_root = out_root / config.get("step2", {}).get("folder_name", "step2_masks")
    step3_root = out_root / config.get("step3", {}).get("folder_name", "step3")
    step4_cfg = config.get("step4", {})
    step4_root = out_root / step4_cfg.get("folder_name", "step4_raw_and_signals")

    for p in [step1_root, step2_root, step3_root, step4_root]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "analysis_root": analysis_root,
        "out_root": out_root,
        "step1_root": step1_root,
        "step2_root": step2_root,
        "step3_root": step3_root,
        "step4_root": step4_root,
    }


def build_standard_paths(config: dict[str, Any], project_paths: dict[str, Path]) -> dict[str, Path]:
    """Construct conventional input and output paths for Steps 1–4.

    Parameters
    ----------
    config : dict[str, Any]
        YAML mapping containing ``step1`` through ``step4`` folder names.
    project_paths : dict[str, pathlib.Path]
        Directory mapping returned by :func:`create_or_load_project`.

    Returns
    -------
    dict[str, pathlib.Path]
        Manifest, reference image, mask, registration, QC, raw-Zarr, and
        processed-signal paths in addition to the step roots.

    Notes
    -----
    Creates the reference-image, mask, manual-mask, raw-Zarr, and
    processed-signal directories if they do not exist."""
    step1_cfg = config.get("step1", {}) or {}
    step2_cfg = config.get("step2", {}) or {}
    step3_cfg = config.get("step3", {}) or {}
    step4_cfg = config.get("step4", {}) or {}
    step1_ref_cfg = step1_cfg.get("reference_selection", {}) or {}

    step1_root = project_paths["step1_root"]
    step2_root = project_paths["step2_root"]
    step3_root = project_paths["step3_root"]
    step4_root = project_paths["step4_root"]

    manifest_path = project_paths["out_root"] / step1_cfg.get("manifest_name", "manifest_recordings.csv")
    reference_images_root = step1_root / step1_ref_cfg.get("output_folder_name", "ilastik_reference_images")
    masks_root = step2_root / step2_cfg.get("masks_folder", "masks")
    metadata_registration_csv = step2_root / step2_cfg.get("metadata_registration_name", "metadata_registration.csv")
    frame_qc_csv = step3_root / step3_cfg.get("frame_qc_name", "frame_qc.csv")
    reference_annotations_csv = step3_root / step3_cfg.get("reference_annotations_name", "reference_annotations.csv")
    manual_masks_root = step3_root / step3_cfg.get("manual_brain_masks_folder", step3_cfg.get("manual_masks_folder", "manual_brain_masks"))
    raw_zarr_root = step4_root / step4_cfg.get("raw_zarr_folder", "raw_zarr")
    processed_signals_root = step4_root / step4_cfg.get("processed_signals_folder", "processed_signals")

    for p in [reference_images_root, masks_root, manual_masks_root, raw_zarr_root, processed_signals_root]:
        p.mkdir(parents=True, exist_ok=True)

    return {
        "out_root": project_paths["out_root"],
        "step1_root": step1_root,
        "step2_root": step2_root,
        "step3_root": step3_root,
        "step4_root": step4_root,
        "manifest_path": manifest_path,
        "reference_images_root": reference_images_root,
        "masks_root": masks_root,
        "metadata_registration_csv": metadata_registration_csv,
        "frame_qc_csv": frame_qc_csv,
        "reference_annotations_csv": reference_annotations_csv,
        "manual_masks_root": manual_masks_root,
        "raw_zarr_root": raw_zarr_root,
        "processed_signals_root": processed_signals_root,
    }


# ============================================================
# STEM / GROUPING
# ============================================================

def get_stem_keys(config: dict[str, Any]) -> list[str]:
    """Return the ordered metadata keys used to identify recordings.

    Parameters
    ----------
    config : dict[str, Any]
        Project configuration containing ``metadata.stem_keys``.

    Returns
    -------
    list[str]
        Ordered, string-converted recording identity fields.

    Raises
    ------
    ValueError
        If ``metadata.stem_keys`` is empty or absent."""
    keys = list((config.get("metadata", {}) or {}).get("stem_keys", []) or [])
    if not keys:
        raise ValueError("metadata.stem_keys is required in v2 pipeline")
    return [str(k) for k in keys]


def build_group_id(row: Any, group_by: Sequence[str]) -> str:
    """Join the configured grouping fields to form a registration group ID.

    Parameters
    ----------
    row : dict or pandas.Series
        Metadata values for a recording.
    group_by : sequence of str
        Ordered group fields; these are not required to match stem keys.

    Returns
    -------
    str
        Underscore-joined values in ``group_by`` order.

    Raises
    ------
    ValueError
        If a selected field is missing or has an empty value.

    Notes
    -----
    This function does not normalize underscores or otherwise escape the
    component values. Group definitions must therefore be consistent."""
    vals: list[str] = []
    for key in group_by:
        if isinstance(row, dict):
            value = row.get(key, "")
        else:
            value = row[key] if key in row.index else ""
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"Empty group key {key!r}")
        vals.append(str(value))
    return "_".join(vals)


def extract_named_groups(text: str, regex: str) -> dict[str, str]:
    """Extract named regex captures from text.

    Parameters
    ----------
    text : str
        String in which to search for the expression.
    regex : str
        Pattern containing named capture groups.

    Returns
    -------
    dict[str, str]
        Non-None named captures, or an empty mapping when there is no match."""
    m = re.search(regex, text)
    if m is None:
        return {}
    return {k: str(v) for k, v in m.groupdict().items() if v is not None}


def extract_path_metadata(img_dir: str | Path) -> dict[str, str]:
    """Read recording metadata from the last four folder names.

    Parameters
    ----------
    img_dir : str or pathlib.Path
        Directory expected to end in ``exp_day/mouse_id/state/protocol``.

    Returns
    -------
    dict[str, str]
        ``exp_day``, ``mouse_id``, ``state``, and ``protocol`` values.

    Raises
    ------
    ValueError
        If fewer than four path components are available.

    Notes
    -----
    This is a positional fallback; additional parent directories do not
    become metadata fields."""
    p = Path(img_dir)
    parts = p.parts
    if len(parts) < 4:
        raise ValueError(f"Cannot extract folder metadata from path with <4 parts: {p}")
    exp_day, mouse_id, state, protocol = parts[-4:]
    return {"exp_day": exp_day, "mouse_id": mouse_id, "state": state, "protocol": protocol}


def normalize_metadata_values(meta: dict[str, str], replace_underscore_with: str = "-") -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    """Replace underscores in metadata values and record changes.

    Parameters
    ----------
    meta : dict[str, str]
        Original named metadata fields.
    replace_underscore_with : str, optional
        Replacement used within each value; defaults to a hyphen.

    Returns
    -------
    normalized : dict[str, str]
        Metadata with replaced underscore characters.
    replacements : dict[str, dict[str, str]]
        Original and normalized values for fields that changed."""
    out: dict[str, str] = {}
    replacements: dict[str, dict[str, str]] = {}
    for key, value in meta.items():
        value_str = str(value)
        new_value = value_str.replace("_", replace_underscore_with)
        out[key] = new_value
        if new_value != value_str:
            replacements[key] = {"original": value_str, "normalized": new_value}
    return out, replacements


def parse_variant_value(value: Any, separators: Sequence[str] = ("", "-", "_")) -> tuple[str, int]:
    """Split a trailing numeric protocol repeat suffix at configured separators.

    Parameters
    ----------
    value : Any
        Protocol value to parse.
    separators : sequence of str, optional
        Accepted separators before the final integer; the empty separator
        permits values such as ``Stim2``.

    Returns
    -------
    base : str
        Protocol label without the detected repeat suffix.
    repeat : int
        Extracted repeat number, or zero if no suffix is recognized.

    Notes
    -----
    Separators are tried from longest to shortest. A trailing number can
    be interpreted as a repeat when the empty separator is enabled."""
    text = "" if value is None or pd.isna(value) else str(value).strip()
    if not text:
        return "", 0
    for sep in sorted(separators, key=len, reverse=True):
        match = re.fullmatch(r"(.+?)" + re.escape(str(sep)) + r"(\d+)", text)
        if match:
            base = match.group(1).strip()
            if base:
                return base, int(match.group(2))
    return text, 0


def infer_stem(
    img_dir: str | Path,
    *,
    mode: str = "regex",
    regex: str | None = None,
    regex_target: str = "path",
    stem_keys: Sequence[str] = ("mouse_id", "exp_day", "state", "protocol"),
    normalize_underscores: bool = True,
    underscore_replacement: str = "-",
    fallback_to_folder_structure: bool = True,
    protocol_separators: Sequence[str] = ("", "-", "_"),
) -> tuple[str, dict[str, str], dict[str, dict[str, str]]]:
    """Derive a unique recording stem and its metadata from a directory.

    Parameters
    ----------
    img_dir : str or pathlib.Path
        Raw recording directory.
    mode : {'regex', 'folder_structure'}, optional
        Method used to identify metadata fields.
    regex : str or None, optional
        Named-group expression required in ``regex`` mode.
    regex_target : str, optional
        ``path`` searches the full path; other values search the basename.
    stem_keys : sequence of str, optional
        Ordered metadata fields joined into ``stem``.
    normalize_underscores : bool, optional
        Whether to normalize underscores in parsed metadata values.
    underscore_replacement : str, optional
        Replacement used for underscores in metadata values.
    fallback_to_folder_structure : bool, optional
        Whether a failed regex match invokes the four-folder parser.
    protocol_separators : sequence of str, optional
        Separators used to infer ``protocol_base`` and ``protocol_repeat``.

    Returns
    -------
    stem : str
        Recording identifier assembled from ordered stem fields, or the
        recording directory name if any requested field is missing.
    metadata : dict[str, str]
        Parsed and derived fields including ``stem_metadata_source`` and,
        where applicable, ``protocol_base`` and ``protocol_repeat``.
    replacements : dict[str, dict[str, str]]
        Record of underscore normalization per field.

    Notes
    -----
    Missing stem fields do not raise an exception here; the fallback stem
    receives ``stem_fallback_reason``. Duplicate stems are checked later
    when the Step1 manifest is built."""
    img_dir = Path(img_dir)
    if mode not in {"regex", "folder_structure"}:
        raise ValueError("mode must be 'regex' or 'folder_structure'")
    if mode == "folder_structure":
        meta = extract_path_metadata(img_dir)
        metadata_source = "folder_structure"
    else:
        if regex is None:
            raise ValueError("metadata.mode='regex' requires metadata.regex")
        target_text = str(img_dir) if regex_target == "path" else img_dir.name
        meta = extract_named_groups(target_text, regex)
        if not meta:
            if fallback_to_folder_structure:
                meta = extract_path_metadata(img_dir)
                metadata_source = "folder_structure_fallback"
            else:
                raise ValueError(f"Regex did not match target text: {target_text}")
        else:
            metadata_source = "regex"
    replacements: dict[str, dict[str, str]] = {}
    if normalize_underscores:
        meta, replacements = normalize_metadata_values(meta, underscore_replacement)
    # Provenance fields are not stem keys and therefore do not change recording identity.
    meta["stem_metadata_source"] = metadata_source
    if "protocol" in meta and "protocol_base" not in meta:
        base, repeat = parse_variant_value(meta["protocol"], protocol_separators)
        meta["protocol_base"] = base
        meta["protocol_repeat"] = str(repeat)

    missing_keys = [k for k in stem_keys if not meta.get(k)]
    if missing_keys:
        stem = img_dir.name
        meta["stem_fallback_reason"] = f"missing_keys: {missing_keys}"
    else:
        stem = "_".join(str(meta[k]) for k in stem_keys)
    return stem, meta, replacements


def infer_stem_from_config(img_dir: str | Path, config: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, dict[str, str]]]:
    """Apply the project's metadata extraction rules to a recording path.

    Parameters
    ----------
    img_dir : str or pathlib.Path
        Recording directory.
    config : dict[str, Any]
        Configuration blocks ``metadata`` and ``variant_keys.protocol``.

    Returns
    -------
    tuple[str, dict[str, str], dict[str, dict[str, str]]]
        Recording stem, metadata fields, and normalization audit mapping.

    Notes
    -----
    Delegates extraction to :func:`infer_stem` without modifying the
    configuration or forcing protocol repeat selection."""
    meta_cfg = config.get("metadata", {}) or {}
    return infer_stem(
        img_dir=img_dir,
        mode=meta_cfg.get("mode", "regex"),
        regex=meta_cfg.get("regex"),
        regex_target=meta_cfg.get("regex_target", "path"),
        stem_keys=tuple(get_stem_keys(config)),
        normalize_underscores=bool(meta_cfg.get("normalize_underscores", True)),
        underscore_replacement=meta_cfg.get("underscore_replacement", "-"),
        fallback_to_folder_structure=bool(meta_cfg.get("fallback_to_folder_structure", True)),
        protocol_separators=((config.get("variant_keys", {}) or {}).get("protocol", {}) or {}).get(
            "separators", ["", "-", "_"]
        ),
    )


# ============================================================
# PATH RESOLVING / PORTABILITY
# ============================================================

def get_path_resolving_config(config: dict[str, Any], step_name: str | None = None) -> dict[str, Any]:
    """Return the global directory and drive remapping settings.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration containing the top-level ``path_resolving`` block.
    step_name : str or None, optional
        Accepted for caller compatibility; not used to select a block.

    Returns
    -------
    dict[str, Any]
        A shallow copy of the global remapping settings."""
    return dict(config.get("path_resolving", {}) or {})


def _windows_root(value: str) -> PureWindowsPath:
    text = str(value).strip().replace("/", "\\")
    wp = PureWindowsPath(text)
    if wp.drive and text.rstrip("\\").endswith(":"):
        return PureWindowsPath(wp.drive + "\\")
    return wp


def _remapped_windows_candidates(
    original_path: str,
    *,
    from_drive: str | None = None,
    to_drive: str | None = None,
    from_root: str | None = None,
    to_root: str | None = None,
) -> list[Path]:
    """Return remap candidates in explicit priority order: root mapping, then drive mapping."""
    s = str(original_path).strip()
    s_norm = s.replace("/", "\\")
    candidates: list[Path] = []

    if from_root and to_root:
        from_norm = str(from_root).rstrip("\\/").replace("/", "\\")
        lower = s_norm.lower()
        from_lower = from_norm.lower()
        if lower == from_lower or lower.startswith(from_lower + "\\"):
            tail = s_norm[len(from_norm):].lstrip("\\/")
            target = _windows_root(str(to_root))
            candidate_wp = target if not tail else target / PureWindowsPath(tail)
            candidates.append(Path(str(candidate_wp)))

    if from_drive and to_drive:
        wp = PureWindowsPath(s_norm)
        from_drive_norm = str(from_drive).rstrip("\\/").lower()
        if wp.drive and wp.drive.lower() == from_drive_norm:
            target_drive = str(to_drive).rstrip("\\/")
            if not target_drive.endswith(":"):
                target_drive = PureWindowsPath(target_drive).drive or target_drive
            target_root = PureWindowsPath(target_drive + "\\")
            candidate_wp = target_root.joinpath(*wp.parts[1:])
            candidate = Path(str(candidate_wp))
            if all(str(candidate).lower() != str(x).lower() for x in candidates):
                candidates.append(candidate)

    return candidates


def resolve_portable_windows_path(
    original_path: str,
    from_drive: Optional[str] = None,
    to_drive: Optional[str] = None,
    from_root: Optional[str] = None,
    to_root: Optional[str] = None,
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve a Windows-like path using root remap first, then drive remap."""
    if not isinstance(original_path, str) or not original_path.strip():
        raise FileNotFoundError(f"Empty path: {original_path!r}")

    p = Path(original_path)
    if p.exists():
        return p

    candidates = _remapped_windows_candidates(
        original_path,
        from_drive=from_drive,
        to_drive=to_drive,
        from_root=from_root,
        to_root=to_root,
    )
    if must_exist:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        tried = ", ".join(str(x) for x in candidates) or "no remap candidate"
        raise FileNotFoundError(f"Could not resolve path: {original_path}; tried: {tried}")

    return candidates[0] if candidates else p


def resolve_flexible_path(
    value: Any,
    *,
    from_drive: str | None = None,
    to_drive: str | None = None,
    from_root: str | None = None,
    to_root: str | None = None,
    must_exist: bool = False,
) -> Path | None:
    """Resolve input/output paths with one priority rule for root and drive remapping.

    Existing original paths win. Otherwise root remap is attempted before drive remap.
    For outputs (``must_exist=False``), the remapped path is returned even before it
    exists; for required inputs, only an existing candidate is accepted.
    """
    if value is None or not str(value).strip():
        if must_exist:
            raise FileNotFoundError(f"Empty path: {value!r}")
        return None

    s = str(value).strip()
    p = Path(s)
    if p.exists():
        return p

    return resolve_portable_windows_path(
        s,
        from_drive=from_drive,
        to_drive=to_drive,
        from_root=from_root,
        to_root=to_root,
        must_exist=must_exist,
    )




# ============================================================
# RAW GEOMETRY HELPERS
# ============================================================

def raw_rotation_degrees(config: dict[str, Any]) -> float:
    """Return the raw-image rotation specified by the project.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration containing ``raw.rotation``.

    Returns
    -------
    float
        Rotation angle in degrees, defaulting to 0.

    Notes
    -----
    Positive rotation follows the project's ``rotate_bound`` convention:
    +90 degrees is clockwise. Step1 reference images use full resolution."""
    return float((config.get("raw", {}) or {}).get("rotation", 0.0) or 0.0)


def raw_scale_factor(config: dict[str, Any]) -> float:
    """Return the spatial scale applied when constructing raw Zarr data.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration containing ``raw.scale``.

    Returns
    -------
    float
        Spatial scaling factor, defaulting to 1 (dimensionless).

    Notes
    -----
    Step1, Step2, and Step3 retain full-resolution rotated images; the
    scale factor is applied during Step4 raw-Zarr construction."""
    return float((config.get("raw", {}) or {}).get("scale", 1.0) or 1.0)


def rotate_bound_image(image: np.ndarray, angle_deg: float, *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Rotate an image while preserving the expanded image extent.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(y, x)`` or ``(y, x, channels)``.
    angle_deg : float
        Angle in degrees; positive values rotate clockwise.
    interpolation : int, optional
        OpenCV interpolation for the fallback at non-quarter-turn angles.

    Returns
    -------
    numpy.ndarray
        Image with axes ``(y, x[, channels])`` and an expanded canvas when
        required. Spatial dimensions may change after rotation.

    Notes
    -----
    An angle of 0 returns the input unchanged. Exact quarter-turns use
    ``numpy.rot90`` without interpolation. Other angles first attempt
    ``imutils.rotate_bound``; the OpenCV fallback uses ``interpolation``."""
    angle = float(angle_deg or 0.0)
    if abs(angle) < 1e-12:
        return image

    normalized = ((angle + 180.0) % 360.0) - 180.0
    if np.isclose(normalized, 90.0, atol=1e-9):
        return np.rot90(np.asarray(image), k=-1).copy()
    if np.isclose(normalized, -90.0, atol=1e-9):
        return np.rot90(np.asarray(image), k=1).copy()

    try:
        import imutils  # type: ignore
        return imutils.rotate_bound(image, angle)
    except Exception:
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        c_x, c_y = w / 2.0, h / 2.0

        # Same sign convention as imutils.rotate_bound.
        M = cv2.getRotationMatrix2D((c_x, c_y), -angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])

        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        M[0, 2] += (new_w / 2.0) - c_x
        M[1, 2] += (new_h / 2.0) - c_y

        return cv2.warpAffine(
            arr, M, (new_w, new_h), flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )


def apply_raw_rotation(image: np.ndarray, config: dict[str, Any], *, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Apply the configured raw rotation without spatial downsampling.

    Parameters
    ----------
    image : numpy.ndarray
        Full-resolution image with spatial axes ``(y, x)``.
    config : dict[str, Any]
        Configuration containing ``raw.rotation``.
    interpolation : int, optional
        Fallback interpolation method for non-quarter-turn rotations.

    Returns
    -------
    numpy.ndarray
        Rotated, full-resolution image used for reference segmentation and
        geometry in Steps 1–3."""
    return rotate_bound_image(image, raw_rotation_degrees(config), interpolation=interpolation)


def scaled_coordinate_value(value: Any, scale: float) -> float:
    """Scale one y/x coordinate or translation value, preserving NaN."""
    try:
        if pd.isna(value):
            return np.nan
        return float(value) * float(scale)
    except Exception:
        return np.nan



# ============================================================
# CANONICAL STEP2 MASK PATHS
# ============================================================

CANONICAL_MASK_FILENAMES = {
    "brain_auto": "brain_auto.npy",
    "bad_auto": "bad_auto.npy",
    "ring_auto": "ring_auto.npy",
    "bregma_auto": "bregma_auto.npy",
    "vessels_auto": "vessels_auto.npy",
    "brain_manual": "brain_manual.npy",
    "bregma_manual": "bregma_manual.npy",
}

# These columns used to store absolute mask paths in metadata_registration.csv.
# New logic derives all mask paths from paths["masks_root"] + stem, so they should
# not be written to metadata_registration.csv anymore. Registration image paths
# are intentionally not included here: Step3 still needs them.
MASK_PATH_METADATA_COLUMNS = [
    "mask_for_reg_path",
    "brain_path",
    "bad_path",
    "ring_path",
    "background_path",
    "vessels_path",
    "bregma_mask_path",
    "masks_dir",
    "brain_source_path",
    "brain_path_before_step3",
    "background_path_before_step3",
]


def masks_root_from_paths(paths_or_root: Any) -> Path:
    """Return the Step2 masks root from a standard paths dict or a direct path."""
    if isinstance(paths_or_root, dict):
        return Path(paths_or_root["masks_root"])
    return Path(paths_or_root)


def stem_masks_dir(paths_or_root: Any, stem: Any) -> Path:
    """Canonical directory for all masks belonging to one stem."""
    return masks_root_from_paths(paths_or_root) / str(stem)


def canonical_mask_paths(paths_or_root: Any, stem: Any) -> dict[str, Path]:
    """Canonical mask file paths for one stem.

    Auto masks are written by Step2. Manual masks are written by Step3 when a
    user edits the brain mask or bregma point.
    """
    d = stem_masks_dir(paths_or_root, stem)
    return {key: d / filename for key, filename in CANONICAL_MASK_FILENAMES.items()}


def mask_for_reg_union_path(paths_or_root: Any, group_id: Any) -> Path:
    """Canonical group-union registration mask path."""
    return masks_root_from_paths(paths_or_root) / "_group_union" / str(group_id) / "mask_for_reg_union.npy"


def mask_for_reg_per_stem_path(paths_or_root: Any, stem: Any) -> Path:
    """Canonical per-stem registration ROI preview path."""
    return stem_masks_dir(paths_or_root, stem) / "mask_for_reg.npy"


def drop_mask_path_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove absolute mask-path columns from metadata_registration.csv tables."""
    return df.drop(columns=[c for c in MASK_PATH_METADATA_COLUMNS if c in df.columns], errors="ignore")

# ============================================================
# RAW FILE HELPERS
# ============================================================

def parse_raw_filename(path: str | Path, file_regex: str) -> dict[str, str]:
    """Extract named fields from a raw image filename.

    Parameters
    ----------
    path : str or pathlib.Path
        Image path; only its filename is matched.
    file_regex : str
        Regular expression used with ``re.match``.

    Returns
    -------
    dict[str, str]
        Matched non-None named fields or an empty mapping."""
    path = Path(path)
    m = re.match(file_regex, path.name)
    if m is None:
        return {}
    return {k: str(v) for k, v in m.groupdict().items() if v is not None}


def sort_raw_image_paths(paths: Sequence[Path], *, file_regex: str, sort_key: str = "index") -> list[Path]:
    """Order raw images using a captured filename field.

    Parameters
    ----------
    paths : sequence of pathlib.Path
        Image paths to sort.
    file_regex : str
        Named-group filename expression.
    sort_key : str, optional
        Capture used for sorting, defaulting to ``index``.

    Returns
    -------
    list[pathlib.Path]
        Paths sorted numerically for digit-only captures, otherwise as text.

    Raises
    ------
    ValueError
        If a filename has no capture matching ``sort_key``."""
    def key_func(p: Path):
        meta = parse_raw_filename(p, file_regex)
        if sort_key not in meta:
            raise ValueError(f"sort_key={sort_key!r} not found in filename: {p.name}")
        v = meta[sort_key]
        return int(v) if str(v).isdigit() else v
    return sorted(paths, key=key_func)


def list_raw_images(img_dir, *, suffixes=(".bmp",), file_regex=None, sort_key="index"):
    """List raw image files in a recording folder.

    Parameters
    ----------
    img_dir : str or pathlib.Path
        Directory containing image files.
    suffixes : sequence of str, optional
        Case-insensitive allowed file extensions.
    file_regex : str or None, optional
        When provided, retain matching files and sort by the captured key.
    sort_key : str, optional
        Capture to use when ``file_regex`` is set.

    Returns
    -------
    list[pathlib.Path]
        Sorted paths to eligible files within the specified directory."""
    img_dir = Path(img_dir)
    suffixes_lower = {s.lower() for s in suffixes}
    files = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in suffixes_lower]
    if file_regex:
        files = [p for p in files if parse_raw_filename(p, file_regex)]
        return sort_raw_image_paths(files, file_regex=file_regex, sort_key=sort_key)
    return sorted(files)
