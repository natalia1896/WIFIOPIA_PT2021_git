"""Step3 manual registration override widget, v1_6.

This is an optional companion tool for the WIFIOPIA Step3 workflow.
It does not overwrite Step2 automatic registration parameters.  It loads one
Step2 group, shows reference and manually registered moving frames as two normalized grayscale napari layers.  The user can either adjust the widget dx/dy/angle controls or use napari's layer Transform 2D control on the `moving_manual_registered` layer.  Save captures the current layer transform, bakes it into the displayed image, immediately replaces the viewer layer with that baked image, and writes the same image to disk.  Returning to a stem and final QC both reopen the baked correction, so iterative redo passes continue from the saved manual state instead of resetting to Step2 auto.  v1_6 additionally writes final transform coordinates in the same reference-coordinate system as Step2 auto registration, so final_translation_x/y and final_rotation_rad describe the total transform to the reference image, not only the delta from auto.

Typical workflow
----------------
1) Run run_step3_all_groups.py with --ask-registration-qc.
2) Mark bad groups in the console prompt.
3) Run this widget with --bad-registration-only.
4) Downstream tools can prefer manual overrides over Step2 auto transforms.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
)
from skimage.transform import EuclideanTransform, warp

from common import (
    build_standard_paths,
    create_or_load_project,
    get_path_resolving_config,
    load_config,
    resolve_flexible_path,
)
from run_step3_all_groups import (
    latest_bad_registration_groups,
    resolve_metadata_paths,
    reorder_reference_first,
    choose_reference_row,
    registration_qc_csv_path,
    append_registration_qc_row,
)


OVERRIDE_COLUMNS = [
    "group_id", "stem", "ref_stem",
    "auto_translation_x", "auto_translation_y", "auto_rotation_rad",
    "manual_delta_x", "manual_delta_y", "manual_delta_angle_deg",
    "final_translation_x", "final_translation_y", "final_rotation_rad",
    "final_translation_x_ref", "final_translation_y_ref", "final_rotation_deg_ref",
    "manual_total_delta_x_from_auto", "manual_total_delta_y_from_auto", "manual_total_delta_angle_deg_from_auto",
    "rotation_center_mode", "rotation_center_y", "rotation_center_x",
    "napari_layer_transform_used",
    "napari_layer_translate_y", "napari_layer_translate_x",
    "napari_layer_rotation_deg",
    "napari_layer_affine_yx_json",
    "final_transform_xy_json",
    "edit_base_previous_final_transform_xy_json",
    "manual_saved_image_source",
    "viewer_edit_base_source",
    "viewer_edit_base_already_registered",
    "viewer_baked_after_save",
    "fixed_registration_path", "moving_registration_path",
    "manual_registered_path", "manual_json_path",
    "registration_source", "annotator", "notes", "updated_at",
]


def cfg_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return a nested configuration value or a supplied default."""
    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def manual_registration_config(config: dict[str, Any]) -> dict[str, Any]:
    """Read the Step3 manual-registration settings with defaults."""
    step3_cfg = config.get("step3", {}) or {}
    cfg = dict(step3_cfg.get("manual_registration", {}) or {})
    cfg.setdefault("folder_name", "manual_registration")
    cfg.setdefault("overrides_csv_name", "manual_registration_overrides.csv")
    cfg.setdefault("bad_registration_only", True)
    cfg.setdefault("rotation_center", "bregma_ref")
    cfg.setdefault("interpolation_order", 1)
    cfg.setdefault("edge_mode", "constant")
    cfg.setdefault("edge_cval", 0.0)
    cfg.setdefault("moving_opacity", 0.55)
    cfg.setdefault("display_percentiles", [1, 99])
    cfg.setdefault("reference_opacity", 1.0)
    cfg.setdefault("reference_visible", True)
    cfg.setdefault("moving_visible", True)
    cfg.setdefault("capture_napari_layer_transform", True)
    cfg.setdefault("read_saved_bmp_in_qc", True)
    cfg.setdefault("load_saved_bmp_in_viewer", True)
    cfg.setdefault("restore_napari_layer_transform_in_viewer", True)
    cfg.setdefault("remember_overrides_in_session", True)
    cfg.setdefault("bake_saved_override_into_viewer", True)
    cfg.setdefault("reset_layer_transform_after_save", True)
    # Write final transform in reference coordinates (same coordinate system as Step2 auto translation_x/y/rotation_rad).
    cfg.setdefault("save_final_reference_transform", True)
    cfg.setdefault("show_final_reference_transform_in_widget", True)
    cfg.setdefault("qc_after_group", True)
    cfg.setdefault("qc_stack_cmap", "gray")
    cfg.setdefault("save_registered_bmp", True)
    return cfg


def build_manual_registration_paths(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Path]:
    """Create the directories used for manual registration outputs."""
    cfg = manual_registration_config(config)
    root = paths["step3_root"] / str(cfg.get("folder_name", "manual_registration"))
    out = {
        "manual_registration_root": root,
        "manual_registration_overrides_csv": root / str(cfg.get("overrides_csv_name", "manual_registration_overrides.csv")),
        "manual_registered_bmps_root": root / "registered_bmps",
        "manual_json_root": root / "json",
    }
    for p in out.values():
        if p.suffix.lower() in {".csv", ".json"}:
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)
    return out


def _first_existing_path_from_columns(row: pd.Series, columns: Sequence[str], what: str) -> Path:
    """Select an existing image path from prioritized metadata columns."""
    tried: list[str] = []
    for col in columns:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            tried.append(f"{col}={row[col]}")
            p = Path(str(row[col]))
            if p.exists():
                return p
    raise FileNotFoundError(f"{what} not found. Tried: " + "; ".join(tried or list(columns)))


def fixed_image_path(row: pd.Series) -> Path:
    """Resolve the fixed image chosen by the Step2 registration."""
    return _first_existing_path_from_columns(
        row,
        ["fixed_registration_path_resolved", "fixed_registration_path"],
        "fixed/reference registration image",
    )


def moving_image_path(row: pd.Series) -> Path:
    # For the reference row, the moving image is the fixed reference itself.
    """Resolve the moving pre-registration image for a record."""
    stem = str(row.get("stem", ""))
    ref_stem = str(row.get("ref_stem", ""))
    if stem and ref_stem and stem == ref_stem:
        return fixed_image_path(row)
    return _first_existing_path_from_columns(
        row,
        ["moving_registration_path_resolved", "moving_registration_path"],
        "moving/pre-registration image",
    )


def saved_registered_image_path(row: pd.Series) -> Path | None:
    """Find the optional Step2 registered image for a record."""
    cols = ["saved_registered_path_resolved", "saved_registered_path"]
    for col in cols:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            p = Path(str(row[col]))
            if p.exists():
                return p
    return None


def read_gray(path: Path) -> np.ndarray:
    """Read an image and convert supported formats to grayscale float32."""
    arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        return arr[..., :3].mean(axis=-1).astype(np.float32)
    if arr.ndim == 3:
        return arr[0].astype(np.float32)
    raise ValueError(f"Unsupported image shape for {path}: {arr.shape}")


def normalize01(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Scale finite image intensities to the interval [0, 1] for display."""
    img = np.asarray(img, dtype=np.float32)
    vals = img[np.isfinite(img)]
    if vals.size < 10:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(vals, [float(p_low), float(p_high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def display_percentiles_from_cfg(cfg: dict[str, Any]) -> tuple[float, float]:
    """Read image-display percentile limits from widget settings."""
    value = cfg.get("display_percentiles", [1, 99])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return 1.0, 99.0


def display_gray(img: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Normalize an image for grayscale Napari display without affecting saved data."""
    p_low, p_high = display_percentiles_from_cfg(cfg)
    return normalize01(img, p_low=p_low, p_high=p_high)


def make_rgb_overlay(reference: np.ndarray, moving: np.ndarray, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Construct a green/magenta reference-versus-moving diagnostic overlay."""
    cfg = cfg or {}
    ref_n = display_gray(reference, cfg)
    mov_n = display_gray(moving, cfg)
    rgb = np.zeros((*ref_n.shape, 3), dtype=np.float32)
    # reference in green, moving in magenta. Good overlap becomes light/white.
    rgb[..., 1] = ref_n
    rgb[..., 0] = mov_n
    rgb[..., 2] = mov_n
    return rgb


def apply_tform(image: np.ndarray, tform: EuclideanTransform, *, order: int = 1, mode: str = "constant", cval: float = 0.0) -> np.ndarray:
    """Warp an image with an explicitly supplied Euclidean transform."""
    return warp(
        image.astype(np.float32),
        tform.inverse,
        order=int(order),
        preserve_range=True,
        mode=str(mode),
        cval=float(cval),
    ).astype(np.float32)


def save_bmp_uint8(path: Path, image: np.ndarray) -> None:
    """Clip an array to 8-bit intensity values and save a BMP."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.clip(image, 0, 255).astype(np.uint8))


def _identity_affine_yx() -> np.ndarray:
    """Return a 3-by-3 identity affine transform in (y, x) coordinates."""
    return np.eye(3, dtype=np.float64)


def _layer_world_affine_yx(layer: Any) -> np.ndarray:
    """Extract a Napari image layer's 2D affine in (y, x) coordinates.
    
    Parameters
    ----------
    layer : napari.layers.Image
        Image layer being manipulated through Napari's 2D transform tool.
    
    Returns
    -------
    numpy.ndarray
        Homogeneous ``(3, 3)`` affine matrix mapping layer data coordinates
        to world/reference coordinates in ``(y, x)`` order.
    
    Notes
    -----
    The implementation samples ``data_to_world`` at the origin and basis
    vectors. A property-based fallback is used for older Napari releases.
    """
    try:
        pts = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        world = np.asarray([layer.data_to_world(tuple(p)) for p in pts], dtype=np.float64)
        if world.ndim == 2 and world.shape[0] == 3 and world.shape[1] >= 2:
            w0 = world[0, :2]
            wy = world[1, :2]
            wx = world[2, :2]
            mat = np.asarray([
                [wy[0] - w0[0], wx[0] - w0[0], w0[0]],
                [wy[1] - w0[1], wx[1] - w0[1], w0[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            if np.all(np.isfinite(mat)):
                return mat
    except Exception:
        pass

    # Conservative fallback for older napari variants.
    mat = _identity_affine_yx()
    try:
        scale = np.asarray(getattr(layer, "scale", [1.0, 1.0]), dtype=np.float64)[:2]
        mat[0, 0] = float(scale[0])
        mat[1, 1] = float(scale[1])
    except Exception:
        pass
    try:
        rotate = float(getattr(layer, "rotate", 0.0) or 0.0)
        theta = np.deg2rad(rotate)
        rot = np.asarray([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
        mat = rot @ mat
    except Exception:
        pass
    try:
        tr = np.asarray(getattr(layer, "translate", [0.0, 0.0]), dtype=np.float64)[:2]
        mat[0, 2] = float(tr[0])
        mat[1, 2] = float(tr[1])
    except Exception:
        pass
    return mat


def _affine_yx_is_identity(mat_yx: np.ndarray, atol: float = 1e-6) -> bool:
    """Test whether a spatial affine matrix is approximately the identity."""
    try:
        return bool(np.allclose(np.asarray(mat_yx, dtype=float), np.eye(3), atol=float(atol)))
    except Exception:
        return False


def _affine_yx_to_xy(mat_yx: np.ndarray) -> np.ndarray:
    """Convert an affine transform from (y, x) to (x, y) convention."""
    m = np.asarray(mat_yx, dtype=np.float64)
    return np.asarray([
        [m[1, 1], m[1, 0], m[1, 2]],
        [m[0, 1], m[0, 0], m[0, 2]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _apply_layer_affine_to_image(
    image: np.ndarray,
    mat_yx: np.ndarray,
    *,
    output_shape: tuple[int, int],
    cfg: dict[str, Any],
) -> np.ndarray:
    """Rasterize the active Napari layer transform on the reference image grid.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image with shape ``(y, x)``.
    mat_yx : numpy.ndarray
        Homogeneous ``(3, 3)`` affine mapping layer data coordinates
        ``(y, x, 1)`` to reference/world coordinates.
    output_shape : tuple of int
        Target spatial dimensions ``(y, x)``.
    cfg : dict
        Manual-registration configuration; interpolation and edge settings.
    
    Returns
    -------
    numpy.ndarray
        Warped ``float32`` image with the reference grid's shape.
    
    Notes
    -----
    The matrix is converted to OpenCV ``(x, y)`` ordering before
    ``warpAffine``. This image is used for saved manual overrides/QC.
    """
    img = np.asarray(image, dtype=np.float32)
    if _affine_yx_is_identity(mat_yx):
        return img.copy()
    h, w = int(output_shape[0]), int(output_shape[1])
    mat_xy = _affine_yx_to_xy(mat_yx)[:2, :]
    interp = int(cfg.get("interpolation_order", 1))
    flags = cv2.INTER_NEAREST if interp == 0 else cv2.INTER_LINEAR
    edge_mode = str(cfg.get("edge_mode", "constant")).lower()
    border_mode = cv2.BORDER_REPLICATE if edge_mode in {"edge", "nearest", "replicate"} else cv2.BORDER_CONSTANT
    cval = float(cfg.get("edge_cval", 0.0))
    return cv2.warpAffine(img, mat_xy.astype(np.float32), (w, h), flags=flags, borderMode=border_mode, borderValue=cval).astype(np.float32)


def _matrix_json(mat: np.ndarray) -> str:
    """Serialize a spatial transform matrix to JSON."""
    return json.dumps(np.asarray(mat, dtype=float).tolist(), ensure_ascii=False)


def _matrix_from_json(value: Any) -> np.ndarray | None:
    """Decode a stored valid finite 3-by-3 transform matrix."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        arr = np.asarray(json.loads(text), dtype=np.float64)
        if arr.shape == (3, 3) and np.all(np.isfinite(arr)):
            return arr
    except Exception:
        return None
    return None


def _manual_registered_path_from_override(override: pd.Series | None) -> Path | None:
    """Find the saved corrected BMP associated with an override."""
    if override is None:
        return None
    try:
        value = override.get("manual_registered_path", "")
    except Exception:
        return None
    if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
        return None
    path = Path(str(value))
    return path if path.exists() else None


def _layer_transform_summary(mat_yx: np.ndarray) -> tuple[float, float, float]:
    """Extract an approximate translation and angle from a layer affine."""
    m = np.asarray(mat_yx, dtype=float)
    ty = float(m[0, 2])
    tx = float(m[1, 2])
    # Convert to xy and estimate rotation from its linear part.
    mxy = _affine_yx_to_xy(m)
    rot = float(np.degrees(np.arctan2(mxy[1, 0], mxy[0, 0])))
    return ty, tx, rot


def _transform_summary_xy(mat_xy: np.ndarray) -> tuple[float, float, float, float]:
    """Extract translation and rotation from an affine in (x, y) coordinates."""
    m = np.asarray(mat_xy, dtype=np.float64)
    tx = float(m[0, 2])
    ty = float(m[1, 2])
    rot_rad = float(np.arctan2(m[1, 0], m[0, 0]))
    return tx, ty, rot_rad, float(np.degrees(rot_rad))


def _final_transform_from_override(override: pd.Series | None) -> np.ndarray | None:
    """Reconstruct a saved total moving-to-reference spatial transform."""
    if override is None:
        return None
    mat = _matrix_from_json(override.get("final_transform_xy_json", ""))
    if mat is not None:
        return np.asarray(mat, dtype=np.float64)
    try:
        tx = _num(override.get("final_translation_x", np.nan), np.nan)
        ty = _num(override.get("final_translation_y", np.nan), np.nan)
        rot = _num(override.get("final_rotation_rad", np.nan), np.nan)
        if np.isfinite(tx) and np.isfinite(ty) and np.isfinite(rot):
            return np.asarray(EuclideanTransform(rotation=float(rot), translation=(float(tx), float(ty))).params, dtype=np.float64)
    except Exception:
        pass
    return None


def _reset_layer_transform(layer: Any) -> None:
    """Reset a Napari image layer transform to the identity."""
    for attr, value in [
        ("scale", (1.0, 1.0)),
        ("translate", (0.0, 0.0)),
        ("rotate", 0.0),
        ("shear", (0.0,)),
    ]:
        try:
            setattr(layer, attr, value)
        except Exception:
            pass
    try:
        # Some napari versions expose an Affine object.  Do not fail if absent.
        if hasattr(layer, "affine") and hasattr(layer.affine, "affine_matrix"):
            layer.affine.affine_matrix = np.eye(3)
    except Exception:
        pass


def _set_layer_affine_yx(layer: Any, mat_yx: np.ndarray | None) -> bool:
    """Attempt to restore a previously saved Napari layer transform."""
    if mat_yx is None:
        _reset_layer_transform(layer)
        return True
    try:
        mat = np.asarray(mat_yx, dtype=np.float64)
        if mat.shape != (3, 3) or not np.all(np.isfinite(mat)):
            _reset_layer_transform(layer)
            return False
    except Exception:
        _reset_layer_transform(layer)
        return False

    _reset_layer_transform(layer)
    if _affine_yx_is_identity(mat):
        return True

    # Preferred route: set the napari Affine object directly.  This preserves
    # the full Transform 2D matrix, including rotations made with the layer
    # controls.
    try:
        if hasattr(layer, "affine") and hasattr(layer.affine, "affine_matrix"):
            layer.affine.affine_matrix = mat
            return True
    except Exception:
        pass

    try:
        # Some napari versions accept direct assignment.
        layer.affine = mat
        return True
    except Exception:
        pass

    # Conservative fallback: restore approximate translation/rotation.
    try:
        ty, tx, rot_deg = _layer_transform_summary(mat)
        setattr(layer, "translate", (float(ty), float(tx)))
        setattr(layer, "rotate", float(rot_deg))
        return True
    except Exception:
        return False


def _num(value: Any, default: float = 0.0) -> float:
    """Convert a value to float or return the configured fallback."""
    v = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else float(default)


def auto_transform_from_row(row: pd.Series) -> EuclideanTransform:
    """Reconstruct the original Step2 rigid transform for one stem."""
    stem = str(row.get("stem", ""))
    ref_stem = str(row.get("ref_stem", ""))
    if stem and ref_stem and stem == ref_stem:
        return EuclideanTransform()
    tx = _num(row.get("translation_x", 0.0), 0.0)
    ty = _num(row.get("translation_y", 0.0), 0.0)
    rot = _num(row.get("rotation_rad", 0.0), 0.0)
    return EuclideanTransform(rotation=rot, translation=(tx, ty))


def rotation_center_yx(row: pd.Series, image_shape: tuple[int, int], mode: str) -> tuple[float, float, str]:
    """Select the bregma or image center for manual rotation."""
    mode = str(mode or "bregma_ref").strip().lower()
    if mode in {"bregma_ref", "ref_bregma", "bregma"}:
        cy = _num(row.get("bregma_ref_cy", np.nan), np.nan)
        cx = _num(row.get("bregma_ref_cx", np.nan), np.nan)
        if np.isfinite(cy) and np.isfinite(cx):
            return float(cy), float(cx), "bregma_ref"
    h, w = int(image_shape[0]), int(image_shape[1])
    return (h - 1) / 2.0, (w - 1) / 2.0, "image_center"


def manual_delta_transform(
    *,
    dx: float,
    dy: float,
    angle_deg: float,
    center_yx: tuple[float, float],
) -> EuclideanTransform:
    """Construct an incremental rigid correction around a chosen center.
    
    Parameters
    ----------
    dx, dy : float
        Manual translation components in rotated full-resolution pixels.
    angle_deg : float
        Incremental rotation in degrees.
    center_yx : tuple of float
        Center ``(y, x)`` in the current reference image pixel grid.
    
    Returns
    -------
    skimage.transform.EuclideanTransform
        Spatial transform representing the requested manual adjustment.
    
    Notes
    -----
    This is a manual correction, not a replacement of Step2's underlying
    automatic registration metadata.
    """
    cy, cx = float(center_yx[0]), float(center_yx[1])
    theta = np.deg2rad(float(angle_deg))
    rot = (
        EuclideanTransform(translation=(-cx, -cy))
        + EuclideanTransform(rotation=theta)
        + EuclideanTransform(translation=(cx, cy))
    )
    shift = EuclideanTransform(translation=(float(dx), float(dy)))
    # Applies rotation around the chosen reference-space center, then translation.
    return rot + shift


def final_transform(row: pd.Series, image_shape: tuple[int, int], *, dx: float, dy: float, angle_deg: float, center_mode: str) -> tuple[EuclideanTransform, tuple[float, float, str]]:
    """Compose an automatic Step2 transform with a manual correction.
    
    Parameters
    ----------
    row : pandas.Series
        Step2 registration metadata with translation and rotation fields.
    image_shape : tuple of int
        Spatial shape ``(y, x)`` of the registered reference image.
    dx, dy : float
        Manual translation components, in full-resolution pixels.
    angle_deg : float
        Manual rotation in degrees.
    center_mode : str
        Rotation-center selector (for example, configured ``bregma_ref``
        or ``image_center``).
    
    Returns
    -------
    tuple
        Combined ``EuclideanTransform`` and the chosen center information.
    
    Notes
    -----
    This helper expresses the total alignment in Step2 reference-image
    coordinates, preserving the existing transform composition order.
    """
    auto = auto_transform_from_row(row)
    cy, cx, resolved_center = rotation_center_yx(row, image_shape, center_mode)
    delta = manual_delta_transform(dx=dx, dy=dy, angle_deg=angle_deg, center_yx=(cy, cx))
    # skimage transform composition: A + B applies A first, then B.
    # Here manual correction is applied on top of Step2 auto registration.
    return auto + delta, (cy, cx, resolved_center)


def load_overrides_table(path: Path) -> pd.DataFrame:
    """Load manual registration overrides, ensuring the required columns."""
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=OVERRIDE_COLUMNS)
    for col in OVERRIDE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col.endswith(("_x", "_y", "_rad", "_deg")) else ""
    return df[OVERRIDE_COLUMNS + [c for c in df.columns if c not in OVERRIDE_COLUMNS]]


def latest_override_for_stem(path: Path, stem: str) -> pd.Series | None:
    """Retrieve the newest saved manual registration override for a stem."""
    df = load_overrides_table(path)
    if df.empty or "stem" not in df.columns:
        return None
    sub = df[df["stem"].astype(str).eq(str(stem))].copy()
    if sub.empty:
        return None
    if "updated_at" in sub.columns:
        sub = sub.sort_values("updated_at")
    return sub.iloc[-1]


def upsert_override(path: Path, row: dict[str, Any]) -> None:
    """Insert or replace the persisted manual override for a recording."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = load_overrides_table(path)
    # Remove older rows for the same group/stem.  The JSON file is the detailed audit trail.
    if not df.empty and "group_id" in df.columns and "stem" in df.columns:
        keep = ~(
            df["group_id"].astype(str).eq(str(row.get("group_id", "")))
            & df["stem"].astype(str).eq(str(row.get("stem", "")))
        )
        df = df[keep].copy()
    full = {col: row.get(col, "") for col in OVERRIDE_COLUMNS}
    df = pd.DataFrame.from_records([*df.to_dict("records"), full], columns=df.columns)
    df.to_csv(path, index=False)


class ManualRegistrationPanel(QWidget):
    def __init__(
        self,
        *,
        viewer: napari.Viewer,
        df_group: pd.DataFrame,
        config: dict[str, Any],
        paths: dict[str, Path],
        manual_paths: dict[str, Path],
        annotator: str,
    ):
        """Initialize the interactive widget and its reference/moving Napari layers."""
        super().__init__()
        self.viewer = viewer
        self.df_group = df_group.reset_index(drop=True)
        self.config = config
        self.cfg = manual_registration_config(config)
        self.paths = paths
        self.manual_paths = manual_paths
        self.annotator = annotator
        self.index = 0
        self.reference_img: np.ndarray | None = None
        self.moving_img: np.ndarray | None = None
        self.moved_img: np.ndarray | None = None
        self._loading = False
        self._session_overrides: dict[str, pd.Series] = {}
        self._edit_base_is_registered = False
        self._edit_base_source = "moving_registration_path"
        self._edit_base_final_xy: np.ndarray | None = None

        non_ref = [i for i, r in self.df_group.iterrows() if str(r.get("stem", "")) != str(r.get("ref_stem", ""))]
        if non_ref:
            self.index = int(non_ref[0])

        self.reference_layer = viewer.add_image(
            np.zeros((2, 2), dtype=np.float32),
            name="reference",
            colormap="gray",
            contrast_limits=(0, 1),
            opacity=float(self.cfg.get("reference_opacity", 1.0)),
            visible=bool(self.cfg.get("reference_visible", True)),
        )
        self.moving_layer = viewer.add_image(
            np.zeros((2, 2), dtype=np.float32),
            name="moving_manual_registered",
            colormap="gray",
            contrast_limits=(0, 1),
            opacity=float(self.cfg.get("moving_opacity", 0.55)),
            visible=bool(self.cfg.get("moving_visible", True)),
        )
        self._keep_moving_layer_active()

        self._build_ui()
        self.load_current_stem()

    def _build_ui(self) -> None:
        """Create controls for incremental transforms, navigation, and saving."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title = QLabel("")
        self.title.setWordWrap(True)
        layout.addWidget(self.title)

        self.stem_combo = QComboBox()
        for i, row in self.df_group.iterrows():
            label = f"{i}: {row.get('stem', '')}"
            if str(row.get("stem", "")) == str(row.get("ref_stem", "")):
                label += " [reference]"
            self.stem_combo.addItem(label, int(i))
        self.stem_combo.currentIndexChanged.connect(self._combo_changed)
        layout.addWidget(self.stem_combo)

        def spin(label: str, step: float, decimals: int, value_range: tuple[float, float]) -> QDoubleSpinBox:
            """Create a labeled numeric control for incremental transform editing."""
            row = QHBoxLayout()
            lab = QLabel(label)
            box = QDoubleSpinBox()
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setRange(float(value_range[0]), float(value_range[1]))
            box.valueChanged.connect(self.update_preview)
            row.addWidget(lab)
            row.addWidget(box)
            layout.addLayout(row)
            return box

        self.dx_box = spin("edit dx px (incremental)", 0.5, 3, (-300, 300))
        self.dy_box = spin("edit dy px (incremental)", 0.5, 3, (-300, 300))
        self.angle_box = spin("edit angle deg (incremental)", 0.05, 4, (-30, 30))

        self.center_combo = QComboBox()
        self.center_combo.addItems(["bregma_ref", "image_center"])
        self.center_combo.currentTextChanged.connect(self.update_preview)
        center_default = str(self.cfg.get("rotation_center", "bregma_ref"))
        if center_default in {"bregma_ref", "image_center"}:
            self.center_combo.setCurrentText(center_default)
        center_row = QHBoxLayout()
        center_row.addWidget(QLabel("rotation center"))
        center_row.addWidget(self.center_combo)
        layout.addLayout(center_row)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("moving opacity"))
        self.moving_opacity_box = QDoubleSpinBox()
        self.moving_opacity_box.setDecimals(2)
        self.moving_opacity_box.setSingleStep(0.05)
        self.moving_opacity_box.setRange(0.0, 1.0)
        self.moving_opacity_box.setValue(float(self.cfg.get("moving_opacity", 0.55)))
        self.moving_opacity_box.valueChanged.connect(self.set_moving_opacity)
        opacity_row.addWidget(self.moving_opacity_box)
        layout.addLayout(opacity_row)

        self.notes = QLineEdit()
        self.notes.setPlaceholderText("notes, optional")
        layout.addWidget(self.notes)

        buttons1 = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Next")
        self.prev_btn.clicked.connect(self.prev_stem)
        self.next_btn.clicked.connect(self.next_stem)
        buttons1.addWidget(self.prev_btn)
        buttons1.addWidget(self.next_btn)
        layout.addLayout(buttons1)

        buttons2 = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to auto")
        self.copy_prev_btn = QPushButton("Copy previous delta")
        self.save_btn = QPushButton("Save current override")
        self.reset_btn.clicked.connect(self.reset_to_auto)
        self.copy_prev_btn.clicked.connect(self.copy_previous_delta)
        self.save_btn.clicked.connect(self.save_current_override)
        buttons2.addWidget(self.reset_btn)
        buttons2.addWidget(self.copy_prev_btn)
        buttons2.addWidget(self.save_btn)
        layout.addLayout(buttons2)

        self.final_transform_label = QLabel("")
        self.final_transform_label.setWordWrap(True)
        layout.addWidget(self.final_transform_label)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def _combo_changed(self, *args: Any) -> None:
        """Load a new stem when the user changes the selection."""
        if self._loading:
            return
        self.index = int(self.stem_combo.currentData())
        self.load_current_stem()

    def current_row(self) -> pd.Series:
        """Return the metadata row for the currently selected stem."""
        return self.df_group.iloc[int(self.index)]

    def latest_override(self, stem: str) -> pd.Series | None:
        """Look up the latest override, including unsaved session state."""
        if bool(self.cfg.get("remember_overrides_in_session", True)) and str(stem) in self._session_overrides:
            return self._session_overrides[str(stem)]
        return latest_override_for_stem(self.manual_paths["manual_registration_overrides_csv"], stem)

    def _restore_saved_layer_transform(self, override: pd.Series | None) -> None:
        """Restore a stored Napari layer affine for the current stem."""
        if not bool(self.cfg.get("restore_napari_layer_transform_in_viewer", True)):
            _reset_layer_transform(self.moving_layer)
            self._keep_moving_layer_active()
            return
        mat = None
        if override is not None:
            mat = _matrix_from_json(override.get("napari_layer_affine_yx_json", ""))
        ok = _set_layer_affine_yx(self.moving_layer, mat)
        self._keep_moving_layer_active()
        if override is not None and mat is not None and not ok:
            self.status.setText(self.status.text() + "\n[WARN] Could not restore saved napari layer transform in edit viewer.")

    def _set_spin_values(self, dx: float, dy: float, angle: float) -> None:
        """Update incremental translation and rotation controls."""
        for box, val in [(self.dx_box, dx), (self.dy_box, dy), (self.angle_box, angle)]:
            box.blockSignals(True)
            box.setValue(float(val))
            box.blockSignals(False)

    def load_current_stem(self) -> None:
        """Load the selected recording and its current reference-aligned image."""
        self._loading = True
        row = self.current_row()
        stem = str(row.get("stem", ""))
        group_id = str(row.get("group_id", ""))
        ref_stem = str(row.get("ref_stem", ""))
        self.title.setText(f"Group: {group_id}\nStem: {stem}\nReference: {ref_stem}")

        fp = fixed_image_path(row)
        mp = moving_image_path(row)
        self.reference_img = read_gray(fp)
        original_moving = read_gray(mp)
        if self.reference_img.shape != original_moving.shape:
            raise ValueError(f"Image shape mismatch for stem={stem}: fixed={self.reference_img.shape}, moving={original_moving.shape}")

        _reset_layer_transform(self.moving_layer)
        self._edit_base_is_registered = False
        self._edit_base_source = "moving_registration_path"
        self._edit_base_final_xy = np.asarray(auto_transform_from_row(row).params, dtype=np.float64)

        override = self.latest_override(stem)
        saved_manual_path = _manual_registered_path_from_override(override)
        use_saved_as_base = bool(self.cfg.get("load_saved_bmp_in_viewer", True)) and saved_manual_path is not None

        if override is not None:
            notes = str(override.get("notes", "")) if pd.notna(override.get("notes", "")) else ""
            center = str(override.get("rotation_center_mode", self.cfg.get("rotation_center", "bregma_ref")))
            self.notes.setText(notes)
            if center in {"bregma_ref", "image_center"}:
                self.center_combo.setCurrentText(center)
        else:
            self.notes.setText("")
            self.center_combo.setCurrentText(str(self.cfg.get("rotation_center", "bregma_ref")))

        if use_saved_as_base:
            saved_img = read_gray(saved_manual_path)  # already on reference grid
            if saved_img.shape != self.reference_img.shape:
                raise ValueError(
                    f"Saved manual BMP shape mismatch for stem={stem}: manual={saved_img.shape}, reference={self.reference_img.shape}\n"
                    f"manual_registered_path={saved_manual_path}"
                )
            self.moving_img = saved_img
            self._edit_base_is_registered = True
            self._edit_base_source = "manual_registered_path"
            self._edit_base_final_xy = _final_transform_from_override(override)
            # The saved image is already corrected. Further edits are incremental
            # and should start from zero so returning to a stem does not reapply
            # Step2 auto or previous dx/dy/angle a second time.
            dx = dy = angle = 0.0
            restore_layer_transform = False
        else:
            self.moving_img = original_moving
            self._edit_base_is_registered = False
            self._edit_base_source = "moving_registration_path"
            self._edit_base_final_xy = np.asarray(auto_transform_from_row(row).params, dtype=np.float64)
            if override is not None:
                dx = _num(override.get("manual_delta_x", 0.0), 0.0)
                dy = _num(override.get("manual_delta_y", 0.0), 0.0)
                angle = _num(override.get("manual_delta_angle_deg", 0.0), 0.0)
                restore_layer_transform = True
            else:
                dx = dy = angle = 0.0
                restore_layer_transform = False

        self._set_spin_values(dx, dy, angle)
        combo_idx = self.stem_combo.findData(int(self.index))
        if combo_idx >= 0:
            self.stem_combo.blockSignals(True)
            self.stem_combo.setCurrentIndex(combo_idx)
            self.stem_combo.blockSignals(False)
        self._loading = False
        self.update_preview()
        if restore_layer_transform:
            self._restore_saved_layer_transform(override)
        else:
            _reset_layer_transform(self.moving_layer)
            self._keep_moving_layer_active()

    def update_preview(self, *args: Any) -> None:
        """Update the moving image preview from the manual correction controls."""
        if self._loading or self.reference_img is None or self.moving_img is None:
            return
        row = self.current_row()
        dx = float(self.dx_box.value())
        dy = float(self.dy_box.value())
        angle = float(self.angle_box.value())
        if bool(self._edit_base_is_registered):
            cy, cx, center_mode = rotation_center_yx(
                row,
                tuple(self.reference_img.shape),
                str(self.center_combo.currentText()),
            )
            tform = manual_delta_transform(dx=dx, dy=dy, angle_deg=angle, center_yx=(cy, cx))
            status_prefix = f"incremental edit on {self._edit_base_source}"
        else:
            tform, (cy, cx, center_mode) = final_transform(
                row,
                tuple(self.reference_img.shape),
                dx=dx,
                dy=dy,
                angle_deg=angle,
                center_mode=str(self.center_combo.currentText()),
            )
            status_prefix = "Step2 auto + manual delta"
        self.moved_img = apply_tform(
            self.moving_img,
            tform,
            order=int(self.cfg.get("interpolation_order", 1)),
            mode=str(self.cfg.get("edge_mode", "constant")),
            cval=float(self.cfg.get("edge_cval", 0.0)),
        )
        ref_disp = display_gray(self.reference_img, self.cfg)
        mov_disp = display_gray(self.moved_img, self.cfg)
        self.reference_layer.data = ref_disp
        self.moving_layer.data = mov_disp
        self._keep_moving_layer_active()
        tx, ty = map(float, tform.translation)
        # These displayed dx/dy/angle values describe the currently edited layer preview.
        # They are not updated live by napari's Transform 2D controls; the total transform
        # including Transform 2D is captured and written when Save current override is pressed.
        self.status.setText(
            f"{status_prefix}\n"
            f"preview edit dx={tx:.3f}, dy={ty:.3f}, rotation={np.degrees(float(tform.rotation)):.4f} deg\n"
            f"manual center: {center_mode} ({cy:.1f}, {cx:.1f})"
        )
        if bool(self.cfg.get("show_final_reference_transform_in_widget", True)):
            try:
                base_final = self._edit_base_final_xy if self._edit_base_final_xy is not None else np.asarray(auto_transform_from_row(row).params, dtype=np.float64)
                if bool(self._edit_base_is_registered):
                    total_preview_xy = np.asarray(tform.params, dtype=np.float64) @ base_final
                else:
                    total_preview_xy = np.asarray(tform.params, dtype=np.float64)
                ftx, fty, frot, frot_deg = _transform_summary_xy(total_preview_xy)
                self.final_transform_label.setText(
                    "final transform to reference, before napari Transform 2D Save:\n"
                    f"x={ftx:.3f}, y={fty:.3f}, rot={frot_deg:.4f} deg ({frot:.6f} rad)"
                )
            except Exception:
                self.final_transform_label.setText("")

    def _keep_moving_layer_active(self) -> None:
        """Keep the moving image layer visible above the reference."""
        try:
            self.viewer.layers.selection.active = self.moving_layer
        except Exception:
            pass

    def set_moving_opacity(self, value: float) -> None:
        """Update the moving layer opacity for visual alignment."""
        self.moving_layer.opacity = float(value)
        self._keep_moving_layer_active()

    def reset_to_auto(self) -> None:
        """Discard the current preview correction and return to automatic alignment."""
        row = self.current_row()
        if self.reference_img is not None:
            self.moving_img = read_gray(moving_image_path(row))
            self._edit_base_is_registered = False
            self._edit_base_source = "moving_registration_path"
            self._edit_base_final_xy = np.asarray(auto_transform_from_row(row).params, dtype=np.float64)
        self._set_spin_values(0.0, 0.0, 0.0)
        _reset_layer_transform(self.moving_layer)
        self.update_preview()

    def copy_previous_delta(self) -> None:
        """Reuse the previous recording correction as the current starting delta."""
        if self.index <= 0:
            return
        prev_idx = max(0, self.index - 1)
        prev_stem = str(self.df_group.iloc[prev_idx].get("stem", ""))
        override = latest_override_for_stem(self.manual_paths["manual_registration_overrides_csv"], prev_stem)
        if override is None:
            return
        self._set_spin_values(
            _num(override.get("manual_delta_x", 0.0), 0.0),
            _num(override.get("manual_delta_y", 0.0), 0.0),
            _num(override.get("manual_delta_angle_deg", 0.0), 0.0),
        )
        self.update_preview()

    def prev_stem(self) -> None:
        """Select the preceding recording in the active group."""
        self.index = max(0, int(self.index) - 1)
        self.load_current_stem()

    def next_stem(self) -> None:
        """Select the following recording in the active group."""
        self.index = min(len(self.df_group) - 1, int(self.index) + 1)
        self.load_current_stem()

    def save_current_override(self) -> None:
        """Save the displayed manual registration and its total spatial transform.
        
        Notes
        -----
        The current Napari layer affine is baked into an image on the reference
        pixel grid, and the saved BMP is reused on subsequent visits to the
        same stem. Override metadata includes the final moving-to-reference
        transform in Step2 reference coordinates.
        """
        if self.reference_img is None or self.moving_img is None:
            return
        if self.moved_img is None:
            self.update_preview()
        if self.moved_img is None:
            return

        row = self.current_row()
        stem = str(row.get("stem", ""))
        group_id = str(row.get("group_id", ""))
        ref_stem = str(row.get("ref_stem", "")) or stem
        dx = float(self.dx_box.value())
        dy = float(self.dy_box.value())
        angle = float(self.angle_box.value())

        if bool(self._edit_base_is_registered):
            cy, cx, center_mode = rotation_center_yx(
                row,
                tuple(self.reference_img.shape),
                str(self.center_combo.currentText()),
            )
            tform = manual_delta_transform(dx=dx, dy=dy, angle_deg=angle, center_yx=(cy, cx))
        else:
            tform, (cy, cx, center_mode) = final_transform(
                row,
                tuple(self.reference_img.shape),
                dx=dx,
                dy=dy,
                angle_deg=angle,
                center_mode=str(self.center_combo.currentText()),
            )

        # Save the exact current viewer state.  The moving layer data already
        # contains the widget dx/dy/angle preview.  Any napari Transform 2D state
        # is baked on top of that displayed image, then the layer is immediately
        # replaced with the baked image and its transform reset to identity.  This
        # avoids the fragile behavior where returning to a stem depends on napari
        # reapplying a layer-control transform.
        if bool(self.cfg.get("capture_napari_layer_transform", True)):
            layer_affine_yx = _layer_world_affine_yx(self.moving_layer)
        else:
            layer_affine_yx = _identity_affine_yx()
        moved = _apply_layer_affine_to_image(
            self.moved_img,
            layer_affine_yx,
            output_shape=tuple(self.reference_img.shape),
            cfg=self.cfg,
        )

        group_dir = self.manual_paths["manual_registered_bmps_root"] / group_id
        json_dir = self.manual_paths["manual_json_root"] / group_id
        manual_bmp = group_dir / f"{stem}_registered_manual_to_{ref_stem}.bmp"
        manual_json = json_dir / f"{stem}_manual_registration_override.json"
        if bool(self.cfg.get("save_registered_bmp", True)):
            save_bmp_uint8(manual_bmp, moved)

        auto_t = auto_transform_from_row(row)
        auto_tx, auto_ty = map(float, auto_t.translation)
        auto_xy = np.asarray(auto_t.params, dtype=np.float64)
        layer_affine_xy = _affine_yx_to_xy(layer_affine_yx)
        edit_xy = np.asarray(tform.params, dtype=np.float64)
        previous_final_xy = self._edit_base_final_xy
        if previous_final_xy is None:
            previous_final_xy = _final_transform_from_override(self.latest_override(stem))
        if previous_final_xy is None:
            previous_final_xy = auto_xy
        previous_final_xy = np.asarray(previous_final_xy, dtype=np.float64)

        # final_xy is always original moving-image coordinates -> reference-image coordinates,
        # i.e. the same coordinate system as Step2 translation_x / translation_y / rotation_rad.
        # When editing starts from an already saved manual BMP, edit_xy/layer_affine_xy are
        # incremental reference-grid corrections, so they must be composed on top of the
        # previously saved final transform instead of being interpreted as deltas from auto.
        if bool(self._edit_base_is_registered):
            final_xy = layer_affine_xy @ edit_xy @ previous_final_xy
        else:
            final_xy = layer_affine_xy @ edit_xy
        final_tx, final_ty, final_rot, final_rot_deg = _transform_summary_xy(final_xy)
        auto_rot_deg = float(np.degrees(float(auto_t.rotation)))
        manual_total_delta_x = float(final_tx - auto_tx)
        manual_total_delta_y = float(final_ty - auto_ty)
        manual_total_delta_angle_deg = float(final_rot_deg - auto_rot_deg)
        layer_ty, layer_tx, layer_rot_deg = _layer_transform_summary(layer_affine_yx)
        now = _dt.datetime.now().isoformat()
        record = {
            "group_id": group_id,
            "stem": stem,
            "ref_stem": ref_stem,
            "auto_translation_x": auto_tx,
            "auto_translation_y": auto_ty,
            "auto_rotation_rad": float(auto_t.rotation),
            "manual_delta_x": dx,
            "manual_delta_y": dy,
            "manual_delta_angle_deg": angle,
            "final_translation_x": final_tx,
            "final_translation_y": final_ty,
            "final_rotation_rad": final_rot,
            "final_translation_x_ref": final_tx,
            "final_translation_y_ref": final_ty,
            "final_rotation_deg_ref": final_rot_deg,
            "manual_total_delta_x_from_auto": manual_total_delta_x,
            "manual_total_delta_y_from_auto": manual_total_delta_y,
            "manual_total_delta_angle_deg_from_auto": manual_total_delta_angle_deg,
            "rotation_center_mode": center_mode,
            "rotation_center_y": float(cy),
            "rotation_center_x": float(cx),
            "napari_layer_transform_used": not _affine_yx_is_identity(layer_affine_yx),
            "napari_layer_translate_y": layer_ty,
            "napari_layer_translate_x": layer_tx,
            "napari_layer_rotation_deg": layer_rot_deg,
            "napari_layer_affine_yx_json": _matrix_json(layer_affine_yx),
            "final_transform_xy_json": _matrix_json(final_xy),
            "edit_base_previous_final_transform_xy_json": _matrix_json(previous_final_xy),
            "manual_saved_image_source": "baked_current_viewer_layer",
            "viewer_edit_base_source": self._edit_base_source,
            "viewer_edit_base_already_registered": bool(self._edit_base_is_registered),
            "viewer_baked_after_save": True,
            "fixed_registration_path": str(fixed_image_path(row)),
            "moving_registration_path": str(moving_image_path(row)),
            "manual_registered_path": str(manual_bmp),
            "manual_json_path": str(manual_json),
            "registration_source": "manual_override",
            "annotator": self.annotator,
            "notes": self.notes.text().strip(),
            "updated_at": now,
        }
        manual_json.parent.mkdir(parents=True, exist_ok=True)
        manual_json.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        upsert_override(self.manual_paths["manual_registration_overrides_csv"], record)
        if bool(self.cfg.get("remember_overrides_in_session", True)):
            self._session_overrides[stem] = pd.Series(record)

        if bool(self.cfg.get("bake_saved_override_into_viewer", True)):
            self.moving_img = moved.astype(np.float32, copy=True)
            self.moved_img = moved.astype(np.float32, copy=True)
            self._edit_base_is_registered = True
            self._edit_base_source = "manual_registered_path"
            self._edit_base_final_xy = np.asarray(final_xy, dtype=np.float64)
            self._set_spin_values(0.0, 0.0, 0.0)
            if bool(self.cfg.get("reset_layer_transform_after_save", True)):
                _reset_layer_transform(self.moving_layer)
            self.reference_layer.data = display_gray(self.reference_img, self.cfg)
            self.moving_layer.data = display_gray(self.moved_img, self.cfg)
            self._keep_moving_layer_active()

        self.status.setText(
            self.status.text()
            + f"\nSaved and baked manual override into current viewer state:\n{manual_bmp}\n{self.manual_paths['manual_registration_overrides_csv']}"
            + f"\nNapari layer transform baked: {not _affine_yx_is_identity(layer_affine_yx)}"
            + f"\nFinal transform to reference: x={final_tx:.3f}, y={final_ty:.3f}, rot={final_rot_deg:.4f} deg"
        )
        if bool(self.cfg.get("show_final_reference_transform_in_widget", True)):
            self.final_transform_label.setText(
                "saved final transform to reference:\n"
                f"x={final_tx:.3f}, y={final_ty:.3f}, rot={final_rot_deg:.4f} deg ({final_rot:.6f} rad)"
            )


def registered_image_for_qc(row: pd.Series, config: dict[str, Any], manual_paths: dict[str, Path]) -> np.ndarray:
    """Retrieve the latest manually corrected or automatic registered image."""
    stem = str(row.get("stem", ""))
    ref_stem = str(row.get("ref_stem", ""))
    if stem and ref_stem and stem == ref_stem:
        return read_gray(fixed_image_path(row))

    override = latest_override_for_stem(manual_paths["manual_registration_overrides_csv"], stem)
    cfg = manual_registration_config(config)
    if override is not None:
        # Prefer the saved manual BMP when it exists.  This is essential when
        # the correction was made with napari's Transform 2D layer control,
        # because the exact displayed layer transform is baked into this file.
        manual_path_value = override.get("manual_registered_path", "")
        if bool(cfg.get("read_saved_bmp_in_qc", True)) and pd.notna(manual_path_value) and str(manual_path_value).strip():
            manual_path = Path(str(manual_path_value))
            if manual_path.exists():
                return read_gray(manual_path)

        moving = read_gray(moving_image_path(row))
        dx = _num(override.get("manual_delta_x", 0.0), 0.0)
        dy = _num(override.get("manual_delta_y", 0.0), 0.0)
        angle = _num(override.get("manual_delta_angle_deg", 0.0), 0.0)
        center_mode = str(override.get("rotation_center_mode", cfg.get("rotation_center", "bregma_ref")))
        tform, _ = final_transform(
            row,
            tuple(moving.shape),
            dx=dx,
            dy=dy,
            angle_deg=angle,
            center_mode=center_mode,
        )
        moved = apply_tform(
            moving,
            tform,
            order=int(cfg.get("interpolation_order", 1)),
            mode=str(cfg.get("edge_mode", "constant")),
            cval=float(cfg.get("edge_cval", 0.0)),
        )
        mat_yx = _matrix_from_json(override.get("napari_layer_affine_yx_json", ""))
        if mat_yx is not None:
            moved = _apply_layer_affine_to_image(moved, mat_yx, output_shape=tuple(moved.shape), cfg=cfg)
        return moved

    saved = saved_registered_image_path(row)
    if saved is not None:
        return read_gray(saved)

    moving = read_gray(moving_image_path(row))
    auto = auto_transform_from_row(row)
    return apply_tform(
        moving,
        auto,
        order=int(cfg.get("interpolation_order", 1)),
        mode=str(cfg.get("edge_mode", "constant")),
        cval=float(cfg.get("edge_cval", 0.0)),
    )


def build_group_registered_stack_for_qc(
    df_group: pd.DataFrame,
    config: dict[str, Any],
    manual_paths: dict[str, Path],
) -> tuple[np.ndarray, list[str]]:
    """Assemble a group image stack with saved manual corrections applied."""
    images: list[np.ndarray] = []
    labels: list[str] = []
    cfg = manual_registration_config(config)
    for _, row in df_group.iterrows():
        img = registered_image_for_qc(row, config, manual_paths)
        images.append(display_gray(img, cfg))
        labels.append(str(row.get("stem", "")))
    shapes = {tuple(im.shape) for im in images}
    if len(shapes) != 1:
        raise ValueError(f"QC registered images have different shapes: {sorted(shapes)}")
    return np.stack(images, axis=0).astype(np.float32), labels


def show_group_registered_stack_qc(
    *,
    df_group: pd.DataFrame,
    config: dict[str, Any],
    manual_paths: dict[str, Path],
    group_id: str,
    ref_stem: str,
) -> None:
    """Display the final registered group stack for interactive visual QC."""
    stack, labels = build_group_registered_stack_for_qc(df_group, config, manual_paths)
    viewer = napari.Viewer(title=f"Manual registration QC stack: {group_id} -> {ref_stem}")
    layer = viewer.add_image(
        stack,
        name="registered_stack_manual_over_auto",
        colormap=str(manual_registration_config(config).get("qc_stack_cmap", "gray")),
        contrast_limits=(0, 1),
    )
    layer.axis_labels = ("stem_index", "y", "x")
    print()
    print("Manual registration QC stack")
    print("group_id:", group_id)
    print("ref_stem:", ref_stem)
    print("z-index -> stem")
    for i, label in enumerate(labels):
        print(f"  {i}: {label}")
    print("Close napari after checking the final registered stack.")
    napari.run()


def _normalise_manual_qc_answer(text: str, default: str = "ok") -> str:
    """Normalize English and legacy Russian manual QC responses."""
    ans = str(text or "").strip().lower()
    if not ans:
        return default
    if ans in {"y", "yes", "ok", "good", "да", "д", "+", "1"}:
        return "ok"
    if ans in {"n", "no", "bad", "redo", "плохо", "нет", "н", "-", "0"}:
        return "redo"
    if ans in {"q", "quit", "exit", "stop", "стоп", "выход"}:
        return "quit"
    return ans


def ask_manual_registration_group_qc(
    *,
    group_id: str,
    ref_stem: str,
    n_records: int,
    paths: dict[str, Path],
    config: dict[str, Any],
    annotator: str,
) -> str:
    """Request an accept, redo, or quit decision for a manually aligned group.
    
    Parameters
    ----------
    group_id, ref_stem : str
        Group and reference recording identifiers.
    n_records : int
        Group recording count.
    paths : dict
        Standard project path mapping.
    config : dict
        Full configuration, including ``step3.registration_qc``.
    annotator : str
        Operator recorded for the QC decision.
    
    Returns
    -------
    str
        One of ``'ok'``, ``'redo'``, or ``'quit'``.
    
    Notes
    -----
    A new row is appended to the group-level registration QC CSV;
    legacy Russian responses remain accepted.
    """
    print()
    print("Manual registration final QC")
    print(f"group_id: {group_id}")
    print(f"ref_stem: {ref_stem}")
    print("Is the manually corrected image stack acceptable?")
    print("  [Enter/y/yes] = accept the corrected group and proceed")
    print("  [n/no/bad/redo] = reopen this group for further editing")
    print("  [q/quit] = exit the manual registration widget")
    answer = _normalise_manual_qc_answer(input("Answer [ok]: "), default="ok")

    if answer == "ok":
        registration_qc = "manual_ok"
        needs_manual = False
        notes = "manual registration accepted after QC stack"
    elif answer == "redo":
        registration_qc = "manual_redo"
        needs_manual = True
        notes = input("Comment/remaining issue (press Enter to skip): ").strip()
    elif answer == "quit":
        registration_qc = "manual_quit"
        needs_manual = True
        notes = "user stopped during manual-registration QC"
    else:
        registration_qc = str(answer)
        needs_manual = True
        notes = "unrecognized manual QC answer saved as-is"

    qc_csv = registration_qc_csv_path(paths["step3_root"], config.get("step3", {}) or {})
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
        "review_ok": True,
        "annotation_ok": True,
        "error": "",
    })
    print(f"Manual registration QC saved: {qc_csv}")
    if answer == "ok":
        return "ok"
    if answer == "quit":
        return "quit"
    return "redo"


def select_groups_for_widget(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    groups: Sequence[str] | None,
    bad_registration_only: bool | None,
) -> list[str]:
    """Select groups for interactive manual registration using QC metadata."""
    if groups:
        wanted = {str(g) for g in groups}
        return [str(g) for g in df["group_id"].dropna().astype(str).unique().tolist() if str(g) in wanted]
    cfg = manual_registration_config(config)
    if bad_registration_only is None:
        bad_registration_only = bool(cfg.get("bad_registration_only", True))
    if bad_registration_only:
        qc_csv = registration_qc_csv_path(paths["step3_root"], config.get("step3", {}) or {})
        bad_groups = latest_bad_registration_groups(qc_csv)
        if not bad_groups:
            raise RuntimeError(f"No groups marked as bad in registration QC table: {qc_csv}")
        existing = set(df["group_id"].dropna().astype(str).unique().tolist())
        return [g for g in bad_groups if g in existing]
    return sorted(df["group_id"].dropna().astype(str).unique().tolist())


def run_manual_registration_widget(
    config_path: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    groups: Sequence[str] | None = None,
    bad_registration_only: bool | None = None,
    annotator: str | None = None,
) -> None:
    """Run interactive manual registration for selected Step2 groups.
    
    Parameters
    ----------
    config_path : str or pathlib.Path
        YAML containing ``step3.manual_registration`` and project paths.
    metadata_csv : str or pathlib.Path or None, optional
        Override for the registration metadata CSV.
    groups : sequence of str or None, optional
        Explicit group identifiers; if absent, groups follow QC selection.
    bad_registration_only : bool or None, optional
        Restrict selection to groups flagged by Step3 QC, defaulting to
        ``step3.manual_registration.bad_registration_only``.
    annotator : str or None, optional
        Operator label for saved manual transforms.
    
    Notes
    -----
    The widget saves per-stem images and override metadata separately
    from Step2 automatic transforms. Optional final group QC can request
    a further edit of the same group. The default group selection depends
    on existing ``registration_qc_groups.csv`` records.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)
    manual_paths = build_manual_registration_paths(config, paths)

    step3_cfg = config.get("step3", {}) or {}
    if annotator is None:
        annotator = str(step3_cfg.get("annotator", "user"))

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
            print(f"[EXCLUDED] Manual registration stem={stem}: excluded_by_config")
        df = df[~df["stem"].astype(str).isin(excluded_stems)].copy()
        if df.empty:
            print("[WARN] No manual-registration records remain after excluded_stems.")
            return
    if "group_id" not in df.columns:
        raise ValueError("metadata_registration.csv must contain group_id")

    path_cfg = get_path_resolving_config(config, "step3")
    df = resolve_metadata_paths(df, path_cfg)

    group_ids = select_groups_for_widget(
        df,
        config=config,
        paths=paths,
        groups=groups,
        bad_registration_only=bad_registration_only,
    )
    print("Manual registration groups:", group_ids)
    print("Overrides CSV:", manual_paths["manual_registration_overrides_csv"])

    for group_id in group_ids:
        df_group = df[df["group_id"].astype(str).eq(str(group_id))].copy()
        if df_group.empty:
            print(f"[WARN] group_id not found: {group_id}")
            continue
        df_group = reorder_reference_first(df_group, config)
        ref_row = choose_reference_row(df_group)
        ref_stem = str(ref_row.get("ref_stem", "")) or str(ref_row.get("stem", ""))

        while True:
            viewer = napari.Viewer(title=f"Manual registration override: {group_id} -> {ref_stem}")
            panel = ManualRegistrationPanel(
                viewer=viewer,
                df_group=df_group,
                config=config,
                paths=paths,
                manual_paths=manual_paths,
                annotator=str(annotator),
            )
            viewer.window.add_dock_widget(panel, name="manual registration controls", area="right")
            print()
            print("=" * 80)
            print(f"Manual registration widget group_id={group_id}, ref_stem={ref_stem}")
            print("Use napari eye icons to compare the two grayscale layers: reference under moving_manual_registered.")
            print("You may adjust dx/dy/angle in the widget OR use napari Layer Controls → Transform 2D on moving_manual_registered.")
            print("Click 'Save current override' for each corrected stem; Save bakes the current displayed moving layer into a BMP and keeps that baked correction visible when you return to the stem. Numeric final_translation_x/y/final_rotation_rad are saved as the total transform to the reference image, in the same coordinate system as Step2 auto registration.")
            napari.run()

            if not bool(manual_registration_config(config).get("qc_after_group", True)):
                break
            show_group_registered_stack_qc(
                df_group=df_group,
                config=config,
                manual_paths=manual_paths,
                group_id=str(group_id),
                ref_stem=str(ref_stem),
            )
            action = ask_manual_registration_group_qc(
                group_id=str(group_id),
                ref_stem=str(ref_stem),
                n_records=len(df_group),
                paths=paths,
                config=config,
                annotator=str(annotator),
            )
            if action == "ok":
                break
            if action == "quit":
                return
            print("Reopening the same group for another manual-registration pass.")


def main() -> None:
    """Parse CLI arguments for the manual registration widget."""
    parser = argparse.ArgumentParser(description="Step3 manual registration override widget v1_4")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument("--metadata", default=None, help="Override metadata_registration.csv path")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group_id list. If omitted, bad-registration groups from Step3 QC are used by default.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--bad-registration-only", action="store_true", help="Open only latest groups marked bad in Step3 registration_qc_groups.csv")
    group.add_argument("--all-groups", action="store_true", help="Open all groups from metadata_registration.csv")
    parser.add_argument("--annotator", default=None)
    args = parser.parse_args()

    run_manual_registration_widget(
        config_path=args.config,
        metadata_csv=args.metadata,
        groups=args.groups,
        bad_registration_only=True if args.bad_registration_only else (False if args.all_groups else None),
        annotator=args.annotator,
    )


if __name__ == "__main__":
    main()
