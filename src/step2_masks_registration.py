from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import h5py
import imageio.v3 as iio
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from scipy.ndimage import binary_fill_holes, gaussian_filter
from skimage.filters import sobel
from skimage.measure import label, regionprops
from skimage.morphology import closing, dilation, erosion, disk
from skimage.registration import phase_cross_correlation
from skimage.transform import EuclideanTransform, warp

from common import (
    load_config,
    create_or_load_project,
    build_standard_paths,
    get_stem_keys,
    build_group_id,
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


# ============================================================
# IO helpers
# ============================================================

def _as_path(value: Any) -> Path:
    """Convert a nonmissing scalar to a Path, returning an empty Path for NaN."""
    if pd.isna(value):
        return Path("")
    return Path(str(value))


def resolve_optional_path(value: Any, path_cfg: dict[str, Any]) -> Path | None:
    """Resolve an optional Step1 input using the same root/drive rules as common."""
    if pd.isna(value) or not str(value).strip():
        return None
    return resolve_flexible_path(
        value,
        from_drive=path_cfg.get("from_drive"),
        to_drive=path_cfg.get("to_drive"),
        from_root=path_cfg.get("from_root"),
        to_root=path_cfg.get("to_root"),
        must_exist=False,
    )


def read_probs_h5(h5_path: Path, dataset: str = "exported_data") -> np.ndarray:
    """Read an Ilastik probability map from an HDF5 dataset.

    Parameters
    ----------
    h5_path : pathlib.Path
        Path to an Ilastik ``*_Probabilities.h5`` file.
    dataset : str, default='exported_data'
        HDF5 dataset name; configured by ``step2.h5.dataset`` in the runner.

    Returns
    -------
    numpy.ndarray
        Probability values with shape ``(y, x, class)``; a leading singleton
        axis in an array of shape ``(1, y, x, class)`` is removed.

    Raises
    ------
    FileNotFoundError
        If the HDF5 file does not exist.
    KeyError
        If the requested dataset is absent.
    ValueError
        If the resulting array is not three-dimensional.
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Ilastik probability file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset {dataset!r} not found in {h5_path}. Available: {list(f.keys())}")
        arr = f[dataset][:]

    # ilastik can export (1, y, x, c); pipeline expects (y, x, c)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected probs (y,x,c) or (1,y,x,c), got {arr.shape} from {h5_path}")
    return arr


def load_bmp_u8_and_f32(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a reference BMP and return the original image and float32 copy."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reference BMP not found: {path}")
    img_u8 = iio.imread(path)
    return img_u8, img_u8.astype(np.float32)


def save_bmp_uint8(path: Path, img: np.ndarray) -> None:
    """Save an image as an 8-bit BMP after clipping values to [0, 255]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.clip(img, 0, 255).astype(np.uint8))


def save_mask(path: Path, mask: np.ndarray) -> str:
    """Save a Boolean-like mask as uint8 NPY and return its path string."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mask.astype(np.uint8))
    return str(path)


def disk_mask(shape: tuple[int, int], cy: float, cx: float, diameter_px: float = 25.0) -> np.ndarray:
    """Create a circular mask centered on a manually annotated bregma.

    Parameters
    ----------
    shape : tuple[int, int]
        Output shape ``(y, x)`` in full-resolution Step2 pixels.
    cy, cx : float
        Bregma center coordinates in ``(y, x)`` order.
    diameter_px : float, default=25.0
        Diameter of the disk in Step2 pixels.

    Returns
    -------
    numpy.ndarray
        Boolean ``(y, x)`` disk mask.
    """
    h, w = int(shape[0]), int(shape[1])
    yy, xx = np.ogrid[:h, :w]
    radius = float(diameter_px) / 2.0
    return ((yy - float(cy)) ** 2 + (xx - float(cx)) ** 2) <= (radius ** 2)


# ============================================================
# Mask helpers
# ============================================================

def largest_cc(mask: np.ndarray) -> np.ndarray:
    """Retain the largest connected component of a Boolean spatial mask."""
    lab = label(mask.astype(bool))
    if lab.max() == 0:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()


def prob_to_mask(probs: np.ndarray, class_idx: int, thr: float) -> np.ndarray:
    """Threshold one Ilastik probability channel to form a Boolean mask.

    Parameters
    ----------
    probs : numpy.ndarray
        Probability image with the class index on the last axis.
    class_idx : int
        Zero-based probability channel index.
    thr : float
        Strict threshold: pixels are selected where ``probability > thr``.

    Returns
    -------
    numpy.ndarray
        Boolean mask with spatial shape ``probs.shape[:-1]``.
    """
    if class_idx is None:
        raise ValueError("class_idx is None")
    return probs[..., int(class_idx)].astype(np.float32) > float(thr)


def _wave_sort_key(value: Any) -> tuple[int, str]:
    """Return a sortable key placing numeric wavelengths before other strings."""
    text = str(value)
    try:
        return (0, f"{int(float(text)):08d}")
    except Exception:
        return (1, text)


def _role_cfg_enabled(role_cfg: Any) -> bool:
    """Check whether a semantic mask role is enabled in the configuration."""
    if role_cfg is None:
        return False
    if isinstance(role_cfg, bool):
        return bool(role_cfg)
    if not isinstance(role_cfg, dict):
        raise TypeError(f"mask_sources role config must be a mapping, got {type(role_cfg)!r}")
    return bool(role_cfg.get("enabled", True))


def get_mask_sources_config(config_or_step2: dict[str, Any]) -> dict[str, Any]:
    """Read the current semantic roles from step2.mask_sources."""
    step2_cfg = config_or_step2.get("step2", config_or_step2) or {}
    mask_sources = step2_cfg.get("mask_sources")
    if not isinstance(mask_sources, dict) or not mask_sources:
        raise ValueError("step2.mask_sources is required")
    return dict(mask_sources)


def iter_mask_source_entries(
    config_or_step2: dict[str, Any],
    role: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten enabled step2.mask_sources roles into individual wavelength/class entries."""
    mask_sources = get_mask_sources_config(config_or_step2)
    roles = [role] if role is not None else list(mask_sources.keys())
    entries: list[dict[str, Any]] = []

    for role_name in roles:
        role_cfg = mask_sources.get(str(role_name))
        if not _role_cfg_enabled(role_cfg):
            continue
        role_cfg = dict(role_cfg)

        role_defaults = {
            "role": str(role_name),
            "required": bool(role_cfg.get("required", str(role_name) in {"brain", "bregma"})),
            "add_to_bad_mask": bool(role_cfg.get("add_to_bad_mask", False)),
            "combine": role_cfg.get("combine", "union"),
        }

        raw_sources = role_cfg.get("sources")
        if raw_sources is None:
            raw_sources = [role_cfg]
        elif isinstance(raw_sources, dict):
            raw_sources = [raw_sources]

        for i, source in enumerate(raw_sources or []):
            if source is None:
                continue
            source = dict(source)
            if not bool(source.get("enabled", True)):
                continue
            entry = dict(role_defaults)
            entry.update(source)
            entry["role"] = str(role_name)
            entry["source_index"] = i
            entry.setdefault("threshold", 0.5)
            entry.setdefault("dilate", 0)
            entries.append(entry)
    return entries


def get_registration_wavelength(config: dict[str, Any]) -> str:
    """Read the explicitly configured step2.registration.image_wavelength."""
    step2_cfg = config.get("step2", {}) or {}
    reg_cfg = step2_cfg.get("registration", {}) or {}
    if not reg_cfg.get("image_wavelength"):
        raise ValueError("step2.registration.image_wavelength is required")
    return str(reg_cfg["image_wavelength"])


def get_step2_mask_wavelengths(config: dict[str, Any]) -> list[str]:
    """List wavelengths used by active mask source entries."""
    waves = {
        str(e["wavelength"])
        for e in iter_mask_source_entries(config)
        if e.get("wavelength") not in (None, "")
    }
    return sorted(waves, key=_wave_sort_key)


def get_step2_required_probability_wavelengths(config: dict[str, Any]) -> list[str]:
    """List wavelengths with required Ilastik probability sources."""
    waves = {
        str(e["wavelength"])
        for e in iter_mask_source_entries(config)
        if e.get("wavelength") not in (None, "") and bool(e.get("required", False))
    }
    return sorted(waves, key=_wave_sort_key)


def get_step2_needed_wavelengths(config: dict[str, Any]) -> list[str]:
    """Combine mask-source and registration wavelengths."""
    waves = set(get_step2_mask_wavelengths(config))
    waves.add(get_registration_wavelength(config))
    return sorted(waves, key=_wave_sort_key)


def _require_source_fields(entry: dict[str, Any]) -> None:
    """Require wavelength and class_index in a mask source entry."""
    missing = [k for k in ("wavelength", "class_index") if entry.get(k) in (None, "")]
    if missing:
        raise ValueError(f"mask_sources.{entry.get('role')} source is missing required fields: {missing}; entry={entry}")


def mask_from_source_entry(
    probs_by_wave: dict[str, np.ndarray],
    entry: dict[str, Any],
    *,
    required: bool | None = None,
) -> np.ndarray | None:
    """Construct one semantic mask from a configured Ilastik source.

    Parameters
    ----------
    probs_by_wave : dict[str, numpy.ndarray]
        Probability arrays indexed by wavelength; each has shape ``(y, x, class)``.
    entry : dict
        Normalized ``step2.mask_sources`` entry containing ``wavelength``,
        ``class_index`` and optional ``threshold``, ``required`` and ``dilate``.
    required : bool or None, optional
        Override whether a missing wavelength is treated as an error.

    Returns
    -------
    numpy.ndarray or None
        Boolean mask with shape ``(y, x)``, or ``None`` when an optional input
        wavelength is unavailable.

    Notes
    -----
        Thresholding is strict (``>``). Positive ``dilate`` applies a disk-shaped
        morphological dilation; the spatial unit is the Step2 image pixel.
    """
    _require_source_fields(entry)
    role = entry.get("role", "")
    wl = str(entry["wavelength"])
    probs = probs_by_wave.get(wl)
    is_required = bool(entry.get("required", False) if required is None else required)
    if probs is None:
        if is_required:
            raise FileNotFoundError(f"Required probability map for role={role!r}, wavelength={wl!r} is not loaded")
        return None
    class_index = int(entry["class_index"])
    if class_index < 0 or class_index >= probs.shape[-1]:
        raise IndexError(
            f"class_index={class_index} is outside probability channels for role={role!r}, "
            f"wavelength={wl!r}, probs shape={probs.shape}"
        )
    m = prob_to_mask(probs, class_index, float(entry.get("threshold", 0.5)))
    dilate_r = int(entry.get("dilate", 0) or 0)
    if dilate_r > 0:
        m = dilation(m, disk(dilate_r), mode="ignore")
    return m.astype(bool)


def combine_source_masks(
    probs_by_wave: dict[str, np.ndarray],
    config_or_step2: dict[str, Any],
    role: str,
    *,
    required: bool | None = None,
) -> np.ndarray | None:
    """Combine the probability-derived masks for one semantic role.

    Parameters
    ----------
    probs_by_wave : dict[str, numpy.ndarray]
        Probability maps keyed by wavelength.
    config_or_step2 : dict
        Full configuration or ``step2`` subsection containing ``mask_sources``.
    role : str
        Semantic role, such as ``brain``, ``bregma``, ``glare`` or ``vessels``.
    required : bool or None, optional
        Whether the role must be present and readable.

    Returns
    -------
    numpy.ndarray or None
        Boolean ``(y, x)`` union of all available source masks; optional missing
        sources may result in ``None``.

    Notes
    -----
        Masks from multiple enabled source entries are combined using logical OR.
    """
    entries = iter_mask_source_entries(config_or_step2, role=role)
    if not entries:
        if required or role in {"brain", "bregma"}:
            raise ValueError(f"Required mask_sources.{role} is not configured")
        return None

    masks: list[np.ndarray] = []
    for entry in entries:
        m = mask_from_source_entry(probs_by_wave, entry, required=required)
        if m is not None:
            masks.append(m)

    if not masks:
        if required or any(bool(e.get("required", False)) for e in entries):
            raise RuntimeError(f"No probability mask could be built for required role={role!r}")
        return None

    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out.astype(bool)


def build_brain_and_ring_from_sources(
    probs_by_wave: dict[str, np.ndarray],
    config_or_step2: dict[str, Any],
    close_r: int,
    r_ring: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create the brain mask and a ring surrounding its boundary.

    Parameters
    ----------
    probs_by_wave : dict[str, numpy.ndarray]
        Ilastik probability maps, indexed by configured wavelengths.
    config_or_step2 : dict
        Configuration containing ``step2.mask_sources.brain``.
    close_r : int
        Radius in full-resolution Step2 pixels for morphological closing
        (``step2.morphology.close_r_brain``).
    r_ring : int
        Radius in full-resolution Step2 pixels for ring dilation and erosion
        (``step2.morphology.ring_radius``).

    Returns
    -------
    brain : numpy.ndarray
        Boolean filled largest-component brain mask, shape ``(y, x)``.
    ring : numpy.ndarray
        Boolean ring defined as dilated brain minus eroded brain.

    Notes
    -----
        The brain mask undergoes largest-component selection, closing,
        largest-component selection again, then hole filling.
    """
    brain = combine_source_masks(probs_by_wave, config_or_step2, "brain", required=True)
    if brain is None:
        raise RuntimeError("brain mask could not be built")
    brain = largest_cc(brain)
    brain = closing(brain, disk(int(close_r)), mode="ignore")
    brain = largest_cc(brain)
    brain = binary_fill_holes(brain)

    outer = dilation(brain, disk(int(r_ring)), mode="ignore")
    inner = erosion(brain, disk(int(r_ring)), mode="ignore")
    ring = outer & (~inner)
    return brain.astype(bool), ring.astype(bool)


def bregma_centroid_from_sources(
    probs_by_wave: dict[str, np.ndarray],
    config_or_step2: dict[str, Any],
) -> Tuple[Optional[float], Optional[float], np.ndarray, bool]:
    """Estimate bregma position from its configured probability mask.

    Parameters
    ----------
    probs_by_wave : dict[str, numpy.ndarray]
        Probability arrays keyed by wavelength.
    config_or_step2 : dict
        Configuration containing ``step2.mask_sources.bregma``.

    Returns
    -------
    cy, cx : float or None
        Largest connected component centroid in Step2 pixels, in ``(y, x)``
        coordinate order, or ``None`` when the mask is empty.
    mask : numpy.ndarray
        Boolean largest-component bregma mask, shape ``(y, x)``.
    found : bool
        Whether the component has a centroid.
    """
    bregma = combine_source_masks(probs_by_wave, config_or_step2, "bregma", required=True)
    if bregma is None:
        # combine_source_masks(required=True) should already raise, but keep this defensive fallback.
        first = next(iter(probs_by_wave.values()))
        bregma = np.zeros(first.shape[:2], dtype=bool)
    m = largest_cc(bregma)
    props = regionprops(m.astype(np.uint8))
    if not props:
        return None, None, m.astype(bool), False
    cy, cx = props[0].centroid
    return float(cy), float(cx), m.astype(bool), True


def build_bad_mask_from_sources(
    probs_by_wave: dict[str, np.ndarray],
    config_or_step2: dict[str, Any],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Construct the exclusion mask from configured nonstructural roles.

    Parameters
    ----------
    probs_by_wave : dict[str, numpy.ndarray]
        Ilastik probability maps indexed by wavelength.
    config_or_step2 : dict
        Configuration specifying source roles and the common dilation radius.
    image_shape : tuple[int, int]
        Output spatial shape ``(y, x)``.

    Returns
    -------
    numpy.ndarray
        Boolean ``(y, x)`` exclusion mask.

    Notes
    -----
        Only roles with ``add_to_bad_mask: true`` contribute. The ``brain`` and
        ``bregma`` roles are always excluded. The optional common dilation uses
        ``step2.morphology.bad_dilate_common`` in full-resolution pixels.
    """
    bad = np.zeros(image_shape, dtype=bool)
    added = False

    for entry in iter_mask_source_entries(config_or_step2):
        role = str(entry.get("role", ""))
        if role in {"brain", "bregma"}:
            continue
        if not bool(entry.get("add_to_bad_mask", False)):
            continue
        m = mask_from_source_entry(probs_by_wave, entry, required=False)
        if m is None:
            continue
        bad |= m.astype(bool)
        added = True

    d = int((config_or_step2.get("step2", config_or_step2) or {}).get("morphology", {}).get("bad_dilate_common", 0) or 0)
    if added and d > 0:
        bad = dilation(bad, disk(d), mode="ignore")
    return bad.astype(bool)


# ============================================================
# Registration helpers
# ============================================================

def apply_tform(image_f32: np.ndarray, tform: EuclideanTransform, order: int = 1) -> np.ndarray:
    """Warp a two-dimensional image using a forward Euclidean transform.

    Parameters
    ----------
    image_f32 : numpy.ndarray
        Image with shape ``(y, x)``.
    tform : skimage.transform.EuclideanTransform
        Forward source-to-destination transform in ``(x, y)`` coordinates.
    order : int, default=1
        Interpolation order passed to ``skimage.transform.warp``.

    Returns
    -------
    numpy.ndarray
        Transformed image, dtype ``float32``.

    Notes
    -----
        ``warp`` receives ``tform.inverse`` because it samples source pixels
        using inverse coordinates.
    """
    return warp(image_f32, tform.inverse, order=order, preserve_range=True).astype(np.float32)


def rotation_about_point_tform(angle_deg: float, center_yx: Tuple[float, float]) -> EuclideanTransform:
    """Build a rigid rotation around a fixed point in image coordinates.

    Parameters
    ----------
    angle_deg : float
        Rotation angle in degrees.
    center_yx : tuple[float, float]
        Center point in ``(y, x)`` image pixels.

    Returns
    -------
    skimage.transform.EuclideanTransform
        Forward transform, with ``(x, y)`` translation convention.

    Notes
    -----
        Transform composition translates to the origin, rotates, then translates
        back. The function verifies that the center remains stationary.
    """
    cy, cx = center_yx
    theta = np.deg2rad(float(angle_deg))
    tform = (
        EuclideanTransform(translation=(-float(cx), -float(cy)))
        + EuclideanTransform(rotation=theta)
        + EuclideanTransform(translation=(float(cx), float(cy)))
    )

    # Defensive geometry check: the rotation center must be invariant.
    center_xy = np.array([[float(cx), float(cy)]], dtype=float)
    mapped_xy = tform(center_xy)[0]
    center_error = float(np.hypot(mapped_xy[0] - float(cx), mapped_xy[1] - float(cy)))
    if center_error > 1e-6:
        raise RuntimeError(f"rotation_about_point_tform failed to keep center fixed: {center_error:.6g} px")
    return tform


def rotate_about_point(
    image_f32: np.ndarray,
    angle_deg: float,
    center_yx: Tuple[float, float],
    order: int = 1,
) -> tuple[np.ndarray, EuclideanTransform]:
    """Rotate an image around an image-space center.

    Parameters
    ----------
    image_f32 : numpy.ndarray
        Two-dimensional image.
    angle_deg : float
        Angle in degrees.
    center_yx : tuple[float, float]
        Rotation center in ``(y, x)`` pixels.
    order : int, default=1
        Warp interpolation order.

    Returns
    -------
    rotated : numpy.ndarray
        Warped image of shape ``(y, x)``.
    transform : skimage.transform.EuclideanTransform
        Forward transform used for the warp.
    """
    tform = rotation_about_point_tform(angle_deg, center_yx)
    return apply_tform(image_f32, tform, order=order), tform


def translate_mask(mask: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate a spatial mask by a pixel offset.

    Parameters
    ----------
    mask : numpy.ndarray
        Input Boolean-like mask with shape ``(y, x)``.
    dy, dx : float
        Translation in ``(y, x)`` Step2 pixel order.

    Returns
    -------
    numpy.ndarray
        Boolean translated mask using nearest-neighbor interpolation.
    """
    t = EuclideanTransform(translation=(float(dx), float(dy)))
    w = warp(mask.astype(np.float32), t.inverse, order=0, preserve_range=True)
    return w > 0.5


def norm_in_mask(img_f32: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Standardize feature values inside a Boolean registration mask.

    Parameters
    ----------
    img_f32 : numpy.ndarray
        Two-dimensional feature image.
    mask : numpy.ndarray
        Boolean registration ROI with shape ``(y, x)``.

    Returns
    -------
    numpy.ndarray
        Float32 image with values outside the mask set to zero.

    Notes
    -----
        When the mask has more than 50 pixels, its masked values are standardized
        by the masked mean and standard deviation; otherwise values remain.
        Check the implementation before interpreting this as a statistical z-score.
    """
    m = mask.astype(bool)
    x = img_f32.copy()
    x[~m] = 0.0
    vals = x[m]
    if vals.size > 50:
        x[m] = (vals - float(vals.mean())) / (float(vals.std()) + 1e-6)
    return x


def corr_score(a_f32: np.ndarray, b_f32: np.ndarray, mask: np.ndarray) -> float:
    """Calculate masked correlation-like similarity of two feature maps.

    Parameters
    ----------
    a_f32, b_f32 : numpy.ndarray
        Feature images with shape ``(y, x)``.
    mask : numpy.ndarray
        Boolean ROI for scoring.

    Returns
    -------
    float
        Mean product of standardized masked feature values; returns negative
        infinity when fewer than 200 spatial samples are present.
    """
    m = mask.astype(bool)
    aa = a_f32[m].astype(np.float32)
    bb = b_f32[m].astype(np.float32)
    if aa.size < 200:
        return float("-inf")
    aa = (aa - aa.mean()) / (aa.std() + 1e-6)
    bb = (bb - bb.mean()) / (bb.std() + 1e-6)
    return float((aa * bb).mean())




def transform_point_yx(tform: EuclideanTransform, y: float, x: float) -> tuple[float, float]:
    """Apply a skimage EuclideanTransform to one (y, x) point and return (y, x)."""
    xy = np.array([[float(x), float(y)]], dtype=float)
    xy_t = tform(xy)[0]
    return float(xy_t[1]), float(xy_t[0])


def bregma_anchor_error_px(tform: EuclideanTransform, moving_yx: Tuple[float, float], ref_yx: Tuple[float, float]) -> float:
    """Measure transformed bregma displacement from reference in Step2 pixels."""
    yy, xx = transform_point_yx(tform, moving_yx[0], moving_yx[1])
    return float(np.hypot(yy - float(ref_yx[0]), xx - float(ref_yx[1])))


def normalize_for_feature(image: np.ndarray, mask: np.ndarray | None = None, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize a registration image using intensity percentiles.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional image.
    mask : numpy.ndarray or None, optional
        Optional Boolean spatial region used to estimate percentiles.
    p_low, p_high : float
        Lower and upper percentiles.

    Returns
    -------
    numpy.ndarray
        Float32 image clipped to the unit interval, shape ``(y, x)``.

    Notes
    -----
        If fewer than ten finite values are found inside the mask, the function
        uses all finite image values; if there are still fewer than ten, it
        returns zeros.
    """
    img = np.asarray(image, dtype=np.float32)
    if mask is not None:
        vals = img[np.asarray(mask, dtype=bool)]
    else:
        vals = img.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        vals = img[np.isfinite(img)]
    if vals.size < 10:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.nanpercentile(vals, [float(p_low), float(p_high)])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    out = (img - lo) / (hi - lo + 1e-6)
    return np.clip(out, 0, 1).astype(np.float32)


def vessel_sobel_feature(image: np.ndarray, mask: np.ndarray | None = None, cfg: dict[str, Any] | None = None) -> np.ndarray:
    """Construct a Sobel feature image emphasizing vessel-like structures.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional registration image.
    mask : numpy.ndarray or None, optional
        Optional Boolean spatial ROI, shape ``(y, x)``.
    cfg : dict or None, optional
        ``step2.registration.vessel_sobel`` settings for percentile normalization,
        background subtraction, polarity and optional smoothing.

    Returns
    -------
    numpy.ndarray
        Float32 Sobel feature map with shape ``(y, x)``.

    Notes
    -----
        Order: percentile normalization; optional Gaussian background subtraction;
        polarity selection; optional Gaussian feature smoothing; Sobel gradient.
        Pixels outside the supplied mask are set to zero.
    """
    cfg = cfg or {}
    img = normalize_for_feature(
        image,
        mask=mask,
        p_low=float(cfg.get("percentile_low", 1)),
        p_high=float(cfg.get("percentile_high", 99)),
    )

    bg_sigma = cfg.get("background_sigma", 12)
    if bg_sigma is not None and float(bg_sigma) > 0:
        high = img - gaussian_filter(img, sigma=float(bg_sigma))
    else:
        high = img

    polarity = str(cfg.get("polarity", "both")).lower()
    if polarity == "dark":
        high = np.maximum(-high, 0)
    elif polarity == "bright":
        high = np.maximum(high, 0)
    elif polarity == "both":
        high = np.abs(high)
    else:
        raise ValueError("step2.registration.vessel_sobel.polarity must be 'both', 'dark' or 'bright'")

    feature_sigma = float(cfg.get("feature_sigma", 0.0) or 0.0)
    if feature_sigma > 0:
        high = gaussian_filter(high, sigma=feature_sigma)

    feat = sobel(high).astype(np.float32)
    if mask is not None:
        feat = feat.copy()
        feat[~np.asarray(mask, dtype=bool)] = 0
    return feat


def build_registration_feature(image: np.ndarray, mask: np.ndarray | None, reg_cfg: dict[str, Any]) -> np.ndarray:
    """Compute the Sobel-based feature selected by step2.registration.method."""
    method = str(reg_cfg.get("method", "ring_sobel")).strip().lower()
    if method in {"ring_sobel", "raw_sobel", "sobel"}:
        return sobel(normalize_for_feature(image, mask=mask)).astype(np.float32)
    if method == "vessel_sobel":
        return vessel_sobel_feature(image, mask=mask, cfg=reg_cfg.get("vessel_sobel", {}) or {})
    raise ValueError(f"Unsupported step2.registration.method={method!r}. Use 'ring_sobel' or 'vessel_sobel'.")


def build_registration_roi(
    *,
    roi_mode: str,
    mask_mode: str,
    ref_rec: dict,
    mov_rec: dict,
    init_dy: float,
    init_dx: float,
    image_shape: tuple[int, int],
    group_union_ring: np.ndarray | None = None,
    group_union_bad: np.ndarray | None = None,
    ignore_border_px: int = 0,
) -> np.ndarray:
    """Select spatial pixels used to estimate registration.

    Parameters
    ----------
    roi_mode : str
        ROI selection mode: ring-like, full image, brain, or brain excluding bad.
    mask_mode : str
        Pairwise, group-union or hybrid ring-mask logic where applicable.
    ref_rec, mov_rec : dict
        Reference and moving records with brain, bad and ring masks.
    init_dy, init_dx : float
        Initial moving-to-reference bregma shifts in Step2 pixels.
    image_shape : tuple[int, int]
        Image shape ``(y, x)``.
    group_union_ring, group_union_bad : numpy.ndarray or None, optional
        Shared group masks for ring-like modes.
    ignore_border_px : int, default=0
        Excluded outer border width, in Step2 pixels.

    Returns
    -------
    numpy.ndarray
        Boolean registration ROI with shape ``(y, x)``.

    Notes
    -----
        ``step2.registration.roi_mode`` is independent of the registration
        feature method. A full-image ROI does not necessarily use the saved
        ``mask_for_reg_union.npy`` even if ``mask_mode`` is group_union.
    """
    mode = str(roi_mode or "ring").strip().lower()
    if mode in {"ring", "ring_sobel", "mask_mode", "legacy", "old"}:
        roi = build_reg_mask_by_mode(
            mode=mask_mode,
            ref_rec=ref_rec,
            mov_rec=mov_rec,
            init_dy=init_dy,
            init_dx=init_dx,
            group_union_ring=group_union_ring,
            group_union_bad=group_union_bad,
        )
    elif mode in {"full", "full_image", "whole_image", "image"}:
        roi = np.ones(tuple(image_shape), dtype=bool)
    elif mode == "brain":
        roi = np.asarray(ref_rec["brain"], dtype=bool).copy()
    elif mode in {"brain_minus_bad", "brain-no-bad", "brain_without_bad"}:
        roi = np.asarray(ref_rec["brain"], dtype=bool) & (~np.asarray(ref_rec["bad"], dtype=bool))
    else:
        raise ValueError(f"Unsupported registration roi_mode={roi_mode!r}")

    b = int(ignore_border_px or 0)
    if b > 0:
        roi = roi.copy()
        roi[:b, :] = False
        roi[-b:, :] = False
        roi[:, :b] = False
        roi[:, -b:] = False
    return roi.astype(bool)


def estimate_feature_scored(
    fixed_feature: np.ndarray,
    moving0_feature: np.ndarray,
    mask: np.ndarray,
    rot_center_yx: Tuple[float, float],
    angle_range: float,
    angle_step: float,
    upsample: int,
    *,
    anchor_bregma: bool = True,
) -> tuple[EuclideanTransform, float, tuple[float, float], float]:
    """Estimate a scored rotation and optional residual translation.

    Parameters
    ----------
    fixed_feature, moving0_feature : numpy.ndarray
        Registration feature images of shape ``(y, x)`` after initial bregma
        alignment of the moving image.
    mask : numpy.ndarray
        Boolean ``(y, x)`` registration ROI.
    rot_center_yx : tuple[float, float]
        Bregma rotation center in Step2 ``(y, x)`` pixels.
    angle_range, angle_step : float
        Half-range and increment of searched rotation angles, in degrees.
    upsample : int
        Phase cross-correlation upsampling factor when free shifts are enabled.
    anchor_bregma : bool, default=True
        If true, do not estimate a second-stage translation.

    Returns
    -------
    transform : skimage.transform.EuclideanTransform
        Best second-stage forward transform.
    angle_deg : float
        Selected angle, in degrees.
    shift_yx : tuple[float, float]
        Estimated residual translation ``(dy, dx)`` in Step2 pixels.
    score : float
        Masked correlation score of transformed features.

    Notes
    -----
        The search maximizes masked correlation over tested angles. When
        ``anchor_bregma`` is false, phase cross-correlation estimates the free
        translation for each candidate angle.
    """
    fixed_m = norm_in_mask(fixed_feature, mask)
    best: dict[str, Any] | None = None
    angles = np.arange(-float(angle_range), float(angle_range) + 1e-9, float(angle_step), dtype=float)

    for a in angles:
        mov_rot, rot_tform = rotate_about_point(moving0_feature, a, rot_center_yx, order=1)
        if anchor_bregma:
            dy, dx = 0.0, 0.0
            total_tform = rot_tform
            mov_warp = mov_rot
        else:
            mov_m = norm_in_mask(mov_rot, mask)
            shift, _, _ = phase_cross_correlation(fixed_m, mov_m, upsample_factor=int(upsample))
            dy, dx = float(shift[0]), float(shift[1])
            total_tform = rot_tform + EuclideanTransform(translation=(dx, dy))
            mov_warp = apply_tform(moving0_feature, total_tform, order=1)

        sc = corr_score(fixed_feature, mov_warp, mask)
        if best is None or sc > best["score"]:
            best = {"angle": float(a), "dy": float(dy), "dx": float(dx), "score": float(sc), "tform": total_tform}

    if best is None:
        raise RuntimeError("Could not estimate registration transform")
    return best["tform"], best["angle"], (best["dy"], best["dx"]), best["score"]

def build_pair_reg_mask(
    ring_ref: np.ndarray,
    bad_ref: np.ndarray,
    ring_mov: np.ndarray,
    bad_mov: np.ndarray,
    dy: float,
    dx: float,
) -> np.ndarray:
    """Build a pairwise ring registration mask after a bregma shift.

    Parameters
    ----------
    ring_ref, bad_ref, ring_mov, bad_mov : numpy.ndarray
        Boolean masks with shape ``(y, x)``.
    dy, dx : float
        Moving-to-reference shift in Step2 pixels.

    Returns
    -------
    numpy.ndarray
        Boolean union of the aligned rings excluding either bad mask.
    """
    ring_mov_s = translate_mask(ring_mov, dy=dy, dx=dx)
    bad_mov_s = translate_mask(bad_mov, dy=dy, dx=dx)
    return ((ring_ref | ring_mov_s) & (~(bad_ref | bad_mov_s))).astype(bool)


def build_group_union_mask(records_for_group: list[dict], ref_stem: str) -> tuple[np.ndarray, np.ndarray]:
    """Form bregma-aligned unions of group ring and bad masks.

    Parameters
    ----------
    records_for_group : list[dict]
        Records for one registration group, including masks and bregma points.
    ref_stem : str
        Reference recording identity.

    Returns
    -------
    ring_union, bad_union : numpy.ndarray
        Boolean arrays with shape ``(y, x)`` in reference coordinates.

    Notes
    -----
        Records without a bregma point do not contribute to the union.
    """
    ref = next((r for r in records_for_group if r["stem"] == ref_stem), None)
    if ref is None:
        raise ValueError(f"Reference stem not found: {ref_stem}")
    if not ref["bregma_found"]:
        raise RuntimeError(f"Reference bregma missing for {ref_stem}")

    cy0, cx0 = ref["bregma"]
    ring_union = ref["ring"].copy().astype(bool)
    bad_union = ref["bad"].copy().astype(bool)

    for rec in records_for_group:
        if rec["stem"] == ref_stem or not rec["bregma_found"]:
            continue
        cy, cx = rec["bregma"]
        dy = cy0 - cy
        dx = cx0 - cx
        ring_union |= translate_mask(rec["ring"], dy=dy, dx=dx)
        bad_union |= translate_mask(rec["bad"], dy=dy, dx=dx)

    return ring_union.astype(bool), bad_union.astype(bool)


def build_reg_mask_by_mode(
    mode: str,
    ref_rec: dict,
    mov_rec: dict,
    init_dy: float,
    init_dx: float,
    group_union_ring: np.ndarray | None = None,
    group_union_bad: np.ndarray | None = None,
) -> np.ndarray:
    """Choose pairwise, group-union or hybrid mask combination."""
    pair_mask = build_pair_reg_mask(
        ring_ref=ref_rec["ring"],
        bad_ref=ref_rec["bad"],
        ring_mov=mov_rec["ring"],
        bad_mov=mov_rec["bad"],
        dy=init_dy,
        dx=init_dx,
    )

    if mode == "pairwise":
        return pair_mask

    if group_union_ring is None or group_union_bad is None:
        raise ValueError(f"{mode} mode requires group_union_ring and group_union_bad")

    if mode == "group_union":
        return (group_union_ring & (~group_union_bad)).astype(bool)

    if mode == "hybrid":
        bad_mov_s = translate_mask(mov_rec["bad"], dy=init_dy, dx=init_dx)
        bad_pair = ref_rec["bad"] | bad_mov_s
        ring_hybrid = group_union_ring | ref_rec["ring"]
        return (ring_hybrid & (~(group_union_bad | bad_pair))).astype(bool)

    raise ValueError(f"Unknown mask_mode: {mode}")


def estimate_rigid_scored(
    fixed_f32: np.ndarray,
    moving0_f32: np.ndarray,
    mask: np.ndarray,
    rot_center_yx: Tuple[float, float],
    angle_range: float,
    angle_step: float,
    upsample: int,
) -> tuple[EuclideanTransform, float, tuple[float, float], float]:
    """Search rotations with phase-correlation translation on Sobel edges.

    Parameters
    ----------
    fixed_f32, moving0_f32 : numpy.ndarray
        Two-dimensional images, with moving image initially bregma-shifted.
    mask : numpy.ndarray
        Boolean ``(y, x)`` scoring ROI.
    rot_center_yx : tuple[float, float]
        Rotation center in Step2 image pixels.
    angle_range, angle_step : float
        Rotation half-range and step, in degrees.
    upsample : int
        Phase cross-correlation upsampling factor.

    Returns
    -------
    transform : skimage.transform.EuclideanTransform
        Best forward second-stage transform.
    angle_deg : float
        Angle selected by masked correlation, in degrees.
    shift_yx : tuple[float, float]
        Free translation ``(dy, dx)`` in pixels.
    score : float
        Selected masked correlation score.
    """
    fixed_edge = sobel(fixed_f32)
    moving_edge = sobel(moving0_f32)
    fixed_m = norm_in_mask(fixed_edge, mask)

    best: dict[str, Any] | None = None
    angles = np.arange(-angle_range, angle_range + 1e-9, angle_step, dtype=float)

    for a in angles:
        mov_rot, rot_tform = rotate_about_point(moving_edge, a, rot_center_yx, order=1)
        mov_m = norm_in_mask(mov_rot, mask)
        shift, _, _ = phase_cross_correlation(fixed_m, mov_m, upsample_factor=int(upsample))
        dy, dx = float(shift[0]), float(shift[1])

        total_tform = rot_tform + EuclideanTransform(translation=(dx, dy))
        mov_warp = apply_tform(moving_edge, total_tform, order=1)
        sc = corr_score(fixed_edge, mov_warp, mask)

        if best is None or sc > best["score"]:
            best = {"angle": float(a), "dy": dy, "dx": dx, "score": float(sc), "tform": total_tform}

    if best is None:
        raise RuntimeError("Could not estimate registration transform")
    return best["tform"], best["angle"], (best["dy"], best["dx"]), best["score"]


# ============================================================
# Manifest index + protocol/reference helpers
# ============================================================


def get_analysis_group_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize registration grouping and reference-selection settings.

    Parameters
    ----------
    config : dict
        Full project configuration.

    Returns
    -------
    dict
        ``analysis_group`` settings with a list-valued ``group_by`` and explicit
        ``reference.match`` / ``reference.variant_keys`` fields.

    Notes
    -----
        If ``analysis_group.group_by`` is missing, the first value of
        ``metadata.stem_keys`` is used as a fallback; this is a code default,
        not a requirement that the experiment be grouped by mouse ID.
    """
    ag = (config.get("analysis_group", {}) or {}).copy()
    group_by = ag.get("group_by")
    if isinstance(group_by, str):
        group_by = [group_by]
    if not group_by:
        # Conservative default: first stem key is often the subject id, but it is still explicit in output.
        stem_keys = get_stem_keys(config)
        group_by = [stem_keys[0]]
    ag["group_by"] = [str(x) for x in group_by]
    ag.setdefault("reference", {})
    ag["reference"].setdefault("match", {})
    ag["reference"].setdefault("variant_keys", [])
    if isinstance(ag["reference"].get("variant_keys"), str):
        ag["reference"]["variant_keys"] = [ag["reference"]["variant_keys"]]
    return ag


def get_debug_filters(config: dict[str, Any]) -> dict[str, Any]:
    """Read optional Step2 debug filters from the project configuration."""
    return ((config.get("step2", {}) or {}).get("debug", {}) or {}).get("filters", {}) or {}


def _is_manual_bregma_source(value: Any) -> bool:
    """Recognize manual bregma labels, including legacy Cyrillic metadata values."""
    return str(value).strip().lower() in {"manual", "man", "ручная", "вручную"}


def load_manual_bregma_overrides(metadata_csv: Path) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """Load previously saved manual bregma positions for Step2 reruns.

    Parameters
    ----------
    metadata_csv : pathlib.Path
        Existing ``metadata_registration.csv``.

    Returns
    -------
    overrides : dict[str, tuple[float, float]]
        Stem to ``(cy, cx)`` coordinates in full-resolution Step2 pixels.
    manual_groups : set[str]
        Group IDs associated with valid manual corrections.

    Notes
    -----
        Legacy Cyrillic values of ``bregma_source`` are recognized for backward
        compatibility; do not translate those data values in executable code.
    """
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        return {}, set()

    old = pd.read_csv(metadata_csv)
    if old.empty or "bregma_source" not in old.columns:
        return {}, set()

    needed = {"stem", "bregma_cy", "bregma_cx"}
    if not needed.issubset(set(old.columns)):
        return {}, set()

    manual = old[old["bregma_source"].apply(_is_manual_bregma_source)].copy()
    manual["bregma_cy"] = pd.to_numeric(manual["bregma_cy"], errors="coerce")
    manual["bregma_cx"] = pd.to_numeric(manual["bregma_cx"], errors="coerce")
    manual = manual[np.isfinite(manual["bregma_cy"]) & np.isfinite(manual["bregma_cx"])]

    overrides: dict[str, tuple[float, float]] = {}
    for _, r in manual.iterrows():
        overrides[str(r["stem"])] = (float(r["bregma_cy"]), float(r["bregma_cx"]))

    manual_groups: set[str] = set()
    if "group_id" in manual.columns:
        manual_groups = {str(v) for v in manual["group_id"].dropna().astype(str).tolist() if str(v).strip()}

    return overrides, manual_groups


def preserve_manual_bregma_provenance(
    previous: pd.DataFrame | None, current: pd.DataFrame
) -> pd.DataFrame:
    """Carry existing manual-bregma timestamps and annotator names into rerun metadata."""
    if previous is None or "bregma_source" not in previous or "bregma_source" not in current:
        return current
    previous_manual = previous[
        previous["bregma_source"].astype(str).str.lower().eq("manual")
    ].drop_duplicates("stem", keep="last").set_index("stem")
    current_manual = current["bregma_source"].astype(str).str.lower().eq("manual")
    for col in ("bregma_updated_at", "bregma_annotator"):
        if col not in previous_manual:
            continue
        if col not in current:
            current[col] = ""
        missing = current[col].isna() | current[col].astype(str).str.strip().eq("")
        prior = current["stem"].astype(str).map(previous_manual[col])
        selected = current_manual & missing & prior.notna()
        current.loc[selected, col] = prior[selected]
    return current


def apply_metadata_filters_df(
    df: pd.DataFrame, filters: dict[str, Any], config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Filter Step2 rows using metadata fields and configured protocol variants."""
    out = df.copy()
    variants = (config or {}).get("variant_keys", {}) or {}
    for key, allowed in (filters or {}).items():
        if key not in out.columns:
            raise ValueError(f"Debug/filter column not found in Step2 table: {key}")
        if allowed is None:
            continue
        if isinstance(allowed, (list, tuple, set)):
            allowed_set = {str(v) for v in allowed}
        else:
            allowed_set = {str(allowed)}
        variant_cfg = variants.get(key, {}) or {}
        if variant_cfg.get("enabled", False):
            separators = variant_cfg.get("separators", ["", "-", "_"])
            actual = out[key].apply(lambda v: parse_variant_value(v, separators)[0])
            allowed_set = {parse_variant_value(v, separators)[0] for v in allowed_set}
        else:
            actual = out[key].astype(str)
        out = out[actual.isin(allowed_set)].copy()
    return out


def add_variant_columns(df: pd.DataFrame, config: dict[str, Any], variant_keys: Sequence[str] | None = None) -> pd.DataFrame:
    """Append base and repeat fields for configured metadata variant keys."""
    out = df.copy()
    variants_cfg = config.get("variant_keys", {}) or {}
    keys = list(variant_keys or variants_cfg.keys())
    for key in keys:
        if key not in out.columns:
            continue
        key_cfg = variants_cfg.get(key, {}) or {}
        if key_cfg and not key_cfg.get("enabled", True):
            continue
        separators = key_cfg.get("separators", ["", "-", "_"])
        parsed = out[key].apply(lambda v: parse_variant_value(v, separators=separators))
        out[f"{key}_base"] = parsed.apply(lambda x: x[0])
        out[f"{key}_repeat"] = parsed.apply(lambda x: x[1])
    return out


def choose_reference_row_for_group(
    df_group: pd.DataFrame,
    config: dict[str, Any],
    allow_fallback: bool = False,
) -> tuple[pd.Series, bool]:
    """Select a reference recording using configured metadata rules.

    Parameters
    ----------
    df_group : pandas.DataFrame
        Records belonging to a single ``analysis_group.group_by`` identity.
    config : dict
        Complete YAML configuration; reads ``analysis_group.reference.match``
        and ``analysis_group.reference.variant_keys``.
    allow_fallback : bool, default=False
        Permit reference selection when no row exactly matches the criteria.

    Returns
    -------
    reference : pandas.Series
        Selected reference record.
    exact_match : bool
        Whether the selected record satisfies the configured match criteria.

    Notes
    -----
        Configured variant keys are matched by their ``<key>_base`` fields, and
        repeated matches favor the greatest ``<key>_repeat`` value. An allowed
        fallback must not be documented as an exact reference match.
    """
    if df_group.empty:
        raise RuntimeError("Empty analysis group dataframe")

    ag = get_analysis_group_config(config)
    ref_cfg = ag.get("reference", {}) or {}
    match = ref_cfg.get("match", {}) or {}
    variant_keys = [str(k) for k in (ref_cfg.get("variant_keys", []) or [])]

    work = add_variant_columns(df_group, config=config, variant_keys=variant_keys)
    strict = work.copy()

    for key, expected in match.items():
        key = str(key)
        if key not in strict.columns:
            raise ValueError(f"Reference match key not found in metadata_registration table: {key}")
        if key in variant_keys:
            base_col = f"{key}_base"
            separators = ((config.get("variant_keys", {}) or {}).get(key, {}) or {}).get(
                "separators", ["", "-", "_"]
            )
            expected_base, _ = parse_variant_value(expected, separators)
            strict = strict[strict[base_col].astype(str) == str(expected_base)]
        else:
            strict = strict[strict[key].astype(str) == str(expected)]

    exact = not strict.empty
    if strict.empty:
        if not allow_fallback:
            raise RuntimeError(f"No reference found for match={match}")
        strict = work.copy()

    sort_cols = []
    ascending = []
    for key in variant_keys:
        repeat_col = f"{key}_repeat"
        if repeat_col in strict.columns:
            sort_cols.append(repeat_col)
            ascending.append(False)
    # stable deterministic tie-breaker
    sort_cols.append("stem")
    ascending.append(True)
    strict = strict.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
    return strict.iloc[0], bool(exact)


# ============================================================
# Step2 index builder
# ============================================================

def build_step2_index_from_manifest(
    manifest_path: Path,
    cfg: dict,
    paths: dict[str, Path],
    *,
    require_h5: bool = True,
    from_drive: str | None = None,
    to_drive: str | None = None,
) -> pd.DataFrame:
    """Validate and index Step1 recordings for Step2 processing.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to the Step1 ``manifest_recordings.csv``.
    cfg : dict
        Complete configuration; requires ``metadata`` and ``analysis_group``.
    paths : dict[str, pathlib.Path]
        Standard project paths returned by ``build_standard_paths``.
    require_h5 : bool, default=True
        Require configured Ilastik probability inputs during index validation.
    from_drive, to_drive : str or None, optional
        Optional drive remapping overrides for resolving file locations.

    Returns
    -------
    pandas.DataFrame
        Validated stem-level index with group identity, reference image and
        probability file paths, and metadata used by registration.

    Notes
    -----
        Input wavelengths derive from ``step2.mask_sources`` and
        ``step2.registration.image_wavelength``. Failed inputs are reported to
        ``step2_missing_inputs.csv``. The full configuration is required by the
        implementation; passing only ``config['step2']`` is not supported.
    """
    if "metadata" not in cfg or "analysis_group" not in cfg:
        raise ValueError("Step2 v2_4+ requires the full config, including metadata.stem_keys and analysis_group.")
    full_config = cfg
    step2_cfg = full_config.get("step2", {}) or {}
    path_cfg = get_path_resolving_config(full_config, "step2")
    if from_drive is not None:
        path_cfg["from_drive"] = from_drive
    if to_drive is not None:
        path_cfg["to_drive"] = to_drive

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest_recordings.csv not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    # Duplicate stem identities cannot be assigned reliably to masks or registration.
    duplicate_stems = manifest.loc[manifest["stem"].astype(str).duplicated(keep=False), "stem"].astype(str).unique().tolist()
    if duplicate_stems:
        raise ValueError(f"Duplicate stem(s) in manifest_recordings.csv: {duplicate_stems}")

    needed_wavelengths = get_step2_needed_wavelengths(full_config)
    required_probability_wavelengths = set(get_step2_required_probability_wavelengths(full_config))
    registration_wavelength = get_registration_wavelength(full_config)
    required_ref_wavelengths = set(required_probability_wavelengths) | {registration_wavelength}

    required = ["stem"]
    for wl in sorted(required_ref_wavelengths, key=_wave_sort_key):
        required.append(f"ref_{wl}_path")
    for wl in sorted(required_probability_wavelengths, key=_wave_sort_key):
        required.append(f"prob_{wl}_path")

    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest is missing required Step2 columns for configured mask_sources/registration: {missing}. "
            f"needed_wavelengths={needed_wavelengths}, required_probability_wavelengths={sorted(required_probability_wavelengths, key=_wave_sort_key)}, "
            f"registration_wavelength={registration_wavelength}"
        )

    stem_keys = get_stem_keys(full_config)
    ag = get_analysis_group_config(full_config)
    group_by = ag["group_by"]

    for key in group_by:
        if key not in manifest.columns:
            raise ValueError(f"analysis_group.group_by key not found in manifest: {key}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    excluded_stems = set(full_config.get("excluded_stems") or [])

    # Retain only the highest repeat per mouse, day, state and protocol base.
    # This is computed from the WHOLE manifest before H5/excluded_stems checks.
    superseded: dict[str, str] = {}
    protocol_variant = ((full_config.get("variant_keys", {}) or {}).get("protocol", {}) or {})
    if protocol_variant.get("enabled", False):
        keys = ["mouse_id", "exp_day", "state", "protocol"]
        absent = [key for key in keys if key not in manifest.columns]
        if absent:
            raise ValueError(f"Cannot apply protocol repeat selection: missing manifest columns {absent}")
        separators = protocol_variant.get("separators", ["", "-", "_"])
        variants = manifest[keys + ["stem"]].copy()
        parsed = variants["protocol"].apply(lambda v: parse_variant_value(v, separators))
        variants["protocol_base"] = parsed.apply(lambda item: item[0])
        variants["protocol_repeat"] = parsed.apply(lambda item: item[1])
        grouped = ["mouse_id", "exp_day", "state", "protocol_base"]
        max_repeat = variants.groupby(grouped, dropna=False)["protocol_repeat"].transform("max")
        chosen = variants.loc[variants["protocol_repeat"].eq(max_repeat)]
        first_chosen = chosen.groupby(grouped, dropna=False, sort=False)["stem"].first()
        for _, v in variants.loc[variants["protocol_repeat"].lt(max_repeat)].iterrows():
            identity = tuple(v[key] for key in grouped)
            superseded[str(v["stem"])] = str(first_chosen.loc[identity])

    technical_cols = {
        "metadata_json", "replacements_json", "created_at", "stem_keys_json",
    }

    for _, row in manifest.iterrows():
        stem = str(row["stem"])
        out_row: dict[str, Any] = {
            "stem": stem,
            "manifest_path": str(manifest_path),
            "img_dir": row.get("img_dir", ""),
            "raw_zarr_path": row.get("raw_zarr_path", ""),
            "processed_zarr_path": row.get("processed_zarr_path", ""),
            "masks_dir": str(paths["masks_root"] / stem),
            "metadata_json": row.get("metadata_json", ""),
        }

        # Preserve all non-path metadata columns from manifest.
        for col in manifest.columns:
            if col in out_row or col in technical_cols:
                continue
            if col.startswith("ref_") or col.startswith("prob_"):
                continue
            out_row[col] = row.get(col, "")

        try:
            group_id = build_group_id(row, group_by)
        except Exception as e:
            out_row["error"] = f"group_id_error: {e!r}"
            errors.append(out_row)
            continue

        out_row["group_id"] = group_id
        out_row["group_keys_json"] = str(group_by)
        if stem in superseded:
            out_row["error"] = "superseded_by_higher_repeat"
            out_row["replaced_by_stem"] = superseded[stem]
            errors.append(out_row)
            continue
        if stem in excluded_stems:
            out_row["error"] = "excluded_by_config"
            errors.append(out_row)
            continue

        ok = True
        for wl in needed_wavelengths:
            ref_col = f"ref_{wl}_path"
            prob_col = f"prob_{wl}_path"
            ref_p = resolve_optional_path(row[ref_col], path_cfg) if ref_col in manifest.columns else None
            prob_p = resolve_optional_path(row[prob_col], path_cfg) if prob_col in manifest.columns else None
            out_row[f"bmp_{wl}"] = str(ref_p) if ref_p is not None else ""
            out_row[f"h5_{wl}"] = str(prob_p) if prob_p is not None else ""
            out_row[ref_col] = str(ref_p) if ref_p is not None else ""
            out_row[prob_col] = str(prob_p) if prob_p is not None else ""
            out_row[f"ref_{wl}_exists"] = bool(ref_p and ref_p.exists())
            out_row[f"prob_{wl}_exists"] = bool(prob_p and prob_p.exists())

            if wl in required_ref_wavelengths and not out_row[f"ref_{wl}_exists"]:
                ok = False
            if require_h5 and wl in required_probability_wavelengths and not out_row[f"prob_{wl}_exists"]:
                ok = False

        if ok:
            rows.append(out_row)
        else:
            out_row["error"] = "missing_reference_or_probability_file"
            errors.append(out_row)

    df = pd.DataFrame(rows)
    # Always refresh the missing report; an old list must not survive a clean rerun.
    missing_path = paths["step2_root"] / "step2_missing_inputs.csv"
    missing_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(errors, columns=["group_id", "stem", "error", "replaced_by_stem"] + [
        c for c in manifest.columns if c.startswith(("ref_", "prob_")) and c.endswith("_path")
    ]).to_csv(missing_path, index=False)
    if errors:
        print(f"[WARN] Step2 skipped {len(errors)} recording(s); see {missing_path}")
        for e in errors:
            print(f"  stem={e.get('stem')} group_id={e.get('group_id', '?')}: {e.get('error')}")

    if not df.empty and ("analysis_group" in full_config or "metadata" in full_config):
        variant_keys = ag.get("reference", {}).get("variant_keys", []) or []
        df = add_variant_columns(df, full_config, variant_keys=variant_keys)

    return df


# ============================================================
# Main Step 2 runner
# ============================================================

def _is_simple_metadata_value(value: Any) -> bool:
    """Accept scalar values suitable for metadata CSV cells."""
    if isinstance(value, (np.ndarray, Path, list, dict, tuple, set)):
        return False
    return True


def _metadata_from_record(rec: dict[str, Any], *, exclude_prefixes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Copy scalar metadata from an internal Step2 record."""
    technical = {
        "ring", "bad", "brain", "background", "bregma", "bregma_mask", "vessels",
        "registration_bmp", "registration_wavelength",
    }
    out: dict[str, Any] = {}
    for k, v in rec.items():
        if k in technical:
            continue
        if exclude_prefixes and any(str(k).startswith(prefix) for prefix in exclude_prefixes):
            continue
        if _is_simple_metadata_value(v):
            out[k] = v
    return out


def _reference_metadata(ref: dict[str, Any]) -> dict[str, Any]:
    """Create reference-prefixed metadata from the selected reference record."""
    keys = [
        "ref_stem", "group_id", "metadata_json", "stem_keys_json", "group_keys_json",
    ]
    out: dict[str, Any] = {}
    for k, v in _metadata_from_record(ref).items():
        if k in {"stem"}:
            out["ref_stem_original"] = v
        elif k in {"group_id", "metadata_json", "stem_keys_json", "group_keys_json"}:
            out[f"ref_{k}"] = v
        elif k not in {"img_dir", "raw_zarr_path", "processed_zarr_path", "masks_dir"} and not k.startswith(("bmp_", "h5_", "ref_", "prob_")):
            out[f"ref_meta_{k}"] = v
    return out



def add_raw_geometry_metadata(row: dict[str, Any], *, raw_scale: float, rotation_deg: float) -> dict[str, Any]:
    """Add coordinated full-resolution and scaled-image metadata fields.

    Parameters
    ----------
    row : dict[str, Any]
        Step2 output row to update.
    raw_scale : float
        ``raw.scale`` used later when constructing Step4 raw Zarr images.
    rotation_deg : float
        Project rotation in degrees, ``raw.rotation``.

    Returns
    -------
    dict[str, Any]
        The updated metadata dictionary.

    Notes
    -----
        Step2 masks, registration translations and bregma points use rotated
        full-resolution pixels; corresponding ``*_raw`` coordinates are scaled
        for Step4. The function does not resample Step2 masks.
    """
    row["project_raw_rotation_deg"] = float(rotation_deg)
    row["project_raw_scale"] = float(raw_scale)
    row["step2_image_space"] = "rotated_fullres"
    row["mask_image_space"] = "rotated_fullres"
    row["raw_zarr_image_space"] = "rotated_scaled"
    row["mask_to_raw_scale_y"] = float(raw_scale)
    row["mask_to_raw_scale_x"] = float(raw_scale)

    for col in [
        "bregma_ref_cy", "bregma_ref_cx",
        "bregma_cy", "bregma_cx",
        "init_shift_dy", "init_shift_dx",
        "second_stage_free_shift_dy", "second_stage_free_shift_dx",
        "translation_x", "translation_y",
        "bregma_registered_y", "bregma_registered_x",
    ]:
        if col in row:
            row[f"{col}_raw"] = scaled_coordinate_value(row.get(col), raw_scale)

    # Rotation angle itself is not affected by scale.
    if "rotation_rad" in row:
        row["rotation_rad_raw"] = row.get("rotation_rad")
    if "angle_deg_found" in row:
        row["angle_deg_found_raw"] = row.get("angle_deg_found")
    return row

def _base_bad_row(group_id: str, stem: str, ref_stem: str, rec: dict, ref: dict, ref_exact_match: bool, mask_mode: str) -> dict[str, Any]:
    """Initialize metadata for an unusable registration row."""
    row: dict[str, Any] = {
        "group_id": group_id,
        "stem": stem,
        "ref_stem": ref_stem,
        "ref_exact_match": bool(ref_exact_match),
        "bregma_ref_found": bool(ref.get("bregma_found", False)),
        "bregma_found": bool(rec.get("bregma_found", False)),
        "bregma_ref_cy": np.nan,
        "bregma_ref_cx": np.nan,
        "bregma_cy": np.nan,
        "bregma_cx": np.nan,
        "init_shift_dy": np.nan,
        "init_shift_dx": np.nan,
        "angle_deg_found": np.nan,
        "score": np.nan,
        "translation_x": np.nan,
        "translation_y": np.nan,
        "rotation_rad": np.nan,
        "mask_for_reg_area": 0,
        "mask_mode": mask_mode,
        "registration_method": "",
        "registration_roi_mode": "",
        "anchor_bregma": np.nan,
        "strict_bregma_anchor": np.nan,
        "bregma_error_px": np.nan,
        "second_stage_free_shift_dx": np.nan,
        "second_stage_free_shift_dy": np.nan,
        "is_bad": True,
        "bad_reasons": "",
        "fixed_registration_path": str(ref["registration_bmp"]),
        "moving_registration_path": str(rec["registration_bmp"]),
        "saved_registered_path": "",
    }

    # Preserve metadata from the current recording exactly as it came from manifest/Step2 index.
    for k, v in _metadata_from_record(rec).items():
        row.setdefault(k, v)

    # Add reference metadata with explicit ref_meta_ prefix.
    for k, v in _reference_metadata(ref).items():
        row.setdefault(k, v)

    return row


def merge_metadata_rerun_rows(existing: pd.DataFrame, rerun: pd.DataFrame) -> pd.DataFrame:
    """Replace recomputed (group_id, stem) rows while preserving other records."""
    if existing.empty:
        return rerun.copy()
    if rerun.empty:
        return existing.copy()

    old = existing.copy()
    new = rerun.copy()

    if {"group_id", "stem"}.issubset(old.columns) and {"group_id", "stem"}.issubset(new.columns):
        rerun_keys = set(zip(new["group_id"].astype(str), new["stem"].astype(str)))
        old_keys = list(zip(old["group_id"].astype(str), old["stem"].astype(str)))
        kept = old[[key not in rerun_keys for key in old_keys]].copy()
    elif "stem" in old.columns and "stem" in new.columns:
        rerun_stems = set(new["stem"].dropna().astype(str))
        kept = old[~old["stem"].astype(str).isin(rerun_stems)].copy()
    else:
        raise ValueError("Step2 rerun merge requires stem columns")

    # Align column union without dropping columns added by manual review.
    all_cols = list(dict.fromkeys(list(kept.columns) + list(new.columns)))
    kept = kept.reindex(columns=all_cols)
    new = new.reindex(columns=all_cols)
    merged = pd.concat([kept, new], ignore_index=True)

    sort_cols = [c for c in ["group_id", "stem"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return merged


def drop_skipped_manifest_stems(
    metadata: pd.DataFrame, missing_report: Path, excluded_stems: set[str]
) -> pd.DataFrame:
    """Remove excluded or input-invalid stems from saved Step2 metadata."""
    skipped_stems = set(excluded_stems)
    if missing_report.exists():
        missing = pd.read_csv(missing_report)
        if "stem" in missing.columns:
            skipped_stems.update(missing["stem"].dropna().astype(str))
    if skipped_stems and "stem" in metadata.columns:
        return metadata[~metadata["stem"].astype(str).isin(skipped_stems)].copy()
    return metadata


def run_step2(
    config_path: str | Path,
    *,
    require_h5: bool = True,
    group_only: str | None = None,
    from_drive: str | None = None,
    to_drive: str | None = None,
    show_progress: bool = True,
    manual_groups_only: bool = False,
    groups: list[str] | None = None,
) -> pd.DataFrame:
    """Generate canonical masks and group-level image registration metadata.

    Parameters
    ----------
    config_path : str or pathlib.Path
        YAML configuration file path.
    require_h5 : bool, default=True
        Whether required Ilastik probability files must be available.
    group_only : str or None, optional
        Restrict processing to a single ``group_id``.
    from_drive, to_drive : str or None, optional
        Drive remapping overrides for paths.
    show_progress : bool, default=True
        Display per-recording and per-group progress bars.
    manual_groups_only : bool, default=False
        Reprocess groups with previously saved manual bregma coordinates.
    groups : list[str] or None, optional
        Optional selection of group IDs.

    Returns
    -------
    pandas.DataFrame
        Step2 registration metadata also saved as
        ``out/step2_masks/metadata_registration.csv`` (subject to folder settings).

    Notes
    -----
        Ilastik masks and reference BMPs are in rotated, full-resolution Step2
        space. Records are grouped according to ``analysis_group.group_by``;
        reference selection uses ``analysis_group.reference``. Registration
        first aligns bregma and then estimates a scored second-stage transform.
        ``step2.qc.apply_thresholds`` defaults to false when absent; structural
        failures are still reported. Partial reruns merge recomputed rows with
        existing metadata rather than discarding unrelated rows.
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    project_paths = create_or_load_project(config, config_path)
    paths = build_standard_paths(config, project_paths)
    cfg = config.get("step2", {})
    ag = get_analysis_group_config(config)
    rotation_deg = raw_rotation_degrees(config)
    raw_scale = raw_scale_factor(config)

    path_cfg = get_path_resolving_config(config, "step2")
    if from_drive is None:
        from_drive = path_cfg.get("from_drive")
    if to_drive is None:
        to_drive = path_cfg.get("to_drive")

    # Reuse approved manual bregma coordinates when a group is reprocessed.
    manual_bregma_by_stem, manual_groups_from_metadata = load_manual_bregma_overrides(
        paths["metadata_registration_csv"]
    )

    # Validate manifest identities and the required wavelength-specific inputs.
    df = build_step2_index_from_manifest(
        manifest_path=paths["manifest_path"],
        cfg=config,
        paths=paths,
        require_h5=require_h5,
        from_drive=from_drive,
        to_drive=to_drive,
    )

    debug_filters = get_debug_filters(config)
    if group_only is not None:
        debug_filters = dict(debug_filters)
        debug_filters["group_id"] = str(group_only)
    if debug_filters:
        df = apply_metadata_filters_df(df, debug_filters, config)

    if manual_groups_only:
        if not manual_groups_from_metadata:
            raise RuntimeError(
                "No manual bregma groups found in previous metadata_registration.csv. "
                "Run step2_bregma_review_v2_1.py first."
            )
        df = df[df["group_id"].astype(str).isin(manual_groups_from_metadata)].copy()

    if groups:
        wanted_groups = {str(g) for g in groups}
        df = df[df["group_id"].astype(str).isin(wanted_groups)].copy()

    if df.empty:
        print("[WARN] No valid Step2 rows after manifest validation/filtering; see step2_missing_inputs.csv.")
        metadata_csv = paths["metadata_registration_csv"]
        if metadata_csv.exists():
            existing = pd.read_csv(metadata_csv)
            cleaned = drop_skipped_manifest_stems(
                existing, paths["step2_root"] / "step2_missing_inputs.csv",
                set(config.get("excluded_stems") or []),
            )
            if len(cleaned) != len(existing):
                cleaned.to_csv(metadata_csv, index=False)
        return pd.DataFrame()

    records: Dict[str, dict] = {}
    h5_dataset = cfg.get("h5", {}).get("dataset", "exported_data")
    morph = cfg.get("morphology", {})
    io_cfg = cfg.get("io", {})
    registration_wavelength = get_registration_wavelength(config)
    mask_wavelengths = get_step2_mask_wavelengths(config)

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Step2 masks", unit="recording")

    # Derive per-stem masks in rotated full-resolution image space.
    for _, r in iterator:
        stem = r["stem"]
        if show_progress:
            iterator.set_postfix(stem=str(stem)[:40])

        probs_by_wave: dict[str, np.ndarray] = {}
        for wl in mask_wavelengths:
            h5_value = r.get(f"h5_{wl}", "")
            if pd.notna(h5_value) and str(h5_value).strip() and Path(str(h5_value)).exists():
                probs_by_wave[wl] = read_probs_h5(Path(str(h5_value)), dataset=h5_dataset)

        brain, ring = build_brain_and_ring_from_sources(
            probs_by_wave=probs_by_wave,
            config_or_step2=config,
            close_r=int(morph.get("close_r_brain", 3)),
            r_ring=int(morph.get("ring_radius", 10)),
        )
        cy_auto, cx_auto, bregma_auto_mask, bregma_auto_found = bregma_centroid_from_sources(
            probs_by_wave=probs_by_wave,
            config_or_step2=config,
        )
        cy, cx = cy_auto, cx_auto
        bregma_mask = bregma_auto_mask
        bregma_found = bool(bregma_auto_found)
        bregma_source = "auto" if bregma_found else "missing"

        bregma_diameter_px = float(
            ((config.get("step3", {}) or {}).get("manual_mask_updates", {}) or {}).get("bregma_diameter_px", 25.0)
        )
        if str(stem) in manual_bregma_by_stem:
            cy, cx = manual_bregma_by_stem[str(stem)]
            bregma_mask = disk_mask(brain.shape, cy, cx, diameter_px=bregma_diameter_px)
            bregma_found = True
            bregma_source = "manual"

        bad = build_bad_mask_from_sources(probs_by_wave, config, brain.shape)
        vessels = combine_source_masks(probs_by_wave, config, "vessels", required=False)
        if vessels is None:
            vessels = np.zeros_like(brain, dtype=bool)

        mask_paths = canonical_mask_paths(paths, stem)
        d = mask_paths["brain_auto"].parent
        save_mask(mask_paths["brain_auto"], brain)
        save_mask(mask_paths["bad_auto"], bad)
        save_mask(mask_paths["ring_auto"], ring)
        save_mask(mask_paths["bregma_auto"], bregma_auto_mask)
        if bregma_source == "manual":
            save_mask(mask_paths["bregma_manual"], bregma_mask)
        if int(vessels.sum()) > 0 or iter_mask_source_entries(config, role="vessels"):
            save_mask(mask_paths["vessels_auto"], vessels)

        record = {
            k: v
            for k, v in r.to_dict().items()
            if _is_simple_metadata_value(v)
        }
        record.update({
            "stem": stem,
            "group_id": r.get("group_id", ""),
            "img_dir": r.get("img_dir", ""),
            "raw_zarr_path": r.get("raw_zarr_path", ""),
            "processed_zarr_path": r.get("processed_zarr_path", ""),
            "registration_wavelength": registration_wavelength,
            "registration_bmp": Path(r[f"bmp_{registration_wavelength}"]),
            "ring": ring,
            "bad": bad,
            "brain": brain,
            "vessels": vessels,
            "bregma": (cy, cx),
            "bregma_mask": bregma_mask,
            "bregma_found": bool(bregma_found),
            "bregma_source": bregma_source,
            "project_raw_rotation_deg": rotation_deg,
            "project_raw_scale": raw_scale,
            "step2_image_space": "rotated_fullres",
            "mask_image_space": "rotated_fullres",
            "raw_zarr_image_space": "rotated_scaled",
            "mask_file_logic": "canonical_by_stem",
            "mask_sources_json": str(get_mask_sources_config(config)),
        })
        for wl in get_step2_needed_wavelengths(config):
            bmp_key = f"bmp_{wl}"
            h5_key = f"h5_{wl}"
            if bmp_key in r.index and pd.notna(r[bmp_key]) and str(r[bmp_key]).strip():
                record[bmp_key] = Path(str(r[bmp_key]))
                record[f"ref_{wl}_path"] = str(r[bmp_key])
            if h5_key in r.index and pd.notna(r[h5_key]) and str(r[h5_key]).strip():
                record[h5_key] = str(r[h5_key])
                record[f"prob_{wl}_path"] = str(r[h5_key])
        records[stem] = record

    # Configure the second phase: reference selection and rigid registration.
    reg_cfg = cfg.get("registration", {}) or {}
    qc_cfg = cfg.get("qc", {})
    apply_qc_thresholds = bool(qc_cfg.get("apply_thresholds", False))
    ref_cfg = reg_cfg.get("reference", {})
    mask_mode = reg_cfg.get("mask_mode", "pairwise")
    registration_method = str(reg_cfg.get("method", "ring_sobel")).strip().lower()
    # For backward compatibility, old configs without roi_mode keep old ring/mask_mode behavior.
    # The new vessel_sobel option defaults to full-image ROI, matching the tested notebook workflow.
    default_roi_mode = "full_image" if registration_method == "vessel_sobel" else "ring"
    roi_mode = str(reg_cfg.get("roi_mode", default_roi_mode)).strip().lower()
    anchor_bregma = bool(reg_cfg.get("anchor_bregma", True))
    strict_bregma_anchor = bool(reg_cfg.get("strict_bregma_anchor", True))
    bregma_anchor_tol_px = float(reg_cfg.get("bregma_anchor_tol_px", 1e-3))
    ignore_border_px = int(reg_cfg.get("ignore_border_px", 0) or 0)
    rows: list[dict[str, Any]] = []

    group_items = list(df.groupby("group_id", sort=True))
    if show_progress:
        group_items = tqdm(group_items, desc="Step2 registration", unit="group")

    # Estimate transforms independently within each configured analysis group.
    for group_id, dfm in group_items:
        stems_m = list(dfm["stem"].values)
        ref_row, ref_exact_match = choose_reference_row_for_group(
            df_group=dfm,
            config=config,
            allow_fallback=True,
        )
        ref_stem = ref_row["stem"]
        ref = records[ref_stem]
        fixed_u8, fixed = load_bmp_u8_and_f32(ref["registration_bmp"])

        if not ref["bregma_found"]:
            for stem in sorted(stems_m):
                rec = records[stem]
                row = _base_bad_row(group_id, stem, ref_stem, rec, ref, ref_exact_match, mask_mode)
                row.update({"registration_method": registration_method, "registration_roi_mode": roi_mode, "anchor_bregma": anchor_bregma, "strict_bregma_anchor": strict_bregma_anchor, "bregma_anchor_tol_px": bregma_anchor_tol_px})
                row["bad_reasons"] = "ref_bregma_empty"
                cy, cx = rec["bregma"]
                row["bregma_cy"] = np.nan if cy is None else float(cy)
                row["bregma_cx"] = np.nan if cx is None else float(cx)
                row["bregma_source"] = rec.get("bregma_source", "missing")
                row["bregma_ref_source"] = ref.get("bregma_source", "missing")
                rows.append(add_raw_geometry_metadata(row, raw_scale=raw_scale, rotation_deg=rotation_deg))
            continue

        cy0, cx0 = ref["bregma"]
        group_union_ring = None
        group_union_bad = None
        group_union_mask_path = ""

        # Save shared ring masks; the scoring ROI is selected separately by roi_mode.
        if mask_mode in ("group_union", "hybrid"):
            recs_group = [records[s] for s in stems_m if s in records]
            group_union_ring, group_union_bad = build_group_union_mask(recs_group, ref_stem)
            group_union_mask = (group_union_ring & (~group_union_bad)).astype(bool)
            d_group = mask_for_reg_union_path(paths, group_id).parent
            save_mask(d_group / "ring_union.npy", group_union_ring)
            save_mask(d_group / "bad_union.npy", group_union_bad)
            group_union_mask_path = save_mask(mask_for_reg_union_path(paths, group_id), group_union_mask)

        if io_cfg.get("save_registered_bmps", True):
            save_bmp_uint8(paths["step2_root"] / "registered_bmps" / f"{ref_stem}_REF_{registration_wavelength}.bmp", fixed_u8)

        for stem in sorted(stems_m):
            rec = records[stem]
            moving_u8, moving = load_bmp_u8_and_f32(rec["registration_bmp"])

            if not rec["bregma_found"]:
                row = _base_bad_row(group_id, stem, ref_stem, rec, ref, ref_exact_match, mask_mode)
                row.update({"registration_method": registration_method, "registration_roi_mode": roi_mode, "anchor_bregma": anchor_bregma, "strict_bregma_anchor": strict_bregma_anchor, "bregma_anchor_tol_px": bregma_anchor_tol_px})
                row.update({
                    "bregma_ref_found": True,
                    "bregma_ref_cy": float(cy0),
                    "bregma_ref_cx": float(cx0),
                    "bad_reasons": "bregma_empty",
                    "bregma_source": rec.get("bregma_source", "missing"),
                    "bregma_ref_source": ref.get("bregma_source", "missing"),
                })
                rows.append(add_raw_geometry_metadata(row, raw_scale=raw_scale, rotation_deg=rotation_deg))
                continue

            cy, cx = rec["bregma"]
            init_dy = cy0 - cy
            init_dx = cx0 - cx
            init_shift = EuclideanTransform(translation=(init_dx, init_dy))
            moving0 = apply_tform(moving, init_shift, order=1)

            mask_for_reg = build_registration_roi(
                roi_mode=roi_mode,
                mask_mode=mask_mode,
                ref_rec=ref,
                mov_rec=rec,
                init_dy=init_dy,
                init_dx=init_dx,
                image_shape=fixed.shape,
                group_union_ring=group_union_ring,
                group_union_bad=group_union_bad,
                ignore_border_px=ignore_border_px,
            )
            mask_area = int(mask_for_reg.sum())
            per_stem_mask_path = save_mask(mask_for_reg_per_stem_path(paths, stem), mask_for_reg)

            angle_deg = 0.0
            score = float("-inf")
            total = init_shift
            _dy2, _dx2 = 0.0, 0.0
            bregma_error_px = np.nan

            if mask_area > 0:
                fixed_feature = build_registration_feature(fixed, mask_for_reg, reg_cfg)
                moving0_feature = build_registration_feature(moving0, mask_for_reg, reg_cfg)
                tform2, angle_deg, (_dy2, _dx2), score = estimate_feature_scored(
                    fixed_feature=fixed_feature,
                    moving0_feature=moving0_feature,
                    mask=mask_for_reg,
                    rot_center_yx=(cy0, cx0),
                    angle_range=float(reg_cfg.get("angle_range_deg", 3.0)),
                    angle_step=float(reg_cfg.get("angle_step_deg", 0.1)),
                    upsample=int(reg_cfg.get("upsample", 10)),
                    anchor_bregma=anchor_bregma,
                )
                # Correct skimage composition: apply initial bregma shift first,
                # then the second-stage rotation/free-shift transform.
                total = init_shift + tform2

            if anchor_bregma:
                bregma_error_px = bregma_anchor_error_px(total, (float(cy), float(cx)), (float(cy0), float(cx0)))
                if strict_bregma_anchor and np.isfinite(bregma_error_px) and bregma_error_px > bregma_anchor_tol_px:
                    raise RuntimeError(
                        f"Hard bregma anchor failed for stem={stem!r}: "
                        f"bregma_error_px={bregma_error_px:.6g}, tol={bregma_anchor_tol_px:.6g}. "
                        "This indicates transform composition or a free shift moved bregma."
                    )

            moved = apply_tform(moving, total, order=1)
            saved_registered_path = paths["step2_root"] / "registered_bmps" / f"{stem}_{registration_wavelength}_registered_to_{ref_stem}.bmp"
            if io_cfg.get("save_registered_bmps", True):
                save_bmp_uint8(saved_registered_path, moved)

            tx, ty = map(float, total.translation)
            reasons: list[str] = []
            if mask_area == 0:
                reasons.append("empty_registration_roi")
            elif not np.isfinite(score):
                reasons.append("invalid_registration_score")
            if apply_qc_thresholds:
                if not np.isfinite(score) or score < float(qc_cfg.get("min_score", 0.1)):
                    reasons.append("low_score")
                if mask_area < int(qc_cfg.get("min_mask_area", 500)):
                    reasons.append("small_mask")
                if abs(tx) > float(qc_cfg.get("max_abs_dx", 100)):
                    reasons.append("dx_large")
                if abs(ty) > float(qc_cfg.get("max_abs_dy", 100)):
                    reasons.append("dy_large")
                if abs(float(angle_deg)) > float(qc_cfg.get("max_abs_angle_deg", 10)):
                    reasons.append("angle_large")
            if anchor_bregma and np.isfinite(bregma_error_px) and bregma_error_px > bregma_anchor_tol_px:
                reasons.append("bregma_anchor_error")

            row: dict[str, Any] = {
                "group_id": group_id,
                "stem": stem,
                "ref_stem": ref_stem,
                "ref_exact_match": bool(ref_exact_match),
                "bregma_ref_found": True,
                "bregma_found": True,
                "bregma_ref_cy": float(cy0),
                "bregma_ref_cx": float(cx0),
                "bregma_cy": float(cy),
                "bregma_cx": float(cx),
                "bregma_source": rec.get("bregma_source", "auto"),
                "bregma_ref_source": ref.get("bregma_source", "auto"),
                "init_shift_dy": float(init_dy),
                "init_shift_dx": float(init_dx),
                "second_stage_free_shift_dy": float(_dy2),
                "second_stage_free_shift_dx": float(_dx2),
                "angle_deg_found": float(angle_deg),
                "score": float(score),
                "translation_x": float(tx),
                "translation_y": float(ty),
                "rotation_rad": float(total.rotation),
                "bregma_registered_y": transform_point_yx(total, float(cy), float(cx))[0],
                "bregma_registered_x": transform_point_yx(total, float(cy), float(cx))[1],
                "bregma_error_px": float(bregma_error_px) if np.isfinite(bregma_error_px) else np.nan,
                "anchor_bregma": bool(anchor_bregma),
                "strict_bregma_anchor": bool(strict_bregma_anchor),
                "bregma_anchor_tol_px": float(bregma_anchor_tol_px),
                "registration_method": registration_method,
                "registration_roi_mode": roi_mode,
                "registration_feature_method": registration_method,
                "vessel_sobel_params_json": str(reg_cfg.get("vessel_sobel", {}) or {}),
                "mask_for_reg_area": int(mask_area),
                "mask_mode": mask_mode,
                "mask_file_logic": "canonical_by_stem",
                "is_bad": bool(reasons),
                "bad_reasons": ",".join(reasons),
                "registration_wavelength": registration_wavelength,
                "fixed_registration_path": str(ref["registration_bmp"]),
                "moving_registration_path": str(rec["registration_bmp"]),
                "saved_registered_path": str(saved_registered_path) if io_cfg.get("save_registered_bmps", True) else "",
            }
            for k, v in _metadata_from_record(rec).items():
                row.setdefault(k, v)
            for k, v in _reference_metadata(ref).items():
                row.setdefault(k, v)
            rows.append(add_raw_geometry_metadata(row, raw_scale=raw_scale, rotation_deg=rotation_deg))

    out = pd.DataFrame(rows)
    metadata_csv = paths["metadata_registration_csv"]
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    # Preserve unrelated records during partial reruns and record the final metadata.
    previous_metadata = pd.read_csv(metadata_csv) if metadata_csv.exists() else None
    subset_rerun = bool(manual_groups_only or groups or group_only is not None or debug_filters)
    if subset_rerun and previous_metadata is not None:
        out_to_save = merge_metadata_rerun_rows(previous_metadata, out)
        print(
            f"Merged Step2 rerun into existing metadata_registration.csv: "
            f"rerun_rows={len(out)}, saved_rows={len(out_to_save)}"
        )
    else:
        out_to_save = out

    # A partial rerun must not retain previously calculated excluded stems.
    out_to_save = drop_skipped_manifest_stems(
        out_to_save, paths["step2_root"] / "step2_missing_inputs.csv",
        set(config.get("excluded_stems") or []),
    )
    out_to_save = drop_mask_path_metadata_columns(out_to_save)
    out_to_save = preserve_manual_bregma_provenance(previous_metadata, out_to_save)
    out_to_save.to_csv(metadata_csv, index=False)
    if "is_bad" in out_to_save.columns:
        out_to_save[out_to_save["is_bad"].fillna(False).astype(bool)].to_csv(paths["step2_root"] / "qc_bad.csv", index=False)
    else:
        pd.DataFrame().to_csv(paths["step2_root"] / "qc_bad.csv", index=False)
    return out_to_save


def main() -> None:
    """Parse command-line options and run Step2."""
    parser = argparse.ArgumentParser(description="Step 2: build masks and registration metadata from Step 1 manifest")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--group-only", default=None, help="Optional group_id for debugging")
    parser.add_argument("--from-drive", default=None, help='Optional Windows drive remap source, e.g. "D:"')
    parser.add_argument("--to-drive", default=None, help='Optional Windows drive remap target, e.g. "E:"')
    parser.add_argument("--allow-missing-h5", action="store_true", help="Do not require probability h5 files while building the index")
    parser.add_argument("--manual-groups-only", action="store_true", help="Re-run only groups that have bregma_source=manual in previous metadata_registration.csv")
    parser.add_argument("--groups", nargs="*", default=None, help="Optional group_id list to process")
    args = parser.parse_args()

    df = run_step2(
        config_path=args.config,
        require_h5=not args.allow_missing_h5,
        group_only=args.group_only,
        from_drive=args.from_drive,
        to_drive=args.to_drive,
        manual_groups_only=args.manual_groups_only,
        groups=args.groups,
    )
    print("Step2 finished")
    print("rows:", len(df))
    print("bad rows:", int(df["is_bad"].sum()) if "is_bad" in df else "NA")


if __name__ == "__main__":
    main()
