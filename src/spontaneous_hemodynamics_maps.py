"""Map reprojection routines preserved verbatim from the supplied Spontaneous v1.10.
Direct Step6/Step6b dependencies only; no Step7 dependency.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence
import cv2
import numpy as np
import pandas as pd
from skimage.transform import warp
import step6b_fc_geometry_coverage as s6b


def _safe_str(value: Any) -> str:
    """Convert an optional identifier to a safe string value.
    
        ``None`` and pandas missing values become an empty string; other values
        are stringified without altering their contents.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)

def _resize_continuous_nanaware(arr: np.ndarray, shape: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a continuous spatial map with NaN-aware normalization.
    
        Parameters
        ----------
        arr : numpy.ndarray
            Two-dimensional input map; NaN marks unavailable pixels.
        shape : tuple[int, int]
            Target spatial ``(height, width)``.
        interpolation : int, optional
            OpenCV interpolation method.
    
        Returns
        -------
        numpy.ndarray
            Float32 map on the requested grid, with NaN where no finite source
            contribution is available.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if tuple(arr.shape) == tuple(shape):
        return arr.copy()
    h, w = int(shape[0]), int(shape[1])
    valid = np.isfinite(arr).astype(np.float32)
    num = cv2.resize(np.nan_to_num(arr, nan=0.0), (w, h), interpolation=interpolation)
    den = cv2.resize(valid, (w, h), interpolation=interpolation)
    out = np.full((h, w), np.nan, dtype=np.float32)
    good = den > 1e-6
    out[good] = (num[good] / den[good]).astype(np.float32)
    return out


def _warp_continuous_stem_to_ref(
    arr_stem_full: np.ndarray,
    tform,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a full-resolution stem map into reference coordinates.
    
        Parameters
        ----------
        arr_stem_full : numpy.ndarray
            Continuous two-dimensional map in full-resolution stem pixels.
        tform : transform object
            Stem-to-reference transform supporting ``inverse``.
        ref_shape : tuple[int, int]
            Output reference height and width.
    
        Returns
        -------
        numpy.ndarray
            NaN-aware, linearly interpolated reference-space map.
    
        Notes
        -----
        Finite values and finite-pixel weights are warped separately before
        division to prevent missing samples from pulling the interpolated map
        toward zero.
    """
    arr = np.asarray(arr_stem_full, dtype=np.float32)
    val = np.nan_to_num(arr, nan=0.0).astype(np.float32)
    wt = np.isfinite(arr).astype(np.float32)
    num = warp(
        val,
        inverse_map=tform.inverse,
        output_shape=tuple(ref_shape),
        order=1,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    den = warp(
        wt,
        inverse_map=tform.inverse,
        output_shape=tuple(ref_shape),
        order=1,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)
    out = np.full(tuple(ref_shape), np.nan, dtype=np.float32)
    good = den > 1e-4
    out[good] = num[good] / den[good]
    return out


def stem_fc_map_to_reference(
    lag_fc: np.ndarray,
    row: pd.Series,
    geometry,
    config: dict[str, Any],
    project_paths: dict[str, Path],
    s6b_paths: dict[str, Path],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a stem's FC-grid spectral map onto its reference image.
    
        Parameters
        ----------
        lag_fc : numpy.ndarray
            Float spectral-power map on the stem's FC grid, with NaNs for invalid pixels.
        row : pandas.Series
            Recording and registration identifiers.
        geometry : Geometry
            Processed and full-resolution stem grid sizes.
        config : dict
            Main project configuration.
        project_paths, s6b_paths : dict
            Resolved project and Step6b paths.
    
        Returns
        -------
        tuple[numpy.ndarray, dict]
            Reference-space map and registration provenance metadata.
    
        Notes
        -----
        Uses saved Step6b reference labels and final valid-brain masks. The map
        is interpolated continuously but masked to valid labeled reference pixels.
    """
    mask_set = _safe_str(row.get("mask_set"))
    gid = _safe_str(row.get("group_id"))
    stem = _safe_str(row.get("stem"))
    mask_set_dir = Path(s6b_paths["mask_sets_root"]) / s6b.safe_key(mask_set, gid)
    labels_ref_path = mask_set_dir / "analysis_roi_ref_buffered_repaired.npy"
    stem_mask_path = mask_set_dir / "by_stem" / s6b.safe_key(stem) / "bilateral_final_usable_ref.npy"
    if not labels_ref_path.exists() or not stem_mask_path.exists():
        raise FileNotFoundError("Step6b reference geometry missing for spectral-map transform")
    ref_labels = np.load(labels_ref_path).astype(np.int32)
    ref_valid = np.load(stem_mask_path).astype(bool)
    ref_shape = tuple(ref_labels.shape)

    lag_processed = _resize_continuous_nanaware(lag_fc, geometry.processed_shape, cv2.INTER_LINEAR)
    lag_stem_full = _resize_continuous_nanaware(lag_processed, geometry.stem_fullres_shape, cv2.INTER_LINEAR)
    manual = s6b.load_manual_registration_overrides(config, project_paths)
    tform, source, meta = s6b.effective_tform_stem_to_ref(row, manual, config)
    lag_ref = _warp_continuous_stem_to_ref(lag_stem_full, tform, ref_shape)
    lag_ref[~(ref_valid & (ref_labels > 0))] = np.nan
    return lag_ref, {"transform_source": source, **{f"transform_{k}": v for k, v in meta.items()}}


def reference_map_to_canonical(
    lag_ref: np.ndarray,
    row: pd.Series,
    config: dict[str, Any],
    project_paths: dict[str, Path],
    s6b_paths: dict[str, Path],
    atlas_data: Any,
) -> np.ndarray:
    """Sample a reference-space spectral map onto the Allen canonical grid.
    
        Parameters
        ----------
        lag_ref : numpy.ndarray
            Reference-space two-dimensional map.
        row : pandas.Series
            Recording, reference stem and mask-set identifiers.
        config : dict
            Main project configuration for Step6 projection.
        project_paths, s6b_paths : dict
            Resolved project and Step6b paths.
        atlas_data : object
            Loaded atlas with ``flat_label`` defining canonical-grid dimensions.
    
        Returns
        -------
        numpy.ndarray
            Canonical two-dimensional map, NaN-masked outside valid coverage.
    
        Raises
        ------
        RuntimeError
            When the Allen atlas is not provided.
        FileNotFoundError
            If the corresponding Step6 transform metadata is absent.
    
        Notes
        -----
        Uses the saved Step6b canonical-to-reference projection and applies the
        optional stem-specific canonical valid mask and ROI labels.
    """
    gid = _safe_str(row.get("group_id"))
    ref_stem = _safe_str(row.get("ref_stem"))
    mask_set = _safe_str(row.get("mask_set"))
    if atlas_data is None:
        raise RuntimeError("atlas_data is required for canonical projection; load it once per batch")
    transform_path = s6b.group_transform_path(project_paths, gid, ref_stem)
    if not transform_path.exists():
        raise FileNotFoundError(f"Step6 atlas transform missing: {transform_path}")
    transform = s6b.load_json_file(transform_path)
    projection = s6b.project_canonical_grid_to_reference(
        atlas_data, transform, tuple(lag_ref.shape), config
    )
    # Step6b's canonical projection dictionary is intentionally treated as its
    # public geometry contract here.  It is the same object used by
    # sample_ref_mask_on_canonical().
    inside = np.asarray(projection["inside"], dtype=bool)
    ry = np.asarray(projection["ref_y"], dtype=int)
    rx = np.asarray(projection["ref_x"], dtype=int)
    canonical_shape = tuple(atlas_data.flat_label.shape)
    out = np.full(canonical_shape, np.nan, dtype=np.float32)
    out[inside] = lag_ref[ry[inside], rx[inside]]

    # Restrict to this stem's already-saved canonical final usable mask.
    can_stem = (
        Path(s6b_paths["canonical_masks_root"])
        / s6b.safe_key(mask_set, gid)
        / "by_stem"
        / s6b.safe_key(_safe_str(row.get("stem")))
        / "bilateral_final_usable_canonical.npy"
    )
    if can_stem.exists():
        valid_can = np.load(can_stem).astype(bool)
        if valid_can.shape == out.shape:
            out[~valid_can] = np.nan
    can_labels = Path(s6b_paths["canonical_root"]) / "analysis_roi_side_labels_buffered_canonical.npy"
    if can_labels.exists():
        lab = np.load(can_labels)
        if lab.shape == out.shape:
            out[lab == 0] = np.nan
    return out


def _nanmean_stack(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Average spatial maps over their finite samples on each pixel.
    
        Parameters
        ----------
        arrays : sequence of numpy.ndarray
            Maps with identical dimensions; NaN means missing measurement.
    
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Pixelwise float32 mean and int16 count of finite contributing maps.
    
        Raises
        ------
        ValueError
            If no input arrays are provided.
    """
    if not arrays:
        raise ValueError("No arrays to average")
    stack = np.stack([np.asarray(a, dtype=np.float32) for a in arrays], axis=0)
    count = np.sum(np.isfinite(stack), axis=0).astype(np.int16)
    total = np.nansum(stack, axis=0, dtype=np.float64)
    mean = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    good = count > 0
    mean[good] = (total[good] / count[good]).astype(np.float32)
    return mean, count


def load_allen_atlas_safe(config: dict[str, Any]):
    """Load the Allen atlas through the existing Step6 loader once per batch.
    
        Parameters
        ----------
        config : dict
            Main project configuration; reads ``step6.atlas.brainglobe_dir``.
    
        Returns
        -------
        object
            The Step6 atlas-data object expected by Step6b projection functions.
    
        Notes
        -----
        A missing/null/empty ``brainglobe_dir`` is omitted when constructing the
        BrainGlobe atlas to use its default cache directory. The temporary
        loader-global factory replacement is restored in a ``finally`` block.
    """
    loader = s6b.load_allen_atlas
    loader_globals = getattr(loader, "__globals__", {})
    factory = loader_globals.get("BrainGlobeAtlas")
    bg_dir = ((config.get("step6", {}) or {}).get("atlas", {}) or {}).get("brainglobe_dir")
    null_dir = bg_dir is None or str(bg_dir).strip().lower() in {"", "none", "null"}

    if factory is None or not null_dir:
        return loader(config)

    def _factory_without_null_dir(atlas_name, *args, **kwargs):
        """Call the BrainGlobe factory without an explicit null atlas directory.
        
                Parameters
                ----------
                atlas_name : str
                    Atlas identifier forwarded to the original factory.
                *args, **kwargs
                    Remaining BrainGlobe factory arguments.
        
                Returns
                -------
                object
                    Atlas instance created by the saved original factory.
        """
        value = kwargs.get("brainglobe_dir", None)
        if value is None or str(value).strip().lower() in {"", "none", "null"}:
            kwargs.pop("brainglobe_dir", None)
        return factory(atlas_name, *args, **kwargs)

    loader_globals["BrainGlobeAtlas"] = _factory_without_null_dir
    try:
        return loader(config)
    finally:
        loader_globals["BrainGlobeAtlas"] = factory


