"""Step 3 napari QC helper, v2_10.

This file is intentionally based on the original ``step3_review_mouse.py`` UI:
- raw_stack_registration
- registered_stack_registration
- mask_for_reg
- bregma_auto_raw / bregma_manual_raw
- bregma_auto_registered / bregma_manual_registered_preview / bregma_reference_anchor_registered
- brain_mask_auto_raw / brain_edge_auto_raw / brain_mask_manual_raw

Only the grouping/metadata logic was changed from old ``mouse`` to new
``group_id`` / ``stem`` / ``ref_stem``. Manual bregma corrections and manual brain mask paths are returned directly to the Step 3 runner, which patches only the corresponding
stem rows in metadata_registration.csv with bregma_source='manual'.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries
from skimage.transform import EuclideanTransform


FRAME_QC_COLUMNS = [
    "group_id", "stem", "ref_stem", "group_keys_json",
    "z_index",
    "bregma_auto_x", "bregma_auto_y",
    "bregma_manual_x", "bregma_manual_y",
    "bregma_source", "bregma_qc",
    "brain_mask_qc", "brain_mask_source", "brain_mask_manual_path",
    "vessel_mask_qc", "vessel_mask_notes",
    "registration_qc", "registration_notes",
    "needs_rerun", "annotator", "notes", "updated_at",
]


# -------------------------------------------------
# IO helpers
# -------------------------------------------------

def load_frame_qc_table(path: Path) -> pd.DataFrame:
    """Load a frame-level QC table, adding missing schema columns."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=FRAME_QC_COLUMNS)

    for col in FRAME_QC_COLUMNS:
        if col not in df.columns:
            if col in {"bregma_auto_x", "bregma_auto_y", "bregma_manual_x", "bregma_manual_y"}:
                df[col] = np.nan
            elif col == "needs_rerun":
                df[col] = False
            else:
                df[col] = ""

    # Keep extra metadata columns if they already exist.
    ordered = FRAME_QC_COLUMNS + [c for c in df.columns if c not in FRAME_QC_COLUMNS]
    return df[ordered]


def save_frame_qc_table(df: pd.DataFrame, path: Path) -> None:
    """Write frame-level QC records to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def pick_path_value(row: pd.Series, preferred: str, fallback: str) -> str:
    """Select a preferred metadata path or fall back to another column."""
    v = row.get(preferred, "")
    if isinstance(v, str) and v.strip():
        return v
    return row[fallback]


# -------------------------------------------------
# Stack loading
# -------------------------------------------------

def load_registered_images(df_group: pd.DataFrame) -> np.ndarray:
    """Load registered intensity images aligned with the Step2 reference.
    
    Parameters
    ----------
    df_group : pandas.DataFrame
        Group metadata in the required display order.
    
    Returns
    -------
    numpy.ndarray
        Stack with shape ``(stem_index, y, x)`` in full-resolution,
        rotated Step2/Step3 coordinates.
    """
    imgs = []
    for _, r in df_group.iterrows():
        if str(r["stem"]) == str(r["ref_stem"]):
            path = pick_path_value(r, "fixed_registration_path_resolved", "fixed_registration_path")
        else:
            path = pick_path_value(r, "saved_registered_path_resolved", "saved_registered_path")
        if not isinstance(path, str) or not path.strip():
            raise FileNotFoundError(
                f"Missing registered image path for stem={r.get('stem', '')}. "
                "Run Step2 with step2.io.save_registered_bmps: true."
            )
        imgs.append(iio.imread(path))
    return np.stack(imgs, axis=0)


def load_raw_images(df_group: pd.DataFrame) -> np.ndarray:
    """Load unregistered intensity images for the group.
    
    Parameters
    ----------
    df_group : pandas.DataFrame
        Group metadata in the required display order.
    
    Returns
    -------
    numpy.ndarray
        Stack with shape ``(stem_index, y, x)`` in full-resolution,
        rotated Step2/Step3 coordinates.
    """
    imgs = []
    for _, r in df_group.iterrows():
        if str(r["stem"]) == str(r["ref_stem"]):
            path = pick_path_value(r, "fixed_registration_path_resolved", "fixed_registration_path")
        else:
            path = pick_path_value(r, "moving_registration_path_resolved", "moving_registration_path")
        if not isinstance(path, str) or not path.strip():
            raise FileNotFoundError(f"Missing raw registration image path for stem={r.get('stem', '')}")
        imgs.append(iio.imread(path))
    return np.stack(imgs, axis=0)


def load_mask_stack(mask_path: Path | Sequence[Path], n_frames: int) -> np.ndarray:
    """Build a display-aligned registration-mask stack.
    
    Parameters
    ----------
    mask_path : pathlib.Path or sequence of pathlib.Path
        Shared union mask or one mask path per recording.
    n_frames : int
        Number of stems in the image stack.
    
    Returns
    -------
    numpy.ndarray
        Array of spatial masks with shape ``(stem_index, y, x)``.
    
    Raises
    ------
    ValueError
        If per-stem mask counts or shapes disagree.
    FileNotFoundError
        If a required mask file is absent.
    """
    if isinstance(mask_path, (list, tuple)):
        paths = [Path(p) for p in mask_path]
        if len(paths) != int(n_frames):
            raise ValueError(f"Expected {n_frames} per-stem mask paths, got {len(paths)}")
        masks = []
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Registration mask not found: {p}")
            masks.append(np.load(p))
        shapes = {tuple(m.shape) for m in masks}
        if len(shapes) != 1:
            raise ValueError(f"Per-stem registration masks have different shapes: {sorted(shapes)}")
        return np.stack(masks, axis=0)

    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Registration mask not found: {mask_path}")
    mask = np.load(mask_path)
    return np.repeat(mask[None, :, :], n_frames, axis=0)


# -------------------------------------------------
# Bregma points
# -------------------------------------------------

def load_auto_bregma_points_raw(df_group: pd.DataFrame) -> np.ndarray:
    """Build 3D points for automatically detected bregma positions."""
    pts = []
    for i, r in enumerate(df_group.itertuples()):
        cy = getattr(r, "bregma_cy", np.nan)
        cx = getattr(r, "bregma_cx", np.nan)
        if pd.notna(cy) and pd.notna(cx):
            pts.append([i, float(cy), float(cx)])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)




def load_reference_anchor_points_registered(df_group: pd.DataFrame) -> np.ndarray:
    """Create reference-bregma anchor points for each registered frame."""
    pts = []
    for i, r in enumerate(df_group.itertuples()):
        cy = getattr(r, "bregma_ref_cy", np.nan)
        cx = getattr(r, "bregma_ref_cx", np.nan)
        if pd.notna(cy) and pd.notna(cx):
            pts.append([i, float(cy), float(cx)])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)


def print_bregma_anchor_summary(df_group: pd.DataFrame) -> None:
    """Print the available Step2 bregma anchoring error statistics."""
    if "bregma_error_px" not in df_group.columns:
        return
    vals = pd.to_numeric(df_group["bregma_error_px"], errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return
    print(
        "Step2 bregma anchor error px: "
        f"max={float(vals.max()):.6g}, median={float(vals.median()):.6g}, n={len(vals)}"
    )

def load_existing_manual_bregma_points_raw(df_group: pd.DataFrame, qc_df: pd.DataFrame) -> np.ndarray:
    """Read manual bregma points from a pre-existing frame QC table."""
    pts = []
    if qc_df.empty:
        return np.empty((0, 3), dtype=float)

    for i, r in enumerate(df_group.itertuples()):
        q = qc_df[qc_df["stem"].astype(str) == str(r.stem)]
        if "group_id" in q.columns:
            q = q[q["group_id"].astype(str) == str(getattr(r, "group_id", ""))]
        if q.empty:
            continue
        q0 = q.iloc[-1]
        cy = q0.get("bregma_manual_y", np.nan)
        cx = q0.get("bregma_manual_x", np.nan)
        if pd.notna(cy) and pd.notna(cx):
            pts.append([i, float(cy), float(cx)])

    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)


# -------------------------------------------------
# Brain masks
# -------------------------------------------------


def load_existing_manual_bregma_points_from_metadata(df_group: pd.DataFrame) -> np.ndarray:
    """Read manual bregma positions stored in Step2 metadata."""
    pts = []
    if df_group.empty or "bregma_source" not in df_group.columns:
        return np.empty((0, 3), dtype=float)
    for i, r in enumerate(df_group.itertuples()):
        source = str(getattr(r, "bregma_source", "")).strip().lower()
        if source != "manual":
            continue
        cy = getattr(r, "bregma_cy", np.nan)
        cx = getattr(r, "bregma_cx", np.nan)
        if pd.notna(cy) and pd.notna(cx):
            pts.append([i, float(cy), float(cx)])
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)


def extract_manual_bregma_updates(df_group: pd.DataFrame, manual_points_layer) -> dict[str, tuple[float, float]]:
    """Collect user-placed bregma locations from the editable Napari layer.
    
    Parameters
    ----------
    df_group : pandas.DataFrame
        Metadata table ordered as in the image stack.
    manual_points_layer : napari.layers.Points
        Points with columns ``(stem_index, y, x)``.
    
    Returns
    -------
    dict of str to tuple of float
        ``stem -> (cy, cx)`` in rotated full-resolution image pixels.
    
    Notes
    -----
    When several points address one stem, the last point wins.
    """
    manual_pts = np.asarray(manual_points_layer.data, dtype=float)
    out: dict[str, tuple[float, float]] = {}
    for p in manual_pts:
        if len(p) != 3:
            continue
        t, y, x = p
        t = int(round(float(t)))
        if 0 <= t < len(df_group):
            stem = str(df_group.iloc[t]["stem"])
            out[stem] = (float(y), float(x))
    return out

def load_auto_brain_masks(
    df_group: pd.DataFrame,
    binmasks_dir: Path,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Load a complete stack of canonical automatic brain masks."""
    stack = np.zeros((len(df_group), image_shape[0], image_shape[1]), dtype=np.uint8)
    for i, r in enumerate(df_group.itertuples()):
        stem = str(r.stem)
        p = Path(binmasks_dir) / stem / "brain_auto.npy"
        if not p.is_file():
            raise FileNotFoundError(f"missing brain_auto.npy: stem={stem}, path={p}")
        m = np.load(p)
        if tuple(m.shape) != tuple(image_shape):
            raise ValueError(f"brain_auto.npy shape_mismatch: stem={stem}, path={p}, shape={m.shape}, expected={image_shape}")
        stack[i] = (m > 0).astype(np.uint8)
    return stack

def build_brain_edge_stack(brain_auto_stack: np.ndarray) -> np.ndarray:
    """Compute outer boundaries of binary brain masks for display."""
    edges = np.zeros_like(brain_auto_stack, dtype=np.uint8)
    for i in range(brain_auto_stack.shape[0]):
        edges[i] = find_boundaries(brain_auto_stack[i] > 0, mode="outer").astype(np.uint8)
    return edges


def load_existing_manual_brain_masks(
    df_group: pd.DataFrame,
    qc_df: pd.DataFrame,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Load previously saved manual brain masks from frame QC paths."""
    stack = np.zeros((len(df_group), image_shape[0], image_shape[1]), dtype=np.uint8)

    if qc_df.empty:
        return stack

    for i, r in enumerate(df_group.itertuples()):
        q = qc_df[qc_df["stem"].astype(str) == str(r.stem)]
        if "group_id" in q.columns:
            q = q[q["group_id"].astype(str) == str(getattr(r, "group_id", ""))]
        if q.empty:
            continue

        q0 = q.iloc[-1]
        p = q0.get("brain_mask_manual_path", "")
        if isinstance(p, str) and p.strip():
            path = Path(p)
            if path.exists():
                m = np.load(path)
                if m.shape == image_shape:
                    stack[i] = (m > 0).astype(np.uint8)

    return stack


# -------------------------------------------------
# Coordinate transforms
# -------------------------------------------------

def transform_point_row(row: pd.Series, y: float, x: float) -> tuple[float, float]:
    """Transform one landmark from recording to reference coordinates.
    
    Parameters
    ----------
    row : pandas.Series
        Step2 registration transform metadata for a recording.
    y, x : float
        Point coordinates in rotated full-resolution pixels.
    
    Returns
    -------
    tuple of float
        Registered ``(y, x)`` point in reference pixels, or NaN values
        if a needed transform parameter is missing.
    
    Notes
    -----
    Reference recordings are unchanged; other recordings use the Step2
    ``EuclideanTransform`` defined by ``rotation_rad`` and translations.
    """
    if str(row["stem"]) == str(row["ref_stem"]):
        return float(y), float(x)

    tx = row.get("translation_x", np.nan)
    ty = row.get("translation_y", np.nan)
    rot = row.get("rotation_rad", np.nan)

    if pd.isna(tx) or pd.isna(ty) or pd.isna(rot):
        return np.nan, np.nan

    tform = EuclideanTransform(rotation=float(rot), translation=(float(tx), float(ty)))
    xy = np.array([[float(x), float(y)]], dtype=float)
    xy_t = tform(xy)[0]
    return float(xy_t[1]), float(xy_t[0])


def transform_points_raw_to_registered(df_group: pd.DataFrame, pts_raw: np.ndarray) -> np.ndarray:
    """Map Napari points into the group's registered image space.
    
    Parameters
    ----------
    df_group : pandas.DataFrame
        Step2 rows ordered as in the Napari stack.
    pts_raw : numpy.ndarray
        Coordinates with shape ``(n_points, 3)`` and columns
        ``(stem_index, y, x)`` in the unregistered image space.
    
    Returns
    -------
    numpy.ndarray
        Transformed point array with shape ``(n_valid_points, 3)``;
        records with missing transforms are excluded.
    """
    if pts_raw.size == 0:
        return np.empty((0, 3), dtype=float)

    reg_pts = []
    for p in pts_raw:
        t, y, x = p
        t = int(round(float(t)))
        if t < 0 or t >= len(df_group):
            continue
        row = df_group.iloc[t]
        y_t, x_t = transform_point_row(row, y, x)
        if pd.notna(y_t) and pd.notna(x_t):
            reg_pts.append([t, y_t, x_t])

    return np.asarray(reg_pts, dtype=float) if reg_pts else np.empty((0, 3), dtype=float)


# -------------------------------------------------
# QC row helpers
# -------------------------------------------------

def _metadata_payload(row: pd.Series, exclude: set[str] | None = None) -> dict[str, Any]:
    """Copy scalar metadata fields suitable for inclusion in a QC row."""
    exclude = exclude or set()
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in exclude:
            continue
        if k.endswith("_resolved") or k.endswith("_exists"):
            continue
        # Avoid storing huge or technical path fields twice in frame_qc.
        if k in {
            "fixed_registration_path", "moving_registration_path", "saved_registered_path",
            "mask_for_reg_path", "brain_path", "bad_path", "ring_path",
            "background_path", "bregma_mask_path",
        }:
            continue
        if isinstance(v, (str, int, float, bool, np.integer, np.floating)) or pd.isna(v):
            out[k] = v
    return out


def make_base_qc_row(row: pd.Series, z_index: int, annotator: str, now: str) -> dict[str, Any]:
    """Construct default frame QC metadata for one group recording."""
    base = {
        "group_id": str(row["group_id"]),
        "stem": row["stem"],
        "ref_stem": row["ref_stem"],
        "group_keys_json": row.get("group_keys_json", ""),
        "z_index": int(z_index),
        "bregma_auto_x": row.get("bregma_cx", np.nan),
        "bregma_auto_y": row.get("bregma_cy", np.nan),
        "bregma_manual_x": np.nan,
        "bregma_manual_y": np.nan,
        "bregma_source": "auto",
        "bregma_qc": "ok",
        "brain_mask_qc": "ok",
        "brain_mask_source": "auto",
        "brain_mask_manual_path": "",
        "vessel_mask_qc": "ok",
        "vessel_mask_notes": "",
        "registration_qc": "ok",
        "registration_notes": "",
        "needs_rerun": False,
        "annotator": annotator,
        "notes": "",
        "updated_at": now,
    }
    extra = _metadata_payload(row, exclude=set(base))
    base.update(extra)
    return base


def ensure_qc_row(qc_df: pd.DataFrame, row: pd.Series, z_index: int, annotator: str, now: str) -> tuple[pd.DataFrame, int]:
    """Find or append the QC row corresponding to a recording."""
    if "stem" not in qc_df.columns:
        qc_df["stem"] = ""
    existing = qc_df[qc_df["stem"].astype(str) == str(row["stem"])]
    if "group_id" in existing.columns:
        existing = existing[existing["group_id"].astype(str) == str(row["group_id"])]

    if existing.empty:
        qc_df = pd.concat([qc_df, pd.DataFrame([make_base_qc_row(row, z_index, annotator, now)])], ignore_index=True)
        idx = qc_df.index[-1]
    else:
        idx = existing.index[-1]
        qc_df.at[idx, "z_index"] = int(z_index)
        qc_df.at[idx, "group_id"] = str(row["group_id"])
        qc_df.at[idx, "ref_stem"] = row["ref_stem"]
        qc_df.at[idx, "group_keys_json"] = row.get("group_keys_json", "")

    return qc_df, int(idx)


# -------------------------------------------------
# Save user edits
# -------------------------------------------------

def save_manual_bregma_from_layer(
    df_group: pd.DataFrame,
    qc_df: pd.DataFrame,
    manual_points_layer,
    annotator: str = "user",
) -> pd.DataFrame:
    """Update frame QC records from a Napari bregma points layer."""
    manual_pts = np.asarray(manual_points_layer.data, dtype=float)
    manual_map: dict[int, tuple[float, float]] = {}

    # If several red points are placed on one frame, the last one wins.
    for p in manual_pts:
        if len(p) != 3:
            continue
        t, y, x = p
        t = int(round(float(t)))
        if 0 <= t < len(df_group):
            manual_map[t] = (float(y), float(x))

    now = datetime.now().isoformat()

    for i, (_, row) in enumerate(df_group.iterrows()):
        qc_df, idx = ensure_qc_row(qc_df, row, i, annotator, now)

        if i in manual_map:
            y, x = manual_map[i]
            qc_df.at[idx, "bregma_manual_x"] = x
            qc_df.at[idx, "bregma_manual_y"] = y
            qc_df.at[idx, "bregma_source"] = "manual"
            qc_df.at[idx, "bregma_qc"] = "corrected"
            qc_df.at[idx, "needs_rerun"] = True
        else:
            if pd.isna(qc_df.at[idx, "bregma_manual_x"]) or pd.isna(qc_df.at[idx, "bregma_manual_y"]):
                qc_df.at[idx, "bregma_source"] = "auto"
                if pd.isna(qc_df.at[idx, "bregma_qc"]) or str(qc_df.at[idx, "bregma_qc"]).strip() == "":
                    qc_df.at[idx, "bregma_qc"] = "ok"

        qc_df.at[idx, "annotator"] = annotator
        qc_df.at[idx, "updated_at"] = now

    return qc_df


def save_manual_brain_masks_from_layer(
    df_group: pd.DataFrame,
    qc_df: pd.DataFrame,
    brain_mask_layer,
    manual_masks_dir: Path,
    annotator: str = "user",
) -> pd.DataFrame:
    """Save edited brain label masks and update frame QC records."""
    manual_masks_dir = Path(manual_masks_dir)
    manual_masks_dir.mkdir(parents=True, exist_ok=True)
    data = np.asarray(brain_mask_layer.data)
    now = datetime.now().isoformat()

    for i, (_, row) in enumerate(df_group.iterrows()):
        qc_df, idx = ensure_qc_row(qc_df, row, i, annotator, now)
        m = data[i]

        if np.any(m > 0):
            out_path = manual_masks_dir / f"{row['stem']}_brain_mask.npy"
            np.save(out_path, (m > 0).astype(np.uint8))

            qc_df.at[idx, "brain_mask_manual_path"] = str(out_path)
            qc_df.at[idx, "brain_mask_source"] = "manual"
            qc_df.at[idx, "brain_mask_qc"] = "corrected"
            qc_df.at[idx, "needs_rerun"] = True
        else:
            existing_path = qc_df.at[idx, "brain_mask_manual_path"]
            if not isinstance(existing_path, str) or not existing_path.strip():
                qc_df.at[idx, "brain_mask_source"] = "auto"
                if pd.isna(qc_df.at[idx, "brain_mask_qc"]) or str(qc_df.at[idx, "brain_mask_qc"]).strip() == "":
                    qc_df.at[idx, "brain_mask_qc"] = "ok"

        qc_df.at[idx, "annotator"] = annotator
        qc_df.at[idx, "updated_at"] = now

    return qc_df


# -------------------------------------------------
# Main review function
# -------------------------------------------------

def review_group_df(
    df_group: pd.DataFrame,
    group_id: str,
    mask_path: Path | Sequence[Path],
    manual_masks_dir: Path,
    binmasks_dir: Path,
    annotator: str = "user",
    auto_bregma_size: int = 6,
    manual_bregma_size: int = 8,
) -> dict[str, Any]:
    """Display a group for mask/bregma QC and collect optional manual edits.
    
    Parameters
    ----------
    df_group : pandas.DataFrame
        Step2 registration metadata for a single ``group_id``.
    group_id : str
        Identifier displayed in the Napari window.
    mask_path : pathlib.Path or sequence of pathlib.Path
        Registration ROI mask used as a visual overlay.
    manual_masks_dir : pathlib.Path
        Directory for editable per-stem manual brain masks.
    binmasks_dir : pathlib.Path
        Canonical Step2 mask root containing ``<stem>/brain_auto.npy``.
    annotator : str, optional
        Operator label for review provenance.
    auto_bregma_size, manual_bregma_size : int, optional
        Point sizes used for Napari visualization, in screen/image units.
    
    Returns
    -------
    dict
        ``manual_bregma_updates`` maps stems to ``(cy, cx)`` points;
        ``manual_brain_mask_paths`` maps stems to edited NPY files;
        ``manual_brain_reverted_stems`` lists stems returned to auto masks.
    
    Notes
    -----
    The editable layer is in rotated full-resolution Step2/Step3 space,
    not ``raw.scale``-reduced Step4 space. Existing manual masks are shown
    on the initial layer. The runner must forward reverted stems to the
    canonical-mask writer to remove an old manual mask completely.
    """
    if df_group.empty:
        raise RuntimeError("df_group is empty")
    if "group_id" not in df_group.columns:
        raise ValueError("df_group must contain group_id")
    if "stem" not in df_group.columns or "ref_stem" not in df_group.columns:
        raise ValueError("df_group must contain stem and ref_stem")

    group_id = str(group_id)
    df_group = df_group.copy().reset_index(drop=True)

    raw_stack = load_raw_images(df_group)
    reg_stack = load_registered_images(df_group)
    mask_stack = load_mask_stack(mask_path, len(df_group))

    auto_raw = load_auto_bregma_points_raw(df_group)
    manual_raw = load_existing_manual_bregma_points_from_metadata(df_group)

    auto_reg = transform_points_raw_to_registered(df_group, auto_raw)
    manual_reg = transform_points_raw_to_registered(df_group, manual_raw)
    ref_anchor_reg = load_reference_anchor_points_registered(df_group)

    brain_auto_stack = load_auto_brain_masks(
        df_group=df_group,
        binmasks_dir=binmasks_dir,
        image_shape=raw_stack.shape[1:],
    )
    brain_edge_stack = build_brain_edge_stack(brain_auto_stack)

    brain_manual_stack = np.zeros_like(brain_auto_stack, dtype=np.uint8)
    for i, r in df_group.iterrows():
        stem = str(r["stem"])
        candidates = [
            Path(manual_masks_dir) / f"{stem}_brain_mask.npy",
            Path(binmasks_dir) / stem / "brain_manual.npy",
        ]
        for p in candidates:
            if p.exists():
                m = np.load(p)
                if m.shape == raw_stack.shape[1:]:
                    brain_manual_stack[i] = (m > 0).astype(np.uint8)
                    break

    # Keep original behavior: if no manual mask exists, initialize manual layer from auto mask.
    empty_frames = (brain_manual_stack.sum(axis=(1, 2)) == 0)
    brain_manual_stack[empty_frames] = brain_auto_stack[empty_frames]

    viewer = napari.Viewer(title=f"Group QC: {group_id}")

    # Keep the original layer names/layout so the UI feels the same.
    viewer.add_image(raw_stack, name="raw_stack_registration", visible=False)
    viewer.add_image(reg_stack, name="registered_stack_registration")

    viewer.add_labels(mask_stack.astype(np.uint8), name="mask_for_reg", opacity=0.30)

    viewer.add_points(
        auto_raw,
        name="bregma_auto_raw",
        size=auto_bregma_size,
        face_color="cyan",
        visible=False,
    )

    manual_raw_layer = viewer.add_points(
        manual_raw,
        name="bregma_manual_raw",
        size=manual_bregma_size,
        face_color="red",
        visible=False,
    )

    viewer.add_points(
        auto_reg,
        name="bregma_auto_registered",
        size=auto_bregma_size,
        face_color="cyan",
    )

    viewer.add_points(
        ref_anchor_reg,
        name="bregma_reference_anchor_registered",
        size=max(3, int(auto_bregma_size * 0.75)),
        face_color="yellow",
        visible=True,
    )

    viewer.add_points(
        manual_reg,
        name="bregma_manual_registered_preview",
        size=manual_bregma_size,
        face_color="red",
    )

    viewer.add_labels(
        brain_auto_stack.astype(np.uint8),
        name="brain_mask_auto_raw",
        opacity=0.20,
        visible=False,
    )

    viewer.add_labels(
        brain_edge_stack.astype(np.uint8),
        name="brain_edge_auto_raw",
        opacity=0.70,
        visible=False,
    )

    brain_manual_layer = viewer.add_labels(
        brain_manual_stack.astype(np.uint8),
        name="brain_mask_manual_raw",
        opacity=0.45,
        visible=False,
    )

    print()
    print("Group:", group_id)
    print("Use raw_stack_registration for editing (rotated full-resolution Step2/Step3 space):")
    print("  - bregma_manual_raw")
    print("  - brain_mask_manual_raw")
    print("Useful raw overlays:")
    print("  - brain_mask_auto_raw")
    print("  - brain_edge_auto_raw")
    print("Use registered_stack_registration for QC of alignment.")
    print("  - bregma_auto_registered should overlap bregma_reference_anchor_registered when Step2 anchor_bregma=True")
    print("Manual data are saved in Step2 coordinates: rotated full-resolution, not raw.scale-resized.")
    print_bregma_anchor_summary(df_group)
    print()
    print("Frame order:")
    for i, r in df_group.iterrows():
        print(i, r["stem"])

    napari.run()

    manual_bregma_updates = extract_manual_bregma_updates(df_group, manual_raw_layer)

    # Save manual brain masks as predictable per-stem files.
    # A manual mask is saved only when the current label layer differs from the
    # Step2 auto brain mask for that stem. If an old manual file exists but the
    # user reverts the layer back to auto, remove the stale manual file.
    manual_masks_dir = Path(manual_masks_dir)
    manual_masks_dir.mkdir(parents=True, exist_ok=True)
    brain_data = np.asarray(brain_manual_layer.data)
    manual_brain_mask_paths: dict[str, str] = {}
    manual_brain_reverted_stems: list[str] = []
    for i, (_, row) in enumerate(df_group.iterrows()):
        stem = str(row["stem"])
        out_path = manual_masks_dir / f"{stem}_brain_mask.npy"
        canonical_manual_path = Path(binmasks_dir) / stem / "brain_manual.npy"
        m = (brain_data[i] > 0).astype(np.uint8)
        auto_m = brain_auto_stack[i].astype(np.uint8)
        if m.shape != auto_m.shape:
            raise RuntimeError(f"Manual brain mask shape mismatch for stem={stem}: {m.shape} vs {auto_m.shape}")

        changed_from_auto = bool(np.any(m != auto_m))
        if changed_from_auto:
            np.save(out_path, m)
            manual_brain_mask_paths[stem] = str(out_path)
        else:
            # Allow reverting a previous manual mask to auto by editing the layer
            # back to the auto mask before closing napari. The Step3 runner will
            # also remove canonical brain_manual.npy if it exists.
            if out_path.exists():
                out_path.unlink()
            if canonical_manual_path.exists():
                manual_brain_reverted_stems.append(stem)

    print()
    print(f"Manual bregma updates: {len(manual_bregma_updates)} stem(s)")
    print(f"Manual brain masks saved: {len(manual_brain_mask_paths)} stem(s)")
    print(f"Manual brain masks reverted to auto: {len(manual_brain_reverted_stems)} stem(s)")
    print(f"Manual masks folder -> {manual_masks_dir}")
    return {
        "manual_bregma_updates": manual_bregma_updates,
        "manual_brain_mask_paths": manual_brain_mask_paths,
        "manual_brain_reverted_stems": manual_brain_reverted_stems,
    }


# -------------------------------------------------
# CLI
# -------------------------------------------------

def review_group(
    group_id: str,
    metadata_csv: Path,
    mask_path: Path,
    manual_masks_dir: Path,
    binmasks_dir: Path,
    annotator: str = "user",
    auto_bregma_size: int = 6,
    manual_bregma_size: int = 8,
) -> None:
    """Load one group from Step2 metadata and open interactive QC."""
    df = pd.read_csv(metadata_csv)
    if "group_id" not in df.columns:
        raise ValueError("metadata_registration.csv must contain group_id")
    df_group = df[df["group_id"].astype(str) == str(group_id)].copy()

    if df_group.empty:
        raise RuntimeError(f"No rows for group_id={group_id}")

    review_group_df(
        df_group=df_group,
        group_id=group_id,
        mask_path=mask_path,
        manual_masks_dir=manual_masks_dir,
        binmasks_dir=binmasks_dir,
        annotator=annotator,
        auto_bregma_size=auto_bregma_size,
        manual_bregma_size=manual_bregma_size,
    )


def main() -> None:
    """Parse standalone group-review CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--manual-masks-dir", required=True)
    parser.add_argument("--binmasks-dir", required=True)
    parser.add_argument("--annotator", default="user")
    parser.add_argument("--auto-bregma-size", type=int, default=6)
    parser.add_argument("--manual-bregma-size", type=int, default=8)
    args = parser.parse_args()

    review_group(
        group_id=args.group_id,
        metadata_csv=Path(args.metadata),
        mask_path=Path(args.mask),
        manual_masks_dir=Path(args.manual_masks_dir),
        binmasks_dir=Path(args.binmasks_dir),
        annotator=args.annotator,
        auto_bregma_size=args.auto_bregma_size,
        manual_bregma_size=args.manual_bregma_size,
    )


if __name__ == "__main__":
    main()
