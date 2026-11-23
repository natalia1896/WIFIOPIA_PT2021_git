"""Step 3 reference anatomical annotation helper, v2_9.

Based on the original ``step3_annotation.py`` UI and behavior, but keyed by
``group_id`` + ``ref_stem`` instead of old ``mouse``. Images are expected in rotated full-resolution Step2/Step3 space.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import napari
import numpy as np
import pandas as pd


ANNOTATION_COLUMNS = [
    "group_id",
    "ref_stem",
    "group_keys_json",
    "bregma_x", "bregma_y",
    "lambda_x", "lambda_y",
    "frontal_vessel_cross_x", "frontal_vessel_cross_y",
    "axis_p1_x", "axis_p1_y",
    "axis_p2_x", "axis_p2_y",
    "axis_angle_deg",
    "annotator",
    "notes",
    "updated_at",
]


# -------------------------------------------------
# IO helpers
# -------------------------------------------------

def load_reference_table(path: Path) -> pd.DataFrame:
    """Load the anatomical landmark table, ensuring the expected columns."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=ANNOTATION_COLUMNS)

    for col in ANNOTATION_COLUMNS:
        if col not in df.columns:
            if col.endswith(("_x", "_y")) or col == "axis_angle_deg":
                df[col] = np.nan
            else:
                df[col] = ""

    # New logic only: remove old mouse column if an older file is loaded.
    if "mouse" in df.columns:
        df = df.drop(columns=["mouse"])

    ordered = ANNOTATION_COLUMNS + [c for c in df.columns if c not in ANNOTATION_COLUMNS]
    return df[ordered]


def save_reference_table(df: pd.DataFrame, path: Path) -> None:
    """Persist the anatomical landmark table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------------------------------
# axis angle helper
# -------------------------------------------------

def compute_axis_angle(p1, p2):
    """Calculate the displayed midline angle from two (y, x) points.
    
    Parameters
    ----------
    p1, p2 : array-like
        Two endpoints with coordinate order ``(y, x)``, in image pixels.
    
    Returns
    -------
    float
        Midline orientation in degrees from ``atan2(dy, dx)``.
    
    Notes
    -----
    The definition follows the coordinate ordering used by the existing
    Napari annotation code; no anatomical orientation convention is inferred.
    """
    dy = p2[0] - p1[0]
    dx = p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def _valid_number(v: Any) -> bool:
    """Check whether a metadata scalar is non-missing."""
    try:
        return not pd.isna(v)
    except Exception:
        return False


def _point_or_nan(layer) -> tuple[float, float]:
    """Read the first Napari point as (y, x), or return missing coordinates."""
    if len(layer.data) == 0:
        return float("nan"), float("nan")
    y, x = np.asarray(layer.data, dtype=float)[0][:2]
    return float(y), float(x)


def _add_single_axis_line(axis_layer, p1_yx: tuple[float, float], p2_yx: tuple[float, float]) -> None:
    """Replace the Napari axis shapes with a single two-point line."""
    axis = np.asarray([p1_yx, p2_yx], dtype=float)
    axis_layer.data = []
    axis_layer.add(axis, shape_type="line")


def _extract_axis_line(axis_layer) -> tuple[np.ndarray, np.ndarray]:
    """Validate and retrieve the two endpoints of the midline axis."""
    if len(axis_layer.data) == 0:
        raise RuntimeError("midline axis not drawn")
    axis = np.asarray(axis_layer.data[0], dtype=float)
    if axis.ndim != 2 or axis.shape[0] < 2 or axis.shape[1] < 2:
        raise RuntimeError(f"midline axis has invalid shape: {axis.shape}")
    if axis.shape[0] != 2 or str(axis_layer.shape_type[0]) != "line":
        raise RuntimeError("midline axis must be exactly one line with two endpoints")
    return axis[0, :2], axis[1, :2]


def _require_annotation_config(annotation_cfg: dict[str, Any] | None) -> dict[str, bool]:
    """Build the required-landmark flags used during annotation."""
    cfg = annotation_cfg or {}
    return {
        "bregma": bool(cfg.get("require_bregma", True)),
        "lambda": bool(cfg.get("require_lambda", True)),
        "frontal_vessel_cross": bool(cfg.get("require_frontal_vessel_cross", True)),
        "axis": bool(cfg.get("require_axis", True)),
    }


# -------------------------------------------------
# annotation function
# -------------------------------------------------

def annotate_reference_group(
    group_id: str,
    ref_stem: str,
    ref_image_path: Path,
    annotations_csv: Path,
    annotator: str = "user",
    group_keys_json: str = "",
    annotation_cfg: dict[str, Any] | None = None,
):
    """Collect anatomical landmarks for the group's chosen reference image.
    
    Parameters
    ----------
    group_id : str
        Step2 registration group identifier.
    ref_stem : str
        Unique Step2 reference recording for the group.
    ref_image_path : pathlib.Path
        Existing rotated full-resolution reference image.
    annotations_csv : pathlib.Path
        Output table; one row is replaced for the same group/reference pair.
    annotator : str, optional
        Person or operator identifier stored with the annotation.
    group_keys_json : str, optional
        Serialized registration-group key metadata.
    annotation_cfg : dict or None, optional
        Landmark requirements from ``step3.annotation``.
    
    Notes
    -----
    Napari points are stored in ``(y, x)`` but CSV coordinates are named
    ``*_x`` and ``*_y``. A single midline line supplies two endpoints and
    an angle in degrees. The required landmarks are controlled by YAML,
    not by an assumed experiment. Closing Napari triggers validation and
    CSV persistence.
    """
    group_id = str(group_id)
    ref_stem = str(ref_stem)
    ref_image_path = Path(ref_image_path)
    annotations_csv = Path(annotations_csv)

    if not ref_image_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_image_path}")

    required = _require_annotation_config(annotation_cfg)
    img = iio.imread(ref_image_path)
    df = load_reference_table(annotations_csv)

    existing = df[(df["group_id"].astype(str) == group_id) & (df["ref_stem"].astype(str) == ref_stem)]

    viewer = napari.Viewer(title=f"Reference annotation: {group_id}")
    viewer.add_image(img, name="reference_image")

    # Keep original layer names/colors.
    bregma_layer = viewer.add_points(name="bregma", ndim=2, size=7, face_color="red")
    lambda_layer = viewer.add_points(name="lambda", ndim=2, size=7, face_color="blue")
    vessel_layer = viewer.add_points(name="frontal_vessel_cross", ndim=2, size=7, face_color="green")
    axis_layer = viewer.add_shapes(name="midline_axis", shape_type="line", edge_color="yellow", edge_width=2)

    # -------------------------------------------------
    # preload existing annotation
    # -------------------------------------------------
    if not existing.empty:
        r = existing.iloc[0]

        if _valid_number(r.get("bregma_x")):
            bregma_layer.data = np.array([[float(r.bregma_y), float(r.bregma_x)]])

        if _valid_number(r.get("lambda_x")):
            lambda_layer.data = np.array([[float(r.lambda_y), float(r.lambda_x)]])

        if _valid_number(r.get("frontal_vessel_cross_x")):
            vessel_layer.data = np.array([[float(r.frontal_vessel_cross_y), float(r.frontal_vessel_cross_x)]])

        if _valid_number(r.get("axis_p1_x")) and _valid_number(r.get("axis_p2_x")):
            _add_single_axis_line(
                axis_layer,
                (float(r.axis_p1_y), float(r.axis_p1_x)),
                (float(r.axis_p2_y), float(r.axis_p2_x)),
            )

    print()
    print("Annotate reference landmarks:")
    print("bregma -> red")
    print("lambda -> blue", "[required]" if required["lambda"] else "[optional]")
    print("frontal vessel cross -> green", "[required]" if required["frontal_vessel_cross"] else "[optional]")
    print("draw midline axis -> yellow", "[required]" if required["axis"] else "[optional]")
    print("close napari when finished")
    print()

    napari.run()

    # -------------------------------------------------
    # read annotations
    # -------------------------------------------------
    if required["bregma"] and len(bregma_layer.data) == 0:
        raise RuntimeError("bregma not placed")
    if required["lambda"] and len(lambda_layer.data) == 0:
        raise RuntimeError("lambda not placed")
    if required["frontal_vessel_cross"] and len(vessel_layer.data) == 0:
        raise RuntimeError("frontal vessel cross not placed")
    if required["axis"] and len(axis_layer.data) == 0:
        raise RuntimeError("midline axis not drawn")

    # Exactly one final landmark/axis; optional lambda is zero or one point.
    for name, layer, required_one in [
        ("bregma", bregma_layer, required["bregma"]),
        ("lambda", lambda_layer, required["lambda"]),
        ("frontal_vessel_cross", vessel_layer, required["frontal_vessel_cross"]),
    ]:
        count = len(layer.data)
        if count > 1 or (required_one and count != 1):
            raise RuntimeError(f"{name}: expected {'exactly one' if required_one else 'at most one'} point, got {count}")
    if len(axis_layer.data) > 1:
        raise RuntimeError("midline_axis: expected at most one line")
    if len(axis_layer.data) == 1:
        _extract_axis_line(axis_layer)

    bregma_y, bregma_x = _point_or_nan(bregma_layer)
    lambda_y, lambda_x = _point_or_nan(lambda_layer)
    vessel_y, vessel_x = _point_or_nan(vessel_layer)

    if len(axis_layer.data) > 0:
        p1, p2 = _extract_axis_line(axis_layer)
        axis_angle = compute_axis_angle(p1, p2)
    else:
        p1 = np.asarray([np.nan, np.nan], dtype=float)
        p2 = np.asarray([np.nan, np.nan], dtype=float)
        axis_angle = float("nan")

    row = {
        "group_id": group_id,
        "ref_stem": ref_stem,
        "group_keys_json": group_keys_json,
        "bregma_x": float(bregma_x),
        "bregma_y": float(bregma_y),
        "lambda_x": float(lambda_x),
        "lambda_y": float(lambda_y),
        "frontal_vessel_cross_x": float(vessel_x),
        "frontal_vessel_cross_y": float(vessel_y),
        "axis_p1_x": float(p1[1]),
        "axis_p1_y": float(p1[0]),
        "axis_p2_x": float(p2[1]),
        "axis_p2_y": float(p2[0]),
        "axis_angle_deg": axis_angle,
        "annotator": annotator,
        "notes": "",
        "updated_at": datetime.now().isoformat(),
    }

    keep = ~((df["group_id"].astype(str) == group_id) & (df["ref_stem"].astype(str) == ref_stem))
    df = df[keep].copy()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    ordered = ANNOTATION_COLUMNS + [c for c in df.columns if c not in ANNOTATION_COLUMNS]
    save_reference_table(df[ordered], annotations_csv)

    print()
    print("Saved annotation for group:", group_id)
    print("Reference stem:", ref_stem)
    print("Axis angle:", axis_angle)


# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    """Parse CLI inputs and open the reference-annotation viewer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-id", required=True)
    parser.add_argument("--ref-stem", required=True)
    parser.add_argument("--ref-image", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--annotator", default="user")
    parser.add_argument("--group-keys-json", default="")
    parser.add_argument("--require-bregma", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-lambda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-frontal-vessel-cross", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-axis", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    annotate_reference_group(
        group_id=args.group_id,
        ref_stem=args.ref_stem,
        ref_image_path=Path(args.ref_image),
        annotations_csv=Path(args.out_csv),
        annotator=args.annotator,
        group_keys_json=args.group_keys_json,
        annotation_cfg={
            "require_bregma": args.require_bregma,
            "require_lambda": args.require_lambda,
            "require_frontal_vessel_cross": args.require_frontal_vessel_cross,
            "require_axis": args.require_axis,
        },
    )


if __name__ == "__main__":
    main()
