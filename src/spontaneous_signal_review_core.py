"""Read-only PT2021 spontaneous-signal visual QC core.

Uses Step6b accepted ROIs and the same reference->stem->processed->FC-grid,
mask-aware area resampling, linear detrending and SOS filtering as the project
Step7 functional-connectivity implementation. Never updates project files or analysis decisions.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.signal import butter, detrend, sosfiltfilt
from skimage.transform import AffineTransform, EuclideanTransform, warp


SIGNALS = ["dcHbT", "dcHbO", "dcHHb", "dcHbT_vreg", "dcHbO_vreg", "dcHHb_vreg"]
FILTERS_REST = ["No filter", "Low-pass ≤0.4 Hz", "Band-pass 0.08–0.4 Hz"]
FILTERS_FULL = FILTERS_REST + ["Slow band 0.009–0.08 Hz"]


def truth(value):
    """Interpret a saved table value as a truth-like inclusion flag.
    
        Only the case-insensitive strings 1, true, yes and y count as true."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def skey(*parts):
    """Create a filesystem-safe key for a combination of identifiers.
    
        Joins parts with double underscores and replaces path separators and colons."""
    return "__".join(str(p) for p in parts).replace("/", "-").replace("\\", "-").replace(":", "-")


def read_table(path: Path, required=True):
    """Read a Step6 or Step6b CSV table.
    
        Parameters
        ----------
        path : Path
            CSV file to load.
        required : bool, optional
            Raise FileNotFoundError when absent; otherwise return an empty table.
    
        Returns
        -------
        pandas.DataFrame
            The CSV contents or an empty table for an optional missing file."""
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required Step6b/Step6 table not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    return df


class Study:
    """Read-only project context for spontaneous-signal visual quality control.
    
        Reads the project YAML, Step6 index, Step6b selection/eligibility CSVs
        and optional final REST segments. It does not update these sources."""
    def __init__(self, config_path: str | Path):
        """Load the project YAML, resolve analysis directories and read Step6 metadata.
        
                Parameters
                ----------
                config_path : str or Path
                    Path to the project YAML. The project root is read from
                    ``project.directory_to_save_main_analysis``, with optional
                    ``path_resolving`` substitutions.
        
                Raises
                ------
                ValueError
                    If YAML is not a mapping or ``raw.fps`` is invalid.
                FileNotFoundError
                    If required analysis paths or Step6 tables are unavailable."""
        self.config_path = Path(config_path).expanduser().resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(self.cfg, dict):
            raise ValueError("Expected YAML mapping")
        project = self.cfg.get("project", {})
        root = self.locate(project.get("directory_to_save_main_analysis"))
        if root is None or not root.is_dir():
            raise FileNotFoundError(f"Analysis folder not found: {root}; check path_resolving")
        self.out = root / project.get("out_folder_name", "out")
        self.s6b_root = self.out / self.cfg.get("step6b_fc_geometry", {}).get("folder_name", "step6b_fc_geometry")
        s6root = self.out / self.cfg.get("step6", {}).get("folder_name", "step6_atlas_transform")
        s6_index_name = self.cfg.get("step6", {}).get("output", {}).get("step6_index_name", "step6_index.csv")
        self.s6_index = read_table(s6root / "tables" / s6_index_name)
        if self.s6_index["stem"].duplicated().any():
            # group/stem identity must not be ambiguous
            if self.s6_index.duplicated(["group_id", "stem"]).any():
                raise ValueError("Duplicate (group_id,stem) rows in Step6 index")
        self.rest = read_table(self.out / "behavior_review" / "rest_segments_final.csv", required=False)
        self.fps = float(self.cfg.get("raw", {}).get("fps", 0))
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("raw.fps must be positive")
        self._tables = {}

    def locate(self, raw):
        """Resolve a stored path, applying configured drive/root substitutions.
        
                Returns the first existing candidate, or the first candidate even if
                it does not exist; returns None for missing or blank input."""
        if raw is None or not str(raw).strip() or str(raw).strip().lower() == 'nan':
            return None
        raw = str(raw).strip()
        pr = self.cfg.get("path_resolving", {}) or {}
        options = [raw]
        a, b = pr.get("from_root"), pr.get("to_root")
        if a and b and raw.lower().startswith(str(a).lower()):
            options.insert(0, str(b) + raw[len(str(a)):])
        a, b = pr.get("from_drive"), pr.get("to_drive")
        if a and b and raw[:2].lower() == str(a)[:2].lower():
            options.insert(0, str(b)[:2] + raw[2:])
        for val in options:
            p = Path(val)
            if p.exists():
                return p
        return Path(options[0])

    def tables(self, variant):
        """Read and cache selection and eligibility tables for one Step6b branch.
        
                Parameters
                ----------
                variant : {"with_rest", "without_rest"}
                    Step6b processing branch.
        
                Returns
                -------
                dict
                    Selection, ROI inclusion, homotopic/global coverage and optional
                    hemisphere-assignment tables plus the branch root directory."""
        if variant not in ("with_rest", "without_rest"):
            raise ValueError(variant)
        if variant not in self._tables:
            root = self.s6b_root / variant
            t = root / "tables"
            self._tables[variant] = {
                "root": root,
                "selection": read_table(t / "mask_set_selection.csv"),
                "selected": read_table(t / "selected_rois_by_mask_set.csv"),
                "hom": read_table(t / "roi_homotopic_stem_usable_coverage.csv"),
                "global": read_table(t / "roi_global_stem_usable_coverage.csv"),
                "hemi": read_table(t / "hemisphere_assignment.csv", required=False),
            }
        return self._tables[variant]

    def mask_sets(self, variant):
        """List mask sets whose Step6b selection status is ``ok``."""
        t = self.tables(variant)
        return sorted(t["selection"].loc[t["selection"]["status"].astype(str).eq("ok"), "mask_set"].astype(str).unique())

    def stems(self, variant, mask_set):
        """List eligible recording rows for a selected Step6b branch and mask set.
        
                Results are sorted by mouse and stem and include a display description."""
        t = self.tables(variant)
        sel = t["selection"]
        s = sel[(sel["mask_set"].astype(str) == str(mask_set)) & sel["status"].astype(str).eq("ok")].copy()
        if s.empty:
            return []
        s["description"] = s["mouse_id"].astype(str) + " | " + s["stem"].astype(str) + " | " + s["slot"].astype(str)
        return s.sort_values(["mouse_id", "stem"]).to_dict("records")

    def source_row(self, stem, group_id):
        """Find the unique Step6 source-index row for a stem and group.
        
                Raises RuntimeError if no row or more than one row matches."""
        data = self.s6_index
        sel = data[data["stem"].astype(str).eq(str(stem)) & data["group_id"].astype(str).eq(str(group_id))]
        if len(sel) != 1:
            raise RuntimeError(f"Expected one Step6 source row: {stem}, group={group_id}, found={len(sel)}")
        return sel.iloc[0]

    def accepted_rois(self, variant, mask_set, stem, group_id):
        """Select ROI entries that passed both cohort and recording eligibility.
        
                Parameters
                ----------
                variant : {"with_rest", "without_rest"}
                    Step6b branch containing saved selection and coverage decisions.
                mask_set : str
                    Selected Step6b mask-set name.
                stem, group_id : str
                    Recording and registration-group identifiers.
        
                Returns
                -------
                list of dict
                    Selected global or homotopic ROI metadata. The group/cohort
                    ``include`` flag and per-stem PASS flag must both be true.
        
                Notes
                -----
                Each homotopic ROI is one selection entry and subsequently produces
                separate left and right temporal traces."""
        t = self.tables(variant)
        inc = t["selected"]
        inc = inc[(inc["mask_set"].astype(str) == str(mask_set)) & inc["include"].map(truth)]
        out = []
        for _, record in inc.iterrows():
            analysis = str(record["analysis"])
            if analysis not in {"homotopic", "global"}:
                continue
            key = str(record["roi_key"])
            src = t["hom"] if analysis == "homotopic" else t["global"]
            stem_key = "pair_key" if analysis == "homotopic" else "global_roi_key"
            pass_key = "homotopic_stem_pass" if analysis == "homotopic" else "global_stem_pass"
            matched = src[(src["stem"].astype(str) == str(stem)) &
                          (src["group_id"].astype(str) == str(group_id)) &
                          (src["mask_set"].astype(str) == str(mask_set)) &
                          src[stem_key].astype(str).eq(key)]
            if not len(matched) or not matched[pass_key].map(truth).any():
                continue
            rid = int(record["broad_structure_id"])
            display = f"{analysis}: {key}"
            out.append({"analysis": analysis, "key": key, "roi_id": rid,
                        "label": display, "hemisphere": str(record.get("hemisphere", ""))})
        return out

    def segments(self, stem, state, nframes):
        """List time windows for the current recording.
        
                Iso recordings yield nonoverlapping complete 60-second windows
                starting at zero. Other states use the saved
                ``behavior_review/rest_segments_final.csv`` start/end times.
                Candidate windows exceeding ``nframes`` are excluded. The GUI
                controls whether these windows are shown or the full recording."""
        if str(state).strip().lower() == "iso":
            step = int(round(60 * self.fps))
            return [{"label": f"Iso {k+1}: {a/self.fps:.1f}–{(a+step)/self.fps:.1f}s",
                     "t0": a/self.fps, "t1": (a+step)/self.fps}
                    for k, a in enumerate(range(0, nframes-step+1, step))]
        if self.rest.empty:
            return []
        table = self.rest[self.rest["stem"].astype(str).eq(str(stem))]
        out = []
        for _, r in table.sort_values("t0_sec").iterrows():
            start, end = float(r["t0_sec"]), float(r["t1_sec"])
            if end <= start or int(round(end*self.fps)) > nframes:
                continue
            out.append({"label": f"REST {len(out)+1}: {start:.2f}–{end:.2f}s",
                        "t0": start, "t1": end})
        return out

    def signal_path(self, row):
        """Find the processed Step4 Zarr directory for a Step6 index row.
        
                Tries the resolved/stored path first, followed by the conventional
                Step4 processed-signals path within the analysis output directory."""
        p = self.locate(row.get("processed_zarr_path_resolved", row.get("processed_zarr_path", None)))
        if p is not None and p.is_dir():
            return p
        stem = str(row["stem"])
        cfg4 = self.cfg.get("step4", {})
        folder = cfg4.get("folder_name", "step4_raw_and_signals")
        proc = cfg4.get("processed_signals_folder", "processed_signals")
        fallback = self.out / folder / proc / (stem + "_processed.zarr")
        if fallback.is_dir():
            return fallback
        raise FileNotFoundError(f"Processed Zarr not found. Declared={p}, fallback={fallback}")


@dataclass
class Geometry:
    """Final per-recording masks reconstructed on the FC analysis grid.
    
        Attributes store processed-resolution and FC-resolution usable masks,
        left/right hemisphere masks, region labels, mirrored pixel pairs and
        the explicitly assigned healthy/lesion hemisphere sides."""
    fc_shape: tuple
    mask_processed: np.ndarray
    mask_fc: np.ndarray
    left_fc: np.ndarray
    right_fc: np.ndarray
    labels_fc: np.ndarray
    pairs: dict
    healthy_side: str
    lesion_side: str


def transform_from_step6(row):
    """Reconstruct the saved Step3 spatial transform referenced by Step6.
    
        Parameters
        ----------
        row : pandas.Series or mapping
            Step6 index row. Saved 3-by-3 affine parameters take precedence
            over manual translation/rotation, followed by registration columns.
    
        Returns
        -------
        skimage.transform.AffineTransform
            Transform from the saved registration metadata; identity for a
            reference recording. Malformed or incomplete manual overrides raise."""
    if str(row["stem"]) == str(row["ref_stem"]):
        return AffineTransform()
    raw = row.get("step3_final_transform_xy_json")
    if isinstance(raw, str) and raw.strip():
        matrix = np.asarray(json.loads(raw), dtype=float)
        if (matrix.shape != (3,3) or not np.all(np.isfinite(matrix)) or
            not np.allclose(matrix[2], [0,0,1], atol=1e-9) or
            abs(np.linalg.det(matrix[:2,:2])) < 1e-10):
            raise ValueError("Malformed saved Step3 affine: " + str(row["stem"]))
        return AffineTransform(matrix=matrix)
    keys = ["step3_final_translation_x", "step3_final_translation_y", "step3_final_rotation_rad"]
    vals = [row.get(k, np.nan) for k in keys]
    present = [pd.notna(v) and str(v).strip() for v in vals]
    if any(present):
        if not all(present):
            raise ValueError("Incomplete Step3 manual registration: " + str(row["stem"]))
        return AffineTransform(matrix=EuclideanTransform(rotation=float(vals[2]), translation=(float(vals[0]),float(vals[1]))).params)
    if truth(row.get("step3_manual_override_present", False)):
        raise ValueError("Manual override present but without valid transform: " + str(row["stem"]))
    keys = ["translation_x", "translation_y", "rotation_rad"]
    vals = [float(row.get(k, np.nan)) for k in keys]
    if not all(np.isfinite(vals)):
        raise ValueError("Missing Step2 transform: " + str(row["stem"]))
    return AffineTransform(matrix=EuclideanTransform(rotation=vals[2],translation=(vals[0],vals[1])).params)


def nearest_bool(a, shape):
    """Resize a 2D Boolean mask using nearest-neighbor interpolation."""
    return cv2.resize(a.astype(np.uint8), (shape[1],shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def nearest_labels(a, shape):
    """Resize a 2D region-label map with nearest-neighbor interpolation."""
    return np.rint(cv2.resize(a.astype(np.float32), (shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)).astype(np.int32)


def project_ref(arr, tform, shape, boolean):
    # Output stem -> reference maps through tform (same direction as Step7).
    """Warp a reference-space mask or label map into the stem image grid.
    
        ``boolean`` selects binary thresholding versus integer label output;
        nearest-neighbor resampling avoids mixing region identifiers."""
    out = warp(np.asarray(arr,dtype=np.float32),inverse_map=tform,output_shape=shape,
               order=0,mode="constant",cval=0,preserve_range=True)
    return out > 0.5 if boolean else np.rint(out).astype(np.int32)


def reflect(points, origin, direction):
    """Reflect XY pixel coordinates across an axis given by origin/direction.
    
        Both the input coordinates and returned coordinates use (x, y) order."""
    delta = points - origin
    projected = origin + np.outer(delta @ direction, direction)
    return 2*projected - points


def homotopic_pairs(labels, bilateral, left, right, origin, direction, side,
                    require_same_roi=True, mutual_nearest=True):
    """Build mutually mirrored valid pixel pairs on the FC grid.
    
        Parameters
        ----------
        labels : numpy.ndarray, shape (y, x)
            Integer atlas-region labels at FC resolution.
        bilateral, left, right : numpy.ndarray of bool
            Final usable bilateral and hemisphere masks on that grid.
        origin, direction : array_like, shape (2,)
            Point and unit direction of the projected AP/midline axis, in (x, y).
        side : {"left", "right"}
            Hemisphere whose pixels are stored as source indices.
        require_same_roi : bool, optional
            Require source and mirrored partner to have equal region labels.
        mutual_nearest : bool, optional
            Require a rounded reverse reflection to return to the original pixel.
    
        Returns
        -------
        dict
            Flattened source/destination pixel indices, source ROI IDs and the
            selected source side. Only valid bilateral, contralateral pairs remain.
    
        Notes
        -----
        This reconstruction uses the same pairing rules as the Step7 FC analysis.
        It does not recalculate Step6b inclusion or coverage decisions."""
    h,w = labels.shape
    src = left if side == "left" else right
    opp = right if side == "left" else left
    sy,sx = np.nonzero(src & (labels>0))
    src_xy = np.stack((sx,sy),axis=1).astype(float)
    mirrored = reflect(src_xy,origin,direction)
    dx,dy = np.rint(mirrored[:,0]).astype(int), np.rint(mirrored[:,1]).astype(int)
    inside = (dx>=0)&(dx<w)&(dy>=0)&(dy<h)
    partner_roi = np.zeros(len(sy),dtype=np.int32)
    partner_hemi = np.zeros(len(sy),dtype=bool)
    partner_good = np.zeros(len(sy),dtype=bool)
    partner_roi[inside] = labels[dy[inside],dx[inside]]
    partner_hemi[inside] = opp[dy[inside],dx[inside]]
    partner_good[inside] = bilateral[dy[inside],dx[inside]]
    rid = labels[sy,sx]
    potential = inside & partner_hemi & (partner_roi>0)
    if require_same_roi:
        potential &= rid == partner_roi
    if mutual_nearest and np.any(potential):
        ix = np.flatnonzero(potential)
        back = reflect(np.stack((dx[ix],dy[ix]),axis=1).astype(float),origin,direction)
        pass_back = (np.rint(back[:,0]).astype(int)==sx[ix]) & (np.rint(back[:,1]).astype(int)==sy[ix])
        keep = np.zeros(len(potential),dtype=bool)
        keep[ix[pass_back]] = True
        potential &= keep
    valid = potential & bilateral[sy,sx] & partner_good
    ix = np.flatnonzero(valid)
    return {"src": np.ravel_multi_index((sy[ix],sx[ix]),(h,w)),
            "dst": np.ravel_multi_index((dy[ix],dx[ix]),(h,w)),
            "roi": rid[ix].astype(np.int32), "map_side":side}


def load_geometry(study: Study, variant, mask_set, row, proc_shape):
    """Reconstruct final per-stem ROI geometry from saved Step6b products.
    
        Parameters
        ----------
        study : Study
            Read-only metadata and mask-set tables.
        variant : {"with_rest", "without_rest"}
            Step6b processing branch.
        mask_set : str
            Name of the selected mask set.
        row : pandas.Series or mapping
            Step6 source row with the stem/reference registration and bregma/AP axis.
        proc_shape : tuple of int
            Processed Step4 signal image shape in (y, x) order.
    
        Returns
        -------
        Geometry
            Final usable masks and analysis ROI labels in the processed grid and
            in the FC grid, with mirrored pixel pairs and hemisphere identities.
    
        Notes
        -----
        Reads the saved reference-space masks, projects them through the Step3/6
        transform to each stem, and resizes them to the FC grid using
        ``functional_connectivity.spatial.scale``. The midline uses saved bregma
        and AP-axis annotations. Step6b eligibility is not recomputed here."""
    t = study.tables(variant)
    gid,ref,stem = str(row["group_id"]),str(row["ref_stem"]),str(row["stem"])
    base = t["root"] / "mask_sets_refspace" / skey(mask_set,gid)
    stemdir = base / "by_stem" / skey(stem)
    refdir = t["root"] / "reference" / skey(gid,ref)
    required = {
        "bilateral":stemdir/"bilateral_final_usable_ref.npy",
        "labels":base/"analysis_roi_ref_buffered_repaired.npy",
        "left":refdir/"left_hemisphere_ref.npy",
        "right":refdir/"right_hemisphere_ref.npy",
        "healthy":stemdir/"global_healthy_final_usable_ref.npy",
    }
    for key,p in required.items():
        if not p.is_file():
            raise FileNotFoundError(f"Step6b mask missing ({key}): {p}")
    raw = {k: np.load(v) for k,v in required.items()}
    shape = raw["bilateral"].shape
    if any(v.shape != shape for v in raw.values()):
        raise ValueError("Reference-space masks have differing shapes")
    tform = transform_from_step6(row)
    # Step7 requires moving registration image for full-resolution stem shape.
    stem_shape = shape
    for col in ("moving_registration_path_resolved", "moving_registration_path", "fixed_registration_path_resolved", "fixed_registration_path"):
        p = study.locate(row.get(col))
        if p is not None and p.is_file():
            img = cv2.imread(str(p),cv2.IMREAD_UNCHANGED)
            if img is not None:
                stem_shape = img.shape[:2]
                break
    if str(row["stem"]) != str(row["ref_stem"]) and stem_shape == shape:
        # Same image dimension is valid; warn caller only if no accessible path.
        pass
    scale = float(study.cfg.get("functional_connectivity", {}).get("spatial",{}).get("scale",0.5))
    if scale <= 0:
        raise ValueError("FC spatial.scale must be positive")
    fc_shape = tuple(max(1,int(round(v*scale))) for v in proc_shape)
    def resize(k, boolean):
        """Project one named reference-space mask to processed and FC image grids.
        
                Returns a pair of resampled arrays at processed and FC resolution."""
        full = project_ref(raw[k],tform,stem_shape,boolean)
        proc = nearest_bool(full,proc_shape) if boolean else nearest_labels(full,proc_shape)
        fc = nearest_bool(proc,fc_shape) if boolean else nearest_labels(proc,fc_shape)
        return proc, fc
    proc_mask, bilat = resize("bilateral",True)
    _, label = resize("labels",False)
    _, left = resize("left",True)
    _, right = resize("right",True)
    _, healthy = resize("healthy",True)
    hemi = t["hemi"]
    hemi = hemi[hemi["group_id"].astype(str).eq(gid)] if not hemi.empty else hemi
    if not hemi.empty:
        lesion = str(hemi.iloc[0].get("lesion_side", "right")).lower()
    else:
        lesion = str(study.cfg.get("step6b_fc_geometry",{}).get("hemisphere",{}).get("lesion_side","right")).lower()
    if lesion not in {"left", "right"}:
        raise ValueError("Missing explicit lesion side for geometry")
    healthy_side = "left" if lesion == "right" else "right"
    left &= bilat
    right &= bilat
    healthy &= (left if healthy_side == "left" else right)
    if healthy_side == "left":
        left &= healthy  # same constraint as Step7
    else:
        right &= healthy
    setting = study.cfg.get("step6b_fc_geometry", {})
    hom_mode = setting.get("mask_sets",{}).get(mask_set,{}).get("homotopic_map_hemisphere") or setting.get("hemisphere",{}).get("homotopic_map_hemisphere","lesion")
    display_side = {"healthy":healthy_side, "lesion":lesion}.get(str(hom_mode),str(hom_mode))
    if display_side not in {"left", "right"}:
        raise ValueError(f"Unknown homotopic hemisphere: {display_side}")
    # Reproduce Step7 midline basis using saved bregma/AP axis (direction sign irrelevant).
    ax1 = np.asarray([float(row["ann_axis_p1_x"]),float(row["ann_axis_p1_y"])])
    ax2 = np.asarray([float(row["ann_axis_p2_x"]),float(row["ann_axis_p2_y"])])
    origin = np.asarray([float(row["ann_bregma_x"]),float(row["ann_bregma_y"])])
    ap = ax2-ax1
    ap /= np.linalg.norm(ap)
    origin = ax1 + np.dot(origin-ax1,ap)*ap
    length = max(stem_shape)*0.75
    refs = np.stack((origin-ap*length,origin+ap*length))
    stempts = tform.inverse(refs)
    scale_xy = np.array([(fc_shape[1]-1)/max(stem_shape[1]-1,1),(fc_shape[0]-1)/max(stem_shape[0]-1,1)])
    fpts = stempts * scale_xy
    direction = fpts[1]-fpts[0]
    direction /= np.linalg.norm(direction)
    hc = study.cfg.get("functional_connectivity",{}).get("homotopic",{})
    pairs = homotopic_pairs(label,bilat,left,right,fpts[0],direction,display_side,
                            hc.get("require_same_analysis_roi",True), hc.get("mutual_nearest_partner",True))
    return Geometry(fc_shape,proc_mask,bilat,left,right,label,pairs,healthy_side,lesion)


def roi_mask_items(geometry, accepted):
    """Convert accepted ROI metadata into final usable FC-grid masks.
    
        Parameters
        ----------
        geometry : Geometry
            Reconstructed per-stem FC-grid masks and mirrored pixel pairs.
        accepted : sequence of dict
            ROI selections returned by ``Study.accepted_rois``.
    
        Returns
        -------
        list of tuple
            Each item is ``(display_label, binary_mask_yx)``. Global ROIs provide
            one hemisphere-specific mask; homotopic entries provide separate
            left and right masks formed only from valid mirrored pixel pairs.
            ROI entries without usable pixels are skipped."""
    out = []
    for r in accepted:
        rid = int(r["roi_id"])
        if r["analysis"] == "global":
            key = str(r["key"])
            hemi = key.rsplit("_",1)[-1]
            if hemi not in ("left","right"):
                continue
            side_mask = geometry.left_fc if hemi=="left" else geometry.right_fc
            m = (geometry.labels_fc == rid) & geometry.mask_fc & side_mask
            if m.any():
                out.append((r["label"], m))
        else:
            p = geometry.pairs
            selected = p["roi"] == rid
            if not selected.any():
                continue
            left = np.zeros(geometry.fc_shape,dtype=bool)
            right = np.zeros_like(left)
            flat_l = p["src"][selected] if p["map_side"]=="left" else p["dst"][selected]
            flat_r = p["src"][selected] if p["map_side"]=="right" else p["dst"][selected]
            left.ravel()[flat_l] = True
            right.ravel()[flat_r] = True
            out.append((r["label"]+" · left",left))
            out.append((r["label"]+" · right",right))
    return out


def make_sos(name, fps, order):
    """Construct the second-order-section filter selected in the GUI.
    
        Parameters
        ----------
        name : str
            One of ``FILTERS_FULL``. ``No filter`` returns None.
        fps : float
            Temporal sampling frequency in Hz.
        order : int
            Butterworth filter order.
    
        Returns
        -------
        numpy.ndarray or None
            SOS filter coefficients, or None if filtering is disabled.
    
        Notes
        -----
        The low-pass cutoff is 0.4 Hz; band-pass options are 0.08--0.4 Hz
        and 0.009--0.08 Hz. These are interface selections, not equivalent
        to the possibly different spectral analysis settings."""
    if name == "No filter":
        return None
    if name == "Low-pass ≤0.4 Hz":
        return butter(order,0.4,btype="lowpass",fs=fps,output="sos")
    bands = {"Band-pass 0.08–0.4 Hz":(0.08,0.4),"Slow band 0.009–0.08 Hz":(0.009,0.08)}
    if name not in bands:
        raise ValueError(name)
    low,high = bands[name]
    if not (0 < low < high < fps/2):
        raise ValueError(f"Invalid filter {name} for fps={fps}")
    return butter(order,[low,high],btype="bandpass",fs=fps,output="sos")


def filtered_columns(data, fps, name, order=4, pad_sec=20):
    """Linearly detrend and optionally zero-phase filter pixel time series.
    
        Parameters
        ----------
        data : array_like, shape (time, pixels)
            One or more temporal pixel traces.
        fps : float
            Frames per second.
        name : str
            GUI filter label from ``FILTERS_FULL``.
        order : int, optional
            Butterworth filter order (default 4).
        pad_sec : float, optional
            Maximum synthetic odd-reflection padding duration in seconds.
    
        Returns
        -------
        numpy.ndarray, shape (time, pixels)
            Float32 detrended/filtered time series. ``No filter`` still removes
            a fitted linear trend. The original input is not modified."""
    x = detrend(np.asarray(data,dtype=np.float32),axis=0,type="linear").astype(np.float32)
    sos = make_sos(name,fps,order)
    if sos is None:
        return x
    requested = max(0,int(round(pad_sec*fps)))
    actual = min(requested,max(0,len(x)-2))
    if actual:
        return sosfiltfilt(sos,x,axis=0,padtype="odd",padlen=actual).astype(np.float32)
    return sosfiltfilt(sos,x,axis=0,padtype=None,padlen=0).astype(np.float32)


def resize_area(movie, static, shape, fraction):
    """Exactly Step7 mask-aware area resampling (chunked for memory)."""
    arr = np.asarray(movie,dtype=np.float32)
    h,w = shape
    out = np.full((len(arr),h,w),np.nan,dtype=np.float32)
    static_fraction = cv2.resize(static.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
    for i,frame in enumerate(arr):
        finite = static & np.isfinite(frame)
        numer = cv2.resize(np.where(finite,frame,0).astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
        denom = cv2.resize(finite.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
        good = (denom>=fraction)&(static_fraction>0)
        out[i,good] = numer[good] / np.maximum(denom[good],1e-12)
    return out


class _LazyMovieNoDask:
    """Minimal read-only fallback for environments without Dask (mainly tests)."""
    def __init__(self, disk):
        """Expose the frame count, spatial shape and dtype of a disk cache."""
        self.disk = disk
        self.shape = (disk.nframes, *disk.shape)
        self.ndim = 3
        self.dtype = np.dtype("float32")

    def __getitem__(self, index):
        # Napari normally uses Dask; this fallback reads only indexed frames.
        """Read requested frames lazily from storage, then apply spatial indexing."""
        if isinstance(index, tuple):
            t = index[0]
            spatial = index[1:]
        else:
            t, spatial = index, ()
        if isinstance(t, (int, np.integer)):
            result = self.disk.frame(int(t))
        else:
            indices = np.arange(self.shape[0])[t]
            result = np.stack([self.disk.frame(int(i)) for i in np.atleast_1d(indices)])
        return result[(..., *spatial)] if spatial else result

    def __array__(self, dtype=None, copy=None):
        """Materialize the complete movie, optionally casting or copying its data."""
        data = self[:]
        if dtype is not None:
            data = data.astype(dtype, copy=False)
        return data.copy() if copy else data


class _DiskMovieStorage:
    """Chunked on-disk matrices, without Windows giant-file memory mapping.

    HDF5 (h5py) is used when available, otherwise Zarr, already required by
    xarray.open_zarr. Both keep [time, pixel] storage chunked and allow Dask
    to serve Napari frames without mapping two entire .npy files on Windows.
    """

    def __init__(self, folder: Path, nframes: int, shape: tuple[int, int], *, backend: str = "auto"):
        """Create two chunked scratch datasets of shape (time, flattened pixels).
        
                Parameters
                ----------
                folder : Path
                    Writable temporary directory.
                nframes : int
                    Temporal length.
                shape : tuple of int
                    FC grid dimensions in (y, x) order.
                backend : {"auto", "hdf5", "zarr"}, optional
                    Scratch format. Auto prefers h5py when installed.
        
                Notes
                -----
                Scratch stores are not analytical outputs and are removed by the GUI
                when its temporary directory is cleaned up."""
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.nframes = int(nframes)
        self.shape = tuple(map(int, shape))
        self.npixels = int(np.prod(self.shape))
        self.backend = str(backend or "auto").lower()
        self.h5 = None
        self.raw = None
        self.filtered = None
        self.raw_filename = None
        self.filtered_filename = None
        self._lazy_filtered = None
        if self.backend not in {"auto", "hdf5", "zarr"}:
            raise ValueError(f"Unsupported disk cache backend: {self.backend}")
        if self.backend == "auto":
            try:
                import h5py  # noqa: F401
                self.backend = "hdf5"
            except ImportError:
                self.backend = "zarr"
        # Small pixel chunks make temporal [all frames, ~128 pixels] reads
        # efficient for detrend + zero-phase filtering.
        chunks = (min(self.nframes, 256), min(self.npixels, 128))
        try:
            if self.backend == "hdf5":
                import h5py
                self.raw_filename = self.filtered_filename = self.folder / "movie_cache.h5"
                self.h5 = h5py.File(str(self.raw_filename), "w")
                for name in ("resized", "filtered"):
                    ds = self.h5.create_dataset(
                        name, shape=(self.nframes, self.npixels), dtype="f4",
                        chunks=chunks, compression="lzf", shuffle=True,
                        fillvalue=np.float32(np.nan),
                    )
                    if name == "resized":
                        self.raw = ds
                    else:
                        self.filtered = ds
            else:
                import zarr
                self.raw_filename = self.folder / "resized.zarr"
                self.filtered_filename = self.folder / "filtered.zarr"
                kw = dict(shape=(self.nframes, self.npixels), chunks=chunks,
                          dtype="f4", fill_value=float("nan"), mode="w")
                self.raw = zarr.open_array(str(self.raw_filename), **kw)
                self.filtered = zarr.open_array(str(self.filtered_filename), **kw)
        except Exception:
            self.close()
            raise

    def write_frames(self, first: int, last: int, frames: np.ndarray) -> None:
        """Write a time block of resized frames into the unfiltered scratch store."""
        data = np.asarray(frames, dtype=np.float32).reshape(last-first, self.npixels)
        self.raw[first:last, :] = data

    def read_columns(self, what: str, columns: np.ndarray) -> np.ndarray:
        """Read all time points for selected flattened pixel indices.
        
                The ``what`` selector chooses raw versus filtered cache data."""
        target = self.raw if what == "raw" else self.filtered
        indices = np.asarray(columns, dtype=np.int64)
        if not indices.size:
            return np.empty((self.nframes, 0), dtype=np.float32)
        if self.backend == "zarr":
            return np.asarray(target.oindex[:, indices], dtype=np.float32)
        return np.asarray(target[:, indices], dtype=np.float32)

    def write_filtered_columns(self, columns: np.ndarray, data: np.ndarray) -> None:
        """Write selected pixel time courses into the filtered scratch dataset."""
        indices = np.asarray(columns, dtype=np.int64)
        if not indices.size:
            return
        if self.backend == "zarr":
            self.filtered.oindex[:, indices] = np.asarray(data, dtype=np.float32)
        else:
            self.filtered[:, indices] = np.asarray(data, dtype=np.float32)

    def frame(self, index: int) -> np.ndarray:
        """Read one filtered frame and reshape it to the spatial FC grid."""
        return np.asarray(self.filtered[int(index), :], dtype=np.float32).reshape(self.shape)

    def lazy_movie(self):
        # Both HDF5 and Zarr present lazy frames to napari, so scrolling need
        # not read/load the complete movie into RAM or virtual address space.
        """Provide a lazy time-by-Y-by-X view of filtered scratch frames.
        
                Uses Dask when present; otherwise falls back to indexed disk reads."""
        if self._lazy_filtered is None:
            try:
                import dask.array as da
            except ImportError:
                self._lazy_filtered = _LazyMovieNoDask(self)
            else:
                # One complete spatial frame per Dask chunk (24 consecutive
                # frames), so reshaping flat pixels to Y/X cannot explode the
                # task graph or require huge rechunks on large FC grids.
                chunks = (min(self.nframes, 24), self.npixels)
                self._lazy_filtered = da.from_array(self.filtered, chunks=chunks,
                                                     asarray=True, lock=True).reshape(
                                                         (self.nframes, *self.shape))
        return self._lazy_filtered

    def flush(self) -> None:
        """Flush the HDF5 scratch file if HDF5 storage is active."""
        if self.h5 is not None:
            self.h5.flush()

    def close(self) -> None:
        """Release the lazy view and close all scratch dataset handles."""
        self._lazy_filtered = None
        if self.h5 is not None:
            try:
                self.h5.close()
            finally:
                self.h5 = None
        self.raw = None
        self.filtered = None


class MovieCache:
    """Disk-backed signal data; no giant Windows np.memmap mappings."""

    def __init__(self, study, geometry, dataset, signal_var, start, end, filt, full,
                 folder, progress: Callable[[str], None] | None = None,
                 backend: str = "auto"):
        """Build disk-backed FC-grid visual signals for a selected time interval.
        
                Parameters
                ----------
                study : Study
                    Project context and temporal/spatial processing configuration.
                geometry : Geometry
                    Stem-specific final usable masks and FC grid dimensions.
                dataset : xarray.Dataset
                    Read-only processed Step4 signals.
                signal_var : str
                    Hemodynamic variable from ``SIGNALS``.
                start, end : int
                    Half-open interval of frame indices in the source recording.
                filt : str
                    Filter selection. Slow-band is valid only for ``full=True``.
                full : bool
                    Whether processing the complete recording rather than a window.
                folder : str or Path
                    Writable scratch directory.
                progress : callable, optional
                    Receives progress messages for the GUI.
                backend : str, optional
                    Chunked scratch backend (``auto``, ``hdf5`` or ``zarr``).
        
                Notes
                -----
                Reads the source Zarr without modifying it, projects frames to the
                FC grid using valid-area interpolation, checks finite variance,
                detrends and filters per pixel, and stores temporary results on disk.
                The valid-pixel mask excludes pixels failing these checks."""
        self.signal_var = signal_var
        self.filt = filt
        self.start = int(start)
        self.end = int(end)
        self.fps = study.fps
        self.shape = geometry.fc_shape
        self.storage = None
        self.closed = False
        if filt not in (FILTERS_FULL if full else FILTERS_REST):
            raise ValueError("Slow-band 0.009–0.08 Hz is FULL RECORD ONLY, not valid for 60-s REST/Iso windows")
        var = dataset[signal_var].transpose("time", "y", "x")
        self.nframes = self.end-self.start
        if self.nframes < 4:
            raise ValueError("Interval too short")
        self.time = np.arange(self.start, self.end, dtype=np.float64)/self.fps
        self.temp = Path(folder)
        self.temp.mkdir(parents=True, exist_ok=True)
        self.valid = np.zeros(self.shape, dtype=bool)
        try:
            self.storage = _DiskMovieStorage(self.temp, self.nframes, self.shape,
                                             backend=backend)
            self.path_filter = self.storage.filtered_filename
            frac = float(study.cfg.get("functional_connectivity", {}).get("spatial", {}).get("min_valid_fraction", 0.5))
            raw_block = 96
            for a in range(0, self.nframes, raw_block):
                b = min(self.nframes, a + raw_block)
                arr = var.isel(time=slice(self.start+a,self.start+b)).load().values.astype(np.float32)
                self.storage.write_frames(a, b, resize_area(arr, geometry.mask_processed, self.shape, frac))
                if progress and (a//raw_block)%6 == 0:
                    progress(f"{signal_var}: resizing frames {b}/{self.nframes}")
            eps = float(study.cfg.get("functional_connectivity", {}).get("temporal", {}).get("variance_epsilon", 1e-12))
            pixel_indices = np.flatnonzero(geometry.mask_fc.ravel())
            tc = study.cfg.get("functional_connectivity", {}).get("temporal", {})
            order = int(tc.get("order", 4))
            pad_sec = float(tc.get("synthetic_odd_pad_sec", 20.0))
            if filt == "Slow band 0.009–0.08 Hz":
                pad_sec = max(pad_sec, 120.0)
            for i in range(0, len(pixel_indices), 128):
                inds = pixel_indices[i:i+128]
                block = self.storage.read_columns("raw", inds)
                good = (np.all(np.isfinite(block), axis=0) &
                        (np.std(np.where(np.isfinite(block), block, 0), axis=0, ddof=0)>eps))
                if good.any():
                    data = filtered_columns(block[:,good], self.fps, filt, order, pad_sec)
                    g2 = np.all(np.isfinite(data), axis=0) & (np.std(data, axis=0, ddof=1)>eps)
                    final_inds = inds[good][g2]
                    self.storage.write_filtered_columns(final_inds, data[:,g2])
                    self.valid.ravel()[final_inds] = True
                if progress and (i//128)%8 == 0:
                    progress(f"{signal_var}: detrend/filter {min(i+128,len(pixel_indices))}/{len(pixel_indices)} pixels")
            if not self.valid.any():
                raise RuntimeError(f"No nonconstant finite FC pixels in {signal_var} interval")
            self.storage.flush()
            self.filtered = self.storage.lazy_movie()
        except Exception:
            self.close()
            raise

    def trace(self, mask, detrended=False):
        """Average selected valid FC pixels into one temporal trace.
        
                Parameters
                ----------
                mask : numpy.ndarray of bool
                    ROI mask on the FC grid, in (y, x) order.
                detrended : bool, optional
                    If true, use the unfiltered resized data and remove a linear
                    trend from the average trace; otherwise use filtered data.
        
                Returns
                -------
                tuple
                    (time series of float32 values, number of valid ROI pixels)."""
        valid = np.asarray(mask, dtype=bool) & self.valid
        if not valid.any():
            return np.full(self.nframes, np.nan, dtype=np.float32), 0
        idx = np.flatnonzero(valid.ravel())
        total = np.zeros(self.nframes, dtype=np.float64)
        for a in range(0, len(idx), 128):
            total += self.storage.read_columns(
                "raw" if detrended else "filtered", idx[a:a+128]
            ).sum(axis=1, dtype=np.float64)
        avg = (total / len(idx)).astype(np.float32)
        if detrended:
            avg = detrend(avg, type="linear").astype(np.float32)
        return avg, int(len(idx))

    def frame(self, k):
        """Read a single filtered FC-grid frame from the scratch cache."""
        k = max(0,min(int(k), self.nframes-1))
        return self.storage.frame(k)

    def close(self):
        """Release scratch storage and mark this movie cache as closed."""
        if self.closed:
            return
        self.closed = True
        if self.storage is not None:
            self.storage.close()


class MovieWindow:
    """Zero-copy temporal view of a MovieCache filtered in its source context.

    In Iso slow-band mode the cache covers the WHOLE recording, then this view
    selects an exact 60-second half-open interval without filtering it again.
    """
    def __init__(self, cache: MovieCache, local_start: int, local_stop: int):
        """Create a half-open temporal view into an existing MovieCache.
        
                ``local_start`` and ``local_stop`` index cache-relative frames.
                The source cache remains responsible for scratch storage lifetime."""
        self.cache = cache
        self.local_start = int(local_start)
        self.local_stop = int(local_stop)
        if not (0 <= self.local_start < self.local_stop <= cache.nframes):
            raise ValueError("Invalid MovieWindow interval")
        self.nframes = self.local_stop - self.local_start
        self.time = cache.time[self.local_start:self.local_stop]
        self.signal_var = cache.signal_var
        self.filt = cache.filt
        self.shape = cache.shape
        self.path_filter = cache.path_filter

    @property
    def stack(self):
        """Return the lazy filtered time-by-Y-by-X movie for this window."""
        return self.cache.filtered[self.local_start:self.local_stop]

    def frame(self, k):
        """Read a time-window-relative frame from the backing full cache."""
        k = int(np.clip(k, 0, self.nframes - 1))
        return self.cache.frame(self.local_start + k)

    def trace(self, mask, *, detrended=False):
        """Return the selected ROI trace cropped to this temporal window."""
        x, pixels = self.cache.trace(mask, detrended=detrended)
        return x[self.local_start:self.local_stop], pixels
