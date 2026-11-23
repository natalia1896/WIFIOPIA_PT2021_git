"""Re-register ONLY specified existing Step2 stems after manual bregma review.

This does not rerun mask creation, group-level registration, or reference selection.
Uses the original step2_masks_registration calculation functions and exactly the
historical ref_stem for each selected record. Only the selected stem's ROI,
registered BMP, bregma_manual mask, and metadata CSV row can be replaced.

Dry run (default):
  python step2_reregister_stems.py --config config.yaml --stems STEM1 STEM2
Commit:
  python step2_reregister_stems.py --config config.yaml --stems STEM1 STEM2 --commit
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.transform import EuclideanTransform

import step2_masks_registration as core
from common import canonical_mask_paths, mask_for_reg_per_stem_path
from step2_add_exp_day import _paths, _resolve_old_image, _existing_ref, _as_point
from common import load_config


def _sha256(path: Path) -> str:
    """Calculate the SHA-256 fingerprint of an on-disk file.

    Parameters
    ----------
    path : Path
        Input file or destination path.

    Returns
    -------
    str
        Result of the operation described above."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _csv_patch_exact_rows(original: bytes, patches: dict[str, dict[str, Any]]) -> bytes:
    """Patch selected Step2 metadata CSV records without reserializing others.

    Parameters
    ----------
    original : bytes
        Historical ``metadata_registration.csv`` contents.
    patches : dict of str to dict
        Replacement fields for the specified stem records.

    Returns
    -------
    bytes
        Updated CSV with byte-identical non-target physical rows.

    Notes
    -----
    Unlike incremental appends, re-registration intentionally replaces only
    the selected existing stem records. The original UTF-8 BOM and line-ending
    conventions are retained."""
    if not original:
        raise ValueError('Empty metadata CSV')
    bom = original.startswith(b'\xef\xbb\xbf')
    text = original.decode('utf-8-sig')
    physical_lines = text.splitlines(keepends=True)
    reader = csv.reader(io.StringIO(text, newline=''))
    eol = '\r\n' if '\r\n' in text[:10000] else '\n'
    out: list[str] = []
    prev = 0
    header: list[str] | None = None
    seen: set[str] = set()
    for record_no, values in enumerate(reader):
        end = reader.line_num
        raw = ''.join(physical_lines[prev:end])
        prev = end
        if record_no == 0:
            header = values
            if len(set(header)) != len(header) or 'stem' not in header:
                raise ValueError('Invalid metadata header')
            out.append(raw)
            continue
        assert header is not None
        if len(values) != len(header):
            raise ValueError(f'Bad metadata CSV record #{record_no}')
        row = dict(zip(header, values))
        stem = row['stem']
        if stem not in patches:
            out.append(raw)
            continue
        if stem in seen:
            raise ValueError(f'Duplicate stem in metadata: {stem}')
        seen.add(stem)
        unknown = set(patches[stem]) - set(header)
        if unknown:
            raise ValueError(f'Calculated fields missing in existing metadata: {sorted(unknown)}')
        row.update({k: str(v) for k, v in patches[stem].items()})
        sio = io.StringIO(newline='')
        writer = csv.writer(sio, lineterminator=eol)
        writer.writerow([row[k] for k in header])
        out.append(sio.getvalue())
    if seen != set(patches):
        raise ValueError(f'Selected stems missing from metadata: {sorted(set(patches)-seen)}')
    return (b'\xef\xbb\xbf' if bom else b'') + ''.join(out).encode('utf-8')


def _load_masks(paths: dict, stem: str, *, ref: bool = False) -> dict:
    """Load the already existing automatic brain, bad-pixel, and ring masks for a recording.

    Parameters
    ----------
    paths : dict
        Resolved project output paths.
    stem : str
        Recording stem identifier.
    ref : bool
        Whether the masks belong to the reference recording.

    Returns
    -------
    dict
        Result of the operation described above."""
    cp = canonical_mask_paths(paths, stem)
    needed = ('brain_auto', 'bad_auto', 'ring_auto')
    for key in needed:
        if not cp[key].is_file():
            raise FileNotFoundError(f'Missing existing Step2 mask for {stem}: {cp[key]}')
    return {
        'brain': np.load(cp['brain_auto']).astype(bool),
        'bad': np.load(cp['bad_auto']).astype(bool),
        'ring': np.load(cp['ring_auto']).astype(bool),
    }


def _target_plan(config_path: Path, stems: list[str]):
    """Validate the target stems and find their stored reference records and images.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration for an existing project.
    stems : list[str]
        Unique recording stem identifiers.

    Returns
    -------
    object
        Computed value or prepared data for the described operation."""
    config = load_config(config_path)
    paths = _paths(config)
    meta = paths['metadata_registration_csv']
    df = pd.read_csv(meta, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    if df['stem'].duplicated().any():
        raise ValueError('Duplicate stems in metadata_registration.csv')
    if not stems or len(set(stems)) != len(stems):
        raise ValueError('Supply unique --stems')
    targets = []
    wave = core.get_registration_wavelength(config)
    for stem in stems:
        rows = df[df['stem'].eq(stem)]
        if len(rows) != 1:
            raise ValueError(f'Exactly one existing metadata row required for {stem}; found {len(rows)}')
        row = rows.iloc[0]
        if str(row.get('exp_day','')) != 'PT':
            raise ValueError(f'Only exp_day=PT was requested; refusing {stem} exp_day={row.get("exp_day")}')
        if str(row.get('bregma_source', '')).lower() != 'manual':
            raise ValueError(f'Manual bregma not saved for {stem}; bregma_source={row.get("bregma_source")}')
        if _as_point(row) is None:
            raise ValueError(f'Manual bregma coordinates missing for {stem}')
        gid = str(row['group_id'])
        group = df[df['group_id'].eq(gid)]
        ref_stem, ref_row, _ = _existing_ref(group, gid)
        if ref_stem != str(row['ref_stem']):
            raise ValueError(f'Historical ref mismatch for {stem}: {ref_stem} != {row["ref_stem"]}')
        if ref_stem == stem:
            raise ValueError(f'Refusing to recompute a group reference itself: {stem}')
        if _as_point(ref_row) is None:
            raise ValueError(f'Reference bregma missing: {ref_stem}')
        fixed_path = _resolve_old_image(ref_row['fixed_registration_path'], config)
        moving_path = _resolve_old_image(row['moving_registration_path'], config)
        cp = canonical_mask_paths(paths, stem)
        for key in ('brain_auto', 'bad_auto', 'ring_auto'):
            if not cp[key].is_file():
                raise FileNotFoundError(cp[key])
        for key in ('brain_auto', 'bad_auto', 'ring_auto'):
            ref_cp = canonical_mask_paths(paths, ref_stem)
            if not ref_cp[key].is_file():
                raise FileNotFoundError(ref_cp[key])
        targets.append((row, ref_row, fixed_path, moving_path))
    return config, paths, meta, targets


def _compute(row: pd.Series, ref_row: pd.Series, fixed_path: Path, moving_path: Path,
             config: dict, paths: dict, staging: Path) -> tuple[dict[str, Any], dict[Path, Path]]:
    """Recompute the selected stem registration and prepare staged output files.

    Parameters
    ----------
    row : pd.Series
        Metadata row for the moving recording.
    ref_row : pd.Series
        Metadata row for the fixed reference.
    fixed_path : Path
        Reference BMP used as the fixed image.
    moving_path : Path
        BMP image to be registered.
    config : dict
        Parsed project configuration mapping.
    paths : dict
        Resolved project output paths.
    staging : Path
        Temporary directory for generated files before commit.

    Returns
    -------
    tuple[dict[str, Any], dict[Path, Path]]
        Result of the operation described above."""
    gid = str(row['group_id']); stem = str(row['stem'])
    reg = config['step2']['registration']; qc = config['step2'].get('qc', {}) or {}
    method = str(reg.get('method', 'ring_sobel')).strip().lower()
    roi_mode = str(reg.get('roi_mode', 'full_image' if method == 'vessel_sobel' else 'ring')).strip().lower()
    mask_mode = str(reg.get('mask_mode', 'pairwise'))
    anchor = bool(reg.get('anchor_bregma', True))
    strict = bool(reg.get('strict_bregma_anchor', True))
    tol = float(reg.get('bregma_anchor_tol_px', 0.001))
    wave = core.get_registration_wavelength(config)
    ref_stem = str(ref_row['stem'])
    ref = {**_load_masks(paths, ref_stem), 'stem': ref_stem}
    mov = {**_load_masks(paths, stem), 'stem': stem}
    cy0, cx0 = _as_point(ref_row)
    cy, cx = _as_point(row)
    _, fixed = core.load_bmp_u8_and_f32(fixed_path)
    _, moving = core.load_bmp_u8_and_f32(moving_path)
    for name, arr in [('moving', moving), *[(f'ref.{k}', v) for k, v in ref.items() if k!='stem'],
                      *[(f'mov.{k}', v) for k, v in mov.items() if k!='stem']]:
        if arr.shape != fixed.shape:
            raise ValueError(f'Shape mismatch {name} for {stem}: {arr.shape} != {fixed.shape}')
    ref['bregma']=(cy0,cx0)
    mov['bregma']=(cy,cx)
    dy, dx = cy0-cy, cx0-cx
    initial = EuclideanTransform(translation=(dx,dy))
    moving0 = core.apply_tform(moving, initial, order=1)
    union_ring = union_bad = None
    # This targeted mode is for the current full-image registration config.
    # For ring/group-union ROI, the original group-wide union would have to change
    # after a new bregma point, which contradicts the old-data preservation goal.
    if roi_mode not in ('full', 'full_image', 'whole_image', 'image'):
        raise ValueError('Selected-stem re-registration supports only the existing full_image ROI configuration; do not change it')
    roi = core.build_registration_roi(
        roi_mode=roi_mode, mask_mode=mask_mode, ref_rec=ref, mov_rec=mov,
        init_dy=dy, init_dx=dx, image_shape=fixed.shape,
        group_union_ring=union_ring, group_union_bad=union_bad,
        ignore_border_px=int(reg.get('ignore_border_px',0) or 0))
    area=int(roi.sum())
    angle=0.0; score=float('-inf'); total=initial; dy2=dx2=0.0
    if area > 0:
        f_feature=core.build_registration_feature(fixed,roi,reg)
        m_feature=core.build_registration_feature(moving0,roi,reg)
        t2,angle,(dy2,dx2),score=core.estimate_feature_scored(
            fixed_feature=f_feature, moving0_feature=m_feature,
            mask=roi,rot_center_yx=(cy0,cx0),
            angle_range=float(reg.get('angle_range_deg',3.0)),
            angle_step=float(reg.get('angle_step_deg',0.1)),
            upsample=int(reg.get('upsample',10)), anchor_bregma=anchor)
        total=initial+t2
    bregma_error=(core.bregma_anchor_error_px(total,(cy,cx),(cy0,cx0)) if anchor else np.nan)
    if anchor and strict and np.isfinite(bregma_error) and bregma_error > tol:
        raise RuntimeError(f'Hard bregma anchor failed for {stem}: {bregma_error} > {tol}')
    tx,ty=map(float,total.translation)
    reasons=[]
    if area==0: reasons.append('empty_registration_roi')
    elif not np.isfinite(score): reasons.append('invalid_registration_score')
    if bool(qc.get('apply_thresholds',False)):
        if not np.isfinite(score) or score < float(qc.get('min_score',0.1)): reasons.append('low_score')
        if area<int(qc.get('min_mask_area',500)): reasons.append('small_mask')
        if abs(tx)>float(qc.get('max_abs_dx',100)): reasons.append('dx_large')
        if abs(ty)>float(qc.get('max_abs_dy',100)): reasons.append('dy_large')
        if abs(float(angle))>float(qc.get('max_abs_angle_deg',10)): reasons.append('angle_large')
    if anchor and np.isfinite(bregma_error) and bregma_error > tol:
        reasons.append('bregma_anchor_error')
    updates = {
        'bregma_ref_found': True, 'bregma_ref_cy': cy0, 'bregma_ref_cx':cx0,
        'bregma_ref_source': ref_row.get('bregma_source','auto'),
        'bregma_found': True, 'bregma_source': 'manual',
        'init_shift_dy': dy, 'init_shift_dx': dx,
        'second_stage_free_shift_dy': dy2, 'second_stage_free_shift_dx': dx2,
        'angle_deg_found': angle, 'score':score,
        'translation_x': tx,'translation_y':ty,'rotation_rad':float(total.rotation),
        'bregma_registered_y':core.transform_point_yx(total,cy,cx)[0],
        'bregma_registered_x':core.transform_point_yx(total,cy,cx)[1],
        'bregma_error_px':bregma_error,
        'anchor_bregma':anchor,'strict_bregma_anchor':strict,'bregma_anchor_tol_px':tol,
        'registration_method':method,'registration_roi_mode':roi_mode,
        'registration_feature_method':method,
        'vessel_sobel_params_json':str(reg.get('vessel_sobel',{}) or {}),
        'mask_for_reg_area':area,'mask_mode':mask_mode,
        'mask_file_logic':'canonical_by_stem',
        'is_bad':bool(reasons),'bad_reasons':','.join(reasons),
        'registration_wavelength':wave,
        'fixed_registration_path': str(ref_row['fixed_registration_path']),
        'moving_registration_path': str(row['moving_registration_path']),
        'saved_registered_path':str(row['saved_registered_path']),
    }
    derived=core.add_raw_geometry_metadata(dict(updates),raw_scale=core.raw_scale_factor(config),
                                           rotation_deg=core.raw_rotation_degrees(config))
    updates.update({k:v for k,v in derived.items() if k.endswith('_raw')})
    header = pd.read_csv(paths['metadata_registration_csv'],nrows=0).columns.tolist()
    updates={k:v for k,v in updates.items() if k in header}
    artifacts={}
    staged_roi=staging/f'{stem}.mask_for_reg.npy'
    core.save_mask(staged_roi,roi)
    artifacts[staged_roi]=mask_for_reg_per_stem_path(paths,stem)
    if config['step2'].get('io',{}).get('save_registered_bmps',True):
        moved=core.apply_tform(moving,total,order=1)
        staged_bmp=staging/f'{stem}.registered.bmp'
        core.save_bmp_uint8(staged_bmp,moved)
        dst=paths['step2_root']/'registered_bmps'/f'{stem}_{wave}_registered_to_{ref_stem}.bmp'
        original_target=_resolve_old_image(row['saved_registered_path'],config)
        if dst.resolve()!=original_target.resolve():
            raise ValueError(f'Registered BMP name mismatch for {stem}: {dst} vs {original_target}')
        artifacts[staged_bmp]=dst
    diam=float(((config.get('step3',{}) or {}).get('manual_mask_updates',{}) or {}).get('bregma_diameter_px',25.0))
    manual_bregma=core.disk_mask(mov['brain'].shape,cy,cx,diameter_px=diam)
    staged_bregma=staging/f'{stem}.bregma_manual.npy'
    core.save_mask(staged_bregma,manual_bregma)
    artifacts[staged_bregma]=canonical_mask_paths(paths,stem)['bregma_manual']
    return updates, artifacts


def rerun(config_path: Path, stems: list[str], commit: bool = False) -> dict:
    """Run dry-run validation or explicitly commit registration changes for selected stems.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration for an existing project.
    stems : list[str]
        Unique recording stem identifiers.
    commit : bool
        Whether to write changes after validation (default: dry-run).

    Returns
    -------
    dict
        Result of the operation described above.

    Notes
    -----
    Without ``commit``, the function reports the requested stems and reference
    images without changing the registration. The targeted operation is limited
    to records with ``exp_day == 'PT'`` by ``_target_plan``."""
    cfg, paths, meta, targets = _target_plan(config_path, stems)
    prior_sha=_sha256(meta)
    data=[]
    for row,ref,fp,mp in targets:
        data.append({'stem':row['stem'],'bregma':_as_point(row),
                     'ref_stem':ref['stem'],'ref_bregma':_as_point(ref),
                     'moving_bmp':str(mp),'reference_bmp':str(fp)})
    if not commit:
        return {'status':'dry_run','n_targets':len(data),'targets':data,
                'note':'No file has been changed'}
    log_root=paths['step2_root']/'incremental_step2_logs'
    log_root.mkdir(parents=True,exist_ok=True)
    folder=log_root/('reregister_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    folder.mkdir(parents=False)
    staging=folder/'staging'; staging.mkdir()
    backup=folder/'backup'; backup.mkdir()
    patches={}; artifacts={}
    try:
        for row,ref,fp,mp in targets:
            p, a = _compute(row,ref,fp,mp,cfg,paths,staging)
            patches[str(row['stem'])]=p
            artifacts.update(a)
        if _sha256(meta)!=prior_sha:
            raise RuntimeError('metadata_registration.csv changed during calculations; aborting')
        new_meta=_csv_patch_exact_rows(meta.read_bytes(), patches)
        (backup/'metadata_registration.csv').write_bytes(meta.read_bytes())
        originals={}
        for src,dst in artifacts.items():
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists():
                saved=backup/(f'{len(originals):03d}_'+dst.name)
                shutil.copy2(dst,saved)
                originals[dst]=saved
            else:
                originals[dst]=None
        moved=[]
        try:
            for src,dst in artifacts.items():
                os.replace(src,dst)
                moved.append(dst)
            temp_meta=folder/'metadata_new.tmp'
            temp_meta.write_bytes(new_meta)
            if _sha256(meta)!=prior_sha:
                raise RuntimeError('metadata CSV changed before final replacement')
            os.replace(temp_meta,meta)
        except Exception:
            for dst in reversed(moved):
                backed=originals[dst]
                if backed is None:
                    dst.unlink(missing_ok=True)
                else:
                    shutil.copy2(backed,dst)
            raise
        result={'status':'updated','n_updated':len(patches),'stems':list(patches),
                'updated_files':list(map(str,artifacts.values())),
                'metadata_path':str(meta),'backup_dir':str(backup),
                'old_metadata_sha256':prior_sha}
        (folder/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
        return result
    finally:
        shutil.rmtree(staging,ignore_errors=True)


def main():
    """Parse the targeted re-registration CLI and print its JSON result."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--stems',required=True,nargs='+')
    p.add_argument('--commit',action='store_true',help='Actually update the selected stems after dry-run')
    args=p.parse_args()
    print(json.dumps(rerun(args.config,args.stems,args.commit),ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':
    main()
