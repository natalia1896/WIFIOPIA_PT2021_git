"""Add only new exp_day stems to an EXISTING Step2 project.

Uses unmodified mask and registration functions from step2_masks_registration.
The actual reference per mouse is taken from existing metadata_registration.csv,
including a historical fallback reference (not necessarily Day8).

No old mask, registered BMP, union mask, metadata row, or Step3 annotation is
recomputed.  New files are staged; old metadata CSV bytes are retained verbatim
and new records are appended after all calculations succeed.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.transform import EuclideanTransform

from common import (
    canonical_mask_paths, get_path_resolving_config,
    mask_for_reg_per_stem_path, mask_for_reg_union_path, raw_rotation_degrees,
    raw_scale_factor, resolve_flexible_path, drop_mask_path_metadata_columns,
    load_config,
)
import step2_masks_registration as core


def _sha256(path: Path) -> str:
    """Compute a SHA-256 digest for a file.

    Parameters
    ----------
    path : Path
        Input file or destination path.

    Returns
    -------
    str
        Result of the operation described above."""
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _paths(config: dict) -> dict[str, Path]:
    """Resolve an existing Step2 analysis root and its output locations.

    Parameters
    ----------
    config : dict
        Project YAML configuration containing an existing
        ``project.directory_to_save_main_analysis``.

    Returns
    -------
    dict of str to pathlib.Path
        Paths to Step1/Step2/Step3/Step4 folders, the manifest, registration
        metadata CSV, and mask root.

    Raises
    ------
    FileNotFoundError
        If the historical project outputs are missing.

    Notes
    -----
    This helper never creates a new analysis project."""
    pcfg = config.get('project', {}) or {}
    original = pcfg.get('directory_to_save_main_analysis')
    if not original:
        raise ValueError('Existing project.directory_to_save_main_analysis required')
    remap = get_path_resolving_config(config)
    root = resolve_flexible_path(original, from_drive=remap.get('from_drive'),
                                 to_drive=remap.get('to_drive'), from_root=remap.get('from_root'),
                                 to_root=remap.get('to_root'), must_exist=True)
    root = Path(root)
    out = root / str(pcfg.get('out_folder_name', 'out'))
    if not out.is_dir():
        raise FileNotFoundError(f'Existing project out folder missing: {out}')
    step1 = config.get('step1', {}) or {}
    step2 = config.get('step2', {}) or {}
    step3 = config.get('step3', {}) or {}
    step4 = config.get('step4', {}) or {}
    s2 = out / step2.get('folder_name', 'step2_masks')
    result = {
        'out_root': out,
        'step1_root': out / step1.get('folder_name', 'step1_project'),
        'step2_root': s2,
        'step3_root': out / step3.get('folder_name', 'step3'),
        'step4_root': out / step4.get('folder_name', 'step4_raw_and_signals'),
        'manifest_path': out / step1.get('manifest_name', 'manifest_recordings.csv'),
        'metadata_registration_csv': s2 / step2.get('metadata_registration_name', 'metadata_registration.csv'),
        'masks_root': s2 / step2.get('masks_folder', 'masks'),
    }
    for key in ('manifest_path', 'metadata_registration_csv'):
        if not result[key].is_file():
            raise FileNotFoundError(f'Existing {key} required: {result[key]}')
    return result


def _csv_header(path: Path) -> list[str]:
    """Read a comma-separated CSV header and validate its column names.

    Parameters
    ----------
    path : Path
        Input file or destination path.

    Returns
    -------
    list[str]
        Result of the operation described above."""
    with path.open(encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader(f), [])
    if not header or 'stem' not in header or len(set(header)) != len(header):
        raise ValueError(f'Invalid comma-separated CSV header: {path}. Do not save it with ; delimiters.')
    return header


def _existing_ref(group: pd.DataFrame, group_id: str) -> tuple[str, pd.Series, pd.Series]:
    """Recover the stored reference recording for a group from historical registration metadata.

    Parameters
    ----------
    group : pd.DataFrame
        Historical registration-group records.
    group_id : str
        Registration-group identifier.

    Returns
    -------
    tuple[str, pd.Series, pd.Series]
        Result of the operation described above."""
    refs = {str(x) for x in group['ref_stem'] if str(x).strip()}
    if len(refs) != 1:
        raise ValueError(f'Existing group={group_id} has inconsistent historical ref_stem values: {sorted(refs)}')
    ref_stem = refs.pop()
    ref_rows = group[group['stem'].eq(ref_stem)]
    if len(ref_rows) != 1:
        raise ValueError(f'Existing group={group_id} has no unique reference metadata row: {ref_stem}')
    return ref_stem, ref_rows.iloc[0], group.iloc[0]


def _resolve_old_image(raw: str, config: dict) -> Path:
    """Resolve an existing reference BMP path using the project path mapping.

    Parameters
    ----------
    raw : str
        Historical image path value.
    config : dict
        Parsed project configuration mapping.

    Returns
    -------
    Path
        Result of the operation described above."""
    cfg = get_path_resolving_config(config, 'step2')
    p = resolve_flexible_path(raw, from_drive=cfg.get('from_drive'), to_drive=cfg.get('to_drive'),
                              from_root=cfg.get('from_root'), to_root=cfg.get('to_root'), must_exist=True)
    return Path(p)


def _as_bool(v: Any) -> bool:
    """Convert a historical CSV value to a Boolean flag.

    Parameters
    ----------
    v : Any
        Value to interpret.

    Returns
    -------
    bool
        Result of the operation described above."""
    return str(v).strip().lower() in {'true', 'yes', '1'}


def _as_point(row: pd.Series) -> tuple[float, float] | None:
    """Read bregma image coordinates from an existing registration row.

    Parameters
    ----------
    row : pd.Series
        Metadata row for the moving recording.

    Returns
    -------
    tuple[float, float] | None
        Result of the operation described above."""
    try:
        y, x = float(row['bregma_cy']), float(row['bregma_cx'])
    except (ValueError, TypeError, KeyError):
        return None
    return (y, x) if np.isfinite(y) and np.isfinite(x) else None


def _new_artifact_paths(paths: dict, stem: str, ref_stem: str, wave: str, config: dict) -> list[Path]:
    """Enumerate all destination artifacts for a new registered recording.

    Parameters
    ----------
    paths : dict
        Resolved project output paths.
    stem : str
        Recording stem identifier.
    ref_stem : str
        Reference recording stem identifier.
    wave : str
        Wavelength label used for registration.
    config : dict
        Parsed project configuration mapping.

    Returns
    -------
    list[Path]
        Result of the operation described above."""
    c = canonical_mask_paths(paths, stem)
    files = [c[k] for k in ('brain_auto', 'bad_auto', 'ring_auto', 'bregma_auto',
                           'vessels_auto', 'bregma_manual')]
    files.append(mask_for_reg_per_stem_path(paths, stem))
    if (config.get('step2', {}).get('io', {}) or {}).get('save_registered_bmps', True):
        files.append(paths['step2_root'] / 'registered_bmps' /
                     f'{stem}_{wave}_registered_to_{ref_stem}.bmp')
    return files


@dataclasses.dataclass
class Step2AddPlan:
    """Preflight plan for incremental Step2 registration.

        Attributes
        ----------
        exp_day : str
            Experimental-day label being appended.
        candidates : pandas.DataFrame
            New recordings selected for registration.
        references : dict
            Original reference stem for each registration group.
        collisions : list
            Conflicting output paths detected before committing."""
    config_path: Path
    exp_day: str
    config: dict
    paths: dict[str, Path]
    candidates: pd.DataFrame
    old_metadata: pd.DataFrame
    metadata_columns: list[str]
    metadata_sha256: str
    manifest_sha256: str
    config_sha256: str
    references: dict[str, str]
    collisions: list[str]
    report_dir: Path

    def summary(self) -> dict[str, Any]:
        """Describe candidate recordings, reference choices, and output collisions.

        Returns
        -------
        dict[str, Any]
            Result of the operation described above."""
        return {
            'exp_day': self.exp_day,
            'candidate_stems': self.candidates['stem'].astype(str).tolist(),
            'n_new': len(self.candidates),
            'existing_reference_by_group': self.references,
            'existing_outputs_collisions': self.collisions,
            'metadata_csv': str(self.paths['metadata_registration_csv']),
            'report_dir': str(self.report_dir),
        }


def plan_add_step2_day(config_path: str | Path, exp_day: str) -> Step2AddPlan:
    """Preflight an incremental Step2 registration for one experimental day.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Configuration of the existing project.
    exp_day : str
        Exact ``exp_day`` value to append.

    Returns
    -------
    Step2AddPlan
        New candidate stems, historical reference selection per group,
        CSV fingerprints, and any destination collisions.

    Notes
    -----
    The original ``metadata_registration.csv`` determines the true reference
    for each existing mouse group, including any historical fallback. This
    preflight reads previous Step2 results but does not update them."""
    config_path = Path(config_path)
    config = load_config(config_path)
    paths = _paths(config)
    day = str(exp_day).strip()
    if not day or any(x in day for x in ('/', '\\')):
        raise ValueError('exp_day must be a single non-empty day label')
    old_columns = _csv_header(paths['metadata_registration_csv'])
    _csv_header(paths['manifest_path'])
    old = pd.read_csv(paths['metadata_registration_csv'], dtype=str, keep_default_na=False,
                      encoding='utf-8-sig')
    if old.empty or not {'group_id', 'stem', 'ref_stem'}.issubset(old.columns):
        raise ValueError('Existing metadata_registration.csv is empty or lacks required group columns')
    if old['stem'].duplicated().any():
        raise ValueError('Existing metadata has duplicate stem entries')
    mfest = pd.read_csv(paths['manifest_path'], dtype=str, keep_default_na=False,
                        encoding='utf-8-sig')
    if 'exp_day' not in mfest.columns:
        raise ValueError('Manifest lacks exp_day')
    raw_day = mfest[mfest['exp_day'].eq(day)].copy()
    if raw_day.empty:
        raise ValueError(f'No manifest stems with exp_day={day!r}')
    if raw_day['stem'].duplicated().any():
        raise ValueError('Manifest has duplicate stem for selected day')
    previously_done = set(old['stem'].astype(str))
    target = raw_day[~raw_day['stem'].isin(previously_done)].copy()
    if target.empty:
        raise ValueError('No NEW stems for this exp_day: all already present in Step2 metadata')

    # Run the ORIGINAL Step2 index builder, but redirect its optional missing-input
    # report away from existing Step2. All its validation/variant rules stay intact.
    with tempfile.TemporaryDirectory(prefix='step2_add_preflight_') as tmp:
        isolated = dict(paths)
        isolated['step2_root'] = Path(tmp)
        index = core.build_step2_index_from_manifest(paths['manifest_path'], config, isolated,
                                                     require_h5=True)
    found = index[index['stem'].astype(str).isin(set(target['stem']))].copy()
    missed = set(target['stem']) - set(found['stem'].astype(str))
    if missed:
        raise ValueError(f'Step2 rejected new stems (missing H5/BMP, superseded repeat or excluded): {sorted(missed)}')
    # The Step2 index performs the ORIGINAL group_by/variant handling.
    allowed_day = found[found['exp_day'].astype(str).eq(day)].copy()
    if len(allowed_day) != len(found):
        raise ValueError('Step2 index returned a different experimental day')
    candidates = allowed_day.reset_index(drop=True)
    references: dict[str, str] = {}
    collisions = []
    reg_wave = core.get_registration_wavelength(config)
    for group_id, group_rows in candidates.groupby('group_id', sort=True):
        prior = old[old['group_id'].eq(str(group_id))]
        if prior.empty:
            raise ValueError(f'New stem belongs to unknown group_id={group_id}. This mode only adds days to existing mice.')
        ref_stem, ref_row, _ = _existing_ref(prior, str(group_id))
        references[str(group_id)] = ref_stem
        if _as_point(ref_row) is None:
            raise ValueError(f'Historical reference bregma missing: {ref_stem}. Resolve it using original workflow first.')
        if not ref_row.get('fixed_registration_path', ''):
            raise ValueError(f'Historical reference BMP path missing for {ref_stem}')
        ref_p = _resolve_old_image(ref_row['fixed_registration_path'], config)
        # Existing reference masks are used as read-only inputs for registration.
        cm = canonical_mask_paths(paths, ref_stem)
        for key in ('brain_auto', 'ring_auto', 'bad_auto'):
            if not cm[key].is_file():
                raise FileNotFoundError(f'Historical ref mask missing: {cm[key]}')
        if (config['step2']['registration'].get('mask_mode') in ('group_union', 'hybrid')
            and str(config['step2']['registration'].get('roi_mode', 'ring')).lower() not in
              {'full', 'full_image', 'whole_image', 'image', 'brain', 'brain_minus_bad', 'brain-no-bad', 'brain_without_bad'}):
            union = mask_for_reg_union_path(paths, group_id).parent
            for filename in ('ring_union.npy', 'bad_union.npy'):
                if not (union / filename).is_file():
                    raise FileNotFoundError(f'Historical group union input missing: {union / filename}')
        for stem in group_rows['stem'].astype(str):
            # Guard ALL existing contents in per-stem mask directories, including
            # Step3 manual outputs. Never silently reuse/overwrite an old stem folder.
            md = canonical_mask_paths(paths, stem)['brain_auto'].parent
            if md.exists() and any(md.iterdir()):
                collisions.append(str(md))
            for p in _new_artifact_paths(paths, stem, ref_stem, reg_wave, config):
                if p.exists():
                    collisions.append(str(p))
    report_dir = paths['step2_root'] / 'incremental_step2_logs'
    return Step2AddPlan(config_path, day, config, paths, candidates, old, old_columns,
                        _sha256(paths['metadata_registration_csv']), _sha256(paths['manifest_path']),
                        _sha256(config_path), references, list(dict.fromkeys(collisions)), report_dir)


def _old_reference_record(plan: Step2AddPlan, group_id: str) -> tuple[dict, pd.Series, pd.Series]:
    """Load historical per-group reference metadata for a new registration.

    Parameters
    ----------
    plan : Step2AddPlan
        Validated incremental-operation preflight plan.
    group_id : str
        Registration-group identifier.

    Returns
    -------
    tuple[dict, pd.Series, pd.Series]
        Result of the operation described above."""
    group = plan.old_metadata[plan.old_metadata['group_id'].eq(group_id)]
    ref_stem, ref_row, any_row = _existing_ref(group, group_id)
    c = canonical_mask_paths(plan.paths, ref_stem)
    bregma = _as_point(ref_row)
    assert bregma is not None
    return {
        'stem': ref_stem,
        'group_id': group_id,
        'registration_bmp': _resolve_old_image(ref_row['fixed_registration_path'], plan.config),
        'ring': np.load(c['ring_auto']).astype(bool),
        'bad': np.load(c['bad_auto']).astype(bool),
        'brain': np.load(c['brain_manual'] if c['brain_manual'].exists() else c['brain_auto']).astype(bool),
        'bregma': bregma,
        'bregma_found': True,
        'bregma_source': ref_row.get('bregma_source', 'auto'),
    }, ref_row, any_row


def _new_record(plan: Step2AddPlan, idx: pd.Series, staging: Path) -> dict:
    """Build one registered new recording in the temporary staging area.

    Parameters
    ----------
    plan : Step2AddPlan
        Validated incremental-operation preflight plan.
    idx : pd.Series
        Row index in the new-recording table.
    staging : Path
        Temporary directory for generated files before commit.

    Returns
    -------
    dict
        Result of the operation described above."""
    cfg = plan.config
    step = cfg['step2']
    stem = str(idx['stem'])
    h5_dataset = (step.get('h5', {}) or {}).get('dataset', 'exported_data')
    wave_paths = {}
    for wl in core.get_step2_mask_wavelengths(cfg):
        raw = str(idx.get(f'h5_{wl}', ''))
        if raw and Path(raw).is_file():
            wave_paths[wl] = core.read_probs_h5(Path(raw), dataset=h5_dataset)
    morph = step.get('morphology', {}) or {}
    brain, ring = core.build_brain_and_ring_from_sources(
        wave_paths, cfg, close_r=int(morph.get('close_r_brain', 3)),
        r_ring=int(morph.get('ring_radius', 10)))
    cy, cx, auto_mask, bfound = core.bregma_centroid_from_sources(wave_paths, cfg)
    bad = core.build_bad_mask_from_sources(wave_paths, cfg, brain.shape)
    vessels = core.combine_source_masks(wave_paths, cfg, 'vessels', required=False)
    if vessels is None:
        vessels = np.zeros_like(brain, bool)
    registration_wl = core.get_registration_wavelength(cfg)
    bmp = Path(str(idx[f'bmp_{registration_wl}']))
    img, _ = core.load_bmp_u8_and_f32(bmp)
    if brain.shape != img.shape or ring.shape != img.shape or bad.shape != img.shape:
        raise ValueError(f'Ilastik mask/BMP shape mismatch: {stem}: mask={brain.shape}, bmp={img.shape}')
    stage_paths = {'masks_root': staging / 'masks'}
    maskp = canonical_mask_paths(stage_paths, stem)
    core.save_mask(maskp['brain_auto'], brain)
    core.save_mask(maskp['ring_auto'], ring)
    core.save_mask(maskp['bad_auto'], bad)
    core.save_mask(maskp['bregma_auto'], auto_mask)
    if int(vessels.sum()) > 0 or core.iter_mask_source_entries(cfg, role='vessels'):
        core.save_mask(maskp['vessels_auto'], vessels)
    rec = {k: v for k, v in idx.to_dict().items() if core._is_simple_metadata_value(v)}
    rec.update({
        'stem': stem, 'group_id': str(idx['group_id']),
        'img_dir': idx.get('img_dir', ''),
        'raw_zarr_path': idx.get('raw_zarr_path', ''),
        'processed_zarr_path': idx.get('processed_zarr_path', ''),
        'registration_wavelength': registration_wl,
        'registration_bmp': bmp,
        'ring': ring, 'bad': bad, 'brain': brain, 'vessels': vessels,
        'bregma': (cy, cx), 'bregma_mask': auto_mask,
        'bregma_found': bool(bfound), 'bregma_source': 'auto' if bfound else 'missing',
        'project_raw_rotation_deg': core.raw_rotation_degrees(cfg),
        'project_raw_scale': core.raw_scale_factor(cfg),
        'step2_image_space': 'rotated_fullres',
        'mask_image_space': 'rotated_fullres',
        'raw_zarr_image_space': 'rotated_scaled',
        'mask_file_logic': 'canonical_by_stem',
        'mask_sources_json': str(core.get_mask_sources_config(cfg)),
    })
    for wl in core.get_step2_needed_wavelengths(cfg):
        if f'bmp_{wl}' in idx.index and pd.notna(idx[f'bmp_{wl}']):
            rec[f'bmp_{wl}'] = Path(str(idx[f'bmp_{wl}']))
            rec[f'ref_{wl}_path'] = str(idx[f'bmp_{wl}'])
        if f'h5_{wl}' in idx.index and pd.notna(idx[f'h5_{wl}']):
            rec[f'h5_{wl}'] = str(idx[f'h5_{wl}'])
            rec[f'prob_{wl}_path'] = str(idx[f'h5_{wl}'])
    return rec


def _union_arrays(plan: Step2AddPlan, gid: str, recs: dict[str, dict], ref: dict,
                  ref_bmp_shape: tuple[int, int], reg: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Update staged group-level union masks while retaining historical masks.

    Parameters
    ----------
    plan : Step2AddPlan
        Validated incremental-operation preflight plan.
    gid : str
        Registration-group identifier.
    recs : dict[str, dict]
        Registration records for the group.
    ref : dict
        Whether the masks belong to the reference recording.
    ref_bmp_shape : tuple[int, int]
        Pixel dimensions of the reference BMP image.
    reg : dict
        Registration settings from YAML.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        Result of the operation described above."""
    if reg.get('mask_mode') not in ('group_union', 'hybrid'):
        return None, None
    roi = str(reg.get('roi_mode', 'ring')).lower()
    if roi in {'full', 'full_image', 'whole_image', 'image', 'brain', 'brain_minus_bad', 'brain-no-bad', 'brain_without_bad'}:
        # In the original algorithm this union is computed/saved but not used by
        # build_registration_roi. For an incremental update it must NOT be saved.
        return None, None
    union_dir = mask_for_reg_union_path(plan.paths, gid).parent
    ring = np.load(union_dir / 'ring_union.npy').astype(bool).copy()
    bad = np.load(union_dir / 'bad_union.npy').astype(bool).copy()
    if ring.shape != ref_bmp_shape or bad.shape != ref_bmp_shape:
        raise ValueError(f'Historical group-union shape mismatch: group={gid}')
    y0, x0 = ref['bregma']
    for rec in recs.values():
        if not rec['bregma_found']:
            continue
        cy, cx = rec['bregma']
        ring |= core.translate_mask(rec['ring'], dy=y0-cy, dx=x0-cx)
        bad |= core.translate_mask(rec['bad'], dy=y0-cy, dx=x0-cx)
    return ring, bad


def _original_reference_fields(old_row: pd.Series) -> dict:
    # Retain reference identity/metadata from the HISTORICAL Step2 row, not a
    # freshly selected reference. The real ref_stem may be a fallback day.
    """Preserve the original reference-identifying metadata fields.

    Parameters
    ----------
    old_row : pd.Series
        Stored historical metadata row.

    Returns
    -------
    dict
        Result of the operation described above."""
    return {k: v for k, v in old_row.items()
            if k.startswith('ref_meta_') or k in {
                'ref_stem_original', 'ref_group_id', 'ref_group_keys_json',
                'ref_metadata_json', 'ref_stem_keys_json'}}


def _compute_rows(plan: Step2AddPlan, staging: Path) -> tuple[pd.DataFrame, dict[Path, Path]]:
    """Run registration for new stems and construct the appended metadata rows.

    Parameters
    ----------
    plan : Step2AddPlan
        Validated incremental-operation preflight plan.
    staging : Path
        Temporary directory for generated files before commit.

    Returns
    -------
    tuple[pd.DataFrame, dict[Path, Path]]
        Result of the operation described above."""
    cfg = plan.config
    scfg = cfg['step2']
    reg = scfg.get('registration', {}) or {}
    qc = scfg.get('qc', {}) or {}
    io_cfg = scfg.get('io', {}) or {}
    mask_mode = reg.get('mask_mode', 'pairwise')
    method = str(reg.get('method', 'ring_sobel')).strip().lower()
    roi_mode = str(reg.get('roi_mode', 'full_image' if method == 'vessel_sobel' else 'ring')).strip().lower()
    anchor = bool(reg.get('anchor_bregma', True))
    strict = bool(reg.get('strict_bregma_anchor', True))
    tol = float(reg.get('bregma_anchor_tol_px', 1e-3))
    border = int(reg.get('ignore_border_px', 0) or 0)
    qc_thresholds = bool(qc.get('apply_thresholds', False))
    rotation_deg = core.raw_rotation_degrees(cfg)
    raw_scale = core.raw_scale_factor(cfg)
    wave = core.get_registration_wavelength(cfg)
    staging_paths = {'masks_root': staging / 'masks'}
    rows: list[dict] = []
    artifacts: dict[Path, Path] = {}  # staging source -> final destination

    for gid, g in plan.candidates.groupby('group_id', sort=True):
        gid = str(gid)
        ref, ref_row, old_any = _old_reference_record(plan, gid)
        ref_stem = ref['stem']
        _, fixed = core.load_bmp_u8_and_f32(ref['registration_bmp'])
        if any(x.shape != fixed.shape for x in [ref['ring'], ref['bad'], ref['brain']]):
            raise ValueError(f'Historical reference masks do not match BMP shape: {ref_stem}')
        recs = {str(r['stem']): _new_record(plan, r, staging) for _, r in g.iterrows()}
        for rec in recs.values():
            if rec['brain'].shape != fixed.shape:
                raise ValueError(f'New and historical reference images have different shape: {rec["stem"]}')
        union_ring, union_bad = _union_arrays(plan, gid, recs, ref, fixed.shape, {**reg, 'mask_mode': mask_mode})
        try:
            ref_exact = _as_bool(old_any['ref_exact_match'])
        except KeyError:
            ref_exact = False
        cy0, cx0 = ref['bregma']
        for stem in sorted(recs):
            rec = recs[stem]
            stage_mask_paths = canonical_mask_paths(staging_paths, stem)
            final_mask_paths = canonical_mask_paths(plan.paths, stem)
            for key in ('brain_auto','bad_auto','ring_auto','bregma_auto','vessels_auto','bregma_manual'):
                if stage_mask_paths[key].is_file():
                    artifacts[stage_mask_paths[key]] = final_mask_paths[key]

            if not rec['bregma_found']:
                row = core._base_bad_row(gid, stem, ref_stem, rec, ref, ref_exact, mask_mode)
                row.update({
                    'registration_method': method, 'registration_roi_mode': roi_mode,
                    'anchor_bregma': anchor, 'strict_bregma_anchor': strict,
                    'bregma_anchor_tol_px': tol, 'bregma_ref_cy': float(cy0),
                    'bregma_ref_cx': float(cx0), 'bregma_ref_found': True,
                    'bregma_source': 'missing', 'bregma_ref_source': ref['bregma_source'],
                    'bad_reasons': 'bregma_empty',
                })
                row.update(_original_reference_fields(old_any))
                rows.append(core.add_raw_geometry_metadata(row, raw_scale=raw_scale, rotation_deg=rotation_deg))
                continue
            cy, cx = rec['bregma']
            dy, dx = cy0-cy, cx0-cx
            initial = EuclideanTransform(translation=(dx, dy))
            _, moving = core.load_bmp_u8_and_f32(rec['registration_bmp'])
            moving0 = core.apply_tform(moving, initial, order=1)
            roi = core.build_registration_roi(
                roi_mode=roi_mode, mask_mode=mask_mode, ref_rec=ref, mov_rec=rec,
                init_dy=dy, init_dx=dx, image_shape=fixed.shape,
                group_union_ring=union_ring, group_union_bad=union_bad,
                ignore_border_px=border)
            mask_area = int(roi.sum())
            stage_roi = mask_for_reg_per_stem_path(staging_paths, stem)
            core.save_mask(stage_roi, roi)
            artifacts[stage_roi] = mask_for_reg_per_stem_path(plan.paths, stem)
            angle = 0.0; score = float('-inf'); total = initial; dy2 = dx2 = 0.0
            if mask_area > 0:
                fixed_feature = core.build_registration_feature(fixed, roi, reg)
                moving_feature = core.build_registration_feature(moving0, roi, reg)
                t2, angle, (dy2, dx2), score = core.estimate_feature_scored(
                    fixed_feature=fixed_feature, moving0_feature=moving_feature,
                    mask=roi, rot_center_yx=(cy0,cx0),
                    angle_range=float(reg.get('angle_range_deg',3.0)),
                    angle_step=float(reg.get('angle_step_deg',0.1)),
                    upsample=int(reg.get('upsample',10)), anchor_bregma=anchor)
                total = initial + t2
            bregma_error = (core.bregma_anchor_error_px(total, (float(cy), float(cx)), (float(cy0),float(cx0)))
                            if anchor else np.nan)
            if anchor and strict and np.isfinite(bregma_error) and bregma_error > tol:
                raise RuntimeError(f'Historical Step2 bregma anchor failed for new stem={stem}: {bregma_error} > {tol}')
            moved = core.apply_tform(moving, total, order=1)
            dest_reg = plan.paths['step2_root'] / 'registered_bmps' / f'{stem}_{wave}_registered_to_{ref_stem}.bmp'
            if io_cfg.get('save_registered_bmps', True):
                src_reg = staging / 'registered_bmps' / dest_reg.name
                core.save_bmp_uint8(src_reg, moved)
                artifacts[src_reg] = dest_reg
            tx, ty = map(float, total.translation)
            reasons = []
            if mask_area == 0:
                reasons.append('empty_registration_roi')
            elif not np.isfinite(score):
                reasons.append('invalid_registration_score')
            if qc_thresholds:
                if not np.isfinite(score) or score < float(qc.get('min_score',0.1)):
                    reasons.append('low_score')
                if mask_area < int(qc.get('min_mask_area',500)):
                    reasons.append('small_mask')
                if abs(tx) > float(qc.get('max_abs_dx',100)):
                    reasons.append('dx_large')
                if abs(ty) > float(qc.get('max_abs_dy',100)):
                    reasons.append('dy_large')
                if abs(float(angle)) > float(qc.get('max_abs_angle_deg',10)):
                    reasons.append('angle_large')
            if anchor and np.isfinite(bregma_error) and bregma_error > tol:
                reasons.append('bregma_anchor_error')
            row = {
                'group_id': gid, 'stem': stem, 'ref_stem': ref_stem,
                'ref_exact_match': ref_exact, 'bregma_ref_found': True,
                'bregma_found': True, 'bregma_ref_cy': float(cy0), 'bregma_ref_cx': float(cx0),
                'bregma_cy': float(cy), 'bregma_cx': float(cx),
                'bregma_source': rec['bregma_source'], 'bregma_ref_source': ref['bregma_source'],
                'init_shift_dy': float(dy), 'init_shift_dx': float(dx),
                'second_stage_free_shift_dy': float(dy2), 'second_stage_free_shift_dx': float(dx2),
                'angle_deg_found': float(angle), 'score': float(score),
                'translation_x': tx, 'translation_y': ty, 'rotation_rad': float(total.rotation),
                'bregma_registered_y': core.transform_point_yx(total, float(cy),float(cx))[0],
                'bregma_registered_x': core.transform_point_yx(total, float(cy),float(cx))[1],
                'bregma_error_px': float(bregma_error) if np.isfinite(bregma_error) else np.nan,
                'anchor_bregma': anchor, 'strict_bregma_anchor': strict,
                'bregma_anchor_tol_px': tol,
                'registration_method': method, 'registration_roi_mode': roi_mode,
                'registration_feature_method': method,
                'vessel_sobel_params_json': str(reg.get('vessel_sobel', {}) or {}),
                'mask_for_reg_area': mask_area, 'mask_mode': mask_mode,
                'mask_file_logic': 'canonical_by_stem',
                'is_bad': bool(reasons), 'bad_reasons': ','.join(reasons),
                'registration_wavelength': wave,
                'fixed_registration_path': str(ref['registration_bmp']),
                'moving_registration_path': str(rec['registration_bmp']),
                'saved_registered_path': str(dest_reg) if io_cfg.get('save_registered_bmps',True) else '',
            }
            for key, value in core._metadata_from_record(rec).items():
                row.setdefault(key, value)
            row.update(_original_reference_fields(old_any))
            rows.append(core.add_raw_geometry_metadata(row, raw_scale=raw_scale, rotation_deg=rotation_deg))
    result = pd.DataFrame(rows)
    result = drop_mask_path_metadata_columns(result)
    return result, artifacts


def _append_preserving_old_bytes(original: bytes, columns: list[str], new_rows: pd.DataFrame) -> bytes:
    """Append Step2 CSV rows without changing any original CSV bytes.

    Parameters
    ----------
    original : bytes
        Original CSV bytes or source file.
    columns : list[str]
        Expected CSV column names and order.
    new_rows : pd.DataFrame
        New records to append.

    Returns
    -------
    bytes
        Result of the operation described above."""
    unexpected = [c for c in new_rows.columns if c not in columns]
    if unexpected:
        # The old Step2 metadata schema must remain authoritative. Never silently
        # drop computed output fields or change historical metadata column format.
        populated = [c for c in unexpected if new_rows[c].notna().any()]
        if populated:
            raise ValueError('New Step2 metadata fields absent from old CSV schema: ' + ', '.join(populated))
    # Duplicate new stems and existing rows already checked in preflight.
    text = original.decode('utf-8-sig')
    eol = '\r\n' if '\r\n' in text[:10000] else '\n'
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator=eol)
    for _, r in new_rows.iterrows():
        obj = {}
        for c in columns:
            v = r.get(c, '')
            if v is None or (not isinstance(v, (list,dict)) and pd.isna(v)):
                v = ''
            obj[c] = v
        writer.writerow(obj)
    separator = b'' if original.endswith((b'\r\n', b'\n')) else eol.encode('ascii')
    return original + separator + output.getvalue().encode('utf-8')


def commit_add_step2_day(plan: Step2AddPlan) -> dict[str, Any]:
    """Register and commit only stems approved by the incremental Step2 plan.

    Parameters
    ----------
    plan : Step2AddPlan
        Current preflight plan with existing reference metadata.

    Returns
    -------
    dict of str to object
        Commit status, new stems, artifact count, backup, and reference choices.

    Notes
    -----
    Checks SHA-256 digests of the config, Step1 manifest and Step2 metadata.
    New masks and registered outputs are computed in a staging directory and
    only then moved to previously unused paths. The original metadata CSV
    bytes remain unchanged at the beginning of the newly appended CSV."""
    if plan.collisions:
        raise FileExistsError(f'Existing Step2 paths for new stems: {plan.collisions[:10]}')
    meta = plan.paths['metadata_registration_csv']
    manifest = plan.paths['manifest_path']
    if (_sha256(meta) != plan.metadata_sha256 or _sha256(manifest) != plan.manifest_sha256
            or _sha256(plan.config_path) != plan.config_sha256):
        raise RuntimeError('Config, manifest or existing metadata changed after preflight. Run plan_add_step2_day again.')
    # Check all potential destinations BEFORE expensive calculations.
    wave = core.get_registration_wavelength(plan.config)
    for _, r in plan.candidates.iterrows():
        ref = plan.references[str(r['group_id'])]
        for p in _new_artifact_paths(plan.paths, str(r['stem']), ref, wave, plan.config):
            if p.exists():
                raise FileExistsError(f'Refusing to overwrite existing Step2 file: {p}')
    plan.report_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f'add_{plan.exp_day.replace("/","-")}_', dir=plan.report_dir))
    moved: list[Path] = []
    tmp_metadata: Path | None = None
    try:
        new_rows, artifacts = _compute_rows(plan, staging)
        if set(new_rows['stem'].astype(str)) != set(plan.candidates['stem'].astype(str)):
            raise RuntimeError('Calculated Step2 stems differ from preflight')
        new_bytes = _append_preserving_old_bytes(meta.read_bytes(), plan.metadata_columns, new_rows)
        if (_sha256(meta) != plan.metadata_sha256 or _sha256(manifest) != plan.manifest_sha256
                or _sha256(plan.config_path) != plan.config_sha256):
            raise RuntimeError('Config, manifest or metadata changed during calculations. Nothing has been committed.')
        for src, dst in artifacts.items():
            if not src.exists() or dst.exists():
                raise FileExistsError(f'Step2 file missing/collision before commit: {src} -> {dst}')
        # Write complete replacement into a unique file on the same volume; old bytes
        # remain verbatim at the start. Replace at very end of commit.
        with tempfile.NamedTemporaryFile(prefix=meta.name+'.incremental.', suffix='.tmp',
                                         dir=meta.parent, delete=False) as f:
            tmp_metadata = Path(f.name)
            f.write(new_bytes)
            f.flush(); os.fsync(f.fileno())
        for src, dst in artifacts.items():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                raise FileExistsError(f'Concurrent file collision: {dst}')
            shutil.move(str(src), str(dst))
            moved.append(dst)
        if _sha256(meta) != plan.metadata_sha256:
            raise RuntimeError('Metadata changed immediately before commit; will roll back new files')
        # Preserve a restorable, byte-identical copy of the historical metadata.
        backup = plan.report_dir / (meta.stem + '_before_add_' +
                                    dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.csv')
        with backup.open('xb') as bf:
            bf.write(meta.read_bytes())
        os.replace(tmp_metadata, meta)
        tmp_metadata = None
        # Inspect newly appended rows without touching old results.
        return {'status': 'added', 'exp_day': plan.exp_day, 'new_stems': list(new_rows['stem'].astype(str)),
                'n_added': len(new_rows), 'reference_by_group': plan.references,
                'new_files': len(moved), 'metadata_csv': str(meta), 'backup': str(backup),
                'log_dir': str(plan.report_dir)}
    except Exception:
        # No historical file is ever deleted. Undo ONLY newly moved files.
        for f in reversed(moved):
            try:
                f.unlink()
            except OSError:
                pass
        raise
    finally:
        if tmp_metadata is not None:
            tmp_metadata.unlink(missing_ok=True)
        # Keep only logs/empty staging folder; nothing historical lives here.
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--exp-day', required=True)
    parser.add_argument('--commit', action='store_true', help='Explicitly append after successful preflight')
    args = parser.parse_args()
    p = plan_add_step2_day(args.config, args.exp_day)
    print(json.dumps(p.summary(), ensure_ascii=False, indent=2))
    if args.commit:
        print(json.dumps(commit_add_step2_day(p), ensure_ascii=False, indent=2))
