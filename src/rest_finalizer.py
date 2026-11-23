"""REST behavior summary and authoritative REST-Review-only finalization.

No primary-QC import/fallback; unlabeled time is NOT-REST, not inferred RUN.
Inspect missing processed Zarr without raising. Never silently export usable windows
for stems with invalid or missing processed data. Designed for stepwise Jupyter.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

FINAL_COLUMNS = [
    'stem','group_id','ref_stem','mouse_id','experimental_group','exp_day','state','protocol',
    'segment_number_source','t0_sec','t1_sec','duration_sec','source','shadow_mask_path',
    'segment_number','duration_matches_expected','slot_id','parent_rest_index','nominal_t0_sec',
]
STATUS_COLUMNS = [
    'stem','mouse_id','experimental_group','exp_day','state','protocol',
    'processed_status','processed_path','n_frames','n_candidates','n_reviewed',
    'n_accepted','n_rejected','n_unclear','n_overlaps',
    'review_status','final_state','reason','behavior_rest_sec',
]
SHADOW_COLUMNS = ['stem','shadow_mask_path','shadow_status','source','review_status',
                  'n_pixels','mask_shape','reason']


def text(v: Any) -> str:
    """Normalize a nullable metadata value to a stripped string.

    Parameters
    ----------
    v : Any
        Scalar CSV or metadata field.

    Returns
    -------
    str
        Empty string for missing values; otherwise stripped text.
    """
    if v is None: return ''
    try:
        if pd.isna(v): return ''
    except (TypeError, ValueError): pass
    return str(v).strip()


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read an optional CSV without substituting any alternative source.

    Parameters
    ----------
    path : str or pathlib.Path
        Expected CSV location.

    Returns
    -------
    pandas.DataFrame
        Text-preserving CSV contents (UTF-8 with optional BOM), or an empty
        DataFrame if the file does not exist. Other read errors propagate.
    """
    path=Path(path)
    return pd.read_csv(path, dtype=object, keep_default_na=False, encoding='utf-8-sig') if path.is_file() else pd.DataFrame()


def unique_latest(df: pd.DataFrame, key='stem') -> dict[str, dict]:
    """Index the last row for each nonempty recording identifier.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table, potentially containing repeated keys.
    key : str, optional
        Identity column, ``stem`` by default.

    Returns
    -------
    dict[str, dict]
        Mapping from identity to the last CSV row.

    Raises
    ------
    ValueError
        If a nonempty input table lacks the requested key.
    """
    if df.empty: return {}
    if key not in df: raise ValueError(f'Missing {key} in CSV')
    return {text(r[key]):r.to_dict() for _,r in df.drop_duplicates(key, keep='last').iterrows() if text(r[key])}


def assign_groups(config: dict) -> dict[str,str]:
    """Map mouse IDs to the experimental groups defined in YAML.

    Parameters
    ----------
    config : dict
        Project configuration. Only
        ``analysis_selection.experimental_groups`` supplies assignments.

    Returns
    -------
    dict[str, str]
        Mouse-to-experimental-group mapping.

    Raises
    ------
    ValueError
        If one mouse is assigned to different groups.
    """
    groups = config.get('analysis_selection', {}).get('experimental_groups', {}) or {}
    mp={}
    for group,mice in groups.items():
        for mouse in mice or []:
            mouse=text(mouse)
            if mouse in mp and mp[mouse]!=str(group):
                raise ValueError(f'Mouse {mouse} appears in both {mp[mouse]} and {group}')
            mp[mouse]=str(group)
    return mp


def load_inputs(config_path: str | Path) -> dict[str,Any]:
    """Load the current Step4 REST-state index and manual annotation CSVs.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Project YAML passed to ``build_step4_index``.

    Returns
    -------
    dict[str, Any]
        ``index``, ``config``, ``paths``, output ``root``, behavioral
        ``annotations``, and manually reviewed REST ``review``.

    Notes
    -----
    Includes technically flagged Step2 records for diagnostics. The only
    REST review source is ``behavior_review/rest_segment_rescue_60s/``
    ``rest_segment_rescue_review.csv``; no legacy primary-QC fallback.
    """
    from step4_signal_processing import build_step4_index
    # Do not equate Step2 automatic is_bad flag with finalized scientific QC.
    index,cfg,paths=build_step4_index(config_path, include_bad=True, filters={})
    protocol=index['protocol_base'] if 'protocol_base' in index else index['protocol']
    index=index.loc[protocol.astype(str).str.fullmatch(r'(?i)rest[-_ ]?state',na=False)].copy()
    groups=assign_groups(cfg)
    index['experimental_group']=index['mouse_id'].map(groups).fillna('UNASSIGNED')
    root=Path(paths['out_root'])/'behavior_review'
    return dict(index=index.reset_index(drop=True),config=cfg,paths=paths,root=root,
                annotations=read_csv(root/'behavior_review_annotations.csv'),
                review=read_csv(root/'rest_segment_rescue_60s'/'rest_segment_rescue_review.csv'))


def inspect_index(index:pd.DataFrame, config:dict, *,signal:str|None=None) -> pd.DataFrame:
    """Inspect processed and raw Zarr availability without filtering records.

    Parameters
    ----------
    index : pandas.DataFrame
        REST-state rows with resolved raw/processed Zarr locations.
    config : dict
        YAML providing ``raw.fps``, optional ``raw.zarr_key`` and
        ``functional_connectivity.signal`` (fallback ``dcHbT``).
    signal : str or None, optional
        Signal variable to validate in the processed dataset.

    Returns
    -------
    pandas.DataFrame
        Input metadata with file statuses, frame count, image size in pixels,
        duration in seconds, source of duration, and diagnostic errors.

    Notes
    -----
    A valid processed signal must have ``time``, ``y`` and ``x`` dimensions.
    If processed frame count is unavailable, raw Zarr can supply frame count
    for recording-duration diagnostics. Missing data are reported, not excluded.
    """
    import xarray as xr
    signal=signal or text(config.get('functional_connectivity',{}).get('signal','dcHbT')) or 'dcHbT'
    rows=[]
    fps=float(config['raw']['fps'])
    for _,r in index.iterrows():
        rec=r.to_dict()
        pp=Path(text(rec.get('processed_zarr_path_resolved')) or '/path/not/specified')
        rp=Path(text(rec.get('raw_zarr_selected_for_processing_path_resolved')) or
                text(rec.get('raw_zarr_path_resolved')) or '/path/not/specified')
        rec.update(processed_path=str(pp),raw_path=str(rp),processed_status='missing',
                   processed_signal=signal,raw_status='missing',n_frames=np.nan,
                   height=np.nan,width=np.nan,recording_sec=np.nan,duration_source='',inspection_error='')
        if pp.is_dir():
            ds=None
            try:
                ds=xr.open_zarr(pp,chunks=None,consolidated=False,mask_and_scale=False)
                if signal not in ds:
                    rec['processed_status']='signal_missing'
                else:
                    v=ds[signal]
                    if not {'time','y','x'}.issubset(v.dims):
                        rec['processed_status']='invalid_dimensions'
                    else:
                        rec.update(processed_status='ok',n_frames=int(v.sizes['time']),
                                   height=int(v.sizes['y']),width=int(v.sizes['x']),
                                   duration_source='processed_zarr')
            except Exception as exc:
                rec['processed_status']='open_error';rec['inspection_error']=str(exc)
            finally:
                if ds is not None: ds.close()
        if rp.is_dir():
            ds=None
            try:
                ds=xr.open_zarr(rp,chunks=None,consolidated=False,mask_and_scale=False)
                k=text(config.get('raw',{}).get('zarr_key','raw_record'))
                if k in ds and 'time' in ds[k].dims:
                    rec['raw_status']='ok'
                    if not np.isfinite(rec['n_frames']):
                        rec.update(n_frames=int(ds[k].sizes['time']),duration_source='raw_zarr')
                else: rec['raw_status']='signal_missing'
            except Exception as exc:
                rec['raw_status']='open_error'
                rec['inspection_error'] = (rec['inspection_error']+'; raw: '+str(exc)).strip('; ')
            finally:
                if ds is not None: ds.close()
        if np.isfinite(rec['n_frames']):rec['recording_sec']=float(rec['n_frames'])/fps
        rows.append(rec)
    return pd.DataFrame(rows)


def _json_list(value:Any) -> list:
    """Parse a nullable CSV JSON field and require a list result.

    Parameters
    ----------
    value : Any
        Serialized JSON; empty values represent an empty list.

    Returns
    -------
    list
        Parsed list.

    Raises
    ------
    ValueError
        If the JSON is invalid or the result is not a list.
    """
    v=json.loads(text(value) or '[]')
    if not isinstance(v,list):raise ValueError('JSON must contain a list')
    return v


def behavior_intervals(value:Any, *, duration:float=np.nan, tol:float=.08) -> list[dict]:
    """Validate and normalize manually annotated REST/RUN intervals.

    Parameters
    ----------
    value : Any
        JSON list of ``label``, ``t0_sec``, and ``t1_sec`` dictionaries.
    duration : float, optional
        Full recording duration in seconds; NaN disables the end check.
    tol : float, optional
        Allowed tolerance in seconds for the end-of-recording check.

    Returns
    -------
    list[dict]
        Validated intervals with lowercase labels and times in seconds.

    Notes
    -----
    Supports explicit ``rest`` and ``run`` labels. Unlabeled time is
    **not REST** but is not automatically assigned ``run``. Overlap between
    REST and RUN is checked by ``summarize_behavior``.
    """
    out=[]
    for item in _json_list(value):
        if not isinstance(item,dict):raise ValueError('behavior interval is not a dictionary')
        label=text(item.get('label')).lower()
        if label not in {'rest','run'}:raise ValueError(f'Unsupported behavior label: {label}')
        a,b=float(item['t0_sec']),float(item['t1_sec'])
        if not(math.isfinite(a) and math.isfinite(b) and a>=0 and b>a):
            raise ValueError('Invalid interval time')
        if math.isfinite(duration) and b>duration+tol:
            raise ValueError(f'Interval extends beyond recording: {b:g}>{duration:g}')
        out.append(dict(label=label,t0_sec=a,t1_sec=b))
    return out


def merge_intervals(intervals:list[tuple[float,float]]) -> list[tuple[float,float]]:
    """Combine overlapping or touching time intervals.

    Parameters
    ----------
    intervals : list[tuple[float, float]]
        Start/end pairs in seconds.

    Returns
    -------
    list[tuple[float, float]]
        Sorted, merged intervals in seconds.

    Notes
    -----
    Treats boundaries within ``1e-9`` seconds as contiguous.
    """
    out=[]
    for a,b in sorted(intervals):
        if out and a <= out[-1][1]+1e-9:out[-1]=(out[-1][0],max(b,out[-1][1]))
        else:out.append((a,b))
    return out


def summarize_behavior(annotations:pd.DataFrame, inspected_index:pd.DataFrame, config:dict,
                       *,exclude_unassigned:bool=False) -> dict[str,pd.DataFrame]:
    """Calculate per-recording and per-mouse/day REST measures.

    Parameters
    ----------
    annotations : pandas.DataFrame
        Manual behavior annotations, including time intervals in seconds.
    inspected_index : pandas.DataFrame
        Step4 REST-state diagnostic table with recording duration.
    config : dict
        Uses ``raw.fps``, ``analysis_selection.experimental_groups`` and
        optional ``rest_qc.anesthesia_states`` (default ``['Iso']``).
    exclude_unassigned : bool, optional
        Omit ``UNASSIGNED`` from mouse-day aggregation; stem diagnostics remain.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``stem``: per-recording measures and statuses; ``periods``: merged
        continuous REST intervals; ``mouse_day``: mouse/day/state aggregation.

    Notes
    -----
    REST percentage is 100 * total REST seconds / full recording seconds,
    using raw or processed Zarr duration, never the last annotation boundary.
    Mean period duration is total REST seconds / number of merged REST periods.
    Undefined duration yields NaN percentage and a diagnostic status.
    Unlabeled time is not called RUN. Anesthesia states are marked
    ``not_applicable_anest`` and do not contribute REST measures.
    """
    fps=float(config['raw']['fps'])
    lookup=unique_latest(inspected_index)
    groups=assign_groups(config)
    anesthesia_states={text(s).lower() for s in config.get('rest_qc',{}).get('anesthesia_states',['Iso'])}
    stem_rows=[];period_rows=[]
    for stem,a in unique_latest(annotations).items():
        proto=text(a.get('protocol_base') or a.get('protocol'))
        if proto.lower().replace('_','-') != 'rest-state': continue
        src=lookup.get(stem,{})
        meta={k:text(a.get(k)) or text(src.get(k)) for k in ['mouse_id','exp_day','state','protocol','group_id']}
        mouse=meta['mouse_id']
        if not mouse:mouse=text(src.get('mouse_id'))
        group=groups.get(mouse,'UNASSIGNED')
        dur=float(src.get('recording_sec',np.nan)) if src else np.nan
        status=text(a.get('review_status')).lower()
        base=dict(stem=stem,**meta,experimental_group=group,processed_status=text(src.get('processed_status')),
                  recording_sec=dur,review_status=status,excluded_from_behavior=bool(exclude_unassigned and group=='UNASSIGNED'),
                  behavior_status='ok',behavior_issue='')
        if meta['state'].lower() in anesthesia_states:
            base.update(behavior_status='not_applicable_anest',behavior_issue='anesthesia_state_no_rest_qc',
                        rest_sec=np.nan,n_rest_periods=0,mean_rest_period_sec=np.nan,
                        rest_percent_recording=np.nan,rest_fraction_recording=np.nan,
                        recording_duration_available=bool(math.isfinite(dur) and dur>0),n_run_labeled=0)
            stem_rows.append(base)
            continue
        intervals=[]
        try:
            if status not in {'reviewed','complete'}:raise ValueError(f'behavior_not_reviewed:{status}')
            intervals=behavior_intervals(a.get('label_intervals_json'),duration=dur,tol=1/fps+1e-7)
            rest=merge_intervals([(x['t0_sec'],x['t1_sec']) for x in intervals if x['label']=='rest'])
            run=merge_intervals([(x['t0_sec'],x['t1_sec']) for x in intervals if x['label']=='run'])
            for x in intervals:
                if x['label']=='rest' and any(x['t0_sec']<r[1]-1e-9 and x['t1_sec']>r[0]+1e-9 for r in run):
                    raise ValueError('overlapping REST and RUN labels')
            lens=[b-a for a,b in rest]
            rest_sec=float(sum(lens))
            pct=100*rest_sec/dur if math.isfinite(dur) and dur>0 else np.nan
            base.update(rest_sec=rest_sec,n_rest_periods=len(rest),
                        mean_rest_period_sec=float(np.mean(lens)) if lens else np.nan,
                        rest_percent_recording=pct,rest_fraction_recording=pct/100 if math.isfinite(pct) else np.nan,
                        recording_duration_available=bool(math.isfinite(dur) and dur>0),
                        n_run_labeled=len(run))
            for j,(lo,hi) in enumerate(rest):
                period_rows.append(dict(stem=stem,mouse_id=mouse,experimental_group=group,
                                        exp_day=meta['exp_day'],state=meta['state'],
                                        rest_index=j,t0_sec=lo,t1_sec=hi,duration_sec=hi-lo,
                                        excluded_from_behavior=bool(exclude_unassigned and group=='UNASSIGNED')))
            if not math.isfinite(dur):
                base.update(behavior_status='missing_recording_duration',behavior_issue='No raw/processed duration; percent unavailable')
        except Exception as exc:
            base.update(behavior_status='invalid',behavior_issue=str(exc),rest_sec=np.nan,
                        n_rest_periods=0,mean_rest_period_sec=np.nan,rest_percent_recording=np.nan,
                        rest_fraction_recording=np.nan,recording_duration_available=False,n_run_labeled=0)
        stem_rows.append(base)
    stems=pd.DataFrame(stem_rows)
    periods=pd.DataFrame(period_rows,columns=['stem','mouse_id','experimental_group','exp_day','state','rest_index',
                                               't0_sec','t1_sec','duration_sec','excluded_from_behavior'])
    if stems.empty:return dict(stem=stems,periods=periods,mouse_day=pd.DataFrame())
    accepted=stems[(stems['behavior_status']=='ok') & ~stems['excluded_from_behavior']].copy()
    if accepted.empty:return dict(stem=stems,periods=periods,mouse_day=pd.DataFrame())
    keys=['mouse_id','experimental_group','exp_day','state']
    md=(accepted.groupby(keys,dropna=False,as_index=False)
        .agg(n_stems=('stem','nunique'),recording_sec=('recording_sec','sum'),
             rest_sec=('rest_sec','sum'),n_rest_periods=('n_rest_periods','sum')))
    md['rest_percent_recording']=100*md['rest_sec']/md['recording_sec']
    md['mean_rest_period_sec']=md['rest_sec']/md['n_rest_periods'].replace(0,np.nan)
    return dict(stem=stems,periods=periods,mouse_day=md)


def summarize_values(df:pd.DataFrame,by:list[str],metrics=('rest_percent_recording','mean_rest_period_sec')) -> pd.DataFrame:
    """Compute descriptive statistics across mouse-level observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with ``mouse_id`` and the requested metric columns.
    by : list[str]
        Grouping columns; an empty list describes all observations.
    metrics : tuple[str, ...], optional
        Metric columns, typically REST percentage and period duration in s.

    Returns
    -------
    pandas.DataFrame
        Number of mice, valid metric count, mean, sample SD (``ddof=1``),
        SEM, median, quartiles, and extrema for each group.

    Notes
    -----
    NaN values are omitted separately for each metric. SD and SEM require
    at least two nonmissing observations.
    """
    if df.empty:return pd.DataFrame()
    rows=[]
    iterator=df.groupby(by,dropna=False) if by else [((),df)]
    for key,group in iterator:
        key = key if isinstance(key,tuple) else (key,)
        info={k:v for k,v in zip(by,key)}
        info['n_mice_total']=int(group['mouse_id'].nunique())
        for metric in metrics:
            vals=pd.to_numeric(group[metric],errors='coerce').dropna()
            info.update({f'{metric}_{suffix}':val for suffix,val in {
                'n':int(vals.size),'mean':float(vals.mean()) if len(vals) else np.nan,
                'sd':float(vals.std(ddof=1)) if len(vals)>1 else np.nan,
                'sem':float(vals.std(ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else np.nan,
                'median':float(vals.median()) if len(vals) else np.nan,
                'q25':float(vals.quantile(.25)) if len(vals) else np.nan,
                'q75':float(vals.quantile(.75)) if len(vals) else np.nan,
                'min':float(vals.min()) if len(vals) else np.nan,
                'max':float(vals.max()) if len(vals) else np.nan,
            }.items()})
        rows.append(info)
    return pd.DataFrame(rows)


def behavior_comparisons(mouse_day:pd.DataFrame, *,days:tuple|list) -> dict[str,pd.DataFrame]:
    """Build Awake-only descriptive and paired behavior comparisons.

    Parameters
    ----------
    mouse_day : pandas.DataFrame
        Per-mouse/day/state behavior metrics.
    days : tuple or list
        Experimental days to include in descriptive summaries.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Before-PT description, paired Before-PT/Day-PT observations, paired
        difference summary, exclusions from pairs, day and group/day summaries.

    Notes
    -----
    Pairing uses ``mouse_id`` and requires Awake observations at both days.
    Any mouse with Day-PT Iso is excluded from that paired comparison.
    Differences are Day-PT minus Before-PT. Pair selection is not an
    exclusion from all other descriptive analyses.
    """
    if mouse_day.empty:return {k:pd.DataFrame() for k in ['before_description','paired','paired_summary','day_summary','group_day_summary','excluded_from_pair']}
    awake=mouse_day[mouse_day['state'].astype(str).str.lower()=='awake'].copy()
    before=awake[awake['exp_day']=='Before-PT']
    day_pt_states=mouse_day[mouse_day['exp_day']=='Day-PT'].groupby('mouse_id')['state'].agg(list)
    excluded_iso=set(day_pt_states[day_pt_states.map(lambda s:any(str(x).lower()=='iso' for x in s))].index)
    exclusions=pd.DataFrame({'mouse_id':sorted(excluded_iso),'reason':'Day-PT recorded under Iso'})
    # Merge mouse/day only, no merging on group to avoid dropping inconsistent group annotations.
    pair=(before[~before['mouse_id'].isin(excluded_iso)]
          .merge(awake[(awake['exp_day']=='Day-PT') & ~awake['mouse_id'].isin(excluded_iso)],
                 on='mouse_id',suffixes=('_before','_daypt'),validate='one_to_one'))
    if not pair.empty:
        for metric in ['rest_percent_recording','mean_rest_period_sec']:
            pair[f'delta_{metric}']=pair[f'{metric}_daypt']-pair[f'{metric}_before']
    diffs=[]
    for metric in ['rest_percent_recording','mean_rest_period_sec']:
        name=f'delta_{metric}'
        vals=pd.to_numeric(pair[name],errors='coerce').dropna() if name in pair else pd.Series(dtype=float)
        diffs.append(dict(metric=metric,n_paired=len(vals),mean_difference=float(vals.mean()) if len(vals) else np.nan,
                          sd_difference=float(vals.std(ddof=1)) if len(vals)>1 else np.nan,
                          median_difference=float(vals.median()) if len(vals) else np.nan))
    return dict(before_description=summarize_values(before,[]),
                paired=pair,paired_summary=pd.DataFrame(diffs),excluded_from_pair=exclusions,
                day_summary=summarize_values(awake[awake['exp_day'].isin(days)],['exp_day']),
                group_day_summary=summarize_values(awake[awake['exp_day'].isin(days)],['experimental_group','exp_day']))



def rest_retention(behavior:dict[str,pd.DataFrame],*,thresholds:tuple|list) -> pd.DataFrame:
    """Estimate pre-QC availability of continuous REST by duration threshold.

    Parameters
    ----------
    behavior : dict[str, pandas.DataFrame]
        Output from ``summarize_behavior``.
    thresholds : tuple or list
        Minimum continuous REST durations in seconds; defined by the caller.

    Returns
    -------
    pandas.DataFrame
        Overall, per-day and group/day mouse-day retention counts and percent.

    Notes
    -----
    This is based on annotated continuous REST periods, before manual
    ``USE/REJECT/UNCLEAR`` segment QC. Thresholds are not read from YAML here.
    """
    md=behavior['mouse_day'];periods=behavior['periods']
    if md.empty:return pd.DataFrame()
    selected=periods.loc[~periods['excluded_from_behavior']].copy()
    keys=['mouse_id','exp_day','state']
    result=[]
    for thr in thresholds:
        eligible=set(tuple(v) for v in selected.loc[selected['duration_sec']+1e-8>=thr,keys].itertuples(index=False,name=None))
        tmp=md.copy()
        tmp['has_continuous_rest']=tmp[keys].apply(lambda x:tuple(x) in eligible,axis=1)
        tmp['min_rest_sec']=int(thr)
        for group_keys in [[],['exp_day'],['experimental_group','exp_day']]:
            iterator=tmp.groupby(group_keys,dropna=False) if group_keys else [((),tmp)]
            for key,part in iterator:
                key=key if isinstance(key,tuple) else (key,)
                entry={k:v for k,v in zip(group_keys,key)}
                n=len(part);retained=int(part['has_continuous_rest'].sum())
                entry.update(scope='overall' if not group_keys else 'day' if len(group_keys)==1 else 'group_day',
                             min_rest_sec=int(thr),n_mouse_days=n,n_retained=retained,
                             retained_percent=100*retained/n if n else np.nan)
                result.append(entry)
    return pd.DataFrame(result)


def _candidates(stem:str,intervals:list[dict],seconds:float) -> list[dict]:
    """Partition annotated REST parents into nonoverlapping nominal slots.

    Parameters
    ----------
    stem : str
        Recording identifier used in deterministic slot IDs.
    intervals : list[dict]
        Validated behavior intervals, with bounds in seconds.
    seconds : float
        Nominal candidate duration in seconds.

    Returns
    -------
    list[dict]
        Candidate slots with parent/chunk indices and nominal start/end times.

    Notes
    -----
    Only complete consecutive slots are generated; leftover shorter time at
    the end of a REST parent does not form a candidate.
    """
    slots=[]
    for pi,rec in enumerate(intervals):
        if rec['label']!='rest':continue
        p0,p1=rec['t0_sec'],rec['t1_sec']
        for k in range(int(math.floor((p1-p0+1e-9)/seconds))):
            a=p0+k*seconds
            slots.append(dict(slot_id=f'{stem}|parent{pi}|chunk{k}',parent_index=pi,chunk_index=k,
                              parent_t0=p0,parent_t1=p1,t0=a,t1=a+seconds))
    return slots



def select_final_cohort(inspected_index:pd.DataFrame, annotations:pd.DataFrame,
                        review:pd.DataFrame, config:dict, *,
                        apply_analysis_selection:bool=False, seconds:float=60.) -> dict[str,pd.DataFrame]:
    """Reconstruct the manual REST-Review target cohort and diagnostics.

    Parameters
    ----------
    inspected_index : pandas.DataFrame
        Step4 REST-state index with processed-file status.
    annotations : pandas.DataFrame
        Manual behavior annotations.
    review : pandas.DataFrame
        Manual REST segment decisions.
    config : dict
        Uses optional ``rest_qc.anesthesia_states`` (default ``['Iso']``);
        ``analysis_selection`` is used only when explicitly requested.
    apply_analysis_selection : bool, optional
        Whether to also apply global day/mouse/metadata selection (False by
        default to avoid silently omitting reviewed REST recordings).
    seconds : float, optional
        Nominal candidate window duration in seconds, default 60.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``cohort``, row-level ``diagnostics``, and records with reviews outside
        the selected cohort (``reviewed_outside_cohort``).

    Notes
    -----
    Anesthesia states are retained for not-applicable diagnostics. Awake
    stems enter when they have nominal candidates or existing manual reviews,
    with processed data available unless already reviewed. Does not apply
    scientific ``analysis_exclusions`` itself; the notebook does so first.
    """
    selected=inspected_index.copy()
    selection=config.get('analysis_selection',{}) or {}
    if apply_analysis_selection:
        days=selection.get('exp_days')
        if days:selected=selected[selected['exp_day'].astype(str).isin(set(map(str,days)))].copy()
        mice=selection.get('include_mice')
        if mice:selected=selected[selected['mouse_id'].astype(str).isin(set(map(str,mice)))].copy()
        for key,val in (selection.get('filters') or {}).items():
            if key not in selected.columns:raise KeyError(f'analysis_selection.filters column absent: {key}')
            vals=val if isinstance(val,(list,tuple,set)) else [val]
            selected=selected[selected[key].astype(str).isin({str(v) for v in vals})].copy()
        if (selection.get('experimental_groups') and
           bool(selection.get('require_group_assignment',True))):
            selected=selected[selected['experimental_group'].astype(str)!='UNASSIGNED'].copy()
    ann=unique_latest(annotations);reviewed=unique_latest(review)
    anesthesia_states = {text(s).lower() for s in
                         config.get('rest_qc',{}).get('anesthesia_states',['Iso'])}
    eligible_ids=set(selected['stem'].astype(str))
    diag=[];keep=[]
    for _,r in inspected_index.iterrows():
        stem=text(r['stem'])
        eligible_selected=stem in eligible_ids
        row=ann.get(stem)
        n=0;issue=''
        if row is not None:
            try:n=len(_candidates(stem,behavior_intervals(row.get('label_intervals_json')),seconds))
            except Exception as ex:issue='invalid_behavior_intervals:'+str(ex)
        else:issue='missing_behavior_annotation'
        has_review=stem in reviewed
        proc_ok=text(r.get('processed_status'))=='ok'
        is_anest=text(r.get('state')).lower() in anesthesia_states
        active=bool(eligible_selected and (is_anest or ((n>0 or has_review) and (proc_ok or has_review))))
        if active:keep.append(stem)
        reason = ('not_applicable_anest' if is_anest and active else
                  issue or ('in_qc_cohort' if active else
                            'outside_analysis_selection' if not eligible_selected else
                            'no_60s_rest_candidate' if n==0 and not has_review else
                            'missing_processed_before_review'))
        diag.append(dict(stem=stem,mouse_id=text(r.get('mouse_id')),exp_day=text(r.get('exp_day')),
                         state=text(r.get('state')),processed_status=text(r.get('processed_status')),
                         in_analysis_selection=eligible_selected,
                         n_nominal_candidates=n,has_rest_review=has_review,
                         in_final_cohort=active,reason=reason))
    diag=pd.DataFrame(diag)
    cohort=inspected_index[inspected_index['stem'].astype(str).isin(keep)].copy()
    outside=diag[diag['has_rest_review'] & ~diag['in_final_cohort']].copy()
    return dict(cohort=cohort.reset_index(drop=True),diagnostics=diag,
                reviewed_outside_cohort=outside)


def _validate_approved(t0:float,t1:float,slot:dict,*,fps:float,n_frames:int,seconds:float):
    """Validate a manually accepted REST window against its slot and frames.

    Parameters
    ----------
    t0, t1 : float
        Accepted window boundaries in seconds.
    slot : dict
        Current nominal slot and enclosing REST parent.
    fps : float
        Frame rate in frames per second.
    n_frames : int
        Number of frames in the processed Zarr signal.
    seconds : float
        Expected window duration in seconds.

    Raises
    ------
    ValueError
        If times, duration, parent inclusion or frame-count alignment fail.

    Notes
    -----
    Boundary tolerance is approximately one frame (``1/fps`` seconds).
    """
    tol=1/fps+1e-8
    if not (math.isfinite(t0) and math.isfinite(t1) and t0>=0 and t1>t0):raise ValueError('bad_time')
    if abs(t1-t0-seconds)>tol:raise ValueError('window_duration_not_60s')
    if t0<slot['parent_t0']-tol or t1>slot['parent_t1']+tol:raise ValueError('window_outside_current_REST_parent')
    lo,hi=round(t0*fps),round(t1*fps)
    if not(0<=lo<hi<=n_frames):raise ValueError('window_outside_processed_frames')
    if abs(hi-lo-seconds*fps)>1.01:raise ValueError('window_inconsistent_frame_count')


def finalize_rest(inspected_index:pd.DataFrame, annotations:pd.DataFrame, review:pd.DataFrame,
                  config:dict, *, seconds:float=60.) -> dict[str,pd.DataFrame]:
    """Validate manual REST QC and construct authoritative output tables.

    Parameters
    ----------
    inspected_index : pandas.DataFrame
        Eligible Step4 REST-state rows with processed-file inspection results.
    annotations : pandas.DataFrame
        Behavior annotations with REST parent intervals in seconds.
    review : pandas.DataFrame
        Manual REST-Review decisions and optional shadow masks.
    config : dict
        YAML with ``raw.fps``, optional ``rest_qc.anesthesia_states``, and
        global ``path_resolving`` for stored mask paths.
    seconds : float, optional
        Expected REST segment duration in seconds (default 60).

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``rest_segments_final``: validated accepted windows;
        ``rest_stem_status``: review/file status for each selected stem;
        ``rest_shadow_masks_by_stem``: shadow-mask provenance and QC.

    Notes
    -----
    Uses only the manual ``rest_segment_rescue_review.csv`` decisions, never
    legacy primary QC. Accepted intervals are checked against current parent
    annotations, nominal slots, frame count and expected duration.
    Missing processed data receive ``missing_data`` and produce no FC windows.
    Unfinished/invalid QC is kept visible for export checks.
    Anesthesia recordings are classified as not applicable. Scientific
    exclusions must be applied by the caller before finalization.
    """
    if seconds<=0:raise ValueError('seconds must be positive')
    fps=float(config['raw']['fps'])
    ann=unique_latest(annotations)
    qc=unique_latest(review)
    anesthesia_states = {text(s).lower() for s in
                         config.get('rest_qc',{}).get('anesthesia_states',['Iso'])}
    # Paths inside the saved REST Review CSV may refer to the old drive.
    # Apply the same global remapping as Step1-Step4 before checking masks.
    from common import get_path_resolving_config, resolve_flexible_path
    path_cfg = get_path_resolving_config(config)
    remap_kwargs = {k: path_cfg.get(k) for k in ('from_drive','to_drive','from_root','to_root')}
    result=[];statuses=[];shadows=[]
    for _,r in inspected_index.iterrows():
        m=r.to_dict();stem=text(m.get('stem'));a=ann.get(stem);rr=qc.get(stem)
        base={k:text(m.get(k)) for k in ['stem','group_id','ref_stem','mouse_id','experimental_group','exp_day','state','protocol']}
        st={k:base.get(k,'') for k in STATUS_COLUMNS if k in base}
        st.update(processed_status=text(m.get('processed_status')),processed_path=text(m.get('processed_path')),
                  n_frames=m.get('n_frames',np.nan),n_candidates=0,n_reviewed=0,n_accepted=0,
                  n_rejected=0,n_unclear=0,n_overlaps=0,review_status=text(rr.get('review_status')) if rr else '',
                  final_state='invalid',reason='',behavior_rest_sec=np.nan)
        shadow=dict(stem=stem,shadow_mask_path='',shadow_status='unreviewed',source='rest_review',
                    review_status=st['review_status'],n_pixels=0,mask_shape='',reason='')
        local=[]
        # Anesthesia is not behavioral REST. Preserve the stem in diagnostics,
        # with zero fake REST windows and without requiring behavior/Rescue QC.
        if text(m.get('state')).lower() in anesthesia_states:
            st.update(final_state='not_applicable_anest',reason='anesthesia_state_no_rest_qc',
                      review_status='',n_candidates=0,n_reviewed=0,n_accepted=0)
            shadow.update(shadow_status='not_applicable_anest',source='none',
                          review_status='',reason='anesthesia_state_no_rest_qc')
            statuses.append(st);shadows.append(shadow)
            continue
        try:
            if a is None:raise ValueError('missing_behavior_annotation')
            if text(a.get('review_status')).lower() not in {'reviewed','complete'}:
                raise ValueError('behavior_not_reviewed')
            intervals=behavior_intervals(a.get('label_intervals_json'),
                duration=(float(m['n_frames'])/fps if st['processed_status']=='ok' else np.nan),
                tol=1/fps+1e-8)
            st['behavior_rest_sec']=float(sum(b-a for a,b in merge_intervals([
                (x['t0_sec'],x['t1_sec']) for x in intervals if x['label']=='rest'])))
            slots=_candidates(stem,intervals,seconds)
            st['n_candidates']=len(slots)
            if rr is not None:
                shadow['shadow_status']='reviewed_absent' if st['review_status'].lower()=='complete' else 'unreviewed'
                p=text(rr.get('shadow_mask_path'))
                if p:
                    try:
                        pp = resolve_flexible_path(p, **remap_kwargs, must_exist=True)
                    except FileNotFoundError:
                        # Preserve a useful remapped candidate for diagnostics;
                        # do not change the manual CSV or pretend it exists.
                        pp = resolve_flexible_path(p, **remap_kwargs, must_exist=False)
                    # Downstream Step6b/Step7 consume these exported paths, so
                    # export the resolved path, not the old literal CSV value.
                    shadow['shadow_mask_path'] = str(pp)
                    if not pp.is_file():
                        msg = 'shadow_mask_missing:' + p
                        if str(pp) != p:
                            msg += '; resolved_candidate:' + str(pp)
                        shadow.update(shadow_status='missing',reason=msg)
                    else:
                        arr=np.load(pp,allow_pickle=False)
                        if arr.ndim!=2:
                            shadow.update(shadow_status='invalid_shape',reason='shadow_mask_not_2d')
                        else:
                            if np.isfinite(m.get('height',np.nan)) and np.isfinite(m.get('width',np.nan)):
                                expected=(int(m['height']),int(m['width']))
                                if arr.shape!=expected:
                                    shadow.update(shadow_status='invalid_shape',reason=f'shadow_mask_shape_{arr.shape}_expected_{expected}')
                            if shadow['shadow_status'] not in ('invalid_shape',):
                                shadow.update(shadow_status='reviewed_present' if np.any(arr) else 'reviewed_absent',
                                              n_pixels=int(np.count_nonzero(arr)),mask_shape=str(tuple(arr.shape)))
            # Check manually saved slot decisions even when the processed Zarr
            # disappeared: useful diagnostics and counts survive in the report.
            if slots:
                if rr is None:
                    st.update(final_state='missing_review',reason='missing_rest_review_row_no_fallback')
                else:
                    decisions=_json_list(rr.get('segment_decisions_json'))
                    by_id={}
                    expected={s['slot_id']:s for s in slots}
                    for d in decisions:
                        sid=text(d.get('slot_id'))
                        if sid not in expected:raise ValueError('unknown_or_outdated_slot_id:'+sid)
                        if sid in by_id:raise ValueError('duplicate_slot_id:'+sid)
                        by_id[sid]=d
                    st['n_reviewed']=len(by_id)
                    for sid,slot in expected.items():
                        if sid not in by_id:continue
                        d=by_id[sid]
                        if (int(d.get('parent_rest_index',-1))!=slot['parent_index']
                          or int(d.get('chunk_index_in_parent',-1))!=slot['chunk_index']):
                            raise ValueError('outdated_parent_or_chunk')
                        for dk,sk in [('parent_rest_t0_sec','parent_t0'),('parent_rest_t1_sec','parent_t1'),
                                      ('nominal_t0_sec','t0'),('nominal_t1_sec','t1')]:
                            val=float(d.get(dk,float('nan')))
                            if not math.isfinite(val) or abs(val-slot[sk])>1/fps:
                                raise ValueError('outdated_'+dk)
                        decision=text(d.get('decision')).lower()
                        if decision not in {'use','reject','unclear'}:raise ValueError('unknown_decision')
                        st['n_accepted' if decision=='use' else ('n_rejected' if decision=='reject' else 'n_unclear')]+=1
                        if decision=='use':
                            lo,hi=float(d['t0_sec']),float(d['t1_sec'])
                            if st['processed_status']=='ok':
                                _validate_approved(lo,hi,slot,fps=fps,n_frames=int(m['n_frames']),seconds=seconds)
                            else:
                                _validate_without_frames(lo,hi,slot,fps=fps,seconds=seconds)
                            local.append((slot,lo,hi))
                    if st['review_status'].lower()!='complete' or set(by_id)!=set(expected):
                        st.update(final_state='incomplete_review',reason='rest_review_not_complete_or_missing_slots')
                    else:
                        accepted=_json_list(rr.get('label_intervals_json'))
                        recorded=sorted((round(float(v['t0_sec']),6),round(float(v['t1_sec']),6)) for v in accepted)
                        approved=sorted((round(x,6),round(y,6)) for _,x,y in local)
                        if recorded!=approved:raise ValueError('accepted_intervals_disagree_with_decisions')
                        st.update(final_state='complete',reason='ok')
            else:
                if rr is not None and _json_list(rr.get('segment_decisions_json')):
                    raise ValueError('rest_review_has_decisions_but_no_current_60s_candidates')
                st.update(final_state='complete',reason='no_60s_candidates')

            if st['processed_status']!='ok' and st['final_state'] in {'complete','missing_review'}:
                original_reason=st['reason']
                st.update(final_state='missing_data',reason='processed_unavailable:'+st['processed_status']+
                          (('; '+original_reason) if original_reason!='ok' else ''))

            if shadow['shadow_status'] in {'missing','invalid_shape'} and st['final_state']=='complete':
                st.update(final_state='invalid',reason=shadow['reason'])

            if st['final_state']=='complete':
                local.sort(key=lambda z:z[1])
                st['n_overlaps']=sum(int(y[1]<x[2]-1/fps) for x,y in zip(local,local[1:]))
                for idx,(slot,lo,hi) in enumerate(local,1):
                    result.append({**base,'segment_number_source':slot['chunk_index']+1,
                                   't0_sec':lo,'t1_sec':hi,'duration_sec':hi-lo,
                                   'source':'rest_review','shadow_mask_path':shadow['shadow_mask_path'],
                                   'segment_number':idx,'duration_matches_expected':True,
                                   'slot_id':slot['slot_id'],'parent_rest_index':slot['parent_index'],
                                   'nominal_t0_sec':slot['t0']})
        except Exception as exc:
            st.update(final_state='invalid',reason=str(exc))
            if shadow['shadow_status'] not in {'reviewed_present','reviewed_absent'}:
                shadow['reason']=str(exc)
        statuses.append(st);shadows.append(shadow)
    return {'rest_segments_final':pd.DataFrame(result,columns=FINAL_COLUMNS),
            'rest_stem_status':pd.DataFrame(statuses,columns=STATUS_COLUMNS),
            'rest_shadow_masks_by_stem':pd.DataFrame(shadows,columns=SHADOW_COLUMNS)}


def _validate_without_frames(t0:float,t1:float,slot:dict,*,fps:float,seconds:float):
    """Validate accepted-window times when processed frames are unavailable.

    Parameters
    ----------
    t0, t1 : float
        Window boundaries in seconds.
    slot : dict
        Candidate slot and REST parent bounds, in seconds.
    fps : float
        Frame rate used to set boundary tolerance.
    seconds : float
        Required duration, in seconds.

    Raises
    ------
    ValueError
        If interval timing, duration or REST-parent containment is invalid.
    """
    tol=1/fps+1e-8
    if not (math.isfinite(t0) and math.isfinite(t1) and t0>=0 and t1>t0):raise ValueError('bad_time')
    if abs(t1-t0-seconds)>tol:raise ValueError('window_duration_not_60s')
    if t0<slot['parent_t0']-tol or t1>slot['parent_t1']+tol:raise ValueError('window_outside_current_REST_parent')


def qc_statistics(final:dict[str,pd.DataFrame], *, exclude_unassigned:bool=False) -> dict[str,pd.DataFrame]:
    """Summarize post-review REST retention for eligible recordings.

    Parameters
    ----------
    final : dict[str, pandas.DataFrame]
        Finalization tables, including ``rest_stem_status``. Caller must
        exclude scientific ``analysis_excluded`` statuses from denominators.
    exclude_unassigned : bool, optional
        Exclude ``UNASSIGNED`` from group and mouse-day summaries, while
        retaining them in stem-level diagnostics.

    Returns
    -------
    dict[str, pandas.DataFrame]
        ``stem``: per-recording usable counts; ``mouse_day``: aggregated
        recording counts; ``group_day``: retention and dropout counts.

    Notes
    -----
    A recording is kept for FC only if final_state is ``complete`` and it has
    at least one accepted segment. Anesthesia is marked not applicable,
    rather than counted as dropout.
    """
    s=final['rest_stem_status'].copy()
    if s.empty:return {'stem':s,'mouse_day':pd.DataFrame(),'group_day':pd.DataFrame()}
    s['kept_for_fc']=s['n_accepted'].gt(0) & s['final_state'].eq('complete')
    s['is_not_applicable_anest']=s['final_state'].eq('not_applicable_anest')
    s['n_dropped_stem']=~s['kept_for_fc'] & ~s['is_not_applicable_anest']
    s['n_usable_60s']=s['n_accepted'].where(s['final_state'].eq('complete'),0)
    # Diagnostic stem table preserves UNASSIGNED; optional filter affects group summaries.
    analysis_s=s[s['experimental_group'].ne('UNASSIGNED')].copy() if exclude_unassigned else s
    # Mouse-day rows remain even with zero accepted segments.
    keys=['experimental_group','exp_day','state','mouse_id']
    mouse=analysis_s.groupby(keys,dropna=False,as_index=False).agg(n_stems=('stem','nunique'),
        n_candidates=('n_candidates','sum'),n_reviewed=('n_reviewed','sum'),
        n_use_decisions=('n_accepted','sum'),n_usable_60s=('n_usable_60s','sum'),
        n_reject=('n_rejected','sum'),n_unclear=('n_unclear','sum'),
        n_stems_with_usable=('kept_for_fc','sum'),
        n_not_applicable_anest=('is_not_applicable_anest','sum'))
    mouse['kept_for_fc']=mouse['n_stems_with_usable']>0
    mouse['rest_qc_applicable']=mouse['n_stems']>mouse['n_not_applicable_anest']
    group=mouse.groupby(['experimental_group','exp_day','state'],dropna=False,as_index=False).agg(
        n_mice=('mouse_id','nunique'),n_mice_rest_qc_applicable=('rest_qc_applicable','sum'),
        n_mice_retained=('kept_for_fc','sum'),
        n_accepted=('n_usable_60s','sum'),n_use_decisions=('n_use_decisions','sum'),
        n_rejected=('n_reject','sum'),n_unclear=('n_unclear','sum'))
    group['n_mice_dropped']=group['n_mice_rest_qc_applicable']-group['n_mice_retained']
    return dict(stem=s,mouse_day=mouse,group_day=group)


def export_authoritative(final:dict[str,pd.DataFrame],root:str|Path,*,save:bool=False)->dict[str,Path]:
    """Write validated REST outputs as frozen CSV tables.

    Parameters
    ----------
    final : dict[str, pandas.DataFrame]
        Tables from ``finalize_rest``, potentially augmented by the notebook
        with scientific-exclusion audit fields.
    root : str or pathlib.Path
        Destination directory.
    save : bool, optional
        False returns without writing; True saves all three CSV files.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths to saved CSV files, or empty mapping in preview mode.

    Raises
    ------
    ValueError
        If QC is invalid/incomplete or required shadow masks are unusable.

    Notes
    -----
    Validates before writing. Uses temporary ``.csv.tmp`` files followed by
    replacement of the three destination CSV files. A missing processed Zarr
    alone is not a blocking QC error; no FC window is exported for that stem.
    """
    state=final['rest_stem_status']
    bad=state[state['final_state'].isin({'invalid','incomplete_review','missing_review'})]
    if not bad.empty:
        raise ValueError('Cannot freeze REST: QC requires review: '+repr(bad[['stem','reason']].to_dict('records')[:25]))
    shadows=final['rest_shadow_masks_by_stem']
    bad_shadow=shadows[shadows['shadow_status'].isin({'missing','invalid_shape'})]
    if not bad_shadow.empty:
        raise ValueError('Cannot freeze REST: missing/invalid shadow mask: '+
                         repr(bad_shadow[['stem','shadow_status','reason']].to_dict('records')[:25]))
    if not save:return {}
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    paths={}
    for key in ('rest_segments_final','rest_stem_status','rest_shadow_masks_by_stem'):
        dst=root/f'{key}.csv';tmp=dst.with_suffix('.csv.tmp')
        final[key].to_csv(tmp,index=False,encoding='utf-8-sig')
        paths[key]=dst
    for key,dst in paths.items():
        dst.with_suffix('.csv.tmp').replace(dst)
    return paths
