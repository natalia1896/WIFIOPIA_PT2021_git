"""Step Spontaneous Hemodynamics: independent of Step7 and Step6c.

Configuration comes ONLY from step_spontaneous_hemodynamics in the main project YAML.
ROI/whole brain PSD = mean pixelwise PSD; correlation traces use genuine
Step6b bilateral homotopically paired canonical-pixel masks.

Article outcomes: old v1.10 Hann periodogram PSD (including mouse curves),
existing pixel-wise normalized bands 0.05-0.10 / 0.11-0.20 / 0.08-0.40 Hz,
and six pairwise correlations of MOp/MOs/SS/RSP for left/right/bilateral traces.
Scientific spatial QC is read exclusively from saved Step6b tables and masks.
"""
from __future__ import annotations
import argparse
import copy
import itertools
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cv2
from skimage.transform import warp
import step6b_fc_geometry_coverage as s6b
from spontaneous_hemodynamics_core import (
    load_geometry,prepare_segment,roi_trace,roi_pixel_columns,pixelwise_psd_low_frequency,
    homotopic_pairable_fc_from_step6b,compute_psd,relative_psd,band_fractions,
    pixelwise_relative_band_power_maps,filter_for_correlation,trim_correlation_window,pairwise_correlations,
    integrate,pearson_fisher)
from spontaneous_hemodynamics_maps import (load_allen_atlas_safe,
    stem_fc_map_to_reference,reference_map_to_canonical,_nanmean_stack)


DEFAULT={
    'folder_name':'step_spontaneous_hemodynamics', 'signal':'dcHbT',
    'rest_variant':'with_rest','mask_set':'daypt_anesthesia',
    'rois':['MOp','MOs','SS','RSP'],
    'segments':{'primary_sec':60.0,'duration_tolerance_sec':.75,
                'rest_segments_final_csv':None},
    'spatial':{'scale':.5,'signal_resize':'mask_aware_area','min_valid_fraction':.5},
    'preprocessing':{'detrend_enabled':True,'detrend_type':'linear','variance_epsilon':1e-12},
    'spectral':{'method':'periodogram','window':'hann',
                'reference_band_hz':[.05,.5],
                'bands':{'0p05_0p10':[.05,.10],'0p11_0p20':[.11,.20],'0p08_0p40':[.08,.40]},
                'max_saved_frequency_hz':.5,'min_roi_pixels_fc':5,
                'pixel_chunk_size':2048},
    'relative_power_maps':{'enabled':True,'bands':{'0p05_0p10':[.05,.10],
                   '0p11_0p20':[.11,.20],'0p08_0p40':[.08,.40]},
                   'reference_band_hz':[.05,.5], 'pixel_chunk_size':2048,
                   'min_group_n_per_pixel':2,'save_difference_map':True},
    'correlations':{'enabled':True,'filter_type':'bandpass',
                    'band_hz':[.08,.4],'lowpass_high_hz':.4,
                    'order':4,'synthetic_odd_pad_sec':30.0,
                    'trim_left_sec':0.0,'trim_right_sec':0.0},
    'output':{'overwrite':False,'save_segment_spectra_long':True},
    'batch':{'continue_on_error':True},
}


def merge(d,add):
    """Recursively merge nested configuration dictionaries in place.
    
        Parameters
        ----------
        d : dict
            Destination mapping, modified in place.
        add : dict or None
            Overrides; nested mappings are merged and other values deep-copied.
    
        Returns
        -------
        dict
            The same destination mapping after applying overrides.
    """
    for k,v in (add or {}).items():
        if isinstance(v,dict) and isinstance(d.get(k),dict):merge(d[k],v)
        else:d[k]=copy.deepcopy(v)
    return d


def load_config(project_path):
    """Load and validate the spontaneous-hemodynamics project settings.
    
        Parameters
        ----------
        project_path : str or pathlib.Path
            Main project YAML path.
    
        Returns
        -------
        tuple[dict, dict]
            Full project configuration and the resolved ``step_spontaneous_hemodynamics``
            block, with code defaults merged beneath YAML overrides.
    
        Notes
        -----
        The selected ``rest_variant`` is also applied to
        ``step6b_fc_geometry.active_rest_variant`` in the in-memory configuration.
        Correlation filtering supports ``bandpass`` and ``lowpass``. In the latter
        mode, ``lowpass_high_hz`` is the upper cutoff and ``band_hz`` does not
        define the actual passband. No project YAML is written.
    """
    main=s6b.load_config(project_path)
    cfg=copy.deepcopy(DEFAULT)
    merge(cfg,(main.get('step_spontaneous_hemodynamics') or {}))
    variant=str(cfg['rest_variant'])
    if variant not in ('with_rest','without_rest'):raise ValueError(f'Invalid rest_variant {variant}')
    if cfg['signal']!='dcHbT':
        # Explicit signal variant, never silently substitute HbT/GSR.
        warnings.warn(f'Configured signal is {cfg["signal"]}, not dcHbT')
    corr = cfg['correlations']
    mode = str(corr.get('filter_type', 'bandpass')).lower().strip()
    if mode not in ('bandpass', 'lowpass'):
        raise ValueError(f'correlations.filter_type must be bandpass or lowpass; got {mode!r}')
    corr['filter_type'] = mode
    left = float(corr.get('trim_left_sec', 0.0)); right = float(corr.get('trim_right_sec', 0.0))
    sec = float(cfg['segments']['primary_sec'])
    if not (np.isfinite(left) and np.isfinite(right) and left >= 0 and right >= 0
            and left + right < sec):
        raise ValueError('correlation trim must be nonnegative and leave part of each segment')
    if not (np.isfinite(float(corr['synthetic_odd_pad_sec'])) and
            float(corr['synthetic_odd_pad_sec']) >= 0):
        raise ValueError('correlations.synthetic_odd_pad_sec must be nonnegative')
    band = corr['band_hz']
    if len(band) != 2 or not (0 < float(band[0]) < float(band[1])):
        raise ValueError('correlations.band_hz must contain increasing positive bounds')
    if not (0 < float(corr['lowpass_high_hz'])):
        raise ValueError('correlations.lowpass_high_hz must be positive')
    main.setdefault('step6b_fc_geometry',{})['active_rest_variant']=variant
    main['step_spontaneous_hemodynamics']=cfg
    return main,cfg


def build_index(project_path,mice=None,stems=None):
    """Select Step6b-eligible recordings for spontaneous-hemodynamics analysis.
    
        Parameters
        ----------
        project_path : str or pathlib.Path
            Path to the main YAML configuration.
        mice, stems : sequence of str or None, optional
            Optional mouse or recording-stem restrictions after Step6b selection.
    
        Returns
        -------
        tuple
            Recording index, project configuration, spontaneous settings, Step6
            paths, Step6b paths, and saved Step6b selection/eligibility tables.
    
        Raises
        ------
        RuntimeError
            If no matching Step6b selection is available.
        ValueError
            If a selected stem cannot be uniquely matched or has an unsupported
            ``state`` (only Awake and Iso are accepted).
    """
    config,cfg=load_config(project_path)
    step6,_,paths=s6b.build_step6_index(project_path,include_bad=True,filters={},require_response_roi=False)
    # Config passed to the mapping functions must carry the selected REST branch.
    paths6b=s6b.build_step6b_paths(config,paths)
    tables=s6b.load_step6b_tables(project_path,rest_variant=cfg['rest_variant'])
    select=tables.get('selection',pd.DataFrame())
    if select.empty:raise RuntimeError(f'Step6b selection missing: {paths6b["selection_csv"]}')
    sel=select[(select['mask_set'].astype(str)==cfg['mask_set']) &
               (select['status'].astype(str)=='ok')].copy()
    if mice:sel=sel[sel.mouse_id.astype(str).isin(set(mice))]
    if stems:sel=sel[sel.stem.astype(str).isin(set(stems))]
    if sel.empty:raise RuntimeError('No selected stems for the requested Step6b variant/mask_set')
    rows=[]
    for _,sr in sel.iterrows():
        hit=step6[(step6.stem.astype(str)==str(sr['stem'])) &
                  (step6.group_id.astype(str)==str(sr['group_id']))]
        if len(hit)!=1:raise ValueError(f'Step6 index nonunique/missing {sr["stem"]}')
        row=hit.iloc[0].to_dict()
        row.update({k:sr.get(k) for k in ('mask_set','slot','experimental_group','status')})
        state=str(row.get('state','')).strip().lower()
        if state not in ('awake','iso'):
            raise ValueError(f'Stem {row["stem"]}: unknown state {row.get("state")}')
        row['spontaneous_condition']='Awake' if state=='awake' else 'Iso'
        row['spontaneous_comparison']=str(cfg['mask_set'])
        rows.append(row)
    idx=pd.DataFrame(rows).sort_values(['spontaneous_condition','mouse_id','stem']).reset_index(drop=True)
    return idx,config,cfg,paths,paths6b,tables


def _rest_csv_path(cfg,paths):
    """Resolve the final REST-segment CSV path without generating it.
    
        An explicit ``segments.rest_segments_final_csv`` takes precedence over
        ``<out_root>/behavior_review/rest_segments_final.csv``.
    """
    raw=cfg['segments'].get('rest_segments_final_csv')
    if raw:return Path(raw)
    return Path(paths['out_root'])/'behavior_review'/'rest_segments_final.csv'


def accepted_rest_windows(index,cfg,paths):
    """Read accepted Awake REST windows from the existing Finalizer CSV.
    
        Parameters
        ----------
        index : pandas.DataFrame
            Selected analysis recordings including ``spontaneous_condition``.
        cfg, paths : dict
            Resolved spontaneous settings and project paths.
    
        Returns
        -------
        pandas.DataFrame
            Accepted, non-overlapping Awake windows matching ``primary_sec``
            within ``duration_tolerance_sec``. ``without_rest`` yields an empty
            table; Iso recording windows are generated separately.
    
        Raises
        ------
        FileNotFoundError
            If the final CSV is required but missing.
        KeyError
            If required timing or identity columns are absent.
        ValueError
            If accepted windows overlap for a recording stem.
    
        Notes
        -----
        This function never invokes REST Finalizer and does not recheck an
        additional status column. It trusts membership in the final CSV.
    """
    if cfg['rest_variant']=='without_rest':return pd.DataFrame(columns=['stem','t0_sec','t1_sec','source'])
    path=_rest_csv_path(cfg,paths)
    if not path.exists():raise FileNotFoundError(f'Final REST CSV absent (will NOT auto-build): {path}')
    df=pd.read_csv(path)
    required={'stem','state','t0_sec','t1_sec'}
    if required-set(df):raise KeyError(f'Final REST columns missing: {required-set(df)}')
    awake=set(index.loc[index.spontaneous_condition=='Awake','stem'].astype(str))
    d=df[(df.stem.astype(str).isin(awake)) & (df.state.astype(str).str.lower()=='awake')].copy()
    d['t0_sec']=pd.to_numeric(d['t0_sec'],errors='raise')
    d['t1_sec']=pd.to_numeric(d['t1_sec'],errors='raise')
    sec=float(cfg['segments']['primary_sec']);tol=float(cfg['segments']['duration_tolerance_sec'])
    duration=d.t1_sec-d.t0_sec
    d=d[(duration-sec).abs()<=tol].copy()
    # Reject overlapping/duplicate accepted segments for a stem, not double-count.
    for stem,g in d.sort_values('t0_sec').groupby('stem'):
        if np.any(g.t0_sec.to_numpy()[1:] < g.t1_sec.to_numpy()[:-1]-1e-6):
            raise ValueError(f'Overlapping final REST windows for {stem}')
    return d.sort_values(['stem','t0_sec']).reset_index(drop=True)


def windows_for_stem(row,nframes,fps,rest,cfg):
    """Generate complete, non-overlapping analysis windows for a recording.
    
        Parameters
        ----------
        row : mapping
            Recording metadata including ``stem`` and ``spontaneous_condition``.
        nframes : int
            Number of frames in the processed signal.
        fps : float
            Temporal sampling rate in frames per second.
        rest : pandas.DataFrame
            Accepted Awake REST intervals, in seconds.
        cfg : dict
            Resolved spontaneous settings.
    
        Returns
        -------
        list[tuple[float, float, str]]
            ``(start_sec, end_sec, source)`` triples. ``with_rest`` selects saved
            intervals only for Awake; Iso and ``without_rest`` use complete
            non-overlapping full-recording windows.
    
        Notes
        -----
        The default window duration is 60 s but the implementation reads
        ``segments.primary_sec`` from the configuration.
    """
    sec=float(cfg['segments']['primary_sec']);nwin=int(round(sec*fps))
    if nwin<=0:raise ValueError('Invalid segment fps/duration')
    if cfg['rest_variant']=='with_rest' and row['spontaneous_condition']=='Awake':
        if rest.empty:return []
        subset=rest[rest.stem.astype(str)==str(row['stem'])]
        return [(float(r.t0_sec),float(r.t1_sec),'accepted_final_rest') for r in subset.itertuples()]
    # Full recording, complete nonoverlapping 60-second windows, no REST for Iso.
    count=int(nframes)//nwin
    return [(i*nwin/fps,(i+1)*nwin/fps,'full_recording_nonoverlap') for i in range(count)]


def roi_id_map(tables,rois):
    """Map requested ROI acronyms to unique saved Step6b structure IDs.
    
        Uses ``broad_structure_id`` when present, otherwise
        ``analysis_structure_id``. Missing or ambiguous IDs raise ``ValueError``.
    """
    defs=tables.get('analysis_roi_definitions',pd.DataFrame())
    if defs.empty:raise RuntimeError('Step6b ROI definitions missing')
    namecol='analysis_roi_acronym';idcol='broad_structure_id' if 'broad_structure_id' in defs else 'analysis_structure_id'
    result={}
    for roi in rois:
        hit=defs[defs[namecol].astype(str)==roi]
        ids=pd.to_numeric(hit[idcol],errors='coerce').dropna().astype(int).unique()
        if len(ids)!=1:raise ValueError(f'ROI {roi} missing or ambiguous Step6b id: {ids}')
        result[roi]=int(ids[0])
    return result


def truth(val):"""Interpret text and numeric values as Boolean flags.   Only case-insensitive ``true``, ``1``, ``yes`` and ``y`` count as true.""";return str(val).strip().lower() in ('true','1','yes','y')


def selected_side_pass(tables,row,roi,side):
    """Check both cohort-wide and stem-specific Step6b ROI eligibility.
    
        Parameters
        ----------
        tables : dict[str, pandas.DataFrame]
            Saved Step6b tables including ``selected_rois`` and
            ``global_stem_usable_coverage``.
        row : mapping
            Recording identifiers (``mask_set``, ``group_id``, ``stem``).
        roi : str
            ROI acronym.
        side : str
            ``left`` or ``right``.
    
        Returns
        -------
        tuple[bool, str]
            Eligibility flag and machine-readable status. Missing or repeated
            cohort/stem rows fail closed rather than implying PASS.
    """
    selected=tables.get('selected_rois',pd.DataFrame())
    stems=tables.get('global_stem_usable_coverage',pd.DataFrame())
    key=f'{roi}_{side}'
    if selected.empty or stems.empty: return False,'missing_step6b_tables'
    s=selected[(selected.mask_set.astype(str)==str(row['mask_set'])) &
               (selected.analysis.astype(str)=='global') & (selected.roi_key.astype(str)==key)]
    if len(s)!=1:return False,'missing_or_duplicate_cohort_roi'
    if not truth(s.iloc[0].get('include',False)):return False,'cohort_ROI_FAIL'
    # Stem table is side-specific and must match the concrete physical stem.
    st=stems[(stems.mask_set.astype(str)==str(row['mask_set'])) &
             (stems.group_id.astype(str)==str(row['group_id'])) &
             (stems.stem.astype(str)==str(row['stem'])) &
             (stems.roi_key.astype(str)==key)]
    if len(st)!=1:return False,'missing_or_duplicate_stem_roi'
    if not truth(st.iloc[0].get('global_stem_pass',False)):return False,'stem_ROI_FAIL'
    return True,'PASS'


def _zarr_open(row,cfg):
    """Open the selected processed Zarr variable as ``(time, y, x)``.
    
        Returns
        -------
        tuple[xarray.Dataset, xarray.DataArray, pathlib.Path]
            Open dataset, configured signal variable, and store directory.
    
        Raises
        ------
        FileNotFoundError
            If the processed Zarr directory is absent.
        KeyError
            If ``cfg['signal']`` is not a variable in the store.
    
        Notes
        -----
        Callers must close the returned dataset. Chunked opening falls back to
        non-chunked access when the first attempt fails.
    """
    path=Path(str(row.get('processed_zarr_path_resolved','')))
    if not path.is_dir():raise FileNotFoundError(f'Processed Zarr absent: {path}')
    try:ds=xr.open_zarr(path,chunks='auto',consolidated=False)
    except Exception:ds=xr.open_zarr(path,chunks=None,consolidated=False)
    name=str(cfg['signal'])
    if name not in ds.data_vars:
        ds.close();raise KeyError(f'Configured signal {name} absent in {path}')
    return ds,ds[name].transpose('time','y','x'),path


def _to_canonical(fc_map,row,geo,config,paths,paths6b,atlas):
    """Reproject a two-dimensional FC map through reference to Allen space.
    
        Returns
        -------
        tuple[numpy.ndarray, dict]
            Canonical map and metadata describing the stem-to-reference transform.
    """
    ref,meta=stem_fc_map_to_reference(fc_map,row,geo,config,paths,paths6b)
    can=reference_map_to_canonical(ref,row,config,paths,paths6b,atlas)
    return can,meta


def _output_paths(cfg,paths):
    """Construct output directories for the selected REST branch.
    
        Directory paths are derived from ``folder_name`` and ``rest_variant``;
        the function does not create the directories.
    """
    root=Path(paths['out_root'])/str(cfg['folder_name'])/str(cfg['rest_variant'])
    p={'root':root,'tables':root/'tables','figures':root/'figures',
       'maps':root/'maps'/'relative_power','mouse_maps':root/'maps'/'relative_power'/'mouse_canonical',
       'group_maps':root/'maps'/'relative_power'/'group_canonical'}
    return p


def preflight(project_config_path,mice=None,stems=None):
    """Inspect Step6b ROI eligibility and accepted REST input without processing Zarr.
    
        Parameters
        ----------
        project_config_path : str or pathlib.Path
            Main YAML configuration path.
        mice, stems : sequence of str or None, optional
            Optional inclusion filters.
    
        Returns
        -------
        dict
            ``index``, per-stem/ROI ``eligibility``, accepted ``rest60`` intervals,
            full ``config``, output ``paths``, ``s6b_paths``, and saved Step6b
            ``tables``.
    
        Notes
        -----
        PASS at this stage checks Step6b cohort and stem criteria, not whether
        enough finite functional pixels exist in each later 60-s segment.
    """
    idx,config,cfg,paths,paths6b,tables=build_index(project_config_path,mice,stems)
    rest=accepted_rest_windows(idx,cfg,paths)
    rois=cfg['rois'];rows=[]
    for _,r in idx.iterrows():
        rec={'mouse_id':r['mouse_id'],'stem':r['stem'],'state':r['spontaneous_condition'],
             'rest_variant':cfg['rest_variant'], 'n_final_rest':int((rest.stem.astype(str)==str(r['stem'])).sum()) if not rest.empty else 0}
        for roi in rois:
            for side in ('left','right'):
                ok,reason=selected_side_pass(tables,r,roi,side)
                rec[f'{roi}_{side}']='PASS' if ok else reason
        rows.append(rec)
    return dict(index=idx,eligibility=pd.DataFrame(rows),rest60=rest,config=config,paths=_output_paths(cfg,paths),
                s6b_paths=paths6b,tables=tables)


def _group_mean_spectra(mouse_df):
    """Aggregate mouse PSD values by condition, ROI, side, and frequency.
    
        Returns
        -------
        pandas.DataFrame
            Per-frequency ``mean``, sample ``sd``, ``n`` and ``sem`` across
            mouse-level rows, or an empty table when no rows are supplied.
    """
    if mouse_df.empty:return pd.DataFrame()
    keys=['rest_variant','state','roi','hemisphere','frequency_hz','psd_kind']
    tmp=mouse_df.groupby(keys,dropna=False).agg(mean=('value','mean'),sd=('value','std'),
                                                  n=('value','count')).reset_index()
    tmp['sem']=tmp['sd']/np.sqrt(tmp['n'])
    return tmp


def plot_psd(mouse_psd,filename,kind='relative_psd'):
    """Plot absolute or relative pixel-mean PSD for individual mice and groups.
    
        Parameters
        ----------
        mouse_psd : pandas.DataFrame
            Mouse-level spectra indexed by ROI, hemisphere, condition and Hz.
        filename : pathlib.Path
            Destination image path; parent directories are created as needed.
        kind : str, optional
            ``relative_psd`` or ``absolute_psd``.
    
        Returns
        -------
        bool
            True when a plot was written; False when no matching PSD exists.
    
        Notes
        -----
        Individual mice are shown as thin lines. Condition averages and SEM use
        mice as the independent units; ``whole_brain`` is bilateral only.
    """
    d=mouse_psd[mouse_psd.psd_kind==kind]
    if d.empty:return False
    rois=[x for x in ('whole_brain','MOp','MOs','SS','RSP') if x in set(d.roi)]
    hemis=['left','right','bilateral']
    fig,axs=plt.subplots(len(rois),3,figsize=(14,2.9*len(rois)),squeeze=False)
    for i,roi in enumerate(rois):
        for j,hem in enumerate(hemis):
            ax=axs[i,j];sub=d[(d.roi==roi)&(d.hemisphere==hem)]
            if roi=='whole_brain' and hem!='bilateral':
                ax.set_axis_off()
                continue
            for state,g in sub.groupby('state'):
                color={'Awake':'#1877a5','Iso':'#cf7726'}.get(state,'#777777')
                for mouse,gm in g.groupby('mouse_id'):
                    gm=gm.sort_values('frequency_hz')
                    ax.plot(gm.frequency_hz,gm.value,color=color,alpha=.19,lw=.7)
                piv=g.pivot_table(index='mouse_id',columns='frequency_hz',values='value',aggfunc='mean')
                x=piv.columns.to_numpy(float);yy=piv.to_numpy(float);mean=np.nanmean(yy,axis=0)
                ax.plot(x,mean,color=color,lw=2,label=f'{state} (n={len(piv)})')
                if len(piv)>1:
                    sem=np.nanstd(yy,axis=0,ddof=1)/np.sqrt(len(piv))
                    ax.fill_between(x,mean-sem,mean+sem,color=color,alpha=.13)
            ax.axvspan(.05,.10,color='#d7eaf8',alpha=.35)
            ax.axvspan(.11,.20,color='#f6e4d2',alpha=.35)
            ax.set_title(f'{roi} | {hem}');ax.set_xlim(0,.5);ax.set_xlabel('Frequency, Hz')
            ax.set_ylabel('relative PSD (1/Hz)' if kind=='relative_psd' else 'absolute PSD (Hb²/Hz)')
            ax.legend(fontsize=7,loc='upper right') if not sub.empty else None
    fig.suptitle(f'{kind} · individual mouse traces and group mean ± SEM')
    fig.tight_layout();filename.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filename,dpi=165,bbox_inches='tight');plt.close(fig);return True


def plot_correlations(mouse_corr,filename):
    """Plot six ROI-correlation matrices from mouse-level Fisher z values.
    
        The two conditions (Awake, Iso) are shown for ``left``, ``right`` and
        ``bilateral_trace_mean``. Group values are averaged in Fisher-z space and
        back-transformed using ``tanh`` to the displayed Pearson-r scale.
    
        Returns
        -------
        bool
            Whether an image was created.
    """
    if mouse_corr.empty:return False
    modes=['left','right','bilateral_trace_mean'];states=['Awake','Iso'];rois=['MOp','MOs','SS','RSP']
    fig,axs=plt.subplots(2,3,figsize=(13,8),squeeze=False)
    last=None
    for i,state in enumerate(states):
        for j,mode in enumerate(modes):
            ax=axs[i,j];tmp=mouse_corr[(mouse_corr.state==state)&(mouse_corr.variant==mode)]
            matrix=np.full((4,4),np.nan)
            # A diagonal of 1 is valid only when this ROI has real observed pairs.
            for roi in rois:
                available=tmp[(tmp.roi_a==roi)|(tmp.roi_b==roi)]
                if pd.to_numeric(available.get('fisher_z',pd.Series(dtype=float)),errors='coerce').notna().any():
                    matrix[rois.index(roi),rois.index(roi)]=1.0
            for a,b in itertools.combinations(rois,2):
                g=tmp[(tmp.roi_a==a)&(tmp.roi_b==b)]
                z=pd.to_numeric(g.fisher_z,errors='coerce').dropna()
                if len(z):matrix[rois.index(a),rois.index(b)]=matrix[rois.index(b),rois.index(a)]=np.tanh(z.mean())
            last=ax.imshow(matrix,cmap='coolwarm',vmin=-1,vmax=1)
            ax.set_xticks(range(4),rois);ax.set_yticks(range(4),rois)
            ax.set_title(f'{state} | {mode} · n={tmp.mouse_id.nunique()}')
            for x in range(4):
                for y in range(4):
                    if np.isfinite(matrix[x,y]):ax.text(y,x,f'{matrix[x,y]:.2f}',ha='center',va='center',fontsize=8)
    fig.subplots_adjust(left=.07,right=.84,bottom=.07,top=.92,hspace=.32,wspace=.3)
    color_ax=fig.add_axes([.875,.13,.018,.72])
    fig.colorbar(last,cax=color_ax,label='Fisher-z mean, back-transformed r')
    filename.parent.mkdir(parents=True,exist_ok=True);fig.savefig(filename,dpi=165,bbox_inches='tight');plt.close(fig)
    return True


def plot_group_maps(map_index,filename,bands=('0p05_0p10','0p11_0p20')):
    """Plot canonical relative-power maps for each requested frequency band.
    
        Parameters
        ----------
        map_index : pandas.DataFrame
            Paths to saved group NPY maps with ``band`` and ``state`` columns.
        filename : pathlib.Path
            Destination figure path.
        bands : sequence of str, optional
            Band keys to display in row order.
    
        Returns
        -------
        bool
            True if plotted; False if the map index is empty.
    
        Notes
        -----
        Awake/Iso maps use a 0–1 scale, while the optional Iso-minus-Awake
        difference uses a separate symmetric -1–1 color scale.
    """
    if map_index.empty:return False
    use_diff=map_index.state.astype(str).eq('Iso_minus_Awake').any()
    states=['Awake','Iso']+(['Iso_minus_Awake'] if use_diff else [])
    fig,axs=plt.subplots(len(bands),len(states),figsize=(5.4*len(states),4.4*len(bands)),squeeze=False)
    for i,band in enumerate(bands):
        for j,state in enumerate(states):
            ax=axs[i,j]
            g=map_index[(map_index.band==band)&(map_index.state==state)]
            if not g.empty:
                data=np.load(g.iloc[0]['path'])
                if state=='Iso_minus_Awake':
                    im=ax.imshow(data,cmap='coolwarm',vmin=-1,vmax=1)
                    ax.set_title(f'Iso − Awake | {band}')
                else:
                    im=ax.imshow(data,cmap='turbo',vmin=0,vmax=1)
                    ax.set_title(f'{state} | {band} | n={int(g.iloc[0]["n_mice"])}')
                fig.colorbar(im,ax=ax,fraction=.035,pad=.02)
            else:ax.set_title(f'{state} | {band} — no data')
            ax.set_axis_off()
    fig.suptitle('Relative spectral power maps · PSD band / 0.05–0.50 Hz')
    fig.tight_layout();filename.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(filename,dpi=160,bbox_inches='tight');plt.close(fig);return True


def mouse_band_metrics(mouse_psd,cfg):
    """Integrate mouse-level absolute ROI PSD over configured frequency bands.
    
        Returns
        -------
        pandas.DataFrame
            One row per REST branch, condition, mouse, ROI, and hemisphere.
            Contains absolute integrals, normalized band fractions, and the
            maximum-PSD frequency inside 0.08–0.40 Hz where available.
    
        Notes
        -----
        The denominator is the integral of the mouse-averaged absolute spectrum
        over ``spectral.reference_band_hz``; these scalar metrics complement
        separate per-pixel relative-power maps.
    """
    if mouse_psd.empty:return pd.DataFrame()
    d=mouse_psd[mouse_psd.psd_kind=='absolute_psd']
    group=['rest_variant','state','mouse_id','roi','hemisphere']
    rows=[]
    for vals,g in d.groupby(group,dropna=False):
        rec=dict(zip(group,vals));g=g.sort_values('frequency_hz')
        f=g.frequency_hz.to_numpy(float);p=g.value.to_numpy(float)
        den=integrate(f,p,*cfg['spectral']['reference_band_hz'])
        rec.update(mask_set=cfg['mask_set'],analysis_roi_acronym=rec['roi'],
                   spontaneous_condition=rec['state'],
                   relative_reference_power_0p05_0p50=float(den))
        for name,(lo,hi) in cfg['spectral']['bands'].items():
            bp=integrate(f,p,lo,hi)
            rec[f'band_power_{name}']=float(bp)
            rec[f'relative_power_{name}']=float(bp/den) if np.isfinite(den) and den>0 and np.isfinite(bp) else np.nan
        rec['band_power_main']=rec.get('band_power_0p08_0p40',np.nan)
        rec['relative_band_power_main']=rec.get('relative_power_0p08_0p40',np.nan)
        mask=(f>=.08)&(f<=.4)&np.isfinite(p)
        rec['f_peak_main_hz']=float(f[np.flatnonzero(mask)[np.argmax(p[mask])]]) if mask.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_correlations(corr):
    """Aggregate successful segment correlations to stems and then mice.
    
        Returns
        -------
        tuple[pandas.DataFrame, pandas.DataFrame]
            Stem-level and mouse-level tables containing averaged ``fisher_z``,
            counts, and ``r = tanh(fisher_z)``. Failed segments are excluded.
    
        Notes
        -----
        Averaging is performed in Fisher-z space; Pearson r is not averaged
        directly. Each stem contributes one mean at the mouse aggregation step.
    """
    cols=['rest_variant','state','mouse_id','stem','variant','roi_a','roi_b']
    if corr.empty:return pd.DataFrame(),pd.DataFrame()
    valid=corr[corr.status=='PASS'].copy()
    if valid.empty:return pd.DataFrame(),pd.DataFrame()
    stem=valid.groupby(cols,dropna=False).agg(fisher_z=('fisher_z','mean'),
                                              n_segments=('fisher_z','count')).reset_index()
    stem['r']=np.tanh(stem.fisher_z)
    cols_mouse=[k for k in cols if k!='stem']
    mouse=stem.groupby(cols_mouse,dropna=False).agg(fisher_z=('fisher_z','mean'),
                         n_stems=('stem','nunique'),n_segments=('n_segments','sum')).reset_index()
    mouse['r']=np.tanh(mouse.fisher_z)
    return stem,mouse


def execute(project_config_path,mice=None,stems=None,overwrite=None):
    """Run spontaneous-hemodynamic analysis and write its scientific outputs.
    
        Parameters
        ----------
        project_config_path : str or pathlib.Path
            Main YAML with the ``step_spontaneous_hemodynamics`` block.
        mice, stems : sequence of str or None, optional
            Optional subset of the saved Step6b selection.
        overwrite : bool or None, optional
            Whether existing results may be replaced. ``None`` uses
            ``output.overwrite`` from the YAML/defaults.
    
        Returns
        -------
        dict
            ``tables`` (in-memory DataFrames), ``figures`` (figure paths),
            ``paths`` (output directories) and resolved ``config``.
    
        Raises
        ------
        FileExistsError
            If ``run_summary.csv`` already exists and overwrite is disabled.
        ValueError
            If ``raw.fps`` or a processed segment is invalid.
    
        Notes
        -----
        Step6b cohort/side/stem eligibility gates ROI calculations. For each
        segment, the mean of pixelwise Hann periodograms produces ROI and
        whole-brain PSD; stems and then mice are averaged before mouse-level
        relative PSD normalization. Correlation uses separate Step6b homotopic
        masks and configured full-segment Butterworth filtering, followed by
        optional edge trim and zero-lag Pearson/Fisher z. Spectral maps use the
        unfiltered 60-s preparation and are projected to canonical space.
    
        Successful results for a stem are appended only after its window loop.
        Missing processed Zarr is reported as ``skipped_missing_zarr``; other
        errors are recorded and optionally raised by ``batch.continue_on_error``.
        Outputs include CSV tables, canonical NPY maps and PNG figures.
    """
    idx,project,cfg,paths,paths6b,tables=build_index(project_config_path,mice,stems)
    out=_output_paths(cfg,paths); summary_file=out['tables']/'run_summary.csv'
    if overwrite is None:overwrite=bool(cfg['output']['overwrite'])
    if summary_file.exists() and not overwrite:raise FileExistsError(f'{summary_file}; specify --overwrite')
    for name,p in out.items():
        if name!='root':p.mkdir(parents=True,exist_ok=True)
    rest=accepted_rest_windows(idx,cfg,paths)
    ids=roi_id_map(tables,cfg['rois'])
    fps=float(project.get('raw',{}).get('fps',0))
    if fps<=0:raise ValueError('raw.fps missing/invalid')
    need_maps=bool(cfg['relative_power_maps']['enabled'])
    atlas=load_allen_atlas_safe(project) if (need_maps or cfg['correlations']['enabled']) else None
    mouse_psd_source=[];corr_rows=[];map_records=[];summ=[];errors=[];segment_qc=[];windows_used=[]
    relbands=cfg['spectral']['bands'];mapbands=cfg['relative_power_maps']['bands']
    for _,row in idx.iterrows():
        stem=str(row['stem']);state=row['spontaneous_condition'];ds=None
        # Commit scientific outputs per stem atomically; an exception discards its partial rows.
        local_psd=[];local_corr=[];local_maps=[];local_qc=[];local_windows=[];n_ok=0
        try:
            ds,var,zpath=_zarr_open(row,cfg)
            geo=load_geometry(row,cfg['mask_set'],project,paths,paths6b,
                              (int(var.sizes['y']),int(var.sizes['x'])),scale=float(cfg['spatial']['scale']))
            windows=windows_for_stem(row,int(var.sizes['time']),fps,rest,cfg)
            if not windows:
                summ.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,status='no_rest' if state=='Awake' else 'no_full_window',n_windows=0))
                continue
            # Load genuine Step6b canonical L/R paired-pixel geometry for correlations.
            pairable_fc = (homotopic_pairable_fc_from_step6b(
                row, geo, project, paths, paths6b, atlas)
                if cfg['correlations']['enabled'] else None)
            # Gate only per-side ROI; do not invent pass where row is missing.
            side_pass={(roi,side):selected_side_pass(tables,row,roi,side)
                       for roi in cfg['rois'] for side in ('left','right')}
            segmaps={band:[] for band in mapbands}
            for iw,(t0,t1,source) in enumerate(windows):
                prep=prepare_segment(var,t0,t1,fps,geo,cfg)
                expected_frames=int(round(float(cfg['segments']['primary_sec'])*fps))
                if prep['n_frames']!=expected_frames:
                    raise ValueError(f'{stem} window {iw}: {prep["n_frames"]} frames; expected {expected_frames}. ' 
                                     'Cannot average PSDs on different frequency grids.')
                # Compute pixel-level PSD ONCE and reuse for every ROI and whole brain.
                f_pix, p_pix = pixelwise_psd_low_frequency(
                    prep['X_det'], fps, cfg['spectral']['max_saved_frequency_hz'],
                    cfg['spectral']['pixel_chunk_size'])
                side_psds={};nroi={}
                for roi in cfg['rois']:
                    for side in ('left','right'):
                        passed,reason=side_pass[(roi,side)]
                        if not passed:
                            local_qc.append(dict(mouse_id=row['mouse_id'],stem=stem,segment=iw,roi=roi,side=side,analysis='psd',status=reason))
                            continue
                        cols=roi_pixel_columns(prep['valid_flat'],geo,ids[roi],side)
                        npx=len(cols)
                        if npx < int(cfg['spectral']['min_roi_pixels_fc']):
                            local_qc.append(dict(mouse_id=row['mouse_id'],stem=stem,segment=iw,roi=roi,side=side,analysis='psd',status='insufficient_functional_pixels',n_pixels=npx))
                            continue
                        nroi[(roi,side)]=npx
                        f,p=f_pix,np.mean(p_pix[:,cols],axis=1);side_psds[(roi,side)]=(f,p)
                        local_qc.append(dict(mouse_id=row['mouse_id'],stem=stem,segment=iw,roi=roi,side=side,analysis='psd',status='PASS',n_pixels=npx))
                        for ff,v,rp in zip(f,p,relative_psd(f,p,cfg['spectral']['reference_band_hz'])):
                            if ff<=cfg['spectral']['max_saved_frequency_hz']:
                                for kind,value in (('absolute_psd',v),('relative_psd',rp)):
                                    local_psd.append(dict(rest_variant=cfg['rest_variant'],state=state,mouse_id=row['mouse_id'],
                                     stem=stem,segment=iw,roi=roi,hemisphere=side,psd_kind=kind,frequency_hz=float(ff),value=float(value)))
                    if (roi,'left') in side_psds and (roi,'right') in side_psds:
                        f,lp=side_psds[(roi,'left')];_,rp=side_psds[(roi,'right')]
                        bp=(lp+rp)/2 # Equal hemisphere weights; each side is mean-of-pixel PSD.
                        for ff,absval,relval in zip(f,bp,relative_psd(f,bp,cfg['spectral']['reference_band_hz'])):
                            if ff<=cfg['spectral']['max_saved_frequency_hz']:
                                for kind,val in (('absolute_psd',absval),('relative_psd',relval)):
                                    local_psd.append(dict(rest_variant=cfg['rest_variant'],state=state,mouse_id=row['mouse_id'],
                                     stem=stem,segment=iw,roi=roi,hemisphere='bilateral',psd_kind=kind,frequency_hz=float(ff),value=float(val)))
                # Whole-brain = arithmetic mean of all technically usable PIXEL PSDs.
                if cfg['spectral'].get('reference_spectra',{'whole_brain':True}).get('whole_brain',True):
                    f,p=f_pix,np.mean(p_pix,axis=1)
                    for ff,absval,relval in zip(f,p,relative_psd(f,p,cfg['spectral']['reference_band_hz'])):
                        if ff<=cfg['spectral']['max_saved_frequency_hz']:
                            for kind,val in (('absolute_psd',absval),('relative_psd',relval)):
                                local_psd.append(dict(rest_variant=cfg['rest_variant'],state=state,mouse_id=row['mouse_id'],
                                  stem=stem,segment=iw,roi='whole_brain',hemisphere='bilateral',psd_kind=kind,
                                  frequency_hz=float(ff),value=float(val)))
                if cfg['correlations']['enabled']:
                    corr_cfg = cfg['correlations']
                    mode = str(corr_cfg['filter_type'])
                    low_hz, band_high_hz = [float(x) for x in corr_cfg['band_hz']]
                    high_hz = (band_high_hz if mode == 'bandpass'
                               else float(corr_cfg['lowpass_high_hz']))
                    # Preserve 60-s detrended input for PSD/maps; filter complete
                    # segment and trim ONLY the correlation signal afterward.
                    filtered = filter_for_correlation(
                        prep['X_det'], fps, low=low_hz, high=high_hz,
                        order=int(corr_cfg['order']),
                        pad_sec=float(corr_cfg['synthetic_odd_pad_sec']),
                        filter_type=mode)
                    fc, trim_meta = trim_correlation_window(
                        filtered, fps,
                        left_sec=float(corr_cfg['trim_left_sec']),
                        right_sec=float(corr_cfg['trim_right_sec']))
                    corr_provenance = {
                        'correlation_filter_type': mode,
                        'correlation_filter_low_hz': low_hz if mode == 'bandpass' else np.nan,
                        'correlation_filter_high_hz': high_hz,
                        'trim_left_sec_requested': float(corr_cfg['trim_left_sec']),
                        'trim_right_sec_requested': float(corr_cfg['trim_right_sec']),
                        'correlation_t0_sec_actual': float(t0) + trim_meta['trim_left_sec_actual'],
                        'correlation_t1_sec_actual': float(t1) - trim_meta['trim_right_sec_actual'],
                        **trim_meta,
                    }
                    corr_tr={}
                    for roi in cfg['rois']:
                        for side in ('left','right'):
                            passed,_ = side_pass[(roi,side)]
                            if passed:
                                t,npair=roi_trace(fc,prep['valid_flat'],geo,ids[roi],side,
                                       min_pixels=int(cfg['spectral']['min_roi_pixels_fc']),
                                       spatial_mask_fc=pairable_fc)
                            else:
                                t,npair=None,0
                            corr_tr[(roi,side)] = t
                            local_qc.append(dict(mouse_id=row['mouse_id'],stem=stem,segment=iw,
                                roi=roi,side=side,analysis='correlation_homotopic',
                                status='PASS' if t is not None else ('insufficient_paired_pixels' if passed else 'side_ROI_FAIL'),
                                n_pixels=npair,**corr_provenance))
                    for rec in pairwise_correlations(corr_tr):
                        local_corr.append(dict(rest_variant=cfg['rest_variant'],state=state,mouse_id=row['mouse_id'],
                            stem=stem,segment=iw,t0_sec=t0,t1_sec=t1,window_source=source,**corr_provenance,**rec))
                if need_maps:
                    maps=pixelwise_relative_band_power_maps(prep,geo,fps,mapbands,
                        reference_band=cfg['relative_power_maps']['reference_band_hz'],
                        chunk=cfg['relative_power_maps']['pixel_chunk_size'])
                    for name,arr in maps.items():segmaps[name].append(arr)
                local_windows.append(dict(rest_variant=cfg['rest_variant'],mouse_id=row['mouse_id'],
                    stem=stem,state=state,segment=iw,t0_sec=float(t0),t1_sec=float(t1),
                    window_source=source,n_frames=int(prep['n_frames'])))
                n_ok+=1
            # Only successful stem results get appended; all windows processed.
            if n_ok==0:
                summ.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,status='no_valid_windows',n_windows=0))
                continue
            if need_maps:
                for band,arrays in segmaps.items():
                    mean,_=_nanmean_stack(arrays)
                    can,_=_to_canonical(mean,row,geo,project,paths,paths6b,atlas)
                    p=out['maps']/ 'by_stem'/str(row['mouse_id'])/stem/f'{band}_canonical.npy'
                    p.parent.mkdir(parents=True,exist_ok=True);np.save(p,can)
                    local_maps.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,band=band,path=str(p)))
            mouse_psd_source.extend(local_psd);corr_rows.extend(local_corr)
            map_records.extend(local_maps);segment_qc.extend(local_qc);windows_used.extend(local_windows)
            summ.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,status='ok',n_windows=n_ok))
        except Exception as exc:
            # Missing input Zarr is an expected scientific exclusion for some stems
            # (e.g. m16). Keep the stem in the cohort accounting, but do not
            # include it in PSD, maps, correlations or the processing-errors table.
            if isinstance(exc, FileNotFoundError) and str(exc).startswith('Processed Zarr absent:'):
                summ.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,
                                 status='skipped_missing_zarr',n_windows=0))
            else:
                errors.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,error=repr(exc)))
                summ.append(dict(mouse_id=row['mouse_id'],stem=stem,state=state,status='error',n_windows=0))
                if not cfg['batch']['continue_on_error']:raise
        finally:
            if ds is not None:ds.close()
    source=pd.DataFrame(mouse_psd_source);corr=pd.DataFrame(corr_rows);mi=pd.DataFrame(map_records)
    if not source.empty:
        # Preserve the v1.10 scientific contract: normalize *mouse-level* mean PSD,
        # not average already normalized 60-s curves.
        source=source[source.psd_kind=='absolute_psd'].copy()
        base=['rest_variant','state','mouse_id','roi','hemisphere','psd_kind','frequency_hz']
        stemcols=base+['stem']
        stem_psd=source.groupby(stemcols,dropna=False).value.mean().reset_index()
        mouse_abs=stem_psd.groupby(base,dropna=False).value.mean().reset_index()
        rel=mouse_abs.copy();rel['psd_kind']='relative_psd'
        cols=['rest_variant','state','mouse_id','roi','hemisphere']
        for _,g in rel.groupby(cols,dropna=False):
            order=g.sort_values('frequency_hz')
            denom=integrate(order.frequency_hz.to_numpy(float),order.value.to_numpy(float),
                            *cfg['spectral']['reference_band_hz'])
            rel.loc[g.index,'value']=g['value']/denom if np.isfinite(denom) and denom>0 else np.nan
        mouse_psd=pd.concat([mouse_abs,rel],ignore_index=True)
    else:stem_psd=mouse_psd=pd.DataFrame()
    band_mouse=mouse_band_metrics(mouse_psd,cfg)
    corr_stem,corr_mouse=summarize_correlations(corr)
    gindex=[]
    if not mi.empty:
        for (band,state,mouse),g in mi.groupby(['band','state','mouse_id']):
            mean,_=_nanmean_stack([np.load(p) for p in g.path])
            p=out['mouse_maps']/state/f'{mouse}_{band}.npy';p.parent.mkdir(parents=True,exist_ok=True);np.save(p,mean)
            gindex.append(dict(band=band,state=state,mouse_id=mouse,path=str(p)))
    mouse_maps=pd.DataFrame(gindex);groups=[]
    if not mouse_maps.empty:
        for (band,state),g in mouse_maps.groupby(['band','state']):
            mean,num=_nanmean_stack([np.load(p) for p in g.path])
            mean[num<int(cfg['relative_power_maps']['min_group_n_per_pixel'])]=np.nan
            p=out['group_maps']/f'{state}_{band}.npy';np.save(p,mean)
            groups.append(dict(band=band,state=state,path=str(p),n_mice=len(g)))
    group_maps=pd.DataFrame(groups)
    # Existing v1.10 descriptive contrast is retained: Iso minus Awake,
    # only at pixels with valid means in BOTH condition maps.
    if cfg['relative_power_maps'].get('save_difference_map',True) and not group_maps.empty:
        for band in sorted(set(group_maps.band)):
            d=group_maps[group_maps.band==band]
            ii=d[d.state=='Iso'];aa=d[d.state=='Awake']
            if len(ii)==1 and len(aa)==1:
                a=np.load(aa.iloc[0]['path']);b=np.load(ii.iloc[0]['path'])
                diff=np.where(np.isfinite(a)&np.isfinite(b),b-a,np.nan).astype(np.float32)
                p=out['group_maps']/f'Iso_minus_Awake_{band}.npy';np.save(p,diff)
                groups.append(dict(band=band,state='Iso_minus_Awake',path=str(p),
                                   n_mice=min(int(ii.iloc[0]['n_mice']),int(aa.iloc[0]['n_mice']))))
    group_maps=pd.DataFrame(groups)
    method_contract=pd.DataFrame([dict(
        psd_roi='mean_per_pixel_hann_periodogram',
        psd_whole_brain='mean_all_valid_pixel_hann_periodograms',
        psd_bilateral='equal_weight_left_right_mean_of_pixel_psd',
        correlation_roi_mask='step6b_homotopic_pairable_bilateral_canonical_per_stem',
        correlation_spatial_grid='projected_to_stem_fc_nearest',
        correlation_statistic='pearson_then_fisher_z_segment_stem_mouse',
        correlation_filter_type=str(cfg['correlations']['filter_type']),
        correlation_band_hz=str(cfg['correlations']['band_hz']),
        correlation_lowpass_high_hz=float(cfg['correlations']['lowpass_high_hz']),
        correlation_filter_low_hz=(float(cfg['correlations']['band_hz'][0])
                                   if cfg['correlations']['filter_type'] == 'bandpass' else np.nan),
        correlation_filter_high_hz=(float(cfg['correlations']['band_hz'][1])
                                    if cfg['correlations']['filter_type'] == 'bandpass'
                                    else float(cfg['correlations']['lowpass_high_hz'])),
        correlation_trim_left_sec=float(cfg['correlations']['trim_left_sec']),
        correlation_trim_right_sec=float(cfg['correlations']['trim_right_sec']),
        correlation_filter_trim_order='detrend_full_segment_then_filter_then_trim_then_roi_pearson',
        rest_variant=cfg['rest_variant'],
    )])
    tables_out={'analysis_index':idx,'rest_segments_used':rest,'windows_used_60s':pd.DataFrame(windows_used),'run_summary':pd.DataFrame(summ),
                'analysis_methods':method_contract,
                'errors':pd.DataFrame(errors),'roi_stem_segment_qc':pd.DataFrame(segment_qc),
                'spectra_segment':source,'spectra_stem':stem_psd,'spectra_mouse':mouse_psd,
                'relative_band_power_mouse_values_60s':band_mouse,
                'spectral_mouse_values_60s':band_mouse,
                'roi_correlation_segments':corr,'roi_correlation_stems':corr_stem,
                'roi_correlation_mice':corr_mouse,'mouse_map_index':mouse_maps,'group_map_index':group_maps}
    for name,df in tables_out.items():
        if name=='spectra_segment' and not cfg['output']['save_segment_spectra_long']:continue
        df.to_csv(out['tables']/f'{name}.csv',index=False)
    figs={'PSD_relative':out['figures']/'PSD_relative_individual_mice.png',
          'PSD_absolute':out['figures']/'PSD_absolute_individual_mice.png',
          'ROI_correlations':out['figures']/'ROI_correlations_Awake_Iso.png',
          'spectral_power_maps':out['figures']/'relative_band_maps_Awake_Iso.png'}
    plot_psd(mouse_psd,figs['PSD_relative'],'relative_psd') if not mouse_psd.empty else None
    plot_psd(mouse_psd,figs['PSD_absolute'],'absolute_psd') if not mouse_psd.empty else None
    plot_correlations(corr_mouse,figs['ROI_correlations'])
    plot_group_maps(group_maps,figs['spectral_power_maps'],bands=tuple(mapbands))
    return dict(tables=tables_out,figures=figs,paths=out,config=cfg)


def main():
    """Provide the command-line interface for preflight and batch execution.
    
        ``--preflight-only`` prints Step6b/REST checks without processing Zarr;
        ``--overwrite`` explicitly permits replacing previous analysis outputs.
    """
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--mice',nargs='*');p.add_argument('--stems',nargs='*')
    p.add_argument('--overwrite',action='store_true');p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args()
    if a.preflight_only:
        pf=preflight(a.config,a.mice,a.stems)
        print(pf['eligibility'].to_string(index=False));print('accepted Awake REST:',len(pf['rest60']));return
    result=execute(a.config,a.mice,a.stems,overwrite=a.overwrite)
    print(result['tables']['run_summary'].to_string(index=False))
    print('figures:',*[str(x) for x in result['figures'].values()],sep='\n')

if __name__=='__main__':main()
