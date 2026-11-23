"""Standalone numerical/geometry helpers for Step Spontaneous Hemodynamics.

Uses Step6b's saved reference-space products and Step6's registration functions.
No Step7 or Step6c module is imported. Spatial resize and per-segment detrending
are numerically equivalent to the original standalone v1.10 + Step7 path.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from scipy import signal
from skimage.transform import warp


def nearest(mask, shape, labels=False):
    """Resize a binary mask or integer label map using nearest-neighbor sampling.
    
        Parameters
        ----------
        mask : array_like
            Two-dimensional reference array.
        shape : tuple[int, int]
            Output ``(height, width)`` in pixels.
        labels : bool, optional
            If True, round to integer labels; otherwise return a Boolean mask.
    
        Returns
        -------
        numpy.ndarray
            Resized array, or a copy if the requested size already matches.
    """
    a = np.asarray(mask)
    if a.shape == tuple(shape): return a.copy()
    x = cv2.resize(a.astype(np.float32 if labels else np.uint8),
                   (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
    return np.rint(x).astype(np.int32) if labels else x.astype(bool)


def infer_full_shape(row, fallback):
    """Infer original full-resolution recording shape from available image paths.
    
        Tries registration and reference image columns in order, reading the first
        available image via OpenCV; ``fallback`` is used if none can be read.
    
        Returns
        -------
        tuple[int, int]
            Full-resolution image height and width.
    """
    for col in ('moving_registration_path_resolved','moving_registration_path',
                'fixed_registration_path_resolved','fixed_registration_path','ref_image_path'):
        val = str(row.get(col,'') or '')
        if not val or val.lower() in ('nan','none'): continue
        p = Path(val)
        if p.exists():
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is not None and im.ndim >= 2: return tuple(im.shape[:2])
    return tuple(fallback)


def warp_ref_to_stem(a, tform, shape, labels=False):
    # tform maps stem -> reference; skimage's inverse_map accepts this directly.
    """Warp a reference-space mask or label array into stem space.
    
        Parameters
        ----------
        a : array_like
            Reference-space binary mask or integer labels.
        tform : skimage.transform.ProjectiveTransform or compatible transform
            Stem-to-reference transform passed as the ``inverse_map`` argument
            to ``skimage.transform.warp``.
        shape : tuple[int, int]
            Output stem image shape ``(height, width)``.
        labels : bool, optional
            Use integer label output instead of a binary mask.
    
        Returns
        -------
        numpy.ndarray
            Nearest-neighbor warped labels or binary mask.
    """
    b=warp(np.asarray(a,dtype=np.float32), inverse_map=tform,
           output_shape=tuple(shape), order=0,preserve_range=True,mode='constant',cval=0)
    return np.rint(b).astype(np.int32) if labels else (b > .5)


@dataclass
class Geometry:
    """Spatial masks and dimensions for one recording on the FC grid.
    
        Attributes
        ----------
        fc_shape, processed_shape, stem_fullres_shape : tuple[int, int]
            Height and width of the FC, Step4 processed and full stem grids.
        bilateral_processed : numpy.ndarray
            Saved usable-brain mask projected to Step4 processed pixels.
        bilateral_fc, left_fc, right_fc : numpy.ndarray
            Boolean technical and hemisphere masks on the FC grid.
        labels_fc : numpy.ndarray
            Integer analysis-ROI labels on the FC grid.
    """
    fc_shape: tuple[int,int]
    processed_shape: tuple[int,int]
    stem_fullres_shape: tuple[int,int]
    bilateral_processed: np.ndarray
    bilateral_fc: np.ndarray
    left_fc: np.ndarray
    right_fc: np.ndarray
    labels_fc: np.ndarray


def load_geometry(row, mask_set, config, project_paths, step6b_paths, processed_shape, scale=.5):
    """Load and project saved Step6b masks into processed and FC spaces.
    
        Parameters
        ----------
        row : mapping
            Recording metadata with ``group_id``, ``stem`` and ``ref_stem``.
        mask_set : str
            Step6b mask-set name.
        config, project_paths, step6b_paths : dict
            Main project configuration and resolved path dictionaries.
        processed_shape : tuple[int, int]
            Step4 spatial grid ``(height, width)`` in pixels.
        scale : float, optional
            Relative scale used to derive the FC analysis grid.
    
        Returns
        -------
        Geometry
            Reference-derived masks and ROI labels warped into stem coordinates.
    
        Raises
        ------
        FileNotFoundError
            If required saved Step6b products are missing.
        ValueError
            If their shapes differ or projected hemispheres overlap.
    
        Notes
        -----
        Uses Step6b saved final usable masks and registration overrides, not
        Step7 products; labels and masks are mapped with nearest-neighbor sampling.
    """
    import step6b_fc_geometry_coverage as s6b
    gid,stem,ref = (str(row[x]) for x in ('group_id','stem','ref_stem'))
    base=Path(step6b_paths['mask_sets_root'])/s6b.safe_key(mask_set,gid)
    root=base/'by_stem'/s6b.safe_key(stem)
    refdir=Path(step6b_paths['reference_root'])/s6b.safe_key(gid,ref)
    files={'bilateral':root/'bilateral_final_usable_ref.npy',
           'labels':base/'analysis_roi_ref_buffered_repaired.npy',
           'left':refdir/'left_hemisphere_ref.npy','right':refdir/'right_hemisphere_ref.npy'}
    for name,p in files.items():
        if not p.is_file(): raise FileNotFoundError(f'Step6b {name} missing: {p}')
    arrays={name: np.load(p) for name,p in files.items()}
    shapes={x.shape for x in arrays.values()}
    if len(shapes)!=1: raise ValueError(f'Step6b reference geometry shape mismatch {shapes}')
    tform,_,_=s6b.effective_tform_stem_to_ref(
        row,s6b.load_manual_registration_overrides(config,project_paths),config)
    full=infer_full_shape(row,arrays['bilateral'].shape)
    fc=tuple(max(1,int(round(n*scale))) for n in processed_shape)
    mapped={}
    for name in arrays:
        a=warp_ref_to_stem(arrays[name],tform,full,labels=name=='labels')
        mapped[name+'_processed']=nearest(a,processed_shape,labels=name=='labels')
        mapped[name+'_fc']=nearest(mapped[name+'_processed'],fc,labels=name=='labels')
    bilateral=mapped['bilateral_fc'].astype(bool)
    l=mapped['left_fc'].astype(bool);r=mapped['right_fc'].astype(bool)
    if np.any(l&r):
        raise ValueError('Projected Step6b left/right hemisphere masks overlap')
    return Geometry(fc,tuple(processed_shape),full,mapped['bilateral_processed'].astype(bool),
                    bilateral, l, r, mapped['labels_fc'].astype(np.int32))


def mask_aware_resize_movie_area(movie, valid_mask_processed, fc_shape, min_valid_fraction=.5):
    """Downsample a time-by-image movie using finite-pixel-weighted area averaging.
    
        Parameters
        ----------
        movie : array_like
            Signal array with axes ``(time, y, x)``.
        valid_mask_processed : array_like of bool
            Static valid-brain mask on the movie's spatial grid.
        fc_shape : tuple[int, int]
            Output FC grid ``(height, width)`` in pixels.
        min_valid_fraction : float, optional
            Minimum valid source-pixel area fraction for an output pixel.
    
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            FC movie ``(time, y_fc, x_fc)`` with invalid pixels as NaN,
            and the static source-brain area fraction per FC pixel.
    """
    arr=np.asarray(movie,dtype=np.float32)
    valid_static=np.asarray(valid_mask_processed,dtype=bool)
    if arr.ndim!=3 or valid_static.shape!=arr.shape[1:]: raise ValueError('movie/mask shape mismatch')
    h,w=map(int,fc_shape);out=np.full((arr.shape[0],h,w),np.nan,dtype=np.float32)
    if not 0<=min_valid_fraction<=1: raise ValueError('Invalid min_valid_fraction')
    static=cv2.resize(valid_static.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
    for i,frame in enumerate(arr):
        good=valid_static&np.isfinite(frame)
        num=cv2.resize(np.where(good,frame,0).astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
        den=cv2.resize(good.astype(np.float32),(w,h),interpolation=cv2.INTER_AREA)
        ok=(den>=min_valid_fraction)&(static>0)
        out[i,ok]=num[ok]/np.maximum(den[ok],1e-12)
    return out,static


def prepare_segment(var,t0,t1,fps,geo,config):
    """Extract and detrend one temporal segment for spectral analysis.
    
        Parameters
        ----------
        var : xarray.DataArray
            Step4 signal with ``time``, ``y`` and ``x`` dimensions.
        t0, t1 : float
            Segment start and end in seconds.
        fps : float
            Sampling rate in frames per second.
        geo : Geometry
            Projected technical brain masks and FC dimensions.
        config : dict
            Resolved spontaneous configuration, including spatial/variance gates.
    
        Returns
        -------
        dict
            ``X_det`` with shape ``(time, valid FC pixels)``, ``valid_flat``
            indexing the flattened FC image, validity mask, frame bounds, frame
            count, and per-pixel static area fractions.
    
        Notes
        -----
        Excludes pixels that are non-finite at any frame or have near-zero
        variance; optional detrending occurs after mask-aware spatial resizing.
        The returned unfiltered matrix is reused for PSD and spectral maps.
    """
    from scipy.signal import detrend
    n=int(var.sizes['time']);i0=int(round(t0*fps));i1=i0+int(round((t1-t0)*fps))
    if i0<0 or i0>=n or i1>n: raise IndexError(f'Segment [{i0},{i1}) out of {n}')
    movie=var.isel(time=slice(i0,i1)).load().values.astype(np.float32,copy=False)
    if movie.shape[0]<4: raise ValueError('Too few frames')
    movie_fc,valid_fraction=mask_aware_resize_movie_area(movie,geo.bilateral_processed,geo.fc_shape,
        float(config['spatial']['min_valid_fraction']))
    finite=np.all(np.isfinite(movie_fc),axis=0)
    std=np.zeros(geo.fc_shape,dtype=np.float32)
    if finite.any(): std[finite]=np.std(movie_fc[:,finite],axis=0)
    func=geo.bilateral_fc&finite&(std>float(config['preprocessing']['variance_epsilon']))
    valid_flat=np.flatnonzero(func.ravel())
    if not len(valid_flat): raise RuntimeError('No finite non-constant brain pixels')
    X=movie_fc.reshape(len(movie),-1)[:,valid_flat].astype(np.float32)
    if config['preprocessing']['detrend_enabled']:
        X=detrend(X,axis=0,type=str(config['preprocessing']['detrend_type'])).astype(np.float32)
    return dict(X_det=X,valid_flat=valid_flat,functional_valid_fc=func,
                n_frames=len(movie),i0=i0,i1=i1,spatial_valid_fraction=valid_fraction)


def roi_pixel_columns(valid_flat, geometry, roi_id, side, spatial_mask_fc=None):
    """Select FC columns belonging to one Step6b ROI and hemisphere.
    
        Parameters
        ----------
        valid_flat : numpy.ndarray
            Flattened FC indices of valid functional pixels.
        geometry : Geometry
            FC-grid labels and bilateral/hemisphere masks.
        roi_id : int
            Saved Step6b analysis structure ID.
        side : {'left', 'right'}
            Requested hemisphere.
        spatial_mask_fc : numpy.ndarray of bool or None, optional
            Extra FC-grid eligibility mask, used only for homotopic correlations.
    
        Returns
        -------
        numpy.ndarray
            Column indices into the prepared ``X_det`` matrix.
    """
    if side not in ('left', 'right'):
        raise ValueError(f'Invalid ROI hemisphere {side!r}')
    mask = (geometry.labels_fc == int(roi_id)) & geometry.bilateral_fc & (
        geometry.left_fc if side == 'left' else geometry.right_fc)
    if spatial_mask_fc is not None:
        additional = np.asarray(spatial_mask_fc, dtype=bool)
        if additional.shape != geometry.fc_shape:
            raise ValueError('Homotopic pair FC mask has wrong shape')
        mask &= additional
    fullcols = np.full(geometry.bilateral_fc.size, -1, dtype=np.int64)
    fullcols[np.asarray(valid_flat, dtype=np.int64)] = np.arange(len(valid_flat))
    cols = fullcols[np.flatnonzero(mask.ravel())]
    return cols[cols >= 0]


def roi_trace(X,valid_flat,geometry,roi_id,side,min_pixels=5,spatial_mask_fc=None):
    """Average valid ROI pixel time series, optionally restricting homotopic pixels.
    
        Parameters
        ----------
        X : numpy.ndarray
            Prepared matrix ``(frames, valid FC pixels)``.
        valid_flat : numpy.ndarray
            Flattened FC indices of columns in ``X``.
        geometry : Geometry
            Spatial ROI/hemisphere masks and labels.
        roi_id : int
            Analysis ROI structure identifier.
        side : {'left', 'right'}
            Hemisphere to extract.
        min_pixels : int, optional
            Minimum number of contributing functional pixels.
        spatial_mask_fc : numpy.ndarray or None, optional
            Additional eligibility mask, used for correlations.
    
        Returns
        -------
        tuple[numpy.ndarray or None, int]
            Mean ROI time series (frames,) and pixel count; trace is None when
            fewer than ``min_pixels`` columns are available.
    """
    cols = roi_pixel_columns(valid_flat, geometry, roi_id, side, spatial_mask_fc)
    if len(cols)<min_pixels: return None,len(cols)
    return np.mean(X[:,cols],axis=1).astype(np.float32),len(cols)


def pixelwise_psd_low_frequency(X, fps, max_saved_frequency_hz=0.5, chunk=2048):
    """Compute one Hann periodogram per functional pixel.
    
        Parameters
        ----------
        X : numpy.ndarray
            Prepared time-by-valid-pixel matrix, before correlation filtering.
        fps : float
            Sampling frequency in Hz.
        max_saved_frequency_hz : float, optional
            Upper saved frequency limit in Hz, inclusive.
        chunk : int, optional
            Number of pixel columns processed per periodogram call.
    
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequency vector (Hz) and PSD matrix ``(frequency, pixel)`` with
            density scaling (signal-unit squared per Hz).
    
        Notes
        -----
        ROI and whole-brain spectra are calculated by averaging these pixel
        periodograms, not by computing a periodogram of an averaged time series.
    """
    arr = np.asarray(X)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 1:
        raise ValueError('Expected [time, pixels] nonempty matrix for pixelwise PSD')
    chunk = max(1, int(chunk))
    freq = None
    cols = []
    for start in range(0, arr.shape[1], chunk):
        f, pxx = signal.periodogram(
            np.asarray(arr[:, start:start+chunk], dtype=np.float64),
            fs=float(fps), window='hann', detrend=False, scaling='density', axis=0)
        keep = f <= float(max_saved_frequency_hz)
        if freq is None:
            freq = f[keep].astype(np.float64)
        elif not np.array_equal(freq, f[keep]):
            raise ValueError('Pixel PSD frequency grids differ')
        cols.append(pxx[keep].astype(np.float64, copy=False))
    return freq, np.concatenate(cols, axis=1)


def homotopic_pairable_fc_from_step6b(row, geometry, config, project_paths,
                                      step6b_paths, atlas_data):
    """Project saved canonical homotopic-pair eligibility to a stem's FC grid.
    
        Parameters
        ----------
        row : mapping
            Stem, group, reference-stem, and mask-set identifiers.
        geometry : Geometry
            FC, processed, and full-resolution stem dimensions.
        config, project_paths, step6b_paths : dict
            Main YAML configuration and Step6/Step6b path dictionaries.
        atlas_data : object
            Loaded Allen atlas with ``flat_label`` on the canonical grid.
    
        Returns
        -------
        numpy.ndarray of bool
            Bilateral paired-pixel eligibility mask on the stem FC grid.
    
        Notes
        -----
        Only canonical pairs already accepted by Step6b are projected to the
        reference and stem grids. This procedure does not create synthetic
        opposite-hemisphere partners in FC space.
    """
    import step6b_fc_geometry_coverage as s6b
    if atlas_data is None:
        raise ValueError('Allen atlas required for canonical homotopic pairs')
    group, stem, ref_stem = (str(row[k]) for k in ('group_id', 'stem', 'ref_stem'))
    setname = str(row['mask_set'])
    pair_file = (Path(step6b_paths['canonical_masks_root']) / s6b.safe_key(setname, group)
                 / 'by_stem' / s6b.safe_key(stem) / 'homotopic_pairable_bilateral_canonical.npy')
    if not pair_file.is_file():
        raise FileNotFoundError(f'Step6b bilateral homotopic paired mask absent: {pair_file}')
    pair_can = np.load(pair_file).astype(bool)
    if pair_can.shape != tuple(atlas_data.flat_label.shape):
        raise ValueError('Homotopic paired canonical mask shape != Allen atlas grid')
    maskset_root = (Path(step6b_paths['mask_sets_root']) / s6b.safe_key(setname, group))
    ref_labels_file = maskset_root / 'analysis_roi_ref_buffered_repaired.npy'
    if not ref_labels_file.is_file():
        raise FileNotFoundError(f'Step6b reference ROI labels absent: {ref_labels_file}')
    ref_labels = np.load(ref_labels_file)
    trans_path = s6b.group_transform_path(project_paths, group, ref_stem)
    if not trans_path.is_file():
        raise FileNotFoundError(f'Step6 atlas projection transform absent: {trans_path}')
    tinfo = s6b.load_json_file(trans_path)
    proj = s6b.project_canonical_grid_to_reference(
        atlas_data, tinfo, tuple(ref_labels.shape), config)
    inside = proj['inside'] & pair_can
    paired_ref = np.zeros(ref_labels.shape, dtype=bool)
    paired_ref[proj['ref_y'][inside], proj['ref_x'][inside]] = True
    paired_ref &= (ref_labels > 0)
    stem_tform, _, _ = s6b.effective_tform_stem_to_ref(
        row, s6b.load_manual_registration_overrides(config, project_paths), config)
    paired_full = warp_ref_to_stem(paired_ref, stem_tform,
                                   geometry.stem_fullres_shape, labels=False)
    paired_processed = nearest(paired_full, geometry.processed_shape)
    paired_fc = nearest(paired_processed, geometry.fc_shape)
    return np.asarray(paired_fc & geometry.bilateral_fc, dtype=bool)


def compute_psd(trace,fps):
    """Compute the legacy Hann-window periodogram of one temporal trace.
    
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Frequencies (Hz) and density-scaled power spectrum. The signal is
            assumed already detrended; SciPy detrending is disabled.
    """
    f,p=signal.periodogram(np.asarray(trace,dtype=np.float64),fs=float(fps),
                         window='hann',detrend=False,scaling='density')
    return f.astype(np.float64),p.astype(np.float64)


def integrate(f,p,lo,hi,axis=0):
    """Numerically integrate spectral density over an inclusive frequency range.
    
        Parameters
        ----------
        f : array_like
            Frequency vector in Hz.
        p : numpy.ndarray
            Power spectral density values; frequency occupies axis 0.
        lo, hi : float
            Lower and upper integration bounds in Hz.
        axis : int, optional
            Integration axis, forwarded to NumPy trapezoidal integration.
    
        Returns
        -------
        float or numpy.ndarray
            Trapezoidal integral, or NaN when fewer than two bins are inside
            the requested interval.
    """
    m=np.isfinite(f)&(f>=lo)&(f<=hi)
    if int(m.sum())<2:return np.full(p.shape[1],np.nan) if np.ndim(p)>1 else np.nan
    fn=getattr(np,'trapezoid',np.trapz)
    return fn(p[m],f[m],axis=axis)


def relative_psd(f,p,den=(.05,.5)):
    """Normalize a PSD by its integrated reference-band power.
    
        Parameters
        ----------
        f, p : array_like
            Frequency vector (Hz) and spectral density.
        den : tuple[float, float], optional
            Normalization interval in Hz, by default 0.05–0.50 Hz.
    
        Returns
        -------
        numpy.ndarray
            Spectral density divided by the reference integral, or NaNs if the
            denominator is missing or non-positive.
    """
    value=integrate(f,p,*den)
    return p/value if np.isfinite(value) and value>0 else np.full_like(p,np.nan)


def band_fractions(f,p,bands,den=(.05,.5)):
    """Compute normalized integrated spectral power for named frequency bands.
    
        Returns
        -------
        dict[str, float]
            Band-name to ``integral(band PSD) / integral(reference PSD)``.
            Invalid normalization gives NaN.
    """
    denom=integrate(f,p,*den)
    return {name:float(integrate(f,p,*lim)/denom) if np.isfinite(denom) and denom>0 else np.nan
            for name,lim in bands.items()}


def pixelwise_relative_band_power_maps(prep,geo,fps,bands,reference_band=(.05,.5),chunk=2048):
    """Compute spatial maps of per-pixel normalized band power.
    
        Parameters
        ----------
        prep : dict
            Prepared segment containing ``X_det`` and ``valid_flat``.
        geo : Geometry
            FC grid dimensions and final usable-brain mask.
        fps : float
            Frame rate in Hz.
        bands : mapping
            Band name to lower/upper frequencies in Hz.
        reference_band : tuple[float, float], optional
            Frequency interval used for the normalization integral.
        chunk : int, optional
            Number of pixels per periodogram batch.
    
        Returns
        -------
        dict[str, numpy.ndarray]
            One NaN-masked FC spatial map per band.
    
        Notes
        -----
        Each pixel is normalized independently using its Hann periodogram;
        correlations-only filtering and edge trimming are not applied.
    """
    X=prep['X_det'].astype(np.float64);n=X.shape[1]
    vals={k:np.full(n,np.nan,np.float32) for k in bands}
    for k in range(0,n,max(1,int(chunk))):
        sl=slice(k,min(n,k+max(1,int(chunk))))
        f,p=signal.periodogram(X[:,sl],fs=float(fps),window='hann',detrend=False,scaling='density',axis=0)
        ref=integrate(f,p,*reference_band)
        for name,(lo,hi) in bands.items():
            bp=integrate(f,p,lo,hi)
            ok=np.isfinite(ref)&(ref>0)&np.isfinite(bp)
            vv=np.full(len(ref),np.nan)
            vv[ok]=bp[ok]/ref[ok]
            vals[name][sl]=vv.astype(np.float32)
    out={}
    for name,v in vals.items():
        arr=np.full(geo.fc_shape,np.nan,np.float32);arr.ravel()[prep['valid_flat']]=v
        arr[~geo.bilateral_fc]=np.nan;out[name]=arr
    return out


def filter_for_correlation(X, fps, low=.08, high=.4, order=4, pad_sec=30,
                           *, filter_type='bandpass'):
    """Apply zero-phase Butterworth filtering before ROI correlation.
    
        Parameters
        ----------
        X : numpy.ndarray
            Complete detrended segment, ``(frames, valid FC pixels)``.
        fps : float
            Temporal sampling frequency in Hz.
        low, high : float, optional
            Lower and upper cutoff for ``bandpass``; only ``high`` is used by
            ``lowpass``. The caller resolves the latter from ``lowpass_high_hz``.
        order : int, optional
            Butterworth filter order.
        pad_sec : float, optional
            Maximum synthetic odd-padding duration, in seconds.
        filter_type : {'bandpass', 'lowpass'}, optional
            Configured filter family.
    
        Returns
        -------
        numpy.ndarray
            Filtered segment with the same shape as the input, as float32.
    
        Notes
        -----
        Uses SciPy SOS forward/backward filtering with odd padding when nonzero.
        Filtering precedes optional correlation-only edge trimming. Neither PSD
        nor power maps call this function.
    """
    filter_type = str(filter_type).lower().strip()
    if filter_type not in ('bandpass', 'lowpass'):
        raise ValueError(f'Invalid correlations.filter_type={filter_type!r}; use bandpass or lowpass')
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError('Invalid sampling frequency for correlation filter')
    if int(order) != order or int(order) < 1:
        raise ValueError('correlations.order must be a positive integer')
    if not (np.isfinite(high) and 0 < high < fps / 2):
        raise ValueError(f'Invalid correlation filter upper cutoff {high} Hz for fps={fps}')
    if filter_type == 'bandpass':
        if not (np.isfinite(low) and 0 < low < high):
            raise ValueError(f'Invalid correlation filter lower cutoff {low} Hz')
        wn = [float(low), float(high)]
    else:
        wn = float(high)
    if not np.isfinite(pad_sec) or pad_sec < 0:
        raise ValueError('correlations.synthetic_odd_pad_sec must be nonnegative')
    sos = signal.butter(int(order), wn, btype=filter_type, fs=float(fps), output='sos')
    pad = min(max(0, int(round(float(pad_sec) * fps))), max(0, X.shape[0] - 2))
    y = signal.sosfiltfilt(sos, X, axis=0,
                          padtype='odd' if pad else None,
                          padlen=pad)
    return y.astype(np.float32)


def trim_correlation_window(X, fps, left_sec=0.0, right_sec=0.0):
    """Remove requested frame counts from filtered correlation data only.
    
        Parameters
        ----------
        X : numpy.ndarray
            Full filtered segment with time along axis 0.
        fps : float
            Frames per second.
        left_sec, right_sec : float, optional
            Temporal margins in seconds, converted via ``round`` to frames.
    
        Returns
        -------
        tuple[numpy.ndarray, dict]
            Trimmed time matrix and metadata containing requested-frame counts,
            actual durations, and remaining frame count.
    
        Raises
        ------
        ValueError
            If the margins are invalid or leave fewer than three frames.
    """
    if not (np.isfinite(fps) and fps > 0):
        raise ValueError('Invalid sampling frequency for correlation trim')
    left = float(left_sec); right = float(right_sec)
    if not (np.isfinite(left) and np.isfinite(right) and left >= 0 and right >= 0):
        raise ValueError('correlation trim seconds must be finite and nonnegative')
    n = int(X.shape[0])
    nl = int(round(left * fps)); nr = int(round(right * fps))
    if n - nl - nr < 3:
        raise ValueError(f'Correlation edge trim leaves fewer than 3 frames: '
                         f'n={n}, left={nl}, right={nr}')
    end = n - nr if nr else n
    return X[nl:end, ...], {
        'trim_left_frames': nl, 'trim_right_frames': nr,
        'n_frames_before_trim': n, 'n_frames_after_trim': end - nl,
        'trim_left_sec_actual': nl / fps, 'trim_right_sec_actual': nr / fps,
    }


def pearson_fisher(a,b):
    """Calculate zero-lag Pearson correlation and its Fisher-z transform.
    
        Parameters
        ----------
        a, b : array_like
            Paired temporal ROI traces, with missing samples allowed.
    
        Returns
        -------
        tuple[float, float, int]
            Pearson r, Fisher z and number of finite paired frames. Undefined
            correlation returns NaN values when there are fewer than three
            frames or either trace is constant.
    
        Notes
        -----
        Fisher z uses ``arctanh`` after clipping r to ±0.999999.
    """
    x=np.asarray(a,float);y=np.asarray(b,float)
    valid=np.isfinite(x)&np.isfinite(y)
    if valid.sum()<3:return np.nan,np.nan,int(valid.sum())
    x=x[valid];y=y[valid]
    if np.std(x)<=0 or np.std(y)<=0:return np.nan,np.nan,len(x)
    r=float(np.corrcoef(x,y)[0,1]);r=float(np.clip(r,-1,1))
    z=float(np.arctanh(np.clip(r,-0.999999,0.999999)))
    return r,z,len(x)


def pairwise_correlations(traces):
    """Calculate six pairwise ROI correlations in three hemisphere variants.
    
        Parameters
        ----------
        traces : dict[tuple[str, str], numpy.ndarray or None]
            Mean ROI traces keyed by (ROI acronym, hemisphere).
    
        Returns
        -------
        list[dict]
            Pairwise zero-lag correlations and Fisher z for MOp, MOs, SS, RSP,
            separately for left, right, and bilateral mean traces.
    
        Notes
        -----
        Bilateral traces average left/right TIME SERIES before correlation, not
        their Pearson coefficients; missing either side invalidates a bilateral
        ROI. Each side is derived from Step6b paired-pixel eligibility.
    """
    from itertools import combinations
    rois=('MOp','MOs','SS','RSP');out=[]
    for mode in ('left','right','bilateral_trace_mean'):
        tmp={}
        for roi in rois:
            l=traces.get((roi,'left'));r=traces.get((roi,'right'))
            if mode=='left': tmp[roi]=l
            elif mode=='right':tmp[roi]=r
            else:tmp[roi]=(l+r)/2 if l is not None and r is not None else None
        for a,b in combinations(rois,2):
            if tmp[a] is None or tmp[b] is None: rr,zz,n=np.nan,np.nan,0
            else:rr,zz,n=pearson_fisher(tmp[a],tmp[b])
            out.append(dict(variant=mode,roi_a=a,roi_b=b,r=rr,fisher_z=zz,n_valid_frames=n,
                            status='PASS' if np.isfinite(zz) else 'ROI_or_signal_unavailable'))
    return out
