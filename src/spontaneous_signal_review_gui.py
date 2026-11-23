"""PT2021 read-only Napari viewer for Step6b-selected spontaneous/REST ROI signals.

Use the project's Napari environment:
    python src/spontaneous_signal_review_gui.py --config path/to/config.yaml

Napari's native time slider controls a turbo-colour Hb signal stack; a Matplotlib
Qt dock tracks the current frame on averaged Step6b ROI traces. Slow band can
be displayed for Iso 60-second windows ONLY after processing the full recording.
Scientific CSV, masks and Zarr inputs are never modified.
"""
from __future__ import annotations

import argparse
import gc
import faulthandler
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import xarray as xr

try:
    import napari
    from qtpy import QtCore, QtWidgets
    from matplotlib import colormaps as mpl_colormaps
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError as exc:
    raise ImportError(
        'Napari, qtpy, a Qt binding and Matplotlib Qt backend are required. '
        'Run in the same Python environment as rest_segment_review_gui.py.'
    ) from exc

from spontaneous_signal_review_core import (
    FILTERS_REST, FILTERS_FULL, SIGNALS, Study, MovieCache,
    load_geometry, roi_mask_items, MovieWindow,
)

SLOW = 'Slow band 0.009–0.08 Hz'
REST_MODE = 'REST 60s'
ISO_MODE = 'Iso 60s'
FULL_MODE = 'Full recording'


def console_unicode_fallback() -> None:
    """Preserve unicode labels on Windows consoles with non-UTF8 stdout."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='backslashreplace')
        except (AttributeError, OSError, ValueError):
            pass


def turbo_colormap():
    """Create a 256-entry Matplotlib turbo colormap for Napari image display."""
    from napari.utils.colormaps import Colormap
    rgba = mpl_colormaps['turbo'](np.linspace(0, 1, 256))
    return Colormap(colors=np.asarray(rgba), name='PT2021_turbo')


class ReviewController(QtWidgets.QWidget):
    """Read-only Napari controller for selecting and comparing Step6b ROI signals.
    
        Holds the current Step6b selection, scratch movie caches, displayed
        Napari image/overlay layers, and the Matplotlib time-course panel.
        All scientific CSV, NPY and processed-Zarr inputs remain untouched."""
    def __init__(self, viewer: napari.Viewer, config_path: Path,
                 cache_dir: Path | None = None, cache_backend: str = "auto"):
        """Initialize the study, scratch directory, GUI widgets and time-slider hooks.
        
                Parameters
                ----------
                viewer : napari.Viewer
                    Active Napari image viewer.
                config_path : pathlib.Path
                    Path to project YAML.
                cache_dir : pathlib.Path or None, optional
                    Parent for disposable scratch movie caches.
                cache_backend : {"auto", "hdf5", "zarr"}, optional
                    Chunked temporary storage format."""
        super().__init__()
        self.viewer = viewer
        self.study = Study(config_path)
        if cache_dir is not None:
            cache_dir = Path(cache_dir).expanduser().resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='pt2021_napari_signal_',
                                                dir=str(cache_dir) if cache_dir else None)
        self.cache_backend = str(cache_backend)
        self._cache_serial = 0
        self.cache: dict[tuple, MovieCache] = {}
        self.products: dict[str, MovieWindow] = {}
        self.trace_cache = {}
        self.dataset: xr.Dataset | None = None
        self.geometry = None
        self.entries: list[dict] = []
        self.masks: list[tuple[str, np.ndarray]] = []
        self.stem_rows: list[dict] = []
        self.segments: list[dict] = []
        self.source = None
        self._busy = False
        self._shown_stem = None
        self._current_window_identity = None
        self._layer_a = None
        self._layer_b = None
        self._boundary_layer = None
        self._usable_layer = None
        self._outside_layer = None
        self._cursor = None
        self._plot_ax = None
        self._mouse_handler_active = False
        self._fixed_limits_cache = {}
        self._build_controls()
        self._build_trace_dock()
        self.viewer.dims.events.current_step.connect(self._on_time_step)
        # Napari's Qt window is retained by the viewer until close.
        try:
            self.viewer.window._qt_window.destroyed.connect(lambda *a: self.close_inputs())
        except Exception:
            pass
        self._load_variants()

    def close_inputs(self):
        """Close the current Zarr dataset and dispose of temporary movie caches."""
        if self.dataset is not None:
            try: self.dataset.close()
            except Exception: pass
            self.dataset = None
        self._release_cache()
        try: self.temp.cleanup()
        except Exception: pass

    def _build_controls(self):
        # A scrollable control dock is crucial on 1080p Windows displays:
        # otherwise the dock's minimum size adds up with the plot dock's height
        # and forces Qt to request a window taller than the physical screen.
        """Build the scrollable controls for branch, ROI, signal, filter and time selection."""
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.controls_scroll = QtWidgets.QScrollArea(self)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.controls_panel = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(self.controls_panel)
        col.setContentsMargins(6, 6, 6, 6)
        col.setSpacing(5)
        self.controls_scroll.setWidget(self.controls_panel)
        outer.addWidget(self.controls_scroll)
        title = QtWidgets.QLabel('Spontaneous / REST signal review — READ ONLY')
        title.setWordWrap(True)
        title.setStyleSheet('font-weight: bold; font-size: 10pt;')
        col.addWidget(title)
        self.variant = self._combo(col, 'Step6b branch', ['with_rest', 'without_rest'])
        self.set_box = self._combo(col, 'Mask set', [])
        self.stem_box = self._combo(col, 'Mouse / stem', [])
        self.mode_box = self._combo(col, 'Time mode', [REST_MODE, FULL_MODE])
        self.window_box = self._combo(col, '60-second window', [])
        self.sig_a = self._combo(col, 'Image A (Napari)', SIGNALS)
        self.filt_a = self._combo(col, 'Filter A', FILTERS_REST)
        self.sig_b = self._combo(col, 'Comparison B', ['(off)'] + SIGNALS)
        self.filt_b = self._combo(col, 'Filter B', FILTERS_REST)
        self.sig_a.setCurrentText('dcHbT')
        self.sig_b.setCurrentText('(off)')
        self.filt_a.setCurrentText('Band-pass 0.08–0.4 Hz')
        self.filt_b.setCurrentText('Band-pass 0.08–0.4 Hz')
        btnrow = QtWidgets.QHBoxLayout()
        self.load_btn = QtWidgets.QPushButton('Load signal and ROIs')
        self.load_btn.setToolTip('Reads processed Zarr and saved Step6b masks. Does not modify project files.')
        btnrow.addWidget(self.load_btn)
        col.addLayout(btnrow)
        self.show_b = QtWidgets.QCheckBox('Show comparison B image instead of A')
        self.show_b.setChecked(False)
        col.addWidget(self.show_b)
        self.show_detrended = QtWidgets.QCheckBox('Overlay detrended A trace')
        self.show_detrended.setChecked(False)
        col.addWidget(self.show_detrended)
        self.show_boundaries = QtWidgets.QCheckBox('ROI outlines (accepted only)')
        self.show_boundaries.setChecked(True)
        col.addWidget(self.show_boundaries)
        self.show_usable = QtWidgets.QCheckBox('Usable brain boundary')
        self.show_usable.setChecked(False)
        col.addWidget(self.show_usable)
        self.fixed_scale = QtWidgets.QCheckBox('Fixed turbo contrast across time')
        self.fixed_scale.setChecked(True)
        col.addWidget(self.fixed_scale)
        col.addWidget(QtWidgets.QLabel('Accepted ROI — multiple selection'))
        self.roi_list = QtWidgets.QListWidget()
        self.roi_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.roi_list.setMinimumHeight(100)
        col.addWidget(self.roi_list, 1)
        self.select_all_btn = QtWidgets.QPushButton('Select all accepted ROI')
        col.addWidget(self.select_all_btn)
        self.info = QtWidgets.QLabel('')
        self.info.setWordWrap(True)
        col.addWidget(self.info)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.status.setStyleSheet('color: #888')
        col.addWidget(self.status)
        col.addWidget(QtWidgets.QLabel('Napari slider = current frame; plot cursor follows it.\n'
                                        'Click on trace to jump to that frame.\n'
                                        'B switches the image; both A and B can appear on the plot.'))
        self.setMinimumWidth(340)
        self.setMinimumHeight(200)
        self.resize(410, 570)
        self.variant.currentTextChanged.connect(self._load_variants)
        self.set_box.currentTextChanged.connect(self._load_stems)
        self.stem_box.currentIndexChanged.connect(self._load_stem_choices)
        self.mode_box.currentTextChanged.connect(self._mode_changed)
        self.window_box.currentIndexChanged.connect(self._window_changed)
        self.load_btn.clicked.connect(self.load)
        self.show_b.toggled.connect(self._change_image_visibility)
        self.show_detrended.toggled.connect(self._plot)
        self.show_boundaries.toggled.connect(self._update_overlays)
        self.show_usable.toggled.connect(self._update_overlays)
        self.fixed_scale.toggled.connect(self._apply_contrast)
        self.roi_list.itemSelectionChanged.connect(self._roi_changed)
        self.select_all_btn.clicked.connect(self._select_all)

    def _combo(self, col, title, options):
        """Add a labeled Qt combobox to the supplied layout and return it."""
        row = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(title)
        lbl.setMinimumWidth(110)
        combo = QtWidgets.QComboBox()
        combo.addItems(list(options))
        combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        row.addWidget(lbl)
        row.addWidget(combo, 1)
        col.addLayout(row)
        return combo

    def _build_trace_dock(self):
        """Create and attach the Matplotlib ROI time-series dock."""
        self.plot_widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.plot_widget)
        lay.setContentsMargins(4, 3, 4, 3)
        self.figure = Figure(figsize=(9.5, 3.7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(310)
        lay.addWidget(NavigationToolbar(self.canvas, self.plot_widget))
        lay.addWidget(self.canvas, 1)
        self.fig_click_id = self.canvas.mpl_connect('button_press_event', self._on_plot_click)
        self.plot_dock = self.viewer.window.add_dock_widget(
            self.plot_widget, name='ROI time series (Step6b)', area='bottom'
        )

    def _say(self, message):
        """Update status text without reentering the Qt event loop."""
        self.status.setText(str(message))
        try: self.viewer.status = str(message)
        except Exception: pass
        # Avoid QApplication.processEvents during large synchronous Zarr reads;
        # deferred repaint means no nested napari slider/layer callback.
        self.status.update()

    def _error(self, title, error):
        """Report a named exception to the GUI and standard error."""
        traceback.print_exc()
        self._say(f'{title}: {error}')
        QtWidgets.QMessageBox.warning(self, title, str(error))

    @staticmethod
    def _set_choices(combo, texts, current=None):
        """Replace combobox entries while preserving the desired value when possible."""
        combo.blockSignals(True)
        old = current if current is not None else combo.currentText()
        combo.clear()
        combo.addItems(list(texts))
        if combo.findText(old) >= 0:
            combo.setCurrentText(old)
        elif combo.count():
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _release_cache(self, keep: set | None = None):
        """Close obsolete HDF5 stores before removing scratch data.

        Keep active filters when changing only the selected ROI or Iso window.
        """
        keep = keep or set()
        for key, movie in list(self.cache.items()):
            if key in keep:
                continue
            self.cache.pop(key, None)
            folder = movie.temp
            try:
                movie.close()
            except Exception as exc:
                print(f'[QC cache] Could not close {folder}: {exc}', flush=True)
            try:
                shutil.rmtree(folder)
            except OSError:
                # Windows may briefly retain a Napari/Dask file handle after
                # the image layer switches. Cleanup of the root occurs at exit.
                pass

    def _remove_image_layers(self):
        """Remove the current signal/overlay image layers from Napari."""
        for layer in (self._boundary_layer, self._usable_layer, self._outside_layer,
                      self._layer_b, self._layer_a):
            if layer is not None and layer in self.viewer.layers:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    try: layer.visible = False
                    except Exception: pass
        self._layer_a = self._layer_b = None
        self._boundary_layer = self._usable_layer = self._outside_layer = None

    def _reset_view_data(self):
        # A changed stem/mask_set must never leave the previous mouse's
        # pseudo-colour maps visible under the newly selected stem name.
        """Reset the current recording, ROI geometry, plots and cached presentation state."""
        self._remove_image_layers()
        self.products.clear()
        self._release_cache()
        self.geometry = None
        self.entries = []
        self.masks = []
        self.trace_cache.clear()
        self.roi_list.clear()
        self._current_window_identity = None
        self._cursor = None
        if hasattr(self, 'figure'):
            self._plot()

    def _load_variants(self, *_):
        """Refresh eligible Step6b mask sets when the branch selection changes."""
        if self._busy: return
        try:
            self._reset_view_data()
            sets = self.study.mask_sets(self.variant.currentText())
            self._set_choices(self.set_box, sets)
            self._load_stems()
        except Exception as e:
            self._error('Could not read Step6b branches', e)

    def _load_stems(self, *_):
        """Refresh the available recording list for the selected branch and mask set."""
        if self._busy: return
        try:
            self._reset_view_data()
            self.stem_rows = self.study.stems(self.variant.currentText(), self.set_box.currentText())
            self._set_choices(self.stem_box, [r['description'] for r in self.stem_rows])
            self._load_stem_choices()
        except Exception as e:
            self._error('Could not read selected stems', e)

    def _chosen_stem(self):
        """Return the currently selected recording metadata, if any."""
        idx = self.stem_box.currentIndex()
        return self.stem_rows[idx] if 0 <= idx < len(self.stem_rows) else None

    def _load_stem_choices(self, *_):
        """Update time-mode options and window choices for the selected recording."""
        if self._busy: return
        self._reset_view_data()
        chosen = self._chosen_stem()
        if chosen is None: return
        try:
            self.source = self.study.source_row(chosen['stem'], chosen['group_id'])
            state = str(self.source.get('state', '')).strip().lower()
            self._set_choices(self.mode_box, [FULL_MODE, ISO_MODE] if state=='iso' else [REST_MODE, FULL_MODE],
                              ISO_MODE if state=='iso' else REST_MODE)
            self._mode_changed()
            self._say('Ready to load: ' + str(chosen['stem']))
        except Exception as e:
            self._error('Could not select stem', e)

    def _mode_changed(self, *_):
        """Offer the valid window and filter choices for the selected time mode.
        
                Slow-band is restricted to full-recording processing; the GUI may
                show cropped Iso windows from a full-recording slow-band cache."""
        if self._busy: return
        mode = self.mode_box.currentText()
        # Slow band is valid in Iso 60s only with full-record filtering + window view.
        options = FILTERS_FULL if mode in (FULL_MODE, ISO_MODE) else FILTERS_REST
        self._set_choices(self.filt_a, options, 'Band-pass 0.08–0.4 Hz')
        self._set_choices(self.filt_b, options, 'Band-pass 0.08–0.4 Hz')
        self.window_box.setEnabled(mode != FULL_MODE)
        self._reset_view_data()
        self._set_choices(self.window_box, [])
        self._say('Mode changed — Load signal and ROIs (full/REST/Iso).')

    def _window_changed(self, *_):
        # Do not recompute when changing stem/mode; the Load button handles it.
        """Schedule reloading the selected REST/Iso window when a movie is loaded."""
        if not self._busy and self.products:
            QtCore.QTimer.singleShot(0, self.load)

    def _select_all(self):
        """Select all currently listed Step6b-accepted ROI entries."""
        self.roi_list.selectAll()
        self._roi_changed()

    def _chosen_rois(self):
        """Return metadata for the ROI entries selected in the GUI list."""
        selected = {item.text() for item in self.roi_list.selectedItems()}
        return [r for r in self.entries if r['label'] in selected]

    def _roi_changed(self, *_):
        """Update displayed ROI masks and redraw selected ROI traces."""
        if self._busy: return
        if self.geometry is not None:
            self.masks = roi_mask_items(self.geometry, self._chosen_rois())
            self.trace_cache.clear()
            self._update_overlays()
            self._plot()

    def _window_index(self, seg, nframes):
        """Convert a time-window record into a bounded half-open frame interval."""
        a = int(round(float(seg['t0']) * self.study.fps))
        b = int(round(float(seg['t1']) * self.study.fps))
        if not (0 <= a < b <= nframes):
            raise ValueError(f'Window outside signal bounds: {seg}')
        if b-a != int(round(60.0*self.study.fps)):
            raise ValueError('Expected a complete 60-second sample window')
        return a,b

    def _compute_movie(self, stem, group_id, signal, filt, start, stop, nframes):
        """Reuse or create a cached filtered movie for the requested selection.
        
                In an Iso 60-second view with slow-band filtering, the full
                recording is processed first and the chosen window is then cropped;
                directly filtering a 60-second slow-band interval is prohibited."""
        full_context = (self.mode_box.currentText()==FULL_MODE or
                        (self.mode_box.currentText()==ISO_MODE and filt==SLOW))
        cs,ce = (0,nframes) if full_context else (start,stop)
        key = (stem, group_id, self.variant.currentText(), self.set_box.currentText(),
               signal, filt, cs, ce)
        if key not in self.cache:
            ny,nx = self.geometry.fc_shape
            estimated = (ce-cs)*ny*nx*4*2
            available = shutil.disk_usage(self.temp.name).free
            # Conservative bound: two uncompressed float32 matrices.
            # Chunked HDF5/Zarr may use substantially less space.
            if estimated > 0.80*available:
                raise RuntimeError(f'Not enough scratch disk for {signal}/{filt}: '
                                   f'~{estimated/1e9:.2f}GB uncompressed estimate, '
                                   f'{available/1e9:.2f}GB free on {self.temp.name}. '
                                   'Use --cache-dir with a drive having more free space, '
                                   'disable comparison B, or select a shorter window.')
            folder = Path(self.temp.name) / f'movie_{self._cache_serial:04d}'
            self._cache_serial += 1
            self._say(f'Processing {signal} | {filt} | {ce-cs} frames...')
            try:
                cache = MovieCache(self.study, self.geometry, self.dataset, signal, cs, ce,
                                   filt, full_context, folder, self._say,
                                   backend=self.cache_backend)
            except Exception as exc:
                try: shutil.rmtree(folder)
                except OSError: pass
                raise RuntimeError(
                    f'Failed to prepare the chunked disk cache for {signal} / {filt} '
                    f'in {folder}. Check free space and write permissions. '
                    f'Choose another scratch disk with --cache-dir F:\\temp\\PT2021_signal_cache. '
                    f'Underlying error: {exc}'
                ) from exc
            self.cache[key] = cache
        cache = self.cache[key]
        return MovieWindow(cache, start-cs, stop-cs)

    def load(self):
        """Load and display the selected recording, catching GUI-facing errors."""
        if self._busy: return
        self._busy = True
        self.load_btn.setEnabled(False)
        try:
            self._load_impl()
        except Exception as e:
            self._error('Could not load signal / masks', e)
        finally:
            self._busy = False
            self.load_btn.setEnabled(True)

    def _load_impl(self):
        """Read Step4 and Step6b inputs, resolve ROI masks and build display movies.
        
                Validates time-window selection, filters and eligible ROI pixels,
                then updates signal layers, Matplotlib traces and status messages.
                This operation reads input data but may write disposable disk cache."""
        chosen = self._chosen_stem()
        if chosen is None:
            raise RuntimeError('No stem selected in Step6b mask_set_selection')
        source = self.study.source_row(chosen['stem'], chosen['group_id'])
        stem, gid = str(chosen['stem']), str(chosen['group_id'])
        processed = self.study.signal_path(source)
        # Do not reload current stem's dataset when switching Iso windows.
        if self.dataset is None or self._shown_stem != (stem, str(processed)):
            if self.dataset is not None:
                try: self.dataset.close()
                except Exception: pass
            try:
                self.dataset = xr.open_zarr(processed, chunks='auto', consolidated=False, mask_and_scale=False)
            except Exception:
                self.dataset = xr.open_zarr(processed, chunks=None, consolidated=False, mask_and_scale=False)
            self._shown_stem = (stem, str(processed))
            # Don't retain unrelated animals' large arrays on disk.
            self._release_cache()
            self.trace_cache.clear()
            self._fixed_limits_cache.clear()
        names = [self.sig_a.currentText()]
        b_name = self.sig_b.currentText()
        if b_name != '(off)': names.append(b_name)
        missing = sorted(set(names)-set(self.dataset.data_vars))
        if missing:
            raise KeyError(f'Signals absent from processed Zarr {processed}: {missing}; '
                           f'available={list(self.dataset.data_vars)}')
        signal = self.dataset[names[0]].transpose('time', 'y', 'x')
        nframes = int(signal.sizes['time'])
        shape = (int(signal.sizes['y']), int(signal.sizes['x']))
        mode = self.mode_box.currentText()
        self.segments = self.study.segments(stem, source.get('state',''), nframes)
        labels = [s['label'] for s in self.segments]
        previous = self.window_box.currentText()
        self._set_choices(self.window_box, labels, previous)
        if mode == FULL_MODE:
            start,stop = 0,nframes
        else:
            if not self.segments:
                raise ValueError(f'No complete {mode} intervals in current stem. Select Full recording.')
            idx = self.window_box.currentIndex()
            if idx < 0: idx=0
            start,stop = self._window_index(self.segments[idx],nframes)
        filters = [self.filt_a.currentText(), self.filt_b.currentText()]
        if mode == REST_MODE and SLOW in filters:
            raise ValueError('Slow band is not available for accepted Awake REST 60s')
        if mode == ISO_MODE and str(source.get('state','')).strip().lower()!='iso':
            raise ValueError('Iso windows are only available for Iso state')
        self.geometry = load_geometry(self.study, self.variant.currentText(), self.set_box.currentText(), source, shape)
        self.entries = self.study.accepted_rois(self.variant.currentText(),self.set_box.currentText(),stem,gid)
        if not self.entries:
            raise RuntimeError('No Step6b cohort-selected + stem-pass ROI for this recording')
        selected_labels = {item.text() for item in self.roi_list.selectedItems()}
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for entry in self.entries:
            self.roi_list.addItem(entry['label'])
        for i in range(self.roi_list.count()):
            item=self.roi_list.item(i)
            if item.text() in selected_labels or (not selected_labels and i==0):
                item.setSelected(True)
        self.roi_list.blockSignals(False)
        self.masks = roi_mask_items(self.geometry, self._chosen_rois())
        if not self.masks:
            raise RuntimeError('Selected Step6b ROI has no final usable pixels on FC grid')
        products = {}
        products['A'] = self._compute_movie(stem,gid,names[0],filters[0],start,stop,nframes)
        if b_name!='(off)':
            products['B'] = self._compute_movie(stem,gid,b_name,filters[1],start,stop,nframes)
        self.products=products
        identity = (stem,self.variant.currentText(),self.set_box.currentText(),start,stop,tuple(names),tuple(filters))
        self._configure_layers(identity)
        self._plot()
        active_movies = {id(p.cache) for p in self.products.values()}
        self._release_cache(keep={k for k, v in self.cache.items() if id(v) in active_movies})
        self.viewer.title = f'PT2021 spontaneous signal QC | {stem} | {mode}'
        slow_used = any(p.filt==SLOW for p in products.values())
        slow_note = ('Iso slow: full recording filtered once, then 60s window cropped.'
                     if mode==ISO_MODE and slow_used else '')
        if slow_used and nframes/self.study.fps < (3.0/0.009):
            slow_note += '\nCAUTION: less than 3 cycles of 0.009 Hz fit into the recording; slow-band QC is exploratory.'
        self.info.setText(f'{stem}  |  {mode}  |  time {start/self.study.fps:.2f}–{stop/self.study.fps:.2f}s\n'
                          f'Grid: {self.geometry.fc_shape} | accepted ROI: {len(self.entries)} | '
                          f'ROI traces: {len(self.masks)} | valid pixels A: {int(products["A"].cache.valid.sum())}\n'
                          + slow_note)
        self._say('Loaded. Move Napari time slider or click on the ROI trace to seek.')

    def _configure_layers(self, identity):
        # Keep native napari slider alive by updating image-layer data in place.
        """Add or update Napari signal layers while preserving the time slider."""
        if self._layer_a is None or self._layer_a not in self.viewer.layers:
            self.viewer.layers.clear()
            self._layer_a = self.viewer.add_image(
                self.products['A'].stack, name='A — signal',
                colormap=turbo_colormap(), blending='translucent', visible=True)
        else:
            self._layer_a.data = self.products['A'].stack
        self._layer_a.name = f'A · {self.products["A"].signal_var} · {self.products["A"].filt}'
        if 'B' in self.products:
            if self._layer_b is None or self._layer_b not in self.viewer.layers:
                self._layer_b = self.viewer.add_image(
                    self.products['B'].stack, name='B — comparison',
                    colormap=turbo_colormap(), blending='translucent', visible=False)
            else:
                self._layer_b.data = self.products['B'].stack
            self._layer_b.name = f'B · {self.products["B"].signal_var} · {self.products["B"].filt}'
        elif self._layer_b is not None and self._layer_b in self.viewer.layers:
            self.viewer.layers.remove(self._layer_b)
            self._layer_b=None
        self._apply_contrast()
        self._update_overlays()
        self._change_image_visibility()
        self.viewer.dims.set_current_step(0,min(self.products['A'].nframes//2,self.products['A'].nframes-1))
        self._current_window_identity = identity

    def _limits(self, product):
        """Estimate symmetric turbo contrast limits for a filtered movie.
        
                With fixed scaling, uses up to 20 evenly spaced frames; otherwise
                derives limits from the current frame. These settings affect only
                visual display, not the underlying signal data."""
        key=(str(product.path_filter), product.local_start, product.local_stop,
             self.fixed_scale.isChecked())
        if key in self._fixed_limits_cache: return self._fixed_limits_cache[key]
        if self.fixed_scale.isChecked():
            frames=np.linspace(0,product.nframes-1,min(product.nframes,20),dtype=int)
        else:
            frames=[self._frame_index()]
        vals=[]
        for idx in frames:
            arr=product.frame(int(idx))
            gd=arr[np.isfinite(arr)]
            if gd.size:
                vals.append(float(np.percentile(np.abs(gd),99)))
        peak = max(float(np.median(vals)) if vals else 1e-8, 1e-8)
        lim=(-peak,peak)
        if self.fixed_scale.isChecked(): self._fixed_limits_cache[key]=lim
        return lim

    def _apply_contrast(self, *_):
        """Apply computed contrast limits to visible Napari signal layers."""
        for name,layer in (('A',self._layer_a),('B',self._layer_b)):
            if layer is not None and name in self.products:
                try: layer.contrast_limits = self._limits(self.products[name])
                except Exception: pass

    def _update_overlays(self, *_):
        """Draw final usable brain boundaries and selected ROI outlines in Napari.
        
                A black RGBA overlay hides pixels outside the usable brain.
                The colored one-pixel ROI contours are overlays only; no mask or
                processed signal values are modified."""
        if self.geometry is None or self._layer_a is None: return
        h,w=self.geometry.fc_shape
        if self._outside_layer is None or self._outside_layer not in self.viewer.layers:
            # An opaque black RGBA layer ensures no pseudocolour outside the actual
            # per-stem final-usable brain, regardless of NaN handling in VisPy.
            rgba=np.zeros((h,w,4),dtype=np.float32)
            rgba[:,:,3]=(~self.geometry.mask_fc).astype(np.float32)
            self._outside_layer=self.viewer.add_image(rgba,rgb=True,name='Outside final usable brain',
                                                       blending='translucent',visible=True)
        else:
            rgba=np.zeros((h,w,4),dtype=np.float32)
            rgba[:,:,3]=(~self.geometry.mask_fc).astype(np.float32)
            self._outside_layer.data=rgba
        # Static 2D mask overlays share y/x of the time-dependent image.
        if self._usable_layer is None or self._usable_layer not in self.viewer.layers:
            self._usable_layer=self.viewer.add_labels(self.geometry.mask_fc.astype(np.uint8),
                                                       name='Final usable brain outline',opacity=.65,visible=False)
            self._usable_layer.editable=False
            try: self._usable_layer.contour=1
            except Exception: pass
        else:
            self._usable_layer.data=self.geometry.mask_fc.astype(np.uint8)
        self._usable_layer.visible=self.show_usable.isChecked()
        # Draw a one-pixel perimeter ONLY for ROI currently selected in the list.
        # RGBA outlines (rather than filled Napari Labels) do not obscure
        # cortical activity or broaden a narrow region with a 2-px contour.
        outlines = np.zeros((h, w, 4), dtype=np.float32)
        colors = mpl_colormaps['tab10']
        for j, (_, mask) in enumerate(self.masks):
            valid = np.asarray(mask, dtype=bool)
            if not valid.any():
                continue
            inside = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                               iterations=1, borderType=cv2.BORDER_CONSTANT,
                               borderValue=0).astype(bool)
            edge = valid & ~inside
            rgba = colors(j % 10)
            outlines[edge, 0] = rgba[0]
            outlines[edge, 1] = rgba[1]
            outlines[edge, 2] = rgba[2]
            outlines[edge, 3] = 0.95
        if self._boundary_layer is None or self._boundary_layer not in self.viewer.layers:
            self._boundary_layer = self.viewer.add_image(
                outlines, rgb=True, name='Selected ROI contours (1 px)',
                blending='translucent', visible=self.show_boundaries.isChecked(),
            )
        else:
            self._boundary_layer.data = outlines
        self._boundary_layer.visible=self.show_boundaries.isChecked()
        self._outside_layer.visible = True
        # Guarantee overlay layering: outside mask below boundaries, all above maps.
        for layer in (self._outside_layer,self._usable_layer,self._boundary_layer):
            if layer in self.viewer.layers:
                self.viewer.layers.move(self.viewer.layers.index(layer),len(self.viewer.layers)-1)

    def _change_image_visibility(self, *_):
        """Switch the visible Napari image between signal A and comparison B."""
        visible_b=bool(self.show_b.isChecked() and 'B' in self.products)
        if self._layer_a is not None: self._layer_a.visible = not visible_b
        if self._layer_b is not None: self._layer_b.visible = visible_b
        if self.products and self._layer_a is not None:
            self.viewer.status='Showing B' if visible_b else 'Showing A'

    def _frame_index(self):
        """Get the current bounded time index from the native Napari slider."""
        a=self.products.get('A')
        if a is None: return 0
        try:
            return int(np.clip(int(self.viewer.dims.current_step[0]),0,a.nframes-1))
        except Exception:
            return 0

    def _on_time_step(self,event=None):
        """Update the plot time cursor and status when the Napari frame changes."""
        if self._busy or not self.products: return
        a=self.products['A']
        k=self._frame_index()
        if self._cursor is not None:
            t=float(a.time[k])
            try:
                self._cursor.set_xdata([t,t])
                self.canvas.draw_idle()
            except Exception: pass
        if not self.fixed_scale.isChecked():
            self._apply_contrast()
        self._say(f'Time: {float(a.time[k]):.3f}s | frame {k+1}/{a.nframes} | {a.signal_var}')

    def _plot(self, *_):
        """Plot mean traces of selected ROI masks for signal A and optional B.
        
                Uses solid lines for A, dashed lines for B and optional dotted
                detrended-but-unfiltered A traces; coloring distinguishes ROI masks.
                Time is the original recording time in seconds."""
        self.figure.clear()
        ax=self.figure.add_subplot(111)
        self._plot_ax=ax
        if not self.products or not self.masks:
            ax.text(.5,.5,'Select one or several Step6b-accepted ROI and Load',
                    ha='center',va='center',transform=ax.transAxes)
        else:
            # Persist overlays as one trace for each of A/B × selected ROI.
            colors=mpl_colormaps['tab10']
            for ri,(label,mask) in enumerate(self.masks):
                color=colors(ri%10)
                for tag,style in (('A','-'),('B','--')):
                    product=self.products.get(tag)
                    if product is None: continue
                    key=(str(product.path_filter),product.local_start,product.local_stop,
                         label,False)
                    if key not in self.trace_cache:
                        self.trace_cache[key]=product.trace(mask)
                    value,n=self.trace_cache[key]
                    if n<1: continue
                    stride=max(1,len(value)//5500)
                    ax.plot(product.time[::stride],value[::stride],linewidth=1.0,
                            color=color,linestyle=style,label=f'{tag} {label} ({n}px)')
                    if tag=='A' and self.show_detrended.isChecked():
                        key_det=(*key[:-1],True)
                        if key_det not in self.trace_cache:
                            self.trace_cache[key_det]=product.trace(mask,detrended=True)
                        det,_=self.trace_cache[key_det]
                        ax.plot(product.time[::stride],det[::stride],linewidth=.65,
                                color=color,linestyle=':',alpha=.55)
            a=self.products['A']
            t=float(a.time[self._frame_index()])
            self._cursor=ax.axvline(t,color='#d62728',linewidth=1.2,alpha=.95)
            ax.set_xlim(a.time[0],a.time[-1] if a.nframes>1 else a.time[0]+1)
            ax.set_title('Accepted ROI mean traces | A solid · B dashed · A detrend dotted',fontsize=10)
            ax.legend(loc='upper right',fontsize=7,ncol=2)
        ax.set_xlabel('Recording time (s)')
        ax.set_ylabel('Hb signal (processed Zarr units)')
        ax.grid(linestyle=':',alpha=.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_plot_click(self,event):
        """Jump the Napari time slider to the sample nearest a clicked plot time."""
        if not self.products or event.inaxes is not self._plot_ax or event.xdata is None:
            return
        a=self.products['A']
        k=int(np.argmin(np.abs(a.time-float(event.xdata))))
        try: self.viewer.dims.set_current_step(0,k)
        except Exception: pass


def adjust_plot_dock_size(viewer, controller):
    """Give ROI traces ~45% of available vertical area, without resizing main window."""
    try:
        qmain = viewer.window._qt_window
        dock = controller.plot_dock
        available = qmain.height()
        target = min(550, max(350, int(available * 0.48)))
        qmain.resizeDocks([dock], [target], QtCore.Qt.Vertical)
    except (AttributeError, TypeError, RuntimeError):
        pass  # Works without private napari APIs on other versions.


def main(argv=None):
    """Start the read-only spontaneous-signal review GUI.
    
        Parameters
        ----------
        argv : sequence of str or None, optional
            Command-line arguments; None reads the process command line.
    
        Notes
        -----
        ``--config`` chooses the project YAML; without it a file picker opens.
        ``--cache-dir`` and ``--cache-backend`` control disposable disk storage.
        The GUI reads existing scientific results but never writes to them."""
    console_unicode_fallback()
    try: faulthandler.enable(all_threads=True)
    except Exception: pass
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='Project YAML; without it a Qt file picker opens')
    parser.add_argument('--cache-dir', type=Path, default=None,
                        help='Scratch drive for temporary movie cache (e.g. F:\\PT2021_signal_cache). '
                             'Automatically cleaned at shutdown.')
    parser.add_argument('--cache-backend', choices=['auto', 'hdf5', 'zarr'], default='auto',
                        help='Chunked disk backend; auto uses h5py if installed, otherwise Zarr.')
    args=parser.parse_args(argv)
    path=args.config
    if path is None:
        app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        filename,_=QtWidgets.QFileDialog.getOpenFileName(None,'Select PT2021 config.yaml','',
                                                          'YAML configuration (*.yaml *.yml)')
        if not filename: return
        path=Path(filename)
    viewer=napari.Viewer(title='PT2021 spontaneous signals — read only')
    try:
        controller=ReviewController(viewer,Path(path),
                                    cache_dir=args.cache_dir,
                                    cache_backend=args.cache_backend)
        viewer.window.add_dock_widget(controller,name='Spontaneous signal review',area='right')
        QtCore.QTimer.singleShot(200, lambda: adjust_plot_dock_size(viewer, controller))
    except Exception:
        traceback.print_exc()
        try: viewer.close()
        except Exception: pass
        raise
    napari.run()


if __name__=='__main__':
    main()
