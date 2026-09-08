import logging
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pydoc import text
from weakref import ref

import cv2
import numpy as np
import ome_types as ome
import pyqtgraph as pg
import tifffile as tf
import tqdm
from pystackreg import StackReg
from skimage import transform
from tqdm import tqdm

from microEye import __version__
from microEye.analysis.clustering.plot import (
    ClusterPlotOptionsDialog,
    ClustersScatter,
    PlotResult,
    RulerPlotOptionsDialog,
    add_plot_result,
)
from microEye.analysis.fitting.results import DataColumns, FittingResults
from microEye.analysis.processing import FRC_resolution_binomial, plot_frc
from microEye.analysis.viewer.localizations import (
    LocalizationRenderHelper,
    LocalizationTab,
    plot_drift,
)
from microEye.hardware.stages import view
from microEye.images import ImageSequenceBase, TiffSeqHandler, create_array
from microEye.qt import Qt, QtCore, QtGui, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.display import DisplayManager, fast_autolevels_opencv
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


def display_frame(
    name,
    frame: np.ndarray,
    histogram: bool = True,
    autoLevels: bool = True,
    stats: bool = False,
):
    thresholds, hist, cdf = fast_autolevels_opencv(frame, levels=autoLevels)

    # cv2.imshow(name, frame)
    DisplayManager.instance().image_update_signal.emit(
        name,
        frame,
        {
            'aspect_ratio': 1.0,
            'width': frame.shape[1],
            'height': frame.shape[0],
            'show_stats': stats,
            'autoLevels': autoLevels,
            'isSingleView': False,
            'isROIsView': False,
            'isOverlaidView': True,
            'plot_type': False,
            'threshold': thresholds,
            'plot': hist if histogram else cdf,
        },
    )


def batch_cluster_dbscan(tables: list[FittingResults], **kwargs):
    try:
        with tqdm(
            total=len(tables), desc='Batch DBSCAN Clustering', unit='table'
        ) as pbar:
            for _, table in enumerate(tables):
                table.cluster_dbscan(**kwargs.get('dbscanOptions', {}))

                pbar.update(1)
    except Exception:
        traceback.print_exc()


def batch_drift_correction(tables: list[FittingResults], mode: str, **kwargs):
    try:
        keys = kwargs.get('keys', [])

        drift_args = {}

        with tqdm(
            total=len(tables), desc='Batch Drift Correction', unit='table'
        ) as pbar:
            for i, table in enumerate(tables):
                if mode == 'aim':
                    res = table.drift_aim(
                        segment_size=kwargs.get('aimSegmentSize'),
                        intersect_d=kwargs.get('aimIntersectD'),
                        z_scale=kwargs.get('aimZScale'),
                    )
                elif mode == 'cross correlation':
                    res = table.drift_cross_correlation(
                        n_bins=kwargs.get('crossBins'),
                        pixelSize=kwargs.get('crossPixelSize'),
                        upsampling=kwargs.get('crossUpSampling'),
                        rmax=kwargs.get('crossRMAX'),
                        method=kwargs.get('crossMethod'),
                    )
                else:
                    res = table.drift_fiducial_marker()

                if res is not None:
                    drift_args[keys[i] if keys else i] = res[1]

                pbar.update(1)

    except Exception:
        traceback.print_exc()

    return drift_args


def batch_export_localizations(tables: list[FittingResults], suffix: str, **kwargs):
    if suffix is None or suffix.strip() == '':
        suffix = 'exported'

    # use tqdm to show progress bar
    with tqdm(
        total=len(tables), desc='Batch Exporting localizations', unit='table'
    ) as pbar:
        for _, table in enumerate(tables):
            base_name = os.path.splitext(os.path.basename(table.path))[0]
            dir_name = os.path.dirname(table.path)
            file_path = os.path.join(dir_name, f'{base_name}_{suffix}.h5')

            table.dataFrame().to_hdf(
                file_path, key='microEye', index=False, complevel=0
            )

            pbar.update(1)


def batch_frc(tables: list[FittingResults], **kwargs):
    try:
        res = []

        frc_method = kwargs.get('frcMethod', 'binomial')
        frc_pixel_size = kwargs.get('binXY', 6)

        with tqdm(total=len(tables), desc='Batch FRC Analysis', unit='table') as pbar:
            for _, table in enumerate(tables):
                res.append(
                    FRC_resolution_binomial(
                        np.c_[
                            table.data[DataColumns.X],
                            table.data[DataColumns.Y],
                            table.data[DataColumns.INTENSITY],
                        ],
                        pixel_size=frc_pixel_size,
                        method=frc_method,
                    )
                )

                pbar.update(1)
    except Exception:
        traceback.print_exc()

    return res


def batch_loc_per_frame(tables: list[FittingResults], **kwargs):
    try:
        res: list[dict] = []

        with tqdm(total=len(tables), desc='Batch Locs per Frame', unit='table') as pbar:
            for _, table in enumerate(tables):
                res.append(table.extended_stats())

                pbar.update(1)
    except Exception:
        traceback.print_exc()

    return res


class BatchLocalizations(QtWidgets.QDialog):
    NAME = 'Batch Localizations Tool'

    def __init__(self, parent=None, loc_tables: dict[str, ref[FittingResults]] = None):
        super().__init__(parent)

        self.setMinimumWidth(500)

        self.loc_tables: dict[str, ref[FittingResults]] = (
            loc_tables if isinstance(loc_tables, dict) else {}
        )

        self.setWindowTitle(f'{self.NAME} - microEye v{__version__}')
        self.resize(1200, 720)

        self._init_layout()

        self._threadpool = QtCore.QThreadPool.globalInstance()
        self._worker = None  # Placeholder for the worker thread

        self._clustDlg = ClusterPlotOptionsDialog(self, batch_mode=True)
        self._rulerDlg = RulerPlotOptionsDialog(self, batch_mode=True)

        # timer to refresh display
        self._timer_interval = 100  # 10 FPS
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._timer_update)
        self._timer.setSingleShot(True)
        self._timer.start(self._timer_interval)

    def _init_layout(self):
        layout = QtWidgets.QHBoxLayout(self)
        self.setLayout(layout)

        # Add tables checklist
        self.tables_checklist = QtWidgets.QListWidget(self)
        self.tables_checklist.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        layout.addWidget(self.tables_checklist)

        # Populate the checklist with localization table names
        for name in self.loc_tables:
            item = QtWidgets.QListWidgetItem(name)
            self.tables_checklist.addItem(item)

        # Localization Tab
        self.localization_tab = LocalizationTab()

        for i in [3, 4]:
            self.localization_tab.setSectionVisible(i, False)

        for btn in ['__tardis_btn', '__NeNA_btn']:
            self.localization_tab.setBtnEnabled(btn, False)

        self.localization_tab.export_loc_btn.setVisible(True)
        self.localization_tab.import_loc_btn.setVisible(False)

        self.localization_tab.setSectionExpanded(5, True)

        self.localization_tab.exportActivated.connect(
            lambda: self.batch_export_localizations()
        )

        self.localization_tab.renderActivated.connect(
            lambda: self.batch_render_localizations()
        )

        self.localization_tab.driftCrossActivated.connect(
            lambda: self.batch_drift_correction(mode='Cross Correlation')
        )
        self.localization_tab.driftFiducialsActivated.connect(
            lambda: self.batch_drift_correction(mode='Fiducial Markers')
        )
        self.localization_tab.aimDriftCorrectionActivated.connect(
            lambda: self.batch_drift_correction(mode='AIM')
        )

        self.localization_tab.dbscanActivated.connect(
            lambda: self.batch_clustering_dbscan()
        )
        self.localization_tab.renderClustersActivated.connect(
            lambda: self.render_clusters()
        )
        self.localization_tab.renderRulersActivated.connect(
            lambda: self.render_ruler_clusters()
        )

        self.localization_tab.frcActivated.connect(lambda: self.batch_frc())

        layout.addWidget(self.localization_tab)

        self._init_context_menu()

    def _init_context_menu(self):
        self.context_menu = QtWidgets.QMenu(self)

        self.context_menu.addAction(
            'Select All', lambda: self.tables_checklist.selectAll()
        )
        self.context_menu.addAction(
            'Deselect All', lambda: self.tables_checklist.clearSelection()
        )
        self.context_menu.addSeparator()
        self.context_menu.addAction(
            'Extended Statistics', lambda: self.batch_extended_stats()
        )

        self.tables_checklist.setContextMenuPolicy(
            QtCore.Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.tables_checklist.customContextMenuRequested.connect(
            lambda pos: self.context_menu.exec(self.tables_checklist.mapToGlobal(pos))
        )

    @property
    def selected_names(self) -> list[str]:
        selected_items = self.tables_checklist.selectedItems()
        return [item.text() for item in selected_items]

    @property
    def selected_tables(self) -> list[FittingResults]:
        return [
            self.loc_tables[name]()
            for name in self.selected_names
            if name in self.loc_tables
        ]

    def runAsync(self, func, *args, **kwargs):
        selected_names = self.selected_names
        selected_tables = self.selected_tables

        if not selected_tables:
            return

        if len(selected_tables) <= 0:
            return

        kwargs['tables'] = selected_tables
        kwargs['keys'] = selected_names

        kwargs.update(self.localization_tab.toDict())

        done = kwargs.pop('done', lambda results: None)
        error = kwargs.pop('error', lambda e: None)

        def finished():
            self.localization_tab.setButtonsEnabled(True)
            logger.info(kwargs.pop('finished_message', 'Batch processing completed.'))

        self.worker = QThreadWorker(func, *args, **kwargs)
        self.worker.signals.result.connect(done)
        self.worker.signals.error.connect(error)
        self.worker.signals.finished.connect(finished)
        # Execute
        self.localization_tab.setButtonsEnabled(False)
        self._threadpool.start(self.worker)

    def batch_export_localizations(self):
        # dialog to get suffix
        suffix, ok = QtWidgets.QInputDialog.getText(
            self,
            'Export Localizations',
            'Enter suffix for exported files:',
            text='exported',
        )

        if not ok or not suffix.strip():
            return

        def done(results):
            QtWidgets.QMessageBox.information(
                self,
                'Export Completed',
                'Export completed for selected tables.',
            )

        self.runAsync(
            batch_export_localizations,
            done=done,
            suffix=suffix,
            finished_message='Export completed for selected tables.',
        )

    def batch_render_localizations(self):
        def done(results):
            if results is not None:
                images, aspects = results

                batch_rendered = pg.GraphicsLayoutWidget(
                    show=True, title='Batch Rendered Localizations'
                )
                batch_rendered.setBackground(None)

                image_items: list[pg.ImageItem] = []

                for idx, (rendered_img, aspect) in enumerate(zip(images, aspects)):
                    if rendered_img is not None:
                        image_item = pg.ImageItem(rendered_img, axisOrder='row-major')
                        image_items.append(image_item)

                        plot_item = batch_rendered.ci.addViewBox(
                            row=0, col=idx, invertY=True
                        )
                        plot_item.setAspectLocked(True, aspect)
                        plot_item.setAutoVisible(True)
                        plot_item.enableAutoRange()

                        plot_item.addItem(image_item)

                        if idx > 0:
                            plot_item.setXLink(batch_rendered.ci.getItem(0, 0))
                            plot_item.setYLink(batch_rendered.ci.getItem(0, 0))

                        # title_item = pg.TextItem(text)
                        # plot_item.addItem(title_item)

                if not image_items:
                    QtWidgets.QMessageBox.warning(
                        self,
                        'Warning',
                        'No images were rendered for the selected tables.',
                    )
                    return

                histogram_item = pg.HistogramLUTItem(
                    gradientPosition='bottom', orientation='horizontal'
                )
                histogram_item.setImageItem(image_items[0])
                batch_rendered.ci.addItem(
                    histogram_item, row=1, col=0, colspan=len(image_items)
                )

                def update_luts():
                    for item in image_items[1:]:
                        item.setLookupTable(histogram_item.getLookupTable(item.image))
                        item.setLevels(image_items[0].getLevels())

                histogram_item.sigLookupTableChanged.connect(update_luts)
                histogram_item.sigLevelsChanged.connect(update_luts)

                self.batch_rendered = batch_rendered
                batch_rendered.show()

        self.runAsync(
            LocalizationRenderHelper.batch_render_loc,
            done=done,
            finished_message='Rendering completed for selected tables.',
        )

    def batch_drift_correction(self, mode: str):
        def done(results):
            if results is None:
                return

            for key in results:
                if results[key] is None:
                    continue

                plot_drift(
                    *results[key],
                    title=f'{mode} Drift Correction',
                    subtitle=f'{key}',
                )

        self.runAsync(
            batch_drift_correction,
            mode=mode.lower(),
            done=done,
            finished_message='Drift correction completed for selected tables.',
        )

    def batch_frc(self):
        def done(results):
            if results is None:
                return

            frc_plots = pg.GraphicsLayoutWidget(show=True, title='FRC Analysis')
            frc_plots.setBackground(None)

            for i, result in enumerate(results):
                frequencies, FRC, smoothed, resolution = result
                items = plot_frc(
                    frequencies, FRC, smoothed, resolution, return_plot_items=True
                )

                plot = frc_plots.ci.addPlot(row=i // 3, col=i % 3)
                plot.setLabel('left', 'FRC', units='')
                plot.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')
                plot.setTitle(f'FRC resolution: {np.round(resolution, 1)} nm')
                for item in items:
                    plot.addItem(item)

            self.frc_plots = frc_plots
            frc_plots.show()

        self.runAsync(
            batch_frc,
            done=done,
            finished_message='FRC analysis completed for selected tables.',
        )

    def batch_extended_stats(self):
        def done(results: list[dict]):
            from matplotlib import pyplot as plt

            if results is None:
                return

            # violin plot of localizations per frame for each table
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            x_tick_labels = []

            for i, result in enumerate(results):
                if result is None:
                    continue

                # Get the table parent directory name not full path
                table_name = self.selected_names[i].split('[')[0].strip()
                if not table_name:
                    table_name = f'Dataset {i + 1}'

                counts_per_frame = result['Localizations per frame'].get('counts')

                if counts_per_frame is not None:
                    ax.violinplot(
                        counts_per_frame,
                        positions=[i],
                        showmeans=True,
                        showmedians=True,
                        showextrema=True,
                    )
                    x_tick_labels.append(table_name)

            ax.set_xticks(range(len(x_tick_labels)))
            ax.set_xticklabels(x_tick_labels)
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Localizations per Frame')
            ax.set_title('Extended Statistics')
            ax.set_xlim(-0.5, len(x_tick_labels) - 0.5)
            ax.set_ylim(bottom=0)

            ax.grid()
            fig.show()

        self.runAsync(
            batch_loc_per_frame,
            done=done,
            finished_message='Extended statistics completed for selected tables.',
        )

    def batch_clustering_dbscan(self):
        self.runAsync(
            batch_cluster_dbscan,
            finished_message='DBSCAN clustering completed for selected tables.',
        )

    def render_clusters(self):
        selected_tables = self.selected_tables

        if not selected_tables:
            return

        if len(selected_tables) <= 0:
            return

        if any(table.data[DataColumns.CLUSTER_ID] is None for table in selected_tables):
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'Please perform DBSCAN clustering before rendering.',
            )
            return

        res = self._clustDlg.exec()
        if not res:
            return

        options = self._clustDlg.values()

        def done(results):
            if results is not None:
                cluster_overlay_items = pg.GraphicsLayoutWidget(
                    show=True, title='DBSCAN'
                )
                cluster_overlay_items.setBackground(None)

                for i, result in enumerate(results):
                    column = i % 3
                    row = i // 3
                    add_plot_result(cluster_overlay_items, result, row, column)

                self.cluster_overlay_items = cluster_overlay_items
                cluster_overlay_items.show()

        self.runAsync(
            ClustersScatter.render_clusters_batch,
            finished_message='Cluster rendering completed for selected tables.',
            done=done,
            **options,
        )

    def render_ruler_clusters(self):
        selected_tables = self.selected_tables

        if not selected_tables:
            return

        if len(selected_tables) <= 0:
            return

        if any(table.data[DataColumns.CLUSTER_ID] is None for table in selected_tables):
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                'Please perform DBSCAN clustering before rendering.',
            )
            return

        res = self._rulerDlg.exec()
        if not res:
            return

        options = self._rulerDlg.values()

        worker_kwargs = {
            'leaf_size': self.localization_tab.toDict()['dbscanOptions']['leaf_size'],
            'algorithm': self.localization_tab.toDict()['dbscanOptions']['algorithm'],
            'scalebar_length': options['ruler_spacing'],
        }
        worker_kwargs.update(options)

        def done(results: list[PlotResult]):
            if results is not None:
                ruler_overlay_items = pg.GraphicsLayoutWidget(show=True, title='Rulers')
                ruler_overlay_items.setBackground(None)

                for i in range(len(results)):
                    add_plot_result(
                        ruler_overlay_items, results[i], 0, i, **worker_kwargs
                    )

                self.ruler_overlay_items = ruler_overlay_items
                ruler_overlay_items.show()

        self.runAsync(
            ClustersScatter.render_rulers_batch,
            finished_message='Ruler rendering completed for selected tables.',
            done=done,
            **worker_kwargs,
        )

    def _timer_update(self):
        self._timer.start(self._timer_interval)
