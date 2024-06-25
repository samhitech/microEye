import os
import re
import traceback

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tifffile as tf

from microEye.analysis.checklist_dialog import ChecklistDialog
from microEye.analysis.fitting.nena import NeNA_Widget
from microEye.analysis.fitting.processing import plot_drift
from microEye.analysis.fitting.results import UNIQUE_COLUMNS, FittingResults
from microEye.analysis.fitting.results_stats import resultsStatsWidget
from microEye.analysis.fitting.tardis import TARDIS_Widget
from microEye.analysis.rendering import *
from microEye.analysis.tools.kymograms import KymogramWidget
from microEye.analysis.viewer.layers_widget import ImageParamsWidget
from microEye.qt import Qt, QtCore, QtGui, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.gui_helper import *
from microEye.utils.thread_worker import QThreadWorker


class LocalizationsView(QtWidgets.QWidget):
    '''
    A class for viewing and interacting with SMLM localizations in a PyQt5 application.
    '''

    def __init__(self, path: str, fittingResults: FittingResults = None):
        '''Initialize the StackView.

        Parameters
        ----------
        path : str
            The path to the image stack.
        fittingResults : FittingResults, optional
            Fitting results, by default None.
        '''
        super().__init__()

        self.setWindowTitle(path.split('/')[-1])

        self.path = path
        self._threadpool = QtCore.QThreadPool.globalInstance()
        # Initialize variables
        self.fittingResults = fittingResults

        self.main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.main_layout)

        # Graphics layout
        self.init_graphics()

        # Tab Widget
        self.tab_widget = QtWidgets.QTabWidget()

        self.main_layout.addWidget(self.tab_widget, 1)

        # Localization / Render tab layout
        self.setup_localization_tab()

        # Results stats tab layout
        self.setup_data_filters_tab()

        # Layers tab
        self.setup_layers_tab()

        for idx in range(2):
            image_item = self.add_image_item(
                self.empty_image if idx < 1 else self.empty_alpha, 1, 'SourceOver'
            )

            histogram_item = pg.HistogramLUTItem(
                gradientPosition='bottom', orientation='horizontal'
            )
            histogram_item.setImageItem(image_item)
            image_item.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_SourceOver
            )

            self.image_widget.addItem(histogram_item, row=1 + idx, col=0)
            self.histogram_items.append(histogram_item)

        if fittingResults:
            self.results_plot.setData(fittingResults.dataFrame())
            self.render_loc()

    def init_graphics(self):
        '''Initialize the graphics layout for the main window.'''
        # A plot area (ViewBox + axes) for displaying the image
        self.image_widget = pg.GraphicsLayoutWidget()
        self.image_widget.setMinimumWidth(600)

        # Create the ViewBox
        self.view_box: pg.ViewBox = self.image_widget.addViewBox(row=0, col=0)

        self.view_box.setAspectLocked(True)
        self.view_box.setAutoVisible(True)
        self.view_box.enableAutoRange()
        self.view_box.invertY(True)

        self.empty_image = np.zeros((128, 128), dtype=np.uint8)
        self.empty_alpha = np.zeros((128, 128, 4), dtype=np.uint8)
        self.empty_image[0, 0] = 255
        self.image_items: list[tuple[pg.ImageItem, np.ndarray]] = []
        self.histogram_items: list[pg.HistogramLUTItem] = []

        self.roi = pg.RectROI(
            [-8, 14], [6, 5], scaleSnap=True, translateSnap=True, movable=False
        )
        self.roi.addTranslateHandle([0, 0], [0.5, 0.5])
        self.view_box.addItem(self.roi)
        self.roi.setZValue(1000)
        # self.roi.sigRegionChangeFinished.connect(self.slider_changed)
        self.roi.setVisible(False)

        # Add the two sub-main layouts
        self.main_layout.addWidget(self.image_widget, 4)

    def add_image_item(
        self, image: np.ndarray, opacity: float = 1.0, compMode='SourceOver'
    ):
        """Add an image item to the view.

        Parameters
        ----------
        image : np.ndarray
            The image data.
        opacity : float, optional
            The opacity of the image, by default 1.0.
        compMode : str, optional
            The composition mode, by default 'SourceOver'.

        Returns
        -------
        pg.ImageItem
            The added image item.
        """
        # Create the ImageItem and set its view to self.view_box
        image_item = pg.ImageItem(
            image, axisOrder='row-major', opacity=max(min(opacity, 1), 0)
        )
        # Add the ImageItem to the ViewBox
        self.view_box.addItem(image_item)

        self.image_items.append([image_item, image])
        self.image_layers.add_layer(compMode)

        return image_item

    def setup_layers_tab(self):
        '''Set up the Layers tab.'''
        # Layers tab
        self.image_layers = ImageParamsWidget()

        self.image_layers.paramsChanged.connect(self.update_layer)

        self.tab_widget.addTab(self.image_layers, 'Layers')

    def update_layer(self, param, changes: list):
        '''Update the layer settings.

        Parameters
        ----------
        param : QParameter
            The parameter being changed.
        changes : list
            List of changes made.
        '''
        for param, _, data in changes:
            path = self.image_layers.param_tree.childPath(param)

            if len(path) > 1:
                parameter = path[-1]
                parent = path[-2]
            else:
                parameter = '.'.join(path) if path is not None else param.name()
                parent = None

            if parameter in ['Opacity', 'Visible']:
                idx = int(parent.split(' ')[-1]) - 1
                layer = self.image_layers.param_tree.param('Layers').child(parent)
                opacity = (
                    data if parameter == 'Opacity' else layer.child('Opacity').value()
                )
                visible = (
                    int(data)
                    if parameter == 'Visible'
                    else int(layer.child('Visible').value())
                )

                if idx < len(self.image_items):
                    self.image_items[idx][0].setOpts(opacity=visible * (opacity / 100))
            elif parameter == 'CompositionMode':
                idx = int(parent.split(' ')[-1]) - 1
                self.image_items[idx][0].setCompositionMode(
                    getattr(
                        QtGui.QPainter.CompositionMode, 'CompositionMode_' + str(data)
                    )
                )

    def render_loc(self):
        '''Update the rendered super-res image.

        Returns
        -------
        ndarray or None
            Rendered super-res image or None.
        '''
        if self.fittingResults is None:
            return None
        elif len(self.fittingResults) > 0:
            render_idx = self.render_cbox.currentData()
            if render_idx == 0:
                renderClass = hist2D_render(self.super_px_size.value())
            elif render_idx == 1:
                renderClass = gauss_hist_render(self.super_px_size.value())
            img = renderClass.render(*self.fittingResults.toRender())
            # TODO: here
            self.image_items[0][0].setImage(img, autoLevels=True)
            return img
        else:
            return None

    def render_tracks(self):
        pass

    def setup_localization_tab(self):
        '''Set up the Localization tab.'''
        # Render tab layout
        self.localization_widget, self.localization_form = create_widget(
            QtWidgets.QFormLayout
        )
        self.tab_widget.addTab(self.localization_widget, 'Localization')

        # Render layout

        self.render_cbox = QtWidgets.QComboBox()
        self.render_cbox.addItem('2D Histogram', 0)
        self.render_cbox.addItem('2D Gaussian Histogram', 1)

        self.super_px_size = create_spin_box(
            min_value=0, max_value=200, initial_value=10
        )

        self.refresh_btn = QtWidgets.QPushButton(
            'Refresh SuperRes Image', clicked=lambda: self.render_loc()
        )

        # Render GroupBox
        localization = QtWidgets.QGroupBox('Render')
        flocalization = QtWidgets.QFormLayout()
        localization.setLayout(flocalization)

        flocalization.addRow(QtWidgets.QLabel('Rendering Method:'), self.render_cbox)
        flocalization.addRow(
            QtWidgets.QLabel('S-res pixel-size [nm]:'), self.super_px_size
        )
        flocalization.addRow(self.refresh_btn)
        # End Localization GroupBox

        self.drift_cross_args = QtWidgets.QHBoxLayout()

        self.drift_cross_bins = create_spin_box(initial_value=10)
        self.drift_cross_px = create_spin_box(initial_value=10)
        self.drift_cross_up = create_spin_box(
            min_value=0, max_value=1000, initial_value=100
        )

        self.drift_cross_args.addWidget(self.drift_cross_bins)
        self.drift_cross_args.addWidget(self.drift_cross_px)
        self.drift_cross_args.addWidget(self.drift_cross_up)

        self.drift_cross_btn = QtWidgets.QPushButton(
            'Drift cross-correlation', clicked=lambda: self.drift_cross()
        )
        self.drift_fdm_btn = QtWidgets.QPushButton(
            'Fiducial marker drift correction', clicked=lambda: self.drift_fdm()
        )

        # Drift GroupBox
        drift = QtWidgets.QGroupBox('Drift Correction')
        fdrift = QtWidgets.QFormLayout()
        drift.setLayout(fdrift)

        fdrift.addRow(QtWidgets.QLabel('Drift X-Corr. (bins, pixelSize, upsampling):'))
        fdrift.addRow(self.drift_cross_args)
        fdrift.addRow(self.drift_cross_btn)
        fdrift.addRow(self.drift_fdm_btn)
        # End Drift GroupBox

        self.frc_cbox = QtWidgets.QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Odd/Even')
        self.frc_cbox.addItem('Halves')
        self.frc_res_btn = QtWidgets.QPushButton(
            'FRC Resolution', clicked=lambda: self.FRC_estimate()
        )

        self.NeNA_widget = None
        self.NeNA_btn = QtWidgets.QPushButton(
            'NeNA Loc. Prec. Estimate', clicked=lambda: self.NeNA_estimate()
        )
        self.tardis_btn = QtWidgets.QPushButton(
            'TARDIS', clicked=lambda: self.TARDIS_analysis()
        )

        # Precision GroupBox
        precision = QtWidgets.QGroupBox('Loc. Precision')
        fprecision = QtWidgets.QFormLayout()
        precision.setLayout(fprecision)

        fprecision.addRow(QtWidgets.QLabel('FRC Method:'), self.frc_cbox)
        fprecision.addWidget(self.frc_res_btn)
        fprecision.addWidget(self.NeNA_btn)
        fprecision.addWidget(self.tardis_btn)
        # End Precision GroupBox

        self.nneigh_merge_args = QtWidgets.QHBoxLayout()

        self.nn_neighbors = create_spin_box(max_value=20000, initial_value=1)
        self.nn_min_distance = create_double_spin_box(max_value=20000, initial_value=0)
        self.nn_max_distance = create_double_spin_box(max_value=20000, initial_value=30)
        self.nn_max_off = create_spin_box(max_value=20000, initial_value=1)
        self.nn_max_length = create_spin_box(max_value=20000, initial_value=500)

        self.nneigh_merge_args.addWidget(self.nn_neighbors)
        self.nneigh_merge_args.addWidget(self.nn_min_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_off)
        self.nneigh_merge_args.addWidget(self.nn_max_length)

        self.nn_layout = QtWidgets.QHBoxLayout()
        self.nneigh_btn = QtWidgets.QPushButton(
            'Nearest-neighbour', clicked=lambda: self.nneigh()
        )
        self.merge_btn = QtWidgets.QPushButton(
            'Merge Tracks', clicked=lambda: self.merge()
        )
        self.nneigh_merge_btn = QtWidgets.QPushButton(
            'NM + Merging', clicked=lambda: self.nneigh_merge()
        )

        self.nn_layout.addWidget(self.nneigh_btn)
        self.nn_layout.addWidget(self.merge_btn)
        self.nn_layout.addWidget(self.nneigh_merge_btn)

        # Precision GroupBox
        nearestN = QtWidgets.QGroupBox('NN Analysis')
        fnearestN = QtWidgets.QFormLayout()
        nearestN.setLayout(fnearestN)

        fnearestN.addRow(
            QtWidgets.QLabel('NN (n-neighbor, min, max-distance, max-off, max-len):')
        )
        fnearestN.addRow(self.nneigh_merge_args)
        fnearestN.addRow(self.nn_layout)
        # End Precision GroupBox

        self.export_options = ChecklistDialog(
            'Exported Columns',
            [
                'Super-res image',
            ]
            + UNIQUE_COLUMNS,
            checked=True,
            parent=self,
        )

        self.im_exp_layout = QtWidgets.QHBoxLayout()
        self.import_loc_btn = QtWidgets.QPushButton(
            'Import', clicked=lambda: self.import_loc()
        )
        self.export_loc_btn = QtWidgets.QPushButton(
            'Export', clicked=lambda: self.export_loc()
        )

        self.im_exp_layout.addWidget(self.import_loc_btn)
        self.im_exp_layout.addWidget(self.export_loc_btn)

        self.localization_form.addRow(localization)
        self.localization_form.addRow(drift)
        self.localization_form.addRow(precision)
        self.localization_form.addRow(nearestN)

        self.localization_form.addRow(self.im_exp_layout)

    def setup_data_filters_tab(self):
        '''Set up the Data Filters tab.'''
        # Results stats tab layout
        self.data_filters_widget, self.data_filters_layout = create_widget(
            QtWidgets.QVBoxLayout
        )
        self.tab_widget.addTab(self.data_filters_widget, 'Data Filters')

        # results stats widget
        self.results_plot_scroll = QtWidgets.QScrollArea()
        self.results_plot = resultsStatsWidget()
        self.results_plot.dataFilterUpdated.connect(self.filter_updated)
        self.results_plot_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.results_plot_scroll.setWidgetResizable(True)
        self.results_plot_scroll.setWidget(self.results_plot)

        self.apply_filters_btn = QtWidgets.QPushButton(
            'Apply Filters', clicked=lambda: self.apply_filters()
        )
        self.apply_filters_btn.setToolTip(
            'Applies the filters permanently to fitting results.'
        )
        self.zero_coords_btn = QtWidgets.QPushButton(
            'Zero Coordinates', clicked=lambda: self.zero_coordinates()
        )

        data_btn_layout = create_hbox_layout(
            self.apply_filters_btn, self.zero_coords_btn
        )

        self.data_filters_layout.addWidget(self.results_plot_scroll)
        self.data_filters_layout.addLayout(data_btn_layout)

    def FRC_estimate(self):
        '''Estimate FRC (Fourier Ring Correlation) resolution.'''
        frc_method = self.frc_cbox.currentText()
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            data = self.fittingResults.toRender()

            def work_func(**kwargs):
                try:
                    return FRC_resolution_binomial(
                        np.c_[data[0], data[1], data[2]],
                        self.super_px_size.value(),
                        frc_method,
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.frc_res_btn.setDisabled(False)
                if results is not None:
                    plotFRC(*results)
        else:
            return

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.frc_res_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def NeNA_estimate(self):
        '''Estimate Nearest-Neighbor Analysis for localization precision.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:

            def work_func(**kwargs):
                try:
                    return self.fittingResults.nn_trajectories(0, 200, 0, 1)
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.NeNA_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(self.fittingResults.dataFrame())

                    self.NeNA_widget = NeNA_Widget(
                        self.fittingResults.neighbour_dist, self.fittingResults.trackID
                    )

                    res = self.NeNA_widget.exec()

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.NeNA_btn.setDisabled(True)
            self._threadpool.start(self.worker)

    def TARDIS_analysis(self):
        '''Perform TARDIS analysis.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            self.tardis = TARDIS_Widget(
                self.fittingResults.frames,
                self.fittingResults.locX,
                self.fittingResults.locY,
                self.fittingResults.locZ,
            )
            self.tardis.startWorker.connect(
                lambda worker: self._threadpool.start(worker)
            )
            self.tardis.show()

    def drift_cross(self):
        '''Perform drift cross-correlation.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:

            def work_func(**kwargs):
                try:
                    return self.fittingResults.drift_cross_correlation(
                        self.drift_cross_bins.value(),
                        self.drift_cross_px.value(),
                        self.drift_cross_up.value(),
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.drift_cross_btn.setDisabled(False)
                if results is not None:
                    self.render_loc()
                    self.fittingResults = results[0]
                    self.results_plot.setData(self.fittingResults.dataFrame())
                    plot_drift(*results[2])

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.drift_cross_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def drift_fdm(self):
        '''Perform drift correction using fiducial markers.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:

            def work_func(**kwargs):
                try:
                    return self.fittingResults.drift_fiducial_marker()
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.drift_fdm_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results[0]
                    self.results_plot.setData(self.fittingResults.dataFrame())
                    plot_drift(*results[1])
                    self.render_loc()

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.drift_fdm_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def nneigh_merge(self):
        '''Perform nearest-neighbor merging.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:

            def work_func(**kwargs):
                try:
                    return self.fittingResults.nearest_neighbour_merging(
                        self.nn_min_distance.value(),
                        self.nn_max_distance.value(),
                        self.nn_max_off.value(),
                        self.nn_max_length.value(),
                        self.nn_neighbors.value(),
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.nneigh_merge_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(self.fittingResults.dataFrame())

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.nneigh_merge_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def nneigh(self):
        '''Perform nearest-neighbor analysis.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:

            def work_func(**kwargs):
                try:
                    return self.fittingResults.nn_trajectories(
                        self.nn_min_distance.value(),
                        self.nn_max_distance.value(),
                        self.nn_max_off.value(),
                        self.nn_neighbors.value(),
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.nneigh_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(self.fittingResults.dataFrame())

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.nneigh_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def merge(self):
        '''Merge tracks.'''
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:

            def work_func(**kwargs):
                try:
                    return self.fittingResults.merge_tracks(self.nn_max_length.value())
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.merge_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(self.fittingResults.dataFrame())

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.merge_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def apply_filters(self):
        '''Apply data filters to the fitting results.'''
        if self.results_plot.filtered is not None:
            self.fittingResults = FittingResults.fromDataFrame(
                self.results_plot.filtered, 1
            )
            self.results_plot.setData(self.fittingResults.dataFrame())
            self.render_loc()
            print('Filters applied.')

    def zero_coordinates(self):
        '''Zero the fitting results coordinates.'''
        if self.fittingResults is not None:
            self.fittingResults.zero_coordinates()
            self.results_plot.setData(self.fittingResults.dataFrame())
            self.render_loc()
            print('Coordinates reset.')

    def filter_updated(self, df: pd.DataFrame):
        '''Update the view when data filters are updated.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered DataFrame.
        '''
        if df is not None:
            if df.count().min() > 1:
                render_idx = self.render_cbox.currentData()
                if render_idx == 0:
                    renderClass = hist2D_render(self.super_px_size.value())
                elif render_idx == 1:
                    renderClass = gauss_hist_render(self.super_px_size.value())
                img = renderClass.render(
                    df['x'].to_numpy(), df['y'].to_numpy(), df['intensity'].to_numpy()
                )
                # img_norm = cv2.normalize(
                #     img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                self.image_items[0][0].setImage(img, autoLevels=False)
            else:
                # Create a black image
                img = np.zeros(
                    (
                        self.image_items[0][0].height(),
                        self.image_items[0][0].width(),
                        3,
                    ),
                    np.uint8,
                )

                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 50)
                fontScale = 1
                fontColor = (255, 255, 255)
                thickness = 1
                lineType = 2

                cv2.putText(
                    img,
                    'EMPTY!',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType,
                )
                self.image_items[0][0].setImage(img, autoLevels=False)

    def export_loc(self, filename=None):
        '''Export the fitting results into a file.

        Parameters
        ----------
        filename : str, optional
            File path; if None, a save file dialog is shown, by default None.
        '''
        if self.fittingResults is None:
            return

        if filename is None:
            if not self.export_options.exec():
                return

            filename, _ = getSaveFileName(
                self,
                'Export localizations',
                filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
                directory=os.path.dirname(self.path),
            )

        if len(filename) > 0:
            options = self.export_options.toList()

            dataFrame = self.fittingResults.dataFrame()
            exp_columns = []
            for col in dataFrame.columns:
                if col in options:
                    exp_columns.append(col)

            if '.tsv' in filename:
                dataFrame.to_csv(
                    filename,
                    index=False,
                    columns=exp_columns,
                    float_format=self.export_options.export_precision.text(),
                    sep='\t',
                    encoding='utf-8',
                )
            elif '.h5' in filename:
                dataFrame[exp_columns].to_hdf(
                    filename, key='microEye', index=False, complevel=0
                )

            if 'Super-res image' in options:
                sres_img = self.render_loc()
                tf.imsave(
                    re.sub(r'(\.h5|\.tsv)$', '_super_res.tif', filename),
                    sres_img,
                    photometric='minisblack',
                    bigtiff=True,
                    ome=False,
                )

    def import_loc(self):
        '''Import fitting results from a file.'''
        filename, _ = getOpenFileName(
            self,
            'Import localizations',
            filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
            directory=os.path.dirname(self.path),
        )

        if len(filename) > 0:
            results = FittingResults.fromFile(filename, 1)

            if results is not None:
                self.fittingResults = results
                self.results_plot.setData(results.dataFrame())
                self.render_loc()
                print('Done importing results.')
            else:
                print('Error importing results.')
