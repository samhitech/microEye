import contextlib
import logging
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
from microEye.analysis.fitting.processing import plot_animation_stats, plot_drift
from microEye.analysis.fitting.psf import stats
from microEye.analysis.fitting.results import (
    DataColumns,
    FittingResults,
)
from microEye.analysis.fitting.results_stats import resultsStatsWidget
from microEye.analysis.fitting.tardis import TARDIS_Widget
from microEye.analysis.processing import FRC_resolution_binomial, plot_frc
from microEye.analysis.rendering import *
from microEye.analysis.viewer.layers_widget import ImageItemsWidget
from microEye.analysis.viewer.volume import PointCloudViewer, VolumeViewerWindow
from microEye.qt import (
    Qt,
    QtCore,
    QtGui,
    QtWidgets,
    Signal,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.expandable_groupbox import SideBar
from microEye.utils.gui_helper import *
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


class LocalizationTab(SideBar):
    projectionChanged = Signal()

    renderActivated = Signal()
    render3DActivated = Signal()
    animationActivated = Signal()

    extractZActivated = Signal()

    driftCrossActivated = Signal()
    driftFiducialsActivated = Signal()

    frcActivated = Signal()
    nenaActivated = Signal()
    tardisActivated = Signal()

    nearestNeighActivated = Signal()
    mergeActivated = Signal()
    nn_mergeActivated = Signal()

    dbscanActivated = Signal()
    renderClustersActivated = Signal()

    importActivated = Signal()
    exportActivated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.__setup_ui()

    def __setup_ui(self):
        self.__init_render()

        self.__init_drift()

        self.__init_precision()

        self.__init_nearest_neighbors()

        self.__init_z_extraction()

        self.__init_dbscan_clustering()

        self.__init_buttons()

        self.addStretch()

    def __init_render(self):
        # Render layout
        self.__render_cbox = QtWidgets.QComboBox()
        self.__render_cbox.addItem('2D Histogram (Intensity)', 0)
        self.__render_cbox.addItem('2D Histogram (Events)', 1)
        self.__render_cbox.addItem('2D Gaussian Histogram', 2)

        self.__projection_cbox = QtWidgets.QComboBox()
        self.__projection_cbox.addItem('XY', 0)
        self.__projection_cbox.addItem('XY (slice)', 3)
        self.__projection_cbox.addItem('XZ', 1)
        self.__projection_cbox.addItem('XZ (slice)', 4)
        self.__projection_cbox.addItem('YZ', 2)
        self.__projection_cbox.addItem('YZ (slice)', 5)

        self.__projection_cbox.currentIndexChanged.connect(
            lambda: self.projectionChanged.emit()
        )

        self.__auto_level = QtWidgets.QCheckBox('Auto-stretch')
        self.__auto_level.setChecked(True)

        self.__frame_bins = create_spin_box(
            min_value=2, max_value=1000, initial_value=10
        )
        self.__xy_binsize = create_spin_box(
            min_value=0, max_value=250, initial_value=10
        )
        self.__z_binsize = create_spin_box(min_value=0, max_value=250, initial_value=50)

        self.__position_label = QtWidgets.QLabel('Position [nm]:')
        self.__position = create_double_spin_box(
            min_value=0, max_value=200, single_step=10, decimals=6, initial_value=0
        )
        self.__position_label.setVisible(False)
        self.__position.setVisible(False)
        self.__position.valueChanged.connect(lambda: self.renderActivated.emit())

        self.__refresh_btn = QtWidgets.QPushButton(
            'Refresh 2D View', clicked=lambda: self.renderActivated.emit()
        )

        self.__display_3d_btn = QtWidgets.QPushButton(
            '3D View', clicked=lambda: self.render3DActivated.emit()
        )

        self.__animation_btn = QtWidgets.QPushButton(
            'Animation', clicked=lambda: self.animationActivated.emit()
        )

        # Render GroupBox
        flocalization = QtWidgets.QFormLayout()

        flocalization.addRow('Rendering Method:', self.__render_cbox)
        flocalization.addRow('Projection:', self.__projection_cbox)
        flocalization.addRow('Frame bins:', self.__frame_bins)
        flocalization.addRow('XY binsize [nm]:', self.__xy_binsize)
        flocalization.addRow('Z binsize [nm]:', self.__z_binsize)
        flocalization.addRow(self.__position_label, self.__position)
        flocalization.addRow(
            create_hbox_layout(
                self.__auto_level,
                self.__refresh_btn,
                self.__display_3d_btn,
                self.__animation_btn,
            )
        )
        # End Localization GroupBox

        self.addSection('Render', flocalization, True)

    def __init_z_extraction(self):
        # Z Extraction
        self.z_cal_curve = None  # LEAVE IT!!!

        z_extract_form = QtWidgets.QFormLayout()

        self.__cal_curve_cbox = QtWidgets.QComboBox()
        for c in [stats.CurveFitMethod.LINEAR, stats.CurveFitMethod.CSPLINE]:
            self.__cal_curve_cbox.addItem(c.name, c)
        self.__cal_curve_cbox.currentIndexChanged.connect(
            lambda: [
                z_extract_form.setRowVisible(
                    idx, self.__cal_curve_cbox.currentIndex() == 0
                )
                for idx in [1, 2]
            ]
        )

        self.__z_cal_slope = create_double_spin_box(
            min_value=0, max_value=200, decimals=8, initial_value=0.000800867
        )
        self.__z_cal_intercept = create_double_spin_box(
            min_value=0, max_value=200, decimals=8, initial_value=1.00927
        )
        self.__load_cal_btn = QtWidgets.QPushButton(
            'Load Z Calibration', clicked=lambda: self.__load_calibration()
        )
        self.__plot_cal_btn = QtWidgets.QPushButton(
            'Calibration Curve', clicked=lambda: self.__plot_calibration()
        )
        # display loaded file path in multiline readonly edit box
        self.__cal_file_path = QtWidgets.QTextEdit()
        self.__cal_file_path.setReadOnly(True)
        self.__cal_file_path.setFixedHeight(60)
        self.__cal_file_path.setLineWrapMode(
            QtWidgets.QTextEdit.LineWrapMode.WidgetWidth
        )
        self.__cal_file_path.setPlaceholderText('No file loaded')

        self.__extract_z_btn = QtWidgets.QPushButton(
            'Extract Z', clicked=lambda: self.extractZActivated.emit()
        )
        z_extract_form.addRow(
            'Calibration Method:',
            self.__cal_curve_cbox,
        )
        z_extract_form.addRow(
            'Z Calibration (slope):',
            self.__z_cal_slope,
        )
        z_extract_form.addRow(
            'Z Calibration (intercept):',
            self.__z_cal_intercept,
        )
        z_extract_form.addRow(
            create_hbox_layout(
                self.__extract_z_btn, self.__load_cal_btn, self.__plot_cal_btn
            )
        )
        z_extract_form.addRow(self.__cal_file_path)
        # End Z Extraction

        self.addSection('Z Extraction', z_extract_form)

    def __init_drift(self):
        drift_cross_args = QtWidgets.QHBoxLayout()

        self.__drift_cross_bins = create_spin_box(initial_value=10)
        self.__drift_cross_px = create_spin_box(initial_value=10)
        self.__drift_cross_up = create_spin_box(
            min_value=0, max_value=1000, initial_value=100
        )
        self.__drift_rmax = create_spin_box(
            min_value=0, max_value=1000, initial_value=23
        )

        drift_cross_args.addWidget(self.__drift_cross_bins)
        drift_cross_args.addWidget(self.__drift_cross_px)
        drift_cross_args.addWidget(self.__drift_cross_up)
        drift_cross_args.addWidget(self.__drift_rmax)

        self.__cross_method_cbox = QtWidgets.QComboBox()
        self.__cross_method_cbox.addItem('Phase', 'phase')
        self.__cross_method_cbox.addItem('Direct', 'dcc')
        self.__cross_method_cbox.addItem('Mean', 'mcc')
        self.__cross_method_cbox.addItem('Redundant', 'rcc')

        self.__drift_cross_btn = QtWidgets.QPushButton(
            'Drift X-Corr', clicked=lambda: self.driftCrossActivated.emit()
        )
        self.__drift_fdm_btn = QtWidgets.QPushButton(
            'Fiducial markers', clicked=lambda: self.driftFiducialsActivated.emit()
        )

        # Drift GroupBox
        fdrift = QtWidgets.QFormLayout()

        fdrift.addRow(
            QtWidgets.QLabel('Drift X-Corr. (bins, pixelSize, upsampling, rmax):')
        )
        fdrift.addRow(drift_cross_args)
        fdrift.addRow(
            create_hbox_layout(
                self.__cross_method_cbox, self.__drift_cross_btn, self.__drift_fdm_btn
            )
        )
        # End Drift GroupBox

        self.addSection('Drift Correction', fdrift, True)

    def __init_precision(self):
        self.__frc_cbox = QtWidgets.QComboBox()
        self.__frc_cbox.addItem('Binomial')
        self.__frc_cbox.addItem('Odd/Even')
        self.__frc_cbox.addItem('Halves')

        precision_btns = QtWidgets.QHBoxLayout()
        self.__frc_res_btn = QtWidgets.QPushButton(
            'FRC Resolution', clicked=lambda: self.frcActivated.emit()
        )

        self.__NeNA_btn = QtWidgets.QPushButton(
            'NeNA Loc. Prec. Estimate',
            clicked=lambda: self.nenaActivated.emit(),
        )
        self.__tardis_btn = QtWidgets.QPushButton(
            'TARDIS', clicked=lambda: self.tardisActivated.emit()
        )

        # Precision GroupBox
        fprecision = QtWidgets.QFormLayout()

        fprecision.addRow(QtWidgets.QLabel('FRC Method:'), self.__frc_cbox)
        precision_btns.addWidget(self.__frc_res_btn)
        precision_btns.addWidget(self.__NeNA_btn)
        precision_btns.addWidget(self.__tardis_btn)
        fprecision.addRow(precision_btns)
        # End Precision GroupBox

        self.addSection('Loc. Precision', fprecision, True)

    def __init_nearest_neighbors(self):
        nneigh_merge_args = QtWidgets.QHBoxLayout()

        self.__nn_neighbors = create_spin_box(max_value=20000, initial_value=1)
        self.__nn_min_distance = create_double_spin_box(
            max_value=20000, initial_value=0
        )
        self.__nn_max_distance = create_double_spin_box(
            max_value=20000, initial_value=30
        )
        self.__nn_max_off = create_spin_box(max_value=20000, initial_value=1)
        self.__nn_max_length = create_spin_box(max_value=20000, initial_value=500)

        nneigh_merge_args.addWidget(self.__nn_neighbors)
        nneigh_merge_args.addWidget(self.__nn_min_distance)
        nneigh_merge_args.addWidget(self.__nn_max_distance)
        nneigh_merge_args.addWidget(self.__nn_max_off)
        nneigh_merge_args.addWidget(self.__nn_max_length)

        nn_layout = QtWidgets.QHBoxLayout()
        self.__nneigh_btn = QtWidgets.QPushButton(
            'Nearest-neighbour',
            clicked=lambda: self.nearestNeighActivated.emit(),
        )
        self.__merge_btn = QtWidgets.QPushButton(
            'Merge Tracks', clicked=lambda: self.mergeActivated.emit()
        )
        self.__nneigh_merge_btn = QtWidgets.QPushButton(
            'NM + Merging',
            clicked=lambda: self.nn_mergeActivated.emit(),
        )

        nn_layout.addWidget(self.__nneigh_btn)
        nn_layout.addWidget(self.__merge_btn)
        nn_layout.addWidget(self.__nneigh_merge_btn)

        # NN GroupBox
        fnearestN = QtWidgets.QFormLayout()

        fnearestN.addRow(
            QtWidgets.QLabel('NN (n-neighbor, min, max-distance, max-off, max-len):')
        )
        fnearestN.addRow(nneigh_merge_args)
        fnearestN.addRow(nn_layout)
        # End NN GroupBox

        self.addSection('NN Analysis', fnearestN, True)

    def __init_dbscan_clustering(self):
        clustering = QtWidgets.QFormLayout()

        self.__dbscan_eps = QtWidgets.QDoubleSpinBox()
        self.__dbscan_eps.setMinimum(0)
        self.__dbscan_eps.setMaximum(5000)
        self.__dbscan_eps.setValue(50)

        self.__dbscan_min_samples = QtWidgets.QSpinBox()
        self.__dbscan_min_samples.setMinimum(2)
        self.__dbscan_min_samples.setMaximum(1000)
        self.__dbscan_min_samples.setValue(5)

        self.__dbscan_metric = QtWidgets.QComboBox()
        metrics = [
            'euclidean',
        ]

        for metric in metrics:
            self.__dbscan_metric.addItem(metric)

        self.__dbscan_algorithm = QtWidgets.QComboBox()
        algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for alg in algos:
            self.__dbscan_algorithm.addItem(alg)

        self.__dbscan_leaf_size = QtWidgets.QSpinBox()
        self.__dbscan_leaf_size.setMinimum(0)
        self.__dbscan_leaf_size.setMaximum(1000)
        self.__dbscan_leaf_size.setValue(30)

        self.__dbscan_scale_data = QtWidgets.QCheckBox('Scale Data ?')

        self.__dbscan_btn = QtWidgets.QPushButton(
            'DBSCAN', clicked=lambda: self.dbscanActivated.emit()
        )
        self.__dbscan_render_btn = QtWidgets.QPushButton(
            'Show Clusters',
            clicked=lambda: self.renderClustersActivated.emit(),
        )

        clustering.addRow(QtWidgets.QLabel('Epsilon'), self.__dbscan_eps)
        clustering.addRow(QtWidgets.QLabel('Min Samples'), self.__dbscan_min_samples)
        clustering.addRow(QtWidgets.QLabel('Leaf Size'), self.__dbscan_leaf_size)
        # clustering.addRow(QtWidgets.QLabel('Metric'), self.__dbscan_metric)
        clustering.addRow(QtWidgets.QLabel('Algorithm'), self.__dbscan_algorithm)
        clustering.addWidget(self.__dbscan_scale_data)
        clustering.addWidget(self.__dbscan_btn)
        clustering.addWidget(self.__dbscan_render_btn)

        self.addSection('DBSCAN Clustering', clustering, False)

    def __init_buttons(self):
        self.import_loc_btn = QtWidgets.QPushButton(
            'Import', clicked=lambda: self.importActivated.emit()
        )
        self.export_loc_btn = QtWidgets.QPushButton(
            'Export', clicked=lambda: self.exportActivated.emit()
        )

        self.addLayout(create_hbox_layout(self.import_loc_btn, self.export_loc_btn))

    def __load_calibration(self):
        '''Load a calibration file.'''
        self.z_cal_curve, path = stats.import_fit_curve(
            self,
            os.path.dirname(self.currentResults.path),
        )
        if self.z_cal_curve is not None:
            self.__cal_file_path.setText(path)
            if self.z_cal_curve.method == stats.CurveFitMethod.LINEAR:
                self.__cal_file_path.setText(path)
                self.__cal_curve_cbox.setCurrentIndex(0)
                self.__z_cal_slope.setValue(self.z_cal_curve.slope)
                self.__z_cal_intercept.setValue(self.z_cal_curve.intercept)
            elif self.z_cal_curve.method == stats.CurveFitMethod.CSPLINE:
                self.__cal_file_path.setText(path)
                self.__cal_curve_cbox.setCurrentIndex(1)

    def __plot_calibration(self):
        '''Plot the calibration curve.'''
        if self.z_cal_curve is not None:
            plt = pg.plot()
            plt.showGrid(x=True, y=True)
            legend = plt.addLegend()
            legend.anchor(itemPos=(1, 0), parentPos=(1, 0))
            plt.setWindowTitle('Z Calibration')
            plt.setLabel('left', 'Sigma Ratio (X/Y)')
            plt.setLabel('bottom', 'Z [nm]')

            # Zero plane line
            plt.plotItem.addLine(x=0, pen='w')

            if self.z_cal_curve.method == stats.CurveFitMethod.LINEAR:
                pen = pg.mkPen(color=(0, 0, 255, 150), width=2)
                plt.plot(
                    self.z_cal_curve.data['z'],
                    self.z_cal_curve.data['ratio'],
                    pen=pen,
                    name='data',
                )
                pen = pg.mkPen(color=(0, 255, 0, 150), width=2)
                plt.plot(
                    self.z_cal_curve.data['z'],
                    np.array(self.z_cal_curve.data['z']) * self.z_cal_curve.slope
                    + self.z_cal_curve.intercept,
                    pen=pen,
                    name='fit',
                )

                # Confidence interval fill
                fill = pg.FillBetweenItem(
                    pg.PlotCurveItem(
                        self.z_cal_curve.data['z'],
                        self.z_cal_curve.data['lower_bounds'],
                    ),
                    pg.PlotCurveItem(
                        self.z_cal_curve.data['z'],
                        self.z_cal_curve.data['upper_bounds'],
                    ),
                    brush=pg.mkBrush((0, 255, 0, 50)),  # More transparent for fill
                )
                plt.addItem(fill)

                plt.show()
            else:
                pen = pg.mkPen(color=(0, 0, 255, 150), width=2)
                plt.plot(
                    self.z_cal_curve.parameters['x_data'],
                    self.z_cal_curve.parameters['y_data'],
                    pen=pen,
                    name='data',
                )
                pen = pg.mkPen(color=(0, 255, 0, 150), width=2)
                plt.plot(
                    self.z_cal_curve.parameters['x_data'],
                    self.z_cal_curve.get_data(self.z_cal_curve.parameters['x_data']),
                    pen=pen,
                    name='fit',
                )

                plt.show()

    def update_position(self, currentResults: FittingResults):
        '''Update the position label.'''
        if currentResults is None or self.__projection_cbox.currentData() not in [
            3,
            4,
            5,
        ]:
            self.__position_label.setVisible(False)
            self.__position.setVisible(False)
            return

        projection = self.__projection_cbox.currentData() - 3
        column = (
            DataColumns.Z
            if projection == 0
            else DataColumns.Y
            if projection == 1
            else DataColumns.X
        )

        coord_data = currentResults.data.get(column)
        if coord_data is None or len(coord_data) == 0:
            self.__position_label.setVisible(False)
            self.__position.setVisible(False)
            return

        self.__position_label.setVisible(True)
        self.__position.setVisible(True)
        self.__position_label.setText('Z Position [nm]:')
        self.__position.setRange(coord_data.min(), coord_data.max())
        self.__position.setValue(self.__position.minimum())

    @property
    def bin_frames(self) -> int:
        return self.__frame_bins.value()

    @property
    def bin_xy(self) -> int:
        return self.__xy_binsize.value()

    @property
    def bin_z(self) -> int:
        return self.__z_binsize.value()

    @property
    def slicePosition(self) -> float:
        return self.__position.value()

    @property
    def projection(self) -> int:
        return self.__projection_cbox.currentData()

    @property
    def renderMode(self):
        return RenderModes(self.__render_cbox.currentData())

    @property
    def autoStretch(self):
        return self.__auto_level.isChecked()

    @property
    def frcMethod(self) -> str:
        return self.__frc_cbox.currentText()

    @property
    def crossMethod(self):
        return self.__cross_method_cbox.currentData()

    @property
    def crossLabel(self) -> str:
        return self.__cross_method_cbox.currentText()

    @property
    def crossBins(self) -> int:
        return self.__drift_cross_bins.value()

    @property
    def crossPixelSize(self) -> int:
        return self.__drift_cross_px.value()

    @property
    def crossUpSampling(self) -> int:
        return self.__drift_cross_up.value()

    @property
    def crossRMAX(self) -> int:
        return self.__drift_rmax.value()

    @property
    def nnMinDistance(self):
        return self.__nn_min_distance.value()

    @property
    def nnMaxDistance(self):
        return self.__nn_max_distance.value()

    @property
    def nnMaxOff(self):
        return self.__nn_max_off.value()

    @property
    def nnMaxLength(self):
        return self.__nn_max_length.value()

    @property
    def nearestNeighbors(self):
        return self.__nn_neighbors.value()

    @property
    def curveFitMethod(self):
        return self.__cal_curve_cbox.currentData()

    @property
    def zCalIntercept(self):
        return self.__z_cal_intercept.value()

    @property
    def zCalSlope(self):
        return self.__z_cal_slope.value()

    @property
    def dbscanEps(self):
        return self.__dbscan_eps.value()

    @property
    def dbscanMinSamples(self):
        return self.__dbscan_min_samples.value()

    @property
    def dbscanLeafSize(self):
        return self.__dbscan_leaf_size.value()

    @property
    def dbscanMetric(self):
        return self.__dbscan_metric.currentText()

    @property
    def dbscanAlgorithm(self):
        return self.__dbscan_algorithm.currentText()

    @property
    def dbscanScaleData(self):
        return self.__dbscan_scale_data.isChecked()

    def setButtonsEnabled(self, enabled=True):
        self.__refresh_btn.setEnabled(enabled)
        self.__display_3d_btn.setEnabled(enabled)
        self.__extract_z_btn.setEnabled(enabled)
        self.__load_cal_btn.setEnabled(enabled)
        self.__plot_cal_btn.setEnabled(enabled)
        self.__drift_cross_btn.setEnabled(enabled)
        self.__drift_fdm_btn.setEnabled(enabled)
        self.__nneigh_btn.setEnabled(enabled)
        self.__merge_btn.setEnabled(enabled)
        self.__nneigh_merge_btn.setEnabled(enabled)
        self.__frc_res_btn.setEnabled(enabled)
        self.__NeNA_btn.setEnabled(enabled)
        self.__tardis_btn.setEnabled(enabled)
        self.__animation_btn.setEnabled(enabled)
        self.__dbscan_btn.setEnabled(enabled)
        self.__dbscan_render_btn.setEnabled(enabled)
        self.import_loc_btn.setEnabled(enabled)
        self.export_loc_btn.setEnabled(enabled)


class ClusterPlotOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Cluster Plot Options')

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItem('Full FOV', 'fov')
        self.mode_box.addItem('Superimposed', 'superimposed')
        self.mode_box.setCurrentIndex(1)

        self.is_rasterized = QtWidgets.QCheckBox('Rasterize?')
        self.is_rasterized.setChecked(True)

        self.render_box = QtWidgets.QComboBox()
        self.render_box.addItem('Histogram', RenderModes.HISTOGRAM)
        self.render_box.addItem('Event Histogram', RenderModes.EVENT_HISTOGRAM)
        self.render_box.addItem('Gaussian', RenderModes.GAUSSIAN)

        self.pixel_size = create_double_spin_box(
            min_value=0.1, max_value=50, decimals=2, initial_value=0.5
        )

        layout = QtWidgets.QFormLayout(self)
        layout.addRow('Mode:', self.mode_box)
        layout.addWidget(self.is_rasterized)
        layout.addRow('Render mode:', self.render_box)
        layout.addRow('Pixel size:', self.pixel_size)

        self.is_rasterized.checkStateChanged.connect(self.is_rasterized_toggled)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def is_rasterized_toggled(self):
        layout: QtWidgets.QFormLayout = self.layout()

        layout.setRowVisible(2, self.is_rasterized.isChecked())
        layout.setRowVisible(3, self.is_rasterized.isChecked())

    def values(self):
        return {
            'mode': self.mode_box.currentData(),
            'rasterized': self.is_rasterized.isChecked(),
            'render_mode': self.render_box.currentData(),
            'pixel_size': self.pixel_size.value(),
        }


class LocalizationsView(QtWidgets.QWidget):
    '''
    A class for viewing and interacting with SMLM localizations in a PyQt5 application.
    '''

    __INSTANCES = 0

    def __init__(self, fittingResults: FittingResults = None):
        '''Initialize the StackView.

        Parameters
        ----------
        path : str
            The path to the image stack.
        fittingResults : FittingResults, optional
            Fitting results, by default None.
        '''
        super().__init__()

        LocalizationsView.__INSTANCES += 1
        self.setWindowTitle(f'Loc Viewer {LocalizationsView.__INSTANCES:04d}')

        self._threadpool = QtCore.QThreadPool.globalInstance()
        # Initialize variables
        self.fittingResults: list[FittingResults] = []

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

        if fittingResults:
            self.add_data_layer(fittingResults)

    @property
    def currentResults(self) -> FittingResults:
        '''Get the current fitting results.'''
        if self.image_layers.currentIndex < 0:
            return None

        return self.fittingResults[self.image_layers.currentIndex]

    @currentResults.setter
    def currentResults(self, value: FittingResults):
        '''Set the current fitting results.'''
        if self.image_layers.currentIndex < 0:
            return

        value.colormap = self.currentResults.colormap
        value.intensity_range = self.currentResults.intensity_range

        self.fittingResults[self.image_layers.currentIndex] = value

    def setImage(self, image: np.ndarray, autoLevels: bool = True, **kwargs):
        '''Set the image data.'''
        if image is None or self.image_layers.currentLayer is None:
            return

        index = kwargs.get('index', self.image_layers.currentIndex)

        colorMap = self.fittingResults[index].colormap
        if colorMap is not None:
            kwargs['colorMap'] = colorMap

        self.image_layers.getImageItemAt(index).setImage(
            image, autoLevels=autoLevels, **kwargs
        )

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

        # histogram item
        self.histogram_item = pg.HistogramLUTItem(
            gradientPosition='bottom', orientation='horizontal'
        )

        self.image_widget.addItem(self.histogram_item, row=1, col=0)

        # Add ROI
        self.roi = pg.RectROI(
            [-8, 14], [6, 5], scaleSnap=True, translateSnap=True, movable=False
        )
        self.roi.addTranslateHandle([0, 0], [0.5, 0.5])
        self.view_box.addItem(self.roi)
        self.roi.setZValue(1000)
        self.roi.setVisible(False)

        # Add LinePlotItem for trajectories
        self.trajectory_item = pg.PlotDataItem(pen=pg.mkPen(color='r', width=2))
        self.view_box.addItem(self.trajectory_item)
        self.trajectory_item.setZValue(999)  # Ensure lines are on top of image

        # Add the two sub-main layouts
        self.main_layout.addWidget(self.image_widget, 4)

    def add_data_layer(self, data: FittingResults):
        if data is None:
            return

        self.fittingResults.append(data)

        self.updateStatsLabel()

        count = self.image_layers.count()
        # Get the last part of the path
        path = (
            data.path
            if len(data.path.split('/')) < 3
            else '/'.join(data.path.split('/')[-2:])
        )

        image_item = self.add_image_item(
            self.empty_image if count < 1 else self.empty_alpha,
            compMode='SourceOver' if count < 1 else 'Plus',
            name=path,
        )

        self.render_loc()

    def add_image_item(
        self,
        image: np.ndarray,
        opacity: float = 1.0,
        compMode='SourceOver',
        name='Layer',
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
        name : str, optional
            The name of the image item layer, by default 'Layer'.

        Returns
        -------
        pg.ImageItem
            The added image item.
        """
        # Create the ImageItem and set its view to self.view_box
        image_item = self.image_layers.add_layer(image, opacity, compMode, name=name)

        # Add the ImageItem to the ViewBox
        self.view_box.addItem(image_item)

        return image_item

    def setup_layers_tab(self):
        '''Set up the Layers tab.'''
        # Layers tab
        self.image_layers = ImageItemsWidget()

        self.image_layers.layerChanged.connect(self.update_layer)
        self.image_layers.layerRemoved.connect(
            lambda index: self.fittingResults.pop(index)
        )

        self.tab_widget.addTab(self.image_layers, 'Layers')

    def update_layer(self, current: int, previous: int, imageItem: pg.ImageItem):
        '''Update the layer settings.

        Parameters
        ----------
        current : int
            The current layer index.
        previous : int
            The previous layer index.
        imageItem : pg.ImageItem
            The image item.
        '''
        if current == -1 or imageItem is None:
            return

        if previous >= 0 and previous < len(self.fittingResults):
            self.fittingResults[
                previous
            ].colormap = self.histogram_item.gradient.colorMap()
            self.fittingResults[
                previous
            ].intensity_range = self.histogram_item.getLevels()

        self.histogram_item.setImageItem(imageItem)

        if self.currentResults.colormap is not None:
            self.histogram_item.gradient.setColorMap(self.currentResults.colormap)
            self.histogram_item.setLevels(*self.currentResults.intensity_range)

        self.results_plot.clear()

    def update_position(self):
        '''Update the position label.'''
        self.localization_widget.update_position(self.currentResults)

    def render_loc(self):
        '''Update the rendered super-res image.

        Returns
        -------
        ndarray or None
            Rendered super-res image or None.
        '''
        return LocalizationRenderHelper.render_loc(self)

    def toAnimation(self):
        '''
        Render an animation of the super-res image.
        '''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        filename = self.currentResults.path

        if filename is None:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.toAnimation(
                    self.localization_widget.bin_frames,
                    self.localization_widget.bin_xy,
                    z_pixel_size=self.localization_widget.bin_z,
                    projection=Projection(self.localization_widget.projection),
                    renderMode=self.localization_widget.renderMode,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.localization_widget.setButtonsEnabled(True)
            # results might be tuple of three nones
            if results is not None and all([r is not None for r in results]):
                stack, frame_bins, locs_per_bin = results

                # save stack to tiff to the same directory as the fitting results
                tf.imwrite(
                    re.sub(r'(\.h5|\.tsv)$', '_animation.tif', filename),
                    stack,
                    photometric='minisblack',
                    bigtiff=True,
                    ome=False,
                )

                plot_animation_stats(frame_bins, locs_per_bin)

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.localization_widget.setButtonsEnabled(False)
        self._threadpool.start(self.worker)

        self.currentResults.toAnimation()

    def render_tracks(self):
        # self.trajectory_item.setData(
        #     x=np.linspace(0, 10, 100),
        #     y=np.sin(np.linspace(0, 10, 100))
        # )
        pass

    def setup_localization_tab(self):
        '''Set up the Localization tab.'''
        # Render tab layout
        self.localization_widget = LocalizationTab()
        self.tab_widget.addTab(self.localization_widget, 'Localization')

        self.NeNA_widget = None

        self.export_options = ChecklistDialog(
            'Exported Columns',
            [
                'Super-res image',
            ]
            + DataColumns.unique_columns(),
            checked=True,
            parent=self,
        )

        self.localization_widget.projectionChanged.connect(
            lambda: self.update_position()
        )
        self.localization_widget.renderActivated.connect(lambda: self.render_loc())
        self.localization_widget.render3DActivated.connect(
            lambda: LocalizationRenderHelper.render_3D(self)
        )
        self.localization_widget.animationActivated.connect(lambda: self.toAnimation())

        self.localization_widget.extractZActivated.connect(
            lambda: LocalizationProcessing.extract_z(self)
        )

        self.localization_widget.driftCrossActivated.connect(
            lambda: LocalizationProcessing.drift_cross(self)
        )
        self.localization_widget.driftFiducialsActivated.connect(
            lambda: LocalizationProcessing.drift_fdm(self)
        )

        self.localization_widget.frcActivated.connect(
            lambda: LocalizationProcessing.FRC_estimate(self)
        )
        self.localization_widget.nenaActivated.connect(
            lambda: LocalizationProcessing.NeNA_estimate(self)
        )
        self.localization_widget.tardisActivated.connect(
            lambda: LocalizationProcessing.TARDIS_analysis(self)
        )

        self.localization_widget.nearestNeighActivated.connect(
            lambda: LocalizationProcessing.nneigh(self)
        )
        self.localization_widget.mergeActivated.connect(
            lambda: LocalizationProcessing.merge(self)
        )
        self.localization_widget.nn_mergeActivated.connect(
            lambda: LocalizationProcessing.nneigh_merge(self)
        )

        self.localization_widget.dbscanActivated.connect(
            lambda: LocalizationProcessing.dbscan_clustering(self)
        )
        self.localization_widget.renderClustersActivated.connect(
            lambda: LocalizationRenderHelper.render_clusters(self)
        )

        self.localization_widget.importActivated.connect(
            lambda: LocalizationIO.import_loc(self)
        )
        self.localization_widget.exportActivated.connect(
            lambda: LocalizationIO.export_loc(self)
        )

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

        self.populate_data_btn = QtWidgets.QPushButton(
            'Populate Data', clicked=lambda: self.populate_data()
        )
        self.apply_filters_btn = QtWidgets.QPushButton(
            'Apply Filters', clicked=lambda: LocalizationProcessing.apply_filters(self)
        )
        self.apply_filters_btn.setToolTip(
            'Applies the filters permanently to fitting results.'
        )
        self.zero_coords_btn = QtWidgets.QPushButton(
            'Zero Coordinates',
            clicked=lambda: LocalizationProcessing.zero_coordinates(self),
        )
        self.toggle_tracking_btn = QtWidgets.QPushButton(
            'Track Plots', clicked=lambda: self.toggle_track_plots()
        )

        data_btn_layout = create_hbox_layout(
            self.populate_data_btn,
            self.apply_filters_btn,
            self.zero_coords_btn,
            self.toggle_tracking_btn,
        )

        # create x/y/z coordinate shift
        self.data_shift = create_double_spin_box(
            min_value=1, max_value=10000, decimals=0, initial_value=200
        )
        self.data_shift_label = QtWidgets.QLabel('Data Shift (nm)')
        # axis to shift
        self.data_shift_axis = QtWidgets.QComboBox()
        self.data_shift_axis.addItem('X', DataColumns.X)
        self.data_shift_axis.addItem('Y', DataColumns.Y)
        self.data_shift_axis.addItem('Z', DataColumns.Z)
        self.data_shift_axis.setToolTip('Select the axis to shift the data.')
        # direction to shift
        self.data_shift_dir = QtWidgets.QCheckBox('Negative')
        self.data_shift_dir.setToolTip('Shift the data in the negative direction.')
        # apply shift button
        self.data_shift_btn = QtWidgets.QPushButton(
            'Shift Data', clicked=lambda: LocalizationProcessing.shift_data(self)
        )
        self.data_shift_btn.setToolTip('Shift the data by the specified value.')

        data_shift_layout = create_hbox_layout(
            self.data_shift_label,
            self.data_shift,
            self.data_shift_axis,
            self.data_shift_dir,
            self.data_shift_btn,
        )

        # Create stats label with HTML formatting
        self.stats_label = QtWidgets.QLabel()

        self.data_filters_layout.addWidget(self.results_plot_scroll)
        self.data_filters_layout.addLayout(data_btn_layout)
        self.data_filters_layout.addLayout(data_shift_layout)
        self.data_filters_layout.addWidget(self.stats_label)

    def updateStatsLabel(self):
        total = len(self.currentResults) if self.currentResults is not None else 0
        filtered = (
            len(self.results_plot.filtered)
            if hasattr(self.results_plot, 'filtered')
            and self.results_plot.filtered is not None
            else total
        )

        self.stats_label.setText(
            f'''<div style='text-align: center;'>
                <b>Localizations:</b> {filtered:,} / {total:,}
            </div>'''
        )

    def populate_data(self):
        '''Populate the data filters.'''
        self.results_plot.setData(self.currentResults.dataFrame())

    def toggle_track_plots(self):
        '''Toggle the track plots.'''
        if self.currentResults is not None:
            self.results_plot.toggle_track_plots()

    def filter_updated(self, df: pd.DataFrame):
        '''Update the view when data filters are updated.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered DataFrame.
        '''
        if df is not None:
            self.updateStatsLabel()

            if df.count().min() > 1:
                render_mode = self.localization_widget.renderMode
                renderClass = BaseRenderer(
                    self.localization_widget.bin_xy, render_mode
                )
                img = renderClass.render_xy(
                    df['x'].to_numpy(), df['y'].to_numpy(), df['intensity'].to_numpy()
                )
                self.setImage(img, autoLevels=False)
            else:
                # Create a black image
                img = np.zeros(
                    (
                        self.image_layers.currentLayer.height(),
                        self.image_layers.currentLayer.width(),
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
                self.setImage(img, autoLevels=False)


class LocalizationRenderHelper:
    '''Helper class for rendering localizations.'''

    class ClustersScatter:
        @classmethod
        def scatter(
            cls, x: np.ndarray, y: np.ndarray, labels: np.ndarray, **kwargs
        ) -> list[pg.GraphicsObject]:
            try:
                # Filter out noise (label < 0) safely
                valid_mask = labels >= 0
                x = x[valid_mask]
                y = y[valid_mask]
                labels = labels[valid_mask]

                if len(labels) == 0:
                    return None

                unique_labels = np.unique(labels)
                max_label = unique_labels.max()

                COLORS = 64

                hues = np.linspace(0, 1, COLORS, endpoint=False)
                color_lookup = np.zeros((COLORS, 3), dtype=np.uint8)
                for idx, h in enumerate(hues):
                    i = int(h * 6.0)
                    f = (h * 6.0) - i
                    t = int(229 * f)
                    q = int(229 * (1.0 - f))
                    i = i % 6
                    if i == 0:
                        r, g, b = 229, t, 0
                    elif i == 1:
                        r, g, b = q, 229, 0
                    elif i == 2:
                        r, g, b = 0, 229, t
                    elif i == 3:
                        r, g, b = 0, q, 229
                    elif i == 4:
                        r, g, b = t, 0, 229
                    else:
                        r, g, b = 229, 0, q
                    color_lookup[idx] = [r, g, b]

                scatter_layers = []
                for color_idx in range(COLORS):
                    mask = (labels % COLORS) == color_idx
                    if not np.any(mask):
                        continue

                    scatter = pg.ScatterPlotItem(
                        x=x[mask],
                        y=y[mask],
                        size=3,
                        pen=None,
                        brush=pg.mkBrush(
                            color_lookup[color_idx]
                        ),  # Single uniform brush per layer!
                        pxMode=True,
                    )
                    scatter_layers.append(scatter)

                return scatter_layers

                # Map colors dynamically
                # mapped_colors = color_lookup[labels % 256]
                # brushes = [pg.mkColor(tuple(c)) for c in mapped_colors]

                # return x, y, brushes
            except Exception:
                traceback.print_exc()
                return None

        @classmethod
        def superimposed(
            cls,
            x: np.ndarray,
            y: np.ndarray,
            intensity: np.ndarray,
            labels: np.ndarray,
            **kwargs,
        ) -> list[pg.GraphicsObject]:
            try:
                # Filter out noise (label < 0) safely
                valid_mask = labels >= 0
                x = x[valid_mask]
                y = y[valid_mask]
                labels = labels[valid_mask]

                if len(labels) == 0:
                    return None

                _arg_sort = np.argsort(labels)
                x = x[_arg_sort]
                y = y[_arg_sort]
                intensity = intensity[_arg_sort]

                label_ids = np.cumsum(np.bincount(labels.astype(np.int64)))

                # min_label = int(np.min(labels))
                max_label = int(np.max(labels))

                for i in range(max_label + 1):
                    _slice = slice(label_ids[i - 1] if i > 0 else 0, label_ids[i])

                    mu_x = np.average(x[_slice], weights=intensity[_slice])
                    mu_y = np.average(y[_slice], weights=intensity[_slice])

                    x[_slice] -= mu_x
                    y[_slice] -= mu_y

                scatter_layers = []

                if not kwargs.get('rasterized', True):
                    scatter = pg.ScatterPlotItem(
                        x=x,
                        y=y,
                        size=3,
                        pen=None,
                        brush=pg.mkBrush(pg.mkColor(255, 0, 0, 127)),
                        pxMode=True,
                    )
                    scatter_layers.append(scatter)
                else:
                    renderer = BaseRenderer(
                        kwargs.get('pixel_size', 0.5),
                        None,
                        kwargs.get('render_mode', RenderModes.HISTOGRAM),
                    )

                    mu_x = np.average(x, weights=intensity)
                    mu_y = np.average(y, weights=intensity)

                    var_x = np.average((x - mu_x) ** 2, weights=intensity)
                    var_y = np.average((y - mu_y) ** 2, weights=intensity)

                    std_x = np.sqrt(var_x)
                    std_y = np.sqrt(var_y)

                    radius = 2.0 * max(std_x, std_y)  # symmetric square crop

                    mask = (np.abs(x - mu_x) <= radius) & (np.abs(y - mu_y) <= radius)

                    x_crop = x[mask] - mu_x
                    y_crop = y[mask] - mu_y
                    image = renderer.render_xy(x_crop, y_crop, intensity[mask])

                    image_item = pg.ImageItem(image, axisOrder='row-major')

                    scatter_layers.append(image_item)

                return scatter_layers
            except Exception:
                traceback.print_exc()
                return None

    @classmethod
    def render_loc(cls, view: LocalizationsView):
        '''Update the rendered super-res image.

        Returns
        -------
        ndarray or None
            Rendered super-res image or None.
        '''
        if not view.fittingResults:
            return

        images = [None] * len(view.fittingResults)

        renderMode = view.localization_widget.renderMode
        projection = view.localization_widget.projection
        xy_bin = view.localization_widget.bin_xy
        z_bin = view.localization_widget.bin_z
        auto_levels = view.localization_widget.autoStretch

        renderer = BaseRenderer(xy_bin, z_bin, renderMode)

        for idx, localizations in enumerate(view.fittingResults):
            if not localizations or len(localizations) <= 0:
                continue

            if 0 <= projection < 3:
                rendered_img = renderer.render(
                    Projection(projection),
                    localizations.data[DataColumns.X],
                    localizations.data[DataColumns.Y],
                    localizations.data[DataColumns.Z],
                    localizations.data[DataColumns.INTENSITY],
                )

                aspect = 1 if projection == 0 else xy_bin / z_bin
                view.view_box.setAspectLocked(True, aspect)
            elif 3 <= projection < 6:
                slice_proj = projection - 3
                aspect = 1 if slice_proj == 0 else xy_bin / z_bin
                view.view_box.setAspectLocked(True, aspect)
                width = z_bin if slice_proj == 0 else xy_bin

                rendered_img = renderer.render_slice(
                    Projection(slice_proj),
                    localizations.data[DataColumns.X],
                    localizations.data[DataColumns.Y],
                    localizations.data[DataColumns.Z],
                    localizations.data[DataColumns.INTENSITY],
                    view.localization_widget.slicePosition,
                    width,
                )
            else:
                continue

            view.setImage(rendered_img, index=idx, autoLevels=auto_levels)
            images[idx] = rendered_img

        return images

    @classmethod
    def render_clusters(
        cls,
        view: LocalizationsView,
    ):
        if view.currentResults is None or len(view.currentResults) <= 0:
            return

        labels = view.currentResults.data.get(DataColumns.CLUSTER_ID)
        if labels is None:
            logger.warning('No cluster labels found. Run DBSCAN first.')
            return None

        labels = np.asarray(labels, dtype=np.int32)
        x = np.asarray(view.currentResults.data[DataColumns.X], dtype=np.float64)
        y = np.asarray(view.currentResults.data[DataColumns.Y], dtype=np.float64)

        dlg = ClusterPlotOptionsDialog(view)
        res = dlg.exec()
        if not res:
            return

        options = dlg.values()
        mode = options['mode']
        rasterized = options['rasterized']
        pixel_size = options['pixel_size']
        render_mode = options['render_mode']

        if mode in [
            'fov',
        ]:
            work_func = cls.ClustersScatter.scatter

            args = [x, y, labels]
            worker_kwargs = {}
        elif mode == 'superimposed':
            work_func = cls.ClustersScatter.superimposed

            intensity = np.asarray(
                view.currentResults.data[DataColumns.INTENSITY], dtype=np.float64
            )

            args = [x, y, intensity, labels]
            worker_kwargs = {
                'rasterized': rasterized,
                'pixel_size': pixel_size,
                'render_mode': render_mode,
            }
        else:
            logger.warning('Mode unavailable.')
            return

        def done(results):
            view.localization_widget.setButtonsEnabled(True)

            if results is not None:
                cluster_overlay_items = pg.GraphicsLayoutWidget(
                    show=True, title='DBSCAN'
                )
                cluster_overlay_items.setBackground(None)

                if any([isinstance(r, pg.ScatterPlotItem) for r in results]):
                    plot_item = cluster_overlay_items.ci.addPlot(
                        title='DBSCAN Clusters'
                    )
                    plot_item.showGrid(x=True, y=True, alpha=0.3)
                    plot_item.getViewBox().setAspectLocked(True, 1)
                    plot_item.getViewBox().invertY(True)
                    plot_item.setLabel('top', 'X [nm]')
                    plot_item.setLabel('left', 'Y [nm]')

                    for layer in results:
                        plot_item.addItem(layer)
                elif any([isinstance(r, pg.ImageItem) for r in results]):
                    plot_item = cluster_overlay_items.ci.addViewBox(
                        row=0, col=0, invertY=True
                    )

                    plot_item.setAspectLocked(True)
                    plot_item.setAutoVisible(True)
                    plot_item.enableAutoRange()

                    image_item: pg.ImageItem = results[0]

                    plot_item.addItem(image_item)

                    histogram_item = pg.HistogramLUTItem(
                        gradientPosition='bottom', orientation='horizontal'
                    )
                    histogram_item.setImageItem(image_item)

                    h_line = pg.InfiniteLine(
                        angle=0,
                        pen=pg.mkPen(color='y', style=QtCore.Qt.PenStyle.DotLine),
                    )
                    v_line = pg.InfiniteLine(
                        angle=90,
                        pen=pg.mkPen(color='y', style=QtCore.Qt.PenStyle.DotLine),
                    )
                    h_line.setY(image_item.image.shape[0] // 2)
                    v_line.setX(image_item.image.shape[1] // 2)

                    plot_item.addItem(h_line)
                    plot_item.addItem(v_line)

                    cluster_overlay_items.addItem(histogram_item, row=1, col=0)

                view.cluster_overlay_items = cluster_overlay_items
                cluster_overlay_items.show()

        view.worker = QThreadWorker(work_func, *args, **worker_kwargs)
        view.worker.signals.result.connect(done)
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def render_3D(cls, view: LocalizationsView):
        '''
        Render a 3D view of the super-res image.
        '''
        if (
            view.currentResults is None
            or view.currentResults.data[DataColumns.Z] is None
        ):
            return None

        if len(view.currentResults) <= 0:
            return

        renderClass = PointCloudRenderer(
            view.localization_widget.bin_xy, view.localization_widget.bin_z
        )
        # Render point cloud
        points, intensities, metadata = renderClass.render(
            view.currentResults.data[DataColumns.X],
            view.currentResults.data[DataColumns.Y],
            view.currentResults.data[DataColumns.Z],
            view.currentResults.data[DataColumns.INTENSITY],
        )
        return PointCloudViewer(points, intensities, metadata)


class LocalizationProcessing:
    '''Controller class for processing localizations.'''

    @classmethod
    def FRC_estimate(cls, view: LocalizationsView):
        '''Estimate FRC (Fourier Ring Correlation) resolution.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        frc_method = view.localization_widget.frcMethod
        data = view.currentResults.toRender()

        def work_func(**kwargs):
            try:
                return FRC_resolution_binomial(
                    np.c_[data[0], data[1], data[2]],
                    view.localization_widget.bin_xy,
                    frc_method,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                plot_frc(*results)

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def NeNA_estimate(clas, view: LocalizationsView):
        '''Estimate Nearest-Neighbor Analysis for localization precision.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.nn_trajectories(0, 200, 0, 1)
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

                view.NeNA_widget = NeNA_Widget(
                    view.currentResults.neighbour_dist, view.currentResults.trackID
                )

                res = view.NeNA_widget.exec()

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def TARDIS_analysis(cls, view: LocalizationsView):
        '''Perform TARDIS analysis.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        view.tardis = TARDIS_Widget(
            view.currentResults.frames,
            view.currentResults.locX,
            view.currentResults.locY,
            view.currentResults.locZ,
        )
        view.tardis.startWorker.connect(lambda worker: view._threadpool.start(worker))
        view.tardis.show()

    @classmethod
    def drift_cross(cls, view: LocalizationsView):
        '''Perform drift cross-correlation.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        method = view.localization_widget.crossMethod
        label = view.localization_widget.crossLabel

        def work_func(**kwargs):
            try:
                return view.currentResults.drift_cross_correlation(
                    view.localization_widget.crossBins,
                    view.localization_widget.crossPixelSize,
                    view.localization_widget.crossUpSampling,
                    view.localization_widget.crossRMAX,
                    method=method,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                view.render_loc()
                view.currentResults = results[0]
                view.results_plot.setData(view.currentResults.dataFrame())
                plot_drift(
                    *results[2],
                    title=f'{label} Cross-Correlation',
                )

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def drift_fdm(cls, view: LocalizationsView):
        '''Perform drift correction using fiducial markers.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.drift_fiducial_marker()
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                view.currentResults = results[0]
                view.results_plot.setData(view.currentResults.dataFrame())
                plot_drift(*results[1], title='Fiducial Marker Drift Correction')
                view.render_loc()

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def nneigh_merge(cls, view: LocalizationsView):
        '''Perform nearest-neighbor merging.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.nearest_neighbour_merging(
                    view.localization_widget.nnMinDistance,
                    view.localization_widget.nnMaxDistance,
                    view.localization_widget.nnMaxOff,
                    view.localization_widget.nnMaxLength,
                    view.localization_widget.nearestNeighbors,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def nneigh(cls, view: LocalizationsView):
        '''Perform nearest-neighbor analysis.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.nn_trajectories(
                    view.localization_widget.nnMinDistance,
                    view.localization_widget.nnMaxDistance,
                    view.localization_widget.nnMaxOff,
                    view.localization_widget.nearestNeighbors,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def merge(cls, view: LocalizationsView):
        '''Merge tracks.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.merge_tracks(
                    view.localization_widget.nnMaxLength
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)

    @classmethod
    def extract_z(cls, view: LocalizationsView):
        '''Extract Z values from the fitting results.'''
        if view.currentResults is None:
            return None
        if len(view.currentResults) <= 0:
            return

        curveFitMethod = view.localization_widget.curveFitMethod

        if curveFitMethod == stats.CurveFitMethod.LINEAR:
            view.currentResults.data[DataColumns.Z] = (
                (
                    view.currentResults.data[DataColumns.SIGMA_X]
                    / view.currentResults.data[DataColumns.SIGMA_Y]
                )
                - view.localization_widget.zCalIntercept
            ) / view.localization_widget.zCalSlope
            view.results_plot.setData(view.currentResults.dataFrame())
        elif curveFitMethod:
            if view.localization_widget.z_cal_curve:
                view.currentResults.data[DataColumns.Z] = (
                    stats.CurveAnalyzer.get_z_from_y_array_optimized(
                        view.currentResults.data[DataColumns.SIGMA_X]
                        / view.currentResults.data[DataColumns.SIGMA_Y],
                        view.localization_widget.z_cal_curve,
                        region=[-450, 450],
                    )
                )
                view.results_plot.setData(view.currentResults.dataFrame())

    @classmethod
    def apply_filters(cls, view: LocalizationsView):
        '''Apply data filters to the fitting results.'''
        if (
            not hasattr(view.results_plot, 'filtered')
            or view.results_plot.filtered is None
        ):
            view.results_plot.update()

        view.currentResults = FittingResults.fromDataFrame(
            view.results_plot.filtered, 1, path=view.currentResults.path
        )
        view.results_plot.setData(view.currentResults.dataFrame())
        view.render_loc()
        logger.info('Filters applied.')

    @classmethod
    def zero_coordinates(cls, view: LocalizationsView):
        '''Zero the fitting results coordinates.'''
        if view.currentResults is not None:
            view.currentResults.zero_coordinates()
            view.results_plot.setData(view.currentResults.dataFrame())
            view.render_loc()
            logger.info('Coordinates reset.')

    @classmethod
    def shift_data(cls, view: LocalizationsView):
        '''Shift the data by a specified amount.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        axis: DataColumns = view.data_shift_axis.currentData()
        shift = view.data_shift.value()
        direction = -1 if view.data_shift_dir.isChecked() else 1

        view.currentResults.data[axis] += shift * direction

        view.results_plot.clear()
        view.render_loc()

    @classmethod
    def dbscan_clustering(cls, view: LocalizationsView):
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.cluster_dbscan(
                    eps=view.localization_widget.dbscanEps,
                    metric=view.localization_widget.dbscanMetric,
                    algorithm=view.localization_widget.dbscanAlgorithm,
                    min_samples=view.localization_widget.dbscanMinSamples,
                    leaf_size=view.localization_widget.dbscanLeafSize,
                    scale_data=view.localization_widget.dbscanScaleData,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            view.localization_widget.setButtonsEnabled(True)
            # if results is not None:
            #     view.currentResults = results
            #     view.results_plot.setData(view.currentResults.dataFrame())

        view.worker = QThreadWorker(work_func)
        view.worker.signals.result.connect(done)
        # Execute
        view.localization_widget.setButtonsEnabled(False)
        view._threadpool.start(view.worker)


class LocalizationIO:
    @classmethod
    def export_loc(cls, view: LocalizationsView, filename=None):
        '''Export the fitting results into a file.

        Parameters
        ----------
        filename : str, optional
            File path; if None, a save file dialog is shown, by default None.
        '''
        if view.currentResults is None:
            return

        if filename is None:
            if not view.export_options.exec():
                return

            filename, _ = getSaveFileName(
                view,
                'Export localizations',
                filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
                directory=os.path.dirname(view.currentResults.path),
            )

        if len(filename) > 0:
            options = view.export_options.toList()

            dataFrame = view.currentResults.dataFrame()
            exp_columns = []
            for col in dataFrame.columns:
                if col in options:
                    exp_columns.append(col)

            if exp_columns:
                if '.tsv' in filename:
                    dataFrame.to_csv(
                        filename,
                        index=False,
                        columns=exp_columns,
                        float_format=view.export_options.export_precision.text(),
                        sep='\t',
                        encoding='utf-8',
                    )
                elif '.h5' in filename:
                    dataFrame[exp_columns].to_hdf(
                        filename, key='microEye', index=False, complevel=0
                    )

            if 'Super-res image' in options:
                sres_img = view.render_loc()[view.image_layers.currentIndex]
                tf.imwrite(
                    re.sub(r'(\.h5|\.tsv)$', '_super_res.tif', filename),
                    sres_img,
                    photometric='minisblack',
                    bigtiff=True,
                    ome=False,
                )

    @classmethod
    def import_loc(cls, view: LocalizationsView):
        '''Import fitting results from a file.'''
        filename, _ = getOpenFileName(
            view,
            'Import localizations',
            filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
            directory=os.path.dirname(view.currentResults.path),
        )

        if len(filename) > 0:
            results = FittingResults.fromFile(filename, 1)

            if results is not None:
                view.add_data_layer(results)
                logger.info('Done importing results.')
            else:
                logger.error('Error importing results.')
