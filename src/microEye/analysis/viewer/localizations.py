import logging
import os
import re
import traceback
import weakref

import cv2
import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
import tifffile as tf

from microEye.analysis.checklist_dialog import ChecklistDialog
from microEye.analysis.clustering.density import (
    DensityAnalysisPlot,
    get_cluster_density,
)
from microEye.analysis.clustering.dialogs import OptionsDialog, OptionsModes
from microEye.analysis.clustering.pipeline import (
    render_grids_batch,
    render_rulers_batch,
)
from microEye.analysis.clustering.viewer import (
    ClusterRenderData,
    ClusterResultsWidget,
    ClustersScatter,
    ClustersViewer,
    PipelineResult,
    add_plot_result,
)
from microEye.analysis.drift import DriftOptionsDialog, plot_drift
from microEye.analysis.fitting.nena import NeNA_Widget
from microEye.analysis.fitting.processing import plot_animation_stats
from microEye.analysis.fitting.psf import stats
from microEye.analysis.fitting.results import (
    DataColumns,
    FittingResults,
)
from microEye.analysis.fitting.results_stats import resultsStatsWidget
from microEye.analysis.fitting.tardis import TARDIS_Widget
from microEye.analysis.processing import FRC_resolution_binomial, plot_frc
from microEye.analysis.rendering import (
    Axis,
    BaseRenderer,
    PointCloudRenderer,
    Projection,
    RenderModes,
    SpatialIndex3D,
)
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
from microEye.utils.gui_helper import (
    create_double_spin_box,
    create_grid_layout,
    create_hbox_layout,
    create_spin_box,
    create_widget,
)
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


class LocalizationTab(SideBar):
    projectionChanged = Signal()

    renderActivated = Signal()
    render3DActivated = Signal()
    animationActivated = Signal()
    viewUpdateRequested = Signal()

    extractZActivated = Signal()

    driftCorrectionActivated = Signal(str, dict)

    frcActivated = Signal()
    nenaActivated = Signal()
    tardisActivated = Signal()

    nearestNeighActivated = Signal()
    mergeActivated = Signal()
    nn_mergeActivated = Signal()

    dbscanActivated = Signal()
    hdbscanActivated = Signal()
    opticsActivated = Signal()
    densityActivated = Signal()
    renderClustersActivated = Signal()
    renderGridActivated = Signal()
    renderRulersActivated = Signal()

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

        self.__init_clustering()

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

        self.__view_cx = create_double_spin_box(
            min_value=-1e7, max_value=1e7, decimals=1
        )
        self.__view_cy = create_double_spin_box(
            min_value=-1e7, max_value=1e7, decimals=1
        )
        self.__view_w = create_double_spin_box(
            min_value=1, max_value=int(1e7), decimals=1, initial_value=1000
        )
        self.__view_h = create_double_spin_box(
            min_value=1, max_value=int(1e7), decimals=1, initial_value=1000
        )
        self.__goto_btn = QtWidgets.QPushButton(
            'Go to View', clicked=lambda: self.viewUpdateRequested.emit()
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

        flocalization.addRow(
            'Center X/Y [nm]:', create_hbox_layout(self.__view_cx, self.__view_cy)
        )
        flocalization.addRow(
            'Width/Height [nm]:', create_hbox_layout(self.__view_w, self.__view_h)
        )
        flocalization.addRow(self.__goto_btn)
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
        self._drift_dialog = DriftOptionsDialog(parent=self)
        self._drift_dialog.drift_requested.connect(
            lambda method, params: self.driftCorrectionActivated.emit(method, params)
        )

        self._drift_correction_btn = QtWidgets.QPushButton(
            'Drift Correction', clicked=lambda: self._drift_dialog.exec()
        )

        # Drift GroupBox
        fdrift = QtWidgets.QVBoxLayout()
        fdrift.addWidget(self._drift_correction_btn)

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

    def __init_clustering(self):
        clustering = QtWidgets.QFormLayout()

        self.__dbscan_options = OptionsDialog(mode=OptionsModes.DBSCAN)
        self.__optics_options = OptionsDialog(mode=OptionsModes.OPTICS)
        self.__hdbscan_options = OptionsDialog(mode=OptionsModes.HDBSCAN)
        self.__density_options = OptionsDialog(mode=OptionsModes.CLUSTER_DENSITY)

        self.__cluster_options_dialog = OptionsDialog(mode=OptionsModes.RENDERING)
        self.__ruler_options_dialog = OptionsDialog(
            mode=OptionsModes.RULER, export=True
        )
        self.__grid_options_dialog = OptionsDialog(mode=OptionsModes.GRID, export=True)

        def dbscan_emit():
            res = self.__dbscan_options.exec()

            if res == QtWidgets.QDialog.DialogCode.Accepted:
                self.dbscanActivated.emit()

        self.__dbscan_btn = QtWidgets.QPushButton(
            'DBSCAN', clicked=lambda: dbscan_emit()
        )

        def hdbscan_emit():
            res = self.__hdbscan_options.exec()

            if res == QtWidgets.QDialog.DialogCode.Accepted:
                self.hdbscanActivated.emit()

        self.__hdbscan_btn = QtWidgets.QPushButton(
            'HDBSCAN', clicked=lambda: hdbscan_emit()
        )

        def density_emit():
            res = self.__density_options.exec()

            if res == QtWidgets.QDialog.DialogCode.Accepted:
                self.densityActivated.emit()

        self.__density_btn = QtWidgets.QPushButton(
            'Density Analysis', clicked=lambda: density_emit()
        )

        def optics_emit():
            res = self.__optics_options.exec()

            if res == QtWidgets.QDialog.DialogCode.Accepted:
                self.opticsActivated.emit()

        self.__optics_btn = QtWidgets.QPushButton(
            'OPTICS', clicked=lambda: optics_emit()
        )
        self.__cluster_render_btn = QtWidgets.QPushButton(
            'Show Clusters',
            clicked=lambda: self.renderClustersActivated.emit(),
        )
        self.__cluster_ruler_btn = QtWidgets.QPushButton(
            'Show Rulers',
            clicked=lambda: self.renderRulersActivated.emit(),
        )
        self.__cluster_grid_btn = QtWidgets.QPushButton(
            'Show Grid',
            clicked=lambda: self.renderGridActivated.emit(),
        )

        clustering.addWidget(self.__dbscan_btn)
        clustering.addWidget(self.__hdbscan_btn)
        clustering.addWidget(self.__optics_btn)
        clustering.addWidget(self.__density_btn)
        clustering.addWidget(self.__cluster_render_btn)
        clustering.addWidget(self.__cluster_ruler_btn)
        clustering.addWidget(self.__cluster_grid_btn)
        self.addSection('Clustering', clustering, False)

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

    def setBtnEnabled(self, btn_attr_name: str, enabled: bool = False):
        '''Disable a button by its attribute name.'''
        btn = getattr(self, f'_LocalizationTab{btn_attr_name}', None)
        if isinstance(btn, QtWidgets.QPushButton):
            btn.setEnabled(enabled)

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
        return self._drift_dialog.method_combo.currentData()

    @property
    def crossLabel(self) -> str:
        return self._drift_dialog.method_combo.currentText()

    @property
    def crossBins(self) -> int:
        return self._drift_dialog.cross_bins.value()

    @property
    def crossPixelSize(self) -> int:
        return self._drift_dialog.cross_px.value()

    @property
    def crossUpSampling(self) -> int:
        return self._drift_dialog.cross_up.value()

    @property
    def crossRMAX(self) -> int:
        return self._drift_dialog.cross_rmax.value()

    @property
    def aimSegmentSize(self) -> int:
        return self._drift_dialog.aim_segment_size.value()

    @property
    def aimIntersectD(self) -> float:
        return self._drift_dialog.aim_intersect_d.value()

    @property
    def aimZScale(self) -> float:
        return self._drift_dialog.aim_z_scale.value()

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

    def set_view_readout(self, cx: float, cy: float, w: float, h: float):
        for box, val in zip(
            (self.__view_cx, self.__view_cy, self.__view_w, self.__view_h),
            (cx, cy, w, h),
        ):
            box.blockSignals(True)
            box.setValue(val)
            box.blockSignals(False)

    @property
    def viewCenterSize(self):
        return (
            self.__view_cx.value(),
            self.__view_cy.value(),
            self.__view_w.value(),
            self.__view_h.value(),
        )

    def setButtonsEnabled(self, enabled=True):
        # check attributes of type QPushButton and set their enabled state
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, QtWidgets.QPushButton):
                attr.setEnabled(enabled)

    def toDict(self):
        return {
            'renderMode': self.renderMode,
            'projection': self.projection,
            'binFrames': self.bin_frames,
            'binXY': self.bin_xy,
            'binZ': self.bin_z,
            'slicePosition': self.slicePosition,
            'autoStretch': self.autoStretch,
            'frcMethod': self.frcMethod,
            'crossMethod': self.crossMethod,
            'crossBins': self.crossBins,
            'crossPixelSize': self.crossPixelSize,
            'crossUpSampling': self.crossUpSampling,
            'crossRMAX': self.crossRMAX,
            'aimSegmentSize': self.aimSegmentSize,
            'aimIntersectD': self.aimIntersectD,
            'aimZScale': self.aimZScale,
            'nearestNeighbors': self.nearestNeighbors,
            'nnMinDistance': self.nnMinDistance,
            'nnMaxDistance': self.nnMaxDistance,
            'nnMaxOff': self.nnMaxOff,
            'nnMaxLength': self.nnMaxLength,
            'curveFitMethod': self.curveFitMethod,
            'zCalIntercept': self.zCalIntercept,
            'zCalSlope': self.zCalSlope,
            'dbscanOptions': self.__dbscan_options.values(),
            'hdbscanOptions': self.__hdbscan_options.values(),
            'opticsOptions': self.__optics_options.values(),
            'densityOptions': self.__density_options.values(),
        }

    def optionsDialog(self, mode: OptionsModes):
        if mode == OptionsModes.DBSCAN:
            return self.__dbscan_options
        elif mode == OptionsModes.HDBSCAN:
            return self.__hdbscan_options
        elif mode == OptionsModes.OPTICS:
            return self.__optics_options
        elif mode == OptionsModes.CLUSTER_DENSITY:
            return self.__density_options
        elif mode == OptionsModes.RENDERING:
            return self.__cluster_options_dialog
        elif mode == OptionsModes.RULER:
            return self.__ruler_options_dialog
        elif mode == OptionsModes.GRID:
            return self.__grid_options_dialog
        else:
            logger.warning(f'No options dialog for mode: {mode}')
            return None


class LocalizationsView(QtWidgets.QWidget):
    '''
    A class for viewing and interacting with SMLM localizations in a PyQt5 application.
    '''

    __INSTANCES = 0

    def __init__(self, fittingResults: FittingResults = None, path=None):
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

        if path is not None and isinstance(path, str):

            def get_data(event):
                return FittingResults.fromFile(path, 1)

            worker = QThreadWorker(get_data)
            worker.signals.result.connect(self.on_stack_ready)

            self._threadpool.start(worker)

    def runAsync(self, func, *args, **kwargs):
        done = kwargs.pop('done', lambda results: None)
        error = kwargs.pop('error', lambda e: None)

        def finished():
            self.localization_widget.setButtonsEnabled(True)
            logger.info(kwargs.pop('finished_message', 'Processing completed.'))

        self.worker = QThreadWorker(func, *args, **kwargs)
        self.worker.signals.result.connect(done)
        self.worker.signals.error.connect(error)
        self.worker.signals.finished.connect(finished)
        # Execute
        self.localization_widget.setButtonsEnabled(False)
        self._threadpool.start(self.worker)

    def on_stack_ready(self, results: FittingResults):
        if results is not None and isinstance(results, FittingResults):
            logger.info(
                f'Done importing results (Number of localizations: {len(results)}).'
            )
            self.add_data_layer(results)
            self.show()
        else:
            logger.error('Error importing results.')
            self.close()

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

        index = kwargs.pop('index', self.image_layers.currentIndex)
        rect = kwargs.pop('rect', None)

        colorMap = self.fittingResults[index].colormap
        if colorMap is not None:
            kwargs['colorMap'] = colorMap

        item = self.image_layers.getImageItemAt(index)
        item.setImage(image, autoLevels=autoLevels, **kwargs)

        if rect is not None:
            item.setRect(QtCore.QRectF(*rect))  # (x0, y0, width_nm, height_nm)

    def init_graphics(self):
        '''Initialize the graphics layout for the main window.'''
        # A plot area (ViewBox + axes) for displaying the image
        self.image_widget = pg.GraphicsLayoutWidget()
        self.image_widget.setMinimumWidth(600)

        # Create the ViewBox
        self.view_box = self.image_widget.ci.addViewBox(row=0, col=0)

        self.view_box.setAspectLocked(True, 1)
        self.view_box.setAutoVisible(True)
        self.view_box.enableAutoRange(enable=False)
        self.view_box.invertY(True)

        self._scale_bar = pg.ScaleBar(
            size=1000,
            suffix='nm',
            brush=pg.mkBrush('w'),
            offset=(-20, -20),
        )
        self._scale_bar.setParentItem(self.view_box)

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

        self._render_debounce = QtCore.QTimer(self)
        self._render_debounce.setSingleShot(True)
        self._render_debounce.setInterval(50)
        self._render_debounce.timeout.connect(self.render_loc)

        self._render_token = 0

        def _on_view_range_changed(*_):
            (x0, x1), (y0, y1) = self.view_box.viewRange()
            self.localization_widget.set_view_readout(
                (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            )

            self._render_debounce.start()

        self.view_box.sigRangeChanged.connect(_on_view_range_changed)

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

        self.render_loc(full_extent=True)
        # self.view_box.autoRange()

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

    def goto_view(self):
        cx, cy, w, h = self.localization_widget.viewCenterSize
        self.view_box.setRange(
            xRange=(cx - w / 2, cx + w / 2),
            yRange=(cy - h / 2, cy + h / 2),
            padding=0,
        )

    def render_loc(self, full_extent=False):
        '''Update the rendered super-res image.

        Parameters
        ----------
        full_extent : bool, optional
            Whether to render the full extent of the image, by default False.

        Returns
        -------
        ndarray or None
            Rendered super-res image or None.
        '''
        return LocalizationRenderHelper.render_loc(self, full_extent=full_extent)

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

        self.runAsync(
            work_func, done=done, finished_message='Animation rendering completed.'
        )

        # self.currentResults.toAnimation()

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
        self.localization_widget.viewUpdateRequested.connect(self.goto_view)

        self.localization_widget.driftCorrectionActivated.connect(
            lambda method, params: LocalizationProcessing.drift_correction(
                self, method, params
            )
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
        self.localization_widget.hdbscanActivated.connect(
            lambda: LocalizationProcessing.hdbscan_clustering(self)
        )
        self.localization_widget.opticsActivated.connect(
            lambda: LocalizationProcessing.optics_clustering(self)
        )

        self.localization_widget.densityActivated.connect(
            lambda: LocalizationProcessing.density_analysis(self)
        )

        self.localization_widget.renderClustersActivated.connect(
            lambda: LocalizationRenderHelper.render_clusters(self)
        )
        self.localization_widget.renderRulersActivated.connect(
            lambda: LocalizationRenderHelper.render_rulers(self)
        )
        self.localization_widget.renderGridActivated.connect(
            lambda: LocalizationRenderHelper.render_grid(self)
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
        self.localizations_per_frame_btn = QtWidgets.QPushButton(
            'More Stats', clicked=lambda: LocalizationProcessing.extended_stats(self)
        )

        data_btn_layout = create_grid_layout(
            2,
            3,
            self.populate_data_btn,
            self.apply_filters_btn,
            self.zero_coords_btn,
            self.toggle_tracking_btn,
            self.localizations_per_frame_btn,
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

    def _show_empty_filter_image(self):
        img = np.zeros(
            (
                self.image_layers.currentLayer.height(),
                self.image_layers.currentLayer.width(),
                3,
            ),
            np.uint8,
        )
        cv2.putText(
            img, 'EMPTY!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2
        )
        self.setImage(img, autoLevels=False)

    def filter_updated(self, df: pd.DataFrame):
        '''Update the view when data filters are updated.

        Parameters
        ----------
        df : pd.DataFrame
            The filtered DataFrame.
        '''
        if df is None:
            return

        self.updateStatsLabel()

        if df.count().min() <= 1:
            self._show_empty_filter_image()
            return

        filtered_table = FittingResults.fromDataFrame(df)
        kwargs = self.localization_widget.toDict()
        kwargs['view_range'] = tuple(self.view_box.viewRange())

        images, rects = LocalizationRenderHelper.batch_render_loc(
            [filtered_table], **kwargs
        )
        img, rect = (images[0], rects[0]) if images else (None, None)

        if img is None:
            self._show_empty_filter_image()
            return

        self.setImage(img, autoLevels=False, rect=rect)

    @classmethod
    def get_data_view(cls, path: str) -> QtWidgets.QWidget:
        with h5py.File(path, 'r') as f:
            if 'microEye' in f and 'PYTABLES_FORMAT_VERSION' in f.attrs:
                type = 'Localizations'
                return LocalizationsView(path=path)
            elif (
                'file_type' in f.attrs
                and f.attrs['file_type'] == 'microEye.analysis.clustering.model'
            ):
                type = f.attrs['file_type']
                return ClusterResultsWidget(path=path)
        return None


class LocalizationRenderHelper:
    '''Helper class for rendering localizations.'''

    _PROJECTION_AXES = {
        0: (Axis.X, Axis.Y, None),  # XY
        1: (Axis.X, Axis.Z, None),  # XZ
        2: (Axis.Y, Axis.Z, None),  # YZ
        3: (Axis.X, Axis.Y, Axis.Z),  # XY (slice)
        4: (Axis.X, Axis.Z, Axis.Y),  # XZ (slice)
        5: (Axis.Y, Axis.Z, Axis.X),  # YZ (slice)
    }
    _AXIS_COLUMN = {Axis.X: DataColumns.X, Axis.Y: DataColumns.Y, Axis.Z: DataColumns.Z}

    @staticmethod
    def _effective_bin_pair(
        requested_x, requested_y, extent_x, extent_y, max_px, force=False
    ):
        '''
        Calculate the effective bin sizes for rendering based on
        the requested bin sizes, the extent of the data, and the
        maximum pixel size allowed.

        Parameters
        ----------
        requested_x : float
            The requested bin size in the x-direction.
        requested_y : float
            The requested bin size in the y-direction.
        extent_x : float
            The extent of the data in the x-direction.
        extent_y : float
            The extent of the data in the y-direction.
        max_px : int
            The maximum pixel size allowed.

        Returns
        -------
        tuple of float
            The effective bin sizes in the x and y directions.
        '''
        requested_x = max(requested_x, 1e-3)
        requested_y = max(requested_y, 1e-3)
        needed_x = extent_x / requested_x if extent_x > 0 else 0.0
        needed_y = extent_y / requested_y if extent_y > 0 else 0.0
        needed = max(needed_x, needed_y)
        if needed <= max_px or force:
            return requested_x, requested_y
        scale = needed / max_px
        return requested_x * scale, requested_y * scale

    _spatial_index_cache = weakref.WeakKeyDictionary()

    @classmethod
    def _get_spatial_index(cls, table: FittingResults) -> SpatialIndex3D:
        '''
        Spatial indices are used for efficient spatial queries on the localization data.

        This method caches the spatial index to avoid redundant computations.

        When `FittingResults` is updated, the cache is automatically
        invalidated due to the use of `WeakKeyDictionary`.'''
        index = cls._spatial_index_cache.get(table)
        if index is None:
            index = SpatialIndex3D(
                table.data[DataColumns.X],
                table.data[DataColumns.Y],
                table.data[DataColumns.Z],
            )
            cls._spatial_index_cache[table] = index
        return index

    @classmethod
    def render_loc(cls, view: LocalizationsView, full_extent: bool = False):
        '''Update the rendered super-res image.

        Returns
        -------
        ndarray or None
            Rendered super-res image or None.
        '''
        if not view.fittingResults:
            return

        view._render_token += 1
        token = view._render_token

        kwargs = view.localization_widget.toDict()
        kwargs['view_range'] = None if full_extent else tuple(view.view_box.viewRange())

        def work(**_):
            return LocalizationRenderHelper.batch_render_loc(
                view.fittingResults, **kwargs
            )

        def done(result):
            if token != view._render_token or result is None:
                return
            images, rects = result
            auto_levels = view.localization_widget.autoStretch
            for idx, (img, rect) in enumerate(zip(images, rects)):
                if img is not None:
                    # Compute the histogram levels for auto-stretching
                    # Conisder data is SMLM and linear stretching is not ideal,
                    # we can use percentile-based stretching
                    levels = (
                        np.percentile(img, [0, 99.9])
                        if auto_levels
                        else view.histogram_item.getLevels()
                    )

                    view.setImage(
                        img, index=idx, rect=rect, autoLevels=False, levels=levels
                    )

                    if view._scale_bar:
                        # sacale bar size in nm based on the FOV
                        scale_size = max(rect[2], rect[3]) / 10
                        # Round to nearest 1, 2, 5, 10, 20, 50, 100, etc.
                        magnitude = 10 ** np.floor(np.log10(scale_size))
                        nice_sizes = np.array([1, 2, 5, 10, 20, 50, 100]) * magnitude
                        closest_size = nice_sizes[
                            np.argmin(np.abs(nice_sizes - scale_size))
                        ]
                        view._scale_bar.size = closest_size
                        view._scale_bar.text.setText(
                            f'{closest_size:.0f} nm'
                            if closest_size < 1000
                            else f'{closest_size / 1000:.1f} μm'
                        )
                        view._scale_bar.updateBar()

            if full_extent:
                view.view_box.autoRange()

        def error(exc):
            logger.error('Rendering failed.')
            traceback.print_exc()

        worker = QThreadWorker(work)
        worker.signals.result.connect(done)
        worker.signals.error.connect(error)
        view._threadpool.start(worker)

    @classmethod
    def batch_render_loc(cls, tables: list[FittingResults], **kwargs):
        '''Batch render localizations for multiple FittingResults.

        Parameters
        ----------
        tables : list[FittingResults]
            List of FittingResults to render.
        **kwargs : dict
            Additional keyword arguments for rendering.

        Returns
        -------
        tuple[list[np.ndarray], list[float]]
            A tuple containing the rendered images and their corresponding aspects.
        '''
        if not tables:
            return [], []

        images = [None] * len(tables)
        rects = [None] * len(tables)
        aspects = [1] * len(tables)

        renderMode = kwargs.get('renderMode', RenderModes.HISTOGRAM)
        projection = kwargs.get('projection', 0)
        xy_bin = kwargs.get('binXY', 1) or 1
        z_bin = kwargs.get('binZ', 1) or 1
        slicePosition = kwargs.get('slicePosition', 0)

        # ((x0,x1),(y0,y1)) of the *displayed* axes, nm, or None
        view_range = kwargs.get('view_range')
        max_render_px = kwargs.get('max_render_px', 2048)

        axes = cls._PROJECTION_AXES.get(projection)
        if axes is None:
            return images, rects

        view_x_axis, view_y_axis, slice_axis = axes

        requested_bin = {Axis.X: xy_bin, Axis.Y: xy_bin, Axis.Z: z_bin}
        slice_width = requested_bin.get(slice_axis) if slice_axis is not None else None

        for idx, localizations in enumerate(tables):
            if not localizations or len(localizations) <= 0:
                continue

            index = cls._get_spatial_index(localizations)
            if not (index.has_axis(view_x_axis) and index.has_axis(view_y_axis)):
                continue  # e.g. no Z column but an XZ/YZ mode was requested
            if slice_axis is not None and not index.has_axis(slice_axis):
                continue

            col_x = localizations.data[cls._AXIS_COLUMN[view_x_axis]]
            col_y = localizations.data[cls._AXIS_COLUMN[view_y_axis]]

            if view_range is not None:
                (x0, x1), (y0, y1) = view_range
            else:
                x0, x1 = float(np.min(col_x)), float(np.max(col_x))
                y0, y1 = float(np.min(col_y)), float(np.max(col_y))

            eff_x, eff_y = cls._effective_bin_pair(
                requested_bin[view_x_axis],
                requested_bin[view_y_axis],
                x1 - x0,
                y1 - y0,
                max_render_px,
                force=kwargs.get('force_bin_size', False),
            )

            origin_x = np.floor(x0 / eff_x) * eff_x
            origin_y = np.floor(y0 / eff_y) * eff_y

            ranges = {view_x_axis: (origin_x, x1), view_y_axis: (origin_y, y1)}
            if slice_axis is not None:
                half_w = slice_width / 2
                ranges[slice_axis] = (slicePosition - half_w, slicePosition + half_w)

            sel = index.query(ranges)
            if sel.size == 0:
                continue

            n_cols = max(1, int(np.ceil((x1 - origin_x) / eff_x)) + 4)
            n_rows = max(1, int(np.ceil((y1 - origin_y) / eff_y)) + 4)

            renderer = BaseRenderer(eff_x, eff_y, renderMode)
            rendered_img = renderer.render_xy(
                col_x[sel],
                col_y[sel],
                localizations.data[DataColumns.INTENSITY][sel],
                shape=(n_rows, n_cols),
                origin=(origin_x, origin_y),
            )

            images[idx] = rendered_img
            rects[idx] = (origin_x, origin_y, n_cols * eff_x, n_rows * eff_y)

        return images, rects

    @classmethod
    def render_clusters(
        cls,
        view: LocalizationsView,
    ):
        if view.currentResults is None or len(view.currentResults) <= 0:
            return

        labels = view.currentResults.data.get(DataColumns.CLUSTER_ID)
        if labels is None or len(np.unique(labels)) <= 2:
            logger.warning('No cluster labels found. Run Clustering first.')
            return None

        view.cluster_viewer = ClustersViewer()

        view.cluster_viewer.load_cluster_render(
            cluster_render_result=ClusterRenderData(
                tables=[view.currentResults.toClusterRender()],
            )
        )

        view.cluster_viewer.show()

    @classmethod
    def render_rulers(cls, view: LocalizationsView):
        if view.currentResults is None or len(view.currentResults) <= 0:
            return

        labels = view.currentResults.data.get(DataColumns.CLUSTER_ID)
        if labels is None or len(np.unique(labels)) <= 2:
            logger.warning('No cluster labels found. Run Clustering first.')
            return None

        dialog = view.localization_widget.optionsDialog(OptionsModes.RULER)

        if dialog is None:
            logger.warning('Ruler options dialog not found.')
            return

        res = dialog.exec()
        if not res:
            return

        work_func = render_rulers_batch

        args = [[view.currentResults]]
        worker_kwargs = dialog.values()

        def done(results):
            if results is not None and isinstance(results, PipelineResult):
                widget = ClusterResultsWidget()
                widget.load_pipeline_result(results)
                widget.resize(1100, 600)
                widget.show()
                view.ruler_overlay_items = widget

        view.runAsync(
            work_func,
            *args,
            **worker_kwargs,
            done=done,
            finished_message='Ruler rendering completed.',
        )

    @classmethod
    def render_grid(cls, view: LocalizationsView):
        if view.currentResults is None or len(view.currentResults) <= 0:
            return

        labels = view.currentResults.data.get(DataColumns.CLUSTER_ID)
        if labels is None or len(np.unique(labels)) <= 2:
            logger.warning('No cluster labels found. Run Clustering first.')
            return None

        dialog = view.localization_widget.optionsDialog(OptionsModes.GRID)
        if dialog is None:
            logger.warning('Grid options dialog not found.')
            return

        res = dialog.exec()
        if not res:
            return

        work_func = render_grids_batch

        args = [[view.currentResults]]
        worker_kwargs = dialog.values()

        def done(results):
            if results is not None and isinstance(results, PipelineResult):
                widget = ClusterResultsWidget()
                widget.load_pipeline_result(results)
                widget.resize(1100, 600)
                widget.show()
                view.ruler_overlay_items = widget

        view.runAsync(
            work_func,
            *args,
            **worker_kwargs,
            done=done,
            finished_message='Grid rendering completed.',
        )

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
            if results is not None:
                plot_frc(*results)

        view.runAsync(
            work_func, done=done, finished_message='FRC estimation completed.'
        )

    @classmethod
    def NeNA_estimate(cls, view: LocalizationsView):
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
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

                view.NeNA_widget = NeNA_Widget(
                    view.currentResults.neighbour_dist, view.currentResults.trackID
                )

                res = view.NeNA_widget.exec()

        view.runAsync(
            work_func, done=done, finished_message='NeNA estimation completed.'
        )

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
    def drift_correction(cls, view: LocalizationsView, method: str, params: dict):
        '''Perform drift correction using the specified method.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        label = view.localization_widget._drift_dialog.method_combo.currentText()

        def work_func(**kwargs):
            try:
                if method in ['rcc', 'dcc', 'mcc', 'phase']:
                    return view.currentResults.drift_cross_correlation(
                        method=method, **params
                    )
                elif method == 'fdm':
                    return view.currentResults.drift_fiducial_marker()
                elif method == 'aim':
                    return view.currentResults.drift_aim(**params)
                elif method == 'comet':
                    return view.currentResults.drift_comet(**params)
                else:
                    logger.warning(f'Unknown drift correction method: {method}')

            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            if results is not None:
                view.currentResults = results[0]
                view.results_plot.setData(view.currentResults.dataFrame())
                plot_drift(
                    *results[1],
                    title=f'{label}',
                )
                view.render_loc()

        view.runAsync(
            work_func, done=done, finished_message='Drift correction completed.'
        )

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
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

        view.runAsync(
            work_func, done=done, finished_message='Nearest-neighbor merging completed.'
        )

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
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

        view.runAsync(
            work_func,
            done=done,
            finished_message='Nearest-neighbor analysis completed.',
        )

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
            if results is not None:
                view.currentResults = results
                view.results_plot.setData(view.currentResults.dataFrame())

        view.runAsync(work_func, done=done, finished_message='Track merging completed.')

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
    def extended_stats(cls, view: LocalizationsView):
        '''Display extended statistics for the fitting results.'''
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        # localization per frame
        def work_func(**kwargs):
            try:
                return view.currentResults.extended_stats()
            except Exception:
                traceback.print_exc()
                return None

        def done(results: dict):
            if results is not None:
                gl = pg.GraphicsLayoutWidget(show=True, title='Extended Statistics')

                for idx, (key, value) in enumerate(results.items()):
                    plot = gl.ci.addPlot(title=key, row=0, col=idx)

                    items = ClustersScatter.histogram(value['counts'], bin_size=1)

                    plot.setLabel('bottom', value['xlabel'])
                    plot.setLabel('left', value['ylabel'])
                    plot.showGrid(x=True, y=True, alpha=0.2)
                    for item in items:
                        plot.addItem(item)

                view.extended_stats_plot = gl
                gl.show()

        view.runAsync(
            work_func,
            finished_message='Extended statistics calculated.',
            done=done,
        )

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
                    **view.localization_widget.optionsDialog(
                        OptionsModes.DBSCAN
                    ).values()
                )
            except Exception:
                traceback.print_exc()
                return None

        view.runAsync(
            work_func,
            finished_message='DBSCAN clustering completed.',
        )

    @classmethod
    def hdbscan_clustering(cls, view: LocalizationsView):
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.cluster_hdbscan(
                    **view.localization_widget.optionsDialog(
                        OptionsModes.HDBSCAN
                    ).values()
                )
            except Exception:
                traceback.print_exc()
                return None

        view.runAsync(
            work_func,
            finished_message='HDBSCAN clustering completed.',
        )

    @classmethod
    def optics_clustering(cls, view: LocalizationsView):
        if view.currentResults is None:
            return

        if len(view.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return view.currentResults.cluster_optics(
                    **view.localization_widget.optionsDialog(
                        OptionsModes.OPTICS
                    ).values()
                )
            except Exception:
                traceback.print_exc()
                return None

        view.runAsync(
            work_func,
            finished_message='OPTICS clustering completed.',
        )

    @classmethod
    def density_analysis(cls, view: LocalizationsView):
        '''
        Batch density analysis for multiple FittingResults.

        Parameters
        ----------
        tables : list[FittingResults]
            List of FittingResults to analyze.
        **kwargs : dict
            Additional keyword arguments for density analysis.
        '''
        if view.currentResults is None:
            return

        if (
            len(view.currentResults) <= 0
            or view.currentResults.data[DataColumns.CLUSTER_ID] is None
        ):
            return

        labels = view.currentResults.data.get(DataColumns.CLUSTER_ID)
        if labels is None or len(np.unique(labels)) <= 2:
            logger.warning('No cluster labels found. Run Clustering first.')
            return None


        table = view.currentResults

        options = view.localization_widget.optionsDialog(
            OptionsModes.CLUSTER_DENSITY
        ).values()

        def worker_func(**kwargs):
            try:
                return get_cluster_density(
                    table.data[DataColumns.X],
                    table.data[DataColumns.Y],
                    table.data[DataColumns.Z],
                    table.data[DataColumns.CLUSTER_ID],
                    **options,
                )
            except Exception:
                logger.error('Error during density analysis.')
                traceback.print_exc()

        def done(results):
            if results is not None:
                density_plot = DensityAnalysisPlot.plot_density_results(
                    table.data[DataColumns.X],
                    table.data[DataColumns.Y],
                    table.data[DataColumns.Z],
                    table.data[DataColumns.CLUSTER_ID],
                    *results,
                )

                view._density_plot_widget = density_plot

                density_plot.show()

        view.runAsync(
            worker_func,
            done=done,
            finished_message='Density analysis completed.',
        )


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
                kwargs = view.localization_widget.toDict()
                sres_img = LocalizationRenderHelper.batch_render_loc(
                    view.fittingResults, force_bin_size=True, **kwargs
                )[view.image_layers.currentIndex]
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
