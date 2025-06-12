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
    UNIQUE_COLUMNS,
    DataColumns,
    FittingResults,
)
from microEye.analysis.fitting.results_stats import resultsStatsWidget
from microEye.analysis.fitting.tardis import TARDIS_Widget
from microEye.analysis.processing import FRC_resolution_binomial, plot_frc
from microEye.analysis.rendering import *
from microEye.analysis.viewer.layers_widget import ImageItemsWidget
from microEye.analysis.viewer.volume import PointCloudViewer, VolumeViewerWindow
from microEye.qt import Qt, QtCore, QtGui, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.gui_helper import *
from microEye.utils.thread_worker import QThreadWorker


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
        if self.projection_cbox.currentData() in [3, 4, 5]:
            self.position_label.setVisible(True)
            self.position.setVisible(True)

            projection = self.projection_cbox.currentData() - 3
            if projection == 0:
                column = DataColumns.Z
            elif projection == 1:
                column = DataColumns.Y
            else:
                column = DataColumns.X

            self.position_label.setText('Z Position [nm]:')
            self.position.setRange(
                self.currentResults.data[column].min(),
                self.currentResults.data[column].max(),
            )
            self.position.setValue(self.position.minimum())
        else:
            self.position_label.setVisible(False)
            self.position.setVisible(False)

    def render_loc(self):
        '''Update the rendered super-res image.

        Returns
        -------
        ndarray or None
            Rendered super-res image or None.
        '''
        if not self.fittingResults:
            return

        images = [None] * len(self.fittingResults)

        render_idx = self.render_cbox.currentData()
        projection = self.projection_cbox.currentData()
        xy_bin = self.xy_binsize.value()
        z_bin = self.z_binsize.value()
        auto_levels = self.auto_level.isChecked()

        renderer = BaseRenderer(xy_bin, z_bin, RenderModes(render_idx))

        for idx, localizations in enumerate(self.fittingResults):
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
                self.view_box.setAspectLocked(True, aspect)
            elif 3 <= projection < 6:
                slice_proj = projection - 3
                aspect = 1 if slice_proj == 0 else xy_bin / z_bin
                self.view_box.setAspectLocked(True, aspect)
                width = z_bin if slice_proj == 0 else xy_bin

                rendered_img = renderer.render_slice(
                    Projection(projection),
                    localizations.data[DataColumns.X],
                    localizations.data[DataColumns.Y],
                    localizations.data[DataColumns.Z],
                    localizations.data[DataColumns.INTENSITY],
                    self.position.value(),
                    width,
                )
            else:
                continue

            self.setImage(rendered_img, index=idx, autoLevels=auto_levels)
            images[idx] = rendered_img

        return images

    def extract_z(self):
        '''Extract Z values from the fitting results.'''
        if self.currentResults is None:
            return None
        if len(self.currentResults) <= 0:
            return

        if self.cal_curve_cbox.currentData() == stats.CurveFitMethod.LINEAR:
            self.currentResults.data[DataColumns.Z] = (
                (
                    self.currentResults.data[DataColumns.X_SIGMA]
                    / self.currentResults.data[DataColumns.Y_SIGMA]
                )
                - self.z_cal_intercept.value()
            ) / self.z_cal_slope.value()
            self.results_plot.setData(self.currentResults.dataFrame())
        elif self.cal_curve_cbox.currentData() == stats.CurveFitMethod.CSPLINE:
            if self.z_cal_curve:
                self.currentResults.data[DataColumns.Z] = (
                    stats.CurveAnalyzer.get_z_from_y_array_optimized(
                        self.currentResults.data[DataColumns.X_SIGMA]
                        / self.currentResults.data[DataColumns.Y_SIGMA],
                        self.z_cal_curve,
                        region=[-450, 450],
                    )
                )
                self.results_plot.setData(self.currentResults.dataFrame())

    def _load_calibration(self):
        '''Load a calibration file.'''
        self.z_cal_curve, path = stats.import_fit_curve(
            self,
            os.path.dirname(self.currentResults.path),
        )
        if self.z_cal_curve is not None:
            self.cal_file_path.setText(path)
            if self.z_cal_curve.method == stats.CurveFitMethod.LINEAR:
                self.cal_file_path.setText(path)
                self.cal_curve_cbox.setCurrentIndex(0)
                self.z_cal_slope.setValue(self.z_cal_curve.slope)
                self.z_cal_intercept.setValue(self.z_cal_curve.intercept)
            elif self.z_cal_curve.method == stats.CurveFitMethod.CSPLINE:
                self.cal_file_path.setText(path)
                self.cal_curve_cbox.setCurrentIndex(1)

    def _plot_calibration(self):
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

    def render_3D(self):
        '''
        Render a 3D view of the super-res image.
        '''
        if (
            self.currentResults is None
            or self.currentResults.data[DataColumns.Z] is None
        ):
            return None

        if len(self.currentResults) <= 0:
            return

        renderClass = PointCloudRenderer(
            self.xy_binsize.value(), self.z_binsize.value()
        )
        # Render point cloud
        points, intensities, metadata = renderClass.render(
            self.currentResults.data[DataColumns.X],
            self.currentResults.data[DataColumns.Y],
            self.currentResults.data[DataColumns.Z],
            self.currentResults.data[DataColumns.INTENSITY],
        )
        volume_viewer = PointCloudViewer(points, intensities, metadata)

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
                    self.frame_bins.value(), self.xy_binsize.value(),
                    z_pixel_size=self.z_binsize.value(),
                    projection=Projection(self.projection_cbox.currentData()),
                    renderMode=RenderModes(self.render_cbox.currentData())
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.animation_btn.setDisabled(False)
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
        self.animation_btn.setDisabled(True)
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
        self.localization_widget, self.localization_form = create_widget(
            QtWidgets.QVBoxLayout
        )
        self.tab_widget.addTab(self.localization_widget, 'Localization')

        # Render layout

        self.render_cbox = QtWidgets.QComboBox()
        self.render_cbox.addItem('2D Histogram (Intensity)', 0)
        self.render_cbox.addItem('2D Histogram (Events)', 1)
        self.render_cbox.addItem('2D Gaussian Histogram', 2)

        self.projection_cbox = QtWidgets.QComboBox()
        self.projection_cbox.addItem('XY', 0)
        self.projection_cbox.addItem('XY (slice)', 3)
        self.projection_cbox.addItem('XZ', 1)
        self.projection_cbox.addItem('XZ (slice)', 4)
        self.projection_cbox.addItem('YZ', 2)
        self.projection_cbox.addItem('YZ (slice)', 5)

        self.projection_cbox.currentIndexChanged.connect(lambda: self.update_position())

        self.auto_level = QtWidgets.QCheckBox('Auto-stretch')
        self.auto_level.setChecked(True)

        self.frame_bins = create_spin_box(min_value=2, max_value=1000, initial_value=10)
        self.xy_binsize = create_spin_box(min_value=0, max_value=250, initial_value=10)
        self.z_binsize = create_spin_box(min_value=0, max_value=250, initial_value=50)

        self.position_label = QtWidgets.QLabel('Position [nm]:')
        self.position = create_double_spin_box(
            min_value=0, max_value=200, single_step=10, decimals=6, initial_value=0
        )
        self.position_label.setVisible(False)
        self.position.setVisible(False)
        self.position.valueChanged.connect(lambda: self.render_loc())

        self.refresh_btn = QtWidgets.QPushButton(
            'Refresh 2D View', clicked=lambda: self.render_loc()
        )

        self.display_3d_btn = QtWidgets.QPushButton(
            '3D View', clicked=lambda: self.render_3D()
        )

        self.animation_btn = QtWidgets.QPushButton(
            'Animation', clicked=lambda: self.toAnimation()
        )

        # Render GroupBox
        localization = QtWidgets.QGroupBox('Render')
        flocalization = QtWidgets.QFormLayout()
        localization.setLayout(flocalization)

        flocalization.addRow('Rendering Method:', self.render_cbox)
        flocalization.addRow('Projection:', self.projection_cbox)
        flocalization.addRow('Frame bins:', self.frame_bins)
        flocalization.addRow('XY binsize [nm]:', self.xy_binsize)
        flocalization.addRow('Z binsize [nm]:', self.z_binsize)
        flocalization.addRow(self.position_label, self.position)
        flocalization.addRow(
            create_hbox_layout(
                self.auto_level,
                self.refresh_btn,
                self.display_3d_btn,
                self.animation_btn,
            )
        )
        # End Localization GroupBox

        # Z Extraction
        self.z_cal_curve = None
        z_extract = QtWidgets.QGroupBox('Z Extraction')
        z_extract_form = QtWidgets.QFormLayout()
        z_extract.setLayout(z_extract_form)

        self.cal_curve_cbox = QtWidgets.QComboBox()
        for c in [stats.CurveFitMethod.LINEAR, stats.CurveFitMethod.CSPLINE]:
            self.cal_curve_cbox.addItem(c.name, c)
        self.cal_curve_cbox.currentIndexChanged.connect(
            lambda: [
                z_extract_form.setRowVisible(
                    idx, self.cal_curve_cbox.currentIndex() == 0
                )
                for idx in [1, 2]
            ]
        )

        self.z_cal_slope = create_double_spin_box(
            min_value=0, max_value=200, decimals=8, initial_value=0.000800867
        )
        self.z_cal_intercept = create_double_spin_box(
            min_value=0, max_value=200, decimals=8, initial_value=1.00927
        )
        self.load_cal_btn = QtWidgets.QPushButton(
            'Load Z Calibration', clicked=lambda: self._load_calibration()
        )
        self.plot_cal_btn = QtWidgets.QPushButton(
            'Calibration Curve', clicked=lambda: self._plot_calibration()
        )
        # display loaded file path in multiline readonly edit box
        self.cal_file_path = QtWidgets.QTextEdit()
        self.cal_file_path.setReadOnly(True)
        self.cal_file_path.setFixedHeight(60)
        self.cal_file_path.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.cal_file_path.setPlaceholderText('No file loaded')

        self.extract_z_btn = QtWidgets.QPushButton(
            'Extract Z', clicked=lambda: self.extract_z()
        )
        z_extract_form.addRow(
            'Calibration Method:',
            self.cal_curve_cbox,
        )
        z_extract_form.addRow(
            'Z Calibration (slope):',
            self.z_cal_slope,
        )
        z_extract_form.addRow(
            'Z Calibration (intercept):',
            self.z_cal_intercept,
        )
        z_extract_form.addRow(
            create_hbox_layout(self.extract_z_btn, self.load_cal_btn, self.plot_cal_btn)
        )
        z_extract_form.addRow(self.cal_file_path)

        # End Z Extraction

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
        fdrift.addRow(create_hbox_layout(self.drift_cross_btn, self.drift_fdm_btn))
        # End Drift GroupBox

        self.frc_cbox = QtWidgets.QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Odd/Even')
        self.frc_cbox.addItem('Halves')

        precision_btns = QtWidgets.QHBoxLayout()
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
        precision_btns.addWidget(self.frc_res_btn)
        precision_btns.addWidget(self.NeNA_btn)
        precision_btns.addWidget(self.tardis_btn)
        fprecision.addRow(precision_btns)
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

        self.import_loc_btn = QtWidgets.QPushButton(
            'Import', clicked=lambda: self.import_loc()
        )
        self.export_loc_btn = QtWidgets.QPushButton(
            'Export', clicked=lambda: self.export_loc()
        )

        self.localization_form.addWidget(localization)
        self.localization_form.addWidget(z_extract)
        self.localization_form.addWidget(drift)
        self.localization_form.addWidget(precision)
        self.localization_form.addWidget(nearestN)

        self.localization_form.addLayout(
            create_hbox_layout(self.import_loc_btn, self.export_loc_btn)
        )
        self.localization_form.addStretch()

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
            'Apply Filters', clicked=lambda: self.apply_filters()
        )
        self.apply_filters_btn.setToolTip(
            'Applies the filters permanently to fitting results.'
        )
        self.zero_coords_btn = QtWidgets.QPushButton(
            'Zero Coordinates', clicked=lambda: self.zero_coordinates()
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
            'Shift Data', clicked=lambda: self.shift_data()
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

    def FRC_estimate(self):
        '''Estimate FRC (Fourier Ring Correlation) resolution.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        frc_method = self.frc_cbox.currentText()
        data = self.currentResults.toRender()

        def work_func(**kwargs):
            try:
                return FRC_resolution_binomial(
                    np.c_[data[0], data[1], data[2]],
                    self.xy_binsize.value(),
                    frc_method,
                )
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.frc_res_btn.setDisabled(False)
            if results is not None:
                plot_frc(*results)

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.frc_res_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def NeNA_estimate(self):
        '''Estimate Nearest-Neighbor Analysis for localization precision.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.nn_trajectories(0, 200, 0, 1)
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.NeNA_btn.setDisabled(False)
            if results is not None:
                self.currentResults = results
                self.results_plot.setData(self.currentResults.dataFrame())

                self.NeNA_widget = NeNA_Widget(
                    self.currentResults.neighbour_dist, self.currentResults.trackID
                )

                res = self.NeNA_widget.exec()

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.NeNA_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def TARDIS_analysis(self):
        '''Perform TARDIS analysis.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        self.tardis = TARDIS_Widget(
            self.currentResults.frames,
            self.currentResults.locX,
            self.currentResults.locY,
            self.currentResults.locZ,
        )
        self.tardis.startWorker.connect(
            lambda worker: self._threadpool.start(worker)
        )
        self.tardis.show()

    def drift_cross(self):
        '''Perform drift cross-correlation.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.drift_cross_correlation(
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
                self.currentResults = results[0]
                self.results_plot.setData(self.currentResults.dataFrame())
                plot_drift(*results[2])

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.drift_cross_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def drift_fdm(self):
        '''Perform drift correction using fiducial markers.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.drift_fiducial_marker()
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.drift_fdm_btn.setDisabled(False)
            if results is not None:
                self.currentResults = results[0]
                self.results_plot.setData(self.currentResults.dataFrame())
                plot_drift(*results[1])
                self.render_loc()

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.drift_fdm_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def nneigh_merge(self):
        '''Perform nearest-neighbor merging.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.nearest_neighbour_merging(
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
                self.currentResults = results
                self.results_plot.setData(self.currentResults.dataFrame())

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.nneigh_merge_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def nneigh(self):
        '''Perform nearest-neighbor analysis.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.nn_trajectories(
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
                self.currentResults = results
                self.results_plot.setData(self.currentResults.dataFrame())

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.nneigh_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def merge(self):
        '''Merge tracks.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        def work_func(**kwargs):
            try:
                return self.currentResults.merge_tracks(self.nn_max_length.value())
            except Exception:
                traceback.print_exc()
                return None

        def done(results):
            self.merge_btn.setDisabled(False)
            if results is not None:
                self.currentResults = results
                self.results_plot.setData(self.currentResults.dataFrame())

        self.worker = QThreadWorker(work_func)
        self.worker.signals.result.connect(done)
        # Execute
        self.merge_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def populate_data(self):
        '''Populate the data filters.'''
        self.results_plot.setData(self.currentResults.dataFrame())

    def apply_filters(self):
        '''Apply data filters to the fitting results.'''
        if (
            not hasattr(self.results_plot, 'filtered')
            or self.results_plot.filtered is None
        ):
            self.results_plot.update()

        self.currentResults = FittingResults.fromDataFrame(
            self.results_plot.filtered, 1, path=self.currentResults.path
        )
        self.results_plot.setData(self.currentResults.dataFrame())
        self.render_loc()
        print('Filters applied.')

    def zero_coordinates(self):
        '''Zero the fitting results coordinates.'''
        if self.currentResults is not None:
            self.currentResults.zero_coordinates()
            self.results_plot.setData(self.currentResults.dataFrame())
            self.render_loc()
            print('Coordinates reset.')

    def toggle_track_plots(self):
        '''Toggle the track plots.'''
        if self.currentResults is not None:
            self.results_plot.toggle_track_plots()

    def shift_data(self):
        '''Shift the data by a specified amount.'''
        if self.currentResults is None:
            return

        if len(self.currentResults) <= 0:
            return

        axis: DataColumns = self.data_shift_axis.currentData()
        shift = self.data_shift.value()
        direction = -1 if self.data_shift_dir.isChecked() else 1

        self.currentResults.data[axis] += shift * direction

        self.results_plot.clear()
        self.render_loc()

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
                render_idx = self.render_cbox.currentData()
                renderClass = BaseRenderer(
                    self.xy_binsize.value(), RenderModes(render_idx)
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

    def export_loc(self, filename=None):
        '''Export the fitting results into a file.

        Parameters
        ----------
        filename : str, optional
            File path; if None, a save file dialog is shown, by default None.
        '''
        if self.currentResults is None:
            return

        if filename is None:
            if not self.export_options.exec():
                return

            filename, _ = getSaveFileName(
                self,
                'Export localizations',
                filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
                directory=os.path.dirname(self.currentResults.path),
            )

        if len(filename) > 0:
            options = self.export_options.toList()

            dataFrame = self.currentResults.dataFrame()
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
                        float_format=self.export_options.export_precision.text(),
                        sep='\t',
                        encoding='utf-8',
                    )
                elif '.h5' in filename:
                    dataFrame[exp_columns].to_hdf(
                        filename, key='microEye', index=False, complevel=0
                    )

            if 'Super-res image' in options:
                sres_img = self.render_loc()[self.image_layers.currentIndex]
                tf.imwrite(
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
            directory=os.path.dirname(self.currentResults.path),
        )

        if len(filename) > 0:
            results = FittingResults.fromFile(filename, 1)

            if results is not None:
                self.add_data_layer(results)
                print('Done importing results.')
            else:
                print('Error importing results.')
