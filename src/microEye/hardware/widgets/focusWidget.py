from time import perf_counter_ns

import numpy as np
import pyqtgraph as pg

from microEye.hardware.stages.stabilizer import (
    Axis,
    FocusPlot,
    FocusStabilizer,
    FocusStabilizerParams,
    FocusStabilizerView,
    RejectionMethod,
    StabilizationMethods,
)
from microEye.qt import Qt, QtCore, QtWidgets
from microEye.utils.gui_helper import GaussianOffSet
from microEye.utils.thread_worker import QThreadWorker


class focusWidget(QtWidgets.QDockWidget):
    __plot_refs = {
        FocusPlot.LOCALIZATIONS: None,
        FocusPlot.LINE_PROFILE: None,
        FocusPlot.LINE_PROFILE_FIT: None,
        FocusPlot.X_SHIFT: {
            'mean': None,
            'rois': [],
        },
        FocusPlot.Y_SHIFT: {
            'mean': None,
            'rois': [],
        },
        FocusPlot.Z_SHIFT: None,
        FocusPlot.XY_POINTS: None,
        FocusPlot.Z_HISTOGRAM: None,
    }

    def __init__(self):
        '''
        Initialize the focusWidget instance.

        Set up the GUI layout, including the ROI settings, buttons, and graph widgets.
        '''
        super().__init__('Focus Stabilization')

        # Remove close button from dock widgets
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        self.focusStabilizerView = FocusStabilizerView()

        self._updating = False
        self._last_range_update = perf_counter_ns()

        self.init_layout()

        self.connectUpdateGui()

    def init_layout(self):
        # display tab
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Graphics Layout
        pg.setConfigOptions(antialias=False, imageAxisOrder='row-major')

        self.image_widget = pg.GraphicsView()
        # remove margins from the plot widget
        self.image_widget.setContentsMargins(0, 0, 0, 0)

        # IR Camera GraphView
        self.view_box = pg.ViewBox()
        self.image_widget.setCentralItem(self.view_box)
        self.view_box.setAspectLocked()
        self.view_box.setAutoVisible(True)
        self.view_box.setDefaultPadding(0.005)
        self.view_box.enableAutoRange()
        self.view_box.invertY(True)

        self._image_item = pg.ImageItem(axisOrder='row-major')
        self._image_item.setImage(np.random.normal(size=(512, 512)))

        # --- ROI Items ---
        # Line ROI for REFLECTION
        self.line_roi = pg.ROI(
            pg.Point(50.5, 25.5), pg.Point(100, 100), angle=0, pen='r'
        )
        self.line_roi.addTranslateHandle([0, 0], [0, 1])
        self.line_roi.addScaleRotateHandle([0, 1], [0, 0])

        # Rect ROI for BEADS/ASTIGMATIC/HYBRID
        self.rect_roi_z = pg.RectROI([50.5, 25.5], [100, 100], pen='g')
        self.rect_roi_z.setZValue(10)
        self.rect_roi_z.setVisible(False)
        self.rect_roi_xy = pg.RectROI([150.5, 150.5], [100, 100], pen='b')
        self.rect_roi_xy.setZValue(10)
        self.rect_roi_xy.setVisible(False)

        self.set_rois()

        # Line profile graph
        self._line_profile = pg.PlotWidget()
        self._line_profile.setContentsMargins(0, 0, 0, 0)
        self._line_profile.plotItem.setLabel('bottom', 'Pixel', **self.labelStyle)
        self._line_profile.plotItem.setLabel(
            'left', 'Intensity [ADU]', **self.labelStyle
        )
        self._line_profile.plotItem.showGrid(x=True, y=True)

        # X Shift Graph
        self._x_shift = pg.PlotWidget()
        self._x_shift.setContentsMargins(0, 0, 0, 0)
        self._x_shift.plotItem.setLabel('bottom', 'Time [s]', **self.labelStyle)
        self._x_shift.plotItem.setLabel('left', 'X', **self.labelStyle)
        self._x_shift.plotItem.showGrid(x=True, y=True)

        # Y Shift Graph
        self._y_shift = pg.PlotWidget()
        self._y_shift.setContentsMargins(0, 0, 0, 0)
        self._y_shift.plotItem.setLabel('bottom', 'Time [s]', **self.labelStyle)
        self._y_shift.plotItem.setLabel('left', 'Y', **self.labelStyle)
        self._y_shift.plotItem.showGrid(x=True, y=True)

        # Z Shift Graph
        self._z_shift = pg.PlotWidget()
        self._z_shift.setContentsMargins(0, 0, 0, 0)
        self._z_shift.plotItem.setLabel('bottom', 'Time [s]', **self.labelStyle)
        self._z_shift.plotItem.showGrid(x=True, y=True)
        self._z_shift.plotItem.setLabel('left', 'Z', **self.labelStyle)

        # XY Scatter Graph
        self._xy_scatter = pg.PlotWidget()
        self._xy_scatter.setContentsMargins(0, 0, 0, 0)
        self._xy_scatter.plotItem.setLabel('bottom', 'X', **self.labelStyle)
        self._xy_scatter.plotItem.setLabel('left', 'Y', **self.labelStyle)
        self._xy_scatter.plotItem.showGrid(x=True, y=True)

        # Z Histogram Graph
        self._z_hist = pg.PlotWidget()
        self._z_hist.setContentsMargins(0, 0, 0, 0)
        self._z_hist.plotItem.setLabel('bottom', 'Z', **self.labelStyle)
        self._z_hist.plotItem.setLabel('left', 'Counts', **self.labelStyle)

        self._init_plot_refs()

        def roiChanged(cls):
            x, y = self.getRoiCoords(cls)

            if cls in [self.rect_roi_z, self.line_roi]:
                roi_manager = FocusStabilizer.instance().roi_manager
                z_roi = roi_manager.get_roi('z')
                z_roi.x1, z_roi.x2 = x
                z_roi.y1, z_roi.y2 = y

            if cls is self.rect_roi_xy:
                roi_manager = FocusStabilizer.instance().roi_manager
                xy_roi = roi_manager.get_roi('xy')
                xy_roi.x1, xy_roi.x2 = x
                xy_roi.y1, xy_roi.y2 = y

            self.set_rois()

        self.line_roi.sigRegionChangeFinished.connect(roiChanged)
        self.rect_roi_z.sigRegionChangeFinished.connect(roiChanged)
        self.rect_roi_xy.sigRegionChangeFinished.connect(roiChanged)

        self.view_box.addItem(self._image_item)
        self.view_box.addItem(self.line_roi)
        self.view_box.addItem(self.rect_roi_z)
        self.view_box.addItem(self.rect_roi_xy)

        graphicsLayoutWidget = self._init_plots_layout()

        splitter.addWidget(graphicsLayoutWidget)
        splitter.addWidget(self.focusStabilizerView)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        self.setWidget(splitter)

    def _init_plots_layout(self):
        graphicsLayoutWidget = QtWidgets.QWidget()

        plots_layout = QtWidgets.QVBoxLayout()
        plots_layout.setSpacing(0)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        graphicsLayoutWidget.setLayout(plots_layout)

        # First row with image and scatter/histogram
        row_0 = QtWidgets.QHBoxLayout()
        row_0.setSpacing(0)
        row_0.setContentsMargins(0, 0, 0, 0)
        row_0.addWidget(self.image_widget, 2)

        row_0_col_1 = QtWidgets.QVBoxLayout()
        row_0_col_1.addWidget(self._xy_scatter, 1)
        row_0_col_1.addWidget(self._z_hist, 1)
        row_0.addLayout(row_0_col_1, 1)

        plots_layout.addLayout(row_0, 2)
        plots_layout.addWidget(self._line_profile, 1)

        # Third row with X, Y, Z shift plots
        row_2 = QtWidgets.QHBoxLayout()
        row_2.setSpacing(0)
        row_2.setContentsMargins(0, 0, 0, 0)
        row_2.addWidget(self._x_shift, 1)
        row_2.addWidget(self._y_shift, 1)
        row_2.addWidget(self._z_shift, 1)

        plots_layout.addLayout(row_2, 2)

        qss = graphicsLayoutWidget.styleSheet()

        qss += r'''
        PlotWidget {
            padding: 0px;
            margin: 0px;
        }
        '''
        graphicsLayoutWidget.setStyleSheet(qss)

        self._update_plot_visibility(FocusStabilizer.instance().method())

        return graphicsLayoutWidget

    def _init_plot_refs(self):
        # Add ScatterPlotItem for localizations
        scatter_locs = pg.ScatterPlotItem()
        scatter_locs.setBrush(color='b')
        scatter_locs.setSymbol('x')
        scatter_locs.setZValue(999)  # Ensure points are on top of image
        self.view_box.addItem(scatter_locs)

        self.__plot_refs = {
            FocusPlot.LOCALIZATIONS: scatter_locs,
            FocusPlot.LINE_PROFILE: self._line_profile.plotItem.plot(pen='r'),
            FocusPlot.LINE_PROFILE_FIT: self._line_profile.plotItem.plot(pen='b'),
            FocusPlot.X_SHIFT: {
                'mean': self._x_shift.plotItem.plot(pen='r'),
                'rois': [],
            },
            FocusPlot.Y_SHIFT: {
                'mean': self._y_shift.plotItem.plot(pen='r'),
                'rois': [],
            },
            FocusPlot.Z_SHIFT: self._z_shift.plotItem.plot(pen='r'),
            FocusPlot.XY_POINTS: self._xy_scatter.plotItem.plot(
                [], pen=None, symbolBrush='r', symbolSize=4, symbolPen=None
            ),
            # z hist fill under the curve
            FocusPlot.Z_HISTOGRAM: self._z_hist.plotItem.plot(
                [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                [1, 1, 5, 5, 2, 2, 4, 4, 3, 3],
                brush='#00b9e7',
                fillLevel=0,  # Fill down to y=0
            ),
        }

    def _update_plot_visibility(self, method: StabilizationMethods):
        is_line = method in [
            StabilizationMethods.REFLECTION,
            StabilizationMethods.HYBRID,
        ]
        # Update ROI display according to method
        self.line_roi.setVisible(is_line)
        self._line_profile.setVisible(is_line)

        is_xy = method in [
            StabilizationMethods.BEADS,
            StabilizationMethods.BEADS_ASTIGMATIC,
            StabilizationMethods.HYBRID,
        ]

        self._xy_scatter.setVisible(is_xy)
        self._x_shift.setVisible(is_xy)
        self._y_shift.setVisible(is_xy)
        self.rect_roi_xy.setVisible(is_xy)

        self.rect_roi_z.setVisible(False)  # Always hide the Z ROI as it's not used

    def connectUpdateGui(self):
        FocusStabilizer.instance().updateViewBox.connect(self.updateViewBox)
        FocusStabilizer.instance().updatePlots.connect(self._update_event)
        self.focusStabilizerView.methodChanged.connect(self._update_plot_visibility)
        self.focusStabilizerView.setRoiActivated.connect(self.set_rois)

    def updateViewBox(self, data: np.ndarray):
        self._image_item.setImage(data, _callSync='off')

    def _update_event(self, kwargs: dict = None):
        '''Worker function to update the plots.'''
        if kwargs is None or self._updating:
            return

        try:
            self._updating = True
            self._update_plots(**kwargs)
        finally:
            self._updating = False

    def _update_plots(self, **kwargs):
        '''Updates the graphs.'''

        _time: np.ndarray = kwargs.get('time')
        _time -= _time[0]  # start from 0

        positions: np.ndarray = kwargs.get('positions')
        X, Y, Z = positions[:, 0], positions[:, 1], positions[:, 2]

        if FocusStabilizer.instance().isFocusStabilized(Axis.Z):
            Z = Z - FocusStabilizer.instance().getParameter()

        Localizations: dict = kwargs.get('localizations')
        line_profile: dict = kwargs.get('line_profile')

        # Line profile
        data = line_profile.get('y')
        xdata = np.arange(len(data))
        self.__plot_refs[FocusPlot.LINE_PROFILE].setData(xdata, data)

        if line_profile.get('fit_params') is not None:
            self.__plot_refs[FocusPlot.LINE_PROFILE_FIT].setData(
                xdata, GaussianOffSet(xdata, *line_profile.get('fit_params'))
            )

        # Shifts
        self.__plot_refs[FocusPlot.X_SHIFT]['mean'].setData(_time, X)
        self.__plot_refs[FocusPlot.Y_SHIFT]['mean'].setData(_time, Y)
        self.__plot_refs[FocusPlot.Z_SHIFT].setData(_time, Z)

        # plot Z histogram
        hist, bin_edges = np.histogram(
            Z, bins=30, range=(np.nanmin(Z), np.nanmax(Z)), density=True
        )

        # Convert bar data to step plot coordinates
        # Double the bin edges and repeat histogram values to create step effect
        x_step = []
        y_step = []

        for i in range(len(hist)):
            x_step.append(bin_edges[i])
            x_step.append(bin_edges[i + 1])
            y_step.append(hist[i])
            y_step.append(hist[i])

        self.__plot_refs[FocusPlot.Z_HISTOGRAM].setData(x_step, y_step)

        # XY Scatter
        self.__plot_refs[FocusPlot.XY_POINTS].setData(X, Y)

        # Localizations
        self.__plot_refs[FocusPlot.LOCALIZATIONS].setData(
            Localizations.get('x', []), Localizations.get('y', [])
        )

        if (
            perf_counter_ns() - self._last_range_update
        ) / 1e9 >= 1.0 and self.focusStabilizerView.get_param_value(
            FocusStabilizerParams.AUTO_RANGE, default=True
        ):
            # update plot ranges every 1 second
            self._x_shift.setXRange(_time[0], _time[-1], padding=0.04)
            self._y_shift.setXRange(_time[0], _time[-1], padding=0.04)
            self._z_shift.setXRange(_time[0], _time[-1], padding=0.04)
            self._x_shift.setYRange(np.nanmin(X), np.nanmax(X), padding=0.04)
            self._y_shift.setYRange(np.nanmin(Y), np.nanmax(Y), padding=0.04)
            self._z_shift.setYRange(np.nanmin(Z), np.nanmax(Z), padding=0.04)
            self._xy_scatter.setXRange(np.nanmin(X), np.nanmax(X), padding=0.04)
            self._xy_scatter.setYRange(np.nanmin(Y), np.nanmax(Y), padding=0.04)

            if data is not None and len(data) > 0:
                self._line_profile.setYRange(0, np.nanmax(data), padding=0.04)
                self._line_profile.setXRange(xdata[0], xdata[-1], padding=0.04)

            self._z_hist.setXRange(bin_edges[0], bin_edges[-1], padding=0.04)
            self._z_hist.setYRange(0, np.nanmax(hist), padding=0.04)
            self._last_range_update = perf_counter_ns()

        QtCore.QThread.msleep(10)  # slight delay to ensure GUI updates

    def getRoiCoords(self, roi: pg.ROI = None):
        if isinstance(roi, pg.RectROI):
            x1, y1 = roi.pos()
            x2, y2 = roi.pos() + roi.size()
            return [x1, x2], [y1, y2]
        else:
            x1, y1 = roi.pos()
            angle_rad = np.radians(-roi.angle())
            length = roi.size()[1]  # FIXED: use [0] for length
            dx = length * np.sin(angle_rad)
            dy = length * np.cos(angle_rad)
            x2, y2 = x1 + dx, y1 + dy
            return [x1, x2], [y1, y2]

    def set_rois(self):
        roi_manager = FocusStabilizer.instance().roi_manager

        z_roi = roi_manager.get_roi('z')

        dx = z_roi.x2 - z_roi.x1
        dy = z_roi.y2 - z_roi.y1
        length = np.hypot(dx, dy)
        angle = -np.degrees(np.arctan2(dx, dy))

        self.line_roi.setPos([z_roi.x1, z_roi.y1], finish=False)
        self.line_roi.setSize([1, length], finish=False)
        self.line_roi.setAngle(angle, finish=False)

        self.rect_roi_z.setPos([z_roi.x1, z_roi.y1], finish=False)
        self.rect_roi_z.setSize([dx, dy], finish=False)

        xy_roi = roi_manager.get_roi('xy')

        self.rect_roi_xy.setPos([xy_roi.x1, xy_roi.y1], finish=False)
        self.rect_roi_xy.setSize(
            [xy_roi.x2 - xy_roi.x1, xy_roi.y2 - xy_roi.y1], finish=False
        )

    def get_config(self) -> dict:
        return self.focusStabilizerView.get_config()

    def load_config(self, config: dict) -> None:
        if not isinstance(config, dict):
            raise TypeError('Configuration must be a dictionary.')

        self.focusStabilizerView.load_config(config)

    def __str__(self):
        return 'Focus Stabilization Widget'
