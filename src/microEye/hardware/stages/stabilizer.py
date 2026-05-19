import json
import logging
import time
import traceback
from enum import Enum, auto
from queue import LifoQueue
from threading import Event
from typing import Optional, Union

import numpy as np
from pyqtgraph.parametertree import Parameter

from microEye.hardware.stages.stabilization.controller import (
    Axis,
    PIDController,
    RejectionMethod,
)
from microEye.hardware.stages.stabilization.methods import (
    CalibrationManager,
    DataLogger,
    FiducialStrategy,
    HybridStrategy,
    PositionTracker,
    ReflectionStrategy,
    ROIManager,
    StabilizationMethods,
    StabilizationStrategy,
)
from microEye.qt import (
    QApplication,
    QtCore,
    QtWidgets,
    Signal,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.parameter_tree import Tree
from microEye.utils.thread_worker import QThreadWorker

FRAME_STATS = ['mean', 'median']


class FocusStabilizerParams(Enum):
    '''
    Enum class defining FocusStabilizer parameters.
    '''

    N_FRAMES = 'Number of Frames'
    FRAMES_STATS = 'Frames Statistics'
    LINE_WIDTH = 'Line ROI Width'
    EXTERNAL_PREVIEW = 'External Preview'
    SAVE = 'Save'
    LOAD = 'Load'

    FOCUS_TRACKING = 'Focus Stabilization'
    FOCUS_PARAMETER = 'Focus Parameter'
    Z_STABILIZATION = 'Stabilization (Z)'
    XY_STABILIZATION = 'Stabilization (XY)'

    STABILIZATION_METHOD = 'Method'
    METHOD_PARAMS = 'Method Parameters'

    CALIBRATION = 'Calibration'

    USE_CAL = 'Calibration.Active'
    X_CALIBRATION = 'Calibration.(X) [nm/pixel]'
    Y_CALIBRATION = 'Calibration.(Y) [nm/pixel]'
    Z_CALIBRATION = 'Calibration.(Z) [nm/pixel]'
    ADJUST_SET_POINT = 'Calibration.Adjust Set Point'

    XY_OUTLIER_REJECTION = 'XY Outlier Rejection'
    XY_OUTLIER_REJECTION_METHOD = 'XY Outlier Rejection.Method'
    XY_OUTLIER_REJECTION_THRESHOLD = 'XY Outlier Rejection.Threshold'
    XY_OUTLIER_REJECTION_MIN_POINTS = 'XY Outlier Rejection.Min Points'

    X = 'X'
    X_K_P = 'X.K_P'
    X_K_I = 'X.K_I'
    X_K_D = 'X.K_D'
    X_INVERTED = 'X.Inverted'

    Y = 'Y'
    Y_K_P = 'Y.K_P'
    Y_K_I = 'Y.K_I'
    Y_K_D = 'Y.K_D'
    Y_INVERTED = 'Y.Inverted'

    Z = 'Z'
    Z_K_P = 'Z.K_P'
    Z_K_I = 'Z.K_I'
    Z_K_D = 'Z.K_D'
    Z_INVERTED = 'Z.Inverted'

    FT_TAU = 'Tau'
    FT_ERROR_TH = 'Error Threshold'
    PEAK_ACQUIRE = 'Start Peak Acquisition'
    PEAK_STOP = 'Stop Peak Acquisition'

    PREVIEW_IMAGE = 'Preview Image'
    AUTO_RANGE = 'Auto Range'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')


class FocusPlot(Enum):
    LINE_PROFILE = auto()
    LINE_PROFILE_FIT = auto()
    X_SHIFT = auto()
    Y_SHIFT = auto()
    Z_SHIFT = auto()
    XY_POINTS = auto()
    Z_HISTOGRAM = auto()
    LOCALIZATIONS = auto()


class FocusStabilizer(QtCore.QObject):
    # Class attribute to hold the single instance
    _instance = None

    TIME_POINTS = 1000

    updateViewBox = Signal(object)
    updatePlots = Signal(dict)
    moveStage = Signal(bool, int)
    moveXYZ = Signal(float, float, float)
    parameterChanged = Signal(object)
    calCoeffChanged = Signal(np.ndarray)
    focusStabilizationToggled = Signal(bool, Axis)

    def __new__(cls, *args, **kwargs):
        # If the single instance doesn't exist, create a new one
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        # Return the single instance
        return cls._instance

    @classmethod
    def instance(cls):
        if cls._instance is None:
            return FocusStabilizer()

        return cls._instance

    def __init__(self):
        super().__init__()

        self.__buffer = LifoQueue()

        # History -> tracker
        self.__tracker = PositionTracker(self.TIME_POINTS)

        # ROI manager
        self.roi_manager = ROIManager()

        # Default method
        self.__method = StabilizationMethods.REFLECTION

        # PID controller
        self.__controller = PIDController(Kp=6.5, Ki=0.0, Kd=0.0)

        # Calibration manager
        self.__cal_manager = CalibrationManager()

        # Adjust set point flag
        self.__adjust_set_point = False

        self.__inverted = np.zeros((3,), dtype=bool)

        self.__n_frames = 1
        self.__frames_stats_method = 'mean'

        # Stabilization flags
        self.__z_stabilization = False
        self.__xy_stabilization = False

        # Initial positions
        self.__initial_z_position = None
        self.__initial_xy_positions = None

        self.__tau = 0.0
        self.__err_th = 0.001

        self.__preview = True

        self._exec_time = 0

        # Data logging
        self.__logger = DataLogger()

        self.file = None
        self.num_frames_saved = 0

        # Method strategies
        self._strategies: dict[StabilizationMethods, StabilizationStrategy] = {
            StabilizationMethods.REFLECTION: ReflectionStrategy(),
            StabilizationMethods.BEADS: FiducialStrategy(False),
            StabilizationMethods.BEADS_ASTIGMATIC: FiducialStrategy(True),
            StabilizationMethods.HYBRID: HybridStrategy(),
        }

        self.exit_event = Event()

        # Stop worker when app is closing
        app = QApplication.instance()
        app.aboutToQuit.connect(FocusStabilizer.instance().exit_event.set)

    @property
    def buffer(self):
        return self.__buffer

    @property
    def tracker_snapshot(self):
        return self.__tracker.snapshot()

    def get_current_strategy(self) -> Optional[StabilizationStrategy]:
        return self._strategies.get(self.__method, None)

    def get_current_strategy_param_defs(self) -> list[dict]:
        strategy = self.get_current_strategy()
        return [] if strategy is None else strategy.get_tunable_params()

    def get_current_strategy_param_values(self) -> dict:
        strategy = self.get_current_strategy()
        return {} if strategy is None else strategy.get_param_values()

    def set_current_strategy_param(self, name: str, value) -> bool:
        strategy = self.get_current_strategy()
        if strategy is None:
            return False
        return bool(strategy.set_param(name, value))

    def set_current_strategy_params(self, values: dict):
        strategy = self.get_current_strategy()
        if strategy is not None:
            strategy.set_param_values(values)

    def method(self):
        '''
        Gets the current method for focus stabilization.

        Returns
        -------
        str
            The current method for focus stabilization.
        '''
        return self.__method

    def setMethod(self, method: Union[str, StabilizationMethods]):
        '''
        Sets the method for focus stabilization.

        Parameters
        ----------
        method : StabilizationMethods
            The new method for focus stabilization.
        '''
        if isinstance(method, str) and method in [
            m.value for m in StabilizationMethods
        ]:
            method = StabilizationMethods(method)

        if not isinstance(method, StabilizationMethods):
            raise ValueError(
                f'Invalid method: {method}.'
                f'Options: {[str(m) for m in StabilizationMethods]}'
            )

        self.__method = method
        self.__initial_z_position = None
        self.__initial_xy_positions = None

        self.__controller.reset()
        self.__tracker.reset()

    def calCoeff(self, axis: Optional[Axis]):
        '''
        Gets the current calibration coefficient.

        Returns
        -------
        float
            The current calibration coefficient.
        '''
        return self.__cal_manager.get_coefficient(axis)

    def setCalCoeff(self, value: float, axis: Axis):
        '''
        Sets the calibration coefficient.

        Parameters
        ----------
        value : float
            The new calibration coefficient.
        '''
        self.__cal_manager.set_coefficient(axis, value)
        self.calCoeffChanged.emit(self.__cal_manager.get_coefficient())

    def preview(self):
        '''
        Check if preview image is enabled.

        Returns
        -------
        bool
            True if preview image is enabled, False otherwise.
        '''
        return self.__preview

    def setPreview(self, value: bool):
        '''
        Set the flag indicating whether to preview image.

        Parameters
        ----------
        value : bool
            The flag value.
        '''
        self.__preview = value

    def useCal(self):
        '''
        Check if pixel calibration coefficient is used.

        Returns
        -------
        bool
            True if pixel calibration coefficient is used, False otherwise.
        '''
        return self.__cal_manager.is_calibration_enabled()

    def setUseCal(self, use_cal: bool):
        '''
        Set the flag indicating whether to use pixel calibration coefficient.

        Parameters
        ----------
        use_cal : bool
            The flag value.
        '''
        self.__cal_manager.enable_calibration(use_cal)

    def isAdjustSetPoint(self):
        '''
        Check if adjust set point option is enabled.

        Returns
        -------
        bool
            True if adjust set point option is enabled, False otherwise.
        '''
        return self.__adjust_set_point

    def setAdjustSetPoint(self, adjust: bool):
        '''
        Set the flag indicating whether to adjust the set point.

        Parameters
        ----------
        adjust : bool
            The flag value.
        '''
        self.__adjust_set_point = adjust

    def isInverted(self, axis: Optional[Axis]):
        '''
        Check if the direction is inverted.

        Returns
        -------
        bool
            True if the direction is inverted, False otherwise.
        '''
        if axis is None:
            return self.__inverted.copy()

        return self.__inverted[axis.axis_index()]

    def setInverted(self, inverted: bool, axis: Axis):
        '''
        Set the flag indicating whether the direction is inverted.

        Parameters
        ----------
        inverted : bool
            The flag value.
        '''
        self.__inverted[axis.axis_index()] = inverted

    def set_n_frames(self, n_frames: int):
        '''
        Set the number of frames to be used for statistics.

        Parameters
        ----------
        n_frames : int
            The number of frames.
        '''
        self.__n_frames = max(1, n_frames)

    def set_frames_stats_method(self, method: str):
        """
        Set the method to calculate the statistics of frames.

        Parameters
        ----------
        method : str
            The method name. Options: 'mean', 'median'.
        """
        self.__frames_stats_method = method if method in FRAME_STATS else 'mean'

    def __len__(self):
        return len(self.__buffer.queue)

    def put(self, value: np.ndarray):
        self.__buffer.put(value)

    def bufferSize(self):
        return len(self.__buffer.queue)

    def isEmpty(self):
        return self.__buffer.empty()

    def hasFrames(self):
        return self.bufferSize() >= self.__n_frames

    def getImage(self) -> np.ndarray:
        '''Gets image from frames Queue in LIFO manner.

        Returns
        -------
        np.ndarray
            The image last in buffer.
        '''
        res = None

        frames = []
        for _ in range(self.__n_frames):
            if not self.__buffer.empty():
                frames.append(self.__buffer.get())
        if len(frames) > 1:
            methods = {'mean': np.mean, 'median': np.median}
            res = methods.get(self.__frames_stats_method, np.mean)(
                np.stack(frames), axis=0
            )
        elif frames and len(frames) == 1:
            res = frames[0]
        self.__buffer.queue.clear()

        return res

    def setP(self, P: float, axis: Axis):
        '''
        Set the P value.

        Parameters
        ----------
        P : float
            The P value.
        '''
        self.__controller.set_Kp(P, axis)

    def setI(self, _I: float, axis: Axis):
        '''
        Set the I value.

        Parameters
        ----------
        _I : float
            The I value.
        '''
        self.__controller.set_Ki(_I, axis)

    def setD(self, D: float, axis: Axis):
        '''
        Set the D value.

        Parameters
        ----------
        D : float
            The D value.
        '''
        self.__controller.set_Kd(D, axis)

    def set_gains(
        self,
        axis: Optional[Axis],
        Kp: Union[float, tuple] = 1.0,
        Ki: Union[float, tuple] = 0.0,
        Kd: Union[float, tuple] = 0.0,
    ):
        '''
        Set the controller gains for a specific axis.
        '''
        self.__controller.set_gains(axis, Kp, Ki, Kd)

    def setTau(self, tau: float):
        '''
        Set the tau value.

        Parameters
        ----------
        tau : float
            The tau value.
        '''
        self.__tau = tau

    def setErrorTh(self, error_th: float):
        '''
        Set the error threshold value.

        Parameters
        ----------
        error_th : float
            The error threshold value.
        '''
        self.__err_th = error_th

    def setRejectionMethod(self, method: RejectionMethod):
        '''
        Set the outlier rejection method.

        Parameters
        ----------
        method : RejectionMethod
            The outlier rejection method.
        '''
        self.__controller.rejection_method = method

    def setOutlierThreshold(self, value: float):
        '''
        Set the outlier rejection threshold.

        Parameters
        ----------
        value : float
            The outlier rejection threshold.
        '''
        self.__controller.outlier_threshold = value

    def setOutlierMinPoints(self, value: int):
        '''
        Set the minimum number of points for outlier rejection.

        Parameters
        ----------
        value : int
            The minimum number of points for outlier rejection.
        '''
        self.__controller.outlier_min_points = value

    def getParameter(self):
        '''
        Get the current focus parameter.

        Returns
        -------
        float
            The current focus parameter.
        '''
        return self.__initial_z_position

    def setParameter(self, value: Union[float, np.ndarray], incerement: bool = False):
        '''
        Set the focus parameter.

        Parameters
        ----------
        value : Union[float, np.ndarray]
            The new focus parameter value.
        axis : Axis
            The axis for which to set the focus parameter.
        incerement : bool, optional
            If True, increment the current focus parameter value by the given value,
            otherwise set the focus parameter to the given value.
        '''
        if incerement and isinstance(self.__initial_z_position, (float, int)):
            self.__initial_z_position += value
        else:
            self.__initial_z_position = value
        self.parameterChanged.emit(self.__initial_z_position)

    def isFocusStabilized(self, axis: Axis):
        '''
        Check if automatic Focus Stabilization is enabled.

        Returns
        -------
        bool
            True if automatic Focus Stabilization is enabled, False otherwise.
        '''
        if axis == Axis.Z:
            return self.__z_stabilization
        elif axis in [Axis.X, Axis.Y]:
            return self.__xy_stabilization

    def toggleFocusStabilization(self, axis: Axis, value: bool = None):
        '''
        Toggles automatic Focus Stabilization.

        If `value` is not provided or set to `None`, the method toggles the
        autofocus tracking option based on its current state. If `value` is a
        Boolean value, the method sets the option to that state.

        Parameters
        ----------
        axis : Axis
            The axis for which to toggle the stabilization.
        value : bool, optional
            The new state of the automatic Focus Stabilization. Default is `None`.

        Returns
        -------
        bool
            The current state of the automatic Focus Stabilization.
        '''
        ref = self.__z_stabilization if axis == Axis.Z else self.__xy_stabilization

        ref = not ref if value is None else value

        if axis == Axis.Z:
            self.__z_stabilization = ref
        else:
            self.__xy_stabilization = ref

        self.__controller.reset([Axis.Z] if axis == Axis.Z else [Axis.X, Axis.Y])

        self.focusStabilizationToggled.emit(ref, axis)

        return ref

    def get_config(self):
        '''
        Get the current configuration values of controller and calibration.

        Returns
        -------
        dict
            The current configuration values as a dictionary.
        '''
        return {
            'Kp': self.__controller.get_Kp(None).tolist(),
            'Ki': self.__controller.get_Ki(None).tolist(),
            'Kd': self.__controller.get_Kd(None).tolist(),
            'tau': self.__tau,
            'error_th': self.__err_th,
            'cal_coeff': self.__cal_manager.get_coefficient().tolist(),
            'use_cal': self.useCal(),
            'adjust_set_point': self.isAdjustSetPoint(),
            'inverted': self.__inverted.tolist(),
            'method': str(self.__method),
            'strategy_params': self.get_current_strategy_param_values(),
            'rejection_method': str(self.__controller.rejection_method),
            'rejection_threshold': self.__controller.outlier_threshold,
            'rejection_min_points': self.__controller.outlier_min_points,
            'n_frames': self.__n_frames,
            'frames_stats_method': str(self.__frames_stats_method),
        }

    def startWorker(self):
        self.worker = QThreadWorker(FocusStabilizer.instance().worker_function)
        # Execute
        QtCore.QThreadPool.globalInstance().start(self.worker)

    def worker_function(self, **kwargs):
        '''A worker function running in the threadpool.

        Handles the IR peak fitting and piezo autofocus tracking.
        '''
        counter = 0
        self._exec_time = 0
        self.__tracker.fill_time(time.monotonic_ns() / 1e9)
        QtCore.QThread.msleep(100)
        while not self.exit_event.is_set():
            try:
                # proceed only if the buffer is not empty
                if not self.isEmpty() and self.hasFrames():
                    self.__tracker.advance_time(time.monotonic_ns() / 1e9)

                    self._exec_time = self.__tracker.last_interval * 1e3  # in ms

                    method = self.method()

                    image = self.getImage()

                    if self.preview():
                        self.updateViewBox.emit(image.copy())

                    # fit the parameter and stabilize
                    shifts = self.fit_parameter(image.copy(), method)
                    # stabilize the focus
                    self.stabilize(shifts)
                    # append parameter to the file if file is set
                    self.log_to_file()

                    counter = counter + 1
                QtCore.QThread.usleep(100)  # sleep for 100us
            except Exception as e:
                logging.getLogger(__name__).error(f'Error in worker_function!')
                traceback.print_exc()

        logging.getLogger(__name__).info(
            f'{FocusStabilizer.__name__} worker is terminating.'
        )

    def fit_parameter(self, data: np.ndarray, method: StabilizationMethods):
        """
        Fit the data to the specified method and update the parameter buffer.

        Parameters
        ----------
        data : np.ndarray
            The input data to fit.
        method : StabilizationMethods
            The method to use for fitting. For options check `StabilizationMethods`.

            - 'reflection': Fit a linear ROI to a GaussianOffSet function.
            - 'beads': Fit beads using a 2D Gaussian function and extract average sigma.
        """
        if not isinstance(method, StabilizationMethods):
            raise ValueError(
                f'Invalid method: {method}.'
                f'Options: {[str(m) for m in StabilizationMethods]}'
            )

        shifts = {
            Axis.X: 0.0,
            Axis.Y: 0.0,
            Axis.Z: 0.0,
        }

        append = {
            Axis.X: 0.0,
            Axis.Y: 0.0,
            Axis.Z: 0.0,
        }

        Localizations = {
            'x': [],
            'y': [],
        }

        line_profile = {
            'y': [],
            'fit_params': None,
        }

        if method in self._strategies:
            res = self._strategies[method].fit(
                data, self.roi_manager, self.__cal_manager, self.__controller
            )

            if res is not None:
                z_param = res['params'].get('z', None)
                xy_params = res['params'].get('xy', None)

                fit_params = res.get('fit_params', None)

                line_profile['y'] = res.get('line_profile', [])
                if line_profile['y']:
                    line_profile['fit_params'] = fit_params

                Localizations = res.get('localizations', Localizations)

                _shifts = self._strategies[method].get_shifts(
                    self.getParameter(),
                    z_param,
                    self.__initial_xy_positions,
                    xy_params,
                    self.__cal_manager,
                    self.__controller,
                )

                if z_param is not None:
                    append[Axis.Z] = z_param

                    if not self.isFocusStabilized(Axis.Z):
                        self.setParameter(z_param)
                    else:
                        shifts[Axis.Z] = _shifts[Axis.Z]

                if xy_params is not None:
                    append[Axis.X] = _shifts[Axis.X]
                    append[Axis.Y] = _shifts[Axis.Y]

                    if not self.isFocusStabilized(Axis.X):
                        self.__initial_xy_positions = xy_params
                    else:
                        shifts[Axis.X] = _shifts[Axis.X]
                        shifts[Axis.Y] = _shifts[Axis.Y]

        self.__tracker.append_axis(Axis.X, append[Axis.X])
        self.__tracker.append_axis(Axis.Y, append[Axis.Y])
        self.__tracker.append_axis(Axis.Z, append[Axis.Z])

        time, positions = self.__tracker.snapshot()

        self.updatePlots.emit(
            {
                'time': time,
                'positions': positions,
                'localizations': Localizations,
                'line_profile': line_profile,
            }
        )

        return shifts

    def stabilize(self, shifts: dict[Axis, float]):
        try:
            output = self.__controller.response(
                self.__tracker.last_time, shifts[Axis.X], shifts[Axis.Y], shifts[Axis.Z]
            )

            # output[Axis.X.axis_index()] *= self.isFocusStabilized(Axis.X)
            # output[Axis.Y.axis_index()] *= self.isFocusStabilized(Axis.Y)
            # output[Axis.Z.axis_index()] *= self.isFocusStabilized(Axis.Z)

            for axis in Axis:
                output[axis.axis_index()] *= self.isFocusStabilized(axis)
                if self.isInverted(axis):
                    output[axis.axis_index()] *= -1

            self.moveXYZ.emit(
                output[Axis.X.axis_index()],
                output[Axis.Y.axis_index()],
                output[Axis.Z.axis_index()],
            )
        except Exception as e:
            pass

    def log_to_file(self):
        '''
        Log the current time and positions to the file if logging is active.
        '''
        if self.__logger.is_active():
            self.__logger.log(self.__tracker.last_row)

    def stop_logging(self):
        '''
        Stop logging data to the file.
        '''
        self.__logger.stop()

    def start_logging(self, file: str):
        '''
        Start logging data to the specified file.

        Parameters
        ----------
        file : str
            The path to the file where data will be logged.
        '''
        self.__logger.start(file)

    def is_logging(self):
        '''
        Check if logging is active.

        Returns
        -------
        bool
            True if logging is active, False otherwise.
        '''
        return self.__logger.is_active()


class FocusStabilizerView(Tree):
    PARAMS = FocusStabilizerParams
    setRoiActivated = Signal()
    actionActivated = Signal(str)
    saveActivated = Signal()
    loadActivated = Signal()
    methodChanged = Signal(StabilizationMethods)

    def __init__(
        self,
        parent: Optional['QtWidgets.QWidget'] = None,
        focusStabilizer: FocusStabilizer = None,
    ):
        '''
        Initialize the FocusStabilizerView instance.

        This method initializes the `FocusStabilizer` instance, sets up the signals,
        creates the parameter tree, and sets up the GUI elements.

        Parameters
        ----------
        parent : Optional[QWidget]
            The parent widget.
        focusStabilizer : Optional[`FocusStabilizer`]
            The Focus Stabilizer to be controlled by the GUI.
            If None, a new `FocusStabilizer` instance is created.
        '''
        super().__init__(parent=parent)

        FocusStabilizer.instance().parameterChanged.connect(self.update_focus_parameter)

        FocusStabilizer.instance().calCoeffChanged.connect(
            self.update_calibration_coefficients
        )

        self.setMinimumWidth(325)

    def __str__(self):
        return f'Focus Stabilization View'

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `FocusStabilizerView` class.
        '''
        params = [
            {
                'name': str(FocusStabilizerParams.AUTO_RANGE),
                'type': 'bool',
                'tooltip': r'''
                    Automatically adjust the plots range based on data.\n
                    Keep on for best performance.\n
                    DON'T USE THE PlotWidget BUILT-IN AUTO RANGE IT WILL CAUSE LAGS!
                    (Button with capital A in the plots)''',
                'value': True,
            },
            {
                'name': str(FocusStabilizerParams.PREVIEW_IMAGE),
                'type': 'bool',
                'value': True,
            },
            {
                'name': str(FocusStabilizerParams.N_FRAMES),
                'type': 'int',
                'value': 1,
                'limits': [1, 100],
            },
            {
                'name': str(FocusStabilizerParams.FRAMES_STATS),
                'type': 'list',
                'value': 'mean',
                'limits': FRAME_STATS,
            },
            {
                'name': str(FocusStabilizerParams.LINE_WIDTH),
                'type': 'int',
                'value': 1,
                'limits': [1, 100],
                'step': 1,
            },
            {'name': str(FocusStabilizerParams.EXTERNAL_PREVIEW), 'type': 'action'},
            {'name': str(FocusStabilizerParams.SAVE), 'type': 'action'},
            {'name': str(FocusStabilizerParams.LOAD), 'type': 'action'},
            {
                'name': str(FocusStabilizerParams.FOCUS_TRACKING),
                'type': 'group',
                'children': [],
            },
            {
                'name': str(FocusStabilizerParams.FOCUS_PARAMETER),
                'type': 'float',
                'value': 0.0,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
            },
            {
                'name': str(FocusStabilizerParams.XY_STABILIZATION),
                'type': 'bool',
                'value': False,
            },
            {
                'name': str(FocusStabilizerParams.Z_STABILIZATION),
                'type': 'bool',
                'value': False,
            },
            {
                'name': str(FocusStabilizerParams.STABILIZATION_METHOD),
                'type': 'list',
                'value': StabilizationMethods.REFLECTION.value,
                'limits': [method.value for method in StabilizationMethods],
            },
            {
                'name': str(FocusStabilizerParams.METHOD_PARAMS),
                'type': 'group',
                'expanded': False,
                'children': [],
            },
            {
                'name': str(FocusStabilizerParams.CALIBRATION),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(FocusStabilizerParams.USE_CAL),
                        'type': 'bool',
                        'value': False,
                    },
                    {
                        'name': str(FocusStabilizerParams.X_CALIBRATION),
                        'type': 'float',
                        'value': 1.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.Y_CALIBRATION),
                        'type': 'float',
                        'value': 1.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.Z_CALIBRATION),
                        'type': 'float',
                        'value': 1.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.ADJUST_SET_POINT),
                        'type': 'bool',
                        'value': False,
                    },
                ],
            },
            {
                'name': str(FocusStabilizerParams.XY_OUTLIER_REJECTION),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(FocusStabilizerParams.XY_OUTLIER_REJECTION_METHOD),
                        'type': 'list',
                        'limits': [method.value for method in RejectionMethod],
                        'value': RejectionMethod.NONE.value,
                    },
                    {
                        'name': str(
                            FocusStabilizerParams.XY_OUTLIER_REJECTION_THRESHOLD
                        ),
                        'type': 'float',
                        'value': 2.0,
                    },
                    {
                        'name': str(
                            FocusStabilizerParams.XY_OUTLIER_REJECTION_MIN_POINTS
                        ),
                        'type': 'int',
                        'value': 4,
                        'limits': [3, 25],
                        'step': 1,
                    },
                ],
            },
            {
                'name': str(FocusStabilizerParams.X),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(FocusStabilizerParams.X_INVERTED),
                        'type': 'bool',
                        'value': False,
                    },
                    {
                        'name': str(FocusStabilizerParams.X_K_P),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.X_K_I),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.X_K_D),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                ],
            },
            {
                'name': str(FocusStabilizerParams.Y),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(FocusStabilizerParams.Y_INVERTED),
                        'type': 'bool',
                        'value': False,
                    },
                    {
                        'name': str(FocusStabilizerParams.Y_K_P),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.Y_K_I),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.Y_K_D),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                ],
            },
            {
                'name': str(FocusStabilizerParams.Z),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(FocusStabilizerParams.Z_INVERTED),
                        'type': 'bool',
                        'value': False,
                    },
                    {
                        'name': str(FocusStabilizerParams.Z_K_P),
                        'type': 'float',
                        'value': 6.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.Z_K_I),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                    {
                        'name': str(FocusStabilizerParams.Z_K_D),
                        'type': 'float',
                        'value': 0.0,
                        'step': 0.01,
                        'dec': False,
                        'decimals': 6,
                    },
                ],
            },
            {
                'name': str(FocusStabilizerParams.FT_TAU),
                'type': 'float',
                'value': 0.0,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
            },
            {
                'name': str(FocusStabilizerParams.FT_ERROR_TH),
                'type': 'float',
                'value': 0.001,
                'step': 0.001,
                'dec': False,
                'decimals': 6,
            },
            {'name': str(FocusStabilizerParams.PEAK_ACQUIRE), 'type': 'action'},
            {'name': str(FocusStabilizerParams.PEAK_STOP), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(FocusStabilizerParams.EXTERNAL_PREVIEW).sigActivated.connect(
            lambda: self.actionActivated.emit(
                FocusStabilizerParams.EXTERNAL_PREVIEW.value
            )
        )
        self.get_param(FocusStabilizerParams.SAVE).sigActivated.connect(
            self.export_config
        )
        self.get_param(FocusStabilizerParams.LOAD).sigActivated.connect(
            self.import_config
        )

        self.get_param(FocusStabilizerParams.PEAK_ACQUIRE).sigActivated.connect(
            self.start_IR
        )
        self.get_param(FocusStabilizerParams.PEAK_STOP).sigActivated.connect(
            self.stop_IR
        )

        # Build method-specific params on first UI creation.
        self.rebuild_method_params()

    def rebuild_method_params(self):
        group = self.get_param(FocusStabilizerParams.METHOD_PARAMS)
        if group is None:
            return

        self.param_tree.blockSignals(True)
        try:
            for child in list(group.children()):
                group.removeChild(child)

            for (
                child_def
            ) in FocusStabilizer.instance().get_current_strategy_param_defs():
                group.addChild(child_def)
        finally:
            self.param_tree.blockSignals(False)

    def apply_method_params(self, values: dict):
        if not isinstance(values, dict):
            return

        FocusStabilizer.instance().set_current_strategy_params(values)

        current = FocusStabilizer.instance().get_current_strategy_param_values()
        for name, value in current.items():
            p = self.get_param(f'{FocusStabilizerParams.METHOD_PARAMS.value}.{name}')
            if p is not None:
                p.setValue(value)

    def change(self, param: Parameter, changes: list):
        '''
        Handle parameter changes as needed.

        This method handles the changes made to the parameters in the parameter
        tree.

        Parameters
        ----------
        param : Parameter
            The parameter that triggered the change.
        changes : list
            List of changes.

        Returns
        -------
        None
        '''
        # Handle parameter changes as needed
        for p, _, data in changes:
            path = self.param_tree.childPath(p)
            if not path:
                continue

            if path[0] == FocusStabilizerParams.METHOD_PARAMS.value:
                if len(path) > 1:
                    FocusStabilizer.instance().set_current_strategy_param(path[1], data)
                continue

            try:
                fsParam = FocusStabilizerParams('.'.join(path))
            except ValueError:
                continue

            if fsParam == FocusStabilizerParams.Z_STABILIZATION:
                FocusStabilizer.instance().toggleFocusStabilization(Axis.Z, data)
            if fsParam == FocusStabilizerParams.XY_STABILIZATION:
                FocusStabilizer.instance().toggleFocusStabilization(Axis.X, data)
            if fsParam == FocusStabilizerParams.X_K_P:
                FocusStabilizer.instance().setP(data, Axis.X)
            if fsParam == FocusStabilizerParams.X_K_I:
                FocusStabilizer.instance().setI(data, Axis.X)
            if fsParam == FocusStabilizerParams.X_K_D:
                FocusStabilizer.instance().setD(data, Axis.X)
            if fsParam == FocusStabilizerParams.Y_K_P:
                FocusStabilizer.instance().setP(data, Axis.Y)
            if fsParam == FocusStabilizerParams.Y_K_I:
                FocusStabilizer.instance().setI(data, Axis.Y)
            if fsParam == FocusStabilizerParams.Y_K_D:
                FocusStabilizer.instance().setD(data, Axis.Y)
            if fsParam == FocusStabilizerParams.Z_K_P:
                FocusStabilizer.instance().setP(data, Axis.Z)
            if fsParam == FocusStabilizerParams.Z_K_I:
                FocusStabilizer.instance().setI(data, Axis.Z)
            if fsParam == FocusStabilizerParams.Z_K_D:
                FocusStabilizer.instance().setD(data, Axis.Z)
            if fsParam == FocusStabilizerParams.FT_TAU:
                FocusStabilizer.instance().setTau(data)
            if fsParam == FocusStabilizerParams.FT_ERROR_TH:
                FocusStabilizer.instance().setErrorTh(data)

            if fsParam == FocusStabilizerParams.N_FRAMES:
                FocusStabilizer.instance().set_n_frames(data)
            if fsParam == FocusStabilizerParams.FRAMES_STATS:
                FocusStabilizer.instance().set_frames_stats_method(data)

            if fsParam == FocusStabilizerParams.LINE_WIDTH:
                FocusStabilizer.instance().roi_manager.set_linewidth(data)

            if fsParam == FocusStabilizerParams.STABILIZATION_METHOD:
                method = StabilizationMethods(data)
                FocusStabilizer.instance().setMethod(method)
                self.rebuild_method_params()
                self.methodChanged.emit(method)

            if fsParam == FocusStabilizerParams.XY_OUTLIER_REJECTION_METHOD:
                FocusStabilizer.instance().setRejectionMethod(RejectionMethod(data))
            if fsParam == FocusStabilizerParams.XY_OUTLIER_REJECTION_THRESHOLD:
                FocusStabilizer.instance().setOutlierThreshold(data)
            if fsParam == FocusStabilizerParams.XY_OUTLIER_REJECTION_MIN_POINTS:
                FocusStabilizer.instance().setOutlierMinPoints(data)

            if fsParam == FocusStabilizerParams.X_INVERTED:
                FocusStabilizer.instance().setInverted(data, Axis.X)
            if fsParam == FocusStabilizerParams.Y_INVERTED:
                FocusStabilizer.instance().setInverted(data, Axis.Y)
            if fsParam == FocusStabilizerParams.Z_INVERTED:
                FocusStabilizer.instance().setInverted(data, Axis.Z)
            if fsParam == FocusStabilizerParams.USE_CAL:
                FocusStabilizer.instance().setUseCal(data)
            if fsParam == FocusStabilizerParams.ADJUST_SET_POINT:
                FocusStabilizer.instance().setAdjustSetPoint(data)
            if fsParam == FocusStabilizerParams.FOCUS_PARAMETER:
                FocusStabilizer.instance().setParameter(data)

            if fsParam == FocusStabilizerParams.X_CALIBRATION:
                FocusStabilizer.instance().setCalCoeff(data, Axis.X)
            if fsParam == FocusStabilizerParams.Y_CALIBRATION:
                FocusStabilizer.instance().setCalCoeff(data, Axis.Y)
            if fsParam == FocusStabilizerParams.Z_CALIBRATION:
                FocusStabilizer.instance().setCalCoeff(data, Axis.Z)

            if fsParam == FocusStabilizerParams.PREVIEW_IMAGE:
                FocusStabilizer.instance().setPreview(data)

    def update_focus_parameter(self, value: float):
        '''
        Update the focus parameter in the parameter tree.

        Parameters
        ----------
        value : float
            The new focus parameter value.

        Returns
        -------
        None
        '''
        if isinstance(value, (float, int)):
            self.set_param_value(FocusStabilizerParams.FOCUS_PARAMETER, value)
        else:
            self.set_param_value(FocusStabilizerParams.FOCUS_PARAMETER, np.nan)

    def update_calibration_coefficients(self, coeffs: np.ndarray):
        '''
        Update the calibration coefficients in the parameter tree.

        Parameters
        ----------
        coeffs : np.ndarray
            The new calibration coefficients.

        Returns
        -------
        None
        '''
        self.set_param_value(FocusStabilizerParams.X_CALIBRATION, coeffs[0])
        self.set_param_value(FocusStabilizerParams.Y_CALIBRATION, coeffs[1])
        self.set_param_value(FocusStabilizerParams.Z_CALIBRATION, coeffs[2])

    def toggleFocusStabilization(self, axis: Axis):
        '''Toggles the focus stabilization.'''
        param = (
            FocusStabilizerParams.Z_STABILIZATION
            if axis == Axis.Z
            else FocusStabilizerParams.XY_STABILIZATION
        )
        self.set_param_value(
            param,
            not self.get_param_value(param),
        )

    def start_IR(self):
        '''Starts the IR peak position acquisition and
        creates a file in the current directory.
        '''
        if not FocusStabilizer.instance().is_logging():
            filename = None
            if filename is None:
                filename, _ = getSaveFileName(
                    self, 'Save Focus Peak Data', filter='CSV Files (*.csv)'
                )

                if len(filename) > 0:
                    FocusStabilizer.instance().start_logging(filename)

    def stop_IR(self):
        '''Stops the IR peak position acquisition and closes the file.'''
        if FocusStabilizer.instance().is_logging():
            FocusStabilizer.instance().stop_logging()

    def get_config(self) -> dict:
        '''
        Get the current configuration as a dictionary.

        Returns
        -------
        dict
            The current configuration dictionary.
        '''
        return {
            'ROIs': FocusStabilizer.instance().roi_manager.get_config(),
            'Stabilizer': FocusStabilizer.instance().get_config(),
        }

    def load_config(self, config: dict):
        '''
        Load the configuration from a dictionary.

        Parameters
        ----------
        config : dict
            The configuration dictionary.

        Returns
        -------
        None
        '''
        if 'ROIs' in config:
            FocusStabilizer.instance().roi_manager.load_config(config['ROIs'])

            self.set_param_value(
                FocusStabilizerParams.LINE_WIDTH,
                FocusStabilizer.instance().roi_manager.get_linewidth(),
            )

            self.setRoiActivated.emit()

        if 'Stabilizer' in config:
            stabilizer: dict = config['Stabilizer']

            Kp = stabilizer.get('Kp', (1.0,) * 3)
            Ki = stabilizer.get('Ki', (0.0,) * 3)
            Kd = stabilizer.get('Kd', (0.0,) * 3)
            tau = stabilizer.get('tau', 0)
            error_th = stabilizer.get('error_th', 0.0)
            cal_coeff = stabilizer.get('cal_coeff', (1.0,) * 3)
            inverted = stabilizer.get('inverted', (False,) * 3)

            cal_params = {
                Axis.X: FocusStabilizerParams.X_CALIBRATION,
                Axis.Y: FocusStabilizerParams.Y_CALIBRATION,
                Axis.Z: FocusStabilizerParams.Z_CALIBRATION,
            }

            for axis in Axis:
                self.set_param_value(
                    FocusStabilizerParams(f'{axis.name}.K_P'),
                    float(Kp[axis.axis_index()]),
                )
                self.set_param_value(
                    FocusStabilizerParams(f'{axis.name}.K_I'),
                    float(Ki[axis.axis_index()]),
                )
                self.set_param_value(
                    FocusStabilizerParams(f'{axis.name}.K_D'),
                    float(Kd[axis.axis_index()]),
                )
                self.set_param_value(
                    cal_params[axis],
                    float(cal_coeff[axis.axis_index()]),
                )
                self.set_param_value(
                    FocusStabilizerParams(f'{axis.name}.Inverted'),
                    bool(inverted[axis.axis_index()]),
                )

            method = stabilizer.get('method', StabilizationMethods.REFLECTION.value)

            rejection_method = stabilizer.get(
                'rejection_method', RejectionMethod.NONE.value
            )
            rejection_threshold = stabilizer.get('rejection_threshold', 2.0)
            rejection_min_points = stabilizer.get('rejection_min_points', 4)

            self.set_param_value(
                FocusStabilizerParams.USE_CAL, bool(stabilizer.get('use_cal'))
            )
            self.set_param_value(
                FocusStabilizerParams.ADJUST_SET_POINT,
                bool(stabilizer.get('adjust_set_point')),
            )
            self.set_param_value(FocusStabilizerParams.FT_TAU, float(tau))
            self.set_param_value(FocusStabilizerParams.FT_ERROR_TH, float(error_th))
            self.set_param_value(FocusStabilizerParams.STABILIZATION_METHOD, method)
            # Ensure params group exists even when method stays unchanged.
            self.rebuild_method_params()
            strategy_params = stabilizer.get('strategy_params', {})
            self.apply_method_params(strategy_params)
            self.set_param_value(
                FocusStabilizerParams.XY_OUTLIER_REJECTION_METHOD,
                rejection_method,
            )
            self.set_param_value(
                FocusStabilizerParams.XY_OUTLIER_REJECTION_THRESHOLD,
                float(rejection_threshold),
            )
            self.set_param_value(
                FocusStabilizerParams.XY_OUTLIER_REJECTION_MIN_POINTS,
                int(rejection_min_points),
            )

            n_frames = stabilizer.get('n_frames', 1)
            frames_stats_method = stabilizer.get('frames_stats_method', 'mean')
            self.set_param_value(FocusStabilizerParams.N_FRAMES, int(n_frames))
            self.set_param_value(
                FocusStabilizerParams.FRAMES_STATS, frames_stats_method
            )

    def export_config(self):
        '''
        Save the current configuration to a JSON file.
        '''
        filename, _ = getSaveFileName(
            self, 'Save Focus Stabilizer Config', filter='JSON Files (*.json)'
        )
        if filename:
            config = self.get_config()
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)

    def import_config(self):
        '''
        Load the configuration from a JSON file.
        '''
        filename, _ = getOpenFileName(
            self, 'Load Focus Stabilizer Config', filter='JSON Files (*.json)'
        )
        if filename:
            with open(filename) as f:
                config = json.load(f)
                self.load_config(config)
