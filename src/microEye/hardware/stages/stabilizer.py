import time
import traceback
from enum import Enum
from queue import LifoQueue
from threading import Event
from typing import Optional

import numpy as np
from pyqtgraph.parametertree import Parameter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from microEye.analysis.tools.kymograms import get_kymogram_row
from microEye.qt import (
    QApplication,
    QDateTime,
    QtCore,
    QtWidgets,
    Signal,
    getSaveFileName,
)
from microEye.utils import Tree
from microEye.utils.gui_helper import GaussianOffSet
from microEye.utils.thread_worker import QThreadWorker


class FocusStabilizerParams(Enum):
    '''
    Enum class defining FocusStabilizer parameters.
    '''

    ROI = 'Line ROI'
    X1 = 'X1'
    Y1 = 'Y1'
    X2 = 'X2'
    Y2 = 'Y2'
    LINE_WIDTH = 'Line Width'
    LENGTH = 'Length'
    ANGLE = 'Angle [deg]'
    SET_ROI = 'Set ROI'
    SAVE = 'Save'
    LOAD = 'Load'

    FOCUS_TRACKING = 'Focus Stabilization'
    FOCUS_PEAK = 'Focus Peak [pixel]'
    PIXEL_CAL = 'Calibration [pixel/nm]'
    USE_CAL = 'Use Calibration'
    INVERTED = 'Inverted'
    TRACKING_ENABLED = 'Stabilization Enabled'
    FT_P = 'P'
    FT_I = 'I'
    FT_D = 'D'
    FT_TAU = 'Tau'
    FT_ERROR_TH = 'Error Threshold'
    PEAK_ACQUIRE = 'Start Peak Acquisition'
    PEAK_STOP = 'Stop Peak Acquisition'

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


class FocusStabilizer(QtCore.QObject):
    # Class attribute to hold the single instance
    _instance = None

    TIME_POINTS = 500

    updateViewBox = Signal(object)
    updatePlots = Signal(object)
    moveStage = Signal(bool, int)
    peakPositionChanged = Signal(float)
    pixelCalChanged = Signal(float)

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
        self.peak_time = np.array(list(range(self.TIME_POINTS)))
        self.peak_positions = np.ones((self.TIME_POINTS,))
        self.error_buffer = np.zeros((20,))
        self.error_integral = 0
        self.fit_params = None

        self.X = np.array([25.5, 25.5])
        self.Y = np.array([25.5, 281.5])
        self.line_width = 1

        self.__pixel_cal_coeff = 0.01
        self.__use_cal = False
        self.__inverted = False
        self.__peak_position = 64
        self.__stabilization = False
        self.__P = 6.45
        self.__I = 0.0
        self.__D = 0.0
        self.__tau = 0.0
        self.__err_th = 0.001

        self._exec_time = 0

        self.file = None
        self.num_frames_saved = 0

        self.exit_event = Event()

        # Stop worker when app is closing
        app = QApplication.instance()
        app.aboutToQuit.connect(FocusStabilizer.instance().exit_event.set)

    @property
    def buffer(self):
        return self.__buffer

    def pixelCalCoeff(self):
        '''
        Gets the current pixel calibration coefficient.

        Returns
        -------
        float
            The current pixel calibration coefficient.
        '''
        return self.__pixel_cal_coeff

    def setPixelCalCoeff(self, value: float):
        '''
        Sets the pixel calibration coefficient.

        Parameters
        ----------
        value : float
            The new pixel calibration coefficient.
        '''
        self.__pixel_cal_coeff = value
        self.pixelCalChanged.emit(value)

    def useCal(self):
        '''
        Check if pixel calibration coefficient is used.

        Returns
        -------
        bool
            True if pixel calibration coefficient is used, False otherwise.
        '''
        return self.__use_cal

    def setUseCal(self, use_cal: bool):
        '''
        Set the flag indicating whether to use pixel calibration coefficient.

        Parameters
        ----------
        use_cal : bool
            The flag value.
        '''
        self.__use_cal = use_cal

    def isInverted(self):
        '''
        Check if the direction is inverted.

        Returns
        -------
        bool
            True if the direction is inverted, False otherwise.
        '''
        return self.__inverted

    def setInverted(self, inverted: bool):
        '''
        Set the flag indicating whether the direction is inverted.

        Parameters
        ----------
        inverted : bool
            The flag value.
        '''
        self.__inverted = inverted

    def __len__(self):
        return len(self.__buffer.queue)

    def put(self, value: np.ndarray):
        self.__buffer.put(value)

    def isEmpty(self):
        return self.__buffer.empty()

    def getImage(self) -> np.ndarray:
        '''Gets image from frames Queue in LIFO manner.

        Returns
        -------
        np.ndarray
            The image last in buffer.
        '''
        res = None
        if not self.__buffer.empty():
            res = self.__buffer.get()
            self.__buffer.queue.clear()
        return res

    def isImage(self, data: np.ndarray):
        if data.ndim > 2:
            raise ValueError('Data should be one or two dimensional array.')

        if data.ndim == 1:
            return False
        elif data.ndim == 2:
            return True

    def setP(self, P: float):
        '''
        Set the P value.

        Parameters
        ----------
        P : float
            The P value.
        '''
        self.__P = P

    def setI(self, _I: float):
        '''
        Set the I value.

        Parameters
        ----------
        _I : float
            The I value.
        '''
        self.__I = _I

    def setD(self, D: float):
        '''
        Set the D value.

        Parameters
        ----------
        D : float
            The D value.
        '''
        self.__D = D

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

    def getPID(self):
        '''
        Get the current PID values.

        Returns
        -------
        tuple of float
            The current PID values as a tuple (P, I, D, tau, error_th).
        '''
        return (self.__P, self.__I, self.__D, self.__tau, self.__err_th)

    def getPeakPosition(self):
        '''
        Get the current focus peak position.

        Returns
        -------
        float
            The current focus peak.
        '''
        return self.__peak_position

    def setPeakPosition(self, value: float, incerement: bool = False):
        '''
        Set the focus peak position.

        Parameters
        ----------
        value : float
            The new focus peak value.
        incerement : bool, optional
            If True, increment the current focus peak value by the given value,
            otherwise set the focus peak to the given value.
        '''
        if incerement:
            self.__peak_position += value
        else:
            self.__peak_position = value
        self.peakPositionChanged.emit(self.__peak_position)

    def isFocusStabilized(self):
        '''
        Check if automatic Focus Stabilization is enabled.

        Returns
        -------
        bool
            True if automatic Focus Stabilization is enabled, False otherwise.
        '''
        return self.__stabilization

    def toggleFocusStabilization(self, value: bool = None):
        '''
        Toggles automatic Focus Stabilization.

        If `value` is not provided or set to `None`, the method toggles the
        autofocus tracking option based on its current state. If `value` is a
        Boolean value, the method sets the option to that state.

        Parameters
        ----------
        value : bool, optional
            The new state of the automatic Focus Stabilization. Default is `None`.

        Returns
        -------
        bool
            The current state of the automatic Focus Stabilization.
        '''
        if value is None:
            self.__stabilization = not self.__stabilization
        else:
            self.__stabilization = value

        return self.__stabilization

    def startWorker(self):
        self.worker = QThreadWorker(
            FocusStabilizer.instance().worker_function
        )
        # Execute
        QtCore.QThreadPool.globalInstance().start(self.worker)

    def worker_function(self, **kwargs):
        '''A worker function running in the threadpool.

        Handles the IR peak fitting and piezo autofocus tracking.
        '''
        counter = 0
        self._exec_time = 0
        now = QDateTime.currentDateTime()
        QtCore.QThread.msleep(100)
        while not self.exit_event.isSet():
            try:
                # proceed only if the buffer is not empty
                if not self.isEmpty():
                    self._exec_time = now.msecsTo(QDateTime.currentDateTime())
                    now = QDateTime.currentDateTime()

                    image = self.getImage()

                    if self.isImage(image):
                        self.updateViewBox.emit(image.copy())
                        line_roi = get_kymogram_row(
                            image, self.X, self.Y, self.line_width
                        )
                    self.peak_fit(line_roi.copy())
                    if self.file is not None:
                        with open(self.file, 'ab') as file:
                            np.savetxt(
                                file,
                                np.array((self._exec_time, self.fit_params[1])).reshape(
                                    (1, 2)
                                ),
                                delimiter=';',
                            )
                        self.num_frames_saved = 1 + self.num_frames_saved
                    counter = counter + 1
                    self.updatePlots.emit(line_roi.copy())
                QtCore.QThread.usleep(100)  # sleep for 100us
            except Exception as e:
                traceback.print_exc()

        print(f'{FocusStabilizer.__name__} worker is terminating.')

    def peak_fit(self, data: np.ndarray):
        '''
        Finds the peak position through fitting and adjusts the piezostage accordingly.

        Parameters
        ----------
        data : np.ndarray
            The input data to fit.

        Raises
        ------
        Exception
            If an error occurs during the fitting process.
        '''
        try:
            # find IR peaks above a specific height
            peaks = find_peaks(data, height=1)
            nPeaks = len(peaks[0])  # number of peaks
            maxPeakIdx = np.argmax(peaks[1]['peak_heights'])  # highest peak
            x0 = 64 if nPeaks == 0 else peaks[0][maxPeakIdx]
            a0 = 1 if nPeaks == 0 else peaks[1]['peak_heights'][maxPeakIdx]

            # curve_fit to GaussianOffSet
            self.fit_params, _ = curve_fit(
                GaussianOffSet, np.arange(data.shape[0]), data, p0=[a0, x0, 1, 0]
            )
            self.peak_positions = np.roll(self.peak_positions, -1)
            self.peak_positions[-1] = self.fit_params[1]

            if self.isFocusStabilized():
                err = np.average(self.peak_positions[-1] - self.getPeakPosition())
                tau = self.__tau if self.__tau > 0 else (max(self._exec_time, 1) / 1000)
                self.error_integral += err * tau
                diff = (err - self.error_buffer[-1]) / tau
                self.error_buffer = np.roll(self.error_buffer, -1)
                self.error_buffer[-1] = err
                step = int(
                    (err * self.__P)
                    + (self.error_integral * self.__I)
                    + (diff * self.__D)
                )
                if abs(err) > self.__err_th:
                    direction = (step <= 0) if not self.isInverted() else (step > 0)
                    self.moveStage.emit(direction, abs(step))
            else:
                self.setPeakPosition(self.peak_positions[-1])
                self.error_buffer = np.zeros((20,))
                self.error_integral = 0

        except Exception as e:
            pass
            # print('Failed Gauss. fit: ' + str(e))


class FocusStabilizerView(Tree):
    PARAMS = FocusStabilizerParams
    setRoiActivated = Signal()
    saveActivated = Signal()
    loadActivated = Signal()

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

        FocusStabilizer.instance().peakPositionChanged.connect(
            lambda value: self.set_param_value(FocusStabilizerParams.FOCUS_PEAK, value)
        )
        FocusStabilizer.instance().pixelCalChanged.connect(
            lambda value: self.set_param_value(FocusStabilizerParams.PIXEL_CAL, value)
        )

        self.setMinimumWidth(250)

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `PzFocView` class.
        '''
        params = [
            {'name': str(FocusStabilizerParams.ROI), 'type': 'group', 'children': []},
            {
                'name': str(FocusStabilizerParams.X1),
                'type': 'float',
                'value': 25.5,
                'step': 1,
            },
            {
                'name': str(FocusStabilizerParams.Y1),
                'type': 'float',
                'value': 25.5,
                'step': 1,
            },
            {
                'name': str(FocusStabilizerParams.X2),
                'type': 'float',
                'value': 25.5,
                'step': 1,
                'readonly': True,
            },
            {
                'name': str(FocusStabilizerParams.Y2),
                'type': 'float',
                'value': 281.5,
                'step': 1,
                'readonly': True,
            },
            {
                'name': str(FocusStabilizerParams.LINE_WIDTH),
                'type': 'int',
                'value': 1,
                'limits': [1, 100],
                'step': 1,
            },
            {
                'name': str(FocusStabilizerParams.LENGTH),
                'type': 'int',
                'value': 256,
                'limits': [64, 4096],
                'step': 1,
            },
            {
                'name': str(FocusStabilizerParams.ANGLE),
                'type': 'float',
                'value': 0,
                'limits': [0, 360],
                'step': 1,
            },
            {'name': str(FocusStabilizerParams.SET_ROI), 'type': 'action'},
            {'name': str(FocusStabilizerParams.SAVE), 'type': 'action'},
            {'name': str(FocusStabilizerParams.LOAD), 'type': 'action'},
            {
                'name': str(FocusStabilizerParams.FOCUS_TRACKING),
                'type': 'group',
                'children': [],
            },
            {
                'name': str(FocusStabilizerParams.FOCUS_PEAK),
                'type': 'float',
                'value': 0.0,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
            },
            {
                'name': str(FocusStabilizerParams.PIXEL_CAL),
                'type': 'float',
                'value': 0.01,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
            },
            {
                'name': str(FocusStabilizerParams.USE_CAL),
                'type': 'bool',
                'value': False,
            },
            {
                'name': str(FocusStabilizerParams.INVERTED),
                'type': 'bool',
                'value': False,
            },
            {
                'name': str(FocusStabilizerParams.TRACKING_ENABLED),
                'type': 'bool',
                'value': False,
            },
            {
                'name': str(FocusStabilizerParams.FT_P),
                'type': 'float',
                'value': 6.450,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
            },
            {
                'name': str(FocusStabilizerParams.FT_I),
                'type': 'float',
                'value': 0.0,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
            },
            {
                'name': str(FocusStabilizerParams.FT_D),
                'type': 'float',
                'value': 0.0,
                'step': 0.01,
                'dec': False,
                'decimals': 6,
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

        self.get_param(FocusStabilizerParams.SET_ROI).sigActivated.connect(
            lambda: self.setRoiActivated.emit()
        )
        self.get_param(FocusStabilizerParams.SAVE).sigActivated.connect(
            self.export_json
        )
        self.get_param(FocusStabilizerParams.LOAD).sigActivated.connect(self.load_json)

        self.get_param(FocusStabilizerParams.PEAK_ACQUIRE).sigActivated.connect(
            self.start_IR
        )
        self.get_param(FocusStabilizerParams.PEAK_STOP).sigActivated.connect(
            self.stop_IR
        )

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

            if path == FocusStabilizerParams.get_path(
                FocusStabilizerParams.TRACKING_ENABLED
            ):
                FocusStabilizer.instance().toggleFocusStabilization(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.FT_P):
                FocusStabilizer.instance().setP(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.FT_I):
                FocusStabilizer.instance().setI(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.FT_D):
                FocusStabilizer.instance().setD(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.FT_TAU):
                FocusStabilizer.instance().setTau(data)
            if path == FocusStabilizerParams.get_path(
                FocusStabilizerParams.FT_ERROR_TH
            ):
                FocusStabilizer.instance().setErrorTh(data)

            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.X1):
                FocusStabilizer.instance().X[0] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.X2):
                FocusStabilizer.instance().X[1] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.Y1):
                FocusStabilizer.instance().Y[0] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.Y2):
                FocusStabilizer.instance().Y[1] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.LINE_WIDTH):
                FocusStabilizer.instance().line_width = data

            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.INVERTED):
                FocusStabilizer.instance().setInverted(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.USE_CAL):
                FocusStabilizer.instance().setUseCal(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.FOCUS_PEAK):
                FocusStabilizer.instance().setPeakPosition(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.PIXEL_CAL):
                FocusStabilizer.instance().setPixelCalCoeff(data)

    def start_IR(self):
        '''Starts the IR peak position acquisition and
        creates a file in the current directory.
        '''
        if FocusStabilizer.instance().file is None:
            filename = None
            if filename is None:
                filename, _ = getSaveFileName(
                    self, 'Save Focus Peak Data', filter='CSV Files (*.csv)'
                )

                if len(filename) > 0:
                    FocusStabilizer.instance().file = filename

    def stop_IR(self):
        '''Stops the IR peak position acquisition and closes the file.'''
        if FocusStabilizer.instance().file is not None:
            FocusStabilizer.instance().file = None
            FocusStabilizer.instance().num_frames_saved = 0
