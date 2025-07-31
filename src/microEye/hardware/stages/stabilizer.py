import time
import traceback
import warnings
from enum import Enum
from queue import LifoQueue
from threading import Event
from typing import Optional

import cv2
import numba as nb
import numpy as np
from pyqtgraph.parametertree import Parameter
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

from microEye.analysis.filters import FourierFilter
from microEye.analysis.fitting import pyfit3Dcspline
from microEye.analysis.fitting.fit import CV_BlobDetector
from microEye.analysis.fitting.processing import nn_trajectories
from microEye.analysis.tools.kymograms import get_kymogram_row
from microEye.qt import (
    QApplication,
    QDateTime,
    QtCore,
    QtWidgets,
    Signal,
    getSaveFileName,
)
from microEye.utils.gui_helper import GaussianOffSet
from microEye.utils.parameter_tree import Tree
from microEye.utils.thread_worker import QThreadWorker

warnings.filterwarnings('ignore', category=OptimizeWarning)


@nb.njit(parallel=True)
def fast_nn_assignment(currentFrame, nextFrame, minDistance=0.0, maxDistance=30.0):
    N = currentFrame.shape[0]
    M = nextFrame.shape[0]
    assigned = np.full(N, -1, dtype=np.int64)
    distances = np.full(N, np.inf, dtype=np.float64)
    for i in nb.prange(N):
        min_dist = np.inf
        min_idx = -1
        for j in range(M):
            dist = np.sqrt(np.sum((currentFrame[i, :2] - nextFrame[j, :2]) ** 2))
            if minDistance <= dist <= maxDistance and dist < min_dist:
                min_dist = dist
                min_idx = j
        if min_idx != -1:
            assigned[i] = min_idx
            distances[i] = min_dist
    return assigned, distances


@nb.njit
def getXYshift(
    previous: np.ndarray, current: np.ndarray, minDistance=0.0, maxDistance=30.0
):
    '''
    Estimate X and Y shifts using fast Numba nearest neighbor assignment.

    Returns
    -------
    tuple of float
        The X and Y shifts.
    '''
    if (
        previous is None
        or current is None
        or previous.shape[0] == 0
        or current.shape[0] == 0
    ):
        return None

    assigned, distances = fast_nn_assignment(
        previous, current, minDistance, maxDistance
    )
    x_shifts = []
    y_shifts = []
    for i in range(previous.shape[0]):
        idx = assigned[i]
        if idx != -1:
            x_shifts.append(current[idx, 0] - previous[i, 0])
            y_shifts.append(current[idx, 1] - previous[i, 1])
    if len(x_shifts) == 0 or len(y_shifts) == 0:
        return None
    return np.mean(np.array(x_shifts)), np.mean(np.array(y_shifts))


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
    STABILIZATION_METHOD = 'Method'
    FOCUS_PARAMETER = 'Focus Parameter [pixel]'
    CAL_COEFF = 'Calibration [pixel/nm]'
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

    PREVIEW_IMAGE = 'Preview Image'

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


def shift_and_append(arr: np.ndarray, new_value: float):
    '''
    Rolls the array and appends a new value at the end.

    Parameters
    ----------
    arr : np.ndarray
        The input array to roll.
    new_value : float
        The new value to append at the end of the rolled array.

    Returns
    -------
    np.ndarray
        The rolled array with the new value appended.
    '''
    arr = np.roll(arr, -1)
    arr[-1] = new_value
    return arr


class FocusStabilizer(QtCore.QObject):
    # Class attribute to hold the single instance
    _instance = None

    TIME_POINTS = 500

    METHODS = ['reflection', 'beads', 'beads astigmatic']

    updateViewBox = Signal(object)
    updatePlots = Signal(object, str)
    moveStage = Signal(bool, int)
    parameterChanged = Signal(float)
    calCoeffChanged = Signal(float)
    focusStabilizationToggled = Signal(bool)

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
        self.time_points = np.array(list(range(self.TIME_POINTS)))
        self.parameter_buffer = np.ones((self.TIME_POINTS,))

        self.total_shift = np.array([0.0, 0.0])
        self.x_buffer = np.zeros((self.TIME_POINTS,))
        self.y_buffer = np.zeros((self.TIME_POINTS,))

        self.error_buffer = np.zeros((20,))
        self.error_integral = 0
        self.fit_params = None

        self.X_ROI = np.array([25.5, 25.5])
        self.Y_ROI = np.array([25.5, 281.5])
        self.line_width = 1

        self.__method = FocusStabilizer.METHODS[0]

        self.__cal_coeff = 0.01
        self.__use_cal = False
        self.__inverted = False
        self.__parameter = 64
        self.__stabilization = False
        self.__P = 6.45
        self.__I = 0.0
        self.__D = 0.0
        self.__tau = 0.0
        self.__err_th = 0.001

        self.__preview = True

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

    def method(self):
        '''
        Gets the current method for focus stabilization.

        Returns
        -------
        str
            The current method for focus stabilization.
        '''
        return self.__method

    def setMethod(self, method: str):
        '''
        Sets the method for focus stabilization.

        Parameters
        ----------
        method : str
            The new method for focus stabilization.
        '''
        if method in FocusStabilizer.METHODS:
            self.__method = method
            self.fit_params = None

            self.total_shift = np.array([0.0, 0.0])
            self.x_buffer = np.zeros((self.TIME_POINTS,))
            self.y_buffer = np.zeros((self.TIME_POINTS,))
        else:
            raise ValueError(
                f'Invalid method: {method}. Options: {FocusStabilizer.METHODS}'
            )

    def calCoeff(self):
        '''
        Gets the current calibration coefficient.

        Returns
        -------
        float
            The current calibration coefficient.
        '''
        return self.__cal_coeff

    def setCalCoeff(self, value: float):
        '''
        Sets the calibration coefficient.

        Parameters
        ----------
        value : float
            The new calibration coefficient.
        '''
        self.__cal_coeff = value
        self.calCoeffChanged.emit(value)

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

    def bufferSize(self):
        return len(self.__buffer.queue)

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

    def getParameter(self):
        '''
        Get the current focus parameter.

        Returns
        -------
        float
            The current focus parameter.
        '''
        return self.__parameter

    def setParameter(self, value: float, incerement: bool = False):
        '''
        Set the focus parameter.

        Parameters
        ----------
        value : float
            The new focus parameter value.
        incerement : bool, optional
            If True, increment the current focus parameter value by the given value,
            otherwise set the focus parameter to the given value.
        '''
        if incerement:
            self.__parameter += value
        else:
            self.__parameter = value
        self.parameterChanged.emit(self.__parameter)

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

        self.focusStabilizationToggled.emit(self.__stabilization)

        return self.__stabilization

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
        now = QDateTime.currentDateTime()
        QtCore.QThread.msleep(100)
        while not self.exit_event.isSet():
            try:
                # proceed only if the buffer is not empty
                if not self.isEmpty():
                    self._exec_time = now.msecsTo(QDateTime.currentDateTime())
                    now = QDateTime.currentDateTime()

                    method = self.method()

                    image = self.getImage()

                    if self.preview():
                        self.updateViewBox.emit(image.copy())

                    if self.isImage(image):
                        if method == 'reflection':
                            # get the kymogram row from the image
                            image = get_kymogram_row(
                                image, self.X_ROI, self.Y_ROI, self.line_width
                            )
                        elif method == 'beads':
                            # crop using the ROI two corners
                            image = image[
                                int(self.Y_ROI[0]) : int(self.Y_ROI[1]),
                                int(self.X_ROI[0]) : int(self.X_ROI[1]),
                            ]

                    # fit the parameter and stabilize
                    self.fit_parameter(image.copy(), method)
                    # stabilize the focus
                    self.stabilize()
                    # append parameter to the file if file is set
                    self.saveToFile()

                    counter = counter + 1

                    # update the plots with the current parameter
                    self.updatePlots.emit(
                        image.copy()
                        if method == 'reflection'
                        else np.c_[self.x_buffer, self.y_buffer],
                        method,
                    )
                QtCore.QThread.usleep(100)  # sleep for 100us
            except Exception as e:
                traceback.print_exc()

        print(f'{FocusStabilizer.__name__} worker is terminating.')

    def fit_parameter(self, data: np.ndarray, method: str):
        """
        Fit the data to the specified method and update the parameter buffer.

        Parameters
        ----------
        data : np.ndarray
            The input data to fit.
        method : str
            The method to use for fitting. For options check `FocusStabilizer.METHODS`.

            - 'reflection': Fit a linear ROI to a GaussianOffSet function.
            - 'beads': Fit beads using a 2D Gaussian function and extract average sigma.
        """
        if method not in FocusStabilizer.METHODS:
            raise ValueError(
                f'Invalid method: {method}. Options: {FocusStabilizer.METHODS}'
            )

        if method == 'reflection':
            self.peak_fit(data)

        elif method in ['beads', 'beads astigmatic']:
            self.beads_fit(data, astigmatic=method == 'beads astigmatic')

    def stabilize(self):
        try:
            if self.isFocusStabilized():
                err = np.average(self.parameter_buffer[-1] - self.getParameter())
                tau = self.__tau if self.__tau > 0 else (max(self._exec_time, 1) / 1000)
                self.error_integral += err * tau
                diff = (err - self.error_buffer[-1]) / tau

                self.error_buffer = shift_and_append(self.error_buffer, err)

                step = int(
                    (err * self.__P)
                    + (self.error_integral * self.__I)
                    + (diff * self.__D)
                )
                if abs(err) > self.__err_th:
                    direction = (step <= 0) if not self.isInverted() else (step > 0)
                    self.moveStage.emit(direction, abs(step))
            else:
                self.setParameter(self.parameter_buffer[-1])
                self.error_buffer = np.zeros((20,))
                self.error_integral = 0
        except Exception as e:
            pass

    def saveToFile(self):
        '''
        Save the current execution time and fit parameters to a file.
        '''
        if self.file is not None:
            with open(self.file, 'ab') as file:
                if self.num_frames_saved == 0:
                    header = ';'.join(
                        [
                            'Execution Time [ms]',
                            'Focus Parameter [pixel]',
                            'X Shift [pixel]',
                            'Y Shift [pixel]',
                        ]
                    )
                    # Write header if this is the first frame
                    file.write(header.encode('utf-8') + b'\n')
                np.savetxt(
                    file,
                    np.array(
                        (
                            self._exec_time,
                            self.parameter_buffer[-1],
                            self.x_buffer[-1],
                            self.y_buffer[-1],
                        )
                    ).reshape((1, 4)),
                    delimiter=';',
                )
            self.num_frames_saved = 1 + self.num_frames_saved

    def peak_fit(self, data: np.ndarray):
        '''
        Fit the data to a GaussianOffSet function and update the parameter buffer.

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
            self.parameter_buffer = shift_and_append(
                self.parameter_buffer, self.fit_params[1]
            )
        except Exception as e:
            pass
            # print('Failed Gauss. fit: ' + str(e))

    def beads_fit(self, data: np.ndarray, sigma: float = 1.0, astigmatic: bool = False):
        '''
        Fit the data to a 2D Gaussian function and update the parameter buffer.

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
            # apply Fourier filter to the data
            filtered = FourierFilter().run(data)

            # Threshold the image
            _, th_img = cv2.threshold(
                filtered,
                np.quantile(filtered, 1 - 1e-4) * 0.4,
                255,
                cv2.THRESH_BINARY,
            )

            # Detect blobs
            points, im_with_keypoints = CV_BlobDetector().find_peaks_preview(
                th_img, None
            )

            if len(points) < 1:
                self.fit_params = None
                return

            # Get the ROI list for fitting
            roi_sz = 21
            rois, coords = pyfit3Dcspline.get_roi_list(data, points, roi_sz)

            Params, CRLBs, LogLikelihood = pyfit3Dcspline.CPUmleFit_LM(
                rois, 4 if astigmatic else 2, np.array([sigma]), None, 0
            )

            if Params is None and len(Params) < 1:
                self.fit_params = None
                return
            elif Params.ndim == 1:
                Params = Params[np.newaxis, :]

            mean_params = np.mean(Params, axis=0)
            self.parameter_buffer = shift_and_append(
                self.parameter_buffer,
                (
                    mean_params[4]
                    if not astigmatic
                    else (mean_params[4] ** 2 - mean_params[5] ** 2)
                ),
            )

            Params[:, 0] += coords[:, 0]  # X coordinate
            Params[:, 1] += coords[:, 1]  # Y coordinate

            # Calculate the X and Y shifts between previous and current parameters
            result = getXYshift(self.fit_params, Params)

            self.fit_params = Params

            if result is None:
                return

            x_shift, y_shift = result
            self.total_shift += np.array([x_shift, y_shift])

            self.x_buffer = shift_and_append(self.x_buffer, self.total_shift[0])
            self.y_buffer = shift_and_append(self.y_buffer, self.total_shift[1])

        except Exception as e:
            traceback.print_exc()
            print('Failed beads fit: ' + str(e))


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

        FocusStabilizer.instance().parameterChanged.connect(
            lambda value: self.set_param_value(
                FocusStabilizerParams.FOCUS_PARAMETER, value
            )
        )
        FocusStabilizer.instance().calCoeffChanged.connect(
            lambda value: self.set_param_value(FocusStabilizerParams.CAL_COEFF, value)
        )

        self.setMinimumWidth(325)

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `FocusStabilizerView` class.
        '''
        params = [
            {
                'name': str(FocusStabilizerParams.PREVIEW_IMAGE),
                'type': 'bool',
                'value': True,
            },
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
                'name': str(FocusStabilizerParams.STABILIZATION_METHOD),
                'type': 'list',
                'value': FocusStabilizer.METHODS[0],
                'limits': FocusStabilizer.METHODS,
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
                'name': str(FocusStabilizerParams.CAL_COEFF),
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
                FocusStabilizer.instance().X_ROI[0] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.X2):
                FocusStabilizer.instance().X_ROI[1] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.Y1):
                FocusStabilizer.instance().Y_ROI[0] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.Y2):
                FocusStabilizer.instance().Y_ROI[1] = data
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.LINE_WIDTH):
                FocusStabilizer.instance().line_width = data

            if path == FocusStabilizerParams.get_path(
                FocusStabilizerParams.STABILIZATION_METHOD
            ):
                FocusStabilizer.instance().setMethod(data)

            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.INVERTED):
                FocusStabilizer.instance().setInverted(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.USE_CAL):
                FocusStabilizer.instance().setUseCal(data)
            if path == FocusStabilizerParams.get_path(
                FocusStabilizerParams.FOCUS_PARAMETER
            ):
                FocusStabilizer.instance().setParameter(data)
            if path == FocusStabilizerParams.get_path(FocusStabilizerParams.CAL_COEFF):
                FocusStabilizer.instance().setCalCoeff(data)
            if path == FocusStabilizerParams.get_path(
                FocusStabilizerParams.PREVIEW_IMAGE
            ):
                FocusStabilizer.instance().setPreview(data)

    def toggleFocusStabilization(self):
        '''Toggles the focus stabilization.'''
        self.set_param_value(
            FocusStabilizerParams.TRACKING_ENABLED,
            not self.get_param_value(FocusStabilizerParams.TRACKING_ENABLED),
        )

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
