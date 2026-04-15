import logging
import traceback
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Union

import cv2
import numba as nb
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks

from microEye.analysis.filters.spatial import FourierFilter
from microEye.analysis.fitting import pyfit3Dcspline
from microEye.analysis.fitting.fit import CV_BlobDetector
from microEye.analysis.tools.kymograms import get_kymogram_row
from microEye.hardware.stages.stabilization.controller import BaseController
from microEye.hardware.stages.stage import Axis
from microEye.utils.gui_helper import GaussianOffSet

warnings.filterwarnings('ignore', category=OptimizeWarning)


class ROI:
    '''Represents a Region of Interest (ROI)'''

    def __init__(self, x1: float, y1: float, x2: float, y2: float, linewidth: int = 1):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.linewidth = linewidth

    def get_coordinates(self) -> tuple:
        '''Return the coordinates of the ROI'''
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def X(self) -> np.ndarray:
        '''Return the X coordinates of the ROI'''
        return np.array([self.x1, self.x2])

    @property
    def Y(self) -> np.ndarray:
        '''Return the Y coordinates of the ROI'''
        return np.array([self.y1, self.y2])

    @property
    def width(self) -> float:
        '''Return the width of the ROI'''
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        '''Return the height of the ROI'''
        return abs(self.y2 - self.y1)

    @property
    def top_left(self) -> tuple:
        '''Return the top-left corner of the ROI'''
        return (min(self.x1, self.x2), min(self.y1, self.y2))

    @property
    def bottom_right(self) -> tuple:
        '''Return the bottom-right corner of the ROI'''
        return (max(self.x1, self.x2), max(self.y1, self.y2))


class ROIManager:
    '''Manages Regions of Interest (ROIs)'''

    def __init__(self):
        self._rois = {
            'z': ROI(25.5, 25.5, 25.5, 281.5),  # Default Z ROI
            'xy': ROI(50.5, 25.5, 150.5, 150.5),  # Default XY ROI
        }

    def add_roi(self, name: str, roi: ROI):
        '''Add a new ROI'''
        self._rois[name] = roi

    def get_roi(self, name: str) -> ROI:
        '''Get an ROI by name'''
        return self._rois.get(name)

    def remove_roi(self, name: str):
        '''Remove an ROI by name'''
        self._rois.pop(name, None)

    def get_line_profile(self, image: np.ndarray) -> np.ndarray:
        """Extract line profile from the image using the 'z' ROI"""
        roi = self.get_roi('z')
        if roi is None:
            raise ValueError("ROI 'z' not found")

        return get_kymogram_row(image, roi.X, roi.Y, roi.linewidth)

    def get_roi_region(self, name: str, image: np.ndarray) -> np.ndarray:
        '''Extract ROI region from image'''
        roi = self.get_roi(name)
        if roi is None:
            raise ValueError(f"ROI '{name}' not found")

        x1, y1, x2, y2 = map(int, roi.get_coordinates())
        return image[y1 : y2 + 1, x1 : x2 + 1]

    def set_linewidth(self, linewidth: int, name: str = 'z'):
        '''Set the line width of an ROI'''
        roi = self.get_roi(name)
        if roi is None:
            raise ValueError(f"ROI '{name}' not found")
        roi.linewidth = linewidth

    def get_linewidth(self, name: str = 'z') -> int:
        '''Get the line width of an ROI'''
        roi = self.get_roi(name)
        if roi is None:
            raise ValueError(f"ROI '{name}' not found")
        return roi.linewidth

    def get_config(self) -> dict:
        '''Get the configuration of all ROIs'''
        config = {}
        for name, roi in self._rois.items():
            config[name] = {
                'x1': roi.x1,
                'y1': roi.y1,
                'x2': roi.x2,
                'y2': roi.y2,
                'linewidth': roi.linewidth,
            }
        return config

    def load_config(self, config: dict[str, dict]):
        '''Load the configuration of all ROIs'''
        for name, roi_data in config.items():
            self._rois[name] = ROI(
                roi_data['x1'],
                roi_data['y1'],
                roi_data['x2'],
                roi_data['y2'],
                roi_data.get('linewidth', 1),
            )


class CalibrationManager:
    def __init__(self):
        self.__cal_coeffs = np.ones((3,))  # nm/pixel

    def get_coefficient(self, axis: Axis = None) -> Union[float, np.ndarray]:
        '''Get calibration coefficient for axis'''
        if axis is None:
            return self.__cal_coeffs.copy()

        return self.__cal_coeffs[axis.axis_index()]

    def set_coefficient(self, axis: Axis, value: float) -> None:
        '''Set calibration coefficient for axis'''
        self.__cal_coeffs[axis.axis_index()] = value

    def convert_to_physical(self, pixels: float, axis: Axis) -> float:
        '''Convert pixel measurement to physical units'''
        return pixels * self.get_coefficient(axis)


class PositionTracker:
    def __init__(self, history_length: int = 1000):
        self.length = history_length
        self.time_points = np.zeros((self.length,), dtype=float)
        self.positions = np.zeros((self.length, 3), dtype=float)

    def advance_time(self, t: float):
        self.time_points = self._shift_and_append(self.time_points, t)

    def append_axis(self, axis: Axis, value: float):
        idx = axis.axis_index()
        self.positions[:, idx] = self._shift_and_append(self.positions[:, idx], value)

    def snapshot(self):
        return self.time_points.copy(), self.positions.copy()

    def reset(self):
        self.positions.fill(0.0)

    def fill_time(self, t: float):
        self.time_points.fill(t)

    @property
    def last_interval(self):
        return self.time_points[-1] - self.time_points[-2]

    @property
    def last_time(self):
        return self.time_points[-1]

    @property
    def last_row(self) -> np.ndarray:
        return np.array([self.last_time, *self.positions[-1, :].tolist()])

    @staticmethod
    def _shift_and_append(arr: np.ndarray, new_value: float):
        arr = np.roll(arr, -1)
        arr[-1] = new_value
        return arr


class DataLogger:
    def __init__(self):
        self.file = None
        self.num_frames_saved = 0

    def start(self, filename: str):
        self.file = filename
        self.num_frames_saved = 0

    def stop(self):
        self.file = None
        self.num_frames_saved = 0

    def is_active(self) -> bool:
        return self.file is not None

    def log(self, row: np.ndarray):
        '''
        row is shape (4,) -> [T, X, Y, Z]
        '''
        if self.file is None:
            return
        with open(self.file, 'ab') as f:
            if self.num_frames_saved == 0:
                header = ';'.join(
                    [
                        'Execution Time [s]',
                        'X Shift [nm]',
                        'Y Shift [nm]',
                        'Z Parameter [nm]',
                    ]
                )
                f.write(header.encode('utf-8') + b'\n')
            np.savetxt(f, row.reshape(1, -1), delimiter=';')
        self.num_frames_saved += 1


class StabilizationMethods(Enum):
    REFLECTION = 'reflection'
    BEADS = 'beads'
    BEADS_ASTIGMATIC = 'beads astigmatic'
    HYBRID = 'hybrid'

    def __str__(self):
        return self.value


class StabilizationStrategy(ABC):
    '''Base class for stabilization method strategies'''

    PARAMS : dict[str, dict] = {}

    def __init__(self):
        # Isolate mutable nested defaults between strategy instances.
        self.__tunable_params = deepcopy(self.PARAMS)

    @abstractmethod
    def fit(
        self,
        image: np.ndarray,
        roi_manager: ROIManager,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        '''Fit parameters from image'''
        raise NotImplementedError('Subclasses must implement this')

    def is_image(self, data: np.ndarray):
        if data.ndim > 2:
            raise ValueError('Data should be one or two dimensional array.')

        if data.ndim == 1:
            return False
        elif data.ndim == 2:
            return True

    @abstractmethod
    def get_shifts(
        self,
        old_z: float,
        new_z: float,
        old_xy: np.ndarray,
        new_xy: np.ndarray,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        '''Get shifts'''
        raise NotImplementedError('Subclasses must implement this')

    def get_tunable_params(self) -> list[dict]:
        return [
            {
                'name': key,
                'type': param.get('type', 'str'),
                'value': param.get('value'),
                'step': param.get('step'),
                'limits': param.get('limits'),
            }
            for key, param in self.__tunable_params.items()
        ]

    def get_param_values(self) -> dict:
        return {key: param['value'] for key, param in self.__tunable_params.items()}

    def set_param_values(self, values: dict):
        if not isinstance(values, dict):
            return
        for name, value in values.items():
            self.set_param(name, value)

    def set_param(self, name: str, value) -> bool:
        if name in self.__tunable_params:
            self.__tunable_params[name]['value'] = value
            return True
        return False

    def get_param(self, name: str, default=None):
        if name in self.__tunable_params:
            return self.__tunable_params[name].get('value', default)
        return default


class ReflectionStrategy(StabilizationStrategy):
    PARAMS = {
        'prominence': {
            'type': 'float',
            'value': 1.0,
            'step': 0.1,
            'limits': [0.0, 4096.0],
        },
        'min_distance': {
            'type': 'int',
            'value': 20,
            'step': 1,
            'limits': [1, 1024],
        },
        'fit_half_window': {
            'type': 'int',
            'value': 18,
            'step': 1,
            'limits': [3, 256],
        },
        'sigma_min': {
            'type': 'float',
            'value': 1.0,
            'step': 0.1,
            'limits': [0.1, 1024.0],
        },
        'sigma_max': {
            'type': 'float',
            'value': 100.0,
            'step': 0.1,
            'limits': [0.1, 2048.0],
        },
        'maxfev': {
            'type': 'int',
            'value': 1000,
            'step': 50,
            'limits': [50, 100000],
        },
    }

    def fit(
        self,
        image: np.ndarray,
        roi_manager: ROIManager,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        '''Fit parameters using reflection method'''
        # Get line profile
        line_profile = (
            roi_manager.get_line_profile(image) if self.is_image(image) else image
        )

        # Find peaks and fit Gaussian
        fit_params = self.peak_fit(line_profile)

        # Process results
        params = {'xy': None, 'z': None}
        if fit_params is not None:
            z_center = fit_params[1]  # x0 from GaussianOffSet
            z_physical = z_center * calibration_manager.get_coefficient(Axis.Z)
            params['z'] = z_physical

        return {
            'params': params,
            'fit_params': fit_params,
            'line_profile': line_profile.copy().tolist(),
        }

    def get_shifts(
        self,
        old_z: float,
        new_z: float,
        old_xy: np.ndarray,
        new_xy: np.ndarray,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        shifts = {
            Axis.X: 0.0,
            Axis.Y: 0.0,
            Axis.Z: 0.0,
        }

        if isinstance(old_z, (int, float)) and isinstance(new_z, (int, float)):
            shifts[Axis.Z] = new_z - old_z

        return shifts

    def _detect_candidates(self, y: np.ndarray, **kwargs):
        med = float(np.nanmedian(y))
        mad = float(np.nanmedian(np.abs(y - med)))
        sigma_n = max(1.4826 * mad, 1e-6)

        q50 = float(np.nanpercentile(y, 50))
        q99 = float(np.nanpercentile(y, 99))
        dyn = max(q99 - q50, 1e-6)

        # Adaptive thresholds
        k_noise = float(kwargs.get('k_noise', 6.0))
        k_dyn = float(kwargs.get('k_dyn', 0.05))

        prom_auto = max(k_noise * sigma_n, k_dyn * dyn)

        # Keep user floor, then adaptive value
        prom = max(float(self.get_param('prominence')), prom_auto)
        prom = float(np.clip(prom, 1.0, 2048.0))

        # Height gate to avoid baseline noise lobes
        min_height = med + max(2.0 * sigma_n, 0.05 * dyn)

        peaks, props = find_peaks(
            y,
            height=min_height,
            prominence=prom,
            distance=self.get_param('min_distance'),
        )

        # Fallback: at least one candidate
        if len(peaks) == 0:
            idx = int(np.nanargmax(y))
            peaks = np.array([idx], dtype=int)
            props = {'prominences': np.array([max(y[idx] - med, 1e-6)], dtype=float)}

        prominences = np.asarray(
            props.get('prominences', np.ones_like(peaks, dtype=float)),
            dtype=float,
        )
        return peaks, prominences

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
            y = np.asarray(data, dtype=np.float64).squeeze()
            if y.ndim != 1 or y.size < 5:
                return None

            finite = np.isfinite(y)
            if not np.any(finite):
                return None
            if not np.all(finite):
                fill = float(np.nanmedian(y[finite]))
                y = np.where(finite, y, fill)

            peaks, prominences = self._detect_candidates(y)
            pk = int(peaks[int(np.argmax(prominences))])  # strongest prominence only

            left = max(0, pk - self.get_param('fit_half_window'))
            right = min(y.size, pk + self.get_param('fit_half_window') + 1)
            if right - left < 5:
                return None

            x_full = np.arange(y.size, dtype=np.float64)
            x = x_full[left:right]
            yy = y[left:right]

            med_local = float(np.nanmedian(yy))
            mad_local = float(np.nanmedian(np.abs(yy - med_local)))
            sigma_n_local = max(1.4826 * mad_local, 1e-6)

            base = float(np.nanpercentile(yy, 10))
            p90 = float(np.nanpercentile(yy, 90))
            p99 = float(np.nanpercentile(yy, 99))

            amp0 = max(p99 - base, 1.0)
            x0 = float(pk)
            s0 = float(
                np.clip(
                    (right - left) / 6.0,
                    self.get_param('sigma_min'),
                    self.get_param('sigma_max'),
                )
            )
            c0 = base

            p0 = [amp0, x0, s0, c0]

            amp_hi = max(4.0 * amp0, 10.0)

            # Tight offset bounds to prevent fit dropping below baseline
            offset_lo = base
            offset_hi = p90 + 2.5 * sigma_n_local

            lower = [0.0, float(left), self.get_param('sigma_min'), float(offset_lo)]
            upper = [
                float(amp_hi),
                float(right - 1),
                self.get_param('sigma_max'),
                float(offset_hi),
            ]

            fit_params, _ = curve_fit(
                GaussianOffSet,
                x,
                yy,
                p0=p0,
                bounds=(lower, upper),
                maxfev=self.get_param('maxfev'),
            )

            return fit_params

        except Exception:
            return None


@nb.njit(parallel=True)
def fast_nn_assignment(
    currentFrame: np.ndarray, nextFrame: np.ndarray, minDistance=0.0, maxDistance=30.0
):
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
def get_xy_shift(
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
    return np.array(x_shifts), np.array(y_shifts)


class FiducialStrategy(StabilizationStrategy):
    PARAMS = {
        # roi_size should be odd to have a center pixel
        'roi_size': {
            'type': 'int',
            'value': 13,
            'step': 2,
            'limits': [7, 51],
        },
        'relative_threshold': {
            'type': 'float',
            'value': 0.5,
            'step': 0.05,
            'limits': [0.0, 1.0],
        },
        'fit_initial_sigma': {
            'type': 'float',
            'value': 1.0,
            'step': 0.1,
            'limits': [0.1, 100.0],
        },
        'filter_center': {
            'type': 'int',
            'value': 36,
            'step': 1,
            'limits': [1, 512],
        },
        'filter_width': {
            'type': 'int',
            'value': 120,
            'step': 1,
            'limits': [1, 512],
        },
        'filter_profile': {
            'type': 'list',
            'value': 'Gaussian',
            'step': None,
            'limits': ['Gaussian', 'Butterworth', 'Ideal'],
        },
        'filter_bandtype': {
            'type': 'list',
            'value': 'Band',
            'step': None,
            'limits': ['Low', 'High', 'Band'],
        },
    }

    def __init__(self, fit_method: bool = True):
        '''
        Parameters
        ----------
        fit_method : bool
            - If True, use astigmatic fitting method.
            - If False, use standard 2D Gaussian fitting.
        '''
        super().__init__()

        self.fit_method = fit_method
        self.fourier_filter = FourierFilter(
            center=self.get_param('filter_center', 36),
            width=self.get_param('filter_width', 120),
        )

    def fit(
        self,
        image: np.ndarray,
        roi_manager: ROIManager,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        '''Fit parameters using fiducial method'''
        params = {'xy': None, 'z': None}

        localizations = {'x': [], 'y': []}

        if not self.is_image(image):
            return {'params': params, 'fit_params': None}

        # Get ROI region
        roi_region = roi_manager.get_roi_region('xy', image)

        # Find peaks and fit Gaussian
        fit_params = self.fiducials_fit(
            roi_region,
            x=roi_manager.get_roi('xy').top_left[0],
            y=roi_manager.get_roi('xy').top_left[1],
        )

        # Process results
        if fit_params is not None:
            z_param = (
                fit_params[:, 4] ** 2 - fit_params[:, 5] ** 2
                if self.fit_method
                else fit_params[:, 4]
            ) * calibration_manager.get_coefficient(Axis.Z)

            z_param = np.nanmean(controller.outlier_rejection(z_param))

            params['z'] = float(z_param)

            params['xy'] = fit_params

            localizations['x'] = fit_params[:, 0].tolist()
            localizations['y'] = fit_params[:, 1].tolist()

        return {
            'params': params,
            'fit_params': fit_params,
            'localizations': localizations,
        }

    def get_shifts(
        self,
        old_z: float,
        new_z: float,
        old_xy: np.ndarray,
        new_xy: np.ndarray,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        shifts = {
            Axis.X: 0.0,
            Axis.Y: 0.0,
            Axis.Z: 0.0,
        }

        xy_shift = get_xy_shift(old_xy, new_xy)

        if xy_shift is not None:
            x_shifts, y_shifts = xy_shift

            x_shift = np.nanmean(
                controller.outlier_rejection(x_shifts)
            ) * calibration_manager.get_coefficient(Axis.X)
            y_shift = np.nanmean(
                controller.outlier_rejection(y_shifts)
            ) * calibration_manager.get_coefficient(Axis.Y)

            shifts[Axis.X] = x_shift
            shifts[Axis.Y] = y_shift

        if isinstance(old_z, (int, float)) and isinstance(new_z, (int, float)):
            shifts[Axis.Z] = new_z - old_z

        return shifts

    def __update_filter_params(self):
        to_update = {
            'center': 'filter_center',
            'width': 'filter_width',
            'type': 'filter_profile',
            'pass_type': 'filter_bandtype',
        }

        for attr, param_name in to_update.items():
            value = self.get_param(param_name)
            if value is not None and value != self.fourier_filter.parameters.get(attr):
                self.fourier_filter.parameters[attr] = value

    def fiducials_fit(
        self,
        data: np.ndarray,
        x: float = 0.0,
        y: float = 0.0,
    ):
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
            self.__update_filter_params()
            filtered = self.fourier_filter.run(data)

            # Threshold the image
            _, th_img = cv2.threshold(
                filtered,
                np.quantile(filtered, 1 - 1e-4)
                * float(self.get_param('relative_threshold', 0.5)),
                255,
                cv2.THRESH_BINARY,
            )

            # Detect blobs
            points, _ = CV_BlobDetector().find_peaks_preview(th_img, None)

            if len(points) < 1:
                return None

            # Get the ROI list for fitting
            rois, coords = pyfit3Dcspline.get_roi_list(
                data, points, int(self.get_param('roi_size', 13))
            )

            Params, _, _ = pyfit3Dcspline.CPUmleFit_LM(
                rois,
                4 if self.fit_method else 2,
                np.array([self.get_param('fit_initial_sigma', 1.0)]),
                None,
                0,
            )

            if Params is None and len(Params) < 1:
                return None
            elif Params.ndim == 1:
                Params = Params[np.newaxis, :]

            Params[:, 0] += coords[:, 0] + int(x) + 0.5  # X coordinate
            Params[:, 1] += coords[:, 1] + int(y) + 0.5  # Y coordinate

            return Params
        except Exception as e:
            logging.getLogger(__name__).error('Failed beads fit!')
            traceback.print_exc()
            return None


class HybridStrategy(StabilizationStrategy):
    '''
    Hybrid strategy:
    - Z from ReflectionStrategy
    - XY from FiducialStrategy
    '''
    PARAMS = {
        **ReflectionStrategy.PARAMS,
        **FiducialStrategy.PARAMS,
    }

    def __init__(
        self,
        z_strategy: ReflectionStrategy = None,
        xy_strategy: FiducialStrategy = None,
        xy_fit_method: bool = False,
    ):
        super().__init__()
        self.z_strategy = z_strategy if z_strategy is not None else ReflectionStrategy()
        self.xy_strategy = (
            xy_strategy if xy_strategy is not None else FiducialStrategy(xy_fit_method)
        )

    def fit(
        self,
        image: np.ndarray,
        roi_manager: ROIManager,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        z_res = self.z_strategy.fit(image, roi_manager, calibration_manager, controller)
        xy_res = self.xy_strategy.fit(
            image, roi_manager, calibration_manager, controller
        )

        if z_res is None:
            z_res = {'params': {'z': None}, 'fit_params': None, 'line_profile': []}
        if xy_res is None:
            xy_res = {
                'params': {'xy': None},
                'localizations': {'x': [], 'y': []},
            }

        params = {
            'z': z_res.get('params', {}).get('z', None),
            'xy': xy_res.get('params', {}).get('xy', None),
        }

        return {
            'params': params,
            'fit_params': z_res.get('fit_params'),
            'line_profile': z_res.get('line_profile', []),
            'localizations': xy_res.get('localizations', {'x': [], 'y': []}),
        }

    def get_shifts(
        self,
        old_z: float,
        new_z: float,
        old_xy: np.ndarray,
        new_xy: np.ndarray,
        calibration_manager: CalibrationManager,
        controller: BaseController,
    ) -> dict:
        z_shifts = self.z_strategy.get_shifts(
            old_z,
            new_z,
            old_xy,
            new_xy,
            calibration_manager,
            controller,
        )
        xy_shifts = self.xy_strategy.get_shifts(
            old_z,
            new_z,
            old_xy,
            new_xy,
            calibration_manager,
            controller,
        )

        return {
            Axis.X: xy_shifts.get(Axis.X, 0.0),
            Axis.Y: xy_shifts.get(Axis.Y, 0.0),
            Axis.Z: z_shifts.get(Axis.Z, 0.0),
        }
