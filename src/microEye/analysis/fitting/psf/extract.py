import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Union

import cv2
import h5py
import numba as nb
import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, peak_prominences
from tqdm import tqdm

from microEye.analysis.fitting.psf.stats import *
from microEye.analysis.fitting.results import PARAMETER_HEADERS
from microEye.utils.uImage import TiffSeqHandler, ZarrImageSequence, uImage


def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


class PSFdata:
    def __init__(
        self, data_dict: dict[str, Union[int, np.ndarray, list, float]]
    ) -> None:
        self._data = data_dict
        self._last_field = 0

        self.stats_calculator = StatsCalculator(
            zero_plane=self.zero_plane,
            z_step=self.z_step,
            fitting_method=self.fitting_method,
        )

    def get_field_psf(self, grid_size: int = 3):
        '''
        Get PSF field for the given grid size.
        '''
        if self._last_field == grid_size:
            return

        width, height = self.dim
        width *= self.upsample
        height *= self.upsample
        offset = self.roi_size / 2

        def get_grid_cell_index(x_start, y_start):
            if not np.isfinite(x_start) or not np.isfinite(y_start):
                return -1
            cell_width = width // grid_size
            cell_height = height // grid_size
            i = (y_start + offset) // cell_height
            j = (x_start + offset) // cell_width
            return int(i * grid_size + j)

        for zslice in self._data['zslices']:
            zslice['field'] = [[] for _ in range(grid_size**2)]
            zslice['field_idx'] = np.zeros(zslice['rois'].shape[0], dtype=np.uint16)
            for i in range(zslice['rois'].shape[0]):
                idx = get_grid_cell_index(*zslice['coords'][i])
                if idx >= 0:
                    zslice['field'][idx].append(zslice['rois'][i].copy())
                zslice['field_idx'][i] = idx

        self._last_field = grid_size

    @property
    def zslices(self) -> list[dict[str, Union[int, np.ndarray, list, float]]]:
        return self._data.get('zslices')

    @property
    def fitting_method(self) -> float:
        return self._data.get('fit_method')

    @property
    def available_stats(self) -> list[str]:
        # self.stats_calculator.available_stats keys
        return list(self.stats_calculator.available_stats.keys())

    @property
    def pixel_size(self) -> float:
        return self._data.get('pixel_size')

    @property
    def z_step(self) -> float:
        step = self._data.get('z_step', 10)
        if step:
            return step
        else:
            return 10

    @property
    def roi_size(self) -> int:
        return self._data.get('roi_size')

    @property
    def roi_info(self):
        return self._data.get('roi_info')

    @property
    def origin(self):
        if self.roi_info:
            return self.roi_info[0]
        else:
            return (0, 0)

    @property
    def dim(self):
        if self.roi_info:
            return tuple(self.roi_info[1])
        else:
            return self.shape[-1], self.shape[-2]

    @property
    def upsample(self) -> int:
        return self._data.get('upsample')

    @property
    def zero_plane(self) -> Optional[int]:
        return self._data.get('zero_plane')

    @zero_plane.setter
    def zero_plane(self, value: int):
        self._data['zero_plane'] = value

    @property
    def shape(self) -> tuple:
        return self._data.get('shape')

    @property
    def path(self) -> str:
        return self._data.get('stack')

    @property
    def headers(self) -> str:
        return PARAMETER_HEADERS[self.fitting_method]

    @property
    def rois(self):
        return [zslice['rois'] for zslice in self.zslices]

    @property
    def counts(self) -> np.ndarray:
        return np.array([zslice['count'] for zslice in self.zslices])

    @property
    def mean(self) -> np.ndarray:
        return np.array([zslice['mean'] for zslice in self.zslices])

    @property
    def median(self) -> np.ndarray:
        return np.array([zslice['median'] for zslice in self.zslices])

    @property
    def std(self) -> np.ndarray:
        return np.array([zslice['std'] for zslice in self.zslices])

    @property
    def coords(self) -> np.ndarray:
        return np.array([zslice['coords'] for zslice in self.zslices])

    @property
    def params(self) -> np.ndarray:
        return np.array([zslice['params'] for zslice in self.zslices])

    @property
    def crlbs(self) -> np.ndarray:
        return np.array([zslice['crlbs'] for zslice in self.zslices])

    @property
    def loglike(self) -> np.ndarray:
        return np.array([zslice['loglike'] for zslice in self.zslices])

    @property
    def field_rois(self) -> list:
        return [zslice.get('field') for zslice in self.zslices]

    @property
    def field_indecies(self) -> np.ndarray:
        return np.array([zslice['field_idx'] for zslice in self.zslices])

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __repr__(self) -> str:
        return f'PSFdata({self._data})'

    def __str__(self) -> str:
        return f'PSFdata({self._data})'

    def __len__(self) -> int:
        return len(self.zslices)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.zslices):
            result = self.zslices[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

    def get_z_slice(
        self,
        z_index: int,
        type: str = 'mean',
        roi_index: int = 0,
        grid_size: int = 1,
        normalize: bool = False,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the XY slice for the given z-index.

        Parameters:
        -----------
        z_index : int
            The z-index of the slice
        type : str, optional
            The type of data to return, by default 'mean'
            Options: 'mean', 'std', 'median', 'roi'
        roi_index : int, optional
            The index of the ROI to return, by default 0
        grid_size : int, optional
            The grid size for field PSF, by default 1
        normalize : bool, optional
            Whether to normalize the data, by default False

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            The data array and grid overlay
        """
        # check if Z index is in bounds
        if z_index >= len(self.zslices):
            raise IndexError(f'Z index {z_index} out of bounds')

        # get the z-slice
        zslice = self.zslices[z_index]

        # check if roi index is in bounds
        if roi_index >= zslice['count']:
            raise IndexError(
                f'ROI index {roi_index} out of bounds for z-slice {z_index}'
            )

        # limit grid_size from 1 to 5
        grid_size = max(1, min(grid_size, 5))

        # get the ROI
        if type in ['mean', 'median', 'std', 'roi']:
            if grid_size == 1:
                if type == 'roi':
                    data = zslice['rois'][roi_index]
                    return None if np.isnan(data).any() else data, None
                return zslice[type], None
            else:
                self.get_field_psf(grid_size)
                colors = [
                    pg.intColor(i, grid_size**2).getRgb() for i in range(grid_size**2)
                ]

                roi_size = self.roi_size
                size = roi_size * grid_size

                # Create empty arrays for visualization
                psf_image = np.zeros((size, size))
                # RGBA for grid overlay
                grid_overlay = np.zeros((size, size, 4))

                for field_idx, field_rois in enumerate(zslice['field']):
                    if not field_rois:
                        continue

                    field_rois = np.array(field_rois)

                    if type == 'mean':
                        stat = np.nanmean(field_rois, axis=0)
                    elif type == 'median':
                        stat = np.nanmedian(field_rois, axis=0)
                    elif type == 'std':
                        stat = np.nanstd(field_rois, axis=0)
                    else:  # Single ROI
                        stat = (
                            field_rois[roi_index]
                            if roi_index < len(field_rois)
                            else None
                        )

                    if stat is not None:
                        y = field_idx // grid_size
                        x = field_idx % grid_size
                        psf_image[
                            y * roi_size : (y + 1) * roi_size,
                            x * roi_size : (x + 1) * roi_size,
                        ] = 2**16 * stat / np.sum(stat) if normalize else stat

                        # Add colored overlay for grid visualization
                        color = colors[field_idx]
                        grid_overlay[
                            y * roi_size : (y + 1) * roi_size,
                            x * roi_size : (x + 1) * roi_size,
                        ] = [color[0], color[1], color[2], 128]
                return psf_image, grid_overlay
        else:
            raise ValueError(f'Invalid type {type}')

    def get_longitudinal_slice(
        self,
        index: int,
        type: str = 'mean',
        grid_size: int = 1,
        sagittal: bool = True,
        normalize: bool = False,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the longitudinal slice for the given index.

        Parameters
        ----------
        index : int
            The index of the slice
        type : str, optional
            The type of data to return, by default 'mean'
            Options: 'mean', 'median', 'std'
        grid_size : int, optional
            The grid size for field PSF, by default 1
        sagittal : bool, optional
            Whether to get the sagittal slice, or coronal, by default True
        normalize : bool, optional
            Whether to normalize the data, by default False

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The data array and grid overlay
        """
        # check if index is in bounds
        if index >= self.roi_size:
            raise IndexError(f'Y index {index} out of bounds')

        # limit grid_size from 1 to 5
        grid_size = np.clip(grid_size, 1, 5)

        roi_height, roi_width = self.roi_size, len(self.zslices)

        if type not in ['mean', 'median', 'std']:
            raise ValueError(f'Invalid type {type}')

        if grid_size == 1:
            data = np.array(
                [
                    zslice[type][index] if sagittal else zslice[type][:, index]
                    for zslice in self.zslices
                ]
            ).T
            return data, None

        self.get_field_psf(grid_size)
        colors = [pg.intColor(i, grid_size**2).getRgb() for i in range(grid_size**2)]

        height, width = roi_height * grid_size, roi_width * grid_size

        psf_image = np.zeros((height, width))
        grid_overlay = np.zeros((height, width, 4), dtype=np.uint8)

        stat_func = {'mean': np.nanmean, 'median': np.nanmedian, 'std': np.nanstd}[type]

        grid_painted = []
        for i, zslice in enumerate(self.zslices):
            for field_idx, field_rois in enumerate(zslice['field']):
                if not field_rois:
                    continue

                field_rois = (
                    np.array(field_rois)[:, index, :]
                    if sagittal
                    else np.array(field_rois)[..., index]
                )
                stat = stat_func(field_rois, axis=0)

                if stat is not None:
                    y, x = divmod(field_idx, grid_size)
                    psf_image[
                        y * roi_height : (y + 1) * roi_height, x * roi_width + i
                    ] = stat

                    if field_idx not in grid_painted:
                        color = colors[field_idx]
                        grid_overlay[
                            y * roi_height : (y + 1) * roi_height,
                            x * roi_width : (x + 1) * roi_width,
                        ] = [*color[:3], 128]
                        grid_painted.append(field_idx)

        return psf_image, grid_overlay

    def get_x_slice(
        self, x_index: int, type: str = 'mean', grid_size: int = 1
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the YZ slice for the given x-index.

        Parameters
        ----------
        x_index : int
            The x-index of the slice
        type : str, optional
            The type of data to return, by default 'mean'
            Options: 'mean', 'median', 'std'
        grid_size : int, optional
            The grid size for field PSF, by default 1

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The data array and grid overlay
        """
        return self.get_longitudinal_slice(
            x_index, type=type, grid_size=grid_size, sagittal=False
        )

    def get_y_slice(
        self, y_index: int, type: str = 'mean', grid_size: int = 1
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the XZ slice for the given y-index.

        Parameters
        ----------
        y_index : int
            The y-index of the slice
        type : str, optional
            The type of data to return, by default 'mean'
            Options: 'mean', 'median', 'std'
        grid_size : int, optional
            The grid size for field PSF, by default 1

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The data array and grid overlay
        """
        return self.get_longitudinal_slice(
            y_index, type=type, grid_size=grid_size, sagittal=True
        )

    def get_volume(self, type: str = 'mean'):
        """
        Get the 3D volume for the given type.

        Parameters
        ----------
        type : str, optional
            The type of data to return, by default 'mean'

            Options:
                'mean' - Mean intensity

                'median' - Median intensity

                'std' - Standard deviation of intensity

        Returns
        -------
        np.ndarray
            The 3D volume data
        """
        if type not in ['mean', 'median', 'std']:
            raise ValueError(f'Invalid type {type}')

        volume_data = np.zeros((len(self), self.roi_size, self.roi_size))

        for i, z_slice in enumerate(self.zslices):
            volume_data[i] = (
                z_slice[type]
                if z_slice[type] is not None
                else np.zeros((self.roi_size, self.roi_size))
            )

        return volume_data

    def get_ratio(self) -> float:
        '''
        Get the ratio of Z step to lateral pixel size.

        Returns
        -------
        float
            The ratio of Z step to lateral pixel size
        '''
        return self.z_step * self.upsample / self.pixel_size

    def get_intensity_stats(self):
        '''
        Get the intensity statistics for all z-slices.

        Returns
        -------
        _indices, _mean, _median, _std: tuple[np.ndarray, list, list, list]
            The indices, mean, median, and standard deviation of the intensity values
        '''
        _indices = (np.arange(len(self)) - self.zero_plane) * self.z_step
        _mean = []
        _median = []
        _std = []

        for z_slice in self.zslices:
            if z_slice['rois'] is not None:
                valid_rois = z_slice['rois'][
                    ~np.isnan(z_slice['rois']).any(axis=(1, 2))
                ]
                if len(valid_rois) > 0:
                    _mean.append(np.mean(valid_rois))
                    _median.append(np.median(valid_rois))
                    _std.append(np.std(valid_rois))
                else:
                    _mean.append(np.nan)
                    _median.append(np.nan)
                    _std.append(np.nan)
            else:
                _mean.append(np.nan)
                _median.append(np.nan)
                _std.append(np.nan)

        return _indices, _mean, _median, _std

    def get_stats(
        self,
        selected_stat: str,
        confidence_method: ConfidenceMethod = ConfidenceMethod.NONE,
        confidence_level: float = 0.95,
    ):
        """
        Get the statistic data for all z-slices.

        Parameters
        ----------
        selected_stat : str
            The selected statistic to return
            Options:
                'Counts', 'Sigma', 'Sigma (sum)', 'Sigma (diff)', 'Sigma (abs(diff))',
                'Intensity', 'Background'
        confidence_method : ConfidenceMethod, optional
            The method to use for confidence interval calculation,
            by default ConfidenceMethod.NONE

        Returns
        -------
        z_indices, param_stat: tuple[np.ndarray, np.ndarray]
            The z-indices and the statistic data for all z-slices
        """
        # Simply delegate to the calculator
        return self.stats_calculator.get_stats(
            self.zslices,
            selected_stat,
            confidence_method,
            confidence_level,
        )

    def get_z_cal(
        self,
        selected_stat: str,
        region: tuple[int, int],
        confidence_method: ConfidenceMethod = ConfidenceMethod.NONE,
        confidence_level: float = 0.95,
        **kwargs,
    ) -> Union[SlopeResult, CurveResult, dict]:
        '''
        Get the slope or curve fit for the selected statistic.

        Parameters
        ----------
        selected_stat : str
            The selected statistic to use for slope or curve fit
        region : tuple[int, int]
            The region to use for slope or curve fit
        confidence_method : ConfidenceMethod, optional
            The method to use for confidence interval calculation,
            by default ConfidenceMethod.NONE
        confidence_level : float, optional
            The confidence level for the confidence interval calculation,
            by default 0.95

        Keyword Arguments
        -----------------
        method : CurveFitMethod, optional
            The method to use for curve fitting, by default CurveFitMethod.CSPLINE
        derivative_threshold : float, optional
            The threshold for derivative-based zero crossing detection,
            by default 0.01
        smoothing : float, optional
            The smoothing factor for curve fitting, by default None

        Returns
        -------
        Union[SlopeResult, CurveResult]
            The slope or curve fit result
        '''
        method: CurveFitMethod = kwargs.get('method', CurveFitMethod.LINEAR)

        if method == CurveFitMethod.LINEAR:
            return SlopeAnalyzer.fit_stat_slope(
                selected_stat,
                lambda: self.get_stats(
                    selected_stat, confidence_method, confidence_level
                ),
                region,
                confidence_method,
                confidence_level,
            )
        elif method == CurveFitMethod.ASTIGMATIC_PSF:
            return None
        else:
            derivative_threshold: float = kwargs.get('derivative_threshold', 0.01)
            smoothing_factor: float = kwargs.get('smoothing', 0.1)
            return CurveAnalyzer.fit_stat_curve(
                selected_stat,
                lambda: self.get_stats(
                    selected_stat, confidence_method, confidence_level
                ),
                region,
                method,
                derivative_threshold,
                smoothing_factor,
            )

    def adjust_zero_plane(
        self, selected_stat: str, method: str, region: tuple[int, int]
    ):
        """
        Adjust the zero plane based on the selected statistic and method.

        Parameters
        ----------
        selected_stat : str
            The selected statistic to use for zero plane adjustment

            Options:

                'Counts', 'Sigma', 'Sigma (sum)', 'Sigma (diff)', 'Sigma (abs(diff))',
                'Intensity', 'Background'
        method : str
            The method to use for zero plane adjustment

            Options: 'Peak', 'Valley', 'Gaussian Fit', 'Gaussian Fit (Inverted)',
            'Manual'
        region : tuple[int, int]
            The region to use for zero plane adjustment

        Returns
        -------
        int
            The new zero plane
        """
        if selected_stat == 'Sigma':
            selected_stat = 'Sigma (sum)'

        start, end = map(
            int,
            [v / self.z_step + self.zero_plane for v in region],
        )

        # Store the old zero plane for potential restoration
        self.old_zero_plane = self.zero_plane

        _, stat_data, _, _ = self.get_stats(selected_stat, ConfidenceMethod.NONE)

        # Slice the data according to the specified range
        x = np.arange(start, end)
        y = stat_data[start:end]

        if method == 'Peak':
            # Find peaks
            peaks, _ = find_peaks(y)
            if len(peaks) > 0:
                # Get peak prominences
                prominences = peak_prominences(y, peaks)[0]
                # Select the most prominent peak
                max_peak = peaks[np.argmax(prominences)]
                new_zero_plane = start + max_peak
            else:
                new_zero_plane = start + np.argmax(y)
        elif method == 'Valley':
            # Invert the data to find valleys as peaks
            inverted_y = -y
            valleys, _ = find_peaks(inverted_y)
            if len(valleys) > 0:
                # Get valley prominences
                prominences = peak_prominences(inverted_y, valleys)[0]
                # Select the most prominent valley
                max_valley = valleys[np.argmax(prominences)]
                new_zero_plane = start + max_valley
            else:
                new_zero_plane = start + np.argmin(y)
        elif 'Gaussian Fit' in method:
            if 'Inverted' in method:
                y = -y
            try:
                # Initial guess for Gaussian parameters
                a_init = np.max(y) - np.min(y)
                x0_init = x[np.argmax(y)]
                sigma_init = (end - start) / 4
                offset_init = np.min(y)

                # Perform Gaussian fit
                popt, _ = curve_fit(
                    gaussian, x, y, p0=[a_init, x0_init, sigma_init, offset_init]
                )
                new_zero_plane = int(popt[1])  # x0 is the center of the Gaussian
            except Exception:
                # Fallback to original plane if fitting fails
                new_zero_plane = self.zero_plane
        else:
            return self.zero_plane  # Keep current zero plane for 'Manual' method

        # Update the zero plane
        self.zero_plane = new_zero_plane
        self.stats_calculator.zero_plane = new_zero_plane

        return self.zero_plane

    def get_consistent_rois(self) -> 'PSFdata':
        '''
        Creates a new PSFdata object with consistent ROIs across all z-slices.
        Only keeps coordinates that exist in all slices and ensures consistent ordering.

        Returns
        -------
        PSFdata
            A new PSFdata object with consistent ROIs
        '''
        # Create a copy of the original data
        new_data = self._data.copy()

        # Tolerance for matching coordinates
        tolerance = 2.0 * self.upsample

        # First, identify coordinates that exist in all slices
        all_coords = []
        for zslice in self.zslices:
            params = zslice['params']
            # Only consider non-NaN coordinates
            valid_coords = params[~np.isnan(params[:, 0])][:, :2]
            all_coords.append(valid_coords)

        # Function to find matching coordinates within a tolerance
        def find_matching_coords(coord, coords_list):
            matches = []
            for slice_idx, slice_coords in enumerate(coords_list):
                distances = np.sqrt(np.sum((slice_coords - coord) ** 2, axis=1))
                match_idx = np.argmin(distances)
                if distances[match_idx] <= tolerance:
                    matches.append((slice_idx, match_idx))
            return matches if len(matches) == len(coords_list) else None

        # Find coordinates that exist in all slices
        consistent_coords = []
        consistent_indices = [[] for _ in range(len(all_coords))]

        for _, coord in enumerate(all_coords[0]):
            matches = find_matching_coords(coord, all_coords)
            if matches:
                consistent_coords.append(coord)
                for slice_idx, match_idx in matches:
                    consistent_indices[slice_idx].append(match_idx)

        # Create new zslices with only consistent ROIs
        new_data['zslices'] = []
        for slice_idx, zslice in enumerate(self.zslices):
            indices = consistent_indices[slice_idx]

            new_zslice = {
                'index': zslice['index'],
                'rois': zslice['rois'][indices] if zslice['rois'] is not None else None,
                'coords': zslice['coords'][indices]
                if zslice['coords'] is not None
                else None,
                'params': zslice['params'][indices],
                'crlbs': zslice['crlbs'][indices],
                'loglike': zslice['loglike'][indices],
                'count': len(indices),
            }

            # Recalculate statistics for the consistent ROIs
            if new_zslice['rois'] is not None:
                new_zslice['mean'] = np.nanmean(new_zslice['rois'], axis=0)
                new_zslice['median'] = np.nanmedian(new_zslice['rois'], axis=0)
                new_zslice['std'] = np.nanstd(new_zslice['rois'], axis=0)
            else:
                new_zslice['mean'] = None
                new_zslice['median'] = None
                new_zslice['std'] = None

            new_data['zslices'].append(new_zslice)

        return PSFdata(new_data)

    def save_hdf(self, filename: str):
        '''
        Save PSF data to HDF5 file.
        '''
        with h5py.File(filename, 'w') as hdf:
            # Save scalar attributes
            for key in ['pixel_size', 'roi_size', 'upsample', 'z_step']:
                hdf.attrs[key] = self._data[key]

            hdf.attrs['shape'] = json.dumps(self.shape)
            hdf.attrs['stack'] = self.path
            hdf.attrs['fit_method'] = self.fitting_method
            hdf.attrs['zero_plane'] = self.zero_plane
            hdf.attrs['roi_info'] = json.dumps(self.roi_info)
            hdf.attrs['params_headers'] = json.dumps(self.headers)

            # Save zslices data
            zslices_group = hdf.create_group('zslices')
            for i, zslice in enumerate(self.zslices):
                slice_group = zslices_group.create_group(f'slice_{i}')
                for key, value in zslice.items():
                    if isinstance(value, np.ndarray):
                        slice_group.create_dataset(key, data=value)
                    elif value is None:
                        slice_group.attrs[key] = 'None'
                    else:
                        slice_group.attrs[key] = value

        print(f'PSF data saved to {filename}')

    @staticmethod
    def load_hdf(filename: str):
        '''
        Load PSF data from HDF5 file.
        '''
        if not os.path.exists(filename):
            raise FileNotFoundError(f'File not found: {filename}')

        with h5py.File(filename, 'r') as hdf:
            data = {
                'zslices': [],
            }

            for key in [
                'pixel_size',
                'z_step',
                'roi_size',
                'upsample',
                'stack',
                'fit_method',
                'zero_plane',
            ]:
                data[key] = hdf.attrs.get(key, None)

            for key in ['shape', 'roi_info', 'params_headers']:
                data[key] = json.loads(hdf.attrs.get(key, '[]'))

            zslices_group = hdf['zslices']
            # Sort slice names numerically
            slice_names = sorted(
                zslices_group.keys(), key=lambda x: int(x.split('_')[1])
            )

            for slice_name in slice_names:
                slice_group = zslices_group[slice_name]
                zslice_data = {}
                for key in slice_group:
                    zslice_data[key] = slice_group[key][()]
                for key in slice_group.attrs:
                    if slice_group.attrs[key] == 'None':
                        zslice_data[key] = None
                    else:
                        zslice_data[key] = slice_group.attrs[key]
                data['zslices'].append(zslice_data)

            # Validate required attributes
            required_attrs = ['pixel_size', 'roi_size', 'upsample', 'shape', 'stack']
            missing_attrs = [attr for attr in required_attrs if attr not in data]
            if missing_attrs:
                raise ValueError(f'Missing required attributes: {missing_attrs}')

        return PSFdata(data)


@nb.njit(cache=True)
def get_roi_list(image: np.ndarray, points: np.ndarray, roi_size=7):
    '''
    Gets the roi list of specific size around the supplied (x, y) points

    Parameters
    ----------
    image : np.ndarray
        The single channel image
    points : np.ndarray
        The points list of preliminary detection
    roi_size : int, optional
        roi size, by default 7

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        roi_list array of shape (nRoi, roi_size**2),
        coord_list of roi top left corner
    '''
    if len(points) < 1:
        return None

    assert len(image.shape) == 2, 'image should be a 2D ndarray!'

    roi_list = np.zeros((points.shape[0], roi_size, roi_size), np.float32)
    coord_list = np.zeros_like(points)
    mask = np.zeros(points.shape[0])

    half_size = roi_size // 2
    y_max, x_max = image.shape

    for r in nb.prange(points.shape[0]):
        x, y = points[r, :]

        x_start = int(x - half_size)
        y_start = int(y - half_size)
        x_end = x_start + roi_size
        y_end = y_start + roi_size

        # Ensure ROI is within image bounds
        # if x_start < 0:
        #     x_start = 0
        #     x_end = roi_size
        # elif x_end > x_max:
        #     x_end = x_max
        #     x_start = x_end - roi_size

        # if y_start < 0:
        #     y_start = 0
        #     y_end = roi_size
        # elif y_end > y_max:
        #     y_end = y_max
        #     y_start = y_end - roi_size

        if x_start < 0 or x_end > x_max or y_start < 0 or y_end > y_max:
            coord_list[r, :] = [np.nan, np.nan]
            roi_list[r] = np.nan
        else:
            coord_list[r, :] = [x_start, y_start]
            roi_list[r] = image[y_start:y_end, x_start:x_end]
            mask[r] = 1

    return roi_list, coord_list, mask


def find_best_z(point_params: np.ndarray, z0_criteria: str, headers: list[str]):
    '''
    Find the best z0 index based on the given criteria.
    '''
    z0_criteria = z0_criteria.lower().replace(' ', '_')

    def get_column(header: str):
        return point_params[:, headers.index(header)]

    if z0_criteria == 'intensity':
        intensities = get_column('intensity')
        return np.argmax(intensities)
    elif z0_criteria == 'min_sigma' and 'sigmax' in headers:
        sigmax = get_column('sigmax')
        return np.argmin(sigmax)
    elif z0_criteria == 'min_sum_sigma' and 'sigmax' in headers and 'sigmay' in headers:
        sigmax = get_column('sigmax')
        sigmay = get_column('sigmay')
        return np.argmin(sigmax + sigmay)
    else:
        argmins = []
        if 'sigmax' in headers and 'sigmay' in headers:
            sigmax = get_column('sigmax')
            sigmay = get_column('sigmay')
            intensities = get_column('intensity')
            argmins.append(np.argmin(sigmax + sigmay))
            argmins.append(np.argmax(intensities))
        else:
            argmins.append(np.argmax(get_column('intensity')))
            if 'sigmax' in headers:
                argmins.append(np.argmin(get_column('sigmax')))

        # return average index
        return int(np.mean(argmins))


def get_psf_rois(
    stack_handler: Union[TiffSeqHandler, ZarrImageSequence],
    frame_list,
    params,
    crlbs,
    loglike,
    fit_method: int,
    pixel_size: float = 114.17,
    z_step: float = 10,
    roi_size: int = 13,
    upsample: int = 1,
    roi_info: Optional[tuple] = None,
    find_z0: bool = False,
    z0_criteria: str = 'all',
    channel: int = 0,
) -> 'PSFdata':
    '''
    Get PSF ROIs for the given frame list.
    '''
    if upsample > 1:
        roi_size = roi_size * upsample | 1
        # This uses the bitwise OR operator to ensure the least significant bit is
        # set to 1, making the number odd.
    else:
        upsample = 1

    headers = PARAMETER_HEADERS[fit_method]
    frame_ids = np.cumsum(np.bincount(np.array(frame_list, np.int64) - 1))
    frames = np.arange(0, max(frame_list), 1, dtype=np.uint32)

    if find_z0:
        frame_params = np.zeros((len(frames), len(headers)))
        for i in range(len(frames)):
            start_idx = frame_ids[i - 1] if i > 0 else 0
            end_idx = frame_ids[i]
            frame_params[i] = np.nanmean(params[slice(start_idx, end_idx)], axis=0)

        z0 = find_best_z(frame_params, z0_criteria, headers)
    else:
        # z0 = max(frame_list) // 2
        z0 = None

    def process_frame(index: int) -> dict[str, Union[int, np.ndarray, list, float]]:
        image = stack_handler.getSlice(index, channel, 0)

        if roi_info is not None:
            origin, dim = roi_info
            slice_y = slice(int(origin[1]), int(origin[1] + dim[1]))
            slice_x = slice(int(origin[0]), int(origin[0] + dim[0]))
            image = image[slice_y, slice_x]
        else:
            origin = (0, 0)
            dim = (image.shape[1], image.shape[0])

        if upsample > 1:
            image = cv2.resize(
                image,
                (0, 0),
                fx=upsample,
                fy=upsample,
                interpolation=cv2.INTER_NEAREST,
            )

        param = params[
            slice(frame_ids[index - 1] if index > 0 else 0, frame_ids[index])
        ]
        crlb = crlbs[slice(frame_ids[index - 1] if index > 0 else 0, frame_ids[index])]
        logl = loglike[
            slice(frame_ids[index - 1] if index > 0 else 0, frame_ids[index])
        ]

        if z0 is None:
            points = upsample * (param[..., :2] - origin) + (upsample - 1) / 2
        else:
            points = (
                upsample
                * (
                    params[
                        slice(frame_ids[z0 - 1] if z0 > 0 else 0, frame_ids[z0]),
                        :2,
                    ]
                    - origin
                )
                + (upsample - 1) / 2
            )

        res = get_roi_list(image, points, roi_size)

        if res:
            rois, coords, mask = res
            count = rois.shape[0]

            mask = mask.astype(bool)

            param = param[mask]
            crlb = crlb[mask]
            logl = logl[mask]
        else:
            rois, coords = None, None
            count = 0

        return {
            'index': index,
            'rois': rois,
            'count': count,
            'mean': np.nanmean(rois, axis=0) if count else None,
            'median': np.nanmedian(rois, axis=0) if count else None,
            'std': np.nanstd(rois, axis=0) if count else None,
            'coords': coords,
            'params': param,
            'crlbs': crlb,
            'loglike': logl,
        }

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_frame, i) for i in frames]
        results = [None] * len(frames)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc='Extrating PSF ROIs'
        ):
            idx = futures.index(future)
            results[idx] = future.result()

    return PSFdata(
        {
            'zslices': results,
            'shape': stack_handler.shapeTCZYX(),
            'pixel_size': pixel_size,
            'roi_size': roi_size,
            'upsample': upsample,
            'stack': stack_handler.path,
            'roi_info': roi_info,
            'fit_method': fit_method,
            'zero_plane': z0 if z0 else max(frame_list) // 2,
            'z_step': z_step,
            'params_headers': headers,
        }
    )
