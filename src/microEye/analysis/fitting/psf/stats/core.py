from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Union

import numpy as np
from scipy import stats

from microEye.analysis.fitting.results import PARAMETER_HEADERS


class ConfidenceMethod(Enum):
    NONE = 'None'
    MIN_MAX = 'min_max'
    T_DIST = 't_dist'
    MEDIAN_STD = 'median_std'
    PERCENTILE = 'percentile'
    BOOTSTRAP = 'bootstrap'


@dataclass
class StatConfig:
    '''Configuration for a statistic calculation'''

    name: str
    required_params: list[str]  # Parameters required from header
    calculator: Callable  # Function to calculate the statistic
    multi_output: bool = False  # Whether the stat returns multiple values


class StatsCalculator:
    def __init__(self, zero_plane: int, z_step: float, fitting_method: str):
        self.zero_plane = zero_plane
        self.z_step = z_step
        self.fitting_method = fitting_method
        self.header = PARAMETER_HEADERS[fitting_method]
        self._register_stats()

    def _register_stats(self):
        '''Register all available statistics calculations'''
        self.available_stats: dict[str, StatConfig] = {
            'Counts': StatConfig(
                name='Counts', required_params=[], calculator=lambda rois, _: len(rois)
            ),
            'Sigma': StatConfig(
                name='Sigma',
                required_params=['sigmax', 'sigmay'],
                calculator=self._calc_sigma,
                multi_output=True,
            ),
            'Sigma (sum)': StatConfig(
                name='Sigma (sum)',
                required_params=['sigmax', 'sigmay'],
                calculator=lambda data_x, data_y: data_x + data_y,
            ),
            'Sigma (diff)': StatConfig(
                name='Sigma (diff)',
                required_params=['sigmax', 'sigmay'],
                calculator=lambda data_x, data_y: data_x - data_y,
            ),
            'Sigma (abs(diff))': StatConfig(
                name='Sigma (abs(diff))',
                required_params=['sigmax', 'sigmay'],
                calculator=lambda data_x, data_y: np.abs(data_x - data_y),
            ),
            'Sigma (x/y)': StatConfig(
                name='Sigma (x/y)',
                required_params=['sigmax', 'sigmay'],
                calculator=lambda data_x, data_y: np.divide(
                    data_x, data_y, out=np.zeros_like(data_x), where=data_y != 0
                ),
            ),
            'Sigma² (diff)': StatConfig(
                name='Sigma² (diff)',
                required_params=['sigmax', 'sigmay'],
                calculator=lambda data_x, data_y: np.square(data_x) - np.square(data_y),
            ),
            'Intensity': StatConfig(
                name='Intensity',
                required_params=['intensity'],
                calculator=lambda data: data,
            ),
            'Background': StatConfig(
                name='Background',
                required_params=['background'],
                calculator=lambda data: data,
            ),
        }

    def _calc_sigma(
        self, data_x: Optional[np.ndarray], data_y: Optional[np.ndarray]
    ) -> list[np.ndarray]:
        '''Calculate sigma statistics for x and y dimensions'''
        results = []
        for data in [data_x, data_y]:
            if data is not None:
                results.append(data)
        return results

    def _get_param_data(self, z_slice: dict, param_name: str) -> Optional[np.ndarray]:
        '''Extract parameter data from z_slice if parameter exists'''
        if param_name in self.header:
            return z_slice['params'][:, self.header.index(param_name)]
        return None

    def _calculate_confidence_interval(
        self,
        data: np.ndarray,
        method: ConfidenceMethod,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        '''Calculate confidence intervals using various methods'''
        if len(data) == 0 or method == ConfidenceMethod.NONE:
            return np.nan, np.nan

        if method == ConfidenceMethod.MIN_MAX:
            return np.nanmin(data), np.nanmax(data)

        elif method == ConfidenceMethod.T_DIST:
            if len(data) < 2:
                return np.nan, np.nan
            mean = np.nanmean(data)
            std_err = np.nanstd(data, ddof=1) / np.sqrt(len(data))
            t_value = stats.t.ppf((1 + confidence_level) / 2, len(data) - 1)
            margin = t_value * std_err
            return mean - margin, mean + margin

        elif method == ConfidenceMethod.MEDIAN_STD:
            median = np.nanmedian(data)
            std = np.nanstd(data)
            return median - std, median + std

        elif method == ConfidenceMethod.PERCENTILE:
            lower = (1 - confidence_level) / 2 * 100
            upper = (1 + confidence_level) / 2 * 100
            return np.nanpercentile(data, [lower, upper])

        elif method == ConfidenceMethod.BOOTSTRAP:
            if len(data) < 2:
                return np.nan, np.nan
            bootstrap_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.nanmean(sample))
            return np.percentile(
                bootstrap_means,
                [(1 - confidence_level) / 2 * 100, (1 + confidence_level) / 2 * 100],
            )

    def _calculate_stats(
        self,
        z_slice: dict[str, Union[int, np.ndarray, list, float]],
        stat_config: StatConfig,
        confidence_method: ConfidenceMethod = ConfidenceMethod.MIN_MAX,
        confidence_level: float = 0.95,
    ) -> tuple[list[float], list[float], list[float]]:
        '''Calculate statistics for a single z-slice'''
        if z_slice['rois'] is None:
            return self._get_nan_result(stat_config.multi_output)

        valid_rois = z_slice['rois'][~np.isnan(z_slice['rois']).any(axis=(1, 2))]

        if len(valid_rois) == 0 and stat_config.name != 'Counts':
            return self._get_nan_result(stat_config.multi_output)

        if stat_config.name == 'Counts':
            return [len(valid_rois)], [len(valid_rois)], [len(valid_rois)]

        # Extract required parameters
        param_data = [
            self._get_param_data(z_slice, param)
            for param in stat_config.required_params
        ]

        if (
            any(data is None for data in param_data)
            and len(stat_config.required_params) > 0
        ):
            return self._get_nan_result(stat_config.multi_output)

        # Calculate statistics
        result = stat_config.calculator(*param_data)
        if stat_config.multi_output:
            means = [np.nanmean(r) for r in result]
            conf_intervals = [
                self._calculate_confidence_interval(
                    r, confidence_method, confidence_level
                )
                for r in result
            ]
            return (
                means,
                [ci[0] for ci in conf_intervals],
                [ci[1] for ci in conf_intervals],
            )

        mean = np.nanmean(result)
        lower, upper = self._calculate_confidence_interval(
            result, confidence_method, confidence_level
        )
        return [mean], [lower], [upper]

    def _get_nan_result(
        self, multi_output: bool
    ) -> tuple[list[float], list[float], list[float]]:
        '''Return NaN results based on whether multiple outputs are expected'''
        if multi_output:
            return [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]
        return [np.nan], [np.nan], [np.nan]

    def get_stats(
        self,
        zslices: list[dict[str, Union[int, np.ndarray, list, float]]],
        selected_stat: str,
        confidence_method: ConfidenceMethod = ConfidenceMethod.MIN_MAX,
        confidence_level: float = 0.95,
    ):
        '''
        Get the statistic data for all z-slices.

        Parameters
        ----------
        selected_stat : str
            The selected statistic to return

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Z-indices, statistic data, minimum data, maximum
        '''
        if selected_stat not in self.available_stats:
            raise ValueError(f'Unknown statistic: {selected_stat}')

        z_indices = (np.arange(len(zslices)) - self.zero_plane) * self.z_step
        stat_config = self.available_stats[selected_stat]

        param_stat = []
        param_lower = []
        param_upper = []

        # Initialize lists for multi-output statistics
        if stat_config.multi_output:
            param_stat = [[] for _ in range(len(stat_config.required_params))]
            param_lower = [[] for _ in range(len(stat_config.required_params))]
            param_upper = [[] for _ in range(len(stat_config.required_params))]

        for z_slice in zslices:
            stats, mins, maxs = self._calculate_stats(
                z_slice, stat_config, confidence_method, confidence_level
            )

            if stat_config.multi_output:
                for i, (stat, min_val, max_val) in enumerate(zip(stats, mins, maxs)):
                    param_stat[i].append(stat)
                    param_lower[i].append(min_val)
                    param_upper[i].append(max_val)
            else:
                param_stat.extend(stats)
                param_lower.extend(mins)
                param_upper.extend(maxs)

        return (
            z_indices,
            np.array(param_stat),
            np.array(param_lower),
            np.array(param_upper),
        )

    def add_statistic(
        self,
        name: str,
        required_params: list[str],
        calculator: Callable,
        multi_output: bool = False,
    ):
        '''
        Add a new statistic calculation method.

        Parameters
        ----------
        name : str
            Name of the new statistic
        required_params : list[str]
            List of parameter names required from the header
        calculator : Callable
            Function to calculate the statistic
        multi_output : bool, optional
            Whether the statistic returns multiple values, by default False
        '''
        self.available_stats[name] = StatConfig(
            name=name,
            required_params=required_params,
            calculator=calculator,
            multi_output=multi_output,
        )
