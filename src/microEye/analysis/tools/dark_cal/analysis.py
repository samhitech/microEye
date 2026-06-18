import logging
import os

import numpy as np

from microEye.analysis.tools.dark_cal.constants import (
    FILE_NAMES,
    HISTOGRAM_DATA_TYPES,
    DataTypes,
)

logger = logging.getLogger(__name__)


def has_plot_data(data: dict) -> bool:
    if not isinstance(data, dict):
        return False

    if (
        DataTypes.EXPOSURE in data
        and DataTypes.MEAN in data
        and DataTypes.VARIANCE in data
    ):
        return True

    return all(data_type in data for data_type in HISTOGRAM_DATA_TYPES)


def infer_results_mode(data: dict):
    if not isinstance(data, dict):
        return None

    if (
        DataTypes.EXPOSURE in data
        and DataTypes.MEAN in data
        and DataTypes.VARIANCE in data
    ):
        return 'Standard'

    if all(data_type in data for data_type in HISTOGRAM_DATA_TYPES):
        return 'Histograms'

    return None


def resolve_results_mode(directories: dict) -> str:
    modes = set()

    for data in directories.values():
        mode = infer_results_mode(data)
        if mode is not None:
            modes.add(mode)

    if not modes:
        return ''

    if len(modes) > 1:
        raise ValueError('Found mixed result modes. Re-run analysis in a single mode.')

    return modes.pop()


def get_centered_data(data: np.ndarray) -> tuple[np.ndarray, float, float]:
    finite_data = data[np.isfinite(data)]
    median = np.nanmedian(finite_data)
    mad = 1.4826 * np.nanmedian(np.abs(finite_data - median))
    return finite_data - median, median, mad


def get_robust_hist_range(
    centered_data: np.ndarray, median: float, mad: float
) -> tuple[float, float]:
    finite_data = centered_data[np.isfinite(centered_data)]

    if finite_data.size == 0:
        return -1.0, 1.0

    if np.isfinite(mad) and mad > 0:
        # Robust, symmetric range around the median.
        span = 6.0 * mad
        return -span, span

    # Fallback for near-constant or degenerate data.
    return (
        np.percentile(finite_data, 0.5),
        np.percentile(finite_data, 99.5),
    )


def get_maps(data: dict[DataTypes, np.ndarray]):
    exposure_times = data[DataTypes.EXPOSURE] / 1000.0  # convert ms to s

    x = np.vstack([np.ones_like(exposure_times), exposure_times]).T  # [n_exp, 2]
    pinv = np.linalg.pinv(x)  # reuse for every pixel

    flat_means = data[DataTypes.MEAN].reshape(len(exposure_times), -1)
    flat_vars = data[DataTypes.VARIANCE].reshape(len(exposure_times), -1)

    coeff_mean = (pinv @ flat_means).reshape(2, *data[DataTypes.MEAN].shape[1:])
    coeff_var = (pinv @ flat_vars).reshape(2, *data[DataTypes.VARIANCE].shape[1:])

    return (
        coeff_mean[0],  # intercept map
        coeff_mean[1],  # slope map
        coeff_var[0],  # variance intercept map
        coeff_var[1],  # variance slope map
    )


def get_histogram(data: np.ndarray, bins=600):
    centered_data, median, mad = get_centered_data(data)
    hist_range = get_robust_hist_range(centered_data, median, mad)

    hist, bin_edges = np.histogram(centered_data.flatten(), bins=bins, range=hist_range)

    mode = bin_edges[np.argmax(hist)] + (bin_edges[1] - bin_edges[0]) / 2.0

    return {
        'hist': hist,
        'bin_edges': bin_edges - mode,
        'median': median - mode,
        'mad': mad,
    }


def perform_analysis(directories, mode, progress_callback, event):
    count = len(directories)

    for i, directory in enumerate(directories):
        logger.info(f'Analyzing directory: {directory}')

        paths = {
            data_type: os.path.join(directory, FILE_NAMES[data_type])
            for data_type in DataTypes
            if data_type in FILE_NAMES
        }

        if any(
            not os.path.exists(path)
            for data_type, path in paths.items()
            if data_type != DataTypes.TEMPERATURE
        ):
            continue

        mean = np.load(paths[DataTypes.MEAN], mmap_mode='r').squeeze()
        logger.info(f'Mean shape: {mean.shape}')
        variance = np.load(paths[DataTypes.VARIANCE], mmap_mode='r').squeeze()
        logger.info(f'Variance shape: {variance.shape}')
        exposure_times = np.load(paths[DataTypes.EXPOSURE], mmap_mode='r').squeeze()
        logger.info(f'Exposure times shape: {exposure_times.shape}')

        if os.path.exists(paths[DataTypes.TEMPERATURE]):
            temps = np.load(paths[DataTypes.TEMPERATURE], mmap_mode='r').squeeze()
            logger.info(f'Temperatures shape: {temps.shape}')
        else:
            temps = None
            logger.info('No temperature data found.')

        if mode == 'Histograms':
            baseline, darkcurrent, dark_var, thermal_var = get_maps(
                {
                    DataTypes.MEAN: mean,
                    DataTypes.VARIANCE: variance,
                    DataTypes.EXPOSURE: exposure_times,
                }
            )

            directories[directory] = {
                DataTypes.BASELINE: get_histogram(baseline),
                DataTypes.DARK_CURRENT: get_histogram(darkcurrent),
                DataTypes.DARK_VARIANCE: get_histogram(dark_var),
                DataTypes.THERMAL_VARIANCE: get_histogram(thermal_var),
                DataTypes.DARK_NOISE: get_histogram(np.sqrt(dark_var)),
                DataTypes.THERMAL_NOISE: get_histogram(np.sqrt(thermal_var)),

            }
        else:
            directories[directory] = {
                DataTypes.EXPOSURE: exposure_times,
                DataTypes.MEAN: np.nanmean(mean, axis=(1, 2)),
                DataTypes.VARIANCE: np.nanmean(variance, axis=(1, 2)),
                DataTypes.TEMPERATURE: np.nanmean(temps, axis=1)
                if temps is not None
                else None,
            }

        progress_callback.emit((i + 1) / count * 100)
