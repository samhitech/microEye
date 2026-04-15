import logging

import numpy as np
import scipy.constants as sc
from sklearn.linear_model import LinearRegression

from microEye.analysis.tools.photon_cal.models import CalibrationDatasetMeta
from microEye.analysis.tools.sphere_cal import (
    SphereCalibrationTool,
    _find_best_linear_segment,
)

logger = logging.getLogger(__name__)


def photon_flux_from_irradiance(E_W_per_cm2, wavelength_nm):
    lam = wavelength_nm * 1e-9  # m
    E_photon = sc.Planck * sc.speed_of_light / lam  # J/photon
    return E_W_per_cm2 / E_photon


def photons_per_pixel_from_irradiance(flux, pixel_size_um, exposure_time_s):
    pixel_area_cm2 = (pixel_size_um * 1e-4) ** 2  # cm^2
    return flux * pixel_area_cm2 * exposure_time_s


def fit_ptc_model(signal_path, dark_path, **kwargs):
    signal = np.load(signal_path, mmap_mode='r').astype(np.float32)
    dark = np.load(dark_path, mmap_mode='r').astype(np.float32)

    if signal.shape[0] != 201:
        signal = signal[:201, ...]

    if dark.shape[0] != 201:
        dark = dark[:201, ...]

    s1, s2 = signal[:, 0, ...], signal[:, 1, ...]
    d1, d2 = dark[:, 0, ...], dark[:, 1, ...]

    mean_dn = (0.5 * (s1 + s2) - 0.5 * (d1 + d2)).mean(axis=(-2, -1), dtype=np.float64)
    var_total = 0.5 * np.var(s1 - s2, axis=(-2, -1), ddof=1, dtype=np.float64)
    var_dark = 0.5 * np.var(d1 - d2, axis=(-2, -1), ddof=1, dtype=np.float64)
    var_shot = var_total - var_dark

    variance_source = str(kwargs.get('variance_source', 'shot')).lower()
    if variance_source not in ('shot', 'total'):
        logger.warning(
            f'Unknown variance source "{variance_source}", defaulting to "shot".'
        )
        variance_source = 'shot'

    variance_for_fit = var_shot if variance_source == 'shot' else var_total

    valid = (
        (mean_dn > 0)
        & np.isfinite(mean_dn)
        & (variance_for_fit > 0)
        & np.isfinite(variance_for_fit)
        & (mean_dn < kwargs.get('linearity_limit_dn', 4096 * 0.9))
    )

    i0, i1, _model, r2 = _find_best_linear_segment(
        mean_dn[valid],
        variance_for_fit[valid],
        min_points=100,
        r2_threshold=kwargs.get('r2_threshold', 0.999),
    )
    logger.info(
        f'Gain ({variance_source}) | Best linear segment: '
        f'indices {i0} to {i1}, R²={r2:.6f}'
    )

    model = LinearRegression(fit_intercept=True)
    model.fit(
        mean_dn[valid][i0:i1].reshape(-1, 1),
        variance_for_fit[valid][i0:i1],
    )

    return {
        'mean_dn': mean_dn,
        'mean_e': mean_dn * (1.0 / model.coef_[0]),
        'var_total_dn2': var_total,
        'var_dark_dn2': var_dark,
        'var_shot_dn2': var_shot,
        'fit_variance_source': variance_source,
        'fit_variance_dn2': variance_for_fit,
        'gain_e_per_dn': float(1.0 / model.coef_[0]),
        'read_noise_dn': float(model.intercept_),
        'read_noise_e': float(
            np.sqrt(max(model.intercept_, 0)) * (1.0 / model.coef_[0])
        ),
    }


def perform_analysis(
    datasets: dict[str, CalibrationDatasetMeta],
    cal_tool: SphereCalibrationTool,
    progress_callback,
    event,
    **kwargs,
):
    count = len(datasets)
    returned_data = {}
    exposure_times_s = np.arange(0.001, 0.201, 0.001)

    r2_threshold = kwargs.get('r2_threshold', 0.999)

    for i, (name, dataset) in enumerate(datasets.items()):
        logger.info(f'Analyzing dataset: {name}')
        returned_data[name] = fit_ptc_model(
            dataset.signal_path, dataset.dark_path, **kwargs
        )

        irradiance = (
            cal_tool.port_power_to_camera_irradiance(
                np.array([[dataset.port_power_mW]])
            )[0]
            if dataset.port_power_mW is not None
            else cal_tool.power_to_camera_irradiance(
                np.array([[dataset.laser_power_mW]])
            )[0]
        )

        flux = photon_flux_from_irradiance(irradiance * 1e-3, dataset.wavelength_nm)
        returned_data[name]['photons_per_pixel'] = photons_per_pixel_from_irradiance(
            flux, dataset.pixel_size_um, exposure_times_s
        )

        params = returned_data[name]
        photons = params['photons_per_pixel']
        mean_e = params['mean_e'][1:]
        valid = (
            (mean_e > 0)
            & np.isfinite(mean_e)
            & (params['mean_dn'][1:] < kwargs.get('linearity_limit_dn', 4096 * 0.8))
        )

        i0, i1, _model, r2 = _find_best_linear_segment(
            photons[valid], mean_e[valid], min_points=100, r2_threshold=r2_threshold
        )
        logger.info(
            f'Responsivity | Best linear segment: indices {i0} to {i1}, R²={r2:.6f}'
        )

        model = LinearRegression(fit_intercept=True)
        model.fit(photons[valid][i0:i1].reshape(-1, 1), mean_e[valid][i0:i1])
        params['qe'] = float(model.coef_[0])
        params['qe_intercept'] = float(model.intercept_)

        model.fit(photons[valid].reshape(-1, 1), params['mean_dn'][1:][valid])
        params['responsivity'] = float(model.coef_[0])
        params['responsivity_intercept'] = float(model.intercept_)

        progress_callback.emit((i + 1) / count * 100)

    return exposure_times_s, returned_data
