
import cv2
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import emcee

from scipy.optimize import curve_fit, minimize, least_squares
from scipy.signal import find_peaks

from ..Rendering import radial_cordinate


def gaussian_2D_fit(
        image: np.ndarray, points: np.ndarray, roi_size=7):
    '''Sub-pixel Simple 2D Gaussian fit
    '''
    if len(points) < 1:
        return None

    return gaussian_2D_fit_numba_worker(
        image, points, roi_size)


def gaussian_2D_fit_numba_worker(
        image: np.ndarray, points: np.ndarray, roi_size=7,
        upsampling=1, defaultSigma=1, MLE=True):

    sub_fit = np.zeros((points.shape[0], 9), dtype=np.float64)
    roi_center = roi_size/2

    with numba.objmode(X='int32[:,:]', Y='int32[:,:]'):
        Y, X = np.indices((roi_size * upsampling,)*2)

    for r in numba.prange(points.shape[0]):
        x, y = points[r, :]

        idx = int(x - roi_center)
        idy = int(y - roi_center)
        if idx < 0:
            idx = 0
        if idy < 0:
            idy = 0
        if idx + roi_size > image.shape[1]:
            idx = image.shape[1] - roi_size
        if idy + roi_size > image.shape[0]:
            idy = image.shape[0] - roi_size
        roi = image[idy:idy+roi_size, idx:idx+roi_size]

        # col = roi[:, int(x-idx)]
        # width_y = np.sqrt(
        #     np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
        # row = roi[int(y-idy), :]
        # width_x = np.sqrt(
        #     np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())

        params = (
            (x-idx) * upsampling,
            (y-idy) * upsampling,
            defaultSigma * upsampling,
            defaultSigma * upsampling,
            np.sum(roi),
            roi.min())

        bnds = (
            (0, roi_size * upsampling),
            (0, roi_size * upsampling),
            (0.25 * upsampling, 20 * upsampling),
            (0.25 * upsampling, 20 * upsampling),
            (roi.min(), 1.5 * params[4]),
            (0, 0.5 * roi.max())
        )

        with numba.objmode():
            data = np.repeat(
                roi, upsampling, axis=0).repeat(
                    upsampling, axis=1)
            if MLE:
                res = minimize(
                    log_probability,
                    x0=params,
                    args=(data, X, Y, upsampling),
                    # bounds=bnds,
                    options={'disp': False})
            else:
                res = least_squares(
                    residuals,
                    x0=params,
                    args=(data, X, Y,),
                    bounds=(0, np.inf)
                )

            if res.success:
                x = idx + res.x[0] / upsampling
                y = idy + res.x[1] / upsampling
                sub_fit[r, :2] = [x, y]
                sub_fit[r, 3] = res.x[4]
                sub_fit[r, 5] = res.x[2] / res.x[3]
                sub_fit[r, 6] = res.x[2] / upsampling
                sub_fit[r, 7] = res.x[3] / upsampling
                sub_fit[r, 8] = res.x[5]
            else:
                sub_fit[r, 3] = -1

    return sub_fit[sub_fit[:, 3] >= 0, :]


@numba.njit
def gauss_2D_explicit_numba(
        xc, yc, sigma_x, sigma_y, rho: float, flux: float,
        offset: float, X: np.ndarray, Y: np.ndarray):
    '''An explicity written 2D Bivariate Gaussian (numba)

    Parameters
    ----------
    vc : list[float] | tuple[float]
        the center point vector (X, Y) (origin is top-left corner)
    sigma : list[float] | tuple[float]
        the standard deviation (sigma_x, sigma_y) > 0
    rho : float
        the correlation between X and Y
    flux : float
        the amplitude
    offset : float
        the offset
    shape : list[int] | tuple(int)
        dimensions of generated 2D array

    Returns
    -------
    numpy.ndarray
        the 2D Bivariate Gaussian array
    '''
    tmp = rho * sigma_x * sigma_y
    covar = np.array(
        [[sigma_x, tmp], [tmp, sigma_y]], dtype=np.float64)

    cov_inv = np.linalg.inv(covar)  # inverse of covariance matrix
    cov_det = np.linalg.det(covar)  # determinant of covariance matrix

    coe = flux / ((2 * np.pi)**2 * cov_det)**0.5
    return (coe * np.e ** (-0.5 * (cov_inv[0, 0]*(X-xc)**2 + (cov_inv[0, 1]
            + cov_inv[1, 0])*(X-xc)*(Y-yc) + cov_inv[1, 1]*(Y-yc)**2))
            ) + offset


def gauss_2D_simple(
        xc, yc, sigma_x, sigma_y, flux: float,
        offset: float, X: np.ndarray, Y: np.ndarray):
    '''A 2D Gaussian (numba) without covariance.

    Parameters
    ----------
    vc : list[float] | tuple[float]
        the center point vector (X, Y) (origin is top-left corner)
    sigma : list[float] | tuple[float]
        the standard deviation (sigma_x, sigma_y) > 0
    flux : float
        the amplitude
    offset : float
        the offset
    shape : list[int] | tuple(int)
        dimensions of generated 2D array

    Returns
    -------
    numpy.ndarray
        the 2D Bivariate Gaussian array
    '''
    return gauss_2D_explicit_numba(
        vc=np.array([xc, yc]),
        sigma=np.array([sigma_x, sigma_y]),
        rho=0,
        flux=flux,
        offset=offset,
        X=X,
        Y=Y)


def model(xc, yc, sigma_x, sigma_y, flux, offset, X, Y):
    y_gauss = gauss_1d(Y[:, 0], yc, sigma_y)
    x_gauss = gauss_1d(X[0, :], xc, sigma_x)

    return flux * np.einsum('i,j->ij', y_gauss, x_gauss) + offset


def jac(params, *args):
    xc, yc, sigma_x, sigma_y, flux, offset = params
    data, X, Y = args[0], args[1], args[2]

    mod = model(
        xc, yc,
        sigma_x, sigma_y,
        flux=1,
        offset=0,
        X=X,
        Y=Y)

    dx = (-(X*X - X*xc) * (flux / sigma_x**2)) * mod
    dy = (-(Y*Y - Y*yc) * (flux / sigma_y**2)) * mod
    dsig_x = ((- flux / sigma_x) +
              (flux / sigma_x ** 3) * ((X - xc)**2)) * mod
    dsig_y = ((- flux / sigma_y) +
              (flux / sigma_y ** 3) * ((Y - yc)**2)) * mod

    d = [
        np.sum(dx),
        np.sum(dy),
        np.sum(dsig_x),
        np.sum(dsig_y),
        np.sum(mod),
        np.sum(1),
    ]

    return d


@numba.njit()
def gauss_1d(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * \
           np.exp(-0.5 * (x - mu)**2 / sigma**2)


def log_likelihood(
        params, data: np.ndarray,
        X: np.ndarray, Y: np.ndarray):
    xc, yc, sigma_x, sigma_y, flux, offset = params

    mod = model(
        xc, yc,
        sigma_x, sigma_y,
        flux=flux,
        offset=offset,
        X=X,
        Y=Y)

    err = np.ravel(data)
    err[err == 0] = np.mean(err)

    likelihood = -0.5 * (data - mod).ravel()**2 / err

    return - np.sum(likelihood)


def residuals(
        params, data: np.ndarray,
        X: np.ndarray, Y: np.ndarray):
    xc, yc, sigma_x, sigma_y, flux, offset = params

    mod = model(
        xc, yc,
        sigma_x, sigma_y,
        flux=flux,
        offset=offset,
        X=X,
        Y=Y)

    res = np.abs(data - mod)

    return res.sum()


def log_prior(params, Upsampling):
    xc, yc, sigma_x, sigma_y, flux, offset = params
    if 0 <= (xc / Upsampling) < 11 and \
       0 <= (yc / Upsampling) < 11 and \
       0.25 < (sigma_x / Upsampling) < 20 and \
       0.25 < (sigma_y / Upsampling) < 20 and \
       flux / Upsampling > 0.0 and \
       0.0 <= offset < 1000:
        return 0.0   # likelihood 1
    return np.inf  # likelihood 0


def log_probability(
        params, data: np.ndarray,
        X: np.ndarray, Y: np.ndarray, Upsampling):
    lp = log_prior(params, Upsampling)
    if not np.isfinite(lp):
        return 1e10
    return lp + log_likelihood(params, data, X, Y)


def emcee_walker(
        params, data: np.ndarray,
        X: np.ndarray, Y: np.ndarray, Upsampling):
    ndim = len(params)   # 5 free params
    nwalkers = 2 * ndim  # at least 2 x ndim

    pos = np.random.normal(0, 1e-4, (nwalkers, ndim)) + params

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        log_probability,
        args=(data, X, Y, Upsampling)
    )

    sampler.run_mcmc(pos, 1000, progress=True)

    flat_samples = sampler.get_chain(flat=True)

    mcmc = np.percentile(flat_samples, [16, 50, 84], axis=0)
    q = np.diff(mcmc, axis=0)

    # fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()
    # for i in range(ndim):
    #     ax = axes[i]
    #     ax.plot(samples[:, :, i], 'k', alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.yaxis.set_label_coords(-0.1, 0.5)

    return mcmc[1, :]
