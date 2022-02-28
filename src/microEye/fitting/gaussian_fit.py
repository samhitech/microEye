
import cv2
import numba
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# @numba.njit(parallel=True)
def gaussian_2D_fit_numba_worker(
        image: np.ndarray, points: np.ndarray, roi_size=7):

    sub_fit = np.zeros((points.shape[0], 5), dtype=np.float64)
    roi_center = roi_size/2

    with numba.objmode(X='int32[:,:]', Y='int32[:,:]'):
        X, Y = np.indices((roi_size,)*2)

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

        col = roi[:, int(x-idx)]
        width_y = np.sqrt(
            np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
        row = roi[int(y-idy), :]
        width_x = np.sqrt(
            np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())

        params = (x-idx, y-idy, width_x, width_y, roi.sum(), roi.min())

        with numba.objmode():
            res = minimize(
                log_probability,
                x0=params,
                args=(roi, X, Y,),)

            if res.success:
                x = idx + res.x[0]
                y = idy + res.x[1]
                sub_fit[r, :2] = [x, y]
                sub_fit[r, 3] = res.x[4]
            else:
                sub_fit[r, 3] = -1

    return sub_fit[sub_fit[:, 3] >= 0, :]


def gauss_2D_explicit(
        vc, sigma, rho: float, flux: float,
        offset: float, shape):
    '''An explicity written 2D Bivariate Gaussian

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
    tmp = rho * sigma[0] * sigma[1]
    covar = np.array(
        [[sigma[0], tmp], [tmp, sigma[1]]], dtype=np.float64)

    cov_inv = np.linalg.inv(covar)  # inverse of covariance matrix
    cov_det = np.linalg.det(covar)  # determinant of covariance matrix

    y_len = np.arange(0, shape[0], dtype=np.int32)
    x_len = np.arange(0, shape[1], dtype=np.int32)

    # mesh grid XY cordinates
    X, Y = np.meshgrid(x_len, y_len)

    coe = flux / ((2 * np.pi)**2 * cov_det)**0.5
    return (coe * np.e ** (-0.5 * (cov_inv[0, 0]*(X-vc[0])**2 + (cov_inv[0, 1]
            + cov_inv[1, 0])*(X-vc[0])*(Y-vc[1]) + cov_inv[1, 1]*(Y-vc[1])**2))
            ) + offset


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

    x, y = np.meshgrid(x_gauss, y_gauss)
    return flux * y * x + offset


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

    err = np.sqrt(data)
    err[err == 0] = np.mean(err[err > 0])

    likelihood = gauss_1d(
        data,
        mod,
        err
    )

    if np.all(likelihood == 0):
        return np.inf

    return -np.sum(np.log(likelihood[np.nonzero(likelihood)]))


def log_prior(params):
    xc, yc, sigma_x, sigma_y, flux, offset = params
    if 0 <= xc < 20 and \
       0 <= yc < 20 and \
       0.01 < sigma_x < 50 and \
       0.01 < sigma_y < 50 and \
       65536 > flux > 0.0 and \
       0.0 <= offset < 100:
        return 0.0   # likelihood 1
    return np.inf  # likelihood 0


def log_probability(
        params, data: np.ndarray,
        X: np.ndarray, Y: np.ndarray):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return 1e8
    return lp + log_likelihood(params, data, X, Y)
