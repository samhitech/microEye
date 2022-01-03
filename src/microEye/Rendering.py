import re
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import numba

from PyQt5.QtCore import *
from .Fitting import radial_cordinate


def explicit2dGauss(mu, sigma, corr, flux, offset, shape):
    '''Generates a 2D Multivariate Gaussian

    Params
    -------
    mu (tuple | list)
        vector holding [yc, xc] the distribution mean
    sigma (tuple | list)
        the standard deviation [y_sigma, x_sigma]
    corr (float)
        the correlation between X and Y
    flux (float)
        the flux
    offset (float)
        the distribution offset
    shape (tuple | list)
        the shape of 2D produced gaussian array

    Returns
    -------
    Z (np.ndarray)
        a 2D Multivariate Gaussian array
    '''
    diag2nd = corr * sigma[0] * sigma[1]
    covar = np.array([[sigma[0], diag2nd], [diag2nd, sigma[1]]])

    cov_inv = np.linalg.inv(covar)  # inverse of covariance matrix
    cov_det = np.linalg.det(covar)  # determinant of covariance matrix
    # Plotting
    y_len = np.arange(0, shape[0])
    x_len = np.arange(0, shape[1])
    X, Y = np.meshgrid(x_len, y_len)
    coe = flux / ((2 * np.pi)**2 * cov_det)**0.5
    return (coe * np.exp(-0.5 * (cov_inv[0, 0]*(X-mu[0])**2 + (cov_inv[0, 1]
            + cov_inv[1, 0])*(X-mu[0])*(Y-mu[1]) + cov_inv[1, 1]*(Y-mu[1])**2))
            ) + offset


class gauss_hist_render:

    def __init__(self, pixelSize=10):
        self._pixel_size = pixelSize
        self._std = 10  # nm
        self._gauss_std = self._std / self._pixel_size
        self._gauss_len = 1 + np.ceil(self._gauss_std * 6)
        self._gauss_shape = [int(self._gauss_len)] * 2
        self._gauss_2d = explicit2dGauss(
            mu=[(self._gauss_len - 1) / 2] * 2,
            sigma=[self._gauss_std] * 2,
            corr=0,
            flux=1,
            offset=0,
            shape=self._gauss_shape)

    def render(self, X_loc, Y_loc, Intensity):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        X_loc (np.ndarray)
            Sub-pixel localized points X coordinates
        Y_loc (np.ndarray)
            Sub-pixel localized points Y coordinates
        Intensity (np.ndarray)
            Sub-pixel localized points intensity estimate

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        if not len(X_loc) == len(Y_loc) == len(Intensity):
            raise Exception('The supplied arguments are of different lengths.')

        x_max = int((np.max(X_loc) / self._pixel_size) +
                    np.ceil(self._gauss_std * 12))
        y_max = int((np.max(Y_loc) / self._pixel_size) +
                    np.ceil(self._gauss_std * 12))

        step = int(np.ceil(self._gauss_std * 3))

        image = np.zeros([y_max, x_max])

        for idx in range(len(X_loc)):
            x = round(X_loc[idx] / self._pixel_size) + 2 * step
            y = round(Y_loc[idx] / self._pixel_size) + 2 * step

            image[y - step:y + step + 1, x - step:x + step + 1] += \
                Intensity[idx] * self._gauss_2d

        return image


# g = gauss_hist_render(10)
# img = cv2.normalize(g._gauss_2d, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# img = cv2.resize(
#     img, (0, 0), fx=100, fy=100, interpolation=cv2.INTER_NEAREST)
# cv2.imshow("k", img)
# cv2.waitKey(0)
