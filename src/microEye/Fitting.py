import re
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import numba

from PyQt5.QtCore import *


def phasor_fit(image: np.ndarray, points: np.ndarray,
               intensity=True, roi_size=7):
    '''Sub-pixel Phasor 2D fit

    More details:
        see doi.org/10.1063/1.5005899 (Martens et al., 2017)
    '''
    if len(points) < 1:
        return

    sub_fit = np.zeros((points.shape[0], 4), points.dtype)

    if intensity:
        bg_mask, sig_mask = roi_mask(roi_size)

    for r in range(points.shape[0]):
        x, y = points[r, :]
        idx = int(x - roi_size//2)
        idy = int(y - roi_size//2)
        if idx < 0:
            idx = 0
        if idy < 0:
            idy = 0
        if idx + roi_size > image.shape[1]:
            idx = image.shape[1] - roi_size
        if idy + roi_size > image.shape[0]:
            idy = image.shape[0] - roi_size
        roi = image[idy:idy+roi_size, idx:idx+roi_size]
        fft_roi = np.fft.fft2(roi)
        theta_x = np.angle(fft_roi[0, 1])
        theta_y = np.angle(fft_roi[1, 0])
        if theta_x > 0:
            theta_x = theta_x - 2 * np.pi
        if theta_y > 0:
            theta_y = theta_y - 2 * np.pi
        x = idx + np.abs(theta_x) / (2 * np.pi / roi_size)
        y = idy + np.abs(theta_y) / (2 * np.pi / roi_size)
        sub_fit[r, :2] = [x, y]

        if intensity:
            sub_fit[r, 2] = intensity_estimate(roi, bg_mask, sig_mask)

    return sub_fit


def radial_cordinate(shape):
    '''Generates a 2D array with radial cordinates
    with according to the first two axis of the
    supplied shape tuple

    Returns
    -------
    R, Rsq
        Radius 2d matrix (R) and radius squared matrix (Rsq)
    '''
    y_len = np.arange(-shape[0]//2, shape[0]//2)
    x_len = np.arange(-shape[1]//2, shape[1]//2)

    X, Y = np.meshgrid(x_len, y_len)

    Rsq = (X**2 + Y**2)

    return np.sqrt(Rsq), Rsq


def roi_mask(roi_size=7):

    roi_shape = [roi_size] * 2
    roi_radius = roi_size / 2

    radius_map, _ = radial_cordinate(roi_shape)

    bg_mask = radius_map > (roi_radius - 0.5)
    sig_mask = radius_map <= roi_radius

    return bg_mask, sig_mask


def intensity_estimate(roi: np.ndarray, bg_mask, sig_mask, percentile=56):

    background_map = roi[bg_mask]
    background = np.percentile(
        background_map, percentile)

    intensity = np.sum(roi[sig_mask]) - (np.sum(sig_mask) * background)

    return max(0, intensity)

# @numba.jit(nopython=True)
# def phasor_fit_numba(image: np.ndarray, points: np.ndarray, roi_size=7):
#     for r in range(points.shape[0]):
#         x, y = points[r, :]
#         idx = int(x - roi_size//2)
#         idy = int(y - roi_size//2)
#         if idx < 0:
#             idx = 0
#         if idy < 0:
#             idy = 0
#         if idx + roi_size > image.shape[1]:
#             idx = image.shape[1] - roi_size
#         if idy + roi_size > image.shape[0]:
#             idy = image.shape[0] - roi_size
#         roi = image[idy:idy+roi_size, idx:idx+roi_size]
#         with numba.objmode(fft_roi='complex128[:,:]'):
#             fft_roi = np.fft.fft2(roi)
#         with numba.objmode(theta_x='float64'):
#             theta_x = np.angle(fft_roi[0, 1])
#         with numba.objmode(theta_y='float64'):
#             theta_y = np.angle(fft_roi[1, 0])
#         if theta_x > 0:
#             theta_x = theta_x - 2 * np.pi
#         if theta_y > 0:
#             theta_y = theta_y - 2 * np.pi
#         x = idx + np.abs(theta_x) / (2 * np.pi / roi_size)
#         y = idy + np.abs(theta_y) / (2 * np.pi / roi_size)
#         points[r, :] = [x, y]
