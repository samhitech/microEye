import re
import cv2
import numpy as np
import pyqtgraph as pg
from numpy.ma import count
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.interpolate import interp1d
from skimage.registration import phase_cross_correlation
import numba
import pandas as pd

from PyQt5.QtCore import *

from .Rendering import *


def phasor_fit(image: np.ndarray, points: np.ndarray,
               intensity=True, roi_size=7):
    '''Sub-pixel Phasor 2D fit

    More details:
        see doi.org/10.1063/1.5005899 (Martens et al., 2017)
    '''
    if len(points) < 1:
        return None

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


class ResultsUnits:
    Pixel = 0
    Nanometer = 1


class FittingResults:

    columns = np.array([
        'frame', 'x [pixel]', 'y [pixel]', 'x [nm]', 'y [nm]', 'intensity'
    ])

    def __init__(self, unit=ResultsUnits.Pixel, pixelSize=130.0):
        '''Fitting Results

        Parameters
        ----------
        unit : int, optional
            unit of localized points, by default ResultsUnits.Pixel
        pixelSize : float, optional
            pixel size in nanometers, by default 130.0
        '''
        self.unit = unit
        self.pixelSize = pixelSize
        self.locX = []
        self.locY = []
        self.locX_nm = []
        self.locY_nm = []
        self.frame = []
        self.intensity = []

    def extend(self, data: np.ndarray):
        '''Extend results by contents of data array

        Parameters
        ----------
        data : np.ndarray
            array of shape (n, m=4), columns (X, Y, Intensity, Frame)
        '''
        if self.unit is ResultsUnits.Pixel:
            self.locX.extend(data[:, 0])
            self.locY.extend(data[:, 1])
        else:
            self.locX_nm.extend(data[:, 0])
            self.locY_nm.extend(data[:, 1])

        self.intensity.extend(data[:, 2])
        self.frame.extend(data[:, 3])

    def dataFrame(self):
        '''Return fitting results as Pandas DataFrame

        Returns
        -------
        DataFrame
            fitting results DataFrame with columns FittingResults.columns
        '''
        if self.unit is ResultsUnits.Pixel:
            loc = np.c_[
                    np.array(self.frame),
                    np.array(self.locX),
                    np.array(self.locY),
                    np.array(self.locX) * self.pixelSize,
                    np.array(self.locY) * self.pixelSize,
                    np.array(self.intensity)]
        else:
            loc = np.c_[
                    np.array(self.frame),
                    np.array(self.locX_nm) / self.pixelSize,
                    np.array(self.locY_nm) / self.pixelSize,
                    np.array(self.locX_nm),
                    np.array(self.locY_nm),
                    np.array(self.intensity)]

        return pd.DataFrame(loc, columns=FittingResults.columns)

    def toRender(self):
        '''Returns columns for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains X [nm], Y [nm], Intensity columns
        '''
        if self.unit is ResultsUnits.Pixel:
            return np.array(self.locX) * self.pixelSize, \
                np.array(self.locY) * self.pixelSize, \
                np.array(self.intensity)
        else:
            return np.array(self.locX_nm), \
                np.array(self.locY_nm), \
                np.array(self.intensity)

    def __len__(self):
        counts = [len(self.locX), len(self.locY),
                  len(self.locX_nm), len(self.locY_nm)]
        return np.max(counts)

    def fromFile(filename: str, pixelSize: float):
        '''Populates fitting results from a tab seperated values
        (tsv) file.

        Parameters
        ----------
        filename : str
            path to tab seperated file (.tsv)
        pixelSize : float
            projected pixel size in nanometers

        Returns
        -------
        FittingResults
            FittingResults with imported data
        '''
        dataFrame = pd.read_csv(
                filename,
                sep='\t',
                engine='python')

        fittingResults = None

        if FittingResults.columns[3] in dataFrame and \
                FittingResults.columns[4] in dataFrame:
            fittingResults = FittingResults(
                ResultsUnits.Nanometer,
                pixelSize)
            fittingResults.locX_nm = \
                dataFrame[FittingResults.columns[3]]
            fittingResults.locY_nm = \
                dataFrame[FittingResults.columns[4]]
        elif FittingResults.columns[1] in dataFrame and \
                FittingResults.columns[2] in dataFrame:
            fittingResults = FittingResults(
                ResultsUnits.Pixel,
                pixelSize)
            fittingResults.locX = \
                dataFrame[FittingResults.columns[1]]
            fittingResults.locY = \
                dataFrame[FittingResults.columns[2]]
        else:
            return None

        if FittingResults.columns[0] in dataFrame:
            fittingResults.frame = dataFrame[FittingResults.columns[0]]
        else:
            fittingResults.frame = np.zeros(len(fittingResults))

        if FittingResults.columns[5] in dataFrame:
            fittingResults.intensity = dataFrame[FittingResults.columns[5]]
        else:
            fittingResults.intensity = np.ones(len(fittingResults))

        return fittingResults

    def drift_cross_correlation(self, n_bins=10, pixelSize=10, upsampling=100):
        '''Corrects the XY drift using cross-correlation measurments

        Parameters
        ----------
        n_bins : int, optional
            Number of frame bins, by default 10
        pixelSize : int, optional
            Super-res image pixel size in nanometers, by default 10
        upsampling : int, optional
            phase_cross_correlation upsampling (check skimage.registration),
            by default 100

        Returns
        -------
        tuple(FittingResults, np.ndarray)
            returns the drift corrected fittingResults and recontructed image
        '''
        unique_frames = np.unique(self.frame)
        if len(unique_frames) < 2:
            print('Drift cross-correlation failed: no frame info.')
            return

        frames_per_bin = np.floor(np.max(unique_frames) / n_bins)

        if frames_per_bin < 1:
            print('Drift cross-correlation failed: large number of bins.')
            return

        renderEngine = gauss_hist_render(pixelSize)

        data = self.dataFrame().to_numpy()

        x_max = int((np.max(data[:, 3]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)
        y_max = int((np.max(data[:, 4]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)

        grouped_data = []
        sub_images = []
        shifts = []
        frames = []

        for f in range(0, n_bins):
            group = data[(data[:, 0] >= f * frames_per_bin) &
                         (data[:, 0] < (f + 1) * frames_per_bin + 1)]
            image = renderEngine.fromArray(group[:, 3:6], (y_max, x_max))
            frames.append(f * frames_per_bin + frames_per_bin/2)
            grouped_data.append(group)
            sub_images.append(image)
            print(
                'Bins: {:d}/{:d}'.format(f + 1, n_bins),
                end="\r")

        for idx, img in enumerate(sub_images):
            shift = phase_cross_correlation(
                img, sub_images[0], upsample_factor=upsampling)
            shifts.append(shift[0] * pixelSize)
            print(
                'Shift Est.: {:d}/{:d}'.format(idx + 1, len(sub_images)),
                end="\r")

        shifts = np.c_[shifts, np.array(frames)]
        print(
            'Shift Correction ...',
            end="\r")

        # An one-dimensional interpolation is applied
        # to drift traces in X and Y dimensions separately.
        interpy = interp1d(
            shifts[:, -1], shifts[:, 0],
            kind='quadratic', fill_value='extrapolate')
        interpx = interp1d(
            shifts[:, -1], shifts[:, 1],
            kind='quadratic', fill_value='extrapolate')
        # And this interpolation is used to get the shift at every frame-point
        frames_new = np.arange(0, np.max(unique_frames), 1)
        interpx = interpx(frames_new)
        interpy = interpy(frames_new)

        for i, (shift_x, shift_y) in enumerate(zip(interpx, interpy)):
            data[data[:, 0] == i, 3] -= shift_x
            data[data[:, 0] == i, 4] -= shift_y

        drift_corrected = FittingResults(
            ResultsUnits.Nanometer, self.pixelSize)
        drift_corrected.frame = data[:, 0]
        drift_corrected.locX_nm = data[:, 3]
        drift_corrected.locY_nm = data[:, 4]
        drift_corrected.intensity = data[:, 5]

        x_max = int((np.max(data[:, 3]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)
        y_max = int((np.max(data[:, 4]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)

        drift_corrected_image = renderEngine.fromArray(
            data[:, 3:6], (y_max, x_max))

        print(
            'Shift Plot ...',
            end="\r")

        # plot results
        plt = pg.plot()

        plt.showGrid(x=True, y=True)
        plt.addLegend()

        # set properties of the label for y axis
        plt.setLabel('left', 'drift', units='nm')

        # set properties of the label for x axis
        plt.setLabel('bottom', 'frame', units='')

        plt.setWindowTitle('Drift Cross-Correlation')

        # setting horizontal range
        plt.setXRange(0, np.max(unique_frames))

        # setting vertical range
        plt.setYRange(0, 1)

        line1 = plt.plot(
            frames_new, interpx,
            pen='r', symbol=None,
            symbolBrush=0.2, name='x-drift')
        line1 = plt.plot(
            frames_new, interpy,
            pen='y', symbol=None,
            symbolBrush=0.2, name='y-drift')

        print(
            'Done ...',
            end="\r")

        return drift_corrected, drift_corrected_image


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
