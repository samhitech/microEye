import re
import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import numba

from PyQt5.QtCore import *


class uImage():

    def __init__(self, image: np.ndarray):
        self._image = image
        self._height = image.shape[0]
        self._width = image.shape[1]
        self._min = 0
        self._max = 2**16 - 1 if image.dtype == np.uint16 else 255
        self._view = np.zeros(image.shape, dtype=np.uint8)
        self._hist = None
        self.n_bins = None
        self._cdf = None
        self._stats = {}
        self._pixel_w = 1.0
        self._pixel_h = 1.0

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value: np.ndarray):
        if value.dtype is not np.uint16 or np.uint8:
            raise Exception(
                'Image must be a numpy array of type np.uint16 or np.uint8.')
        self._image = value
        self._height = self._image.shape[0]
        self._width = self._image.shape[1]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def calcHist(self):
        self.n_bins = 2**16 if self._image.dtype == np.uint16 else 256
        # calculate image histogram
        self._hist = cv2.calcHist(
            [self.image], [0], None,
            [self.n_bins], [0, self.n_bins]) / float(np.prod(self.image.shape))
        # calculate the cdf
        self._cdf = self._hist[:, 0].cumsum()

        self._min = np.where(self._cdf >= 0.00001)[0][0]
        self._max = np.where(self._cdf >= 0.9999)[0][0]

    def fastHIST(self):
        self.n_bins = 4096 if self._image.dtype == np.uint16 else 256
        cv2.normalize(
            src=self._image, dst=self._view,
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # calculate image histogram
        self._hist = cv2.calcHist(
            [self._view], [0], None,
            [self.n_bins], [0, self.n_bins]) / float(np.prod(self._view.shape))
        # calculate the cdf
        self._cdf = self._hist[:, 0].cumsum()

        self._min = np.where(self._cdf >= 0.00001)[0][0]
        self._max = np.where(self._cdf >= 0.9999)[0][0]

    def equalizeLUT(self, range=None, nLUT=False):
        if nLUT:
            self.calcHist()

            if range is not None:
                self._min = min(max(range[0], 0), 4095)
                self._max = min(max(range[1], 0), 4095)

            self._LUT = np.zeros((self.n_bins), dtype=np.uint8)
            self._LUT[self._min:self._max] = np.linspace(
                0, 255, self._max - self._min, dtype=np.uint8)
            self._LUT[self._max:] = 255

            self._view = self._LUT[self._image]
        else:
            self.fastHIST()

            if range is not None:
                self._min = min(max(range[0], 0), 255)
                self._max = min(max(range[1], 0), 255)

            self._LUT = np.zeros((256), dtype=np.uint8)
            self._LUT[self._min:self._max] = np.linspace(
                0, 255, self._max - self._min, dtype=np.uint8)
            self._LUT[self._max:] = 255

            cv2.LUT(self._view, self._LUT, self._view)

    def getStatistics(self):
        if self._hist is None:
            self.calcHist()

        _sum = 0.0
        _sum_of_sq = 0.0
        _count = 0
        for i, count in enumerate(self._hist):
            _sum += float(i * count)
            _sum_of_sq += (i**2)*count
            _count += count

        self._stats['Mean'] = _sum / _count
        self._stats['Area'] = _count * self._pixel_w * self._pixel_h
        self.calcStdDev(_count, _sum, _sum_of_sq)

    def calcStdDev(self, n, sum, sum_of_sq):
        if n > 0.0:
            stdDev = sum_of_sq - (sum**2 / n)
            if stdDev > 0:
                self._stats['StdDev'] = np.sqrt(stdDev / (n - 1))
            else:
                self._stats['StdDev'] = 0.0
        else:
            self._stats['StdDev'] = 0.0

    def fromUINT8(buffer, height, width):
        res = np.ndarray(shape=(height, width), dtype='u1', buffer=buffer)
        return uImage(res)

    def fromUINT16(buffer, height, width):
        res = np.ndarray(shape=(height, width), dtype='<u2', buffer=buffer)
        return uImage(res)

    def fromBuffer(buffer, height, width, bytes_per_pixel):
        if bytes_per_pixel == 1:
            return uImage.fromUINT8(buffer, height, width)
        else:
            return uImage.fromUINT16(buffer, height, width)


class BandpassFilter:

    def __init__(self) -> None:
        self._radial_cordinates = None
        self._filter = None

    def radial_cordinate(self, shape):
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

        self._radial_cordinates = (np.sqrt(Rsq), Rsq)

        return self._radial_cordinates

    def ideal_bandpass_filter(self, shape, cutoff, cuton):
        '''Generates an ideal bandpass filter of shape

            Params
            -------
            shape
                the shape of filter matrix
            cutoff
                the cutoff frequency (low)
            cuton
                the cuton frequency (high)

            Returns
            -------
            filter (np.ndarray)
                the filter in fourier space
        '''

        cir_filter = np.zeros(shape, dtype=np.float32)
        cir_center = (
            shape[1] // 2,
            shape[0] // 2)
        cir_filter = cv2.circle(
            cir_filter, cir_center, cuton, 1, -1)
        cir_filter = cv2.circle(
            cir_filter, cir_center, cutoff, 0, -1)
        # cir_filter = cv2.GaussianBlur(cir_filter, (0, 0), sigmaX=1, sigmaY=1)
        return cir_filter

    def gauss_bandpass_filter(self, shape, center, width):
        '''Generates a Gaussian bandpass filter of shape

            Params
            -------
            shape
                the shape of filter matrix
            center
                the center frequency
            width
                the filter bandwidth

            Returns
            -------
            filter (np.ndarray)
                the filter in fourier space
        '''
        if self._radial_cordinates is None:
            R, Rsq = self.radial_cordinate(shape)
        else:
            R, Rsq = self._radial_cordinates

        with np.errstate(divide='ignore', invalid='ignore'):
            filter = np.exp(-((Rsq - center**2)/(R * width))**2)
            filter[filter == np.inf] = 0

        a, b = np.unravel_index(R.argmin(), R.shape)

        filter[a, b] = 1

        return filter

    def butterworth_bandpass_filter(self, shape, center, width):
        '''Generates a Gaussian bandpass filter of shape

            Params
            -------
            shape
                the shape of filter matrix
            center
                the center frequency
            width
                the filter bandwidth

            Returns
            -------
            filter (np.ndarray)
                the filter in fourier space
        '''
        if self._radial_cordinates is None:
            R, Rsq = self.radial_cordinate(shape)
        else:
            R, Rsq = self._radial_cordinates

        with np.errstate(divide='ignore', invalid='ignore'):
            filter = 1 - (1 / (1+((R * width)/(Rsq - center**2))**10))
            filter[filter == np.inf] = 0

        a, b = np.unravel_index(R.argmin(), R.shape)

        filter[a, b] = 1

        return filter

    def run(self, image: np.ndarray, center: int, width: int, type='gauss',
            refresh=True, show_filter=True):
        '''Applies a bandpass filter using fft of the 2D image.

            Params
            -------
            image (np.ndarray)
                the image to be filtered.
            center (int)
                the center frequency.
            width (int)
                the filter bandwidth.
            filter (str)
                the filter type ('gauss' for gaussian,
                'butter' for butterworth, 'ideal')
        '''

        # time = QDateTime.currentDateTime()

        rows, cols = image.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        nimg = np.zeros((nrows, ncols))
        nimg[:rows, :cols] = image

        ft = fftshift(cv2.dft(np.float64(nimg), flags=cv2.DFT_COMPLEX_OUTPUT))

        filter = np.ones(ft.shape)

        if self._filter is None or refresh:
            if 'gauss' in type:
                filter = self.gauss_bandpass_filter(ft.shape, center, width)
            elif 'butter' in type:
                filter = self.butterworth_bandpass_filter(
                    ft.shape, center, width)
            else:
                cutoff = int(max(0, center - width // 2))
                cuton = int(center + width // 2)
                filter = self.ideal_bandpass_filter(ft.shape, cutoff, cuton)
        else:
            filter = self._filter

        if show_filter:
            cv2.imshow('test', (filter*255).astype(np.uint8))

        img = np.zeros(nimg.shape, dtype=np.uint8)
        ft[:, :, 0] *= filter
        ft[:, :, 1] *= filter
        idft = cv2.idft(ifftshift(ft))
        idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        cv2.normalize(
            idft, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # exex = time.msecsTo(QDateTime.currentDateTime())
        return img[:rows, :cols]


@numba.jit(nopython=True)
def phasor_fit_numba(image: np.ndarray, points: np.ndarray, roi_size=7):
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
        with numba.objmode(fft_roi='complex128[:,:]'):
            fft_roi = np.fft.fft2(roi)
        with numba.objmode(theta_x='float64'):
            theta_x = np.angle(fft_roi[0, 1])
        with numba.objmode(theta_y='float64'):
            theta_y = np.angle(fft_roi[1, 0])
        if theta_x > 0:
            theta_x = theta_x - 2 * np.pi
        if theta_y > 0:
            theta_y = theta_y - 2 * np.pi
        x = idx + np.abs(theta_x) / (2 * np.pi / roi_size)
        y = idy + np.abs(theta_y) / (2 * np.pi / roi_size)
        points[r, :] = [x, y]


def phasor_fit(image: np.ndarray, points: np.ndarray, roi_size=7):
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
        points[r, :] = [x, y]
