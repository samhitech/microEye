import math
from enum import Enum

import cv2
import numpy as np
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

from microEye.analysis.filters.base import AbstractFilter


class PassFilter(AbstractFilter):
    def run(self, image: np.ndarray) -> np.ndarray:
        return image

    def get_metadata(self):
        '''Return metadata about the Pass filter.'''
        return {
            'name': 'Pass Filter'
        }

class DoG_Filter(AbstractFilter):
    def __init__(self, sigma: float = 1, factor: float = 2.5) -> None:
        '''Difference of Gauss init

        Parameters
        ----------
        sigma : float, optional
            sigma_min, by default 1
        factor : float, optional
            factor = sigma_max/sigma_min, by default 2.5
        '''
        self.set_params(sigma, factor)

    def set_params(self, sigma: float, factor: float):
        '''Set filter parameters.

        Parameters
        ----------
        sigma : float
            sigma_min
        factor : float
            factor = sigma_max/sigma_min
        '''
        self._show_filter = False
        self.sigma = sigma
        self.factor = factor
        self.rsize = max(math.ceil(6 * self.sigma + 1), 3)
        self.dog = DoG_Filter.gaussian_kernel(
            self.rsize, self.sigma
        ) - DoG_Filter.gaussian_kernel(self.rsize, max(1, self.factor * self.sigma))

    def gaussian_kernel(dim, sigma):
        x = cv2.getGaussianKernel(dim, sigma)
        kernel = x.dot(x.T)
        return kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        rows, cols = image.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        pad_rows = nrows - rows
        pad_cols = ncols - cols

        # Ensure that pad_cols[1] and pad_rows[1] are at least 1
        pad_rows = (pad_rows // 2, max(1, pad_rows - pad_rows // 2))
        pad_cols = (pad_cols // 2, max(1, pad_cols - pad_cols // 2))

        nimg = np.pad(image, (pad_rows, pad_cols), mode='reflect')

        res = cv2.normalize(
            signal.convolve2d(nimg, np.rot90(self.dog), mode='same')[
                pad_rows[0] : -pad_rows[1], pad_cols[0] : -pad_cols[1]
            ],
            None,
            0,
            255,
            cv2.NORM_MINMAX,
            cv2.CV_8U,
        )
        return res

    def get_metadata(self):
        '''Return metadata about the Difference of Gaussians filter.'''
        return {
            'name': 'Difference of Gaussains Filter',
            'sigma': self.sigma,
            'factor': self.factor,
        }


class GaussFilter(AbstractFilter):
    def __init__(self, sigma=1) -> None:
        '''Gaussian filter initialization.'''
        self.set_sigma(sigma)

    def set_sigma(self, sigma):
        '''Set the sigma parameter for the Gaussian filter.'''
        self.sigma = 1
        self.gauss = GaussFilter.gaussian_kernel(
            max(3, math.ceil(3 * sigma + 1)), sigma
        )

    def gaussian_kernel(dim, sigma):
        x = cv2.getGaussianKernel(dim, sigma)
        kernel = x.dot(x.T)
        return kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        return signal.convolve2d(image, np.rot90(self.gauss), mode='same')


class BANDPASS_TYPES(Enum):
    Gaussian = 1
    Butterworth = 2
    Ideal = 3

    @classmethod
    def from_string(cls, s: str):
        for column in cls:
            if column.name.lower() == s.lower() or s.lower() in column.name.lower():
                return column
        raise ValueError(f'{cls.__name__} has no value matching "{s}"')

    @classmethod
    def values(cls):
        return [column.name for column in cls]


class BandpassFilter(AbstractFilter):
    def __init__(
        self, center=40.0, width=90.0, filter_type=BANDPASS_TYPES.Gaussian, show=False
    ) -> None:
        '''Bandpass filter initialization.'''
        self._radial_coordinates = None
        self._filter = None
        self._center = center
        self._width = width
        self._type = filter_type
        self._show_filter = show
        self._refresh = True

    def radial_coordinates(self, shape):
        '''Generates a 2D array with radial cordinates
        with according to the first two axis of the
        supplied shape tuple

        Returns
        -------
        R, Rsq
            Radius 2d matrix (R) and radius squared matrix (Rsq)
        '''
        y_len = np.arange(-shape[0] // 2, shape[0] // 2)
        x_len = np.arange(-shape[1] // 2, shape[1] // 2)

        X, Y = np.meshgrid(x_len, y_len)

        Rsq = X**2 + Y**2

        self._radial_coordinates = (np.sqrt(Rsq), Rsq)

        return self._radial_coordinates

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

        cir_filter = np.zeros((shape[0], shape[1]), dtype=np.float32)
        cir_center = (shape[1] // 2, shape[0] // 2)
        cir_filter = cv2.circle(cir_filter, cir_center, cuton, 1, -1)
        cir_filter = cv2.circle(cir_filter, cir_center, cutoff, 0, -1)
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
        if self._radial_coordinates is None:
            R, Rsq = self.radial_coordinates(shape)
        elif self._radial_coordinates[0].shape != shape[:2]:
            R, Rsq = self.radial_coordinates(shape)
        else:
            R, Rsq = self._radial_coordinates

        with np.errstate(divide='ignore', invalid='ignore'):
            filter = np.exp(-(((Rsq - center**2) / (R * width)) ** 2))
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
        if self._radial_coordinates is None:
            R, Rsq = self.radial_coordinates(shape)
        elif self._radial_coordinates[0].shape != shape[:2]:
            R, Rsq = self.radial_coordinates(shape)
        else:
            R, Rsq = self._radial_coordinates

        with np.errstate(divide='ignore', invalid='ignore'):
            filter = 1 - (1 / (1 + ((R * width) / (Rsq - center**2)) ** 10))
            filter[filter == np.inf] = 0

        a, b = np.unravel_index(R.argmin(), R.shape)

        filter[a, b] = 1

        return filter

    def run(self, image: np.ndarray):
        '''Applies an FFT bandpass filter to the 2D image.

        Params
        -------
        image (np.ndarray)
            the image to be filtered.
        '''

        # time = QDateTime.currentDateTime()

        rows, cols = image.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        pad_rows = nrows - rows
        pad_cols = ncols - cols

        # Ensure that pad_cols[1] and pad_rows[1] are at least 1
        pad_rows = (pad_rows // 2, max(1, pad_rows - pad_rows // 2))
        pad_cols = (pad_cols // 2, max(1, pad_cols - pad_cols // 2))

        nimg = np.pad(image, (pad_rows, pad_cols), mode='reflect')
        # nimg = np.zeros((nrows, ncols))
        # nimg[:rows, :cols] = image

        ft = fftshift(cv2.dft(np.float64(nimg), flags=cv2.DFT_COMPLEX_OUTPUT))

        filter = np.ones(ft.shape[:2])

        refresh = self._refresh

        if self._filter is None:
            refresh = True
        elif self._filter.shape != ft.shape[:2]:
            refresh = True

        if refresh:
            if self._type == BANDPASS_TYPES.Gaussian.name:
                filter = self.gauss_bandpass_filter(ft.shape, self._center, self._width)
            elif self._type == BANDPASS_TYPES.Butterworth.name:
                filter = self.butterworth_bandpass_filter(
                    ft.shape, self._center, self._width
                )
            else:
                cutoff = int(max(0, self._center - self._width // 2))
                cuton = int(self._center + self._width // 2)
                filter = self.ideal_bandpass_filter(ft.shape, cutoff, cuton)

            self._filter = filter
        else:
            filter = self._filter

        if self._show_filter:
            cv2.namedWindow('BandpassFilter', cv2.WINDOW_NORMAL)
            cv2.imshow('BandpassFilter', (filter * 255).astype(np.uint8))

        img = np.zeros(nimg.shape, dtype=np.uint8)
        ft[:, :, 0] *= filter
        ft[:, :, 1] *= filter
        idft = cv2.idft(ifftshift(ft))
        idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        cv2.normalize(idft, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # exex = time.msecsTo(QDateTime.currentDateTime())
        return img[pad_rows[0] : -pad_rows[1], pad_cols[0] : -pad_cols[1]]

    def set_params(
        self, center: float, width: float, filter_type: BANDPASS_TYPES, show: bool
    ):
        self._center = center
        self._width = width
        self._type = filter_type
        self._show_filter = show

    def get_metadata(self):
        '''Return metadata about the Bandpass Fourier Filter.'''
        return {
            'name': 'Fourier Bandpass Filter',
            'type': self._type,
            'band center': self._center,
            'band width': self._width,
            'show filter': self._show_filter,
        }
