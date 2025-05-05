import math
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from skimage import filters
from skimage.morphology import disk, white_tophat
from skimage.restoration import rolling_ball

from microEye.analysis.filters.base import SpatialFilter


class PassFilter(SpatialFilter):
    def __init__(self) -> None:
        '''Pass filter initialization.'''
        super().__init__(name='No Filter (Pass)')

    def run(self, image: np.ndarray) -> np.ndarray:
        return image


class DoG_Filter(SpatialFilter):
    def __init__(self, sigma: float = 1, factor: float = 2.5) -> None:
        '''Difference of Gauss init

        Parameters
        ----------
        sigma : float, optional
            sigma_min, by default 1
        factor : float, optional
            factor = sigma_max/sigma_min, by default 2.5
        '''
        params = {
            'sigma': sigma,
            'factor': factor,
        }
        super().__init__(name='Difference of Gaussians', parameters=params)
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
        self.parameters['sigma'] = sigma
        self.parameters['factor'] = factor
        self.rsize = max(math.ceil(6 * sigma + 1), 3)
        self.dog = DoG_Filter.gaussian_kernel(
            self.rsize, sigma
        ) - DoG_Filter.gaussian_kernel(self.rsize, max(1, factor * sigma))

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

    def get_tree_parameters(self):
        '''Return the parameters for the pyqtgraph tree view.'''
        return {
            'name': self.name,
            'type': 'group',
            'visible': False,
            'children': [
                {
                    'name': 'Sigma',
                    'type': 'float',
                    'value': self.parameters['sigma'],
                    'limits': [0.0, 100.0],
                    'step': 0.1,
                    'decimals': 2,
                    'tip': 'Standard deviation (\u03c3) min for the filter',
                },
                {
                    'name': 'Factor',
                    'type': 'float',
                    'value': self.parameters['factor'],
                    'limits': [0.0, 100.0],
                    'step': 0.1,
                    'decimals': 2,
                    'tip': 'Ratio of max \u03c3 to min \u03c3.',
                },
            ],
        }


# class GaussFilter(SpatialFilter):
#     def __init__(self, sigma=1) -> None:
#         '''Gaussian filter initialization.'''
#         params = {'sigma': sigma}
#         super().__init__(name='Gaussian Filter', parameters=params)
#         self.set_params(sigma)

#     def set_params(self, sigma):
#         '''Set the sigma parameter for the Gaussian filter.'''
#         self._show_filter = False
#         self.parameters['sigma'] = sigma
#         self.gauss = GaussFilter.gaussian_kernel(
#             max(3, math.ceil(3 * sigma + 1)), sigma
#         )

#     def gaussian_kernel(dim, sigma):
#         x = cv2.getGaussianKernel(dim, sigma)
#         kernel = x.dot(x.T)
#         return kernel

#     def run(self, image: np.ndarray) -> np.ndarray:
#         return cv2.normalize(
#             signal.convolve2d(image, np.rot90(self.gauss), mode='same'),
#             None,
#             0,
#             255,
#             cv2.NORM_MINMAX,
#             cv2.CV_8U,
#         )

#     def get_tree_parameters(self):
#         '''Return the parameters for the pyqtgraph tree view.'''
#         return {
#             'name': self.name,
#             'type': 'group',
#             'visible': False,
#             'children': [
#                 {
#                     'name': 'Sigma',
#                     'type': 'float',
#                     'value': self.parameters['sigma'],
#                     'limits': [0.0, 100.0],
#                     'step': 0.1,
#                     'decimals': 2,
#                     'tip': 'Standard deviation (\u03c3) for the filter',
#                 },
#             ],
#         }


class FourierFilter(SpatialFilter):
    class PROFILES(Enum):
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

    class PASS_TYPES(Enum):
        Band = 0
        Low = 1
        High = 2

        @classmethod
        def from_string(cls, s: str):
            for column in cls:
                if column.name.lower() == s.lower() or s.lower() in column.name.lower():
                    return column
            raise ValueError(f'{cls.__name__} has no value matching "{s}"')

        @classmethod
        def values(cls):
            return [column.name for column in cls]

    def __init__(
        self,
        center=40.0,
        width=90.0,
        filter_type=PROFILES.Gaussian,
        pass_type=PASS_TYPES.Band,
        order=5,
        show_filter=False,
    ) -> None:
        '''Fourier filter initialization.'''
        params = {
            'center': center,
            'width': width,
            'type': filter_type.name
            if isinstance(filter_type, FourierFilter.PROFILES)
            else filter_type,
            'pass_type': pass_type.name
            if isinstance(pass_type, FourierFilter.PASS_TYPES)
            else pass_type,
            'order': order,
        }
        super().__init__(name='Fourier Filter', parameters=params)
        self._radial_coordinates = None
        self._filter: Optional[np.ndarray] = None
        self._show_filter = show_filter
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

    def ideal_filter(self, shape, cutoff, cuton):
        '''Generates an ideal pass filter of shape

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

        if self.parameters['pass_type'] == FourierFilter.PASS_TYPES.Low.name:
            cir_filter = cv2.circle(cir_filter, cir_center, cutoff, 1, -1)
        elif self.parameters['pass_type'] == FourierFilter.PASS_TYPES.High.name:
            cir_filter = cv2.circle(1 + cir_filter, cir_center, cutoff, 0, -1)
        else:
            cir_filter = cv2.circle(cir_filter, cir_center, cuton, 1, -1)
            cir_filter = cv2.circle(cir_filter, cir_center, cutoff, 0, -1)

        return cir_filter

    def gaussian_filter(self, shape, center, width):
        '''Generates a Gaussian pass filter of shape

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
            if self.parameters['pass_type'] == FourierFilter.PASS_TYPES.Low.name:
                filter = np.exp(-(Rsq) / (2 * (center**2)))
            elif self.parameters['pass_type'] == FourierFilter.PASS_TYPES.High.name:
                filter = 1 - np.exp(-(Rsq) / (2 * (center**2)))
            else:
                filter = np.exp(-(((Rsq - center**2) / (R * width)) ** 2))

            filter[filter == np.inf] = 0

        a, b = np.unravel_index(R.argmin(), R.shape)

        filter[a, b] = 1

        return filter

    def butterworth_filter(self, shape, center, width, order=5):
        '''Generates a Gaussian pass filter of shape

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
            if self.parameters['pass_type'] == FourierFilter.PASS_TYPES.Low.name:
                filter = 1 / (1 + (R / center) ** (2 * order))
            elif self.parameters['pass_type'] == FourierFilter.PASS_TYPES.High.name:
                filter = 1 / (1 + (center / np.where(R == 0, 1e-10, R)) ** (2 * order))
            else:
                filter = 1 - (
                    1 / (1 + ((R * width) / (Rsq - center**2)) ** (2 * order))
                )

            filter[filter == np.inf] = 0

        a, b = np.unravel_index(R.argmin(), R.shape)

        filter[a, b] = 1

        return filter

    def run(self, image: np.ndarray):
        '''Applies an FFT pass filter to the 2D image.

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
            if self.parameters['type'] == FourierFilter.PROFILES.Gaussian.name:
                filter = self.gaussian_filter(
                    ft.shape, self.parameters['center'], self.parameters['width']
                )
            elif self.parameters['type'] == FourierFilter.PROFILES.Butterworth.name:
                filter = self.butterworth_filter(
                    ft.shape,
                    self.parameters['center'],
                    self.parameters['width'],
                    self.parameters['order'],
                )
            else:
                cutoff = int(
                    max(0, self.parameters['center'] - self.parameters['width'] // 2)
                )
                cuton = int(self.parameters['center'] + self.parameters['width'] // 2)
                filter = self.ideal_filter(ft.shape, cutoff, cuton)

            self._filter = filter
        else:
            filter = self._filter

        if self._show_filter:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
            cv2.imshow(self.name, (filter * 255).astype(np.uint8))

        img = np.zeros(nimg.shape, dtype=np.uint8)
        ft[:, :, 0] *= filter
        ft[:, :, 1] *= filter
        idft = cv2.idft(ifftshift(ft))
        idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        cv2.normalize(idft, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # exex = time.msecsTo(QDateTime.currentDateTime())
        return img[pad_rows[0] : -pad_rows[1], pad_cols[0] : -pad_cols[1]]

    def set_params(
        self,
        center: float,
        width: float,
        filter_type: PROFILES,
        pass_type: PASS_TYPES,
        show_filter: bool,
        order: int = 5,
    ):
        '''
        Set the parameters for the filter.
        '''
        self.parameters['center'] = center
        self.parameters['width'] = width
        self.parameters['order'] = order
        self.parameters['type'] = (
            filter_type.name
            if isinstance(filter_type, FourierFilter.PROFILES)
            else filter_type
        )
        self.parameters['pass_type'] = (
            pass_type.name
            if isinstance(pass_type, FourierFilter.PASS_TYPES)
            else pass_type
        )
        self._show_filter = show_filter

    def get_tree_parameters(self):
        return {
            'name': self.name,
            'type': 'group',
            'visible': False,
            'children': [
                {
                    'name': 'Filter Type',
                    'type': 'list',
                    'limits': FourierFilter.PROFILES.values(),
                    'value': FourierFilter.PROFILES.Gaussian.name,
                    'tip': 'Select the type of fourier filter',
                },
                {
                    'name': 'Pass Type',
                    'type': 'list',
                    'limits': FourierFilter.PASS_TYPES.values(),
                    'value': FourierFilter.PASS_TYPES.Band.name,
                    'tip': 'Select the type of pass filter',
                },
                {
                    'name': 'Center',
                    'type': 'float',
                    'value': self.parameters['center'],
                    'limits': [0.0, 2096.0],
                    'step': 0.5,
                    'decimals': 2,
                    'tip': 'Center frequency in pixels.',
                },
                {
                    'name': 'Width',
                    'type': 'float',
                    'value': 90,
                    'limits': [0.0, 2096.0],
                    'step': 0.5,
                    'decimals': 2,
                    'tip': 'The width of the band in pixels',
                },
                {
                    'name': 'Order',
                    'type': 'int',
                    'value': 5,
                    'limits': [1, 10],
                    'step': 1,
                    'tip': 'The order of the Butterworth filter',
                },
                {
                    'name': 'Show Filter',
                    'type': 'bool',
                    'value': self._show_filter,
                    'tip': 'Toggle to show or hide the filter',
                },
            ],
        }


class BackgroundReduction(SpatialFilter):
    METHODS = [
        'Rolling Ball',
        'White Top Hat',
    ]

    def __init__(self, radius: float = 30.0) -> None:
        '''Rolling ball filter initialization.'''
        params = {
            'radius': radius,
            'method': 'Rolling Ball',
        }
        super().__init__(name='Background Reduction', parameters=params)
        self.set_params(radius)

    def set_params(self, radius: float, method='Rolling Ball'):
        '''Set the radius parameter fand the method.'''
        self.parameters['radius'] = radius

        if method not in self.METHODS:
            raise ValueError(f'Method {method} not supported.')

        self.parameters['method'] = method

    def run(self, image: np.ndarray) -> np.ndarray:
        if self.parameters['method'] == self.METHODS[0]:
            return image - rolling_ball(image, radius=self.parameters['radius'])
        elif self.parameters['method'] == self.METHODS[1]:
            return white_tophat(image, disk(self.parameters['radius']))

        return image

    def get_tree_parameters(self):
        '''Return the parameters for the pyqtgraph tree view.'''
        return {
            'name': self.name,
            'type': 'group',
            'visible': False,
            'children': [
                {
                    'name': 'Radius',
                    'type': 'float',
                    'value': self.parameters['radius'],
                    'limits': [0.0, 250.0],
                    'step': 0.1,
                    'decimals': 2,
                    'tip': 'Radius of the rolling ball in pixels',
                },
                {
                    'name': 'Method',
                    'type': 'list',
                    'limits': self.METHODS,
                    'value': self.parameters['method'],
                    'tip': 'Select the method for background reduction',
                },
            ],
        }
