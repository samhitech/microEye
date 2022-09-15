import math

import cv2
import dask.array
import numba
import numpy as np
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QWidget
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

from .uImage import ZarrImageSequence


class AbstractFilter:

    def run(self, image: np.ndarray) -> np.ndarray:
        return image

    def getWidget(self):
        return QWidget()


class DoG_Filter(AbstractFilter):

    def __init__(self, sigma=1, factor=2.5) -> None:
        '''Difference of Gauss init

        Parameters
        ----------
        sigma : int, optional
            sigma_min, by default 1
        factor : float, optional
            factor = sigma_max/sigma_min, by default 2.5
        '''
        self.setParams(sigma, factor)

    def setParams(self, sigma, factor):
        self._show_filter = False
        self.sigma = sigma
        self.factor = factor
        self.rsize = max(math.ceil(6*self.sigma+1), 3)
        self.dog = \
            DoG_Filter.gaussian_kernel(self.rsize, self.sigma) - \
            DoG_Filter.gaussian_kernel(
                self.rsize, max(1, self.factor*self.sigma))

    def gaussian_kernel(dim, sigma):
        x = cv2.getGaussianKernel(dim, sigma)
        kernel = x.dot(x.T)
        return kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        return cv2.normalize(
            signal.convolve2d(image, np.rot90(self.dog), mode='same'),
            None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


class DoG_FilterWidget(QGroupBox):
    update = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.setTitle('DoG Filter')

        self.filter = DoG_Filter()

        self._layout = QFormLayout()

        self.arg_1 = QDoubleSpinBox()
        self.arg_1.setMinimum(0)
        self.arg_1.setMaximum(1000)
        self.arg_1.setSingleStep(0.1)
        self.arg_1.setValue(1.0)
        self.arg_1.valueChanged.connect(self.value_changed)

        self.arg_2 = QDoubleSpinBox()
        self.arg_2.setMinimum(1)
        self.arg_2.setMaximum(100)
        self.arg_2.setSingleStep(0.1)
        self.arg_2.setValue(2.5)
        self.arg_2.valueChanged.connect(self.value_changed)

        self.arg_3 = QCheckBox('Show Filter')
        self.arg_3.setChecked(True)
        self.arg_3.stateChanged.connect(self.value_changed)

        self.arg_1.valueChanged.emit(1.0)
        self.arg_2.valueChanged.emit(2.5)

        self._layout.addRow(
            QLabel('\u03C3 min:'),
            self.arg_1)
        self._layout.addRow(
            QLabel('Factor (\u03C3 max/\u03C3 min):'),
            self.arg_2)
        self._layout.addWidget(self.arg_3)

        self.setLayout(self._layout)

    def value_changed(self, value):
        self.filter.setParams(
            self.arg_1.value(), self.arg_2.value()
        )
        self.filter._show_filter = self.arg_3.isChecked()

        self.update.emit()


class GaussFilter(AbstractFilter):

    def __init__(self, sigma=1) -> None:
        self.setSigma(sigma)

    def setSigma(self, sigma):
        self.sigma = 1
        self.gauss = GaussFilter.gaussian_kernel(
            max(3, math.ceil(3*sigma+1)), sigma)

    def gaussian_kernel(dim, sigma):
        x = cv2.getGaussianKernel(dim, sigma)
        kernel = x.dot(x.T)
        return kernel

    def run(self, image: np.ndarray) -> np.ndarray:
        return signal.convolve2d(image, np.rot90(self.gauss), mode='same')


class BandpassFilter(AbstractFilter):

    def __init__(self) -> None:
        self._radial_cordinates = None
        self._filter = None
        self._center = 40.0
        self._width = 90.0
        self._type = 'gauss'
        self._show_filter = True
        self._refresh = True

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

        cir_filter = np.zeros((shape[0], shape[1]), dtype=np.float32)
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
        elif self._radial_cordinates[0].shape != shape[:2]:
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
        elif self._radial_cordinates[0].shape != shape[:2]:
            R, Rsq = self.radial_cordinate(shape)
        else:
            R, Rsq = self._radial_cordinates

        with np.errstate(divide='ignore', invalid='ignore'):
            filter = 1 - (1 / (1+((R * width)/(Rsq - center**2))**10))
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
        nimg = np.zeros((nrows, ncols))
        nimg[:rows, :cols] = image

        ft = fftshift(cv2.dft(np.float64(nimg), flags=cv2.DFT_COMPLEX_OUTPUT))

        filter = np.ones(ft.shape[:2])

        refresh = self._refresh

        if self._filter is None:
            refresh = True
        elif self._filter.shape != ft.shape[:2]:
            refresh = True

        if refresh:
            if 'gauss' in self._type:
                filter = self.gauss_bandpass_filter(
                    ft.shape, self._center, self._width)
            elif 'butter' in self._type:
                filter = self.butterworth_bandpass_filter(
                    ft.shape, self._center, self._width)
            else:
                cutoff = int(max(0, self._center - self._width // 2))
                cuton = int(self._center + self._width // 2)
                filter = self.ideal_bandpass_filter(ft.shape, cutoff, cuton)

            self._filter = filter
        else:
            filter = self._filter

        if self._show_filter:
            cv2.namedWindow("BandpassFilter", cv2.WINDOW_NORMAL)
            cv2.imshow('BandpassFilter', (filter*255).astype(np.uint8))

        img = np.zeros(nimg.shape, dtype=np.uint8)
        ft[:, :, 0] *= filter
        ft[:, :, 1] *= filter
        idft = cv2.idft(ifftshift(ft))
        idft = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
        cv2.normalize(
            idft, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # exex = time.msecsTo(QDateTime.currentDateTime())
        return img[:rows, :cols]


class BandpassFilterWidget(QGroupBox):
    update = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.setTitle('Bandpass Filter')

        self.filter = BandpassFilter()

        self._layout = QFormLayout()

        self.arg_1 = QComboBox()
        self.arg_1.addItems(['gauss', 'butter', 'ideal'])
        self.arg_1.setCurrentText('gauss')
        self.arg_1.currentTextChanged.connect(self.value_changed)
        self.arg_1.currentTextChanged.emit('gauss')

        self.arg_2 = QDoubleSpinBox()
        self.arg_2.setMinimum(0)
        self.arg_2.setMaximum(2096)
        self.arg_2.setSingleStep(2)
        self.arg_2.setValue(40.0)
        self.arg_2.valueChanged.connect(self.value_changed)
        self.arg_2.valueChanged.emit(40.0)

        self.arg_3 = QDoubleSpinBox()
        self.arg_3.setMinimum(0)
        self.arg_3.setMaximum(2096)
        self.arg_3.setSingleStep(2)
        self.arg_3.setValue(90.0)
        self.arg_3.valueChanged.connect(self.value_changed)
        self.arg_3.valueChanged.emit(90.0)

        self.arg_4 = QCheckBox('Show Filter')
        self.arg_4.setChecked(True)
        self.arg_4.stateChanged.connect(self.value_changed)

        self._layout.addRow(
            QLabel('Filter type:'),
            self.arg_1)
        self._layout.addRow(
            QLabel('Center:'),
            self.arg_2)
        self._layout.addRow(
            QLabel('Width:'),
            self.arg_3)
        self._layout.addWidget(self.arg_4)

        self.setLayout(self._layout)

    def value_changed(self, value):
        if self.sender() is self.arg_1:
            self.filter._type = self.arg_1.currentText()
        elif self.sender() is self.arg_2:
            self.filter._center = value
        elif self.sender() is self.arg_3:
            self.filter._width = value
        elif self.sender() is self.arg_4:
            self.filter._show_filter = self.arg_4.isChecked()

        self.update.emit()


class TemporalMedianFilter(AbstractFilter):

    def __init__(self) -> None:
        super().__init__()

        self._temporal_window = 3
        self._frames = None
        self._median = None

    def getFrames(self, index, dataHandler: ZarrImageSequence):
        if self._temporal_window < 2:
            self._frames = None
            self._median = None
            self._start = None
            return

        maximum = dataHandler.shape[0]
        step = 0
        if self._temporal_window % 2:
            step = self._temporal_window // 2
        else:
            step = (self._temporal_window // 2) - 1

        self._start = min(
            max(index-step, 0),
            maximum - self._temporal_window)

        data_slice = slice(
                self._start,
                (self._start + self._temporal_window),
                1)
        return dask.array.from_array(
            dataHandler.getSlice(data_slice, 0, 0))

    def run(self, image: np.ndarray, frames: np.ndarray, roiInfo=None):
        if frames is not None:
            if roiInfo is None:
                median = np.array(
                    dask.array.median(frames, axis=0))

                img = image - median
            else:
                origin = roiInfo[0]  # ROI (x,y)
                dim = roiInfo[1]  # ROI (w,h)
                median = np.array(
                    dask.array.median(frames[
                        :,
                        int(origin[1]):int(origin[1] + dim[1]),
                        int(origin[0]):int(origin[0] + dim[0])
                    ], axis=0))

                img = image.astype(np.float64)
                img[
                    int(origin[1]):int(origin[1] + dim[1]),
                    int(origin[0]):int(origin[0] + dim[0])] -= median

            img[img < 0] = 0
            # max_val = np.iinfo(image.dtype).max
            # img *= 10  # 0.9 * max_val / img.max()
            return img.astype(image.dtype)
        else:
            return image


class TemporalMedianFilterWidget(QGroupBox):
    update = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.setTitle('Temporal Median Filter')

        self.filter = TemporalMedianFilter()

        self._layout = QFormLayout()

        self.window_size = QSpinBox()
        self.window_size.setMinimum(1)
        self.window_size.setMaximum(2096)
        self.window_size.setSingleStep(1)
        self.window_size.setValue(3)
        self.window_size.valueChanged.connect(self.value_changed)
        self.window_size.valueChanged.emit(3)

        self.enabled = QCheckBox('Enable Filter')
        self.enabled.setChecked(False)
        self.enabled.stateChanged.connect(self.value_changed)

        self._layout.addRow(
            QLabel('Temporal Window Size:'),
            self.window_size)
        self._layout.addWidget(self.enabled)

        self.setLayout(self._layout)

    def value_changed(self, value):
        if self.sender() is self.window_size:
            self.filter._temporal_window = value

        self.update.emit()
