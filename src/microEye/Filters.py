import re
from PyQt5.QtWidgets import QWidget
import cv2
from numba.core.types import abstract
import numpy as np
from numpy.lib.type_check import imag
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import numba
import tifffile as tf

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class AbstractFilter:

    def run(self, image: np.ndarray) -> np.ndarray:
        return image

    def getWidget(self):
        return QWidget()


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


class BandpassFilterWidget(QGroupBox):
    update = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.setTitle('Bandpass Filter')

        self.filter = BandpassFilter()

        self._layout = QVBoxLayout()

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

        self._layout.addWidget(QLabel('Filter arg 1 (type)'))
        self._layout.addWidget(self.arg_1)
        self._layout.addWidget(QLabel('Filter arg 2 (center)'))
        self._layout.addWidget(self.arg_2)
        self._layout.addWidget(QLabel('Filter arg 3 (width)'))
        self._layout.addWidget(self.arg_3)
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

    def getFrames(self, index, maximum, file: str):
        if self._temporal_window < 2:
            self._frames = None
            self._median = None
            self._start = None
            return

        step = 0
        if self._temporal_window % 2:
            step = self._temporal_window // 2
        else:
            step = (self._temporal_window // 2) - 1

        self._start = min(
            max(index-step, 0),
            maximum - self._temporal_window)

        # self._frames = np.array(
        #     [x.asarray() for x in
        #         file.pages[self._start:self._start + self._temporal_window]])
        self._frames = tf.imread(
            file,
            key=slice(
                self._start,
                (self._start + self._temporal_window),
                1))

    def run(self, image: np.ndarray):
        if self._frames is not None:
            self._median = np.median(self._frames, axis=0)

            img = image - self._median
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

        self._layout = QVBoxLayout()

        self.window_size = QSpinBox()
        self.window_size.setMinimum(1)
        self.window_size.setMaximum(2096)
        self.window_size.setSingleStep(1)
        self.window_size.setValue(3)
        self.window_size.valueChanged.connect(self.value_changed)
        self.window_size.valueChanged.emit(3)

        self.enabled = QCheckBox('Enable Filter')
        self.enabled.setChecked(True)
        self.enabled.stateChanged.connect(self.value_changed)

        self._layout.addWidget(QLabel('Window Temporal Size:'))
        self._layout.addWidget(self.window_size)
        self._layout.addWidget(self.enabled)

        self.setLayout(self._layout)

    def value_changed(self, value):
        if self.sender() is self.window_size:
            self.filter._temporal_window = value

        self.update.emit()
