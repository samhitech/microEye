import cv2
import numpy as np
from numpy.lib.function_base import meshgrid
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


class uImage():

    def __init__(self, image: np.ndarray):
        self._image = image
        self._height = image.shape[0]
        self._width = image.shape[1]
        self._min = 0
        self._max = 2**16 if image.dtype == np.uint16 else 255
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


def fft_bandpass(image: np.ndarray, center: int, width: int):
    '''Applies a bandpass filter using fft of the 2D image.

        Params
        -------
        image (np.ndarray)
            the image to be filtered.
        min (np.ndarray)
            the inner radius of ring.
        max (np.ndarray)
            the outer radius of ring.
    '''
    ft = fftshift(fft2(image))

    y_len = np.arange(-ft.shape[0]//2, ft.shape[0]//2)
    x_len = np.arange(-ft.shape[1]//2, ft.shape[1]//2)
    X, Y = np.meshgrid(x_len, y_len)
    Rsq = (X**2 + Y**2)
    R = np.sqrt(Rsq)
    filter = np.exp(-((Rsq - center**2)/(R * width))**2)
    # filter = 1 - (1 / (1+((R * width)/(Rsq - center**2))**10))
    a, b = np.unravel_index(R.argmin(), R.shape)
    filter[a, b] = 1

    # cir_filter = np.zeros(ft.shape, dtype=np.float32)
    # cir_center = (
    #     int(ft.shape[1] / 2.0),
    #     int(ft.shape[0] / 2.0))
    # cir_filter = cv2.circle(
    #     cir_filter, cir_center, max, 1, -1)
    # cir_filter = cv2.circle(
    #     cir_filter, cir_center, min, 0, -1)
    # cir_filter = cv2.GaussianBlur(cir_filter, (0, 0), sigmaX=1, sigmaY=1)

    cv2.imshow('test', (filter*255).astype(np.uint8))

    img = np.zeros(image.shape, dtype=np.uint8)
    ft *= filter
    cv2.normalize(
        np.abs(ifft2(ft)), img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img
