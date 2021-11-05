import cv2
import numpy as np


class uImage():

    def __init__(self, image: np.ndarray):
        self._image = image
        self._height = image.shape[0]
        self._width = image.shape[1]
        self._min = 0
        self._max = 4095 if image.dtype == np.uint16 else 255
        self._view = np.zeros(image.shape, dtype=np.uint8)

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
        self.n_bins = 4096 if self._image.dtype == np.uint16 else 256
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
