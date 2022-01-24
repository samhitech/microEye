from sys import flags
import cv2
import numba
import numpy as np
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import select
from pandas.core import frame
import tifffile as tf
import zarr
from PyQt5.QtCore import *
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift


class uImage():

    def __init__(self, image: np.ndarray):
        self._image = image
        self._height = image.shape[0]
        self._width = image.shape[1]
        self._min = 0
        self._max = 255 if image.dtype == np.uint8 else 2**16 - 1
        self._isfloat = (image.dtype == np.float16
                         or image.dtype == np.float32
                         or image.dtype == np.float64)
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
        self.n_bins = 2**16 if self._image.dtype == np.uint16 else 256
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
        if nLUT and not self._isfloat:
            self.calcHist()

            if range is not None:
                self._min = min(max(range[0], 0), self.n_bins - 1)
                self._max = min(max(range[1], 0), self.n_bins - 1)

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


class TiffSeqHandler:

    def __init__(self, tiffSeq: tf.TiffSequence) -> None:
        self._tiffSeq = tiffSeq
        self._stores = [None] * len(tiffSeq.files)
        self._zarr = [None] * len(tiffSeq.files)
        self._frames = [None] * len(tiffSeq.files)
        self._data = None
        self._shape = None
        self._dtype = None

    def open(self):
        for idx, file in enumerate(self._tiffSeq.files):
            self._stores[idx] = tf.imread(file, aszarr=True)
            self._zarr[idx] = zarr.open(self._stores[idx], mode='r')
            self._zarr[idx][0]
            self._frames[idx] = self._zarr[idx].shape[0]

        self._shape = (sum(self._frames),) + self._zarr[0].shape[1:]
        self._dtype = self._zarr[0].dtype
        self._cum_frames = np.cumsum(self._frames)

    def close(self):
        for store in self._stores:
            store.close()
        self._tiffSeq.close()

    def __getitem__(self, i):
        if not isinstance(i, slice):
            file_idx = 0
            for idx, cum_frames in enumerate(self._cum_frames):
                if i <= cum_frames - 1:
                    file_idx = idx
                    i -= (cum_frames - self._frames[idx])
                    break

            return self._zarr[file_idx][i]
        else:
            start = i.start
            stop = i.stop
            if stop <= self._cum_frames[0]:
                return self._zarr[0][i]
            else:
                indices = np.arange(start, stop)
                result = np.empty(
                    shape=(0,) + self._zarr[0].shape[1:], dtype=self._dtype)
                for idx, cum_frames in enumerate(self._cum_frames):
                    mask = np.logical_and(
                        cum_frames - self._frames[idx] <= indices,
                        indices < cum_frames)
                    if np.sum(mask) > 0:
                        r = indices[mask] - (cum_frames - self._frames[idx])
                        result = np.concatenate((
                            result,
                            self._zarr[idx][np.min(r):np.max(r)+1]), axis=0)

            return result

    def __len__(self):
        return sum(self._frames)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
