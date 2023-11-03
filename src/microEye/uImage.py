
from typing import Union
import cv2
import numpy as np
import tifffile as tf
import zarr
from PyQt5.QtCore import *
from numba import cuda


class uImage():

    def __init__(self, image: np.ndarray):
        self._isfloat = (image.dtype == np.float16
                         or image.dtype == np.float32
                         or image.dtype == np.float64)
        if self._isfloat:
            self._image = image.astype(np.float32)
            self._norm = (
                (2**16 - 1) * self._image / self._image.max()
                ).astype(np.uint16)
        else:
            self._image = image
            self._norm = None
        self._height = image.shape[0]
        self._width = image.shape[1]
        if len(image.shape) > 2:
            self._channels = image.shape[2]
        else:
            self._channels = 1
        self._min = 0
        self._max = 255 if image.dtype == np.uint8 else 2**16 - 1
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
        self._height = value.shape[0]
        self._width = value.shape[1]
        if len(value.shape) > 2:
            self._channels = value.shape[2]
        else:
            self._channels = 1

    @property
    def width(self):
        return self._image.shape[1]

    @property
    def height(self):
        return self._image.shape[0]

    @property
    def channels(self):
        return self._channels

    def calcHist_GPU(self):
        self.n_bins = 256 if self._image.dtype == np.uint8 else 2**16
        # Calculate the range of the input image
        min_value = np.min(self.image)
        max_value = np.max(self.image)
        value_range = max_value - min_value

        # Compute the bin width
        bin_width = value_range / self.n_bins

        # Allocate memory for the LUT on the GPU
        lut_device = cuda.to_device(np.zeros((self.n_bins,), dtype=np.int32))

        # Configure kernel launch parameters
        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            self.image.shape[0] + threads_per_block[0] - 1
            ) // threads_per_block[0]
        blocks_per_grid_y = (
            self.image.shape[1] + threads_per_block[1] - 1
            ) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        # Launch the generate LUT kernel
        generate_lut_kernel[blocks_per_grid, threads_per_block](
            self.image, lut_device, min_value, bin_width, self.n_bins)

        # Copy the LUT from GPU device to host memory
        self._hist = lut_device.copy_to_host().astype(np.float64)

        # Normalize the LUT values
        self._hist = self._hist / np.sum(self._hist)

        # calculate the cdf
        self._cdf = self._hist.cumsum()

        self._min = np.where(self._cdf >= 0.00001)[0][0]
        self._max = np.where(self._cdf >= 0.9999)[0][0]

    def calcHist(self):
        self.n_bins = 256 if self._image.dtype == np.uint8 else 2**16
        # calculate image histogram
        self._hist = cv2.calcHist(
            [self.image], [0], None,
            [self.n_bins],
            [0, int(self._image.max()) if self._isfloat else self.n_bins]
            ) / float(np.prod(self.image.shape))
        # calculate the cdf
        self._cdf = self._hist[:, 0].cumsum()

        self._min = np.where(self._cdf >= 0.00001)[0][0]
        self._max = np.where(self._cdf >= 0.999)[0][0]

    def fastHIST(self):
        self.n_bins = 256 if self._image.dtype == np.uint8 else 2**16
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
                self._min = min(max(range[0], 0), self.n_bins - 1)
                self._max = min(max(range[1], 0), self.n_bins - 1)

            self._LUT = np.zeros((self.n_bins), dtype=np.uint8)
            self._LUT[self._min:self._max] = np.linspace(
                0, 255, self._max - self._min, dtype=np.uint8)
            self._LUT[self._max:] = 255
            if not self._isfloat:
                self._view = self._LUT[self._image]
            else:
                self._view = self._LUT[self._norm]
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

    def hsplitData(self):
        mid = self._image.shape[1] // 2
        left_view = self._image[:, :mid]
        right_view = self._image[:, mid:]
        RGB_img = np.zeros(
            left_view.shape[:2] + (3,), dtype=np.uint8)

        RGB_img[..., 0] = left_view
        RGB_img[..., 1] = np.fliplr(right_view)

        return uImage(RGB_img)

    def hsplitViewOverlay(self, RGB=True) -> np.ndarray:
        mid = self._view.shape[1] // 2
        left_view = self._view[:, :mid]
        right_view = self._view[:, mid:]

        _img = np.zeros(
            left_view.shape[:2] + (3,), dtype=np.uint8)
        if RGB:
            _img[..., 1] = left_view
            _img[..., 2] = np.fliplr(right_view)
        else:
            _img[..., 1] = left_view
            _img[..., 0] = np.fliplr(right_view)
        return _img

    def hsplitView(self):
        mid = self.image.shape[1] // 2
        left_view = self.image[:, :mid]
        right_view = self.image[:, mid:]

        return uImage(left_view), uImage(np.fliplr(right_view))


@cuda.jit
def generate_lut_kernel(image, lut, min_value, bin_width, num_bins):
    row, col = cuda.grid(2)

    if row < image.shape[0] and col < image.shape[1]:
        # Calculate the bin index for the pixel value
        bin_index = int((image[row, col] - min_value) / bin_width)
        if bin_index >= num_bins:
            bin_index = num_bins - 1
        cuda.atomic.add(lut, bin_index, 1)


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
            n_dim = len(self._zarr[idx].shape)
            if n_dim > 2:
                self._frames[idx] = self._zarr[idx].shape[0]
            else:
                self._zarr[idx] = self._zarr[idx][:, :][np.newaxis, ...]
                self._frames[idx] = 1

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
            start = 0 if i.start is None else i.start
            stop = sum(self._frames) if start is None else i.stop
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

    def getSlice(
            self, timeSlice=slice(None), channelSlice=slice(None),
            zSlice=slice(None), ySlice=slice(None), xSlice=slice(None)):

        res = self[timeSlice]

        if np.ndim(res) == 2:
            return res[ySlice, xSlice]
        else:
            return res[:, ySlice, xSlice]

    def __len__(self):
        return sum(self._frames)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def shape(self):
        if self._zarr is None:
            return None

        return (sum(self._frames),) + self._zarr[0].shape[1:]


class ZarrImageSequence:
    def __init__(self, path: str) -> None:

        self.path = path

        self.data = None

    def __getitem__(self, i):
        if self.data is None:
            return None

        return zarr.open(self.path, 'r').__getitem__(i)

    def getSlice(
            self, timeSlice=slice(None), channelSlice=slice(None),
            zSlice=slice(None), ySlice=slice(None), xSlice=slice(None)):
        za = zarr.open(self.path, 'r')
        if len(za.shape) == 5:
            return za[timeSlice, channelSlice, zSlice, ySlice, xSlice]
        elif len(za.shape) == 4:
            return za[timeSlice, zSlice, ySlice, xSlice]
        else:
            return za[timeSlice, ySlice, xSlice]

    def open(self):
        self.data = zarr.open(self.path, 'r')

    def close(self):
        del self.data

    def __len__(self):
        if self.data is None:
            return 0
        return self.data.shape[0]

    @property
    def shape(self):
        if self.data is None:
            return None
        return self.data.shape


def saveZarrImage(
        path: str,
        imgSeq: Union[ZarrImageSequence, TiffSeqHandler],
        timeSlice: slice = slice(None),
        channelSlice: slice = slice(None),
        zSlice: slice = slice(None),
        ySlice: slice = slice(None),
        xSlice: slice = slice(None)):

    def ifnone(a, b):
        return b if a is None else a

    if isinstance(imgSeq, TiffSeqHandler):
        shape = (
            ifnone(
                timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
            1,
            1,
            ifnone(ySlice.stop, imgSeq.shape[1]) - ifnone(ySlice.start, 0),
            ifnone(xSlice.stop, imgSeq.shape[2]) - ifnone(xSlice.start, 0)
        )
        chunks = (
            min(10, shape[0]),
            min(10, shape[1]),
            min(10, shape[2]),
            shape[3],
            shape[4],
        )

        zarrImg = zarr.open(
            path, mode='w-',
            shape=shape, chunks=chunks,
            compressor=None, dtype=imgSeq._dtype)

        timeSlice = slice(ifnone(timeSlice.start, 0), shape[0])

        for idx in np.arange(len(imgSeq._zarr)):
            offset = imgSeq._cum_frames[idx] - imgSeq._frames[idx]
            zarrSlice = slice(
                max(
                    timeSlice.start,
                    offset,
                ),
                min(
                    timeSlice.stop,
                    imgSeq._cum_frames[idx]
                )
            )
            tiffSlice = slice(
                zarrSlice.start - offset,
                zarrSlice.stop - offset)
            zarrImg[zarrSlice, 0, 0] = \
                imgSeq._zarr[idx][tiffSlice, ySlice, xSlice]
            print('Saving ...      ', end='\r')

        print('Done ...      ', end='\r')
        return True
    elif isinstance(imgSeq, ZarrImageSequence):

        print('Saving ...      ', end='\r')
        shape = (
            ifnone(
                timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
            ifnone(
                channelSlice.stop, imgSeq.shape[1]
                ) - ifnone(channelSlice.start, 0),
            ifnone(zSlice.stop, imgSeq.shape[2]) - ifnone(zSlice.start, 0),
            ifnone(ySlice.stop, imgSeq.shape[3]) - ifnone(ySlice.start, 0),
            ifnone(xSlice.stop, imgSeq.shape[4]) - ifnone(xSlice.start, 0)
        )
        chunks = (
            min(10, shape[0]),
            min(10, shape[1]),
            min(10, shape[2]),
            shape[3],
            shape[4],
        )
        zarrImg = zarr.open(
            path, mode='w-',
            shape=shape, chunks=chunks,
            compressor=None, dtype=imgSeq.data.dtype)
        zarrImg[:] = imgSeq.getSlice(
            timeSlice, channelSlice, zSlice, ySlice, xSlice)

        print('Done ...      ', end='\r')
        return True

    print('Failed ...      ', end='\r')
    return False
