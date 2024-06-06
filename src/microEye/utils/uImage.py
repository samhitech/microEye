
from typing import Union

import cv2
import numpy as np
import tifffile as tf
import zarr
from numba import cuda

BYTE = 256
WORD = 2**16


class uImage:
    '''
    Class for handling image processing and analysis.

    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array.

    Attributes
    ----------
    _image : np.ndarray
        Processed image data.
    _isfloat : bool
        Flag indicating if the image data type is floating-point.
    _norm : np.ndarray or None
        Normalized image data.
    _min : int
        Minimum pixel value.
    _max : int
        Maximum pixel value.
    _view : np.ndarray
        View of the image data.
    _hist : np.ndarray or None
        Image histogram.
    n_bins : int or None
        Number of bins in the histogram.
    _cdf : np.ndarray or None
        Cumulative distribution function of the histogram.
    _stats : dict
        Image statistics.
    _pixel_w : float
        Width of a pixel.
    _pixel_h : float
        Height of a pixel.
    '''


    def __init__(self, image: np.ndarray, axis: str = 'CYX'):
        '''
        Initialize the uImage object.

        Parameters
        ----------
        image : np.ndarray
            Input image as a NumPy array.
        '''
        if axis == 'CYX' and image.ndim == 3:
            # Swap axes for CYX configuration
            image = np.swapaxes(image, 1, 2).swapaxes(0, 2)
        elif axis != 'YXC' and image.ndim == 3:
            raise ValueError("Invalid axis. Use 'YXC' or 'CYX'.")

        self._axis = 'YXC'  # Force the internal representation to 'YXC'
        self.image = image



        self._min = 0
        self._max = BYTE - 1 if image.dtype == np.uint8 else WORD - 1
        self._view = np.zeros(image.shape, dtype=np.uint8)
        self._hist = None
        self.n_bins = None
        self._cdf = None
        self._stats = {}
        self._pixel_w = 1.0
        self._pixel_h = 1.0

    def normalize_image(self):
        '''
        Normalize the input image.

        Returns
        -------
        np.ndarray
            Normalized image.
        '''
        if self._isfloat:
            if self._image.ndim == 3:
                # For YXC format, normalize along channels
                image_min = np.min(self._image, axis=(0, 1), keepdims=True)
                image_max = np.max(self._image, axis=(0, 1), keepdims=True)

                # Avoid division by zero
                non_zero_mask = (image_max != image_min).squeeze()

                normalized_image = np.zeros_like(self._image, dtype=np.uint16)

                # Perform normalization for each channel separately
                normalized_image[..., non_zero_mask] = (
                    (WORD - 1) * (self._image[..., non_zero_mask] - image_min) /
                    (image_max - image_min)
                ).astype(np.uint16)
            else:
                image_min = self._image.min()
                image_max = self._image.max()

                if image_max != image_min:
                    normalized_image = (
                        (2**16 - 1) * (
                            self._image - image_min) / (image_max - image_min)
                    ).astype(np.uint16)
                else:
                    normalized_image = np.zeros_like(self._image, dtype=np.uint16)
        else:
            normalized_image = None

        return normalized_image

    @property
    def image(self):
        '''
        Get the image data.

        Returns
        -------
        np.ndarray
            Image data.
        '''
        return self._image

    @image.setter
    def image(self, value: np.ndarray):
        '''
        Set the image data.

        Parameters
        ----------
        value : np.ndarray
            New image data.
        '''
        if value.dtype not in [np.float64, np.float32, np.uint16, np.uint8]:
            raise Exception(
                '''Image must be a numpy array of type np.float64,
                np.float32, np.uint16 or np.uint8.''')

        self._isfloat = (value.dtype == np.float16
                or value.dtype == np.float32
                or value.dtype == np.float64)

        if self._isfloat:
            self._image = value.astype(np.float32)
        else:
            self._image = value

        self._norm = self.normalize_image()
        if value.ndim > 3:
            raise ValueError('Unexpected image dimensions. Expected 2D or 3D array.')

    @property
    def width(self):
        '''
        Get the width of the image.

        Returns
        -------
        int
            Width of the image.
        '''
        if self._image.ndim == 3:
            return self._image.shape[2]
        else:
            return self._image.shape[1]

    @property
    def height(self):
        '''
        Get the height of the image.

        Returns
        -------
        int
            Height of the image.
        '''
        if self._image.ndim == 3:
            return self._image.shape[1]
        else:
            return self._image.shape[0]

    @property
    def channels(self):
        '''
        Get the number of channels in the image.

        Returns
        -------
        int
            Number of channels.
        '''
        if self._image.ndim == 3:
            return self._image.shape[2]
        else:
            return None

    @property
    def cdf_min(self) -> float:
        '''
        Get the minimum value for CDF calculation.

        Returns
        -------
        float
            Minimum value for CDF calculation.
        '''
        return 0.00001

    @property
    def cdf_max(self) -> float:
        '''
        Get the maximum value for CDF calculation.

        Returns
        -------
        float
            Maximum value for CDF calculation.
        '''
        return 0.999 if self._isfloat else 0.9999

    def norm_to_float(self, value):
        '''
        Convert a value from _norm to the original float value.

        Parameters
        ----------
        value : Union[int, np.ndarray]
            The normalized value(s) from _norm.

        Returns
        -------
        Union[float, np.ndarray]
            The original float value(s).
        '''
        if self._isfloat:
            image_min = self._image.min()
            image_max = self._image.max()

            if isinstance(value, (int, float)):
                return image_min + value * (image_max - image_min) / (WORD - 1)
            elif isinstance(value, np.ndarray):
                return image_min + value * (image_max - image_min) / (WORD - 1)
            else:
                raise ValueError(
                    'Input type not supported. Use int, float, or np.ndarray.')
        else:
            return value

    def update_range(self):
        '''
        Update the minimum and maximum pixel values based
        on the cumulative distribution function (CDF).
        '''
        if self._cdf.ndim == 1:
            self._min = np.where(self._cdf >= self.cdf_min)[0][0]
            self._max = np.where(self._cdf >= self.cdf_max)[0][0]
        elif self._cdf.ndim == 2:
            # Considering the cumulative distribution function across all channels
            min_indices = np.where(np.all(self._cdf >= self.cdf_min, axis=1))[0]
            max_indices = np.where(np.all(self._cdf >= self.cdf_max, axis=1))[0]

            # Update _min and _max based on all channels
            self._min, self._max = min(min_indices, default=0), min(
                max_indices, default=self._cdf.shape[0] - 1)

        # Ensure _max is greater than _min
        self._max = max(self._min + 1, self._max)

    def calcHist_GPU(self):
        '''
        Calculate the image histogram using GPU acceleration.
        '''
        image_to_use = self._norm if self._isfloat else self.image

        self.n_bins = BYTE if self._image.dtype == np.uint8 else WORD

        # Configure kernel launch parameters
        threads_per_block = (16, 16)
        blocks_per_grid_x = (
            image_to_use.shape[0] + threads_per_block[0] - 1
            ) // threads_per_block[0]
        blocks_per_grid_y = (
            image_to_use.shape[1] + threads_per_block[1] - 1
            ) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        if self.channels is None:
            # Allocate memory for the LUT on the GPU
            lut_device = cuda.to_device(np.zeros((self.n_bins,), dtype=np.int32))

            # Single-channel image
            generate_lut_single_channel_kernel[blocks_per_grid, threads_per_block](
                image_to_use, lut_device, 0, 1, self.n_bins)
        else:
            # Allocate memory for the LUT on the GPU
            lut_device = cuda.to_device(
                np.zeros((self.n_bins, self.channels), dtype=np.int32))
            # Multi-channel image
            generate_lut_multi_channel_kernel[blocks_per_grid, threads_per_block](
                image_to_use, lut_device, 0, 1, self.n_bins)

        # Copy the LUT from GPU device to host memory
        self._hist = lut_device.copy_to_host().astype(np.float64)

        # Normalize the LUT values
        self._hist = self._hist / float(
                    np.prod(image_to_use.shape[:2]))

        # calculate the cdf
        self._cdf = self._hist.cumsum(axis=0)

        self.update_range()

    def calcHist(self):
        '''
        Calculate the image histogram using CPU.
        '''
        image_to_use = self._norm if self._isfloat else self.image

        self.n_bins = BYTE if self._image.dtype == np.uint8 else WORD

        if image_to_use.ndim == 3 and self.channels > 1:
            # Multi-channel image
            self._hist = np.zeros((self.n_bins, self.channels), dtype=np.float32)
            for channel in range(self.channels):
                hist_channel = cv2.calcHist(
                    [image_to_use[:, :, channel]],
                    [0], None,
                    [self.n_bins],
                    [0, self.n_bins]
                )
                self._hist[:, channel] = hist_channel.squeeze() / float(
                    np.prod(image_to_use.shape[:2]))
            # Calculate CDF for each channel
            self._cdf = self._hist.cumsum(axis=0)
        else:
            # Calculate histogram for the entire image (single-channel)
            self._hist = cv2.calcHist(
                [image_to_use],
                [0], None,
                [self.n_bins],
                [0, self.n_bins]
            ).squeeze() / float(np.prod(image_to_use.shape))

            # Calculate CDF for the single channel
            self._cdf = self._hist.cumsum(axis=0).squeeze()

        self.update_range()

    def fastHIST(self):
        '''
        Calculate the image histogram using a fast method.
        '''
        image_to_use = self._norm if self._isfloat else self.image

        self.n_bins = BYTE if self._image.dtype == np.uint8 else WORD
        cv2.normalize(
            src=image_to_use, dst=self._view,
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if image_to_use.ndim == 3 and self.channels > 1:
            # Multi-channel image
            self._hist = np.zeros((self.n_bins, self.channels), dtype=np.float32)
            for channel in range(self.channels):
                hist_channel = cv2.calcHist(
                    [self._view[:, :, channel]],
                    [0], None,
                    [self.n_bins],
                    [0, self.n_bins]
                )
                self._hist[:, channel] = hist_channel.squeeze() / float(
                    np.prod(self._view.shape[:2]))
            # Calculate CDF for each channel
            self._cdf = self._hist.cumsum(axis=0)
        else:
            # Single-channel image
            # calculate image histogram
            self._hist = cv2.calcHist(
                [self._view], [0], None,
                [self.n_bins], [0, self.n_bins]
                ).squeeze() / float(np.prod(self._view.shape))
            # calculate the cdf
            self._cdf = self._hist.cumsum()

        self.update_range()

    def equalizeLUT(self, range=None, nLUT=False):
        '''
        Equalize the image using a Look-Up Table (LUT).

        Parameters
        ----------
        range : tuple or None, optional
            Range of values to consider for equalization.
        nLUT : bool, optional
            Flag indicating whether to use the normalized histogram for LUT creation.
        '''
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

            self._LUT = np.zeros((BYTE), dtype=np.uint8)
            self._LUT[self._min:self._max] = np.linspace(
                0, 255, self._max - self._min, dtype=np.uint8)
            self._LUT[self._max:] = 255

            cv2.LUT(self._view, self._LUT, self._view)

    def getStatistics(self):
        '''
        Calculate and update image statistics.
        '''
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
        '''
        Calculate standard deviation based on the count, sum, and sum of squares.

        Parameters
        ----------
        n : int
            Count of pixels.
        sum : float
            Sum of pixel values.
        sum_of_sq : float
            Sum of squares of pixel values.
        '''
        if n > 0.0:
            stdDev = sum_of_sq - (sum**2 / n)
            if stdDev > 0:
                self._stats['StdDev'] = np.sqrt(stdDev / (n - 1))
            else:
                self._stats['StdDev'] = 0.0
        else:
            self._stats['StdDev'] = 0.0

    def hsplitData(self):
        '''
        Split the image data horizontally and create a new uImage object.

        Returns
        -------
        uImage
            uImage object containing the horizontally split image data.
        '''
        mid = self._image.shape[1] // 2
        left_view = self._image[:, :mid]
        right_view = self._image[:, mid:]
        RGB_img = np.zeros(
            left_view.shape[:2] + (3,), dtype=np.uint8)

        RGB_img[..., 0] = left_view
        RGB_img[..., 1] = np.fliplr(right_view)

        return uImage(RGB_img)

    def hsplitViewOverlay(self, RGB=True) -> np.ndarray:
        '''
        Create a horizontally split view overlay.

        Parameters
        ----------
        RGB : bool, optional
            Flag indicating whether to create an RGB overlay, by default True.

        Returns
        -------
        np.ndarray
            Horizontally split view overlay.
        '''
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
        '''
        Split the image horizontally and create new uImage objects.

        Returns
        -------
        Tuple[uImage, uImage]
            Two uImage objects containing the horizontally split image data.
        '''
        mid = self.image.shape[1] // 2
        left_view = self.image[:, :mid]
        right_view = self.image[:, mid:]

        return uImage(left_view), uImage(np.fliplr(right_view))

    @staticmethod
    def fromUINT8(buffer, height, width):
        '''
        Create a uImage object from a UINT8 buffer.

        Parameters
        ----------
        buffer : object
            Buffer containing the image data.
        height : int
            Height of the image.
        width : int
            Width of the image.

        Returns
        -------
        uImage
            uImage object created from the UINT8 buffer.
        '''
        res = np.ndarray(shape=(height, width), dtype='u1', buffer=buffer)
        return uImage(res)

    @staticmethod
    def fromUINT16(buffer, height, width):
        '''
        Create a uImage object from a UINT16 buffer.

        Parameters
        ----------
        buffer : object
            Buffer containing the image data.
        height : int
            Height of the image.
        width : int
            Width of the image.

        Returns
        -------
        uImage
            uImage object created from the UINT16 buffer.
        '''
        res = np.ndarray(shape=(height, width), dtype='<u2', buffer=buffer)
        return uImage(res)

    @staticmethod
    def fromBuffer(buffer, height, width, bytes_per_pixel):
        '''
        Create a uImage object from a buffer.

        Parameters
        ----------
        buffer : object or None
            Buffer containing the image data.
        height : int
            Height of the image.
        width : int
            Width of the image.
        bytes_per_pixel : int
            Number of bytes per pixel.

        Returns
        -------
        uImage or np.ndarray
            uImage object created from the buffer or a zero-filled array.
        '''
        if buffer is not None:
            if bytes_per_pixel == 1:
                return uImage.fromUINT8(buffer, height, width)
            else:
                return uImage.fromUINT16(buffer, height, width)
        else:
            return np.zeros((height, width), dtype=np.uint16)


@cuda.jit
def generate_lut_single_channel_kernel(image, lut, min_value, bin_width, num_bins):
    row, col = cuda.grid(2)

    if row < image.shape[0] and col < image.shape[1]:
        # Calculate the bin index for the pixel value
        bin_index = int((image[row, col] - min_value) / bin_width)
        if bin_index >= num_bins:
            bin_index = num_bins - 1
        cuda.atomic.add(lut, bin_index, 1)

@cuda.jit
def generate_lut_multi_channel_kernel(image, lut, min_value, bin_width, num_bins):
    row, col = cuda.grid(2)
    channels = lut.shape[1]

    if row < image.shape[0] and col < image.shape[1]:
        for channel in range(channels):
            # Calculate the bin index for the pixel value in each channel
            bin_index = int(
                (image[row, col, channel] - min_value) / bin_width)
            if bin_index >= num_bins:
                bin_index = num_bins - 1
            cuda.atomic.add(lut, (bin_index, channel), 1)


class TiffSeqHandler:
    '''
    Class for handling TIFF sequences using tifffile and zarr libraries.

    Parameters
    ----------
    tiff_seq : tf.TiffSequence
        The TIFF sequence to be handled.

    Methods
    -------
    open()
        Opens the TIFF files and initializes necessary attributes.
    close()
        Closes the TIFF files and resets attributes.
    __getitem__(i)
        Gets an item or slice from the TIFF sequence.
    get_slice(...)
        Gets a slice from the TIFF sequence based on specified indices.
    __len__()
        Returns the total number of frames in the TIFF sequence.
    __enter__()
        Enters the context manager.
    __exit__(exc_type, exc_value, traceback)
        Exits the context manager.

    Attributes
    ----------
    shape : tuple
        Shape of the TIFF sequence.

    Raises
    ------
    ValueError
        If the shapes of TIFF files in the sequence do not match.
    '''

    def __init__(self, tiff_seq: tf.TiffSequence) -> None:
        '''
        Initializes the TiffSeqHandler object.

        Parameters
        ----------
        tiff_seq : tf.TiffSequence
            The TIFF sequence to be handled.
        '''
        self._tiff_seq = tiff_seq
        self._initialize_arrays()

    def _initialize_arrays(self):
        '''Initializes arrays and attributes.'''
        self._stores = [None] * len(self._tiff_seq.files)
        self._zarr = [None] * len(self._tiff_seq.files)
        self._frames = [None] * len(self._tiff_seq.files)
        self._data = None
        self._shape = None
        self._dtype = None
        self._cum_frames = None

    def open(self):
        '''
        Opens the TIFF files and initializes necessary attributes.

        Raises
        ------
        ValueError
            If the shapes of TIFF files do not match.
        '''
        for idx, file in enumerate(self._tiff_seq.files):
            self._stores[idx] = tf.imread(file, aszarr=True)
            self._zarr[idx] = zarr.open(self._stores[idx], mode='r')
            n_dim = len(self._zarr[idx].shape)
            if n_dim > 2:
                self._frames[idx] = self._zarr[idx].shape[0]
            else:
                self._zarr[idx] = self._zarr[idx][:, :][np.newaxis, ...]
                self._frames[idx] = 1

        # Check if shapes match
        first_shape = self._zarr[0].shape[1:]
        if any(shape != first_shape for shape in [
                arr.shape[1:] for arr in self._zarr[1:]]):
            raise ValueError('Shapes of TIFF files do not match.')

        self._update_properties()

    def close(self):
        '''
        Closes the TIFF files and resets attributes.
        '''
        for store in self._stores:
            if store is not None:
                store.close()
        self._tiff_seq.close()
        self._initialize_arrays()

    def _update_properties(self):
        '''Updates properties such as shape, dtype, and cumulative frames.'''
        self._shape = (sum(self._frames),) + self._zarr[0].shape[1:]
        self._dtype = self._zarr[0].dtype
        self._cum_frames = np.cumsum(self._frames)

    def __getitem__(self, i):
        '''
        Gets an item or slice from the TIFF sequence.

        Parameters
        ----------
        i : int, slice
            Index or slice to retrieve.

        Returns
        -------
        np.ndarray
            Retrieved data.
        '''
        if isinstance(i, int):
            return self._get_slice(slice(i, i + 1))
        elif isinstance(i, slice):
            return self._get_slice(i)
        else:
            return self._get_slice(slice(None))

    def _get_file_and_adjust_index(self, i):
        '''
        Determines the file index and adjusts the index for retrieval.

        Parameters
        ----------
        i : int
            Index to adjust.

        Returns
        -------
        tuple
            File index and adjusted index.
        '''
        file_idx = 0
        for idx, cum_frames in enumerate(self._cum_frames):
            if i <= cum_frames - 1:
                file_idx = idx
                i -= (cum_frames - self._frames[idx])
                break
        return file_idx, i

    def _get_slice(self, i):
        '''
        Retrieves a slice from the TIFF sequence.

        Parameters
        ----------
        i : slice
            Slice to retrieve.

        Returns
        -------
        np.ndarray
            Retrieved data.
        '''
        start = 0 if i.start is None else i.start
        stop = sum(self._frames) if start is None else i.stop
        if stop <= self._cum_frames[0]:
            indices = slice(start, stop)
            return self._zarr[0][indices]
        else:
            return self._get_concatenated_slice(i)

    def _get_concatenated_slice(self, i):
        '''
        Retrieves a concatenated slice from multiple TIFF files.

        Parameters
        ----------
        i : slice
            Slice to retrieve.

        Returns
        -------
        np.ndarray
            Retrieved data.
        '''
        indices = np.arange(i.start or 0, i.stop)
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
            self, timeSlice=None, channelSlice=None,
            zSlice=None, ySlice=None, xSlice=None,
            squeezed=True, broadcasted=False,
            four='TCYX', three='TYX'):
        '''
        Retrieves a slice from the Zarr array based on specified indices.

        Parameters
        ----------
        timeSlice : slice or None
            Slice for the time dimension.
        channelSlice : slice or None
            Slice for the channel dimension.
        zSlice : slice or None
            Slice for the z dimension.
        ySlice : slice or None
            Slice for the y dimension.
        xSlice : slice or None
            Slice for the x dimension.
        squeezed : bool (optional)
            Squeeze returned slice, default is True.
        broadcasted : bool (optional)
            Broad cast returned slice according to TCZYX, default is False.
        four : str
            String representing the axis configuration for four dimensions.
        three : str
            String representing the axis configuration for three dimensions.

        Returns
        -------
        np.ndarray
            Retrieved slice.
        '''
        t = ifnone(timeSlice, slice(None))
        c = ifnone(channelSlice, slice(None))
        z = ifnone(zSlice, slice(None))
        y = ifnone(ySlice, slice(None))
        x = ifnone(xSlice, slice(None))

        if self.shape is None:
            raise ValueError(f'The handler was not initializd correctly.')

        ndim = len(self._shape)
        data = None

        if ndim == 5:
            data = self[t][
                ...,
                c, z, y, x]
            new_slice = (slice(None),) * 5
        elif ndim == 4:
            if four == 'TCYX':
                data = self[t][
                    ...,
                    c, y, x]
                new_slice = (slice(None),) * 2 + (np.newaxis,) + (slice(None),) * 2
            elif four == 'CZYX':
                data = self[c][
                    ...,
                    z, y, x]
                new_slice = (np.newaxis,) + (slice(None),) * 4
            elif four == 'TZYX':
                data = self[t][
                    ...,
                    z, y, x]
                new_slice = (slice(None), np.newaxis,) + (slice(None),) * 3
            else:
                raise ValueError(
                    f'Unsupported dimensions format: {four}')
        elif ndim == 3:
            if three == 'TYX':
                data = self[t][
                    ...,
                    y, x]
                new_slice = (slice(None), np.newaxis, np.newaxis,) + (slice(None),) * 2
            elif three == 'CYX':
                data = self[c][
                    ...,
                    y, x]
                new_slice = (np.newaxis, slice(None), np.newaxis,) + (slice(None),) * 2
            elif three == 'ZYX':
                data = self[z][
                    ...,
                    y, x]
                new_slice = (np.newaxis, np.newaxis,) + (slice(None),) * 3
            else:
                raise ValueError(
                    f'Unsupported dimensions format: {three}')
        elif ndim == 2:
                data = self[0][
                    y, x]
                new_slice = (np.newaxis,) * 3 + (slice(None),) * 2
        else:
            raise ValueError(
                f'Unsupported number of dimensions: {len(self._shape)}')

        if broadcasted:
            return data[new_slice]
        else:
            if squeezed:
                return data.squeeze()
            else:
                return data

    def __len__(self):
        '''
        Returns the total number of frames in the TIFF sequence.

        Returns
        -------
        int
            Total number of frames.
        '''
        return sum(self._frames)

    @property
    def shape(self):
        '''
        Gets the shape of the TIFF sequence.

        Returns
        -------
        tuple
            Shape of the TIFF sequence.
        '''
        return self._shape if self._zarr is not None else None

    def shapeTCZYX(self, four='TCYX', three='TYX'):
        '''
        Returns the shape of the Zarr array.

        Returns
        -------
        tuple or None
            Shape of the Zarr array.
        '''
        if self._shape:
            if len(self._shape) == 5:
                return self._shape
            elif len(self._shape) == 4:
                if four == 'TCYX':
                    return (
                        self._shape[0], self._shape[1],
                        1,
                        self._shape[2], self._shape[3])
                elif four == 'CZYX':
                    return (
                        1,
                        self._shape[0], self._shape[1],
                        self._shape[2], self._shape[3])
                elif four == 'TZYX':
                    return (
                        self._shape[0],
                        1, self._shape[1],
                        self._shape[2], self._shape[3])
                else:
                    raise ValueError(
                        f'Unsupported dimensions format: {four}')
            elif len(self._shape) == 3:
                if three == 'TYX':
                    return (
                        self._shape[0], 1, 1,
                        self._shape[1], self._shape[2])
                elif three == 'CYX':
                    return (
                        1, self._shape[0], 1,
                        self._shape[1], self._shape[2])
                elif three == 'ZYX':
                    return (
                        1, 1, self._shape[0],
                        self._shape[1], self._shape[2])
                else:
                    raise ValueError(
                        f'Unsupported dimensions format: {three}')
            elif len(self._shape) == 2:
                return (
                    1, 1, 1,
                    self._shape[0], self._shape[1])
            else:
                raise ValueError(
                    f'Unsupported number of dimensions: {len(self._shape)}')
        else:
            raise ValueError(f'The handler was not initializd correctly.')


    def __enter__(self):
        '''
        Enters the context manager.

        Returns
        -------
        TiffSeqHandler
            The TiffSeqHandler object.
        '''
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        '''
        Exits the context manager.

        Parameters
        ----------
        exc_type : type
            Exception type.
        exc_value : Exception
            Exception instance.
        traceback : Traceback
            Traceback object.
        '''
        self.close()

class ZarrImageSequence:
    '''
    A class for handling image sequences stored in Zarr format.

    Parameters
    ----------
    path : str
        The path to the Zarr store.

    Attributes
    ----------
    path : str
        The path to the Zarr store.
    data : zarr.Array or None
        The Zarr array containing the image sequence data.
    '''

    def __init__(self, path: str) -> None:
        '''
        Initializes the ZarrImageSequence object.

        Parameters
        ----------
        path : str
            The path to the Zarr store.
        '''
        self.path = path
        self.data = None

    def __getitem__(self, i):
        '''
        Retrieves a specific item or slice from the Zarr array.

        Parameters
        ----------
        i : Index or slice

        Returns
        -------
        np.ndarray or None
            Retrieved data.
        '''
        if self.data is None:
            return None
        return zarr.open(self.path, 'r').__getitem__(i)

    def getSlice(
            self, timeSlice=None, channelSlice=None,
            zSlice=None, ySlice=None, xSlice=None, squeezed=True,
            four='TCYX', three='TYX'):
        '''
        Retrieves a slice from the Zarr array based on specified indices.

        Parameters
        ----------
        timeSlice : slice or None
            Slice for the time dimension.
        channelSlice : slice or None
            Slice for the channel dimension.
        zSlice : slice or None
            Slice for the z dimension.
        ySlice : slice or None
            Slice for the y dimension.
        xSlice : slice or None
            Slice for the x dimension.
        squeezed : bool (optional)
            Squeeze returned slice, default is True.
        four : str
            String representing the axis configuration for four dimensions.
        three : str
            String representing the axis configuration for three dimensions.

        Returns
        -------
        np.ndarray
            Retrieved slice.
        '''
        za = zarr.open(self.path, 'r')

        # Handle None values and replace with default slices
        timeSlice = ifnone(timeSlice, slice(None))
        channelSlice = ifnone(channelSlice, slice(None))
        zSlice = ifnone(zSlice, slice(None))
        ySlice = ifnone(ySlice, slice(None))
        xSlice = ifnone(xSlice, slice(None))

        if len(za.shape) == 5:
            data = za[timeSlice, channelSlice, zSlice, ySlice, xSlice]
        elif len(za.shape) == 4:
            if four == 'TCYX':
                data = za[timeSlice, channelSlice, ySlice, xSlice]
            elif four == 'CZYX':
                data = za[channelSlice, zSlice, ySlice, xSlice]
            elif four == 'TZYX':
                data = za[timeSlice, zSlice, ySlice, xSlice]
            else:
                raise ValueError(f'Unsupported dimensions format: {four}')
        elif len(za.shape) == 3:
            if three == 'TYX':
                data = za[timeSlice, ySlice, xSlice]
            elif three == 'CYX':
                data = za[channelSlice, ySlice, xSlice]
            elif three == 'ZYX':
                data = za[zSlice, ySlice, xSlice]
            else:
                raise ValueError(f'Unsupported dimensions format: {three}')
        elif len(za.shape) == 2:
            data = za[ySlice, xSlice]
        else:
            raise ValueError(f'Unsupported number of dimensions: {len(za.shape)}')

        if squeezed:
            return data.squeeze()
        else:
            return data

    def open(self):
        """
        Opens the Zarr array and assigns it to the 'data' attribute.
        """
        self.data = zarr.open(self.path, 'r')

    def close(self):
        """
        Closes the Zarr array by deleting the 'data' attribute.
        """
        del self.data

    def __len__(self):
        '''
        Returns the length of the Zarr array.

        Returns
        -------
        int
            Length of the Zarr array.
        '''
        return 0 if self.data is None else self.data.shape[0]

    @property
    def shape(self):
        '''
        Returns the shape of the Zarr array.

        Returns
        -------
        tuple or None
            Shape of the Zarr array.
        '''
        return None if self.data is None else self.data.shape

    def shapeTCZYX(self, four='TCYX', three='TYX'):
        '''
        Returns the shape of the Zarr array.

        Returns
        -------
        tuple or None
            Shape of the Zarr array.
        '''
        if self.data:
            if len(self.data.shape) == 5:
                return self.data.shape
            elif len(self.data.shape) == 4:
                if four == 'TCYX':
                    return (
                        self.data.shape[0], self.data.shape[1],
                        1,
                        self.data.shape[2], self.data.shape[3])
                elif four == 'CZYX':
                    return (
                        1,
                        self.data.shape[0], self.data.shape[1],
                        self.data.shape[2], self.data.shape[3])
                elif four == 'TZYX':
                    return (
                        self.data.shape[0],
                        1, self.data.shape[1],
                        self.data.shape[2], self.data.shape[3])
                else:
                    raise ValueError(
                        f'Unsupported dimensions format: {four}')
            elif len(self.data.shape) == 3:
                if three == 'TYX':
                    return (
                        self.data.shape[0], 1, 1,
                        self.data.shape[1], self.data.shape[2])
                elif three == 'CYX':
                    return (
                        1, self.data.shape[0], 1,
                        self.data.shape[1], self.data.shape[2])
                elif three == 'ZYX':
                    return (
                        1, 1, self.data.shape[0],
                        self.data.shape[1], self.data.shape[2])
                else:
                    raise ValueError(
                        f'Unsupported dimensions format: {three}')
            elif len(self.data.shape) == 2:
                return (
                    1, 1, 1,
                    self.data.shape[0], self.data.shape[1])
            else:
                raise ValueError(
                    f'Unsupported number of dimensions: {len(self.data.shape)}')
        else:
            raise ValueError(f'The zarr array was closed.')

def ifnone(a, b):
    return b if a is None else a

def saveZarrImage(
        path: str,
        imgSeq: Union[TiffSeqHandler, ZarrImageSequence],
        timeSlice: slice = None,
        channelSlice: slice = None,
        zSlice: slice = None,
        ySlice: slice = None,
        xSlice: slice = None):
    '''
    Saves an image sequence represented by either a TiffSeqHandler
    or ZarrImageSequence to a Zarr store.

    Parameters
    ----------
    path : str
        The path to the Zarr store.
    imgSeq : TiffSeqHandler or ZarrImageSequence
        The image sequence to save.
    timeSlice : slice or None
        Slice for the time dimension.
    channelSlice : slice or None
        Slice for the channel dimension.
    zSlice : slice or None
        Slice for the z dimension.
    ySlice : slice or None
        Slice for the y dimension.
    xSlice : slice or None
        Slice for the x dimension.

    Returns
    -------
    bool
        True if the save operation is successful, False otherwise.
    '''
    # Handle None values and replace with default slices
    timeSlice = ifnone(timeSlice, slice(None))
    channelSlice = ifnone(channelSlice, slice(None))
    zSlice = ifnone(zSlice, slice(None))
    ySlice = ifnone(ySlice, slice(None))
    xSlice = ifnone(xSlice, slice(None))

    if isinstance(imgSeq, TiffSeqHandler):
        ndim = len(imgSeq.shape)
        if ndim == 2:
            shape = (
                1,
                1,
                1,
                ifnone(ySlice.stop, imgSeq.shape[0]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[1]) - ifnone(xSlice.start, 0)
            )
            chunks = (1, 1, 1, shape[3], shape[4])
        elif ndim == 3:
            shape = (
                ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
                1,
                1,
                ifnone(ySlice.stop, imgSeq.shape[1]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[2]) - ifnone(xSlice.start, 0)
            )
            chunks = (
                min(10, shape[0]),
                1,
                1,
                shape[3],
                shape[4]
            )
        elif ndim == 4:
            shape = (
                ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
                ifnone(
                    channelSlice.stop, imgSeq.shape[1]) - ifnone(channelSlice.start, 0),
                1,
                ifnone(ySlice.stop, imgSeq.shape[2]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[3]) - ifnone(xSlice.start, 0)
            )
            chunks = (
                min(10, shape[0]),
                min(10, shape[1]),
                1,
                shape[3],
                shape[4]
            )
        elif ndim == 5:
            shape = (
                ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
                ifnone(
                    channelSlice.stop, imgSeq.shape[1]) - ifnone(channelSlice.start, 0),
                ifnone(zSlice.stop, imgSeq.shape[2]) - ifnone(zSlice.start, 0),
                ifnone(ySlice.stop, imgSeq.shape[3]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[4]) - ifnone(xSlice.start, 0)
            )
            chunks = (
                min(10, shape[0]),
                min(10, shape[1]),
                min(10, shape[2]),
                shape[3],
                shape[4]
            )
        else:
            raise ValueError(f'Unsupported number of dimensions: {ndim}')

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

            # Adjust the tiffSlice based on the offset
            tiffSlice = slice(
                max(
                    zarrSlice.start - offset, 0
                ),
                min(
                    zarrSlice.stop - offset,
                    imgSeq._zarr[idx].shape[0]
                )
            )

            # Use tuple unpacking to apply the slices to the image
            print('Saving ...', end='\r')
            zarrImg[zarrSlice, ...] = imgSeq.getSlice(
                tiffSlice, channelSlice, zSlice, ySlice, xSlice,
                broadcasted=True)

        print('Done ...', end='\r')
        return True
    elif isinstance(imgSeq, ZarrImageSequence):
        print('Saving ...', end='\r')
        shape = (
            ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
            ifnone(
                channelSlice.stop, imgSeq.shape[1]) - ifnone(channelSlice.start, 0),
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

        print('Done ...', end='\r')
        return True
    else:
        print('Failed ...', end='\r')
        return False
