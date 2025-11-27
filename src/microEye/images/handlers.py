import logging
import os
from typing import Optional, Union

import numpy as np
import ome_types as ome
import ome_types.model as ome_models
import tifffile as tf
import zarr
import zarr.storage

from microEye import __version__

logger = logging.getLogger(__name__)

class ImageSequenceBase:
    '''
    A base class for handling image sequences.

    Attributes
    ----------
    shape : tuple or None
        Shape of the image sequence.
    dtype : np.dtype or None
        Data type of the image sequence.
    '''

    def __init__(self):
        '''
        Initializes the ImageSequenceBase object.
        '''
        self._shape = None
        self._dtype = None
        self._path = ''

    @property
    def path(self) -> str:
        '''
        Get the path of the image sequence.

        Returns
        -------
        str
            Path of the image sequence.
        '''
        return self._path

    @path.setter
    def path(self, value: str):
        '''
        Set the path of the image sequence.

        Parameters
        ----------
        value : str
            New path of the image sequence.
        '''
        self._path = value

    def __getitem__(self, i):
        '''
        Retrieves a specific item or slice from the image sequence.

        Parameters
        ----------
        i : Index or slice

        Returns
        -------
        np.ndarray or None
            Retrieved data.
        '''
        if isinstance(i, slice):
            return self.getSlice(i)
        elif isinstance(i, int):
            return self.getSlice(slice(i, i + 1, 1))
        else:
            raise IndexError('Index must be an integer or a slice')

    def getSlice(
        self,
        timeSlice=None,
        channelSlice=None,
        zSlice=None,
        ySlice=None,
        xSlice=None,
        squeezed=True,
        four='TCYX',
        three='TYX',
    ) -> Optional[np.ndarray]:
        '''
        Retrieves a slice from the image sequence based on specified indices.

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
        raise NotImplementedError('This method should be implemented by subclasses.')

    def open(self):
        '''
        Opens the image sequence and initializes the data structure.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        '''
        raise NotImplementedError('This method should be implemented by subclasses.')

    def close(self):
        '''
        Closes the image sequence and releases any resources.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        '''
        raise NotImplementedError('This method should be implemented by subclasses.')

    def __len__(self):
        '''
        Returns the length of the image sequence.

        Returns
        -------
        int
            Length of the image sequence.
        '''
        if self._shape is None:
            return 0
        return self._shape[0]

    @property
    def shape(self):
        '''
        Returns the shape of the image sequence.

        Returns
        -------
        tuple or None
            Shape of the image sequence.
        '''
        if self._shape is None:
            return None
        return self._shape

    def shapeTCZYX(self, four='TCYX', three='TYX'):
        '''
        Returns the shape of the image sequence in a specific format.

        Returns
        -------
        tuple or None
            Shape of the image sequence.
        '''
        if self._shape:
            if len(self._shape) == 5:
                return self._shape
            elif len(self._shape) == 4:
                if four == 'TCYX':
                    return (
                        self._shape[0],
                        self._shape[1],
                        1,
                        self._shape[2],
                        self._shape[3],
                    )
                elif four == 'CZYX':
                    return (
                        1,
                        self._shape[0],
                        self._shape[1],
                        self._shape[2],
                        self._shape[3],
                    )
                elif four == 'TZYX':
                    return (
                        self._shape[0],
                        1,
                        self._shape[1],
                        self._shape[2],
                        self._shape[3],
                    )
                else:
                    raise ValueError(f'Unsupported dimensions format: {four}')
            elif len(self._shape) == 3:
                if three == 'TYX':
                    return (self._shape[0], 1, 1, self._shape[1], self._shape[2])
                elif three == 'CYX':
                    return (1, self._shape[0], 1, self._shape[1], self._shape[2])
                elif three == 'ZYX':
                    return (1, 1, self._shape[0], self._shape[1], self._shape[2])
                else:
                    raise ValueError(f'Unsupported dimensions format: {three}')
            elif len(self._shape) == 2:
                return (1, 1, 1, self._shape[0], self._shape[1])
            else:
                raise ValueError(
                    f'Unsupported number of dimensions: {len(self._shape)}'
                )
        else:
            return None

    def minimal_metadata(
        self, name: str, description: str, channels: int = None
    ) -> dict:
        ''' '''
        ome_obj = ome.OME(creator=f'microEye Python Package v{__version__}')

        channel = ome_models.Channel()

        shape = self.shapeTCZYX()

        dtype = str(np.dtype(self._dtype))

        dtype = {
            'float32': 'float',
            'float64': 'double',
            'complex64': 'complex',
            'complex128': 'double-complex',
            'bool': 'bit',
        }.get(dtype, dtype)

        pixels = ome_models.Pixels(
            size_c=shape[1] if channels is None else channels,
            size_t=shape[0],
            size_x=shape[4],
            size_y=shape[3],
            size_z=shape[2],
            type=ome_models.PixelType._member_map_[dtype],
            dimension_order='TCZYX',
            physical_size_x_unit=ome_models.UnitsLength.MICROMETER,
            physical_size_y_unit=ome_models.UnitsLength.MICROMETER,
            time_increment_unit=ome_models.UnitsTime.MILLISECOND,
        )
        pixels.tiff_data_blocks.append(ome_models.TiffData())
        pixels.channels.append(channel)

        planes = [
            ome_models.Plane(
                the_c=0,
                the_t=i,
                the_z=0,
                # exposure_time=self.get_param_value(MetaParams.EXPOSURE),
                exposure_time_unit=ome_models.UnitsTime.MILLISECOND,
            )
            for i in range(shape[0])
        ]
        pixels.planes.extend(planes)

        img = ome_models.Image(
            id='Image:1',
            name=name,
            pixels=pixels,
            description=description,
        )
        ome_obj.images.append(img)

        ome_obj = ome.OME.model_validate(ome_obj)

        return ome_obj.to_xml()

    def __enter__(self):
        '''
        Enters the context manager.

        Returns
        -------
        ImageSequenceBase
            ImageSequenceBase object.
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


class TiffSeqHandler(ImageSequenceBase):
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
        super().__init__()

        self._tiff_seq = tiff_seq
        self.path = ','.join(self._tiff_seq.files)
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
            self._zarr[idx] = zarr.open(store=self._stores[idx], mode='r')
            n_dim = len(self._zarr[idx].shape)
            if n_dim > 2:
                self._frames[idx] = self._zarr[idx].shape[0]
            else:
                self._zarr[idx] = self._zarr[idx][:, :][np.newaxis, ...]
                self._frames[idx] = 1

        # Check if shapes match
        first_shape = self._zarr[0].shape[1:]
        if any(
            shape != first_shape for shape in [arr.shape[1:] for arr in self._zarr[1:]]
        ):
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
        if isinstance(i, slice):
            return self._get_slice(i)
        elif isinstance(i, int) or np.issubdtype(i, np.integer):
            return self._get_slice(slice(i, i + 1, 1))
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
                i -= cum_frames - self._frames[idx]
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
        result = np.empty(shape=(0,) + self._zarr[0].shape[1:], dtype=self._dtype)
        for idx, cum_frames in enumerate(self._cum_frames):
            mask = np.logical_and(
                cum_frames - self._frames[idx] <= indices, indices < cum_frames
            )
            if np.sum(mask) > 0:
                r = indices[mask] - (cum_frames - self._frames[idx])
                result = np.concatenate(
                    (result, self._zarr[idx][np.min(r) : np.max(r) + 1]), axis=0
                )
        return result

    def getSlice(
        self,
        timeSlice=None,
        channelSlice=None,
        zSlice=None,
        ySlice=None,
        xSlice=None,
        squeezed=True,
        broadcasted=False,
        four='TCYX',
        three='TYX',
    ):
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
            data = self[t][..., c, z, y, x]
            new_slice = (slice(None),) * 5
        elif ndim == 4:
            if four == 'TCYX':
                data = self[t][..., c, y, x]
                new_slice = (slice(None),) * 2 + (np.newaxis,) + (slice(None),) * 2
            elif four == 'CZYX':
                data = self[c][..., z, y, x]
                new_slice = (np.newaxis,) + (slice(None),) * 4
            elif four == 'TZYX':
                data = self[t][..., z, y, x]
                new_slice = (
                    slice(None),
                    np.newaxis,
                ) + (slice(None),) * 3
            else:
                raise ValueError(f'Unsupported dimensions format: {four}')
        elif ndim == 3:
            if three == 'TYX':
                data = self[t][..., y, x]
                new_slice = (
                    slice(None),
                    np.newaxis,
                    np.newaxis,
                ) + (slice(None),) * 2
            elif three == 'CYX':
                data = self[c][..., y, x]
                new_slice = (
                    np.newaxis,
                    slice(None),
                    np.newaxis,
                ) + (slice(None),) * 2
            elif three == 'ZYX':
                data = self[z][..., y, x]
                new_slice = (
                    np.newaxis,
                    np.newaxis,
                ) + (slice(None),) * 3
            else:
                raise ValueError(f'Unsupported dimensions format: {three}')
        elif ndim == 2:
            data = self[0][y, x]
            new_slice = (np.newaxis,) * 3 + (slice(None),) * 2
        else:
            raise ValueError(f'Unsupported number of dimensions: {len(self._shape)}')

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


class ZarrImageSequence(ImageSequenceBase):
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
        super().__init__()
        self.path = path
        self.data = None

    def getSlice(
        self,
        timeSlice=None,
        channelSlice=None,
        zSlice=None,
        ySlice=None,
        xSlice=None,
        squeezed=True,
        four='TCYX',
        three='TYX',
    ):
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
        store = (
            zarr.storage.LocalStore(self.path)
            if os.path.isdir(self.path)
            else zarr.storage.ZipStore(self.path)
        )
        za = zarr.open_array(store=store)

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

        del za
        if squeezed:
            return data.squeeze()
        else:
            return data

    def open(self):
        '''
        Opens the zarr file.
        '''
        store = (
            zarr.storage.LocalStore(self.path)
            if os.path.isdir(self.path)
            else zarr.storage.ZipStore(self.path)
        )
        data = zarr.open_array(store=store)
        self._shape = data.shape
        self._dtype = data.dtype

    def close(self):
        '''
        Closes the zarr file.
        '''
        pass


def ifnone(a, b):
    return b if a is None else a


def saveZarrImage(
    path: str,
    imgSeq: Union[TiffSeqHandler, ZarrImageSequence],
    timeSlice: slice = None,
    channelSlice: slice = None,
    zSlice: slice = None,
    ySlice: slice = None,
    xSlice: slice = None,
):
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
                ifnone(xSlice.stop, imgSeq.shape[1]) - ifnone(xSlice.start, 0),
            )
            chunks = (1, 1, 1, shape[3], shape[4])
        elif ndim == 3:
            shape = (
                ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
                1,
                1,
                ifnone(ySlice.stop, imgSeq.shape[1]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[2]) - ifnone(xSlice.start, 0),
            )
            chunks = (min(10, shape[0]), 1, 1, shape[3], shape[4])
        elif ndim == 4:
            shape = (
                ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
                ifnone(channelSlice.stop, imgSeq.shape[1])
                - ifnone(channelSlice.start, 0),
                1,
                ifnone(ySlice.stop, imgSeq.shape[2]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[3]) - ifnone(xSlice.start, 0),
            )
            chunks = (min(10, shape[0]), min(10, shape[1]), 1, shape[3], shape[4])
        elif ndim == 5:
            shape = (
                ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
                ifnone(channelSlice.stop, imgSeq.shape[1])
                - ifnone(channelSlice.start, 0),
                ifnone(zSlice.stop, imgSeq.shape[2]) - ifnone(zSlice.start, 0),
                ifnone(ySlice.stop, imgSeq.shape[3]) - ifnone(ySlice.start, 0),
                ifnone(xSlice.stop, imgSeq.shape[4]) - ifnone(xSlice.start, 0),
            )
            chunks = (
                min(10, shape[0]),
                min(10, shape[1]),
                min(10, shape[2]),
                shape[3],
                shape[4],
            )
        else:
            raise ValueError(f'Unsupported number of dimensions: {ndim}')

        zarrImg = zarr.open(
            store=path,
            mode='w-',
            shape=shape,
            chunks=chunks,
            compressor=None,
            dtype=imgSeq._dtype,
        )

        timeSlice = slice(ifnone(timeSlice.start, 0), shape[0])

        for idx in np.arange(len(imgSeq._zarr)):
            offset = imgSeq._cum_frames[idx] - imgSeq._frames[idx]
            zarrSlice = slice(
                max(
                    timeSlice.start,
                    offset,
                ),
                min(timeSlice.stop, imgSeq._cum_frames[idx]),
            )

            # Adjust the tiffSlice based on the offset
            tiffSlice = slice(
                max(zarrSlice.start - offset, 0),
                min(zarrSlice.stop - offset, imgSeq._zarr[idx].shape[0]),
            )

            # Use tuple unpacking to apply the slices to the image
            logger.info('Saving ...')
            zarrImg[zarrSlice, ...] = imgSeq.getSlice(
                tiffSlice, channelSlice, zSlice, ySlice, xSlice, broadcasted=True
            )

        logger.info('Done ...')
        return True
    elif isinstance(imgSeq, ZarrImageSequence):
        logger.info('Saving ...')
        shape = (
            ifnone(timeSlice.stop, imgSeq.shape[0]) - ifnone(timeSlice.start, 0),
            ifnone(channelSlice.stop, imgSeq.shape[1]) - ifnone(channelSlice.start, 0),
            ifnone(zSlice.stop, imgSeq.shape[2]) - ifnone(zSlice.start, 0),
            ifnone(ySlice.stop, imgSeq.shape[3]) - ifnone(ySlice.start, 0),
            ifnone(xSlice.stop, imgSeq.shape[4]) - ifnone(xSlice.start, 0),
        )
        chunks = (
            min(10, shape[0]),
            min(10, shape[1]),
            min(10, shape[2]),
            shape[3],
            shape[4],
        )
        zarrImg = zarr.open(
            store=path,
            mode='w-',
            shape=shape,
            chunks=chunks,
            compressor=None,
            dtype=imgSeq._dtype,
        )
        zarrImg[:] = imgSeq.getSlice(timeSlice, channelSlice, zSlice, ySlice, xSlice)

        logger.info('Done ...')
        return True
    else:
        logger.error('Failed ...')
        return False
