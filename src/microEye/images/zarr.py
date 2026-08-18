import logging
import os
import shutil
from pathlib import Path

import numpy as np
import zarr
import zarr.storage

# from zarr.api import asynchronous

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.handlers[0].setFormatter(
    logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
)


class ZarrAcquisition:
    def __init__(self, filename: str, max_frames, channels, depth, height, width):
        self.filename = filename
        self.max_frames = max_frames
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        self.actual_frames = 0

        # Create array for streaming acquisition (no sharding)
        self.zarr_array = zarr.create_array(
            self.filename,
            zarr_format=3,
            shape=(max_frames, channels, depth, height, width),
            chunks=(1, 1, 1, height, width),
            dtype=np.uint16,
        )

    def write_frame(self, frame_idx, channel_idx, z_idx, frame):
        '''Write a single frame during acquisition.'''
        self.zarr_array[frame_idx, channel_idx, z_idx] = frame
        self.actual_frames = max(self.actual_frames, frame_idx + 1)

    def finalize(
        self
    ):
        '''
        Finalize the acquisition by trimming unused frames and
        optionally adding sharding.

        Parameters
        -----------
        trim_zeros : bool
            Remove surplus zero frames from pre-allocated array
        add_sharding : bool
            Create a sharded copy for long-term storage
        shard_size : tuple
            Shard dimensions (T, C, Z, Y, X). Use None to keep original dimension.
        '''
        logger.info(
            'Finalizing acquisition.'
            f' Actual frames: {self.actual_frames}/{self.max_frames}'
        )

        # Step 1: Trim surplus frames if needed
        logger.info(f'Storing Array {self.actual_frames}/{self.max_frames} frames...')
        self._store_array()

    def _store_array(self):
        '''Trim the array to actual acquired frames.'''
        temp_path = f'{self.filename}_temp'

        # Create new array with correct size
        store = zarr.storage.ZipStore(path=temp_path, mode='w')
        trimmed = zarr.create_array(
            store=store,
            zarr_format=3,
            shape=(
                self.actual_frames,
                self.channels,
                self.depth,
                self.height,
                self.width,
            ),
            chunks=(1, 1, 1, self.height, self.width),
            dtype=np.uint16,
        )

        # Copy actual data
        trimmed[:] = self.zarr_array[: self.actual_frames]

        store.close()
        self.zarr_array.store.close()

        logger.info(f'  Original size: {self._get_dir_size(self.filename):.1f} MB')
        logger.info(f'  Stored size: {self._get_dir_size(temp_path):.1f} MB')

        # Replace original with trimmed version
        shutil.rmtree(self.filename)
        shutil.move(temp_path, self.filename)

        # Reopen the trimmed array
        self.zarr_array = trimmed

    def _get_dir_size(self, path):
        '''Get total size of directory in MB.'''
        if os.path.isfile(path):
            return os.path.getsize(path) / (1024 * 1024)

        total = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())
        return total / (1024 * 1024)


def create_array(
    filename: str,
    shape: tuple,
    chunks: tuple = (1, 1, 1, 512, 512),
    dtype=np.uint16,
    is_zip: bool = True,
):
    '''
    Create an empty zarr array.

    Parameters
    -----------
    filename : str
        Output zarr filename (directory or .zip).
    shape : tuple
        Shape of the array.
    chunks : tuple
        Chunk size for zarr storage.
    dtype : data-type
        Data type for storage.
    is_zip : bool
        Whether to store as a zip file or directory.
    '''
    if os.path.exists(filename):
        raise FileExistsError(f'File {filename} already exists.')

    store = (
        zarr.storage.ZipStore(filename, mode='w')
        if is_zip
        else zarr.storage.LocalStore(filename)
    )


    zarr_array = zarr.create_array(
        store=store,
        zarr_format=3,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        overwrite=False,
    )
    return zarr_array


def store_zarr_array(
    array: np.ndarray,
    filename: str,
    chunks: tuple = (1, 1, 1, 512, 512),
    dtype=np.uint16,
    is_zip: bool = True,
):
    '''
    Store a numpy array as a zarr file.

    Parameters
    -----------
    array : np.ndarray
        Input array to store.
    filename : str
        Output zarr filename (directory or .zip).
    chunks : tuple
        Chunk size for zarr storage.
    dtype : data-type
        Data type for storage.
    is_zip : bool
        Whether to store as a zip file or directory.
    '''
    zarr_array = create_array(
        filename=filename, shape=array.shape, chunks=chunks, dtype=dtype, is_zip=is_zip
    )
    zarr_array[:] = array

    zarr_array.store.close()
