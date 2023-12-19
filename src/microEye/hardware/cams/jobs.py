import json
import os
import threading
import time
from queue import Queue
from typing import Callable

import numpy as np
import ome_types.model as om
import tifffile as tf
import zarr
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .camera_calibration import dark_calibration


class AcquisitionJob:
    '''A class holding cross-thread info required for an Acquisition Job.

    Attributes
    ----------
    temp_queue : Queue[float]
        The queue for temporary data.
    display_queue : Queue[np.ndarray]
        The queue for display data.
    save_queue : Queue[tuple[np.ndarray, float]]
        The queue for data to be saved along with associated timestamps.
    path : str
        The base path where files will be saved.
    height : int
        The height of the image frames.
    width : int
        The width of the image frames.
    biggTiff : bool, optional
        Flag indicating whether to use big TIFF format, by default True.
    bytes_per_pixel : int, optional
        The number of bytes per pixel, by default 1.
    exposure : float, optional
        The exposure time in milliseconds, by default 50.0.
    frames : int, optional
        The number of frames, by default 100.
    full_tif_meta : bool, optional
        Flag indicating whether to include full TIFF metadata, by default True.
    is_dark_cal : bool, optional
        Flag indicating whether dark calibration is enabled, by default False.
    meta_file : str, optional
        Additional metadata file path, by default ''.
    meta_func : Callable, optional
        A function to generate metadata, by default a lambda function.
    name : str, optional
        The name of the acquisition job, by default 'Camera'.
    prefix : str, optional
        A prefix for filenames, by default ''.
    save : bool, optional
        Flag indicating whether to save data, by default False.
    Zarr : bool, optional
        Flag indicating whether to use Zarr format, by default False.

    Methods
    -------
    getExt()
        Return the file extension based on the Zarr attribute.
    getFilename()
        Generate the filename based on the current state.
    getTempFilename()
        Generate the temperatures filename based on the current state.
    addDarkFrame(frame)
        Add a dark frame for calibration.
    addFrame(frame)
        Add a frame to the acquisition job.
    addTemp(temp)
        Add temperature data to the acquisition job.
    getMetaFilename()
        Return the metadata file path.
    writeMetaFile()
        Write metadata to a file.
    getTiffWriter()
        Return a TiffWriter instance for saving TIFF files.
    getZarrArray()
        Return the Zarr array for saving Zarr files.
    saveMetadata()
        Save metadata to the current file.
    finalize()
        Finalize and clean up resources.
    setDone(index, value)
        Set the completion status for a specific stage.
    '''

    def __init__(self, temp_queue: Queue[float], display_queue: Queue[np.ndarray],
             save_queue: Queue[tuple[np.ndarray, float]], path: str,
             height: int, width: int, **kwargs) -> None:
        '''Initialize the AcquisitionJob instance.

        Parameters
        ----------
        temp_queue : Queue[float]
            The queue for temporary data.
        display_queue : Queue[np.ndarray]
            The queue for display data.
        save_queue : Queue[tuple[np.ndarray, float]]
            The queue for data to be saved along with associated timestamps.
        path : str
            The base path where files will be saved.
        height : int
            The height of the image frames.
        width : int
            The width of the image frames.
        **kwargs
            Optional keyword arguments for customization.

            Optional Parameters
            -------------------
            biggTiff : bool, optional
                Flag indicating whether to use big TIFF format, by default True.
            bytes_per_pixel : int, optional
                The number of bytes per pixel, by default 1.
            exposure : float, optional
                The exposure time in milliseconds, by default 50.0.
            frames : int, optional
                The number of frames, by default 100.
            full_tif_meta : bool, optional
                Flag indicating whether to include full TIFF metadata, by default True.
            is_dark_cal : bool, optional
                Flag indicating whether dark calibration is enabled, by default False.
            meta_file : str, optional
                Additional metadata file path, by default ''.
            meta_func : Callable, optional
                A function to generate metadata, by default a lambda function.
            name : str, optional
                The name of the acquisition job, by default 'Camera'.
            prefix : str, optional
                A prefix for filenames, by default ''.
            save : bool, optional
                Flag indicating whether to save data, by default False.
            Zarr : bool, optional
                Flag indicating whether to use Zarr format, by default False.
            rois : list[list[int]], optional
                List of regions of interest, by default None.
        '''
        self.display_queue = display_queue
        self.height = height
        self.index = 0
        self.lock = threading.Lock()
        self.major = 0
        self.path = path
        self.save_queue = save_queue
        self.temp_queue = temp_queue
        self.timestamp = time.strftime('_%Y_%m_%d_%H%M%S')
        self.width = width
        self.zarr_array = None

        self.biggTiff: bool = kwargs.get('biggTiff', True)
        self.bytes_per_pixel: int = kwargs.get('bytes_per_pixel', 1)
        self.exposure: float = kwargs.get('exposure', 50.0)
        self.frames: int = kwargs.get('frames', 100)
        self.full_tif_meta: bool = kwargs.get('full_tif_meta', True)
        self.is_dark_cal: bool = kwargs.get('is_dark_cal', False)
        self.meta_file: str = kwargs.get('meta_file', '')
        self.meta_func: Callable = kwargs.get('meta_func', lambda *args: '')
        self.name: str = kwargs.get('name', 'Camera')
        self.prefix: str = kwargs.get('prefix', '')
        self.save: bool = kwargs.get('save', False)
        '''cross-thread attribute (use lock)'''
        self.zarr: bool = kwargs.get('Zarr', False)
        self.rois: list[list[int]] = kwargs.get('rois', None)

        if self.rois is not None:
            same_width = all(
                [self.rois[0][2] == roi[2] for roi in self.rois])
            same_height = all(
                [self.rois[0][3] == roi[3] for roi in self.rois])
            if not same_width or not same_height:
                self.rois = None

        self.capture_done = False
        '''cross-thread attribute (use lock)'''
        self.capture_time = 0.0
        '''cross-thread attribute (use lock)'''
        self.display_done = False
        '''cross-thread attribute (use lock)'''
        self.display_time = 0.0
        '''cross-thread attribute (use lock)'''
        self.frame = None
        '''cross-thread attribute (use lock)'''
        self.frames_captured = 0
        '''cross-thread attribute (use lock)'''
        self.frames_saved = 0
        '''cross-thread attribute (use lock)'''
        self.save_time = 0.0
        '''cross-thread attribute (use lock)'''
        self.stop_threads = False
        '''cross-thread attribute (use lock)'''

        self.tiffWriter = None
        self.tempFile = None
        self.dark_cal = None


        if self.save:
            if not os.path.exists(self.path):
                os.makedirs(self.path)

            while os.path.exists(self.getFilename()):
                self.major += 1

            self.writeMetaFile()

    def getExt(self):
        '''Return the file extension based on the Zarr attribute.

        Returns
        -------
        str
            File extension ('ome.tif' or 'zarr').
        '''
        return 'ome.tif' if not self.zarr else 'zarr'

    def getFilename(self) -> str:
        '''Generate the filename based on the current state.

        Returns
        -------
        str
            Generated filename.
        '''
        return self.path + \
            f'{self.major:02d}_{self.prefix}_image_{self.index:05d}.{self.getExt()}'

    def getTempFilename(self) -> str:
        '''Generate the temperature log filename based on the current state.

        Returns
        -------
        str
            Generated temperature log filename.
        '''
        return self.path + self.prefix + \
            f'{self.major:02d}_{self.prefix}_temp_log.csv'

    def addDarkFrame(self, frame: np.ndarray):
        '''Add a dark frame for calibration.

        Parameters
        ----------
        frame : np.ndarray
            Dark frame data.
        '''
        if self.is_dark_cal:
            if self.dark_cal is None:
                self.dark_cal = dark_calibration(
                    frame.shape, self.exposure)
            self.dark_cal.addFrame(frame)

    def addFrame(self, frame: np.ndarray):
        '''Add a frame to the acquisition job.

        Parameters
        ----------
        frame : np.ndarray
            Image frame data.
        '''
        if self.zarr:
            self.getZarrArray()[self.frames_saved] = frame
        else:
            if self.tiffWriter is None:
                self.tiffWriter = self.getTiffWriter()
            # append frame to tiff
            try:
                self.tiffWriter.write(
                    data=frame[np.newaxis, :],
                    photometric='minisblack')
            except ValueError as ve:
                if str(ve) == \
                        'data too large for standard TIFF file':
                    self.tiffWriter.close()
                    self.saveMetadata()
                    self.frames_saved = 0
                    self.index += 1
                    self.tiffWriter = self.getTiffWriter()
                    self.tiffWriter.write(
                        data=frame[np.newaxis, :],
                        photometric='minisblack')
                else:
                    raise ve

    def addTemp(self, temp: float):
        '''Add temperature data to the log file.

        Parameters
        ----------
        temp : float
            Temperature data.
        '''
        # open csv file and append sensor temp and close
        with open(self.getTempFilename(), 'ab') as f:
            np.savetxt(f, [temp], delimiter=';')

    def getMetaFilename(self) -> str:
        '''Return the metadata file path.

        Returns
        -------
        str
            Metadata file path.
        '''
        return self.path + self.name + self.timestamp + '.txt'

    def writeMetaFile(self):
        '''Write metadata to a file.'''
        with open(self.getMetaFilename(), 'w+') as metaFile:
            json.dump(self.meta_file, metaFile)

    def getTiffWriter(self) -> tf.TiffWriter:
        '''Return a TiffWriter instance for saving TIFF files.

        Returns
        -------
        tf.TiffWriter
            TiffWriter instance.
        '''
        return tf.TiffWriter(
            self.getFilename(), append=False,
            bigtiff=self.biggTiff, ome=False)

    def getZarrArray(self) -> zarr.Array:
        '''Return the Zarr array for saving Zarr files.

        Returns
        -------
        zarr.Array
            Zarr array.
        '''
        if self.zarr_array is None:
            self.zarr_array = zarr.open_array(
                self.getFilename(),
                shape=(self.frames, self.height, self.width),
                compressor=None,
                chunks=(1, self.height, self.width),
                dtype=np.uint16)
        return self.zarr_array

    def saveMetadata(self):
        '''Save metadata to the current file.'''
        ome: om.OME = self.meta_func(
            self.frames_saved, self.width, self.height)

        tf.tiffcomment(self.getFilename(), ome.to_xml())

    def finalize(self):
        '''Finalize and clean up resources.'''
        if self.tempFile is not None:
            self.tempFile.close()
        if self.tiffWriter is not None:
            self.tiffWriter.close()
        if self.dark_cal is not None:
            if self.dark_cal._counter > 1:
                self.dark_cal.saveResults(
                    self.path,
                    f'{self.major:02d}_{self.prefix}')

        if self.save and not self.zarr:
            self.saveMetadata()

    def setDone(self, index: int, value: bool):
        '''Set the completion status for a specific stage.

        Parameters
        ----------
        index : int
            The index representing the stage (0 for capture, 1 for display, etc.).
        value : bool
            The completion status.
        '''
        if index == 0:
            self.capture_done = value
        elif index == 1:
            self.display_done = value
