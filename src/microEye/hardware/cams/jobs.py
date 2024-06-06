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

from microEye.hardware.cams.camera_calibration import dark_calibration


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
    rois : list[list[int]], optional
        List of regions of interest, by default empty list.
    seperate_rois : bool, optional
        Flag indicating whether to save ROIs into seperate files,
        by default False.
        (For Zarr format different ROIs are treated as channels)
    flip_rois : bool, optional
        Flag indicating whether to flip n-th ROIs horizontally for n > 1,
        by default True.


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
                List of regions of interest, by default empty list.
            seperate_rois : bool, optional
                Flag indicating whether to save ROIs into seperate files,
                by default False.
                (For Zarr format different ROIs are treated as channels)
            flip_rois : bool, optional
                Flag indicating whether to flip n-th ROIs horizontally for n > 1,
                by default True.
        '''
        self.display_queue = display_queue
        self.height = height
        self.cam_height = height
        self.index = 0
        self.lock = threading.Lock()
        self.major = 0
        self.path = path
        self.save_queue = save_queue
        self.temp_queue = temp_queue
        self.timestamp = time.strftime('_%Y_%m_%d_%H%M%S')
        self.width = width
        self.cam_width = width
        self.channels = 1
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
        self.rois: list[list[int]] = kwargs.get('rois', [])
        self.seperate_rois: bool = kwargs.get('seperate_rois', False)
        self.flip_rois: bool = kwargs.get('flip_rois', True)

        if self.rois:
            same_width = all(
                [self.rois[0][2] == roi[2] for roi in self.rois])
            same_height = all(
                [self.rois[0][3] == roi[3] for roi in self.rois])
            if not same_width or not same_height:
                self.rois = None
            else:
                self.width = self.rois[0][2]
                self.height = self.rois[0][3]
                self.channels = len(self.rois)

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

        self.c_event = threading.Event()
        self.s_event = kwargs.get('s_event', threading.Event())

        self.tiffWriter: list[tf.TiffWriter] = []
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

    def getFilename(self, roi_index: int=None) -> str:
        '''Generate the filename based on the current state.

        Parameters
        ----------
        roi_index : int, optional
            the roi index for file name suffix, if None then not added. Default is None.

        Returns
        -------
        str
            Generated filename.
        '''
        roi_suffix = '' if roi_index is None else f'_ROI_{roi_index:02d}'
        return self.path + \
            f'{self.major:02d}_{self.prefix}_image_{self.index:05d}{roi_suffix}.{self.getExt()}'

    def getTempFilename(self) -> str:
        '''Generate the temperature log filename based on the current state.

        Returns
        -------
        str
            Generated temperature log filename.
        '''
        return self.path + \
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
            if self.rois:
                for idx, roi in enumerate(self.rois):
                    roi_data = frame[
                            roi[1]: roi[1] + roi[3],
                            roi[0]: roi[0] + roi[2]]
                    if idx > 0 and self.flip_rois:
                        roi_data = np.fliplr(roi_data)

                    self.getZarrArray()[self.frames_saved, idx, 0] = roi_data
            else:
                self.getZarrArray()[self.frames_saved, 0, 0] = frame
        else:
            if not self.tiffWriter:
                if self.rois:
                    if self.seperate_rois:
                        for idx in range(len(self.rois)):
                            self.tiffWriter.append(
                                self.getTiffWriter(idx))
                    else:
                        self.tiffWriter.append(self.getTiffWriter())
                else:
                    self.tiffWriter.append(self.getTiffWriter())
            # append frame to tiff
            if self.rois:
                if self.seperate_rois:
                    for idx, roi in enumerate(self.rois):
                        roi_data = frame[
                                roi[1]: roi[1] + roi[3],
                                roi[0]: roi[0] + roi[2]]
                        if idx > 0 and self.flip_rois:
                            roi_data = np.fliplr(roi_data)

                        self.tiffWriter[idx] = self.writeTiffFrame(
                            self.tiffWriter[idx],
                            roi_data, idx)
                else:
                    data = np.zeros(
                        (len(self.rois),
                         self.rois[0][3], self.rois[0][2]), dtype=frame.dtype)
                    for idx, roi in enumerate(self.rois):
                        data[idx] = frame[
                                roi[1]: roi[1] + roi[3],
                                roi[0]: roi[0] + roi[2]]
                        if idx > 0 and self.flip_rois:
                            data[idx] = np.fliplr(data[idx])

                    self.tiffWriter[0] = self.writeTiffFrame(
                        self.tiffWriter[0], data)
            else:
                self.tiffWriter[0] = self.writeTiffFrame(
                    self.tiffWriter[0], frame)

    def writeTiffFrame(
            self,
            writer: tf.TiffWriter, frame: np.ndarray, roi_index=None):
        '''writes an image into a Tiff file.

        Parameters
        ----------
        writer : tf.TiffWriter
            the TiffWriter class of the Tiff file.
        frame : np.ndarray
            Image frame data.
        roi_index : int, optional
            the roi index for file name suffix, if None then not added. Default is None.
        '''

        try:
            writer.write(
                data=frame[np.newaxis, np.newaxis, np.newaxis, ...
                           ] if frame.ndim == 2 else frame[
                               np.newaxis, :, np.newaxis, ...],
                photometric='minisblack')
        except ValueError as ve:
            if str(ve) == \
                    'data too large for standard TIFF file':
                writer.close()
                self.saveMetadata(roi_index)
                self.frames_saved = 0
                self.index += 1
                writer = self.getTiffWriter(roi_index)
                writer.write(
                    data=frame[np.newaxis, np.newaxis, np.newaxis, ...
                               ] if frame.ndim == 2 else frame[
                                   np.newaxis, :, np.newaxis, ...],
                    photometric='minisblack')
            else:
                raise ve

        return writer

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
        return self.path + \
            f'{self.major:02d}_{self.prefix}' + self.name.replace(
                ' ', '_') + self.timestamp + '.json'

    def writeMetaFile(self):
        '''Write metadata to a file.'''
        with open(self.getMetaFilename(), 'w+') as metaFile:
            json.dump(self.meta_file, metaFile, indent=2)

    def getTiffWriter(self, roi_index: int =None) -> tf.TiffWriter:
        '''Return a TiffWriter instance for saving TIFF files.

        Parameters
        ----------
        roi_index : int, optional
            the roi index for file name suffix, if None then not added. Default is None.

        Returns
        -------
        tf.TiffWriter
            TiffWriter instance.
        '''
        return tf.TiffWriter(
            self.getFilename(roi_index), append=False,
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
                shape=(self.frames, self.channels, 1, self.height, self.width),
                compressor=None,
                chunks=(1, 1, 1, self.height, self.width),
                dtype=np.uint16)
        return self.zarr_array

    def saveMetadata(self, roi_index: int =None):
        '''
        Save metadata to the current file.

        Parameters
        ----------
        roi_index : int, optional
            the roi index for file name suffix, if None then not added. Default is None.
        '''
        ome: om.OME = self.meta_func(
            self.frames_saved, self.width, self.height,
            1 if self.seperate_rois else self.channels)

        tf.tiffcomment(self.getFilename(roi_index), ome.to_xml())

    def finalize(self):
        '''Finalize and clean up resources.'''
        if self.dark_cal is not None:
            if self.dark_cal._counter > 1:
                self.dark_cal.saveResults(
                    self.path,
                    f'{self.major:02d}_{self.prefix}')

        if self.tiffWriter:
            if self.rois and self.seperate_rois:
                for idx, writer in enumerate(self.tiffWriter):
                    writer.close()
                    if self.save and not self.zarr:
                        self.saveMetadata(idx)
            else:
                self.tiffWriter[0].close()
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
