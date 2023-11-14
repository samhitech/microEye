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
    '''
    def __init__(
            self,
            temp_queue: Queue[float],
            display_queue: Queue[np.ndarray],
            save_queue: Queue[tuple[np.ndarray, float]], path: str,
            height: int, width: int, biggTiff: bool = True,
            bytes_per_pixel: int = 1, exposure: float = 50.0,
            frames: int = 100, full_tif_meta: bool = True,
            is_dark_cal: bool = False, meta_file: str = '',
            meta_func: Callable = lambda *args: '',
            name: str = 'Camera', prefix: str = '',
            save: bool = False, Zarr: bool = False) -> None:

        self.biggTiff = biggTiff
        self.bytes_per_pixel = bytes_per_pixel
        self.display_queue = display_queue
        self.exposure = exposure
        self.frames = frames
        self.full_tif_meta = full_tif_meta
        self.height = height
        self.index = 0
        self.is_dark_cal = is_dark_cal
        self.lock = threading.Lock()
        self.major = 0
        self.meta_file = meta_file
        self.meta_func = meta_func
        self.name = name
        self.path = path
        self.prefix = prefix
        self.save_queue = save_queue
        self.temp_queue = temp_queue
        self.timestamp = time.strftime('_%Y_%m_%d_%H%M%S')
        self.width = width
        self.zarr = Zarr
        self.zarr_array = None

        self.capture_done = False  # cross-thread attribute (use lock)
        self.capture_time = 0.0  # cross-thread attribute (use lock)
        self.display_done = False  # cross-thread attribute (use lock)
        self.display_time = 0.0  # cross-thread attribute (use lock)
        self.frame = None  # cross-thread attribute (use lock)
        self.frames_captured = 0  # cross-thread attribute (use lock)
        self.frames_saved = 0  # cross-thread attribute (use lock)
        self.save = save  # cross-thread attribute (use lock)
        # self.save_done = False  # cross-thread attribute (use lock)
        self.save_time = 0.0  # cross-thread attribute (use lock)
        self.stop_threads = False  # cross-thread attribute (use lock)

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
        return 'ome.tif' if not self.zarr else 'zarr'

    def getFilename(self) -> str:
        return self.path + \
            f'{self.major:02d}_{self.prefix}_image_{self.index:05d}.{self.getExt()}'

    def getTempFilename(self) -> str:
        return self.path + self.prefix + \
            f'{self.major:02d}_{self.prefix}_temp_log.csv'

    def addDarkFrame(self, frame: np.ndarray):
        if self.is_dark_cal:
            if self.dark_cal is None:
                self.dark_cal = dark_calibration(
                    frame.shape, self.exposure)
            self.dark_cal.addFrame(frame)

    def addFrame(self, frame: np.ndarray):
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
        # open csv file and append sensor temp and close
        if self.tempFile is None:
            with open(self.getTempFilename(), 'ab') as f:
                self.tempFile = f

        np.savetxt(self.tempFile, [temp], delimiter=';')

    def getMetaFilename(self) -> str:
        return self.path + self.name + self.timestamp + '.txt'

    def writeMetaFile(self):
        with open(self.getMetaFilename(), 'w+') as metaFile:
            json.dump(self.meta_file, metaFile)

    def getTiffWriter(self) -> tf.TiffWriter:
        return tf.TiffWriter(
            self.getFilename(), append=False,
            bigtiff=self.biggTiff, ome=False)

    def getZarrArray(self) -> zarr.Array:
        if self.zarr_array is None:
            self.zarr_array = zarr.open_array(
                self.getFilename(),
                shape=(self.frames, self.height, self.width),
                compressor=None,
                chunks=(1, self.height, self.width),
                dtype=np.uint16)
        return self.zarr_array

    def saveMetadata(self):
        ome: om.OME = self.meta_func(
            self.frames_saved, self.width, self.height)

        tf.tiffcomment(self.getFilename(), ome.to_xml())

    def finalize(self):
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
        if index == 0:
            self.capture_done = value
        elif index == 1:
            self.display_done = value
