import logging
import sys
from typing import Any, Optional, Union

import numpy as np
import pco
from pco.sdk import Sdk, shared_library_loader

from microEye.hardware.cams.micam import miCamera
from microEye.hardware.cams.pco.enums import *


class pco_cam(miCamera):
    def __init__(self, Cam_ID=0, **kwargs) -> None:
        super().__init__(Cam_ID)

        self.bytes_per_pixel = 2

        self.cam = pco.Camera(
            interface=kwargs.get('interface'), serial=kwargs.get('serial')
        )

        self.initialize()

    def initialize(self):
        '''Initialize the camera'''
        self.cam.default_configuration()
        self.getExposure()
        self.getDelay()

        self.name = self.cam.camera_name.replace(' ', '_').replace('.', '_')
        self.serial = self.cam.camera_serial
        self.interface = self.cam.interface

        self.print_status()

    @property
    def exposure_increment(self) -> float:
        '''Get the exposure increment of the camera

        Returns
        -------
        float
            The exposure increment of the camera
        '''
        return self.getDescByKey(DescriptionKeys.MIN_EXPOSURE_STEP)

    @property
    def exposure_range(self) -> tuple[float, float]:
        '''
        Get the minimum and maximum exposure time of the camera

        Returns
        -------
        tuple[float, float]
            The minimum and maximum exposure time of the camera
        '''
        return (
            self.getDescByKey(DescriptionKeys.MIN_EXPOSURE_TIME),
            self.getDescByKey(DescriptionKeys.MAX_EXPOSURE_TIME),
        )

    @exposure_range.setter
    def exposure_range(self, value: tuple[float, float]):
        '''
        Set the minimum and maximum exposure time of the camera

        Parameters
        ----------
        value : tuple[float, float]
            The minimum and maximum exposure time to set
        '''
        pass

    @property
    def pixel_rates(self) -> list[int]:
        '''Get the pixel rates of the camera

        Returns
        -------
        list[int]
            The pixel rates of the camera
        '''
        return self.getDescByKey(DescriptionKeys.PIXELRATES)

    @property
    def delay_increment(self) -> float:
        '''Get the delay increment of the camera

        Returns
        -------
        float
            The delay increment of the camera
        '''
        return self.getDescByKey(DescriptionKeys.MIN_DELAY_STEP)

    @property
    def delay_range(self) -> tuple[float, float]:
        '''Get the minimum and maximum delay time of the camera

        Returns
        -------
        tuple[float, float]
            The minimum and maximum delay time of the camera
        '''
        return (
            self.getDescByKey(DescriptionKeys.MIN_DELAY_TIME),
            self.getDescByKey(DescriptionKeys.MAX_DELAY_TIME),
        )

    @property
    def min_dim(self) -> tuple[int, int]:
        '''Get the minimum dimensions of the camera

        Returns
        -------
        tuple[int, int]
            The minimum width and height of the camera
        '''
        return (
            self.getDescByKey(DescriptionKeys.MIN_WIDTH),
            self.getDescByKey(DescriptionKeys.MIN_HEIGHT),
        )

    @property
    def max_dim(self) -> tuple[int, int]:
        '''Get the maximum dimensions of the camera

        Returns
        -------
        tuple[int, int]
            The maximum width and height of the camera
        '''
        return (
            self.getDescByKey(DescriptionKeys.MAX_WIDTH),
            self.getDescByKey(DescriptionKeys.MAX_HEIGHT),
        )

    @property
    def height(self):
        '''Get the height of the camera ROI'''
        return self.ROI[3]

    @height.setter
    def height(self, value):
        '''Invalid set height, use `setROI` instead'''
        print('Invalid set height, use setROI instead')
        pass

    @property
    def width(self):
        '''Get the width of the camera ROI'''
        return self.ROI[2]

    @width.setter
    def width(self, value):
        '''Invalid set width, use `setROI` instead'''
        print('Invalid set width, use setROI instead')
        pass

    @property
    def roi_steps(self) -> tuple[int, int]:
        '''Get the steps for the ROI

        Returns
        -------
        tuple[int, int]
            The steps for the ROI in the horizontal and vertical direction
        '''
        return self.getDescByKey(DescriptionKeys.ROI_STEPS)

    @property
    def binning_steps(self) -> tuple[int, int]:
        return (
            self.getDescByKey(DescriptionKeys.BINNING_HORZ_VEC),
            self.getDescByKey(DescriptionKeys.BINNING_VERT_VEC),
        )

    @property
    def bit_res(self):
        '''Get the bit resolution of the camera'''
        return self.getDescByKey(DescriptionKeys.BIT_RESOLUTION)

    @property
    def ROI(self) -> tuple[int, int, int, int]:
        '''
        Get the current region of interest (ROI) of the camera

        Returns
        -------
        tuple[int, int, int, int]
            The current region of interest (x, y, width, height),
            Note that x and y are 1-indexed.
        '''
        return self.getConfigByKey(ConfigKeys.ROI)

    def close(self):
        '''Close the camera'''
        if self.cam is not None:
            self.cam.close()

    def populate_status(self):
        self.status['Camera'] = {
            'Name': self.cam.camera_name,
            'Serial': self.cam.camera_serial,
            'Interface': self.cam.interface,
        }

        self.status['Exposure'] = {
            'Value': self.cam.exposure_time,
            'Delay': self.cam.delay_time,
            'Unit': 's',
        }

        self.status['Temperature'] = self.cam.sdk.get_temperature()

        self.status['Description'] = self.cam.description

        self.status['Configuration'] = self.cam.configuration

    def get_temperature(self) -> float:
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.temperature = -127
        if self.cam is not None:
            self.temperature = self.cam.sdk.get_temperature()['sensor temperature']
        return self.temperature

    def getExposure(self) -> float:
        '''Get the current exposure time of the camera

        Returns
        -------
        float
            The current exposure time of the camera
        '''
        exp = -127
        try:
            if self.cam is not None:
                self.exposure_current = self.cam.exposure_time
                self.exposure_unit = 's'
                return self.exposure_current
        except Exception:
            print('Exposure Get ERROR')
        return exp

    def setExposure(self, exp: float) -> float:
        '''Set the exposure time of the camera

        Parameters
        ----------
        exp : float
            The exposure time to set in seconds

        Returns
        -------
        float
            The current exposure time of the camera
        '''
        try:
            if self.cam is not None:
                self.cam.exposure_time = exp
                self.exposure_current = self.cam.exposure_time
                return self.exposure_current
        except Exception:
            print('Exposure Set ERROR')
        return -127

    def getDelay(self) -> float:
        '''Get the current delay time of the camera

        Returns
        -------
        float
            The current delay time of the camera
        '''
        delay = -127
        try:
            if self.cam is not None:
                self.delay_current = self.cam.delay_time
                return self.delay_current
        except Exception:
            print('Delay Get ERROR')
        return delay

    def setDelay(self, delay: float) -> float:
        '''Set the delay time of the camera

        Parameters
        ----------
        delay : float
            The delay time to set in seconds

        Returns
        -------
        float
            The current delay time of the camera
        '''
        try:
            if self.cam is not None:
                self.cam.delay_time = delay
                self.delay_current = self.cam.delay_time
                return self.delay_current
        except Exception:
            print('Delay Set ERROR')
        return -127

    def isRecording(self):
        '''Check if the camera is recording'''
        return self.cam.is_recording

    def isColor(self):
        '''Check if the camera is color'''
        return self.cam.is_color

    @property
    def recorded_image_count(self):
        '''Get the number of recorded images'''
        return self.cam.recorded_image_count

    def getDescByKey(self, key: DescriptionKeys):
        '''
        Get the description of the camera by key

        Parameters
        ----------
        key : DescriptionKeys
            The key of the description to get

        Returns
        -------
        Any
            The description value of the camera
        '''
        return self.cam.description[key.value[0]]

    def getConfigByKey(self, key: ConfigKeys):
        '''
        Get the configuration of the camera by key

        Parameters
        ----------
        key : ConfigKeys
            The key of the configuration to get

        Returns
        -------
        Any
            The configuration value of the camera
        '''
        return self.cam.configuration[key.value[0]]

    def setConfigByKey(self, key: ConfigKeys, value):
        if not isinstance(value, key.value[1]):
            raise ValueError(
                f'Value {value} is not of type {key.value[1]} for key {key.name}'
            )
        self.cam.configuration = {key.value[0]: value}

    def getPixelRate(self) -> int:
        '''Get the pixel rate of the camera'''
        return self.getConfigByKey(ConfigKeys.PIXEL_RATE)

    def setPixelRate(self, rate: int) -> None:
        '''Set the pixel rate of the camera

        Parameters
        ----------
        rate : int
            The pixel rate to set

        Raises
        ------
        ValueError
            If the pixel rate is not supported by the camera
        '''
        if rate not in self.getDescByKey(DescriptionKeys.PIXELRATES):
            raise ValueError(f'Pixelrate {rate} is not supported by the camera')
        self.setConfigByKey(ConfigKeys.PIXEL_RATE, rate)

    def getTrigger(self) -> str:
        '''
        Get the trigger mode of the camera

        Returns
        -------
        str
            The trigger mode of the camera
        '''
        return self.getConfigByKey(ConfigKeys.TRIGGER)

    def setTrigger(self, trigger: str) -> None:
        '''
        Set the trigger mode of the camera
        '''
        self.setConfigByKey(ConfigKeys.TRIGGER, trigger)

    def getAcquire(self) -> str:
        '''
        Get the acquire mode of the camera'''
        return self.getConfigByKey(ConfigKeys.ACQUIRE)

    def setAcquire(self, acquire: str) -> None:
        '''
        Set the acquire mode of the camera
        '''
        self.setConfigByKey(ConfigKeys.ACQUIRE, acquire)

    def getMetadata(self) -> str:
        '''
        Get the metadata of the camera
        '''
        return self.getConfigByKey(ConfigKeys.METADATA)

    def setMetadata(self, metadata: str) -> None:
        '''
        Set the metadata of the camera
        '''
        self.setConfigByKey(ConfigKeys.METADATA, metadata)

    def getNoiseFilter(self) -> str:
        '''
        Get the noise filter of the camera
        '''
        return self.getConfigByKey(ConfigKeys.NOISE_FILTER)

    def setNoiseFilter(self, noise_filter: str) -> None:
        '''
        Set the noise filter of the camera
        '''
        self.setConfigByKey(ConfigKeys.NOISE_FILTER, noise_filter)

    def getBinning(self) -> tuple[int, int, str]:
        '''
        Get the binning of the camera

        Returns
        -------
        tuple[int, int, str]
            The binning of the camera, horizontal, vertical and mode
        '''
        return self.getConfigByKey(ConfigKeys.BINNING)

    def setBinning(self, binning: tuple[int, int, str]) -> None:
        '''
        Set the binning of the camera

        Parameters
        ----------
        binning : tuple[int, int, str]
            The binning to set, horizontal, vertical and mode
        '''
        self.setConfigByKey(ConfigKeys.BINNING, binning)

    def getAutoExposure(self) -> tuple[str, int, int]:
        '''
        Get the auto exposure of the camera

        Returns
        -------
        tuple[str, int, int]
            The auto exposure of the camera, mode, min and max exposure
        '''
        return self.getConfigByKey(ConfigKeys.AUTO_EXPOSURE)

    def setAutoExposure(self, auto_exposure: tuple[str, int, int]) -> None:
        '''
        Set the auto exposure of the camera

        Parameters
        ----------
        auto_exposure : tuple[str, int, int]
            The auto exposure to set, mode, min and max exposure
        '''
        self.setConfigByKey(ConfigKeys.AUTO_EXPOSURE, auto_exposure)

    def getTimestamp(self) -> str:
        '''
        Get the timestamp of the camera
        '''
        return self.getConfigByKey(ConfigKeys.TIMESTAMP)

    def setTimestamp(self, timestamp: str) -> None:
        '''
        Set the timestamp of the camera

        Parameters
        ----------
        timestamp : str
            The timestamp to set
        '''
        self.setConfigByKey(ConfigKeys.TIMESTAMP, timestamp)

    def setROI(self, roi: tuple[int, int, int, int]) -> None:
        '''
        Set the region of interest (ROI) of the camera

        Parameters
        ----------
        roi : tuple[int, int, int, int]
            The region of interest (x, y, width, height),
            Note that x and y are 1-indexed.
        '''
        self.setConfigByKey(ConfigKeys.ROI, roi)

    def resetROI(self) -> None:
        '''Reset the region of interest (ROI) of the camera'''
        self.setConfigByKey(ConfigKeys.ROI, (1, 1, self.max_dim[0], self.max_dim[1]))

    def configureExposureTrigger(self, status: bool, rising_or_falling: bool):
        '''Configure the exposure trigger of the camera

        Parameters
        ----------
        status : bool
            The status of the exposure trigger
        rising_or_falling : bool
            The rising or falling edge of the exposure trigger
        '''
        self.cam.configureHWIO_1_exposureTrigger(
            status, 'rising edge' if rising_or_falling else 'falling edge'
        )

    def configureAcquireTrigger(self, status: bool, high_or_low: bool):
        '''Configure the acquire trigger of the camera

        Parameters
        ----------
        status : bool
            The status of the acquire trigger
        high_or_low : bool
            The high or low level of the acquire trigger
        '''
        self.cam.configureHWIO_2_acquireEnable(
            status, 'high level' if high_or_low else 'low level'
        )

    def configureStatusBusy(self, status: bool, high_or_low: bool, signal_type: str):
        '''Configure the status busy of the camera

        Parameters
        ----------
        status : bool
            The status of the status busy
        high_or_low : bool
            The high or low level of the status busy
        signal_type : str
            The signal type of the status busy
        '''
        if signal_type not in ['status busy', 'status line', 'status armed']:
            raise ValueError(f'Invalid signal type {signal_type}')
        return self.cam.configureHWIO_3_statusBusy(status, high_or_low, signal_type)

    def configureStatusExpos(
        self,
        status: bool,
        high_or_low: bool,
        signal_type: str,
        signal_timing: Optional[str] = None,
    ):
        '''Configure the status expos of the camera

        Parameters
        ----------
        status : bool
            The status of the status expos
        high_or_low : bool
            The high or low level of the status expos
        signal_type : str
            The signal type of the status expos
        signal_timing : str, optional
            The signal timing of the status expos
        '''
        if signal_type not in ['status expos', 'status line', 'status armed']:
            raise ValueError(f'Invalid signal type {signal_type}')
        if signal_timing is not None and signal_timing not in [
            'first line',
            'global',
            'last line',
            'all lines',
        ]:
            raise ValueError(f'Invalid signal timing {signal_timing}')
        return self.cam.configureHWIO_4_statusExpos(
            status, high_or_low, signal_type, signal_timing
        )

    def configureAutoExposure(
        self, region_type: ExposureRegionType, min_exp: int, max_exp: int
    ):
        '''Configure the auto exposure of the camera

        Parameters
        ----------
        region_type : ExposureRegionType
            The region type of the auto exposure
        min_exp : int
            The minimum exposure time of the auto exposure
        max_exp : int
            The maximum exposure time of the auto exposure
        '''
        return self.cam.configure_auto_exposure(region_type.value, min_exp, max_exp)

    def setAutoExposure(self, on: bool):
        '''Set the auto exposure of the camera

        Parameters
        ----------
        on : bool
            The status of the auto exposure
        '''
        if on:
            self.cam.auto_exposure_on()
        else:
            self.cam.auto_exposure_off()

    def stop(self):
        '''Stop the camera'''
        self.cam.stop()

    def record(
        self,
        n_images: int = 1,
        mode: RecorderModes = RecorderModes.SEQUENCE,
        filename: Optional[str] = None,
    ):
        '''Record images with the camera

        Parameters
        ----------
        n_images : int, optional
            The number of images to record, by default 1
        mode : RecorderModes, optional
            The mode of the recorder, by default RecorderModes.SEQUENCE
        filename : str, optional
            The filename to save the images, by default None
        '''
        self.cam.record(n_images, mode.value[0], filename)

    def wait_for_first_image(self, delay: bool = True, timeout: int = 60):
        '''Wait for the first image from the camera

        Parameters
        ----------
        delay : bool, optional
            The delay for the first image, by default True
        timeout : int, optional
            The timeout for the first image, by default 60
        '''
        return self.cam.wait_for_first_image(delay, timeout)

    def wait_for_new_image(self, delay: bool = True, timeout: int = 60):
        '''Wait for a new image from the camera

        Parameters
        ----------
        delay : bool, optional
            The delay for the new image, by default True
        timeout : int, optional
            The timeout for the new image, by default 60
        '''
        return self.cam.wait_for_new_image(delay, timeout)

    def get_convert_control(self, data_format: ImageFormats):
        '''Get the convert control of the camera

        Parameters
        ----------
        data_format : ImageFormats
            The image format to get the convert control

        Returns
        -------
        dict[str, Any]
            The convert control of the camera
        '''
        return self.cam.get_convert_control(data_format.value)

    def set_convert_control(
        self, data_format: ImageFormats, convert_control: dict[str, Any]
    ):
        '''Set the convert control of the camera

        Parameters
        ----------
        data_format : ImageFormats
            The image format to set the convert control
        convert_control : dict[str, Any]
            The convert control to set
        '''
        return self.cam.set_convert_control(data_format.value, convert_control)

    def load_lut(self, data_format: ImageFormats, filename: str):
        '''Load the look-up table (LUT) of the camera

        Parameters
        ----------
        data_format : ImageFormats
            The image format to load the LUT
        filename : str
            The filename of the LUT
        '''
        return self.cam.load_lut(data_format.value, filename)

    def get_image(
        self,
        idx: int,
        roi: Optional[tuple[int, int, int, int]] = None,
        data_format: ImageFormats = ImageFormats.MONO16,
        Not_implemented_yet=None,
    ) -> Union[np.ndarray, dict[str, Any]]:
        '''Get the image from the camera

        Parameters
        ----------
        idx : int
            The index of the image
        roi : tuple[int, int, int, int], optional
            The region of interest (ROI) of the image, by default None
        data_format : ImageFormats, optional
            The image format to get the image, by default ImageFormats.MONO16
        Not_implemented_yet : Any, optional
            Not implemented yet, by default None

        Returns
        -------
        Union[np.ndarray, dict[str, Any]]
            The image from the camera, as a numpy array and metadata as a dictionary
        '''
        return self.cam.image(idx, roi, data_format.value, Not_implemented_yet)

    def image_average(
        self,
        roi: Optional[tuple[int, int, int, int]] = None,
        data_format: ImageFormats = ImageFormats.MONO16,
    ):
        '''Get the average image from the camera

        Parameters
        ----------
        roi : tuple[int, int, int, int], optional
            The region of interest (ROI) of the image, by default None
        data_format : ImageFormats, optional
            The image format to get the image, by default ImageFormats.MONO16

        Returns
        -------
        np.ndarray
            The average image from the camera
        '''
        return self.cam.image_average(roi, data_format.value)


if __name__ == '__main__':
    cam = pco_cam()

    print(cam.get_temperature())
