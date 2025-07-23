from typing import Any, Optional, Union

import numpy as np
from pypylon import pylon

from microEye.hardware.cams.micam import miCamera


class basler_cam(miCamera):
    def __init__(self, Cam_ID=0, **kwargs) -> None:
        super().__init__(Cam_ID)

        self.bytes_per_pixel = 2

        self.cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        self.cam.Open()

        self.initialize()

    def initialize(self):
        '''Initialize the camera'''
        self.cam.default_configuration()
        self.getExposure()
        self.getDelay()

        self.name = self.cam
        self.serial = self.cam.camera_serial
        self.interface = self.cam.interface

        self.print_status()

    @staticmethod
    def get_camera_list():
        '''
        Gets the list of available Basler cameras

        Returns
        -------
        dict[str, object]
            dictionary containing
            {Camera ID, Device ID, Sensor ID, Status, InUse, Model, Serial}
        '''
        cam_list = []
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        if devices:
            for index, device in enumerate(devices):
                cam_list.append(
                    {
                        'Camera ID': device.GetSerialNumber(),
                        'Device ID': index,
                        'Sensor ID': 'NA',
                        'Status': 'NA',
                        'InUse': 0,
                        'Model': device.GetModelName(),
                        'Serial': device.GetSerialNumber(),
                        'Driver': device.GetVendorName(),
                    }
                )
        return cam_list

    @property
    def exposure_increment(self) -> float:
        '''Get the exposure increment of the camera

        Returns
        -------
        float
            The exposure increment of the camera
        '''
        return self.cam.ExposureTime.Inc

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
            self.cam.ExposureTime.Min,
            self.cam.ExposureTime.Max,
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
            self.cam.Width.Min,
            self.cam.Height.Min,
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
            self.cam.Width.Max,
            self.cam.Height.Max,
        )

    @property
    def height(self):
        '''Get the height of the camera'''
        return self.cam.Height.Value

    @height.setter
    def height(self, value):
        '''Set the height of the camera'''
        self.cam.Height.Value = value

    @property
    def width(self):
        '''Get the width of the camera ROI'''
        return self.cam.Width.Value

    @width.setter
    def width(self, value):
        '''Set the width of the camera'''
        self.cam.Width.Value = value

    @property
    def roi_steps(self) -> tuple[int, int]:
        '''Get the steps for the ROI

        Returns
        -------
        tuple[int, int]
            The steps for the ROI in the horizontal and vertical direction
        '''
        return (
            self.cam.Width.Inc,
            self.cam.Height.Inc,
        )

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
        return (
            self.cam.OffsetX.Value,
            self.cam.OffsetY.Value,
            self.cam.Width.Value,
            self.cam.Height.Value,
        )

    def close(self):
        '''Close the camera'''
        if self.cam is not None:
            self.cam.Close()

    def populate_status(self):
        self.status['Camera'] = {
            'Name': self.cam.camera_name,
            'Serial': self.cam.camera_serial,
            'Interface': self.cam.interface,
        }

        self.status['Exposure'] = {
            'Value': self.cam.ExposureTime.Value / 1000,
            'Unit': 'ms',
        }

        self.status['Temperature'] = self.get_temperature()

    def get_temperature(self) -> float:
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.temperature = -127
        if self.cam is not None:
            self.temperature = self.cam.BslTemperatureStatus.Value
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
                self.exposure_current = self.cam.ExposureTime.Value
                self.exposure_unit = 'us'
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
                self.cam.ExposureTime.Value = exp
                self.exposure_current = self.cam.ExposureTime.Value
                return self.exposure_current
        except Exception:
            print('Exposure Set ERROR')
        return -127

    def AcquisitionStatus(self):
        '''Get the acquisition status of the camera

        Returns
        -------
        bool
            The acquisition status of the camera
        '''
        return self.cam.AcquisitionStatus.Value

    @property
    def recorded_image_count(self):
        '''Get the number of recorded images'''
        return self.cam.recorded_image_count


    def setROI(self, roi: tuple[int, int, int, int]) -> None:
        '''
        Set the region of interest (ROI) of the camera

        Parameters
        ----------
        roi : tuple[int, int, int, int]
            The region of interest (x, y, width, height),
            Note that x and y are 0-indexed.
        '''
        self.cam.OffsetX.Value = roi[0]
        self.cam.OffsetY.Value = roi[1]
        self.cam.Width.Value = roi[2]
        self.cam.Height.Value = roi[3]

    def resetROI(self) -> None:
        '''Reset the region of interest (ROI) of the camera'''
        self.cam.OffsetX.Value = 0
        self.cam.OffsetY.Value = 0
        self.cam.Width.Value = self.cam.Width.Max
        self.cam.Height.Value = self.cam.Height.Max

    def stop(self):
        '''Stop the camera'''
        self.cam.StopGrabbing()

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
