import logging
import os
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.micam import miCamera

try:
    from ids_peak import ids_peak, ids_peak_ipl_extension
    from ids_peak_ipl import ids_peak_ipl

    ids_peak.Library.Initialize()
except Exception:
    ids_peak = None
    ids_peak_ipl = None
    ids_peak_ipl_extension = None
    logging.getLogger(__name__).warning('IDS peak library could not be loaded.')

logger = logging.getLogger(__name__)


TARGET_PIXEL_FORMAT = ids_peak_ipl.PixelFormatName_Mono16


class peakParams(Enum):
    FREERUN = 'Acquisition.Start (Freerun)'
    SOFTWARE_TRIGGER = 'Acquisition.Software Trigger'
    STOP = 'Acquisition.Stop'
    CLOCK_FREQ = 'Acquisition Settings.Clock Frequency'
    CLOCK_FREQ_SELECTOR = 'Acquisition Settings.Clock Frequency Selector'
    FRAMERATE = 'Acquisition Settings.Framerate'
    FRAME_AVERAGING = 'Acquisition Settings.Frame Averaging'
    TRIGGER_MODE = 'Acquisition Settings.Trigger Mode'
    TRIGGER_SOURCE = 'Acquisition Settings.Trigger Source'
    PIXEL_FORMAT = 'Acquisition Settings.Pixel Format'
    GAIN = 'Acquisition Settings.Gain'
    HOT_PIXEL_CORRECTION = 'Acquisition Settings.Hot Pixel Correction'
    FLASH_REF = 'Acquisition Settings.Flash Reference'
    FLASH_DURATION = 'Acquisition Settings.Flash Duration'
    FLASH_START_DELAY = 'Acquisition Settings.Flash Start Delay'
    BINNING_SELECTOR = 'Acquisition Settings.Binning Selector'
    BINNING_HORIZONTAL = 'Acquisition Settings.Binning Horizontal'
    BINNING_H_MODE = 'Acquisition Settings.Binning Horizontal Mode'
    BINNING_VERTICAL = 'Acquisition Settings.Binning Vertical'
    BINNING_V_MODE = 'Acquisition Settings.Binning Vertical Mode'
    LOAD = 'Acquisition Settings.Load'
    SAVE = 'Acquisition Settings.Save'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')


def available_entries(node) -> list[str]:
    '''Get available entries for a given node.'''
    allEntries = node.Entries()
    availableEntries = []
    for entry in allEntries:
        if (
            entry.AccessStatus() != ids_peak.NodeAccessStatus_NotAvailable
            and entry.AccessStatus() != ids_peak.NodeAccessStatus_NotImplemented
        ):
            availableEntries.append(entry.SymbolicValue())

    return availableEntries


class PeakCamera(miCamera):
    '''A class to handle an IDS Peak / uEye+ camera.'''

    def __init__(self, serial: str = None):
        '''Initializes a new PeakCamera with the specified Cam_ID.


        Parameters
        ----------
        Cam_ID : int, optional
            camera ID, by default 0

            0: first available camera

            1-254: The camera with the specified camera ID
        '''
        if serial is None:
            raise ValueError('Camera serial number must be provided.')

        # Variables
        super().__init__(0)

        self._device_manager = ids_peak.DeviceManager.Instance()
        self._device_manager.Update()
        self._device = None
        self._datastream = None
        # self._buffer_list = []
        self._killed = False

        # Find the device with the specified serial number
        for device in self._device_manager.Devices():
            if device.SerialNumber() == serial:
                self._device = device.OpenDevice(ids_peak.DeviceAccessType_Control)
                break

        if self._device is None:
            raise ValueError(f'Camera with serial number {serial} not found.')

        self.acquisition = False
        self.trigger_mode = None
        self.temperature = -127

        self.initialize()

    def initialize(self):
        '''Starts the driver and establishes the connection to the camera'''
        try:
            self.node_map.FindNode('UserSetSelector').SetCurrentEntry('Default')
            self.node_map.FindNode('UserSetLoad').Execute()
            self.node_map.FindNode('UserSetLoad').WaitUntilDone()
        except ids_peak.Exception as e:
            logger.warning(f'Could not load default userset: {e}')

        self.set_pixel_format('Mono12')
        self.get_roi()

        self.name = (
            f'{self._device.VendorName()}_{self._device.ModelName()}'
            f'_{self._device.SerialNumber()}'
        ).replace('-', '_')

        self.refresh_info(False)

        self.print_status()

    def __del__(self):
        '''Destructor to properly close the camera connection'''
        if self._device is not None:
            self._device = None
        if self._device_manager is not None:
            self._device_manager = None

    @property
    def node_map(self):
        """Returns the camera's node map"""
        return self._device.RemoteDevice().NodeMaps()[0]

    # Prints out some information about the camera and the sensor
    def populate_status(self):
        self.status['Camera'] = {
            'Model': self._device.ModelName(),
            'Serial no.:\t': self._device.SerialNumber(),
            'Pixel Clock [MHz]': self.get_device_clock_frequency(False),
        }

        self.status['Temperature'] = {
            'Value': self.get_temperature(),
            'Unit': 'Celsius',
        }

        exposure_range = self.get_exposure_range(False)
        self.status['Exposure'] = {
            'Value': self.get_exposure(False),
            'Unit': self.get_exposure_unit(),
            'Range': exposure_range[:2],
            'Increment': exposure_range[2],
        }

        framerate_range = self.get_framerate_range(False)
        self.status['Framerate'] = {
            'Value': self.get_framerate(False),
            'Unit': 'Hz',
            'Range': framerate_range[:2],
        }

        self.status['Image Format'] = {
            'Pixel Format': self.get_pixel_format(),
            'Bit Depth': self.get_pixel_format(),
        }

        self.status['Image Size'] = {
            'Width': self.width,
            'Height': self.height,
        }

    def refresh_info(self, output=False):
        """Refreshes the camera's pixel clock,
        framerate, exposure and flash ranges.

        Parameters
        ----------
        output : bool, optional
            True to printout errors, by default False
        """
        if output:
            logger.info(self.name)
        self.get_device_clock_frequency(output)
        self.get_framerate_range(output)
        self.get_exposure_range(output)
        self.get_exposure(output)

    def get_metadata(self):
        '''
        Returns the metadata for the camera.

        Returns
        -------
        dict
            A dictionary containing the metadata for the camera.
        '''
        return {
            'CHANNEL_NAME': self.name,
            'DET_MANUFACTURER': self._device.VendorName(),
            'DET_MODEL': self._device.ModelName(),
            'DET_SERIAL': self._device.SerialNumber(),
            'DET_TYPE': 'CMOS',
            'PX_SIZE': self.pixel_width,
            'PY_SIZE': self.pixel_height,
        }

    @classmethod
    def get_camera_list(cls, output=False):
        '''Gets the list of available IDS cameras

        Parameters
        ----------
        output : bool, optional
            print out camera list, by default False

        Returns
        -------
        dict[str, object]
            dictionary containing
            {Camera ID, Device ID, Sensor ID, Status, InUse, Model, Serial}
        '''
        cam_list: list[dict[str, Any]] = []
        if ids_peak is None or ids_peak_ipl is None or ids_peak_ipl_extension is None:
            return cam_list

        try:
            ids_peak.Library.Initialize()

            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            devices = device_manager.Devices()

            if hasattr(devices, 'empty') and devices.empty():
                raise RuntimeError('No IDS Peak devices found.')

            for device in devices:
                cam_list.append(
                    {
                        'Camera ID': device.SerialNumber(),
                        'Device ID': device.SerialNumber(),
                        'Sensor ID': '(unknown)',
                        'Status': device.ParentInterface().DisplayName(),
                        'InUse': not device.IsOpenable(),
                        'Model': device.ModelName(),
                        'Serial': device.SerialNumber(),
                        'Driver': 'IDS Peak',
                    }
                )
        except Exception:
            logger.error('Error enumerating IDS Peak devices.', exc_info=True)
        finally:
            ids_peak.Library.Close()

        return cam_list

    @property
    def pixel_width(self):
        '''Gets the pixel width in micrometers

        Returns
        -------
        float
            pixel width in um.
        '''
        return self.node_map.FindNode('SensorPixelWidth').Value()

    @property
    def pixel_height(self):
        '''Gets the pixel height in micrometers

        Returns
        -------
        float
            pixel height in um.
        '''
        return self.node_map.FindNode('SensorPixelHeight').Value()

    def get_pixel_formats(self):
        '''Gets available pixel formats.

        Returns
        -------
        list of str
            List of available pixel formats.
        '''
        return available_entries(self.node_map.FindNode('PixelFormat'))

    def get_pixel_format(self):
        '''Gets current pixel format.

        Returns
        -------
        str
            current pixel format.
        '''
        format = self.node_map.FindNode('PixelFormat').CurrentEntry().SymbolicValue()
        return format

    def set_pixel_format(self, format: str):
        '''Sets pixel format.

        Parameters
        ----------
        format : str
            The pixel format to set. Must be the symbolic name.

        Returns
        -------
        int
            is_PixelFormat return code.
        '''
        try:
            if isinstance(format, str) and format in self.get_pixel_formats():
                self.node_map.FindNode('PixelFormat').SetCurrentEntry(format)
                if 'Mono8' in format:
                    self.bytes_per_pixel = 1
                else:
                    self.bytes_per_pixel = 2
            else:
                raise ValueError('Format must be a string and valid.')
            return True
        except Exception as e:
            logger.error(f'Error setting pixel format: {e}')
            return False

    def get_gain(self):
        '''Gets current gain value.

        Returns
        -------
        float
            current gain value.
        '''
        gain = self.node_map.FindNode('Gain').Value()
        return gain

    def set_gain(self, value: float):
        '''Sets gain value.

        Parameters
        ----------
        value : float
            The gain value to set.

        Returns
        -------
        int
            is_Gain return code.
        '''
        gain = self.node_map.FindNode('Gain')

        value = max(min(value, gain.Maximum()), gain.Minimum())

        gain.SetValue(value)

        return gain.Value()

    def get_hot_pixel_correction_modes(self):
        '''Gets available hot pixel correction modes.'''
        return available_entries(self.node_map.FindNode('HotpixelCorrectionMode'))

    def get_hot_pixel_correction_mode(self):
        '''Gets current hot pixel correction mode.'''
        mode = (
            self.node_map.FindNode('HotpixelCorrectionMode')
            .CurrentEntry()
            .SymbolicValue()
        )
        return mode

    def set_hot_pixel_correction_mode(self, mode: str):
        '''Sets hot pixel correction mode.'''
        try:
            if isinstance(mode, str) and mode in self.get_hot_pixel_correction_modes():
                self.node_map.FindNode('HotpixelCorrectionMode').SetCurrentEntry(mode)
            else:
                raise ValueError('Mode must be a string and valid entry.')
            return True
        except Exception as e:
            logger.error(f'Error setting hot pixel correction mode: {e}')
            return False

    def get_binning_selectors(self):
        '''Gets available binning selectors.

        Returns
        -------
        list of str
            List of available binning selectors.
        '''
        return available_entries(self.node_map.FindNode('BinningSelector'))

    def get_binning_selector(self):
        '''Gets current binning selector.

        Returns
        -------
        str
            current binning selector.
        '''
        selector = (
            self.node_map.FindNode('BinningSelector').CurrentEntry().SymbolicValue()
        )
        return selector

    def set_binning_selector(self, selector: str):
        '''Sets binning selector.

        Parameters
        ----------
        selector : str
            The selector to set. Must be the symbolic name.

        Returns
        -------
        int
            is_BinningSelector return code.
        '''
        try:
            if isinstance(selector, str) and selector in self.get_binning_selectors():
                self.node_map.FindNode('BinningSelector').SetCurrentEntry(selector)
            else:
                raise ValueError('Selector must be a string and valid.')
            return True
        except Exception as e:
            logger.error(f'Error setting binning selector: {e}')
            return False

    def get_binning_horizontal_modes(self):
        '''Gets available binning horizontal modes.'''
        return available_entries(self.node_map.FindNode('BinningHorizontalMode'))

    def get_binning_horizontal_mode(self):
        '''Gets current binning horizontal mode.'''
        mode = (
            self.node_map.FindNode('BinningHorizontalMode')
            .CurrentEntry()
            .SymbolicValue()
        )
        return mode

    def set_binning_horizontal_mode(self, mode: str):
        '''Sets binning horizontal mode.'''
        try:
            if isinstance(mode, str) and mode in self.get_binning_horizontal_modes():
                self.node_map.FindNode('BinningHorizontalMode').SetCurrentEntry(mode)
            else:
                raise ValueError('Mode must be a string and valid entry.')
            return True
        except Exception as e:
            logger.error(f'Error setting binning horizontal mode: {e}')
            return False

    def get_binning_vertical_modes(self):
        '''Gets available binning vertical modes.'''
        return available_entries(self.node_map.FindNode('BinningVerticalMode'))

    def get_binning_vertical_mode(self):
        '''Gets current binning vertical mode.'''
        mode = (
            self.node_map.FindNode('BinningVerticalMode').CurrentEntry().SymbolicValue()
        )
        return mode

    def set_binning_vertical_mode(self, mode: str):
        '''Sets binning vertical mode.'''
        try:
            if isinstance(mode, str) and mode in self.get_binning_vertical_modes():
                self.node_map.FindNode('BinningVerticalMode').SetCurrentEntry(mode)
            else:
                raise ValueError('Mode must be a string and valid entry.')
            return True
        except Exception as e:
            logger.error(f'Error setting binning vertical mode: {e}')
            return False

    def get_binning_horizontal(self):
        '''Gets current binning horizontal value.

        Returns
        -------
        int
            current binning horizontal value.
        '''
        return self.node_map.FindNode('BinningHorizontal').Value()

    def get_binning_vertical(self):
        '''Gets current binning vertical value.

        Returns
        -------
        int
            current binning vertical value.
        '''
        return self.node_map.FindNode('BinningVertical').Value()

    def set_binning_horizontal(self, value: int):
        '''Sets binning horizontal value.

        Parameters
        ----------
        value : int
            The binning horizontal value to set.

        Returns
        -------
        int
            is_BinningHorizontal return code.
        '''
        bin_h = self.node_map.FindNode('BinningHorizontal')

        value = max(min(value, bin_h.Maximum()), bin_h.Minimum())

        bin_h.SetValue(value)

        return bin_h.Value()

    def set_binning_vertical(self, value: int):
        '''Sets binning vertical value.

        Parameters
        ----------
        value : int
            The binning vertical value to set.

        Returns
        -------
        int
            is_BinningVertical return code.
        '''
        bin_v = self.node_map.FindNode('BinningVertical')

        value = max(min(value, bin_v.Maximum()), bin_v.Minimum())

        bin_v.SetValue(value)

        return bin_v.Value()

    def get_temperature(self):
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.temperature = self.node_map.FindNode('DeviceTemperature').Value()
        return self.temperature

    def get_roi(self):
        '''Can be used to get the size and position
        of an "area of interest" (ROI) within an image.

        For ROI rectangle check PeakCamera.rectROI

        Returns
        -------
        int
            is_ROI return code.
        '''
        self.width = self.node_map.FindNode('Width').Value()
        self.height = self.node_map.FindNode('Height').Value()
        return (
            self.node_map.FindNode('OffsetX').Value(),
            self.node_map.FindNode('OffsetY').Value(),
            self.width,
            self.height,
        )

    def set_roi(self, x, y, width, height):
        '''Sets the size and position of an
        "area of interest"(ROI) within an image.

        Parameters
        ----------
        x : int
            ROI x position.
        y : int
            ROI y position.
        width : int
            ROI width.
        height : int
            ROI height.

        Returns
        -------
        int
            is_AOI return code.
        '''
        try:
            min_offset_x = self.node_map.FindNode('OffsetX_MinReg').Value()
            min_offset_y = self.node_map.FindNode('OffsetY_MinReg').Value()

            max_offset_x = self.node_map.FindNode('OffsetX_MaxReg').Value()
            max_offset_y = self.node_map.FindNode('OffsetY_MaxReg').Value()

            inc_offset_x = self.node_map.FindNode('OffsetX_IncReg').Value()
            inc_offset_y = self.node_map.FindNode('OffsetY_IncReg').Value()

            min_width = self.node_map.FindNode('WidthMinReg').Value()
            min_height = self.node_map.FindNode('HeightMinReg').Value()
            max_width = self.node_map.FindNode('WidthMaxReg').Value()
            max_height = self.node_map.FindNode('HeightMaxReg').Value()

            inc_width = self.node_map.FindNode('WidthIncReg').Value()
            inc_height = self.node_map.FindNode('HeightIncReg').Value()

            if x + width > max_width:
                x = max_width - width
            if y + height > max_height:
                y = max_height - height

            x = max(min_offset_x, min(x, max_offset_x))
            y = max(min_offset_y, min(y, max_offset_y))
            width = max(min_width, min(width, max_width))
            height = max(min_height, min(height, max_height))

            # correct to the nearest increment
            x = min_offset_x + round((x - min_offset_x) / inc_offset_x) * inc_offset_x
            y = min_offset_y + round((y - min_offset_y) / inc_offset_y) * inc_offset_y
            width = min_width + round((width - min_width) / inc_width) * inc_width
            height = min_height + round((height - min_height) / inc_height) * inc_height

            self.node_map.FindNode('OffsetX').SetValue(x)
            self.node_map.FindNode('OffsetY').SetValue(y)
            self.node_map.FindNode('Width').SetValue(width)
            self.node_map.FindNode('Height').SetValue(height)

            self.width = self.node_map.FindNode('Width').Value()
            self.height = self.node_map.FindNode('Height').Value()

            return True
        except Exception as e:
            logger.error(f'Error setting ROI: {e}')
            return False

    def reset_roi(self):
        '''Resets the ROI.

        Returns
        -------
        int
            is_AOI return code.
        '''
        try:
            self.node_map.FindNode('OffsetX').SetValue(0)
            self.node_map.FindNode('OffsetY').SetValue(0)

            max_width = self.node_map.FindNode('WidthMaxReg').Value()
            max_height = self.node_map.FindNode('HeightMaxReg').Value()

            self.node_map.FindNode('Width').SetValue(max_width)
            self.node_map.FindNode('Height').SetValue(max_height)

            self.width = self.node_map.FindNode('Width').Value()
            self.height = self.node_map.FindNode('Height').Value()

            return True
        except Exception as e:
            logger.error(f'Error resetting ROI: {e}')
            return False

    def get_device_clock_selector(self):
        '''Gets the current device clock selector.

        Returns
        -------
        int
            is_DeviceClockSelector return code.
        '''
        selector = (
            self.node_map.FindNode('DeviceClockSelector').CurrentEntry().SymbolicValue()
        )
        return selector

    def get_device_clock_selectors(self):
        '''Gets the available device clock selectors.

        Returns
        -------
        list of str
            List of available device clock selectors.
        '''
        return available_entries(self.node_map.FindNode('DeviceClockSelector'))

    def set_device_clock_selector(self, selector: str):
        '''Sets the device clock selector.

        Parameters
        ----------
        selector : str
            The selector to set. Must be the symbolic name.

        Returns
        -------
        int
            is_DeviceClockSelector return code.
        '''
        try:
            if (
                isinstance(selector, str)
                and selector in self.get_device_clock_selectors()
            ):
                self.node_map.FindNode('DeviceClockSelector').SetCurrentEntry(selector)
            else:
                raise ValueError('Selector must be a string and valid.')
            return True
        except Exception as e:
            logger.error(f'Error setting device clock selector: {e}')
            return False

    def get_device_clock_frequency(self, output=True):
        '''
        Gets the current device clock frequency.

        Returns
        -------
        int
            current device clock frequency in MHz.
        '''
        return self.node_map.FindNode('DeviceClockFrequencyInt').Value()

    def get_device_clock_frequency_range(self, output=True):
        '''
        Gets the device clock frequency range.

        Returns
        -------
        list
            [min, max, increment] device clock frequency in MHz.
        '''
        min_freq = self.node_map.FindNode('DeviceClockFrequencyInt').Minimum()
        max_freq = self.node_map.FindNode('DeviceClockFrequencyInt').Maximum()
        inc_freq = self.node_map.FindNode('DeviceClockFrequencyInt').Increment()

        if output:
            logger.info('Device Clock Frequency Range:')
            logger.info(f'Min: {min_freq} MHz')
            logger.info(f'Max: {max_freq} MHz')
            logger.info(f'Increment: {inc_freq} MHz')

        return [min_freq, max_freq, inc_freq]

    def get_device_clk_freq_valid_values(self):
        values = [
            int(e.real)
            for e in self.node_map.FindNode('DeviceClockFrequencyInt').ValidValues()
        ]
        if len(values) == 0:
            values.append(self.node_map.FindNode('DeviceClockFrequencyInt').Value())
        return values

    def set_device_clock_frequency(self, value: int):
        """Sets the camera's pixel clock speed.

        Parameters
        ----------
        value : uint
            The supported clock speed to set in MHz.

        Returns
        -------
        int
            is_PixelClock return code.
        """
        freq = self.node_map.FindNode('DeviceClockFrequencyInt')

        if value not in self.get_device_clk_freq_valid_values():
            return freq.Value()

        value = max(min(value, freq.Maximum()), freq.Minimum())

        freq.SetValue(value)

        return freq.Value()

    def get_exposure_range(self, output=True):
        '''Gets exposure range

        Parameters
        ----------
        output : bool, optional
            [description], by default True

        Returns
        -------
        list
            [min, max, increment] exposure time.
        '''
        min_exp = self.node_map.FindNode('ExposureTime').Minimum()
        max_exp = self.node_map.FindNode('ExposureTime').Maximum()
        inc_exp = self.node_map.FindNode('ExposureTime').Increment()

        if output:
            logger.info('Exposure Time Range:')
            logger.info(f'Min: {min_exp} {self.get_exposure_unit()}')
            logger.info(f'Max: {max_exp} {self.get_exposure_unit()}')
            logger.info(f'Increment: {inc_exp} {self.get_exposure_unit()}')

        return [min_exp, max_exp, inc_exp]

    def get_exposure_unit(self):
        '''Gets exposure unit

        Returns
        -------
        str
            exposure unit.
        '''
        return self.node_map.FindNode('ExposureTime').Unit()

    def get_exposure(self, output=True):
        '''
        Gets exposure time

        Parameters
        ----------
        output : bool, optional
            print out exposure time, by default True

        Returns
        -------
        float
            exposure time.
        '''
        self.exposure_current = self.node_map.FindNode('ExposureTime').Value()
        if output:
            logger.info(
                f'Current Exposure: {self.exposure_current} {self.get_exposure_unit()}'
            )
        return self.exposure_current

    def set_exposure(self, value: float):
        exp = self.node_map.FindNode('ExposureTime')

        value = max(min(value, exp.Maximum()), exp.Minimum())

        exp.SetValue(value)

        return self.get_exposure(False)

    def get_flash_duration_unit(self):
        '''Gets flash duration unit

        Returns
        -------
        str
            flash duration unit.
        '''
        return self.node_map.FindNode('FlashDuration').Unit()

    def get_flash_delay_unit(self):
        '''Gets flash delay unit

        Returns
        -------
        str
            flash delay unit.
        '''
        return self.node_map.FindNode('FlashStartDelay').Unit()

    def get_flash_duration_range(self, output=True):
        if not self.node_map.FindNode('FlashDuration').IsReadable():
            if output:
                logger.info('Flash Duration not readable.')
            return self.get_exposure_range(output)

        min_duration = self.node_map.FindNode('FlashDuration').Minimum()
        max_duration = self.node_map.FindNode('FlashDuration').Maximum()

        if output:
            unit = self.get_flash_duration_unit()
            logger.info('Flash Duration Range:')
            logger.info(f'Min: {min_duration} {unit}')
            logger.info(f'Max: {max_duration} {unit}')

        return [min_duration, max_duration]

    def get_flash_delay_range(self, output=True):
        min_delay = self.node_map.FindNode('FlashStartDelay').Minimum()
        max_delay = self.node_map.FindNode('FlashStartDelay').Maximum()
        inc_delay = self.node_map.FindNode('FlashStartDelay').Increment()

        if output:
            unit = self.get_flash_delay_unit()
            logger.info('Flash Delay Range:')
            logger.info(f'Min: {min_delay} {unit}')
            logger.info(f'Max: {max_delay} {unit}')
            logger.info(f'Increment: {inc_delay} {unit}')
        return [min_delay, max_delay, inc_delay]

    def get_flash_duration(self, output=True):
        if not self.node_map.FindNode('FlashDuration').IsReadable():
            if output:
                logger.info('Flash Duration not readable.')
            return self.get_exposure(output)

        duration = self.node_map.FindNode('FlashDuration').Value()
        if output:
            logger.info(
                f'Current Flash Duration: {duration} {self.get_flash_duration_unit()}'
            )
        return duration

    def get_flash_delay(self, output=True):
        delay = self.node_map.FindNode('FlashStartDelay').Value()
        if output:
            logger.info(f'Current Flash Delay: {delay} {self.get_flash_delay_unit()}')
        return delay

    def set_flash_duration(self, value: int):
        if not self.node_map.FindNode('FlashDuration').IsWriteable():
            logger.warning('Flash Duration not writable.')
            return 0

        dur = self.node_map.FindNode('FlashDuration')

        value = max(min(value, dur.Maximum()), dur.Minimum())

        dur.SetValue(value)

        return dur.Value()

    def set_flash_delay(self, value: int):
        delay = self.node_map.FindNode('FlashStartDelay')

        value = max(min(value, delay.Maximum()), delay.Minimum())

        delay.SetValue(value)

        return delay.Value()

    def get_flash_references(self):
        '''Gets available flash reference sources.'''
        return available_entries(self.node_map.FindNode('FlashReference'))

    def get_flash_reference(self, output=True):
        '''Gets current flash reference source.'''
        reference = (
            self.node_map.FindNode('FlashReference').CurrentEntry().SymbolicValue()
        )
        if output:
            logger.info(f'Current Flash Reference: {reference}')
        return reference

    def set_flash_reference(self, reference: str):
        '''Sets flash reference source.'''
        try:
            if isinstance(reference, str) and reference in self.get_flash_references():
                self.node_map.FindNode('FlashReference').SetCurrentEntry(reference)
            else:
                raise ValueError('Reference must be a string and valid entry.')
            return True
        except Exception as e:
            logger.error(f'Error setting flash reference: {e}')
            return False

    def get_framerate_range(self, output=True):
        min_fr = self.node_map.FindNode('AcquisitionFrameRate').Minimum()
        max_fr = self.node_map.FindNode('AcquisitionFrameRateMaxFunc').Maximum()

        if output:
            logger.info('Framerate Range:')
            logger.info(f'Min: {min_fr} Hz')
            logger.info(f'Max: {max_fr} Hz')

        return [min_fr, max_fr]

    def get_framerate(self, output=True):
        self.current_framerate = self.node_map.FindNode('AcquisitionFrameRate').Value()
        if output:
            logger.info('Framerate:')
            logger.info(f'Current FrameRate {self.current_framerate} Hz')
        return self.current_framerate

    def set_framerate(self, value):
        fr = self.node_map.FindNode('AcquisitionFrameRate')

        value = max(min(value, fr.Maximum()), fr.Minimum())

        fr.SetValue(value)

        return fr.Value()

    def get_trigger_modes(self):
        '''Gets available trigger modes.'''
        return available_entries(self.node_map.FindNode('TriggerMode'))

    def get_trigger_mode(self):
        self.trigger_mode = (
            self.node_map.FindNode('TriggerMode').CurrentEntry().SymbolicValue()
        )
        return self.trigger_mode

    def set_trigger_mode(self, mode: str):
        '''Sets trigger mode.'''
        try:
            if isinstance(mode, str) and mode in self.get_trigger_modes():
                self.node_map.FindNode('TriggerMode').SetCurrentEntry(mode)
            else:
                raise ValueError('Mode must be a string and valid entry.')
            return True
        except Exception as e:
            logger.error(f'Error setting trigger mode: {e}')
            return False

    def get_trigger_sources(self):
        '''Gets available trigger sources.'''
        return available_entries(self.node_map.FindNode('TriggerSource'))

    def get_trigger_source(self):
        '''Gets current trigger source.'''
        source = self.node_map.FindNode('TriggerSource').CurrentEntry().SymbolicValue()
        return source

    def set_trigger_source(self, source: str):
        '''Sets trigger source.'''
        try:
            if isinstance(source, str) and source in self.get_trigger_sources():
                self.node_map.FindNode('TriggerSource').SetCurrentEntry(source)
            else:
                raise ValueError('Source must be a string and valid entry.')
            return True
        except Exception as e:
            logger.error(f'Error setting trigger source: {e}')
            return False

    def _init_data_stream(self):
        try:
            if self._datastream is None:
                self._datastream = self._device.DataStreams()[0].OpenDataStream()

            payload_size = self.node_map.FindNode('PayloadSize').Value()
            buffer_amount = self._datastream.NumBuffersAnnouncedMinRequired() * 5

            for _ in range(buffer_amount):
                buffer = self._datastream.AllocAndAnnounceBuffer(payload_size)
                self._datastream.QueueBuffer(buffer)

            logger.info('Allocated buffers!')
        except Exception as e:
            logger.error(f'Exception in revoke_and_allocate_buffer: {e}')

    def software_trigger(self):
        if self.acquisition and self.get_trigger_source() == 'Software':
            logger.info('Executing software trigger...')
            self.node_map.FindNode('TriggerSoftware').Execute()
            self.node_map.FindNode('TriggerSoftware').WaitUntilDone()
            logger.info('Finished.')

    def start_acquisition(self):
        if self._device is None:
            return False
        if self.acquisition is True:
            return True

        self._init_data_stream()

        try:
            # Lock writable nodes, which could influence the payload size or
            # similar information during acquisition.
            # self.node_map.FindNode('TLParamsLocked').SetValue(1)

            image_width = self.node_map.FindNode('Width').Value()
            image_height = self.node_map.FindNode('Height').Value()
            input_pixel_format = ids_peak_ipl.PixelFormat(
                self.node_map.FindNode('PixelFormat').CurrentEntry().Value()
            )

            # Pre-allocate conversion buffers to speed up first image conversion
            # while the acquisition is running
            # NOTE: Re-create the image converter, so old conversion buffers
            #       get freed
            self._image_converter = ids_peak_ipl.ImageConverter()
            self._image_converter.PreAllocateConversion(
                input_pixel_format, TARGET_PIXEL_FORMAT, image_width, image_height
            )

            # Start acquisition both locally and on device.
            self._datastream.StartAcquisition()
            self.node_map.FindNode('AcquisitionStart').Execute()
            self.node_map.FindNode('AcquisitionStart').WaitUntilDone()
            self.acquisition = True

            logger.info('Acquisition started!')
        except Exception as e:
            logger.error(f'Exception (start acquisition): {str(e)}')
            return False
        return True

    def stop_acquisition(self):
        if self._device is None:
            return

        try:
            self.node_map.FindNode('AcquisitionStop').Execute()

            # Stop and flush the `DataStream`.
            # `KillWait` will cancel pending `WaitForFinishedBuffer` calls.
            # NOTE:
            #   One call to `KillWait` will cancel one pending `WaitForFinishedBuffer'
            #   For more information, refer to the documentation of `KillWait`.
            self._datastream.StopAcquisition(ids_peak.AcquisitionStopMode_Default)
            self._datastream.FlushPendingKillWaits()
            # Discard all buffers from the acquisition engine.
            # They remain in the announced buffer pool.
            self._datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)

            if self._datastream is not None:
                try:
                    for buffer in self._datastream.AnnouncedBuffers():
                        self._datastream.RevokeBuffer(buffer)
                except Exception as e:
                    logger.error(f'Exception revoking buffers: {str(e)}')

            self.acquisition = False

            # Unlock parameters
            # self.node_map.FindNode('TLParamsLocked').SetValue(0)
        except Exception as e:
            logger.error(f'Exception (stop acquisition): {str(e)}')

    def snap_image(self):
        if self.acquisition:
            return None

        image = None

        tigger_mode = self.get_trigger_mode()
        trigger_source = self.get_trigger_source()

        self.set_trigger_source('Software')
        self.set_trigger_mode('On')

        try:
            self.start_acquisition()

            self.software_trigger()

            # Wait until the next buffer is received.
            buffer = self._datastream.WaitForFinishedBuffer(1000)
            logger.info('Buffered image!')

            # Get image from buffer (shallow copy)
            self.ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)

            image = (
                self._image_converter.Convert(self.ipl_image, TARGET_PIXEL_FORMAT)
                .get_numpy_2D_16()
                .copy()
            )

            self._datastream.QueueBuffer(buffer)
        except Exception as e:
            logger.error(f'Exception in snap_image: {e}')
        finally:
            self.stop_acquisition()
            self.set_trigger_source(trigger_source)
            self.set_trigger_mode(tigger_mode)

        return image

    def property_tree(self) -> list[dict[str, Any]]:
        CLOCK_FREQ_SELECTOR = {
            'name': str(peakParams.CLOCK_FREQ_SELECTOR),
            'type': 'list',
            'limits': self.get_device_clock_selectors(),
            'value': self.get_device_clock_selector(),
        }
        CLOCK_FREQ = {
            'name': str(peakParams.CLOCK_FREQ),
            'type': 'list',
            'value': int(self.get_device_clock_frequency(False)),
            'limits': self.get_device_clk_freq_valid_values(),
            'suffix': 'MHz',
        }
        FRAMERATE = {
            'name': str(peakParams.FRAMERATE),
            'type': 'float',
            'value': self.get_framerate(False),
            'dec': False,
            'decimals': 6,
            'limits': [
                self.get_framerate_range(False)[0],
                self.get_framerate_range(False)[1],
            ],
            'suffix': 'Hz',
        }
        FRAME_AVERAGING = {
            'name': str(peakParams.FRAME_AVERAGING),
            'type': 'int',
            'value': 1,
            'dec': False,
            'decimals': 6,
            'limits': [1, 512],
            'suffix': 'frame',
        }
        TRIGGER_SOURCE = {
            'name': str(peakParams.TRIGGER_SOURCE),
            'type': 'list',
            'limits': self.get_trigger_sources(),
            'value': self.get_trigger_source(),
        }
        TRIGGER_MODE = {
            'name': str(peakParams.TRIGGER_MODE),
            'type': 'list',
            'limits': self.get_trigger_modes(),
            'value': self.get_trigger_mode(),
        }
        FLASH_REF = {
            'name': str(peakParams.FLASH_REF),
            'type': 'list',
            'limits': self.get_flash_references(),
            'value': self.get_flash_reference(),
        }
        FLASH_DURATION = {
            'name': str(peakParams.FLASH_DURATION),
            'type': 'int',
            'value': self.get_flash_duration(False),
            'dec': False,
            'decimals': 6,
            'suffix': self.get_flash_duration_unit(),
            'limits': [
                self.get_flash_duration_range(False)[0],
                self.get_flash_duration_range(False)[1],
            ],
        }
        FLASH_DELAY = {
            'name': str(peakParams.FLASH_START_DELAY),
            'type': 'int',
            'value': self.get_flash_delay(False),
            'dec': False,
            'decimals': 6,
            'suffix': self.get_flash_delay_unit(),
            'limits': [
                self.get_flash_delay_range(False)[0],
                self.get_flash_delay_range(False)[1],
            ],
        }

        PIXEL_FORMAT = {
            'name': str(peakParams.PIXEL_FORMAT),
            'type': 'list',
            'limits': self.get_pixel_formats(),
            'value': self.get_pixel_format(),
        }

        HOT_PIXEL_CORRECTION = {
            'name': str(peakParams.HOT_PIXEL_CORRECTION),
            'type': 'list',
            'limits': self.get_hot_pixel_correction_modes(),
            'value': self.get_hot_pixel_correction_mode(),
        }

        BINNING_SELECTOR = {
            'name': str(peakParams.BINNING_SELECTOR),
            'type': 'list',
            'limits': self.get_binning_selectors(),
            'value': self.get_binning_selector(),
        }
        BINNING_HORIZONTAL_MODE = {
            'name': str(peakParams.BINNING_H_MODE),
            'type': 'list',
            'limits': self.get_binning_horizontal_modes(),
            'value': self.get_binning_horizontal_mode(),
        }
        BINNING_HORIZONTAL = {
            'name': str(peakParams.BINNING_HORIZONTAL),
            'type': 'int',
            'value': self.get_binning_horizontal(),
            'limits': [
                self.node_map.FindNode('BinningHorizontal').Minimum(),
                self.node_map.FindNode('BinningHorizontal').Maximum(),
            ],
        }
        BINNING_VERTICAL_MODE = {
            'name': str(peakParams.BINNING_V_MODE),
            'type': 'list',
            'limits': self.get_binning_vertical_modes(),
            'value': self.get_binning_vertical_mode(),
        }
        BINNING_VERTICAL = {
            'name': str(peakParams.BINNING_VERTICAL),
            'type': 'int',
            'value': self.get_binning_vertical(),
            'limits': [
                self.node_map.FindNode('BinningVertical').Minimum(),
                self.node_map.FindNode('BinningVertical').Maximum(),
            ],
        }
        FREERUN = {
            'name': str(peakParams.FREERUN),
            'type': 'action',
            'parent': CamParams.ACQUISITION,
            'event': 'Event',
        }
        SOFTWARE_TRIGGER = {
            'name': str(peakParams.SOFTWARE_TRIGGER),
            'type': 'action',
            'parent': CamParams.ACQUISITION,
        }
        STOP = {
            'name': str(peakParams.STOP),
            'type': 'action',
            'parent': CamParams.ACQUISITION,
        }

        return [
            CLOCK_FREQ_SELECTOR,
            CLOCK_FREQ,
            FRAMERATE,
            FRAME_AVERAGING,
            TRIGGER_SOURCE,
            TRIGGER_MODE,
            FLASH_REF,
            FLASH_DURATION,
            FLASH_DELAY,
            PIXEL_FORMAT,
            HOT_PIXEL_CORRECTION,
            BINNING_SELECTOR,
            BINNING_HORIZONTAL_MODE,
            BINNING_HORIZONTAL,
            BINNING_VERTICAL_MODE,
            BINNING_VERTICAL,
            FREERUN,
            SOFTWARE_TRIGGER,
            STOP,
        ]

    def update_cam(self, param, path, param_value):
        if path is None:
            return

        param_value = param.value()

        try:
            param_name = peakParams('.'.join(path))
        except ValueError:
            return

        if param_name == peakParams.PIXEL_FORMAT:
            self.set_pixel_format(param_value)
        elif param_name == peakParams.HOT_PIXEL_CORRECTION:
            self.set_hot_pixel_correction_mode(param_value)
        elif param_name == peakParams.BINNING_SELECTOR:
            self.set_binning_selector(param_value)
            self.get_roi()
        elif param_name == peakParams.BINNING_H_MODE:
            self.set_binning_horizontal_mode(param_value)
            self.get_roi()
        elif param_name == peakParams.BINNING_HORIZONTAL:
            self.set_binning_horizontal(param_value)
            self.get_roi()
        elif param_name == peakParams.BINNING_V_MODE:
            self.set_binning_vertical_mode(param_value)
            self.get_roi()
        elif param_name == peakParams.BINNING_VERTICAL:
            self.set_binning_vertical(param_value)
            self.get_roi()
