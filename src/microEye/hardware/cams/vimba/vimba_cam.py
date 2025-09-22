import contextlib
import logging
from enum import Enum
from typing import Any, Optional

import numpy as np
from tabulate import tabulate

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.micam import miCamera
from microEye.hardware.cams.vimba import INSTANCE, vb


class VimbaParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    EXPOSURE_MODE = 'Acquisition Settings.Exposure Mode'
    EXPOSURE_AUTO = 'Acquisition Settings.Exposure Auto'
    FRAMERATE_ENABLED = 'Acquisition Settings.Frame Rate Enabled'
    FRAMERATE = 'Acquisition Settings.Frame Rate'
    TRIGGER_MODE = 'Acquisition Settings.Trigger Mode'
    TRIGGER_SOURCE = 'Acquisition Settings.Trigger Source'
    TRIGGER_SELECTOR = 'Acquisition Settings.Trigger Selector'
    TRIGGER_ACTIVATION = 'Acquisition Settings.Trigger Activation'
    PIXEL_FORMAT = 'Acquisition Settings.Pixel Format'
    LOAD = 'Acquisition Settings.Load Config'
    SAVE = 'Acquisition Settings.Save Config'
    LINE_SELECTOR = 'GPIOs.Line Selector'
    LINE_MODE = 'GPIOs.Line Mode'
    LINE_SOURCE = 'GPIOs.Line Source'
    LINE_INVERTER = 'GPIOs.Line Inverter'
    SET_IO_CONFIG = 'GPIOs.Set IO Config'
    TIMER_SELECTOR = 'Timers.Timer Selector'
    TIMER_ACTIVATION = 'Timers.Timer Activation'
    TIMER_SOURCE = 'Timers.Timer Source'
    TIMER_DELAY = 'Timers.Timer Delay'
    TIMER_DURATION = 'Timers.Timer Duration'
    TIMER_RESET = 'Timers.Timer Reset'
    SET_TIMER_CONFIG = 'Timers.Set Timer Config'

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


class vimba_cam(miCamera):
    '''A class to handle an Allied Vision camera.'''

    def __init__(self, camera_id=None):
        super().__init__(camera_id)

        self._vmb_context = None
        self._cam_context = None
        self.cam: Optional[vb.Camera] = None
        self.Cam_ID = camera_id
        self.cam = vimba_cam.get_camera(camera_id)
        self.name = self.cam.get_name()
        self.name = self.name.replace(' ', '_').replace('-', '_')

        self.temperature = -127

        self.exposure_current = 0
        self.exposure_increment = 0.1
        self.exposure_unit = 'us'

        self.exposure_mode = 'Timed'

        self.exposure_auto = 'Off'

        self.frameRateEnabled = False

        self.frameRate = 0.0

        self.frameRate_increment = 1
        self.frameRate_unit = ''
        self.frameRate_range = None

        self.trigger_source = 'Software'

        self.trigger_selector = 'FrameStart'

        self.trigger_source = 'On'

        self.trigger_activation = ''

        self.acquisition_mode = 'Continuous'
        self.acquisition = False

        self.pixel_format = None
        self.pixel_size = None
        self.bytes_per_pixel = 1

        self.width = None
        self.width_max = None
        self.width_range = None
        self.width_inc = None
        self.height = None
        self.height_max = None
        self.height_range = None
        self.height_inc = None
        self.offsetX = None
        self.offsetX_range = None
        self.offsetX_inc = None
        self.offsetY = None
        self.offsetY_range = None
        self.offsetY_inc = None

        self.trigger_modes = []
        self.trigger_sources = []
        self.trigger_selectors = []
        self.trigger_activations = []

        self.exposure_modes = []
        self.exposure_auto_entries = []

        self.pixel_formats = []

        self.initialize()

    def __enter__(self):
        '''Enter the context - initialize Vimba system and camera'''
        try:
            if INSTANCE is None:
                return None

            self._vmb_context = INSTANCE.__enter__()

            # Get camera
            if self.cam:
                # self.cam = vimba_cam.get_camera(self.Cam_ID)
                self._cam_context = self.cam.__enter__()

                return self.cam
            else:
                return INSTANCE
        except Exception as e:
            # Cleanup on error
            self.__exit__(type(e), e, e.__traceback__)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''Exit the context - cleanup camera and Vimba system'''
        # Exit camera context first
        if self._cam_context is not None and self.cam is not None:
            try:
                self.cam.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logging.warning(f'Error exiting camera context: {e}')
            finally:
                self._cam_context = None

        # Exit Vimba system context
        if self._vmb_context is not None and INSTANCE is not None:
            try:
                INSTANCE.__exit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logging.warning(f'Error exiting Vimba context: {e}')
            finally:
                self._vmb_context = None

        # Return False to propagate any exceptions
        return False

    @classmethod
    def get_camera_list(cls):
        cam_list = []

        if INSTANCE is None:
            return cam_list

        with INSTANCE as vimba:
            cams = vimba.get_all_cameras()
            for cam in cams:
                access_modes = cam.get_permitted_access_modes()
                in_use = 0 if vb.AccessMode.Full in access_modes else 1
                cam_list.append(
                    {
                        'Camera ID': cam.get_id(),
                        'Device ID': cam.get_interface_id(),
                        'Sensor ID': 'NA',
                        'Status': 'NA',
                        'InUse': in_use,
                        'Model': cam.get_model(),
                        'Serial': cam.get_serial(),
                        'Driver': 'Vimba',
                        'Name': cam.get_name(),
                    }
                )
        return cam_list

    @classmethod
    def get_camera(cls, camera_id: Optional[str]):
        if INSTANCE is None:
            return None

        with INSTANCE as vimba:
            if camera_id:
                try:
                    return vimba.get_camera_by_id(camera_id)
                except vb.VmbCameraError:
                    print(f"Failed to access Camera '{camera_id}'. Abort.")
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    print('No Cameras accessible. Abort.')
                    return None

                return cams[0]

    def initialize(self):
        with self:
            self.default()
            self.get_acquisition_framerate_enable(False)
            self.get_framerate(False)
            self.get_exposure(False)
            self.get_exposure_mode(False)
            self.get_exposure_auto(False)
            self.get_trigger_source(False)
            self.get_trigger_selector(False)
            self.get_trigger_mode(False)
            self.get_trigger_activation(False)
            self.get_acquisition_mode(False)
            self.get_pixel_format(False)
            self.get_pixel_size(False)
            self.get_roi(False)
            self.get_temperature()

            self.trigger_modes = self.get_trigger_modes()
            self.trigger_sources = self.get_trigger_sources()
            self.trigger_selectors = self.get_trigger_selectors()
            self.trigger_activations = self.get_trigger_activations()

            self.exposure_modes = self.get_exposure_modes()
            self.exposure_auto_entries = self.get_exposure_auto_entries()

            self.pixel_formats = self.get_pixel_formats()

            if 'Mono12' in self.pixel_formats:
                self.set_pixel_format('Mono12')

            self.print_status()

    def populate_status(self):
        self.status['Camera'] = {'Name': self.name}

        self.status['Temperature'] = {'Value': self.temperature, 'Unit': 'Celsius'}

        self.status['Exposure'] = {
            'Mode': self.exposure_mode,
            'Value': self.exposure_current,
            'Unit': self.exposure_unit,
            'Increment': self.exposure_increment,
            'Auto': self.exposure_auto,
        }

        self.status['Framerate'] = {
            'Value': self.frameRate,
            'Unit': self.frameRate_unit,
            'Range': self.frameRate_range,
            'Increment': self.frameRate_increment,
            'Enabled': self.frameRateEnabled,
        }

        self.status['Acquisition'] = {
            'Mode': self.acquisition_mode,
            'Enabled': self.acquisition,
        }

        # self.status['Trigger'] = {
        #     'source': self.trigger_source,
        #     'selector': self.trigger_selector,
        #     'activation': self.trigger_activation,
        # }
        self.status['Image Format'] = {
            'Pixel Format': self.pixel_format,
            'Pixel Size': self.pixel_size,
            'Bytes per Pixel': self.bytes_per_pixel,
        }

        self.status['Image Size'] = {
            'Width': self.width,
            'Width_max': self.width_max,
            'Width_range': self.width_range,
            'Width_inc': self.width_inc,
            'Height': self.height,
            'Height_max': self.height_max,
            'Height_range': self.height_range,
            'Height_inc': self.height_inc,
            'Offset_x': self.offsetX,
            'Offset_x_range': self.offsetX_range,
            'Offset_x_inc': self.offsetX_inc,
            'Offset_y': self.offsetY,
            'Offset_y_range': self.offsetY_range,
            'Offset_y_inc': self.offsetY_inc,
        }

    def default(self):
        with self:
            try:
                self.cam.UserSetSelector.set('Default')

            except (AttributeError, vb.VmbFeatureError):
                print("Failed to set Feature 'UserSetSelector'")

            try:
                self.cam.UserSetLoad.run()
                print('--> All feature values have been restored to default')

            except (AttributeError, vb.VmbFeatureError):
                print("Failed to run Feature 'UserSetLoad'")

    def get_metadata(self):
        return {
            'CHANNEL_NAME': self.name,
            'DET_MANUFACTURER': 'Allied Vision',
            'DET_MODEL': self.cam.get_model(),
            'DET_SERIAL': self.cam.get_serial(),
            'DET_TYPE': 'CMOS',
        }

    def get_temperature(self):
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.temperature = -127
        # with self.cam:
        with contextlib.suppress(Exception):
            self.temperature = self.cam.get_feature_by_name('DeviceTemperature').get()
        return self.temperature

    # Get exposure
    def get_exposure(self, output=True):
        exp = -127
        try:
            # with self.cam:
            try:
                exposure = self.cam.ExposureTime
                self.exposure_current = exposure.get()
                self.exposure_increment = exposure.get_increment()
                self.exposure_unit = exposure.get_unit()
                self.exposure_range = exposure.get_range()
            except Exception:
                exposure = self.cam.ExposureTimeAbs
                self.exposure_current = exposure.get()
                self.exposure_increment = self.cam.ExposureTimeIncrement.get()
                self.exposure_unit = exposure.get_unit()
                self.exposure_range = exposure.get_range()
            if output:
                print('Current Exposure ', self.exposure_current, self.exposure_unit)
            return self.exposure_current
        except Exception:
            print('Exposure Get ERROR')
        return exp

    def set_exposure(self, value: float):
        try:
            # with self.cam:
            try:
                exposure = self.cam.ExposureTime
                self.exposure_increment = exposure.get_increment()
                self.exposure_unit = exposure.get_unit()
                self.exposure_range = exposure.get_range()
            except Exception:
                exposure = self.cam.ExposureTimeAbs
                self.exposure_increment = self.cam.ExposureTimeIncrement.get()
                self.exposure_unit = exposure.get_unit()
                self.exposure_range = exposure.get_range()
            floor_value = (
                self.exposure_increment
                * np.floor((value - self.exposure_range[0]) / self.exposure_increment)
                + self.exposure_range[0]
            )
            ceil_value = (
                self.exposure_increment
                * np.ceil((value - self.exposure_range[0]) / self.exposure_increment)
                + self.exposure_range[0]
            )
            set_value = 0
            if np.abs(floor_value - value) < np.abs(ceil_value - value):
                set_value = floor_value
            else:
                set_value = ceil_value
            set_value = max(
                min(set_value, self.exposure_range[1]), self.exposure_range[0]
            )
            exposure.set(set_value)
            self.get_exposure()
            self.get_framerate()
            return 1
        except Exception:
            print('Exposure Set ERROR')
            return 0

    def set_black_level(self, value: float):
        try:
            blackLevel = self.cam.BlackLevel
            blackLevel.set(value)
            return 1
        except Exception:
            print('Black Level Set ERROR')
            return 0

    def set_exposure_mode(self, value: str = 'Timed'):
        try:
            # with self.cam:
            ExposureMode = self.cam.ExposureMode
            entries = map(str, ExposureMode.get_available_entries())
            if value in entries:
                ExposureMode.set(value)
            return 1
        except Exception:
            print('ExposureMode Set ERROR')
            return 0

    def get_exposure_mode(self, output=True):
        try:
            # with self.cam:
            ExposureMode = self.cam.ExposureMode
            self.exposure_mode = ExposureMode.get()
            if output:
                print('Exposure mode ', self.exposure_mode)
            return self.exposure_mode
        except Exception:
            print('ExposureMode get ERROR')
            return 'NA'

    def get_exposure_modes(self):
        modes = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            ExposureMode = self.cam.ExposureMode
            entries = ExposureMode.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('ExposureModes Get ERROR')

        return [modes, displayNames, tooltips]

    def set_exposure_auto(self, value: str = 'Off'):
        try:
            # with self.cam:
            ExposureAuto = self.cam.ExposureAuto
            entries = map(str, ExposureAuto.get_available_entries())
            if value in entries:
                ExposureAuto.set(value)
                return 1
            else:
                return 0
        except Exception:
            print('ExposureAuto Set ERROR')
            return 0

    def get_exposure_auto(self, output=True):
        try:
            # with self.cam:
            ExposureAuto = self.cam.ExposureAuto
            self.exposure_auto = ExposureAuto.get()
            if output:
                print('Exposure Auto ', self.exposure_auto)
            return self.exposure_auto
        except Exception:
            print('ExposureAuto get ERROR')
            return 'NA'

    def get_exposure_auto_entries(self):
        modes = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            ExposureAuto = self.cam.ExposureAuto
            entries = ExposureAuto.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('ExposureAutos Get ERROR')

        return [modes, displayNames, tooltips]

    def set_acquisition_framerate_enable(self, enabled: bool = False):
        try:
            AcquisitionFrameRateEnable = self.cam.AcquisitionFrameRateEnable
            AcquisitionFrameRateEnable.set(enabled)
            self.frameRateEnabled = enabled
        except Exception:
            print('AcquisitionFrameRateEnable set ERROR')

    def get_acquisition_framerate_enable(self, output: bool = True):
        try:
            AcquisitionFrameRateEnable = self.cam.AcquisitionFrameRateEnable
            self.frameRateEnabled = AcquisitionFrameRateEnable.get()
            if output:
                print('AcquisitionFrameRate Enabled ', self.frameRateEnabled)
        except Exception:
            print('AcquisitionFrameRateEnable get ERROR')

    def get_framerate(self, output: bool = True):
        try:
            frameRate = self.cam.AcquisitionFrameRate
            self.frameRate = frameRate.get()
            self.frameRate_unit = frameRate.get_unit()
            self.frameRate_range = frameRate.get_range()
            if output:
                print('Current FrameRate ', self.frameRate, self.frameRate_unit)
            return self.frameRate
        except Exception:
            print('AcquisitionFrameRate get ERROR')
            self.frameRate = 0.0
            return self.frameRate

    def set_framerate(self, value: float):
        if not self.frameRateEnabled:
            return 0

        try:
            frameRate = self.cam.AcquisitionFrameRate
            self.frameRate_unit = frameRate.get_unit()
            self.frameRate_range = frameRate.get_range()

            set_value = max(
                min(value, self.frameRate_range[1]), self.frameRate_range[0]
            )
            frameRate.set(set_value)
            self.get_framerate(False)
            self.get_exposure(False)
            return 1
        except Exception:
            print('AcquisitionFrameRate Set ERROR')
            return 0

    def set_trigger_mode(self, value: str = 'On'):
        try:
            # with self.cam:
            TriggerMode = self.cam.TriggerMode
            entries = map(str, TriggerMode.get_available_entries())
            if value in entries:
                TriggerMode.set(value)
                return 1
            else:
                return 0
        except Exception:
            print('trigger Set ERROR')
            return 0

    def get_trigger_mode(self, output=True):
        try:
            # with self.cam:
            TriggerMode = self.cam.TriggerMode
            self.trigger_mode = TriggerMode.get()
            if output:
                print('Trigger mode ', self.trigger_mode)
            return self.trigger_mode
        except Exception:
            print('trigger get ERROR')
            return 'NA'

    def get_trigger_modes(self):
        modes = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            TriggerMode = self.cam.TriggerMode
            entries = TriggerMode.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('trigger modes Get ERROR')

        return [modes, displayNames, tooltips]

    def set_trigger_source(self, value: str = 'Software'):
        try:
            # with self.cam:
            TriggerSource = self.cam.TriggerSource
            entries = map(str, TriggerSource.get_available_entries())
            if value in entries:
                TriggerSource.set(value)
                return 1
            else:
                return 0
        except Exception:
            print('trigger source Set ERROR')
            return 0

    def get_trigger_source(self, output=True):
        try:
            # with self.cam:
            TriggerSource = self.cam.TriggerSource
            self.trigger_source = TriggerSource.get()
            if output:
                print('Trigger source ', self.trigger_source)
            return self.trigger_source
        except Exception:
            print('trigger source get ERROR')
            return 'NA'

    def get_trigger_sources(self):
        sources = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            TriggerSource = self.cam.TriggerSource
            entries = TriggerSource.get_available_entries()
            sources += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('trigger sources Get ERROR')

        return [sources, displayNames, tooltips]

    def set_trigger_selector(self, value: str = 'FrameStart'):
        try:
            # with self.cam:
            TriggerSelector = self.cam.TriggerSelector
            entries = map(str, TriggerSelector.get_available_entries())
            if value in entries:
                TriggerSelector.set(value)
                return 1
            else:
                return 0
        except Exception:
            print('trigger selector Set ERROR')
            return 0

    def get_trigger_selector(self, output=True):
        try:
            # with self.cam:
            TriggerSelector = self.cam.TriggerSelector
            self.trigger_selector = TriggerSelector.get()
            if output:
                print('Trigger selector ', self.trigger_selector)
            return self.trigger_selector
        except Exception:
            print('trigger selector get ERROR')
            return 'NA'

    def get_trigger_selectors(self):
        selectors = []
        displayNames = []
        tooltips = []
        try:
            # with self.cam:
            TriggerSelector = self.cam.TriggerSelector
            entries = TriggerSelector.get_available_entries()
            selectors += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('trigger selectors Get ERROR')

        return [selectors, displayNames, tooltips]

    def set_trigger_activation(self, value: str = 'RisingEdge'):
        try:
            # with self.cam:
            TriggerActivation = self.cam.TriggerActivation
            entries = map(str, TriggerActivation.get_available_entries())
            if value in entries:
                TriggerActivation.set(value)
                return 1
            else:
                return 0
        except Exception:
            print('TriggerActivation Set ERROR')
            return 0

    def get_trigger_activation(self, output=True):
        try:
            # with self.cam:
            TriggerActivation = self.cam.TriggerActivation
            self.trigger_activation = TriggerActivation.get()
            if output:
                print('Trigger activation ', self.trigger_activation)
            return self.trigger_activation
        except Exception:
            print('TriggerActivation get ERROR')
            return 'NA'

    def get_trigger_activations(self):
        activations = []
        displayNames = []
        tooltips = []
        try:
            # with self.cam:
            TriggerActivation = self.cam.TriggerActivation
            entries = TriggerActivation.get_available_entries()
            activations += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('TriggerActivation Get ERROR')

        return [activations, displayNames, tooltips]

    def set_acquisition_mode(self, value: str = 'Continuous'):
        try:
            # with self.cam:
            AcquisitionMode = self.cam.AcquisitionMode
            entries = map(str, AcquisitionMode.get_available_entries())
            if value in entries:
                AcquisitionMode.set(value)
                return 1
            else:
                return 0
        except Exception:
            print('AcquisitionMode Set ERROR')
            return 0

    def get_acquisition_mode(self, output=True):
        try:
            # with self.cam:
            AcquisitionMode = self.cam.AcquisitionMode
            self.acquisition_mode = AcquisitionMode.get()
            if output:
                print('Acquisition mode ', self.acquisition_mode)
            return self.acquisition_mode
        except Exception:
            print('AcquisitionMode get ERROR')
            return 'NA'

    def get_acquisition_modes(self):
        modes = []
        displayNames = []
        tooltips = []
        try:
            # with self.cam:
            AcquisitionMode = self.cam.AcquisitionMode
            entries = AcquisitionMode.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip for entry in entries]
        except Exception:
            print('AcquisitionModes Get ERROR')

        return [modes, displayNames, tooltips]

    def get_pixel_formats(self):
        formats = []
        try:
            # with self.cam:
            fmts = self.cam.get_pixel_formats()
            formats += map(str, fmts)
        except Exception:
            print('Pixel Formats Get ERROR')

        return formats

    def get_pixel_format(self, output=True):
        formats = []
        try:
            # with self.cam:
            self.pixel_format = self.cam.get_pixel_format()
            if output:
                print('Pixel Format', self.pixel_format)
            return self.pixel_format
        except Exception:
            print('Pixel Format Get ERROR')
            return 'NA'

    def set_pixel_format(self, value: str):
        try:
            # with self.cam:
            if value in vb.PixelFormat.__members__:
                self.cam.set_pixel_format(vb.PixelFormat[value])
                self.get_pixel_format()
                self.get_pixel_size()
                return 1
            else:
                return 0
        except Exception:
            pFormat = str(self.cam.get_pixel_format())
            if '8' in pFormat:
                self.pixel_size = 8
                self.bytes_per_pixel = 1
                return self.pixel_size
            elif '12p' in pFormat:
                print('Pixel Format Not supported.')
                return 'NA'
            elif '12' in pFormat:
                self.pixel_size = 16
                self.bytes_per_pixel = 2
                return self.pixel_size
            else:
                print('Pixel Format Not supported.')
                return 'NA'
        finally:
            self.set_black_level(5.0)

    def get_pixel_size(self, output=True):
        try:
            # with self.cam:
            self.pixel_size = self.cam.PixelSize.get()
            self.bytes_per_pixel = int(np.ceil(int(self.pixel_size) / 8))
            if output:
                print('Pixel Size', self.pixel_size)
            return self.pixel_size
        except Exception:
            print('Pixel Size Get ERROR')
            return 'NA'

    def get_io_lines(self):
        try:
            lines = []
            for line in self.cam.LineSelector.get_available_entries():
                lines.append(str(line))
            return lines
        except Exception:
            print('get_io_lines ERROR')
            return None

    def get_line_modes(self):
        try:
            modes = []
            for mode in self.cam.LineMode.get_available_entries():
                modes.append(str(mode))
            return modes
        except Exception:
            print('get_line_modes ERROR')
            return None

    def get_line_sources(self):
        try:
            sources = []
            for source in self.cam.LineSource.get_available_entries():
                sources.append(str(source))
            return sources
        except Exception:
            print('get_line_sources ERROR')
            return None

    def get_line_source(self):
        try:
            return str(self.cam.LineSource.get())
        except Exception:
            print('get_line_source ERROR')
            return None

    def get_line_mode(self):
        try:
            return str(self.cam.LineMode.get())
        except Exception:
            print('get_line_mode ERROR')
            return None

    def get_line_inverter(self):
        try:
            return self.cam.LineInverter.get()
        except Exception:
            print('get_line_inverter ERROR')
            return None

    def get_line_status(self):
        try:
            return self.cam.LineStatus.get()
        except Exception:
            print('get_line_status ERROR')
            return None

    def set_line_inverter(self, value: bool):
        try:
            self.cam.LineInverter.set(value)
            return True
        except Exception:
            print('set_line_inverter ERROR')
            return False

    def set_line_source(self, value: str):
        try:
            self.cam.LineSource.set(value)
            return True
        except Exception:
            print('set_line_source ERROR')
            return False

    def set_line_mode(self, value: str):
        try:
            self.cam.LineMode.set(value)
            return True
        except Exception:
            print('set_line_mode ERROR')
            return False

    def select_io_line(self, value: str):
        try:
            self.cam.LineSelector.set(value)
            return True
        except Exception:
            print('select_io_line ERROR')
            return False

    def get_timers(self):
        try:
            timers = []
            for timer in self.cam.TimerSelector.get_available_entries():
                timers.append(str(timer))
            return timers
        except Exception:
            print('get_timers ERROR')
            return None

    def get_timer_trigger_activations(self):
        try:
            modes = []
            for mode in self.cam.TimerTriggerActivation.get_available_entries():
                modes.append(str(mode))
            return modes
        except Exception:
            print('get_timer_trigger_activations ERROR')
            return None

    def get_timer_trigger_sources(self):
        try:
            sources = []
            for source in self.cam.TimerTriggerSource.get_available_entries():
                sources.append(str(source))
            return sources
        except Exception:
            print('get_timer_trigger_sources ERROR')
            return None

    def get_timer_trigger_source(self):
        try:
            return str(self.cam.TimerTriggerSource.get())
        except Exception:
            print('get_timer_trigger_source ERROR')
            return None

    def get_timer_trigger_activation(self):
        try:
            return str(self.cam.TimerTriggerActivation.get())
        except Exception:
            print('get_timer_trigger_activation ERROR')
            return None

    def get_timer_status(self):
        try:
            return self.cam.TimerStatus.get()
        except Exception:
            print('get_timer_status ERROR')
            return None

    def set_timer_trigger_source(self, value: str):
        try:
            self.cam.TimerTriggerSource.set(value)
            return True
        except Exception:
            print('set_timer_trigger_source ERROR')
            return False

    def set_timer_trigger_activation(self, value: str):
        try:
            self.cam.TimerTriggerActivation.set(value)
            return True
        except Exception:
            print('set_line_mode ERROR')
            return False

    def get_timer_duration(self):
        try:
            duration = self.cam.TimerDuration
            return duration.get(), duration.get_range()
        except Exception:
            print('get_timer_duration ERROR')

    def get_timer_delay(self):
        try:
            delay = self.cam.TimerDelay
            return delay.get(), delay.get_range()
        except Exception:
            print('get_timer_delay ERROR')

    def set_timer_duration(self, value: float):
        try:
            self.cam.TimerDuration.set(value)
            return True
        except Exception:
            print('set_timer_duration ERROR')
            return False

    def set_timer_delay(self, value: float):
        try:
            self.cam.TimerDelay.set(value)
            return True
        except Exception:
            print('set_timer_delay ERROR')
            return False

    def select_timer(self, value: str):
        try:
            self.cam.TimerSelector.set(value)
            return True
        except Exception:
            print('select_timer ERROR')
            return False

    def reset_timer(self):
        try:
            self.cam.TimerReset.run()
            return True
        except Exception:
            print('reset_timer ERROR')
            return False

    def get_roi(self, output=True):
        try:
            # with self.cam:
            self.width = self.cam.Width.get()
            self.width_max = self.cam.WidthMax.get()
            self.width_range = self.cam.Width.get_range()
            self.width_inc = self.cam.Width.get_increment()
            self.height = self.cam.Height.get()
            self.height_max = self.cam.HeightMax.get()
            self.height_range = self.cam.Height.get_range()
            self.height_inc = self.cam.Height.get_increment()
            self.offsetX = self.cam.OffsetX.get()
            self.offsetX_range = self.cam.OffsetX.get_range()
            self.offsetX_inc = self.cam.OffsetX.get_increment()
            self.offsetY = self.cam.OffsetY.get()
            self.offsetY_range = self.cam.OffsetY.get_range()
            self.offsetY_inc = self.cam.OffsetY.get_increment()
            if output:
                print(
                    f'X = {self.offsetX} Y = {self.offsetY}',
                    f'W = {self.width} H = {self.height}',
                )
            return (
                int(self.offsetX),
                int(self.offsetY),
                int(self.width),
                int(self.height),
            )
        except Exception:
            print('ROI Get ERROR')
            return 'NA'

    def set_roi(self, x, y, width, height):
        try:
            self.reset_roi()

            self.width_range = self.cam.Width.get_range()
            self.width_inc = self.cam.Width.get_increment()
            self.width = self.get_nearest(self.width_range, self.width_inc, width)
            self.cam.Width.set(self.width)

            self.height_range = self.cam.Height.get_range()
            self.height_inc = self.cam.Height.get_increment()
            self.height = self.get_nearest(self.height_range, self.height_inc, height)
            self.cam.Height.set(self.height)

            if x is None:
                x = (self.width_range[1] - self.width) / 2

            self.offsetX_range = self.cam.OffsetX.get_range()
            self.offsetX_inc = self.cam.OffsetX.get_increment()
            self.offsetX = self.get_nearest(self.offsetX_range, self.offsetX_inc, x)
            self.cam.OffsetX.set(self.offsetX)

            if y is None:
                y = (self.height_range[1] - self.height) / 2

            self.offsetY_range = self.cam.OffsetY.get_range()
            self.offsetY_inc = self.cam.OffsetY.get_increment()
            self.offsetY = self.get_nearest(self.offsetY_range, self.offsetY_inc, y)
            self.cam.OffsetY.set(self.offsetY)
            return 1
        except Exception:
            print('ROI Set ERROR')
            return 0

    def reset_roi(self):
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        self.cam.Width.set(self.width_max)
        self.cam.Height.set(self.height_max)

    def get_nearest(self, vrange, step, value):
        values = np.arange(vrange[0], vrange[1] + step / 4, step)
        return values[np.abs(values - value).argmin()]

    def property_tree(self) -> list[dict[str, Any]]:
        '''Return a list of parameter dicts for Vimba camera options.'''
        EXPOSURE_MODE = {
            'name': str(VimbaParams.EXPOSURE_MODE),
            'type': 'list',
            'limits': self.exposure_modes[0] if self.exposure_modes else [],
            'value': self.exposure_mode,
        }
        EXPOSURE_AUTO = {
            'name': str(VimbaParams.EXPOSURE_AUTO),
            'type': 'list',
            'limits': self.exposure_auto_entries[0]
            if self.exposure_auto_entries
            else [],
            'value': self.exposure_auto,
        }
        FRAMERATE_ENABLED = {
            'name': str(VimbaParams.FRAMERATE_ENABLED),
            'type': 'bool',
            'value': False,
        }
        FRAMERATE = {
            'name': str(VimbaParams.FRAMERATE),
            'type': 'float',
            'value': self.frameRate,
            'dec': False,
            'decimals': 6,
            'step': 0.1,
            'limits': [self.frameRate_range[0], self.frameRate_range[1]],
            'suffix': self.frameRate_unit,
            'enabled': False,
        }
        TRIGGER_MODE = {
            'name': str(VimbaParams.TRIGGER_MODE),
            'type': 'list',
            'limits': self.trigger_modes[0],
        }
        TRIGGER_SOURCE = {
            'name': str(VimbaParams.TRIGGER_SOURCE),
            'type': 'list',
            'limits': self.trigger_sources[0],
        }
        TRIGGER_SELECTOR = {
            'name': str(VimbaParams.TRIGGER_SELECTOR),
            'type': 'list',
            'limits': self.trigger_selectors[0],
        }
        TRIGGER_ACTIVATION = {
            'name': str(VimbaParams.TRIGGER_ACTIVATION),
            'type': 'list',
            'limits': self.trigger_activations[0],
        }
        PIXEL_FORMAT = {
            'name': str(VimbaParams.PIXEL_FORMAT),
            'type': 'list',
            'limits': self.pixel_formats if self.pixel_formats else [],
            'value': self.pixel_format.name if self.pixel_format else None,
        }

        FREERUN = {
            'name': str(VimbaParams.FREERUN),
            'type': 'action',
            'parent': CamParams.ACQUISITION,
        }
        STOP = {
            'name': str(VimbaParams.STOP),
            'type': 'action',
            'parent': CamParams.ACQUISITION,
        }

        LOAD = {'name': str(VimbaParams.LOAD), 'type': 'action'}
        SAVE = {'name': str(VimbaParams.SAVE), 'type': 'action'}

        # GPIOs
        with self:
            LINE_SELECTOR = {
                'name': str(VimbaParams.LINE_SELECTOR),
                'type': 'list',
                'limits': self.get_io_lines(),
                'parent': CamParams.CAMERA_GPIO,
            }
            LINE_MODE = {
                'name': str(VimbaParams.LINE_MODE),
                'type': 'list',
                'limits': self.get_line_modes(),
                'parent': CamParams.CAMERA_GPIO,
            }
            LINE_SOURCE = {
                'name': str(VimbaParams.LINE_SOURCE),
                'type': 'list',
                'limits': self.get_line_sources(),
                'parent': CamParams.CAMERA_GPIO,
            }
            LINE_INVERTER = {
                'name': str(VimbaParams.LINE_INVERTER),
                'type': 'bool',
                'value': False,
                'parent': CamParams.CAMERA_GPIO,
            }

        SET_IO_CONFIG = {
            'name': str(VimbaParams.SET_IO_CONFIG),
            'type': 'action',
            'parent': CamParams.CAMERA_GPIO,
        }

        # Timers
        TIMER_SELECTOR = {
            'name': str(VimbaParams.TIMER_SELECTOR),
            'type': 'list',
            'parent': CamParams.CAMERA_TIMERS,
        }
        TIMER_ACTIVATION = {
            'name': str(VimbaParams.TIMER_ACTIVATION),
            'type': 'list',
            'parent': CamParams.CAMERA_TIMERS,
        }
        TIMER_SOURCE = {
            'name': str(VimbaParams.TIMER_SOURCE),
            'type': 'list',
            'parent': CamParams.CAMERA_TIMERS,
        }
        TIMER_DELAY = {
            'name': str(VimbaParams.TIMER_DELAY),
            'type': 'float',
            'dec': False,
            'decimals': 6,
            'parent': CamParams.CAMERA_TIMERS,
        }
        TIMER_DURATION = {
            'name': str(VimbaParams.TIMER_DURATION),
            'type': 'float',
            'dec': False,
            'decimals': 6,
            'parent': CamParams.CAMERA_TIMERS,
        }
        TIMER_RESET = {
            'name': str(VimbaParams.TIMER_RESET),
            'type': 'action',
            'parent': CamParams.CAMERA_TIMERS,
        }
        SET_TIMER_CONFIG = {
            'name': str(VimbaParams.SET_TIMER_CONFIG),
            'type': 'action',
            'parent': CamParams.CAMERA_TIMERS,
        }

        with self:
            timers = self.get_timers()
            if timers:
                TIMER_SELECTOR['limits'] = timers
                TIMER_ACTIVATION['limits'] = self.get_timer_trigger_activations()
                TIMER_SOURCE['limits'] = self.get_timer_trigger_sources()

        return [
            EXPOSURE_MODE,
            EXPOSURE_AUTO,
            FRAMERATE_ENABLED,
            FRAMERATE,
            TRIGGER_MODE,
            TRIGGER_SOURCE,
            TRIGGER_SELECTOR,
            TRIGGER_ACTIVATION,
            PIXEL_FORMAT,
            FREERUN,
            STOP,
            LOAD,
            SAVE,
            LINE_SELECTOR,
            LINE_MODE,
            LINE_SOURCE,
            LINE_INVERTER,
            SET_IO_CONFIG,
            TIMER_SELECTOR,
            TIMER_ACTIVATION,
            TIMER_SOURCE,
            TIMER_DELAY,
            TIMER_DURATION,
            TIMER_RESET,
            SET_TIMER_CONFIG,
        ]

    def update_cam(self, param, path, param_value):
        if path is None:
            return

        param_value = param.value()

        try:
            param_name = VimbaParams('.'.join(path))
        except ValueError:
            return

        with self:
            if param_name == VimbaParams.TRIGGER_MODE:
                self.set_trigger_mode(param_value)
                self.get_trigger_mode()
            elif param_name == VimbaParams.TRIGGER_SOURCE:
                self.set_trigger_source(param_value)
                self.get_trigger_source()
            elif param_name == VimbaParams.TRIGGER_SELECTOR:
                self.set_trigger_selector(param_value)
                self.get_trigger_selector()
            elif param_name == VimbaParams.TRIGGER_ACTIVATION:
                self.set_trigger_activation(param_value)
                self.get_trigger_activation()
            elif param_name == VimbaParams.EXPOSURE_MODE:
                self.set_exposure_mode(param_value)
                self.get_exposure_mode()
            elif param_name == VimbaParams.EXPOSURE_AUTO:
                self.set_exposure_auto(param_value)
                self.get_exposure_auto()
            elif param_name == VimbaParams.PIXEL_FORMAT:
                self.set_pixel_format(param_value)


# if __name__ == '__main__':
# camera = vimba_cam('')

# with camera.vimba, camera.cam:
#     camera.set_exposure(-1)
#     camera.get_exposure()
#     camera.set_pixel_format('Mono12')
#     camera.get_pixel_format()
#     camera.set_trigger_mode()
#     camera.set_roi(515, 515)
#     camera.get_roi()
#     print(camera.get_pixel_formats())
