
from typing import Optional
import numpy as np

from .micam import miCamera

try:
    import vimba as vb

    def get_camera_list():
        cam_list = []
        with vb.Vimba.get_instance() as vimba:
            cams = vimba.get_all_cameras()
            for cam in cams:
                cam_list.append({
                            "camID": cam.get_id(),
                            "devID": cam.get_interface_id(),
                            "senID": 'NA',
                            "Status": 'NA',
                            "InUse": 0,
                            "Model": cam.get_model(),
                            "Serial": cam.get_serial(),
                            "Driver": 'Vimba',
                            "Name": cam.get_name()})
        return cam_list

    def get_camera(camera_id: Optional[str]) -> vb.Camera:
        with vb.Vimba.get_instance() as vimba:
            if camera_id:
                try:
                    return vimba.get_camera_by_id(camera_id)
                except vb.VimbaCameraError:
                    print(
                        'Failed to access Camera \'{}\'. Abort.'.format(
                            camera_id))
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    print('No Cameras accessible. Abort.')
                    return None

                return cams[0]

except Exception:
    vb = None

    def get_camera_list():
        return []

    def get_camera(camera_id: Optional[str]):
        return None


class vimba_cam(miCamera):
    '''A class to handle an Allied Vision camera.'''

    def __init__(self, camera_id=None):
        super().__init__(camera_id)

        self.vimba = vb.Vimba.get_instance()
        self.cam = get_camera(camera_id)
        self.Cam_ID = self.cam.get_id()
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

    def initialize(self):
        with self.cam:
            self.default()
            self.getAcquisitionFrameRateEnable()
            self.getFrameRate()
            self.get_exposure()
            self.get_exposure_mode()
            self.get_exposure_auto()
            self.get_trigger_source()
            self.get_trigger_selector()
            self.get_trigger_mode()
            self.get_trigger_activation()
            self.get_acquisition_mode()
            self.get_pixel_format()
            self.get_pixel_size()
            self.get_roi()
            self.get_temperature()

            self.trigger_modes = self.get_trigger_modes()
            self.trigger_sources = self.get_trigger_sources()
            self.trigger_selectors = self.get_trigger_selectors()
            self.trigger_activations = self.get_trigger_activations()

            self.exposure_modes = self.get_exposure_modes()
            self.exposure_auto_entries = self.get_exposure_auto_entries()

            self.pixel_formats = self.get_pixel_formats()

    def default(self):
        # with self.cam:
        # Restore settings to initial value.
        try:
            self.cam.UserSetSelector.set('Default')

        except (AttributeError, vb.VimbaFeatureError):
            print('Failed to set Feature \'UserSetSelector\'')

        try:
            self.cam.UserSetLoad.run()
            print("--> All feature values have been restored to default")

        except (AttributeError, vb.VimbaFeatureError):
            print('Failed to run Feature \'UserSetLoad\'')

    def get_temperature(self):
        '''Reads out the sensor temperature value

        Returns
        -------
        float
            camera sensor temperature in C.
        '''
        self.temperature = -127
        # with self.cam:
        try:
            self.temperature = self.cam.get_feature_by_name(
                'DeviceTemperature').get()
        except Exception as e:
            pass
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
                print(
                    "Current Exposure ",
                    self.exposure_current,
                    self.exposure_unit)
            return self.exposure_current
        except Exception:
            print("Exposure Get ERROR")
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
            floor_value = self.exposure_increment * np.floor(
                (value - self.exposure_range[0])/self.exposure_increment)\
                + self.exposure_range[0]
            ceil_value = self.exposure_increment * np.ceil(
                (value - self.exposure_range[0])/self.exposure_increment)\
                + self.exposure_range[0]
            set_value = 0
            if np.abs(floor_value - value) < np.abs(ceil_value - value):
                set_value = floor_value
            else:
                set_value = ceil_value
            set_value = max(
                min(set_value, self.exposure_range[1]),
                self.exposure_range[0])
            exposure.set(set_value)
            self.get_exposure()
            self.getFrameRate()
            return 1
        except Exception:
            print("Exposure Set ERROR")
            return 0

    def set_black_level(self, value: float):
        try:
            blackLevel = self.cam.BlackLevel
            blackLevel.set(value)
            return 1
        except Exception:
            print("Black Level Set ERROR")
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
            print("ExposureMode Set ERROR")
            return 0

    def get_exposure_mode(self):
        try:
            # with self.cam:
            ExposureMode = self.cam.ExposureMode
            self.exposure_mode = ExposureMode.get()
            print("Exposure mode ", self.exposure_mode)
            return self.exposure_mode
        except Exception:
            print("ExposureMode get ERROR")
            return "NA"

    def get_exposure_modes(self):
        modes = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            ExposureMode = self.cam.ExposureMode
            entries = ExposureMode.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("ExposureModes Get ERROR")

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
            print("ExposureAuto Set ERROR")
            return 0

    def get_exposure_auto(self):
        try:
            # with self.cam:
            ExposureAuto = self.cam.ExposureAuto
            self.exposure_auto = ExposureAuto.get()
            print("Exposure Auto ", self.exposure_auto)
            return self.exposure_auto
        except Exception:
            print("ExposureAuto get ERROR")
            return "NA"

    def get_exposure_auto_entries(self):
        modes = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            ExposureAuto = self.cam.ExposureAuto
            entries = ExposureAuto.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("ExposureAutos Get ERROR")

        return [modes, displayNames, tooltips]

    def setAcquisitionFrameRateEnable(self, enabled: bool = False):
        try:
            AcquisitionFrameRateEnable = self.cam.AcquisitionFrameRateEnable
            AcquisitionFrameRateEnable.set(enabled)
            self.frameRateEnabled = enabled
        except Exception:
            print("AcquisitionFrameRateEnable set ERROR")

    def getAcquisitionFrameRateEnable(self, output: bool = True):
        try:
            AcquisitionFrameRateEnable = self.cam.AcquisitionFrameRateEnable
            self.frameRateEnabled = AcquisitionFrameRateEnable.get()
            if output:
                print(
                    "AcquisitionFrameRate Enabled ",
                    self.frameRateEnabled)
        except Exception:
            print("AcquisitionFrameRateEnable get ERROR")

    def getFrameRate(self, output: bool = True):
        try:
            frameRate = self.cam.AcquisitionFrameRate
            self.frameRate = frameRate.get()
            self.frameRate_unit = frameRate.get_unit()
            self.frameRate_range = frameRate.get_range()
            if output:
                print(
                    "Current FrameRate ",
                    self.frameRate,
                    self.frameRate_unit)
            return self.frameRate
        except Exception:
            print("AcquisitionFrameRate get ERROR")
            self.frameRate = 0.0
            return self.frameRate

    def setFrameRate(self, value: float):
        if not self.frameRateEnabled:
            return 0

        try:
            frameRate = self.cam.AcquisitionFrameRate
            self.frameRate_unit = frameRate.get_unit()
            self.frameRate_range = frameRate.get_range()

            set_value = max(
                min(value, self.frameRate_range[1]),
                self.frameRate_range[0])
            frameRate.set(set_value)
            self.getFrameRate(False)
            self.get_exposure(False)
            return 1
        except Exception:
            print("AcquisitionFrameRate Set ERROR")
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
            print("trigger Set ERROR")
            return 0

    def get_trigger_mode(self):
        try:
            # with self.cam:
            TriggerMode = self.cam.TriggerMode
            self.trigger_mode = TriggerMode.get()
            print("Trigger mode ", self.trigger_mode)
            return self.trigger_mode
        except Exception:
            print("trigger get ERROR")
            return "NA"

    def get_trigger_modes(self):
        modes = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            TriggerMode = self.cam.TriggerMode
            entries = TriggerMode.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("trigger modes Get ERROR")

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
            print("trigger source Set ERROR")
            return 0

    def get_trigger_source(self):
        try:
            # with self.cam:
            TriggerSource = self.cam.TriggerSource
            self.trigger_source = TriggerSource.get()
            print("Trigger source ", self.trigger_source)
            return self.trigger_source
        except Exception:
            print("trigger source get ERROR")
            return "NA"

    def get_trigger_sources(self):
        sources = []
        displayNames = []
        tooltips = []

        try:
            # with self.cam:
            TriggerSource = self.cam.TriggerSource
            entries = TriggerSource.get_available_entries()
            sources += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("trigger sources Get ERROR")

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
            print("trigger selector Set ERROR")
            return 0

    def get_trigger_selector(self):
        try:
            # with self.cam:
            TriggerSelector = self.cam.TriggerSelector
            self.trigger_selector = TriggerSelector.get()
            print("Trigger selector ", self.trigger_selector)
            return self.trigger_selector
        except Exception:
            print("trigger selector get ERROR")
            return "NA"

    def get_trigger_selectors(self):
        selectors = []
        displayNames = []
        tooltips = []
        try:
            # with self.cam:
            TriggerSelector = self.cam.TriggerSelector
            entries = TriggerSelector.get_available_entries()
            selectors += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("trigger selectors Get ERROR")

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
            print("TriggerActivation Set ERROR")
            return 0

    def get_trigger_activation(self):
        try:
            # with self.cam:
            TriggerActivation = self.cam.TriggerActivation
            self.trigger_activation = TriggerActivation.get()
            print("Trigger activation ", self.trigger_activation)
            return self.trigger_activation
        except Exception:
            print("TriggerActivation get ERROR")
            return "NA"

    def get_trigger_activations(self):
        activations = []
        displayNames = []
        tooltips = []
        try:
            # with self.cam:
            TriggerActivation = self.cam.TriggerActivation
            entries = TriggerActivation.get_available_entries()
            activations += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("TriggerActivation Get ERROR")

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
            print("AcquisitionMode Set ERROR")
            return 0

    def get_acquisition_mode(self):
        try:
            # with self.cam:
            AcquisitionMode = self.cam.AcquisitionMode
            self.acquisition_mode = AcquisitionMode.get()
            print("Acquisition mode ", self.acquisition_mode)
            return self.acquisition_mode
        except Exception:
            print("AcquisitionMode get ERROR")
            return "NA"

    def get_acquisition_modes(self):
        modes = []
        displayNames = []
        tooltips = []
        try:
            # with self.cam:
            AcquisitionMode = self.cam.AcquisitionMode
            entries = AcquisitionMode.get_available_entries()
            modes += map(str, entries)
            displayNames += [entry._EnumEntry__info.displayName
                             for entry in entries]
            tooltips += [entry._EnumEntry__info.tooltip
                         for entry in entries]
        except Exception:
            print("AcquisitionModes Get ERROR")

        return [modes, displayNames, tooltips]

    def get_pixel_formats(self):
        formats = []
        try:
            # with self.cam:
            fmts = self.cam.get_pixel_formats()
            formats += map(str, fmts)
        except Exception:
            print("Pixel Formats Get ERROR")

        return formats

    def get_pixel_format(self):
        formats = []
        try:
            # with self.cam:
            self.pixel_format = self.cam.get_pixel_format()
            print("Pixel Format", self.pixel_format)
            return self.pixel_format
        except Exception:
            print("Pixel Format Get ERROR")
            return 'NA'

    def set_pixel_format(self, value: str):
        try:
            # with self.cam:
            if value in vb.PixelFormat.__members__:
                self.cam.set_pixel_format(vb.PixelFormat[value])
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
                print("Pixel Format Not supported.")
                return 'NA'
            elif '12' in pFormat:
                self.pixel_size = 16
                self.bytes_per_pixel = 2
                return self.pixel_size
            else:
                print("Pixel Format Not supported.")
                return 'NA'
        finally:
            self.set_black_level(5.0)

    def get_pixel_size(self):
        try:
            # with self.cam:
            self.pixel_size = self.cam.PixelSize.get()
            self.bytes_per_pixel = int(np.ceil(int(self.pixel_size)/8))
            print("Pixel Format", self.pixel_size)
            return self.pixel_size
        except Exception:
            print("Pixel Size Get ERROR")
            return 'NA'

    def get_io_lines(self):
        try:
            lines = []
            for line in self.cam.LineSelector.get_available_entries():
                lines.append(str(line))
            return lines
        except Exception:
            print("get_io_lines ERROR")
            return None

    def get_line_modes(self):
        try:
            modes = []
            for mode in self.cam.LineMode.get_available_entries():
                modes.append(str(mode))
            return modes
        except Exception:
            print("get_line_modes ERROR")
            return None

    def get_line_sources(self):
        try:
            sources = []
            for source in self.cam.LineSource.get_available_entries():
                sources.append(str(source))
            return sources
        except Exception:
            print("get_line_sources ERROR")
            return None

    def get_line_source(self):
        try:
            return str(self.cam.LineSource.get())
        except Exception:
            print("get_line_source ERROR")
            return None

    def get_line_mode(self):
        try:
            return str(self.cam.LineMode.get())
        except Exception:
            print("get_line_mode ERROR")
            return None

    def get_line_inverter(self):
        try:
            return self.cam.LineInverter.get()
        except Exception:
            print("get_line_inverter ERROR")
            return None

    def get_line_status(self):
        try:
            return self.cam.LineStatus.get()
        except Exception:
            print("get_line_status ERROR")
            return None

    def set_line_inverter(self, value: bool):
        try:
            self.cam.LineInverter.set(value)
            return True
        except Exception:
            print("set_line_inverter ERROR")
            return False

    def set_line_source(self, value: str):
        try:
            self.cam.LineSource.set(value)
            return True
        except Exception:
            print("set_line_source ERROR")
            return False

    def set_line_mode(self, value: str):
        try:
            self.cam.LineMode.set(value)
            return True
        except Exception:
            print("set_line_mode ERROR")
            return False

    def select_io_line(self, value: str):
        try:
            self.cam.LineSelector.set(value)
            return True
        except Exception:
            print("select_io_line ERROR")
            return False

    def get_timers(self):
        try:
            timers = []
            for timer in self.cam.TimerSelector.get_available_entries():
                timers.append(str(timer))
            return timers
        except Exception:
            print("get_timers ERROR")
            return None

    def get_timer_trigger_activations(self):
        try:
            modes = []
            for mode in \
                    self.cam.TimerTriggerActivation.get_available_entries():
                modes.append(str(mode))
            return modes
        except Exception:
            print("get_timer_trigger_activations ERROR")
            return None

    def get_timer_trigger_sources(self):
        try:
            sources = []
            for source in self.cam.TimerTriggerSource.get_available_entries():
                sources.append(str(source))
            return sources
        except Exception:
            print("get_timer_trigger_sources ERROR")
            return None

    def get_timer_trigger_source(self):
        try:
            return str(self.cam.TimerTriggerSource.get())
        except Exception:
            print("get_timer_trigger_source ERROR")
            return None

    def get_timer_trigger_activation(self):
        try:
            return str(self.cam.TimerTriggerActivation.get())
        except Exception:
            print("get_timer_trigger_activation ERROR")
            return None

    def get_timer_status(self):
        try:
            return self.cam.TimerStatus.get()
        except Exception:
            print("get_timer_status ERROR")
            return None

    def set_timer_trigger_source(self, value: str):
        try:
            self.cam.TimerTriggerSource.set(value)
            return True
        except Exception:
            print("set_timer_trigger_source ERROR")
            return False

    def set_timer_trigger_activation(self, value: str):
        try:
            self.cam.TimerTriggerActivation.set(value)
            return True
        except Exception:
            print("set_line_mode ERROR")
            return False

    def get_timer_duration(self):
        try:
            duration = self.cam.TimerDuration
            return duration.get(), duration.get_range()
        except Exception:
            print("get_timer_duration ERROR")

    def get_timer_delay(self):
        try:
            delay = self.cam.TimerDelay
            return delay.get(), delay.get_range()
        except Exception:
            print("get_timer_delay ERROR")

    def set_timer_duration(self, value: float):
        try:
            self.cam.TimerDuration.set(value)
            return True
        except Exception:
            print("set_timer_duration ERROR")
            return False

    def set_timer_delay(self, value: float):
        try:
            self.cam.TimerDelay.set(value)
            return True
        except Exception:
            print("set_timer_delay ERROR")
            return False

    def select_timer(self, value: str):
        try:
            self.cam.TimerSelector.set(value)
            return True
        except Exception:
            print("select_timer ERROR")
            return False

    def reset_timer(self):
        try:
            self.cam.TimerReset.run()
            return True
        except Exception:
            print("reset_timer ERROR")
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
                    "W H X Y", self.width, self.height,
                    self.offsetX, self.offsetY)
            return (self.width, self.height, self.offsetX, self.offsetY)
        except Exception:
            print("ROI Get ERROR")
            return 'NA'

    def set_roi(self, width, height, x=None, y=None):
        try:
            # with self.cam:
            self.width_range = self.cam.Width.get_range()
            self.width_inc = self.cam.Width.get_increment()
            self.width = self.get_nearest(
                self.width_range, self.width_inc, width)
            self.cam.Width.set(self.width)

            self.height_range = self.cam.Height.get_range()
            self.height_inc = self.cam.Height.get_increment()
            self.height = self.get_nearest(
                self.height_range, self.height_inc, height)
            self.cam.Height.set(self.height)

            if x is None:
                x = (self.width_range[1] - self.width) / 2

            self.offsetX_range = self.cam.OffsetX.get_range()
            self.offsetX_inc = self.cam.OffsetX.get_increment()
            self.offsetX = self.get_nearest(
                self.offsetX_range, self.offsetX_inc, x)
            self.cam.OffsetX.set(self.offsetX)

            if y is None:
                y = (self.height_range[1] - self.height) / 2

            self.offsetY_range = self.cam.OffsetY.get_range()
            self.offsetY_inc = self.cam.OffsetY.get_increment()
            self.offsetY = self.get_nearest(
                self.offsetY_range, self.offsetY_inc, y)
            self.cam.OffsetY.set(self.offsetY)
            return 1
        except Exception:
            print("ROI Set ERROR")
            return 0

    def get_nearest(self, vrange, step, value):
        values = np.arange(vrange[0], vrange[1] + step / 4, step)
        return values[np.abs(values - value).argmin()]


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
