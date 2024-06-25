import logging
import traceback
from enum import Enum

from microEye.analysis.tools.roi_selectors import (
    MultiRectangularROISelector,
    convert_pos_size_to_rois,
    convert_rois_to_pos_size,
)
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.vimba_cam import vimba_cam
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.gui_helper import get_scaling_factor
from microEye.utils.metadata_tree import MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage

try:
    import vimba as vb
except Exception:
    vb = None


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


class Vimba_Panel(Camera_Panel):
    '''
    A Qt Widget for controlling an Allied Vision Camera through Vimba
     | Inherits Camera_Panel
    '''
    PARAMS = VimbaParams

    def __init__(self, cam: vimba_cam, mini=False, *args, **kwargs):
        '''
        Initializes a new Vimba_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : vimba_cam
            Vimba Camera python adapter.

        mini : bool, optional
            Flag indicating if this is a mini camera panel, by default False.

        Other Parameters
        ---------------
        *args
            Arguments to pass to the Camera_Panel constructor.

        **kwargs
            Keyword arguments to pass to the Camera_Panel constructor.
        '''
        super().__init__(cam, mini, *args, **kwargs)

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'Allied Vision')
        self.OME_tab.set_param_value(MetaParams.DET_MODEL, self._cam.cam.get_model())
        self.OME_tab.set_param_value(MetaParams.DET_SERIAL, self._cam.cam.get_serial())
        self.OME_tab.set_param_value(MetaParams.DET_TYPE, 'CMOS')

        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_increment, suffix=self._cam.exposure_unit
        )
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        # exposure mode combobox
        exposure_mode = {
            'name': str(VimbaParams.EXPOSURE_MODE),
            'type': 'list',
            'limits': self._cam.exposure_modes[0],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, exposure_mode)
        # for x in range(len(self._cam.exposure_modes[2])):
        #     self.cam_exposure_mode_cbox.setItemData(
        #         x, self._cam.exposure_modes[2][x], Qt.ToolTipRole)
        self.camera_options.get_param(
            VimbaParams.EXPOSURE_MODE
        ).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.EXPOSURE_MODE)
        )

        # exposure auto mode combobox
        exposure_auto = {
            'name': str(VimbaParams.EXPOSURE_AUTO),
            'type': 'list',
            'limits': self._cam.exposure_auto_entries[0],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, exposure_auto)
        # for x in range(len(self._cam.exposure_auto_entries[2])):
        #     self.cam_exposure_auto_cbox.setItemData(
        #         x, self._cam.exposure_auto_entries[2][x], Qt.ToolTipRole)
        self.camera_options.get_param(
            VimbaParams.EXPOSURE_AUTO
        ).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.EXPOSURE_AUTO)
        )

        # Frame Rate
        framerate_enabled = {
            'name': str(VimbaParams.FRAMERATE_ENABLED),
            'type': 'bool',
            'value': False,
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, framerate_enabled)
        self.camera_options.get_param(
            VimbaParams.FRAMERATE_ENABLED
        ).sigStateChanged.connect(self.cam_framerate_changed)

        framerate = {
            'name': str(VimbaParams.FRAMERATE),
            'type': 'float',
            'value': self._cam.frameRate,
            'dec': False,
            'decimals': 6,
            'step': 0.1,
            'limits': [self._cam.frameRate_range[0], self._cam.frameRate_range[1]],
            'suffix': self._cam.frameRate_unit,
            'enabled': False,
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, framerate)
        self.camera_options.get_param(VimbaParams.FRAMERATE).sigValueChanged.connect(
            self.framerate_spin_changed
        )

        # trigger mode combobox
        trigger_mode = {
            'name': str(VimbaParams.TRIGGER_MODE),
            'type': 'list',
            'limits': self._cam.trigger_modes[0],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, trigger_mode)
        # for x in range(len(self._cam.trigger_modes[2])):
        #     self.cam_trigger_mode_cbox.setItemData(
        #         x, self._cam.trigger_modes[2][x], Qt.ToolTipRole)
        self.camera_options.get_param(VimbaParams.TRIGGER_MODE).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.TRIGGER_MODE)
        )

        # trigger source combobox
        trigger_source = {
            'name': str(VimbaParams.TRIGGER_SOURCE),
            'type': 'list',
            'limits': self._cam.trigger_sources[0],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, trigger_source)
        # for x in range(len(self._cam.trigger_sources[2])):
        #     self.cam_trigger_source_cbox.setItemData(
        #         x, self._cam.trigger_sources[2][x], Qt.ToolTipRole)
        self.camera_options.get_param(
            VimbaParams.TRIGGER_SOURCE
        ).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.TRIGGER_SOURCE)
        )

        # trigger selector combobox
        trigger_selector = {
            'name': str(VimbaParams.TRIGGER_SELECTOR),
            'type': 'list',
            'limits': self._cam.trigger_selectors[0],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, trigger_selector)
        # for x in range(len(self._cam.trigger_selectors[2])):
        #     self.cam_trigger_selector_cbox.setItemData(
        #         x, self._cam.trigger_selectors[2][x], Qt.ToolTipRole)
        self.camera_options.get_param(
            VimbaParams.TRIGGER_SELECTOR
        ).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.TRIGGER_SELECTOR)
        )

        # trigger activation combobox
        trigger_activation = {
            'name': str(VimbaParams.TRIGGER_ACTIVATION),
            'type': 'list',
            'limits': self._cam.trigger_activations[0],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, trigger_activation)
        # for x in range(len(self._cam.trigger_activations[2])):
        #     self.cam_trigger_activation_cbox.setItemData(
        #         x, self._cam.trigger_activations[2][x], Qt.ToolTipRole)
        self.camera_options.get_param(
            VimbaParams.TRIGGER_ACTIVATION
        ).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.TRIGGER_ACTIVATION)
        )

        # pixel formats combobox
        pixel_format = {
            'name': str(VimbaParams.PIXEL_FORMAT),
            'type': 'list',
            'limits': self._cam.pixel_formats,
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, pixel_format)
        self.camera_options.get_param(VimbaParams.PIXEL_FORMAT).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(VimbaParams.PIXEL_FORMAT)
        )

        # start freerun mode button
        freerun = self.get_event_action(VimbaParams.FREERUN)
        self.camera_options.add_param_child(CamParams.ACQUISITION, freerun)
        self.camera_options.get_param(VimbaParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        stop = {'name': str(VimbaParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQUISITION, stop)
        self.camera_options.get_param(VimbaParams.STOP).sigActivated.connect(self.stop)

        # config buttons
        load = {'name': str(VimbaParams.LOAD), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, load)
        self.camera_options.get_param(VimbaParams.LOAD).sigActivated.connect(
            self.load_config
        )

        save = {'name': str(VimbaParams.SAVE), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, save)
        self.camera_options.get_param(VimbaParams.SAVE).sigActivated.connect(
            self.save_config
        )

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.width_range[1]),
            (0, self.cam.height_range[1]),
            (self.cam.width_range[0], self.cam.width_range[1]),
            (self.cam.height_range[0], self.cam.height_range[1]),
        )

        # GPIOs
        lineSelector = {'name': str(VimbaParams.LINE_SELECTOR), 'type': 'list'}
        lineMode = {'name': str(VimbaParams.LINE_MODE), 'type': 'list'}
        lineSource = {'name': str(VimbaParams.LINE_SOURCE), 'type': 'list'}
        lineInverter = {
            'name': str(VimbaParams.LINE_INVERTER),
            'type': 'bool',
            'value': False,
        }

        with self._cam.cam:
            lineSelector['limits'] = self.cam.get_io_lines()
            lineMode['limits'] = self.cam.get_line_modes()
            lineSource['limits'] = self.cam.get_line_sources()

        self.camera_options.add_param_child(CamParams.CAMERA_GPIO, lineSelector)
        self.camera_options.add_param_child(CamParams.CAMERA_GPIO, lineMode)
        self.camera_options.add_param_child(CamParams.CAMERA_GPIO, lineSource)
        self.camera_options.add_param_child(CamParams.CAMERA_GPIO, lineInverter)

        self.camera_options.get_param(
            VimbaParams.LINE_SELECTOR
        ).sigValueChanged.connect(self.io_line_changed)

        set_io_config = {'name': str(VimbaParams.SET_IO_CONFIG), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.CAMERA_GPIO, set_io_config)
        self.camera_options.get_param(VimbaParams.SET_IO_CONFIG).sigActivated.connect(
            self.set_io_line_config
        )

        # Timers
        timerSelector = {'name': str(VimbaParams.TIMER_SELECTOR), 'type': 'list'}
        timerActivation = {'name': str(VimbaParams.TIMER_ACTIVATION), 'type': 'list'}
        timerSource = {'name': str(VimbaParams.TIMER_SOURCE), 'type': 'list'}
        timerDelay = {
            'name': str(VimbaParams.TIMER_DELAY),
            'type': 'float',
            'dec': False,
            'decimals': 6,
        }
        timerDuration = {
            'name': str(VimbaParams.TIMER_DURATION),
            'type': 'float',
            'dec': False,
            'decimals': 6,
        }
        timerReset = {'name': str(VimbaParams.TIMER_RESET), 'type': 'action'}
        set_timer_config = {'name': str(VimbaParams.SET_TIMER_CONFIG), 'type': 'action'}

        with self._cam.cam:
            timers = self.cam.get_timers()
            if timers:
                timerSelector['limits'] = timers
                timerActivation['limits'] = self.cam.get_timer_trigger_activations()
                timerSource['limits'] = self.cam.get_timer_trigger_sources()

        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, timerSelector)
        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, timerActivation)
        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, timerSource)
        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, timerDelay)
        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, timerDuration)
        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, timerReset)
        self.camera_options.add_param_child(CamParams.CAMERA_TIMERS, set_timer_config)

        def reset_timer():
            with self._cam.cam:
                self.cam.reset_timer()

        def update_timer():
            with self._cam.cam:
                timer = self.camera_options.get_param_value(VimbaParams.TIMER_SELECTOR)
                self.cam.select_timer(timer)
                delay = self.cam.get_timer_delay()
                duration = self.cam.get_timer_duration()

                delay_param = self.camera_options.get_param(VimbaParams.TIMER_DELAY)
                delay_param.setLimits((delay[1][0], delay[1][1]))
                delay_param.setValue(delay[0])

                duration_param = self.camera_options.get_param(
                    VimbaParams.TIMER_DURATION
                )
                duration_param.setLimits((duration[1][0], duration[1][1]))
                duration_param.setValue(duration[0])

                self.camera_options.set_param_value(
                    VimbaParams.TIMER_ACTIVATION,
                    self.cam.get_timer_trigger_activation(),
                )
                self.camera_options.set_param_value(
                    VimbaParams.TIMER_SOURCE, self.cam.get_timer_trigger_source()
                )

        def set_timer():
            with self._cam.cam:
                timer = self.camera_options.get_param_value(VimbaParams.TIMER_SELECTOR)
                self.cam.select_timer(timer)
                act = self.camera_options.get_param_value(VimbaParams.TIMER_ACTIVATION)
                source = self.camera_options.get_param_value(VimbaParams.TIMER_SOURCE)
                self.cam.set_timer_trigger_activation(act)
                self.cam.set_timer_trigger_source(source)
                self.cam.set_timer_duration(
                    self.camera_options.get_param_value(VimbaParams.TIMER_DURATION)
                )
                self.cam.set_timer_delay(
                    self.camera_options.get_param_value(VimbaParams.TIMER_DELAY)
                )

        self.camera_options.get_param(
            VimbaParams.TIMER_SELECTOR
        ).sigValueChanged.connect(update_timer)
        self.camera_options.get_param(VimbaParams.TIMER_RESET).sigActivated.connect(
            reset_timer
        )
        self.camera_options.get_param(
            VimbaParams.SET_TIMER_CONFIG
        ).sigActivated.connect(set_timer)

    @property
    def cam(self):
        '''The vimba_cam property.

        Returns
        -------
        vimba_cam
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: vimba_cam):
        '''The vimba_cam property.

        Parameters
        ----------
        cam : vimba_cam
            the vimba_cam to set as panel camera.
        '''
        self._cam = cam

    def setExposure(self, value):
        '''Sets the exposure time widget of camera

        Parameters
        ----------
        value : float
            selected exposure time
        '''
        super().setExposure(value * 1e3)

    def set_ROI(self):
        '''Sets the ROI for the slected vimba_cam'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(*self.camera_options.get_roi_info(True))

            self.cam.get_roi()

        self.camera_options.set_roi_info(
            int(self.cam.offsetX),
            int(self.cam.offsetY),
            int(self.cam.width),
            int(self.cam.height),
        )

        self.refresh_framerate()

    def reset_ROI(self):
        '''Resets the ROI for the slected IDS_Camera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(self.cam.width_max, self.cam.height_max)

            self.cam.get_roi()

        self.camera_options.set_roi_info(
            0, 0, int(self.cam.width), int(self.cam.height)
        )

        self.refresh_framerate()

    def center_ROI(self):
        '''Sets the ROI for the slected vimba_cam'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(*self.camera_options.get_roi_info(True)[:2])

        self.camera_options.set_roi_info(
            self.cam.offsetX, self.cam.offsetY, self.cam.width, self.cam.height
        )

        self.refresh_framerate()

    def select_ROI(self):
        if self.acq_job is not None:
            try:

                def work_func(**kwargs):
                    try:
                        image = uImage(self.acq_job.frame.image)

                        image.equalizeLUT()

                        scale_factor = get_scaling_factor(image.height, image.width)

                        selector = MultiRectangularROISelector.get_selector(
                            image._view, scale_factor, max_rois=1
                        )
                        # if old_rois:
                        #     selector.rois = old_rois

                        rois = selector.select_rectangular_rois()

                        rois = convert_rois_to_pos_size(rois)

                        if len(rois) > 0:
                            return rois[0]
                        else:
                            return None
                    except Exception:
                        traceback.print_exc()
                        return None

                def done(result: list):
                    if result is not None:
                        # x, y, w, h = result
                        self.camera_options.set_roi_info(*result)

                self.worker = QThreadWorker(work_func)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    def select_ROIs(self):
        if self.acq_job is not None:
            try:

                def work_func(**kwargs):
                    try:
                        image = uImage(self.acq_job.frame.image)

                        image.equalizeLUT()

                        scale_factor = get_scaling_factor(image.height, image.width)

                        selector = MultiRectangularROISelector.get_selector(
                            image._view, scale_factor, max_rois=4, one_size=True
                        )

                        old_rois = self.camera_options.get_export_rois()

                        if len(old_rois) > 0:
                            old_rois = convert_pos_size_to_rois(old_rois)
                            selector.rois = old_rois

                        rois = selector.select_rectangular_rois()

                        rois = convert_rois_to_pos_size(rois)

                        if len(rois) > 0:
                            return rois
                        else:
                            return None
                    except Exception:
                        traceback.print_exc()
                        return None

                def done(results: list[list]):
                    if results is not None:
                        rois_param = self.camera_options.get_param(
                            CamParams.EXPORTED_ROIS
                        )
                        rois_param.clearChildren()
                        for x, y, w, h in results:
                            self.camera_options.add_param_child(
                                CamParams.EXPORTED_ROIS,
                                {
                                    'name': 'ROI 1',
                                    'type': 'str',
                                    'readonly': True,
                                    'removable': True,
                                    'value': f'{x}, {y}, {w}, {h}',
                                },
                            )

                self.worker = QThreadWorker(work_func)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    def cam_cbox_changed(self, param: VimbaParams):
        '''
        Slot for changed combobox values

        Parameters
        ----------
        param : VimbaParams
            selected parameter enum
        '''
        value = self.camera_options.get_param_value(param)
        with self._cam.cam:
            if param == VimbaParams.TRIGGER_MODE:
                self._cam.set_trigger_mode(value)
                self._cam.get_trigger_mode()
            elif param == VimbaParams.TRIGGER_SOURCE:
                self._cam.set_trigger_source(value)
                self._cam.get_trigger_source()
            elif param == VimbaParams.TRIGGER_SELECTOR:
                self._cam.set_trigger_selector(value)
                self._cam.get_trigger_selector()
            elif param == VimbaParams.TRIGGER_ACTIVATION:
                self._cam.set_trigger_activation(value)
                self._cam.get_trigger_activation()
            elif param == VimbaParams.EXPOSURE_MODE:
                self._cam.set_exposure_mode(value)
                self._cam.get_exposure_mode()
            elif param == VimbaParams.EXPOSURE_AUTO:
                self._cam.set_exposure_auto(value)
                self._cam.get_exposure_auto()
            elif param == VimbaParams.PIXEL_FORMAT:
                self._cam.set_pixel_format(value)
                self.refresh_framerate()

    def exposure_spin_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in micro-seconds
        '''
        with self._cam.cam:
            self._cam.set_exposure(value)

        self.refresh_exposure()
        self.refresh_framerate()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, float(self._cam.exposure_current / 1000)
        )
        if self.master:
            self.exposureChanged.emit()

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_spin_changed
        )

    def refresh_framerate(self, value=None):
        with self._cam.cam:
            if value:
                self._cam.setFrameRate(value)
            self._cam.getFrameRate(False)

        framerate = self.camera_options.get_param(VimbaParams.FRAMERATE)
        framerate.setLimits(
            (self._cam.frameRate_range[0], self._cam.frameRate_range[1])
        )
        framerate.setValue(self._cam.frameRate)

    def cam_framerate_changed(self, value):
        with self._cam.cam:
            self.cam.setAcquisitionFrameRateEnable(
                self.camera_options.get_param_value(VimbaParams.FRAMERATE_ENABLED)
            )

        self.camera_options.get_param(VimbaParams.FRAMERATE).setOpts(
            enabled=self.camera_options.get_param_value(VimbaParams.FRAMERATE_ENABLED)
        )
        self.refresh_framerate()

    def framerate_spin_changed(self, param, value: float):
        self.refresh_framerate(value)

    def io_line_changed(self, value):
        with self._cam.cam:
            line = self.camera_options.get_param_value(VimbaParams.LINE_SELECTOR)
            self.cam.select_io_line(line)
            mode = self.cam.get_line_mode()
            self.camera_options.set_param_value(VimbaParams.LINE_MODE, mode)
            if 'Out' in mode:
                self.camera_options.set_param_value(
                    VimbaParams.LINE_SOURCE, self.cam.get_line_source()
                )
                self.camera_options.set_param_value(
                    VimbaParams.LINE_INVERTER, self.cam.get_line_inverter()
                )

    def set_io_line_config(self):
        with self._cam.cam:
            line = self.camera_options.get_param_value(VimbaParams.LINE_SELECTOR)
            self.cam.select_io_line(line)
            mode = self.camera_options.get_param_value(VimbaParams.LINE_MODE)
            self.cam.set_line_mode(mode)
            if 'Out' in mode:
                self.cam.set_line_source(
                    self.camera_options.get_param_value(VimbaParams.LINE_SOURCE)
                )
                self.cam.set_line_inverter(
                    self.camera_options.get_param_value(VimbaParams.LINE_INVERTER)
                )

    def _capture_handler(self, cam, frame):
        if self.acq_job.frames_captured < self.acq_job.frames or self.mini:
            self._buffer.put(frame.as_numpy_ndarray()[..., 0])
            cam.queue_frame(frame)
            # add sensor temperature to the stack
            self._temps.put(self.cam.get_temperature())
            self.acq_job.frames_captured = self.acq_job.frames_captured + 1
        if self.acq_job.frames_captured > self.acq_job.frames - 1 and not self.mini:
            self.acq_job.c_event.set()
            self.acq_job.stop_threads = True
            logging.debug('Stop')
            print('Capture Stopped!')

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : vimba_cam
            the vimba_cam used to acquire frames.
        '''
        try:
            # Continuous image capture
            with self._cam.cam:
                self._cam.cam.start_streaming(self._capture_handler)

                self.acq_job.c_event.wait()
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            with self._cam.cam:
                self._cam.cam.stop_streaming()
            self._cam.acquisition = False
            QtCore.QThreadPool.globalInstance().releaseThread()
        return QDateTime.currentDateTime()

    def getCaptureArgs(self) -> list:
        '''User specific arguments to be passed to the parallelized
        Camera_Panel.cam_capture function.

        Example
        ------
        check child panels (vimba, ueye or thorlabs)

        Returns
        -------
        list
            list of args to be passed in order to
            Camera_Panel.cam_capture function.

        Raises
        ------
        NotImplementedError
            Has to be implemented by the use in child class.
        '''
        args = []
        return args

    def save_config(self):
        filename, _ = getSaveFileName(
            self, 'Save config', filter='XML Files (*.xml);;'
        )

        if len(filename) > 0:
            with self.cam.cam:
                self.cam.cam.save_settings(filename, vb.PersistType.All)

            QtWidgets.QMessageBox.information(self, 'Info', 'Config saved.')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Config not saved.')

    def load_config(self):
        filename, _ = getOpenFileName(
            self, 'Load config', filter='XML Files (*.xml);;'
        )

        if len(filename) > 0:
            with self.cam.cam:
                self.cam.default()

                # Load camera settings from file.
                self.cam.cam.load_settings(filename, vb.PersistType.All)

                w, h, x, y = self.cam.get_roi()

                self.camera_options.set_roi_info(int(x), int(y), int(w), int(h))

                self.cam_trigger_mode_cbox.setCurrentText(self.cam.get_trigger_mode())

                self.camera_options.set_param_value(
                    CamParams.EXPOSURE, self.cam.get_exposure()
                )
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No file selected.')
