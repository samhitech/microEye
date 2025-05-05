import json
import traceback
from enum import Enum

from microEye.analysis.tools.roi_selectors import (
    MultiRectangularROISelector,
    convert_pos_size_to_rois,
    convert_rois_to_pos_size,
)
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.thorlabs.thorlabs import (
    CMD,
    FLASH_MODE,
    IS_DONT_WAIT,
    TRIGGER,
    thorlabs_camera,
)
from microEye.qt import (
    QDateTime,
    QtCore,
    QtWidgets,
    Slot,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.gui_helper import get_scaling_factor
from microEye.utils.metadata_tree import MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage


class ThorCamParams(Enum):
    FREERUN = 'Acquisition.Start (Freerun)'
    TRIGGERED = 'Acquisition.Start (Triggered)'
    STOP = 'Acquisition.Stop'
    PIXEL_CLOCK = 'Acquisition Settings.Pixel Clock (MHz)'
    FRAMERATE = 'Acquisition Settings.Framerate Slider'
    FRAME_AVERAGING = 'Acquisition Settings.Frame Averaging'
    TRIGGER_MODE = 'Acquisition Settings.Trigger Mode'
    FLASH_MODE = 'Acquisition Settings.Flash Mode'
    FLASH_DURATION = 'Acquisition Settings.Flash Duration Slider'
    FLASH_DELAY = 'Acquisition Settings.Flash Delay Slider'
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


class Thorlabs_Panel(Camera_Panel):
    '''
    A Qt Widget for controlling a Thorlabs Camera | Inherits Camera_Panel
    '''
    PARAMS = ThorCamParams

    def __init__(self, cam: thorlabs_camera, mini: bool = False, *args, **kwargs):
        '''
        Initializes a new Thorlabs_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : thorlabs_camera
            Thorlabs Camera python adapter.

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

        # flag true to close camera adapter and dispose it
        self._dispose_cam = False

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'Thorlabs')
        self.OME_tab.set_param_value(
            MetaParams.DET_MODEL, self._cam.sInfo.strSensorName.decode('utf-8')
        )
        self.OME_tab.set_param_value(
            MetaParams.DET_SERIAL, self._cam.cInfo.SerNo.decode('utf-8')
        )
        self.OME_tab.set_param_value(MetaParams.DET_TYPE, 'CMOS')

        # pixel clock label and combobox
        pixel_clock = {
            'name': str(ThorCamParams.PIXEL_CLOCK),
            'type': 'list',
            'limits': list(map(str, self._cam.pixel_clock_list[:])),
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, pixel_clock)
        self.camera_options.get_param(
            ThorCamParams.PIXEL_CLOCK
        ).sigValueChanged.connect(
            lambda: self.cam_pixel_cbox_changed(ThorCamParams.PIXEL_CLOCK)
        )

        # framerate slider control
        framerate = {
            'name': str(ThorCamParams.FRAMERATE),
            'type': 'float',
            'value': int(self._cam.current_framerate.value),
            'dec': False,
            'decimals': 6,
            'step': self._cam.increment_framerate.value,
            'limits': [self._cam.min_framerate.value, self._cam.max_framerate.value],
            'suffix': 'Hz',
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, framerate)
        self.camera_options.get_param(ThorCamParams.FRAMERATE).sigValueChanged.connect(
            self.cam_framerate_value_changed
        )

        # exposure text box
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(step=self._cam.exposure_range[2], suffix='ms')
        exposure.setValue(self._cam.exposure_current.value)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        # trigger mode combobox
        trigger_mode = {
            'name': str(ThorCamParams.TRIGGER_MODE),
            'type': 'list',
            'limits': list(TRIGGER.TRIGGER_MODES.keys()),
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, trigger_mode)
        self.camera_options.get_param(
            ThorCamParams.TRIGGER_MODE
        ).sigValueChanged.connect(
            lambda: self.cam_trigger_cbox_changed(ThorCamParams.TRIGGER_MODE)
        )

        # flash mode combobox
        flash_mode = {
            'name': str(ThorCamParams.FLASH_MODE),
            'type': 'list',
            'limits': list(FLASH_MODE.FLASH_MODES.keys()),
            'enabled': not self.mini,
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, flash_mode)
        self.camera_options.get_param(ThorCamParams.FLASH_MODE).sigValueChanged.connect(
            lambda: self.cam_flash_cbox_changed(ThorCamParams.FLASH_MODE)
        )

        # flash duration slider
        falsh_duration = {
            'name': str(ThorCamParams.FLASH_DURATION),
            'type': 'int',
            'value': 0,
            'dec': False,
            'decimals': 6,
            'suffix': 'us',
            'enabled': not self.mini,
            'step': self._cam.flash_inc.u32Duration,
            'limits': [0, self._cam.flash_max.u32Duration],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, falsh_duration)
        self.camera_options.get_param(
            ThorCamParams.FLASH_DURATION
        ).sigValueChanged.connect(self.cam_flash_duration_value_changed)

        # flash delay
        falsh_delay = {
            'name': str(ThorCamParams.FLASH_DELAY),
            'type': 'int',
            'value': 0,
            'dec': False,
            'decimals': 6,
            'suffix': 'us',
            'enabled': not self.mini,
            'step': self._cam.flash_inc.s32Delay,
            'limits': [0, self._cam.flash_max.s32Delay],
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, falsh_delay)
        self.camera_options.get_param(
            ThorCamParams.FLASH_DELAY
        ).sigValueChanged.connect(self.cam_flash_delay_value_changed)

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[-1])
        )

        # start freerun mode
        freerun = self.get_event_action(ThorCamParams.FREERUN)
        self.camera_options.add_param_child(CamParams.ACQUISITION, freerun)
        self.camera_options.get_param(ThorCamParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # start trigger mode button
        triggered = self.get_event_action(ThorCamParams.TRIGGERED)
        self.camera_options.add_param_child(CamParams.ACQUISITION, triggered)
        self.camera_options.get_param(ThorCamParams.TRIGGERED).sigActivated.connect(
            self.start_software_triggered
        )

        # stop acquisition
        stop = {'name': str(ThorCamParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQUISITION, stop)
        self.camera_options.get_param(ThorCamParams.STOP).sigActivated.connect(
            lambda: self.stop()
        )

        # config buttons
        load = {'name': str(ThorCamParams.LOAD), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, load)
        self.camera_options.get_param(ThorCamParams.LOAD).sigActivated.connect(
            lambda: self.load_config()
        )

        save = {'name': str(ThorCamParams.SAVE), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, save)
        self.camera_options.get_param(ThorCamParams.SAVE).sigActivated.connect(
            lambda: self.save_config()
        )

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.rectROI.s32Width),
            (0, self.cam.rectROI.s32Height),
            (32, self.cam.rectROI.s32Width),
            (32, self.cam.rectROI.s32Height),
        )

    @property
    def cam(self):
        '''The thorlabs_camera property.

        Returns
        -------
        thorlabs_camera
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: thorlabs_camera):
        '''The thorlabs_camera property.

        Parameters
        ----------
        cam : thorlabs_camera
            the thorlabs_camera to set as panel camera.
        '''
        self._cam = cam

    def set_ROI(self):
        '''Sets the ROI for the slected thorlabs_camera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.set_roi(*self.camera_options.get_roi_info())

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[0])
        )
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[-1])
        )

        self.camera_options.set_roi_info(
            self.cam.set_rectROI.s32X,
            self.cam.set_rectROI.s32Y,
            self.cam.set_rectROI.s32Width,
            self.cam.set_rectROI.s32Height,
        )

    def reset_ROI(self):
        '''Resets the ROI for the slected thorlabs_camera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.reset_roi()
        self.camera_options.set_roi_info(0, 0, self.cam.width, self.cam.height)

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[0])
        )
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[-1])
        )

    def center_ROI(self):
        '''Calculates the x, y values for a centered ROI'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot center ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        _, _, w, h = self.camera_options.get_roi_info()
        x = (self.cam.rectROI.s32Width - w) // 2
        y = (self.cam.rectROI.s32Height - h) // 2

        self.camera_options.set_roi_info(x, y, w, h)

        self.set_ROI()

    def select_ROI(self):
        '''
        Opens a dialog to select a ROI from the last image.
        '''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        if self.acq_job.frame is not None:
            try:

                def work_func(**kwargs):
                    try:
                        image = uImage(self.acq_job.frame.image)

                        image.equalizeLUT(nLUT=True)

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

    @Slot(str)
    def cam_trigger_cbox_changed(self, param: ThorCamParams):
        '''
        Slot for changed trigger mode

        Parameters
        ----------
        param : ThorCamParams
            parameter path to index of selected trigger mode from _TRIGGER.TRIGGER_MODES
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_trigger_mode(TRIGGER.TRIGGER_MODES[value])
        self._cam.get_trigger_mode()

    @Slot(str)
    def cam_flash_cbox_changed(self, param: ThorCamParams):
        '''
        Slot for changed flash mode

        Parameters
        ----------
        param : ThorCamParams
            parameter path to index of selected flash mode from _TRIGGER.FLASH_MODES
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_flash_mode(FLASH_MODE.FLASH_MODES[value])
        self._cam.get_flash_mode(output=True)

    @Slot(str)
    def cam_pixel_cbox_changed(self, param: ThorCamParams):
        '''
        Slot for changed pixel clock

        Parameters
        ----------
        param : ThorCamParams
            parameter path to selected pixel clock in MHz
            (note that only certain values are allowed by camera)
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_pixel_clock(int(value))
        self._cam.get_pixel_clock_info(False)

        self.refresh_framerate(True)

    def cam_framerate_value_changed(self, param, value):
        '''
        Slot for changed framerate

        Parameters
        ----------
        param : Parameter
            frames per second parameter
        Value : double
            selected frames per second
        '''
        self._cam.set_framerate(value)

        self.camera_options.set_param_value(
            ThorCamParams.FRAMERATE,
            self._cam.current_framerate.value,
            self.cam_framerate_value_changed,
        )

        self.refresh_exposure()

    def exposure_spin_changed(self, param, value):
        '''
        Slot for changed exposure

        Parameters
        ----------
        param : Parameter
            exposure parameter
        Value : double
            selected exposure in milliseconds
        '''
        self._cam.set_exposure(value)
        self._cam.get_exposure(False)

        self.camera_options.set_param_value(
            CamParams.EXPOSURE,
            self._cam.exposure_current.value,
            self.exposure_spin_changed,
        )

        self.refresh_flash()

        self.OME_tab.set_param_value(MetaParams.EXPOSURE, self._cam.exposure_current)
        if self.master:
            self.exposureChanged.emit()

    def refresh_framerate(self, range=False):
        self._cam.get_framerate_range(False)

        framerate = self.camera_options.get_param(ThorCamParams.FRAMERATE)
        framerate.setLimits(
            (self._cam.min_framerate.value, self._cam.max_framerate.value)
        )
        framerate.setOpts(step=self._cam.increment_framerate.value)

        self.cam_framerate_value_changed(framerate, self._cam.max_framerate.value)

    def refresh_exposure(self, range=False):
        self._cam.get_exposure_range(False)

        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(step=self._cam.exposure_range[2])
        exposure.setValue(self.cam.exposure_range[1])

    def refresh_flash(self):
        self._cam.get_flash_range(False)

        duration = self.camera_options.get_param(ThorCamParams.FLASH_DURATION)
        duration.setLimits((0, self._cam.flash_max.u32Duration))
        duration.setOpts(step=self._cam.flash_inc.u32Duration)
        duration.setValue(0)

        delay = self.camera_options.get_param(ThorCamParams.FLASH_DURATION)
        delay.setLimits((0, self._cam.flash_max.s32Delay))
        delay.setOpts(step=self._cam.flash_inc.s32Delay)
        delay.setValue(0)

    def cam_flash_duration_value_changed(self, param, value):
        '''
        Slot for changed flash duration

        Parameters
        ----------
        param : Parameter
            flash duration parameter
        Value : double
            selected flash duration in micro-seconds
        '''
        self._cam.set_flash_params(self._cam.flash_cur.s32Delay, value)
        self._cam.get_flash_params()

        self.camera_options.set_param_value(
            ThorCamParams.FLASH_DURATION,
            self._cam.flash_cur.u32Duration,
            self.cam_flash_duration_value_changed,
        )

    def cam_flash_delay_value_changed(self, param, value):
        '''
        Slot for changed flash delay

        Parameters
        ----------
        param : Parameter
            flash delay parameter
        Value : double
            selected flash delay in micro-seconds
        '''
        self._cam.set_flash_params(value, self._cam.flash_cur.u32Duration)

        self.camera_options.set_param_value(
            ThorCamParams.FLASH_DELAY,
            self._cam.flash_cur.s32Delay,
            self.cam_flash_delay_value_changed,
        )

    def start_free_run(self, Prefix=''):
        '''
        Starts free run acquisition mode

        Parameters
        ----------
        cam : IDS_Camera
            IDS Camera python adapter
        '''
        nRet = 0
        if self._cam.acquisition:
            return  # if acquisition is already going on

        if not self._cam.memory_allocated:
            self._cam.allocate_memory_buffer()  # allocate memory

        if not self._cam.capture_video:
            self._cam.start_live_capture()  # start live capture (freerun mode)

        self._cam.refresh_info()  # refresh adapter info

        self._save_prefix = Prefix
        self.acq_job = self.getAcquisitionJob()

        self._cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers()

    def start_software_triggered(self, Prefix=''):
        '''
        Starts trigger acquisition mode

        Parameters
        ----------
        cam : IDS_Camera
            IDS Camera python adapter
        '''

        nRet = 0
        if self._cam.acquisition:
            return  # if acquisition is already going on

        if not self._cam.memory_allocated:
            self._cam.allocate_memory_buffer()  # allocate memory

        self._cam.refresh_info()  # refresh adapter info

        self._save_prefix = Prefix
        self.acq_job = self.getAcquisitionJob()

        self._cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers()

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        nRet : int
            return code from thorlabs_camera, ueye.IS_SUCCESS = 0 to run.
        cam : thorlabs_camera
            the thorlabs_camera used to acquire frames.
        '''
        try:
            nRet = 0
            time = QDateTime.currentDateTime()
            # Continuous image capture
            while nRet == 0:
                self.acq_job.capture_time = time.msecsTo(QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                if not self._cam.capture_video:
                    self._cam.uc480.is_FreezeVideo(self._cam.hCam, IS_DONT_WAIT)

                nRet = self._cam.wait_for_next_image(500, not self.mini)
                if nRet == CMD.IS_SUCCESS:
                    self._cam.get_pitch()
                    data = self._cam.get_data()

                    # if not np.array_equal(temp, data):
                    self._buffer.put(data.copy())
                    # add sensor temperature to the stack
                    self._temps.put(self._cam.get_temperature())
                    self.acq_job.frames_captured = self.acq_job.frames_captured + 1
                    if (
                        self.acq_job.frames_captured >= self.acq_job.frames
                        and not self.mini
                    ):
                        self._stop_thread = True
                    self._cam.unlock_buffer()

                QtCore.QThread.usleep(100)  # sleep 100us

                if self.acq_job.stop_threads:
                    break  # in case stop threads is initiated
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            self._cam.acquisition = False
            if self._cam.capture_video:
                self._cam.stop_live_capture()
            self._cam.free_memory()
            if self._dispose_cam:
                self._cam.dispose()

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
            self, 'Save config', filter='JSON Files (*.json);;'
        )

        if len(filename) > 0:
            config = {
                'model': self.cam.sInfo.strSensorName.decode('utf-8'),
                'serial': self.cam.cInfo.SerNo.decode('utf-8'),
                'clock speed': self.cam.pixel_clock.value,
                'trigger': self.camera_options.get_param_value(
                    ThorCamParams.TRIGGER_MODE
                ),
                'flash mode': self.camera_options.get_param_value(
                    ThorCamParams.FLASH_MODE
                ),
                'framerate': self.cam.current_framerate.value,
                'exposure': self.cam.exposure_current.value,
                'flash duration': self.cam.flash_cur.u32Duration,
                'flash delay': self.cam.flash_cur.s32Delay,
                'ROI w': self.cam.set_rectROI.s32Width,
                'ROI h': self.cam.set_rectROI.s32Height,
                'ROI x': self.cam.set_rectROI.s32X,
                'ROI y': self.cam.set_rectROI.s32Y,
                'Zoom': self.camera_options.get_param_value(CamParams.RESIZE_DISPLAY),
            }

            with open(filename, 'w') as file:
                json.dump(config, file)

            QtWidgets.QMessageBox.information(self, 'Info', 'Config saved.')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Config not saved.')

    def load_config(self):
        filename, _ = getOpenFileName(
            self, 'Load config', filter='JSON Files (*.json);;'
        )

        if len(filename) > 0:
            config: dict = None
            keys = [
                'model',
                'serial',
                'clock speed',
                'trigger',
                'flash mode',
                'framerate',
                'exposure',
                'flash duration',
                'flash delay',
                'ROI w',
                'ROI h',
                'ROI x',
                'ROI y',
                'Zoom',
            ]
            with open(filename) as file:
                config = json.load(file)
            if all(key in config for key in keys):
                if self.cam.sInfo.strSensorName.decode('utf-8') == config['model']:
                    self.camera_options.set_roi_info(
                        int(config['ROI x']),
                        int(config['ROI y']),
                        int(config['ROI w']),
                        int(config['ROI h']),
                    )
                    self.set_ROI()

                    self.camera_options.set_param_value(
                        ThorCamParams.PIXEL_CLOCK, str(config['clock speed'])
                    )
                    self.camera_options.set_param_value(
                        ThorCamParams.TRIGGER_MODE, config['trigger']
                    )
                    self.camera_options.set_param_value(
                        ThorCamParams.FLASH_DURATION, config['flash mode']
                    )

                    self.camera_options.set_param_value(
                        ThorCamParams.FRAMERATE, config['framerate']
                    )

                    self.camera_options.set_param_value(
                        CamParams.EXPOSURE, config['exposure']
                    )

                    self.camera_options.set_param_value(
                        ThorCamParams.FLASH_DURATION, config['flash duration']
                    )

                    self.camera_options.set_param_value(
                        ThorCamParams.FLASH_DELAY, config['flash delay']
                    )

                    self.camera_options.set_param_value(
                        CamParams.RESIZE_DISPLAY, float(config['Zoom'])
                    )
                else:
                    QtWidgets.QMessageBox.warning(
                        self, 'Warning', 'Camera model is different.'
                    )
            else:
                QtWidgets.QMessageBox.warning(
                    self, 'Warning', 'Wrong or corrupted config file.'
                )
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'No file selected.')
