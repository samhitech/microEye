import json
import logging
import traceback

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.ueye.ueye_camera import IDS_Camera, uEyeParams
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.metadata_tree import MetaParams

try:
    from pyueye import ueye
except Exception:
    ueye = None


class IDS_Panel(Camera_Panel):
    '''
    A Qt Widget for controlling an IDS Camera | Inherits Camera_Panel
    '''
    PARAMS = uEyeParams

    def __init__(self, cam: IDS_Camera, mini=False, *args, **kwargs):
        '''
        Initializes a new IDS_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : IDS_Camera
            IDS Camera python adapter.

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


    def _init_camera_specific(self):
        # flag true to close camera adapter and dispose it
        self._dispose_cam = False

        for prop in self.cam.property_tree():
            self.camera_options.add_param_child(
                prop.pop('parent', CamParams.ACQ_SETTINGS), prop)

        # pixel clock label and combobox
        self.camera_options.get_param(uEyeParams.PIXEL_CLOCK).sigValueChanged.connect(
            lambda: self.cam_pixel_cbox_changed(uEyeParams.PIXEL_CLOCK)
        )

        # framerate slider control
        self.camera_options.get_param(uEyeParams.FRAMERATE).sigValueChanged.connect(
            self.cam_framerate_value_changed
        )

        # exposure text box
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (self._cam.exposure_range[0].value, self._cam.exposure_range[1].value)
        )
        exposure.setOpts(step=self._cam.exposure_range[2].value, suffix='ms')
        exposure.setValue(self._cam.exposure_current.value)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        # trigger mode combobox
        self.camera_options.get_param(uEyeParams.TRIGGER_MODE).sigValueChanged.connect(
            lambda: self.cam_trigger_cbox_changed(uEyeParams.TRIGGER_MODE)
        )

        # flash mode combobox
        flash_mode = self.camera_options.get_param(uEyeParams.FLASH_MODE)
        flash_mode.setOpts(enabled=not self.mini)
        flash_mode.sigValueChanged.connect(
            lambda: self.cam_flash_cbox_changed(uEyeParams.FLASH_MODE)
        )

        # flash duration slider
        falsh_duration = self.camera_options.get_param(
            uEyeParams.FLASH_DURATION
        )
        falsh_duration.setOpts(enabled=not self.mini)
        falsh_duration.sigValueChanged.connect(self.cam_flash_duration_value_changed)

        # flash delay
        flash_delay = self.camera_options.get_param(uEyeParams.FLASH_DELAY)
        flash_delay.setOpts(enabled=not self.mini)
        flash_delay.sigValueChanged.connect(
            self.cam_flash_delay_value_changed
        )

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            uEyeParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[-1].value)
        )

        # start freerun mode
        self.camera_options.get_param(uEyeParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # start trigger mode button
        self.camera_options.get_param(uEyeParams.TRIGGERED).sigActivated.connect(
            self.start_software_triggered
        )

        # stop acquisition
        self.camera_options.get_param(uEyeParams.STOP).sigActivated.connect(
            lambda: self.stop()
        )

        # config buttons
        self.camera_options.get_param(uEyeParams.LOAD).sigActivated.connect(
            lambda: self.load_config()
        )

        self.camera_options.get_param(uEyeParams.SAVE).sigActivated.connect(
            lambda: self.save_config()
        )

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.rectROI.s32Width.value),
            (0, self.cam.rectROI.s32Height.value),
            (32, self.cam.rectROI.s32Width.value),
            (32, self.cam.rectROI.s32Height.value),
        )

    @property
    def cam(self):
        '''The IDS_Camera property.

        Returns
        -------
        IDS_Camera
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: IDS_Camera):
        '''The IDS_Camera property.

        Parameters
        ----------
        cam : IDS_Camera
            the IDS_Camera to set as panel camera.
        '''
        self._cam = cam

    def set_ROI(self):
        '''Sets the ROI for the slected IDS_Camera'''
        super().set_ROI()

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            uEyeParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[0].value)
        )
        self.camera_options.set_param_value(
            uEyeParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[-1].value)
        )

    def reset_ROI(self):
        '''Resets the ROI for the slected IDS_Camera'''
        super().reset_ROI()

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            uEyeParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[0].value)
        )
        self.camera_options.set_param_value(
            uEyeParams.PIXEL_CLOCK, str(self._cam.pixel_clock_list[-1].value)
        )

    def cam_trigger_cbox_changed(self, param: uEyeParams):
        '''
        Slot for changed trigger mode

        Parameters
        ----------
        param : uEyeParams
            parameter path to index of selected trigger mode
            from IDS_Camera.TRIGGER_MODES.
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_trigger_mode(IDS_Camera.TRIGGER_MODES[value])
        self._cam.get_trigger_mode()

    def cam_flash_cbox_changed(self, param: uEyeParams):
        '''
        Slot for changed flash mode

        Parameters
        ----------
        param : uEyeParams
            parameter path to index of selected flash mode from IDS_Camera.FLASH_MODES
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_flash_mode(IDS_Camera.FLASH_MODES[value])
        self._cam.get_flash_mode(output=True)

    def cam_pixel_cbox_changed(self, param: uEyeParams):
        '''
        Slot for changed pixel clock

        Parameters
        ----------
        param : uEyeParams
            parameter path Enum value pointing to pixel clock parameter in MHz
            (note that only certain values are allowed by camera)
        '''
        value = self.camera_options.get_param_value(param)
        self._cam.set_pixel_clock(int(value))
        self._cam.get_pixel_clock_info(False)

        self.refresh_framerate()

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
        # self._cam.get_framerate(False)

        self.camera_options.set_param_value(
            uEyeParams.FRAMERATE,
            self._cam.current_framerate.value,
            self.cam_framerate_value_changed,
        )

        self.refresh_exposure()

    def exposure_spin_changed(self, param, value: float):
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

    def refresh_framerate(self, value=None):
        self._cam.get_framerate_range(False)

        framerate = self.camera_options.get_param(uEyeParams.FRAMERATE)
        framerate.setLimits(
            (self._cam.min_framerate.value, self._cam.max_framerate.value)
        )
        framerate.setOpts(step=self._cam.increment_framerate.value)

        self.cam_framerate_value_changed(framerate, self._cam.max_framerate.value)

    def refresh_exposure(self):
        self._cam.get_exposure_range(False)

        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (self._cam.exposure_range[0].value, self._cam.exposure_range[1].value)
        )
        exposure.setOpts(step=self._cam.exposure_range[2].value)
        exposure.setValue(self.cam.exposure_range[1].value)

    def refresh_flash(self):
        self._cam.get_flash_range(False)

        duration = self.camera_options.get_param(uEyeParams.FLASH_DURATION)
        duration.setLimits((0, self._cam.flash_max.u32Duration.value))
        duration.setOpts(step=self._cam.flash_inc.u32Duration.value)
        duration.setValue(0)

        delay = self.camera_options.get_param(uEyeParams.FLASH_DURATION)
        delay.setLimits((0, self._cam.flash_max.s32Delay.value))
        delay.setOpts(step=self._cam.flash_inc.s32Delay.value)
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
        self._cam.set_flash_params(self._cam.flash_cur.s32Delay.value, value)
        self._cam.get_flash_params()

        self.camera_options.set_param_value(
            uEyeParams.FLASH_DURATION,
            self._cam.flash_cur.u32Duration.value,
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
        self._cam.set_flash_params(value, self._cam.flash_cur.u32Duration.value)

        self.camera_options.set_param_value(
            uEyeParams.FLASH_DELAY,
            self._cam.flash_cur.s32Delay.value,
            self.cam_flash_delay_value_changed,
        )

    def start_free_run(self, param=None, Prefix=''):
        '''
        Starts free run acquisition mode

        Parameters
        ----------
        param : Parameter
            the parameter that was activated.
        Prefix : str
            an extra prefix added to the image stack file name.
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

    def start_software_triggered(self, param=None, Prefix=''):
        '''
        Starts trigger acquisition mode

        Parameters
        ----------
        param : Parameter
            the parameter that was activated.
        Prefix : str
            an extra prefix added to the image stack file name.
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
            return code from IDS_Camera, ueye.IS_SUCCESS = 0 to run.
        cam : IDS_Camera
            the IDS_Camera used to acquire frames.
        '''
        try:
            nRet = ueye.IS_SUCCESS
            time = QDateTime.currentDateTime()
            # Continuous image capture
            while nRet == ueye.IS_SUCCESS:
                self.acq_job.capture_time = time.msecsTo(QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                if not self._cam.capture_video:
                    ueye.is_FreezeVideo(self._cam.hCam, ueye.IS_WAIT)

                nRet = self._cam.wait_for_next_image(500, not self.mini)
                if nRet == ueye.IS_SUCCESS:
                    self._cam.get_pitch()
                    data = self._cam.get_data()

                    self._buffer.put(data.copy())
                    # add sensor temperature to the stack
                    self._temps.put(self._cam.get_temperature())
                    self.acq_job.frames_captured = self.acq_job.frames_captured + 1
                    if (
                        self.acq_job.frames_captured >= self.acq_job.frames
                        and not self.mini
                    ):
                        self.acq_job.c_event.set()
                        self.acq_job.stop_threads = True
                        logging.debug('Stop')
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

            self._threadpool.releaseThread()

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

    def dispose(self):
        self.cam.free_memory()
        self.cam.dispose()

    def save_config(self):
        filename, _ = getSaveFileName(
            self, 'Save config', filter='JSON Files (*.json);;'
        )

        if len(filename) > 0:
            config = {
                'model': self.cam.sInfo.strSensorName.decode('utf-8'),
                'serial': self.cam.cInfo.SerNo.decode('utf-8'),
                'clock speed': self.cam.pixel_clock.value,
                'trigger': self.camera_options.get_param_value(uEyeParams.TRIGGER_MODE),
                'flash mode': self.camera_options.get_param_value(
                    uEyeParams.FLASH_MODE
                ),
                'framerate': self.cam.current_framerate.value,
                'exposure': self.cam.exposure_current.value,
                'flash duration': self.cam.flash_cur.u32Duration.value,
                'flash delay': self.cam.flash_cur.s32Delay.value,
                'ROI w': self.cam.set_rectROI.s32Width.value,
                'ROI h': self.cam.set_rectROI.s32Height.value,
                'ROI x': self.cam.set_rectROI.s32X.value,
                'ROI y': self.cam.set_rectROI.s32Y.value,
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
                        uEyeParams.PIXEL_CLOCK, str(config['clock speed'])
                    )
                    self.camera_options.set_param_value(
                        uEyeParams.TRIGGER_MODE, config['trigger']
                    )
                    self.camera_options.set_param_value(
                        uEyeParams.FLASH_DURATION, config['flash mode']
                    )

                    self.camera_options.set_param_value(
                        uEyeParams.FRAMERATE, config['framerate']
                    )

                    self.camera_options.set_param_value(
                        CamParams.EXPOSURE, config['exposure']
                    )

                    self.camera_options.set_param_value(
                        uEyeParams.FLASH_DURATION, config['flash duration']
                    )

                    self.camera_options.set_param_value(
                        uEyeParams.FLASH_DELAY, config['flash delay']
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
