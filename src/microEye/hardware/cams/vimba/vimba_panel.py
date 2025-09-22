import logging
import traceback

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.vimba import INSTANCE, vb
from microEye.hardware.cams.vimba.vimba_cam import VimbaParams, vimba_cam
from microEye.qt import (
    QDateTime,
    QtCore,
    QtGui,
    QtWidgets,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.metadata_tree import MetaParams


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


    def _init_camera_specific(self):
        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_increment, suffix=self._cam.exposure_unit
        )
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_changed)

        for prop in self.cam.property_tree():
            parent = prop.pop('parent', CamParams.ACQ_SETTINGS)
            self.camera_options.add_param_child(
                parent, prop)


        # Frame Rate
        self.camera_options.get_param(
            VimbaParams.FRAMERATE_ENABLED
        ).sigStateChanged.connect(self.framerate_enabled)

        self.camera_options.get_param(VimbaParams.FRAMERATE).sigValueChanged.connect(
            self.framerate_changed
        )

        # pixel formats combobox
        self.camera_options.get_param(VimbaParams.PIXEL_FORMAT).sigValueChanged.connect(
            lambda: self.refresh_framerate()
        )

        # start freerun mode button
        self.camera_options.get_param(VimbaParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        self.camera_options.get_param(VimbaParams.STOP).sigActivated.connect(self.stop)

        # config buttons
        self.camera_options.get_param(VimbaParams.LOAD).sigActivated.connect(
            self.load_config
        )
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
        self.camera_options.get_param(
            VimbaParams.LINE_SELECTOR
        ).sigValueChanged.connect(self.io_line_changed)

        self.camera_options.get_param(VimbaParams.SET_IO_CONFIG).sigActivated.connect(
            self.set_io_line_config
        )

        # Timers
        def reset_timer():
            with self._cam:
                self.cam.reset_timer()

        def update_timer():
            with self._cam:
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
            with self._cam:
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
        with self._cam:
            super().set_ROI()

    def reset_ROI(self):
        '''Resets the ROI for the slected IDS_Camera'''
        with self._cam:
            super().reset_ROI()

    def exposure_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in micro-seconds
        '''
        with self._cam:
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
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_changed
        )

    def refresh_framerate(self, value=None):
        with self._cam:
            if value:
                self._cam.set_framerate(value)
            self._cam.get_framerate(False)

        framerate = self.camera_options.get_param(VimbaParams.FRAMERATE)
        framerate.setLimits(
            (self._cam.frameRate_range[0], self._cam.frameRate_range[1])
        )
        framerate.setValue(self._cam.frameRate)

    def framerate_enabled(self, value):
        with self._cam:
            self.cam.set_acquisition_framerate_enable(
                self.camera_options.get_param_value(VimbaParams.FRAMERATE_ENABLED)
            )

        self.camera_options.get_param(VimbaParams.FRAMERATE).setOpts(
            enabled=self.camera_options.get_param_value(VimbaParams.FRAMERATE_ENABLED)
        )
        self.refresh_framerate()

    def framerate_changed(self, param, value: float):
        self.refresh_framerate(value)

    def io_line_changed(self, value):
        with self._cam:
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
        with self._cam:
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

    def _capture_handler(self, cam: vb.Camera, stream: vb.Stream, frame: vb.Frame):
        if self.acq_job.frames_captured < self.acq_job.frames or self.mini:
            self._buffer.put(frame.as_numpy_ndarray().squeeze())
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
            with self._cam:
                self._cam.cam.start_streaming(self._capture_handler)

                self.acq_job.c_event.wait()
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            with self._cam:
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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        with self._cam:
            self.cam.cam._close()
        return super().closeEvent(event)

    def save_config(self):
        filename, _ = getSaveFileName(
            self, 'Save config', filter='XML Files (*.xml);;'
        )

        if len(filename) > 0:
            with self._cam:
                self.cam.cam.save_settings(filename, vb.PersistType.All)

            QtWidgets.QMessageBox.information(self, 'Info', 'Config saved.')
        else:
            QtWidgets.QMessageBox.warning(self, 'Warning', 'Config not saved.')

    def load_config(self):
        filename, _ = getOpenFileName(
            self, 'Load config', filter='XML Files (*.xml);;'
        )

        if len(filename) > 0:
            with self._cam:
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
