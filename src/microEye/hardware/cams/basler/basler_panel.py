import logging
import traceback
from time import perf_counter_ns

from microEye.hardware.cams.basler.basler_cam import BaslerParams, basler_cam, pylon
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import AcquisitionJob, Camera_Panel, Queue
from microEye.qt import (
    QDateTime,
    QtCore,
    QtGui,
    QtWidgets,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.gui_helper import get_scaling_factor
from microEye.utils.metadata_tree import MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage


class ImageHandler(pylon.ImageEventHandler):
    def __init__(self, images: Queue, temps: Queue, acq_job: AcquisitionJob):
        super().__init__()
        self.images = images
        self.temps = temps
        self.acq_job = acq_job
        self.time = perf_counter_ns()

    def OnImageGrabbed(self, camera: pylon.InstantCamera, grabResult: pylon.GrabResult):
        time = perf_counter_ns()
        self.acq_job.capture_time = (time - self.time) / 1e6
        self.time = time
        try:
            if (
                grabResult.GrabSucceeded()
                and self.acq_job.frames_captured < self.acq_job.frames
            ):
                # check image contents
                self.images.put_nowait(grabResult.Array)
                # add sensor temperature to the stack
                self.temps.put_nowait(camera.DeviceTemperature())
                self.acq_job.frames_captured = self.acq_job.frames_captured + 1
        except Exception as e:
            traceback.print_exc()


class Basler_Panel(Camera_Panel):
    '''
    A Qt Widget for controlling a Basler Camera through pyPylon
     | Inherits Camera_Panel
    '''

    PARAMS = BaslerParams

    def __init__(self, cam: basler_cam, mini=False, *args, **kwargs):
        '''
        Initializes a new Basler_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : basler_cam
            Basler Camera python adapter.

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
            signal = prop.pop('signal', None)
            self.camera_options.add_param_child(parent, prop)
            if signal:
                path = parent.value + f".{prop['name']}"
                self.camera_options.get_param(path).sigActivated.connect(signal)

        # Frame Rate
        self.camera_options.get_param(
            BaslerParams.ACQUISITION_FRAMERATE_ENABLE
        ).sigStateChanged.connect(self.framerate_enabled)

        self.camera_options.get_param(
            BaslerParams.ACQUISITION_FRAMERATE
        ).sigValueChanged.connect(self.framerate_changed)

        # pixel formats combobox
        self.camera_options.get_param(
            BaslerParams.PIXEL_FORMAT
        ).sigValueChanged.connect(lambda: self.refresh_framerate())

        # start freerun mode button
        self.camera_options.get_param(BaslerParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        self.camera_options.get_param(BaslerParams.STOP).sigActivated.connect(self.stop)

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.max_dim[0]),
            (0, self.cam.max_dim[1]),
            (self.cam.min_dim[0], self.cam.max_dim[0]),
            (self.cam.min_dim[1], self.cam.max_dim[1]),
        )

        # GPIOs
        self.camera_options.get_param(
            BaslerParams.LINE_SELECTOR
        ).sigValueChanged.connect(self.io_line_changed)

        def update_timer():
            delay_param = self.camera_options.get_param(BaslerParams.TIMER_DELAY)
            delay_param.setLimits((self.cam.TimerDelay.Min, self.cam.TimerDelay.Max))
            delay_param.setValue(self.cam.TimerDelay())

            duration_param = self.camera_options.get_param(BaslerParams.TIMER_DURATION)
            duration_param.setLimits(
                (self.cam.TimerDuration.Min, self.cam.TimerDuration.Max)
            )
            duration_param.setValue(self.cam.TimerDuration())

            arm_delay = self.camera_options.get_param(
                BaslerParams.TIMER_TRIGGER_ARM_DELAY
            )
            arm_delay.setLimits(
                (self.cam.TimerTriggerArmDelay.Min, self.cam.TimerTriggerArmDelay.Max)
            )
            arm_delay.setValue(self.cam.TimerTriggerArmDelay())

            self.camera_options.set_param_value(
                BaslerParams.TIMER_TRIGGER_ACTIVATION,
                self.cam.TimerTriggerActivation(),
            )
            self.camera_options.set_param_value(
                BaslerParams.TIMER_TRIGGER_SOURCE, self.cam.TimerTriggerSource()
            )
            self.camera_options.set_param_value(
                BaslerParams.TIMER_TRIGGER_SOURCE, self.cam.TimerTriggerSource()
            )

        self.camera_options.get_param(
            BaslerParams.TIMER_SELECTOR
        ).sigValueChanged.connect(update_timer)

        # start a timer to update the camera params
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_params)
        self.timer.start()

    def update_params(self):
        if self._dispose_cam:
            return

        def update(event):
            '''
            Fetches the current readings and settings from the laser device.
            '''
            for prop in self.cam.property_tree():
                if prop['type'] == 'action':
                    continue

                parent = prop.pop('parent', CamParams.ACQ_SETTINGS)
                name = prop.pop('name', None)
                value = prop.pop('value', None)
                limits = prop.pop('limits', None)
                enabled = prop.pop('enabled', False)

                param = self.camera_options.get_param('.'.join([parent.value, name]))

                with param.treeChangeBlocker():
                    if value != param.value():
                        param.setValue(value)
                    if enabled != param.opts.get('enabled', True):
                        param.setOpts(enabled=enabled)
                    if limits and (
                        len(limits) != len(param.opts.get('limits', []))
                        or any(
                            old != new
                            for old, new in zip(param.opts.get('limits', []), limits)
                        )
                    ):
                        param.setLimits(limits)

            self.refresh_exposure()

        worker = QThreadWorker(update)
        worker.signals.finished.connect(
            self.timer.start
        )  # Restart the timer after fetching stats

        QtCore.QThreadPool.globalInstance().start(worker)

    @property
    def cam(self):
        '''The basler_cam property.

        Returns
        -------
        basler_cam
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: basler_cam):
        '''The basler_cam property.

        Parameters
        ----------
        cam : basler_cam
            the basler_cam to set as panel camera.
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

    def exposure_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in micro-seconds
        '''
        self._cam.setExposure(value)

        self.refresh_exposure()
        self.refresh_framerate()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, float(self._cam.exposure_current / 1000)
        )
        if self.master:
            self.exposureChanged.emit()

    def refresh_exposure(self):
        param = self.camera_options.get_param(CamParams.EXPOSURE)
        with param.treeChangeBlocker():
            param.setValue(self._cam.exposure_current)

    def refresh_framerate(self, value=None):
        if value is not None:
            self.cam.framerate = value

        framerate = self.camera_options.get_param(BaslerParams.ACQUISITION_FRAMERATE)
        framerate.setLimits((self.cam.min_framerate, self.cam.max_framerate))
        framerate.setValue(self.cam.framerate, self.framerate_changed)

    def framerate_enabled(self, value):
        self.camera_options.get_param(BaslerParams.ACQUISITION_FRAMERATE).setOpts(
            enabled=self.camera_options.get_param_value(
                BaslerParams.ACQUISITION_FRAMERATE_ENABLE
            )
        )
        self.refresh_framerate()

    def framerate_changed(self, param, value: float):
        self.refresh_framerate(value)

    def io_line_changed(self, value):
        line = self.camera_options.get_param_value(BaslerParams.LINE_SELECTOR)
        self.camera_options.set_param_value(
            BaslerParams.LINE_FORMAT, self.cam.cam.LineFormat()
        )
        self.camera_options.set_param_value(
            BaslerParams.LINE_STATUS, self.cam.cam.LineStatus()
        )
        self.camera_options.set_param_value(
            BaslerParams.LINE_INVERTER, self.cam.cam.LineInverter()
        )
        self.camera_options.set_param_value(
            BaslerParams.LINE_MODE, self.cam.cam.LineMode()
        )
        self.camera_options.set_param_value(
            BaslerParams.LINE_SOURCE, self.cam.cam.LineSource()
        )

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : basler_cam
            the basler_cam used to acquire frames.
        '''
        try:
            # instantiate callback handler
            # handler = ImageHandler(self._buffer, self._temps, self.acq_job)
            # register with the pylon loop
            # self.cam.cam.RegisterImageEventHandler(
            #     handler, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete
            # )

            # Continuous image capture
            if not self.mini:
                self.cam.StartGrabbingMax(self.acq_job.frames, grabLoopType=1)
            else:
                self.cam.StartGrabbing(grabLoopType=1)

            self.time = perf_counter_ns()

            while self.cam.IsGrabbing() and not self.acq_job.c_event.is_set():
                cycle_time = perf_counter_ns()
                self.acq_job.capture_time = (cycle_time - self.time) / 1e6
                self.time = cycle_time
                with self.cam.cam.RetrieveResult(100) as res:
                    if res.GrabSucceeded():
                        self._buffer.put(res.Array.copy())
                        # add sensor temperature to the stack
                        self._temps.put(self.cam.get_temperature())
                        self.acq_job.frames_captured += 1

                if (
                    self.acq_job.frames_captured >= self.acq_job.frames
                    and not self.mini
                ):
                    self.acq_job.c_event.set()
                    self.acq_job.stop_threads = True
                    logging.debug('Stop')
                    print('Capture Stopped!')

                QtCore.QThread.usleep(100)
        except Exception:
            traceback.print_exc()
        finally:
            self.cam.StopGrabbing()
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
        self.cam.close()
        self.timer.stop()
        self.timer.deleteLater()
        self.timer = None
        super().closeEvent(event)
