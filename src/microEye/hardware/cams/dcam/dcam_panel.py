import logging
import traceback
from time import perf_counter_ns
from typing import Optional

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel, Parameter
from microEye.hardware.cams.dcam.dcam import DcamParams, dcamera
from microEye.qt import (
    QDateTime,
    QtCore,
    QtGui,
    QtWidgets,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.metadata_tree import MetaParams
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)


class Dcam_Panel(Camera_Panel):
    '''
    A Qt Widget for controlling a Hamamatsu camera using the DCAM API.
     | Inherits Camera_Panel
    '''

    PARAMS = DcamParams
    '''
    exposure time factor to convert to ms
    '''

    FACTOR = 1e-3

    def __init__(self, cam: dcamera, mini=False, *args, **kwargs):
        '''
        Initializes a new Hamamatsu Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : dcamera
            Hamamatsu Camera python adapter.

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
            parent = prop.pop('parent', CamParams.ACQ_SETTINGS.value)
            signal = prop.pop('signal', None)
            self.camera_options.add_param_child(parent, prop)
            if signal:
                path = parent + f".{prop['name']}"
                self.camera_options.get_param(path).sigActivated.connect(signal)

        self.camera_options.get_param(DcamParams.FRAMERATE).sigValueChanged.connect(
            self.framerate_changed
        )

        # start freerun mode button
        self.camera_options.get_param(DcamParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        self.camera_options.get_param(DcamParams.STOP).sigActivated.connect(self.stop)

        # ROI
        self.camera_options.set_roi_limits(
            (self.cam.subarrayh_pos_range[0], self.cam.subarrayh_pos_range[1]),
            (self.cam.subarrayv_pos_range[0], self.cam.subarrayv_pos_range[1]),
            (self.cam.subarrayh_range[0], self.cam.subarrayh_range[1]),
            (self.cam.subarrayv_range[0], self.cam.subarrayv_range[1]),
        )

        # OME pixel size
        MAGNIFICATION = 60
        self.OME_tab.set_param_value(
            MetaParams.PX_SIZE,
            float(1e3 * self._cam.get_sensor_pixel_width_um() / MAGNIFICATION),
        )
        self.OME_tab.set_param_value(
            MetaParams.PY_SIZE,
            float(1e3 * self._cam.get_sensor_pixel_height_um() / MAGNIFICATION),
        )

        # start a timer to update the camera params
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(250)  # Update every 250 ms after a change
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
                try:
                    if prop['type'] == 'action':
                        continue

                    parent = prop.pop('parent', CamParams.ACQ_SETTINGS)
                    name = prop.pop('name', None)
                    value = prop.pop('value', None)
                    limits = prop.pop('limits', None)
                    enabled = prop.pop('enabled', False)

                    param = self.camera_options.get_param('.'.join([parent, name]))

                    # if 'Source' in name or 'Active' in name:
                    #     print(
                    #         f'Updating parameter {name}: {value}, {limits}, {enabled}'
                    #     )

                    with param.treeChangeBlocker():
                        if limits and (
                            len(limits) != len(param.opts.get('limits', []))
                            or any(
                                old != new
                                for old, new in zip(
                                    param.opts.get('limits', []), limits
                                )
                            )
                        ):
                            param.setLimits(limits)
                        if value != param.value():
                            param.setValue(value)
                        if enabled != param.opts.get('enabled', True):
                            param.setOpts(enabled=enabled)
                except Exception as e:
                    logging.getLogger(__name__).error(
                        f"Error updating parameter {prop.get('name', 'unknown')}: {e}"
                    )
                    traceback.print_exc()

            self.refresh_exposure()
            self.refresh_framerate()

        worker = QThreadWorker(update)

        QtCore.QThreadPool.globalInstance().start(worker)

    def update_cam(self, param: Optional[Parameter], param_value):
        path = super().update_cam(param, param_value)

        # Restart the timer to update the camera params after a change
        if hasattr(self, 'timer') and self.timer is not None:
            self.timer.start()

        if path is None:
            return

    @property
    def cam(self):
        '''The dcamera property.

        Returns
        -------
        dcamera
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: dcamera):
        '''The dcamera property.

        Parameters
        ----------
        cam : dcamera
            the dcamera to set as panel camera.
        '''
        self._cam = cam

    def exposure_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in seconds
        '''
        with self._cam:
            self._cam.set_exposure(value)

        self.refresh_exposure()
        self.refresh_framerate()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, float(self._cam.exposure_current / self.FACTOR)
        )  # Meta Exposure in ms
        if self.master:
            self.exposureChanged.emit()

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_changed
        )

    def framerate_changed(self, param, value: float):
        self.refresh_framerate(value)

    def refresh_framerate(self, value=None):
        if value:
            self._cam.set_framerate(value)
        self._cam.get_framerate()

        framerate = self.camera_options.get_param(DcamParams.FRAMERATE)
        framerate.setLimits(
            (self._cam.framerate_range[0], self._cam.framerate_range[1])
        )
        framerate.setValue(self._cam.framerate_current)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cam.close()
        return super().closeEvent(event)

    def _capture_handler(self, frame):
        """Push one already-retrieved frame into the display/save pipeline,
        mirroring vimba_panel._capture_handler's budget check so free-run
        (mini) and finite-count acquisitions share the same stop logic."""
        if self.acq_job.frames_captured < self.acq_job.frames or self.mini:
            self._buffer.put(frame)
            self._temps.put(self.cam.get_temperature())
            self.acq_job.frames_captured += 1

        if self.acq_job.frames_captured > self.acq_job.frames - 1 and not self.mini:
            self.acq_job.c_event.set()
            self.acq_job.stop_threads = True
            logger.debug('Capture Stopped!')

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Polls the DCAM ring buffer and transfers acquired frames to the
        display processing stack.

        Parameters
        ----------
        cam : dcamera
            the dcamera used to acquire frames.
        '''
        try:
            self.cam.buffer_frames = self.cam.suggest_buffer_frames(
                expected_stall_ms=250
            )

            if not self.cam.start_freerun():
                logger.error('DCAM start_freerun() failed.')
                return QDateTime.currentDateTime()

            self.time = perf_counter_ns()

            while not self.acq_job.c_event.is_set():
                cycle_time = perf_counter_ns()
                self.acq_job.capture_time = (cycle_time - self.time) / 1e6
                self.time = cycle_time

                frames = self.cam.wait_and_pop_frames(
                    timeout_ms=100 + int(self.cam.exposure_current / Dcam_Panel.FACTOR)
                )

                for frame in frames:
                    self._capture_handler(frame)
                    if self.acq_job.c_event.is_set():
                        break

                QtCore.QThread.usleep(100)
        except Exception:
            traceback.print_exc()
        finally:
            self.cam.stop_freerun()
            self._cam.acquisition = False
            QtCore.QThreadPool.globalInstance().releaseThread()
        return QDateTime.currentDateTime()

    def getCaptureArgs(self) -> list:
        args = []
        return args
