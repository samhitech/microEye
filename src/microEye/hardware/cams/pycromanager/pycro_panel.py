import logging
import time
import traceback
from enum import Enum

from microEye.analysis.tools.roi_selectors import (
    MultiRectangularROISelector,
    convert_pos_size_to_rois,
    convert_rois_to_pos_size,
)
from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.pycromanager.pycro_cam import PycroCamera
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


class PycroParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    LOAD = 'Acquisition Settings.Load Config'
    SAVE = 'Acquisition Settings.Save Config'

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


class PycroPanel(Camera_Panel):
    '''
    A Qt Widget for a pycro-manager camera
     | Inherits Camera_Panel
    '''

    PARAMS = PycroParams

    def __init__(self, cam: PycroCamera, mini=False, *args, **kwargs):
        '''
        Initializes a new Dummy_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : PycroCamera
            The camera to be used in the panel.
        mini : bool, optional
            Flag indicating if this is a mini camera panel, by default False.

        Other Parameters
        ---------------
        *args
            Arguments to pass to the Camera_Panel constructor.

        **kwargs
            Keyword arguments to pass to the Camera_Panel constructor.
        '''
        if not isinstance(cam, PycroCamera) or cam is None:
            raise ValueError('Invalid camera type!')

        super().__init__(cam, mini, *args, **kwargs)


    def _init_camera_specific(self):
        self.camera_options.set_param_value(CamParams.FRAMES, 10000)

        for property in self._cam.property_tree():
            if '.' in property['name']:
                continue

            self.camera_options.add_param_child(
                CamParams.ACQ_SETTINGS,
                property,
            )
            self.camera_options.get_param(
                '.'.join([CamParams.ACQ_SETTINGS.value, property['name']])
            ).sigValueChanged.connect(self.update_cam)

        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(step=0.1, suffix='ms')
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_changed)

        # start freerun mode button
        freerun = self.get_event_action(PycroParams.FREERUN)
        self.camera_options.add_param_child(CamParams.ACQUISITION, freerun)
        self.camera_options.get_param(PycroParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        stop = {'name': str(PycroParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQUISITION, stop)
        self.camera_options.get_param(PycroParams.STOP).sigActivated.connect(self.stop)

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.getWidth()),
            (0, self.cam.getHeight()),
            (32, self.cam.getWidth()),
            (32, self.cam.getHeight()),
        )

        # start a timer to update the camera params
        self.timer = QtCore.QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_params)
        self.timer.start()

    @property
    def cam(self):
        '''The PycroCamera property.

        Returns
        -------
        PycroCamera
            the PycroCamera used as panel camera.
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: PycroCamera):
        '''The PycroCamera setter.

        Parameters
        ----------
        cam : PycroCamera
            the PycroCamera to set as panel camera.
        '''
        self._cam = cam

    def update_cam(self, param, value):
        if self._cam.has_property(param.name()):
            self._cam.set_property(param.name(), value)

            param.setValue(self._cam.get_property(param.name()), self.update_cam)

    def update_params(self):
        if self._dispose_cam:
            return

        def update(event):
            '''
            Fetches the current readings and settings from the laser device.
            '''
            for property in self._cam.get_prop_names():
                if '.' in property:
                    continue

                param = self.camera_options.get_param(
                    '.'.join([CamParams.ACQ_SETTINGS.value, property])
                )

                new_value = type(param.value())(self._cam.get_property(property))
                if param and new_value != param.value():
                    param.setValue(new_value, self.update_cam)

            self.refresh_exposure()
            self.updateInfo()

        worker = QThreadWorker(update)
        worker.signals.finished.connect(
            self.timer.start
        )  # Restart the timer after fetching stats

        QtCore.QThreadPool.globalInstance().start(worker)

    def exposure_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in milliseconds
        '''
        self._cam.setExposure(value)

        self.refresh_exposure()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, float(self._cam.exposure_current)
        )

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_changed
        )

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : PycroCamera
            the PycroCamera used to acquire frames.
        '''
        try:
            if self.mini:
                self._cam.start_continuous_acquisition(0)
            else:
                self._cam.start_sequence_acquisition(0, self.acq_job.frames)

            start_time = time.perf_counter_ns()
            # Continuous image capture
            while self._cam.is_sequence_running() or self.mini:
                # Check if there are images before trying to get one
                if self._cam.image_count > 0:
                    self.acq_job.capture_time = (
                        time.perf_counter_ns() - start_time
                    ) / 1e6
                    start_time = time.perf_counter_ns()
                    # Get the next image from the camera
                    self._buffer.put(self._cam.pop_image().tobytes())
                    # add sensor temperature to the stack
                    self._temps.put(self._cam.get_temperature())
                    self.acq_job.frames_captured += 1

                if self.acq_job.stop_threads or self.acq_job.c_event.is_set():
                    self._cam.stop_sequence_acquisition()
                    break  # in case stop threads is initiated

                QtCore.QThread.usleep(500)

            while self._cam.image_count > 0:
                self._buffer.put(self._cam.pop_image().tobytes())
                self._temps.put(self.cam.get_temperature())
                self.acq_job.frames_captured += 1

            self.acq_job.stop_threads = True
            logging.debug('Stop')
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            self._cam.acquisition = False
            QtCore.QThreadPool.globalInstance().releaseThread()
        return QDateTime.currentDateTime()

    def getCaptureArgs(self) -> list:
        '''User specific arguments to be passed to the parallelized
        Camera_Panel.cam_capture function.

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
        self.timer.stop()
        self.timer.deleteLater()
        self.timer = None
        super().closeEvent(event)


if __name__ == '__main__':
    import os
    import time

    import cv2

    from microEye.hardware.pycromanager.devices import (
        PycroCore,
        start_headless,
        stop_headless,
    )

    mm_path = 'C:/Program Files/Micro-Manager-2.0'
    config = os.path.join(mm_path, 'MMConfig_demo.cfg')
    port = 4827

    try:
        start_headless(
            mm_path,
            config,
            port=port,
            # python_backend=True,
        )

        cam = PycroCamera('Camera', port=port)

        app = QtWidgets.QApplication([])

        panel = PycroPanel(cam)

        panel.show()

        app.exec()
    except Exception as e:
        import traceback

        traceback.print_exc()
    finally:
        PycroCore._instances[port].close()
        # Stop the camera and close the acquisition
        stop_headless()
