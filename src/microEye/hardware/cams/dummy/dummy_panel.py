import logging
import time
import traceback

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.micam import DummyParams, miDummy
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.metadata_tree import MetaParams


class Dummy_Panel(Camera_Panel):
    '''
    A Qt Widget for a dummy camera
     | Inherits Camera_Panel
    '''

    PARAMS = DummyParams

    def __init__(self, mini=False, *args, **kwargs):
        '''
        Initializes a new Dummy_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        mini : bool, optional
            Flag indicating if this is a mini camera panel, by default False.

        Other Parameters
        ---------------
        *args
            Arguments to pass to the Camera_Panel constructor.

        **kwargs
            Keyword arguments to pass to the Camera_Panel constructor.
        '''
        super().__init__(miDummy(), mini, *args, **kwargs)

    def _init_camera_specific(self):
        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_increment, suffix=self._cam.exposure_unit
        )
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        for prop in self.cam.property_tree():
            self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, prop)

        # start freerun mode button
        freerun = self.get_event_action(DummyParams.FREERUN)
        self.camera_options.add_param_child(CamParams.ACQUISITION, freerun)
        self.camera_options.get_param(DummyParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        stop = {'name': str(DummyParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQUISITION, stop)
        self.camera_options.get_param(DummyParams.STOP).sigActivated.connect(self.stop)

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.getWidth()),
            (0, self.cam.getHeight()),
            (32, self.cam.getWidth()),
            (32, self.cam.getHeight()),
        )

    @property
    def cam(self):
        '''The miDummy property.

        Returns
        -------
        miDummy
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: miDummy):
        '''The miDummy property.

        Parameters
        ----------
        cam : miDummy
            the miDummy to set as panel camera.
        '''
        self._cam = cam

    def exposure_spin_changed(self, param, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in milliseconds
        '''
        self._cam.exposure = value

        self.refresh_exposure()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, float(self._cam.exposure_current)
        )

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_spin_changed
        )

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : miDummy
            the miDummy used to acquire frames.
        '''
        try:
            # Continuous image capture
            while self.acq_job.frames_captured < self.acq_job.frames or self.mini:
                cycle_time = time.perf_counter_ns()

                self._buffer.put(
                    self._cam.get_dummy_image_from_pattern(
                        self.acq_job.frames_captured / 100
                    ).tobytes()
                )
                # add sensor temperature to the stack
                self._temps.put(self.cam.get_temperature())
                self.acq_job.frames_captured += 1

                diff = (
                    int(self._cam.exposure * 1000)
                    - (time.perf_counter_ns() - cycle_time) // 1000
                )
                if diff > 0:
                    QtCore.QThread.usleep(diff)
                else:
                    QtCore.QThread.usleep(100)  # sleep 100us

                if self.acq_job.stop_threads:
                    break  # in case stop threads is initiated

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

    def dispose(self):
        miDummy.instances.remove(self.cam)
