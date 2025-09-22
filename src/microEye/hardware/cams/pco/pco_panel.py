import logging
import traceback

from microEye.hardware.cams.camera_options import CamParams
from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.hardware.cams.pco.pco_cam import (
    ConfigKeys,
    DescriptionKeys,
    RecorderModes,
    pco_cam,
    pcoParams,
)
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.metadata_tree import MetaParams


class PCO_Panel(Camera_Panel):
    '''
    A Qt Widget for controlling an PCO Camera through pco.python
     | Inherits Camera_Panel
    '''

    PARAMS = pcoParams

    def __init__(self, cam: pco_cam, mini=False, *args, **kwargs):
        '''
        Initializes a new PCO_Panel Qt widget.

        Inherits Camera_Panel.

        Parameters
        ----------
        cam : pco_cam
            PCO Camera python adapter.

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

        # exposure auto

        # trigger mode

        # trigger source

        # trigger selector

        # trigger activation

        # start freerun mode button
        self.camera_options.get_param(pcoParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        self.camera_options.get_param(pcoParams.STOP).sigActivated.connect(self.stop)

        # config buttons
        # load = {'name': str(pcoParams.LOAD), 'type': 'action'}
        # self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, load)
        # self.camera_options.get_param(pcoParams.LOAD).sigActivated.connect(
        #     self.load_config
        # )

        # save = {'name': str(pcoParams.SAVE), 'type': 'action'}
        # self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, save)
        # self.camera_options.get_param(pcoParams.SAVE).sigActivated.connect(
        #     self.save_config
        # )

        # ROI
        self.camera_options.set_roi_limits(
            (1, self.cam.max_dim[0]),
            (1, self.cam.max_dim[1]),
            (self.cam.min_dim[0], self.cam.max_dim[0]),
            (self.cam.min_dim[1], self.cam.max_dim[1]),
        )

        # GPIOs

        # Timers

    @property
    def cam(self):
        '''The pco_cam property.

        Returns
        -------
        pco_cam
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: pco_cam):
        '''The pco_cam property.

        Parameters
        ----------
        cam : pco_cam
            the pco_cam to set as panel camera.
        '''
        self._cam = cam

    def setExposure(self, value):
        '''Sets the exposure time widget of camera

        Parameters
        ----------
        value : float
            selected exposure time
        '''
        super().setExposure(value / 1e3)


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
            MetaParams.EXPOSURE, float(self._cam.exposure_current * 1000)
        )
        if self.master:
            self.exposureChanged.emit()

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_changed
        )

    def refresh_framerate(self, value=None):
        pass

    def cam_capture(self, *args, **kwargs):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : pco_cam
            the pco_cam used to acquire frames.
        '''
        try:
            # Continuous image capture
            self._cam.record(
                # self.acq_job.frames,
                100,
                RecorderModes.RING_BUFFER,
            )

            while not self.acq_job.c_event.isSet():
                if self.acq_job.frames_captured < self.acq_job.frames or self.mini:
                    self._cam.wait_for_new_image()
                    self._buffer.put(self.cam.get_image(0xFFFFFFFF)[0])
                    # add sensor temperature to the stack
                    self._temps.put(self.cam.get_temperature())
                    self.acq_job.frames_captured = self.acq_job.frames_captured + 1
                if (
                    self.acq_job.frames_captured > self.acq_job.frames - 1
                    and not self.mini
                ):
                    self.acq_job.c_event.set()
                    self.acq_job.stop_threads = True
                    logging.debug('Stop')
                    print('Capture Stopped!')

            self.acq_job.c_event.wait()
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            self._cam.stop()
            self._cam.cam.rec.cleanup()
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

if __name__ == '__main__':
    import sys

    from microEye.hardware.cams.pco.pco_cam import pco_cam
    from microEye.hardware.cams.pco.pco_panel import PCO_Panel
    from microEye.qt import QApplication

    app = QApplication(sys.argv)

    cam = pco_cam()

    panel = PCO_Panel(cam)
    panel.show()

    sys.exit(app.exec())
