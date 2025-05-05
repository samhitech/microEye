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
from microEye.hardware.cams.pco.pco_cam import (
    ConfigKeys,
    DescriptionKeys,
    RecorderModes,
    pco_cam,
)
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
from microEye.utils.gui_helper import get_scaling_factor
from microEye.utils.metadata_tree import MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage


class pcoParams(Enum):
    FREERUN = 'Acquisition.Freerun'
    STOP = 'Acquisition.Stop'
    # EXPOSURE_MODE = 'Acquisition Settings.Exposure Mode'
    # EXPOSURE_AUTO = 'Acquisition Settings.Exposure Auto'
    DELAY = 'Acquisition Settings.Delay'
    # TRIGGER_MODE = 'Acquisition Settings.Trigger Mode'
    # PIXEL_FORMAT = 'Acquisition Settings.Pixel Format'

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

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'Excelitas PCO')
        self.OME_tab.set_param_value(MetaParams.DET_MODEL, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_SERIAL, self._cam.serial)
        self.OME_tab.set_param_value(MetaParams.DET_TYPE, 'CMOS')

        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits((self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_increment, suffix=self._cam.exposure_unit
        )
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        # exposure auto

        delay = {
            'name': str(pcoParams.DELAY),
            'type': 'float',
            'value': self._cam.delay_current,
            'dec': False,
            'decimals': 6,
            'step': 0.1,
            'limits': [self._cam.delay_range[0], self._cam.delay_range[1]],
            'suffix': self._cam.exposure_unit,
            'enabled': True,
        }
        self.camera_options.add_param_child(CamParams.ACQ_SETTINGS, delay)
        self.camera_options.get_param(pcoParams.DELAY).sigValueChanged.connect(
            lambda: self.cam_cbox_changed(pcoParams.DELAY)
        )

        # trigger mode

        # trigger source

        # trigger selector

        # trigger activation

        # start freerun mode button
        freerun = self.get_event_action(pcoParams.FREERUN)
        self.camera_options.add_param_child(CamParams.ACQUISITION, freerun)
        self.camera_options.get_param(pcoParams.FREERUN).sigActivated.connect(
            self.start_free_run
        )

        # stop acquisition button
        stop = {'name': str(pcoParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(CamParams.ACQUISITION, stop)
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

    def set_ROI(self):
        '''Sets the ROI for the slected pco_cam'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.setROI(self.camera_options.get_roi_info())

        self.camera_options.set_roi_info(*self.cam._roi)

    def reset_ROI(self):
        '''Resets the ROI for the slected IDS_Camera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.resetROI()

        self.camera_options.set_roi_info(*self.cam._roi)

    def center_ROI(self):
        '''Sets the ROI for the slected pco_cam'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        width, height = self.camera_options.get_roi_info(True)[:2]

        self.cam.setROI(
            (
                (self.cam.max_dim[0] - width) // 2 + 1,
                (self.cam.max_dim[1] - height) // 2 + 1,
                width,
                height,
            )
        )

        self.camera_options.set_roi_info(*self.cam._roi)

    def select_ROI(self):
        '''
        Opens a dialog to select a ROI from the last image.
        '''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        if self.acq_job is not None:
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
                        x, y, w, h = result
                        self.camera_options.set_roi_info(x + 1, y + 1, w, h)

                self.worker = QThreadWorker(work_func)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    def cam_cbox_changed(self, param: pcoParams):
        '''
        Slot for changed combobox values

        Parameters
        ----------
        param : pcoParams
            selected parameter enum
        '''
        value = self.camera_options.get_param_value(param)

        if param == pcoParams.DELAY:
            self._cam.setDelay(value)

    def exposure_spin_changed(self, param, value: float):
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
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_spin_changed
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
