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
from microEye.hardware.pycromanager.devices import PycroCamera
from microEye.qt import QDateTime, QtCore, QtWidgets, getOpenFileName, getSaveFileName
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

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'Pycromanager')
        self.OME_tab.set_param_value(MetaParams.DET_MODEL, cam.label)
        self.OME_tab.set_param_value(MetaParams.DET_SERIAL, 'N/A')
        self.OME_tab.set_param_value(MetaParams.DET_TYPE, 'CMOS')

        self.camera_options.set_param_value(CamParams.FRAMES, 10000)

        for property in self._cam.property_tree():
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
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

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
        self.timer = QtCore.QTimer(self, timeout=self.update_params, interval=1000)
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
        for property in self._cam.get_prop_names():
            param = self.camera_options.get_param(
                '.'.join([CamParams.ACQ_SETTINGS.value, property])
            )

            new_value = type(param.value())(self._cam.get_property(property))
            if param and new_value != param.value():
                param.setValue(new_value, self.update_cam)

        self.refresh_exposure()
        self.updateInfo()

    def set_ROI(self):
        '''Sets the ROI for the slected PycroCamera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.set_roi(*self.camera_options.get_roi_info(False))

    def reset_ROI(self):
        '''Resets the ROI for the slected PycroCamera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        self.cam.reset_roi()

        self.camera_options.set_roi_info(
            0, 0, int(self.cam.width), int(self.cam.height)
        )

    def center_ROI(self):
        '''Sets the ROI for the slected PycroCamera'''
        if self.cam.acquisition:
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!'
            )
            return  # if acquisition is already going on

        x, y, width, height = self.camera_options.get_roi_info(False)
        x = (int(self.cam.width) - width) // 2
        y = (int(self.cam.height) - height) // 2

        self.cam.set_roi(x, y, width, height)

        self.camera_options.set_roi_info(x, y, width, height)

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
                        # x, y, w, h = result
                        self.camera_options.set_roi_info(*result)

                self.worker = QThreadWorker(work_func)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    def select_ROIs(self):
        '''
        Opens a dialog to select multiple ROIs from the last image.
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

    def exposure_spin_changed(self, param, value: float):
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
            CamParams.EXPOSURE, self._cam.exposure_current, self.exposure_spin_changed
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
