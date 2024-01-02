import json
import os
import time
import traceback
from enum import Enum
from queue import Queue

import cv2
import numpy as np
import pyqtgraph as pg
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ...analysis.tools.roi_selectors import (
    MultiRectangularROISelector,
    convert_pos_size_to_rois,
    convert_rois_to_pos_size,
)
from ...shared.gui_helper import get_scaling_factor
from ...shared.metadata_tree import MetaParams
from ...shared.thread_worker import thread_worker
from ...shared.uImage import uImage
from ..widgets.qlist_slider import *
from . import Camera_Panel
from .camera_options import CamParams
from .thorlabs import *


class ThorCamParams(Enum):
    PIXEL_CLOCK = 'Camera.Pixel Clock (MHz)'
    FRAMERATE = 'Camera.Framerate Slider'
    FRAME_AVERAGING = 'Camera.Frame Averaging'
    TRIGGER_MODE = 'Camera.Trigger Mode'
    FLASH_MODE = 'Camera.Flash Mode'
    FLASH_DURATION = 'Camera.Flash Duration Slider'
    FLASH_DELAY = 'Camera.Flash Delay Slider'
    FREERUN = 'Camera.Start (Freerun)'
    TRIGGERED = 'Camera.Start (Triggered)'
    LOAD = 'Camera.Load'
    SAVE = 'Camera.Save'
    STOP = 'Camera.Stop'

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

    def __init__(self, threadpool, cam: thorlabs_camera,
                 mini: bool = False, *args, **kwargs):
        '''
        Initializes a new Thorlabs_Panel Qt widget
        | Inherits Camera_Panel

        Parameters
        ----------
        threadpool : QThreadPool
            The threadpool for multithreading
        cam : thorlabs_camera
            Thorlabs Camera python adapter
        '''
        super().__init__(
            threadpool, cam, mini,
            *args, **kwargs)

        # flag true to close camera adapter and dispose it
        self._dispose_cam = False

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'Thorlabs')
        self.OME_tab.set_param_value(
            MetaParams.DET_MODEL,
            self._cam.sInfo.strSensorName.decode('utf-8'))
        self.OME_tab.set_param_value(MetaParams.DET_SERIAL,
            self._cam.cInfo.SerNo.decode('utf-8'))
        self.OME_tab.set_param_value(MetaParams.DET_TYPE,
            'CMOS')

        # pixel clock label and combobox
        pixel_clock = {
            'name': str(ThorCamParams.PIXEL_CLOCK), 'type': 'list',
            'values': list(map(str, self._cam.pixel_clock_list[:]))}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, pixel_clock)
        self.camera_options.get_param(
            ThorCamParams.PIXEL_CLOCK).sigValueChanged.connect(
                lambda: self.cam_pixel_cbox_changed(ThorCamParams.PIXEL_CLOCK))

        # framerate slider control
        framerate = {
            'name': str(ThorCamParams.FRAMERATE), 'type': 'float',
            'value': int(self._cam.currentFrameRate.value), 'dec': False, 'decimals': 6,
            'step': self._cam.incFrameRate.value,
            'limits': [self._cam.minFrameRate.value, self._cam.maxFrameRate.value],
            'suffix': 'Hz'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, framerate)
        self.camera_options.get_param(
            ThorCamParams.FRAMERATE).sigValueChanged.connect(
                self.cam_framerate_value_changed)

        # exposure text box
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_range[2],
            suffix='ms')
        exposure.setValue(self._cam.exposure_current.value)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        # trigger mode combobox
        trigger_mode = {
            'name': str(ThorCamParams.TRIGGER_MODE), 'type': 'list',
            'values': list(TRIGGER.TRIGGER_MODES.keys())}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, trigger_mode)
        self.camera_options.get_param(
            ThorCamParams.TRIGGER_MODE).sigValueChanged.connect(
                lambda: self.cam_trigger_cbox_changed(ThorCamParams.TRIGGER_MODE))

        # flash mode combobox
        flash_mode = {
            'name': str(ThorCamParams.FLASH_MODE), 'type': 'list',
            'values': list(FLASH_MODE.FLASH_MODES.keys()),
            'enabled': not self.mini}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, flash_mode)
        self.camera_options.get_param(
            ThorCamParams.FLASH_MODE).sigValueChanged.connect(
                lambda: self.cam_flash_cbox_changed(ThorCamParams.FLASH_MODE))

        # flash duration slider
        falsh_duration = {
            'name': str(ThorCamParams.FLASH_DURATION), 'type': 'int',
            'value': 0,
            'dec': False, 'decimals': 6, 'suffix': 'us', 'enabled': not self.mini,
            'step': self._cam.flash_inc.u32Duration,
            'limits': [
                0,
                self._cam.flash_max.u32Duration]}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, falsh_duration)
        self.camera_options.get_param(
            ThorCamParams.FLASH_DURATION).sigValueChanged.connect(
                self.cam_flash_duration_value_changed)

        # flash delay
        falsh_delay = {
            'name': str(ThorCamParams.FLASH_DELAY), 'type': 'int',
            'value': 0,
            'dec': False, 'decimals': 6, 'suffix': 'us', 'enabled': not self.mini,
            'step': self._cam.flash_inc.s32Delay,
            'limits': [
                0,
                self._cam.flash_max.s32Delay]}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, falsh_delay)
        self.camera_options.get_param(
            ThorCamParams.FLASH_DELAY).sigValueChanged.connect(
                self.cam_flash_delay_value_changed)

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK,
            str(self._cam.pixel_clock_list[-1]))

        # start freerun mode
        freerun = {'name': str(ThorCamParams.FREERUN), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, freerun)
        self.camera_options.get_param(
            ThorCamParams.FREERUN).sigActivated.connect(
                self.start_free_run)

        # start trigger mode button
        triggered = {'name': str(ThorCamParams.TRIGGERED), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, triggered)
        self.camera_options.get_param(
            ThorCamParams.TRIGGERED).sigActivated.connect(
                self.start_software_triggered)

        # stop acquisition
        stop = {'name': str(ThorCamParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, stop)
        self.camera_options.get_param(
            ThorCamParams.STOP).sigActivated.connect(
                lambda: self.stop())

        # config buttons
        load = {'name': str(ThorCamParams.LOAD), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, load)
        self.camera_options.get_param(
            ThorCamParams.LOAD).sigActivated.connect(
                lambda: self.load_config())

        save = {'name': str(ThorCamParams.SAVE), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, save)
        self.camera_options.get_param(
            ThorCamParams.SAVE).sigActivated.connect(
                lambda: self.save_config())

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
        '''Sets the ROI for the slected thorlabs_camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!')
            return  # if acquisition is already going on

        self.cam.set_ROI(
            *self.camera_options.get_roi_info())

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK,
            str(self._cam.pixel_clock_list[0]))
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK,
            str(self._cam.pixel_clock_list[-1]))

        self.camera_options.set_roi_info(
            self.cam.set_rectROI.s32X,
            self.cam.set_rectROI.s32Y,
            self.cam.set_rectROI.s32Width,
            self.cam.set_rectROI.s32Height)

    def reset_ROI(self):
        '''Resets the ROI for the slected thorlabs_camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!')
            return  # if acquisition is already going on

        self.cam.reset_ROI()
        self.camera_options.set_roi_info(
            0, 0,
            self.cam.width,
            self.cam.height)

        # setting the highest pixel clock as default
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK,
            str(self._cam.pixel_clock_list[0]))
        self.camera_options.set_param_value(
            ThorCamParams.PIXEL_CLOCK,
            str(self._cam.pixel_clock_list[-1]))

    def center_ROI(self):
        '''Calculates the x, y values for a centered ROI'''
        _, _, w, h = self.camera_options.get_roi_info()
        x = (self.cam.rectROI.s32Width - w) // 2
        y = (self.cam.rectROI.s32Height - h) // 2

        self.camera_options.set_roi_info(x, y, w, h)

        self.set_ROI()

    def select_ROI(self):
        if self.acq_job.frame is not None:
            try:
                def work_func():
                    try:
                        image = uImage(self.acq_job.frame.image)

                        image.equalizeLUT()

                        scale_factor = get_scaling_factor(image.height, image.width)

                        selector = MultiRectangularROISelector.get_selector(
                            image._view, scale_factor, max_rois=1)
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

                self.worker = thread_worker(
                    work_func,
                    progress=False, z_stage=False)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    def select_ROIs(self):
        if self.acq_job is not None:
            try:
                def work_func():
                    try:
                        image = uImage(self.acq_job.frame.image)

                        image.equalizeLUT()

                        scale_factor = get_scaling_factor(image.height, image.width)

                        selector = MultiRectangularROISelector.get_selector(
                            image._view, scale_factor, max_rois=4, one_size=True)

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
                            CamParams.EXPORTED_ROIS)
                        rois_param.clearChildren()
                        for x, y, w, h in results:
                            self.camera_options.add_param_child(
                                CamParams.EXPORTED_ROIS,
                                {'name': 'ROI 1', 'type': 'str',
                                 'readonly': True, 'removable': True,
                                 'value': f'{x}, {y}, {w}, {h}'}
                            )

                self.worker = thread_worker(
                    work_func,
                    progress=False, z_stage=False)
                self.worker.signals.result.connect(done)
                # Execute
                self._threadpool.start(self.worker)
            except Exception:
                traceback.print_exc()

    @pyqtSlot(str)
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

    @pyqtSlot(str)
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

    @pyqtSlot(str)
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
            self._cam.currentFrameRate.value,
            self.cam_framerate_value_changed)

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
            self.exposure_spin_changed)

        self.refresh_flash()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, self._cam.exposure_current)
        if self.master:
            self.exposureChanged.emit()

    def refresh_framerate(self, range=False):
        self._cam.get_framerate_range(False)

        framerate = self.camera_options.get_param(ThorCamParams.FRAMERATE)
        framerate.setLimits(
            (self._cam.minFrameRate.value,
            self._cam.maxFrameRate.value))
        framerate.setOpts(step=self._cam.incFrameRate.value)

        self.cam_framerate_value_changed(framerate, self._cam.maxFrameRate.value)

    def refresh_exposure(self, range=False):
        self._cam.get_exposure_range(False)

        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (self._cam.exposure_range[0],
             self._cam.exposure_range[1]))
        exposure.setOpts(step=self._cam.exposure_range[2])
        exposure.setValue(self.cam.exposure_range[1])

    def refresh_flash(self):
        self._cam.get_flash_range(False)

        duration = self.camera_options.get_param(ThorCamParams.FLASH_DURATION)
        duration.setLimits(
            (0,
             self._cam.flash_max.u32Duration))
        duration.setOpts(step=self._cam.flash_inc.u32Duration)
        duration.setValue(0)

        delay = self.camera_options.get_param(ThorCamParams.FLASH_DURATION)
        delay.setLimits(
            (0,
             self._cam.flash_max.s32Delay))
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
            self.cam_flash_duration_value_changed)

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
        self._cam.set_flash_params(
            value, self._cam.flash_cur.u32Duration)

        self.camera_options.set_param_value(
            ThorCamParams.FLASH_DELAY,
            self._cam.flash_cur.s32Delay,
            self.cam_flash_delay_value_changed)

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

    def cam_capture(self, *args):
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
            while (nRet == 0):
                self.acq_job.capture_time = time.msecsTo(
                    QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                if not self._cam.capture_video:
                    self._cam.uc480.is_FreezeVideo(
                        self._cam.hCam, IS_DONT_WAIT)

                nRet = self._cam.is_WaitForNextImage(500, not self.mini)
                if nRet == CMD.IS_SUCCESS:
                    self._cam.get_pitch()
                    data = self._cam.get_data()

                    # if not np.array_equal(temp, data):
                    self._buffer.put(data.copy())
                    # add sensor temperature to the stack
                    self._temps.put(self._cam.get_temperature())
                    self.acq_job.frames_captured = \
                        self.acq_job.frames_captured + 1
                    if self.acq_job.frames_captured >= self.acq_job.frames \
                            and not self.mini:
                        self._stop_thread = True
                        logging.debug('Stop')
                    self._cam.unlock_buffer()

                QThread.usleep(100)  # sleep 100us

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

    def get_meta(self):
        meta = {
            'Frames': self.camera_options.get_param_value(CamParams.FRAMES),
            'model': self.cam.sInfo.strSensorName.decode('utf-8'),
            'serial': self.cam.cInfo.SerNo.decode('utf-8'),
            'clock speed': self.cam.pixel_clock.value,
            'trigger': self.camera_options.get_param_value(ThorCamParams.TRIGGER_MODE),
            'flash mode': self.camera_options.get_param_value(ThorCamParams.FLASH_MODE),
            'framerate': self.cam.currentFrameRate.value,
            'exposure': self.cam.exposure_current.value,
            'flash duration': self.cam.flash_cur.u32Duration,
            'flash delay': self.cam.flash_cur.s32Delay,
            'ROI w': self.cam.set_rectROI.s32Width,
            'ROI h': self.cam.set_rectROI.s32Height,
            'ROI x': self.cam.set_rectROI.s32X,
            'ROI y': self.cam.set_rectROI.s32Y,
            'Zoom': self.camera_options.get_param_value(CamParams.RESIZE_DISPLAY),
        }
        return meta

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save config', filter='JSON Files (*.json);;')

        if len(filename) > 0:

            config = {
                'model': self.cam.sInfo.strSensorName.decode('utf-8'),
                'serial': self.cam.cInfo.SerNo.decode('utf-8'),
                'clock speed': self.cam.pixel_clock.value,
                'trigger': self.camera_options.get_param_value(
                    ThorCamParams.TRIGGER_MODE),
                'flash mode': self.camera_options.get_param_value(
                    ThorCamParams.FLASH_MODE),
                'framerate': self.cam.currentFrameRate.value,
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

            QMessageBox.information(
                self, 'Info', 'Config saved.')
        else:
            QMessageBox.warning(
                self, 'Warning', 'Config not saved.')

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load config', filter='JSON Files (*.json);;')

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
                if self.cam.sInfo.strSensorName.decode('utf-8') == \
                        config['model']:
                    self.camera_options.set_roi_info(
                        int(config['ROI x']), int(config['ROI y']),
                        int(config['ROI w']), int(config['ROI h']))
                    self.set_ROI()

                    self.camera_options.set_param_value(
                        ThorCamParams.PIXEL_CLOCK,
                        str(config['clock speed']))
                    self.camera_options.set_param_value(
                        ThorCamParams.TRIGGER_MODE,
                        config['trigger'])
                    self.camera_options.set_param_value(
                        ThorCamParams.FLASH_DURATION,
                        config['flash mode'])

                    self.camera_options.set_param_value(
                        ThorCamParams.FRAMERATE,
                        config['framerate'])

                    self.camera_options.set_param_value(
                        CamParams.EXPOSURE,
                        config['exposure'])

                    self.camera_options.set_param_value(
                        ThorCamParams.FLASH_DURATION,
                        config['flash duration'])

                    self.camera_options.set_param_value(
                        ThorCamParams.FLASH_DELAY,
                        config['flash delay'])

                    self.camera_options.set_param_value(
                        CamParams.RESIZE_DISPLAY,
                        float(config['Zoom']))
                else:
                    QMessageBox.warning(
                        self, 'Warning', 'Camera model is different.')
            else:
                QMessageBox.warning(
                    self, 'Warning', 'Wrong or corrupted config file.')
        else:
            QMessageBox.warning(
                self, 'Warning', 'No file selected.')
