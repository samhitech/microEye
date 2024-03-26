import json
import logging
import time
import traceback
from enum import Enum
from typing import overload

import cv2
import numpy as np
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
from .camera_options import CamParams
from .camera_panel import Camera_Panel
from .micam import miDummy

try:
    import vimba as vb
except Exception:
    vb = None

class DummyParams(Enum):
    FREERUN = 'Camera.Freerun'
    LOAD = 'Camera.Load Config'
    SAVE = 'Camera.Save Config'
    STOP = 'Camera.Stop'
    GAIN = 'Camera.Gain'
    OFFSET = 'Camera.Offset'
    HEIGHT = 'Camera.Height'
    WIDTH = 'Camera.Width'
    BINNING_HORIZONTAL = 'Camera.Binning Horizontal'
    BINNING_VERTICAL = 'Camera.Binning Vertical'
    BIT_DEPTH = 'Camera.Bit Depth'
    FULL_WELL_CAPACITY = 'Camera.Full Well Capacity'
    QUANTUM_EFFICIENCY = 'Camera.Quantum Efficiency'
    DARK_CURRENT = 'Camera.Dark Current'
    READOUT_NOISE = 'Camera.Readout Noise'
    NOISE_BASELINE = 'Camera.Noise Baseline'
    FLUX = 'Camera.Flux'
    PATTERN_TYPE = 'Camera.Pattern Type'
    PATTERN_OFFSET = 'Camera.Pattern Offset'
    PATTERN_SINUSOIDAL = 'Camera.Sinusoidal Pattern'
    SINUSOIDAL_FREQUENCY = 'Camera.Sinusoidal Pattern.Frequency'
    SINUSOIDAL_PHASE = 'Camera.Sinusoidal Pattern.Phase'
    SINUSOIDAL_AMPLITUDE = 'Camera.Sinusoidal Pattern.Amplitude'
    SINUSOIDAL_DIRECTION = 'Camera.Sinusoidal Pattern.Direction'

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

class Dummy_Panel(Camera_Panel):
    '''
    A Qt Widget for a dummy camera
     | Inherits Camera_Panel
    '''

    def __init__(self, mini=False,
                 *args, **kwargs):
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
        super().__init__(
            miDummy(), mini,
            *args, **kwargs)

        self.OME_tab.set_param_value(MetaParams.CHANNEL_NAME, self._cam.name)
        self.OME_tab.set_param_value(MetaParams.DET_MANUFACTURER, 'MicroEye')
        self.OME_tab.set_param_value(MetaParams.DET_MODEL, 'Dummy')
        self.OME_tab.set_param_value(MetaParams.DET_SERIAL, '123456789')
        self.OME_tab.set_param_value(MetaParams.DET_TYPE, 'CMOS')

        # exposure init
        exposure = self.camera_options.get_param(CamParams.EXPOSURE)
        exposure.setLimits(
            (self._cam.exposure_range[0], self._cam.exposure_range[1]))
        exposure.setOpts(
            step=self._cam.exposure_increment,
            suffix=self._cam.exposure_unit)
        exposure.setValue(self._cam.exposure_current)
        exposure.sigValueChanged.connect(self.exposure_spin_changed)

        HEIGHT = {
            'name': str(DummyParams.HEIGHT),
            'type': 'int', 'value': 512, 'limits': [0, 4096]}
        WIDTH = {
            'name': str(DummyParams.WIDTH),
            'type': 'int', 'value': 512, 'limits': [0, 4096]}
        BINNING_HORIZONTAL = {
            'name': str(DummyParams.BINNING_HORIZONTAL),
            'type': 'int', 'value': 1, 'limits': [0, 10]}
        BINNING_VERTICAL = {
            'name': str(DummyParams.BINNING_VERTICAL),
            'type': 'int', 'value': 1, 'limits': [0, 10]}
        BIT_DEPTH = {
            'name': str(DummyParams.BIT_DEPTH),
            'type': 'list', 'values': [8, 10, 12, 16], 'value': 12}
        GAIN = {
            'name': str(DummyParams.GAIN),
            'type': 'float', 'value': 2.23}
        FULL_WELL_CAPACITY = {
            'name': str(DummyParams.FULL_WELL_CAPACITY),
            'type': 'int', 'value': 9200, 'suffix': 'e-'}
        QUANTUM_EFFICIENCY = {
            'name': str(DummyParams.QUANTUM_EFFICIENCY),
            'type': 'float', 'value': 0.8, 'limits': [0, 1]}
        DARK_CURRENT = {
            'name': str(DummyParams.DARK_CURRENT),
            'type': 'float', 'value': 0.0001, 'suffix': ' e-/s'}
        READOUT_NOISE = {
            'name': str(DummyParams.READOUT_NOISE),
            'type': 'float', 'value': 2.1, 'suffix': ' e-'}
        NOISE_BASELINE = {
            'name': str(DummyParams.NOISE_BASELINE),
            'type': 'float', 'value': 5.0, 'limits': [0, 2**16], 'suffix': ' ADU'}
        FLUX = {
            'name': str(DummyParams.FLUX),
            'type': 'float', 'value': 0, 'suffix': ' e-/p/s'}
        PATTERN_TYPE = {
            'name': str(DummyParams.PATTERN_TYPE),
            'type': 'list',
            'values': ['Constant Flux', 'Sinusoidal'],
            'value': 'Sinusoidal'}
        PATTERN_OFFSET = {
            'name': str(DummyParams.PATTERN_OFFSET),
            'type': 'float', 'value': 0.0, 'suffix': ' e-'}
        PATTERN_SINUSOIDAL = {
            'name': str(DummyParams.PATTERN_SINUSOIDAL), 'type': 'group', 'children': [
                {'name': str(DummyParams.SINUSOIDAL_FREQUENCY),
                 'type': 'float', 'value': 0.01, 'suffix': ' Hz'},
                {'name': str(DummyParams.SINUSOIDAL_PHASE), 'type': 'float',
                 'value': 0.0, 'suffix': ' deg'},
                {'name': str(DummyParams.SINUSOIDAL_AMPLITUDE),
                 'type': 'float', 'value': 0.1e5, 'suffix': ' e-/p/s'},
                {'name': str(DummyParams.SINUSOIDAL_DIRECTION), 'type': 'list',
                 'values': ['d', 'h', 'v', 'r'], 'value': 'd'}
            ]}

        # add parameters
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, HEIGHT)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, WIDTH)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, BINNING_HORIZONTAL)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, BINNING_VERTICAL)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, BIT_DEPTH)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, GAIN)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, FULL_WELL_CAPACITY)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, QUANTUM_EFFICIENCY)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, DARK_CURRENT)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, READOUT_NOISE)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, NOISE_BASELINE)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, FLUX)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, PATTERN_TYPE)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, PATTERN_OFFSET)
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, PATTERN_SINUSOIDAL)

        # update params
        self.camera_options.get_param(
            DummyParams.HEIGHT).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(DummyParams.HEIGHT, new_value))
        self.camera_options.get_param(
            DummyParams.WIDTH).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(DummyParams.WIDTH, new_value))
        self.camera_options.get_param(
            DummyParams.BINNING_HORIZONTAL).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.BINNING_HORIZONTAL, new_value))
        self.camera_options.get_param(
            DummyParams.BINNING_VERTICAL).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.BINNING_VERTICAL, new_value))
        self.camera_options.get_param(
            DummyParams.BIT_DEPTH).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(DummyParams.BIT_DEPTH, new_value))
        self.camera_options.get_param(
            DummyParams.GAIN).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(DummyParams.GAIN, new_value))
        self.camera_options.get_param(
            DummyParams.FULL_WELL_CAPACITY).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.FULL_WELL_CAPACITY, new_value))
        self.camera_options.get_param(
            DummyParams.QUANTUM_EFFICIENCY).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.QUANTUM_EFFICIENCY, new_value))
        self.camera_options.get_param(
            DummyParams.DARK_CURRENT).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.DARK_CURRENT, new_value))
        self.camera_options.get_param(
            DummyParams.READOUT_NOISE).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.READOUT_NOISE, new_value))
        self.camera_options.get_param(
            DummyParams.NOISE_BASELINE).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.NOISE_BASELINE, new_value))
        self.camera_options.get_param(
            DummyParams.FLUX).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(DummyParams.FLUX, new_value))
        self.camera_options.get_param(
            DummyParams.PATTERN_TYPE).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.PATTERN_TYPE, new_value))
        self.camera_options.get_param(
            DummyParams.PATTERN_OFFSET).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.PATTERN_OFFSET, new_value))
        self.camera_options.get_param(
            DummyParams.SINUSOIDAL_AMPLITUDE).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.SINUSOIDAL_AMPLITUDE, new_value))
        self.camera_options.get_param(
            DummyParams.SINUSOIDAL_FREQUENCY).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.SINUSOIDAL_FREQUENCY, new_value))
        self.camera_options.get_param(
            DummyParams.SINUSOIDAL_PHASE).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.SINUSOIDAL_PHASE, new_value))
        self.camera_options.get_param(
            DummyParams.SINUSOIDAL_DIRECTION).sigValueChanged.connect(
                lambda _, new_value: self.update_cam(
                    DummyParams.SINUSOIDAL_DIRECTION, new_value))

        # start freerun mode button
        freerun = {'name': str(DummyParams.FREERUN), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, freerun)
        self.camera_options.get_param(
            DummyParams.FREERUN).sigActivated.connect(
                self.start_free_run)

        # stop acquisition button
        stop = {'name': str(DummyParams.STOP), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, stop)
        self.camera_options.get_param(
            DummyParams.STOP).sigActivated.connect(
                self.stop)

        # config buttons
        load = {'name': str(DummyParams.LOAD), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, load)
        self.camera_options.get_param(
            DummyParams.LOAD).sigActivated.connect(
                self.load_config)

        save = {'name': str(DummyParams.SAVE), 'type': 'action'}
        self.camera_options.add_param_child(
            CamParams.CAMERA_OPTIONS, save)
        self.camera_options.get_param(
            DummyParams.SAVE).sigActivated.connect(
                self.save_config)

        # ROI
        self.camera_options.set_roi_limits(
            (0, self.cam.getWidth()),
            (0, self.cam.getHeight()),
            (32, self.cam.getWidth()),
            (32, self.cam.getHeight()))

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

    def update_cam(self, param_name, param_value):
        if param_name == DummyParams.HEIGHT:
            self._cam.height = param_value
        elif param_name ==  DummyParams.WIDTH:
            self._cam.width = param_value
        elif param_name ==  DummyParams.BINNING_HORIZONTAL:
            self._cam.binning_horizontal = param_value
        elif param_name ==  DummyParams.BINNING_VERTICAL:
            self._cam.binning_vertical = param_value
        elif param_name ==  DummyParams.BIT_DEPTH:
            self._cam.bit_depth = param_value
        elif param_name ==  DummyParams.GAIN:
            self._cam.gain = param_value
        elif param_name ==  DummyParams.FULL_WELL_CAPACITY:
            self._cam.full_well_capacity = param_value
        elif param_name ==  DummyParams.QUANTUM_EFFICIENCY:
            self._cam.quantum_efficiency = param_value
        elif param_name ==  DummyParams.DARK_CURRENT:
            self._cam.dark_current = param_value
        elif param_name ==  DummyParams.READOUT_NOISE:
            self._cam.readout_noise = param_value
        elif param_name ==  DummyParams.NOISE_BASELINE:
            self._cam.noise_baseline = param_value
        elif param_name ==  DummyParams.FLUX:
            self._cam.flux = param_value
        elif param_name ==  DummyParams.PATTERN_TYPE:
            self._cam.pattern_type = param_value
        elif param_name ==  DummyParams.PATTERN_OFFSET:
            self._cam.pattern_offset = param_value
        elif param_name == DummyParams.SINUSOIDAL_AMPLITUDE:
            self._cam.sinusoidal_amplitude = param_value
        elif param_name == DummyParams.SINUSOIDAL_FREQUENCY:
            self._cam.sinusoidal_frequency = param_value
        elif param_name == DummyParams.SINUSOIDAL_PHASE:
            self._cam.sinusoidal_phase = param_value
        elif param_name == DummyParams.SINUSOIDAL_DIRECTION:
            self._cam.sinusoidal_direction = param_value

    def set_ROI(self):
        '''Sets the ROI for the slected miDummy
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!')
            return  # if acquisition is already going on

        self.cam.set_roi(
            *self.camera_options.get_roi_info(False))

    def reset_ROI(self):
        '''Resets the ROI for the slected miDummy
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!')
            return  # if acquisition is already going on

        self.cam.reset_roi()

        self.camera_options.set_roi_info(
            0, 0, int(self.cam.width), int(self.cam.height))

    def center_ROI(self):
        '''Sets the ROI for the slected miDummy
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, 'Warning', 'Cannot set ROI while acquiring images!')
            return  # if acquisition is already going on

        x, y, width, height = self.camera_options.get_roi_info(False)
        x = (int(self.cam.width)-width)//2
        y = (int(self.cam.height)-height)//2

        self.cam.set_roi(x, y, width, height)

        self.camera_options.set_roi_info(x, y, width, height)

    def select_ROI(self):
        if self.acq_job is not None:
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
            MetaParams.EXPOSURE, float(self._cam.exposure_current))

    def refresh_exposure(self):
        self.camera_options.set_param_value(
            CamParams.EXPOSURE,
            self._cam.exposure_current,
            self.exposure_spin_changed)

    def cam_capture(self, *args):
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
            while self.acq_job.frames_captured < self.acq_job.frames or \
                self.mini:
                cycle_time = time.perf_counter_ns()

                self._buffer.put(
                    self._cam.get_dummy_image_from_pattern(
                        self.acq_job.frames_captured / 100).tobytes())
                # add sensor temperature to the stack
                self._temps.put(self.cam.get_temperature())
                self.acq_job.frames_captured += 1

                diff = int(self._cam.exposure * 1000) - (
                    time.perf_counter_ns() - cycle_time) // 1000
                if diff > 0:
                    QThread.usleep(diff)
                else:
                    QThread.usleep(100)  # sleep 100us

                if self.acq_job.stop_threads:
                    break  # in case stop threads is initiated

            self.acq_job.stop_threads = True
            logging.debug('Stop')
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            self._cam.acquisition = False
            QThreadPool.globalInstance().releaseThread()
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

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save config', filter='XML Files (*.xml);;')

        if len(filename) > 0:

            with self.cam.cam:
                self.cam.cam.save_settings(filename, vb.PersistType.All)

            QMessageBox.information(
                self, 'Info', 'Config saved.')
        else:
            QMessageBox.warning(
                self, 'Warning', 'Config not saved.')

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load config', filter='XML Files (*.xml);;')

        if len(filename) > 0:
            with self.cam.cam:
                self.cam.default()

                # Load camera settings from file.
                self.cam.cam.load_settings(filename, vb.PersistType.All)

                w, h, x, y = self.cam.get_roi()

                self.camera_options.set_roi_info(
                    int(x), int(y), int(w), int(h))

                self.cam_trigger_mode_cbox.setCurrentText(
                    self.cam.get_trigger_mode())

                self.camera_options.set_param_value(
                    CamParams.EXPOSURE, self.cam.get_exposure())
        else:
            QMessageBox.warning(
                self, 'Warning', 'No file selected.')
