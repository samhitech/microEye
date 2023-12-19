import json
import os
import time
import traceback
from queue import Queue

import cv2
import numpy as np
import pyqtgraph as pg
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ...shared.metadata_tree import MetaParams
from ...shared.thread_worker import *
from ..widgets.qlist_slider import *
from . import Camera_Panel
from .thorlabs import *


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
        self.cam_pixel_clock_lbl = QLabel('Pixel Clock MHz')
        self.cam_pixel_clock_cbox = QComboBox()
        self.cam_pixel_clock_cbox.addItems(
            map(str, self._cam.pixel_clock_list[:]))
        self.cam_pixel_clock_cbox.currentIndexChanged[str] \
            .connect(self.cam_pixel_cbox_changed)

        # framerate slider control
        self.cam_framerate_lbl = DragLabel(
            'Framerate FPS', parent_name='cam_framerate_slider')
        self.cam_framerate_slider = QListSlider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_framerate_slider.values = np.arange(
            self._cam.minFrameRate.value, self._cam.maxFrameRate.value,
            self._cam.incFrameRate.value * 100)
        self.cam_framerate_slider.elementChanged[int, float] \
            .connect(self.cam_framerate_value_changed)

        # framerate text box control
        self.cam_framerate_ledit = QLineEdit(f'{self._cam.currentFrameRate.value:.6f}')
        self.cam_framerate_ledit.setCompleter(
            QCompleter(map('{:.6f}'.format, self.cam_framerate_slider.values)))
        self.cam_framerate_ledit.setValidator(QDoubleValidator())
        self.cam_framerate_ledit.returnPressed \
            .connect(self.cam_framerate_return)

        # exposure slider control
        self.cam_exposure_lbl = DragLabel(
            'Exposure ms', parent_name='cam_exposure_slider')
        self.cam_exposure_slider = QListSlider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_exposure_slider.values = np.arange(
            self._cam.exposure_range[0],
            self._cam.exposure_range[1],
            self._cam.exposure_range[2])
        self.cam_exposure_slider.elementChanged[int, float] \
            .connect(self.cam_exposure_value_changed)

        # exposure text box
        self.cam_exposure_qs.setMinimum(self._cam.exposure_range[0].value)
        self.cam_exposure_qs.setMaximum(self._cam.exposure_range[1].value)
        self.cam_exposure_qs.setSingleStep(self._cam.exposure_range[2].value)
        self.cam_exposure_qs.setValue(self._cam.exposure_current.value)
        self.cam_exposure_qs.valueChanged.connect(self.exposure_spin_changed)

        # trigger mode combobox
        self.cam_trigger_mode_lbl = QLabel('Trigger Mode')
        self.cam_trigger_mode_cbox = QComboBox()
        self.cam_trigger_mode_cbox.addItems(TRIGGER.TRIGGER_MODES.keys())
        self.cam_trigger_mode_cbox.currentIndexChanged[str] \
            .connect(self.cam_trigger_cbox_changed)

        # flash mode combobox
        self.cam_flash_mode_lbl = QLabel('Flash Mode')
        self.cam_flash_mode_cbox = QComboBox()
        self.cam_flash_mode_cbox.addItems(FLASH_MODE.FLASH_MODES.keys())
        self.cam_flash_mode_cbox.currentIndexChanged[str] \
            .connect(self.cam_flash_cbox_changed)

        # flash duration slider
        self.cam_flash_duration_lbl = QLabel('Flash Duration us')
        self.cam_flash_duration_slider = QListSlider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_flash_duration_slider.values = \
            np.append([0], np.arange(self._cam.flash_min.u32Duration,
                      self._cam.flash_max.u32Duration,
                      self._cam.flash_inc.u32Duration))
        self.cam_flash_duration_slider.elementChanged[int, int] \
            .connect(self.cam_flash_duration_value_changed)

        # flash duration text box
        self.cam_flash_duration_ledit = QLineEdit(
            f'{self._cam.flash_cur.u32Duration:d}')
        self.cam_flash_duration_ledit.setValidator(QIntValidator())
        self.cam_flash_duration_ledit.returnPressed \
            .connect(self.cam_flash_duration_return)

        # flash delay slider
        self.cam_flash_delay_lbl = QLabel('Flash Delay us')
        self.cam_flash_delay_slider = QListSlider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_flash_delay_slider.values = np.append([0], np.arange(
            self._cam.flash_min.s32Delay,
            self._cam.flash_max.s32Delay,
            self._cam.flash_inc.s32Delay))
        self.cam_flash_delay_slider.elementChanged[int, int] \
            .connect(self.cam_flash_delay_value_changed)

        # flash delay text box
        self.cam_flash_delay_ledit = QLineEdit(f'{self._cam.flash_cur.s32Delay:d}')
        self.cam_flash_delay_ledit.setValidator(QIntValidator())
        self.cam_flash_delay_ledit.returnPressed \
            .connect(self.cam_flash_delay_return)

        # setting the highest pixel clock as default
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[-1]))

        # start freerun mode button
        self.cam_freerun_btn = QPushButton(
            'Freerun Mode (Start)',
            clicked=lambda: self.start_free_run()
        )

        # start trigger mode button
        self.cam_trigger_btn = QPushButton(
            'Trigger Mode (Start)',
            clicked=lambda: self.start_software_triggered()
        )

        # config buttons
        self.cam_load_btn = QPushButton(
            'Load Config.',
            clicked=lambda: self.load_config()
        )
        self.cam_save_btn = QPushButton(
            'Save Config.',
            clicked=lambda: self.save_config()
        )
        self.config_Hlay = QHBoxLayout()
        self.config_Hlay.addWidget(self.cam_save_btn)
        self.config_Hlay.addWidget(self.cam_load_btn)

        # stop acquisition button
        self.cam_stop_btn = QPushButton(
            'Stop',
            clicked=lambda: self.stop()
        )

        # ROI
        self.ROI_x_tbox.setMinimum(0)
        self.ROI_x_tbox.setMaximum(self.cam.width)
        self.ROI_y_tbox.setMinimum(0)
        self.ROI_y_tbox.setMaximum(self.cam.height)
        self.ROI_width_tbox.setMinimum(self.cam.minROI.s32Width)
        self.ROI_width_tbox.setMaximum(self.cam.width)
        self.ROI_height_tbox.setMinimum(self.cam.minROI.s32Height)
        self.ROI_height_tbox.setMaximum(self.cam.height)

        # adding widgets to the main layout
        self.first_tab_Layout.addWidget(self.cam_pixel_clock_lbl)
        self.first_tab_Layout.addWidget(self.cam_pixel_clock_cbox)
        self.first_tab_Layout.addWidget(self.cam_trigger_mode_lbl)
        self.first_tab_Layout.addWidget(self.cam_trigger_mode_cbox)
        self.first_tab_Layout.addWidget(self.cam_framerate_lbl)
        self.first_tab_Layout.addWidget(self.cam_framerate_ledit)
        self.first_tab_Layout.addWidget(self.cam_framerate_slider)
        self.first_tab_Layout.addWidget(self.cam_exposure_lbl)
        self.first_tab_Layout.addWidget(self.cam_exposure_qs)
        self.first_tab_Layout.addWidget(self.cam_exposure_slider)
        self.first_tab_Layout.addRow(self.cam_exp_shortcuts)
        if not self.mini:
            self.first_tab_Layout.addWidget(self.cam_flash_mode_lbl)
            self.first_tab_Layout.addWidget(self.cam_flash_mode_cbox)
            self.first_tab_Layout.addWidget(self.cam_flash_duration_lbl)
            self.first_tab_Layout.addWidget(self.cam_flash_duration_ledit)
            self.first_tab_Layout.addWidget(self.cam_flash_duration_slider)
            self.first_tab_Layout.addWidget(self.cam_flash_delay_lbl)
            self.first_tab_Layout.addWidget(self.cam_flash_delay_ledit)
            self.first_tab_Layout.addWidget(self.cam_flash_delay_slider)
        self.first_tab_Layout.addLayout(self.config_Hlay)
        self.first_tab_Layout.addWidget(self.cam_freerun_btn)
        self.first_tab_Layout.addWidget(self.cam_trigger_btn)
        self.first_tab_Layout.addWidget(self.cam_stop_btn)

        self.addFirstTabItems()

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
            int(self.ROI_x_tbox.text()),
            int(self.ROI_y_tbox.text()),
            int(self.ROI_width_tbox.text()),
            int(self.ROI_height_tbox.text()))

        # setting the highest pixel clock as default
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[0]))
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[-1]))

    def reset_ROI(self):
        '''Resets the ROI for the slected thorlabs_camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, 'Warning', 'Cannot reset ROI while acquiring images!')
            return  # if acquisition is already going on

        self.cam.reset_ROI()
        self.ROI_x_tbox.setText('0')
        self.ROI_y_tbox.setText('0')
        self.ROI_width_tbox.setText(str(self.cam.width))
        self.ROI_height_tbox.setText(str(self.cam.height))

        # setting the highest pixel clock as default
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[0]))
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[-1]))

    def center_ROI(self):
        '''Calculates the x, y values for a centered ROI'''
        self.ROI_x_tbox.setText(
            str(
                int(
                    (self.cam.rectROI.s32Width -
                     int(self.ROI_width_tbox.text()))/2)))
        self.ROI_y_tbox.setText(
            str(
                int(
                    (self.cam.rectROI.s32Height -
                     int(self.ROI_height_tbox.text()))/2)))

    def select_ROI(self):
        if self.acq_job.frame is not None:
            roi = cv2.selectROI(self.acq_job.frame._view)
            cv2.destroyWindow('ROI selector')

            z = self.zoom_box.value()
            self.ROI_x_tbox.setValue(int(roi[0] / z))
            self.ROI_y_tbox.setValue(int(roi[1] / z))
            self.ROI_width_tbox.setValue(int(roi[2] / z))
            self.ROI_height_tbox.setValue(int(roi[3] / z))

    @pyqtSlot(str)
    def cam_trigger_cbox_changed(self, value):
        '''
        Slot for changed trigger mode

        Parameters
        ----------
        Value : int
            index of selected trigger mode from _TRIGGER.TRIGGER_MODES
        '''
        self._cam.set_trigger_mode(TRIGGER.TRIGGER_MODES[value])
        self._cam.get_trigger_mode()

    @pyqtSlot(str)
    def cam_flash_cbox_changed(self, value):
        '''
        Slot for changed flash mode

        Parameters
        ----------
        Value : int
            index of selected flash mode from _TRIGGER.FLASH_MODES
        '''
        self._cam.set_flash_mode(FLASH_MODE.FLASH_MODES[value])
        self._cam.get_flash_mode(output=True)

    @pyqtSlot(str)
    def cam_pixel_cbox_changed(self, value):
        '''
        Slot for changed pixel clock

        Parameters
        ----------
        Value : int
            selected pixel clock in MHz
            (note that only certain values are allowed by camera)
        '''
        self._cam.set_pixel_clock(int(value))
        self._cam.get_pixel_clock_info(False)
        self._cam.get_framerate_range(False)

        self.refresh_framerate(True)
        self.cam_framerate_slider.setValue(
            len(self.cam_framerate_slider.values) - 1)

    @pyqtSlot(int, float)
    def cam_framerate_value_changed(self, index, value):
        '''
        Slot for changed framerate

        Parameters
        ----------
        Index : ont
            selected frames per second index in the slider values list
        Value : double
            selected frames per second
        '''
        self._cam.set_framerate(value)
        self._cam.get_exposure_range(False)
        self.cam_framerate_ledit.setText(f'{self._cam.currentFrameRate.value:.6f}')
        self.refresh_exposure(True)
        self.cam_exposure_slider.setValue(
            len(self.cam_exposure_slider.values) - 1)

    def cam_framerate_return(self):
        '''
        Sets the framerate slider with the entered text box value
        '''
        self.cam_framerate_slider.setNearest(self.cam_framerate_ledit.text())

    @pyqtSlot(int, float)
    def cam_exposure_value_changed(self, index, value):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Index : ont
            selected exposure index in the slider values list
        Value : double
            selected exposure in milli-seconds
        '''
        self._cam.set_exposure(value)
        self._cam.get_exposure(False)
        self._cam.get_flash_range(False)

        self.refresh_exposure()
        self.refresh_flash()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, self._cam.exposure_current)
        if self.master:
            self.exposureChanged.emit()

    @pyqtSlot(float)
    def exposure_spin_changed(self, value: float):
        '''
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in micro-seconds
        '''
        self._cam.set_exposure(value)
        self._cam.get_exposure(False)
        self._cam.get_flash_range(False)

        self.refresh_exposure()
        self.refresh_flash()

        self.OME_tab.set_param_value(
            MetaParams.EXPOSURE, self._cam.exposure_current)
        if self.master:
            self.exposureChanged.emit()

    def refresh_framerate(self, range=False):
        self.cam_framerate_slider.blockSignals(True)
        if range:
            self.cam_framerate_slider.values = np.arange(
                self._cam.minFrameRate.value,
                self._cam.maxFrameRate.value,
                self._cam.incFrameRate.value * 100)
        self.cam_framerate_slider.setNearest(self._cam.currentFrameRate.value)
        self.cam_framerate_slider.blockSignals(False)

    def refresh_exposure(self, range=False):
        self.cam_exposure_slider.blockSignals(True)
        if range:
            self.cam_exposure_slider.values = np.arange(
                self._cam.exposure_range[0].value,
                self._cam.exposure_range[1].value,
                self._cam.exposure_range[2].value)
        self.cam_exposure_slider.setNearest(self._cam.exposure_current.value)
        self.cam_exposure_slider.blockSignals(False)

        self.cam_exposure_qs.blockSignals(True)
        if range:
            self.cam_exposure_qs.setMinimum(self._cam.exposure_range[0].value)
            self.cam_exposure_qs.setMaximum(self._cam.exposure_range[1].value)
            self.cam_exposure_qs.setSingleStep(
                self._cam.exposure_range[2].value)
        self.cam_exposure_qs.setValue(self._cam.exposure_current.value)
        self.cam_exposure_qs.blockSignals(False)

    def refresh_flash(self):
        self.cam_flash_duration_slider.values = np.append([0], np.arange(
            self._cam.flash_min.u32Duration,
            self._cam.flash_max.u32Duration,
            self._cam.flash_inc.u32Duration))
        self.cam_flash_delay_slider.values = np.append([0], np.arange(
            self._cam.flash_min.s32Delay,
            self._cam.flash_max.s32Delay,
            self._cam.flash_inc.s32Delay))

    @pyqtSlot(int, int)
    def cam_flash_duration_value_changed(self, index, value):
        '''
        Slot for changed flash duration

        Parameters
        ----------
        Index : ont
            selected flash duration index in the slider values list
        Value : double
            selected flash duration in micro-seconds
        '''
        self._cam.set_flash_params(self._cam.flash_cur.s32Delay, value)
        self._cam.get_flash_params()
        self.cam_flash_duration_ledit.setText(f'{self._cam.flash_cur.u32Duration:d}')

    def cam_flash_duration_return(self):
        '''
        Sets the flash duration slider with the entered text box value
        '''
        self.cam_flash_duration_slider.setNearest(
            self.cam_flash_duration_ledit.text())

    @pyqtSlot(int, int)
    def cam_flash_delay_value_changed(self, index, value):
        '''
        Slot for changed flash delay

        Parameters
        ----------
        Index : ont
            selected flash delay index in the slider values list
        Value : double
            selected flash delay in micro-seconds
        '''
        self._cam.set_flash_params(
            value, self._cam.flash_cur.u32Duration)
        self._cam.get_flash_params()
        self.cam_flash_delay_ledit.setText(f'{self._cam.flash_cur.s32Delay:d}')

    def cam_flash_delay_return(self):
        '''
        Sets the flash delay slider with the entered text box value
        '''
        self.cam_flash_delay_slider.setNearest(
            self.cam_flash_delay_ledit.text())

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
            'Frames': self.frames_tbox.value(),
            'model': self.cam.sInfo.strSensorName.decode('utf-8'),
            'serial': self.cam.cInfo.SerNo.decode('utf-8'),
            'clock speed': self.cam.pixel_clock.value,
            'trigger': self.cam_trigger_mode_cbox.currentText(),
            'flash mode': self.cam_flash_mode_cbox.currentText(),
            'framerate': self.cam.currentFrameRate.value,
            'exposure': self.cam.exposure_current.value,
            'flash duration': self.cam.flash_cur.u32Duration,
            'flash delay': self.cam.flash_cur.s32Delay,
            'ROI w': self.cam.set_rectROI.s32Width,
            'ROI h': self.cam.set_rectROI.s32Height,
            'ROI x': self.cam.set_rectROI.s32X,
            'ROI y': self.cam.set_rectROI.s32Y,
            'Zoom': self.zoom_box.value(),
        }
        return meta

    @staticmethod
    def find_nearest(array, value):
        '''
        find nearest value in array to the supplied one

        Parameters
        ----------
        array : ndarray
            numpy array searching in
        value : type of ndarray.dtype
            value to find nearest to in the array
        '''
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save config', filter='JSON Files (*.json);;')

        if len(filename) > 0:

            config = {
                'model': self.cam.sInfo.strSensorName.decode('utf-8'),
                'serial': self.cam.cInfo.SerNo.decode('utf-8'),
                'clock speed': self.cam.pixel_clock.value,
                'trigger': self.cam_trigger_mode_cbox.currentText(),
                'flash mode': self.cam_flash_mode_cbox.currentText(),
                'framerate': self.cam.currentFrameRate.value,
                'exposure': self.cam.exposure_current.value,
                'flash duration': self.cam.flash_cur.u32Duration,
                'flash delay': self.cam.flash_cur.s32Delay,
                'ROI w': self.cam.set_rectROI.s32Width,
                'ROI h': self.cam.set_rectROI.s32Height,
                'ROI x': self.cam.set_rectROI.s32X,
                'ROI y': self.cam.set_rectROI.s32Y,
                'Zoom': self.zoom_box.value(),
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
                    self.ROI_x_tbox.setText(str(config['ROI x']))
                    self.ROI_y_tbox.setText(str(config['ROI y']))
                    self.ROI_width_tbox.setText(str(config['ROI w']))
                    self.ROI_height_tbox.setText(str(config['ROI h']))
                    self.set_ROI()

                    self.cam_pixel_clock_cbox.setCurrentText(
                        str(config['clock speed']))
                    self.cam_trigger_mode_cbox.setCurrentText(
                        config['trigger'])
                    self.cam_flash_mode_cbox.setCurrentText(
                        config['flash mode'])

                    self.cam_framerate_slider.setNearest(config['framerate'])

                    self.cam_exposure_slider.setNearest(config['exposure'])

                    self.cam_flash_duration_slider.setNearest(
                        config['flash duration'])

                    self.cam_flash_delay_slider.setNearest(
                        config['flash delay'])

                    self.zoom_box.setValue(float(config['Zoom']))
                else:
                    QMessageBox.warning(
                        self, 'Warning', 'Camera model is different.')
            else:
                QMessageBox.warning(
                    self, 'Warning', 'Wrong or corrupted config file.')
        else:
            QMessageBox.warning(
                self, 'Warning', 'No file selected.')
