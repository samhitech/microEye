import json
import logging
import traceback

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ...qlist_slider import *
from . import Camera_Panel, vimba_cam

try:
    import vimba as vb
except Exception:
    vb = None


class Vimba_Panel(Camera_Panel):
    """
    A Qt Widget for controlling an Allied Vision Camera through Vimba
     | Inherits Camera_Panel
    """

    def __init__(self, threadpool: QThreadPool, cam: vimba_cam, mini=False,
                 *args, **kwargs):
        """
        Initializes a new Vimba_Panel Qt widget
        | Inherits Camera_Panel

        Parameters
        ----------
        threadpool : QThreadPool
            The threadpool for multithreading
        cam : vimba_cam
            Vimba Camera python adapter
        """
        super().__init__(
            threadpool, cam, mini,
            *args, **kwargs)

        self.OME_tab.channel_name.setText(self._cam.name)
        self.OME_tab.det_manufacturer.setText('Allied Vision')
        self.OME_tab.det_model.setText(
            self._cam.cam.get_model())
        self.OME_tab.det_serial.setText(
            self._cam.cam.get_serial())

        # exposure slider control
        self.cam_exposure_lbl = DragLabel(
            "Exposure " + self.cam.exposure_unit,
            parent_name='cam_exposure_slider')
        self.cam_exposure_slider = qlist_slider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_exposure_slider.values = np.arange(
            self._cam.exposure_range[0],
            min(self._cam.exposure_range[1], 2e+6),
            self._cam.exposure_increment)
        self.cam_exposure_slider.elementChanged[int, float] \
            .connect(self.cam_exposure_value_changed)

        # exposure text box control
        self.cam_exposure_qs.setMinimum(self._cam.exposure_range[0])
        self.cam_exposure_qs.setMaximum(self._cam.exposure_range[1])
        self.cam_exposure_qs.setSingleStep(self._cam.exposure_increment)
        self.cam_exposure_qs.setValue(self._cam.exposure_current)
        self.cam_exposure_qs.setSuffix(self._cam.exposure_unit)
        self.cam_exposure_qs.valueChanged.connect(self.exposure_spin_changed)

        # exposure mode combobox
        self.cam_exposure_mode_lbl = QLabel("Exposure Mode")
        self.cam_exposure_mode_cbox = QComboBox()
        self.cam_exposure_mode_cbox.addItems(self._cam.exposure_modes[0])
        for x in range(len(self._cam.exposure_modes[2])):
            self.cam_exposure_mode_cbox.setItemData(
                x, self._cam.exposure_modes[2][x], Qt.ToolTipRole)
        self.cam_exposure_mode_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # exposure auto mode combobox
        self.cam_exposure_auto_lbl = QLabel("Exposure Auto Mode")
        self.cam_exposure_auto_cbox = QComboBox()
        self.cam_exposure_auto_cbox.addItems(
            self._cam.exposure_auto_entries[0])
        for x in range(len(self._cam.exposure_auto_entries[2])):
            self.cam_exposure_auto_cbox.setItemData(
                x, self._cam.exposure_auto_entries[2][x], Qt.ToolTipRole)
        self.cam_exposure_auto_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # Frame Rate
        self.cam_framerate_enabled = QCheckBox('Frame Rate')
        self.cam_framerate_enabled.setChecked(False)
        self.cam_framerate_enabled.stateChanged.connect(
            self.cam_framerate_changed)

        self.cam_framerate_qs = QDoubleSpinBox()
        self.cam_framerate_qs.setMinimum(self._cam.frameRate_range[0])
        self.cam_framerate_qs.setMaximum(self._cam.frameRate_range[1])
        self.cam_framerate_qs.setSingleStep(0.1)
        self.cam_framerate_qs.setDecimals(5)
        self.cam_framerate_qs.setValue(self._cam.frameRate)
        self.cam_framerate_qs.setSuffix(self._cam.frameRate_unit)
        self.cam_framerate_qs.valueChanged.connect(self.framerate_spin_changed)
        self.cam_framerate_qs.setEnabled(False)

        # trigger mode combobox
        self.cam_trigger_mode_lbl = QLabel("Trigger Mode")
        self.cam_trigger_mode_cbox = QComboBox()
        self.cam_trigger_mode_cbox.addItems(self._cam.trigger_modes[0])
        for x in range(len(self._cam.trigger_modes[2])):
            self.cam_trigger_mode_cbox.setItemData(
                x, self._cam.trigger_modes[2][x], Qt.ToolTipRole)
        self.cam_trigger_mode_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # trigger source combobox
        self.cam_trigger_source_lbl = QLabel("Trigger Source")
        self.cam_trigger_source_cbox = QComboBox()
        self.cam_trigger_source_cbox.addItems(self._cam.trigger_sources[0])
        for x in range(len(self._cam.trigger_sources[2])):
            self.cam_trigger_source_cbox.setItemData(
                x, self._cam.trigger_sources[2][x], Qt.ToolTipRole)
        self.cam_trigger_source_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # trigger selector combobox
        self.cam_trigger_selector_lbl = QLabel("Trigger Selector")
        self.cam_trigger_selector_cbox = QComboBox()
        self.cam_trigger_selector_cbox.addItems(self._cam.trigger_selectors[0])
        for x in range(len(self._cam.trigger_selectors[2])):
            self.cam_trigger_selector_cbox.setItemData(
                x, self._cam.trigger_selectors[2][x], Qt.ToolTipRole)
        self.cam_trigger_selector_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # trigger activation combobox
        self.cam_trigger_activation_lbl = QLabel("Trigger Activation")
        self.cam_trigger_activation_cbox = QComboBox()
        self.cam_trigger_activation_cbox.addItems(
            self._cam.trigger_activations[0])
        for x in range(len(self._cam.trigger_activations[2])):
            self.cam_trigger_activation_cbox.setItemData(
                x, self._cam.trigger_activations[2][x], Qt.ToolTipRole)
        self.cam_trigger_activation_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # pixel formats combobox
        self.cam_pixel_format_lbl = QLabel("Pixel Format")
        self.cam_pixel_format_cbox = QComboBox()
        self.cam_pixel_format_cbox.addItems(
            self._cam.pixel_formats)
        self.cam_pixel_format_cbox.currentIndexChanged[str] \
            .connect(self.cam_cbox_changed)

        # start freerun mode button
        self.cam_freerun_btn = QPushButton(
            "Freerun Mode (Start)",
            clicked=lambda: self.start_free_run()
        )

        # config buttons
        self.cam_load_btn = QPushButton(
            "Load Config.",
            clicked=lambda: self.load_config()
        )
        self.cam_save_btn = QPushButton(
            "Save Config.",
            clicked=lambda: self.save_config()
        )
        self.config_Hlay = QHBoxLayout()
        self.config_Hlay.addWidget(self.cam_save_btn)
        self.config_Hlay.addWidget(self.cam_load_btn)

        # stop acquisition button
        self.cam_stop_btn = QPushButton(
            "Stop",
            clicked=lambda: self.stop()
        )

        # AOI        
        self.AOI_x_tbox.setMinimum(0)
        self.AOI_x_tbox.setMaximum(self.cam.width_range[1])
        self.AOI_y_tbox.setMinimum(0)
        self.AOI_y_tbox.setMaximum(self.cam.height_range[1])
        self.AOI_width_tbox.setMinimum(self.cam.width_range[0])
        self.AOI_width_tbox.setMaximum(self.cam.width_range[1])
        self.AOI_height_tbox.setMinimum(self.cam.height_range[0])
        self.AOI_height_tbox.setMaximum(self.cam.height_range[1])

        # GPIOs
        self.lineSelector = QComboBox()
        self.lineMode = QComboBox()
        self.lineSource = QComboBox()
        self.lineInverter = QCheckBox('Inverter')
        self.lineInverter.setChecked(False)

        with self._cam.cam:
            self.lineSelector.addItems(self.cam.get_io_lines())
            self.lineMode.addItems(self.cam.get_line_modes())
            self.lineSource.addItems(self.cam.get_line_sources())

        self.lineSelector.currentIndexChanged.connect(self.io_line_changed)

        self.set_io_config = QPushButton(
            'Set Config.',
            clicked=lambda: self.set_io_line_config())

        # Timers
        self.timerSelector = QComboBox()
        self.timerActivation = QComboBox()
        self.timerSource = QComboBox()
        self.timerDelay = QDoubleSpinBox()
        self.timerDuration = QDoubleSpinBox()
        self.timerReset = QPushButton('Timer Reset')

        with self._cam.cam:
            timers = self.cam.get_timers()
            if timers:
                self.timerSelector.addItems(
                    timers)
                self.timerActivation.addItems(
                    self.cam.get_timer_trigger_activations())
                self.timerSource.addItems(
                    self.cam.get_timer_trigger_sources())

        def reset_timer():
            with self._cam.cam:
                self.cam.reset_timer()

        def update_timer():
            with self._cam.cam:
                timer = self.timerSelector.currentText()
                self.cam.select_timer(timer)
                delay = self.cam.get_timer_delay()
                duration = self.cam.get_timer_duration()
                self.timerDelay.setMinimum(delay[1][0])
                self.timerDelay.setMaximum(delay[1][1])
                self.timerDelay.setValue(delay[0])
                self.timerDuration.setMinimum(duration[1][0])
                self.timerDuration.setMaximum(duration[1][1])
                self.timerDuration.setValue(duration[0])

                self.timerActivation.setCurrentText(
                    self.cam.get_timer_trigger_activation())
                self.timerSource.setCurrentText(
                    self.cam.get_timer_trigger_activation())

        def set_timer():
            with self._cam.cam:
                timer = self.timerSelector.currentText()
                self.cam.select_timer(timer)
                act = self.timerActivation.currentText()
                source = self.timerSource.currentText()
                self.cam.set_timer_trigger_activation(act)
                self.cam.set_timer_trigger_source(source)
                self.cam.set_timer_duration(
                    self.timerDuration.value())
                self.cam.set_timer_delay(
                    self.timerDelay.value())

        self.timerSelector.currentIndexChanged.connect(update_timer)
        self.timerReset.clicked.connect(reset_timer)

        self.set_timer_config = QPushButton(
            'Set Timer Config.',
            clicked=set_timer)

        # adding widgets to the main layout
        self.first_tab_Layout.addRow(
            self.cam_trigger_mode_lbl,
            self.cam_trigger_mode_cbox)
        self.first_tab_Layout.addRow(
            self.cam_trigger_source_lbl,
            self.cam_trigger_source_cbox)
        self.first_tab_Layout.addRow(
            self.cam_trigger_selector_lbl,
            self.cam_trigger_selector_cbox)
        self.first_tab_Layout.addRow(
            self.cam_trigger_activation_lbl,
            self.cam_trigger_activation_cbox)

        self.first_tab_Layout.addRow(
            self.cam_exposure_lbl,
            self.cam_exposure_qs)
        self.first_tab_Layout.addRow(self.cam_exposure_slider)
        self.first_tab_Layout.addRow(self.cam_exp_shortcuts)
        self.first_tab_Layout.addRow(
            self.cam_exposure_mode_lbl,
            self.cam_exposure_mode_cbox)
        self.first_tab_Layout.addRow(
            self.cam_exposure_auto_lbl,
            self.cam_exposure_auto_cbox)
        self.first_tab_Layout.addRow(
            self.cam_pixel_format_lbl,
            self.cam_pixel_format_cbox)

        self.first_tab_Layout.addRow(
            self.cam_framerate_enabled,
            self.cam_framerate_qs)

        self.first_tab_Layout.addRow(self.config_Hlay)
        self.first_tab_Layout.addRow(self.cam_freerun_btn)
        self.first_tab_Layout.addRow(self.cam_stop_btn)

        self.addFirstTabItems()

        self.fourth_tab_Layout.addRow(
            QLabel('Selected Line:'),
            self.lineSelector)
        self.fourth_tab_Layout.addRow(
            QLabel('Line Mode:'),
            self.lineMode)
        self.fourth_tab_Layout.addRow(
            QLabel('Line Source:'),
            self.lineSource)
        self.fourth_tab_Layout.addWidget(
            self.set_io_config)

        self.fourth_tab_Layout.addRow(
            QLabel('Selected Timer:'),
            self.timerSelector)
        self.fourth_tab_Layout.addRow(
            QLabel('Timer Delay (us):'),
            self.timerDelay)
        self.fourth_tab_Layout.addRow(
            QLabel('Timer Duration (us):'),
            self.timerDuration)
        self.fourth_tab_Layout.addRow(
            QLabel('Timer Trigger Activation:'),
            self.timerActivation)
        self.fourth_tab_Layout.addRow(
            QLabel('Timer Trigger Source:'),
            self.timerSource)
        self.fourth_tab_Layout.addWidget(
            self.timerReset)
        self.fourth_tab_Layout.addWidget(
            self.set_timer_config)

    @property
    def cam(self):
        '''The IDS_Camera property.

        Returns
        -------
        IDS_Camera
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: vimba_cam):
        '''The vimba_cam property.

        Parameters
        ----------
        cam : vimba_cam
            the vimba_cam to set as panel camera.
        '''
        self._cam = cam

    def setExposure(self, value):
        self.cam_exposure_qs.setValue(value*1e3)

    def set_AOI(self):
        '''Sets the AOI for the slected vimba_cam
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot set AOI while acquiring images!")
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(
                self.AOI_width_tbox.value(),
                self.AOI_height_tbox.value(),
                self.AOI_x_tbox.value(),
                self.AOI_y_tbox.value())

        self.AOI_x_tbox.setValue(int(self.cam.offsetX))
        self.AOI_y_tbox.setValue(int(self.cam.offsetY))
        self.AOI_width_tbox.setValue(int(self.cam.width))
        self.AOI_height_tbox.setValue(int(self.cam.height))

        self.refresh_framerate()

    def reset_AOI(self):
        '''Resets the AOI for the slected IDS_Camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot reset AOI while acquiring images!")
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(self.cam.width_max, self.cam.height_max)
        self.AOI_x_tbox.setValue(0)
        self.AOI_y_tbox.setValue(0)
        self.AOI_width_tbox.setValue(int(self.cam.width))
        self.AOI_height_tbox.setValue(int(self.cam.height))

        self.refresh_framerate()

    def center_AOI(self):
        '''Sets the AOI for the slected vimba_cam
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot set AOI while acquiring images!")
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(
                int(self.AOI_width_tbox.text()),
                int(self.AOI_height_tbox.text()))

        self.AOI_x_tbox.setValue(self.cam.offsetX)
        self.AOI_y_tbox.setValue(self.cam.offsetY)
        self.AOI_width_tbox.setValue(self.cam.width)
        self.AOI_height_tbox.setValue(self.cam.height)

        self.refresh_framerate()

    def select_AOI(self):
        if self.frame is not None:
            aoi = cv2.selectROI(self.frame._view)
            cv2.destroyWindow('ROI selector')

            z = self.zoom_box.value()
            self.AOI_x_tbox.setValue(int(aoi[0] / z))
            self.AOI_y_tbox.setValue(int(aoi[1] / z))
            self.AOI_width_tbox.setValue(int(aoi[2] / z))
            self.AOI_height_tbox.setValue(int(aoi[3] / z))

    @pyqtSlot(str)
    def cam_cbox_changed(self, value):
        """
        Slot for changed combobox values

        Parameters
        ----------
        Value : str
            selected enum value
        """
        with self._cam.cam:
            if self.sender() is self.cam_trigger_mode_cbox:
                self._cam.set_trigger_mode(value)
                self._cam.get_trigger_mode()
            elif self.sender() is self.cam_trigger_source_cbox:
                self._cam.set_trigger_source(value)
                self._cam.get_trigger_source()
            elif self.sender() is self.cam_trigger_selector_cbox:
                self._cam.set_trigger_selector(value)
                self._cam.get_trigger_selector()
            elif self.sender() is self.cam_trigger_activation_cbox:
                self._cam.set_trigger_activation(value)
                self._cam.get_trigger_activation()
            elif self.sender() is self.cam_exposure_mode_cbox:
                self._cam.set_exposure_mode(value)
                self._cam.get_exposure_mode()
            elif self.sender() is self.cam_exposure_auto_cbox:
                self._cam.set_exposure_auto(value)
                self._cam.get_exposure_auto()
            elif self.sender() is self.cam_pixel_format_cbox:
                self._cam.set_pixel_format(value)
                self.refresh_framerate()

    @pyqtSlot(int, float)
    def cam_exposure_value_changed(self, index, value):
        """
        Slot for changed exposure

        Parameters
        ----------
        Index : ont
            selected exposure index in the slider values list
        Value : double
            selected exposure in micro-seconds
        """
        with self._cam.cam:
            self._cam.set_exposure(value)
            self._cam.get_exposure(False)

        self.refresh_exposure()
        self.refresh_framerate()

        self.OME_tab.exposure.setValue(self._cam.exposure_current / 1000)
        if self.master:
            self.exposureChanged.emit()

    @pyqtSlot(float)
    def exposure_spin_changed(self, value: float):
        """
        Slot for changed exposure

        Parameters
        ----------
        Value : double
            selected exposure in micro-seconds
        """
        with self._cam.cam:
            self._cam.set_exposure(value)
            self._cam.get_exposure(False)

        self.refresh_exposure()
        self.refresh_framerate()

        self.OME_tab.exposure.setValue(self._cam.exposure_current / 1000)
        if self.master:
            self.exposureChanged.emit()

    def refresh_exposure(self):

        self.cam_exposure_slider.elementChanged[int, float] \
            .disconnect(self.cam_exposure_value_changed)
        self.cam_exposure_slider.setNearest(self._cam.exposure_current)
        self.cam_exposure_slider.elementChanged[int, float] \
            .connect(self.cam_exposure_value_changed)

        self.cam_exposure_qs.valueChanged.disconnect(
            self.exposure_spin_changed)
        self.cam_exposure_qs.setValue(self._cam.exposure_current)
        self.cam_exposure_qs.valueChanged.connect(
            self.exposure_spin_changed)

    def refresh_framerate(self, value=None):
        with self._cam.cam:
            if value:
                self._cam.setFrameRate(value)
            self._cam.getFrameRate(False)

        self.cam_framerate_qs.setMinimum(self._cam.frameRate_range[0])
        self.cam_framerate_qs.setMaximum(self._cam.frameRate_range[1])
        self.cam_framerate_qs.setValue(self._cam.frameRate)

    def cam_framerate_changed(self, value):
        with self._cam.cam:
            self.cam.setAcquisitionFrameRateEnable(
                self.cam_framerate_enabled.isChecked())

        self.cam_framerate_qs.setEnabled(
            self.cam_framerate_enabled.isChecked())
        self.refresh_framerate()

    @pyqtSlot(float)
    def framerate_spin_changed(self, value: float):
        self.refresh_framerate(value)

    def io_line_changed(self, value):
        with self._cam.cam:
            line = self.lineSelector.currentText()
            self.cam.select_io_line(line)
            mode = self.cam.get_line_mode()
            self.lineMode.setCurrentText(
                mode)
            if 'Out' in mode:
                self.lineSource.setCurrentText(
                    self.cam.get_line_source()
                )
                self.lineInverter.setChecked(
                    self.cam.get_line_inverter()
                )

    def set_io_line_config(self):
        with self._cam.cam:
            line = self.lineSelector.currentText()
            self.cam.select_io_line(line)
            mode = self.lineMode.currentText()
            self.cam.set_line_mode(
                mode)
            if 'Out' in mode:
                self.cam.set_line_source(
                    self.lineSource.currentText())
                self.cam.set_line_inverter(
                    self.lineInverter.isChecked())

    def _capture_handler(self, cam, frame):
        if self.acq_job.frames_captured < self.acq_job.frames or \
                self.mini:
            self._buffer.put(frame.as_numpy_ndarray()[..., 0])
            cam.queue_frame(frame)
            # add sensor temperature to the stack
            self._temps.put(self.cam.get_temperature())
            self.acq_job.frames_captured = self.acq_job.frames_captured + 1
        if self.acq_job.frames_captured > self.acq_job.frames - 1 and \
                not self.mini:
            self.c_event.set()
            self.acq_job.stop_threads = True
            logging.debug('Stop')

    def cam_capture(self, *args):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : vimba_cam
            the vimba_cam used to acquire frames.
        '''
        try:
            # Continuous image capture
            with self._cam.cam:
                self._cam.cam.start_streaming(self._capture_handler)

                self.c_event.wait()
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            with self._cam.cam:
                self._cam.cam.stop_streaming()
            self._cam.acquisition = False
            QThreadPool.globalInstance().releaseThread()
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

    def get_meta(self):
        with self._cam.cam:
            meta = {
                'Exposure': self._cam.exposure_current,
                'ROI': self._cam.get_roi(False),
                'Frames': self.frames_tbox.value()
            }
        return meta

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save config", filter="XML Files (*.xml);;")

        if len(filename) > 0:

            with self.cam.cam:
                self.cam.cam.save_settings(filename, vb.PersistType.All)

            QMessageBox.information(
                self, "Info", "Config saved.")
        else:
            QMessageBox.warning(
                self, "Warning", "Config not saved.")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load config", filter="XML Files (*.xml);;")

        if len(filename) > 0:
            with self.cam.cam:
                self.cam.default()

                # Load camera settings from file.
                self.cam.cam.load_settings(filename, vb.PersistType.All)

                w, h, x, y = self.cam.get_roi()

                self.AOI_x_tbox.setValue(int(x))
                self.AOI_y_tbox.setValue(int(y))
                self.AOI_width_tbox.setValue(int(w))
                self.AOI_height_tbox.setValue(int(h))

                self.cam_trigger_mode_cbox.setCurrentText(
                    self.cam.get_trigger_mode())

                self.cam_exposure_qs.setValue(self.cam.get_exposure())
        else:
            QMessageBox.warning(
                self, "Warning", "No file selected.")
