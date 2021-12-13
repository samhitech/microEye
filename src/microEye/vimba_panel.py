import json
import logging
import os
import time
import traceback
from queue import Queue

import cv2
import numpy as np
import pyqtgraph as pg
import tifffile as tf
from numpy import random, true_divide
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph import PlotWidget, plot
from pyqtgraph.metaarray.MetaArray import axis
from tifffile.tifffile import astype

from .metadata import MetadataEditor
from .qlist_slider import *
from .thread_worker import *
from .vimba_cam import vimba_cam
from .uImage import uImage

try:
    import vimba as vb
except Exception:
    vb = None


class Vimba_Panel(QGroupBox):
    """
    A Qt Widget for controlling an Allied Vision Camera through Vimba
     | Inherits QGroupBox
    """

    exposureChanged = pyqtSignal()

    def __init__(self, threadpool, cam: vimba_cam,
                 *args, mini=False, **kwargs):
        """
        Initializes a new Vimba_Panel Qt widget
        | Inherits QGroupBox

        Parameters
        ----------
        threadpool : QThreadPool
            The threadpool for multithreading
        cam : vimba_cam
            Vimba Camera python adapter
        """
        super().__init__(*args, **kwargs)
        self._cam = cam  # Vimba Camera
        # flag true if master (always the first added camera is master)
        self.master = False

        self.c_worker = None  # worker for capturing
        self.d_worker = None  # worker for display
        self.s_worker = None  # worker for saving

        # number of bins for the histogram (4096 is set for 12bit mono-camera)
        self._nBins = 2**12
        self._hist = np.arange(0, self._nBins) * 0  # arrays for the Hist plot
        self._bins = np.arange(0, self._nBins)      # arrays for the Hist plot

        self._threadpool = threadpool  # the threadpool for workers
        self._stop_thread = False  # flag true to stop capture/display threads

        # flag true to close camera adapter and dispose it
        self._dispose_cam = False
        self._counter = 0  # a frame counter for acquisition
        self._buffer = Queue()  # data memory stack
        self._temps = Queue()  # camera sensor temperature stack
        # stack of (frame, temperature) tuples (for saving)
        self._frames = Queue()
        # reserved for testing/displaying execution time
        self._exec_time = 0
        # reserved for testing/displaying execution time
        self._save_time = 0
        # reserved for testing/displaying execution time
        self._dis_time = 0
        self._zoom = 0.50  # display resize
        self._directory = ""  # save directory
        self._save_path = ""  # save path

        self.mini = mini

        # main layout
        self.main_layout = QVBoxLayout()

        # set main layout
        self.setLayout(self.main_layout)

        # the main tab widget
        self.main_tab_view = QTabWidget()
        self.main_layout.addWidget(self.main_tab_view)

        # tab widgets
        self.first_tab = QWidget()
        self.second_tab = QWidget()
        self.third_tab = QWidget()

        self.OME_tab = MetadataEditor()
        self.OME_tab.channel_name.setText(self._cam.name)
        self.OME_tab.det_manufacturer.setText('Allied Vision')
        self.OME_tab.det_model.setText(
            self._cam.cam.get_model())
        self.OME_tab.det_serial.setText(
            self._cam.cam.get_serial())
        # add tabs
        self.main_tab_view.addTab(self.first_tab, "Main")
        self.main_tab_view.addTab(self.second_tab, "Preview")
        self.main_tab_view.addTab(self.third_tab, "Area of Interest (AOI)")
        if not self.mini:
            self.main_tab_view.addTab(self.OME_tab, "OME-XML metadata")

        # first tab vertical layout
        self.first_tab_Layout = QVBoxLayout()
        # set as first tab layout
        self.first_tab.setLayout(self.first_tab_Layout)

        # second tab vertical layout
        self.second_tab_Layout = QVBoxLayout()
        # set as second tab layout
        self.second_tab.setLayout(self.second_tab_Layout)

        # third tab vertical layout
        self.third_tab_Layout = QVBoxLayout()
        # set as third tab layout
        self.third_tab.setLayout(self.third_tab_Layout)

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
        self.cam_exposure_qs = QDoubleSpinBox()
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
            clicked=lambda: self.start_free_run(self._cam)
        )

        # start trigger mode button
        self.cam_trigger_btn = QPushButton(
            "Trigger Mode (Start)",
            clicked=lambda: self.start_software_triggered(self._cam)
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

        self.experiment_name = QLineEdit("Experiment_001")
        self.experiment_name.textChanged.connect(
            lambda x: self.OME_tab.experiment.setText(x))

        self.save_dir_layout = QHBoxLayout()

        self._directory = os.path.dirname(os.path.realpath(__file__))
        self.save_dir_edit = QLineEdit(self._directory)
        self.save_dir_edit.setReadOnly(True)

        self.save_browse_btn = QPushButton(
            "...", clicked=lambda: self.save_browse_clicked())

        self.save_dir_layout.addWidget(self.save_dir_edit)
        self.save_dir_layout.addWidget(self.save_browse_btn)

        self.frames_tbox = QLineEdit("1000")
        self.frames_tbox.setValidator(QIntValidator())

        # save to hard drive checkbox
        self.cam_save_temp = QCheckBox("Save to dir")
        self.cam_save_temp.setChecked(self.mini)
        self.cam_save_meta = QCheckBox("Write OME-XML")
        self.cam_save_meta.setChecked(self.mini)

        # preview checkbox
        self.preview_ch_box = QCheckBox("Preview")
        self.preview_ch_box.setChecked(not self.mini)

        # preview checkbox
        self.slow_lut_rbtn = QRadioButton("LUT Numpy (12bit)")
        self.slow_lut_rbtn.setChecked(True)
        self.fast_lut_rbtn = QRadioButton("LUT Opencv (8bit)")
        self.lut_btns = QButtonGroup()
        self.lut_btns.addButton(self.slow_lut_rbtn)
        self.lut_btns.addButton(self.fast_lut_rbtn)

        # display size
        self.zoom_layout = QHBoxLayout()
        self.zoom_lbl = QLabel("Resize " + "{:.0f}%".format(self._zoom*100))
        self.zoom_in_btn = QPushButton(
            "+",
            clicked=lambda: self.zoom_in()
        )
        self.zoom_out_btn = QPushButton(
            "-",
            clicked=lambda: self.zoom_out()
        )
        self.zoom_layout.addWidget(self.zoom_lbl, 4)
        self.zoom_layout.addWidget(self.zoom_out_btn, 1)
        self.zoom_layout.addWidget(self.zoom_in_btn, 1)

        # controls for histogram stretching and plot
        self.histogram_lbl = QLabel("Histogram")
        self.alpha = QSlider(orientation=Qt.Orientation.Horizontal)
        self.alpha.setMinimum(0)
        self.alpha.setMaximum(self._nBins)
        self.alpha.setValue(0)
        self.beta = QSlider(orientation=Qt.Orientation.Horizontal)
        self.beta.setMinimum(0)
        self.beta.setMaximum(self._nBins)
        self.beta.setValue(self._nBins)
        # autostretch checkbox
        self.auto_stretch = QCheckBox("Auto Stretch")
        self.auto_stretch.setChecked(True)
        # Hist plotWidget
        self.histogram = pg.PlotWidget()
        self.hist_cdf = pg.PlotWidget()
        self._plot_ref = self.histogram.plot(self._bins, self._hist)
        self._cdf_plot_ref = self.hist_cdf.plot(self._bins, self._hist)

        # Stats. info
        self.info_cap = QLabel("Capture {:d} | {:.2f} ms ".format(
            self._counter, self._exec_time))
        self.info_disp = QLabel("Display {:d} | {:.2f} ms ".format(
            self._buffer.qsize(), self._dis_time))
        self.info_save = QLabel("Save {:d} | {:.2f} ms ".format(
            self._frames.qsize(), self._save_time))
        with self._cam.cam:
            self.info_temp = QLabel(
                " T {:.2f} °C".format(self._cam.get_temperature()))

        # AOI controls
        self.AOI_x_tbox = QLineEdit("0")
        self.AOI_y_tbox = QLineEdit("0")
        self.AOI_width_tbox = QLineEdit("0")
        self.AOI_height_tbox = QLineEdit("0")
        self.AOI_x_tbox.setValidator(
            QIntValidator(0, self.cam.width_range[1]))
        self.AOI_y_tbox.setValidator(
            QIntValidator(0, self.cam.height_range[1]))
        self.AOI_width_tbox.setValidator(
            QIntValidator(self.cam.width_range[0], self.cam.width_range[1]))
        self.AOI_height_tbox.setValidator(
            QIntValidator(self.cam.height_range[0], self.cam.height_range[1]))
        self.AOI_set_btn = QPushButton(
            "Set AOI",
            clicked=lambda: self.set_AOI()
        )
        self.AOI_reset_btn = QPushButton(
            "Reset AOI",
            clicked=lambda: self.reset_AOI()
        )
        self.AOI_center_btn = QPushButton(
            "Center AOI",
            clicked=lambda: self.center_AOI()
        )

        # adding widgets to the main layout
        self.first_tab_Layout.addWidget(self.cam_trigger_mode_lbl)
        self.first_tab_Layout.addWidget(self.cam_trigger_mode_cbox)
        self.first_tab_Layout.addWidget(self.cam_trigger_source_lbl)
        self.first_tab_Layout.addWidget(self.cam_trigger_source_cbox)
        self.first_tab_Layout.addWidget(self.cam_trigger_selector_lbl)
        self.first_tab_Layout.addWidget(self.cam_trigger_selector_cbox)
        self.first_tab_Layout.addWidget(self.cam_trigger_activation_lbl)
        self.first_tab_Layout.addWidget(self.cam_trigger_activation_cbox)

        self.first_tab_Layout.addWidget(self.cam_exposure_lbl)
        self.first_tab_Layout.addWidget(self.cam_exposure_qs)
        self.first_tab_Layout.addWidget(self.cam_exposure_slider)
        self.first_tab_Layout.addWidget(self.cam_exposure_mode_lbl)
        self.first_tab_Layout.addWidget(self.cam_exposure_mode_cbox)
        self.first_tab_Layout.addWidget(self.cam_exposure_auto_lbl)
        self.first_tab_Layout.addWidget(self.cam_exposure_auto_cbox)
        self.first_tab_Layout.addWidget(self.cam_pixel_format_lbl)
        self.first_tab_Layout.addWidget(self.cam_pixel_format_cbox)
        self.first_tab_Layout.addLayout(self.config_Hlay)
        self.first_tab_Layout.addWidget(self.cam_freerun_btn)
        self.first_tab_Layout.addWidget(self.cam_trigger_btn)
        self.first_tab_Layout.addWidget(self.cam_stop_btn)
        if not self.mini:
            self.first_tab_Layout.addWidget(QLabel("Experiment:"))
            self.first_tab_Layout.addWidget(self.experiment_name)
            self.first_tab_Layout.addWidget(QLabel("Save Directory:"))
            self.first_tab_Layout.addLayout(self.save_dir_layout)
            self.first_tab_Layout.addWidget(self.frames_tbox)
            self.first_tab_Layout.addWidget(self.cam_save_temp)
            self.first_tab_Layout.addWidget(self.cam_save_meta)
        self.first_tab_Layout.addStretch()

        if not self.mini:
            self.second_tab_Layout.addWidget(self.preview_ch_box)
        self.second_tab_Layout.addWidget(self.slow_lut_rbtn)
        self.second_tab_Layout.addWidget(self.fast_lut_rbtn)
        if not self.mini:
            self.second_tab_Layout.addLayout(self.zoom_layout)

        self.second_tab_Layout.addWidget(self.histogram_lbl)
        self.second_tab_Layout.addWidget(self.alpha)
        self.second_tab_Layout.addWidget(self.beta)
        self.second_tab_Layout.addWidget(self.auto_stretch)
        self.second_tab_Layout.addWidget(self.histogram)
        self.second_tab_Layout.addWidget(self.hist_cdf)
        self.second_tab_Layout.addWidget(QLabel("Stats"))
        self.second_tab_Layout.addWidget(self.info_temp)
        self.second_tab_Layout.addWidget(self.info_cap)
        self.second_tab_Layout.addWidget(self.info_disp)
        self.second_tab_Layout.addWidget(self.info_save)
        self.second_tab_Layout.addStretch()

        self.third_tab_Layout.addWidget(QLabel("X"))
        self.third_tab_Layout.addWidget(self.AOI_x_tbox)
        self.third_tab_Layout.addWidget(QLabel("Y"))
        self.third_tab_Layout.addWidget(self.AOI_y_tbox)
        self.third_tab_Layout.addWidget(QLabel("Width"))
        self.third_tab_Layout.addWidget(self.AOI_width_tbox)
        self.third_tab_Layout.addWidget(QLabel("Height"))
        self.third_tab_Layout.addWidget(self.AOI_height_tbox)
        self.third_tab_Layout.addWidget(self.AOI_set_btn)
        self.third_tab_Layout.addWidget(self.AOI_reset_btn)
        self.third_tab_Layout.addWidget(self.AOI_center_btn)
        self.third_tab_Layout.addStretch()

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

    @property
    def buffer(self):
        if self._frames is not None:
            return self._frames
        else:
            return Queue()

    @property
    def isEmpty(self) -> bool:
        return self.buffer.empty()

    def get(self) -> np.ndarray:
        if not self.isEmpty:
            return self._frames.get()

    @property
    def isOpen(self) -> bool:
        return self.cam.acquisition

    def set_AOI(self):
        '''Sets the AOI for the slected vimba_cam
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot set AOI while acquiring images!")
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(
                int(self.AOI_width_tbox.text()),
                int(self.AOI_height_tbox.text()),
                int(self.AOI_x_tbox.text()),
                int(self.AOI_y_tbox.text()))

        self.AOI_x_tbox.setText(str(self.cam.offsetX))
        self.AOI_y_tbox.setText(str(self.cam.offsetY))
        self.AOI_width_tbox.setText(str(self.cam.width))
        self.AOI_height_tbox.setText(str(self.cam.height))

    def reset_AOI(self):
        '''Resets the AOI for the slected IDS_Camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot reset AOI while acquiring images!")
            return  # if acquisition is already going on

        with self._cam.cam:
            self.cam.set_roi(self.cam.width_max, self.cam.height_max)
        self.AOI_x_tbox.setText("0")
        self.AOI_y_tbox.setText("0")
        self.AOI_width_tbox.setText(str(self.cam.width))
        self.AOI_height_tbox.setText(str(self.cam.height))

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

        self.AOI_x_tbox.setText(str(self.cam.offsetX))
        self.AOI_y_tbox.setText(str(self.cam.offsetY))
        self.AOI_width_tbox.setText(str(self.cam.width))
        self.AOI_height_tbox.setText(str(self.cam.height))

    def zoom_in(self):
        """Increase image display size"""
        self._zoom = min(self._zoom + 0.05, 4)
        self.zoom_lbl.setText("Resize " + "{:.0f}%".format(self._zoom*100))

    def zoom_out(self):
        """Decrease image display size"""
        self._zoom = max(self._zoom - 0.05, 0.25)
        self.zoom_lbl.setText("Resize " + "{:.0f}%".format(self._zoom*100))

    @pyqtSlot()
    def save_browse_clicked(self):
        """Slot for browse clicked event"""
        self._directory = ""

        while len(self._directory) == 0:
            self._directory = str(
                QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.save_dir_edit.setText(self._directory)

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

        self.cam_exposure_qs.valueChanged.disconnect(
            self.exposure_spin_changed)
        self.cam_exposure_qs.setValue(self._cam.exposure_current)
        self.cam_exposure_qs.valueChanged.connect(
            self.exposure_spin_changed)

        self.OME_tab.exposure.setText(str(self._cam.exposure_current / 1000))
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

        self.cam_exposure_slider.elementChanged[int, float] \
            .connect(self.cam_exposure_value_changed)
        self.cam_exposure_slider.setNearest(self._cam.exposure_current)
        self.cam_exposure_slider.elementChanged[int, float] \
            .disconnect(self.cam_exposure_value_changed)

        self.OME_tab.exposure.setText(str(self._cam.exposure_current / 1000))
        if self.master:
            self.exposureChanged.emit()

    def start_free_run(self, cam: vimba_cam):
        """
        Starts free run acquisition mode

        Parameters
        ----------
        cam : vimba_cam
            Vimba Camera python adapter
        """
        nRet = 0
        if cam.acquisition:
            return  # if acquisition is already going on

        self._save_path = (self._directory + "\\"
                           + self.experiment_name.text()
                           + "\\" + self.cam.name
                           + time.strftime("_%Y_%m_%d_%H%M%S"))

        cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers()

    def start_software_triggered(self, cam: vimba_cam):
        """
        Starts trigger acquisition mode

        Parameters
        ----------
        cam : vimba_cam
            Vimba Camera python adapter
        """

        nRet = 0
        if cam.acquisition:
            return  # if acquisition is already going on

        self._save_path = (self._directory + "\\"
                           + self.experiment_name.text() + "\\"
                           + self.cam.name
                           + time.strftime("_%Y_%m_%d_%H%M%S"))

        cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers()

    def stop(self):
        self._stop_thread = True  # set stop acquisition workers flag to true

    def start_all_workers(self):
        """
        Starts all workers
        """

        self._stop_thread = False  # set stop acquisition workers flag to false

        # Pass the display function to be executed
        if self.d_worker is None or self.d_worker.done:
            self.d_worker = thread_worker(
                self._display, self.cam, progress=False, z_stage=False)
            # Execute
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.done:
                self.s_worker = thread_worker(
                    self._save, progress=False, z_stage=False)
                # Execute
                self._threadpool.start(self.s_worker)

        #  Pass the capture function to be executed
        if self.c_worker is None or self.c_worker.done:
            # Any other args, kwargs are passed to the run function
            self.c_worker = thread_worker(
                self._capture, self.cam, progress=False, z_stage=False)
            # Execute
            self._threadpool.start(self.c_worker)

        # Giving the capture thread a head start over the display one
        QThread.msleep(500)

    def start_dis_save_workers(self):
        """
        Starts both the display and save workers only
        """
        self._stop_thread = False  # set stop acquisition workers flag to false

        # Pass the display function to be executed
        if self.d_worker is None or self.d_worker.done:
            self.d_worker = thread_worker(
                self._display, self.cam, progress=False, z_stage=False)
            # Execute
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.done:
                self.s_worker = thread_worker(
                    self._save, progress=False, z_stage=False)
                # Execute
                self._threadpool.start(self.s_worker)

    def _capture_handler(self, cam: vb.Camera, frame):
        self._buffer.put(frame.as_numpy_ndarray())
        cam.queue_frame(frame)
        # add sensor temperature to the stack
        self._temps.put(self.cam.get_temperature())
        self._counter = self._counter + 1
        if self._counter > self.nFrames - 1 and not self.mini:
            self._stop_thread = True
            logging.debug('Stop')

    def _capture(self, cam: vimba_cam):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        cam : vimba_cam
            the vimba_cam used to acquire frames.
        '''
        with self._cam.cam:
            try:
                self._buffer.queue.clear()
                self._temps.queue.clear()
                self._frames.queue.clear()
                self._counter = 0
                self.time = QDateTime.currentDateTime()
                self.nFrames = int(self.frames_tbox.text())
                # Continuous image capture
                cam.cam.start_streaming(self._capture_handler)
                while(True):
                    QThread.usleep(500)

                    if self._stop_thread:
                        cam.cam.stop_streaming()
                        break  # in case stop threads is initiated
                self._exec_time = self.time.msecsTo(
                    QDateTime.currentDateTime()) / self._counter
            except Exception:
                traceback.print_exc()
            finally:
                # reset flags and release resources
                cam.acquisition = False

    def _display(self, cam: vimba_cam):
        '''Display function executed by the display worker.

        Processes the acquired frame, displays it, and sends it to the save
        stack.

        Parameters
        ----------
        cam : vimba_cam
            the vimba_cam used to acquire frames.
        '''
        try:
            time = QDateTime.currentDateTime()
            # Continuous image display
            while(True):
                # for display time estimations

                # proceed only if the buffer is not empty
                if not self._buffer.empty():
                    self._dis_time = time.msecsTo(QDateTime.currentDateTime())
                    time = QDateTime.currentDateTime()

                    # reshape image into proper shape
                    # (height, width, bytes per pixel)
                    frame = uImage(self._buffer.get()) \
                        if self.cam.bytes_per_pixel > 1\
                        else uImage.fromUINT8(
                            self._buffer.get(),
                            self.cam.height, self.cam.width)

                    # add to saving stack
                    if self.cam_save_temp.isChecked():
                        if not self.mini:
                            self._frames.put((frame.image, self._temps.get()))
                        else:
                            self._frames.put(frame.image)

                    if self.preview_ch_box.isChecked():
                        _range = None
                        # image stretching
                        if not self.auto_stretch.isChecked():
                            _range = (self.alpha.value(), self.beta.value())

                        frame.equalizeLUT(
                            _range, self.slow_lut_rbtn.isChecked())
                        if self.auto_stretch.isChecked():
                            self.alpha.setValue(frame._min)
                            self.beta.setValue(frame._max)
                        self.histogram.setXRange(frame._min, frame._max)
                        self.hist_cdf.setXRange(frame._min, frame._max)

                        self._plot_ref.setData(frame._hist[:, 0])
                        self._cdf_plot_ref.setData(frame._cdf)

                        # resizing the image
                        frame._view = cv2.resize(
                            frame._view, (0, 0), fx=self._zoom, fy=self._zoom)

                        # display it
                        cv2.imshow(cam.name, frame._view)
                        cv2.waitKey(1)
                    else:
                        cv2.waitKey(1)
                else:
                    cv2.waitKey(1)

                QThread.usleep(100)
                # Flag that ends the loop
                if self._stop_thread & self._buffer.empty() \
                        & self.c_worker.done:
                    break
        except Exception:
            traceback.print_exc()
        finally:
            if not self.mini:
                cv2.destroyWindow(cam.name)

    def _save(self):
        '''Save function executed by the save worker.

        Saves the acquired frames in bigtiff format associated
        with a csv file containing the sensor temp for each frame.

        Parameters
        ----------
        nRet : int
            return code from IDS_Camera, ueye.IS_SUCCESS = 0 to run.
        '''
        try:
            frames_saved = 0
            while(True):
                # save in case frame stack is not empty
                if not self._frames.empty():
                    # for save time estimations
                    time = QDateTime.currentDateTime()

                    # get frame and temp to save from bottom of stack
                    frame, temp = self._frames.get()

                    # creates dir
                    if not os.path.exists(self._save_path):
                        os.makedirs(self._save_path)

                    # append frame to tiff
                    tf.imwrite(
                        self._save_path + "\\image.ome.tif",
                        data=frame[np.newaxis, :],
                        photometric='minisblack',
                        append=True,
                        bigtiff=True,
                        ome=False)

                    # open csv file and append sensor temp and close
                    file = open(self._save_path + '\\temps.csv', 'ab')
                    np.savetxt(file, [temp], delimiter=";")
                    file.close()

                    # for save time estimations
                    self._save_time = time.msecsTo(QDateTime.currentDateTime())

                    frames_saved = frames_saved + 1

                QThread.usleep(100)
                # Flag that ends the loop
                if self._frames.empty() & self._stop_thread \
                        & self.d_worker.done:
                    break
        except Exception:
            traceback.print_exc()
        finally:
            if self.cam_save_meta.isChecked():
                ome = self.OME_tab.gen_OME_XML(
                    frames_saved,
                    self._cam.width,
                    self._cam.height)
                tf.tiffcomment(
                    self._save_path + "\\image.ome.tif",
                    ome.to_xml())

    @staticmethod
    def find_nearest(array, value):
        """
        find nearest value in array to the supplied one

        Parameters
        ----------
        array : ndarray
            numpy array searching in
        value : type of ndarray.dtype
            value to find nearest to in the array
        """
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

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

                self.AOI_x_tbox.setText(str(x))
                self.AOI_y_tbox.setText(str(y))
                self.AOI_width_tbox.setText(str(w))
                self.AOI_height_tbox.setText(str(h))

                self.cam_trigger_mode_cbox.setCurrentText(
                    self.cam.get_trigger_mode())

                self.cam_exposure_ledit.setValue(self.cam.get_exposure())
        else:
            QMessageBox.warning(
                self, "Warning", "No file selected.")