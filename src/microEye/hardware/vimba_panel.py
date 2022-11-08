import json
import logging
import os
import threading
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

from .camera_calibration import dark_calibration

from ..metadata import MetadataEditor
from ..qlist_slider import *
from ..thread_worker import *
from .vimba_cam import vimba_cam
from ..uImage import uImage

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

    def __init__(self, threadpool: QThreadPool, cam: vimba_cam,
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
        self.frame = None
        self._frames = Queue()
        # reserved for testing/displaying execution time
        self._exec_time = 0
        # reserved for testing/displaying execution time
        self._save_time = 0
        # reserved for testing/displaying execution tim
        self._directory = ""  # save directory
        self._save_path = ""  # save path

        self._dis_time = 0

        self._counter = 0
        self._nFrames = 1

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
        self.fourth_tab = QWidget()

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
        self.main_tab_view.addTab(self.fourth_tab, "GPIOs / Timers")
        if not self.mini:
            self.main_tab_view.addTab(self.OME_tab, "OME-XML metadata")

        # first tab vertical layout
        self.first_tab_Layout = QFormLayout()
        # set as first tab layout
        self.first_tab.setLayout(self.first_tab_Layout)

        # second tab vertical layout
        self.second_tab_Layout = QFormLayout()
        # set as second tab layout
        self.second_tab.setLayout(self.second_tab_Layout)

        # third tab vertical layout
        self.third_tab_Layout = QFormLayout()
        # set as third tab layout
        self.third_tab.setLayout(self.third_tab_Layout)

        # fourth tab vertical layout
        self.fourth_tab_Layout = QFormLayout()
        # set as fourth tab layout
        self.fourth_tab.setLayout(self.fourth_tab_Layout)

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

        self.cam_exp_shortcuts = QHBoxLayout()
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                'min',
                clicked=lambda: self.cam_exposure_qs.setValue(
                    self._cam.exposure_range[0]))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '1ms',
                clicked=lambda: self.cam_exposure_qs.setValue(1e3))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '10ms',
                clicked=lambda: self.cam_exposure_qs.setValue(1e4))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '30ms',
                clicked=lambda: self.cam_exposure_qs.setValue(3e4))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '50ms',
                clicked=lambda: self.cam_exposure_qs.setValue(5e4))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '100ms',
                clicked=lambda: self.cam_exposure_qs.setValue(1e5))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '150ms',
                clicked=lambda: self.cam_exposure_qs.setValue(1.5e5))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '200ms',
                clicked=lambda: self.cam_exposure_qs.setValue(2e5))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '500ms',
                clicked=lambda: self.cam_exposure_qs.setValue(5e5))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '1s',
                clicked=lambda: self.cam_exposure_qs.setValue(1e6))
        )

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

        self._directory = os.path.dirname(os.path.realpath(__package__))
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
        self.dark_cal = QCheckBox("Dark Calibration")
        self.dark_cal.setToolTip('Generates mean and variance images')
        self.dark_cal.setChecked(False)
        self.save_bigg_tiff = QCheckBox("BiggTiff Format")
        self.save_bigg_tiff.setChecked(True)
        self.cam_save_meta = QCheckBox("Write full OME-XML")
        self.cam_save_meta.setChecked(self.mini)

        # preview checkbox
        self.preview_ch_box = QCheckBox("Preview")
        self.preview_ch_box.setChecked(not self.mini)

        self.dual_view_bx = QCheckBox("Dual Channel (Splits the AOI in half).")
        self.dual_view_bx.setChecked(False)

        # preview checkbox
        self.slow_lut_rbtn = QRadioButton("LUT Numpy (12bit)")
        self.slow_lut_rbtn.setChecked(True)
        self.fast_lut_rbtn = QRadioButton("LUT Opencv (8bit)")
        self.lut_btns = QButtonGroup()
        self.lut_btns.addButton(self.slow_lut_rbtn)
        self.lut_btns.addButton(self.fast_lut_rbtn)

        # display size
        self.zoom_lbl = QLabel("Resize Display:")
        self.zoom_box = QDoubleSpinBox()
        self.zoom_box.setSingleStep(0.02)
        self.zoom_box.setMinimum(0.1)
        self.zoom_box.setMaximum(4.0)
        self.zoom_box.setDecimals(2)
        self.zoom_box.setValue(0.5)

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
        self.AOI_x_tbox = QSpinBox()
        self.AOI_y_tbox = QSpinBox()
        self.AOI_width_tbox = QSpinBox()
        self.AOI_height_tbox = QSpinBox()
        self.AOI_x_tbox.setMinimum(0)
        self.AOI_x_tbox.setMaximum(self.cam.width_range[1])
        self.AOI_y_tbox.setMinimum(0)
        self.AOI_y_tbox.setMaximum(self.cam.height_range[1])
        self.AOI_width_tbox.setMinimum(self.cam.width_range[0])
        self.AOI_width_tbox.setMaximum(self.cam.width_range[1])
        self.AOI_height_tbox.setMinimum(self.cam.height_range[0])
        self.AOI_height_tbox.setMaximum(self.cam.height_range[1])
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
        self.AOI_select_btn = QPushButton(
            "Select AOI",
            clicked=lambda: self.select_AOI()
        )

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
        self.first_tab_Layout.addRow(self.config_Hlay)
        self.first_tab_Layout.addRow(self.cam_freerun_btn)
        self.first_tab_Layout.addRow(self.cam_trigger_btn)
        self.first_tab_Layout.addRow(self.cam_stop_btn)
        if not self.mini:
            self.first_tab_Layout.addRow(
                QLabel("Experiment:"),
                self.experiment_name)
            self.first_tab_Layout.addRow(
                QLabel("Save Directory:"),
                self.save_dir_layout)
            self.first_tab_Layout.addRow(
                QLabel('Frames:'),
                self.frames_tbox)
            self.first_tab_Layout.addWidget(self.cam_save_temp)
            self.first_tab_Layout.addWidget(self.dark_cal)
            self.first_tab_Layout.addWidget(self.save_bigg_tiff)
            self.first_tab_Layout.addWidget(self.cam_save_meta)

        if not self.mini:
            self.second_tab_Layout.addRow(self.preview_ch_box)
            self.second_tab_Layout.addRow(self.dual_view_bx)

        self.second_tab_Layout.addRow(self.slow_lut_rbtn)
        self.second_tab_Layout.addRow(self.fast_lut_rbtn)
        if not self.mini:
            self.second_tab_Layout.addRow(
                self.zoom_lbl,
                self.zoom_box)

        self.second_tab_Layout.addRow(
            self.histogram_lbl,
            self.alpha)
        self.second_tab_Layout.addWidget(self.beta)
        self.second_tab_Layout.addWidget(self.auto_stretch)
        self.second_tab_Layout.addRow(self.histogram)
        self.second_tab_Layout.addRow(self.hist_cdf)
        self.second_tab_Layout.addRow(
            QLabel("Stats:"), self.info_temp)
        self.second_tab_Layout.addWidget(self.info_cap)
        self.second_tab_Layout.addWidget(self.info_disp)
        self.second_tab_Layout.addWidget(self.info_save)

        self.third_tab_Layout.addRow(
            QLabel("X:"),
            self.AOI_x_tbox)
        self.third_tab_Layout.addRow(
            QLabel("Y:"),
            self.AOI_y_tbox)
        self.third_tab_Layout.addRow(
            QLabel("Width:"),
            self.AOI_width_tbox)
        self.third_tab_Layout.addRow(
            QLabel("Height:"),
            self.AOI_height_tbox)
        self.third_tab_Layout.addRow(self.AOI_set_btn)
        self.third_tab_Layout.addRow(self.AOI_reset_btn)
        self.third_tab_Layout.addRow(self.AOI_center_btn)
        self.third_tab_Layout.addRow(self.AOI_select_btn)

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
                self.AOI_width_tbox.value(),
                self.AOI_height_tbox.value(),
                self.AOI_x_tbox.value(),
                self.AOI_y_tbox.value())

        self.AOI_x_tbox.setValue(int(self.cam.offsetX))
        self.AOI_y_tbox.setValue(int(self.cam.offsetY))
        self.AOI_width_tbox.setValue(int(self.cam.width))
        self.AOI_height_tbox.setValue(int(self.cam.height))

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

    def select_AOI(self):
        if self.frame is not None:
            aoi = cv2.selectROI(self.frame._view)
            cv2.destroyWindow('ROI selector')

            z = self.zoom_box.value()
            self.AOI_x_tbox.setValue(int(aoi[0] / z))
            self.AOI_y_tbox.setValue(int(aoi[1] / z))
            self.AOI_width_tbox.setValue(int(aoi[2] / z))
            self.AOI_height_tbox.setValue(int(aoi[3] / z))

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

        self.refresh_exposure()

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
        if self._event is not None:
            self._event.set()
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
            self.d_worker.setAutoDelete(True)
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.done:
                self.s_worker = thread_worker(
                    self._save, progress=False, z_stage=False)
                self.s_worker.setAutoDelete(True)
                # Execute
                self._threadpool.start(self.s_worker)

        #  Pass the capture function to be executed
        if self.c_worker is None or self.c_worker.done:
            # Any other args, kwargs are passed to the run function
            self.c_worker = thread_worker(
                self._capture, self.cam, progress=False, z_stage=False)
            self.c_worker.setAutoDelete(True)
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

    def _capture_handler(self, cam, frame):
        self._buffer.put(frame.as_numpy_ndarray().copy())
        cam.queue_frame(frame)
        # add sensor temperature to the stack
        self._temps.put(self.cam.get_temperature())
        self._counter = self._counter + 1
        if self._counter > self._nFrames - 1 and not self.mini:
            self._event.set()
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
        try:
            self._buffer.queue.clear()
            self._temps.queue.clear()
            self._frames.queue.clear()
            self._counter = 0
            self._event = threading.Event()
            self.time = QDateTime.currentDateTime()
            self._nFrames = int(self.frames_tbox.text())
            # Continuous image capture

            with self._cam.cam:
                cam.cam.start_streaming(self._capture_handler)

                self._event.wait()
            # while(True):
            #     QThread.usleep(500)

            #     if self._stop_thread:
            #         break  # in case stop threads is initiated
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            with self._cam.cam:
                cam.cam.stop_streaming()
            self._exec_time = self.time.msecsTo(
                QDateTime.currentDateTime()) / self._counter
            cam.acquisition = False
            self._threadpool.releaseThread()

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
                    self.frame = uImage(self._buffer.get()[..., 0]) \
                        if self.cam.bytes_per_pixel > 1\
                        else uImage.fromUINT8(
                            self._buffer.get()[..., 0],
                            self.cam.height, self.cam.width)

                    # add to saving stack
                    if self.cam_save_temp.isChecked():
                        if not self.mini:
                            self._frames.put(
                                (self.frame.image, self._temps.get()))
                        else:
                            self._frames.put(self.frame.image)

                    if self.preview_ch_box.isChecked():
                        _range = None
                        # image stretching
                        if not self.auto_stretch.isChecked():
                            _range = (self.alpha.value(), self.beta.value())

                        self.frame.equalizeLUT(
                            _range, self.slow_lut_rbtn.isChecked())
                        if self.auto_stretch.isChecked():
                            self.alpha.setValue(self.frame._min)
                            self.beta.setValue(self.frame._max)
                        self.histogram.setXRange(
                            self.frame._min, self.frame._max)
                        self.hist_cdf.setXRange(
                            self.frame._min, self.frame._max)

                        self._plot_ref.setData(self.frame._hist[:, 0])
                        self._cdf_plot_ref.setData(self.frame._cdf)

                        if self.dual_view_bx.isChecked():
                            BGR_img = self.frame.hsplitView(False)

                            BGR_img = cv2.resize(
                                BGR_img,
                                (0, 0),
                                fx=self.zoom_box.value(),
                                fy=self.zoom_box.value(),
                                interpolation=cv2.INTER_NEAREST)

                            cv2.imshow(cam.name, BGR_img)
                        else:
                            # resizing the image
                            self.frame._view = cv2.resize(
                                self.frame._view, (0, 0),
                                fx=self.zoom_box.value(),
                                fy=self.zoom_box.value(),
                                interpolation=cv2.INTER_NEAREST)

                            # display it
                            cv2.imshow(cam.name, self.frame._view)
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
            path = self._save_path
            tiffWriter = None
            tempFile = None
            darkCal = None
            index = 0
            biggTiff = self.save_bigg_tiff.isChecked()

            def getFilename(index: int):
                return path + \
                       '\\image_{:05d}.ome.tif'.format(index)

            def saveMetadata(index: int):
                if self.cam_save_meta.isChecked():
                    ome = self.OME_tab.gen_OME_XML(
                        frames_saved,
                        self._cam.width,
                        self._cam.height)
                    tf.tiffcomment(
                        getFilename(index),
                        ome.to_xml())
                else:
                    ome = self.OME_tab.gen_OME_XML_short(
                        frames_saved,
                        self._cam.width,
                        self._cam.height)
                    tf.tiffcomment(
                        getFilename(index),
                        ome.to_xml())

            while(True):
                # save in case frame stack is not empty
                if not self._frames.empty():
                    # for save time estimations
                    time = QDateTime.currentDateTime()

                    # get frame and temp to save from bottom of stack
                    frame, temp = self._frames.get()

                    if tempFile is None:
                        if not os.path.exists(path):
                            os.makedirs(path)

                        tempFile = open(path + '\\temp_log.csv', 'ab')

                    if tiffWriter is None:
                        tiffWriter = tf.TiffWriter(
                            getFilename(index),
                            append=False,
                            bigtiff=biggTiff,
                            ome=False)

                    if self.dark_cal.isChecked():
                        if darkCal is None:
                            exp = self._cam.exposure_current
                            darkCal = dark_calibration(frame.shape, exp)

                        darkCal.addFrame(frame)

                    # append frame to tiff
                    try:
                        tiffWriter.write(
                            data=frame[np.newaxis, :],
                            photometric='minisblack')
                    except ValueError as ve:
                        if str(ve) == 'data too large for standard TIFF file':
                            tiffWriter.close()
                            saveMetadata(index, frames_saved)
                            frames_saved = 0
                            index += 1
                            tiffWriter = tf.TiffWriter(
                                getFilename(index),
                                append=False,
                                bigtiff=biggTiff,
                                ome=False)
                            tiffWriter.write(
                                data=frame[np.newaxis, :],
                                photometric='minisblack')
                        else:
                            raise ve

                    # open csv file and append sensor temp and close
                    np.savetxt(tempFile, [temp], delimiter=";")

                    # for save time estimations
                    self._save_time = time.msecsTo(
                        QDateTime.currentDateTime())

                    frames_saved = frames_saved + 1

                QThread.usleep(100)
                # Flag that ends the loop
                if self._frames.empty() & self._stop_thread \
                        & self.d_worker.done:
                    break
        except Exception:
            traceback.print_exc()
        finally:
            if tempFile is not None:
                tempFile.close()
            if tiffWriter is not None:
                tiffWriter.close()
            if darkCal is not None:
                if darkCal._counter > 1:
                    darkCal.saveResults(path)

            if self.cam_save_temp.isChecked():
                saveMetadata(index)

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
