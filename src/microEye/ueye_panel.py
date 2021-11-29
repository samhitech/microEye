import json
import logging
import os
import time
import traceback
from queue import Queue

import cv2
from lmfit.minimizer import minimize
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

from .qlist_slider import *
from .thread_worker import *
from .ueye_camera import IDS_Camera
from .uImage import uImage
from .metadata import MetadataEditor

try:
    from pyueye import ueye
except Exception:
    ueye = None


class IDS_Panel(QGroupBox):
    """
    A Qt Widget for controlling an IDS Camera | Inherits QGroupBox
    """

    exposureChanged = pyqtSignal()

    def __init__(self, threadpool, cam: IDS_Camera,
                 *args, mini=False, **kwargs):
        """
        Initializes a new IDS_Panel Qt widget
        | Inherits QGroupBox

        Parameters
        ----------
        threadpool : QThreadPool
            The threadpool for multithreading
        cam : IDS_Camera
            IDS Camera python adapter
        """
        super().__init__(*args, **kwargs)
        self._cam = cam  # IDS Camera
        # flag true if master (always the first added camera is master)
        self.master = False
        # list of slave camera for exposure sync
        self.slaves: list[IDS_Camera] = []
        self.c_worker = None  # worker for capturing
        self.d_worker = None  # worker for display
        self.s_worker = None  # worker for saving
        # number of bins for the histogram (4096 is set for 12bit mono-camera)
        self._nBins = 2**cam.bit_depth.value
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
        self._zoom = 0.25  # display resize
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
        self.OME_tab.det_manufacturer.setText('IDS uEye')
        self.OME_tab.det_model.setText(
            self._cam.sInfo.strSensorName.decode('utf-8'))
        self.OME_tab.det_serial.setText(
            self._cam.cInfo.SerNo.decode('utf-8'))
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

        # pixel clock label and combobox
        self.cam_pixel_clock_lbl = QLabel("Pixel Clock MHz")
        self.cam_pixel_clock_cbox = QComboBox()
        self.cam_pixel_clock_cbox.addItems(
            map(str, self._cam.pixel_clock_list[:]))
        self.cam_pixel_clock_cbox.currentIndexChanged[str] \
            .connect(self.cam_pixel_cbox_changed)

        # framerate slider control
        self.cam_framerate_lbl = DragLabel(
            "Framerate FPS",
            parent_name='cam_framerate_slider')
        self.cam_framerate_slider = qlist_slider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_framerate_slider.values = np.arange(
            self._cam.minFrameRate, self._cam.maxFrameRate,
            self._cam.incFrameRate.value * 100)
        self.cam_framerate_slider.elementChanged[int, float] \
            .connect(self.cam_framerate_value_changed)

        # framerate text box control
        self.cam_framerate_ledit = QLineEdit("{:.6f}".format(
            self._cam.currentFrameRate.value))
        self.cam_framerate_ledit.setCompleter(
            QCompleter(map("{:.6f}".format, self.cam_framerate_slider.values)))
        self.cam_framerate_ledit.setValidator(QDoubleValidator())
        self.cam_framerate_ledit.returnPressed \
            .connect(self.cam_framerate_return)

        self.cam_framerate_ledit.returnPressed.emit()

        # exposure slider control
        self.cam_exposure_lbl = DragLabel(
            "Exposure ms", parent_name='cam_exposure_slider')
        self.cam_exposure_slider = qlist_slider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_exposure_slider.values = np.arange(
            self._cam.exposure_range[0],
            self._cam.exposure_range[1],
            self._cam.exposure_range[2].value)
        self.cam_exposure_slider.elementChanged[int, float] \
            .connect(self.cam_exposure_value_changed)

        # exposure text box control
        self.cam_exposure_ledit = QLineEdit("{:.6f}".format(
            self._cam.exposure_current.value))
        self.cam_exposure_ledit.setCompleter(
            QCompleter(map("{:.6f}".format, self.cam_exposure_slider.values)))
        self.cam_exposure_ledit.setValidator(QDoubleValidator())
        self.cam_exposure_ledit.returnPressed.connect(self.cam_exposure_return)

        # trigger mode combobox
        self.cam_trigger_mode_lbl = QLabel("Trigger Mode")
        self.cam_trigger_mode_cbox = QComboBox()
        self.cam_trigger_mode_cbox.addItems(IDS_Camera.TRIGGER_MODES.keys())
        self.cam_trigger_mode_cbox.currentIndexChanged[str] \
            .connect(self.cam_trigger_cbox_changed)

        # flash mode combobox
        self.cam_flash_mode_lbl = QLabel("Flash Mode")
        self.cam_flash_mode_cbox = QComboBox()
        self.cam_flash_mode_cbox.addItems(IDS_Camera.FLASH_MODES.keys())
        self.cam_flash_mode_cbox.currentIndexChanged[str] \
            .connect(self.cam_flash_cbox_changed)

        # flash duration slider
        self.cam_flash_duration_lbl = QLabel("Flash Duration us")
        self.cam_flash_duration_slider = qlist_slider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_flash_duration_slider.values = \
            np.append([0], np.arange(self._cam.flash_min.u32Duration.value,
                      self._cam.flash_max.u32Duration.value,
                      self._cam.flash_inc.u32Duration.value))
        self.cam_flash_duration_slider.elementChanged[int, int] \
            .connect(self.cam_flash_duration_value_changed)

        # flash duration text box
        self.cam_flash_duration_ledit = QLineEdit("{:d}".format(
            self._cam.flash_cur.u32Duration.value))
        self.cam_flash_duration_ledit.setValidator(QIntValidator())
        self.cam_flash_duration_ledit.returnPressed \
            .connect(self.cam_flash_duration_return)

        # flash delay slider
        self.cam_flash_delay_lbl = QLabel("Flash Delay us")
        self.cam_flash_delay_slider = qlist_slider(
            orientation=Qt.Orientation.Horizontal)
        self.cam_flash_delay_slider.values = np.append([0], np.arange(
            self._cam.flash_min.s32Delay.value,
            self._cam.flash_max.s32Delay.value,
            self._cam.flash_inc.s32Delay.value))
        self.cam_flash_delay_slider.elementChanged[int, int] \
            .connect(self.cam_flash_delay_value_changed)

        # flash delay text box
        self.cam_flash_delay_ledit = QLineEdit("{:d}".format(
            self._cam.flash_cur.s32Delay.value))
        self.cam_flash_delay_ledit.setValidator(QIntValidator())
        self.cam_flash_delay_ledit.returnPressed \
            .connect(self.cam_flash_delay_return)

        # setting the highest pixel clock as default
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[-1].value))

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
        self.info_temp = QLabel(" T {:.2f} °C".format(self.cam.temperature))

        # AOI controls
        self.AOI_x_tbox = QLineEdit("0")
        self.AOI_y_tbox = QLineEdit("0")
        self.AOI_width_tbox = QLineEdit("0")
        self.AOI_height_tbox = QLineEdit("0")
        self.AOI_x_tbox.setValidator(
            QIntValidator(0, self.cam.width.value))
        self.AOI_y_tbox.setValidator(
            QIntValidator(0, self.cam.height.value))
        self.AOI_width_tbox.setValidator(
            QIntValidator(10, self.cam.width.value))
        self.AOI_height_tbox.setValidator(
            QIntValidator(10, self.cam.height.value))
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
        self.first_tab_Layout.addWidget(self.cam_pixel_clock_lbl)
        self.first_tab_Layout.addWidget(self.cam_pixel_clock_cbox)
        self.first_tab_Layout.addWidget(self.cam_trigger_mode_lbl)
        self.first_tab_Layout.addWidget(self.cam_trigger_mode_cbox)
        self.first_tab_Layout.addWidget(self.cam_framerate_lbl)
        self.first_tab_Layout.addWidget(self.cam_framerate_ledit)
        self.first_tab_Layout.addWidget(self.cam_framerate_slider)
        self.first_tab_Layout.addWidget(self.cam_exposure_lbl)
        self.first_tab_Layout.addWidget(self.cam_exposure_ledit)
        self.first_tab_Layout.addWidget(self.cam_exposure_slider)
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
    def cam(self, cam: IDS_Camera):
        '''The IDS_Camera property.

        Parameters
        ----------
        cam : IDS_Camera
            the IDS_Camera to set as panel camera.
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
        '''Sets the AOI for the slected IDS_Camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot set AOI while acquiring images!")
            return  # if acquisition is already going on

        self.cam.set_AOI(
            int(self.AOI_x_tbox.text()),
            int(self.AOI_y_tbox.text()),
            int(self.AOI_width_tbox.text()),
            int(self.AOI_height_tbox.text()))

        # setting the highest pixel clock as default
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[0].value))
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[-1].value))

    def reset_AOI(self):
        '''Resets the AOI for the slected IDS_Camera
        '''
        if self.cam.acquisition:
            QMessageBox.warning(
                self, "Warning", "Cannot reset AOI while acquiring images!")
            return  # if acquisition is already going on

        self.cam.reset_AOI()
        self.AOI_x_tbox.setText("0")
        self.AOI_y_tbox.setText("0")
        self.AOI_width_tbox.setText(str(self.cam.width.value))
        self.AOI_height_tbox.setText(str(self.cam.height.value))

        # setting the highest pixel clock as default
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[0].value))
        self.cam_pixel_clock_cbox.setCurrentText(
            str(self._cam.pixel_clock_list[-1].value))

    def center_AOI(self):
        '''Calculates the x, y values for a centered AOI'''
        self.AOI_x_tbox.setText(
            str(
                int(
                    (self.cam.rectAOI.s32Width.value -
                     int(self.AOI_width_tbox.text()))/2)))
        self.AOI_y_tbox.setText(
            str(
                int(
                    (self.cam.rectAOI.s32Height.value -
                     int(self.AOI_height_tbox.text()))/2)))

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
    def cam_trigger_cbox_changed(self, value):
        """
        Slot for changed trigger mode

        Parameters
        ----------
        Value : int
            index of selected trigger mode from IDS_Camera.TRIGGER_MODES
        """
        self._cam.set_trigger_mode(IDS_Camera.TRIGGER_MODES[value])
        self._cam.get_trigger_mode()

    @pyqtSlot(str)
    def cam_flash_cbox_changed(self, value):
        """
        Slot for changed flash mode

        Parameters
        ----------
        Value : int
            index of selected flash mode from IDS_Camera.FLASH_MODES
        """
        self._cam.set_flash_mode(IDS_Camera.FLASH_MODES[value])
        self._cam.get_flash_mode(output=True)

    @pyqtSlot(str)
    def cam_pixel_cbox_changed(self, value):
        """
        Slot for changed pixel clock

        Parameters
        ----------
        Value : int
            selected pixel clock in MHz
            (note that only certain values are allowed by camera)
        """
        self._cam.set_pixel_clock(int(value))
        self._cam.get_pixel_clock_info(False)
        self._cam.get_framerate_range(False)
        self.cam_framerate_slider.elementChanged[int, float].disconnect()
        self.cam_framerate_slider.values = np.arange(
            self._cam.minFrameRate,
            self._cam.maxFrameRate,
            self._cam.incFrameRate.value * 100)
        self.cam_framerate_slider.elementChanged[int, float] \
            .connect(self.cam_framerate_value_changed)
        self.cam_framerate_slider.setValue(
            len(self.cam_framerate_slider.values) - 1)

    @pyqtSlot(int, float)
    def cam_framerate_value_changed(self, index, value):
        """
        Slot for changed framerate

        Parameters
        ----------
        Index : ont
            selected frames per second index in the slider values list
        Value : double
            selected frames per second
        """
        self._cam.set_framerate(value)
        self._cam.get_exposure_range(False)
        self.cam_framerate_ledit.setText("{:.6f}".format(
            self._cam.currentFrameRate.value))
        self.cam_exposure_slider.elementChanged[int, float].disconnect()
        self.cam_exposure_slider.values = np.arange(
            self._cam.exposure_range[0],
            self._cam.exposure_range[1],
            self._cam.exposure_range[2].value)
        self.cam_exposure_slider.elementChanged[int, float] \
            .connect(self.cam_exposure_value_changed)
        self.cam_exposure_slider.setValue(
            len(self.cam_exposure_slider.values) - 1)

    def cam_framerate_return(self):
        """
        Sets the framerate slider with the entered text box value
        """
        self.cam_framerate_slider.setNearest(self.cam_framerate_ledit.text())

    @pyqtSlot(int, float)
    def cam_exposure_value_changed(self, index, value):
        """
        Slot for changed exposure

        Parameters
        ----------
        Index : ont
            selected exposure index in the slider values list
        Value : double
            selected exposure in milli-seconds
        """
        self._cam.set_exposure(value)
        self._cam.get_exposure(False)
        self._cam.get_flash_range(False)
        self.cam_exposure_ledit.setText("{:.6f}".format(
            self._cam.exposure_current.value))
        self.OME_tab.exposure.setText(self.cam_exposure_ledit.text())
        self.cam_flash_duration_slider.values = np.append([0], np.arange(
            self._cam.flash_min.u32Duration.value,
            self._cam.flash_max.u32Duration.value,
            self._cam.flash_inc.u32Duration.value))
        self.cam_flash_delay_slider.values = np.append([0], np.arange(
            self._cam.flash_min.s32Delay.value,
            self._cam.flash_max.s32Delay.value,
            self._cam.flash_inc.s32Delay.value))
        if self.master:
            self.exposureChanged.emit()

    def cam_exposure_return(self):
        """
        Sets the exposure slider with the entered text box value
        """
        self.cam_exposure_slider.setNearest(self.cam_exposure_ledit.text())

    @pyqtSlot(int, int)
    def cam_flash_duration_value_changed(self, index, value):
        """
        Slot for changed flash duration

        Parameters
        ----------
        Index : ont
            selected flash duration index in the slider values list
        Value : double
            selected flash duration in micro-seconds
        """
        self._cam.set_flash_params(self._cam.flash_cur.s32Delay.value, value)
        self._cam.get_flash_params()
        self.cam_flash_duration_ledit.setText("{:d}".format(
            self._cam.flash_cur.u32Duration.value))

    def cam_flash_duration_return(self):
        """
        Sets the flash duration slider with the entered text box value
        """
        self.cam_flash_duration_slider.setNearest(
            self.cam_flash_duration_ledit.text())

    @pyqtSlot(int, int)
    def cam_flash_delay_value_changed(self, index, value):
        """
        Slot for changed flash delay

        Parameters
        ----------
        Index : ont
            selected flash delay index in the slider values list
        Value : double
            selected flash delay in micro-seconds
        """
        self._cam.set_flash_params(
            value, self._cam.flash_cur.u32Duration.value)
        self._cam.get_flash_params()
        self.cam_flash_delay_ledit.setText("{:d}".format(
            self._cam.flash_cur.s32Delay.value))

    def cam_flash_delay_return(self):
        """
        Sets the flash delay slider with the entered text box value
        """
        self.cam_flash_delay_slider.setNearest(
            self.cam_flash_delay_ledit.text())

    def start_free_run(self, cam: IDS_Camera):
        """
        Starts free run acquisition mode

        Parameters
        ----------
        cam : IDS_Camera
            IDS Camera python adapter
        """
        nRet = 0
        if cam.acquisition:
            return  # if acquisition is already going on

        if not cam.memory_allocated:
            cam.allocate_memory_buffer()  # allocate memory

        if not cam.capture_video:
            cam.start_live_capture()  # start live capture (freerun mode)

        cam.refresh_info()  # refresh adapter info

        self._save_path = (self._directory + "\\"
                           + self.experiment_name.text()
                           + "\\" + self.cam.name
                           + time.strftime("_%Y_%m_%d_%H%M%S"))

        # if not os.path.exists(self._save_path):
        #     os.makedirs(self._save_path)

        cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers(nRet)

    def start_software_triggered(self, cam: IDS_Camera):
        """
        Starts trigger acquisition mode

        Parameters
        ----------
        cam : IDS_Camera
            IDS Camera python adapter
        """

        nRet = 0
        if cam.acquisition:
            return  # if acquisition is already going on

        if not cam.memory_allocated:
            cam.allocate_memory_buffer()  # allocate memory

        # nRet = cam.enable_queue_mode()  # enable queue mode

        cam.refresh_info()  # refresh adapter info

        self._save_path = (self._directory + "\\"
                           + self.experiment_name.text() + "\\"
                           + self.cam.name
                           + time.strftime("_%Y_%m_%d_%H%M%S"))

        # if not os.path.exists(self._save_path):
        #     os.makedirs(self._save_path)

        cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers(nRet)

    def stop(self):
        self._stop_thread = True  # set stop acquisition workers flag to true

    def start_all_workers(self, nRet):
        """
        Starts all workers

        Parameters
        ----------
        nRet : int
            IDS Camera error/success code
        """

        self._stop_thread = False  # set stop acquisition workers flag to false

        # Pass the display function to be executed
        if self.d_worker is None or self.d_worker.done:
            self.d_worker = thread_worker(
                self._display, nRet, self.cam, progress=False, z_stage=False)
            # Execute
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.done:
                self.s_worker = thread_worker(
                    self._save, nRet, progress=False, z_stage=False)
                # Execute
                self._threadpool.start(self.s_worker)

        #  Pass the capture function to be executed
        if self.c_worker is None or self.c_worker.done:
            # Any other args, kwargs are passed to the run function
            self.c_worker = thread_worker(
                self._capture, nRet, self.cam, progress=False, z_stage=False)
            # Execute
            self._threadpool.start(self.c_worker)

        # Giving the capture thread a head start over the display one
        QThread.msleep(500)

    def start_dis_save_workers(self, nRet):
        """
        Starts both the display and save workers only

        Parameters
        ----------
        nRet : int
            IDS Camera error/success code
        """
        self._stop_thread = False  # set stop acquisition workers flag to false

        # Pass the display function to be executed
        if self.d_worker is None or self.d_worker.done:
            self.d_worker = thread_worker(
                self._display, nRet, self.cam, progress=False, z_stage=False)
            # Execute
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.done:
                self.s_worker = thread_worker(
                    self._save, nRet, progress=False, z_stage=False)
                # Execute
                self._threadpool.start(self.s_worker)

    def _capture(self, nRet, cam: IDS_Camera):
        '''Capture function executed by the capture worker.

        Sends software trigger signals and transfers the
        acquired frame to the display processing stack.

        Parameters
        ----------
        nRet : int
            return code from IDS_Camera, ueye.IS_SUCCESS = 0 to run.
        cam : IDS_Camera
            the IDS_Camera used to acquire frames.
        '''
        try:
            self._buffer.queue.clear()
            self._temps.queue.clear()
            self._frames.queue.clear()
            self._counter = 0

            temp = np.zeros(
                (cam.height.value * cam.width.value * cam.bytes_per_pixel))
            time = QDateTime.currentDateTime()
            nFrames = int(self.frames_tbox.text())
            # Continuous image capture
            while(nRet == ueye.IS_SUCCESS):
                self._exec_time = time.msecsTo(QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                if not cam.capture_video:
                    ueye.is_FreezeVideo(cam.hCam, ueye.IS_WAIT)

                nRet = cam.is_WaitForNextImage(500, not self.mini)
                if nRet == ueye.IS_SUCCESS:
                    cam.get_pitch()
                    data = cam.get_data()

                    self._buffer.put(data.copy())
                    # add sensor temperature to the stack
                    self._temps.put(cam.get_temperature())
                    temp = data.copy()
                    self._counter = self._counter + 1
                    if self._counter >= nFrames and not self.mini:
                        self._stop_thread = True
                        logging.debug('Stop')
                    cam.unlock_buffer()

                QThread.usleep(100)  # sleep 100us

                if self._stop_thread:
                    break  # in case stop threads is initiated
        except Exception:
            traceback.print_exc()
        finally:
            # reset flags and release resources
            cam.acquisition = False
            if cam.capture_video:
                cam.stop_live_capture()
            cam.free_memory()
            if self._dispose_cam:
                cam.dispose()

    def _display(self, nRet, cam: IDS_Camera):
        '''Display function executed by the display worker.

        Processes the acquired frame, displays it, and sends it to the save
        stack.

        Parameters
        ----------
        nRet : int
            return code from IDS_Camera, ueye.IS_SUCCESS = 0 to run.
        cam : IDS_Camera
            the IDS_Camera used to acquire frames.
        '''
        try:
            time = QDateTime.currentDateTime()
            # Continuous image display
            while(nRet == ueye.IS_SUCCESS):
                # for display time estimations

                # proceed only if the buffer is not empty
                if not self._buffer.empty():
                    self._dis_time = time.msecsTo(QDateTime.currentDateTime())
                    time = QDateTime.currentDateTime()

                    # reshape image into proper shape
                    # (height, width, bytes per pixel)
                    frame = uImage.fromBuffer(
                        self._buffer.get(),
                        cam.height.value, cam.width.value, cam.bytes_per_pixel)

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

    def _save(self, nRet):
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
            while(nRet == ueye.IS_SUCCESS):
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
            self, "Save config", filter="JSON Files (*.json);;")

        if len(filename) > 0:

            config = {
                'model': self.cam.sInfo.strSensorName.decode('utf-8'),
                'serial': self.cam.cInfo.SerNo.decode('utf-8'),
                'clock speed': self.cam.pixel_clock.value,
                'trigger': self.cam_trigger_mode_cbox.currentText(),
                'flash mode': self.cam_flash_mode_cbox.currentText(),
                'framerate': self.cam.currentFrameRate.value,
                'exposure': self.cam.exposure_current.value,
                'flash duration': self.cam.flash_cur.u32Duration.value,
                'flash delay': self.cam.flash_cur.s32Delay.value,
                'AOI w': self.cam.set_rectAOI.s32Width.value,
                'AOI h': self.cam.set_rectAOI.s32Height.value,
                'AOI x': self.cam.set_rectAOI.s32X.value,
                'AOI y': self.cam.set_rectAOI.s32Y.value,
                'Zoom': self._zoom,
            }

            with open(filename, 'w') as file:
                json.dump(config, file)

            QMessageBox.information(
                self, "Info", "Config saved.")
        else:
            QMessageBox.warning(
                self, "Warning", "Config not saved.")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load config", filter="JSON Files (*.json);;")

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
                'AOI w',
                'AOI h',
                'AOI x',
                'AOI y',
                'Zoom',
            ]
            with open(filename, 'r') as file:
                config = json.load(file)
            if all(key in config for key in keys):
                if self.cam.sInfo.strSensorName.decode('utf-8') == \
                        config['model']:
                    self.AOI_x_tbox.setText(str(config['AOI x']))
                    self.AOI_y_tbox.setText(str(config['AOI y']))
                    self.AOI_width_tbox.setText(str(config['AOI w']))
                    self.AOI_height_tbox.setText(str(config['AOI h']))
                    self.set_AOI()

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

                    self._zoom = float(config['Zoom'])
                    self.zoom_lbl.setText(
                        "Resize " + "{:.0f}%".format(self._zoom*100))
                else:
                    QMessageBox.warning(
                        self, "Warning", "Camera model is different.")
            else:
                QMessageBox.warning(
                    self, "Warning", "Wrong or corrupted config file.")
        else:
            QMessageBox.warning(
                self, "Warning", "No file selected.")
