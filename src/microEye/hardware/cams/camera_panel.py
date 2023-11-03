import logging
import math
import os
import threading
import time
import traceback
from queue import Queue

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ...metadata import MetadataEditor
from ...qlist_slider import *
from ...thread_worker import *
from ...uImage import uImage
from .jobs import *
from .line_profiler import LineProfiler
from .micam import *


class Camera_Panel(QGroupBox):
    """
    A Qt Widget base class for controlling a camera | Inherits QGroupBox
    """

    exposureChanged = pyqtSignal()

    def __init__(self, threadpool: QThreadPool, cam: miCamera, mini=False,
                 *args, **kwargs):
        """
        Initializes a new Vimba_Panel Qt widget
        | Inherits QGroupBox

        Parameters
        ----------
        threadpool : QThreadPool
            The threadpool for multithreading
        cam : miCamera
            Camera python adapter
        """
        super().__init__(*args, **kwargs)
        self._cam = cam  # miCamera
        # flag true if master (always the first added camera is master)
        self.master = False

        self.c_worker = None  # worker for capturing
        self.d_worker = None  # worker for display
        self.s_worker = None  # worker for saving

        self.c_event = None
        self.s_event = None

        # number of bins for the histogram (4096 is set for 12bit mono-camera)
        self._nBins = 2**12
        self._hist = np.arange(0, self._nBins) * 0  # arrays for the Hist plot
        self._bins = np.arange(0, self._nBins)      # arrays for the Hist plot

        self._threadpool = threadpool  # the threadpool for workers

        # flag true to close camera adapter and dispose it
        self._dispose_cam = False
        self._buffer = Queue()  # data memory stack
        self._temps = Queue()  # camera sensor temperature stack
        # stack of (frame, temperature) tuples (for saving)
        self.frame = None
        self._frames = Queue()
        # reserved for saving
        self._directory = ""  # save directory
        self._save_path = ""  # save path
        self._save_prefix = ""  # save prefix

        # acquisition job info class
        self.acq_job = None

        self.mini = mini

        # Line Profiler
        self.lineProfiler = LineProfiler()

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

        # exposure text box control
        self.cam_exposure_qs = QDoubleSpinBox()

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
                clicked=lambda: self.setExposure(1))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '5ms',
                clicked=lambda: self.setExposure(5))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '10ms',
                clicked=lambda: self.setExposure(10))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '30ms',
                clicked=lambda: self.setExposure(30))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '50ms',
                clicked=lambda: self.setExposure(50))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '100ms',
                clicked=lambda: self.setExposure(100))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '150ms',
                clicked=lambda: self.setExposure(150))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '200ms',
                clicked=lambda: self.setExposure(200))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '500ms',
                clicked=lambda: self.setExposure(500))
        )
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                '1s',
                clicked=lambda: self.setExposure(1000))
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

        self.frames_tbox = QSpinBox()
        self.frames_tbox.setMaximum(1e9)
        self.frames_tbox.setMinimum(1)
        self.frames_tbox.setValue(1e6)

        # save to hard drive checkbox
        self.save_data_chbx = QCheckBox("Save Data")
        self.save_data_chbx.setChecked(self.mini)
        self.dark_cal = QCheckBox("Dark Calibration")
        self.dark_cal.setToolTip('Generates mean and variance images')
        self.dark_cal.setChecked(False)
        self.save_formats_layout = QHBoxLayout()
        self.save_as_tiff = QRadioButton('Tiff')
        self.save_as_zarr = QRadioButton('Zarr')
        self.save_formats = QButtonGroup()
        self.save_formats.addButton(self.save_as_tiff)
        self.save_formats.addButton(self.save_as_zarr)
        self.save_formats_layout.addWidget(self.save_as_tiff)
        self.save_formats_layout.addWidget(self.save_as_zarr)
        self.save_as_tiff.setChecked(True)
        self.save_bigg_tiff = QCheckBox("BiggTiff")
        self.save_bigg_tiff.setChecked(True)
        self.cam_save_meta = QCheckBox("Full OME-XML")
        self.cam_save_meta.setChecked(self.mini)

        # preview checkbox
        self.preview_ch_box = QCheckBox("Preview")
        self.preview_ch_box.setChecked(not self.mini)

        self.line_profiler_ch_box = QCheckBox("Line Profiler")
        self.line_profiler_ch_box.setChecked(False)
        self.line_profiler_ch_box.stateChanged.connect(
            lambda: self.lineProfiler.show()
            if self.line_profiler_ch_box.isChecked()
            else self.lineProfiler.hide()
        )

        self.single_view_rbtn = QRadioButton("Single View")
        self.single_view_rbtn.setChecked(True)
        self.dual_view_rbtn = QRadioButton("Dual Channel (Side by Side).")
        self.dual_view_overlap_rbtn = QRadioButton(
            "Dual Channel (Overlapped).")
        self.view_btns = QButtonGroup()
        self.view_btns.addButton(self.single_view_rbtn)
        self.view_btns.addButton(self.dual_view_rbtn)
        self.view_btns.addButton(self.dual_view_overlap_rbtn)
        self.view_btns.buttonToggled.connect(self.view_toggled)

        self.view_rbtns = QHBoxLayout()
        self.view_rbtns.addWidget(self.single_view_rbtn)
        self.view_rbtns.addWidget(self.dual_view_rbtn)
        self.view_rbtns.addWidget(self.dual_view_overlap_rbtn)

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
        # autostretch checkbox
        self.auto_stretch = QCheckBox("Auto Stretch")
        self.auto_stretch.setChecked(True)
        # Hist plotWidget
        self.histogram = pg.PlotWidget()
        self.hist_cdf = pg.PlotWidget()
        greenP = pg.mkPen(color='g')
        blueP = pg.mkPen(color='b')
        greenB = pg.mkBrush(0, 255, 0, 32)
        blueB = pg.mkBrush(0, 0, 255, 32)
        self._plot_ref = self.histogram.plot(
            self._bins, self._hist, pen=greenP)
        self._cdf_plot_ref = self.hist_cdf.plot(
            self._bins, self._hist, pen=greenP)
        self._plot_ref_2 = self.histogram.plot(
            self._bins, self._hist, pen=blueP)
        self._cdf_plot_ref_2 = self.hist_cdf.plot(
            self._bins, self._hist, pen=blueP)

        self.lr_0 = pg.LinearRegionItem(
            (0, self._nBins),
            bounds=(0, self._nBins),
            pen=greenP, brush=greenB,
            movable=True, swapMode="push", span=(0.0, 1))
        self.lr_1 = pg.LinearRegionItem(
            (0, self._nBins),
            bounds=(0, self._nBins),
            pen=blueP, brush=blueB,
            movable=True, swapMode="push", span=(1, 1))
        self.histogram.addItem(self.lr_0)
        self.histogram.addItem(self.lr_1)

        # Stats. info
        self.info_cap = QLabel("Capture {:d} | {:.2f} ms ".format(
            0, 0))
        self.info_disp = QLabel("Display {:d} | {:.2f} ms ".format(
            self._buffer.qsize(), 0))
        self.info_save = QLabel("Save {:d} | {:.2f} ms ".format(
            self._frames.qsize(), 0))
        self.info_temp = QLabel(
            " T {:.2f} °C".format(-127))

        # AOI controls
        self.AOI_x_tbox = QSpinBox()
        self.AOI_y_tbox = QSpinBox()
        self.AOI_width_tbox = QSpinBox()
        self.AOI_height_tbox = QSpinBox()

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

        # adding widgets to the main layout

        if not self.mini:
            self.second_tab_Layout.addRow(self.preview_ch_box)
            self.second_tab_Layout.addRow(QLabel('View Options:'))
            self.second_tab_Layout.addRow(self.view_rbtns)
            self.second_tab_Layout.addRow(self.line_profiler_ch_box)

        self.second_tab_Layout.addRow(self.slow_lut_rbtn)
        self.second_tab_Layout.addRow(self.fast_lut_rbtn)
        if not self.mini:
            self.second_tab_Layout.addRow(
                self.zoom_lbl,
                self.zoom_box)

        self.second_tab_Layout.addRow(
            self.histogram_lbl,
            self.auto_stretch)
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

    def addFirstTabItems(self):
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
            self.first_tab_Layout.addWidget(self.save_data_chbx)
            self.first_tab_Layout.addRow(
                QLabel("Format:"),
                self.save_formats_layout)
            self.first_tab_Layout.addWidget(self.dark_cal)
            self.first_tab_Layout.addWidget(self.save_bigg_tiff)
            self.first_tab_Layout.addWidget(self.cam_save_meta)

    @property
    def cam(self):
        '''The Camera property.

        Returns
        -------
        object
            the cam property value
        '''
        return self._cam

    @cam.setter
    def cam(self, cam: miCamera):
        '''The Camera property.

        Parameters
        ----------
        cam : object
            the object to set as panel camera.
        '''
        self._cam = cam

    @property
    def buffer(self):
        if self._frames is not None:
            return self._frames
        else:
            return Queue()

    @property
    def bufferSize(self):
        if self._frames is not None:
            return self._frames.qsize()
        else:
            return 0

    @property
    def isEmpty(self) -> bool:
        return self._frames.empty()

    def get(self, last=False) -> np.ndarray:
        res = None
        if not self._frames.empty():
            if last:
                while not self.isEmpty:
                    res = self._frames.get()
            else:
                res = self._frames.get()
        else:
            res = np.random.normal(size=(256, 256))
        return res

    def setExposure(self, value):
        self.cam_exposure_qs.setValue(value)

    @property
    def isOpen(self) -> bool:
        return self.cam.acquisition

    def set_AOI(self):
        '''Sets the AOI for the slected vimba_cam
        '''
        pass

    def reset_AOI(self):
        '''Resets the AOI for the slected IDS_Camera
        '''
        pass

    def center_AOI(self):
        '''Sets the AOI for the slected vimba_cam
        '''
        pass

    def select_AOI(self):
        pass

    def get_meta(self):
        meta = dict[str, any]()
        return meta

    def getAcquisitionJob(self) -> AcquisitionJob:
        self._save_path = (self._directory + "/"
                           + self.experiment_name.text() + "/")
        return AcquisitionJob(
            self._temps, self._buffer, self._frames,
            self._save_path, self._cam.getHeight(),
            self._cam.getWidth(),
            biggTiff=self.save_bigg_tiff.isChecked(),
            bytes_per_pixel=self._cam.bytes_per_pixel,
            exposure=self._cam.getExposure(),
            frames=self.frames_tbox.value(),
            full_tif_meta=self.cam_save_meta.isChecked(),
            is_dark_cal=self.dark_cal.isChecked(),
            meta_file=self.get_meta(),
            meta_func=self.OME_tab.gen_OME_XML if
            self.cam_save_meta.isChecked() else self.OME_tab.gen_OME_XML_short,
            name=self._cam.name, prefix=self._save_prefix,
            save=self.save_data_chbx.isChecked(),
            Zarr=self.save_as_zarr.isChecked()
        )

    def view_toggled(self, params):
        if self.single_view_rbtn.isChecked():
            self.lr_0.setSpan(0, 1)
            self.lr_1.setMovable(False)
            self.lr_1.setRegion((0, self._nBins))
            self.lr_1.setSpan(1.0, 1.0)
        else:
            self.lr_0.setSpan(0, 0.5)
            self.lr_1.setMovable(True)
            self.lr_1.setSpan(0.5, 1.0)

    @pyqtSlot()
    def save_browse_clicked(self):
        """Slot for browse clicked event"""
        self._directory = ""

        while len(self._directory) == 0:
            self._directory = str(
                QFileDialog.getExistingDirectory(self, "Select Directory"))

        self.save_dir_edit.setText(self._directory)

    def start_free_run(self, Prefix=''):
        """
        Starts free run acquisition mode

        Parameters
        ----------
        Prefix : str
            an extra prefix added to the image stack file name
        """
        if self._cam.acquisition:
            return  # if acquisition is already going on

        self._save_prefix = Prefix
        self.acq_job = self.getAcquisitionJob()

        self._cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers()

    def stop(self):
        if self.c_event is not None:
            self.c_event.set()
        # set stop acquisition workers flag to true
        if self.acq_job is not None:
            self.acq_job.stop_threads = True

    def start_all_workers(self):
        """
        Starts all workers
        """

        self._buffer.queue.clear()
        self._temps.queue.clear()
        self._frames.queue.clear()
        self.c_event = threading.Event()
        self.s_event = threading.Event()
        self.time = QDateTime.currentDateTime()

        # Pass the display function to be executed
        if self.d_worker is None or self.d_worker.done:
            self.d_worker = thread_worker(
                cam_display,
                self.acq_job, self,
                progress=False, z_stage=False)

            self.d_worker.signals.finished.connect(
                lambda: self.acq_job.setDone(1, True)
            )
            # Execute
            self.d_worker.setAutoDelete(True)
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.done:
                self.s_worker = thread_worker(
                    cam_save,
                    self.acq_job,
                    progress=False, z_stage=False)
                self.s_worker.setAutoDelete(True)
                # Execute
                self._threadpool.start(self.s_worker)

        #  Pass the capture function to be executed
        if self.c_worker is None or self.c_worker.done:
            # Any other args, kwargs are passed to the run function
            self.c_worker = thread_worker(
                self.cam_capture,
                *self.getCaptureArgs(),
                progress=False, z_stage=False)
            self.c_worker.signals.finished.connect(
                lambda: self.acq_job.setDone(0, True)
            )
            self.c_worker.setAutoDelete(True)
            # Execute
            self._threadpool.start(self.c_worker)

    def updateInfo(self):
        if isinstance(self.acq_job, AcquisitionJob):
            self.info_temp.setText(
                    " T {:.2f} °C".format(self._cam.temperature))
            self.info_cap.setText(
                " Capture {:d}/{:d} {:.2%} | {:.2f} ms ".format(
                    self.acq_job.frames_captured,
                    self.acq_job.frames,
                    self.acq_job.frames_captured / self.acq_job.frames,
                    self.acq_job.capture_time))
            self.info_disp.setText(
                " Display {:d} | {:.2f} ms ".format(
                    self._buffer.qsize(), self.acq_job.display_time))
            self.info_save.setText(
                " Save {:d} | {:.2f} ms ".format(
                    self._frames.qsize(), self.acq_job.save_time))

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
        raise NotImplementedError(
            "The getCaptureArgs function is not implemented yet.")

    def cam_capture(self, *args):
        '''User specific implemented logic for frame capture
        from a miCamera adapter.

        Example
        ------
        check child panels (vimba, ueye or thorlabs)

        Raises
        ------
        NotImplementedError
            Has to be implemented by the use in child class.
        '''
        raise NotImplementedError(
            "The cam_capture function is not implemented yet.")


def cam_save(params: AcquisitionJob):
    '''Save function executed by the save worker thread.

    Saves the acquired frames using the config in AcquisitionJob
    with a csv file containing the sensor temp for each frame.

    Parameters
    ----------
    params : AcquisitionJob
        class holding cross-thread info required for an Acquisition Job.
    '''
    try:
        while True:
            # save in case frame stack is not empty
            if not params.save_queue.empty():
                # for save time estimations
                time = QDateTime.currentDateTime()

                # get frame and temp to save from bottom of stack
                frame, temp = params.save_queue.get()

                if params.frames_saved < params.frames:
                    # opens csv file and appends sensor temp and close
                    params.addTemp(temp)
                    # adds dark frame for camera calibration if needed
                    params.addDarkFrame(frame)
                    # saves frame to storage
                    params.addFrame(frame)

                # for save time estimations
                with params.lock:
                    params.save_time = time.msecsTo(
                        QDateTime.currentDateTime())

                params.frames_saved += 1

            QThread.usleep(100)
            # Flag that ends the loop
            if params.save_queue.empty() and \
                    params.stop_threads and params.display_done:
                print('Save thread break.')
                break
    except Exception:
        traceback.print_exc()
    finally:
        params.finalize()
        print('Save thread finally finished.')


def cam_display(params: AcquisitionJob, camp: Camera_Panel):
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
        while (True):

            # proceed only if the buffer is not empty
            if not params.display_queue.empty():
                # for display time estimations
                with params.lock:
                    params.display_time = time.msecsTo(
                        QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                # reshape image into proper shape
                # (height, width, bytes per pixel)
                with params.lock:
                    params.frame = uImage.fromBuffer(
                        params.display_queue.get(),
                        params.height, params.width,
                        params.bytes_per_pixel)

                # add to saving stack
                if params.save:
                    if not camp.mini:
                        params.save_queue.put(
                            (params.frame.image, params.temp_queue.get()))
                    else:
                        params.save_queue.put(params.frame.image)

                if camp.preview_ch_box.isChecked():
                    if camp.line_profiler_ch_box.isChecked():
                        camp.lineProfiler.setData(params.frame._image)

                    if camp.single_view_rbtn.isChecked():
                        _range = None

                        # image stretching
                        if not camp.auto_stretch.isChecked():
                            _range = tuple(
                                map(math.ceil, camp.lr_0.getRegion()))

                        params.frame.equalizeLUT(
                            _range, camp.slow_lut_rbtn.isChecked())

                        if camp.auto_stretch.isChecked():
                            camp.lr_0.setRegion(
                                (params.frame._min, params.frame._max))
                            camp.histogram.setXRange(
                                params.frame._min, params.frame._max)
                            camp.hist_cdf.setXRange(
                                params.frame._min, params.frame._max)

                        camp._plot_ref.setData(params.frame._hist[:, 0])
                        camp._cdf_plot_ref.setData(params.frame._cdf)
                        camp._plot_ref_2.setData(camp._hist)
                        camp._cdf_plot_ref_2.setData(camp._hist)

                        # resizing the image
                        params.frame._view = cv2.resize(
                            params.frame._view, (0, 0),
                            fx=camp.zoom_box.value(),
                            fy=camp.zoom_box.value(),
                            interpolation=cv2.INTER_NEAREST)

                        # display it
                        cv2.imshow(params.name, params.frame._view)
                    else:
                        _rang_left = None
                        _rang_right = None

                        # image stretching
                        if not camp.auto_stretch.isChecked():
                            _rang_left = tuple(
                                map(math.ceil, camp.lr_0.getRegion()))
                            _rang_right = tuple(
                                map(math.ceil, camp.lr_1.getRegion()))

                        left, right = params.frame.hsplitView()

                        left.equalizeLUT(
                            _rang_left, camp.slow_lut_rbtn.isChecked())
                        right.equalizeLUT(
                            _rang_right, camp.slow_lut_rbtn.isChecked())
                        if camp.auto_stretch.isChecked():
                            camp.lr_0.setRegion((left._min, left._max))
                            camp.lr_1.setRegion((right._min, right._max))
                            camp.histogram.setXRange(
                                min(left._min, right._min),
                                max(left._max, right._max))
                            camp.hist_cdf.setXRange(
                                min(left._min, right._min),
                                max(left._max, right._max))

                        camp._plot_ref.setData(left._hist[:, 0])
                        camp._cdf_plot_ref.setData(left._cdf)
                        camp._plot_ref_2.setData(right._hist[:, 0])
                        camp._cdf_plot_ref_2.setData(right._cdf)

                        if camp.dual_view_overlap_rbtn.isChecked():
                            _img = np.zeros(
                                left._view.shape[:2] + (3,),
                                dtype=np.uint8)
                            _img[..., 1] = left._view
                            _img[..., 0] = right._view
                            _img = cv2.resize(
                                _img,
                                (0, 0),
                                fx=camp.zoom_box.value(),
                                fy=camp.zoom_box.value(),
                                interpolation=cv2.INTER_NEAREST)

                            cv2.imshow(params.name, _img)
                        else:
                            _img = cv2.resize(
                                np.concatenate(
                                    [left._view, right._view], axis=1),
                                (0, 0),
                                fx=camp.zoom_box.value(),
                                fy=camp.zoom_box.value(),
                                interpolation=cv2.INTER_NEAREST)

                            cv2.imshow(params.name, _img)
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(1)
            else:
                cv2.waitKey(1)

            QThread.usleep(100)
            # Flag that ends the loop
            if params.stop_threads and params.display_queue.empty() \
                    and camp.c_worker.done:
                print('Display thread break.')
                break
    except Exception:
        traceback.print_exc()
    finally:
        try:
            if not camp.mini:
                cv2.destroyWindow(params.name)
        except Exception:
            traceback.print_exc()
        print('Display thread finally finished.')
