import logging
import math
import os
import threading
import time
import traceback
from enum import Enum
from queue import Queue

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from ...shared.expandable_groupbox import ExpandableGroupBox
from ...shared.metadata_tree import MetadataEditorTree, MetaParams
from ...shared.thread_worker import *
from ...shared.uImage import uImage
from ..widgets.qlist_slider import *
from .camera_options import CameraOptions, CamParams
from .jobs import *
from .line_profiler import LineProfiler
from .micam import *

EXPOSURE_SHORTCUTS = [
    1, 5, 10, 20, 30, 50, 100, 150, 200, 300, 500, 1000]

class Camera_Panel(QGroupBox):
    '''
    A Qt Widget base class for controlling a camera | Inherits QGroupBox
    '''

    exposureChanged = pyqtSignal()

    def __init__(self, threadpool: QThreadPool, cam: miCamera, mini=False,
                 *args, **kwargs):
        '''
        Initializes a new Vimba_Panel Qt widget
        | Inherits QGroupBox

        Parameters
        ----------
        threadpool : QThreadPool
            The threadpool for multithreading
        cam : miCamera
            Camera python adapter
        '''
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
        self._directory = ''  # save directory
        self._save_path = ''  # save path
        self._save_prefix = ''  # save prefix

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

        self.OME_tab = MetadataEditorTree()

        # add tabs
        self.main_tab_view.addTab(self.first_tab, 'Main')
        if not self.mini:
            self.main_tab_view.addTab(self.OME_tab, 'OME-XML metadata')

        # first tab vertical layout
        self.first_tab_Layout = QFormLayout()
        # set as first tab layout
        self.first_tab.setLayout(self.first_tab_Layout)

        self.camera_options = CameraOptions()


        self.cam_exp_shortcuts = QHBoxLayout()
        self.cam_exp_shortcuts.addWidget(
            QPushButton(
                'min',
                clicked=lambda: self.camera_options.set_param_value(
                    CamParams.EXPOSURE,
                    self._cam.exposure_range[0]))
        )

        def add_button(layout: QHBoxLayout, value):
            layout.addWidget(
                QPushButton(
                    f'{value:.0f} ms',
                    clicked=lambda: self.setExposure(value))
            )

        for value in EXPOSURE_SHORTCUTS:
            add_button(self.cam_exp_shortcuts, value)

        self.first_tab_Layout.addRow(self.cam_exp_shortcuts)
        self.first_tab_Layout.addRow(self.camera_options)

        self.camera_options.get_param(CamParams.EXPERIMENT_NAME).sigValueChanged.connect(
            lambda x: self.OME_tab.set_param_value(
                MetaParams.EXPERIMENT_NAME, x))

        self.save_dir_layout = QHBoxLayout()

        self._directory = os.path.dirname(os.path.realpath(__file__))
        self.camera_options.set_param_value(
            CamParams.SAVE_DIRECTORY, self._directory)
        self.camera_options.directoryChanged.connect(
            lambda value: self.directory_changed(value))

        save_param = self.camera_options.get_param(CamParams.SAVE_DATA)
        save_param.setOpts(enabled=not self.mini)

        preview_param = self.camera_options.get_param(CamParams.PREVIEW)
        preview_param.setValue(not self.mini)
        preview_param.setOpts(enabled=not self.mini)

        self.camera_options.get_param(CamParams.DISPLAY_STATS_OPTION).setOpts(
            enabled=not self.mini)
        self.camera_options.get_param(CamParams.VIEW_OPTIONS).setOpts(
            visible=not self.mini)
        self.camera_options.get_param(CamParams.RESIZE_DISPLAY).setOpts(
            enabled=not self.mini)

        profiler_param = self.camera_options.get_param(CamParams.LINE_PROFILER)
        profiler_param.setOpts(enabled=not self.mini)
        profiler_param.sigStateChanged.connect(
            lambda: self.lineProfiler.show()
            if self.camera_options.get_param_value(CamParams.LINE_PROFILER)
            else self.lineProfiler.hide()
        )

        self.camera_options.setROI.connect(lambda: self.set_ROI())
        self.camera_options.resetROI.connect(lambda: self.reset_ROI())
        self.camera_options.centerROI.connect(lambda: self.center_ROI())
        self.camera_options.selectROI.connect(lambda: self.select_ROI())
        self.camera_options.selectROIs.connect(lambda: self.select_ROIs())

        # controls for histogram and cdf
        self.histogram_group = ExpandableGroupBox('Histogram')
        self.first_tab_Layout.addRow(self.histogram_group)
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
            movable=True, swapMode='push', span=(0.0, 1))
        self.lr_1 = pg.LinearRegionItem(
            (0, self._nBins),
            bounds=(0, self._nBins),
            pen=blueP, brush=blueB,
            movable=True, swapMode='push', span=(1, 1))
        self.histogram.addItem(self.lr_0)
        self.histogram.addItem(self.lr_1)

        self.histogram_group.layout().addWidget(self.histogram)
        self.histogram_group.layout().addWidget(self.hist_cdf)

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
        '''Gets the frames Queue

        Returns
        -------
        Queue
            Frames buffer.
        '''
        if self._frames is not None:
            return self._frames
        else:
            return Queue()

    @property
    def bufferSize(self):
        '''Gets frames Queue size.

        Returns
        -------
        int
            Frames Queue size.
        '''
        if self._frames is not None:
            return self._frames.qsize()
        else:
            return 0

    @property
    def isEmpty(self) -> bool:
        '''Is frames Queue empty.

        Returns
        -------
        bool
            True if frames Queue is empty, otherwise False.
        '''
        return self._frames.empty()

    def get(self, last=False) -> np.ndarray:
        '''Gets image from frames Queue in FIFO or LIFO manner.

        Parameters
        ----------
        last : bool, optional
            Gets image from frames Queue in LIFO manner if True, by default False.

        Returns
        -------
        np.ndarray
            The image first or last in buffer.
        '''
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

    def setExposure(self, value: float):
        '''Sets the exposure time widget of camera

        Parameters
        ----------
        value : float
            selected exposure time
        '''
        self.camera_options.set_param_value(
            CamParams.EXPOSURE, value)

    @property
    def isOpen(self) -> bool:
        '''Returns is acquisition active?

        Returns
        -------
        bool
            True if acquiring, otherwise False.
        '''
        return self.cam.acquisition

    def set_ROI(self):
        '''Sets the ROI for the slected camera
        '''
        pass

    def reset_ROI(self):
        '''Resets the ROI for the slected camera
        '''
        pass

    def center_ROI(self):
        '''Sets the ROI for the slected camera
        '''
        pass

    def select_ROI(self):
        '''Selects the ROI for the camera
        '''
        raise NotImplementedError(
            'The cam_capture function is not implemented yet.')

    def select_ROIs(self):
        '''Selects the ROI for the camera
        '''
        raise NotImplementedError(
            'The cam_capture function is not implemented yet.')

    def get_meta(self):
        meta = dict[str, any]()
        return meta

    def getAcquisitionJob(self) -> AcquisitionJob:
        self._save_path = os.path.join(
            self._directory,
            self.camera_options.get_param_value(CamParams.EXPERIMENT_NAME) + '\\')
        return AcquisitionJob(
            self._temps, self._buffer, self._frames,
            self._save_path, self._cam.getHeight(),
            self._cam.getWidth(),
            biggTiff=self.camera_options.isBiggTiff,
            bytes_per_pixel=self._cam.bytes_per_pixel,
            exposure=self._cam.getExposure(),
            frames=self.camera_options.get_param_value(CamParams.FRAMES),
            full_tif_meta=self.camera_options.isFullMetadata,
            is_dark_cal=self.camera_options.isDarkCalibration,
            meta_file=self.get_meta(),
            meta_func=self.OME_tab.gen_OME_XML if
            self.camera_options.isFullMetadata else self.OME_tab.gen_OME_XML_short,
            name=self._cam.name, prefix=self._save_prefix,
            save=self.camera_options.isSaveData,
            Zarr=not self.camera_options.isTiff
        )

    def view_toggled(self, params):
        if self.camera_options.isSingleView:
            self.lr_0.setSpan(0, 1)
            self.lr_1.setMovable(False)
            self.lr_1.setRegion((0, self._nBins))
            self.lr_1.setSpan(1.0, 1.0)
        else:
            self.lr_0.setSpan(0, 0.5)
            self.lr_1.setMovable(True)
            self.lr_1.setSpan(0.5, 1.0)

    @pyqtSlot()
    def directory_changed(self, value: str):
        '''Slot for directory changed signal'''
        if len(value) > 0:
            self._directory = value

    def start_free_run(self, param=None, Prefix=''):
        '''
        Starts free run acquisition mode

        Parameters
        ----------
        Prefix : str
            an extra prefix added to the image stack file name
        '''
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
        '''
        Starts all workers
        '''

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
            self.camera_options.set_param_value(
                CamParams.TEMPERATURE,
                f'T {self._cam.temperature:.2f} °C')
            self.camera_options.set_param_value(
                CamParams.CAPTURE_STATS,
                'Capture {:d}/{:d} {:.2%} | {:.2f} ms '.format(
                    self.acq_job.frames_captured,
                    self.acq_job.frames,
                    self.acq_job.frames_captured / self.acq_job.frames,
                    self.acq_job.capture_time))
            self.camera_options.set_param_value(
                CamParams.DISPLAY_STATS,
                'Display {:d} | {:.2f} ms '.format(
                    self._buffer.qsize(), self.acq_job.display_time))
            self.camera_options.set_param_value(
                CamParams.SAVE_STATS,
                f'Save {self._frames.qsize():d} | {self.acq_job.save_time:.2f} ms ')

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
            'The getCaptureArgs function is not implemented yet.')

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
            'The cam_capture function is not implemented yet.')


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

                if camp.camera_options.isPreview:
                    if camp.camera_options.isLineProfiler:
                        camp.lineProfiler.setData(params.frame._image)

                    if camp.camera_options.isSingleView:
                        _range = None

                        # image stretching
                        if not camp.camera_options.isAutostretch:
                            _range = tuple(
                                map(math.ceil, camp.lr_0.getRegion()))

                        params.frame.equalizeLUT(
                            _range, camp.camera_options.isNumpyLUT)

                        if camp.camera_options.isAutostretch:
                            camp.lr_0.setRegion(
                                (params.frame._min, params.frame._max))
                            camp.histogram.setXRange(
                                params.frame._min, params.frame._max)
                            camp.hist_cdf.setXRange(
                                params.frame._min, params.frame._max)

                        camp._plot_ref.setData(params.frame._hist)
                        camp._cdf_plot_ref.setData(params.frame._cdf)
                        camp._plot_ref_2.setData(camp._hist)
                        camp._cdf_plot_ref_2.setData(camp._hist)

                        # resizing the image
                        zoom = camp.camera_options.get_param_value(
                            CamParams.RESIZE_DISPLAY)
                        params.frame._view = cv2.resize(
                            params.frame._view, (0, 0),
                            fx=zoom,
                            fy=zoom,
                            interpolation=cv2.INTER_NEAREST)

                        # display it
                        cv2.imshow(params.name, params.frame._view)
                    else:
                        _rang_left = None
                        _rang_right = None

                        # image stretching
                        if not camp.camera_options.isAutostretch:
                            _rang_left = tuple(
                                map(math.ceil, camp.lr_0.getRegion()))
                            _rang_right = tuple(
                                map(math.ceil, camp.lr_1.getRegion()))

                        left, right = params.frame.hsplitView()

                        left.equalizeLUT(
                            _rang_left, camp.camera_options.isNumpyLUT)
                        right.equalizeLUT(
                            _rang_right, camp.camera_options.isNumpyLUT)
                        if camp.camera_options.isAutostretch:
                            camp.lr_0.setRegion((left._min, left._max))
                            camp.lr_1.setRegion((right._min, right._max))
                            camp.histogram.setXRange(
                                min(left._min, right._min),
                                max(left._max, right._max))
                            camp.hist_cdf.setXRange(
                                min(left._min, right._min),
                                max(left._max, right._max))

                        camp._plot_ref.setData(left._hist)
                        camp._cdf_plot_ref.setData(left._cdf)
                        camp._plot_ref_2.setData(right._hist)
                        camp._cdf_plot_ref_2.setData(right._cdf)

                        zoom = camp.camera_options.get_param_value(
                            CamParams.RESIZE_DISPLAY)
                        if camp.camera_options.isOverlaidView:
                            _img = np.zeros(
                                left._view.shape[:2] + (3,),
                                dtype=np.uint8)
                            _img[..., 1] = left._view
                            _img[..., 0] = right._view
                            _img = cv2.resize(
                                _img,
                                (0, 0),
                                fx=zoom,
                                fy=zoom,
                                interpolation=cv2.INTER_NEAREST)

                            cv2.imshow(params.name, _img)
                        else:
                            _img = cv2.resize(
                                np.concatenate(
                                    [left._view, right._view], axis=1),
                                (0, 0),
                                fx=zoom,
                                fy=zoom,
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