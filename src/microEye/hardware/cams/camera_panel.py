import math
import os
import re
import threading
import time
import traceback
from functools import lru_cache
from queue import Queue
from typing import Union

import cv2
import numpy as np
import pyqtgraph as pg

from microEye.hardware.cams.camera_options import CameraOptions, CamParams
from microEye.hardware.cams.jobs import AcquisitionJob
from microEye.hardware.cams.line_profiler import LineProfiler
from microEye.hardware.cams.micam import miCamera
from microEye.hardware.cams.shortcuts import (
    CameraShortcutsWidget,
)
from microEye.qt import QDateTime, Qt, QtCore, QtWidgets, Signal, Slot
from microEye.utils.expandable_groupbox import ExpandableGroupBox
from microEye.utils.metadata_tree import MetadataEditorTree, MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import uImage


class Camera_Panel(QtWidgets.QGroupBox):
    '''
    A Qt Widget base class for controlling a camera | Inherits QGroupBox
    '''

    exposureChanged = Signal()
    asyncFreerun = Signal(str, threading.Event)
    updateStatsSignal = Signal(tuple)
    updateRangeSignal = Signal(int, int)

    def __init__(self, cam: miCamera, mini=False, *args, **kwargs):
        '''
        Initializes a new Camera_Panel Qt widget.

        Inherits QGroupBox.

        Parameters
        ----------
        cam : miCamera
            Camera python adapter.

        mini : bool, optional
            Flag indicating if this is a mini camera panel, by default False.

        Other Parameters
        ---------------
        *args
            Arguments to pass to the QGroupBox constructor.

        **kwargs
            Keyword arguments to pass to the QGroupBox constructor.
        '''
        super().__init__(*args, **kwargs)

        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)

        self._cam = cam  # miCamera
        # flag true if master (always the first added camera is master)
        self.master = False

        self.c_worker = None  # worker for capturing
        self.d_worker = None  # worker for display
        self.s_worker = None  # worker for saving

        self.asyncFreerun.connect(
            lambda prefix, event: self.start_free_run(Prefix=prefix, Event=event)
        )

        # number of bins for the histogram (4096 is set for 12bit mono-camera)
        self._nBins = 2**12
        self._hist = np.arange(0, self._nBins) * 0  # arrays for the Hist plot
        self._bins = np.arange(0, self._nBins)  # arrays for the Hist plot

        self.__update_timer = time.time()
        self.updateStatsSignal.connect(self.updateStats)
        self.updateRangeSignal.connect(self.updateRange)

        self._threadpool = (
            QtCore.QThreadPool.globalInstance()
        )  # the threadpool for workers

        # flag true to close camera adapter and dispose it
        self._dispose_cam = False
        self._buffer = Queue()  # data memory stack
        self._temps = Queue()  # camera sensor temperature stack
        # stack of (frame, temperature) tuples (for saving)
        self.frame = None
        self._frames = Queue()
        # reserved for saving
        home = os.path.expanduser('~')

        self._directory = os.path.join(home, 'Desktop')  # save directory
        self._save_path = ''  # save path
        self._save_prefix = ''

        # acquisition job info class
        self.acq_job = None

        self.mini = mini

        # Line Profiler
        self.lineProfiler = LineProfiler()

        # main layout
        self.main_layout = QtWidgets.QVBoxLayout()

        # set main layout
        self.setLayout(self.main_layout)

        # the main tab widget
        self.main_tab_view = QtWidgets.QTabWidget()
        self.main_layout.addWidget(self.main_tab_view)

        # tab widgets
        self.first_tab = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.OME_tab = MetadataEditorTree()

        # add tabs
        self.main_tab_view.addTab(self.first_tab, 'Main')
        if not self.mini:
            self.main_tab_view.addTab(self.OME_tab, 'OME-XML metadata')

        self.camera_options = CameraOptions()

        # Create the camera shortcuts widget
        self.cam_shortcuts = CameraShortcutsWidget(
            camera=self._cam,
        )

        def convinient_naming(mode: str):
            name: str = self.camera_options.get_param_value(CamParams.EXPERIMENT_NAME)

            if mode == 'exposure':
                match = re.search(
                    r'(\d+)(?:_ms|ms)', name
                )  # find any number of format '000_ms' or '00ms'
                if match:
                    unit = self.camera_options.get_param(CamParams.EXPOSURE).opts[
                        'suffix'
                    ]
                    exposure = self._cam.getExposure()
                    if unit == 's':
                        exposure *= 1000
                    elif unit == 'us':
                        exposure /= 1000
                    # update the exposure time in the name
                    name = name.replace(match.group(0), f'{exposure:03.0f}ms')
                else:
                    name = f'{name}_{self._cam.getExposure():03.0f}ms'

            elif mode == 'index':
                # only from the start of the name
                match = re.search(r'^\d+', name)
                if match:
                    index = int(match.group())
                    index += 1
                    name = re.sub(r'^\d+', f'{index:03d}', name)
                else:
                    name = f'001_{name}'
            elif mode == 'unique_index':
                # Check if folder exists within the chosen directory
                directory = self.camera_options.get_param_value(
                    CamParams.SAVE_DIRECTORY
                )
                match = re.search(r'^\d+', name)
                if match:
                    index = 1
                    while any(
                        folder.startswith(f'{index:03d}_')
                        for folder in os.listdir(directory)
                        if os.path.isdir(os.path.join(directory, folder))
                    ):
                        index += 1
                    name = re.sub(r'^\d+', f'{index:03d}', name)
                else:
                    index = 1
                    while os.path.exists(
                        os.path.join(directory, f'{index:03d}_{name}')
                    ):
                        index += 1
                    name = f'{index:03d}_{name}'

            self.camera_options.set_param_value(CamParams.EXPERIMENT_NAME, name)

        self.cam_shortcuts.exposureChanged.connect(self.setExposure)
        self.cam_shortcuts.autostretchChanged.connect(
            self.camera_options.toggleAutostretch
        )
        self.cam_shortcuts.previewChanged.connect(self.camera_options.togglePreview)
        self.cam_shortcuts.displayStatsChanged.connect(
            self.camera_options.toggleDisplayStats
        )
        self.cam_shortcuts.displayModeChanged.connect(
            lambda mode: self.camera_options.set_param_value(
                CamParams.VIEW_OPTIONS, mode
            )
        )
        self.cam_shortcuts.saveDataChanged.connect(self.camera_options.toggleSaveData)
        self.cam_shortcuts.zoomChanged.connect(self.camera_options.setZoom)
        self.cam_shortcuts.acquisitionStart.connect(self.start_free_run)
        self.cam_shortcuts.acquisitionStop.connect(self.stop)
        self.cam_shortcuts.adjustName.connect(convinient_naming)

        # Create the context menu
        menu = self.cam_shortcuts.create_context_menu(self.camera_options)

        def show_shortcuts(pos):
            self.cam_shortcuts.blockSignals(True)
            self.cam_shortcuts.set_exposure(
                self.camera_options.get_param_value(CamParams.EXPOSURE)
            )
            self.cam_shortcuts.set_zoom(
                self.camera_options.get_param_value(CamParams.RESIZE_DISPLAY)
            )
            self.cam_shortcuts.set_autostretch(
                self.camera_options.get_param_value(CamParams.AUTO_STRETCH)
            )
            self.cam_shortcuts.set_preview(
                self.camera_options.get_param_value(CamParams.PREVIEW)
            )
            self.cam_shortcuts.set_display_stats(
                self.camera_options.get_param_value(CamParams.DISPLAY_STATS_OPTION)
            )
            self.cam_shortcuts.set_display_mode(
                self.camera_options.get_param_value(CamParams.VIEW_OPTIONS)
            )
            self.cam_shortcuts.set_save_data(
                self.camera_options.get_param_value(CamParams.SAVE_DATA)
            )
            self.cam_shortcuts.blockSignals(False)

            # display menu but centered horizontally
            menu.exec(
                self.main_tab_view.mapToGlobal(
                    QtCore.QPoint(
                        self.main_tab_view.rect().left()
                        + (self.main_tab_view.rect().width() - menu.rect().width())
                        // 2,
                        pos.y(),
                    )
                )
            )

        self.camera_options.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.camera_options.customContextMenuRequested.connect(show_shortcuts)

        self.first_tab.addWidget(self.camera_options)
        self.first_tab.setStretchFactor(0, 1)

        self.camera_options.get_param(
            CamParams.EXPERIMENT_NAME
        ).sigValueChanged.connect(
            lambda x: self.OME_tab.set_param_value(
                MetaParams.EXPERIMENT_NAME, x.value()
            )
        )

        self.save_dir_layout = QtWidgets.QHBoxLayout()

        self.camera_options.set_param_value(CamParams.SAVE_DIRECTORY, self._directory)
        self.camera_options.directoryChanged.connect(
            lambda value: self.directory_changed(value)
        )

        save_param = self.camera_options.get_param(CamParams.SAVE_DATA)
        save_param.setOpts(enabled=not self.mini)
        save_param.setValue(self.mini)

        preview_param = self.camera_options.get_param(CamParams.PREVIEW)
        preview_param.setValue(not self.mini)
        preview_param.setOpts(enabled=not self.mini)

        self.camera_options.get_param(CamParams.DISPLAY_STATS_OPTION).setOpts(
            enabled=not self.mini
        )
        self.camera_options.get_param(CamParams.VIEW_OPTIONS).setOpts(
            visible=not self.mini
        )
        self.camera_options.get_param(CamParams.RESIZE_DISPLAY).setOpts(
            enabled=not self.mini
        )

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
        self.camera_options.viewOptionChanged.connect(lambda: self.view_toggled())

        # controls for histogram and cdf

        # Create a single plot with shared X-axis and
        # separate Y-axes for histogram and CDF
        self.histogram = pg.PlotWidget()
        self.hist_cdf = pg.ViewBox()

        # Add the CDF ViewBox to the histogram plot
        self.histogram.getPlotItem().scene().addItem(self.hist_cdf)
        self.histogram.getPlotItem().getAxis('left').setLabel('HIST')
        self.histogram.getPlotItem().getAxis('right').setLabel('CDF')
        self.histogram.getPlotItem().showAxis('right')
        self.histogram.getPlotItem().getAxis('right').linkToView(self.hist_cdf)

        # Link the X-axis of the CDF ViewBox to the main plot
        self.hist_cdf.setXLink(self.histogram.getPlotItem().getViewBox())

        # Adjust the CDF ViewBox to match the main plot
        def update_views():
            self.hist_cdf.setGeometry(
                self.histogram.getPlotItem().getViewBox().sceneBoundingRect()
            )
            self.hist_cdf.linkedViewChanged(
                self.histogram.getPlotItem().getViewBox(), self.hist_cdf.XAxis
            )

        self.histogram.getPlotItem().getViewBox().sigResized.connect(update_views)

        # Set up pens and brushes for histogram and CDF
        greenP = pg.mkPen(color='g')
        blueP = pg.mkPen(color='b')
        cyanP = pg.mkPen(color='c')
        yellowP = pg.mkPen(color='y')
        greenB = pg.mkBrush(0, 255, 0, 32)
        blueB = pg.mkBrush(0, 0, 255, 32)

        # Add histogram and CDF plots
        self._plot_ref = self.histogram.plot(self._bins, self._hist, pen=greenP)
        self._plot_ref_2 = self.histogram.plot(self._bins, self._hist, pen=blueP)
        self._cdf_plot_ref = pg.PlotDataItem(self._bins, self._hist, pen=cyanP)
        self._cdf_plot_ref_2 = pg.PlotDataItem(self._bins, self._hist, pen=yellowP)
        self.hist_cdf.addItem(self._cdf_plot_ref)
        self.hist_cdf.addItem(self._cdf_plot_ref_2)

        # Add linear regions for histogram and CDF
        self.lr_0 = pg.LinearRegionItem(
            (0, self._nBins),
            bounds=(0, self._nBins),
            pen=greenP,
            brush=greenB,
            movable=True,
            swapMode='push',
            span=(0.0, 1),
        )
        self.lr_1 = pg.LinearRegionItem(
            (0, self._nBins),
            bounds=(0, self._nBins),
            pen=blueP,
            brush=blueB,
            movable=True,
            swapMode='push',
            span=(1, 1),
        )
        self.histogram.addItem(self.lr_0)
        self.histogram.addItem(self.lr_1)

        # Add the histogram plot to the layout
        self.first_tab.addWidget(self.histogram)
        self.first_tab.setStretchFactor(1, 1)

        CameraOptions.combine_params(self.PARAMS)

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

    @property
    def Event(self):
        '''The acquisition job threading Event.'''
        if hasattr(self, '_acqEvent'):
            return self._acqEvent
        else:
            return None

    @Event.setter
    def Event(self, value: threading.Event):
        '''The acquisition job threading Event.'''
        if value is None:
            self._acqEvent = threading.Event()
        else:
            self._acqEvent = value

    def get_event_action(self, param: str):
        return {'name': str(param), 'type': 'action', 'event': 'Event'}

    def __str__(self):
        return f'{self.cam.name}'

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
        self.camera_options.set_param_value(CamParams.EXPOSURE, value)

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
        '''Sets the ROI for the slected camera'''
        pass

    def reset_ROI(self):
        '''Resets the ROI for the slected camera'''
        pass

    def center_ROI(self):
        '''Sets the ROI for the slected camera'''
        pass

    def select_ROI(self):
        '''Selects the ROI for the camera'''
        raise NotImplementedError('The select_ROI function is not implemented yet.')

    def select_ROIs(self):
        '''Selects the ROI for the camera'''
        raise NotImplementedError('The select_ROIs function is not implemented yet.')

    def get_meta(self):
        return self.camera_options.get_json()

    def getAcquisitionJob(self, event=None) -> AcquisitionJob:
        self._save_path = os.path.join(
            self._directory,
            self.camera_options.get_param_value(CamParams.EXPERIMENT_NAME) + '\\',
        )
        self.Event = event
        return AcquisitionJob(
            self._temps,
            self._buffer,
            self._frames,
            self._save_path,
            self._cam.getHeight(),
            self._cam.getWidth(),
            biggTiff=self.camera_options.isBiggTiff,
            bytes_per_pixel=self._cam.bytes_per_pixel,
            exposure=self._cam.getExposure(),
            frames=self.camera_options.get_param_value(CamParams.FRAMES),
            full_tif_meta=self.camera_options.isFullMetadata,
            is_dark_cal=self.camera_options.isDarkCalibration,
            meta_file=self.get_meta(),
            meta_func=self.OME_tab.gen_OME_XML
            if self.camera_options.isFullMetadata
            else self.OME_tab.gen_OME_XML_short,
            name=self._cam.name,
            prefix=self._save_prefix,
            save=self.camera_options.isSaveData,
            Zarr=not self.camera_options.isTiff,
            rois=self.camera_options.get_export_rois(),
            seperate_rois=self.camera_options.get_param_value(
                CamParams.EXPORT_ROIS_SEPERATE
            ),
            flip_rois=self.camera_options.get_param_value(
                CamParams.EXPORT_ROIS_FLIPPED
            ),
            s_event=self.Event,
        )

    def view_toggled(self):
        if self.camera_options.isSingleView or self.camera_options.isROIsView:
            self.lr_0.setSpan(0, 1)
            self.lr_1.setMovable(False)
            self.lr_1.setRegion((0, self._nBins))
            self.lr_1.setSpan(1.0, 1.0)
        else:
            self.lr_0.setSpan(0, 0.5)
            self.lr_1.setMovable(True)
            self.lr_1.setSpan(0.5, 1.0)

    @Slot()
    def directory_changed(self, value: str):
        '''Slot for directory changed signal'''
        if len(value) > 0:
            self._directory = value

    def start_free_run(self, param=None, Prefix='', Event=None):
        '''
        Starts free run acquisition mode

        Parameters
        ----------
        param : Parameter
            the parameter that was activated.
        Prefix : str
            an extra prefix added to the image stack file name.
        '''
        if self._cam.acquisition:
            return  # if acquisition is already going on

        self._save_prefix = Prefix
        self.acq_job = self.getAcquisitionJob(Event)

        self._cam.acquisition = True  # set acquisition flag to true

        # start both capture and display workers
        self.start_all_workers()

    def stop(self):
        if self.acq_job is not None:
            self.acq_job.c_event.set()
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
        self.time = QDateTime.currentDateTime()

        # Pass the display function to be executed
        if self.d_worker is None or self.d_worker.is_set():
            self.d_worker = QThreadWorker(cam_display, self.acq_job, self)

            self.d_worker.signals.finished.connect(
                lambda: self.acq_job.setDone(1, True)
            )
            # Execute
            self.d_worker.setAutoDelete(True)
            self._threadpool.start(self.d_worker)

        # Pass the save function to be executed
        if not self.mini:
            if self.s_worker is None or self.s_worker.is_set():
                self.s_worker = QThreadWorker(cam_save, self.acq_job)
                self.s_worker.setAutoDelete(True)
                # Execute
                self._threadpool.start(self.s_worker)

        #  Pass the capture function to be executed
        if self.c_worker is None or self.c_worker.is_set():
            # Any other args, kwargs are passed to the run function
            self.c_worker = QThreadWorker(self.cam_capture, *self.getCaptureArgs())
            self.c_worker.signals.finished.connect(
                lambda: self.acq_job.setDone(0, True)
            )
            self.c_worker.setAutoDelete(True)
            # Execute
            self._threadpool.start(self.c_worker)

    def updateInfo(self):
        if isinstance(self.acq_job, AcquisitionJob):
            self.camera_options.set_param_value(
                CamParams.TEMPERATURE, f'T {self._cam.temperature:.2f} °C'
            )
            self.camera_options.set_param_value(
                CamParams.CAPTURE_STATS,
                (
                    f'Capture {self.acq_job.frames_captured}/{self.acq_job.frames} '
                    + f'{self.acq_job.frames_captured / self.acq_job.frames:.2%} '
                    + f'| {self.acq_job.capture_time:.2f} ms'
                ),
            )
            self.camera_options.set_param_value(
                CamParams.DISPLAY_STATS,
                f'Display {self._buffer.qsize()} | {self.acq_job.display_time:.2f} ms',
            )
            self.camera_options.set_param_value(
                CamParams.SAVE_STATS,
                f'Save {self._frames.qsize():d} | {self.acq_job.save_time:.2f} ms ',
            )

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
        raise NotImplementedError('The getCaptureArgs function is not implemented yet.')

    def cam_capture(self, *args, **kwargs):
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
        raise NotImplementedError('The cam_capture function is not implemented yet.')

    def updateStats(self, frames: tuple[uImage]):
        now = time.time()
        if now - self.__update_timer < 0.1:
            return

        self.__update_timer = now
        for idx, frame in enumerate(frames):
            if idx < 2:
                _lr = self.lr_0 if idx == 0 else self.lr_1
                _plot_ref = self._plot_ref if idx == 0 else self._plot_ref_2
                _cdf_plot_ref = self._cdf_plot_ref if idx == 0 else self._cdf_plot_ref_2

                if frame is not None:
                    if self.camera_options.isAutostretch:
                        _lr.setRegion((frame._min, frame._max))

                    _plot_ref.setData(frame._hist)
                    _cdf_plot_ref.setData(frame._cdf)
                else:
                    _plot_ref.setData(self._hist)
                    _cdf_plot_ref.setData(self._hist)
            else:
                break

    def updateRange(self, rmin: int, rmax: int):
        if time.time() - self.__update_timer < 0.1:
            return

        if self.camera_options.isAutostretch:
            self.histogram.setXRange(rmin, rmax)
            self.hist_cdf.setXRange(rmin, rmax)


def cam_save(params: AcquisitionJob, **kwargs):
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
                save_time = QDateTime.currentDateTime()

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
                    params.save_time = save_time.msecsTo(QDateTime.currentDateTime())

                params.frames_saved += 1

            QtCore.QThread.usleep(100)
            # Flag that ends the loop
            if (
                params.save_queue.empty()
                and params.stop_threads
                and params.display_done
            ):
                print('Save thread break.')
                break
    except Exception:
        traceback.print_exc()
    finally:
        params.s_event.set()
        params.finalize()
        print('Save thread finally finished.')


@lru_cache(maxsize=128)
def get_font_settings():
    return cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 255


def drawStats(frame: uImage):
    lines = [
        f'min/max: {np.min(frame.image)}/{np.max(frame.image)}',
        f'mean: {np.mean(frame.image):.3f}',
        f'median: {np.median(frame.image):.3f}',
        f'std: {np.std(frame.image):.4f}',
    ]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Reduced font size
    font_thickness = 1
    font_color = (255, 255, 255)  # White text

    # Position to start adding text
    x, y = 10, 20

    # Add each line of text to the image with a background
    for line in lines:
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_x, text_y = x, y - text_size[1]
        cv2.rectangle(
            frame._view,
            (text_x - 2, text_y - 2),
            (text_x + text_size[0] + 2, text_y + text_size[1] + 2),
            (0, 0, 0),  # Black background
            -1,
        )
        cv2.putText(
            frame._view, line, (x, y), font, font_scale, font_color, font_thickness
        )
        y += int(1.5 * text_size[1])  # Adjust vertical spacing


def update_histogram_and_display(params: AcquisitionJob, camp: Camera_Panel):
    '''
    Update histogram and display based on camera options.

    Parameters
    ----------
    params : AcquisitionJob
        Class holding cross-thread info required for an Acquisition Job.
    camp : Camera_Panel
        The camera panel to display data.
    '''

    def process_frame(frame: uImage, region=None, zoom=None):
        '''Helper to process a single frame.'''
        frame.equalizeLUT(region, camp.camera_options.isNumpyLUT)
        if zoom is None:
            zoom = camp.camera_options.get_param_value(CamParams.RESIZE_DISPLAY)
        frame._view = cv2.resize(
            frame._view, (0, 0), fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST
        )

        if camp.camera_options.isDisplayStats:
            drawStats(frame)
        return frame

    def display_frame(name, frame: Union[uImage, np.ndarray]):
        '''Helper to display a single frame.'''
        if isinstance(frame, uImage):
            frame = frame._view

        cv2.imshow(name, frame)

    if camp.camera_options.isSingleView or (
        camp.camera_options.isROIsView and len(params.rois) == 0
    ):
        region = (
            None
            if camp.camera_options.isAutostretch
            else tuple(map(math.ceil, camp.lr_0.getRegion()))
        )
        params.frame = process_frame(params.frame, region)
        camp.updateRangeSignal.emit(params.frame._min, params.frame._max)
        camp.updateStatsSignal.emit((params.frame, None))
        display_frame(params.name, params.frame)

    elif camp.camera_options.isROIsView:
        images, mins, maxs = [], [], []
        for idx, roi in enumerate(params.rois):
            uimage = uImage(
                params.frame.image[roi[1] : roi[1] + roi[3], roi[0] : roi[0] + roi[2]]
            )
            if idx > 0 and params.flip_rois:
                uimage.image = np.fliplr(uimage.image)
            region = (
                None
                if camp.camera_options.isAutostretch
                else tuple(map(math.ceil, camp.lr_0.getRegion()))
            )
            uimage = process_frame(uimage, region)
            mins.append(uimage._min)
            maxs.append(uimage._max)
            images.append(uimage)
            display_frame(f'ROI {idx + 1}', uimage)
        camp.updateRangeSignal.emit(min(mins), max(maxs))
        camp.updateStatsSignal.emit(tuple(images))

    else:
        left, right = params.frame.hsplitView()
        zoom = camp.camera_options.get_param_value(CamParams.RESIZE_DISPLAY)
        left = process_frame(
            left,
            None
            if camp.camera_options.isAutostretch
            else tuple(map(math.ceil, camp.lr_0.getRegion())),
            zoom=zoom,
        )
        right = process_frame(
            right,
            None
            if camp.camera_options.isAutostretch
            else tuple(map(math.ceil, camp.lr_1.getRegion())),
            zoom=zoom,
        )
        camp.updateRangeSignal.emit(
            min(left._min, right._min), max(left._max, right._max)
        )
        camp.updateStatsSignal.emit((left, right))

        if camp.camera_options.isOverlaidView:
            overlaid_img = np.zeros(left._view.shape[:2] + (3,), dtype=np.uint8)
            overlaid_img[..., 1] = left._view
            overlaid_img[..., 0] = right._view
            display_frame(params.name, overlaid_img)
        else:
            concatenated_img = np.concatenate([left._view, right._view], axis=1)
            display_frame(params.name, concatenated_img)


def cam_display(params: AcquisitionJob, camp: Camera_Panel, **kwargs):
    '''Display function executed by the display worker.

    Processes the acquired frame, displays it, and sends it to the save
    stack.

    Parameters
    ----------
    params : AcquisitionJob
        class holding cross-thread info required for an Acquisition Job.
    camp : Camera_Panel
        the camera panel to display data.
    '''
    try:
        time = QDateTime.currentDateTime()

        # Continuous image display
        while True:
            # proceed only if the buffer is not empty
            if not params.display_queue.empty():
                # for display time estimations
                with params.lock:
                    params.display_time = time.msecsTo(QDateTime.currentDateTime())
                    time = QDateTime.currentDateTime()

                    # reshape image into proper shape
                    # (height, width, bytes per pixel)
                    params.frame = uImage.fromBuffer(
                        params.display_queue.get(),
                        params.cam_height,
                        params.cam_width,
                        params.bytes_per_pixel,
                    )

                    # add to saving stack
                    if params.save:
                        if not camp.mini:
                            params.save_queue.put(
                                (params.frame.image, params.temp_queue.get())
                            )
                        else:
                            params.save_queue.put(params.frame.image)

                    if camp.camera_options.isPreview:
                        if camp.camera_options.isLineProfiler:
                            camp.lineProfiler.imageUpdate.emit(params.frame._image)

                        update_histogram_and_display(params, camp)
                        cv2.waitKey(1)
                    else:
                        cv2.waitKey(1)
            else:
                cv2.waitKey(1)

            QtCore.QThread.usleep(100)
            # Flag that ends the loop
            if (
                params.stop_threads
                and params.display_queue.empty()
                and params.capture_done
            ):
                print('Display thread break.')
                break
    except Exception:
        traceback.print_exc()
    finally:
        try:
            if not camp.mini:
                cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()
        print('Display thread finally finished.')
