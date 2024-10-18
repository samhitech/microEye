import json
import math
import os
import threading
import traceback
from typing import Union

import cv2
import numpy as np
import pyqtgraph as pg
import tifffile as tf
from numba import cuda
from ome_types import OME

if cuda.is_available():
    from microEye.analysis.fitting.pyfit3Dcspline.mainfunctions import GPUmleFit_LM
else:

    def GPUmleFit_LM(*args):
        pass


from microEye.analysis.checklist_dialog import ChecklistDialog
from microEye.analysis.cmosMaps import cmosMaps
from microEye.analysis.filters import BandpassFilter, TemporalMedianFilter
from microEye.analysis.fitting import psf, pyfit3Dcspline
from microEye.analysis.fitting.fit import *
from microEye.analysis.fitting.phasor_fit import phasor_fit
from microEye.analysis.fitting.results import (
    FittingMethod,
    FittingResults,
    ResultsUnits,
)
from microEye.analysis.tools.kymograms import KymogramWidget
from microEye.analysis.viewer.image_options_widget import FittingOptions, Parameters
from microEye.analysis.viewer.layers_widget import ImageParamsWidget
from microEye.qt import (
    QDateTime,
    Qt,
    QtCore,
    QtGui,
    QtWidgets,
    Signal,
    Slot,
    getOpenFileName,
    getSaveFileName,
)
from microEye.utils.gui_helper import *
from microEye.utils.labelled_slider import LabelledSlider
from microEye.utils.metadata_tree import MetadataEditorTree, MetaParams
from microEye.utils.thread_worker import QThreadWorker
from microEye.utils.uImage import (
    WORD,
    TiffSeqHandler,
    ZarrImageSequence,
    saveZarrImage,
    uImage,
)


class StackView(QtWidgets.QWidget):
    '''
    A class for viewing and interacting with image stacks in a PyQt5 application.
    '''

    localizedData = Signal(str)

    def __init__(self, path: str, mask_pattern: str = None):
        '''
        Initialize the StackView.

        Parameters
        ----------
        path : str
            The path to the image stack.
        mask_pattern : str, optional
            A pattern to the mask file, by default None.
        '''
        super().__init__()

        self.setWindowTitle(path.split('/')[-1])

        self.path = path
        self.stack_handler = self.getStackHandler(path, mask_pattern)
        self.stack_handler.open()
        self._threadpool = QtCore.QThreadPool.globalInstance()
        self._lock = threading.Lock()

        self.main_layout = QtWidgets.QHBoxLayout()
        self.setLayout(self.main_layout)

        # Graphics layout
        self.initGraphics()

        # Tab Widget
        self.tab_widget = QtWidgets.QTabWidget()

        self.main_layout.addWidget(self.tab_widget, 2)

        # initialize the Prefit/Fitting Options tab
        self.setupOptionsTab()

        # Layers tab
        self.setupLayersTab()

        # Kymogram tab
        self.setupKymogramTab()

        # CMOS maps tab
        self.setupCMOSMapsTab()

        # Load Metadata
        self.load_ome_metadata()

        nTime = self.stack_handler.shapeTCZYX()[1]
        for idx in range(nTime + 1):
            image_item = self.addImageItem(
                self.empty_image if idx < nTime else self.empty_alpha,
                1,
                'Plus' if 0 < idx < nTime else 'SourceOver',
            )

            histogram_item = pg.HistogramLUTItem(
                gradientPosition='bottom', orientation='horizontal'
            )
            histogram_item.setImageItem(image_item)
            if 0 < idx < nTime:
                image_item.setCompositionMode(
                    QtGui.QPainter.CompositionMode.CompositionMode_Plus
                )

            self.image_widget.addItem(histogram_item, row=1 + idx, col=0)
            self.histogram_items.append(histogram_item)

        self.connnectSignals()
        self.centerROI()

    def connnectSignals(self):
        self.roi.sigRegionChanged.connect(self.region_changed)
        self.roi.sigRegionChangeFinished.connect(self.slider_changed)

        self.fitting_options_widget.localizeData.connect(self.localize)
        self.fitting_options_widget.extractPSF.connect(self.extract_psf)
        self.fitting_options_widget.saveCropped.connect(self.save_cropped_img)
        self.fitting_options_widget.roiEnabled.connect(self.change_roi_visibility)
        self.fitting_options_widget.roiChanged.connect(self.roi_changed)
        self.fitting_options_widget.paramsChanged.connect(self.slider_changed)

        self.frames_slider.valueChanged.connect(self.slider_changed)
        self.z_slider.valueChanged.connect(self.slider_changed)
        self.lr_0.sigRegionChangeFinished.connect(self.slider_changed)

    def initGraphics(self):
        # A plot area (ViewBox + axes) for displaying the image
        self.uImage = None
        self.image_widget = pg.GraphicsLayoutWidget()
        graphicsLayout = self.image_widget.ci
        self.image_widget.setMinimumWidth(300)

        # Create the ViewBox
        self.view_box: pg.ViewBox = graphicsLayout.addViewBox(
            row=0, col=0, invertY=True
        )

        self.view_box.setAspectLocked()
        self.view_box.invertY(True)

        self.empty_image = np.zeros(self.stack_handler.shape[-2:], dtype=np.uint8)
        self.empty_alpha = np.zeros(
            self.stack_handler.shape[-2:] + (4,), dtype=np.uint8
        )
        # self.empty_image[0, 0] = 255
        self.image_items: list[tuple[pg.ImageItem, np.ndarray]] = []
        self.histogram_items: list[pg.HistogramLUTItem] = []

        self.roi = pg.RectROI(
            [-8, 14], [6, 5], scaleSnap=True, translateSnap=True, movable=False
        )
        self.roi.addTranslateHandle([0, 0], [0.5, 0.5])
        self.roi.setZValue(1000)
        self.roi.setVisible(False)
        self.view_box.addItem(self.roi)

        # Add the two sub-main layouts
        self.main_layout.addWidget(self.image_widget, 3)

    def addImageItem(
        self, image: np.ndarray, opacity: float = 1.0, compMode='SourceOver'
    ):
        # Create the ImageItem and set its view to self.view_box
        image_item = pg.ImageItem(
            image, axisOrder='row-major', opacity=max(min(opacity, 1), 0)
        )
        # Add the ImageItem to the ViewBox
        self.view_box.addItem(image_item)

        self.image_items.append([image_item, image])
        self.image_layers.add_layer(compMode)

        return image_item

    def setupOptionsTab(self):
        self.image_control_layout = QtWidgets.QFormLayout()
        self.image_control_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.frames_label = QtWidgets.QLabel('Time Axis:')
        self.frames_slider = LabelledSlider(
            None, True, (0, self.stack_handler.shapeTCZYX()[0] - 1)
        )
        self.z_label = QtWidgets.QLabel('Z Axis:')
        self.z_slider = LabelledSlider(
            None, True, (0, self.stack_handler.shapeTCZYX()[2] - 1)
        )

        # Hist plotWidget
        self.histogram = pg.PlotWidget()
        green_pen = pg.mkPen(color='g')
        self._bins = np.arange(WORD)
        green_brush = pg.mkBrush(0, 255, 0, 32)

        self.plot_refs = []
        # Add hist channels plot references
        colors = 'brgcyk'
        for idx in range(self.stack_handler.shapeTCZYX()[1]):
            ref = self.histogram.plot(
                self._bins, np.zeros_like(self._bins), pen=pg.mkPen(color=colors[idx])
            )
            self.plot_refs.append(ref)

        self.lr_0 = pg.LinearRegionItem(
            (0, WORD - 1),
            bounds=(0, WORD - 1),
            pen=green_pen,
            brush=green_brush,
            movable=True,
            swapMode='push',
            span=(0.0, 1),
        )
        self.histogram.addItem(self.lr_0)

        if self.frames_slider.maximum() > 0:
            self.image_control_layout.addRow(self.frames_label, self.frames_slider)
        if self.z_slider.maximum() > 0:
            self.image_control_layout.addRow(self.z_label, self.z_slider)
        self.image_control_layout.addRow(self.histogram)

        self.fitting_options_widget = FittingOptions(
            shape=self.stack_handler.shapeTCZYX()[-2:]
        )

        # Localization GroupBox
        self.export_options = ChecklistDialog(
            'Exported Columns',
            [
                'Super-res image',
            ]
            + UNIQUE_COLUMNS,
            checked=True,
            parent=self,
        )

        self.prefit_widget, self.prefit_options_layout = create_widget(
            QtWidgets.QVBoxLayout
        )
        layout_add_elements(
            self.prefit_options_layout,
            self.image_control_layout,
            self.fitting_options_widget,
        )
        self.prefit_options_layout.addStretch()

        self.tab_widget.addTab(self.prefit_widget, 'Prefit/Fitting Options')

    def get_param(self, param_name: Parameters):
        '''
        Get a parameter by name from the cache of image_prefit_widget.
        '''
        return self.fitting_options_widget.get_param(param_name)

    @Slot()
    def save_cropped_img(self):
        if self.stack_handler is None:
            return

        filename, _ = getSaveFileName(
            self,
            'Save Cropped Image',
            directory=os.path.dirname(self.path),
            filter='Zarr Files (*.zarr)',
        )

        if len(filename) > 0:
            roiInfo = self.get_roi_info()

            def work_func(**kwargs):
                try:
                    if roiInfo is not None:
                        origin, dim = roiInfo

                        saveZarrImage(
                            filename,
                            self.stack_handler,
                            ySlice=slice(int(origin[1]), int(origin[1] + dim[1])),
                            xSlice=slice(int(origin[0]), int(origin[0] + dim[0])),
                        )
                    else:
                        origin, dim = None, None

                        saveZarrImage(filename, self.stack_handler)
                except Exception:
                    traceback.print_exc()

            def done(results):
                self.get_param(Parameters.SAVE_CROPPED_IMAGE).setOpts(enabled=True)

            self.worker = QThreadWorker(work_func)
            self.worker.signals.result.connect(done)
            # Execute
            self.get_param(Parameters.SAVE_CROPPED_IMAGE).setOpts(enabled=False)
            self._threadpool.start(self.worker)

    def setupLayersTab(self):
        # Layers tab
        self.image_layers = ImageParamsWidget()

        self.image_layers.paramsChanged.connect(self.updateLayer)

        self.tab_widget.addTab(self.image_layers, 'Layers')

    def updateLayer(self, param, changes: list):
        for param, _, data in changes:
            path = self.image_layers.param_tree.childPath(param)

            if len(path) > 1:
                parameter = path[-1]
                parent = path[-2]
            else:
                parameter = '.'.join(path) if path is not None else param.name()
                parent = None

            if parameter in ['Opacity', 'Visible']:
                idx = int(parent.split(' ')[-1]) - 1
                layer = self.image_layers.param_tree.param('Layers').child(parent)
                opacity = (
                    data if parameter == 'Opacity' else layer.child('Opacity').value()
                )
                visible = (
                    int(data)
                    if parameter == 'Visible'
                    else int(layer.child('Visible').value())
                )

                if idx < len(self.image_items):
                    self.image_items[idx][0].setOpts(opacity=visible * (opacity / 100))
            elif parameter == 'CompositionMode':
                idx = int(parent.split(' ')[-1]) - 1
                self.image_items[idx][0].setCompositionMode(
                    getattr(
                        QtGui.QPainter.CompositionMode, 'CompositionMode_' + str(data)
                    )
                )

    def setupKymogramTab(self):
        # Creating the kymogram tab layout
        self.kymogram_widget = KymogramWidget(
            max_window=self.stack_handler.shapeTCZYX()[0]
        )

        # self.kymogram_widget.setMaximum(self.stack_handler.shapeTCZYX()[0] - 1)
        self.kymogram_widget.displayClicked.connect(self.kymogram_display_clicked)
        self.kymogram_widget.extractClicked.connect(self.kymogram_btn_clicked)

        self.tab_widget.addTab(self.kymogram_widget, 'Kymogram')

    def kymogram_display_clicked(self, data: np.ndarray):
        cv2.destroyAllWindows()

        uImg = uImage(data)
        uImg.equalizeLUT()
        # self.image.setImage(uImg._view, autoLevels=True)
        cv2.imshow('Kymogram', uImg._view)

    def kymogram_btn_clicked(self):
        if self.stack_handler is None:
            return

        cv2.destroyAllWindows()

        self.kymogram_widget.extract_kymogram(
            self.stack_handler,
            self.frames_slider.value(),
            self.frames_slider.maximum(),
            self.get_roi_info(),
        )

    def setupCMOSMapsTab(self):
        # CMOS maps tab
        self.cmos_maps_group = cmosMaps()
        self.tab_widget.addTab(self.cmos_maps_group, 'CMOS Maps')

    def centerROI(self):
        '''
        Centers the region of interest (ROI) and fits it to the image.
        '''
        x = self.stack_handler.shape[-1]
        y = self.stack_handler.shape[-2]

        self.roi.setSize([x, y])
        self.roi.setPos([0, 0])
        self.roi.maxBounds = QtCore.QRectF(0, 0, x, y)

    def get_roi_info(self):
        if self.get_param(Parameters.ENABLE_ROI).value():
            origin = self.roi.pos()  # ROI (x,y)
            dim = self.roi.size()  # ROI (w,h)
            return (round(origin[0]), round(origin[1])), (int(dim[0]), int(dim[1]))
        else:
            return None

    def set_roi_info(self, x: int, y: int, width: int, height: int):
        shape = self.stack_handler.shapeTCZYX()
        x = min(max(0, x), shape[-1] - 1)
        y = min(max(0, y), shape[-2] - 1)
        width = min(max(1, width), shape[-1] - x)
        height = min(max(1, height), shape[-2] - y)
        self.roi.setPos([x, y])  # ROI (x,y)
        self.roi.setSize([width, height])  # ROI (w,h)

    def get_roi_txt(self):
        if self.get_param(Parameters.ENABLE_ROI).value():
            pixel_size = self.get_param(Parameters.PIXEL_SIZE).value()
            return (
                ' | ROI Pos. ('
                + '{:.0f}, {:.0f}), '
                + 'Size ({:.0f}, {:.0f})/({:.3f} um, {:.3f} um)'
            ).format(
                *self.roi.pos(),
                *self.roi.size(),
                *(self.roi.size() * pixel_size / 1000),
            )

        return ''

    @Slot(tuple)
    def roi_changed(self, value: tuple[int, int, int, int]):
        self.set_roi_info(*value)

    @Slot()
    def region_changed(self):
        x, y = self.roi.pos()
        w, h = self.roi.size()
        self.fitting_options_widget.set_roi(x, y, w, h)

    @Slot()
    def change_roi_visibility(self):
        if self.get_param(Parameters.ENABLE_ROI).value():
            self.roi.setVisible(True)
        else:
            self.roi.setVisible(False)

    def load_ome_metadata(self):
        if isinstance(self.stack_handler, TiffSeqHandler):
            with tf.TiffFile(self.stack_handler._tiff_seq.files[0]) as fl:
                if fl.is_ome:
                    ome_metadata = OME.from_xml(fl.ome_metadata)

                    self.metadata_editor = MetadataEditorTree()
                    self.metadata_editor.pop_OME_XML(ome_metadata)

                    self.tab_widget.addTab(self.metadata_editor, 'OME Metadata')
                    self.get_param(Parameters.PIXEL_SIZE).setValue(
                        self.metadata_editor.get_param_value(MetaParams.PX_SIZE)
                    )

    @Slot()
    def slider_changed(self):
        if self.stack_handler is not None:
            self.update_display()

    def update_histogram(self):
        self.lr_0.setBounds([0, self.uImage._max])

        min_max = None
        if not self.get_param(Parameters.AUTO_STRETCH).value():
            min_max = tuple(map(math.ceil, self.lr_0.getRegion()))

        self.uImage.equalizeLUT(min_max, True)

        # TODO: Update plotrefs
        if self.uImage._hist.ndim == 1:
            self.plot_refs[0].setData(self.uImage._hist)
        elif self.uImage._hist.ndim == 2:
            for idx in range(self.uImage._hist.shape[1]):
                self.plot_refs[idx].setData(self.uImage._hist[:, idx])

        if self.get_param(Parameters.AUTO_STRETCH).value():
            self.lr_0.sigRegionChangeFinished.disconnect(self.slider_changed)
            self.lr_0.setRegion([self.uImage._min, self.uImage._max])
            self.histogram.setXRange(self.uImage._min, self.uImage._max)
            self.lr_0.sigRegionChangeFinished.connect(self.slider_changed)

    def update_images(self):
        if self.uImage._view.ndim == 2:
            self.image_items[0][1] = self.uImage._view
            self.image_items[0][0].setImage(self.uImage._view)
        elif self.uImage._view.ndim == 3:
            for idx in range(self.uImage._view.shape[2]):
                self.image_items[idx][1] = self.uImage._view[..., idx]
                self.image_items[idx][0].setImage(self.uImage._view[..., idx])

    def apply_cmos_maps(self, image):
        varim = None
        if self.cmos_maps_group.active.isChecked():
            res = self.cmos_maps_group.getMaps()
            if res is not None and res[0].shape == image.shape:
                image = image * res[0]
                image = image - res[1]
                varim = res[2]
        return varim

    def apply_temporal_median_filter(self, image, roiInfo):
        if self.get_param(Parameters.TM_FILTER_ENABLED).value():
            filter = TemporalMedianFilter(
                self.get_param(Parameters.TM_FILTER_WINDOW_SIZE).value()
            )
            frames = filter.getFrames(
                self.frames_slider.value(),
                self.stack_handler,
                None,
                self.z_slider.value(),
            )
            image = filter.run(image, frames, roiInfo)
        return image

    def update_display(self):
        image = self.stack_handler.getSlice(
            self.frames_slider.value(), None, self.z_slider.value()
        )

        varim = self.apply_cmos_maps(image)

        roiInfo = self.get_roi_info()

        image = self.apply_temporal_median_filter(image, roiInfo)

        self.uImage = uImage(image)

        self.update_histogram()

        self.update_images()

        if self.get_param(Parameters.REALTIME_LOCALIZATION).value():
            if image.ndim == 2:
                self.apply_realtime_localization(image, varim, roiInfo)
        else:
            self.image_items[-1][0].setImage(self.empty_alpha, autoLevels=False)

    def preprocess_image(self, image: uImage, roi_info):
        if roi_info is not None:
            origin, dim = roi_info
            img = image._view[
                int(origin[1]) : int(origin[1] + dim[1]),
                int(origin[0]) : int(origin[0] + dim[0]),
            ]
        else:
            origin, dim = None, None
            img = image._view

        # Apply bandpass filter
        img = self.fitting_options_widget.get_image_filter().run(img)

        # Threshold the image
        _, th_img = cv2.threshold(
            img,
            np.quantile(img, 1 - 1e-4)
            * self.get_param(Parameters.RELATIVE_THRESHOLD_MIN).value(),
            255,
            cv2.THRESH_BINARY,
        )

        if self.get_param(Parameters.RELATIVE_THRESHOLD_MAX).value() < 1.0:
            _, th2 = cv2.threshold(
                img,
                np.max(img) * self.get_param(Parameters.RELATIVE_THRESHOLD_MAX).value(),
                1,
                cv2.THRESH_BINARY_INV,
            )
            th_img = th_img * th2

        if self.get_param(Parameters.SHOW_FILTER).value():
            cv2.namedWindow('Thresholded filtered Img.', cv2.WINDOW_NORMAL)
            cv2.imshow('Thresholded filtered Img.', th_img)

        return th_img, img, origin

    def detect_and_display_keypoints(self, th_img, img, origin):
        # Detect blobs
        points, im_with_keypoints = (
            self.fitting_options_widget.get_detector().find_peaks_preview(th_img, img)
        )

        # Show keypoints
        if self.get_param(Parameters.SHOW_FILTER).value():
            cv2.namedWindow('Approx. Loc.', cv2.WINDOW_NORMAL)
            cv2.imshow('Approx. Loc.', im_with_keypoints)

        if len(points) > 0 and origin is not None:
            points[:, 0] += origin[0]
            points[:, 1] += origin[1]

        return points

    def fit_keypoints(self, image, varim, points):
        # method
        method = self.fitting_options_widget.get_fitting_method()

        if method == FittingMethod._2D_Phasor_CPU:
            sz = self.get_param(Parameters.ROI_SIZE).value()
            sub_fit = phasor_fit(image, points, True, sz)

            if sub_fit is not None:
                keypoints = [cv2.KeyPoint(*point, size=1.0) for point in sub_fit[:, :2]]

                # Draw detected blobs as red circles.
                im_with_keypoints = cv2.drawKeypoints(
                    self.empty_image,
                    keypoints,
                    None,
                    (0, 0, 255),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )
            else:
                im_with_keypoints = None
        else:
            sz = self.get_param(Parameters.ROI_SIZE).value()
            if varim is None:
                varims = None
                rois, coords = pyfit3Dcspline.get_roi_list(image, points, sz)
            else:
                rois, varims, coords = pyfit3Dcspline.get_roi_list_CMOS(
                    image, varim, points, sz
                )
            Params = None

            PSF_param = np.array([self.get_param(Parameters.INITIAL_SIGMA).value()])

            if method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
                Params, CRLBs, LogLikelihood = pyfit3Dcspline.CPUmleFit_LM(
                    rois, 1, PSF_param, varims, 0
                )
            elif method == FittingMethod._2D_Gauss_MLE_free_sigma:
                Params, CRLBs, LogLikelihood = pyfit3Dcspline.CPUmleFit_LM(
                    rois, 2, PSF_param, varims, 0
                )
            elif method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                Params, CRLBs, LogLikelihood = pyfit3Dcspline.CPUmleFit_LM(
                    rois, 4, PSF_param, varims, 0
                )
            elif method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
                Params, CRLBs, LogLikelihood = pyfit3Dcspline.CPUmleFit_LM(
                    rois, 5, np.ones((64, 4, 4, 4), dtype=np.float32), varims, 0
                )

            if Params is not None:
                keypoints = [
                    cv2.KeyPoint(
                        Params[idx, 0] + coords[idx, 0],
                        Params[idx, 1] + coords[idx, 1],
                        size=1.0,
                    )
                    for idx in range(rois.shape[0])
                ]

                # Draw detected blobs as red circles.
                im_with_keypoints = cv2.drawKeypoints(
                    self.empty_image,
                    keypoints,
                    None,
                    (0, 0, 255),
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                )
            else:
                im_with_keypoints = None
        return im_with_keypoints

    def apply_realtime_localization(self, image, varim, roiInfo):
        th_img, img, origin = self.preprocess_image(self.uImage, roiInfo)

        # Detect blobs.
        points = self.detect_and_display_keypoints(th_img, img, origin)

        im_with_keypoints = (
            self.fit_keypoints(image, varim, points) if len(points) > 0 else None
        )

        if im_with_keypoints is not None:
            alpha = self.empty_alpha.copy()
            alpha[..., :3] = im_with_keypoints
            alpha[..., 3] = im_with_keypoints[..., 2]
            self.image_items[-1][0].setImage(alpha, autoLevels=False)

    @Slot()
    def extract_psf(self):
        '''Initiates the PSF extraction main thread worker.'''
        if self.stack_handler is None:
            return

        filename, _ = getSaveFileName(
            self,
            'Save PSF',
            filter='PSF HDF5 files (*.psf.h5)',
            directory=os.path.dirname(self.path),
        )

        if len(filename) > 0:

            def done(res):
                self.get_param(Parameters.LOCALIZE).setOpts(enabled=True)
                self.get_param(Parameters.EXTRACT_PSF).setOpts(enabled=True)
                if res is not None:
                    self.localizedData.emit(filename)

            print('\nPSF Extraction Protocol:')
            # Any other args, kwargs are passed to the run function
            self.worker = QThreadWorker(self.extraction_protocol, filename)
            self.worker.signals.result.connect(done)

            # Execute
            self.get_param(Parameters.LOCALIZE).setOpts(enabled=False)
            self.get_param(Parameters.EXTRACT_PSF).setOpts(enabled=False)
            self._threadpool.start(self.worker)

    def extraction_protocol(self, filename: str, **kwargs):
        method = self.fitting_options_widget.get_fitting_method()
        frames_list = None
        params = None
        crlbs = None
        loglike = None

        if (
            method == FittingMethod._2D_Phasor_CPU
            or not cuda.is_available()
            or not self.get_param(Parameters.LOCALIZE_GPU).value()
        ):
            print('\nCPU Fit')
            # Any other args, kwargs are passed to the run function
            frames_list, params, crlbs, loglike = self.localizeStackCPU()
        else:
            print('\nGPU Fit')
            # Any other args, kwargs are passed to the run function
            frames_list, params, crlbs, loglike = self.localizeStackGPU()

        print('\nPSF Extraction:')
        results = psf.get_psf_rois(
            self.stack_handler,
            frames_list,
            params,
            crlbs,
            loglike,
            method.value,
            self.get_param(Parameters.PIXEL_SIZE).value(),
            self.get_param(Parameters.PSF_ZSTEP).value(),
            self.get_param(Parameters.ROI_SIZE).value(),
            self.get_param(Parameters.PSF_UPSAMPLE).value(),
            self.get_roi_info(),
            self.get_param(Parameters.Z0_ENABLED).value(),
            self.get_param(Parameters.Z0_METHOD).value(),
        )

        results.save_hdf(filename)

        return results

    @Slot()
    def localize(self):
        '''Initiates the localization main thread worker.'''
        if self.stack_handler is None:
            return

        if not self.export_options.exec():
            return

        filename, _ = getSaveFileName(
            self,
            'Save localizations',
            filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
            directory=os.path.dirname(self.path),
        )

        if len(filename) > 0:
            self.export_metadata_to_file(filename)
            method = self.fitting_options_widget.get_fitting_method()

            def done(res):
                self.get_param(Parameters.LOCALIZE).setOpts(enabled=True)
                self.get_param(Parameters.EXTRACT_PSF).setOpts(enabled=True)
                if res is not None:
                    self.fittingResults.extend(res)
                    self.export_loc(filename)
                    self.localizedData.emit(filename)

            if (
                method == FittingMethod._2D_Phasor_CPU
                or not cuda.is_available()
                or not self.get_param(Parameters.LOCALIZE_GPU).value()
            ):
                print('\nCPU Fit')
                # Any other args, kwargs are passed to the run function
                self.worker = QThreadWorker(self.localizeStackCPU)
                self.worker.signals.result.connect(done)
            else:
                print('\nGPU Fit')
                # Any other args, kwargs are passed to the run function
                self.worker = QThreadWorker(self.localizeStackGPU)
                self.worker.signals.result.connect(done)

            # Execute
            self.get_param(Parameters.LOCALIZE).setOpts(enabled=False)
            self.get_param(Parameters.EXTRACT_PSF).setOpts(enabled=False)
            self._threadpool.start(self.worker)

    def _initialize_parameters(self):
        '''
        Common method to initialize parameters, filters, and detectors.
        '''
        # method
        method = self.fitting_options_widget.get_fitting_method()

        # new instance of FittingResults
        self.fittingResults = FittingResults(
            ResultsUnits.Pixel,  # unit is pixels
            self.get_param(Parameters.PIXEL_SIZE).value(),  # pixel projected size
            method,
        )

        # Filters + Blob detector params
        filter_obj = self.fitting_options_widget.get_image_filter()

        tm_enabled = self.get_param(Parameters.TM_FILTER_ENABLED).value()

        tm_window = (
            self.get_param(Parameters.TM_FILTER_WINDOW_SIZE).value()
            if tm_enabled
            else 0
        )

        detector = self.fitting_options_widget.get_detector()

        # ROI
        roi_info = self.get_roi_info()

        min_max = None
        if not self.get_param(Parameters.AUTO_STRETCH).value():
            min_max = tuple(map(math.ceil, self.lr_0.getRegion()))

        # varim
        varim = None
        offset = 0
        gain = 1
        if self.cmos_maps_group.active.isChecked():
            res = self.cmos_maps_group.getMaps()
            if res:
                gain = res[0]
                offset = res[1]
                varim = res[2]

        rel_threshold = self.get_param(Parameters.RELATIVE_THRESHOLD_MIN).value()
        max_threshold = self.get_param(Parameters.RELATIVE_THRESHOLD_MAX).value()
        roi_size = self.get_param(Parameters.ROI_SIZE).value()

        PSFparam = np.array([self.get_param(Parameters.INITIAL_SIGMA).value()])

        return {
            'method': method,
            'gain': gain,
            'offset': offset,
            'varim': varim,
            'roi_info': roi_info,
            'irange': min_max,
            'rel_threshold': rel_threshold,
            'max_threshold': max_threshold,
            'tm_window': tm_window,
            'roi_size': roi_size,
            'PSFparam': PSFparam,
            'filter': filter_obj,
            'detector': detector,
        }

    def localizeStackCPU(self, **kwargs):
        '''
        CPU Localization main thread worker function.
        '''
        options = self._initialize_parameters()

        frames_list, params, crlbs, loglike = localizeStackCPU(
            self.stack_handler, options['filter'], options['detector'], options
        )

        return frames_list, params, crlbs, loglike

    def localizeStackGPU(self, **kwargs):
        '''GPU Localization main thread worker function.'''
        options = self._initialize_parameters()

        print('\nCollecting Prefit ROIs...')
        start = QDateTime.currentDateTime()  # timer

        PSFparam = options['PSFparam']

        # ROI
        roi_info = options['roi_info']
        if roi_info is not None:
            origin = roi_info[0]  # ROI (x,y)

        kwargs = {
            'gain': options['gain'],
            'offset': options['offset'],
            'varim': options['varim'],
            'roi_info': options['roi_info'],
            'irange': options['irange'],
            'rel_threshold': options['rel_threshold'],
            'max_threshold': options['max_threshold'],
            'tm_window': options['tm_window'],
            'roi_size': options['roi_size'],
        }

        roi_list, varim_list, coord_list, frames_list = get_rois_lists_GPU(
            self.stack_handler, options['filter'], options['detector'], **kwargs
        )

        roi_list = np.vstack(roi_list)
        coord_list = np.vstack(coord_list)
        varim_list = None if options['varim'] is None else np.vstack(varim_list)

        print(
            '\nROIs collected in',
            time_string_ms(start.msecsTo(QDateTime.currentDateTime())),
        )

        if options['method'] == FittingMethod._2D_Gauss_MLE_fixed_sigma:
            params, crlbs, loglike = GPUmleFit_LM(roi_list, 1, PSFparam, varim_list, 0)
        elif options['method'] == FittingMethod._2D_Gauss_MLE_free_sigma:
            params, crlbs, loglike = GPUmleFit_LM(roi_list, 2, PSFparam, varim_list, 0)
        elif options['method'] == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
            params, crlbs, loglike = GPUmleFit_LM(roi_list, 4, PSFparam, varim_list, 0)
        elif options['method'] == FittingMethod._3D_Gauss_MLE_cspline_sigma:
            params, crlbs, loglike = GPUmleFit_LM(roi_list, 5, PSFparam, varim_list, 0)

        params = params.astype(np.float64, copy=False)
        crlbs = crlbs.astype(np.float64, copy=False)
        loglike = loglike.astype(np.float64, copy=False)
        frames_list = np.array(frames_list, dtype=np.int64)

        if params is not None:
            params[:, :2] += np.array(coord_list)
            if len(params) > 0 and roi_info is not None:
                params[:, 0] += origin[0]
                params[:, 1] += origin[1]

        print('\nDone...', time_string_ms(start.msecsTo(QDateTime.currentDateTime())))

        return frames_list, params, crlbs, loglike

    def export_loc(self, filename=None):
        '''Exports the fitting results into a file.

        Parameters
        ----------
        filename : str, optional
            file path if None a save file dialog is shown, by default None
        '''
        if self.fittingResults is None:
            return

        if filename is None:
            if not self.export_options.exec():
                return

            filename, _ = getSaveFileName(
                self,
                'Export localizations',
                filter='HDF5 files (*.h5);;TSV Files (*.tsv)',
                directory=os.path.dirname(self.path),
            )

        if len(filename) > 0:
            options = self.export_options.toList()

            dataFrame = self.fittingResults.dataFrame()
            exp_columns = []
            for col in dataFrame.columns:
                if col in options:
                    exp_columns.append(col)

            if '.tsv' in filename:
                dataFrame.to_csv(
                    filename,
                    index=False,
                    columns=exp_columns,
                    float_format=self.export_options.export_precision.text(),
                    sep='\t',
                    encoding='utf-8',
                )
            elif '.h5' in filename:
                dataFrame[exp_columns].to_hdf(
                    filename, key='microEye', index=False, complevel=0
                )

            if 'Super-res image' in options:
                pass
                # sres_img = self.renderLoc()
                # tf.imsave(
                #     filename.replace('.tsv', '_super_res.tif'),
                #     sres_img,
                #     photometric='minisblack',
                #     append=True,
                #     bigtiff=True,
                #     ome=False)

    def update_lists(self, result: np.ndarray):
        '''Extends the fitting results by results emitted
        by a thread worker.

        Parameters
        ----------
        result : np.ndarray
            [description]
        '''
        if result is not None:
            with self._lock:
                self.fittingResults.extend(result)
        self.thread_done += 1

    def get_protocol_metadata(self, path):
        state = self.fitting_options_widget.param_tree.saveState(filter='user')
        metadata = {
            'images': self.path,
            'localizations file': path,
            'intensity range': tuple(map(math.ceil, self.lr_0.getRegion())),
            'cmos maps': {
                'enabled': self.cmos_maps_group.active.isChecked(),
            },
        }
        dir_name = os.path.dirname(path)
        root_name, _ = os.path.splitext(os.path.basename(path))
        metadata['path'] = f'{dir_name}/{root_name}_protocol.json'

        state['Others'] = metadata
        return state

    def set_protocol_from_metadata(self, metadata: dict):
        others = metadata.pop('Others', None)
        self.fitting_options_widget.param_tree.restoreState(metadata)
        # Set intensity scaling
        lr_range = others.get('intensity range', None)
        if lr_range:
            self.lr_0.setRegion(tuple(map(math.ceil, lr_range)))

        # Set cmos maps
        cmos_maps_metadata: dict = others.get('cmos maps', {})
        self.cmos_maps_group.active.setChecked(
            cmos_maps_metadata.get('enabled', self.cmos_maps_group.active.isChecked())
        )

    def import_metadata_from_file(self, filepath=None):
        if filepath is None:
            filepath, _ = getOpenFileName(
                self,
                'Import Metadata',
                filter='JSON files (*.json);;All Files (*)',
                directory=os.path.dirname(self.path),
            )

        if len(filepath) > 0:
            try:
                with open(filepath) as file:
                    metadata = json.load(file)
                    # Do something with the imported metadata
                    self.set_protocol_from_metadata(metadata)
            except Exception as e:
                # Handle any errors that might occur during file reading
                print(f'Error importing metadata: {e}')

    def export_metadata_to_file(self, filepath=None):
        if filepath is None:
            filepath, _ = getSaveFileName(
                self,
                'Export Metadata',
                filter='JSON files (*.json);;All Files (*)',
                directory=os.path.dirname(self.path),
            )

        if len(filepath) > 0:
            try:
                metadata = self.get_protocol_metadata(filepath)
                with open(metadata['Others']['path'], 'w') as file:
                    json.dump(metadata, file, indent=2)
            except Exception as e:
                # Handle any errors that might occur during file writing
                print(f'Error exporting metadata: {e}')

    def closeEvent(self, event):
        if (
            self.get_param(Parameters.SAVE_CROPPED_IMAGE).opts['enabled']
            and self.get_param(Parameters.LOCALIZE).opts['enabled']
            and self.get_param(Parameters.EXTRACT_PSF).opts['enabled']
        ):
            # Ask the user if they really want to close the widget
            reply = QtWidgets.QMessageBox.question(
                self,
                'Confirmation',
                'Are you sure you want to close the widget?',
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No,
            )

            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                # User clicked "Yes," close the widget
                self.stack_handler.close()
                event.accept()
            else:
                # User clicked "No," ignore the close event
                event.ignore()
        else:
            event.ignore()
            QtWidgets.QMessageBox.warning(
                self, 'Warning', 'Cannot close while workers are active!'
            )

    def getStackHandler(self, path: str, mask_pattern: str = None):
        '''
        Get the image sequence handler based on the provided
        path and mask pattern.

        Parameters
        ----------
        path : str
            The path to the image sequence.
        mask_pattern : str, optional
            The pattern for masking the image sequence in case of Tiff sequence,
            by default None.

        Returns
        -------
        Union[ZarrImageSequence, TiffSeqHandler]
            An instance of the image sequence handler.
        '''
        if os.path.isdir(path) and not path.endswith('.zarr'):
            if not mask_pattern:
                raise ValueError('No mask pattern provided for a directory path')
            return TiffSeqHandler(
                tf.TiffSequence(f'{path}/{mask_pattern}')
            )
        elif os.path.isdir(path) and path.endswith('.zarr'):
            return ZarrImageSequence(path)
        elif os.path.isfile(path) and (path.endswith('.tif') or path.endswith('.tiff')):
            return TiffSeqHandler(tf.TiffSequence([path]))
        else:
            raise ValueError('Unsupported file format')
