
import cv2
import numba
import numpy as np
import pandas as pd
import pyqtgraph as pg

from ..Filters import *
from ..uImage import uImage
from .phasor_fit import *
from .gaussian_fit import *
from .results import *


def get_blob_detector(
        params: cv2.SimpleBlobDetector_Params = None) \
        -> cv2.SimpleBlobDetector:
    if params is None:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        params.minDistBetweenBlobs = 0

        # Change thresholds
        # params.minThreshold = 0
        # params.maxThreshold = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 2.0
        params.maxArea = 80.0

        # Filter by Circularity
        params.filterByCircularity = False
        # params.minCircularity = 1

        # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 1

        # Filter by Inertia
        params.filterByInertia = False
        # params.minInertiaRatio = 1

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        return cv2.SimpleBlobDetector(params)
    else:
        return cv2.SimpleBlobDetector_create(params)


def localize_frame(
            index,
            image: np.ndarray,
            filtered: np.ndarray,
            filter: AbstractFilter,
            params: cv2.SimpleBlobDetector_Params,
            threshold=255,
            method=FittingMethod._2D_Phasor_CPU):

    uImg = uImage(filtered)

    uImg.equalizeLUT(None, True)

    if filter is BandpassFilter:
        filter._show_filter = False
        filter._refresh = False

    img = filter.run(uImg._view)

    # Detect blobs.
    _, th_img = cv2.threshold(
            img,
            threshold,
            255,
            cv2.THRESH_BINARY)

    keypoints = get_blob_detector(params).detect(th_img)

    points: np.ndarray = cv2.KeyPoint_convert(keypoints)

    if method == FittingMethod._2D_Phasor_CPU:
        result = phasor_fit(image, points)
    elif method == FittingMethod._2D_Gauss_MLE_CPU:
        result = gaussian_2D_fit(image, points)

    if result is not None:
        result[:, 4] = [index + 1] * result.shape[0]

    return result


class CV_BlobDetector:
    def __init__(self, **kwargs) -> None:
        self.set_blob_detector_params(**kwargs)

    def set_blob_detector_params(
            self, min_threshold=0, max_threshold=255,
            minArea=1.5, maxArea=80,
            minCircularity=None, minConvexity=None,
            minInertiaRatio=None, blobColor=255,
            minDistBetweenBlobs=0) -> cv2.SimpleBlobDetector_Params:
        # Setup SimpleBlobDetector parameters.
        self.params = cv2.SimpleBlobDetector_Params()

        self.params.filterByColor = True
        self.params.blobColor = blobColor

        self.params.minDistBetweenBlobs = minDistBetweenBlobs

        # Change thresholds
        self.params.minThreshold = float(min_threshold)
        self.params.maxThreshold = max_threshold

        # Filter by Area.
        self.params.filterByArea = True
        self.params.minArea = minArea
        self.params.maxArea = maxArea

        # Filter by Circularity
        if minCircularity is None:
            self.params.filterByCircularity = False
            self.params.minCircularity = 0
        else:
            self.params.filterByCircularity = True
            self.params.minCircularity = minCircularity

        # Filter by Convexity
        if minConvexity is None:
            self.params.filterByConvexity = False
            self.params.minConvexity = 0
        else:
            self.params.filterByConvexity = True
            self.params.minConvexity = minConvexity

        # Filter by Inertia
        if minInertiaRatio is None:
            self.params.filterByInertia = False
            self.params.minInertiaRatio = 0
        else:
            self.params.filterByInertia = True
            self.params.minInertiaRatio = minInertiaRatio

        self.detector = get_blob_detector(self.params)

    def find_peaks(self, img: np.ndarray):

        keypoints = self.detector.detect(img)

        return cv2.KeyPoint_convert(keypoints)

    def find_peaks_preview(self, th_img: np.ndarray, img: np.ndarray):
        keypoints = self.detector.detect(th_img)

        im_with_keypoints = cv2.drawKeypoints(
            img, keypoints, np.array([]),
            (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return cv2.KeyPoint_convert(keypoints), im_with_keypoints


class BlobDetectionWidget(QGroupBox):
    update = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.detector = CV_BlobDetector()

        self._layout = QFormLayout()
        self.setTitle('OpenCV Blob Approx. Localization')
        self.setLayout(self._layout)

        self.minArea = QDoubleSpinBox()
        self.minArea.setMinimum(0)
        self.minArea.setMaximum(1024)
        self.minArea.setSingleStep(0.05)
        self.minArea.setValue(1.5)
        self.minArea.valueChanged.connect(self.value_changed)

        self.maxArea = QDoubleSpinBox()
        self.maxArea.setMinimum(0)
        self.maxArea.setMaximum(1024)
        self.maxArea.setSingleStep(0.05)
        self.maxArea.setValue(80.0)
        self.maxArea.valueChanged.connect(self.value_changed)

        self.minCircularity = QDoubleSpinBox()
        self.minCircularity.setMinimum(0)
        self.minCircularity.setMaximum(1)
        self.minCircularity.setSingleStep(0.05)
        self.minCircularity.setValue(0)
        self.minCircularity.valueChanged.connect(self.value_changed)

        self.minConvexity = QDoubleSpinBox()
        self.minConvexity.setMinimum(0)
        self.minConvexity.setMaximum(1)
        self.minConvexity.setSingleStep(0.05)
        self.minConvexity.setValue(0)
        self.minConvexity.valueChanged.connect(self.value_changed)

        self.minInertiaRatio = QDoubleSpinBox()
        self.minInertiaRatio.setMinimum(0)
        self.minInertiaRatio.setMaximum(1)
        self.minInertiaRatio.setSingleStep(0.05)
        self.minInertiaRatio.setValue(0)
        self.minInertiaRatio.valueChanged.connect(self.value_changed)

        self._layout.addRow(
            QLabel('Min area:'),
            self.minArea)
        self._layout.addRow(
            QLabel('Max area:'),
            self.maxArea)
        # self.controls_layout.addWidget(self.minCircularity)
        # self.controls_layout.addWidget(self.minConvexity)
        # self.controls_layout.addWidget(self.minInertiaRatio)

    def value_changed(self, value):
        self.detector.set_blob_detector_params(
            minArea=self.minArea.value(),
            maxArea=self.maxArea.value()
        )
        self.update.emit()
