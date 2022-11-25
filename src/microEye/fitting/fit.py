
from typing import Union
import cv2
import numpy as np

from microEye.fitting.pyfit3Dcspline.mainfunctions import get_roi_list_CMOS

from ..Filters import *
from ..uImage import TiffSeqHandler, uImage
from .phasor_fit import *
from .results import *
from .pyfit3Dcspline import CPUmleFit_LM, get_roi_list


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


class AbstractDetector:
    def __init__(self) -> None:
        pass

    def find_peaks(self, img: np.ndarray):
        return None

    def find_peaks_preview(self, th_img: np.ndarray, img: np.ndarray):
        return None, img


class CV_BlobDetector(AbstractDetector):
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


def pre_localize_frame(
        index,
        tiffSeq_Handler: Union[TiffSeqHandler, ZarrImageSequence],
        image: np.ndarray, varim: np.ndarray,
        temp: TemporalMedianFilter,
        filter: AbstractFilter,
        detector: AbstractDetector,
        roiInfo,
        rel_threshold=0.4,
        roiSize=13,
        method=FittingMethod._2D_Phasor_CPU):

    filtered = image.copy()

    # apply the median filter
    if temp is not None:
        frames = temp.getFrames(index, tiffSeq_Handler)
        filtered = temp.run(image, frames, roiInfo)

    # crop image to ROI
    if roiInfo is not None:
        origin = roiInfo[0]  # ROI (x,y)
        dim = roiInfo[1]  # ROI (w,h)
        image = image[
            int(origin[1]):int(origin[1] + dim[1]),
            int(origin[0]):int(origin[0] + dim[0])]
        filtered = filtered[
            int(origin[1]):int(origin[1] + dim[1]),
            int(origin[0]):int(origin[0] + dim[0])]

    frames, params, crlbs, loglike = localize_frame(
        index=index,
        image=image,
        filtered=filtered,
        varim=None,
        filter=filter,
        detector=detector,
        rel_threshold=rel_threshold,
        method=method,
        roiSize=roiSize
    )

    if params is not None:
        if len(params) > 0 and roiInfo is not None:
            params[:, 0] += origin[0]
            params[:, 1] += origin[1]

    return frames, params, crlbs, loglike


def localize_frame(
            index,
            image: np.ndarray,
            filtered: np.ndarray,
            varim: np.ndarray,
            filter: AbstractFilter,
            detector: AbstractDetector,
            rel_threshold=0.4,
            PSFparam=np.array([1.5]),
            roiSize=13,
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
            np.quantile(img, 1-1e-4) * rel_threshold,
            255,
            cv2.THRESH_BINARY)

    points: np.ndarray = detector.find_peaks(th_img)

    if method == FittingMethod._2D_Phasor_CPU:
        params = phasor_fit(image, points, roi_size=roiSize)
        crlbs, loglike = None, None
    else:
        if varim is None:
            rois, coords = get_roi_list(image, points, roiSize)
        else:
            rois, varim, coords = get_roi_list_CMOS(
                image, varim, points, roiSize)

        if method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 1, PSFparam, varim, 0)
        elif method == FittingMethod._2D_Gauss_MLE_free_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 2, PSFparam, varim, 0)
        elif method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 4, PSFparam, varim, 0)
        elif method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 5, PSFparam, varim, 0)

        params[:, :2] += coords

    if params is not None:
        frames = [index + 1] * params.shape[0]
    else:
        frames = None

    return frames, params, crlbs, loglike


# def get_approx_fit_list():
