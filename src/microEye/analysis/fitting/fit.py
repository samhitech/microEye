import concurrent.futures
import threading
from typing import Union

import cv2
import numpy as np

from ...shared.uImage import TiffSeqHandler, ZarrImageSequence, uImage
from ..filters import *
from .phasor_fit import *
from .pyfit3Dcspline import CPUmleFit_LM, get_roi_list, get_roi_list_CMOS
from .results import *


def get_blob_detector(
        params: cv2.SimpleBlobDetector_Params = None) \
        -> cv2.SimpleBlobDetector:
    if params is None:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        params.minDistBetweenBlobs = 1

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
        '''
        Initialize the CV_BlobDetector instance.

        Parameters
        ----------
        **kwargs : Any
            Additional keyword arguments to customize blob detector parameters.
            Check CV_BlobDetector.set_blob_detector_params for more info.
        '''
        self.set_blob_detector_params(**kwargs)

    def set_blob_detector_params(
            self, min_threshold=0, max_threshold=255,
            minArea=1.5, maxArea=80,
            minCircularity=None, minConvexity=None,
            minInertiaRatio=None, blobColor=255,
            minDistBetweenBlobs=1) -> cv2.SimpleBlobDetector_Params:
        '''
        Set parameters for the blob detector.

        Parameters
        ----------
        min_threshold : float, optional
            Minimum threshold for blob detection (default: 0).
        max_threshold : float, optional
            Maximum threshold for blob detection (default: 255).
        minArea : float, optional
            Minimum area of blobs (default: 1.5).
        maxArea : float, optional
            Maximum area of blobs (default: 80).
        minCircularity : float or None, optional
            Minimum circularity of blobs. If None,
            circularity filtering is disabled (default: None).
        minConvexity : float or None, optional
            Minimum convexity of blobs. If None,
            convexity filtering is disabled (default: None).
        minInertiaRatio : float or None, optional
            Minimum inertia ratio of blobs. If None,
            inertia ratio filtering is disabled (default: None).
        blobColor : int, optional
            Blob color (default: 255).
        minDistBetweenBlobs : float, optional
            Minimum distance between blobs (default: 0).
        '''
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
            # self.params.minCircularity = 0
        else:
            self.params.filterByCircularity = True
            self.params.minCircularity = minCircularity

        # Filter by Convexity
        if minConvexity is None:
            self.params.filterByConvexity = False
            # self.params.minConvexity = 0
        else:
            self.params.filterByConvexity = True
            self.params.minConvexity = minConvexity

        # Filter by Inertia
        if minInertiaRatio is None:
            self.params.filterByInertia = False
            # self.params.minInertiaRatio = 0
        else:
            self.params.filterByInertia = True
            self.params.minInertiaRatio = minInertiaRatio

        self.detector = get_blob_detector(self.params)

    def find_peaks(self, img: np.ndarray):
        '''
        Find blob peaks in the given image.

        Parameters
        ----------
        img : numpy.ndarray
            Input image.

        Returns
        -------
        keypoints : numpy.ndarray
            Array of blob keypoints.
        '''
        keypoints = self.detector.detect(img)

        return cv2.KeyPoint_convert(keypoints)

    def find_peaks_preview(self, th_img: np.ndarray, img: np.ndarray):
        '''
        Find blob peaks in the thresholded image and return keypoints and preview image.

        Parameters
        ----------
        th_img : numpy.ndarray
            Thresholded input image.
        img : numpy.ndarray
            Original input image.

        Returns
        -------
        keypoints : numpy.ndarray
            Array of blob keypoints.
        im_with_keypoints : numpy.ndarray
            Image with keypoints drawn.
        '''
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

        self._layout = QHBoxLayout()
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

        min_label = QLabel('Min area:')
        max_label = QLabel('Max area:')
        min_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        max_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        self._layout.addWidget(min_label)
        self._layout.addWidget(self.minArea)
        self._layout.addWidget(max_label)
        self._layout.addWidget(self.maxArea)
        # self.controls_layout.addWidget(self.minCircularity)
        # self.controls_layout.addWidget(self.minConvexity)
        # self.controls_layout.addWidget(self.minInertiaRatio)

    def value_changed(self, value):
        self.detector.set_blob_detector_params(
            minArea=self.minArea.value(),
            maxArea=self.maxArea.value()
        )
        self.update.emit()

    def get_metadata(self):
        return {
            'name': 'OpenCV Blob Detection',
            'min area': self.minArea.value(),
            'max area': self.maxArea.value(),
            'filter by area': True,
            'other': 'Check default parameters in CV_BlobDetector'
        }


def pre_localize_frame(**kwargs):
    '''
    Perform pre-localization processing on a frame.

    Parameters
    ----------
    index : int
        Index of the frame.
    image : np.ndarray
        Input frame image.
    varim : np.ndarray
        Variance image.
    filter : AbstractFilter
        Image filter.
    detector : AbstractDetector
        Detection algorithm.
    roi_info : tuple, optional
        Region of interest information ((x,y), (w,h)) and (default is None).
    temp : TemporalMedianFilter, optional
        Temporal median filter (default is None).
    stack_handler : Union[TiffSeqHandler, ZarrImageSequence], optional
        Handler for the image stack needed in case of using TemporalMedianFilter
        (default is None).
    irange : tuple[int, int], optional
        Intensity range (default is None for autoscaling).
    rel_threshold : float, optional
        Relative intensity threshold (default is 0.4).
    max_threshold : float, optional
        Maximum intensity threshold (default is 1.0).
    roi_size : int, optional
        Size of the region of interest (default is 13).
    method : FittingMethod, optional
        Fitting method (default is FittingMethod._2D_Phasor_CPU).

    Returns
    -------
    frames : np.ndarray
        Localized frames.
    params : np.ndarray
        Localization parameters.
    crlbs : np.ndarray
        Cramer-Rao lower bounds.
    loglike : np.ndarray
        Log-likelihood values.
    '''
    index: int = kwargs['index']
    image: np.ndarray = kwargs['image']
    varim: np.ndarray = kwargs['varim']
    filter: AbstractFilter = kwargs['filter']
    detector: AbstractDetector = kwargs['detector']
    roi_info = kwargs.get('roi_info', None)
    temp: TemporalMedianFilter = kwargs.get('temp', None)
    stack_handler: Union[
        TiffSeqHandler, ZarrImageSequence] = kwargs.get('stack_handler', None)
    irange = kwargs.get('irange', None)
    rel_threshold: float = kwargs.get('rel_threshold', 0.4)
    max_threshold: float = kwargs.get('max_threshold', 1.0)
    roi_size: int = kwargs.get('roi_size', 13)
    method: FittingMethod = kwargs.get('method', FittingMethod._2D_Phasor_CPU)

    # apply the temporal median filter
    if temp is not None and stack_handler is not None:
        frames = temp.getFrames(index, stack_handler)
        filtered = temp.run(image, frames, roi_info)
    else:
        filtered = image.copy()


    # crop image to ROI
    if roi_info is not None:
        origin = roi_info[0]  # ROI (x,y)
        dim = roi_info[1]  # ROI (w,h)
        image = image[
            int(origin[1]):int(origin[1] + dim[1]),
            int(origin[0]):int(origin[0] + dim[0])]
        filtered = filtered[
            int(origin[1]):int(origin[1] + dim[1]),
            int(origin[0]):int(origin[0] + dim[0])]

    frames, params, crlbs, loglike = localize_frame(
        index,
        image,
        filtered,
        varim,
        filter,
        detector,
        irange=irange,
        rel_threshold=rel_threshold,
        max_threshold=max_threshold,
        method=method,
        roi_size=roi_size
    )

    if params is not None:
        if len(params) > 0 and roi_info is not None:
            params[:, 0] += origin[0]
            params[:, 1] += origin[1]

    return frames, params, crlbs, loglike

def localize_frame(
        index: int, image: np.ndarray, filtered: np.ndarray,
        varim: np.ndarray, filter: AbstractFilter,
        detector: AbstractDetector, **kwargs):
    '''
    Localize features in a frame using various fitting methods.

    Parameters
    ----------
    index : int
        Index of the frame.
    image : np.ndarray
        Original frame image.
    filtered : np.ndarray
        Filtered frame image.
    varim : np.ndarray
        Variance image.
    filter : AbstractFilter
        Image filter.
    detector : AbstractDetector
        Detection algorithm.
    irange : tuple[int, int], optional
        Intensity range (default is None for autoscaling).
    rel_threshold : float, optional
        Relative intensity threshold (default is 0.4).
    max_threshold : float, optional
        Maximum intensity threshold (default is 1.0).
    PSFparam : Optional
        Parameters for the point spread function.
    roi_size : int, optional
        Size of the region of interest (default is 13).
    method : FittingMethod, optional
        Fitting method (default is FittingMethod._2D_Phasor_CPU).

    Returns
    -------
    frames : np.ndarray
        Localized frames.
    params : np.ndarray
        Localization parameters.
    crlbs : np.ndarray
        Cramer-Rao lower bounds.
    loglike : np.ndarray
        Log-likelihood values.
    '''
    PSFparam = kwargs.get('PSFparam', np.array([1.5]))
    irange: tuple[int, int] = kwargs.get('irange', None)
    rel_threshold: float = kwargs.get('rel_threshold', 0.4)
    max_threshold: float = kwargs.get('max_threshold', 1.0)
    roi_size: int = kwargs.get('roi_size', 13)
    method: int = kwargs.get('method', FittingMethod._2D_Phasor_CPU)

    uImg = uImage(filtered)

    uImg.equalizeLUT(irange, True)

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
    if max_threshold < 1.0:
        _, th2 = cv2.threshold(
            img,
            np.max(img) * max_threshold,
            1,
            cv2.THRESH_BINARY_INV)
        th_img = th_img * th2

    points: np.ndarray = detector.find_peaks(th_img)

    if method == FittingMethod._2D_Phasor_CPU:
        params = phasor_fit(image, points, roi_size=roi_size)
        crlbs, loglike = None, None
    else:
        if varim is None:
            rois, coords = get_roi_list(image, points, roi_size)
            varims = None
        else:
            rois, varims, coords = get_roi_list_CMOS(
                image, varim, points, roi_size)

        if method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 1, PSFparam, varims, 0)
        elif method == FittingMethod._2D_Gauss_MLE_free_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 2, PSFparam, varims, 0)
        elif method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 4, PSFparam, varims, 0)
        elif method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
            params, crlbs, loglike = CPUmleFit_LM(
                rois, 5, PSFparam, varims, 0)

        params[:, :2] += coords

    frames = [index + 1] * params.shape[0] if params is not None else None

    return frames, params, crlbs, loglike

def get_rois_lists_GPU(
        stack_handler: Union[TiffSeqHandler, ZarrImageSequence],
        filter: AbstractFilter,
        detector: AbstractDetector, **kwargs):
    gain: Union[float, np.ndarray] = kwargs.get('gain', 1)
    offset: Union[float, np.ndarray] = kwargs.get('offset', 0)
    varim: np.ndarray = kwargs.get('varim', None)
    roi_info = kwargs.get('roi_info', None)
    irange = kwargs.get('irange', None)
    rel_threshold: float = kwargs.get('rel_threshold', 0.4)
    max_threshold: float = kwargs.get('max_threshold', 1.0)
    tm_window: int = kwargs.get('tm_window', 0)
    roi_size: int = kwargs.get('roi_size', 13)

    roi_list = []
    varim_list = []
    coord_list = []
    frames_list = []
    counter = np.zeros(1)

    for k in range(len(stack_handler)):
        cycle = time.time_ns()
        image = stack_handler.getSlice(k, 0, 0)

        image = image * gain
        image = image - offset

        # apply the median filter
        if tm_window > 1:
            temp = TemporalMedianFilter()
            temp._temporal_window = tm_window
            frames = temp.getFrames(k, stack_handler)
            filtered = temp.run(image, frames, roi_info)
        else:
            filtered = image.copy()

        # crop image to ROI
        if roi_info is not None:
            origin = roi_info[0]  # ROI (x,y)
            dim = roi_info[1]  # ROI (w,h)
            image = image[
                int(origin[1]):int(origin[1] + dim[1]),
                int(origin[0]):int(origin[0] + dim[0])]
            filtered = filtered[
                int(origin[1]):int(origin[1] + dim[1]),
                int(origin[0]):int(origin[0] + dim[0])]

        uImg = uImage(filtered)

        uImg.equalizeLUT(irange, True)

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
        if max_threshold < 1.0:
            _, th2 = cv2.threshold(
                img,
                np.max(img) * max_threshold,
                1,
                cv2.THRESH_BINARY_INV)
            th_img = th_img * th2

        points: np.ndarray = detector.find_peaks(th_img)

        if len(points) > 0:
            if varim is None:
                rois, coords = get_roi_list(image, points, roi_size)
            else:
                rois, varims, coords = get_roi_list_CMOS(
                    image, varim, points, roi_size)
                varim_list += [varims]

            roi_list += [rois]
            coord_list += [coords]
            frames_list += [k + 1] * rois.shape[0]

        counter[0] = counter[0] + 1

        cycle = time.time_ns() - cycle
        print(
            f'index: {100*counter[0]/len(stack_handler):.2f}% {cycle/1e9:.6f} s    ',
            end='\r')

    return roi_list, varim_list, coord_list, frames_list
