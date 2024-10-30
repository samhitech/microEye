import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional, Union

import cv2
import numpy as np
from tqdm import tqdm

from microEye.analysis.filters import *
from microEye.analysis.fitting.phasor_fit import *
from microEye.analysis.fitting.pyfit3Dcspline import (
    CPUmleFit_LM,
    get_roi_list,
    get_roi_list_CMOS,
)
from microEye.analysis.fitting.results import *
from microEye.qt import Qt, QtWidgets, Signal
from microEye.utils.uImage import TiffSeqHandler, ZarrImageSequence, uImage


def get_blob_detector(
    params: cv2.SimpleBlobDetector_Params = None,
) -> cv2.SimpleBlobDetector:
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
        self,
        min_threshold=0,
        max_threshold=255,
        minArea=1.5,
        maxArea=80,
        minCircularity=None,
        minConvexity=None,
        minInertiaRatio=None,
        blobColor=255,
        minDistBetweenBlobs=1,
    ) -> cv2.SimpleBlobDetector_Params:
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
            img,
            keypoints,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        return cv2.KeyPoint_convert(keypoints), im_with_keypoints


class BlobDetectionWidget(QtWidgets.QGroupBox):
    update = Signal()

    def __init__(self) -> None:
        super().__init__()

        self.detector = CV_BlobDetector()

        self._layout = QtWidgets.QHBoxLayout()
        self.setTitle('OpenCV Blob Approx. Localization')
        self.setLayout(self._layout)

        self.minArea = QtWidgets.QDoubleSpinBox()
        self.minArea.setMinimum(0)
        self.minArea.setMaximum(1024)
        self.minArea.setSingleStep(0.05)
        self.minArea.setValue(1.5)
        self.minArea.valueChanged.connect(self.value_changed)

        self.maxArea = QtWidgets.QDoubleSpinBox()
        self.maxArea.setMinimum(0)
        self.maxArea.setMaximum(1024)
        self.maxArea.setSingleStep(0.05)
        self.maxArea.setValue(80.0)
        self.maxArea.valueChanged.connect(self.value_changed)

        self.minCircularity = QtWidgets.QDoubleSpinBox()
        self.minCircularity.setMinimum(0)
        self.minCircularity.setMaximum(1)
        self.minCircularity.setSingleStep(0.05)
        self.minCircularity.setValue(0)
        self.minCircularity.valueChanged.connect(self.value_changed)

        self.minConvexity = QtWidgets.QDoubleSpinBox()
        self.minConvexity.setMinimum(0)
        self.minConvexity.setMaximum(1)
        self.minConvexity.setSingleStep(0.05)
        self.minConvexity.setValue(0)
        self.minConvexity.valueChanged.connect(self.value_changed)

        self.minInertiaRatio = QtWidgets.QDoubleSpinBox()
        self.minInertiaRatio.setMinimum(0)
        self.minInertiaRatio.setMaximum(1)
        self.minInertiaRatio.setSingleStep(0.05)
        self.minInertiaRatio.setValue(0)
        self.minInertiaRatio.valueChanged.connect(self.value_changed)

        min_label = QtWidgets.QLabel('Min area:')
        max_label = QtWidgets.QLabel('Max area:')
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
            minArea=self.minArea.value(), maxArea=self.maxArea.value()
        )
        self.update.emit()

    def get_metadata(self):
        return {
            'name': 'OpenCV Blob Detection',
            'min area': self.minArea.value(),
            'max area': self.maxArea.value(),
            'filter by area': True,
            'other': 'Check default parameters in CV_BlobDetector',
        }


def pre_process_frame(
    index: int,
    stack_handler: Union['TiffSeqHandler', 'ZarrImageSequence'],
    options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Optional[tuple[tuple[int, int], tuple[int, int]]]]:
    '''
    Preprocess a frame before localization.

    Parameters
    ----------
    index : int
        Index of the frame.
    stack_handler : Union[TiffSeqHandler, ZarrImageSequence]
        Image stack handler.
    options : Dict[str, Any]
        Dictionary of preprocessing options.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, Optional[tuple[tuple[int, int], tuple[int, int]]]]
        Preprocessed image, filtered image, and ROI information (if applicable).
    '''
    gain = options.get('gain', 1)
    offset = options.get('offset', 0)
    tm_window = options.get('tm_window', 0)
    roi_info = options.get('roi_info')

    image = stack_handler.getSlice(index, 0, 0)
    image = image * gain - offset

    if tm_window > 1:
        temp = TemporalMedianFilter(tm_window)
        frames = temp.getFrames(index, stack_handler)
        filtered = temp.run(image, frames, roi_info)
    else:
        filtered = image.copy()

    if roi_info is not None:
        origin, dim = roi_info
        slice_y = slice(int(origin[1]), int(origin[1] + dim[1]))
        slice_x = slice(int(origin[0]), int(origin[0] + dim[0]))
        image = image[slice_y, slice_x]
        filtered = filtered[slice_y, slice_x]

    return image, filtered, roi_info


def detect_points(
    filtered_image: np.ndarray,
    filter_obj: 'AbstractFilter',
    detector: 'AbstractDetector',
    options: dict[str, Any],
) -> np.ndarray:
    '''
    Detect points in the filtered image.

    Parameters
    ----------
    filtered_image : np.ndarray
        Filtered image.
    filter_obj : AbstractFilter
        Image filter object.
    detector : AbstractDetector
        Point detection algorithm.
    options : Dict[str, Any]
        Detection options.

    Returns
    -------
    np.ndarray
        Detected points.
    '''
    irange = options.get('irange')
    rel_threshold = options.get('rel_threshold', 0.4)
    max_threshold = options.get('max_threshold', 1.0)

    uImg = uImage(filtered_image)
    uImg.equalizeLUT(irange, True)

    if isinstance(filter_obj, BandpassFilter):
        filter_obj._show_filter = False
        filter_obj._refresh = False

    img = filter_obj.run(uImg._view)

    _, th_img = cv2.threshold(
        img, np.quantile(img, 1 - 1e-4) * rel_threshold, 255, cv2.THRESH_BINARY
    )
    if max_threshold < 1.0:
        _, th2 = cv2.threshold(
            img, np.max(img) * max_threshold, 1, cv2.THRESH_BINARY_INV
        )
        th_img *= th2

    return detector.find_peaks(th_img)


def fit_points(
    image: np.ndarray,
    points: np.ndarray,
    varim: Optional[np.ndarray],
    options: dict[str, Any],
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    '''
    Fit detected points using specified method.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    points : np.ndarray
        Detected points.
    varim : Optional[np.ndarray]
        Variance image (if applicable).
    options : Dict[str, Any]
        Fitting options.

    Returns
    -------
    tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        Fitted parameters, CRLBs, and log-likelihood values.
    '''
    method : FittingMethod = options.get('method', FittingMethod._2D_Phasor_CPU)
    roi_size = options.get('roi_size', 13)
    PSFparam = options.get('PSFparam', np.array([1]))

    if method == FittingMethod._2D_Phasor_CPU:
        params = phasor_fit(image, points, roi_size=roi_size)
        return params, None, None

    if varim is None:
        rois, coords = get_roi_list(image, points, roi_size)
        varims = None
    else:
        rois, varims, coords = get_roi_list_CMOS(image, varim, points, roi_size)

    fit_type = method.value

    params, crlbs, loglike = CPUmleFit_LM(rois, fit_type, PSFparam, varims, 0)
    params[:, :2] += coords

    return params, crlbs, loglike


def localize_frame(
    index: int,
    stack_handler: Union['TiffSeqHandler', 'ZarrImageSequence'],
    filter_obj: 'AbstractFilter',
    detector: 'AbstractDetector',
    options: dict[str, Any],
) -> tuple[list[int], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    '''
    Localize features in a frame.

    Parameters
    ----------
    index : int
        Frame index.
    stack_handler : Union[TiffSeqHandler, ZarrImageSequence]
        Image stack handler.
    filter_obj : AbstractFilter
        Image filter object.
    detector : AbstractDetector
        Point detection algorithm.
    options : Dict[str, Any]
        Localization options.

    Options
    -------
    gain : Union[float, np.ndarray]
        Gain value.
    offset : Union[float, np.ndarray]
        Offset value.
    varim : np.ndarray
        Variance image.
    roi_info : tuple, optional
        Region of interest information ((x,y), (w,h)) and (default is None).
    tm_window : int, optional
        Temporal median filter window size (default is 0).
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
    image, filtered, roi_info = pre_process_frame(index, stack_handler, options)
    points = detect_points(filtered, filter_obj, detector, options)

    if len(points) == 0:
        return None, None, None, None

    params, crlbs, loglike = fit_points(image, points, options.get('varim'), options)

    if params is not None and len(params) > 0 and roi_info is not None:
        origin = roi_info[0]
        params[:, 0] += origin[0]
        params[:, 1] += origin[1]

    frames = [index + 1] * params.shape[0] if params is not None else None

    return frames, params, crlbs, loglike


def time_string_ms(ms):
    s = int(ms // 1000)
    ms = int(ms % 1000)
    m = int(s // 60)
    s = int(s % 60)
    h = int(m // 60)
    m = int(m % 60)
    return f'{h:02d}:{m:02d}:{s:02d}.{ms:03d}'


def localizeStackCPU(
    stack_handler: Union[TiffSeqHandler, ZarrImageSequence],
    filter_obj: AbstractFilter,
    detector: AbstractDetector,
    options: dict[str, Any],
) -> tuple[list, list, list, list]:
    '''
    Localize features in an image stack using CPU parallelization.

    Parameters
    ----------
    stack_handler : Union[TiffSeqHandler, ZarrImageSequence]
        Handler for the image stack.
    filter_obj : AbstractFilter
        Image filter object.
    detector : AbstractDetector
        Point detection algorithm.
    options : Dict[str, Any]
        Dictionary of localization options.
    n_threads : int, optional
        Number of threads to use for parallel processing.
        If None, uses all available threads minus 2.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        lists of frame indices, localization parameters,
        CRLBs, and log-likelihood values.
    '''
    if options.get('n_threads') is None:
        n_threads = max(1, ThreadPoolExecutor()._max_workers - 2)

    start = datetime.now()

    all_frames, all_params, all_crlbs, all_loglike = [], [], [], []

    def process_frame(
        frame_index: int,
    ) -> tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        return localize_frame(frame_index, stack_handler, filter_obj, detector, options)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [
            executor.submit(process_frame, i) for i in range(stack_handler.shape[0])
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc='Processing frames'
        ):
            frames, params, crlbs, loglike = future.result()
            if frames is not None:
                all_frames.extend(frames)
                all_params.extend(params)
                if crlbs is not None:
                    all_crlbs.extend(crlbs)
                if loglike is not None:
                    all_loglike.extend(loglike)

    duration = (datetime.now() - start).total_seconds() * 1000
    print(f'\nDone... {time_string_ms(duration)}')

    return (
        np.array(all_frames, dtype=np.int64),
        np.array(all_params, np.float64),
        np.array(all_crlbs, np.float64) if all_crlbs else None,
        np.array(all_loglike, np.float64) if all_loglike else None,
    )


def get_rois_lists_GPU(
    stack_handler: Union[TiffSeqHandler, ZarrImageSequence],
    filter: AbstractFilter,
    detector: AbstractDetector,
    **kwargs,
):
    gain: Union[float, np.ndarray] = kwargs.get('gain', 1)
    offset: Union[float, np.ndarray] = kwargs.get('offset', 0)
    varim: np.ndarray = kwargs.get('varim')
    roi_info = kwargs.get('roi_info')
    irange = kwargs.get('irange')
    rel_threshold: float = kwargs.get('rel_threshold', 0.4)
    max_threshold: float = kwargs.get('max_threshold', 1.0)
    tm_window: int = kwargs.get('tm_window', 0)
    roi_size: int = kwargs.get('roi_size', 13)

    def process_image(k: int) -> tuple[list, list, list, list]:
        image = stack_handler.getSlice(k, 0, 0)
        image = (image * gain) - offset

        if tm_window > 1:
            temp = TemporalMedianFilter()
            temp._temporal_window = tm_window
            frames = temp.getFrames(k, stack_handler)
            filtered = temp.run(image, frames, roi_info)
        else:
            filtered = image.copy()

        if roi_info is not None:
            origin, dim = roi_info
            slice_y = slice(int(origin[1]), int(origin[1] + dim[1]))
            slice_x = slice(int(origin[0]), int(origin[0] + dim[0]))
            image = image[slice_y, slice_x]
            filtered = filtered[slice_y, slice_x]

        uImg = uImage(filtered)
        uImg.equalizeLUT(irange, True)

        if isinstance(filter, BandpassFilter):
            filter._show_filter = False
            filter._refresh = False

        img = filter.run(uImg._view)

        _, th_img = cv2.threshold(
            img, np.quantile(img, 1 - 1e-4) * rel_threshold, 255, cv2.THRESH_BINARY
        )
        if max_threshold < 1.0:
            _, th2 = cv2.threshold(
                img, np.max(img) * max_threshold, 1, cv2.THRESH_BINARY_INV
            )
            th_img *= th2

        points: np.ndarray = detector.find_peaks(th_img)

        if len(points) > 0:
            if varim is None:
                rois, coords = get_roi_list(image, points, roi_size)
                return rois, [], coords, [k + 1] * rois.shape[0]
            else:
                rois, varims, coords = get_roi_list_CMOS(image, varim, points, roi_size)
                return rois, varims, coords, [k + 1] * rois.shape[0]
        return [], [], [], []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, k) for k in range(len(stack_handler))]
        results = []
        for future in tqdm(
            as_completed(futures), total=len(stack_handler), desc='Processing images'
        ):
            results.append(future.result())

    roi_list, varim_list, coord_list, frames_list = zip(*results)
    return (
        [item for sublist in roi_list for item in sublist],
        [item for sublist in varim_list for item in sublist],
        [item for sublist in coord_list for item in sublist],
        [item for sublist in frames_list for item in sublist],
    )
