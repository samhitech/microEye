
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
