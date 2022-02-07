
from distutils.log import debug
from typing import Iterable

import cv2
import numba
import numpy as np
import pandas as pd
import pyqtgraph as pg
from numpy.ma import count
from PyQt5.QtCore import *
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift
from scipy.interpolate import interp1d
from skimage.registration import phase_cross_correlation
from sklearn.neighbors import NearestNeighbors

from .Rendering import *
from .Filters import *
from .uImage import uImage


def phasor_fit(image: np.ndarray, points: np.ndarray,
               intensity=True, roi_size=7):
    '''Sub-pixel Phasor 2D fit

    More details:
        see doi.org/10.1063/1.5005899 (Martens et al., 2017)
    '''
    if len(points) < 1:
        return None

    sub_fit = np.zeros((points.shape[0], 4), points.dtype)

    if intensity:
        bg_mask, sig_mask = roi_mask(roi_size)

    for r in range(points.shape[0]):
        x, y = points[r, :]
        idx = int(x - roi_size//2)
        idy = int(y - roi_size//2)
        if idx < 0:
            idx = 0
        if idy < 0:
            idy = 0
        if idx + roi_size > image.shape[1]:
            idx = image.shape[1] - roi_size
        if idy + roi_size > image.shape[0]:
            idy = image.shape[0] - roi_size
        roi = image[idy:idy+roi_size, idx:idx+roi_size]
        fft_roi = np.fft.fft2(roi)
        theta_x = np.angle(fft_roi[0, 1])
        theta_y = np.angle(fft_roi[1, 0])
        if theta_x > 0:
            theta_x = theta_x - 2 * np.pi
        if theta_y > 0:
            theta_y = theta_y - 2 * np.pi
        x = idx + np.abs(theta_x) / (2 * np.pi / roi_size)
        y = idy + np.abs(theta_y) / (2 * np.pi / roi_size)
        sub_fit[r, :2] = [x, y]

        if intensity:
            sub_fit[r, 2] = intensity_estimate(roi, bg_mask, sig_mask)

    return sub_fit


def roi_mask(roi_size=7):

    roi_shape = [roi_size] * 2
    roi_radius = roi_size / 2

    radius_map, _ = radial_cordinate(roi_shape)

    bg_mask = radius_map > (roi_radius - 0.5)
    sig_mask = radius_map <= roi_radius

    return bg_mask, sig_mask


def intensity_estimate(roi: np.ndarray, bg_mask, sig_mask, percentile=56):

    background_map = roi[bg_mask]
    background = np.percentile(
        background_map, percentile)

    intensity = np.sum(roi[sig_mask]) - (np.sum(sig_mask) * background)

    return max(0, intensity)


class ResultsUnits:
    Pixel = 0
    Nanometer = 1


class FittingResults:

    columns = np.array([
        'frame', 'x [pixel]', 'y [pixel]', 'x [nm]', 'y [nm]', 'intensity',
        'trackID', 'neighbour_dist [nm]', 'n_merged'
    ])

    def __init__(self, unit=ResultsUnits.Pixel, pixelSize=130.0):
        '''Fitting Results

        Parameters
        ----------
        unit : int, optional
            unit of localized points, by default ResultsUnits.Pixel
        pixelSize : float, optional
            pixel size in nanometers, by default 130.0
        '''
        self.unit = unit
        self.pixelSize = pixelSize
        self.locX = []
        self.locY = []
        self.locX_nm = []
        self.locY_nm = []
        self.frame = []
        self.intensity = []
        self.trackID = []
        self.neighbour_dist = []
        self.n_merged = []

    def extend(self, data: np.ndarray):
        '''Extend results by contents of data array

        Parameters
        ----------
        data : np.ndarray
            array of shape (n, m=4), columns (X, Y, Intensity, Frame)
        '''
        if self.unit is ResultsUnits.Pixel:
            self.locX.extend(data[:, 0])
            self.locY.extend(data[:, 1])
        else:
            self.locX_nm.extend(data[:, 0])
            self.locY_nm.extend(data[:, 1])

        self.intensity.extend(data[:, 2])
        self.frame.extend(data[:, 3])

        self.trackID.extend([0] * data.shape[0])
        self.neighbour_dist.extend([0] * data.shape[0])
        self.n_merged.extend([0] * data.shape[0])

    def dataFrame(self):
        '''Return fitting results as Pandas DataFrame

        Returns
        -------
        DataFrame
            fitting results DataFrame with columns FittingResults.columns
        '''
        if self.unit is ResultsUnits.Pixel:
            loc = np.c_[
                    np.array(self.frame),
                    np.array(self.locX),
                    np.array(self.locY),
                    np.array(self.locX) * self.pixelSize,
                    np.array(self.locY) * self.pixelSize,
                    np.array(self.intensity),
                    np.array(self.trackID),
                    np.array(self.neighbour_dist),
                    np.array(self.n_merged)]
        else:
            loc = np.c_[
                    np.array(self.frame),
                    np.array(self.locX_nm) / self.pixelSize,
                    np.array(self.locY_nm) / self.pixelSize,
                    np.array(self.locX_nm),
                    np.array(self.locY_nm),
                    np.array(self.intensity),
                    np.array(self.trackID),
                    np.array(self.neighbour_dist),
                    np.array(self.n_merged)]

        return pd.DataFrame(
            loc, columns=FittingResults.columns).sort_values(
            by=FittingResults.columns[0])

    def toRender(self):
        '''Returns columns for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains X [nm], Y [nm], Intensity columns
        '''
        if self.unit is ResultsUnits.Pixel:
            return np.array(self.locX) * self.pixelSize, \
                np.array(self.locY) * self.pixelSize, \
                np.array(self.intensity)
        else:
            return np.array(self.locX_nm), \
                np.array(self.locY_nm), \
                np.array(self.intensity)

    def __len__(self):
        counts = [len(self.locX), len(self.locY),
                  len(self.locX_nm), len(self.locY_nm)]
        return np.max(counts)

    def fromFile(filename: str, pixelSize: float):
        '''Populates fitting results from a tab seperated values
        (tsv) file.

        Parameters
        ----------
        filename : str
            path to tab seperated file (.tsv)
        pixelSize : float
            projected pixel size in nanometers

        Returns
        -------
        FittingResults
            FittingResults with imported data
        '''
        dataFrame = pd.read_csv(
                filename,
                sep='\t',
                engine='python')

        return FittingResults.fromDataFrame(dataFrame, pixelSize)

    def fromDataFrame(dataFrame: pd.DataFrame, pixelSize: float):
        fittingResults = None

        if FittingResults.columns[3] in dataFrame and \
                FittingResults.columns[4] in dataFrame:
            fittingResults = FittingResults(
                ResultsUnits.Nanometer,
                pixelSize)
            fittingResults.locX_nm = \
                dataFrame[FittingResults.columns[3]]
            fittingResults.locY_nm = \
                dataFrame[FittingResults.columns[4]]
        elif FittingResults.columns[1] in dataFrame and \
                FittingResults.columns[2] in dataFrame:
            fittingResults = FittingResults(
                ResultsUnits.Pixel,
                pixelSize)
            fittingResults.locX = \
                dataFrame[FittingResults.columns[1]]
            fittingResults.locY = \
                dataFrame[FittingResults.columns[2]]
        else:
            return None

        if FittingResults.columns[0] in dataFrame:
            fittingResults.frame = dataFrame[FittingResults.columns[0]]
        else:
            fittingResults.frame = np.zeros(len(fittingResults))

        if FittingResults.columns[5] in dataFrame:
            fittingResults.intensity = dataFrame[FittingResults.columns[5]]
        else:
            fittingResults.intensity = np.ones(len(fittingResults))

        if FittingResults.columns[6] in dataFrame:
            fittingResults.trackID = dataFrame[FittingResults.columns[6]]
        else:
            fittingResults.trackID = np.zeros(len(fittingResults))

        if FittingResults.columns[7] in dataFrame:
            fittingResults.neighbour_dist = \
                dataFrame[FittingResults.columns[7]]
        else:
            fittingResults.neighbour_dist = np.zeros(len(fittingResults))

        if FittingResults.columns[8] in dataFrame:
            fittingResults.n_merged = dataFrame[FittingResults.columns[8]]
        else:
            fittingResults.n_merged = np.zeros(len(fittingResults))

        return fittingResults

    def drift_cross_correlation(self, n_bins=10, pixelSize=10, upsampling=100):
        '''Corrects the XY drift using cross-correlation measurments

        Parameters
        ----------
        n_bins : int, optional
            Number of frame bins, by default 10
        pixelSize : int, optional
            Super-res image pixel size in nanometers, by default 10
        upsampling : int, optional
            phase_cross_correlation upsampling (check skimage.registration),
            by default 100

        Returns
        -------
        tuple(FittingResults, np.ndarray)
            returns the drift corrected fittingResults and recontructed image
        '''
        unique_frames = np.unique(self.frame)
        if len(unique_frames) < 2:
            print('Drift cross-correlation failed: no frame info.')
            return

        frames_per_bin = np.floor(np.max(unique_frames) / n_bins)

        if frames_per_bin < 1:
            print('Drift cross-correlation failed: large number of bins.')
            return

        renderEngine = gauss_hist_render(pixelSize)

        data = self.dataFrame().to_numpy()

        x_max = int((np.max(data[:, 3]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)
        y_max = int((np.max(data[:, 4]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)

        grouped_data = []
        sub_images = []
        shifts = []
        frames = []

        for f in range(0, n_bins):
            group = data[(data[:, 0] >= f * frames_per_bin) &
                         (data[:, 0] < (f + 1) * frames_per_bin + 1)]
            image = renderEngine.fromArray(group[:, 3:6], (y_max, x_max))
            frames.append(f * frames_per_bin + frames_per_bin/2)
            grouped_data.append(group)
            sub_images.append(image)
            print(
                'Bins: {:d}/{:d}'.format(f + 1, n_bins),
                end="\r")

        print(
            'Shift Estimation ...',
            end="\r")
        shifts = shift_estimation(np.array(sub_images), pixelSize, upsampling)
        # for idx, img in enumerate(sub_images):
        #     shift = phase_cross_correlation(
        #         img, sub_images[0], upsample_factor=upsampling)
        #     shifts.append(shift[0] * pixelSize)
        #     print(
        #         'Shift Est.: {:d}/{:d}'.format(idx + 1, len(sub_images)),
        #         end="\r")

        shifts = np.c_[shifts, np.array(frames)]
        print(
            'Shift Correction ...',
            end="\r")

        # An one-dimensional interpolation is applied
        # to drift traces in X and Y dimensions separately.
        interpy = interp1d(
            shifts[:, -1], shifts[:, 0],
            kind='quadratic', fill_value='extrapolate')
        interpx = interp1d(
            shifts[:, -1], shifts[:, 1],
            kind='quadratic', fill_value='extrapolate')
        # And this interpolation is used to get the shift at every frame-point
        frames_new = np.arange(0, np.max(unique_frames), 1)
        interpx = interpx(frames_new)
        interpy = interpy(frames_new)

        shift_correction(interpx, interpy, data)
        # for i, (shift_x, shift_y) in enumerate(zip(interpx, interpy)):
        #     data[data[:, 0] == i, 3] -= shift_x
        #     data[data[:, 0] == i, 4] -= shift_y

        df = pd.DataFrame(
            data,
            columns=FittingResults.columns)

        drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        drift_corrected_image = renderEngine.fromArray(
            data[:, 3:6])

        return drift_corrected, drift_corrected_image, \
            (frames_new, interpx, interpy)

    def nn_trajectories(
            self, maxDistance=30, maxOff=1, neighbors=1):
        data = self.dataFrame().to_numpy()
        data[:, 6] = 0
        counter = 0
        max_frame = np.max(self.frame)

        for frameID in np.arange(0, max_frame + 1):
            for offset in np.arange(0, maxOff + 1):
                counter = nn_trajectories(
                    data,
                    frameID=frameID, nextID=frameID + offset + 1,
                    counter=counter, maxDistance=maxDistance,
                    neighbors=neighbors)
            print(
                'NN {:.2%} ...               '.format(frameID / max_frame),
                end="\r")

        df = pd.DataFrame(
            data,
            columns=FittingResults.columns).sort_values(
                by=FittingResults.columns[0])

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def merge_tracks(self, maxLength=500):
        data = self.dataFrame().to_numpy()

        print('Merging ...                         ')
        finalData = merge_localizations(data, maxLength)

        df = pd.DataFrame(
            finalData,
            columns=FittingResults.columns).sort_values(
                by=FittingResults.columns[0])

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def nearest_neighbour_merging(
            self, maxDistance=30, maxOff=1, maxLength=500, neighbors=1):
        data = self.dataFrame().to_numpy()
        data[:, 6] = 0
        counter = 0
        max_frame = np.max(self.frame)

        for frameID in np.arange(0, max_frame + 1):
            for offset in np.arange(0, maxOff + 1):
                counter = nn_trajectories(
                    data,
                    frameID=frameID, nextID=frameID + offset + 1,
                    counter=counter, maxDistance=maxDistance,
                    neighbors=neighbors)
            print(
                'NN {:.2%} ...               '.format(frameID / max_frame),
                end="\r")

        print('Merging ...                         ')
        finalData = merge_localizations(data, maxLength)

        df = pd.DataFrame(
            finalData,
            columns=FittingResults.columns).sort_values(
                by=FittingResults.columns[0])

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def drift_fiducial_marker(self):
        data = self.dataFrame().to_numpy()

        unique_frames, frame_counts = \
            np.unique(data[:, 0], return_counts=True)
        if len(unique_frames) < 2:
            print('Drift correction failed: no frame info.')
            return

        if np.max(data[:, 6]) < 1:
            print('Drift correction failed: please perform NN, no trackIDs.')
            return

        unique_tracks, track_counts = \
            np.unique(data[:, 6], return_counts=True)
        fiducial_tracks_mask = np.logical_and(
            unique_tracks > 0,
            track_counts == len(unique_frames)
        )

        fiducial_trackIDs = unique_tracks[fiducial_tracks_mask]

        if len(fiducial_trackIDs) < 1:
            print(
                'Drift correction failed: no ' +
                'fiducial markers tracks detected.')
            return
        else:
            print('{:d} tracks detected.'.format(
                len(fiducial_trackIDs)), end='\r')

        fiducial_markers = np.array((
            len(fiducial_trackIDs),
            len(unique_frames),
            data.shape[-1]))

        print(
            'Fiducial markers ...                 ', end='\r')
        for idx in np.arange(fiducial_markers.shape[0]):
            fiducial_markers[idx] = \
                data[data[:, 6] == fiducial_trackIDs[idx]]

            fiducial_markers[idx, :, 3] -= fiducial_markers[idx, 0, 3]
            fiducial_markers[idx, :, 4] -= fiducial_markers[idx, 0, 4]

        print(
            'Drift estimate ...                 ', end='\r')
        drift_x = np.mean(fiducial_markers[:, :, 3], axis=0)
        drift_y = np.mean(fiducial_markers[:, :, 4], axis=0)

        print(
            'Drift correction ...                 ', end='\r')
        for idx in np.arange(len(unique_frames)):
            frame = unique_frames[idx]
            data[data[:, 0] == frame, 3] -= drift_x[idx]
            data[data[:, 0] == frame, 4] -= drift_y[idx]

        df = pd.DataFrame(
            data,
            columns=FittingResults.columns)

        drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        return drift_corrected, \
            (unique_frames, drift_x, drift_y)


@numba.njit(parallel=True)
def nn_trajectories(
        data: np.ndarray, frameID: int, nextID: int, counter: int = 0,
        maxDistance=30, neighbors=1):
    currentFrame = data[data[:, 0] == frameID, :]
    nextFrame = data[data[:, 0] == nextID, :]

    if len(currentFrame) > 0 and len(nextFrame) > 0:
        with numba.objmode(
                currentIDs='int64[:]', neighbourIDs='int64[:]',
                distance='float64[:]'):
            nNeighbors = NearestNeighbors(n_neighbors=neighbors)
            nNeighbors.fit(nextFrame[:, 3:5])
            foundnn = nNeighbors.kneighbors(currentFrame[:, 3:5])
            foundnn = np.asarray(foundnn)

            currentIDs = np.where(foundnn[0] < maxDistance)[0]
            neighbourIDs = foundnn[
                :, foundnn[0] < maxDistance][1].astype(np.int64)
            distance = foundnn[0]

        for idx in numba.prange(len(currentIDs)):
            if nextFrame[neighbourIDs[idx], 6] == 0:
                if currentFrame[currentIDs[idx], 6] == 0:
                    currentFrame[currentIDs[idx], 6] = counter + idx + 1
                    nextFrame[neighbourIDs[idx], 6] = counter + idx + 1
                else:
                    nextFrame[neighbourIDs[idx], 6] = \
                        currentFrame[currentIDs[idx], 6]

            currentFrame[currentIDs[idx], 7] = distance[currentIDs[idx]]

        data[data[:, 0] == frameID, :] = currentFrame
        data[data[:, 0] == nextID, :] = nextFrame

    return counter + len(currentIDs)


@numba.njit(parallel=True)
def merge_localizations(data: np.ndarray, maxLength):

    with numba.objmode(
            trackIDs='float64[:]'):
        trackIDs, inverse, trackCounts = np.unique(
            data[:, 6], return_counts=True, return_inverse=True)
        mask = np.logical_and(trackIDs > 0, trackCounts <= maxLength)
        trackIDs[np.logical_not(mask)] = 0
        data[:, 6] = trackIDs[inverse]
        trackIDs, trackCounts = trackIDs[mask], trackCounts[mask]

    mergedData = np.empty((len(trackIDs),) + data.shape[1:], dtype=np.float64)

    for idx in numba.prange(len(trackIDs)):
        trackGroup = data[data[:, 6] == trackIDs[idx], :]

        mergedData[idx, 0] = np.min(trackGroup[:, 0])
        weights = trackGroup[:, 5]
        mergedData[idx, 5] = np.sum(weights)
        if mergedData[idx, 5] <= 0:
            weights = np.ones(weights.shape)
        mergedData[idx, 3] = np.sum(
            trackGroup[:, 3] * weights) / np.sum(weights)
        mergedData[idx, 4] = np.sum(
            trackGroup[:, 4] * weights) / np.sum(weights)
        mergedData[idx, 8] = len(trackGroup)

        # print(
        #     'Merging ', (idx + 1), ' / ', len(trackIDs))

    leftIndices = np.where(data[:, 6] == 0)[0]
    leftData = data[leftIndices, :]

    return np.append(mergedData, leftData, axis=0)


@numba.njit(parallel=True)
def shift_estimation(sub_images, pixelSize, upsampling):
    shifts = np.zeros((len(sub_images), 2))
    for idx in numba.prange(0, len(sub_images)):
        with numba.objmode(shift='float64[:]'):
            shift = phase_cross_correlation(
                sub_images[idx], sub_images[0], upsample_factor=upsampling)[0]
        shifts[idx, :] = shift * pixelSize
    return shifts


@numba.jit(nopython=True, parallel=True)
def shift_correction(interpx, interpy, data):
    for idx in numba.prange(0, len(interpx)):
        shift_x = interpx[idx]
        shift_y = interpy[idx]
        data[data[:, 0] == idx, 3] -= shift_x
        data[data[:, 0] == idx, 4] -= shift_y


def plot_drift(
        frames_new, interpx, interpy,
        title='Drift Cross-Correlation'):
    print(
        'Shift Plot ...',
        end="\r")

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'drift', units='nm')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'frame', units='')

    plt.setWindowTitle(title)

    # setting horizontal range
    plt.setXRange(0, np.max(frames_new))

    # setting vertical range
    plt.setYRange(0, 1)

    line1 = plt.plot(
        frames_new, interpx,
        pen='r', symbol=None,
        symbolBrush=0.2, name='x-drift')
    line1 = plt.plot(
        frames_new, interpy,
        pen='y', symbol=None,
        symbolBrush=0.2, name='y-drift')

    print(
        'Done ...',
        end="\r")


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
            filter: AbstractFilter,
            params: cv2.SimpleBlobDetector_Params,
            threshold=255):

    # median fitler is removed
    # if temp is not None:
    #     temp.getFrames(index, self.tiffSeq_Handler)
    #     image = temp.run(image)

    uImg = uImage(image)

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

    time = QDateTime.currentDateTime()
    result = phasor_fit(uImg._image, points)

    if result is not None:
        result[:, 3] = [index + 1] * points.shape[0]

    return result
