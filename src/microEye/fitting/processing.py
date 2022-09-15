
import cv2
import numba
import numpy as np
import pyqtgraph as pg
from skimage.registration import phase_cross_correlation
from sklearn.neighbors import NearestNeighbors


@numba.njit(parallel=True)
def nn_trajectories(
        data: np.ndarray, trackID: np.ndarray, nn_dist: np.ndarray,
        frameID: int, nextID: int, counter: int = 0,
        maxDistance=30, neighbors=1):
    currentFrame = data[:, 0] == frameID
    nextFrame = data[:, 0] == nextID

    if currentFrame.any() and nextFrame.any():
        with numba.objmode(
                currentIDs='int64[:]', neighbourIDs='int64[:]',
                distance='float64[:]'):
            nNeighbors = NearestNeighbors(
                n_neighbors=min(neighbors, np.sum(nextFrame)))
            nNeighbors.fit(data[nextFrame, 1:])
            foundnn = nNeighbors.kneighbors(data[currentFrame, 1:])
            foundnn = np.asarray(foundnn)

            currentIDs = np.where(foundnn[0] < maxDistance)[0]
            neighbourIDs = foundnn[
                :, foundnn[0] < maxDistance][1].astype(np.int64)
            distance = foundnn[0]

        for idx in numba.prange(len(currentIDs)):
            if trackID[nextFrame][neighbourIDs[idx]] == 0:
                if trackID[currentFrame][currentIDs[idx]] == 0:
                    trackID[currentFrame][currentIDs[idx]] = counter + idx + 1
                    trackID[nextFrame][neighbourIDs[idx]] = counter + idx + 1
                else:
                    trackID[nextFrame][neighbourIDs[idx]] = \
                        trackID[currentFrame][currentIDs[idx]]

            nn_dist[currentFrame][currentIDs[idx]] = distance[currentIDs[idx]]

        # data[data[:, 0] == frameID, :] = currentFrame
        # data[data[:, 0] == nextID, :] = nextFrame

    return counter + len(currentIDs)


@numba.njit(parallel=True)
def merge_localizations(data: np.ndarray, columns: list[str], maxLength):

    with numba.objmode(
            trackIDs='float64[:]'):
        frame_idx = columns.index('frame')
        track_idx = columns.index('trackID')
        n_idx = columns.index('n_merged')
        if 'I' in columns:
            int_idx = columns.index('I')
        elif 'intensity' in columns:
            int_idx = columns.index('intensity')
        trackIDs, inverse, trackCounts = np.unique(
            data[:, track_idx], return_counts=True, return_inverse=True)
        mask = np.logical_and(trackIDs > 0, trackCounts <= maxLength)
        trackIDs[np.logical_not(mask)] = 0
        data[:, track_idx] = trackIDs[inverse]
        trackIDs, trackCounts = trackIDs[mask], trackCounts[mask]

    mergedData = np.empty((len(trackIDs),) + data.shape[1:], dtype=np.float64)

    for idx in numba.prange(len(trackIDs)):
        trackGroup = data[data[:, track_idx] == trackIDs[idx], :]

        weights = trackGroup[:, int_idx]
        if mergedData[idx, int_idx] <= 0:
            weights = np.ones(weights.shape)
        mergedData[idx, :] = np.average(
            trackGroup[:, :], axis=0, weights=weights)

        mergedData[idx, frame_idx] = np.min(trackGroup[:, frame_idx])
        mergedData[idx, int_idx] = np.sum(weights)
        mergedData[idx, n_idx] = len(trackGroup)

        # print(
        #     'Merging ', (idx + 1), ' / ', len(trackIDs))

    leftIndices = np.where(data[:, trackIDs] == 0)[0]
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
def shift_correction(interpx, interpy, frames, datax, datay):
    for idx in numba.prange(0, len(interpx)):
        shift_x = interpx[idx]
        shift_y = interpy[idx]
        datax[frames == idx] -= shift_x
        datay[frames == idx] -= shift_y


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
