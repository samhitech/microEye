
import cv2
import numba
import numpy as np
import pyqtgraph as pg
from skimage.registration import phase_cross_correlation
from sklearn.neighbors import NearestNeighbors


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
            nNeighbors = NearestNeighbors(
                n_neighbors=min(neighbors, nextFrame.shape[0]))
            nNeighbors.fit(nextFrame[:, 3:5])
            foundnn = nNeighbors.kneighbors(currentFrame[:, 3:5])
            foundnn = np.asarray(foundnn)

            currentIDs = np.where(foundnn[0] < maxDistance)[0]
            neighbourIDs = foundnn[
                :, foundnn[0] < maxDistance][1].astype(np.int64)
            distance = foundnn[0]

        for idx in numba.prange(len(currentIDs)):
            if nextFrame[neighbourIDs[idx], 7] == 0:
                if currentFrame[currentIDs[idx], 7] == 0:
                    currentFrame[currentIDs[idx], 7] = counter + idx + 1
                    nextFrame[neighbourIDs[idx], 7] = counter + idx + 1
                else:
                    nextFrame[neighbourIDs[idx], 7] = \
                        currentFrame[currentIDs[idx], 7]

            currentFrame[currentIDs[idx], 8] = distance[currentIDs[idx]]

        data[data[:, 0] == frameID, :] = currentFrame
        data[data[:, 0] == nextID, :] = nextFrame

    return counter + len(currentIDs)


@numba.njit(parallel=True)
def merge_localizations(data: np.ndarray, maxLength):

    with numba.objmode(
            trackIDs='float64[:]'):
        trackIDs, inverse, trackCounts = np.unique(
            data[:, 7], return_counts=True, return_inverse=True)
        mask = np.logical_and(trackIDs > 0, trackCounts <= maxLength)
        trackIDs[np.logical_not(mask)] = 0
        data[:, 7] = trackIDs[inverse]
        trackIDs, trackCounts = trackIDs[mask], trackCounts[mask]

    mergedData = np.empty((len(trackIDs),) + data.shape[1:], dtype=np.float64)

    for idx in numba.prange(len(trackIDs)):
        trackGroup = data[data[:, 7] == trackIDs[idx], :]

        mergedData[idx, 0] = np.min(trackGroup[:, 0])
        weights = trackGroup[:, 6]
        mergedData[idx, 6] = np.sum(weights)
        if mergedData[idx, 6] <= 0:
            weights = np.ones(weights.shape)
        mergedData[idx, 3] = np.sum(
            trackGroup[:, 3] * weights) / np.sum(weights)
        mergedData[idx, 4] = np.sum(
            trackGroup[:, 4] * weights) / np.sum(weights)
        mergedData[idx, 9] = len(trackGroup)

        # print(
        #     'Merging ', (idx + 1), ' / ', len(trackIDs))

    leftIndices = np.where(data[:, 7] == 0)[0]
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
