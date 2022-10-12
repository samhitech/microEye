
import weakref
import cv2
import numpy as np
import numba as nb
import pyqtgraph as pg
from skimage.registration import phase_cross_correlation
from sklearn.neighbors import NearestNeighbors


@nb.njit(nb.types.Tuple((nb.int64, nb.int64[:], nb.int64[:], nb.float64[:]))(
    nb.float64[:, :], nb.float64[:, :], nb.int64[:], nb.int64[:],
    nb.float64[:], nb.int64, nb.float64,
    nb.float64, nb.int64), parallel=True)
def nn_trajectories(
        currentFrame: np.ndarray, nextFrame: np.ndarray,
        c_trackID: np.ndarray, n_trackID: np.ndarray,
        nn_dist: np.ndarray, counter: int = 0, minDistance=0,
        maxDistance=30, neighbors=1):

    currentIDs = None

    if len(currentFrame) > 0 and len(nextFrame) > 0:
        with nb.objmode(
                foundnn='float64[:,:]', dist_mask='boolean[:]'):
            nNeighbors = NearestNeighbors(
                n_neighbors=max(1, min(neighbors, nextFrame.shape[0])))
            nNeighbors.fit(nextFrame[:, 1:])
            foundnn = nNeighbors.kneighbors(currentFrame[:, 1:])
            foundnn = np.asarray(foundnn, dtype=np.float64)[..., 0]

            dist_mask = np.logical_and(
                foundnn[0] >= minDistance,
                foundnn[0] <= maxDistance
            )

        currentIDs = np.where(dist_mask)[0].astype(np.int64)

        neighbourIDs = foundnn[1][dist_mask].astype(np.int64)
        distance = foundnn[0]

        if np.isnan(distance).any():
            print("Nan!\n\n")

        for idx in range(len(currentIDs)):
            if n_trackID[neighbourIDs[idx]] == 0:
                if c_trackID[currentIDs[idx]] == 0:
                    counter += 1
                    c_trackID[currentIDs[idx]] = counter
                    n_trackID[neighbourIDs[idx]] = counter
                else:
                    n_trackID[neighbourIDs[idx]] = \
                        c_trackID[currentIDs[idx]]

                nn_dist[neighbourIDs[idx]] = distance[currentIDs[idx]]

    # counter += 0 if currentIDs is None else len(currentIDs)

    return counter, c_trackID, n_trackID, nn_dist


@nb.njit(nb.float64[:, :](nb.float64[:, :], nb.int32[:], nb.int64),
         parallel=True)
def merge_localizations(data: np.ndarray, columns: np.ndarray, maxLength):

    frame_idx = columns[0]
    track_idx = columns[1]
    n_idx = columns[2]
    int_idx = columns[3]

    with nb.objmode(trackIDs='float64[:]'):
        trackIDs, inverse, trackCounts = np.unique(
            data[:, track_idx], return_counts=True, return_inverse=True)
        mask = np.logical_and(trackIDs > 0, trackCounts <= maxLength)
        trackIDs[np.logical_not(mask)] = 0
        data[:, track_idx] = trackIDs[inverse]
        trackIDs, trackCounts = trackIDs[mask], trackCounts[mask]

    mergedData = np.empty((len(trackIDs),) + data.shape[1:], dtype=np.float64)

    for idx in nb.prange(len(trackIDs)):
        trackGroup = data[data[:, track_idx] == trackIDs[idx], :]

        weights = trackGroup[:, int_idx]
        for idw in range(len(weights)):
            if weights[idw] < 1:
                weights[idw] = 1
        if np.sum(weights) <= 0:
            weights = np.ones(len(weights), dtype=np.float64)
        for col in range(trackGroup.shape[1]):
            mergedData[idx, col] = np.sum(
                trackGroup[:, col] * weights) / np.sum(weights)

        mergedData[idx, frame_idx] = np.min(trackGroup[:, frame_idx])
        mergedData[idx, int_idx] = np.sum(weights)
        mergedData[idx, n_idx] = len(trackGroup)
        mergedData[idx, track_idx] = trackIDs[idx]

        # print(
        #     'Merging ', (idx + 1), ' / ', len(trackIDs))

    if np.isnan(mergedData).any():
        print("Nan!\n\n")

    # with nb.objmode(leftData='float64[:,:]'):
    leftIndices = np.where(data[:, track_idx] == 0)[0]
    leftData = data[leftIndices, :]

    return np.append(mergedData, leftData, axis=0)


@nb.njit(parallel=True)
def shift_estimation(sub_images, pixelSize, upsampling):
    shifts = np.zeros((len(sub_images), 2))
    for idx in nb.prange(0, len(sub_images)):
        with nb.objmode(shift='float64[:]'):
            shift = phase_cross_correlation(
                sub_images[idx], sub_images[0], upsample_factor=upsampling)[0]
        shifts[idx, :] = shift * pixelSize
    return shifts


@nb.jit(nopython=True, parallel=True)
def shift_correction(interpx, interpy, frames, datax, datay):
    for idx in nb.prange(0, len(interpx)):
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
