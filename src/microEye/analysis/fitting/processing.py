import logging
import math
import time

import numba as nb
import numpy as np
import pyqtgraph as pg
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

@nb.njit(
    nb.types.Tuple((nb.int64, nb.int64[:], nb.float64[:]))(
        nb.float64[:, :],  # data: frame, x, y
        nb.int64[:],  # ids (frame boundary indices)
        nb.int64[:],  # trackID array to update
        nb.float64[:],  # neighbour_dist to update
        nb.int64,  # min_frame
        nb.int64,  # max_frame
        nb.int64,  # maxOff
        nb.float64,  # minDistance
        nb.float64,  # maxDistance
    ),
    cache=True,
)
def run_global_tracking(
    data: np.ndarray,
    ids: np.ndarray,
    trackID: np.ndarray,
    neighbour_dist: np.ndarray,
    min_frame: int,
    max_frame: int,
    maxOff: int,
    minDistance: float,
    maxDistance: float,
):
    counter = 0

    for frameID in range(min_frame, max_frame):
        c_start = ids[frameID - 1] if frameID > 0 else 0
        c_end = ids[frameID]

        n_current = c_end - c_start
        if n_current == 0:
            continue

        coords_current = data[c_start:c_end, 1:]

        for offset in range(0, maxOff + 1):
            nextID = frameID + offset + 1
            if nextID > max_frame:
                continue

            n_start = ids[nextID - 1]
            n_end = ids[nextID]

            n_next = n_end - n_start
            if n_next == 0:
                continue

            coords_next = data[n_start:n_end, 1:]

            with nb.objmode(distances='float64[:]', indices='int64[:]'):
                tree = KDTree(coords_next)
                distances, indices = tree.query(
                    coords_current, k=1, distance_upper_bound=maxDistance
                )
                distances = np.asarray(distances, dtype=np.float64)
                indices = np.asarray(indices, dtype=np.int64)

            for idx in range(n_current):
                dist = distances[idx]
                neighbour_idx = indices[idx]

                # no neighbour within maxDistance: KDTree returns index n_next
                if neighbour_idx < n_next and dist >= minDistance:
                    glob_curr = c_start + idx
                    glob_next = n_start + neighbour_idx

                    if trackID[glob_next] == 0:
                        if trackID[glob_curr] == 0:
                            counter += 1
                            trackID[glob_curr] = counter
                            trackID[glob_next] = counter
                        else:
                            trackID[glob_next] = trackID[glob_curr]

                        neighbour_dist[glob_next] = dist

    return counter, trackID, neighbour_dist


# @nb.njit(parallel=True)
def tardis_data(
    frames: np.ndarray,
    locX: np.ndarray,
    locY: np.ndarray,
    locZ: np.ndarray = None,
    dts=None,
    range=1500,
    bins=1500,
):
    if dts is None:
        dts = [1]
    result = np.zeros((len(dts), bins), np.float64)
    edges = np.linspace(0, range, bins + 1)
    if locZ is not None:
        data: np.ndarray = np.column_stack((locX, locY, locZ))
    else:
        data: np.ndarray = np.column_stack((locX, locY))
    data = data[np.argsort(frames)]

    ids = np.cumsum(np.bincount(frames.astype(np.int64)))

    min_frame = int(np.min(frames))
    max_frame = int(np.max(frames))

    for idx, dt in enumerate(dts):
        timer = []
        for t in nb.prange(min_frame, max_frame - dt + 1):
            # next_frame = current_frame + delta_t
            nt = t + dt

            if nt > max_frame:
                continue

            # with nb.objmode(dist='float64[:]'):
            start = time.perf_counter_ns()
            dist = calc_dist(
                data[slice(ids[t - 1] if t > 0 else 0, ids[t])],
                data[slice(ids[nt - 1], ids[nt])],
            )
            stop = time.perf_counter_ns()
            timer.append((stop - start) / 1e6)

            result[idx] += histogram(dist, 0, range, bins)

            print(
                f'{idx + 1:d}/',
                f'{len(dts):d} TARDIS ',
                f'{(t + 1) / (max_frame - dt + 1):.2%}...               ',
                end='\r',
            )

        result[idx] /= np.sum(result[idx])
        print(
            f'{idx + 1:d}/{len(dts):d} TARDIS {1.0:.2%}... ',
            f'{np.mean(timer):.3f}ms | ',
            f'{np.sum(timer):.3f}ms     ',
            end='\r',
        )
    return result, edges


@nb.vectorize([nb.int64(nb.float64, nb.float64, nb.float64, nb.float64)])
def f(x, min_val, max_val, num_bins):
    c = num_bins / (max_val - min_val)
    return (x - min_val) * c


def histogram(data, min_val, max_val, num_bins):
    c = f(data[(data >= min_val) & (data < max_val)], min_val, max_val, num_bins)
    counts = np.bincount(c, minlength=num_bins)
    return counts


@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def diff_sq(x, y):
    return (x - y) ** 2


@nb.njit(parallel=True, cache=True)
def calc_dist(origin: np.ndarray, dest: np.ndarray):
    res = np.zeros((origin.shape[0] * dest.shape[0]), np.float64)
    for i in nb.prange(origin.shape[0]):
        for j in nb.prange(dest.shape[0]):
            res[i * dest.shape[0] + j] = math.sqrt(
                np.sum((dest[j, :] - origin[i, :]) ** 2)
            )
    return res


@nb.njit(
    nb.float64[:, :](nb.float64[:, :], nb.int32[:], nb.int64), parallel=True, cache=True
)
def merge_localizations(data: np.ndarray, columns: np.ndarray, maxLength):
    frame_idx = columns[0]
    track_idx = columns[1]
    n_idx = columns[2]
    int_idx = columns[3]

    with nb.objmode(trackIDs='float64[:]'):
        trackIDs, inverse, trackCounts = np.unique(
            data[:, track_idx], return_counts=True, return_inverse=True
        )
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
            mergedData[idx, col] = np.sum(trackGroup[:, col] * weights) / np.sum(
                weights
            )

        mergedData[idx, frame_idx] = np.min(trackGroup[:, frame_idx])
        mergedData[idx, int_idx] = np.sum(weights)
        mergedData[idx, n_idx] = len(trackGroup)
        mergedData[idx, track_idx] = trackIDs[idx]

        # print(
        #     'Merging ', (idx + 1), ' / ', len(trackIDs))

    if np.isnan(mergedData).any():
        print('Nan!\n\n')

    # with nb.objmode(leftData='float64[:,:]'):
    leftIndices = np.where(data[:, track_idx] == 0)[0]
    leftData = data[leftIndices, :]

    return np.append(mergedData, leftData, axis=0)



def plot_animation_stats(
    frame_bin_centers, locs_per_bin, title='Localizations per Frames Bin'
):
    logger.info('Plotting Stats ...')

    # Calculate bin edges from centers
    bin_width = frame_bin_centers[1] - frame_bin_centers[0]
    bin_edges = np.concatenate(
        [frame_bin_centers - bin_width / 2, [frame_bin_centers[-1] + bin_width / 2]]
    )

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'Localizations', units='counts')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Frame', units='')

    plt.setWindowTitle(title)

    # plot bar plot using bin edges instead of centers
    plt.plot(
        bin_edges,
        locs_per_bin,
        stepMode='center',
        fillLevel=0,
        fillOutline=True,
        brush=(0, 0, 255, 150),
    )

    logger.info('Done ...')
