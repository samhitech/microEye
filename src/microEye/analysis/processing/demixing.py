import logging

import numba as nb
import numpy as np
from scipy.spatial import KDTree

from microEye.utils.gui_helper import _get_progress_bar

logger = logging.getLogger(__name__)


@nb.njit(
    nb.int64(
        nb.float32[:],  # current x
        nb.float32[:],  # current y
        nb.float32[:],  # current z
        nb.float32[:],  # current intensity
        nb.float64[:],  # distances
        nb.int64[:],  # neighbor indices
        nb.int32[:],  # previous labels
        nb.int32[:],  # labels
        nb.int32[:],  # molecule observation counts
        nb.float64[:],  # x sums
        nb.float64[:],  # y sums
        nb.float64[:],  # z sums
        nb.float64[:],  # intensity sums
        nb.float64[:],  # first intensities
        nb.float64[:],  # last intensities
        nb.int64,  # current offset
        nb.int64,  # molecule count
        nb.float32,  # radius
    ),
    cache=True,
)
def _assign_and_accumulate(
    current_x,
    current_y,
    current_z,
    current_intensity,
    distances,
    indices,
    previous_labels,
    labels,
    molecule_on,
    molecule_x,
    molecule_y,
    molecule_z,
    molecule_flux,
    molecule_flux_first,
    molecule_flux_last,
    current_offset,
    molecule_count,
    radius,
):
    n_previous = previous_labels.shape[0]

    for current_idx in range(distances.shape[0]):
        previous_idx = indices[current_idx]

        is_match = (
            distances[current_idx] < radius
            and previous_idx >= 0
            and previous_idx < n_previous
        )

        if is_match:
            molecule_id = previous_labels[previous_idx]

            molecule_flux_last[molecule_id] = current_intensity[current_idx]
        else:
            molecule_id = molecule_count
            molecule_count += 1

            molecule_on[molecule_id] = 0
            molecule_x[molecule_id] = 0.0
            molecule_y[molecule_id] = 0.0
            molecule_z[molecule_id] = 0.0
            molecule_flux[molecule_id] = 0.0
            molecule_flux_first[molecule_id] = current_intensity[current_idx]

            # Leave molecule_flux_last[molecule_id] equal to zero.

        global_idx = current_offset + current_idx
        labels[global_idx] = molecule_id

        molecule_on[molecule_id] += 1
        molecule_x[molecule_id] += current_x[current_idx]
        molecule_y[molecule_id] += current_y[current_idx]
        molecule_z[molecule_id] += current_z[current_idx]
        molecule_flux[molecule_id] += current_intensity[current_idx]

    return molecule_count


def brightness_demixing(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    intensity: np.ndarray,
    Radius: float,
    n_on: int,
):
    '''Track localizations between adjacent frames and calculate brightness data.

    Parameters
    ----------
    frames, x, y, z, intensity:
        One-dimensional localization arrays.
    Radius:
        Maximum XY distance for linking localizations.
    n_on:
        Minimum number of observations required to retain a molecule.

        It excludes the first and last observations when calculating the
        average transient intensity.

    Returns
    -------
    tuple
        labels, molecule on-counts, molecule X/Y/Z positions,
        average intensity, transient intensity, and rejection rate.
    '''

    frames = np.asarray(frames)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)

    n_points = frames.size

    if not (x.size == y.size == intensity.size == n_points):
        raise ValueError('frames, x, y, and intensity must have equal length.')

    if z is None:
        z = np.zeros(n_points, dtype=np.float32)
    else:
        z = np.asarray(z, dtype=np.float32)
        if z.size != n_points:
            raise ValueError('z must have the same length as frames.')

    if n_points == 0:
        empty_labels = np.empty((0, 1), dtype=np.int32)
        empty_values = np.empty((0, 1), dtype=np.float64)
        return (
            empty_labels,
            empty_labels.copy(),
            empty_values.copy(),
            empty_values.copy(),
            empty_values.copy(),
            empty_values.copy(),
            empty_values.copy(),
            0.0,
        )

    if Radius <= 0:
        raise ValueError('Radius must be greater than zero.')

    if n_on < 1:
        raise ValueError('n_on must be at least one.')

    # Sort once. All frame-local indexing uses these sorted arrays.
    sorted_idx = np.argsort(frames, kind='stable')

    frames_sorted = frames[sorted_idx]
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    z_sorted = z[sorted_idx]
    intensity_sorted = intensity[sorted_idx]

    # Reusable XY coordinate matrix; avoids column_stack per frame.
    xy_sorted = np.empty((n_points, 2), dtype=np.float32)
    xy_sorted[:, 0] = x_sorted
    xy_sorted[:, 1] = y_sorted

    # Frame boundaries.
    frame_breaks = np.flatnonzero(frames_sorted[1:] != frames_sorted[:-1]) + 1

    frame_starts = np.empty(frame_breaks.size + 1, dtype=np.int64)
    frame_starts[0] = 0
    frame_starts[1:] = frame_breaks

    frame_ends = np.empty_like(frame_starts)
    frame_ends[:-1] = frame_breaks
    frame_ends[-1] = n_points

    n_frames = frame_starts.size

    # Every localization can become a molecule, so n_points is the upper bound.
    labels_sorted = np.full(n_points, -1, dtype=np.int32)

    molecule_on = np.zeros(n_points, dtype=np.int32)
    molecule_x = np.zeros(n_points, dtype=np.float64)
    molecule_y = np.zeros(n_points, dtype=np.float64)
    molecule_z = np.zeros(n_points, dtype=np.float64)
    molecule_flux = np.zeros(n_points, dtype=np.float64)
    molecule_flux_first = np.zeros(n_points, dtype=np.float64)
    molecule_flux_last = np.zeros(n_points, dtype=np.float64)

    # Initialize molecules from the first frame.
    first_start = frame_starts[0]
    first_end = frame_ends[0]
    first_count = first_end - first_start

    labels_sorted[first_start:first_end] = np.arange(
        first_count,
        dtype=np.int32,
    )

    molecule_on[:first_count] = 1
    molecule_x[:first_count] = x_sorted[first_start:first_end]
    molecule_y[:first_count] = y_sorted[first_start:first_end]
    molecule_z[:first_count] = z_sorted[first_start:first_end]
    molecule_flux[:first_count] = intensity_sorted[first_start:first_end]
    molecule_flux_first[:first_count] = intensity_sorted[first_start:first_end]

    molecule_count = int(first_count)
    radius = np.float32(Radius)

    with _get_progress_bar() as progress:
        task = progress.add_task('[#d000d0]Demixing ...', total=n_frames)
        progress.update(task, advance=1)  # First frame is already initialized.

        # Process each adjacent frame pair.
        for frame_idx in range(1, n_frames):
            previous_start = frame_starts[frame_idx - 1]
            previous_end = frame_ends[frame_idx - 1]
            current_start = frame_starts[frame_idx]
            current_end = frame_ends[frame_idx]

            previous_xy = xy_sorted[previous_start:previous_end]
            current_xy = xy_sorted[current_start:current_end]

            tree = KDTree(previous_xy)
            distances, indices = tree.query(
                current_xy,
                k=1,
                distance_upper_bound=float(Radius),
            )

            previous_labels = labels_sorted[previous_start:previous_end]

            molecule_count = _assign_and_accumulate(
                x_sorted[current_start:current_end],
                y_sorted[current_start:current_end],
                z_sorted[current_start:current_end],
                intensity_sorted[current_start:current_end],
                np.asarray(distances, dtype=np.float64),
                np.asarray(indices, dtype=np.int64),
                previous_labels,
                labels_sorted,
                molecule_on,
                molecule_x,
                molecule_y,
                molecule_z,
                molecule_flux,
                molecule_flux_first,
                molecule_flux_last,
                int(current_start),
                int(molecule_count),
                radius,
            )
            progress.update(task, advance=1)

    # Convert labels back to the original input order without a second argsort.
    labels = np.empty(n_points, dtype=np.int32)
    labels[sorted_idx] = labels_sorted + 1
    labels = labels.reshape(-1, 1)

    # Keep only used molecule slots.
    molecule_on = molecule_on[:molecule_count]
    molecule_x = molecule_x[:molecule_count]
    molecule_y = molecule_y[:molecule_count]
    molecule_z = molecule_z[:molecule_count]
    molecule_flux = molecule_flux[:molecule_count]
    molecule_flux_first = molecule_flux_first[:molecule_count]
    molecule_flux_last = molecule_flux_last[:molecule_count]

    # Calculate averages.
    molecule_x /= molecule_on
    molecule_y /= molecule_on
    molecule_z /= molecule_on

    molecule_flux_small = np.zeros(molecule_count, dtype=np.float64)
    transient_mask = molecule_on > 2

    molecule_flux_small[transient_mask] = (
        molecule_flux[transient_mask]
        - molecule_flux_first[transient_mask]
        - molecule_flux_last[transient_mask]
    ) / (molecule_on[transient_mask] - 2)

    molecule_flux /= molecule_on

    # Filter molecules by minimum number of observations.
    keep_mask = molecule_on >= n_on

    mol_on = molecule_on[keep_mask].reshape(-1, 1)
    mol_x = molecule_x[keep_mask].reshape(-1, 1)
    mol_y = molecule_y[keep_mask].reshape(-1, 1)
    mol_z = molecule_z[keep_mask].reshape(-1, 1)
    mol_flux = molecule_flux[keep_mask].reshape(-1, 1)
    mol_flux_small = molecule_flux_small[keep_mask].reshape(-1, 1)

    accepted_observations = np.sum(mol_on[mol_on > 2])

    rejection_rate = (
        100.0 - (100.0 * accepted_observations / n_points) if n_points > 0 else 0.0
    )

    return (
        labels,
        mol_on,
        mol_x,
        mol_y,
        mol_z,
        mol_flux,
        mol_flux_small,
        rejection_rate,
    )


if __name__ == '__main__':
    import os

    from microEye.analysis.fitting.results import DataColumns, FittingResults

    filepath = os.environ.get('LOCAL_DATA_PATH')
    # filepath = os.environ.get('GATTA_DATA_PATH')

    table = FittingResults.fromFile(filepath, 1.0)

    res = brightness_demixing(
        table.data[DataColumns.FRAME],
        table.data[DataColumns.X],
        table.data[DataColumns.Y],
        table.data[DataColumns.Z],
        table.data[DataColumns.INTENSITY],
        Radius=10,
        n_on=3,
    )

    if res is not None:
        (
            labels,
            mol_on,
            mol_x,
            mol_y,
            mol_z,
            mol_flux,
            mol_flux_small,
            rejection_rate,
        ) = res

        logger.info(f'Labels shape: {labels.shape}')
        logger.info(f'Molecule on-counts shape: {mol_on.shape}')
        logger.info(f'Molecule X positions shape: {mol_x.shape}')
        logger.info(f'Molecule Y positions shape: {mol_y.shape}')
        logger.info(f'Molecule Z positions shape: {mol_z.shape}')
        logger.info(f'Average intensity shape: {mol_flux.shape}')
        logger.info(f'Transient intensity shape: {mol_flux_small.shape}')
        logger.info(f'Transient intensity > 0: {np.sum(mol_flux_small > 0)}')
        logger.info(f'Rejection rate: {rejection_rate:.2f}%')

        import sys

        import pyqtgraph as pg

        from microEye.qt import QApplication

        app = QApplication([sys.argv[0]])

        # plot I hist, average flux hist, and transient flux hist, and on counts hist

        graph = pg.GraphicsLayoutWidget()

        # plot I hist
        p1 = graph.ci.addPlot(title='Intensity Histogram')
        hist, bins = np.histogram(
            table.data[DataColumns.INTENSITY], bins=200, density=True
        )

        p1.plot(
            bins,
            hist,
            stepMode=True,
            fillLevel=0,
            brush=(0, 0, 255, 120),
            name='Intensity Histogram',
        )
        p1.setLabel('left', 'Count')
        p1.setLabel('bottom', 'Intensity')

        graph.ci.nextColumn()
        p2 = graph.ci.addPlot(title='Average Intensity Histogram')
        p2.setLabel('left', 'Count')
        p2.setLabel('bottom', 'Average Intensity')

        hist, bins = np.histogram(mol_flux, bins=200, density=True)
        p2.plot(
            bins,
            hist,
            stepMode=True,
            fillLevel=0,
            brush=(255, 0, 0, 120),
            name='Average Intensity Histogram',
        )

        graph.ci.nextColumn()
        p3 = graph.ci.addPlot(title='Transient Intensity Histogram')
        p3.setLabel('left', 'Count')
        p3.setLabel('bottom', 'Transient Intensity')

        hist, bins = np.histogram(
            mol_flux_small[mol_flux_small > 0], bins=200, density=True
        )
        p3.plot(
            bins,
            hist,
            stepMode=True,
            fillLevel=0,
            brush=(255, 0, 255, 120),
            name='Transient Intensity Histogram',
        )

        # plot on counts hist
        graph.ci.nextRow()
        p4 = graph.ci.addPlot(title='On Counts Histogram', colspan=3)
        hist, bins = np.histogram(mol_on, bins=200, density=True)
        p4.plot(bins, hist, stepMode=True, fillLevel=0, brush=(0, 0, 255, 120))
        p4.setLabel('left', 'Count')
        p4.setLabel('bottom', 'On Counts')


        graph.show()

        sys.exit(app.exec())
