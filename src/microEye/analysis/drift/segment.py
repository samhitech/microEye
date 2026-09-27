import logging

import numpy as np

logger = logging.getLogger(__name__)

MIN_WINDOW_SIZE = 5
MIN_NUM_WINDOWS = 3


def segment_dataset(
    # frames: np.ndarray,
    milestones: np.ndarray,
    frame_boundaries: np.ndarray,
    max_locs: int | None = None,
    return_loc_counts: bool = False,
    return_frame_counts: bool = False,
    return_seg_centers: bool = False,
    sort_indices: np.ndarray = None,
) -> tuple:
    '''
    Segments the dataset into contiguous slices based on the provided milestones.

    Parameters
    ----------
    milestones : np.ndarray
        Array of shape (num_windows + 1,) containing the indices of the milestones.
    frame_boundaries : np.ndarray
        Array of shape (num_windows + 1,) containing the boundaries of the frames.
    max_locs : int | None, optional
        The maximum number of localizations to keep per window.
    return_seg_centers : bool, optional
        Whether to return the center frame of each segment.
    return_loc_counts : bool, optional
        Whether to return the number of localizations in each window.
    return_frame_counts : bool, optional
        Whether to return the number of frames in each window.
    sort_indices : np.ndarray, optional
        Array of indices to sort the original frames array.

    Returns
    -------
    tuple
        A tuple containing:
        - `slices`: Pre-computed contiguous slices.
        - `seg_centers`: Center frame of each segment.
        - `loc_counts`: Number of localizations in each window.
        - `frame_counts`: Number of frames in each window.
        - `sort_indices`: Indices to sort the original frames array.
    '''
    num_windows = len(milestones) - 1

    # frame counts for each window if requested
    frame_counts = np.diff(frame_boundaries).tolist() if return_frame_counts else None

    slices: list[slice | np.ndarray] = []
    loc_counts: list[int] = [] if return_loc_counts else None
    seg_centers: list[int] = [] if return_seg_centers else None

    # single-pass segmentation filtering
    for i in range(num_windows):
        start, stop = int(milestones[i]), int(milestones[i + 1])
        window_size = stop - start

        # If downsampling is triggered and current window density exceeds the cap
        if max_locs is not None and window_size > max_locs:
            # Generate random subset indices relative to the sorted array positioning
            chosen_indices = np.random.choice(
                np.arange(start, stop), max_locs, replace=False
            )
            slices.append(chosen_indices)
            if return_loc_counts:
                loc_counts.append(max_locs)
        else:
            # Keep the perfect contiguous memory block view
            slices.append(slice(start, stop))
            if return_loc_counts:
                loc_counts.append(window_size)

        if return_seg_centers:
            # Calculate the center frame of the current segment
            center_frame = (frame_boundaries[i] + frame_boundaries[i + 1]) // 2
            seg_centers.append(center_frame)

    return tuple(
        r
        for r in [slices, seg_centers, loc_counts, frame_counts, sort_indices]
        if r is not None
    )


def frame_segmentation(
    frames: np.ndarray,
    parameter: int,
    is_window_size: bool = True,
    return_seg_centers=False,
    return_loc_counts=False,
    return_frame_counts=False,
    sort_frames=False,
    equal_locs_per_window=False,
):
    '''
    Segments the dataset into contiguous slices based on a specified window size.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (N,) containing integer frame indices.
    parameter : int
        If `is_window_size` is True, this is the window size in frames.
        If `is_window_size` is False, this is the number of windows to create.
    is_window_size : bool, optional
        If True, `parameter` is treated as the window size.
    return_seg_centers : bool, optional
        Whether to return the center frame of each segment.
    return_loc_counts : bool, optional
        Whether to return the number of localizations in each window.
    return_frame_counts : bool, optional
        Whether to return the number of frames in each window.
    sort_frames : bool, optional
        Whether to sort the frames before segmentation.
        If True, the function will return
        the indices to sort the original frames array.

    Returns
    -------
    tuple
        A tuple containing:
        - `slices`: Pre-computed contiguous slices.
        - `seg_centers`: Center frame of each segment.
        - `loc_counts`: Number of localizations in each window.
        - `frame_counts`: Number of frames in each window.
        - `sort_indices`: Indices to sort the original frames array.
    '''
    if not is_window_size and parameter < MIN_NUM_WINDOWS:
        raise ValueError(f'Number of windows must be at least {MIN_NUM_WINDOWS}.')
    if is_window_size and parameter < MIN_WINDOW_SIZE:
        raise ValueError(f'Window size must be at least {MIN_WINDOW_SIZE} frames.')

    sort_indices = np.argsort(frames) if sort_frames else None
    sorted_frames = frames[sort_indices] if sort_frames else frames

    min_frame = int(sorted_frames[0])
    max_frame = int(sorted_frames[-1])
    total_frames = max_frame - min_frame + 1

    num_windows = (
        max(MIN_NUM_WINDOWS, round(total_frames / parameter))
        if is_window_size
        else parameter
    )

    # Generate balanced frame milestones
    frame_boundaries = np.linspace(
        min_frame, max_frame + 1, num_windows + 1, dtype=np.int64
    )
    milestones = np.searchsorted(sorted_frames, frame_boundaries)
    milestones[-1] = len(sorted_frames)

    max_locs = None
    if equal_locs_per_window:
        # find minimum number of localizations in any window
        max_locs = np.diff(milestones).min()

    return segment_dataset(
        milestones,
        frame_boundaries,
        max_locs,
        return_seg_centers=return_seg_centers,
        return_loc_counts=return_loc_counts,
        return_frame_counts=return_frame_counts,
        sort_indices=sort_indices,
    )


def adaptive_segmentation(
    frames: np.ndarray,
    locs_per_window=5000,
    return_seg_centers=False,
    return_loc_counts=False,
    return_frame_counts=False,
    sort_frames=False,
    downsample=1.0,
) -> tuple:
    '''
    Perform adaptive segmentation of the dataset based on the number
    of localizations per window.

    If `downsample` is less than `1.0`,
    the dataset will be downsampled before segmentation.

    Important: `frames` must be sorted in ascending order before calling this function.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (N,) containing integer frame indices.
    locs_per_window : int, optional
        Number of localizations to include in each window.
    return_seg_centers : bool, optional
        Whether to return the center frame of each segment.
    return_loc_counts : bool, optional
        Whether to return the number of localizations in each window.
    return_frame_counts : bool, optional
        Whether to return the number of frames in each window.
    sort_frames : bool, optional
        Whether to sort the frames before segmentation.
        If `True`, the function will return
        the indices to sort the original frames array.
    downsample : float, optional
        Factor by which to downsample the dataset for segmentation.
        The dataset will be randomly downsampled:
        - If 0.0 < `downsample` < 1.0, then `max_locs = downsample * locs_per_window`.
        - If 1.0 < `downsample` < `locs_per_window`, then `max_locs = downsample`.

    Returns
    -------
    tuple
        A tuple containing:
        - `slices`: Pre-computed contiguous slices.
        - `seg_centers`: Center frame of each segment.
        - `loc_counts`: Number of localizations in each window.
        - `frame_counts`: Number of frames in each window.
        - `sort_indices`: Indices to sort the original frames array.
    '''
    sort_indices = np.argsort(frames) if sort_frames else None
    sorted_frames = frames[sort_indices] if sort_frames else frames

    counts_per_frame = np.bincount(sorted_frames)
    cumulative_frames = np.cumsum(counts_per_frame)
    total_locs = cumulative_frames[-1]

    num_windows = max(1, round(total_locs / locs_per_window))

    # Adaptive targets scale based on localization counts rather than timelines
    milestones = np.linspace(0, total_locs, num_windows + 1, dtype=np.int64)
    frame_boundaries = np.searchsorted(cumulative_frames, milestones)
    frame_boundaries[-1] = len(counts_per_frame)

    max_locs = None
    if 0.0 < downsample < 1.0:
        # Use the explicit structural baseline
        max_locs = int(locs_per_window * downsample)
    elif 1.0 < downsample < locs_per_window:
        max_locs = int(downsample)

    return segment_dataset(
        milestones,
        frame_boundaries,
        max_locs=max_locs,
        return_seg_centers=return_seg_centers,
        return_loc_counts=return_loc_counts,
        return_frame_counts=return_frame_counts,
        sort_indices=sort_indices,
    )


if __name__ == '__main__':
    import os

    import pandas as pd

    data = pd.read_hdf(
        os.environ.get('LOCAL_DATA_PATH'),
    )

    frames = data['frame'].to_numpy(dtype=np.int64)
    slices, loc_counts, frame_counts = frame_segmentation(
        frames, 100, True, True, True, equal_locs_per_window=True
    )

    print(f'Number of slices: {len(slices)}')

    # plot counts histogram for locs and frames
    import matplotlib.pyplot as plt

    ax = plt.subplot(2, 1, 1)

    ax.hist(loc_counts, edgecolor='black', alpha=0.7, label='Localizations')
    # no scientific notation for x-axis
    ax.ticklabel_format(style='', axis='both')
    ax.set_xlabel('Number of Localizations')
    ax.set_ylabel('Counts')

    ax = plt.subplot(2, 1, 2)
    ax.hist(frame_counts, bins=50, edgecolor='black', alpha=0.7, label='Frames')
    ax.set_xlabel('Number of Frames')
    ax.set_ylabel('Counts')
    ax.legend()
    plt.tight_layout()
    plt.show()
