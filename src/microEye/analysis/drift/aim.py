# AIM: Adaptive Intersection Maximization
#
# References
# [1] H. Ma, M. Chen, P. Nguyen, and Y. Liu,
# “Toward drift-free high-throughput nanoscopy through adaptive
# intersection maximization,” Sci. Adv. 10(21), eadm7765 (2024).
#
# https://www.science.org/doi/10.1126/sciadv.adm7765
# https://github.com/YangLiuLab/AIM
#
import numba as nb
import numpy as np

from microEye.analysis.drift.core import _compute_frame_drift, _shift_correction


@nb.njit('int64(int64[:], int64[:], int64[:], int64[:])', cache=True)
def _intersections_two_pointer(a, b, a_counts, b_counts) -> int:
    '''
    Fastest for Big vs Big (Linear Time: O(M + N))

    Parameters
    ----------
    a : np.ndarray (int64)
        unique values of the reference localizations.
    b : np.ndarray (int64)
        unique values of the target localizations.
    a_counts : np.ndarray (int64)
        counts of the unique values of reference localizations.
    b_counts : np.ndarray (int64)
        counts of the unique values of target localizations.

    Returns
    -------
    int
        number of intersecting points.
    '''
    intersection_count = 0
    i, j = 0, 0
    len_a, len_b = len(a), len(b)

    while i < len_a and j < len_b:
        if a[i] == b[j]:
            intersection_count += min(a_counts[i], b_counts[j])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return intersection_count


@nb.njit('int64(int64[:], int64[:], int64[:], int64[:])', cache=True)
def _intersections_binary_search(a, b, a_counts, b_counts) -> int:
    '''
    Fastest when `a` is highly filtered/localized and `b` is massive (O(M log N))

    Parameters
    ----------
    a : np.ndarray (int64)
        unique values of the reference localizations.
    b : np.ndarray (int64)
        unique values of the target localizations.
    a_counts : np.ndarray (int64)
        counts of the unique values of reference localizations.
    b_counts : np.ndarray (int64)
        counts of the unique values of target localizations.

    Returns
    -------
    int
        number of intersecting points.
    '''
    intersection_count = 0
    len_b = len(b)

    for i in range(len(a)):
        target = a[i]
        left, right = 0, len_b - 1
        while left <= right:
            mid = (left + right) // 2
            if b[mid] == target:
                intersection_count += min(a_counts[i], b_counts[mid])
                break
            elif b[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
    return intersection_count


@nb.njit('int64(int64[:], int64[:], int64[:], int64[:])', cache=True)
def _intersections(a, b, a_counts, b_counts) -> int:
    '''
    Adaptive router ensuring optimal performance for any combination of array sizes.

    Parameters
    ----------
    a : np.ndarray (int64)
        unique values of the reference localizations.
    b : np.ndarray (int64)
        unique values of the target localizations.
    a_counts : np.ndarray (int64)
        counts of the unique values of reference localizations.
    b_counts : np.ndarray (int64)
        counts of the unique values of target localizations.

    Returns
    -------
    int
        number of intersecting points.
    '''
    M = len(a)
    N = len(b)

    if M == 0 or N == 0:
        return 0

    # Rule of thumb threshold: If one array is roughly 15x smaller than the other,
    # Binary Search beats Two-Pointer by avoiding stepping through the giant array.
    if N > 15 * M:
        return _intersections_binary_search(a, b, a_counts, b_counts)
    elif M > 15 * N:
        # Re-use binary search by swapping arguments so the smaller array
        # queries the larger one
        return _intersections_binary_search(b, a, b_counts, a_counts)
    else:
        # Default to Two-Pointer for similar-sized arrays (linear time)
        return _intersections_two_pointer(a, b, a_counts, b_counts)


# @nb.njit(parallel=True)
def _calc_intersections(
    ref_coords: np.ndarray,
    ref_counts: np.ndarray,
    target: np.ndarray,
    target_counts: np.ndarray,
    shifts: np.ndarray,
    roi_size: int,
    is_2d: bool = True,
):
    target_coords_shifted = target[:, np.newaxis] + shifts

    intersection_counts: np.ndarray = np.zeros(shifts.shape[0], dtype=np.int64)

    for shift_idx in nb.prange(shifts.shape[0]):
        intersection_counts[shift_idx] = _intersections(
            ref_coords, target_coords_shifted[:, shift_idx], ref_counts, target_counts
        )

    if not is_2d:
        return intersection_counts

    return intersection_counts.reshape(roi_size, roi_size)


def _point_intersect_2d(
    ref_coords: np.ndarray,
    ref_counts: np.ndarray,
    tar_coords: np.ndarray,
    tar_counts: np.ndarray,
    shifts: np.ndarray,
    roi_size: int,
) -> np.ndarray:
    '''
    Performs 2D point intersection analysis.

    Parameters
    ----------
    ref_coords : np.ndarray
        Unique values of the reference localizations.
    ref_counts : np.ndarray
        Counts of the unique values of reference localizations.
    tar_coords : np.ndarray
        Unique values of the target localizations.
    tar_counts : np.ndarray
        Counts of the unique values of target localizations.
    intersect_d : float
        Intersect distance in camera pixels.
    width : float
        the width of the image in intersect_d units.
    shifts : np.ndarray
        1D array with x and y shifts.
    roi_size : int
        Size of the local search region.

    Returns
    -------
    np.ndarray
        2D array of intersection counts for each shift in the local search region.
    '''
    return _calc_intersections(
        ref_coords.astype(np.int64),
        ref_counts.astype(np.int64),
        tar_coords.astype(np.int64),
        tar_counts.astype(np.int64),
        shifts.astype(np.int64),
        int(roi_size),
        is_2d=True,
    )


def _point_intersect_1d(
    ref_coords: np.ndarray,
    ref_counts: np.ndarray,
    tar_coords: np.ndarray,
    tar_counts: np.ndarray,
    shifts: np.ndarray,
    roi_size: int,
) -> np.ndarray:
    '''
    Performs 1D point intersection analysis.
    '''
    return _calc_intersections(
        ref_coords.astype(np.int64),
        ref_counts.astype(np.int64),
        tar_coords.astype(np.int64),
        tar_counts.astype(np.int64),
        shifts.astype(np.int64),
        int(roi_size),
        is_2d=False,
    )


def _get_fft_peak(roi: np.ndarray, roi_size: int) -> tuple[float, float]:
    '''
    Estimate sub-pixel peak position using 2D FFT phase.
    roi shape is (roi_size, roi_size) where row=Y, col=X.
    '''
    fft_spectrum = np.fft.fft2(roi)  # Do NOT transpose!
    grid_dimensions = np.array(roi.shape)

    # Extract primary frequencies:
    # Axis 1 (cols) = X frequency component [0, 1]
    # Axis 0 (rows) = Y frequency component [1, 0]
    phase_angles = np.array(
        [
            np.angle(fft_spectrum[0, 1]),  # X-frequency
            np.angle(fft_spectrum[1, 0]),  # Y-frequency
        ]
    )
    spatial_resolutions = 2 * np.pi / grid_dimensions

    peak_pos = np.zeros(2, dtype=np.float64)

    for d in range(2):
        if phase_angles[d] > 0:
            phase_angles[d] -= 2 * np.pi

        peak_pos[d] = (np.abs(phase_angles[d]) / spatial_resolutions[d]) - (
            grid_dimensions[d] - 1
        ) / 2

    return float(peak_pos[0]), float(peak_pos[1])


def _get_fft_peak_z(roi: np.ndarray, roi_size: int) -> float:
    '''
    Estimate the precise sub-pixel position of the peak of
    `roi` using the phase of the 1D FFT along the z-axis.

    Parameters
    ----------
    roi : np.ndarray
        1D array with numbers of intersections in the local search region.
    roi_size : int
        Size of the local search region.

    Returns
    -------
    peak_pos : float
        Estimated z-coordinate of the peak.
    '''
    fft_spectrum = np.fft.fft(roi)
    phase_angle = np.angle(fft_spectrum[1])
    phase_angle = phase_angle - 2 * np.pi * (phase_angle > 0)  # normalize
    peak_pos = np.abs(phase_angle) / (2 * np.pi / roi.size) - (roi.size - 1) / 2
    peak_pos *= roi_size / roi.size  # scale to intersect_d units
    return peak_pos


def intersection_max(
    x: np.ndarray,
    y: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    frame_ids: np.ndarray,
    slices: list[slice],
    seg_centers: np.ndarray,
    image_width: int,
    Intersect_d: float = 20.0,
    roi_r: int = 3,
):
    '''
    Parameters
    ----------
    x : np.ndarray
        Array of x-coordinates of localizations.
    y : np.ndarray
        Array of y-coordinates of localizations.
    ref_x : np.ndarray
        Array of x-coordinates of reference localizations.
    ref_y : np.ndarray
        Array of y-coordinates of reference localizations.
    frame_ids : np.ndarray
        Array for fast slice indexing of frames.
    slices : list[slice]
        List of slices defining temporal segments of the dataset.
    seg_centers : np.ndarray
        Array of center frame numbers for each segment.
    image_width : int
        Width of the image in BIN units,
        i.e. already `ceil(width_physical / Intersect_d)`.

        Do not divide this by Intersect_d again anywhere downstream - it is the
        row-stride used consistently for both `ref` and the target/shift arrays.
    Intersect_d : float, optional
        Intersection distance, by default 20.0
    roi_r : int, optional
        Radius of the region of interest for local search, by default 3
    second_stage : bool, optional
        Whether to use the second stage of the intersection algorithm, by default False
    '''
    num_segs = len(slices)

    # transfer 2D local search region to 1D
    roi_size = 2 * roi_r + 1

    shifts = np.zeros((roi_size, roi_size), dtype=np.int64)
    for i, r in enumerate(range(-roi_r, roi_r + 1)):
        for j, c in enumerate(range(-roi_r, roi_r + 1)):
            shifts[i, j] = r * image_width + c
    shifts = shifts.reshape(roi_size**2)

    # drift in x and y (physical units)
    drift_x = np.zeros(num_segs)
    drift_y = np.zeros(num_segs)

    relative_drift_x = 0.0
    relative_drift_y = 0.0

    ref = (
        np.round(ref_x / Intersect_d).astype(np.int64)
        + np.round(ref_y / Intersect_d).astype(np.int64) * image_width
    )
    ref_coords, ref_counts = np.unique(ref, return_counts=True)

    x_work = x.copy()
    y_work = y.copy()

    for seg in range(1, len(slices)):
        s = slices[seg]
        x1 = x_work[s]
        y1 = y_work[s]

        if len(x1) == 0:
            drift_x[seg] = drift_x[seg - 1]
            drift_y[seg] = drift_y[seg - 1]
            continue

        # convert target coordinates to a 1D array in intersect_d units
        x_scaled = np.round(relative_drift_x + x1 / Intersect_d).astype(np.int64)
        y_scaled = np.round(relative_drift_y + y1 / Intersect_d).astype(np.int64)

        target = x_scaled + y_scaled * int(image_width)

        # target counts
        tar_coords, tar_counts = np.unique(target, return_counts=True)

        # compute intersection counts for each shift in the local search region
        roi = _point_intersect_2d(
            ref_coords=ref_coords,
            ref_counts=ref_counts,
            tar_coords=tar_coords,
            tar_counts=tar_counts,
            shifts=shifts,
            roi_size=roi_size,
        )

        # estimate the precise sub-pixel position of the peak using FFT.
        # peak_x/peak_y come back in BIN units - convert to physical units
        # (same units as x/y) before accumulating, or the correction applied
        # to the coordinates will be off by a factor of Intersect_d.
        peak_x, peak_y = _get_fft_peak(roi, roi_size)

        relative_drift_x += peak_x
        relative_drift_y += peak_y
        drift_x[seg] = -relative_drift_x
        drift_y[seg] = -relative_drift_y

    drift_x *= Intersect_d  # convert to physical units
    drift_y *= Intersect_d  # convert to physical units

    frames_full, frame_drift = _compute_frame_drift(
        seg_centers, np.column_stack([drift_x, drift_y]), len(frame_ids), kind='pchip'
    )

    # Correct from the ORIGINAL, untouched x/y - not x_work.
    x_corrected, y_corrected = x.copy(), y.copy()

    _shift_correction(frame_ids, x_corrected, y_corrected, None, frame_drift)

    return {
        'x_corrected': x_corrected,
        'y_corrected': y_corrected,
        'full_frames': frames_full,
        'frame_drift': frame_drift,
        'drift_x': drift_x,
        'drift_y': drift_y,
    }


def intersection_max_z(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_z: np.ndarray,
    frame_ids: np.ndarray,
    slices: list[slice],
    seg_centers: np.ndarray,
    image_width: int,
    image_height: int,
    Intersect_d: float = 20.0,
    roi_r: int = 3,
):
    '''
    Parameters
    ----------
    x, y, z : np.ndarray
        Arrays of x, y, and z coordinates of localizations.
    ref_x, ref_y, ref_z : np.ndarray
        Arrays of x, y, and z coordinates of reference localizations.
    frame_ids : np.ndarray
        Array for fast slice indexing of frames.
    slices : list[slice]
        List of slices defining temporal segments of the dataset.
    seg_centers : np.ndarray
        Array of center frame numbers for each segment.
    image_width, image_height : int
        Bin counts in x and y, i.e. already `ceil(extent_physical / Intersect_d)`.

        Do not divide by Intersect_d again here - see intersection_max for why.
    Intersect_d : float, optional
        Intersection distance, by default 20.0
    roi_r : int, optional
        Radius of the region of interest for local search, by default 3
    '''
    num_segs = len(slices)

    # transfer 2D local search region to 1D
    roi_size = 2 * roi_r + 1

    # image_width / image_height are ALREADY unitless bin counts - use them
    # directly as the row/plane strides, consistent with how `ref` is built.
    width = image_width
    height = image_height

    shifts_z = np.arange(-roi_r, roi_r + 1, 1)
    shifts_z = (shifts_z * width * height).astype(np.int64)

    # drift in z (physical units)
    drift_z = np.zeros(num_segs)

    relative_drift_z = 0.0

    ref = (
        np.round(ref_x / Intersect_d).astype(np.int64)
        + np.round(ref_y / Intersect_d).astype(np.int64) * width
        + np.round(ref_z / Intersect_d).astype(np.int64) * width * height
    )
    ref_coords, ref_counts = np.unique(ref, return_counts=True)

    # Private working copy - see intersection_max for why this matters.
    z_work = z.copy()

    for seg in range(1, len(slices)):
        s = slices[seg]
        x1 = x[s]
        y1 = y[s]
        z1 = z_work[s]

        if len(x1) == 0:
            drift_z[seg] = drift_z[seg - 1]
            continue

        x_scaled = np.round(x1 / Intersect_d).astype(np.int64)
        y_scaled = np.round(y1 / Intersect_d).astype(np.int64)
        z_scaled = np.round(z1 / Intersect_d).astype(np.int64)

        target = x_scaled + y_scaled * int(width) + z_scaled * int(width) * int(height)

        # target counts
        tar_coords, tar_counts = np.unique(target, return_counts=True)

        tar_coords = tar_coords + round(relative_drift_z) * width * height

        # compute intersection counts for each shift in the local search region
        roi = _point_intersect_1d(
            ref_coords=ref_coords,
            ref_counts=ref_counts,
            tar_coords=tar_coords,
            tar_counts=tar_counts,
            shifts=shifts_z,
            roi_size=roi_size,
        )

        # peak_z comes back in BIN units - convert to physical units.
        peak_z = _get_fft_peak_z(roi, roi_size)

        relative_drift_z += peak_z
        drift_z[seg] = -relative_drift_z

    drift_z *= Intersect_d  # convert to physical units

    frames_full, frame_drift = _compute_frame_drift(
        seg_centers, np.column_stack([drift_z]), len(frame_ids), kind='pchip'
    )

    # Correct from the ORIGINAL, untouched z - not z_work.
    z_corrected = z.copy()[:, np.newaxis]

    _shift_correction(frame_ids, z_corrected, None, None, frame_drift)

    return {
        'z_corrected': z_corrected.squeeze(),
        'full_frames': frames_full,
        'frame_drift': frame_drift,
        'drift_z': drift_z,
    }


def _run_two_stage_aim_engine(
    coords: np.ndarray,
    slices: list[slice],
    intersect_d: float,
    max_search: float,
    frame_ids: np.ndarray,
    seg_centers: np.ndarray,
    image_width: int,
    image_height: int,
    perform_z_correction: bool = False,
) -> np.ndarray:
    '''
    Generic 2-Stage AIM engine. Operates on any shape (N, D) - 1D, 2D, or 3D.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, D) containing localization coordinates, in
        PHYSICAL units.
    slices : list[slice]
        List of slices defining temporal segments of the dataset.
    intersect_d : float
        Intersection radius threshold in physical units (typically 15-30 nm).
    max_search : float
        Maximum search distance (currently informational only - see note in
        aim_drift_correction about roi_r).
    frame_ids : np.ndarray
        Array for fast slice indexing of frames.
    seg_centers : np.ndarray
        Array of center frame numbers for each segment.
    image_width, image_height : int
        Width/height of the image in BIN units (already divided by intersect_d
        once, upstream in aim_drift_correction).

    Returns
    -------
    shifts_s2 : np.ndarray
        Array of shape (num_segs, D) containing the estimated shifts for each segment.
    '''
    num_segs = len(slices)
    ndim = coords.shape[1]

    segments = [coords[s] for s in slices]

    # STAGE 1: Preliminary Shift Estimation against Segment 0
    ref_s1 = segments[0]

    res = intersection_max(
        coords[:, 0],
        coords[:, 1],
        ref_s1[:, 0],
        ref_s1[:, 1],
        frame_ids,
        slices,
        seg_centers,
        image_width,
        Intersect_d=intersect_d,
        roi_r=3,  # int(max_search / intersect_d)
    )

    # STAGE 2: Global Refinement - use the stage-1 corrected dataset as its
    # own (much denser) reference.
    res2 = intersection_max(
        res['x_corrected'],
        res['y_corrected'],
        res['x_corrected'],
        res['y_corrected'],
        frame_ids,
        slices,
        seg_centers,
        image_width,
        Intersect_d=intersect_d,
        roi_r=3,  # int(max_search / intersect_d)
    )

    drift_x = res['drift_x'] + res2['drift_x']
    drift_y = res['drift_y'] + res2['drift_y']

    if ndim == 3 and perform_z_correction:
        ref_s2_x = res2['x_corrected'][slices[0]]
        ref_s2_y = res2['y_corrected'][slices[0]]
        ref_s1_z = coords[:, 2][slices[0]]  # Z is not yet corrected

        res_z = intersection_max_z(
            res2['x_corrected'],
            res2['y_corrected'],
            coords[:, 2],
            ref_s2_x,
            ref_s2_y,
            ref_s1_z,
            frame_ids,
            slices,
            seg_centers,
            image_width,
            image_height,
            Intersect_d=intersect_d,
            roi_r=3,
        )

        res_z2 = intersection_max_z(
            res2['x_corrected'],
            res2['y_corrected'],
            res_z['z_corrected'],
            res2['x_corrected'],
            res2['y_corrected'],
            res_z['z_corrected'],
            frame_ids,
            slices,
            seg_centers,
            image_width,
            image_height,
            Intersect_d=intersect_d,
            roi_r=3,
        )
        drift_z = res_z['drift_z'] + res_z2['drift_z']
    else:
        drift_z = np.zeros(num_segs)

    # Force absolute re-anchoring of the global drift to Segment 0
    drift_x -= drift_x[0]
    drift_y -= drift_y[0]
    drift_z -= drift_z[0]

    return np.column_stack([drift_x, drift_y, drift_z])


def aim_drift_correction(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray = None,
    segment_size: int = 100,
    intersect_d: float = 20.0,
    z_scale: float = 0.5,
    max_search: float = 100.0,
    interp_kind: str = 'pchip',
) -> dict:
    """
    Generalized AIM Drift Correction supporting 2D and 3D datasets.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (N,) containing integer frame indices.
    x : np.ndarray
        Array of shape (N,) containing x-coordinates, in physical units (e.g. nm).
    y : np.ndarray
        Array of shape (N,) containing y-coordinates, in physical units.
    z : np.ndarray, optional
        Array of shape (N,) containing z-coordinates, in physical units.
        Required for 3D datasets. If provided, z-correction is performed.
    segment_size : int
        Number of frames per temporal block.
    intersect_d : float
        Intersection radius threshold in physical units (typically 15-30 nm).
    z_scale : float
        Scaling factor for z-coordinates to account for axial distortion.
    max_search : float
        Currently informational only. The actual per-round search radius is
        controlled by `roi_r` inside intersection_max/intersection_max_z
        (hardcoded to 3 bins there) - if you need a wider or narrower search
        window, adjust roi_r rather than max_search.
    interp_kind : str
        Interpolation method for per-frame drift: 'pchip' (default), 'linear',
        'cubic', etc.

    Returns
    -------
    dict
        Dictionary containing:
        - `block_shifts`: Coarse (num_segs, ndim) drift points.
        - `seg_centers`: X-axis frame indices for plotting coarse points.
        - `frame_full`: Frame indices for the full drift array.
        - `frame_drift`: Continuous (max_frame, ndim) per-frame drift curves.
        - `slices`: Pre-computed contiguous slices.
        - `frame_ids`: Frame IDs for fast slice indexing.
    """
    coords = np.column_stack([x, y])
    if z is not None:
        coords = np.column_stack([coords, z])
    frames = np.asarray(frames)
    ndim = coords.shape[1]

    # sort frames and coords together to ensure temporal order
    sort_indices = np.argsort(frames)
    # check if frames are already sorted
    if not np.all(sort_indices == np.arange(len(frames))):
        coords = coords[sort_indices]
        frames = frames[sort_indices]

    # Fast slice creation
    frame_ids = np.cumsum(np.bincount(frames.astype(np.int64)))
    min_frame = int(frames.min())
    max_frame = int(frames.max()) + 1
    seg_edges = np.arange(min_frame, max_frame + segment_size, segment_size)
    num_segs = len(seg_edges) - 1
    seg_centers = seg_edges[:-1] + (segment_size / 2.0)

    # Generate contiguous memory slices
    slices = [
        slice(
            frame_ids[min(seg_edges[i], len(frame_ids) - 1)],
            frame_ids[min(seg_edges[i + 1], len(frame_ids) - 1)],
        )
        for i in range(num_segs)
    ]

    # image_width/image_height are unitless BIN counts. This is the one and
    # only place they get divided by intersect_d - every downstream function
    # must treat them as already-unitless from here on.
    #
    # Add a safety margin (+10) to the bin dimensions so shifted
    # coordinates never wrap around rows
    offset = 10
    image_width = int((np.round(coords[:, 0].max()) + offset) / intersect_d)
    image_height = int((np.round(coords[:, 1].max()) + offset) / intersect_d)

    scaled_coords = coords.copy()
    if ndim == 3 and z_scale != 1.0:
        scaled_coords[:, 2] *= z_scale

    block_shifts = _run_two_stage_aim_engine(
        scaled_coords,
        slices,
        intersect_d,
        max_search,
        frame_ids,
        seg_centers,
        image_width,
        image_height,
        perform_z_correction=(ndim == 3 and z is not None),
    )

    if ndim == 3 and z_scale != 1.0:
        block_shifts[:, 2] /= z_scale

    # Interpolate to per-frame drift array (max_frame, ndim)
    frame_full, frame_drift = _compute_frame_drift(
        seg_centers, block_shifts, max_frame, kind=interp_kind
    )

    # Return structured results dict for plotting, saving, and your Numba kernel
    return {
        # Coarse (num_segs, ndim) drift points
        'block_shifts': block_shifts,
        # X-axis frame indices for plotting coarse points
        'seg_centers': seg_centers,
        # Frame indices for the full drift array
        'frame_full': frame_full,
        # Continuous (max_frame, ndim) per-frame drift curves
        'frame_drift': frame_drift,
        # Pre-computed contiguous slices
        'slices': slices,
        # frame_ids for fast slice indexing
        'frame_ids': frame_ids,
    }
