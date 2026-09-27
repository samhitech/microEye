import numba as nb
import numpy as np
from scipy.interpolate import PchipInterpolator, interp1d


def _compute_frame_drift(
    seg_centers: np.ndarray,
    drift_segments: np.ndarray,
    max_frame: int,
    kind: str = 'pchip',
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates coarse block drift vectors down to a per-frame
    drift array (max_frame, ndim).

    Parameters
    ----------
    seg_centers : np.ndarray
        Array of shape (num_segs,) containing the center frame of each segment.
    drift_segments : np.ndarray
        Array of shape (num_segs, ndim) containing the estimated drift for each segment.
    max_frame : int
        The maximum frame index in the dataset.
    kind : str
        Interpolation method: 'pchip' (default) or 'linear', 'cubic', etc.

    Returns
    -------
    frames_full : np.ndarray
        Array of shape (max_frame,) containing all frame indices.
    frame_drift : np.ndarray
        Array of shape (max_frame, ndim) containing the interpolated
        drift for each frame.
    """
    ndim = drift_segments.shape[1]
    frames_full = np.arange(0, max_frame)
    frame_drift = np.zeros((max_frame, ndim), dtype=np.float64)

    for d in range(ndim):
        if kind == 'pchip':
            # PCHIP prevents cubic extrapolation ripples at boundary frames
            interp_func = PchipInterpolator(
                seg_centers, drift_segments[:, d], extrapolate=True
            )
        else:
            interp_func = interp1d(
                seg_centers, drift_segments[:, d], kind=kind, fill_value='extrapolate'
            )
        frame_drift[:, d] = interp_func(frames_full)

    return frames_full, frame_drift


@nb.jit(nopython=True, parallel=True)
def _shift_correction(frame_bins, x, y, z, shifts: np.ndarray):
    for idx in nb.prange(0, len(shifts)):
        mask = slice(frame_bins[idx - 1] if idx > 0 else 0, frame_bins[idx])
        x[mask] -= shifts[idx, 0]
        if y is not None and shifts.ndim > 1 and shifts.shape[1] > 1:
            y[mask] -= shifts[idx, 1]
        if z is not None and shifts.ndim > 1 and shifts.shape[1] > 2:
            z[mask] -= shifts[idx, 2]
