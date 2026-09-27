import numba
import numpy as np


def radial_coordinate(shape, dtype=np.float32):
    y_len = np.arange(-shape[0] // 2, shape[0] // 2, dtype=dtype)
    x_len = np.arange(-shape[1] // 2, shape[1] // 2, dtype=dtype)

    # square in-place as 1D vectors to save operations
    y_sq = y_len[:, np.newaxis] ** 2  # shape: (rows, 1)
    x_sq = x_len**2  # shape: (cols,)

    Rsq = y_sq + x_sq

    return np.sqrt(Rsq), Rsq

@numba.njit(parallel=True, fastmath=True, cache=True)
def get_rounded_radial_coord(shape):
    rows, cols = shape[0], shape[1]

    # Pre-allocate ONLY the final output array as int32
    R = np.empty((rows, cols), dtype=np.int32)

    # Midpoints for centering
    y_mid = rows // 2
    x_mid = cols // 2

    # Parallelized, vectorized loop
    for i in numba.prange(rows):
        y_val = float(i - y_mid)
        y_sq = y_val * y_val

        for j in range(cols):
            x_val = float(j - x_mid)
            r_float = np.sqrt(y_sq + (x_val * x_val))

            # Round and cast to int32 on the fly
            R[i, j] = np.int32(np.round(r_float))

    return R
