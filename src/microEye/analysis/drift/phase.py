import numba as nb
import numpy as np
import pyqtgraph as pg
from skimage.registration import phase_cross_correlation


@nb.njit
def _process_single_image(image, reference, pixelSize, upsampling):
    with nb.objmode(shift='float64[:]'):
        shift = phase_cross_correlation(image, reference, upsample_factor=upsampling)[0]
    return shift * pixelSize


@nb.njit(parallel=True, cache=True)
def phase_shift_estimation(sub_images, pixelSize, upsampling):
    shifts = np.zeros((len(sub_images), 2))
    reference = sub_images[0]
    for idx in nb.prange(0, len(sub_images)):
        shifts[idx] = _process_single_image(
            sub_images[idx], reference, pixelSize, upsampling
        )
    return shifts
