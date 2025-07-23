import cv2
import numba as nb
import numpy as np

from microEye.analysis.fitting.pyfit3Dcspline.CPU.CPUmleFit_LM import (
    kernel_MLEFit_LM_Sigma,
)


def cross_correlation(image1, image2):
    """
    Compute the cross-correlation of two images using FFT.
    Equivalent to MATLAB's CrossCorrelation function.
    """
    f1 = np.fft.fft2(image1)
    f2 = np.fft.fft2(image2)
    corr = np.fft.fftshift(np.real(np.fft.ifft2(f1 * np.conj(f2))))
    return corr


@nb.njit
def _process_single_image(image, reference, pixelSize, box):
    with nb.objmode(result='float32[:]'):
        xcorr = cross_correlation(image, reference)
        y_max_, x_max_ = np.unravel_index(xcorr.argmax(), xcorr.shape)

        halfBoxSize = int(box / 2)

        xcorr = xcorr[
            y_max_ - halfBoxSize : y_max_ + halfBoxSize + 1,
            x_max_ - halfBoxSize : x_max_ + halfBoxSize + 1,
        ]

        result, crb, ll = kernel_MLEFit_LM_Sigma(
            xcorr.flatten().astype(np.float32), 1.0, int(box + 1), 60, None
        )

        result[0] += x_max_ - halfBoxSize
        result[1] += y_max_ - halfBoxSize

    shift = result[0:2] * pixelSize

    return shift


@nb.njit(parallel=True, cache=True)
def rcc_shift_estimation(sub_images, pixelSize, box=12):
    n_bins = sub_images.shape[0]
    total_pairs = n_bins * (n_bins - 1) // 2

    # Pre-compute all (i,j) pairs and their corresponding linear indices
    pairs = []
    for i in range(n_bins - 1):
        for j in range(i + 1, n_bins):
            pairs.append((i, j))

    imshift = np.zeros((total_pairs, 2), dtype=np.float32)
    A = np.zeros((total_pairs, n_bins - 1), dtype=np.float32)

    # Process reference image once
    origins = {}
    for i in nb.prange(n_bins - 1):
        yorigin, xorigin = _process_single_image(
            sub_images[i], sub_images[i], pixelSize, box
        )
        origins[i] = (yorigin, xorigin)

    # Process all pairs
    for idx in nb.prange(len(pairs)):
        i, j = pairs[idx]
        y, x = _process_single_image(sub_images[i], sub_images[j], pixelSize, box)
        yorigin, xorigin = origins[i]
        imshift[idx, 0] = xorigin - x
        imshift[idx, 1] = yorigin - y
        A[idx, i:j] = 1

    return imshift, A


def rcc_solve(imshift, A, n_bins, rmax=10.0):
    # imshift: (N, 2) or (N, 3)
    # A: (N, n_bins-1)
    # use_z: if True, expects imshift.shape[1] == 3

    # Initial solve
    drift = np.matmul(np.linalg.pinv(A), imshift)
    error = np.matmul(A, drift) - imshift

    # Compute error magnitude
    rowerr = np.zeros((A.shape[0], 2))
    if error.shape[1] == 3:
        rowerr[:, 0] = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2 + error[:, 2] ** 2)
    else:
        rowerr[:, 0] = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
    rowerr[:, 1] = np.arange(A.shape[0])

    # Sort by error (descending)
    rowerr = np.flipud(np.sort(rowerr, axis=0))

    # Outlier rejection
    mask = rowerr[:, 0] > rmax
    index = rowerr[mask, 1].astype(np.int64)

    # Remove outliers (one by one, check rank)
    for idx in range(len(index)):
        flag = int(index[idx])
        tmp = np.delete(A, flag, axis=0)
        if np.linalg.matrix_rank(tmp, tol=1) == (n_bins - 1):
            A = np.delete(A, flag, axis=0)
            imshift = np.delete(imshift, flag, axis=0)
            index[index > flag] -= 1

    # Final solve and cumulative sum
    return np.cumsum(np.matmul(np.linalg.pinv(A), imshift), axis=0)


@nb.njit(parallel=True, cache=True)
def dcc_shift_estimation(sub_images, pixelSize, box=12):
    n_bins = sub_images.shape[0]

    imshift = np.zeros((n_bins - 1, 2), dtype=np.float32)

    yorigin, xorigin = _process_single_image(
        sub_images[0], sub_images[0], pixelSize, box
    )
    # Process
    for i in range(1, n_bins):
        y, x = _process_single_image(sub_images[0], sub_images[i], pixelSize, box)
        imshift[i - 1, 0] = xorigin - x
        imshift[i - 1, 1] = yorigin - y

    return imshift

@nb.njit(parallel=True, cache=True)
def mcc_shift_estimation(sub_images, pixelSize, box=12):
    n_bins = sub_images.shape[0]
    total_pairs = n_bins * (n_bins - 1) // 2

    # Pre-compute all (i,j) pairs and their corresponding linear indices
    pairs = []
    for i in range(n_bins - 1):
        for j in range(i + 1, n_bins):
            pairs.append((i, j))

    imshift = np.zeros((total_pairs, 2), dtype=np.float32)
    drift = np.zeros((n_bins, 2), dtype=np.float32)

    # Process reference image once
    origins = {}
    for i in nb.prange(n_bins - 1):
        yorigin, xorigin = _process_single_image(
            sub_images[i], sub_images[i], pixelSize, box
        )
        origins[i] = (yorigin, xorigin)

    # Process all pairs
    for idx in nb.prange(len(pairs)):
        i, j = pairs[idx]
        y, x = _process_single_image(sub_images[i], sub_images[j], pixelSize, box)
        yorigin, xorigin = origins[i]
        imshift[idx, 0] = xorigin - x
        imshift[idx, 1] = yorigin - y

    # For each bin, sum the shifts as per MCC
    for i in nb.prange(n_bins):
        # Subtract shifts to later bins
        for j in range(i + 1, n_bins):
            idx = (i * (2 * n_bins - i - 1)) // 2 + (j - i - 1)
            drift[i, 0] -= imshift[idx, 0]
            drift[i, 1] -= imshift[idx, 1]
        # Add shifts from earlier bins
        for j in range(i):
            idx = (j * (2 * n_bins - j - 1)) // 2 + (i - j - 1)
            drift[i, 0] += imshift[idx, 0]
            drift[i, 1] += imshift[idx, 1]


    drift /= n_bins

    drift[:, 0] -= drift[0, 0]
    drift[:, 1] -= drift[0, 1]

    return drift
