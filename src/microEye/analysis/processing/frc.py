import logging
from time import perf_counter_ns

import cv2
import numba
import numpy as np
import pyqtgraph as pg
from numba import cuda
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter

from microEye.analysis.rendering import BaseRenderer
from microEye.analysis.utils import get_rounded_radial_coord, hamming_rect

logger = logging.getLogger(__name__)

if cuda.is_available():
    import gc

    import cupy as cp
    import cupyx as cpx

    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)

    FRC_METHOD = 'GPU'

    _frc_kernel = cp.ElementwiseKernel(
        in_params='complex64 f1, complex64 f2, S r_int, S r_max',
        out_params='raw float64 out_a, raw float64 out_b, raw float64 out_c',
        operation='''
            if (r_int <= r_max) {
                double r1 = f1.real();
                double i1 = f1.imag();
                double r2 = f2.real();
                double i2 = f2.imag();

                // Compute atomic additions in hardware right away
                atomicAdd(&out_a[r_int], r1 * r2 + i1 * i2);
                atomicAdd(&out_b[r_int], r1 * r1 + i1 * i1);
                atomicAdd(&out_c[r_int], r2 * r2 + i2 * i2);
            }
            ''',
        name='frc_fused_kernel',
    )

    _image_norm_windows_kernel = cp.ElementwiseKernel(
        in_params='float32 img1, float32 img2, float32 win, float32 inv_sum1, float32 inv_sum2',  # noqa: E501
        out_params='float32 out_img1, float32 out_img2',
        operation='''
            out_img1 = img1 * inv_sum1 * win;
            out_img2 = img2 * inv_sum2 * win;
            ''',
        name='image_norm_windows_kernel',
    )

    def FRC_compute(image_1, image_2, window, R_int, R_max):
        start_time = perf_counter_ns()
        # Transfer array to GPU memory (do this once at ingestion)
        gpu_image_1 = cp.asarray(image_1, dtype=cp.float32)
        gpu_image_2 = cp.asarray(image_2, dtype=cp.float32)

        gpu_window = cp.asarray(window, dtype=cp.float32)
        logger.info(
            f'Images Transferred to GPU {(perf_counter_ns() - start_time) / 1e6:.2f} ms'
        )

        start_time = perf_counter_ns()
        inv_sum1 = cp.float32(1.0) / cp.sum(gpu_image_1)
        inv_sum2 = cp.float32(1.0) / cp.sum(gpu_image_2)

        _image_norm_windows_kernel(
            gpu_image_1, gpu_image_2, gpu_window, inv_sum1, inv_sum2,
            gpu_image_1, gpu_image_2
        )
        logger.info(
            f'Images norm and win in {(perf_counter_ns() - start_time) / 1e6:.2f} ms'
        )

        start_time = perf_counter_ns()
        # Compute 2D FFT instantly on the GPU hardware
        fft_1: np.ndarray = cp.fft.fft2(gpu_image_1)
        fft_2: np.ndarray = cp.fft.fft2(gpu_image_2)


        r_int = cp.asarray(np.fft.fftshift(R_int)).ravel()


        logger.info(f'FFT computed in {(perf_counter_ns() - start_time) / 1e6:.2f} ms')

        # allocate accumulator bins on the GPU
        a = cp.zeros(R_max + 1, dtype=cp.float64)
        b = cp.zeros(R_max + 1, dtype=cp.float64)
        c = cp.zeros(R_max + 1, dtype=cp.float64)

        # parallelized hardware-level atomic additions
        _frc_kernel(fft_1.ravel(), fft_2.ravel(), r_int, R_max, a, b, c)

        numerator = a[1 : R_max + 1]
        denominator = cp.sqrt(b[1 : R_max + 1] * c[1 : R_max + 1])

        frc_gpu = numerator / denominator

        res = frc_gpu.get()

        del (
            gpu_image_1,
            gpu_image_2,
            fft_1,
            fft_2,
            r_int,
            a,
            b,
            c,
            numerator,
            denominator,
            frc_gpu,
        )
        gc.collect()

        return res
else:
    FRC_METHOD = 'CPU'

    @numba.njit(
        'float64[:](float32[:, :], float32[:, :], float32[:, :], float32[:, :], int32[:, :], int64)',  # noqa: E501
        cache=True,
    )
    def FRC_compute_kernel(re1, im1, re2, im2, R_int, R_max):
        a = np.zeros(R_max + 1)
        b = np.zeros(R_max + 1)
        c = np.zeros(R_max + 1)
        re1f, im1f = re1.ravel(), im1.ravel()
        re2f, im2f = re2.ravel(), im2.ravel()
        rf = R_int.ravel()
        for i in range(rf.shape[0]):
            r = rf[i]
            if r <= R_max:
                a[r] += re1f[i] * re2f[i] + im1f[i] * im2f[i]
                b[r] += re1f[i] * re1f[i] + im1f[i] * im1f[i]
                c[r] += re2f[i] * re2f[i] + im2f[i] * im2f[i]
        return a[1 : R_max + 1] / np.sqrt(b[1 : R_max + 1] * c[1 : R_max + 1])

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def window_and_normalize(image_1, image_2, window):
        sum_1 = np.sum(image_1)
        sum_2 = np.sum(image_2)

        inv_sum_1 = 1.0 / sum_1
        inv_sum_2 = 1.0 / sum_2

        flat_img1 = image_1.ravel()
        flat_img2 = image_2.ravel()
        flat_win = window.ravel()

        for i in numba.prange(flat_img1.size):
            flat_img1[i] = flat_img1[i] * inv_sum_1 * flat_win[i]
            flat_img2[i] = flat_img2[i] * inv_sum_2 * flat_win[i]

    def FRC_compute(image_1, image_2, window, R, R_max):
        start_time = perf_counter_ns()
        window_and_normalize(image_1, image_2, window)
        logger.info(
            f'Images norm and win in {(perf_counter_ns() - start_time) / 1e6:.2f} ms'
        )

        start_time = perf_counter_ns()
        fft_1 = cv2.dft(image_1, flags=cv2.DFT_COMPLEX_OUTPUT)
        re1, im1 = fft_1[..., 0], fft_1[..., 1]
        fft_2 = cv2.dft(image_2, flags=cv2.DFT_COMPLEX_OUTPUT)
        re2, im2 = fft_2[..., 0], fft_2[..., 1]
        logger.info(f'FFT computed in {(perf_counter_ns() - start_time) / 1e6:.2f} ms')

        # Process the FFT results
        return FRC_compute_kernel(re1, im1, re2, im2, np.fft.fftshift(R), R_max)


def get_frc_resolution_savgol(
    frequencies: np.ndarray,
    frc_vals: np.ndarray,
    threshold: float = 1 / 7,
    window_ratio: float = 0.08,  # ~8% of total points
    polyorder: int = 2,
) -> tuple[float, np.ndarray]:
    '''Extracts sub-bin FRC resolution using Savitzky-Golay smoothing

    and exact sub-interval crossing interpolation.

    Parameters
    ----------
    frequencies : np.ndarray
        Spatial frequencies.
    frc_vals : np.ndarray
        Raw FRC values.
    threshold : float
        Resolution threshold (default 1/7).
    window_ratio : float
        Fraction of array length to use for smoothing window (0.05 - 0.10 works best).
    polyorder : int
        Order of polynomial
        (use 1 for a cleaner local average, 2 for curvature preservation).

    Returns
    -------
    resolution : float
        Spatial resolution (1 / q_cross).
    frc_smoothed : np.ndarray
        The smoothed FRC curve array for plotting.
    '''
    frequencies = np.asarray(frequencies)
    frc_vals = np.asarray(frc_vals)
    n_points = len(frc_vals)

    # 1. Calculate an optimal odd window length
    win_len = int(n_points * window_ratio)
    if win_len <= polyorder:
        win_len = polyorder + 2
    if win_len % 2 == 0:
        win_len += 1  # Sav-Gol requires an odd window

    # 2. Smooth curve preserving local slope
    frc_smoothed = savgol_filter(frc_vals, window_length=win_len, polyorder=polyorder)

    # 3. Find the first downward threshold crossing
    below = frc_smoothed <= threshold
    if not np.any(below):
        return np.nan, frc_smoothed

    idx = np.argmax(below)

    if idx == 0:
        return np.nan, frc_smoothed  # Dropped immediately at origin

    # 4. Sub-bin exact linear crossing calculation
    # Interpolate exact q between the bin right above and right below 1/7
    q1, q2 = frequencies[idx - 1], frequencies[idx]
    f1, f2 = frc_smoothed[idx - 1], frc_smoothed[idx]

    q_cross = q1 + (threshold - f1) * (q2 - q1) / (f2 - f1)

    return float(1.0 / q_cross), frc_smoothed


def FRC_resolution_binomial(data: np.ndarray, pixel_size=10, method='Binomial'):
    """Fourier Ring Correlation based on
    https://www.frontiersin.org/articles/10.3389/fbinf.2021.817254/

    Parameters
    ----------
    data : np.ndarray
        data to be resolution estimated, columns (X, Y, intensity)
    pixelSize : int, optional
        super resolution image pixel size in nanometers, by default 10
    method : str, optional
        method to split data, by default 'Binomial'
        - 'Binomial' : randomly split data into two halves
        - 'Odd/Even' : split data into odd and even indices
        - 'Halves' : split data into first and second halves

    Returns
    -------
    frequencies : np.ndarray
        spatial frequencies
    FRC : np.ndarray
        Fourier Ring Correlation values
    smoothed : np.ndarray
        smoothed FRC values
    FRC_res : float
        estimated resolution
    """
    init_time = perf_counter_ns()

    n_points = data.shape[0]

    # Split data based on method
    start_time = perf_counter_ns()
    if 'Binomial' in method:
        mask = np.random.rand(n_points) < 0.5
        data1, data2 = data[mask], data[~mask]
    elif 'Odd/Even' in method:
        data1, data2 = data[::2], data[1::2]
    elif 'Halves' in method:
        data1, data2 = data[: data.shape[0] // 2], data[data.shape[0] // 2 :]
    logger.info(f'Data split in {(perf_counter_ns() - start_time) / 1e6:.2f} ms')

    # Generate and process images
    hist = BaseRenderer(pixel_size)
    start_time = perf_counter_ns()
    max_dims = [
        cv2.getOptimalDFTSize(
            np.ceil((data1[:, i].max() - data1[:, i].min()) / pixel_size).astype(int)
        )
        for i in range(2)
    ]
    image_1 = hist.from_array(data1, shape=max_dims)
    logger.info(f'Image 1 generated in {(perf_counter_ns() - start_time) / 1e6:.2f} ms')
    start_time = perf_counter_ns()
    image_2 = hist.from_array(data2, shape=max_dims)
    logger.info(f'Image 2 generated in {(perf_counter_ns() - start_time) / 1e6:.2f} ms')

    assert image_1.shape == image_2.shape, 'Images must have the same shape for FRC.'

    # Process results
    start_time = perf_counter_ns()
    R = get_rounded_radial_coord(image_1.shape)
    frequencies = np.fft.rfftfreq(R.shape[0], d=pixel_size)
    R_max = frequencies.shape[0]
    logger.info(
        f'R coordinates computed in {(perf_counter_ns() - start_time) / 1e6:.2f} ms'
    )

    # windowing
    window = hamming_rect(image_1.shape[1], image_1.shape[0]).astype(np.float32)

    # Compute FFT
    FRC_res = FRC_compute(
        image_1, image_2, window, R, R_max
    )  # Returns FRC values for each frequency bin

    # Interpolate and smooth results
    start_time = perf_counter_ns()
    pchip = PchipInterpolator(frequencies, FRC_res)
    FRC = pchip(frequencies)

    resolution, smoothed = get_frc_resolution_savgol(
        frequencies, FRC, threshold=1 / 7, window_ratio=0.1, polyorder=2
    )
    logger.info(
        f'FRC ({FRC_METHOD}) resolution: {np.round(resolution, 1)} nm '
        f'({(perf_counter_ns() - init_time) / 1e9:.2f} s total)'
    )

    return frequencies, FRC, smoothed, resolution


def plot_frc(frequencies, FRC, smoothed, FRC_res, return_plot_items=False):

    items = [
        pg.PlotCurveItem(x=frequencies, y=FRC, pen=pg.mkPen('r', width=2), name='FRC'),
        pg.PlotCurveItem(
            x=frequencies, y=smoothed, pen=pg.mkPen('b', width=2), name='FRC smoothed'
        ),
        pg.InfiniteLine(
            pos=1 / 7, angle=0, pen=pg.mkPen('y', width=2), name='1/7 threshold'
        ),
    ]

    if return_plot_items:
        return items

    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    legend = plt.addLegend()
    legend.anchor(itemPos=(1, 0), parentPos=(1, 0))

    plt.setLabel('left', 'FRC', units='')
    plt.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')
    plt.setWindowTitle(f'FRC resolution: {np.round(FRC_res, 1)} nm')
    plt.setXRange(0, frequencies[-1])
    plt.setYRange(0, 1)

    for item in items:
        plt.addItem(item)
