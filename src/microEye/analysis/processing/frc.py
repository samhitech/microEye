import cv2
import numba
import numpy as np
import pyqtgraph as pg
from scipy.interpolate import PchipInterpolator, UnivariateSpline
from scipy.signal import savgol_filter

from microEye.analysis.rendering import BaseRenderer
from microEye.analysis.utils import *


def FRC_compute(
    fft_12: np.ndarray, fft_11: np.ndarray, fft_22: np.ndarray, FRC_res, R, R_max
):
    R = R.reshape(-1)
    argsort = np.argsort(R)
    R = R[argsort]
    fft_12 = fft_12.reshape(-1)[argsort]
    fft_11 = fft_11.reshape(-1)[argsort]
    fft_22 = fft_22.reshape(-1)[argsort]
    ids = np.cumsum(np.bincount(R.astype(np.int64)))
    for idx in range(1, R_max + 1):
        a = np.sum(fft_12[ids[idx - 1] : ids[idx]])
        b = np.sum(fft_11[ids[idx - 1] : ids[idx]])
        c = np.sum(fft_22[ids[idx - 1] : ids[idx]])
        FRC_res[idx - 1] = a / np.sqrt(b * c)


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
    n_points = data.shape[0]

    # Split data based on method
    if 'Binomial' in method:
        coin = np.random.binomial(1, 0.5, (n_points))
        data1, data2 = data[coin == 0], data[coin == 1]
    elif 'Odd/Even' in method:
        data1, data2 = data[::2], data[1::2]
    elif 'Halves' in method:
        data1, data2 = data[: data.shape[0] // 2], data[data.shape[0] // 2 :]

    # Generate and process images
    hist = BaseRenderer(pixel_size)
    image_1 = hist.from_array(data1)
    image_2 = hist.from_array(data2)

    image_1, image_2 = match_shape(image_1, image_2)
    image_1 /= np.sum(image_1)
    image_2 /= np.sum(image_2)

    window = hamming_rect(image_1.shape[1], image_1.shape[0])
    image_1 *= window
    image_2 *= window

    # Compute FFT
    fft_1 = cv2.dft(np.float64(image_1), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_1 = fft_1[..., 0] + 1j * fft_1[..., 1]
    fft_2 = cv2.dft(np.float64(image_2), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_2 = fft_2[..., 0] + 1j * fft_2[..., 1]

    fft_12 = np.fft.fftshift(np.real(fft_1 * np.conj(fft_2)))
    fft_11 = np.fft.fftshift(np.abs(fft_1) ** 2)
    fft_22 = np.fft.fftshift(np.abs(fft_2) ** 2)

    # Process results
    R, _ = radial_coordinate(image_1.shape)
    R = np.round(R)
    frequencies = np.fft.rfftfreq(R.shape[0], d=pixel_size)
    R_max = frequencies.shape[0]
    FRC_res = np.zeros(R_max)

    FRC_compute(fft_12, fft_11, fft_22, FRC_res, R, R_max)

    # Interpolate and smooth results
    pchip = PchipInterpolator(frequencies, FRC_res)
    FRC = pchip(frequencies)

    resolution, smoothed = get_frc_resolution_savgol(
        frequencies, FRC, threshold=1 / 7, window_ratio=0.1, polyorder=2
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
