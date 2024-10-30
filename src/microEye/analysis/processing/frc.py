import cv2
import numba
import numpy as np
import pyqtgraph as pg
from scipy.interpolate import interp1d
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


def FRC_resolution_binomial(data: np.ndarray, pixel_size=10, method='Binomial'):
    '''Fourier Ring Correlation based on
    https://www.frontiersin.org/articles/10.3389/fbinf.2021.817254/

    Parameters
    ----------
    data : np.ndarray
        data to be resolution estimated, columns (X, Y, intensity)
    pixelSize : int, optional
        super resolution image pixel size in nanometers, by default 10
    '''
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

    window = hamming_2Dwindow(image_1.shape[0])
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
    interpy = interp1d(frequencies, FRC_res, kind='cubic', fill_value='extrapolate')
    FRC = interpy(frequencies)

    window_length = min(len(FRC_res) // 20, 101)
    window_length = window_length if window_length % 2 == 1 else window_length + 1
    smoothed = savgol_filter(FRC_res, window_length, 3)

    idx = np.where(smoothed <= (1 / 7))[0]
    if idx is not None and len(idx) > 0:
        idx = idx.min()
        FRC_res = 1 / frequencies[idx]
    else:
        FRC_res = np.nan

    return frequencies, FRC, smoothed, FRC_res

def plot_frc(frequencies, FRC, smoothed, FRC_res):
    plt = pg.plot()
    plt.showGrid(x=True, y=True)
    legend = plt.addLegend()
    legend.anchor(itemPos=(1, 0), parentPos=(1, 0))

    plt.setLabel('left', 'FRC', units='')
    plt.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')
    plt.setWindowTitle(f'FRC resolution: {np.round(FRC_res, 1)} nm')
    plt.setXRange(0, frequencies[-1])
    plt.setYRange(0, 1)

    plt.plot(
        frequencies, FRC,
        pen=pg.mkPen('r', width=2), symbol=None, symbolPen='r',
        symbolBrush=0, name='FRC')
    plt.plot(
        frequencies, smoothed,
        pen=pg.mkPen('b', width=2), symbol=None, symbolPen='b',
        symbolBrush=0, name='FRC smoothed')
    plt.plotItem.addLine(y=1/7, pen='y')
