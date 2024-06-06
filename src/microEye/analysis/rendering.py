
from os import name

import cv2
import numba
import numpy as np
import pyqtgraph as pg
from scipy.interpolate import BSpline, interp1d, splrep

from microEye.qt import QDateTime


def model(xc, yc, sigma_x, sigma_y, flux, offset, X, Y):
    y_gauss = gauss_1d(Y[:, 0], yc, sigma_y)
    x_gauss = gauss_1d(X[0, :], xc, sigma_x)

    return flux * np.einsum('i,j->ij', y_gauss, x_gauss) + offset


@numba.njit(cache=True)
def gauss_1d(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * \
           np.exp(-0.5 * (x - mu)**2 / sigma**2)


def radial_cordinate(shape):
    '''Generates a 2D array with radial cordinates
    with according to the first two axis of the
    supplied shape tuple

    Returns
    -------
    R, Rsq
        Radius 2d matrix (R) and radius squared matrix (Rsq)
    '''
    y_len = np.arange(-shape[0]//2, shape[0]//2)
    x_len = np.arange(-shape[1]//2, shape[1]//2)

    X, Y = np.meshgrid(x_len, y_len)

    Rsq = (X**2 + Y**2)

    return np.sqrt(Rsq), Rsq


class gauss_hist_render:

    def __init__(self, pixelSize=10, is2D_hist=False):
        self._pixel_size = pixelSize
        self._std = pixelSize  # nm
        self._gauss_std = self._std / self._pixel_size
        self._gauss_len = 1 + np.ceil(self._gauss_std * 6)
        if self._gauss_len % 2 == 0:
            self._gauss_len += 1
        self._gauss_shape = [int(self._gauss_len)] * 2

        xy_len = np.arange(0, self._gauss_shape[0])
        X, Y = np.meshgrid(xy_len, xy_len)
        self._gauss_2d = model(
            (self._gauss_len - 1) / 2,
            (self._gauss_len - 1) / 2,
            self._gauss_std,
            self._gauss_std,
            1,
            0,
            X, Y)
        self._image = None

    def render(self, X_loc, Y_loc, Intensity, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        X_loc (np.ndarray)
            Sub-pixel localized points X coordinates
        Y_loc (np.ndarray)
            Sub-pixel localized points Y coordinates
        Intensity (np.ndarray)
            Sub-pixel localized points intensity estimate
        shape tuple(int, int), optional
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        if not len(X_loc) == len(Y_loc) == len(Intensity):
            raise Exception('The supplied arguments are of different lengths.')

        x_min = np.min(X_loc)
        y_min = np.min(Y_loc)

        if x_min < 0:
            X_loc -= x_min
        if y_min < 0:
            Y_loc -= y_min

        if shape is None:
            x_max = int((np.max(X_loc) / self._pixel_size) +
                        4 * self._gauss_len)
            y_max = int((np.max(Y_loc) / self._pixel_size) +
                        4 * self._gauss_len)
        else:
            x_max = shape[1]
            y_max = shape[0]
        n_max = max(x_max, y_max)

        step = int((self._gauss_len - 1) // 2)

        self._image = np.zeros([n_max, n_max])

        X = np.round(X_loc / self._pixel_size) + 4 * step
        Y = np.round(Y_loc / self._pixel_size) + 4 * step

        render_compute(
            np.c_[X, Y, Intensity],
            step, self._gauss_2d,
            self._image)

        # for idx in range(len(X_loc)):
        #     x = round(X_loc[idx] / self._pixel_size) + 4 * step
        #     y = round(Y_loc[idx] / self._pixel_size) + 4 * step

        #     self._image[y - step:y + step + 1, x - step:x + step + 1] += \
        #         Intensity[idx] * self._gauss_2d

        return self._image

    def fromArray(self, data: np.ndarray, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        data (np.ndarray)
            Array with sub-pixel localization data columns (X, Y, Intensity)
        shape tuple(int, int), optional
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        return self.render(data[:, 0], data[:, 1], data[:, 2], shape)


class hist2D_render:

    def __init__(self, pixelSize=10):
        self._pixel_size = pixelSize
        self._image = None

    def render(self, X_loc, Y_loc, Intensity, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        X_loc (np.ndarray)
            Sub-pixel localized points X coordinates
        Y_loc (np.ndarray)
            Sub-pixel localized points Y coordinates
        Intensity (np.ndarray)
            Sub-pixel localized points intensity estimate
        shape tuple(int, int), optional
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        if not len(X_loc) == len(Y_loc) == len(Intensity):
            raise Exception('The supplied arguments are of different lengths.')

        x_min = np.min(X_loc)
        y_min = np.min(Y_loc)

        if x_min < 0:
            X_loc -= x_min
        if y_min < 0:
            Y_loc -= y_min

        if shape is None:
            x_max = int((np.max(X_loc) / self._pixel_size) +
                        4)
            y_max = int((np.max(Y_loc) / self._pixel_size) +
                        4)
        else:
            x_max = shape[1]
            y_max = shape[0]
        n_max = max(x_max, y_max)

        step = 2

        self._image = np.zeros([n_max, n_max])

        X = np.round(X_loc / self._pixel_size) + 2
        Y = np.round(Y_loc / self._pixel_size) + 2

        render_compute(
            np.c_[X, Y, Intensity],
            0, 1,
            self._image)

        return self._image

    def fromArray(self, data: np.ndarray, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        data (np.ndarray)
            Array with sub-pixel localization data columns (X, Y, Intensity)
        shape tuple(int, int), optional
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        return self.render(data[:, 0], data[:, 1], data[:, 2], shape)


@numba.jit(nopython=True)
def render_compute(data, step, gauss_2d, out_img):
    for x, y, Intensity in data:
        out_img[y - step:y + step + 1, x - step:x + step + 1] += \
            Intensity * gauss_2d


def FRC_resolution_binomial(data: np.ndarray, pixelSize=10, method='Binomial'):
    '''Fourier Ring Correlation based on
    https://www.frontiersin.org/articles/10.3389/fbinf.2021.817254/

    Parameters
    ----------
    data : np.ndarray
        data to be resolution estimated, columns (X, Y, intensity)
    pixelSize : int, optional
        super resolution image pixel size in nanometers, by default 10
    '''
    all = QDateTime.currentDateTime()
    print('Initialization ... ')

    n_points = data.shape[0]

    if 'Binomial' in method:
        coin = np.random.binomial(1, 0.5, (n_points))

        # Two separate datsets based on coin flip
        data1, data2 = data[coin == 0], data[coin == 1]
    elif 'Odd/Even' in method:
        data1, data2 = data[::2], data[1::2]
    elif 'Halves' in method:
        data1, data2 = data[:data.shape[0]//2], data[data.shape[0]//2:]

    gaussHist = hist2D_render(pixelSize)

    image_1 = gaussHist.fromArray(data1)
    image_2 = gaussHist.fromArray(data2)

    image_1, image_2 = match_shape(image_1, image_2)

    image_1 /= np.sum(image_1)
    image_2 /= np.sum(image_2)

    window = hamming_2Dwindow(image_1.shape[0])

    image_1 *= window
    image_2 *= window

    start = QDateTime.currentDateTime()
    print('FFT ... ')

    fft_1 = cv2.dft(
            np.float64(image_1), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_1 = fft_1[..., 0] + 1j * fft_1[..., 1]
    fft_2 = cv2.dft(
            np.float64(image_2), flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_2 = fft_2[..., 0] + 1j * fft_2[..., 1]
    # fft_1 = np.fft.fft2(image_1)
    # fft_2 = np.fft.fft2(image_2)
    fft_12 = np.fft.fftshift(
        np.real(fft_1 * np.conj(fft_2)))

    fft_11 = np.fft.fftshift(np.abs(fft_1)**2)
    fft_22 = np.fft.fftshift(np.abs(fft_2)**2)

    R, _ = radial_cordinate(image_1.shape)
    R = np.round(R)

    frequencies = np.fft.rfftfreq(R.shape[0], d=pixelSize)
    freq_nyq = frequencies.max()
    R_max = frequencies.shape[0]

    FRC_res = np.zeros(R_max)

    print(f'{start.msecsTo(QDateTime.currentDateTime()) * 1e-3:.3f} s')

    start = QDateTime.currentDateTime()
    print('FRC ... ')
    FRC_compute(fft_12, fft_11, fft_22, FRC_res, R, R_max)

    print(f'{start.msecsTo(QDateTime.currentDateTime()) * 1e-3:.3f} s')

    print('Interpolation ... ')
    interpy = interp1d(
            frequencies, FRC_res,
            kind='cubic', fill_value='extrapolate')
    FRC = interpy(frequencies)
    tck = splrep(frequencies, FRC_res, s=1/pixelSize)
    bspline = BSpline(*tck)
    smoothed = bspline(frequencies)

    idx = np.where(smoothed <= (1/7))[0]
    if idx is not None:
        idx = idx.min()
        FRC_res = 1 / frequencies[idx]
    else:
        FRC_res = np.nan

    print(
        f'Done ... {all.msecsTo(QDateTime.currentDateTime()) * 1e-3:.3f} s')
    return frequencies, FRC, smoothed, FRC_res


def FRC_compute(
        fft_12: np.ndarray, fft_11: np.ndarray, fft_22: np.ndarray,
        FRC_res: np.ndarray, R: np.ndarray, R_max):
    R = R.reshape(-1)
    argsort = np.argsort(R)
    R = R[argsort]
    fft_12 = fft_12.reshape(-1)[argsort]
    fft_11 = fft_11.reshape(-1)[argsort]
    fft_22 = fft_22.reshape(-1)[argsort]
    ids = np.cumsum(np.bincount(R.astype(np.int64)))
    for idx in range(1, R_max + 1):
        a = np.sum(fft_12[ids[idx-1]:ids[idx]])
        b = np.sum(fft_11[ids[idx-1]:ids[idx]])
        c = np.sum(fft_22[ids[idx-1]:ids[idx]])
        FRC_res[idx-1] = (a / np.sqrt(b * c))


@numba.jit(nopython=True, parallel=True)
def masked_sum(array: np.ndarray, mask: np.ndarray):
    a = array.flatten()
    m = np.where(mask.flatten())[0]
    return np.sum(a[m])


def plotFRC(frequencies, FRC, smoothed, FRC_res):

    # print(
    #     'Plot ...               ',
    #     end="\r")
    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    legend = plt.addLegend()
    legend.anchor(
        itemPos=(1, 0), parentPos=(1, 0))

    # set properties of the label for y axis
    plt.setLabel('left', 'FRC', units='')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')

    plt.setWindowTitle(
        f'FRC resolution: {np.round(FRC_res, 1)} nm'
        )

    # setting horizontal range
    plt.setXRange(0, frequencies[-1])

    # setting vertical range
    plt.setYRange(0, 1)

    line1 = plt.plot(
        frequencies, FRC,
        pen=pg.mkPen('r', width=2), symbol=None, symbolPen='r',
        symbolBrush=0, name='FRC')
    line1 = plt.plot(
        frequencies, smoothed,
        pen=pg.mkPen('b', width=2), symbol=None, symbolPen='b',
        symbolBrush=0, name='FRC smoothed')
    line2 = plt.plotItem.addLine(y=1/7, pen='y')

def plotFRC_(frequencies, FRC, FRC_res, cut_off_corrections):
    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'FRC', units='')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Spatial Frequency [1/nm]', units='')

    plt.setWindowTitle(
        'FRC resolution (1st, 2nd, avg): {} | {} | {} nm'.format(
            *np.round(FRC_res, 1))
        )

    # setting horizontal range
    plt.setXRange(0, frequencies[-1])

    # setting vertical range
    plt.setYRange(0, 1)

    line1 = plt.plot(
        frequencies * cut_off_corrections[0], FRC[0],
        pen='g', symbol='x', symbolPen='g',
        symbolBrush=0.2, name='1st Set')
    line2 = plt.plot(
        frequencies * cut_off_corrections[1], FRC[1],
        pen='b', symbol='o', symbolPen='b',
        symbolBrush=0.2, name='2nd Set')
    line3 = plt.plot(
        frequencies * cut_off_corrections[2], FRC[2],
        pen='r', symbol='+', symbolPen='r',
        symbolBrush=0.2, name='Average')
    line4 = plt.plotItem.addLine(y=1/7, pen='y')


def checker_pairs(image: np.ndarray):

    shape = image.shape

    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    # first set
    odd = image[odd_index[0], :][:, odd_index[1]]
    even = image[even_index[0], :][:, even_index[1]]

    odd, even = match_shape(odd, even)

    # reverse set
    oddeven = image[odd_index[0], :][:, even_index[1]]
    evenodd = image[even_index[0], :][:, odd_index[1]]

    oddeven, evenodd = match_shape(oddeven, evenodd)

    return odd, even, oddeven, evenodd


def expand_image(image: np.ndarray, shape):
    if image.shape == shape:
        return image
    else:
        ret = np.zeros(shape)
        ret[:image.shape[0], :image.shape[1]] = image
        return ret


def match_shape(image1: np.ndarray, image2: np.ndarray):
    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape))

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = expand_image(image1, shape)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = expand_image(image2, shape)

    return image1, image2


def hamming_2Dwindow(size: int):
    window1d = np.hamming(size)
    return np.sqrt(np.outer(window1d, window1d))
