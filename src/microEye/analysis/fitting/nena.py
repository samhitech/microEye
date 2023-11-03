import numpy as np
import numba as nb
from scipy.optimize import minimize
from scipy.stats import norm, rayleigh, ncx2, chi2
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import pyqtgraph as pg

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


@nb.njit()
def Rayleigh_dist(x: np.ndarray, x0: float, sig: float):
    '''Rayleigh distribution PDF

        X = (x - x0) / sig

        X[X < 0] = 0

        PDF = 0.5 * X / sig * np.exp(-0.25 * X**2)

    Parameters
    ----------
    x : np.ndarray
        x values array.
    x0 : float
        loc parameter.
    sig : float
        scale parameter.

    Returns
    -------
    np.ndarray
        the Rayleigh distribution PDF
    '''
    X = (x - x0) / sig
    X[X < 0] = 0
    return 0.5 * X / sig * np.exp(-0.25 * X**2)


@nb.njit()
def Gaussian_dist(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    '''Gaussian distribution PDF

        1 / (sqrt(pi / 2) * sigma) * exp(-2 * (x - mu)**2 / sigma**2)

    Parameters
    ----------
    x : np.ndarray
        x values array.
    mu : np.ndarray
        mu parameter.
    sigma : np.ndarray
        sigma parameter.

    Returns
    -------
    np.ndarray
        the Gaussian distribution PDF
    '''
    return 1 / (np.sqrt(np.pi / 2) * sigma) * \
        np.exp(-2 * (x - mu)**2 / sigma**2)


@nb.njit()
def NeNA_model(x: np.ndarray, params):
    '''The NeNA model

    NeNA(x) = a0 * Rayleigh_dist(x, x0=0, sig0) +
    a1 * Gaussian_dist(x, x1, sig1) + a2 * x

    Parameters
    ----------
    x : np.ndarray
        x values array.
    params : list[float]
        list or array of parameters [a0, a1, a2, x1, sig0, sig1]

    Returns
    -------
    np.ndarray
        the NeNA model
    '''
    a0, a1, a2 = params[:3]
    x0 = 0
    x1 = params[3]
    sig0, sig1 = params[4:6]

    return (
        a0 * Rayleigh_dist(x, x0, sig0) +
        a1 * Gaussian_dist(x, x1, sig1) +
        a2 * x)


def NeNA_log_like(params, args):
    '''The NeNA model fitting loglikihood function

    Parameters
    ----------
    params : list[float]
        list or array of parameters [a0, a1, a2, x1, sig0, sig1, sd]
    args : tuple[np.ndarray, np.ndarray]
        tuple of (x, y) data used for fitting

    Returns
    -------
    float
        loglikelihood value
    '''
    sd = params[6]

    x, data = args[0], args[1]

    # only positive coeffs
    for param in params[:3]:
        if not 0 <= param < 100:
            return np.inf

    # only positive sigma
    for param in params[4:6]:
        if param < 0:
            return np.inf

    dataPred = NeNA_model(x, params[:7])

    # Calculate negative log likelihood
    LL = -np.sum(norm.logpdf(data, loc=dataPred, scale=sd))

    return (LL)


def NeNA_fit(x, y, params):
    '''The NeNA model fitting MLE minimization function

    Parameters
    ----------
    x : np.ndarray
        values of x used for fitting.
    y : np.ndarray
        values of y used for fitting.
    params : list[float]
        list or array of parameters [a0, a1, a2, x1, sig0, sig1, sd]

    Returns
    -------
    OptimizeResult
        The optimization result represented as an OptimizeResult object.
    '''
    return minimize(
        NeNA_log_like, params, args=[x, y], method='Nelder-Mead', tol=1e-8)


def get_bincenters(edges: np.ndarray):
    '''Bin centers of histogram bin edges.

    Parameters
    ----------
    edges : np.ndarray
        array of histogram bin edges.

    Returns
    -------
    np.ndarray
        array of bin centers of histogram bin edges.
    '''
    return np.array([(edges[i]+edges[i+1])/2. for i in range(len(edges)-1)])


def NeNA_resolution_estimate(
        distances: np.ndarray, trackIDs: np.ndarray,
        minDist=0.5, range=[0, 200], bins=500,
        a_ray=0.5, a_gauss=0.25, a_lin=1, xc_gauss=15,
        sig_ray=None, sig_gauss=30, sd_fit=1e-4):
    dist = distances[np.logical_and(trackIDs > 0, distances > minDist)]

    x = np.linspace(*range, bins)

    n, bin_edges = np.histogram(dist, density=True, range=range, bins=bins)

    if sig_ray is None:
        sig_ray = np.sqrt(np.sum(np.square(dist)) / (2 * dist.shape[0]))

    res = NeNA_fit(
        get_bincenters(bin_edges), n,
        [a_ray, a_gauss, a_lin, xc_gauss, sig_ray, sig_gauss, sd_fit])

    print('NeNA resolution estimate')
    print('A0 {:.5f} Loc0 {:.5f} Sig0 {:.5f}'.format(res.x[0], 0, res.x[4]))
    print('A1 {:.5f} Loc1 {:.5f} Sig1 {:.5f}'.format(
        res.x[1], res.x[3], res.x[5]))
    print('A2 {:e} SD {:e}'.format(res.x[2], res.x[6]))

    return res, (bin_edges, n, get_bincenters(bin_edges))


class NeNA_Widget(QDialog):

    def __init__(
            self,
            neighbourDists: np.ndarray,
            trackIDs: np.ndarray,
            parent=None,
            ):
        super(NeNA_Widget, self).__init__(parent)

        self.setWindowTitle('NeNA localization precision estimate')
        self.nDists = neighbourDists
        self.trackIDs = trackIDs
        self.maxDist = 200
        self.res = None

        self.flayout = QHBoxLayout()
        self.setLayout(self.flayout)

        self.histogram = pg.PlotWidget()
        self.flayout.addWidget(self.histogram)

        self.fitgroup = QGroupBox('Fitting parameters')
        self.flayout.addWidget(self.fitgroup)

        self.fitlay = QFormLayout()
        self.fitgroup.setLayout(self.fitlay)

        self.bins = QSpinBox()
        self.bins.setMinimum(5)
        self.bins.setMaximum(1e4)
        self.bins.setValue(200)

        self.fitlay.addRow(
            QLabel('N of bins:'), self.bins)

        self.A0 = QDoubleSpinBox()
        self.A0.setMinimum(0)
        self.A0.setMaximum(10)
        self.A0.setDecimals(4)
        self.A0.setSingleStep(0.01)
        self.A0.setValue(0.75)

        self.sig0 = QDoubleSpinBox()
        self.sig0.setMinimum(0)
        self.sig0.setMaximum(self.maxDist)
        self.sig0.setDecimals(4)
        self.sig0.setSingleStep(0.01)
        self.sig0.setValue(
            np.sqrt(
                np.sum(np.square(self.nDists)
                       ) / (2 * self.nDists.shape[0])))

        self.A1 = QDoubleSpinBox()
        self.A1.setMinimum(0)
        self.A1.setMaximum(10)
        self.A1.setDecimals(4)
        self.A1.setSingleStep(0.01)
        self.A1.setValue(0.25)

        self.sig1 = QDoubleSpinBox()
        self.sig1.setMinimum(0)
        self.sig1.setMaximum(self.maxDist)
        self.sig1.setDecimals(4)
        self.sig1.setSingleStep(0.01)
        self.sig1.setValue(100)

        self.loc1 = QDoubleSpinBox()
        self.loc1.setMinimum(0)
        self.loc1.setMaximum(self.maxDist)
        self.loc1.setDecimals(4)
        self.loc1.setSingleStep(0.01)
        self.loc1.setValue(
            np.sqrt(
                np.sum(np.square(self.nDists)
                       ) / (2 * self.nDists.shape[0])))

        self.A2 = QDoubleSpinBox()
        self.A2.setMinimum(0)
        self.A2.setMaximum(10)
        self.A2.setDecimals(4)
        self.A2.setSingleStep(0.01)
        self.A2.setValue(1.0)

        self.sd = QDoubleSpinBox()
        self.sd.setMinimum(0)
        self.sd.setMaximum(10)
        self.sd.setDecimals(6)
        self.sd.setSingleStep(0.00001)
        self.sd.setValue(1e-4)

        self.fitlay.addRow(
            QLabel('Rayleigh Coeff.:'), self.A0)
        self.fitlay.addRow(
            QLabel('Rayleigh Sigma:'), self.sig0)
        self.fitlay.addRow(
            QLabel('Rayleigh Loc:'), QLabel('0.00'))
        self.fitlay.addRow(
            QLabel('Gaussian Coeff.:'), self.A1)
        self.fitlay.addRow(
            QLabel('Gaussian Sigma:'), self.sig1)
        self.fitlay.addRow(
            QLabel('Gaussian Loc:'), self.loc1)
        self.fitlay.addRow(
            QLabel('Linear Coeff.:'), self.A2)
        self.fitlay.addRow(
            QLabel('MLE SD:'), self.sd)

        self.res_guess = QCheckBox('Result as initial guess.')
        self.fitlay.addWidget(self.res_guess)

        self.fit_btn = QPushButton('Fit', clicked=self.fit_data)
        self.fitlay.addWidget(self.fit_btn)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.log.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.fitlay.addRow(self.log)

        greenP = pg.mkPen(color='g')
        redP = pg.mkPen(color='r')
        yellowP = pg.mkPen(color='y')
        whiteP = pg.mkPen(color='w')
        label_style = {'color': '#EEE', 'font-size': '10pt'}

        legend = self.histogram.addLegend()
        legend.anchor(itemPos=(1, 0), parentPos=(1, 0))
        self.histogram.setLabel('bottom', 'Distance [nm]', **label_style)
        self.histogram.setLabel('left', 'PDF', **label_style)
        _bins = np.linspace(0, self.maxDist, self.bins.value())
        self._hist_ref = self.histogram.plot(
            _bins, np.zeros(_bins.shape[0]-1), stepMode='center',
            fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150),
            name='NN Dinstances')
        self._fit_ref = self.histogram.plot(
            _bins, np.ones_like(_bins), pen=greenP, name='NeNA fit')
        self._ray_ref = self.histogram.plot(
            _bins, np.ones_like(_bins), pen=redP, name='Rayleigh')
        self._gauss_ref = self.histogram.plot(
            _bins, np.ones_like(_bins), pen=yellowP, name='Gauss')
        self._line_ref = self.histogram.plot(
            _bins, np.ones_like(_bins), pen=whiteP, name='Linear')

    def fit_data(self):
        self.log.appendPlainText(
            QDateTime.currentDateTime().toString(
                '>>> yyyy/MM/dd hh:mm:ss') + ' NeNA fit started \n'
        )

        if self.res is None or not self.res_guess.isChecked():
            res, (edges, n, x) = NeNA_resolution_estimate(
                self.nDists, self.trackIDs, bins=self.bins.value(),
                a_ray=self.A0.value(), sig_ray=self.sig0.value(),
                a_gauss=self.A1.value(), sig_gauss=self.sig1.value(),
                xc_gauss=self.loc1.value(), a_lin=self.A2.value(),
                sd_fit=self.sd.value())
        else:
            res, (edges, n, x) = NeNA_resolution_estimate(
                self.nDists, self.trackIDs, bins=self.bins.value(),
                a_ray=self.res.x[0], sig_ray=self.res.x[4],
                a_gauss=self.res.x[1], sig_gauss=self.res.x[5],
                xc_gauss=self.res.x[3], a_lin=self.res.x[2],
                sd_fit=self.res.x[6])

        self.res = res

        self._hist_ref.setData(edges, n)
        self._fit_ref.setData(edges, NeNA_model(edges, res.x[:7]))
        self._ray_ref.setData(
            edges, res.x[0] * Rayleigh_dist(edges, 0, res.x[4]))
        self._gauss_ref.setData(
            edges, res.x[1] * Gaussian_dist(edges, res.x[3], res.x[5]))
        self._line_ref.setData(edges, res.x[2] * edges)

        self.log.appendPlainText(
            '    NeNA resolution estimate \n')
        self.log.appendPlainText(
            '    A0 {:.5f} Loc0 {:.5f} Sig0 {:.5f}\n'.format(
                res.x[0], 0, res.x[4]))
        self.log.appendPlainText(
            '    A1 {:.5f} Loc1 {:.5f} Sig1 {:.5f}\n'.format(
                res.x[1], res.x[3], res.x[5]))
        self.log.appendPlainText(
            '    A2 {:e} SD {:e}\n'.format(res.x[2], res.x[6]))
