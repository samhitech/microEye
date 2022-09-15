import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg

from .results import FittingMethod


ParametersHeaders = {
    0: ['x', 'y', 'bg', 'I', 'ratio x/y', 'frame'],
    1: ['x', 'y', 'bg', 'I'],
    2: ['x', 'y', 'bg', 'I', 'sigma'],
    4: ['x', 'y', 'bg', 'I', 'sigmax', 'sigmay'],
    5: ['x', 'y', 'bg', 'I', 'z']
}


class resultsStatsWidget(QWidget):

    def __init__(self) -> None:
        super().__init__()

        minHeight = 125

        self._layout = QGridLayout()
        self.setLayout(self._layout)

        self.params_pw = []
        self.params_pl = []
        self.params_lr = []

        self.crlbs_pw = []
        self.crlbs_pl = []
        self.crlbs_lr = []

        for x in range(6):
            param_pw = pg.PlotWidget()
            crlb_pw = pg.PlotWidget()

            param_lr = pg.LinearRegionItem(
                [1, 6], bounds=[0, 100], movable=True)
            crlb_lr = pg.LinearRegionItem(
                [1, 6], bounds=[0, 100], movable=True)

            param_pw.addItem(param_lr)
            crlb_pw.addItem(crlb_lr)

            param_pw.setLabel('left', 'Counts', units='')
            param_pw.setMinimumHeight(minHeight)
            crlb_pw.setLabel('left', 'Counts', units='')
            crlb_pw.setMinimumHeight(minHeight)

            self.params_pw.append(param_pw)
            self.crlbs_pw.append(crlb_pw)

            self.params_pl.append(param_pw.plot())
            self.crlbs_pl.append(crlb_pw.plot())

            self.params_lr.append(param_lr)
            self.crlbs_lr.append(crlb_lr)

            self._layout.addWidget(param_pw, x, 0)
            self._layout.addWidget(crlb_pw, x, 1)

        self.log_like_pw = pg.PlotWidget()
        self.iter_pw = pg.PlotWidget()
        self.frames_pw = pg.PlotWidget()

        self.log_like_pl = self.log_like_pw.plot()
        self.iter_pl = self.iter_pw.plot()
        self.frames_pl = self.frames_pw.plot()

        self.log_like_lr = pg.LinearRegionItem(
            [1, 6], bounds=[0, 100], movable=True)
        self.iter_lr = pg.LinearRegionItem(
            [1, 6], bounds=[0, 100], movable=True)
        self.frames_lr = pg.LinearRegionItem(
            [1, 6], bounds=[0, 100], movable=True)

        self.log_like_pw.addItem(self.log_like_lr)
        self.iter_pw.addItem(self.iter_lr)
        self.frames_pw.addItem(self.frames_lr)

        self.log_like_pw.setMinimumHeight(minHeight)
        self.iter_pw.setMinimumHeight(minHeight)
        self.frames_pw.setMinimumHeight(minHeight)

        self.log_like_pw.setLabel('left', 'Counts', units='')
        self.iter_pw.setLabel('left', 'Counts', units='')
        self.frames_pw.setLabel('left', 'Counts', units='')

        self.log_like_pw.setLabel('bottom', 'Log Likelihood', units='')
        self.iter_pw.setLabel('bottom', 'Fitting Iterations', units='')
        self.frames_pw.setLabel('bottom', 'Frame', units='')

        self._layout.addWidget(self.log_like_pw, 6, 0)
        self._layout.addWidget(self.iter_pw, 6, 1)
        self._layout.addWidget(self.frames_pw, 7, 0)

    def setData(
            self, fittingMethod,
            frames=None, params=None, crlbs=None, loglike=None):
        self.headers = ParametersHeaders[fittingMethod]
        self.fittingMethod = fittingMethod
        self.frames = frames
        self.params = params
        self.crlbs = crlbs
        self.loglike = loglike

        for x in range(6):
            if fittingMethod == FittingMethod._2D_Phasor_CPU:
                self.crlbs_pw[x].setVisible(False)
            else:
                if x >= len(self.headers):
                    self.crlbs_pw[x].setVisible(False)
                else:
                    self.crlbs_pw[x].setVisible(True)
                    self.crlbs_pw[x].setLabel(
                        'bottom', 'CRLB ' + self.headers[x], units='')
            if x >= len(self.headers):
                self.params_pw[x].setVisible(False)
            else:
                self.params_pw[x].setVisible(True)
                self.params_pw[x].setLabel('bottom', self.headers[x], units='')

        if fittingMethod == FittingMethod._2D_Phasor_CPU:
            self.log_like_pw.setVisible(False)
            self.iter_pw.setVisible(False)
        else:
            self.log_like_pw.setVisible(True)
            self.iter_pw.setVisible(True)

    def update(self):
        if self.fittingMethod == FittingMethod._2D_Phasor_CPU:
            if self.frames is None:
                if self.frames_pl is not None:
                    self.frames_pl.setData(None, None)
            else:
                hist, bins = np.histogram(self.frames)
                self.frames_lr.setBounds([np.min(bins), np.max(bins)])
                if self.frames_pl is not None:
                    self.frames_pl.setData(x=bins, y=hist)
                else:
                    self.frames_pl = self.frames_pw.plot(
                        bins, hist, stepMode="center",
                        fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))

            if self.params is None:
                for x in range(6):
                    if x < len(self.headers):
                        hist, bins = np.histogram(self.params[:, x])
                        self.params_lr[x].setBounds(
                            [np.min(bins), np.max(bins)])
                        if self.params_lr[x] is not None:
                            self.params_lr[x].setData(x=bins, y=hist)
                        else:
                            self.params_lr[x] = self.params_pw[x].plot(
                                bins, hist, stepMode="center",
                                fillLevel=0, fillOutline=True,
                                brush=(0, 0, 255, 150))

        elif self.fittingMethod == FittingMethod._2D_Gauss_MLE_fixed_sigma:
            pass
        elif self.fittingMethod == FittingMethod._2D_Gauss_MLE_free_sigma:
            pass
        elif self.fittingMethod == \
                FittingMethod._2D_Gauss_MLE_elliptical_sigma:
            pass
        elif self.fittingMethod == FittingMethod._3D_Gauss_MLE_cspline_sigma:
            pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = resultsStatsWidget()
    win.show()
    win.setData(5)

    app.exec_()
