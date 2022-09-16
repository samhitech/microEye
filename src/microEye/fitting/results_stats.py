import sys
import numpy as np
import pandas as pd
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
    dataFilterUpdated = pyqtSignal(pd.DataFrame)

    def __init__(self) -> None:
        super().__init__()

        minHeight = 125

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.plot_widgets = []
        self.plot_lr = []

    def setData(
            self, df: pd.DataFrame):
        self.plot_widgets = []
        self.plot_lr: list[pg.LinearRegionItem] = []
        self.clearLayout()

        self.df = df

        for idx, column in enumerate(df.columns):
            # row = idx // 2
            # col = idx % 2

            pw = pg.PlotWidget()
            pw.setMinimumHeight(125)
            pw.setLabel('left', 'Counts', units='')
            pw.setLabel('bottom', column, units='')

            bounds = [df[column].min(), df[column].max()]
            lr = pg.LinearRegionItem(
                bounds,
                bounds=bounds, movable=True)
            lr.sigRegionChangeFinished.connect(self.update)
            pw.addItem(lr)

            self.plot_widgets.append(pw)
            self.plot_lr.append(lr)

            self._layout.addWidget(pw)
            # self._layout.addWidget(pw, row, col)

            hist, bins = np.histogram(df[column].to_numpy())

            pw.plot(
                bins, hist, stepMode="center",
                fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))

    def clearLayout(self):
        for i in reversed(range(self._layout.count())):
            widgetToRemove = self._layout.itemAt(i).widget()
            # remove it from the layout list
            self._layout.removeWidget(widgetToRemove)
            # remove it from the gui
            widgetToRemove.setParent(None)

    def update(self):
        mask = np.ones(self.df.count()[0], dtype=bool)
        for idx, column in enumerate(self.df.columns):
            Rmin, Rmax = self.plot_lr[idx].getRegion()
            mask &= np.logical_and(
                self.df[column].to_numpy() >= Rmin,
                self.df[column].to_numpy() <= Rmax)

        self.filtered = self.df[mask]

        self.dataFilterUpdated.emit(self.filtered)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = resultsStatsWidget()
    win.show()
    win.setData(5)

    app.exec_()
