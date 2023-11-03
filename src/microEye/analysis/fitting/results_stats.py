import sys
import typing
import traceback

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

ParametersHeaders = {
    0: ['x', 'y', 'I', 'bg', 'ratio x/y', 'frame'],
    1: ['x', 'y', 'I', 'bg'],
    2: ['x', 'y', 'I', 'bg', 'sigma'],
    4: ['x', 'y', 'I', 'bg', 'sigmax', 'sigmay'],
    5: ['x', 'y', 'I', 'bg', 'z']
}


class resultsStatsWidget(QWidget):
    dataFilterUpdated = pyqtSignal(pd.DataFrame)

    def __init__(self, parent: typing.Optional['QWidget'] = None):
        super().__init__(parent=parent)

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

            pw = pg.PlotWidget(self)
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

            if column == 'trackID':
                data = df[column].to_numpy()
                data = data[data.nonzero()]

                if len(data) > 0:
                    uniq, counts = np.unique(
                        data, return_counts=True)

                    bars = pg.BarGraphItem(
                        x=uniq, height=counts,
                        width=0.9, brush='b')
                    pw.addItem(bars)

                    # pw.plot(
                    #     uniq, counts, stepMode="center",
                    #     fillLevel=0, fillOutline=True,
                    #     brush=(0, 0, 255, 150))
                else:
                    pw.plot(
                        [0], [1],
                        fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
            else:
                try:
                    min_val = np.floor(df[column].min())
                    max_val = np.ceil(df[column].max())
                    counts = df[column].count()
                    hist, bins = np.histogram(
                        df[column].to_numpy(),
                        bins=min(counts+1, 1024))

                    pw.plot(
                        bins, hist / np.max(hist), stepMode="center",
                        fillLevel=0, fillOutline=True, brush=(0, 0, 255, 150))
                except Exception:
                    traceback.print_exc()
                    print(column)

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
            mask = np.logical_and(
                mask,
                np.logical_and(
                    self.df[column].to_numpy() >= Rmin,
                    self.df[column].to_numpy() <= Rmax))

        self.filtered = self.df[mask]

        self.dataFilterUpdated.emit(self.filtered)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = resultsStatsWidget()
    win.show()
    win.setData(5)

    app.exec_()
