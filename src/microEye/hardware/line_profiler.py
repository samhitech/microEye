import pyqtgraph as pg
import qdarkstyle
import numpy as np
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtSerialPort import *
from PyQt5.QtGui import *
# from ..uImage import uImage


class LineProfiler(QWidget):

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle('Line Profiler')

        self.lineProfiles = []
        self.image = np.random.normal(50, 2, (512, 512))

        layout = QFormLayout()
        self.setLayout(layout)

        params = QHBoxLayout()

        self.average = QSpinBox()
        self.average.setMinimum(1)
        self.average.setMaximum(128)
        self.average.setValue(1)

        # params.addWidget(
        #     QLabel('Average [frames]'))
        # params.addWidget(
        #     self.average)

        self.data = np.random.normal(
            size=(512, 512))
        self.remote_view = pg.RemoteGraphicsView()
        self.remote_view.pg.setConfigOptions(
            antialias=True, imageAxisOrder='row-major')
        self.remote_plt = self.remote_view.pg.ViewBox(invertY=True)
        self.remote_plt._setProxyOptions(deferGetattr=True)
        self.remote_view.setCentralItem(self.remote_plt)
        self.remote_plt.setAspectLocked()
        self.remote_img = self.remote_view.pg.ImageItem(axisOrder='row-major')
        self.remote_img.setImage(
            self.data, _callSync='off')

        self.roi = self.remote_view.pg.ROI(
            [10, 10], [128, 20],
            angle=0, pen='r')
        self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.roi.addRotateHandle([1, 0.5], [0.5, 0.5])
        self.roi.setZValue(10)

        self.remote_plt.addItem(self.remote_img)
        self.remote_plt.addItem(self.roi)

        self.line = pg.RemoteGraphicsView()
        self.line.setMaximumHeight(250)
        self.lineP = self.line.pg.PlotItem()
        self.lineP._setProxyOptions(deferGetattr=True)
        self.line.setCentralItem(self.lineP)
        self.lineP_ref = self.lineP.plot()

        def updatePlot(cls=None):
            selected, _ = self.roi.getArrayRegion(
                self.data, self.remote_img, returnMappedCoords=True)
            self.lineP_ref.setData(selected.mean(axis=0), _callSync='off')

        self.lr_proxy = pg.multiprocess.proxy(
            updatePlot, callSync='off', autoProxy=True)
        self.roi.sigRegionChangeFinished.connect(self.lr_proxy)
        self.remote_img.sigImageChanged.connect(self.lr_proxy)

        self.setMinimumHeight(700)
        layout.addRow(params)
        layout.addRow(self.remote_view)
        layout.addRow(self.line)

    def setData(self, data: np.ndarray):
        if self.data.shape == data.shape:
            self.data[:, :] = data
        else:
            self.data = data.copy()
        self.remote_img.setImage(
            self.data, _callSync='off')


if __name__ == '__main__':

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    win = LineProfiler()
    win.show()

    app.exec_()
