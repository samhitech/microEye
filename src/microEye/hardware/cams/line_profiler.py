import numpy as np
import pyqtgraph as pg
import qdarkstyle
from pyqtgraph.multiprocess import proxy

from microEye.analysis.tools.kymograms import get_kymogram_row
from microEye.qt import QApplication, QtWidgets, Signal, Slot


class LineProfiler(QtWidgets.QWidget):
    imageUpdate = Signal(np.ndarray)

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle('Line Profiler')

        self.lineProfiles = []
        self.image = np.random.normal(50, 2, (512, 512))

        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)

        params = QtWidgets.QHBoxLayout()

        self.average = QtWidgets.QSpinBox()
        self.average.setMinimum(1)
        self.average.setMaximum(128)
        self.average.setValue(1)

        # params.addWidget(
        #     QLabel('Average [frames]'))
        # params.addWidget(
        #     self.average)
        self.data = np.random.normal(size=(512, 512))
        self.remote_view = pg.GraphicsView()
        pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')
        self.remote_plt = pg.ViewBox(invertY=True)
        self.remote_view.setCentralItem(self.remote_plt)
        self.remote_plt.setAspectLocked()
        self.remote_img = pg.ImageItem(axisOrder='row-major')
        self.remote_img.setImage(self.data)

        self.roi = pg.ROI([10, 10], [0, 128], angle=0, pen='r')
        self.roi.addTranslateHandle([0, 0], [0, 1])
        self.roi.addScaleRotateHandle([0, 1], [0, 0])
        self.roi.setZValue(10)

        self.X, self.Y = self.getRoiCoords()
        self.line_width = 16

        self.remote_plt.addItem(self.remote_img)
        self.remote_plt.addItem(self.roi)

        self.line = pg.GraphicsView()
        self.line.setMaximumHeight(250)
        self.lineP = pg.PlotItem()
        self.line.setCentralItem(self.lineP)
        self.lineP_ref = self.lineP.plot()

        def updateRoiCoords(cls=None):
            self.X, self.Y = self.getRoiCoords()

        self.roi.sigRegionChangeFinished.connect(updateRoiCoords)

        self.setMinimumHeight(700)
        layout.addRow(params)
        layout.addRow(self.remote_view)
        layout.addRow(self.line)

        def setData(data: np.ndarray):
            # if self.data.shape == data.shape:
            #     self.data[:, :] = data
            # else:
            #     self.data = data.copy()
            self.remote_img.setImage(data)

            line_roi = get_kymogram_row(
                data, self.X, self.Y, self.line_width)
            self.lineP_ref.setData(line_roi)

        self.imageUpdate.connect(setData)

    def getRoiCoords(self):
        dy = self.roi.size()[1] * np.cos(np.pi * self.roi.angle() / 180)
        dx = self.roi.size()[1] * np.sin(np.pi * self.roi.angle() / 180)
        x = self.roi.x()
        y = self.roi.y()
        return np.array([x, x - dx]), np.array([y, y + dy])


if __name__ == '__main__':
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    win = LineProfiler()
    win.show()

    app.exec()
