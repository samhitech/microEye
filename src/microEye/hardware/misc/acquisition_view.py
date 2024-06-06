import numpy as np
import pyqtgraph as pg
import qdarkstyle

from microEye.qt import QApplication, QtWidgets


class AcquisitionView(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle('Image View (Snap/Live)')

        pg.setConfigOption('imageAxisOrder', 'row-major')

        size = QApplication.primaryScreen().availableGeometry()
        minDim = int(min(size.width(), size.height()) * 0.9)
        self.setGeometry(
            (size.width() - minDim) // 2, (size.height() - minDim) // 2, minDim, minDim
        )

        layout = QtWidgets.QGridLayout()
        self.setLayout(layout)

        self.view = pg.GraphicsView()
        self.vb = pg.ViewBox()
        self.vb.setAspectLocked()
        self.view.setCentralItem(self.vb)
        layout.addWidget(self.view, 0, 0, 1, 3)

        self.hist = pg.HistogramLUTWidget(
            gradientPosition='top', orientation='horizontal'
        )
        layout.addWidget(self.hist, 1, 0)

        self.imageItem = pg.ImageItem(AcquisitionView.dummyData())
        self.vb.addItem(self.imageItem)
        self.vb.autoRange()

        self.hist.setImageItem(self.imageItem)

    def setData(self, image: np.ndarray, autoLevels=True):
        self.imageItem.setImage(image, autoLevels)

        if len(image.shape) < 3:
            self.hist.setLevelMode('mono')
        elif image.shape[2] > 1:
            self.hist.setLevelMode('rgba')
        else:
            self.hist.setLevelMode('mono')

    def dummyData():
        x = np.linspace(255, 0, 256)
        y = np.linspace(0, 255, 256)
        z = np.concatenate([x, y])
        return np.tile(z, (512, 1))


if __name__ == '__main__':
    x = np.linspace(255, 0, 256)
    y = np.linspace(0, 255, 256)
    z = np.concatenate([x, y])
    image = np.tile(z, (512, 1)).T

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    win = AcquisitionView()
    win.show()

    app.exec()
