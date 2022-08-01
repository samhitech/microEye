import pyqtgraph as pg
import qdarkstyle
import numpy as np
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtSerialPort import *
from PyQt5.QtGui import *
from ..uImage import uImage


class TileImage:
    def __init__(self, uImage: uImage, index, position) -> None:
        self.uImage = uImage
        self.index = index
        self.position = position


class TiledImageSelector(QWidget):
    positionSelected = pyqtSignal(float, float)

    def __init__(self, images: list[TileImage]) -> None:
        super().__init__()

        self.images = images

        central_layout = QHBoxLayout()
        self.setLayout(central_layout)

        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.ci.setContentsMargins(0, 0, 0, 0)
        imageWidget.ci.setSpacing(0)
        imageWidget.sceneObj.sigMouseClicked.connect(self.clicked)

        for tImg in images:
            vb: pg.ViewBox = imageWidget.addViewBox(*tImg.index)
            vb.setMouseEnabled(False, False)
            vb.setDefaultPadding(0.004)
            vb.setAspectLocked(True)
            vb.invertY()
            menu: QMenu = vb.getMenu(None)
            vb.action = QAction("Save Raw Data (.tif)")
            menu.addAction(vb.action)
            vb.action.triggered.connect(
                lambda: self.save_raw_data(tImg.uImage._image))
            img = pg.ImageItem(tImg.uImage._view.T)
            vb.addItem(img)
            vb.item = tImg

        self.imgView = pg.ImageView()
        self.setWindowTitle('Tiled Image Selector')

        central_layout.addWidget(imageWidget, 3)
        central_layout.addWidget(self.imgView, 4)

    def clicked(self, event):
        if event.double():
            self.positionSelected.emit(*event.currentItem.item.position)
        else:
            self.setWindowTitle(
                'Tiled Image Selector ({}, {}) ({}, {})'.format(
                    *event.currentItem.item.index,
                    *event.currentItem.item.position))
            self.imgView.setImage(event.currentItem.addedItems[0].image)

    def save_raw_data(self, data):
        filename = None
        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Raw Data", filter="Tiff Files (*.tif)")

        if len(filename) > 0:
            tf.imwrite(
                filename,
                data,
                photometric='minisblack')


class ScanAcquisitionWidget(QGroupBox):
    startAcquisition = pyqtSignal(tuple)
    stopAcquisition = pyqtSignal()
    openLastTile = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()

        self.setTitle('Scan Acquisition')

        layout = QFormLayout()
        self.setLayout(layout)

        self.x_steps = QSpinBox()
        self.x_steps.setMinimum(1)
        self.x_steps.setMaximum(20)
        self.x_steps.setValue(4)

        self.y_steps = QSpinBox()
        self.y_steps.setMinimum(1)
        self.y_steps.setMaximum(20)
        self.y_steps.setValue(4)

        self.x_stepsize = QDoubleSpinBox()
        self.x_stepsize.setDecimals(1)
        self.x_stepsize.setMinimum(0.1)
        self.x_stepsize.setMaximum(500)
        self.x_stepsize.setValue(50)

        self.y_stepsize = QDoubleSpinBox()
        self.y_stepsize.setDecimals(1)
        self.y_stepsize.setMinimum(0.1)
        self.y_stepsize.setMaximum(500)
        self.y_stepsize.setValue(50)

        self.delay = QSpinBox()
        self.delay.setMinimum(0)
        self.delay.setMaximum(2000)
        self.delay.setValue(200)

        self.average = QSpinBox()
        self.average.setMinimum(1)
        self.average.setMaximum(32)
        self.average.setValue(1)

        layout.addRow(
            QLabel('Number of X steps'),
            self.x_steps
        )
        layout.addRow(
            QLabel('Number of Y steps'),
            self.y_steps
        )
        layout.addRow(
            QLabel('X Step size [um]'),
            self.x_stepsize
        )
        layout.addRow(
            QLabel('Y Step size [um]'),
            self.y_stepsize
        )
        layout.addRow(
            QLabel('Delay [ms]'),
            self.delay
        )
        layout.addRow(
            QLabel('Average [frames]'),
            self.average
        )

        buttons = QHBoxLayout()

        self.acquire_btn = QPushButton(
            'Acquire',
            clicked=lambda: self.startAcquisition.emit((
                self.x_steps.value(),
                self.y_steps.value(),
                self.x_stepsize.value(),
                self.y_stepsize.value(),
                self.delay.value(),
                self.average.value())
            )
        )
        self.last_btn = QPushButton(
            'Last Image Scan',
            clicked=lambda: self.openLastTile.emit()
        )

        self.stop_btn = QPushButton(
            'STOP!',
            clicked=lambda: self.stopAcquisition.emit()
        )

        buttons.addWidget(self.acquire_btn)
        buttons.addWidget(self.last_btn)
        buttons.addWidget(self.stop_btn)
        layout.addRow(buttons)


if __name__ == '__main__':
    x = np.linspace(255, 0, 256)
    y = np.linspace(0, 255, 256)
    z = np.concatenate([x, y])
    image = np.tile(z, (512, 1)).T

    data = []

    for i in range(10):
        for j in range(10):
            uImg = uImage(image)
            uImg.equalizeLUT()
            tImg = TileImage(uImg, [i, j], [17, 17])
            data.append(tImg)

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    win = TiledImageSelector(data)
    win.positionSelected.connect(lambda x, y: print(x, y))
    win.show()

    app.exec_()
