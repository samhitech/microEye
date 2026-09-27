import numpy as np
import pyqtgraph as pg

from microEye.qt import (
    QtCore,
    QtWidgets,
)


class NavigatorOverlay(QtWidgets.QWidget):
    '''Fixed-size overview thumbnail with a rectangle marking the
    current viewport, pinned to a corner regardless of main zoom/pan.'''

    def __init__(self, host: QtWidgets.QWidget, size: int = 140):
        super().__init__(host)
        self.setFixedSize(size, size)

        self._plot = pg.GraphicsLayoutWidget(self)
        self._plot.setFixedSize(size, size)
        self._plot.setStyleSheet('background: transparent;')
        self._mini_vb = self._plot.ci.addViewBox(row=0, col=0)
        self._mini_vb.setMouseEnabled(False, False)
        self._mini_vb.setMenuEnabled(False)
        self._mini_vb.invertY(True)
        self._mini_vb.setAspectLocked(True)

        self._overview_item = pg.ImageItem()
        self._mini_vb.addItem(self._overview_item)

        self._view_rect = QtWidgets.QGraphicsRectItem()
        self._view_rect.setPen(pg.mkPen('y', width=2))
        self._view_rect.setBrush(pg.mkBrush(255, 255, 0, 40))
        self._mini_vb.addItem(self._view_rect)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)
        self.hide()  # nothing to show until the first overview render lands

    def set_overview(self, image: np.ndarray, rect: tuple):
        self._overview_item.setImage(image, autoLevels=True)
        self._overview_item.setRect(QtCore.QRectF(*rect))
        self._mini_vb.setRange(QtCore.QRectF(*rect), padding=0)
        self.show()

    def update_viewport(self, view_range):
        (x0, x1), (y0, y1) = view_range
        self._view_rect.setRect(QtCore.QRectF(x0, y0, x1 - x0, y1 - y0))
