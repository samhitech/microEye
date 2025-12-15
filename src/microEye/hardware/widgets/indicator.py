from microEye.qt import QtCore, QtGui, QtWidgets, Signal
from microEye.tools.color import wavelength_to_rgb


class LaserIndicator(QtWidgets.QWidget):
    toggled = Signal(bool)
    _size = 24

    def __init__(
        self,
        wavelength_nm: float,
        on: bool = False,
        read_only: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._wavelength = wavelength_nm
        self._on = on
        self._color = QtGui.QColor(*wavelength_to_rgb(wavelength_nm, max_intensity=255))
        self._read_only = read_only
        if self._read_only:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(self._size * 2, int(self._size * 1.5))

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @property
    def is_on(self) -> bool:
        return self._on

    def set_on(self, value: bool):
        if self._on != value:
            self._on = value
            self.toggled.emit(self._on)
            self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and not self._read_only:
            self.set_on(not self._on)

    def paintEvent(self, _):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(3, 3, -3, -3)

        base = QtGui.QColor(self._color)
        base.setAlpha(255 if self._on else 10)
        painter.setBrush(base)
        pen = QtGui.QPen(base.darker(150) if self._on else QtGui.QColor('#555'))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 4, 4)

        painter.setPen(base.darker(200) if self._on else QtGui.QColor('#FFF'))

        painter.drawText(
            rect,
            QtCore.Qt.AlignmentFlag.AlignCenter,
            f'{int(self._wavelength)}',
        )


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)

    window = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(window)

    for wl in [405, 488, 520, 532, 561, 638]:
        indicator = LaserIndicator(wavelength_nm=wl, on=(wl % 2 == 0))
        layout.addWidget(indicator)

    window.show()
    sys.exit(app.exec())
