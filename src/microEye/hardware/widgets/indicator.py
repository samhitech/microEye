from microEye.qt import QtCore, QtGui, QtWidgets, Signal
from microEye.tools.color import wavelength_to_rgb


class LaserIndicator(QtWidgets.QWidget):
    toggleSignal = Signal(bool)
    _size = 24

    def __init__(
        self,
        wavelength_nm: float,
        on: bool = False,
        read_only: bool = False,
        label: str = None,
        parent=None,
    ):
        super().__init__(parent)
        self._wavelength = wavelength_nm
        self._on = on
        self._color = QtGui.QColor(*wavelength_to_rgb(wavelength_nm, max_intensity=255))
        self._read_only = read_only
        self._label = label or f'{int(wavelength_nm)}'
        self._sync_cursor()
        self.setFixedSize(self._size * 2, int(self._size * 1.5))

    def _sync_cursor(self):
        self.setCursor(
            QtCore.Qt.CursorShape.ArrowCursor
            if self._read_only
            else QtCore.Qt.CursorShape.PointingHandCursor
        )

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @property
    def is_on(self) -> bool:
        return self._on

    def set_on(self, value: bool, *, emit: bool = True):
        value = bool(value)
        if self._on != value:
            self._on = value
            if emit:
                self.toggleSignal.emit(self._on)
            self.update()

    def set_label(self, label: str | None):
        label = label or f'{int(self._wavelength)}'
        if self._label != label:
            self._label = label
            self.update()

    def set_read_only(self, read_only: bool):
        read_only = bool(read_only)
        if self._read_only != read_only:
            self._read_only = read_only
            self._sync_cursor()

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
            self._label,
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
