from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QWidget,
)


class LabelledSlider(QWidget):
    '''
    A custom widget representing a labeled slider with optional range labels.

    Attributes:
        finished (pyqtSignal): Signal emitted when editing is finished.
        valueChanged (pyqtSignal): Signal emitted when value is changed.
    '''
    finished = pyqtSignal()
    valueChanged = pyqtSignal(int)

    def __init__(
            self, parent=None,
            show_range_labels=True, vrange=(0, 100), offset=1):
        '''
        Initialize the LabelledSlider widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None.
        show_range_labels : bool, optional
            Whether to show range labels, by default True.
        '''
        super().__init__(parent)
        self._offset = offset
        self.initUI(vrange)
        self.showRangeLabels(show_range_labels)

    def initUI(self, vrange: tuple[int, int]):
        '''
        Initialize the UI components of the widget.
        '''
        self.slider = QSlider(tickPosition=QSlider.TicksLeft, orientation=Qt.Horizontal)
        self.slider.setMaximum(0)

        self.label = QSpinBox()
        self.label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.label.valueChanged.connect(self.changeValue)
        self.slider.valueChanged.connect(self.changeValue)
        self.slider.sliderReleased.connect(
            lambda: self.finished.emit())
        self.label.editingFinished.connect(
            lambda: self.finished.emit())

        self.label_minimum = QLabel()
        self.label_maximum = QLabel()

        self.slider.rangeChanged.connect(
            lambda min, max: self.label_minimum.setNum(vrange[0]+self._offset))
        self.slider.rangeChanged.connect(
            lambda min, max: self.label_maximum.setNum(vrange[1]+self._offset))

        slider_hbox = QHBoxLayout(self)
        slider_hbox.setContentsMargins(0, 0, 0, 0)
        slider_hbox.setSpacing(0)

        slider_hbox.addWidget(self.label_minimum)
        slider_hbox.addWidget(self.slider)
        slider_hbox.addWidget(self.label_maximum)
        slider_hbox.addWidget(self.label)

        self.setRange(*vrange)

    def setRange(self, min: int, max: int):
        '''
        Set the range for the slider.

        Parameters
        ----------
        min : int
            Minimum value of the range.
        max : int
            Maximum value of the range.
        '''
        self.label.setMinimum(min + self._offset)
        self.label.setMaximum(max + self._offset)
        self.slider.setMinimum(min)
        self.slider.setMaximum(max)

    def changeValue(self, value):
        '''
        Change the value of the slider.

        Parameters
        ----------
        value : int
            New value of the slider.
        '''
        self.label.blockSignals(True)
        self.slider.blockSignals(True)

        if self.sender() is self.slider:
            self.label.setValue(value + self._offset)
        else:
            self.slider.setValue(value - self._offset)

        self.label.blockSignals(False)
        self.slider.blockSignals(False)

        self.valueChanged.emit(value)

    def showRangeLabels(self, show=True):
        '''
        Show or hide the range labels.

        Parameters
        ----------
        show : bool, optional
            Whether to show range labels, by default True.
        '''
        self.label_minimum.setVisible(show)
        self.label_maximum.setVisible(show)

    def value(self):
        return self.slider.value()

    def maximum(self):
        return self.slider.maximum()

    def minimum(self):
        return self.slider.minimum()

    def setValue(self, value: int):
        return self.slider.setValue(value)
