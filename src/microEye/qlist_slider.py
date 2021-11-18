from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np


class qlist_slider(QSlider):
    '''QList Slider

    Inherits QSlider allows selecting over a range of
    non-integer values using a slider using indecies.

    Parameters
    ----------
    values : list | ndarray
        list or array of values.
    '''
    elementChanged = pyqtSignal([int, int], [int, float])
    '''Element changed pyqtSignal,
    supports (index: int, value: int) or (index: int, value: float) functions.
    '''

    def __init__(self, values=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMinimum(0)
        self._values = []
        self.valueChanged.connect(self._on_value_changed)
        self.values = values or []

    @property
    def values(self):
        '''Gets values property.

        Returns
        -------
        list | ndarray
            list or array of values.
        '''
        return self._values

    @values.setter
    def values(self, values):
        '''Sets values property.

        Parameters
        ----------
        values : list | ndarray
            list or array of values.
        '''
        self._values = values
        maximum = max(0, len(self._values) - 1)
        self.setMaximum(maximum)
        self.setValue(0)

    @pyqtSlot(int)
    def _on_value_changed(self, index):
        '''On value changed emits the element changed signal.

        Parameters
        ----------
        index : int
            the selected element index.
        '''
        value = self.values[index]
        self.elementChanged[int, int].emit(index, value)
        self.elementChanged[int, float].emit(index, value)

    def find_nearest(self, array, value):
        """
        find nearest value in array to the supplied one

        Parameters
        ----------
        array : ndarray
            numpy array searching in
        value : type of ndarray.dtype
            value to find nearest to in the array
        """
        array = np.asarray(array)
        return (np.abs(array - value)).argmin()

    def setNearest(self, value):
        self.setValue(self.find_nearest(
                        self.values,
                        float(value)))


class DragLabel(QLabel):
    def __init__(self, parent=None, parent_name: str = ''):
        super(QLabel, self).__init__(parent)
        self.parent_name = parent_name
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton):
            return
        if (event.pos() - self.drag_start_position).manhattanLength() < \
                QApplication.startDragDistance():
            return
        drag = QDrag(self)
        mimedata = QMimeData()
        mimedata.setText(self.parent_name)
        drag.setMimeData(mimedata)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())
        drag.exec_(Qt.CopyAction | Qt.MoveAction)
