import pyqtgraph as pg
import qdarkstyle
import numpy as np
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtSerialPort import *
from PyQt5.QtGui import *
# from ..uImage import uImage


class LineProfiler(QGroupBox):

    def __init__(self) -> None:
        super().__init__()

        self.setTitle('Line Profiler')

        self.lineProfiles = []

        layout = QFormLayout()
        self.setLayout(layout)

        params = QHBoxLayout()

        self.x_1 = QSpinBox()
        self.x_1.setMinimum(0)
        self.x_1.setMaximum(10**5)
        self.x_1.setValue(0)
        self.x_2 = QSpinBox()
        self.x_2.setMinimum(0)
        self.x_2.setMaximum(10**5)
        self.x_2.setValue(0)

        self.y_1 = QSpinBox()
        self.y_1.setMinimum(0)
        self.y_1.setMaximum(10**5)
        self.y_1.setValue(0)
        self.y_2 = QSpinBox()
        self.y_2.setMinimum(0)
        self.y_2.setMaximum(10**5)
        self.y_2.setValue(256)

        self.lineWidth = QSpinBox()
        self.lineWidth.setMinimum(0)
        self.lineWidth.setMaximum(255)
        self.lineWidth.setValue(1)

        self.average = QSpinBox()
        self.average.setMinimum(1)
        self.average.setMaximum(128)
        self.average.setValue(1)

        self.add = QPushButton('Add', clicked=lambda: self.lineProfiles.append(
            {
                'P1': (self.x_1.value(), self.y_1.value()),
                'P2': (self.x_2.value(), self.y_2.value()),
                'Width': self.lineWidth.value()
            }
        ))

        self.clear = QPushButton(
            'Clear',
            clicked=lambda: self.lineProfiles.clear())

        params.addWidget(
            QLabel('X 1'))
        params.addWidget(
            self.x_1)
        params.addWidget(
            QLabel('Y 1'))
        params.addWidget(
            self.y_1)
        params.addWidget(
            QLabel('X 2'))
        params.addWidget(
            self.x_2)
        params.addWidget(
            QLabel('Y 2'))
        params.addWidget(
            self.y_2)
        params.addWidget(
            QLabel('Line Width'))
        params.addWidget(
            self.lineWidth)
        params.addWidget(
            QLabel('Average [frames]'))
        params.addWidget(
            self.average)

        self.plot = pg.PlotWidget()
        greenP = pg.mkPen(color='g')
        self.plot_ref = self.plot.plot(
            np.zeros((256)), np.zeros((256)), pen=greenP)

        layout.addRow(params)
        layout.addRow(self.plot)

    def save_browse_clicked(self):
        """Slot for browse clicked event"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory")

        if len(directory) > 0:
            self._directory = directory
            self.save_dir_edit.setText(self._directory)

    def get_params(self):
        return (
                (self.x_1.value(), self.y_1.value()),
                (self.x_2.value(), self.y_2.value()),
                self.lineWidth.value(),
                self.average.value())


if __name__ == '__main__':

    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))

    win = LineProfiler()
    win.show()

    app.exec_()
