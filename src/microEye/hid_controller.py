import sys

from enum import Enum
import hid
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class hid_controller(QWidget):

    def __init__(self) -> None:
        super().__init__()

        minHeight = 125

        self._layout = QFormLayout()
        self.setLayout(self._layout)

        self.hid_device = None

        self.devices_cbox = QComboBox()
        self.refresh_list()

        self.refresh_btn = QPushButton(
            'Refresh',
            clicked=lambda: self.refresh_list()
        )
        self.open_btn = QPushButton(
            'Open',
            clicked=lambda: self.open_HID()
        )
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.open_btn)
        self.btn_layout.addWidget(self.refresh_btn)

        self._layout.addRow(
            QLabel('HID Devices'),
            self.devices_cbox
        )
        self._layout.addRow(
            self.btn_layout)

        self.timer = QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def recurring_timer(self):
        if self.hid_device is not None:
            report = self.hid_device.read(64)
            if report:
                if len(report) == 14:
                    print(Buttons.from_value(0, report[-4]))
                    print(Buttons.from_value(1, report[-3]))

    def open_HID(self):
        data = self.devices_cbox.currentData()
        if data is not None:
            if self.hid_device is not None:
                self.hid_device.close()
            else:
                self.hid_device = hid.device()
            self.hid_device.open(data[0], data[1])
            self.hid_device.set_nonblocking(True)

    def refresh_list(self):
        self.devices_cbox.clear()
        for device in hid.enumerate():
            data = (
                device['vendor_id'],
                device['product_id'],
                device['product_string'])
            self.devices_cbox.addItem(
                data[2],
                data
            )


class Buttons(Enum):
    A = (1, 0, 'X/A')
    B = (2, 0, 'O/B')
    X = (4, 0, 'Sq/X')
    Y = (8, 0, 'Tri/Y')
    L1 = (16, 0, 'L1')
    R1 = (32, 0, 'R1')
    Share = (64, 0, 'Share')
    Options = (128, 0, 'Options')

    L3 = (1, 1, 'L3')
    R3 = (2, 1, 'R3')
    UP = (4, 1, 'D-Pad UP')
    RIGHT = (12, 1, 'D-Pad RIGHT')
    DOWN = (20, 1, 'D-Pad DOWN')
    LEFT = (28, 1, 'D-Pad LEFT')

    def __str__(self):
        return self.value[-1]

    @classmethod
    def from_string(cls, s):
        for column in cls:
            if column.value[-1] == s:
                return column
        raise ValueError(cls.__name__ + ' has no value matching "' + s + '"')

    @classmethod
    def from_value(cls, ch, id):
        for column in cls:
            if column.value[0] == id and column.value[1] == ch:
                return column
        return None

    @classmethod
    def strings(cls):
        res = []
        for column in cls:
            res.append(column.value[-1])
        return np.array(res)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = hid_controller()
    win.show()

    app.exec_()
