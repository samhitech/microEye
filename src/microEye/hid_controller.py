import sys

from enum import Enum
import hid
import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


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


class hid_controller(QWidget):
    reportEvent = pyqtSignal(Buttons)
    reportLStickPosition = pyqtSignal(int, int)
    reportRStickPosition = pyqtSignal(int, int)

    def __init__(self) -> None:
        super().__init__()

        self._layout = QFormLayout()
        self.setLayout(self._layout)

        self.hid_device = None

        self.last_btn = None
        self.last_aux = None
        self.left_analog = (128, 127)
        self.right_analog = (128, 127)

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
        self.close_btn = QPushButton(
            'Close',
            clicked=lambda: self.close_HID()
        )
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.open_btn)
        self.btn_layout.addWidget(self.close_btn)
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
                # print(report)
                if len(report) == 14:
                    res = Buttons.from_value(0, report[-4])
                    if res:
                        if res != self.last_btn:
                            self.reportEvent.emit(res)
                        self.last_btn = res
                    else:
                        self.last_btn = None
                    res = Buttons.from_value(1, report[-3])
                    if res:
                        if res != self.last_aux:
                            self.reportEvent.emit(res)
                        self.last_aux = res
                    else:
                        self.last_aux = None

                    self.left_analog = report[1], report[3]
                    self.right_analog = report[5], report[7]
                    self.reportLStickPosition.emit(
                        *self.left_analog)
                    self.reportRStickPosition.emit(
                        *self.right_analog)

    def close_HID(self):
        if self.hid_device is not None:
            self.hid_device.close()
            self.hid_device = None

    def open_HID(self):
        data = self.devices_cbox.currentData()
        if data is not None:
            if self.hid_device is not None:
                self.hid_device.close()
                self.hid_device = None

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


def map_range(value, args):
    old_min, old_max, new_min, new_max = args
    return (new_min +
            (new_max - new_min) * (value - old_min) / (old_max - old_min))


def dz_scaled_radial(stick_input, deadzone):
    input_magnitude = np.linalg.norm(stick_input)
    if input_magnitude < deadzone:
        return 0, 0
    else:
        input_normalized = stick_input / input_magnitude
        # Formula:
        # max_value = 1
        # min_value = 0
        # retval = input_normalized *
        # (min_value + (max_value - min_value) * ((input_magnitude - deadzone)
        #  / (max_value - deadzone)))
        retval = input_normalized * map_range(
            input_magnitude, (deadzone, 1, 0, 1))
        return retval[0], retval[1]


def dz_sloped_scaled_axial(stick_input, deadzone, n=1):
    x_val = 0
    y_val = 0
    deadzone_x = deadzone * np.power(abs(stick_input[1]), n)
    deadzone_y = deadzone * np.power(abs(stick_input[0]), n)
    sign = np.sign(stick_input)
    if abs(stick_input[0]) > deadzone_x:
        x_val = sign[0] * map_range(abs(stick_input[0]), (deadzone_x, 1, 0, 1))
    if abs(stick_input[1]) > deadzone_y:
        y_val = sign[1] * map_range(abs(stick_input[1]), (deadzone_y, 1, 0, 1))
    return x_val, y_val


def dz_hybrid(stick_input, deadzone):
    # First, check that input does not fall within deadzone
    input_magnitude = np.linalg.norm(stick_input)
    if input_magnitude < deadzone:
        return 0, 0

    # Then apply a scaled_radial transformation
    partial_output = dz_scaled_radial(stick_input, deadzone)

    return partial_output


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = hid_controller()
    win.show()

    app.exec_()
