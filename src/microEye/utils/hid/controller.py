import sys
from typing import Optional

import hid
import numpy as np
from pyqtgraph.parametertree import Parameter

from microEye.qt import QApplication, QtCore, QtWidgets, Signal
from microEye.utils.hid.device import hidDevice
from microEye.utils.hid.enums import Buttons, hidParams
from microEye.utils.hid.utils import dz_hybrid, dz_scaled_radial, dz_sloped_scaled_axial
from microEye.utils.parameter_tree import Tree


class hidController(Tree):
    '''QWidget for handling HID controller input.'''
    # Define signals with docstrings
    reportEvent = Signal(Buttons)
    '''Signal emitted when a controller button event occurs.

    Parameters
    ----------
    Buttons: :class:`Buttons`
        The enum member representing the button.
    '''

    reportLStickPosition = Signal(int, int)
    '''Signal emitted when the left stick position changes.

    Parameters
    ----------
    int: int
        X-axis position of the left stick.
    int: int
        Y-axis position of the left stick.
    '''

    reportRStickPosition = Signal(int, int)
    '''Signal emitted when the right stick position changes.

    Parameters
    ----------
    int: int
        X-axis position of the right stick.
    int: int
        Y-axis position of the right stick.
    '''
    PARAMS = hidParams

    def __init__(self, parent: Optional['QtWidgets.QWidget'] = None):
        '''Initialize the HID controller widget.'''
        super().__init__(parent=parent)

        self.refresh_list()

        self.hid_device = None

        self.last_btn = None
        self.last_aux = None
        self.left_analog = (128, 127)
        self.right_analog = (128, 127)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(1)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {'name': str(hidParams.TITLE), 'type': 'group',
                'children': []},
            {'name': str(hidParams.DEVICE), 'type': 'list',
                'limits': []},
            {'name': str(hidParams.REFRESH), 'type': 'action'},
            {'name': str(hidParams.OPEN), 'type': 'action'},
            {'name': str(hidParams.CLOSE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.get_param(
            hidParams.REFRESH).sigActivated.connect(self.refresh_list)
        self.get_param(
            hidParams.OPEN).sigActivated.connect(self.open_HID)
        self.get_param(
            hidParams.CLOSE).sigActivated.connect(self.close_HID)

    def recurring_timer(self):
        '''Read and process controller input in a recurring timer.'''
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
        '''Close the HID device.'''
        if self.hid_device is not None:
            self.hid_device.close()
            self.hid_device = None

    def open_HID(self):
        '''Open the selected HID device.'''
        device: hidDevice = self.get_param_value(hidParams.DEVICE)
        if device is not None:
            if self.hid_device is not None:
                self.hid_device.close()
                self.hid_device = None

            self.hid_device = device.getHID()

    def refresh_list(self):
        '''Refresh the list of available HID devices.'''
        devicesParam = self.get_param(hidParams.DEVICE)

        devices = [hidDevice(device) for device in hid.enumerate()]

        devicesParam.setLimits(devices)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = hidController()
    win.show()

    app.exec()
