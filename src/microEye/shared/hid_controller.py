import sys
from enum import Enum
from typing import Optional

import hid
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from pyqtgraph.parametertree import Parameter

from .parameter_tree import Tree


class Buttons(Enum):
    '''
    Enum representing controller buttons.

    Each enum member has a unique value representing its state in the report,
    along with additional information like identifier, description, etc.

    Attributes
    ----------
    A : Buttons
        Represents the X/A button on the controller.
    B : Buttons
        Represents the O/B button on the controller.
    X : Buttons
        Represents the Sq/X button on the controller.
    Y : Buttons
        Represents the Tri/Y button on the controller.
    L1 : Buttons
        Represents the L1 button on the controller.
    R1 : Buttons
        Represents the R1 button on the controller.
    Share : Buttons
        Represents the Share button on the controller.
    Options : Buttons
        Represents the Options button on the controller.
    L3 : Buttons
        Represents the L3 button on the controller.
    R3 : Buttons
        Represents the R3 button on the controller.
    UP : Buttons
        Represents the D-Pad UP button on the controller.
    RIGHT : Buttons
        Represents the D-Pad RIGHT button on the controller.
    DOWN : Buttons
        Represents the D-Pad DOWN button on the controller.
    LEFT : Buttons
        Represents the D-Pad LEFT button on the controller.

    Methods
    -------
    __str__()
        String representation of the button.
    from_string(s)
        Get an enum member based on its string representation.
    from_value(ch, id)
        Get an enum member based on its channel and identifier.
    strings()
        Get an array of string representations of all enum members.
    '''
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
        '''String representation of the button.'''
        return self.value[-1]

    @classmethod
    def from_string(cls, s):
        '''Get an enum member based on its string representation.'''
        for column in cls:
            if column.value[-1] == s:
                return column
        raise ValueError(cls.__name__ + ' has no value matching "' + s + '"')

    @classmethod
    def from_value(cls, ch, id):
        '''Get an enum member based on its channel and identifier.

        Parameters
        ----------
        ch : int
            The channel of the button.
        id : int
            The identifier of the button.

        Returns
        -------
        Buttons
            The enum member representing the button with the
            given channel and identifier.

        Raises
        ------
        ValueError
            If no enum member is found with the given channel and identifier.
        '''
        for column in cls:
            if column.value[0] == id and column.value[1] == ch:
                return column
        return None

    @classmethod
    def strings(cls):
        '''Get an array of string representations of all enum members.'''
        res = []
        for column in cls:
            res.append(column.value[-1])
        return np.array(res)

class hidParams(Enum):
    '''
    Enum class defining hidController parameters.
    '''
    TITLE = 'HID Controller'
    DEVICE = 'Device'
    REFRESH = 'Refresh'
    OPEN = 'Open'
    CLOSE = 'Close'

    REMOVE = 'Remove Device'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')

class hidDevice:
    def __init__(self, device: dict) -> None:
        '''
        Initialize a new hidDevice object.

        Parameters
        ----------
        device : dict
            A dictionary containing information about the HID device.
        '''
        self.data = device

    @property
    def vendorID(self):
        '''
        Get the vendor ID of the HID device.

        Returns
        -------
        int or None
            The vendor ID of the HID device, or None if it is not available.
        '''
        return self.data.get('vendor_id', None)

    @property
    def productID(self):
        '''
        Get the product ID of the HID device.

        Returns
        -------
        int or None
            The product ID of the HID device, or None if it is not available.
        '''
        return self.data.get('product_id', None)

    def getHID(self):
        '''
        Get a hid.device object for the HID device.

        Returns
        -------
        hid.device
            A hid.device object for the HID device.
        '''
        hid_device = hid.device()
        hid_device.open(self.vendorID, self.productID)
        hid_device.set_nonblocking(True)
        return hid_device

    def __str__(self) -> str:
        '''
        Get a string representation of the HID device.

        Returns
        -------
        str
            A string representation of the HID device.
        '''
        return self.data.get('product_string', 'N/A')

class hidController(Tree):
    '''QWidget for handling HID controller input.'''
    # Define signals with docstrings
    reportEvent = pyqtSignal(Buttons)
    '''Signal emitted when a controller button event occurs.

    Parameters
    ----------
    Buttons: :class:`Buttons`
        The enum member representing the button.
    '''

    reportLStickPosition = pyqtSignal(int, int)
    '''Signal emitted when the left stick position changes.

    Parameters
    ----------
    int: int
        X-axis position of the left stick.
    int: int
        Y-axis position of the left stick.
    '''

    reportRStickPosition = pyqtSignal(int, int)
    '''Signal emitted when the right stick position changes.

    Parameters
    ----------
    int: int
        X-axis position of the right stick.
    int: int
        Y-axis position of the right stick.
    '''

    def __init__(self, parent: Optional['QWidget'] = None):
        '''Initialize the HID controller widget.'''
        super().__init__(parent=parent)

        self.refresh_list()

        self.hid_device = None

        self.last_btn = None
        self.last_aux = None
        self.left_analog = (128, 127)
        self.right_analog = (128, 127)

        self.timer = QTimer()
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
                'values': []},
            {'name': str(hidParams.REFRESH), 'type': 'action'},
            {'name': str(hidParams.OPEN), 'type': 'action'},
            {'name': str(hidParams.CLOSE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='root', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

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

def map_range(value, args):
    '''Map a value from one range to another.

    This function maps a value from one range to another by linearly
    interpolating between the two ranges. It takes a value and a tuple
    of four arguments: the old minimum, old maximum, new minimum, and
    new maximum.

    Parameters
    ----------
    value : float
        The value to be mapped.
    args : tuple
        A tuple of four arguments: old_min, old_max, new_min, new_max.

    Returns
    -------
    float
        The mapped value.

    Examples
    --------
    >>> map_range(0.5, (0, 1, 0, 10))
    5.0
    '''
    old_min, old_max, new_min, new_max = args
    return (new_min +
            (new_max - new_min) * (value - old_min) / (old_max - old_min))


def dz_scaled_radial(stick_input, deadzone):
    '''Apply a scaled radial transformation with deadzone.

    This function applies a scaled radial transformation with a deadzone to a
    stick input, which is a 2D vector. The transformation scales the input vector
    based on its magnitude and applies a deadzone, where inputs within the
    deadzone are mapped to zero.

    Parameters
    ----------
    stick_input : numpy.ndarray
        A 2D vector representing the stick input.
    deadzone : float
        The radius of the deadzone.

    Returns
    -------
    numpy.ndarray
        A 2D vector representing the transformed stick input.

    Examples
    --------
    >>> dz_scaled_radial(np.array([1.0, 0.0]), 0.5)
    array([0.5, 0.  ])
    '''
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
    '''Apply a sloped scaled axial transformation with deadzone.

    This function applies a sloped scaled axial transformation with a deadzone to a
    stick input, which is a 2D vector. The transformation scales the input vector
    based on its magnitude and applies a deadzone, where inputs within the
    deadzone are mapped to zero. The scaling is sloped, meaning that the scaling
    factor depends on the direction of the input vector.

    Parameters
    ----------
    stick_input : numpy.ndarray
        A 2D vector representing the stick input.
    deadzone : float
        The radius of the deadzone.
    n : float, optional
        The slope of the scaling factor. Default is 1.

    Returns
    -------
    numpy.ndarray
        A 2D vector representing the transformed stick input.

    Examples
    --------
    >>> dz_sloped_scaled_axial(np.array([1.0, 0.0]), 0.5)
    array([0.5, 0.  ])

    '''
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
    '''Apply a hybrid transformation with deadzone.

    This function applies a hybrid transformation with a deadzone to a stick input,
    which is a 2D vector. The transformation first checks if the input falls within
    the deadzone, and if so, maps it to zero. Otherwise, it applies a scaled radial
    transformation to the input.

    Parameters
    ----------
    stick_input : numpy.ndarray
        A 2D vector representing the stick input.
    deadzone : float
        The radius of the deadzone.

    Returns
    -------
    numpy.ndarray
        A 2D vector representing the transformed stick input.

    Examples
    --------
    >>> dz_hybrid(np.array([1.0, 0.0]), 0.5)
    array([0.5, 0.  ])
    '''
    # First, check that input does not fall within deadzone
    input_magnitude = np.linalg.norm(stick_input)
    if input_magnitude < deadzone:
        return 0, 0

    # Then apply a scaled_radial transformation
    partial_output = dz_scaled_radial(stick_input, deadzone)

    return partial_output


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = hidController()
    win.show()

    app.exec_()
