from enum import Enum

import numpy as np


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
