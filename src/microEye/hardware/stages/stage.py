from enum import Enum

from microEye.qt import QtWidgets


class stage:
    def __init__(self) -> None:
        '''
        Initialize the stage object.

        This method initializes the `stage` object with the starter values.
        '''
        self._connect_btn = QtWidgets.QPushButton()
        self.ZPosition = 50000
        self.LastCmd = ''
        self.Received = ''

    def isOpen(self):
        '''
        Check if the stage is open.

        Returns
        -------
        is_open : bool
            A boolean value indicating whether the stage is open.
        '''
        return False


class XYStageParams(Enum):
    '''
    Enum class defining XY Stage parameters.
    '''

    MODEL = 'Model'
    STATUS = 'Status'

    X_POSITION = 'X Position'
    Y_POSITION = 'Y Position'
    GET_POSITION = 'Get Position'

    MOVE = 'Move'
    HOME = 'Home'
    CENTER = 'Center'
    STOP = 'Stop !'

    CONTROLS = 'Controls'
    X_JUMP_P = 'Controls.X++'
    X_JUMP_N = 'Controls.X--'
    X_STEP_P = 'Controls.X+'
    X_STEP_N = 'Controls.X-'

    Y_JUMP_P = 'Controls.Y++'
    Y_JUMP_N = 'Controls.Y--'
    Y_STEP_P = 'Controls.Y+'
    Y_STEP_N = 'Controls.Y-'

    OPTIONS = 'Options'
    STEP = 'Options.Step Size'
    JUMP = 'Options.Jump Size'
    ID_X = 'Options.ID X'
    ID_Y = 'Options.ID Y'

    SERIAL_PORT = 'Serial Port'
    PORT = 'Serial Port.Port'
    BAUDRATE = 'Serial Port.Baudrate'
    REFRESH = 'Serial Port.Refresh Ports'
    SET_PORT = 'Serial Port.Set Config'
    SET_PORT_X = 'Serial Port.Set Config X'
    SET_PORT_Y = 'Serial Port.Set Config Y'
    OPEN = 'Serial Port.Connect'
    CLOSE = 'Serial Port.Disconnect'
    PORT_STATE = 'Serial Port.State'

    INFO = 'Info'

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
