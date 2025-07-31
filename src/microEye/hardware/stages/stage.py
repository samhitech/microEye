from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Union

from pyqtgraph.parametertree import Parameter

from microEye.hardware.stages.stabilizer import FocusStabilizer
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree


class Axis(Enum):
    '''
    Enum class defining the axis of the stage.
    '''

    X = 'X'
    Y = 'Y'
    Z = 'Z'

    def __str__(self):
        '''
        Return the last part of the enum value (Axis name).
        '''
        return self.value.split('.')[-1]

    @classmethod
    def from_string(cls, axis_str: str):
        '''
        Create an Axis enum from a string.

        Parameters
        ----------
        axis_str : str
            The string representation of the axis.

        Returns
        -------
        Axis
            The corresponding Axis enum value.
        '''
        axis_str = axis_str.upper()
        if axis_str in cls.__members__:
            return cls[axis_str]
        else:
            raise ValueError(f'Invalid axis: {axis_str}')


class Units(Enum):
    '''
    Enum class defining the units of the stage.
    '''

    NANOMETERS = 1  # 1 nm = 1 nm
    MICROMETERS = 1000  # 1 um = 1000 nm
    MILIMETERS = 1000000  # 1 mm = 1,000,000 nm


class ZStageSignals(QtCore.QObject):
    positionChanged = Signal(float)


class AbstractStage(ABC):
    '''
    Abstract base class for all stages.
    '''

    def __init__(
        self,
        name: str = 'Abstract Base Stage',
        max_range: tuple[float] = 0,
        units: tuple[Units] = Units.NANOMETERS,
        readyRead: Callable = None,
    ):
        """
        Initialize the stage object.

        Parameters
        ----------
        name : str, optional
            The name of the stage (default is 'Abstract Base Stage').
        max_range : int, optional
            The maximum range of the stage (default is 0).
        units : tuple[Units], optional
            The units of the stage (default is Units.NANOMETERS).
        readyRead : Callable, optional
            A callable function to be executed when the serial port is ready to read
            data (default is None).
        """
        self._name = name

        def get_dict(props: Union[tuple, list, Any]) -> dict[Axis, Any]:
            if isinstance(props, (tuple, list)) and len(props) == 3:
                return {axis: props[i] for i, axis in enumerate(Axis)}
            else:
                return {axis: props for axis in Axis}

        self._max_range = get_dict(max_range)
        self._units: dict[Axis, Units] = get_dict(units)
        self._position = {axis: 0 for axis in Axis}

        self.signals = ZStageSignals()

        self.serial = QtSerialPort.QSerialPort(None)
        if callable(readyRead):
            self.serial.readyRead.connect(readyRead)
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM5')

        self._is_serial = readyRead is not None

    def __str__(self):
        return self._name

    def is_serial(self) -> bool:
        '''
        Check if the stage is serial.

        Returns
        -------
        bool
            True if the stage is serial, False otherwise.
        '''
        return self._is_serial

    @abstractmethod
    def is_open(self) -> bool:
        '''
        Check if the stage is open.

        Returns
        -------
        bool
            True if the stage is open, False otherwise.
        '''
        pass

    def setPortName(self, name: str):
        '''Sets the serial port name.'''
        if not self.is_open():
            self.serial.setPortName(name)

    def setBaudRate(self, baudRate: int):
        '''Sets the serial port baudrate.'''
        if not self.is_open():
            self.serial.setBaudRate(baudRate)

    def get_position(self, axis: Axis) -> float:
        '''
        Get the current position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to query.

        Returns
        -------
        float
            The current position.
        '''
        return self._position[axis]

    def set_position(self, axis: Axis, position: float, incremental=False) -> None:
        '''
        Set the current position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to set.
        position : float
            The target position.
        '''
        if incremental:
            position = self._position[axis] + position

        if position < 0 or position > self._max_range[axis]:
            return

        self._position[axis] = max(0, min(position, self._max_range[axis]))

    def convert_to_nm(self, value: float, axis: Axis = Axis.Z) -> float:
        """
        Convert a value from the stage's native units to nanometers.

        Parameters
        ----------
        value : float
            The value in the stage's native units.
        axis : Axis
            The axis for which the conversion is being performed.

        Returns
        -------
        float
            The value converted to nanometers.
        """
        return value * self._units[axis].value

    def convert_from_nm(self, value: float, axis: Axis = Axis.Z) -> float:
        """
        Convert a value from nanometers to the stage's native units.

        Parameters
        ----------
        value : float
            The value in nanometers.
        axis : Axis
            The axis for which the conversion is being performed.

        Returns
        -------
        float
            The value converted to the stage's native units.
        """
        return value / self._units[axis].value

    @abstractmethod
    def home(self) -> None:
        '''
        Move the stage to its home position.
        '''
        pass

    @abstractmethod
    def connect(self):
        '''
        Connect to the stage.
        '''
        pass

    @abstractmethod
    def disconnect(self):
        '''
        Disconnect from the stage.
        '''
        pass

    @abstractmethod
    def move_absolute(self, pos):
        '''
        Move the stage to an absolute position.

        Parameters
        ----------
        pos : any
            The target position.
        '''
        pass

    @abstractmethod
    def move_relative(self, pos):
        '''
        Move the stage to a relative position.

        Parameters
        ----------
        pos : any
            The target position.
        '''
        pass

    @abstractmethod
    def refresh_position(self):
        '''
        Refreshes the current position of the stage.

        **Example implementation:**

        ```python
        self.move_absolute(self.position)
        ```
        '''
        pass

    def move_up(self, step: int):
        '''
        Moves the stage up (positive direction) by a specified number of steps.

        Parameters
        ----------
        step : int
            The number of steps to move up in nanometers.

        Raises
        ------
        IOError
            If the serial port is not open or an error occurs while writing.
        '''
        self.move_relative(step)

    def move_down(self, step: int):
        '''
        Moves the stage down (negative direction) by a specified number of steps.

        Parameters
        ----------
        step : int
            The number of steps to move down in nanometers.

        Raises
        ------
        IOError
            If the serial port is not open or an error occurs while writing.
        '''
        self.move_relative(-step)

    def set_max_range(self, axis: Axis, max_range: float) -> None:
        '''
        Set the maximum range of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to set.
        max_range : float
            The maximum range.
        '''
        self._max_range[axis] = max_range

    @property
    def z_max_um(self) -> float:
        '''Maximum stage position in micrometers.'''
        return self.z_max_nm / 1000

    @property
    def z_max_nm(self) -> int:
        '''Maximum stage position in nanometers.'''
        return self._max_range[Axis.Z]

    @abstractmethod
    def stop(self) -> None:
        '''
        Stop the stage movement.
        '''
        pass


class ZStageParams(Enum):
    '''
    Enum class defining Stage parameters.
    '''

    MODEL = 'Model'

    HOME = 'Home'
    REFRESH = 'Refresh'
    JUMP_P = 'Jump +'
    JUMP_N = 'Jump -'
    STEP_P = 'Step +'
    STEP_N = 'Step -'

    SERIAL_PORT = 'Serial Port'
    PORT = 'Serial Port.Port'
    BAUDRATE = 'Serial Port.Baudrate'
    SET_PORT = 'Serial Port.Set Config'
    OPEN = 'Serial Port.Connect'
    CLOSE = 'Serial Port.Disconnect'
    PORT_STATE = 'Serial Port.State'

    OPTIONS = 'Options'
    STEP = 'Options.Step Size [nm]'
    JUMP = 'Options.Jump Size [um]'

    READINGS = 'Readings'
    Z_POSITION = 'Readings.Z Position [um]'

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


class ZStageView(Tree):
    PARAMS = ZStageParams
    removed = Signal(object)

    def __init__(
        self,
        stage: AbstractStage,
        parent: Optional['QtWidgets.QWidget'] = None,
    ):
        '''
        Initialize the ZStageView instance.

        This method initializes the `ZStageView` instance, sets up the stage signals,
        creates the parameter tree, and sets up the GUI elements.

        Parameters
        ----------
        stage : Optional[`AbstractStage`]
            The stage to be controlled by the GUI. If None, a new stage instance is
            created.
        parent : Optional[QWidget]
            The parent widget.
        '''
        if stage is None:
            raise ValueError('Stage cannot be None.')

        self.stage = stage

        super().__init__(parent=parent)

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `ZStageView` class.
        '''
        params = [
            {
                'name': str(ZStageParams.MODEL),
                'type': 'str',
                'value': str(self.stage),
                'readonly': True,
            },
            {'name': str(ZStageParams.HOME), 'type': 'action'},
            {'name': str(ZStageParams.REFRESH), 'type': 'action'},
            {'name': str(ZStageParams.JUMP_P), 'type': 'action'},
            {'name': str(ZStageParams.STEP_P), 'type': 'action'},
            {'name': str(ZStageParams.STEP_N), 'type': 'action'},
            {'name': str(ZStageParams.JUMP_N), 'type': 'action'},
            {
                'name': str(ZStageParams.SERIAL_PORT),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(ZStageParams.PORT),
                        'type': 'list',
                        'limits': [
                            info.portName()
                            for info in QtSerialPort.QSerialPortInfo.availablePorts()
                        ],
                    },
                    {
                        'name': str(ZStageParams.BAUDRATE),
                        'type': 'list',
                        'value': 115200,
                        'limits': [
                            bd
                            for bd in QtSerialPort.QSerialPortInfo.standardBaudRates()
                        ],
                    },
                    {'name': str(ZStageParams.SET_PORT), 'type': 'action'},
                    {'name': str(ZStageParams.OPEN), 'type': 'action'},
                    {'name': str(ZStageParams.CLOSE), 'type': 'action'},
                    {
                        'name': str(ZStageParams.PORT_STATE),
                        'type': 'str',
                        'value': 'closed',
                        'readonly': True,
                    },
                ],
            },
            {
                'name': str(ZStageParams.READINGS),
                'type': 'group',
                'children': [
                    {
                        'name': str(ZStageParams.Z_POSITION),
                        'type': 'float',
                        'value': 0.0,
                        'readonly': True,
                        'decimals': 6,
                    },
                ],
            },
            {
                'name': str(ZStageParams.OPTIONS),
                'type': 'group',
                'children': [
                    {
                        'name': str(ZStageParams.STEP),
                        'type': 'int',
                        'value': 100,
                        'limits': [1, 1000],
                        'step': 5,
                    },
                    {
                        'name': str(ZStageParams.JUMP),
                        'type': 'int',
                        'value': 1,
                        'limits': [1, 100],
                        'step': 5,
                    },
                ],
            },
            {'name': str(ZStageParams.REMOVE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

    def remove_widget(self):
        '''
        Remove the widget from the parent layout.

        This method removes the widget from the parent layout and deletes it.
        '''
        if self.parent() and not self.stage.is_open():
            self.parent().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Cannot remove {self.stage} widget. Try to disconnect first.')

    def __str__(self):
        return f'{self.stage} View'


class ZStageController:
    def __init__(self, stage: AbstractStage, view: ZStageView = None):
        '''
        Initialize the ZStageController class.

        This class controls the interaction between the PzFoc stage
        and the ZStageView widget.

        Parameters
        ----------
        stage: AbstractStage
            The stage to be controlled.
        view: ZStageView
            The GUI/view for the stage.
            Default is a new ZStageView instance with the stage as an argument.
        '''
        if stage is None:
            raise ValueError('Stage cannot be None.')

        self.stage = stage
        self.view = view if view else ZStageView(stage=self.stage)

        self.view.get_param(ZStageParams.JUMP_P).sigActivated.connect(
            lambda: self.moveStage(
                True, self.view.get_param_value(ZStageParams.JUMP) * 1000, True
            )
        )
        self.view.get_param(ZStageParams.JUMP_N).sigActivated.connect(
            lambda: self.moveStage(
                False, self.view.get_param_value(ZStageParams.JUMP) * 1000, True
            )
        )
        self.view.get_param(ZStageParams.STEP_P).sigActivated.connect(
            lambda: self.moveStage(
                True, self.view.get_param_value(ZStageParams.STEP), True
            )
        )
        self.view.get_param(ZStageParams.STEP_N).sigActivated.connect(
            lambda: self.moveStage(
                False, self.view.get_param_value(ZStageParams.STEP), True
            )
        )

        self.view.get_param(ZStageParams.SET_PORT).sigActivated.connect(self.set_config)
        self.view.get_param(ZStageParams.OPEN).sigActivated.connect(
            lambda: self.stage.connect()
        )
        self.view.get_param(ZStageParams.CLOSE).sigActivated.connect(
            lambda: self.stage.disconnect()
        )

        self.view.get_param(ZStageParams.HOME).sigActivated.connect(self.stage.home)
        self.view.get_param(ZStageParams.REFRESH).sigActivated.connect(
            self.stage.refresh_position
        )

        self.stage.signals.positionChanged.connect(
            lambda value: self.view.set_param_value(ZStageParams.Z_POSITION, value)
        )

    def set_config(self):
        '''Sets the serial port configuration.

        This method sets the serial port configuration based on the current
        settings in the parameter tree.
        '''
        if (
            not self.stage.is_open()
            and hasattr(self.stage, 'setPortName')
            and hasattr(self.stage, 'setBaudRate')
        ):
            self.stage.setPortName(self.view.get_param_value(ZStageParams.PORT))
            self.stage.setBaudRate(self.view.get_param_value(ZStageParams.BAUDRATE))

    def setPortName(self, value: str):
        '''
        Set the name of the serial port.

        Parameters
        ----------
        value : str
            The name of the serial port.
        '''
        if not hasattr(self.stage, 'setPortName'):
            self.stage.setPortName(value)
        self.view.set_param_value(ZStageParams.PORT, value)

    def setBaudRate(self, value: int):
        '''
        Set the baud rate of the serial port.

        Parameters
        ----------
        value : int
            The baud rate of the serial port.
        '''
        if not hasattr(self.stage, 'setBaudRate'):
            self.stage.setBaudRate(value)
        self.view.set_param_value(ZStageParams.BAUDRATE, value)

    def connect(self):
        '''
        Opens the stage serial port.
        '''
        self.stage.connect()

    def disconnect(self):
        '''
        Closes the stage serial port.
        '''
        self.stage.disconnect()

    def isOpen(self):
        '''
        Check if the stage is open.

        Returns
        -------
        bool
            True if the stage is open, False otherwise.
        '''
        return self.stage.is_open()

    def isSerial(self):
        '''
        Check if the stage is serial.

        Returns
        -------
        bool
            True if the stage is serial, False otherwise.
        '''
        return self.stage.is_serial()

    def getStep(self):
        '''
        Get the current step size.

        Returns
        -------
        float
            The current step size.
        '''
        return self.view.get_param_value(ZStageParams.STEP)

    def setStep(self, value: float, incerement: bool = False):
        '''
        Set the step size.

        Parameters
        ----------
        value : float
            The new step size value.
        incerement : bool, optional
            If True, increment the current step size value by the given value, otherwise
            set the step size to the given value.
        '''
        if incerement:
            step = self.view.get_param_value(ZStageParams.STEP)
            self.view.set_param_value(ZStageParams.STEP, step + value)
        else:
            self.view.set_param_value(ZStageParams.STEP, value)

    def moveAbsolute(self, pos: int):
        '''
        Move the stage to an absolute position.

        Parameters
        ----------
        pos : int
            The absolute position to move to in nanometers.
        '''
        focusStabilizer = FocusStabilizer.instance()
        if focusStabilizer is not None and focusStabilizer.isFocusStabilized():
            return

        self.stage.move_absolute(pos)

    def moveStage(self, dir: bool, step_arg: Union[int, bool], interface: bool = False):
        '''
        Move the stage in a specified direction by a specified
        number of steps in nanometers. Optional boolean argument
        to specify jump or step move. If FocusStabilizer is stabilizing
        and use calibration is set to True, moves the center pixel value
        instead of the Z position when `interface` is True.

        Parameters
        ----------
        dir: bool
            Direction of the movement. If True, moves the stage up, else moves it down.
        step_arg: Union[int, bool]
            Number of steps to move in the specified direction.
            If provided as a `bool`:
            - `True`: Moves the stage by the value of the JUMP parameter.
            - `False`: Moves the stage by the value of the STEP parameter.
        interface : bool, optional
            If True, moves the center pixel value instead of the Z position
            when FocusStabilizer is stabilizing and use calibration is set to True.
        '''
        if isinstance(step_arg, bool):
            if step_arg:
                step_arg = 1000 * self.view.get_param_value(ZStageParams.JUMP)
            else:
                step_arg = self.view.get_param_value(ZStageParams.STEP)

        focusStabilizer = FocusStabilizer.instance()
        if (
            focusStabilizer is not None
            and focusStabilizer.isFocusStabilized()
            and focusStabilizer.useCal()
            and interface
        ):
            sign = 1 if dir else -1
            focusStabilizer.setParameter(
                focusStabilizer.calCoeff() * step_arg * sign, True
            )
        else:
            self.stage.move_up(step_arg) if dir else self.stage.move_down(step_arg)

    def updatePortState(self):
        '''
        Update the port state in the parameter tree.

        This method updates the port state in the parameter tree
        to reflect the current state of the serial port.
        '''
        self.view.set_param_value(
            ZStageParams.PORT_STATE, 'open' if self.isOpen() else 'closed'
        )

    def refreshPorts(self):
        """
        Refreshes the available serial ports list in the GUI.

        This method updates the list of available serial ports in the GUI by fetching
        the current list of available ports and setting it as the options for the
        'QSerial Port' parameter in the parameter tree.
        """
        if not self.isOpen():
            self.view.get_param(ZStageParams.PORT).setLimits(
                [
                    info.portName()
                    for info in QtSerialPort.QSerialPortInfo.availablePorts()
                ]
            )

    def get_config(self) -> dict:
        '''Get the current configuration of the stage.'''
        return {
            'port': self.stage.serial.portName(),
            'baudrate': self.stage.serial.baudRate(),
            'max_nm': self.stage.z_max_nm,
        }

    def load_config(self, config: dict) -> None:
        '''Load the configuration of the stage from a dictionary.'''
        self.stage.setPortName(config.get('port', 'COM5'))
        self.stage.setBaudRate(config.get('baudrate', 115200))
        self.stage.set_max_range(Axis.Z, config.get('max_nm', 100000))

    def __str__(self):
        return f'{self.stage} Stage Controller'
