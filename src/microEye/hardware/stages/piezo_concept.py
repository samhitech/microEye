import re
from enum import Enum
from typing import Optional, Union, overload

from pyqtgraph.parametertree import Parameter

from microEye.hardware.stages.stabilizer import FocusStabilizer
from microEye.hardware.stages.stage import stage
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils import Tree


class StageParams(Enum):
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


class PzFocSignals(QtCore.QObject):
    positionChanged = Signal(float)


class PzFoc(stage):
    '''PiezoConcept FOC 1-axis stage adapter.'''

    def __init__(self):
        '''
        Initializes a PiezoConcept FOC 1-axis stage adapter.
        '''
        self.signals = PzFocSignals()

        super().__init__()

        self.max = 100 * 1000

        self.serial = QtSerialPort.QSerialPort(None, readyRead=self.rx_piezo)
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM5')

    def isOpen(self):
        '''Returns True if connected.'''
        return self.serial.isOpen()

    def open(self):
        '''Opens the serial port.'''
        if not self.isOpen():
            self.serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

    def close(self):
        '''Closes the supplied serial port.'''
        if self.isOpen():
            self.serial.close()

    def setPortName(self, name: str):
        '''Sets the serial port name.'''
        if not self.isOpen():
            self.serial.setPortName(name)

    def setBaudRate(self, baudRate: int):
        '''Sets the serial port baudrate.'''
        if not self.isOpen():
            self.serial.setBaudRate(baudRate)

    def write(self, value):
        '''
        Writes data to the serial port.

        Parameters
        ----------
        data : QByteArray | bytes | bytearray
            The data to be written to the serial port.

        Returns
        -------
        int
            The number of bytes written to the serial port, or a negative value
            if an error occurs.

        Raises
        ------
        IOError
            If the serial port is not open.
        '''
        return self.serial.write(value)

    def GETZ(self):
        '''Gets the current stage position along the Z axis.'''
        if self.isOpen():
            self.write(b'GET_Z\n')
            self.LastCmd = 'GETZ'

    def HOME(self):
        '''Centers the stage position along the Z axis.'''
        if self.isOpen():
            self.ZPosition = self.max // 2
            self.write(b'MOVEZ 50u\n')
            self.LastCmd = 'MOVRZ'

    def REFRESH(self):
        '''Refresh the stage position
        to the set value in case of discrepancy.
        '''
        if self.isOpen():
            self.write(('MOVEZ ' + str(self.ZPosition) + 'n\n').encode('utf-8'))
            self.LastCmd = 'MOVEZ'

    def UP(self, step: int):
        '''
        Moves the stage up by a specified number of steps.

        Parameters
        ----------
        step : int
            The number of steps to move up in nanometers.

        Raises
        ------
        IOError
            If the serial port is not open or an error occurs while writing.
        '''
        if self.isOpen():
            self.ZPosition = min(max(self.ZPosition + step, 0), self.max)
            self.write(('MOVEZ ' + str(self.ZPosition) + 'n\n').encode('utf-8'))
            self.LastCmd = 'MOVEZ'

    def DOWN(self, step: int):
        '''
        Moves the stage down by a specified number of steps.

        Parameters
        ----------
        step : int
            The number of steps to move down in nanometers.

        Raises
        ------
        IOError
            If the serial port is not open or an error occurs while writing.
        '''
        if self.isOpen():
            self.ZPosition = min(max(self.ZPosition - step, 0), self.max)
            self.write(('MOVEZ ' + str(self.ZPosition) + 'n\n').encode('utf-8'))
            self.LastCmd = 'MOVEZ'

    def rx_piezo(self):
        '''
        Reads data from the serial port and processes it.

        This method is called automatically when new data is available in the
        serial port input buffer. It reads the data, parses the response, and
        updates the appropriate attributes of the `PzFoc` object as needed.

        Raises
        ------
        IOError
            If the serial port is not open, or an error occurs while reading.
        '''
        self.Received = str(self.serial.readAll(), encoding='utf8')
        if self.LastCmd != 'GETZ':
            self.GETZ()
        else:
            match = re.search(r' *(\d+\.\d+).*um.*', self.Received)
            if match:
                self.signals.positionChanged.emit(float(match.group(1)))

    def getQWidget(self):
        '''Generates a PzFocView with stage controls.'''
        return PzFocView(stage=self)


class PzFocView(Tree):
    PARAMS = StageParams
    removed = Signal(object)

    def __init__(
        self, parent: Optional['QtWidgets.QWidget'] = None, stage: PzFoc = None
    ):
        '''
        Initialize the PzFocView instance.

        This method initializes the `PzFocView` instance, sets up the stage signals,
        creates the parameter tree, and sets up the GUI elements.

        Parameters
        ----------
        parent : Optional[QWidget]
            The parent widget.
        stage : Optional[`PzFoc`]
            The stage to be controlled by the GUI. If None, a new stage instance is
            created.
        '''
        self.stage = stage if stage else PzFoc()

        super().__init__(parent=parent)

        self.stage.signals.positionChanged.connect(
            lambda value: self.set_param_value(StageParams.Z_POSITION, value)
        )

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `PzFocView` class.
        '''
        params = [
            {
                'name': str(StageParams.MODEL),
                'type': 'str',
                'value': 'PiezoConcept FOC 1-axis',
                'readonly': True,
            },
            {'name': str(StageParams.HOME), 'type': 'action'},
            {'name': str(StageParams.REFRESH), 'type': 'action'},
            {'name': str(StageParams.JUMP_P), 'type': 'action'},
            {'name': str(StageParams.STEP_P), 'type': 'action'},
            {'name': str(StageParams.STEP_N), 'type': 'action'},
            {'name': str(StageParams.JUMP_N), 'type': 'action'},
            {
                'name': str(StageParams.SERIAL_PORT),
                'type': 'group',
                'children': [
                    {
                        'name': str(StageParams.PORT),
                        'type': 'list',
                        'limits': [
                            info.portName()
                            for info in QtSerialPort.QSerialPortInfo.availablePorts()
                        ],
                    },
                    {
                        'name': str(StageParams.BAUDRATE),
                        'type': 'list',
                        'value': 115200,
                        'limits': [
                            baudrate
                            for baudrate in \
                                QtSerialPort.QSerialPortInfo.standardBaudRates()
                        ],
                    },
                    {'name': str(StageParams.SET_PORT), 'type': 'action'},
                    {'name': str(StageParams.OPEN), 'type': 'action'},
                    {'name': str(StageParams.CLOSE), 'type': 'action'},
                    {
                        'name': str(StageParams.PORT_STATE),
                        'type': 'str',
                        'value': 'closed',
                        'readonly': True,
                    },
                ],
            },
            {
                'name': str(StageParams.READINGS),
                'type': 'group',
                'children': [
                    {
                        'name': str(StageParams.Z_POSITION),
                        'type': 'float',
                        'value': 0.0,
                        'readonly': True,
                        'decimals': 6,
                    },
                ],
            },
            {
                'name': str(StageParams.OPTIONS),
                'type': 'group',
                'children': [
                    {
                        'name': str(StageParams.STEP),
                        'type': 'int',
                        'value': 100,
                        'limits': [1, 1000],
                        'step': 5,
                    },
                    {
                        'name': str(StageParams.JUMP),
                        'type': 'int',
                        'value': 1,
                        'limits': [1, 100],
                        'step': 5,
                    },
                ],
            },
            {'name': str(StageParams.REMOVE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(StageParams.SET_PORT).sigActivated.connect(self.set_config)
        self.get_param(StageParams.OPEN).sigActivated.connect(lambda: self.stage.open())
        self.get_param(StageParams.CLOSE).sigActivated.connect(
            lambda: self.stage.close()
        )

        self.get_param(StageParams.REMOVE).sigActivated.connect(self.remove_widget)

        self.get_param(StageParams.HOME).sigActivated.connect(self.stage.HOME)
        self.get_param(StageParams.REFRESH).sigActivated.connect(self.stage.REFRESH)

    def set_config(self):
        '''Sets the serial port configuration.

        This method sets the serial port configuration based on the current
        settings in the parameter tree.
        '''
        if not self.stage.isOpen():
            self.stage.setPortName(self.get_param_value(StageParams.PORT))
            self.stage.setBaudRate(self.get_param_value(StageParams.BAUDRATE))

    def remove_widget(self):
        '''
        Remove the widget from the parent layout.

        This method removes the widget from the parent layout and deletes it.
        '''
        if self.parent() and not self.stage.isOpen():
            self.parent().layout().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Disconnect FOC stage before removing!')

    def __str__(self):
        return f'Piezo Concept FOC Stage ({self.stage.serial.portName()})'

class PzFocController:
    def __init__(self, stage: PzFoc = None, view: PzFocView = None):
        '''
        Initialize the PzFocController class.

        This class controls the interaction between the PzFoc stage
        and the PzFocView widget.

        Parameters
        ----------
        stage: pz_foc
            The stage to be controlled.
            Default is a new PzFoc instance.
        view: FOC_View
            The GUI/view for the stage.
            Default is a new PzFocView instance with the stage as an argument.
        '''
        self.stage = stage if stage else PzFoc()
        self.view = view if view else PzFocView(stage=self.stage)

        self.view.get_param(StageParams.JUMP_P).sigActivated.connect(
            lambda: self.moveStage(
                True, self.view.get_param_value(StageParams.JUMP) * 1000, True
            )
        )
        self.view.get_param(StageParams.JUMP_N).sigActivated.connect(
            lambda: self.moveStage(
                False, self.view.get_param_value(StageParams.JUMP) * 1000, True
            )
        )
        self.view.get_param(StageParams.STEP_P).sigActivated.connect(
            lambda: self.moveStage(
                True, self.view.get_param_value(StageParams.STEP), True
            )
        )
        self.view.get_param(StageParams.STEP_N).sigActivated.connect(
            lambda: self.moveStage(
                False, self.view.get_param_value(StageParams.STEP), True
            )
        )

    def setPortName(self, value: str):
        '''
        Set the name of the serial port.

        Parameters
        ----------
        value : str
            The name of the serial port.
        '''
        self.stage.serial.setPortName(value)
        self.view.set_param_value(StageParams.PORT, value)

    def setBaudRate(self, value: int):
        '''
        Set the baud rate of the serial port.

        Parameters
        ----------
        value : int
            The baud rate of the serial port.
        '''
        self.stage.serial.setBaudRate(value)
        self.view.set_param_value(StageParams.BAUDRATE, value)

    def connect(self):
        '''
        Opens the stage serial port.
        '''
        self.stage.open()

    def isOpen(self):
        '''
        Check if the stage is open.

        Returns
        -------
        bool
            True if the stage is open, False otherwise.
        '''
        return self.stage.isOpen()

    def getStep(self):
        '''
        Get the current step size.

        Returns
        -------
        float
            The current step size.
        '''
        return self.view.get_param_value(StageParams.STEP)

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
            step = self.view.get_param_value(StageParams.STEP)
            self.view.set_param_value(StageParams.STEP, step + value)
        else:
            self.view.set_param_value(StageParams.STEP, value)

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
            If provided as a boolean and True, moves the stage up by the value of the
            JUMP parameter, otherwise, moves the stage up or down by the value of the
            STEP parameter.
        interface : bool, optional
            If True, moves the center pixel value instead of the Z position
            when FocusStabilizer is stabilizing and use calibration is set to True.
        '''
        if isinstance(step_arg, bool):
            if step_arg:
                step_arg = 1000 * self.view.get_param_value(StageParams.JUMP)
            else:
                step_arg = self.view.get_param_value(StageParams.STEP)

        focusStabilizer = FocusStabilizer.instance()
        if (
            focusStabilizer is not None
            and focusStabilizer.isFocusStabilized()
            and focusStabilizer.useCal()
            and interface
        ):
            sign = 1 if dir else -1
            focusStabilizer.setPeakPosition(
                focusStabilizer.pixelCalCoeff() * step_arg * sign, True
            )
        else:
            self.stage.UP(step_arg) if dir else self.stage.DOWN(step_arg)

    def updatePortState(self):
        '''
        Update the port state in the parameter tree.

        This method updates the port state in the parameter tree
        to reflect the current state of the serial port.
        '''
        self.view.set_param_value(
            StageParams.PORT_STATE, 'open' if self.isOpen() else 'closed'
        )

    def refreshPorts(self):
        """
        Refreshes the available serial ports list in the GUI.

        This method updates the list of available serial ports in the GUI by fetching
        the current list of available ports and setting it as the options for the
        'QSerial Port' parameter in the parameter tree.
        """
        if not self.isOpen():
            self.view.get_param(StageParams.PORT).setLimits(
                [
                    info.portName()
                    for info in QtSerialPort.QSerialPortInfo.availablePorts()
                ]
            )

    def __str__(self):
        return 'Piezo Concept FOC Stage Controller'
