import os
import sys
from enum import Enum
from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils import StartGUI, Tree


class LaserRelay:
    def __init__(self) -> None:
        """
        Initialize a new LaserRelay instance.

        Sets up the serial port with a baud rate of 115200 and a port of 'COM6'.
        Also initializes the last sent configuration, the current laser relay state,
        and the ALEX flag.
        """
        self.__port = QtSerialPort.QSerialPort()
        self.__port.setBaudRate(115200)
        self.__port.setPortName('COM6')

        self.__last = ''
        self.laserRelay_curr = ''
        self.__alex = False

    @property
    def lastCommand(self):
        return self.__last

    def isOpen(self) -> bool:
        '''
        Check if the serial port is open.

        Returns
        -------
        bool
            True if the serial port is open, False otherwise.
        '''
        return self.__port.isOpen()

    def isALEX(self) -> bool:
        '''
        Check if the ALEX flag is set.

        Returns
        -------
        bool
            True if the ALEX flag is set, False otherwise.
        '''
        return self.__alex

    def setALEX(self, value: bool):
        '''
        Set the ALEX flag.

        Parameters
        ----------
        value : bool
            The new value for the ALEX flag.

        Raises
        ------
        TypeError
            If the value is not a boolean.
        '''
        if not isinstance(value, bool):
            raise TypeError('ALEX value should be boolean.')

        self.__alex = value

    def open(self):
        '''
        Open the serial port.

        If the serial port is not already open,
        this method will open it in read-write mode.
        '''
        if not self.isOpen():
            self.__port.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

    def close(self):
        '''
        Close the serial port.

        If the serial port is already open, this method will close it.
        '''
        if self.isOpen():
            self.__port.close()

    def portName(self):
        '''
        Get the name of the serial port.

        Returns
        -------
        str
            The name of the serial port.
        '''
        return self.__port.portName()

    def baudRate(self):
        '''
        Get the baud rate of the serial port.

        Returns
        -------
        int
            The baud rate of the serial port.
        '''
        return self.__port.baudRate()

    def setBaudRate(self, value: int):
        '''
        Set the baud rate of the serial port.

        Parameters
        ----------
        value : int
            The new baud rate for the serial port.

        Raises
        ------
        ValueError
            If the value is not a valid baud rate.
        '''
        if not self.isOpen():
            self.__port.setBaudRate(value)

    def setPortName(self, value: str):
        '''
        Set the port of the serial port.

        Parameters
        ----------
        value : str
            The new port for the serial port.

        Raises
        ------
        ValueError
            If the value is not a valid port.
        '''
        if not self.isOpen():
            self.__port.setPortName(value)

    def sendConfig(self, config: str = None):
        '''
        Send the RelayBox configuration command.

        Parameters
        ----------
        config : str, optional
            The configuration to send. If None, the current configuration will be sent.

        Raises
        ------
        Exception
            If there is an error sending the configuration.
        '''
        try:
            if not self.isOpen():
                raise ConnectionError('Device is not open.')

            if config is None:
                config = ('ALEXON' if self.isALEX() else 'ALEXOFF') + '\r'

            self.__port.write(config.encode('utf-8'))
            self.__last = config
            print(str(self.__port.readAll(), encoding='utf-8'))
        except Exception as e:
            print('Failed Laser Relay Send Config: ' + str(e))


class RelayParams(Enum):
    '''
    Enum class defining Laser Relay parameters.
    '''

    MODEL = 'Model'
    ALEX = 'ALEX'
    SEND_COMMAND = 'Send Command'

    SERIAL_PORT = 'Serial Port'
    PORT = 'Serial Port.Port'
    BAUDRATE = 'Serial Port.Baudrate'
    SET_PORT = 'Serial Port.Set Config'
    OPEN = 'Serial Port.Connect'
    CLOSE = 'Serial Port.Disconnect'
    PORT_STATE = 'Serial Port.State'

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


class LaserRelayView(Tree):
    PARAMS = RelayParams
    sendCommandActivated = Signal()
    removed = Signal(object)

    def __init__(
        self, parent: Optional['QtWidgets.QWidget'] = None, relay: LaserRelay = None
    ):
        self.__laserRelay = relay if relay else LaserRelay()

        super().__init__(parent=parent)

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {
                'name': str(RelayParams.MODEL),
                'type': 'str',
                'value': 'Laser Relay Box',
                'readonly': True,
            },
            {'name': str(RelayParams.ALEX), 'type': 'bool', 'value': False},
            {'name': str(RelayParams.SEND_COMMAND), 'type': 'action'},
            {
                'name': str(RelayParams.SERIAL_PORT),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(RelayParams.PORT),
                        'type': 'list',
                        'limits': [
                            info.portName()
                            for info in QtSerialPort.QSerialPortInfo.availablePorts()
                        ],
                    },
                    {
                        'name': str(RelayParams.BAUDRATE),
                        'type': 'list',
                        'value': 115200,
                        'limits': [
                            baudrate
                            for baudrate in \
                                QtSerialPort.QSerialPortInfo.standardBaudRates()
                        ],
                    },
                    {'name': str(RelayParams.SET_PORT), 'type': 'action'},
                    {'name': str(RelayParams.OPEN), 'type': 'action'},
                    {'name': str(RelayParams.CLOSE), 'type': 'action'},
                    {
                        'name': str(RelayParams.PORT_STATE),
                        'type': 'str',
                        'value': 'closed',
                        'readonly': True,
                    },
                ],
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(RelayParams.CLOSE).sigActivated.connect(self.__laserRelay.close)
        self.get_param(RelayParams.SEND_COMMAND).sigActivated.connect(
            lambda: self.sendCommandActivated.emit()
        )

    def remove_widget(self):
        '''
        Remove the widget from its parent.

        This method removes the view from its parent widget if
        the laser relay is not open.
        '''
        if self.parent() and not self.__laserRelay.isOpen():
            self.parent().layout().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Disconnect Laser Relay before removing!')

    def StartGUI():
        '''Initializes a new :class:`QApplication` and :class:`LaserRelay`.

        Use
        -------
        app, window = LaserRelay.StartGUI()

        app.exec()

        Returns
        -------
        tuple (:class:`QApplication`, :class:`LaserRelay`)
            Returns a tuple with QApp and LaserRelay main window.
        '''
        return StartGUI(LaserRelay)


class LaserRelayController:
    def __init__(self, relay: LaserRelay = None, view: LaserRelayView = None):
        '''
        Initialize the LaserRelayController class.

        This class controls the interaction between the LaserRelay
        and the LaserRelayView widget.

        Parameters
        ----------
        relay: LaserRelay
            The laser relay to be controlled.
            Default is a new LaserRelay instance.
        view: LaserRelayView
            The GUI/view for the laser relay.
            Default is a new LaserRelayView instance with the relay as an argument.
        '''
        self.__relay = relay if relay else LaserRelay()
        self.__view = view if view else LaserRelayView(relay=self.__relay)

        self.__view.get_param(RelayParams.SET_PORT).sigActivated.connect(
            self.set_config
        )
        self.__view.get_param(RelayParams.OPEN).sigActivated.connect(self.connect)

    @property
    def sendCommandActivated(self):
        '''
        PyQt signal emitted when the "Send Command" action is
        activated in the parameter tree.
        '''
        return self.__view.sendCommandActivated

    @property
    def lastCommand(self):
        '''
        The last command sent by the LaserRelay instance.

        Returns
        -------
        str
            The last command sent by the LaserRelay instance.
        '''
        return self.__relay.lastCommand

    @property
    def view(self):
        '''
        The GUI/view for the laser relay.

        Returns
        -------
        :class:`LaserRelayView`
            The GUI/view for the laser relay.
        '''
        return self.__view

    def portName(self):
        '''
        Get the name of the serial port.

        Returns
        -------
        str
            The name of the serial port.
        '''
        return self.__relay.portName()

    def setPortName(self, value: str):
        '''
        Set the name of the serial port.

        Parameters
        ----------
        value : str
            The name of the serial port.
        '''
        self.__relay.setPortName(value)
        self.__view.set_param_value(RelayParams.PORT, value)

    def baudRate(self):
        '''
        Get the baud rate of the serial port.

        Returns
        -------
        int
            The baud rate of the serial port.
        '''
        return self.__relay.baudRate()

    def setBaudRate(self, value: int):
        '''
        Set the baud rate of the serial port.

        Parameters
        ----------
        value : int
            The baud rate of the serial port.
        '''
        self.__relay.setBaudRate(value)
        self.__view.set_param_value(RelayParams.BAUDRATE, value)

    def isOpen(self):
        '''
        Check if the laser relay is open.

        Returns
        -------
        bool
            True if the laser relay is open, False otherwise.
        '''
        return self.__relay.isOpen()

    def set_config(self):
        '''Sets the serial port configuration.

        This method sets the serial port configuration based on the current
        settings in the parameter tree.
        '''
        if not self.__relay.isOpen():
            self.__relay.setPortName(self.__view.get_param_value(RelayParams.PORT))
            self.__relay.setBaudRate(self.__view.get_param_value(RelayParams.BAUDRATE))

    def connect(self):
        '''
        Connect to the laser relay.

        This method opens the serial port and sets the port state to "open" in the
        parameter tree.
        '''
        self.__relay.open()

        self.__view.set_param_value(RelayParams.PORT_STATE, 'open')

    def isALEX(self) -> bool:
        '''
        Check if the ALEX flag is set.

        Returns
        -------
        bool
            True if the ALEX flag is set, False otherwise.
        '''
        return self.__relay.isALEX()

    def sendCommand(self, config: str):
        '''
        Send a configuration command to the laser relay.

        Parameters
        ----------
        config : str
            The configuration command to send to the laser relay.
        '''
        self.__relay.sendConfig(config)

    def updatePortState(self):
        '''
        Update the port state in the parameter tree.

        This method updates the port state in the parameter
        tree to reflect the current state of the laser relay.
        '''
        color = '#004CB6' if self.isOpen() else 'black'

        self.__view.set_param_value(
            RelayParams.PORT_STATE, 'open' if self.isOpen() else 'closed'
        )

        next(self.__view.get_param(RelayParams.OPEN).items.keys()).button.setStyleSheet(
            f'background-color: {color}'
        )

    def updateHighlight(self, config: str):
        '''
        Update the highlight style of the "Send Command" action.

        This method updates the highlight style of the "Send Command" action
        in the parameter tree based on whether the given configuration matches
        the last command sent.

        Parameters
        ----------
        config : str
            The configuration to compare with the last command sent.
        '''
        style = ''
        if config is not None:
            if config == self.lastCommand:
                style = 'background-color: #004CB6'
            else:
                style = 'background-color: black'

        next(
            self.__view.get_param(RelayParams.SEND_COMMAND).items.keys()
        ).button.setStyleSheet(style)

    def refreshPorts(self):
        """
        Refreshes the available serial ports list in the GUI.

        This method updates the list of available serial ports in the GUI by fetching
        the current list of available ports and setting it as the options for the
        'Serial Port' parameter in the parameter tree.
        """
        if not self.isOpen():
            self.__view.get_param(RelayParams.PORT).setLimits(
                [
                    info.portName()
                    for info in QtSerialPort.QSerialPortInfo.availablePorts()
                ]
            )

    def getCommand(self, config: str):
        '''
        Get the full configuration command to send to the laser relay.

        This method returns the full configuration command to send to the laser relay,
        including the ALEXON/ALEXOFF command based on the current ALEX flag.

        Parameters
        ----------
        config : str
            The configuration command to send to the laser relay,
            excluding the ALEXON/ALEXOFF command.

        Returns
        -------
        str
            The full configuration command to send to the laser relay,
            including the ALEXON/ALEXOFF command.
        '''
        return config + ('ALEXON' if self.isALEX() else 'ALEXOFF') + '\r'

    def __str__(self):
        return 'Laser Relay Controller'
