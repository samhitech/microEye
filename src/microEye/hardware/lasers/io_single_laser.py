from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.hardware.lasers.io_params import LaserState, MB_Params
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils import StartGUI, Tree


class io_single_laser(QtSerialPort.QSerialPort):
    '''
    Class representing a single laser MatchBox device | Inherits QSerialPort
    '''

    INFO = b'r i'
    '''Get MatchBox Info
    '''
    ON_DIS = b'e 2'
    '''Enable the combiner
    '''
    OFF = b'e 0'
    '''Disable the combiner
    '''
    READ = b'r r'
    SETTINGS = b'r s'
    ON_EN = b'e 1'
    '''Enable 1st Laser Diode
    '''

    MAX_CUR = b'Lm?'
    '''Get maximum current
    '''
    P_SET = b'c 4 '
    '''Set the power value in mW
    '''
    P_MAX = b'r 4'
    '''Get the max power value
    '''
    STATUS = b'Le'
    '''Get the laser diodes enabled(1)/disabled(0) states
    '''
    # START = b'c u 2 35488'
    START = b'c u 1 1234'
    '''TBA
    '''

    DataReady = Signal(str, bytes)

    Max_Power = 0.0

    R_Diode_Temp = 0.0
    R_Crystal_Temp = 0.0
    R_Body_Temp = 0.0

    R_LD_Current = '0.0mA'

    R_Crystal_TEC_Load = '0%'
    R_LD_TEC_Load = '0%'

    R_Status = 'OFF'

    R_Fan_Load = '0%'

    R_Input_Voltage = '5V'

    S_Crystal_Temp = 0.0
    S_Diode_Temp = 0.0

    S_LD_Current = '0.0mA'
    S_Feedback_DAC = 0

    S_Power = 0.0

    S_LD_MaxCurrent = 0.0

    S_Autostart_Mode = 'OFF'

    S_Access_Level = 1

    S_Fan_Temp = 0.0

    Firmware = ''
    Serial = ''
    Model = ''
    Operation_Time = ''
    ON_Times = ''

    def SendCommand(self, command, log_print: bool = True, delay: int = 1):
        '''
        Sends a specific command to the device and waits for the response.
        Then emits the DataReady signal with the response and command as arguments.

        Parameters
        ----------
        command : bytes
            Command to be sent to the device.
        log_print : bool, optional
            If True, prints the response and command to the console.
        delay : int, optional
            Delay in milliseconds before reading the response.
        '''
        if self.isOpen():
            self.write(command)
            self.waitForBytesWritten(500)
            QtCore.QThread.msleep(delay)
            while self.bytesAvailable() < 5:
                self.waitForReadyRead(500)

            response = str(self.readAll(), encoding='utf8').strip('\r\n').strip()
            if log_print:
                print(response, command)

            self.DataReady.emit(response, command)

            return response

    def OpenCOM(self):
        '''Opens the serial port and initializes the combiner.'''
        if not self.isOpen():
            self.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)
            self.flush()

            if self.isOpen():
                self.SendCommand(io_single_laser.ON_DIS)
                self.SendCommand(io_single_laser.START)
                # self.SendCommand(io_single_laser.STATUS)
                self.GetInfo()
                self.GetMaxPower()
                # self.SetPower(1)

    def CloseCOM(self):
        '''Closes the serial port.'''
        if self.isOpen():
            self.SendCommand(io_single_laser.OFF)
            self.waitForBytesWritten(500)

            self.close()

    def GetMaxPower(self):
        '''
        Gets the maximum power that can be set for the laser diode.

        This method sends the `P_MAX` command to the device and waits for the response.
        If the response does not contain an error message,
        the method sets the `Max_Power` attribute of the `io_single_laser`
        instance to the value contained in the response.
        '''
        if self.isOpen():
            res = self.SendCommand(io_single_laser.P_MAX)

            if not ('<ERR>' in res or '<ACK>' in res):
                self.Max_Power = float(res)

    def GetReadings(self, log_print: bool = True):
        '''
        Gets the current readings from the device.

        This method first checks if the serial port is open. If it is, it sends the
        `READ` command to the device and waits for the response. If the response
        does not contain an error message, the method sets the various attributes
        of the `io_single_laser` instance with the values contained in the
        response.
        '''
        if self.isOpen():
            res = self.SendCommand(io_single_laser.READ, log_print)

            if not ('<ERR>' in res or '<ACK>' in res):
                readings = res.split(' ')
                if len(readings) >= 10:
                    self.R_Diode_Temp = float(readings[1])
                    self.R_Crystal_Temp = float(readings[2])
                    self.R_Body_Temp = float(readings[3])
                    self.R_LD_Current = readings[4]
                    self.R_Crystal_TEC_Load = readings[5]
                    self.R_LD_TEC_Load = readings[6]
                    self.R_Status = readings[7]
                    self.R_Fan_Load = readings[8]
                    self.R_Input_Voltage = readings[9]

                    return {
                        MB_Params.LD_TEMP: self.R_Diode_Temp,
                        MB_Params.CRYSTAL_TEMP: self.R_Crystal_Temp,
                        MB_Params.BODY_TEMP: self.R_Body_Temp,
                        MB_Params.LD_CURRENT: self.R_LD_Current,
                        MB_Params.CRYSTAL_TEC_LOAD: self.R_Crystal_TEC_Load,
                        MB_Params.LD_TEC_LOAD: self.R_LD_TEC_Load,
                        MB_Params.STATUS: self.R_Status,
                        MB_Params.FAN_LOAD: self.R_Fan_Load,
                        MB_Params.IN_VOLTAGE: self.R_Input_Voltage,
                    }

        return {}

    def GetSettings(self, log_print: bool = True):
        '''Gets the current settings from the device.

        This method first checks if the serial port is open. If it is, it sends the
        `SETTINGS` command to the device and waits for the response. If the response
        does not contain an error message, the method sets the various attributes
        of the `io_single_laser` instance with the values contained in the
        response.
        '''
        if self.isOpen():
            res = self.SendCommand(io_single_laser.SETTINGS, log_print)

            if not ('<ERR>' in res or '<ACK>' in res):
                readings = res.split(' ')
                if len(readings) >= 10:
                    self.S_Crystal_Temp = float(readings[1]) / 100
                    self.S_Diode_Temp = float(readings[2]) / 100
                    self.S_LD_Current = float(readings[3])
                    self.S_Feedback_DAC = float(readings[4])
                    self.S_Power = float(readings[5])
                    self.S_LD_MaxCurrent = float(readings[6])
                    self.S_Autostart_Mode = readings[7]
                    self.S_Access_Level = int(readings[8])
                    self.S_Fan_Temp = float(readings[9]) / 100

                    return {
                        MB_Params.CRYSTAL_TEMP_SET: self.S_Crystal_Temp,
                        MB_Params.LD_TEMP_SET: self.S_Diode_Temp,
                        MB_Params.LD_CURRENT_SET: self.S_LD_Current,
                        MB_Params.FEEDBACK_DAC: self.S_Feedback_DAC,
                        MB_Params.POWER_READ: self.S_Power,
                        MB_Params.LD_CURRENT_MAX: self.S_LD_MaxCurrent,
                        MB_Params.AUTO_MODE: self.S_Autostart_Mode,
                        MB_Params.ACCESS_LEVEL: self.S_Access_Level,
                        MB_Params.FAN_TEMP_SET: self.S_Fan_Temp,
                    }

        return {}

    def SetPower(self, value: float) -> bool:
        '''Sets the power level of the laser diode.

        This method sends the `P_SET` command to the device followed by the desired
        power level (in milliwatts) as a floating point number. It then waits for
        the response and returns a boolean indicating whether the command was
        successful (`True`) or not (`False`).

        Parameters
        ----------
        value : float
            The desired power level in milliwatts.

        Returns
        -------
        bool
            `True` if the command was successful, `False` otherwise.
        '''
        if self.isOpen():
            if not isinstance(value, (int, float)):
                raise TypeError('Power must be a number!')
            if value < 0 or value > self.Max_Power:
                raise ValueError(f'Power must be between 0 and {self.Max_Power}')

            res = self.SendCommand(io_single_laser.P_SET + f'{value:.2f}'.encode())

            return '<ACK>' in res

    def GetInfo(self):
        '''Gets the information about the device.

        This method sends the `INFO` command to the device and waits for the response.
        If the response does not contain an error message, the method sets the
        `Firmware`, `Serial`, `Model`, `Operation_Time`, and `ON_Times`
        attributes of the `io_single_laser` instance with the values contained in
        the response.
        '''
        if self.isOpen():
            res = self.SendCommand(io_single_laser.INFO, delay=50)

            if not ('<ERR>' in res or '<ACK>' in res):
                info = res.split('\r\n')

                self.Firmware = info[0]
                self.Serial = info[1].split(':')[1]
                self.Model = info[2].split(':')[1]
                self.Operation_Time = info[3]
                self.ON_Times = info[4]


class SingleMatchBox(Tree):
    '''A class representing a single MatchBox device.

    This class provides a high-level interface for controlling a single MatchBox
    device. It uses the `io_single_laser` class to communicate with the device
    over a serial port.

    Attributes
    ----------
    Laser : io_single_laser
        The `io_single_laser` instance used to communicate with the device.

    Methods
    -------
    create_parameters()
        Creates the parameter tree for the MatchBox GUI.

    update_stats()
        Updates the statistics displayed in the MatchBox GUI.

    set_config()
        Sets the serial port configuration.

    laser_connect()
        Connects to the MatchBox device.

    laser_disconnect()
        Disconnects from the MatchBox device.

    set_power(value)
        Sets the power level of the laser diode.

    get_relay_state()
        Gets the state of the relay.
    '''

    PARAMS = MB_Params
    removed = Signal(object)

    def __init__(self, parent: Optional['QtWidgets.QWidget'] = None):
        '''Initializes a new SingleMatchBox instance.

        Parameters
        ----------
        parent : Optional[QWidget]
            The parent widget for this SingleMatchBox instance.

        Attributes
        ----------
        Laser : io_single_laser
            The `io_single_laser` instance used to communicate with the device.
        '''
        super().__init__(parent=parent)

        self.Laser = io_single_laser()

        self.wavelength = '000'

        # Statues Bar Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start()

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {
                'name': str(MB_Params.MODEL),
                'type': 'str',
                'value': 'N/A',
                'readonly': True,
            },
            {
                'name': str(MB_Params.WAVELENGTH),
                'type': 'int',
                'value': 0,
                'suffix': 'nm',
                'readonly': True,
            },
            {
                'name': str(MB_Params.SERIAL_PORT),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(MB_Params.PORT),
                        'type': 'list',
                        'limits': [
                            info.portName()
                            for info in QtSerialPort.QSerialPortInfo.availablePorts()
                        ],
                    },
                    {
                        'name': str(MB_Params.BAUDRATE),
                        'type': 'list',
                        'value': 115200,
                        'limits': [
                            baudrate
                            for baudrate in QtSerialPort.QSerialPortInfo.standardBaudRates()
                        ],
                    },
                    {'name': str(MB_Params.SET_PORT), 'type': 'action'},
                    {'name': str(MB_Params.OPEN), 'type': 'action'},
                    {'name': str(MB_Params.CLOSE), 'type': 'action'},
                    {
                        'name': str(MB_Params.PORT_STATE),
                        'type': 'str',
                        'value': 'closed',
                        'readonly': True,
                    },
                ],
            },
            {
                'name': str(MB_Params.OPTIONS),
                'type': 'group',
                'children': [
                    {
                        'name': str(MB_Params.STATE),
                        'type': 'list',
                        'value': LaserState.OFF,
                        'limits': LaserState.get_list(),
                    },
                    {
                        'name': str(MB_Params.POWER),
                        'type': 'float',
                        'value': 0,
                        'limits': [0, 100],
                        'step': 0.1,
                        'dec': False,
                    },
                    {'name': str(MB_Params.SET_POWER), 'type': 'action'},
                ],
            },
            {
                'name': str(MB_Params.READINGS),
                'type': 'group',
                'children': [
                    {
                        'name': str(MB_Params.POWER_READ),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.LD_CURRENT),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.LD_CURRENT_SET),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.LD_CURRENT_MAX),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.LD_TEMP),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.LD_TEMP_SET),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.LD_TEC_LOAD),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.CRYSTAL_TEMP),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.CRYSTAL_TEMP_SET),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.CRYSTAL_TEC_LOAD),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.BODY_TEMP),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.FAN_TEMP_SET),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.FAN_LOAD),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.FEEDBACK_DAC),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.AUTO_MODE),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.ACCESS_LEVEL),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.STATUS),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.IN_VOLTAGE),
                        'type': 'str',
                        'value': '0',
                        'readonly': True,
                    },
                ],
            },
            {
                'name': str(MB_Params.INFO),
                'type': 'group',
                'children': [
                    {
                        'name': str(MB_Params.FIRMWARE),
                        'type': 'str',
                        'value': 'N/A',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.SERIAL),
                        'type': 'str',
                        'value': 'N/A',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.OPERATION_TIME),
                        'type': 'str',
                        'value': 'N/A',
                        'readonly': True,
                    },
                    {
                        'name': str(MB_Params.ON_TIMES),
                        'type': 'str',
                        'value': 'N/A',
                        'readonly': True,
                    },
                ],
            },
            {'name': str(MB_Params.REMOVE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(MB_Params.STATE).sigValueChanged.connect(
            lambda: self.laser_state_changed()
        )

        self.get_param(MB_Params.SET_PORT).sigActivated.connect(self.set_config)
        self.get_param(MB_Params.OPEN).sigActivated.connect(self.laser_connect)
        self.get_param(MB_Params.CLOSE).sigActivated.connect(
            lambda: self.Laser.CloseCOM()
        )
        self.get_param(MB_Params.SET_POWER).sigActivated.connect(self.set_power)

        self.get_param(MB_Params.REMOVE).sigActivated.connect(self.remove_widget)

    def set_config(self):
        '''Sets the serial port configuration.

        This method sets the serial port configuration based on the current
        settings in the parameter tree.
        '''
        if not self.Laser.isOpen():
            self.Laser.setPortName(self.get_param_value(MB_Params.PORT))
            self.Laser.setBaudRate(self.get_param_value(MB_Params.BAUDRATE))

    def laser_connect(self):
        '''Connects to the MatchBox device.

        This method opens the serial port and initializes the laser.
        '''
        self.Laser.OpenCOM()

        self.wavelength = self.Laser.Model.split('L')[0]
        self.get_param(MB_Params.POWER).setLimits([0, self.Laser.Max_Power])

        self.set_param_value(MB_Params.MODEL, self.Laser.Model)
        self.set_param_value(MB_Params.WAVELENGTH, int(self.wavelength))
        self.set_param_value(MB_Params.FIRMWARE, self.Laser.Firmware)
        self.set_param_value(MB_Params.SERIAL, self.Laser.Serial)

    def update_stats(self):
        """
        Updates the statistics displayed in the MatchBox GUI.

        This method retrieves the current readings and settings from the device, and
        updates the corresponding parameter tree items with the new values. It also
        sets the limits of the power parameter based on the maximum power that can be
        set for the laser diode.

        If the laser is not connected, the method sets the 'Port State' parameter to
        'closed'.
        """
        if self.Laser.isOpen():
            readings = self.Laser.GetReadings(False)
            settings = self.Laser.GetSettings(False)

            for key, value in readings.items():
                if isinstance(value, (int, float)):
                    self.set_param_value(key, f'{value:.2f}')
                elif isinstance(value, str):
                    self.set_param_value(key, value)

            for key, value in settings.items():
                if isinstance(value, (int, float)):
                    self.set_param_value(key, f'{value:.2f}')
                elif isinstance(value, str):
                    self.set_param_value(key, value)

            self.set_param_value(MB_Params.OPERATION_TIME, self.Laser.Operation_Time)
            self.set_param_value(MB_Params.ON_TIMES, self.Laser.ON_Times)

            self.set_param_value(MB_Params.PORT_STATE, 'open')
        else:
            self.set_param_value(MB_Params.PORT_STATE, 'closed')

        self.RefreshPorts()

    def RefreshPorts(self):
        """
        Refreshes the available serial ports list in the GUI.

        This method updates the list of available serial ports in the GUI by fetching
        the current list of available ports and setting it as the options for the
        'Serial Port' parameter in the parameter tree.
        """
        if not self.Laser.isOpen():
            self.get_param(MB_Params.PORT).setLimits(
                [
                    info.portName()
                    for info in QtSerialPort.QSerialPortInfo.availablePorts()
                ]
            )

    def laser_state_changed(self):
        '''Sends enable/disable signals to the
        laser combiner according to selected state setting
        '''
        if self.get_param_value(MB_Params.STATE) == LaserState.OFF:
            self.Laser.SendCommand(io_single_laser.ON_DIS)
        else:
            self.Laser.SendCommand(io_single_laser.ON_EN)

    def set_power(self):
        '''
        Sets the power level of the laser diode
        based on the current value in the GUI
        '''
        self.Laser.SetPower(self.get_param_value(MB_Params.POWER))

    def GetRelayState(self):
        '''
        Returns the current state as a string to send to the laser relay box.
        '''
        try:
            return f'L{self.wavelength}{self.get_param_value(MB_Params.STATE)}'
        except Exception:
            return ''

    def get_config(self) -> dict:
        '''
        Returns the current configuration of the laser device as a dictionary.
        '''
        return {
            'port': self.Laser.portName(),
            'baudrate': self.Laser.baudRate(),
            'class': self.__class__.__name__,
        }

    def load_config(self, config: dict) -> bool:
        '''
        Loads the configuration from the given dictionary.
        '''
        port = config.get('port')
        baudrate = config.get('baudrate')
        if port:
            self.Laser.setPortName(port)
            self.set_param_value(MB_Params.PORT, port)
        if baudrate:
            self.Laser.setBaudRate(baudrate)
            self.set_param_value(MB_Params.BAUDRATE, baudrate)

    def remove_widget(self):
        if self.parent() and not self.Laser.isOpen():
            self.parent().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Disconnect device {self.Laser.Model} before removing!')

    def __str__(self):
        return self.Laser.Model.strip() or self.__class__.__name__

    def StartGUI():
        '''Initializes a new QApplication and SingleMatchBox.

        Use
        -------
        app, window = SingleMatchBox.StartGUI()

        app.exec()

        Returns
        -------
        tuple (QApplication, SingleMatchBox)
            Returns a tuple with QApp and SingleMatchBox main window.
        '''
        return StartGUI(SingleMatchBox)


if __name__ == '__main__':
    app, widget = SingleMatchBox.StartGUI()

    app.exec()
