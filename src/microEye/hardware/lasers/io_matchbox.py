from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.hardware.lasers.io_params import LaserState, MB_Params
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils import StartGUI, Tree


class io_combiner(QtSerialPort.QSerialPort):
    '''
    Class representing a laser combiner MatchBox device | Inherits QSerialPort
    '''

    INFO = b'r i'
    '''Get MatchBox Info
    '''
    ON = b'e 1'
    '''Enable the combiner
    '''
    OFF = b'e 0'
    '''Disable the combiner
    '''
    READ = b'r r'
    SETTINGS = b'r s'

    ENABLE_1 = b'L1E'
    '''Enable 1st Laser Diode
    '''
    ENABLE_2 = b'L2E'
    '''Enable 2nd Laser Diode
    '''
    ENABLE_3 = b'L3E'
    '''Enable 3rd Laser Diode
    '''
    ENABLE_4 = b'L4E'
    '''Enable 4th Laser Diode
    '''
    DISABLE_1 = b'L1D'
    '''Disable 1st Laser Diode
    '''
    DISABLE_2 = b'L2D'
    '''Disable 2nd Laser Diode
    '''
    DISABLE_3 = b'L3D'
    '''Disable 3rd Laser Diode
    '''
    DISABLE_4 = b'L4D'
    '''Disable 4th Laser Diode
    '''

    GET_WAVELENGTHS = b'Ln?'
    '''Get available wavelengths
    '''

    MAX_CUR = b'Lm?'
    '''Get maximum current
    '''
    CUR_SET = b'Lc?'
    '''Get the current set value
    '''
    CUR_CUR = b'Lr'
    '''Get the current reading
    '''
    STATUS = b'Le'
    '''Get the laser diodes enabled(1)/disabled(0) states
    '''
    START = b'c u 2 35488'
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

    Current = [0, 0, 0, 0]
    Setting = [0, 0, 0, 0]
    Max = [0, 0, 0, 0]
    Wavelengths = [0, 0, 0, 0]

    def SendCommand(self, command, log_print: bool = True, delay: int = 1):
        '''Sends a specific command to the device and waits for
        the response then emits the DataReady signal.
        The DataReady signal passes the response and command.

        Parameters
        ----------
        command : [bytes]
            command to be sent, please check the constants
            implemented in the MatchBox class.
        '''
        if self.isOpen():
            self.readAll()
            self.write(command)
            self.waitForBytesWritten(500)
            QtCore.QThread.msleep(delay)
            while self.bytesAvailable() < 5:
                self.waitForReadyRead(500)

            response = (
                str(
                    self.readAll().replace(b'\xff', b''),
                    encoding='utf-8',
                    errors='replace',
                )
                .strip('\r\n')
                .strip()
            )
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
                self.SendCommand(io_combiner.ON)
                self.SendCommand(io_combiner.START)
                # self.SendCommand(io_single_laser.STATUS)
                self.GetInfo()
                self.GetWavelengths()
                self.SetCurrent(1, 0)
                self.SetCurrent(2, 0)
                self.SetCurrent(3, 0)
                self.SetCurrent(4, 0)
                self.GetMaxCurrent()

    def CloseCOM(self):
        '''Closes the serial port.'''
        if self.isOpen():
            self.SendCommand(io_combiner.OFF)
            self.waitForBytesWritten(500)

            self.close()

    def GetMaxCurrent(self):
        if self.isOpen():
            res = self.SendCommand(io_combiner.MAX_CUR)

            if not ('<ERR>' in res or '<ACK>' in res):
                values = res.strip('<').strip('>').split(' ')
                if len(values) >= 4:
                    self.Max = list(map(int, values))

    def GetCurrent(self, log_print: bool = False):
        if self.isOpen():
            res = self.SendCommand(io_combiner.CUR_CUR, log_print)

            if not ('<ERR>' in res or '<ACK>' in res):
                values = res.strip('<').strip(' mA>').split(' ')
                if len(values) >= 4:
                    self.Current = list(map(float, values))

    def GetSetCurrent(self, log_print: bool = False):
        if self.isOpen():
            res = self.SendCommand(io_combiner.CUR_SET, log_print)

            if not ('<ERR>' in res or '<ACK>' in res):
                values = res.strip('<').strip('>').split(' ')
                if len(values) >= 4:
                    self.Setting = list(map(int, values))

    def GetWavelengths(self):
        if self.isOpen():
            res = self.SendCommand(io_combiner.GET_WAVELENGTHS)

            if not ('<ERR>' in res or '<ACK>' in res):
                values = res.strip('<').strip(' >').strip().split(' ')
                self.Wavelengths = []
                if len(values) == 3:
                    self.Wavelengths.append(0)
                for x in range(len(values)):
                    if values[x].isdigit():
                        self.Wavelengths.append(int(values[x]))
                    else:
                        self.Wavelengths.append(0)
                # if len(values) >= 3:
                #     self.Wavelengths = list(map(int, values))

    def GetReadings(self, log_print: bool = True):
        if self.isOpen():
            res = self.SendCommand(io_combiner.READ, log_print)

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
        if self.isOpen():
            res = self.SendCommand(io_combiner.SETTINGS, log_print)

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

    def SetCurrent(self, index: int, value: int) -> bool:
        if self.isOpen():
            if not isinstance(value, int):
                raise TypeError('Current must be an int!')
            if value < 0 or value > self.Max[index - 1]:
                raise ValueError(f'Current must be between 0 and {self.Max[index-1]}')

            res = self.SendCommand(f'Lc{index:.0f} {value:.0f}'.encode())

            return '<ACK>' in res

    def SetDisabled(self, index: int) -> bool:
        if self.isOpen():
            res = self.SendCommand(f'L{index:.0f}D'.encode())

            return '<ACK>' in res

    def SetEnabled(self, index: int) -> bool:
        if self.isOpen():
            res = self.SendCommand(f'L{index:.0f}E'.encode())

            return '<ACK>' in res

    def GetInfo(self):
        if self.isOpen():
            res = self.SendCommand(io_combiner.INFO, delay=50)

            if not ('<ERR>' in res or '<ACK>' in res):
                info = res.split('\r\n')

                self.Firmware = info[0]
                self.Serial = info[1].split(':')[1]
                self.Model = info[2].split(':')[1]
                self.Operation_Time = info[3]
                self.ON_Times = info[4]


class LaserSwitches:
    def __init__(self, Laser: io_combiner, index=1, wavelength=638) -> None:
        super().__init__()

        self.Laser = Laser
        self.index = index
        self.wavelength = wavelength
        self.state = LaserState.OFF
        self.max_current = Laser.Max[index - 1]

        self.STATE = f'Options.{self.wavelength:d}nm State'
        self.CURRENT = f'Options.{self.wavelength:d}nm Current [mA]'

    def laser_state_changed(self, param: Parameter, value):
        '''Sends enable/disable signals to the
        laser combiner according to selected setting

        Parameters
        ----------
        object : [QRadioButton]
            the radio button toggled
        '''
        self.state = value
        if self.state == LaserState.OFF:
            self.Laser.SetDisabled(self.index)
        else:
            self.Laser.SetEnabled(self.index)

    def SetCurrent(self, value):
        self.Laser.SetCurrent(self.index, value)

    def GetRelayState(self):
        return f'L{self.wavelength:d}{self.state}'

    def add_params(self, parent: Tree):
        params = [
            {
                'name': self.STATE.split('.')[1],
                'type': 'list',
                'value': LaserState.OFF,
                'limits': LaserState.get_list(),
            },
            {
                'name': self.CURRENT.split('.')[1],
                'type': 'int',
                'value': 0,
                'limits': [0, self.max_current],
            },
        ]
        for param in params:
            parent.add_param_child(MB_Params.OPTIONS, param)

        parent.get_param(self.STATE).sigValueChanged.connect(self.laser_state_changed)


class CombinerLaserWidget(Tree):
    PARAMS = MB_Params
    removed = Signal(object)

    def __init__(self, parent: Optional['QtWidgets.QWidget'] = None):
        '''Initializes a new CombinerLaserWidget instance.

        Parameters
        ----------
        parent : Optional[QWidget]
            The parent widget for this CombinerLaserWidget instance.

        Attributes
        ----------
        Laser : io_combiner
            The `io_combiner` instance used to communicate with the device.
        '''
        super().__init__()

        self.Laser = io_combiner()

        self._laserSwitches: list[LaserSwitches] = []

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
                'name': str(MB_Params.WAVELENGTHS),
                'type': 'str',
                'value': 'N/A',
                'readonly': True,
            },
            {
                'name': str(MB_Params.SERIAL_PORT),
                'type': 'group',
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
                            for baudrate in \
                                QtSerialPort.QSerialPortInfo.standardBaudRates()
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
                    {'name': str(MB_Params.SET_CURRENT), 'type': 'action'},
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
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

        self.get_param(MB_Params.SET_PORT).sigActivated.connect(self.set_config)
        self.get_param(MB_Params.OPEN).sigActivated.connect(self.laser_connect)
        self.get_param(MB_Params.CLOSE).sigActivated.connect(
            lambda: self.Laser.CloseCOM()
        )
        self.get_param(MB_Params.SET_CURRENT).sigActivated.connect(self.set_current)

        self.get_param(MB_Params.REMOVE).sigActivated.connect(self.remove_widget)

    def laser_connect(self):
        '''Connects to the MatchBox device.

        This method opens the serial port and initializes the laser.
        '''
        self.Laser.OpenCOM()

        self.set_param_value(MB_Params.MODEL, self.Laser.Model)
        self.set_param_value(MB_Params.WAVELENGTHS, str(self.Laser.Wavelengths))
        self.set_param_value(MB_Params.FIRMWARE, self.Laser.Firmware)
        self.set_param_value(MB_Params.SERIAL, self.Laser.Serial)

        if not self._laserSwitches:
            for idx in range(len(self.Laser.Wavelengths)):
                if self.Laser.Wavelengths[idx] > 0:
                    switch = LaserSwitches(
                        self.Laser, idx + 1, self.Laser.Wavelengths[idx]
                    )
                    switch.add_params(self)
                    self._laserSwitches.append(switch)

    def set_config(self):
        '''Sets the serial port configuration.

        This method sets the serial port configuration based on the current
        settings in the parameter tree.
        '''
        if not self.Laser.isOpen():
            self.Laser.setPortName(self.get_param_value(MB_Params.PORT))
            self.Laser.setBaudRate(self.get_param_value(MB_Params.BAUDRATE))

    def set_current(self):
        for switch in self._laserSwitches:
            value = self.get_param(switch.CURRENT).value()
            switch.SetCurrent(value)

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
            self.Laser.GetCurrent()

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

            self.set_param_value(
                MB_Params.LD_CURRENT,
                '{:.2f} mA, {:.2f} mA, {:.2f} mA, {:.2f} mA'.format(
                    *self.Laser.Current
                ),
            )
            self.set_param_value(
                MB_Params.LD_CURRENT_MAX,
                '{:.2f} mA, {:.2f} mA, {:.2f} mA, {:.2f} mA'.format(*self.Laser.Max),
            )
            self.set_param_value(
                MB_Params.LD_CURRENT_SET,
                '{:.2f} mA, {:.2f} mA, {:.2f} mA, {:.2f} mA'.format(
                    *self.Laser.Setting
                ),
            )

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

    def GetRelayState(self):
        states = ''
        for switch in self._laserSwitches:
            states += switch.GetRelayState()
        return states

    def remove_widget(self):
        if self.parent() and not self.Laser.isOpen():
            self.parent().layout().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Disconnect device {self.Laser.Model} before removing!')

    def __str__(self):
        return self.Laser.Model.strip() or self.__class__.__name__

    def StartGUI():
        '''Initializes a new QApplication and CombinerLaserWidget.

        Use
        -------
        app, window = CombinerLaserWidget.StartGUI()

        app.exec()

        Returns
        -------
        tuple (QApplication, CombinerLaserWidget)
            Returns a tuple with QApp and CombinerLaserWidget main window.
        '''
        return StartGUI(CombinerLaserWidget)


if __name__ == '__main__':
    app, widget = CombinerLaserWidget.StartGUI()

    app.exec()
