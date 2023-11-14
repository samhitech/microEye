import os
import sys

import qdarkstyle
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *


class io_single_laser(QSerialPort):
    '''
    MatchBox Class | Inherits QSerialPort
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

    DataReady = pyqtSignal(str, bytes)

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
        '''Sends a specific command to the device and waits for
        the response then emits the DataReady signal.
        The DataReady signal passes the response and command.

        Parameters
        ----------
        command : [bytes]
            command to be sent, please check the constants
            implemented in the MatchBox class.
        '''
        if (self.isOpen()):
            self.write(command)
            self.waitForBytesWritten(500)
            QThread.msleep(delay)
            while self.bytesAvailable() < 5:
                self.waitForReadyRead(500)

            response = str(self.readAll(),
                           encoding='utf8').strip('\r\n').strip()
            if log_print:
                print(response, command)

            self.DataReady.emit(response, command)

            return response

    def OpenCOM(self):
        '''Opens the serial port and initializes the combiner.
        '''
        if not self.isOpen():
            self.open(QIODevice.ReadWrite)
            self.flush()

            if (self.isOpen()):
                self.SendCommand(io_single_laser.ON_DIS)
                self.SendCommand(io_single_laser.START)
                # self.SendCommand(io_single_laser.STATUS)
                self.GetInfo()
                self.SetPower(1)
                self.GetMaxPower()

    def CloseCOM(self):
        '''Closes the serial port.
        '''
        if (self.isOpen()):
            self.SendCommand(io_single_laser.OFF)
            self.waitForBytesWritten(500)

            self.close()

    def GetMaxPower(self):
        if (self.isOpen()):
            res = self.SendCommand(io_single_laser.P_MAX)

            if not ('<ERR>' in res or '<ACK>' in res):
                self.Max_Power = float(res)

    def GetReadings(self, log_print: bool = True):
        if (self.isOpen()):
            res = self.SendCommand(
                io_single_laser.READ,
                log_print)

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

    def GetSettings(self, log_print: bool = True):
        if (self.isOpen()):
            res = self.SendCommand(
                io_single_laser.SETTINGS,
                log_print)

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

    def SetPower(self, value: float) -> bool:
        if (self.isOpen()):
            res = self.SendCommand(
                io_single_laser.P_SET +
                f'{value:.2f}'.encode())

            return '<ACK>' in res

    def GetInfo(self):
        if (self.isOpen()):
            res = self.SendCommand(
                io_single_laser.INFO,
                delay=50)

            if not ('<ERR>' in res or '<ACK>' in res):
                info = res.split('\r\n')

                self.Firmware = info[0]
                self.Serial = info[1].split(':')[1]
                self.Model = info[2].split(':')[1]
                self.Operation_Time = info[3]
                self.ON_Times = info[4]


class SingleLaserWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__()

        self.Laser = io_single_laser()

        self.V_Layout = QFormLayout()

        self.portname_comboBox = QComboBox()    # port name combobox
        self.baudrate_comboBox = QComboBox()    # baudrate combobox

        # adding available serial ports
        for info in QSerialPortInfo.availablePorts():
            self.portname_comboBox.addItem(info.portName())

        # adding default baudrates (default 115200)
        for baudrate in QSerialPortInfo.standardBaudRates():
            if baudrate == 115200:
                self.baudrate_comboBox.addItem(str(baudrate), baudrate)

        self.SetConfigBtn = QPushButton(
            'Set Config.',
            clicked=lambda: self.set_config())

        # IO MatchBox controls
        self.mbox_connect_btn = QPushButton(
            'Connect',
            clicked=lambda: self.laser_connect()
        )
        self.mbox_disconnect_btn = QPushButton(
            'Disconnect',
            clicked=lambda: self.Laser.CloseCOM()
        )

        self.V_Layout.addRow(
            QLabel('Serial Port:'), self.portname_comboBox)
        self.V_Layout.addRow(
            QLabel('Baurate:'), self.baudrate_comboBox)
        self.V_Layout.addRow(self.SetConfigBtn)
        self.V_Layout.addRow(self.mbox_connect_btn)
        self.V_Layout.addRow(self.mbox_disconnect_btn)

        # main vertical layout
        L_Layout = QVBoxLayout()

        # on with cam 1 flash
        self.CAM1 = QRadioButton('CAM 1')
        L_Layout.addWidget(self.CAM1)

        # on with cam 2 flash
        self.CAM2 = QRadioButton('CAM 2')
        L_Layout.addWidget(self.CAM2)

        # off regardless
        self.OFF = QRadioButton('OFF')
        self.OFF.setChecked(True)
        L_Layout.addWidget(self.OFF)

        # on regardless
        self.ON = QRadioButton('ON')
        L_Layout.addWidget(self.ON)

        # self.CAM1.state = "L{:d}F1".format(wavelength)
        # self.CAM2.state = "L{:d}F2".format(wavelength)
        # self.OFF.state = "L{:d}OFF".format(wavelength)
        # self.ON.state = "L{:d}ON".format(wavelength)

        # Create a button group for radio buttons
        self.L_button_group = QButtonGroup()
        self.L_button_group.addButton(self.CAM1, 1)
        self.L_button_group.addButton(self.CAM2, 2)
        self.L_button_group.addButton(self.OFF, 3)
        self.L_button_group.addButton(self.ON, 4)

        self.L_button_group.buttonPressed.connect(self.laser_radio_changed)

        self.V_Layout.addRow(
            QLabel('Status:'), L_Layout)

        # Power control
        self.L_power_label = QLabel('Power (mW):')
        self.L_power = QDoubleSpinBox()
        self.L_power.label = self.L_power_label
        self.L_power.setDecimals(2)
        self.L_power.setMinimum(0)
        self.L_power.setMaximum(100)
        self.L_power.setValue(0)
        self.L_set_curr_btn = QPushButton(
            'Set Power',
            clicked=lambda:
            self.Laser.SetPower(
                self.L_power.value()))

        self.V_Layout.addRow(
            self.L_power_label, self.L_power)
        self.V_Layout.addRow(self.L_set_curr_btn)

        self.R_Temps_Label = QLabel('NA')
        self.S_Temps_Label = QLabel('NA')
        self.R_TEC_Label = QLabel('NA')

        self.V_Layout.addRow(
            QLabel('Temp. Set (LD, Crystal, Fan):'), self.S_Temps_Label)
        self.V_Layout.addRow(
            QLabel('Temp. Read (LD, Crystal, Body):'), self.R_Temps_Label)
        self.V_Layout.addRow(
            QLabel('TEC Load (LD, Crystal):'), self.R_TEC_Label)

        self.setLayout(self.V_Layout)

        # Statues Bar Timer
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start()

    def set_config(self):
        if not self.Laser.isOpen():
            self.Laser.setPortName(
                self.portname_comboBox.currentText())
            self.Laser.setBaudRate(
                self.baudrate_comboBox.currentData())

    def laser_connect(self):
        self.Laser.OpenCOM()

        wavelength = self.Laser.Model.split('L')[0]

        self.CAM1.state = f'L{wavelength}F1'
        self.CAM2.state = f'L{wavelength}F2'
        self.OFF.state = f'L{wavelength}OFF'
        self.ON.state = f'L{wavelength}ON'

    def update_stats(self):
        if self.Laser.isOpen():
            self.Laser.GetReadings(False)
            self.Laser.GetSettings(False)

            self.L_power.setMaximum(self.Laser.Max_Power)
            self.L_power_label.setText(
                f'Power {self.Laser.S_Power:.2f} (mW):')
            self.R_Temps_Label.setText(
                '{:.2f} C, {:.2f} C, {:.2f} C'.format(
                    self.Laser.R_Diode_Temp,
                    self.Laser.R_Crystal_Temp,
                    self.Laser.R_Body_Temp))
            self.S_Temps_Label.setText(
                '{:.2f} C, {:.2f} C, {:.2f} C'.format(
                    self.Laser.S_Diode_Temp,
                    self.Laser.S_Crystal_Temp,
                    self.Laser.S_Fan_Temp))
            self.R_TEC_Label.setText(
                f'{self.Laser.R_LD_TEC_Load} , {self.Laser.R_Crystal_TEC_Load}')

            self.setTitle(self.Laser.Model)
            self.mbox_connect_btn.setStyleSheet('background-color: green')
        else:
            self.mbox_connect_btn.setStyleSheet('background-color: red')

        self.RefreshPorts()

    def RefreshPorts(self):
        avPorts = QSerialPortInfo.availablePorts()
        if self.portname_comboBox.count() > len(avPorts):
            self.portname_comboBox.clear()
            for info in avPorts:
                self.portname_comboBox.addItem(info.portName())
        else:
            for info in avPorts:
                if self.portname_comboBox.findText(info.portName()) == -1:
                    self.portname_comboBox.addItem(info.portName())

    def laser_radio_changed(self, object):
        '''Sends enable/disable signals to the
        laser combiner according to selected setting

        Parameters
        ----------
        object : [QRadioButton]
            the radio button toggled
        '''
        if ('OFF' in object.state):
            self.Laser.SendCommand(io_single_laser.ON_DIS)
        else:
            self.Laser.SendCommand(io_single_laser.ON_EN)

    def GetRelayState(self):
        try:
            return self.L_button_group.checkedButton().state
        except Exception:
            return ''

    def StartGUI():
        '''Initializes a new QApplication and control_module.

        Use
        -------
        app, window = control_module.StartGUI()

        app.exec_()

        Returns
        -------
        tuple (QApplication, microEye.control_module)
            Returns a tuple with QApp and control_module main window.
        '''
        # create a QApp
        app = QApplication(sys.argv)
        # set darkmode from *qdarkstyle* (not compatible with pyqt6)
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
        # sets the app icon
        dirname = os.path.dirname(os.path.abspath(__file__))
        app_icon = QIcon()
        app_icon.addFile(
            os.path.join(dirname, '../icons/16.png'), QSize(16, 16))
        app_icon.addFile(
            os.path.join(dirname, '../icons/24.png'), QSize(24, 24))
        app_icon.addFile(
            os.path.join(dirname, '../icons/32.png'), QSize(32, 32))
        app_icon.addFile(
            os.path.join(dirname, '../icons/48.png'), QSize(48, 48))
        app_icon.addFile(
            os.path.join(dirname, '../icons/64.png'), QSize(64, 64))
        app_icon.addFile(
            os.path.join(dirname, '../icons/128.png'), QSize(128, 128))
        app_icon.addFile(
            os.path.join(dirname, '../icons/256.png'), QSize(256, 256))
        app_icon.addFile(
            os.path.join(dirname, '../icons/512.png'), QSize(512, 512))

        app.setWindowIcon(app_icon)

        if sys.platform.startswith('win'):
            import ctypes
            myappid = 'samhitech.mircoEye.control_module'  # appid
            ctypes.windll.shell32.\
                SetCurrentProcessExplicitAppUserModelID(myappid)

        widget = SingleLaserWidget()
        widget.show()
        return app, widget


if __name__ == '__main__':
    app, widget = SingleLaserWidget.StartGUI()

    app.exec_()
