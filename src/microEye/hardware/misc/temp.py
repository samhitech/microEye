from os import name

import numpy as np
from pyqtgraph.widgets.PlotWidget import PlotWidget

from microEye.hardware.port_config import port_config
from microEye.qt import (
    QApplication,
    QDateTime,
    QMainWindow,
    QtCore,
    QtSerialPort,
    QtWidgets,
    Slot,
)
from microEye.utils import StartGUI


class temperature_monitor(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Define main window layout
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.temp_log_X = np.arange(100)
        self.temp_log = np.array([20.0] * 100)

        self.serial = QtSerialPort.QSerialPort(None, readyRead=self.receive)
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM4')

        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.temp_log_graph = PlotWidget()
        self.temp_log_graph.setLabel('bottom', 'Frame', **self.labelStyle)
        self.temp_log_graph.setLabel('left', 'Temperature [C]', **self.labelStyle)
        self.temp_log_graph_ref = None

        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.connect_btn = QtWidgets.QPushButton(
            'Connect', clicked=lambda: self.OpenCOM()
        )
        self.disconnect_btn = QtWidgets.QPushButton(
            'Disconnect', clicked=lambda: self.CloseCOM()
        )
        self.config_btn = QtWidgets.QPushButton(
            'Config.', clicked=lambda: self.open_dialog(self.serial)
        )

        self.buttons_layout.addWidget(self.connect_btn)
        self.buttons_layout.addWidget(self.disconnect_btn)
        self.buttons_layout.addWidget(self.config_btn)

        self.settings_group = QtWidgets.QGroupBox('Settings')
        self.settings_layout = QtWidgets.QFormLayout()
        self.settings_group.setLayout(self.settings_layout)
        self.settings_group.setMinimumWidth(250)

        self.setT = QtWidgets.QDoubleSpinBox()
        self.setT.setMinimum(-2000)
        self.setT.setMaximum(10000)
        self.setT.setValue(10)

        self.kP = QtWidgets.QDoubleSpinBox()
        self.kP.setMinimum(-2000)
        self.kP.setMaximum(10000)
        self.kP.setValue(90)
        self.kI = QtWidgets.QDoubleSpinBox()
        self.kI.setMinimum(-2000)
        self.kI.setMaximum(10000)
        self.kI.setValue(0)
        self.kD = QtWidgets.QDoubleSpinBox()
        self.kD.setMinimum(-2000)
        self.kD.setMaximum(10000)
        self.kD.setValue(0)

        self.send_btn = QtWidgets.QPushButton(
            'Send Settings', clicked=lambda: self.SendCommand()
        )

        self.settings_layout.addRow(self.connect_btn)
        self.settings_layout.addRow(self.disconnect_btn)
        self.settings_layout.addRow(self.config_btn)
        self.settings_layout.addRow('Set T [C]:', self.setT)
        self.settings_layout.addRow('kP:', self.kP)
        self.settings_layout.addRow('kI:', self.kI)
        self.settings_layout.addRow('kD:', self.kD)
        self.settings_layout.addRow(self.send_btn)

        self.main_layout.addWidget(self.settings_group)
        self.main_layout.addWidget(self.temp_log_graph, 3)

        # Statues Bar Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update)
        self.timer.start()

        self.show()
        self.center()

    def center(self):
        '''Centers the window within the screen.'''
        qtRectangle = self.frameGeometry()
        centerPoint = QApplication.primaryScreen().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def receive(self):
        while self.serial.canReadLine():
            line = self.serial.readLine().data().decode('utf-8')
            temp = line.rstrip('\r\n')
            try:
                temp = float(temp)
                self.temp_log = np.roll(self.temp_log, -1)
                self.temp_log[-1] = temp
            except ValueError:
                return

    def update(self):
        self.statusBar().showMessage(
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz')
        )

        if self.temp_log_graph_ref is None:
            # create plot reference when None
            self.temp_log_graph_ref = self.temp_log_graph.plot(
                self.temp_log_X, self.temp_log
            )
        else:
            # use the plot reference to update the data for that line.
            self.temp_log_graph_ref.setData(self.temp_log_X, self.temp_log)

    @Slot()
    def open_dialog(self, serial: QtSerialPort.QSerialPort):
        '''Opens a port config dialog for the provided serial port.

        Parameters
        ----------
        serial : QSerialPort
            the serial port to be configured.
        '''
        dialog = port_config()
        if not serial.isOpen():
            if dialog.exec():
                portname, baudrate = dialog.get_results()
                serial.setPortName(portname)
                serial.setBaudRate(baudrate)

    def OpenCOM(self):
        '''Opens the serial port and initializes the combiner.'''
        if not self.serial.isOpen():
            self.serial.open(
                QtCore.QIODevice.OpenModeFlag.ReadWrite)
            self.serial.flush()

    def CloseCOM(self):
        '''Closes the serial port.'''
        self.serial.close()

    def SendCommand(self):
        if self.serial.isOpen():
            data = dict()
            data['P'] = self.kP.value()
            data['I'] = self.kI.value()
            data['D'] = self.kD.value()
            data['T'] = self.setT.value()
            command = str(data) + '\r'
            self.serial.write(command.encode('utf-8'))

    def StartGUI():
        '''Initializes a new QApplication and temperature_monitor.

        Use
        -------
        app, window = temperature_monitor.StartGUI()

        app.exec()

        Returns
        -------
        tuple (QApplication, temperature_monitor)
            Returns a tuple with QApp and temperature_monitor main window.
        '''
        return StartGUI(temperature_monitor)
