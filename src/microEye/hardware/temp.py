
import os
import sys
from os import name

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
from pyqtgraph.widgets.PlotWidget import PlotWidget

import qdarkstyle

from .port_config import *


class temperature_monitor(QMainWindow):

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        # Define main window layout
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.temp_log_X = np.arange(100)
        self.temp_log = np.array([20.0] * 100)

        self.serial = QSerialPort(
            None,
            readyRead=self.receive
        )
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM4')

        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.temp_log_graph = PlotWidget()
        self.temp_log_graph.setLabel(
            "bottom", "Frame", **self.labelStyle)
        self.temp_log_graph.setLabel(
            "left", "Temperature [C]", **self.labelStyle)
        self.temp_log_graph_ref = None

        self.buttons_layout = QHBoxLayout()
        self.connect_btn = QPushButton(
            "Connect",
            clicked=lambda: self.OpenCOM()
        )
        self.disconnect_btn = QPushButton(
            "Disconnect",
            clicked=lambda: self.CloseCOM()
        )
        self.config_btn = QPushButton(
            "Config.",
            clicked=lambda: self.open_dialog(self.serial)
        )

        self.buttons_layout.addWidget(self.connect_btn)
        self.buttons_layout.addWidget(self.disconnect_btn)
        self.buttons_layout.addWidget(self.config_btn)

        self.settings_group = QGroupBox('Settings')
        self.settings_layout = QFormLayout()
        self.settings_group.setLayout(self.settings_layout)
        self.settings_group.setMinimumWidth(250)

        self.setT = QDoubleSpinBox()
        self.setT.setMinimum(-2000)
        self.setT.setMaximum(10000)
        self.setT.setValue(10)

        self.kP = QDoubleSpinBox()
        self.kP.setMinimum(-2000)
        self.kP.setMaximum(10000)
        self.kP.setValue(90)
        self.kI = QDoubleSpinBox()
        self.kI.setMinimum(-2000)
        self.kI.setMaximum(10000)
        self.kI.setValue(0)
        self.kD = QDoubleSpinBox()
        self.kD.setMinimum(-2000)
        self.kD.setMaximum(10000)
        self.kD.setValue(0)

        self.send_btn = QPushButton(
            'Send Settings',
            clicked=lambda: self.SendCommand())

        self.settings_layout.addRow(self.connect_btn)
        self.settings_layout.addRow(self.disconnect_btn)
        self.settings_layout.addRow(self.config_btn)
        self.settings_layout.addRow(
            'Set T [C]:',
            self.setT
        )
        self.settings_layout.addRow(
            'kP:',
            self.kP
        )
        self.settings_layout.addRow(
            'kI:',
            self.kI
        )
        self.settings_layout.addRow(
            'kD:',
            self.kD
        )
        self.settings_layout.addRow(self.send_btn)

        self.main_layout.addWidget(self.settings_group)
        self.main_layout.addWidget(self.temp_log_graph, 3)

        # Statues Bar Timer
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update)
        self.timer.start()

        self.show()
        self.center()

    def center(self):
        '''Centers the window within the screen.
        '''
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def receive(self):
        while(self.serial.canReadLine()):
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
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz'))

        if self.temp_log_graph_ref is None:
            # create plot reference when None
            self.temp_log_graph_ref = self.temp_log_graph.plot(
                self.temp_log_X, self.temp_log)
        else:
            # use the plot reference to update the data for that line.
            self.temp_log_graph_ref.setData(
                self.temp_log_X, self.temp_log)

    @pyqtSlot()
    def open_dialog(self, serial: QSerialPort):
        '''Opens a port config dialog for the provided serial port.

        Parameters
        ----------
        serial : QSerialPort
            the serial port to be configured.
        '''
        dialog = port_config()
        if not serial.isOpen():
            if dialog.exec_():
                portname, baudrate = dialog.get_results()
                serial.setPortName(portname)
                serial.setBaudRate(baudrate)

    def OpenCOM(self):
        '''Opens the serial port and initializes the combiner.
        '''
        if not self.serial.isOpen():
            self.serial.open(QIODevice.ReadWrite)
            self.serial.flush()

    def CloseCOM(self):
        '''Closes the serial port.
        '''
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
            myappid = u'samhitech.mircoEye.temperature_monitor'  # appid
            ctypes.windll.shell32.\
                SetCurrentProcessExplicitAppUserModelID(myappid)

        window = temperature_monitor()
        return app, window
