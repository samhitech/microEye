from typing import Optional

import serial

from microEye.hardware.port_config import port_config
from microEye.qt import QtCore, QtWidgets
from microEye.utils.thread_worker import *


class KinesisDevice:
    """Class for controlling Thorlab's Z825B 25mm actuator by a KDC101"""

    def __init__(self, port='COM12', baudrate=115200) -> None:
        self.serial = serial.Serial()
        self.serial.port = port
        self.serial.baudrate = baudrate
        # self.serial.open()

    def identify(self, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _IDENTIFY = [0x23, 0x2, channelID, 0x0, dist, source]
            self.serial.write(_IDENTIFY)

    def home(self, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _HOME = [0x43, 0x04, channelID, 0x0, dist, source]
            self.serial.write(_HOME)

    def jog_fw(self, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _JOG_FW = [0x6A, 0x04, channelID, 0x1, dist, source]
            self.serial.write(_JOG_FW)

    def jog_bw(self, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _JOG_BW = [0x6A, 0x04, channelID, 0x2, dist, source]
            self.serial.write(_JOG_BW)

    def move_absolute(self, distance=0.1, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _distance = int(34554.96 * distance)
            _ABSOLUTE = [0x53, 0x04, 0x06, 0x00, dist | 0x80, source]
            _Params = [channelID, 0x0] + list(
                _distance.to_bytes(4, 'little', signed=True)
            )
            self.serial.write(_ABSOLUTE)
            self.serial.write(_Params)

    def move_relative(self, distance=0.1, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _distance = int(34554.96 * distance)
            _RELATIVE = [0x48, 0x04, 0x06, 0x00, dist | 0x80, source]
            _Params = [channelID, 0x0] + list(
                _distance.to_bytes(4, 'little', signed=True)
            )
            self.serial.write(_RELATIVE)
            self.serial.write(_Params)

    def move_stop(self, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _STOP = [0x65, 0x04, channelID, 0x2, dist, source]
            self.serial.write(_STOP)

    def wait(self):
        if self.isOpen():
            return self.serial.read(6)

    def isOpen(self):
        '''Returns True if connected.'''
        return self.serial.isOpen()

    def open(self):
        '''Opens the serial port.'''
        self.serial.open()

    def close(self):
        '''Closes the supplied serial port.'''
        self.serial.close()


class KinesisXY:
    '''Class for controlling Two Kinesis Devices as an XY Stage'''

    def __init__(self, x_port='COM12', y_port='COM11'):
        self.X_Kinesis = KinesisDevice(x_port)
        self.Y_Kinesis = KinesisDevice(y_port)
        self.position = [0, 0]
        self.min = [0, 0]
        self.max = [25, 25]
        self.prec = 4

    def home(self):
        self.X_Kinesis.home()
        self.Y_Kinesis.home()
        self.position = [0, 0]
        return self.X_Kinesis.wait(), self.Y_Kinesis.wait()

    def center(self, x_center=17, y_center=17):
        return self.move_absolute(x_center, y_center, True)

    def move_absolute(self, x, y, force=False):
        x = round(max(self.min[0], min(self.max[0], x)), self.prec)
        y = round(max(self.min[1], min(self.max[1], y)), self.prec)
        if x != self.position[0] or force:
            self.X_Kinesis.move_absolute(x)
        if y != self.position[1] or force:
            self.Y_Kinesis.move_absolute(y)
        if x != self.position[0] or force:
            self.X_Kinesis.wait()
        if y != self.position[1] or force:
            self.Y_Kinesis.wait()

        self.position = [x, y]
        return self.position

    def move_relative(self, x, y):
        x = self.position[0] + x
        y = self.position[1] + y
        return self.move_absolute(x, y)

    def stop(self):
        self.X_Kinesis.move_stop()
        self.Y_Kinesis.move_stop()

    def open(self):
        '''Opens the serial ports.'''
        self.X_Kinesis.open()
        self.Y_Kinesis.open()

    def close(self):
        '''Closes the serial ports.'''
        self.X_Kinesis.close()
        self.Y_Kinesis.close()

    def isOpen(self):
        return self.X_Kinesis.isOpen(), self.Y_Kinesis.isOpen()

    def open_dialog(self):
        '''Opens a port config dialog
        for the serial port.
        '''
        if not self.X_Kinesis.isOpen() and not self.Y_Kinesis.isOpen():
            x_dialog = port_config(title='X Controller Config.')
            y_dialog = port_config(title='Y Controller Config.')
            if x_dialog.exec():
                portname, baudrate = x_dialog.get_results()
                self.X_Kinesis.serial.port = portname
                self.X_Kinesis.serial.baudrate = baudrate
            if y_dialog.exec():
                portname, baudrate = y_dialog.get_results()
                self.Y_Kinesis.serial.port = portname
                self.Y_Kinesis.serial.baudrate = baudrate

    def getViewWidget(self):
        '''Generates a QGroupBox with XY
        stage controls.'''
        view = KinesisView(stage=self)

        return view


class KinesisView(QtWidgets.QGroupBox):
    '''View class for the Kinesis stage controller.'''

    def __init__(
        self, parent: Optional['QtWidgets.QWidget'] = None, stage: KinesisXY = None
    ):
        super().__init__(parent)

        self.setTitle(KinesisXY.__name__)
        self.stage = stage if stage else KinesisXY()
        self.threadpool = QtCore.QThreadPool.globalInstance()
        self.init_ui()

    def init_ui(self):
        container = QtWidgets.QVBoxLayout()
        self.setLayout(container)

        self._connect_btn = QtWidgets.QPushButton(
            'Connect', clicked=lambda: self.stage.open()
        )
        self._disconnect_btn = QtWidgets.QPushButton(
            'Disconnect', clicked=lambda: self.stage.close()
        )
        self._config_btn = QtWidgets.QPushButton(
            'Config.', clicked=lambda: self.stage.open_dialog()
        )

        self._stop_btn = QtWidgets.QPushButton(
            'STOP!', clicked=lambda: self.stage.stop()
        )

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._connect_btn)
        btns.addWidget(self._disconnect_btn)
        btns.addWidget(self._config_btn)
        btns.addWidget(self._stop_btn)
        container.addLayout(btns)

        self.controlsWidget = QtWidgets.QWidget()
        hLayout = QtWidgets.QVBoxLayout()
        self.controlsWidget.setLayout(hLayout)
        formLayout = QtWidgets.QFormLayout()
        hLayout.addLayout(formLayout, 3)
        container.addWidget(self.controlsWidget)
        container.addStretch()

        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setDecimals(self.stage.prec)
        self.y_spin.setDecimals(self.stage.prec)
        self.x_spin.setSingleStep(10 ** (-self.stage.prec))
        self.y_spin.setSingleStep(10 ** (-self.stage.prec))
        self.x_spin.setValue(self.stage.position[0])
        self.y_spin.setValue(self.stage.position[1])
        self.x_spin.setMinimum(self.stage.min[0])
        self.y_spin.setMinimum(self.stage.min[1])
        self.x_spin.setMaximum(self.stage.max[0])
        self.y_spin.setMaximum(self.stage.max[1])

        formLayout.addRow(QtWidgets.QLabel('X [mm]'), self.x_spin)
        formLayout.addRow(QtWidgets.QLabel('Y [mm]'), self.y_spin)

        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.jump_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setDecimals(self.stage.prec)
        self.jump_spin.setDecimals(self.stage.prec)
        self.step_spin.setSingleStep(10 ** (-self.stage.prec))
        self.jump_spin.setSingleStep(10 ** (-self.stage.prec))
        self.step_spin.setMinimum(self.stage.min[0])
        self.jump_spin.setMinimum(self.stage.min[1])
        self.step_spin.setMaximum(self.stage.max[0])
        self.jump_spin.setMaximum(self.stage.max[1])
        self.step_spin.setValue(0.050)
        self.jump_spin.setValue(0.5)

        formLayout.addRow(QtWidgets.QLabel('Step [mm]'), self.step_spin)
        formLayout.addRow(QtWidgets.QLabel('Jump [mm]'), self.jump_spin)

        self._move_btn = QtWidgets.QPushButton(
            'Move',
            clicked=lambda: self.doAsync(
                self.stage.move_absolute,
                self.x_spin.value(),
                self.y_spin.value(),
            ),
        )
        self._home_btn = QtWidgets.QPushButton(
            'Home', clicked=lambda: self.doAsync(self.stage.home)
        )
        self._center_btn = QtWidgets.QPushButton(
            'Center',
            clicked=lambda: self.doAsync(self.stage.center),
        )
        self.x_id_btn = QtWidgets.QPushButton(
            'ID X', clicked=lambda: self.stage.X_Kinesis.identify()
        )
        self.y_id_btn = QtWidgets.QPushButton(
            'ID Y', clicked=lambda: self.stage.Y_Kinesis.identify()
        )

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._move_btn)
        controls.addWidget(self._home_btn)
        controls.addWidget(self._center_btn)
        controls.addWidget(self.x_id_btn)
        controls.addWidget(self.y_id_btn)
        formLayout.addRow(controls)

        self.n_x_jump_btn = QtWidgets.QPushButton(
            'x--',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, -self.jump_spin.value(), 0
            ),
        )
        self.n_x_step_btn = QtWidgets.QPushButton(
            'x-',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, -self.step_spin.value(), 0
            ),
        )
        self.p_x_step_btn = QtWidgets.QPushButton(
            'x+',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, self.step_spin.value(), 0
            ),
        )
        self.p_x_jump_btn = QtWidgets.QPushButton(
            'x++',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, self.jump_spin.value(), 0
            ),
        )

        self.n_y_jump_btn = QtWidgets.QPushButton(
            'y--',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, 0, -self.jump_spin.value()
            ),
        )
        self.n_y_step_btn = QtWidgets.QPushButton(
            'y-',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, 0, -self.step_spin.value()
            ),
        )
        self.p_y_step_btn = QtWidgets.QPushButton(
            'y+',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, 0, self.step_spin.value()
            ),
        )
        self.p_y_jump_btn = QtWidgets.QPushButton(
            'y++',
            clicked=lambda: self.doAsync(
                self.stage.move_relative, 0, self.jump_spin.value()
            ),
        )

        self.n_x_step_btn.setStyleSheet('background-color: #004CB6')
        self.n_y_step_btn.setStyleSheet('background-color: #004CB6')
        self.p_x_step_btn.setStyleSheet('background-color: #004CB6')
        self.p_y_step_btn.setStyleSheet('background-color: #004CB6')

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self.n_x_jump_btn, 2, 0)
        grid.addWidget(self.n_x_step_btn, 2, 1)
        grid.addWidget(self.p_x_step_btn, 2, 3)
        grid.addWidget(self.p_x_jump_btn, 2, 4)

        grid.addWidget(self.n_y_jump_btn, 4, 2)
        grid.addWidget(self.n_y_step_btn, 3, 2)
        grid.addWidget(self.p_y_step_btn, 1, 2)
        grid.addWidget(self.p_y_jump_btn, 0, 2)

        hLayout.addLayout(grid, 1)
        hLayout.addStretch()

    def doAsync(self, callback, *args):
        res = self.stage.isOpen()
        if res[0] and res[1]:
            self.stage.X_Kinesis.serial.read_all()
            self.stage.Y_Kinesis.serial.read_all()
        if self.controlsWidget is not None:
            self.controlsWidget.setEnabled(False)
        _worker = QThreadWorker(callback, *args, nokwargs=True)
        # Execute
        _worker.signals.result.connect(lambda: self.update())

        _worker.setAutoDelete(True)

        _worker.signals.finished.connect(lambda: self.threadpool.clear())

        self.threadpool.start(_worker)

    def update(self):
        self.x_spin.setValue(self.stage.position[0])
        self.y_spin.setValue(self.stage.position[1])
        if self.controlsWidget is not None:
            self.controlsWidget.setEnabled(True)


if __name__ == '__main__':
    XY = KinesisXY('COM11', 'COM12')

    # print(XY.home())
    print(XY.center())

    # for i in range(10):
    #     k = 0.01
    #     XY.move_relative(k, k)
    #     sleep(1)
    # for i in range(10):
    #     k = - 0.01
    #     XY.move_relative(k, k)
    #     sleep(1)
