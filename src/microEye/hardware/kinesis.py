import serial

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *


from .port_config import port_config
from ..thread_worker import *


class KinesisDevice:
    '''Class for controlling Thorlab's Z825B 25mm actuator by a KDC101'''
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

    def move_absolute(
            self, distance=0.1, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _distance = int(34554.96 * distance)
            _ABSOLUTE = [0x53, 0x04, 0x06, 0x00, dist | 0x80, source]
            _Params = [channelID, 0x0] + \
                list(_distance.to_bytes(4, 'little', signed=True))
            self.serial.write(_ABSOLUTE)
            self.serial.write(_Params)

    def move_relative(
            self, distance=0.1, channelID=0x0, dist=0x50, source=0x1):
        if self.isOpen():
            _distance = int(34554.96 * distance)
            _RELATIVE = [0x48, 0x04, 0x06, 0x00, dist | 0x80, source]
            _Params = [channelID, 0x0] + \
                list(_distance.to_bytes(4, 'little', signed=True))
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

    def __init__(self, x_port='COM12', y_port='COM11',
                 threadpool: QThreadPool = None):
        self.X_Kinesis = KinesisDevice(x_port)
        self.Y_Kinesis = KinesisDevice(y_port)
        self.position = [0, 0]
        self.min = [0, 0]
        self.max = [25, 25]
        self.prec = 4
        self.threadpool = threadpool

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
            if x_dialog.exec_():
                portname, baudrate = x_dialog.get_results()
                self.X_Kinesis.serial.port = portname
                self.X_Kinesis.serial.baudrate = baudrate
            if y_dialog.exec_():
                portname, baudrate = y_dialog.get_results()
                self.Y_Kinesis.serial.port = portname
                self.Y_Kinesis.serial.baudrate = baudrate

    def doAsync(self, group, callback, *args):
        res = self.isOpen()
        if res[0] and res[1]:
            self.X_Kinesis.serial.read_all()
            self.Y_Kinesis.serial.read_all()
        if group is not None:
            group.setEnabled(False)
        _worker = thread_worker(
                callback, *args, progress=False, z_stage=False)
        # Execute
        _worker.signals.result.connect(
            lambda: self.update(group))

        _worker.setAutoDelete(True)

        _worker.signals.finished.connect(
            lambda: self.threadpool.clear())

        self.threadpool.start(_worker)

    def update(self, group=None):
        self.x_spin.setValue(self.position[0])
        self.y_spin.setValue(self.position[1])
        if group is not None:
            group.setEnabled(True)

    def getQWidget(self):
        '''Generates a QGroupBox with XY
        stage controls.'''
        group = QGroupBox('Kinesis XY Stage')
        container = QVBoxLayout()
        group.setLayout(container)

        self._connect_btn = QPushButton(
            "Connect",
            clicked=lambda: self.open()
        )
        self._disconnect_btn = QPushButton(
            "Disconnect",
            clicked=lambda: self.close()
        )
        self._config_btn = QPushButton(
            "Config.",
            clicked=lambda: self.open_dialog()
        )

        self._stop_btn = QPushButton(
            "STOP!",
            clicked=lambda: self.stop()
        )

        btns = QHBoxLayout()
        btns.addWidget(self._connect_btn)
        btns.addWidget(self._disconnect_btn)
        btns.addWidget(self._config_btn)
        btns.addWidget(self._stop_btn)
        container.addLayout(btns)

        controlsWidget = QWidget()
        hLayout = QHBoxLayout()
        controlsWidget.setLayout(hLayout)
        formLayout = QFormLayout()
        hLayout.addLayout(formLayout, 3)
        container.addWidget(controlsWidget)
        container.addStretch()

        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.x_spin.setDecimals(self.prec)
        self.y_spin.setDecimals(self.prec)
        self.x_spin.setSingleStep(10**(-self.prec))
        self.y_spin.setSingleStep(10**(-self.prec))
        self.x_spin.setValue(self.position[0])
        self.y_spin.setValue(self.position[1])
        self.x_spin.setMinimum(self.min[0])
        self.y_spin.setMinimum(self.min[1])
        self.x_spin.setMaximum(self.max[0])
        self.y_spin.setMaximum(self.max[1])

        formLayout.addRow(
            QLabel('X [mm]'),
            self.x_spin
        )
        formLayout.addRow(
            QLabel('Y [mm]'),
            self.y_spin
        )

        self.step_spin = QDoubleSpinBox()
        self.jump_spin = QDoubleSpinBox()
        self.step_spin.setDecimals(self.prec)
        self.jump_spin.setDecimals(self.prec)
        self.step_spin.setSingleStep(10**(-self.prec))
        self.jump_spin.setSingleStep(10**(-self.prec))
        self.step_spin.setMinimum(self.min[0])
        self.jump_spin.setMinimum(self.min[1])
        self.step_spin.setMaximum(self.max[0])
        self.jump_spin.setMaximum(self.max[1])
        self.step_spin.setValue(0.050)
        self.jump_spin.setValue(0.5)

        formLayout.addRow(
            QLabel('Step [mm]'),
            self.step_spin
        )
        formLayout.addRow(
            QLabel('Jump [mm]'),
            self.jump_spin
        )

        self._move_btn = QPushButton(
            "Move",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_absolute,
                self.x_spin.value(),
                self.y_spin.value()
            )
        )
        self._home_btn = QPushButton(
            "Home",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.home
            )
        )
        self._center_btn = QPushButton(
            "Center",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.center
            )
        )
        self.x_id_btn = QPushButton(
            "ID X",
            clicked=lambda: self.X_Kinesis.identify()
        )
        self.y_id_btn = QPushButton(
            "ID Y",
            clicked=lambda: self.Y_Kinesis.identify()
        )

        controls = QHBoxLayout()
        controls.addWidget(self._move_btn)
        controls.addWidget(self._home_btn)
        controls.addWidget(self._center_btn)
        controls.addWidget(self.x_id_btn)
        controls.addWidget(self.y_id_btn)
        formLayout.addRow(controls)

        self.n_x_jump_btn = QPushButton(
            "x--",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                - self.jump_spin.value(),
                0
            )
        )
        self.n_x_step_btn = QPushButton(
            "x-",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                - self.step_spin.value(),
                0
            )
        )
        self.p_x_step_btn = QPushButton(
            "x+",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                self.step_spin.value(),
                0
            )
        )
        self.p_x_jump_btn = QPushButton(
            "x++",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                self.jump_spin.value(),
                0
            )
        )

        self.n_y_jump_btn = QPushButton(
            "y--",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                0,
                - self.jump_spin.value()
            )
        )
        self.n_y_step_btn = QPushButton(
            "y-",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                0,
                - self.step_spin.value()
            )
        )
        self.p_y_step_btn = QPushButton(
            "y+",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                0,
                self.step_spin.value()
            )
        )
        self.p_y_jump_btn = QPushButton(
            "y++",
            clicked=lambda: self.doAsync(
                controlsWidget,
                self.move_relative,
                0,
                self.jump_spin.value()
            )
        )

        self.n_x_step_btn.setStyleSheet(
            "background-color: green")
        self.n_y_step_btn.setStyleSheet(
            "background-color: green")
        self.p_x_step_btn.setStyleSheet(
            "background-color: green")
        self.p_y_step_btn.setStyleSheet(
            "background-color: green")

        grid = QGridLayout()
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

        return group


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
