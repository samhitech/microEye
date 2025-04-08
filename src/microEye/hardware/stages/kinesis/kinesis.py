from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional, Union

import serial
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ActionParameter

from microEye.hardware.port_config import port_config
from microEye.hardware.stages.kinesis.kdc101 import KDC101Controller
from microEye.hardware.stages.stage import XYStageParams
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree
from microEye.utils.thread_worker import *


class KinesisXY(QtCore.QObject):
    '''Class for controlling Two Kinesis Devices as an XY Stage'''

    asyncStarted = Signal()
    asyncFinished = Signal()

    def __init__(self, x_port='COM12', y_port='COM11', parent=None, **kwargs):
        super().__init__(parent=parent)
        self.X = KDC101Controller(portName=x_port)
        self.Y = KDC101Controller(portName=y_port)
        self.min = kwargs.get('min', [0, 0])
        self.max = kwargs.get('max', [25, 25])
        self.center_pos = kwargs.get('center', [17, 13])
        self.prec = 4
        self.busy = False

    @property
    def position(self) -> tuple[float, float]:
        return self.X.position, self.Y.position

    def home(self, is_async=True):
        self.run_async(self.X.home, is_async=is_async)
        self.run_async(self.Y.home, is_async=is_async)

    def center(self, x_center=None, y_center=None, is_async=True):
        '''Move to the center of the stage.'''
        self.move_absolute(
            x_center if x_center is not None else self.center_pos[0],
            y_center if y_center is not None else self.center_pos[1],
            is_async=is_async,
        )

    def move_absolute(self, x, y, force=False, is_async=True):
        x = round(max(self.min[0], min(self.max[0], x)), self.prec)
        y = round(max(self.min[1], min(self.max[1], y)), self.prec)
        if x != self.X.position or force:
            self.run_async(self.X.move_absolute, x, is_async=is_async)
        if y != self.Y.position or force:
            self.run_async(self.Y.move_absolute, y, is_async=is_async)

    def move_relative(self, x, y, is_async=True):
        x = round(x, self.prec)
        y = round(y, self.prec)
        if x != 0:
            self.run_async(self.X.move_relative, x, is_async=is_async)
        if y != 0:
            self.run_async(self.Y.move_relative, y, is_async=is_async)

    def stop(self):
        self.X.move_stop()
        self.Y.move_stop()

    def open(self):
        '''Opens the serial ports.'''
        res = self.X.open() and self.Y.open()
        return res

    def close(self):
        '''Closes the serial ports.'''
        self.X.close()
        self.Y.close()

    def run_async(self, callback, *args, is_async=True):
        if self.isOpen() and not self.busy:
            self.X.clearResponses()
            self.Y.clearResponses()

            def worker_callback():
                try:
                    self.busy = True
                    self.asyncStarted.emit()
                    callback(*args)
                except Exception as e:
                    traceback.print_exc()
                finally:
                    self.busy = False
                    self.asyncFinished.emit()

            if is_async:
                _worker = QThreadWorker(worker_callback, nokwargs=True)

                QtCore.QThreadPool.globalInstance().start(_worker)
            else:
                worker_callback()

    def isOpen(self):
        return all([self.X.isOpen(), self.Y.isOpen()])

    def get_config(self) -> dict:
        return {
            'X': {
                'port': self.X.portName(),
                'baudrate': self.X.baudRate(),
            },
            'Y': {
                'port': self.Y.portName(),
                'baudrate': self.Y.baudRate(),
            },
        }

    def load_config(self, config: dict):
        if isinstance(config, dict):
            x_config = config.get('X', {})
            y_config = config.get('Y', {})

            if isinstance(x_config, dict):
                self.X.setBaudRate(x_config.get('baudrate', 115200))
                self.X.setPortName(x_config.get('port', 'COM12'))

            if isinstance(y_config, dict):
                self.Y.setBaudRate(y_config.get('baudrate', 115200))
                self.Y.setPortName(y_config.get('port', 'COM13'))

    def getViewWidget(self):
        '''Generates a QGroupBox with XY
        stage controls.'''
        view = KinesisView(stage=self)

        return view


class KinesisView(Tree):
    '''View class for the Kinesis stage controller.'''

    def __init__(
        self, parent: Optional['QtWidgets.QWidget'] = None, stage: KinesisXY = None
    ):
        if not isinstance(stage, KinesisXY):
            raise ValueError('stage must be an instance of KinesisXY.')

        self.__stage = stage if stage else KinesisXY()
        self.__threadpool = QtCore.QThreadPool.globalInstance()

        super().__init__(parent)

    def create_parameters(self):
        params = [
            {
                'name': str(XYStageParams.MODEL),
                'type': 'str',
                'value': str(self),
                'readonly': True,
            },
            {
                'name': str(XYStageParams.STATUS),
                'type': 'str',
                'value': 'Idle',
                'readonly': True,
            },
            {
                'name': str(XYStageParams.X_POSITION),
                'type': 'float',
                'value': 0.0,
                'decimals': 6,
                'suffix': 'mm',
            },
            {
                'name': str(XYStageParams.Y_POSITION),
                'type': 'float',
                'value': 0.0,
                'decimals': 6,
                'suffix': 'mm',
            },
            {
                'name': str(XYStageParams.MOVE),
                'type': 'action',
            },
            {
                'name': str(XYStageParams.HOME),
                'type': 'action',
            },
            {
                'name': str(XYStageParams.CENTER),
                'type': 'action',
            },
            {
                'name': str(XYStageParams.STOP),
                'type': 'action',
            },
            {
                'name': str(XYStageParams.CONTROLS),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(XYStageParams.X_JUMP_P),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.X_STEP_P),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.X_STEP_N),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.X_JUMP_N),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.Y_JUMP_P),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.Y_STEP_P),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.Y_STEP_N),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.Y_JUMP_N),
                        'type': 'action',
                    },
                ],
            },
            {
                'name': str(XYStageParams.OPTIONS),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(XYStageParams.STEP),
                        'type': 'float',
                        'value': 0.05,
                        'suffix': 'mm',
                    },
                    {
                        'name': str(XYStageParams.JUMP),
                        'type': 'float',
                        'value': 0.5,
                        'suffix': 'mm',
                    },
                    {
                        'name': str(XYStageParams.ID_X),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.ID_Y),
                        'type': 'action',
                    },
                ],
            },
            {
                'name': str(XYStageParams.SERIAL_PORT),
                'type': 'group',
                'expanded': False,
                'children': [
                    {
                        'name': str(XYStageParams.PORT),
                        'type': 'list',
                        'limits': KinesisView._get_available_ports(),
                    },
                    {
                        'name': str(XYStageParams.BAUDRATE),
                        'type': 'list',
                        'default': 115200,
                        'limits': KinesisView._get_baudrates(),
                    },
                    {
                        'name': str(XYStageParams.SET_PORT_X),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.SET_PORT_Y),
                        'type': 'action',
                    },
                    {'name': str(XYStageParams.OPEN), 'type': 'action'},
                    {'name': str(XYStageParams.CLOSE), 'type': 'action'},
                    {
                        'name': str(XYStageParams.REFRESH),
                        'type': 'action',
                    },
                    {
                        'name': str(XYStageParams.PORT_STATE),
                        'type': 'str',
                        'value': 'closed',
                        'readonly': True,
                    },
                ],
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        # self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.initializeSignals()

        self.__stage.asyncStarted.connect(
            lambda: self.set_param_value(XYStageParams.STATUS, 'busy')
        )
        self.__stage.asyncFinished.connect(self.updateControls)

    def initializeSignals(self):
        self._connect_movement_signals()
        self._connect_identification_signals()
        self._connect_configuration_signals()

    def _connect_movement_signals(self):
        movement_params = [
            XYStageParams.MOVE,
            XYStageParams.HOME,
            XYStageParams.CENTER,
            XYStageParams.STOP,
            XYStageParams.X_JUMP_P,
            XYStageParams.X_JUMP_N,
            XYStageParams.X_STEP_P,
            XYStageParams.X_STEP_N,
            XYStageParams.Y_JUMP_P,
            XYStageParams.Y_JUMP_N,
            XYStageParams.Y_STEP_P,
            XYStageParams.Y_STEP_N,
        ]
        for param in movement_params:
            self.get_param(param).sigActivated.connect(self._handle_movement_signal)

    def _handle_movement_signal(self, action: Parameter):
        path = self.get_param_path(action)
        if not path:
            return

        param = XYStageParams('.'.join(path))

        if param == XYStageParams.MOVE:
            self.__stage.move_absolute(
                self.get_param_value(XYStageParams.X_POSITION),
                self.get_param_value(XYStageParams.Y_POSITION),
            )
        elif param == XYStageParams.HOME:
            self.__stage.home()
        elif param == XYStageParams.CENTER:
            self.__stage.center()
        elif param == XYStageParams.STOP:
            self.__stage.stop()
        elif param == XYStageParams.X_JUMP_P:
            self.__stage.move_relative(self.get_param_value(XYStageParams.JUMP), 0)
        elif param == XYStageParams.X_JUMP_N:
            self.__stage.move_relative(-self.get_param_value(XYStageParams.JUMP), 0)
        elif param == XYStageParams.X_STEP_P:
            self.__stage.move_relative(self.get_param_value(XYStageParams.STEP), 0)
        elif param == XYStageParams.X_STEP_N:
            self.__stage.move_relative(-self.get_param_value(XYStageParams.STEP), 0)
        elif param == XYStageParams.Y_JUMP_P:
            self.__stage.move_relative(0, self.get_param_value(XYStageParams.JUMP))
        elif param == XYStageParams.Y_JUMP_N:
            self.__stage.move_relative(0, -self.get_param_value(XYStageParams.JUMP))
        elif param == XYStageParams.Y_STEP_P:
            self.__stage.move_relative(0, self.get_param_value(XYStageParams.STEP))
        elif param == XYStageParams.Y_STEP_N:
            self.__stage.move_relative(0, -self.get_param_value(XYStageParams.STEP))

    def _connect_identification_signals(self):
        identification_params = [
            XYStageParams.ID_X,
            XYStageParams.ID_Y,
        ]
        for param in identification_params:
            self.get_param(param).sigActivated.connect(
                self._handle_identification_signal
            )

    def _handle_identification_signal(self, action: Parameter):
        path = self.get_param_path(action)
        if not path:
            return

        param = XYStageParams('.'.join(path))

        if param == XYStageParams.ID_X:
            self.__stage.X.identify()
        elif param == XYStageParams.ID_Y:
            self.__stage.Y.identify()

    def _connect_configuration_signals(self):
        configuration_params = [
            XYStageParams.SET_PORT_X,
            XYStageParams.SET_PORT_Y,
            XYStageParams.OPEN,
            XYStageParams.CLOSE,
            XYStageParams.REFRESH,
        ]
        for param in configuration_params:
            self.get_param(param).sigActivated.connect(
                self._handle_configuration_signal
            )

    def _handle_configuration_signal(self, action: Parameter):
        path = self.get_param_path(action)
        if not path:
            return

        param = XYStageParams('.'.join(path))

        if param == XYStageParams.SET_PORT_X:
            self.setConfig()
        elif param == XYStageParams.SET_PORT_Y:
            self.setConfig(X=False)
        elif param == XYStageParams.OPEN:
            self.open()
        elif param == XYStageParams.CLOSE:
            self.__stage.close()
        elif param == XYStageParams.REFRESH:
            self._refresh_ports()

    def isOpen(self):
        return self.__stage.isOpen()

    def setConfig(self, X: bool = True):
        if not self.__stage.isOpen():
            self.__stage.close()

            self.setPortName(self.get_param_value(XYStageParams.PORT), X)
            # self.setBaudRate(self.get_param_value(XYStageParams.BAUDRATE), X)

    def setPortName(self, name: str, X: bool = True):
        serial: KDC101Controller = self.__stage.X if X else self.__stage.Y
        if not serial.isOpen():
            serial.setPortName(name)

    def open(self):
        res = self.__stage.open()

        if res:
            self.set_param_value(XYStageParams.X_POSITION, self.__stage.position[0])
            self.set_param_value(XYStageParams.Y_POSITION, self.__stage.position[1])
            self.set_expanded(XYStageParams.SERIAL_PORT)

    def getStep(self):
        return self.get_param_value(XYStageParams.STEP)

    def setStep(self, value: float):
        self.set_param_value(XYStageParams.STEP, value)

    def getJump(self):
        return self.get_param_value(XYStageParams.JUMP)

    def setJump(self, value: float):
        self.set_param_value(XYStageParams.JUMP, value)

    def center(self):
        self.get_param(XYStageParams.CENTER).activate()

    def stop(self):
        self.get_param(XYStageParams.STOP).activate()

    def move(self, X: bool, jump: bool, direction: bool):
        if jump:
            param = (
                XYStageParams.X_JUMP_P
                if X and direction
                else XYStageParams.X_JUMP_N
                if X and not direction
                else XYStageParams.Y_JUMP_P
                if not X and direction
                else XYStageParams.Y_JUMP_N
            )
        else:
            param = (
                XYStageParams.X_STEP_P
                if X and direction
                else XYStageParams.X_STEP_N
                if X and not direction
                else XYStageParams.Y_STEP_P
                if not X and direction
                else XYStageParams.Y_STEP_N
            )

        self.get_param(param).activate()

    def moveRelative(self, x: float, y: float):
        self.__stage.move_relative(x, y)

    def moveAbsolute(self, x: float, y: float):
        self.__stage.move_absolute(x, y)

    def updateHighlight(self):
        '''
        Updates the highlight style of the "Connect" action.
        '''
        style = ''
        if self.isOpen():
            style = 'background-color: #004CB6'
        else:
            style = 'background-color: black'

        next(self.get_param(XYStageParams.OPEN).items.keys()).button.setStyleSheet(
            style
        )

    def updateControls(self, value: bool):
        step_params = [
            XYStageParams.X_STEP_N,
            XYStageParams.X_STEP_P,
            XYStageParams.Y_STEP_N,
            XYStageParams.Y_STEP_P,
        ]

        for param in step_params:
            next(self.get_param(param).items.keys()).button.setStyleSheet(
                'background-color: #004CB6' if not value else ''
            )

        jump_params = [
            XYStageParams.X_JUMP_N,
            XYStageParams.X_JUMP_P,
            XYStageParams.Y_JUMP_N,
            XYStageParams.Y_JUMP_P,
        ]

        for param in jump_params:
            next(self.get_param(param).items.keys()).button.setStyleSheet(
                'background-color: #004CB6' if value else ''
            )

    def updatePositions(self):
        self.set_param_value(XYStageParams.STATUS, 'idle')

        self.set_param_value(XYStageParams.X_POSITION, self.__stage.X.state.position)
        self.set_param_value(XYStageParams.Y_POSITION, self.__stage.Y.state.position)

    def __str__(self):
        return 'Kinesis KDC101 + Z825B'

    @staticmethod
    def _get_available_ports():
        return [
            info.portName() for info in QtSerialPort.QSerialPortInfo.availablePorts()
        ]

    @staticmethod
    def _get_baudrates():
        return [
            baudrate for baudrate in QtSerialPort.QSerialPortInfo.standardBaudRates()
        ]

    def _refresh_ports(self):
        self.get_param(XYStageParams.PORT).setLimits(self._get_available_ports())


# class OldKinesisView(QtWidgets.QGroupBox):
#     '''View class for the Kinesis stage controller.'''

#     def __init__(
#         self, parent: Optional['QtWidgets.QWidget'] = None, stage: KinesisXY = None
#     ):
#         super().__init__(parent)

#         self.setTitle(KinesisXY.__name__)
#         self.stage = stage if stage else KinesisXY()
#         self.threadpool = QtCore.QThreadPool.globalInstance()
#         self.init_ui()

#     def init_ui(self):
#         container = QtWidgets.QVBoxLayout()
#         self.setLayout(container)

#         self._connect_btn = QtWidgets.QPushButton(
#             'Connect', clicked=lambda: self.stage.open()
#         )
#         self._disconnect_btn = QtWidgets.QPushButton(
#             'Disconnect', clicked=lambda: self.stage.close()
#         )
#         self._config_btn = QtWidgets.QPushButton(
#             'Config.', clicked=lambda: self.stage.open_dialog()
#         )

#         self._stop_btn = QtWidgets.QPushButton(
#             'STOP!', clicked=lambda: self.stage.stop()
#         )

#         btns = QtWidgets.QHBoxLayout()
#         btns.addWidget(self._connect_btn)
#         btns.addWidget(self._disconnect_btn)
#         btns.addWidget(self._config_btn)
#         btns.addWidget(self._stop_btn)
#         container.addLayout(btns)

#         self.controlsWidget = QtWidgets.QWidget()
#         hLayout = QtWidgets.QVBoxLayout()
#         self.controlsWidget.setLayout(hLayout)
#         formLayout = QtWidgets.QFormLayout()
#         hLayout.addLayout(formLayout, 3)
#         container.addWidget(self.controlsWidget)
#         container.addStretch()

#         self.x_spin = QtWidgets.QDoubleSpinBox()
#         self.y_spin = QtWidgets.QDoubleSpinBox()
#         self.x_spin.setDecimals(self.stage.prec)
#         self.y_spin.setDecimals(self.stage.prec)
#         self.x_spin.setSingleStep(10 ** (-self.stage.prec))
#         self.y_spin.setSingleStep(10 ** (-self.stage.prec))
#         self.x_spin.setValue(self.stage.position[0])
#         self.y_spin.setValue(self.stage.position[1])
#         self.x_spin.setMinimum(self.stage.min[0])
#         self.y_spin.setMinimum(self.stage.min[1])
#         self.x_spin.setMaximum(self.stage.max[0])
#         self.y_spin.setMaximum(self.stage.max[1])

#         formLayout.addRow(QtWidgets.QLabel('X [mm]'), self.x_spin)
#         formLayout.addRow(QtWidgets.QLabel('Y [mm]'), self.y_spin)

#         self.step_spin = QtWidgets.QDoubleSpinBox()
#         self.jump_spin = QtWidgets.QDoubleSpinBox()
#         self.step_spin.setDecimals(self.stage.prec)
#         self.jump_spin.setDecimals(self.stage.prec)
#         self.step_spin.setSingleStep(10 ** (-self.stage.prec))
#         self.jump_spin.setSingleStep(10 ** (-self.stage.prec))
#         self.step_spin.setMinimum(self.stage.min[0])
#         self.jump_spin.setMinimum(self.stage.min[1])
#         self.step_spin.setMaximum(self.stage.max[0])
#         self.jump_spin.setMaximum(self.stage.max[1])
#         self.step_spin.setValue(0.050)
#         self.jump_spin.setValue(0.5)

#         formLayout.addRow(QtWidgets.QLabel('Step [mm]'), self.step_spin)
#         formLayout.addRow(QtWidgets.QLabel('Jump [mm]'), self.jump_spin)

#         self._move_btn = QtWidgets.QPushButton(
#             'Move',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_absolute,
#                 self.x_spin.value(),
#                 self.y_spin.value(),
#             ),
#         )
#         self._home_btn = QtWidgets.QPushButton(
#             'Home', clicked=lambda: self.doAsync(self.stage.home)
#         )
#         self._center_btn = QtWidgets.QPushButton(
#             'Center',
#             clicked=lambda: self.doAsync(self.stage.center),
#         )
#         self.x_id_btn = QtWidgets.QPushButton(
#             'ID X', clicked=lambda: self.stage.X_Kinesis.identify()
#         )
#         self.y_id_btn = QtWidgets.QPushButton(
#             'ID Y', clicked=lambda: self.stage.Y_Kinesis.identify()
#         )

#         controls = QtWidgets.QHBoxLayout()
#         controls.addWidget(self._move_btn)
#         controls.addWidget(self._home_btn)
#         controls.addWidget(self._center_btn)
#         controls.addWidget(self.x_id_btn)
#         controls.addWidget(self.y_id_btn)
#         formLayout.addRow(controls)

#         self.n_x_jump_btn = QtWidgets.QPushButton(
#             'x--',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, -self.jump_spin.value(), 0
#             ),
#         )
#         self.n_x_step_btn = QtWidgets.QPushButton(
#             'x-',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, -self.step_spin.value(), 0
#             ),
#         )
#         self.p_x_step_btn = QtWidgets.QPushButton(
#             'x+',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, self.step_spin.value(), 0
#             ),
#         )
#         self.p_x_jump_btn = QtWidgets.QPushButton(
#             'x++',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, self.jump_spin.value(), 0
#             ),
#         )

#         self.n_y_jump_btn = QtWidgets.QPushButton(
#             'y--',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, 0, -self.jump_spin.value()
#             ),
#         )
#         self.n_y_step_btn = QtWidgets.QPushButton(
#             'y-',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, 0, -self.step_spin.value()
#             ),
#         )
#         self.p_y_step_btn = QtWidgets.QPushButton(
#             'y+',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, 0, self.step_spin.value()
#             ),
#         )
#         self.p_y_jump_btn = QtWidgets.QPushButton(
#             'y++',
#             clicked=lambda: self.doAsync(
#                 self.stage.move_relative, 0, self.jump_spin.value()
#             ),
#         )

#         self.n_x_step_btn.setStyleSheet('background-color: #004CB6')
#         self.n_y_step_btn.setStyleSheet('background-color: #004CB6')
#         self.p_x_step_btn.setStyleSheet('background-color: #004CB6')
#         self.p_y_step_btn.setStyleSheet('background-color: #004CB6')

#         grid = QtWidgets.QGridLayout()
#         grid.addWidget(self.n_x_jump_btn, 2, 0)
#         grid.addWidget(self.n_x_step_btn, 2, 1)
#         grid.addWidget(self.p_x_step_btn, 2, 3)
#         grid.addWidget(self.p_x_jump_btn, 2, 4)

#         grid.addWidget(self.n_y_jump_btn, 4, 2)
#         grid.addWidget(self.n_y_step_btn, 3, 2)
#         grid.addWidget(self.p_y_step_btn, 1, 2)
#         grid.addWidget(self.p_y_jump_btn, 0, 2)

#         hLayout.addLayout(grid, 1)
#         hLayout.addStretch()

#     def doAsync(self, callback, *args):
#         res = self.stage.isOpen()
#         if res[0] and res[1]:
#             self.stage.X_Kinesis.serial.read_all()
#             self.stage.Y_Kinesis.serial.read_all()
#         if self.controlsWidget is not None:
#             self.controlsWidget.setEnabled(False)
#         _worker = QThreadWorker(callback, *args, nokwargs=True)
#         # Execute
#         _worker.signals.result.connect(lambda: self.update())

#         _worker.setAutoDelete(True)

#         _worker.signals.finished.connect(lambda: self.threadpool.clear())

#         self.threadpool.start(_worker)

#     def update(self):
#         self.x_spin.setValue(self.stage.position[0])
#         self.y_spin.setValue(self.stage.position[1])
#         if self.controlsWidget is not None:
#             self.controlsWidget.setEnabled(True)


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
