import weakref
from enum import Enum
from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.hardware.port_config import port_config
from microEye.hardware.stages.stage import stage
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils import Tree


class ElliptecStage(stage):
    def __init__(self):
        super().__init__()

        self.widget = None
        self.serial = QtSerialPort.QSerialPort(None, readyRead=self.rx_piezo)
        self.serial.setBaudRate(9600)
        self.serial.setPortName('COM9')  # Replace with the appropriate port name

    def isOpen(self):
        return self.serial.isOpen()

    def open(self):
        self.serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

    def close(self):
        self.serial.close()

    def setPortName(self, name: str):
        self.serial.setPortName(name)

    def setBaudRate(self, baudRate: int):
        self.serial.setBaudRate(baudRate)

    def write(self, value):
        self.serial.write(value)

    def home(self, address):
        '''Homes the stage at a specific address'''
        if self.isOpen():
            self.LastCmd = f'{address}ho0'
            self.write(self.LastCmd.encode('utf-8'))

    def forward(self, address):
        '''Moves the stage at a specific address one step FORWARD'''
        if self.isOpen():
            self.LastCmd = f'{address}fw'
            self.write(self.LastCmd.encode('utf-8'))

    def backward(self, address):
        '''Moves the stage at a specific address one step BACKWARD'''
        if self.isOpen():
            self.LastCmd = f'{address}bw'
            self.write(self.LastCmd.encode('utf-8'))

    def set_slot(self, address, slot):
        '''Moves the stage at a specific address one step BACKWARD'''
        if self.isOpen():
            self.LastCmd = f'{address}ma000000{slot * 2}0'
            self.write(self.LastCmd.encode('utf-8'))

    def rx_piezo(self):
        '''Controller dataReady signal.'''
        self.Received = str(self.serial.readAll(), encoding='utf8')

    def getViewWidget(self):
        view = ElliptecStageView(stage=self)

        # Elliptec init config
        view.set_param_value(ElliptecStageParams.STAGE_ADDRESS, 2)
        view.set_param_value(ElliptecStageParams.STAGE_TYPE, 'ELL6')
        view.get_param(ElliptecStageParams.ADD_STAGE).activate()
        view.set_param_value(ElliptecStageParams.STAGE_ADDRESS, 0)
        view.set_param_value(ElliptecStageParams.STAGE_TYPE, 'ELL6')
        view.get_param(ElliptecStageParams.ADD_STAGE).activate()

        return view


class ElliptecStageParams(Enum):
    MODEL = 'Model'
    STAGE_TYPE = 'Stage Type'
    STAGE_ADDRESS = 'Stage Address'
    ADD_STAGE = 'Add Stage'
    STAGES = 'Stages'
    HOME = 'Home'
    FORWARD = 'Forward'
    BACKWARD = 'Backward'
    SLOT = 'Slot'
    SET_SLOT = 'Set Slot'
    SERIAL_PORT = 'Serial Port'
    PORT = 'Serial Port.Port'
    BAUDRATE = 'Serial Port.Baudrate'
    SET_PORT = 'Serial Port.Set Config'
    OPEN = 'Serial Port.Connect'
    CLOSE = 'Serial Port.Disconnect'
    PORT_STATE = 'Serial Port.State'
    REMOVE = 'Remove Device'

    def __str__(self):
        return self.value.split('.')[-1]

    def get_path(self):
        return self.value.split('.')


class ElliptecStageView(Tree):
    PARAMS = ElliptecStageParams
    removed = Signal(object)

    STAGE_TYPES = ['ELL6', 'ELL9']

    def __init__(
        self, parent: Optional['QtWidgets.QWidget'] = None, stage: ElliptecStage = None
    ):
        super().__init__(parent=parent)
        self.stage = stage if stage else ElliptecStage()

        self.stages = {}

    def create_parameters(self):
        params = [
            {
                'name': str(ElliptecStageParams.MODEL),
                'type': 'str',
                'value': 'Elliptec Stages Controller',
                'readonly': True,
            },
            {
                'name': str(ElliptecStageParams.STAGE_ADDRESS),
                'type': 'int',
                'default': 0,
                'limits': [0, 9],
            },
            {
                'name': str(ElliptecStageParams.STAGE_TYPE),
                'type': 'list',
                'limits': self.STAGE_TYPES,
            },
            {'name': str(ElliptecStageParams.ADD_STAGE), 'type': 'action'},
            {'name': str(ElliptecStageParams.STAGES), 'type': 'group', 'children': []},
            {
                'name': str(ElliptecStageParams.SERIAL_PORT),
                'type': 'group',
                'children': [
                    {
                        'name': str(ElliptecStageParams.PORT),
                        'type': 'list',
                        'limits': [
                            info.portName()
                            for info in QtSerialPort.QSerialPortInfo.availablePorts()
                        ],
                    },
                    {
                        'name': str(ElliptecStageParams.BAUDRATE),
                        'type': 'list',
                        'default': 9600,
                        'limits': [9600, 19200, 38400, 57600, 115200],
                    },
                    {'name': str(ElliptecStageParams.SET_PORT), 'type': 'action'},
                    {'name': str(ElliptecStageParams.OPEN), 'type': 'action'},
                    {'name': str(ElliptecStageParams.CLOSE), 'type': 'action'},
                    {
                        'name': str(ElliptecStageParams.PORT_STATE),
                        'type': 'str',
                        'value': 'closed',
                        'readonly': True,
                    },
                ],
            },
            # {'name': str(ElliptecStageParams.REMOVE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        # self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(ElliptecStageParams.SET_PORT).sigActivated.connect(
            self.set_config
        )
        self.get_param(ElliptecStageParams.OPEN).sigActivated.connect(
            lambda: self.stage.open()
        )
        self.get_param(ElliptecStageParams.CLOSE).sigActivated.connect(
            lambda: self.stage.close()
        )
        # self.get_param(ElliptecStageParams.REMOVE).sigActivated.connect(
        #     self.remove_widget
        # )
        self.get_param(ElliptecStageParams.ADD_STAGE).sigActivated.connect(
            self.add_stage
        )

    def isOpen(self):
        return self.stage.isOpen()

    def set_config(self):
        if not self.stage.isOpen():
            self.setPortName(self.get_param_value(ElliptecStageParams.PORT))
            self.setBaudRate(self.get_param_value(ElliptecStageParams.BAUDRATE))

    def setPortName(self, name: str):
        self.stage.serial.setPortName(name)

    def setBaudRate(self, baudRate: int):
        self.stage.serial.setBaudRate(baudRate)

    def open(self):
        self.stage.serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

    def add_stage(self):
        stage_type = self.get_param_value(ElliptecStageParams.STAGE_TYPE)
        stage_address = self.get_param_value(ElliptecStageParams.STAGE_ADDRESS)
        if stage_address not in self.stages:
            self.add_stage_params(stage_type, stage_address)

    def add_stage_params(self, stage_type, address):
        if stage_type == 'ELL6':
            self.add_ell6_widget(address)
        elif stage_type == 'ELL9':
            self.add_ell9_widget(address)

    def add_ell6_widget(self, address):
        # Implement logic to add ELL6 stage widget
        ell6 = {
            'name': f'ELL6 - {address}',
            'type': 'group',
            'children': [
                {'name': str(ElliptecStageParams.HOME), 'type': 'action'},
                {'name': str(ElliptecStageParams.FORWARD), 'type': 'action'},
                {'name': str(ElliptecStageParams.BACKWARD), 'type': 'action'},
                {'name': str(ElliptecStageParams.REMOVE), 'type': 'action'},
            ],
        }
        self.add_param_child(ElliptecStageParams.STAGES, ell6)

        def get_path(enum: ElliptecStageParams):
            return '.'.join(
                [str(ElliptecStageParams.STAGES), f'ELL6 - {address}', str(enum)]
            )

        self.get_param(get_path(ElliptecStageParams.HOME)).sigActivated.connect(
            lambda: self.stage.home(address)
        )
        self.get_param(get_path(ElliptecStageParams.FORWARD)).sigActivated.connect(
            lambda: self.stage.forward(address)
        )
        self.get_param(get_path(ElliptecStageParams.BACKWARD)).sigActivated.connect(
            lambda: self.stage.backward(address)
        )
        self.get_param(get_path(ElliptecStageParams.REMOVE)).sigActivated.connect(
            lambda: self.get_param(
                f'{str(ElliptecStageParams.STAGES)}.ELL6 - {address}'
            ).remove()
        )

    def add_ell9_widget(self, address):
        # Implement logic to add ELL9 stage widget
        name = f'ELL9 - {address}'
        ell9 = {
            'name': name,
            'type': 'group',
            'children': [
                {'name': str(ElliptecStageParams.HOME), 'type': 'action'},
                {'name': str(ElliptecStageParams.FORWARD), 'type': 'action'},
                {'name': str(ElliptecStageParams.BACKWARD), 'type': 'action'},
                {
                    'name': str(ElliptecStageParams.SLOT),
                    'type': 'int',
                    'default': 0,
                    'limits': [0, 3],
                },
                {'name': str(ElliptecStageParams.SET_SLOT), 'type': 'action'},
                {'name': str(ElliptecStageParams.REMOVE), 'type': 'action'},
            ],
        }
        self.add_param_child(ElliptecStageParams.STAGES, ell9)

        def get_path(enum: ElliptecStageParams):
            return '.'.join([str(ElliptecStageParams.STAGES), name, str(enum)])

        self.get_param(get_path(ElliptecStageParams.HOME)).sigActivated.connect(
            lambda: self.stage.home(address)
        )
        self.get_param(get_path(ElliptecStageParams.FORWARD)).sigActivated.connect(
            lambda: self.stage.forward(address)
        )
        self.get_param(get_path(ElliptecStageParams.BACKWARD)).sigActivated.connect(
            lambda: self.stage.backward(address)
        )
        self.get_param(get_path(ElliptecStageParams.SET_SLOT)).sigActivated.connect(
            lambda: self.stage.set_slot(
                address, self.get_param_value(get_path(ElliptecStageParams.SLOT))
            )
        )
        self.get_param(get_path(ElliptecStageParams.REMOVE)).sigActivated.connect(
            lambda: self.get_param(
                f'{str(ElliptecStageParams.STAGES)}.ELL9 - {address}'
            ).remove()
        )

    def updateHighlight(self):
        '''
        Updates the highlight style of the "Connect" action.
        '''
        style = ''
        if self.isOpen():
            style = 'background-color: #004CB6'
        else:
            style = 'background-color: black'

        next(
            self.get_param(ElliptecStageParams.OPEN).items.keys()
        ).button.setStyleSheet(style)

    def remove_widget(self):
        if self.parent() and not self.stage.isOpen():
            self.parent().layout().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Disconnect Elliptec stage before removing!')

    def __str__(self):
        return f'Elliptec Stage ({self.stage.serial.portName()})'
