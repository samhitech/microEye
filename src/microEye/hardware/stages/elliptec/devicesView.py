from enum import Enum
from typing import Any, Callable, Optional

from pyqtgraph.parametertree import Parameter

from microEye import __version__
from microEye.hardware.stages.elliptec.baseDevice import DeviceDirection
from microEye.hardware.stages.elliptec.deviceStatus import DeviceStatusValues
from microEye.hardware.stages.elliptec.ellDevices import *
from microEye.hardware.stages.elliptec.motorInfo import MotorInfo
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils import Tree


class ElliptecStageParams(Enum):
    MODEL = 'Controller Settings.Model'

    # Serial Port Settings
    SERIAL_PORT = 'Controller Settings.Serial Port'
    PORT = 'Controller Settings.Serial Port.Port'
    REFRESH = 'Controller Settings.Serial Port.Refresh Ports'
    BAUDRATE = 'Controller Settings.Serial Port.Baudrate'
    SET_PORT = 'Controller Settings.Serial Port.Set Config'
    OPEN = 'Controller Settings.Serial Port.Connect'
    CLOSE = 'Controller Settings.Serial Port.Disconnect'
    PORT_STATE = 'Controller Settings.Serial Port.State'

    # Stage Management
    MIN_ADDRESS = 'Stage Management.Range (min)'
    MAX_ADDRESS = 'Stage Management.Range (max)'
    SCAN = 'Stage Management.Scan Devices'

    # Devices
    DEVICES = 'Connected Devices'

    # Device Specific Params
    HOME = 'Home'
    HOME_DIRECTION = 'Homing Direction'
    FORWARD = 'Forward'
    BACKWARD = 'Backward'
    STOP_JOG = 'Stop Jog'
    SLOT = 'Slot'
    SET_SLOT = 'Set Slot'
    POSITION = 'Position'
    SET_POSITION = 'Set Position'
    STATUS = 'Status'
    JOG_STEP_SIZE = 'Jog Step Size'
    SET_JOG = 'Set Jog Step Size'
    GET_JOG = 'Get Jog Step Size'
    VELOCITY = 'Velocity'
    REMOVE = 'Remove Device'

    def __str__(self):
        return self.value.split('.')[-1]

    def get_path(self):
        return self.value.split('.')


class ElliptecStageView(Tree):
    PARAMS = ElliptecStageParams
    removed = Signal(object)

    # STAGE_TYPES = ['ELL6', 'ELL9', 'ELL14']

    def __init__(
        self, controller: ELLDevices, parent: Optional['QtWidgets.QWidget'] = None
    ):
        super().__init__(parent=parent)

        if not isinstance(controller, ELLDevices):
            raise ValueError('Invalid controller type!')

        self.controller = controller

    def create_parameters(self):
        params = [
            {
                'name': 'Controller Settings',
                'type': 'group',
                'children': [
                    {
                        'name': str(ElliptecStageParams.MODEL),
                        'type': 'str',
                        'value': 'Elliptec Devices Controller',
                        'readonly': True,
                    },
                    {
                        'name': str(ElliptecStageParams.SERIAL_PORT),
                        'type': 'group',
                        'expanded': False,
                        'children': [
                            {
                                'name': str(ElliptecStageParams.PORT),
                                'type': 'list',
                                'limits': self._get_available_ports(),
                            },
                            {
                                'name': str(ElliptecStageParams.BAUDRATE),
                                'type': 'list',
                                'default': 9600,
                                'limits': [9600, 19200, 38400, 57600, 115200],
                            },
                            {
                                'name': str(ElliptecStageParams.SET_PORT),
                                'type': 'action',
                            },
                            {'name': str(ElliptecStageParams.OPEN), 'type': 'action'},
                            {'name': str(ElliptecStageParams.CLOSE), 'type': 'action'},
                            {
                                'name': str(ElliptecStageParams.REFRESH),
                                'type': 'action',
                            },
                            {
                                'name': str(ElliptecStageParams.PORT_STATE),
                                'type': 'str',
                                'value': 'closed',
                                'readonly': True,
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'Stage Management',
                'type': 'group',
                'children': [
                    {
                        'name': str(ElliptecStageParams.MIN_ADDRESS),
                        'type': 'list',
                        'default': '0',
                        'limits': EllDevice.VALID_ADDRESSES,
                    },
                    {
                        'name': str(ElliptecStageParams.MAX_ADDRESS),
                        'type': 'list',
                        'default': 'F',
                        'limits': EllDevice.VALID_ADDRESSES,
                    },
                    {'name': str(ElliptecStageParams.SCAN), 'type': 'action'},
                ],
            },
            {'name': str(ElliptecStageParams.DEVICES), 'type': 'group', 'children': []},
            # {'name': str(ElliptecStageParams.REMOVE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        # self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.initializeSignals()

    def initializeSignals(self):
        self.get_param(ElliptecStageParams.SET_PORT).sigActivated.connect(
            lambda: self.setConfig()
        )
        self.get_param(ElliptecStageParams.OPEN).sigActivated.connect(
            lambda: self.open()
        )
        self.get_param(ElliptecStageParams.CLOSE).sigActivated.connect(
            lambda: DevicePort.close()
        )
        self.get_param(ElliptecStageParams.REFRESH).sigActivated.connect(
            lambda: self._refresh_ports()
        )
        self.get_param(ElliptecStageParams.SCAN).sigActivated.connect(
            lambda: self.findDevices()
        )

    def isOpen(self):
        return DevicePort.isOpen()

    def setConfig(self):
        if not DevicePort.isOpen():
            DevicePort.setPortName(self.get_param_value(ElliptecStageParams.PORT))
            DevicePort.setBaudRate(self.get_param_value(ElliptecStageParams.BAUDRATE))

    def setPortName(self, name: str):
        DevicePort.setPortName(name)

    def setBaudRate(self, baudRate: int):
        DevicePort.setBaudRate(baudRate)

    def open(self):
        res = DevicePort.open()

        if res:
            self.set_expanded(ElliptecStageParams.SERIAL_PORT)

    def findDevices(self):
        if not DevicePort.isOpen():
            print('Open the serial port first!')
            return

        self.get_param(ElliptecStageParams.DEVICES).clearChildren()

        min_address = self._get_valid_address(ElliptecStageParams.MIN_ADDRESS, '0')
        max_address = self._get_valid_address(ElliptecStageParams.MAX_ADDRESS, 'F')

        devices = self.controller.scanAddresses(min_address, max_address)
        for device in devices:
            # Configure each device found
            if self.controller.configure(device):
                self.configureDeviceParams(self.controller.addressedDevice(device[0]))

    def configureDeviceParams(self, device: EllDevice):
        if device.deviceInfo.DeviceType in [
            DeviceID.DeviceTypes.Shutter2,
            DeviceID.DeviceTypes.Shutter4,
            DeviceID.DeviceTypes.Shutter6,
        ]:
            self.buildShutterParams(device)
        elif device.deviceInfo.DeviceType in [
            DeviceID.DeviceTypes.OpticsRotator,
            DeviceID.DeviceTypes.LinearStage25mm,
            DeviceID.DeviceTypes.LinearStage28mm,
            DeviceID.DeviceTypes.LinearStage60mm,
            DeviceID.DeviceTypes.LinearStage60mm_10,
        ]:
            self.buildStageParams(device)

    def buildShutterParams(self, device: EllDevice):
        # Implement logic to add device params
        params = {
            'name': f'ELL{device.deviceInfo.DeviceType.value} - {device.address}',
            'type': 'group',
            'children': [
                {'name': str(ElliptecStageParams.HOME), 'type': 'action'},
                {'name': str(ElliptecStageParams.FORWARD), 'type': 'action'},
                {'name': str(ElliptecStageParams.BACKWARD), 'type': 'action'},
                {
                    'name': str(ElliptecStageParams.POSITION),
                    'type': 'float',
                    'value': 0.0,
                    'readonly': True,
                },
                {
                    'name': str(ElliptecStageParams.STATUS),
                    'type': 'str',
                    'value': device.deviceStatus.name,
                    'readonly': True,
                },
            ],
        }

        if device.deviceInfo.DeviceType in [
            DeviceID.DeviceTypes.Shutter4,
            DeviceID.DeviceTypes.Shutter6,
        ] and isinstance(params['children'], list):
            params['children'].extend(
                [
                    {
                        'name': str(ElliptecStageParams.SLOT),
                        'type': 'int',
                        'default': 0,
                        'limits': [
                            0,
                            3
                            if device.deviceInfo.DeviceType
                            == DeviceID.DeviceTypes.Shutter4
                            else 5,
                        ],
                    },
                    {'name': str(ElliptecStageParams.SET_SLOT), 'type': 'action'},
                ]
            )
        self.add_param_child(ElliptecStageParams.DEVICES, params)

        self.connectDeviceSignals(device)

    def buildStageParams(self, device: EllDevice):
        # Implement logic to add device params
        params = {
            'name': f'ELL{device.deviceInfo.DeviceType.value} - {device.address}',
            'type': 'group',
            'children': [],
        }
        if device.deviceInfo.DeviceType not in [
            DeviceID.DeviceTypes.OpticsRotator,
            DeviceID.DeviceTypes.LinearStage25mm,
            DeviceID.DeviceTypes.LinearStage28mm,
            DeviceID.DeviceTypes.LinearStage60mm,
            DeviceID.DeviceTypes.LinearStage60mm_10,
        ]:
            return

        device.get_position()
        device.get_jogstep_size()

        params['children'] = [
            {'name': str(ElliptecStageParams.HOME), 'type': 'action'},
            {'name': str(ElliptecStageParams.FORWARD), 'type': 'action'},
            {'name': str(ElliptecStageParams.BACKWARD), 'type': 'action'},
            {
                'name': str(ElliptecStageParams.POSITION),
                'type': 'float',
                'value': device.position,
                'suffix': device.deviceInfo.Units,
            },
            {'name': str(ElliptecStageParams.SET_POSITION), 'type': 'action'},
            {
                'name': str(ElliptecStageParams.STATUS),
                'type': 'str',
                'value': device.deviceStatus.name,
                'readonly': True,
            },
            {
                'name': str(ElliptecStageParams.JOG_STEP_SIZE),
                'type': 'float',
                'value': device.jogstep_size,
                'limits': [0.0, device.deviceInfo.Travel],
                'suffix': device.deviceInfo.Units,
            },
            {'name': str(ElliptecStageParams.GET_JOG), 'type': 'action'},
            {'name': str(ElliptecStageParams.SET_JOG), 'type': 'action'},
            {'name': str(ElliptecStageParams.STOP_JOG), 'type': 'action'},
            {
                'name': str(ElliptecStageParams.HOME_DIRECTION),
                'type': 'list',
                'default': 'Clockwise',
                'limits': [
                    'Clockwise',
                    'AntiClockwise',
                ],
                'visible': device.deviceInfo.DeviceType
                in [
                    DeviceID.DeviceTypes.OpticsRotator,
                    DeviceID.DeviceTypes.RotaryStage8,
                    DeviceID.DeviceTypes.RotaryStage18,
                ],
            },
        ]
        self.add_param_child(ElliptecStageParams.DEVICES, params)

        self.connectDeviceSignals(device)

    def connectParamSignal(self, paramPath: str, callback: Callable):
        param_obj = self.get_param(paramPath)
        if param_obj is not None:
            param_obj.sigActivated.connect(callback)
            return True

        return False

    def connectDeviceSignals(self, device: EllDevice):
        address = device.address
        dev_type = device.deviceInfo.DeviceType
        stage_path = (
            f'{str(ElliptecStageParams.DEVICES)}.ELL{dev_type.value} - {address}'
        )

        self.connectParamSignal(
            f'{stage_path}.{str(ElliptecStageParams.FORWARD)}',
            lambda: device.jog_forward(),
        )
        self.connectParamSignal(
            f'{stage_path}.{str(ElliptecStageParams.BACKWARD)}',
            lambda: device.jog_backward(),
        )

        if dev_type in [
            DeviceID.DeviceTypes.Shutter2,
            DeviceID.DeviceTypes.Shutter4,
            DeviceID.DeviceTypes.Shutter6,
        ]:
            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.HOME)}',
                lambda: device.home(),
            )

            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.SET_SLOT)}',
                lambda: device.move_absolute(
                    self.get_param_value(
                        f'{stage_path}.{str(ElliptecStageParams.SLOT)}'
                    )
                    * SHUTTER_POSITION_FACTOR[dev_type],
                ),
            )
        elif dev_type in [
            DeviceID.DeviceTypes.OpticsRotator,
            DeviceID.DeviceTypes.LinearStage25mm,
            DeviceID.DeviceTypes.LinearStage28mm,
            DeviceID.DeviceTypes.LinearStage60mm,
            DeviceID.DeviceTypes.LinearStage60mm_10,
        ]:
            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.STOP_JOG)}',
                lambda: device.jog_stop(),
            )

            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.SET_POSITION)}',
                lambda: device.move_absolute(
                    self.get_param_value(
                        f'{stage_path}.{str(ElliptecStageParams.POSITION)}'
                    )
                ),
            )
            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.HOME)}',
                lambda: device.home(
                    DeviceDirection[
                        self.get_param_value(
                            f'{stage_path}.{str(ElliptecStageParams.HOME_DIRECTION)}'
                        )
                    ]
                ),
            )

            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.GET_JOG)}',
                lambda: device.get_jogstep_size(),
            )
            self.connectParamSignal(
                f'{stage_path}.{str(ElliptecStageParams.SET_JOG)}',
                lambda: device.set_jogstep_size(
                    self.get_param_value(
                        f'{stage_path}.{str(ElliptecStageParams.JOG_STEP_SIZE)}'
                    )
                ),
            )

    def remove_widget(self):
        if self.parent() and not DevicePort.isOpen():
            self.parent().layout().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Disconnect Elliptec stage before removing!')

    def __str__(self):
        return f'Elliptec Devices ({DevicePort.instance().serial.portName()})'

    @staticmethod
    def _get_available_ports():
        return [
            info.portName() for info in QtSerialPort.QSerialPortInfo.availablePorts()
        ]

    def _refresh_ports(self):
        self.get_param(ElliptecStageParams.PORT).setLimits(self._get_available_ports())

    def _get_valid_address(self, param: ElliptecStageParams, default: Any):
        address = self.get_param_value(param)
        return address if EllDevice.is_valid_address(address) else default


class ElliptecView(QtWidgets.QTabWidget):
    HEADER = '> <span style="color:#0f0;">Elliptec Driver ('
    HEADER += f'<span style="color:#aaf;">microEye v{__version__}</span>)</span>'

    def __init__(
        self,
        elleptic: Optional[ELLDevices] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.initUI(elleptic)

    def initUI(self, elleptic: Optional[ELLDevices] = None):
        '''
        Initializes the ElliptecView UI.
        '''

        # Set window properties
        self.setWindowTitle('Elliptec Driver')

        # Create the output terminal
        self._output = QtWidgets.QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.appendHtml(self.HEADER)
        self._output.setStyleSheet('QPlainTextEdit { background-color: #111;}')

        # Create the ELLDevices controller
        if elleptic:
            self._ellDevices = elleptic
        else:
            self._ellDevices = ELLDevices()
        self._ellDevices.MessageUpdates.outputUpdated.connect(self.handleOutputUpdate)
        self._ellDevices.MessageUpdates.parameterUpdated.connect(
            self.handleParameterUpdate
        )
        DevicePort.instance().dataSent.connect(self.handleDataSent)
        DevicePort.instance().dataReceived.connect(self.handleDataReceived)

        # Create the devices tab
        self._elliptecStageView = ElliptecStageView(controller=self._ellDevices)

        # Set up the layout
        self.addTab(self._elliptecStageView, 'Devices')
        self.addTab(self._output, 'Output')


    def isOpen(self):
        return DevicePort.isOpen()

    def portName(self):
        return DevicePort.instance().serial.portName()

    def setPortName(self, name: str):
        DevicePort.setPortName(name)

    def baudRate(self):
        return DevicePort.instance().serial.baudRate()

    def setBaudRate(self, baudRate: int):
        DevicePort.setBaudRate(baudRate)

    def open(self):
        DevicePort.open()

    def closePort(self):
        DevicePort.close()

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
            self._elliptecStageView.get_param(ElliptecStageParams.OPEN).items.keys()
        ).button.setStyleSheet(style)

    def clearTerminal(self):
        '''
        Clears the terminal.
        '''
        self._output.clear()
        self._output.appendHtml(self.HEADER)

    def handleDataSent(self, message: str):
        '''
        Handles the data sent event.
        '''
        self._output.appendHtml(f'<span style="color: cyan;">Tx: {message}</span>')

    def handleDataReceived(self, message: str):
        '''
        Handles the data received event.
        '''
        self._output.appendHtml(f'<span style="color: green;">Rx: {message}</span>')

    def handleOutputUpdate(self, messages: list[str], error: bool):
        '''
        Handles the output update event.
        '''
        for message in messages:
            if 'Scanning for devices' in message:
                self.clearTerminal()

            self._output.appendHtml(
                f'<span style="color: yellow;">Output-> {message}</span>'
            )

    def handleParameterUpdate(
        self, update_type: MessageUpdater.UpdateTypes, address: str, data: Any
    ):
        if update_type == MessageUpdater.UpdateTypes.DeviceInfo and isinstance(
            data, DeviceID
        ):
            # if data and address in _device_view_models:
            #     _device_view_models[address].description = data.description()
            pass
        elif update_type == MessageUpdater.UpdateTypes.MotorInfo and isinstance(
            data, MotorInfo
        ):
            # if data and address in _device_view_models:
            #     motor_view_model = next(
            #         (
            #             motor
            #             for motor in _device_view_models[address].motors
            #             if motor.motor_id == data.motor_id
            #         ),
            #         None,
            #     )
            #     if motor_view_model:
            #         motor_view_model.update_info(data)
            pass
        elif update_type == MessageUpdater.UpdateTypes.Status and isinstance(
            data, DeviceStatusValues
        ):
            if data:
                device = self._ellDevices.addressedDevice(address)
                stage_path = (
                    f'{str(ElliptecStageParams.DEVICES)}.'
                    f'ELL{device.deviceInfo.DeviceType.value} - {address}'
                )

                param = self._elliptecStageView.get_param(
                    f'{stage_path}.{str(ElliptecStageParams.STATUS)}'
                )
                if param:
                    param.setValue(data.name)

                if data not in [DeviceStatusValues.OK, DeviceStatusValues.Busy]:
                    self._output.appendHtml(
                        f'<span style="color: red;">'
                        f'Device #{address} error: {data.name}</span>'
                    )
        elif update_type == MessageUpdater.UpdateTypes.Position and isinstance(
            data, (float, int)
        ):
            try:
                device = self._ellDevices.addressedDevice(address)
                stage_path = (
                    f'{str(ElliptecStageParams.DEVICES)}.'
                    f'ELL{device.deviceInfo.DeviceType.value} - {address}'
                )

                position = float(data)
                param = self._elliptecStageView.get_param(
                    f'{stage_path}.{str(ElliptecStageParams.POSITION)}'
                )
                if param:
                    param.setValue(position)
                if device.deviceInfo.DeviceType in [
                    DeviceID.DeviceTypes.Shutter4,
                    DeviceID.DeviceTypes.Shutter6,
                ]:
                    n = (
                        4
                        if device.deviceInfo.DeviceType == DeviceID.DeviceTypes.Shutter4
                        else 6
                    )
                    increment = device.deviceInfo.Travel / (n - 1)
                    slot = position // increment
                    param = self._elliptecStageView.get_param(
                        f'{stage_path}.{str(ElliptecStageParams.SLOT)}'
                    )
                    if param:
                        param.setValue(slot)
            except Exception as e:
                self._output.appendHtml(
                    f'<span style="color: red;">Device #{address} error: {e}</span>'
                )
        elif update_type in [
            MessageUpdater.UpdateTypes.PolarizerPositions,
            MessageUpdater.UpdateTypes.PaddlePosition,
        ]:
            pass
        elif update_type == MessageUpdater.UpdateTypes.HomeOffset and isinstance(
            data, (float, int)
        ):
            try:
                home_offset = float(data)
                # if address in _device_view_models:
                #     _device_view_models[address].update_home_offset(home_offset)
            except Exception as e:
                self._output.appendHtml(
                    f'<span style="color: red;">Device #{address} error: {e}</span>'
                )
        elif update_type == MessageUpdater.UpdateTypes.JogstepSize and isinstance(
            data, (float, int)
        ):
            try:
                device = self._ellDevices.addressedDevice(address)
                stage_path = (
                    f'{str(ElliptecStageParams.DEVICES)}.'
                    f'ELL{device.deviceInfo.DeviceType.value} - {address}'
                )

                jog_step = float(data)
                param = self._elliptecStageView.get_param(
                    f'{stage_path}.{str(ElliptecStageParams.JOG_STEP_SIZE)}'
                )
                if param:
                    param.setValue(jog_step)
                # if address in _device_view_models:
                #     _device_view_models[address].update_jogstep_size(jog_step)
            except Exception as e:
                self._output.appendHtml(
                    f'<span style="color: red;">Device #{address} error: {e}</span>'
                )

    def __str__(self):
        return 'Elliptec Devices'


if __name__ == '__main__':
    import sys

    from microEye.qt import QApplication

    try:
        print('Starting application...')
        app = QApplication(sys.argv)

        print('Creating ElliptecView...')
        view = ElliptecView()

        print('Showing window...')
        view.show()

        print('Entering event loop...')
        sys.exit(app.exec())
    except Exception as e:
        print(f'An error occurred: {e}')
