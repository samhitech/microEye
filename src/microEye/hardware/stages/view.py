import weakref
from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.hardware.stages.stage import AbstractStage, Axis, StageParams
from microEye.qt import QtCore, QtSerialPort, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree


class StageView(Tree):
    PARAMS = StageParams
    removed = Signal(object)

    def __init__(
        self,
        stage: AbstractStage,
        parent: Optional['QtWidgets.QWidget'] = None,
    ):
        '''
        Initialize the ZStageView instance.

        This method initializes the `ZStageView` instance, sets up the stage signals,
        creates the parameter tree, and sets up the GUI elements.

        Parameters
        ----------
        stage : Optional[`AbstractStage`]
            The stage to be controlled by the GUI. If None, a new stage instance is
            created.
        parent : Optional[QWidget]
            The parent widget.
        '''
        if stage is None:
            raise ValueError('Stage cannot be None.')

        self._stage = weakref.ref(stage)

        super().__init__(parent=parent)

    def __str__(self):
        return f'Stage View: {self.stage}'

    @property
    def stage(self) -> AbstractStage:
        '''Get the stage instance.'''
        # check if the weak reference is still valid
        if self._stage() is None:
            raise ReferenceError('The stage reference is no longer valid.')
        return self._stage()

    @property
    def position(self) -> tuple[float, ...]:
        '''Get the current position of the stage.'''
        x = self.get_param_value(StageParams.X_POSITION, 0.0)
        y = self.get_param_value(StageParams.Y_POSITION, 0.0)
        z = self.get_param_value(StageParams.Z_POSITION, 0.0)
        return x, y, z

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `ZStageView` class.
        '''
        params = [
            {
                'name': str(StageParams.MODEL),
                'type': 'str',
                'value': str(self.stage),
                'readonly': True,
            },
            {
                'name': str(StageParams.STATUS),
                'type': 'str',
                'value': 'Idle',
                'readonly': True,
            },
            *self._get_postions(),
            {
                'name': str(StageParams.MOVE),
                'type': 'action',
            },
            {'name': str(StageParams.HOME), 'type': 'action'},
            {
                'name': str(StageParams.CENTER),
                'type': 'action',
            },
            {'name': str(StageParams.REFRESH), 'type': 'action'},
            {
                'name': str(StageParams.STOP),
                'type': 'action',
            },
            {'name': str(StageParams.OPEN), 'type': 'action'},
            {'name': str(StageParams.CLOSE), 'type': 'action'},
            *self._get_options(),
            *self._get_serial_port(),
            # {'name': str(StageParams.REMOVE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self._initialize_signals()

    def _get_options(self) -> dict:
        options = {
            'name': str(StageParams.OPTIONS),
            'type': 'group',
            'expanded': False,
            'children': [],
        }
        mapping = {
            Axis.X: (StageParams.X_MAX, StageParams.X_CENTER),
            Axis.Y: (StageParams.Y_MAX, StageParams.Y_CENTER),
            Axis.Z: (StageParams.Z_MAX, StageParams.Z_CENTER),
        }

        def get_float_param(name, value, axis):
            return {
                'name': str(name),
                'type': 'float',
                'value': value,
                'readonly': False,
                'suffix': f'{self.stage.get_unit(axis).suffix()}',
                'decimals': 8,
            }

        for axis, (max_param, center_param) in mapping.items():
            if axis in self.stage.axes:
                options['children'].append(
                    get_float_param(str(max_param), self.stage.get_max(axis), axis)
                )
                options['children'].append(
                    get_float_param(
                        str(center_param), self.stage.get_center(axis), axis
                    )
                )
        if self.stage and self.stage.driver.is_dual_serial():
            options['children'].extend(
                [
                    {
                        'name': str(StageParams.ID_X),
                        'type': 'action',
                    },
                    {
                        'name': str(StageParams.ID_Y),
                        'type': 'action',
                    },
                ]
            )
        return [
            options,
        ]

    def _get_postions(self) -> list[dict]:
        positions = []
        mapping = {
            Axis.X: StageParams.X_POSITION,
            Axis.Y: StageParams.Y_POSITION,
            Axis.Z: StageParams.Z_POSITION,
        }
        for axis, param in mapping.items():
            if axis in self.stage.axes:
                positions.append(
                    {
                        'name': str(param),
                        'type': 'float',
                        'value': self.stage.x
                        if axis == Axis.X
                        else self.stage.y
                        if axis == Axis.Y
                        else self.stage.z,
                        'readonly': False,
                        'suffix': f'{self.stage.get_unit(axis).suffix()}',
                        'decimals': 8,
                    }
                )
        return positions

    def _get_serial_port(self) -> dict:
        if self.stage and self.stage.driver.is_serial():
            return [
                {
                    'name': str(StageParams.SERIAL_PORT),
                    'type': 'group',
                    'expanded': False,
                    'children': [
                        {
                            'name': str(StageParams.PORT),
                            'type': 'list',
                            'limits': StageView._get_available_ports(),
                        },
                        {
                            'name': str(StageParams.BAUDRATE),
                            'type': 'list',
                            'default': 115200,
                            'limits': StageView._get_baudrates(),
                        },
                        *(
                            [
                                {'name': str(StageParams.SET_PORT), 'type': 'action'},
                            ]
                            if not self.stage.driver.is_dual_serial()
                            else [
                                {
                                    'name': str(StageParams.SET_PORT_X),
                                    'type': 'action',
                                },
                                {
                                    'name': str(StageParams.SET_PORT_Y),
                                    'type': 'action',
                                },
                            ]
                        ),
                        {
                            'name': str(StageParams.PORT_STATE),
                            'type': 'str',
                            'value': 'closed',
                            'readonly': True,
                        },
                    ],
                }
            ]

        return []

    def _initialize_signals(self):
        self.stage.signals.moveFinished.connect(
            lambda: QtCore.QTimer.singleShot(250, self._update_positions)
        )
        self.stage.signals.positionChanged.connect(
            lambda: QtCore.QTimer.singleShot(250, self._update_positions)
        )

        self.stage.signals.asyncStarted.connect(
            lambda: self.set_param_value(StageParams.STATUS, 'busy')
        )
        self.stage.signals.asyncFinished.connect(self._update_positions)
        self.stage.signals.stageRemoved.connect(self._remove_widget)

        self._connect_signals()

    def _connect_signals(self):
        movement_params = [
            StageParams.OPEN,
            StageParams.CLOSE,
            StageParams.MOVE,
            StageParams.HOME,
            StageParams.CENTER,
            StageParams.REFRESH,
            StageParams.STOP,
            StageParams.SET_PORT,
            StageParams.SET_PORT_X,
            StageParams.SET_PORT_Y,
            StageParams.ID_X,
            StageParams.ID_Y,
        ]
        for stage_param in movement_params:
            param = self.get_param(stage_param)
            if param is not None:
                param.sigActivated.connect(self._handle_signal)

    def _handle_signal(self, action: Parameter):
        path = self.get_param_path(action)
        if not path and not self.stage:
            return

        param = StageParams('.'.join(path))

        if param == StageParams.OPEN:
            self._open()
        elif param == StageParams.CLOSE:
            self._close()
        elif param == StageParams.MOVE:
            self.stage.move_absolute(*self.position)
        elif param == StageParams.HOME:
            self.stage.home()
        elif param == StageParams.CENTER:
            self.stage.center()
        elif param == StageParams.REFRESH:
            self.stage.refresh_position()
        elif param == StageParams.STOP:
            self.stage.stop()
        elif param == StageParams.SET_PORT:
            self.stage.setPortName(self.get_param_value(StageParams.PORT))
            self.stage.setBaudRate(self.get_param_value(StageParams.BAUDRATE))
        elif param == StageParams.SET_PORT_X:
            self.stage.X.setPortName(self.get_param_value(StageParams.PORT))
            self.stage.X.setBaudRate(self.get_param_value(StageParams.BAUDRATE))
        elif param == StageParams.SET_PORT_Y:
            self.stage.Y.setPortName(self.get_param_value(StageParams.PORT))
            self.stage.Y.setBaudRate(self.get_param_value(StageParams.BAUDRATE))
        elif param == StageParams.ID_X:
            self.stage.X.identify()
        elif param == StageParams.ID_Y:
            self.stage.Y.identify()

    def _update_positions(self):
        self.set_param_value(StageParams.STATUS, 'idle')

        if not self.stage:
            return

        mapping = {
            Axis.X: StageParams.X_POSITION,
            Axis.Y: StageParams.Y_POSITION,
            Axis.Z: StageParams.Z_POSITION,
        }

        for axis in self.stage.axes:
            param = self.get_param(mapping[axis])
            if param is not None:
                with param.treeChangeBlocker():
                    param.setValue(
                        self.stage.x
                        if axis == Axis.X
                        else self.stage.y
                        if axis == Axis.Y
                        else self.stage.z
                    )

    def _update_highlight(self):
        '''
        Updates the highlight style of the "Connect" action.
        '''
        style = ''
        if self.stage and self.stage.is_open():
            style = 'background-color: #004CB6'
        else:
            style = 'background-color: black'

        next(self.get_param(StageParams.OPEN).items.keys()).button.setStyleSheet(style)

    def _open(self):
        if self.stage and not self.stage.is_open():
            self.stage.open()

            self._update_positions()

    def _close(self):
        if self.stage and self.stage.is_open():
            self.stage.close()

            port_state = self.get_param(StageParams.PORT_STATE)
            if port_state is not None:
                port_state.setValue('open' if self.stage.is_open() else 'closed')

    def change(self, param, changes):
        for p, _, data in changes:
            path = self.param_tree.childPath(p)

            if path == StageParams.get_path(StageParams.X_MAX):
                self.stage.set_max_range(Axis.X, data)
            elif path == StageParams.get_path(StageParams.Y_MAX):
                self.stage.set_max_range(Axis.Y, data)
            elif path == StageParams.get_path(StageParams.Z_MAX):
                self.stage.set_max_range(Axis.Z, data)
            elif path == StageParams.get_path(StageParams.X_CENTER):
                self.stage.set_center(Axis.X, data)
            elif path == StageParams.get_path(StageParams.Y_CENTER):
                self.stage.set_center(Axis.Y, data)
            elif path == StageParams.get_path(StageParams.Z_CENTER):
                self.stage.set_center(Axis.Z, data)

            self.paramsChanged.emit(p, data)

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

    def _update_gui(self):
        port = self.get_param(StageParams.PORT)
        if port is not None:
            port.setLimits(self._get_available_ports())

        port_state = self.get_param(StageParams.PORT_STATE)
        if port_state is not None:
            port_state.setValue('open' if self.stage.is_open() else 'closed')

        self._update_positions()

    def _remove_widget(self):
        '''
        Remove the widget from the parent layout.

        This method removes the widget from the parent layout and deletes it.
        '''
        if self.parent() and not self.stage.is_open():
            self.parent().removeWidget(self)
            self.removed.emit(self)
            self.deleteLater()
        else:
            print(f'Cannot remove {self.stage} widget. Try to disconnect first.')

    def __str__(self):
        return f'{self.stage} View'
