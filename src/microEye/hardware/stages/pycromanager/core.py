from microEye.hardware.pycromanager.core import DEFAULT_BRIDGE_PORT, PycroCore
from microEye.hardware.pycromanager.devices import PycroDevice
from microEye.hardware.pycromanager.enums import DeviceType
from microEye.hardware.stages.stage import AbstractStage, Axis, Units, emit_after_signal
from microEye.qt import QtWidgets


class PycroStageZ(PycroDevice, AbstractStage):
    NAME = 'PycroStage Z'

    def __init__(self, label: str = None, port: int = DEFAULT_BRIDGE_PORT) -> None:
        PycroDevice.__init__(self, label, DeviceType.StageDevice, port=port)

        AbstractStage.__init__(
            self,
            f'{PycroStageZ.NAME} {label}',
            max_range=(100,),
            units=Units.MICROMETERS,
            axes=(Axis.Z,),
        )

    def is_open(self) -> bool:
        return False

    def open(self):
        pass

    def close(self):
        pass

    @property
    def z(self) -> int:
        pos = self._core.get_position(self.label)
        self.set_position(axis=Axis.Z, position=pos)
        return pos

    @z.setter
    def z(self, value: int):
        self._core.set_position(value, self.label)
        self.set_position(axis=Axis.Z, position=value)

    @emit_after_signal('moveFinished')
    def move_relative(self, x, y, z, **kwargs):
        self._core.set_relative_position(z, self.label)
        return self.z

    @emit_after_signal('moveFinished')
    def move_absolute(self, x, y, z, **kwargs):
        self._core.set_position(z, self.label)
        return self.z

    def home(self):
        # self._core.home(self.label)
        self.move_absolute(0, 0, 0)
        return self.z

    def stop(self):
        self._core.stop(self.label)
        return self.z

    def set_adapter_origin_z(self, new_z_um: float):
        self._core.set_adapter_origin(new_z_um, self.label)

    def set_origin(self):
        self._core.set_origin(self.label)

    def refresh_position(self):
        self.move_absolute(0, 0, self.z)

    @classmethod
    def get_stage(cls, **kwargs):
        instances = PycroCore._instances.keys().__len__()
        if instances == 0:
            return None
        elif instances == 1:
            return cls(port=list(PycroCore._instances.keys())[0], **kwargs)
        else:
            # If there are multiple instances, prompt the user to select one
            port, ok = QtWidgets.QInputDialog.getItem(
                None,
                'Select PycroManager Instance',
                'Select the PycroManager instance to use:',
                list(map(str, PycroCore._instances.keys())),
            )
            if ok and port:
                return cls(port=int(port), **kwargs)
            else:
                return None

    def get_config(self) -> dict:
        config = super().get_config()

        config['pycromanager'] = {
            'label': self.label,
            'port': self._port,
        }

        return config

    def load_config(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError('Config must be a dictionary.')

        super().load_config(config)

        pycromanager_config = config.get('pycromanager', {})

        if isinstance(pycromanager_config, dict):
            self._label = pycromanager_config.get('label', self.label)
            self._port = pycromanager_config.get('port', self._port)


class PycroStageXY(PycroDevice, AbstractStage):
    NAME = 'PycroStage XY'

    def __init__(self, label: str = None, port: int = DEFAULT_BRIDGE_PORT) -> None:
        PycroDevice.__init__(self, label, DeviceType.XYStageDevice, port=port)

        AbstractStage.__init__(
            self,
            f'{PycroStageXY.NAME} {label}',
            max_range=(10 * 1000, 10 * 1000),
            units=Units.MICROMETERS,
            axes=(Axis.X, Axis.Y),
        )

    def is_open(self) -> bool:
        return False

    def open(self):
        pass

    def close(self):
        pass

    @property
    def x(self) -> float:
        pos = self._core.get_x_position(self.label)
        self.set_position(axis=Axis.X, position=pos)
        return pos

    @property
    def y(self) -> float:
        pos = self._core.get_y_position(self.label)
        self.set_position(axis=Axis.Y, position=pos)
        return pos

    @x.setter
    def x(self, value: float):
        x = value
        y = self.y
        self._core.set_xy_position(x, y, self.label)
        return self.x

    @y.setter
    def y(self, value: float):
        x = self.x
        y = value
        self._core.set_xy_position(x, y, self.label)
        return self.y

    @emit_after_signal('moveFinished')
    def move_relative(self, x: float, y: float, z: float = 0, **kwargs):
        self._core.set_relative_xy_position(x, y, self.label)
        return self.x, self.y

    @emit_after_signal('moveFinished')
    def move_absolute(self, x: float, y: float, z: float = 0, **kwargs):
        self._core.set_xy_position(x, y, self.label)
        return self.x, self.y

    def refresh_position(self):
        self.move_absolute(self.x, self.y, 0)

    def home(self):
        self._core.home(self.label)
        return self.x, self.y

    def stop(self):
        self._core.stop(self.label)
        return self.x, self.y

    def set_adapter_origin(self, new_x_um: float, new_y_um: float):
        self._core.set_adapter_origin_xy(new_x_um, new_y_um, self.label)

    def set_origin(self):
        self._core.set_origin_xy(self.label)

    def set_as_xy_stage_device(self):
        self._core.set_xy_stage_device(self.label)

    @classmethod
    def get_stage(cls, **kwargs):
        instances = PycroCore._instances.keys().__len__()
        if instances == 0:
            return None
        elif instances == 1:
            return cls(port=list(PycroCore._instances.keys())[0], **kwargs)
        else:
            # If there are multiple instances, prompt the user to select one
            port, ok = QtWidgets.QInputDialog.getItem(
                None,
                'Select PycroManager Instance',
                'Select the PycroManager instance to use:',
                list(map(str, PycroCore._instances.keys())),
            )
            if ok and port:
                return cls(port=int(port), **kwargs)
            else:
                return None

    def get_config(self) -> dict:
        config = super().get_config()

        config['pycromanager'] = {
            'label': self.label,
            'port': self._port,
        }

        return config

    def load_config(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError('Config must be a dictionary.')

        super().load_config(config)

        pycromanager_config = config.get('pycromanager', {})

        if isinstance(pycromanager_config, dict):
            self._label = pycromanager_config.get('label', self.label)
            self._port = pycromanager_config.get('port', self._port)
