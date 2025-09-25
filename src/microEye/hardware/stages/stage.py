from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Optional, Union

from microEye.hardware.stages.stabilizer import FocusStabilizer
from microEye.qt import QtCore, QtSerialPort, Signal


class Axis(Enum):
    '''
    Enum class defining the axis of the stage.
    '''

    X = 'X'
    Y = 'Y'
    Z = 'Z'

    def __str__(self):
        '''
        Return the last part of the enum value (Axis name).
        '''
        return self.value.split('.')[-1]

    @classmethod
    def from_string(cls, axis_str: str):
        '''
        Create an Axis enum from a string.

        Parameters
        ----------
        axis_str : str
            The string representation of the axis.

        Returns
        -------
        Axis
            The corresponding Axis enum value.
        '''
        axis_str = axis_str.upper()
        if axis_str in cls.__members__:
            return cls[axis_str]
        else:
            raise ValueError(f'Invalid axis: {axis_str}')


class Units(Enum):
    '''
    Enum class defining the units of the stage.
    '''

    PICOMETERS = 0.001  # 1 pm = 0.001 nm
    NANOMETERS = 1  # 1 nm = 1 nm
    MICROMETERS = 1000  # 1 um = 1000 nm
    MILLIMETERS = 1000000  # 1 mm = 1,000,000 nm

    SLIDES_PIXELS  = 100000

    def suffix(self) -> str:
        '''
        Return the suffix of the unit.

        Returns
        -------
        str
            The suffix of the unit.
        '''
        return {
            Units.PICOMETERS: 'pm',
            Units.NANOMETERS: 'nm',
            Units.MICROMETERS: 'µm',
            Units.MILLIMETERS: 'mm',
        }[self]

    @classmethod
    def convert(cls, value: float, from_unit: 'Units', to_unit: 'Units') -> float:
        '''
        Convert a value from one unit to another.

        Parameters
        ----------
        value : float
            The value to convert.
        from_unit : Units
            The unit of the input value.
        to_unit : Units
            The unit to convert to.

        Returns
        -------
        float
            The converted value.
        '''
        if not isinstance(from_unit, Units) or not isinstance(to_unit, Units):
            raise ValueError('from_unit and to_unit must be instances of Units enum.')

        return value * (from_unit.value / to_unit.value)


class StageDriver(Enum):
    '''
    Enum class defining the stage drivers.
    '''

    SERIALPORT = 'Serial Port'
    DUAL_SERIALPORT = 'Dual Serial Port'
    OTHER = 'Other'

    def is_serial(self) -> bool:
        '''
        Check if the driver is a serial port driver.

        Returns
        -------
        bool
            True if the driver is a serial port driver, False otherwise.
        '''
        return self in {StageDriver.SERIALPORT, StageDriver.DUAL_SERIALPORT}

    def is_dual_serial(self) -> bool:
        '''
        Check if the driver is a dual serial port driver.

        Returns
        -------
        bool
            True if the driver is a dual serial port driver, False otherwise.
        '''
        return self == StageDriver.DUAL_SERIALPORT


class StageParams(Enum):
    '''
    Enum class defining Stage parameters.
    '''

    MODEL = 'Model'
    STATUS = 'Status'

    X_POSITION = 'X Position'
    Y_POSITION = 'Y Position'
    Z_POSITION = 'Z Position'

    GET_POSITION = 'Get Position'

    MOVE = 'Move'

    HOME = 'Home'
    REFRESH = 'Refresh'
    CENTER = 'Center'
    STOP = 'Stop !'

    OPEN = 'Connect'
    CLOSE = 'Disconnect'

    OPTIONS = 'Options'
    STEP = 'Options.Step Size'
    JUMP = 'Options.Jump Size'
    ID_X = 'Options.ID X'
    ID_Y = 'Options.ID Y'

    X_MAX = 'Options.X Max'
    Y_MAX = 'Options.Y Max'
    Z_MAX = 'Options.Z Max'

    X_CENTER = 'Options.X Center'
    Y_CENTER = 'Options.Y Center'
    Z_CENTER = 'Options.Z Center'

    SERIAL_PORT = 'Serial Port'
    PORT = 'Serial Port.Port'
    BAUDRATE = 'Serial Port.Baudrate'
    SET_PORT = 'Serial Port.Set Config'
    SET_PORT_X = 'Serial Port.Set Config X'
    SET_PORT_Y = 'Serial Port.Set Config Y'
    PORT_STATE = 'Serial Port.State'

    INFO = 'Info'

    REMOVE = 'Remove Device'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')


class StageSignals(QtCore.QObject):
    positionChanged = Signal(object, float, Axis)
    moveFinished = Signal()
    '''Signal emitted when the stage position changes.

    Parameters
    ----------
    AbstractStage
        The stage object.
    float
        The new position of the stage.
    Axis
        The axis along which the position has changed.
    '''
    asyncStarted = Signal()
    asyncFinished = Signal()
    stageRemoved = Signal()


def emit_after_signal(signal_name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            signal = getattr(self.signals, signal_name)
            signal.emit()
            return result

        return wrapper

    return decorator


class AbstractStage(ABC):
    '''
    Abstract base class for all stages.
    '''

    NAME = 'Abstract Base Stage'

    def __init__(
        self,
        name: str = 'Abstract Base Stage',
        max_range: tuple[float] = 0,
        units: tuple[Units] = Units.NANOMETERS,
        axes: Optional[Union[tuple[Axis], list[Axis]]] = None,
        readyRead: Callable = None,
        center: Optional[Union[float, list, tuple]] = None,
    ):
        """
        Initialize the stage object.

        Parameters
        ----------
        name : str, optional
            The name of the stage (default is 'Abstract Base Stage').
        max_range : int, optional
            The maximum range of the stage (default is 0).
        units : tuple[Units], optional
            The units of the stage (default is Units.NANOMETERS).
        axes : Optional[Iterable[Axis]], optional
            Iterable of Axis members that this stage supports. Defaults to all axes.
        readyRead : Callable, optional
            A callable function to be executed when the serial port is ready to read
            data (default is None).
        center : Optional[Union[float, list, tuple]], optional
            The center position of the stage. If None, it is set to half of max_range.
        """
        self._name = name

        # determine which axes this stage actually supports
        self._axes: tuple[Axis, ...] = tuple(axes) if axes is not None else tuple(Axis)

        if len(self._axes) == 0:
            raise ValueError('At least one axis must be specified.')

        self._axes = tuple(sorted(self._axes, key=lambda axis: axis.value))

        def get_dict(
            props: Union[tuple, list, Any], factor: float = 1.0
        ) -> dict[Axis, Any]:
            if isinstance(props, (tuple, list)) and len(props) == 3:
                return {axis: props[i] * factor for i, axis in enumerate(self._axes)}
            else:
                return {axis: props * factor for axis in self._axes}

        self._metadata = {}
        for i, axis in enumerate(self._axes):
            self._metadata[axis] = {
                'max': max_range[i]
                if isinstance(max_range, (tuple, list))
                else max_range,
                'unit': units[i] if isinstance(units, (tuple, list)) else units,
            }
            if center is not None:
                self._metadata[axis]['center'] = (
                    center[i] if isinstance(center, (tuple, list)) else center
                )
            else:
                self._metadata[axis]['center'] = self._metadata[axis]['max'] * 0.5

        self._position = {axis: 0 for axis in self._axes}

        self.signals = StageSignals()

        if readyRead is not None and callable(readyRead):
            self.serial = QtSerialPort.QSerialPort(None)
            self.serial.readyRead.connect(readyRead)
            self.serial.setBaudRate(115200)
            self.serial.setPortName('COM5')
            self._driver = StageDriver.SERIALPORT
        else:
            self.serial = None
            self._driver = StageDriver.OTHER

        self._busy = False

    def __str__(self):
        return self._name

    # disconnect signals and serial port on deletion
    def __del__(self):
        if self.serial is not None and isinstance(
            self.serial, QtSerialPort.QSerialPort
        ):
            self.serial.readyRead.disconnect()

        self.signals = None

    @classmethod
    def get_stage(cls, **kwargs) -> 'AbstractStage':
        '''
        Factory method to create a stage instance.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments to pass to the stage constructor.

        Returns
        -------
        AbstractStage
            The created stage instance.
        '''
        return cls(**kwargs)

    def is_serial(self) -> bool:
        '''
        Check if the stage is serial.

        Returns
        -------
        bool
            True if the stage is serial, False otherwise.
        '''
        return self.serial is not None and isinstance(
            self.serial, QtSerialPort.QSerialPort
        )

    def setPortName(self, name: str):
        '''Sets the serial port name.'''
        if self.is_serial() and not self.is_open():
            self.serial.setPortName(name)

    def setBaudRate(self, baudRate: int):
        '''Sets the serial port baudrate.'''
        if self.is_serial() and not self.is_open():
            self.serial.setBaudRate(baudRate)

    @abstractmethod
    def is_open(self) -> bool:
        '''
        Check if the stage is open.

        Returns
        -------
        bool
            True if the stage is open, False otherwise.
        '''
        pass

    def get_position(self, axis: Axis) -> float:
        '''
        Get the current position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to query.

        Returns
        -------
        float
            The current position.
        '''
        if axis not in self._position:
            raise ValueError(f'Axis {axis} not supported by this stage')

        return self._position[axis]

    def set_position(self, axis: Axis, position: float, incremental=False) -> None:
        '''
        Set the current position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to set.
        position : float
            The target position.
        '''
        if axis not in self._position:
            raise ValueError(f'Axis {axis} not supported by this stage')

        if incremental:
            position = self._position[axis] + position

        if position < 0 or position > self._metadata[axis]['max']:
            return

        self._position[axis] = max(0, min(position, self._metadata[axis]['max']))

    @abstractmethod
    def home(self) -> None:
        '''
        Move the stage to its home position.
        '''
        pass

    @abstractmethod
    def open(self):
        '''
        Connect to the stage.
        '''
        pass

    @abstractmethod
    def close(self):
        '''
        Disconnect from the stage.
        '''
        pass

    @abstractmethod
    def stop(self) -> None:
        '''
        Stop the stage movement.
        '''
        pass

    @abstractmethod
    def move_absolute(
        self, x: Union[int, float], y: Union[int, float], z: Union[int, float], **kwargs
    ):
        '''
        Move the stage to an absolute position.

        Parameters
        ----------
        x : Union[int, float]
            The target x position.
        y : Union[int, float]
            The target y position.
        z : Union[int, float]
            The target z position.
        kwargs : dict
            Additional keyword arguments.
        '''
        pass

    @abstractmethod
    def move_relative(
        self, x: Union[int, float], y: Union[int, float], z: Union[int, float], **kwargs
    ):
        '''
        Move the stage to a relative position.

        Parameters
        ----------
        x : Union[int, float]
            The relative x position.
        y : Union[int, float]
            The relative y position.
        z : Union[int, float]
            The relative z position.
        kwargs : dict
            Additional keyword arguments.
        '''
        pass

    @abstractmethod
    def refresh_position(self):
        '''
        Refreshes the current position of the stage.

        **Example implementation:**

        ```python
        self.move_absolute(self.x, self.y, self.z)
        ```
        '''
        pass

    def center(self):
        '''
        Move the stage to its center position.
        '''
        center_xyz = (
            self._metadata.get(Axis.X, {}).get('center', 0),
            self._metadata.get(Axis.Y, {}).get('center', 0),
            self._metadata.get(Axis.Z, {}).get('center', 0),
        )
        self.move_absolute(*center_xyz)

    def move_right(self, step: int):
        '''
        Moves the stage right (+x) by a specified number of steps.
        '''
        self.move_relative(step, 0, 0)

    def move_left(self, step: int):
        '''
        Moves the stage left (-x) by a specified number of steps.
        '''
        self.move_relative(-step, 0, 0)

    def move_up(self, step: int):
        '''
        Moves the stage up (+y) by a specified number of steps.
        '''
        self.move_relative(0, step, 0)

    def move_down(self, step: int):
        '''
        Moves the stage down (-y) by a specified number of steps.
        '''
        self.move_relative(0, -step, 0)

    def move_higher(self, step: int):
        '''
        Moves the stage higher (+z) by a specified number of steps.
        '''
        self.move_relative(0, 0, step)

    def move_lower(self, step: int):
        '''
        Moves the stage lower (-z) by a specified number of steps.
        '''
        self.move_relative(0, 0, -step)

    def set_max_range(self, axis: Axis, max_range: float) -> None:
        '''
        Set the maximum range of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to set.
        max_range : float
            The maximum range.
        '''
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        self._metadata[axis]['max'] = max_range

    @property
    def busy(self) -> bool:
        '''Return True if the stage is currently busy.'''
        return self._busy

    @property
    def driver(self) -> StageDriver:
        '''Return the driver type for the stage.'''
        return self._driver

    @property
    def axes(self) -> tuple[Axis, ...]:
        '''Tuple of Axis members this stage supports (order preserved).'''
        return self._axes

    @property
    def axis_label(self) -> str:
        """String label of axes this stage supports (e.g. 'XYZ', 'Z', 'XY')."""
        return ''.join([axis.value for axis in self._axes])

    @property
    def n_axes(self) -> int:
        '''Number of supported axes.'''
        return len(self._axes)

    def has_axis(self, axis: Axis) -> bool:
        '''Return True if the stage supports the given axis.'''
        return axis in self._axes

    @property
    def position(self) -> dict[Axis, float]:
        '''Current stage position as a dictionary.'''
        return self._position.copy()

    @property
    def x(self) -> float:
        '''Current stage X position.'''
        return self.get_position(Axis.X)

    @property
    def dx(self) -> float:
        '''Current stage delta X position from center.'''
        return self.x - self.get_center(Axis.X)

    @x.setter
    def x(self, value: float):
        '''Set the stage X position.'''
        self.set_position(Axis.X, value)

    @property
    def y(self) -> float:
        '''Current stage Y position.'''
        return self.get_position(Axis.Y)

    @property
    def dy(self) -> float:
        '''Current stage delta Y position from center.'''
        return self.y - self.get_center(Axis.Y)

    @y.setter
    def y(self, value: float):
        '''Set the stage Y position.'''
        self.set_position(Axis.Y, value)

    @property
    def z(self) -> float:
        '''Current stage Z position.'''
        return self.get_position(Axis.Z)

    @property
    def dz(self) -> float:
        '''Current stage delta Z position from center.'''
        return self.z - self.get_center(Axis.Z)

    @z.setter
    def z(self, value: float):
        '''Set the stage Z position.'''
        self.set_position(Axis.Z, value)

    @property
    def x_max(self) -> int:
        '''Maximum stage position.'''
        return self._metadata[Axis.X]['max']

    @property
    def y_max(self) -> int:
        '''Maximum stage position.'''
        return self._metadata[Axis.Y]['max']

    @property
    def z_max(self) -> int:
        '''Maximum stage position.'''
        return self._metadata[Axis.Z]['max']

    def get_unit(self, axis: Axis) -> Units:
        '''
        Get the unit of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to query.

        Returns
        -------
        Units
            The unit of the stage.
        '''
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        return self._metadata[axis]['unit']

    def get_center(self, axis: Axis) -> float:
        '''
        Get the center position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to query.

        Returns
        -------
        float
            The center position of the stage.
        '''
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        return self._metadata[axis]['center']

    def set_center(self, axis: Axis, center: float) -> None:
        '''
        Set the center position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to set.
        center : float
            The center position.
        '''
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        self._metadata[axis]['center'] = center

    def get_max(self, axis: Axis) -> float:
        '''
        Get the maximum position of the stage along a specific axis.

        Parameters
        ----------
        axis : Axis
            The axis to query.

        Returns
        -------
        float
            The maximum position of the stage.
        '''
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        return self._metadata[axis]['max']

    def convert_units(self, x, y, z, from_unit: Optional[Units]):
        '''
        Convert stage coordinates from one unit to another.

        Parameters
        ----------
        x : float
            The x coordinate in the original units.
        y : float
            The y coordinate in the original units.
        z : float
            The z coordinate in the original units.
        from_unit : Optional[Units]
            The unit of the original coordinates.

        Returns
        -------
        tuple[float, float, float]
            The converted coordinates in the target units.
        '''
        if from_unit is None:
            return x, y, z

        if not isinstance(from_unit, Units):
            raise ValueError('from_unit must be an instance of Units enum.')

        mapping = {
            Axis.X: x,
            Axis.Y: y,
            Axis.Z: z,
        }

        return (
            Units.convert(value, from_unit, self.get_unit(axis))
            if axis in self.axes
            else value
            for axis, value in mapping.items()
        )

    def convert_to_nm(self, value: float, axis: Axis = Axis.Z) -> float:
        """
        Convert a value from the stage's native units to nanometers.

        Parameters
        ----------
        value : float
            The value in the stage's native units.
        axis : Axis
            The axis for which the conversion is being performed.

        Returns
        -------
        float
            The value converted to nanometers.
        """
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        return value * self._metadata[axis]['unit'].value

    def convert_from_nm(self, value: float, axis: Axis = Axis.Z) -> float:
        """
        Convert a value from nanometers to the stage's native units.

        Parameters
        ----------
        value : float
            The value in nanometers.
        axis : Axis
            The axis for which the conversion is being performed.

        Returns
        -------
        float
            The value converted to the stage's native units.
        """
        if axis not in self._metadata:
            raise ValueError(f'Axis {axis} not supported by this stage')

        return value / self._metadata[axis]['unit'].value

    def __str__(self):
        return f'{self._name} ({self.axis_label})'

    def get_config(self) -> dict:
        '''
        Get the current configuration of the stage.

        Returns
        -------
        dict
            The current configuration of the stage.
        '''
        config = {}

        if (
            self.driver.is_serial()
            and not self.driver.is_dual_serial
            and self.serial is not None
        ):
            config['serial'] = {
                'port': self.serial.portName(),
                'baudrate': self.serial.baudRate(),
            }

        for axis in self._axes:
            config[axis.value] = {
                'max': self._metadata[axis]['max'],
                'center': self._metadata[axis]['center'],
            }

        return config

    def load_config(self, config: dict):
        '''
        Load a configuration into the stage.

        Parameters
        ----------
        config : dict
            The configuration to load.
        '''
        if not isinstance(config, dict):
            raise ValueError('Config must be a dictionary.')

        if (
            self.driver.is_serial()
            and not self.driver.is_dual_serial
            and self.serial is not None
        ):
            serial_config = config.get('serial', {})
            if isinstance(serial_config, dict):
                self.setBaudRate(serial_config.get('baudrate', 115200))
                self.setPortName(serial_config.get('port', 'COM5'))

        for axis in self._axes:
            if axis.value in config:
                axis_config = config[axis.value]
                if not isinstance(axis_config, dict):
                    continue

                if 'max' in axis_config:
                    self.set_max_range(axis, axis_config['max'])
                if 'center' in axis_config:
                    self.set_center(axis, axis_config['center'])
