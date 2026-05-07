import logging
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Callable, Optional, Union

from matplotlib.pyplot import axes

from microEye.hardware.stages.stage import AbstractStage, Axis, Units, emit_after_signal
from microEye.qt import QtWidgets

try:
    import smaract.ctl as ctl
except ImportError:
    ctl = None

logger = logging.getLogger(__name__)


def assert_lib_compatibility():
    '''
    Checks that the major version numbers of the Python API and the
    loaded shared library are the same to avoid errors due to
    incompatibilities.
    Raises a RuntimeError if the major version numbers are different.
    '''
    if ctl is None:
        raise ImportError('SmarActCTL library not found.')

    vapi = ctl.api_version
    vlib = [int(i) for i in ctl.GetFullVersionString().split('.')]
    if vapi[0] != vlib[0]:
        raise RuntimeError('Incompatible SmarActCTL python api and library version.')


class MCS2Stage(AbstractStage):
    '''
    Class for controlling SmarAct MCS2 stages using the SmarActCTL library.

    **Notes**
        - The MCS2 stage can be configured with different types of channels,
          such as stick-slip piezo drivers, piezo scanner drivers, and magnetic drivers.
        - This class assumes a default configuration of 3 channels (X, Y, Z)
          with stick-slip piezo drivers.
        - Should work out of the box, but it can be adapted to other configurations by
          modifying the channel initialization and movement methods accordingly.
    '''
    NAME = 'SmarAct MCS2'

    CTL_LOCK = threading.RLock()

    DEFAULT_MAX_RANGE_PM = {
        Axis.X: 40_000_000_000,
        Axis.Y: 21_000_000_000,
        Axis.Z: 10_000_000_000,
    }

    CALIBRATING_STATE_MASK = (
        ctl.ChannelState.CALIBRATING
        | ctl.ChannelState.ACTIVELY_MOVING
        # | ctl.ChannelState.CLOSED_LOOP_ACTIVE
    ) if ctl is not None else 0

    REFERNCING_STATE_MASK = (
        ctl.ChannelState.CLOSED_LOOP_ACTIVE
        | ctl.ChannelState.REFERENCING
        | ctl.ChannelState.ACTIVELY_MOVING
    ) if ctl is not None else 0

    MOVE_STATE_MASK = (
        ctl.ChannelState.ACTIVELY_MOVING | ctl.ChannelState.CLOSED_LOOP_ACTIVE
    ) if ctl is not None else 0

    def _ctl_call(self, func: Callable, *args):
        with MCS2Stage.CTL_LOCK:
            return func(*args)

    def _get_channel_state(self, channel: int):
        try:
            return self._ctl_call(
                ctl.GetProperty_i32, self.__handle, channel, ctl.Property.CHANNEL_STATE
            )
        except ctl.Error as exc:
            logger.warning(f'MCS2 channel {channel} state read failed: {exc}')
            return None

    def __init__(
        self,
        locator: str,
        max_range: Optional[
            Union[Mapping[Union[Axis, str], float], Iterable[float], float]
        ] = None,
        units: Union[
            Units, Mapping[Union[Axis, str], Units], Iterable[Units]
        ] = Units.PICOMETERS,
        center: Optional[
            Union[Mapping[Union[Axis, str], float], Iterable[float], float]
        ] = 0,
    ):
        self._axis_map = {
            Axis.X: 0,
            Axis.Y: 1,
            Axis.Z: 2,
        }

        axes = [axis for axis in self._axis_map]
        max_range = (
            tuple(MCS2Stage.DEFAULT_MAX_RANGE_PM.values())
            if max_range is None
            else max_range
        )

        super().__init__(
            name=MCS2Stage.NAME,
            max_range=max_range,
            units=units,
            axes=axes,
            readyRead=None,
            center=center,
            min_factor=-1,
        )

        if locator is None:
            raise ValueError('MCS2 locator must be specified.')

        self.__handle = None
        self.locator = locator

    def is_open(self):
        return ctl is not None and self.__handle is not None

    def open(self):
        if not self.is_open():
            assert_lib_compatibility()
            locators = MCS2Stage.find_devices()
            if locators and self.locator in locators:
                try:
                    self.__handle = ctl.Open(self.locator)
                    self._init_info()
                    self._initialize_channels()
                    self._update_soft_limits()
                    logger.info(f'MCS2 connected to {self.locator}.')
                except Exception as e:
                    self.__handle = None
                    logger.error(f'MCS2 failed to connect to {self.locator}. {e}')
            else:
                logger.error(f'MCS2 device {self.locator} not found.')

    def _init_info(self):
        if not self.is_open():
            return

        # the index parameter is unused and must be set to zero.
        self._serial = self._ctl_call(
            ctl.GetProperty_s, self.__handle, 0, ctl.Property.DEVICE_SERIAL_NUMBER
        )
        self._name = self._ctl_call(
            ctl.GetProperty_s, self.__handle, 0, ctl.Property.DEVICE_NAME
        )

        logger.info(f'MCS2 device info - Serial: {self._serial}, Name: {self._name}.')

    def close(self):
        if self.is_open():
            try:
                self._ctl_call(ctl.Close, self.__handle)
                logger.info(f'MCS2 disconnected from {self.locator}.')
            finally:
                self.__handle = None

    @property
    def x(self) -> float:
        if Axis.X in self._axis_map:
            pos = self._read_axis_position(Axis.X)
            if pos is not None:
                self.set_position(Axis.X, pos)
        return self.get_position(Axis.X)

    @property
    def y(self) -> float:
        if Axis.Y in self._axis_map:
            pos = self._read_axis_position(Axis.Y)
            if pos is not None:
                self.set_position(Axis.Y, pos)
        return self.get_position(Axis.Y)

    @property
    def z(self) -> float:
        if Axis.Z in self._axis_map:
            pos = self._read_axis_position(Axis.Z)
            if pos is not None:
                self.set_position(Axis.Z, pos)
        return self.get_position(Axis.Z)

    def stop(self):
        if not self.is_open():
            logger.warning('MCS2 stop ignored: device not open.')
            return

        for axis in self._axis_map:
            channel = self._axis_map[axis]
            self._ctl_call(ctl.Stop, self.__handle, channel)

    @emit_after_signal('moveFinished')
    def home(self, is_async: bool = True):
        if not self.is_open():
            logger.warning('MCS2 home ignored: device not open.')
            return

        axes_to_wait = []

        def worker():
            for axis in self._axis_map:
                channel = self._axis_map[axis]

                if not self._has_sensor(channel):
                    logger.warning(
                        f'MCS2 channel {channel} has no sensor, skipping referencing.'
                    )
                    continue
                if self.is_axis_busy(axis):
                    logger.info(f'MCS2 axis {axis} busy, skipping referencing.')
                    continue

                ref_options = (
                    ctl.ReferencingOption.START_DIR | ctl.ReferencingOption.AUTO_ZERO
                )
                self._ctl_call(
                    ctl.SetProperty_i32,
                    self.__handle,
                    channel,
                    ctl.Property.REFERENCING_OPTIONS,
                    ref_options,
                )
                self._ctl_call(
                    ctl.SetProperty_i64,
                    self.__handle,
                    channel,
                    ctl.Property.MOVE_VELOCITY,
                    10000000000,
                )  # Set move velocity to 10 mm/s for referencing.
                self._ctl_call(
                    ctl.SetProperty_i64,
                    self.__handle,
                    channel,
                    ctl.Property.MOVE_ACCELERATION,
                    10000000000,
                )  # Set move acceleration to 10 mm/s2 for referencing.
                self._ctl_call(ctl.Reference, self.__handle, channel)
                axes_to_wait.append(axis)

        self.run_async(
            worker,
            is_async=is_async,
            wait_func=lambda: self.wait(axes=axes_to_wait),
        )

    @emit_after_signal('moveFinished')
    def calibrate(self, channel=None, is_async: bool = True):
        if not self.is_open():
            logger.warning('MCS2 calibrate ignored: device not open.')
            return

        if channel is not None and channel not in self._axis_map.values():
            logger.warning(f'MCS2 calibrate ignored: invalid channel {channel}.')
            return

        axes_to_wait = []

        def worker():
            for axis in self._axis_map:
                channel = self._axis_map[axis]

                if channel is not None and channel != channel:
                    continue

                self._calibrate(channel)
                axes_to_wait.append(axis)

        self.run_async(
            worker,
            is_async=is_async,
            wait_func=lambda: self.wait(axes=axes_to_wait),
        )

    def _calibrate(self, channel):
        if not self._has_sensor(channel):
            logger.warning(
                f'MCS2 channel {channel} has no sensor, skipping calibration.'
            )
            return

        logger.info(f'MCS2 start calibration on channel: {channel}.')
        # Set calibration options (start direction: forward)
        self._ctl_call(
            ctl.SetProperty_i32,
            self.__handle,
            channel,
            ctl.Property.CALIBRATION_OPTIONS,
            ctl.CalibrationOption.DIRECTION,
        )
        # Start calibration sequence
        self._ctl_call(ctl.Calibrate, self.__handle, channel)

    def is_axis_busy(self, axis: Axis) -> bool:
        channel = self._axis_map[axis]
        state = self._get_channel_state(channel)
        # only first 5 bits of state are used for flags, so we can ignore the rest
        state = state & 0b11111 if state is not None else None

        return bool(
            state is not None
            and state
            in [
                MCS2Stage.CALIBRATING_STATE_MASK,
                MCS2Stage.REFERNCING_STATE_MASK,
                MCS2Stage.MOVE_STATE_MASK,
            ]
        )

    def any_busy(self, axes: Optional[Iterable[Axis]] = None):
        if not self.is_open():
            return False

        axes = tuple(axes) if axes is not None else tuple(self._axis_map.keys())

        for axis in axes:  # noqa: SIM110
            if self.is_axis_busy(axis):
                return True

        return False

    def wait(
        self,
        timeout: float = 60.0,
        axes: Optional[Iterable[Axis]] = None,
    ):
        if not self.is_open():
            logger.warning('MCS2 wait ignored: device not open.')
            return

        time.sleep(0.2)
        start_time = time.monotonic()

        while self.any_busy(axes=axes):
            if (time.monotonic() - start_time) > timeout:
                logger.warning('MCS2 wait timeout exceeded.')
                break
            time.sleep(0.2)

        self.refresh_position()

    def wait_for_move_events(self, axes: Iterable[Axis], timeout_ms: int = 30000):
        if not self.is_open():
            return
        pending = {self._axis_map[a] for a in axes}
        deadline = time.monotonic() + (timeout_ms / 1000.0)

        while pending and time.monotonic() < deadline:
            try:
                event = self._ctl_call(ctl.WaitForEvent, self.__handle, 1000)
            except ctl.Error as exc:
                logger.warning(f'MCS2 WaitForEvent failed: {exc}')
                continue

            if event.type == ctl.EventType.MOVEMENT_FINISHED and event.idx in pending:
                if event.i32 != ctl.ErrorCode.NONE:
                    logger.warning(
                        f'MCS2 channel {event.idx} movement error: {event.i32}'
                    )
                pending.remove(event.idx)

        if pending:
            logger.warning(f'MCS2 wait_for_move_events timeout for channels: {pending}')
        self.refresh_position()

    @emit_after_signal('moveFinished')
    def move_absolute(self, x, y, z=0, **kwargs):
        if not self.is_open():
            logger.warning('MCS2 move ignored: device not open.')
            return

        is_async = kwargs.get('is_async', True)
        force = kwargs.get('force', False)
        targets = {Axis.X: x, Axis.Y: y, Axis.Z: z}

        axes_to_wait = []

        def worker():
            for axis, value in targets.items():
                if axis not in self._axis_map:
                    continue
                if value is None:
                    continue
                if self.is_axis_busy(axis) and not force:
                    logger.info(f'MCS2 axis {axis} busy, skipping movement.')
                    continue

                target = self._clamp(axis, value)
                self._move_axis_absolute(axis, target)
                axes_to_wait.append(axis)

        self.run_async(
            worker, is_async=is_async, wait_func=lambda: self.wait(axes=axes_to_wait)
        )

    @emit_after_signal('moveFinished')
    def move_relative(self, x, y, z=0, **kwargs):
        if not self.is_open():
            logger.warning('MCS2 move ignored: device not open.')
            return

        is_async = kwargs.get('is_async', True)
        deltas = {Axis.X: x, Axis.Y: y, Axis.Z: z}

        axes_to_wait = []

        def worker():
            for axis, delta in deltas.items():
                if axis not in self._axis_map:
                    continue
                if delta is None:
                    continue
                if self.is_axis_busy(axis):
                    logger.info(f'MCS2 axis {axis} busy, skipping movement.')
                    continue

                current = self.get_position(axis)
                target = self._clamp(axis, current + delta)
                delta = target - current
                if delta == 0:
                    continue

                self._move_axis_relative(axis, delta)
                axes_to_wait.append(axis)

        self.run_async(
            worker, is_async=is_async, wait_func=lambda: self.wait(axes=axes_to_wait)
        )

    def refresh_position(self):
        if not self.is_open():
            return

        for axis in self._axis_map:
            pos = self._read_axis_position(axis)
            if pos is None:
                continue
            self.set_position(axis, pos)
            self.signals.positionChanged.emit(self, pos, axis)

    @classmethod
    def find_devices(cls):
        '''
        Finds available MCS2 devices to connect to.

        Returns
        -------
        list[str]
            List of available device locators.
        '''
        try:
            assert_lib_compatibility()

            buffer = ctl.FindDevices()
            if len(buffer) == 0:
                logger.warning('MCS2 no devices found.')
                raise ConnectionError
            locators = buffer.split('\n')
            for locator in locators:
                logger.info(f'MCS2 available devices: {locator}')
        except Exception as e:
            locators = []
            logger.error(f'MCS2 failed to find devices. {e}')

        return locators.copy()

    @classmethod
    def get_stage(cls, **kwargs):
        locators = cls.find_devices()
        if len(locators) == 0:
            return None
        if len(locators) == 1:
            return cls(locator=locators[0], **kwargs)
        else:
            # If there are multiple instances, prompt the user to select one
            locator, ok = QtWidgets.QInputDialog.getItem(
                None,
                'Select MCS2 Device',
                'Select the MCS2 device to use:',
                locators,
            )
            if ok and locator:
                return cls(locator=locator, **kwargs)
            else:
                return None

    def get_config(self) -> dict:
        config = super().get_config()

        config['locator'] = self.locator

        return config

    def load_config(self, config: dict):
        if not isinstance(config, dict):
            raise ValueError('Config must be a dictionary.')

        super().load_config(config)

        locator = config.get('locator')
        if locator is not None:
            self.close()
            self.locator = locator

    def _initialize_channels(self):
        if not self.is_open():
            return

        no_of_channels = ctl.GetProperty_i32(
            self.__handle, 0, ctl.Property.NUMBER_OF_CHANNELS
        )
        logger.info(f'MCS2 number of channels: {no_of_channels}.')

        info = {}

        for channel in range(no_of_channels):
            name = ctl.GetProperty_s(
                self.__handle, channel, ctl.Property.POSITIONER_TYPE_NAME
            )
            pos_type = ctl.GetProperty_i32(
                self.__handle, channel, ctl.Property.POSITIONER_TYPE
            )
            state = ctl.GetProperty_i32(
                self.__handle, channel, ctl.Property.CHANNEL_STATE
            )
            type = ctl.GetProperty_i32(
                self.__handle, channel, ctl.Property.CHANNEL_TYPE
            )

            info[channel] = {
                'name': name,
                'positioner_type': pos_type,
                'state': state,
                'type': ctl.ChannelModuleType(type).name,
                'velocity': 5e9,
                'acceleration': 5e10,
                'has_sensor': (state & ctl.ChannelState.SENSOR_PRESENT) != 0,
            }

            # Leave default velocity and acceleration for now,
            # to the values set in the volatile memory of the controller
            # for best performance, but we can set them here if needed.
            # ctl.SetProperty_i64(
            #     self.__handle,
            #     channel,
            #     ctl.Property.MOVE_VELOCITY,
            #     int(info[channel]['velocity']),
            # )
            # ctl.SetProperty_i64(
            #     self.__handle,
            #     channel,
            #     ctl.Property.MOVE_ACCELERATION,
            #     int(info[channel]['acceleration']),
            # )

            # According to SmarAct support, setting the actuator mode to quiet
            # on a stick-slip piezo driver can reduce its lifetime and is just for demo
            # purposes, so I commented it out.

            # ctl.SetProperty_i32(
            #     self.__handle,
            #     channel,
            #     ctl.Property.ACTUATOR_MODE,
            #     ctl.ActuatorMode.QUIET,
            # )

            if type == ctl.ChannelModuleType.STICK_SLIP_PIEZO_DRIVER:
                # Set move mode to closed-loop relative movement.
                ctl.SetProperty_i32(
                    self.__handle,
                    channel,
                    ctl.Property.MOVE_MODE,
                    ctl.MoveMode.CL_RELATIVE,
                )
                # ctl.SetProperty_i32(
                #     self.__handle, channel, ctl.Property.MAX_CL_FREQUENCY, 6000
                # ) # valid range from 5-7 KHz, setting it to 6 KHz.
            elif type == ctl.ChannelModuleType.PIEZO_SCANNER_DRIVER:
                # Enable the amplifier for each channel to allow movement.
                ctl.SetProperty_i32(
                    self.__handle, channel, ctl.Property.AMPLIFIER_ENABLED, ctl.TRUE
                )
                ctl.SetProperty_i32(
                    self.__handle, channel, ctl.Property.HOLD_TIME, 1000
                )
            elif type == ctl.ChannelModuleType.MAGNETIC_DRIVER:
                # Enable the amplifier (and start the phasing sequence).
                ctl.SetProperty_i32(
                    self.__handle, channel, ctl.Property.AMPLIFIER_ENABLED, ctl.TRUE
                )

    def _update_axis_limits(self, axis: Axis):
        if not self.is_open():
            return

        channel = self._axis_map[axis]
        try:
            max_axis_value = int(self.get_max(axis))
            min_axis_value = int(self.get_min(axis))

            ctl.SetProperty_i64(
                self.__handle,
                channel,
                ctl.Property.RANGE_LIMIT_MAX,
                max_axis_value,
            )
            ctl.SetProperty_i64(
                self.__handle,
                channel,
                ctl.Property.RANGE_LIMIT_MIN,
                min_axis_value,
            )
            mm_max = self._from_device_units(axis, max_axis_value, Units.MILLIMETERS)
            mm_min = self._from_device_units(axis, min_axis_value, Units.MILLIMETERS)

            logger.info(
                f'MCS2 channel {channel} range: min={mm_min} mm, max={mm_max} mm.'
            )
        except Exception as exc:
            logger.debug(f'MCS2 failed to read max range for channel {channel}: {exc}')

    def _update_soft_limits(self):
        if not self.is_open():
            return

        for axis in self._axis_map:
            self._update_axis_limits(axis)

    def set_max_range(self, axis: Axis, max_range: float) -> None:
        super().set_max_range(axis, max_range)
        if self.is_open():
            self._update_axis_limits(axis)

    def set_min_range(self, axis: Axis, min_range: float) -> None:
        super().set_min_range(axis, min_range)
        if self.is_open():
            self._update_axis_limits(axis)

    def _move_axis_absolute(self, axis: Axis, position: float):
        channel = self._axis_map[axis]
        move_mode = ctl.MoveMode.CL_ABSOLUTE

        self._move_channel(channel, position, move_mode=move_mode)

    def _move_axis_relative(self, axis: Axis, delta: float):
        channel = self._axis_map[axis]

        move_mode = ctl.MoveMode.CL_RELATIVE

        self._move_channel(channel, delta, move_mode=move_mode)

    def _move_channel(
        self, channel: int, target: int, move_mode=None
    ):
        if move_mode is None:
            move_mode = ctl.MoveMode.CL_RELATIVE

        if not self._has_sensor(channel):
            logger.warning(f'MCS2 channel {channel} has no sensor, skipping movement.')
            return

        self._ctl_call(
            ctl.SetProperty_i32,
            self.__handle,
            channel,
            ctl.Property.MOVE_MODE,
            move_mode,
        )
        self._ctl_call(ctl.Move, self.__handle, channel, int(target), 0)

    def _has_sensor(self, channel):
        state = self._get_channel_state(channel)
        return (state & ctl.ChannelState.SENSOR_PRESENT) != 0

    def _read_axis_position(self, axis: Axis) -> Optional[float]:
        if not self.is_open():
            return None

        channel = self._axis_map[axis]

        if not self._has_sensor(channel):
            return None

        # position = self._ctl_call(
        #     ctl.GetProperty_i64, self.__handle, channel, ctl.Property.POSITION
        # )
        rID = self._ctl_call(
            ctl.RequestReadProperty, self.__handle, channel, ctl.Property.POSITION, 0
        )
        position = self._ctl_call(ctl.ReadProperty_i64, self.__handle, rID)

        return position

    def _clamp(self, axis: Axis, value: float, min_factor: int = -1) -> float:
        max_range = self.get_max(axis)
        return max(max_range * min_factor, min(value, max_range))

    def _to_device_units(
        self, axis: Axis, value: float, from_unit: Units = Units.NANOMETERS
    ) -> float:
        return Units.convert(value, from_unit, self.get_unit(axis))

    def _from_device_units(
        self, axis: Axis, value: float, target_unit: Units = Units.NANOMETERS
    ) -> float:
        return Units.convert(value, self.get_unit(axis), target_unit)


if __name__ == '__main__':
    import sys

    def print_position(stage: AbstractStage):
        time.sleep(0.5)

        for axis in stage.axes:
            pos = stage.get_position(axis)
            logger.info(f'Position changed - Axis: {axis}, New Position: {pos}')

    app = QtWidgets.QApplication(sys.argv)
    stage = MCS2Stage.get_stage()
    if stage is not None:
        stage.open()
        stage.calibrate()
        stage.wait(axes=stage.axes)
        print_position(stage)
        stage.home()
        stage.wait(axes=stage.axes)
        print_position(stage)

        step = stage._to_device_units(Axis.X, 10, Units.MILLIMETERS)
        for i in range(5):
            factor = 1 if i % 2 == 0 else 0
            stage.move_absolute(x=int(step) * factor, y=0, z=0)

            stage.wait(axes=stage.axes)
            print_position(stage)
        logger.info('Centering stage.')
        stage.center()
        stage.wait(axes=stage.axes)
        print_position(stage)

        stage.close()
    else:
        logger.error('No MCS2 stage found.')
