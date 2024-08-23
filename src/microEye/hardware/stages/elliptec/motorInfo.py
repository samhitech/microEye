import struct
from enum import Enum


class SwitchState(Enum):
    ON = '1'
    OFF = '0'


class MotorInfoStruct:
    '''Structure containing stage axis parameters.'''

    FORMAT = 'c2scc4s4s4s4s4s'
    SIZE = struct.calcsize(FORMAT)

    def __init__(
        self,
        address: bytes,
        command: bytes,
        loop_state: bytes,
        motor_state: bytes,
        current: bytes,
        ramp_up: bytes,
        ramp_down: bytes,
        fwd_period: bytes,
        rev_period: bytes,
    ):
        self.address = address.decode('utf-8')
        self.command = command.decode('utf-8')
        self.motor_id = command.decode('utf-8')[1]
        self.loop_state = SwitchState(loop_state.decode('utf-8'))
        self.motor_state = SwitchState(motor_state.decode('utf-8'))
        self.current = int(current, 16) / 1866
        self.ramp_up = int(ramp_up, 16)
        self.ramp_down = int(ramp_down, 16)
        self.fwd_period = int(fwd_period, 16)
        self.rev_period = int(rev_period, 16)

    @classmethod
    def from_bytes(cls, data):
        return cls(*struct.unpack(MotorInfoStruct.FORMAT, data))

class MotorInfo:
    def __init__(self, motor_id: str = None):
        self.motor_id = motor_id
        self.motor_state = SwitchState.OFF
        self.is_valid = False

        self.loop_state = None
        self.current = 0.0
        self.ramp_up = 0
        self.ramp_down = 0
        self.fwd_freq = 0
        self.rev_freq = 0

    @classmethod
    def from_string(cls, id_string: str):
        motor_info_struct = MotorInfoStruct.from_bytes(id_string.encode('ascii'))
        motor_info = cls('')
        motor_info.motor_id = motor_info_struct.motor_id
        motor_info.motor_state = motor_info_struct.motor_state
        motor_info.loop_state = motor_info_struct.loop_state
        motor_info.current = motor_info_struct.current
        motor_info.ramp_up = motor_info_struct.ramp_up
        motor_info.ramp_down = motor_info_struct.ramp_down
        motor_info.fwd_freq = 14740 / motor_info_struct.fwd_period
        motor_info.rev_freq = 14740 / motor_info_struct.rev_period
        motor_info.is_valid = True
        return motor_info

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @is_valid.setter
    def is_valid(self, value: bool):
        self._is_valid = value

    @property
    def loop_state(self) -> SwitchState:
        return self._loop_state

    @loop_state.setter
    def loop_state(self, value: SwitchState):
        self._loop_state = value

    @property
    def motor_state(self) -> SwitchState:
        return self._motor_state

    @motor_state.setter
    def motor_state(self, value: SwitchState):
        self._motor_state = value

    @property
    def current(self) -> float:
        return self._current

    @current.setter
    def current(self, value: float):
        self._current = value

    @property
    def ramp_up(self) -> int:
        return self._ramp_up

    @ramp_up.setter
    def ramp_up(self, value: int):
        self._ramp_up = value

    @property
    def ramp_down(self) -> int:
        return self._ramp_down

    @ramp_down.setter
    def ramp_down(self, value: int):
        self._ramp_down = value

    @property
    def fwd_freq(self) -> float:
        return self._fwd_freq

    @fwd_freq.setter
    def fwd_freq(self, value: float):
        self._fwd_freq = value

    @property
    def rev_freq(self) -> float:
        return self._rev_freq

    @rev_freq.setter
    def rev_freq(self, value: float):
        self._rev_freq = value

    @property
    def motor_id(self) -> str:
        return self._motor_id

    @motor_id.setter
    def motor_id(self, value: str):
        self._motor_id = value

    def description(self) -> list:
        return [
            f'Motor ID {self.motor_id}',
            f'Loop State {self.loop_state}',
            f'Motor State {self.motor_state}',
            f'Current {self.current:.2f}A',
            f'Fwd Frequency {self.fwd_freq:.1f}kHz',
            f'Rev Frequency {self.rev_freq:.1f}kHz',
            f'RampUp {self.ramp_up}',
            f'RampDown {self.ramp_down}',
        ]
