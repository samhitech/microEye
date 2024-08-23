from collections import defaultdict
from typing import Optional

from microEye.hardware.stages.elliptec.baseDevice import (
    BaseDevice,
    DeviceDirection,
    StatusResult,
)
from microEye.hardware.stages.elliptec.deviceID import DeviceID
from microEye.hardware.stages.elliptec.devicePort import DevicePort
from microEye.hardware.stages.elliptec.deviceStatus import (
    DeviceStatusValues,
)
from microEye.hardware.stages.elliptec.messageUpdater import MessageUpdater
from microEye.qt import QtCore, Signal

SHUTTER_POSITION_FACTOR = defaultdict(
    lambda: 1,
    {
        DeviceID.DeviceTypes.Shutter2: 32,
        DeviceID.DeviceTypes.Shutter4: 32,
        DeviceID.DeviceTypes.Shutter6: 19,
    },
)
'''
Dictionary mapping shutter device types to their respective position factors.

This factor is used for shutters when performing absolute movements. For example,
if the position index `i` is 2, the actual position `pos` is calculated as:
    pos = i * factor

Where `factor` is the value from this dictionary corresponding to the device type.
'''


class EllDevice(BaseDevice):
    cleaningUpdate = Signal(bool)

    def __init__(
        self,
        device_id: DeviceID,
        motor_count: int,
        message_updater: Optional[MessageUpdater] = None,
    ):
        super().__init__(device_id, motor_count, message_updater)
        self._position = 0
        self._home_offset = 0
        self._jogstep_size = 0

        self._is_jogging = False
        self._is_thermal_locked = False
        self._is_cleaning = False

        # self._jog_timer = None

    @property
    def home_offset(self):
        return self._home_offset

    @property
    def jogstep_size(self):
        return self._jogstep_size

    @property
    def position(self):
        return self._position

    def set_home_offset(self, offset: float) -> bool:
        self._update_output(
            'Set Home Offset to '
            f'{self.deviceInfo.format_position(offset, True, True)}...'
        )
        pulses = self.deviceInfo.unit_to_pulse(offset)
        DevicePort.send_string_i32(self.address, 'so', pulses)
        if not self.wait_for_home_offset('gs', False):
            return False
        self._home_offset = offset
        return True

    def get_home_offset(self) -> bool:
        self._update_output('Get Home Offset...')
        DevicePort.send_command(self.address, 'go')
        return self.wait_for_home_offset('go', True)

    def home(self, direction: DeviceDirection = DeviceDirection.Linear) -> bool:
        if self.is_device_busy():
            return False
        self._update_output('Homing device ...')
        DevicePort.send_command(self.address, f'ho{direction.value}')
        return self.wait_for_position(BaseDevice.HOME_TIMEOUT)

    def home_multiple(
        self, addresses: list[str], direction: DeviceDirection = DeviceDirection.Linear
    ) -> bool:
        if self.is_device_busy():
            return False
        if not addresses:
            return self.home(direction)
        self.set_to_group_address(addresses)
        self._update_output('Homing device ...')
        DevicePort.send_command(self.address, f'ho{direction.value}')
        return self.wait_for_positions(addresses)

    def set_jogstep_size(self, jogstep_size: float) -> bool:
        self._update_output(
            'Set Jog Step to ' f'{self.deviceInfo.format_position(jogstep_size)}...'
        )
        pulses = self.deviceInfo.unit_to_pulse(jogstep_size)
        DevicePort.send_string_i32(self.address, 'sj', pulses)
        if not self.wait_for_jogstep_size('gs', False):
            return False
        self._jogstep_size = jogstep_size

        return True

    def get_jogstep_size(self) -> bool:
        self._update_output('Get Jogstep Size...')
        DevicePort.send_command(self.address, 'gj')
        return self.wait_for_jogstep_size('gj', True)

    def start_jogging(self, command: str, message: str):
        self._is_jogging = True
        self._update_output(message)
        DevicePort.send_command(self.address, command)

        QtCore.QTimer.singleShot(100, self.jog_time_update)

    def jog_time_update(self):
        if self._is_jogging:
            self.get_position()
            QtCore.QTimer.singleShot(500, self.jog_time_update)

    def jog_stop_internal(self):
        self._is_jogging = False

    def jog_forward_start(self) -> bool:
        if self.is_device_busy() or self._is_jogging:
            return False
        self.start_jogging('fw', 'Jog Forward Start')
        return True

    def jog_backward_start(self) -> bool:
        if self.is_device_busy() or self._is_jogging:
            return False
        self.start_jogging('bw', 'Jog Backward Start')
        return True

    def jog_stop(self) -> bool:
        if not self._is_jogging:
            return False
        self.jog_stop_internal()
        self._update_output('Jog Stop')
        DevicePort.send_command(self.address, 'ms')
        return self.get_position()

    def jog_forward(self) -> bool:
        if self.is_device_busy():
            return False
        value = self.deviceInfo.format_position(self._jogstep_size, True, True)
        self._update_output(f'Jog Forward {value}')
        DevicePort.send_command(self.address, 'fw')
        return self.wait_for_position(BaseDevice.MOVE_TIMEOUT)

    def jog_backward(self) -> bool:
        if self.is_device_busy():
            return False
        value = self.deviceInfo.format_position(self._jogstep_size, True, True)
        self._update_output(f'Jog Backward {value}')
        DevicePort.send_command(self.address, 'bw')
        return self.wait_for_position(BaseDevice.MOVE_TIMEOUT)

    def get_position(self) -> bool:
        self._update_output('Get Positions...')
        DevicePort.send_command(self.address, 'gp')
        return self.wait_for_position(BaseDevice.MOVE_TIMEOUT)

    def move_absolute(self, position: float) -> bool:
        if self.is_device_busy():
            return False
        value = self.deviceInfo.format_position(position, True, True)
        self._update_output(f'Move device to {value}...')
        pulses = self.deviceInfo.unit_to_pulse(position)
        DevicePort.send_string_i32(self.address, 'ma', pulses)
        return self.wait_for_position(BaseDevice.MOVE_TIMEOUT)

    def move_relative(self, step: float) -> bool:
        if self.is_device_busy():
            return False
        self._update_output(
            f'Move device by {self.deviceInfo.format_position(step, True, True)}...'
        )
        pulses = self.deviceInfo.unit_to_pulse(step)
        if pulses == 0:
            return True
        DevicePort.send_string_i32(self.address, 'mr', pulses)
        return self.wait_for_position(BaseDevice.MOVE_TIMEOUT)

    def is_device_busy(self) -> bool:
        return self.test_thermal_initialized() or self.test_cleaning_state()

    def test_thermal_initialized(self) -> bool:
        if self._is_thermal_locked:
            self._update_output('Thermal Lockout')
            return True
        return False

    def test_cleaning_state(self) -> bool:
        if self._is_cleaning:
            self._update_output('Cleaning, Please wait...')
            return True
        return False

    def set_cleaning_state(self, is_cleaning: bool):
        self._is_cleaning = is_cleaning
        self.cleaningUpdate.emit(self._is_cleaning)

    def send_clean_machine(self) -> bool:
        if self.is_device_busy():
            return False
        self.set_cleaning_state(True)
        self._update_output('Performing clean mechanics...')
        DevicePort.send_command(self.address, 'cm')
        retval = self.wait_for_cleaning()
        self.set_cleaning_state(False)
        return retval

    def send_clean_and_optimize(self) -> bool:
        if self.is_device_busy():
            return False
        self.set_cleaning_state(True)
        self._update_output('Performing clean & optimize...')
        DevicePort.send_command(self.address, 'om')
        retval = self.wait_for_cleaning()
        self.set_cleaning_state(False)
        return retval

    def send_stop_cleaning(self) -> bool:
        if not self._is_cleaning:
            return False
        self._update_output('Stop cleaning...')
        DevicePort.send_command(self.address, 'st')
        self.set_sleep_delay_counter(2)
        DevicePort.send_command(self.address, 'gs')
        return True

    def set_to_group_address(self, addresses: list[str]) -> bool:
        for address in addresses:
            if address != self.address:
                self._update_output(f'Set GroupAddress {address}->{self.address}...')
                DevicePort.send_string_b(address, 'ga', ord(self.address))
                if not self._wait_for_status(self.address):
                    return False
        return True

    def process_device_status(
        self, device_status: DeviceStatusValues, resend_cmd: str, getter: bool
    ) -> StatusResult:
        if device_status == DeviceStatusValues.Busy:
            self._update_device_status(device_status)
            if getter:
                DevicePort.send_command(self.address, resend_cmd)
            return StatusResult.Busy
        elif device_status == DeviceStatusValues.OK:
            self._is_thermal_locked = False
            self._update_device_status(device_status)
            if getter:
                DevicePort.send_command(self.address, resend_cmd)
            return StatusResult.OK
        elif device_status == DeviceStatusValues.ThermalError:
            self.jog_stop_internal()
            self._is_thermal_locked = True
            self._update_device_status(device_status)
            QtCore.QThread.msleep(2000)
            DevicePort.send_command(self.address, 'gs')
        elif device_status == DeviceStatusValues.ModuleIsolated:
            self._update_device_status(device_status)
            return StatusResult.Fail
        else:
            self._update_device_status(device_status)
            QtCore.QThread.msleep(500)
            DevicePort.send_command(self.address, resend_cmd)
        return StatusResult.Error

    def wait_for_position(self, ms_timeout: int) -> bool:
        temp_to = ms_timeout
        if self.deviceInfo.DeviceType == DeviceID.DeviceTypes.Actuator:
            temp_to = 10000
        return self.wait_for_positions([self.address], temp_to)

    def wait_for_positions(self, addresses: list[str], ms_timeout: int = 30000) -> bool:
        temp_to = ms_timeout
        try:
            is_completed_list = {c: False for c in addresses}
            responses = ['GS', 'PO']
            counter = 10 * len(addresses)

            while True:
                if self.deviceInfo.DeviceType == DeviceID.DeviceTypes.Actuator:
                    temp_to = 10000

                msg = DevicePort.wait_for_response(addresses, responses, temp_to)
                if msg:
                    return_value, counter = self._test_status(msg, 'gp', True, counter)
                    if return_value:
                        return return_value

                    address = msg[0]
                    if msg[1:3] == 'PO' and address in addresses:
                        if len(msg) != 11:
                            return False
                        self._position = self.deviceInfo.pulse_to_unit(
                            int.from_bytes(
                                bytes.fromhex(msg[3:]), byteorder='big', signed=True
                            )
                        )
                        self._update_parameter(
                            MessageUpdater.UpdateTypes.Position, address, self._position
                        )
                        is_completed_list[address] = True
                        if all(is_completed_list.values()):
                            return True
        except Exception as ex:
            self._update_output(f'Get Device Status: {str(ex)}', True)
        return False

    def wait_for_home_offset(
        self, cmd: str, getter: bool, ms_timeout: int = 30000
    ) -> bool:
        try:
            responses = ['GS', 'HO']
            counter = 10
            while True:
                msg = DevicePort.wait_for_response(self.address, responses, ms_timeout)
                if msg:
                    return_value, counter = self._test_status(msg, cmd, getter, counter)
                    if return_value:
                        return return_value
                    if msg[1:3] == 'HO':
                        if len(msg) != 11:
                            return False
                        self._home_offset = self.deviceInfo.pulse_to_unit(
                            int.from_bytes(
                                bytes.fromhex(msg[3:]), byteorder='big', signed=True
                            )
                        )
                        self._update_parameter(
                            MessageUpdater.UpdateTypes.HomeOffset,
                            self.address,
                            self._home_offset,
                        )
                        return True
        except Exception as ex:
            self._update_output(f'Get Device Status: {str(ex)}', True)
        return False

    def wait_for_jogstep_size(
        self, cmd: str, getter: bool, ms_timeout: int = 30000
    ) -> bool:
        try:
            responses = ['GS', 'GJ']
            counter = 10
            while True:
                msg = DevicePort.wait_for_response(self.address, responses, ms_timeout)
                if msg:
                    return_value, counter = self._test_status(msg, cmd, getter, counter)
                    if return_value:
                        return return_value
                    if msg[1:3] == 'GJ':
                        if len(msg) != 11:
                            return False
                        self._jogstep_size = self.deviceInfo.pulse_to_unit(
                            int.from_bytes(
                                bytes.fromhex(msg[3:]), byteorder='big', signed=True
                            )
                        )
                        self._update_parameter(
                            MessageUpdater.UpdateTypes.JogstepSize,
                            self.address,
                            self._jogstep_size,
                        )
                        return True
        except Exception as ex:
            self._update_output(f'Get Device Status: {str(ex)}', True)
        return False
