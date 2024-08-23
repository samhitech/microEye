from enum import Enum
from threading import Lock
from typing import Any, Optional

from microEye.hardware.stages.elliptec.deviceID import DeviceID
from microEye.hardware.stages.elliptec.devicePort import DevicePort, ELLException
from microEye.hardware.stages.elliptec.deviceStatus import (
    DeviceStatus,
    DeviceStatusValues,
)
from microEye.hardware.stages.elliptec.messageUpdater import MessageUpdater
from microEye.hardware.stages.elliptec.motorInfo import MotorInfo
from microEye.qt import QtCore


class StatusResult(Enum):
    '''Values that represent status results.'''

    OK = 1
    Busy = 2
    Error = 3
    Fail = 4


class DeviceDirection(Enum):
    '''Values that represent Device Direction.

    These are used to set the homing direction for a device
    '''

    Linear = '0'
    Clockwise = '0'
    AntiClockwise = '1'

    def __str__(self):
        return {
            '0': 'Linear',
            '0': 'Clockwise',
            '1': 'AntiClockwise',
        }[self.value]


class BaseDevice(QtCore.QObject):
    VALID_ADDRESSES = tuple(
        [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            'A',
            'B',
            'C',
            'D',
            'E',
            'F',
        ]
    )

    CLEAN_TIMEOUT = 10000  # The clean timeout
    MOVE_TIMEOUT = 4000  # The move timeout
    SEARCH_TIMEOUT = 8000  # The search timeout
    HOME_TIMEOUT = 6000  # The home timeout
    STATUS_TIMEOUT = 1000  # The status timeout

    def __init__(
        self,
        device_id: DeviceID,
        motor_count: int,
        message_updater: Optional[MessageUpdater] = None,
    ):
        super().__init__()
        self._lock = Lock()
        self._message_updater = message_updater
        self._motor_info: dict[Any, MotorInfo] = {}
        self.deviceInfo = device_id
        self.address = device_id.Address
        self.deviceStatus = DeviceStatusValues.OK
        self.auto_save = False
        self._initialize_class(device_id, motor_count)

    def _initialize_class(self, device_id, motor_count: int):
        for c in range(1, motor_count + 1):
            self._motor_info[str(c)] = MotorInfo(str(c))

    @staticmethod
    def configure(device_id, message_updater: MessageUpdater):
        if not device_id:
            return None
        address = device_id[0]
        if not BaseDevice.is_valid_address(address):
            return None
        device_id_obj = DeviceID(device_id)
        if message_updater:
            message_updater.update_parameter(
                MessageUpdater.UpdateTypes.DeviceInfo, address, device_id_obj
            )
        return device_id_obj

    @property
    def valid_addresses(self):
        return self.VALID_ADDRESSES

    @staticmethod
    def is_valid_address(address):
        return address in BaseDevice.VALID_ADDRESSES

    @property
    def is_address_valid(self):
        return self.address in self.VALID_ADDRESSES

    def save_user_data(self):
        self._update_output('Save user status...')
        DevicePort.send_command(self.address, 'us')
        return self._wait_for_status()

    def set_address(self, new_address):
        if new_address == self.address:
            return True
        self._update_output(f'changing address to {new_address}...')
        msg = DevicePort.send_string_b(self.address, 'ca', new_address)
        self.address = new_address
        return msg

    def is_motor_id_valid(self, c):
        return c in self._motor_info

    def get_motor_info(self, motor_id):
        if not self.is_motor_id_valid(motor_id):
            return False
        self._update_output(f'Requesting Motor{motor_id} info...')
        DevicePort.send_command(self.address, 'i', motor_id)
        try:
            msg = DevicePort.wait_for_response(
                self.address, 'I' + motor_id, self.MOVE_TIMEOUT
            )
            if msg:
                info = MotorInfo.from_string(msg)
                self._motor_info[motor_id] = info
                self._update_output('Get Device Status: ', info.description())
                self._update_parameter(
                    MessageUpdater.UpdateTypes.MotorInfo, self.address, info
                )
                return True
        except ELLException:
            self._update_output(f'Requesting Motor{motor_id}: FAILED')
        return False

    def set_period(self, motor_id, fwd, frequency, permanent, hard_save_frequency):
        if not self.is_motor_id_valid(motor_id):
            return False
        period = int(14740 / frequency)
        self._update_output(
            f"Motor{motor_id} - Setting {'Fwd' if fwd else 'Bwd'} period to {period:X}"
            f"({frequency}kHz) ..."
        )
        if hard_save_frequency:
            period |= 0x8000
        DevicePort.send_string_i16(
            self.address, ('f' if fwd else 'b') + motor_id, period
        )
        if not self._wait_for_status():
            return False
        if permanent:
            self.save_user_data()
        return True

    def search_period(self, motor_id, permanent):
        if not self.is_motor_id_valid(motor_id):
            return False
        self._update_output(f'Motor{motor_id} - Search period ...')
        DevicePort.send_command(self.address, 's', motor_id)
        if not self._wait_for_status(self.SEARCH_TIMEOUT):
            return False
        self.get_motor_info(motor_id)
        if permanent:
            self.save_user_data()
        return True

    def reset_period(self):
        motors = [
            key
            for key in self._motor_info
            if self._motor_info[key] and self._motor_info[key].is_valid
        ]
        for motor_id in motors:
            self._update_output(f'Motor{motor_id} - Reset default period ...')
            period = int(14740 / 100) | 0x8000
            DevicePort.send_string_i16(self.address, 'f' + motor_id, period)
            if not self._wait_for_status(self.SEARCH_TIMEOUT):
                return False
            DevicePort.send_string_i16(self.address, 'b' + motor_id, period)
            if not self._wait_for_status(self.SEARCH_TIMEOUT):
                return False
            self.get_motor_info(motor_id)
        self.save_user_data()
        return True

    def scan_current_curve(self, motor_id):
        if not self.is_motor_id_valid(motor_id):
            return False
        self._update_output(f'Motor{motor_id} - Scan Current ...')
        DevicePort.send_command(self.address, 'c', motor_id)
        if not self._wait_for_status(6000):
            return False
        self.get_motor_info(motor_id)
        return True

    def _update_parameter(self, update_type, address, data):
        if self._message_updater:
            self._message_updater.update_parameter(update_type, address, data)

    def _update_output(self, message, error=False):
        if self._message_updater:
            self._message_updater.update_output(message, error)

    def _update_device_status(self, device_status: DeviceStatusValues):
        error = device_status not in [
            DeviceStatusValues.OK,
            DeviceStatusValues.Busy,
        ]
        self._update_output('Get Device Status: ' + device_status.name, error)
        self._update_parameter(
            MessageUpdater.UpdateTypes.Status, self.address, device_status
        )
        self.deviceStatus = device_status

    def get_device_status(self):
        self._update_output('Get Device Status...')
        DevicePort.send_command(self.address, 'gs')
        return self._wait_for_status()

    def _wait_for_status(self, ms_timeout=MOVE_TIMEOUT):
        return self._wait_for_status_internal(self.address, ms_timeout)

    def _wait_for_status_internal(self, address, ms_timeout=MOVE_TIMEOUT, getter=False):
        try:
            counter = 10
            while True:
                msg = DevicePort.wait_for_response(address, 'GS', ms_timeout)
                if msg:
                    return_value, counter = self._test_status(
                        msg, 'gs', getter, counter
                    )
                    if return_value is not None:
                        return return_value
        except ELLException as ex:
            self._update_output(f'Get Device Status: {ex}', True)
        return False

    def _test_status(self, msg: str, cmd: str, getter: bool, counter: int):
        if msg[1:3] == 'GS':
            result = self.process_device_status(DeviceStatus(msg).Status, cmd, getter)
            if result in [
                StatusResult.Fail,
                StatusResult.Busy,
            ]:
                return False, counter
            if result == StatusResult.OK and not getter:
                return True, counter
            if result == StatusResult.Error:
                return False, counter
            counter -= 1
            if counter == 0:
                return False, counter
        return None, counter

    def process_device_status(
        self, device_status: DeviceStatus, resend_cmd: str, getter: bool
    ):
        raise NotImplementedError('This method should be implemented by subclasses.')

    def set_sleep_delay_counter(self, delay_counter: int):
        with self._lock:
            self._sleep_duration_counter = delay_counter

    def get_sleep_delay_counter(self):
        with self._lock:
            return self._sleep_duration_counter

    def wait_for_cleaning(self, ms_timeout=None):
        if ms_timeout is None:
            ms_timeout = BaseDevice.CLEAN_TIMEOUT

        try:
            counter = 60
            while True:
                msg = DevicePort.wait_for_response(self.address, 'GS', ms_timeout)
                if msg:
                    if msg[1:5] == 'GS09':
                        self.set_sleep_delay_counter(10)
                        while self.get_sleep_delay_counter() >= 0:
                            self.set_sleep_delay_counter(
                                self.get_sleep_delay_counter() - 1
                            )
                            QtCore.QThread.msleep(1000)
                        DevicePort.send_command(self.address, 'gs')
                    else:
                        return_value, counter = self._test_status(
                            msg, 'gs', False, counter
                        )
                        if return_value is not None:
                            return return_value
        except ELLException as ex:
            if self._message_updater:
                self._message_updater.update_output(
                    f'Get Device Status: {str(ex)}', True
                )
        return False
