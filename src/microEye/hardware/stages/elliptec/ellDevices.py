from typing import Optional

from microEye.hardware.stages.elliptec.baseDevice import BaseDevice
from microEye.hardware.stages.elliptec.device import SHUTTER_POSITION_FACTOR, EllDevice
from microEye.hardware.stages.elliptec.deviceID import DeviceID
from microEye.hardware.stages.elliptec.devicePort import DevicePort, ELLException
from microEye.hardware.stages.elliptec.messageUpdater import MessageUpdater
from microEye.qt import QtCore


class ELLDevices(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self._selected_device: Optional[EllDevice] = None
        self._devices: dict[str, EllDevice] = {}
        self._message_updater = MessageUpdater()

    @property
    def IsOpen(self) -> bool:
        return DevicePort.isOpen()

    def SendFreeCommand(self, free_command: str) -> bool:
        self._message_updater.update_output('Sending free command...')
        DevicePort.sendFreeString(free_command)
        return True

    def scanAddresses(
        self, min_device_address: str, max_device_address: str
    ) -> list[str]:
        connections = []
        if not DevicePort.isOpen():
            return connections

        self._message_updater.update_output('Scanning for devices')
        for address in [
            a
            for a in BaseDevice.VALID_ADDRESSES
            if min_device_address <= a <= max_device_address
        ]:
            DevicePort.send_command(address, 'in')
            try:
                msg = DevicePort.wait_for_response(
                    address, 'IN', BaseDevice.STATUS_TIMEOUT
                )
                if msg:
                    connections.append(msg)
            except ELLException as e:
                DevicePort.instance().logger.error(f'Error in scanAddresses: {str(e)}')
                DevicePort.instance().errorOccurred.emit(str(e))
                # device address not used

        count = len(connections)
        self._message_updater.update_output(
            f"{count} device{'s' if count != 1 else ''} found"
        )
        return connections

    def ClearDevices(self):
        self._devices.clear()

    def configure(self, device_id: str) -> bool:
        if not device_id:
            return False

        address = device_id[0]
        if address in BaseDevice.VALID_ADDRESSES:
            di = BaseDevice.configure(device_id, self._message_updater)
            if di.DeviceType == DeviceID.DeviceTypes.Paddle:
                return False
                # self._devices[address] = ELLPaddlePolariser(di, self._message_updater)
            elif di.DeviceType == DeviceID.DeviceTypes.Actuator:
                self._devices[address] = EllDevice(di, 1, self._message_updater)
            else:
                self._devices[address] = EllDevice(di, 2, self._message_updater)
            self._selected_device = self._devices[address]
        return True

    @property
    def SelectedDevice(self):
        return self._selected_device

    def ReaddressDevice(self, old_address: str, new_address: str) -> bool:
        device = self.addressedDevice(old_address)
        if not device:
            return False
        if not device.set_address(new_address):
            return False
        self._devices[new_address] = self._devices.pop(old_address)
        return True

    def addressedDevice(self, address: str):
        return self._devices.get(address)

    def ValidAddresses(self, addresses: list[str]) -> list[str]:
        return [addr for addr in addresses if addr in self._devices]

    @property
    def MessageUpdates(self) -> MessageUpdater:
        return self._message_updater

    def open(self) -> bool:
        DevicePort.open()

    def close(self) -> bool:
        DevicePort.close()

    def baudrate(self) -> int:
        return DevicePort.baudRate()

    def set_baudrate(self, baudrate: int) -> bool:
        return DevicePort.setBaudRate(baudrate)

    def portname(self) -> str:
        return DevicePort.portName()

    def set_portname(self, portname: str) -> bool:
        return DevicePort.setPortName(portname)

    def get_config(self) -> dict:
        return {
            'port': DevicePort.portName(),
            'baudrate': DevicePort.baudRate(),
        }

    def load_config(self, config: dict) -> bool:
        if 'port' in config:
            self.set_portname(config['port'])
        if 'baudrate' in config:
            self.set_baudrate(config['baudrate'])
        return True

    @property
    def ValidAddress(self) -> list[str]:
        return BaseDevice.VALID_ADDRESSES
