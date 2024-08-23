import struct
from enum import Enum


class DeviceStatusValues(Enum):
    '''Values that represent Device Status.'''

    OK = 0
    CommunicationError = 1
    MechanicalTimeOut = 2
    CommandError = 3
    ValueOutOfRange = 4
    ModuleIsolated = 5
    ModuleOutOfIsolation = 6
    InitializationError = 7
    ThermalError = 8
    Busy = 9
    SensorError = 10
    MotorError = 11
    OutOfRangeError = 12
    OverCurrentError = 13
    GeneralError = 14

    def __str__(self):
        return {
            0: 'Ok',
            1: 'Communication Error',
            2: 'Mechanical Time Out',
            3: 'Command Error',
            4: 'Value Out Of Range',
            5: 'Module Isolated',
            6: 'Module Out Of Isolation',
            7: 'Initialization Error',
            8: 'Thermal Error',
            9: 'Busy',
            10: 'Sensor Error',
            11: 'Motor Error',
            12: 'Out of range Error',
            13: 'Over current Error',
            14: 'General Error',
        }[self.value]


class DeviceStatusStruct:
    '''Structure containing device stage axis parameters.'''

    FORMAT = 'c2s2s'

    def __init__(self, address, command, status):
        self.address: str = address
        self.command: str = command
        self.status: int = status

    @classmethod
    def from_bytes(cls, data):
        address, command, status = struct.unpack(cls.FORMAT, data)
        return cls(address.decode('utf-8'), command.decode('utf-8'), int(status, 16))

class DeviceStatus:
    def __init__(self, message: str = None) -> None:
        if message is None:
            self.Status = DeviceStatusValues.OK
        else:
            ds = DeviceStatusStruct.from_bytes(message.encode('utf-8'))
            self.Status = DeviceStatusValues(ds.status)
