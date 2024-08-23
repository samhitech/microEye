from enum import Enum

from microEye.qt import QtCore, Signal


class ThreadType(Enum):
    Imperial = 1
    Metric = 0

class ElliptecStage(QtCore.QObject):
    positionChanged = Signal(int)
    '''Signal emitted when position is updated'''
    statusChanged = Signal(int, str)
    '''Signal emitted when status is updated'''

    def __init__(self, address: str, **kwargs):
        super().__init__()
        self.address = str(address)
        # self.type = ElliptecType(kwargs.get('Model'))
        self.serial_number = kwargs.get('Serial_number', 'N/A')
        self.year = kwargs.get('Year', 'N/A')
        self.firmware_release = kwargs.get('Firmware_release', 'N/A')
        self.hardware_release = kwargs.get('Hardware_release', -1)
        self.thread_type = ThreadType(kwargs.get('Thread_type', 0))
        self.travel = kwargs.get('Travel', -1)
        self.pulses_per_mu = kwargs.get('Pulses_per_mu', -1)
        self._position = 0

        self.last_response = None
        self.last_status = None

    @property
    def position(self):
        return self._position / self.pulses_per_unit()

    @position.setter
    def position(self, value: int):
        if isinstance(value, int):
            self._position = value
            self.positionChanged.emit(self.position)

    def update_status(self, status_code: int, status_message: str):
        self.last_status = status_message
        self.statusChanged.emit(status_code, status_message)

    def pulses_per_unit(self):
        return self.pulses_per_mu / self.travel

    def __str__(self):
        return (
            f'ElliptecStage(address={self.address}, '
            f'type={self.type.name}, '
            # f'serial_number={self.serial_number}, '
            # f'year={self.year}, '
            # f'firmware_release={self.firmware_release}, '
            # f'hardware_release={self.hardware_release}, '
            # f'thread_type={self.thread_type.name}, '
            # f'travel={self.travel}, '
            # f'pulses_per_mu={self.pulses_per_mu}, '
            f'position={self.position / self.pulses_per_unit()}, '
            f'last_status={self.last_status})'
        )
