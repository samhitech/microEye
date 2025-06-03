import re

from microEye.hardware.stages.stage import (
    AbstractStage,
    Axis,
    Units,
    ZStageParams,
    ZStageView,
)
from microEye.qt import QtCore


class PzFoc(AbstractStage):
    '''PiezoConcept FOC 1-axis stage adapter.'''

    def __init__(self, max_um: int = 100):
        '''
        Initializes a PiezoConcept FOC 1-axis stage adapter.

        Parameters
        ----------
        max_um : int, optional
            Maximum stage position in micrometers (default is 100 um).
        '''
        super().__init__(
            name='PiezoConcept FOC 1-axis',
            max_range=(0, 0, max_um * 1000),
            units=Units.NANOMETERS,
            readyRead=self.read_serial_data,
        )

        self.position = self.z_max_nm // 2

    @property
    def position(self) -> int:
        '''Current stage position in nanometers.'''
        return self.get_position(Axis.Z)

    @position.setter
    def position(self, value: int):
        '''Set the current stage position in nanometers.'''
        self.set_position(Axis.Z, value)

    def is_open(self):
        return self.serial.isOpen()

    def connect(self):
        if not self.is_open():
            self.serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

    def disconnect(self):
        if self.is_open():
            self.serial.close()

    def send_command(self, value):
        '''
        Writes data to the serial port.

        Parameters
        ----------
        data : QByteArray | bytes | bytearray
            The data to be written to the serial port.

        Returns
        -------
        int
            The number of bytes written to the serial port, or a negative value
            if an error occurs.

        Raises
        ------
        IOError
            If the serial port is not open.
        '''
        return self.serial.write(value)

    def get_z(self):
        '''
        Requests the current position of the stage.
        This method sends a command to the stage to retrieve its current position.

        Returns
        -------
        None
        '''
        if self.is_open():
            self.send_command(b'GET_Z\n')
            self.last_command = 'GETZ'

    def move_absolute(self, pos: int):
        if self.is_open():
            self.position = min(max(pos, 0), self.z_max_nm)
            self.send_command(f'MOVEZ {pos}n\n'.encode())
            self.last_command = 'MOVEZ'

    def home(self):
        self.move_absolute(self.z_max_nm // 2)

    def stop(self):
        return super().stop()

    def move_relative(self, pos: int):
        if self.is_open():
            self.position = min(max(self.position + pos, 0), self.z_max_nm)
            self.send_command(f'MOVEZ {self.position}n\n'.encode())
            self.last_command = 'MOVEZ'

    def refresh_position(self):
        self.move_absolute(self.position)

    def read_serial_data(self):
        '''
        Reads data from the serial port and processes it.

        This method is called automatically when new data is available in the
        serial port input buffer. It reads the data, parses the response, and
        updates the appropriate attributes of the `PzFoc` object as needed.

        Raises
        ------
        IOError
            If the serial port is not open, or an error occurs while reading.
        '''
        self.received_data = str(self.serial.readAll(), encoding='utf8')
        if self.last_command != 'GETZ':
            self.get_z()
        else:
            match = re.search(r' *(\d+\.\d+).*um.*', self.received_data)
            if match:
                # self.ZPosition = int(float(match.group(1)) * 1000)
                self.signals.positionChanged.emit(float(match.group(1)))

    def getQWidget(self):
        '''Generates a ZStageView with stage controls.'''
        return ZStageView(stage=self)
