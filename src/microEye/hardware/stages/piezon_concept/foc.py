import re

from microEye.hardware.stages.stage import (
    AbstractStage,
    Axis,
    Units,
    emit_after_signal,
)
from microEye.qt import QtCore


class PiezoConceptFOC(AbstractStage):
    '''PiezoConcept FOC 1-axis stage adapter.'''

    NAME = 'PiezoConcept FOC 1-axis'

    def __init__(self, max_um: int = 100):
        '''
        Initializes a PiezoConcept FOC 1-axis stage adapter.

        Parameters
        ----------
        max_um : int, optional
            Maximum stage position in micrometers (default is 100 um).
        '''
        super().__init__(
            name=PiezoConceptFOC.NAME,
            max_range=(max_um * 1000,),
            units=Units.NANOMETERS,
            axes=(Axis.Z,),
            readyRead=self.read_serial_data,
        )

        self.z = self.z_max // 2

    def is_open(self):
        return self.serial.isOpen()

    def open(self):
        if not self.is_open():
            self.serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

    def close(self):
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

    @emit_after_signal('moveFinished')
    def move_absolute(self, x, y, z, **kwargs):
        if self.is_open():
            self.z = min(max(z, 0), self.z_max)
            self.send_command(f'MOVEZ {z}n\n'.encode())
            self.last_command = 'MOVEZ'

    def home(self):
        self.move_absolute(0, 0, self.z_max // 2)

    def stop(self):
        pass

    @emit_after_signal('moveFinished')
    def move_relative(self, x, y, z, **kwargs):
        if self.is_open():
            self.z = min(max(self.z + z, 0), self.z_max)
            self.send_command(f'MOVEZ {self.z}n\n'.encode())
            self.last_command = 'MOVEZ'

    def refresh_position(self):
        self.move_absolute(0, 0, self.z)

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
                self.signals.positionChanged.emit(self, float(match.group(1)), Axis.Z)
