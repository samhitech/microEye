from PyQt5.QtCore import QIODevice, QThread, pyqtSignal
from PyQt5.QtSerialPort import *


class io_matchbox(QSerialPort):
    '''
    MatchBox Class | Inherits QSerialPort
    '''
    INFO = b'r i'
    '''Get MatchBox Info
    '''
    ON = b'e 1'
    '''Enable the combiner
    '''
    OFF = b'e 0'
    '''Disable the combiner
    '''
    READ = b'r r'
    SETTINGS = b'r s'
    ENABLE_1 = b'L1E'
    '''Enable 1st Laser Diode
    '''
    ENABLE_2 = b'L2E'
    '''Enable 2nd Laser Diode
    '''
    ENABLE_3 = b'L3E'
    '''Enable 3rd Laser Diode
    '''
    ENABLE_4 = b'L4E'
    '''Enable 4th Laser Diode
    '''
    DISABLE_1 = b'L1D'
    '''Disable 1st Laser Diode
    '''
    DISABLE_2 = b'L2D'
    '''Disable 2nd Laser Diode
    '''
    DISABLE_3 = b'L3D'
    '''Disable 3rd Laser Diode
    '''
    DISABLE_4 = b'L4D'
    '''Disable 4th Laser Diode
    '''
    MAX_CUR = b'Lm?'
    '''Get maximum current
    '''
    CUR_SET = b'Lc?'
    '''Get the current set value
    '''
    CUR_CUR = b'Lr'
    '''Get the current reading
    '''
    STATUS = b'Le'
    '''Get the laser diodes enabled(1)/disabled(0) states
    '''
    START = b'c u 2 35488'
    '''TBA
    '''

    DataReady = pyqtSignal(str, bytes)

    Current = [0, 0, 0, 0]
    Setting = [0, 0, 0, 0]

    def SendCommand(self, command):
        '''Sends a specific command to the device and waits for
        the response then emits the DataReady signal.
        The DataReady signal passes the response and command.

        Parameters
        ----------
        command : [bytes]
            command to be sent, please check the constants
            implemented in the MatchBox class.
        '''
        if(self.isOpen()):
            self.write(command)
            self.waitForBytesWritten(500)
            while self.bytesAvailable() < 5:
                self.waitForReadyRead(500)

            response = str(self.readLine(),
                           encoding='utf8').strip('\r\n').strip()
            print(response, command)

            self.DataReady.emit(response, command)

    def OpenCOM(self):
        '''Opens the serial port and initializes the combiner.
        '''
        if not self.isOpen():
            self.open(QIODevice.ReadWrite)
            self.flush()

            if(self.isOpen()):
                self.SendCommand(io_matchbox.ON)
                self.SendCommand(io_matchbox.START)
                self.SendCommand(io_matchbox.STATUS)
                self.SendCommand(io_matchbox.CUR_CUR)
                self.SendCommand(io_matchbox.CUR_SET)

    def CloseCOM(self):
        '''Closes the serial port.
        '''
        self.SendCommand(io_matchbox.OFF)
        self.waitForBytesWritten(500)

        self.close()
