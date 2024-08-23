import logging
import struct
from collections import deque
from datetime import datetime
from typing import Optional, Union

from microEye.qt import QDateTime, QtCore, QtSerialPort, Signal


class ELLException(Exception):
    pass


class DevicePort(QtCore.QObject):
    _instance = None
    '''Class variable to hold the single instance'''
    _responses: Optional[deque[str]] = None
    '''Class variable to hold the responses'''

    dataSent = Signal(str)
    '''Signal emitted when data is sent'''
    dataReceived = Signal(str)
    '''Signal emitted when data is received'''
    logUpdated = Signal(str)
    '''Signal to notify when the TX/RX log is updated'''
    errorOccurred = Signal(str)
    '''Signal for error reporting'''

    COMMAND_RESPONSE_LENGTHS = {
        'GS': 5,  # Example: GetStatus command
        'PO': 11,  # Example: Position command
        'IN': 33,  # Example: Identify command
        'BS': 5,  # Button status
        'BO': 11,  # Button position
        'I1': 25,
        'I2': 25,
        'I3': 25,
        'HO': 11,
        'GJ': 11,
        'GV': 5,
        'C1': 525,
        'C2': 525,
        # Add more commands as needed
    }
    '''A dictionary mapping commands to their expected response lengths'''

    @classmethod
    def instance(cls, port_name='COM1', baud_rate=9600, max_log_entries=1000):
        '''Class method to get the singleton instance'''
        if cls._instance is None:
            cls._instance = cls(port_name, baud_rate, max_log_entries)
        return cls._instance

    def __init__(self, port_name='COM1', baud_rate=9600, max_entries=1000):
        """
        Initialize the Elliptec Communication Layer.

        Parameters
        ----------
        port_name : str, optional
            The name of the serial port to use (default is 'COM1').
        baud_rate : int, optional
            The baud rate for the serial port (default is 9600).
        max_entries : int, optional
            The maximum number of entries to keep (default is 1000).
        """
        if DevicePort._instance is not None:
            raise Exception('This class is a singleton!')
        super().__init__()

        self.serial = QtSerialPort.QSerialPort(None, readyRead=self.rx_piezo)
        self.serial.setBaudRate(baud_rate)
        self.serial.setPortName(port_name)
        self.serial.setParity(QtSerialPort.QSerialPort.Parity.NoParity)
        self.serial.setStopBits(QtSerialPort.QSerialPort.StopBits.OneStop)
        self.serial.setDataBits(QtSerialPort.QSerialPort.DataBits.Data8)
        self.serial.setFlowControl(QtSerialPort.QSerialPort.FlowControl.NoFlowControl)
        self.buffer = ''

        # New attributes for logging
        DevicePort._responses = deque(maxlen=max_entries)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

    @classmethod
    def open(cls, portName: str = None):
        '''
        Open the serial port.
        '''
        if portName:
            cls.instance().serial.setPortName(portName)

        if not cls.instance().isOpen():
            cls.instance().serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

        return cls.instance().isOpen()

    @classmethod
    def close(cls):
        '''
        Close the serial port.
        '''
        if cls.instance().isOpen():
            cls.instance().serial.close()

    @classmethod
    def isOpen(cls):
        '''
        Check if the serial port is open.

        Returns
        -------
        bool
            True if the serial port is open, False otherwise.
        '''
        return cls.instance().serial.isOpen()

    @classmethod
    def setPortName(cls, name: str):
        '''
        Set the name of the serial port.

        Parameters
        ----------
        name : str
            The new port name.
        '''
        if not cls.instance().isOpen():
            cls.instance().serial.setPortName(name)

    @classmethod
    def setBaudRate(cls, baudRate: int):
        '''
        Set the baud rate for the serial port.

        Parameters
        ----------
        baudRate : int
            The new baud rate.
        '''
        if not cls.instance().isOpen():
            cls.instance().serial.setBaudRate(baudRate)

    @classmethod
    def log_communication(cls, direction, message):
        """
        Log a communication event.

        Parameters:
        direction (str): Either 'TX' for transmitted or 'RX' for received.
        message (str): The content of the message.
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        log_entry = f'{timestamp} - {direction}: {message}'
        cls.instance().logUpdated.emit(log_entry)

    @classmethod
    def write(cls, value: str):
        '''
        Write a value to the serial port.

        Parameters
        ----------
        value : bytes
            The value to write.
        '''
        if cls.instance().isOpen():
            cls.instance().serial.write(value.encode('utf-8'))
            cls.instance().dataSent.emit(value)
            cls.instance().log_communication('TX', value)

    @classmethod
    def addResponse(cls, response: str):
        '''
        Add a response to the response buffer.

        Parameters
        ----------
        response : str
            The response to add.
        '''
        cls._responses.append(response)

    @classmethod
    def hasResponse(cls, addresses: Optional[list[str]] = None):
        '''
        Check if there is a response in the buffer.

        Returns
        -------
        bool
            True if there is a response, False otherwise.
        '''
        if (
            addresses is not None
            and isinstance(addresses, list)
            and all(isinstance(address, str) for address in addresses)
        ):
            return any(response[0] in addresses for response in cls._responses)
        else:
            return len(cls._responses) > 0

    @classmethod
    def getNextResponse(cls, address: Optional[Union[str, list[str]]] = None):
        '''
        Get the next response from the buffer.

        Parameters
        ----------
        address : str or list[str], optional
            The address of the response to get
            (default is None, which gets the next response).

        Returns
        -------
        str
            The next response from the buffer.
        '''
        if address is not None:
            if isinstance(address, str):
                address = [address]

        if not cls.hasResponse(address):
            raise ELLException('No Responses available')

        if address is None:
            return cls._responses.popleft()
        elif isinstance(address, list) and all(isinstance(add, str) for add in address):
            for i, response in enumerate(cls._responses):
                if response[0] in address:
                    result = cls._responses[i]
                    del cls._responses[i]
                    return result

            raise ELLException('No Responses available for the given address')
        else:
            raise ELLException('Invalid address')

    @classmethod
    def clearResponses(cls, address: Optional[Union[str, list[str]]] = None):
        '''
        Clear the response buffer.

        Parameters
        ----------
        address : str or list[str], optional
            The address of the response to clear
            (default is None, which clears all responses).
        '''
        if address is not None:
            if isinstance(address, str):
                address = [address]

        if address is None:
            cls._responses.clear()
        elif isinstance(address, list) and all(isinstance(add, str) for add in address):
            cls._responses = deque(
                response for response in cls._responses if response[0] not in address
            )
        else:
            raise ELLException('Invalid address')

    def rx_piezo(self):
        '''
        Handle incoming data from the serial port.
        '''
        try:
            self.buffer += str(self.serial.readAll(), encoding='utf8')

            eol = self.buffer.find('\n')
            while eol >= 0:
                message = self.buffer[: eol + 1].strip(' \r\n')
                self.buffer = self.buffer[eol + 1 :]
                eol = self.buffer.find('\n')

                if message:
                    self.log_communication('RX', message)
                    DevicePort.addResponse(message)
                    self.dataReceived.emit(message)
        except Exception as e:
            self.logger.error(f'Error in rx_piezo: {str(e)}')
            self.errorOccurred.emit(str(e))

    @classmethod
    def wait_for_response(
        cls,
        addresses: Union[str, list[str]],
        responses: Union[str, list[str]],
        timeout: int,
    ):
        if isinstance(responses, str):
            responses = [responses]
        if isinstance(addresses, str):
            addresses = [addresses]

        headers = [
            f'{address}{response}' for address in addresses for response in responses
        ]

        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(timeout, loop.quit)
        cls.instance().dataReceived.connect(loop.quit)
        loop.exec()
        cls.instance().dataReceived.disconnect(loop.quit)

        if cls.hasResponse(addresses):
            message = cls.getNextResponse(addresses)
            if message[:3] in headers:
                return message

        raise ELLException('Response timeout')

    @classmethod
    def send_command(
        cls, address: str, command: str, param: str = '', clearResponses=True
    ) -> Optional[str]:
        '''
        Send a command to the stage.

        Parameters
        ----------
        address : str
            The stage address 0-F.
        command : str
            The command to send.
        param : str, optional
            The parameter to send (default is "").
        '''
        try:
            full_command = f'{address}{command}{param}'

            if clearResponses:
                cls.clearResponses(address)

            cls.instance().write(full_command)
        except Exception as e:
            cls.instance().logger.error(f'Error in send_command: {str(e)}')
            cls.instance().errorOccurred.emit(str(e))

    @classmethod
    def sendFreeString(cls, text: str):
        cls.instance().send_command(text, '')

    @classmethod
    def send_string_i16(cls, address: str, command: str, i: int):
        cls.instance().send_command(address, command, f'{i:04X}')

    @classmethod
    def send_string_i32(cls, address: str, command: str, i: int):
        cls.instance().send_command(address, command, f'{i:08X}')

    @classmethod
    def send_string_b(cls, address: str, command: str, b: int):
        cls.instance().send_command(address, command, f'{chr(b)}')
