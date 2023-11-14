import struct
from os import name
from queue import Queue

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *

from ..port_config import *


class IR_Cam:
    '''An abstract class for IR cameras.'''

    def __init__(self) -> None:
        self.name = 'Dummy'
        self._buffer = Queue()
        self._buffer.put(np.array([0 for i in range(128)]))
        self._connect_btn = QPushButton()

    def isDummy(self) -> bool:
        return True

    @property
    def buffer(self):
        return self._buffer

    @property
    def isEmpty(self) -> bool:
        return self._buffer.empty()

    def get(self) -> np.ndarray:
        return self._buffer.get()

    @property
    def isOpen(self) -> bool:
        return False


class ParallaxLineScanner(IR_Cam):
    '''A class for Parallax CCD Array (TSL1401)
    connected via the Arduino LineScanner.'''

    def __init__(self) -> None:
        super().__init__()

        self.name = 'Parallax CCD Array (TSL1401) LineScanner'
        self._buffer.put(np.array([0 for i in range(128)]))
        self.serial = QSerialPort(
            None,
            readyRead=self.receive
        )
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM4')

    def isDummy(self) -> bool:
        return False

    def open(self):
        '''Opens the serial port.'''
        self.serial.open(QIODevice.ReadWrite)

    @property
    def isOpen(self) -> bool:
        '''Returns True if connected.'''
        return self.serial.isOpen()

    def close(self):
        '''Closes the supplied serial port.'''
        self.serial.close()

    def setPortName(self, name: str):
        '''Sets the serial port name.'''
        self.serial.setPortName(name)

    def setBaudRate(self, baudRate: int):
        '''Sets the serial port baudrate.'''
        self.serial.setBaudRate(baudRate)

    def receive(self):
        '''IR CCD array serial port data ready signal
        '''
        if self.serial.bytesAvailable() >= 260:
            barray = self.serial.read(260)
            temp = np.array(np.array(struct.unpack(
                'h'*(len(barray)//2), barray)) * 5.0 / 1023.0)
            # array realignment
            if (temp[0] != 0 or temp[-1] != 0) and \
               self.serial.bytesAvailable() >= 2:
                self.serial.read(2)
            # byte-wise realignment
            if temp.max() > 5:
                self.serial.read(1)
            self.buffer.put(temp[1:129])

    def open_dialog(self):
        '''Opens a port config dialog
        for the serial port.
        '''
        dialog = port_config()
        if not self.isOpen:
            if dialog.exec_():
                portname, baudrate = dialog.get_results()
                self.setPortName(portname)
                self.setBaudRate(baudrate)

    def getQWidget(self, parent=None) -> QGroupBox:
        '''Generates a QGroupBox with
        connect/disconnect/config buttons.'''
        group = QGroupBox('Parallax CCD Array')
        layout = QVBoxLayout()
        group.setLayout(layout)

        # IR CCD array arduino buttons
        self._connect_btn = QPushButton(
            'Connect',
            parent,
            clicked=lambda: self.open()
        )
        disconnect_btn = QPushButton(
            'Disconnect',
            clicked=lambda: self.close()
        )
        config_btn = QPushButton(
            'Port Config.',
            clicked=lambda: self.open_dialog()
        )

        btns = QHBoxLayout()
        btns.addWidget(self._connect_btn)
        btns.addWidget(disconnect_btn)
        btns.addWidget(config_btn)
        layout.addLayout(btns)

        return group
