from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *


class port_config(QDialog):
    '''A dialog for setting the serial port config | Inherits QDialog
    '''
    def __init__(self, parent=None):
        super(port_config, self).__init__(parent)

        self.setWindowTitle('Serial Port Config.')
        self.portname_comboBox = QComboBox()    # port name combobox
        self.baudrate_comboBox = QComboBox()    # baudrate combobox

        # adding available serial ports
        for info in QSerialPortInfo.availablePorts():
            self.portname_comboBox.addItem(info.portName())

        # adding default baudrates (default 115200)
        for baudrate in QSerialPortInfo.standardBaudRates():
            if baudrate == 115200:
                self.baudrate_comboBox.addItem(str(baudrate), baudrate)

        # dialog buttons
        buttonBox = QDialogButtonBox()
        buttonBox.setOrientation(Qt.Horizontal)
        buttonBox.setStandardButtons(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        # dialog layout
        lay = QFormLayout(self)
        lay.addRow("Port Name:", self.portname_comboBox)
        lay.addRow("BaudRate:", self.baudrate_comboBox)
        lay.addRow(buttonBox)
        self.setFixedSize(
            self.sizeHint().width() + 50, self.sizeHint().height())
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowTitleHint)

    def get_results(self):
        '''Get the selected serial port config.

        Returns
        -------
        tuple (str, int)
            the selected port and baudrate respectively.
        '''
        return self.portname_comboBox.currentText(), \
            self.baudrate_comboBox.currentData()
