from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *


class port_config(QDialog):
    '''A dialog for setting the serial port config | Inherits QDialog
    '''
    def __init__(self, parent=None, title='Serial Port Config.'):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.portname_comboBox = QComboBox()    # port name combobox
        self.baudrate_comboBox = QComboBox()    # baudrate combobox

        # adding available serial ports
        for info in QSerialPortInfo.availablePorts():
            self.portname_comboBox.addItem(info.portName())

        # adding default baudrates (default 115200)
        for baudrate in QSerialPortInfo.standardBaudRates():
            # if baudrate == 115200:
            self.baudrate_comboBox.addItem(str(baudrate), baudrate)

        self.baudrate_comboBox.setCurrentText('115200')

        # dialog buttons
        buttonBox = QDialogButtonBox()
        buttonBox.setOrientation(Qt.Horizontal)
        buttonBox.setStandardButtons(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        # dialog layout
        lay = QFormLayout(self)
        lay.addRow('Port Name:', self.portname_comboBox)
        lay.addRow('BaudRate:', self.baudrate_comboBox)
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


class tracking_config(QDialog):
    '''A dialog for setting the stage tracking config | Inherits QDialog
    '''
    def __init__(self, *args, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Tracking Config.')
        valid = QDoubleValidator(0, 1e6, 3)
        self.pConst = QLineEdit(f'{args[0]:.3f}')
        self.pConst.setValidator(valid)
        self.iConst = QLineEdit(f'{args[1]:.3f}')
        self.iConst.setValidator(valid)
        self.dConst = QLineEdit(f'{args[2]:.3f}')
        self.dConst.setValidator(valid)
        self.tau = QLineEdit(f'{args[3]:.3f}')
        self.tau.setValidator(valid)
        self.thresh = QLineEdit(f'{args[4]:.3f}')
        self.thresh.setValidator(valid)

        # dialog buttons
        buttonBox = QDialogButtonBox()
        buttonBox.setOrientation(Qt.Horizontal)
        buttonBox.setStandardButtons(
            QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        # dialog layout
        lay = QFormLayout(self)
        lay.addRow('P:', self.pConst)
        lay.addRow('I:', self.iConst)
        lay.addRow('D:', self.dConst)
        lay.addRow('tau:', self.tau)
        lay.addRow('Error Threshold:', self.thresh)
        lay.addRow(buttonBox)
        self.setFixedSize(
            self.sizeHint().width() + 50, self.sizeHint().height())
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowTitleHint)

    def get_results(self):
        '''Get the results

        Returns
        -------
        tuple (float, float, float, float, float)
            the tracking config values list (pConst, iConst, dConst,
            tau, threshold)
        '''
        return (float(self.pConst.text()),
                float(self.iConst.text()),
                float(self.dConst.text()),
                float(self.tau.text()),
                float(self.thresh.text()))
