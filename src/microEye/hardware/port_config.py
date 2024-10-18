from microEye.qt import Qt, QtSerialPort, QtWidgets


class port_config(QtWidgets.QDialog):
    '''A dialog for setting the serial port config | Inherits QDialog'''

    def __init__(self, parent=None, title='Serial Port Config.', **kwargs):
        super().__init__(parent)

        self.setWindowTitle(title)
        self.portname_comboBox = QtWidgets.QComboBox()  # port name combobox
        self.baudrate_comboBox = QtWidgets.QComboBox()  # baudrate combobox

        # adding available serial ports
        for info in QtSerialPort.QSerialPortInfo.availablePorts():
            self.portname_comboBox.addItem(info.portName())

        self.portname_comboBox.setCurrentText(
            kwargs.get(
                'portname', QtSerialPort.QSerialPortInfo.availablePorts()[0].portName()
            )
        )

        # adding default baudrates (default 115200)
        for baudrate in QtSerialPort.QSerialPortInfo.standardBaudRates():
            # if baudrate == 115200:
            self.baudrate_comboBox.addItem(str(baudrate), baudrate)

        self.baudrate_comboBox.setCurrentText(kwargs.get('baudrate', '115200'))

        # dialog buttons
        buttonBox = QtWidgets.QDialogButtonBox()
        buttonBox.setOrientation(Qt.Orientation.Horizontal)
        buttonBox.setStandardButtons(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        # dialog layout
        lay = QtWidgets.QFormLayout(self)
        lay.addRow('Port Name:', self.portname_comboBox)
        lay.addRow('BaudRate:', self.baudrate_comboBox)
        lay.addRow(buttonBox)
        self.setFixedSize(self.sizeHint().width() + 50, self.sizeHint().height())
        self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowTitleHint)

    def get_results(self):
        '''Get the selected serial port config.

        Returns
        -------
        tuple (str, int)
            the selected port and baudrate respectively.
        '''
        return (
            self.portname_comboBox.currentText(),
            self.baudrate_comboBox.currentData(),
        )
