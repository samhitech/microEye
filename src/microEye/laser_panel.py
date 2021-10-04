from .io_matchbox import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *


class laser_panel(QGroupBox):
    '''A laser control panel | Inherits QGroupBox
    '''

    def __init__(self, index: int, wavelength: int,
                 mbox: io_matchbox, maximum: int, *args, **kwargs):
        '''Initializes a new laser control panel for
        a specific laser diode in a laser combiner.

        Parameters
        ----------
        index : int
            The laser number.
        wavelength : int
            The wavelength of the laser.
        mbox : io_matchbox
            The laser combiner device adapter.
        maximum : int
            Maximum current.
        '''
        super(QGroupBox, self).__init__(*args, **kwargs)

        self.wavelength = wavelength
        self.index = index
        self.match_box = mbox

        # main vertical layout
        L_Layout = QVBoxLayout()
        self.setLayout(L_Layout)

        # on with cam 1 flash
        self.CAM1 = QRadioButton("CAM 1 Flash")
        self.CAM1.state = "L{:d}F1".format(wavelength)
        L_Layout.addWidget(self.CAM1)

        # on with cam 2 flash
        self.CAM2 = QRadioButton("CAM 2 Flash")
        self.CAM2.state = "L{:d}F2".format(wavelength)
        L_Layout.addWidget(self.CAM2)

        # off regardless
        self.OFF = QRadioButton("OFF")
        self.OFF.state = "L{:d}OFF".format(wavelength)
        self.OFF.setChecked(True)
        L_Layout.addWidget(self.OFF)

        # on regardless
        self.ON = QRadioButton("ON")
        self.ON.state = "L{:d}ON".format(wavelength)
        L_Layout.addWidget(self.ON)

        # Laser diode set current label + slider
        self.L_set_cur_label = QLabel("Set Current 0 mA")
        self.L_cur_slider = QSlider(Qt.Orientation.Horizontal)
        self.L_cur_slider.label = self.L_set_cur_label
        self.L_cur_slider.setMinimum(0)
        self.L_cur_slider.setMaximum(maximum)
        self.L_cur_slider.setValue(0)
        self.L_cur_slider.valueChanged.connect(self.current_valuechange)

        L_Layout.addWidget(self.L_set_cur_label)
        L_Layout.addWidget(self.L_cur_slider)

        # Laser diode current reading label
        self.L_cur_label = QLabel("Current 0 mA")

        L_Layout.addWidget(self.L_cur_label)

        # Laser diode current set button
        self.L_set_curr_btn = QPushButton(
            "Set Current",
            clicked=lambda:
            self.match_box.SendCommand(
                ("Lc{:d} ".format(self.index) + str(self.L_cur_slider.value()))
                .encode("utf-8"))
        )

        L_Layout.addWidget(self.L_set_curr_btn)

        L_Layout.addStretch()

        # Create a button group for radio buttons
        self.L_button_group = QButtonGroup()
        self.L_button_group.addButton(self.CAM1, 1)
        self.L_button_group.addButton(self.CAM2, 2)
        self.L_button_group.addButton(self.OFF, 3)
        self.L_button_group.addButton(self.ON, 4)

        self.L_button_group.buttonPressed.connect(self.laser_radio_changed)

    def current_valuechange(self):
        '''Updates the set current label upon slider changes
        '''
        self.sender().label.setText(
            "Current " + str(self.sender().value()) + " mA")

    def laser_radio_changed(self, object):
        '''Sends enable/disable signals to the
        laser combiner according to selected setting

        Parameters
        ----------
        object : [QRadioButton]
            the radio button toggled
        '''
        if ("OFF" in object.state):
            if ("L405" in object.state):
                self.match_box.SendCommand(io_matchbox.DISABLE_4)
            elif ("L488" in object.state):
                self.match_box.SendCommand(io_matchbox.DISABLE_3)
            elif ("L520" in object.state):
                self.match_box.SendCommand(io_matchbox.DISABLE_2)
            elif ("L638" in object.state):
                self.match_box.SendCommand(io_matchbox.DISABLE_1)
        else:
            if ("L405" in object.state):
                self.match_box.SendCommand(io_matchbox.ENABLE_4)
            elif ("L488" in object.state):
                self.match_box.SendCommand(io_matchbox.ENABLE_3)
            elif ("L520" in object.state):
                self.match_box.SendCommand(io_matchbox.ENABLE_2)
            elif ("L638" in object.state):
                self.match_box.SendCommand(io_matchbox.ENABLE_1)
