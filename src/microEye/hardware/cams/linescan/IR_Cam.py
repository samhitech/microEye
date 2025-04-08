import struct
import weakref
from os import name
from queue import LifoQueue, Queue
from typing import Union

import numpy as np

from microEye.hardware.port_config import *
from microEye.qt import QtCore, QtSerialPort, QtWidgets


class IR_Cam:
    '''An abstract class for IR cameras.'''

    def __init__(self, buffer: Union[Queue, LifoQueue]) -> None:
        self.name = 'Dummy'
        if buffer is None:
            self._buffer = LifoQueue()
        else:
            # add buffer in weakref to avoid circular reference
            self._buffer = weakref.ref(buffer)
        self.buffer.put(np.array([0 for i in range(128)]))
        self._connect_btn = QtWidgets.QPushButton()

    def isDummy(self) -> bool:
        return True

    @property
    def buffer(self):
        if isinstance(self._buffer, weakref.ref):
            # dereference the weakref to get the actual buffer
            return self._buffer()

        return self._buffer

    @property
    def isEmpty(self) -> bool:
        return self.buffer.empty()

    def get(self) -> np.ndarray:
        return self.buffer.get()

    @property
    def isOpen(self) -> bool:
        return False


class ParallaxLineScanner(IR_Cam):
    '''A class for Parallax CCD Array (TSL1401)
    connected via the Arduino LineScanner.'''

    def __init__(self, buffer: Union[Queue, LifoQueue]) -> None:
        super().__init__(buffer=buffer)

        self.name = 'Parallax CCD Array (TSL1401) LineScanner'
        self.buffer.put(np.array([0 for i in range(128)]))
        self.serial = QtSerialPort.QSerialPort(None, readyRead=self.receive)
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM4')

    def isDummy(self) -> bool:
        return False

    def open(self):
        '''Opens the serial port.'''
        self.serial.open(QtCore.QIODevice.OpenModeFlag.ReadWrite)

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
        '''IR CCD array serial port data ready signal'''
        if self.serial.bytesAvailable() >= 260:
            barray = self.serial.read(260)
            temp = np.array(
                np.array(struct.unpack('h' * (len(barray) // 2), barray)) * 5.0 / 1023.0
            )
            # array realignment
            if (temp[0] != 0 or temp[-1] != 0) and self.serial.bytesAvailable() >= 2:
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
            if dialog.exec():
                portname, baudrate = dialog.get_results()
                self.setPortName(portname)
                self.setBaudRate(baudrate)

    def getQWidget(self, parent=None) -> QtWidgets.QGroupBox:
        '''Generates a QGroupBox with
        connect/disconnect/config buttons.'''
        group = QtWidgets.QGroupBox('Parallax CCD Array')
        layout = QtWidgets.QVBoxLayout()
        group.setLayout(layout)

        # IR CCD array arduino buttons
        self._connect_btn = QtWidgets.QPushButton(
            'Connect', parent, clicked=lambda: self.open()
        )
        disconnect_btn = QtWidgets.QPushButton(
            'Disconnect', clicked=lambda: self.close()
        )
        config_btn = QtWidgets.QPushButton(
            'Port Config.', clicked=lambda: self.open_dialog()
        )

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._connect_btn)
        btns.addWidget(disconnect_btn)
        btns.addWidget(config_btn)
        layout.addLayout(btns)

        return group


class DemoLineScanner(IR_Cam):
    '''A demo camera class that generates random frames every 50 ms.
    Useful for testing and demonstrations without hardware.'''

    def __init__(self, buffer: Union[Queue, LifoQueue], array_length=128) -> None:
        super().__init__(buffer=buffer)

        self.name = 'Demo Line Scanner'
        self._array_length = array_length  # Adjustable array length
        self.buffer.put(np.array([0 for i in range(self._array_length)]))
        self._isOpen = False
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.generate_frame)
        self._timer.setInterval(50)  # 50 ms interval
        self._noise_level = 0.1  # Default noise level

        # Signal pattern options
        self.PATTERN_NOISE = 0
        self.PATTERN_STATIC_PEAK = 1
        self.PATTERN_MOVING_PEAK = 2
        self.PATTERN_DRIFT = 3
        self._pattern_type = self.PATTERN_NOISE

        # Parameters for peaks
        self._peak_width = self._array_length // 12  # Width of Gaussian peak (sigma)
        self._peak_amplitude = 0.8  # Height of peak (0-1)
        self._peak_position = (
            self._array_length // 2
        )  # Current position for moving peak
        self._peak_direction = 1  # Direction of movement (1 or -1)
        self._peak_speed = 2  # Pixels per frame

        # Microscope drift simulation parameters
        self._drift_speed = 0.1  # Base drift speed (pixels per frame)
        self._drift_variation = 0.3  # Random variation in drift
        self._drift_momentum = (
            0.95  # How much previous movement affects current (inertia)
        )
        self._current_drift_velocity = 0.0  # Current drift velocity
        self._vibration_amplitude = 0.3  # Amplitude of high-frequency vibrations
        self._drift_time = 0  # Time counter for drift simulation
        self._random_walk_factor = 0.1  # Factor for random walk component

        # Initialize fake drift with some variation
        self._drift_center = self._array_length // 2
        self._drift_period = np.random.uniform(
            50, 120
        )  # Period in frames for slow drift cycle
        self._drift_phase = np.random.uniform(0, 2 * np.pi)  # Random starting phase

    def isDummy(self) -> bool:
        return True

    def open(self):
        '''Starts the frame generation timer.'''
        self._isOpen = True
        self._timer.start()
        if self._connect_btn is not None:
            self._connect_btn.setEnabled(False)

    @property
    def isOpen(self) -> bool:
        '''Returns True if the demo camera is running.'''
        return self._isOpen

    def close(self):
        '''Stops the frame generation timer.'''
        self._timer.stop()
        self._isOpen = False
        if self._connect_btn is not None:
            self._connect_btn.setEnabled(True)

    @property
    def array_length(self) -> int:
        '''Get the current array length'''
        return self._array_length

    def set_array_length(self, length: int) -> None:
        '''Set a new array length'''

        self._array_length = length if length > 32 else 32

        # Adjust peak position and width to match new array length
        self._peak_position = self._array_length // 2
        self._peak_width = self._array_length // 12

    def generate_frame(self):
        '''Generates a frame based on the selected pattern type.'''
        # Base array with zeros
        frame = np.zeros(self._array_length)
        x = np.arange(self._array_length)

        # Add noise to all patterns
        if self._noise_level > 0:
            noise = np.random.random(self._array_length) * self._noise_level
            frame += noise

        # Add pattern based on selection
        if self._pattern_type == self.PATTERN_NOISE:
            # Just noise, already added
            pass

        elif self._pattern_type == self.PATTERN_STATIC_PEAK:
            # Add static Gaussian peak at center
            center = self._array_length // 2
            frame += self._peak_amplitude * np.exp(
                -((x - center) ** 2) / (2 * self._peak_width**2)
            )

        elif self._pattern_type == self.PATTERN_MOVING_PEAK:
            # Add moving Gaussian peak
            frame += self._peak_amplitude * np.exp(
                -((x - self._peak_position) ** 2) / (2 * self._peak_width**2)
            )

            # Update peak position for next frame
            self._peak_position += self._peak_speed * self._peak_direction

            # Reverse direction when reaching edges (with margin)
            edge_margin = max(8, self._array_length // 16)
            if (
                self._peak_position >= self._array_length - edge_margin
                or self._peak_position <= edge_margin
            ):
                self._peak_direction *= -1

        elif self._pattern_type == self.PATTERN_DRIFT:
            # Simulate microscope stage drift
            self._drift_time += 1

            # Combine multiple drift components:

            # 1. Slow sinusoidal drift (thermal/mechanical creep)
            slow_drift = np.sin(
                2 * np.pi * self._drift_time / self._drift_period + self._drift_phase
            )

            # 2. Brownian motion / random walk component
            # (use momentum to avoid jerky movement)
            random_component = np.random.normal(0, self._random_walk_factor)
            self._current_drift_velocity = (
                self._drift_momentum * self._current_drift_velocity
                + (1 - self._drift_momentum) * random_component
            )

            # 3. Small high-frequency vibrations (building vibrations, etc)
            vibration = (
                self._vibration_amplitude
                * np.sin(0.1 * self._drift_time)
                * np.random.random()
            )

            # Combine all drift components
            drift_amount = (
                slow_drift * self._drift_speed
                + self._current_drift_velocity
                + vibration
            )

            # Update drift center with bounds checking
            self._drift_center += drift_amount
            edge_margin = max(10, self._array_length // 10)
            if self._drift_center < edge_margin:
                self._drift_center = edge_margin
                self._current_drift_velocity *= -0.5  # Bounce back with damping
            elif self._drift_center > self._array_length - edge_margin:
                self._drift_center = self._array_length - edge_margin
                self._current_drift_velocity *= -0.5  # Bounce back with damping

            # Add Gaussian peak at current drift position
            frame += self._peak_amplitude * np.exp(
                -((x - self._drift_center) ** 2) / (2 * self._peak_width**2)
            )

        # Scale to 0-5V range
        frame = frame * 5.0

        # Add to buffer
        self.buffer.put(frame)

    def set_pattern_type(self, pattern_type):
        '''Set the pattern type for the generated frames'''
        self._pattern_type = pattern_type

    def set_peak_width(self, width):
        '''Set the width (sigma) of the Gaussian peak'''
        self._peak_width = max(1, min(self._array_length // 4, width))

    def set_peak_amplitude(self, amplitude):
        '''Set the amplitude of the Gaussian peak'''
        self._peak_amplitude = max(0.0, min(1.0, amplitude))

    def set_peak_speed(self, speed):
        '''Set the speed of the moving peak'''
        self._peak_speed = max(1, min(10, speed))

    def set_noise_level(self, level):
        '''Set the noise level (0.0 to 1.0)'''
        self._noise_level = max(0.0, min(1.0, level))  # Clamp between 0 and 1

    def set_drift_speed(self, speed):
        '''Set the base speed of the drift simulation'''
        self._drift_speed = max(0.01, min(1.0, speed))

    def set_drift_variation(self, variation):
        '''Set the random variation amount in drift simulation'''
        self._random_walk_factor = max(0.01, min(1.0, variation))

    def set_vibration_amplitude(self, amplitude):
        '''Set the amplitude of high-frequency vibrations'''
        self._vibration_amplitude = max(0.0, min(1.0, amplitude))

    def getQWidget(self, parent=None) -> QtWidgets.QGroupBox:
        '''Generates a QGroupBox with controls for the demo camera.'''
        group = QtWidgets.QGroupBox('Demo Line Scanner')
        layout = QtWidgets.QVBoxLayout()
        group.setLayout(layout)

        # Connect/Disconnect buttons
        self._connect_btn = QtWidgets.QPushButton(
            'Start Demo', parent, clicked=lambda: self.open()
        )
        disconnect_btn = QtWidgets.QPushButton(
            'Stop Demo', clicked=lambda: self.close()
        )

        # Array length control
        array_length_layout = QtWidgets.QHBoxLayout()
        array_length_label = QtWidgets.QLabel('Array Length:')
        array_length_spinbox = QtWidgets.QSpinBox()
        array_length_spinbox.setRange(16, 2048)  # Reasonable range
        array_length_spinbox.setValue(self._array_length)
        array_length_spinbox.valueChanged.connect(self.set_array_length)
        array_length_layout.addWidget(array_length_label)
        array_length_layout.addWidget(array_length_spinbox)

        # Pattern selection
        pattern_layout = QtWidgets.QHBoxLayout()
        pattern_label = QtWidgets.QLabel('Pattern:')
        pattern_combo = QtWidgets.QComboBox()
        pattern_combo.addItems(
            ['Noise Only', 'Static Peak', 'Moving Peak', 'Microscope Drift']
        )
        pattern_combo.currentIndexChanged.connect(self.set_pattern_type)
        pattern_layout.addWidget(pattern_label)
        pattern_layout.addWidget(pattern_combo)

        # Noise level slider
        noise_layout = QtWidgets.QHBoxLayout()
        noise_label = QtWidgets.QLabel('Noise Level:')
        noise_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        noise_slider.setMinimum(0)
        noise_slider.setMaximum(100)
        noise_slider.setValue(int(self._noise_level * 100))
        noise_slider.valueChanged.connect(lambda v: self.set_noise_level(v / 100.0))
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(noise_slider)

        # Peak width slider
        width_layout = QtWidgets.QHBoxLayout()
        width_label = QtWidgets.QLabel('Peak Width:')
        width_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        width_slider.setMinimum(1)
        width_slider.setMaximum(self._array_length // 4)
        width_slider.setValue(self._peak_width)
        width_slider.valueChanged.connect(self.set_peak_width)
        width_layout.addWidget(width_label)
        width_layout.addWidget(width_slider)

        # Peak amplitude slider
        amp_layout = QtWidgets.QHBoxLayout()
        amp_label = QtWidgets.QLabel('Peak Amplitude:')
        amp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        amp_slider.setMinimum(0)
        amp_slider.setMaximum(100)
        amp_slider.setValue(int(self._peak_amplitude * 100))
        amp_slider.valueChanged.connect(lambda v: self.set_peak_amplitude(v / 100.0))
        amp_layout.addWidget(amp_label)
        amp_layout.addWidget(amp_slider)

        # Drift speed slider (for microscope drift simulation)
        drift_speed_layout = QtWidgets.QHBoxLayout()
        drift_speed_label = QtWidgets.QLabel('Drift Speed:')
        drift_speed_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        drift_speed_slider.setMinimum(1)
        drift_speed_slider.setMaximum(100)
        drift_speed_slider.setValue(int(self._drift_speed * 100))
        drift_speed_slider.valueChanged.connect(
            lambda v: self.set_drift_speed(v / 100.0)
        )
        drift_speed_layout.addWidget(drift_speed_label)
        drift_speed_layout.addWidget(drift_speed_slider)

        # Drift variation slider
        drift_var_layout = QtWidgets.QHBoxLayout()
        drift_var_label = QtWidgets.QLabel('Drift Randomness:')
        drift_var_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        drift_var_slider.setMinimum(1)
        drift_var_slider.setMaximum(100)
        drift_var_slider.setValue(int(self._random_walk_factor * 100))
        drift_var_slider.valueChanged.connect(
            lambda v: self.set_drift_variation(v / 100.0)
        )
        drift_var_layout.addWidget(drift_var_label)
        drift_var_layout.addWidget(drift_var_slider)

        # Vibration amplitude slider
        vib_layout = QtWidgets.QHBoxLayout()
        vib_label = QtWidgets.QLabel('Vibration:')
        vib_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        vib_slider.setMinimum(0)
        vib_slider.setMaximum(100)
        vib_slider.setValue(int(self._vibration_amplitude * 100))
        vib_slider.valueChanged.connect(
            lambda v: self.set_vibration_amplitude(v / 100.0)
        )
        vib_layout.addWidget(vib_label)
        vib_layout.addWidget(vib_slider)

        # Arrange buttons
        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self._connect_btn)
        btns.addWidget(disconnect_btn)

        # Add all layouts
        layout.addLayout(btns)
        layout.addLayout(array_length_layout)
        layout.addLayout(pattern_layout)
        layout.addLayout(noise_layout)
        layout.addLayout(width_layout)
        layout.addLayout(amp_layout)
        layout.addLayout(drift_speed_layout)
        layout.addLayout(drift_var_layout)
        layout.addLayout(vib_layout)

        return group
