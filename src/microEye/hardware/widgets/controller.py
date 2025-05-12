import json

import numpy as np
import pyqtgraph as pg

from microEye.hardware.stages.stabilizer import *
from microEye.qt import QtWidgets
from microEye.utils.gui_helper import GaussianOffSet


class Controller(QtWidgets.QDockWidget):
    stage_move_requested = Signal(str, int, float, bool)
    stage_stop_requested = Signal(str)
    stage_home_requested = Signal(str)
    stage_toggle_lock = Signal(str)

    def __init__(self):
        ''' '''
        super().__init__('Controller Unit')

        # Remove close button from dock widgets
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        self.init_layout()

        self.connect_signals()

    def init_layout(self):
        # display tab
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}

        main_layout = QtWidgets.QVBoxLayout()
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(main_layout)
        self.setWidget(main_widget)

        # Movement settings group
        settings_group = QtWidgets.QGroupBox('Movement Settings')
        settings_layout = QtWidgets.QGridLayout()

        # XY Step size controls
        self.xy_coarse_step = QtWidgets.QDoubleSpinBox()
        self.xy_coarse_step.setRange(0, 5000.0)  # 1µm to 5mm
        self.xy_coarse_step.setValue(500.0)
        self.xy_coarse_step.setSingleStep(100.0)
        self.xy_coarse_step.setSuffix(' µm')

        self.xy_fine_step = QtWidgets.QDoubleSpinBox()
        self.xy_fine_step.setRange(0.0, 100.0)  # 100nm to 10µm
        self.xy_fine_step.setValue(50.0)
        self.xy_fine_step.setSingleStep(5.0)
        self.xy_fine_step.setSuffix(' µm')
        self.xy_fine_step.setDecimals(3)

        # Z Step size controls
        self.z_coarse_step = QtWidgets.QDoubleSpinBox()
        self.z_coarse_step.setRange(0.0, 10.0)  # 100nm to 10µm
        self.z_coarse_step.setValue(1.0)
        self.z_coarse_step.setSingleStep(0.5)
        self.z_coarse_step.setSuffix(' µm')

        self.z_fine_step = QtWidgets.QSpinBox()
        self.z_fine_step.setRange(0, 1000)  # 1nm to 1µm
        self.z_fine_step.setValue(100)
        self.z_fine_step.setSingleStep(10)
        self.z_fine_step.setSuffix(' nm')

        # Layout for step controls
        settings_layout.addWidget(QtWidgets.QLabel('XY Fine:'), 0, 0)
        settings_layout.addWidget(self.xy_fine_step, 0, 1)
        settings_layout.addWidget(QtWidgets.QLabel('XY Coarse:'), 1, 0)
        settings_layout.addWidget(self.xy_coarse_step, 1, 1)
        settings_layout.addWidget(QtWidgets.QLabel('Z Fine:'), 2, 0)
        settings_layout.addWidget(self.z_fine_step, 2, 1)
        settings_layout.addWidget(QtWidgets.QLabel('Z Coarse:'), 3, 0)
        settings_layout.addWidget(self.z_coarse_step, 3, 1)

        settings_group.setLayout(settings_layout)

        # XY Controls group
        xy_group = QtWidgets.QGroupBox('XY Control')
        xy_layout = QtWidgets.QGridLayout()

        # XY Buttons
        self.btn_y_up = QtWidgets.QPushButton('++')
        self.btn_y_down = QtWidgets.QPushButton('--')
        self.btn_x_left = QtWidgets.QPushButton('--')
        self.btn_x_right = QtWidgets.QPushButton('++')

        # Fine movement buttons
        self.btn_y_up_fine = QtWidgets.QPushButton('+')
        self.btn_y_down_fine = QtWidgets.QPushButton('-')
        self.btn_x_left_fine = QtWidgets.QPushButton('-')
        self.btn_x_right_fine = QtWidgets.QPushButton('+')

        self.btn_xy_stop = QtWidgets.QPushButton('⚠')
        self.btn_xy_stop.setToolTip('Stop XY movement immediately!')

        # Set size policies and styling
        for btn in [
            self.btn_y_up,
            self.btn_y_down,
            self.btn_x_left,
            self.btn_x_right,
            self.btn_y_up_fine,
            self.btn_y_down_fine,
            self.btn_x_left_fine,
            self.btn_x_right_fine,
            self.btn_xy_stop,
        ]:
            btn.setFixedSize(50, 50)
            btn.setStyleSheet('''
                QPushButton {
                    font-weight: bold;
                    font-size: 22px;
                    padding: 0px;
                }
            ''')

        # Layout XY controls
        xy_layout.addWidget(self.btn_y_up_fine, 1, 2)
        xy_layout.addWidget(self.btn_y_up, 0, 2)
        xy_layout.addWidget(self.btn_y_down, 4, 2)
        xy_layout.addWidget(self.btn_y_down_fine, 3, 2)

        xy_layout.addWidget(self.btn_x_left_fine, 2, 1)
        xy_layout.addWidget(self.btn_x_left, 2, 0)
        xy_layout.addWidget(self.btn_x_right, 2, 4)
        xy_layout.addWidget(self.btn_x_right_fine, 2, 3)

        xy_layout.addWidget(self.btn_xy_stop, 2, 2)

        xy_group.setLayout(xy_layout)

        # Z Controls group
        z_group = QtWidgets.QGroupBox('Z Control')
        z_layout = QtWidgets.QVBoxLayout()

        self.btn_z_up = QtWidgets.QPushButton('++')
        self.btn_z_down = QtWidgets.QPushButton('--')
        self.btn_z_up_fine = QtWidgets.QPushButton('+')
        self.btn_z_down_fine = QtWidgets.QPushButton('-')
        self.btn_z_home = QtWidgets.QPushButton('🏠︎')
        self.btn_z_home.setToolTip('Move to home position')
        self.btn_z_toggle_stabilizer = QtWidgets.QPushButton('🔓')
        self.btn_z_toggle_stabilizer.setToolTip('Toggle Z-axis stabilizer')

        for btn in [
            self.btn_z_up,
            self.btn_z_down,
            self.btn_z_up_fine,
            self.btn_z_down_fine,
            self.btn_z_home,
            self.btn_z_toggle_stabilizer,
        ]:
            btn.setFixedSize(50, 50)
            btn.setStyleSheet('''
                QPushButton {
                    font-weight: bold;
                    font-size: 22px;
                    padding: 0px;
                }
            ''')

        z_layout.addWidget(self.btn_z_up)
        z_layout.addWidget(self.btn_z_up_fine)
        z_layout.addWidget(self.btn_z_down_fine)
        z_layout.addWidget(self.btn_z_down)
        z_layout.addWidget(self.btn_z_home)
        z_layout.addWidget(self.btn_z_toggle_stabilizer)
        z_group.setLayout(z_layout)

        # Create a container widget for XY and Z controls
        controls_layout = QtWidgets.QHBoxLayout()

        # Add XY and Z groups to horizontal layout
        controls_layout.addWidget(xy_group)
        controls_layout.addWidget(z_group)

        # Options group
        options_group = QtWidgets.QGroupBox('Options')
        options_layout = QtWidgets.QVBoxLayout()
        options_group.setLayout(options_layout)

        # snap_image after movement checkbox
        self.snap_image_after_movement = QtWidgets.QCheckBox(
            'Snap Image After Movement'
        )
        self.snap_image_after_movement.setChecked(False)
        options_layout.addWidget(self.snap_image_after_movement)

        # Add all groups to main layout
        main_layout.addWidget(settings_group)
        main_layout.addLayout(controls_layout)
        main_layout.addWidget(options_group)
        main_layout.addStretch()

    def connect_signals(self):
        # XY coarse movements
        self.btn_x_left.clicked.connect(lambda: self.move_stage('x', -1, 'coarse'))
        self.btn_x_right.clicked.connect(lambda: self.move_stage('x', 1, 'coarse'))
        self.btn_y_up.clicked.connect(lambda: self.move_stage('y', 1, 'coarse'))
        self.btn_y_down.clicked.connect(lambda: self.move_stage('y', -1, 'coarse'))

        # XY fine movements
        self.btn_x_left_fine.clicked.connect(lambda: self.move_stage('x', -1, 'fine'))
        self.btn_x_right_fine.clicked.connect(lambda: self.move_stage('x', 1, 'fine'))
        self.btn_y_up_fine.clicked.connect(lambda: self.move_stage('y', 1, 'fine'))
        self.btn_y_down_fine.clicked.connect(lambda: self.move_stage('y', -1, 'fine'))

        self.btn_xy_stop.clicked.connect(lambda: self.stage_stop_requested.emit('xy'))

        # Z movements
        self.btn_z_up.clicked.connect(lambda: self.move_stage('z', 1, 'coarse'))
        self.btn_z_down.clicked.connect(lambda: self.move_stage('z', -1, 'coarse'))
        self.btn_z_up_fine.clicked.connect(lambda: self.move_stage('z', 1, 'fine'))
        self.btn_z_down_fine.clicked.connect(lambda: self.move_stage('z', -1, 'fine'))

        self.btn_z_home.clicked.connect(lambda: self.stage_home_requested.emit('z'))
        self.btn_z_toggle_stabilizer.clicked.connect(self.toggle_stabilizer)

    def toggle_stabilizer(self):
        '''Toggle the stabilizer lock on the Z-axis. This method emits a signal to
        request the toggling of the stabilizer lock.

        Returns
        -------
        None
        '''
        self.stage_toggle_lock.emit('z')

    def set_stabilizer_lock(self, lock: bool):
        '''Set the stabilizer lock state on the Z-axis. This method updates the
        button icon and tooltip based on the lock state.

        Parameters
        ----------
        lock : bool
            True to lock the stabilizer, False to unlock it.

        Returns
        -------
        None
        '''
        self.btn_z_toggle_stabilizer.setText('🔒' if lock else '🔓')

    def move_stage(self, axis, direction, mode):
        if axis in ['x', 'y']:
            step = (
                self.xy_coarse_step.value()
                if mode == 'coarse'
                else self.xy_fine_step.value()
            )
            step /= 1000  # Convert to mm
            # print(f'Moving {axis} axis by {distance} µm')
        else:  # z-axis
            if mode == 'coarse':
                step = int(self.z_coarse_step.value() * 1000)
                # print(f'Moving {axis} axis by {distance} µm')
            else:
                step = self.z_fine_step.value()
                # print(f'Moving {axis} axis by {distance} nm')

        self.stage_move_requested.emit(
            axis, direction, step, self.snap_image_after_movement.isChecked()
        )

    def __str__(self):
        return 'Controller Unit Widget'


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    controller = Controller()
    controller.show()
    app.exec()
