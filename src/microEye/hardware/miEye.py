import json
import os
import threading
import traceback
import warnings
import webbrowser

import numpy as np
import pyqtgraph as pg
import tabulate
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
from scipy.optimize import curve_fit
from scipy.optimize.optimize import OptimizeWarning
from scipy.signal import find_peaks

from ..shared.gui_helper import GaussianOffSet
from ..shared.hid_controller import *
from ..shared.pyscripting import *
from ..shared.start_gui import StartGUI
from ..shared.thread_worker import *
from .cams import *
from .lasers import *
from .port_config import *
from .stages import *
from .widgets import *

try:
    from pyueye import ueye

    from .cams import IDS_Camera
    from .cams.ueye_panel import IDS_Panel
except Exception:
    ueye = None
    IDS_Camera = None
    IDS_Panel = None

try:
    import vimba as vb

    from .cams import vimba_cam
    from .cams.vimba_panel import *
except Exception:
    vb = None

warnings.filterwarnings('ignore', category=OptimizeWarning)


class miEye_module(QMainWindow):
    '''The main GUI for miEye combines control and acquisition modules.

    Inherits `QMainWindow`

    Attributes:
        - devicesDock (`QDockWidget`):
            - devicesWidget (`QWidget`):
                - devicesLayout (`QHBoxLayout`):
                    - hid_controller (`hidController`)
                    - devicesView (`DevicesView`)

        - ir_widget (`QDockWidget`, optional):
            - `QWidget`

        - stagesDock (`QDockWidget`):
            - stagesWidget (`QWidget`):
                - stages_Layout (`QHBoxLayout`):
                    - z-stage (`FocPzView`)
                    - elliptec_controller (`elliptec_controller`)
                    - kinesisXY (`KinesisXY`)
                    - scanAcqWidget (`ScanAcquisitionWidget`)

        - pyDock (`QDockWidget`):
            - pyEditor (`pyEditor`)

        - lasersDock (`QDockWidget`):
            - lasersWidget (`QWidget`):
                - lasersLayout (`QHBoxLayout`):
                    - laserRelayCtrllr (`LaserRelayController`)
                    - lasers ...

        - camDock (`QDockWidget`):
            - camList (`CameraList`)

        - focus (`focusWidget`)
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setting title
        self.setWindowTitle(
            'microEye control module \
            (https://github.com/samhitech/microEye)')

        # setting geometry
        self.setGeometry(0, 0, 1200, 920)

        # Statusbar time
        self.statusBar().showMessage(
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz'))

        # Threading
        self._threadpool = QThreadPool.globalInstance()
        print('Multithreading with maximum %d threads'
              % self._threadpool.maxThreadCount())

        # Z-Stage e.g., PiezoConcept
        self.stage = None

        # XY-Stage e.g., kinesis
        self.kinesisXY = KinesisXY()

        self.lastTile = None
        self._stop_scan = False
        self._scanning = False

        # Serial Port IR linear CCD array
        self.IR_Cam = IR_Cam()
        # IR Detector Widget
        self.IR_Widget = None

        # IR 2D Camera
        self.cam_dock = None

        # Serial Port Laser Relay
        self.laserRelayCtrllr = LaserRelayController()
        self.laserRelayCtrllr.sendCommandActivated.connect(
            lambda: self.laserRelayCtrllr.sendCommand(
                self.getRelaySettings())
        )

        # Elliptec controller
        self._elliptec_controller = elliptec_controller()

        # Layout
        self.LayoutInit()

        # Statues Bar Timer
        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

        # Any other args, kwargs are passed to the run function
        FocusStabilizer.instance().moveStage.connect(self.moveStage)
        FocusStabilizer.instance().startWorker()

        self.show()

        # centered
        self.center()

    def center(self):
        '''Centers the window within the screen.
        '''
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def init_devices_dock(self):
        # General settings groupbox
        self.devicesDock = QDockWidget('Devices', self)
        self.devicesDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        devicesWidget = QWidget()
        self.devicesDock.setWidget(devicesWidget)
        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea,
            self.devicesDock)

        # vertical layout
        devicesLayout = QHBoxLayout()
        devicesWidget.setLayout(devicesLayout)

        self.hid_controller = hidController()
        self.hid_controller.reportEvent.connect(self.hid_report)
        self.hid_controller.reportRStickPosition.connect(
            self.hid_RStick_report)
        self.hid_controller.reportLStickPosition.connect(
            self.hid_LStick_report)
        self.hid_controller_toggle = False

        devicesLayout.addWidget(self.hid_controller)

        self.devicesView = DevicesView()
        self.devicesView.setDetectorActivated.connect(
            self.setIRcam)
        self.devicesView.resetDetectorActivated.connect(
            self.resetIRcam)
        self.devicesView.addLaserActivated.connect(
            self.add_laser_panel)
        self.devicesView.setStageActivated.connect(
            self.setStage)

        devicesLayout.addWidget(self.devicesView)

    def init_stages_dock(self):
        # Stages Tab (Elliptec + Kinesis Tab + Scan Acquisition)
        self.stagesDock = QDockWidget('Stages', self)
        self.stagesDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        self.stages_Layout = QHBoxLayout()
        self.stagesWidget = QWidget()
        self.stagesWidget.setLayout(self.stages_Layout)
        self.stagesDock.setWidget(self.stagesWidget)

        self.setStage(
            self.devicesView.get_param_value(devicesParams.STAGE))

        self.stages_Layout.addWidget(self._elliptec_controller.getQWidget())
        # Elliptec init config
        self._elliptec_controller.address_bx.setValue(2)
        self._elliptec_controller.stage_type.setCurrentText('ELL6')
        self._elliptec_controller._add_btn.click()
        self._elliptec_controller.address_bx.setValue(0)
        self._elliptec_controller.stage_type.setCurrentText('ELL6')
        self._elliptec_controller._add_btn.click()
        # Scan Acquisition
        self.scanAcqWidget = ScanAcquisitionWidget()
        self.scanAcqWidget.startAcquisitionXY.connect(
            self.start_scan_acquisitionXY)
        self.scanAcqWidget.startAcquisitionZ.connect(
            self.start_scan_acquisitionZ)
        self.scanAcqWidget.startCalibrationZ.connect(
            self.start_calibration_Z)
        self.scanAcqWidget.stopAcquisitionXY.connect(
            self.stop_scan_acquisition)
        self.scanAcqWidget.openLastTileXY.connect(self.show_last_tile)

        self.scanAcqWidget.directoryChanged.connect(self.update_directories)

        self.scanAcqWidget.moveZ.connect(self.moveStage)

        self.stages_Layout.addWidget(self.kinesisXY.getQWidget())
        self.stages_Layout.addWidget(self.scanAcqWidget)

        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea,
            self.stagesDock)

    def init_py_dock(self):
        # Py Script Editor
        self.pyDock = QDockWidget('PyScript', self)
        self.pyDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        self.pyEditor = pyEditor()
        self.pyEditor.exec_btn.clicked.connect(lambda: self.scriptTest())
        self.pyDock.setWidget(self.pyEditor)

        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea,
            self.pyDock)

    def init_lasers_dock(self):
        # Lasers Tab
        self.lasersDock = QDockWidget('Lasers', self)
        self.lasersDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        self.lasersLayout = QHBoxLayout()
        # self.lasersLayout.addStretch()
        self.lasersWidget = QWidget()
        self.lasersWidget.setLayout(self.lasersLayout)

        self.lasersDock.setWidget(self.lasersWidget)

        self.laserPanels = []

        self.lasersLayout.addWidget(self.laserRelayCtrllr.view)

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.lasersDock)

    def init_cam_dock(self):
        # cameras tab
        self.camDock = QDockWidget('Cameras List', self)
        self.camDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)

        self.camList = CameraList()
        self.camList.cameraAdded.connect(self.add_camera)
        self.camList.cameraRemoved.connect(self.remove_camera)

        self.camDock.setWidget(self.camList)

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.camDock)

    def init_focus_dock(self):
        # focusWidget
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.focus = focusWidget()

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea, self.focus)

    def init_menubar(self):
        # Create menu bar
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
        help_menu = menu_bar.addMenu('Help')

        # Create exit action
        save_config = QAction('Save Config.', self)
        save_config.triggered.connect(lambda: generateConfig(self))
        load_config = QAction('Load Config.', self)
        load_config.triggered.connect(lambda: loadConfig(self))

        devices_act = self.devicesDock.toggleViewAction()
        devices_act.setEnabled(True)
        pyDock_act = self.pyDock.toggleViewAction()
        pyDock_act.setEnabled(True)
        stages_act = self.stagesDock.toggleViewAction()
        stages_act.setEnabled(True)
        cams_act = self.camDock.toggleViewAction()
        cams_act.setEnabled(True)
        lasers_act = self.lasersDock.toggleViewAction()
        lasers_act.setEnabled(True)
        focus_act = self.focus.toggleViewAction()
        focus_act.setEnabled(True)

        github = QAction('microEye Github', self)
        github.triggered.connect(
            lambda: webbrowser.open('https://github.com/samhitech/microEye'))
        pypi = QAction('microEye PYPI', self)
        pypi.triggered.connect(
            lambda: webbrowser.open('https://pypi.org/project/microEye/'))

        # Add exit action to file menu
        file_menu.addAction(save_config)
        file_menu.addAction(load_config)
        view_menu.addAction(devices_act)
        view_menu.addAction(pyDock_act)
        view_menu.addAction(stages_act)
        view_menu.addAction(lasers_act)
        view_menu.addAction(cams_act)
        view_menu.addAction(focus_act)

        help_menu.addAction(github)
        help_menu.addAction(pypi)

    def tabifyDocks(self):
        self.tabifyDockWidget(self.pyDock, self.devicesDock)
        self.tabifyDockWidget(self.pyDock, self.stagesDock)

        self.setTabPosition(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            QTabWidget.TabPosition.North)

        self.tabifyDockWidget(self.lasersDock, self.camDock)
        self.tabifyDockWidget(self.lasersDock, self.focus)

    def LayoutInit(self):
        '''Initializes the window layout
        '''

        self.init_devices_dock()
        # self.init_ir_dock()
        self.init_stages_dock()
        self.init_py_dock()
        self.init_lasers_dock()
        self.init_cam_dock()
        self.init_focus_dock()
        self.tabifyDocks()

        self.init_menubar()

    def scriptTest(self):
        exec(self.pyEditor.toPlainText())

    def getDockWidget(self, text: str, content: QWidget):
        dock = QDockWidget(text, self)
        dock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        dock.setWidget(content)
        return dock

    def add_camera(self, panel: Camera_Panel, ir: bool):
        if ir:
            panel._frames = FocusStabilizer.instance().buffer
            self.cam_dock = self.getDockWidget(
                panel._cam.name, panel)
            self.addDockWidget(
                Qt.DockWidgetArea.BottomDockWidgetArea,
                self.cam_dock)
            self.tabifyDockWidget(
                self.lasersDock, self.cam_dock)
            self.focus.graph_IR.setLabel(
                'left', 'Signal', '', **self.labelStyle)
        else:
            panel.show()

    def remove_camera(self, cam: dict, ir: bool):
        if ir:
            self.removeDockWidget(self.cam_dock)
            self.cam_dock.deleteLater()
            self.cam_dock = None
        else:
            pass

    def isEmpty(self):
        if self.camList.autofocusCam:
            return self.camList.autofocusCam.isEmpty
        elif not self.IR_Cam.isDummy():
            return self.IR_Cam.isEmpty
        else:
            return True

    def isImage(self):
        if self.camList.autofocusCam:
            return True
        elif not self.IR_Cam.isDummy():
            return False
        else:
            return False

    def BufferGet(self):
        if self.camList.autofocusCam:
            return self.camList.autofocusCam.get(True)
        elif not self.IR_Cam.isDummy():
            return self.IR_Cam.buffer.get()
        else:
            return np.zeros((256, 256), dtype=np.uint16)

    def BufferSize(self):
        if self.camList.autofocusCam:
            return self.camList.autofocusCam.bufferSize
        elif not self.IR_Cam.isDummy():
            return 0
        else:
            return 0

    def getRelaySettings(self):
        '''Returns the RelayBox setting command.

        Returns
        -------
        str
            the RelayBox setting command.
        '''
        config = ''
        for panel in self.laserPanels:
            config += panel.GetRelayState()
        return self.laserRelayCtrllr.getCommand(config)

    def add_laser_panel(self, value: str):
        if 'IO MatchBox' in value:
            if 'Combiner' in value:
                combiner = CombinerLaserWidget()
                self.laserPanels.append(combiner)
                self.lasersLayout.addWidget(combiner)
            elif 'Single' in value:
                laser = SingleMatchBox()
                self.laserPanels.append(laser)
                self.lasersLayout.addWidget(laser)

    def update_gui(self):
        '''Recurring timer updates the status bar and GUI
        '''
        IR = ('    |  IR Cam ' +
              ('connected' if self.IR_Cam.isOpen else 'disconnected'))

        RelayBox = ('    |  Relay ' + ('connected' if self.laserRelayCtrllr.isOpen()
                    else 'disconnected'))

        Position = ''
        Frames = '    | Frames Saved: ' + str(
            FocusStabilizer.instance().num_frames_saved)

        Worker = f'    | Execution time: {FocusStabilizer.instance()._exec_time:.0f}'
        if self.camList.autofocusCam:
            Worker += f'    | Frames Buffer: {self.BufferSize():d}'
        self.statusBar().showMessage(
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz')
            + IR + RelayBox + Position + Frames + Worker)

        # update indicators
        if self.IR_Cam.isOpen:
            self.IR_Cam._connect_btn.setStyleSheet('background-color: #004CB6')
        else:
            self.IR_Cam._connect_btn.setStyleSheet('background-color: black')
        if self._elliptec_controller.isOpen():
            self._elliptec_controller._connect_btn.setStyleSheet(
                'background-color: #004CB6')
        else:
            self._elliptec_controller._connect_btn.setStyleSheet(
                'background-color: black')

        self.laserRelayCtrllr.updatePortState()
        if not self.laserRelayCtrllr.isOpen():
            self.laserRelayCtrllr.refreshPorts()
            self.laserRelayCtrllr.updateHighlight(self.getRelaySettings())
        else:
            self.laserRelayCtrllr.updateHighlight(self.getRelaySettings())

        for _, cam_list in CameraList.cameras.items():
            for cam in cam_list:
                cam['Panel'].updateInfo()

        if self.stage:
            self.stage.updatePortState()
            self.stage.refreshPorts()

    def setIRcam(self, value: str):
        if self.camList.autofocusCam:
            QMessageBox.warning(
                self,
                'Warning',
                f'Please remove {self.camList.autofocusCam.title()}.',
                QMessageBox.StandardButton.Ok)
            return

        if self.IR_Cam.isOpen:
            QMessageBox.warning(
                self,
                'Warning',
                f'Please disconnect {self.IR_Cam.name}.',
                QMessageBox.StandardButton.Ok)
            return

        if 'TSL1401' in value:
            self.IR_Cam = ParallaxLineScanner()
            if self.IR_Widget is not None:
                self.removeDockWidget(self.IR_Widget)
                self.IR_Widget.deleteLater()
            self.IR_Widget = QDockWidget('IR Cam')
            self.IR_Widget.setFeatures(
                QDockWidget.DockWidgetFloatable |
                QDockWidget.DockWidgetMovable)
            self.IR_Widget.setWidget(
                self.IR_Cam.getQWidget())
            self.addDockWidget(
                Qt.DockWidgetArea.TopDockWidgetArea,
                self.IR_Widget)
            self.tabifyDockWidget(
                self.devicesDock, self.IR_Widget)
            self.focus.graph_IR.setLabel(
                'left', 'Signal', 'V', **self.labelStyle)

    def resetIRcam(self):
        if self.IR_Cam.isOpen:
            QMessageBox.warning(
                self,
                'Warning',
                f'Please disconnect {self.IR_Cam.name}.',
                QMessageBox.StandardButton.Ok)
            return

        if self.IR_Widget is not None:
            self.removeDockWidget(self.IR_Widget)
            self.IR_Widget.deleteLater()
        self.IR_Widget = None
        self.IR_Cam = IR_Cam()

    def setStage(self, value: str):
        if self.stage and self.stage.isOpen():
            return

        if self.stage:
            self.stage.view.remove_widget()

        if 'FOC100' in value:
            self.stage = PzFocController()
            self.stages_Layout.insertWidget(0, self.stage.view)

    def moveStage(self, dir: bool, steps: int):
        if isinstance(self.stage, PzFocController):
            self.stage.moveStage(dir, steps)

    def hid_report(self, reportedEvent: Buttons):
        if isinstance(self.stage, PzFocController):
            if reportedEvent == Buttons.X:
                self.stage.moveStage(True, True, True)
            elif reportedEvent == Buttons.B:
                self.stage.moveStage(False, True, True)
            elif reportedEvent == Buttons.Y:
                self.stage.moveStage(True, False, True)
            elif reportedEvent == Buttons.A:
                self.stage.moveStage(False, False, True)
            elif reportedEvent == Buttons.Options:
                self.stage.stage.HOME()
            elif reportedEvent == Buttons.R3:
                self.hid_controller_toggle = not self.hid_controller_toggle

        if reportedEvent == Buttons.LEFT:
            if not self.hid_controller_toggle:
                self.kinesisXY.n_x_step_btn.click()
            else:
                self.kinesisXY.n_x_jump_btn.click()
        elif reportedEvent == Buttons.RIGHT:
            if not self.hid_controller_toggle:
                self.kinesisXY.p_x_step_btn.click()
            else:
                self.kinesisXY.p_x_jump_btn.click()
        elif reportedEvent == Buttons.UP:
            if not self.hid_controller_toggle:
                self.kinesisXY.p_y_step_btn.click()
            else:
                self.kinesisXY.p_y_jump_btn.click()
        elif reportedEvent == Buttons.DOWN:
            if not self.hid_controller_toggle:
                self.kinesisXY.n_y_step_btn.click()
            else:
                self.kinesisXY.n_y_jump_btn.click()
        elif reportedEvent == Buttons.R1:
            self.kinesisXY._center_btn.click()
        elif reportedEvent == Buttons.L1:
            self.kinesisXY._stop_btn.click()
        elif reportedEvent == Buttons.L3:
            self.hid_controller_toggle = not self.hid_controller_toggle
            if self.hid_controller_toggle:
                self.kinesisXY.n_x_jump_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.n_y_jump_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.p_x_jump_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.p_y_jump_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.n_x_step_btn.setStyleSheet(
                    '')
                self.kinesisXY.n_y_step_btn.setStyleSheet(
                    '')
                self.kinesisXY.p_x_step_btn.setStyleSheet(
                    '')
                self.kinesisXY.p_y_step_btn.setStyleSheet(
                    '')
            else:
                self.kinesisXY.n_x_jump_btn.setStyleSheet(
                    '')
                self.kinesisXY.n_y_jump_btn.setStyleSheet(
                    '')
                self.kinesisXY.p_x_jump_btn.setStyleSheet(
                    '')
                self.kinesisXY.p_y_jump_btn.setStyleSheet(
                    '')
                self.kinesisXY.n_x_step_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.n_y_step_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.p_x_step_btn.setStyleSheet(
                    'background-color: #004CB6')
                self.kinesisXY.p_y_step_btn.setStyleSheet(
                    'background-color: #004CB6')

    def hid_LStick_report(self, x, y):
        diff_x = x - 128
        diff_y = y - 127
        deadzone = 16
        res = dz_hybrid([diff_x, diff_y], deadzone)
        diff = res[1]
        if abs(diff) > 0:
            if self.hid_controller_toggle:
                val = 0.0001 * diff
                val += self.kinesisXY.jump_spin.value()
                self.kinesisXY.jump_spin.setValue(val)
            else:
                val = 0.0001 * diff
                val += self.kinesisXY.step_spin.value()
                self.kinesisXY.step_spin.setValue(val)
        else:
            if self.hid_controller_toggle:
                val = self.kinesisXY.jump_spin.value()
                val -= val % 0.0005
                self.kinesisXY.jump_spin.setValue(val)
            else:
                val = self.kinesisXY.step_spin.value()
                val -= val % 0.0005
                self.kinesisXY.step_spin.setValue(val)

    def hid_RStick_report(self, x, y):
        diff_x = x - 128
        diff_y = y - 127
        deadzone = 16
        res = dz_hybrid([diff_x, diff_y], deadzone)
        diff = res[1]
        if self.hid_controller_toggle:
            if FocusStabilizer.instance().isFocusStabilized():
                value = 1e-3 * diff
                FocusStabilizer.instance().setPeakPosition(value, True)
        else:
            if abs(diff) > 0 and abs(diff) < 100:
                value = 0.25 * diff
                self.stage.setStep(value, True)
            else:
                value = self.stage.getStep()
                value -= value % 5
                self.stage.setStep(value)

    def StartGUI():
        '''Initializes a new QApplication and miEye_module.

        Use
        -------
        app, window = miEye_module.StartGUI()


        app.exec_()

        Returns
        -------
        tuple (QApplication, microEye.miEye_module)
            Returns a tuple with QApp and miEye_module main window.
        '''
        return StartGUI(miEye_module)

    def result_scan_acquisition(self, data):
        self._scanning = False
        self.scanAcqWidget.setActionsStatus(True)

        if data:
            self.lastTile = TiledImageSelector(data)
            self.lastTile.positionSelected.connect(
                lambda x, y: self.kinesisXY.doAsync(
                    None, self.kinesisXY.move_absolute, x, y)
            )
            self.lastTile.show()

    def result_z_calibration(self, data):
        self._scanning = False
        self.scanAcqWidget.setActionsStatus(True)

        if data is not None:
            coeff = np.polyfit(data[:, 0], data[:, 1], 1)
            FocusStabilizer.instance().setPixelCalCoeff(coeff[0])
            plot_z_cal(data, coeff)

    def result_scan_export(self, data: list[TileImage]):
        self._scanning = False
        self.scanAcqWidget.setActionsStatus(True)

        if data:
            if len(self.scanAcqWidget._directory) > 0:
                path = self.scanAcqWidget._directory
                index = 0
                while (os.path.exists(path + f'/{index:03d}_XY/')):
                    index += 1
                path = path + f'/{index:03d}_XY/'
                if not os.path.exists(path):
                    os.makedirs(path)
                for idx, tImg in enumerate(data):
                    tf.imwrite(
                        path +
                        f'{idx:03d}_image_y{tImg.index[0]:02d}_x{tImg.index[1]:02d}.tif',
                        tImg.uImage.image,
                        photometric='minisblack')

    def start_scan_acquisitionXY(
            self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = thread_worker(
                scanAcquisition, self,
                [params[0], params[1]],
                [params[2], params[3]],
                params[4],
                params[5], progress=False, z_stage=False)
            self.scan_worker.signals.result.connect(
                self.result_scan_acquisition)
            # Execute
            self._threadpool.start(self.scan_worker)

            self.scanAcqWidget.setActionsStatus(False)

    def start_scan_acquisitionZ(
            self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = thread_worker(
                z_stack_acquisition, self,
                params[0], params[1],
                params[2], params[3],
                params[4], progress=False, z_stage=False)
            self.scan_worker.signals.result.connect(
                self.result_scan_acquisition)
            # Execute
            self._threadpool.start(self.scan_worker)

            self.scanAcqWidget.setActionsStatus(False)

    def start_calibration_Z(
            self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = thread_worker(
                z_calibration, self,
                params[0], params[1],
                params[2], params[3],
                params[4], progress=False, z_stage=False)
            self.scan_worker.signals.result.connect(
                self.result_z_calibration)
            # Execute
            self._threadpool.start(self.scan_worker)

            self.scanAcqWidget.setActionsStatus(False)

    def stop_scan_acquisition(self):
        self._stop_scan = True

    def update_directories(self, value: str):
        for _, cam_list in CameraList.cameras.items():
            for cam in cam_list:
                panel: Camera_Panel = cam['Panel']
                panel._directory = value
                panel.camera_options.set_param_value(
                    CamParams.SAVE_DIRECTORY, value)

    def show_last_tile(self):
        if self.lastTile is not None:
            self.lastTile.show()


def scanAcquisition(miEye: miEye_module, steps, step_size, delay, average=1):
    '''Scan Acquisition (works with Allied Vision Cams only)

    Parameters
    ----------
    miEye : miEye_module
        the miEye module
    steps : (int, int)
        number of grid steps (x ,y)
    step_size : (float, float)
        step size in um (x ,y)
    delay : float
        delay in ms after moving before acquiring images
    average : int
        number of frames to average, default 1 (no averaging)

    Returns
    -------
    list[TileImage]
        result data list of TileImages
    '''
    try:
        data = []
        vimba_cams = [cam for cam in CameraList.cameras['Vimba'] if not cam['IR']]
        if miEye.kinesisXY.isOpen()[0] and miEye.kinesisXY.isOpen()[1] \
                and len(vimba_cams) > 0:
            cam: vimba_cam = vimba_cams[0]['Camera']
            for x in range(steps[0]):
                miEye.kinesisXY.move_relative(
                    round(step_size[0] / 1000, 4), 0)
                for y in range(steps[1]):
                    if y > 0:
                        miEye.kinesisXY.move_relative(0, ((-1)**x) * round(
                            step_size[1] / 1000, 4))
                    frame = None
                    with cam.cam:
                        QThread.msleep(delay)
                        if average > 1:
                            frames_avg = []
                            for _n in range(average):
                                frames_avg.append(
                                    cam.cam.get_frame().as_numpy_ndarray()[..., 0])
                            frame = uImage(
                                np.array(
                                    frames_avg).mean(
                                        axis=0, dtype=np.uint16))
                        else:
                            frame = uImage(
                                cam.cam.get_frame().as_numpy_ndarray()[..., 0])
                        frame.equalizeLUT(None, True)
                    frame._view = cv2.resize(
                        frame._view, (0, 0),
                        fx=0.5,
                        fy=0.5,
                        interpolation=cv2.INTER_NEAREST)
                    Y = (x % 2) * (steps[1] - 1) + ((-1)**x) * y
                    data.append(
                        TileImage(frame, [Y, x], miEye.kinesisXY.position))
                    cv2.imshow(cam.name, frame._view)
                    cv2.waitKey(1)

            miEye.kinesisXY.update()
        else:
            return
    except Exception:
        traceback.print_exc()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            traceback.print_exc()

    return data

def z_stack_acquisition(
        miEye: miEye_module, n,
        step_size, delay=100, nFrames=1,
        reverse=False):
    '''Z-Stack Acquisition (works with Allied Vision Cams only)

    Parameters
    ----------
    miEye : miEye_module
        the miEye module
    n : int
        number of z-stacks
    step_size : int
        step size in nm along z-axis
    delay : float
        delay in ms after moving before acquiring images
    nFrames : int
        number of frames for each stack
    '''
    try:
        data = []
        peak = None
        vimba_cams = [cam for cam in CameraList.cameras['Vimba'] if not cam['IR']]
        if miEye.stage.isOpen() and len(vimba_cams) > 0 and nFrames > 0:
            cam: miCamera = vimba_cams[0]['Camera']
            cam_pan: Camera_Panel = vimba_cams[0]['Panel']
            if cam.acquisition:
                return

            cam_pan.camera_options.set_param_value(
                CamParams.FRAMES, nFrames)
            cam_pan.camera_options.set_param_value(
                CamParams.SAVE_DATA, True)

            peak = FocusStabilizer.instance().getPeakPosition()
            for x in range(n):
                if x > 0:
                    if FocusStabilizer.instance().isFocusStabilized() and \
                            FocusStabilizer.instance().useCal():
                        value = FocusStabilizer.instance().pixelCalCoeff() * step_size
                        if reverse:
                            value *= -1
                        FocusStabilizer.instance().setPeakPosition(value, True)
                        QThread.msleep(delay)
                    else:
                        if FocusStabilizer.instance().isFocusStabilized():
                            FocusStabilizer.instance().toggleFocusStabilization(False)
                        miEye.scanAcqWidget.moveZ.emit(reverse, step_size)
                        QThread.msleep(delay)
                        FocusStabilizer.instance().toggleFocusStabilization(True)
                frame = None
                prefix = f'Z_{x:04d}_'

                event = threading.Event()

                cam_pan.asyncFreerun.emit(prefix, event)

                event.wait()
                QThread.msleep(100)
        else:
            print('Z-scan failed!')
            info = [{
                'Z-Stage Open' : miEye.stage.isOpen(),
                'Camera Available' : len(vimba_cams) > 0,
                'Frames > 0' : nFrames > 0,
            }]
            print(tabulate.tabulate(info, headers='keys', tablefmt='rounded_grid'))
    except Exception:
        traceback.print_exc()
    finally:
        if peak:
            FocusStabilizer.instance().setPeakPosition(peak)
    return


def z_calibration(
        miEye: miEye_module, n,
        step_size, delay=100, nFrames=50,
        reverse=False):
    '''Z-Stack Acquisition (works with Allied Vision Cams only)

    Parameters
    ----------
    miEye : miEye_module
        the miEye module
    n : int
        number of z-stacks
    step_size : int
        step size in nm along z-axis
    delay : float
        delay in ms per measurement
    nFrames : int
        number of frames used for each measurement
    '''
    positions = np.zeros((n, 2))
    try:
        data = []
        if miEye.stage.isOpen():
            if FocusStabilizer.instance().isFocusStabilized():
                FocusStabilizer.instance().toggleFocusStabilization(False)
            for x in range(n):
                if x > 0:
                    miEye.scanAcqWidget.moveZ.emit(reverse, step_size)
                QThread.msleep(delay * nFrames)
                positions[x, 0] = x * step_size
                positions[x, 1] = np.mean(
                    FocusStabilizer.instance().peak_positions[-nFrames:])
    except Exception:
        traceback.print_exc()
        positions = None
    return positions


def plot_z_cal(data, coeff):

    model = np.poly1d(coeff)

    x = np.linspace(data[0, 0], data[-1, 0], 1001)

    # plot results
    plt = pg.plot()

    plt.showGrid(x=True, y=True)
    plt.addLegend()

    # set properties of the label for y axis
    plt.setLabel('left', 'Central Pixel', units='')

    # set properties of the label for x axis
    plt.setLabel('bottom', 'Z [nm]', units='')

    plt.setWindowTitle(
        f'Slope: {coeff[0]} | Intercept {coeff[1]}'
        )

    # setting horizontal range
    plt.setXRange(data[0, 0], data[-1, 0])

    # setting vertical range
    plt.setYRange(data[0, 1], data[-1, 1])

    line1 = plt.plot(
        data[:, 0], data[:, 1],
        pen='g', symbol='x', symbolPen='g',
        symbolBrush=0.2, name='Data')
    line2 = plt.plot(
        x, model(x),
        pen='b', name='Fit')


def generateConfig(mieye: miEye_module):
    filename = 'config.json'
    config = {
            'LaserRelay': (mieye.laserRelayCtrllr.portName(),
                           mieye.laserRelayCtrllr.baudRate()),
            'Elliptic': (mieye._elliptec_controller.serial.portName(),
                         mieye._elliptec_controller.serial.baudRate()),
            'PiezoStage': (mieye.stage.stage.serial.portName(),
                           mieye.stage.stage.serial.baudRate()),
            'KinesisX': (mieye.kinesisXY.X_Kinesis.serial.port,
                         mieye.kinesisXY.X_Kinesis.serial.baudrate),
            'KinesisY': (mieye.kinesisXY.Y_Kinesis.serial.port,
                         mieye.kinesisXY.Y_Kinesis.serial.baudrate),
            'FocusStabilizer' : {
                'ROI_x': mieye.focus.roi.x(),
                'ROI_y': mieye.focus.roi.y(),
                'ROI_length': mieye.focus.roi.state['size'][1],
                'ROI_angle': mieye.focus.roi.state['angle'] % 360,
                'ROI_Width': FocusStabilizer.instance().line_width,
                'PID': FocusStabilizer.instance().getPID(),
                'PixelCalCoeff': FocusStabilizer.instance().pixelCalCoeff(),
                'UseCal': FocusStabilizer.instance().useCal(),
                'Inverted': FocusStabilizer.instance().isInverted(),
            }
        }

    config['miEye_module'] = (
        (mieye.mapToGlobal(QPoint(0, 0)).x(),
         mieye.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.geometry().width(),
         mieye.geometry().height()),
        mieye.isMaximized())

    config['LaserPanels'] = [
        (panel.Laser.portName(),
         panel.Laser.baudRate(),
         type(panel) is CombinerLaserWidget)
        for panel in mieye.laserPanels]

    config['LasersDock'] = (
        mieye.lasersDock.isFloating(),
        (mieye.lasersDock.mapToGlobal(QPoint(0, 0)).x(),
         mieye.lasersDock.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.lasersDock.geometry().width(),
         mieye.lasersDock.geometry().height()),
        mieye.lasersDock.isVisible())
    config['devicesDock'] = (
        mieye.devicesDock.isFloating(),
        (mieye.devicesDock.mapToGlobal(QPoint(0, 0)).x(),
         mieye.devicesDock.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.devicesDock.geometry().width(),
         mieye.devicesDock.geometry().height()),
        mieye.devicesDock.isVisible())
    config['stagesDock'] = (
        mieye.stagesDock.isFloating(),
        (mieye.stagesDock.mapToGlobal(QPoint(0, 0)).x(),
         mieye.stagesDock.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.stagesDock.geometry().width(),
         mieye.stagesDock.geometry().height()),
        mieye.stagesDock.isVisible())
    config['focus'] = (
        mieye.focus.isFloating(),
        (mieye.focus.mapToGlobal(QPoint(0, 0)).x(),
         mieye.focus.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.focus.geometry().width(),
         mieye.focus.geometry().height()),
        mieye.focus.isVisible())
    config['camDock'] = (
        mieye.camDock.isFloating(),
        (mieye.camDock.mapToGlobal(QPoint(0, 0)).x(),
         mieye.camDock.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.camDock.geometry().width(),
         mieye.camDock.geometry().height()),
        mieye.camDock.isVisible())
    config['pyDock'] = (
        mieye.pyDock.isFloating(),
        (mieye.pyDock.mapToGlobal(QPoint(0, 0)).x(),
         mieye.pyDock.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.pyDock.geometry().width(),
         mieye.pyDock.geometry().height()),
        mieye.pyDock.isVisible())

    with open(filename, 'w') as file:
        json.dump(config, file, indent=2)

    print('Config.json file generated!')


def loadConfig(mieye: miEye_module):
    filename = 'config.json'

    if not os.path.exists(filename):
        print('Config.json not found!')
        return

    config: dict = None

    with open(filename) as file:
        config = json.load(file)

    if 'miEye_module' in config:
        if bool(config['miEye_module'][2]):
            mieye.showMaximized()
        else:
            mieye.setGeometry(
                config['miEye_module'][0][0],
                config['miEye_module'][0][1],
                config['miEye_module'][1][0],
                config['miEye_module'][1][1])

    if 'FocusStabilizer' in config:
        fStabilizer = config['FocusStabilizer']
        if isinstance(fStabilizer, dict):
            mieye.focus.updateRoiParams(fStabilizer)
            mieye.focus.set_roi()

    if 'LaserRelay' in config:
        mieye.laserRelayCtrllr.setPortName(str(config['LaserRelay'][0]))
        mieye.laserRelayCtrllr.setBaudRate(int(config['LaserRelay'][1]))
    if 'Elliptic' in config:
        mieye._elliptec_controller.serial.setPortName(
            config['Elliptic'][0])
        mieye._elliptec_controller.serial.setBaudRate(
            int(config['Elliptic'][1]))
    if 'PiezoStage' in config:
        mieye.stage.stage.serial.setPortName(str(config['PiezoStage'][0]))
        mieye.stage.stage.serial.setBaudRate(int(config['PiezoStage'][1]))
    if 'KinesisX' in config:
        mieye.kinesisXY.X_Kinesis.serial.port = str(config['KinesisX'][0])
        mieye.kinesisXY.X_Kinesis.serial.baudrate = int(config['KinesisX'][1])
    if 'KinesisY' in config:
        mieye.kinesisXY.Y_Kinesis.serial.port = str(config['KinesisY'][0])
        mieye.kinesisXY.Y_Kinesis.serial.baudrate = int(config['KinesisY'][1])

    if 'LaserPanels' in config:
        if config['LaserPanels'] is not None:
            for panel in mieye.laserPanels:
                panel.Laser.CloseCOM()
                panel.remove_widget()

            mieye.laserPanels.clear()

            for _panel in config['LaserPanels']:
                panel = CombinerLaserWidget() if bool(_panel[2]) else SingleMatchBox()
                mieye.laserPanels.append(panel)
                mieye.lasersLayout.addWidget(panel)
                panel.set_param_value(RelayParams.PORT, str(_panel[0]))
                panel.set_param_value(RelayParams.BAUDRATE, int(_panel[1]))
                panel.set_config()

    if 'LasersDock' in config:
        mieye.lasersDock.setVisible(
            bool(config['LasersDock'][3]))
        if bool(config['LasersDock'][0]):
            mieye.lasersDock.setFloating(True)
            mieye.lasersDock.setGeometry(
                config['LasersDock'][1][0],
                config['LasersDock'][1][1],
                config['LasersDock'][2][0],
                config['LasersDock'][2][1])
        else:
            mieye.lasersDock.setFloating(False)
    if 'devicesDock' in config:
        mieye.devicesDock.setVisible(
            bool(config['devicesDock'][3]))
        if bool(config['devicesDock'][0]):
            mieye.devicesDock.setFloating(True)
            mieye.devicesDock.setGeometry(
                config['devicesDock'][1][0],
                config['devicesDock'][1][1],
                config['devicesDock'][2][0],
                config['devicesDock'][2][1])
        else:
            mieye.devicesDock.setFloating(False)
    if 'stagesDock' in config:
        mieye.stagesDock.setVisible(
            bool(config['stagesDock'][3]))
        if bool(config['stagesDock'][0]):
            mieye.stagesDock.setFloating(True)
            mieye.stagesDock.setGeometry(
                config['stagesDock'][1][0],
                config['stagesDock'][1][1],
                config['stagesDock'][2][0],
                config['stagesDock'][2][1])
        else:
            mieye.stagesDock.setFloating(False)
    if 'focus' in config:
        mieye.focus.setVisible(
            bool(config['focus'][3]))
        if bool(config['focus'][0]):
            mieye.focus.setFloating(True)
            mieye.focus.setGeometry(
                config['focus'][1][0],
                config['focus'][1][1],
                config['focus'][2][0],
                config['focus'][2][1])
        else:
            mieye.focus.setFloating(False)
    if 'camDock' in config:
        mieye.camDock.setVisible(
            bool(config['camDock'][3]))
        if bool(config['camDock'][0]):
            mieye.camDock.setFloating(True)
            mieye.camDock.setGeometry(
                config['camDock'][1][0],
                config['camDock'][1][1],
                config['camDock'][2][0],
                config['camDock'][2][1])
        else:
            mieye.camDock.setFloating(False)
    if 'pyDock' in config:
        mieye.pyDock.setVisible(
            bool(config['pyDock'][3]))
        if bool(config['pyDock'][0]):
            mieye.pyDock.setFloating(True)
            mieye.pyDock.setGeometry(
                config['pyDock'][1][0],
                config['pyDock'][1][1],
                config['pyDock'][2][0],
                config['pyDock'][2][1])
        else:
            mieye.pyDock.setFloating(False)

    print('Config.json file loaded!')
