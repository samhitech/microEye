import json
import os
import warnings
import webbrowser

import numpy as np
from scipy.optimize import OptimizeWarning

from microEye.hardware.cams import *
from microEye.hardware.lasers import *
from microEye.hardware.mieye.acquisition_manager import AcquisitionManager
from microEye.hardware.mieye.devices_manager import DeviceManager
from microEye.hardware.protocols import ExperimentDesigner, WeakObjects
from microEye.hardware.stages import FocusStabilizer, PzFocController
from microEye.hardware.widgets import DevicesView, devicesParams, focusWidget
from microEye.qt import (
    QT_API,
    QAction,
    QApplication,
    QDateTime,
    QMainWindow,
    Qt,
    QtCore,
    QtWidgets,
)
from microEye.utils.pyscripting import pyEditor
from microEye.utils.retry_exec import retry_exec
from microEye.utils.start_gui import StartGUI

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

        self.device_manager = DeviceManager(self)
        self.acquisition_manager = AcquisitionManager(self.device_manager)

        # setting title
        self.setWindowTitle('miEye module')

        # setting geometry
        self.setGeometry(0, 0, 1200, 920)

        # Statusbar time
        self.statusBar().showMessage(
            f'{QT_API} | Time: '
            + QtCore.QDateTime.currentDateTime().toString('hh:mm:ss,zzz')
        )

        # Threading
        print(
            'Multithreading with maximum %d threads'
            % QtCore.QThreadPool.globalInstance().maxThreadCount()
        )

        # IR Detector Widget
        self.IR_Widget = None

        # IR 2D Camera
        self.cam_dock = None

        # Layout
        self.LayoutInit()

        # Statues Bar Timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

        self.show()

        # centered
        self.center()

    def center(self):
        '''Centers the window within the screen.'''
        qtRectangle = self.frameGeometry()
        centerPoint = QApplication.primaryScreen().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def init_devices_dock(self):
        # General settings groupbox
        self.devicesDock = QtWidgets.QDockWidget('Devices', self)
        self.devicesDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        devicesWidget = QtWidgets.QWidget()
        self.devicesDock.setWidget(devicesWidget)
        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.devicesDock)

        # vertical layout
        devicesLayout = QtWidgets.QHBoxLayout()
        devicesWidget.setLayout(devicesLayout)

        devicesLayout.addWidget(self.device_manager.hid_controller)

        self.devicesView = DevicesView()
        self.devicesView.setDetectorActivated.connect(self.setIRcam)
        self.devicesView.resetDetectorActivated.connect(self.resetIRcam)
        self.devicesView.addLaserActivated.connect(self.add_laser_panel)
        self.devicesView.setStageActivated.connect(self.setStage)

        devicesLayout.addWidget(self.devicesView)

    def init_stages_dock(self):
        # Stages Tab (Elliptec + Kinesis Tab + Scan Acquisition)
        self.stagesDock = QtWidgets.QDockWidget('Stages', self)
        self.stagesDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.stages_Layout = QtWidgets.QHBoxLayout()
        self.stagesWidget = QtWidgets.QWidget()
        self.stagesWidget.setLayout(self.stages_Layout)
        self.stagesDock.setWidget(self.stagesWidget)

        self.setStage(self.devicesView.get_param_value(devicesParams.STAGE))

        self.stages_Layout.addWidget(
            DeviceManager.WIDGETS[DeviceManager.DEVICES.ELLIPTEC]
        )

        self.stages_Layout.addWidget(
            DeviceManager.WIDGETS[DeviceManager.DEVICES.XY_STAGE]
        )
        self.stages_Layout.addWidget(self.acquisition_manager.acquisitionWidget)

        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.stagesDock)

    def init_py_dock(self):
        # Py Script Editor
        self.pyDock = QtWidgets.QDockWidget('PyScript', self)
        self.pyDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.pyEditor = pyEditor()
        self.pyEditor.exec_btn.clicked.connect(lambda: self.scriptTest())
        self.pyDock.setWidget(self.pyEditor)

        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.pyDock)

    def init_protocol_dock(self):
        self.protocolDock = QtWidgets.QDockWidget('Protocols', self)
        self.protocolDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.actionEditor = ExperimentDesigner()
        self.protocolDock.setWidget(self.actionEditor)

        self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.protocolDock)

    def init_lasers_dock(self):
        # Lasers Tab
        self.lasersDock = QtWidgets.QDockWidget('Lasers', self)
        self.lasersDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.lasersLayout = QtWidgets.QHBoxLayout()
        # self.lasersLayout.addStretch()
        self.lasersWidget = QtWidgets.QWidget()
        self.lasersWidget.setLayout(self.lasersLayout)

        self.lasersDock.setWidget(self.lasersWidget)

        self.laserPanels = []

        self.lasersLayout.addWidget(self.device_manager.laserRelayCtrllr.view)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.lasersDock)

    def init_cam_dock(self):
        # cameras tab
        self.camDock = QtWidgets.QDockWidget('Cameras List', self)
        self.camDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        self.camList = CameraList()
        self.camList.cameraAdded.connect(self.add_camera)
        self.camList.cameraRemoved.connect(self.remove_camera)

        self.camDock.setWidget(self.camList)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.camDock)

    def init_focus_dock(self):
        # focusWidget
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.focus = focusWidget()
        WeakObjects.addObject(self.focus)

        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.focus)

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
        load_config.triggered.connect(lambda: loadConfig(self, False))
        auto_load_config = QAction('Load Config. && Connect', self)
        auto_load_config.triggered.connect(lambda: loadConfig(self, True))

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
            lambda: webbrowser.open('https://github.com/samhitech/microEye')
        )
        pypi = QAction('microEye PYPI', self)
        pypi.triggered.connect(
            lambda: webbrowser.open('https://pypi.org/project/microEye/')
        )

        # Add exit action to file menu
        file_menu.addAction(save_config)
        file_menu.addAction(load_config)
        file_menu.addAction(auto_load_config)
        view_menu.addAction(devices_act)
        view_menu.addAction(pyDock_act)
        view_menu.addAction(stages_act)
        view_menu.addAction(lasers_act)
        view_menu.addAction(cams_act)
        view_menu.addAction(focus_act)

        help_menu.addAction(github)
        help_menu.addAction(pypi)

    def tabifyDocks(self):
        self.tabifyDockWidget(self.pyDock, self.protocolDock)
        self.tabifyDockWidget(self.pyDock, self.devicesDock)
        self.tabifyDockWidget(self.pyDock, self.stagesDock)

        self.setTabPosition(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            QtWidgets.QTabWidget.TabPosition.North,
        )

        self.tabifyDockWidget(self.lasersDock, self.camDock)
        self.tabifyDockWidget(self.lasersDock, self.focus)

    def LayoutInit(self):
        '''Initializes the window layout'''

        self.init_devices_dock()
        # self.init_ir_dock()
        self.init_stages_dock()
        self.init_py_dock()
        self.init_protocol_dock()
        self.init_lasers_dock()
        self.init_cam_dock()
        self.init_focus_dock()
        self.tabifyDocks()

        self.init_menubar()

    def scriptTest(self):
        exec(self.pyEditor.toPlainText())

    def getDockWidget(self, text: str, content: QtWidgets.QWidget):
        dock = QtWidgets.QDockWidget(text, self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        dock.setWidget(content)
        return dock

    def add_camera(self, panel: Camera_Panel, ir: bool):
        if ir:
            panel._frames = FocusStabilizer.instance().buffer
            self.cam_dock = self.getDockWidget(panel._cam.name, panel)
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.cam_dock)
            self.tabifyDockWidget(self.lasersDock, self.cam_dock)
            self.focus.graph_IR.setLabel('left', 'Signal', '', **self.labelStyle)
        else:
            panel.setWindowTitle(panel.title())
            panel.show()

        WeakObjects.addObject(panel)

    def remove_camera(self, panel: Camera_Panel, ir: bool):
        if ir:
            self.removeDockWidget(self.cam_dock)
            self.cam_dock.deleteLater()
            self.cam_dock = None
        else:
            pass

        WeakObjects.removeObject(panel)

    def isEmpty(self):
        if self.camList.autofocusCam:
            return self.camList.autofocusCam.isEmpty
        elif not self.device_manager.IR_Cam.isDummy():
            return self.device_manager.IR_Cam.isEmpty
        else:
            return True

    def isImage(self):
        if self.camList.autofocusCam:
            return True
        elif not self.device_manager.IR_Cam.isDummy():
            return False
        else:
            return False

    def BufferGet(self):
        if self.camList.autofocusCam:
            return self.camList.autofocusCam.get(True)
        elif not self.device_manager.IR_Cam.isDummy():
            return self.device_manager.IR_Cam.buffer.get()
        else:
            return np.zeros((256, 256), dtype=np.uint16)

    def BufferSize(self):
        if self.camList.autofocusCam:
            return self.camList.autofocusCam.bufferSize
        elif not self.device_manager.IR_Cam.isDummy():
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
        return self.device_manager.laserRelayCtrllr.getCommand(config)

    def add_laser_panel(self, value: str):
        if 'IO MatchBox' in value:
            if 'Combiner' in value:
                combiner = CombinerLaserWidget()
                combiner.removed.connect(self.laser_panel_removed)
                self.laserPanels.append(combiner)
                self.lasersLayout.addWidget(combiner)
                WeakObjects.addObject(combiner)
            elif 'Single' in value:
                laser = SingleMatchBox()
                laser.removed.connect(self.laser_panel_removed)
                self.laserPanels.append(laser)
                self.lasersLayout.addWidget(laser)
                WeakObjects.addObject(laser)

    def laser_panel_removed(self, panel):
        WeakObjects.removeObject(panel)

    def update_gui(self):
        '''Recurring timer updates the status bar and GUI'''

        RelayBox = '    |  Relay ' + (
            'connected'
            if self.device_manager.laserRelayCtrllr.isOpen()
            else 'disconnected'
        )

        Position = ''
        Frames = '    | Frames Saved: ' + str(
            FocusStabilizer.instance().num_frames_saved
        )

        Worker = f'    | Execution time: {FocusStabilizer.instance()._exec_time:.0f}'
        if self.camList.autofocusCam:
            Worker += f'    | Frames Buffer: {self.BufferSize():d}'
        self.statusBar().showMessage(
            f'{QT_API} | '
            + 'Time: '
            + QDateTime.currentDateTime().toString('hh:mm:ss,zzz')
            + RelayBox
            + Position
            + Frames
            + Worker
        )

        # update indicators
        if self.device_manager.elliptec_controller.isOpen():
            self.device_manager.elliptec_controller._connect_btn.setStyleSheet(
                'background-color: #004CB6'
            )
        else:
            self.device_manager.elliptec_controller._connect_btn.setStyleSheet(
                'background-color: black'
            )

        self.device_manager.laserRelayCtrllr.updatePortState()
        if not self.device_manager.laserRelayCtrllr.isOpen():
            self.device_manager.laserRelayCtrllr.refreshPorts()
            self.device_manager.laserRelayCtrllr.updateHighlight(
                self.getRelaySettings()
            )
        else:
            self.device_manager.laserRelayCtrllr.updateHighlight(
                self.getRelaySettings()
            )

        for _, cam_list in CameraList.cameras.items():
            for cam in cam_list:
                cam['Panel'].updateInfo()

        if self.device_manager.stage:
            self.device_manager.stage.updatePortState()
            self.device_manager.stage.refreshPorts()

    def setIRcam(self, value: str):
        if self.camList.autofocusCam:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                f'Please remove {self.camList.autofocusCam.title()}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        if self.device_manager.IR_Cam.isOpen:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                f'Please disconnect {self.device_manager.IR_Cam.name}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        if 'TSL1401' in value:
            self.device_manager.IR_Cam = ParallaxLineScanner()
            if self.IR_Widget is not None:
                self.removeDockWidget(self.IR_Widget)
                self.IR_Widget.deleteLater()
            self.IR_Widget = QtWidgets.QDockWidget('IR Cam')
            self.IR_Widget.setFeatures(
                QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            )
            self.IR_Widget.setWidget(self.device_manager.IR_Cam.getQWidget())
            self.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, self.IR_Widget)
            self.tabifyDockWidget(self.devicesDock, self.IR_Widget)
            self.focus.graph_IR.setLabel('left', 'Signal', 'V', **self.labelStyle)

    def resetIRcam(self):
        if self.device_manager.IR_Cam.isOpen:
            QtWidgets.QMessageBox.warning(
                self,
                'Warning',
                f'Please disconnect {self.device_manager.IR_Cam.name}.',
                QtWidgets.QMessageBox.StandardButton.Ok,
            )
            return

        if self.IR_Widget is not None:
            self.removeDockWidget(self.IR_Widget)
            self.IR_Widget.deleteLater()
        self.IR_Widget = None
        self.device_manager.IR_Cam = IR_Cam()

    def setStage(self, value: str):
        if self.device_manager.stage and self.device_manager.stage.isOpen():
            return

        if self.device_manager.stage:
            WeakObjects.removeObject(self.device_manager.stage.view)
            self.device_manager.stage.view.remove_widget()

        if 'FOC100' in value:
            self.device_manager.stage = PzFocController()
            self.device_manager.stage.view.removed.connect(
                lambda: WeakObjects.addObject(self.device_manager.stage.view)
            )
            self.stages_Layout.insertWidget(0, self.device_manager.stage.view)
            WeakObjects.addObject(self.device_manager.stage.view)

    def StartGUI():
        '''Initializes a new QApplication and miEye_module.

        Use
        -------
        app, window = miEye_module.StartGUI()


        app.exec()

        Returns
        -------
        tuple (QApplication, microEye.miEye_module)
            Returns a tuple with QApp and miEye_module main window.
        '''
        return StartGUI(miEye_module)


def generateConfig(mieye: miEye_module):
    filename = 'config.json'
    config = {
        'LaserRelay': (
            mieye.device_manager.laserRelayCtrllr.portName(),
            mieye.device_manager.laserRelayCtrllr.baudRate(),
        ),
        'Elliptic': (
            mieye.device_manager.elliptec_controller.serial.portName(),
            mieye.device_manager.elliptec_controller.serial.baudRate(),
        ),
        'PiezoStage': (
            mieye.device_manager.stage.stage.serial.portName(),
            mieye.device_manager.stage.stage.serial.baudRate(),
        ),
        'KinesisX': (
            mieye.device_manager.kinesisXY.X_Kinesis.serial.port,
            mieye.device_manager.kinesisXY.X_Kinesis.serial.baudrate,
        ),
        'KinesisY': (
            mieye.device_manager.kinesisXY.Y_Kinesis.serial.port,
            mieye.device_manager.kinesisXY.Y_Kinesis.serial.baudrate,
        ),
        'FocusStabilizer': {
            'ROI_x': mieye.focus.roi.x(),
            'ROI_y': mieye.focus.roi.y(),
            'ROI_length': mieye.focus.roi.state['size'][1],
            'ROI_angle': mieye.focus.roi.state['angle'] % 360,
            'ROI_Width': FocusStabilizer.instance().line_width,
            'PID': FocusStabilizer.instance().getPID(),
            'PixelCalCoeff': FocusStabilizer.instance().pixelCalCoeff(),
            'UseCal': FocusStabilizer.instance().useCal(),
            'Inverted': FocusStabilizer.instance().isInverted(),
        },
    }

    config['miEye_module'] = (
        (
            mieye.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.geometry().width(), mieye.geometry().height()),
        mieye.isMaximized(),
    )

    config['LaserPanels'] = [
        (
            panel.Laser.portName(),
            panel.Laser.baudRate(),
            type(panel) is CombinerLaserWidget,
        )
        for panel in mieye.laserPanels
    ]

    config['LasersDock'] = (
        mieye.lasersDock.isFloating(),
        (
            mieye.lasersDock.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.lasersDock.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.lasersDock.geometry().width(), mieye.lasersDock.geometry().height()),
        mieye.lasersDock.isVisible(),
    )
    config['devicesDock'] = (
        mieye.devicesDock.isFloating(),
        (
            mieye.devicesDock.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.devicesDock.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.devicesDock.geometry().width(), mieye.devicesDock.geometry().height()),
        mieye.devicesDock.isVisible(),
    )
    config['stagesDock'] = (
        mieye.stagesDock.isFloating(),
        (
            mieye.stagesDock.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.stagesDock.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.stagesDock.geometry().width(), mieye.stagesDock.geometry().height()),
        mieye.stagesDock.isVisible(),
    )
    config['focus'] = (
        mieye.focus.isFloating(),
        (
            mieye.focus.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.focus.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.focus.geometry().width(), mieye.focus.geometry().height()),
        mieye.focus.isVisible(),
    )
    config['camDock'] = (
        mieye.camDock.isFloating(),
        (
            mieye.camDock.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.camDock.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.camDock.geometry().width(), mieye.camDock.geometry().height()),
        mieye.camDock.isVisible(),
    )
    config['pyDock'] = (
        mieye.pyDock.isFloating(),
        (
            mieye.pyDock.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            mieye.pyDock.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        ),
        (mieye.pyDock.geometry().width(), mieye.pyDock.geometry().height()),
        mieye.pyDock.isVisible(),
    )

    with open(filename, 'w') as file:
        json.dump(config, file, indent=2)

    print('Config.json file generated!')


def loadConfig(mieye: miEye_module, auto_connect=True):
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
                config['miEye_module'][1][1],
            )

    if 'FocusStabilizer' in config:
        fStabilizer = config['FocusStabilizer']
        if isinstance(fStabilizer, dict):
            mieye.focus.updateRoiParams(fStabilizer)
            mieye.focus.set_roi()

    if 'LaserRelay' in config:
        mieye.device_manager.laserRelayCtrllr.setPortName(str(config['LaserRelay'][0]))
        mieye.device_manager.laserRelayCtrllr.setBaudRate(int(config['LaserRelay'][1]))
    if 'Elliptic' in config:
        mieye.device_manager.elliptec_controller.setPortName(config['Elliptic'][0])
        mieye.device_manager.elliptec_controller.setBaudRate(int(config['Elliptic'][1]))
    if 'PiezoStage' in config:
        mieye.device_manager.stage.setPortName(str(config['PiezoStage'][0]))
        mieye.device_manager.stage.setBaudRate(int(config['PiezoStage'][1]))
    if 'KinesisX' in config:
        mieye.device_manager.kinesisXY.X_Kinesis.serial.port = str(
            config['KinesisX'][0]
        )
        mieye.device_manager.kinesisXY.X_Kinesis.serial.baudrate = int(
            config['KinesisX'][1]
        )
    if 'KinesisY' in config:
        mieye.device_manager.kinesisXY.Y_Kinesis.serial.port = str(
            config['KinesisY'][0]
        )
        mieye.device_manager.kinesisXY.Y_Kinesis.serial.baudrate = int(
            config['KinesisY'][1]
        )

    funcs = [
        mieye.device_manager.laserRelayCtrllr.connect,
        mieye.device_manager.elliptec_controller.open,
        mieye.device_manager.stage.connect,
        mieye.device_manager.kinesisXY.open,
    ]

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
                funcs.append(panel.laser_connect)

    if 'LasersDock' in config:
        mieye.lasersDock.setVisible(bool(config['LasersDock'][3]))
        if bool(config['LasersDock'][0]):
            mieye.lasersDock.setFloating(True)
            mieye.lasersDock.setGeometry(
                config['LasersDock'][1][0],
                config['LasersDock'][1][1],
                config['LasersDock'][2][0],
                config['LasersDock'][2][1],
            )
        else:
            mieye.lasersDock.setFloating(False)
    if 'devicesDock' in config:
        mieye.devicesDock.setVisible(bool(config['devicesDock'][3]))
        if bool(config['devicesDock'][0]):
            mieye.devicesDock.setFloating(True)
            mieye.devicesDock.setGeometry(
                config['devicesDock'][1][0],
                config['devicesDock'][1][1],
                config['devicesDock'][2][0],
                config['devicesDock'][2][1],
            )
        else:
            mieye.devicesDock.setFloating(False)
    if 'stagesDock' in config:
        mieye.stagesDock.setVisible(bool(config['stagesDock'][3]))
        if bool(config['stagesDock'][0]):
            mieye.stagesDock.setFloating(True)
            mieye.stagesDock.setGeometry(
                config['stagesDock'][1][0],
                config['stagesDock'][1][1],
                config['stagesDock'][2][0],
                config['stagesDock'][2][1],
            )
        else:
            mieye.stagesDock.setFloating(False)
    if 'focus' in config:
        mieye.focus.setVisible(bool(config['focus'][3]))
        if bool(config['focus'][0]):
            mieye.focus.setFloating(True)
            mieye.focus.setGeometry(
                config['focus'][1][0],
                config['focus'][1][1],
                config['focus'][2][0],
                config['focus'][2][1],
            )
        else:
            mieye.focus.setFloating(False)
    if 'camDock' in config:
        mieye.camDock.setVisible(bool(config['camDock'][3]))
        if bool(config['camDock'][0]):
            mieye.camDock.setFloating(True)
            mieye.camDock.setGeometry(
                config['camDock'][1][0],
                config['camDock'][1][1],
                config['camDock'][2][0],
                config['camDock'][2][1],
            )
        else:
            mieye.camDock.setFloating(False)
    if 'pyDock' in config:
        mieye.pyDock.setVisible(bool(config['pyDock'][3]))
        if bool(config['pyDock'][0]):
            mieye.pyDock.setFloating(True)
            mieye.pyDock.setGeometry(
                config['pyDock'][1][0],
                config['pyDock'][1][1],
                config['pyDock'][2][0],
                config['pyDock'][2][1],
            )
        else:
            mieye.pyDock.setFloating(False)

    if auto_connect:
        for func in funcs:
            retry_exec(func)

    print('Config.json file loaded!')


if __name__ == '__main__':
    try:
        import vimba as vb
    except Exception:
        vb = None

    if vb:
        with vb.Vimba.get_instance() as vimba:
            app, window = miEye_module.StartGUI()
            app.exec()
    else:
        app, window = miEye_module.StartGUI()
        app.exec()
