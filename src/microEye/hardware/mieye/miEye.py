import json
import os
import weakref
from enum import Enum

from microEye.hardware.mieye.acquisition_manager import AcquisitionManager
from microEye.hardware.mieye.devices_manager import (
    DEVICES,
    Camera_Panel,
    CameraList,
    DeviceManager,
    FocusStabilizer,
)
from microEye.hardware.protocols.designer import ExperimentDesigner
from microEye.hardware.pycromanager.widgets import BridgesWidget, HeadlessManagerWidget
from microEye.hardware.widgets import (
    Controller,
    DevicesView,
)
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
from microEye.tools.microscopy import ObjectiveCalculator
from microEye.utils.pyscripting import pyEditor
from microEye.utils.start_gui import StartGUI


class DOCKS(Enum):
    DEVICES = 0
    LASERS = 1
    STAGES = 2
    FOCUS = 3
    CAMERAS = 4
    PYSCRIPT = 5
    PROTOCOLS = 6
    CONTROLLER = 7
    FOCUS_CAMERA = 8

    def get_key(self):
        return self.name.lower() + '_dock'


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

        self.device_manager = DeviceManager()
        self.acquisition_manager = AcquisitionManager()

        self.device_manager.widgetAdded.connect(self._add_widgets)
        self.device_manager.widgetRemoved.connect(self._remove_widgets)

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
        max_threads = QtCore.QThreadPool.globalInstance().maxThreadCount()
        print(f'Multithreading with maximum {max_threads} threads')

        # IR 1D array
        self.ir_array_dock = None

        # IR 2D Camera
        self.cam_dock = None

        # Layout
        self.LayoutInit()

        self.device_manager.init_devices()

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

    def LayoutInit(self):
        '''Initializes the window layout'''

        self.dock_widgets: weakref.WeakValueDictionary[DOCKS, QtWidgets.QDockWidget] = (
            weakref.WeakValueDictionary()
        )

        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}

        self.init_controller()
        self.init_devices_dock()
        # self.init_ir_dock()
        self.init_stages_dock()
        self.init_py_dock()
        self.init_protocol_dock()
        self.init_lasers_dock()
        self.init_cam_dock()
        self.tabifyDocks()

        self.init_menubar()

    def init_controller(self):
        self.controller = Controller()
        self.controller.stage_move_requested.connect(
            DeviceManager.instance().moveRequest
        )
        self.controller.stage_stop_requested.connect(
            DeviceManager.instance().stopRequest
        )
        self.controller.stage_home_requested.connect(
            DeviceManager.instance().homeRequest
        )
        self.controller.stage_toggle_lock.connect(DeviceManager.instance().toggleLock)
        FocusStabilizer.instance().focusStabilizationToggled.connect(
            self.controller.set_stabilizer_lock
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.controller, DOCKS.CONTROLLER
        )

    def init_devices_dock(self):
        # General settings groupbox
        self.devicesDock = QtWidgets.QDockWidget('Devices', self)
        self.devicesDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        devicesWidget = QtWidgets.QWidget()
        self.devicesDock.setWidget(devicesWidget)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.devicesDock, DOCKS.DEVICES
        )

        # vertical layout
        devicesLayout = QtWidgets.QGridLayout()
        devicesWidget.setLayout(devicesLayout)

        self.devicesView = DevicesView()
        self.devicesView.setDetectorActivated.connect(
            self.device_manager._set_ir_array_detector
        )
        self.devicesView.resetDetectorActivated.connect(
            self.device_manager._remove_ir_array_detector
        )
        self.devicesView.addLaserActivated.connect(self.device_manager._add_laser)
        self.devicesView.setStageActivated.connect(self.device_manager._set_z_stage)

        devicesLayout.addWidget(self.devicesView, 0, 1)

    def init_stages_dock(self):
        # Stages Tab (Elliptec + Kinesis Tab + Scan Acquisition)
        self.stagesDock = QtWidgets.QDockWidget('Stages', self)
        self.stagesDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.stagesWidget = QtWidgets.QTabWidget()
        self.stagesWidget.setMinimumWidth(350)
        self.stagesDock.setWidget(self.stagesWidget)

        self.stagesWidget.addTab(
            self.acquisition_manager.acquisitionWidget, 'Scan Acquistion'
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self.stagesDock, DOCKS.STAGES
        )

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

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.pyDock, DOCKS.PYSCRIPT
        )

    def init_protocol_dock(self):
        self.protocolDock = QtWidgets.QDockWidget('Protocols', self)
        self.protocolDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.actionEditor = ExperimentDesigner()
        self.protocolDock.setWidget(self.actionEditor)

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.protocolDock, DOCKS.PROTOCOLS
        )

    def init_lasers_dock(self):
        # Lasers Tab
        self.lasersDock = QtWidgets.QDockWidget('Lasers', self)
        self.lasersDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.lasersLayout = QtWidgets.QGridLayout()

        self.lasersWidget = QtWidgets.QWidget()
        self.lasersWidget.setLayout(self.lasersLayout)

        self.lasersTabs = QtWidgets.QTabWidget()

        self.lasersDock.setWidget(self.lasersWidget)

        self.lasersLayout.addWidget(self.lasersTabs, 0, 1)

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.lasersDock, DOCKS.LASERS
        )

    def init_cam_dock(self):
        # cameras tab
        self.camDock = QtWidgets.QDockWidget('Cameras List', self)
        self.camDock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, self.camDock, DOCKS.CAMERAS
        )

    def open_link(self, url: str):
        '''Open a URL in the default web browser.'''
        import webbrowser

        if not webbrowser.open(url):
            print(f'Failed to open URL: {url}')
            QtWidgets.QMessageBox.warning(
                self, 'Error', f'Could not open URL: {url}'
            )

    def init_menubar(self):
        # Create menu bar
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
        tools_menu = menu_bar.addMenu('Tools')
        help_menu = menu_bar.addMenu('Help')

        # Create exit action
        save_config = QAction('Save Config.', self)
        save_config.triggered.connect(lambda: generateConfig(self))
        load_config = QAction('Load Config.', self)
        load_config.triggered.connect(lambda: loadConfig(self, False))
        auto_load_config = QAction('Load Config. && Connect', self)
        auto_load_config.triggered.connect(lambda: loadConfig(self, True))
        disconnect_devices = QAction('Disconnect Devices', self)
        disconnect_devices.triggered.connect(
            lambda: self.device_manager.shutdown(False)
        )
        shutdown_and_exit = QAction('Exit & Disconnect Devices', self)
        shutdown_and_exit.triggered.connect(lambda: self.device_manager.shutdown())

        github = QAction('microEye Github', self)
        github.triggered.connect(
            lambda: self.open_link('https://github.com/samhitech/microEye')
        )
        pypi = QAction('microEye PYPI', self)
        pypi.triggered.connect(
            lambda: self.open_link('https://pypi.org/project/microEye/')
        )

        # Add exit action to file menu
        file_menu.addAction(save_config)
        file_menu.addAction(load_config)
        file_menu.addAction(auto_load_config)
        file_menu.addAction(disconnect_devices)
        file_menu.addAction(shutdown_and_exit)

        docks: list[QtWidgets.QDockWidget] = [
            self.controller,
            self.devicesDock,
            self.pyDock,
            self.stagesDock,
            self.camDock,
            self.lasersDock,
        ]

        def connect(action: QAction, dock: QtWidgets.QDockWidget):
            action.triggered.connect(lambda: dock.setVisible(action.isChecked()))

        for dock in docks:
            dock_act = dock.toggleViewAction()
            dock_act.setEnabled(True)
            if '6' in QT_API:
                connect(dock_act, dock)
            view_menu.addAction(dock_act)

        # Tools menu
        pycro_headless = HeadlessManagerWidget.get_menu_action(self)
        core_instances = BridgesWidget.get_menu_action(self)
        objective_tool = ObjectiveCalculator.get_menu_action(self)

        tools_menu.addAction(pycro_headless)
        tools_menu.addAction(core_instances)
        tools_menu.addSeparator()
        tools_menu.addAction(objective_tool)

        help_menu.addAction(github)
        help_menu.addAction(pypi)

    def tabifyDocks(self):
        self.setTabPosition(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            QtWidgets.QTabWidget.TabPosition.North,
        )
        self.setTabPosition(
            Qt.DockWidgetArea.RightDockWidgetArea,
            QtWidgets.QTabWidget.TabPosition.North,
        )

        self.tabifyDockWidget(self.lasersDock, self.devicesDock)
        self.tabifyDockWidget(self.lasersDock, self.camDock)
        self.tabifyDockWidget(self.lasersDock, self.protocolDock)
        self.tabifyDockWidget(self.lasersDock, self.pyDock)

        self.tabifyDockWidget(self.stagesDock, self.controller)

        self.stagesDock.raise_()

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

    def _add_widgets(self, device: DEVICES, widget: QtWidgets.QWidget):
        if device == DEVICES.LASER:
            self.lasersTabs.addTab(widget, f'Laser #{self.lasersTabs.count() + 1}')

            self.lasersTabs.setCurrentIndex(self.lasersTabs.count() - 1)

        elif device == DEVICES.IR_CAM:
            if isinstance(widget, Camera_Panel):
                # IR camera widget
                self.cam_dock = self.getDockWidget(widget._cam.name, widget)
                self.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea,
                    self.cam_dock,
                    DOCKS.FOCUS_CAMERA,
                )
                self.tabifyDockWidget(self.lasersDock, self.cam_dock)
                self.cam_dock.raise_()  # Bring to front
                self.device_manager.focus.graph_IR.setLabel(
                    'left', 'Signal', '', **self.labelStyle
                )

            else:
                if self.ir_array_dock is None:
                    self.ir_array_dock = QtWidgets.QDockWidget('Array Detector')
                    self.ir_array_dock.setFeatures(
                        QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                        | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                    )
                    self.addDockWidget(
                        Qt.DockWidgetArea.RightDockWidgetArea,
                        self.ir_array_dock,
                        DOCKS.FOCUS_CAMERA,
                    )
                    self.tabifyDockWidget(self.devicesDock, self.ir_array_dock)

                self.ir_array_dock.setWidget(widget)
                self.ir_array_dock.raise_()  # Bring to front
                self.device_manager.focus.graph_IR.setLabel(
                    'left', 'Signal', 'V', **self.labelStyle
                )

        elif device == DEVICES.Z_STAGE:
            self.stagesWidget.insertTab(0, widget, 'Z-Stage')
            self.stagesWidget.setCurrentIndex(0)

        elif device == DEVICES.XY_STAGE:
            self.stagesWidget.insertTab(1, widget, 'XY-Stage')
            self.stagesWidget.setCurrentIndex(1)

        elif device == DEVICES.ELLIPTEC:
            self.stagesWidget.insertTab(2, widget, 'Elliptec Devices')
            self.stagesWidget.setCurrentIndex(2)

        elif device == DEVICES.HID_CONTROLLER:
            devicesLayout = self.devicesDock.widget().layout()

            devicesLayout.addWidget(widget, 0, 0)

        elif device == DEVICES.LASER_RELAY:
            self.lasersLayout.addWidget(widget, 0, 0)

        elif device == DEVICES.FocusStabilizer:
            self.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, widget, DOCKS.FOCUS
            )
            self.tabifyDockWidget(self.lasersDock, widget)
            widget.raise_()  # Bring to front
        elif device == DEVICES.CAM_LIST:
            self.camDock.setWidget(widget)

    def _remove_widgets(self, device: DEVICES, widget: QtWidgets.QWidget):
        if device == DEVICES.LASER:
            index = self.lasersTabs.indexOf(widget)
            self.lasersTabs.removeTab(index)
        elif device == DEVICES.IR_CAM:
            if isinstance(widget, Camera_Panel):
                self.removeDockWidget(self.cam_dock)
                self.cam_dock.deleteLater()
                self.cam_dock = None

            else:
                self.removeDockWidget(self.ir_array_dock)
                self.ir_array_dock.deleteLater()
                self.ir_array_dock = None

        elif device == DEVICES.Z_STAGE:
            pass

    def update_gui(self):
        '''Recurring timer updates the status bar and GUI'''

        RelayBox = '    |  Relay ' + (
            'connected'
            if DeviceManager.instance().laser_relay.isOpen()
            else 'disconnected'
        )

        Position = ''
        Frames = '    | Frames Saved: ' + str(
            FocusStabilizer.instance().num_frames_saved
        )

        Worker = f'    | Execution time: {FocusStabilizer.instance()._exec_time:.0f}'
        if DeviceManager.instance().camList.autofocusCam:
            Worker += (
                f'    | Frames Buffer: {FocusStabilizer.instance().bufferSize():d}'
            )
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
        DeviceManager.instance().elliptecView.updateHighlight()

        DeviceManager.instance().laser_relay.updatePortState()
        if not DeviceManager.instance().laser_relay.isOpen():
            DeviceManager.instance().laser_relay.refreshPorts()
            DeviceManager.instance().laser_relay.updateHighlight(
                self.device_manager.laser_relay_settings()
            )
        else:
            DeviceManager.instance().laser_relay.updateHighlight(
                self.device_manager.laser_relay_settings()
            )

        for _, cam_list in CameraList.CAMERAS.items():
            for cam in cam_list:
                cam['Panel'].updateInfo()

        if DeviceManager.instance().stage:
            DeviceManager.instance().stage.updatePortState()
            DeviceManager.instance().stage.refreshPorts()

    def addDockWidget(
        self, area: Qt.DockWidgetArea, widget: QtWidgets.QDockWidget, key: DOCKS
    ):
        super().addDockWidget(area, widget)

        if key and key not in self.dock_widgets:
            self.dock_widgets[key] = widget

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

    config = mieye.device_manager.get_config()

    # mieye.dock_widgets = [

    config['miEye_module'] = {
        'position': {
            'x': mieye.mapToGlobal(QtCore.QPoint(0, 0)).x(),
            'y': mieye.mapToGlobal(QtCore.QPoint(0, 0)).y(),
        },
        'size': {
            'width': mieye.geometry().width(),
            'height': mieye.geometry().height(),
        },
        'is_maximized': mieye.isMaximized(),
    }

    for _, (key, widget) in enumerate(mieye.dock_widgets.items()):
        if isinstance(widget, QtWidgets.QDockWidget):
            config[key.get_key()] = {
                'is_floating': widget.isFloating(),
                'position': {
                    'x': widget.mapToGlobal(QtCore.QPoint(0, 0)).x(),
                    'y': widget.mapToGlobal(QtCore.QPoint(0, 0)).y(),
                },
                'size': {
                    'width': widget.geometry().width(),
                    'height': widget.geometry().height(),
                },
                'is_visible': widget.isVisible(),
            }

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
        if (
            'is_maximized' in config['miEye_module']
            and config['miEye_module']['is_maximized']
        ):
            mieye.showMaximized()
        else:
            mieye.setGeometry(
                config['miEye_module'].get('position', {}).get('x', 100),
                config['miEye_module'].get('position', {}).get('y', 100),
                config['miEye_module'].get('size', {}).get('width', 1200),
                config['miEye_module'].get('size', {}).get('height', 920),
            )

    for dock in DOCKS:
        if dock.get_key() in config:
            dock_config: dict = config[dock.get_key()]
            widget = mieye.dock_widgets[dock]
            widget.setVisible(bool(dock_config.get('is_visible', True)))
            if bool(dock_config.get('is_floating', False)):
                widget.setFloating(True)
                widget.setGeometry(
                    dock_config['position']['x'],
                    dock_config['position']['y'],
                    dock_config['size']['width'],
                    dock_config['size']['height'],
                )
            else:
                widget.setFloating(False)

    mieye.device_manager.load_config(config)
    if auto_connect:
        mieye.device_manager.auto_connect()

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
