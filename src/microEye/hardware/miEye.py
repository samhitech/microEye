import json
import os
import sys
import traceback
import warnings
import webbrowser
from queue import Queue

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
from pyqtgraph.widgets.PlotWidget import PlotWidget
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView
from scipy.optimize import curve_fit
from scipy.optimize.optimize import OptimizeWarning
from scipy.signal import find_peaks

from ..hid_controller import *
from ..pyscripting import *
from ..qlist_slider import *
from ..thread_worker import *
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

warnings.filterwarnings("ignore", category=OptimizeWarning)


class miEye_module(QMainWindow):
    '''The main GUI for miEye combines control and acquisition modules

    | Inherits QMainWindow
    '''

    def __init__(self, *args, **kwargs):
        super(miEye_module, self).__init__(*args, **kwargs)

        # setting title
        self.setWindowTitle(
            "microEye control module \
            (https://github.com/samhitech/microEye)")

        # setting geometry
        self.setGeometry(0, 0, 1200, 920)

        # Statusbar time
        self.statusBar().showMessage(
            "Time: " + QDateTime.currentDateTime().toString("hh:mm:ss,zzz"))

        # Threading
        self._threadpool = QThreadPool.globalInstance()
        print("Multithreading with maximum %d threads"
              % self._threadpool.maxThreadCount())

        # PiezoConcept
        self.stage = stage()

        # kinesis XY Stage
        self.kinesisXY = KinesisXY(threadpool=self._threadpool)

        self.lastTile = None
        self._stop_scan = False
        self._scanning = False

        # Serial Port IR linear CCD array
        self.IR_Cam = IR_Cam()

        # Acquisition Cameras lists
        self.ids_cams: list[IDS_Camera] = []
        self.thorlabs_cams: list[thorlabs_camera] = []
        self.vimba_cams: list[vimba_cam] = []

        # Acquisition Cameras Panels
        self.ids_panels: list[IDS_Panel] = []
        self.thor_panels: list[Thorlabs_Panel] = []
        self.vimba_panels: list[Vimba_Panel] = []

        # IR 2D Camera
        self.cam = None
        self.cam_panel = None
        self.cam_dock = None

        # Serial Port Laser Relay
        self.laserRelay = QSerialPort(
            self)
        self.laserRelay.setBaudRate(115200)
        self.laserRelay.setPortName('COM6')
        self.laserRelay_last = ""
        self.laserRelay_curr = ""

        # Elliptec controller
        self._elliptec_controller = elliptec_controller()

        # Layout
        self.LayoutInit()

        # File object for saving frames
        self.file = None

        # Data variables
        n_data = 128
        self.xdata = np.array(list(range(n_data)))
        # self.ydata_buffer = Queue()
        # self.ydata_buffer.put(np.array([0 for i in range(n_data)]))
        self.y_index = 0
        self.centerDataX = np.array(list(range(500)))
        self.centerData = np.ones((500,))
        self.error_buffer = np.zeros((20,))
        self.error_integral = 0
        self.ydata_temp = None
        self.popt = None
        self.frames_saved = 0

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None
        self._plotfit_ref = None
        self._center_ref = None

        # Statues Bar Timer
        self.timer = QTimer()
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

        # Any other args, kwargs are passed to the run function
        self.worker = thread_worker(self.worker_function)
        self.worker.signals.progress.connect(self.update_graphs)
        self.worker.signals.move_stage_z.connect(self.movez_stage)
        # Execute
        self._threadpool.start(self.worker)

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

    def eventFilter(self, source, event):
        if qApp.activePopupWidget() is None:
            if event.type() == QEvent.MouseMove:
                if self.menuBar().isHidden():
                    rect = self.geometry()
                    rect.setHeight(40)

                    if rect.contains(event.globalPos()):
                        self.menuBar().show()
                else:
                    rect = QRect(
                        self.menuBar().mapToGlobal(QPoint(0, 0)),
                        self.menuBar().size()
                    )

                    if not rect.contains(event.globalPos()):
                        self.menuBar().hide()
            elif event.type() == QEvent.Leave and source is self:
                self.menuBar().hide()
        return super().eventFilter(source, event)

    def LayoutInit(self):
        '''Initializes the window layout
        '''

        Hlayout = QHBoxLayout()

        # General settings groupbox
        self.devicesDock = QDockWidget('Devices', self)
        self.devicesDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        LPanel_GBox = QWidget()
        self.devicesDock.setWidget(LPanel_GBox)
        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea,
            self.devicesDock)

        # vertical layout
        Left_Layout = QFormLayout()
        LPanel_GBox.setLayout(Left_Layout)

        # IR Cam combobox
        self.ir_cam_cbox = QComboBox()
        self.ir_cam_cbox.addItem('Parallax CCD (TSL1401)')
        self.ir_cam_set_btn = QPushButton(
            "Set",
            clicked=self.setIRcam
        )
        self.ir_cam_reset_btn = QPushButton(
            "Reset",
            clicked=self.resetIRcam
        )
        self.ir_widget = None

        # ALEX checkbox
        self.ALEX = QCheckBox("ALEX")
        self.ALEX.state = "ALEX"
        self.ALEX.setChecked(False)

        # IR CCD array arduino buttons

        # records IR peak-fit position
        self.start_IR_btn = QPushButton(
            "Start IR Acquisition",
            clicked=self.start_IR
        )
        self.stop_IR_btn = QPushButton(
            "Stop IR Acquisition",
            clicked=self.stop_IR
        )

        # Add Laser controls
        self.lasers_cbox = QComboBox()
        self.lasers_cbox.addItem('IO MatchBox Single Laser')
        self.lasers_cbox.addItem('IO MatchBox Laser Combiner')
        self.add_laser_btn = QPushButton(
            "Add Laser",
            clicked=lambda: self.add_laser_panel()
        )

        # Arduino RelayBox controls
        self.laser_relay_connect_btn = QPushButton(
            "Connect",
            clicked=lambda: self.connectToPort(self.laserRelay)
        )
        self.laser_relay_disconnect_btn = QPushButton(
            "Disconnect",
            clicked=lambda: self.disconnectFromPort(self.laserRelay)
        )
        self.laser_relay_btn = QPushButton(
            "Config.",
            clicked=lambda: self.open_dialog(self.laserRelay)
        )
        self.send_laser_relay_btn = QPushButton(
            "Send Setting",
            clicked=lambda: self.sendConfig(self.laserRelay)
        )

        # Piezostage controls
        self.stage_cbox = QComboBox()
        self.stage_cbox.addItem('PiezoConcept FOC100')
        self.stage_set_btn = QPushButton(
            "Set",
            clicked=self.setStage
        )
        self.stage_widget = None
        self.stage_dock = None
        self.stage_set_btn.click()

        self.hid_controller = hid_controller()
        self.hid_controller.reportEvent.connect(self.hid_report)
        self.hid_controller.reportRStickPosition.connect(
            self.hid_RStick_report)
        self.hid_controller.reportLStickPosition.connect(
            self.hid_LStick_report)
        self.hid_controller_toggle = False

        Left_Layout.addWidget(self.hid_controller)

        Left_Layout.addRow(
            QLabel('IR Camera:'), self.ir_cam_cbox)

        ir_cam_layout_0 = QHBoxLayout()
        ir_cam_layout_0.addWidget(self.ir_cam_set_btn)
        ir_cam_layout_0.addWidget(self.ir_cam_reset_btn)
        Left_Layout.addRow(ir_cam_layout_0)

        ir_cam_layout_1 = QHBoxLayout()
        ir_cam_layout_1.addWidget(self.start_IR_btn)
        ir_cam_layout_1.addWidget(self.stop_IR_btn)
        Left_Layout.addRow(ir_cam_layout_1)

        Left_Layout.addRow(
            DragLabel('Stage:', parent_name='self.stage_set_btn'),
            self.stage_cbox)
        Left_Layout.addWidget(self.stage_set_btn)

        Left_Layout.addRow(
            QLabel('Lasers:'),
            self.lasers_cbox)
        Left_Layout.addWidget(
            self.add_laser_btn)

        relay_btns_0 = QHBoxLayout()
        relay_btns_1 = QHBoxLayout()
        relay_btns_0.addWidget(self.laser_relay_connect_btn)
        relay_btns_0.addWidget(self.laser_relay_disconnect_btn)
        relay_btns_0.addWidget(self.laser_relay_btn)
        relay_btns_1.addWidget(self.ALEX)
        relay_btns_1.addWidget(self.send_laser_relay_btn, 1)
        Left_Layout.addRow(
            QLabel('Laser Relay:'), relay_btns_0)
        Left_Layout.addRow(relay_btns_1)

        # Stages Tab (Elliptec + Kinesis Tab + Scan Acquisition)
        self.stagesDock = QDockWidget('Stages', self)
        self.stagesDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        self.stages_Layout = QHBoxLayout()
        self.stagesWidget = QWidget()
        self.stagesWidget.setLayout(self.stages_Layout)
        self.stagesDock.setWidget(self.stagesWidget)

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

        self.scanAcqWidget.moveZ.connect(self.movez_stage)

        self.stages_Layout.addWidget(self.kinesisXY.getQWidget())
        self.stages_Layout.addWidget(self.scanAcqWidget)

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
        self.addDockWidget(
            Qt.DockWidgetArea.TopDockWidgetArea,
            self.stagesDock)
        self.tabifyDockWidget(
            self.pyDock, self.stagesDock)

        # Lasers Tab
        self.lasersDock = QDockWidget('Lasers', self)
        self.lasersDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        self.lasersLayout = QHBoxLayout()
        self.lasersLayout.addStretch()
        self.lasersWidget = QWidget()
        self.lasersWidget.setLayout(self.lasersLayout)

        self.lasersDock.setWidget(self.lasersWidget)

        self.laserPanels = []

        # cameras tab
        self.camDock = QDockWidget('Cameras List', self)
        self.camDock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)

        self.camListWidget = CameraListWidget()
        self.camListWidget.addCamera.connect(self.add_camera)
        self.camListWidget.removeCamera.connect(self.remove_camera)

        self.camDock.setWidget(self.camListWidget)
        self.setTabPosition(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            QTabWidget.TabPosition.North)

        # display tab
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        self.focus = focusWidget()

        # Tabs
        self.addDockWidget(0x8, self.lasersDock)
        self.addDockWidget(0x8, self.camDock)
        self.tabifyDockWidget(self.lasersDock, self.camDock)
        self.addDockWidget(0x8, self.focus)
        self.tabifyDockWidget(self.lasersDock, self.focus)

        AllWidgets = QWidget()
        AllWidgets.setLayout(Hlayout)

        # self.setCentralWidget(AllWidgets)

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
        stage_act = self.stage_dock.toggleViewAction()
        stage_act.setEnabled(True)
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
        view_menu.addAction(stage_act)
        view_menu.addAction(pyDock_act)
        view_menu.addAction(stages_act)
        view_menu.addAction(lasers_act)
        view_menu.addAction(cams_act)
        view_menu.addAction(focus_act)

        help_menu.addAction(github)
        help_menu.addAction(pypi)

        qApp.installEventFilter(self)

        self.menuBar().hide()

    def scriptTest(self):
        exec(self.pyEditor.toPlainText())

    def getDockWidget(self, text: str, content: QWidget):
        dock = QDockWidget(text, self)
        dock.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)
        dock.setWidget(content)
        return dock

    def add_camera(self, cam):
        res = QMessageBox.information(
            self,
            'Acquisition / IR camera',
            'Add acquisition or IR camera? (Yes/No)',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Cancel)

        if res == QMessageBox.StandardButton.Yes:
            self.add_camera_clicked(cam)
        elif res == QMessageBox.StandardButton.No:
            self.add_IR_camera(cam)
        else:
            pass

    def remove_camera(self, cam):
        res = QMessageBox.warning(
            self,
            'Acquisition / IR camera',
            'Remove acquisition or IR camera? (Yes/No)',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No |
            QMessageBox.StandardButton.Cancel)

        if res == QMessageBox.StandardButton.Yes:
            self.remove_camera_clicked(cam)
        elif res == QMessageBox.StandardButton.No:
            self.remove_IR_camera(cam)
        else:
            pass

    def add_IR_camera(self, cam):
        if self.cam is not None:
            QMessageBox.warning(
                self,
                "Warning",
                "Please remove {}.".format(self.cam.name),
                QMessageBox.StandardButton.Ok)
            return

        if not self.IR_Cam.isDummy():
            QMessageBox.warning(
                self,
                "Warning",
                "Please remove {}.".format(self.IR_Cam.name),
                QMessageBox.StandardButton.Ok)
            return

        # print(cam)
        if cam["InUse"] == 0:
            if 'uEye' in cam["Driver"]:
                ids_cam = IDS_Camera(cam["camID"])
                nRet = ids_cam.initialize()
                self.cam = ids_cam
                ids_panel = IDS_Panel(
                    self._threadpool,
                    ids_cam, True,
                    cam["Model"] + " " + cam["Serial"])
                # ids_panel._directory = self.save_directory
                ids_panel.master = False
                self.cam_panel = ids_panel
                self.cam_dock = self.getDockWidget(
                    ids_cam.name, ids_panel)
            if 'UC480' in cam["Driver"]:
                thor_cam = thorlabs_camera(cam["camID"])
                nRet = thor_cam.initialize()
                if nRet == CMD.IS_SUCCESS:
                    self.cam = thor_cam
                    thor_panel = Thorlabs_Panel(
                        self._threadpool,
                        thor_cam, True,
                        cam["Model"] + " " + cam["Serial"])
                    # thor_panel._directory = self.save_directory
                    thor_panel.master = False
                    self.cam_panel = thor_panel
                    self.cam_dock = self.getDockWidget(
                        thor_cam.name, thor_panel)
            if 'Vimba' in cam["Driver"]:
                v_cam = vimba_cam(cam["camID"])
                self.cam = v_cam
                v_panel = Vimba_Panel(
                        self._threadpool,
                        v_cam, True, cam["Model"] + " " + cam["Serial"])
                v_panel.master = False
                self.cam_panel = v_panel
                self.cam_dock = self.getDockWidget(
                    v_cam.name, v_panel)

            self.addDockWidget(
                Qt.DockWidgetArea.BottomDockWidgetArea,
                self.cam_dock)
            self.tabifyDockWidget(
                self.lasersDock, self.cam_dock)
            self.focus.graph_IR.setLabel(
                "left", "Signal", "", **self.labelStyle)
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Device is in use or already added.",
                QMessageBox.StandardButton.Ok)

    def remove_IR_camera(self, cam):
        if self.cam.cInfo.SerNo.decode('utf-8') == cam["Serial"]:
            if not self.cam.acquisition:
                if self.cam.free_memory:
                    self.cam.free_memory()
                    self.cam.dispose()

            self.cam_panel._dispose_cam = True
            self.cam_panel._stop_thread = True
            self.removeDockWidget(self.cam_dock)
            self.cam_dock.deleteLater()
            self.cam = None
            self.cam_panel = None
            self.cam_dock = None

    def add_camera_clicked(self, cam):
        home = os.path.expanduser('~')
        _directory = os.path.join(home, 'Desktop')

        if cam["InUse"] == 0:
            if 'uEye' in cam["Driver"]:
                ids_cam = IDS_Camera(cam["camID"])
                nRet = ids_cam.initialize()
                self.ids_cams.append(ids_cam)
                ids_panel = IDS_Panel(
                    self._threadpool,
                    ids_cam, False, cam["Model"] + " " + cam["Serial"])
                ids_panel._directory = _directory
                ids_panel.master = False
                ids_panel.show()
                self.ids_panels.append(ids_panel)
            if 'UC480' in cam["Driver"]:
                thor_cam = thorlabs_camera(cam["camID"])
                nRet = thor_cam.initialize()
                if nRet == CMD.IS_SUCCESS:
                    self.thorlabs_cams.append(thor_cam)
                    thor_panel = Thorlabs_Panel(
                        self._threadpool,
                        thor_cam, False, cam["Model"] + " " + cam["Serial"])
                    thor_panel._directory = _directory
                    thor_panel.master = False
                    thor_panel.show()
                    self.thor_panels.append(thor_panel)
            if 'Vimba' in cam["Driver"]:
                v_cam = vimba_cam(cam["camID"])
                self.vimba_cams.append(v_cam)
                v_panel = Vimba_Panel(
                        self._threadpool,
                        v_cam, False, cam["Model"] + " " + cam["Serial"])
                v_panel._directory = _directory
                v_panel.master = False
                v_panel.show()
                self.vimba_panels.append(v_panel)
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Device is in use or already added.",
                QMessageBox.StandardButton.Ok)

    def remove_camera_clicked(self, cam):

        if 'uEye' in cam["Driver"]:
            for pan in self.ids_panels:
                if pan.cam.Cam_ID == cam["camID"]:
                    if not pan.cam.acquisition:
                        pan.cam.free_memory()
                        pan.cam.dispose()

                        pan._dispose_cam = True
                        pan._stop_thread = True
                        self.ids_cams.remove(pan.cam)
                        self.ids_panels.remove(pan)
                        pan.close()
                        pan.setParent(None)
                        break
        if 'UC480' in cam["Driver"]:
            for pan in self.thor_panels:
                # if pan.cam.hCam.value == cam["camID"]:
                if pan.cam.cInfo.SerNo.decode('utf-8') == cam["Serial"]:
                    if not pan.cam.acquisition:
                        pan.cam.free_memory()
                        pan.cam.dispose()

                        pan._dispose_cam = True
                        pan._stop_thread = True
                        self.thorlabs_cams.remove(pan.cam)
                        self.thor_panels.remove(pan)
                        pan.close()
                        pan.setParent(None)
                        break
        if 'Vimba' in cam["Driver"]:
            for pan in self.vimba_panels:
                with pan.cam.cam:
                    if pan.cam.cam.get_serial() == cam["Serial"]:
                        if not pan.cam.acquisition:
                            pan._dispose_cam = True
                            if pan.acq_job is not None:
                                pan.acq_job.stop_threads = True
                            self.vimba_cams.remove(pan.cam)
                            self.vimba_panels.remove(pan)
                            pan.close()
                            pan.setParent(None)
                            break

    def isEmpty(self):
        if self.cam_panel is not None:
            return self.cam_panel.isEmpty
        elif not self.IR_Cam.isDummy():
            return self.IR_Cam.isEmpty
        else:
            return True

    def isImage(self):
        if self.cam is not None:
            return True
        elif not self.IR_Cam.isDummy():
            return False
        else:
            return False

    def BufferGet(self) -> Queue:
        if self.cam is not None:
            return self.cam_panel.get(True)
        elif not self.IR_Cam.isDummy():
            return self.IR_Cam.buffer.get()
        else:
            return np.zeros((256, 256), dtype=np.uint16)

    def BufferSize(self) -> Queue:
        if self.cam is not None:
            return self.cam_panel.bufferSize
        elif not self.IR_Cam.isDummy():
            return 0
        else:
            return 0

    def relaySettings(self):
        '''Returns the RelayBox setting command.

        Returns
        -------
        str
            the RelayBox setting command.
        '''
        settings = ''
        for panel in self.laserPanels:
            settings += panel.GetRelayState()
        return settings + \
            ("ALEXON" if self.ALEX.isChecked() else "ALEXOFF") + "\r"

    def sendConfig(self, serial: QSerialPort):
        '''Sends the RelayBox setting command.

        Parameters
        ----------
        serial : QSerialPort
            the RelayBox serial port.
        '''
        try:

            message = self.relaySettings()

            serial.write(message.encode('utf-8'))
            self.laserRelay_last = message
            print(str(serial.readAll(), encoding="utf-8"))
        except Exception as e:
            print('Failed Laser Relay Send Config: ' + str(e))

    def add_laser_panel(self):
        selected = self.lasers_cbox.currentText()
        if 'IO MatchBox' in selected:
            if 'Combiner' in selected:
                combiner = CombinerLaserWidget()
                self.laserPanels.append(combiner)
                self.lasersLayout.insertWidget(0, combiner)
            elif 'Single' in selected:
                laser = SingleLaserWidget()
                self.laserPanels.append(laser)
                self.lasersLayout.insertWidget(0, laser)

    def worker_function(self, progress_callback, movez_callback):
        '''A worker function running in the threadpool.

        Handles the IR peak fitting and piezo autofocus tracking.
        '''
        counter = 0
        self._exec_time = 0
        time = QDateTime.currentDateTime()
        QThread.msleep(100)
        while (self.isVisible()):
            try:
                # dt = Gaussian(
                #     np.array(range(512)), 255, np.random.normal() + 256, 50)
                # x, y = np.meshgrid(dt, dt)
                # dt = x * y
                # self.remote_img.setImage(dt)
                # ax, pos = self.focus.roi.getArrayRegion(
                #     dt, self.remote_img, returnMappedCoords=True)
                # self.IR_Cam._buffer.put(ax)
                # proceed only if the buffer is not empty
                if not self.isEmpty():
                    self._exec_time = time.msecsTo(QDateTime.currentDateTime())
                    time = QDateTime.currentDateTime()

                    data = self.BufferGet()

                    if self.isImage():
                        self.focus.remote_img.setImage(
                            data.copy(), _callSync='off')
                        data, _ = self.focus.roi.getArrayRegion(
                            data, self.focus.remote_img,
                            returnMappedCoords=True)
                        data = np.squeeze(data)
                    # self.ydata_temp = self.ydata
                    self.peak_fit(movez_callback, data.copy())
                    if (self.file is not None):
                        np.savetxt(self.file,
                                   np.array((self._exec_time, self.popt[1]))
                                   .reshape((1, 2)), delimiter=";")
                        self.frames_saved = 1 + self.frames_saved
                    counter = counter + 1
                    progress_callback.emit(data)
                QThread.usleep(100)
            except Exception as e:
                traceback.print_exc()

    def peak_fit(self, movez_callback: pyqtSignal(bool, int), data):
        '''Finds the peak position through fitting
        and adjsuts the piezostage accordingly.
        '''
        try:
            # find IR peaks above a specific height
            peaks = find_peaks(data, height=1)
            popt = None  # fit parameters
            nPeaks = len(peaks[0])  # number of peaks
            maxPeakIdx = np.argmax(peaks[1]['peak_heights'])  # highest peak
            x0 = 64 if nPeaks == 0 else peaks[0][maxPeakIdx]
            a0 = 1 if nPeaks == 0 else peaks[1]['peak_heights'][maxPeakIdx]

            # gmodel = Model(GaussianOffSet)
            # self.result = gmodel.fit(
            #     data, x=self.xdata, a=a0, x0=x0, sigma=1, offset=0)

            # curve_fit to GaussianOffSet
            self.popt, pcov = curve_fit(
                GaussianOffSet,
                self.xdata,
                data,
                p0=[a0, x0, 1, 0])
            self.centerData = np.roll(self.centerData, -1)
            self.centerData[-1] = self.popt[1]  # self.result.best_values['x0']

            if self.stage.piezoTracking:
                err = np.average(self.centerData[-1] - self.stage.center_pixel)
                tau = self.stage.tau if self.stage.tau > 0 \
                    else (self._exec_time / 1000)
                self.error_integral += err * tau
                diff = (err - self.error_buffer[-1]) / tau
                self.error_buffer = np.roll(self.error_buffer, -1)
                self.error_buffer[-1] = err
                step = int((err * self.stage.pConst) +
                           (self.error_integral * self.stage.iConst) +
                           (diff * self.stage.dConst))
                if abs(err) > self.stage.threshold:
                    if step > 0:
                        movez_callback.emit(False, abs(step))
                    else:
                        movez_callback.emit(True, abs(step))
            else:
                self.stage.center_pixel = self.centerData[-1]
                self.error_buffer = np.zeros((20,))
                self.error_integral = 0

        except Exception as e:
            pass
            # print('Failed Gauss. fit: ' + str(e))

    def movez_stage(self, dir: bool, step: int):
        # print(up, step)
        if self.stage._inverted.isChecked():
            dir = not dir

        if dir:
            self.stage.UP(step)
        else:
            self.stage.DOWN(step)

    def update_graphs(self, data):
        '''Updates the graphs.
        '''
        self.xdata = range(len(data))
        # updates the IR graph with data
        if self._plot_ref is None:
            # create plot reference when None
            self._plot_ref = self.focus.graph_IR.plot(self.xdata, data)
        else:
            # use the plot reference to update the data for that line.
            self._plot_ref.setData(self.xdata, data)

        # updates the IR graph with data fit
        if self._plotfit_ref is None:
            # create plot reference when None
            self._plotfit_ref = self.focus.graph_IR.plot(
                self.xdata,
                GaussianOffSet(self.xdata, *self.popt))
        else:
            # use the plot reference to update the data for that line.
            self._plotfit_ref.setData(
                self.xdata,
                GaussianOffSet(self.xdata, *self.popt))

        if self._center_ref is None:
            # create plot reference when None
            self._center_ref = self.focus.graph_Peak.plot(
                self.centerDataX, self.centerData)
        else:
            # use the plot reference to update the data for that line.
            self._center_ref.setData(
                self.centerDataX, self.centerData)

    def update_gui(self):
        '''Recurring timer updates the status bar and GUI
        '''
        IR = ("    |  IR Cam " +
              ('connected' if self.IR_Cam.isOpen else 'disconnected'))

        RelayBox = ("    |  Relay " + ('connected' if self.laserRelay.isOpen()
                    else 'disconnected'))
        Piezo = ("    |  Piezo " + ('connected' if self.stage.isOpen()
                 else 'disconnected'))

        Position = ''
        if self.stage.isOpen():
            # self.piezoConcept.GETZ()
            Position = "    |  Position " + self.stage.Received
        Frames = "    | Frames Saved: " + str(self.frames_saved)

        Worker = "    | Execution time: {:d}".format(self._exec_time)
        if self.cam is not None:
            Worker += "    | Frames Buffer: {:d}".format(self.BufferSize())
        self.statusBar().showMessage(
            "Time: " + QDateTime.currentDateTime().toString("hh:mm:ss,zzz")
            + IR + RelayBox
            + Piezo + Position + Frames + Worker)

        # update indicators
        if self.IR_Cam.isOpen:
            self.IR_Cam._connect_btn.setStyleSheet("background-color: green")
        else:
            self.IR_Cam._connect_btn.setStyleSheet("background-color: red")
        if self.stage.isOpen():
            self.stage._connect_btn.setStyleSheet("background-color: green")
        else:
            self.stage._connect_btn.setStyleSheet("background-color: red")
        if self.laserRelay.isOpen():
            self.laser_relay_connect_btn.setStyleSheet(
                "background-color: green")
        else:
            self.laser_relay_connect_btn.setStyleSheet("background-color: red")
        if self._elliptec_controller.isOpen():
            self._elliptec_controller._connect_btn.setStyleSheet(
                "background-color: green")
        else:
            self._elliptec_controller._connect_btn.setStyleSheet(
                "background-color: red")

        if self.laserRelay.isOpen():
            if self.laserRelay_last == self.relaySettings():
                self.send_laser_relay_btn.setStyleSheet(
                    "background-color: green")
            else:
                self.send_laser_relay_btn.setStyleSheet(
                    "background-color: red")
        else:
            self.send_laser_relay_btn.setStyleSheet("")

        if self.cam_panel is not None:
            self.cam_panel.updateInfo()

        for panel in self.vimba_panels:
            panel.updateInfo()
        for panel in self.ids_panels:
            panel.updateInfo()
        for panel in self.thor_panels:
            panel.updateInfo()

    @pyqtSlot()
    def open_dialog(self, serial: QSerialPort):
        '''Opens a port config dialog for the provided serial port.

        Parameters
        ----------
        serial : QSerialPort
            the serial port to be configured.
        '''
        dialog = port_config()
        if not serial.isOpen():
            if dialog.exec_():
                portname, baudrate = dialog.get_results()
                serial.setPortName(portname)
                serial.setBaudRate(baudrate)

    @pyqtSlot(QSerialPort)
    def connectToPort(self, serial: QSerialPort):
        '''Opens the supplied serial port.

        Parameters
        ----------
        serial : QSerialPort
            the serial port to open.
        '''
        serial.open(QIODevice.ReadWrite)

    @pyqtSlot(QSerialPort)
    def disconnectFromPort(self, serial: QSerialPort):
        '''Closes the supplied serial port.

        Parameters
        ----------
        serial : QSerialPort
            the serial port to close.
        '''
        serial.close()

    @pyqtSlot()
    def start_IR(self):
        '''Starts the IR peak position acquisition and
        creates a file in the current directory.
        '''
        if (self.file is None):
            filename = None
            if filename is None:
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save IR Track Data", filter="CSV Files (*.csv)")

                if len(filename) > 0:
                    self.file = open(
                        filename, 'ab')

    @pyqtSlot()
    def stop_IR(self):
        '''Stops the IR peak position acquisition and closes the file.
        '''
        if (self.file is not None):
            self.file.close()
            self.file = None
            self.frames_saved = 0

    @pyqtSlot()
    def setIRcam(self):
        if self.cam is not None:
            QMessageBox.warning(
                self,
                "Warning",
                "Please remove {}.".format(self.cam.name),
                QMessageBox.StandardButton.Ok)
            return

        if self.IR_Cam.isOpen:
            QMessageBox.warning(
                self,
                "Warning",
                "Please disconnect {}.".format(self.IR_Cam.name),
                QMessageBox.StandardButton.Ok)
            return

        if 'TSL1401' in self.ir_cam_cbox.currentText():
            self.IR_Cam = ParallaxLineScanner()
            if self.ir_widget is not None:
                self.removeDockWidget(self.ir_widget)
                self.ir_widget.deleteLater()
            self.ir_widget = QDockWidget('IR Cam')
            self.ir_widget.setFeatures(
                QDockWidget.DockWidgetFloatable |
                QDockWidget.DockWidgetMovable)
            self.ir_widget.setWidget(
                self.IR_Cam.getQWidget())
            self.addDockWidget(
                Qt.DockWidgetArea.TopDockWidgetArea,
                self.ir_widget)
            self.tabifyDockWidget(
                self.devicesDock, self.ir_widget)
            self.focus.graph_IR.setLabel(
                "left", "Signal", "V", **self.labelStyle)

    @pyqtSlot()
    def resetIRcam(self):
        if self.IR_Cam.isOpen:
            QMessageBox.warning(
                self,
                "Warning",
                "Please disconnect {}.".format(self.IR_Cam.name),
                QMessageBox.StandardButton.Ok)
            return

        if self.ir_widget is not None:
            self.removeDockWidget(self.ir_widget)
            self.ir_widget.deleteLater()
        self.ir_widget = None
        self.IR_Cam = IR_Cam()

    @pyqtSlot()
    def setStage(self):
        if 'FOC100' in self.stage_cbox.currentText():
            if not self.stage.isOpen():
                self.stage = piezo_concept()
                if self.stage_dock is not None:
                    self.removeDockWidget(self.stage_dock)
                    self.stage_dock.deleteLater()
                self.stage_widget = self.stage.getQWidget()
                self.stage_dock = QDockWidget('Z-Stage', self)
                self.stage_dock.setFeatures(
                    QDockWidget.DockWidgetFloatable |
                    QDockWidget.DockWidgetMovable)
                self.stage_dock.setWidget(self.stage_widget)
                self.addDockWidget(
                    Qt.DockWidgetArea.TopDockWidgetArea,
                    self.stage_dock)
                self.tabifyDockWidget(
                    self.devicesDock, self.stage_dock)

    def hid_report(self, reportedEvent: Buttons):
        if type(self.stage) is piezo_concept:
            if reportedEvent == Buttons.X:
                self.stage.piezo_B_UP_btn.click()
            elif reportedEvent == Buttons.B:
                self.stage.piezo_B_DOWN_btn.click()
            elif reportedEvent == Buttons.Y:
                self.stage.piezo_S_UP_btn.click()
            elif reportedEvent == Buttons.A:
                self.stage.piezo_S_DOWN_btn.click()
            elif reportedEvent == Buttons.Options:
                self.stage.piezo_HOME_btn.click()
            elif reportedEvent == Buttons.R3:
                self.hid_controller_toggle = not self.hid_controller_toggle
                if self.hid_controller_toggle:
                    self.stage.fine_steps_slider.setStyleSheet(
                        '')
                    self.stage.pixel_slider.setStyleSheet(
                        'background-color: green')
                else:
                    self.stage.fine_steps_slider.setStyleSheet(
                        'background-color: green')
                    self.stage.pixel_slider.setStyleSheet(
                        '')

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
                    "background-color: green")
                self.kinesisXY.n_y_jump_btn.setStyleSheet(
                    "background-color: green")
                self.kinesisXY.p_x_jump_btn.setStyleSheet(
                    "background-color: green")
                self.kinesisXY.p_y_jump_btn.setStyleSheet(
                    "background-color: green")
                self.kinesisXY.n_x_step_btn.setStyleSheet(
                    "")
                self.kinesisXY.n_y_step_btn.setStyleSheet(
                    "")
                self.kinesisXY.p_x_step_btn.setStyleSheet(
                    "")
                self.kinesisXY.p_y_step_btn.setStyleSheet(
                    "")
            else:
                self.kinesisXY.n_x_jump_btn.setStyleSheet(
                    "")
                self.kinesisXY.n_y_jump_btn.setStyleSheet(
                    "")
                self.kinesisXY.p_x_jump_btn.setStyleSheet(
                    "")
                self.kinesisXY.p_y_jump_btn.setStyleSheet(
                    "")
                self.kinesisXY.n_x_step_btn.setStyleSheet(
                    "background-color: green")
                self.kinesisXY.n_y_step_btn.setStyleSheet(
                    "background-color: green")
                self.kinesisXY.p_x_step_btn.setStyleSheet(
                    "background-color: green")
                self.kinesisXY.p_y_step_btn.setStyleSheet(
                    "background-color: green")

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
            if self.stage.piezoTracking:
                val = 1e-3 * diff
                self.stage.center_pixel += val
        else:
            if abs(diff) > 0 and abs(diff) < 100:
                val = 0.25 * diff
                val += self.stage.fine_steps_slider.value()
                # val -= val % 5
                self.stage.fine_steps_slider.setValue(val)
            else:
                val = self.stage.fine_steps_slider.value()
                val -= val % 5
                self.stage.fine_steps_slider.setValue(val)

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
        # create a QApp
        app = QApplication(sys.argv)
        # set darkmode from *qdarkstyle* (not compatible with pyqt6)
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
        # sets the app icon
        dirname = os.path.dirname(os.path.abspath(__file__))
        app_icon = QIcon()
        app_icon.addFile(
            os.path.join(dirname, '../icons/16.png'), QSize(16, 16))
        app_icon.addFile(
            os.path.join(dirname, '../icons/24.png'), QSize(24, 24))
        app_icon.addFile(
            os.path.join(dirname, '../icons/32.png'), QSize(32, 32))
        app_icon.addFile(
            os.path.join(dirname, '../icons/48.png'), QSize(48, 48))
        app_icon.addFile(
            os.path.join(dirname, '../icons/64.png'), QSize(64, 64))
        app_icon.addFile(
            os.path.join(dirname, '../icons/128.png'), QSize(128, 128))
        app_icon.addFile(
            os.path.join(dirname, '../icons/256.png'), QSize(256, 256))
        app_icon.addFile(
            os.path.join(dirname, '../icons/512.png'), QSize(512, 512))

        app.setWindowIcon(app_icon)

        if sys.platform.startswith('win'):
            import ctypes
            myappid = u'samhitech.mircoEye.miEye_module'  # appid
            ctypes.windll.shell32.\
                SetCurrentProcessExplicitAppUserModelID(myappid)

        window = miEye_module()
        return app, window

    def result_scan_acquisition(self, data):
        self._scanning = False
        self.scanAcqWidget.acquire_btn.setEnabled(True)
        self.scanAcqWidget.z_acquire_btn.setEnabled(True)
        self.scanAcqWidget.z_cal_btn.setEnabled(True)

        if data:
            self.lastTile = TiledImageSelector(data)
            self.lastTile.positionSelected.connect(
                lambda x, y: self.kinesisXY.doAsync(
                    None, self.kinesisXY.move_absolute, x, y)
            )
            self.lastTile.show()

    def result_z_calibration(self, data):
        self._scanning = False
        self.scanAcqWidget.acquire_btn.setEnabled(True)
        self.scanAcqWidget.z_acquire_btn.setEnabled(True)
        self.scanAcqWidget.z_cal_btn.setEnabled(True)

        if data is not None:
            coeff = np.polyfit(data[:, 0], data[:, 1], 1)
            self.stage.coeff_pixel = coeff[0]
            plot_z_cal(data, coeff)

    def result_scan_export(self, data: list[TileImage]):
        self._scanning = False
        self.scanAcqWidget.acquire_btn.setEnabled(True)
        self.scanAcqWidget.z_acquire_btn.setEnabled(True)
        self.scanAcqWidget.z_cal_btn.setEnabled(True)

        if data:
            if len(self.scanAcqWidget._directory) > 0:
                path = self.scanAcqWidget._directory
                index = 0
                while (os.path.exists(path + '/{:03d}_XY/'.format(index))):
                    index += 1
                path = path + '/{:03d}_XY/'.format(index)
                if not os.path.exists(path):
                    os.makedirs(path)
                for idx, tImg in enumerate(data):
                    tf.imwrite(
                        path +
                        '{:03d}_image_y{:02d}_x{:02d}.tif'.format(
                            idx, tImg.index[0], tImg.index[1]
                        ),
                        tImg.uImage.image,
                        photometric='minisblack')

    def start_scan_acquisitionXY(
            self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = thread_worker(
                scan_acquisition, self,
                [params[0], params[1]],
                [params[2], params[3]],
                params[4],
                params[5], progress=False, z_stage=False)
            self.scan_worker.signals.result.connect(
                self.result_scan_acquisition)
            # Execute
            self._threadpool.start(self.scan_worker)

            self.scanAcqWidget.acquire_btn.setEnabled(False)
            self.scanAcqWidget.z_acquire_btn.setEnabled(False)
            self.scanAcqWidget.z_cal_btn.setEnabled(False)

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

            self.scanAcqWidget.acquire_btn.setEnabled(False)
            self.scanAcqWidget.z_acquire_btn.setEnabled(False)
            self.scanAcqWidget.z_cal_btn.setEnabled(False)

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

            self.scanAcqWidget.acquire_btn.setEnabled(False)
            self.scanAcqWidget.z_acquire_btn.setEnabled(False)
            self.scanAcqWidget.z_cal_btn.setEnabled(False)

    def stop_scan_acquisition(self):
        self._stop_scan = True

    def update_directories(self, value: str):
        for panel in self.vimba_panels:
            panel._directory = value
            panel.save_dir_edit.setText(value)
        for panel in self.ids_panels:
            panel._directory = value
            panel.save_dir_edit.setText(value)
        for panel in self.thor_panels:
            panel._directory = value
            panel.save_dir_edit.setText(value)

    def show_last_tile(self):
        if self.lastTile is not None:
            self.lastTile.show()


def scan_acquisition(miEye: miEye_module, steps, step_size, delay, average=1):
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
        if miEye.kinesisXY.isOpen()[0] and miEye.kinesisXY.isOpen()[1] \
                and len(miEye.vimba_cams) > 0:
            cam = miEye.vimba_cams[0]
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
                            for n in range(average):
                                frames_avg.append(
                                    cam.cam.get_frame().as_numpy_ndarray())
                            frame = uImage(
                                np.array(
                                    frames_avg).mean(
                                        axis=0, dtype=np.uint16))
                        else:
                            frame = uImage(
                                cam.cam.get_frame().as_numpy_ndarray())
                        frame.equalizeLUT(None, True)
                    frame._view = cv2.resize(
                        frame._view, (0, 0),
                        fx=0.5,
                        fy=0.5)
                    Y = (x % 2) * (steps[1] - 1) + ((-1)**x) * y
                    data.append(
                        TileImage(frame, [Y, x], miEye.kinesisXY.position))
                    cv2.imshow(cam.name, frame._view)
                    cv2.waitKey(1)

            miEye.kinesisXY.update()

    except Exception:
        traceback.print_exc()
    finally:
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
        if miEye.stage.isOpen() and len(miEye.vimba_cams) > 0 and nFrames > 0:
            cam = miEye.vimba_cams[0]
            cam_pan = miEye.vimba_panels[0]
            if cam.acquisition:
                return

            peak = miEye.stage.center_pixel
            for x in range(n):
                if x > 0:
                    if miEye.stage.piezoTracking and \
                            miEye.stage.pixel_translation.isChecked():
                        value = miEye.stage.coeff_pixel * step_size
                        if reverse:
                            value *= -1
                        miEye.stage.center_pixel += value
                        QThread.msleep(delay)
                    else:
                        if miEye.stage.piezoTracking:
                            miEye.stage.autoFocusTracking()
                        miEye.scanAcqWidget.moveZ.emit(reverse, step_size)
                        QThread.msleep(delay)
                        miEye.stage.autoFocusTracking()
                frame = None
                cam_pan.frames_tbox.setValue(nFrames)
                cam_pan.save_data_chbx.setChecked(True)
                prefix = 'Z_{:04d}_'.format(x)
                cam_pan.start_free_run(prefix)

                cam_pan.s_event.wait()
    except Exception:
        traceback.print_exc()
    finally:
        miEye.stage.center_pixel = peak
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
            if miEye.stage.piezoTracking:
                miEye.stage.autoFocusTracking()
            for x in range(n):
                if x > 0:
                    miEye.scanAcqWidget.moveZ.emit(reverse, step_size)
                QThread.msleep(delay * nFrames)
                positions[x, 0] = x * step_size
                positions[x, 1] = np.mean(miEye.centerData[-nFrames:])
    except Exception:
        traceback.print_exc()
        positions = None
    finally:
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
        "Slope: {0} | Intercept {1}".format(
            coeff[0], coeff[1])
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
            'ROI_x': mieye.focus.ROI_x.value(),
            'ROI_y': mieye.focus.ROI_y.value(),
            'ROI_length': mieye.focus.ROI_length.value(),
            'ROI_angle': mieye.focus.ROI_angle.value(),
            'LaserRelay': (mieye.laserRelay.portName(),
                           mieye.laserRelay.baudRate()),
            'Elliptic': (mieye._elliptec_controller.serial.portName(),
                         mieye._elliptec_controller.serial.baudRate()),
            'PiezoStage': (mieye.stage.serial.portName(),
                           mieye.stage.serial.baudRate(),
                           mieye.stage.pConst,
                           mieye.stage.iConst,
                           mieye.stage.dConst,
                           mieye.stage.tau,
                           mieye.stage.threshold),
            'KinesisX': (mieye.kinesisXY.X_Kinesis.serial.port,
                         mieye.kinesisXY.X_Kinesis.serial.baudrate),
            'KinesisY': (mieye.kinesisXY.Y_Kinesis.serial.port,
                         mieye.kinesisXY.Y_Kinesis.serial.baudrate)
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
    config['stage_dock'] = (
        mieye.stage_dock.isFloating(),
        (mieye.stage_dock.mapToGlobal(QPoint(0, 0)).x(),
         mieye.stage_dock.mapToGlobal(QPoint(0, 0)).y()),
        (mieye.stage_dock.geometry().width(),
         mieye.stage_dock.geometry().height()),
        mieye.stage_dock.isVisible())
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

    with open(filename, 'r') as file:
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

    if 'ROI_x' in config:
        mieye.focus.ROI_x.setValue(float(config['ROI_x']))
    if 'ROI_y' in config:
        mieye.focus.ROI_y.setValue(float(config['ROI_y']))
    if 'ROI_length' in config:
        mieye.focus.ROI_length.setValue(float(config['ROI_length']))
    if 'ROI_angle' in config:
        mieye.focus.ROI_angle.setValue(float(config['ROI_angle']))
    if 'ROI_x' in config:
        mieye.focus.ROI_x.setValue(float(config['ROI_x']))
    mieye.focus.set_roi()

    if 'LaserRelay' in config:
        mieye.laserRelay.setPortName(str(config['LaserRelay'][0]))
        mieye.laserRelay.setBaudRate(int(config['LaserRelay'][1]))
    if 'Elliptic' in config:
        mieye._elliptec_controller.serial.setPortName(
            (config['Elliptic'][0]))
        mieye._elliptec_controller.serial.setBaudRate(
            int(config['Elliptic'][1]))
    if 'PiezoStage' in config:
        mieye.stage.serial.setPortName(str(config['PiezoStage'][0]))
        mieye.stage.serial.setBaudRate(int(config['PiezoStage'][1]))
        if len(config['PiezoStage']) > 2:
            mieye.stage.pConst = float(config['PiezoStage'][2])
            mieye.stage.iConst = float(config['PiezoStage'][3])
            mieye.stage.dConst = float(config['PiezoStage'][4])
            mieye.stage.tau = float(config['PiezoStage'][5])
            mieye.stage.threshold = float(config['PiezoStage'][6])
    if 'KinesisX' in config:
        mieye.kinesisXY.X_Kinesis.serial.port = str(config['KinesisX'][0])
        mieye.kinesisXY.X_Kinesis.serial.baudrate = int(config['KinesisX'][1])
    if 'KinesisY' in config:
        mieye.kinesisXY.Y_Kinesis.serial.port = str(config['KinesisY'][0])
        mieye.kinesisXY.Y_Kinesis.serial.baudrate = int(config['KinesisY'][1])

    if 'LaserPanels' in config:
        if config['LaserPanels'] is not None:
            for panel in mieye.laserPanels:
                mieye.lasersLayout.removeWidget(panel)
                panel.Laser.CloseCOM()

            mieye.laserPanels.clear()

            for panel in config['LaserPanels']:
                if bool(panel[2]):
                    combiner = CombinerLaserWidget()
                    mieye.laserPanels.append(combiner)
                    mieye.lasersLayout.insertWidget(0, combiner)
                    combiner.Laser.setPortName(str(panel[0]))
                    combiner.Laser.setBaudRate(int(panel[1]))
                else:
                    laser = SingleLaserWidget()
                    mieye.laserPanels.append(laser)
                    mieye.lasersLayout.insertWidget(0, laser)
                    laser.Laser.setPortName(str(panel[0]))
                    laser.Laser.setBaudRate(int(panel[1]))

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
    if 'stage_dock' in config:
        mieye.stage_dock.setVisible(
            bool(config['stage_dock'][3]))
        if bool(config['stage_dock'][0]):
            mieye.stage_dock.setFloating(True)
            mieye.stage_dock.setGeometry(
                config['stage_dock'][1][0],
                config['stage_dock'][1][1],
                config['stage_dock'][2][0],
                config['stage_dock'][2][1])
        else:
            mieye.stage_dock.setFloating(False)
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
