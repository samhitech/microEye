import ctypes
import os
import struct
import sys
import time
import warnings
import traceback
import subprocess
from queue import Queue

from lmfit import Model

import numpy as np
import pyqtgraph
from pyqtgraph.graphicsItems.ROI import Handle
from pyqtgraph.widgets.PlotWidget import PlotWidget
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView
import qdarkstyle
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
# from pyqtgraph import *
from scipy.optimize import curve_fit
from scipy.optimize.optimize import OptimizeWarning
from scipy.signal import find_peaks

from pyqode.python.widgets import PyCodeEdit
from pyqode.python import panels as pypanels
from pyqode.core import api, modes, panels

from .io_matchbox import *
from .IR_Cam import *
from .laser_panel import *
from .piezo_concept import *
from .port_config import *
from .thread_worker import *
from .CameraListWidget import *
from .qlist_slider import *
from .pyscripting import *


warnings.filterwarnings("ignore", category=OptimizeWarning)


class control_module(QMainWindow):
    '''The main GUI for laser control, autofocus IR tracking and
    Piezostage control.

    Works with Integrated Optics MatchBox Combiner,
    PiezoConcept FOC100 controller,
    and our Arduino RelayBox and Arduino CCD camera driver.
    | Inherits QMainWindow
    '''

    def __init__(self, *args, **kwargs):
        super(control_module, self).__init__(*args, **kwargs)

        # setting title
        self.setWindowTitle(
            "microEye control module \
            (https://github.com/samhitech/microEye)")

        # setting geometry
        self.setGeometry(0, 0, 1200, 600)

        # centered
        self.center()

        # Statusbar time
        self.statusBar().showMessage(
            "Time: " + QDateTime.currentDateTime().toString("hh:mm:ss,zzz"))

        # PiezoConcept
        self.stage = stage()

        # MatchBox
        self.match_box = io_matchbox(self)  # readyRead=self.rx_mbox)
        self.match_box.DataReady.connect(self.rx_mbox)
        self.match_box.setBaudRate(115200)
        self.match_box.setPortName('COM3')

        # Serial Port IR CCD array
        self.IR_Cam = IR_Cam()

        # Camera
        self.cam = None
        self.cam_panel = None

        # Serial Port Laser Relay
        self.laserRelay = QSerialPort(
            self)
        self.laserRelay.setBaudRate(115200)
        self.laserRelay.setPortName('COM6')
        self.laserRelay_last = ""
        self.laserRelay_curr = ""

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
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.update_gui)
        self.timer.start()

        # Threading
        self._threadpool = QThreadPool()
        print("Multithreading with maximum %d threads"
              % self._threadpool.maxThreadCount())

        # Any other args, kwargs are passed to the run function
        self.worker = thread_worker(self.worker_function)
        self.worker.signals.progress.connect(self.update_graphs)
        self.worker.signals.move_stage_z.connect(self.movez_stage)
        # Execute
        self._threadpool.start(self.worker)

        self.show()

    def center(self):
        '''Centers the window within the screen.
        '''
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def LayoutInit(self):
        '''Initializes the window layout
        '''
        Hlayout = QHBoxLayout()
        self.VL_layout = QGridLayout()

        self.Tab_Widget = QTabWidget()

        # General settings groupbox
        Reg_GBox = QGroupBox("Devices")
        self.VL_layout.addWidget(Reg_GBox, 0, 0)

        # vertical layout
        Reg_Layout = QVBoxLayout()
        Reg_GBox.setLayout(Reg_Layout)

        # IR Cam combobox
        ir_cam_layout = QHBoxLayout()
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
        ir_cam_layout.addWidget(self.ir_cam_cbox, 3)
        ir_cam_layout.addWidget(self.ir_cam_set_btn, 1)
        ir_cam_layout.addWidget(self.ir_cam_reset_btn, 1)
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

        # IO MatchBox controls
        self.mbox_connect_btn = QPushButton(
            "Connect",
            clicked=lambda: self.match_box.OpenCOM()
        )
        self.mbox_disconnect_btn = QPushButton(
            "Disconnect",
            clicked=lambda: self.match_box.CloseCOM()
        )
        self.mbox_config_btn = QPushButton(
            "Config.",
            clicked=lambda: self.open_dialog(self.match_box)
        )
        self.get_set_curr_btn = QPushButton(
            "Get Set Current",
            clicked=lambda: self.match_box.SendCommand(io_matchbox.CUR_SET)
        )
        self.get_curr_curr_btn = QPushButton(
            "Get Current",
            clicked=lambda: self.match_box.SendCommand(io_matchbox.CUR_CUR)
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
        stage_layout = QHBoxLayout()
        self.stage_cbox = QComboBox()
        self.stage_cbox.addItem('PiezoConcept FOC100')
        self.stage_set_btn = QPushButton(
            "Set",
            clicked=self.setStage
        )
        stage_layout.addWidget(self.stage_cbox, 3)
        stage_layout.addWidget(self.stage_set_btn, 1)
        self.stage_widget = None

        Reg_Layout.addWidget(QLabel('IR Camera:'))
        Reg_Layout.addLayout(ir_cam_layout)
        Reg_Layout.addWidget(
            DragLabel('Stage:', parent_name='self.stage_set_btn'))
        Reg_Layout.addLayout(stage_layout)

        Reg_Layout.addWidget(self.start_IR_btn)
        Reg_Layout.addWidget(self.stop_IR_btn)

        mbox_btns_0 = QHBoxLayout()
        mbox_btns_1 = QHBoxLayout()
        mbox_btns_0.addWidget(self.mbox_connect_btn)
        mbox_btns_0.addWidget(self.mbox_disconnect_btn)
        mbox_btns_0.addWidget(self.mbox_config_btn)
        mbox_btns_1.addWidget(self.get_set_curr_btn)
        mbox_btns_1.addWidget(self.get_curr_curr_btn)
        Reg_Layout.addWidget(QLabel('IO MatchBox:'))
        Reg_Layout.addLayout(mbox_btns_0)
        Reg_Layout.addLayout(mbox_btns_1)

        relay_btns_0 = QHBoxLayout()
        relay_btns_1 = QHBoxLayout()
        relay_btns_0.addWidget(self.laser_relay_connect_btn)
        relay_btns_0.addWidget(self.laser_relay_disconnect_btn)
        relay_btns_0.addWidget(self.laser_relay_btn)
        relay_btns_1.addWidget(self.ALEX)
        relay_btns_1.addWidget(self.send_laser_relay_btn, 1)
        Reg_Layout.addWidget(QLabel('Laser Relay:'))
        Reg_Layout.addLayout(relay_btns_0)
        Reg_Layout.addLayout(relay_btns_1)

        Reg_Layout.addStretch()

        # Lasers Tab
        lasersLayout = QGridLayout()
        lasers_tab = QWidget()
        lasers_tab.setLayout(lasersLayout)
        # Laser #1 405nm panel
        self.L1_GBox = laser_panel(4, 405,
                                   self.match_box, 280, "Laser #1 (405nm)")

        # Laser #2 488nm panel
        self.L2_GBox = laser_panel(3, 488,
                                   self.match_box, 105, "Laser #2 (488nm)")

        # Laser #3 520nm panel
        self.L3_GBox = laser_panel(2, 520,
                                   self.match_box, 200, "Laser #3 (520nm)")

        # Laser #4 638nm panel
        self.L4_GBox = laser_panel(1, 638,
                                   self.match_box, 280, "Laser #4 (638nm)")

        lasersLayout.addWidget(self.L1_GBox, 0, 0)
        lasersLayout.addWidget(self.L2_GBox, 1, 0)
        lasersLayout.addWidget(self.L3_GBox, 0, 1)
        lasersLayout.addWidget(self.L4_GBox, 1, 1)

        # cameras tab
        self.camListWidget = CameraListWidget()
        self.camListWidget.addCamera.connect(self.add_camera)
        self.camListWidget.removeCamera.connect(self.remove_camera)

        # display tab
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        graphs_layout = QGridLayout()
        graphs_tab = QWidget()
        graphs_tab.setLayout(graphs_layout)
        self.graph_IR = PlotWidget()
        self.graph_IR.setLabel("bottom", "Pixel", **self.labelStyle)
        self.graph_IR.setLabel("left", "Signal", "V", **self.labelStyle)
        self.graph_Peak = PlotWidget()
        self.graph_Peak.setLabel("bottom", "Frame", **self.labelStyle)
        self.graph_Peak.setLabel("left", "Center Pixel", **self.labelStyle)
        self.remote_view = RemoteGraphicsView()
        self.remote_view.pg.setConfigOptions(
            antialias=True, imageAxisOrder='row-major')
        pyqtgraph.setConfigOption('imageAxisOrder', 'row-major')
        self.remote_plt = self.remote_view.pg.ViewBox(invertY=True)
        self.remote_plt._setProxyOptions(deferGetattr=True)
        self.remote_view.setCentralItem(self.remote_plt)
        self.remote_plt.setAspectLocked()
        self.remote_img = self.remote_view.pg.ImageItem(axisOrder='row-major')
        self.roi_handles = [None, None]
        self.roi = self.remote_view.pg.LineSegmentROI(
            [[10, 256], [138, 256]],
            pen='r', handles=self.roi_handles)
        self.remote_plt.addItem(self.remote_img)
        self.remote_plt.addItem(self.roi)

        graphs_layout.addWidget(self.remote_view, 0, 0, 2, 1)
        graphs_layout.addWidget(self.graph_IR, 0, 1)
        graphs_layout.addWidget(self.graph_Peak, 1, 1)

        app = QApplication.instance()
        app.aboutToQuit.connect(self.remote_view.close)
        # Tabs
        self.Tab_Widget.addTab(lasers_tab, 'Lasers')
        self.Tab_Widget.addTab(self.camListWidget, 'Cameras List')
        self.Tab_Widget.addTab(graphs_tab, 'Graphs')

        self.pyEditor = pyEditor()
        self.pyEditor.exec_btn.clicked.connect(lambda: self.scriptTest())
        self.Tab_Widget.addTab(self.pyEditor, 'PyScript')

        # Create a placeholder widget to hold the controls
        Hlayout.addLayout(self.VL_layout, 1)
        Hlayout.addWidget(self.Tab_Widget, 3)

        AllWidgets = QWidget()
        AllWidgets.setLayout(Hlayout)

        self.setCentralWidget(AllWidgets)

    def scriptTest(self):
        exec(self.pyEditor.toPlainText())

    def add_camera(self, cam):
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
                    ids_cam, cam["Model"] + " " + cam["Serial"],
                    mini=True)
                # ids_panel._directory = self.save_directory
                ids_panel.master = False
                self.cam_panel = ids_panel
                self.Tab_Widget.addTab(ids_panel, ids_cam.name)
            if 'UC480' in cam["Driver"]:
                thor_cam = thorlabs_camera(cam["camID"])
                nRet = thor_cam.initialize()
                if nRet == CMD.IS_SUCCESS:
                    self.cam = thor_cam
                    thor_panel = Thorlabs_Panel(
                        self._threadpool,
                        thor_cam, cam["Model"] + " " + cam["Serial"],
                        mini=True)
                    # thor_panel._directory = self.save_directory
                    thor_panel.master = False
                    self.cam_panel = thor_panel
                    self.Tab_Widget.addTab(thor_panel, thor_cam.name)
            self.graph_IR.setLabel(
                "left", "Signal", "", **self.labelStyle)
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Device is in use or already added.",
                QMessageBox.StandardButton.Ok)

    def remove_camera(self, cam):
        if self.cam.cInfo.SerNo.decode('utf-8') == cam["Serial"]:
            if not self.cam.acquisition:
                self.cam.free_memory()
                self.cam.dispose()

            self.cam_panel._dispose_cam = True
            self.cam_panel._stop_thread = True
            idx = self.Tab_Widget.indexOf(self.cam_panel)
            self.Tab_Widget.removeTab(idx)
            self.cam = None
            self.cam_panel = None

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

    def Buffer(self) -> Queue:
        if self.cam is not None:
            return self.cam_panel.buffer
        elif not self.IR_Cam.isDummy():
            return self.IR_Cam.buffer
        else:
            return Queue()

    def relaySettings(self):
        '''Returns the RelayBox setting command.

        Returns
        -------
        str
            the RelayBox setting command.
        '''
        return "" + self.L1_GBox.L_button_group.checkedButton().state \
                  + self.L2_GBox.L_button_group.checkedButton().state \
                  + self.L3_GBox.L_button_group.checkedButton().state \
                  + self.L4_GBox.L_button_group.checkedButton().state \
                  + ("ALEXON" if self.ALEX.isChecked() else "ALEXOFF") + "\r"

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

    def worker_function(self, progress_callback, movez_callback):
        '''A worker function running in the threadpool.

        Handles the IR peak fitting and piezo autofocus tracking.
        '''
        counter = 0
        self._exec_time = 0
        time = QDateTime.currentDateTime()
        QThread.sleep(1)
        while(self.isVisible()):
            try:
                # dt = Gaussian(
                #     np.array(range(512)), 255, np.random.normal() + 256, 50)
                # x, y = np.meshgrid(dt, dt)
                # dt = x * y
                # self.remote_img.setImage(dt)
                # ax, pos = self.roi.getArrayRegion(
                #     dt, self.remote_img, returnMappedCoords=True)
                # self.IR_Cam._buffer.put(ax)
                # proceed only if the buffer is not empty
                if not self.isEmpty():
                    self._exec_time = time.msecsTo(QDateTime.currentDateTime())
                    time = QDateTime.currentDateTime()

                    data = self.Buffer().get()

                    if self.isImage():
                        self.remote_img.setImage(data, _callSync='off')
                        data, _ = self.roi.getArrayRegion(
                            data, self.remote_img,
                            axes=(1, 0), returnMappedCoords=True)
                    # self.ydata_temp = self.ydata
                    self.peak_fit(movez_callback, data.copy())
                    if(self.file is not None):
                        np.savetxt(self.file,
                                   np.concatenate(
                                    (data, self.popt),
                                    axis=0)
                                   .reshape((1, 131)), delimiter=";")
                        self.frames_saved = 1 + self.frames_saved
                    counter = counter + 1
                    progress_callback.emit(data)
                QThread.msleep(5)
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
            self._plot_ref = self.graph_IR.plot(self.xdata, data)
        else:
            # use the plot reference to update the data for that line.
            self._plot_ref.setData(self.xdata, data)

        # updates the IR graph with data fit
        if self._plotfit_ref is None:
            # create plot reference when None
            self._plotfit_ref = self.graph_IR.plot(
                self.xdata,
                GaussianOffSet(self.xdata, *self.popt))
        else:
            # use the plot reference to update the data for that line.
            self._plotfit_ref.setData(
                self.xdata,
                GaussianOffSet(self.xdata, *self.popt))

        if self._center_ref is None:
            # create plot reference when None
            self._center_ref = self.graph_Peak.plot(
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
        MBox = ("    |  MBox " +
                ('connected' if self.match_box.isOpen() else 'disconnected'))
        MBox_RESPONSE = ''
        # if self.match_box.isOpen() and len(self.match_box.RESPONSE) > 0:
        #     MBox_RESPONSE = "    | " + self.match_box.RESPONSE[-1]

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
            Worker += "    | Frames Buffer: {:d}".format(self.Buffer().qsize())
        self.statusBar().showMessage(
            "Time: " + QDateTime.currentDateTime().toString("hh:mm:ss,zzz")
            + IR + MBox + MBox_RESPONSE + RelayBox
            + Piezo + Position + Frames + Worker)

        # update indicators
        if self.IR_Cam.isOpen:
            self.IR_Cam._connect_btn.setStyleSheet("background-color: green")
        else:
            self.IR_Cam._connect_btn.setStyleSheet("background-color: red")
        if self.match_box.isOpen():
            self.mbox_connect_btn.setStyleSheet("background-color: green")
        else:
            self.mbox_connect_btn.setStyleSheet("background-color: red")
        if self.stage.isOpen():
            self.stage._connect_btn.setStyleSheet("background-color: green")
        else:
            self.stage._connect_btn.setStyleSheet("background-color: red")
        if self.laserRelay.isOpen():
            self.laser_relay_connect_btn.setStyleSheet(
                "background-color: green")
        else:
            self.laser_relay_connect_btn.setStyleSheet("background-color: red")

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
            self.cam_panel.info_temp.setText(
                " T {:.2f} °C".format(self.cam_panel.cam.temperature))
            self.cam_panel.info_cap.setText(
                " Capture {:d} | {:.2f} ms ".format(
                    self.cam_panel._counter,
                    self.cam_panel._exec_time))
            self.cam_panel.info_disp.setText(
                " Display {:d} | {:.2f} ms ".format(
                    self.cam_panel._buffer.qsize(), self.cam_panel._dis_time))
            self.cam_panel.info_save.setText(
                " Save {:d} | {:.2f} ms ".format(
                    self.cam_panel._frames.qsize(), self.cam_panel._save_time))

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

    @pyqtSlot(str, bytes)
    def rx_mbox(self, res, cmd):
        '''IO MatchBox laser combiner DataReady signal,
        emitted after recieving a command's response.

        Parameters
        ----------
        res : str
            response from the combiner.
        cmd : bytes
            command sent to the combiner.
        '''
        print('Event ', res, cmd)

        # sets the GUI controls to match the laser enabled/disabled status
        if cmd == io_matchbox.STATUS:
            res = res.strip("<").strip(">").split()
            if len(res) >= 4:
                if bool(int(res[0])):
                    self.L4_GBox.ON.setChecked(True)
                else:
                    self.L4_GBox.ON.setChecked(False)
                if bool(int(res[1])):
                    self.L3_GBox.ON.setChecked(True)
                else:
                    self.L3_GBox.ON.setChecked(False)
                if bool(int(res[2])):
                    self.L2_GBox.ON.setChecked(True)
                else:
                    self.L2_GBox.ON.setChecked(False)
                if bool(int(res[3])):
                    self.L1_GBox.ON.setChecked(True)
                else:
                    self.L1_GBox.ON.setChecked(False)
        # sets the GUI controls to match the laser set currents
        elif cmd == io_matchbox.CUR_SET:
            res = res.strip("<").strip(">").split()
            if len(res) > 0:
                if res[0] != 'ERR':
                    self.match_box.Setting = res
                    self.L1_GBox.L_cur_slider.setValue(
                        int(self.match_box.Setting[3]))
                    self.L2_GBox.L_cur_slider.setValue(
                        int(self.match_box.Setting[2]))
                    self.L3_GBox.L_cur_slider.setValue(
                        int(self.match_box.Setting[1]))
                    self.L4_GBox.L_cur_slider.setValue(
                        int(self.match_box.Setting[0]))
                else:
                    self.match_box.SendCommand(io_matchbox.CUR_SET)
        # updates the GUI to show the laser current readings
        elif cmd == io_matchbox.CUR_CUR:
            res = res.strip("<").strip(">").strip('mA').split()
            if res[0] != 'ERR':
                self.match_box.Current = res

                self.L1_GBox.L_cur_label.setText(
                    "Current " + str(self.match_box.Current[3]) + " mA")
                self.L2_GBox.L_cur_label.setText(
                    "Current " + str(self.match_box.Current[2]) + " mA")
                self.L3_GBox.L_cur_label.setText(
                    "Current " + str(self.match_box.Current[1]) + " mA")
                self.L4_GBox.L_cur_label.setText(
                    "Current " + str(self.match_box.Current[0]) + " mA")
            else:
                self.match_box.SendCommand(io_matchbox.CUR_CUR)

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
        if(self.file is None):
            self.file = open(
                'Data_' + time.strftime("%Y_%m_%d_%H%M%S") + '.csv', 'ab')

    @pyqtSlot()
    def stop_IR(self):
        '''Stops the IR peak position acquisition and closes the file.
        '''
        if(self.file is not None):
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
                self.VL_layout.removeWidget(self.ir_widget)
            self.ir_widget = self.IR_Cam.getQWidget()
            self.VL_layout.addWidget(self.ir_widget, 1, 0)
            self.graph_IR.setLabel(
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
            self.VL_layout.removeWidget(self.ir_widget)
        self.ir_widget.deleteLater()
        self.ir_widget = None
        self.IR_Cam = IR_Cam()

    @pyqtSlot()
    def setStage(self):
        if 'FOC100' in self.stage_cbox.currentText():
            if not self.stage.isOpen():
                self.stage = piezo_concept()
                if self.stage_widget is not None:
                    self.VL_layout.removeWidget(self.stage_widget)
                self.stage_widget = self.stage.getQWidget()
                self.VL_layout.addWidget(self.stage_widget, 2, 0)

    def StartGUI():
        '''Initializes a new QApplication and control_module.

        Use
        -------
        app, window = control_module.StartGUI()

        app.exec_()

        Returns
        -------
        tuple (QApplication, microEye.control_module)
            Returns a tuple with QApp and control_module main window.
        '''
        # create a QApp
        app = QApplication(sys.argv)
        # set darkmode from *qdarkstyle* (not compatible with pyqt6)
        app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
        # sets the app icon
        dirname = os.path.dirname(__file__)
        app_icon = QIcon()
        app_icon.addFile(
            os.path.join(dirname, 'icons/16.png'), QSize(16, 16))
        app_icon.addFile(
            os.path.join(dirname, 'icons/24.png'), QSize(24, 24))
        app_icon.addFile(
            os.path.join(dirname, 'icons/32.png'), QSize(32, 32))
        app_icon.addFile(
            os.path.join(dirname, 'icons/48.png'), QSize(48, 48))
        app_icon.addFile(
            os.path.join(dirname, 'icons/64.png'), QSize(64, 64))
        app_icon.addFile(
            os.path.join(dirname, 'icons/128.png'), QSize(128, 128))
        app_icon.addFile(
            os.path.join(dirname, 'icons/256.png'), QSize(256, 256))
        app_icon.addFile(
            os.path.join(dirname, 'icons/512.png'), QSize(512, 512))

        app.setWindowIcon(app_icon)

        myappid = u'samhitech.mircoEye.control_module'  # appid
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        window = control_module()
        return app, window
