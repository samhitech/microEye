import struct
import os
import sys
import time
import traceback
import ctypes

import numpy as np
import qdarkstyle
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
from pyqtgraph import PlotWidget, plot
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .io_matchbox import *
from .piezo_concept import *
from .thread_worker import *
from .port_config import *
from .laser_panel import *

from queue import Queue


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
        self.piezoConcept = piezo_concept(self, readyRead=self.rx_piezo)
        self.piezoConcept.setBaudRate(115200)
        self.piezoConcept.setPortName('COM5')
        self.piezoConcept.ZPosition = 50000
        self.piezoConcept.Received = ''
        self.PzRes = ''

        # MatchBox
        self.match_box = io_matchbox(self)  # readyRead=self.rx_mbox)
        self.match_box.DataReady.connect(self.rx_mbox)
        self.match_box.setBaudRate(115200)
        self.match_box.setPortName('COM3')

        # Serial Port IR CAM
        self.serial = QSerialPort(
            self,
            readyRead=self.receive
        )
        self.serial.ydata = None
        self.serial.setBaudRate(115200)
        self.serial.setPortName('COM4')

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
        self.ydata_buffer = Queue()
        self.ydata_buffer.put(np.array([0 for i in range(n_data)]))
        self.y_index = 0
        self.centerDataX = np.array(list(range(500)))
        self.centerData = np.ones((500,))
        self.error_buffer = np.zeros((20,))
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

        # self.peak_fit(self.worker.signals.move_stage_z, data)
        # self.update_graphs()

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

        # General settings groupbox
        Reg_GBox = QGroupBox("Control")

        # vertical layout
        Reg_Layout = QVBoxLayout()
        Reg_GBox.setLayout(Reg_Layout)

        # ALEX checkbox
        self.ALEX = QCheckBox("ALEX")
        self.ALEX.state = "ALEX"
        self.ALEX.setChecked(False)
        Reg_Layout.addWidget(self.ALEX)

        # IR CCD array arduino buttons
        self.cam_connect_btn = QPushButton(
            "IR Connect",
            clicked=lambda: self.connectToPort(self.serial)
        )
        self.cam_disconnect_btn = QPushButton(
            "IR Disconnect",
            clicked=lambda: self.disconnectFromPort(self.serial)
        )
        self.cam_config_btn = QPushButton(
            "IR CAM Config.",
            clicked=lambda: self.open_dialog(self.serial)
        )

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
            "MBox Connect",
            clicked=lambda: self.match_box.OpenCOM()
        )
        self.mbox_disconnect_btn = QPushButton(
            "MBox Disconnect",
            clicked=lambda: self.match_box.CloseCOM()
        )
        self.mbox_config_btn = QPushButton(
            "MBox Config.",
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
            "Laser Relay Connect",
            clicked=lambda: self.connectToPort(self.laserRelay)
        )
        self.laser_relay_disconnect_btn = QPushButton(
            "Laser Relay Disconnect",
            clicked=lambda: self.disconnectFromPort(self.laserRelay)
        )
        self.laser_relay_btn = QPushButton(
            "Laser Relay Config.",
            clicked=lambda: self.open_dialog(self.laserRelay)
        )
        self.send_laser_relay_btn = QPushButton(
            "Laser Relay Send",
            clicked=lambda: self.sendConfig(self.laserRelay)
        )

        # Piezostage controls
        self.piezo_connect_btn = QPushButton(
            "Piezo Concept Connect",
            clicked=lambda: self.connectToPort(self.piezoConcept)
        )
        self.piezo_disconnect_btn = QPushButton(
            "Piezo Concept Disconnect",
            clicked=lambda: self.disconnectFromPort(self.piezoConcept)
        )
        self.piezo_config_btn = QPushButton(
            "Piezo Concept Config.",
            clicked=lambda: self.open_dialog(self.piezoConcept)
        )

        self.piezo_tracking_btn = QPushButton(
            "Focus Tracking Off",
            clicked=lambda: self.autoFocusTracking()
        )
        self.center_pixel = 57
        self.piezoTracking = False

        fine_step = 25
        coarse_step = 1
        self.fine_steps_label = QLabel("Fine step " + str(fine_step) + " nm")
        self.fine_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fine_steps_slider.setMinimum(1)
        self.fine_steps_slider.setMaximum(1000)
        self.fine_steps_slider.setValue(fine_step)
        self.fine_steps_slider.setTickInterval(200)
        self.fine_steps_slider.valueChanged.connect(
            self.fine_steps_valuechange)

        self.coarse_steps_label = QLabel(
            "Coarse step " + str(coarse_step) + " um")
        self.coarse_steps_slider = QSlider(Qt.Orientation.Horizontal)
        self.coarse_steps_slider.setMinimum(1)
        self.coarse_steps_slider.setMaximum(20)
        self.coarse_steps_slider.setValue(coarse_step)
        self.coarse_steps_slider.setTickInterval(4)
        self.coarse_steps_slider.valueChanged.connect(
            self.coarse_steps_valuechange)

        self.piezo_HOME_btn = QPushButton(
            "⌂",
            clicked=lambda: self.piezoConcept.HOME()
        )
        self.piezo_REFRESH_btn = QPushButton(
            "R",
            clicked=lambda: self.piezoConcept.REFRESH()
        )
        self.piezo_B_UP_btn = QPushButton(
            "<<",
            clicked=lambda: self.piezoConcept.UP(
                self.coarse_steps_slider.value() * 1000)
        )
        self.piezo_S_UP_btn = QPushButton(
            "<",
            clicked=lambda: self.piezoConcept.UP(
                self.fine_steps_slider.value())
        )
        self.piezo_S_DOWN_btn = QPushButton(
            ">",
            clicked=lambda: self.piezoConcept.DOWN(
                self.fine_steps_slider.value())
        )
        self.piezo_B_DOWN_btn = QPushButton(
            ">>",
            clicked=lambda: self.piezoConcept.DOWN(
                self.coarse_steps_slider.value() * 1000)
        )
        self.Directions = QHBoxLayout()
        self.Directions.addWidget(self.piezo_HOME_btn)
        self.Directions.addWidget(self.piezo_REFRESH_btn)
        self.Directions.addWidget(self.piezo_B_UP_btn)
        self.Directions.addWidget(self.piezo_S_UP_btn)
        self.Directions.addWidget(self.piezo_S_DOWN_btn)
        self.Directions.addWidget(self.piezo_B_DOWN_btn)

        Reg_Layout.addWidget(self.cam_connect_btn)
        Reg_Layout.addWidget(self.cam_disconnect_btn)
        Reg_Layout.addWidget(self.cam_config_btn)
        Reg_Layout.addWidget(self.start_IR_btn)
        Reg_Layout.addWidget(self.stop_IR_btn)

        Reg_Layout.addWidget(self.mbox_connect_btn)
        Reg_Layout.addWidget(self.mbox_disconnect_btn)
        Reg_Layout.addWidget(self.mbox_config_btn)
        Reg_Layout.addWidget(self.get_set_curr_btn)
        Reg_Layout.addWidget(self.get_curr_curr_btn)

        Reg_Layout.addWidget(self.laser_relay_connect_btn)
        Reg_Layout.addWidget(self.laser_relay_disconnect_btn)
        Reg_Layout.addWidget(self.laser_relay_btn)
        Reg_Layout.addWidget(self.send_laser_relay_btn)

        Reg_Layout.addWidget(self.piezo_connect_btn)
        Reg_Layout.addWidget(self.piezo_disconnect_btn)
        Reg_Layout.addWidget(self.piezo_config_btn)
        Reg_Layout.addWidget(self.fine_steps_label)
        Reg_Layout.addWidget(self.fine_steps_slider)
        Reg_Layout.addWidget(self.coarse_steps_label)
        Reg_Layout.addWidget(self.coarse_steps_slider)
        temp = QWidget()
        temp.setLayout(self.Directions)
        Reg_Layout.addWidget(temp)
        Reg_Layout.addWidget(self.piezo_tracking_btn)

        Reg_Layout.addStretch()

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

        # Plot Canvas

        layout = QVBoxLayout()
        self.graph_IR = PlotWidget()
        self.graph_Peak = PlotWidget()
        layout.addWidget(self.graph_IR)
        layout.addWidget(self.graph_Peak)
        # layout.addWidget(toolbar)
        # layout.addWidget(self.canvas)
        # layout.addWidget(toolbar2)
        # layout.addWidget(self.canvas2)

        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QWidget()
        widget.setLayout(layout)

        # Create a placeholder widget to hold the controls
        Hlayout.addWidget(Reg_GBox, 1)
        Hlayout.addWidget(self.L1_GBox, 1)
        Hlayout.addWidget(self.L2_GBox, 1)
        Hlayout.addWidget(self.L3_GBox, 1)
        Hlayout.addWidget(self.L4_GBox, 1)
        Hlayout.addWidget(widget, 5)

        AllWidgets = QWidget()
        AllWidgets.setLayout(Hlayout)

        self.setCentralWidget(AllWidgets)

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

    def fine_steps_valuechange(self):
        '''Updates the fine step label.
        '''
        self.fine_steps_label.setText(
            "Fine step " + str(self.fine_steps_slider.value()) + " nm")

    def coarse_steps_valuechange(self):
        '''Updates the coarse step label.
        '''
        self.coarse_steps_label.setText(
            "Coarse step " + str(self.coarse_steps_slider.value()) + " um")

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
                # proceed only if the buffer is not empty
                if not self.ydata_buffer.empty():
                    # if (self.ydata != self.ydata_temp).any():
                    data: np.ndarray = self.ydata_buffer.get()
                    self._exec_time = time.msecsTo(QDateTime.currentDateTime())
                    time = QDateTime.currentDateTime()
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
            maxPeakIdx = np.argmax(peaks[1])  # highest peak
            x0 = 64 if nPeaks == 0 else peaks[0][maxPeakIdx]
            a0 = 1 if nPeaks == 0 else peaks[1]['peak_heights'][maxPeakIdx]

            # curve_fit to GaussianOffSet
            self.popt, pcov = curve_fit(
                GaussianOffSet,
                self.xdata,
                data,
                p0=[a0, x0, 1, 0])
            self.centerData = np.roll(self.centerData, -1)
            self.centerData[-1] = self.popt[1]

            if self.piezoTracking:
                avg = np.average(self.centerData[-1] - self.center_pixel)
                self.error_buffer = np.roll(self.error_buffer, -1)
                self.error_buffer[-1] = avg
                # step = max(
                #     1, abs(int(avg*self.fine_steps_slider.value()/10)))
                step = max(1, abs(int(avg*26.45)))
                # step = max(1, abs(int(avg*(90 + 108*0.05))))
                # step = max(
                #     1,
                #     abs(int(avg*(120 + 240*0.05) +
                #         15*(avg - self.error_buffer[-2]))))
                if abs(avg) > 0.016:
                    if avg > 0:
                        movez_callback.emit(True, step)
                    else:
                        movez_callback.emit(False, step)
                # if avg > 2:
                #     print(avg)
                #     movez_callback.emit(True, 25)
                # if avg > 1:
                #     print(avg)
                #     movez_callback.emit(True, 10)
                # elif avg > 0.1:
                #     print(avg)
                #     movez_callback.emit(True, 5)
                # elif avg > 0.03:
                #     print(avg)
                #     movez_callback.emit(True, 2)
                # elif avg > 0.01:
                #     print(avg)
                #     movez_callback.emit(True, 1)
                # elif avg < -2:
                #     print(avg)
                #     movez_callback.emit(False, 25)
                # elif avg < -1:
                #     print(avg)
                #     movez_callback.emit(False, 10)
                # elif avg < -0.1:
                #     print(avg)
                #     movez_callback.emit(False, 5)
                # elif avg < -0.03:
                #     print(avg)
                #     movez_callback.emit(False, 2)
                # elif avg < -0.01:
                #     print(avg)
                #     movez_callback.emit(False, 1)

        except Exception as e:
            print('Failed Gauss. fit: ' + str(e))

    def movez_stage(self, up: bool, step: int):
        print(up, step)
        if not up:
            self.piezoConcept.UP(step)
        else:
            self.piezoConcept.DOWN(step)

    def update_graphs(self, data):
        '''Updates the graphs.
        '''
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
        IR = ("    |  IR " +
              ('connected' if self.serial.isOpen() else 'disconnected'))
        MBox = ("    |  MBox " +
                ('connected' if self.match_box.isOpen() else 'disconnected'))
        MBox_RESPONSE = ''
        # if self.match_box.isOpen() and len(self.match_box.RESPONSE) > 0:
        #     MBox_RESPONSE = "    | " + self.match_box.RESPONSE[-1]

        RelayBox = ("    |  Relay " + ('connected' if self.laserRelay.isOpen()
                    else 'disconnected'))
        Piezo = ("    |  Piezo " + ('connected' if self.piezoConcept.isOpen()
                 else 'disconnected'))

        Position = ''
        if self.piezoConcept.isOpen():
            # self.piezoConcept.GETZ()
            Position = "    |  Position " + self.piezoConcept.Received
        Frames = "    | Frames Saved: " + str(self.frames_saved)

        Worker = "    | Execution time: {:d}".format(self._exec_time)
        self.statusBar().showMessage(
            "Time: " + QDateTime.currentDateTime().toString("hh:mm:ss,zzz")
            + IR + MBox + MBox_RESPONSE + RelayBox
            + Piezo + Position + Frames + Worker)

        # update indicators
        if self.serial.isOpen():
            self.cam_connect_btn.setStyleSheet("background-color: green")
        else:
            self.cam_connect_btn.setStyleSheet("background-color: red")
        if self.match_box.isOpen():
            self.mbox_connect_btn.setStyleSheet("background-color: green")
        else:
            self.mbox_connect_btn.setStyleSheet("background-color: red")
        if self.piezoConcept.isOpen():
            self.piezo_connect_btn.setStyleSheet("background-color: green")
        else:
            self.piezo_connect_btn.setStyleSheet("background-color: red")
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

    @pyqtSlot()
    def receive(self):
        '''IR CCD array serial port data ready signal
        '''
        if self.serial.bytesAvailable() >= 260:
            barray = self.serial.read(260)
            temp = np.array((np.array(struct.unpack(
                'h'*(len(barray)//2), barray)) * 5.0 / 1023.0))
            if (temp[0] != 0 or temp[-1] != 0) and \
               self.serial.bytesAvailable() >= 2:
                self.serial.read(2)
            self.ydata_buffer.put(temp[1:129])

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

    @pyqtSlot()
    def rx_piezo(self):
        '''PiezoConcept stage dataReady signal.
        '''
        self.piezoConcept.Received = str(
            self.piezoConcept.readAll(),
            encoding='utf8')
        if self.piezoConcept.LastCmd != "GETZ":
            self.piezoConcept.GETZ()

    def autoFocusTracking(self):
        '''Toggles autofocus tracking option.
        '''
        if self.piezoTracking:
            self.piezoTracking = False
            self.piezo_tracking_btn.setText("Focus Tracking Off")
        else:
            self.piezoTracking = True
            self.piezo_tracking_btn.setText("Focus Tracking On")
            self.center_pixel = self.centerData[-1]

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
