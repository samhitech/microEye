# Libraries
import os
import sys
import traceback
from logging import exception
from math import exp
from typing import final

import cv2
import numpy as np
import qdarkstyle
import tifffile as tf
from pyqode.python.widgets import PyCodeEdit
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from .CameraListWidget import CameraListWidget
from ..pyscripting import *
from .thorlabs import *
from .thorlabs_panel import Thorlabs_Panel
from ..thread_worker import *
from .ueye_camera import IDS_Camera
from .ueye_panel import IDS_Panel
from .vimba_cam import *
from .vimba_panel import *
from .kinesis import *
from .scan_acquisition import *
from ..hid_controller import *

try:
    from pyueye import ueye
except Exception:
    ueye = None

try:
    import vimba as vb
except Exception:
    vb = None


class acquisition_module(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(acquisition_module, self).__init__(*args, **kwargs)

        # setting title
        self.setWindowTitle(
            "microEye acquisition module \
            (https://github.com/samhitech/microEye)")

        # setting geometry
        self.setGeometry(0, 0, 800, 600)

        # Statusbar time
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss,zzz")
        self.statusBar().showMessage("Time: " + current_time)

        # Cameras
        self.ids_cams: list[IDS_Camera] = []
        self.thorlabs_cams: list[thorlabs_camera] = []
        self.vimba_cams: list[vimba_cam] = []

        # Panels
        self.ids_panels: list[IDS_Panel] = []
        self.thor_panels: list[Thorlabs_Panel] = []
        self.vimba_panels: list[Vimba_Panel] = []

        # Threading
        self.threadpool = QThreadPool()
        self._mcam_acq_worker = None
        print(
            "Multithreading with maximum %d threads"
            % self.threadpool.maxThreadCount())

        self._stop_mcam_thread = True
        self._exec_time = 0.0
        self._calc = ""

        # XY Stage
        self.kinesisXY = KinesisXY(threadpool=self.threadpool)

        self.lastTile = None
        self._stop_scan = False
        self._scanning = False

        #  Layout
        self.LayoutInit()

        # Statues Bar Timer
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.recurring_timer)
        self.timer.start()

        # Statues Bar Timer
        self.timer_2 = QTimer()
        self.timer_2.setInterval(250)
        self.timer_2.timeout.connect(self.recurring_timer_2)
        self.timer_2.start()

        self.show()

        # centered
        self.center()

    def center(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def LayoutInit(self):
        # Main layout
        self.Hlayout = QHBoxLayout()

        # Tablayout
        self.tabView = QTabWidget()

        self.first_tab = QWidget()
        self.second_tab = QWidget()

        # first tab vertical layout
        self.first_tab_Layout = QFormLayout()
        # set as first tab layout
        self.first_tab.setLayout(self.first_tab_Layout)

        # second tab vertical layout
        self.second_tab_Layout = QVBoxLayout()
        # set as second tab layout
        self.second_tab.setLayout(self.second_tab_Layout)

        self.scanAcqWidget = ScanAcquisitionWidget()
        self.scanAcqWidget.startAcquisition.connect(
            self.start_scan_acquisition)
        self.scanAcqWidget.stopAcquisition.connect(self.stop_scan_acquisition)
        self.scanAcqWidget.openLastTile.connect(self.show_last_tile)

        self.hid_controller = hid_controller()
        self.hid_controller.reportEvent.connect(self.hid_report)
        self.hid_controller.reportLStickPosition.connect(
            self.hid_LStick_report)
        self.hid_controller_toggle = False

        self.second_tab_Layout.addWidget(self.hid_controller)
        self.second_tab_Layout.addWidget(self.kinesisXY.getQWidget())
        self.second_tab_Layout.addWidget(self.scanAcqWidget)

        # second tab vertical layout
        self.pyEditor = pyEditor()
        self.pyEditor.exec_btn.clicked.connect(lambda: self.exec_script())

        self.tabView.addTab(self.first_tab, "Main")
        self.tabView.addTab(self.second_tab, "Stage Controls")
        self.tabView.addTab(self.pyEditor, "Scripting")

        # CAM Table
        self.camWidget = CameraListWidget()
        self.camWidget.addCamera.connect(self.add_camera_clicked)
        self.camWidget.removeCamera.connect(self.remove_camera_clicked)

        self.start_macq = QPushButton(
            "Start Multi-Cam Acquisition",
            clicked=lambda: self.start_multi_cam_acq())
        self.start_macq.setToolTip("Trigger mode acquisition | \
        First Cam Must Be Software Triggered | \
        Second Cam Externally Triggered by the First Flash Optocoupler.")

        self.stop_macq = QPushButton(
            "Stop Acquisition",
            clicked=lambda: self.stop_multi_cam_acq())

        self.camWidget.HL_buttons.addWidget(self.start_macq, 2)
        self.camWidget.HL_buttons.addWidget(self.stop_macq, 2)

        self.acq_mode_radio = QHBoxLayout()

        self.strigger_rbox = QRadioButton("Software Triggered")
        self.strigger_rbox.setChecked(True)
        self.freerun_rbox = QRadioButton("Freerun")

        self.acq_mode_radio.addWidget(self.strigger_rbox)
        self.acq_mode_radio.addWidget(self.freerun_rbox)
        self.acq_mode_radio.addStretch()

        self.experiment_name = QLineEdit("Experiment_001")
        self.experiment_name.textChanged[str].connect(
            self.experiment_name_changed)

        self.save_dir_layout = QHBoxLayout()

        self.save_directory = os.path.dirname(os.path.realpath(__package__))
        self.save_dir_edit = QLineEdit(self.save_directory)
        self.save_dir_edit.setReadOnly(True)

        self.save_browse_btn = QPushButton("...",
                                           clicked=lambda:
                                           self.save_browse_clicked())

        self.save_dir_layout.addWidget(self.save_dir_edit)
        self.save_dir_layout.addWidget(self.save_browse_btn)

        self.frames_tbox = QLineEdit("1000")
        self.frames_tbox.textChanged[str].connect(self.frames_changed)
        self.frames_tbox.setValidator(QIntValidator())

        self.stack_to_stats = QPushButton("Convert Tiff stack to var & mean",
                                          clicked=lambda:
                                          self.stack_to_stats_clicked())

        self.first_tab_Layout.addRow(self.camWidget)
        self.first_tab_Layout.addRow(self.acq_mode_radio)
        self.first_tab_Layout.addRow(
            QLabel("Experiment:"),
            self.experiment_name)
        self.first_tab_Layout.addRow(
            QLabel("Save Directory:"),
            self.save_dir_layout)
        self.first_tab_Layout.addRow(
            QLabel("Number of frames:"),
            self.frames_tbox)
        # self.first_tab_Layout.addWidget(self.stack_to_stats)

        self.Hlayout.addWidget(self.tabView, 1)

        AllWidgets = QWidget()
        AllWidgets.setLayout(self.Hlayout)

        self.setCentralWidget(AllWidgets)

    def hid_report(self, reportedEvent: Buttons):
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
        # elif reportedEvent == Buttons.X:
        #     self.kinesisXY.n_x_jump_btn.click()
        # elif reportedEvent == Buttons.B:
        #     self.kinesisXY.p_x_jump_btn.click()
        # elif reportedEvent == Buttons.Y:
        #     self.kinesisXY.p_y_jump_btn.click()
        # elif reportedEvent == Buttons.A:
        #     self.kinesisXY.n_y_jump_btn.click()
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

    def scan_acquisition(self, steps, step_size, delay, average):
        try:
            data = []
            if self.kinesisXY.isOpen()[0] and self.kinesisXY.isOpen()[1] \
                    and len(self.vimba_cams) > 0:
                cam = self.vimba_cams[0]
                for x in range(steps[0]):
                    self.kinesisXY.move_relative(
                        round(step_size[0] / 1000, 4), 0)
                    for y in range(steps[1]):
                        if y > 0:
                            self.kinesisXY.move_relative(0, ((-1)**x) * round(
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
                            TileImage(frame, [Y, x], self.kinesisXY.position))

                self.kinesisXY.update()

        except Exception:
            traceback.print_exc()
        finally:
            return data

    def result_scan_acquisition(self, data):
        self._scanning = False
        self.scanAcqWidget.acquire_btn.setEnabled(True)

        self.lastTile = TiledImageSelector(data)
        self.lastTile.positionSelected.connect(
            lambda x, y: self.kinesisXY.doAsync(
                None, self.kinesisXY.move_absolute, x, y)
        )
        self.lastTile.show()

    def start_scan_acquisition(
            self, params):
        if not self._scanning:
            self._stop_scan = False
            self._scanning = True

            self.scan_worker = thread_worker(
                self.scan_acquisition,
                [params[0], params[1]],
                [params[2], params[3]],
                params[4],
                params[5], progress=False, z_stage=False)
            self.scan_worker.signals.result.connect(
                self.result_scan_acquisition)
            # Execute
            self.threadpool.start(self.scan_worker)

            self.scanAcqWidget.acquire_btn.setEnabled(False)

    def stop_scan_acquisition(self):
        self._stop_scan = True

    def show_last_tile(self):
        if self.lastTile is not None:
            self.lastTile.show()

    def exec_script(self):
        exec(self.pyEditor.toPlainText())

    def stack_to_stats_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Tiff Stack", filter="Tiff Files (*.tiff);;")

        if len(filename) > 0:
            self.threadpool.start(
                thread_worker(
                    self.image_stats, filename, progress=False, z_stage=False))

    def frames_changed(self, value):
        for panel in self.ids_panels:
            panel.frames_tbox.setText(value)

        for panel in self.thor_panels:
            panel.frames_tbox.setText(value)

        for panel in self.vimba_panels:
            panel.frames_tbox.setText(value)

    def experiment_name_changed(self, value):
        for panel in self.ids_panels:
            panel.experiment_name.setText(value)

        for panel in self.thor_panels:
            panel.experiment_name.setText(value)

        for panel in self.vimba_panels:
            panel.experiment_name.setText(value)

    def image_stats(self, filename):
        with tf.TiffFile(filename) as tiff_file:
            # x = np.zeros(
            #     (len(tiff_file.pages),) + tiff_file.asarray().shape,
            #     dtype=np.uint16)
            # for page in tiff_file.pages:
            #     x[page.index, :, :] = page.asarray()

            # mean = x.mean(axis=0)
            # var = x.var(axis=0)

            x = np.zeros(tiff_file.asarray().shape, dtype=np.uint32)
            x2 = np.zeros(tiff_file.asarray().shape, dtype=np.uint32)
            for page in tiff_file.pages:
                x = x + page.asarray()
                x2 = x2 + np.square(page.asarray())

                self._calc = " | Calculating {:d}/{:d}".format(
                    page.index + 1, len(tiff_file.pages))

            mean = x / len(tiff_file.pages)
            x3 = x2 / len(tiff_file.pages)
            var = x3 - np.square(mean)

            self._calc = " | Saving ... "
            # append frame to tiff
            tf.imwrite(filename.replace(".tiff", "avg.tiff"),
                       data=mean, photometric='minisblack',
                       append=True, bigtiff=True)
            tf.imwrite(filename.replace(".tiff", "var.tiff"),
                       data=var, photometric='minisblack',
                       append=True, bigtiff=True)
            self._calc = ""

    def save_browse_clicked(self):
        self.save_directory = ""

        while len(self.save_directory) == 0:
            self.save_directory = str(QFileDialog.getExistingDirectory(
                self, "Select Directory"))

        self.save_dir_edit.setText(self.save_directory)

        for panel in self.ids_panels:
            panel._directory = self.save_directory
            panel.save_dir_edit.setText(self.save_directory)

        for panel in self.thor_panels:
            panel._directory = self.save_directory
            panel.save_dir_edit.setText(self.save_directory)

        for panel in self.vimba_panels:
            panel._directory = self.save_directory
            panel.save_dir_edit.setText(self.save_directory)

    def remove_camera_clicked(self, cam):
        if not self._stop_mcam_thread:
            QMessageBox.warning(
                self, "Warning",
                "Please stop Multi-Cam acquisition.",
                QMessageBox.StandardButton.Ok)
            return

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
                        self.Hlayout.removeWidget(pan)
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
                        self.Hlayout.removeWidget(pan)
                        pan.setParent(None)
                        break
        if 'Vimba' in cam["Driver"]:
            for pan in self.vimba_panels:
                with pan.cam.cam:
                    if pan.cam.cam.get_serial() == cam["Serial"]:
                        if not pan.cam.acquisition:

                            pan._dispose_cam = True
                            pan._stop_thread = True
                            self.vimba_cams.remove(pan.cam)
                            self.vimba_panels.remove(pan)
                            self.Hlayout.removeWidget(pan)
                            pan.setParent(None)
                            break

    def add_camera_clicked(self, cam):
        if not self._stop_mcam_thread:
            QMessageBox.warning(
                self,
                "Warning",
                "Please stop Multi-Cam acquisition.",
                QMessageBox.StandardButton.Ok)
            return

        # print(cam)
        if cam["InUse"] == 0:
            if 'uEye' in cam["Driver"]:
                ids_cam = IDS_Camera(cam["camID"])
                nRet = ids_cam.initialize()
                self.ids_cams.append(ids_cam)
                ids_panel = IDS_Panel(
                    self.threadpool,
                    ids_cam, cam["Model"] + " " + cam["Serial"])
                ids_panel._directory = self.save_directory
                if len(self.ids_panels) == 0:
                    ids_panel.master = True
                else:
                    ids_panel.master = False
                ids_panel.exposureChanged.connect(
                    self.master_exposure_changed)
                self.ids_panels.append(ids_panel)
                self.Hlayout.addWidget(ids_panel, 1)
            if 'UC480' in cam["Driver"]:
                thor_cam = thorlabs_camera(cam["camID"])
                nRet = thor_cam.initialize()
                if nRet == CMD.IS_SUCCESS:
                    self.thorlabs_cams.append(thor_cam)
                    thor_panel = Thorlabs_Panel(
                        self.threadpool,
                        thor_cam, cam["Model"] + " " + cam["Serial"])
                    thor_panel._directory = self.save_directory
                    thor_panel.master = False
                    thor_panel.exposureChanged.connect(
                        self.master_exposure_changed)
                    self.thor_panels.append(thor_panel)
                    self.Hlayout.addWidget(thor_panel, 1)
            if 'Vimba' in cam["Driver"]:
                v_cam = vimba_cam(cam["camID"])
                self.vimba_cams.append(v_cam)
                v_panel = Vimba_Panel(
                        self.threadpool,
                        v_cam, cam["Model"] + " " + cam["Serial"])
                v_panel._directory = self.save_directory
                v_panel.master = False
                v_panel.exposureChanged.connect(
                    self.master_exposure_changed)
                self.vimba_panels.append(v_panel)
                self.Hlayout.addWidget(v_panel, 1)
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Device is in use or already added.",
                QMessageBox.StandardButton.Ok)

    def stop_multi_cam_acq(self):
        self._stop_mcam_thread = True

    def start_multi_cam_acq(self):

        p_count = len(self.ids_panels)

        if p_count < 2:
            QMessageBox.warning(
                self,
                "Warning",
                "Two cameras has to be added at least.",
                QMessageBox.StandardButton.Ok)
            return

        if self.ids_panels[0].cam.trigger_mode != ueye.IS_SET_TRIGGER_SOFTWARE:
            QMessageBox.warning(
                self,
                "Warning",
                "First camera has to be set to Software Trigger Mode.",
                QMessageBox.StandardButton.Ok)
            return

        for p in range(p_count - 1):
            if self.ids_panels[p + 1].cam.trigger_mode == \
               ueye.IS_SET_TRIGGER_OFF:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Other cameras has to be set to a certain Trigger Mode.",
                    QMessageBox.StandardButton.Ok)
                return

        for cam in self.ids_cams:
            if cam.acquisition:
                QMessageBox.warning(
                    self,
                    "Warning",
                    cam.name + " is in acquisiton mode.",
                    QMessageBox.StandardButton.Ok)
                return

        for cam in self.ids_cams:
            if not cam.memory_allocated:
                cam.allocate_memory()

            nRet = cam.enable_queue_mode()

            cam.refresh_info()

            cam.acquisition = True

        self._stop_mcam_thread = False
        # Pass the function to execute
        self._mcam_acq_worker = thread_worker(
            self.multi_cam_acq, ueye.IS_SUCCESS, progress=False, z_stage=False)

        # Execute
        self.threadpool.start(self._mcam_acq_worker)

        QThread.msleep(500)

        for panel in self.ids_panels:
            panel.start_dis_save_workers(0)

    def multi_cam_acq(self, nRet):
        try:
            time = QDateTime.currentDateTime()
            datetime_str = "\\" + time.toString("_yyyy_MM_dd_hhmmss")
            nFrames = int(self.frames_tbox.text())

            for panel in reversed(self.ids_panels):
                panel._buffer.queue.clear()
                panel._temps.queue.clear()
                panel._frames.queue.clear()
                panel._counter = 0

                panel._save_path = (self.save_directory + "\\"
                                    + self.experiment_name.text()
                                    + "\\" + panel.cam.name
                                    + datetime_str)

                if not panel.master:
                    panel.cam.start_live_capture()
                elif self.freerun_rbox.isChecked():
                    panel.cam.start_live_capture()
                # if not os.path.exists(panel._save_path):
                #     os.makedirs(panel._save_path)
            # Continuous image display
            while(nRet == ueye.IS_SUCCESS):
                self._exec_time = time.msecsTo(QDateTime.currentDateTime())
                time = QDateTime.currentDateTime()

                if not self.freerun_rbox.isChecked():
                    ueye.is_FreezeVideo(
                        self.ids_panels[0].cam.hCam,
                        ueye.IS_WAIT)

                # In order to display the image in an OpenCV window
                # we need to extract the data of our image memory
                for panel in self.ids_panels:
                    panel._buffer.put(panel.cam.get_data().copy())
                    panel._temps.put(panel.cam.get_temperature())
                    panel._counter = panel._counter + 1
                    if panel._counter >= nFrames:
                        self._stop_mcam_thread = True

                QThread.usleep(100)

                if self._stop_mcam_thread:
                    break
        finally:
            for panel in self.ids_panels:
                panel._stop_thread = True
                panel.cam.acquisition = False
                if panel.cam.capture_video:
                    panel.cam.stop_live_capture()
                panel.cam.free_memory()
                if panel._dispose_cam:
                    panel.cam.dispose()

    def master_exposure_changed(self):
        if len(self.ids_panels) > 1:
            master: IDS_Panel = self.ids_panels[0]
            for panel in self.ids_panels:
                if panel != master:
                    if master.cam_framerate_ledit.text() != \
                       panel.cam_framerate_ledit.text():
                        panel.cam_framerate_slider.setValue(
                            IDS_Panel.find_nearest(
                                panel.cam_framerate_slider.values,
                                float(master.cam_framerate_ledit.text())))
                    elif master.cam_exposure_ledit.text() != \
                            panel.cam_exposure_ledit.text():
                        panel.cam_exposure_slider.setValue(
                            IDS_Panel.find_nearest(
                                panel.cam_exposure_slider.values,
                                float(master.cam_exposure_ledit.text()))
                                )

    def recurring_timer_2(self):
        for cam in self.ids_cams:
            if not cam.acquisition:
                cam.get_temperature()

    def recurring_timer(self):
        exe = ""
        if not self._stop_mcam_thread:
            exe = " | Execution time (ms): " + \
                "{:.3f}".format(self._exec_time) + \
                " | FPS: " + "{:.3f}".format(1000.0/self._exec_time) + \
                " | Frames: " + str(self.ids_panels[0]._counter)

        for panel in self.ids_panels:
            panel.info_temp.setText(
                " T {:.2f} °C".format(panel.cam.temperature))
            panel.info_cap.setText(
                " Capture {:d}/{:d} {:.2%} | {:.2f} ms ".format(
                    panel._counter,
                    panel._nFrames,
                    panel._counter / panel._nFrames,
                    panel._exec_time))
            panel.info_disp.setText(
                " Display {:d} | {:.2f} ms ".format(
                    panel._buffer.qsize(), panel._dis_time))
            panel.info_save.setText(
                " Save {:d} | {:.2f} ms ".format(
                    panel._frames.qsize(), panel._save_time))
            exe = exe + " | CAM " + str(panel.cam.Cam_ID) + \
                panel.info_temp.text() + panel.info_cap.text() + \
                panel.info_disp.text() + panel.info_save.text()

        for panel in self.thor_panels:
            panel.info_temp.setText(
                " T {:.2f} °C".format(panel.cam.temperature))
            panel.info_cap.setText(
                " Capture {:d}/{:d} {:.2%} | {:.2f} ms ".format(
                    panel._counter,
                    panel._nFrames,
                    panel._counter / panel._nFrames,
                    panel._exec_time))
            panel.info_disp.setText(
                " Display {:d} | {:.2f} ms ".format(
                    panel._buffer.qsize(), panel._dis_time))
            panel.info_save.setText(
                " Save {:d} | {:.2f} ms ".format(
                    panel._frames.qsize(), panel._save_time))
            exe = exe + " | CAM " + str(panel.cam.hCam.value) + \
                panel.info_temp.text() + panel.info_cap.text() + \
                panel.info_disp.text() + panel.info_save.text()

        for panel in self.vimba_panels:
            panel.info_temp.setText(
                " T {:.2f} °C".format(panel.cam.temperature))
            panel.info_cap.setText(
                " Capture {:d}/{:d} {:.2%} | {:.2f} ms ".format(
                    panel._counter,
                    panel._nFrames,
                    panel._counter / panel._nFrames,
                    panel._exec_time))
            panel.info_disp.setText(
                " Display {:d} | {:.2f} ms ".format(
                    panel._buffer.qsize(), panel._dis_time))
            panel.info_save.setText(
                " Save {:d} | {:.2f} ms ".format(
                    panel._frames.qsize(), panel._save_time))
            exe = exe + " | CAM " + str(panel.cam.Cam_ID) + \
                panel.info_temp.text() + panel.info_cap.text() + \
                panel.info_disp.text() + panel.info_save.text()

        self.statusBar().showMessage("Time: " +
                                     QDateTime.currentDateTime()
                                     .toString("hh:mm:ss,zzz") +
                                     exe + self._calc)

    def StartGUI():
        '''Initializes a new QApplication and acquisition_module.

        Use
        -------
        app, window = acquisition_module.StartGUI()

        app.exec_()

        Returns
        -------
        tuple (QApplication, microEye.acquisition_module)
            Returns a tuple with QApp and acquisition_module main window.
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
            myappid = u'samhitech.mircoEye.acquisition_module'  # appid
            ctypes.windll.shell32.\
                SetCurrentProcessExplicitAppUserModelID(myappid)

        window = acquisition_module()
        return app, window
