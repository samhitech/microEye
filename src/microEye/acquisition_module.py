# Libraries
import ctypes
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
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyueye import ueye

from .thread_worker import *
from .ueye_camera import IDS_Camera
from .ueye_panel import IDS_Panel


class acquisition_module(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(acquisition_module, self).__init__(*args, **kwargs)

        # setting title
        self.setWindowTitle(
            "microEye acquisition module \
            (https://github.com/samhitech/microEye)")

        # setting geometry
        self.setGeometry(0, 0, 800, 600)

        # centered
        self.center()

        # Statusbar time
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss,zzz")
        self.statusBar().showMessage("Time: " + current_time)

        # Cameras
        self.ids_cams: list[IDS_Camera] = []

        # Panels
        self.ids_panels: list[IDS_Panel] = []

        # Threading
        self.threadpool = QThreadPool()
        self._mcam_acq_worker = None
        print(
            "Multithreading with maximum %d threads"
            % self.threadpool.maxThreadCount())

        self._stop_mcam_thread = True
        self._exec_time = 0.0
        self._calc = ""

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

    def center(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def LayoutInit(self):
        self.Hlayout = QHBoxLayout()
        self.Vlayout = QVBoxLayout()

        # CAM Table
        self.cam_list = IDS_Camera.get_camera_list()

        if self.cam_list is None:
            self.cam_list = []

        self.ids_model = QStandardItemModel(len(self.cam_list), 7)

        self.ids_model.setHorizontalHeaderLabels(
            ["In Use", "Camera ID", "Device ID",
             "Model", "Serial", "Status", "Sensor ID"])

        for i in range(len(self.cam_list)):
            self.ids_model.setItem(
                i, 0,
                QStandardItem(str(self.cam_list[i]["InUse"])))
            self.ids_model.setItem(
                i, 1,
                QStandardItem(str(self.cam_list[i]["camID"])))
            self.ids_model.setItem(
                i, 2,
                QStandardItem(str(self.cam_list[i]["devID"])))
            self.ids_model.setItem(
                i, 3,
                QStandardItem(self.cam_list[i]["Model"]))
            self.ids_model.setItem(
                i, 4,
                QStandardItem(self.cam_list[i]["Serial"]))
            self.ids_model.setItem(
                i, 5,
                QStandardItem(str(self.cam_list[i]["Status"])))
            self.ids_model.setItem(
                i, 6,
                QStandardItem(str(self.cam_list[i]["senID"])))

        self.ids_table = QTableView()
        self.ids_table.setModel(self.ids_model)
        self.ids_table.clearSelection()
        self.ids_table.horizontalHeader().setStretchLastSection(True)
        self.ids_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)
        self.ids_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.ids_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)

        self.ids_buttons = QHBoxLayout()

        self.ids_add_cam = QPushButton("Add Camera",
                                       clicked=lambda:
                                       self.add_camera_clicked())

        self.ids_remove_cam = QPushButton("Remove Camera",
                                          clicked=lambda:
                                          self.remove_camera_clicked())

        self.ids_refresh = QPushButton("Refresh List",
                                       clicked=lambda:
                                       self.refresh_list_clicked())

        self.ids_start_macq = QPushButton("Start Multi-Cam Acquisition",
                                          clicked=lambda:
                                          self.start_multi_cam_acq())
        self.ids_start_macq.setToolTip("Trigger mode acquisition | \
        First Cam Must Be Software Triggered | \
        Second Cam Externally Triggered by the First Flash Optocoupler.")

        self.ids_stop_macq = QPushButton("Stop Acquisition",
                                         clicked=lambda:
                                         self.stop_multi_cam_acq())

        self.ids_buttons.addWidget(self.ids_add_cam)
        self.ids_buttons.addWidget(self.ids_remove_cam)
        self.ids_buttons.addWidget(self.ids_refresh)
        self.ids_buttons.addWidget(self.ids_start_macq, 2)
        self.ids_buttons.addWidget(self.ids_stop_macq, 2)

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

        self.save_directory = os.path.dirname(os.path.realpath(__file__))
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

        self.Vlayout.addWidget(self.ids_table)
        self.Vlayout.addLayout(self.ids_buttons)
        self.Vlayout.addLayout(self.acq_mode_radio)
        self.Vlayout.addWidget(QLabel("Experiment:"))
        self.Vlayout.addWidget(self.experiment_name)
        self.Vlayout.addWidget(QLabel("Save Directory:"))
        self.Vlayout.addLayout(self.save_dir_layout)
        self.Vlayout.addWidget(QLabel("Number of frames:"))
        self.Vlayout.addWidget(self.frames_tbox)
        self.Vlayout.addWidget(self.stack_to_stats)
        self.Vlayout.addStretch()

        self.Hlayout.addLayout(self.Vlayout, 1)

        AllWidgets = QWidget()
        AllWidgets.setLayout(self.Hlayout)

        self.setCentralWidget(AllWidgets)

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

    def experiment_name_changed(self, value):
        for panel in self.ids_panels:
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

    def remove_camera_clicked(self):
        if not self._stop_mcam_thread:
            QMessageBox.warning(
                self, "Warning",
                "Please stop Multi-Cam acquisition.",
                QMessageBox.StandardButton.Ok)
            return

        if len(self.ids_table.selectedIndexes()) > 0:
            cam = self.cam_list[self.ids_table.currentIndex().row()]

            for pan in self.ids_panels:
                if pan.cam.Cam_ID == cam["camID"]:
                    if not pan.cam.acquisition:
                        pan.cam.dispose()
                    # if not pan.master:
                    #     self.ids_panels[0].slaves.remove(pan.cam)
                    pan._dispose_cam = True
                    pan._stop_thread = True
                    self.ids_cams.remove(pan.cam)
                    self.ids_panels.remove(pan)
                    self.Hlayout.removeWidget(pan)
                    pan.setParent(None)
                    self.refresh_list_clicked()
                    break
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Please select a device.",
                QMessageBox.StandardButton.Ok)

    def add_camera_clicked(self):
        if not self._stop_mcam_thread:
            QMessageBox.warning(
                self,
                "Warning",
                "Please stop Multi-Cam acquisition.",
                QMessageBox.StandardButton.Ok)
            return

        if len(self.ids_table.selectedIndexes()) > 0:
            cam = self.cam_list[self.ids_table.currentIndex().row()]
            # print(cam)
            if cam["InUse"] == 0:
                ids_cam = IDS_Camera(cam["camID"])
                ids_cam.initialize()
                self.ids_cams.append(ids_cam)
                ids_panel = IDS_Panel(
                    self.threadpool,
                    ids_cam, cam["Model"] + " " + cam["Serial"])
                ids_panel._directory = self.save_directory
                if len(self.ids_panels) == 0:
                    ids_panel.master = True
                else:
                    ids_panel.master = False
                ids_panel.exposureChanged.connect(self.master_exposure_changed)
                self.ids_panels.append(ids_panel)
                self.Hlayout.addWidget(ids_panel, 1)
                self.refresh_list_clicked()
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Device is in use or already added.",
                    QMessageBox.StandardButton.Ok)
        else:
            QMessageBox.warning(
                self,
                "Warning",
                "Please select a device.",
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

    def refresh_list_clicked(self):
        self.cam_list = IDS_Camera.get_camera_list()

        if self.cam_list is None:
            self.cam_list = []
            print('No cameras connected.')

        self.ids_model = QStandardItemModel(len(self.cam_list), 7)

        self.ids_model.setHorizontalHeaderLabels(
            ["In Use", "Camera ID", "Device ID",
             "Model", "Serial", "Status", "Sensor ID"])

        for i in range(len(self.cam_list)):
            self.ids_model.setItem(
                i, 0,
                QStandardItem(str(self.cam_list[i]["InUse"])))
            self.ids_model.setItem(
                i, 1,
                QStandardItem(str(self.cam_list[i]["camID"])))
            self.ids_model.setItem(
                i, 2,
                QStandardItem(str(self.cam_list[i]["devID"])))
            self.ids_model.setItem(
                i, 3,
                QStandardItem(self.cam_list[i]["Model"]))
            self.ids_model.setItem(
                i, 4,
                QStandardItem(self.cam_list[i]["Serial"]))
            self.ids_model.setItem(
                i, 5,
                QStandardItem(str(self.cam_list[i]["Status"])))
            self.ids_model.setItem(
                i, 6,
                QStandardItem(str(self.cam_list[i]["senID"])))

        self.ids_table.setModel(self.ids_model)

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
                " Capture {:d} | {:.2f} ms ".format(
                    panel._counter,
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

        myappid = u'samhitech.mircoEye.acquisition_module'  # appid
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        window = acquisition_module()
        return app, window
