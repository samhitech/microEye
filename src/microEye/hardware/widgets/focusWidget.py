import json

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
from pyqtgraph.widgets.PlotWidget import PlotWidget
from pyqtgraph.widgets.RemoteGraphicsView import RemoteGraphicsView


class focusWidget(QDockWidget):
    def __init__(self):
        super().__init__('IR Autofocus')

        # Remove close button from dock widgets
        self.setFeatures(
            QDockWidget.DockWidgetFloatable |
            QDockWidget.DockWidgetMovable)

        # display tab
        self.labelStyle = {'color': '#FFF', 'font-size': '10pt'}
        graphs_layout = QGridLayout()

        self.ROI_x = QDoubleSpinBox()
        self.ROI_x.setMaximum(5000)
        self.ROI_x.setValue(25)
        self.ROI_y = QDoubleSpinBox()
        self.ROI_y.setMaximum(5000)
        self.ROI_y.setValue(25)
        self.ROI_length = QDoubleSpinBox()
        self.ROI_length.setMaximum(5000)
        self.ROI_length.setValue(256)
        self.ROI_angle = QDoubleSpinBox()
        self.ROI_angle.setMaximum(5000)
        self.ROI_angle.setMinimum(-5000)
        self.ROI_angle.setValue(0)

        self.ROI_set_btn = QPushButton(
            ' Set ROI ',
            clicked=self.set_roi)
        self.ROI_save_btn = QPushButton(
            ' Save ',
            clicked=self.save_config)
        self.ROI_load_btn = QPushButton(
            ' Load ',
            clicked=self.load_config)

        self.IR_VLayout = QVBoxLayout()
        self.IR_HLayout = QHBoxLayout()
        self.IR_HLayout.addWidget(QLabel('Position X/Y'))
        self.IR_HLayout.addWidget(self.ROI_x)
        self.IR_HLayout.addWidget(self.ROI_y)
        self.IR_HLayout.addWidget(QLabel('Length'))
        self.IR_HLayout.addWidget(self.ROI_length)
        self.IR_HLayout.addWidget(QLabel('Angle'))
        self.IR_HLayout.addWidget(self.ROI_angle)
        self.IR_HLayout.addWidget(self.ROI_set_btn)
        self.IR_HLayout.addWidget(self.ROI_save_btn)
        self.IR_HLayout.addWidget(self.ROI_load_btn)
        self.IR_HLayout.addStretch()

        # IR LineROI Graph
        self.graph_IR = PlotWidget()
        self.graph_IR.setLabel("bottom", "Pixel", **self.labelStyle)
        self.graph_IR.setLabel("left", "Signal", "V", **self.labelStyle)
        # IR Peak Position Graph
        self.graph_Peak = PlotWidget()
        self.graph_Peak.setLabel("bottom", "Frame", **self.labelStyle)
        self.graph_Peak.setLabel(
            "left", "Center Pixel Error", **self.labelStyle)
        # IR Camera GraphView
        self.remote_view = RemoteGraphicsView()
        self.remote_view.pg.setConfigOptions(
            antialias=True, imageAxisOrder='row-major')
        pg.setConfigOption('imageAxisOrder', 'row-major')
        self.remote_plt = self.remote_view.pg.ViewBox(invertY=True)
        self.remote_plt._setProxyOptions(deferGetattr=True)
        self.remote_view.setCentralItem(self.remote_plt)
        self.remote_plt.setAspectLocked()
        self.remote_img = self.remote_view.pg.ImageItem(axisOrder='row-major')
        self.remote_img.setImage(
            np.random.normal(size=(512, 512)), _callSync='off')
        # IR LineROI
        self.roi = self.remote_view.pg.ROI(
            self.remote_view.pg.Point(25, 25),
            size=self.remote_view.pg.Point(0.5, 256),
            angle=0, pen='r')
        self.roi.addTranslateHandle([0.5, 0], [0.5, 1])
        self.roi.addScaleRotateHandle([0.5, 1], [0.5, 0])
        self.roi.updateFlag = False

        # self.roi.maxBounds = QRectF(0, 0, 513, 513)

        def roiChanged(cls):
            if not self.roi.updateFlag:
                pos = self.roi.pos()
                self.ROI_x.setValue(pos[0])
                self.ROI_y.setValue(pos[1])
                self.ROI_length.setValue(self.roi.size()[1])
                self.ROI_angle.setValue(self.roi.angle() % 360)

        self.lr_proxy = pg.multiprocess.proxy(
            roiChanged, callSync='off', autoProxy=True)
        self.roi.sigRegionChangeFinished.connect(self.lr_proxy)
        self.remote_plt.addItem(self.remote_img)
        self.remote_plt.addItem(self.roi)

        graphs_layout.addWidget(self.remote_view, 0, 0, 2, 1)
        graphs_layout.addWidget(self.graph_IR, 0, 1)
        graphs_layout.addWidget(self.graph_Peak, 1, 1)

        self.IR_VLayout.addLayout(self.IR_HLayout)
        self.IR_VLayout.addLayout(graphs_layout)

        container = QWidget(self)
        container.setLayout(self.IR_VLayout)
        self.setWidget(container)

        app = QApplication.instance()
        app.aboutToQuit.connect(self.remote_view.close)

    def set_roi(self):
        self.roi.updateFlag = True
        self.roi.setPos(
            self.ROI_x.value(),
            self.ROI_y.value())
        self.roi.setSize(
                [0.5, self.ROI_length.value()])
        self.roi.setAngle(
            self.ROI_angle.value())
        self.roi.updateFlag = False

    def save_config(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save config", filter="JSON Files (*.json);;")

        if len(filename) > 0:
            config = {
                'ROI_x': self.ROI_x.value(),
                'ROI_y': self.ROI_y.value(),
                'ROI_length': self.ROI_length.value(),
                'ROI_angle': self.ROI_angle.value(),
            }

            with open(filename, 'w') as file:
                json.dump(config, file)

            QMessageBox.information(
                self, "Info", "Config saved.")
        else:
            QMessageBox.warning(
                self, "Warning", "Config not saved.")

    def load_config(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load config", filter="JSON Files (*.json);;")

        if len(filename) > 0:
            config: dict = None
            keys = [
                'ROI_x',
                'ROI_y',
                'ROI_length',
                'ROI_angle',
            ]
            with open(filename, 'r') as file:
                config = json.load(file)
            if all(key in config for key in keys):
                self.ROI_x.setValue(float(config['ROI_x']))
                self.ROI_y.setValue(float(config['ROI_y']))
                self.ROI_length.setValue(float(config['ROI_length']))
                self.ROI_angle.setValue(float(config['ROI_angle']))
                self.set_roi()
            else:
                QMessageBox.warning(
                    self, "Warning", "Wrong or corrupted config file.")
        else:
            QMessageBox.warning(
                self, "Warning", "No file selected.")
