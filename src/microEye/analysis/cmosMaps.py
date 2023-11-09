import typing

import numpy as np
import tifffile as tf
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class cmosMaps(QWidget):

    def __init__(self, parent: typing.Optional['QWidget'] = None):
        super().__init__(parent=parent)

        self.invGain = None
        self.baseline = None
        self.darkCurrent = None
        self.readNoiseSQ = None
        self.thermalNoiseSQ = None
        self.expTime = 100.00699

        self.offsetMap = None
        self.varMap = None

        self.InitLayout()

    def InitLayout(self):

        self.main_layout = QFormLayout(self)

        self.setLayout(self.main_layout)

        self.invg_le = QLineEdit()
        self.baseline_le = QLineEdit()
        self.darkc_le = QLineEdit()
        self.readnsq_le = QLineEdit()
        self.thermnsq_le = QLineEdit()

        self.invg_le.setReadOnly(True)
        self.baseline_le.setReadOnly(True)
        self.darkc_le.setReadOnly(True)
        self.readnsq_le.setReadOnly(True)
        self.thermnsq_le.setReadOnly(True)

        self.invg_btn = QPushButton(
            'open',
            clicked=lambda: self.browseImg(0)
        )
        self.baseline_btn = QPushButton(
            'open',
            clicked=lambda: self.browseImg(1)
        )
        self.darkc_btn = QPushButton(
            'open',
            clicked=lambda: self.browseImg(2)
        )
        self.readnsq_btn = QPushButton(
            'open',
            clicked=lambda: self.browseImg(3)
        )
        self.thermnsq_btn = QPushButton(
            'open',
            clicked=lambda: self.browseImg(4)
        )

        self.main_layout.addRow(
            QLabel('inverse Gain map:'),
            self.invg_le
        )
        self.main_layout.addWidget(self.invg_btn)
        self.main_layout.addRow(
            QLabel('Baseline map [ADU]:'),
            self.baseline_le
        )
        self.main_layout.addWidget(self.baseline_btn)
        self.main_layout.addRow(
            QLabel('Dark current map [ADU/s]:'),
            self.darkc_le
        )
        self.main_layout.addWidget(self.darkc_btn)
        self.main_layout.addRow(
            QLabel('Read noise sq map [ADU^2]:'),
            self.readnsq_le
        )
        self.main_layout.addWidget(self.readnsq_btn)
        self.main_layout.addRow(
            QLabel('Thermal noise sq map [ADU^2/s]:'),
            self.thermnsq_le
        )
        self.main_layout.addWidget(self.thermnsq_btn)

        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setMinimum(0)
        self.exp_spin.setMaximum(1e4)
        self.exp_spin.setDecimals(5)
        self.exp_spin.setValue(self.expTime)

        self.main_layout.addRow(
            QLabel('Exposure [ms]:'),
            self.exp_spin
        )

        self.calc_btn = QPushButton(
            'gen. offset / var maps',
            clicked=lambda: self.calcMaps()
        )

        self.main_layout.addWidget(
            self.calc_btn
        )

        self.X = QSpinBox()
        self.Y = QSpinBox()
        self.W = QSpinBox()
        self.H = QSpinBox()
        self.X.setMinimum(0)
        self.Y.setMinimum(0)
        self.W.setMinimum(0)
        self.H.setMinimum(0)
        self.X.setMaximum(1e4)
        self.Y.setMaximum(1e4)
        self.W.setMaximum(1e4)
        self.H.setMaximum(1e4)

        self.main_layout.addRow(
            QLabel('ROI X:'),
            self.X
        )
        self.main_layout.addRow(
            QLabel('ROI Y:'),
            self.Y
        )
        self.main_layout.addRow(
            QLabel('ROI Width:'),
            self.W
        )
        self.main_layout.addRow(
            QLabel('ROI Height:'),
            self.H
        )

        self.active = QCheckBox('Use Maps?')
        self.active.setChecked(False)
        self.gain = QCheckBox('Use Gain?')
        self.gain.setChecked(False)

        self.main_layout.addWidget(
            self.active
        )
        self.main_layout.addWidget(
            self.gain
        )

    def browseImg(self, index):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load Image', filter='Tiff Image Files (*.tif);')

        if len(filename) > 0:
            img = tf.imread(filename)
            self.H.setValue(img.shape[0])
            self.W.setValue(img.shape[1])

            if index == 0:
                self.invg_le.setText(filename)
                self.invg_le.setToolTip(str(img.shape))
                self.invGain = img
            elif index == 1:
                self.baseline_le.setText(filename)
                self.baseline_le.setToolTip(str(img.shape))
                self.baseline = img
            elif index == 2:
                self.darkc_le.setText(filename)
                self.darkc_le.setToolTip(str(img.shape))
                self.darkCurrent = img
            elif index == 3:
                self.readnsq_le.setText(filename)
                self.readnsq_le.setToolTip(str(img.shape))
                self.readNoiseSQ = img
            elif index == 4:
                self.thermnsq_le.setText(filename)
                self.thermnsq_le.setToolTip(str(img.shape))
                self.thermalNoiseSQ = img

    def missingMap(self):
        print('Missing Maps!')

    def calcMaps(self, export=True):
        if self.invGain is None:
            self.missingMap()
            return False
        if self.baseline is None:
            self.missingMap()
            return False
        if self.darkCurrent is None:
            self.missingMap()
            return False
        if self.readNoiseSQ is None:
            self.missingMap()
            return False
        if self.thermalNoiseSQ is None:
            self.missingMap()
            return False

        shapes = []
        shapes.append(self.invGain.shape)
        shapes.append(self.baseline.shape)
        shapes.append(self.darkCurrent.shape)
        shapes.append(self.readNoiseSQ.shape)
        shapes.append(self.thermalNoiseSQ.shape)
        shapes = np.vstack(shapes)

        if not np.all(shapes[:, 0] == shapes[0, 0]) or \
                not np.all(shapes[:, 1] == shapes[0, 1]):
            print('Maps have unmatching dimensions!')
            return False

        self.expTime = self.exp_spin.value()

        offsetMap = self.baseline + self.expTime * self.darkCurrent

        varMap = self.readNoiseSQ + self.expTime * self.thermalNoiseSQ

        if self.gain.isChecked():
            self.offsetMap = offsetMap * self.invGain
            self.varMap = varMap * np.square(self.invGain)
        else:
            self.offsetMap = offsetMap
            self.varMap = varMap

        if not export:
            return True

        _directory = str(
                QFileDialog.getExistingDirectory(self, 'Select Directory'))

        if len(_directory) < 1:
            return

        tf.imwrite(
            _directory + f'/offset_{self.expTime:.5f}ms'.replace('.', '_') + '.tif',
            self.offsetMap)
        tf.imwrite(
            _directory + f'/var_{self.expTime:.5f}ms'.replace('.', '_') + '.tif',
            self.varMap)

        return True

    def getMaps(self):
        if self.calcMaps(False):
            x = self.X.value()
            y = self.Y.value()
            w = self.W.value()
            h = self.H.value()

            offset = self.offsetMap[y:y+h, x:x+w]
            var = self.varMap[y:y+h, x:x+w]

            if self.gain.isChecked():
                gain = self.invGain[y:y+h, x:x+w]
            else:
                gain = np.ones_like(offset)
            return gain, offset, var
        else:
            return None
