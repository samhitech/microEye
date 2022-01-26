import ctypes
import os
import sys
from enum import auto

import cv2
import numba
import numpy as np
import ome_types.model as om
import pandas as pd
import pyqtgraph as pg
import qdarkstyle
from sympy import true
import tifffile as tf
import zarr
from numpy.lib.type_check import imag
from ome_types.model.channel import Channel
from ome_types.model.ome import OME
from ome_types.model.simple_types import UnitsLength, UnitsTime
from ome_types.model.tiff_data import TiffData
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import *
from pyqtgraph.colormap import ColorMap
from scipy.fftpack import fft2, fftshift, ifft2, ifftshift

from .Filters import *
from .Fitting import *
from .metadata import MetadataEditor
from .Rendering import *
from .thread_worker import *
from .uImage import *


class tiff_viewer(QMainWindow):

    def __init__(self, path=os.path.dirname(os.path.abspath(__file__))):
        super().__init__()
        self.title = 'microEye tiff viewer'
        self.left = 0
        self.top = 0
        self.width = 1024
        self.height = 600
        self._zoom = 1  # display resize
        self._n_levels = 4096

        self.fittingResults = None

        self.tiff = None

        self.tiffSequence = None
        self.tiffSeq_Handler = None
        # self.tiffStore = None
        # self.tiffZar = None

        # Threading
        self._threadpool = QThreadPool()
        print("Multithreading with maximum %d threads"
              % self._threadpool.maxThreadCount())

        self.initialize(path)

        self.status()

        # Statues Bar Timer
        self.timer = QTimer()
        self.timer.setInterval(10)
        self.timer.timeout.connect(self.status)
        self.timer.start()

    def initialize(self, path):
        # Set Title / Dimensions / Center Window
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.center()

        # Define main window layout
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # Initialize the file system model / tree
        self.path = path
        self.model = QFileSystemModel()
        self.model.setRootPath(self.path)
        self.model.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.Files |
            QDir.Filter.NoDotAndDotDot)
        self.model.setNameFilters(['*.tif', '*.tiff'])
        self.model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(self.path))

        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)

        self.tree.doubleClicked.connect(self._open_file)

        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.hideColumn(3)

        self.tree.setWindowTitle("Dir View")
        self.tree.resize(640, 480)

        # Metadata Viewer / Editor
        self.metadataEditor = MetadataEditor()

        # Side TabView
        self.tabView = QTabWidget()

        # Graphical layout
        self.g_layout_widget = QVBoxLayout()

        # Add the two sub-main layouts
        self.main_layout.addWidget(self.tabView, 2)
        self.main_layout.addLayout(self.g_layout_widget, 3)

        # Tiff File system tree viewer tab
        self.file_tree = QWidget()
        self.file_tree_layout = QVBoxLayout()
        self.file_tree.setLayout(self.file_tree_layout)

        # Tiff Options tab layout
        self.controls_group = QWidget()
        self.controls_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_layout)

        # Localization / Render tab layout
        self.loc_group = QWidget()
        self.loc_layout = QVBoxLayout()
        self.loc_group.setLayout(self.loc_layout)

        # Add Tabs
        self.tabView.addTab(self.file_tree, 'File system')
        self.tabView.addTab(self.metadataEditor, 'OME-XML Metadata')
        self.tabView.addTab(self.controls_group, 'Tiff Options')
        self.tabView.addTab(self.loc_group, 'Localization / Render')

        # Add the File system tab contents
        self.imsq_pattern = QLineEdit('/image_0*.ome.tif')

        self.file_tree_layout.addWidget(QLabel('Image Sequence pattern:'))
        self.file_tree_layout.addWidget(self.imsq_pattern)
        self.file_tree_layout.addWidget(self.tree)

        self.series_slider = QSlider(Qt.Horizontal)
        self.series_slider.setMinimum(0)
        self.series_slider.setMaximum(0)

        self.pages_label = QLabel('Pages')
        self.pages_slider = QSlider(Qt.Horizontal)
        self.pages_slider.setMinimum(0)
        self.pages_slider.setMaximum(0)
        self.pages_slider.valueChanged.connect(self.slider_changed)

        self.min_label = QLabel('Min')
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(self._n_levels - 1)
        self.min_slider.valueChanged.connect(self.slider_changed)
        self.min_slider.valueChanged.emit(0)

        self.max_label = QLabel('Max')
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(self._n_levels - 1)
        self.max_slider.setValue(self._n_levels - 1)
        self.max_slider.valueChanged.connect(self.slider_changed)
        self.max_slider.valueChanged.emit(self._n_levels - 1)

        self.autostretch = QCheckBox('Auto-Stretch')
        self.autostretch.setChecked(True)
        self.autostretch.stateChanged.connect(self.slider_changed)

        # display size
        self.zoom_layout = QHBoxLayout()
        self.zoom_lbl = QLabel("Scale " + "{:.0f}%".format(self._zoom * 100))
        self.zoom_in_btn = QPushButton(
            "+",
            clicked=lambda: self.zoom_in()
        )
        self.zoom_out_btn = QPushButton(
            "-",
            clicked=lambda: self.zoom_out()
        )
        self.zoom_layout.addWidget(self.zoom_lbl, 4)
        self.zoom_layout.addWidget(self.zoom_out_btn, 1)
        self.zoom_layout.addWidget(self.zoom_in_btn, 1)

        self.average_btn = QPushButton(
            'Average stack',
            clicked=lambda: self.average_stack())

        self.controls_layout.addWidget(self.pages_label)
        self.controls_layout.addWidget(self.pages_slider)
        self.controls_layout.addWidget(self.autostretch)
        self.controls_layout.addWidget(self.min_label)
        self.controls_layout.addWidget(self.min_slider)
        self.controls_layout.addWidget(self.max_label)
        self.controls_layout.addWidget(self.max_slider)

        self.th_min_label = QLabel('Min')
        self.th_min_slider = QSlider(Qt.Horizontal)
        self.th_min_slider.setMinimum(0)
        self.th_min_slider.setMaximum(255)
        self.th_min_slider.valueChanged.connect(self.slider_changed)
        self.th_min_slider.valueChanged.emit(0)

        self.th_max_label = QLabel('Max')
        self.th_max_slider = QSlider(Qt.Horizontal)
        self.th_max_slider.setMinimum(0)
        self.th_max_slider.setMaximum(255)
        self.th_max_slider.setValue(255)
        self.th_max_slider.valueChanged.connect(self.slider_changed)
        self.th_max_slider.valueChanged.emit(255)

        self.detection = QCheckBox('Blob Detection')
        self.detection.setChecked(False)
        self.detection.stateChanged.connect(self.slider_changed)

        self.controls_layout.addWidget(self.detection)
        self.controls_layout.addWidget(self.th_min_label)
        self.controls_layout.addWidget(self.th_min_slider)
        self.controls_layout.addWidget(self.th_max_label)
        self.controls_layout.addWidget(self.th_max_slider)

        self.tempMedianFilter = TemporalMedianFilterWidget()
        self.tempMedianFilter.update.connect(lambda: self.update_display())

        self.bandpassFilter = BandpassFilterWidget()
        self.bandpassFilter.update.connect(lambda: self.update_display())

        self.controls_layout.addWidget(self.tempMedianFilter)
        self.controls_layout.addWidget(self.bandpassFilter)

        self.minArea = QDoubleSpinBox()
        self.minArea.setMinimum(0)
        self.minArea.setMaximum(1024)
        self.minArea.setSingleStep(0.05)
        self.minArea.setValue(3.0)
        self.minArea.valueChanged.connect(self.slider_changed)

        self.maxArea = QDoubleSpinBox()
        self.maxArea.setMinimum(0)
        self.maxArea.setMaximum(1024)
        self.maxArea.setSingleStep(0.05)
        self.maxArea.setValue(16)
        self.maxArea.valueChanged.connect(self.slider_changed)

        self.minCircularity = QDoubleSpinBox()
        self.minCircularity.setMinimum(0)
        self.minCircularity.setMaximum(1)
        self.minCircularity.setSingleStep(0.05)
        self.minCircularity.setValue(0)
        self.minCircularity.valueChanged.connect(self.slider_changed)

        self.minConvexity = QDoubleSpinBox()
        self.minConvexity.setMinimum(0)
        self.minConvexity.setMaximum(1)
        self.minConvexity.setSingleStep(0.05)
        self.minConvexity.setValue(0)
        self.minConvexity.valueChanged.connect(self.slider_changed)

        self.minInertiaRatio = QDoubleSpinBox()
        self.minInertiaRatio.setMinimum(0)
        self.minInertiaRatio.setMaximum(1)
        self.minInertiaRatio.setSingleStep(0.05)
        self.minInertiaRatio.setValue(0)
        self.minInertiaRatio.valueChanged.connect(self.slider_changed)

        self.controls_layout.addWidget(
            QLabel('Min/Max Area:'))
        self.controls_layout.addWidget(self.minArea)
        self.controls_layout.addWidget(self.maxArea)
        # self.controls_layout.addWidget(self.minCircularity)
        # self.controls_layout.addWidget(self.minConvexity)
        # self.controls_layout.addWidget(self.minInertiaRatio)

        # self.controls_layout.addWidget(self.average_btn)
        # self.controls_layout.addLayout(self.zoom_layout)
        self.controls_layout.addStretch()

        # Localization / Render layout
        self.fitting_cbox = QComboBox()
        self.fitting_cbox.addItem('2D Phasor-Fit')

        self.render_cbox = QComboBox()
        self.render_cbox.addItem('2D Gaussian Histogram')

        self.frc_cbox = QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Check Pattern')

        self.export_frm = QCheckBox('Frame')
        self.export_frm.setChecked(True)
        self.export_loc_px = QCheckBox('Localizations (px)')
        self.export_loc_px.setChecked(False)
        self.export_loc_nm = QCheckBox('Localizations (nm)')
        self.export_loc_nm.setChecked(True)
        self.export_int = QCheckBox('Intensity')
        self.export_int.setChecked(True)

        self.export_precision = QLineEdit('%10.5f')

        self.export_image = QCheckBox('Super-res Image')
        self.export_image.setChecked(True)

        self.px_size = QDoubleSpinBox()
        self.px_size.setMinimum(0)
        self.px_size.setMaximum(10000)
        self.px_size.setValue(130.0)

        self.super_px_size = QSpinBox()
        self.super_px_size.setMinimum(0)
        self.super_px_size.setValue(10)

        self.drift_cross_args = QHBoxLayout()
        self.drift_cross_bins = QSpinBox()
        self.drift_cross_bins.setValue(10)
        self.drift_cross_px = QSpinBox()
        self.drift_cross_px.setValue(10)
        self.drift_cross_up = QSpinBox()
        self.drift_cross_up.setMaximum(1000)
        self.drift_cross_up.setValue(100)
        self.drift_cross_args.addWidget(self.drift_cross_bins)
        self.drift_cross_args.addWidget(self.drift_cross_px)
        self.drift_cross_args.addWidget(self.drift_cross_up)

        self.nneigh_merge_args = QHBoxLayout()
        self.nn_neighbors = QSpinBox()
        self.nn_neighbors.setValue(1)
        self.nn_max_distance = QDoubleSpinBox()
        self.nn_max_distance.setMaximum(10000)
        self.nn_max_distance.setValue(30)
        self.nn_max_off = QSpinBox()
        self.nn_max_off.setValue(1)
        self.nn_max_length = QSpinBox()
        self.nn_max_length.setMaximum(10000)
        self.nn_max_length.setValue(500)
        self.nneigh_merge_args.addWidget(self.nn_neighbors)
        self.nneigh_merge_args.addWidget(self.nn_max_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_off)
        self.nneigh_merge_args.addWidget(self.nn_max_length)

        self.loc_btn = QPushButton(
            'Localize',
            clicked=lambda: self.localize())
        self.refresh_btn = QPushButton(
            'Display / Refresh Super-res Image',
            clicked=lambda: self.update_loc())
        self.frc_res_btn = QPushButton(
            'FRC Resolution',
            clicked=lambda: self.FRC_estimate())
        self.drift_cross_btn = QPushButton(
            'Drift cross-correlation',
            clicked=lambda: self.drift_cross())
        self.nneigh_merge_btn = QPushButton(
            'Nearest-neighbour Merging',
            clicked=lambda: self.nneigh_merge())

        self.export_loc_btn = QPushButton(
            'Export Localizations',
            clicked=lambda: self.export_loc())
        self.import_loc_btn = QPushButton(
            'Import Localizations',
            clicked=lambda: self.import_loc())

        self.loc_layout.addWidget(QLabel('Fitting:'))
        self.loc_layout.addWidget(self.fitting_cbox)
        self.loc_layout.addWidget(QLabel('Rendering Method:'))
        self.loc_layout.addWidget(self.render_cbox)
        self.loc_layout.addWidget(QLabel('FRC Method:'))
        self.loc_layout.addWidget(self.frc_cbox)
        self.loc_layout.addWidget(QLabel('Pixel-size [nm]:'))
        self.loc_layout.addWidget(self.px_size)
        self.loc_layout.addWidget(QLabel('Super resolution pixel-size [nm]:'))
        self.loc_layout.addWidget(self.super_px_size)
        self.loc_layout.addWidget(QLabel('Exported:'))
        self.loc_layout.addWidget(self.export_frm)
        self.loc_layout.addWidget(self.export_loc_px)
        self.loc_layout.addWidget(self.export_loc_nm)
        self.loc_layout.addWidget(self.export_int)
        self.loc_layout.addWidget(self.export_image)
        self.loc_layout.addWidget(QLabel('Format:'))
        self.loc_layout.addWidget(self.export_precision)
        self.loc_layout.addWidget(self.loc_btn)
        self.loc_layout.addWidget(self.refresh_btn)
        self.loc_layout.addWidget(self.frc_res_btn)
        self.loc_layout.addWidget(
            QLabel('Drift X-Corr. (bins, pixelSize, upsampling):'))
        self.loc_layout.addLayout(self.drift_cross_args)
        self.loc_layout.addWidget(self.drift_cross_btn)
        self.loc_layout.addWidget(
            QLabel('NN (n-neighbor, max-distance, max-off, max-len):'))
        self.loc_layout.addLayout(self.nneigh_merge_args)
        self.loc_layout.addWidget(self.nneigh_merge_btn)
        self.loc_layout.addWidget(self.export_loc_btn)
        self.loc_layout.addWidget(self.import_loc_btn)
        self.loc_layout.addStretch()

        # graphics layout
        # # A plot area (ViewBox + axes) for displaying the image
        self.uImage = None
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image_plot = pg.ImageView(imageItem=self.image)
        self.image_plot.setLevels(0, 255)
        # self.image_plot.setColorMap(pg.colormap.getFromMatplotlib('jet'))
        self.g_layout_widget.addWidget(self.image_plot)
        # Item for displaying image data

        self.show()

    def center(self):
        '''Centers the window within the screen.
        '''
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def status(self):
        # Statusbar time
        self.statusBar().showMessage(
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz') +
            ' | Results: ' +
            ('None' if self.fittingResults is None else str(
                len(self.fittingResults)))
            )

    def _open_file(self, i):
        if not self.model.isDir(i):
            cv2.destroyAllWindows()
            index = self.model.index(i.row(), 0, i.parent())
            self.path = self.model.filePath(index)
            if self.tiffSeq_Handler is not None:
                self.tiffSeq_Handler.close()
            self.tiffSequence = tf.TiffSequence([self.path])
            self.tiffSeq_Handler = TiffSeqHandler(self.tiffSequence)
            self.tiffSeq_Handler.open()
            # self.tiffStore = self.tiffSequence.aszarr(axestiled={0: 0})
            # self.tiffZar = zarr.open(self.tiffStore, mode='r')

            self.pages_slider.setMaximum(len(self.tiffSeq_Handler) - 1)
            self.pages_slider.valueChanged.emit(0)

            with tf.TiffFile(self.tiffSequence.files[0]) as fl:
                if fl.is_ome:
                    ome = OME.from_xml(fl.ome_metadata)
                    self.metadataEditor.pop_OME_XML(ome)
                    self.px_size.setValue(
                        self.metadataEditor.px_size.value())

            # self.update_display()
            # self.genOME()
        else:
            index = self.model.index(i.row(), 0, i.parent())
            self.path = self.model.filePath(index)
            if self.tiffSeq_Handler is not None:
                self.tiffSeq_Handler.close()
            try:
                self.tiffSequence = tf.TiffSequence(
                    self.path + '/' + self.imsq_pattern.text())
            except ValueError:
                self.tiffSequence = None

            if self.tiffSequence is not None:
                self.tiffSeq_Handler = TiffSeqHandler(self.tiffSequence)
                self.tiffSeq_Handler.open()
                # self.tiffStore = self.tiffSequence.aszarr(axestiled={0: 0})
                # chunks = (1,) + self.tiffStore._chunks[1:]
                # self.tiffZar = zarr.open(
                #     self.tiffStore, mode='r', chunks=chunks)

                self.pages_slider.setMaximum(
                    self.tiffSeq_Handler.__len__() - 1)
                self.pages_slider.valueChanged.emit(0)

                with tf.TiffFile(self.tiffSequence.files[0]) as fl:
                    if fl.is_ome:
                        ome = OME.from_xml(fl.ome_metadata)
                        self.metadataEditor.pop_OME_XML(ome)
                        self.px_size.setValue(
                            self.metadataEditor.px_size.value())

    def average_stack(self):
        if self.tiffSeq_Handler is not None:
            sum = np.array([page.asarray() for page in self.tiff.pages])
            avg = sum.mean(axis=0, dtype=np.float32)

            self.image.setImage(avg, autoLevels=False)

    def genOME(self):
        if self.tiffSeq_Handler is not None:
            frames = len(self.tiffSeq_Handler)
            width = self.image.image.shape[1]
            height = self.image.image.shape[0]
            ome = self.metadataEditor.gen_OME_XML(frames, width, height)
            # tf.tiffcomment(
            #     self.path,
            #     ome.to_xml())
            # print(om.OME.from_tiff(self.tiff.filename))

    def zoom_in(self):
        """Increase image display size"""
        self._zoom = min(self._zoom + 0.05, 4)
        self.zoom_lbl.setText("Scale " + "{:.0f}%".format(self._zoom*100))
        self.update_display()

    def zoom_out(self):
        """Decrease image display size"""
        self._zoom = max(self._zoom - 0.05, 0.25)
        self.zoom_lbl.setText("Scale " + "{:.0f}%".format(self._zoom*100))
        self.update_display()

    def slider_changed(self, value):
        if self.tiffSeq_Handler is not None:
            self.update_display()
        if self.sender() is self.pages_slider:
            self.pages_label.setText(
                'Page: {:d}/{:d}'.format(
                    value + 1,
                    self.pages_slider.maximum() + 1))
        elif self.sender() is self.min_slider:
            self.min_label.setText(
                'Min: {:d}/{:d}'.format(
                    value,
                    self.min_slider.maximum()))
        elif self.sender() is self.max_slider:
            self.max_label.setText(
                'Max: {:d}/{:d}'.format(
                    value,
                    self.max_slider.maximum()))
        elif self.sender() is self.th_min_slider:
            self.th_min_label.setText(
                'Min det. threshold: {:d}/{:d}'.format(
                    value,
                    self.th_min_slider.maximum()))
        elif self.sender() is self.th_max_slider:
            self.th_max_label.setText(
                'Max det. threshold: {:d}/{:d}'.format(
                    value,
                    self.th_max_slider.maximum()))

    def update_display(self, image=None):
        if image is None:
            image = self.tiffSeq_Handler[self.pages_slider.value()]

        if self.tempMedianFilter.enabled.isChecked():
            self.tempMedianFilter.filter.getFrames(
                self.pages_slider.value(), self.tiffSeq_Handler)

            image = self.tempMedianFilter.filter.run(image)

        self.uImage = uImage(image)

        self.max_slider.setMaximum(self.uImage._max)
        self.min_slider.setMaximum(self.uImage._max)

        min_max = None
        if not self.autostretch.isChecked():
            min_max = (self.min_slider.value(), self.max_slider.value())

        self.uImage.equalizeLUT(min_max, True)

        if self.autostretch.isChecked():
            self.min_slider.valueChanged.disconnect(self.slider_changed)
            self.max_slider.valueChanged.disconnect(self.slider_changed)
            self.min_slider.setValue(self.uImage._min)
            self.max_slider.setValue(self.uImage._max)
            self.min_slider.valueChanged.connect(self.slider_changed)
            self.max_slider.valueChanged.connect(self.slider_changed)

        # cv2.imshow(self.path, image)
        self.image.setImage(self.uImage._view, autoLevels=False)

        if self.detection.isChecked():
            # bandpass filter
            img = self.bandpassFilter.filter.run(self.uImage._view)

            _, th_img = cv2.threshold(
                img,
                self.th_min_slider.value(),
                self.th_max_slider.value(),
                cv2.THRESH_BINARY)

            cv2.imshow("Keypoints_TH", th_img)
            # print(ex, end='\r')
            # Detect blobs.
            keypoints = self.blob_detector().detect(th_img)

            im_with_keypoints = cv2.drawKeypoints(
                img, keypoints, np.array([]),
                (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Show keypoints
            cv2.imshow("Keypoints_CV", im_with_keypoints)

            points = cv2.KeyPoint_convert(keypoints)

            sub_fit = phasor_fit(self.uImage._image, points, False)

            if sub_fit is not None:

                keypoints = [cv2.KeyPoint(*point, size=1.0) for point in
                             sub_fit[:, :2]]

                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
                # the size of the circle corresponds to the size of blob
                im_with_keypoints = cv2.drawKeypoints(
                    self.uImage._view, keypoints, np.array([]),
                    (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                # Show keypoints
                # cv2.imshow("Keypoints_PH", im_with_keypoints)

                self.image.setImage(im_with_keypoints, autoLevels=False)

                # return np.array(points, copy=True)

    def blob_detector(self) -> cv2.SimpleBlobDetector:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        params.minDistBetweenBlobs = 0

        # Change thresholds
        params.minThreshold = float(self.th_min_slider.value())
        params.maxThreshold = float(self.th_max_slider.value())

        # Filter by Area.
        params.filterByArea = True
        params.minArea = self.minArea.value()
        params.maxArea = self.maxArea.value()

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = self.minCircularity.value()

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = self.minConvexity.value()

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = self.minInertiaRatio.value()

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            return cv2.SimpleBlobDetector(params)
        else:
            return cv2.SimpleBlobDetector_create(params)

    def FRC_estimate(self):
        frc_method = self.frc_cbox.currentText()
        if 'Check' in frc_method:
            img = self.update_loc()

            if img is not None:
                def work_func():
                    FRC_resolution_check_pattern(
                        img, self.super_px_size.value())
            else:
                return
        elif 'Binomial' in frc_method:
            if self.fittingResults is None:
                return
            elif len(self.fittingResults) > 0:
                data = self.fittingResults.toRender()

                def work_func():
                    return FRC_resolution_binomial(
                        np.c_[data[0], data[1], data[2]],
                        self.super_px_size.value()
                    )
            else:
                return
        else:
            return

        self.worker = thread_worker(
            work_func,
            progress=False, z_stage=False)
        self.worker.signals.result.connect(lambda x: plotFRC(*x))
        # Execute
        self._threadpool.start(self.worker)

    def drift_cross(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                return self.fittingResults.drift_cross_correlation(
                    self.drift_cross_bins.value(),
                    self.drift_cross_px.value(),
                    self.drift_cross_up.value(),
                )

            def done(results):
                if results is not None:
                    img_norm = cv2.normalize(
                        results[1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    self.image.setImage(img_norm, autoLevels=False)
                    self.fittingResults = results[0]
                    plotDriftXCorr(*results[2])

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self._threadpool.start(self.worker)
        else:
            return

    def nneigh_merge(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            self.fittingResults = \
                self.fittingResults.nearest_neighbour_merging(
                    self.nn_max_distance.value(),
                    self.nn_max_off.value(),
                    self.nn_max_length.value(),
                    self.nn_neighbors.value()
                )
        else:
            return

    def import_loc(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import localizations", filter="TSV Files (*.tsv);;")

        if len(filename) > 0:

            results = FittingResults.fromFile(
                filename,
                self.px_size.value())

            if results is not None:
                self.fittingResults = results
                self.update_loc()
                print('Done importing results.')
            else:
                print('Error importing results.')

    def export_loc(self, filename=None):
        if self.fittingResults is None:
            return

        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export localizations", filter="TSV Files (*.tsv);;")

        if len(filename) > 0:
            sres_img = self.update_loc()

            if sres_img is not None:
                tf.imsave(
                    filename.replace('.tsv', '_super_res.tif'),
                    sres_img,
                    photometric='minisblack',
                    append=True,
                    bigtiff=True,
                    ome=False)

            exp_columns = [
                self.export_frm.isChecked(),
                self.export_loc_px.isChecked(),
                self.export_loc_px.isChecked(),
                self.export_loc_nm.isChecked(),
                self.export_loc_nm.isChecked(),
                self.export_int.isChecked(),
                True,
                True,
                True
            ]

            dataFrame = self.fittingResults.dataFrame()
            dataFrame.to_csv(
                filename, index=False,
                columns=FittingResults.columns[exp_columns],
                float_format=self.export_precision.text(),
                sep='\t',
                encoding='utf-8')

    def localize(self):
        if self.tiffSeq_Handler is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save localizations", filter="TSV Files (*.tsv);;")

        if len(filename) > 0:
            # Any other args, kwargs are passed to the run function
            self.worker = thread_worker(
                self.proccess_loc, filename,
                progress=True, z_stage=False)
            self.worker.signals.progress.connect(self.update_loc)
            # Execute
            self._threadpool.start(self.worker)

    def update_loc(self):
        if self.fittingResults is None:
            return None
        elif len(self.fittingResults) > 0:
            renderClass = gauss_hist_render(self.super_px_size.value())
            img = renderClass.render(
                *self.fittingResults.toRender())
            img_norm = cv2.normalize(
                img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            self.image.setImage(img_norm, autoLevels=False)
            return img
        else:
            return None

    def update_lists(self, result: np.ndarray):
        if result is not None:
            self.fittingResults.extend(result)
        self.thread_done += 1

    def proccess_loc(self, filename: str, progress_callback):
        self.fittingResults = FittingResults(
            ResultsUnits.Pixel,
            self.px_size.value()
        )
        self.thread_done = 0
        time = QDateTime.currentDateTime()
        filter = self.bandpassFilter.filter
        tempEnabled = self.tempMedianFilter.enabled.isChecked()
        blob_detector = self.blob_detector()
        threads = self._threadpool.maxThreadCount() - 2
        print('Threads', threads)
        for i in range(
                0, int(np.ceil(len(self.tiffSeq_Handler) / threads) * threads),
                threads):
            time = QDateTime.currentDateTime()
            workers = []
            self.thread_done = 0
            for k in range(threads):
                if i + k < len(self.tiffSeq_Handler):
                    img = self.tiffSeq_Handler[i + k].copy()

                    temp = None
                    if tempEnabled:
                        temp = TemporalMedianFilter()
                        temp._temporal_window = \
                            self.tempMedianFilter.filter._temporal_window

                    worker = thread_worker(
                        self.localize_frame,
                        img,
                        temp,
                        filter,
                        i + k,
                        progress=False, z_stage=False)
                    worker.signals.result.connect(self.update_lists)
                    workers.append(worker)
                    QThreadPool.globalInstance().start(worker)

            while self.thread_done < len(workers):
                QThread.msleep(10)

            exex = time.msecsTo(QDateTime.currentDateTime())

            print(
                'index: {:d}/{:d}, Time: {:d}  '.format(
                    i + len(workers), len(self.tiffSeq_Handler), exex),
                end="\r")
            if (i // threads) % 40 == 0:
                progress_callback.emit(self.uImage._image.shape)

        QThread.msleep(5000)

        self.export_loc(filename)

    def localize_frame(
            self, image: np.ndarray,
            temp: TemporalMedianFilter,
            filter: AbstractFilter,
            index):

        if temp is not None:
            temp.getFrames(index, self.tiffSeq_Handler)
            image = temp.run(image)

        uImg = uImage(image)

        uImg.equalizeLUT(None, False)

        if filter is BandpassFilter:
            filter._show_filter = False
            filter._refresh = False

        img = filter.run(uImg._view)

        # Detect blobs.
        _, th_img = cv2.threshold(
                img,
                self.th_min_slider.value(),
                self.th_max_slider.value(),
                cv2.THRESH_BINARY)

        keypoints = self.blob_detector().detect(th_img)

        points: np.ndarray = cv2.KeyPoint_convert(keypoints)

        time = QDateTime.currentDateTime()
        result = phasor_fit(uImg._image, points)

        if result is not None:
            result[:, 3] = [index + 1] * points.shape[0]
            # print(index + 1, result[:, 3].shape)

        exex = time.msecsTo(QDateTime.currentDateTime())

        return result

    def StartGUI(path=None):
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
        font = app.font()
        font.setPointSize(10)
        app.setFont(font)
        # sets the app icon
        dirname = os.path.dirname(os.path.abspath(__package__))
        app_icon = QIcon()
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/16.png'), QSize(16, 16))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/24.png'), QSize(24, 24))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/32.png'), QSize(32, 32))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/48.png'), QSize(48, 48))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/64.png'), QSize(64, 64))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/128.png'), QSize(128, 128))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/256.png'), QSize(256, 256))
        app_icon.addFile(
            os.path.join(dirname, 'microEye/icons/512.png'), QSize(512, 512))

        app.setWindowIcon(app_icon)

        myappid = u'samhitech.mircoEye.tiff_viewer'  # appid
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        window = tiff_viewer(path)
        return app, window


if __name__ == '__main__':
    app, window = tiff_viewer.StartGUI()
    app.exec_()
