import os
import sys

import cv2
import numba
import numpy as np
import pandas as pd
import pyqtgraph as pg
import qdarkstyle
import tifffile as tf
import zarr
from ome_types.model.ome import OME
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import *
from sympy import zeros

from microEye.fitting.gaussian_fit import gaussian_2D_fit

from .fitting.results import *
from .fitting.fit import *

from .checklist_dialog import Checklist
from .Filters import *
from .metadata import MetadataEditor
from .Rendering import *
from .thread_worker import *
from .uImage import *


class tiff_viewer(QMainWindow):

    def __init__(self, path=os.path.dirname(os.path.abspath(__package__))):
        super().__init__()
        self.title = 'microEye tiff viewer'
        self.left = 0
        self.top = 0
        self.width = 1024
        self.height = 512
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
        self.tree.resize(512, 256)

        # Metadata Viewer / Editor
        self.metadataEditor = MetadataEditor()

        # Side TabView
        self.tabView = QTabWidget()
        self.tabView.setMinimumWidth(350)
        self.tabView.setMaximumWidth(400)

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
        self.loc_form = QFormLayout()
        self.loc_group.setLayout(self.loc_form)

        # Add Tabs
        self.tabView.addTab(self.file_tree, 'File system')
        self.tabView.addTab(self.controls_group, 'Tiff Options')
        self.tabView.addTab(self.loc_group, 'Localization / Render')
        self.tabView.addTab(self.metadataEditor, 'OME-XML Metadata')

        # Add the File system tab contents
        self.imsq_pattern = QLineEdit('/image_0*.ome.tif')

        self.file_tree_layout.addWidget(QLabel('Image Sequence pattern:'))
        self.file_tree_layout.addWidget(self.imsq_pattern)
        self.file_tree_layout.addWidget(self.tree)

        self.image_control_layout = QFormLayout()

        self.pages_label = QLabel('Pages')
        self.pages_slider = QSlider(Qt.Horizontal)
        self.pages_slider.setMinimum(0)
        self.pages_slider.setMaximum(0)

        self.min_label = QLabel('Min level:')
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(self._n_levels - 1)
        self.min_slider.valueChanged.emit(0)

        self.max_label = QLabel('Max level:')
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(self._n_levels - 1)
        self.max_slider.setValue(self._n_levels - 1)
        self.max_slider.valueChanged.emit(self._n_levels - 1)

        self.autostretch = QCheckBox('Auto-Stretch')
        self.autostretch.setChecked(True)

        self.enableROI = QCheckBox('Enable ROI')
        self.enableROI.setChecked(False)
        self.enableROI.stateChanged.connect(self.enableROI_changed)

        self.saveCropped = QPushButton(
            'Save Cropped Image',
            clicked=lambda: self.save_cropped_img())

        self.image_control_layout.addRow(
            self.pages_label,
            self.pages_slider)
        self.image_control_layout.addRow(
            self.min_label,
            self.min_slider)
        self.image_control_layout.addRow(
            self.max_label,
            self.max_slider)
        self.image_control_layout.addWidget(self.autostretch)
        self.image_control_layout.addWidget(self.enableROI)
        self.image_control_layout.addWidget(self.saveCropped)

        self.controls_layout.addLayout(
            self.image_control_layout)

        self.tempMedianFilter = TemporalMedianFilterWidget()
        self.tempMedianFilter.update.connect(lambda: self.update_display())
        self.controls_layout.addWidget(self.tempMedianFilter)

        self.detection_layout = QFormLayout()
        self.detection_group = QGroupBox('Blob Detection / Fitting')
        self.detection_group.setLayout(self.detection_layout)

        self.detection = QCheckBox('Enabled')
        self.detection.setChecked(False)

        self.th_min_label = QLabel('Detection threshold:')
        self.th_min_slider = QSpinBox()
        self.th_min_slider.setMinimum(0)
        self.th_min_slider.setMaximum(255)
        self.th_min_slider.setValue(127)

        self.detection_layout.addWidget(self.detection)
        self.detection_layout.addRow(
            self.th_min_label,
            self.th_min_slider)

        self.controls_layout.addWidget(self.detection_group)

        self.minArea = QDoubleSpinBox()
        self.minArea.setMinimum(0)
        self.minArea.setMaximum(1024)
        self.minArea.setSingleStep(0.05)
        self.minArea.setValue(1.5)
        self.minArea.valueChanged.connect(self.slider_changed)

        self.maxArea = QDoubleSpinBox()
        self.maxArea.setMinimum(0)
        self.maxArea.setMaximum(1024)
        self.maxArea.setSingleStep(0.05)
        self.maxArea.setValue(80.0)
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

        self.detection_layout.addRow(
            QLabel('Min area:'),
            self.minArea)
        self.detection_layout.addRow(
            QLabel('Max area:'),
            self.maxArea)
        # self.controls_layout.addWidget(self.minCircularity)
        # self.controls_layout.addWidget(self.minConvexity)
        # self.controls_layout.addWidget(self.minInertiaRatio)

        self.bandpassFilter = BandpassFilterWidget()
        self.bandpassFilter.update.connect(lambda: self.update_display())
        self.controls_layout.addWidget(self.bandpassFilter)

        self.pages_slider.valueChanged.connect(self.slider_changed)
        self.min_slider.valueChanged.connect(self.slider_changed)
        self.max_slider.valueChanged.connect(self.slider_changed)
        self.autostretch.stateChanged.connect(self.slider_changed)
        self.th_min_slider.valueChanged.connect(self.slider_changed)
        self.detection.stateChanged.connect(self.slider_changed)

        self.controls_layout.addStretch()

        # Localization / Render layout
        self.fitting_cbox = QComboBox()
        self.fitting_cbox.addItem(
            '2D Phasor-Fit (CPU)', FittingMethod._2D_Phasor_CPU)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit (CPU)', FittingMethod._2D_Gauss_MLE_CPU)

        self.render_cbox = QComboBox()
        self.render_cbox.addItem('2D Gaussian Histogram')

        self.frc_cbox = QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Check Pattern')

        self.export_options = Checklist(
                'Exported Columns',
                ['Frame', 'Coordinates [Pixel]', 'Coordinates [nm]',
                 'Intensity', 'Super-res image', 'Track ID',
                 'Next NN Distance', 'Number of Merged NNs'], checked=True)

        self.export_precision = QLineEdit('%10.5f')

        self.px_size = QDoubleSpinBox()
        self.px_size.setMinimum(0)
        self.px_size.setMaximum(10000)
        self.px_size.setValue(117.5)

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

        self.nn_layout = QHBoxLayout()
        self.nneigh_btn = QPushButton(
            'Nearest-neighbour',
            clicked=lambda: self.nneigh())
        self.merge_btn = QPushButton(
            'Merge Tracks',
            clicked=lambda: self.merge())
        self.nneigh_merge_btn = QPushButton(
            'NM + Merging',
            clicked=lambda: self.nneigh_merge())

        self.drift_fdm_btn = QPushButton(
            'Fiducial marker drift correction',
            clicked=lambda: self.drift_fdm())

        self.nn_layout.addWidget(self.nneigh_btn)
        self.nn_layout.addWidget(self.merge_btn)
        self.nn_layout.addWidget(self.nneigh_merge_btn)

        self.im_exp_layout = QHBoxLayout()
        self.export_loc_btn = QPushButton(
            'Export Localizations',
            clicked=lambda: self.export_loc())
        self.import_loc_btn = QPushButton(
            'Import Localizations',
            clicked=lambda: self.import_loc())

        self.im_exp_layout.addWidget(self.import_loc_btn)
        self.im_exp_layout.addWidget(self.export_loc_btn)

        self.loc_form.addRow(
            QLabel('Fitting:'),
            self.fitting_cbox
        )
        self.loc_form.addRow(
            QLabel('Rendering Method:'),
            self.render_cbox
        )
        self.loc_form.addRow(
            QLabel('Pixel-size [nm]:'),
            self.px_size
        )
        self.loc_form.addRow(
            QLabel('S-res pixel-size [nm]:'),
            self.super_px_size
        )
        self.loc_form.addRow(self.loc_btn)
        self.loc_form.addRow(self.refresh_btn)
        self.loc_form.addRow(
            QLabel('FRC Method:'),
            self.frc_cbox
        )
        self.loc_form.addRow(self.frc_res_btn)

        self.loc_form.addRow(
            QLabel('Drift X-Corr. (bins, pixelSize, upsampling):'))
        self.loc_form.addRow(self.drift_cross_args)
        self.loc_form.addRow(self.drift_cross_btn)
        self.loc_form.addRow(
            QLabel('NN (n-neighbor, max-distance, max-off, max-len):'))
        self.loc_form.addRow(self.nneigh_merge_args)
        self.loc_form.addRow(self.nn_layout)
        self.loc_form.addRow(self.drift_fdm_btn)
        self.loc_form.addRow(self.export_options)
        self.loc_form.addRow(
            QLabel('Format:'),
            self.export_precision)
        self.loc_form.addRow(self.im_exp_layout)

        # graphics layout
        # # A plot area (ViewBox + axes) for displaying the image
        self.uImage = None
        self.image = pg.ImageItem(np.zeros((256, 256)), axisOrder='row-major')
        self.image_plot = pg.ImageView(imageItem=self.image)
        self.image_plot.setLevels(0, 255)
        self.image_plot.setMinimumWidth(600)
        self.image_plot.ui.roiBtn.deleteLater()
        self.image_plot.ui.menuBtn.deleteLater()
        self.roi = pg.RectROI(
            [-8, 14], [6, 5],
            scaleSnap=True, translateSnap=True,
            movable=False)
        self.roi.addTranslateHandle([0, 0], [0.5, 0.5])
        self.image_plot.addItem(self.roi)
        self.roi.setZValue(100)
        self.roi.sigRegionChangeFinished.connect(self.slider_changed)
        self.roi.setVisible(False)

        # self.image_plot.setColorMap(pg.colormap.getFromMatplotlib('jet'))
        self.g_layout_widget.addWidget(self.image_plot)
        # Item for displaying image data

        self.show()
        self.center()

    def center(self):
        '''Centers the window within the screen.
        '''
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def centerROI(self):
        '''Centers the ROI and fits it to the image.
        '''
        image = self.tiffSeq_Handler.getSlice(self.pages_slider.value(), 0, 0)

        self.roi.setSize([image.shape[1], image.shape[0]])
        self.roi.setPos([0, 0])
        self.roi.maxBounds = QRectF(0, 0, image.shape[1], image.shape[0])

    def enableROI_changed(self, state):
        if self.enableROI.isChecked():
            self.roi.setVisible(True)
        else:
            self.roi.setVisible(False)

    def get_roi_txt(self):
        if self.enableROI.isChecked():
            return (' | ROI Pos. (' + '{:.0f}, {:.0f}), ' +
                    'Size ({:.0f}, {:.0f})/({:.3f} um, {:.3f} um)').format(
                    *self.roi.pos(), *self.roi.size(),
                    *(self.roi.size()*self.px_size.value() / 1000))

        return ''

    def get_roi_info(self):
        if self.enableROI.isChecked():
            origin = self.roi.pos()  # ROI (x,y)
            dim = self.roi.size()  # ROI (w,h)
            return origin, dim
        else:
            return None

    def status(self):
        # Statusbar time
        self.statusBar().showMessage(
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz') +
            self.get_roi_txt() +
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

            self.centerROI()

            # self.update_display()
            # self.genOME()
        else:
            index = self.model.index(i.row(), 0, i.parent())
            self.path = self.model.filePath(index)

            if self.tiffSeq_Handler is not None:
                self.tiffSeq_Handler.close()
                self.tiffSeq_Handler = None

            if self.path.endswith('.zarr'):
                self.tiffSeq_Handler = ZarrImageSequence(self.path)
                self.tiffSeq_Handler.open()

                self.pages_slider.setMaximum(
                        self.tiffSeq_Handler.shape[0] - 1)
                self.pages_slider.valueChanged.emit(0)

                self.centerROI()
            else:
                try:
                    self.tiffSequence = tf.TiffSequence(
                        self.path + '/' + self.imsq_pattern.text())
                except ValueError:
                    self.tiffSequence = None

                if self.tiffSequence is not None:
                    self.tiffSeq_Handler = TiffSeqHandler(self.tiffSequence)
                    self.tiffSeq_Handler.open()

                    self.pages_slider.setMaximum(
                        self.tiffSeq_Handler.__len__() - 1)
                    self.pages_slider.valueChanged.emit(0)

                    with tf.TiffFile(self.tiffSequence.files[0]) as fl:
                        if fl.is_ome:
                            ome = OME.from_xml(fl.ome_metadata)
                            self.metadataEditor.pop_OME_XML(ome)
                            self.px_size.setValue(
                                self.metadataEditor.px_size.value())

                    self.centerROI()

    def save_cropped_img(self):
        if self.tiffSeq_Handler is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Cropped Image",
            directory=self.path,
            filter="Zarr Files (*.zarr)")

        roiInfo = self.get_roi_info()

        def work_func():
            if roiInfo is not None:
                origin, dim = roiInfo

                ySlice = slice(int(origin[1]), int(origin[1] + dim[1]), 1)
                xSlice = slice(int(origin[0]), int(origin[0] + dim[0]), 1)

                saveZarrImage(
                    filename, self.tiffSeq_Handler,
                    ySlice=ySlice,
                    xSlice=xSlice
                )
            else:
                origin, dim = None, None

                saveZarrImage(
                    filename, self.tiffSeq_Handler)

        def done(results):
            self.saveCropped.setDisabled(False)

        self.worker = thread_worker(
            work_func,
            progress=False, z_stage=False)
        self.worker.signals.result.connect(done)
        # Execute
        self.saveCropped.setDisabled(True)
        self._threadpool.start(self.worker)

    def average_stack(self):
        if self.tiffSeq_Handler is not None:
            sum = np.array([page.asarray() for page in self.tiff.pages])
            avg = sum.mean(axis=0, dtype=np.float32)

            self.image.setImage(avg, autoLevels=False)

    def genOME(self):
        if self.tiffSeq_Handler is not None:
            frames = self.tiffSeq_Handler.shape[0]
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

        if self.pages_slider is not None:
            self.pages_label.setText(
                'Page: {:d}/{:d}'.format(
                    self.pages_slider.value() + 1,
                    self.pages_slider.maximum() + 1))
        if self.min_slider is not None:
            self.min_label.setText(
                'Min: {:d}/{:d}'.format(
                    self.min_slider.value(),
                    self.min_slider.maximum()))
        if self.max_slider is not None:
            self.max_label.setText(
                'Max: {:d}/{:d}'.format(
                    self.max_slider.value(),
                    self.max_slider.maximum()))

    def update_display(self, image=None):
        if image is None:
            image = self.tiffSeq_Handler.getSlice(
                self.pages_slider.value(), 0, 0)

        roiInfo = self.get_roi_info()
        if roiInfo is not None:
            origin, dim = roiInfo
        else:
            origin, dim = None, None

        if self.tempMedianFilter.enabled.isChecked():
            frames = self.tempMedianFilter.filter.getFrames(
                self.pages_slider.value(), self.tiffSeq_Handler)

            image = self.tempMedianFilter.filter.run(image, frames, roiInfo)

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

            if roiInfo is not None:
                origin = self.roi.pos()  # ROI (x,y)
                dim = self.roi.size()  # ROI (w,h)
                img = self.uImage._view[
                    int(origin[1]):int(origin[1] + dim[1]),
                    int(origin[0]):int(origin[0] + dim[0])]
            else:
                origin = None
                dim = None
                img = self.uImage._view

            # bandpass filter
            img = self.bandpassFilter.filter.run(img)

            _, th_img = cv2.threshold(
                img,
                self.th_min_slider.value(),
                255,
                cv2.THRESH_BINARY)

            cv2.namedWindow("Keypoints_TH", cv2.WINDOW_NORMAL)
            cv2.imshow("Keypoints_TH", th_img)
            # print(ex, end='\r')
            # Detect blobs.

            keypoints = self.get_blob_detector()[0].detect(th_img)

            im_with_keypoints = cv2.drawKeypoints(
                img, keypoints, np.array([]),
                (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Show keypoints
            cv2.namedWindow("Keypoints_CV", cv2.WINDOW_NORMAL)
            cv2.imshow("Keypoints_CV", im_with_keypoints)

            points = cv2.KeyPoint_convert(keypoints)

            if len(points) > 0 and origin is not None:
                points[:, 0] += origin[0]
                points[:, 1] += origin[1]

            # method
            method = self.fitting_cbox.currentData()

            if method == FittingMethod._2D_Phasor_CPU:
                sub_fit = phasor_fit(image, points, False)
            elif method == FittingMethod._2D_Gauss_MLE_CPU:
                sub_fit = gaussian_2D_fit(image, points)

            if sub_fit is not None:

                keypoints = [cv2.KeyPoint(*point, size=1.0) for
                             point in sub_fit[:, :2]]

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

    def get_blob_detector_params(self) -> cv2.SimpleBlobDetector_Params:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.filterByColor = True
        params.blobColor = 255

        params.minDistBetweenBlobs = 0

        # Change thresholds
        params.minThreshold = float(self.th_min_slider.value())
        params.maxThreshold = 255

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
        return params

    def get_blob_detector(self) \
            -> tuple[cv2.SimpleBlobDetector, cv2.SimpleBlobDetector_Params]:
        params = self.get_blob_detector_params()
        return get_blob_detector(params), params

    def FRC_estimate(self):
        frc_method = self.frc_cbox.currentText()
        time = QDateTime.currentDateTime()
        if 'Check' in frc_method:
            img = self.update_loc()

            if img is not None:
                def work_func():
                    try:
                        return FRC_resolution_check_pattern(
                            img, self.super_px_size.value())
                    except Exception:
                        traceback.print_exc()
                        return None

                def done(results):
                    plotFRC_(*results)
                    self.frc_res_btn.setDisabled(False)
            else:
                return
        elif 'Binomial' in frc_method:
            if self.fittingResults is None:
                return
            elif len(self.fittingResults) > 0:
                data = self.fittingResults.toRender()

                def work_func():
                    try:
                        return FRC_resolution_binomial(
                            np.c_[data[0], data[1], data[2]],
                            self.super_px_size.value()
                        )
                    except Exception:
                        traceback.print_exc()
                        return None

                def done(results):
                    plotFRC(*results)
                    self.frc_res_btn.setDisabled(False)
            else:
                return
        else:
            return

        self.worker = thread_worker(
            work_func,
            progress=False, z_stage=False)
        self.worker.signals.result.connect(done)
        # Execute
        self.frc_res_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def drift_cross(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                try:
                    return self.fittingResults.drift_cross_correlation(
                        self.drift_cross_bins.value(),
                        self.drift_cross_px.value(),
                        self.drift_cross_up.value(),
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                if results is not None:
                    img_norm = cv2.normalize(
                        results[1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    self.image.setImage(img_norm, autoLevels=False)
                    self.fittingResults = results[0]
                    plot_drift(*results[2])
                self.drift_cross_btn.setDisabled(False)

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.drift_cross_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def drift_fdm(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                try:
                    return self.fittingResults.drift_fiducial_marker()
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                if results is not None:
                    self.fittingResults = results[0]
                    plot_drift(*results[1])
                    self.update_loc()
                self.drift_fdm_btn.setDisabled(False)

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.drift_fdm_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def nneigh_merge(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                try:
                    return self.fittingResults.nearest_neighbour_merging(
                        self.nn_max_distance.value(),
                        self.nn_max_off.value(),
                        self.nn_max_length.value(),
                        self.nn_neighbors.value()
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.fittingResults = results
                self.nneigh_merge_btn.setDisabled(False)

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.nneigh_merge_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def nneigh(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                try:
                    return self.fittingResults.nn_trajectories(
                        self.nn_max_distance.value(),
                        self.nn_max_off.value(),
                        self.nn_neighbors.value()
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.fittingResults = results
                self.nneigh_btn.setDisabled(False)

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.nneigh_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def merge(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                try:
                    return self.fittingResults.merge_tracks(
                        self.nn_max_length.value()
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.fittingResults = results
                self.merge_btn.setDisabled(False)

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.merge_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        else:
            return

    def import_loc(self):
        '''Imports fitting results from a (*.tsv) file.
        '''
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import localizations", filter="TSV Files (*.tsv)")

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
        '''Exports the fitting results into a (*.tsv) file.

        Parameters
        ----------
        filename : str, optional
            file path if None a save file dialog is shown, by default None
        '''
        if self.fittingResults is None:
            return

        if filename is None:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export localizations", filter="TSV Files (*.tsv)")

        if len(filename) > 0:
            options = self.export_options.toList()

            exp_columns = [
                'Frame' in options,
                'Coordinates [Pixel]' in options,
                'Coordinates [Pixel]' in options,
                'Coordinates [nm]' in options,
                'Coordinates [nm]' in options,
                'Coordinates [nm]' in options,
                'Intensity' in options,
                'Track ID' in options,
                'Next NN Distance' in options,
                'Number of Merged NNs' in options
            ]

            dataFrame = self.fittingResults.dataFrame()
            dataFrame.to_csv(
                filename, index=False,
                columns=FittingResults.columns[exp_columns],
                float_format=self.export_precision.text(),
                sep='\t',
                encoding='utf-8')

            if 'Super-res image' in options:
                sres_img = self.update_loc()
                tf.imsave(
                    filename.replace('.tsv', '_super_res.tif'),
                    sres_img,
                    photometric='minisblack',
                    append=True,
                    bigtiff=True,
                    ome=False)

    def localize(self):
        '''Initiates the localization main thread worker.
        '''
        if self.tiffSeq_Handler is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save localizations", filter="TSV Files (*.tsv)")

        if len(filename) > 0:
            def done(res):
                self.loc_btn.setDisabled(False)
            # Any other args, kwargs are passed to the run function
            self.worker = thread_worker(
                self.proccess_loc, filename,
                progress=True, z_stage=False)
            self.worker.signals.progress.connect(self.update_loc)
            self.worker.signals.result.connect(done)
            # Execute
            self.loc_btn.setDisabled(True)
            self._threadpool.start(self.worker)

    def update_loc(self):
        '''Updates the rendered super-res image

        Returns
        -------
        ndarray | None
            rendered super-res image
        '''
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
        '''Extends the fitting results by results emitted
        by a thread worker.

        Parameters
        ----------
        result : np.ndarray
            [description]
        '''
        if result is not None:
            self.fittingResults.extend(result)
        self.thread_done += 1

    def proccess_loc(self, filename: str, progress_callback):
        '''Localization main thread worker function.

        Parameters
        ----------
        filename : str
            filename where the fitting results would be saved.
        progress_callback : func
            a progress callback emitted at a certain interval.
        '''

        # new instance of FittingResults
        self.fittingResults = FittingResults(
            ResultsUnits.Pixel,  # unit is pixels
            self.px_size.value()  # pixel projected size
        )

        self.thread_done = 0  # number of threads done
        start = QDateTime.currentDateTime()  # timer
        time = start

        # Filters + Blob detector params
        filter = self.bandpassFilter.filter
        self.bandpassFilter.arg_4.setChecked(False)
        tempEnabled = self.tempMedianFilter.enabled.isChecked()
        params = self.get_blob_detector_params()

        # method
        method = self.fitting_cbox.currentData()

        # ROI
        roiInfo = self.get_roi_info()
        self.enableROI.setChecked(False)

        # uses only n_threads - 2
        threads = self._threadpool.maxThreadCount() - 2
        print('Threads', threads)
        for i in range(
                0, int(
                    np.ceil(
                        self.tiffSeq_Handler.shape[0] / threads
                        ) * threads
                        ),
                threads):
            time = QDateTime.currentDateTime()
            workers = []
            self.thread_done = 0
            for k in range(threads):
                if i + k < len(self.tiffSeq_Handler):
                    img = self.tiffSeq_Handler.getSlice(
                        i + k, 0, 0)

                    temp = None
                    if tempEnabled:
                        temp = TemporalMedianFilter()
                        temp._temporal_window = \
                            self.tempMedianFilter.filter._temporal_window

                    worker = thread_worker(
                        self.localize_frame,
                        i + k,
                        img,
                        temp,
                        filter,
                        params,
                        roiInfo,
                        method,
                        progress=False, z_stage=False)
                    worker.signals.result.connect(self.update_lists)
                    workers.append(worker)
                    QThreadPool.globalInstance().start(worker)

            while self.thread_done < len(workers):
                QThread.msleep(10)

            exex = time.msecsTo(QDateTime.currentDateTime())
            duration = start.msecsTo(QDateTime.currentDateTime())

            print(
                'index: {:d}/{:d}, Time: {:d}  '.format(
                    i + len(workers), self.tiffSeq_Handler.shape[0], exex),
                end="\r")
            if (i // threads) % 40 == 0:
                progress_callback.emit(self)

        QThread.msleep(5000)

        self.export_loc(filename)

    def localize_frame(
            self, index, image: np.ndarray,
            temp: TemporalMedianFilter,
            filter: AbstractFilter,
            params: cv2.SimpleBlobDetector_Params,
            roiInfo,
            method=FittingMethod._2D_Phasor_CPU):

        filtered = image.copy()

        # apply the median filter
        if temp is not None:
            frames = temp.getFrames(index, self.tiffSeq_Handler)
            filtered = temp.run(image, frames, roiInfo)
        # else:
        #     filtered_img = None

        # crop image to ROI
        if roiInfo is not None:
            origin = roiInfo[0]  # ROI (x,y)
            dim = roiInfo[1]  # ROI (w,h)
            image = image[
                int(origin[1]):int(origin[1] + dim[1]),
                int(origin[0]):int(origin[0] + dim[0])]
            filtered = filtered[
                int(origin[1]):int(origin[1] + dim[1]),
                int(origin[0]):int(origin[0] + dim[0])]

            # if filtered_img is not None:
            #     filtered_img = filtered_img[
            #         int(origin[1]):int(origin[1] + dim[1]),
            #         int(origin[0]):int(origin[0] + dim[0])]

        results = localize_frame(
            index=index,
            image=image,
            filtered=filtered,
            filter=filter,
            params=params,
            threshold=self.th_min_slider.value(),
            method=method
        )

        if results is not None:
            if len(results) > 0 and roiInfo is not None:
                results[:, 0] += origin[0]
                results[:, 1] += origin[1]

        return results

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
        dirname = os.path.dirname(os.path.abspath(__file__))
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

        if sys.platform.startswith('win'):
            import ctypes
            myappid = u'samhitech.mircoEye.tiff_viewer'  # appid
            ctypes.windll.shell32.\
                SetCurrentProcessExplicitAppUserModelID(myappid)

        window = tiff_viewer(path)
        return app, window


if __name__ == '__main__':
    app, window = tiff_viewer.StartGUI()
    app.exec_()
