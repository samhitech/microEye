import os
import sys

import cv2
import numba as nb
from numba import cuda
import numpy as np
import pandas as pd
import pyqtgraph as pg
import qdarkstyle
import tifffile as tf
import zarr
from numba import cuda
from ome_types.model.ome import OME
from PyQt5.QtCore import *
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import *

if cuda.is_available(): 
    from .fitting.pyfit3Dcspline.mainfunctions import GPUmleFit_LM
else:
    def GPUmleFit_LM(*args):
        pass

from .checklist_dialog import Checklist
from .Filters import *
from .fitting import pyfit3Dcspline
from .fitting.fit import *
from .fitting.results import *
from .fitting.results_stats import resultsStatsWidget
from .metadata import MetadataEditor
from .Rendering import *
from .thread_worker import *
from .uImage import *
from microEye import fitting


class tiff_viewer(QMainWindow):

    def __init__(self, path=os.path.dirname(os.path.abspath(__package__))):
        super().__init__()
        self.title = 'microEye tiff viewer'
        self.left = 0
        self.top = 0
        self.width = 1280
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
        self.tabView.setMaximumWidth(450)

        # Graphical layout
        self.g_layout_widget = QTabWidget()

        # Add the two sub-main layouts
        self.main_layout.addWidget(self.tabView, 3)
        self.main_layout.addWidget(self.g_layout_widget, 4)

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

        # results stats tab layout
        self.data_filters = QWidget()
        self.data_filters_layout = QVBoxLayout()
        self.data_filters.setLayout(self.data_filters_layout)

        # Add Tabs
        self.tabView.addTab(self.file_tree, 'File system')
        self.tabView.addTab(self.controls_group, 'Prefit Options')
        self.tabView.addTab(self.loc_group, 'Fitting')
        self.tabView.addTab(
            self.data_filters, 'Data Filters')
        self.tabView.addTab(self.metadataEditor, 'Metadata')

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

        self.detection = QCheckBox('Enable Realtime localization.')
        self.detection.setChecked(False)

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
        self.image_control_layout.addWidget(self.detection)
        self.image_control_layout.addWidget(self.saveCropped)

        self.controls_layout.addLayout(
            self.image_control_layout)

        self.blobDetectionWidget = BlobDetectionWidget()
        self.blobDetectionWidget.update.connect(
            lambda: self.update_display())

        self.detection_method = QComboBox()
        # self.detection_method.currentIndexChanged.connect()
        self.detection_method.addItem(
            'OpenCV Blob Detection',
            self.blobDetectionWidget
        )

        self.doG_FilterWidget = DoG_FilterWidget()
        self.doG_FilterWidget.update.connect(
            lambda: self.update_display())
        self.bandpassFilterWidget = BandpassFilterWidget()
        self.bandpassFilterWidget.setVisible(False)
        self.bandpassFilterWidget.update.connect(
            lambda: self.update_display())

        self.image_filter = QComboBox()
        self.image_filter.addItem(
            'Difference of Gaussians',
            self.doG_FilterWidget)
        self.image_filter.addItem(
            'Fourier Bandpass Filter',
            self.bandpassFilterWidget)

        # displays the selected item
        def update_visibility(box: QComboBox):
            for idx in range(box.count()):
                box.itemData(idx).setVisible(
                    idx == box.currentIndex())

        self.detection_method.currentIndexChanged.connect(
            lambda: update_visibility(self.detection_method))
        self.image_filter.currentIndexChanged.connect(
            lambda: update_visibility(self.image_filter))

        self.image_control_layout.addRow(
            QLabel('Approx. Loc. Method:'),
            self.detection_method)
        self.image_control_layout.addRow(
            QLabel('Image filter:'),
            self.image_filter)

        self.th_min_label = QLabel('Relative threshold:')
        self.th_min_slider = QDoubleSpinBox()
        self.th_min_slider.setMinimum(0)
        self.th_min_slider.setMaximum(100)
        self.th_min_slider.setSingleStep(0.01)
        self.th_min_slider.setDecimals(3)
        self.th_min_slider.setValue(0.2)
        self.th_min_slider.valueChanged.connect(self.slider_changed)

        self.image_control_layout.addRow(
            self.th_min_label,
            self.th_min_slider)

        self.tempMedianFilter = TemporalMedianFilterWidget()
        self.tempMedianFilter.update.connect(lambda: self.update_display())
        self.controls_layout.addWidget(self.tempMedianFilter)

        self.controls_layout.addWidget(self.blobDetectionWidget)
        self.controls_layout.addWidget(self.doG_FilterWidget)
        self.controls_layout.addWidget(self.bandpassFilterWidget)

        self.pages_slider.valueChanged.connect(self.slider_changed)
        self.min_slider.valueChanged.connect(self.slider_changed)
        self.max_slider.valueChanged.connect(self.slider_changed)
        self.autostretch.stateChanged.connect(self.slider_changed)
        self.detection.stateChanged.connect(self.slider_changed)

        self.controls_layout.addStretch()

        # Localization / Render layout
        self.fitting_cbox = QComboBox()
        self.fitting_cbox.addItem(
            '2D Phasor-Fit (CPU)',
            FittingMethod._2D_Phasor_CPU)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit fixed sigma (GPU/CPU)',
            FittingMethod._2D_Gauss_MLE_fixed_sigma)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit free sigma (GPU/CPU)',
            FittingMethod._2D_Gauss_MLE_free_sigma)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit elliptical sigma (GPU/CPU)',
            FittingMethod._2D_Gauss_MLE_elliptical_sigma)
        self.fitting_cbox.addItem(
            '2D MLE Gauss-Fit cspline (GPU/CPU)',
            FittingMethod._3D_Gauss_MLE_cspline_sigma)

        self.render_cbox = QComboBox()
        self.render_cbox.addItem('2D Histogram', 0)
        self.render_cbox.addItem('2D Gaussian Histogram', 1)

        self.frc_cbox = QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Check Pattern')

        self.export_options = Checklist(
                'Exported Columns',
                ['Super-res image', ] + FittingResults.uniqueKeys(None),
                checked=True)

        self.export_precision = QLineEdit('%10.5f')

        self.px_size = QDoubleSpinBox()
        self.px_size.setMinimum(0)
        self.px_size.setMaximum(20000)
        self.px_size.setValue(117.5)

        self.super_px_size = QSpinBox()
        self.super_px_size.setMinimum(0)
        self.super_px_size.setMaximum(200)
        self.super_px_size.setValue(10)

        self.fit_roi_size = QSpinBox()
        self.fit_roi_size.setMinimum(7)
        self.fit_roi_size.setMaximum(99)
        self.fit_roi_size.setSingleStep(2)
        self.fit_roi_size.lineEdit().setReadOnly(True)
        self.fit_roi_size.setValue(13)

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
        self.nn_min_distance = QDoubleSpinBox()
        self.nn_min_distance.setMaximum(20000)
        self.nn_min_distance.setValue(0)
        self.nn_max_distance = QDoubleSpinBox()
        self.nn_max_distance.setMaximum(20000)
        self.nn_max_distance.setValue(30)
        self.nn_max_off = QSpinBox()
        self.nn_max_off.setValue(1)
        self.nn_max_length = QSpinBox()
        self.nn_max_length.setMaximum(20000)
        self.nn_max_length.setValue(500)
        self.nneigh_merge_args.addWidget(self.nn_neighbors)
        self.nneigh_merge_args.addWidget(self.nn_min_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_off)
        self.nneigh_merge_args.addWidget(self.nn_max_length)

        self.loc_btn = QPushButton(
            'Localize',
            clicked=lambda: self.localize())
        self.refresh_btn = QPushButton(
            'Refresh SuperRes Image',
            clicked=lambda: self.renderLoc())
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
        self.import_loc_btn = QPushButton(
            'Import',
            clicked=lambda: self.import_loc())
        self.export_loc_btn = QPushButton(
            'Export',
            clicked=lambda: self.export_loc())

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
            QLabel('Fitting roi-size [pixel]:'),
            self.fit_roi_size
        )
        self.loc_form.addRow(
            QLabel('Pixel-size [nm]:'),
            self.px_size
        )
        self.loc_form.addRow(
            QLabel('S-res pixel-size [nm]:'),
            self.super_px_size
        )
        self.loc_ref_lay = QHBoxLayout()
        self.loc_ref_lay.addWidget(self.loc_btn)
        self.loc_ref_lay.addWidget(self.refresh_btn)
        self.loc_form.addRow(self.loc_ref_lay)
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
            QLabel('NN (n-neighbor, min, max-distance, max-off, max-len):'))
        self.loc_form.addRow(self.nneigh_merge_args)
        self.loc_form.addRow(self.nn_layout)
        self.loc_form.addRow(self.drift_fdm_btn)
        self.loc_form.addRow(self.export_options)
        self.loc_form.addRow(
            QLabel('Format:'),
            self.export_precision)
        self.loc_form.addRow(self.im_exp_layout)

        # graphics layout
        # A plot area (ViewBox + axes) for displaying the image
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

        # results stats widget
        self.results_plot_scroll = QScrollArea()
        self.results_plot = resultsStatsWidget()
        self.results_plot.dataFilterUpdated.connect(
            self.filter_updated)
        self.results_plot_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results_plot_scroll.setWidgetResizable(True)
        self.results_plot_scroll.setWidget(self.results_plot)

        self.apply_filters_btn = QPushButton(
            'Apply Filters',
            clicked=lambda: self.apply_filters())
        self.apply_filters_btn.setToolTip(
            'Applies the filters permanently to fitting results.')

        self.data_filters_layout.addWidget(self.results_plot_scroll)
        self.data_filters_layout.addWidget(self.apply_filters_btn)

        # self.image_plot.setColorMap(pg.colormap.getFromMatplotlib('jet'))
        self.g_layout_widget.addTab(self.image_plot, 'Image Preview')
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

            self.image.setImage(avg, autoLevels=True)

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
        self.image.setImage(self.uImage._view, autoLevels=True)

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
            img = self.image_filter.currentData().filter.run(img)

            _, th_img = cv2.threshold(
                img,
                np.quantile(img, 1-1e-4) * self.th_min_slider.value(),
                255,
                cv2.THRESH_BINARY)

            cv2.namedWindow("Thresholded filtered Img.", cv2.WINDOW_NORMAL)
            cv2.imshow("Thresholded filtered Img.", th_img)
            # print(ex, end='\r')
            # Detect blobs.

            points, im_with_keypoints = self.detection_method.currentData()\
                .detector.find_peaks_preview(th_img, img)

            # Show keypoints
            cv2.namedWindow("Approx. Loc.", cv2.WINDOW_NORMAL)
            cv2.imshow("Approx. Loc.", im_with_keypoints)

            if len(points) > 0 and origin is not None:
                points[:, 0] += origin[0]
                points[:, 1] += origin[1]

            # method
            method = self.fitting_cbox.currentData()

            if method == FittingMethod._2D_Phasor_CPU:
                sub_fit = phasor_fit(image, points, False)

                if sub_fit is not None:

                    keypoints = [cv2.KeyPoint(*point, size=1.0) for
                                 point in sub_fit[:, :2]]

                    # Draw detected blobs as red circles.
                    im_with_keypoints = cv2.drawKeypoints(
                        self.uImage._view, keypoints, np.array([]),
                        (0, 0, 255),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    self.image.setImage(im_with_keypoints, autoLevels=True)
            else:
                rois, coords = pyfit3Dcspline.get_roi_list(image, points, 13)
                Parameters = None

                if method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 1, np.array([1]), None, 0)
                elif method == FittingMethod._2D_Gauss_MLE_free_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 2, np.array([1]), None, 0)
                elif method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 4, np.array([1]), None, 0)
                elif method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 5,
                            np.ones((64, 4, 4, 4), dtype=np.float32), None, 0)

                if Parameters is not None:
                    keypoints = [cv2.KeyPoint(
                        Parameters[idx, 0] + coords[idx, 0],
                        Parameters[idx, 1] + coords[idx, 1],
                        size=1.0) for idx in range(rois.shape[0])]

                    # Draw detected blobs as red circles.
                    im_with_keypoints = cv2.drawKeypoints(
                        self.uImage._view, keypoints, np.array([]),
                        (0, 0, 255),
                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    self.image.setImage(im_with_keypoints, autoLevels=True)

    def FRC_estimate(self):
        frc_method = self.frc_cbox.currentText()
        time = QDateTime.currentDateTime()
        if 'Check' in frc_method:
            img = self.renderLoc()

            if img is not None:
                def work_func():
                    try:
                        return FRC_resolution_check_pattern(
                            img, self.super_px_size.value())
                    except Exception:
                        traceback.print_exc()
                        return None

                def done(results):
                    self.frc_res_btn.setDisabled(False)
                    if results is not None:
                        plotFRC_(*results)
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
                    self.frc_res_btn.setDisabled(False)
                    if results is not None:
                        plotFRC(*results)
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
                self.drift_cross_btn.setDisabled(False)
                if results is not None:
                    self.renderLoc()
                    self.fittingResults = results[0]
                    self.results_plot.setData(
                        self.fittingResults.dataFrame())
                    plot_drift(*results[2])

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
                self.drift_fdm_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results[0]
                    self.results_plot.setData(
                        self.fittingResults.dataFrame())
                    plot_drift(*results[1])
                    self.renderLoc()

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
                        self.nn_min_distance.value(),
                        self.nn_max_distance.value(),
                        self.nn_max_off.value(),
                        self.nn_max_length.value(),
                        self.nn_neighbors.value()
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.nneigh_merge_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(
                        self.fittingResults.dataFrame())

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
                        self.nn_min_distance.value(),
                        self.nn_max_distance.value(),
                        self.nn_max_off.value(),
                        self.nn_neighbors.value()
                    )
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.nneigh_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(
                        self.fittingResults.dataFrame())

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
                self.merge_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(
                        self.fittingResults.dataFrame())

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
                self.results_plot.setData(results.dataFrame())
                self.renderLoc()
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

            dataFrame = self.fittingResults.dataFrame()
            exp_columns = []
            for col in dataFrame.columns:
                if col in options:
                    exp_columns.append(col)

            dataFrame.to_csv(
                filename, index=False,
                columns=exp_columns,
                float_format=self.export_precision.text(),
                sep='\t',
                encoding='utf-8')

            if 'Super-res image' in options:
                sres_img = self.renderLoc()
                tf.imsave(
                    filename.replace('.tsv', '_super_res.tif'),
                    sres_img,
                    photometric='minisblack',
                    append=True,
                    bigtiff=True,
                    ome=False)

    def apply_filters(self):
        if self.results_plot.filtered is not None:
            self.fittingResults = FittingResults.fromDataFrame(
                self.results_plot.filtered, 1)
            self.results_plot.setData(self.fittingResults.dataFrame())
            self.renderLoc()
            print('Filters applied.')

    def localize(self):
        '''Initiates the localization main thread worker.
        '''
        if self.tiffSeq_Handler is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save localizations", filter="TSV Files (*.tsv)")

        if len(filename) > 0:

            if self.fitting_cbox.currentData() == \
                    FittingMethod._2D_Phasor_CPU or not cuda.is_available():
                def done(res):
                    self.loc_btn.setDisabled(False)
                    if res is not None:
                        self.results_plot.setData(
                            self.fittingResults.dataFrame())
                print('\nCPU Fit')
                # Any other args, kwargs are passed to the run function
                self.worker = thread_worker(
                    self.localizeStackCPU, filename,
                    progress=True, z_stage=False)
                self.worker.signals.progress.connect(self.renderLoc)
                self.worker.signals.result.connect(done)
                # Execute
                self.loc_btn.setDisabled(True)
                self._threadpool.start(self.worker)
            else:
                def done(res):
                    self.loc_btn.setDisabled(False)
                    if res is not None:
                        self.fittingResults.extend(res)
                        self.export_loc(filename)
                        self.results_plot.setData(
                            self.fittingResults.dataFrame())
                print('\nGPU Fit')
                # Any other args, kwargs are passed to the run function
                # self.localizeStackGPU(filename, None)
                self.worker = thread_worker(
                    self.localizeStackGPU, filename,
                    progress=True, z_stage=False)
                self.worker.signals.progress.connect(self.renderLoc)
                self.worker.signals.result.connect(done)
                # Execute
                self.loc_btn.setDisabled(True)
                self._threadpool.start(self.worker)

    def renderLoc(self):
        '''Updates the rendered super-res image

        Returns
        -------
        ndarray | None
            rendered super-res image
        '''
        if self.fittingResults is None:
            return None
        elif len(self.fittingResults) > 0:
            render_idx = self.render_cbox.currentData()
            if render_idx == 0:
                renderClass = hist2D_render(self.super_px_size.value())
            elif render_idx == 1:
                renderClass = gauss_hist_render(self.super_px_size.value())
            img = renderClass.render(
                *self.fittingResults.toRender())
            # img_norm = cv2.normalize(
            #     img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            self.image.setImage(img, autoLevels=True)
            return img
        else:
            return None

    def filter_updated(self, df: pd.DataFrame):
        if df is not None:
            if df.count()[0] > 1:
                render_idx = self.render_cbox.currentData()
                if render_idx == 0:
                    renderClass = hist2D_render(self.super_px_size.value())
                elif render_idx == 1:
                    renderClass = gauss_hist_render(self.super_px_size.value())
                img = renderClass.render(
                    df['x'].to_numpy(),
                    df['y'].to_numpy(),
                    df['I'].to_numpy())
                # img_norm = cv2.normalize(
                #     img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                self.image.setImage(img, autoLevels=False)
            else:
                # Create a black image
                img = np.zeros(
                    (self.image.height(), self.image.width(), 3),
                    np.uint8)

                # Write some Text
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 50)
                fontScale = 1
                fontColor = (255, 255, 255)
                thickness = 1
                lineType = 2

                cv2.putText(
                    img, 'EMPTY!',
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                self.image.setImage(img, autoLevels=False)

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

    def localizeStackCPU(self, filename: str, progress_callback):
        '''CPU Localization main thread worker function.

        Parameters
        ----------
        filename : str
            filename where the fitting results would be saved.
        progress_callback : func
            a progress callback emitted at a certain interval.
        '''
        # method
        method = self.fitting_cbox.currentData()

        # new instance of FittingResults
        self.fittingResults = FittingResults(
            ResultsUnits.Pixel,  # unit is pixels
            self.px_size.value(),  # pixel projected size
            method
        )

        self.thread_done = 0  # number of threads done
        start = QDateTime.currentDateTime()  # timer
        time = start

        # Filters + Blob detector params
        filter = self.image_filter.currentData().filter
        tempEnabled = self.tempMedianFilter.enabled.isChecked()
        detector = self.detection_method.currentData().detector

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
                        pre_localize_frame,
                        i + k,
                        self.tiffSeq_Handler,
                        img,
                        None,
                        temp,
                        filter,
                        detector,
                        roiInfo,
                        self.th_min_slider.value(),
                        self.fit_roi_size.value(),
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

    def localizeStackGPU(self, filename: str, progress_callback):
        '''CPU Localization main thread worker function.

        Parameters
        ----------
        filename : str
            filename where the fitting results would be saved.
        progress_callback : func
            a progress callback emitted at a certain interval.
        '''
        # method
        method = self.fitting_cbox.currentData()

        # new instance of FittingResults
        self.fittingResults = FittingResults(
            ResultsUnits.Pixel,  # unit is pixels
            self.px_size.value(),  # pixel projected size
            method
        )

        print('\nCollecting Prefit ROIs...')
        start = QDateTime.currentDateTime()  # timer

        # Filters + Blob detector params
        filter = self.image_filter.currentData().filter
        tempEnabled = self.tempMedianFilter.enabled.isChecked()
        detector = self.detection_method.currentData().detector

        # Vars
        roiSize = self.fit_roi_size.value()
        rel_threshold = self.th_min_slider.value()
        varim = None
        PSFparam = np.array([1.5])

        roi_list = []
        varim_list = []
        coord_list = []
        frames_list = []

        # ROI
        roiInfo = self.get_roi_info()
        self.enableROI.setChecked(False)

        for k in range(len(self.tiffSeq_Handler)):
            cycle = QDateTime.currentDateTime()
            image = self.tiffSeq_Handler.getSlice(k, 0, 0)

            temp = None
            if tempEnabled:
                temp = TemporalMedianFilter()
                temp._temporal_window = \
                    self.tempMedianFilter.filter._temporal_window

            filtered = image.copy()

            # apply the median filter
            if temp is not None:
                frames = temp.getFrames(k, self.tiffSeq_Handler)
                filtered = temp.run(image, frames, roiInfo)

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

            uImg = uImage(filtered)

            uImg.equalizeLUT(None, True)

            if filter is BandpassFilter:
                filter._show_filter = False
                filter._refresh = False

            img = filter.run(uImg._view)

            # Detect blobs.
            _, th_img = cv2.threshold(
                    img,
                    np.quantile(img, 1-1e-4) * rel_threshold,
                    255,
                    cv2.THRESH_BINARY)

            points: np.ndarray = detector.find_peaks(th_img)

            if varim is None:
                rois, coords = get_roi_list(image, points, roiSize)
            else:
                rois, varims, coords = get_roi_list_CMOS(
                    image, varim, points, roiSize)
                varim_list += [varims]

            roi_list += [rois]
            coord_list += [coords]
            frames_list += [k + 1] * rois.shape[0]

            print(
                'index: {:.2f}% {:d} ms    '.format(
                    100*(k+1)/len(self.tiffSeq_Handler),
                    cycle.msecsTo(QDateTime.currentDateTime())),
                end="\r")

        roi_list = np.vstack(roi_list)
        coord_list = np.vstack(coord_list)
        if varim is None:
            varim_list = None
        else:
            varim_list = np.vstack(varim_list)

        print(
            '\n',
            start.msecsTo(QDateTime.currentDateTime()),
            ' ms')

        if method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
            params, crlbs, loglike = GPUmleFit_LM(
                roi_list, 1, PSFparam, varim_list, 0)
        elif method == FittingMethod._2D_Gauss_MLE_free_sigma:
            params, crlbs, loglike = GPUmleFit_LM(
                roi_list, 2, PSFparam, varim_list, 0)
        elif method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
            params, crlbs, loglike = GPUmleFit_LM(
                roi_list, 4, PSFparam, varim_list, 0)
        elif method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
            params, crlbs, loglike = GPUmleFit_LM(
                roi_list, 5, PSFparam, varim_list, 0)

        params = params.astype(np.float64, copy=False)
        crlbs = crlbs.astype(np.float64, copy=False)
        loglike = loglike.astype(np.float64, copy=False)
        frames_list = np.array(frames_list, dtype=np.int64)

        if params is not None:
            params[:, :2] += np.array(coord_list)
            if len(params) > 0 and roiInfo is not None:
                params[:, 0] += origin[0]
                params[:, 1] += origin[1]

        print(
            '\nDone... ',
            start.msecsTo(QDateTime.currentDateTime()) / 1000,
            ' s')

        return frames_list, params, crlbs, loglike

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
