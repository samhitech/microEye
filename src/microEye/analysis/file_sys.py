import json
import os
import sys
import webbrowser
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pyqtgraph as pg
import qdarkstyle
import tifffile as tf
from numba import cuda
from ome_types.model import OME
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *

from ..shared import start_gui

if cuda.is_available():
    from .fitting.pyfit3Dcspline.mainfunctions import GPUmleFit_LM
else:
    def GPUmleFit_LM(*args):
        pass

from ..shared.gui_helper import *
from ..shared.metadata import MetadataEditor
from ..shared.thread_worker import *
from ..shared.uImage import *
from .checklist_dialog import ChecklistDialog
from .cmosMaps import cmosMaps
from .filters import *
from .fitting import pyfit3Dcspline
from .fitting.fit import *
from .fitting.nena import NeNA_Widget
from .fitting.results import *
from .fitting.results_stats import resultsStatsWidget
from .fitting.tardis import TARDIS_Widget
from .rendering import *
from .tools.kymograms import (
    Kymogram,
    KymogramWidget,
)


class DockKeys(Enum):
    FILE_SYSTEM = 'File System'
    PREFIT_OPTIONS = 'Prefit Options'
    IMAGE_TOOLS = 'Image Processing Tools'
    SMLM_ANALYSIS = 'SMLM Analysis'
    CMOS_MAPS = 'CMOS Maps'
    DATA_FILTERS = 'Data Filters'
    METADATA = 'Metadata'


class tiff_viewer(QMainWindow):

    def __init__(self, path=None):
        super().__init__()
        # Set window properties
        self.title = 'microEye tiff viewer'
        self.left = 0
        self.top = 0
        self._width = 1300
        self._height = 650
        self._zoom = 1
        self._n_levels = 4096

        # Initialize variables
        self.fittingResults = None
        self.tiff = None
        self.tiffSequence = None
        self.stack_handler = None

        # Threading
        self._threadpool = QThreadPool.globalInstance()
        print('Multithreading with maximum %d threads'
              % self._threadpool.maxThreadCount())

        # Set the path
        if path is None:
            path = os.path.dirname(os.path.abspath(__package__))
        self.initialize(path)

        # Set up the status bar
        self.status()

        # Status Bar Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.status)
        self.timer.start(10)

        # Set main window properties
        self.setStatusBar(self.statusBar())

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

    def initialize(self, path):
        # Set Title / Dimensions / Center Window
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self._width, self._height)

        # Define main window layout
        self.setupMainWindowLayout()

        # Initialize the file system model / tree
        self.setupFileSystemTab(path)

        # Creating the Prefit Options tab layout
        self.setupPrefitOptionsTab()

        # Creating the Image Stack Processing Tools tab layout
        self.setupImageToolsTab()

        # Localization / Render tab layout
        self.setupLocalizationTab()

        # CMOS maps tab
        self.setupCMOSMapsTab()

        # Results stats tab layout
        self.setupDataFiltersTab()

        # Metadata Viewer / Editor
        self.setupMetadataTab()

        # Tabify docks
        self.tabifyDocks()

        # Set tab positions
        self.setTabPositions()

        # Raise docks
        self.raiseDocks()

        # Create menu bar
        self.createMenuBar()

        self.show()
        self.center()

    def setupMainWindowLayout(self):
        self.main_widget = QWidget()
        self.main_layout = QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # Graphical layout
        self.g_layout_widget = QTabWidget()

        # graphics layout
        # A plot area (ViewBox + axes) for displaying the image
        self.uImage = None
        self.imageWidget = pg.GraphicsLayoutWidget()
        self.imageWidget.setMinimumWidth(600)

        # Create the ViewBox
        self.vb : pg.ViewBox = self.imageWidget.addViewBox(row=0, col=0)

        # Create the ImageItem and set its view to self.vb
        self.image = pg.ImageItem(
            np.zeros((256, 256)), axisOrder='row-major', view=self.vb)

        # Add the ImageItem to the ViewBox
        self.vb.addItem(self.image)
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)

        self.hist = pg.HistogramLUTItem(
            gradientPosition='bottom', orientation='horizontal')
        self.hist.setImageItem(self.image)
        self.imageWidget.addItem(self.hist, row=1, col=0)
        # self.image_plot = pg.ImageView(imageItem=self.image)
        # self.image_plot.setLevels(0, 255)
        # self.image_plot.ui.histogram.hide()
        # self.image_plot.ui.roiBtn.deleteLater()
        # self.image_plot.ui.menuBtn.deleteLater()
        self.roi = pg.RectROI(
            [-8, 14], [6, 5],
            scaleSnap=True, translateSnap=True,
            movable=False)
        self.roi.addTranslateHandle([0, 0], [0.5, 0.5])
        self.vb.addItem(self.roi)
        self.roi.setZValue(100)
        self.roi.sigRegionChangeFinished.connect(self.slider_changed)
        self.roi.setVisible(False)


        # self.image_plot.setColorMap(pg.colormap.getFromMatplotlib('jet'))
        self.g_layout_widget.addTab(self.imageWidget, 'Image Preview')

        # Add the two sub-main layouts
        self.main_layout.addWidget(self.g_layout_widget, 1)

        self.docks : dict[str, QDockWidget] = {}  # Dictionary to store created docks
        self.layouts = {}

    def setupFileSystemTab(self, path):
        # Tiff File system tree viewer tab layout
        self.file_tree_layout = self.create_tab(
            DockKeys.FILE_SYSTEM, QVBoxLayout,
            'LeftDockWidgetArea', widget=None)

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

        self.tree.setWindowTitle('Dir View')
        self.tree.resize(512, 256)

        # Add the File system tab contents
        self.imsq_pattern = QLineEdit('/image_0*.ome.tif')

        self.file_tree_layout.addWidget(QLabel('Image Sequence pattern:'))
        self.file_tree_layout.addWidget(self.imsq_pattern)
        self.file_tree_layout.addWidget(self.tree)

    def setupPrefitOptionsTab(self):
        # Creating the Prefit Options tab layout
        self.prefit_options_layout = self.create_tab(
            DockKeys.PREFIT_OPTIONS, QVBoxLayout,
            'LeftDockWidgetArea', widget=None)

        self.image_control_layout = QFormLayout()

        self.pages_label = QLabel('Pages')
        self.pages_slider = QSlider(Qt.Horizontal)
        self.pages_slider.setMinimum(0)
        self.pages_slider.setMaximum(0)

        # Hist plotWidget
        self.histogram = pg.PlotWidget()
        self.hist_cdf = pg.PlotWidget()
        green_pen = pg.mkPen(color='g')
        blue_pen = pg.mkPen(color='b')
        green_brush = pg.mkBrush(0, 255, 0, 32)
        _bins = np.arange(self._n_levels - 1)
        self._plot_ref = self.histogram.plot(
            _bins, np.ones_like(_bins), pen=blue_pen)

        self.lr_0 = pg.LinearRegionItem(
            (0, self._n_levels - 1),
            bounds=(0, self._n_levels - 1),
            pen=green_pen, brush=green_brush,
            movable=True, swapMode='push', span=(0.0, 1))
        self.histogram.addItem(self.lr_0)

        self.autostretch = create_check_box(
            'Auto-Stretch', initial_state=True,
            state_changed_slot=self.slider_changed)

        self.enable_roi = create_check_box(
            'Enable ROI', initial_state=False,
            state_changed_slot=self.enable_roi_changed)

        self.detection = create_check_box(
            'Realtime localization.', initial_state=False,
            state_changed_slot=self.slider_changed)

        check_boxes_layout = QHBoxLayout()

        self.save_cropped = QPushButton(
            'Save Cropped Image',
            clicked=lambda: self.save_cropped_img())

        self.image_control_layout.addRow(
            self.pages_label,
            self.pages_slider)
        self.image_control_layout.addRow(
            self.histogram)
        check_boxes_layout.addWidget(self.autostretch)
        check_boxes_layout.addWidget(self.enable_roi)
        check_boxes_layout.addWidget(self.detection)
        self.image_control_layout.addRow(check_boxes_layout)
        self.image_control_layout.addWidget(self.save_cropped)

        self.prefit_options_layout.addLayout(
            self.image_control_layout)

        self.blob_detection_widget = BlobDetectionWidget()
        self.blob_detection_widget.update.connect(
            lambda: self.update_display())

        self.detection_method = QComboBox()
        # self.detection_method.currentIndexChanged.connect()
        self.detection_method.addItem(
            'OpenCV Blob Detection',
            self.blob_detection_widget
        )

        self.do_g_filter_widget = DoG_FilterWidget()
        self.do_g_filter_widget.update.connect(
            lambda: self.update_display())
        self.bandpass_filter_widget = BandpassFilterWidget()
        self.bandpass_filter_widget.setVisible(False)
        self.bandpass_filter_widget.update.connect(
            lambda: self.update_display())

        self.image_filter = QComboBox()
        self.image_filter.addItem(
            'Difference of Gaussians',
            self.do_g_filter_widget)
        self.image_filter.addItem(
            'Fourier Bandpass Filter',
            self.bandpass_filter_widget)

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

        self.th_min_label = QLabel('Relative threshold (min/max):')

        self.th_min_slider = create_double_spin_box(
            max_value=1, initial_value=0.4, slot=self.slider_changed)
        self.th_max_slider = create_double_spin_box(
            max_value=1, initial_value=1, slot=self.slider_changed)

        self.image_control_layout.addRow(
            self.th_min_label,
            self.th_min_slider)
        self.image_control_layout.addWidget(
            self.th_max_slider)

        self.temp_median_filter = TemporalMedianFilterWidget()
        self.temp_median_filter.update.connect(lambda: self.update_display())
        self.prefit_options_layout.addWidget(self.temp_median_filter)

        self.prefit_options_layout.addWidget(self.blob_detection_widget)
        self.prefit_options_layout.addWidget(self.do_g_filter_widget)
        self.prefit_options_layout.addWidget(self.bandpass_filter_widget)

        self.pages_slider.valueChanged.connect(self.slider_changed)
        self.lr_0.sigRegionChangeFinished.connect(self.region_changed)

        self.prefit_options_layout.addStretch()

    def setupImageToolsTab(self):
        # Creating the Image Stack Processing Tools tab layout
        self.image_tools_layout = self.create_tab(
            DockKeys.IMAGE_TOOLS, QVBoxLayout,
            'LeftDockWidgetArea', widget=None)

        self.kymogram_widget = KymogramWidget()

        self.kymogram_widget.displayClicked.connect(self.kymogram_display_clicked)
        self.kymogram_widget.extractClicked.connect(self.kymogram_btn_clicked)

        self.image_tools_layout.addWidget(self.kymogram_widget)

    def setupLocalizationTab(self):
        # Localization / Render tab layout
        self.localization_form = self.create_tab(
            DockKeys.SMLM_ANALYSIS, QFormLayout,
            'RightDockWidgetArea', widget=None)

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

        self.px_size = create_double_spin_box(
            min_value=0, max_value=20000, initial_value=117.5)

        self.super_px_size = create_spin_box(
            min_value=0, max_value=200, initial_value=10)

        self.fit_roi_size = create_spin_box(
            min_value=7, max_value=99, single_step=2, initial_value=13)
        self.fit_roi_size.lineEdit().setReadOnly(True)

        self.loc_btn = QPushButton(
            'Localize',
            clicked=lambda: self.localize())
        self.refresh_btn = QPushButton(
            'Refresh SuperRes Image',
            clicked=lambda: self.renderLoc())
        self.loc_ref_lay = QHBoxLayout()
        self.loc_ref_lay.addWidget(self.loc_btn)
        self.loc_ref_lay.addWidget(self.refresh_btn)

        # Localization GroupBox
        localization = QGroupBox('Localization')
        flocalization = QFormLayout()
        localization.setLayout(flocalization)

        flocalization.addRow(
            QLabel('Fitting:'),
            self.fitting_cbox
        )
        flocalization.addRow(
            QLabel('Rendering Method:'),
            self.render_cbox
        )
        flocalization.addRow(
            QLabel('Fitting roi-size [pixel]:'),
            self.fit_roi_size
        )
        flocalization.addRow(
            QLabel('Pixel-size [nm]:'),
            self.px_size
        )
        flocalization.addRow(
            QLabel('S-res pixel-size [nm]:'),
            self.super_px_size
        )
        flocalization.addRow(self.loc_ref_lay)
        # End Localization GroupBox

        self.drift_cross_args = QHBoxLayout()

        self.drift_cross_bins = create_spin_box(
            initial_value=10)
        self.drift_cross_px = create_spin_box(
            initial_value=10)
        self.drift_cross_up = create_spin_box(
            min_value=0, max_value=1000, initial_value=100)

        self.drift_cross_args.addWidget(self.drift_cross_bins)
        self.drift_cross_args.addWidget(self.drift_cross_px)
        self.drift_cross_args.addWidget(self.drift_cross_up)

        self.drift_cross_btn = QPushButton(
            'Drift cross-correlation',
            clicked=lambda: self.drift_cross())
        self.drift_fdm_btn = QPushButton(
            'Fiducial marker drift correction',
            clicked=lambda: self.drift_fdm())

        # Drift GroupBox
        drift = QGroupBox('Drift Correction')
        fdrift = QFormLayout()
        drift.setLayout(fdrift)

        fdrift.addRow(
            QLabel('Drift X-Corr. (bins, pixelSize, upsampling):'))
        fdrift.addRow(self.drift_cross_args)
        fdrift.addRow(self.drift_cross_btn)
        fdrift.addRow(self.drift_fdm_btn)
        # End Drift GroupBox

        self.frc_cbox = QComboBox()
        self.frc_cbox.addItem('Binomial')
        self.frc_cbox.addItem('Odd/Even')
        self.frc_cbox.addItem('Halves')
        self.frc_res_btn = QPushButton(
            'FRC Resolution',
            clicked=lambda: self.FRC_estimate())

        self.NeNA_widget = None
        self.NeNA_btn = QPushButton(
            'NeNA Loc. Prec. Estimate',
            clicked=lambda: self.NeNA_estimate())
        self.tardis_btn = QPushButton(
            'TARDIS',
            clicked=lambda: self.TARDIS_analysis())

        # Precision GroupBox
        precision = QGroupBox('Loc. Precision')
        fprecision = QFormLayout()
        precision.setLayout(fprecision)

        fprecision.addRow(
            QLabel('FRC Method:'),
            self.frc_cbox
        )
        fprecision.addWidget(self.frc_res_btn)
        fprecision.addWidget(self.NeNA_btn)
        fprecision.addWidget(self.tardis_btn)
        # End Precision GroupBox

        self.nneigh_merge_args = QHBoxLayout()

        self.nn_neighbors = create_spin_box(
            max_value=20000, initial_value=1)
        self.nn_min_distance = create_double_spin_box(
            max_value=20000, initial_value=0)
        self.nn_max_distance = create_double_spin_box(
            max_value=20000, initial_value=30)
        self.nn_max_off = create_spin_box(
            max_value=20000, initial_value=1)
        self.nn_max_length = create_spin_box(
            max_value=20000, initial_value=500)

        self.nneigh_merge_args.addWidget(self.nn_neighbors)
        self.nneigh_merge_args.addWidget(self.nn_min_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_distance)
        self.nneigh_merge_args.addWidget(self.nn_max_off)
        self.nneigh_merge_args.addWidget(self.nn_max_length)

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

        self.nn_layout.addWidget(self.nneigh_btn)
        self.nn_layout.addWidget(self.merge_btn)
        self.nn_layout.addWidget(self.nneigh_merge_btn)

        # Precision GroupBox
        nearestN = QGroupBox('NN Analysis')
        fnearestN = QFormLayout()
        nearestN.setLayout(fnearestN)

        fnearestN.addRow(
            QLabel('NN (n-neighbor, min, max-distance, max-off, max-len):'))
        fnearestN.addRow(self.nneigh_merge_args)
        fnearestN.addRow(self.nn_layout)
        # End Precision GroupBox

        self.export_options = ChecklistDialog(
                'Exported Columns',
                ['Super-res image', ] + UNIQUE_COLUMNS,
                checked=True, parent=self)

        self.im_exp_layout = QHBoxLayout()
        self.import_loc_btn = QPushButton(
            'Import',
            clicked=lambda: self.import_loc())
        self.export_loc_btn = QPushButton(
            'Export',
            clicked=lambda: self.export_loc())

        self.im_exp_layout.addWidget(self.import_loc_btn)
        self.im_exp_layout.addWidget(self.export_loc_btn)

        self.localization_form.addRow(localization)
        self.localization_form.addRow(drift)
        self.localization_form.addRow(precision)
        self.localization_form.addRow(nearestN)

        self.localization_form.addRow(self.im_exp_layout)

    def setupCMOSMapsTab(self):
        # CMOS maps tab
        self.cmos_maps_group = cmosMaps()
        self.cmos_maps_layout = self.create_tab(
            DockKeys.CMOS_MAPS, None,
            'RightDockWidgetArea', widget=self.cmos_maps_group)

    def setupDataFiltersTab(self):
        # Results stats tab layout
        self.data_filters_layout = self.create_tab(
            DockKeys.DATA_FILTERS, QVBoxLayout, 'RightDockWidgetArea', widget=None)

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

    def setupMetadataTab(self):
        # Metadata Viewer / Editor
        self.metadata_editor = MetadataEditor()
        self.metadata_layout = self.create_tab(
            DockKeys.METADATA, None,
            'RightDockWidgetArea', visible=False, widget=self.metadata_editor)

    def tabifyDocks(self):
        # Tabify docks
        self.tabifyDockWidget(
            self.docks[DockKeys.FILE_SYSTEM],
            self.docks[DockKeys.PREFIT_OPTIONS])
        self.tabifyDockWidget(
            self.docks[DockKeys.FILE_SYSTEM],
            self.docks[DockKeys.IMAGE_TOOLS])
        self.tabifyDockWidget(
            self.docks[DockKeys.SMLM_ANALYSIS],
            self.docks[DockKeys.CMOS_MAPS])
        self.tabifyDockWidget(
            self.docks[DockKeys.SMLM_ANALYSIS],
            self.docks[DockKeys.DATA_FILTERS])
        self.tabifyDockWidget(
            self.docks[DockKeys.SMLM_ANALYSIS],
            self.docks[DockKeys.METADATA])

    def setTabPositions(self):
        self.setTabPosition(
            Qt.LeftDockWidgetArea, QTabWidget.East)
        self.setTabPosition(
            Qt.RightDockWidgetArea, QTabWidget.West)

    def raiseDocks(self):
        # Raise docks
        self.docks[DockKeys.FILE_SYSTEM].raise_()
        self.docks[DockKeys.SMLM_ANALYSIS].raise_()

    def createMenuBar(self):
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
        help_menu = menu_bar.addMenu('Help')

        # Create exit action
        save_config = QAction('Save Config.', self)
        save_config.triggered.connect(lambda: saveConfig(self))
        load_config = QAction('Load Config.', self)
        load_config.triggered.connect(lambda: loadConfig(self))

        github = QAction('microEye Github', self)
        github.triggered.connect(
            lambda: webbrowser.open('https://github.com/samhitech/microEye'))
        pypi = QAction('microEye PYPI', self)
        pypi.triggered.connect(
            lambda: webbrowser.open('https://pypi.org/project/microEye/'))

        # Add exit action to file menu
        file_menu.addAction(save_config)
        file_menu.addAction(load_config)

        # Create toggle view actions for each dock
        dock_toggle_actions = {}
        for key, dock in self.docks.items():
            toggle_action = dock.toggleViewAction()
            toggle_action.setEnabled(True)
            dock_toggle_actions[key] = toggle_action
            view_menu.addAction(toggle_action)

        help_menu.addAction(github)
        help_menu.addAction(pypi)

        qApp.installEventFilter(self)

        self.menuBar().hide()

    def create_tab(
        self,
        key: DockKeys,
        layout_type: Optional[type[Union[QVBoxLayout, QFormLayout]]] = None,
        dock_area: str = 'LeftDockWidgetArea',
        widget: Optional[QWidget] = None,
        visible: bool = True,
    ) -> Optional[type[Union[QVBoxLayout, QFormLayout]]]:
        '''
        Create a tab with a dock widget.

        Parameters
        ----------
        key : DockKeys
            The unique identifier for the tab.
        layout_type : Optional[Type[Union[QVBoxLayout, QFormLayout]]], optional
            The layout type for the group in the dock. If provided,
            widget should be None.
        dock_area : str, optional
            The dock area where the tab will be added.
        widget : Optional[QWidget], optional
            The widget to be placed in the dock. If provided,
            layout_type should be None.
        visible : bool, optional
            Whether the dock widget should be visible.

        Returns
        -------
        Optional[Type[Union[QVBoxLayout, QFormLayout]]]
            The layout if layout_type is provided, otherwise None.
        '''
        if widget:
            group = widget
        else:
            group = QWidget()
            group_layout = layout_type() if layout_type else QVBoxLayout()
            group.setLayout(group_layout)
            self.layouts[key] = group_layout

        dock = QDockWidget(str(key.value), self)
        dock.setFeatures(
            QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetMovable)
        dock.setWidget(group)
        dock.setVisible(visible)
        self.addDockWidget(getattr(Qt.DockWidgetArea, dock_area), dock)

        # Store the dock in the dictionary
        self.docks[key] = dock

        # Return the layout if layout_type is provided, otherwise None
        return None if widget else group_layout

    def center(self):
        '''Centers the window within the screen using setGeometry.'''
        # Get the screen geometry
        screen_geometry = QDesktopWidget().availableGeometry()

        # Calculate the center point
        center_point = screen_geometry.center()

        # Set the window geometry
        self.setGeometry(
            center_point.x() - self.width() / 2,
            center_point.y() - self.height() / 2,
            self.width(),
            self.height()
        )

    def centerROI(self):
        '''Centers the ROI and fits it to the image.
        '''
        image = self.stack_handler.getSlice(self.pages_slider.value(), 0, 0)

        self.roi.setSize([image.shape[1], image.shape[0]])
        self.roi.setPos([0, 0])
        self.roi.maxBounds = QRectF(0, 0, image.shape[1], image.shape[0])

    def enable_roi_changed(self, state):
        if self.enable_roi.isChecked():
            self.roi.setVisible(True)
        else:
            self.roi.setVisible(False)

    def get_roi_txt(self):
        if self.enable_roi.isChecked():
            return (' | ROI Pos. (' + '{:.0f}, {:.0f}), ' +
                    'Size ({:.0f}, {:.0f})/({:.3f} um, {:.3f} um)').format(
                    *self.roi.pos(), *self.roi.size(),
                    *(self.roi.size()*self.px_size.value() / 1000))

        return ''

    def get_roi_info(self):
        if self.enable_roi.isChecked():
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

    def set_maximum(self, frames):
        self.pages_slider.setMaximum(frames)
        self.kymogram_widget.setMaximum(frames)

    def _open_file(self, i):
        if not self.model.isDir(i):
            cv2.destroyAllWindows()
            index = self.model.index(i.row(), 0, i.parent())
            self.path = self.model.filePath(index)
            if self.stack_handler is not None:
                self.stack_handler.close()
            self.tiffSequence = tf.TiffSequence([self.path])
            self.stack_handler = TiffSeqHandler(self.tiffSequence)
            self.stack_handler.open()
            # self.tiffStore = self.tiffSequence.aszarr(axestiled={0: 0})
            # self.tiffZar = zarr.open(self.tiffStore, mode='r')

            self.set_maximum(len(self.stack_handler) - 1)
            self.pages_slider.valueChanged.emit(0)

            with tf.TiffFile(self.tiffSequence.files[0]) as fl:
                if fl.is_ome:
                    ome = OME.from_xml(fl.ome_metadata)
                    self.metadata_editor.pop_OME_XML(ome)
                    self.px_size.setValue(
                        self.metadata_editor.px_size.value())

            self.centerROI()

            # self.update_display()
            # self.genOME()
        else:
            index = self.model.index(i.row(), 0, i.parent())
            self.path = self.model.filePath(index)

            if self.stack_handler is not None:
                self.stack_handler.close()
                self.stack_handler = None

            if self.path.endswith('.zarr'):
                self.stack_handler = ZarrImageSequence(self.path)
                self.stack_handler.open()

                self.set_maximum(
                        self.stack_handler.shape[0] - 1)
                self.pages_slider.valueChanged.emit(0)

                self.centerROI()
            else:
                try:
                    self.tiffSequence = tf.TiffSequence(
                        self.path + '/' + self.imsq_pattern.text())
                except ValueError:
                    self.tiffSequence = None

                if self.tiffSequence is not None:
                    self.stack_handler = TiffSeqHandler(self.tiffSequence)
                    self.stack_handler.open()

                    self.set_maximum(
                        self.stack_handler.__len__() - 1)
                    self.pages_slider.valueChanged.emit(0)

                    with tf.TiffFile(self.tiffSequence.files[0]) as fl:
                        if fl.is_ome:
                            ome = OME.from_xml(fl.ome_metadata)
                            self.metadata_editor.pop_OME_XML(ome)
                            self.px_size.setValue(
                                self.metadata_editor.px_size.value())

                    self.centerROI()

    def save_cropped_img(self):
        if self.stack_handler is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save Cropped Image',
            directory=self.path,
            filter='Zarr Files (*.zarr)')

        roiInfo = self.get_roi_info()

        def work_func():
            try:
                if roiInfo is not None:
                    origin, dim = roiInfo

                    ySlice = slice(int(origin[1]), int(origin[1] + dim[1]), 1)
                    xSlice = slice(int(origin[0]), int(origin[0] + dim[0]), 1)

                    saveZarrImage(
                        filename, self.stack_handler,
                        ySlice=ySlice,
                        xSlice=xSlice
                    )
                else:
                    origin, dim = None, None

                    saveZarrImage(
                        filename, self.stack_handler)
            except Exception:
                traceback.print_exc()

        def done(results):
            self.save_cropped.setDisabled(False)

        self.worker = thread_worker(
            work_func,
            progress=False, z_stage=False)
        self.worker.signals.result.connect(done)
        # Execute
        self.save_cropped.setDisabled(True)
        self._threadpool.start(self.worker)

    def average_stack(self):
        if self.stack_handler is not None:
            sum = np.array([page.asarray() for page in self.tiff.pages])
            avg = sum.mean(axis=0, dtype=np.float32)

            self.image.setImage(avg, autoLevels=True)

    def kymogram_display_clicked(self, data: np.ndarray):
        self.enable_roi.setChecked(False)
        uImg = uImage(data)
        uImg.equalizeLUT()
        self.image.setImage(uImg._view, autoLevels=True)

    def kymogram_btn_clicked(self):
        if self.stack_handler is None:
            return

        self.kymogram_widget.extract_kymogram(
            self.stack_handler,
            self.pages_slider.value(),
            self.pages_slider.maximum(),
            self.get_roi_info()
        )

    def genOME(self):
        if self.stack_handler is not None:
            frames = self.stack_handler.shape[0]
            width = self.image.image.shape[1]
            height = self.image.image.shape[0]
            ome = self.metadata_editor.gen_OME_XML(frames, width, height)
            # tf.tiffcomment(
            #     self.path,
            #     ome.to_xml())
            # print(om.OME.from_tiff(self.tiff.filename))

    def region_changed(self, value):
        if self.stack_handler is not None:
            self.update_display()

    def slider_changed(self, value):
        if self.stack_handler is not None:
            self.update_display()

        if self.pages_slider is not None:
            self.pages_label.setText(
                'Page: {:d}/{:d}'.format(
                    self.pages_slider.value() + 1,
                    self.pages_slider.maximum() + 1))

    def update_display(self, image=None):
        if image is None:
            image = self.stack_handler.getSlice(
                self.pages_slider.value(), 0, 0)

        varim = None
        if self.cmos_maps_group.active.isChecked():
            res = self.cmos_maps_group.getMaps()
            if res is not None:
                if res[0].shape == image.shape:
                    image = image * res[0]
                    image = image - res[1]
                    varim = res[2]

        roiInfo = self.get_roi_info()
        if roiInfo is not None:
            origin, dim = roiInfo
        else:
            origin, dim = None, None

        if self.temp_median_filter.enabled.isChecked():
            frames = self.temp_median_filter.filter.getFrames(
                self.pages_slider.value(), self.stack_handler)

            image = self.temp_median_filter.filter.run(image, frames, roiInfo)

        self.uImage = uImage(image)

        self.lr_0.setBounds([0, self.uImage._max])

        min_max = None
        if not self.autostretch.isChecked():
            min_max = tuple(map(math.ceil, self.lr_0.getRegion()))

        self.uImage.equalizeLUT(min_max, True)

        self._plot_ref.setData(self.uImage._hist)

        # if self.uImage._isfloat:
        #     new_tick_values = np.arange(
        #         self.uImage.n_bins) * np.max(image) / 2**16
        #     new_tick_labels = [f'{val:.2f}' for val in new_tick_values]

        #     # Set the ticks on the x-axis using the plot reference
        #     self._plot_ref.setData(
        #         np.arange(self.uImage.n_bins) * np.max(image) / 2**16,
        #         self.uImage._hist)

        if self.autostretch.isChecked():
            self.lr_0.sigRegionChangeFinished.disconnect(self.region_changed)
            self.lr_0.setRegion([self.uImage._min, self.uImage._max])
            self.histogram.setXRange(
                self.uImage._min, self.uImage._max)
            self.lr_0.sigRegionChangeFinished.connect(self.region_changed)

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
            if self.th_max_slider.value() < 1.0:
                _, th2 = cv2.threshold(
                    img,
                    np.max(img) * self.th_max_slider.value(),
                    1,
                    cv2.THRESH_BINARY_INV)
                th_img = th_img * th2

            if self.image_filter.currentData().filter._show_filter:
                cv2.namedWindow('Thresholded filtered Img.', cv2.WINDOW_NORMAL)
                cv2.imshow('Thresholded filtered Img.', th_img)

            # Detect blobs.

            points, im_with_keypoints = self.detection_method.currentData()\
                .detector.find_peaks_preview(th_img, img)

            # Show keypoints
            if self.image_filter.currentData().filter._show_filter:
                cv2.namedWindow('Approx. Loc.', cv2.WINDOW_NORMAL)
                cv2.imshow('Approx. Loc.', im_with_keypoints)

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
                sz = self.fit_roi_size.value()
                if varim is None:
                    varims = None
                    rois, coords = pyfit3Dcspline.get_roi_list(
                        image, points, sz)
                else:
                    rois, varims, coords = pyfit3Dcspline.get_roi_list_CMOS(
                        image, varim, points, sz)
                Parameters = None

                if method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 1, np.array([1]), varims, 0)
                elif method == FittingMethod._2D_Gauss_MLE_free_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 2, np.array([1]), varims, 0)
                elif method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 4, np.array([1]), varims, 0)
                elif method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
                    Parameters, CRLBs, LogLikelihood = \
                        pyfit3Dcspline.CPUmleFit_LM(
                            rois, 5,
                            np.ones((64, 4, 4, 4), dtype=np.float32),
                            varims, 0)

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
        # if 'Check' in frc_method:
        #     img = self.renderLoc()

        #     if img is not None:
        #         def work_func():
        #             try:
        #                 return FRC_resolution_check_pattern(
        #                     img, self.super_px_size.value())
        #             except Exception:
        #                 traceback.print_exc()
        #                 return None

        #         def done(results):
        #             self.frc_res_btn.setDisabled(False)
        #             if results is not None:
        #                 plotFRC_(*results)
        #     else:
        #         return
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            data = self.fittingResults.toRender()

            def work_func():
                try:
                    return FRC_resolution_binomial(
                        np.c_[data[0], data[1], data[2]],
                        self.super_px_size.value(),
                        frc_method)
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.frc_res_btn.setDisabled(False)
                if results is not None:
                    plotFRC(*results)
        else:
            return

        self.worker = thread_worker(
            work_func,
            progress=False, z_stage=False)
        self.worker.signals.result.connect(done)
        # Execute
        self.frc_res_btn.setDisabled(True)
        self._threadpool.start(self.worker)

    def NeNA_estimate(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            def work_func():
                try:
                    return self.fittingResults.nn_trajectories(
                        0, 200, 0, 1)
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results):
                self.NeNA_btn.setDisabled(False)
                if results is not None:
                    self.fittingResults = results
                    self.results_plot.setData(
                        self.fittingResults.dataFrame())

                    self.NeNA_widget = NeNA_Widget(
                        self.fittingResults.neighbour_dist,
                        self.fittingResults.trackID
                    )

                    res = self.NeNA_widget.exec_()

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.NeNA_btn.setDisabled(True)
            self._threadpool.start(self.worker)

    def TARDIS_analysis(self):
        if self.fittingResults is None:
            return
        elif len(self.fittingResults) > 0:
            self.tardis = TARDIS_Widget(
                self.fittingResults.frames,
                self.fittingResults.locX,
                self.fittingResults.locY,
                self.fittingResults.locZ,
            )
            self.tardis.startWorker.connect(
                lambda worker: self._threadpool.start(worker))
            self.tardis.show()

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
        '''Imports fitting results from a file.
        '''
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Import localizations',
            filter='HDF5 files (*.h5);;TSV Files (*.tsv)')

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
        '''Exports the fitting results into a file.

        Parameters
        ----------
        filename : str, optional
            file path if None a save file dialog is shown, by default None
        '''
        if self.fittingResults is None:
            return

        if filename is None:
            if not self.export_options.exec_():
                return

            filename, _ = QFileDialog.getSaveFileName(
                self, 'Export localizations',
                filter='HDF5 files (*.h5);;TSV Files (*.tsv)')

        if len(filename) > 0:
            options = self.export_options.toList()

            dataFrame = self.fittingResults.dataFrame()
            exp_columns = []
            for col in dataFrame.columns:
                if col in options:
                    exp_columns.append(col)

            if '.tsv' in filename:
                dataFrame.to_csv(
                    filename, index=False,
                    columns=exp_columns,
                    float_format=self.export_options.export_precision.text(),
                    sep='\t',
                    encoding='utf-8')
            elif '.h5' in filename:
                dataFrame[exp_columns].to_hdf(
                    filename, key='microEye', index=False,
                    complevel=0)

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
        if self.stack_handler is None:
            return

        if not self.export_options.exec_():
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Save localizations',
            filter='HDF5 files (*.h5);;TSV Files (*.tsv)')

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
        tempEnabled = self.temp_median_filter.enabled.isChecked()
        detector = self.detection_method.currentData().detector

        # ROI
        roiInfo = self.get_roi_info()
        self.enable_roi.setChecked(False)

        min_max = None
        if not self.autostretch.isChecked():
            min_max = tuple(map(math.ceil, self.lr_0.getRegion()))

        # varim
        varim = None
        offset = 0
        gain = 1
        if self.cmos_maps_group.active.isChecked():
            res = self.cmos_maps_group.getMaps()
            if res:
                gain = res[0]
                offset = res[1]
                varim = res[2]

        rel_threshold = self.th_min_slider.value()
        max_threshold = self.th_max_slider.value()
        roi_size  = self.fit_roi_size.value()

        # uses only n_threads - 2
        threads = self._threadpool.maxThreadCount() - 2
        print('Threads', threads)
        for i in range(
                0, int(
                    np.ceil(
                        self.stack_handler.shape[0] / threads
                        ) * threads
                        ),
                threads):
            time = QDateTime.currentDateTime()
            workers = []
            self.thread_done = 0
            for k in range(threads):
                if i + k < len(self.stack_handler):
                    img = self.stack_handler.getSlice(
                        i + k, 0, 0)
                    img = img * gain
                    img = img - offset

                    temp = None
                    if tempEnabled:
                        temp = TemporalMedianFilter()
                        temp._temporal_window = \
                            self.temp_median_filter.filter._temporal_window

                    options = {
                        'index': i + k,
                        'tiffSeq_Handler': self.stack_handler,
                        'image': img,
                        'varim': varim,
                        'temp': temp,
                        'filter': filter,
                        'detector': detector,
                        'roiInfo': roiInfo,
                        'irange': min_max,
                        'rel_threshold': rel_threshold,
                        'max_threshold': max_threshold,
                        'roi_size': roi_size,
                        'method': method
                    }

                    worker = thread_worker(
                        pre_localize_frame,
                        progress=False, z_stage=False,
                        **options)
                    worker.signals.result.connect(self.update_lists)
                    workers.append(worker)
                    QThreadPool.globalInstance().start(worker)

            while self.thread_done < len(workers):
                QThread.msleep(10)

            exex = time.msecsTo(QDateTime.currentDateTime())
            duration = start.msecsTo(QDateTime.currentDateTime())

            print(
                'index: {:d}/{:d}, Time: {:d}  '.format(
                    i + len(workers), self.stack_handler.shape[0], exex),
                end='\r')
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
        tempEnabled = self.temp_median_filter.enabled.isChecked()
        detector = self.detection_method.currentData().detector

        # Vars
        roiSize = self.fit_roi_size.value()
        rel_threshold = self.th_min_slider.value()
        max_threshold = self.th_max_slider.value()
        PSFparam = np.array([1.5])

        min_max = None
        if not self.autostretch.isChecked():
            min_max = tuple(map(math.ceil, self.lr_0.getRegion()))

        roi_list = []
        varim_list = []
        coord_list = []
        frames_list = []

        # ROI
        roiInfo = self.get_roi_info()
        self.enable_roi.setChecked(False)

        # varim
        varim = None
        offset = 0
        gain = 1
        if self.cmos_maps_group.active.isChecked():
            res = self.cmos_maps_group.getMaps()
            if res:
                gain = res[0]
                offset = res[1]
                varim = res[2]

        for k in range(len(self.stack_handler)):
            cycle = QDateTime.currentDateTime()
            image = self.stack_handler.getSlice(k, 0, 0)

            image = image * gain
            image = image - offset

            temp = None
            if tempEnabled:
                temp = TemporalMedianFilter()
                temp._temporal_window = \
                    self.temp_median_filter.filter._temporal_window

            filtered = image.copy()

            # apply the median filter
            if temp is not None:
                frames = temp.getFrames(k, self.stack_handler)
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

            uImg.equalizeLUT(min_max, True)

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
            if max_threshold < 1.0:
                _, th2 = cv2.threshold(
                    img,
                    np.max(img) * max_threshold,
                    1,
                    cv2.THRESH_BINARY_INV)
                th_img = th_img * th2

            points: np.ndarray = detector.find_peaks(th_img)

            if len(points) > 0:
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
                    100*(k+1)/len(self.stack_handler),
                    cycle.msecsTo(QDateTime.currentDateTime())),
                end='\r')

        roi_list = np.vstack(roi_list)
        coord_list = np.vstack(coord_list)
        varim_list = None if varim is None else np.vstack(varim_list)

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
        '''
        Initializes a new QApplication and tiff_viewer.

        Parameters
        ----------
        path : str, optional
            The path to a file to be loaded initially.

        Returns
        -------
        tuple of QApplication and tiff_viewer
            Returns a tuple with QApplication and tiff_viewer main window.
        '''
        return start_gui(tiff_viewer, path)


def get_dock_config(dock: QDockWidget):
    '''
    Get the configuration dictionary for a QDockWidget.

    Parameters
    ----------
    dock : QDockWidget
        The QDockWidget to get the configuration for.

    Returns
    -------
    dict
        The configuration dictionary containing isFloating,
        position, size, and isVisible.
    '''
    if dock:
        return {
            'isFloating': dock.isFloating(),
            'position': (
                dock.mapToGlobal(QPoint(0, 0)).x(),
                dock.mapToGlobal(QPoint(0, 0)).y()),
            'size': (dock.geometry().width(), dock.geometry().height()),
            'isVisible': dock.isVisible()
        }

def get_widget_config(widget: QWidget):
    '''
    Get the configuration dictionary for a QWidget.

    Parameters
    ----------
    widget : QWidget
        The QWidget to get the configuration for.

    Returns
    -------
    dict
        The configuration dictionary containing position, size, and isMaximized.
    '''
    if widget:
        return {
            'position': (
                widget.mapToGlobal(QPoint(0, 0)).x(),
                widget.mapToGlobal(QPoint(0, 0)).y()),
            'size': (widget.geometry().width(), widget.geometry().height()),
            'isMaximized': widget.isMaximized()
        }

def saveConfig(window: tiff_viewer, filename: str = 'config_tiff.json'):
    """
    Save the configuration for the tiff_viewer application.

    Parameters
    ----------
    window : tiff_viewer
        The main application window.
    filename : str, optional
        The filename of the configuration file, by default 'config_tiff.json'.
    """
    config = dict()

    # Save tiff_viewer widget config
    config['tiff_viewer'] = get_widget_config(window)

    # Save docks config
    for key in DockKeys:
        dock = window.docks.get(key)
        if dock:
            config[key.value] = get_dock_config(dock)

    with open(filename, 'w') as file:
        json.dump(config, file, indent=2)

    print(f'{filename} file generated!')

def load_widget_config(widget: QWidget, widget_config):
    '''
    Load configuration for a QWidget.

    Parameters
    ----------
    widget : QWidget
        The QWidget to apply the configuration to.
    widget_config : dict
        The configuration dictionary containing position, size, and maximized status.

    Returns
    -------
    None
    '''
    widget.setGeometry(
        widget_config['position'][0],
        widget_config['position'][1],
        widget_config['size'][0],
        widget_config['size'][1]
    )
    if bool(widget_config['isMaximized']):
        widget.showMaximized()

def loadConfig(window: tiff_viewer, filename: str = 'config_tiff.json'):
    """
    Load the configuration for the tiff_viewer application.

    Parameters
    ----------
    window : tiff_viewer
        The main application window.
    filename : str, optional
        The filename of the configuration file, by default 'config_tiff.json'.
    """
    if not os.path.exists(filename):
        print(f'{filename} not found!')
        return

    config: dict = None

    with open(filename) as file:
        config = json.load(file)

    # Loading tiff_viewer widget config
    if 'tiff_viewer' in config:
        load_widget_config(window, config['tiff_viewer'])

    # Loading docks
    for dock_key, dock_config in config.items():
        dock_enum_key = None
        try:
            dock_enum_key = DockKeys(dock_key)
        except ValueError:
            # Skip processing if dock_key is not a valid DockKeys enum
            continue

        if dock_enum_key in window.docks:
            dock = window.docks[dock_enum_key]
            dock.setVisible(bool(dock_config.get('isVisible', False)))
            if bool(dock_config.get('isFloating', False)):
                dock.setFloating(True)
                dock.setGeometry(
                    dock_config.get('position', (0, 0))[0],
                    dock_config.get('position', (0, 0))[1],
                    dock_config.get('size', (0, 0))[0],
                    dock_config.get('size', (0, 0))[1]
                )
            else:
                dock.setFloating(False)

    print(f'{filename} file loaded!')


if __name__ == '__main__':
    app, window = tiff_viewer.StartGUI()
    app.exec_()
