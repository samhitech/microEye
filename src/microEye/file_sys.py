import ctypes
import os
import sys
from enum import auto

import cv2
import numba
import numpy as np
import ome_types.model as om
import pyqtgraph as pg
import qdarkstyle
import tifffile as tf
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
from .thread_worker import *
from .uImage import uImage
from .Rendering import gauss_hist_render


class tiff_viewer(QWidget):

    def __init__(self, path=os.path.dirname(os.path.abspath(__file__))):
        super().__init__()
        self.title = 'microEye tiff viewer'
        self.left = 0
        self.top = 0
        self.width = 1024
        self.height = 600
        self._zoom = 1  # display resize
        self._n_levels = 4096

        self.tiff = None

        # Threading
        self._threadpool = QThreadPool()
        print("Multithreading with maximum %d threads"
              % self._threadpool.maxThreadCount())

        self.initialize(path)

    def initialize(self, path):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.center()
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

        self.metadataEditor = MetadataEditor()

        self.main_layout = QHBoxLayout()
        self.g_layout_widget = QVBoxLayout()

        self.tabView = QTabWidget()
        self.tabView.addTab(self.tree, 'File system')
        self.tabView.addTab(self.metadataEditor, 'OME-XML Metadata')

        self.main_layout.addWidget(self.tabView, 2)
        self.main_layout.addLayout(self.g_layout_widget, 3)

        self.controls_group = QWidget()
        self.controls_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_layout)

        self.tabView.addTab(self.controls_group, 'Tiff Options')

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

        self.export_loc_btn = QPushButton(
            'Export Localizations',
            clicked=lambda: self.export_loc())

        self.controls_layout.addWidget(self.export_loc_btn)

        # self.controls_layout.addWidget(self.average_btn)
        # self.controls_layout.addLayout(self.zoom_layout)
        self.controls_layout.addStretch()

        # graphics layout
        # # A plot area (ViewBox + axes) for displaying the image
        self.uImage = None
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image_plot = pg.ImageView(imageItem=self.image)
        self.image_plot.setLevels(0, 255)
        # self.image_plot.setColorMap(pg.colormap.getFromMatplotlib('jet'))
        self.g_layout_widget.addWidget(self.image_plot)
        # Item for displaying image data

        self.setLayout(self.main_layout)

        self.show()

    def center(self):
        '''Centers the window within the screen.
        '''
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def _open_file(self, i):
        if not self.model.isDir(i):
            cv2.destroyAllWindows()
            index = self.model.index(i.row(), 0, i.parent())
            self.path = self.model.filePath(index)
            if self.tiff is not None:
                self.tiff.close()
            self.tiff = tf.TiffFile(self.path)
            self.pages_slider.setMaximum(len(self.tiff.pages) - 1)
            self.pages_slider.valueChanged.emit(0)

            if self.tiff.ome_metadata is not None:
                ome = OME.from_xml(self.tiff.ome_metadata)
                self.metadataEditor.pop_OME_XML(ome)

            # self.update_display()
            # self.genOME()

    def average_stack(self):
        if self.tiff is not None:
            sum = np.array([page.asarray() for page in self.tiff.pages])
            avg = sum.mean(axis=0, dtype=np.float32)

            self.image.setImage(avg, autoLevels=False)

    def genOME(self):
        if self.tiff is not None:

            frames = len(self.tiff.pages)
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
        if self.tiff is not None:
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
            image = self.tiff.pages[self.pages_slider.value()].asarray()

        if self.tempMedianFilter.enabled.isChecked():
            self.tempMedianFilter.filter.getFrames(
                self.pages_slider.value(), len(self.tiff.pages), self.path)

            image = self.tempMedianFilter.filter.run(image)

        self.uImage = uImage(image)

        self.max_slider.setMaximum(self.uImage._max)
        self.min_slider.setMaximum(self.uImage._max)

        min_max = None
        if not self.autostretch.isChecked():
            min_max = (self.min_slider.value(), self.max_slider.value())

        self.uImage.equalizeLUT(min_max, True)

        if self.autostretch.isChecked():
            self.min_slider.setValue(self.uImage._min)
            self.max_slider.setValue(self.uImage._max)

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

    def export_loc(self):
        if self.tiff is None:
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

    def update_loc(self, data):
        shape = data
        renderClass = gauss_hist_render()
        img = renderClass.render(
            np.array(self.locX).copy() *
            float(self.metadataEditor.px_size.text()),
            np.array(self.locY).copy() *
            float(self.metadataEditor.py_size.text()),
            np.array(self.intensity).copy()
        )
        # H, _, _ = np.histogram2d(
        #     np.array(self.locX).copy() * 11.5,
        #     np.array(self.locY).copy() * 11.5,
        #     np.array(shape).copy() * 12)
        img_norm = cv2.normalize(
            img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        self.image.setImage(img_norm, autoLevels=False)
        # cv2.imshow('localization', H)
        # cv2.waitKey(1)
        return img

    def update_lists(self, result: np.ndarray):
        self.locX.extend(result[:, 0])
        self.locY.extend(result[:, 1])
        self.intensity.extend(result[:, 2])
        self.frame.extend(result[:, 3])
        self.thread_done += 1

    def proccess_loc(self, filename: str, progress_callback):
        self.locX = []
        self.locY = []
        self.frame = []
        self.intensity = []
        self.thread_done = 0
        time = QDateTime.currentDateTime()
        filter = self.bandpassFilter.filter
        tempEnabled = self.tempMedianFilter.enabled.isChecked()
        blob_detector = self.blob_detector()
        threads = self._threadpool.maxThreadCount() - 2
        print('Threads', threads)
        for i in range(
                0, int(np.ceil(len(self.tiff.pages) / threads) * threads),
                threads):
            time = QDateTime.currentDateTime()
            workers = []
            self.thread_done = 0
            for k in range(threads):
                if i + k < len(self.tiff.pages):
                    img = self.tiff.pages[i + k].asarray().copy()

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

            exex = time.msecsTo(QDateTime.currentDateTime())

            # uImg = uImage(self.tiff.pages[i].asarray())

            # uImg.equalizeLUT(None, False)

            # img = filter.run(
            #     uImg._view,
            #     self.fft_min_slider.value(),
            #     self.fft_max_slider.value(),
            #     type=self.fft_type_cbox.currentText(),
            #     refresh=False,
            #     show_filter=False)

            # # Detect blobs.

            # keypoints = blob_detector.detect(img)

            # points = cv2.KeyPoint_convert(keypoints)

            # time = QDateTime.currentDateTime()
            # phasor_fit(uImg._image, points)
            # exex = time.msecsTo(QDateTime.currentDateTime())

            # locX.extend(points[:, 0])
            # locY.extend(points[:, 1])
            # frame.extend([i] * len(points[:, 0]))

            print(
                'index: {:d}/{:d}, Time: {:d}  '.format(
                    i + len(workers), len(self.tiff.pages), exex),
                end="\r")
            if (i // threads) % 40 == 0:
                progress_callback.emit(self.uImage._image.shape)

        sres_img = self.update_loc(None)
        tf.imsave(
            filename.replace('.tsv', '_super_res.tif'),
            sres_img,
            photometric='minisblack',
            append=True,
            bigtiff=True,
            ome=False)

        loc = np.c_[np.array(self.locX), np.array(self.locY)]
        np.savetxt(filename, loc, delimiter='\t', encoding='utf-8')

        loc = np.c_[
            np.array(self.frame),
            np.array(self.locX) * float(self.metadataEditor.px_size.text()),
            np.array(self.locY) * float(self.metadataEditor.py_size.text()),
            np.array(self.intensity)]
        np.savetxt(
            filename.replace('.tsv', '_full.tsv') + '.full',
            loc, delimiter='\t', encoding='utf-8')

    def localize_frame(
            self, image: np.ndarray,
            temp: TemporalMedianFilter,
            filter: AbstractFilter,
            index):

        if temp is not None:
            temp.getFrames(index, len(self.tiff.pages), self.path)
            image = temp.run(image)

        uImg = uImage(image)

        uImg.equalizeLUT(None, False)

        if filter is BandpassFilter:
            filter._show_filter = False
            filter._refresh = False

        img = filter.run(uImg._view)

        # Detect blobs.

        keypoints = self.blob_detector().detect(img)

        points: np.ndarray = cv2.KeyPoint_convert(keypoints)

        time = QDateTime.currentDateTime()
        result = phasor_fit(uImg._image, points)
        result[:, 3] = [index + 1] * points.shape[0]

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

        myappid = u'samhitech.mircoEye.tiff_viewer'  # appid
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        window = tiff_viewer(path)
        return app, window


if __name__ == '__main__':
    app, window = tiff_viewer.StartGUI()
    app.exec_()
