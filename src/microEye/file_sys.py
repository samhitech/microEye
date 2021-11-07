import sys
import os
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QIcon
import pyqtgraph as pg
from pyqtgraph.colormap import ColorMap
import qdarkstyle
import ctypes
import tifffile as tf
import cv2
import numpy as np


class tiff_viewer(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'microEye tiff viewer'
        self.left = 0
        self.top = 0
        self.width = 1024
        self.height = 600
        self._zoom = 1  # display resize

        self.tiff = None

        self.initialize()

    def initialize(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.center()
        self.path = os.path.dirname(os.path.abspath(__file__))
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

        self.main_layout = QHBoxLayout()
        self.right_layout = QVBoxLayout()
        self.g_layout_widget = QVBoxLayout()

        self.main_layout.addWidget(self.tree, 1)
        self.main_layout.addLayout(self.right_layout, 1)
        self.main_layout.addLayout(self.g_layout_widget, 3)

        self.controls_group = QGroupBox('Tiff Options')
        self.controls_layout = QVBoxLayout()
        self.controls_group.setLayout(self.controls_layout)

        self.series_slider = QSlider(Qt.Horizontal)
        self.series_slider.setMinimum(0)
        self.series_slider.setMaximum(0)

        self.pages_label = QLabel('Pages')
        self.pages_slider = QSlider(Qt.Horizontal)
        self.pages_slider.setMinimum(0)
        self.pages_slider.setMaximum(0)
        self.pages_slider.valueChanged.connect(self.slider_changed)

        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(4096)
        self.min_slider.valueChanged.connect(self.slider_changed)

        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(4096)
        self.max_slider.setValue(4096)
        self.max_slider.valueChanged.connect(self.slider_changed)

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

        # self.controls_layout.addWidget(QLabel('Series'))
        # self.controls_layout.addWidget(self.series_slider)

        self.controls_layout.addWidget(self.pages_label)
        self.controls_layout.addWidget(self.pages_slider)
        self.controls_layout.addWidget(self.autostretch)
        self.controls_layout.addWidget(QLabel('Min'))
        self.controls_layout.addWidget(self.min_slider)
        self.controls_layout.addWidget(QLabel('Max'))
        self.controls_layout.addWidget(self.max_slider)
        # self.controls_layout.addLayout(self.zoom_layout)
        self.controls_layout.addStretch()

        self.right_layout.addWidget(self.controls_group)

        # graphics layout
        # # A plot area (ViewBox + axes) for displaying the image
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image_plot = pg.ImageView(imageItem=self.image)
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

            self.update_display()

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
            self.pages_label.setText('Page: {:d}'.format(value))

    def update_display(self):
        image = self.tiff.pages[self.pages_slider.value()].asarray()

        if self.autostretch.isChecked():
            # calculate image histogram
            _hist = cv2.calcHist(
                [image], [0], None,
                [4096], [0, 4096]) / float(np.prod(image.shape))
            # calculate the cdf
            cdf = _hist[:, 0].cumsum()

            p2 = np.where(cdf >= 0.00001)[0][0]
            p98 = np.where(cdf >= 0.9999)[0][0]
            self.min_slider.setValue(p2)
            self.max_slider.setValue(p98)

            c = 255.0 / (p98 - p2)
            ac = p2 * c
            image = np.subtract(
                image.dot(c), ac).astype(np.uint8)
        else:
            p2 = self.min_slider.value()
            p98 = self.max_slider.value()

            c = 255.0 / (p98 - p2)
            ac = p2 * c
            image = np.subtract(
                image.dot(c), ac).astype(np.uint8)

        # resizing the image
        image = cv2.resize(
            image, (0, 0), fx=self._zoom, fy=self._zoom)

        # cv2.imshow(self.path, image)
        self.image.setImage(image)

        # # Detect blobs.
        # keypoints = self.blob_detector().detect(image)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        # the size of the circle corresponds to the size of blob
        # im_with_keypoints = cv2.drawKeypoints(
        #     image, keypoints, np.array([]),
        #     (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # # Show keypoints
        # cv2.imshow("Keypoints", im_with_keypoints)

    def blob_detector(self) -> cv2.SimpleBlobDetector:
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 100
        params.maxThreshold = 4096

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 20

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.75

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 1

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.5

        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3:
            return cv2.SimpleBlobDetector(params)
        else:
            return cv2.SimpleBlobDetector_create(params)

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
        font = app.font()
        font.setPointSize(10)
        app.setFont(font)
        # sets the app icon
        app_icon = QIcon()
        app_icon.addFile('icons/16.png', QSize(16, 16))
        app_icon.addFile('icons/24.png', QSize(24, 24))
        app_icon.addFile('icons/32.png', QSize(32, 32))
        app_icon.addFile('icons/48.png', QSize(48, 48))
        app_icon.addFile('icons/64.png', QSize(64, 64))
        app_icon.addFile('icons/128.png', QSize(128, 128))
        app_icon.addFile('icons/256.png', QSize(256, 256))
        app_icon.addFile('icons/512.png', QSize(512, 512))

        app.setWindowIcon(app_icon)

        myappid = u'samhitech.mircoEye.tiff_viewer'  # appid
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        window = tiff_viewer()
        return app, window


if __name__ == '__main__':
    app, window = tiff_viewer.StartGUI()
    app.exec_()
