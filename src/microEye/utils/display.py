from time import perf_counter
from typing import Optional

import numpy as np
import pyqtgraph as pg

from microEye.qt import QtCore, QtGui, QtWidgets, Signal


class FrameCounter(QtCore.QObject):
    sigFpsUpdate = Signal(object)

    def __init__(self, interval=1000):
        super().__init__()
        self.count = 0
        self.last_update = 0
        self.interval = interval

    def update(self):
        self.count += 1

        if self.last_update == 0:
            self.last_update = perf_counter()
            self.startTimer(self.interval)

    def timerEvent(self, evt):
        now = perf_counter()
        elapsed = now - self.last_update
        fps = self.count / elapsed
        self.last_update = now
        self.count = 0
        self.sigFpsUpdate.emit(fps)


class PyQtGraphDisplay(QtWidgets.QWidget):
    '''A PyQtGraph-based widget for displaying images.'''

    image_update_signal = Signal(np.ndarray)  # Signal to receive new images
    close_signal = Signal()  # Signal to close the display

    def __init__(self, title: str = 'PyQtGraph Display', parent=None, **kwargs):
        super().__init__(parent)
        self.setWindowTitle(title)

        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        self.image_aspect_ratio = width / height
        self.resize(width, height)

        # Setup pyqtgraph with optimizations
        pg.setConfigOptions(antialias=False, imageAxisOrder='row-major')

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(layout)

        self.setStatusTip('Press "Esc" to close the window.')
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)

        # Use PlotWidget instead which should be more compatible with PyQt6
        self.plot_widget = pg.GraphicsView()
        # remove margins from the plot widget
        self.plot_widget.setContentsMargins(0, 0, 0, 0)

        self.view_box = pg.ViewBox()
        self.plot_widget.setCentralItem(self.view_box)
        self.view_box.setAspectLocked(True, ratio=kwargs.get('aspect_ratio', 1.0))
        self.view_box.setAutoVisible(True)
        self.view_box.setDefaultPadding(0.005)
        self.view_box.enableAutoRange()
        self.view_box.invertY(True)

        layout.addWidget(self.plot_widget)

        # Create image item
        self.image_item = pg.ImageItem(
            axisOrder='row-major', border=kwargs.get('border', 'y')
        )
        self.view_box.addItem(self.image_item)

        # Add text item in viewbox for FPS display
        self.text_item = pg.TextItem()
        self.text_item.setPos(16, 16)  # Adjust position
        self.text_item.setZValue(100)  # Ensure text is on top of the image
        self.plot_widget.addItem(self.text_item)

        # Create a timer to update the FPS
        self.frame_counter = FrameCounter()
        self.frame_counter.sigFpsUpdate.connect(self.update_fps)

        # Connect signals
        self.image_update_signal.connect(self.update_image)

    def setStatsVisible(self, visible: bool):
        '''hide or remove the stats text item.'''
        if visible:
            self.text_item.show()
        else:
            self.text_item.hide()

    def update_fps(self, fps: float):
        '''Update the FPS display.'''
        # set html text for better formatting
        # includes time info + image size + image min/max/mean/std
        self.text_item.setHtml(
            '<div style="color: white; font-size: 14px;'
            'background-color: rgba(0, 0, 0, 0.5);">'
            f'<span style="font-weight: bold;">'
            f'FPS: {fps:.2f} ({1000 / fps if fps > 0 else np.nan:.2f} ms)</span><br>'
            f'<span style="font-size: 11px;">'
            f'Image Stats: {self.image_item.image.shape[1]} x '
            f'{self.image_item.image.shape[0]} | '
            f'Min: {np.min(self.image_item.image):.2f} | '
            f'Max: {np.max(self.image_item.image):.2f} | '
            f'Mean: {np.mean(self.image_item.image):.2f} | '
            f'Std: {np.std(self.image_item.image):.2f}</span></div>'
        )
        # set widget minimum size to fit the text
        self.setMinimumWidth(self.text_item.boundingRect().width() * 1.25)

    def update_image(self, image: np.ndarray):
        '''Update the main image view.'''
        self.image_item.setImage(
            image,
            autoLevels=False,
            # levels=useScale,
            # lut=useLut,
            # autoDownsample=downsample,
        )
        self.frame_counter.update()  # Update frame counter

    def adjust_widget_aspect_ratio(self, event: QtGui.QResizeEvent):
        """Adjust the widget's size to maintain the image's aspect ratio."""

        # Get the new size of the widget
        new_width = event.size().width()
        new_height = event.size().height()

        # old size of the widget
        old_width = event.oldSize().width()
        old_height = event.oldSize().height()

        # Calculate the new aspect ratio
        new_aspect_ratio = new_width / new_height

        # If the new aspect ratio is different from the image's aspect ratio,
        # adjust the widget's size to maintain the image's aspect ratio
        if new_aspect_ratio != self.image_aspect_ratio:
            if new_aspect_ratio > self.image_aspect_ratio:
                # Too wide, adjust width
                new_width = int(new_height * self.image_aspect_ratio)
            else:
                # Too tall, adjust height
                new_height = int(new_width / self.image_aspect_ratio)

            # Resize the widget to maintain the aspect ratio
            self.resize(new_width, new_height)

            event = QtGui.QResizeEvent(
                QtCore.QSize(new_width, new_height), event.oldSize()
            )

        super().resizeEvent(event)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        '''Override resizeEvent to maintain the aspect ratio.'''
        self.adjust_widget_aspect_ratio(event)

    def closeEvent(self, event: QtGui.QCloseEvent):
        '''Override closeEvent to emit the close signal.'''
        self.close_signal.emit()
        event.accept()


class DisplayManager(QtCore.QObject):
    '''Manager singleton class to handle multiple PyQtGraph displays.'''

    # Signal to update the displays with new images and kwargs
    image_update_signal = Signal(str, object, dict)

    DISPLAYS: dict[str, PyQtGraphDisplay] = {}

    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent=parent)

        self.image_update_signal.connect(self.im_show)

    @classmethod
    def instance(cls):
        '''Get the singleton instance of DisplayManager.'''
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def im_show(cls, window_name: str, image: np.ndarray, kwargs: dict):
        '''Show an image in a new or existing display window.'''
        if window_name not in cls.DISPLAYS:
            cls.DISPLAYS[window_name] = PyQtGraphDisplay(window_name, **kwargs)
            cls.DISPLAYS[window_name].close_signal.connect(
                lambda: cls.display_closed(window_name)
            )
            cls.DISPLAYS[window_name].show()

        cls.DISPLAYS[window_name].setStatsVisible(kwargs.get('show_stats', True))

        # Update the display with the new image
        cls.DISPLAYS[window_name].image_update_signal.emit(image)

    @classmethod
    def display_closed(cls, display_name: str):
        '''Handle display closed event.'''
        if display_name in cls.DISPLAYS:
            del cls.DISPLAYS[display_name]


# class ImageProcessor(QtCore.QObject):
#     '''A worker class for processing and sending images.'''

#     image_ready = Signal(np.ndarray)  # Signal to send processed images
#     roi_images_ready = Signal(list)  # Signal to send ROI images

#     def __init__(self, samples=1000):
#         super().__init__()
#         self.running = False
#         self.samples = samples

#         # Simulate receiving an image (replace with actual camera frame)
#         self.images = self.generate_test_image(self.samples)
#         self.idx = 0

#     def start_processing(self):
#         '''Start processing images.'''
#         self.image_ready.emit(self.images[self.idx % self.images.shape[0]])
#         self.idx += 1

#     @staticmethod
#     def generate_test_image(samples=1000, width=640, height=480):
#         '''Generate a test image with random noise.'''
#         return (np.random.rand(samples, height, width) * 255).astype(np.uint8)


# class PyQtGraphApp(QtWidgets.QApplication):
#     '''Main application to run the PyQtGraph-based widget.'''

#     def __init__(self):
#         super().__init__([])

#         # Create the main display widget
#         self.display = PyQtGraphDisplay(width=640, height=480)
#         self.display.show()

#         # Create the image processor
#         self.processor = ImageProcessor()

#         # Connect signals
#         self.processor.image_ready.connect(self.display.new_image_signal)

#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.processor.start_processing)
#         self.timer.start(0)

#     def stop(self):
#         '''Stop the application and clean up.'''
#         self.timer.stop()


# if __name__ == '__main__':
#     app = PyQtGraphApp()
#     app.exec()
#     app.stop()
