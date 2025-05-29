from time import perf_counter, perf_counter_ns
from typing import Optional, Union

import cv2
import numpy as np
import pyqtgraph as pg

from microEye.qt import QtCore, QtGui, QtWidgets, Signal


def fast_autolevels_opencv(
    image: np.ndarray, low_percent=0.001, high_percent=99.99, levels=False
):
    nbins = max(256, int(np.ceil(image.max())) + 1)

    if image.ndim == 3:
        hist = np.array(
            [
                cv2.calcHist(
                    [image[:, :, channel]],
                    [0],
                    None,
                    [nbins],
                    [0, nbins],
                ).squeeze()
                for channel in range(image.shape[2])
            ]
        ).T
    else:
        hist = cv2.calcHist([image], [0], None, [nbins], [0, nbins])
    cumsum = np.cumsum(hist, axis=0)
    total = cumsum[-1]

    hist /= np.max(hist, axis=0)

    thresholds = []

    if levels:
        if image.ndim == 3:
            # Considering the cumulative distribution function across all channels
            for idx in range(cumsum.shape[1]):
                # Find indices for cdf_min and cdf_max for each channel
                min_idx, max_idx = np.searchsorted(
                    cumsum[:, idx],
                    [total[idx] * low_percent / 100, total[idx] * high_percent / 100],
                    side='left',
                ).squeeze()

                # Append the results
                thresholds.append([min_idx, max(min_idx + 1, max_idx)])
        else:
            low_thresh, high_thresh = np.searchsorted(
                cumsum[:, 0],
                [total * low_percent / 100, total * high_percent / 100],
                side='left',
            ).squeeze()
            thresholds.append([low_thresh, max(low_thresh + 1, high_thresh)])
    else:
        low_thresh, high_thresh = 0, nbins - 1
        thresholds.append([low_thresh, high_thresh])

    return thresholds, hist, cumsum / total


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

    image_update_signal = Signal(np.ndarray, dict)  # Signal to receive new images
    close_signal = Signal()  # Signal to close the display

    BINS = {
        8: np.arange(0, 256, 1),
        10: np.arange(0, 1024, 1),
        11: np.arange(0, 2048, 1),
        12: np.arange(0, 4096, 1),
        14: np.arange(0, 2**14, 1),
        16: np.arange(0, 2**16, 1),
    }

    def __init__(self, title: str = 'PyQtGraph Display', parent=None, **kwargs):
        super().__init__(parent)
        self.setWindowTitle(title)

        width = kwargs.get('width', 512)
        height = kwargs.get('height', 512)
        self.image_aspect_ratio = width / height

        # set maximum size to current screen size
        screen = QtWidgets.QApplication.primaryScreen()
        screen_size = screen.size()
        self.setMaximumSize(screen_size.width(), screen_size.height())
        self.resize(
            int((screen_size.height() - 100) * self.image_aspect_ratio),
            screen_size.height() - 100,
        )

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

        layout.addWidget(self.plot_widget, 6)

        # Create image item
        self.image_item = pg.ImageItem(
            axisOrder='row-major', border=kwargs.get('border', 'y')
        )
        self.image_item.setImage(
            np.zeros(
                (height, width),
            )
        )  # Initialize with a blank image
        self.view_box.addItem(self.image_item)

        # Add text item in viewbox for FPS display
        self.text_item = pg.TextItem()
        self.text_item.setPos(16, 16)  # Adjust position
        self.text_item.setZValue(100)  # Ensure text is on top of the image
        self.text_item.setHtml(
            '<div style="color: white; font-size: 14px;'
            'background-color: rgba(0, 0, 0, 0.5);">'
            '<span style="font-weight: bold;">'
            'FPS: 0.00 (0.00 ms)</span><br>'
            '<span style="font-size: 11px;">'
            'Image Stats: 0 x 0 | '
            'Min: 0.00 | '
            'Max: 0.00 | '
            'Mean: 0.00 | '
            'Std: 0.00</span></div>'
        )
        self.plot_widget.addItem(self.text_item)

        # Create a timer to update the FPS
        self.frame_counter = FrameCounter()
        self.frame_counter.sigFpsUpdate.connect(self.update_fps)

        # hist and cdf plot
        self.histogram = pg.PlotWidget()

        # Set up pens and brushes for histogram and CDF
        colors = [
            '#FF0000',
            '#00FF00',
            '#0000FF',
        ]

        # Add histogram and CDF plots
        self._plot_refs = [
            self.histogram.plotItem.plot(
                np.zeros_like(self.BINS[16]),
                pen=pg.mkPen(color=colors[idx]),
            )
            for idx in range(3)
        ]

        # Add linear regions for histogram and CDF
        self.histogram_region_items = [
            pg.LinearRegionItem(
                (0, 256),
                bounds=(0, 2**16),
                pen=pg.mkPen(color=colors[idx]),
                brush=pg.mkBrush(color=f'{colors[idx]}20'),
                movable=True,
                swapMode='push',
                span=(0.0, 1),
            )
            for idx in range(3)
        ]
        for idx in range(3):
            self.histogram.addItem(self.histogram_region_items[idx])

        layout.addWidget(self.histogram, 1)

        # Connect signals
        self.image_update_signal.connect(self.update_image)

    def showEvent(self, event: QtGui.QShowEvent):
        '''Ensure minimum width is set after the widget is shown.'''
        super().showEvent(event)
        # set widget minimum size to fit the text item
        self.setMinimumWidth(int(self.text_item.boundingRect().width() * 1.25))

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
        if self.text_item.isVisible():
            self.text_item.setHtml(
                '<div style="color: white; font-size: 14px;'
                'background-color: rgba(0, 0, 0, 0.5);">'
                '<span style="font-weight: bold;">'
                f'FPS: {fps:.2f} ({1000 / fps if fps > 0 else np.nan:.2f} ms)'
                '</span><br>'
                '<span style="font-size: 11px;">'
                f'Image Stats: {self.image_item.image.shape[1]} x '
                f'{self.image_item.image.shape[0]} | '
                f'Min: {np.min(self.image_item.image):.2f} | '
                f'Max: {np.max(self.image_item.image):.2f} | '
                f'Mean: {np.mean(self.image_item.image):.2f} | '
                f'Std: {np.std(self.image_item.image):.2f}</span></div>'
            )

    def get_regions(self):
        '''Get the regions of interest from the histogram region items.'''
        regions = [
            item.getRegion() for item in self.histogram_region_items if item.isVisible()
        ]
        return regions[0] if len(regions) == 1 else regions

    def set_regions(self, thresholds: list[list[float]]):
        '''Set the regions of interest for the histogram region items.'''
        count = len(thresholds)

        for idx, item in enumerate(self.histogram_region_items):
            if idx < len(thresholds):
                item.setRegion(thresholds[idx])
                item.setVisible(True)
                item.setSpan(idx / count, (idx + 1) / count)
            else:
                item.setVisible(False)

    def update_image(self, image: np.ndarray, kwargs: dict):
        '''Update the main image view.'''

        thresholds = kwargs.get('threshold', [0, 255])
        plot_data: np.ndarray = kwargs.get('plot')

        # Update histogram and CDF plots each second
        if self.frame_counter.count % 1000 == 0:
            # Update plots
            for idx in range(plot_data.shape[1]):
                self._plot_refs[idx].setData(plot_data[:, idx])

        if kwargs.get('autoLevels', True):
            # autoLevels = True, set levels to min/max of the image
            kwargs['levels'] = np.array(thresholds).squeeze()
            self.set_regions(thresholds)
            self.histogram.setXRange(np.min(thresholds), np.max(thresholds))
        else:
            # autoLevels = False, set levels to the region of interest
            kwargs['levels'] = self.get_regions()

        self.image_item.setImage(
            image,
            autoLevels=False,
            levels=kwargs.get('levels'),
            # lut=useLut,
            # autoDownsample=True,
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

    # def resizeEvent(self, event: QtGui.QResizeEvent):
    #     '''Override resizeEvent to maintain the aspect ratio.'''
    #     self.adjust_widget_aspect_ratio(event)

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
        cls.DISPLAYS[window_name].image_update_signal.emit(image, kwargs)

    @classmethod
    def display_closed(cls, display_name: str):
        '''Handle display closed event.'''
        if display_name in cls.DISPLAYS:
            del cls.DISPLAYS[display_name]

    @classmethod
    def tile_displays(cls, screen_idx: int = 0):
        '''Tile all open displays on screen resize as needed.'''
        screen = QtWidgets.QApplication.screens()[screen_idx]
        screen_size = screen.size()
        screen_width = screen_size.width()
        screen_height = screen_size.height()

        num_displays = len(cls.DISPLAYS)
        if num_displays == 0:
            return
        elif num_displays == 1:
            # If only one display, center it on the screen
            display = list(cls.DISPLAYS.values())[0]
            display.resize(screen_width, screen_height)
            display.move(0, 0)
        else:
            # Tile the displays in a grid layout
            cols = int(num_displays)
            rows = (num_displays + cols - 1) // cols

            display_width = screen_width // cols
            display_height = (screen_height - 100) // rows

            for i, display in enumerate(cls.DISPLAYS.values()):
                x = (i // rows) * display_width
                y = (i % rows) * display_height
                display.resize(display_width, display_height)
                display.move(x, y)

    @classmethod
    def close_all_displays(cls):
        '''Close all open displays.'''
        for display in list(cls.DISPLAYS.values()):
            display.close()


# class ImageProcessor(QtCore.QObject):
#     '''A worker class for processing and sending images.'''

#     image_ready = Signal(np.ndarray, dict)  # Signal to send processed images
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
#         self.image_ready.emit(self.images[self.idx % self.images.shape[0]], {})
#         self.idx += 1

#     @staticmethod
#     def generate_test_image(samples=1000, width=640, height=480):
#         '''Generate a test image with random noise.'''
#         return (np.random.rand(samples, height, width) * 4095).astype(np.uint16)


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
#         self.processor.image_ready.connect(self.display.image_update_signal)

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
