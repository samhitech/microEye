import json
import traceback
from queue import Queue

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import tifffile as tf
from PyQt5.QtCore import Qt, QThreadPool, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.functions import interpolateArray

from ...shared.gui_helper import *
from ...shared.thread_worker import thread_worker
from ...shared.uImage import TiffSeqHandler, ZarrImageSequence, uImage


class MultiLineROISelector:
    def __init__(self, image_data: np.ndarray, resize_factor=1.0):
        '''
        Initialize the MultiLineROISelector.

        Parameters
        ----------
        image_data : np.ndarray
            The image data as a NumPy array.
        resize_factor : float, optional
            The initial resize factor for displaying the image, by default 1.0.
        '''
        self.original = image_data.copy()
        self.resize_factor = resize_factor
        self.resized_empty: np.ndarray = cv2.resize(
            self.original, (0, 0),
            fx=self.resize_factor, fy=self.resize_factor,
            interpolation=cv2.INTER_NEAREST)
        self.resized_image = self.resized_empty.copy()

        self.rois = []  # List store multiple ROIs, each represented as a list of points
        self.active_roi_idx = -1  # Index of the currently active ROI
        self.dragging = False
        self.drag_idx = -1
        self.modes = Queue()
        for mode in ['Add', 'Append', 'Edit', 'Insert', 'Remove']:
            self.modes.put(mode)
        self.current_mode = self.modes.get()

    def insert_index(self, cursor, threshold=1.005):
        '''
        Find the index to insert a new point based on cursor proximity to line segments.

        Parameters
        ----------
        cursor : Tuple (x, y)
            The cursor position represented as a tuple of (x, y) coordinates.
        threshold : float, optional
            Threshold distance for considering the cursor above a line segment,
            by default 3.0.

        Returns
        -------
        int
            The index at which to add the point if the cursor is above a line segment,
            otherwise -1.
        '''
        x, y = cursor
        n_points = len(self.rois[self.active_roi_idx])

        # Check if there are at least two points to form line segments
        if n_points < 2:
            return -1

        distances = []
        indices = []

        # Iterate through existing points to calculate distances to line segments
        for idx in range(n_points - 1):
            x1, y1 = self.rois[self.active_roi_idx][idx]
            x2, y2 = self.rois[self.active_roi_idx][idx + 1]

            segment_length = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            sum_distance = np.sqrt((x1 - x)**2 + (y1 - y)**2) + \
                np.sqrt((x2 - x)**2 + (y2 - y)**2)

            # Calculate the distance from the cursor to the line segment
            distance = sum_distance / segment_length

            # Check if the distance is below the threshold
            if distance < threshold:
                distances.append(distance)
                indices.append(idx + 1)

        # If no suitable segment found, return -1
        if len(distances) < 1:
            return -1
        else:
            # Return the index of the closest line segment
            return indices[distances.index(min(distances))]

    def remove_point(self, x, y, threshold=5.0):
        '''
        Find the index of the point to remove based on cursor proximity.

        Parameters
        ----------
        x : float
            The x-coordinate of the cursor position.
        y : float
            The y-coordinate of the cursor position.
        threshold : float, optional
            Threshold distance for considering the cursor above a point, by default 5.0.

        Returns
        -------
        int
            The index of the point to remove if the cursor is above a point,
            otherwise -1.
        '''
        if self.active_roi_idx >= 0 and self.active_roi_idx < len(self.rois):
            for idx, point in enumerate(self.rois[self.active_roi_idx]):
                dist = np.linalg.norm(np.array(point) - [x, y])
                if dist < threshold:
                    return idx

        return -1

    def update_polyline(self, image, points, color):
        '''
        Update the image by drawing the polyline with the updated points.

        Parameters
        ----------
        image : np.ndarray
            The image data as a NumPy array.
        points : list of Tuple (x, y)
            List of points representing the polyline.
        color : Tuple (B, G, R)
            The color of polyline in BGR order.
        '''
        if len(points) > 1:
            line_thickness = int(1 * self.resize_factor)
            cv2.polylines(
                image, [(np.array(points) * self.resize_factor).astype(int)],
                isClosed=False, color=color, thickness=line_thickness)

    def redraw(self):
        '''
        Redraw the image with points and polyline.
        '''
        if self.resized_image.shape == self.resized_empty.shape:
            self.resized_image[:] = self.resized_empty
        else:
            self.resized_image = self.resized_empty.copy()

        for roi_idx, roi in enumerate(self.rois):
            color = (0, 0, 255) if roi_idx != self.active_roi_idx else (0, 255, 255)

            for _idx, point in enumerate(roi):
                circle_radius = int(3 * self.resize_factor)
                cv2.circle(
                    self.resized_image,
                    np.array(point * self.resize_factor).astype(int),
                    circle_radius, color, -1)
            self.update_polyline(self.resized_image, roi, color)

            # Add text annotation for ROI number/index
            if roi:
                text_position = tuple(np.array(roi[0]) * self.resize_factor)
                text_position = (int(text_position[0]), int(text_position[1] - 10))
                cv2.putText(
                    self.resized_image,
                    f'ROI {roi_idx}',
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA
                )

        cv2.imshow('Select Line ROIs', self.resized_image)
        cv2.setWindowTitle(
            'Select Line ROIs',
            f'Select Line ROIs \"Mode: {self.current_mode}\"' + \
                ' (Space: change mode | Q: quit)')

    def mouse_callback(self, event, x, y, flags, param):
        '''
        Mouse callback function for handling mouse events.

        Parameters
        ----------
        event : int
            The mouse event.
        x : int
            The x-coordinate of the mouse cursor.
        y : int
            The y-coordinate of the mouse cursor.
        flags : int
            Additional flags for the mouse event.
        param : object
            Additional parameters.
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'Edit':
                for roi_idx, roi in enumerate(self.rois):
                    for idx, point in enumerate(roi):
                        dist = np.linalg.norm(
                            np.array(point) - [
                                x / self.resize_factor, y / self.resize_factor])
                        if dist < 10:
                            self.dragging = True
                            self.drag_idx = idx
                            self.active_roi_idx = roi_idx
                            break
                return
            elif self.current_mode == 'Add':
                self.rois.append([])
                self.active_roi_idx = len(self.rois) - 1
                self.rois[self.active_roi_idx].append(
                    np.array((x / self.resize_factor, y / self.resize_factor)))

                self.cycle_mode()
            elif self.current_mode == 'Append':
                if self.active_roi_idx != -1:
                    self.rois[self.active_roi_idx].append(
                        np.array((x / self.resize_factor, y / self.resize_factor)))
            elif self.current_mode == 'Remove':
                if self.active_roi_idx != -1:
                    remove_idx = self.remove_point(
                        x / self.resize_factor, y / self.resize_factor)
                    if remove_idx != -1:
                        del self.rois[self.active_roi_idx][remove_idx]

                        # Check if the ROI is empty after deletion
                        if not self.rois[self.active_roi_idx]:
                            del self.rois[self.active_roi_idx]

                            # Adjust active ROI index only if there are remaining ROIs
                            if self.rois:
                                self.active_roi_idx = max(0, self.active_roi_idx - 1)
                            else:
                                self.active_roi_idx = -1
            elif self.current_mode == 'Insert':
                if self.active_roi_idx != -1:
                    res = self.insert_index(
                        (x / self.resize_factor, y / self.resize_factor))
                    if res > 0:
                        self.rois[self.active_roi_idx].insert(
                            res, np.array(
                                (x / self.resize_factor, y / self.resize_factor)))
            self.redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and 0 <= self.drag_idx < len(
                    self.rois[self.active_roi_idx]):
                self.rois[self.active_roi_idx][self.drag_idx] = np.array(
                    (x / self.resize_factor, y / self.resize_factor))
            self.redraw()
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.resize_factor += 0.05
            else:
                self.resize_factor = max(0.1, self.resize_factor - 0.05)
            self.resized_empty = cv2.resize(
                self.original, (0, 0),
                fx=self.resize_factor, fy=self.resize_factor,
                interpolation=cv2.INTER_NEAREST)
            self.redraw()

    def cycle_mode(self):
        self.modes.put(self.current_mode)
        self.current_mode = self.modes.get()

    def select_line_rois(self, offset: tuple[int, int]=None):
        '''
        Start the interactive process of selecting multiple polyline ROIs.

        Parameters
        ----------
        offset : tuple[int, int]
            The points offset (x, y)

        Returns
        -------
        list
            List of arrays of points, where each array represents a polyline ROI.
        '''
        cv2.namedWindow('Select Line ROIs')
        cv2.setMouseCallback('Select Line ROIs', self.mouse_callback)

        while True:
            self.redraw()
            key = cv2.waitKeyEx(200)

            if key == ord('q') or key == ord('Q'):
                break
            elif key == 32:  # Space
                self.cycle_mode()
            elif key == 2490368:  # Up arrow
                if len(self.rois) > 0:
                    self.active_roi_idx = (self.active_roi_idx - 1) % len(self.rois)
                    self.redraw()
            elif key == 2621440:  # Down arrow
                if len(self.rois) > 0:
                    self.active_roi_idx = (self.active_roi_idx + 1) % len(self.rois)
                    self.redraw()
            elif key == 3014656:  # Delete
                if self.active_roi_idx != -1 and len(
                    self.rois[self.active_roi_idx]) > 0:
                    del self.rois[self.active_roi_idx][-1]

                    # Check if the ROI is empty after deletion
                    if not self.rois[self.active_roi_idx]:
                        del self.rois[self.active_roi_idx]

                        # Adjust active ROI index only if there are remaining ROIs
                        if self.rois:
                            self.active_roi_idx = max(0, self.active_roi_idx - 1)
                        else:
                            self.active_roi_idx = -1
            elif key == ord('a'):
                while self.current_mode != 'Add':
                    self.cycle_mode()
            elif key == ord('p'):
                while self.current_mode != 'Append':
                    self.cycle_mode()
            elif key == ord('e'):
                while self.current_mode != 'Edit':
                    self.cycle_mode()
            elif key == ord('i'):
                while self.current_mode != 'Insert':
                    self.cycle_mode()
            elif key == ord('r'):
                while self.current_mode != 'Remove':
                    self.cycle_mode()
            elif key == ord('c'):
                self.rois = []
                while self.current_mode != 'Add':
                    self.cycle_mode()

        cv2.destroyAllWindows()

        if offset is not None:
            return [np.array(roi) + offset for roi in self.rois]
        else:
            return [np.array(roi) for roi in self.rois]

    @staticmethod
    def get_selector(image: np.ndarray, resize_factor=1.0):
        if image is None:
            raise ValueError('Parameter image is None.')

        if image.ndim != 2:
            raise ValueError(
                f'Parameter image should be an array with ndim = 2, not {image.ndim}.')

        # Convert monochromatic image to RGB
        _image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Select a multi-point line ROI with the specified resize factor
        return MultiLineROISelector(_image, resize_factor)

@nb.njit
def get_move_lines(linewidth: int):
    '''
    Generate an array representing movement lines for a given linewidth.

    Parameters
    ----------
    linewidth : int
        The width of the line.

    Returns
    -------
    np.ndarray
        An array of movement lines. For linewidth 0 or 1, no movement is required, and
        the array contains only [0.0]. For other linewidths, the array includes
        adjusted movement lines based on the linewidth. If the linewidth is even,
        the central line is excluded.

    Notes
    -----
    The function is optimized for use with Numba (nb.njit).

    Examples
    --------
    >>> get_move_lines(1)
    array([0.0])

    >>> get_move_lines(3)
    array([-1.0, 0.0, 1.0])

    >>> get_move_lines(4)
    array([-1.5, -0.5, 0.5, 1.5])
    '''
    if linewidth <= 1:
        # For linewidth 0 or 1, no movement is required
        return np.array([0.0])

    half_width = linewidth // 2
    width_indices = np.arange(-half_width, half_width + 1)

    # Adjust for even linewidth to exclude the central line
    if linewidth % 2 == 0:
        width_indices = np.delete(width_indices, half_width)
        return width_indices * 0.5
    else:
        return width_indices * 1.0  # Ensure it's a Numba-compatible float

@nb.njit
def get_kymogram_row(
        Data: np.ndarray, X: np.ndarray, Y: np.ndarray,
        linewidth=1, method='average'):
    '''
    Generate a polyline ROI data to construct a kymogram from the given data
    along specified X and Y coordinates.

    Parameters
    ----------
    Data : np.ndarray
        2D array representing the data.
    X : np.ndarray
        1D array of X coordinates.
    Y : np.ndarray
        1D array of Y coordinates.
    linewidth : int, optional
        Width of the kymogram lines, by default 1.
    method : str, optional
        Method to generate the kymogram, either 'average' (default) or 'maximum'.

    Returns
    -------
    np.ndarray
        1D array representing a kymogram row.
    '''
    diffX, diffY = np.diff(X), np.diff(Y)
    lengths = np.hypot(diffX, diffY)

    dX = -diffY / lengths
    dY = diffX / lengths
    move_lines = get_move_lines(linewidth)

    n_points = lengths.astype(np.int64)
    lut = np.concatenate((np.array([0]), np.cumsum(n_points))).astype(np.int64)

    Roi = np.zeros(np.sum(n_points))

    for idx in range(X.shape[0] - 1):
        x = np.linspace(X[idx], X[idx + 1], n_points[idx])
        y = np.linspace(Y[idx], Y[idx + 1], n_points[idx])

        dx = dX[idx]
        dy = dY[idx]

        temp = np.zeros(n_points[idx])
        for move_line in move_lines:
            x_idx = np.round(x + move_line * dx).astype(np.int64)
            y_idx = np.round(y + move_line * dy).astype(np.int64)
            with nb.objmode():
                if method == 'maximum':
                    temp[:] = np.maximum(temp, Data[y_idx, x_idx])
                else:
                    temp[:] = temp + Data[y_idx, x_idx]
        if method == 'average' and linewidth > 1:
            Roi[lut[idx]:lut[idx+1]] = temp / len(move_lines)
        else:
            Roi[lut[idx]:lut[idx+1]] = temp

    return Roi

class Kymogram:
    def __init__(
            self, data: np.ndarray, rois: list[np.ndarray],
            linewidth: int, method: str, window: int, at_index: int,
            window_location: str) -> None:
        '''
        Initialize a Kymogram object.

        Parameters
        ----------
        data : np.ndarray
            The input data for kymogram generation.
        rois : np.ndarray
            The rois including points used for kymogram extraction.
        linewidth : int
            The linewidth for kymogram extraction.
        method : str
            The method used for kymogram generation.
        window : int
            The window size for kymogram extraction.
        at_index: int
            The frame index at which data was extracted.
        window_location: str
            The window location relative to at_index.
        '''
        self.data = data
        self.rois = rois
        self.linewidth = linewidth
        self.method = method
        self.window = window
        self.at_index = at_index
        self.window_location = window_location

    def get_metadata(self):
        '''
        Get metadata of the Kymogram object.

        Returns
        -------
        dict
            A dictionary containing metadata, including the original
            attributes and generated kymogram data.
        '''
        kymogram_dict = {
            'Kymogram': {
                'points': [points.tolist() for points in self.rois],
                'linewidth': self.linewidth,
                'method': self.method,
                'window': self.window,
                'window_location': self.window_location,
                'at_index': self.at_index,
            }
        }
        return kymogram_dict

    def export_to_json(self, filename):
        '''
        Export Kymogram metadata to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the JSON file.
        '''
        metadata = self.get_metadata()
        json_string = json.dumps(metadata, indent=2)

        with open(filename, 'w') as json_file:
            json_file.write(json_string)

    def to_tiff(self, filename):
        '''
        Export the kymogram data as a TIFF image with metadata.

        Parameters
        ----------
        filename : str
            The name of the TIFF file to be created.
        '''
        if self.data is None:
            raise ValueError('Kymogram data is not available.')

        # Create a dictionary containing metadata
        metadata = self.get_metadata()

        # Write the metadata to the TIFF file
        tf.imwrite(filename, self.data, bigtiff=True, metadata=metadata)

    @classmethod
    def from_json(cls, filename):
        '''
        Create a Kymogram object from a JSON file.

        Parameters
        ----------
        filename : str
            The name of the JSON file.

        Returns
        -------
        Kymogram or None
            A Kymogram object created from the metadata in the JSON file,
            or None if the 'Kymogram' key is not present.
        '''
        with open(filename) as json_file:
            metadata = json.load(json_file)

        kymogram_metadata = metadata.get('Kymogram', None)
        if kymogram_metadata is None:
            # 'Kymogram' key is not present
            return None

        rois = [np.array(roi) for roi in kymogram_metadata.get('points', [])]
        linewidth = kymogram_metadata.get('linewidth', 0)
        method = kymogram_metadata.get('method', 'N/A')
        window = kymogram_metadata.get('window', -1)
        window_location = kymogram_metadata.get(
            'window_location', 'From Current')
        at_index = kymogram_metadata.get('at_index', 0)

        kymogram = cls(
            None, rois, linewidth, method, window, at_index, window_location)

        return kymogram

    @classmethod
    def from_tiff(cls, filename):
        '''
        Create a Kymogram object from a TIFF file with metadata.

        Parameters
        ----------
        filename : str
            The name of the TIFF file.

        Returns
        -------
        Kymogram or None
            A Kymogram object created from the metadata in the TIFF file,
            or None if the 'Kymogram' key is not present.
        '''
        try:
            # Read TIFF file and extract metadata
            with tf.TiffFile(filename) as tiff:
                data = tiff.asarray()

            metadata = tf.tiffcomment(filename)

            # Parse JSON metadata
            metadata_dict = json.loads(metadata)

            kymogram_metadata = metadata_dict.get('Kymogram', None)
            if kymogram_metadata is None:
                # 'Kymogram' key is not present
                kymogram = cls(
                    data, np.array([]), 1, 'N/A', data.shape[0], 0, 'N/A')

                return kymogram
            else:
                rois = [
                    np.array(roi) for roi in kymogram_metadata.get('points', [])]
                linewidth = kymogram_metadata.get('linewidth', 0)
                method = kymogram_metadata.get('method', 'N/A')
                window = kymogram_metadata.get('window', -1)
                window_location = kymogram_metadata.get(
                    'window_location', 'From Current')
                at_index = kymogram_metadata.get('at_index', 0)

                kymogram = cls(
                    data, rois, linewidth, method, window, at_index, window_location)

                return kymogram

        except Exception as e:
            print(f'Error reading TIFF file: {e}')
            return None

class KymogramWidget(QGroupBox):
    extractClicked = pyqtSignal()
    displayClicked = pyqtSignal(np.ndarray)

    controls_description = [
        '<b>Q:</b> Quit',
        '<b>Space:</b> Cycle modes',
        '<b>A:</b> Add new ROI at cursor',
        '<b>P:</b> Append to active ROI',
        '<b>E:</b> Edit points (dragging)',
        '<b>I:</b> Insert at segment',
        '<b>R:</b> Remove points',
        '<b>C:</b> Clear all ROIs',
        '<b>Delete:</b> Delete the last added point',
        '<b>Up/Down Arrows:</b> Select active ROI.'
    ]

    def __init__(self, title: str, threadpool: QThreadPool, parent=None):
        super().__init__(title, parent)
        self._threadpool = threadpool

        self.kymogram_temporal_window = create_spin_box(0, 100, 1, 50)
        self.kymogram_linewidth = create_spin_box(1, 100, 1, 1)

        self.temporal_window_location = QComboBox()
        self.temporal_window_location.addItems(
            ['From Current', 'Current Central', 'To Current'])

        self.kymogram_roi_selector = QComboBox()
        self.kymogram_roi_selector.addItems(
            ['current', 'average', 'maximum', 'sum', 'std'])

        self.kymogram_roi_method = QComboBox()
        self.kymogram_roi_method.addItems(['average', 'maximum', 'sum'])

        self.kymogram_btn = QPushButton(
            'Extract', clicked=self.kymogram_btn_clicked)
        self.kymogram_display_btn = QPushButton(
            'Display', clicked=self.kymogram_display_clicked)
        self.kymogram_save_btn = QPushButton(
            'Save', clicked=self.kymogram_save_clicked)
        self.kymogram_load_btn = QPushButton(
            'Load', clicked=self.kymogram_load_clicked)

        self.previous_selector = create_check_box('Use previous ROI selector?', True)

        kymogram_btns = create_hbox_layout(
            self.kymogram_btn,
            self.kymogram_display_btn,
            self.kymogram_save_btn,
            self.kymogram_load_btn
        )

        kymo_layout = QFormLayout(self)
        kymo_layout.addRow(
            QLabel('Temporal Window [frames]:'), self.kymogram_temporal_window)
        kymo_layout.addRow(
            QLabel('Window Range:'), self.temporal_window_location)
        kymo_layout.addRow(
            QLabel('Selector Displays:'), self.kymogram_roi_selector)
        kymo_layout.addRow(
            QLabel('ROI Linewidth [pixels]:'), self.kymogram_linewidth)
        kymo_layout.addRow(
            QLabel('ROI Extracts:'), self.kymogram_roi_method)
        kymo_layout.addWidget(self.previous_selector)
        kymo_layout.addRow(
            kymogram_btns)

        description_html = '; '.join(
            KymogramWidget.controls_description)
        wrapped_label = QLabel(
            '<b>ROI Selector Controls:</b><br>' + description_html)
        wrapped_label.setAlignment(Qt.AlignTop)
        wrapped_label.setWordWrap(True)
        kymo_layout.addRow(wrapped_label)

        # Create a QTableWidget
        self.tableWidget = QTableWidget()

        # Set the column count
        self.tableWidget.setColumnCount(3)

        # Set the headers
        self.tableWidget.setHorizontalHeaderLabels(['ROI index', 'X', 'Y'])
        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        kymo_layout.addRow(self.tableWidget)


        self._kymogram: Kymogram = None  # Assuming you have a Kymogram instance

    def setMaximum(self, value: int):
        self.kymogram_temporal_window.setMaximum(value)

    def extract_kymogram(
            self,
            tiffSeq_Handler: Union[ZarrImageSequence, TiffSeqHandler],
            current_frame: int, max_frame: int,
            roi_info: tuple[int, int]=None):
        if tiffSeq_Handler is None:
            return

        try:
            window_location = self.temporal_window_location.currentText()
            window_range = self.get_kymogram_window(current_frame, max_frame)

            linewidth = self.kymogram_linewidth.value()

            roi_extracts = self.kymogram_roi_method.currentText()

            previous_selector = self.previous_selector.isChecked()
            if previous_selector and self._kymogram is not None:
                if roi_info is not None:
                    origin, _ = roi_info
                else:
                    origin = (0, 0)
                old_rois = [
                    [np.array(point) - origin for point in roi.tolist()
                     ] for roi in self._kymogram.rois]
            else:
                old_rois = None

            def work_func():
                try:
                    image = uImage(self.get_selector_image(
                        tiffSeq_Handler, current_frame, window_range))

                    image.equalizeLUT()

                    if roi_info is not None:
                        origin, dim = roi_info
                        view = image._view[
                            int(origin[1]):int(origin[1] + dim[1]),
                            int(origin[0]):int(origin[0] + dim[0])]
                    else:
                        origin = None
                        view = image._view

                    scale_factor = get_scaling_factor(view.shape[0], view.shape[1])

                    selector = MultiLineROISelector.get_selector(view, scale_factor)
                    if old_rois:
                        selector.rois = old_rois

                    rois = selector.select_line_rois(origin)

                    if rois:
                        res = None
                        zero = np.zeros(1)
                        t_points = window_range[1] - window_range[0] + 1
                        for frame_idx in range(window_range[0], window_range[1] + 1):
                            data = tiffSeq_Handler.getSlice(
                                frame_idx, 0, 0)
                            rois_row = None
                            for points in rois:
                                if len(points) > 1:
                                    row = get_kymogram_row(
                                        data,
                                        points[:, 0].flatten(),  # X points
                                        points[:, 1].flatten(),  # Y points
                                        linewidth,
                                        roi_extracts
                                    )
                                    rois_row = row if rois_row is None else \
                                        np.concatenate([rois_row, zero, row])
                            res = rois_row if res is None else np.vstack(
                                [res, rois_row])
                            print(
                                'Generating Kymogram ... {:.2f}%'.format(
                                    100 * (res.shape[0] if res.ndim ==2 else 1
                                     )/t_points), end='\r')
                        return Kymogram(
                            res, rois, linewidth, roi_extracts,
                            t_points, current_frame, window_location)
                    else:
                        return None
                except Exception:
                    traceback.print_exc()
                    return None

            def done(results: Kymogram):
                self.kymogram_btn.setDisabled(False)
                if results is not None:
                    self._kymogram = results
                    self.displayClicked.emit(results.data)

            self.worker = thread_worker(
                work_func,
                progress=False, z_stage=False)
            self.worker.signals.result.connect(done)
            # Execute
            self.kymogram_btn.setDisabled(True)
            self._threadpool.start(self.worker)
        except Exception:
            traceback.print_exc()

    def kymogram_display_clicked(self):
        if self._kymogram is None:
            return

        self.displayClicked.emit(self._kymogram.data)

    def kymogram_btn_clicked(self):
        self.extractClicked.emit()

    def kymogram_save_clicked(self):
        if self._kymogram is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Export Kymogram',
            filter='Tiff files (*.tif);;')

        if len(filename) > 0:
            self._kymogram.to_tiff(filename)

    def kymogram_load_clicked(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Load Kymogram',
            filter='Tiff files (*.tif);;')

        if len(filename) > 0:
            kymogram = Kymogram.from_tiff(filename)
            if kymogram is not None:
                self._kymogram = kymogram
                self.kymogram_linewidth.setValue(
                    kymogram.linewidth)
                self.kymogram_temporal_window.setValue(
                    kymogram.window)

                index = self.kymogram_roi_method.findText(
                    kymogram.method)

                if index != -1:
                    self.kymogram_roi_method.setCurrentIndex(index)

                index = self.temporal_window_location.findText(
                    kymogram.window_location)

                if index != -1:
                    self.temporal_window_location.setCurrentIndex(index)

                self.displayClicked.emit(self._kymogram.data)


    def get_kymogram_window(
            self, current_frame: int, max_frame: int):
        '''
        Get the temporal window for generating a kymogram based on the selected option.

        Parameters
        ----------
        current_frame : int
            The current frame index of the stack.
        max_frame : int
            The maximum frame index of the stack.
        Returns
        -------
        Tuple[int, int]:
            A tuple representing the start and end frames
            of the kymogram window.
        '''
        selected_option = self.temporal_window_location.currentText()

        n_frames = self.kymogram_temporal_window.value()

        if max_frame + 1 <= n_frames:
            # return the entire range.
            return 0, max_frame
        else:
            if selected_option == 'To Current':
                # Return frames from (current_frame - n_frames) to current_frame
                start_frame = max(0, current_frame - n_frames)
                return start_frame, current_frame
            elif selected_option == 'Current Central':
                # Return frames centered around current_frame (window size = n_frames)
                half_window = n_frames // 2
                start_frame = max(0, current_frame - half_window)
                end_frame = min(max_frame, current_frame + half_window)
                return start_frame, end_frame
            else:  # Assuming the third option is 'From Current'
                # Return frames from current_frame to (current_frame + n_frames)
                start_frame = current_frame
                end_frame = min(max_frame, current_frame + n_frames)
                return start_frame, end_frame

    def get_selector_image(
            self,
            tiffSeq_Handler: Union[ZarrImageSequence, TiffSeqHandler],
            current_frame: int,
            window_range: tuple[int, int]):
        condition = self.kymogram_roi_selector.currentText()

        if condition == 'current':
            return tiffSeq_Handler.getSlice(current_frame, 0, 0)
        elif condition == 'average':
            return np.mean(
                tiffSeq_Handler.getSlice(
                    slice(window_range[0], window_range[1] + 1),
                    0, 0),
                axis=0
                )
        elif condition == 'maximum':
            return np.max(
                tiffSeq_Handler.getSlice(
                    slice(window_range[0], window_range[1] + 1),
                    0, 0),
                axis=0
                )
        elif condition == 'sum':
            return np.sum(
                tiffSeq_Handler.getSlice(
                    slice(window_range[0], window_range[1] + 1),
                    0, 0),
                axis=0
                )
        elif condition == 'std':
            return np.std(
                tiffSeq_Handler.getSlice(
                    slice(window_range[0], window_range[1] + 1),
                    0, 0),
                axis=0
                )

        return image

if __name__ == '__main__':
    # Example usage
    image = np.random.choice(range(256), (512, 512)).astype(np.uint8)

    resize_factor = 1

    selector = MultiLineROISelector.get_selector(image, resize_factor)

    # Select a multi-point line ROI with the specified resize factor
    roi_points = selector.select_line_rois()

    # Draw the selected line on the resized image
    if len(roi_points) > 1:
        for roi in roi_points:
            if  len(roi) > 1:
                #-- Plot...
                fig, axes = plt.subplots(nrows=2)
                axes[0].imshow(image)
                axes[0].plot(roi[:, 0], roi[:, 1], 'ro-')
                axes[0].axis('image')

                # linewidth
                linewidth = 1
                axes[1].plot(
                    get_kymogram_row(
                        image,
                        roi[:, 0].flatten(),  # X points
                        roi[:, 1].flatten(),  # Y points
                        linewidth),
                    label=f'linewidth={linewidth}')
                plt.legend()
                plt.show()
            else:
                print('Error: Please select at least two points.')
