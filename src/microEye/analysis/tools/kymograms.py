import json
from queue import Queue

import cv2
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import tifffile as tf
from pyqtgraph.functions import interpolateArray


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


def insert_index(cursor, points, threshold=3):
    '''
    Find the index to insert a new point based on cursor proximity to line segments.

    Parameters
    ----------
    cursor : Tuple (x, y)
        The cursor position represented as a tuple of (x, y) coordinates.
    points : list of Tuple (x, y)
        List of previously added points,
        each represented as a tuple of (x, y) coordinates.
    threshold : float, optional
        Threshold distance for considering the cursor above a line segment.
        Default is 3.0.

    Returns
    -------
    int
        The index at which to add the point if the cursor is above a line segment,
        otherwise -1.
    '''
    x, y = cursor
    n_points = len(points)

    # Check if there are at least two points to form line segments
    if n_points < 2:
        return -1

    distances = []
    indices = []

    # Iterate through existing points to calculate distances to line segments
    for idx in range(n_points - 1):
        x1, y1 = points[idx]
        x2, y2 = points[idx + 1]

        # Calculate the distance from the cursor to the line segment
        distance = np.abs(
            (y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1
        ) / np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

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


def remove_point(cursor, points, threshold=5.0):
    '''
    Find the index of the point to remove based on cursor proximity.

    Parameters
    ----------
    cursor : Tuple (x, y)
        The cursor position represented as a tuple of (x, y) coordinates.
    points : list of Tuple (x, y)
        List of previously added points,
        each represented as a tuple of (x, y) coordinates.
    threshold : float, optional
        Threshold distance for considering the cursor above a point.
        Default is 5.0.

    Returns
    -------
    int
        The index of the point to remove if the cursor is above a point,
        otherwise -1.
    '''
    x, y = cursor

    # Iterate through existing points to find the index of the point to remove
    for idx, point in enumerate(points):
        dist = np.linalg.norm(np.array(point) - [x, y])
        if dist < threshold:
            return idx

    # If no suitable point found, return -1
    return -1


def select_line_roi(image, resize_factor=1.0):
    original = image.copy()
    resized_empty = cv2.resize(
        original, (0, 0), fx=resize_factor, fy=resize_factor,
        interpolation=cv2.INTER_NEAREST)
    resized_image = resized_empty.copy()
    points = []

    modes = Queue()
    for mode in ['Add', 'Edit', 'Insert', 'Remove']:
        modes.put(mode)
    current_mode = modes.get()  # track the current mode (adding or removing)

    dragging = False  # Flag to track if dragging is in progress
    drag_idx = -1  # Index of the point being dragged

    def update_polyline(image):
        # Draw the polyline with the updated points
        if len(points) > 1:
            line_thickness = int(1 * resize_factor)  # Scale the line thickness
            cv2.polylines(
                image,
                [(np.array(points) * resize_factor).astype(int)],
                isClosed=False, color=(0, 255, 255), thickness=line_thickness)

    def redraw():
        nonlocal resized_image
        # Clear the image and redraw the points and polyline
        if resized_image.shape == resized_empty.shape:
            resized_image[:] = resized_empty
        else:
            resized_image = resized_empty.copy()
        for idx, point in enumerate(points):
            circle_radius = int(3 * resize_factor)
            cv2.circle(
                resized_image, np.array(point * resize_factor).astype(int),
                circle_radius,
                (0, 255, 255) if idx != len(points) - 1 else (0, 0, 255),
                -1)
        update_polyline(resized_image)

        cv2.imshow('Select Line ROI', resized_image)

        # Update window title to reflect the mode
        cv2.setWindowTitle(
            'Select Line ROI',
            f'Select Line ROI \"Mode: {current_mode}\" (Space: change mode | Q: quit)')

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, current_mode, dragging, drag_idx, resize_factor, resized_empty
        if event == cv2.EVENT_LBUTTONDOWN:
            if current_mode == 'Edit':
                # If right button is clicked, find the closest point for dragging
                for idx, point in enumerate(points):
                    dist = np.linalg.norm(
                        np.array(point) - [x / resize_factor, y / resize_factor])
                    if dist < 10:  # You can adjust this threshold as needed
                        dragging = True
                        drag_idx = idx
                        break
                return
            elif current_mode == 'Add':
                # If left button is clicked and in adding mode, add a new point
                points.append(np.array((x / resize_factor, y / resize_factor)))
            elif current_mode == 'Remove':
                # If left button is clicked and in removing mode
                remove_idx = remove_point(
                    (x / resize_factor, y / resize_factor), points)
                if remove_idx != -1:
                    del points[remove_idx]
            elif current_mode == 'Insert':
                res = insert_index((x / resize_factor, y / resize_factor), points)
                if res > 0:
                    points.insert(
                        res,
                        np.array((x / resize_factor, y / resize_factor)))
            redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            # Release the dragging flag
            dragging = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                # If dragging, update the position of the dragged point
                points[drag_idx] = np.array(
                    (x / resize_factor, y / resize_factor))
            redraw()
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Check the direction of the scroll
            if flags > 0:
                # Scroll up, zoom in
                resize_factor += 0.05
            else:
                # Scroll down, zoom out
                resize_factor = max(0.1, resize_factor - 0.05)
            resized_empty = cv2.resize(
                original, (0, 0), fx=resize_factor, fy=resize_factor,
                interpolation=cv2.INTER_NEAREST)
            redraw()

    cv2.namedWindow('Select Line ROI')
    cv2.setMouseCallback('Select Line ROI', mouse_callback)

    while True:
        redraw()
        key = cv2.waitKey(200) & 0xFF

        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):  # Check for the Space key to toggle modes
            modes.put(current_mode)
            current_mode = modes.get()
        elif key == ord('r'):
            del points[-1]

    cv2.destroyAllWindows()

    return np.array(points)

def select_polyline_roi(image: np.ndarray, resize_factor=1.0):
    if image is None:
        raise ValueError('Parameter image is None.')

    if image.ndim != 2:
        raise ValueError(
            f'Parameter image should be an np.ndarray with ndim = 2, not {image.ndim}.')

    # Convert monochromatic image to RGB
    _image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Select a multi-point line ROI with the specified resize factor
    return select_line_roi(_image, resize_factor)

class Kymogram:
    def __init__(
            self, data: np.ndarray, points: np.ndarray,
            linewidth: int, method: str, window: int) -> None:
        '''
        Initialize a Kymogram object.

        Parameters
        ----------
        data : np.ndarray
            The input data for kymogram generation.
        points : np.ndarray
            The points used for kymogram extraction.
        linewidth : int
            The linewidth for kymogram extraction.
        method : str
            The method used for kymogram generation.
        window : int
            The window size for kymogram extraction.
        '''
        self.data = data
        self.points = points
        self.linewidth = linewidth
        self.method = method
        self.window = window

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
                'points': self.points.tolist(),
                'linewidth': self.linewidth,
                'method': self.method,
                'window': self.window,
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

        points = np.array(kymogram_metadata.get('points', []))
        linewidth = kymogram_metadata.get('linewidth', 0)
        method = kymogram_metadata.get('method', 'N/A')
        window = kymogram_metadata.get('window', -1)

        kymogram = cls(None, points, linewidth, method, window)

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
                metadata = tiff[0].image_description

            # Parse JSON metadata
            metadata_dict = json.loads(metadata)

            kymogram_metadata = metadata_dict.get('Kymogram', None)
            if kymogram_metadata is None:
                # 'Kymogram' key is not present
                kymogram = cls(data, np.array([]), 1, 'N/A', data.shape[0])

                return kymogram
            else:
                points = np.array(kymogram_metadata.get('points', []))
                linewidth = kymogram_metadata.get('linewidth', 0)
                method = kymogram_metadata.get('method', 'N/A')
                window = kymogram_metadata.get('window', -1)

                kymogram = cls(data, points, linewidth, method, window)

                return kymogram

        except Exception as e:
            print(f'Error reading TIFF file: {e}')
            return None


if __name__ == '__main__':
    # Example usage
    image = np.random.choice(range(256), (512, 512)).astype(np.uint8)

    resize_factor = 1

    # Select a multi-point line ROI with the specified resize factor
    roi_points = select_polyline_roi(image, resize_factor)

    # Draw the selected line on the resized image
    if len(roi_points) > 1:
        #-- Plot...
        fig, axes = plt.subplots(nrows=2)
        axes[0].imshow(image)
        axes[0].plot(roi_points[:, 0], roi_points[:, 1], 'ro-')
        axes[0].axis('image')

        # linewidth
        linewidth = 1
        axes[1].plot(
            get_kymogram_row(
                image,
                roi_points[:, 0].flatten(),  # X points
                roi_points[:, 1].flatten(),  # Y points
                linewidth),
            label=f'linewidth={linewidth}')
        plt.legend()
        plt.show()
    else:
        print('Error: Please select at least two points.')
