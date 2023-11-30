
from queue import Queue

import cv2
import numpy as np


class MultiRectangularROISelector:
    '''
    Class for interactively selecting and manipulating
    multiple rectangular ROIs in an image.

    Parameters
    ----------
    image_data : np.ndarray
        The original image data.
    resize_factor : float, optional
        The resize factor for the image. Defaults to 1.0.
    '''

    def __init__(self, image_data: np.ndarray, resize_factor=1.0):
        '''
        Initialize the MultiRectangularROISelector instance.

        Parameters
        ----------
        image_data : np.ndarray
            The original image data.
        resize_factor : float, optional
            The resize factor for the image. Defaults to 1.0.
        '''
        self.original = image_data.copy()
        self.shape = self.original.shape
        self.resize_factor = resize_factor
        self.resized_empty: np.ndarray = cv2.resize(
            self.original, (0, 0),
            fx=self.resize_factor, fy=self.resize_factor,
            interpolation=cv2.INTER_NEAREST)
        self.resized_image = self.resized_empty.copy()

        self.rois = []  # stores multiple ROIs, represented as a list [x1, y1, x2, y2]
        self.active_roi_idx = -1  # Index of the currently active ROI
        self.dragging = False
        self.drag_idx = -1
        self.modes = Queue()
        for mode in ['Add', 'Edit', 'Insert', 'Remove']:
            self.modes.put(mode)
        self.current_mode = self.modes.get()

    def update_rectangles(self, image, rectangles, color):
        '''
        Update the image with the specified rectangles.

        Parameters
        ----------
        image : np.ndarray
            The image to be updated.
        rectangles : List[List[float]]
            List of rectangles represented as [x1, y1, x2, y2].
        color : Tuple[int, int, int]
            Color for drawing rectangles.

        Returns
        -------
        None
        '''
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            cv2.rectangle(
                image,
                (int(x1 * self.resize_factor), int(y1 * self.resize_factor)),
                (int(x2 * self.resize_factor), int(y2 * self.resize_factor)),
                color,
                int(1 * self.resize_factor),
            )

            # Add text annotation for ROI number/index
            text_position = (
                int(x1 * self.resize_factor),
                int(y1 * self.resize_factor) - 10,
            )
            cv2.putText(
                image,
                f'ROI {self.rois.index(rect)}',
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_edit_circles(self, image, rect, color):
        '''
        Draw edit circles at the corners of the rectangle.

        Parameters
        ----------
        image : np.ndarray
            The image to draw on.
        rect : List[float]
            The rectangle coordinates.
        color : Tuple[int, int, int]
            Color for drawing circles.

        Returns
        -------
        None
        '''
        x1, y1, x2, y2 = rect
        circle_radius = int(3 * self.resize_factor)

        # Draw circles at the corners of the rectangle
        for corner in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            cv2.circle(
                image,
                (
                    int(corner[0] * self.resize_factor),
                    int(corner[1] * self.resize_factor)),
                circle_radius,
                color,
                -1,
            )

    def redraw(self):
        '''
        Redraw the image with rectangles and edit circles.

        Returns
        -------
        None
        '''
        if self.resized_image.shape == self.resized_empty.shape:
            self.resized_image[:] = self.resized_empty
        else:
            self.resized_image = self.resized_empty.copy()

        for roi_idx, roi in enumerate(self.rois):
            color = (255, 0, 0) if roi_idx != self.active_roi_idx else (0, 255, 255)

            self.draw_edit_circles(self.resized_image, roi, color)
            self.update_rectangles(self.resized_image, [roi], color)

        cv2.imshow('Select Rectangular ROIs', self.resized_image)
        cv2.setWindowTitle(
            'Select Rectangular ROIs',
            f'Select Rectangular ROIs \"Mode: {self.current_mode}\"',
        )

    def mouse_callback(self, event, x, y, flags, param):
        '''
        Mouse callback function for handling user interactions.

        Returns
        -------
        None
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == 'Edit':
                if self.active_roi_idx >= 0 and self.active_roi_idx < len(self.rois):
                        self.drag_idx = self.is_hit(x, y)
                        self.dragging = self.drag_idx != -1
            elif self.current_mode == 'Add':
                if not self.dragging:
                    self.rois.append([
                        x / self.resize_factor, y / self.resize_factor,
                        x / self.resize_factor, y / self.resize_factor])
                    self.dragging = True
                    self.drag_idx = 2  # Assign drag index to bottom-right corner
                    self.active_roi_idx = len(self.rois) - 1
            elif self.current_mode == 'Insert':
                if not self.dragging:
                    if self.active_roi_idx == -1:
                        # Add the first ROI
                        self.rois.append([
                            x / self.resize_factor, y / self.resize_factor,
                            x / self.resize_factor, y / self.resize_factor])
                        self.dragging = True
                        self.drag_idx = 2  # Assign drag index to bottom-right corner
                    else:
                        # For subsequent ROIs, set initial size to match the first ROI
                        x1, y1, x2, y2 = self.rois[-1]  # Get the first ROI
                        self.rois.append([
                            x / self.resize_factor, y / self.resize_factor,
                            (x + (x2 - x1)) / self.resize_factor,
                            (y + (y2 - y1)) / self.resize_factor])
                    self.active_roi_idx = len(self.rois) - 1
            elif self.current_mode == 'Remove':
                if self.active_roi_idx >= 0 and self.active_roi_idx < len(self.rois):
                    if self.is_hit(x, y) > -1:
                        self.delete_active()
            self.redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            if self.active_roi_idx >= 0:
                x1, y1, x2, y2 = self.rois[self.active_roi_idx]
                if (x1 == x2 and y1 == y2) or \
                        x2 > self.shape[1] or y2 > self.shape[0]:
                    self.delete_active()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and 0 <= self.drag_idx < 4:
                offset = 2 / self.resize_factor
                x1, y1, x2, y2 = self.rois[self.active_roi_idx]
                x_ratio, y_ratio = x / self.resize_factor, y / self.resize_factor

                if self.drag_idx == 0:
                    self.rois[self.active_roi_idx][0] = min(x_ratio, x2 - offset)
                    self.rois[self.active_roi_idx][1] = min(y_ratio, y2 - offset)
                elif self.drag_idx == 1:
                    self.rois[self.active_roi_idx][2] = max(x_ratio, x1 + offset)
                    self.rois[self.active_roi_idx][1] = min(y_ratio, y2 - offset)
                elif self.drag_idx == 2:
                    self.rois[self.active_roi_idx][2] = max(x_ratio, x1 + offset)
                    self.rois[self.active_roi_idx][3] = max(y_ratio, y1 + offset)
                elif self.drag_idx == 3:
                    self.rois[self.active_roi_idx][0] = min(x_ratio, x2 - offset)
                    self.rois[self.active_roi_idx][3] = max(y_ratio, y1 + offset)
            self.redraw()
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.resize_factor += 0.05
            else:
                self.resize_factor = max(0.1, self.resize_factor - 0.05)
            self.resized_empty = cv2.resize(
                self.original,
                (0, 0),
                fx=self.resize_factor,
                fy=self.resize_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            self.redraw()

    def is_hit(self, x: int, y: int):
        '''
        Check if the point (x, y) is within the active rectangle.

        Parameters
        ----------
        x : int
            X-coordinate of the point.
        y : int
            Y-coordinate of the point.

        Returns
        -------
        int
            Index of the hit corner, or -1 if not hit.
        '''
        x1, y1, x2, y2 = self.rois[self.active_roi_idx]
        delta = np.sqrt(
            (np.array([x1, x2, x2, x1]) - x / self.resize_factor)**2 + \
            (np.array([y1, y1, y2, y2]) - y / self.resize_factor)**2)
        return np.argmin(delta) if any(delta < 10) else -1

    def delete_active(self):
        '''
        Delete the currently active rectangle.

        Returns
        -------
        None
        '''
        if 0 <= self.active_roi_idx < len(self.rois):
            del self.rois[self.active_roi_idx]

            # Adjust active ROI index only if there are remaining ROIs
            if self.rois:
                self.active_roi_idx = max(0, self.active_roi_idx - 1)
            else:
                self.active_roi_idx = -1

    def cycle_mode(self):
        '''
        Cycle through different modes of operation.

        Returns
        -------
        None
        '''
        self.modes.put(self.current_mode)
        self.current_mode = self.modes.get()

    def select_rectangular_rois(self, offset: tuple[int, int] = None):
        '''
        Select rectangular ROIs interactively.

        Parameters
        ----------
        offset : Tuple[int, int], optional
            Offset to apply to selected ROIs.

        Returns
        -------
        List[np.ndarray]
            List of selected ROIs as NumPy arrays.
        '''
        cv2.namedWindow('Select Rectangular ROIs')
        cv2.setMouseCallback('Select Rectangular ROIs', self.mouse_callback)

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
                if self.active_roi_idx != -1:
                    self.delete_active()
            elif key == ord('a'):
                while self.current_mode != 'Add':
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

        cv2.destroyAllWindows()

        if offset is not None:
            return [
                np.array([
                    rect[0] + offset[0], rect[1] + offset[1],
                    rect[2] + offset[0], rect[3] + offset[1]])
                for rect in self.rois
            ]
        else:
            return [np.array(rect) for rect in self.rois]

    @staticmethod
    def get_selector(image: np.ndarray, resize_factor=1.0):
        '''
        Get a MultiRectangularROISelector instance for the specified image.

        Parameters
        ----------
        image : np.ndarray
            The image for ROI selection.
        resize_factor : float, optional
            The resize factor for the image. Defaults to 1.0.

        Returns
        -------
        MultiRectangularROISelector
            An instance of the selector.
        '''
        if image is None:
            raise ValueError('Parameter image is None.')

        if image.ndim != 2:
            raise ValueError(
                f'Parameter image should be an array with ndim = 2, not {image.ndim}.')

        # Convert monochromatic image to RGB
        _image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Select a multi-rectangle ROI with the specified resize factor
        return MultiRectangularROISelector(_image, resize_factor)

if __name__ == '__main__':
    # Example usage
    image = np.random.choice(range(256), (512, 512)).astype(np.uint8)

    resize_factor = 1

    selector = MultiRectangularROISelector.get_selector(image, resize_factor)

    # Select a multi-point line ROI with the specified resize factor
    roi_points = selector.select_rectangular_rois()

    # Draw the selected line on the resized image
    if len(roi_points) > 1:
        pass
    #     for roi in roi_points:
    #         if  len(roi) > 1:
    #             #-- Plot...
    #             fig, axes = plt.subplots(nrows=2)
    #             axes[0].imshow(image)
    #             axes[0].plot(roi[:, 0], roi[:, 1], 'ro-')
    #             axes[0].axis('image')

    #             # linewidth
    #             linewidth = 1
    #             axes[1].plot(
    #                 get_kymogram_row(
    #                     image,
    #                     roi[:, 0].flatten(),  # X points
    #                     roi[:, 1].flatten(),  # Y points
    #                     linewidth),
    #                 label=f'linewidth={linewidth}')
    #             plt.legend()
    #             plt.show()
    #         else:
    #             print('Error: Please select at least two points.')
