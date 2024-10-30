import numba
import numpy as np


@numba.jit(nopython=True)
def compute_point_cloud_histogram(data, bin_edges_x, bin_edges_y, bin_edges_z):
    '''
    Compute the 3D histogram using numba for acceleration.
    Returns coordinates and intensities of non-zero bins.
    '''
    # Initialize arrays to store results
    max_points = len(data)  # Maximum possible number of points
    points = np.zeros((max_points, 3), dtype=np.float64)
    intensities = np.zeros(max_points, dtype=np.float64)
    point_count = 0

    # Create temporary volume for binning
    hist = np.zeros((len(bin_edges_z) - 1, len(bin_edges_y) - 1, len(bin_edges_x) - 1))

    # Accumulate points into bins
    for x, y, z, intensity in data:
        # Find bin indices
        x_idx = np.searchsorted(bin_edges_x, x) - 1
        y_idx = np.searchsorted(bin_edges_y, y) - 1
        z_idx = np.searchsorted(bin_edges_z, z) - 1

        # Check if point is within bounds
        if (
            0 <= x_idx < len(bin_edges_x) - 1
            and 0 <= y_idx < len(bin_edges_y) - 1
            and 0 <= z_idx < len(bin_edges_z) - 1
        ):
            hist[z_idx, y_idx, x_idx] += intensity

    # Extract non-zero bins
    for z_idx in range(len(bin_edges_z) - 1):
        for y_idx in range(len(bin_edges_y) - 1):
            for x_idx in range(len(bin_edges_x) - 1):
                if hist[z_idx, y_idx, x_idx] > 0:
                    # Calculate center coordinates of the bin
                    x_center = (bin_edges_x[x_idx] + bin_edges_x[x_idx + 1]) / 2
                    y_center = (bin_edges_y[y_idx] + bin_edges_y[y_idx + 1]) / 2
                    z_center = (bin_edges_z[z_idx] + bin_edges_z[z_idx + 1]) / 2

                    points[point_count] = np.array([x_center, y_center, z_center])
                    intensities[point_count] = hist[z_idx, y_idx, x_idx]
                    point_count += 1

    return points[:point_count], intensities[:point_count]


class PointCloudRenderer:
    '''
    Point cloud renderer for rendering 3D super resolution images as binned point clouds
    '''

    def __init__(self, xy_bin_size=10, z_bin_size=50):
        '''
        Initialize the point cloud renderer.

        Parameters:
        -----------
        xy_bin_size : float
            Size of each bin in the lateral (XY) dimensions
        z_bin_size : float
            Size of each bin in the axial (Z) dimension
        '''
        self._xy_bin_size = xy_bin_size
        self._z_bin_size = z_bin_size
        self._points = None
        self._intensities = None

    def _validate_inputs(self, X, Y, Z, Intensity):
        '''
        Validate the inputs for rendering.
        '''
        if not len(X) == len(Y) == len(Z) == len(Intensity):
            raise ValueError('The supplied coordinate arrays are of different lengths.')

    def _normalize_xy(self, X, Y):
        '''
        Normalize XY coordinates to start from zero.
        '''
        x_min, y_min = np.min(X), np.min(Y)

        if x_min < 0:
            X = X - x_min
        if y_min < 0:
            Y = Y - y_min

        return X, Y

    def render(self, X, Y, Z, Intensity, shape=None):
        '''
        Generates a point cloud from 3D single-molecule localizations using binning.

        Parameters:
        -----------
        X : np.ndarray
            Array of X coordinates for the localizations.
        Y : np.ndarray
            Array of Y coordinates for the localizations.
        Z : np.ndarray
            Array of Z coordinates for the localizations (preserving the zero reference)
        Intensity : np.ndarray
            Array of intensity values corresponding to each localization.
        shape : tuple[int, int, int], optional
            Number of bins in each dimension as (depth, height, width).
            Defaults to None.

        Returns:
        --------
        points : np.ndarray
            Nx3 array of point coordinates representing non-zero bins
        intensities : np.ndarray
            N array of intensity values for each point
        metadata : dict
            Dictionary containing bin sizes and coordinate ranges
        '''
        self._validate_inputs(X, Y, Z, Intensity)
        X, Y = self._normalize_xy(X, Y)

        # Calculate bin edges
        if shape is None:
            x_max = np.max(X) + 2 * self._xy_bin_size
            y_max = np.max(Y) + 2 * self._xy_bin_size

            # For Z, calculate number of bins needed on each side of zero
            z_min, z_max = np.min(Z), np.max(Z)

            # Handle cases where all values fall within one bin centered at zero
            if -self._z_bin_size / 2 <= z_min and z_max <= self._z_bin_size / 2:
                n_bins_below = 1
                n_bins_above = (
                    1  # Create one bin centered at zero [-bin_size/2, +bin_size/2]
                )
            else:
                # Calculate bins needed on each side, accounting for half bins
                n_bins_below = int(
                    np.ceil((-z_min + self._z_bin_size / 2) / self._z_bin_size)
                )
                n_bins_above = int(
                    np.ceil((z_max + self._z_bin_size / 2) / self._z_bin_size)
                )

            n_bins_x = int(np.ceil(x_max / self._xy_bin_size))
            n_bins_y = int(np.ceil(y_max / self._xy_bin_size))
            n_bins_z = n_bins_below + n_bins_above
            shape = (n_bins_z, n_bins_y, n_bins_x)

        # Create bin edges
        bin_edges_x = np.linspace(0, shape[2] * self._xy_bin_size, shape[2] + 1)
        bin_edges_y = np.linspace(0, shape[1] * self._xy_bin_size, shape[1] + 1)
        bin_edges_z = np.arange(
            -n_bins_below * self._z_bin_size
            - self._z_bin_size / 2,  # start at first bin edge
            (n_bins_above + 1) * self._z_bin_size
            - self._z_bin_size / 2,  # end at last bin edge
            self._z_bin_size,
        )

        # Compute point cloud histogram
        points, intensities = compute_point_cloud_histogram(
            np.c_[X, Y, Z, Intensity], bin_edges_x, bin_edges_y, bin_edges_z
        )

        self._points = points
        self._intensities = intensities

        metadata = {
            'bin_size': {
                'x': self._xy_bin_size,
                'y': self._xy_bin_size,
                'z': self._z_bin_size,
            },
            'coordinates': {
                'z_min': bin_edges_z[0],
                'z_max': bin_edges_z[-1],
            },
            'point_count': len(points),
        }

        return points, intensities, metadata

    def from_array(self, data: np.ndarray, shape=None):
        '''
        Renders a point cloud from an array of localizations.

        Parameters:
        -----------
        data : np.ndarray
            Array with columns (X, Y, Z, Intensity)
        shape : tuple[int, int, int], optional
            Number of bins in each dimension (depth, height, width), by default None

        Returns:
        --------
        points : np.ndarray
            Nx3 array of point coordinates
        intensities : np.ndarray
            N array of intensity values
        metadata : dict
            Dictionary containing bin sizes and coordinate ranges
        '''
        return self.render(data[:, 0], data[:, 1], data[:, 2], data[:, 3], shape)


@numba.jit(nopython=True)
def normalize_intensities(intensities):
    '''
    Normalize intensity values to range [0,1] using Numba acceleration.
    '''
    min_val = np.min(intensities)
    max_val = np.max(intensities)
    return (intensities - min_val) / (max_val - min_val)
