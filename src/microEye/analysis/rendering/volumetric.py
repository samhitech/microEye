from enum import Enum

import numba
import numpy as np


@numba.jit(nopython=True)
def render_volume_compute(data, out_volume):
    '''
    Compute the volumetric histogram using numba for acceleration.
    '''
    for x, y, z, intensity in data:
        out_volume[int(z), int(y), int(x)] += intensity


class VolumeRenderer:
    '''
    Volume renderer for rendering 3D super resolution images
    '''

    def __init__(self, xy_voxel_size=10, z_voxel_size=50):
        '''
        Initialize the volume renderer.

        Parameters:
        -----------
        xy_voxel_size : float
            Size of each voxel in the lateral (XY) dimensions
        z_voxel_size : float
            Size of each voxel in the axial (Z) dimension
        '''
        self._xy_voxel_size = xy_voxel_size
        self._z_voxel_size = z_voxel_size
        self._volume = None

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
        """
        Generates a volumetric histogram from 3D single-molecule localizations.

        Parameters
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
            Dimensions of the volume as (depth, height, width). Defaults to None.

        Returns
        --------
        np.ndarray
            3D volume array suitable for visualization
        metadata : dict
            Dictionary containing voxel sizes and coordinate ranges
            - 'voxel_size' (dict): Voxel sizes in each dimension ('x', 'y', 'z').
            - 'coordinates' (dict): Z-coordinate range ('z_min', 'z_max').

        """
        self._validate_inputs(X, Y, Z, Intensity)
        X, Y = self._normalize_xy(X, Y)

        if shape is None:
            # Calculate XY dimensions based on data extent
            x_max = int((np.max(X) / self._xy_voxel_size) + 4)
            y_max = int((np.max(Y) / self._xy_voxel_size) + 4)

            # Calculate Z dimension to maintain zero reference
            z_min = int(np.floor(np.min(Z) / self._z_voxel_size)) - 2
            z_max = int(np.ceil(np.max(Z) / self._z_voxel_size)) + 2
            z_size = z_max - z_min

            shape = (z_size, y_max, x_max)

        self._volume = np.zeros(shape)

        # Convert XY coordinates to voxel space (positive-only indices)
        X_voxels = X / self._xy_voxel_size + 2
        Y_voxels = Y / self._xy_voxel_size + 2

        # Convert Z coordinates to voxel space maintaining zero reference
        z_min = int(np.floor(np.min(Z) / self._z_voxel_size)) - 2
        Z_voxels = Z / self._z_voxel_size - z_min

        render_volume_compute(
            np.c_[X_voxels, Y_voxels, Z_voxels, Intensity], self._volume
        )

        # Return metadata for proper scaling and positioning in visualization
        metadata = {
            'voxel_size': {
                'x': self._xy_voxel_size,
                'y': self._xy_voxel_size,
                'z': self._z_voxel_size,
            },
            'coordinates': {
                'z_min': z_min
                * self._z_voxel_size,  # actual z coordinate where volume starts
                'z_max': (z_min + shape[0])
                * self._z_voxel_size,  # actual z coordinate where volume ends
            },
        }

        return self._volume, metadata

    def from_array(self, data: np.ndarray, shape=None):
        '''
        Renders a volumetric histogram from an array of localizations.

        Parameters:
        -----------
        data : np.ndarray
            Array with columns (X, Y, Z, Intensity)
        shape : tuple[int, int, int], optional
            Volume dimensions (depth, height, width), by default None

        Returns:
        --------
        np.ndarray
            3D volume array suitable for visualization
        metadata : dict
            Dictionary containing voxel sizes and coordinate ranges
        '''
        return self.render(data[:, 0], data[:, 1], data[:, 2], data[:, 3], shape)

def normalize_volume(volume_data: np.ndarray):
    '''
    Normalize volume data to range [0,1] using Numba acceleration.
    '''
    # data_min = np.min(volume_data)
    # data_max = np.max(volume_data)
    q1, q3 = np.nanpercentile(volume_data[volume_data > 0], [1, 99])
    return (np.clip(volume_data, q1, q3) - q1) / (q3 - q1)


@numba.jit(nopython=True, parallel=True)
def create_rgba_volume(normalized_data, colors, opacity):
    '''
    Create RGBA volume with colormap using Numba acceleration.

    Parameters:
    -----------
    normalized_data : np.ndarray
        3D array of normalized data values (0-1)
    colors : np.ndarray
        256x3 array of RGB colors for the colormap
    opacity : float
        Global opacity multiplier (0-1)

    Returns:
    --------
    rgba_volume : np.ndarray
        4D array containing the RGBA volume
    '''
    shape = normalized_data.shape
    rgba_volume = np.empty(shape + (4,), dtype=np.uint8)

    # Convert normalized data to color indices
    color_indices = (normalized_data * 255).astype(np.uint8)

    # Fill RGB channels
    for i in numba.prange(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                idx = color_indices[i, j, k]
                rgba_volume[i, j, k, 0] = colors[idx, 0]  # Red
                rgba_volume[i, j, k, 1] = colors[idx, 1]  # Green
                rgba_volume[i, j, k, 2] = colors[idx, 2]  # Blue
                # Alpha channel with opacity scaling
                rgba_volume[i, j, k, 3] = min(
                    255, int(normalized_data[i, j, k] * 255 * opacity)
                )

    return rgba_volume

