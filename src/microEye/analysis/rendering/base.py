from enum import Enum

import numba
import numpy as np

from microEye.analysis.rendering.core import model


@numba.njit()
def render_compute(data, step, gauss_2d, out_img):
    for x, y, Intensity in data:
        out_img[y - step : y + step + 1, x - step : x + step + 1] += (
            Intensity * gauss_2d
        )


@numba.njit()
def render_compute_projection(data, step_lateral, step_axial, gauss_2d, out_img):
    for lateral, axial, Intensity in data:
        out_img[
            axial - step_axial : axial + step_axial + 1,
            lateral - step_lateral : lateral + step_lateral + 1,
        ] += Intensity * gauss_2d


@numba.njit()
def filter_points_in_range(coords, values, center, half_width):
    '''Filter points within a range along one dimension.'''
    mask = (coords >= center - half_width) & (coords <= center + half_width)
    return mask


class RenderModes(Enum):
    '''
    Enum class for the rendering modes.
    '''

    HISTOGRAM = 0
    '''Intensity Histogram'''
    EVENT_HISTOGRAM = 1
    '''Event Histogram'''
    GAUSSIAN = 2
    '''Gaussian Rendering'''


class Projection(Enum):
    '''
    Enum class for different projections.
    '''

    XY = 0
    '''XY Projection'''
    XZ = 1
    '''XZ Projection'''
    YZ = 2
    '''YZ Projection'''


class BaseRenderer:
    '''
    Base class for rendering super resolution images
    from single molecule localizations.
    '''

    def __init__(self, pixel_size=10, z_pixel_size=None, mode=RenderModes.HISTOGRAM):
        '''
        Initializes the renderer.

        Parameters:
        -----------
        pixel_size : float
            Pixel size of the rendered image in lateral dimensions (XY).
        z_pixel_size : float, optional
            Pixel size for axial dimension (Z). If None, uses pixel_size.
        mode : RenderModes
            Rendering mode.
        '''
        self._pixel_size = pixel_size
        self._z_pixel_size = z_pixel_size if z_pixel_size is not None else pixel_size
        self._mode = mode
        self._image = None
        self._origin = None

        if self._mode == RenderModes.GAUSSIAN:
            self.generate_gaussian_kernels()

    def generate_gaussian_kernels(self):
        '''
        Generates Gaussian kernels for rendering both lateral and axial projections.
        '''
        # Lateral kernel (XY)
        self._std = self._pixel_size  # nm
        self._gauss_std = self._std / self._pixel_size
        self._gauss_len = 1 + np.ceil(self._gauss_std * 6)
        if self._gauss_len % 2 == 0:
            self._gauss_len += 1
        self._gauss_shape = [int(self._gauss_len)] * 2

        # Axial kernel (Z)
        self._z_std = self._z_pixel_size  # nm
        self._z_gauss_std = self._z_std / self._z_pixel_size
        self._z_gauss_len = 1 + np.ceil(self._z_gauss_std * 6)
        if self._z_gauss_len % 2 == 0:
            self._z_gauss_len += 1

        # Generate kernels for different projections
        xy_len = np.arange(0, self._gauss_shape[0])
        X, Y = np.meshgrid(xy_len, xy_len)

        # Standard XY kernel
        self._gauss_2d = model(
            (self._gauss_len - 1) / 2,
            (self._gauss_len - 1) / 2,
            self._gauss_std,
            self._gauss_std,
            1,
            0,
            X,
            Y,
        )

        # XZ/YZ projection kernel (asymmetric)
        xz_len_lateral = np.arange(0, self._gauss_len)
        xz_len_axial = np.arange(0, self._z_gauss_len)
        X_proj, Z_proj = np.meshgrid(xz_len_lateral, xz_len_axial)

        self._gauss_2d_projection = model(
            (self._gauss_len - 1) / 2,
            (self._z_gauss_len - 1) / 2,
            self._gauss_std,
            self._z_gauss_std,
            1,
            0,
            X_proj,
            Z_proj,
        )

    def _validate_inputs(self, X, Y, Intensity):
        '''
        Validate the inputs for rendering.
        '''
        if any([X is None, Y is None, Intensity is None]):
            raise Exception('One or more of the inputs are None.')
        if not len(X) == len(Y) == len(Intensity):
            raise Exception('The supplied arguments are of different lengths.')

    def _zero_origin(self, X, Y):
        '''
        Zero the origin of the coordinates.
        '''
        x_min = np.min(X)
        y_min = np.min(Y)

        if x_min < 0:
            X = X - x_min
        if y_min < 0:
            Y = Y - y_min

        return X, Y

    def render(
        self,
        projection: Projection,
        X,
        Y,
        Z,
        Intensity,
        shape=None,
    ):
        '''
        Renders super resolution image from single
        molecule localizations.

        Parameters
        ----------
        projection : Projection
            Projection type
        X, Y, Z : np.ndarray
            Coordinates of localizations
        Intensity : np.ndarray
            Intensity values
        shape : tuple[int, int], optional
            Output image shape (height, width)

        Returns
        -------
        np.ndarray
            Rendered image
        '''
        if projection == Projection.XY:
            return self.render_xy(X, Y, Intensity, shape)
        elif projection == Projection.XZ:
            return self.render_xz(X, Z, Intensity, shape)
        elif projection == Projection.YZ:
            return self.render_yz(Y, Z, Intensity, shape)
        else:
            raise ValueError('Invalid projection type.')

    def render_xy(self, X, Y, Intensity, shape=None):
        '''
        Renders super resolution image (XY Projection) from single
        molecule localizations.


        Params
        -------
        X (np.ndarray)
            Sub-pixel localized points X coordinates
        Y (np.ndarray)
            Sub-pixel localized points Y coordinates
        Intensity (np.ndarray)
            Sub-pixel localized points intensity estimate
        shape (tuple[int, int], optional)
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        self._validate_inputs(X, Y, Intensity)
        X, Y = self._zero_origin(X, Y)

        if self._mode in [RenderModes.HISTOGRAM, RenderModes.EVENT_HISTOGRAM]:
            if shape is None:
                x_max = int((np.max(X) / self._pixel_size) + 4)
                y_max = int((np.max(Y) / self._pixel_size) + 4)
            else:
                x_max, y_max = shape[1], shape[0]
            n_max = max(x_max, y_max)

            self._image = np.zeros([n_max, n_max])
            X = np.round(X / self._pixel_size) + 2
            Y = np.round(Y / self._pixel_size) + 2

            if self._mode == RenderModes.EVENT_HISTOGRAM:
                Intensity = np.ones_like(Intensity)
            render_compute(np.c_[X, Y, Intensity], 0, 1, self._image)
        elif self._mode == RenderModes.GAUSSIAN:
            if shape is None:
                x_max = int((np.max(X) / self._pixel_size) + 4 * self._gauss_len)
                y_max = int((np.max(Y) / self._pixel_size) + 4 * self._gauss_len)
            else:
                x_max, y_max = shape[1], shape[0]
            n_max = max(x_max, y_max)

            step = int((self._gauss_len - 1) // 2)
            self._image = np.zeros([n_max, n_max])
            X = np.round(X / self._pixel_size) + 4 * step
            Y = np.round(Y / self._pixel_size) + 4 * step

            render_compute(np.c_[X, Y, Intensity], step, self._gauss_2d, self._image)

        return self._image

    def render_xz(self, X, Z, Intensity, shape=None):
        '''
        Renders XZ projection of super resolution image.

        Parameters
        ----------
        X : np.ndarray
            X coordinates
        Z : np.ndarray
            Z coordinates
        Intensity : np.ndarray
            Intensity values
        shape : tuple[int, int], optional
            Output image shape (z_height, x_width)

        Returns
        -------
        np.ndarray
            Rendered XZ projection
        '''
        self._validate_inputs(X, Z, Intensity)
        X, Z = self._zero_origin(X, Z)

        if self._mode in [RenderModes.HISTOGRAM, RenderModes.EVENT_HISTOGRAM]:
            if shape is None:
                x_max = int((np.max(X) / self._pixel_size) + 4)
                z_max = int((np.max(Z) / self._z_pixel_size) + 4)
            else:
                x_max, z_max = shape[1], shape[0]

            self._image = np.zeros([z_max, x_max])
            X = np.round(X / self._pixel_size) + 2
            Z = np.round(Z / self._z_pixel_size) + 2

            if self._mode == RenderModes.EVENT_HISTOGRAM:
                Intensity = np.ones_like(Intensity)
            render_compute(np.c_[X, Z, Intensity], 0, 1, self._image)

        elif self._mode == RenderModes.GAUSSIAN:
            if shape is None:
                x_max = int((np.max(X) / self._pixel_size) + 4 * self._gauss_len)
                z_max = int((np.max(Z) / self._z_pixel_size) + 4 * self._z_gauss_len)
            else:
                x_max, z_max = shape[1], shape[0]

            step_lateral = int((self._gauss_len - 1) // 2)
            step_axial = int((self._z_gauss_len - 1) // 2)

            self._image = np.zeros([z_max, x_max])
            X = np.round(X / self._pixel_size) + 4 * step_lateral
            Z = np.round(Z / self._z_pixel_size) + 4 * step_axial

            render_compute_projection(
                np.c_[X, Z, Intensity],
                step_lateral,
                step_axial,
                self._gauss_2d_projection,
                self._image,
            )

        return self._image

    def render_yz(self, Y, Z, Intensity, shape=None):
        '''
        Renders YZ projection of super resolution image.

        Parameters
        ----------
        Y : np.ndarray
            Y coordinates
        Z : np.ndarray
            Z coordinates
        Intensity : np.ndarray
            Intensity values
        shape : tuple[int, int], optional
            Output image shape (z_height, y_width)

        Returns
        -------
        np.ndarray
            Rendered YZ projection
        '''
        return self.render_xz(Y, Z, Intensity, shape)

    def render_slice(
        self,
        projection: Projection,
        X,
        Y,
        Z,
        Intensity,
        position,
        width=None,
        shape=None,
    ):
        '''
        Renders selected projection image at specific othogonal position
        with given bin width.

        Parameters
        ----------
        projection : Projection
            Projection type
        X, Y, Z : np.ndarray
            Coordinates of localizations
        Intensity : np.ndarray
            Intensity values
        position : float
            Position of the slice in the orthogonal axis
        width : float, optional
            Width of the slice, by default None
        shape : tuple[int, int], optional
            Output image shape (height, width)

        Returns
        -------
        np.ndarray
            Rendered XY slice at specified Z position
        '''
        # Validate inputs
        if not len(X) == len(Y) == len(Z) == len(Intensity):
            raise Exception('The supplied arguments are of different lengths.')

        # Set slice width
        if width is None:
            if projection == Projection.XY:
                width = self._z_pixel_size
            else:
                width = self._pixel_size

        # Filter points within Z range
        half_width = width / 2
        if projection == Projection.XY:
            mask = filter_points_in_range(Z, Intensity, position, half_width)

            if not np.any(mask):
                return None

            # Only render points within the Z range
            X_filtered = X[mask]
            Y_filtered = Y[mask]
            I_filtered = Intensity[mask]

            # Render the filtered points using existing render method
            return self.render_xy(X_filtered, Y_filtered, I_filtered, shape)
        elif projection == Projection.XZ:
            mask = filter_points_in_range(Y, Intensity, position, half_width)

            if not np.any(mask):
                return None

            # Filter points within Y range
            half_width = width / 2
            y_mask = filter_points_in_range(Y, Intensity, position, half_width)

            # Only render points within the Y range
            X_filtered = X[y_mask]
            Z_filtered = Z[y_mask]
            I_filtered = Intensity[y_mask]

            return self.render_xz(X_filtered, Z_filtered, I_filtered, shape)
        elif projection == Projection.YZ:
            mask = filter_points_in_range(X, Intensity, position, half_width)

            if not np.any(mask):
                return None

            # Filter points within X range
            half_width = width / 2
            x_mask = filter_points_in_range(X, Intensity, position, half_width)

            # Only render points within the X range
            Y_filtered = Y[x_mask]
            Z_filtered = Z[x_mask]
            I_filtered = Intensity[x_mask]

            return self.render_yz(Y_filtered, Z_filtered, I_filtered, shape)
        else:
            raise ValueError('Invalid projection type.')

    def from_array(self, data: np.ndarray, shape=None):
        '''Renders as super resolution image from
        single molecule localizations.

        Params
        -------
        data (np.ndarray)
            Array with sub-pixel localization data columns (X, Y, Intensity)
        shape (tuple[int, int], optional)
            Super-res image (height, width), by default None

        Returns
        -------
        Image (np.ndarray)
            the rendered 2D super-res image array
        '''
        return self.render_xy(data[:, 0], data[:, 1], data[:, 2], shape)
