import logging
from enum import Enum

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from microEye.analysis.clustering.dbscan import (
    perform_dbscan_clustering,
    perform_hdbscan_clustering,
    perform_optics_clustering,
)
from microEye.analysis.fitting.aim import _shift_correction, aim_drift_correction
from microEye.analysis.fitting.processing import (
    merge_localizations,
    nn_trajectories,
    shift_correction,
    shift_estimation,
)
from microEye.analysis.fitting.rcc import (
    dcc_shift_estimation,
    mcc_shift_estimation,
    rcc_shift_estimation,
    rcc_solve,
)
from microEye.analysis.rendering import BaseRenderer, Projection, RenderModes

logger = logging.getLogger(__name__)


class DataColumns(Enum):
    FRAME = 1
    CHANNEL = 2
    X = 3
    Y = 4
    Z = 5
    BACKGROUND = 6
    INTENSITY = 7
    SIGMA_X = 8
    SIGMA_Y = 9
    XY_RATIO = 10
    LOGLIKE = 11
    ITERATION = 12
    TRACK_ID = 13
    NEIGHBOUR_DISTANCE = 14
    N_MERGED = 15
    CRLB_X = 16
    CRLB_Y = 17
    CRLB_Z = 18
    CRLB_INTENSITY = 19
    CRLB_BACKGROUND = 20
    CRLB_SIGMA_X = 21
    CRLB_SIGMA_Y = 22
    CRLB_SIGMA_Z = 23
    CLUSTER_ID = 24

    @classmethod
    def unique_columns(cls) -> list[str]:
        return [column.name.replace('_', ' ').lower() for column in cls]

    def __str__(self):
        return self.name.replace('_', ' ').lower()

    @classmethod
    def from_string(cls, s):
        for column in cls:
            if column.name == s:
                return column
        raise ValueError(f'{cls.__name__} has no value matching "{s}"')

    @classmethod
    def values(cls):
        return np.array([column.name for column in cls])


class ResultsUnits:
    Pixel = 0
    Nanometer = 1


class FittingMethod(Enum):
    External = -1
    Phasor_2D_CPU = 0
    MLE_2D_fixed_sigma = 1
    MLE_2D_free_sigma = 2
    not_used = 3
    MLE_2D_elliptical_sigma = 4
    MLE_3D_cspline_sigma = 5

    def __str__(self):
        '''
        Return the string representation of method.
        '''
        return {
            FittingMethod.External: 'External',
            FittingMethod.Phasor_2D_CPU: '2D Phasor-Fit',
            FittingMethod.MLE_2D_fixed_sigma: '2D MLE fixed sigma',
            FittingMethod.MLE_2D_free_sigma: '2D MLE free sigma',
            FittingMethod.not_used: 'Not Available',
            FittingMethod.MLE_2D_elliptical_sigma: '2D MLE elliptical sigma',
            FittingMethod.MLE_3D_cspline_sigma: '3D MLE cSpline',
        }.get(self, 'External')

    @staticmethod
    def get_strings():
        return [
            '2D Phasor-Fit',
            '2D MLE free sigma',
            '2D MLE elliptical sigma',
            '3D MLE cSpline',
        ]

    @staticmethod
    def from_string(s: str):
        '''
        Return the fitting method from string representation.
        '''
        for method in FittingMethod:
            if str(method) == s:
                return method
        raise ValueError(f'Fitting method "{s}" not found.')

    def __lt__(self, other):
        if isinstance(other, (FittingMethod, int)):
            return self.value < (
                other.value if isinstance(other, FittingMethod) else other
            )
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, (FittingMethod, int)):
            return self.value <= (
                other.value if isinstance(other, FittingMethod) else other
            )
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, (FittingMethod, int)):
            return self.value > (
                other.value if isinstance(other, FittingMethod) else other
            )
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, (FittingMethod, int)):
            return self.value >= (
                other.value if isinstance(other, FittingMethod) else other
            )
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, (FittingMethod, int)):
            return self.value == (
                other.value if isinstance(other, FittingMethod) else other
            )
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, (FittingMethod, int)):
            return self.value != (
                other.value if isinstance(other, FittingMethod) else other
            )
        return NotImplemented

    def __hash__(self):
        return super().__hash__()


PARAMETER_HEADERS = {
    FittingMethod.Phasor_2D_CPU: [
        str(DataColumns.X),
        str(DataColumns.Y),
        str(DataColumns.INTENSITY),
        str(DataColumns.BACKGROUND),
        str(DataColumns.XY_RATIO),
    ],
    FittingMethod.MLE_2D_fixed_sigma: [
        str(DataColumns.X),
        str(DataColumns.Y),
        str(DataColumns.INTENSITY),
        str(DataColumns.BACKGROUND),
        str(DataColumns.ITERATION),
    ],
    FittingMethod.MLE_2D_free_sigma: [
        str(DataColumns.X),
        str(DataColumns.Y),
        str(DataColumns.INTENSITY),
        str(DataColumns.BACKGROUND),
        str(DataColumns.SIGMA_X),
        str(DataColumns.ITERATION),
    ],
    FittingMethod.MLE_2D_elliptical_sigma: [
        str(DataColumns.X),
        str(DataColumns.Y),
        str(DataColumns.INTENSITY),
        str(DataColumns.BACKGROUND),
        str(DataColumns.SIGMA_X),
        str(DataColumns.SIGMA_Y),
        str(DataColumns.ITERATION),
    ],
    FittingMethod.MLE_3D_cspline_sigma: [
        str(DataColumns.X),
        str(DataColumns.Y),
        str(DataColumns.INTENSITY),
        str(DataColumns.BACKGROUND),
        str(DataColumns.Z),
        str(DataColumns.ITERATION),
    ],
}

ALIASES = {
    'frame': DataColumns.FRAME,
    'frames': DataColumns.FRAME,
    't': DataColumns.FRAME,
    'time': DataColumns.FRAME,
    'channel': DataColumns.CHANNEL,
    'c': DataColumns.CHANNEL,
    'x': DataColumns.X,
    'x [nm]': DataColumns.X,
    'locx': DataColumns.X,
    'y': DataColumns.Y,
    'y [nm]': DataColumns.Y,
    'locy': DataColumns.Y,
    'z': DataColumns.Z,
    'z [nm]': DataColumns.Z,
    'locz': DataColumns.Z,
    'bg': DataColumns.BACKGROUND,
    'background': DataColumns.BACKGROUND,
    'i': DataColumns.INTENSITY,
    'intensity': DataColumns.INTENSITY,
    'sigma': DataColumns.SIGMA_X,
    'sigmax': DataColumns.SIGMA_X,
    'sigma x': DataColumns.SIGMA_X,
    'x_sigma': DataColumns.SIGMA_X,
    'sigmay': DataColumns.SIGMA_Y,
    'sigma y': DataColumns.SIGMA_Y,
    'y_sigma': DataColumns.SIGMA_Y,
    'ratio': DataColumns.XY_RATIO,
    'xy_ratio': DataColumns.XY_RATIO,
    'xy ratio': DataColumns.XY_RATIO,
    'ratio x/y': DataColumns.XY_RATIO,
    'loglike': DataColumns.LOGLIKE,
    'iteration': DataColumns.ITERATION,
    'trackid': DataColumns.TRACK_ID,
    'track id': DataColumns.TRACK_ID,
    'neighbour_distance': DataColumns.NEIGHBOUR_DISTANCE,
    'neighbour_dist': DataColumns.NEIGHBOUR_DISTANCE,
    'neighbour distance': DataColumns.NEIGHBOUR_DISTANCE,
    'n_merged': DataColumns.N_MERGED,
    'crlb_x': DataColumns.CRLB_X,
    'crlb x': DataColumns.CRLB_X,
    'crlb_y': DataColumns.CRLB_Y,
    'crlb y': DataColumns.CRLB_Y,
    'crlb_z': DataColumns.CRLB_Z,
    'crlb z': DataColumns.CRLB_Z,
    'crlb_i': DataColumns.CRLB_INTENSITY,
    'crlb i': DataColumns.CRLB_INTENSITY,
    'crlb intensity': DataColumns.CRLB_INTENSITY,
    'crlb_bg': DataColumns.CRLB_BACKGROUND,
    'crlb bg': DataColumns.CRLB_BACKGROUND,
    'crlb background': DataColumns.CRLB_BACKGROUND,
    'crlb_sigx': DataColumns.CRLB_SIGMA_X,
    'crlb sigma': DataColumns.CRLB_SIGMA_X,
    'crlb sigmax': DataColumns.CRLB_SIGMA_X,
    'crlb sigma x': DataColumns.CRLB_SIGMA_X,
    'crlb_sigy': DataColumns.CRLB_SIGMA_Y,
    'crlb sigmay': DataColumns.CRLB_SIGMA_Y,
    'crlb sigma y': DataColumns.CRLB_SIGMA_Y,
    'crlb_sigz': DataColumns.CRLB_SIGMA_Z,
    'crlb sigmaz': DataColumns.CRLB_SIGMA_Z,
    'crlb sigma z': DataColumns.CRLB_SIGMA_Z,
    'clusterid': DataColumns.CLUSTER_ID,
    'cluster id': DataColumns.CLUSTER_ID,
}


def map_column_alias(alias: str) -> DataColumns:
    alias_lower = alias.lower()
    if alias_lower in ALIASES:
        return ALIASES[alias_lower]
    else:
        return None


class FittingResults:
    def __init__(
        self,
        unit=ResultsUnits.Pixel,
        pixelSize=130.0,
        fittingMethod=FittingMethod.Phasor_2D_CPU,
        path=None,
    ):
        '''Fitting Results

        Parameters
        ----------
        unit : int, optional
            unit of localized points, by default ResultsUnits.Pixel
        pixelSize : float, optional
            pixel size in nanometers, by default 130.0
        fittingMethod : FittingMethod, optional
            fitting method used, by default FittingMethod._2D_Phasor_CPU
        path : str, optional
            path to file, by default None
        '''
        self.unit = unit
        self.pixel_size = pixelSize
        self.fitting_method = fittingMethod
        self.path = path
        self.colormap = None
        self.intensity_range = None
        if fittingMethod > -1:
            self.parameterHeader = PARAMETER_HEADERS[fittingMethod]
        else:
            self.parameterHeader = []

        # Initialize the dictionary with None for each key
        self.data = {key: None for key in DataColumns}

    def __len__(self):
        '''Number of localized positions'''
        return len(self.data[self.uniqueKeys()[0]])

    def __getattr__(self, name):
        key = map_column_alias(name)
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'."
            )

    def __setattr__(self, name, value):
        key = map_column_alias(name)
        if key is None:
            super().__setattr__(name, value)
        else:
            self.data[key] = value

    def uniqueKeys(self) -> list[DataColumns]:
        '''Returns a list of unique keys in FittingResults'''
        keys = []
        for key in self.data:
            if self.data[key] is not None:
                keys.append(key)

        return keys

    def extend_column(self, key, value):
        if key not in self.data or self.data[key] is None:
            self.data[key] = value
        else:
            self.data[key] = np.concatenate((self.data[key], value))

    def _scale_array(self, array: np.ndarray):
        if array is not None:
            array[:, :2] *= self.pixel_size
            if (
                self.fitting_method == FittingMethod.MLE_2D_free_sigma
                or self.fitting_method == FittingMethod.MLE_3D_cspline_sigma
            ):
                array[:, 4] *= self.pixel_size
            elif self.fitting_method == FittingMethod.MLE_2D_elliptical_sigma:
                array[:, 4:6] *= self.pixel_size

    def get_column(self, key: DataColumns):
        if not isinstance(key, DataColumns):
            raise TypeError(f'Expected DataColumns type, got {type(key)}')
        if self.data[key] is None:
            return np.zeros(self.__len__())
        else:
            return self.data[key]

    def extend(self, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        '''Extend results by contents of data array

        Parameters
        ----------
        data : tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            tuple of arrays (frames, params, crlbs, loglike),
            single-column (frames, loglike),
            multi-column headers for params see ParametersHeaders
            for each fitting method,
            multi-column headers for crlbs same as params without iterations.

            For FittingMethod._2D_Phasor_CPU use data:
            (None, params, None, None).
        '''
        frames, params, crlbs, loglike = data

        if self.unit == ResultsUnits.Pixel:
            self._scale_array(params)
            self._scale_array(crlbs)

        for idx, var in enumerate(
            [
                DataColumns.X,
                DataColumns.Y,
                DataColumns.INTENSITY,
                DataColumns.BACKGROUND,
            ]
        ):
            self.extend_column(var, params[:, idx])

        self.extend_column(DataColumns.FRAME, frames)

        if self.fitting_method == FittingMethod.Phasor_2D_CPU:
            self.extend_column(DataColumns.XY_RATIO, params[:, -1])
        else:
            self.extend_column(DataColumns.ITERATION, params[:, -1])

        # Extend last column based on FittingMethod
        if self.fitting_method != FittingMethod.Phasor_2D_CPU:
            self.extend_column(DataColumns.LOGLIKE, loglike)

            for idx, var in enumerate(
                [
                    DataColumns.CRLB_X,
                    DataColumns.CRLB_Y,
                    DataColumns.CRLB_INTENSITY,
                    DataColumns.CRLB_BACKGROUND,
                ]
            ):
                self.extend_column(var, crlbs[:, idx])

            if self.fitting_method == FittingMethod.MLE_2D_free_sigma:
                self.extend_column(DataColumns.SIGMA_X, params[:, 4])
                self.extend_column(DataColumns.CRLB_SIGMA_X, crlbs[:, 4])
            elif self.fitting_method == FittingMethod.MLE_2D_elliptical_sigma:
                self.extend_column(DataColumns.SIGMA_X, params[:, 4])
                self.extend_column(DataColumns.SIGMA_Y, params[:, 5])
                self.extend_column(DataColumns.CRLB_SIGMA_X, crlbs[:, 4])
                self.extend_column(DataColumns.CRLB_SIGMA_Y, crlbs[:, 5])
            elif self.fitting_method == FittingMethod.MLE_3D_cspline_sigma:
                self.extend_column(DataColumns.Z, params[:, 4])
                self.extend_column(DataColumns.CRLB_Z, crlbs[:, 4])

    def dataFrame(self):
        '''Return fitting results as Pandas DataFrame

        Returns
        -------
        DataFrame
            fitting results DataFrame with columns FittingResults.columns
        '''
        keys = self.uniqueKeys()

        if self.fitting_method == FittingMethod.Phasor_2D_CPU:
            loc = np.c_[
                self.data[DataColumns.FRAME],
                self.data[DataColumns.X],
                self.data[DataColumns.Y],
                self.data[DataColumns.INTENSITY],
                self.data[DataColumns.BACKGROUND],
                self.data[DataColumns.XY_RATIO],
                self.get_column(DataColumns.TRACK_ID),
                self.get_column(DataColumns.NEIGHBOUR_DISTANCE),
                self.get_column(DataColumns.N_MERGED),
                self.get_column(DataColumns.CLUSTER_ID),
            ]

            columns = (
                [
                    'frame',
                ]
                + self.parameterHeader
                + [
                    str(DataColumns.TRACK_ID),
                    str(DataColumns.NEIGHBOUR_DISTANCE),
                    str(DataColumns.N_MERGED),
                    str(DataColumns.CLUSTER_ID),
                ]
            )
        elif self.fitting_method == -1:
            loc = None
            columns = []
            for key in keys:
                col = self.data[key]
                columns.append(str(key))
                loc = col if loc is None else np.c_[loc, col]
        else:
            loc = np.c_[
                self.data[DataColumns.FRAME],
                self.data[DataColumns.X],
                self.data[DataColumns.Y],
                self.data[DataColumns.INTENSITY],
                self.data[DataColumns.BACKGROUND],
            ]

            if self.fitting_method == FittingMethod.MLE_2D_fixed_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_INTENSITY],
                    self.data[DataColumns.CRLB_BACKGROUND],
                ]
            elif self.fitting_method == FittingMethod.MLE_2D_free_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.SIGMA_X],
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_INTENSITY],
                    self.data[DataColumns.CRLB_BACKGROUND],
                    self.data[DataColumns.CRLB_SIGMA_X],
                ]
            elif self.fitting_method == FittingMethod.MLE_2D_elliptical_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.SIGMA_X],
                    self.data[DataColumns.SIGMA_Y],
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_INTENSITY],
                    self.data[DataColumns.CRLB_BACKGROUND],
                    self.data[DataColumns.CRLB_SIGMA_X],
                    self.data[DataColumns.CRLB_SIGMA_Y],
                ]
            elif self.fitting_method == FittingMethod.MLE_3D_cspline_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.Z],
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_INTENSITY],
                    self.data[DataColumns.CRLB_BACKGROUND],
                    self.data[DataColumns.CRLB_Z],
                ]

            loc = np.c_[
                loc,
                self.data[DataColumns.LOGLIKE],
                self.get_column(DataColumns.TRACK_ID),
                self.get_column(DataColumns.NEIGHBOUR_DISTANCE),
                self.get_column(DataColumns.N_MERGED),
                self.get_column(DataColumns.CLUSTER_ID),
            ]

            columns = (
                [
                    'frame',
                ]
                + self.parameterHeader
                + [f'crlb {x}' for x in self.parameterHeader[:-1]]
                + [
                    'loglike',
                    str(DataColumns.TRACK_ID),
                    str(DataColumns.NEIGHBOUR_DISTANCE),
                    str(DataColumns.N_MERGED),
                    str(DataColumns.CLUSTER_ID),
                ]
            )

        return pd.DataFrame(loc, columns=columns).sort_values(by=str(DataColumns.FRAME))

    def toRender(self):
        '''Returns columns for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains X [nm], Y [nm], Intensity columns
        '''
        # TODO: update for 3D render later
        return (
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            self.data[DataColumns.INTENSITY],
        )

    def toAnimation(self, n_bins=10, pixelSize=10, **kwargs):
        '''Returns columns for animation
        Parameters
        ----------
        n_bins : int, optional
            Number of bins, by default 10
        pixelSize : int, optional
            Pixel size, by default 10
        **kwargs : dict
            Additional keyword arguments for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains animation stack, frame bins, locs per bin
        '''
        # Get kwargs for rendering
        renderMode: RenderModes = kwargs.get('renderMode', RenderModes.HISTOGRAM)
        z_pixel_size = kwargs.get('z_pixel_size')
        projection: Projection = kwargs.get('projection', Projection.XY)

        # Get unique frames
        unique_frames = np.unique(self.data[DataColumns.FRAME])

        # Check edge cases
        if len(unique_frames) < 2:
            logger.error('Animation failed: 2 or more frames are required.')
            return None, None, None

        if n_bins < 2:
            logger.error('Animation failed: 2 or more bins are required.')
            return None, None, None

        frames_per_bin = int(np.floor(np.max(unique_frames) / n_bins))

        if frames_per_bin < 2:
            logger.error('Animation failed: 2 or more frames per bin are required.')
            return None, None, None

        renderEngine = BaseRenderer(pixelSize, z_pixel_size, renderMode)

        x_max = int((np.max(self.data[DataColumns.X]) / renderEngine._pixel_size) + 4)
        y_max = int((np.max(self.data[DataColumns.Y]) / renderEngine._pixel_size) + 4)

        frame_ids = np.cumsum(
            np.bincount(self.data[DataColumns.FRAME].astype(np.int64))
        )

        sub_images = []
        locs_per_bin = []
        frames = []

        for f in range(0, n_bins):
            mask = slice(
                frame_ids[f * frames_per_bin - 1] if f * frames_per_bin > 0 else 0,
                frame_ids[(f + 1) * frames_per_bin],
            )
            image = renderEngine.render(
                projection,
                self.data[DataColumns.X][mask],
                self.data[DataColumns.Y][mask],
                self.data[DataColumns.Z][mask] if self.data[DataColumns.Z] else None,
                self.data[DataColumns.INTENSITY][mask],
                (y_max, x_max),
            )
            locs_per_bin.append(mask.stop - mask.start)
            frames.append(f * frames_per_bin + frames_per_bin / 2)
            sub_images.append(image)
            logger.info(f'Bins: {f + 1:d}/{n_bins:d}')

        return (
            np.array(sub_images),
            np.array(frames),
            np.array(locs_per_bin),
        )

    def extended_stats(self):
        statistics = {}

        unique_frames, counts = np.unique(
            self.data[DataColumns.FRAME], return_counts=True
        )

        # percintile of localizations per frame
        p1, p2 = np.percentile(counts, [1, 99])
        p1 = int(np.floor(p1))
        p2 = int(np.ceil(p2))

        # use counts to make a histogram of average localizations per frame
        hist, bin_edges = np.histogram(
            counts, bins=np.arange(p1, p2 + 1, 1), density=True
        )

        statistics['Localizations per frame'] = {
            'counts': counts,
            'hist': hist,
            'bin_edges': bin_edges,
            'mean': np.mean(counts),
            'std': np.std(counts),
            'min': np.min(counts),
            'max': np.max(counts),
            'median': np.median(counts),
            'percentiles': {
                '1st': p1,
                '99th': p2,
            },
            'xlabel': 'Localizations per frame',
            'ylabel': 'Probability density',
        }

        return statistics

    def drift_cross_correlation(
        self, n_bins=10, pixelSize=10, upsampling=100, rmax=20, method='phase'
    ):
        """Corrects the XY drift using cross-correlation measurments

        Parameters
        ----------
        n_bins : int, optional
            Number of frame bins, by default 10
        pixelSize : int, optional
            Super-res image pixel size in nanometers, by default 10
        upsampling : int, optional
            phase_cross_correlation upsampling (check skimage.registration),
            by default 100
        rmax : int, optional
            Maximum error for redundant cross-correlation, by default 20, should be
            0.2 of projected pixel size.
        method : str, optional
            Drift estimation method, by default 'phase'
            Options: 'rcc', 'mcc', 'dcc', 'phase'

            - 'rcc' - redundant cross-correlation [1]
            - 'mcc' - mean cross-correlation [1]
            - 'dcc' - drift cross-correlation [1]
            - 'phase' - phase cross-correlation

        Returns
        -------
        tuple(FittingResults, tuple(np.ndarray, np.ndarray, np.ndarray, None))
            FittingResults with corrected drift and tuple of (frames, drift_x,
            drift_y, None) where drift_x and drift_y are the estimated drifts
            in nanometers for each frame.


        References
        ----------
        [1] Yina Wang, Joerg Schnitzbauer, Zhe Hu, Xueming Li, Yifan Cheng,
            Zhen-Li Huang, and Bo Huang, "Localization events-based sample
            drift correction for localization microscopy with redundant
            cross-correlation algorithm," Opt. Express 22, 15982-15991 (2014)
        """
        unique_frames = np.unique(self.data[DataColumns.FRAME])
        if len(unique_frames) < 2:
            logger.error('Drift cross-correlation failed: no frame info.')
            return

        frames_per_bin = int(np.floor(np.max(unique_frames) / n_bins))

        if frames_per_bin < 2:
            logger.error('Drift cross-correlation failed: 2 or more bins are required.')
            return

        renderEngine = BaseRenderer(pixelSize, RenderModes.HISTOGRAM)

        x_max = int((np.max(self.data[DataColumns.X]) / renderEngine._pixel_size) + 4)
        y_max = int((np.max(self.data[DataColumns.Y]) / renderEngine._pixel_size) + 4)

        frame_ids = np.cumsum(
            np.bincount(self.data[DataColumns.FRAME].astype(np.int64))
        )
        max_frame = np.max(self.data[DataColumns.FRAME])

        sub_images = []
        frames = []

        for f in range(0, n_bins):
            mask = slice(
                frame_ids[f * frames_per_bin - 1] if f * frames_per_bin > 0 else 0,
                frame_ids[(f + 1) * frames_per_bin],
            )
            image = renderEngine.from_array(
                np.c_[
                    self.data[DataColumns.X][mask],
                    self.data[DataColumns.Y][mask],
                    self.data[DataColumns.INTENSITY][mask],
                ],
                (y_max, x_max),
            )
            frames.append(f * frames_per_bin + frames_per_bin / 2)
            sub_images.append(image)
            logger.info(f'Bins: {f + 1:d}/{n_bins:d}')

        sub_images = np.array(sub_images)

        logger.info('Shift Estimation ...')
        try:
            if method == 'rcc':
                imshift, A = rcc_shift_estimation(sub_images, pixelSize)

                logger.info('Shift Optimization ...')
                drift = rcc_solve(imshift, A, n_bins, rmax=rmax)
            elif method == 'mcc':
                drift = mcc_shift_estimation(sub_images, pixelSize)
            elif method == 'dcc':
                drift = dcc_shift_estimation(sub_images, pixelSize)
            elif method == 'phase':
                drift = shift_estimation(np.array(sub_images), pixelSize, upsampling)
            else:
                raise ValueError(f'Unknown drift estimation method: {method}')
        except MemoryError as e:
            logger.error(f'Drift cross-correlation failed: {e}')
            return

        logger.info('Shift Correction ...')

        frames_interp = np.concatenate(([0], np.array(frames), [max_frame - 1]))

        # Prepare drift arrays with padding zeros at start and last value at end
        base = [
            0,
        ] * (1 if method in ['mcc', 'phase'] else 2)

        drift_x = np.concatenate((base, drift[:, 0], [drift[-1, 0]]))
        drift_y = np.concatenate((base, drift[:, 1], [drift[-1, 1]]))

        # An one-dimensional interpolation is applied
        # to drift traces in X and Y dimensions separately.

        # 2026-07-27: Use Piecewise Cubic Hermite Interpolating Polynomial
        #             instead of CubicSpline to avoid overshooting and
        #             negative values in drift correction.

        interpy = PchipInterpolator(frames_interp, drift_x, axis=0, extrapolate=True)
        interpx = PchipInterpolator(frames_interp, drift_y, axis=0, extrapolate=True)
        # And this interpolation is used to get the shift at every frame-point
        frames_new = np.arange(0, max_frame, 1)
        frame_drift = np.c_[
            interpx(frames_new),
            interpy(frames_new)
        ]

        _shift_correction(
            frame_ids,
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            None,
            frame_drift
        )

        return self, (frames_new, frame_drift[:, 0], frame_drift[:, 1], None)

    def drift_fiducial_marker(self):
        df = self.dataFrame()
        columns = df.columns
        data = df.to_numpy()

        unique_frames, frame_counts = np.unique(
            self.data[DataColumns.FRAME], return_counts=True
        )
        if len(unique_frames) < 2:
            logger.error('Drift correction failed: no frame info.')
            return

        if self.data[DataColumns.TRACK_ID] is None:
            logger.error('Drift correction failed: please perform NN, no trackIDs.')
            return
        if np.max(self.data[DataColumns.TRACK_ID]) < 1:
            logger.error('Drift correction failed: please perform NN, no trackIDs.')
            return

        unique_tracks, track_counts = np.unique(
            self.data[DataColumns.TRACK_ID], return_counts=True
        )
        fiducial_tracks_mask = np.logical_and(
            unique_tracks > 0, track_counts == len(unique_frames)
        )

        fiducial_trackIDs = unique_tracks[fiducial_tracks_mask]

        if len(fiducial_trackIDs) < 1:
            logger.error(
                'Drift correction failed: no ' + 'fiducial markers tracks detected.'
            )
            return
        else:
            logger.info(f'{len(fiducial_trackIDs):d} tracks detected.')

        fiducial_markers = np.zeros((len(fiducial_trackIDs), len(unique_frames), 2))

        logger.info('Fiducial markers ...')
        for idx in np.arange(fiducial_markers.shape[0]):
            mask = self.data[DataColumns.TRACK_ID] == fiducial_trackIDs[idx]
            fiducial_markers[idx] = np.c_[
                self.data[DataColumns.X][mask], self.data[DataColumns.Y][mask]
            ]

            fiducial_markers[idx, :, 0] -= fiducial_markers[idx, 0, 0]
            fiducial_markers[idx, :, 1] -= fiducial_markers[idx, 0, 1]

        logger.info('Drift estimate ...')
        drift_x = np.mean(fiducial_markers[:, :, 0], axis=0)
        drift_y = np.mean(fiducial_markers[:, :, 1], axis=0)

        error_bar_x = np.std(fiducial_markers[:, :, 0], axis=0)
        error_bar_y = np.std(fiducial_markers[:, :, 1], axis=0)

        logger.info('Drift correction ...')
        ids = np.cumsum(np.bincount(self.data[DataColumns.FRAME].astype(np.int64)))

        for idx in np.arange(len(unique_frames)):
            frame_id = int(unique_frames[idx])
            mask = slice(ids[frame_id - 1] if frame_id > 0 else 0, ids[frame_id])
            self.data[DataColumns.X][mask] -= drift_x[idx]
            self.data[DataColumns.Y][mask] -= drift_y[idx]

        # df = pd.DataFrame(
        #     data,
        #     columns=columns)

        # drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        return self, (unique_frames, drift_x, drift_y, None, error_bar_x, error_bar_y)

    def drift_aim(
        self,
        segment_size: int = 100,
        intersect_d: float = 20.0,
        z_scale: float = 0.5,
        mode: str = 'sequential',  # "sequential" or "joint"
        max_search: float = 100.0,
        interp_kind: str = 'pchip',
    ):
        frames = self.data[DataColumns.FRAME]
        x = self.data[DataColumns.X]
        y = self.data[DataColumns.Y]
        z = self.data[DataColumns.Z] if self.data[DataColumns.Z] is not None else None

        logger.info('Drift correction using AIM method ...')

        res = aim_drift_correction(
            frames=frames,
            x=x,
            y=y,
            z=z,
            segment_size=segment_size,
            intersect_d=intersect_d,
            z_scale=z_scale,
            max_search=max_search,
            interp_kind=interp_kind,
        )

        if res is None or not isinstance(res, dict):
            logger.error(
                'Drift correction failed: AIM method returned None or invalid result.'
            )
            return

        frame_ids: np.ndarray = res['frame_ids']
        frame_full: np.ndarray = res['frame_full']
        frame_drift: np.ndarray = res['frame_drift']

        logger.info('Applying drift correction ...')

        # Apply drift correction
        _shift_correction(
            frame_ids,
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            self.data[DataColumns.Z] if self.data[DataColumns.Z] is not None else None,
            frame_drift,
        )

        logger.info('Drift correction using AIM method completed successfully.')

        return self, (
            frame_full,
            frame_drift[:, 0],
            frame_drift[:, 1],
            frame_drift[:, 2] if z is not None else None,
        )

    def nn_trajectories(
        self, minDistance: float = 0, maxDistance: float = 30, maxOff=1, neighbors=1
    ):
        '''
        Perform nearest neighbor trajectory analysis.

        Parameters
        ----------
        minDistance : float
            Minimum distance between neighbors in nanometers
        maxDistance : float
            Maximum distance between neighbors in nanometers
        maxOff : int
            Maximum length of trajectory in frames
        neighbors : int
            Number of nearest neighbors to consider

        Returns
        -------
        FittingResults
            Updated FittingResults object with trajectory information

        Raises
        ------
        ValueError
            If parameters are invalid
        '''
        self.trackID = np.zeros(self.__len__(), dtype=np.int64)
        self.neighbour_dist = np.zeros(self.__len__(), dtype=np.float64)
        # data = self.dataFrame().to_numpy()
        data = np.c_[
            self.data[DataColumns.FRAME],
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
        ]

        ids = np.cumsum(np.bincount(self.data[DataColumns.FRAME].astype(np.int64)))

        counter = 0
        min_frame = np.min(self.data[DataColumns.FRAME])
        max_frame = np.max(self.data[DataColumns.FRAME])

        for frameID in np.arange(min_frame, max_frame, dtype=np.int64):
            for offset in np.arange(0, maxOff + 1, dtype=np.int64):
                nextID = frameID + offset + 1

                if nextID > max_frame:
                    continue

                currentMask = slice(
                    ids[frameID - 1] if frameID > 0 else 0, ids[frameID]
                )
                nextMask = slice(ids[nextID - 1], ids[nextID])

                (
                    counter,
                    self.trackID[currentMask],
                    self.trackID[nextMask],
                    self.neighbour_dist[nextMask],
                ) = nn_trajectories(
                    data[currentMask, :],
                    data[nextMask, :],
                    c_trackID=self.trackID[currentMask],
                    n_trackID=self.trackID[nextMask],
                    nn_dist=self.neighbour_dist[nextMask],
                    counter=counter,
                    minDistance=minDistance,
                    maxDistance=maxDistance,
                    neighbors=neighbors,
                )
            print(f'NN {frameID / max_frame:.2%} ...               ', end='\r')

        logger.info('\nDone ...')

        return self

    def merge_tracks(self, maxLength=500):
        self.n_merged = np.zeros(self.__len__(), dtype=np.int64)
        df = self.dataFrame()
        columns = list(df.columns)

        column_ids = np.zeros(4, dtype=np.int32)
        column_ids[0] = columns.index(str(DataColumns.FRAME))
        column_ids[1] = columns.index(str(DataColumns.TRACK_ID))
        column_ids[2] = columns.index(str(DataColumns.N_MERGED))
        if 'I' in columns:
            column_ids[3] = columns.index('I')
        elif str(DataColumns.INTENSITY) in columns:
            column_ids[3] = columns.index(str(DataColumns.INTENSITY))

        data = df.to_numpy()

        logger.info('Merging ...\n')
        finalData = merge_localizations(data, column_ids, maxLength)

        df = pd.DataFrame(finalData, columns=columns).sort_values(
            by=str(DataColumns.FRAME)
        )

        logger.info('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixel_size)

    def nearest_neighbour_merging(
        self, minDistance=0, maxDistance=30, maxOff=1, maxLength=500, neighbors=1
    ):
        # Call nn_trajectories
        self.nn_trajectories(minDistance, maxDistance, maxOff, neighbors)

        # Call merge_tracks
        return self.merge_tracks(maxLength)

    def cluster_dbscan(self, **kwargs):
        logger.info('DBSCAN clustering ...')

        eps = kwargs.pop('eps', 50)
        min_samples = kwargs.pop('min_samples', 5)

        kwargs['n_jobs'] = -1

        labels = perform_dbscan_clustering(
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            self.data[DataColumns.Z],
            eps,
            min_samples,
            **kwargs,
        )

        self.data[DataColumns.CLUSTER_ID] = labels

    def cluster_hdbscan(self, **kwargs):
        logger.info('HDBSCAN clustering ...')

        labels = perform_hdbscan_clustering(
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            self.data[DataColumns.Z],
            **kwargs,
        )

        self.data[DataColumns.CLUSTER_ID] = labels

    def cluster_optics(self, **kwargs):
        logger.info('OPTICS clustering ...')

        min_samples = kwargs.pop('min_samples', 5)
        xi = kwargs.pop('xi', 0.05)

        kwargs['n_jobs'] = -1

        labels = perform_optics_clustering(
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            self.data[DataColumns.Z],
            min_samples,
            xi,
            **kwargs,
        )

        self.data[DataColumns.CLUSTER_ID] = labels

    def zero_coordinates(self):
        # Use vectorized operations with optional masking
        for coord in (DataColumns.X, DataColumns.Y, DataColumns.Z):
            if self.data[coord] is not None:
                self.data[coord] -= np.min(self.data[coord])

    @staticmethod
    def fromFile(filename: str, pixelSize: float):
        '''Populates fitting results from a tab seperated values
        (tsv) file.

        Parameters
        ----------
        filename : str
            path to tab seperated file (.tsv)
        pixelSize : float
            projected pixel size in nanometers

        Returns
        -------
        FittingResults
            FittingResults with imported data
        '''
        if '.tsv' in filename:
            dataFrame = pd.read_csv(filename, sep='\t', engine='python')
        elif '.h5' in filename:
            dataFrame = pd.read_hdf(filename, mode='r')
        else:
            raise Exception(
                'Supported file types are dataframes ' + 'stored as .tsv or .h5 files.'
            )

        return FittingResults.fromDataFrame(dataFrame, pixelSize, path=filename)

    @staticmethod
    def fromDataFrame(dataFrame: pd.DataFrame, pixelSize: float = 1, **kwargs):
        fittingResults = None
        fittingResults = FittingResults(
            ResultsUnits.Nanometer,
            pixelSize,
            FittingMethod.External,
            kwargs.get('path'),
        )

        for column in dataFrame.columns.values.tolist():
            key = map_column_alias(column)
            if key is not None:
                fittingResults.data[key] = dataFrame[column].to_numpy()

        return fittingResults
