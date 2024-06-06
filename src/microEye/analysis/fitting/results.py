
from enum import Enum

import numpy as np
import pandas as pd
from numpy.core._exceptions import _ArrayMemoryError
from scipy.interpolate import interp1d

from microEye.analysis.fitting.processing import *
from microEye.analysis.rendering import gauss_hist_render, hist2D_render

UNIQUE_COLUMNS = [
            'frame', 'channel', 'x', 'y', 'z', 'background', 'intensity',
            'sigmax', 'sigmay', 'ratio', 'loglike', 'iteration', 'trackID',
            'neighbour_distance', 'n_merged', 'CRLB x', 'CRLB y', 'CRLB z',
            'CRLB background', 'CRLB intensity', 'CRLB sigmax',
            'CRLB sigmay', 'CRLB sigmaz'
            ]

class DataColumns(Enum):
    FRAME = 1
    C = 2
    X = 3
    Y = 4
    Z = 5
    BACKGROUND = 6
    INTENSITY = 7
    X_SIGMA = 8
    Y_SIGMA = 9
    XY_RATIO = 10
    LOG_LIKE = 11
    ITERATION = 12
    TRACK_ID = 13
    NEIGHBOUR_DISTANCE = 14
    N_MERGED = 15
    CRLB_X = 16
    CRLB_Y = 17
    CRLB_Z = 18
    CRLB_I = 19
    CRLB_BG = 20
    CRLB_SIG_X = 21
    CRLB_SIG_Y = 22
    CRLB_SIG_Z = 23

    def __str__(self):
        return UNIQUE_COLUMNS[self.value-1]

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

class FittingMethod:
    _External = -1
    _2D_Phasor_CPU = 0
    _2D_Gauss_MLE_fixed_sigma = 1
    _2D_Gauss_MLE_free_sigma = 2
    _2D_Gauss_MLE_elliptical_sigma = 4
    _3D_Gauss_MLE_cspline_sigma = 5

PARAMETER_HEADERS = {
    0: ['x', 'y', 'background', 'intensity', 'ratio'],
    1: ['x', 'y', 'background', 'intensity', 'iteration'],
    2: ['x', 'y', 'background', 'intensity', 'sigma', 'iteration'],
    4: ['x', 'y', 'background', 'intensity', 'sigmax', 'sigmay', 'iteration'],
    5: ['x', 'y', 'background', 'intensity', 'z', 'iteration']
}

def map_column_alias(alias: str) -> DataColumns:
    alias_lower = alias.lower()
    if alias_lower in ('frame', 'frames', 't', 'time'):
        return DataColumns.FRAME
    elif alias_lower in ('channel', 'c'):
        return DataColumns.C
    elif alias_lower in ('x', 'x [nm]', 'locx'):
        return DataColumns.X
    elif alias_lower in ('y', 'y [nm]', 'locy'):
        return DataColumns.Y
    elif alias_lower in ('z', 'z [nm]', 'locz'):
        return DataColumns.Z
    elif alias_lower in ('bg', 'background'):
        return DataColumns.BACKGROUND
    elif alias_lower in ('i', 'intensity'):
        return DataColumns.INTENSITY
    elif alias_lower in ('sigma', 'sigmax', 'sigma x', 'x_sigma'):
        return DataColumns.X_SIGMA
    elif alias_lower in ('sigmay', 'sigma y', 'y_sigma'):
        return DataColumns.Y_SIGMA
    elif alias_lower in ('ratio', 'xy_ratio', 'ratio x/y'):
        return DataColumns.XY_RATIO
    elif alias_lower in ('loglike',):
        return DataColumns.LOG_LIKE
    elif alias_lower in ('iteration',):
        return DataColumns.ITERATION
    elif alias_lower in ('trackid', 'track id'):
        return DataColumns.TRACK_ID
    elif alias_lower in (
            'neighbour_distance', 'neighbour_dist', 'neighbour distance'):
        return DataColumns.NEIGHBOUR_DISTANCE
    elif alias_lower in ('n_merged',):
        return DataColumns.N_MERGED
    elif alias_lower in ('crlb_x', 'crlb x'):
        return DataColumns.CRLB_X
    elif alias_lower in ('crlb_y', 'crlb y'):
        return DataColumns.CRLB_Y
    elif alias_lower in ('crlb_z', 'crlb z'):
        return DataColumns.CRLB_Z
    elif alias_lower in ('crlb_i', 'crlb i', 'crlb intensity'):
        return DataColumns.CRLB_I
    elif alias_lower in ('crlb_bg', 'crlb bg', 'crlb background'):
        return DataColumns.CRLB_BG
    elif alias_lower in ('crlb_sigx', 'crlb sigmax', 'crlb sigmax'):
        return DataColumns.CRLB_SIG_X
    elif alias_lower in ('crlb_sigy',  'crlb sigmay'):
        return DataColumns.CRLB_SIG_Y
    elif alias_lower in ('crlb_sigz', 'crlb sigmaz'):
        return DataColumns.CRLB_SIG_Z
    else:
        # raise ValueError(f'Unrecognized column alias: {alias}')
        return None


class FittingResults:

    def __init__(
            self,
            unit=ResultsUnits.Pixel, pixelSize=130.0,
            fittingMethod=FittingMethod._2D_Phasor_CPU):
        '''Fitting Results

        Parameters
        ----------
        unit : int, optional
            unit of localized points, by default ResultsUnits.Pixel
        pixelSize : float, optional
            pixel size in nanometers, by default 130.0
        '''
        self.unit = unit
        self.pixel_size = pixelSize
        self.fitting_method = fittingMethod
        if fittingMethod > -1:
            self.parameterHeader = PARAMETER_HEADERS[fittingMethod]
        else:
            self.parameterHeader = []

        # Initialize the dictionary with None for each key
        self.data = {key: None for key in DataColumns}

    def __len__(self):
        '''Number of localized positions
        '''
        return len(self.data[self.uniqueKeys()[0]])

    def __getattr__(self, name):
        key = map_column_alias(name)
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'.")

    def __setattr__(self, name, value):
        key = map_column_alias(name)
        if key is None:
            super().__setattr__(name, value)
        else:
            self.data[key] = value

    def uniqueKeys(self) -> list[DataColumns]:
        '''Returns a list of unique keys in FittingResults
        '''
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
            if self.fitting_method == FittingMethod._2D_Gauss_MLE_free_sigma or \
               self.fitting_method == FittingMethod._3D_Gauss_MLE_cspline_sigma:
                array[:, 4] *= self.pixel_size
            elif self.fitting_method == FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                array[:, 4:6] *= self.pixel_size

    def get_column(self, key: DataColumns):
        if self.data[key] is None:
            return np.zeros(self.__len__())
        else:
            return self.data[key]

    def extend(self, data: np.ndarray):
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

        for idx, var in enumerate([
                DataColumns.X, DataColumns.Y,
                DataColumns.INTENSITY, DataColumns.BACKGROUND]):
            self.extend_column(var, params[:, idx])

        self.extend_column(DataColumns.FRAME, frames)

        if self.fitting_method == FittingMethod._2D_Phasor_CPU:
            self.extend_column(DataColumns.XY_RATIO, params[:, -1])
        else:
            self.extend_column(DataColumns.ITERATION, params[:, -1])

        # Extend last column based on FittingMethod
        if self.fitting_method != FittingMethod._2D_Phasor_CPU:
            self.extend_column(DataColumns.LOG_LIKE, loglike)

            for idx, var in enumerate([
                    DataColumns.CRLB_X, DataColumns.CRLB_Y,
                    DataColumns.CRLB_I, DataColumns.CRLB_BG]):
                self.extend_column(var, crlbs[:, idx])

            if self.fitting_method == FittingMethod._2D_Gauss_MLE_free_sigma:
                self.extend_column(DataColumns.X_SIGMA, params[:, 4])
                self.extend_column(DataColumns.CRLB_SIG_X, crlbs[:, 4])
            elif self.fitting_method == \
                    FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                self.extend_column(DataColumns.X_SIGMA, params[:, 4])
                self.extend_column(DataColumns.Y_SIGMA, params[:, 5])
                self.extend_column(DataColumns.CRLB_SIG_X, crlbs[:, 4])
                self.extend_column(DataColumns.CRLB_SIG_Y, crlbs[:, 5])
            elif self.fitting_method == \
                    FittingMethod._3D_Gauss_MLE_cspline_sigma:
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

        if self.fitting_method == FittingMethod._2D_Phasor_CPU:
            loc = np.c_[
                self.data[DataColumns.FRAME],
                self.data[DataColumns.X],
                self.data[DataColumns.Y],
                self.data[DataColumns.BACKGROUND],
                self.data[DataColumns.INTENSITY],
                self.data[DataColumns.XY_RATIO],
                self.get_column(DataColumns.TRACK_ID),
                self.get_column(DataColumns.NEIGHBOUR_DISTANCE),
                self.get_column(DataColumns.N_MERGED)]

            columns = ['frame', ] + \
                self.parameterHeader + \
                [str(DataColumns.TRACK_ID),
                 str(DataColumns.NEIGHBOUR_DISTANCE),
                 str(DataColumns.N_MERGED)]
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
                self.data[DataColumns.BACKGROUND],
                self.data[DataColumns.INTENSITY]]

            if self.fitting_method == FittingMethod._2D_Gauss_MLE_fixed_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_BG],
                    self.data[DataColumns.CRLB_I]]
            elif self.fitting_method == FittingMethod._2D_Gauss_MLE_free_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.X_SIGMA],
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_BG],
                    self.data[DataColumns.CRLB_I],
                    self.data[DataColumns.CRLB_SIG_X]]
            elif self.fitting_method == \
                    FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.X_SIGMA],
                    self.data[DataColumns.Y_SIGMA],
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_BG],
                    self.data[DataColumns.CRLB_I],
                    self.data[DataColumns.CRLB_SIG_X],
                    self.data[DataColumns.CRLB_SIG_Y]]
            elif self.fitting_method == \
                    FittingMethod._3D_Gauss_MLE_cspline_sigma:
                loc = np.c_[
                    loc,
                    self.data[DataColumns.Z],
                    self.data[DataColumns.ITERATION],
                    self.data[DataColumns.CRLB_X],
                    self.data[DataColumns.CRLB_Y],
                    self.data[DataColumns.CRLB_BG],
                    self.data[DataColumns.CRLB_I],
                    self.data[DataColumns.CRLB_Z]]

            loc = np.c_[
                loc,
                self.data[DataColumns.LOG_LIKE],
                self.get_column(DataColumns.TRACK_ID),
                self.get_column(DataColumns.NEIGHBOUR_DISTANCE),
                self.get_column(DataColumns.N_MERGED)]

            columns = ['frame', ] + \
                self.parameterHeader + \
                ['CRLB ' + x for x in self.parameterHeader[:-1]] + \
                ['loglike',
                 str(DataColumns.TRACK_ID),
                 str(DataColumns.NEIGHBOUR_DISTANCE),
                 str(DataColumns.N_MERGED)]

        return pd.DataFrame(
            loc, columns=columns).sort_values(
            by=str(DataColumns.FRAME))

    def toRender(self):
        '''Returns columns for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains X [nm], Y [nm], Intensity columns
        '''
        # TODO: update for 3D render later
        return self.data[DataColumns.X], \
            self.data[DataColumns.Y], \
            self.data[DataColumns.INTENSITY]

    def drift_cross_correlation(self, n_bins=10, pixelSize=10, upsampling=100):
        '''Corrects the XY drift using cross-correlation measurments

        Parameters
        ----------
        n_bins : int, optional
            Number of frame bins, by default 10
        pixelSize : int, optional
            Super-res image pixel size in nanometers, by default 10
        upsampling : int, optional
            phase_cross_correlation upsampling (check skimage.registration),
            by default 100

        Returns
        -------
        tuple(FittingResults, np.ndarray)
            returns the drift corrected fittingResults and recontructed image
        '''
        unique_frames = np.unique(self.data[DataColumns.FRAME])
        if len(unique_frames) < 2:
            print('Drift cross-correlation failed: no frame info.')
            return

        frames_per_bin = int(np.floor(np.max(unique_frames) / n_bins))

        if frames_per_bin < 2:
            print('Drift cross-correlation failed: 2 or more bins are required.')
            return

        renderEngine = hist2D_render(pixelSize)

        x_max = int((np.max(self.data[DataColumns.X]) / renderEngine._pixel_size) +
                    4)
        y_max = int((np.max(self.data[DataColumns.Y]) / renderEngine._pixel_size) +
                    4)

        frame_ids = np.cumsum(
            np.bincount(self.data[DataColumns.FRAME].astype(np.int64)))
        max_frame = np.max(self.data[DataColumns.FRAME])

        # grouped_data = []
        sub_images = []
        shifts = []
        frames = []

        for f in range(0, n_bins):
            mask = slice(
                frame_ids[f * frames_per_bin - 1] if f * frames_per_bin > 0 else 0,
                frame_ids[(f + 1) * frames_per_bin])
            image = renderEngine.fromArray(
                np.c_[self.data[DataColumns.X][mask],
                      self.data[DataColumns.Y][mask],
                      self.data[DataColumns.INTENSITY][mask]],
                (y_max, x_max))
            frames.append(f * frames_per_bin + frames_per_bin/2)
            sub_images.append(image)
            print(
                f'Bins: {f + 1:d}/{n_bins:d}',
                end='\r')

        print(
            'Shift Estimation ...',
            end='\r')
        try:
            shifts = shift_estimation(
                np.array(sub_images), pixelSize, upsampling)
        except _ArrayMemoryError as e:
            print(f'Drift cross-correlation failed: {e}')
            return


        shifts = np.c_[shifts, np.array(frames)]
        print(
            'Shift Correction ...',
            end='\r')

        # An one-dimensional interpolation is applied
        # to drift traces in X and Y dimensions separately.
        interpy = interp1d(
            shifts[:, -1], shifts[:, 0],
            kind='quadratic', fill_value='extrapolate')
        interpx = interp1d(
            shifts[:, -1], shifts[:, 1],
            kind='quadratic', fill_value='extrapolate')
        # And this interpolation is used to get the shift at every frame-point
        frames_new = np.arange(0, max_frame, 1)
        interpx = interpx(frames_new)
        interpy = interpy(frames_new)

        shift_correction(
            interpx, interpy,
            frame_ids,
            self.data[DataColumns.X], self.data[DataColumns.Y])

        drift_corrected_image = renderEngine.render(
            self.data[DataColumns.X],
            self.data[DataColumns.Y],
            self.data[DataColumns.INTENSITY])

        return self, drift_corrected_image, \
            (frames_new, interpx, interpy)

    def nn_trajectories(
            self, minDistance: float = 0, maxDistance: float = 30,
            maxOff=1, neighbors=1):
        self.trackID = np.zeros(self.__len__(), dtype=np.int64)
        self.neighbour_dist = np.zeros(self.__len__(), dtype=np.float64)
        # data = self.dataFrame().to_numpy()
        data = np.c_[
            self.data[DataColumns.FRAME],
            self.data[DataColumns.X],
            self.data[DataColumns.Y]]

        ids = np.cumsum(
            np.bincount(self.data[DataColumns.FRAME].astype(np.int64)))

        counter = 0
        min_frame = np.min(self.data[DataColumns.FRAME])
        max_frame = np.max(self.data[DataColumns.FRAME])

        for frameID in np.arange(min_frame, max_frame, dtype=np.int64):
            for offset in np.arange(0, maxOff + 1, dtype=np.int64):
                nextID = frameID + offset + 1

                if nextID > max_frame:
                    continue

                currentMask = slice(
                    ids[frameID-1] if frameID > 0 else 0, ids[frameID])
                nextMask = slice(ids[nextID-1], ids[nextID])

                counter, self.trackID[currentMask], \
                    self.trackID[nextMask], \
                    self.neighbour_dist[nextMask] = nn_trajectories(
                        data[currentMask, :], data[nextMask, :],
                        c_trackID=self.trackID[currentMask],
                        n_trackID=self.trackID[nextMask],
                        nn_dist=self.neighbour_dist[nextMask],
                        counter=counter, minDistance=minDistance,
                        maxDistance=maxDistance,
                        neighbors=neighbors)
            print(
                f'NN {frameID / max_frame:.2%} ...               ',
                end='\r')

        print('Done ...                         ')

        return self

    def merge_tracks(self, maxLength=500):
        self.n_merged = np.zeros(self.__len__(), dtype=np.int64)
        df = self.dataFrame()
        columns = list(df.columns)

        column_ids = np.zeros(4, dtype=np.int32)
        column_ids[0] = columns.index('frame')
        column_ids[1] = columns.index('trackID')
        column_ids[2] = columns.index('n_merged')
        if 'I' in columns:
            column_ids[3] = columns.index('I')
        elif 'intensity' in columns:
            column_ids[3] = columns.index('intensity')

        data = df.to_numpy()

        print('Merging ...                         ')
        finalData = merge_localizations(data, column_ids, maxLength)

        df = pd.DataFrame(
            finalData,
            columns=columns).sort_values(
                by=str(DataColumns.FRAME))

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixel_size)

    def nearest_neighbour_merging(
            self, minDistance=0, maxDistance=30,
            maxOff=1, maxLength=500, neighbors=1):
        # Call nn_trajectories
        self.nn_trajectories(minDistance, maxDistance, maxOff, neighbors)

        # Call merge_tracks
        return self.merge_tracks(maxLength)

    def drift_fiducial_marker(self):
        df = self.dataFrame()
        columns = df.columns
        data = df.to_numpy()

        unique_frames, frame_counts = \
            np.unique(self.data[DataColumns.FRAME], return_counts=True)
        if len(unique_frames) < 2:
            print('Drift correction failed: no frame info.')
            return

        if self.data[DataColumns.TRACK_ID] is None:
            print('Drift correction failed: please perform NN, no trackIDs.')
            return
        if np.max(self.data[DataColumns.TRACK_ID]) < 1:
            print('Drift correction failed: please perform NN, no trackIDs.')
            return

        unique_tracks, track_counts = \
            np.unique(self.data[DataColumns.TRACK_ID], return_counts=True)
        fiducial_tracks_mask = np.logical_and(
            unique_tracks > 0,
            track_counts == len(unique_frames)
        )

        fiducial_trackIDs = unique_tracks[fiducial_tracks_mask]

        if len(fiducial_trackIDs) < 1:
            print(
                'Drift correction failed: no ' +
                'fiducial markers tracks detected.')
            return
        else:
            print(f'{len(fiducial_trackIDs):d} tracks detected.', end='\r')

        fiducial_markers = np.zeros((
            len(fiducial_trackIDs),
            len(unique_frames), 2))

        print(
            'Fiducial markers ...                 ', end='\r')
        for idx in np.arange(fiducial_markers.shape[0]):
            mask = self.data[DataColumns.TRACK_ID] == fiducial_trackIDs[idx]
            fiducial_markers[idx] = np.c_[
                self.data[DataColumns.X][mask],
                self.data[DataColumns.Y][mask]]

            fiducial_markers[idx, :, 0] -= fiducial_markers[idx, 0, 0]
            fiducial_markers[idx, :, 1] -= fiducial_markers[idx, 0, 1]

        print(
            'Drift estimate ...                 ', end='\r')
        drift_x = np.mean(fiducial_markers[:, :, 0], axis=0)
        drift_y = np.mean(fiducial_markers[:, :, 1], axis=0)

        print(
            'Drift correction ...                 ', end='\r')
        ids = np.cumsum(
            np.bincount(self.data[DataColumns.FRAME].astype(np.int64)))

        for idx in np.arange(len(unique_frames)):
            frame_id = int(unique_frames[idx])
            mask = slice(
                    ids[frame_id-1] if frame_id > 0 else 0, ids[frame_id])
            self.data[DataColumns.X][mask] -= drift_x[idx]
            self.data[DataColumns.Y][mask] -= drift_y[idx]

        # df = pd.DataFrame(
        #     data,
        #     columns=columns)

        # drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        return self, \
            (unique_frames, drift_x, drift_y)

    def zero_coordinates(self):
        if self.data[DataColumns.X] is not None:
            self.data[DataColumns.X] -= np.min(self.data[DataColumns.X])
        if self.data[DataColumns.Y] is not None:
            self.data[DataColumns.Y] -= np.min(self.data[DataColumns.Y])
        if self.data[DataColumns.Z] is not None:
            self.data[DataColumns.Z] -= np.min(self.data[DataColumns.Z])

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
            dataFrame = pd.read_csv(
                    filename,
                    sep='\t',
                    engine='python')
        elif '.h5' in filename:
            dataFrame = pd.read_hdf(
                    filename,
                    mode='r')
        else:
            raise Exception(
                'Supported file types are dataframes ' +
                'stored as .tsv or .h5 files.')

        return FittingResults.fromDataFrame(dataFrame, pixelSize)

    @staticmethod
    def fromDataFrame(dataFrame: pd.DataFrame, pixelSize: float):
        fittingResults = None
        fittingResults = FittingResults(
            ResultsUnits.Nanometer, 1, FittingMethod._External
        )

        for column in dataFrame.columns.values.tolist():
            key = map_column_alias(column)
            if key is not None:
                fittingResults.data[key] = dataFrame[column].to_numpy()

        return fittingResults
