
import numpy as np
import pandas as pd
from PyQt5.QtCore import *
from pkg_resources import yield_lines
from scipy.interpolate import interp1d

from ..Rendering import gauss_hist_render
from .processing import *

from enum import Enum


class DataColumns(Enum):
    Frame = (0, 'frame')
    X = (1, 'x')
    Y = (2, 'y')
    Z = (3, 'z')
    X_nm = (4, 'x [nm]')
    Y_nm = (5, 'y [nm]')
    Z_nm = (6, 'z [nm]')
    Background = (7, 'background')
    Intensity = (8, 'intensity')
    TrackID = (9, 'trackID')
    NeighbourDist = (10, 'neighbour_dist')
    NMerged = (11, 'n_merged')

    XY_Ratio = (10, 'ratio x/y')
    Sigma = (11, 'sigma')
    X_Sigma = (11, 'X_Sigma')
    Y_Sigma = (12, 'Y_Sigma')

    def __str__(self):
        return self.value[1]

    @classmethod
    def from_string(cls, s):
        for column in cls:
            if column.value[1] == s:
                return column
        raise ValueError(cls.__name__ + ' has no value matching "' + s + '"')

    @classmethod
    def values(cls):
        res = []
        for column in cls:
            res.append(column.value[1])
        return np.array(res)


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


ParametersHeaders = {
    0: ['x', 'y', 'bg', 'I', 'ratio x/y', 'frame'],
    1: ['x', 'y', 'bg', 'I', 'iteration'],
    2: ['x', 'y', 'bg', 'I', 'sigma', 'iteration'],
    4: ['x', 'y', 'bg', 'I', 'sigmax', 'sigmay', 'iteration'],
    5: ['x', 'y', 'bg', 'I', 'z', 'iteration']
}


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
        self.pixelSize = pixelSize
        self.fittingMethod = fittingMethod
        if fittingMethod > -1:
            self.parameterHeader = ParametersHeaders[fittingMethod]
        else:
            self.parameterHeader = []

        self.frames = None
        self.locX = None
        self.locY = None
        self.locZ = None
        self.intensity = None
        self.background = None

        self.xy_Ratio = None
        self.x_Sigma = None
        self.y_Sigma = None

        self.crlb_X = None
        self.crlb_Y = None
        self.crlb_Z = None
        self.crlb_I = None
        self.crlb_BG = None
        self.crlb_SigX = None
        self.crlb_SigY = None

        self.loglike = None
        self.iteration = None

        self.trackID = None
        self.neighbour_dist = None
        self.n_merged = None

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
            if params is not None:
                params[:, :2] *= self.pixelSize
                if self.fittingMethod == \
                        FittingMethod._2D_Gauss_MLE_free_sigma:
                    params[:, 4] *= self.pixelSize
                elif self.fittingMethod == \
                        FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                    params[:, 4:6] *= self.pixelSize

            if crlbs is not None:
                crlbs[:, :2] *= self.pixelSize
                if self.fittingMethod == \
                        FittingMethod._2D_Gauss_MLE_free_sigma:
                    crlbs[:, 4] *= self.pixelSize
                elif self.fittingMethod == \
                        FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                    crlbs[:, 4:6] *= self.pixelSize

        if self.locX is None:
            self.locX = params[:, 0]
        else:
            self.locX = np.concatenate(
                (self.locX, params[:, 0]))

        if self.locY is None:
            self.locY = params[:, 1]
        else:
            self.locY = np.concatenate(
                (self.locY, params[:, 1]))

        if self.background is None:
            self.background = params[:, 2]
        else:
            self.background = np.concatenate(
                (self.background, params[:, 2]))

        if self.intensity is None:
            self.intensity = params[:, 3]
        else:
            self.intensity = np.concatenate(
                (self.intensity, params[:, 3]))

        if self.frames is None:
            self.frames = frames
        else:
            self.frames = np.concatenate(
                (self.frames, frames))

        if self.fittingMethod == FittingMethod._2D_Phasor_CPU:
            if self.xy_Ratio is None:
                self.xy_Ratio = params[:, 4]
            else:
                self.xy_Ratio = np.concatenate(
                    (self.xy_Ratio, params[:, 4]))
        else:
            if self.iteration is None:
                self.iteration = params[:, -1]
            else:
                self.iteration = np.concatenate(
                    (self.iteration, params[:, -1]))

            if self.loglike is None:
                self.loglike = loglike
            else:
                self.loglike = np.concatenate(
                    (self.loglike, loglike))

            if self.crlb_X is None:
                self.crlb_X = crlbs[:, 0]
            else:
                self.crlb_X = np.concatenate(
                    (self.crlb_X, crlbs[:, 0]))

            if self.crlb_Y is None:
                self.crlb_Y = crlbs[:, 1]
            else:
                self.crlb_Y = np.concatenate(
                    (self.crlb_Y, crlbs[:, 1]))

            if self.crlb_BG is None:
                self.crlb_BG = crlbs[:, 2]
            else:
                self.crlb_BG = np.concatenate(
                    (self.crlb_BG, crlbs[:, 2]))

            if self.crlb_I is None:
                self.crlb_I = crlbs[:, 3]
            else:
                self.crlb_I = np.concatenate(
                    (self.crlb_I, crlbs[:, 3]))

            if self.fittingMethod == FittingMethod._2D_Gauss_MLE_free_sigma:
                if self.x_Sigma is None:
                    self.x_Sigma = params[:, -2]
                else:
                    self.x_Sigma = np.concatenate(
                        (self.x_Sigma, params[:, -2]))

                if self.crlb_SigX is None:
                    self.crlb_SigX = crlbs[:, -1]
                else:
                    self.crlb_SigX = np.concatenate(
                        (self.crlb_SigX, crlbs[:, -1]))
            elif self.fittingMethod == \
                    FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                if self.x_Sigma is None:
                    self.x_Sigma = params[:, -3]
                else:
                    self.x_Sigma = np.concatenate(
                        (self.x_Sigma, params[:, -3]))
                if self.y_Sigma is None:
                    self.y_Sigma = params[:, -2]
                else:
                    self.y_Sigma = np.concatenate(
                        (self.y_Sigma, params[:, -2]))

                if self.crlb_SigX is None:
                    self.crlb_SigX = crlbs[:, -2]
                else:
                    self.crlb_SigX = np.concatenate(
                        (self.crlb_SigX, crlbs[:, -2]))
                if self.crlb_SigY is None:
                    self.crlb_SigY = crlbs[:, -1]
                else:
                    self.crlb_SigY = np.concatenate(
                        (self.crlb_SigY, crlbs[:, -1]))
            elif self.fittingMethod == \
                    FittingMethod._3D_Gauss_MLE_cspline_sigma:
                if self.locZ is None:
                    self.locZ = params[:, -2]
                else:
                    self.locZ = np.concatenate(
                        (self.locZ, params[:, -2]))

                if self.crlb_Z is None:
                    self.crlb_Z = crlbs[:, -1]
                else:
                    self.crlb_Z = np.concatenate(
                        (self.crlb_Z, crlbs[:, -1]))

    def dataFrame(self):
        '''Return fitting results as Pandas DataFrame

        Returns
        -------
        DataFrame
            fitting results DataFrame with columns FittingResults.columns
        '''
        if self.fittingMethod == FittingMethod._2D_Phasor_CPU:
            loc = np.c_[
                self.locX,
                self.locY,
                self.background,
                self.intensity,
                self.xy_Ratio,
                self.frames,
                np.zeros(
                    self.__len__()
                    ) if self.trackID is None else self.trackID,
                np.zeros(
                    self.__len__()
                    ) if self.neighbour_dist is None else self.neighbour_dist,
                np.zeros(
                    self.__len__()
                    ) if self.n_merged is None else self.n_merged]

            columns = self.parameterHeader + \
                [str(DataColumns.TrackID),
                 str(DataColumns.NeighbourDist),
                 str(DataColumns.NMerged)]
        elif self.fittingMethod == -1:
            loc = None
            keys = self.uniqueKeys()
            columns = []
            for key in keys:
                col = self.getColumn(key)
                if col is not None:
                    columns.append(key)
                    if loc is None:
                        loc = col
                    else:
                        loc = np.c_[loc, col]
        else:
            loc = np.c_[
                self.frames,
                self.locX,
                self.locY,
                self.background,
                self.intensity]

            if self.fittingMethod == FittingMethod._2D_Gauss_MLE_fixed_sigma:
                loc = np.c_[
                    loc,
                    self.iteration,
                    self.crlb_X,
                    self.crlb_Y,
                    self.crlb_BG,
                    self.crlb_I]
            elif self.fittingMethod == FittingMethod._2D_Gauss_MLE_free_sigma:
                loc = np.c_[
                    loc,
                    self.x_Sigma,
                    self.iteration,
                    self.crlb_X,
                    self.crlb_Y,
                    self.crlb_BG,
                    self.crlb_I,
                    self.crlb_SigX]
            elif self.fittingMethod == \
                    FittingMethod._2D_Gauss_MLE_elliptical_sigma:
                loc = np.c_[
                    loc,
                    self.x_Sigma,
                    self.y_Sigma,
                    self.iteration,
                    self.crlb_X,
                    self.crlb_Y,
                    self.crlb_BG,
                    self.crlb_I,
                    self.crlb_SigX,
                    self.crlb_SigY]
            elif self.fittingMethod == \
                    FittingMethod._3D_Gauss_MLE_cspline_sigma:
                loc = np.c_[
                    loc,
                    self.locZ,
                    self.iteration,
                    self.crlb_X,
                    self.crlb_Y,
                    self.crlb_BG,
                    self.crlb_I,
                    self.crlb_Z]

            loc = np.c_[
                loc,
                self.loglike,
                np.zeros(
                    self.__len__()
                    ) if self.trackID is None else self.trackID,
                np.zeros(
                    self.__len__()
                    ) if self.neighbour_dist is None else self.neighbour_dist,
                np.zeros(
                    self.__len__()
                    ) if self.n_merged is None else self.n_merged]

            columns = ['frame', ] + \
                self.parameterHeader + \
                ['CRLB ' + x for x in self.parameterHeader[:-1]] + \
                ['loglike',
                 str(DataColumns.TrackID),
                 str(DataColumns.NeighbourDist),
                 str(DataColumns.NMerged)]

        return pd.DataFrame(
            loc, columns=columns).sort_values(
            by=str(DataColumns.Frame))

    def toRender(self):
        '''Returns columns for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains X [nm], Y [nm], Intensity columns
        '''
        return self.locX, \
            self.locY, \
            self.intensity

    def __len__(self):
        if self.locX is None:
            return 0
        else:
            return len(self.locX)

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
        dataFrame = pd.read_csv(
                filename,
                sep='\t',
                engine='python')

        return FittingResults.fromDataFrame(dataFrame, pixelSize)

    def setColumn(self, key, value):
        if key == 'frame':
            self.frames = value
        elif key == 'x' or key == 'x [nm]':
            self.locX = value
        elif key == 'y' or key == 'y [nm]':
            self.locY = value
        elif key == 'z' or key == 'z [nm]':
            self.locZ = value
        elif key == 'bg' or key == 'background':
            self.background = value
        elif key == 'I' or key == 'intensity':
            self.intensity = value
        elif key == 'sigmax' or key == 'sigma':
            self.x_Sigma = value
        elif key == 'sigmay':
            self.y_Sigma = value
        elif 'ratio' in key and 'iteration' not in key:
            self.xy_Ratio = value
        elif key == 'loglike':
            self.loglike = value
        elif key == 'iteration':
            self.iteration = value
        elif key == 'trackID':
            self.trackID = value
        elif 'neighbour_dist' in key:
            self.neighbour_dist = value
        elif key == 'n_merged':
            self.n_merged = value
        elif key == 'CRLB x':
            self.crlb_X = value
        elif key == 'CRLB y':
            self.crlb_Y = value
        elif key == 'CRLB z':
            self.crlb_Z = value
        elif key == 'CRLB I':
            self.crlb_I = value
        elif key == 'CRLB bg':
            self.crlb_BG = value
        elif key == 'CRLB sigma' or key == 'CRLB sigmax':
            self.crlb_SigX = value
        elif key == 'CRLB sigmay':
            self.crlb_SigY = value

    def getColumn(self, key):
        if key == 'frame':
            return self.frames
        elif key == 'x' or key == 'x [nm]':
            return self.locX
        elif key == 'y' or key == 'y [nm]':
            return self.locY
        elif key == 'z' or key == 'z [nm]':
            return self.locZ
        elif key == 'bg' or key == 'background':
            return self.background
        elif key == 'I' or key == 'intensity':
            return self.intensity
        elif key == 'sigmax' or key == 'sigma':
            return self.x_Sigma
        elif key == 'sigmay':
            return self.y_Sigma
        elif 'ratio' in key and 'iteration' not in key:
            return self.xy_Ratio
        elif key == 'loglike':
            return self.loglike
        elif key == 'iteration':
            return self.iteration
        elif key == 'trackID':
            return self.trackID
        elif 'neighbour_dist' in key:
            return self.neighbour_dist
        elif key == 'n_merged':
            return self.n_merged
        elif key == 'CRLB x':
            return self.crlb_X
        elif key == 'CRLB y':
            return self.crlb_Y
        elif key == 'CRLB z':
            return self.crlb_Z
        elif key == 'CRLB I':
            return self.crlb_I
        elif key == 'CRLB bg':
            return self.crlb_BG
        elif key == 'CRLB sigma' or key == 'CRLB sigmax':
            return self.crlb_SigX
        elif key == 'CRLB sigmay':
            return self.crlb_SigY

    def columnKeys(self):
        return [
            'frame', 'x', 'y', 'z', 'x [nm]', 'y [nm]', 'z [nm]',
            'bg', 'background', 'I', 'intensity', 'sigma', 'sigmax',
            'sigmay', 'ratio', 'loglike', 'iteration', 'trackID',
            'neighbour_dist', 'n_merged', 'CRLB x', 'CRLB y', 'CRLB z',
            'CRLB I', 'CRLB bg', 'CRLB sigma', 'CRLB sigmax', 'CRLB sigmay']

    def uniqueKeys(self):
        return [
            'frame', 'x', 'y', 'z',
            'bg', 'I', 'sigmax',
            'sigmay', 'ratio', 'loglike', 'iteration', 'trackID',
            'neighbour_dist', 'n_merged', 'CRLB x', 'CRLB y', 'CRLB z',
            'CRLB I', 'CRLB bg', 'CRLB sigmax', 'CRLB sigmay']

    def fromDataFrame(dataFrame: pd.DataFrame, pixelSize: float):
        fittingResults = None
        fittingResults = FittingResults(
            ResultsUnits.Nanometer, 1, FittingMethod._External
        )

        for key in fittingResults.columnKeys():
            if key in dataFrame:
                fittingResults.setColumn(
                    key, dataFrame[key].to_numpy())

        return fittingResults

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
        unique_frames = np.unique(self.frames)
        if len(unique_frames) < 2:
            print('Drift cross-correlation failed: no frame info.')
            return

        frames_per_bin = np.floor(np.max(unique_frames) / n_bins)

        if frames_per_bin < 2:
            print('Drift cross-correlation failed: large number of bins.')
            return

        renderEngine = gauss_hist_render(pixelSize)

        # data = self.dataFrame().to_numpy()

        x_max = int((np.max(self.locX) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)
        y_max = int((np.max(self.locY) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)

        # grouped_data = []
        sub_images = []
        shifts = []
        frames = []

        for f in range(0, n_bins):
            mask = (self.frames >= f * frames_per_bin) & \
                (self.frames < (f + 1) * frames_per_bin + 1)
            image = renderEngine.fromArray(
                np.c_[self.locX[mask], self.locY[mask], self.intensity[mask]],
                (y_max, x_max))
            frames.append(f * frames_per_bin + frames_per_bin/2)
            sub_images.append(image)
            print(
                'Bins: {:d}/{:d}'.format(f + 1, n_bins),
                end="\r")

        print(
            'Shift Estimation ...',
            end="\r")
        shifts = shift_estimation(np.array(sub_images), pixelSize, upsampling)
        # for idx, img in enumerate(sub_images):
        #     shift = phase_cross_correlation(
        #         img, sub_images[0], upsample_factor=upsampling)
        #     shifts.append(shift[0] * pixelSize)
        #     print(
        #         'Shift Est.: {:d}/{:d}'.format(idx + 1, len(sub_images)),
        #         end="\r")

        shifts = np.c_[shifts, np.array(frames)]
        print(
            'Shift Correction ...',
            end="\r")

        # An one-dimensional interpolation is applied
        # to drift traces in X and Y dimensions separately.
        interpy = interp1d(
            shifts[:, -1], shifts[:, 0],
            kind='quadratic', fill_value='extrapolate')
        interpx = interp1d(
            shifts[:, -1], shifts[:, 1],
            kind='quadratic', fill_value='extrapolate')
        # And this interpolation is used to get the shift at every frame-point
        frames_new = np.arange(0, np.max(unique_frames), 1)
        interpx = interpx(frames_new)
        interpy = interpy(frames_new)

        shift_correction(interpx, interpy, self.frames, self.locX, self.locY)
        # for i, (shift_x, shift_y) in enumerate(zip(interpx, interpy)):
        #     data[data[:, 0] == i, 3] -= shift_x
        #     data[data[:, 0] == i, 4] -= shift_y

        drift_corrected_image = renderEngine.render(
            self.locX, self.locY, self.intensity)

        return self, drift_corrected_image, \
            (frames_new, interpx, interpy)

    def nn_trajectories(
            self, minDistance=0, maxDistance=30, maxOff=1, neighbors=1):
        self.trackID = np.zeros(self.__len__(), dtype=np.int64)
        self.neighbour_dist = np.zeros(self.__len__(), dtype=np.float64)
        # data = self.dataFrame().to_numpy()
        data = np.c_[self.frames, self.locX, self.locY]

        counter = 0
        min_frame = np.min(self.frames)
        max_frame = np.max(self.frames)

        for frameID in np.arange(min_frame, max_frame + 1):
            for offset in np.arange(0, maxOff + 1):
                nextID = frameID + offset + 1

                currentMask = data[:, 0] == frameID
                nextMask = data[:, 0] == nextID

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
                'NN {:.2%} ...               '.format(frameID / max_frame),
                end="\r")

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
                by=str(DataColumns.Frame))

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def nearest_neighbour_merging(
            self, minDistance=0, maxDistance=30,
            maxOff=1, maxLength=500, neighbors=1):
        self.trackID = np.zeros(self.__len__(), dtype=np.int64)
        self.neighbour_dist = np.zeros(self.__len__(), dtype=np.float64)
        # data = self.dataFrame().to_numpy()
        data = np.c_[self.frames, self.locX, self.locY]

        counter = 0
        min_frame = np.min(self.frames)
        max_frame = np.max(self.frames)

        for frameID in np.arange(min_frame, max_frame + 1):
            for offset in np.arange(0, maxOff + 1):
                nextID = frameID + offset + 1

                currentMask = data[:, 0] == frameID
                nextMask = data[:, 0] == nextID

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
                'NN {:.2%} ...               '.format(frameID / max_frame),
                end="\r")

        print('Merging ...                         ')
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
                by=str(DataColumns.Frame))

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def drift_fiducial_marker(self):
        df = self.dataFrame()
        columns = df.columns
        data = df.to_numpy()

        unique_frames, frame_counts = \
            np.unique(self.frames, return_counts=True)
        if len(unique_frames) < 2:
            print('Drift correction failed: no frame info.')
            return

        if self.trackID is None:
            print('Drift correction failed: please perform NN, no trackIDs.')
            return
        if np.max(self.trackID) < 1:
            print('Drift correction failed: please perform NN, no trackIDs.')
            return

        unique_tracks, track_counts = \
            np.unique(self.trackID, return_counts=True)
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
            print('{:d} tracks detected.'.format(
                len(fiducial_trackIDs)), end='\r')

        fiducial_markers = np.zeros((
            len(fiducial_trackIDs),
            len(unique_frames), 2))

        print(
            'Fiducial markers ...                 ', end='\r')
        for idx in np.arange(fiducial_markers.shape[0]):
            fiducial_markers[idx] = np.c_[
                self.locX, self.locY
                ][self.trackID == fiducial_trackIDs[idx], ...]

            fiducial_markers[idx, :, 0] -= fiducial_markers[idx, 0, 0]
            fiducial_markers[idx, :, 1] -= fiducial_markers[idx, 0, 1]

        print(
            'Drift estimate ...                 ', end='\r')
        drift_x = np.mean(fiducial_markers[:, :, 0], axis=0)
        drift_y = np.mean(fiducial_markers[:, :, 1], axis=0)

        print(
            'Drift correction ...                 ', end='\r')
        for idx in np.arange(len(unique_frames)):
            frame = unique_frames[idx]
            self.locX[self.frames == frame] -= drift_x[idx]
            self.locY[self.frames == frame] -= drift_y[idx]

        # df = pd.DataFrame(
        #     data,
        #     columns=columns)

        # drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        return self, \
            (unique_frames, drift_x, drift_y)
