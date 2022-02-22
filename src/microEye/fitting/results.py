
import numpy as np
import pandas as pd
from PyQt5.QtCore import *
from scipy.interpolate import interp1d

from ..Rendering import gauss_hist_render
from .processing import *


class ResultsUnits:
    Pixel = 0
    Nanometer = 1


class FittingMethod:
    _2D_Phasor_CPU = 0
    _2D_Gauss_MLE_CPU = 1


class FittingResults:

    columns = np.array([
        'frame', 'x [pixel]', 'y [pixel]', 'x [nm]', 'y [nm]', 'z [nm]',
        'intensity', 'trackID', 'neighbour_dist [nm]', 'n_merged'
    ])

    def __init__(self, unit=ResultsUnits.Pixel, pixelSize=130.0):
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
        self.locX = []
        self.locY = []
        self.locX_nm = []
        self.locY_nm = []
        self.locZ_nm = []
        self.frame = []
        self.intensity = []
        self.trackID = []
        self.neighbour_dist = []
        self.n_merged = []

    def extend(self, data: np.ndarray):
        '''Extend results by contents of data array

        Parameters
        ----------
        data : np.ndarray
            array of shape (n, m=4), columns (X, Y, Intensity, Frame)
        '''
        if self.unit is ResultsUnits.Pixel:
            self.locX.extend(data[:, 0])
            self.locY.extend(data[:, 1])
        else:
            self.locX_nm.extend(data[:, 0])
            self.locY_nm.extend(data[:, 1])

        self.locZ_nm.extend(data[:, 2])
        self.intensity.extend(data[:, 3])
        self.frame.extend(data[:, 4])

        self.trackID.extend([0] * data.shape[0])
        self.neighbour_dist.extend([0] * data.shape[0])
        self.n_merged.extend([0] * data.shape[0])

    def dataFrame(self):
        '''Return fitting results as Pandas DataFrame

        Returns
        -------
        DataFrame
            fitting results DataFrame with columns FittingResults.columns
        '''
        if self.unit is ResultsUnits.Pixel:
            loc = np.c_[
                    np.array(self.frame),
                    np.array(self.locX),
                    np.array(self.locY),
                    np.array(self.locX) * self.pixelSize,
                    np.array(self.locY) * self.pixelSize,
                    np.array(self.locZ_nm),
                    np.array(self.intensity),
                    np.array(self.trackID),
                    np.array(self.neighbour_dist),
                    np.array(self.n_merged)]
        else:
            loc = np.c_[
                    np.array(self.frame),
                    np.array(self.locX_nm) / self.pixelSize,
                    np.array(self.locY_nm) / self.pixelSize,
                    np.array(self.locX_nm),
                    np.array(self.locY_nm),
                    np.array(self.locZ_nm),
                    np.array(self.intensity),
                    np.array(self.trackID),
                    np.array(self.neighbour_dist),
                    np.array(self.n_merged)]

        return pd.DataFrame(
            loc, columns=FittingResults.columns).sort_values(
            by=FittingResults.columns[0])

    def toRender(self):
        '''Returns columns for rendering

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            tuple contains X [nm], Y [nm], Intensity columns
        '''
        if self.unit is ResultsUnits.Pixel:
            return np.array(self.locX) * self.pixelSize, \
                np.array(self.locY) * self.pixelSize, \
                np.array(self.intensity)
        else:
            return np.array(self.locX_nm), \
                np.array(self.locY_nm), \
                np.array(self.intensity)

    def __len__(self):
        counts = [len(self.locX), len(self.locY),
                  len(self.locX_nm), len(self.locY_nm)]
        return np.max(counts)

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

    def fromDataFrame(dataFrame: pd.DataFrame, pixelSize: float):
        fittingResults = None

        if FittingResults.columns[3] in dataFrame and \
                FittingResults.columns[4] in dataFrame:
            fittingResults = FittingResults(
                ResultsUnits.Nanometer,
                pixelSize)
            fittingResults.locX_nm = \
                dataFrame[FittingResults.columns[3]]
            fittingResults.locY_nm = \
                dataFrame[FittingResults.columns[4]]
        elif FittingResults.columns[1] in dataFrame and \
                FittingResults.columns[2] in dataFrame:
            fittingResults = FittingResults(
                ResultsUnits.Pixel,
                pixelSize)
            fittingResults.locX = \
                dataFrame[FittingResults.columns[1]]
            fittingResults.locY = \
                dataFrame[FittingResults.columns[2]]
        else:
            return None

        if FittingResults.columns[0] in dataFrame:
            fittingResults.frame = dataFrame[FittingResults.columns[0]]
        else:
            fittingResults.frame = np.zeros(len(fittingResults))

        if FittingResults.columns[5] in dataFrame:
            fittingResults.locZ_nm = dataFrame[FittingResults.columns[5]]
        else:
            fittingResults.locZ_nm = np.zeros(len(fittingResults))

        if FittingResults.columns[6] in dataFrame:
            fittingResults.intensity = dataFrame[FittingResults.columns[6]]
        else:
            fittingResults.intensity = np.ones(len(fittingResults))

        if FittingResults.columns[7] in dataFrame:
            fittingResults.trackID = dataFrame[FittingResults.columns[7]]
        else:
            fittingResults.trackID = np.zeros(len(fittingResults))

        if FittingResults.columns[8] in dataFrame:
            fittingResults.neighbour_dist = \
                dataFrame[FittingResults.columns[8]]
        else:
            fittingResults.neighbour_dist = np.zeros(len(fittingResults))

        if FittingResults.columns[9] in dataFrame:
            fittingResults.n_merged = dataFrame[FittingResults.columns[9]]
        else:
            fittingResults.n_merged = np.zeros(len(fittingResults))

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
        unique_frames = np.unique(self.frame)
        if len(unique_frames) < 2:
            print('Drift cross-correlation failed: no frame info.')
            return

        frames_per_bin = np.floor(np.max(unique_frames) / n_bins)

        if frames_per_bin < 1:
            print('Drift cross-correlation failed: large number of bins.')
            return

        renderEngine = gauss_hist_render(pixelSize)

        data = self.dataFrame().to_numpy()

        x_max = int((np.max(data[:, 3]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)
        y_max = int((np.max(data[:, 4]) / renderEngine._pixel_size) +
                    4 * renderEngine._gauss_len)

        grouped_data = []
        sub_images = []
        shifts = []
        frames = []

        for f in range(0, n_bins):
            group = data[(data[:, 0] >= f * frames_per_bin) &
                         (data[:, 0] < (f + 1) * frames_per_bin + 1)]
            image = renderEngine.fromArray(group[:, [3, 4, 6]], (y_max, x_max))
            frames.append(f * frames_per_bin + frames_per_bin/2)
            grouped_data.append(group)
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

        shift_correction(interpx, interpy, data)
        # for i, (shift_x, shift_y) in enumerate(zip(interpx, interpy)):
        #     data[data[:, 0] == i, 3] -= shift_x
        #     data[data[:, 0] == i, 4] -= shift_y

        df = pd.DataFrame(
            data,
            columns=FittingResults.columns)

        drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        drift_corrected_image = renderEngine.render(
            *drift_corrected.toRender())

        return drift_corrected, drift_corrected_image, \
            (frames_new, interpx, interpy)

    def nn_trajectories(
            self, maxDistance=30, maxOff=1, neighbors=1):
        data = self.dataFrame().to_numpy()
        data[:, 7] = 0
        counter = 0
        max_frame = np.max(self.frame)

        for frameID in np.arange(0, max_frame + 1):
            for offset in np.arange(0, maxOff + 1):
                counter = nn_trajectories(
                    data,
                    frameID=frameID, nextID=frameID + offset + 1,
                    counter=counter, maxDistance=maxDistance,
                    neighbors=neighbors)
            print(
                'NN {:.2%} ...               '.format(frameID / max_frame),
                end="\r")

        df = pd.DataFrame(
            data,
            columns=FittingResults.columns).sort_values(
                by=FittingResults.columns[0])

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def merge_tracks(self, maxLength=500):
        data = self.dataFrame().to_numpy()

        print('Merging ...                         ')
        finalData = merge_localizations(data, maxLength)

        df = pd.DataFrame(
            finalData,
            columns=FittingResults.columns).sort_values(
                by=FittingResults.columns[0])

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def nearest_neighbour_merging(
            self, maxDistance=30, maxOff=1, maxLength=500, neighbors=1):
        data = self.dataFrame().to_numpy()
        data[:, 7] = 0
        counter = 0
        max_frame = np.max(self.frame)

        for frameID in np.arange(0, max_frame + 1):
            for offset in np.arange(0, maxOff + 1):
                counter = nn_trajectories(
                    data,
                    frameID=frameID, nextID=frameID + offset + 1,
                    counter=counter, maxDistance=maxDistance,
                    neighbors=neighbors)
            print(
                'NN {:.2%} ...               '.format(frameID / max_frame),
                end="\r")

        print('Merging ...                         ')
        finalData = merge_localizations(data, maxLength)

        df = pd.DataFrame(
            finalData,
            columns=FittingResults.columns).sort_values(
                by=FittingResults.columns[0])

        print('Done ...                         ')

        return FittingResults.fromDataFrame(df, self.pixelSize)

    def drift_fiducial_marker(self):
        data = self.dataFrame().to_numpy()

        unique_frames, frame_counts = \
            np.unique(data[:, 0], return_counts=True)
        if len(unique_frames) < 2:
            print('Drift correction failed: no frame info.')
            return

        if np.max(data[:, 7]) < 1:
            print('Drift correction failed: please perform NN, no trackIDs.')
            return

        unique_tracks, track_counts = \
            np.unique(data[:, 7], return_counts=True)
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
            len(unique_frames),
            data.shape[-1]))

        print(
            'Fiducial markers ...                 ', end='\r')
        for idx in np.arange(fiducial_markers.shape[0]):
            fiducial_markers[idx] = \
                data[data[:, 7] == fiducial_trackIDs[idx], ...]

            fiducial_markers[idx, :, 3] -= fiducial_markers[idx, 0, 3]
            fiducial_markers[idx, :, 4] -= fiducial_markers[idx, 0, 4]

        print(
            'Drift estimate ...                 ', end='\r')
        drift_x = np.mean(fiducial_markers[:, :, 3], axis=0)
        drift_y = np.mean(fiducial_markers[:, :, 4], axis=0)

        print(
            'Drift correction ...                 ', end='\r')
        for idx in np.arange(len(unique_frames)):
            frame = unique_frames[idx]
            data[data[:, 0] == frame, 3] -= drift_x[idx]
            data[data[:, 0] == frame, 4] -= drift_y[idx]

        df = pd.DataFrame(
            data,
            columns=FittingResults.columns)

        drift_corrected = FittingResults.fromDataFrame(df, self.pixelSize)

        return drift_corrected, \
            (unique_frames, drift_x, drift_y)
