import dask.array
import numpy as np

from microEye.analysis.filters.base import AbstractFilter
from microEye.utils.uImage import ZarrImageSequence


class TemporalMedianFilter(AbstractFilter):

    def __init__(self, window=3) -> None:
        super().__init__()

        self._temporal_window = window
        self._frames = None
        self._median = None

    def getFrames(
            self, index,
            dataHandler: ZarrImageSequence, c_slice=0, z_slice=0):
        if self._temporal_window < 2:
            self._frames = None
            self._median = None
            self._start = None
            return

        maximum = dataHandler.shapeTCZYX()[0]
        step = 0
        if self._temporal_window % 2:
            step = self._temporal_window // 2
        else:
            step = (self._temporal_window // 2) - 1

        self._start = max(
            0,
            min(
                index - step,
                maximum - self._temporal_window))


        data_slice = slice(
            max(self._start, 0),
            min(self._start + self._temporal_window, maximum),
            1
        )
        return dask.array.from_array(
            dataHandler.getSlice(data_slice, c_slice, z_slice))

    def run(self, image: np.ndarray, frames: np.ndarray, roiInfo=None):
        if frames is not None:
            if roiInfo is None:
                median = np.array(
                    dask.array.median(frames, axis=0))

                img = image - median
            else:
                origin = roiInfo[0]  # ROI (x,y)
                dim = roiInfo[1]  # ROI (w,h)
                median = np.array(
                    dask.array.median(frames[
                        :,
                        int(origin[1]):int(origin[1] + dim[1]),
                        int(origin[0]):int(origin[0] + dim[0])
                    ], axis=0))

                img = image.astype(np.float64)
                img[
                    int(origin[1]):int(origin[1] + dim[1]),
                    int(origin[0]):int(origin[0] + dim[0])] -= median

            img[img < 0] = 0
            # max_val = np.iinfo(image.dtype).max
            # img *= 10  # 0.9 * max_val / img.max()
            return img.astype(image.dtype)
        else:
            return image

