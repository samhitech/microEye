import logging
import os

import numba as nb
import numpy as np
import tifffile as tf


@nb.njit(parallel=True)
def update(count, mean, M2, new_value):
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2
    return (count, mean, M2)

class dark_calibration:
    '''A class holder for pixelwise average and variance.

    Intended to be used for photon-free calibration:
    https://www.biorxiv.org/lookup/doi/10.1101/2021.04.16.440125
    http://dx.doi.org/10.1038/s41467-022-30907-2

    ACCENT (ImageJ/Fiji Plugin):
    https://github.com/ries-lab/Accent
    '''

    def __init__(self, shape, exposure):
        self._count = 0
        self._exposure = exposure
        self._shape = shape
        self._mean = np.zeros(shape=shape, dtype=np.float64)
        self._M2 = np.zeros(shape=shape, dtype=np.float64)

    # Retrieve the mean, variance and sample variance from an aggregate
    def finalize(existing_aggregate):
        (count, mean, M2) = existing_aggregate
        if count < 2:
            return float('nan')
        else:
            (mean, variance, sample_variance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sample_variance)

    def addFrame(self, image: np.ndarray):
        '''Adds an image frame to the mean/variance estimators.

        Parameters
        ----------
        image : np.ndarray
            image of dark_calibration shape to be added.

        Raises
        ------
        ValueError
            image of wrong shape.
        '''
        if image.shape != self._shape:
            raise ValueError('Image of wrong shape.')

        frame = np.asarray(image, dtype=np.float64, order='C')

        self._count, self._mean, self._M2 = update(
            self._count,
            self._mean,
            self._M2,
            frame,
        )

    def getResults(self):
        '''Gets the resulting mean and variance frames.

        Returns
        -------
        tuple(ndarray, ndarray, ndarray)
            the rasults (mean, variance, sample_variance).

        Raises
        ------
        ValueError
            in case zero frames are added.
        '''
        if self._count < 2:
            raise ValueError('At least two frames are required.')

        (mean, variance, sample_variance) = (
            self._mean,
            self._M2 / self._count,
            self._M2 / (self._count - 1),
        )

        return mean, variance, sample_variance

    def saveResults(self, path: str, prefix: str):
        if not os.path.exists(path):
            os.makedirs(path)

        def getFilename(name: str):
            return (
                path
                + prefix
                + f'_image_{name}_{self._exposure:.5f}_ms'.replace('.', '_')
                + '.ome.tif'
            )

        mean, variance, sample_variance = self.getResults()

        self._write_tiff(
            getFilename('mean'),
            mean,
        )

        self._write_tiff(
            getFilename('var'),
            variance,
        )

        self._write_tiff(
            getFilename('samplevar'),
            sample_variance,
        )

    def _write_tiff(self, filename, data):
        try:
            with tf.TiffWriter(
                filename, append=False, bigtiff=False, ome=False
            ) as writer:
                writer.write(data=data, photometric='minisblack')
        except Exception as e:
            logging.getLogger(__name__).error(f'Error writing tiff file: {e}')
