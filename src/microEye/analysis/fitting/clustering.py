import logging

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def perform_dbscan_clustering(X, Y, Z, eps=50, min_samples=2, **kwargs):
    '''
    Performs DBSCAN clustering on 3D localization data.

    Parameters:
    ------------
    X, Y, Z: array-like
        Arrays of x, y, z coordinates of localizations.
    eps: float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood. Default is 50.
    min_samples: int, optional
        The number of samples in a neighborhood for a point to be considered
        a core point. Default is 2.
    **kwargs: additional keyword arguments for DBSCAN.
    '''
    VECTORS = np.column_stack((X, Y, Z) if Z is not None else (X, Y))

    if kwargs.pop('scale_data', False):
        std = np.std(VECTORS)
        if std > 0:
            VECTORS /= std
            eps /= std  # Scale eps accordingly
        else:
            logger.warning('Standard deviation of data is zero. Skipping scaling.')


    # 2. Initialize and fit DBSCAN
    # eps: maximum distance between two samples
    # min_samples: number of samples in a neighborhood for a point to be a core point
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    dbscan_model.fit(VECTORS)

    # 3. Retrieve labels (-1 indicates noise/outliers)
    labels = dbscan_model.labels_
    unique = np.unique(labels).shape[0] - 1
    logger.info(
        'Cluster labels: %d (%d Points)', unique, labels[labels != -1].shape[0]
    )  # Log only non-noise labels

    return labels
