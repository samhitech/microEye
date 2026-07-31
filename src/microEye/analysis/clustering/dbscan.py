import logging
from dataclasses import dataclass
from enum import Enum

import numba as nb
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@nb.njit(cache=True)
def geometric_median(points, max_iter=128, tol=1e-6, eps=1e-12):
    guess = np.median(points, axis=0)
    for _ in range(max_iter):
        try:
            deltas = points - guess
            distances = np.linalg.norm(deltas, axis=1)
            weights = 1.0 / np.maximum(distances, eps)
            new_guess = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
            if np.linalg.norm(new_guess - guess) < tol:
                return new_guess
            guess = new_guess
        except Exception as e:
            logger.error(f'Error in geometric_median: {e}')
            break
    return guess


@nb.njit(fastmath=True, cache=True)
def geometric_median_numba(points, max_iter=128, tol=1e-6, eps=1e-12):
    n_points, n_dims = points.shape

    guess = np.empty(n_dims, dtype=points.dtype)
    for d in range(n_dims):
        guess[d] = np.median(points[:, d])

    new_guess = np.empty(n_dims, dtype=points.dtype)

    # Iterative Weiszfeld's algorithm loop
    for _ in range(max_iter):
        total_weight = 0.0

        # Reset new_guess array to zero for current iteration accumulation
        for d in range(n_dims):
            new_guess[d] = 0.0

        # Compute weights and accumulate values using flat loops
        for i in range(n_points):
            dist_sq = 0.0
            for d in range(n_dims):
                diff = points[i, d] - guess[d]
                dist_sq += diff * diff

            dist = np.sqrt(dist_sq)
            if dist < eps:
                dist = eps

            weight = 1.0 / dist
            total_weight += weight

            for d in range(n_dims):
                new_guess[d] += points[i, d] * weight

        if total_weight == 0.0:
            break

        # Finalize the step calculation
        for d in range(n_dims):
            new_guess[d] /= total_weight

        # Check convergence using Euclidean distance
        delta_sq = 0.0
        for d in range(n_dims):
            diff = new_guess[d] - guess[d]
            delta_sq += diff * diff

        if np.sqrt(delta_sq) < tol:
            return new_guess

        # Update current guess manually via value copies
        for d in range(n_dims):
            guess[d] = new_guess[d]

    return guess


@nb.njit(fastmath=True, cache=True)
def huber_center(points, delta=2.0, max_iter=64, tol=1e-5):
    '''
    Calculates the spatial robust center using M-estimation with a Huber loss.
    Eliminates the single-pixel snapping artifact of the geometric median.

    delta: Distance threshold (in nm) where the loss switches from L2 to L1.
           Set this to roughly 1 to 2 times your expected cluster precision.
    '''
    n_points, n_dims = points.shape

    # Safety Check: If empty array or corrupted inputs passed, exit early
    if n_points == 0:
        return np.zeros(n_dims, dtype=points.dtype)

    # Initialize guess with standard mean
    guess = np.empty(n_dims, dtype=points.dtype)
    for d in range(n_dims):
        guess[d] = np.mean(points[:, d])

    new_guess = np.empty(n_dims, dtype=points.dtype)

    for _ in range(max_iter):
        total_weight = 0.0
        for d in range(n_dims):
            new_guess[d] = 0.0

        for i in range(n_points):
            dist_sq = 0.0
            for d in range(n_dims):
                diff = points[i, d] - guess[d]
                dist_sq += diff * diff
            dist = np.sqrt(dist_sq)

            if dist <= delta:
                weight = 1.0
            else:
                # Numba friendly approach: ensure dist can NEVER be zero here
                # since dist > delta and delta > 0, dist is physically > 0,
                # but we add an explicit fallback for safety.
                safe_dist = dist if dist > 1e-12 else 1e-12
                weight = delta / safe_dist

            total_weight += weight
            for d in range(n_dims):
                new_guess[d] += points[i, d] * weight

        # FIXED: Wrap the division completely inside a strict safety check
        # to prevent Numba from pre-optimizing the division pathway.
        if total_weight > 1e-12:
            for d in range(n_dims):
                new_guess[d] /= total_weight
        else:
            # Fallback if weights collapse to zero (e.g. all points are NaN/Inf)
            break

        # Check Convergence
        delta_sq = 0.0
        for d in range(n_dims):
            diff = new_guess[d] - guess[d]
            delta_sq += diff * diff

        if np.sqrt(delta_sq) < tol:
            return new_guess

        for d in range(n_dims):
            guess[d] = new_guess[d]

    return guess


class CenteringMode(Enum):
    CRLB_WEIGHTED_MEAN = 'crlb_weighted_mean'
    MEAN = 'mean'
    GEOMETRIC_MEDIAN = 'geometric_median'
    HUBER_CENTER = 'huber_center'


def get_center(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    center_mode: CenteringMode = CenteringMode.CRLB_WEIGHTED_MEAN,
    crlb_x: np.ndarray | None = None,
    crlb_y: np.ndarray | None = None,
    eps: float = 1e-12,
):
    '''
    Computes the center of clusters based on the specified centering mode.

    *Methods*

    - CRLB_WEIGHTED_MEAN: Computes the center using a weighted mean based on the
    Cramer-Rao Lower Bound (CRLB) variances for x and y coordinates.
    - GEOMETRIC_MEDIAN: Computes the center as the geometric median of the points.
    - HUBER_CENTER: Computes the center using M-estimation with a Huber loss function.
    - MEAN: Computes the center as the arithmetic mean of the points.

    Parameters
    -----------
    x, y, z: np.ndarray
        Arrays of x, y, z coordinates of localizations.
    center_mode: CenteringMode
        The method to compute the center of clusters.
    crlb_x, crlb_y: np.ndarray, optional
        Arrays of CRLB variance values for x and y coordinates.
        Required if center_mode is CenteringMode.CRLB_WEIGHTED_MEAN.
    eps: float
        A small value to avoid division by zero when computing weights.
        (default: 1e-12)

    Returns
    --------
    mu_x, mu_y, mu_z: np.ndarray
    '''
    if (
        center_mode == CenteringMode.CRLB_WEIGHTED_MEAN
        and crlb_x is not None
        and crlb_y is not None
    ):
        wx = 1.0 / np.maximum(crlb_x, eps)
        wy = 1.0 / np.maximum(crlb_y, eps)
        mu_x = np.sum(wx * x) / np.sum(wx)
        mu_y = np.sum(wy * y) / np.sum(wy)
        mu_z = np.mean(z) if z is not None else None

        # 1. Calculate variance (squared error) for X and Y independently
        var_x_com = 1.0 / np.sum(1.0 / np.maximum(crlb_x, eps))
        var_y_com = 1.0 / np.sum(1.0 / np.maximum(crlb_y, eps))

        # 2. Combine them in quadrature to get true 2D radial uncertainty
        sigma_com = np.sqrt(var_x_com + var_y_com)

    elif center_mode == CenteringMode.GEOMETRIC_MEDIAN:
        if z is not None:
            mu_x, mu_y, mu_z = geometric_median_numba(np.column_stack((x, y, z)))
        else:
            mu_x, mu_y = geometric_median_numba(np.column_stack((x, y)))
            mu_z = None

        var_x = np.mean((x - mu_x) ** 2)
        var_y = np.mean((y - mu_y) ** 2)

        # sqrt(pi/2) factor converts from 1D std to 2D radial std
        sigma_com = 1.253 * np.sqrt(var_x + var_y) / np.sqrt(len(x))

    elif center_mode == CenteringMode.HUBER_CENTER:
        if z is not None:
            mu_x, mu_y, mu_z = huber_center(np.column_stack((x, y, z)))
        else:
            mu_x, mu_y = huber_center(np.column_stack((x, y)))
            mu_z = None

        # Median Absolute Deviation (MAD)
        mad_x = 1.4826 * np.median(np.abs(x - mu_x)) / np.sqrt(len(x))
        mad_y = 1.4826 * np.median(np.abs(y - mu_y)) / np.sqrt(len(y))

        sigma_com = np.sqrt(mad_x**2 + mad_y**2)

    else:
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        mu_z = np.mean(z) if z is not None else None

        var_x = np.mean((x - mu_x) ** 2)
        var_y = np.mean((y - mu_y) ** 2)

        # Standard error of the mean
        sigma_com = np.sqrt(var_x + var_y) / np.sqrt(len(x))

    return mu_x, mu_y, mu_z, sigma_com


@dataclass
class ClusterCenterResult:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    labels: np.ndarray
    intensity: np.ndarray | None
    crlb_x: np.ndarray | None
    crlb_y: np.ndarray | None
    cx: np.ndarray
    cy: np.ndarray
    cz: np.ndarray | None
    c_sigma: np.ndarray | None
    c_label: np.ndarray
    n_points: np.ndarray
    rejected: int


def get_centers(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray | None,
    labels: np.ndarray,
    intensity: np.ndarray | None = None,
    crlb_x: np.ndarray | None = None,
    crlb_y: np.ndarray | None = None,
    center_mode: CenteringMode = CenteringMode.CRLB_WEIGHTED_MEAN,
    eps: float = 1e-12,
    rejection_threshold: float | None = None,
    min_points: int | None = None,
    max_points: int | None = None,
    subtract_com: bool = False,
    **kwargs,
) -> ClusterCenterResult | None:
    try:
        # Filter out noise (label < 0) safely
        valid_mask = labels >= 0
        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask] if z is not None else None
        intensity = intensity[valid_mask] if intensity is not None else None
        labels = labels[valid_mask]

        if crlb_x is not None:
            crlb_x = np.asarray(crlb_x, dtype=np.float64)[valid_mask]
        if crlb_y is not None:
            crlb_y = np.asarray(crlb_y, dtype=np.float64)[valid_mask]

        if len(labels) == 0:
            return None

        _arg_sort = np.argsort(labels)
        x = x[_arg_sort]
        y = y[_arg_sort]
        z = z[_arg_sort] if z is not None else None
        intensity = intensity[_arg_sort] if intensity is not None else None
        if crlb_x is not None:
            crlb_x = crlb_x[_arg_sort]
        if crlb_y is not None:
            crlb_y = crlb_y[_arg_sort]

        labels = labels[_arg_sort]

        label_ids = np.cumsum(np.bincount(labels.astype(np.int64)))

        unique_labels = np.unique(labels.astype(np.int64))

        # array to hold the computed centers for each cluster
        cx = []
        cy = []
        cz = [] if z is not None else None
        c_sigma = []
        c_labels = []
        n_points = []

        rejected = []

        for i in unique_labels:
            s = slice(label_ids[i - 1] if i > 0 else 0, label_ids[i])

            if len(x[s]) == 0:
                logger.debug('Cluster %d has no points. Skipping.', i)
                continue

            x_cluster = x[s]
            y_cluster = y[s]
            z_cluster = z[s] if z is not None else None

            mu_x, mu_y, mu_z, sigma_com = get_center(
                x_cluster,
                y_cluster,
                z_cluster,
                center_mode=center_mode,
                crlb_x=crlb_x[s] if crlb_x is not None else None,
                crlb_y=crlb_y[s] if crlb_y is not None else None,
                eps=eps,
            )

            # Reject clusters with large uncertainty
            if (
                (rejection_threshold is not None and sigma_com > rejection_threshold)
                or (min_points is not None and len(x_cluster) < min_points)
                or (max_points is not None and len(x_cluster) > max_points)
            ):
                rejected.append(i)
                continue

            cx.append(mu_x)
            cy.append(mu_y)
            if cz is not None:
                cz.append(mu_z)
            c_sigma.append(sigma_com)
            c_labels.append(i)
            n_points.append(len(x_cluster))

            if subtract_com:
                x[s] -= mu_x
                y[s] -= mu_y
                if z is not None:
                    z[s] -= mu_z

        if len(rejected) > 0:
            logger.warning(
                'Rejected %d clusters due to uncertainty or insufficient points.',
                len(rejected),
            )

            valid_mask = np.isin(labels, rejected, invert=True)
            x = x[valid_mask]
            y = y[valid_mask]
            intensity = intensity[valid_mask] if intensity is not None else None
            labels = labels[valid_mask]
            crlb_x = crlb_x[valid_mask] if crlb_x is not None else None
            crlb_y = crlb_y[valid_mask] if crlb_y is not None else None
            if z is not None:
                z = z[valid_mask]

        return ClusterCenterResult(
            x=x,
            y=y,
            z=z,
            labels=labels,
            intensity=intensity,
            crlb_x=crlb_x,
            crlb_y=crlb_y,
            cx=np.array(cx),
            cy=np.array(cy),
            cz=np.array(cz) if cz is not None else None,
            c_sigma=np.array(c_sigma),
            c_label=np.array(c_labels),
            n_points=np.array(n_points),
            rejected=len(rejected),
        )
    except Exception as e:
        logger.error(f'Error in get_centers: {e}')
        return None


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
