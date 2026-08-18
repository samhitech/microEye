import logging
import traceback
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numba as nb
import numpy as np
from numpy.linalg import eigh

from microEye.analysis.clustering.dbscan import (
    OPTICS,
    CenteringMode,
    ClusterCenterResult,
    get_center,
    get_centers,
    perform_dbscan_clustering,
    perform_hdbscan_clustering,
)

logger = logging.getLogger(__name__)


@dataclass
class RulerCandidate:
    ruler_id: int
    raw_sites: np.ndarray  # (N, 2) all centroids found
    angle: float | None = None
    best_triplet: np.ndarray | None = None  # (3, 2) chosen sites
    best_score: float = np.inf
    aligned_sites: np.ndarray | None = None  # (3, 2) after PCA
    R: np.ndarray | None = None
    center: np.ndarray | None = None
    passed_qc: bool = False
    reject_reason: str = ''


def align_ruler_pca(site_coords: np.ndarray):
    '''
    site_coords: (n, 2) array of [x, y] centroids for one ruler.
    Returns (n, 2) rotated + translated to canonical frame:
      - centroid at origin
      - long axis along X
      - sites ordered left to right

    For n==2 the axis is computed directly (PCA is ill-conditioned for
    a rank-1 point set and gives an arbitrary minor eigenvector).
    '''
    center = site_coords.mean(axis=0)
    coords_c = site_coords - center

    if len(site_coords) == 2:
        # Direct vector between the two points — exact, no PCA needed
        delta = coords_c[1] - coords_c[0]
        angle = np.arctan2(delta[1], delta[0])
    else:
        cov = np.cov(coords_c.T)
        eigenvalues, eigenvectors = eigh(cov)
        pc1 = eigenvectors[:, -1]
        angle = np.arctan2(pc1[1], pc1[0])

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    aligned = coords_c @ R.T

    order = np.argsort(aligned[:, 0])

    return aligned[order], R, center, angle


def canonical_flip(
    aligned_sites: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Ensure site[0].x < 0 (left site is always negative X).
    When a flip is needed, also updates R so that raw localizations
    transformed with the returned R land in the correct frame.

    Returns (aligned_sites, R) — R may be modified.
    '''
    if aligned_sites[0, 0] > 0:
        # Flip matrix: negate X, keep Y
        F = np.array([[-1.0, 0.0], [0.0, 1.0]])
        aligned_sites = (aligned_sites[::-1]) * np.array([-1.0, 1.0])
        R = F @ R  # compose: first rotate, then flip
    return aligned_sites, R


def score_n_sites(
    sites: np.ndarray,
    expected_spacing: float = 80.0,
    n_sites: int = 3,
) -> float:
    '''
    Score a candidate (n_sites, 2) set. Lower = better.
    Generalizes score_triplet to any N collinear sites.
    '''
    if len(sites) != n_sites:
        return np.inf

    aligned, *_ = align_ruler_pca(sites)

    # All consecutive spacings along X
    spacings = np.diff(aligned[:, 0])

    # Reject if any gap is non-positive (sites out of order or overlapping)
    if np.any(spacings <= 0):
        return np.inf

    # Off-axis spread
    y_dev = np.max(np.abs(aligned[:, 1]))

    spacing_penalty = np.sum((spacings - expected_spacing) ** 2)
    collinear_penalty = y_dev**2
    # Variance of spacings penalises uneven gaps (generalises asymmetry_penalty)
    uniformity_penalty = np.var(spacings)

    return float(spacing_penalty + 0.5 * collinear_penalty + 0.3 * uniformity_penalty)


def best_n_sites_from_candidates(
    site_coords: np.ndarray,
    expected_spacing: float = 80.0,
    n_sites: int = 3,
    spacing_tolerance: float = 25.0,
) -> tuple[Optional[np.ndarray], float]:
    '''
    Given M >= n_sites centroids, enumerate all C(M, n_sites) combinations
    and return the best-scoring one.
    '''
    m = len(site_coords)
    if m < n_sites:
        return None, np.inf

    # Max score budget: each of (n_sites-1) gaps can deviate by spacing_tolerance,
    # plus the collinear budget. Scales naturally with n_sites.
    max_score = (n_sites - 1) * spacing_tolerance**2 + 0.5 * 20.0**2

    best_sites, best_score = None, np.inf

    for combo in combinations(range(m), n_sites):
        candidate = site_coords[list(combo)]
        score = score_n_sites(candidate, expected_spacing, n_sites)
        if score < best_score:
            best_score = score
            best_sites = candidate

    if best_score > max_score:
        return None, best_score

    return best_sites, best_score


def ruler_qc_smart(
    candidate: RulerCandidate,
    expected_spacing: float = 80.0,
    spacing_tolerance: float = 25.0,
    max_y_dev: float = 20.0,
    n_sites: int = 3,  # <-- new param, defaults to 3 for backward compat
) -> RulerCandidate:
    n = len(candidate.raw_sites)

    if n < n_sites:
        candidate.reject_reason = f'too_few_sites (n={n}, need={n_sites})'
        return candidate

    best_sites, score = best_n_sites_from_candidates(
        candidate.raw_sites, expected_spacing, n_sites, spacing_tolerance
    )
    candidate.best_score = score

    if best_sites is None:
        candidate.reject_reason = (
            f'no_{n_sites}_site_combo_fits (n={n}, best_score={score:.1f})'
        )
        return candidate

    aligned, R, center, angle = align_ruler_pca(best_sites)
    aligned, R = canonical_flip(aligned, R)

    candidate.best_triplet = best_sites  # name kept for compat; holds n_sites sites
    candidate.aligned_sites = aligned
    candidate.R = R
    candidate.center = center
    candidate.angle = angle

    # Check every consecutive spacing
    spacings = np.diff(aligned[:, 0])
    y_dev = np.max(np.abs(aligned[:, 1]))

    issues = []
    for i, sp in enumerate(spacings):
        if abs(sp - expected_spacing) > spacing_tolerance:
            issues.append(f'd{i + 1}={sp:.1f}nm off by {sp - expected_spacing:+.1f}')
    if y_dev > max_y_dev:
        issues.append(f'y_dev={y_dev:.1f}nm > {max_y_dev}')

    if issues:
        candidate.reject_reason = '; '.join(issues)
    else:
        candidate.passed_qc = True

    return candidate


def group_sites_into_rulers(
    cx: np.ndarray,
    cy: np.ndarray,
    cz: np.ndarray | None,
    n_sites: int = 3,
    method: str = 'dbscan',  # 'dbscan' or 'optics'
    distance_threshold: float = 180.0,
    xi: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    Groups site centroids into candidate rulers.

    Parameters
    -----------
    cx, cy, cz : np.ndarray
        Arrays of x, y, (optional) z coordinates of site centroids.
    n_sites : int
        Minimum number of sites required to form a ruler.
    method : str
        Clustering method to use: 'dbscan' or 'optics'.
    distance_threshold : float
        Maximum distance between sites to be considered part of the same ruler.
        (Ignored if method is 'optics')
    xi : float
        The xi parameter for OPTICS clustering. (ignored if method is 'dbscan')

    Returns
    --------
    ruler_id : np.ndarray
        Array of the same length as cx/cy/cz, where each element is the ID
        of the ruler that the corresponding site belongs to.
        Noise points are labeled as -1.
    """
    coords = np.column_stack((cx, cy) if cz is None else (cx, cy, cz))

    if method == 'optics':
        optics = OPTICS(min_samples=n_sites, xi=xi, cluster_method='xi')
        optics.fit(coords)
        ruler_id = optics.labels_
    elif method == 'hdbscan':
        ruler_id = perform_hdbscan_clustering(
            cx,
            cy,
            cz,
            min_cluster_size=n_sites,
            cluster_selection_epsilon_max=distance_threshold,
        )
    else:
        ruler_id = perform_dbscan_clustering(
            cx,
            cy,
            cz,
            eps=distance_threshold,
            min_samples=n_sites,
            **kwargs,
        )

    return ruler_id


def get_rulers_from_clusters(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    intensity: np.ndarray | None = None,
    crlb_x: np.ndarray | None = None,
    crlb_y: np.ndarray | None = None,
    **kwargs,
):
    n_sites: int = kwargs.get('n_sites', 3)
    distance_threshold: float = kwargs.get('distance_threshold', 180.0)
    expected_spacing: float = kwargs.get('expected_spacing', 80.0)
    spacing_tolerance: float = kwargs.get('spacing_tolerance', 25.0)
    max_y_dev: float = kwargs.get('max_y_dev', 20.0)
    center_mode: CenteringMode = kwargs.get(
        'center_mode', CenteringMode.CRLB_WEIGHTED_MEAN
    )
    eps: float = kwargs.get('eps', 1e-12)
    scale_data: bool = kwargs.get('scale_data', False)
    leaf_size: int = kwargs.get('leaf_size', 30)
    algorithm: str = kwargs.get('algorithm', 'auto')

    n_localizations = x.size

    centers_result = get_centers(
        x,
        y,
        z,
        labels,
        intensity,
        crlb_x=crlb_x,
        crlb_y=crlb_y,
        center_mode=center_mode,
        eps=eps,
        min_points=n_sites,
        subtract_com=False,
    )

    if centers_result is None or not isinstance(centers_result, ClusterCenterResult):
        raise ValueError('get_centers did not return a valid ClusterCenterResult.')

    ruler_id = group_sites_into_rulers(
        centers_result.cx,
        centers_result.cy,
        centers_result.cz,
        n_sites=n_sites,
        method=kwargs.get('grouping_method', 'dbscan'),
        distance_threshold=distance_threshold,
        xi=kwargs.get('xi', 0.05),
        scale_data=scale_data,
        leaf_size=leaf_size,
        algorithm=algorithm,
    )

    logger.info(
        f'Found {np.unique(ruler_id).size - 1} potential rulers. '
        f'From {n_localizations} localizations, '
        f'{centers_result.cx.size} cluster centers were identified.'
    )

    ruler_aligned = {}
    for ruler_label in np.unique(ruler_id):
        if ruler_label == -1:
            continue  # Skip noise points

        mask = ruler_id == ruler_label
        site_coords = np.column_stack(
            (centers_result.cx[mask], centers_result.cy[mask])
        )

        candidate = ruler_qc_smart(
            RulerCandidate(ruler_id=ruler_label, raw_sites=site_coords),
            expected_spacing,
            spacing_tolerance,
            max_y_dev,
            n_sites=n_sites,  # now forwarded
        )

        if not candidate.passed_qc:
            logger.debug('Ruler %d failed QC: %s', ruler_label, candidate.reject_reason)
            continue

        raw_points_labels = centers_result.c_label[mask]
        raw_points_mask = np.isin(centers_result.labels, raw_points_labels)
        xy_raw = np.column_stack(
            (centers_result.x[raw_points_mask], centers_result.y[raw_points_mask])
        )
        xy_aligned = (xy_raw - candidate.center) @ candidate.R.T

        ruler_aligned[ruler_label] = {
            'aligned_sites': candidate.aligned_sites,
            'R': candidate.R,
            'center': candidate.center,
            'angle': candidate.angle,
            'x': xy_aligned[:, 0],
            'y': xy_aligned[:, 1],
            'z': centers_result.z[raw_points_mask]
            if centers_result.z is not None
            else None,
            'intensity': centers_result.intensity[raw_points_mask]
            if centers_result.intensity is not None
            else None,
            'labels': centers_result.labels[raw_points_mask],
            'crlb_x': centers_result.crlb_x[raw_points_mask]
            if centers_result.crlb_x is not None
            else None,
            'crlb_y': centers_result.crlb_y[raw_points_mask]
            if centers_result.crlb_y is not None
            else None,
        }

    logger.info(
        f'Rulers passing QC: {len(ruler_aligned)} / {np.unique(ruler_id).size - 1}.'
    )

    return ruler_aligned


def rulers_concatenate(ruler_aligned: dict[int, dict]) -> dict[str, np.ndarray]:
    concatenated = {
        'aligned_sites': [],
        'R': [],
        'center': [],
        'angle': [],
        'x': [],
        'y': [],
        'z': [],
        'intensity': [],
        'labels': [],
        'crlb_x': [],
        'crlb_y': [],
    }

    for ruler_data in ruler_aligned.values():
        for key in concatenated:
            if key in ruler_data and ruler_data[key] is not None:
                concatenated[key].append(ruler_data[key])

    for key in concatenated:
        if not concatenated[key]:
            concatenated[key] = None
            continue

        first_item = np.asarray(concatenated[key][0])
        if first_item.ndim == 0:
            concatenated[key] = np.asarray(concatenated[key])
        else:
            concatenated[key] = np.concatenate(concatenated[key], axis=0)

    logger.info(
        'Concatenated rulers: %d rulers, %d total points.',
        len(ruler_aligned),
        concatenated['x'].size if concatenated['x'] is not None else 0,
    )

    return concatenated
