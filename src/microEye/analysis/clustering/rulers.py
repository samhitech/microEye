import logging
from itertools import combinations
from typing import Optional

import numba as nb
import numpy as np
from numpy.linalg import eigh

from microEye.analysis.clustering.model import FitResult

logger = logging.getLogger(__name__)


def align_ruler_pca(site_coords: np.ndarray):
    '''
    site_coords: (n, 2) array of [x, y] centroids for one ruler.
    Returns (n, 2) rotated + translated to canonical frame.
    For n==2 the axis is computed directly (PCA is ill-conditioned for
    a rank-1 point set and gives an arbitrary minor eigenvector).
    '''
    center = site_coords.mean(axis=0)
    coords_c = site_coords - center

    if len(site_coords) == 2:
        delta = coords_c[1] - coords_c[0]
        angle = np.arctan2(delta[1], delta[0])
    else:
        cov = np.cov(coords_c.T)
        _, eigenvectors = eigh(cov)
        pc1 = eigenvectors[:, -1]
        angle = np.arctan2(pc1[1], pc1[0])

    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    aligned = coords_c @ R.T
    order = np.argsort(aligned[:, 0])
    return aligned[order], R, center, angle


def canonical_flip(aligned_sites: np.ndarray, R: np.ndarray):
    '''Ensure site[0].x < 0; updates R so raw-data transforms stay correct.'''
    if aligned_sites[0, 0] > 0:
        F = np.array([[-1.0, 0.0], [0.0, 1.0]])
        aligned_sites = aligned_sites[::-1] * np.array([-1.0, 1.0])
        R = F @ R
    return aligned_sites, R


def score_n_sites(
    sites: np.ndarray, expected_spacing: float = 80.0, n_sites: int = 3
) -> float:
    if len(sites) != n_sites:
        return np.inf
    aligned, *_ = align_ruler_pca(sites)
    spacings = np.diff(aligned[:, 0])
    if np.any(spacings <= 0):
        return np.inf
    y_dev = np.max(np.abs(aligned[:, 1]))
    spacing_penalty = np.sum((spacings - expected_spacing) ** 2)
    collinear_penalty = y_dev**2
    uniformity_penalty = np.var(spacings)
    return float(spacing_penalty + 0.5 * collinear_penalty + 0.3 * uniformity_penalty)


def best_n_sites_from_candidates(
    site_coords: np.ndarray,
    expected_spacing: float = 80.0,
    n_sites: int = 3,
    spacing_tolerance: float = 25.0,
) -> tuple[Optional[np.ndarray], float, Optional[tuple]]:
    '''
    Enumerate all C(m, n_sites) combinations and return the best-scoring one,
    now including the original indices chosen (needed to build inlier_mask).
    '''
    m = len(site_coords)
    if m < n_sites:
        return None, np.inf, None

    max_score = (n_sites - 1) * spacing_tolerance**2 + 0.5 * 20.0**2
    best_sites, best_score, best_idx = None, np.inf, None

    for combo in combinations(range(m), n_sites):
        candidate = site_coords[list(combo)]
        score = score_n_sites(candidate, expected_spacing, n_sites)
        if score < best_score:
            best_score, best_sites, best_idx = score, candidate, combo

    if best_score > max_score:
        return None, best_score, None
    return best_sites, best_score, best_idx


class RulerModel:
    '''GeometricModel for an N-site collinear DNA-origami nanoruler.'''

    def __init__(
        self,
        n_sites: int,
        expected_spacing: float = 80.0,
        spacing_tolerance: float = 25.0,
        max_y_dev: float = 20.0,
    ):
        self.n_sites = n_sites
        self.expected_spacing = expected_spacing
        self.spacing_tolerance = spacing_tolerance
        self.max_y_dev = max_y_dev

    @property
    def min_points(self) -> int:
        return self.n_sites

    @property
    def span_hint(self) -> float:
        return self.expected_spacing * (self.n_sites - 1)

    def reference_lines(self) -> dict[str, list[float]]:
        '''Expected site positions (nm) in the canonical frame, for overlay drawing.'''
        offsets = (
            np.arange(self.n_sites) - (self.n_sites - 1) / 2.0
        ) * self.expected_spacing
        return {'x': offsets.tolist(), 'y': []}

    def fit(self, site_coords: np.ndarray) -> Optional[FitResult]:
        n = len(site_coords)
        if n < self.n_sites:
            return None

        best_sites, score, combo_idx = best_n_sites_from_candidates(
            site_coords, self.expected_spacing, self.n_sites, self.spacing_tolerance
        )
        if best_sites is None:
            return None

        aligned, R, center, angle = align_ruler_pca(best_sites)
        aligned, R = canonical_flip(aligned, R)

        spacings = np.diff(aligned[:, 0])
        y_dev = np.max(np.abs(aligned[:, 1]))
        if np.any(np.abs(spacings - self.expected_spacing) > self.spacing_tolerance):
            return None
        if y_dev > self.max_y_dev:
            return None

        # Re-express in combo (original) point order so slot indices align
        # with inlier_mask, which is built from combo_idx below.
        aligned_combo_order = (best_sites - center) @ R.T
        slot = np.argsort(np.argsort(aligned_combo_order[:, 0]))

        inlier_mask = np.zeros(n, dtype=bool)
        inlier_mask[list(combo_idx)] = True

        return FitResult(
            R=R,
            origin=center,
            inlier_mask=inlier_mask,
            indices=slot.reshape(-1, 1),
            n_matched=self.n_sites,
            residual=score,
            meta={'angle': angle},
        )

    def get_model_meta(self) -> dict:
        return {
            'type': 'ruler',
            'n_sites': self.n_sites,
            'expected_spacing': self.expected_spacing,
            'spacing_tolerance': self.spacing_tolerance,
            'max_y_dev': self.max_y_dev,
        }

    def get_filename_suffix(self) -> str:
        return f'{self.__class__.__name__}_{self.n_sites}_{self.expected_spacing}'

    @classmethod
    def from_model_meta(cls, meta: dict) -> 'RulerModel':
        return cls(
            n_sites=meta['n_sites'],
            expected_spacing=meta['expected_spacing'],
            spacing_tolerance=meta['spacing_tolerance'],
            max_y_dev=meta['max_y_dev'],
        )
