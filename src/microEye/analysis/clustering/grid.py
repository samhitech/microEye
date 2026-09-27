import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from microEye.analysis.clustering.model import FitResult

logger = logging.getLogger(__name__)


def circular_mean_offset(coord_1d: np.ndarray, d: float) -> tuple[float, float]:
    '''Robust phase offset of a 1D periodic signal (spacing d) with unknown origin.'''
    phi = 2 * np.pi * (coord_1d % d) / d
    C, S = np.mean(np.cos(phi)), np.mean(np.sin(phi))
    R = np.hypot(C, S)
    offset = (np.arctan2(S, C) / (2 * np.pi)) * d
    return offset, 1.0 - R


def lattice_snap_residual(theta: float, coords_c: np.ndarray, d: float) -> float:
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    Rm = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = coords_c @ Rm.T
    _, res_u = circular_mean_offset(rotated[:, 0], d)
    _, res_v = circular_mean_offset(rotated[:, 1], d)
    return res_u + res_v


def fit_lattice_orientation(
    coords_c: np.ndarray, d: float, coarse_steps: int = 181
) -> float:
    '''
    Search space is [0, pi/2) since the infinite square lattice repeats every 90deg.
    '''
    thetas = np.linspace(0, np.pi / 2, coarse_steps, endpoint=False)
    residuals = [lattice_snap_residual(t, coords_c, d) for t in thetas]
    best_coarse = thetas[int(np.argmin(residuals))]
    result = minimize_scalar(
        lattice_snap_residual,
        args=(coords_c, d),
        bounds=(best_coarse - np.pi / 90, best_coarse + np.pi / 90),
        method='bounded',
    )
    return float(result.x)


class GridModel:
    '''GeometricModel for an M x N square-pitch DNA-origami docking grid.'''

    def __init__(
        self,
        M: int,
        N: int,
        pitch: float = 20.0,
        min_occupied: int = 4,
        snap_residual_max: float = 6.0,
        min_inlier_fraction: float = 0.8,
    ):
        self.M, self.N, self.pitch = M, N, pitch
        self.min_occupied = min_occupied
        self.snap_residual_max = snap_residual_max
        self.min_inlier_fraction = min_inlier_fraction

    @property
    def min_points(self) -> int:
        return self.min_occupied

    @property
    def span_hint(self) -> float:
        return self.pitch * np.hypot(self.M - 1, self.N - 1)

    def reference_lines(self) -> dict[str, list[float]]:
        '''Expected lattice node positions (nm) along each axis, for overlay drawing.'''
        x_offsets = (np.arange(self.N) - (self.N - 1) / 2.0) * self.pitch
        y_offsets = (np.arange(self.M) - (self.M - 1) / 2.0) * self.pitch
        return {'x': x_offsets.tolist(), 'y': y_offsets.tolist()}

    def fit(self, site_coords: np.ndarray) -> Optional[FitResult]:
        n = len(site_coords)
        if n < self.min_occupied:
            return None

        center = site_coords.mean(axis=0)
        coords_c = site_coords - center
        theta = fit_lattice_orientation(coords_c, self.pitch)
        cos_a, sin_a = np.cos(theta), np.sin(theta)
        Rm = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = coords_c @ Rm.T

        offset_u, _ = circular_mean_offset(rotated[:, 0], self.pitch)
        offset_v, _ = circular_mean_offset(rotated[:, 1], self.pitch)
        phased = rotated - np.array([offset_u, offset_v])

        raw_idx_u = np.round(phased[:, 0] / self.pitch).astype(int)
        raw_idx_v = np.round(phased[:, 1] / self.pitch).astype(int)
        snap_err = np.hypot(
            phased[:, 0] - raw_idx_u * self.pitch,
            phased[:, 1] - raw_idx_v * self.pitch,
        )

        best = None
        # Try both role assignments (which axis is rows vs cols)
        for idx_col_raw, idx_row_raw, n_try, m_try, u_is_col in [
            (raw_idx_u, raw_idx_v, self.N, self.M, True),
            (raw_idx_v, raw_idx_u, self.M, self.N, False),
        ]:
            idx_row = idx_row_raw - idx_row_raw.min()
            idx_col = idx_col_raw - idx_col_raw.min()
            inliers = (
                (snap_err <= self.snap_residual_max)
                & (idx_row < m_try)
                & (idx_col < n_try)
            )
            if inliers.sum() == 0:
                continue

            rms = float(np.sqrt(np.mean(snap_err[inliers] ** 2)))
            n_occupied = len(
                set(zip(idx_row[inliers].tolist(), idx_col[inliers].tolist()))
            )

            if u_is_col:
                row_off = offset_v + idx_row_raw.min() * self.pitch
                col_off = offset_u + idx_col_raw.min() * self.pitch
                Rrole = Rm
            else:
                row_off = offset_u + idx_row_raw.min() * self.pitch
                col_off = offset_v + idx_col_raw.min() * self.pitch
                Rrole = Rm[::-1].copy()

            grid_center = np.array(
                [
                    (n_try - 1) * self.pitch / 2,
                    (m_try - 1) * self.pitch / 2,
                ]
            )

            origin = center + Rrole.T @ (np.array([col_off, row_off]) + grid_center)

            candidate = dict(
                R=Rrole,
                origin=origin,
                inlier_mask=inliers,
                indices=np.column_stack([idx_row[inliers], idx_col[inliers]]),
                n_matched=n_occupied,
                residual=rms,
                meta={'M': m_try, 'N': n_try, 'angle': theta},
            )
            if best is None or rms < best['residual']:
                best = candidate

        if best is None:
            return None

        inlier_fraction = best['inlier_mask'].sum() / n if n else 0.0
        if best['n_matched'] < self.min_occupied:
            return None
        if best['residual'] > self.snap_residual_max:
            return None
        if inlier_fraction < self.min_inlier_fraction:
            return None

        return FitResult(**best)

    def get_model_meta(self) -> dict:
        return {
            'type': 'grid',
            'M': self.M,
            'N': self.N,
            'pitch': self.pitch,
            'min_occupied': self.min_occupied,
            'snap_residual_max': self.snap_residual_max,
            'min_inlier_fraction': self.min_inlier_fraction,
        }

    def get_filename_suffix(self) -> str:
        return f'{self.__class__.__name__}_{self.M}_{self.N}_{self.pitch}'

    @classmethod
    def from_model_meta(cls, meta: dict) -> 'GridModel':
        return cls(
            M=meta['M'],
            N=meta['N'],
            pitch=meta['pitch'],
            min_occupied=meta.get('min_occupied', 4),
            snap_residual_max=meta.get('snap_residual_max', 6.0),
            min_inlier_fraction=meta.get('min_inlier_fraction', 0.8),
        )
