# COMET: Cost-function Optimized Maximal Overlap Drift Estimation
#
# References
# [1] L. Reinkensmeier, S. Aufmkolk, I. Farabella, A. Egner, and M. Bates,
#     “Cost-function Optimized Maximal Overlap Drift Estimation for
#     Single Molecule Localization Microscopy,” (2026).
#
import gc
import logging
import math
from time import perf_counter_ns

import numba as nb
import numpy as np
import rich.progress as rp
from numba import cuda
from scipy.optimize import minimize
from scipy.spatial import KDTree

from microEye.analysis.drift.core import _compute_frame_drift
from microEye.analysis.drift.segment import adaptive_segmentation

logger = logging.getLogger(__name__)


DRIFT_MAX_BOUND_FACTOR = 2.0  # allow drift to exceed max_drift by this factor

if cuda.is_available():
    import cupy as cp
    import cupyx as cpx

    MEMORY_POOL = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(MEMORY_POOL.malloc)
    cp.cuda.set_pinned_memory_allocator(None)

    MODE = 'cupy'

    def cost_and_grad_cupy(
        d_flat: cp.ndarray,
        x_ij: cp.ndarray,
        si: cp.ndarray,
        sj: cp.ndarray,
        sigma: float,
        n_segments: int,
        ndim: int,
        batch_size=5_000_000,
    ) -> tuple[float, np.ndarray]:
        '''
        Fully vectorized CuPy alternative replacing the Numba implementation.

        Parameters
        ----------
        d_flat : cp.ndarray
            Flattened drift array of shape (n_segments * ndim,).
        x_ij : cp.ndarray
            Array of shape (len(si), ndim) containing coordinate differences
            between neighbor pairs.
        si : cp.ndarray
            Array of shape (len(si),) containing segment indices for the first
            point in each neighbor pair.
        sj : cp.ndarray
            Array of shape (len(sj),) containing segment indices for the second
            point in each neighbor pair.
        sigma : float
            Gaussian length scale for the current optimization round.
        n_segments : int
            Total number of segments.
        ndim : int
            Number of dimensions.
        '''
        d = d_flat.reshape(n_segments, ndim)

        grad = cp.zeros_like(d)

        total_cost = cp.zeros((), dtype=cp.float32)  # 0-d device array

        n_pairs = si.shape[0]
        sigma_sq2 = 2 * (sigma**2)
        inv_sigma_sq = 1.0 / (sigma**2)

        # Process pairs in chunks to bound memory usage
        for start in range(0, n_pairs, batch_size):
            end = min(start + batch_size, n_pairs)

            # Slice only a manageable block of pairs
            si_b = si[start:end]
            sj_b = sj[start:end]
            x_ij_b = x_ij[start:end]

            # Vectorized calculations for this batch only
            delta_b = x_ij_b - (d[si_b] - d[sj_b])
            sq_b = cp.sum(delta_b**2, axis=-1)
            w_b = cp.exp(-sq_b / sigma_sq2)

            # Accumulate scalar cost
            total_cost -= cp.sum(w_b)

            # Compute gradient component for this batch
            c_b = delta_b * (w_b[:, cp.newaxis] * inv_sigma_sq)

            # Scatter add safely to the main gradient matrix
            cpx.scatter_add(grad, si_b, -c_b)
            cpx.scatter_add(grad, sj_b, c_b)

        return float(total_cost), grad.ravel()

    def target_func(d_flat_gpu, *cupy_args):
        cost, grad_gpu = cost_and_grad_cupy(
            cp.asarray(d_flat_gpu, order='C'), *cupy_args
        )
        return cost, grad_gpu.get()
else:
    MODE = 'numba'

    @nb.njit(parallel=True, fastmath=True, cache=True)
    def cost_and_grad_numba(d, x_ij, si, sj, sigma, grad_local, cost_local):
        ndim = d.shape[-1]

        for p in nb.prange(len(si)):
            tid = nb.get_thread_id()
            sq = 0.0
            delta = np.empty(ndim)
            for k in range(ndim):
                delta[k] = x_ij[p, k] - (d[si[p], k] - d[sj[p], k])
                sq += delta[k] * delta[k]
            w = math.exp(-sq / (2 * sigma**2))
            cost_local[tid] -= w
            for k in range(ndim):
                c = delta[k] * w / sigma**2
                grad_local[tid, si[p], k] -= c
                grad_local[tid, sj[p], k] += c

        return cost_local.sum(), grad_local.sum(axis=0)

    def target_func(
        d_flat,
        x_ij,
        si,
        sj,
        sigma,
        n_segments,
        ndim,
        grad_local,
        cost_local,
    ):
        d = d_flat.reshape(n_segments, ndim)
        grad_local.fill(0.0)
        cost_local.fill(0.0)
        cost, grad = cost_and_grad_numba(d, x_ij, si, sj, sigma, grad_local, cost_local)
        return cost, grad.ravel()


def build_segment_ids(n_locs, slices):
    seg_id = np.full(n_locs, -1, dtype=np.int64)  # sentinel, not garbage
    for m, sl in enumerate(slices):
        seg_id[sl] = m
    return seg_id


def build_neighbor_pairs(coords, seg_id, d_max):
    valid = seg_id >= 0
    tree = KDTree(coords[valid])
    pairs = tree.query_pairs(r=d_max, output_type='ndarray')
    if pairs.size == 0:
        raise ValueError('No neighbor pairs found — increase d_max.')
    valid_idx = np.flatnonzero(valid)  # map local KDTree indices back to global
    i, j = valid_idx[pairs[:, 0]], valid_idx[pairs[:, 1]]
    keep = seg_id[i] != seg_id[j]
    return i[keep], j[keep]


def run_comet(
    coords: np.ndarray,
    frames: np.ndarray,
    locs_per_window: int = 250,
    max_drift: float = 300.0,
    sigma_schedule: np.ndarray | None = None,
    downsample: float = 1.0,
    boxcar_width: int = 3,
) -> tuple[np.ndarray, list]:
    '''
    Estimate per-segment drift with COMET.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, ndim) containing coordinates in physical units (e.g. nm).
    frames : np.ndarray
        Array of shape (N,) containing integer frame indices.
    locs_per_window : int, optional
        Target localizations per time segment.
    max_drift : float, optional
        Max expected drift in physical units (e.g. nm);
        also the neighbor-pair cutoff.
    sigma_schedule : np.ndarray, optional
        Gaussian length scales, largest first.
        Defaults to max_drift/3 down to 10 over 5 rounds.
    downsample : float, optional
        Factor by which to downsample the dataset for coarse drift estimation.
    boxcar_width : int, optional
        Width of boxcar smoothing applied to the coarse drift estimate.
        Defaults to 3. To disable smoothing, set to 1.

    Returns
    -------
    tuple
        A tuple containing:
        - `drift`: Estimated per-segment drift, shape (num_segments, ndim).
        - `slices`: Pre-computed contiguous slices for each segment.
    '''
    func_start_time = perf_counter_ns()

    n_locs, ndim = coords.shape
    slices, centers, *_ = adaptive_segmentation(
        frames,
        return_seg_centers=True,
        locs_per_window=locs_per_window,
        downsample=downsample,
    )
    logger.info(f'COMET: {len(slices)} segments, {n_locs:,d} localizations.')
    logger.info(f'Time to segment: {(perf_counter_ns() - func_start_time) / 1e9:.3f} s')

    start_time = perf_counter_ns()
    seg_id = build_segment_ids(n_locs, slices)
    n_segments = len(slices)
    logger.info(f'COMET: {n_segments} segments, {ndim} dimensions.')
    logger.info(
        f'Time to build segment IDs: {(perf_counter_ns() - start_time) / 1e9:.3f} s'
    )

    # filter coords to only those segments with at least one neighbor pair

    start_time = perf_counter_ns()
    pair_i, pair_j = build_neighbor_pairs(coords, seg_id, max_drift)
    logger.debug(f'COMET: {len(pair_i):,d} neighbor pairs, {n_segments} segments.')
    logger.debug(
        f'Time to build neigh-pairs: {(perf_counter_ns() - start_time) / 1e9:.3f} s'
    )

    if sigma_schedule is None:
        sigma_schedule = np.geomspace(max_drift / 3, 10, num=5)

    bounds = [
        (-max_drift * DRIFT_MAX_BOUND_FACTOR, max_drift * DRIFT_MAX_BOUND_FACTOR)
    ] * (ndim * n_segments)

    x_ij = coords[pair_i] - coords[pair_j]  # once, before the sigma loop
    si = seg_id[pair_i].copy()
    sj = seg_id[pair_j].copy()

    if MODE == 'numba':
        drift = np.zeros((n_segments, ndim), dtype=np.float32)
        n_threads = nb.get_num_threads()
        grad_local = np.zeros((n_threads, n_segments, ndim), dtype=np.float32)
        cost_local = np.zeros(n_threads, dtype=np.float32)
    elif MODE == 'cupy':
        drift = np.zeros((n_segments, ndim), dtype=np.float32)
        grad_local = None  # not needed for CuPy implementation
        cost_local = None  # not needed for CuPy implementation

        x_ij = cp.asarray(x_ij, order='C')
        si = cp.asarray(si, order='C')
        sj = cp.asarray(sj, order='C')

    with rp.Progress(
        rp.SpinnerColumn(),  # Animated spinner
        rp.TextColumn('[progress.description]{task.description}'),
        rp.BarColumn(),  # The actual progress bar
        rp.TaskProgressColumn(),  # Percentage complete
        rp.TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            '[cyan]COMET drift estimation...', total=len(sigma_schedule)
        )

        for sigma in sigma_schedule:
            start_time = perf_counter_ns()

            if MODE == 'numba':
                args = (x_ij, si, sj, sigma, n_segments, ndim, grad_local, cost_local)
            elif MODE == 'cupy':
                args = (x_ij, si, sj, sigma, n_segments, ndim)

            result = minimize(
                target_func,
                drift.ravel(),
                args=args,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'gtol': 1e-5, 'ftol': 1e3 * np.finfo(float).eps, 'maxls': 40},
            )

            drift = result.x.reshape(n_segments, ndim)

            if boxcar_width > 1:
                for i in range(ndim):
                    # if MODE == 'numba':
                    drift[:, i] = np.convolve(
                        np.pad(
                            drift[:, i],
                            (boxcar_width // 2, boxcar_width // 2),
                            mode='reflect',
                        ),
                        np.ones(boxcar_width) / boxcar_width,
                        mode='valid',
                    )
                    # elif MODE == 'cupy':
                    #     drift[:, i] = cp.convolve(
                    #         cp.pad(
                    #             drift[:, i],
                    #             (boxcar_width // 2, boxcar_width // 2),
                    #             mode='reflect',
                    #         ),
                    #         cp.ones(boxcar_width) / boxcar_width,
                    #         mode='valid',
                    #     )

            progress.update(task, advance=1)

    logger.info(
        f'Total COMET time: {(perf_counter_ns() - func_start_time) / 1e9:.3f} s'
    )

    drift -= drift[0]  # subtract first segment to anchor drift at zero

    if MODE == 'cupy':
        del x_ij, si, sj  # free GPU memory
        gc.collect()
        MEMORY_POOL.free_all_blocks()  # free GPU memory

    return drift, slices, centers


def comet_drift_correction(
    frames: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray = None,
    **kwargs,
) -> dict:
    '''
    COMET Drift Correction supporting 2D and 3D datasets.

    **References**
    - [1] L. Reinkensmeier, S. Aufmkolk, I. Farabella, A. Egner, and M. Bates,
    “Cost-function Optimized Maximal Overlap Drift Estimation for
    Single Molecule Localization Microscopy,” (2026).

    Parameters
    ----------
    frames : np.ndarray
        Array of shape (N,) containing integer frame indices.
    x : np.ndarray
        Array of shape (N,) containing x-coordinates, in physical units (e.g. nm).
    y : np.ndarray
        Array of shape (N,) containing y-coordinates, in physical units.
    z : np.ndarray, optional
        Array of shape (N,) containing z-coordinates, in physical units.
        Required for 3D datasets. If provided, z-correction is performed.
    max_drift : float, optional
        Maximum expected drift in physical units (e.g. nm). Default is 300.0.
    locs_per_window : int, optional
        Target number of localizations per time segment.
    downsample_factor : int, optional
        Factor by which to downsample the dataset for coarse drift estimation.
    interp_kind : str, optional
        Interpolation method for per-frame drift estimation.
    sigma_start : float, optional
        Starting Gaussian length scale for COMET optimization.
        Default is 40.0.
    sigma_end : float, optional
        Ending Gaussian length scale for COMET optimization.
        Default is 5.0.
    sigma_steps : int, optional
        Number of Gaussian length scales to use in COMET optimization.
        Default is 4.
    boxcar_width : int, optional
        Width of boxcar smoothing applied to the coarse drift estimate.
        Default is 3. To disable smoothing, set to 1.

    **kwargs
        Additional keyword arguments for the drift correction algorithm.

    Returns
    -------
    dict
        Dictionary containing:
        - `block_shifts`: Coarse (num_segs, ndim) drift points.
        - `seg_centers`: X-axis frame indices for plotting coarse points.
        - `frame_full`: Frame indices for the full drift array.
        - `frame_drift`: Continuous (max_frame, ndim) per-frame drift curves.
        - `slices`: Pre-computed contiguous slices.
        - `frame_ids`: Frame IDs for fast slice indexing.
    '''
    coords = np.ascontiguousarray(
        np.column_stack([x, y, z if z is not None else np.zeros_like(x)]),
        dtype=np.float32,
    )

    frames = np.ascontiguousarray(frames, dtype=np.uint64)

    # sort frames and coords together to ensure temporal order
    sort_indices = np.argsort(frames)
    # check if frames are already sorted
    if not np.all(sort_indices == np.arange(len(frames))):
        coords = coords[sort_indices]
        frames = frames[sort_indices]

    # run COMET drift correction
    drift, slices, centers = run_comet(
        coords,
        frames,
        locs_per_window=kwargs.get('locs_per_window', 500),
        max_drift=kwargs.get('max_drift', 300.0),  # nm
        sigma_schedule=np.geomspace(
            kwargs.get('sigma_start', 40),
            kwargs.get('sigma_end', 5),
            num=kwargs.get('sigma_steps', 4),
        ),
        downsample=kwargs.get('downsample_factor', 1.0),
        boxcar_width=kwargs.get('boxcar_width', 3),
    )

    # Interpolate to per-frame drift array (max_frame, ndim)
    frame_ids = np.cumsum(np.bincount(frames.astype(np.int64)))
    frame_full, frame_drift = _compute_frame_drift(
        centers, drift, frames.max(), kind=kwargs.get('interp_kind', 'pchip')
    )

    return {
        # Coarse (num_segs, ndim) drift points
        'block_shifts': drift,
        # X-axis frame indices for plotting coarse points
        'seg_centers': centers,
        # Frame indices for the full drift array
        'frame_full': frame_full,
        # Continuous (max_frame, ndim) per-frame drift curves
        'frame_drift': frame_drift,
        # Pre-computed contiguous slices
        'slices': slices,
        # frame_ids for fast slice indexing
        'frame_ids': frame_ids,
    }


if __name__ == '__main__':
    import os

    import pandas as pd

    data = pd.read_hdf(
        os.environ.get('LOCAL_DATA_PATH_DRIFTED'),
    )

    frames = data['frame'].to_numpy(dtype=np.int64)
    x = data['x'].to_numpy(dtype=np.float64)
    y = data['y'].to_numpy(dtype=np.float64)
    if 'z' in data.columns:
        z = data['z'].to_numpy(dtype=np.float64)
    else:
        z = np.zeros_like(x)

    coords = np.ascontiguousarray(np.column_stack([x, y, z]))

    # drift, slices = run_comet(
    #     coords,
    #     frames,
    #     locs_per_window=5000,  # start here; paper's practical default
    #     max_drift=300.0,  # nm, set to your expected max drift
    #     sigma_schedule=np.geomspace(40, 5, num=4),
    #     downsample=0.5,
    # )

    # # quick sanity plot of the estimated drift vs. segment
    # import matplotlib.pyplot as plt

    # labels = ['x', 'y', 'z']
    # for k in range(drift.shape[1]):
    #     plt.plot(drift[:, k], label=labels[k])
    # plt.xlabel('segment')
    # plt.ylabel('drift (nm)')
    # plt.legend()
    # plt.show()

    # import time

    # import numpy as np

    # # --- benchmark cost_and_grad, isolated from scipy/L-BFGS-B ---
    # slices, centers, *_ = adaptive_segmentation(
    #     frames, return_seg_centers=True, locs_per_window=5000, downsample=0.5
    # )
    # seg_id = build_segment_ids(coords.shape[0], slices)
    # n_segments, ndim = len(slices), coords.shape[1]
    # pair_i, pair_j = build_neighbor_pairs(coords, seg_id, 300.0)

    # x_ij = coords[pair_i] - coords[pair_j]
    # si, sj = seg_id[pair_i].copy(), seg_id[pair_j].copy()
    # d0 = np.zeros(n_segments * ndim, dtype=np.float32)

    # if MODE == 'numba':
    #     n_threads = nb.get_num_threads()
    #     grad_local = np.zeros((n_threads, n_segments, ndim), dtype=np.float32)
    #     cost_local = np.zeros(n_threads, dtype=np.float32)
    #     args = (x_ij, si, sj, 40.0, n_segments, ndim, grad_local, cost_local)
    # elif MODE == 'cupy':
    #     x_ij, si, sj, d0 = (
    #         cp.asarray(x_ij),
    #         cp.asarray(si),
    #         cp.asarray(sj),
    #         cp.asarray(d0),
    #     )
    #     args = (x_ij, si, sj, 40.0, n_segments, ndim, 10_000_000)

    # for _ in range(3):
    #     target_func(d0, *args)  # warmup
    # if MODE == 'cupy':
    #     cp.cuda.Stream.null.synchronize()

    # t0 = perf_counter_ns()
    # for _ in range(20):
    #     target_func(d0, *args)
    # if MODE == 'cupy':
    #     cp.cuda.Stream.null.synchronize()
    # print(f'{MODE}: {(perf_counter_ns() - t0) / 20 / 1e6:.2f} ms/call')
    # # --- end benchmark ---
