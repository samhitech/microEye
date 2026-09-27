from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

import numpy as np

from microEye.analysis.clustering.core import (
    ClusterCenterResult,
    get_centers,
    perform_dbscan_clustering,
    perform_hdbscan_clustering,
    perform_optics_clustering,
)
from microEye.analysis.clustering.dataclasses import (
    CenteringMode,
    CenteringOptions,
    ClusteringOptions,
    FitResult,
    InstanceStats,
)


def combine_stats(stats_list: list[InstanceStats]) -> InstanceStats:
    '''Merge per-table InstanceStats into one, for the aggregate display case.'''
    combined_log = []
    for t_idx, s in enumerate(stats_list):
        for row in s.candidate_log:
            combined_log.append({**row, 'table': t_idx})
    return InstanceStats(
        n_localizations=sum(s.n_localizations for s in stats_list),
        n_input_clusters=sum(s.n_input_clusters for s in stats_list),
        n_input_cluster_localizations=sum(
            s.n_input_cluster_localizations for s in stats_list
        ),
        n_cluster_centers=sum(s.n_cluster_centers for s in stats_list),
        n_rejected_centers=sum(s.n_rejected_centers for s in stats_list),
        n_candidates=sum(s.n_candidates for s in stats_list),
        n_candidate_centers=sum(s.n_candidate_centers for s in stats_list),
        n_rejected_candidates=sum(s.n_rejected_candidates for s in stats_list),
        n_passed_instances=sum(s.n_passed_instances for s in stats_list),
        n_passed_centers=sum(s.n_passed_centers for s in stats_list),
        n_passed_localizations=sum(s.n_passed_localizations for s in stats_list),
        candidate_log=combined_log,
    )


class GeometricModel(Protocol):
    min_points: int
    '''
    The minimum number of points required to form a valid instance of this model.
    '''
    span_hint: float
    '''
    A rough estimate of the maximum distance between any two points and used to set
    the distance_threshold for clustering.'''

    def fit(self, site_coords: np.ndarray) -> Optional[FitResult]:
        """None if fitting isn't possible (too few/degenerate points)."""
        ...

    def get_model_meta(self) -> dict:
        '''Return a dict of model parameters for logging and saving.'''
        ...

    def get_filename_suffix(self) -> str:
        '''Return a string to append to filenames when saving results for this model.'''
        ...

    def reference_lines(self) -> Optional[dict[str, list[float]]]:
        """
        Return a dict of reference lines for overlay drawing, or None if not applicable.

        The dict keys are 'x' and 'y', and the values are lists of positions in nm.
        """
        ...


def group_sites_into_instances(
    cx,
    cy,
    cz,
    model: GeometricModel,
    clustering_options: ClusteringOptions,
):
    if clustering_options.grouping_method not in {'dbscan', 'hdbscan', 'optics'}:
        raise ValueError(
            f'Unknown grouping method: {clustering_options.grouping_method}'
        )

    distance_threshold = clustering_options.distance_threshold
    if distance_threshold is None or distance_threshold <= 0:
        distance_threshold = model.span_hint * 1.15

    if clustering_options.grouping_method == 'optics':
        return perform_optics_clustering(
            cx,
            cy,
            cz,
            min_samples=model.min_points,
            xi=clustering_options.xi,
            scale_data=clustering_options.scale_data,
            sigma_xy=clustering_options.sigma_xy,
            sigma_z=clustering_options.sigma_z,
        )
    if clustering_options.grouping_method == 'hdbscan':
        return perform_hdbscan_clustering(
            cx,
            cy,
            cz,
            min_cluster_size=model.min_points,
            cluster_selection_epsilon_max=distance_threshold,
            scale_data=clustering_options.scale_data,
            sigma_xy=clustering_options.sigma_xy,
            sigma_z=clustering_options.sigma_z,
        )

    return perform_dbscan_clustering(
        cx,
        cy,
        cz,
        eps=float(distance_threshold),
        min_samples=model.min_points,
        scale_data=clustering_options.scale_data,
        sigma_xy=clustering_options.sigma_xy,
        sigma_z=clustering_options.sigma_z,
    )


def get_instances_from_clusters(
    x,
    y,
    z,
    labels,
    model: GeometricModel,
    intensity=None,
    crlb_x=None,
    crlb_y=None,
    max_residual=None,
    centering_options: CenteringOptions | None = None,
    clustering_options: ClusteringOptions | None = None,
):
    if centering_options is None:
        centering_options = CenteringOptions(
            center_mode=CenteringMode.CRLB_WEIGHTED_MEAN,
            min_points=model.min_points,
        )

    if clustering_options is None:
        clustering_options = ClusteringOptions()

    labels = np.asarray(labels)
    n_localizations = int(x.size)
    input_cluster_mask = labels >= 0
    n_input_clusters = int(np.unique(labels[input_cluster_mask]).size)
    n_input_cluster_localizations = int(input_cluster_mask.sum())

    centers = get_centers(
        x,
        y,
        z,
        labels,
        intensity,
        crlb_x=crlb_x,
        crlb_y=crlb_y,
        options=centering_options,
    )

    if centers is None or not isinstance(centers, ClusterCenterResult):
        raise ValueError('get_centers failed.')

    instance_id = group_sites_into_instances(
        centers.cx, centers.cy, centers.cz, model, clustering_options
    )

    candidates = [label for label in np.unique(instance_id) if label != -1]

    aligned, candidate_log = {}, []

    n_candidate_centers = 0
    n_rejected_candidates = 0
    n_passed_instances = 0
    n_passed_centers = 0
    n_passed_localizations = 0

    for label in candidates:
        candidate_mask = instance_id == label
        n_candidate_centers += int(candidate_mask.sum())

        site_coords = np.column_stack(
            (centers.cx[candidate_mask], centers.cy[candidate_mask])
        )
        fit = model.fit(site_coords)

        residual_rejected = (
            fit is not None and max_residual is not None and fit.residual > max_residual
        )
        passed = fit is not None and not residual_rejected

        row = {
            'candidate_id': int(label),
            'n_sites': int(candidate_mask.sum()),
            'passed': passed,
            'residual': float(fit.residual) if fit is not None else None,
        }

        if not passed:
            n_rejected_candidates += 1
            row['rejection_reason'] = (
                'max_residual' if residual_rejected else 'model_fit_failed'
            )
            candidate_log.append(row)
            continue

        candidate_labels = centers.c_label[candidate_mask]
        inlier_labels = candidate_labels[fit.inlier_mask]
        raw_mask = np.isin(centers.labels, inlier_labels)

        xy_raw = np.column_stack((centers.x[raw_mask], centers.y[raw_mask]))
        xy_aligned = (xy_raw - fit.origin) @ fit.R.T

        candidate_localizations = int(centers.n_points[candidate_mask].sum())
        passed_localizations = int(raw_mask.sum())
        passed_centers = int(fit.inlier_mask.sum())

        n_passed_instances += 1
        n_passed_centers += passed_centers
        n_passed_localizations += passed_localizations

        row.update(
            {
                'n_matched': int(fit.n_matched),
                'n_inlier_centers': passed_centers,
                'n_candidate_localizations': candidate_localizations,
                'n_passed_localizations': passed_localizations,
            }
        )
        candidate_log.append(row)

        aligned[label] = {
            'indices': fit.indices,
            'n_matched': fit.n_matched,
            'residual': fit.residual,
            'R': fit.R,
            'origin': fit.origin,
            'x': xy_aligned[:, 0],
            'y': xy_aligned[:, 1],
            'z': centers.z[raw_mask] if centers.z is not None else None,
            'intensity': (
                centers.intensity[raw_mask] if centers.intensity is not None else None
            ),
            'labels': centers.labels[raw_mask],
            'crlb_x': (
                centers.crlb_x[raw_mask] if centers.crlb_x is not None else None
            ),
            'crlb_y': (
                centers.crlb_y[raw_mask] if centers.crlb_y is not None else None
            ),
        }
        aligned[label].update(fit.meta)

    stats = InstanceStats(
        n_localizations=n_localizations,
        n_input_clusters=n_input_clusters,
        n_input_cluster_localizations=n_input_cluster_localizations,
        n_cluster_centers=int(centers.cx.size),
        n_rejected_centers=int(centers.rejected),
        n_candidates=len(candidates),
        n_candidate_centers=n_candidate_centers,
        n_rejected_candidates=n_rejected_candidates,
        n_passed_instances=n_passed_instances,
        n_passed_centers=n_passed_centers,
        n_passed_localizations=n_passed_localizations,
        candidate_log=candidate_log,
    )

    return aligned, stats


def instances_concatenate(aligned: dict) -> dict:
    keys = {
        'x',
        'y',
        'z',
        'intensity',
        'labels',
        'crlb_x',
        'crlb_y',
        'angle',
        'n_matched',
    }
    out = {k: [] for k in keys}
    for data in aligned.values():
        for k in keys:
            if data.get(k) is not None:
                out[k].append(data[k])

    ret = {}

    for k, v in out.items():
        if v:
            # if list of arrays is not empty, concatenate into a single array
            # if list of scalars make into a single array
            if isinstance(v[0], (np.ndarray, list)):
                ret[k] = np.concatenate(v)
            else:
                ret[k] = np.array(v)
        else:
            ret[k] = None
    return ret


def group_by(aligned: dict, key_fn) -> dict:
    '''Generic stratification — replaces group_by_occupancy and any future variant.'''
    out = {}
    for gid, data in aligned.items():
        out.setdefault(key_fn(data), {})[gid] = data
    return out
