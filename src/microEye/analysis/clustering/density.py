import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numba as nb
import numpy as np
import pyqtgraph as pg
import tqdm
from scipy.spatial import ConvexHull, Delaunay, Voronoi

from microEye.analysis.clustering.plot import ClustersScatter, PlotResult
from microEye.qt import QAction, QtCore, QtGui, QtWidgets, getSaveFileName

logger = logging.getLogger(__name__)


class DensityAnalysisPlot(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Density Analysis Results')
        self.resize(800, 600)

        # Create plots
        self.global_plot = self.ci.addPlot(title='Cluster Scatter Plot')
        self.global_plot.getViewBox().setAspectLocked(True, 1)
        self.density_plot = self.ci.addPlot(title='Density Distribution')
        self.area_plot = self.ci.addPlot(title='Area Distribution')
        self.nextRow()
        self.cv_plot = self.ci.addPlot(title='Voronoi CV Distribution')
        self.counts_plot = self.ci.addPlot(title='Counts Distribution')

        # Set labels
        self.global_plot.setLabel('left', 'Y (nm)')
        self.global_plot.setLabel('bottom', 'X (nm)')
        self.density_plot.setLabel('left', 'Molecule Count')
        self.density_plot.setLabel('bottom', 'Local Density (Molecules / µm²)')

        self.area_plot.setLabel('bottom', 'Area (µm²)')
        self.area_plot.setLabel('left', 'Frequency')

        self.cv_plot.setLabel('bottom', 'Voronoi CV')
        self.cv_plot.setLabel('left', 'Frequency')

        self.counts_plot.setLabel('bottom', 'Counts')
        self.counts_plot.setLabel('left', 'Frequency')

    def update_density_plots(
        self,
        X,
        Y,
        Z,
        labels,
        densities,
        measures,
        counts,
        boundaries,
        cv_data,
        local_densities,
        cluster_id=-1,  # Index of the specific cluster you want to view
    ):
        if cluster_id == -1:
            res: PlotResult = ClustersScatter.scatter(
                X,
                Y,
                Z,
                labels,
                title='All Clusters',
                show_legend=True,
            )

            self.global_plot.setTitle('All Clusters')
            self.global_plot.clear()
            for item in res.scatter:
                self.global_plot.addItem(item)

            plots = [self.density_plot, self.area_plot, self.cv_plot, self.counts_plot]
            data = [densities, measures, cv_data, counts]

            for plot, data_dict in zip(plots, data):
                plot.clear()
                plot.setVisible(True)
                hist = np.histogram(list(data_dict.values()), bins=200)
                plot.plot(
                    hist[1], hist[0], stepMode=True, fillLevel=0, brush=(0, 0, 255, 100)
                )

        else:
            # Filter points for the specific cluster
            mask = labels == cluster_id
            X_cluster = X[mask]
            Y_cluster = Y[mask]
            Z_cluster = Z[mask] if Z is not None else None

            res: PlotResult = ClustersScatter.scatter(
                X_cluster,
                Y_cluster,
                Z_cluster,
                labels[mask],
                title=f'Cluster {cluster_id}',
                show_legend=False,
            )

            self.global_plot.setTitle(f'Cluster {cluster_id}')
            self.global_plot.clear()
            for item in res.scatter:
                self.global_plot.addItem(item)

            self.density_plot.clear()
            local_density_hist = np.histogram(local_densities[cluster_id], bins=200)
            self.density_plot.plot(
                local_density_hist[1],
                local_density_hist[0],
                stepMode=True,
                fillLevel=0,
                brush=(0, 0, 255, 100),
            )
            self.density_plot.addItem(
                pg.InfiniteLine(
                    pos=densities[cluster_id], angle=0, pen=pg.mkPen('r', width=2)
                )
            )

            for plot in [self.area_plot, self.cv_plot, self.counts_plot]:
                plot.clear()
                plot.setVisible(False)

    @classmethod
    def plot_density_results(
        cls,
        X,
        Y,
        Z,
        labels,
        densities,
        measures,
        counts,
        boundaries,
        cv_data,
        local_densities,
    ):
        '''
        Plots the density analysis results.

        Parameters
        ----------
        X, Y, Z : array-like
            Arrays of x, y, and z coordinates of localizations.
        labels : array-like
            Cluster labels for each point; -1 indicates noise.
        densities : dict
            Dictionary mapping cluster labels to their computed densities
            (molecules/µm² or molecules/µm³).
        measures : dict
            Dictionary mapping cluster labels to their computed measures
            (area or volume).
        counts : dict
            Dictionary mapping cluster labels to the number of points in each cluster.
        boundaries : dict
            Dictionary mapping cluster labels to the boundary points of each cluster.
        cv_data : dict
            Dictionary mapping cluster labels to their Voronoi Area
            Coefficient of Variation (CV) scores.
        local_densities : dict
            Dictionary mapping cluster labels to arrays of local densities for each
            point in the cluster.
        '''
        widget = cls()

        widget.update_density_plots(
            X,
            Y,
            Z,
            labels,
            densities,
            measures,
            counts,
            boundaries,
            cv_data,
            local_densities,
        )

        return widget


def calculate_simplex_measure(vertices: np.ndarray) -> float:
    n = vertices.shape[1]
    matrix = np.vstack([vertices.T, np.ones(n + 1)])
    determinant = np.linalg.det(matrix)
    measure = abs(determinant) / math.factorial(n)
    return measure


def alpha_shape(cluster_points: np.ndarray, alpha: float) -> np.ndarray:
    '''
    Computes the alpha shape (concave hull) of a set of points.

    Parameters
    -----------
    cluster_points : np.ndarray
        Array of shape (N, 2) or (N, 3) containing the coordinates
        of points in the cluster.
    alpha : float
        Alpha parameter for the alpha shape. Controls the maximum radius
        allowed for a valid cluster feature. Any triangle/tetrahedron
        larger than this is considered "empty space".

    Returns
    --------
    measure : float
        The area (for 2D) or volume (for 3D) of the alpha shape of the cluster.
    boundary_pts : np.ndarray
        Array of shape (M, 2) or (M, 3) containing the coordinates of the boundary
        points of the alpha shape.
    '''
    if len(cluster_points) < (3 if cluster_points.shape[1] == 2 else 4):
        return 0.0, np.array([])

    try:
        tri = Delaunay(cluster_points)
    except Exception:
        return 0.0, np.array([])

    measure = 0.0
    edges = {}

    IS_2D = cluster_points.shape[1] == 2

    for simplex in tri.simplices:
        vertices = cluster_points[simplex]

        if IS_2D:
            a = np.linalg.norm(vertices[0] - vertices[1])
            b = np.linalg.norm(vertices[1] - vertices[2])
            c = np.linalg.norm(vertices[2] - vertices[0])

            s = (a + b + c) / 2.0
            area_sq = s * (s - a) * (s - b) * (s - c)
            area = np.sqrt(max(0.0, area_sq))

            if area > 1e-9:
                r = (a * b * c) / (4.0 * area)
                if r < alpha:
                    measure += area
                    for i1, i2 in [
                        (simplex[0], simplex[1]),
                        (simplex[1], simplex[2]),
                        (simplex[2], simplex[0]),
                    ]:
                        edge = tuple(sorted((i1, i2)))
                        edges[edge] = edges.get(edge, 0) + 1
        else:
            vol = calculate_simplex_measure(vertices)
            if vol > 1e-9:
                v0, v1, v2, v3 = vertices[0], vertices[1], vertices[2], vertices[3]
                v_a, v_b, v_c = v1 - v0, v2 - v0, v3 - v0
                cross_ab = np.cross(v_a, v_b)
                denom = 2.0 * np.dot(cross_ab, v_c)
                if abs(denom) > 1e-9:
                    num = (
                        np.sum(v_c**2) * cross_ab
                        + np.sum(v_b**2) * np.cross(v_c, v_a)
                        + np.sum(v_a**2) * np.cross(v_b, v_c)
                    )
                    r = np.linalg.norm(num) / abs(denom)
                    if r < alpha:
                        measure += vol
                        for f in [
                            (simplex[0], simplex[1], simplex[2]),
                            (simplex[0], simplex[1], simplex[3]),
                            (simplex[0], simplex[2], simplex[3]),
                            (simplex[1], simplex[2], simplex[3]),
                        ]:
                            facet = tuple(sorted(f))
                            edges[facet] = edges.get(facet, 0) + 1

    # Filter edges that only occur once (the exposed outside shell boundary)
    boundary_indices = [item for item, count in edges.items() if count == 1]
    if len(boundary_indices) > 0:
        if IS_2D:
            # Resolve indices directly into edge coordinate segments
            boundary_pts = np.array(
                [cluster_points[list(edge)] for edge in boundary_indices]
            )
        else:
            # 3D boundary facets array
            boundary_pts = np.array(
                [cluster_points[list(facet)] for facet in boundary_indices]
            )

    return measure, boundary_pts


def voronoi_measure(cluster_points: np.ndarray) -> float:
    '''
    Computes the measure (area for 2D, volume for 3D) of
    the Voronoi tessellation of a set of points.

    Parameters
    -----------
    cluster_points : np.ndarray
        Array of shape (N, 2) or (N, 3) containing the
        coordinates of points in the cluster.

    Returns
    --------
    measure : float
        The total area (for 2D) or volume (for 3D) of
        the Voronoi cells corresponding to the input points.
    '''
    vor = Voronoi(cluster_points)

    measure = 0.0

    # Loop through every individual point's cell region
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]

        # A region index of -1 means it touches infinity
        if -1 not in region and len(region) > 0:
            cell_vertices = vor.vertices[region]
            measure += ConvexHull(cell_vertices).volume

    return measure


def voronoi_local_density(cluster_points: np.ndarray):
    N = len(cluster_points)
    IS_2D = cluster_points.shape[1] == 2
    scale_factor = 1e6 if IS_2D else 1e9  # nm² -> µm² or nm³ -> µm³

    local_densities = np.full(N, np.nan)

    if (3 if IS_2D else 4) > N:
        return 0.0, local_densities, 0.0

    try:
        vor_local = Voronoi(cluster_points)
    except Exception:
        return 0.0, local_densities, 0.0

    valid_cell_areas = []

    for i, region_idx in enumerate(vor_local.point_region):
        region = vor_local.regions[region_idx]
        if -1 not in region and len(region) > 0:
            try:
                cell_vertices = vor_local.vertices[region]
                cell_vol = ConvexHull(cell_vertices).volume
                # Filter out degenerate duplicate-point cell areas (< 1e-4 nm²)
                if cell_vol > 1e-4:
                    valid_cell_areas.append(cell_vol)
                    area_um = cell_vol / scale_factor
                    local_densities[i] = 1.0 / area_um
            except Exception:
                continue

    if len(valid_cell_areas) > 2:
        valid_cell_areas = np.array(valid_cell_areas)
        mean_area = np.mean(valid_cell_areas)
        std_area = np.std(valid_cell_areas)
        cluster_voronoi_cv = std_area / mean_area if mean_area > 0 else 0.0
        bounded_measure = np.sum(valid_cell_areas)
    else:
        cluster_voronoi_cv = 0.0
        bounded_measure = 0.0

    return cluster_voronoi_cv, local_densities, bounded_measure


def process_single_cluster(cluster_points: np.ndarray, method: str, **kwargs):
    try:
        num_molecules = len(cluster_points)
        IS_2D = cluster_points.shape[1] == 2
        min_pts = 3 if IS_2D else 4

        if num_molecules < min_pts:
            return 0.0, num_molecules, None, 0.0, np.full(num_molecules, np.nan)

        hull = ConvexHull(cluster_points)
        hull_boundary = (
            cluster_points[hull.vertices] if IS_2D else cluster_points[hull.simplices]
        )

        voronoi_cv, local_density, bounded_voronoi_measure = voronoi_local_density(
            cluster_points
        )

        if method == 'convex_hull':
            measure = hull.volume
            boundary_pts = hull_boundary
        elif method == 'alpha_shape':
            alpha = kwargs.get('alpha', 200.0)
            measure, boundary_pts = alpha_shape(cluster_points, alpha)
            if boundary_pts is None or len(boundary_pts) == 0:
                boundary_pts = hull_boundary
        elif method == 'voronoi':
            measure = bounded_voronoi_measure
            boundary_pts = hull_boundary
            # Count only bounded molecules so total density isn't inflated
            bounded_count = np.count_nonzero(~np.isnan(local_density))
            if bounded_count > 0:
                num_molecules = bounded_count
        else:
            raise ValueError(
                f"Unknown method '{method}' for cluster density calculation."
            )

        return measure, num_molecules, boundary_pts, voronoi_cv, local_density

    except Exception as e:
        logger.debug(f'Error processing cluster: {e}')
        return 0.0, len(cluster_points), None, 0.0, np.full(len(cluster_points), np.nan)


def get_cluster_density(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    labels: np.ndarray,
    method: str = 'convex_hull',
    **kwargs,
):
    """
    Computes the density of points in each cluster using a Convex Hull approach.

    Parameters
    -----------
    X, Y, Z: array-like
        Arrays of x, y, z coordinates of localizations.
    labels: array-like
        Cluster labels for each point; -1 indicates noise.
    method: str, optional
        Method to compute cluster density.

        - 'convex_hull': Convex Hull to estimate the area of each cluster.
        - 'alpha_shape': Alpha shape to estimate the area of each cluster.
        - 'voronoi': Voronoi tessellation to estimate the area of each cluster.

    For 'alpha_shape':

    alpha: float, optional
        Alpha parameter for the alpha shape. Default is 30.0 nm.

    Returns
    --------
    cluster_densities: dict
        A dictionary mapping cluster labels to their computed densities
        (2D: molecules/µm², 3D: molecules/µm³).
    cluster_measure: dict
        A dictionary mapping cluster labels to their computed measures
        (2D: area µm², 3D: volume µm³).
    cluster_counts: dict
        A dictionary mapping cluster labels to the number of points in each cluster.
    cluster_boundaries: dict
        A dictionary mapping cluster labels to the boundary points of each cluster.
    cluster_voronoi_cv: dict
        A dictionary mapping cluster labels to their Voronoi Area
        Coefficient of Variation (CV)
    cluster_local_density: dict
        A dictionary mapping cluster labels to arrays of local densities for each
        point in the cluster.
    """
    cluster_densities = {}
    cluster_measure = {}
    cluster_counts = {}
    cluster_boundaries = {}
    cluster_voronoi_cv = {}
    cluster_local_density = {}

    # Filter noise points (label -1)
    valid_mask = labels >= 0
    x = X[valid_mask]
    y = Y[valid_mask]
    z = Z[valid_mask] if Z is not None else None
    labels_valid = labels[valid_mask]

    if len(labels_valid) == 0:
        return {}, {}, {}, {}, {}, {}

    # Sort points by cluster label
    sort_idx = np.argsort(labels_valid)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    z_sorted = z[sort_idx] if z is not None else None
    labels_sorted = labels_valid[sort_idx]

    IS_2D = z_sorted is None
    points_sorted = (
        np.column_stack((x_sorted, y_sorted))
        if IS_2D
        else np.column_stack((x_sorted, y_sorted, z_sorted))
    )

    # Safe slicing using np.unique and np.split
    unique_labels, split_indices = np.unique(labels_sorted, return_index=True)
    cluster_points_list = np.split(points_sorted, split_indices[1:])

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_cluster,
                pts,
                method,
                **kwargs,
            ): lbl
            for pts, lbl in zip(cluster_points_list, unique_labels)
        }

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Processing Clusters',
            unit='cluster',
        ):
            lbl = futures[future]
            (
                measure,
                num_molecules,
                boundary_pts,
                voronoi_cv,
                local_density,
            ) = future.result()

            cluster_voronoi_cv[lbl] = voronoi_cv
            cluster_local_density[lbl] = local_density

            # Convert Area (nm² to µm²) or Volume (nm³ to µm³)
            measure_um = measure / (1e6 if IS_2D else 1e9)
            cluster_measure[lbl] = measure_um

            density_per_um = (
                num_molecules / measure_um if measure_um > 0 else 0.0
            )

            cluster_densities[lbl] = density_per_um
            cluster_counts[lbl] = num_molecules
            cluster_boundaries[lbl] = boundary_pts

    return (
        cluster_densities,
        cluster_measure,
        cluster_counts,
        cluster_boundaries,
        cluster_voronoi_cv,
        cluster_local_density,
    )

    cluster_densities = {}  # Units: molecules / µm² or molecules / µm³
    cluster_measure = {}  # Units: area µm² or volume µm³
    cluster_counts = {}
    cluster_boundaries = {}
    cluster_voronoi_cv = {}
    cluster_local_density = {}

    # fitler noise points (label -1) and sort by label
    valid_mask = labels >= 0
    x = X[valid_mask]
    y = Y[valid_mask]
    z = Z[valid_mask] if Z is not None else None
    labels = labels[valid_mask]

    # Sort points by cluster label for efficient processing
    _arg_sort = np.argsort(labels)
    x = x[_arg_sort]
    y = y[_arg_sort]
    z = z[_arg_sort] if z is not None else None

    IS_2D = z is None

    label_ids = np.cumsum(np.bincount(labels.astype(np.int64)))

    unique_labels = np.unique(labels.astype(np.int64))

    points = np.column_stack((x, y)) if z is None else np.column_stack((x, y, z))

    # parallel processing of clusters using ThreadPoolExecutor and tqdm for progress bar
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_single_cluster,
                points[label_ids[i - 1] if i > 0 else 0 : label_ids[i]],
                method,
                **kwargs,
            ): i
            for i in unique_labels
        }

        for future in tqdm.tqdm(
            as_completed(futures),
            total=len(futures),
            desc='Processing Clusters',
            unit='cluster',
        ):
            i = futures[future]
            (
                measure,
                num_molecules,
                boundary_pts,
                voronoi_cv,
                local_density,
            ) = future.result()

            if voronoi_cv is not None:
                cluster_voronoi_cv[i] = voronoi_cv
            if local_density is not None:
                cluster_local_density[i] = local_density

            # Convert Area (nm² to µm²) or Volume (nm³ to µm³) to microns
            cluster_measure[i] = measure / (1e6 if IS_2D else 1e9)

            density_per_um2 = (
                num_molecules / cluster_measure[i] if cluster_measure[i] > 0 else 0
            )

            cluster_densities[i] = density_per_um2
            cluster_counts[i] = num_molecules
            cluster_boundaries[i] = boundary_pts

    return (
        cluster_densities,
        cluster_measure,
        cluster_counts,
        cluster_boundaries,
        cluster_voronoi_cv,
        cluster_local_density,
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 1. SIMULATE AN UNEVEN CLUSTER (Dense Core + Loose Shell)
    np.random.seed(42)
    # Dense core (60 points closely packed)
    X_core = np.random.normal(0, 250, 60)
    Y_core = np.random.normal(0, 250, 60)
    # Loose outer shell (60 points widely dispersed)
    X_shell = np.random.normal(0, 400, 60)
    Y_shell = np.random.normal(0, 400, 60)

    X = np.concatenate([X_core, X_shell])
    Y = np.concatenate([Y_core, Y_shell])
    labels = np.zeros(120)  # Group into a single Cluster '0'

    # 2. EXECUTE THE UPGRADED FUNCTION
    densities, measures, counts, boundaries, cv_data, local_dens_data = (
        get_cluster_density(X, Y, None, labels, method='alpha_shape', alpha=200)
    )

    # 3. SCIENTIFIC INTERPRETATION AND PLOTTING
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: The Global Spatial Distribution
    ax1.scatter(X, Y, color='black', s=15, label='Molecules')
    ax1.set_title(
        f'Cluster 0: Global Properties\n'
        f'Total Molecules: {counts[0]}\n'
        f'Global Area (Alpha Shape): {measures[0]:.4f} µm²\n'
        f'Global Avg Density: {densities[0]:.0f} mol/µm²'
    )
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')

    # Panel B: The Local Internal Density Profile (Histogram)
    local_densities = local_dens_data[0]
    cv_value = cv_data[0]

    ax2.hist(local_densities, bins=20, color='purple', edgecolor='black', alpha=0.7)
    ax2.axvline(
        densities[0],
        color='red',
        linestyle='--',
        lw=2,
        label=f'Global Average ({densities[0]:.0f})',
    )
    ax2.set_title(
        f'Internal Local Density Distribution\nVoronoi Area CV Score: {cv_value:.2f}'
    )
    ax2.set_xlabel('Local Density (Molecules / µm²)')
    ax2.set_ylabel('Molecule Count')
    ax2.legend()

    plt.tight_layout()
    plt.show()
