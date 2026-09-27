from dataclasses import asdict, dataclass, field
from enum import StrEnum

import numpy as np
import pyqtgraph as pg

from microEye.analysis.rendering.base import RenderModes


class CenteringMode(StrEnum):
    CRLB_WEIGHTED_MEAN = 'crlb_weighted_mean'
    MEAN = 'mean'
    GEOMETRIC_MEDIAN = 'geometric_median'
    HUBER_CENTER = 'huber_center'


@dataclass
class CenteringOptions:
    center_mode: CenteringMode = CenteringMode.CRLB_WEIGHTED_MEAN
    eps: float = 1e-12
    rejection_threshold: float | None = None
    min_points: int | None = None
    max_points: int | None = None
    subtract_com: bool = False

    def __post_init__(self):
        if isinstance(self.center_mode, str):
            self.center_mode = CenteringMode(self.center_mode)


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


@dataclass
class ClusteringOptions:
    grouping_method: str = 'dbscan'
    distance_threshold: float | None = None
    sigma_xy: float = 15.0
    sigma_z: float = 45.0
    scale_data: bool = False
    xi : float = 0.05


@dataclass
class RenderOptions:
    mode: str = 'superimposed'
    rasterized: bool = True
    render_mode: RenderModes = RenderModes.HISTOGRAM
    pixel_size: float = 1.0
    extent: float = 200.0
    scalebar_length: float = 80.0

    def __post_init__(self):
        if isinstance(self.render_mode, int):
            self.render_mode = RenderModes(self.render_mode)


@dataclass
class RunOptions:
    clustering: ClusteringOptions
    rendering: RenderOptions
    model: dict
    export: dict

    def __post_init__(self):
        if isinstance(self.clustering, dict):
            self.clustering = ClusteringOptions(**self.clustering)
        if isinstance(self.rendering, dict):
            self.rendering = RenderOptions(**self.rendering)


@dataclass
class FitResult:
    '''Generic outcome of fitting ANY geometric model to a candidate site set.'''

    R: np.ndarray
    '''
    The rotation matrix that transforms the raw coordinates into the canonical frame.
    '''
    origin: np.ndarray
    '''
    The origin of the canonical frame, expressed in raw coordinates.
    '''
    inlier_mask: np.ndarray
    '''
    A boolean array (K,) indicating which input sites were used in the fit.
    '''
    indices: np.ndarray
    '''
    The integer indices (n_inliers, d) of the canonical positions
    corresponding to each inlier site.

    Examples:
    - For a ruler, d=1 and the indices are the slot numbers.
    - For a grid, d=2 and the indices are (row, col) pairs
    - For a ring, d=1 and the indices are the angular slot numbers.
    '''
    n_matched: int
    '''
    The number of distinct canonical positions that were explained by the inlier sites.
    '''
    residual: float
    '''
    The fit residual.
    '''
    meta: dict = field(default_factory=dict)
    '''
    Model-specific metadata for logging and debugging purposes.
    '''


@dataclass
class InstanceStats:
    n_localizations: int
    '''All input localizations, including noise.'''
    n_input_clusters: int
    '''Distinct non-negative labels in the original localization table.'''
    n_input_cluster_localizations: int
    '''Localizations belonging to those original clusters.'''

    n_cluster_centers: int
    '''Centers created by get_centers() after filtering.'''
    n_rejected_centers: int
    '''Centers rejected by center-level QC.'''

    n_candidates: int
    '''Candidate geometric groups produced by DBSCAN/HDBSCAN/OPTICS.'''
    n_candidate_centers: int
    '''Total site centers across all candidates.'''
    n_rejected_candidates: int
    '''Candidates where model.fit() returned None or residual QC failed.'''

    n_passed_instances: int
    '''Candidates that passed the geometric model.'''
    n_passed_centers: int
    '''Centers belonging to passed instances.'''
    n_passed_localizations: int
    '''Raw localizations belonging to passed instances.'''

    candidate_log: list[dict] = field(default_factory=list)


@dataclass
class PlotResult:
    '''
    Unified return type for all ClustersScatter render methods.

    raster      : pg.ImageItem or None   — 2D rasterized render
    scatter     : list[pg.ScatterPlotItem] — raw scatter layers (when not rasterized)
    histogram_x : list[pg.GraphicsObject] — X-projection histogram + KDE items
    histogram_y : list[pg.GraphicsObject] — Y-projection histogram + KDE items
    title       : str                     — plot title
    pixel_size  : float                   — nm per pixel (carried from rasterized_2d)
    '''

    raster: pg.ImageItem | None = None
    scatter: list[pg.ScatterPlotItem | pg.PlotCurveItem] = field(default_factory=list)
    histogram_x: list[pg.GraphicsObject] = field(default_factory=list)
    histogram_y: list[pg.GraphicsObject] = field(default_factory=list)
    title: str = ''
    pixel_size: float = 1.0

    histogram_angle: list[pg.GraphicsObject] = field(default_factory=list)
    histogram_intensity: list[pg.GraphicsObject] = field(default_factory=list)
    histogram_n_matched: list[pg.GraphicsObject] = field(default_factory=list)

    @property
    def is_rasterized(self) -> bool:
        return self.raster is not None

    @property
    def has_histograms(self) -> bool:
        return bool(self.histogram_x or self.histogram_y or self.histogram_angle)

@dataclass
class LabeledLocalizations:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None
    labels: np.ndarray
    intensity: np.ndarray | None
    crlb_x: np.ndarray | None
    crlb_y: np.ndarray | None
    crlb_z: np.ndarray | None

@dataclass
class ClusterRenderData:
    tables: list[LabeledLocalizations]
    run_meta: dict = field(default_factory=dict)

@dataclass
class PipelineResult:
    stats: list[InstanceStats]
    instances: list[dict]
    model_meta: dict
    run_meta: dict

