import logging
import traceback
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from microEye.analysis.clustering.dbscan import (
    CenteringMode,
    ClusterCenterResult,
    get_center,
    get_centers,
)
from microEye.analysis.clustering.rulers import (
    get_rulers_from_clusters,
    rulers_concatenate,
)
from microEye.analysis.fitting.pyfit3Dcspline import CPUmleFit_LM
from microEye.analysis.fitting.results import DataColumns, FittingResults
from microEye.analysis.rendering import BaseRenderer, RenderModes
from microEye.qt import QAction, QtCore, QtGui, QtWidgets, getSaveFileName
from microEye.utils.gui_helper import create_double_spin_box, create_spin_box

logger = logging.getLogger(__name__)


def export_image_as_tiff(image: np.ndarray, base_name: str | None = None):
    filename, _ = getSaveFileName(
        caption='Export Image as TIFF',
        directory=base_name if base_name else '',
        filter='TIFF Files (*.tif *.tiff)',
    )

    if filename:
        try:
            import tifffile as tf

            tf.imwrite(filename, image.astype(np.float32))
            logger.info('Image exported as TIFF: %s', filename)
        except Exception as e:
            logger.error('Error exporting image as TIFF: %s', e)


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

    @property
    def is_rasterized(self) -> bool:
        return self.raster is not None

    @property
    def has_histograms(self) -> bool:
        return bool(self.histogram_x or self.histogram_y or self.histogram_angle)


class ClustersScatter:
    VLINE_COLOR = QtGui.QColor(220, 60, 60, 180)  # red
    HIST_COLOR = QtGui.QColor(70, 130, 180, 180)
    KDE_COLOR = QtGui.QColor(220, 60, 60, 255)
    PEAK_COLOR = QtGui.QColor(255, 165, 0, 220)  # orange
    GRID_COLOR = QtGui.QColor(180, 180, 180, 60)

    @classmethod
    def scatter(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None,
        labels: np.ndarray,
        **kwargs,
    ) -> PlotResult:
        try:
            # Filter out noise (label < 0) safely
            valid_mask = labels >= 0
            x = x[valid_mask]
            y = y[valid_mask]
            z = z[valid_mask] if z is not None else None
            labels = labels[valid_mask]

            if len(labels) == 0:
                return None

            unique_labels = np.unique(labels)
            max_label = unique_labels.max()

            COLORS = 64

            hues = np.linspace(0, 1, COLORS, endpoint=False)
            color_lookup = np.zeros((COLORS, 3), dtype=np.uint8)
            for idx, h in enumerate(hues):
                i = int(h * 6.0)
                f = (h * 6.0) - i
                t = int(229 * f)
                q = int(229 * (1.0 - f))
                i = i % 6
                if i == 0:
                    r, g, b = 229, t, 0
                elif i == 1:
                    r, g, b = q, 229, 0
                elif i == 2:
                    r, g, b = 0, 229, t
                elif i == 3:
                    r, g, b = 0, q, 229
                elif i == 4:
                    r, g, b = t, 0, 229
                else:
                    r, g, b = 229, 0, q
                color_lookup[idx] = [r, g, b]

            scatter_layers = []
            for color_idx in range(COLORS):
                mask = (labels % COLORS) == color_idx
                if not np.any(mask):
                    continue

                scatter = pg.ScatterPlotItem(
                    x=x[mask],
                    y=y[mask],
                    size=3,
                    pen=None,
                    brush=pg.mkBrush(
                        color_lookup[color_idx]
                    ),  # Single uniform brush per layer!
                    pxMode=True,
                )
                scatter_layers.append(scatter)

            return PlotResult(
                scatter=scatter_layers,
                title=kwargs.get('title', 'Cluster Scatter'),
            )
        except Exception:
            logger.error('Error in ClustersScatter.scatter: %s', traceback.format_exc())

    @classmethod
    def rasterized_2d(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None,
        intensity: np.ndarray,
        labels: np.ndarray,
        crlb_x: np.ndarray | None,
        crlb_y: np.ndarray | None,
        **kwargs,
    ) -> tuple[pg.ImageItem, np.ndarray]:
        try:
            pixel_size = kwargs.get('pixel_size', 0.5)
            render_mode = kwargs.get('render_mode', RenderModes.HISTOGRAM)
            center_mode = kwargs.get('center_mode', CenteringMode.CRLB_WEIGHTED_MEAN)
            eps = kwargs.get('eps', 1e-12)
            extent = kwargs.get('extent')
            renderer = BaseRenderer(
                pixel_size,
                None,
                render_mode,
            )

            mu_x, mu_y, mu_z, sigma_com = get_center(
                x,
                y,
                z,
                center_mode=center_mode,
                crlb_x=crlb_x,
                crlb_y=crlb_y,
                eps=eps,
            )

            if kwargs.get('skip_centering', False):
                mu_x, mu_y = 0.0, 0.0

            var_x = np.mean((x - mu_x) ** 2)
            var_y = np.mean((y - mu_y) ** 2)
            radius = 2.0 * max(np.sqrt(var_x), np.sqrt(var_y))

            # Extent is full width/height of the image in nm
            # so we need to divide by 2 to get the radius
            if extent is not None:
                radius = max(radius, extent / 2.0)

            mask = (np.abs(x - mu_x) <= radius) & (np.abs(y - mu_y) <= radius)

            x_crop = x[mask] - mu_x
            y_crop = y[mask] - mu_y

            if not extent:
                x_min = np.min(x_crop)
                x_max = np.max(x_crop)
                y_min = np.min(y_crop)
                y_max = np.max(y_crop)

                sym_x = max(abs(x_min), abs(x_max))
                sym_y = max(abs(y_min), abs(y_max))

                # shape of image should be odd and centeral
                # pixel should be the center of the image
                shape_x = int(2 * np.ceil(sym_x / pixel_size)) + 1
                shape_y = int(2 * np.ceil(sym_y / pixel_size)) + 1

                # find offset in original coordinates to center the image
                shift_x = sym_x
                shift_y = sym_y
            else:
                shape_x = int(2 * np.ceil(radius / pixel_size)) + 1
                shape_y = int(2 * np.ceil(radius / pixel_size)) + 1

                # find offset in original coordinates to center the image
                shift_x = radius
                shift_y = radius

            options = {
                'x_padding': kwargs.get('x_padding', 0),
                'y_padding': kwargs.get('y_padding', 0),
                'x_offset': kwargs.get('x_offset', 0),
                'y_offset': kwargs.get('y_offset', 0),
            }

            image = renderer.render_xy(
                x_crop + shift_x,
                y_crop + shift_y,
                intensity[mask],
                shape=(shape_y, shape_x),
                zero_origin=False,
                **options,
            )

            item = pg.ImageItem(image, axisOrder='row-major')
            item.pixel_size = pixel_size
            # populate the tiff_filename attribute for later use in export
            # it should contain metadata about the settings used
            item.tiff_filename = (
                f"raster_center_mode={center_mode}_"
                f"render_mode={render_mode}_"
                f"pixel_size={pixel_size:.2f}_"
                f"extent={extent if extent else 'auto'}_"
                f"crlb_x={crlb_x is not None}_"
                f"crlb_y={crlb_y is not None}_"
                f"num_clusters={np.unique(labels).size}_.tiff"
            )

            return item, image
        except Exception:
            logger.error(
                'Error in ClustersScatter.rasterized: %s', traceback.format_exc()
            )

        return None, None

    @classmethod
    def histogram(
        cls, x: np.ndarray, bin_size: float = 1.0, detect_peaks: bool = False, **kwargs
    ) -> list[pg.GraphicsObject]:

        res = []

        verbose = kwargs.get('verbose', True)

        try:
            bins = np.arange(np.min(x), np.max(x) + bin_size, bin_size)
            counts, edges = np.histogram(x, bins=bins, density=True)
            centers = (edges[:-1] + edges[1:]) / 2

            bar_item = pg.BarGraphItem(
                x=edges[:-1],
                height=counts,
                width=bin_size * 0.9,
                brush=pg.mkBrush(cls.HIST_COLOR),
                pen=pg.mkPen(None),
            )
            res.append(bar_item)

            if detect_peaks:
                peak_distance_bins: int = kwargs.get('peak_distance_bins', 30)
                peak_prominence: float = kwargs.get('peak_prominence', 0.001)
                grid_n_points: int = kwargs.get('grid_n_points', 1000)
                axis_label = kwargs.get('axis_label', 'X')

                kde_bw = bin_size / x.std(ddof=1)

                # Perform KDE on the histogram data
                kde = gaussian_kde(x, bw_method=kde_bw)
                grid = np.linspace(np.min(x), np.max(x), grid_n_points)
                kde_y = kde(grid)

                kde_curve = pg.PlotCurveItem(
                    x=grid,
                    y=kde_y,
                    pen=pg.mkPen(cls.KDE_COLOR, width=2),
                )
                res.append(kde_curve)

                # Peak detection on KDE
                peaks, props = find_peaks(
                    kde(grid),
                    distance=peak_distance_bins,
                    prominence=peak_prominence,
                )
                peak_positions = grid[peaks]
                peak_heights = kde_y[peaks]

                if verbose:
                    logger.info(
                        f'[{axis_label}] Peak positions: {np.round(peak_positions, 1)}'
                    )
                    if len(peak_positions) >= 2:
                        logger.info(
                            f'[{axis_label}] Spacings: '
                            f'{np.round(np.diff(peak_positions), 1)}'
                        )

                # Add vertical lines for detected peaks
                for pos in peak_positions:
                    vline = pg.InfiniteLine(
                        pos=pos,
                        angle=90,
                        pen=pg.mkPen(
                            cls.PEAK_COLOR, width=1.5, style=QtCore.Qt.PenStyle.DashLine
                        ),
                    )
                    res.append(vline)

                # estimate sigma from FWHM for each peak
                sigma_values = []
                for peak_idx, (px, ph) in enumerate(zip(peak_positions, peak_heights)):
                    peak_height = kde_y[peaks[peak_idx]]
                    half_max = peak_height / 2

                    left_idx = peaks[peak_idx]
                    right_idx = peaks[peak_idx]
                    while left_idx > 0 and kde_y[left_idx] > half_max:
                        left_idx -= 1
                    while right_idx < len(kde_y) - 1 and kde_y[right_idx] > half_max:
                        right_idx += 1

                    fwhm = grid[right_idx] - grid[left_idx]
                    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
                    sigma_values.append(sigma)

                    label = pg.TextItem(
                        text=f'{axis_label}={px:.1f}\nσ={sigma:.2f}',
                        anchor=(0.5, 0.5),
                        color=pg.mkColor(cls.PEAK_COLOR),
                    )
                    label.setScale(0.35)
                    label.setPos(
                        px - label.boundingRect().width() * 0.5 / 2,
                        ph * 1.1,
                    )
                    res.append(label)

                if verbose:
                    logger.info(f'σ[{axis_label}]: {np.round(sigma_values, 1)}')

        except Exception:
            logger.error(
                'Error in ClustersScatter.histogram [%s]: %s',
                axis_label,
                traceback.format_exc(),
            )

        return res

    @classmethod
    def _build_histograms(
        cls,
        x: np.ndarray,
        y: np.ndarray | None,
        bin_size: float = 1.0,
        detect_peaks: bool = False,
        symmetrize: bool = False,
        **kwargs,
    ) -> tuple[list[pg.GraphicsObject], list[pg.GraphicsObject]]:
        '''
        Build X (and optionally Y) projection histogram items.
        Single implementation used by both superimposed and ruler paths.

        symmetrize : mirror x about 0 before histogramming (symmetric rulers).
        Returns (histogram_x_items, histogram_y_items).
        '''
        x_data = np.concatenate([x, -x]) if symmetrize else x

        hist_x = cls.histogram(
            x_data,
            bin_size=bin_size,
            detect_peaks=detect_peaks,
            axis_label='X',
            **kwargs,
        )

        hist_y = (
            cls.histogram(
                y,
                bin_size=bin_size,
                detect_peaks=detect_peaks,
                axis_label='Y',
                **kwargs,
            )
            if y is not None
            else []
        )
        return hist_x, hist_y

    @classmethod
    def _make_raster_or_scatter(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None,
        intensity: np.ndarray,
        labels: np.ndarray,
        crlb_x: np.ndarray | None,
        crlb_y: np.ndarray | None,
        fit_gaussian: bool = False,
        **kwargs,
    ) -> tuple[pg.ImageItem | None, list[pg.ScatterPlotItem], float]:
        '''
        Returns (raster, scatter_layers, pixel_size).
        Exactly one of raster / scatter_layers will be populated.
        fit_gaussian: run CPUmleFit_LM on the raster image and attach .param.
        '''
        pixel_size = kwargs.get('pixel_size', 1.0)

        if not kwargs.get('rasterized', True):
            scatter = pg.ScatterPlotItem(
                x=x,
                y=y,
                size=3,
                pen=None,
                brush=pg.mkBrush(pg.mkColor(255, 0, 0, 32)),
                pxMode=True,
            )
            return None, [scatter], pixel_size

        item, image = cls.rasterized_2d(
            x, y, z, intensity, labels, crlb_x, crlb_y, **kwargs
        )

        if fit_gaussian and image is not None:
            try:
                params, crlb_fit, loglike = CPUmleFit_LM(
                    np.array([image.flatten()], dtype=np.float32),
                    4,
                    np.array([5], dtype=np.float32),
                    None,
                    0,
                )
                item.param = params[0]
            except Exception:
                logger.warning('CPUmleFit_LM failed: %s', traceback.format_exc())

        return item, [], pixel_size

    @classmethod
    def superimposed(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None,
        intensity: np.ndarray,
        labels: np.ndarray,
        crlb_x: np.ndarray | None = None,
        crlb_y: np.ndarray | None = None,
        **kwargs,
    ) -> PlotResult:
        try:
            centers_result = get_centers(
                x,
                y,
                z,
                labels,
                intensity,
                crlb_x,
                crlb_y,
                subtract_com=True,
                **kwargs,
            )

            if centers_result is None or not isinstance(
                centers_result, ClusterCenterResult
            ):
                raise ValueError(
                    'get_centers did not return a valid ClusterCenterResult.'
                )

            x = centers_result.x
            y = centers_result.y
            z = centers_result.z
            intensity = centers_result.intensity
            labels = centers_result.labels
            crlb_x = centers_result.crlb_x
            crlb_y = centers_result.crlb_y

            raster, scatter, pixel_size = cls._make_raster_or_scatter(
                x,
                y,
                z,
                intensity,
                labels,
                crlb_x,
                crlb_y,
                fit_gaussian=kwargs.get('fit_gaussian', True),
                **kwargs,
            )

            hist_x, hist_y = [], []
            if kwargs.get('show_histogram', True):
                bin_size = kwargs.get('pixel_size', 1.0)
                hist_x, hist_y = cls._build_histograms(
                    x,
                    y,
                    bin_size=bin_size,
                    detect_peaks=kwargs.get('detect_peaks', True),
                    symmetrize=kwargs.get('symmetrize', False),
                    grid_n_points=kwargs.get('grid_n_points', 400),
                    **kwargs,
                )

            return PlotResult(
                raster=raster,
                scatter=scatter,
                histogram_x=hist_x,
                histogram_y=hist_y,
                title=kwargs.get('title', 'Superimposed clusters'),
                pixel_size=pixel_size,
            )

        except Exception:
            logger.error('Error in superimposed: %s', traceback.format_exc())
            return PlotResult()

    @classmethod
    def superimposed_batch(
        cls,
        fitting_results: list['FittingResults'],
        **kwargs,
    ) -> list[PlotResult]:
        results = []
        for i, result in enumerate(fitting_results):
            pr = cls.superimposed(
                result.data[DataColumns.X],
                result.data[DataColumns.Y],
                result.data[DataColumns.Z],
                result.data[DataColumns.INTENSITY],
                result.data[DataColumns.CLUSTER_ID],
                crlb_x=result.data[DataColumns.CRLB_X],
                crlb_y=result.data[DataColumns.CRLB_Y],
                **kwargs,
            )
            results.append(pr)
            logger.info('Superimposed (%d/%d)', i + 1, len(fitting_results))
        return results

    @classmethod
    def superimposed_batch_aggregate(
        cls,
        fitting_results: list['FittingResults'],
        **kwargs,
    ) -> PlotResult:
        '''Concatenate all tables first, then superimpose once.'''
        try:
            is_z = any(r.data[DataColumns.Z] is not None for r in fitting_results)
            x, y, z, labels, intensity, crlb_x, crlb_y = [], [], [], [], [], [], []
            max_label = 0

            for result in fitting_results:
                x.append(result.data[DataColumns.X].copy())
                y.append(result.data[DataColumns.Y].copy())
                if is_z:
                    z.append(result.data[DataColumns.Z].copy())
                ids = result.data[DataColumns.CLUSTER_ID].copy()
                ids[ids >= 0] += max_label
                labels.append(ids)
                max_label = int(ids.max()) + 1
                intensity.append(result.data[DataColumns.INTENSITY].copy())
                crlb_x.append(result.data[DataColumns.CRLB_X].copy())
                crlb_y.append(result.data[DataColumns.CRLB_Y].copy())

            return cls.superimposed(
                x=np.concatenate(x),
                y=np.concatenate(y),
                z=np.concatenate(z) if is_z else None,
                intensity=np.concatenate(intensity),
                labels=np.concatenate(labels),
                crlb_x=np.concatenate(crlb_x),
                crlb_y=np.concatenate(crlb_y),
                **kwargs,
            )
        except Exception:
            logger.error('superimposed_batch_aggregate: %s', traceback.format_exc())
            return PlotResult()

    @classmethod
    def scatter_batch(
        cls,
        fitting_results: list[FittingResults],
        **kwargs,
    ) -> list[PlotResult]:
        try:
            scatter_plots = []

            for i, result in enumerate(fitting_results):
                x = result.data[DataColumns.X]
                y = result.data[DataColumns.Y]
                z = (
                    result.data[DataColumns.Z]
                    if result.data[DataColumns.Z] is not None
                    else None
                )
                labels = result.data[DataColumns.CLUSTER_ID]

                scatter_plots.append(cls.scatter(x, y, z, labels, **kwargs))

                logger.info(
                    'Cluster rendering completed for table: (%d/%d)',
                    i + 1,
                    len(fitting_results),
                )
        except Exception as e:
            logger.error(
                'Error in ClustersScatter.scatter_batch: %s', traceback.format_exc()
            )

        return scatter_plots

    @classmethod
    def rasterize_ruler(cls, ruler_data: dict[str, np.ndarray], **kwargs):
        x = ruler_data['x']
        y = ruler_data['y']

        # Use the mean of all aligned_sites as the true ruler center
        # aligned_sites are (n_sites, 2) per ruler; ruler_data has them stacked
        # so we take the grand mean across all rulers and all sites
        all_sites = ruler_data.get(
            'aligned_sites'
        )  # shape: (N * n_sites, 2) after concat
        if all_sites is not None and len(all_sites) > 0:
            site_mu_x = np.mean(all_sites[:, 0])
            site_mu_y = np.mean(all_sites[:, 1])
        else:
            site_mu_x = np.mean(x)
            site_mu_y = np.mean(y)

        # Shift so the site-averaged center is at (0, 0)
        ruler_data = dict(ruler_data)  # shallow copy — don't mutate original
        ruler_data['x'] = x - site_mu_x
        ruler_data['y'] = y - site_mu_y

        item, _ = cls.rasterized_2d(
            ruler_data['x'],
            ruler_data['y'],
            ruler_data['z'],
            intensity=ruler_data['intensity'],
            labels=ruler_data['labels'],
            crlb_x=ruler_data['crlb_x'],
            crlb_y=ruler_data['crlb_y'],
            skip_centering=True,  # already centered
            **kwargs,
        )
        return item

    @classmethod
    def _render_ruler(
        cls, ruler_concatenated: dict[str, np.ndarray], **kwargs
    ) -> PlotResult:
        item = cls.rasterize_ruler(ruler_concatenated, **kwargs)

        # remove y outliers for histogram calculation
        # mask = np.abs(
        #     ruler_concatenated['y'] - np.mean(ruler_concatenated['y'])
        # ) <= 3 * np.std(ruler_concatenated['y'])

        hist_x, hist_y = cls._build_histograms(
            ruler_concatenated['x'],
            ruler_concatenated['y'],
            bin_size=kwargs.get('pixel_size', 1.0),
            detect_peaks=kwargs.get('detect_peaks', True),
            symmetrize=kwargs.get('symmetrize', False),
            **kwargs,
        )

        if kwargs.get('angle_hist', False):
            # rad to deg conversion and conversion to range [-90, 90]
            angles = np.degrees(ruler_concatenated['angle'])
            min_angle = angles.min()
            max_angle = angles.max()
            angles = (angles + 90) % 180 - 90 if max_angle - min_angle > 180 else angles
            hist_angle = cls.histogram(
                angles,
                bin_size=kwargs.get('angle_bin_size', 1),
                detect_peaks=False,
                axis_label='Angle [rad]',
                **kwargs,
            )
        else:
            hist_angle = None

        if kwargs.get('intensity_hist', False):
            hist_intensity = cls.histogram(
                ruler_concatenated['intensity'],
                bin_size=kwargs.get('intensity_bin_size', 1),
                detect_peaks=False,
                axis_label='Intensity [a.u.]',
                **kwargs,
            )
        else:
            hist_intensity = None

        return PlotResult(
            raster=item,
            histogram_x=hist_x,
            histogram_y=hist_y,
            histogram_angle=hist_angle,
            histogram_intensity=hist_intensity,
            title='Superimposed rulers',
            pixel_size=kwargs.get('pixel_size', 1.0),
        )

    @classmethod
    def render_clusters_batch(
        cls, tables: list[FittingResults], **kwargs
    ) -> list[PlotResult]:
        try:
            mode = kwargs.get('mode', 'fov')
            rasterized = kwargs.get('rasterized', False)
            pixel_size = kwargs.get('pixel_size', 1.0)
            render_mode = kwargs.get('render_mode', 'scatter')
            center_mode = kwargs.get('center_mode', CenteringMode.CRLB_WEIGHTED_MEAN)
            extent = kwargs.get('extent')
            rejection_threshold = kwargs.get('rejection_threshold')
            min_points = kwargs.get('min_points')
            aggregate = kwargs.get('aggregate', False)

            res = []

            if mode == 'superimposed':
                worker_kwargs = {
                    'rasterized': rasterized,
                    'pixel_size': pixel_size,
                    'render_mode': render_mode,
                    'center_mode': center_mode,
                    'extent': extent,
                    'rejection_threshold': rejection_threshold,
                    'min_points': min_points,
                }
                if not aggregate:
                    res = ClustersScatter.superimposed_batch(tables, **worker_kwargs)
                else:
                    res.append(
                        ClustersScatter.superimposed_batch_aggregate(
                            tables, **worker_kwargs
                        )
                    )
            else:
                worker_kwargs = {}
                res = ClustersScatter.scatter_batch(tables, **worker_kwargs)

        except Exception:
            traceback.print_exc()

        return res

    @classmethod
    def render_rulers_batch(
        cls,
        tables: list['FittingResults'],
        **kwargs,
    ) -> list[PlotResult]:
        '''
        Full ruler pipeline per table (or aggregated).
        Returns one PlotResult per output image — same shape as before
        but now each carries its own histograms.
        '''
        try:
            aggregate = kwargs.get('aggregate', False)
            bin_size = kwargs.get('pixel_size', 1.0)

            aggregate_data: dict[int, dict] = {}
            results: list[PlotResult] = []

            for i, table in enumerate(tables):
                ruler_aligned = get_rulers_from_clusters(
                    table.data[DataColumns.X],
                    table.data[DataColumns.Y],
                    table.data[DataColumns.Z],
                    table.data[DataColumns.CLUSTER_ID],
                    intensity=table.data[DataColumns.INTENSITY],
                    crlb_x=table.data[DataColumns.CRLB_X],
                    crlb_y=table.data[DataColumns.CRLB_Y],
                    **kwargs,
                )
                ruler_concat = rulers_concatenate(ruler_aligned)

                if aggregate:
                    aggregate_data[i] = ruler_concat
                else:
                    results.append(cls._render_ruler(ruler_concat, **kwargs))

            if aggregate:
                all_data = rulers_concatenate(aggregate_data)
                results.append(cls._render_ruler(all_data, **kwargs))

            return results

        except Exception:
            traceback.print_exc()
            return []


class ClusterPlotOptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, batch_mode=False):
        super().__init__(parent)
        self.setWindowTitle('Cluster Plot Options')

        self._batch_mode = batch_mode

        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItem('Full FOV', 'fov')
        self.mode_box.addItem('Superimposed', 'superimposed')
        self.mode_box.setCurrentIndex(1)
        self.mode_box.currentIndexChanged.connect(self.mode_changed)

        self.is_rasterized = QtWidgets.QCheckBox('Rasterize?')
        self.is_rasterized.setChecked(True)

        self.render_box = QtWidgets.QComboBox()
        self.render_box.addItem('Histogram', RenderModes.HISTOGRAM)
        self.render_box.addItem('Event Histogram', RenderModes.EVENT_HISTOGRAM)
        self.render_box.addItem('Gaussian', RenderModes.GAUSSIAN)

        self.pixel_size = create_double_spin_box(
            min_value=0.1, max_value=50, decimals=2, initial_value=0.5
        )

        # centering unweighted mean, CRLB-weighted mean, geometric median
        self.centering = QtWidgets.QComboBox()
        for mode in CenteringMode:
            self.centering.addItem(mode.value.replace('_', ' ').title(), mode)

        # extent in nm
        self.extent = create_double_spin_box(
            min_value=50, max_value=1000, decimals=0, initial_value=100
        )
        # rejection threshold for cluster uncertainty
        self.rejection_threshold = create_double_spin_box(
            min_value=0.0, max_value=100, decimals=2, initial_value=0
        )
        self.rejection_threshold.setToolTip(
            'Reject clusters with uncertainty (sigma_com) above this threshold. '
            'Set to 0 to disable.'
        )
        # minimum number of localizations per cluster
        self.min_points = create_spin_box(min_value=1, max_value=1000, initial_value=1)
        self.min_points.setToolTip(
            'Reject clusters with fewer localizations than this threshold. '
            'Set to 1 to disable.'
        )

        # aggregate superimposed data for batch processing
        if self._batch_mode:
            self.aggregate = QtWidgets.QCheckBox('Aggregate superimposed data?')
            self.aggregate.setChecked(True)
            self.aggregate.setToolTip(
                'If checked, all data will be aggregated into a single plot.'
            )
            self.aggregate.setVisible(True)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow('Mode:', self.mode_box)
        layout.addWidget(self.is_rasterized)
        layout.addRow('Render mode:', self.render_box)
        layout.addRow('Pixel size:', self.pixel_size)
        layout.addRow('Extent:', self.extent)
        layout.addRow('Center Mode:', self.centering)
        layout.addRow('Rejection Threshold:', self.rejection_threshold)
        layout.addRow('Minimum Points:', self.min_points)
        if self._batch_mode:
            layout.addWidget(self.aggregate)

        self.is_rasterized.checkStateChanged.connect(self.is_rasterized_toggled)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def setRowVisibility(self, rows: list[int], visible: bool):
        layout: QtWidgets.QFormLayout = self.layout()
        for row in rows:
            layout.setRowVisible(row, visible)

    def is_rasterized_toggled(self):
        is_rasterized = self.is_rasterized.isChecked()

        rows = [2, 3, 4]

        self.setRowVisibility(rows, is_rasterized)

    def mode_changed(self):
        mode = self.mode_box.currentData()
        is_superimposed = mode == 'superimposed'

        rows = [1, 2, 3, 4, 5, 6, 7]

        self.setRowVisibility(rows, is_superimposed)

    def values(self):
        return {
            'mode': self.mode_box.currentData(),
            'rasterized': self.is_rasterized.isChecked(),
            'render_mode': self.render_box.currentData(),
            'pixel_size': self.pixel_size.value(),
            'center_mode': self.centering.currentData(),
            'extent': self.extent.value(),
            'rejection_threshold': self.rejection_threshold.value()
            if self.rejection_threshold.value() > 0
            else None,
            'min_points': self.min_points.value()
            if self.min_points.value() > 1
            else None,
            'aggregate': self.aggregate.isChecked()
            if hasattr(self, 'aggregate')
            else False,
        }


class RulerPlotOptionsDialog(ClusterPlotOptionsDialog):
    def __init__(self, parent=None, batch_mode=False):
        super().__init__(parent, batch_mode=batch_mode)
        self.setWindowTitle('Ruler Plot Options')

        self.pixel_size.setValue(1)
        self.extent.setValue(400)

        # Add additional options specific to ruler plotting
        self.n_sites = create_spin_box(min_value=2, max_value=10, initial_value=3)
        self.n_sites.setToolTip('Number of sites in the ruler.')

        self.distance_threshold = create_double_spin_box(
            min_value=0.0, max_value=1000.0, decimals=2, initial_value=105.0
        )
        self.distance_threshold.setToolTip(
            'Maximum distance between clusters to be considered part of the same ruler.'
        )

        self.grouping_method = QtWidgets.QComboBox()
        self.grouping_method.addItem('DBSCAN', 'dbscan')
        self.grouping_method.addItem('HDBSCAN', 'hdbscan')
        self.grouping_method.addItem('OPTICS', 'optics')


        self.ruler_spacing = create_double_spin_box(
            min_value=0.0, max_value=1000.0, decimals=2, initial_value=80.0
        )
        self.ruler_spacing.setToolTip('Expected spacing between sites in the ruler.')

        self.spacing_tolerance = create_double_spin_box(
            min_value=0.0, max_value=1000.0, decimals=2, initial_value=25.0
        )
        self.spacing_tolerance.setToolTip(
            'Tolerance for the expected spacing between sites in the ruler.'
        )

        self.max_y_dev = create_double_spin_box(
            min_value=0.0, max_value=1000.0, decimals=2, initial_value=20.0
        )
        self.max_y_dev.setToolTip(
            'Maximum deviation in the y-direction for sites in the ruler.'
        )

        self.scale_data = QtWidgets.QCheckBox('Scale data?')
        self.scale_data.setChecked(False)
        self.scale_data.setToolTip(
            'If checked, data will be scaled to fit within the plot area.'
        )

        self.angle_hist = QtWidgets.QCheckBox('Show angle histogram?')
        self.angle_hist.setChecked(False)
        self.angle_hist.setToolTip(
            'If checked, an angle histogram will be displayed for the rulers.'
        )

        self.intensity_hist = QtWidgets.QCheckBox('Show intensity histogram?')
        self.intensity_hist.setChecked(False)
        self.intensity_hist.setToolTip(
            'If checked, an intensity histogram will be displayed for the rulers.'
        )

        layout: QtWidgets.QFormLayout = self.layout()
        layout.insertRow(8, 'Number of Sites:', self.n_sites)
        layout.insertRow(9, 'Grouping Method:', self.grouping_method)
        layout.insertRow(10, 'Distance Threshold:', self.distance_threshold)
        layout.insertRow(11, 'Ruler Spacing:', self.ruler_spacing)
        layout.insertRow(12, 'Spacing Tolerance:', self.spacing_tolerance)
        layout.insertRow(13, 'Maximum Y Deviation:', self.max_y_dev)
        layout.insertRow(15, self.scale_data)
        layout.insertRow(16, self.angle_hist)
        layout.insertRow(17, self.intensity_hist)

    def values(self):
        base_values = super().values()
        base_values.update(
            {
                'n_sites': self.n_sites.value(),
                'grouping_method': self.grouping_method.currentData(),
                'distance_threshold': self.distance_threshold.value(),
                'ruler_spacing': self.ruler_spacing.value(),
                'spacing_tolerance': self.spacing_tolerance.value(),
                'max_y_dev': self.max_y_dev.value(),
                'scale_data': self.scale_data.isChecked(),
                'angle_hist': self.angle_hist.isChecked(),
                'intensity_hist': self.intensity_hist.isChecked(),
            }
        )
        return base_values


class DBSCAN_OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('DBSCAN Options')

        self.sigma_xy = create_double_spin_box(
            min_value=0.1, max_value=100, decimals=2, initial_value=15.0
        )
        self.sigma_z = create_double_spin_box(
            min_value=0.1, max_value=100, decimals=2, initial_value=45.0
        )

        self.eps = create_double_spin_box(
            min_value=0.1, max_value=5000, decimals=2, initial_value=5.0
        )
        self.min_samples = create_spin_box(min_value=2, max_value=1000, initial_value=5)
        self.leaf_size = create_spin_box(min_value=0, max_value=1000, initial_value=30)

        self.algorithm = QtWidgets.QComboBox()
        algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for alg in algos:
            self.algorithm.addItem(alg)

        self.scale_data = QtWidgets.QCheckBox('Scale data?')
        self.scale_data.setChecked(False)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow('Sigma XY [nm]:', self.sigma_xy)
        layout.addRow('Sigma Z [nm]:', self.sigma_z)
        layout.addRow('Epsilon (eps):', self.eps)
        layout.addRow('Minimum Samples:', self.min_samples)
        layout.addRow('Leaf Size:', self.leaf_size)
        layout.addRow('Algorithm:', self.algorithm)
        layout.addWidget(self.scale_data)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def toDict(self):
        return {
            'eps': self.eps.value(),
            'metric': 'euclidean',
            'min_samples': self.min_samples.value(),
            'leaf_size': self.leaf_size.value(),
            'algorithm': self.algorithm.currentText(),
            'sigma_xy': self.sigma_xy.value(),
            'sigma_z': self.sigma_z.value(),
            'scale_data': self.scale_data.isChecked(),
        }


class HDBSCAN_OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('HDBSCAN Options')

        self.sigma_xy = create_double_spin_box(
            min_value=0.1, max_value=100, decimals=2, initial_value=15.0
        )
        self.sigma_z = create_double_spin_box(
            min_value=0.1, max_value=100, decimals=2, initial_value=45.0
        )

        self.min_samples = create_spin_box(min_value=0, max_value=1000, initial_value=0)
        self.min_cluster_size = create_spin_box(
            min_value=2, max_value=1000, initial_value=5
        )
        self.cluster_selection_epsilon = create_double_spin_box(
            min_value=0.0, max_value=1000.0, decimals=2, initial_value=0.0
        )

        self.scale_data = QtWidgets.QCheckBox('Scale data?')
        self.scale_data.setChecked(False)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow('Sigma XY [nm]:', self.sigma_xy)
        layout.addRow('Sigma Z [nm]:', self.sigma_z)
        layout.addRow('Minimum Samples:', self.min_samples)
        layout.addRow('Minimum Cluster Size:', self.min_cluster_size)
        layout.addRow('Cluster Selection Epsilon:', self.cluster_selection_epsilon)
        layout.addWidget(self.scale_data)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def toDict(self):
        return {
            'min_samples': self.min_samples.value()
            if self.min_samples.value() > 0
            else None,
            'min_cluster_size': self.min_cluster_size.value(),
            'sigma_xy': self.sigma_xy.value(),
            'sigma_z': self.sigma_z.value(),
            'scale_data': self.scale_data.isChecked(),
            'cluster_selection_epsilon_max': self.cluster_selection_epsilon.value(),
        }


class OPTICS_OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('OPTICS Options')

        self.sigma_xy = create_double_spin_box(
            min_value=0.1, max_value=100, decimals=2, initial_value=15.0
        )
        self.sigma_z = create_double_spin_box(
            min_value=0.1, max_value=100, decimals=2, initial_value=45.0
        )

        self.min_samples = create_spin_box(min_value=2, max_value=1000, initial_value=5)
        self.max_eps = create_double_spin_box(
            min_value=0.0, max_value=5000, decimals=2, initial_value=0.0
        )

        self.cluster_method = QtWidgets.QComboBox()
        methods = ['xi', 'dbscan']
        for method in methods:
            self.cluster_method.addItem(method)

        self.xi = create_double_spin_box(
            min_value=0.0, max_value=1.0, decimals=2, initial_value=0.05
        )

        self.algorithm = QtWidgets.QComboBox()
        algos = ['auto', 'ball_tree', 'kd_tree', 'brute']
        for alg in algos:
            self.algorithm.addItem(alg)

        self.leaf_size = create_spin_box(min_value=0, max_value=1000, initial_value=30)

        self.scale_data = QtWidgets.QCheckBox('Scale data?')
        self.scale_data.setChecked(False)

        layout = QtWidgets.QFormLayout(self)
        layout.addRow('Sigma XY [nm]:', self.sigma_xy)
        layout.addRow('Sigma Z [nm]:', self.sigma_z)
        layout.addRow('Minimum Samples:', self.min_samples)
        layout.addRow('Maximum Epsilon (max_eps):', self.max_eps)
        layout.addRow('Metric:', QtWidgets.QLabel('minkowski (p=2)'))
        layout.addRow('Cluster Method:', self.cluster_method)
        layout.addRow('Xi:', self.xi)
        layout.addRow('Algorithm:', self.algorithm)
        layout.addRow('Leaf Size:', self.leaf_size)
        layout.addWidget(self.scale_data)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def toDict(self):
        return {
            'algorithm': self.algorithm.currentText(),
            'min_samples': self.min_samples.value(),
            'max_eps': self.max_eps.value() if self.max_eps.value() > 0 else np.inf,
            'metric': 'minkowski',
            'p': 2.0,
            'cluster_method': self.cluster_method.currentText(),
            'sigma_xy': self.sigma_xy.value(),
            'sigma_z': self.sigma_z.value(),
            'scale_data': self.scale_data.isChecked(),
            'xi': self.xi.value(),
            'leaf_size': self.leaf_size.value(),
        }


def add_plot_result(
    graphics_layout: pg.GraphicsLayoutWidget,
    result: PlotResult,
    row: int = 0,
    col: int = 0,
    **kwargs,
) -> int:
    '''
    Place a PlotResult into a GraphicsLayoutWidget.

    Layout (rows consumed, starting at `row`):
      row+0 : raster ViewBox  OR  scatter PlotItem
      row+1 : HistogramLUTItem  (only when rasterized)
      row+2 : X-projection PlotItem  (only when histogram_x present)
      row+3 : Y-projection PlotItem  (only when histogram_y present)

    Returns the next free row index so callers can stack results.
    '''
    next_row = row

    # ── 2D render ─────────────────────────────────────────────────────────────
    if result.is_rasterized:
        plot_item = add_image_item(
            graphics_layout, result.raster, row=next_row, column=col, **kwargs
        )
        next_row += 2  # raster + histogram

        # Ruler site lines — only drawn when n_sites + ruler_spacing provided
        n_sites = kwargs.get('n_sites')
        ruler_spacing = kwargs.get('ruler_spacing')
        if n_sites is not None and ruler_spacing is not None and plot_item is not None:
            offsets = (np.arange(n_sites) - (n_sites - 1) / 2.0) * ruler_spacing
            for offset in offsets:
                x_pos = result.raster.image.shape[1] / 2 + offset / result.pixel_size
                v_line = pg.InfiniteLine(
                    angle=90,
                    pen=pg.mkPen(color='g', style=QtCore.Qt.PenStyle.DashLine),
                )
                v_line.setX(x_pos)
                plot_item.addItem(v_line)
    # ── scatter render ─────────────────────────────────────────────────────────
    elif result.scatter:
        plot_item = graphics_layout.ci.addPlot(
            title=result.title or 'Clusters', row=next_row, col=col
        )
        plot_item.showGrid(x=True, y=True, alpha=0.3)
        plot_item.getViewBox().setAspectLocked(True, 1)
        plot_item.getViewBox().invertY(True)
        plot_item.setLabel('top', 'X [nm]')
        plot_item.setLabel('left', 'Y [nm]')
        for layer in result.scatter:
            plot_item.addItem(layer)
        next_row += 1

    # ── projections ────────────────────────────────────────────────────────────
    if result.histogram_x:
        px = graphics_layout.ci.addPlot(title='X projection', row=next_row, col=col)
        px.setLabel('bottom', 'X position (nm)')
        px.setLabel('left', 'Density')
        px.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_x:
            px.addItem(item)
        next_row += 1

    if result.histogram_y:
        py = graphics_layout.ci.addPlot(title='Y projection', row=next_row, col=col)
        py.setLabel('bottom', 'Y position (nm)')
        py.setLabel('left', 'Density')
        py.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_y:
            py.addItem(item)
        next_row += 1

    if result.histogram_intensity:
        pi = graphics_layout.ci.addPlot(
            title='Intensity Histogram', row=next_row, col=col
        )
        pi.setLabel('bottom', 'Intensity (a.u.)')
        pi.setLabel('left', 'Density')
        pi.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_intensity:
            pi.addItem(item)
        next_row += 1

    if result.histogram_angle:
        pa = graphics_layout.ci.addPlot(title='Angle Histogram', row=next_row, col=col)
        pa.setLabel('bottom', 'Angle (degrees)')
        pa.setLabel('left', 'Density')
        pa.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_angle:
            pa.addItem(item)
        next_row += 1

    return next_row


def add_image_item(
    graphics_layout: pg.GraphicsLayoutWidget,
    image_item: pg.ImageItem,
    row: int = 0,
    column: int = 0,
    **kwargs,
):
    if image_item is not None:
        plot_item = graphics_layout.ci.addViewBox(row=row, col=column, invertY=True)

        plot_item.setAspectLocked(True)
        plot_item.setAutoVisible(True)
        plot_item.enableAutoRange()

        # title is width x height of image in pixels
        image_shape = image_item.image.shape

        text = f'{image_shape[1]} x {image_shape[0]} pixels'

        if hasattr(image_item, 'param'):
            params = image_item.param
            text += (
                f'\nFitted Parameters:\n'
                f'X: {params[0]:.2f} pixel\n'
                f'Y: {params[1]:.2f} pixel\n'
                f'Amplitude: {params[2]:.2f}\n'
                f'Offset: {params[3]:.2f}\n'
                f'Sigma X: {params[4] * image_item.pixel_size:.2f} nm\n'
                f'Sigma Y: {params[5] * image_item.pixel_size:.2f} nm'
            )

        title_item = pg.TextItem(text)

        plot_item.addItem(image_item)

        plot_item.addItem(title_item)

        if hasattr(image_item, 'pixel_size'):
            # add scale bar to the bottom right c`orner of the image
            scalebar_length = kwargs.get('scalebar_length', 10.0)  # in nm
            scalebar_length_pixels = scalebar_length / image_item.pixel_size
            # add a line to the bottom right corner of the image

            scaleBar = pg.ScaleBar(
                size=scalebar_length_pixels,
                suffix=f'pixels ({scalebar_length:.0f} nm)',
                brush=pg.mkBrush('w'),
                offset=(-20, -20),
            )
            scaleBar.setParentItem(plot_item)

        histogram_item = pg.HistogramLUTItem(
            gradientPosition='bottom', orientation='horizontal'
        )
        histogram_item.setImageItem(image_item)
        # set to jet colormap
        histogram_item.gradient.setColorMap(pg.colormap.getFromMatplotlib('jet'))

        h_line = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(color='y', style=QtCore.Qt.PenStyle.DotLine),
        )
        v_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(color='y', style=QtCore.Qt.PenStyle.DotLine),
        )
        h_line.setY(image_item.image.shape[0] / 2)
        v_line.setX(image_item.image.shape[1] / 2)

        plot_item.addItem(h_line)
        plot_item.addItem(v_line)

        graphics_layout.ci.addItem(histogram_item, row=row + 1, col=column)

        # add export button to export image as tiff to context menu of plot_item
        export_action = QAction('Export Image as TIFF', plot_item)
        plot_item.menu.addAction(export_action)

        tiff_filename = getattr(image_item, 'tiff_filename', None)
        export_action.triggered.connect(
            lambda: export_image_as_tiff(image_item.image, base_name=tiff_filename)
        )

        return plot_item
