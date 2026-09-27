import logging
import traceback

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from sklearn.neighbors import NearestNeighbors

from microEye.analysis.clustering.core import (
    ClusterCenterResult,
    get_center,
    get_centers,
)
from microEye.analysis.clustering.dataclasses import (
    CenteringMode,
    CenteringOptions,
    LabeledLocalizations,
    PlotResult,
)
from microEye.analysis.fitting.pyfit3Dcspline import CPUmleFit_LM
from microEye.analysis.fitting.results import FittingResults
from microEye.analysis.rendering import BaseRenderer, RenderModes
from microEye.qt import (
    QAction,
    QtCore,
    QtGui,
    getSaveFileName,
)

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
            scatter_layers = []
            # add 0 gray background layer for noise points
            if kwargs.get('add_noise_background', False) and np.any(labels == -1):
                scatter = pg.ScatterPlotItem(
                    x=x[labels == -1],
                    y=y[labels == -1],
                    size=3,
                    pen=None,
                    brush=pg.mkBrush(pg.mkColor(128, 128, 128, 32)),
                    pxMode=True,
                )
                scatter_layers.append(scatter)

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
        cls, x: np.ndarray, bin_size: float = 1.0, **kwargs
    ) -> list[pg.GraphicsObject]:

        res = []
        axis_label = kwargs.get('axis_label', 'X')

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

            curve = pg.PlotCurveItem(
                x=centers,
                y=gaussian_filter1d(counts, sigma=bin_size),
                pen=pg.mkPen(cls.KDE_COLOR, width=2),
            )
            res.append(curve)
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
            axis_label='X',
            **kwargs,
        )

        hist_y = (
            cls.histogram(
                y,
                bin_size=bin_size,
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
            centering_options = CenteringOptions(
                center_mode=kwargs.pop('center_mode', CenteringMode.CRLB_WEIGHTED_MEAN),
                eps=kwargs.pop('eps', 1e-12),
                rejection_threshold=kwargs.pop('rejection_threshold'),
                min_points=kwargs.pop('min_points'),
                max_points=kwargs.pop('max_points', None),
                subtract_com=True,
            )
            centers_result = get_centers(
                x,
                y,
                z,
                labels,
                intensity,
                crlb_x,
                crlb_y,
                options=centering_options,
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
        fitting_results: list[LabeledLocalizations],
        **kwargs,
    ) -> list[PlotResult]:
        results = []
        for i, result in enumerate(fitting_results):
            pr = cls.superimposed(
                result.x,
                result.y,
                result.z,
                result.intensity,
                result.labels,
                crlb_x=result.crlb_x,
                crlb_y=result.crlb_y,
                **kwargs,
            )
            results.append(pr)
            logger.info('Superimposed (%d/%d)', i + 1, len(fitting_results))
        return results

    @classmethod
    def superimposed_batch_aggregate(
        cls,
        fitting_results: list[LabeledLocalizations],
        **kwargs,
    ) -> PlotResult:
        '''Concatenate all tables first, then superimpose once.'''
        try:
            is_z = any(r.z is not None for r in fitting_results)
            x, y, z, labels, intensity, crlb_x, crlb_y = [], [], [], [], [], [], []
            max_label = 0

            for result in fitting_results:
                x.append(result.x)
                y.append(result.y)
                if result.z is not None:
                    z.append(result.z)

                ids = result.labels.copy()
                ids[ids >= 0] += max_label
                labels.append(ids)
                max_label = int(ids.max()) + 1

                intensity.append(result.intensity)
                crlb_x.append(result.crlb_x)
                crlb_y.append(result.crlb_y)

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
        fitting_results: list[LabeledLocalizations],
        **kwargs,
    ) -> list[PlotResult]:
        try:
            scatter_plots = []

            for i, result in enumerate(fitting_results):
                x = result.x
                y = result.y
                z = result.z if result.z is not None else None
                labels = result.labels

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
    def render_clusters_batch(
        cls, tables: list[FittingResults] | list[LabeledLocalizations], **kwargs
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

            labeled_tables = []
            for result in tables:
                if isinstance(result, FittingResults):
                    val: LabeledLocalizations = result.toClusterRender()
                elif isinstance(result, LabeledLocalizations):
                    val = result
                else:
                    logger.warning(
                        f'Unsupported table type: {type(result)}. Skipping.'
                    )
                    continue

                if val is not None:
                    labeled_tables.append(val)

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
                    res = ClustersScatter.superimposed_batch(
                        labeled_tables, **worker_kwargs
                    )
                else:
                    res.append(
                        ClustersScatter.superimposed_batch_aggregate(
                            labeled_tables, **worker_kwargs
                        )
                    )
            else:
                worker_kwargs = {}
                res = ClustersScatter.scatter_batch(labeled_tables, **worker_kwargs)

        except Exception:
            traceback.print_exc()

        return res

    @classmethod
    def rasterize_instances(cls, concatenated: dict[str, np.ndarray], **kwargs):
        """
        Generic rasterizer for any pooled/aligned model output.
        Data is already centered on each instance's canonical origin by the
        fitting model, so no extra recentering is needed here.
        """
        item, _ = cls.rasterized_2d(
            concatenated['x'],
            concatenated['y'],
            concatenated['z'],
            intensity=concatenated['intensity'],
            labels=concatenated['labels'],
            crlb_x=concatenated['crlb_x'],
            crlb_y=concatenated['crlb_y'],
            skip_centering=True,
            **kwargs,
        )
        return item

    @classmethod
    def _render_instances(
        cls, concatenated: dict[str, np.ndarray], **kwargs
    ) -> PlotResult:

        item = cls.rasterize_instances(concatenated, **kwargs)
        hist_x, hist_y = (
            cls._build_histograms(
                concatenated['x'],
                concatenated['y'],
                bin_size=kwargs.get('pixel_size', 1.0),
                symmetrize=kwargs.get('symmetrize', False),
                **kwargs,
            )
            if kwargs.get('line_profiles', True)
            else ([], [])
        )

        hist_intensity = (
            cls.histogram(
                concatenated['intensity'],
                bin_size=kwargs.get('intensity_bin_size', 1),
                axis_label='Intensity [a.u.]',
                **kwargs,
            )
            if kwargs.get('intensity_histogram', False)
            else None
        )

        if kwargs.get('angle_histogram', False):
            # rad to deg conversion and conversion to range [-90, 90]
            angles = np.degrees(concatenated['angle'])
            min_angle = angles.min()
            max_angle = angles.max()
            angles = (angles + 90) % 180 - 90 if max_angle - min_angle > 180 else angles
            hist_angle = cls.histogram(
                angles,
                bin_size=kwargs.get('angle_bin_size', 1),
                axis_label='Angle [deg]',
                **kwargs,
            )
        else:
            hist_angle = None

        hist_n_matched = (
            cls.histogram(
                concatenated['n_matched'],
                bin_size=kwargs.get('n_matched_bin_size', 1),
                axis_label='N matched',
                **kwargs,
            )
            if kwargs.get('n_matched_histogram', False)
            else None
        )

        return PlotResult(
            raster=item,
            histogram_x=hist_x,
            histogram_y=hist_y,
            histogram_intensity=hist_intensity,
            histogram_angle=hist_angle,
            histogram_n_matched=hist_n_matched,
            title=kwargs.get('title', 'Superimposed instances'),
            pixel_size=kwargs.get('pixel_size', 1.0),
        )


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
        px.setLabel('bottom', 'X position [nm]')
        px.setLabel('left', 'Density')
        px.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_x:
            px.addItem(item)
        next_row += 1

    if result.histogram_y:
        py = graphics_layout.ci.addPlot(title='Y projection', row=next_row, col=col)
        py.setLabel('bottom', 'Y position [nm]')
        py.setLabel('left', 'Density')
        py.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_y:
            py.addItem(item)
        next_row += 1

    if result.histogram_intensity:
        pi = graphics_layout.ci.addPlot(
            title='Intensity Histogram', row=next_row, col=col
        )
        pi.setLabel('bottom', 'Intensity [a.u.]')
        pi.setLabel('left', 'Density')
        pi.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_intensity:
            pi.addItem(item)
        next_row += 1

    if result.histogram_angle:
        pa = graphics_layout.ci.addPlot(title='Angle Histogram', row=next_row, col=col)
        pa.setLabel('bottom', 'Angle [degrees]')
        pa.setLabel('left', 'Density')
        pa.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_angle:
            pa.addItem(item)
        next_row += 1

    if result.histogram_n_matched:
        pn = graphics_layout.ci.addPlot(
            title='N Matched Histogram', row=next_row, col=col
        )
        pn.setLabel('bottom', 'N matched')
        pn.setLabel('left', 'Density')
        pn.showGrid(x=True, y=True, alpha=0.2)
        for item in result.histogram_n_matched:
            pn.addItem(item)
        next_row += 1

    return next_row


def add_image_item(
    graphics_layout: pg.GraphicsLayoutWidget,
    image_item: pg.ImageItem,
    row: int = 0,
    column: int = 0,
    **kwargs,
) -> pg.ViewBox | None:
    if image_item is None or image_item.image is None:
        return None

    plot_item = graphics_layout.ci.addViewBox(row=row, col=column, invertY=True)
    plot_item.setAspectLocked(True)
    plot_item.setAutoVisible(True)
    plot_item.enableAutoRange()
    plot_item.addItem(image_item)

    # title is width x height of image in pixels
    h, w = image_item.image.shape[:2]
    pixel_size = getattr(image_item, 'pixel_size', 1.0)
    text_lines = [f'{w} x {h} pixels']

    spatial_width = w * pixel_size
    spatial_height = h * pixel_size

    image_item.setRect(QtCore.QRectF(0, 0, spatial_width, spatial_height))

    params = getattr(image_item, 'param', None)
    if params is not None and len(params) >= 6:
        text_lines.extend(
            [
                '\nFitted Parameters:',
                f'X: {params[0]:.2f} pixel',
                f'Y: {params[1]:.2f} pixel',
                f'Amplitude: {params[2]:.2f}',
                f'Offset: {params[3]:.2f}',
                f'Sigma X: {params[4] * pixel_size:.2f} nm',
                f'Sigma Y: {params[5] * pixel_size:.2f} nm',
            ]
        )

    title_item = pg.TextItem('\n'.join(text_lines), color='w', anchor=(0, 0))
    plot_item.addItem(title_item)

    if hasattr(image_item, 'pixel_size'):
        scalebar_length_nm = kwargs.get('scalebar_length', 10.0)
        # Size expects the dimensions in the ViewBox's data coordinates
        scalebar_pixels = scalebar_length_nm / pixel_size

        scale_bar = pg.ScaleBar(
            size=scalebar_length_nm,
            suffix=f'nm ({scalebar_pixels:.0f} px)',
            brush=pg.mkBrush('w'),
            offset=(-20, -20),
        )
        scale_bar.setParentItem(plot_item)

    histogram_item = pg.HistogramLUTItem(
        gradientPosition='bottom', orientation='horizontal'
    )
    histogram_item.setImageItem(image_item)
    histogram_item.gradient.setColorMap(pg.colormap.getFromMatplotlib('jet'))
    graphics_layout.ci.addItem(histogram_item, row=row + 1, col=column)

    reference_lines = kwargs.get('reference_lines')
    if reference_lines is not None:
        fit_and_plot_rois(
            plot_item,
            image_item.image,
            roi_size=kwargs.get('roi_size', 21),
            pixel_size=image_item.pixel_size,
            reference_lines=reference_lines,
        )

    # add export button to export image as tiff to context menu of plot_item
    if hasattr(plot_item, 'menu') and plot_item.menu is not None:
        export_action = QAction('Export Image as TIFF', plot_item)
        tiff_filename = getattr(image_item, 'tiff_filename', None)
        export_action.triggered.connect(
            lambda: export_image_as_tiff(image_item.image, base_name=tiff_filename)
        )
        plot_item.menu.addAction(export_action)

    return plot_item


def extract_rois(
    image: np.ndarray, rois: np.ndarray, roi_size: int = 60
) -> tuple[list[np.ndarray], list[list[int]]]:
    if image is None or rois is None or not len(rois):
        return [], []
    h, w = image.shape[:2]

    hs = roi_size // 2
    max_y, max_x = max(0, h - roi_size), max(0, w - roi_size)

    patches, starts = [], []
    for x_off, y_off in rois:
        # Compute exact start pixels and clamp safely inside image boundaries
        y_start = max(0, min(int(h // 2 + y_off) - hs, max_y))
        x_start = max(0, min(int(w // 2 + x_off) - hs, max_x))

        # Slice (using ... keeps it compatible with color channels if present)
        patches.append(
            image[
                y_start : y_start + roi_size, x_start : x_start + roi_size, ...
            ].flatten()
        )
        starts.append([x_start, y_start])

    return patches, np.array(starts, dtype=int)


def reference_lines_to_points(reference_lines: dict) -> np.ndarray:
    '''
    Convert reference lines to points for plotting.

    Parameters
    ----------
    reference_lines : dict
        Dictionary containing reference lines with keys `x` and `y`,
        each mapping to a list of line positions.
    '''
    x = np.array(reference_lines.get('x', []), dtype=int)
    y = np.array(reference_lines.get('y', []), dtype=int)

    if x.size == 0 and y.size == 0:
        return np.empty((0, 2), dtype=int)

    if x.size == 0:
        x = np.array([0], dtype=int)
    elif y.size == 0:
        y = np.array([0], dtype=int)

    # make points of intersection of each x and y line
    points = np.array(np.meshgrid(x, y), dtype=int).T.reshape(-1, 2)
    return points


def fit_and_plot_rois(
    plot_item: pg.ViewBox,
    image: np.ndarray,
    roi_size: int,
    reference_lines: dict,
    pixel_size: float = 1.0,
):
    '''
    Fit Gaussian peaks to the extracted ROIs and plot the
    fitted peaks on the given plot item.
    '''
    rois = reference_lines_to_points(reference_lines) / pixel_size

    roi_size = min(roi_size, image.shape[0], image.shape[1])

    if len(rois) == 1:
        patches = [image.flatten()]

        anchors = np.array([[0, 0]])
    else:
        patches, anchors = extract_rois(image, rois, roi_size=roi_size)

    if not patches:
        return

    # Call MLE fitting routine
    params, _, _ = CPUmleFit_LM(
        np.array(patches, dtype=np.float32),
        4,
        np.array([5], dtype=np.float32),
        None,
        0,
    )
    if params is None:
        return

    colors = ['w', 'g', 'b']

    x_local_fit = params[:, 0]
    y_local_fit = params[:, 1]

    x_global = (anchors[:, 0] + x_local_fit + 0.5) * pixel_size
    y_global = (anchors[:, 1] + y_local_fit + 0.5) * pixel_size

    nn = NearestNeighbors(n_neighbors=min(5, len(x_global)))
    nn.fit(np.column_stack((x_global, y_global)))

    # Scale the global coordinates by the pixel size
    for i in range(len(patches)):
        color = colors[i % len(colors)]
        scatter = pg.ScatterPlotItem(
            x=[x_global[i]],
            y=[y_global[i]],
            size=10,
            pen=pg.mkPen(color=color, width=2),
            brush=pg.mkBrush(color=color),
            pxMode=True,
        )
        plot_item.addItem(scatter)

        distances, indices = nn.kneighbors([[x_global[i], y_global[i]]])

        median_distance = np.median(distances)
        # show distance label between the current point and its nearest neighbor
        for j in range(1, len(indices[0])):
            neighbor_index = indices[0][j]
            if neighbor_index != i and distances[0][j] <= median_distance * 1.1:
                draw_distance_label(
                    plot_item,
                    x_global[i],
                    y_global[i],
                    x_global[neighbor_index],
                    y_global[neighbor_index],
                )


def draw_distance_label(plot_item: pg.ViewBox, x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    txt = pg.TextItem(f'{distance:.1f}', color='w', anchor=(0.5, 0.5))
    txt.setPos((x1 + x2) / 2, (y1 + y2) / 2)
    plot_item.addItem(txt)
