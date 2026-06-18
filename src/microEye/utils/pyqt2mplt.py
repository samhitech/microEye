from __future__ import annotations

import logging
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

import matplotlib
import numpy as np
from cycler import cycler
from matplotlib import rc_context
from matplotlib import style as mpl_style
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qtagg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from pyqtgraph.parametertree import Parameter

from microEye.qt import QtCore, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree

logger = logging.getLogger(__name__)

PlotType = Literal['line', 'scatter', 'bar', 'histogram', 'errorbar']


@dataclass(slots=True)
class PlotSeriesPayload:
    '''Generic series payload consumed by the Matplotlib bridge.'''

    x: Iterable[float] | np.ndarray | None
    y: Iterable[float] | np.ndarray | None = None
    plot_type: PlotType = 'line'
    label: str = ''
    dataset: str = 'default'
    style: dict[str, Any] = field(default_factory=dict)
    visible: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubplotPayload:
    '''Generic subplot payload with fully prepared display metadata.'''

    row: int
    col: int
    title: str = ''
    xlabel: str = ''
    ylabel: str = ''
    xscale: str = 'linear'
    yscale: str = 'linear'
    series: list[PlotSeriesPayload] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FigurePayload:
    '''Container for all subplots to render in the Matplotlib dialog.'''

    title: str = ''
    subplots: list[SubplotPayload] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class PlotterSettings(Tree):
    renderActivated = Signal()
    saveActivated = Signal()

    def __init__(self, parent=None):
        self._plot_labels: list[str] = ['(none)']
        self._dataset_labels: list[str] = []
        self._subplot_defaults: list[SubplotPayload] = []
        self._metric_labels: list[str] = []
        self._reference_labels: list[str] = []
        self._curve_types: list[str] = []
        self._compare_datasets: list[str] = []
        super().__init__(parent)

    def create_parameters(self):
        params = [
            {
                'name': 'Figure',
                'type': 'group',
                'children': [
                    {'name': 'Width (in)', 'type': 'float', 'value': 11.0},
                    {'name': 'Height (in)', 'type': 'float', 'value': 7.0},
                    {'name': 'DPI', 'type': 'int', 'value': 100},
                    {'name': 'Title', 'type': 'str', 'value': ''},
                    {
                        'name': 'Title Alignment',
                        'type': 'list',
                        'limits': ['left', 'center', 'right'],
                        'value': 'center',
                    },
                    {'name': 'Title Bold', 'type': 'bool', 'value': True},
                    {
                        'name': 'Font Family',
                        'type': 'list',
                        'limits': [
                            'DejaVu Sans',
                            'Arial',
                            'Helvetica',
                            'Liberation Sans',
                            'Times New Roman',
                            'DejaVu Serif',
                        ],
                        'value': 'DejaVu Sans',
                    },
                    {
                        'name': 'Debounce (ms)',
                        'type': 'int',
                        'value': 150,
                        'limits': (100, 1000),
                    },
                    {
                        'name': 'Render Now',
                        'type': 'action',
                    },
                    {
                        'name': 'Save Figure',
                        'type': 'action',
                    },
                ],
            },
            {
                'name': 'Layout',
                'type': 'group',
                'children': [
                    {'name': 'Rows', 'type': 'int', 'value': 2, 'limits': (1, 20)},
                    {'name': 'Cols', 'type': 'int', 'value': 2, 'limits': (1, 20)},
                    {'name': 'Row Spacing', 'type': 'float', 'value': 0.3},
                    {'name': 'Column Spacing', 'type': 'float', 'value': 0.25},
                    {'name': 'Left Margin', 'type': 'float', 'value': 0.05},
                    {'name': 'Right Margin', 'type': 'float', 'value': 0.98},
                    {'name': 'Top Margin', 'type': 'float', 'value': 0.9},
                    {'name': 'Bottom Margin', 'type': 'float', 'value': 0.1},
                    {'name': 'Share X', 'type': 'bool', 'value': False},
                    {'name': 'Share Y', 'type': 'bool', 'value': False},
                ],
            },
            {
                'name': 'Style',
                'type': 'group',
                'children': [
                    {
                        'name': 'Theme',
                        'type': 'list',
                        'limits': [
                            'default',
                            'seaborn-v0_8-whitegrid',
                            'seaborn-v0_8-darkgrid',
                            'ggplot',
                            'bmh',
                            'classic',
                        ],
                        'value': 'default',
                    },
                    {
                        'name': 'Color Cycle',
                        'type': 'list',
                        'limits': [
                            'tab10',
                            'tab20',
                            'Set1',
                            'Set2',
                            'Dark2',
                            'Accent',
                        ],
                        'value': 'tab10',
                    },
                    {'name': 'Grid', 'type': 'bool', 'value': True},
                    {'name': 'Legend', 'type': 'bool', 'value': True},
                    {
                        'name': 'Legend Location',
                        'type': 'list',
                        'limits': [
                            'best',
                            'upper right',
                            'upper left',
                            'lower left',
                            'lower right',
                            'center left',
                            'center right',
                            'lower center',
                            'upper center',
                            'center',
                        ],
                        'value': 'best',
                    },
                    {'name': 'Line Width', 'type': 'float', 'value': 1.6},
                    {'name': 'Marker Size', 'type': 'float', 'value': 5.0},
                    {'name': 'Font Size', 'type': 'int', 'value': 10},
                    {'name': 'Left title margin', 'type': 'float', 'value': -0.1},
                    {'name': 'Right title margin', 'type': 'float', 'value': 1.0},
                    {'name': 'Title Y', 'type': 'float', 'value': 1.05},
                ],
            },
            {
                'name': 'Datasets',
                'type': 'group',
                'children': self._dataset_children(),
            },
            {
                'name': 'Subplots',
                'type': 'group',
                'children': self._subplot_children(4),
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.__connect_signals()

    def __connect_signals(self):
        render_param = self.get_param('Figure.Render Now')
        if render_param is not None:
            render_param.sigActivated.connect(self.renderActivated)

        save_param = self.get_param('Figure.Save Figure')
        if save_param is not None:
            save_param.sigActivated.connect(self.saveActivated)

    def change(self, param: Parameter, changes: list):
        self.paramsChanged.emit(param, changes)

    def _dataset_children(self) -> list[dict]:
        return self._dataset_children_from_labels(self._dataset_labels)

    def _dataset_children_from_labels(self, labels: list[str]) -> list[dict]:
        return [{'name': label, 'type': 'bool', 'value': True} for label in labels]

    def _subplot_children(self, count: int) -> list[dict]:
        return [self._subplot_group(index) for index in range(count)]

    def _subplot_group(self, index: int) -> dict:
        defaults = (
            self._subplot_defaults[index]
            if index < len(self._subplot_defaults)
            else None
        )
        default_row = (defaults.row + 1) if defaults is not None else (index // 2) + 1
        default_col = (defaults.col + 1) if defaults is not None else (index % 2) + 1
        default_title = defaults.title if defaults is not None else ''
        default_xlabel = defaults.xlabel if defaults is not None else ''
        default_ylabel = defaults.ylabel if defaults is not None else ''
        default_xtick_labels = (
            defaults.metadata.get('dataset_names', []) if defaults is not None else []
        )
        default_title_alignment = (
            defaults.metadata.get('title_alignment', 'center')
            if defaults is not None
            else 'center'
        )
        default_title_bold = (
            bool(defaults.metadata.get('title_bold', False))
            if defaults is not None
            else False
        )
        default_dataset_labels = []
        if defaults is not None:
            default_dataset_labels = sorted(
                {series.dataset for series in defaults.series if series.dataset}
            )
        default_legend = (
            bool(defaults.metadata.get('legend', True))
            if defaults is not None
            else True
        )
        return {
            'name': f'Plot {index + 1}',
            'type': 'group',
            'children': [
                {'name': 'Visible', 'type': 'bool', 'value': True},
                {'name': 'Row', 'type': 'int', 'value': default_row, 'limits': (1, 20)},
                {'name': 'Col', 'type': 'int', 'value': default_col, 'limits': (1, 20)},
                {'name': 'Title Override', 'type': 'str', 'value': default_title},
                {
                    'name': 'Title Alignment',
                    'type': 'list',
                    'limits': ['left', 'center', 'right'],
                    'value': default_title_alignment,
                },
                {'name': 'Title Bold', 'type': 'bool', 'value': default_title_bold},
                {'name': 'X Label Override', 'type': 'str', 'value': default_xlabel},
                {'name': 'Y Label Override', 'type': 'str', 'value': default_ylabel},
                {
                    'name': 'X Tick Labels',
                    'type': 'str',
                    'value': ', '.join(map(str, default_xtick_labels)),
                },
                {
                    'name': 'Series Colors',
                    'type': 'str',
                    'value': '',
                },
                {'name': 'Use Global Datasets', 'type': 'bool', 'value': False},
                {
                    'name': 'Datasets',
                    'expanded': False,
                    'type': 'group',
                    'children': self._dataset_children_from_labels(
                        default_dataset_labels
                    ),
                },
                {'name': 'Grid Override', 'type': 'bool', 'value': True},
                {'name': 'Legend Override', 'type': 'bool', 'value': default_legend},
            ],
        }

    def rebuild(
        self,
        plot_labels,
        dataset_labels,
        subplot_defaults=None,
        metric_labels=None,
        reference_labels=None,
        curve_types=None,
        compare_datasets=None,
    ):
        self._plot_labels = plot_labels or ['(none)']
        self._dataset_labels = dataset_labels or []
        self._subplot_defaults = list(subplot_defaults or [])
        self._metric_labels = metric_labels or []
        self._reference_labels = reference_labels or []
        self._curve_types = curve_types or []
        self._compare_datasets = compare_datasets or []
        self.create_parameters()
        self.setParameters(self.param_tree, showTop=False)

    def refresh_subplots(self, count: int):
        subplots = self.param_tree.param('Subplots')
        if subplots is None:
            return
        # Ensure the number of subplot parameter groups matches `count`.
        # Previously we only grew the list, which left stale entries when the
        # payload or layout requested fewer subplots. Now we add or remove
        # children to match exactly the requested count.
        current_count = len(subplots.children())
        target_count = int(count)
        if current_count < target_count:
            for index in range(current_count, target_count):
                subplots.addChild(self._subplot_group(index))
        elif current_count > target_count:
            # Remove extra children from the end to keep indexes stable.
            for _ in range(current_count - target_count):
                child = subplots.children()[-1]
                try:
                    subplots.removeChild(child)
                except Exception:
                    # If removal fails for any reason, stop attempting to avoid
                    # leaving the tree in a partial state.
                    break


class MatplotlibPlotterDialog(QtWidgets.QDialog):
    NAME = 'Matplotlib Plotter'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.NAME)
        self.resize(1600, 720)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setWindowFlags(
            self.windowFlags() | QtCore.Qt.WindowType.WindowMaximizeButtonHint
        )

        self._payload = FigurePayload()
        self._subplot_lookup: dict[tuple[int, int], SubplotPayload] = {}
        self._is_refreshing_subplots = False
        self._payload_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None

        self._refresh_timer = QtCore.QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.render_now)

        self.__init_layout()
        self._sync_settings_from_payload()
        self.schedule_render()

    def __init_layout(self):
        layout = QtWidgets.QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        self.plot_settings = PlotterSettings()
        self.plot_settings.paramsChanged.connect(self._on_settings_changed)
        self.plot_settings.renderActivated.connect(self.render_now)
        self.plot_settings.saveActivated.connect(self.save_figure)

        splitter.addWidget(self.plot_settings)

        figure_container = QtWidgets.QWidget()
        figure_layout = QtWidgets.QVBoxLayout(figure_container)
        figure_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(11, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        figure_layout.addWidget(self.toolbar)
        figure_layout.addWidget(self.canvas, 1)
        splitter.addWidget(figure_container)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    @classmethod
    def plot(cls, x, y, **kwargs) -> MatplotlibPlotterDialog:
        dialog = cls()

        series = PlotSeriesPayload(
            x,
            y,
            plot_type=kwargs.get('plot_type','scatter'),
            label=kwargs.get('label', ''),
            dataset=kwargs.get('label', ''),
            style=kwargs.get('style', {}),
            visible=kwargs.get('style', True),
            extras=kwargs.get('extras', {}),
        )

        subplot = SubplotPayload(
            row=0,
            col=0,
            title=kwargs.get('title', ''),
            xlabel=kwargs.get('xlabel', ''),
            ylabel=kwargs.get('ylabel', ''),
            series=[series],
            metadata={
                'dataset_names': [],
                'title_alignment': kwargs.get('title_alignment', 'center'),
                'title_bold': kwargs.get('title_bold', True),
                'legend': kwargs.get('legend', False),
            },
        )

        fig = FigurePayload(
            title=kwargs.get('fig_title', ''),
            subplots=[subplot],
            metadata={'layout': {'rows': 1, 'cols': 1}},
        )

        dialog.set_payload(fig)

        return dialog

    def set_payload(self, payload: FigurePayload):
        self._payload = payload
        self._sync_settings_from_payload(force_rebuild=False)
        self.schedule_render()

    def set_payload_from_dict(self, payload: dict[str, Any]):
        self.set_payload(self._figure_payload_from_dict(payload))

    def set_subplots(self, subplots: list[SubplotPayload], title: str = ''):
        self._payload = FigurePayload(title=title, subplots=subplots)
        self._sync_settings_from_payload(force_rebuild=False)
        self.schedule_render()

    def append_subplot(self, subplot: SubplotPayload):
        self._payload.subplots.append(subplot)
        self._sync_settings_from_payload(force_rebuild=False)
        self.schedule_render()

    def clear_payloads(self):
        self._payload = FigurePayload()
        self._sync_settings_from_payload(force_rebuild=True)
        self.schedule_render()

    def schedule_render(self):
        self._refresh_timer.start()

    def _on_settings_changed(self, _param: Parameter, changes: list):
        if self._is_refreshing_subplots:
            return

        should_refresh_subplots = False
        for change in changes:
            changed_param, change_type, _data = change
            if change_type != 'value':
                continue
            child_path = self.plot_settings.get_param_path(changed_param) or []
            dotted_path = '.'.join(str(item) for item in child_path)
            if dotted_path in {'Layout.Rows', 'Layout.Cols'}:
                should_refresh_subplots = True

            if dotted_path == 'Figure.Debounce (ms)':
                self._refresh_timer.setInterval(int(_data))

        if should_refresh_subplots:
            self._refresh_subplot_settings()

        self.schedule_render()

    def closeEvent(self, event):
        self._refresh_timer.stop()
        super().closeEvent(event)
        self.deleteLater()

    def _refresh_subplot_settings(self):
        self._is_refreshing_subplots = True
        try:
            count = max(
                len(self._payload.subplots),
                self._layout_rows() * self._layout_cols(),
            )
            self.plot_settings.refresh_subplots(count)
        finally:
            self._is_refreshing_subplots = False

    def _layout_rows(self) -> int:
        return int(self.plot_settings.get_param_value('Layout.Rows', 1))

    def _layout_cols(self) -> int:
        return int(self.plot_settings.get_param_value('Layout.Cols', 1))

    def _sync_settings_from_payload(self, force_rebuild: bool = False):
        dataset_labels = sorted(
            {
                series.dataset
                for subplot in self._payload.subplots
                for series in subplot.series
                if series.dataset
            }
        )
        plot_labels = [
            subplot.title or f'({subplot.row}, {subplot.col})'
            for subplot in self._payload.subplots
        ]

        signature = (tuple(plot_labels), tuple(dataset_labels))
        if force_rebuild or signature != self._payload_signature:
            self._payload_signature = signature
            self.plot_settings.rebuild(
                plot_labels=plot_labels,
                dataset_labels=dataset_labels,
                subplot_defaults=self._payload.subplots,
            )
            if (
                not self.plot_settings.get_param_value('Figure.Title', '')
                and self._payload.title
            ):
                figure_title = self.plot_settings.get_param('Figure.Title')
                if figure_title is not None:
                    figure_title.setDefault(self._payload.title)
                    figure_title.setValue(self._payload.title)

            layout_meta = self._payload.metadata.get('layout', {})
            layout_rows = layout_meta.get('rows')
            layout_cols = layout_meta.get('cols')
            if layout_rows is not None:
                rows_param = self.plot_settings.get_param('Layout.Rows')
                if rows_param is not None:
                    rows_param.setValue(int(layout_rows))
            if layout_cols is not None:
                cols_param = self.plot_settings.get_param('Layout.Cols')
                if cols_param is not None:
                    cols_param.setValue(int(layout_cols))

            figure_width = layout_meta.get('width_in')
            figure_height = layout_meta.get('height_in')
            if figure_width is not None:
                width_param = self.plot_settings.get_param('Figure.Width (in)')
                if width_param is not None:
                    width_param.setValue(float(figure_width))
            if figure_height is not None:
                height_param = self.plot_settings.get_param('Figure.Height (in)')
                if height_param is not None:
                    height_param.setValue(float(figure_height))
            self._refresh_subplot_settings()

    def _figure_payload_from_dict(self, payload: dict[str, Any]) -> FigurePayload:
        subplots: list[SubplotPayload] = []
        for subplot_data in payload.get('subplots', []):
            series_payload: list[PlotSeriesPayload] = []
            for series_data in subplot_data.get('series', []):
                series_payload.append(
                    PlotSeriesPayload(
                        x=series_data.get('x'),
                        y=series_data.get('y'),
                        plot_type=series_data.get('plot_type', 'line'),
                        label=series_data.get('label', ''),
                        dataset=series_data.get('dataset', 'default'),
                        style=dict(series_data.get('style', {})),
                        visible=bool(series_data.get('visible', True)),
                        extras=dict(series_data.get('extras', {})),
                    )
                )

            subplots.append(
                SubplotPayload(
                    row=int(subplot_data.get('row', 0)),
                    col=int(subplot_data.get('col', 0)),
                    title=subplot_data.get('title', ''),
                    xlabel=subplot_data.get('xlabel', ''),
                    ylabel=subplot_data.get('ylabel', ''),
                    xscale=subplot_data.get('xscale', 'linear'),
                    yscale=subplot_data.get('yscale', 'linear'),
                    series=series_payload,
                    metadata=dict(subplot_data.get('metadata', {})),
                )
            )

        return FigurePayload(
            title=payload.get('title', ''),
            subplots=subplots,
            metadata=dict(payload.get('metadata', {})),
        )

    @staticmethod
    def _coerce_1d(data: Iterable[float] | np.ndarray | None) -> np.ndarray | None:
        if data is None:
            return None
        arr = np.asarray(data)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr

    def _dataset_enabled(self, subplot_index: int, dataset: str) -> bool:
        global_enabled = self.plot_settings.get_param_value(
            f'Datasets.{dataset}',
            True,
        )

        use_global = self.plot_settings.get_param_value(
            f'Subplots.Plot {subplot_index + 1}.Use Global Datasets',
            True,
        )
        if use_global:
            return bool(global_enabled)

        subplot_enabled = self.plot_settings.get_param_value(
            f'Subplots.Plot {subplot_index + 1}.Datasets.{dataset}',
            global_enabled,
        )
        return bool(subplot_enabled)

    def _style_context(self):
        theme = self.plot_settings.get_param_value('Style.Theme', 'default')
        font_size = self.plot_settings.get_param_value('Style.Font Size', 10)
        linewidth = self.plot_settings.get_param_value('Style.Line Width', 1.6)
        font_family = self.plot_settings.get_param_value(
            'Figure.Font Family',
            'DejaVu Sans',
        )

        color_cycle_name = self.plot_settings.get_param_value(
            'Style.Color Cycle',
            'tab10',
        )
        cmap = matplotlib.colormaps.get(color_cycle_name)
        cycle_colors = getattr(cmap, 'colors', None)

        rc_params = {
            'font.size': font_size,
            'lines.linewidth': linewidth,
            'font.family': font_family,
        }
        if cycle_colors:
            rc_params['axes.prop_cycle'] = cycler(color=list(cycle_colors))

        style_ctx = (
            mpl_style.context(theme) if theme in mpl_style.available else nullcontext()
        )
        return style_ctx, rc_context(rc=rc_params)

    def _subplot_index_from_position(self, row: int, col: int) -> int:
        return row * self._layout_cols() + col

    def _subplot_setting(self, subplot_index: int, key: str, default: Any):
        return self.plot_settings.get_param_value(
            f'Subplots.Plot {subplot_index + 1}.{key}',
            default,
        )

    def _subplot_row_col(self, subplot_index: int) -> tuple[int, int]:
        row = int(self._subplot_setting(subplot_index, 'Row', subplot_index // 2 + 1))
        col = int(self._subplot_setting(subplot_index, 'Col', subplot_index % 2 + 1))
        return row - 1, col - 1

    def _subplot_visible(self, subplot_index: int) -> bool:
        return bool(self._subplot_setting(subplot_index, 'Visible', True))

    def _default_line_style(self, style: dict[str, Any]) -> dict[str, Any]:
        result = dict(style)
        result.setdefault(
            'linewidth',
            self.plot_settings.get_param_value('Style.Line Width', 1.6),
        )
        result.setdefault(
            'markersize',
            self.plot_settings.get_param_value('Style.Marker Size', 5.0),
        )
        return result

    def _subplot_xtick_labels(
        self,
        subplot_index: int,
        fallback_labels: list[str],
    ) -> list[str]:
        raw_labels = str(
            self._subplot_setting(subplot_index, 'X Tick Labels', '')
        ).strip()
        if not raw_labels:
            return fallback_labels

        labels = [part.strip() for part in raw_labels.split(',')]
        cleaned = [label for label in labels if label]
        return cleaned or fallback_labels

    def _subplot_series_colors(self, subplot_index: int) -> list[str]:
        raw_colors = str(
            self._subplot_setting(subplot_index, 'Series Colors', '')
        ).strip()
        if not raw_colors:
            return []

        colors = [part.strip() for part in raw_colors.split(',')]
        return [color for color in colors if color]

    def _draw_series(self, axis: Axes, series: PlotSeriesPayload):
        x = self._coerce_1d(series.x)
        y = self._coerce_1d(series.y)
        style = self._default_line_style(series.style)
        label = series.label or None

        if not series.visible:
            return

        if series.plot_type in {'line', 'scatter', 'bar', 'errorbar'} and (
            x is None or y is None
        ):
            logger.warning(
                'Series %s skipped: x/y required for %s',
                series.label,
                series.plot_type,
            )
            return

        if (
            series.plot_type != 'histogram'
            and x is not None
            and y is not None
            and len(x) != len(y)
        ):
            logger.warning('Series %s skipped: x/y length mismatch', series.label)
            return

        if series.plot_type == 'line':
            axis.plot(x, y, label=label, **style)
            return

        if series.plot_type == 'scatter':
            scatter_style = dict(style)
            marker_size = scatter_style.pop(
                's',
                self.plot_settings.get_param_value('Style.Marker Size', 5.0) ** 2,
            )
            scatter_style.pop('linewidth', None)
            scatter_style.pop('markersize', None)
            axis.scatter(x, y, s=marker_size, label=label, **scatter_style)
            return

        if series.plot_type == 'bar':
            bar_style = dict(style)
            bar_style.pop('markersize', None)
            bar_style.setdefault('width', series.extras.get('width', 0.8))
            axis.bar(x, y, label=label, **bar_style)
            return

        if series.plot_type == 'histogram':
            hist_style = dict(style)
            hist_style.pop('markersize', None)
            bins = series.extras.get('bins', 'auto')
            if y is None:
                axis.hist(x, bins=bins, label=label, **hist_style)
                return
            if len(x) == len(y) + 1:
                axis.stairs(y, x, label=label, **hist_style)
            else:
                axis.bar(x, y, label=label, **hist_style)
            return

        if series.plot_type == 'errorbar':
            error_style = dict(style)
            error_style.pop('markersize', None)
            xerr = series.extras.get('xerr')
            yerr = series.extras.get('yerr')
            axis.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                label=label,
                **error_style,
            )
            return

        if series.plot_type == 'fill':
            fill_style = dict(style)
            alpha = fill_style.pop(
                'alpha',
                150 / 255,
            )
            facecolor = fill_style.pop(
                'facecolor',
                (100 / 255, 100 / 255, 255 / 255),
            )
            axis.fill_between(
                x,
                y,
                y2=0,  # fillLevel: 0
                step='mid',  # stepMode: True
                facecolor=facecolor,  # brush RGB
                alpha=alpha,  # brush Alpha (skaidrumas)
            )

            axis.step(x, y, where='mid', color=facecolor)
            return

        logger.warning(
            'Unsupported plot_type=%s for series=%s',
            series.plot_type,
            series.label,
        )

    @staticmethod
    def _apply_category_ticks(axis: Axes, labels: list[str]):
        if not labels:
            return

        tick_positions = np.arange(len(labels), dtype=float)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(labels, rotation=35, ha='right')

    def render_now(self):
        self._subplot_lookup = {(sp.row, sp.col): sp for sp in self._payload.subplots}

        rows = self._layout_rows()
        cols = self._layout_cols()
        share_x = bool(self.plot_settings.get_param_value('Layout.Share X', False))
        share_y = bool(self.plot_settings.get_param_value('Layout.Share Y', False))

        width_in = float(self.plot_settings.get_param_value('Figure.Width (in)', 11.0))
        height_in = float(self.plot_settings.get_param_value('Figure.Height (in)', 7.0))
        dpi = int(self.plot_settings.get_param_value('Figure.DPI', 100))

        style_ctx, rc_ctx = self._style_context()

        with style_ctx, rc_ctx:
            self.figure.clear()
            self.figure.set_size_inches(width_in, height_in, forward=True)
            self.figure.set_dpi(dpi)

            axes_grid = self.figure.subplots(
                rows,
                cols,
                squeeze=False,
                sharex=share_x,
                sharey=share_y,
            )

            global_grid = bool(self.plot_settings.get_param_value('Style.Grid', True))
            global_legend = bool(
                self.plot_settings.get_param_value('Style.Legend', True)
            )
            legend_location = self.plot_settings.get_param_value(
                'Style.Legend Location',
                'best',
            )
            assigned_axes: set[tuple[int, int]] = set()

            for subplot_index, subplot_payload in enumerate(self._payload.subplots):
                if not self._subplot_visible(subplot_index):
                    continue

                row, col = self._subplot_row_col(subplot_index)
                if row < 0 or col < 0 or row >= rows or col >= cols:
                    continue

                axis: Axes = axes_grid[row][col]
                assigned_axes.add((row, col))
                axis.clear()

                series_colors = self._subplot_series_colors(subplot_index)

                for series_index, series in enumerate(subplot_payload.series):
                    if not self._dataset_enabled(subplot_index, series.dataset):
                        continue
                    if series_index < len(series_colors):
                        styled_series = PlotSeriesPayload(
                            x=series.x,
                            y=series.y,
                            plot_type=series.plot_type,
                            label=series.label,
                            dataset=series.dataset,
                            style={
                                **series.style,
                                'color': series_colors[series_index],
                            },
                            visible=series.visible,
                            extras=series.extras,
                        )
                        self._draw_series(axis, styled_series)
                        continue
                    self._draw_series(axis, series)

                title_override = self._subplot_setting(
                    subplot_index,
                    'Title Override',
                    '',
                )
                x_label_override = self._subplot_setting(
                    subplot_index,
                    'X Label Override',
                    '',
                )
                y_label_override = self._subplot_setting(
                    subplot_index,
                    'Y Label Override',
                    '',
                )

                subplot_title_alignment = self._subplot_setting(
                    subplot_index,
                    'Title Alignment',
                    'center',
                )
                subplot_title_bold = bool(
                    self._subplot_setting(subplot_index, 'Title Bold', False)
                )
                try:
                    title_artist = axis.set_title(
                        title_override,
                        y=float(
                            self.plot_settings.get_param_value('Style.Title Y', 1.05)
                        ),
                        loc=subplot_title_alignment,
                    )
                    if subplot_title_alignment in {'left', 'right'}:
                        title_artist.set_horizontalalignment(
                            subplot_title_alignment,
                        )
                        title_margin = float(
                            self.plot_settings.get_param_value(
                                'Style.Right title margin'
                                if subplot_title_alignment == 'right'
                                else 'Style.Left title margin',
                                1.0 if subplot_title_alignment == 'right' else -0.1,
                            )
                        )
                        title_artist.set_x(title_margin)
                except Exception as e:
                    logger.debug(
                        'Failed to set title alignment for subplot %d: %s',
                        subplot_index + 1,
                        e,
                    )
                    title_artist = axis.set_title(title_override)
                title_artist.set_fontweight(
                    'bold' if subplot_title_bold else 'normal',
                )
                axis.set_xlabel(x_label_override)
                axis.set_ylabel(y_label_override)
                axis.set_xscale(subplot_payload.xscale)
                axis.set_yscale(subplot_payload.yscale)

                category_labels = subplot_payload.metadata.get('dataset_names', [])
                if category_labels:
                    self._apply_category_ticks(
                        axis,
                        self._subplot_xtick_labels(
                            subplot_index,
                            list(category_labels),
                        ),
                    )

                show_grid = bool(
                    self._subplot_setting(
                        subplot_index,
                        'Grid Override',
                        global_grid,
                    )
                )
                show_legend = bool(
                    self._subplot_setting(
                        subplot_index,
                        'Legend Override',
                        global_legend,
                    )
                )

                axis.set_axisbelow(True)
                axis.grid(show_grid)

                handles, labels = axis.get_legend_handles_labels()
                if show_legend and handles and labels:
                    axis.legend(loc=legend_location)

            for row in range(rows):
                for col in range(cols):
                    if (row, col) not in assigned_axes:
                        axes_grid[row][col].set_visible(False)

            title = self.plot_settings.get_param_value(
                'Figure.Title', self._payload.title
            )
            title_alignment = self.plot_settings.get_param_value(
                'Figure.Title Alignment',
                'center',
            )
            title_bold = bool(
                self.plot_settings.get_param_value('Figure.Title Bold', True)
            )
            if title_alignment == 'left':
                self.figure.suptitle(
                    title,
                    x=0.01,
                    ha='left',
                    fontweight='bold' if title_bold else 'normal',
                )
            elif title_alignment == 'right':
                self.figure.suptitle(
                    title,
                    x=0.99,
                    ha='right',
                    fontweight='bold' if title_bold else 'normal',
                )
            else:
                self.figure.suptitle(
                    title,
                    x=0.5,
                    ha='center',
                    fontweight='bold' if title_bold else 'normal',
                )

            column_spacing = float(
                self.plot_settings.get_param_value('Layout.Column Spacing', 0.3)
            )
            row_spacing = float(
                self.plot_settings.get_param_value('Layout.Row Spacing', 0.35)
            )
            left_margin = float(
                self.plot_settings.get_param_value('Layout.Left Margin', 0.05)
            )
            right_margin = float(
                self.plot_settings.get_param_value('Layout.Right Margin', 0.95)
            )
            top_margin = float(
                self.plot_settings.get_param_value('Layout.Top Margin', 0.9)
            )
            bottom_margin = float(
                self.plot_settings.get_param_value('Layout.Bottom Margin', 0.1)
            )
            self.figure.subplots_adjust(
                wspace=column_spacing,
                hspace=row_spacing,
                left=left_margin,
                right=right_margin,
                top=top_margin,
                bottom=bottom_margin,
            )
            # self.figure.tight_layout()
            self.canvas.draw_idle()

    def save_figure(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            'Save Figure',
            '',
            'PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;All Files (*)',
        )
        if not filename:
            return

        self.figure.savefig(filename, dpi=self.figure.dpi)
