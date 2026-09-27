import copy
import itertools
import logging
from dataclasses import asdict, is_dataclass

import pyqtgraph as pg

from microEye.analysis.clustering.dataclasses import (
    ClusterRenderData,
    InstanceStats,
    LabeledLocalizations,
    PlotResult,
)
from microEye.analysis.clustering.dialogs import (
    BASE_PARAMS,
    RENDERING_PARAMS,
    VIEWER_PARAMS,
)
from microEye.analysis.clustering.io import export_run, load_run
from microEye.analysis.clustering.model import (
    combine_stats,
    instances_concatenate,
)
from microEye.analysis.clustering.pipeline import PipelineResult, build_model
from microEye.analysis.clustering.plot import (
    ClustersScatter,
    add_plot_result,
)
from microEye.qt import QtCore, QtWidgets, Signal, getOpenFileName, getSaveFileName
from microEye.utils.parameter_tree import GroupParameter, Parameter, Tree
from microEye.utils.thread_worker import QThreadWorker

logger = logging.getLogger(__name__)

RENDER_KEY_TO_PARAM_NAME = {
    'roi_size': 'ROI Size [px]',
    'pixel_size': 'Pixel Size [nm]',
    'scalebar_length': 'Scale Bar Length [nm]',
    'render_mode': 'Render Mode',
    'line_profiles': 'Line Profiles',
    'aggregate': 'Aggregate View',
    'angle_histogram': 'Angle Histogram',
    'intensity_histogram': 'Intensity Histogram',
    'n_matched_histogram': 'N Matched Histogram',
}


class ViewerParams(Tree):
    loadActivated = Signal()
    saveActivated = Signal()
    updateRendering = Signal()

    def __init__(self, parent=None, **kwargs):
        self._viewer_params = kwargs.pop('viewer_params', VIEWER_PARAMS)
        self._render_map = kwargs.pop('render_map', RENDER_KEY_TO_PARAM_NAME)

        # last step
        super().__init__(parent)

    def create_parameters(self):

        groups = []

        groups.extend(self._viewer_params)
        groups.extend(BASE_PARAMS)

        self.param_tree = Parameter.create(
            name='',
            type='group',
            children=groups,
        )
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        # connect signals for actions
        self.param_tree.param('Load Results').sigActivated.connect(
            self.loadActivated.emit
        )
        self.param_tree.param('Save Results').sigActivated.connect(
            self.saveActivated.emit
        )

    def change(self, param, changes):
        for _, change, _ in changes:
            if change == 'value':
                # if param.name() in [
                #     'Aggregate View',
                #     'Angle Histogram',
                #     'Intensity Histogram',
                # ]:
                self.updateRendering.emit()

    def update_model_metadata(self, model_meta: dict):
        model_group = self.param_tree.param('Model Metadata')
        if not isinstance(model_group, GroupParameter):
            return

        model_group.clearChildren()
        for k, v in model_meta.items():
            model_group.addChild(
                Parameter.create(name=k, type='str', value=str(v), readonly=True)
            )

    def update_data_summary(self, summary_list: list[dict]):
        summary_group = self.param_tree.param('Data Summary')
        if not isinstance(summary_group, GroupParameter):
            return

        summary_group.clearChildren()
        for summary in summary_list:
            parent = GroupParameter.create(
                name=f"Table {summary.get('table', 'N/A')}",
                type='group',
                children=[],
            )
            summary_group.addChild(parent)
            for k, v in summary.items():
                parent.addChild(
                    Parameter.create(name=k, type='str', value=str(v), readonly=True)
                )

    def update_runtime_metadata(self, run_meta: dict):
        runtime_group = self.param_tree.param('Runtime Metadata')
        if not isinstance(runtime_group, GroupParameter):
            return

        runtime_group.clearChildren()
        for k, v in run_meta.items():
            runtime_group.addChild(
                Parameter.create(name=k, type='str', value=str(v), readonly=True)
            )

    def get_rendering_options(self) -> dict:
        return {
            key: self.param_tree.param(param_name).value()
            for key, param_name in self._render_map.items()
            if self.param_tree.param(param_name) is not None
        }

    @property
    def aggregate(self) -> bool:
        return self.param_tree.param('Aggregate View').value()

    @aggregate.setter
    def aggregate(self, value: bool):
        self.param_tree.param('Aggregate View').setValue(value)

    def set_rendering_options(self, **kwargs):
        for key in self._render_map:
            if key in kwargs:
                param_name = self._render_map[key]
                param = self.param_tree.param(param_name)
                if param is not None:
                    param.setValue(kwargs[key])


class BaseViewer(QtWidgets.QWidget):
    '''
    Base class for viewers of clustering results
    '''

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.setWindowTitle('Clustering Results Viewer')
        self._redraw_token = 0
        self._worker = None

        self._redraw_timer = QtCore.QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.timeout.connect(self.request_redraw)
        self._redraw_timer.setInterval(200)  # milliseconds

        self.setup_ui(**kwargs)

    def setup_ui(self, **kwargs):
        '''
        Build the UI layout.
        '''
        self.setMinimumHeight(640)
        self.setMinimumWidth(800)

        self.tab_layout = QtWidgets.QTabWidget()
        self._viewer_params = kwargs.pop('viewer_params', VIEWER_PARAMS)
        self._render_map = kwargs.pop('render_map', RENDER_KEY_TO_PARAM_NAME)

        self.viewer_params = ViewerParams(
            viewer_params=kwargs.pop('viewer_params', self._viewer_params),
            render_map=kwargs.pop('render_map', self._render_map),
        )
        self.viewer_params.loadActivated.connect(self._on_load)
        self.viewer_params.saveActivated.connect(self._on_save)
        self.viewer_params.updateRendering.connect(self._redraw)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self.viewer_params)
        splitter.addWidget(self.tab_layout)
        splitter.setStretchFactor(1, 3)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(splitter)

    def request_redraw(self):
        self._redraw_token += 1
        token = self._redraw_token
        render_kwargs = self.render_kwargs
        payload = self._get_render_payload()

        def done(result):
            if token != self._redraw_token:
                return
            self._display_render_result(result, render_kwargs)

        self._worker = QThreadWorker(
            self._calculate_render_result,
            payload,
            render_kwargs,
        )
        self._worker.signals.result.connect(done)
        QtCore.QThreadPool.globalInstance().start(self._worker)

    def _redraw(self):
        self._redraw_timer.start()

    def _get_render_payload(self):
        raise NotImplementedError

    def _calculate_render_result(
        self, payload, render_kwargs, **kwargs
    ) -> list[PlotResult]:
        raise NotImplementedError

    def _display_render_result(self, result: list[PlotResult], render_kwargs):
        raise NotImplementedError

    def _add_plot_result(self, data, title: str, **render_kwargs):
        '''
        Add a PlotResult to the tab layout with the given title.
        '''
        plot_layout = pg.GraphicsLayoutWidget()
        plot_layout.setBackground(None)

        result = self._get_data_plot_result(data, **render_kwargs)

        if result is None:
            logger.warning(f'No data to render for {title}')
            return

        add_plot_result(plot_layout, result, row=0, col=0, **render_kwargs)

        self.tab_layout.addTab(plot_layout, title)

    def _get_data_plot_result(self, data, **render_kwargs) -> PlotResult:
        '''
        Render the given data (aligned instances) into a PlotResult.
        '''
        if isinstance(data, dict):
            raise NotImplementedError('Rendering from dict is not implemented')
        elif isinstance(data, PlotResult):
            return data
        else:
            raise ValueError('Unsupported data type for rendering')

    def _on_save(self): ...

    def _on_load(self, path: str | None = None): ...

    @staticmethod
    def _normalize_stats(stats_list) -> list[dict]:
        '''
        Callers may hand us either raw InstanceStats objects (e.g. straight from
        render_instances_batch) or plain dicts (e.g. from load_run, which stores
        them as JSON). Every internal method expects dicts, so normalize here —
        this is the single place that decision is made.
        '''
        return [asdict(s) if is_dataclass(s) else s for s in stats_list]

    @property
    def render_kwargs(self) -> dict:
        return self.viewer_params.get_rendering_options()

    @property
    def render_keys(self) -> list:
        return list(self.viewer_params.get_rendering_options().keys())

    @render_kwargs.setter
    def render_kwargs(self, kwargs: dict):
        keys = self.render_keys
        for k in list(kwargs.keys()):
            if k not in keys:
                kwargs.pop(k)

        self.viewer_params.set_rendering_options(**kwargs)


class ClustersViewer(BaseViewer):
    def __init__(self, parent=None, path: str | None = None):
        keys: list[str] = [param['name'] for param in RENDERING_PARAMS]
        names = [key.replace('_', ' ').title() for key in keys]

        mapping = {key: name for key, name in zip(keys, names)}

        mapping['aggregate'] = 'Aggregate View'

        params = copy.deepcopy(RENDERING_PARAMS)
        # update param name with titled ones
        for param in params:
            param['name'] = mapping[param['name']]

        self._data = None

        super().__init__(parent, viewer_params=params, render_map=mapping)

        if path is not None and path:
            self._on_load(path=path)

    def load_cluster_render(
        self,
        cluster_render_result: ClusterRenderData | None = None,
    ):
        if cluster_render_result is None:
            self._data = None
            return

        self._data = cluster_render_result

        self._redraw()

    def _refresh_summary(self):
        # empty summary if no data
        self.viewer_params.update_data_summary([])

    def _get_render_payload(self):
        return self._data

    def _calculate_render_result(
        self, payload: ClusterRenderData, render_kwargs, **kwargs
    ) -> list[PlotResult] | None:
        if self._data is None or not self._data.tables:
            self.tab_layout.clear()
            self._refresh_summary()
            return

        return ClustersScatter.render_clusters_batch(payload.tables, **render_kwargs)

    def _display_render_result(self, result: list[PlotResult], render_kwargs):
        self.tab_layout.clear()

        if result is None or len(result) == 0:
            logger.warning('No data to render')
            self._refresh_summary()
            return

        for r in result:
            self._add_plot_result(r, r.title, **render_kwargs)


class ClusterResultsWidget(BaseViewer):
    '''
    Widget for viewing clustering results, including aligned instances and
    associated statistics. Supports loading and saving results to HDF5 files.
    '''

    def __init__(self, parent=None, path: str | None = None):
        self._tables_aligned: list[dict] = []
        self._tables_stats: list[dict] = []
        self._model_meta: dict = {}

        super().__init__(parent)

        if path is not None and path:
            self._on_load(path=path)

    def _get_render_payload(self):
        return self._tables_aligned, self._model_meta

    def _calculate_render_result(
        self, payload: tuple[list[dict], dict], render_kwargs, **kwargs
    ) -> list[PlotResult] | None:
        tables_aligned, model_meta = payload
        if not tables_aligned or not model_meta:
            return None

        model = build_model(model_meta)
        ref_lines = (
            model.reference_lines() if hasattr(model, 'reference_lines') else None
        )
        if ref_lines:
            render_kwargs['reference_lines'] = ref_lines

        results = []
        if self.viewer_params.aggregate:
            merged = {}
            counter = itertools.count()
            for aligned in tables_aligned:
                for data in aligned.values():
                    merged[next(counter)] = data
            results.append(
                ClustersScatter._render_instances(
                    instances_concatenate(merged), **render_kwargs
                )
            )
        else:
            for _, aligned in enumerate(tables_aligned):
                results.append(
                    ClustersScatter._render_instances(
                        instances_concatenate(aligned), **render_kwargs
                    )
                )

        return results

    def _display_render_result(self, result: list[PlotResult], render_kwargs):
        self.tab_layout.clear()

        if result is None or len(result) == 0:
            logger.warning('No data to render')
            self._refresh_summary()
            return

        for i, r in enumerate(result):
            title = 'Overall' if self.viewer_params.aggregate else f'Table {i}'
            self._add_plot_result(r, title, **render_kwargs)

        self._refresh_summary()

    def load_pipeline_result(
        self,
        pipeline_result: PipelineResult | None = None,
    ):
        '''
        Load precomputed PlotResult objects (e.g., from render_rulers_batch or
        render_grids_batch) and display them.

        This is a convenience method for when the caller has already computed the
        PlotResult objects and does not need to recompute them.

        Parameters
        ----------
        pipeline_result : PipelineResult or None
            The precomputed results to display. If None, the viewer will be cleared.
        **render_kwargs
            Additional keyword arguments for rendering.
        '''
        self._pipeline_result = pipeline_result

        self._tables_aligned = pipeline_result.instances if pipeline_result else []
        self._tables_stats = self._normalize_stats(
            pipeline_result.stats if pipeline_result else []
        )
        self._model_meta = pipeline_result.model_meta if pipeline_result else {}

        self._run_meta = pipeline_result.run_meta.copy() or {}
        self.viewer_params.update_model_metadata(self._model_meta)
        self.viewer_params.update_runtime_metadata(self._run_meta)

        # self.render_kwargs = self._run_meta

        self._redraw()

    def _refresh_summary(self):
        def summary_row(table, stats):
            cluster_localizations = stats['n_input_cluster_localizations']
            candidates = stats['n_candidates']

            return {
                'table': table,
                'localizations': stats['n_localizations'],
                'input_clusters': stats['n_input_clusters'],
                'cluster_localizations': cluster_localizations,
                'cluster_centers': stats['n_cluster_centers'],
                'rejected_centers': stats['n_rejected_centers'],
                'candidates': candidates,
                'candidate_centers': stats['n_candidate_centers'],
                'rejected_candidates': stats['n_rejected_candidates'],
                'passed_instances': stats['n_passed_instances'],
                'passed_centers': stats['n_passed_centers'],
                'passed_localizations': stats['n_passed_localizations'],
                'localization_pass_rate': round(
                    stats['n_passed_localizations'] / cluster_localizations,
                    3,
                )
                if cluster_localizations
                else 0.0,
                'candidate_pass_rate': round(
                    stats['n_passed_instances'] / candidates,
                    3,
                )
                if candidates
                else 0.0,
            }

        rows = [summary_row(i, stats) for i, stats in enumerate(self._tables_stats)]

        if len(self._tables_stats) > 1 and not self.viewer_params.aggregate:
            total = combine_stats(
                [InstanceStats(**stats) for stats in self._tables_stats]
            )
            rows.append(summary_row('TOTAL', asdict(total)))

        self.viewer_params.update_data_summary(rows)

    def _on_save(self):
        if not self._tables_aligned:
            logger.warning('Nothing to save')
            return
        filename, _ = getSaveFileName(
            caption='Save Results', filter='HDF5 Files (*.h5)'
        )
        if filename:
            pairs = list(
                zip(
                    self._tables_aligned,
                    [InstanceStats(**s) for s in self._tables_stats],
                )
            )
            export_run(pairs, filename, self._model_meta, run_meta=self._run_meta)

    def _on_load(self, path: str | None = None):
        if path is None:
            path, _ = getOpenFileName(self, 'Load Results', filter='HDF5 Files (*.h5)')
        if path:
            tables_aligned, tables_stats, model_meta, run_meta = load_run(path)
            self.load_pipeline_result(
                PipelineResult(
                    stats=tables_stats,
                    instances=tables_aligned,
                    model_meta=model_meta,
                    run_meta=run_meta,
                )
            )
