import logging
import traceback

from microEye.analysis.clustering.dataclasses import (
    CenteringMode,
    CenteringOptions,
    ClusteringOptions,
    InstanceStats,
    PipelineResult,
)
from microEye.analysis.clustering.grid import GridModel
from microEye.analysis.clustering.io import export_run
from microEye.analysis.clustering.model import (
    GeometricModel,
    get_instances_from_clusters,
)
from microEye.analysis.clustering.rulers import RulerModel
from microEye.analysis.fitting.results import DataColumns, FittingResults

logger = logging.getLogger(__name__)


MODEL_REGISTRY = {'ruler': RulerModel, 'grid': GridModel}


def build_model(model_meta: dict) -> GeometricModel:
    if not isinstance(model_meta, dict):
        raise TypeError(f'model_meta must be a dict, got {type(model_meta).__name__}')
    kind = model_meta.get('type')

    if kind not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{kind}'. Known: {list(MODEL_REGISTRY)}")

    params = {k: v for k, v in model_meta.items() if k != 'type'}
    return MODEL_REGISTRY[kind](**params)


def render_instances_batch(
    tables: list['FittingResults'],
    model: GeometricModel,
    **kwargs,
) -> PipelineResult:
    '''
    Generic per-model batch pipeline: rulers, grids, or any future
    GeometricModel.

    Returns
    -------
    PipelineResult
        - stats: list of InstanceStats objects, one per table
        - instances: list of dicts, one per table,
          each dict contains the fitted instances
        - model_meta: dict of model parameters for logging and saving
    '''
    try:
        run_meta = kwargs.copy()

        save_results = kwargs.pop('save_results', False)
        max_residual = kwargs.pop('max_residual', None)

        centering_options = CenteringOptions(
            center_mode=CenteringMode(kwargs.pop('center_mode', 'crlb_weighted_mean')),
            eps=kwargs.pop('eps', 1e-12),
            rejection_threshold=kwargs.pop('rejection_threshold', None),
            min_points=model.min_points,
            max_points=None,
            subtract_com=False,
        )

        distance_threshold = kwargs.pop('distance_threshold', None)
        clustering_options = ClusteringOptions(
            grouping_method=kwargs.pop('grouping_method', 'dbscan'),
            distance_threshold=(
                distance_threshold
                if distance_threshold and distance_threshold > 0
                else None
            ),
            sigma_xy=kwargs.pop('sigma_xy', 15.0),
            sigma_z=kwargs.pop('sigma_z', 45.0),
            scale_data=kwargs.pop('scale_data', False),
            xi=kwargs.pop('xi', 0.05),
        )

        per_table_instances: list[dict] = []
        per_table_stats: list[InstanceStats] = []

        for table in tables:
            aligned, table_stats = get_instances_from_clusters(
                table.data[DataColumns.X],
                table.data[DataColumns.Y],
                table.data[DataColumns.Z],
                table.data[DataColumns.CLUSTER_ID],
                model=model,
                intensity=table.data[DataColumns.INTENSITY],
                crlb_x=table.data[DataColumns.CRLB_X],
                crlb_y=table.data[DataColumns.CRLB_Y],
                max_residual=max_residual,
                centering_options=centering_options,
                clustering_options=clustering_options,
            )
            per_table_instances.append(aligned)
            per_table_stats.append(table_stats)

        if save_results:
            suffix = kwargs.pop('save_suffix') or model.get_filename_suffix()
            path = tables[0].path.replace('.h5', f'_{suffix}.h5')
            export_run(
                [
                    (instances, stats)
                    for instances, stats in zip(per_table_instances, per_table_stats)
                ],
                path,
                model.get_model_meta(),
                run_meta=kwargs,
            )
            logger.info(f'Exported results to first table path: {path}')

        return PipelineResult(
            stats=per_table_stats,
            instances=per_table_instances,
            model_meta=model.get_model_meta(),
            run_meta=run_meta,
        )
    except Exception:
        traceback.print_exc()
        return PipelineResult(
            stats=[],
            instances=[],
            model_meta=model.get_model_meta(),
            run_meta=run_meta,
        )


def render_rulers_batch(
    tables: list['FittingResults'],
    **kwargs,
) -> PipelineResult:
    '''
    Full ruler pipeline per table (or aggregated).
    Returns one PlotResult per output image — same shape as before
    but now each carries its own histograms.
    '''
    model_kwargs = {}
    for key in RulerModel.__init__.__code__.co_varnames:
        if key in kwargs:
            model_kwargs[key] = kwargs.pop(key)

    model = RulerModel(**model_kwargs)
    return render_instances_batch(tables, model, **kwargs)


def render_grids_batch(tables, **kwargs) -> PipelineResult:
    model_kwargs = {}
    for key in GridModel.__init__.__code__.co_varnames:
        if key in kwargs:
            model_kwargs[key] = kwargs.pop(key)

    model = GridModel(
        M=model_kwargs.pop('M', 3), N=model_kwargs.pop('N', 4), **model_kwargs
    )
    return render_instances_batch(tables, model, **kwargs)
