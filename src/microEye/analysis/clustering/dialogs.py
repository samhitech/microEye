from enum import Enum

import rich.console as rc
import rich.table as rt

from microEye.analysis.clustering.dataclasses import CenteringMode
from microEye.analysis.rendering import RenderModes
from microEye.qt import QtWidgets
from microEye.utils.parameter_tree import Parameter, Tree


class OptionsModes(Enum):
    RENDERING = 'Cluster Rendering'
    RULER = 'Ruler Model'
    GRID = 'Grid Model'

    DBSCAN = 'DBSCAN Clustering'
    HDBSCAN = 'HDBSCAN Clustering'
    OPTICS = 'OPTICS Clustering'

    CLUSTER_DENSITY = 'Cluster Density Analysis'


BASE_PARAMS = [
    {
        'name': 'Aggregate View',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'Load Results',
        'type': 'action',
    },
    {
        'name': 'Save Results',
        'type': 'action',
    },
    {
        'name': 'Model Metadata',
        'type': 'group',
        'children': [],
    },
    {
        'name': 'Runtime Metadata',
        'type': 'group',
        'children': [],
    },
    {
        'name': 'Data Summary',
        'type': 'group',
        'children': [],
    },
]

VIEWER_PARAMS = [
    {
        'name': 'Pixel Size [nm]',
        'type': 'float',
        'value': 1.0,
    },
    {
        'name': 'Scale Bar Length [nm]',
        'type': 'float',
        'value': 50.0,
    },
    {
        'name': 'ROI Size [px]',
        'type': 'int',
        'value': 21,
    },
    {
        'name': 'Render Mode',
        'type': 'list',
        'limits': {
            'Histogram': RenderModes.HISTOGRAM,
            'Event Histogram': RenderModes.EVENT_HISTOGRAM,
            'Gaussian': RenderModes.GAUSSIAN,
        },
        'value': RenderModes.HISTOGRAM,
    },
    {
        'name': 'Line Profiles',
        'type': 'bool',
        'value': True,
    },
    {
        'name': 'Angle Histogram',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'Intensity Histogram',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'N Matched Histogram',
        'type': 'bool',
        'value': False,
    },
    # {
    #     'name': 'Symmetrize Line Profiles',
    #     'type': 'bool',
    #     'value': False,
    # },
]

RENDERING_PARAMS = [
    {
        'name': 'mode',
        'type': 'list',
        'limits': {
            'Superimposed': 'superimposed',
            'Full FOV': 'fov',
        },
        'value': 'superimposed',
    },
    {'name': 'rasterized', 'type': 'bool', 'value': True},
    {
        'name': 'render_mode',
        'type': 'list',
        'limits': {
            'Histogram': RenderModes.HISTOGRAM,
            'Event Histogram': RenderModes.EVENT_HISTOGRAM,
            'Gaussian': RenderModes.GAUSSIAN,
        },
        'value': RenderModes.HISTOGRAM,
    },
    {
        'name': 'pixel_size',
        'type': 'float',
        'value': 0.5,
        'limits': (0.1, 50.0),
    },
    {
        'name': 'extent',
        'type': 'float',
        'value': 100.0,
        'limits': (25.0, 1000.0),
    },
    {
        'name': 'scalebar_length',
        'type': 'float',
        'value': 40.0,
        'limits': (1.0, 1000.0),
        'tooltip': 'Length of the scale bar in nanometers.',
    },
    {
        'name': 'center_mode',
        'type': 'list',
        'limits': {
            mode.name.replace('_', ' ').title(): mode.value for mode in CenteringMode
        },
        'value': CenteringMode.CRLB_WEIGHTED_MEAN.value,
    },
    {
        'name': 'rejection_threshold',
        'type': 'float',
        'value': 0.0,
        'limits': (0.0, 100.0),
    },
    {
        'name': 'min_points',
        'type': 'int',
        'value': 1,
        'limits': (1, 1000),
    },
]

RULER_PARAMS = [
    {
        'name': 'grouping_method',
        'type': 'list',
        'limits': {
            'DBSCAN': 'dbscan',
            'HDBSCAN': 'hdbscan',
            'OPTICS': 'optics',
        },
        'value': 'dbscan',
    },
    {'name': 'n_sites', 'type': 'int', 'value': 3, 'limits': (2, 10)},
    {
        'name': 'ruler_spacing',
        'type': 'float',
        'value': 80.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'distance_threshold',
        'type': 'float',
        'value': 105.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'spacing_tolerance',
        'type': 'float',
        'value': 25.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'max_y_dev',
        'type': 'float',
        'value': 20.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'leaf_size',
        'type': 'int',
        'value': 30,
        'limits': (0, 1000),
    },
    {
        'name': 'algorithm',
        'type': 'list',
        'limits': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'value': 'auto',
    },
    {
        'name': 'scale_data',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'angle_histogram',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'intensity_histogram',
        'type': 'bool',
        'value': False,
    },
]

GRID_PARAMS = [
    {
        'name': 'grouping_method',
        'type': 'list',
        'limits': {
            'DBSCAN': 'dbscan',
            'HDBSCAN': 'hdbscan',
            'OPTICS': 'optics',
        },
        'value': 'dbscan',
    },
    {'name': 'M', 'type': 'int', 'value': 3, 'limits': (1, 10)},
    {'name': 'N', 'type': 'int', 'value': 3, 'limits': (1, 10)},
    {
        'name': 'pitch',
        'type': 'float',
        'value': 20.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'min_occupied',
        'type': 'int',
        'value': 4,
        'limits': (1, 10),
    },
    {
        'name': 'snap_residual_max',
        'type': 'float',
        'value': 6.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'min_inlier_fraction',
        'type': 'float',
        'value': 0.8,
        'limits': (0.0, 1.0),
    },
    {
        'name': 'leaf_size',
        'type': 'int',
        'value': 30,
        'limits': (0, 1000),
    },
    {
        'name': 'algorithm',
        'type': 'list',
        'limits': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'value': 'auto',
    },
    {
        'name': 'scale_data',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'angle_histogram',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'intensity_histogram',
        'type': 'bool',
        'value': False,
    },
    # hidden distance_threshold with None default
    {
        'name': 'distance_threshold',
        'type': 'float',
        'value': None,
        'limits': (0.0, 1000.0),
        'visible': False,
        'readonly': True,
    },
]

EXPORT_PARAMS = [
    {
        'name': 'save_results',
        'type': 'bool',
        'value': False,
    },
    {
        'name': 'save_suffix',
        'type': 'str',
        'value': '',
    },
    # {
    #     'name': 'save_file',
    #     'type': 'file',
    #     'value': '',
    # },
]

BATCH_PARAMS = [
    {
        'name': 'aggregate',
        'type': 'bool',
        'value': True,
    },
]

DBSCAN_PARAMS = [
    {
        'name': 'eps',
        'type': 'float',
        'value': 5.0,
        'limits': (0.1, 5000.0),
    },
    {
        'name': 'min_samples',
        'type': 'int',
        'value': 5,
        'limits': (2, 1000),
    },
    {
        'name': 'leaf_size',
        'type': 'int',
        'value': 30,
        'limits': (0, 1000),
    },
    {
        'name': 'metric',
        'type': 'list',
        'limits': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
        'value': 'euclidean',
    },
    {
        'name': 'algorithm',
        'type': 'list',
        'limits': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'value': 'auto',
    },
    {
        'name': 'sigma_xy',
        'type': 'float',
        'value': 15.0,
        'limits': (0.1, 100.0),
    },
    {
        'name': 'sigma_z',
        'type': 'float',
        'value': 45.0,
        'limits': (0.1, 100.0),
    },
    {
        'name': 'scale_data',
        'type': 'bool',
        'value': False,
    },
]

HDBSCAN_PARAMS = [
    {
        'name': 'min_samples',
        'type': 'int',
        'value': 0,
        'limits': (0, 1000),
    },
    {
        'name': 'min_cluster_size',
        'type': 'int',
        'value': 5,
        'limits': (2, 1000),
    },
    {
        'name': 'cluster_selection_epsilon_max',
        'type': 'float',
        'value': 0.0,
        'limits': (0.0, 1000.0),
    },
    {
        'name': 'sigma_xy',
        'type': 'float',
        'value': 15.0,
        'limits': (0.1, 100.0),
    },
    {
        'name': 'sigma_z',
        'type': 'float',
        'value': 45.0,
        'limits': (0.1, 100.0),
    },
    {
        'name': 'scale_data',
        'type': 'bool',
        'value': False,
    },
]

OPTICS_PARAMS = [
    {
        'name': 'min_samples',
        'type': 'int',
        'value': 5,
        'limits': (2, 1000),
    },
    {
        'name': 'xi',
        'type': 'float',
        'value': 0.05,
        'limits': (0.0, 1.0),
    },
    {
        'name': 'max_eps',
        'type': 'float',
        'value': 0.0,
        'limits': (0.0, 5000.0),
    },
    {
        'name': 'p',
        'type': 'float',
        'value': 2.0,
        'limits': (0.0, 10.0),
    },
    {
        'name': 'metric',
        'type': 'list',
        'limits': [
            'minkowski',
            'braycurtis',
            'canberra',
            'chebyshev',
            'correlation',
            'dice',
            'hamming',
            'jaccard',
            'kulsinski',
            'mahalanobis',
            'rogerstanimoto',
            'russellrao',
            'seuclidean',
            'sokalmichener',
            'sokalsneath',
            'sqeuclidean',
            'yule',
            'cityblock',
            'cosine',
            'euclidean',
            'l1',
            'l2',
            'manhattan',
        ],
        'value': 'minkowski',
    },
    {
        'name': 'cluster_method',
        'type': 'list',
        'limits': ['xi', 'dbscan'],
        'value': 'xi',
    },
    {
        'name': 'leaf_size',
        'type': 'int',
        'value': 30,
        'limits': (1, 1000),
    },
    {
        'name': 'algorithm',
        'type': 'list',
        'limits': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'value': 'auto',
    },
    {
        'name': 'sigma_xy',
        'type': 'float',
        'value': 15.0,
        'limits': (0.1, 100.0),
    },
    {
        'name': 'sigma_z',
        'type': 'float',
        'value': 45.0,
        'limits': (0.1, 100.0),
    },
    {
        'name': 'scale_data',
        'type': 'bool',
        'value': False,
    },
]

DENSITY_PARAMS = [
    {
        'name': 'method',
        'type': 'list',
        'limits': ['convex_hull', 'alpha_shape', 'voronoi'],
        'value': 'convex_hull',
    },
    {
        'name': 'alpha',
        'type': 'float',
        'value': 30.0,
        'limits': (0.1, 1000.0),
    },
    {
        'name': 'is_2d',
        'type': 'bool',
        'value': False,
    },
]

PARAMS_BY_MODE = {
    OptionsModes.RENDERING: {OptionsModes.RENDERING.value: RENDERING_PARAMS},
    OptionsModes.RULER: {
        OptionsModes.RULER.value: RULER_PARAMS,
    },
    OptionsModes.GRID: {
        OptionsModes.GRID.value: GRID_PARAMS,
    },
    OptionsModes.DBSCAN: {OptionsModes.DBSCAN.value: DBSCAN_PARAMS},
    OptionsModes.HDBSCAN: {OptionsModes.HDBSCAN.value: HDBSCAN_PARAMS},
    OptionsModes.OPTICS: {OptionsModes.OPTICS.value: OPTICS_PARAMS},
    OptionsModes.CLUSTER_DENSITY: {OptionsModes.CLUSTER_DENSITY.value: DENSITY_PARAMS},
}


class OptionsTree(Tree):
    def __init__(self, mode: OptionsModes, parent=None, **kwargs):
        self.mode = mode

        self.__export = kwargs.get('export', False)
        self.__batch = kwargs.get('batch', False)

        # last step
        super().__init__(parent)

    def create_parameters(self):

        groups = []

        params = PARAMS_BY_MODE.get(self.mode, [])

        for group_name, param_group in params.items():
            if group_name == '':
                groups.extend(param_group)
            else:
                groups.append(
                    {
                        'name': group_name,
                        'type': 'group',
                        'children': param_group,
                    }
                )

        if self.__export:
            groups.append(
                {
                    'name': 'Export',
                    'type': 'group',
                    'children': EXPORT_PARAMS,
                }
            )
            # set suffix based on mode
            # suffix_param = self.get_param('Export.save_suffix')
            # if suffix_param is not None:
            #     suffix_param.setValue(self.mode.name.lower())

        if self.__batch:
            groups.append(
                {
                    'name': 'Batch',
                    'type': 'group',
                    'children': BATCH_PARAMS,
                }
            )

        self.param_tree = Parameter.create(
            name='',
            type='group',
            children=groups,
        )
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    def values(self):
        result = {}

        for path in self.get_param_paths():
            parameter = self.get_param(path)
            if parameter is not None:
                split = path.split('.')
                if len(split) > 1:
                    group = split[0]
                    name = '.'.join(split[1:])
                    if group not in result:
                        result[group] = {}
                    result[group][name] = parameter.value()
                else:
                    result[path.split('.')[-1]] = parameter.value()

        return result

    def _update_mode_visibility(self):
        is_superimposed = self.get_param_value('mode') == 'superimposed'

        # Parameters visible only for superimposed rendering
        superimposed_params = (
            'rasterized',
            'render_mode',
            'pixel_size',
            'extent',
            'center_mode',
            'rejection_threshold',
            'min_points',
        )

        for name in superimposed_params:
            parameter = self.get_param(name)
            if parameter is not None:
                parameter.setOpts(visible=is_superimposed)

        aggregate = self.get_param('Batch.aggregate')
        if aggregate is not None:
            aggregate.setOpts(visible=is_superimposed)

    def change(self, param, changes):
        if any(parameter.name() == 'mode' for parameter, _, _ in changes):
            self._update_mode_visibility()


def rich_print_table(data: dict, title: str = None):
    console = rc.Console()
    table = rt.Table(title=title, show_header=True, header_style='bold magenta')
    table.add_column('Parameter')
    table.add_column('Value')

    for key, value in data.items():
        table.add_row(key, str(value))

    console.print(table)


class OptionsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, mode=OptionsModes.RENDERING, **kwargs):
        super().__init__(parent)

        self.__mode = mode
        self.tree = OptionsTree(mode=mode, parent=self, **kwargs)

        # set width and height
        self.setMinimumWidth(400)
        self.setMinimumHeight(600)
        # Title
        self.setWindowTitle(f'{self.__mode.value} Options')

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.tree)
        layout.addWidget(buttons)

    def values(self, mode: OptionsModes = None, verbose=False) -> dict:
        vals = self.tree.values()

        if mode is not None and mode.value in vals:
            # filter the values to only include the specified mode
            vals = {mode.value: vals.get(mode.value, {})}

        # flatten the dictionary to a single level
        # ignored group names too
        flat_vals = {}
        for group, params in vals.items():
            if isinstance(params, dict):
                for name, value in params.items():
                    flat_vals[name] = value
            else:
                flat_vals[group] = params

        if verbose:
            rich_print_table(flat_vals, title=f'{self.__mode.value} Options')

        return flat_vals

    def accept(self):
        vals = self.values(verbose=True)
        super().accept()
