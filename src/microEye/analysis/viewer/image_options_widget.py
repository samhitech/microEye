import inspect
import json
from enum import Enum
from typing import Optional

from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter

from microEye.analysis.filters import (
    DoG_Filter,
    FourierFilter,
    PassFilter,
    SpatialFilter,
)
from microEye.analysis.fitting.fit import AbstractDetector, CV_BlobDetector
from microEye.analysis.fitting.results import FittingMethod
from microEye.qt import QApplication, QtWidgets, Signal, Slot
from microEye.utils import Tree

DETECTORS = {'OpenCV Blob Detector': CV_BlobDetector}

FILTERS = {cls().name: cls for cls in SpatialFilter.__subclasses__()}
'''
Dictionary of available filters.

Keys are the names of the filters, and values are the filter classes.

_Do not change the order of the keys in this dictionary._'''


class Parameters(Enum):
    AUTO_STRETCH = 'Options.Auto-Stretch'
    CHANNEL = 'Options.Channel'
    ENABLE_ROI = 'Options.Enable ROI'
    ROI_X = 'Options.ROI.x'
    ROI_Y = 'Options.ROI.y'
    ROI_WIDTH = 'Options.ROI.width'
    ROI_HEIGHT = 'Options.ROI.height'
    REALTIME_LOCALIZATION = 'Options.Realtime Localization'
    SAVE_CROPPED_IMAGE = 'Options.Save Cropped Image'
    TM_FILTER_ENABLED = 'Filters.Temporal Median Filter.Filter Enabled'
    TM_FILTER_WINDOW_SIZE = 'Filters.Temporal Median Filter.Window Size'
    FILTER = 'Filters.Filter'
    SIGMA = 'Filters.Difference of Gaussians.Sigma'
    FACTOR = 'Filters.Difference of Gaussians.Factor'
    FILTER_TYPE = 'Filters.Fourier Filter.Type'
    CENTER = 'Filters.Fourier Filter.Center'
    WIDTH = 'Filters.Fourier Filter.Width'
    SHOW_FILTER = 'Filters.Fourier Filter.Show Filter'
    PREFIT_DETECTOR = 'Prefit.Prefit Detector'
    MIN_AREA = 'Prefit.OpenCV Blob Detector.Min Area'
    MAX_AREA = 'Prefit.OpenCV Blob Detector.Max Area'
    RELATIVE_THRESHOLD_MIN = 'Prefit.Relative Threshold.min'
    RELATIVE_THRESHOLD_MAX = 'Prefit.Relative Threshold.max'
    FITTING_METHOD = 'Localization.Fitting Method'
    ROI_SIZE = 'Localization.ROI Size'
    PIXEL_SIZE = 'Localization.Pixel-size'
    INITIAL_SIGMA = 'Localization.Initial Sigma'
    LOCALIZE_GPU = 'Localization.GPU'
    LOCALIZE = 'Localization.Localize'
    FIELD_DEPENDENCY = 'PSF.Field Dependency'
    FIELD_GRID_SIZE = 'PSF.Field Dependency.Grid Size'
    FIELD_ENABLED = 'PSF.Field Dependency.Enabled'
    ZERO_PLANE = 'PSF.Zero Plane'
    Z0_ENABLED = 'PSF.Zero Plane.Enabled'
    Z0_METHOD = 'PSF.Zero Plane.Method'
    PSF_ZSTEP = 'PSF.Z Step'
    PSF_UPSAMPLE = 'PSF.Upsample'
    EXTRACT_PSF = 'PSF.Extract'
    EXPORT_STATE = 'Settings.Export'
    IMPORT_STATE = 'Settings.Import'


class FittingOptions(Tree):
    saveCropped = Signal()
    localizeData = Signal()
    extractPSF = Signal()
    paramsChanged = Signal()
    roiEnabled = Signal()
    roiChanged = Signal(tuple)

    DETECTORS = list(DETECTORS.keys())
    FILTERS = list(FILTERS.keys())
    FITTING_METHODS = FittingMethod.get_strings()

    def __init__(
        self,
        shape: tuple = (1, 1, 1, 128, 128),
        debug=False,
        parent: Optional['QtWidgets.QWidget'] = None,
    ):
        self._debug = debug

        if shape is None or len(shape) < 5:
            raise ValueError('Shape must be a 5-tuple')

        self._shape = shape

        self._filters = {key: None for key in self.FILTERS}
        self._detectors = {key: None for key in self.DETECTORS}

        super().__init__(parent)

    @property
    def imageWidth(self):
        length = len(self._shape)
        return self._shape[length - 1]

    @property
    def imageHeight(self):
        length = len(self._shape)
        return self._shape[length - 2]

    @property
    def imageDepth(self):
        length = len(self._shape)
        return self._shape[length - 3]

    @property
    def imageChannels(self):
        length = len(self._shape)
        return self._shape[length - 4]

    @property
    def imageFrames(self):
        return self._shape[0]

    def create_parameters(self):
        # Create an initial parameter tree with a Layers group
        params = [
            {
                'name': 'Options',
                'type': 'group',
                'children': [
                    {'name': 'Auto-Stretch', 'type': 'bool', 'value': True},
                    {'name': 'Enable ROI', 'type': 'bool', 'value': False},
                    {
                        'name': 'ROI',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'x',
                                'type': 'int',
                                'value': 0,
                                'limits': [0, self.imageWidth],
                                'step': 1,
                                'suffix': 'pixel',
                            },
                            {
                                'name': 'y',
                                'type': 'int',
                                'value': 0,
                                'limits': [0, self.imageHeight],
                                'step': 1,
                                'suffix': 'pixel',
                            },
                            {
                                'name': 'width',
                                'type': 'int',
                                'value': self.imageWidth,
                                'limits': [0, self.imageWidth],
                                'step': 1,
                                'suffix': 'pixel',
                            },
                            {
                                'name': 'height',
                                'type': 'int',
                                'value': self.imageHeight,
                                'limits': [0, self.imageHeight],
                                'step': 1,
                                'suffix': 'pixel',
                            },
                        ],
                    },
                    {'name': 'Realtime Localization', 'type': 'bool', 'value': False},
                    {'name': 'Save Cropped Image', 'type': 'action'},
                ],
            },
            {
                'name': 'Filters',
                'type': 'group',
                'children': [
                    {
                        'name': 'Temporal Median Filter',
                        'type': 'group',
                        'children': [
                            {'name': 'Filter Enabled', 'type': 'bool', 'value': False},
                            {
                                'name': 'Window Size',
                                'type': 'int',
                                'value': 3,
                                'limits': [1, 2048],
                                'step': 1,
                                'suffix': 'frame',
                            },
                        ],
                    },
                    {
                        'name': 'Filter',
                        'type': 'list',
                        'limits': self.FILTERS,
                        'value': self.FILTERS[0],
                        'tip': 'Select the type of filter to apply',
                    },
                    *[
                        cls().get_tree_parameters()
                        for cls in SpatialFilter.__subclasses__()
                        if cls().get_tree_parameters()
                    ],
                ],
            },
            {
                'name': 'Prefit',
                'type': 'group',
                'children': [
                    {
                        'name': 'Prefit Detector',
                        'type': 'list',
                        'limits': self.DETECTORS,
                        'value': self.DETECTORS[0],
                    },
                    {
                        'name': self.DETECTORS[0],
                        'type': 'group',
                        'children': [
                            {
                                'name': 'Min Area',
                                'type': 'float',
                                'value': 1.5,
                                'limits': [0.0, 1000.0],
                                'step': 0.1,
                                'decimals': 2,
                                'tip': 'Minimum area of the detected object.',
                            },
                            {
                                'name': 'Max Area',
                                'type': 'float',
                                'value': 80.0,
                                'limits': [0.0, 1000.0],
                                'step': 0.1,
                                'decimals': 2,
                                'tip': 'Maximum area of the detected object.',
                            },
                        ],
                    },
                    {
                        'name': 'Relative Threshold',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'min',
                                'type': 'float',
                                'value': 0.5,
                                'limits': [0.0, 1.0],
                                'step': 0.01,
                                'decimals': 3,
                                'tip': 'Minimum relative threshold for detection.',
                            },
                            {
                                'name': 'max',
                                'type': 'float',
                                'value': 1.0,
                                'limits': [0.0, 1.0],
                                'step': 0.01,
                                'decimals': 3,
                                'tip': 'Maximum relative threshold for detection.',
                            },
                        ],
                    },
                ],
            },
            {
                'name': 'Localization',
                'type': 'group',
                'children': [
                    {
                        'name': 'Fitting Method',
                        'type': 'list',
                        'limits': self.FITTING_METHODS,
                        'value': self.FITTING_METHODS[0],
                    },
                    {
                        'name': 'GPU',
                        'type': 'bool',
                        'value': True,
                        'tip': 'Try GPU-accelerated fitting, if available.',
                    },
                    {
                        'name': 'ROI Size',
                        'type': 'int',
                        'value': 13,
                        'limits': [7, 201],
                        'step': 2,
                        'suffix': 'pixel',
                        'tip': 'Size of the fitting box in pixels.',
                    },
                    {
                        'name': 'Pixel-size',
                        'type': 'float',
                        'value': 114.17,
                        'limits': [0.0, 20000],
                        'step': 1,
                        'decimals': 2,
                        'suffix': 'nm',
                        'tip': 'Size of a pixel in nanometers.',
                    },
                    {
                        'name': 'Initial Sigma',
                        'type': 'float',
                        'value': 1.0,
                        'limits': [0.4, 10],
                        'step': 0.1,
                        'decimals': 3,
                        'suffix': 'pixel',
                        'tip': 'Initial sigma for fitting.',
                    },
                    {'name': 'Localize', 'type': 'action'},
                ],
            },
            {
                'name': 'PSF',
                'type': 'group',
                'children': [
                    {
                        'name': 'Field Dependency',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'Enabled',
                                'type': 'bool',
                                'value': False,
                                'tip': 'Toggle to enable or disable this option',
                            },
                            {
                                'name': 'Grid Size',
                                'type': 'int',
                                'value': 3,
                                'limits': [3, 15],
                                'step': 1,
                                'suffix': 'pixel',
                                'tip': 'Size of the grid for field dependency.',
                            },
                        ],
                    },
                    {
                        'name': 'Zero Plane',
                        'type': 'group',
                        'children': [
                            {
                                'name': 'Enabled',
                                'type': 'bool',
                                'value': False,
                                'tip': 'Toggle to enable or disable this option',
                            },
                            {
                                'name': 'Method',
                                'type': 'list',
                                'limits': [
                                    'All',
                                    'Intensity',
                                    'Min Sigma',
                                    'Min Sum Sigma',
                                ],
                                'value': 'All',
                                'tip': 'Select the method to determine the zero plane.',
                            },
                        ],
                        'tip': (
                            'If checked, the algorithm searches for Z=0 and ROIs are'
                            + ' extracted up and down; otherwise, they are extracted'
                            + ' from each slice localizations.'
                        ),
                    },
                    {
                        'name': 'Z Step',
                        'type': 'int',
                        'value': 10,
                        'limits': [1, 1000],
                        'step': 1,
                        'suffix': 'nm',
                        'tip': 'Step size along the Z axis.',
                    },
                    {
                        'name': 'Upsample',
                        'type': 'int',
                        'value': 1,
                        'limits': [1, 200],
                        'step': 1,
                        'tip': 'Factor by which to upsample the image.',
                    },
                    {
                        'name': 'Extract',
                        'type': 'action',
                        'tip': 'Extract the PSF data from the stack.',
                    },
                ],
            },
            {
                'name': 'Settings',
                'type': 'group',
                'children': [
                    {
                        'name': 'Export',
                        'type': 'action',
                        'tip': 'Export the current settings.',
                    },
                    {
                        'name': 'Import',
                        'type': 'action',
                        'tip': 'Import settings from a file.',
                    },
                ],
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.get_param(Parameters.LOCALIZE).sigActivated.connect(self.onLocalize)
        self.get_param(Parameters.EXTRACT_PSF).sigActivated.connect(self.onExtractPSF)
        self.get_param(Parameters.IMPORT_STATE).sigActivated.connect(self.load_json)
        self.get_param(Parameters.EXPORT_STATE).sigActivated.connect(self.export_json)
        self.get_param(Parameters.SAVE_CROPPED_IMAGE).sigActivated.connect(
            self.onCropped
        )

    @Slot(object)
    def onCropped(self, action: Parameter):
        self.saveCropped.emit()

    @Slot(object)
    def onLocalize(self, action: Parameter):
        self.localizeData.emit()

    @Slot(object)
    def onExtractPSF(self, action: Parameter):
        self.extractPSF.emit()

    @Slot(object, object)
    def change(self, param: Parameter, changes: list):
        # Check if 'Enable ROI' parameter is changed
        if any('Enable ROI' in param.name() for param, _, _ in changes):
            self.roiEnabled.emit()

        if any('Filter' in param.name() for param, _, _ in changes):
            filter_ = self.get_param_value(Parameters.FILTER)
            for name in self.FILTERS:
                if self.get_param(f'Filters.{name}'):
                    self.get_param(f'Filters.{name}').setOpts(visible=name == filter_)

        if any(
            param.name() in ['x', 'y', 'width', 'height'] for param, _, _ in changes
        ):
            self.roiChanged.emit(
                (
                    self.get_param_value(Parameters.ROI_X),
                    self.get_param_value(Parameters.ROI_Y),
                    self.get_param_value(Parameters.ROI_WIDTH),
                    self.get_param_value(Parameters.ROI_HEIGHT),
                )
            )

        if any(
            'Localization' not in self.get_param_path(param)
            and 'Settings' not in self.get_param_path(param)
            for param, _, _ in changes
        ):
            self.paramsChanged.emit()

        if not self._debug:
            return

        print('tree changes:')
        for param, change, data in changes:
            path = self.get_param_path(param)
            if len(path) > 1:
                print(f'  parameter: {path[-1]}')
                print(f'  parent: {path[-2]}')
            else:
                childName = '.'.join(path) if path is not None else param.name()
                print(f'  parameter: {childName}')
            print(f'  change:    {change}')
            print(f'  data:      {str(data)}')
            print('  ----------')

    def set_roi(self, x: int, y: int, width: int, height: int):
        try:
            self.param_tree.sigTreeStateChanged.disconnect(self.change)
            self.set_param_value(Parameters.ROI_X, x)
            self.set_param_value(Parameters.ROI_Y, y)
            self.set_param_value(Parameters.ROI_WIDTH, width)
            self.set_param_value(Parameters.ROI_HEIGHT, height)
        finally:
            self.param_tree.sigTreeStateChanged.connect(self.change)

    def get_image_filter(self) -> SpatialFilter:
        value = self.get_param_value(Parameters.FILTER)
        filter_class = FILTERS.get(value)

        path_str = f'Filters.{value}'

        def get_arg_names(method):
            '''Get the names of arguments for the method.'''
            init_signature = inspect.signature(method)
            return [
                param.name
                for param in init_signature.parameters.values()
                if param.name != 'self'  # Exclude 'self'
            ]

        filter_: SpatialFilter = self._filters.get(value, None)

        if filter_ is None:
            # Create a new filter instance if it doesn't exist
            self._filters[value] = filter_class(
                **{
                    param: self.get_param_value(
                        f"{path_str}.{param.replace('_', ' ').title()}"
                    )
                    for param in get_arg_names(filter_class.__init__)
                }
            )
        elif hasattr(filter_class, 'set_params'):
            # If the filter class has a set_params method, use it to update parameters
            filter_.set_params(
                **{
                    param: self.get_param_value(
                        f"{path_str}.{param.replace('_', ' ').title()}"
                    )
                    for param in get_arg_names(filter_.set_params)
                }
            )

        return self._filters.get(value, None)

    def get_detector(self) -> AbstractDetector:
        value = self.get_param_value(Parameters.PREFIT_DETECTOR)
        detector_class = DETECTORS.get(value)
        if detector_class is CV_BlobDetector:
            detector: CV_BlobDetector = self._detectors.get(value, None)
            if detector is None:
                self._detectors[value] = CV_BlobDetector(
                    minArea=self.get_param_value(Parameters.MIN_AREA),
                    maxArea=self.get_param_value(Parameters.MAX_AREA),
                )
            else:
                detector.set_blob_detector_params(
                    minArea=self.get_param_value(Parameters.MIN_AREA),
                    maxArea=self.get_param_value(Parameters.MAX_AREA),
                )
            return self._detectors[value]
        else:
            return AbstractDetector()

    def get_fitting_method(self):
        value = self.get_param_value(Parameters.FITTING_METHOD)
        return FittingMethod.from_string(value)


if __name__ == '__main__':
    app = QApplication([])
    my_app = FittingOptions()
    my_app.show()
    app.exec()
