import json
from enum import Enum

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
)
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter

from ..filters import BANDPASS_TYPES, AbstractFilter, BandpassFilter, DoG_Filter
from ..fitting.fit import AbstractDetector, CV_BlobDetector
from ..fitting.results import FittingMethod

FITTING_METHODS = {
    '2D Phasor-Fit (CPU)': FittingMethod._2D_Phasor_CPU,
    '2D MLE Gauss-Fit free sigma (GPU/CPU)': \
        FittingMethod._2D_Gauss_MLE_free_sigma,
    '2D MLE Gauss-Fit elliptical sigma (GPU/CPU)': \
        FittingMethod._2D_Gauss_MLE_elliptical_sigma,
    '3D MLE Gauss-Fit cspline (GPU/CPU)': \
        FittingMethod._3D_Gauss_MLE_cspline_sigma
}

DETECTORS = {'OpenCV Blob Detector': CV_BlobDetector}

FILTERS = {
    'Difference of Gaussians': DoG_Filter,
    'Fourier Bandpass Filter': BandpassFilter
}

class Parameters(Enum):
    AUTO_STRETCH = 'Options.Auto-Stretch'
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
    FILTER_TYPE = 'Filters.Fourier Bandpass Filter.Type'
    CENTER = 'Filters.Fourier Bandpass Filter.Center'
    WIDTH = 'Filters.Fourier Bandpass Filter.Width'
    SHOW_FILTER = 'Filters.Fourier Bandpass Filter.Show Filter'
    PREFIT_DETECTOR = 'Prefit.Prefit Detector'
    MIN_AREA = 'Prefit.OpenCV Blob Detector.Min Area'
    MAX_AREA = 'Prefit.OpenCV Blob Detector.Max Area'
    RELATIVE_THRESHOLD_MIN = 'Prefit.Relative Threshold.min'
    RELATIVE_THRESHOLD_MAX = 'Prefit.Relative Threshold.max'
    FITTING_METHOD = 'Localization.Fitting Method'
    ROI_SIZE = 'Localization.ROI Size'
    PIXEL_SIZE = 'Localization.Pixel-size'
    LOCALIZE_GPU = 'Localization.GPU'
    LOCALIZE = 'Localization.Localize'
    EXPORT_STATE = 'Settings.Export'
    IMPORT_STATE = 'Settings.Import'

class ImagePrefitWidget(ParameterTree):
    saveCropped = pyqtSignal()
    localizeData = pyqtSignal()
    paramsChanged = pyqtSignal(GroupParameter, list)
    roiEnabled = pyqtSignal()
    roiChanged = pyqtSignal(tuple)

    DETECTORS = list(DETECTORS.keys())
    FILTERS = list(FILTERS.keys())
    FITTING_METHODS = list(FITTING_METHODS.keys())
    BANDPASS_TYPES = BANDPASS_TYPES.values()

    def __init__(
            self, shape: tuple[int, int]=(128, 128), debug=False):
        super().__init__()

        self._debug = debug
        self._shape = shape

        self._filters = {key: None for key in self.FILTERS}
        self._detectors = {key: None for key in self.DETECTORS}

        # Create an initial parameter tree with a Layers group
        params = [
            {'name': 'Options', 'type': 'group', 'children': [
                {'name': 'Auto-Stretch', 'type': 'bool', 'value': True},
                {'name': 'Enable ROI', 'type': 'bool', 'value': False},
                {'name': 'ROI', 'type': 'group', 'children': [
                    {'name': 'x', 'type': 'int', 'value': 0,
                     'limits': [0, shape[1]], 'step': 1, 'suffix': 'pixel'},
                    {'name': 'y', 'type': 'int', 'value': 0,
                     'limits': [0, shape[0]], 'step': 1, 'suffix': 'pixel'},
                    {'name': 'width', 'type': 'int', 'value': shape[1],
                     'limits': [0, shape[1]], 'step': 1, 'suffix': 'pixel'},
                    {'name': 'height', 'type': 'int', 'value': shape[0],
                     'limits': [0, shape[0]], 'step': 1, 'suffix': 'pixel'},
                    ]},
                {'name': 'Realtime Localization', 'type': 'bool', 'value': False},
                {'name': 'Save Cropped Image', 'type': 'action'},
            ]},
            {'name': 'Filters', 'type': 'group', 'children': [
                {'name': 'Temporal Median Filter', 'type': 'group', 'children': [
                    {'name': 'Filter Enabled', 'type': 'bool', 'value': False},
                    {'name': 'Window Size', 'type': 'int', 'value': 3,
                     'limits': [1, 2048], 'step': 1, 'suffix': 'frame'},
                    ]},
                {'name': 'Filter', 'type': 'list',
                 'values': self.FILTERS,
                 'value': self.FILTERS[0]},
                {'name': self.FILTERS[0], 'type': 'group', 'children': [
                    {'name': 'Sigma', 'type': 'float', 'value': 1.0,
                     'limits': [0.0, 100.0], 'step': 0.1, 'decimals': 2,
                     'tip': '\u03C3 min'},
                    {'name': 'Factor', 'type': 'float', 'value': 2.5,
                     'limits': [0.0, 100.0], 'step': 0.1, 'decimals': 2,
                     'tip': '\u03C3 max/\u03C3 min'}
                    ]},
                {'name': self.FILTERS[1], 'type': 'group', 'children': [
                    {'name': 'Type', 'type': 'list',
                    'values': self.BANDPASS_TYPES,
                    'value': self.BANDPASS_TYPES[0]},
                    {'name': 'Center', 'type': 'float', 'value': 40.0,
                     'limits': [0.0, 2096.0], 'step': 0.5, 'decimals': 2,
                     'tip': 'The center of the band in pixels.'},
                    {'name': 'Width', 'type': 'float', 'value': 90.0,
                     'limits': [0.0, 2096.0], 'step': 0.5, 'decimals': 2,
                     'tip': 'The width of the band in pixels'},
                    {'name': 'Show Filter', 'type': 'bool', 'value': False},
                    ]},
            ]},
            {'name': 'Prefit', 'type': 'group', 'children': [
                {'name': 'Prefit Detector', 'type': 'list',
                 'values': self.DETECTORS,
                 'value': self.DETECTORS[0]},
                {'name': self.DETECTORS[0], 'type': 'group', 'children': [
                    {'name': 'Min Area', 'type': 'float', 'value': 1.5,
                     'limits': [0.0, 1000.0], 'step': 0.1, 'decimals': 2},
                    {'name': 'Max Area', 'type': 'float', 'value': 80.0,
                     'limits': [0.0, 1000.0], 'step': 0.1, 'decimals': 2},
                    ]},
                {'name': 'Relative Threshold', 'type': 'group', 'children': [
                    {'name': 'min', 'type': 'float', 'value': 0.5,
                     'limits': [0.0, 1.0], 'step': 0.01, 'decimals': 3},
                    {'name': 'max', 'type': 'float', 'value': 1.0,
                     'limits': [0.0, 1.0], 'step': 0.01, 'decimals': 3},
                    ]},
            ]},
            {'name': 'Localization', 'type': 'group', 'children': [
                {'name': 'Fitting Method', 'type': 'list',
                 'values': self.FITTING_METHODS,
                 'value': self.FITTING_METHODS[0]},
                {'name': 'GPU', 'type': 'bool', 'value': True,
                 'tip': 'Try GPU-accelerated fitting if possible.'},
                {'name': 'ROI Size', 'type': 'int', 'value': 13,
                    'limits': [7, 99], 'step': 2, 'suffix': 'pixel'},
                {'name': 'Pixel-size', 'type': 'float', 'value': 114.17,
                 'limits': [0.0, 20000], 'step': 1, 'decimals': 2,
                 'suffix': 'nm'},
                {'name': 'Localize', 'type': 'action'},
            ]},
            {'name': 'Settings', 'type': 'group', 'children': [
                {'name': 'Export', 'type': 'action'},
                {'name': 'Import', 'type': 'action'},
            ]},
        ]

        self.param_tree = Parameter.create(name='root', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.setParameters(self.param_tree, showTop=False)
        self.header().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)

        self.get_param(
            Parameters.LOCALIZE.value).sigActivated.connect(self.localizeData.emit)
        self.get_param(
            Parameters.IMPORT_STATE.value).sigActivated.connect(self.load_json)
        self.get_param(
            Parameters.EXPORT_STATE.value).sigActivated.connect(self.export_json)
        self.get_param(
            Parameters.SAVE_CROPPED_IMAGE.value).sigActivated.connect(self.saveCropped.emit)

    def get_param(self, param_name: str) -> Parameter:
        '''
        Get a parameter by name from the cache.
        '''
        return self.param_tree.param(*param_name.split('.'))

    def get_param_path(self, param: Parameter):
        return self.param_tree.childPath(param)

    def change(self, param: Parameter, changes: list):
        # Check if 'Enable ROI' parameter is changed
        if any('Enable ROI' in param.name() for param, _, _ in changes):
            self.roiEnabled.emit()

        if any(param.name() in ['x', 'y', 'width', 'height'
                                ] for param, _, _ in changes):
            self.roiChanged.emit((
                self.get_param(Parameters.ROI_X.value).value(),
                self.get_param(Parameters.ROI_Y.value).value(),
                self.get_param(Parameters.ROI_WIDTH.value).value(),
                self.get_param(Parameters.ROI_HEIGHT.value).value()))

        if any(
            'Localization' not in self.get_param_path(param) and \
            'Settings' not in self.get_param_path(param) for param, _, _ in changes):
            self.paramsChanged.emit(param, changes)

        if not self._debug:
            return

        print('tree changes:')
        for param, change, data in changes:
            path = self.param_tree.childPath(param)
            if len(path) > 1:
                print('  parameter: %s'% path[-1])
                print('  parent: %s'% path[-2])
            else:
                childName = '.'.join(path) if path is not None else param.name()
                print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

    def set_roi(self, x: int, y: int, width: int, height: int):
        self.get_param(Parameters.ROI_X.value).setValue(x)
        self.get_param(Parameters.ROI_Y.value).setValue(y)
        self.get_param(Parameters.ROI_WIDTH.value).setValue(width)
        self.get_param(Parameters.ROI_HEIGHT.value).setValue(height)

    def get_image_filter(self) -> AbstractFilter:
        value = self.get_param(Parameters.FILTER.value).value()
        filter_class = FILTERS.get(value, None)
        if filter_class is DoG_Filter:
            filter: DoG_Filter = self._filters.get(value, None)
            if filter is None:
                self._filters[value] = DoG_Filter(
                    self.get_param(Parameters.SIGMA.value).value(),
                    self.get_param(Parameters.FACTOR.value).value()
                )
            else:
                filter.set_params(
                    self.get_param(Parameters.SIGMA.value).value(),
                    self.get_param(Parameters.FACTOR.value).value()
                )
            return self._filters[value]
        elif filter_class is BandpassFilter:
            filter: BandpassFilter = self._filters.get(value, None)
            if filter is None:
                self._filters[value] = BandpassFilter(
                    self.get_param(Parameters.CENTER.value).value(),
                    self.get_param(Parameters.WIDTH.value).value(),
                    self.get_param(Parameters.FILTER_TYPE.value).value(),
                    self.get_param(Parameters.SHOW_FILTER.value).value()
                )
            else:
                filter.set_params(
                    self.get_param(Parameters.CENTER.value).value(),
                    self.get_param(Parameters.WIDTH.value).value(),
                    self.get_param(Parameters.FILTER_TYPE.value).value(),
                    self.get_param(Parameters.SHOW_FILTER.value).value()
                )
            return self._filters[value]
        else:
            return AbstractFilter()

    def get_detector(self) -> AbstractDetector:
        value = self.get_param(Parameters.PREFIT_DETECTOR.value).value()
        detector_class = DETECTORS.get(value, None)
        if detector_class is CV_BlobDetector:
            detector: CV_BlobDetector = self._detectors.get(value, None)
            if detector is None:
                self._detectors[value] = CV_BlobDetector(
                    minArea=self.get_param(Parameters.MIN_AREA.value).value(),
                    maxArea=self.get_param(Parameters.MAX_AREA.value).value()
                )
            else:
                detector.set_blob_detector_params(
                    minArea=self.get_param(Parameters.MIN_AREA.value).value(),
                    maxArea=self.get_param(Parameters.MAX_AREA.value).value()
                )
            return self._detectors[value]
        else:
            return AbstractDetector()

    def get_fitting_method(self):
        value = self.get_param(Parameters.FITTING_METHOD.value).value()
        return FITTING_METHODS[value]

    def export_json(self):
        filename, _ = QFileDialog.getSaveFileName(
            None,
            'Save Parameters', '', 'JSON Files (*.json);;All Files (*)')
        if not filename:
            return  # User canceled the operation

        state = self.param_tree.saveState(filter='user')
        with open(filename, 'w') as file:
            json.dump(state, file, indent=2)

    # Load parameters from JSON
    def load_json(self):
        filename, _ = QFileDialog.getOpenFileName(
            None, 'Load Parameters',
            '', 'JSON Files (*.json);;All Files (*)')
        if not filename:
            return  # User canceled the operation

        with open(filename) as file:
            state = json.load(file)
        self.param_tree.restoreState(state)

if __name__ == '__main__':
    app = QApplication([])
    my_app = ImagePrefitWidget()
    my_app.show()
    app.exec_()
