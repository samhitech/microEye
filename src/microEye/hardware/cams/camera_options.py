import json
import os
from enum import Enum
from typing import Any, Optional, Union

import ome_types.model as om
from ome_types.model import *
from ome_types.model.simple_types import PixelType, UnitsLength, UnitsTime
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import ActionParameter, GroupParameter


class CamParams(Enum):
    '''
    Enum class defining Camera parameters.
    '''
    CAMERA_OPTIONS = 'Camera'
    CAMERA_GPIO = 'GPIOs'
    CAMERA_TIMERS = 'Timers'
    EXPOSURE = 'Camera.Exposure Time'
    EXPERIMENT_NAME = 'Options.Experiment Name'
    FRAMES = 'Options.Number of Frames'
    SAVE_DIRECTORY = 'Options.Save Directory'
    SAVE_DATA = 'Options.Save Data'
    DARK_CALIBRATION = 'Options.Dark Calibration'
    IMAGE_FORMAT = 'Options.Image Format'
    TIFF_FORMAT = 'Options.Tiff Format'
    ZARR_FORMAT = 'Options.Zarr Format'
    BIGG_TIFF_FORMAT = 'Options.BiggTiff Format'
    FULL_METADATA = 'Options.Full Metadata'
    CAPTURE_STATS = 'Stats.Capture'
    DISPLAY_STATS = 'Stats.Display'
    SAVE_STATS = 'Stats.Save'
    TEMPERATURE = 'Stats.Temperature'
    ROI_X = 'Region of Interest (ROI).X'
    ROI_Y = 'Region of Interest (ROI).Y'
    ROI_WIDTH = 'Region of Interest (ROI).Width'
    ROI_HEIGHT = 'Region of Interest (ROI).Height'
    SET_ROI = 'Region of Interest (ROI).Set ROI'
    RESET_ROI = 'Region of Interest (ROI).Reset ROI'
    CENTER_ROI = 'Region of Interest (ROI).Center ROI'
    SELECT_ROI = 'Region of Interest (ROI).Select ROI'
    SELECT_EXPORT_ROIS = 'Region of Interest (ROI).Select Export ROIs'
    EXPORT_ROIS = 'Region of Interest (ROI).Export ROIs'
    PREVIEW = 'Display.Preview'
    DISPLAY_STATS_OPTION = 'Display.Display Stats'
    AUTO_STRETCH = 'Display.Auto Stretch'
    VIEW_OPTIONS = 'Display.View Options'
    SINGLE_VIEW = 'Display.View Options.Single View'
    DUAL_SIDE = 'Display.View Options.Dual Channel (Side by Side)'
    DUAL_OVERLAID = 'Display.View Options.Dual Channel (Overlapped)'
    LINE_PROFILER = 'Display.Line Profiler'
    LUT = 'Display.LUT'
    LUT_NUMPY = 'Display.LUT Numpy (12bit)'
    LUT_OPENCV = 'Display.LUT Opencv (8bit)'
    RESIZE_DISPLAY = 'Display.Resize Display'

    EXPORT_STATE = 'Export State'
    IMPORT_STATE = 'Import State'

    def __str__(self):
        '''
        Return the last part of the enum value (Param name).
        '''
        return self.value.split('.')[-1]

    def get_path(self):
        '''
        Return the full parameter path.
        '''
        return self.value.split('.')

class CameraOptions(ParameterTree):
    '''
    Tree widget for editing camera parameters.

    Attributes
    ----------
    paramsChanged : pyqtSignal
        Signal for parameter changed event.
    '''

    paramsChanged = pyqtSignal(GroupParameter, list)
    '''Signal emitted when parameters are changed.

    Parameters
    ----------
    GroupParameter
        The group parameter that was changed.
    list
        A list of changes made to the parameter.
    '''
    setROI = pyqtSignal()
    resetROI = pyqtSignal()
    centerROI = pyqtSignal()
    selectROI = pyqtSignal()
    selectROIs = pyqtSignal()
    directoryChanged = pyqtSignal(str)



    def __init__(self, parent: Optional['QWidget'] = None):
        '''
        Initialize the CameraOptions.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None.
        '''
        super().__init__()

        self.setMinimumWidth(50)
        self.create_parameters()
        self.setParameters(self.param_tree, showTop=False)

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {'name': str(CamParams.CAMERA_OPTIONS), 'type': 'group', 'children': [
                {'name': str(CamParams.EXPOSURE), 'type': 'float',
                 'value': 100.0, 'dec': False, 'decimals': 6,
                 'suffixes': [' ns', ' us', ' ms', ' s']},
            ]},
            {'name': 'Options', 'type': 'group', 'children': [
                {'name': str(CamParams.EXPERIMENT_NAME),
                'type': 'str', 'value': 'Experiment_001'},
                {'name': str(CamParams.FRAMES),
                'type': 'int', 'value': 1e6, 'limits': [1, 1e9]},
                {'name': str(CamParams.SAVE_DIRECTORY), 'type': 'file',
                 'directory': os.path.dirname(os.path.realpath(__file__)),
                 'fileMode': 'DirectoryOnly'},
                {'name': str(CamParams.SAVE_DATA), 'type': 'bool', 'value': False},
                {'name': str(CamParams.DARK_CALIBRATION),
                 'type': 'bool', 'value': False},
                {'name': str(CamParams.IMAGE_FORMAT), 'type': 'list', 'values': [
                    str(CamParams.BIGG_TIFF_FORMAT), str(CamParams.TIFF_FORMAT),
                    str(CamParams.ZARR_FORMAT)
                ]},
                {'name': str(CamParams.FULL_METADATA), 'type': 'bool', 'value': True},
                ]},
            {'name': 'Display', 'type': 'group', 'children': [
                {'name': str(CamParams.PREVIEW), 'type': 'bool', 'value': True},
                {'name': str(CamParams.DISPLAY_STATS_OPTION),
                 'type': 'bool', 'value': False},
                {'name': str(CamParams.AUTO_STRETCH), 'type': 'bool', 'value': True},
                {'name': str(CamParams.LUT), 'type': 'list', 'values': [
                    str(CamParams.LUT_NUMPY), str(CamParams.LUT_OPENCV)
                ]},
                {'name': str(CamParams.VIEW_OPTIONS), 'type': 'list', 'values': [
                    str(CamParams.SINGLE_VIEW),
                    str(CamParams.DUAL_SIDE), str(CamParams.DUAL_OVERLAID)]},
                {'name': str(CamParams.LINE_PROFILER), 'type': 'bool', 'value': False},
                {'name': str(CamParams.RESIZE_DISPLAY),
                'type': 'float', 'value': 0.5, 'limits': [0.1, 4.0],
                'step': 0.02, 'dec': False},
            ]},
            {'name': 'Stats', 'type': 'group', 'children': [
                {'name': str(CamParams.CAPTURE_STATS),
                'type': 'str', 'value': '0 | 0.00 ms', 'readonly': True},
                {'name': str(CamParams.DISPLAY_STATS),
                'type': 'str', 'value': '0 | 0.00 ms', 'readonly': True},
                {'name': str(CamParams.SAVE_STATS),
                'type': 'str', 'value': '0 | 0.00 ms', 'readonly': True},
                {'name': str(CamParams.TEMPERATURE),
                'type': 'str', 'value': ' T -127.00 °C', 'readonly': True},
            ]},
            {'name': 'Region of Interest (ROI)', 'type': 'group', 'children': [
                {'name': str(CamParams.ROI_X), 'type': 'int', 'value': 0},
                {'name': str(CamParams.ROI_Y), 'type': 'int', 'value': 0},
                {'name': str(CamParams.ROI_WIDTH), 'type': 'int', 'value': 0},
                {'name': str(CamParams.ROI_HEIGHT), 'type': 'int', 'value': 0},
                {'name': str(CamParams.SET_ROI), 'type': 'action'},
                {'name': str(CamParams.RESET_ROI), 'type': 'action'},
                {'name': str(CamParams.CENTER_ROI), 'type': 'action'},
                {'name': str(CamParams.SELECT_ROI), 'type': 'action'},
                {'name': str(CamParams.SELECT_EXPORT_ROIS), 'type': 'action'},
                {'name': str(CamParams.EXPORT_ROIS), 'type': 'group', 'children': [
                ]},
            ]},
            {'name': str(CamParams.CAMERA_GPIO), 'type': 'group', 'children': [
            ]},
            {'name': str(CamParams.CAMERA_TIMERS), 'type': 'group', 'children': [
            ]},
            {'name': str(CamParams.EXPORT_STATE), 'type': 'action'},
            {'name': str(CamParams.IMPORT_STATE), 'type': 'action'},
        ]

        self.param_tree = Parameter.create(name='root', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)

        self.get_param(
            CamParams.SAVE_DIRECTORY).sigValueChanged.connect(
                lambda: self.directoryChanged.emit(
                    self.get_param_value(CamParams.SAVE_DIRECTORY)
                    ))

        self.get_param(
            CamParams.SET_ROI).sigActivated.connect(lambda: self.setROI.emit())
        self.get_param(
            CamParams.RESET_ROI).sigActivated.connect(lambda: self.resetROI.emit())
        self.get_param(
            CamParams.CENTER_ROI).sigActivated.connect(lambda: self.centerROI.emit())
        self.get_param(
            CamParams.SELECT_ROI).sigActivated.connect(lambda: self.selectROI.emit())
        self.get_param(
            CamParams.SELECT_EXPORT_ROIS).sigActivated.connect(
                lambda: self.selectROIs.emit())

        self.get_param(
            CamParams.IMPORT_STATE).sigActivated.connect(self.load_json)
        self.get_param(
            CamParams.EXPORT_STATE).sigActivated.connect(self.export_json)

    def get_param(
            self, param: CamParams
            ) -> Union[Parameter, ActionParameter]:
        '''Get a parameter by name.

        Parameters
        ----------
        param : CamParams
            Camera parameter.

        Returns
        -------
        Union[Parameter, ActionParameter]
            Retrieved parameter.
        '''
        return self.param_tree.param(*param.value.split('.'))

    def get_param_value(
            self, param: CamParams):
        '''Get a parameter value by name.

        Parameters
        ----------
        param : CamParams
            Camera parameter.

        Returns
        -------
        Any
            The value of the parameter.
        '''
        return self.param_tree.param(*param.value.split('.')).value()

    def set_param_value(
            self, param: CamParams, value, blockSignals: Union[Any, None]= None):
        '''
        Set a camera parameter value by name.

        Parameters
        ----------
        param : CamParams
            Camera parameter.
        value : Any
            The value to set.
        blockSignals : Union[Any, None], optional
            If provided, signals for parameter changes are blocked during the update.
            The interpretation of the value is left to the implementation.

        Returns
        -------
        bool
            True if the value is set successfully, False otherwise.
        '''
        try:
            parameter: Parameter = self.param_tree.param(*param.value.split('.'))
            parameter.setValue(value, blockSignals)
        except Exception:
            import traceback
            traceback.print_exc()
            return False
        else:
            return True

    def get_param_path(self, param: Parameter):
        '''
        Get the child path of a parameter in the parameter tree.

        Parameters
        ----------
        param : Parameter
            The parameter for which to retrieve the child path.

        Returns
        -------
        list
            The child path of the parameter.
        '''
        return self.param_tree.childPath(param)

    def add_param_child(self, parent: CamParams, value: dict):
        '''
        Add a child parameter to the specified parent parameter.

        Parameters
        ----------
        parent : CamParams
            The parent parameter to which the child will be added.
        value : dict
            A dictionary representing the child parameter. It should contain at
            least the following key-value pairs:
            - 'name': str, the name of the parameter.
            - 'type': str, the type of the parameter (e.g., 'int', 'float', 'str').
            - 'value': Any, the initial value of the parameter.
            Additional optional keys can be included based on the desired configuration.

        Returns
        -------
        None
        '''
        parent = self.get_param(parent)
        parent.addChild(value, autoIncrementName=True)

    def get_children(self, param: CamParams):
        '''
        Get the values of all children of a specified parameter.

        Parameters
        ----------
        param : CamParams
            The parameter whose children's values will be retrieved.

        Returns
        -------
        list
            List of values of all children of the specified parameter.
        '''
        res = []
        param = self.get_param(param)
        if isinstance(param, GroupParameter):
            for child in param.children():
                res.append(child.value())
        return res

    def change(self, param: Parameter, changes: list):
        '''
        Handle parameter changes as needed.

        Parameters
        ----------
        param : Parameter
            The parameter that triggered the change.
        changes : list
            List of changes.

        Returns
        -------
        None
        '''
        # Handle parameter changes as needed
        pass

    def export_json(self):
        '''
        Export parameters to a JSON file.

        Returns
        -------
        None
        '''
        filename, _ = QFileDialog.getSaveFileName(
            None,
            'Save Parameters', '', 'JSON Files (*.json);;All Files (*)')
        if not filename:
            return  # User canceled the operation

        state = self.param_tree.saveState()
        with open(filename, 'w', encoding='utf8') as file:
            json.dump(state, file, indent=2)

    # Load parameters from JSON
    def load_json(self):
        '''
        Load parameters from a JSON file.

        Returns
        -------
        None
        '''
        filename, _ = QFileDialog.getOpenFileName(
            None, 'Load Parameters',
            '', 'JSON Files (*.json);;All Files (*)')
        if not filename:
            return  # User canceled the operation

        with open(filename, encoding='utf8') as file:
            state = json.load(file)
        self.param_tree.restoreState(state, blockSignals=False)

    def get_roi_info(self, vimba=False):
        '''
        Get the region of interest (ROI) information.

        Parameters
        ----------
        vimba : bool, optional
            If True, returns ROI information suitable for Vimba API.
            If False (default), returns ROI information in the default order.

        Returns
        -------
        tuple
            Tuple containing ROI X, ROI Y, ROI width, and ROI height.
            If vimba is True, the order is (width, height, x, y).
            If vimba is False, the order is (x, y, width, height).
        '''
        if not vimba:
            info = [
                self.get_param_value(CamParams.ROI_X),
                self.get_param_value(CamParams.ROI_Y),
                self.get_param_value(CamParams.ROI_WIDTH),
                self.get_param_value(CamParams.ROI_HEIGHT),
            ]
            return info
        else:
            info = [
                self.get_param_value(CamParams.ROI_WIDTH),
                self.get_param_value(CamParams.ROI_HEIGHT),
                self.get_param_value(CamParams.ROI_X),
                self.get_param_value(CamParams.ROI_Y),
            ]
            return info

    def set_roi_info(self, x: int, y: int, w: int, h: int):
        '''
        Set the region of interest (ROI) information.

        Parameters
        ----------
        x : int
            X-coordinate of the ROI.
        y : int
            Y-coordinate of the ROI.
        w : int
            Width of the ROI.
        h : int
            Height of the ROI.

        Returns
        -------
        None
        '''
        self.set_param_value(CamParams.ROI_X, x)
        self.set_param_value(CamParams.ROI_Y, y)
        self.set_param_value(CamParams.ROI_WIDTH, w)
        self.set_param_value(CamParams.ROI_HEIGHT, h)

    def set_roi_limits(
            self,
            x: tuple[int, int], y: tuple[int, int],
            w: tuple[int, int], h: tuple[int, int]):
        '''
        Set limits for the Region of Interest (ROI) parameters.

        This function sets limits for the X-coordinate, Y-coordinate, width,
        and height parameters of the Region of Interest (ROI).

        Parameters
        ----------
        x : tuple[int, int]
            Tuple representing the minimum and maximum limits for the X-coordinate.
        y : tuple[int, int]
            Tuple representing the minimum and maximum limits for the Y-coordinate.
        w : tuple[int, int]
            Tuple representing the minimum and maximum limits for the width.
        h : tuple[int, int]
            Tuple representing the minimum and maximum limits for the height.

        Returns
        -------
        None
        '''
        self.get_param(CamParams.ROI_X).setLimits(x)
        self.get_param(CamParams.ROI_Y).setLimits(y)
        self.get_param(CamParams.ROI_WIDTH).setLimits(w)
        self.get_param(CamParams.ROI_HEIGHT).setLimits(h)

    @property
    def isTiff(self):
        '''
        Check if the image format is TIFF.

        Returns
        -------
        bool
            True if the image format is TIFF, False otherwise.
        '''
        return self.get_param_value(CamParams.IMAGE_FORMAT) in [
            str(CamParams.TIFF_FORMAT), str(CamParams.BIGG_TIFF_FORMAT)
        ]

    @property
    def isBiggTiff(self):
        '''
        Check if the image format is BigTIFF.

        Returns
        -------
        bool
            True if the image format is BigTIFF, False otherwise.
        '''
        return self.get_param_value(CamParams.IMAGE_FORMAT) in [
            str(CamParams.BIGG_TIFF_FORMAT)]

    @property
    def isSingleView(self):
        '''
        Check if the view option is set to single view.

        Returns
        -------
        bool
            True if the view option is set to single view, False otherwise.
        '''
        return self.get_param_value(CamParams.VIEW_OPTIONS) in [
            str(CamParams.SINGLE_VIEW)]

    @property
    def isOverlaidView(self):
        '''
        Check if the view option is set to overlaid view.

        Returns
        -------
        bool
            True if the view option is set to overlaid view, False otherwise.
        '''
        return self.get_param_value(CamParams.VIEW_OPTIONS) in [
            str(CamParams.DUAL_OVERLAID)]

    @property
    def isFullMetadata(self):
        '''
        Check if the metadata option is set to full.

        Returns
        -------
        bool
            True if the full metadata option is set, False otherwise.
        '''
        return self.get_param_value(CamParams.FULL_METADATA)

    @property
    def isDarkCalibration(self):
        '''
        Check if the dark calibration option is set.

        Returns
        -------
        bool
            True if the dark calibration option is set, False otherwise.
        '''
        return self.get_param_value(CamParams.DARK_CALIBRATION)

    @property
    def isSaveData(self):
        '''
        Check if the save data option is set.

        Returns
        -------
        bool
            True if the save data option is set, False otherwise.
        '''
        return self.get_param_value(CamParams.SAVE_DATA)

    @property
    def isPreview(self):
        '''
        Check if the preview option is set.

        Returns
        -------
        bool
            True if the preview option is set, False otherwise.
        '''
        return self.get_param_value(CamParams.PREVIEW)

    @property
    def isAutostretch(self):
        '''
        Check if the auto-stretch option is set.

        Returns
        -------
        bool
            True if the auto-stretch option is set, False otherwise.
        '''
        return self.get_param_value(CamParams.AUTO_STRETCH)

    @property
    def isLineProfiler(self):
        '''
        Check if the line profiler option is set.

        Returns
        -------
        bool
            True if the line profiler option is set, False otherwise.
        '''
        return self.get_param_value(CamParams.LINE_PROFILER)

    @property
    def isNumpyLUT(self):
        '''
        Check if the LUT option is set to numpy lut.

        Returns
        -------
        bool
            True if the LUT option is set to numpy lut, False otherwise.
        '''
        return self.get_param_value(CamParams.LUT) in [str(CamParams.LUT_NUMPY)]


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraOptions()
    window.show()
    sys.exit(app.exec_())
