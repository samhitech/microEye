import importlib
import logging
import pkgutil
import uuid
import weakref
from enum import Enum
from typing import Union

from pyqtgraph.parametertree import Parameter

import microEye.hardware.stages
from microEye.hardware.stages.stabilizer import FocusStabilizer
from microEye.hardware.stages.stage import AbstractStage, Axis, StageParams, Units
from microEye.qt import QtCore, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree

for _, modname, ispkg in pkgutil.walk_packages(
    microEye.hardware.stages.__path__, microEye.hardware.stages.__name__ + '.'
):
    if ispkg:  # Only import if it's a package (directory with __init__.py)
        importlib.import_module(modname)


class StageManager(QtCore.QObject):
    __initialized = False
    __singleton = None

    stageAdded = Signal(str)
    stageRemoved = Signal(str)

    STAGES: dict[str, AbstractStage] = {}

    def __new__(cls, *args, **kwargs):
        # If the single instance doesn't exist, create a new one
        if cls.__singleton is None:
            cls.__singleton = super().__new__(cls, *args)
        # Return the single instance
        return cls.__singleton

    @classmethod
    def instance(cls):
        if cls.__singleton is None:
            return StageManager()

        return cls.__singleton

    def __init__(self):
        if not StageManager.__initialized:
            super().__init__()
            self._metadata = {
                Axis.X: {
                    'step': 50.0,
                    'jump': 500.0,
                },
                Axis.Y: {
                    'step': 50.0,
                    'jump': 500.0,
                },
                Axis.Z: {
                    'step': 0.1,
                    'jump': 1.0,
                },
            }

            StageManager.__initialized = True

    def stage_drivers(self):
        # get all subclasses of AbstractStage
        return AbstractStage.__subclasses__()

    def z_stage(self) -> AbstractStage:
        return self._metadata[Axis.Z].get('stage', lambda: None)()

    def xy_stage(self) -> AbstractStage:
        stage = self._metadata[Axis.X].get('stage', lambda: None)()
        if stage is None:
            stage = self._metadata[Axis.Y].get('stage', lambda: None)()

        return stage

    def move_absolute(self, x, y, z, **kwargs):
        if z >= 0 and self.z_stage() is not None:
            args = self.z_stage().convert_units(0, 0, z, kwargs.get('unit'))
            self.z_stage().move_absolute(*args, **kwargs)
        if (x >= 0 or y >= 0) and self.xy_stage() is not None:
            args = self.xy_stage().convert_units(x, y, 0, kwargs.get('unit'))
            self.xy_stage().move_absolute(*args, **kwargs)

    def move_relative(self, x, y, z, **kwargs):
        if z != 0 and self.z_stage() is not None:
            args = self.z_stage().convert_units(0, 0, z, kwargs.get('unit'))
            self.z_stage().move_relative(*args, **kwargs)
        if (x != 0 or y != 0) and self.xy_stage() is not None:
            args = self.xy_stage().convert_units(x, y, 0, kwargs.get('unit'))
            self.xy_stage().move_relative(*args, **kwargs)

    def home(self, axis: Axis):
        if axis == Axis.Z and self.z_stage() is not None:
            self.z_stage().home()
        elif axis in (Axis.X, Axis.Y) and self.xy_stage() is not None:
            self.xy_stage().home()

    def stop(self, axis: Axis):
        if axis == Axis.Z and self.z_stage() is not None:
            self.z_stage().stop()
        elif axis in (Axis.X, Axis.Y) and self.xy_stage() is not None:
            self.xy_stage().stop()

    def is_busy(self, axis: Axis) -> bool:
        if axis == Axis.Z and self.z_stage() is not None:
            return self.z_stage().busy
        elif axis in (Axis.X, Axis.Y) and self.xy_stage() is not None:
            return self.xy_stage().busy
        return False

    def is_open(self, axis: Axis) -> bool:
        if axis == Axis.Z and self.z_stage() is not None:
            return self.z_stage().is_open()
        elif axis in (Axis.X, Axis.Y) and self.xy_stage() is not None:
            return self.xy_stage().is_open()
        return False

    def move_z(self, dir: bool, step_arg: Union[int, bool], interface: bool = False):
        '''
        Move the stage in a specified direction by a specified
        number of steps in nanometers. Optional boolean argument
        to specify jump or step move. If FocusStabilizer is stabilizing
        and use calibration is set to True, moves the center pixel value
        instead of the Z position when `interface` is True.

        Parameters
        ----------
        dir: bool
            Direction of the movement. If True, moves the stage up, else moves it down.
        step_arg: Union[int, bool]
            Step size [nm] to move in the specified direction.
            If provided as a `bool`:
            - `True`: Moves the stage by the value of the JUMP parameter.
            - `False`: Moves the stage by the value of the STEP parameter.
        interface : bool, optional
            If True, moves the center pixel value instead of the Z position
            when FocusStabilizer is stabilizing and use calibration is set to True.
        '''
        if self.z_stage() is None:
            return

        if isinstance(step_arg, bool):
            step_arg = self._metadata[Axis.Z]['jump' if step_arg else 'step']

            step_arg = Units.convert(
                step_arg, Units.MICROMETERS, self.z_stage().get_unit(Axis.Z)
            )
        elif isinstance(step_arg, (int, float)):
            step_arg = Units.convert(
                step_arg, Units.NANOMETERS, self.z_stage().get_unit(Axis.Z)
            )

        focusStabilizer = FocusStabilizer.instance()
        if (
            focusStabilizer is not None
            and focusStabilizer.isFocusStabilized()
            and focusStabilizer.useCal()
            and interface
        ):
            sign = 1 if dir else -1
            focusStabilizer.setParameter(
                focusStabilizer.calCoeff() * step_arg * sign, True
            )
        else:
            self.z_stage().move_higher(step_arg) if dir else self.z_stage().move_lower(
                step_arg
            )

    def move_xy(self, X: bool, jump: bool, direction: bool):
        if self.xy_stage() is None:
            return

        axis = Axis.X if X else Axis.Y

        if isinstance(jump, bool):
            step = self._metadata[axis]['jump' if jump else 'step']
        elif isinstance(jump, (int, float)):
            step = jump
        else:
            step = self._metadata[axis]['step']

        step = Units.convert(step, Units.MICROMETERS, self.xy_stage().get_unit(axis))
        step *= -1 if not direction else 1
        self.xy_stage().move_relative(step if X else 0, step if not X else 0)

    def open_all(self):
        if self.z_stage() is not None and not self.z_stage().is_open():
            self.z_stage().open()
        if self.xy_stage() is not None and not self.xy_stage().is_open():
            self.xy_stage().open()

    def close_all(self):
        if self.z_stage() is not None and self.z_stage().is_open():
            self.z_stage().close()
        if self.xy_stage() is not None and self.xy_stage().is_open():
            self.xy_stage().close()

    def _add_stage(self, class_name: str):
        for stage_class in self.stage_drivers():
            if class_name == stage_class.NAME:
                stage = stage_class.get_stage()
                if stage is not None:
                    key = f'{stage_class.NAME}-{uuid.uuid4().hex[:8]}'
                    StageManager.STAGES[key] = stage
                    self.stageAdded.emit(key)
                    return key
                else:
                    logging.warning(f'Stage class {class_name} returned None.')
                    return None

        logging.warning(f'Stage class {class_name} not found.')
        return None

    def _remove_stage(self, key: str) -> bool:
        stage = StageManager.STAGES.get(key)
        if stage:
            stage.close()
            stage.signals.stageRemoved.emit()
            del StageManager.STAGES[key]
            self.stageRemoved.emit(key)
            return True
        return False

    def _remove_all_stages(self):
        keys = list(StageManager.STAGES.keys())
        for key in keys:
            self._remove_stage(key)

    def _set_axis_stage(self, axis: Axis, key: str):
        stage = StageManager.STAGES.get(key)

        if stage is None or axis not in stage.axes:
            logging.warning('Stage must support the specified axis.')
            return

        self._metadata[axis]['stage'] = weakref.ref(stage)

    def _set_step(self, axis: Axis, step: float):
        if axis in self._metadata:
            self._metadata[axis]['step'] = step

    def _set_jump(self, axis: Axis, jump: float):
        if axis in self._metadata:
            self._metadata[axis]['jump'] = jump

    def _get_stage_axis(self, key):
        stage = StageManager.STAGES.get(key)
        if stage is not None:
            return [
                axis.value
                for axis, metadata in self._metadata.items()
                if stage == metadata.get('stage', lambda: None)()
            ]
        return []

    def get_config(self):
        config = {}
        for key, stage in self.STAGES.items():
            config[key] = {
                'class_name': stage.NAME,
                'config': stage.get_config(),
                'axis': self._get_stage_axis(key),
            }
        return config

    def load_config(self, config=dict):
        self._remove_all_stages()
        for _, entry in config.items():
            class_name = entry['class_name']
            stage_config = entry['config']
            stage_axis = entry.get('axis', [])
            stage_key = self._add_stage(class_name)
            if stage_key and stage_key in self.STAGES:
                self.STAGES[stage_key].load_config(stage_config)
                for axis in stage_axis:
                    self._set_axis_stage(Axis(axis), stage_key)


class StageManagerParams(Enum):
    '''
    Enum class defining Stage Manager parameters.
    '''

    DRIVERS = 'Stage Drivers'
    ADD_STAGE = 'Add Stage'
    REMOVE_STAGE = 'Remove Stage'

    X_STEP = 'X Step Size [um]'
    Y_STEP = 'Y Step Size [um]'
    Z_STEP = 'Z Step Size [um]'

    X_JUMP = 'X Jump Size [um]'
    Y_JUMP = 'Y Jump Size [um]'
    Z_JUMP = 'Z Jump Size [um]'

    STAGES = 'Stages'

    SET_X_STAGE = 'Set X Stage'
    SET_Y_STAGE = 'Set Y Stage'
    SET_Z_STAGE = 'Set Z Stage'

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


class StageManagerView(Tree):
    def __init__(self, parent=None):
        super().__init__(parent)

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        This method creates and sets up the parameter tree structure for the
        `ZStageView` class.
        '''
        params = [
            {
                'name': str(StageManagerParams.DRIVERS),
                'type': 'list',
                'limits': [d.NAME for d in StageManager.instance().stage_drivers()],
                'value': None,
            },
            {'name': str(StageManagerParams.ADD_STAGE), 'type': 'action'},
            {'name': str(StageManagerParams.X_STEP), 'type': 'float', 'value': 50.0},
            {'name': str(StageManagerParams.X_JUMP), 'type': 'float', 'value': 500.0},
            {'name': str(StageManagerParams.Y_STEP), 'type': 'float', 'value': 50.0},
            {'name': str(StageManagerParams.Y_JUMP), 'type': 'float', 'value': 500.0},
            {'name': str(StageManagerParams.Z_STEP), 'type': 'float', 'value': 0.1},
            {'name': str(StageManagerParams.Z_JUMP), 'type': 'float', 'value': 1.0},
            {
                'name': str(StageManagerParams.STAGES),
                'type': 'list',
                'limits': list(StageManager.STAGES.keys()),
                'value': None,
            },
            {'name': str(StageManagerParams.REMOVE_STAGE), 'type': 'action'},
            {
                'name': str(StageManagerParams.SET_X_STAGE),
                'type': 'action',
            },
            {
                'name': str(StageManagerParams.SET_Y_STAGE),
                'type': 'action',
            },
            {
                'name': str(StageManagerParams.SET_Z_STAGE),
                'type': 'action',
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self._initialize_signals()

    def _initialize_signals(self):
        self.get_param(StageManagerParams.ADD_STAGE).sigActivated.connect(
            lambda: StageManager.instance()._add_stage(
                self.get_param(StageManagerParams.DRIVERS).value()
            )
        )

        self.get_param(StageManagerParams.REMOVE_STAGE).sigActivated.connect(
            lambda: StageManager.instance()._remove_stage(
                self.get_param(StageManagerParams.STAGES).value()
            )
        )

        self.get_param(StageManagerParams.SET_X_STAGE).sigActivated.connect(
            lambda: StageManager.instance()._set_axis_stage(
                Axis.X, self.get_param(StageManagerParams.STAGES).value()
            )
        )

        self.get_param(StageManagerParams.SET_Y_STAGE).sigActivated.connect(
            lambda: StageManager.instance()._set_axis_stage(
                Axis.Y, self.get_param(StageManagerParams.STAGES).value()
            )
        )

        self.get_param(StageManagerParams.SET_Z_STAGE).sigActivated.connect(
            lambda: StageManager.instance()._set_axis_stage(
                Axis.Z, self.get_param(StageManagerParams.STAGES).value()
            )
        )

        StageManager.instance().stageAdded.connect(self._update_stages)
        StageManager.instance().stageRemoved.connect(self._update_stages)

    def _add_stage(self):
        StageManager.instance()._add_stage(
            self.get_param(StageManagerParams.DRIVERS).value()
        )

        self.get_param(StageManagerParams.STAGES).setLimits(
            list(StageManager.STAGES.keys())
        )

    def _update_stages(self, key: str):
        stages_param = self.get_param(StageManagerParams.STAGES)
        stages_param.setLimits(list(StageManager.STAGES.keys()))
        if key in StageManager.STAGES:
            stages_param.setValue(key)

    def change(self, param, changes):
        for p, _, data in changes:
            path = self.param_tree.childPath(p)

            _param = StageManagerParams('.'.join(path))

            mapping = {
                StageManagerParams.X_STEP: (Axis.X, StageManager._set_step),
                StageManagerParams.Y_STEP: (Axis.Y, StageManager._set_step),
                StageManagerParams.Z_STEP: (Axis.Z, StageManager._set_step),
                StageManagerParams.X_JUMP: (Axis.X, StageManager._set_jump),
                StageManagerParams.Y_JUMP: (Axis.Y, StageManager._set_jump),
                StageManagerParams.Z_JUMP: (Axis.Z, StageManager._set_jump),
            }
            if _param in mapping:
                axis, func = mapping[_param]
                func(StageManager.instance(), axis, data)

            self.paramsChanged.emit(p, data)


# print([d.NAME for d in StageManager.instance().stage_drivers()])
