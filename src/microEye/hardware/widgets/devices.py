from enum import Enum
from typing import Optional

from pyqtgraph.parametertree import Parameter

from microEye.qt import QtWidgets, Signal
from microEye.utils import Tree


class devicesParams(Enum):
    '''
    Enum class defining Devices parameters.
    '''

    IR_CAM = 'IR Detector'
    IR_DEVICE = 'IR Detector.Device'
    IR_SET = 'IR Detector.Set'
    IR_RESET = 'IR Detector.Reset'

    LASERS = 'Lasers'
    LASER = 'Lasers.Laser'
    ADD_LASER = 'Lasers.Add Laser'

    STAGES = 'Stages'
    STAGE = 'Stages.Stage'
    SET_STAGE = 'Stages.Set Stage'

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


class DevicesView(Tree):
    '''Devices View used in miEye module to add lasers, stages, and IR detector.'''

    PARAMS = devicesParams
    setDetectorActivated = Signal(str)
    resetDetectorActivated = Signal()
    addLaserActivated = Signal(str)
    setStageActivated = Signal(str)

    def __init__(self, parent: Optional['QtWidgets.QWidget'] = None):
        super().__init__(parent=parent)

    def getDetectors(self):
        return ['Demo Line Scanner', 'Parallax CCD (TSL1401)']

    def getLasers(self):
        return ['IO MatchBox Single Laser', 'IO MatchBox Laser Combiner']

    def getStages(self):
        return ['PiezoConcept FOC100']

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {
                'name': str(devicesParams.IR_CAM),
                'type': 'group',
                'children': [
                    {
                        'name': str(devicesParams.IR_DEVICE),
                        'title': '',
                        'type': 'list',
                        'limits': self.getDetectors(),
                    },
                    {'name': str(devicesParams.IR_SET), 'type': 'action'},
                    {'name': str(devicesParams.IR_RESET), 'type': 'action'},
                ],
            },
            {
                'name': str(devicesParams.LASERS),
                'type': 'group',
                'children': [
                    {
                        'name': str(devicesParams.LASER),
                        'title': '',
                        'type': 'list',
                        'limits': self.getLasers(),
                    },
                    {'name': str(devicesParams.ADD_LASER), 'type': 'action'},
                ],
            },
            {
                'name': str(devicesParams.STAGES),
                'type': 'group',
                'children': [
                    {
                        'name': str(devicesParams.STAGE),
                        'title': '',
                        'type': 'list',
                        'limits': self.getStages(),
                    },
                    {'name': str(devicesParams.SET_STAGE), 'type': 'action'},
                ],
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )

        self.get_param(devicesParams.IR_SET).sigActivated.connect(
            lambda: self.setDetectorActivated.emit(
                self.get_param_value(devicesParams.IR_DEVICE)
            )
        )
        self.get_param(devicesParams.IR_RESET).sigActivated.connect(
            lambda: self.resetDetectorActivated.emit()
        )

        self.get_param(devicesParams.ADD_LASER).sigActivated.connect(
            lambda: self.addLaserActivated.emit(
                self.get_param_value(devicesParams.LASER)
            )
        )
        self.get_param(devicesParams.SET_STAGE).sigActivated.connect(
            lambda: self.setStageActivated.emit(
                self.get_param_value(devicesParams.STAGE)
            )
        )
