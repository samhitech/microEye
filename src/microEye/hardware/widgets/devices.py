from enum import Enum
from typing import Optional

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtSerialPort import *
from PyQt5.QtWidgets import *
from pyqtgraph.parametertree import Parameter

from ...shared import Tree


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
    '''Devices View used in miEye module to add lasers, stages, and IR detector.
    '''
    setDetectorActivated = pyqtSignal(str)
    resetDetectorActivated = pyqtSignal()
    addLaserActivated = pyqtSignal(str)
    setStageActivated = pyqtSignal(str)

    def __init__(self, parent: Optional['QWidget'] = None):
        super().__init__(parent=parent)

    def getDetectors(self):
        return ['Parallax CCD (TSL1401)']

    def getLasers(self):
        return ['IO MatchBox Single Laser', 'IO MatchBox Laser Combiner']

    def getStages(self):
        return ['PiezoConcept FOC100']

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {'name': str(devicesParams.IR_CAM), 'type': 'group',
                'children': [
                    {'name': str(devicesParams.IR_DEVICE), 'title': '', 'type': 'list',
                        'values': self.getDetectors()},
                    {'name': str(devicesParams.IR_SET), 'type': 'action'},
                    {'name': str(devicesParams.IR_RESET), 'type': 'action'},
                ]},
            {'name': str(devicesParams.LASERS), 'type': 'group',
                'children': [
                    {'name': str(devicesParams.LASER), 'title': '', 'type': 'list',
                        'values': self.getLasers()},
                    {'name': str(devicesParams.ADD_LASER), 'type': 'action'},
                ]},
            {'name': str(devicesParams.STAGES), 'type': 'group',
                'children': [
                    {'name': str(devicesParams.STAGE), 'title': '', 'type': 'list',
                        'values': self.getStages()},
                    {'name': str(devicesParams.SET_STAGE), 'type': 'action'},
                ]},
        ]

        self.param_tree = Parameter.create(name='root', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents)

        self.get_param(
            devicesParams.IR_SET).sigActivated.connect(
                lambda: self.setDetectorActivated.emit(
                    self.get_param_value(devicesParams.IR_DEVICE)))
        self.get_param(
            devicesParams.IR_RESET).sigActivated.connect(
                self.resetDetectorActivated.emit)

        self.get_param(
            devicesParams.ADD_LASER).sigActivated.connect(
                lambda: self.addLaserActivated.emit(
                    self.get_param_value(devicesParams.LASER)))
        self.get_param(
            devicesParams.SET_STAGE).sigActivated.connect(
                lambda: self.setStageActivated.emit(
                    self.get_param_value(devicesParams.STAGE)))
