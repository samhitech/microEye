from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from microEye.qt import QtWidgets


class AbstractFilter(ABC):
    def __init__(self, name: str = 'Abstract Base Filter', parameters: dict = None):
        self._name = name
        self._parameters = parameters or {}
        self._show_filter = False

    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        '''Apply the filter to the input image.'''
        pass

    def getWidget(self):
        '''Return the QWidget for configuring the filter.'''
        return QtWidgets.QWidget()

    def get_metadata(self) -> dict:
        '''Return metadata about the filter.'''
        return {
            'name': self.name,
            'parameters': self._parameters,
        }

    @classmethod
    def get_tree_parameters(cls) -> dict:
        '''Return the parameters for the pyqtgraph tree view.'''
        return None

    @property
    def name(self):
        return self._name

    @property
    def parameters(self) -> Optional[dict]:
        return self._parameters

class SpatialFilter(AbstractFilter):
    '''Base class for spatial filters.'''
    def __init__(self, name: str = 'Spatial Filter', parameters: dict = None):
        super().__init__(name=name, parameters=parameters)

    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        '''Apply the spatial filter to the input image.'''
        pass


class TemporalFilter(AbstractFilter):
    '''Base class for temporal filters.'''
    def __init__(self, name: str = 'Temporal Filter', parameters: dict = None):
        super().__init__(name=name, parameters=parameters)

    @abstractmethod
    def run(self, image: np.ndarray) -> np.ndarray:
        '''Apply the temporal filter to the input image.'''
        pass
