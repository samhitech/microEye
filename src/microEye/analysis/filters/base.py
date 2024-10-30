from abc import ABC, abstractmethod

import numpy as np
from PyQt5 import QtWidgets


class AbstractFilter(ABC):
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
            'name': 'Abstract Base Filter'
        }
