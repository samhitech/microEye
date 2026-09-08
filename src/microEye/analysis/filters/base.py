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

    def get_kernel_padding(self, kernel: np.ndarray) -> tuple[int, int]:
        '''Return the padding size for the kernel.

        Parameters
        ----------
        kernel : np.ndarray
            The kernel to be used for filtering.

        Returns
        -------
        tuple[int, int]
            The padding size for the kernel (height, width).
        '''
        kh, kw = kernel.shape

        assert kh % 2 == 1 and kw % 2 == 1, (
            'kernel must be odd-sized to avoid coordinate shift'
        )

        ph, pw = kh // 2, kw // 2

        return ph, pw

    def pad_image(
        self, image: np.ndarray, kernel: np.ndarray, mode: str = 'reflect'
    ) -> np.ndarray:
        """Pad the image for filtering.

        Parameters
        ----------
        image : np.ndarray
            The input image to be padded.
        kernel : np.ndarray
            The kernel to be used for filtering.
        mode : str, optional
            The padding mode to be used. Default is 'reflect'.

        Returns
        -------
        np.ndarray
            The padded image.
        """
        ph, pw = self.get_kernel_padding(kernel)
        padded = np.pad(image, ((ph, ph), (pw, pw)), mode=mode)

        return padded


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
