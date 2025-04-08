import os
import sys
from enum import Enum
from typing import Any, Optional

from mmpycorex.launcher import find_existing_mm_install
from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import GroupParameter

from microEye.qt import QApplication, QtWidgets, Signal
from microEye.utils.parameter_tree import Tree


class headlessParams(Enum):
    '''
    Enum class defining headless MM parameters.
    '''

    MM_APP_PATH = 'Micro-Manager Path'
    CONFIG_FILE = 'Config File'
    JAVA_LOC = 'Java Location'
    '''
    On windows plaforms, the Java Runtime Environment will be grabbed automatically
    as it is installed along with the Micro-Manager application.

    On non-windows platforms, it may need to be installed/specified manually in order
    to ensure compatibility.
    Installing Java 11 is the most likely version to work without issue
    '''
    PYTHON_BACKEND = 'Python Backend'
    CORE_LOG_PATH = 'Core Log Path'
    BUFFER_SIZE_MB = 'Buffer Size MB'
    MAX_MEMORY_MB = 'Max Memory MB'
    PORT = 'Port'
    DEBUG = 'Debug'

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

    def label(self):
        '''
        Return the label of the parameter.
        '''
        return self.value.split('.')[-1]


def get_mm_path() -> str:
    '''
    Get the path to the Micro-Manager application.

    Returns
    -------
    str
        Path to the Micro-Manager application.
    '''
    return find_existing_mm_install()


class HeadlessOptions(Tree):
    def __init__(self, parent: Optional['QtWidgets.QWidget'] = None):
        '''
        Initialize the CameraOptions.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None.
        '''
        super().__init__(parent=parent)

        self.setMinimumWidth(700)

    def create_parameters(self):
        '''
        Create the parameter tree structure.
        '''
        params = [
            {
                'name': headlessParams.PORT.label(),
                'type': 'int',
                'value': 4827,
                'limits': [0, 9999],
                'tip': 'Default port to use for ZMQServer.',
            },
            {
                'name': headlessParams.DEBUG.label(),
                'type': 'bool',
                'value': False,
                'tip': 'Print debug messages.',
            },
            {
                'name': headlessParams.MM_APP_PATH.label(),
                'type': 'file',
                'directory': get_mm_path(),
                'fileMode': 'Directory',
                'value': get_mm_path(),
                'tip': 'Path to top level folder of Micro-Manager installation.',
            },
            {
                'name': headlessParams.CONFIG_FILE.label(),
                'type': 'list',
                'limits': [None],
                'value': None,
                'tip': 'Micro-manager config file with which core will be initialized.',
            },
            {
                'name': headlessParams.CORE_LOG_PATH.label(),
                'type': 'file',
                'directory': get_mm_path(),
                'fileMode': 'Directory',
                'value': '',
                'tip': 'Path to where core log files should be created.',
            },
            {
                'name': headlessParams.JAVA_LOC.label(),
                'type': 'file',
                'directory': get_mm_path(),
                'fileMode': 'Directory',
                'value': None,
                'tip': 'Path to the java version that it should be run with.',
            },
            {
                'name': headlessParams.PYTHON_BACKEND.label(),
                'type': 'bool',
                'readonly': True,
                'enabled': False,
                'value': False,
                'visible': False,
                'tip': 'Whether to use the python backend or the Java backend.',
            },
            {
                'name': headlessParams.BUFFER_SIZE_MB.label(),
                'type': 'list',
                'value': 1024,
                'limits': [
                    512,
                    768,
                    1024,
                    1280,
                    1536,
                    1792,
                    2048,
                    2304,
                    2560,
                    2816,
                    3072,
                    3328,
                    3584,
                    3840,
                    4096,
                ],
                'tip': 'Size of circular buffer in MB in MMCore.',
            },
            {
                'name': headlessParams.MAX_MEMORY_MB.label(),
                'type': 'list',
                'value': 2048,
                'limits': [
                    2048,
                    2304,
                    2560,
                    2816,
                    3072,
                    3328,
                    3584,
                    3840,
                    4096,
                    5120,
                    6144,
                    7168,
                    8192,
                    9216,
                ],
                'tip': 'Maximum amount of memory to be allocated to JVM.',
            },
        ]

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        # self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        self.get_param(headlessParams.MM_APP_PATH).sigValueChanged.connect(
            self.update_config_file
        )
        self.update_config_file(None, None)

    def update_config_file(self, param: Parameter, value: Any):
        '''
        Update the config file list based on the selected MM path.

        Parameters
        ----------
        param : Parameter
            The parameter that triggered the change.
        value : Any
            The new value of the parameter.
        '''
        mm_path = self.get_param(headlessParams.MM_APP_PATH).value()
        config_file_param = self.get_param(headlessParams.CONFIG_FILE)
        if os.path.exists(mm_path):
            config_files = [
                os.path.basename(f)
                for f in os.listdir(mm_path)
                if f.endswith('.cfg') and f.startswith('MMConfig')
            ]
            config_file_param.setLimits([None, *config_files])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HeadlessOptions()
    window.show()
    sys.exit(app.exec())
