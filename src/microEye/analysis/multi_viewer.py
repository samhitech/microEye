import json
import logging
import os
import re
import webbrowser
from enum import Enum
from typing import Optional, Union
from weakref import ref

import cv2

from microEye import __version__
from microEye.analysis.file_system import FileSystem, _compact_display_name
from microEye.analysis.fitting.results import FittingResults
from microEye.analysis.tools.batch import BatchLocalizations
from microEye.analysis.tools.dark_cal import DarkCalibration
from microEye.analysis.tools.photon_cal import PhotonTransfer
from microEye.analysis.tools.registration import RegistrationWidget
from microEye.analysis.viewer import LocalizationsView, PSFView, StackView
from microEye.qt import (
    QT_API,
    QAction,
    QApplication,
    QDateTime,
    QFileSystemModel,
    QIcon,
    QMainWindow,
    Qt,
    QtCore,
    QtWidgets,
    Signal,
    Slot,
)
from microEye.utils import StartGUI

logger = logging.getLogger(__name__)


class DockKeys(Enum):
    FILE_SYSTEM = 'File System'
    FILES_LIST = 'Files List'
    SMLM_ANALYSIS = 'SMLM Analysis'
    DATA_FILTERS = 'Data Filters'


class multi_viewer(QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        # Set window properties
        self.title = f'Multi Viewer Module v{__version__}'
        self.left = 0
        self.top = 0
        self._width = 1600
        self._height = 950

        # Threading
        self._threadpool = QtCore.QThreadPool.globalInstance()
        print(
            f'Multithreading with maximum {self._threadpool.maxThreadCount()} threads'
        )

        # Set the path
        if path is None:
            path = os.path.dirname(os.path.abspath(__package__))
        self.initialize(path)

        # Set up the status bar
        self.status()

        # Status Bar Timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.status)
        self.timer.start(200)

        # Set main window properties
        self.setStatusBar(self.statusBar())

    def initialize(self, path):
        # Set Title / Dimensions / Center Window
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self._width, self._height)

        # Define main window layout
        self.setupMainWindowLayout()

        # Initialize the file system model / tree
        self.setupFileSystemTab(path)

        # Tabify docks
        self.tabifyDocks()

        # Set tab positions
        self.setTabPositions()

        # Raise docks
        self.raiseDocks()

        # Create menu bar
        self.createMenuBar()

        self.show()
        self.center()

    def setupMainWindowLayout(self):
        # # Create the MDI area
        self.mdi_area = QtWidgets.QMdiArea()
        self.mdi_area.setViewMode(QtWidgets.QMdiArea.ViewMode.TabbedView)
        self.mdi_area.setTabsClosable(True)
        self.mdi_area.setTabsMovable(True)
        self.mdi_area.setBackground(Qt.GlobalColor.transparent)
        tabs = self.mdi_area.findChild(QtWidgets.QTabBar)
        tabs.setExpanding(False)

        self.setCentralWidget(self.mdi_area)

        self.docks: dict[
            str, QtWidgets.QDockWidget
        ] = {}  # Dictionary to store created docks
        self.layouts = {}

    def setupFileSystemTab(self, path):
        # Tiff File system tree viewer tab layout
        self.file_tree_layout = self.create_tab(
            DockKeys.FILE_SYSTEM,
            QtWidgets.QVBoxLayout,
            'LeftDockWidgetArea',
            widget=None,
        )

        self.path = path

        # Create QTreeView
        self.file_system = FileSystem(path=self.path)

        # Connect double-click signal to a method that uses QTimer
        self.file_system.openFile.connect(self._open_file)
        self.file_system.focusItem.connect(self._focus_opened_file)

        self.file_tree_layout.addWidget(self.file_system)

    def tabifyDocks(self):
        # Tabify docks
        if len(self.docks) > 1:
            first_dock = next(iter(self.docks.values()))
            for dock in list(self.docks.values())[1:]:
                self.tabifyDockWidget(first_dock, dock)

    def setTabPositions(self):
        self.setTabPosition(
            Qt.DockWidgetArea.LeftDockWidgetArea, QtWidgets.QTabWidget.TabPosition.North
        )
        self.setTabPosition(
            Qt.DockWidgetArea.RightDockWidgetArea,
            QtWidgets.QTabWidget.TabPosition.North,
        )

    def raiseDocks(self):
        # Raise docks
        self.docks[DockKeys.FILE_SYSTEM].raise_()

    def createMenuBar(self):
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
        tools_menu = menu_bar.addMenu('Tools')
        help_menu = menu_bar.addMenu('Help')

        # Create exit action
        save_config = QAction('Save Config.', self)
        save_config.triggered.connect(lambda: saveConfig(self))
        load_config = QAction('Load Config.', self)
        load_config.triggered.connect(lambda: loadConfig(self))

        github = QAction('microEye Github', self)
        github.triggered.connect(
            lambda: webbrowser.open('https://github.com/samhitech/microEye')
        )
        pypi = QAction('microEye PYPI', self)
        pypi.triggered.connect(
            lambda: webbrowser.open('https://pypi.org/project/microEye/')
        )

        # Add exit action to file menu
        file_menu.addAction(save_config)
        file_menu.addAction(load_config)

        def connect(action: QAction, dock: QtWidgets.QDockWidget):
            action.triggered.connect(lambda: dock.setVisible(action.isChecked()))

        # Create toggle view actions for each dock
        dock_toggle_actions = {}
        for key, dock in self.docks.items():
            toggle_action = dock.toggleViewAction()
            toggle_action.setEnabled(True)
            dock_toggle_actions[key] = toggle_action
            view_menu.addAction(toggle_action)
            if '6' in QT_API:
                connect(toggle_action, dock)

        # Add tools menu actions
        self.reg_window = None

        def show_registration_tool():
            if self.reg_window is not None:
                self.reg_window.close()
                self.reg_window = None

            stacks = {}
            for path, info in self.file_system._opened_files.items():
                widget = info['widget']
                if info['type'] in ('TIFF', 'ZARR', 'ImageSeq') and isinstance(
                    widget, StackView
                ):
                    channels = widget.stack_handler.shapeTCZYX()[1]
                    for c in range(channels):
                        key = _compact_display_name(path, info['type']) + f' (C{c})'
                        stacks[key.replace('\n', ' ')] = ref(widget.stack_handler)

            if len(stacks) < 2:
                logger.warning('Not enough stacks available for registration.')
                return

            self.reg_window = RegistrationWidget(stacks=stacks)
            self.reg_window.show()

        # Add tools menu actions
        registration_tool = QAction('Registration Tool', self)
        registration_tool.triggered.connect(show_registration_tool)
        tools_menu.addAction(registration_tool)

        # Add Localization Clustering Tool
        self.batch_window = None

        def show_batch_loc_tool():
            if self.batch_window is not None:
                self.batch_window.close()
                self.batch_window = None

            loc_tables = {}
            for path, info in self.file_system._opened_files.items():
                widget = info['widget']
                if info['type'] in ('HDF5', 'TSV') and isinstance(
                    widget, LocalizationsView
                ):
                    key = _compact_display_name(path, info['type'])
                    for table in range(len(widget.fittingResults)):
                        key = f'{key} (Table {table})'
                        loc_tables[key.replace('\n', ' ')] = ref(
                            widget.fittingResults[table]
                        )

            if len(loc_tables) < 1:
                logger.warning(
                    'No localization tables available for batch processing.'
                )
                return

            self.batch_window = BatchLocalizations(
                parent=self, loc_tables=loc_tables
            )
            self.batch_window.show()

        loc_batch_tool = QAction('Localization Batch Tool', self)
        loc_batch_tool.triggered.connect(show_batch_loc_tool)
        tools_menu.addAction(loc_batch_tool)

        # Add Dark Calibration Tool
        dark_cal_tool = QAction('Dark Calibration Tool', self)
        dark_cal_tool.triggered.connect(
            lambda: self.mdi_area.addSubWindow(DarkCalibration()).show()
        )
        tools_menu.addAction(dark_cal_tool)

        # Add PTC Calibration Tool
        ptc_cal_tool = QAction('PTC Calibration Tool', self)
        ptc_cal_tool.triggered.connect(
            lambda: self.mdi_area.addSubWindow(PhotonTransfer()).show()
        )
        tools_menu.addAction(ptc_cal_tool)

        help_menu.addAction(github)
        help_menu.addAction(pypi)

    def create_tab(
        self,
        key: DockKeys,
        layout_type: Optional[
            type[Union[QtWidgets.QVBoxLayout, QtWidgets.QFormLayout]]
        ] = None,
        dock_area: str = 'LeftDockWidgetArea',
        widget: Optional[QtWidgets.QWidget] = None,
        visible: bool = True,
    ) -> Optional[type[Union[QtWidgets.QVBoxLayout, QtWidgets.QFormLayout]]]:
        '''
        Create a tab with a dock widget.

        Parameters
        ----------
        key : DockKeys
            The unique identifier for the tab.
        layout_type : Optional[Type[Union[QVBoxLayout, QFormLayout]]], optional
            The layout type for the group in the dock. If provided,
            widget should be None.
        dock_area : str, optional
            The dock area where the tab will be added.
        widget : Optional[QWidget], optional
            The widget to be placed in the dock. If provided,
            layout_type should be None.
        visible : bool, optional
            Whether the dock widget should be visible.

        Returns
        -------
        Optional[Type[Union[QVBoxLayout, QFormLayout]]]
            The layout if layout_type is provided, otherwise None.
        '''
        if widget:
            group = widget
        else:
            group = QtWidgets.QWidget()
            group_layout = layout_type() if layout_type else QtWidgets.QVBoxLayout()
            group.setLayout(group_layout)
            self.layouts[key] = group_layout

        dock = QtWidgets.QDockWidget(str(key.value), self)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        dock.setWidget(group)
        dock.setVisible(visible)
        self.addDockWidget(getattr(Qt.DockWidgetArea, dock_area), dock)

        # Store the dock in the dictionary
        self.docks[key] = dock

        # Return the layout if layout_type is provided, otherwise None
        return None if widget else group_layout

    def center(self):
        '''Centers the window within the screen using setGeometry.'''
        # Get the screen geometry
        screen_geometry = QApplication.primaryScreen().availableGeometry()

        # Calculate the center point
        center_point = screen_geometry.center()

        # Set the window geometry
        self.setGeometry(
            center_point.x() - self.width() // 2,
            center_point.y() - self.height() // 2,
            self.width(),
            self.height(),
        )

    def status(self):
        # Statusbar time
        self.statusBar().showMessage(
            f'{QT_API} | '
            + 'Time: '
            + QDateTime.currentDateTime().toString('hh:mm:ss,zzz')
        )

    def _focus_opened_file(self, reference: ref):
        widget = reference()

        if not widget:
            return

        for subwin in self.mdi_area.subWindowList():
            if subwin.widget() is widget:
                # subwin.raise_()
                subwin.showMaximized()
                break

    @Slot(QtCore.QModelIndex, str)
    def _open_file(self, index: QtCore.QModelIndex, file_type: str):
        # Set the Qt.WindowFlags for making the subwindow resizable
        view = None

        path = self.file_system.filePath(index)

        if path in self.file_system._opened_files:
            self.file_system._focus_opened_file(path)
            return

        if file_type in ['TIFF', 'ZARR', 'ImageSeq']:
            pattern = self.file_system.file_pattern if file_type == 'ImageSeq' else None

            try:
                view = StackView(path, pattern)
                view.localizedData.connect(self.file_system.localizedData)
            except Exception as e:
                view = None
                logger.error(f'Error opening: {e}')

        elif file_type in ['HDF5', 'TSV']:
            view = LocalizationsView(path=path)
        elif file_type == 'PSF':
            view = PSFView(path)

        if view:
            self.file_system._opened_files[path] = {
                'type': file_type,
                'widget': view,
            }
            window = self.mdi_area.addSubWindow(view)
            # Install event filter
            window.destroyed.connect(lambda: self.file_system._on_subwindow_close(path))
            window.show()

        self.file_system._update_opened_files_list()

    def StartGUI(path=None):
        '''
        Initializes a new QApplication and multi_viewer.

        Parameters
        ----------
        path : str, optional
            The path to a file to be loaded initially.

        Returns
        -------
        tuple of QApplication and multi_viewer
            Returns a tuple with QApplication and multi_viewer main window.
        '''
        return StartGUI(multi_viewer, path)


def get_dock_config(dock: QtWidgets.QDockWidget):
    '''
    Get the configuration dictionary for a QDockWidget.

    Parameters
    ----------
    dock : QDockWidget
        The QDockWidget to get the configuration for.

    Returns
    -------
    dict
        The configuration dictionary containing isFloating,
        position, size, and isVisible.
    '''
    if dock:
        return {
            'isFloating': dock.isFloating(),
            'position': (
                dock.mapToGlobal(QtCore.QPoint(0, 0)).x(),
                dock.mapToGlobal(QtCore.QPoint(0, 0)).y(),
            ),
            'size': (dock.geometry().width(), dock.geometry().height()),
            'isVisible': dock.isVisible(),
        }


def get_widget_config(widget: QtWidgets.QWidget):
    '''
    Get the configuration dictionary for a QWidget.

    Parameters
    ----------
    widget : QWidget
        The QWidget to get the configuration for.

    Returns
    -------
    dict
        The configuration dictionary containing position, size, and isMaximized.
    '''
    if widget:
        return {
            'position': (
                widget.mapToGlobal(QtCore.QPoint(0, 0)).x(),
                widget.mapToGlobal(QtCore.QPoint(0, 0)).y(),
            ),
            'size': (widget.geometry().width(), widget.geometry().height()),
            'isMaximized': widget.isMaximized(),
        }


def saveConfig(window: multi_viewer, filename: str = 'config_tiff.json'):
    """
    Save the configuration for the multi_viewer application.

    Parameters
    ----------
    window : multi_viewer
        The main application window.
    filename : str, optional
        The filename of the configuration file, by default 'config_tiff.json'.
    """
    config = dict()

    # Save multi_viewer widget config
    config['multi_viewer'] = get_widget_config(window)

    # Save docks config
    for key in DockKeys:
        dock = window.docks.get(key)
        if dock:
            config[key.value] = get_dock_config(dock)

    with open(filename, 'w') as file:
        json.dump(config, file, indent=2)

    logger.info(f'{filename} file generated!')


def load_widget_config(widget: QtWidgets.QWidget, widget_config):
    '''
    Load configuration for a QWidget.

    Parameters
    ----------
    widget : QWidget
        The QWidget to apply the configuration to.
    widget_config : dict
        The configuration dictionary containing position, size, and maximized status.

    Returns
    -------
    None
    '''
    widget.setGeometry(
        widget_config['position'][0],
        widget_config['position'][1],
        widget_config['size'][0],
        widget_config['size'][1],
    )
    if bool(widget_config['isMaximized']):
        widget.showMaximized()


def loadConfig(window: multi_viewer, filename: str = 'config_tiff.json'):
    """
    Load the configuration for the multi_viewer application.

    Parameters
    ----------
    window : multi_viewer
        The main application window.
    filename : str, optional
        The filename of the configuration file, by default 'config_tiff.json'.
    """
    if not os.path.exists(filename):
        logger.warning(f'{filename} not found!')
        return

    config: dict = None

    with open(filename) as file:
        config = json.load(file)

    # Loading multi_viewer widget config
    if 'multi_viewer' in config:
        load_widget_config(window, config['multi_viewer'])

    # Loading docks
    for dock_key, dock_config in config.items():
        dock_enum_key = None
        try:
            dock_enum_key = DockKeys(dock_key)
        except ValueError:
            # Skip processing if dock_key is not a valid DockKeys enum
            continue

        if dock_enum_key in window.docks:
            dock = window.docks[dock_enum_key]
            dock.setVisible(bool(dock_config.get('isVisible', False)))
            if bool(dock_config.get('isFloating', False)):
                dock.setFloating(True)
                dock.setGeometry(
                    dock_config.get('position', (0, 0))[0],
                    dock_config.get('position', (0, 0))[1],
                    dock_config.get('size', (0, 0))[0],
                    dock_config.get('size', (0, 0))[1],
                )
            else:
                dock.setFloating(False)

    logger.info(f'{filename} file loaded!')


if __name__ == '__main__':
    app, window = multi_viewer.StartGUI('')
    app.exec()
