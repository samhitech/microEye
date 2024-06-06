import json
import os
import webbrowser
from enum import Enum
from typing import Optional, Union

import cv2

from microEye.analysis.fitting.results import FittingResults
from microEye.analysis.viewer.images import StackView
from microEye.analysis.viewer.localizations import LocalizationsView
from microEye.qt import (
    QAction,
    QApplication,
    QDateTime,
    QFileSystemModel,
    QMainWindow,
    Qt,
    QtCore,
    QtWidgets,
)
from microEye.utils import StartGUI


class DockKeys(Enum):
    FILE_SYSTEM = 'File System'
    SMLM_ANALYSIS = 'SMLM Analysis'
    DATA_FILTERS = 'Data Filters'


class multi_viewer(QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        # Set window properties
        self.title = 'Multi Viewer Module'
        self.left = 0
        self.top = 0
        self._width = 1600
        self._height = 950
        self._zoom = 1
        self._n_levels = 4096

        # Initialize variables
        self.fittingResults = None

        # Threading
        self._threadpool = QtCore.QThreadPool.globalInstance()
        print(
            'Multithreading with maximum %d threads' % self._threadpool.maxThreadCount()
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
        self.timer.start(10)

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
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        # # Create the MDI area
        self.mdi_area = QtWidgets.QMdiArea(self.main_widget)
        self.mdi_area.setViewMode(QtWidgets.QMdiArea.ViewMode.TabbedView)
        self.mdi_area.setTabsClosable(True)
        self.mdi_area.setTabsMovable(True)
        self.mdi_area.setBackground(Qt.GlobalColor.transparent)
        tabs = self.mdi_area.findChild(QtWidgets.QTabBar)
        tabs.setExpanding(False)

        # # Add the two sub-main layouts
        self.main_layout.addWidget(self.mdi_area, 1)

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
        self.model = QFileSystemModel()
        self.model.setRootPath(self.path)
        self.model.setFilter(
            QtCore.QDir.Filter.AllDirs
            | QtCore.QDir.Filter.Files
            | QtCore.QDir.Filter.NoDotAndDotDot
        )
        self.model.setNameFilters(['*.tif', '*.tiff', '*.tsv', '*.h5'])
        self.model.setNameFilterDisables(False)
        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.model)
        self.tree.setRootIndex(self.model.index(self.path))

        self.tree.setAnimated(False)
        self.tree.setIndentation(20)
        self.tree.setSortingEnabled(True)

        self.tree.doubleClicked.connect(self._open_file)

        self.tree.hideColumn(1)
        self.tree.hideColumn(2)
        self.tree.hideColumn(3)

        self.tree.setWindowTitle('Dir View')
        self.tree.setMinimumWidth(400)
        self.tree.resize(512, 256)

        # Add the File system tab contents
        self.imsq_pattern = QtWidgets.QLineEdit('/image_0*.ome.tif')

        self.file_tree_layout.addWidget(QtWidgets.QLabel('Image Sequence pattern:'))
        self.file_tree_layout.addWidget(self.imsq_pattern)
        self.file_tree_layout.addWidget(self.tree)

    def tabifyDocks(self):
        # Tabify docks
        pass

    def setTabPositions(self):
        self.setTabPosition(
            Qt.DockWidgetArea.LeftDockWidgetArea, QtWidgets.QTabWidget.TabPosition.East
        )
        self.setTabPosition(
            Qt.DockWidgetArea.RightDockWidgetArea, QtWidgets.QTabWidget.TabPosition.West
        )

    def raiseDocks(self):
        # Raise docks
        self.docks[DockKeys.FILE_SYSTEM].raise_()

    def createMenuBar(self):
        menu_bar = self.menuBar()

        # Create file menu
        file_menu = menu_bar.addMenu('File')
        view_menu = menu_bar.addMenu('View')
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

        # Create toggle view actions for each dock
        dock_toggle_actions = {}
        for key, dock in self.docks.items():
            toggle_action = dock.toggleViewAction()
            toggle_action.setEnabled(True)
            dock_toggle_actions[key] = toggle_action
            view_menu.addAction(toggle_action)

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
            center_point.x() - self.width() / 2,
            center_point.y() - self.height() / 2,
            self.width(),
            self.height(),
        )

    def status(self):
        # Statusbar time
        self.statusBar().showMessage(
            'Time: ' + QDateTime.currentDateTime().toString('hh:mm:ss,zzz')
        )

    def _open_file(self, i):
        # Set the Qt.WindowFlags for making the subwindow resizable
        view = None

        if not self.model.isDir(i):
            cv2.destroyAllWindows()
            index = self.model.index(i.row(), 0, i.parent())
            path = self.model.filePath(index)

            if path.endswith('.tif') or path.endswith('.tiff'):
                view = StackView.FromImageSequence(path, None)
                view.localizedData.connect(self.localizedData)
            elif path.endswith('.h5') or path.endswith('.tsv'):
                results = FittingResults.fromFile(path, 1)
                if results is not None:
                    view = LocalizationsView(path, results)
                    print('Done importing results.')
                else:
                    print('Error importing results.')
        else:
            index = self.model.index(i.row(), 0, i.parent())
            path = self.model.filePath(index)

            if path.endswith('.zarr'):
                view = StackView.FromZarr(path)
            else:
                view = StackView.FromImageSequence(path, self.imsq_pattern.text())

            view.localizedData.connect(self.localizedData)

        if view:
            window = self.mdi_area.addSubWindow(view, Qt.WindowType.SubWindow)
            window.show()

    def localizedData(self, path):
        index = self.model.index(path)
        self._open_file(index)

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

    print(f'{filename} file generated!')


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
        print(f'{filename} not found!')
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

    print(f'{filename} file loaded!')


if __name__ == '__main__':
    app, window = multi_viewer.StartGUI()
    app.exec()
