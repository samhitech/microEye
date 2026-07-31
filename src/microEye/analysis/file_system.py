import logging
import os
import re
import threading
from weakref import ref

from microEye import __version__
from microEye.analysis.fitting.results import FittingResults
from microEye.analysis.viewer import LocalizationsView, PSFView, StackView
from microEye.qt import (
    QApplication,
    QFileSystemModel,
    QIcon,
    Qt,
    QtCore,
    QtGui,
    QtWidgets,
    Signal,
    Slot,
)

logger = logging.getLogger(__name__)


def _compact_display_name(path, type):
    # extract drive if on windows
    if os.name == 'nt':
        drive, tail = os.path.splitdrive(path)
        parts = tail.strip('/').split('/')
        drive = f'[💻{drive}]'
    else:
        drive = '[💻ROOT]'
        parts = path.strip('/').split('/')

    parent = '' if len(parts) < 2 else f'\n[📁{parts[-2]}]'
    basename = parts[-1]

    # file name pattern 00__image_00000_roi_00.ome.tif
    # extrat roi if exists regex
    match = re.search(r'.*?(\d+)__image_(\d+)_roi_(\d+)', basename)
    if match:
        prefix = match.group(1)
        image_num = match.group(2)
        roi_num = match.group(3)
        return (
            f'{drive} [{type}]\n[{prefix}] [Image {image_num}] [ROI {roi_num}]{parent}'
        )
    elif len(basename) > 20:
        return f'{drive} [{type}]\n...{basename[-20:]}{parent}'
    else:
        return f'{drive} [{type}]\n{basename}{parent}'


class CustomFileSystemModel(QFileSystemModel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def filterAcceptsRow(self, source_row, source_parent):
        index = self.index(source_row, 0, source_parent)
        if not index.isValid():
            return False

        # Get the file/folder name and path
        file_info = self.fileInfo(index)
        file_name = file_info.fileName()
        parent_path = file_info.absolutePath()

        # Check if the parent directory is a .zarr folder
        if parent_path.endswith('.zarr'):
            return False  # Do not show any files within a .zarr folder

        # Check if the current directory is a .zarr folder
        if file_name.endswith('.zarr'):
            return True  # Always accept .zarr folders themselves

        # Use the default filtering behavior for other files/folders
        return super().filterAcceptsRow(source_row, source_parent)

    def isDir(self, index):
        # Override the isDir method to treat .zarr folders as files
        file_info = self.fileInfo(index)
        if file_info.isDir() and file_info.fileName().endswith('.zarr'):
            return False  # Treat .zarr folders as files
        return super().isDir(index)

    def data(self, index, role):
        # Optionally, customize the display role to show the .zarr folders as files
        file_info = self.fileInfo(index)
        if (
            role == QFileSystemModel.Roles.FileIconRole
            and file_info.fileName().endswith('.zarr')
        ):
            # Get the icon for a ZIP file (or any compressed file)
            zip_icon = QIcon.fromTheme('application-zip')  # On Linux
            if zip_icon.isNull():
                zip_icon = self.iconProvider().icon(
                    QtCore.QFileInfo('dummy.zip')
                )  # Fallback for Windows/Mac
            return zip_icon  # Return ZIP icon for .zarr folders

        return super().data(index, role)

    def hasChildren(self, index):
        # Ensure that .zarr folders don't expand by showing no children
        file_info = self.fileInfo(index)
        if file_info.isDir() and file_info.fileName().endswith('.zarr'):
            return False  # Treat .zarr folders as files with no children
        return super().hasChildren(index)


class FileSystemTree(QtWidgets.QTreeView):
    openFile = Signal(QtCore.QModelIndex, str)

    def __init__(self, path: str = None, parent=None, **kwargs):
        super().__init__(parent)

        self.__path = path

        self.__filters = kwargs.get(
            'filters', ['*.tif', '*.tiff', '*.tsv', '*.h5', '*.zarr']
        )

        # Create QFileSystemModel
        self.__model = CustomFileSystemModel()
        self.__model.setRootPath(self.__path)
        self.__model.setFilter(
            QtCore.QDir.Filter.AllDirs
            | QtCore.QDir.Filter.Files
            | QtCore.QDir.Filter.NoDotAndDotDot
        )
        self.__model.setNameFilters(self.__filters)
        self.__model.setNameFilterDisables(False)

        self.setModel(self.__model)
        self.setRootIndex(self.__model.index(path))
        self.setAnimated(False)
        self.setIndentation(20)
        self.setSortingEnabled(False)
        self.hideColumn(1)
        self.hideColumn(2)
        self.hideColumn(3)
        self.setWindowTitle('Dir View')
        self.setMinimumWidth(400)

        # Connect double-click signal to a method that uses QTimer
        self.doubleClicked.connect(self._open_file)

        # Create Tree view context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_tree_context_menu)

    def filePath(self, index: QtCore.QModelIndex):
        return self.__model.filePath(index)

    def index(self, path: str):
        return self.__model.index(path)

    def _show_tree_context_menu(self, position):
        index = self.indexAt(position)
        if not index.isValid():
            return

        menu = QtWidgets.QMenu()
        reveal_action = menu.addAction('Reveal in Explorer')
        new_folder_action = menu.addAction('New Folder')
        menu.addSeparator()

        # Root path shortcuts
        root_action = menu.addAction('This PC (Root Path)')
        home_action = menu.addAction('Home (User Directory)')
        desktop_action = menu.addAction('Desktop')
        initial_action = menu.addAction('Initial Path')
        working_dir_action = menu.addAction('Working Directory')

        action = menu.exec(self.viewport().mapToGlobal(position))
        if action == reveal_action:
            path = self.__model.filePath(index)
            folder = path if os.path.isdir(path) else os.path.dirname(path)
            os.startfile(folder)
        elif action == new_folder_action:
            self._create_new_folder()
        elif action == home_action:
            home = os.path.expanduser('~')
            self.setRootIndex(self.__model.index(home))
        elif action == desktop_action:
            desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
            self.setRootIndex(self.__model.index(desktop))
        elif action == root_action:
            self.setRootIndex(self.__model.index(''))
        elif action == initial_action:
            self.setRootIndex(self.__model.index(self.path))
        elif action == working_dir_action:
            self.setRootIndex(
                self.__model.index(os.path.dirname(os.path.abspath(__package__)))
            )

    def _create_new_folder(self):
        index = self.currentIndex()
        if not index.isValid():
            return

        path = self.__model.filePath(index)
        if not os.path.isdir(path):
            path = os.path.dirname(path)

        new_folder_name, ok = QtWidgets.QInputDialog.getText(
            self, 'New Folder', 'Enter folder name:'
        )
        if ok and new_folder_name:
            new_folder_path = os.path.join(path, new_folder_name)
            try:
                os.makedirs(new_folder_path)
            except OSError as e:
                QtWidgets.QMessageBox.critical(
                    self, 'Error', f'Could not create folder: {e}'
                )

    @Slot(QtCore.QModelIndex)
    def _open_file(self, index: QtCore.QModelIndex):
        path = self.filePath(index)

        if not os.path.isdir(path):
            if path.endswith('.tif') or path.endswith('.tiff'):
                file_type = 'TIFF'
            elif path.endswith('.zarr'):
                file_type = 'ZARR'
            elif (
                path.endswith('.h5') and not path.endswith('.psf.h5')
            ) or path.endswith('.tsv'):
                file_type = 'HDF5' if path.endswith('.h5') else 'TSV'
            elif path.endswith('.psf.h5'):
                file_type = 'PSF'
        else:
            file_type = 'ZARR' if path.endswith('.zarr') else 'ImageSeq'

        self.openFile.emit(index, file_type)


class FileSystem(QtWidgets.QTabWidget):
    openFile = Signal(QtCore.QModelIndex, str)
    focusItem = Signal(ref)

    def __init__(self, path: str = None, parent=None, **kwargs):
        super().__init__(parent)

        self.setWindowTitle('Files')

        # {path: {'type': str, 'widget': QWidget}}
        self._opened_files: dict[str, dict[str, QtWidgets.QWidget]] = {}

        self._opening = threading.Lock()

        self._file_tree = FileSystemTree(path, **kwargs)
        self._file_tree.openFile.connect(self.openFile.emit)

        # Add the File system tab contents
        self._imsq_pattern = QtWidgets.QLineEdit('/image_0*.ome.tif')

        self._file_tree_layout = QtWidgets.QVBoxLayout()
        self._file_tree_widget = QtWidgets.QWidget()
        self._file_tree_widget.setLayout(self._file_tree_layout)

        self._file_tree_layout.addWidget(QtWidgets.QLabel('Image Sequence pattern:'))
        self._file_tree_layout.addWidget(self._imsq_pattern)
        self._file_tree_layout.addWidget(self._file_tree)

        self.addTab(self._file_tree_widget, 'Files System')

        self._opened_files_list = QtWidgets.QListWidget(self)
        self._opened_files_list.setWindowTitle('Opened Files')
        self._opened_files_list.itemClicked.connect(self._focus_opened_file)

        self.addTab(self._opened_files_list, 'Opened Files')

        self._standalone = kwargs.get('standalone', False)
        if self._standalone:
            self.openFile.connect(self._open_file_dialog)

        self.resize(512, 720)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)


    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True

        if self._standalone:
            for key, dic in self._opened_files.items():
                try:
                    dic['widget'].destroy()
                except Exception as e:
                    logger.error(f'Error Destroying {key}: {e}')

        return super().closeEvent(event)


    @property
    def file_pattern(self) -> str:
        return self._imsq_pattern.text()

    def _update_opened_files_list(self):
        if getattr(self, '_opened_files_list', None) is None:
            return

        self._opened_files_list.clear()
        for path, info in self._opened_files.items():
            display_name = _compact_display_name(path, type=info['type'])
            item = QtWidgets.QListWidgetItem(f'{display_name}')
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self._opened_files_list.addItem(item)

    def _focus_opened_file(self, item):
        if isinstance(item, str):
            path = item
        elif isinstance(item, QtWidgets.QListWidgetItem):
            path = item.data(QtCore.Qt.ItemDataRole.UserRole)

        if self._standalone:
            view: QtWidgets.QWidget = self._opened_files[path]['widget']
            view.show()
            view.raise_()

        self.focusItem.emit(ref(self._opened_files[path]['widget']))

    @Slot(QtCore.QModelIndex, str)
    def _open_file_dialog(self, index: QtCore.QModelIndex, file_type: str):
        # Set the Qt.WindowFlags for making the subwindow resizable
        with self._opening:
            view = None

            path = self.filePath(index)

            if path in self._opened_files:
                self._focus_opened_file(path)
                return

            if file_type in ['TIFF', 'ZARR', 'ImageSeq']:
                pattern = self.file_pattern if file_type == 'ImageSeq' else None
                try:
                    view = StackView(path, pattern)
                    view.localizedData.connect(self.localizedData)
                except Exception as e:
                    view = None
                    logger.error(f'Error opening: {e}')

            elif file_type in ['HDF5', 'TSV']:
                view = LocalizationsView(path=path)
            elif file_type == 'PSF':
                view = PSFView(path)

            if view:
                self._opened_files[path] = {'type': file_type, 'widget': view}
                view.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
                view.destroyed.connect(lambda: self._on_subwindow_close(path))

            self._update_opened_files_list()

    def _on_subwindow_close(self, path):
        if getattr(self, '_closing', False):
            return

        if path in self._opened_files:
            del self._opened_files[path]
            self._update_opened_files_list()

    def filePath(self, index: QtCore.QModelIndex):
        return self._file_tree.filePath(index)

    def index(self, path: str):
        return self._file_tree.index(path)

    def localizedData(self, path):
        index = self._file_tree.index(path)
        self._file_tree._open_file(index)

    # @classmethod
    # def get_dialog(cls, path: str, **kwargs):
    #     fs = cls(path=path, standalone=True, **kwargs)

    #     dialog = QtWidgets.QDialog()

    #     dialog.setLayout(QtWidgets.QVBoxLayout())

    #     dialog.layout().addWidget(fs)

    #     return dialog

if __name__ == '__main__':
    app = QApplication([])

    fs = FileSystem(None, standalone=True)
    fs.show()

    app.exec()
