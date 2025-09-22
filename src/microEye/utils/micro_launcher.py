import importlib
import os
import subprocess
import sys
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version

from microEye.qt import QApplication, Qt, QtCore, QtGui, QtSvg, QtWidgets
from microEye.utils.start_gui import StartGUI


def check_modules(modules):
    availability = {}
    for module in modules:
        try:
            imported_module = importlib.import_module(module)
            try:
                _version = (
                    imported_module.__version__
                    if hasattr(imported_module, '__version__')
                    else version(module)
                )
            except PackageNotFoundError:
                _version = 'Version not found'
            availability[module] = _version if _version != 'Version not found' else None
        except ImportError:
            availability[module] = None
    return availability


class microLauncher(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Remove the title bar
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Set round corners, white border, and dark gray background
        self.setStyleSheet('''
            microLauncher {
                border: 1px solid white !important;
                padding: 0;
            }
            QPushButton#closeButton {
                border: none;
            }
            QPushButton#minButton {
                border: none;
            }
        ''')

        self.dirname = os.path.dirname(os.path.abspath(__file__))

        width = 160
        height = 130

        # Create a close button with SVG icon
        self.close_button = QtWidgets.QPushButton()
        self.close_button.setObjectName('closeButton')
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)

        self.minimize_button = QtWidgets.QPushButton()
        self.minimize_button.setObjectName('minButton')
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.clicked.connect(lambda: self.showMinimized())

        # Load SVG and set as icon
        svg_renderer = QtSvg.QSvgRenderer(
            os.path.join(self.dirname, '../icons/close.svg')
        )
        svg_renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        svg_pixmap = QtGui.QPixmap(24, 24)
        svg_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(svg_pixmap)
        svg_renderer.render(painter)
        painter.end()
        self.close_button.setIcon(QtGui.QIcon(svg_pixmap))
        self.close_button.setIconSize(QtCore.QSize(24, 24))

        svg_renderer = QtSvg.QSvgRenderer(
            os.path.join(self.dirname, '../icons/min.svg')
        )
        svg_renderer.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        svg_pixmap = QtGui.QPixmap(24, 24)
        svg_pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        painter = QtGui.QPainter(svg_pixmap)
        svg_renderer.render(painter)
        painter.end()
        self.minimize_button.setIcon(QtGui.QIcon(svg_pixmap))
        self.minimize_button.setIconSize(QtCore.QSize(24, 24))

        # Layout for close button
        self.close_button_layout = QtWidgets.QHBoxLayout()
        self.close_button_layout.addStretch()
        self.close_button_layout.addWidget(self.minimize_button)
        self.close_button_layout.addWidget(self.close_button)

        # Set the main layout
        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.addLayout(self.close_button_layout)

        # Create buttons with images
        self.button1 = QtWidgets.QPushButton()
        pixmap1 = QtGui.QPixmap(
            os.path.join(self.dirname, '../icons/mieye.png')
        ).scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.button1.setIcon(QtGui.QIcon(pixmap1))
        self.button1.setIconSize(QtCore.QSize(width, height))
        self.button1.clicked.connect(lambda: self.launch_module())

        self.button2 = QtWidgets.QPushButton()
        pixmap2 = QtGui.QPixmap(
            os.path.join(self.dirname, '../icons/viewer.png')
        ).scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.button2.setIcon(QtGui.QIcon(pixmap2))
        self.button2.setIconSize(QtCore.QSize(width, height))
        self.button2.clicked.connect(lambda: self.launch_module('viewer'))

        # Create comboboxes
        self.qt_api_combobox = QtWidgets.QComboBox()
        self.populate_combobox(self.qt_api_combobox, ['PySide6', 'PyQt5', 'PyQt6'])

        self.theme_combobox = QtWidgets.QComboBox()
        self.populate_combobox(
            self.theme_combobox, ['qdarktheme', 'qdarkstyle', 'None'], True
        )

        # Layout for buttons
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.buttons_layout.addWidget(self.button1)
        self.buttons_layout.addWidget(self.button2)

        # Add widgets to the main layout
        self.main_layout.addLayout(self.buttons_layout)
        self.main_layout.addWidget(self.qt_api_combobox)
        self.main_layout.addWidget(self.theme_combobox)

        # Set central widget
        central_widget = QtWidgets.QWidget()
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Window size and title
        self.setWindowTitle('microEye Launcher')
        # self.resize(400, 300)
        self.setContentsMargins(0, 0, 0, 0)

        self.show()
        self.center()

    def launch_module(self, module='mieye'):
        '''Launch the selected module'''
        qt_api = self.qt_api_combobox.currentText()
        miTheme = self.theme_combobox.currentText()

        env = dict(os.environ, QT_API=qt_api, PYQTGRAPH_QT_LIB=qt_api, MITHEME=miTheme)

        # Define the command and arguments
        cmd = ['python', '-m', 'microEye.launcher', '--module', module]

        # Get the current working directory
        cwd = os.getcwd()

        # Open a new terminal and run the command
        if os.name == 'nt':  # Windows
            flags = 0
            flags |= subprocess.DETACHED_PROCESS
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
            flags |= subprocess.CREATE_BREAKAWAY_FROM_JOB
            flags |= subprocess.CREATE_NO_WINDOW

            pkwargs = {
                'close_fds': True,  # close stdin/stdout/stderr on child
                'creationflags': flags,
            }
            subprocess.Popen(
                ['start', 'cmd', '/k'] + cmd, cwd=cwd, shell=True, env=env, **pkwargs
            )
            # self.close()
        else:  # Unix-based systems (Linux, macOS, etc.)
            subprocess.Popen(
                ['setsid', 'gnome-terminal', '--working-directory=' + cwd, '--'] + cmd,
                env=env,
            )
            self.close()

    def center(self):
        '''Centers the window within the screen.'''
        qtRectangle = self.frameGeometry()
        centerPoint = QApplication.primaryScreen().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def populate_combobox(self, combobox, items: Iterable[str], theme=False):
        model = QtGui.QStandardItemModel()
        availability = check_modules(items)
        for item in items:
            standard_item = QtGui.QStandardItem(item)
            if availability[item] is None and item.lower() != 'none':
                disabledColor = QtGui.QBrush(QtGui.QColor(Qt.GlobalColor.red))
                standard_item.setData(
                    disabledColor, QtCore.Qt.ItemDataRole.ForegroundRole
                )
                standard_item.setEnabled(False)
                standard_item.setToolTip(f'{standard_item.text()} is not installed')
            elif item.lower() != 'none':
                prefix = (
                    '' if availability[item] == 'Version not found' else 'Version: '
                )
                standard_item.setToolTip(f'{prefix}{availability[item]}')
            model.appendRow(standard_item)

        if theme:
            for item in QtWidgets.QStyleFactory.keys():  # noqa: SIM118
                standard_item = QtGui.QStandardItem(item)
                model.appendRow(standard_item)
        combobox.setModel(model)

    def StartGUI():
        '''
        Initializes a new QApplication and microLauncher.

        Parameters
        ----------
        path : str, optional
            The path to a file to be loaded initially.

        Returns
        -------
        tuple of QApplication and microLauncher
            Returns a tuple with QApplication and microLauncher window.
        '''
        return StartGUI(microLauncher, theme=None)


if __name__ == '__main__':
    app, window = microLauncher.StartGUI()
    app.exec()
