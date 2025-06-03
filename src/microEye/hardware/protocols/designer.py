import json
import sys

from microEye import __version__
from microEye.hardware.protocols.actions import (
    ForLoop,
    ParameterAdjustmentAction,
)
from microEye.hardware.protocols.scene_manager import SceneManager
from microEye.qt import QApplication, Qt, QtCore, QtGui, QtWidgets


class ExperimentDesignerView(QtWidgets.QGraphicsView):
    '''
    Custom QGraphicsView with zooming and key event handling.

    Subclass of QGraphicsView.

    Attributes
    ----------
    root : ActionGroupItem
        The root item in the scene.
    '''

    def __init__(self):
        '''
        Initialize a new ExperimentDesignerView instance.
        '''
        super().__init__()
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)

        # Create an instance of the SceneManager
        self.scene_manager = SceneManager()
        self.setScene(self.scene_manager.scene)

        # Set dark background color for the view
        background_color = QtGui.QColor('#333333')  # Adjust the color as needed
        self.setBackgroundBrush(background_color)

    def wheelEvent(self, event: QtWidgets.QGraphicsSceneWheelEvent):
        '''
        Handle wheel events for zooming.

        Parameters
        ----------
        event : QGraphicsSceneWheelEvent
            The wheel event.
        '''
        zoom_factor = 1.2

        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)

        super().wheelEvent(event)

    def keyPressEvent(self, event):
        '''
        Handle key press events for swapping and deletion.

        Parameters
        ----------
        event : QKeyEvent
            The key event.
        '''
        self.scene_manager.handleKeyPress(event)

        super().keyPressEvent(event)

    def contextMenuEvent(self, event):
        '''
        Handle context menu events.

        Parameters
        ----------
        event : QContextMenuEvent
            The context menu event.
        '''
        context_menu = QtWidgets.QMenu()

        # Add option to add different types
        add_menu = context_menu.addMenu('Add')
        # add_menu.addAction(
        #     'Function Call', lambda: self.scene_manager.add_action(FunctionCall())
        # )
        add_menu.addAction('For Loop', lambda: self.scene_manager.add_action(ForLoop()))
        add_menu.addAction(
            'Parameter Adjustment',
            lambda: self.scene_manager.add_action(ParameterAdjustmentAction()),
        )

        # Add Move submenu
        move_submenu = context_menu.addMenu('Move')
        move_submenu.addAction(
            'Up',
            lambda: self.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    Qt.Key.Key_PageUp,
                    Qt.KeyboardModifier.NoModifier,
                    0,
                    0,
                    0,
                )
            ),
        )
        move_submenu.addAction(
            'Down',
            lambda: self.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    Qt.Key.Key_PageDown,
                    Qt.KeyboardModifier.NoModifier,
                    0,
                    0,
                    0,
                )
            ),
        )

        context_menu.addAction(
            'Delete',
            lambda: self.keyPressEvent(
                QtGui.QKeyEvent(
                    QtCore.QEvent.Type.KeyPress,
                    Qt.Key.Key_Delete,
                    Qt.KeyboardModifier.NoModifier,
                    0,
                    0,
                    0,
                )
            ),
        )

        # Add Execute action
        context_menu.addAction('Execute', self.scene_manager.execute_actions)
        context_menu.addAction('Stop Execution', self.scene_manager.stop_execution)

        # Add Export and Import actions
        context_menu.addAction('Export (protocol)', self.export_protocol)
        context_menu.addAction('Import (protocol)', self.import_protocol)
        context_menu.addAction('Import & Execute', self.import_execute_protocol)

        # Show the context menu
        context_menu.exec(event.globalPos())

    def export_protocol(self):
        '''
        Handle exporting the protocol to a JSON file.
        '''
        file_dialog = QtWidgets.QFileDialog()
        file_name, _ = file_dialog.getSaveFileName(
            self, 'Export Protocol', '', 'JSON Files (*.json)'
        )

        if file_name:
            with open(file_name, 'w') as file:
                protocol_data = self.scene_manager.serialize_protocol()
                json.dump(protocol_data, file, indent=4)

    def import_protocol(self):
        '''
        Handle importing the protocol from a JSON file.
        '''
        file_dialog = QtWidgets.QFileDialog()
        file_name, _ = file_dialog.getOpenFileName(
            self, 'Import Protocol', '', 'JSON Files (*.json)'
        )

        if not file_name:
            return False

        with open(file_name) as file:
            protocol_data = json.load(file)
            self.scene_manager.load_actions(protocol_data)

        return True

    def import_execute_protocol(self):
        '''
        Handle importing and executing the protocol from a JSON file.
        '''
        if self.import_protocol():
            self.scene_manager.execute_actions()

class ExperimentDesigner(QtWidgets.QWidget):
    HEADER = '> <span style="color:#0f0;">Experiment Designer ('
    HEADER += f'<span style="color:#aaf;">microEye v{__version__}</span>)</span>'

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Create the ExperimentDesignerView
        self.experiment_view = ExperimentDesignerView()

        # Create the text terminal
        self.text_terminal = QtWidgets.QPlainTextEdit()
        self.text_terminal.setReadOnly(True)
        self.text_terminal.appendHtml(self.HEADER)
        self.text_terminal.setStyleSheet(
            'QPlainTextEdit { background-color: #111;}'
        )

        self.experiment_view.scene_manager.updateTerminal.connect(self.updateTerminal)
        self.experiment_view.scene_manager.clearTerminal.connect(self.clearTerminal)

        # Set up the layout
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.experiment_view, 2)
        layout.addWidget(self.text_terminal, 1)
        self.setLayout(layout)

        # Set window properties
        self.setWindowTitle('Experiment Designer')

    def updateTerminal(self, text: str, level: int):
        indent = '>' * (level + 1) + ' '
        if ':' in text and len(text.split(':')) == 2:
            text = text.split(':')
            self.text_terminal.appendHtml(
                f'{indent}<span style="color: #0f0;">{text[0]}: </span>'
                f'<span style="color: #aaa;">{text[1]}</span>'
            )
        else:
            self.text_terminal.appendHtml(
                f'{indent}<span style="color: #0f0;">{text}</span>'
            )

    def clearTerminal(self):
        self.text_terminal.clear()
        self.text_terminal.appendHtml(self.HEADER)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ExperimentDesigner()
    editor.show()
    sys.exit(app.exec())
