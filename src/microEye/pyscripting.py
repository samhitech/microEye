
from pyqode.python.widgets import PyCodeEdit
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class pyEditor(QWidget):

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)

        self.mLayout = QVBoxLayout()
        self.setLayout(self.mLayout)

        self.pyEditor = PyCodeEdit(color_scheme='darcula')
        self.pyEditor.setAcceptDrops(True)
        self.pyEditor.dragEnterEvent = self.pydragEnterEvent
        self.pyEditor.dropEvent = (
            lambda e: self.pyEditor.insertPlainText(e.mimeData().text()))

        self.btns_layout = QHBoxLayout()
        self.open_btn = QPushButton(
            'Open',
            clicked=lambda: self.openScript())
        self.save_btn = QPushButton(
            'Save',
            clicked=lambda: self.saveScript())
        self.exec_btn = QPushButton('Execute')
        self.btns_layout.addWidget(self.open_btn)
        self.btns_layout.addWidget(self.save_btn)
        self.btns_layout.addWidget(self.exec_btn)

        self.mLayout.addWidget(self.pyEditor)
        self.mLayout.addLayout(self.btns_layout)

    def openScript(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Script", filter="Python Files (*.py);;")

        if len(filename) > 0:
            with open(filename, 'r', encoding='utf-8') as file:
                self.pyEditor.setPlainText(file.read())

    def saveScript(self):
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Script", filter="Python Files (*.py);;")

        if len(filename) > 0:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(self.pyEditor.toPlainText())

    def pydragEnterEvent(self, e):
        if e.mimeData().hasFormat('text/plain'):
            e.accept()
        else:
            e.ignore()

    def toPlainText(self) -> str:
        return self.pyEditor.toPlainText()
