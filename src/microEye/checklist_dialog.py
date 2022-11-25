
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class ChecklistDialog(QDialog):

    def __init__(
            self,
            name,
            stringlist=None,
            checked=False,
            icon=None,
            parent=None,
            ):
        super(ChecklistDialog, self).__init__(parent)

        self.name = name
        self.icon = icon
        self.model = QStandardItemModel()
        self.listView = QListView()

        for string in stringlist:
            item = QStandardItem(string)
            item.setCheckable(True)
            check = \
                (Qt.Checked if checked else Qt.Unchecked)
            item.setCheckState(check)
            self.model.appendRow(item)

        self.listView.setModel(self.model)

        self.okButton = QPushButton('OK')
        self.cancelButton = QPushButton('Cancel')
        self.selectButton = QPushButton('Select All')
        self.unselectButton = QPushButton('Unselect All')

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)
        hbox.addWidget(self.selectButton)
        hbox.addWidget(self.unselectButton)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.listView)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setWindowTitle(self.name)
        if self.icon:
            self.setWindowIcon(self.icon)

        self.okButton.clicked.connect(self.onAccepted)
        self.cancelButton.clicked.connect(self.reject)
        self.selectButton.clicked.connect(self.select)
        self.unselectButton.clicked.connect(self.unselect)

    def onAccepted(self):
        self.choices = [self.model.item(i).text() for i in
                        range(self.model.rowCount())
                        if self.model.item(i).checkState()
                        == Qt.Checked]
        self.accept()

    def select(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.Checked)

    def unselect(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.Unchecked)


class Checklist(QGroupBox):

    def __init__(
            self,
            name,
            stringlist=None,
            checked=False,
            parent=None,
            ):
        super(QGroupBox, self).__init__(parent)

        self.name = name
        self.model = QStandardItemModel()
        self.listView = QListView()

        for idx, string in enumerate(stringlist):
            item = QStandardItem(string)
            item.setCheckable(True)
            check = \
                (Qt.Checked if checked else Qt.Unchecked)
            item.setCheckState(check)
            self.model.appendRow(item)

        self.listView.setModel(self.model)

        self.okButton = QPushButton('OK')
        self.cancelButton = QPushButton('Cancel')
        self.selectButton = QPushButton('Select All')
        self.unselectButton = QPushButton('Unselect All')

        hbox = QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)
        hbox.addWidget(self.selectButton)
        hbox.addWidget(self.unselectButton)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.listView)
        vbox.addStretch(1)
        vbox.addLayout(hbox)

        self.setTitle(self.name)

        self.selectButton.clicked.connect(self.select)
        self.unselectButton.clicked.connect(self.unselect)

    def toList(self):
        return [self.model.item(i).text() for i in
                range(self.model.rowCount())
                if self.model.item(i).checkState()
                == Qt.Checked]

    def select(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.Checked)

    def unselect(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.Unchecked)
