
from microEye.qt import Qt, QtGui, QtWidgets


class ChecklistDialog(QtWidgets.QDialog):

    def __init__(
            self,
            name,
            stringlist=None,
            checked=False,
            icon=None,
            parent=None,
            ):
        super().__init__(parent)

        self.name = name
        self.icon = icon
        self.model = QtGui.QStandardItemModel()
        self.listView = QtWidgets.QListView()

        for string in stringlist:
            item = QtGui.QStandardItem(string)
            item.setCheckable(True)
            check = \
                (Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
            item.setCheckState(check)
            self.model.appendRow(item)

        self.listView.setModel(self.model)

        self.export_precision = QtWidgets.QLineEdit('%10.5f')

        self.okButton = QtWidgets.QPushButton('OK')
        self.cancelButton = QtWidgets.QPushButton('Cancel')
        self.selectButton = QtWidgets.QPushButton('Select All')
        self.unselectButton = QtWidgets.QPushButton('Unselect All')

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)
        hbox.addWidget(self.selectButton)
        hbox.addWidget(self.unselectButton)

        vbox = QtWidgets.QFormLayout(self)
        vbox.addRow(self.listView)
        vbox.addRow(
            QtWidgets.QLabel('Format:'), self.export_precision)
        vbox.addRow(hbox)

        self.setWindowTitle(self.name)
        if self.icon:
            self.setWindowIcon(self.icon)

        self.okButton.clicked.connect(self.onAccepted)
        self.cancelButton.clicked.connect(self.reject)
        self.selectButton.clicked.connect(self.select)
        self.unselectButton.clicked.connect(self.unselect)

    def onAccepted(self):
        self.choices = self.toList()
        self.accept()

    def select(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def unselect(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)

    def toList(self):
        return [self.model.item(i).text() for i in
                range(self.model.rowCount())
                if self.model.item(i).checkState()
                == Qt.CheckState.Checked]


class Checklist(QtWidgets.QGroupBox):

    def __init__(
            self,
            name,
            stringlist=None,
            checked=False,
            parent=None,
            ):
        super(QtWidgets.QGroupBox, self).__init__(parent)

        self.name = name
        self.model = QtGui.QStandardItemModel()
        self.listView = QtWidgets.QListView()

        for _, string in enumerate(stringlist):
            item = QtGui.QStandardItem(string)
            item.setCheckable(True)
            check = \
                (Qt.Checked if checked else Qt.Unchecked)
            item.setCheckState(check)
            self.model.appendRow(item)

        self.listView.setModel(self.model)

        self.okButton = QtWidgets.QPushButton('OK')
        self.cancelButton = QtWidgets.QPushButton('Cancel')
        self.selectButton = QtWidgets.QPushButton('Select All')
        self.unselectButton = QtWidgets.QPushButton('Unselect All')

        hbox = QtWidgets.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.okButton)
        hbox.addWidget(self.cancelButton)
        hbox.addWidget(self.selectButton)
        hbox.addWidget(self.unselectButton)

        vbox = QtWidgets.QVBoxLayout(self)
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
                == Qt.CheckState.Checked]

    def select(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.CheckState.Checked)

    def unselect(self):
        for i in range(self.model.rowCount()):
            item = self.model.item(i)
            item.setCheckState(Qt.CheckState.Unchecked)
