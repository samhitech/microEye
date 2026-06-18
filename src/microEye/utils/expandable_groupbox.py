from weakref import WeakValueDictionary

from microEye.qt import QApplication, QtCore, QtGui, QtWidgets, Slot


class CollapsibleWidget(QtWidgets.QWidget):
    OPEN_STYLE = '''
            QToolButton {
                padding: 5px;
                spacing: 5px;
                text-align: center;
            }
        '''

    CLOSED_STYLE = '''
            QToolButton {
                padding: 5px;
                spacing: 5px;
                text-align: center;
                font-weight: 700;
            }
        '''

    def __init__(self, title='', parent=None):
        super().__init__(parent)

        self.toggle_button = QtWidgets.QToolButton(
            text=title, checkable=True, checked=False
        )
        self.toggle_button.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.toggle_button.setArrowType(QtCore.Qt.ArrowType.RightArrow)
        self.toggle_button.setIconSize(QtCore.QSize(6, 6))
        self.toggle_button.toggled.connect(self.on_toggled)

        self.toggle_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.toggle_button.setStyleSheet(CollapsibleWidget.CLOSED_STYLE)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QtWidgets.QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed
        )
        self.content_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle_button)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b'minimumHeight')
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self, b'maximumHeight')
        )
        self.toggle_animation.addAnimation(
            QtCore.QPropertyAnimation(self.content_area, b'maximumHeight')
        )

    @Slot()
    def on_toggled(self):
        checked = self.toggle_button.isChecked()

        # animate when user toggles; the heavy lifting is in setExpanded
        self.setExpanded(checked, animate=True)

    def setExpanded(self, expanded: bool, animate: bool = False):
        self.toggle_button.setArrowType(
            QtCore.Qt.ArrowType.DownArrow
            if expanded
            else QtCore.Qt.ArrowType.RightArrow
        )
        self.toggle_animation.setDirection(
            QtCore.QAbstractAnimation.Direction.Forward
            if expanded
            else QtCore.QAbstractAnimation.Direction.Backward
        )
        self.toggle_button.setStyleSheet(
            CollapsibleWidget.OPEN_STYLE
            if expanded
            else CollapsibleWidget.CLOSED_STYLE
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout: QtWidgets.QLayout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(100)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(100)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class SideBar(QtWidgets.QScrollArea):
    __INDEX = 0

    def __init__(self, parent=None):
        super().__init__(parent)

        self._sections = {}

        self.__content = QtWidgets.QWidget()
        self.setWidget(self.__content)
        self.setWidgetResizable(True)
        self._layout = QtWidgets.QVBoxLayout(self.__content)

    def addSection(self, title: str, layout: QtWidgets.QLayout, checked=False):
        section = CollapsibleWidget(title=title)
        self._layout.addWidget(section)
        section.setContentLayout(layout)
        self._sections[SideBar.__INDEX] = section

        section.toggle_button.setChecked(checked)

        SideBar.__INDEX += 1

        return layout

    def addLayout(self, layout: QtWidgets.QLayout):
        self._layout.addLayout(layout)

    def addStretch(self):
        self._layout.addStretch()


if __name__ == '__main__':
    import random
    import sys

    app = QApplication(sys.argv)

    win = QtWidgets.QMainWindow()
    win.setCentralWidget(QtWidgets.QWidget())
    dock = QtWidgets.QDockWidget('Collapsible Demo')
    win.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
    side_bar = SideBar()
    dock.setWidget(side_bar)

    for i in range(10):
        lay = QtWidgets.QVBoxLayout()
        for j in range(8):
            label = QtWidgets.QLabel(f'{j}')
            color = QtGui.QColor(*[random.randint(0, 255) for _ in range(3)])
            label.setStyleSheet(f'background-color: {color.name()}; color : white;')
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            lay.addWidget(label)
        side_bar.addSection(f'Collapsible Box Header-{i}', lay)

    side_bar.addStretch()
    win.resize(720, 480)
    win.show()
    sys.exit(app.exec())
