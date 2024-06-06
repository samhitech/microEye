from microEye.qt import QApplication, QtWidgets


class ExpandableGroupBox(QtWidgets.QGroupBox):
    """
    Custom QGroupBox with expandable functionality.

    Parameters
    ----------
    title : str
        The title of the group box.
    parent : QWidget, optional
        The parent widget.

    Attributes
    ----------
    None

    Signals
    -------
    toggled : bool
        Emitted when the group box is toggled (expanded or collapsed).

    Methods
    -------
    init_ui()
        Initialize the user interface.
    mouseDoubleClickEvent(event)
        Handle the mouse press event to toggle the expansion state.
    setFlat(flat)
        Override the setFlat method to adjust the visibility of child widgets.

    Example
    -------
    >>> app = QApplication([])
    >>> main_widget = QWidget()
    >>> main_layout = QVBoxLayout(main_widget)
    >>> expandable_group = ExpandableGroupBox('Expandable Group')
    >>> button1 = QPushButton('Button 1')
    >>> button2 = QPushButton('Button 2')
    >>> expandable_group.layout().addWidget(button1)
    >>> expandable_group.layout().addWidget(button2)
    >>> main_layout.addWidget(expandable_group)
    >>> main_widget.show()
    >>> app.exec()
    """

    def __init__(self, title: str, parent=None):
        '''
        Parameters
        ----------
        title : str
            The title of the group box.
        parent : QWidget, optional
            The parent widget.
        '''
        super().__init__(title, parent)
        self.init_ui()

    def init_ui(self):
        '''Initialize the user interface.'''
        group_layout = QtWidgets.QVBoxLayout()
        self.setLayout(group_layout)
        self.expanded = True

    def mouseDoubleClickEvent(self, event):
        '''Handle the mouse press event to toggle the expansion state.'''
        self.expanded = not self.expanded
        self.setFlat(not self.expanded)
        # Hide or show child widgets based on the expansion state
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i)
            if item.widget():
                item.widget().setVisible(self.expanded)
                # Set the minimum height dynamically based on the expansion state
                if self.expanded:
                    item.widget().setMinimumHeight(125)
                else:
                    item.widget().setMinimumHeight(0)


if __name__ == '__main__':
    app = QApplication([])
    main_widget = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(main_widget)
    expandable_group = ExpandableGroupBox('Expandable Group')
    button1 = QtWidgets.QPushButton('Button 1')
    button2 = QtWidgets.QPushButton('Button 2')
    expandable_group.layout().addWidget(button1)
    expandable_group.layout().addWidget(button2)
    main_layout.addWidget(expandable_group)
    main_widget.show()
    app.exec()
