import numpy as np
import pyqtgraph as pg

from microEye.qt import QApplication, Qt, QtGui, QtWidgets, Signal


class ImageItemOptions(QtWidgets.QWidget):
    paramsChanged = Signal(int, str, object)  # layer_index, param_name, value

    def __init__(self, layer_index, parent=None):
        super().__init__(parent)
        self.layer_index = layer_index

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Visible checkbox
        self.visible_check = QtWidgets.QCheckBox('Visible')
        self.visible_check.setChecked(True)
        self.visible_check.stateChanged.connect(
            lambda state: self.paramsChanged.emit(
                self.layer_index, 'Visible', bool(state)
            )
        )
        layout.addWidget(self.visible_check)

        # Opacity slider with value
        opacity_widget = QtWidgets.QWidget()
        opacity_layout = QtWidgets.QVBoxLayout()
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        opacity_widget.setLayout(opacity_layout)

        opacity_layout.addWidget(QtWidgets.QLabel('Opacity:'))

        opacity_slider_layout = QtWidgets.QHBoxLayout()
        self.opacity_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(
            lambda value: self.paramsChanged.emit(self.layer_index, 'Opacity', value)
        )

        self.opacity_value = QtWidgets.QSpinBox()
        self.opacity_value.setRange(0, 100)
        self.opacity_value.setValue(100)
        self.opacity_value.setFixedWidth(75)

        self.opacity_value.valueChanged.connect(self.opacity_slider.setValue)
        self.opacity_slider.valueChanged.connect(self.opacity_value.setValue)

        opacity_slider_layout.addWidget(self.opacity_slider)
        opacity_slider_layout.addWidget(self.opacity_value)
        opacity_layout.addLayout(opacity_slider_layout)

        layout.addWidget(opacity_widget)

        # Composition mode
        comp_widget = QtWidgets.QWidget()
        comp_layout = QtWidgets.QVBoxLayout()
        comp_layout.setContentsMargins(0, 0, 0, 0)
        comp_widget.setLayout(comp_layout)

        comp_layout.addWidget(QtWidgets.QLabel('Composition Mode:'))
        self.comp_combo = QtWidgets.QComboBox()
        self.comp_combo.addItems(ImageItemsWidget.COMP_MODES)
        self.comp_combo.setCurrentText('SourceOver')
        self.comp_combo.currentTextChanged.connect(
            lambda text: self.paramsChanged.emit(
                self.layer_index, 'CompositionMode', text
            )
        )
        comp_layout.addWidget(self.comp_combo)

        layout.addWidget(comp_widget)
        layout.addStretch()


class CustomListWidget(QtWidgets.QListWidget):
    '''Custom list widget with enhanced font handling
    and curent item index next to it.'''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QtWidgets.QFrame.Shape.NoFrame)
        self.setSpacing(2)

        # Set larger font size for all items
        font = self.font()
        font.setPointSize(10)  # Adjust size as needed
        self.setFont(font)

    def addItem(self, item):
        '''Override addItem to add index prefix to items'''
        if isinstance(item, str):
            item = QtWidgets.QListWidgetItem(item)

        # Get the new item's index
        index = self.count() + 1  # 1-based index
        # Update the text to include the index
        current_text = item.text()
        item.setText(f'[{index}] {current_text}')

        super().addItem(item)

    def currentChanged(self, current, previous):
        # Update font weight for both items
        if previous:
            item = self.item(previous.row())
            if item:
                font = item.font()
                font.setBold(False)
                item.setFont(font)

        if current:
            item = self.item(current.row())
            if item:
                font = item.font()
                font.setBold(True)
                item.setFont(font)

        super().currentChanged(current, previous)

    def takeItem(self, row):
        super().takeItem(row)

        # Update indices after removal
        self.updateIndices()

    def updateIndices(self):
        '''Update all item indices after removals or reordering'''
        for i in range(self.count()):
            item = self.item(i)
            if item:
                # Remove old index if present and add new one
                text = item.text()
                if ']' in text:
                    text = text.split(']', 1)[1].strip()
                item.setText(f'[{i + 1}] {text}')


class ImageItemsWidget(QtWidgets.QWidget):
    layerRemoved = Signal(int)  # Signal emitted when user wants to remove a layer
    layerChanged = Signal(
        int, int, object
    )  # Signal emitted when user selects a different layer

    COMP_MODES = [
        'Clear',
        'ColorBurn',
        'ColorDodge',
        'Darken',
        'Destination',
        'DestinationAtop',
        'DestinationIn',
        'DestinationOut',
        'DestinationOver',
        'Difference',
        'Exclusion',
        'HardLight',
        'Lighten',
        'Multiply',
        'Overlay',
        'Plus',
        'Screen',
        'SoftLight',
        'Source',
        'SourceAtop',
        'SourceIn',
        'SourceOut',
        'SourceOver',
        'Xor',
    ]

    def __init__(self, debug=False):
        super().__init__()
        self._debug = debug
        self._readonly = False

        # Create main layout
        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # Create splitter
        self.splitter = QtWidgets.QSplitter(Qt.Orientation.Vertical)
        self.splitter.setHandleWidth(1)  # Minimal width for splitter handle
        self.splitter.setStyleSheet('''
            QSplitter::handle {
                background-color: #999999;
            }
        ''')

        # Layer list section
        list_widget = QtWidgets.QWidget()
        list_layout = QtWidgets.QVBoxLayout()
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_widget.setLayout(list_layout)

        # Custom list widget with enhanced font handling
        self.layer_list = CustomListWidget()
        palette = self.layer_list.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(240, 240, 240))
        self.layer_list.setPalette(palette)
        self.layer_list.currentItemChanged.connect(self._on_layer_selection_changed)
        list_layout.addWidget(self.layer_list)

        # Parameters section
        self.params_container = QtWidgets.QWidget()
        self.params_layout = QtWidgets.QVBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)
        self.params_container.setLayout(self.params_layout)

        self.placeholder_label = QtWidgets.QLabel('Select a layer to view parameters')
        self.placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.params_layout.addWidget(self.placeholder_label)

        # Add widgets to splitter
        self.splitter.addWidget(list_widget)
        self.splitter.addWidget(self.params_container)

        # Add splitter to main layout
        main_layout.addWidget(self.splitter)

        # remove layer button
        self.remove_button = QtWidgets.QPushButton('Remove Layer')
        self.remove_button.clicked.connect(self.remove_layer)
        main_layout.addWidget(self.remove_button)

        # Store layer parameter widgets
        self.layer_params: list[ImageItemOptions] = []

    @property
    def currentIndex(self):
        '''
        Returns the index of the currently selected layer.
        Returns -1 if no layer is selected.
        '''
        return self.layer_list.currentRow()

    @currentIndex.setter
    def currentIndex(self, index):
        if not isinstance(index, int):
            raise TypeError('Index must be an integer')

        if index < 0 or index >= self.layer_list.count():
            raise ValueError('Index out of range')

        self.layer_list.setCurrentRow(index)

    @property
    def currentLayer(self) -> pg.ImageItem:
        '''
        Returns the currently selected layer.
        Returns None if no layer is selected.
        '''
        index = self.currentIndex
        if index < 0:
            return None
        return self.layer_list.item(index).data(Qt.ItemDataRole.UserRole)

    def setReadonly(self, readonly: bool = True):
        '''
        Sets the widget to read-only mode.
        In read-only mode, the user cannot remove layers.
        '''
        self._readonly = readonly

        self.remove_button.setEnabled(not readonly)

    def getImageItemAt(self, index: int) -> pg.ImageItem:
        '''
        Returns the image item at the given index.
        Returns None if the index is out of range.
        '''
        # if negative index, count from the end
        if index < 0:
            index = self.layer_list.count() + index

        if index < 0 or index >= self.layer_list.count():
            return None

        return self.layer_list.item(index).data(Qt.ItemDataRole.UserRole)

    def add_layer(
        self,
        image: np.ndarray,
        opacity=1.0,
        composition_mode='SourceOver',
        name='Layer',
    ):
        # Create the ImageItem and set its view to self.view_box
        image_item = pg.ImageItem(
            image, axisOrder='row-major', opacity=max(min(opacity, 1), 0)
        )
        image_item.setCompositionMode(
            getattr(
                QtGui.QPainter.CompositionMode, f'CompositionMode_{composition_mode}'
            )
        )

        layer_index = self.count()

        # Add to list widget
        item = QtWidgets.QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, image_item)  # Store imageItem
        self.layer_list.addItem(item)

        # Create parameter widget
        params_widget = ImageItemOptions(layer_index)
        params_widget.paramsChanged.connect(self._on_params_changed)
        params_widget.comp_combo.setCurrentText(composition_mode)
        self.layer_params.append(params_widget)

        # Select the new layer
        self.layer_list.setCurrentItem(item)

        return image_item

    def remove_layer(self):
        if self.layer_list.count() > 1:
            current_row = self.layer_list.currentRow()
            if current_row >= 0:
                # Get the image item and its parent viewbox
                image_item = self.currentLayer
                if image_item:
                    # Remove from viewbox if it has a parent
                    if image_item.parentWidget():
                        image_item.parentWidget().removeItem(image_item)
                    # Clear the image data
                    image_item.clear()
                    # Set to None to help garbage collection
                    image_item.setParentItem(None)

                    self.layer_list.currentItem().setData(
                        Qt.ItemDataRole.UserRole, None
                    )

                # Remove from list
                self.layer_list.takeItem(current_row)

                # Clean up and reorganize remaining layers
                if current_row < len(self.layer_params):
                    self.layer_params[current_row].paramsChanged.disconnect()
                    self.layer_params[current_row].deleteLater()
                    del self.layer_params[current_row]

                self.layerRemoved.emit(current_row)
                self.layer_list.setCurrentRow(-1)
                self.layer_list.setCurrentRow(current_row - 1)

    def clear_layers(self):
        self.layer_list.clear()
        for widget in self.layer_params:
            widget.paramsChanged.disconnect()
            widget.deleteLater()
        self.layer_params.clear()

    def count(self):
        return self.layer_list.count()

    def _on_layer_selection_changed(self, current, previous):
        # Clear current params layout
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().hide()

        if current:
            current_index = self.layer_list.row(current)
            previous_index = self.layer_list.row(previous)

            imageItem: pg.ImageItem = self.layer_list.currentItem().data(
                Qt.ItemDataRole.UserRole
            )
            if current_index < len(self.layer_params):
                params_widget = self.layer_params[current_index]
                self.params_layout.addWidget(params_widget)
                params_widget.show()
                self.placeholder_label.hide()
                self.layerChanged.emit(current_index, previous_index, imageItem)
        else:
            self.params_layout.addWidget(self.placeholder_label)
            self.placeholder_label.show()
            self.layerChanged.emit(-1, -1, None)

    def _on_params_changed(self, layer_index, parameter, value):
        if self._debug:
            print(f'Parameter changed:')
            print(f'  Layer index: {layer_index}')
            print(f'  Parameter: {parameter}')
            print(f'  Value: {value}')
            print('  ----------')

        if self.layer_list.currentItem() is None:
            return

        imageItem: pg.ImageItem = self.layer_list.currentItem().data(
            Qt.ItemDataRole.UserRole
        )

        if parameter in ['Opacity', 'Visible']:
            value = value / 100 if parameter == 'Opacity' else value
            imageItem.setOpts(opacity=float(value))
        elif parameter == 'CompositionMode':
            imageItem.setCompositionMode(
                getattr(QtGui.QPainter.CompositionMode, f'CompositionMode_{value}')
            )


if __name__ == '__main__':
    app = QApplication([])
    my_app = ImageItemsWidget()
    my_app.show()
    app.exec()
