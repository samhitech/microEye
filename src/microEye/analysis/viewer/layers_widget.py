import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import GroupParameter


class ImageParamsWidget(ParameterTree):
    paramsChanged = pyqtSignal(GroupParameter, list)

    # Define global variable for composition modes
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

        # Create an initial parameter tree with a Layers group
        params = [
            {'name': 'Layers', 'type': 'group', 'children': []},
        ]

        self.param_tree = Parameter.create(name='params', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.setParameters(self.param_tree, showTop=False)
        self.header().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)

        self.setWindowTitle('ParameterTree Example')

    def add_layer(self, compositionMode='SourceOver'):
        # Add a new layer to the Layers group
        layer_name = f'Layer {self.get_layers_count() + 1}'
        new_layer = {'name': layer_name, 'type': 'group', 'children': [
            {'name': 'Visible', 'type': 'bool', 'value': True},
            {'name': 'Opacity', 'type': 'slider', 'value': 100, 'limits': (0, 100)},
            {'name': 'CompositionMode', 'type': 'list',
             'values': ImageParamsWidget.COMP_MODES,
             'value': 'SourceOver'}
        ]}
        self.param_tree.param('Layers').addChild(new_layer)

    def remove_layer(self):
        # Remove the last layer from the Layers group if there is at least one layer
        layers_group = self.param_tree.param('Layers')
        if len(layers_group.children()) > 0:
            layers_group.removeChild(layers_group.children()[-1])

    def clear_layers(self):
        # Remove all layers from the Layers group
        self.param_tree.param('Layers').clearChildren()

    def get_layers_count(self):
        # Get the count of layers in the Layers group
        return len(self.param_tree.param('Layers').children())

    def change(self, param: GroupParameter, changes: list):
        self.paramsChanged.emit(param, changes)
        if not self._debug:
            return

        print('tree changes:')
        for param, change, data in changes:
            path = self.param_tree.childPath(param)
            if len(path) > 1:
                print('  parameter: %s'% path[-1])
                print('  parent: %s'% path[-2])
            else:
                childName = '.'.join(path) if path is not None else param.name()
                print('  parameter: %s'% childName)
            print('  change:    %s'% change)
            print('  data:      %s'% str(data))
            print('  ----------')

if __name__ == '__main__':
    app = QApplication([])
    my_app = ImageParamsWidget()
    my_app.show()
    app.exec_()
