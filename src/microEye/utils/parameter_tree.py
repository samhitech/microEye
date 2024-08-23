import json
from enum import Enum
from typing import Any, Optional, Union

from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.parametertree.parameterTypes import ActionParameter, GroupParameter

from microEye.qt import QtWidgets, Signal, Slot, getOpenFileName, getSaveFileName


class Tree(ParameterTree):
    '''
    Tree widget for editing parameters.

    Attributes
    ----------
    paramsChanged : pyqtSignal
        Signal for parameter changed event.
    '''

    PARAMS: type[Enum] = None

    paramsChanged = Signal(GroupParameter, list)
    '''Signal emitted when parameters are changed.

    Parameters
    ----------
    GroupParameter
        The group parameter that was changed.
    list
        A list of changes made to the parameter.
    '''

    def __init__(self, parent: Optional['QtWidgets.QWidget'] = None):
        '''
        Initialize the Tree.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget, by default None.
        '''
        super().__init__(parent=parent)

        self.setMinimumWidth(50)
        self.create_parameters()
        self.setParameters(self.param_tree, showTop=False)

    def create_parameters(self):
        '''
        Create the parameter tree structure.

        Reimplement in child class.
        '''
        params = []

        self.param_tree = Parameter.create(name='', type='group', children=params)
        self.param_tree.sigTreeStateChanged.connect(self.change)
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    def get_param(self, param: Union[Enum, str]) -> Union[Parameter, ActionParameter]:
        '''Get a parameter by name.

        Parameters
        ----------
        param : Enum
            parameter.

        Returns
        -------
        Union[Parameter, ActionParameter]
            Retrieved parameter.
        '''
        try:
            if isinstance(param, Enum):
                return self.param_tree.param(*param.value.split('.'))
            else:
                return self.param_tree.param(*param.split('.'))
        except KeyError:
            return None

    def get_param_value(self, param: Union[Enum, str]):
        '''Get a parameter value by name.

        Parameters
        ----------
        param : Enum
            parameter.

        Returns
        -------
        Any
            The value of the parameter.
        '''
        try:
            if isinstance(param, Enum):
                return self.param_tree.param(*param.value.split('.')).value()
            else:
                return self.param_tree.param(*param.split('.')).value()
        except KeyError:
            return None

    def set_param_value(
        self, param: Enum, value, blockSignals: Union[Any, None] = None
    ):
        '''
        Set a parameter value by name.

        Parameters
        ----------
        param : Enum
            parameter.
        value : Any
            The value to set.
        blockSignals : Union[Any, None], optional
            If provided, signals for parameter changes are blocked during the update.
            The interpretation of the value is left to the implementation.

        Returns
        -------
        bool
            True if the value is set successfully, False otherwise.
        '''
        try:
            parameter: Parameter = self.param_tree.param(*param.value.split('.'))
            parameter.setValue(value, blockSignals)
        except Exception:
            import traceback

            traceback.print_exc()
            return False
        else:
            return True

    def get_param_path(self, param: Parameter):
        '''
        Get the child path of a parameter in the parameter tree.

        Parameters
        ----------
        param : Parameter
            The parameter for which to retrieve the child path.

        Returns
        -------
        list
            The child path of the parameter.
        '''
        return self.param_tree.childPath(param)

    def add_param_child(self, parent: Enum, value: dict):
        """
        Add a child parameter to the specified parent parameter.

        Parameters
        ----------
        parent : Enum
            The parent parameter to which the child will be added.
        value : dict
            A dictionary representing the child parameter. It should contain at
            least the following key-value pairs:
            - 'name': str, the name of the parameter.
            - 'type': str, the type of the parameter (e.g., 'int', 'float', 'str').
            - 'value': Any, the initial value of the parameter.
            Additional optional keys can be included based on the desired configuration.

        Returns
        -------
        None
        """
        parent = self.get_param(parent)
        parent.addChild(value, autoIncrementName=True)

    def get_children(self, param: Enum):
        """
        Get the values of all children of a specified parameter.

        Parameters
        ----------
        param : Enum
            The parameter whose children's values will be retrieved.

        Returns
        -------
        list
            List of values of all children of the specified parameter.
        """
        res = []
        param = self.get_param(param)
        if isinstance(param, GroupParameter):
            for child in param.children():
                res.append(child.value())
        return res

    def change(self, param: Parameter, changes: list):
        '''
        Handle parameter changes as needed.

        Parameters
        ----------
        param : Parameter
            The parameter that triggered the change.
        changes : list
            List of changes.

        Returns
        -------
        None
        '''
        # Handle parameter changes as needed
        pass


    @Slot(object)
    def export_json(self, action = None):
        '''
        Export parameters to a JSON file.

        Returns
        -------
        None
        '''
        filename, _ = getSaveFileName(
            None, 'Save Parameters', '', 'JSON Files (*.json);;All Files (*)'
        )
        if not filename:
            return  # User canceled the operation

        state = self.param_tree.saveState()
        with open(filename, 'w', encoding='utf8') as file:
            json.dump(state, file, indent=2)

    def get_json(self):
        '''
        Export parameters to a JSON text.

        Returns
        -------
        str
            A string containing all parameters in JSON format.
        '''
        return self.param_tree.saveState()

    # Load parameters from JSON
    @Slot(object)
    def load_json(self, action = None):
        '''
        Load parameters from a JSON file.

        Returns
        -------
        None
        '''
        filename, _ = getOpenFileName(
            None, 'Load Parameters', '', 'JSON Files (*.json);;All Files (*)'
        )
        if not filename:
            return  # User canceled the operation

        with open(filename, encoding='utf8') as file:
            state = json.load(file)
        self.param_tree.restoreState(state, blockSignals=False)

    def get_param_paths(self) -> list[str]:
        '''
        Get a list of all parameter paths in the tree, excluding group parameters.

        Returns
        -------
        list
            A list of all parameter paths, excluding groups.
        '''

        def collect_paths(param, current_path=None):
            paths = []
            if current_path is None:
                current_path = []
            for child in param.children():
                child_path = current_path + [child.name()]
                if not child.hasChildren():  # This checks if parameter is not a group
                    paths.append('.'.join(child_path))
                else:
                    paths.extend(collect_paths(child, child_path))
            return paths

        return collect_paths(self.param_tree)

    def search_param(self, contains: str) -> Union[None, Parameter, ActionParameter]:
        '''
        Find parameter that contains the supplied string
        in the tree and return it if found.

        Parameters
        ----------
        contains : str
            The string to search for in the parameter paths.

        Returns
        -------
        object or None
            The value of the first parameter that contains the supplied string,
            or None if no match is found.
        '''
        path = [
            path
            for path in self.get_param_paths()
            if contains.lower().strip() in path.lower().split('.')[-1]
        ]
        if path:
            self.get_param(path[0])
        else:
            return None

    def set_expanded(self, param: Union[Parameter, Enum, str], value: bool = False):
        if isinstance(param, (Enum, str)):
            param = self.get_param(param)

        if not isinstance(param, Parameter):
            return

        if param:
            for item in self.listAllItems():
                if item.param == param:
                    item.setExpanded(value)
                    break
