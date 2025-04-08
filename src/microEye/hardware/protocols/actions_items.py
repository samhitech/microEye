from typing import Union

from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ActionParameter, GroupParameter

from microEye.hardware.cams import Camera_Panel
from microEye.hardware.protocols.actions import (
    ActionGroup,
    BaseAction,
    ForLoop,
    FunctionCall,
    ParameterAdjustmentAction,
    WeakObjects,
)
from microEye.qt import QtCore, QtGui, QtWidgets
from microEye.utils.parameter_tree import Tree

# Define color constants
BACKGROUND_COLOR = QtGui.QColor('#262626')  # Dark gray #1E1E1E
FOREGROUND_COLOR = QtGui.QColor('#D4D4D4')  # Light gray
BORDER_COLOR = QtGui.QColor('#808080')  # Gray
HIGHLIGHT_COLOR = QtGui.QColor('#4D4D4D')  # Darker gray

# Define font constants
FONT_FAMILY = ', '.join(
    [
        'Courier New',
        'Courier',
        'Monaco',
        'Menlo',
        'system-ui',
        '-apple-system',
        'BlinkMacSystemFont',
        "'Segoe UI'",
        'Roboto',
        'Oxygen',
        'Ubuntu',
        'Cantarell',
        "'Open Sans'",
        "'Helvetica Neue'",
        'sans-serif',
    ]
)
FONT_SIZE = 10
FONT_BOLD = QtGui.QFont.Weight.Bold


class BaseActionItem(QtWidgets.QGraphicsRectItem):
    '''
    Base class for graphical representation of action items in the scene.

    Subclass of QGraphicsRectItem.

    Attributes
    ----------
    TOP_MARGIN : int
        Top margin for layout.
    H_MARGIN : int
        Horizontal margin for layout.
    VSPACING : int
        Vertical spacing for layout.
    MIN_WIDTH : int
        Minimum width of the item.
    MIN_HEIGHT : int
        Minimum height of the item.
    '''

    TOP_MARGIN = 30
    MARGIN = 20
    SPACING = 10
    MIN_WIDTH = 150
    MIN_HEIGHT = 50

    def __init__(
        self,
        action: Union[ActionGroup, ForLoop, FunctionCall, ParameterAdjustmentAction],
        parent=None,
    ):
        '''
        Initialize a new BaseActionItem instance.

        Parameters
        ----------
        action : Union[ActionGroup, ForLoop, FunctionCall, ParameterAdjustmentAction]
            The associated action for this item.
        parent : QGraphicsItem, optional
            The parent item, by default None.
        '''
        super().__init__(parent)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)

        self.action = action
        self.text_item = QtWidgets.QGraphicsTextItem(self)
        self.text_item.setPos(5, 5)
        # Set font for the text item
        font = QtGui.QFont(FONT_FAMILY, FONT_SIZE)
        self.text_item.setFont(font)
        self.text_item.setDefaultTextColor(FOREGROUND_COLOR)
        self.text_item.setHtml(action.toHTML())

        bounding_rect = self.text_item.boundingRect()

        self.MIN_WIDTH = max(self.MIN_WIDTH, bounding_rect.width() + self.MARGIN)
        self.MIN_HEIGHT = max(self.MIN_HEIGHT, bounding_rect.height() + self.MARGIN)

        self.setRect(QtCore.QRectF(0, 0, self.MIN_WIDTH, self.MIN_HEIGHT))

        # Set dark background color
        self.setBrush(BACKGROUND_COLOR)
        self.setPen(QtGui.QPen(BORDER_COLOR, 1))

        # Add a subtle shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5)
        shadow.setColor(QtGui.QColor('#000000'))  # Adjust the shadow color as needed
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        '''
        Handle mouse double-click events for editing properties.

        Opens a dialog to edit customizable properties.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The mouse event.
        '''
        if isinstance(self.action, ForLoop):
            text, ok = QtWidgets.QInputDialog.getInt(
                None,
                'Edit Repeat Count',
                'Enter new repeat count:',
                value=self.action.repeat_count,
            )
            if ok:
                self.action.setRepeatCount(text)
                self.text_item.setHtml(self.action.toHTML())
        else:
            text, ok = QtWidgets.QInputDialog.getText(
                None, 'Edit Properties', 'Enter new name:', text=self.action.name
            )
            if ok and text:
                self.action.name = text
                self.text_item.setHtml(self.action.toHTML())

        self.update_layout()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        """
        Handle mouse move events for dragging.

        Update the item's position and recursively update layout.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The mouse event.
        """
        # Check if the item being moved is the root item
        if self.parentItem() is None:
            # This is the root item, handle the move event
            super().mouseMoveEvent(event)
            new_pos = event.scenePos()

            parent = self.parentItem()

            if parent is not None:
                parent_rect = parent.boundingRect()
                new_pos.setX(parent_rect.left() + self.MARGIN // 2)
                new_pos.setY(
                    max(
                        parent_rect.top() + self.TOP_MARGIN,
                        min(
                            new_pos.y(),
                            parent_rect.bottom()
                            - self.boundingRect().height()
                            - self.SPACING,
                        ),
                    )
                )

                self.setPos(new_pos)
                parent.update_layout_recursive()
        else:
            # This is not the root item, skip the move event
            return

    def max_neighbor_width(self):
        '''
        Calculate the maximum width of the neighboring items.

        Returns
        -------
        int
            The maximum width of the neighboring items.
        '''
        parent = self.parentItem()
        if isinstance(parent, ActionGroupItem):
            return parent.max_child_width()
        else:
            return self.MIN_WIDTH

    def calculate_layout(self):
        '''
        Calculate the layout dimensions based on the provided parameters
        and neighboring items widths.

        Returns
        -------
        tuple
            A tuple containing the calculated width and height for the layout.
        '''
        parent = self.parentItem()
        if isinstance(self, ActionGroupItem):
            content_width = self.max_child_width() + 2 * self.MARGIN
            content_height = self.children_height()
        else:
            content_width = self.text_item.boundingRect().width()
            content_height = self.text_item.boundingRect().height()

        if isinstance(parent, ActionGroupItem):
            neighbor_width = parent.max_child_width() + self.MARGIN
        else:
            neighbor_width = content_width

        width = max([neighbor_width, content_width, self.MIN_WIDTH]) + self.MARGIN
        height = max(self.MIN_HEIGHT, content_height)
        return width, height

    def update_layout(self, update_parent=True):
        '''
        Update the layout of the item.
        '''
        parent = self.parentItem()
        if isinstance(parent, ActionGroupItem):
            max_width, max_height = self.calculate_layout()

            self.setRect(0, 0, max_width, max_height)
            if update_parent:
                parent.update_layout_recursive(False)

    def update_layout_recursive(self, update_self=True):
        '''
        Recursively update layout starting from the current
        item and going up the hierarchy.
        '''
        current_item = self if update_self else self.parentItem()
        while current_item is not None:
            current_item.update_layout()
            current_item = current_item.parentItem()


class ActionGroupItem(BaseActionItem):
    '''
    Graphical representation of an ActionGroup or ForLoop in the scene.

    Subclass of BaseActionItem.

    Attributes
    ----------
    child_items : list
        List of child items.
    '''

    def __init__(self, action: Union[ActionGroup, ForLoop], parent=None):
        '''
        Initialize a new ActionGroupItem instance.

        Parameters
        ----------
        action : Union[ActionGroup, ForLoop]
            The associated action for this item.
        parent : QGraphicsItem, optional
            The parent item, by default None.
        '''
        super().__init__(action, parent)

        self.child_items: list[BaseActionItem] = []

    def add_child_item(self, child_item: BaseActionItem):
        '''
        Add a child item to the group.

        Parameters
        ----------
        child_item : BaseActionItem
            The child item to be added.
        '''
        self.child_items.append(child_item)
        self.action.child_actions.append(child_item.action)
        child_item.setParentItem(self)
        self.update_layout_recursive()

    def remove_child_item(self, child_item):
        '''
        Remove a child item from the group.

        Parameters
        ----------
        child_item : BaseActionItem
            The child item to be removed.
        '''
        self.child_items.remove(child_item)
        self.action.child_actions.remove(child_item.action)
        child_item.setParentItem(None)
        self.update_layout_recursive()

    def children_widths(self):
        '''
        Get the widths of the child items.

        Returns
        -------
        list[int]
            The width of the child items.
        '''
        if self.child_items:
            return [
                child_item.max_child_width() + self.MARGIN
                if isinstance(child_item, ActionGroupItem)
                else child_item.text_item.boundingRect().width()
                for child_item in self.child_items
            ]
        else:
            return [self.MIN_WIDTH]

    def children_height(self):
        '''
        Get the heights of the child items.

        Returns
        -------
        list[int]
            The heights of the child items.
        '''
        if self.child_items:
            return (
                sum(
                    [
                        child_item.boundingRect().height()
                        for child_item in self.child_items
                    ]
                )
                + self.text_item.boundingRect().height()
                + (len(self.child_items) + 1) * self.SPACING
            )
        else:
            return self.MIN_HEIGHT

    def max_child_width(self):
        '''
        Calculate the maximum width of the child items.

        Returns
        -------
        int
            The maximum width of the child items.
        '''
        return max(self.children_widths())

    def update_layout(self, update_parent=True):
        '''
        Update the layout of the item and its child items.
        '''
        max_width, max_height = self.calculate_layout()
        self.setRect(0, 0, max_width, max_height)

        total_height = self.text_item.boundingRect().height() + self.SPACING
        for child_item in self.child_items:
            height = child_item.boundingRect().height()
            child_item.setPos(self.SPACING, total_height)
            total_height += height + self.SPACING
            child_item.update_layout(False)

    def add_child_item_by_type(self, action_type):
        '''
        Add a child item of a specific action type to the group.

        Parameters
        ----------
        action_type : type
            The type of action for the child item.
        '''
        action = action_type()
        action_item = get_action_item(action)
        self.add_child_item(action_item)

    def swap_children(self, index1, index2):
        '''
        Swap positions of two child items in the group.

        Parameters
        ----------
        index1 : int
            Index of the first child item.
        index2 : int
            Index of the second child item.
        '''
        self.child_items[index1], self.child_items[index2] = (
            self.child_items[index2],
            self.child_items[index1],
        )

        self.action.child_actions[index1], self.action.child_actions[index2] = (
            self.action.child_actions[index2],
            self.action.child_actions[index1],
        )

        self.update_layout_recursive()


class ParameterAdjustmentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setWindowTitle('Configure Parameter Adjustment')

        layout = QtWidgets.QVBoxLayout(self)

        # Target Object
        self.target_object_combo = QtWidgets.QComboBox()
        layout.addWidget(QtWidgets.QLabel('Target Object:'))
        layout.addWidget(self.target_object_combo)

        # Parameter Name
        self.parameter_name_combo = QtWidgets.QComboBox()
        layout.addWidget(QtWidgets.QLabel('Parameter Name:'))
        layout.addWidget(self.parameter_name_combo)

        # Value Editor
        self.value_editor_stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.value_editor_stack)

        # Expression Mode
        self.expression_mode_checkbox = QtWidgets.QCheckBox('Expression Mode')
        layout.addWidget(self.expression_mode_checkbox)

        # Expression Editor
        self.expression_editor = QtWidgets.QLineEdit()
        self.expression_editor.setPlaceholderText(
            'Enter expression (e.g., f"Experiment_{i2+1:03d}")'
        )
        layout.addWidget(self.expression_editor)
        self.expression_editor.hide()

        # Delay
        self.delay_spin = QtWidgets.QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 1e6)
        self.delay_spin.setSingleStep(0.1)
        self.delay_spin.setDecimals(3)
        layout.addWidget(QtWidgets.QLabel('Delay (seconds):'))
        layout.addWidget(self.delay_spin)

        # Buttons
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # Connect signals
        self.target_object_combo.currentIndexChanged.connect(
            self.update_parameter_names
        )
        self.parameter_name_combo.currentIndexChanged.connect(
            self.update_parameter_value_editor
        )
        self.expression_mode_checkbox.stateChanged.connect(self.toggle_expression_mode)

        # Initialize
        self.target_object_combo.addItems(WeakObjects.getObjectNames())
        self.update_parameter_names()

    def update_parameter_names(self):
        tree = self.get_paramtree()
        if tree:
            parameter_names = [
                param
                for param in self.get_params()
                if not isinstance(tree.get_param(param), GroupParameter)
                and tree.get_param(param) is not None
            ]
            self.parameter_name_combo.clear()
            self.parameter_name_combo.addItems(parameter_names)
            self.update_parameter_value_editor()

    def update_parameter_value_editor(self):
        parameter_name = self.parameter_name_combo.currentText()

        while self.value_editor_stack.count() > 0:
            self.value_editor_stack.removeWidget(self.value_editor_stack.widget(0))

        tree = self.get_paramtree()
        if tree:
            param = tree.get_param(parameter_name)
            if param is not None:
                value_editor = self.create_value_editor(param)
                if value_editor is not None:
                    self.value_editor_stack.addWidget(value_editor)
                    self.value_editor_stack.setCurrentWidget(value_editor)

                # Disable expression mode for action parameters
                is_action = isinstance(param, ActionParameter)
                self.expression_mode_checkbox.setEnabled(not is_action)
                if is_action:
                    self.expression_mode_checkbox.setChecked(False)
                    self.expression_editor.hide()
                    self.value_editor_stack.show()

    def toggle_expression_mode(self, state):
        is_expression = state == QtCore.Qt.CheckState.Checked.value
        self.value_editor_stack.setVisible(not is_expression)
        self.expression_editor.setVisible(is_expression)

    def create_value_editor(self, param: Parameter):
        param_type = param.opts.get('type', None)
        param_limits = param.opts.get('limits', [])

        if param_type in ['int', 'float']:
            value_editor = (
                QtWidgets.QDoubleSpinBox()
                if param_type == 'float'
                else QtWidgets.QSpinBox()
            )
            if param_limits:
                value_editor.setMinimum(param_limits[0])
                value_editor.setMaximum(param_limits[1])
            else:
                value_editor.setMinimum(0)
                value_editor.setMaximum(int(1e9))
            value_editor.setValue(param.value())
        elif param_type == 'bool':
            value_editor = QtWidgets.QCheckBox()
            value_editor.setChecked(param.value())
        elif param_type == 'action':
            value_editor = QtWidgets.QCheckBox('Try to wait for Event')
        elif param_type == 'str' or param_type == 'file':
            value_editor = QtWidgets.QLineEdit()
            value_editor.setText(param.value())
        elif param_type == 'list':
            value_editor = QtWidgets.QComboBox()
            for limit in param_limits:
                value_editor.addItem(str(limit), limit)
            value_editor.setCurrentText(str(param.value()))
        else:
            value_editor = None

        return value_editor

    def get_paramtree(self):
        target_name = self.target_object_combo.currentText()
        target_object = WeakObjects.getObject(target_name)

        if target_object is None:
            return None

        if isinstance(target_object, Tree):
            return target_object
        if isinstance(target_object, Camera_Panel):
            return target_object.camera_options
        for attr_name in dir(target_object):
            attr = getattr(target_object, attr_name)
            if isinstance(attr, Tree):
                return attr

    def get_params(self):
        target_name = self.target_object_combo.currentText()
        target_object = WeakObjects.getObject(target_name)

        if target_object is None:
            return None

        if isinstance(target_object, Tree):
            return target_object.get_param_paths()
        if isinstance(target_object, Camera_Panel):
            return target_object.camera_options.get_param_paths()

        for attr_name in dir(target_object):
            attr = getattr(target_object, attr_name)
            if hasattr(attr, 'get_param_paths'):
                return attr.get_param_paths()

    def get_parameter_value(self):
        if self.expression_mode_checkbox.isChecked():
            return self.expression_editor.text(), True

        editor = self.value_editor_stack.currentWidget()
        if isinstance(editor, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            return editor.value(), False
        elif isinstance(editor, QtWidgets.QCheckBox):
            return editor.isChecked(), False
        elif isinstance(editor, QtWidgets.QLineEdit):
            return editor.text(), False
        elif isinstance(editor, QtWidgets.QComboBox):
            return editor.currentData(), False
        else:
            return None, False

    def set_parameter_value(self, value, is_expression=False):
        self.expression_mode_checkbox.setChecked(is_expression)
        if is_expression:
            self.expression_editor.setText(value)
        else:
            editor = self.value_editor_stack.currentWidget()
            if isinstance(editor, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                editor.setValue(float(value))
            elif isinstance(editor, QtWidgets.QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QtWidgets.QLineEdit):
                editor.setText(str(value))
            elif isinstance(editor, QtWidgets.QComboBox):
                index = editor.findData(value)
                if index >= 0:
                    editor.setCurrentIndex(index)
            else:
                return None


class ParameterAdjustmentActionItem(BaseActionItem):
    def __init__(
        self,
        action: ParameterAdjustmentAction,
        parent=None,
    ):
        super().__init__(action, parent)

    def tree(self):
        return WeakObjects.OBJECTS

    def mouseDoubleClickEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        config_dialog = ParameterAdjustmentDialog()

        if self.action.target_object:
            index = config_dialog.target_object_combo.findText(
                self.action.target_object
            )
            if index >= 0:
                config_dialog.target_object_combo.setCurrentIndex(index)

        if self.action.parameter_name:
            index = config_dialog.parameter_name_combo.findText(
                self.action.parameter_name
            )
            if index >= 0:
                config_dialog.parameter_name_combo.setCurrentIndex(index)
        if self.action.parameter_value is not None:
            config_dialog.set_parameter_value(
                self.action.parameter_value, self.action.is_expression
            )
        if self.action.delay:
            config_dialog.delay_spin.setValue(self.action.delay)

        if config_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            target_object_index = config_dialog.target_object_combo.currentIndex()
            if target_object_index > -1:
                self.action.target_object = (
                    config_dialog.target_object_combo.currentText()
                )
                self.action.parameter_name = (
                    config_dialog.parameter_name_combo.currentText()
                )
                self.action.parameter_value, self.action.is_expression = (
                    config_dialog.get_parameter_value()
                )
                self.action.delay = config_dialog.delay_spin.value()

                self.text_item.setHtml(self.action.toHTML())

        self.update_layout_recursive()


ACTION_TYPE_TO_ITEM = {
    ActionGroup: ActionGroupItem,
    FunctionCall: BaseActionItem,
    ForLoop: ActionGroupItem,
    ParameterAdjustmentAction: ParameterAdjustmentActionItem,
}


def get_action_item(action: BaseAction):
    '''
    Get the appropriate action item based on the action type.

    Parameters
    ----------
    action : BaseAction
        The action for which to create an item.

    Returns
    -------
    BaseActionItem
        The corresponding action item.
    '''
    action_item_class = ACTION_TYPE_TO_ITEM.get(type(action))
    if action_item_class is None:
        raise ValueError(f'Invalid action type: {type(action)}')

    return action_item_class(action)
