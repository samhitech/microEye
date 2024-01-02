import sys
from typing import Callable, Union

from PyQt5.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPen,
)
from PyQt5.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsSceneWheelEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QInputDialog,
    QMainWindow,
    QMenu,
    QUndoStack,
)


class BaseAction:
    '''
    Base class for all actions in the editor.

    Attributes
    ----------
    id_counter : int
        Global counter for action IDs.
    NAME : str
        The name of the action type.
    id : int
        The unique identifier for each action instance.
    name : str
        The name of the action instance.
    '''
    id_counter = 1  # Global counter for action IDs
    NAME = 'Base Action'

    def __init__(self):
        '''
        Initialize a new BaseAction instance.
        '''
        self.id = BaseAction.id_counter
        BaseAction.id_counter += 1
        self.name = f'{self.__class__.__name__}_{self.id}'

    def execute(self):
        '''
        Execute the action. To be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If subclasses do not implement the execute method.
        '''
        raise NotImplementedError('Subclasses must implement the execute method')

    def __str__(self) -> str:
        '''
        Return a string representation of the action.

        Returns
        -------
        str
            String representation of the action.
        '''
        return self.name


class SimpleAction(BaseAction):
    NAME = 'Simple Action'

    def execute(self):
        print(f'Executing: {str(self)}')


class FunctionCall(BaseAction):
    NAME = 'Function Call'

    def __init__(self, function: Callable = None, Args: tuple = ()):
        '''
        Initialize a new FunctionCall instance.

        Parameters
        ----------
        function : Callable, optional
            The function to be called, by default None.
        Args : tuple, optional
            The arguments to be passed to the function, by default ().

        Raises
        ------
        TypeError
            If the provided function is not callable.
        '''
        super().__init__()
        self.function = function
        self.function_args = Args

    def execute(self):
        print(f'Executing: {str(self)}')
        if self.function is not None:
            self.function(*self.function_args)


class ActionGroup(BaseAction):
    '''
    Action that groups multiple child actions.

    Subclass of BaseAction.

    Attributes
    ----------
    NAME : str
        The name of the action type.
    child_actions : list
        List of child actions.
    '''
    NAME = 'Action Group'

    def __init__(self):
        '''
        Initialize a new ActionGroup instance.
        '''
        super().__init__()
        self.child_actions: list[BaseAction] = []

    def execute(self):
        '''
        Execute the action group by executing all child actions.
        '''
        print(f'Executing: {str(self)}')
        for child_action in self.child_actions:
            child_action.execute()


class ForLoop(ActionGroup):
    '''
    Action that repeats the execution of its child actions.

    Subclass of ActionGroup.

    Attributes
    ----------
    NAME : str
        The name of the action type.
    repeat_count : int
        The number of times to repeat the child actions.
    '''
    NAME = 'For Loop'

    def __init__(self, repeat_count=3):
        '''
        Initialize a new ForLoop instance.

        Parameters
        ----------
        repeat_count : int, optional
            The number of times to repeat the child actions, by default 3.
        '''
        super().__init__()
        self.repeat_count = repeat_count

    def setRepeatCount(self, repeat_count):
        '''
        Set the number of times to repeat the child actions.

        Parameters
        ----------
        repeat_count : int
            The number of times to repeat the child actions.
        '''
        self.repeat_count = repeat_count

    def execute(self):
        '''
        Execute the for loop by repeating the child actions.

        Notes
        -----
        Overrides the execute method in ActionGroup.
        '''
        print(f'Executing: {str(self)}')
        for _ in range(self.repeat_count):
            for child_action in self.child_actions:
                child_action.execute()

    def __str__(self) -> str:
        '''
        Return a string representation of the for loop.

        Returns
        -------
        str
            String representation of the for loop.
        '''
        return f'{self.name}; {self.repeat_count} times'


class BaseActionItem(QGraphicsRectItem):
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
    H_MARGIN = 10
    VSPACING = 5
    MIN_WIDTH = 150
    MIN_HEIGHT = 50

    def __init__(
            self,
            action: Union[ActionGroup, ForLoop, SimpleAction, FunctionCall],
            parent=None):
        '''
        Initialize a new BaseActionItem instance.

        Parameters
        ----------
        action : Union[ActionGroup, ForLoop, SimpleAction, FunctionCall]
            The associated action for this item.
        parent : QGraphicsItem, optional
            The parent item, by default None.
        '''
        super().__init__(parent)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemIsSelectable)

        self.action = action
        self.text_item = QGraphicsSimpleTextItem(str(action), self)
        self.text_item.setPos(5, 5)
        # Set font for the text item
        font = QFont('Consolas', 10)
        text_color = QColor('#FFFFFF')
        self.text_item.setFont(font)
        self.text_item.setBrush(text_color)

        self.MIN_WIDTH = max(
            self.MIN_WIDTH, self.text_item.boundingRect().width() + self.H_MARGIN
        )

        self.setRect(QRectF(
            0, 0,
            self.MIN_WIDTH,
            self.MIN_HEIGHT))

        # Set dark background color
        background_color = QColor('#333333')  # Adjust the color as needed
        self.setBrush(background_color)

        # Set white border
        border_color = QColor('#AAAAAA')  # Set border color to white
        self.setPen(QPen(border_color, 1))  # Adjust the pen width as needed

        # Add a subtle shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(5)
        shadow.setColor(QColor('#000000'))  # Adjust the shadow color as needed
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        '''
        Handle mouse double-click events for editing properties.

        Opens a dialog to edit customizable properties.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The mouse event.
        '''
        if isinstance(self.action, ForLoop):
            text, ok = QInputDialog.getInt(
                None, 'Edit Repeat Count', 'Enter new repeat count:',
                value=self.action.repeat_count)
            if ok:
                self.action.setRepeatCount(text)
                self.text_item.setText(str(self.action))
        else:
            text, ok = QInputDialog.getText(None, 'Edit Properties', 'Enter new name:')
            if ok and text:
                self.action.name = text
                self.text_item.setText(str(self.action))

        self.update_layout()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """
        Handle mouse move events for dragging.

        Update the item's position and recursively update layout.

        Parameters
        ----------
        event : QGraphicsSceneMouseEvent
            The mouse event.
        """
        super().mouseMoveEvent(event)
        new_pos = event.scenePos()

        parent = self.parentItem()

        if parent is not None:
            parent_rect = parent.boundingRect()
            new_pos.setX(parent_rect.left() + self.H_MARGIN // 2)
            new_pos.setY(
                max(parent_rect.top() + self.TOP_MARGIN, min(
                    new_pos.y(),
                    parent_rect.bottom() - self.boundingRect().height() - self.VSPACING)
                ))

            self.setPos(new_pos)
            parent.update_layout_recursive()

    def update_layout(self):
        '''
        Update the layout of the item.
        '''
        if isinstance(self, ActionGroupItem):
            max_width = max(
                [item.boundingRect().width() for item in self.child_items])
        else:
            # Update the minimum width based on the updated text content
            self.MIN_WIDTH = max(
                self.MIN_WIDTH, self.text_item.boundingRect().width() + self.H_MARGIN
            )
            max_width = self.MIN_WIDTH

        self.setRect(
            0, 0,
            max(self.MIN_WIDTH, max_width + self.H_MARGIN),
            self.MIN_HEIGHT)

        if isinstance(self.parentItem(), ActionGroupItem):
            self.parentItem().update_layout_recursive()

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

    def update_layout_recursive(self):
        '''
        Recursively update layout starting from the current
        item and going up the hierarchy.
        '''
        current_item = self
        while current_item is not None:
            current_item.update_layout()
            current_item = current_item.parentItem()

    def update_layout(self):
        '''
        Update the layout of the item and its child items.
        '''
        total_height = self.TOP_MARGIN
        if self.child_items:
            max_width = max(
                [item.boundingRect().width() for item in self.child_items])
        else:
            max_width = self.MIN_WIDTH

        for child_item in self.child_items:
            height = child_item.boundingRect().height()
            child_item.setPos(5, total_height)
            total_height += height + self.VSPACING

        self.setRect(
            0, 0,
            max(self.MIN_WIDTH, max_width + self.H_MARGIN),
            max(self.MIN_HEIGHT, total_height))

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


ACTION_TYPE_TO_ITEM = {
    ActionGroup: ActionGroupItem,
    SimpleAction: BaseActionItem,
    FunctionCall: BaseActionItem,
    ForLoop: ActionGroupItem,
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

class ZoomableGraphicsView(QGraphicsView):
    '''
    Custom QGraphicsView with zooming and key event handling.

    Subclass of QGraphicsView.

    Attributes
    ----------
    root : ActionGroupItem
        The root item in the scene.
    '''

    def __init__(self, scene, root: ActionGroupItem):
        '''
        Initialize a new ZoomableGraphicsView instance.

        Parameters
        ----------
        scene : QGraphicsScene
            The graphics scene.
        root : ActionGroupItem
            The root item in the scene.
        '''
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.TextAntialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.HighQualityAntialiasing, True)

        self.root = root

        # Set dark background color for the view
        background_color = QColor('#1E1E1E')  # Adjust the color as needed
        self.setBackgroundBrush(background_color)

    def wheelEvent(self, event: QGraphicsSceneWheelEvent):
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
        selected_items = self.scene().selectedItems()

        if len(selected_items) == 1 and isinstance(selected_items[0], BaseActionItem):
            selected_item = selected_items[0]
            parent_item = selected_item.parentItem()
            if parent_item is not None and isinstance(parent_item, ActionGroupItem):
                index_self = parent_item.child_items.index(selected_item)

                if event.key() == Qt.Key_PageUp:
                    # Move item up
                    if index_self > 0:
                        parent_item.swap_children(index_self, index_self - 1)
                elif event.key() == Qt.Key_PageDown:
                    # Move item down
                    if index_self < len(parent_item.child_items) - 1:
                        parent_item.swap_children(index_self, index_self + 1)
                elif event.key() == Qt.Key_Delete:
                    # Delete item and its children
                    parent_item.remove_child_item(selected_item)
                    self.scene().removeItem(selected_item)

        super().keyPressEvent(event)

    def add_child(self, action_type):
        '''
        Add a child item of a specific action type.

        Parameters
        ----------
        action_type : type
            The type of action for the child item.
        '''
        selected_items = self.scene().selectedItems()

        if len(selected_items) == 1:
            if isinstance(selected_items[0], ActionGroupItem):
                selected_item = selected_items[0]
                selected_item.add_child_item_by_type(action_type)
            else:
                selected_item = selected_items[0]
                parent_item = selected_item.parentItem()
                if isinstance(parent_item, ActionGroupItem):
                    parent_item.add_child_item_by_type(action_type)

    def contextMenuEvent(self, event):
        '''
        Handle context menu events.

        Parameters
        ----------
        event : QContextMenuEvent
            The context menu event.
        '''
        context_menu = QMenu()

        # Add option to add different types
        add_menu = context_menu.addMenu('Add')
        add_menu.addAction(
            'Simple Action', lambda: self.add_child(SimpleAction))
        add_menu.addAction(
            'Function Call', lambda: self.add_child(FunctionCall))
        add_menu.addAction(
            'For Loop', lambda: self.add_child(ForLoop))

        # Add Move submenu
        move_submenu = context_menu.addMenu('Move')
        move_submenu.addAction('Up', lambda: self.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_PageUp, Qt.NoModifier, 0, 0, 0)
                ))
        move_submenu.addAction('Down', lambda: self.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_PageDown, Qt.NoModifier, 0, 0, 0)
                ))

        context_menu.addAction('Delete', lambda: self.keyPressEvent(
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.NoModifier, 0, 0, 0)
                ))

        # Add Execute action
        context_menu.addAction('Execute', self.execute_item)

        # Add Export and Import actions
        context_menu.addAction('Export (protocol)', self.export_protocol)
        context_menu.addAction('Import (protocol)', self.import_protocol)

        # Show the context menu
        context_menu.exec_(event.globalPos())

    def execute_item(self):
        '''
        Handle executing the topmost action group at the root level.
        '''
        if isinstance(self.root, ActionGroupItem):
            self.root.action.execute()

    def export_protocol(self):
        '''
        Handle exporting protocol.
        '''
        print('Exporting protocol')

    def import_protocol(self):
        '''
        Handle importing protocol.
        '''
        print('Importing protocol')


class ActionEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.undo_stack = QUndoStack(self)

        self.root = ActionGroupItem(ActionGroup())
        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene, self.root)
        self.setCentralWidget(self.view)

        # Add initial actions for testing
        self.scene.addItem(self.root)
        self.add_action(SimpleAction(), self.root)
        self.add_action(FunctionCall(), self.root)
        self.add_action(ForLoop(), self.root)


        # Set dark background color for the scene
        background_color = QColor('#1E1E1E')  # Adjust the color as needed
        self.scene.setBackgroundBrush(background_color)

    def add_action(self, action, parent_item=None):
        # Create the appropriate item based on the action type
        action_item = get_action_item(action)

        # Add the action item to the scene
        if parent_item is None:
            self.scene.addItem(action_item)
        else:
            parent_item.add_child_item(action_item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = ActionEditor()
    editor.show()
    sys.exit(app.exec_())
